"""RLVR base pipeline (TRL GRPO) — the DE-RISK harness for the differentiable organ.

WHAT THIS IS
  A minimal, WORKING RL-from-verifiable-rewards loop on ONE L40S:
      OLMo-2-1B (frozen base) + LoRA   --policy-->   reasoning-gym task
      completions  --our exact verifier (reward)-->  GRPO / Dr.GRPO update
  It proves the integration substrate runs+learns BEFORE we wire in the
  custom differentiable organ. The organ is NOT here yet — see ORGAN INSERTION
  POINT below for exactly where it slots.

WHY THESE CHOICES (per notes/rlvr_landscape.md)
  * use_vllm=False  -> generation goes through HF `model.generate`. This is the
    ONLY path that will later run our custom differentiable organ: vLLM only
    serves *registered* architectures, ours is bespoke. So we eat the slower
    HF-generate now to keep the organ-compatible code path.
  * LoRA-GRPO on a frozen OLMo base ("LoRA Without Regret": adapt ALL linears
    incl MLP, ~10x the full-FT LR).
  * Dr.GRPO objective (loss_type="dr_grpo" + scale_rewards=False) to drop the
    response-length / std-normalization biases of vanilla GRPO.
  * reasoning-gym as an infinite, difficulty-tunable task source; its canonical
    answer-checker (`get_score_answer_fn`) is our EXACT verifier — but we feed it
    a PARSED answer (extract final number), not the raw chatty completion, so the
    reward is a clean 1.0/0.0 instead of fuzzy decimal-similarity.

ORGAN INSERTION POINT  (where the next step plugs in)
  GRPOTrainer owns the policy `model` and calls `model.generate(...)` (HF path,
  because use_vllm=False). To wire the organ you replace the POLICY MODEL passed
  to GRPOTrainer (the `build_model()` return) with the augmented module whose
  forward/generate routes hidden states through the differentiable organ
  (cf. clair.glados_woven / clair.augmented). Everything else here — dataset
  glue, the exact-verifier reward_func, the GRPOConfig, the loop — stays as-is.
  The reward never changes: it only sees text completions + ground-truth columns.

USAGE
  python -m clair.rlvr_pipeline --smoke                 # ~tiny, sanity only
  python -m clair.rlvr_pipeline --steps 300             # the de-risk smoke run
"""
from __future__ import annotations

import argparse, json, re, time
from typing import Optional

import torch
from datasets import Dataset

import reasoning_gym
from reasoning_gym import get_score_answer_fn

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

MODEL_ID = "allenai/OLMo-2-0425-1B"  # base (not -Instruct): no chat template, string prompts

# Instruction wrapper: nudge the base model toward a parseable final answer.
# The reward also falls back to "last number in text", so partial compliance still scores.
PROMPT_TMPL = (
    "Solve the problem. Think briefly, then on the final line write exactly "
    "'Answer: <your answer>'.\n\n{q}\n"
)

# --------------------------------------------------------------------------- answer extraction
_ANS_RE = re.compile(r"answer\s*[:=]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def extract_answer(text: str) -> str:
    """Pull the model's final answer out of a chatty completion.
    Priority: text after the LAST 'Answer:' marker, else the LAST number in the text.
    Returns a clean string handed to reasoning-gym's canonical checker."""
    if not text:
        return ""
    m = list(_ANS_RE.finditer(text))
    if m:
        cand = m[-1].group(1).strip()
        nums = _NUM_RE.findall(cand)
        return nums[-1].replace(",", "") if nums else cand
    nums = _NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else text.strip()


# --------------------------------------------------------------------------- dataset glue
def build_split(task: str, size: int, seed: int, **cfg) -> Dataset:
    """One reasoning-gym task -> a TRL standard (non-conversational) dataset.
    Columns: prompt (str), answer (gold str), source (task name), metadata (json str).
    We stash answer+metadata so the reward can rebuild the exact `entry` dict that
    reasoning-gym's score_answer / get_score_answer_fn expects."""
    rg = reasoning_gym.create_dataset(task, size=size, seed=seed, **cfg)
    rows = {"prompt": [], "answer": [], "source": [], "metadata": []}
    for e in rg:
        rows["prompt"].append(PROMPT_TMPL.format(q=e["question"]))
        rows["answer"].append("" if e["answer"] is None else str(e["answer"]))
        rows["source"].append(e["metadata"].get("source_dataset", task))
        rows["metadata"].append(json.dumps(e["metadata"]))
    return Dataset.from_dict(rows)


# --------------------------------------------------------------------------- EXACT verifier reward
# cache one scorer per source dataset (callable: (answer_str, entry_dict) -> float)
_SCORERS: dict[str, callable] = {}


def _scorer(source: str):
    fn = _SCORERS.get(source)
    if fn is None:
        fn = _SCORERS[source] = get_score_answer_fn(source)
    return fn


def make_reward(strict: bool = True):
    """Build the TRL reward_func. `strict` thresholds reasoning-gym's (sometimes
    partial-credit) score to a hard 1.0/0.0 so the GRPO advantage is crisp."""

    def reward_exact(completions, answer, source, metadata, **kwargs):
        out = []
        for comp, gold, src, meta in zip(completions, answer, source, metadata):
            pred = extract_answer(comp if isinstance(comp, str) else str(comp))
            entry = {"answer": gold, "metadata": json.loads(meta)}
            try:
                s = float(_scorer(src)(pred, entry))
            except Exception:
                s = 0.0
            out.append(1.0 if (strict and s >= 0.999) else s)
        return out

    reward_exact.__name__ = "rg_exact_verifier"
    return reward_exact


# --------------------------------------------------------------------------- model (policy)
def build_model():
    """The POLICY model. *** ORGAN INSERTION POINT ***: swap this returned module
    for the organ-augmented OLMo and the rest of the pipeline is unchanged."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.config.use_cache = False  # required for gradient checkpointing
    return model


def lora_cfg():
    # "LoRA Without Regret": adapt ALL linears incl MLP (gate/up/down), not just attn.
    return LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear",
    )


# --------------------------------------------------------------------------- main
def main(reward_builder=None):
    """reward_builder: optional zero-arg callable returning a TRL reward_func. When provided (e.g. the
    organ-as-process-reward mix from clair.organ.train.rlvr), it REPLACES the default exact-verifier
    reward — this is the wired ORGAN-AS-PROCESS-REWARD path (was only an insertion-point comment)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="chain_sum")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end sanity (few steps)")
    ap.add_argument("--out", default="runs/rlvr_pipeline")
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--per_device_bs", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_completion", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--loss_type", default="dr_grpo")
    args = ap.parse_args()

    if args.smoke:
        args.steps = 8
        args.num_generations = 4
        args.per_device_bs = 8
        args.grad_accum = 1
        args.max_completion = 96

    print(f"[rlvr] task={args.task} steps={args.steps} G={args.num_generations} "
          f"bs={args.per_device_bs} accum={args.grad_accum} loss={args.loss_type}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # EASY train split / harder OOD eval split (difficulty knob = term/digit counts).
    # Train is deliberately easy so the weak base model lands SOME correct rollouts:
    # GRPO needs non-zero variance within a group (all-wrong -> zero advantage -> no signal).
    train_ds = build_split(args.task, size=512, seed=0,
                           min_terms=2, max_terms=3, min_digits=1, max_digits=1)
    eval_ds = build_split(args.task, size=64, seed=999,
                          min_terms=3, max_terms=5, min_digits=1, max_digits=3)
    print(f"[rlvr] train={len(train_ds)} eval={len(eval_ds)}  sample prompt:\n"
          f"{train_ds[0]['prompt']!r}\n  gold={train_ds[0]['answer']!r}")

    cfg = GRPOConfig(
        output_dir=args.out,
        # ---- objective: Dr.GRPO (no length bias, no std-scaling) ----
        loss_type=args.loss_type,
        scale_rewards=False,
        beta=0.0,                       # no KL -> no reference-model copy (saves VRAM)
        num_iterations=1,
        epsilon=0.2,
        # ---- generation (HF generate; vLLM OFF for organ compatibility) ----
        use_vllm=False,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion,
        temperature=1.0,
        top_p=1.0,
        # ---- batching / memory (sized for ONE L40S 46GB) ----
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        # ---- optim / schedule ----
        learning_rate=args.lr,
        max_steps=args.steps,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        # ---- logging / eval ----
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        log_completions=True,
        num_completions_to_print=2,
    )

    reward_funcs = reward_builder() if reward_builder is not None else make_reward(strict=True)
    print(f"[rlvr] reward = {getattr(reward_funcs, '__name__', reward_funcs)}")
    trainer = GRPOTrainer(
        model=build_model(),
        reward_funcs=reward_funcs,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=lora_cfg(),
    )

    t0 = time.time()
    trainer.train()
    print(f"[rlvr] done in {time.time()-t0:.0f}s")

    # ---- post-hoc: a couple sample generations + their reward, on EVAL (OOD) ----
    model = trainer.model
    model.eval()
    reward = make_reward(strict=False)
    print("\n[rlvr] === sample generations (eval/OOD) ===")
    for i in range(3):
        ex = eval_ds[i]
        ids = tok(ex["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=args.max_completion,
                                  do_sample=True, temperature=0.7, top_p=0.95,
                                  pad_token_id=tok.pad_token_id)
        comp = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        r = reward([comp], [ex["answer"]], [ex["source"]], [ex["metadata"]])[0]
        print(f"--- ex{i} gold={ex['answer']!r} extracted={extract_answer(comp)!r} reward={r:.3f}")
        print("    completion:", repr(comp[:300]))


if __name__ == "__main__":
    main()
