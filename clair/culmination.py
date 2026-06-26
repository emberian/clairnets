"""clair/culmination.py — THE CULMINATION EXPERIMENT.

Wire a FROZEN learned deduction organ into the working RLVR pipeline on OLMo-3-Base-7B, using the
organ's OWN narrowing as a process reward (the LSRL lift), and test whether the organ helps a
properly RL-trained 7B model.

Staged build (each stage de-risked + reported):
  STAGE 1  OLMo-3-Base-7B in the base TRL-GRPO loop (LoRA, use_vllm=False, beta=0). Confirm it FITS
           one L40S at 7B and that reward goes UP on an easy reasoning-gym task.
  STAGE 2  Wire the FROZEN organ into the policy: augmented OLMo-3-7B that injects the frozen organ's
           narrowed lattice via the proven structured gamma (clair.oracle_readout.OracleGamma).
           Bootstrap+freeze a general organ (run_glados_staged STAGE-1 recipe). No-op @ init +
           a causal-control check (corrupt the lattice -> generation changes).
  STAGE 3  Organ-as-process-reward: per-step candidate-set cardinality drop (progress) + soundness
           (no dedP survivor eliminated), mixed 0.7*outcome + 0.3*process into GRPO.
  STAGE 4  Matched 2x2 {base-7B, +organ} x {SFT, SFT+RLVR} + random-reward control; pass@1 AND
           pass@k on a held-out OOD split.

The organ-applicable fuel is clair CSP/curriculum problems rendered as text (we KNOW the underlying
CSP, so we can featurize it for the organ and locate entity mentions for gamma) WITH an exact
verifier. reasoning-gym is the base-RLVR sanity fuel for Stage 1 (no alpha=text->program needed).

USAGE
  python -m clair.culmination --stage 1 --smoke           # base 7B RLVR fits + learns
  python -m clair.culmination --stage 1 --steps 60
  python -m clair.culmination --stage 2 --smoke           # organ wired, no-op + causal control
  python -m clair.culmination --stage 3 --smoke           # process reward computed + used
  python -m clair.culmination --stage 4 --smoke           # matched comparison
"""
from __future__ import annotations

import argparse
import json
import time

import torch

MODEL_ID = "allenai/Olmo-3-1025-7B"  # OLMo-3 base 7B: Olmo3ForCausalLM, 32L, hidden 4096


# =============================================================================== STAGE 1
def stage1(args):
    """Base OLMo-3-Base-7B in the TRL-GRPO loop. Confirm it FITS one L40S and reward rises."""
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from . import rlvr_pipeline as RP

    print(f"[stage1] model={MODEL_ID}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # train split tuned so the 7B base lands SOME-but-not-all correct rollouts (non-zero GRPO
    # variance AND headroom for reward to rise); harder OOD eval split.
    train_ds = RP.build_split(args.task, size=args.train_size, seed=0,
                              min_terms=args.min_terms, max_terms=args.max_terms,
                              min_digits=args.min_digits, max_digits=args.max_digits)
    eval_ds = RP.build_split(args.task, size=64, seed=999,
                             min_terms=args.max_terms, max_terms=args.max_terms + 2,
                             min_digits=args.max_digits, max_digits=args.max_digits + 1)
    print(f"[stage1] task={args.task} train={len(train_ds)} eval={len(eval_ds)}\n"
          f"  sample prompt: {train_ds[0]['prompt']!r}\n  gold={train_ds[0]['answer']!r}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False

    cfg = GRPOConfig(
        output_dir=args.out,
        loss_type="dr_grpo", scale_rewards=False,
        beta=0.0,                       # NO KL -> no reference-model copy (saves ~14GB at 7B)
        num_iterations=1, epsilon=0.2,
        use_vllm=False,                 # HF generate (organ-compatible path)
        num_generations=args.group,
        max_completion_length=args.max_completion,
        temperature=1.0, top_p=1.0,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        learning_rate=args.lr, max_steps=args.steps,
        lr_scheduler_type="constant_with_warmup", warmup_steps=3,
        logging_steps=1, save_strategy="no", report_to=[],
        log_completions=True, num_completions_to_print=2,
    )
    trainer = GRPOTrainer(
        model=model, reward_funcs=RP.make_reward(strict=True), args=cfg,
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tok, peft_config=RP.lora_cfg(),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"\n[stage1] DONE in {dt:.0f}s  peak_vram={peak:.1f}GB  "
          f"(L40S has 46GB -> {'FITS' if peak < 45 else 'TIGHT/OOM-RISK'})", flush=True)

    # report reward trajectory from the logged history
    hist = [h for h in trainer.state.log_history if "reward" in h]
    if hist:
        r = [h["reward"] for h in hist]
        first = sum(r[:max(1, len(r)//5)]) / max(1, len(r)//5)
        last = sum(r[-max(1, len(r)//5):]) / max(1, len(r)//5)
        print(f"[stage1] reward: first-fifth {first:.3f} -> last-fifth {last:.3f}  "
              f"({'UP' if last > first else 'flat/down'})  full={[round(x,2) for x in r]}", flush=True)
    print(f"[stage1] config: bs={args.bs} group={args.group} accum={args.grad_accum} "
          f"max_completion={args.max_completion} max_prompt={args.max_prompt} lr={args.lr}", flush=True)


# =============================================================================== organ (frozen)
def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def bootstrap_or_load_organ(args, dev):
    """Bootstrap a GENERAL frozen deduction organ (run_glados_staged STAGE-1 recipe) or load a saved
    one. Returns (organ, meta). The organ is FROZEN (eval, no grad)."""
    import os
    from .proposer import FactorGraphProposer
    from . import run_glados_staged as G

    if os.path.exists(args.organ_pt):
        blob = torch.load(args.organ_pt, map_location=dev)
        m = blob["meta"]
        organ = FactorGraphProposer("full", m["N_MAX"], m["D_MAX"], m["M_MAX"], m["A_MAX"],
                                    d=m["d"], R=m["R"], ds=min(4, m["R"])).to(dev)
        organ.load_state_dict(blob["state"])
        print(f"[organ] loaded {args.organ_pt}  d={m['d']} params={m['params']:,} R={m['R']}", flush=True)
    else:
        print(f"[organ] bootstrapping GENERAL organ ({args.organ_steps} steps)...", flush=True)
        organ, d, npar, log = G.train_organ(dev, G.RUNGS, target=3.0e5, steps=args.organ_steps,
                                            pool=128, R=8, lr=3e-4, seed=args.seed)
        m = {"N_MAX": G.N_MAX, "D_MAX": G.D_MAX, "M_MAX": G.M_MAX, "A_MAX": G.A_MAX,
             "d": d, "params": npar, "R": 8}
        os.makedirs(os.path.dirname(args.organ_pt) or ".", exist_ok=True)
        torch.save({"state": organ.state_dict(), "meta": m}, args.organ_pt)
        print(f"[organ] bootstrapped + saved {args.organ_pt}  d={d} params={npar:,}", flush=True)
    for p in organ.parameters():
        p.requires_grad_(False)
    organ.eval()
    return organ, m


@torch.no_grad()
def organ_trajectories(organ, csps, dev, theta=0.5, R_max=12):
    """Run the FROZEN organ's MONOTONE meet step-by-step from the full domain on each csp, recording
    the per-cell surviving sets after EACH step (the narrowing trajectory). Returns, per csp, a list
    of per-cell domain tuples [S^(0), S^(1), ..., S^(R)] (monotone: S^(t+1) subseteq S^(t))."""
    from . import run_glados_staged as G
    organ.eval()
    feat = G.featurize([(c, c.full()) for c in csps], dev)
    vm = feat["var_mask"].clone()
    traj = [[G.dom_from_mask(vm[i].cpu().numpy(), csps[i]) for i in range(len(csps))]]
    for _ in range(R_max):
        b, cls, _ = G.fwd(organ, feat, vm)
        new_vm = G.meet(vm, b, theta)
        if bool((new_vm == vm).all()):
            break
        vm = new_vm
        traj.append([G.dom_from_mask(vm[i].cpu().numpy(), csps[i]) for i in range(len(csps))])
    # transpose: per-csp list of per-step domains
    return [[traj[t][i] for t in range(len(traj))] for i in range(len(csps))]


# =============================================================================== STAGE 2
def stage2(args):
    """Wire the FROZEN organ into OLMo-3-7B via the proven structured gamma (OracleReadout): confirm
    NO-OP @ init, a SHORT supervised gamma-readout warmup (RLVR fixes calibration, not capacity:
    TransNAR transfers with NO RL -> the interface is pretrained supervised), then the CAUSAL-CONTROL
    table (true vs shuffle/permute/corrupt/zero) -> does generation CAUSALLY read the organ's lattice
    even with the full problem text present?"""
    import types
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from . import run_glados_staged as G
    from . import oracle_readout as O

    dev = _device()
    organ, om = bootstrap_or_load_organ(args, dev)

    print("\n[stage2] FROZEN ORGAN RECALL vs EXACT dedP (sanity, in-dist):", flush=True)
    recall = G.organ_recall_report(organ, dev, G.RUNGS, splits=("id",), n_per=args.organ_recall_n,
                                   seed=args.seed + 5)
    for rg in G.RUNGS:
        print(G._rrow(rg, recall[(rg, "id")]), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    probe = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    D = probe.config.hidden_size
    nL = probe.config.num_hidden_layers
    del probe
    print(f"[stage2] OLMo-3-7B: {nL} layers, hidden {D}; inject at layer {args.inject_layer}", flush=True)

    # ----- build pools (frozen organ's lattice injected); train cells & full renderings -----
    prng = np.random.default_rng(args.seed + 11)
    train_pools = G.build_pool(prng, organ, dev, G.RUNGS, "id", args.per_rung, ("cells", "full"))
    train_cells, train_full = train_pools["cells"], train_pools["full"]
    eval_id = G.build_pool(prng, organ, dev, G.RUNGS, "id", max(8, args.per_rung // 4), ("cells", "full"))

    # ----- supervised gamma-readout warmup on OLMo-3-7B (reuses the proven recipe) -----
    a = types.SimpleNamespace(lora_r=16, inject_layer=args.inject_layer, gamma_hidden=256,
                              lora_lr=2e-4, gamma_lr=1e-3, seed=args.seed, cells_frac=0.4,
                              steps=args.readout_steps, bs=args.bs)
    model = G.train_readout(organ, (MODEL_ID, D, nL), tok, dev, a, train_cells, train_full,
                            eval_id["cells"])

    # ----- DELIVERABLE: causal-control table (intervene on the frozen organ's lattice) -----
    print("\n[stage2] ===== CAUSAL-CONTROL TABLE (woven OLMo-3-7B; full text present; acc %) =====",
          flush=True)
    print(f"  {'suite':22s} {'true':>6s} {'shuffle':>8s} {'permute':>8s} {'corrupt':>8s} {'zero':>6s}",
          flush=True)
    for sname, pool in [("in-dist (full)", eval_id["full"]), ("in-dist (cells)", eval_id["cells"])]:
        row = {c: G.score_gen(model, pool, tok, dev, control=c, inject=True, bs=args.bs)["overall"]
               for c in ("true", "shuffle", "permute", "corrupt", "zero")}
        print(f"  {sname:22s} {row['true']*100:6.1f} {row['shuffle']*100:8.1f} {row['permute']*100:8.1f}"
              f" {row['corrupt']*100:8.1f} {row['zero']*100:6.1f}", flush=True)
    # base/text-LoRA contrast on in-dist full
    w = G.score_gen(model, eval_id["full"], tok, dev, control="true", inject=True, bs=args.bs)
    tl = G.score_gen(model, eval_id["full"], tok, dev, inject=False, use_base=False, bs=args.bs)
    bs_ = G.score_gen(model, eval_id["full"], tok, dev, inject=False, use_base=True,
                      fewshot=G.FEWSHOT, bs=args.bs)
    print(f"\n[stage2] in-dist(full): WOVEN {w['overall']*100:.1f}%  TEXT-LoRA {tl['overall']*100:.1f}%"
          f"  BASE {bs_['overall']*100:.1f}%", flush=True)
    print("[stage2] If true >> corrupt/zero WITH text present => the LM causally WIELDS the organ.",
          flush=True)


# =============================================================================== CSP-as-text fuel
ABSTAIN = "cannot be determined"


def _ser_facts(facts):
    return [["alldiff", list(f[1])] if f[0] == "alldiff" else list(f) for f in facts]


def _deser_facts(sfacts):
    out = []
    for f in sfacts:
        out.append(("alldiff", tuple(f[1])) if f[0] == "alldiff" else tuple(f))
    return out


def build_csp_dataset(rungs, split, per_rung, seed):
    """A reasoning dataset of curriculum CSP problems rendered as text (canonical English, ending
    'Answer:'), with the EXACT gold answer + everything the rewards need (serialized CSP) in `meta`.
    Returns a datasets.Dataset with columns: prompt, answer, meta(json)."""
    import numpy as np
    from datasets import Dataset
    from . import curriculum as CU
    from . import run_glados_staged as G

    rng = np.random.default_rng(seed)
    rows = {"prompt": [], "answer": [], "meta": []}
    for rg in rungs:
        for _ in range(per_rung):
            p = G.make_problem(rng, rg, split)
            prompt = CU.canonical_render(p) + " Answer:"
            rows["prompt"].append(prompt)
            rows["answer"].append(CU.canonical_answer(p))
            rows["meta"].append(json.dumps({
                "relation": p.relation, "n": p.n, "d": p.d, "query": int(p.query),
                "determined": bool(p.determined), "vnames": list(p.vnames),
                "gold": CU.canonical_answer(p), "facts": _ser_facts(p.facts)}))
    return Dataset.from_dict(rows)


def parse_csp_answer(text: str, vnames) -> int:
    """Map a chatty completion to an answer index: i in [0,len(vnames)) for a value name, len(vnames)
    for ABSTAIN, or -1 if unparseable. Priority: text after the LAST 'Answer:'."""
    if not text:
        return -1
    low = text.lower()
    seg = low.rsplit("answer:", 1)[-1] if "answer:" in low else low
    if "cannot" in seg or "undetermined" in seg or "unknown" in seg:
        return len(vnames)
    # match a value name (longest first to avoid 'red' in 'reddish' style collisions)
    for i in sorted(range(len(vnames)), key=lambda j: -len(vnames[j])):
        if vnames[i].lower() in seg:
            return i
    return -1


# =============================================================================== rewards
def _gold_idx(meta):
    """Gold answer index: value index if determined, else ABSTAIN (= len(vnames))."""
    vnames = meta["vnames"]
    gold = meta.get("gold", ABSTAIN)
    return vnames.index(gold) if (meta["determined"] and gold in vnames) else len(vnames)


def outcome_score(meta_json, completion) -> float:
    """Exact-match outcome reward: the parsed answer equals the gold answer (1.0/0.0)."""
    meta = json.loads(meta_json)
    pred = parse_csp_answer(completion if isinstance(completion, str) else str(completion),
                            meta["vnames"])
    return 1.0 if pred == _gold_idx(meta) else 0.0


class _FakeProblem:
    """Minimal carrier so run_glados_staged.organ_csp (reads .n/.d/.facts) can rebuild the CSP."""
    def __init__(self, n, d, facts):
        self.n, self.d, self.facts = n, d, facts


def make_outcome_reward(weight=0.7):
    def reward_outcome(completions, meta, **kw):
        return [weight * outcome_score(mj, c) for c, mj in zip(completions, meta)]
    reward_outcome.__name__ = "outcome"
    return reward_outcome


def make_process_reward(organ, dev, weight=0.3, R_max=12):
    """PROCESS reward = the organ's OWN narrowing used as an exact, unhackable per-step grader (the
    LSRL lift). For each rollout: run the FROZEN organ's monotone meet trajectory on the problem's
    CSP, then score the rollout's PARSED answer by how DEEP into the narrowing it SURVIVES at the
    query cell (survival-depth) -- a dense, per-rollout, exact version of LSRL's PS/IQS:
       PS (progress)  = fraction of narrowing depths t at which the emitted value is still a survivor.
       IQS (soundness)= gate to 0 if the organ ever eliminated a true (exact-dedP) survivor (unsound).
    Outcome stays primary (weight 0.7 vs 0.3); this only SHAPES, anti-hacking."""
    from . import csp as C
    from . import run_glados_staged as G

    def reward_process(completions, meta, **kw):
        # cache the (expensive) organ trajectory + soundness per UNIQUE problem (key = meta json str)
        cache = {}
        for mj in meta:
            if mj in cache:
                continue
            m = json.loads(mj)
            oc = G.organ_csp(_FakeProblem(m["n"], m["d"], _deser_facts(m["facts"])))
            traj = organ_trajectories(organ, [oc], dev, R_max=R_max)[0]       # per-step domains
            q = m["query"]
            qsurv = [set(dom[q]) for dom in traj]                            # query surviving sets
            ded, _ = C.to_fixpoint(C.exact_dedP, oc, oc.full())
            sound = set(ded[q]).issubset(qsurv[0]) and set(qsurv[-1]).issuperset(set(ded[q]))
            cache[mj] = (qsurv, set(ded[q]), bool(sound))
        out = []
        for c, mj in zip(completions, meta):
            m = json.loads(mj)
            qsurv, ded_q, sound = cache[mj]
            if not sound:
                out.append(0.0)
                continue
            vnames = m["vnames"]
            pred = parse_csp_answer(c if isinstance(c, str) else str(c), vnames)
            R = len(qsurv)
            if pred == len(vnames):                       # ABSTAIN: reward iff organ stays >1-valued
                s = 1.0 if len(qsurv[-1]) > 1 else 0.0
            elif 0 <= pred < len(vnames):
                s = sum(1.0 for ss in qsurv if pred in ss) / R       # survival-depth of emitted value
                if len(qsurv[-1]) > 1:
                    s *= 0.5                              # penalize over-commitment when undetermined
            else:
                s = 0.0
            out.append(weight * s)
        return out
    reward_process.__name__ = "organ_process"
    return reward_process


def make_random_reward(weight=0.7, seed=0):
    """Random-reward DUMMY control (per rl_design_lessons / SPUR): on OLMo this should do ~nothing;
    a gain here exposes a clipping-amplified shortcut rather than real organ signal."""
    import random
    rng = random.Random(seed)

    def reward_random(completions, **kw):
        return [weight * float(rng.random() < 0.3) for _ in completions]
    reward_random.__name__ = "random_ctrl"
    return reward_random


# =============================================================================== STAGE 3
def _build_policy_7b(tok):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    return model


def stage3(args):
    """Organ-as-process-reward: demonstrate the process reward is COMPUTED from the organ's own
    narrowing + VARIES per rollout, then run a short GRPO with 0.7*outcome + 0.3*process."""
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from . import rlvr_pipeline as RP

    dev = _device()
    organ, om = bootstrap_or_load_organ(args, dev)

    rungs = ["coloring", "equality", "ordering", "arithmetic", "alldiff"]
    train_ds = build_csp_dataset(rungs, "id", args.per_rung, seed=1)
    print(f"[stage3] CSP dataset: {len(train_ds)} rows  sample:\n  {train_ds[0]['prompt']!r}\n"
          f"  gold={train_ds[0]['answer']!r}", flush=True)

    out_r = make_outcome_reward(weight=0.7)
    proc_r = make_process_reward(organ, dev, weight=0.3, R_max=args.R_max)

    # --- DEMONSTRATION: process reward varies per (synthetic) rollout-answer ---
    print("\n[stage3] === process-reward demo (organ narrowing credits near-misses) ===", flush=True)
    demo = train_ds.select(range(min(4, len(train_ds))))
    for ex in demo:
        meta = json.loads(ex["meta"]); vn = meta["vnames"]
        cands = [f"Answer: {v}" for v in vn] + [f"Answer: {ABSTAIN}"]
        prc = proc_r(cands, [ex["meta"]] * len(cands))
        otc = out_r(cands, [ex["meta"]] * len(cands))
        print(f"  gold={ex['answer']!r:24s} det={meta['determined']}", flush=True)
        for c, p, o in zip(cands, prc, otc):
            print(f"     {c:28s}  outcome(0.7) {o:.2f}   process(0.3) {p:.3f}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = GRPOConfig(
        output_dir=args.out, loss_type="dr_grpo", scale_rewards=False, beta=0.0,
        num_iterations=1, epsilon=0.2, use_vllm=False,
        num_generations=args.group, max_completion_length=args.max_completion,
        temperature=1.0, top_p=1.0,
        per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True, bf16=True,
        learning_rate=args.lr, max_steps=args.steps,
        lr_scheduler_type="constant_with_warmup", warmup_steps=3,
        logging_steps=1, save_strategy="no", report_to=[], log_completions=True,
        num_completions_to_print=2,
    )
    trainer = GRPOTrainer(
        model=_build_policy_7b(tok), reward_funcs=[out_r, proc_r], args=cfg,
        train_dataset=train_ds, processing_class=tok, peft_config=RP.lora_cfg(),
    )
    print("\n[stage3] === short GRPO with outcome+organ-process reward ===", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    hist = trainer.state.log_history
    def col(k):
        return [h[k] for h in hist if k in h]
    print(f"\n[stage3] DONE peak_vram={peak:.1f}GB", flush=True)
    print(f"[stage3] outcome reward:  {[round(x,3) for x in col('rewards/outcome/mean')]}", flush=True)
    print(f"[stage3] process reward:  {[round(x,3) for x in col('rewards/organ_process/mean')]}", flush=True)
    print(f"[stage3] total reward:    {[round(x,3) for x in col('reward')]}", flush=True)


# =============================================================================== STAGE 4
@torch.no_grad()
def passk_eval(model, tok, ds, dev, k=8, max_new=128, temperature=0.8, top_p=0.95, bs=8):
    """pass@1 (greedy) AND pass@k (k samples) exact-match on a held-out split. Returns dict."""
    import numpy as np
    model.eval()
    prev_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = True            # re-enable KV cache for FAST generation (off during train)
    p1 = 0
    solved_any = 0
    n = len(ds)
    for i in range(n):
        ex = ds[i]
        meta = json.loads(ex["meta"]); vn = meta["vnames"]
        gold_idx = _gold_idx(meta)
        ids = tok(ex["prompt"], return_tensors="pt").to(dev)
        # pass@1 greedy
        g = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        comp = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        p1 += int(parse_csp_answer(comp, vn) == gold_idx)
        # pass@k sampled
        hit = False
        for _ in range(k):
            g = model.generate(**ids, max_new_tokens=max_new, do_sample=True,
                               temperature=temperature, top_p=top_p, pad_token_id=tok.pad_token_id)
            comp = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            if parse_csp_answer(comp, vn) == gold_idx:
                hit = True
                break
        solved_any += int(hit)
    if prev_cache is not None:
        model.config.use_cache = prev_cache
    return {"pass@1": p1 / max(1, n), f"pass@{k}": solved_any / max(1, n), "n": n}


def _train_arm(arm, args, tok, organ, dev, train_ds):
    """Train one RLVR arm. arm in {outcome, organ_process, random}. Returns the trained policy."""
    from trl import GRPOConfig, GRPOTrainer
    from . import rlvr_pipeline as RP
    if arm == "outcome":
        rfs = [make_outcome_reward(weight=1.0)]
    elif arm == "organ_process":
        rfs = [make_outcome_reward(weight=0.7), make_process_reward(organ, dev, 0.3, args.R_max)]
    elif arm == "random":
        rfs = [make_random_reward(weight=1.0, seed=args.seed)]
    else:
        raise ValueError(arm)
    cfg = GRPOConfig(
        output_dir=f"{args.out}_{arm}", loss_type="dr_grpo", scale_rewards=False, beta=0.0,
        num_iterations=1, epsilon=0.2, use_vllm=False,
        num_generations=args.group, max_completion_length=args.max_completion,
        temperature=1.0, top_p=1.0,
        per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True, bf16=True, learning_rate=args.lr, max_steps=args.steps,
        lr_scheduler_type="constant_with_warmup", warmup_steps=3,
        logging_steps=5, save_strategy="no", report_to=[],
    )
    trainer = GRPOTrainer(model=_build_policy_7b(tok), reward_funcs=rfs, args=cfg,
                          train_dataset=train_ds, processing_class=tok, peft_config=RP.lora_cfg())
    trainer.train()
    return trainer.model


def stage4(args):
    """Matched comparison: organ-process-reward vs outcome-only vs random-reward control, pass@1 AND
    pass@k on a held-out OOD split. (The 'organ helps under properly-credited RL' question.)"""
    from transformers import AutoTokenizer
    dev = _device()
    organ, om = bootstrap_or_load_organ(args, dev)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rungs = ["coloring", "equality", "ordering", "arithmetic", "alldiff"]
    train_ds = build_csp_dataset(rungs, "id", args.per_rung, seed=1)
    ood_ds = build_csp_dataset(rungs, "ood", args.ood_per, seed=777)
    print(f"[stage4] train={len(train_ds)} ood_eval={len(ood_ds)}  arms={args.arms}", flush=True)

    results = {}
    for arm in args.arms:
        print(f"\n[stage4] ===== ARM: {arm} =====", flush=True)
        model = _train_arm(arm, args, tok, organ, dev, train_ds)
        res = passk_eval(model, tok, ood_ds, dev, k=args.k, max_new=args.eval_max_new)
        results[arm] = res
        print(f"[stage4] {arm}: {res}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[stage4] ===== MATCHED TABLE (held-out OOD) =====", flush=True)
    print(f"  {'arm':16s} {'pass@1':>8s} {f'pass@{args.k}':>9s}", flush=True)
    for arm in args.arms:
        r = results[arm]
        print(f"  {arm:16s} {r['pass@1']*100:7.1f}% {r[f'pass@{args.k}']*100:8.1f}%", flush=True)
    out_path = f"{args.out}_stage4.json"
    json.dump(results, open(out_path, "w"), indent=1)
    print(f"[stage4] wrote {out_path}", flush=True)


# =============================================================================== dispatch
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="runs/culmination")
    # stage-1 / RLVR knobs (sized for ONE L40S at 7B)
    ap.add_argument("--task", default="chain_sum")
    ap.add_argument("--min_terms", type=int, default=4)
    ap.add_argument("--max_terms", type=int, default=8)
    ap.add_argument("--min_digits", type=int, default=2)
    ap.add_argument("--max_digits", type=int, default=3)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--group", type=int, default=6)            # num_generations
    ap.add_argument("--bs", type=int, default=12)              # per_device_train_batch_size (16GB peak leaves room)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_completion", type=int, default=160)
    ap.add_argument("--max_prompt", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--train_size", type=int, default=512)
    # organ knobs (stages 2-4)
    ap.add_argument("--organ_steps", type=int, default=1200)
    ap.add_argument("--organ_pt", default="runs/general_organ.pt")
    ap.add_argument("--inject_layer", type=int, default=20)
    ap.add_argument("--per_rung", type=int, default=120)
    ap.add_argument("--readout_steps", type=int, default=800)
    ap.add_argument("--organ_recall_n", type=int, default=100)
    ap.add_argument("--R_max", type=int, default=12)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--ood_per", type=int, default=12)
    ap.add_argument("--eval_max_new", type=int, default=48)
    ap.add_argument("--arms", nargs="+", default=["outcome", "organ_process", "random"])
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main():
    args = build_argparser().parse_args()
    if args.smoke:
        # tiny end-to-end sanity sized to FIT and run fast at 7B
        args.steps = min(args.steps, 8)
        args.group = 4
        args.bs = 4
        args.grad_accum = 1
        args.max_completion = 96
        args.train_size = 128
        args.organ_steps = min(args.organ_steps, 250)
        args.per_rung = 40
        args.readout_steps = 150
        args.organ_recall_n = 40
        args.k = 4
        args.ood_per = 4
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    if args.stage == 1:
        stage1(args)
    elif args.stage == 2:
        stage2(args)
    elif args.stage == 3:
        stage3(args)
    elif args.stage == 4:
        stage4(args)


if __name__ == "__main__":
    main()
