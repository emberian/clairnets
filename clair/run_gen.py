"""Train the GENERATIVE hybrid-deductive OLMo (LoRA + organ woven into lm_head) and run the
decisive measurements:

  1. Generative accuracy vs BASE OLMo few-shot on the SAME generate-the-answer task (in-dist + OOD).
  2. THE CAUSAL ABLATION: generate twice -- organ gate live vs organ gate forced to 0. A drop when
     the organ is zeroed = the LM's token selection causally depends on the organ.
  3. LoRA needed? compare --lora vs --no_lora for the generative task.

  python -m clair.run_gen --smoke
  python -m clair.run_gen --steps 1200 --train_n 4,6 --test_n 5,8,11
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np
import torch

from . import gen_augmented as G
from . import run_augmented as R
from . import augmented as A
from . import induce as I


@torch.no_grad()
def gen_accuracy(model, pool, tok, Nmax, K, colors, dev, n, organ_on=True, use_base=False,
                 fewshot=None, max_new=8):
    model.eval()
    det_tot = det_right = 0
    ab_tot = ab_right = 0
    pred_ab = pred_ab_wrong = 0
    correct = tot = 0
    junk = 0
    for p in pool[:n]:
        if fewshot is not None:
            cont, pred = model.generate_fewshot(p, tok, Nmax, K, colors, dev, fewshot, max_new=max_new)
        else:
            cont, pred = model.generate(p, tok, Nmax, K, colors, dev, max_new=max_new,
                                        organ_on=organ_on, use_base=use_base)
        true_det = p["determined"]
        if pred == -2:
            junk += 1
        if pred is None:
            pred_ab += 1
            if true_det:
                pred_ab_wrong += 1
        if true_det:
            det_tot += 1
            ok = (pred == p["answer"])
            det_right += int(ok)
        else:
            ab_tot += 1
            ok = (pred is None)
            ab_right += int(ok)
        correct += int(ok); tot += 1
    prec = (pred_ab - pred_ab_wrong) / max(1, pred_ab)
    rec = ab_right / max(1, ab_tot)
    return {"overall": correct / max(1, tot), "det_acc": det_right / max(1, det_tot),
            "abstain_prec": prec, "abstain_rec": rec, "junk": junk / max(1, tot), "n": tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--aux", type=float, default=0.5)
    ap.add_argument("--woven_layers", default="11,13,15")
    ap.add_argument("--woven_heads", type=int, default=8)
    ap.add_argument("--woven_dh", type=int, default=512)
    ap.add_argument("--write_head", choices=["pool", "grounded"], default="grounded")
    ap.add_argument("--edge_window", type=int, default=7)
    ap.add_argument("--latch", dest="latch", action="store_true", default=True,
                    help="vocab-anchored latch: organ verdict -> color-word embedding -> residual")
    ap.add_argument("--no_latch", dest="latch", action="store_false")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--train_n", default="4,6")
    ap.add_argument("--test_n", default="5,8,11")
    ap.add_argument("--pool_size", type=int, default=6000)
    ap.add_argument("--eval_n", type=int, default=160)
    ap.add_argument("--base_n", type=int, default=120)
    ap.add_argument("--max_new", type=int, default=8)
    ap.add_argument("--lora", dest="lora", action="store_true", default=True)
    ap.add_argument("--no_lora", dest="lora", action="store_false")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default="olmo2-1b")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no_base", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.steps = min(a.steps, 80); a.pool_size = 1500; a.eval_n = 48; a.base_n = 32
    colors = A.COLORS
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = A.resolve_host(a.host)
    print("loading host", a.host, "->", model_id, flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    olmo = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(dev)
    cfg = olmo.config
    print(f"host: {cfg.num_hidden_layers} layers, hidden {cfg.hidden_size}, "
          f"{sum(p.numel() for p in olmo.parameters())/1e9:.2f}B params", flush=True)

    # ----- no-op check pre-LoRA: stash base logits on a fixed sample -----
    sample_text = ("Node A is red. Node A and node B must be different colors. "
                   "Question: what color is node B? Answer: cannot be determined")
    s_ids = tok(sample_text, return_tensors="pt").to(dev)
    with torch.no_grad():
        base_logits = olmo(input_ids=s_ids["input_ids"]).logits.float()

    if a.lora:
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
        olmo = get_peft_model(olmo, lcfg)
        n_lora = sum(p.numel() for p in olmo.parameters() if p.requires_grad)
        print(f"LoRA on: r={a.lora_r} alpha={a.lora_alpha}, {n_lora:,} LoRA params", flush=True)
    else:
        for p in olmo.parameters():
            p.requires_grad_(False)
        print("LoRA off: OLMo fully frozen (only organ + woven adapters train)", flush=True)

    Nmax = max(int(x) for x in a.test_n.split(",")) + 1
    model = G.GenAugmentedOLMo(olmo, tok, cfg, Nmax, a.k, T=a.T, write_head=a.write_head,
                               edge_window=a.edge_window, woven_layers=a.woven_layers,
                               woven_heads=a.woven_heads, woven_dh=a.woven_dh, latch=a.latch).to(dev)
    # latch needs OLMo's color-word input embeddings (cached; not trained)
    model.set_color_embeds(G._unwrap_causal(olmo).get_input_embeddings().weight.detach(), colors)
    # organ/woven modules stay fp32 for stable Adam (they cast bf16 host activations internally)
    print(f"woven layers {model.woven_layers}  latch={a.latch}  trainable {G.n_trainable(model):,}  Nmax={Nmax}", flush=True)

    # ----- no-op-at-init: lm_head logits with organ woven (gate=0) + LoRA(=0) vs base OLMo -----
    smoke_batch = G.build_gen_batch(
        [I.gen_problem(5, a.k, np.random.default_rng(1)) for _ in range(2)], tok, Nmax, a.k, colors, dev)
    ws0, wsm0, anc0, _ = model.build_organ(smoke_batch["org"])
    with torch.no_grad():
        gated = model.gen_logits(smoke_batch["f_ids"], smoke_batch["f_attn"], ws0, wsm0, anc0, organ_on=True).float()
        # recompute base on the same f_ids (LoRA disabled) for an exact comparison
        if hasattr(model.olmo, "disable_adapter"):
            with model.olmo.disable_adapter():
                base_f = model.olmo(input_ids=smoke_batch["f_ids"], attention_mask=smoke_batch["f_attn"]).logits.float()
        else:
            base_f = model.olmo(input_ids=smoke_batch["f_ids"], attention_mask=smoke_batch["f_attn"]).logits.float()
    gap = float((gated - base_f).abs().max())
    print(f"NO-OP AT INIT (woven gate=0 + LoRA=0): max|gated_lm_logits - base_lm_logits| = {gap:.3e}", flush=True)

    # ----- pools -----
    n_lo, n_hi = (int(x) for x in a.train_n.split(","))
    prng = np.random.default_rng(a.seed + 999)
    t0 = time.time()
    train_pool = R.build_pool(n_lo, n_hi, a.k, a.pool_size, prng)
    eval_pools = {"train": R.build_pool(n_lo, n_hi, a.k, a.eval_n, prng)}
    for tn in (int(x) for x in a.test_n.split(",")):
        eval_pools[tn] = R.build_pool(tn, tn, a.k, max(a.eval_n, a.base_n), prng)
    print(f"pools built in {time.time()-t0:.0f}s (train {len(train_pool)})", flush=True)

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)

    def lr_at(step):  # linear warmup -> cosine decay to ~0 (stops the post-grounding thrashing)
        if step < a.warmup:
            return step / max(1, a.warmup)
        prog = (step - a.warmup) / max(1, a.steps - a.warmup)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    rng = np.random.default_rng(a.seed)
    log = []
    t0 = time.time()
    model.train()
    for s in range(1, a.steps + 1):
        idxs = rng.integers(0, len(train_pool), a.bs).tolist()
        batch = G.build_gen_batch([train_pool[i] for i in idxs], tok, Nmax, a.k, colors, dev)
        loss, parts, _ = model.loss(batch, aux=a.aux)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0); opt.step(); sched.step()
        if s % max(1, a.steps // 10) == 0 or s == 1:
            ind = gen_accuracy(model, eval_pools["train"], tok, Nmax, a.k, colors, dev,
                               n=min(64, a.eval_n), organ_on=True, max_new=a.max_new)
            model.train()
            print(f"  step {s:5d} loss {loss.item():.3f} ({parts})  GEN in-dist overall {ind['overall']*100:4.1f}%"
                  f"  detAcc {ind['det_acc']*100:4.1f} absR {ind['abstain_rec']*100:4.1f} junk {ind['junk']*100:.0f}"
                  f"  {time.time()-t0:.0f}s", flush=True)
            log.append({"step": s, "loss": float(loss.detach()), **ind})

    test_keys = [int(x) for x in a.test_n.split(",")]
    # ===== 1+2: generative accuracy (organ on) and THE CAUSAL ABLATION (organ off) =====
    print("\n=== GENERATIVE accuracy: ORGAN-ON vs ORGAN-ZEROED (causal ablation) ===", flush=True)
    organ_on, organ_off = {}, {}
    for tn in test_keys:
        on = gen_accuracy(model, eval_pools[tn], tok, Nmax, a.k, colors, dev, n=a.eval_n,
                          organ_on=True, max_new=a.max_new)
        off = gen_accuracy(model, eval_pools[tn], tok, Nmax, a.k, colors, dev, n=a.eval_n,
                           organ_on=False, max_new=a.max_new)
        organ_on[tn] = on; organ_off[tn] = off
        drop = (on["overall"] - off["overall"]) * 100
        print(f"  N={tn:2d}  ORGAN-ON overall {on['overall']*100:4.1f} (det {on['det_acc']*100:4.1f} absR {on['abstain_rec']*100:4.1f})"
              f"  | ORGAN-0 overall {off['overall']*100:4.1f} (det {off['det_acc']*100:4.1f})"
              f"  | CAUSAL DROP {drop:+4.1f} pts  (n={on['n']})", flush=True)

    # ===== 1: base OLMo few-shot generative =====
    base_gen = {}
    if not a.no_base:
        print("\n=== BASE OLMo few-shot (generative, same task) ===", flush=True)
        for tn in test_keys:
            r = gen_accuracy(model, eval_pools[tn], tok, Nmax, a.k, colors, dev, n=a.base_n,
                             fewshot=R.FEWSHOT, max_new=a.max_new)
            base_gen[tn] = r
            print(f"  N={tn:2d}  overall {r['overall']*100:4.1f}  detAcc {r['det_acc']*100:4.1f}"
                  f"  absR {r['abstain_rec']*100:4.1f}  junk {r['junk']*100:.0f}  (n={r['n']})", flush=True)

    res = {"args": vars(a), "noop_gap": gap, "log": log,
           "organ_on": {str(k): v for k, v in organ_on.items()},
           "organ_off": {str(k): v for k, v in organ_off.items()},
           "base_gen": {str(k): v for k, v in base_gen.items()}}
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "runs",
                                f"gen_{'lora' if a.lora else 'frozen'}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("\nwrote", out, flush=True)


if __name__ == "__main__":
    main()
