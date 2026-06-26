"""clair/organ/train.py — THE canonical staged training pipeline for the GLaDOS organ.

Three clean callables, ONE documented flow:  pretrain_organ → weave → rlvr.  Each lifts the
already-validated recipe into a single front-door callable (nothing is re-prototyped — the bodies
delegate to the proven implementations in clair.run_glados_staged / clair.rlvr_pipeline):

  (1) pretrain_organ(...)   STAGE 1 — train the deduction organ STANDALONE on the soundness-asymmetric
                            dominate-dedₚ loss vs the exact per-cell transformer (clair.csp.exact_dedP),
                            on-policy over the 7-rung MIX. Recipe (from the sweep): ~1.5M params, R=12,
                            ~30% composed. Saves the frozen union organ in load_core_organ's format.

  (2) weave(...)            STAGE 2 — graft the organ into a host LM (clair.organ.graft) and train the
                            woven readout: frozen base + LoRA + latent-α compile + FROZEN organ +
                            zero-init γ readout, on the answer-span LM CE. This is the engagement
                            mechanism, lifted out of run_glados_staged's experiment driver into one
                            callable: the TWO-STREAM lever (α reads full text, the LM generates from a
                            fact-ablated cells prompt), J0 direct α-supervision (dominate-dedₚ on α's
                            raw compile, the SATNet grounding fix), and the rich-state calibration /
                            causal-control readout (true/shuffle/permute/corrupt/zero + oracle-override).

  (3) rlvr(...)             STAGE 3 — RL-from-verifiable-rewards (TRL Dr.GRPO) with the exact checker as
                            reward; the organ-as-process-reward (LSRL-style per-step cardinality drop) is
                            the documented insertion point. Sharpens calibration, not capacity; report
                            pass@1 AND pass@k.

Eval is the arbiter, exposed separately as clair.organ.eval.run_eval_suite.

Usage:
  python -m clair.organ.train pretrain --steps 1500 --out runs/general_organ_full.pt
  python -m clair.organ.train weave    --base allenai/OLMo-2-0425-1B --regime hard --steps 2500
  python -m clair.organ.train weave    --smoke                         # tiny end-to-end woven smoke
  python -m clair.organ.train rlvr     --task chain_sum --steps 300
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace


# ============================================================ STAGE 1: pretrain the organ
def pretrain_organ(out="runs/general_organ_full.pt", steps=1500, target=1.5e6, pool=128, R=12,
                   lr=3e-4, seed=0, dev=None):
    """STAGE 1: on-policy dominate-dedₚ training of ONE general narrow organ over the 7-rung MIX, saved
    in the {'state','meta'} format clair.organ.bank.load_core_organ reads. Returns (organ, meta).
    Recipe defaults follow notes/training.md (the sweep): ~1.5M params, R=12, ~30% composed. Delegates
    verbatim to the validated clair.run_glados_staged.train_organ (the dominate-dedₚ loop)."""
    import torch
    from .. import run_glados_staged as G
    dev = dev or G.device()
    print(f"[organ] dominate-dedₚ over rungs={G.RUNGS}  steps={steps} target={target:g} R={R} dev={dev}",
          flush=True)
    organ, d, npar, log = G.train_organ(dev, G.RUNGS, target=target, steps=steps, pool=pool,
                                        R=R, lr=lr, seed=seed)
    for p in organ.parameters():
        p.requires_grad_(False)
    organ.eval()
    meta = {"N_MAX": G.N_MAX, "D_MAX": G.D_MAX, "M_MAX": G.M_MAX, "A_MAX": G.A_MAX,
            "d": d, "params": npar, "R": R}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({"state": organ.state_dict(), "meta": meta}, out)
    print(f"[organ] saved {out}  d={d} params={npar:,} R={R}", flush=True)
    return organ, meta


# back-compat alias (clair.organ.selftest + older notes call train_organ)
train_organ = pretrain_organ


# ============================================================ STAGE 2: weave into a host LM
def _weave_args(base_id, **over):
    """Default woven hyperparameters (the run_glados_staged.main defaults), overridable by kwargs."""
    a = SimpleNamespace(
        base=base_id, steps=2500, warm_steps=400, bs=12, lora_r=16, lora_lr=2e-4, gamma_lr=1e-3,
        alpha_lr=3e-4, gamma_hidden=256, mid_layer=6, inject_layer=12, dctx=256, alpha_dp=384,
        alpha_heads=6, organ_d=128, organ_heads=4, organ_layers=2, organ_T=12, aux_w=0.5,
        alpha_sup_w=1.0, engage_thr=5.0, per_rung_train=400, per_rung_eval=80, det_only=1, seed=0)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def weave(base_id="allenai/OLMo-2-0425-1B", *, regime="hard", two_stream=True, smoke=False,
          out=None, dev=None, **hp):
    """STAGE 2: graft the organ into `base_id` and train the woven readout with the FULL engagement
    mechanism (two-stream lever + J0 α-supervision + calibration / causal-control readout). Returns
    (woven_model, metrics). `regime` selects (rungs, split) from run_glados_staged.LIVE_REGIMES
    {small, hard, large}; `hp` overrides any hyperparameter (steps, warm_steps, lr, layers, ...).

    The host is grafted by clair.organ.graft (which makes train_live_woven model-agnostic via the
    generalized decoder-layer lookup), so this weaves onto ANY of the 7 proven bases identically."""
    import gc
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoConfig
    from .. import run_glados_staged as G

    dev = dev or G.device()
    a = _weave_args(base_id, **hp)
    if smoke:
        a.steps, a.warm_steps, a.per_rung_train, a.per_rung_eval, a.bs = 60, 40, 24, 12, 6
        a.lora_r, a.gamma_hidden, a.organ_d, a.organ_T = 8, 128, 96, 8

    print(f"[weave] base={base_id} regime={regime} two_stream={two_stream} smoke={smoke}", flush=True)
    tok = AutoTokenizer.from_pretrained(base_id)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(base_id)
    D = cfg.hidden_size
    nL = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "num_layers", None)
    a.inject_layer = min(a.inject_layer, nL - 1)
    a.mid_layer = min(a.mid_layer, a.inject_layer - 1)

    rungs, split = G.LIVE_REGIMES[G._REGIME_ALIAS.get(regime, regime)]
    rng_tr = np.random.default_rng(a.seed + 100)
    rng_ev = np.random.default_rng(a.seed + 200)
    train_recs = G.build_live_pool(rng_tr, rungs, split, a.per_rung_train,
                                   det_only=bool(a.det_only), two_stream=two_stream)
    eval_recs = G.build_live_pool(rng_ev, rungs, split, a.per_rung_eval,
                                  det_only=bool(a.det_only), two_stream=two_stream)
    print(f"[weave] data: {len(train_recs)} train / {len(eval_recs)} eval recs  (D={D}, nL={nL})",
          flush=True)

    model = G.train_live_woven((base_id, D, nL), tok, dev, a, train_recs, eval_recs,
                               two_stream=two_stream)
    metrics = G._live_controls(model, eval_recs, tok, dev, a.bs, a.engage_thr, two_stream=two_stream)
    G._report_regime(f"{regime} | two_stream={two_stream}", metrics, a.engage_thr)

    if out:
        from .. import eval_suite as ES
        wcfg = {"kind": "live", "D": D, "K": G.K, "lora_r": a.lora_r, "gamma_hidden": a.gamma_hidden,
                "inject_layer": a.inject_layer, "mid_layer": a.mid_layer, "dctx": a.dctx,
                "alpha_dp": a.alpha_dp, "alpha_heads": a.alpha_heads, "organ_d": a.organ_d,
                "organ_heads": a.organ_heads, "organ_layers": a.organ_layers, "organ_T": a.organ_T}
        ES.save_woven(model, out, base_id=base_id, cfg=wcfg)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, metrics


# ============================================================ STAGE 3: RLVR (organ-as-process-reward)
def rlvr(task="chain_sum", steps=300, *, smoke=False, out="runs/rlvr_pipeline", **over):
    """STAGE 3: RL-from-verifiable-rewards on the woven policy (TRL Dr.GRPO, exact-verifier reward).
    Delegates to the validated clair.rlvr_pipeline harness. The ORGAN-AS-PROCESS-REWARD (LSRL-style
    per-step candidate-set cardinality drop, mixed 0.7·outcome + 0.3·process) plugs in at that file's
    documented ORGAN INSERTION POINT — swap build_model() for the woven policy; the reward loop is
    unchanged."""
    from .. import rlvr_pipeline as RL
    argv = ["clair.rlvr_pipeline", "--task", str(task), "--steps", str(steps), "--out", str(out)]
    if smoke:
        argv.append("--smoke")
    for k, v in over.items():
        argv += [f"--{k}", str(v)]
    old = sys.argv
    try:
        sys.argv = argv
        return RL.main()
    finally:
        sys.argv = old


# ============================================================ CLI
def main():
    ap = argparse.ArgumentParser(description="GLaDOS canonical pipeline: pretrain → weave → rlvr")
    sub = ap.add_subparsers(dest="cmd", required=True)

    po = sub.add_parser("pretrain", aliases=["organ"], help="STAGE 1: train + save the general organ")
    po.add_argument("--out", default="runs/general_organ_full.pt")
    po.add_argument("--steps", type=int, default=1500)
    po.add_argument("--target", type=float, default=1.5e6)
    po.add_argument("--pool", type=int, default=128)
    po.add_argument("--R", type=int, default=12)
    po.add_argument("--lr", type=float, default=3e-4)
    po.add_argument("--seed", type=int, default=0)

    pw = sub.add_parser("weave", help="STAGE 2: graft + train the woven readout (engagement mechanism)")
    pw.add_argument("--base", default="allenai/OLMo-2-0425-1B")
    pw.add_argument("--regime", default="hard", help="small | hard | large (LIVE_REGIMES)")
    pw.add_argument("--two_stream", type=int, default=1, help="1=two-stream (cells-gen) 0=single-stream")
    pw.add_argument("--steps", type=int, default=2500)
    pw.add_argument("--warm_steps", type=int, default=400)
    pw.add_argument("--out", default=None)
    pw.add_argument("--smoke", action="store_true")

    pr = sub.add_parser("rlvr", help="STAGE 3: RLVR (Dr.GRPO, exact-verifier reward)")
    pr.add_argument("--task", default="chain_sum")
    pr.add_argument("--steps", type=int, default=300)
    pr.add_argument("--out", default="runs/rlvr_pipeline")
    pr.add_argument("--smoke", action="store_true")

    a = ap.parse_args()
    if a.cmd in ("pretrain", "organ"):
        pretrain_organ(out=a.out, steps=a.steps, target=a.target, pool=a.pool, R=a.R, lr=a.lr, seed=a.seed)
    elif a.cmd == "weave":
        weave(a.base, regime=a.regime, two_stream=bool(a.two_stream), smoke=a.smoke, out=a.out,
              steps=a.steps, warm_steps=a.warm_steps)
    elif a.cmd == "rlvr":
        rlvr(task=a.task, steps=a.steps, out=a.out, smoke=a.smoke)


if __name__ == "__main__":
    main()
