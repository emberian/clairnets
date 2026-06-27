"""clair/organ/run_shuffled_diag.py — driver: train a bank-woven the validated way, then run codex's
3-arm SHUFFLED diagnostic (clair.organ.shuffled_diag).

  python -m clair.organ.run_shuffled_diag --steps 1500 --warm 400 --regime hard
  python -m clair.organ.run_shuffled_diag --smoke          # tiny: engagement + struct-swap sanity only

Stages:
  1. ensure runs/general_organ_full.pt (the pretrained CoreNarrowOrgan) — the validated bank uses it as
     the gated neural faculty; if absent and --pretrain, train it on the difficulty stream.
  2. weave (organ_mode=bank, two-stream + J0 + rich-state) ~steps on the CSP curriculum; print engagement.
  3. build a fresh held-out determined-query eval pool; run the 3-arm SHUFFLED diagnostic.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--regime", default="hard")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--warm", type=int, default=400)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--per_rung_eval", type=int, default=300)
    ap.add_argument("--diag_bs", type=int, default=8)
    ap.add_argument("--pretrain", action="store_true", help="pretrain the core organ if checkpoint missing")
    ap.add_argument("--pretrain_steps", type=int, default=1500)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional: save diagnostic dict (npz-ish json) here")
    a = ap.parse_args()

    from .. import run_glados_staged as G
    from . import train as T
    from . import bank_woven as BW
    from . import shuffled_diag as SD

    dev = G.device()
    print(f"[diag] device={dev}  base={a.base}  regime={a.regime}  steps={a.steps}  smoke={a.smoke}",
          flush=True)

    # ---- stage 1: core organ checkpoint ----
    ckpt = "runs/general_organ_full.pt"
    if not os.path.exists(ckpt):
        if a.pretrain and not a.smoke:
            print(f"[diag] pretraining core organ -> {ckpt} ({a.pretrain_steps} steps)", flush=True)
            T.pretrain_organ(out=ckpt, steps=a.pretrain_steps, seed=a.seed, dev=dev)
        else:
            print(f"[diag] NOTE: {ckpt} absent — bank runs with the CERTIFIED FLOOR ONLY (Arc/Factor/"
                  f"Modular/GF2/Macro) + α; the CoreNarrowOrgan neural faculty is skipped. The crux test "
                  f"is still valid: the certified floor is exactly the structure-driven solver.", flush=True)

    # ---- stage 2: weave (the validated bank-woven recipe) ----
    hp = dict(steps=a.steps, warm_steps=a.warm, bs=a.bs, per_rung_eval=a.per_rung_eval, seed=a.seed)
    model, metrics = T.weave(a.base, regime=a.regime, two_stream=True, organ_mode="bank",
                             smoke=a.smoke, **hp)
    print("\n[diag] ENGAGEMENT CHECK (bank_controls):", flush=True)
    eng_keys = ["true", "det_acc", "zero", "shuffle", "permute", "corrupt", "oracle", "base",
                "gate", "drop", "lift", "wired", "necessary", "engages", "faculties", "n"]
    for k in eng_keys:
        if k in metrics:
            print(f"    {k:10s} = {metrics[k]}", flush=True)
    engaged = bool(metrics.get("engages"))
    if not engaged:
        print("[diag] WARNING: engagement criteria NOT met (gate/drop/lift). The diagnostic below is "
              "reported regardless, but interpret with this caveat.", flush=True)

    # ---- stage 3: fresh held-out determined eval pool + the 3-arm diagnostic ----
    rungs, split = G.LIVE_REGIMES[G._REGIME_ALIAS.get(a.regime, a.regime)]
    rng = np.random.default_rng(a.seed + 9000)
    per = 30 if a.smoke else a.per_rung_eval
    recs = G.build_live_pool(rng, rungs, split, per, det_only=True, two_stream=True)
    print(f"\n[diag] held-out determined-query eval pool: {len(recs)} recs over {rungs} ({split})",
          flush=True)

    out = SD.run_diagnostic(model, recs, model_tok(a.base), dev, bs=a.diag_bs, two_stream=True,
                            seed=a.seed, verbose=True)
    if a.out:
        import json
        dump = {k: v for k, v in out.items() if not k.startswith("_")}
        with open(a.out, "w") as f:
            json.dump(dump, f, indent=2, default=str)
        print(f"[diag] wrote {a.out}", flush=True)


_TOK = {}


def model_tok(base):
    if base not in _TOK:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(base)
        tok.padding_side = "right"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _TOK[base] = tok
    return _TOK[base]


if __name__ == "__main__":
    main()
