"""clair/organ/reduction_smoke.py — the SMOKE: does the woven solve-by-reduction + generalize?

Trains the REAL woven path (bank_woven.BankWoven: α-compile → composer over CSPState → γ readout) on
the REDUCTION CURRICULUM (clair.datagen.reductions: SAT-family sources reduced X→CSP, solved by the
certified narrowers, decoded back), then measures the SOLVE-VIA-REDUCTION rate on the two OOD splits:

  route_required          a source family (XOR-SAT) with NO direct faculty in train — solvable ONLY by
                          reducing to CSP (a faculty we have). Tests cross-faculty routing.
  heldout_reduction_path  a TRAINED family rendered with an UNSEEN representation. Tests whether the
                          routing/readout generalizes to an encoding of Y never seen.

This is the smoke verdict (adequate length, honest if inconclusive): the WOVEN-true accuracy is the
solve-via-reduction rate; the base/zero controls bound the lift the injected (reduced+solved) lattice
buys. Run on the GPU box.

  python -m clair.organ.reduction_smoke --base allenai/OLMo-2-0425-1B --steps 400 --warm 150
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--warm", type=int, default=150)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--per_cell", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/reduction_smoke.json")
    a = ap.parse_args()

    from .. import run_glados_staged as G
    from . import bank_woven as BW
    from . import train as T
    from ..datagen import reductions as RC
    from transformers import AutoTokenizer, AutoConfig

    dev = G.device()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    print(f"==== REDUCTION SMOKE  base={a.base}  dev={dev} ====", flush=True)

    # [1] build + exact-verify the reduction curriculum (end-to-end, dedup on canonical form)
    splits, crep = RC.build_curriculum(seed=a.seed, per_cell=a.per_cell)
    print(json.dumps({k: v for k, v in crep.items() if k != "train_representations"}, indent=2),
          flush=True)
    # split train into train/eval-in-dist
    rng = np.random.default_rng(a.seed)
    tr = splits["train"]; rng.shuffle(tr)
    cut = int(0.85 * len(tr))
    train_recs, eval_id = tr[:cut], tr[cut:]
    route_req, heldout = splits["route_required"], splits["heldout_reduction_path"]

    # [2] weave the bank-composer woven on the reduction curriculum
    tok = AutoTokenizer.from_pretrained(a.base)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(a.base)
    D = cfg.hidden_size
    nL = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "num_layers", None)

    args = T._weave_args(a.base, steps=a.steps, warm_steps=a.warm, bs=a.bs, seed=a.seed,
                         per_rung_train=a.per_cell, per_rung_eval=40)
    args.inject_layer = min(args.inject_layer, nL - 1)
    args.mid_layer = min(args.mid_layer, args.inject_layer - 1)

    import os
    core = "runs/general_organ_full.pt"
    composer = BW.BankComposerOrgan(dev="cpu", core_ckpt=core, use_core=os.path.exists(core))
    print(f"  composer faculties = {composer.faculties()}", flush=True)

    t0 = time.time()
    model = BW.train_bank_woven((a.base, D, nL), tok, dev, args, train_recs, eval_id,
                                two_stream=True, composer=composer)
    print(f"  [weave] done in {time.time()-t0:.0f}s", flush=True)

    # [3] solve-via-reduction rate on each split (WOVEN-true vs base/zero controls)
    def score(recs, tag):
        if not recs:
            return None
        true = BW.score_bank_woven(model, recs, tok, dev, control="true", bs=a.bs, two_stream=True)
        zero = BW.score_bank_woven(model, recs, tok, dev, control="zero", bs=a.bs, two_stream=True)
        base = BW.score_bank_woven(model, recs, tok, dev, control="zero", use_base=True,
                                   fewshot=G.FEWSHOT, bs=a.bs, two_stream=True)
        oracle = BW.score_bank_woven(model, recs, tok, dev, control="true", override_oracle=True,
                                     bs=a.bs, two_stream=True)
        r = {"n": true["n"], "woven_true": true["overall"], "det_acc": true["det_acc"],
             "zero_noinject": zero["overall"], "base_lora_off": base["overall"],
             "oracle_override": oracle["overall"], "lift_vs_zero": true["overall"] - zero["overall"]}
        print(f"  [{tag:24s}] n={r['n']:4d}  WOVEN-true {r['woven_true']*100:5.1f}%  "
              f"zero {r['zero_noinject']*100:5.1f}%  base {r['base_lora_off']*100:5.1f}%  "
              f"oracle {r['oracle_override']*100:5.1f}%  lift {r['lift_vs_zero']*100:+.1f}pts", flush=True)
        return r

    print("\n== SOLVE-VIA-REDUCTION RATE ==", flush=True)
    out = {"curriculum": crep, "splits": {}}
    out["splits"]["in_dist_eval"] = score(eval_id, "in_dist (trained reps)")
    out["splits"]["route_required"] = score(route_req, "route_required (xorsat)")
    out["splits"]["heldout_reduction_path"] = score(heldout, "heldout_reduction_path")

    import os as _os
    _os.makedirs(_os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nwrote {a.out}", flush=True)
    print("==== REDUCTION SMOKE DONE ====", flush=True)


if __name__ == "__main__":
    main()
