"""clair/sweep_organ.py — SIZING x DEPTH study for the deduction organ (organ-excellence task 2).

Sweeps capacity (param target -> d_model) x depth (R message-passing rounds) for the general
factor-graph organ (proposer.FactorGraphProposer, the 'full' geometric arm), trained multitask on
the whole curriculum ladder, eval per rung. Produces scaling curves for narrowing-recall (vs exact
dedP = completeness) and soundness (false-elim), and the depth-for-width law: how many rounds R the
hard (affine/alldiff) rungs need at each width.

  python -m clair.sweep_organ --out runs/sweep.json --steps 600 --neval 120
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np
import torch

from . import run_general as RG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="5e4,1e5,2e5,4e5")
    ap.add_argument("--rounds", default="2,4,8,16")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--neval", type=int, default=120)
    ap.add_argument("--pool", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--rmax", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true",
                    help="parallel (all-core) dedP targets + GPU-overlapped data path")
    ap.add_argument("--workers", type=int, default=0, help="pool workers for --fast (0=os.cpu_count)")
    ap.add_argument("--cache", default=None, help="on-disk dedP target cache (sqlite path), reused across runs")
    ap.add_argument("--out", default="runs/sweep.json")
    args = ap.parse_args()
    dev = RG.device()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    gen = None
    if args.fast:
        from .datagen import fast as _fast
        gen = _fast.FastGen(workers=args.workers or None,    # one pool reused across all sweep cells
                            cache_path=args.cache or None)
    targets = [float(x) for x in args.targets.split(",")]
    rounds = [int(x) for x in args.rounds.split(",")]
    rungs = RG.RUNGS
    eval_sets = RG.make_eval_sets(rungs, args.neval)
    out = {"budget": [RG.N_MAX, RG.D_MAX, RG.M_MAX, RG.A_MAX], "rungs": rungs,
           "targets": targets, "rounds": rounds, "args": vars(args), "cells": []}
    print(f"device={dev}  sweep targets={targets} rounds={rounds} steps={args.steps}", flush=True)
    for tg in targets:
        for R in rounds:
            t0 = time.time()
            m, d, npar, log = RG.train(rungs, dev, tg, args.steps, pool=args.pool, R=R,
                                       lr=args.lr, theta=args.theta, seed=args.seed, gen=gen)
            ev = RG.eval_all(m, eval_sets, dev, args.theta, args.rmax)
            # aggregate (micro over rungs already per-rung; report per-rung + a mean)
            mean_rec = float(np.mean([ev[rg]["narrowing_recall"] for rg in rungs]))
            mean_fe = float(np.mean([ev[rg]["false_elim"] for rg in rungs]))
            cell = {"target": tg, "R": R, "d_model": d, "params": npar,
                    "mean_recall": mean_rec, "mean_false_elim": mean_fe,
                    "per_rung": {rg: {k: ev[rg][k] for k in
                                ("narrowing_recall", "factor_recall", "match_dedP",
                                 "solved", "uniq_rate", "false_elim", "false_elim_count")}
                                 for rg in rungs}}
            out["cells"].append(cell)
            print(f"\n[target {tg:.0e} R {R:2d}] d={d} params={npar:,}  mean-recall {mean_rec*100:.1f}%  "
                  f"mean-FE {mean_fe:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            for rg in rungs:
                e = ev[rg]
                print(f"    {rg:11s} recall {e['narrowing_recall']*100:5.1f}%  match-fix {e['match_dedP']*100:5.1f}%"
                      f"  FE {e['false_elim']:.4f}", flush=True)
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            json.dump(out, open(args.out, "w"), indent=1)        # checkpoint after every cell
    if gen is not None:
        gen.close()
    print("\nwrote", args.out, flush=True)


if __name__ == "__main__":
    main()
