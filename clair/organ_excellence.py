"""clair/organ_excellence.py — the BEST organ (task 4) + zero-shot reasoning-gym organ eval (task 1).

Integrates the research lessons into ONE config and trains it to excellence:
  * general FACTOR representation (allowed-tuple relations, any arity) — proposer/blade skeleton
  * grade/blade ARITY prior (blade_deductor.BladeFactorDeductor msg=blade): grade-k blade == arity-k
  * sufficient DEPTH (R message rounds) — the affine wall needs composition over rounds
  * set-valued (monotone-meet) readout dominating the exact dedP -> soundness is free

Trains on the full curriculum ladder (incl xor=affine), SAVES the checkpoint (for the assembly to
bootstrap+freeze), evals per rung, and runs ZERO-SHOT on the reasoning-gym CSP adapters
(n_queens / knights_knaves / graph_color that fit the trained budget) — the breadth/generality test.

  python -m clair.organ_excellence --train --R 12 --target 4e5 --steps 1200 --ckpt runs/best_organ.pt
  python -m clair.organ_excellence --eval-rg --ckpt runs/best_organ.pt
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np
import torch

from . import run_general as RG
from . import run_blade as RB
from .blade_deductor import BladeFactorDeductor, size_for
from . import rg_csp as RGC

ALL_RUNGS = RB.ALL_RUNGS                              # incl xor (affine d=2)

# reasoning-gym tasks that fit the trained budget (N<=8,D<=8,M<=28,A<=3), with configs
RG_EVAL = [
    ("n_queens",      RGC.n_queens_to_csp,      {"n": 5, "min_remove": 1, "max_remove": 4}),
    ("n_queens",      RGC.n_queens_to_csp,      {"n": 4, "min_remove": 1, "max_remove": 3}),
    ("knights_knaves", RGC.knights_knaves_to_csp, {"n_people": 2}),
    ("knights_knaves", RGC.knights_knaves_to_csp, {"n_people": 3}),
    ("graph_color",   RGC.graph_color_to_csp,   {"min_num_vertices": 8, "max_num_vertices": 8, "num_colors": 3}),
]


def train_best(args, dev):
    msg = args.msg
    d, npar = size_for(msg, RG.N_MAX, RG.D_MAX, RG.M_MAX, RG.A_MAX, args.target, R=args.R, ds=min(4, args.R))
    print(f"BEST organ: msg={msg} d={d} params={npar:,} R={args.R} steps={args.steps} rungs={ALL_RUNGS}", flush=True)
    m, d, npar, log = RB.train(msg, ALL_RUNGS, dev, d, args.steps, pool=args.pool, R=args.R,
                               lr=args.lr, theta=args.theta, seed=args.seed)
    eval_sets = RB.make_eval_sets(ALL_RUNGS, args.neval)
    ev = {rg: RG.evaluate_rung(m, csps, dev, args.theta, args.rmax) for rg, csps in eval_sets.items()}
    print("\n=== BEST organ per-rung (curriculum) ===", flush=True)
    for rg in ALL_RUNGS:
        print(RG._row(rg, ev[rg]), flush=True)
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    torch.save({"state_dict": m.state_dict(),
                "config": {"msg": msg, "n_max": RG.N_MAX, "d_max": RG.D_MAX, "m_max": RG.M_MAX,
                           "a_max": RG.A_MAX, "d": d, "R": args.R, "ds": min(4, args.R)},
                "params": npar, "rungs": ALL_RUNGS, "eval": ev, "log": log,
                "args": vars(args)}, args.ckpt)
    print(f"\nsaved checkpoint -> {args.ckpt}  ({npar:,} params)", flush=True)
    return m, ev


def load_organ(ckpt, dev):
    blob = torch.load(ckpt, map_location=dev, weights_only=False)
    c = blob["config"]
    m = BladeFactorDeductor(c["msg"], c["n_max"], c["d_max"], c["m_max"], c["a_max"],
                            d=c["d"], R=c["R"], ds=c["ds"]).to(dev)
    m.load_state_dict(blob["state_dict"]); m.eval()
    return m, blob


def eval_rg(args, dev, m=None):
    if m is None:
        m, _ = load_organ(args.ckpt, dev)
    budget = (RG.N_MAX, RG.D_MAX, RG.M_MAX, RG.A_MAX)
    rows = {}
    print("\n=== ZERO-SHOT reasoning-gym organ eval (trained ONLY on curriculum) ===", flush=True)
    print(f"  {'task':22s} {'n':>4s} {'recall(dedP)':>13s} {'recall(fac)':>12s} {'match-fix':>10s} "
          f"{'solved':>8s} {'FALSE-ELIM':>12s}", flush=True)
    for name, adp, cfg in RG_EVAL:
        csps = RGC.rg_eval_csps(name, adp, cfg, args.neval, budget=budget,
                                require_narrowable=(name != "graph_color"))
        if not csps:
            print(f"  {name+str(cfg.get('n',cfg.get('n_people','')))[:18]:22s}  (no budget-fitting narrowable instances)", flush=True)
            continue
        e = RG.evaluate_rung(m, csps, dev, args.theta, args.rmax)
        tag = f"{name}:{cfg.get('n', cfg.get('n_people', cfg.get('max_num_vertices','')))}"
        rows[tag] = e
        print(f"  {tag:22s} {e['n_solvable']:4d} {e['narrowing_recall']*100:11.1f}% "
              f"{e['factor_recall']*100:10.1f}% {e['match_dedP']*100:8.1f}% {e['solved']*100:6.1f}% "
              f"{e['false_elim']:9.4f}({e['false_elim_count']})", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval-rg", dest="eval_rg", action="store_true")
    ap.add_argument("--msg", default="blade", choices=["blade", "table"])
    ap.add_argument("--target", type=float, default=4e5)
    ap.add_argument("--R", type=int, default=12)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--pool", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--neval", type=int, default=150)
    ap.add_argument("--rmax", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="runs/best_organ.pt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = RG.device()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    out = {}
    m = None
    if args.train:
        m, ev = train_best(args, dev)
        out["curriculum_eval"] = ev
    if args.eval_rg:
        out["rg_eval"] = eval_rg(args, dev, m=m)
    if args.out and out:
        json.dump(out, open(args.out, "w"), indent=1, default=str)
        print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
