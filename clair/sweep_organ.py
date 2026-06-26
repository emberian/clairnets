"""clair/sweep_organ.py — ORGAN-RECIPE sweep: CAPACITY x DEPTH x COMPOSITION-MIXUP.

Picks the config for the upcoming organ PRETRAINING run. Extends the original capacity x depth sizing
study (organ_excellence.md §2: R=8 floor, wider amortizes depth) with a THIRD axis — the fraction of
COMPOSED (multi-skill) vs single-skill problems in TRAINING — and evaluates every cell on THREE
graded splits so compositional generalization is measured directly (arXiv 2507.07207):

  in_dist   single-skill CSPs (n 4-6)              the trained distribution
  ood_n     single-skill CSPs (n 7-8)              length generalization (within the N_MAX=8 budget)
  ood_comp  HELD-OUT skill PAIRINGS                compositional generalization (never co-trained)

The organ is the general factor-graph narrower (proposer.FactorGraphProposer), trained on-policy by
dominate-dedP (run_general / datagen.fast overlapped path). "Multi-skill composition" is a CSP that
combines TWO constraint families over a shared variable set (datagen.compose_csp) — the
CSP-expressible, exact-dedP-gradeable analog of the broad corpus's domain chains. Per cell we report
narrowing-RECALL (completeness vs exact dedP) and FALSE-ELIM (soundness) on each split.

  python -m clair.sweep_organ --mode smoke
  python -m clair.sweep_organ --mode sweep --steps 700 --neval 120 --fast \
         --workers 0 --cache runs/dedp_cache.sqlite --out runs/organ_recipe_sweep.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from . import run_general as RG
from .proposer import FactorGraphProposer, size_for
from .datagen import compose_csp as CC
from .datagen import fast as FAST

N_MAX, D_MAX, M_MAX, A_MAX = RG.N_MAX, RG.D_MAX, RG.M_MAX, RG.A_MAX
SPLITS = ("in_dist", "ood_n", "ood_comp")


def train_cell(dev, target, R, steps, mixup, pool, lr, theta, seed, gen):
    """Train one organ cell on a SINGLE+COMPOSED mix (composition fraction = mixup) via the
    overlapped/parallel data path. Composed instances are drawn from TRAIN_PAIRS only; the held-out
    OOD_PAIRS are reserved for eval."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d, npar = size_for("full", N_MAX, D_MAX, M_MAX, A_MAX, target, R=R, ds=min(4, R))
    m = FactorGraphProposer("full", N_MAX, D_MAX, M_MAX, A_MAX, d=d, R=R, ds=min(4, R)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95))
    items, specs = CC.make_corpus(rng, pool, mixup)
    label = f"c{target:.0e}R{R}m{int(mixup*100):02d}"
    print(f"  [{label}] d={d} params={npar:,} pool={pool} R={R} steps={steps} mixup={mixup}", flush=True)
    log = FAST.overlap_loop(m, opt, dev, steps, rng, items, specs, CC.sample_for_spec,
                            theta=theta, gen=gen, label=label)
    return m, d, npar, log


def eval_cell(m, eval_sets, dev, theta, rmax):
    out = {}
    for split, csps in eval_sets.items():
        e = RG.evaluate_rung(m, csps, dev, theta, rmax)
        out[split] = {k: e[k] for k in ("n_solvable", "narrowing_recall", "match_dedP",
                                        "solved", "uniq_rate", "false_elim", "false_elim_count")}
    return out


def run_one(dev, eval_sets, target, R, mixup, args, gen):
    t0 = time.time()
    m, d, npar, log = train_cell(dev, target, R, args.steps, mixup, args.pool,
                                 args.lr, args.theta, args.seed, gen)
    ev = eval_cell(m, eval_sets, dev, args.theta, args.rmax)
    cell = {"target": target, "R": R, "mixup": mixup, "d_model": d, "params": npar, "eval": ev}
    secs = time.time() - t0
    rr = {s: ev[s]["narrowing_recall"] for s in SPLITS}
    fe = {s: ev[s]["false_elim"] for s in SPLITS}
    print(f"  -> [{target:.0e} R{R} mix{int(mixup*100)}] d={d} p={npar:,}  "
          f"recall in/oodN/comp {rr['in_dist']*100:.1f}/{rr['ood_n']*100:.1f}/{rr['ood_comp']*100:.1f}%  "
          f"FE {fe['in_dist']:.4f}/{fe['ood_n']:.4f}/{fe['ood_comp']:.4f}  ({secs:.0f}s)", flush=True)
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cell


# --------------------------------------------------------------------------- the study design
def study_cells(args):
    """The cells: a CAPACITY x COMPOSITION grid (R=8) + a DEPTH sweep (cap-anchor, mix-anchor) +
    a big-capacity point. Deduped by (target, R, mixup)."""
    caps_grid = [float(x) for x in args.caps.split(",")]          # 1e5,4e5,1.5e6
    mixes = [float(x) for x in args.mixes.split(",")]             # 0,0.15,0.35
    rounds = [int(x) for x in args.rounds.split(",")]            # 4,8,12,16
    cap_anchor = float(args.cap_anchor)                          # 4e5
    mix_anchor = float(args.mix_anchor)                         # 0.15
    big_cap = float(args.big_cap)                               # 6e6
    cells = []
    # 1. capacity x composition grid at R=8 (the interaction + both 1D curves through it)
    for tg in caps_grid:
        for mx in mixes:
            cells.append((tg, 8, mx))
    # 2. depth sweep at the capacity/mix anchor
    for R in rounds:
        cells.append((cap_anchor, R, mix_anchor))
    # 3. big-capacity point on the scaling curve (mix/R anchor)
    cells.append((big_cap, 8, mix_anchor))
    # dedup, stable order
    seen, uniq = set(), []
    for c in cells:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return uniq


def _fmt_pct(x):
    return f"{x*100:5.1f}"


def print_tables(out):
    cells = {(c["target"], c["R"], c["mixup"]): c for c in out["cells"]}
    A = out["args"]
    caps = [float(x) for x in A["caps"].split(",")]
    mixes = [float(x) for x in A["mixes"].split(",")]
    rounds = [int(x) for x in A["rounds"].split(",")]
    cap_a, mix_a, big = float(A["cap_anchor"]), float(A["mix_anchor"]), float(A["big_cap"])

    def cell(tg, R, mx):
        return cells.get((tg, R, mx))

    print("\n" + "=" * 92, flush=True)
    print("ORGAN-RECIPE SWEEP RESULTS  (recall = narrowing-recall vs exact dedP; FE = false-elim soundness)",
          flush=True)
    print("=" * 92, flush=True)

    # ---- capacity scaling (mix anchor, R=8): in-dist / OOD-N / OOD-comp ----
    cap_curve = [c for c in caps if cell(c, 8, mix_a)] + ([big] if cell(big, 8, mix_a) else [])
    print(f"\n[A] CAPACITY scaling  (R=8, mix={int(mix_a*100)}%)   recall%% | FE", flush=True)
    print(f"  {'params':>10s} {'d':>4s} | {'in-dist':>14s} {'OOD-N':>14s} {'OOD-comp':>14s}", flush=True)
    for tg in cap_curve:
        c = cell(tg, 8, mix_a); e = c["eval"]
        print(f"  {c['params']:>10,d} {c['d_model']:>4d} | "
              f"{_fmt_pct(e['in_dist']['narrowing_recall'])} {e['in_dist']['false_elim']:.4f} | "
              f"{_fmt_pct(e['ood_n']['narrowing_recall'])} {e['ood_n']['false_elim']:.4f} | "
              f"{_fmt_pct(e['ood_comp']['narrowing_recall'])} {e['ood_comp']['false_elim']:.4f}",
              flush=True)

    # ---- depth sweep (cap anchor, mix anchor) ----
    print(f"\n[B] DEPTH sweep  (params~{cap_a:.0e}, mix={int(mix_a*100)}%)   recall%% | FE", flush=True)
    print(f"  {'R':>3s} | {'in-dist':>14s} {'OOD-N':>14s} {'OOD-comp':>14s}", flush=True)
    for R in rounds:
        c = cell(cap_a, R, mix_a)
        if not c:
            continue
        e = c["eval"]
        print(f"  {R:>3d} | "
              f"{_fmt_pct(e['in_dist']['narrowing_recall'])} {e['in_dist']['false_elim']:.4f} | "
              f"{_fmt_pct(e['ood_n']['narrowing_recall'])} {e['ood_n']['false_elim']:.4f} | "
              f"{_fmt_pct(e['ood_comp']['narrowing_recall'])} {e['ood_comp']['false_elim']:.4f}",
              flush=True)

    # ---- composition-mixup, the KEY axis: OOD-comp recall/FE vs mix%, per capacity ----
    print(f"\n[C] COMPOSITION-MIXUP -> OOD-COMPOSITION generalization  (R=8)   recall%% | FE", flush=True)
    hdr = "  ".join(f"mix{int(m*100):>2d}%%" for m in mixes)
    print(f"  {'params':>10s} | OOD-comp recall across mix: {hdr}", flush=True)
    for tg in caps:
        row = []
        for mx in mixes:
            c = cell(tg, 8, mx)
            row.append(_fmt_pct(c['eval']['ood_comp']['narrowing_recall']) if c else "   - ")
        print(f"  {cells.get((tg,8,mixes[0]),{}).get('params',0):>10,d} | "
              + "  ".join(row), flush=True)
    print(f"\n      (OOD-comp FALSE-ELIM across mix)", flush=True)
    for tg in caps:
        row = []
        for mx in mixes:
            c = cell(tg, 8, mx)
            row.append(f"{c['eval']['ood_comp']['false_elim']:.4f}" if c else "  -   ")
        print(f"  {cells.get((tg,8,mixes[0]),{}).get('params',0):>10,d} | " + "  ".join(row), flush=True)

    # ---- in-dist recall across mix (does composition-mixup cost in-dist?) ----
    print(f"\n[D] IN-DIST recall across mix  (R=8) — composition tax on the trained distribution", flush=True)
    for tg in caps:
        row = []
        for mx in mixes:
            c = cell(tg, 8, mx)
            row.append(_fmt_pct(c['eval']['in_dist']['narrowing_recall']) if c else "   - ")
        print(f"  {cells.get((tg,8,mixes[0]),{}).get('params',0):>10,d} | " + "  ".join(row), flush=True)


def run_sweep(args, dev):
    gen = None
    if args.fast:
        gen = FAST.FastGen(workers=args.workers or None, cache_path=args.cache or None)
    eval_sets = CC.make_eval_sets(args.neval)
    for s in SPLITS:
        n_narrow = len(eval_sets[s])
        print(f"  eval split {s:9s}: {n_narrow} instances", flush=True)
    cells_spec = study_cells(args)
    out = {"budget": [N_MAX, D_MAX, M_MAX, A_MAX], "splits": SPLITS,
           "train_pairs": [list(p) for p in CC.TRAIN_PAIRS],
           "ood_pairs": [list(p) for p in CC.OOD_PAIRS],
           "args": vars(args), "cells": []}
    print(f"\ndevice={dev}  {len(cells_spec)} cells: {cells_spec}", flush=True)
    for (tg, R, mx) in cells_spec:
        cell = run_one(dev, eval_sets, tg, R, mx, args, gen)
        out["cells"].append(cell)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)              # checkpoint after every cell
    if gen is not None:
        gen.close()
    print_tables(out)
    print("\nwrote", args.out, flush=True)
    return out


def run_smoke(args, dev):
    print("\n===== SMOKE: one cell trains + all THREE eval splits compute =====", flush=True)
    gen = FAST.FastGen(workers=args.workers or None) if args.fast else None
    eval_sets = CC.make_eval_sets(args.neval or 50)
    for s in SPLITS:
        print(f"  eval split {s:9s}: {len(eval_sets[s])} instances", flush=True)
    cell = run_one(dev, eval_sets, target=1.5e5, R=8, mixup=0.15, args=args, gen=gen)
    if gen is not None:
        gen.close()
    print("\n  SMOKE eval (recall / FE per split):", flush=True)
    for s in SPLITS:
        e = cell["eval"][s]
        print(f"    {s:9s} n={e['n_solvable']:3d}  recall {e['narrowing_recall']*100:5.1f}%  "
              f"match-fix {e['match_dedP']*100:5.1f}%  solved {e['solved']*100:5.1f}%  "
              f"FALSE-ELIM {e['false_elim']:.4f} ({e['false_elim_count']})", flush=True)
    return {"smoke": cell}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "sweep"], default="smoke")
    ap.add_argument("--caps", default="1e5,4e5,1.5e6", help="capacity grid (param targets)")
    ap.add_argument("--mixes", default="0,0.15,0.35", help="composition-mixup fractions")
    ap.add_argument("--rounds", default="4,8,12,16", help="depth R values")
    ap.add_argument("--cap_anchor", default="4e5", help="capacity anchor for the depth sweep")
    ap.add_argument("--mix_anchor", default="0.15", help="mixup anchor for depth + capacity curves")
    ap.add_argument("--big_cap", default="6e6", help="large-capacity scaling point")
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--neval", type=int, default=120)
    ap.add_argument("--pool", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--rmax", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true",
                    help="parallel (all-core) dedP targets + GPU-overlapped data path")
    ap.add_argument("--workers", type=int, default=0, help="pool workers for --fast (0=os.cpu_count)")
    ap.add_argument("--cache", default=None, help="on-disk dedP target cache (sqlite path)")
    ap.add_argument("--out", default="runs/organ_recipe_sweep.json")
    args = ap.parse_args()
    dev = RG.device()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    print(f"device={dev}  budget N={N_MAX} D={D_MAX} M={M_MAX} A={A_MAX}", flush=True)
    if args.mode == "smoke":
        run_smoke(args, dev)
    else:
        run_sweep(args, dev)


if __name__ == "__main__":
    main()
