"""clair/run_blade.py — BLADE vs TABLE factor-deductor comparison (the expressiveness experiment).

Drives clair.blade_deductor.BladeFactorDeductor in both modes through the SAME factor-graph
narrowing harness (reusing clair.run_general's featurize / targets / loss / monotone-meet eval),
iso-param. The question: does the grade-k geometric-product / blade factor->variable message give
the narrowing organ higher COMPLETENESS on affine/3-way (arity-3) constraints, and does it
GENERALISE to UNSEEN arity-3 relations where the relation-table version collapses?

Rungs (each its own EXACT clair.csp ground truth):
    coloring equality ordering alldiff smt   — arity 1/2 (the 'table-friendly' rungs)
    arithmetic                               — arity 1/3 modular sum  (affine)
    xor                                      — arity 1/3 boolean parity (affine, d=2)   [NEW]

EXPERIMENTS (each run for BOTH msg=blade and msg=table, iso-param, same seeds/eval sets):
  A. multitask     train ALL rungs, eval per rung           -> per-rung completeness table
  B. zero-shot-3   train arity<=2 rungs ONLY, eval {arithmetic, xor} ZERO-SHOT
                   (no arity-3 factor EVER seen — the grade-3 path must generalise unprompted)
  C. cross-affine  train arity<=2 + xor (HOLD OUT arithmetic), eval arithmetic ZERO-SHOT
                   (one arity-3 affine relation seen, a DIFFERENT one tested — does grade-3 transfer?)

Primary metric: narrowing-recall vs exact dedP (completeness; sound => eliminations are a subset).
Soundness tracked as false-elim (must stay ~0). Run:
    python -m clair.run_blade --mode smoke
    python -m clair.run_blade --mode full --steps 1200 --out runs/blade.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .blade_deductor import BladeFactorDeductor, size_for
from . import csp as C
from . import curriculum as CU
from . import run_general as RG

N_MAX, D_MAX, M_MAX, A_MAX = RG.N_MAX, RG.D_MAX, RG.M_MAX, RG.A_MAX
ENUM_CAP = RG.ENUM_CAP

ALL_RUNGS = ["coloring", "equality", "ordering", "arithmetic", "xor", "alldiff", "smt"]
ARITY3 = ["arithmetic", "xor"]                                    # the affine / 3-way rungs
ARITY2 = ["coloring", "equality", "ordering", "alldiff", "smt"]   # arity<=2 (train-only for zero-shot)


# ===================================================================== sampling (xor-aware)
def _ok(csp):
    return (csp.n <= N_MAX and csp.d <= D_MAX and len(csp.cons) <= M_MAX
            and csp.d ** csp.n <= ENUM_CAP and len(C.solutions(csp)) > 0
            and all(len(sc) <= A_MAX for sc, _ in csp.cons))


def sample_csp(rng, rung):
    """Solvable, within-budget CSP for a rung. xor is handled here; the rest delegate to
    run_general.sample_rung_csp (unchanged)."""
    if rung == "xor":
        for _ in range(200):
            n, d, _, facts, s = CU.gen_xor(rng)
            csp = CU.build_csp(n, d, facts)
            if _ok(csp):
                return csp
        return CU.build_csp(3, 2, [("pin", 0, 0)])   # trivial solvable fallback (rare)
    return RG.sample_rung_csp(rng, rung)


def sample_corpus(rng, n, rungs):
    return [(rungs[i % len(rungs)], sample_csp(rng, rungs[i % len(rungs)])) for i in range(n)]


def make_eval_sets(rungs, neval, seed=12345):
    rng = np.random.default_rng(seed)
    return {rg: [sample_csp(rng, rg) for _ in range(neval)] for rg in rungs}


# ===================================================================== sizing (iso-param, blade<=table)
def matched_sizes(target, R, ds):
    """Size the TABLE arm to the largest d under `target`, then size the BLADE arm to the table arm's
    param COUNT — so blade never carries MORE params than table (a conservative bias test).
    Returns {'table': (d, params), 'blade': (d, params)}."""
    dt, pt = size_for("table", N_MAX, D_MAX, M_MAX, A_MAX, target, R=R, ds=ds)
    db, pb = size_for("blade", N_MAX, D_MAX, M_MAX, A_MAX, pt, R=R, ds=ds)
    return {"table": (dt, pt), "blade": (db, pb)}


# ===================================================================== training (mirrors RG.train, blade model)
def train(msg, rungs, dev, d, steps, pool=128, R=8, lr=3e-4, theta=0.5, seed=0,
          ex=RG.SHARED, log=None, gen=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = BladeFactorDeductor(msg, N_MAX, D_MAX, M_MAX, A_MAX, d=d, R=R, ds=min(4, R)).to(dev)
    npar = m.n_params()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95))
    log = log if log is not None else []
    tagged = sample_corpus(rng, pool, rungs)
    items = [(c, c.full()) for _, c in tagged]
    rtags = [rg for rg, _ in tagged]
    if gen is not None:                                          # parallel + GPU-overlapped data path
        from .datagen import fast as _fast
        print(f"  [{msg:5s}] [fast] train rungs={rungs} d={d} params={npar:,} pool={pool} R={R} "
              f"steps={steps} workers={gen.workers}", flush=True)
        log = _fast.overlap_loop(m, opt, dev, steps, rng, items, rtags, sample_csp,
                                 theta=theta, gen=gen, ex=ex, log=log, label=msg)
        return m, d, npar, log
    print(f"  [{msg:5s}] train rungs={rungs} d={d} params={npar:,} pool={pool} R={R} steps={steps}",
          flush=True)
    fe_k = fe_n = 0
    t0 = time.time()
    for s in range(1, steps + 1):
        feat = RG.featurize(items, dev)
        vm = feat["var_mask"]
        tgt, conflict = RG.build_targets(items, ex, dev)
        b, cls, sup = RG.fwd(m, feat, vm)
        loss = RG.loss_fn(sup, vm, feat["var_valid"], tgt, conflict)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        with torch.no_grad():
            new_vm = RG.meet(vm, b, theta)
            k, n = RG.false_elim(vm, new_vm, feat["var_valid"], tgt); fe_k += k; fe_n += n
            nvc = new_vm.cpu().numpy()
            nxt = []
            for bi, (csp, dom) in enumerate(items):
                ndom = RG.dom_from_mask(nvc[bi], csp)
                if C.status(ndom) in ("solved", "conflict") or ndom == dom:
                    rg = rtags[bi]
                    nc = sample_csp(rng, rg)
                    nxt.append((nc, nc.full()))
                else:
                    nxt.append((csp, ndom))
            items = nxt
        if s % max(1, steps // 12) == 0:
            fer = fe_k / max(1, fe_n)
            log.append({"step": s, "loss": float(loss.detach()), "false_elim": fer})
            print(f"    [{msg:5s}] step {s:5d}  loss {float(loss.detach()):.3f}  false_elim {fer:.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            fe_k = fe_n = 0
    return m, d, npar, log


# ===================================================================== experiments
def _eval(m, eval_sets, dev, args):
    return {rg: RG.evaluate_rung(m, csps, dev, args.theta, args.rmax) for rg, csps in eval_sets.items()}


def _ptable(title, blade, table, rungs):
    print(f"\n  {title}", flush=True)
    print(f"  {'rung':11s} {'recall(dedP)':>22s} {'recall(factor)':>22s} {'false-elim':>16s}", flush=True)
    print(f"  {'':11s} {'blade':>10s} {'table':>10s} {'blade':>10s} {'table':>10s} "
          f"{'blade':>7s} {'table':>7s}", flush=True)
    for rg in rungs:
        b, t = blade[rg], table[rg]
        print(f"  {rg:11s} {b['narrowing_recall']*100:9.1f}% {t['narrowing_recall']*100:9.1f}% "
              f"{b['factor_recall']*100:9.1f}% {t['factor_recall']*100:9.1f}% "
              f"{b['false_elim']:7.4f} {t['false_elim']:7.4f}", flush=True)


def _maybe_gen(args):
    """One shared FastGen reused across every train() call when --fast (parallel + overlapped data)."""
    if getattr(args, "fast", False):
        from .datagen import fast as _fast
        return _fast.FastGen(workers=getattr(args, "workers", 0) or None,
                             cache_path=getattr(args, "cache", None) or None)
    return None


def run_full(args, dev):
    out = {"budget": [N_MAX, D_MAX, M_MAX, A_MAX], "all_rungs": ALL_RUNGS, "args": vars(args)}
    gen = _maybe_gen(args)
    eval_all = make_eval_sets(ALL_RUNGS, args.neval)
    eval_3 = {rg: eval_all[rg] for rg in ARITY3}
    modes = ("blade", "table")
    sz = matched_sizes(args.target, args.R, min(4, args.R))
    out["sizes"] = {k: {"d_model": v[0], "params": v[1]} for k, v in sz.items()}
    print(f"iso-param sizes: table d={sz['table'][0]} p={sz['table'][1]:,}  >=  "
          f"blade d={sz['blade'][0]} p={sz['blade'][1]:,} (blade no param advantage)", flush=True)

    # ---- A. MULTITASK: train all rungs, eval per rung ----
    print("\n========== A. MULTITASK (train ALL rungs, eval per rung) ==========", flush=True)
    A = {}
    for msg in modes:
        m, d, npar, log = train(msg, ALL_RUNGS, dev, sz[msg][0], args.steps,
                                 pool=args.pool, R=args.R, lr=args.lr, theta=args.theta, seed=args.seed, gen=gen)
        A[msg] = {"d_model": d, "params": npar, "log": log, "eval": _eval(m, eval_all, dev, args)}
    _ptable("A. MULTITASK per-rung completeness", A["blade"]["eval"], A["table"]["eval"], ALL_RUNGS)
    out["multitask"] = A

    # ---- B. ZERO-SHOT-3: train arity<=2 ONLY, eval {arithmetic, xor} zero-shot ----
    print("\n========== B. ZERO-SHOT to UNSEEN ARITY-3 (train arity<=2 ONLY) ==========", flush=True)
    Bx = {}
    for msg in modes:
        m, d, npar, log = train(msg, ARITY2, dev, sz[msg][0], args.steps,
                                 pool=args.pool, R=args.R, lr=args.lr, theta=args.theta, seed=args.seed, gen=gen)
        Bx[msg] = {"d_model": d, "params": npar, "log": log, "eval": _eval(m, eval_3, dev, args)}
    _ptable("B. ZERO-SHOT arity-3 (NEVER saw an arity-3 factor in training)",
            Bx["blade"]["eval"], Bx["table"]["eval"], ARITY3)
    out["zeroshot3"] = Bx

    # ---- C. CROSS-AFFINE: train arity<=2 + xor (HOLD OUT arithmetic), eval arithmetic ----
    print("\n========== C. CROSS-AFFINE (train arity<=2 + xor, eval arithmetic ZERO-SHOT) ==========",
          flush=True)
    Cx = {}
    train_C = ARITY2 + ["xor"]
    eval_C = {"arithmetic": eval_all["arithmetic"]}
    for msg in modes:
        m, d, npar, log = train(msg, train_C, dev, sz[msg][0], args.steps,
                                 pool=args.pool, R=args.R, lr=args.lr, theta=args.theta, seed=args.seed, gen=gen)
        Cx[msg] = {"d_model": d, "params": npar, "log": log, "eval": _eval(m, eval_C, dev, args)}
    _ptable("C. CROSS-AFFINE (saw xor=arity-3 affine; tested on arithmetic=different arity-3 affine)",
            Cx["blade"]["eval"], Cx["table"]["eval"], ["arithmetic"])
    out["crossaffine"] = Cx

    # ---- verdict scaffold (numbers; the prose verdict is in the report) ----
    print("\n========== HEADLINE (narrowing-recall vs exact dedP) ==========", flush=True)
    for rg in ARITY3:
        print(f"  multitask  {rg:11s}: blade {A['blade']['eval'][rg]['narrowing_recall']*100:5.1f}%  "
              f"table {A['table']['eval'][rg]['narrowing_recall']*100:5.1f}%", flush=True)
    for rg in ARITY3:
        print(f"  zero-shot  {rg:11s}: blade {Bx['blade']['eval'][rg]['narrowing_recall']*100:5.1f}%  "
              f"table {Bx['table']['eval'][rg]['narrowing_recall']*100:5.1f}%", flush=True)
    print(f"  cross-aff  arithmetic : blade {Cx['blade']['eval']['arithmetic']['narrowing_recall']*100:5.1f}%  "
          f"table {Cx['table']['eval']['arithmetic']['narrowing_recall']*100:5.1f}%", flush=True)
    if gen is not None:
        gen.close()
    return out


def run_smoke(args, dev):
    print("\n===== SMOKE (blade & table train; affine rungs present; false-elim drops) =====", flush=True)
    rungs = ["coloring", "arithmetic", "xor"]
    eval_sets = make_eval_sets(rungs, 60)
    sz = matched_sizes(1.2e5, args.R, min(4, args.R))
    gen = _maybe_gen(args)
    print(f"iso-param: table d={sz['table'][0]} p={sz['table'][1]:,} >= "
          f"blade d={sz['blade'][0]} p={sz['blade'][1]:,}", flush=True)
    out = {}
    for msg in ("blade", "table"):
        m, d, npar, log = train(msg, rungs, dev, sz[msg][0], steps=args.steps or 200,
                                pool=64, R=args.R, lr=args.lr, theta=args.theta, seed=0, gen=gen)
        ev = _eval(m, eval_sets, dev, args)
        out[msg] = {"d": d, "params": npar, "log": log, "eval": ev}
        print(f"  [{msg}] false_elim traj: {[round(x['false_elim'],4) for x in log]}", flush=True)
        for rg in rungs:
            print("   " + RG._row(rg, ev[rg]), flush=True)
    if gen is not None:
        gen.close()
    return {"smoke": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--target", type=float, default=2.0e5)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--pool", type=int, default=128)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--neval", type=int, default=200)
    ap.add_argument("--rmax", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true",
                    help="parallel (all-core) dedP targets + GPU-overlapped data path")
    ap.add_argument("--workers", type=int, default=0, help="pool workers for --fast (0=os.cpu_count)")
    ap.add_argument("--cache", default=None, help="on-disk dedP target cache (sqlite path), reused across runs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = RG.device()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    print(f"device={dev}  budget N={N_MAX} D={D_MAX} M={M_MAX} A={A_MAX}  "
          f"rungs={ALL_RUNGS}", flush=True)
    out = run_smoke(args, dev) if args.mode == "smoke" else run_full(args, dev)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
        print("\nwrote", args.out, flush=True)


if __name__ == "__main__":
    main()
