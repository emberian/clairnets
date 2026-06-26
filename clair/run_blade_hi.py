"""clair/run_blade_hi.py — HIGHER-ARITY (grade 4/5) & UN-TRUNCATED blade comparison.

Boolean (d=2) AFFINE/parity rungs of arity 3, 4, 5 (clair.curriculum.gen_parity_k), narrowed by
clair.blade_hi.BladeHiDeductor in three modes through the SAME factor-graph harness, iso-param:

  table : relation-table MLP            (baseline; an arity-k constraint = a 2^k table)
  minor : fixed-K grade-a wedge minors  (the DIRECT grade-4/5 extension — hardcode grade, cap at K)
  scan  : exterior-algebra FOLD         (ONE wedge primitive folded over a factor's cells; grade =
                                         #cells folded — the un-truncated abstraction)

EXPERIMENTS (each for all 3 modes, iso-param, same seeds/eval sets):
  A. multitask    train {par3,par4,par5}, eval each       -> does higher grade help per arity?
  B. zero-shot    train {par3} ONLY, eval {par4, par5}    -> saw ONLY ternary; must do 4-/5-ary
                  affine ZERO-SHOT. The truncation test: minor's grade-4/5 paths were NEVER
                  trained but exist; scan's fold extends to any arity by construction; table must
                  learn a 2^4 / 2^5 relation cold.

Budget: N=9 cells, D=2 (boolean), M=24 factors, A=5 (max arity). Primary metric = narrowing-recall
vs exact dedP. Run: python -m clair.run_blade_hi --mode full --steps 1500 --out runs/blade_hi.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .blade_hi import BladeHiDeductor, size_for
from . import csp as C
from . import curriculum as CU
from . import run_general as RG

N_MAX, D_MAX, M_MAX, A_MAX = 9, 2, 24, 5
ENUM_CAP = 1024
RUNGS = ["par3", "par4", "par5"]
_KOF = {"par3": (3, 3), "par4": (4, 4), "par5": (5, 5)}


# ============================================================ sampling (boolean parity, fixed arity)
def _ok(csp):
    return (csp.n <= N_MAX and csp.d <= D_MAX and len(csp.cons) <= M_MAX
            and csp.d ** csp.n <= ENUM_CAP and len(C.solutions(csp)) > 0
            and all(len(sc) <= A_MAX for sc, _ in csp.cons))


def sample_csp(rng, rung):
    kmin, kmax = _KOF[rung]
    n_lo = max(kmin + 1, 6)
    for _ in range(300):
        n, d, _, facts, s = CU.gen_parity_k(rng, n_lo=n_lo, n_hi=max(n_lo + 2, 8), kmin=kmin, kmax=kmax)
        if not any(len(f[1]) >= kmin for f in facts if f[0] == "par"):
            continue                                          # ensure at least one arity-k factor
        csp = CU.build_csp(n, d, facts)
        if _ok(csp):
            return csp
    return CU.build_csp(kmin + 1, 2, [("pin", 0, 0)])         # trivial fallback (rare)


def sample_corpus(rng, n, rungs):
    return [(rungs[i % len(rungs)], sample_csp(rng, rungs[i % len(rungs)])) for i in range(n)]


def make_eval_sets(rungs, neval, seed=777):
    rng = np.random.default_rng(seed)
    return {rg: [sample_csp(rng, rg) for _ in range(neval)] for rg in rungs}


# ============================================================ featurize / targets (boolean budget)
def featurize(items, dev):
    B = len(items)
    rel_dim = D_MAX ** A_MAX
    var_mask = np.zeros((B, N_MAX, D_MAX), np.float32)
    given = np.zeros((B, N_MAX), np.float32)
    var_valid = np.zeros((B, N_MAX), np.float32)
    fac_rel = np.zeros((B, M_MAX, rel_dim), np.float32)
    fac_arity = np.zeros((B, M_MAX, A_MAX), np.float32)
    fac_valid = np.zeros((B, M_MAX), np.float32)
    edge_var = np.full((B, M_MAX, A_MAX), N_MAX, np.int64)
    edge_valid = np.zeros((B, M_MAX, A_MAX), np.float32)
    for bi, (csp, dom) in enumerate(items):
        for i in range(csp.n):
            var_valid[bi, i] = 1.0
            for v in dom[i]:
                var_mask[bi, i, v] = 1.0
            if len(dom[i]) == 1:
                given[bi, i] = 1.0
        for fi, (sc, al) in enumerate(csp.cons):
            if fi >= M_MAX:
                break
            fac_valid[bi, fi] = 1.0
            fac_arity[bi, fi, len(sc) - 1] = 1.0
            fac_rel[bi, fi] = RG.relation_table(sc, al, D_MAX, A_MAX)
            for p, cell in enumerate(sc):
                edge_var[bi, fi, p] = cell
                edge_valid[bi, fi, p] = 1.0
    t = lambda a: torch.as_tensor(a, device=dev)
    return dict(var_mask=t(var_mask), given=t(given), var_valid=t(var_valid),
                fac_rel=t(fac_rel), fac_arity=t(fac_arity), fac_valid=t(fac_valid),
                edge_var=t(edge_var), edge_valid=t(edge_valid))


def build_targets(items, ex, dev):
    B = len(items)
    tgt = np.zeros((B, N_MAX, D_MAX), np.float32)
    conflict = np.zeros((B,), np.float32)
    for bi, (csp, dom) in enumerate(items):
        ded = ex.dedP(csp, dom)
        if all(len(c) == 0 for c in ded):
            conflict[bi] = 1.0
        for i in range(csp.n):
            for v in ded[i]:
                tgt[bi, i, v] = 1.0
    return torch.as_tensor(tgt, device=dev), torch.as_tensor(conflict, device=dev)


def fwd(m, feat, var_mask):
    return m(var_mask, feat["given"], feat["fac_rel"], feat["fac_arity"],
             feat["edge_var"], feat["edge_valid"], feat["var_valid"], feat["fac_valid"])


# ============================================================ eval (per-rung completeness vs dedP)
@torch.no_grad()
def run_to_fixpoint(m, csps, dev, theta=0.5, R_max=64):
    items = [(c, c.full()) for c in csps]
    feat = featurize(items, dev)
    vm = feat["var_mask"].clone()
    B = len(csps)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    fe_k = fe_n = 0
    for _ in range(R_max):
        b, cls, _ = fwd(m, feat, vm)
        new_vm = RG.meet(vm, b, theta)
        cur = [(csps[i], RG.dom_from_mask(vm[i].cpu().numpy(), csps[i])) for i in range(B)]
        tgt, _ = build_targets(cur, RG.SHARED, dev)
        k, n = RG.false_elim(vm, new_vm, feat["var_valid"], tgt); fe_k += k; fe_n += n
        changed = (new_vm != vm).any(-1).any(-1)
        vm = torch.where(done.view(-1, 1, 1), vm, new_vm)
        done = done | ~changed
        if done.all():
            break
    return [RG.dom_from_mask(vm[i].cpu().numpy(), csps[i]) for i in range(B)], (fe_k, fe_n)


@torch.no_grad()
def evaluate_rung(m, csps, dev, theta=0.5, R_max=64):
    m.eval()
    doms, (fe_k, fe_n) = run_to_fixpoint(m, csps, dev, theta, R_max)
    n_solv = solved = wrong = uniq = match = 0
    rec_num = rec_den = 0
    for csp, dom in zip(csps, doms):
        sols = C.solutions(csp)
        if not sols:
            continue
        n_solv += 1
        full = csp.full()
        ded, _ = C.to_fixpoint(C.exact_dedP, csp, csp.full())
        rm_model = RG._removed(full, dom, csp.n)
        rm_ded = RG._removed(full, ded, csp.n)
        rec_num += len(rm_model & rm_ded); rec_den += len(rm_ded)
        match += int(dom == ded)
        uniq += int(all(len(c) == 1 for c in ded))
        st = C.status(dom)
        is_solved = st == "solved"
        solved += is_solved
        wrong += int(is_solved and tuple(next(iter(dom[i])) for i in range(csp.n)) not in sols)
    m.train()
    ns = max(1, n_solv)
    return {"n_solvable": n_solv,
            "narrowing_recall": rec_num / max(1, rec_den),
            "match_dedP": match / ns, "uniq_rate": uniq / ns, "solved": solved / ns,
            "wrong_return_rate": wrong / ns,
            "false_elim": fe_k / max(1, fe_n), "false_elim_count": int(fe_k)}


# ============================================================ sizing + training
def matched_sizes(target, R, ds):
    dt, pt = size_for("table", N_MAX, D_MAX, M_MAX, A_MAX, target, R=R, ds=ds)
    out = {"table": (dt, pt)}
    for msg in ("minor", "scan"):
        out[msg] = size_for(msg, N_MAX, D_MAX, M_MAX, A_MAX, pt, R=R, ds=ds)
    return out


def train(msg, rungs, dev, d, steps, pool=128, R=8, lr=3e-4, theta=0.5, seed=0, ex=RG.SHARED):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = BladeHiDeductor(msg, N_MAX, D_MAX, M_MAX, A_MAX, d=d, R=R, ds=min(4, R)).to(dev)
    npar = m.n_params()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95))
    log = []
    tagged = sample_corpus(rng, pool, rungs)
    items = [(c, c.full()) for _, c in tagged]
    rtags = [rg for rg, _ in tagged]
    print(f"  [{msg:5s}] train {rungs} d={d} params={npar:,} pool={pool} R={R} steps={steps}", flush=True)
    fe_k = fe_n = 0
    t0 = time.time()
    for s in range(1, steps + 1):
        feat = featurize(items, dev)
        vm = feat["var_mask"]
        tgt, conflict = build_targets(items, ex, dev)
        b, cls, sup = fwd(m, feat, vm)
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
                    rg = rtags[bi]; nc = sample_csp(rng, rg); nxt.append((nc, nc.full()))
                else:
                    nxt.append((csp, ndom))
            items = nxt
        if s % max(1, steps // 10) == 0:
            fer = fe_k / max(1, fe_n)
            log.append({"step": s, "loss": float(loss.detach()), "false_elim": fer})
            print(f"    [{msg:5s}] step {s:5d}  loss {float(loss.detach()):.3f}  fe {fer:.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            fe_k = fe_n = 0
    return m, npar, log


# ============================================================ experiments
def _eval(m, sets, dev, args):
    return {rg: evaluate_rung(m, c, dev, args.theta, args.rmax) for rg, c in sets.items()}


def _ptable(title, res, rungs, modes=("table", "minor", "scan")):
    print(f"\n  {title}", flush=True)
    head = " ".join(f"{md:>9s}" for md in modes)
    print(f"  {'rung':6s}  recall(dedP): {head:>30s}      false-elim: " +
          " ".join(f"{md:>7s}" for md in modes), flush=True)
    for rg in rungs:
        rec = " ".join(f"{res[md][rg]['narrowing_recall']*100:8.1f}%" for md in modes)
        fe = " ".join(f"{res[md][rg]['false_elim']:7.4f}" for md in modes)
        print(f"  {rg:6s}  {rec}      {fe}", flush=True)


def run_full(args, dev):
    out = {"budget": [N_MAX, D_MAX, M_MAX, A_MAX], "args": vars(args)}
    modes = ("table", "minor", "scan")
    sz = matched_sizes(args.target, args.R, min(4, args.R))
    out["sizes"] = {k: {"d": v[0], "params": v[1]} for k, v in sz.items()}
    print("iso-param sizes: " + "  ".join(f"{k} d={v[0]} p={v[1]:,}" for k, v in sz.items()), flush=True)
    eval_all = make_eval_sets(RUNGS, args.neval)

    # ---- A. MULTITASK ----
    print("\n========== A. MULTITASK (train par3+par4+par5, eval each) ==========", flush=True)
    A = {}
    for msg in modes:
        m, npar, log = train(msg, RUNGS, dev, sz[msg][0], args.steps, pool=args.pool, R=args.R,
                             lr=args.lr, theta=args.theta, seed=args.seed)
        A[msg] = _eval(m, eval_all, dev, args)
    _ptable("A. multitask per-arity completeness", A, RUNGS)
    out["multitask"] = A

    # ---- B. ZERO-SHOT: train par3 ONLY, eval par4 + par5 ----
    print("\n========== B. ZERO-SHOT (train par3 ONLY, eval par4 & par5) ==========", flush=True)
    eval_hi = {rg: eval_all[rg] for rg in ("par4", "par5")}
    Bx = {}
    for msg in modes:
        m, npar, log = train(msg, ["par3"], dev, sz[msg][0], args.steps, pool=args.pool, R=args.R,
                             lr=args.lr, theta=args.theta, seed=args.seed)
        Bx[msg] = _eval(m, eval_hi, dev, args)
    _ptable("B. zero-shot to UNSEEN higher arity (trained on arity-3 parity ONLY)", Bx,
            ["par4", "par5"])
    out["zeroshot"] = Bx

    print("\n========== HEADLINE (narrowing-recall vs dedP) ==========", flush=True)
    for rg in RUNGS:
        print(f"  multitask {rg}: " + "  ".join(f"{md} {A[md][rg]['narrowing_recall']*100:5.1f}%"
              for md in modes), flush=True)
    for rg in ("par4", "par5"):
        print(f"  zero-shot {rg}: " + "  ".join(f"{md} {Bx[md][rg]['narrowing_recall']*100:5.1f}%"
              for md in modes), flush=True)
    return out


def run_smoke(args, dev):
    print("\n===== SMOKE hi (3 modes train; parity arity 3/4/5) =====", flush=True)
    sz = matched_sizes(1.2e5, args.R, min(4, args.R))
    print("sizes: " + "  ".join(f"{k} d={v[0]} p={v[1]:,}" for k, v in sz.items()), flush=True)
    eval_sets = make_eval_sets(RUNGS, 50)
    out = {}
    for msg in ("table", "minor", "scan"):
        m, npar, log = train(msg, RUNGS, dev, sz[msg][0], steps=args.steps or 150, pool=64,
                             R=args.R, lr=args.lr, theta=args.theta, seed=0)
        out[msg] = _eval(m, eval_sets, dev, args)
        print(f"  [{msg}] " + "  ".join(f"{rg} {out[msg][rg]['narrowing_recall']*100:.1f}%"
              for rg in RUNGS), flush=True)
    return {"smoke": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--target", type=float, default=2.0e5)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--pool", type=int, default=128)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--neval", type=int, default=200)
    ap.add_argument("--rmax", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = RG.device()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    print(f"device={dev}  budget N={N_MAX} D={D_MAX} M={M_MAX} A={A_MAX}  rungs={RUNGS}", flush=True)
    out = run_smoke(args, dev) if args.mode == "smoke" else run_full(args, dev)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
        print("\nwrote", args.out, flush=True)


if __name__ == "__main__":
    main()
