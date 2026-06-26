"""clair/big_organ.py — the LARGER-BUDGET finite organ (Ember's ask).

organ_excellence capped at N_MAX=8, which blocks the most valuable rg CSPs (4x4 sudoku / futoshiki =
16 cells, 3x3 latin = 9, 5x5 latin = 25). This scales the factor-graph / blade organ's budget up
(N<=25, D<=6, M<=160, A<=3) and trains the best config (blade, R12, dominate-dedP) on a LARGE-N
curriculum that includes the Hall-set weak spot (latin squares + 4x4 sudoku + futoshiki + alldiff).

Everything is witness-first (no reasoning_gym needed): we sample a valid filled grid, reveal a subset
of cells as pins -> always solvable, exact clair.csp.exact_dedP gives ground truth. The earlier
ENUM_CAP=d**n<=4096 gate is dropped: exact dedP early-stops once every alive value is witnessed, so
even fully-open 5x5 latin costs <60ms (probed).

Reports per-rung narrowing-recall / soundness at N up to 25, and OOD on HELD-OUT generators
(futoshiki, n_queens) the organ never trained on. Saves runs/big_organ.pt.

  python -m clair.big_organ --smoke
  python -m clair.big_organ --train --steps 1500 --R 12 --target 4e5 --ckpt runs/big_organ.pt
"""
from __future__ import annotations

import argparse, json, os, time
import itertools as it
import numpy as np
import torch

from . import csp as C
from . import run_general as RG

# ---- the LARGER budget (override run_general's module globals so its featurize/targets use it) ----
N_MAX, D_MAX, M_MAX, A_MAX = 25, 6, 160, 3
RG.N_MAX, RG.D_MAX, RG.M_MAX, RG.A_MAX = N_MAX, D_MAX, M_MAX, A_MAX
RG.ENUM_CAP = 10 ** 18                                    # drop the d**n gate; dedP early-stops

from .blade_deductor import BladeFactorDeductor, size_for   # noqa: E402


# ============================================================ witness-first large-N generators
def _alldiff_cons(cells, d):
    return [C._rel((cells[a], cells[b]), lambda t: t[0] != t[1], d)
            for a in range(len(cells)) for b in range(a + 1, len(cells))]


def random_latin(rng, n):
    """Backtracking-filled n x n latin square (values 0..n-1)."""
    g = [[-1] * n for _ in range(n)]
    def ok(r, c, v):
        return all(g[r][k] != v and g[k][c] != v for k in range(n))
    def fill(idx):
        if idx == n * n:
            return True
        r, c = divmod(idx, n)
        vals = list(range(n)); rng.shuffle(vals)
        for v in vals:
            if ok(r, c, v):
                g[r][c] = v
                if fill(idx + 1):
                    return True
                g[r][c] = -1
        return False
    fill(0)
    return g


def gen_latin(rng, n_lo=3, n_hi=5):
    """n x n latin square (rows+cols alldiff as pairwise !=) + a revealed subset of pins. The Hall-set
    workout: alldiff groups of size n sharing cells, narrowing needs Hall reasoning the binary lattice
    only partly does — the organ must learn to match the exact dedP."""
    n = int(rng.integers(n_lo, n_hi + 1))
    g = random_latin(rng, n)
    cell = lambda r, c: r * n + c
    cons = []
    for r in range(n):
        cons += _alldiff_cons([cell(r, c) for c in range(n)], n)
        cons += _alldiff_cons([cell(c, r) for c in range(n)], n)
    reveal = int(rng.integers(n, 2 * n + 1))
    cells = list(range(n * n)); rng.shuffle(cells)
    for idx in cells[:reveal]:
        r, c = divmod(idx, n); v = g[r][c]
        cons.append(C._rel((idx,), (lambda v: lambda t: t[0] == v)(v), n))
    return C.CSP(n * n, n, tuple(cons))


def gen_sudoku4(rng):
    """4x4 sudoku: latin + four 2x2 boxes alldiff + revealed pins (16 cells, d=4)."""
    while True:
        g = random_latin(rng, 4)
        if all(len({g[br + i][bc + j] for i in range(2) for j in range(2)}) == 4
               for br in (0, 2) for bc in (0, 2)):
            break
    cell = lambda r, c: r * 4 + c
    cons = []
    for r in range(4):
        cons += _alldiff_cons([cell(r, c) for c in range(4)], 4)
        cons += _alldiff_cons([cell(c, r) for c in range(4)], 4)
    for br in (0, 2):
        for bc in (0, 2):
            cons += _alldiff_cons([cell(br + i, bc + j) for i in range(2) for j in range(2)], 4)
    reveal = int(rng.integers(6, 11))
    cells = list(range(16)); rng.shuffle(cells)
    for idx in cells[:reveal]:
        r, c = divmod(idx, 4); v = g[r][c]
        cons.append(C._rel((idx,), (lambda v: lambda t: t[0] == v)(v), 4))
    return C.CSP(16, 4, tuple(cons))


def gen_futoshiki(rng, n_lo=3, n_hi=5):
    """n x n latin + a few adjacent inequality (<) constraints consistent with the witness + pins."""
    n = int(rng.integers(n_lo, n_hi + 1))
    g = random_latin(rng, n)
    cell = lambda r, c: r * n + c
    cons = []
    for r in range(n):
        cons += _alldiff_cons([cell(r, c) for c in range(n)], n)
        cons += _alldiff_cons([cell(c, r) for c in range(n)], n)
    # inequalities between horizontally/vertically adjacent cells, oriented per the witness
    pairs = []
    for r in range(n):
        for c in range(n):
            if c + 1 < n:
                pairs.append(((r, c), (r, c + 1)))
            if r + 1 < n:
                pairs.append(((r, c), (r + 1, c)))
    rng.shuffle(pairs)
    for (r1, c1), (r2, c2) in pairs[:int(rng.integers(n, 2 * n))]:
        a, b = cell(r1, c1), cell(r2, c2)
        if g[r1][c1] < g[r2][c2]:
            cons.append(C._rel((a, b), lambda t: t[0] < t[1], n))
        else:
            cons.append(C._rel((a, b), lambda t: t[0] > t[1], n))
    reveal = int(rng.integers(max(1, n - 1), n + 2))
    cells = list(range(n * n)); rng.shuffle(cells)
    for idx in cells[:reveal]:
        r, c = divmod(idx, n); v = g[r][c]
        cons.append(C._rel((idx,), (lambda v: lambda t: t[0] == v)(v), n))
    return C.CSP(n * n, n, tuple(cons))


def gen_alldiff_big(rng, n_lo=4, n_hi=6):
    """Single alldiff block over n cells, d in [n, 6] + pins (a pure Hall-set rung, small domain)."""
    n = int(rng.integers(n_lo, n_hi + 1))
    d = int(rng.integers(n, min(6, n + 2) + 1))
    s = np.array(rng.permutation(d)[:n])
    cons = _alldiff_cons(list(range(n)), d)
    for i in range(n):
        if rng.random() < 0.45:
            cons.append(C._rel((i,), (lambda v: lambda t: t[0] == v)(int(s[i])), d))
    return C.CSP(n, d, tuple(cons))


def gen_nqueens(rng, n_lo=4, n_hi=6):
    """Witness-first n-queens (HELD-OUT/OOD): valid solution -> one var/row (domain=col), distinct
    cols + off-diagonal pairwise (novel arity-2 relation), some rows pinned. d=n<=6."""
    n = int(rng.integers(n_lo, n_hi + 1))
    cols = list(range(n))
    sol = None
    for _ in range(400):
        rng.shuffle(cols)
        if all(abs(cols[i] - cols[j]) != j - i for i in range(n) for j in range(i + 1, n)):
            sol = list(cols); break
    if sol is None:
        sol = list(range(n))
    cons = []
    for i in range(n):
        for j in range(i + 1, n):
            cons.append(C._rel((i, j), lambda t: t[0] != t[1], n))
            dij = j - i
            cons.append(C._rel((i, j), (lambda d: lambda t: abs(t[0] - t[1]) != d)(dij), n))
    rows = list(range(n)); rng.shuffle(rows)
    for r in rows[:int(rng.integers(1, n))]:
        cons.append(C._rel((r,), (lambda v: lambda t: t[0] == v)(sol[r]), n))
    return C.CSP(n, n, tuple(cons))


def _within_budget(csp):
    return (csp.n <= N_MAX and csp.d <= D_MAX and len(csp.cons) <= M_MAX
            and all(len(sc) <= A_MAX for sc, _ in csp.cons) and len(C.solutions(csp, limit=1)) > 0)


GEN = {
    "coloring": lambda rng: RG.sample_rung_csp(rng, "coloring"),
    "arithmetic": lambda rng: RG.sample_rung_csp(rng, "arithmetic"),
    "ordering": lambda rng: RG.sample_rung_csp(rng, "ordering"),
    "latin": gen_latin,
    "sudoku4": gen_sudoku4,
    "alldiff_big": gen_alldiff_big,
}
TRAIN_RUNGS = ["coloring", "arithmetic", "ordering", "latin", "sudoku4", "alldiff_big"]
OOD_RUNGS = {"futoshiki": gen_futoshiki, "n_queens": gen_nqueens, "latin5": lambda rng: gen_latin(rng, 5, 5)}


def sample(rng, rung):
    for _ in range(60):
        csp = GEN[rung](rng)
        if _within_budget(csp):
            return csp
    return RG.sample_rung_csp(rng, "coloring")


def sample_corpus(rng, n, rungs):
    return [(rungs[i % len(rungs)], sample(rng, rungs[i % len(rungs)])) for i in range(n)]


def make_eval(rungs, neval, seed=4321):
    rng = np.random.default_rng(seed)
    out = {}
    for rg in rungs:
        gen = GEN.get(rg) or OOD_RUNGS[rg]
        cs = []
        while len(cs) < neval:
            csp = gen(rng)
            if _within_budget(csp):
                cs.append(csp)
        out[rg] = cs
    return out


# ============================================================ lean eval (skip O(F^2) factor lattice)
def _sat(csp, assign):
    return all(tuple(assign[c] for c in sc) in al for sc, al in csp.cons)


@torch.no_grad()
def evaluate_big(m, csps, dev, theta=0.5, R_max=96):
    """Per-rung soundness + completeness vs exact dedP, WITHOUT the level-3 factor-fixpoint reference
    (that reference is O(F^2) and explodes on 25-cell latin). dedP early-stops, so it stays cheap."""
    m.eval()
    doms, (fe_k, fe_n) = RG.run_to_fixpoint(m, csps, dev, theta, R_max)
    n_solv = solved = wrong = openab = match = uniq = 0
    rec_num = rec_den = 0
    for csp, dom in zip(csps, doms):
        if not C.solutions(csp, limit=1):
            continue
        n_solv += 1
        full = csp.full()
        ded, _ = C.to_fixpoint(C.exact_dedP, csp, csp.full())
        rm_model = {(i, v) for i in range(csp.n) for v in full[i] if v not in dom[i]}
        rm_ded = {(i, v) for i in range(csp.n) for v in full[i] if v not in ded[i]}
        rec_num += len(rm_model & rm_ded); rec_den += len(rm_ded)
        match += int(dom == ded)
        uniq += int(all(len(c) == 1 for c in ded))
        st = C.status(dom); solved += int(st == "solved"); openab += int(st == "open")
        if st == "solved":
            wrong += int(not _sat(csp, [next(iter(dom[i])) for i in range(csp.n)]))
    m.train()
    ns = max(1, n_solv)
    return {"n_solvable": n_solv, "narrowing_recall": rec_num / max(1, rec_den),
            "factor_recall": 0.0, "match_dedP": match / ns, "uniq_rate": uniq / ns,
            "solved": solved / ns, "abstain_rate": openab / ns, "wrong_return_rate": wrong / ns,
            "false_elim": fe_k / max(1, fe_n), "false_elim_count": int(fe_k)}


# ============================================================ training (blade, dominate-dedP)
def train(rungs, dev, d, steps, pool=64, R=12, lr=3e-4, theta=0.5, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = BladeFactorDeductor("blade", N_MAX, D_MAX, M_MAX, A_MAX, d=d, R=R, ds=min(4, R)).to(dev)
    npar = m.n_params()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95))
    tagged = sample_corpus(rng, pool, rungs)
    items = [(c, c.full()) for _, c in tagged]
    rtags = [rg for rg, _ in tagged]
    print(f"  big organ: d={d} params={npar:,} pool={pool} R={R} steps={steps} budget "
          f"N={N_MAX} D={D_MAX} M={M_MAX}", flush=True)
    fe_k = fe_n = 0; t0 = time.time(); log = []
    for s in range(1, steps + 1):
        feat = RG.featurize(items, dev); vm = feat["var_mask"]
        tgt, conflict = RG.build_targets(items, RG.SHARED, dev)
        b, cls, sup = RG.fwd(m, feat, vm)
        loss = RG.loss_fn(sup, vm, feat["var_valid"], tgt, conflict)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        with torch.no_grad():
            new_vm = RG.meet(vm, b, theta)
            k, nn_ = RG.false_elim(vm, new_vm, feat["var_valid"], tgt); fe_k += k; fe_n += nn_
            nvc = new_vm.cpu().numpy(); nxt = []
            for bi, (csp, dom) in enumerate(items):
                ndom = RG.dom_from_mask(nvc[bi], csp)
                if C.status(ndom) in ("solved", "conflict") or ndom == dom:
                    rg = rtags[bi]; nc = sample(rng, rg); nxt.append((nc, nc.full()))
                else:
                    nxt.append((csp, ndom))
            items = nxt
        if s % max(1, steps // 15) == 0:
            fer = fe_k / max(1, fe_n)
            log.append({"step": s, "loss": float(loss.detach()), "false_elim": fer})
            print(f"    step {s:5d}  loss {float(loss.detach()):.3f}  false_elim {fer:.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            fe_k = fe_n = 0
    return m, npar, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--R", type=int, default=12)
    ap.add_argument("--target", type=float, default=4e5)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--neval", type=int, default=120)
    ap.add_argument("--rmax", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="runs/big_organ.pt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = RG.device()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    print(f"device={dev}  budget N={N_MAX} D={D_MAX} M={M_MAX} A={A_MAX}", flush=True)

    if args.smoke:
        rng = np.random.default_rng(0)
        print("=== big_organ SMOKE: generators within budget + dedP ground truth ===", flush=True)
        for rg in list(GEN) + list(OOD_RUNGS):
            gen = GEN.get(rg) or OOD_RUNGS[rg]
            t0 = time.time(); csp = None
            for _ in range(40):
                c = gen(rng)
                if _within_budget(c):
                    csp = c; break
            ded, _ = C.to_fixpoint(C.exact_dedP, csp, csp.full())
            print(f"  {rg:12s} n={csp.n:2d} d={csp.d} m={len(csp.cons):3d} "
                  f"solved-by-dedP={C.status(ded)=='solved'} ({(time.time()-t0)*1000:.0f}ms)", flush=True)
        d, npar = size_for("blade", N_MAX, D_MAX, M_MAX, A_MAX, args.target, R=args.R, ds=4)
        print(f"  size_for(blade,target={args.target:.0e}) -> d={d} params={npar:,}", flush=True)
        m, npar, log = train(TRAIN_RUNGS, dev, d, steps=120, pool=48, R=args.R, seed=0)
        ev = {rg: evaluate_big(m, cs, dev, args.theta, args.rmax)
              for rg, cs in make_eval(TRAIN_RUNGS, 40).items()}
        for rg in TRAIN_RUNGS:
            print(RG._row(rg, ev[rg]), flush=True)
        return

    if args.train:
        d, npar = size_for("blade", N_MAX, D_MAX, M_MAX, A_MAX, args.target, R=args.R, ds=4)
        m, npar, log = train(TRAIN_RUNGS, dev, d, args.steps, pool=args.pool, R=args.R,
                             lr=args.lr, theta=args.theta, seed=args.seed)
        print("\n=== BIG organ per-rung (in-distribution, N up to 25) ===", flush=True)
        ev = {rg: evaluate_big(m, cs, dev, args.theta, args.rmax)
              for rg, cs in make_eval(TRAIN_RUNGS, args.neval).items()}
        for rg in TRAIN_RUNGS:
            print(RG._row(rg, ev[rg]), flush=True)
        print("\n=== OOD (HELD-OUT generators never trained) — soundness gate ===", flush=True)
        ood = {rg: evaluate_big(m, cs, dev, args.theta, args.rmax)
               for rg, cs in make_eval(list(OOD_RUNGS), args.neval).items()}
        for rg in OOD_RUNGS:
            print(RG._row(rg, ood[rg]), flush=True)
        os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
        torch.save({"state_dict": m.state_dict(),
                    "config": {"msg": "blade", "n_max": N_MAX, "d_max": D_MAX, "m_max": M_MAX,
                               "a_max": A_MAX, "d": d, "R": args.R, "ds": min(4, args.R)},
                    "params": npar, "train_rungs": TRAIN_RUNGS, "eval": ev, "ood": ood,
                    "log": log, "args": vars(args)}, args.ckpt)
        print(f"\nsaved checkpoint -> {args.ckpt} ({npar:,} params)", flush=True)
        if args.out:
            json.dump({"eval": ev, "ood": ood, "log": log, "params": npar},
                      open(args.out, "w"), indent=1, default=str)
            print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
