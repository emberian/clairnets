"""clair/xor_wall.py — IS THE AFFINE WALL ACTUALLY BEATEN? (codex's honest-claim gate)

The earlier claim was "grade-3 / level-2 solves the affine wall". codex's correction: that is only
the LOCAL representation of ONE ternary affine constraint. Arbitrary XOR *systems* (parity equations
over GF(2)) are a WIDTH / algorithmic problem: deciding/solving them requires GAUSSIAN-ELIMINATION /
row-space state, not "triples + depth" at a fixed lattice level. Bounded-width local consistency at
any FIXED level k provably FAILS on affine systems (affine ⇒ unbounded width).

This module tests it rigorously and organ-independently:

  * generate arity-3 XOR-SAT systems over GF(2), uniquely solvable (full rank), swept by SIZE (n),
    and by a controllable WIDTH knob (bandwidth of the constraint incidence graph ⇒ pathwidth);
  * GROUND TRUTH = GF(2) Gaussian elimination (the forced bits = the unique solution). For affine
    systems this EQUALS the exact per-cell transformer dedP (verified on small n).
  * measure the EXACT level-k factor lattice (clair.csp.solve_factor machinery) at k=3,4,5,...: does
    its completeness COLLAPSE as width grows at fixed k, and does the required k track the width?
  * the neural arity-3 factor-graph / blade organ is upper-bounded by the level-3 factor closure
    (it only ever sees arity-<=3 factor marginals), so depth R cannot beat width — confirmed
    empirically by training blade organs at R in {4..24} and showing recall on high-width systems
    stays at/below the level-3 lattice regardless of R.

VERDICT printed at the end: is the affine wall beaten by depth+grade-3 (completeness holds across
rank/width), or does high-rank/high-width XOR need a fundamentally different ROW-SPACE organ?

  python -m clair.xor_wall --smoke
  python -m clair.xor_wall --lattice --out runs/xor_lattice.json
  python -m clair.xor_wall --organ   --out runs/xor_organ.json
"""
from __future__ import annotations

import argparse, json, os, time
import itertools as it
import numpy as np

from . import csp as C


# ============================================================ GF(2) linear algebra (the row-space organ)
def gf2_rref(A, b):
    """Reduced row echelon over GF(2). A: (m,n) uint8, b: (m,) uint8. Returns (R, rb, pivots, rank,
    consistent)."""
    A = (A % 2).astype(np.uint8).copy()
    b = (b % 2).astype(np.uint8).copy()
    m, n = A.shape
    pivots = []
    r = 0
    for c in range(n):
        piv = None
        for i in range(r, m):
            if A[i, c]:
                piv = i; break
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]; b[[r, piv]] = b[[piv, r]]
        for i in range(m):
            if i != r and A[i, c]:
                A[i] ^= A[r]; b[i] ^= b[r]
        pivots.append(c); r += 1
        if r == m:
            break
    consistent = True
    for i in range(r, m):
        if b[i] and not A[i].any():
            consistent = False
    return A, b, pivots, r, consistent


def gf2_forced(A, b, n):
    """Per-variable forced set from the GF(2) system Ax=b: frozenset({v}) if the variable is the same
    in every solution (the unique value), else frozenset({0,1}). This is the EXACT affine dedP."""
    R, rb, pivots, rank, consistent = gf2_rref(A, b)
    if not consistent:
        return tuple(frozenset() for _ in range(n)), rank, False   # ⊥
    pivots = list(pivots)
    free = [c for c in range(n) if c not in set(pivots)]
    # particular solution: free vars = 0, pivots = rb of their row
    x0 = np.zeros(n, np.uint8)
    for ri, pc in enumerate(pivots):
        x0[pc] = rb[ri]
    # null-space basis: one vector per free var
    forced = [True] * n
    for fv in free:
        forced[fv] = False                                          # free var itself varies
        # nz: pivot pc varies with fv iff R[row(pc), fv] == 1
        for ri, pc in enumerate(pivots):
            if R[ri, fv]:
                forced[pc] = False
    out = tuple(frozenset({int(x0[i])}) if forced[i] else frozenset({0, 1}) for i in range(n))
    return out, rank, True


# ============================================================ XOR-SAT system generation (width knob)
def gen_xor_system(rng, n, n_par, band, force_unique=True):
    """Witness-first arity-3 XOR-SAT system. Sample s in GF(2)^n; emit n_par parity constraints
    a^b^c = s[a]^s[b]^s[c], each over 3 distinct vars drawn from a WINDOW of width `band` (band=3 ->
    tight local 'near-tree' chain; band=n -> global high-treewidth). Then pin variables (greedily,
    each increasing GF(2) rank) until the system has rank n (unique solution = s) if force_unique.

    Returns dict with the clair.csp.CSP, the GF(2) (A,b), the witness s, rank, band, and arity (3)."""
    n = int(n); band = int(min(band, n))
    s = rng.integers(0, 2, n).astype(np.uint8)
    rows = []; rhs = []
    cons = []
    seen = set()
    tries = 0
    while len([1 for _ in cons]) < n_par and tries < n_par * 40:
        tries += 1
        start = int(rng.integers(0, max(1, n - band + 1)))
        win = list(range(start, start + band))
        if len(win) < 3:
            win = list(range(n))
        trip = tuple(sorted(rng.choice(win, size=3, replace=False)))
        if trip in seen:
            continue
        seen.add(trip)
        a, bb, c = trip
        p = int(s[a] ^ s[bb] ^ s[c])
        allowed = frozenset(t for t in it.product((0, 1), repeat=3) if (t[0] ^ t[1] ^ t[2]) == p)
        cons.append(((a, bb, c), allowed))
        row = np.zeros(n, np.uint8); row[a] = row[bb] = row[c] = 1
        rows.append(row); rhs.append(p)
    # pin to reach full rank (unique) if requested
    def cur_rank():
        if not rows:
            return 0
        _, _, _, rk, _ = gf2_rref(np.array(rows, np.uint8), np.array(rhs, np.uint8))
        return rk
    if force_unique:
        order = list(range(n)); rng.shuffle(order)
        rk_now = cur_rank()
        for i in order:
            if rk_now >= n:
                break
            row = np.zeros(n, np.uint8); row[i] = 1
            test_rows = rows + [row]; test_rhs = rhs + [int(s[i])]
            _, _, _, rk, _ = gf2_rref(np.array(test_rows, np.uint8), np.array(test_rhs, np.uint8))
            if rk > rk_now:                                          # accept only rank-raising pins
                rows = test_rows; rhs = test_rhs; rk_now = rk
                cons.append(((i,), frozenset({(int(s[i]),)})))
    A = np.array(rows, np.uint8) if rows else np.zeros((0, n), np.uint8)
    b = np.array(rhs, np.uint8) if rhs else np.zeros((0,), np.uint8)
    _, _, _, rank, _ = gf2_rref(A, b)
    csp = C.CSP(n, 2, tuple(cons))
    return {"csp": csp, "A": A, "b": b, "s": s, "n": n, "rank": int(rank), "band": band,
            "n_par": sum(1 for sc, _ in cons if len(sc) == 3), "n_pin": sum(1 for sc, _ in cons if len(sc) == 1)}


def gen_tree_xor_unique(rng, n, band):
    """CHEAP uniquely-solvable arity-3 XOR system (no GF(2) loop): feed-forward parity DAG — each
    non-source cell c = xor of two earlier cells within a window of width `band`, all source cells
    pinned. Forward evaluation makes it unique; `band` controls pathwidth (3=near-tree/low-width,
    n=global/high-width). Same affine wall, consistent train/eval distribution."""
    n = int(n); band = int(min(band, n))
    s = rng.integers(0, 2, n).astype(np.uint8)
    n_src = int(rng.integers(2, max(3, n // 2)))
    cons = []; rows = []; rhs = []
    allowed0 = frozenset(t for t in it.product((0, 1), repeat=3) if (t[0] ^ t[1] ^ t[2]) == 0)
    for c in range(n_src, n):
        lo = max(0, c - band); choices = list(range(lo, c))
        if len(choices) >= 2:
            a, bb = (int(x) for x in rng.choice(choices, size=2, replace=False))
        else:
            a = bb = choices[0]
        s[c] = int(s[a]) ^ int(s[bb])
        cons.append(((a, bb, c), allowed0))
        row = np.zeros(n, np.uint8); row[a] ^= 1; row[bb] ^= 1; row[c] ^= 1
        rows.append(row); rhs.append(0)
    for i in range(n_src):
        cons.append(((i,), frozenset({(int(s[i]),)})))
        row = np.zeros(n, np.uint8); row[i] = 1; rows.append(row); rhs.append(int(s[i]))
    A = np.array(rows, np.uint8) if rows else np.zeros((0, n), np.uint8)
    b = np.array(rhs, np.uint8) if rhs else np.zeros((0,), np.uint8)
    return {"csp": C.CSP(n, 2, tuple(cons)), "A": A, "b": b, "s": s, "n": n, "band": band}


# ============================================================ level-k factor lattice (exact, organ-independent)
def factor_fixpoint_cells(csp, k):
    """Exact level-k factor-consistency fixpoint -> per-cell domains. k=3 is the arity-3 organ's
    ceiling; raising k adds factors spanning unions of constraints (the row-space the lattice lacks)."""
    factors = C.default_factors(csp, k)
    st = C.factor_init(csp, factors)
    for _ in range(csp.n * csp.d * csp.d + 2):
        nxt = C.factor_step(csp, st, factors)
        if nxt == st:
            break
        st = nxt
    return C.factor_cells(csp, st, factors), len(factors)


def removed(full, dom, n):
    return {(i, v) for i in range(n) for v in full[i] if v not in dom[i]}


def recall_vs(forced, dom, n, full):
    """narrowing-recall: of the (cell,value) eliminations the ground-truth (GF(2)/dedP) makes, what
    fraction does `dom` also make. 1.0 = reaches the exact forced state."""
    rm_gt = removed(full, forced, n)
    rm = removed(full, dom, n)
    fe = len(rm - rm_gt)                                            # false elim (should be 0: sound)
    return (len(rm & rm_gt) / max(1, len(rm_gt)), fe, len(rm_gt))


# ============================================================ verification: GF(2) forced == exact dedP
def verify_gf2_equals_dedp(n_inst=40, seed=0):
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(n_inst):
        n = int(rng.integers(4, 9))
        sysd = gen_xor_system(rng, n, n_par=int(rng.integers(2, 2 * n)), band=int(rng.integers(3, n + 1)),
                              force_unique=bool(rng.integers(0, 2)))
        csp = sysd["csp"]
        gf, rank, ok = gf2_forced(sysd["A"], sysd["b"], n)
        ded, _ = C.to_fixpoint(C.exact_dedP, csp, csp.full())
        if tuple(sorted(x) for x in gf) != tuple(sorted(x) for x in ded):
            bad += 1
    return bad, n_inst


# ============================================================ the lattice sweeps
def lattice_width_sweep(n=12, bands=(3, 4, 6, 9, 12), ks=(3, 4, 5, 6), n_inst=8,
                        par_mult=1.4, seed=1):
    """Fixed n, UNIQUELY-determined systems (GF(2) forces all n bits), vary bandwidth (=pathwidth).
    For each band: GF(2) forced-fraction (=1.0), and the level-k factor lattice recall + solved rate.
    The headline: level-3 recall COLLAPSES as band grows; the required k tracks the width."""
    rng = np.random.default_rng(seed)
    rows = []
    for band in bands:
        agg = {k: {"rec": [], "solved": [], "fe": [], "nfac": []} for k in ks}
        ranks = []; gtfrac = []
        for _ in range(n_inst):
            sysd = gen_xor_system(rng, n, n_par=int(par_mult * n), band=band, force_unique=True)
            csp = sysd["csp"]; full = csp.full()
            forced, rank, ok = gf2_forced(sysd["A"], sysd["b"], n)
            ranks.append(rank)
            gtfrac.append(len(removed(full, forced, n)) / max(1, n))
            for k in ks:
                cells, nfac = factor_fixpoint_cells(csp, k)
                rec, fe, _ = recall_vs(forced, cells, n, full)
                agg[k]["rec"].append(rec)
                agg[k]["solved"].append(int(C.status(cells) == "solved"))
                agg[k]["fe"].append(fe); agg[k]["nfac"].append(nfac)
        row = {"band": band, "n": n, "mean_rank": float(np.mean(ranks)),
               "gt_forced_frac": float(np.mean(gtfrac)),
               "levels": {k: {"recall": float(np.mean(agg[k]["rec"])),
                              "solved": float(np.mean(agg[k]["solved"])),
                              "false_elim": int(np.sum(agg[k]["fe"])),
                              "mean_factors": float(np.mean(agg[k]["nfac"]))} for k in ks}}
        rows.append(row)
        msg = "  ".join(f"L{k}={row['levels'][k]['recall']*100:5.1f}%(solv{row['levels'][k]['solved']*100:3.0f})"
                        for k in ks)
        print(f"  band={band:2d} (n={n}, rank~{row['mean_rank']:.1f}, GF2=100%)  {msg}", flush=True)
    return rows


def lattice_size_sweep(ns=(6, 8, 10, 12, 14), band_mode="global", ks=(3, 4), n_inst=8,
                       par_mult=1.4, seed=2):
    """High-width (global band) uniquely-determined systems, vary n. Shows level-3 recall vs system
    SIZE/RANK: GF(2) stays 100%, level-3 collapses toward the local-pin-only floor."""
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        band = n if band_mode == "global" else max(3, n // 2)
        agg = {k: [] for k in ks}; solved = {k: [] for k in ks}; ranks = []
        for _ in range(n_inst):
            sysd = gen_xor_system(rng, n, n_par=int(par_mult * n), band=band, force_unique=True)
            csp = sysd["csp"]; full = csp.full()
            forced, rank, ok = gf2_forced(sysd["A"], sysd["b"], n); ranks.append(rank)
            for k in ks:
                cells, _ = factor_fixpoint_cells(csp, min(k, n))
                rec, fe, _ = recall_vs(forced, cells, n, full)
                agg[k].append(rec); solved[k].append(int(C.status(cells) == "solved"))
        row = {"n": n, "band": band, "mean_rank": float(np.mean(ranks)),
               "levels": {k: {"recall": float(np.mean(agg[k])), "solved": float(np.mean(solved[k]))} for k in ks}}
        rows.append(row)
        msg = "  ".join(f"L{k}={row['levels'][k]['recall']*100:5.1f}%" for k in ks)
        print(f"  n={n:2d} band={band:2d} rank~{row['mean_rank']:.1f} (GF2=100%)  {msg}", flush=True)
    return rows


# ============================================================ the neural-organ depth experiment
def organ_depth_experiment(Rs=(4, 8, 16, 24), steps=450, seed=0,
                           train_band=3, eval_bands=(3, 6, 12), n_eval=60):
    """Train blade (grade-3) XOR organs at increasing DEPTH R on near-tree (low-width) xor, then eval
    recall on low- AND high-width systems. The organ is upper-bounded by the level-3 factor closure
    (arity-<=3 factor graph), so depth must NOT rescue high-width recall. Budget: d=2, n<=14, a=3."""
    import torch
    from .blade_deductor import BladeFactorDeductor, size_for
    from . import run_general as RG
    # local budget for boolean xor systems
    NB, DB, MB, AB = 14, 2, 60, 3
    dev = RG.device()
    # monkeypatch RG budget so featurize/build_targets/run_to_fixpoint use it
    old = (RG.N_MAX, RG.D_MAX, RG.M_MAX, RG.A_MAX, RG.ENUM_CAP)
    RG.N_MAX, RG.D_MAX, RG.M_MAX, RG.A_MAX, RG.ENUM_CAP = NB, DB, MB, AB, 10 ** 18

    # PRE-GENERATE a fixed training bank ONCE (random-triple, uniquely-solvable, low-width affine
    # systems — these DO exhibit the wall, unlike forward-deterministic trees). Paying the GF(2)
    # rank cost once (~seconds) instead of every step is what makes training affordable.
    bankrng = np.random.default_rng(777)
    train_bank = []
    while len(train_bank) < 600:
        n = int(bankrng.integers(8, NB + 1))
        sysd = gen_xor_system(bankrng, n, n_par=int(1.4 * n), band=train_band, force_unique=True)
        if len(C.solutions(sysd["csp"], limit=2)) == 1:        # keep only the uniquely-solvable ones
            train_bank.append(sysd["csp"])
    print(f"  [organ] training bank: {len(train_bank)} unique band={train_band} affine systems", flush=True)

    def draw(rng):
        return train_bank[int(rng.integers(0, len(train_bank)))]

    def make_eval(band, seedv):
        rng = np.random.default_rng(seedv)
        out = []
        tries = 0
        while len(out) < n_eval and tries < n_eval * 30:
            tries += 1
            n = int(rng.integers(10, NB + 1))
            sysd = gen_xor_system(rng, n, n_par=int(1.4 * n), band=min(band, n), force_unique=True)
            if len(C.solutions(sysd["csp"], limit=2)) == 1:
                out.append(sysd)
        return out

    eval_sets = {b: make_eval(b, 9000 + b) for b in eval_bands}
    results = {}
    torch.manual_seed(seed)

    def train_R(R):
        rng = np.random.default_rng(100 + R)
        d, npar = size_for("blade", NB, DB, MB, AB, 2.0e5, R=R, ds=min(4, R))
        m = BladeFactorDeductor("blade", NB, DB, MB, AB, d=d, R=R, ds=min(4, R)).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, betas=(0.9, 0.95))
        pool = [draw(rng) for _ in range(64)]
        items = [(c, c.full()) for c in pool]
        t0 = time.time()
        for s in range(1, steps + 1):
            feat = RG.featurize(items, dev); vm = feat["var_mask"]
            tgt, conflict = RG.build_targets(items, RG.SHARED, dev)
            b, cls, sup = RG.fwd(m, feat, vm)
            loss = RG.loss_fn(sup, vm, feat["var_valid"], tgt, conflict)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            with torch.no_grad():
                nv = RG.meet(vm, b, 0.5).cpu().numpy()
                nxt = []
                for bi, (csp, dom) in enumerate(items):
                    ndom = RG.dom_from_mask(nv[bi], csp)
                    if C.status(ndom) in ("solved", "conflict") or ndom == dom:
                        nc = draw(rng); nxt.append((nc, nc.full()))
                    else:
                        nxt.append((csp, ndom))
                items = nxt
            if s % max(1, steps // 3) == 0:
                print(f"    [R={R}] step {s}/{steps} loss {float(loss.detach()):.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        return m, d, npar

    for R in Rs:
        m, d, npar = train_R(R)
        rrow = {"d": d, "params": npar, "bands": {}}
        for band, sysds in eval_sets.items():
            csps = [s["csp"] for s in sysds]
            doms, (fek, fen) = RG.run_to_fixpoint(m, csps, dev, 0.5, 64)
            recs = []; lat3 = []
            for sysd, dom in zip(sysds, doms):
                n = sysd["n"]; csp = sysd["csp"]; full = csp.full()
                forced, _, _ = gf2_forced(sysd["A"], sysd["b"], n)
                rec, fe, _ = recall_vs(forced, dom, n, full); recs.append(rec)
                cells, _ = factor_fixpoint_cells(csp, 3)
                l3, _, _ = recall_vs(forced, cells, n, full); lat3.append(l3)
            rrow["bands"][band] = {"organ_recall": float(np.mean(recs)),
                                   "lattice_L3_recall": float(np.mean(lat3)),
                                   "false_elim": int(fek)}
        results[R] = rrow
        msg = "  ".join(f"band{b}: organ {rrow['bands'][b]['organ_recall']*100:4.1f}% / L3 "
                        f"{rrow['bands'][b]['lattice_L3_recall']*100:4.1f}%" for b in eval_bands)
        print(f"  R={R:2d} (d={d}, {npar//1000}k)  {msg}", flush=True)

    RG.N_MAX, RG.D_MAX, RG.M_MAX, RG.A_MAX, RG.ENUM_CAP = old
    return results


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lattice", action="store_true")
    ap.add_argument("--organ", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = {}

    if args.smoke:
        print("=== xor_wall SMOKE ===", flush=True)
        bad, tot = verify_gf2_equals_dedp(30)
        print(f"GF(2)-forced == exact dedP on {tot-bad}/{tot} random small systems "
              f"({'OK' if bad==0 else 'MISMATCH!'})", flush=True)
        rng = np.random.default_rng(0)
        for band in (3, 12):
            sysd = gen_xor_system(rng, 12, 17, band, force_unique=True)
            csp = sysd["csp"]; full = csp.full()
            forced, rank, ok = gf2_forced(sysd["A"], sysd["b"], 12)
            print(f"  band={band:2d} rank={rank} npar={sysd['n_par']} npin={sysd['n_pin']}", flush=True)
            for k in (3, 4, 6):
                cells, nf = factor_fixpoint_cells(csp, k)
                rec, fe, ng = recall_vs(forced, cells, 12, full)
                print(f"     L{k}: recall {rec*100:5.1f}%  solved={C.status(cells)=='solved'}  "
                      f"FE={fe}  factors={nf}", flush=True)
        return

    if args.lattice:
        print("\n=== LATTICE WIDTH SWEEP (fixed n=12, vary bandwidth=pathwidth; GF(2)=100%) ===", flush=True)
        out["width_sweep"] = lattice_width_sweep()
        print("\n=== LATTICE SIZE/RANK SWEEP (global high-width, vary n; GF(2)=100%) ===", flush=True)
        out["size_sweep"] = lattice_size_sweep()
        bad, tot = verify_gf2_equals_dedp(40)
        out["gf2_eq_dedp"] = {"bad": bad, "total": tot}
        print(f"\n[verify] GF(2)-forced == exact dedP on {tot-bad}/{tot} systems", flush=True)

    if args.organ:
        print("\n=== ORGAN DEPTH EXPERIMENT (blade R=4..24 on near-tree xor; eval low+high width) ===",
              flush=True)
        out["organ_depth"] = organ_depth_experiment()

    if args.out and out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1, default=str)
        print("\nwrote", args.out, flush=True)


if __name__ == "__main__":
    main()
