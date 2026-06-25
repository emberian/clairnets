"""Map each curriculum relation type onto the abstraction hierarchy.

The Lean LDT files prove: soundness is free; COMPLETENESS depends on the lattice LEVEL (per-cell /
pair / triple) and the constraint's polymorphism KIND (semilattice → per-cell AC solves; affine/Maltsev
→ per-cell must abstain; majority/bounded-width → some local-consistency level solves). csp.py has the
boolean version of the diagnostic; here we generalize it to the curriculum's actual domains and MEASURE,
per relation type, what each lattice level can solve. The verdict tells us which rungs a per-cell organ
can handle vs which need a richer (pair/triple) lattice — the design input for extending the organ.

Pure Python, no torch, no infra.  Run:  python -m clair.levels
"""
from __future__ import annotations

import numpy as np

from . import csp as C
from . import curriculum as CU


def polymorphism_kind(csp: C.CSP):
    """Generalized (any-domain) polymorphism probe. Returns flags for the three classes the
    bounded-width dichotomy keys on (csp.polymorphism_signature is the boolean special case):
      semilattice (min OR max is a polymorphism)  -> AC-solvable / order-Horn
      majority    (median-of-3 is a polymorphism) -> bounded width (some local level solves)
      affine      (x-y+z mod d is a polymorphism)  -> the unbounded-width wall (per-cell abstains)."""
    d = csp.d
    sl = C._is_polymorphism(csp, min, 2) or C._is_polymorphism(csp, max, 2)
    maj = C._is_polymorphism(csp, lambda a, b, c: sorted((a, b, c))[1], 3)
    aff = C._is_polymorphism(csp, lambda a, b, c: (a - b + c) % d, 3)
    return sl, maj, aff


def verdict(sl, maj, aff):
    if aff and not maj:
        return "level-2 (affine wall: per-cell AC abstains)"
    if sl:
        return "level-0 (semilattice: per-cell AC solves)"
    if maj:
        return "bounded (majority: some local-consistency level solves)"
    return "search/pair (no nice polymorphism: needs branching or richer lattice)"


def analyze(n_per=40, seed=0):
    rng = np.random.default_rng(seed)
    print(f"\nrelation     poly SL/Maj/Aff   L0(cell)  L1(pair)  L2(triple)  exact-cell   kind / verdict")
    print("-" * 104)
    for rel in CU.GENERATORS:
        tot = ac = pr = tr = ex = 0
        sl_n = maj_n = aff_n = fe = 0
        for _ in range(n_per):
            p = CU.gen_problem(rng, relation=rel)
            csp = p.csp
            if not C.solutions(csp):
                continue
            tot += 1
            s, m, a = polymorphism_kind(csp)
            sl_n += s; maj_n += m; aff_n += a
            r0 = C.solve(csp, C.ac_step)
            fe += r0["false_elim_vs_exact"]
            ac += r0["outcome"] == "solved"
            try:
                pr += C.solve_pair(csp)["outcome"] == "solved"
            except Exception:
                pass
            tr += C.solve_factor(csp, 3)["outcome"] == "solved"   # level-2 / triple factor lattice
            dom, _ = C.to_fixpoint(C.exact_dedP, csp, csp.full())
            ex += C.status(dom) == "solved"
        if not tot:
            continue
        v = verdict(sl_n > 0.8 * tot, maj_n > 0.8 * tot, aff_n > 0.8 * tot)
        print(f"{rel:11}  {sl_n*100//tot:>3}/{maj_n*100//tot:>3}/{aff_n*100//tot:>3}        "
              f"{ac*100//tot:>4}%    {pr*100//tot:>5}%     {tr*100//tot:>5}%     {ex*100//tot:>5}%      {v}")
    print("\nreads: AC-solve = per-cell arc-consistency pins a unique solution; exact-solve = the best")
    print("per-cell transformer dedP solves it; pair-solve = path-consistency. false_elim must be 0")
    print("(all levels sound). Where AC<exact, per-cell propagation is incomplete; where exact<100,")
    print("per-cell is fundamentally insufficient (multi-solution or needs pair/triple) -> richer organ.")
    print("caveat: pair-solve only ingests BINARY constraints (drops 3-cell ones), so it under-reports")
    print("on arithmetic/alldiff -- a fixable pair_init limitation, not 'pair is worse'.")


def affine_wall_demo():
    """The canonical proof the factor (level-2) lattice breaks the affine wall: XOR is uniquely
    solvable but per-cell AC abstains; the triple lattice solves it, staying sound."""
    xr = C.xor_parity()
    ac = C.solve(xr, C.ac_step)
    f3 = C.solve_factor(xr, 3)
    print("\naffine-wall demo (XOR x=y,y=z,x^y^z=0; unique solution, per-cell must abstain):")
    print(f"  L0 per-cell AC : {ac['outcome']:7} alive={ac['final_alive']} false_elim={ac['false_elim_vs_exact']}")
    print(f"  L2 triple      : {f3['outcome']:7} alive={f3['final_alive']} false_elim={f3['false_elim_vs_exact']}")
    assert ac["outcome"] == "open" and f3["outcome"] == "solved" and f3["false_elim_vs_exact"] == 0
    assert C.solve_factor(C.chain_eq(4), 3)["outcome"] == "solved"   # still solves the easy case, soundly
    print("  -> level-2 SOLVES where level-0 abstains, soundly. The richer lattice is the lever.")


if __name__ == "__main__":
    analyze()
    affine_wall_demo()
