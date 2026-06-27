"""clair/datagen/difficulty.py — the reusable per-instance DIFFICULTY metric + occupancy profiler.

The curriculum audit (notes/curriculum_design.md) showed the corpus occupies SIZE (n) uniformly but
clusters at EASY on every intrinsic-hardness axis: required-lattice-level, treewidth, propagation
depth. This module makes that measurable + reusable so the build/stream mix can be BUCKET-TARGETED and
the OOD splits redefined as "one bucket past each trained max".

    difficulty(csp) -> {"level": int, "treewidth": int, "depth": int, "n": int, "d": int}

  * level     — the cheapest lattice level whose per-cell projection reaches the exact per-cell dedₚ
                (0 = per-cell AC suffices; 1 = pair/path consistency; 2 = triple/factor; 3 = richer,
                e.g. a wide affine system the arity-3 factor lattice still can't crack). This is the
                affine-wall axis: per-cell AC is complete iff level 0.
  * treewidth — min-fill upper bound on the primal (constraint) graph (exact enough at n<=12).
  * depth     — arc-consistency iterations to fixpoint (the propagation distance the organ must run).

Everything is pure clair.csp / clair.levels (no torch, no GPU).
"""
from __future__ import annotations

import itertools as it

from .. import csp as C


# --------------------------------------------------------------------------- treewidth (min-fill UB)
def primal_graph(csp: C.CSP) -> dict:
    """Adjacency of the primal graph: a clique over each constraint's scope."""
    adj = {i: set() for i in range(csp.n)}
    for sc, _ in csp.cons:
        for a, b in it.combinations(sc, 2):
            adj[a].add(b)
            adj[b].add(a)
    return adj


def treewidth_minfill(csp: C.CSP) -> int:
    """Min-fill heuristic UPPER BOUND on treewidth: repeatedly eliminate the vertex adding the fewest
    fill edges, tracking the max elimination degree. Exact enough for the n<=12 corpus."""
    adj = {i: set(v) for i, v in primal_graph(csp).items()}
    tw = 0
    remaining = set(adj)
    while remaining:
        # pick the vertex whose elimination adds the fewest fill edges
        def fill_count(v):
            nb = [u for u in adj[v] if u in remaining]
            return sum(1 for a, b in it.combinations(nb, 2) if b not in adj[a])
        v = min(remaining, key=lambda v: (fill_count(v), len(adj[v] & remaining)))
        nb = [u for u in adj[v] if u in remaining]
        tw = max(tw, len(nb))
        for a, b in it.combinations(nb, 2):          # connect the neighbours (make a clique)
            adj[a].add(b)
            adj[b].add(a)
        remaining.discard(v)
        for u in nb:
            adj[u].discard(v)
    return tw


# --------------------------------------------------------------------------- required lattice level
def _factor_cells(csp: C.CSP, k: int):
    factors = C.default_factors(csp, k)
    st = C.factor_init(csp, factors)
    for _ in range(csp.n * csp.d * csp.d + 2):
        nxt = C.factor_step(csp, st, factors)
        if nxt == st:
            break
        st = nxt
    return C.factor_cells(csp, st, factors)


def required_level(csp: C.CSP, exact=None) -> int:
    """Cheapest lattice level whose per-cell projection EQUALS the exact per-cell dedₚ (the audit's
    '%L0' axis). 0 per-cell AC; 1 pair/path-consistency; 2 triple/factor; else 3 (richer — e.g. a wide
    affine system). Sound at every level, so equality-to-exact is the completeness test."""
    full = csp.full()
    exact = exact if exact is not None else C.exact_dedP(csp, full)
    ac, _ = C.to_fixpoint(C.ac_step, csp, full)
    if ac == exact:
        return 0
    try:
        P, _ = C.pair_to_fixpoint(csp, full)
        if C.pair_cells(csp, P) == exact:
            return 1
    except Exception:
        pass
    try:
        if _factor_cells(csp, 3) == exact:
            return 2
    except Exception:
        pass
    return 3


def prop_depth(csp: C.CSP) -> int:
    """Arc-consistency iterations to fixpoint = the propagation distance (long-range = deep)."""
    _, steps = C.to_fixpoint(C.ac_step, csp, csp.full())
    return int(steps)


# --------------------------------------------------------------------------- the vector
def difficulty(csp: C.CSP) -> dict:
    """The reusable per-instance difficulty vector {level, treewidth, depth, n, d}."""
    exact = C.exact_dedP(csp, csp.full())
    return {"level": required_level(csp, exact=exact),
            "treewidth": treewidth_minfill(csp),
            "depth": prop_depth(csp),
            "n": csp.n, "d": csp.d}


# --------------------------------------------------------------------------- occupancy bucketing
LEVEL_BUCKETS = (0, 1, 2, 3)               # >=3 lumped into the "3" bucket
DEPTH_THIRDS = {"short(<=3)": (0, 3), "med(4-6)": (4, 6), "long(>=7)": (7, 999)}


def depth_bucket(depth: int) -> str:
    for name, (lo, hi) in DEPTH_THIRDS.items():
        if lo <= depth <= hi:
            return name
    return "long(>=7)"


def profile(csps, label="") -> dict:
    """Occupancy profile over a list of CSPs: per-level / per-treewidth / per-depth-third histograms,
    plus the headline %L0 and treewidth median. Returns a plain dict (JSON-safe)."""
    from collections import Counter
    levels, tws, depths = Counter(), Counter(), Counter()
    tw_vals = []
    n = 0
    for csp in csps:
        d = difficulty(csp)
        n += 1
        levels[min(d["level"], 3)] += 1
        tws[d["treewidth"]] += 1
        tw_vals.append(d["treewidth"])
        depths[depth_bucket(d["depth"])] += 1
    tw_vals.sort()
    med_tw = tw_vals[len(tw_vals) // 2] if tw_vals else 0
    return {
        "label": label, "n": n,
        "level_pct": {str(k): round(100 * levels.get(k, 0) / max(1, n), 1) for k in LEVEL_BUCKETS},
        "level_ge1_pct": round(100 * (n - levels.get(0, 0)) / max(1, n), 1),
        "treewidth_hist": {str(k): tws[k] for k in sorted(tws)},
        "treewidth_median": med_tw,
        "treewidth_ge4_pct": round(100 * sum(v for k, v in tws.items() if k >= 4) / max(1, n), 1),
        "depth_thirds_pct": {k: round(100 * depths.get(k, 0) / max(1, n), 1) for k in DEPTH_THIRDS},
    }


def print_profile(p: dict):
    print(f"\n=== difficulty occupancy: {p['label']}  (n={p['n']}) ===")
    print(f"  required-level %:  " + "  ".join(f"L{k}={v}%" for k, v in p["level_pct"].items())
          + f"   (level>=1: {p['level_ge1_pct']}%)")
    print(f"  treewidth:         median={p['treewidth_median']}  tw>=4={p['treewidth_ge4_pct']}%  "
          f"hist={p['treewidth_hist']}")
    print(f"  prop-depth thirds: " + "  ".join(f"{k}={v}%" for k, v in p["depth_thirds_pct"].items()))
