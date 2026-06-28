"""clair/organ/faculty_difficulty.py — PER-FACULTY difficulty metrics + the CROSS-FACULTY composition-
occupancy framework (the foundation for the multi-faculty curriculum expansion).

WHY. `clair/datagen/difficulty.py` made CSP hardness MEASURABLE (level / treewidth / depth) and the audit
(`notes/curriculum_design.md`) found the CSP generators cluster at EASY — 98.7% per-cell-level-0 — so the
organ pretrains where per-cell AC already solves everything. The organ is now a MULTI-FACULTY fabric
(csp · graph · ising · type · reduction-routing + the cross/tri multi-hop flows, `clair/organ/faculty.py`)
but the curriculum is still CSP-difficulty + a handful of hand-picked cross tasks. The same coverage-hole
lesson applies per faculty AND across the composition lattice: **the model only generalizes where the
curriculum spans the multifaculty complexity.** This module generalizes the CSP difficulty metric +
occupancy profiler to EACH faculty's intrinsic hardness axes, and adds the cross-faculty
composition-occupancy framework (which faculties × chain-order × hop-depth the cross tasks cover vs the
full lattice) + the principled OOD-split definitions the expansion will target.

This is the METRICS + FRAMEWORK + AUDIT. It does NOT generate the curriculum (that needs the bank-
completion + owns `faculty_tasks.py`). Everything here is exact/cheap and READ-ONLY on the faculty code:
it imports `faculty` / `ising_organ` / `typeinfer` / `datagen.difficulty` but edits nothing.

    difficulty(struct, faculty) -> {axes...}     # per-faculty intrinsic-hardness vector
    profile_faculty(samples, faculty) -> {hist}  # the occupancy histogram per axis (the audit)
    cross_occupancy(tasks) -> {coverage}         # composition-space occupancy vs the full lattice
    OOD_SPLITS                                    # held-out faculty-combination / hop-depth / routing-trap

Run:  python -m clair.organ.faculty_difficulty        # the per-faculty + cross occupancy audit (numbers)
"""
from __future__ import annotations

import itertools as it
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .. import csp as C
from ..datagen import difficulty as CSPD             # the CSP template we generalize (treewidth/level/depth)
from . import faculty as FAC                          # READ-ONLY: state dataclasses + builders


# ============================================================================ ISING faculty difficulty
# Intrinsic hardness of a ±1 Ising / max-cut instance H = −½sᵀJs − hᵀs:
#   frustration            — fraction of bonds UNSATISFIED in a ground state (Harary frustration index;
#                            0 ⇔ the coupling graph is balanced/2-colourable). The cheap proxy is the
#                            frustrated-triangle density (a triangle with sign-product −1 can NEVER be
#                            satisfied), verified ⊆ the exact ground frustration on small n.
#   coupling_density       — |edges| / C(n,2): how connected the rivalry graph is.
#   ground_degeneracy      — #distinct ground states (brute, small n): a flat/glassy landscape is harder
#                            and makes the RELATIVE-team question well-posed only on the determined ones.
#   abs_J_spread           — std of the nonzero |J|: 0 = uniform (max-cut-clean); large = the SK spin-glass
#                            -hard regime where mean-field has many local minima.
def _ising_parts(struct):
    """Accept an IsingState | (J, h[, source, target]) | dict(J_true,h_true) → (n, J[n,n], h[n])."""
    if isinstance(struct, FAC.IsingState):
        J, h = np.asarray(struct.J, np.float64), np.asarray(struct.h, np.float64)
    elif isinstance(struct, dict):
        J, h = np.asarray(struct["J_true"], np.float64), np.asarray(struct["h_true"], np.float64)
    else:
        J = np.asarray(struct[0], np.float64)
        h = np.asarray(struct[1], np.float64) if len(struct) > 1 and struct[1] is not None \
            else np.zeros(J.shape[0])
    n = J.shape[0]
    J = 0.5 * (J + J.T)
    np.fill_diagonal(J, 0.0)
    return n, J, h


def _ising_edges(J):
    n = J.shape[0]
    return [(i, j) for i in range(n) for j in range(i + 1, n) if abs(J[i, j]) > 1e-12]


def frustrated_triangle_frac(J) -> float:
    """Fraction of fully-coupled triangles whose coupling-sign product is negative (cannot be satisfied
    by ANY spin assignment — the cheap, exact necessary witness of frustration). O(n³)."""
    n = J.shape[0]
    tot = fr = 0
    for i, j, k in it.combinations(range(n), 3):
        a, b, c = J[i, j], J[j, k], J[i, k]
        if abs(a) > 1e-12 and abs(b) > 1e-12 and abs(c) > 1e-12:
            tot += 1
            fr += int(np.sign(a) * np.sign(b) * np.sign(c) < 0)
    return fr / tot if tot else 0.0


def _all_ground_states(J, h):
    """All min-energy ±1 configs by brute enumeration (n ≤ ~18). Returns (list[np.ndarray], min_E)."""
    n = J.shape[0]
    best = None
    mins = []
    for m in range(1 << n):
        s = np.array([1.0 if (m >> i) & 1 else -1.0 for i in range(n)], dtype=np.float64)
        E = -0.5 * (s @ J @ s) - float(h @ s)
        if best is None or E < best - 1e-9:
            best = E
            mins = [s]
        elif abs(E - best) < 1e-9:
            mins.append(s)
    return mins, best


def ground_frustration(J, h, gs=None) -> float:
    """EXACT Harary frustration index: fraction of bonds (i,j) whose ground-state product J_ij·s_i·s_j < 0
    (unsatisfied) at a min-energy config. 0 ⇔ a balanced/2-colourable coupling graph (easy). Brute, small n."""
    edges = _ising_edges(J)
    if not edges:
        return 0.0
    if gs is None:
        gs, _ = _all_ground_states(J, h)
    s = gs[0]
    bad = sum(1 for (i, j) in edges if J[i, j] * s[i] * s[j] < -1e-12)
    return bad / len(edges)


def ising_difficulty(struct, brute_n=18) -> dict:
    """The Ising faculty's intrinsic-hardness vector. Brute axes (degeneracy / exact frustration) only
    when n ≤ brute_n (the generators are n≤8, so always)."""
    n, J, h = _ising_parts(struct)
    edges = _ising_edges(J)
    m = len(edges)
    absJ = np.array([abs(J[i, j]) for (i, j) in edges]) if edges else np.zeros(0)
    out = {
        "n": n,
        "coupling_density": round(m / max(1, n * (n - 1) // 2), 4),
        "frustrated_tri_frac": round(frustrated_triangle_frac(J), 4),
        "abs_J_spread": round(float(absJ.std()) if absJ.size else 0.0, 4),
    }
    if n <= brute_n:
        gs, _E = _all_ground_states(J, h)
        out["ground_degeneracy"] = len(gs)
        out["frustration"] = round(ground_frustration(J, h, gs), 4)
    else:
        out["ground_degeneracy"] = -1
        out["frustration"] = round(frustrated_triangle_frac(J), 4)   # proxy fallback
    return out


# ============================================================================ GRAPH faculty difficulty
# Intrinsic hardness of a directed reachability instance (cells = nodes):
#   reach_depth   — the min-plus / BFS closure depth from the source (= #rounds the certified relaxation
#                   must run; the long-range-propagation axis, analogous to CSP prop-depth).
#   diameter      — the directed diameter (max finite hop-distance over all ordered pairs): the global
#                   long-range requirement.
#   reach_frac    — |reachable(source)| / n: how much of the graph the answer depends on.
#   edge_density  — |edges| / (n(n−1)): sparse (deep chains) vs dense (shallow, trivially reachable).
def _graph_parts(struct):
    """Accept GraphReachState | (n, edges, source) | dict(true_edges, n, source) → (n, edges, source)."""
    if isinstance(struct, FAC.GraphReachState):
        return struct.n, frozenset(struct.edges), struct.source
    if isinstance(struct, dict):
        return int(struct["n"]), frozenset(map(tuple, struct["true_edges"])), int(struct["source"])
    n, edges, source = struct
    return int(n), frozenset(map(tuple, edges)), int(source)


def _bfs_hops(n, adj, s):
    """Hop-distance dict node→#edges from s (BFS over a directed adjacency list). Unreached omitted."""
    dist = {s: 0}
    frontier = [s]
    while frontier:
        nxt = []
        for x in frontier:
            for y in adj.get(x, ()):
                if y not in dist:
                    dist[y] = dist[x] + 1
                    nxt.append(y)
        frontier = nxt
    return dist


def graph_difficulty(struct) -> dict:
    """The graph faculty's intrinsic-hardness vector (reach-depth / diameter / reach-frac / density)."""
    n, edges, s = _graph_parts(struct)
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, []).append(v)
    src_hops = _bfs_hops(n, adj, s)
    diam = 0
    for u in range(n):
        hops = _bfs_hops(n, adj, u)
        if len(hops) > 1:
            diam = max(diam, max(hops.values()))
    return {
        "n": n,
        "reach_depth": max(src_hops.values()) if src_hops else 0,
        "diameter": diam,
        "reach_frac": round(len(src_hops) / max(1, n), 4),
        "edge_density": round(len(edges) / max(1, n * (n - 1)), 4),
    }


# ============================================================================ TYPE faculty difficulty
# Intrinsic hardness of a type-inference instance. typeinfer compiles a typed-λ expr to a CSP over a finite
# monotype universe (certified_step == csp arc-consistency), so the structural axes reuse the CSP machinery:
#   prop_depth    — AC iterations to fixpoint over the type-CSP = unification-chain depth (how many
#                   'has the same type as' hops from a pin to the query — the type faculty's propagation).
#   poly_breadth  — mean |dedP[cell]| at the exact fixpoint: 1.0 = fully monomorphic (every cell forced);
#                   >1 = residual polymorphism / free type-vars (the a→a identity family). Breadth of the
#                   still-open type space.
#   nesting_depth — max structural nesting of the types in play (0 for base int/bool; ≥1 once fun/pair
#                   appear). The richness the type universe affords but the flat eq-tree generator omits.
#   treewidth     — min-fill UB on the type-CSP primal graph (the inference-rank proxy).
def _type_csp(struct):
    """Accept a TypeState | C.CSP | (facts, n) | dict(true_facts, n) → (C.CSP, optional ast)."""
    ast = None
    if isinstance(struct, FAC.TypeState):
        return struct.csp, None
    if isinstance(struct, C.CSP):
        return struct, None
    if isinstance(struct, dict):
        facts = [tuple(f) for f in struct["true_facts"]]
        return FAC.build_type_csp(facts, int(struct["n"])), struct.get("ast")
    first, second = struct                                # (C.CSP, ast)  |  (facts, n)
    if isinstance(first, C.CSP):
        return first, second
    return FAC.build_type_csp([tuple(f) for f in first], int(second)), ast


def _type_term_depth(t) -> int:
    """Structural nesting depth of a type term, treating base types AND free type-vars as depth 0
    (a tvar is ('tvar', k) — an unresolved leaf, not a constructor)."""
    if not isinstance(t, tuple) or t[0] in ("int", "bool", "tvar"):
        return 0
    return 1 + max((_type_term_depth(x) for x in t[1:]), default=0)


def _ast_nesting(ast) -> int:
    """Max type-term nesting depth implied by an AST's inferred principal type (lam/app/pair raise depth)."""
    if ast is None:
        return 0
    from .. import typeinfer as TI
    pt, ok = TI.algorithm_w(ast)
    return _type_term_depth(pt) if (ok and pt is not None) else 0


def type_difficulty(struct) -> dict:
    """The type faculty's intrinsic-hardness vector (prop-depth / poly-breadth / nesting / treewidth)."""
    csp, ast = _type_csp(struct)
    _, steps = C.to_fixpoint(C.ac_step, csp, csp.full())
    ded = C.exact_dedP(csp, csp.full())
    breadth = float(np.mean([len(ded[i]) for i in range(csp.n)])) if csp.n else 1.0
    return {
        "n": csp.n,
        "prop_depth": int(steps),
        "poly_breadth": round(breadth, 3),
        "nesting_depth": _ast_nesting(ast),
        "treewidth": CSPD.treewidth_minfill(csp),
    }


# ============================================================================ the dispatch
def difficulty(struct, faculty: str) -> dict:
    """Per-faculty intrinsic-difficulty vector, dispatched by `faculty`. `struct` is the faculty-native
    structure (a State dataclass, a raw tuple, a C.CSP, or a faculty_tasks record dict).
        csp   → datagen.difficulty.difficulty (level / treewidth / depth — reused unchanged)
        ising → ising_difficulty (frustration / coupling-density / degeneracy / |J|-spread)
        graph → graph_difficulty (reach-depth / diameter / reach-frac / density)
        type  → type_difficulty  (prop-depth / poly-breadth / nesting / treewidth)
    The composite faculties (cross/tri/reduction) decompose into their per-faculty parts via the
    composition framework below — their per-stage hardness is measured on each stage's native struct."""
    if isinstance(struct, dict) and faculty is None:
        faculty = struct.get("faculty")
    if faculty == "csp":
        csp = struct["csp"] if isinstance(struct, dict) else struct
        return CSPD.difficulty(csp)
    if faculty == "ising":
        return ising_difficulty(struct)
    if faculty == "graph":
        return graph_difficulty(struct)
    if faculty == "type":
        return type_difficulty(struct)
    raise ValueError(f"no intrinsic difficulty metric for faculty {faculty!r} "
                     f"(composite faculties decompose via the composition framework)")


# ============================================================================ per-faculty occupancy buckets
# Bucketers per axis, mirroring CSPD.LEVEL_BUCKETS / DEPTH_THIRDS. Each maps a difficulty vector → a label
# per axis so `profile_faculty` can histogram. The thresholds split easy / medium / hard for the audit.
def _band(x, edges, labels):
    for hi, lab in zip(edges, labels):
        if x <= hi:
            return lab
    return labels[-1]


FACULTY_AXES = {
    "ising": {
        "frustration":  lambda d: _band(d["frustration"], [0.0, 0.15, 0.35], ["0", "lo(<=.15)", "mid(<=.35)", "hi"]),
        "coupling_density": lambda d: _band(d["coupling_density"], [0.33, 0.66], ["sparse", "med", "dense"]),
        "ground_degeneracy": lambda d: _band(d["ground_degeneracy"], [2, 4, 16], ["unique(<=2)", "few(<=4)", "many(<=16)", "glassy"]),
        "abs_J_spread": lambda d: ("uniform" if d["abs_J_spread"] < 1e-9 else "spread"),
    },
    "graph": {
        "reach_depth":  lambda d: _band(d["reach_depth"], [1, 2, 3], ["1", "2", "3", "deep(>=4)"]),
        "diameter":     lambda d: _band(d["diameter"], [1, 2, 3], ["1", "2", "3", "deep(>=4)"]),
        "reach_frac":   lambda d: _band(d["reach_frac"], [0.33, 0.66], ["lo", "med", "hi"]),
        "edge_density": lambda d: _band(d["edge_density"], [0.2, 0.4], ["sparse", "med", "dense"]),
    },
    "type": {
        "prop_depth":   lambda d: _band(d["prop_depth"], [2, 4], ["short(<=2)", "med(3-4)", "long(>=5)"]),
        "poly_breadth": lambda d: ("mono" if d["poly_breadth"] <= 1.0 + 1e-9 else "poly"),
        "nesting_depth": lambda d: _band(d["nesting_depth"], [0, 1], ["base(0)", "depth1", "depth>=2"]),
        "treewidth":    lambda d: _band(d["treewidth"], [1, 2, 3], ["1", "2", "3", ">=4"]),
    },
}


def profile_faculty(samples, faculty: str, label="") -> dict:
    """Occupancy profile over a list of faculty-native structs (or records): the per-axis histogram of
    difficulty buckets, like CSPD.profile found 98.7% level-0 for CSP. Returns a JSON-safe dict. For csp
    it delegates to the CSP profiler; for ising/graph/type it histograms each FACULTY_AXES bucketer."""
    if faculty == "csp":
        csps = [(s["csp"] if isinstance(s, dict) else s) for s in samples]
        return CSPD.profile(csps, label=label or "csp")
    axes = FACULTY_AXES[faculty]
    hist = {ax: Counter() for ax in axes}
    raw = []
    n = 0
    for s in samples:
        d = difficulty(s, faculty)
        raw.append(d)
        n += 1
        for ax, fn in axes.items():
            hist[ax][fn(d)] += 1
    pct = {ax: {k: round(100 * c / max(1, n), 1) for k, c in sorted(h.items())} for ax, h in hist.items()}
    return {"label": label or faculty, "faculty": faculty, "n": n, "axis_pct": pct}


def print_faculty_profile(p: dict):
    if "axis_pct" not in p:                                # a CSP profile (CSPD schema)
        CSPD.print_profile(p)
        return
    print(f"\n=== {p['faculty']} occupancy: {p['label']}  (n={p['n']}) ===")
    for ax, h in p["axis_pct"].items():
        print(f"  {ax:18s} " + "  ".join(f"{k}={v}%" for k, v in h.items()))


# ============================================================================ CROSS-FACULTY composition space
# A composite task is a CHAIN of faculties with FLOWS between them (faculty.py: cross = csp→graph,
# tri = csp→graph→ising, reduction = a Karp route mis→ising). The composition space is combinatorial in
# (which faculties × chain-order × hop-depth × reduction-on-path), so we define PRINCIPLED BUCKETS to
# occupy rather than enumerate the whole product.
BASE_FACULTIES = ("csp", "graph", "ising", "type")        # the direct-solver faculties a flow can land on


@dataclass(frozen=True)
class CompositionBucket:
    """A point in the composition lattice: the ordered faculty chain, its hop-depth (#inter-faculty flows
    = len(chain)−1), and whether a Karp REDUCTION edge is on the path (a route, not a direct flow)."""
    chain: tuple                  # e.g. ("csp", "graph", "ising")
    hop_depth: int                # len(chain) - 1
    has_reduction: bool = False

    @property
    def faculty_set(self) -> frozenset:
        return frozenset(self.chain)


# How each LIVE faculty tag (faculty.LIVE_FACULTIES) maps into the composition lattice. The single
# faculties are hop-0; cross/tri are the explicit multi-hop flows; reduction is a hop-1 Karp route.
LIVE_COMPOSITION = {
    "csp":       CompositionBucket(("csp",), 0),
    "graph":     CompositionBucket(("graph",), 0),
    "ising":     CompositionBucket(("ising",), 0),
    "type":      CompositionBucket(("type",), 0),
    "cross":     CompositionBucket(("csp", "graph"), 1),
    "tri":       CompositionBucket(("csp", "graph", "ising"), 2),
    "reduction": CompositionBucket(("graph", "ising"), 1, has_reduction=True),   # mis→ising route
}


def task_bucket(task) -> CompositionBucket:
    """A record dict (with 'faculty') or a faculty-tag string → its CompositionBucket."""
    fac = task.get("faculty") if isinstance(task, dict) else task
    if fac in LIVE_COMPOSITION:
        return LIVE_COMPOSITION[fac]
    raise ValueError(f"unknown faculty tag {fac!r}")


def full_lattice(max_hop=3, base=BASE_FACULTIES) -> list:
    """The PRINCIPLED target composition lattice the expansion should occupy: every ordered chain over the
    base faculties up to `max_hop` flows, with NO immediate repeat (a flow A→A is a no-op). hop-0 = the 4
    singletons; hop-1 = directed pairs; hop-2 = 3-chains; hop-3 = 4-chains. (The reduction-on-path axis is
    orthogonal — a route can substitute for any direct flow — so we track it as a separate occupancy bit,
    not a Cartesian blow-up of this list.)"""
    buckets = []
    for L in range(1, max_hop + 2):
        for chain in it.product(base, repeat=L):
            if any(chain[i] == chain[i + 1] for i in range(L - 1)):
                continue
            buckets.append(CompositionBucket(tuple(chain), L - 1))
    return buckets


def cross_occupancy(tasks, max_hop=3) -> dict:
    """Profile which composition buckets the CURRENT tasks cover vs the full lattice. `tasks` is a list of
    records (or faculty-tag strings). Reports occupancy by hop-depth, by unordered faculty-set, the
    reduction-on-path share, and the SPECIFIC ordered chains covered vs missing — the cross-faculty
    coverage audit the expansion targets (predict: hop-0 + a few hop-1/2, the rest of the lattice empty)."""
    seen_chains = Counter()
    seen_sets = Counter()
    hop_hist = Counter()
    red_share = 0
    n = 0
    for t in tasks:
        b = task_bucket(t)
        n += 1
        seen_chains[b.chain] += 1
        seen_sets[tuple(sorted(b.faculty_set))] += 1
        hop_hist[b.hop_depth] += 1
        red_share += int(b.has_reduction)

    lattice = full_lattice(max_hop)
    by_hop_total = Counter(b.hop_depth for b in lattice)
    by_hop_seen = Counter()
    for chain in seen_chains:
        if any(chain[i] == chain[i + 1] for i in range(len(chain) - 1)):
            continue
        if len(chain) - 1 <= max_hop:
            by_hop_seen[len(chain) - 1] += 1
    covered = {b.chain for b in lattice if b.chain in seen_chains}
    missing = [b.chain for b in lattice if b.chain not in seen_chains]
    return {
        "n_tasks": n,
        "distinct_chains_covered": len(covered),
        "lattice_size": len(lattice),
        "coverage_frac": round(len(covered) / max(1, len(lattice)), 3),
        "by_hop_depth": {str(k): {"covered": by_hop_seen.get(k, 0), "total": by_hop_total[k]}
                         for k in sorted(by_hop_total)},
        "task_hop_hist": {str(k): hop_hist[k] for k in sorted(hop_hist)},
        "faculty_set_hist": {"·".join(k): v for k, v in sorted(seen_sets.items())},
        "reduction_on_path_frac": round(red_share / max(1, n), 3),
        "covered_chains": sorted("→".join(c) for c in covered),
        "missing_chains_sample": ["→".join(c) for c in missing[:24]],
    }


# ============================================================================ OOD-split definitions
# The cross-faculty OOD axis is the COMBINATION, not the size (notes/multifaculty_overhaul §4.4). Each split
# is a predicate over a CompositionBucket: train on its complement, evaluate on the held-out region. These
# are the targets the expansion's generators are gated to fill (train side) + probe (test side).
def _held_out_faculty_combination(b: CompositionBucket) -> bool:
    """Held-out faculty-COMBINATION: a flow-pair never co-trained. Train {csp·graph, csp·ising (in tri)},
    hold out {type·*} and {graph·ising} as a DIRECT pair — a faculty set the cross tasks never span."""
    fs = b.faculty_set
    return ("type" in fs and len(fs) >= 2) or (fs == frozenset({"graph", "ising"}) and not b.has_reduction)


def _held_out_hop_depth(b: CompositionBucket) -> bool:
    """Held-out hop-DEPTH: train hop ≤ 2 (single / cross / tri), extrapolate to hop ≥ 3 (the 4-faculty
    chain, e.g. type→csp→graph→ising) — does emergent wiring compose one flow deeper than trained?"""
    return b.hop_depth >= 3


def _routing_trap(b: CompositionBucket) -> bool:
    """Routing-TRAP: the answer needs a Karp REDUCTION edge on the path (no direct faculty), with the
    obvious single-faculty/greedy route giving a confident WRONG answer. Train the direct routes, hold
    out the reduction-routed ones (sat3→CSP trained vs sat3→MIS→Ising held out)."""
    return b.has_reduction


OOD_SPLITS = {
    "held_out_faculty_combination": _held_out_faculty_combination,
    "held_out_hop_depth": _held_out_hop_depth,
    "routing_trap": _routing_trap,
}


def classify_ood(tasks) -> dict:
    """How the current tasks fall across the OOD-split predicates (which held-out regions are ALREADY
    occupied by train — they should be ~empty if the splits are clean)."""
    out = {name: 0 for name in OOD_SPLITS}
    n = 0
    for t in tasks:
        b = task_bucket(t)
        n += 1
        for name, pred in OOD_SPLITS.items():
            out[name] += int(pred(b))
    return {"n_tasks": n, "held_out_occupancy": {k: f"{v}/{n}" for k, v in out.items()}}


# ============================================================================ ranked generator specs
# The concrete difficulty-controlled generators the expansion should build (per-faculty + cross-faculty
# chains), ranked by leverage × cheapness, each gated on the bank-completion landing. Surfaced as data so
# the expansion (which owns faculty_tasks.py) can consume the spec list directly.
GENERATOR_SPECS = [
    {"rank": 1, "faculty": "ising", "name": "gen_ising_frustrated(target_frustration)",
     "axis": "frustration", "gated_on": "ising bank (live)",
     "spec": "plant a target Harary frustration index: seed a balanced (2-colourable) base then add k "
             "frustrating odd-cycle bonds; reject-sample on ground_frustration. Fills the 0%→hard "
             "frustration tail the current uniform max-cut generator misses."},
    {"rank": 2, "faculty": "ising", "name": "gen_ising_glass(abs_J_spread, density)",
     "axis": "abs_J_spread × coupling_density", "gated_on": "ising bank (live)",
     "spec": "draw SK-style |J|~LogNormal(spread) on a density-controlled graph → the spin-glass-hard "
             "regime (many local minima, high degeneracy). Targets the |J|-spread + density axes jointly."},
    {"rank": 3, "faculty": "graph", "name": "gen_graph_depth(reach_depth=D)",
     "axis": "reach_depth / diameter", "gated_on": "graph faculty (live)",
     "spec": "build a layered DAG of depth D from the source (a reachability chain) + decoy off-path "
             "components, so the answer needs D closure rounds. Lifts reach-depth from the current "
             "shallow p_edge sample to a uniform short/med/deep occupancy (the R-budget earns its keep)."},
    {"rank": 4, "faculty": "type", "name": "gen_type_nested(nesting_depth=k, poly_breadth)",
     "axis": "nesting_depth × poly_breadth", "gated_on": "type faculty (live)",
     "spec": "use typeinfer.gen_typed at universe max_depth≥1 (fun/pair) so inferred types NEST, and "
             "tune the free-tvar count for poly-breadth. The current eq-tree generator is base-only "
             "(nesting 0, mono) — this spans the type universe's actual richness."},
    {"rank": 5, "faculty": "cross", "name": "gen_chain(faculty_chain, hop_depth)",
     "axis": "composition hop-depth / faculty-combo", "gated_on": "bank-completion (all faculties)",
     "spec": "a generic cross-faculty chain factory parametric in the ordered faculty chain (the §4.3 "
             "design): each stage's solution determines the next stage's structure. Fills the empty "
             "hop-1/hop-2 buckets BEYOND the two hand-picked (cross, tri) and enables the held-out-combo "
             "+ held-out-hop OOD splits."},
    {"rank": 6, "faculty": "reduction", "name": "gen_karp_route(src_type, held_out_edge)",
     "axis": "routing-trap", "gated_on": "reductions hub (organ/reductions ALL_EDGES)",
     "spec": "emit instances whose only low-cost route crosses a held-out Karp edge (sat3→MIS→Ising vs "
             "the trained sat3→CSP); the greedy single-faculty answer is a confident wrong. Trains the "
             "router + flow gates on routing, exercising the routing-trap OOD split."},
]


# ============================================================================ the audit (numbers for the notes)
def _ising_samples(rng, n_inst):
    from . import faculty_tasks as FT
    return [FT.gen_ising_record(rng, want_yes=(k % 2 == 0)) for k in range(n_inst)]


def _graph_samples(rng, n_inst):
    from . import faculty_tasks as FT
    return [FT.gen_graph_record(rng) for _ in range(n_inst)]


def _type_samples(rng, n_inst):
    from . import faculty_tasks as FT
    return [FT.gen_type_record(rng) for _ in range(n_inst)]


def _type_rich_samples(seed, n_inst):
    """The TYPE faculty's OWN rich generator (typeinfer.rand_problem, fun/pair universe) — the contrast
    that shows how much intrinsic type-complexity the flat eq-tree multifaculty generator leaves unspanned."""
    import random
    from .. import typeinfer as TI
    U = TI.Universe(max_depth=1, ctors=("fun", "pair"))
    r = random.Random(seed)
    out = []
    for _ in range(n_inst):
        comp = TI.rand_problem(r, U, budget=3)
        out.append((comp.csp, comp.ast))                 # (C.CSP, ast) — type_difficulty reads both
    return out


def _csp_samples(rng_seed, n_inst):
    """CSP faculty records as the multifaculty pool builds them (G.build_live_pool coloring/equality,
    determined). Returns the carried C.CSP per record."""
    import numpy as _np
    from .. import run_glados_staged as G
    rng = _np.random.default_rng(rng_seed)
    pool = G.build_live_pool(rng, ["coloring", "equality"], "id", max(2, n_inst // 2),
                             det_only=True, two_stream=False)
    return [r["csp"] for r in pool[:n_inst]]


def run_audit(n_inst=150, seed=0, verbose=True):
    """The per-faculty coverage audit (does each faculty's generator cluster easy, like CSP did?) + the
    cross-faculty occupancy. Returns a results dict; prints the numbers for notes/multifaculty_curriculum.md."""
    rng = np.random.default_rng(seed)
    profiles = {}

    # csp — reuse the CSP profiler (the 98.7%-L0 baseline, restated here for the multifaculty pool)
    profiles["csp"] = profile_faculty(_csp_samples(seed, n_inst), "csp", label="multifaculty csp pool")
    # ising / graph / type — the new per-faculty profilers on the live generators
    profiles["ising"] = profile_faculty(_ising_samples(rng, n_inst), "ising", label="gen_ising_record")
    profiles["graph"] = profile_faculty(_graph_samples(rng, n_inst), "graph", label="gen_graph_record")
    profiles["type"] = profile_faculty(_type_samples(rng, n_inst), "type", label="gen_type_record (eq-tree)")
    profiles["type_rich"] = profile_faculty(_type_rich_samples(seed, n_inst), "type",
                                            label="typeinfer.rand_problem (fun/pair — the faculty's reach)")

    # cross-faculty occupancy: the live faculties as a 'task mix' (one of each kind the pool ships)
    from .faculty import LIVE_FACULTIES
    live_tasks = list(LIVE_FACULTIES)
    cross = cross_occupancy(live_tasks)
    ood = classify_ood(live_tasks)

    if verbose:
        print("================ PER-FACULTY DIFFICULTY OCCUPANCY AUDIT ================")
        for key in ("csp", "ising", "graph", "type", "type_rich"):
            print_faculty_profile(profiles[key])
        print("\n================ CROSS-FACULTY COMPOSITION OCCUPANCY ================")
        print(f"  live faculty tags: {live_tasks}")
        print(f"  distinct ordered chains covered: {cross['distinct_chains_covered']} / "
              f"{cross['lattice_size']} lattice  ({cross['coverage_frac']*100:.1f}%)")
        print(f"  by hop-depth (covered/total): {cross['by_hop_depth']}")
        print(f"  faculty-set occupancy: {cross['faculty_set_hist']}")
        print(f"  reduction-on-path share: {cross['reduction_on_path_frac']*100:.0f}%")
        print(f"  covered chains: {cross['covered_chains']}")
        print(f"  MISSING (sample): {cross['missing_chains_sample']}")
        print(f"\n  OOD held-out occupancy (should be sparse): {ood['held_out_occupancy']}")
        print("\n================ RANKED GENERATOR SPECS (gated on bank-completion) ================")
        for g in GENERATOR_SPECS:
            print(f"  [{g['rank']}] {g['faculty']:9s} {g['name']}  — axis: {g['axis']}")

    return {"profiles": profiles, "cross_occupancy": cross, "ood": ood, "generator_specs": GENERATOR_SPECS}


# ============================================================================ self-check / verification
def _verify(seed=0):
    """Cheap correctness checks: the ising triangle-frustration proxy is a SOUND ⊆ of the exact ground
    frustration (a frustrated triangle ⇒ ground_frustration>0), and the metrics RUN on real instances."""
    from . import faculty_tasks as FT
    rng = np.random.default_rng(seed)
    # (1) frustrated triangle ⇒ exact ground frustration > 0 (the brute reference), over random instances
    checked = witnessed = 0
    for _ in range(60):
        n = int(rng.integers(4, 8))
        J = rng.standard_normal((n, n)).astype(np.float64)
        J = np.triu(J, 1); J = J + J.T
        J = np.where(rng.random((n, n)) < 0.5, J, 0.0); J = np.triu(J, 1); J = J + J.T
        np.fill_diagonal(J, 0.0)
        h = np.zeros(n)
        ftf = frustrated_triangle_frac(J)
        gf = ground_frustration(J, h)
        checked += 1
        if ftf > 0:
            witnessed += 1
            assert gf > 1e-9, "a frustrated triangle must force ground_frustration > 0 (soundness)"
    # (2) every metric runs on a live instance of its faculty
    di = ising_difficulty(FT.gen_ising_record(rng))
    dg = graph_difficulty(FT.gen_graph_record(rng))
    dt = type_difficulty(FT.gen_type_record(rng))
    assert {"frustration", "coupling_density", "ground_degeneracy", "abs_J_spread"} <= set(di)
    assert {"reach_depth", "diameter", "reach_frac", "edge_density"} <= set(dg)
    assert {"prop_depth", "poly_breadth", "nesting_depth", "treewidth"} <= set(dt)
    # (3) the composition framework round-trips + the lattice is well-formed
    assert task_bucket("tri").hop_depth == 2 and task_bucket("cross").hop_depth == 1
    assert all(b.hop_depth == len(b.chain) - 1 for b in full_lattice())
    return {"frustration_checks": checked, "frustrated_witnessed": witnessed,
            "ising_ok": di, "graph_ok": dg, "type_ok": dt}


if __name__ == "__main__":
    v = _verify()
    print(f"[verify] frustration proxy ⊆ exact on {v['frustration_checks']} instances "
          f"({v['frustrated_witnessed']} frustrated witnessed); metrics run on live instances. OK\n")
    run_audit()
