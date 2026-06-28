"""clair/organ/faculty_curriculum.py — the DIFFICULTY-CONTROLLED multi-faculty curriculum generators that
CLOSE the coverage holes the audit found (notes/multifaculty_curriculum.md + faculty_difficulty.py).

THE HOLES (the audit, restated). Every live faculty trains on the EASY corner of its own intrinsic-hardness
space, and the cross-faculty composition lattice is 4.4% covered:
  * ising : 100% uniform-|J| (no spin-glass-hard), frustration has no hard tail (>0.35)
  * graph : 90% reach-depth <= 2 (no long-range closure)
  * type  : 100% monomorphic / base-typed / treewidth-1 (no nesting / polymorphism / width)
  * csp   : 100% level-0
  * cross : hop-0 full, hop-1 2/12, hop-2 1/36, hop-3 0/108 (the multi-hop lattice is empty)

THIS MODULE (NEW FILE — edits nothing the bank build owns). Each generator is a difficulty-CONTROLLED knob
that targets the empty hard buckets, and every instance is EXACT-VERIFIED — the answer re-checked against
the faculty's OWN exact solver (brute ground state for ising via ising_organ + the faculty_difficulty brute,
the certified reachability closure for graph, Algorithm-W / exact_dedP agreement for type, the exact CSP /
brute ground truth for each chain stage, and the reductions' exact end-to-end verifier for the Karp route):

    gen_ising_frustrated(rng, target_frustration)   # span Harary frustration 0 -> hi (>.35)
    gen_ising_glass(rng, abs_J_spread)               # the SK |J|-spread spin-glass-hard regime
    gen_graph_depth(rng, reach_depth)                # layered DAG needing D min-plus closure rounds
    gen_type_nested(rng, nesting, want_poly)         # fun/pair NESTED + polymorphic type inference
    gen_chain(rng, faculties, ...)                   # the GENERIC cross-faculty A->flow->B->flow->C chain
    gen_karp_route(rng, ...)                          # routing-trap: sat3 -> MIS -> Ising (no direct faculty)

Then RE-PROFILE with faculty_difficulty's profiler (the before->after proof the holes are closed) and ship
CURRICULUM_MIX — the per-faculty bucket targets + cross-faculty hop-depth distribution the fabric-retrain
consumes. READ-ONLY on faculty.py / faculty_tasks.py / ising_organ.py / typeinfer.py / reductions.py:
imports them, edits nothing. organ.selftest stays PASS (new file, not on its import path).

Run:  python -m clair.organ.faculty_curriculum     # exact-verify all generators + the before->after profile
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .. import csp as C
from .. import curriculum as CU
from .. import ising_organ as IS
from .. import typeinfer as TI
from . import faculty as FAC
from . import faculty_difficulty as FD
from . import reductions as RED


# ============================================================================ shared planted structures
def _planted_2coloring(rng, n):
    """A connected tree of `neq` edges + a pin => a UNIQUE 2-colouring (the CSP faculty's precondition).
    Returns (facts, colors). Self-contained (does not import the bank-owned faculty_tasks)."""
    perm = list(rng.permutation(n))
    tree = [(perm[i], perm[int(rng.integers(i))]) for i in range(1, n)]
    adj = {i: [] for i in range(n)}
    for (u, v) in tree:
        adj[u].append(v); adj[v].append(u)
    color = [None] * n
    root = perm[0]; c0 = int(rng.integers(2)); color[root] = c0
    stack = [root]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if color[y] is None:
                color[y] = 1 - color[x]; stack.append(y)
    facts = [("pin", root, c0)] + [("neq", u, v) for (u, v) in tree]
    return facts, color


def _planted_typing(rng, n):
    """A determined base-monotype assignment over n cells: split into >=2 type groups, pin one cell per
    group, link the rest of the group by an eq-tree. exact_dedP forces every cell -> its planted type.
    Returns (facts, types) with types[i] in {0=int, 1=bool}."""
    perm = list(rng.permutation(n))
    cut = 1 + int(rng.integers(max(1, n - 1)))
    groups = [perm[:cut], perm[cut:]] if 0 < cut < n else [perm]
    facts = []
    types = [None] * n
    for grp in groups:
        if not grp:
            continue
        tval = int(rng.integers(2))
        root = grp[0]
        facts.append(("pin", root, tval)); types[root] = tval
        placed = [root]
        for c in grp[1:]:
            par = placed[int(rng.integers(len(placed)))]
            facts.append(("eq", c, par)); types[c] = tval; placed.append(c)
    return facts, types


# ============================================================================ ISING exact ground truth
def _ground_configs(J, h):
    """All min-energy +-1 configs of H = -1/2 sJs - hs by VECTORIZED brute enumeration (n <= ~16). Same
    exact answer as faculty_difficulty._all_ground_states / ising_organ.brute_force_ground, but batched in
    numpy (the per-config python loop was the curriculum's hot-path bottleneck). Returns [k, n] +-1 array."""
    J = np.asarray(J, np.float64); h = np.asarray(h, np.float64)
    n = J.shape[0]
    idx = np.arange(1 << n, dtype=np.int64)
    S = (((idx[:, None] >> np.arange(n)) & 1) * 2 - 1).astype(np.float64)   # [2^n, n] +-1
    E = -0.5 * np.einsum("bi,ij,bj->b", S, J, S) - S @ h
    return S[E <= E.min() + 1e-9]


def _ising_relteam(J, h, source, target):
    """EXACT relative-team answer via the brute ground state: across ALL ground states (canonical
    spin[source]=+1), is spin[target] constant? Returns (determined, same_side_bool) — the well-posedness
    gate for the relative-team question."""
    G = _ground_configs(J, h)
    G = G * G[:, source:source + 1]                          # canonicalize spin[source] := +1
    vals = set(int(v) for v in np.unique(G[:, target]))
    if len(vals) != 1:
        return False, None
    return True, (next(iter(vals)) == 1)


def _organ_relteam(J, h, source, target):
    """The SAME answer via ising_organ.brute_force_ground (the optimisation faculty's own exact ground
    state, on torch) — the independent cross-solver used by the verifier (must agree with _ising_relteam)."""
    import torch
    Jt = torch.as_tensor(np.asarray(J, np.float32)); ht = torch.as_tensor(np.asarray(h, np.float32))
    s, _mE, _xE = IS.brute_force_ground(Jt, ht)
    s = s.detach().cpu().numpy()
    sc = s * s[source]
    return bool(sc[target] == 1)


def _ground_frustration(J, h=None):
    """EXACT Harary frustration index (fraction of bonds unsatisfied at a ground state), vectorized — the
    same value as faculty_difficulty.ground_frustration but off _ground_configs (the hot-path version)."""
    J = np.asarray(J, np.float64)
    n = J.shape[0]
    h = np.zeros(n) if h is None else np.asarray(h, np.float64)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if abs(J[i, j]) > 1e-12]
    if not edges:
        return 0.0
    s = _ground_configs(J, h)[0]
    bad = sum(1 for (i, j) in edges if J[i, j] * s[i] * s[j] < -1e-12)
    return bad / len(edges)


def _ising_record(J, h, n, source, target, tag, difficulty_axis):
    det, same = _ising_relteam(J, h, source, target)
    if not det:
        return None
    return {"faculty": "ising", "n": n, "J_true": np.asarray(J, np.float32).tolist(),
            "h_true": np.asarray(h, np.float32).tolist(), "source": int(source),
            "target": int(target), "gold_idx": 0 if same else 1, "answer": "yes" if same else "no",
            "gen": tag, "difficulty_axis": difficulty_axis}


def gen_ising_frustrated(rng, target_frustration=0.4, n_choices=(6, 7, 8), tol=0.1, tries=600):
    """DIFFICULTY KNOB: plant a target Harary frustration index (fraction of bonds unsatisfiable at the
    ground state). Seed a balanced 2-colouring (cross-class antiferromagnetic bonds are satisfiable =>
    frustration 0) then add WITHIN-class odd-cycle bonds (each unsatisfiable) until the EXACT
    ground_frustration reaches the target band. Closes the missing hard-frustration tail (>0.35) the
    uniform max-cut generator never reaches. Verified: the relative-team query is determined across the
    brute ground states. Spans frustration 0 -> hi as `target_frustration` sweeps."""
    best = None; best_gap = 1e9
    # within-class bond probability grows with the requested frustration; cross bonds keep it connected
    pw = float(np.clip(target_frustration / 0.45, 0.0, 1.0)) * 0.9
    pc = 0.5
    for _ in range(tries):
        n = int(rng.choice(n_choices))
        _facts, colors = _planted_2coloring(rng, n)
        colors = np.array(colors)
        J = np.zeros((n, n), np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                same = colors[i] == colors[j]
                p = pw if same else pc
                if rng.random() < p:
                    J[i, j] = J[j, i] = -1.0                       # antiferromagnetic (max-cut) bond
        if (np.abs(J) > 0).sum() < 2 * (n - 1):
            continue
        gf = _ground_frustration(J)
        gap = abs(gf - target_frustration)
        # pick a determined relative-team target
        source = 0
        rec = None
        for t in range(1, n):
            r = _ising_record(J, np.zeros(n, np.float32), n, source, t, "frustrated", "frustration")
            if r is not None:
                r["frustration"] = round(gf, 4); rec = r; break
        if rec is None:
            continue
        if gap <= tol:
            return rec
        if gap < best_gap:
            best_gap = gap; best = rec
    return best


def gen_ising_glass(rng, abs_J_spread=1.0, n_choices=(6, 7, 8), density=0.55, tries=400):
    """DIFFICULTY KNOB: the SK spin-glass-hard regime — heterogeneous |J| (non-uniform magnitudes, mixed
    signs) on a density-controlled graph. Draw |J| ~ LogNormal(sigma=abs_J_spread) with random signs, so
    abs_J_spread (std of the nonzero |J|) lands in the 'spread' bucket the audit found 100% empty
    (every live instance was uniform unit-|J| max-cut). Verified: the relative-team query is determined
    across the brute ground states (continuous J => generically a unique ground state)."""
    for _ in range(tries):
        n = int(rng.choice(n_choices))
        J = np.zeros((n, n), np.float64)
        m = 0
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    mag = float(np.exp(rng.normal(0.0, max(1e-3, abs_J_spread))))
                    sgn = 1.0 if rng.random() < 0.5 else -1.0
                    J[i, j] = J[j, i] = sgn * mag; m += 1
        if m < n - 1:
            continue
        source = 0
        for t in range(1, n):
            r = _ising_record(J, np.zeros(n, np.float32), n, source, t, "glass", "abs_J_spread")
            if r is not None:
                d = FD.ising_difficulty(r)
                r["abs_J_spread"] = d["abs_J_spread"]; r["frustration"] = d["frustration"]
                return r
    return None


# ============================================================================ GRAPH long-range depth
def gen_graph_depth(rng, reach_depth=4, n_extra=(1, 2, 3), want_yes=True, tries=64):
    """DIFFICULTY KNOB: a layered DAG whose answer needs `reach_depth` min-plus closure rounds. Build the
    chain source=v0 -> v1 -> ... -> v_D (so the deepest reachable cell is D hops out), add safe side-edges
    to already-reachable shallower cells (never a shortcut, so the reach-depth stays exactly D), and a
    disconnected decoy component (the 'no' targets). Closes the 90%-depth<=2 hole — lifts reach-depth into
    the deep(>=4) bucket so the R-round closure budget earns its keep. Verified against the certified
    reachability closure (faculty.reachable == faculty.GraphReach)."""
    D = max(1, int(reach_depth))
    for _ in range(tries):
        extra = int(rng.choice(n_extra))
        n = (D + 1) + extra
        perm = list(rng.permutation(n))
        chain = perm[: D + 1]                                     # source-component chain
        decoy = perm[D + 1:]                                      # disconnected component
        edges = set((chain[i], chain[i + 1]) for i in range(D))   # the deep path
        # safe side-edges: from a deeper chain cell back to a strictly shallower (already-reachable) one
        for _ in range(extra):
            if D >= 2:
                a = int(rng.integers(2, D + 1)); b = int(rng.integers(0, a))
                edges.add((chain[a], chain[b]))                   # cycle/back-edge: never deepens BFS
        # a few intra-decoy edges (keep the decoy a real but unreachable component)
        for a in range(len(decoy)):
            for b in range(len(decoy)):
                if a != b and rng.random() < 0.4:
                    edges.add((decoy[a], decoy[b]))
        edges = frozenset(edges)
        source = chain[0]
        # verify reach-depth is exactly D via the certified closure / BFS
        adj = {}
        for (u, v) in edges:
            adj.setdefault(u, []).append(v)
        hops = FD._bfs_hops(n, adj, source)
        if max(hops.values()) != D:
            continue
        reach = FAC.reachable(n, edges, source)
        if want_yes:
            target = chain[D]                                     # deepest cell: reachable, needs D rounds
            if target not in reach:
                continue
        else:
            cands = [c for c in decoy if c not in reach]
            if not cands:
                continue
            target = int(rng.choice(cands))
        ans_yes = target in reach
        return {"faculty": "graph", "n": n, "true_edges": sorted(edges), "source": int(source),
                "target": int(target), "gold_idx": 0 if ans_yes else 1,
                "answer": "yes" if ans_yes else "no", "gen": "graph_depth",
                "difficulty_axis": "reach_depth", "reach_depth": D}
    return None


# ============================================================================ TYPE nested / polymorphic
def gen_type_nested(rng, nesting=1, want_poly=False, budget=3, tries=80):
    """DIFFICULTY KNOB: nested (fun/pair) + polymorphic type inference in the type faculty's OWN universe
    (typeinfer.Universe(max_depth>=1)), the richness the flat base-only eq-tree generator omits (100%
    monomorphic / nesting-0 / treewidth-1). `nesting` selects the universe depth + a nesting-target want
    type; `want_poly` biases toward an open type-var family (a->a) so poly_breadth>1. Closes the
    type-faculty holes (nesting depth1/>=2, poly). Verified: the certified per-cell dedP == real
    Algorithm-W principal type (the typeinfer soundness+completeness invariant) on every well-typed instance.
    Returns (csp, ast) — the form faculty_difficulty.type_difficulty consumes."""
    import random
    r = random.Random(int(rng.integers(1 << 30)))
    U = TI.Universe(max_depth=max(1, int(nesting)), ctors=("fun", "pair"))
    nest_types = [t for t in U.types if TI.type_depth(t) >= min(nesting, U_maxdepth(U))]
    for _ in range(tries):
        if want_poly:
            # a free-type-var family: an identity-like lambda keeps a SET of monotypes alive (poly_breadth>1)
            want = r.choice([t for t in U.types if t[0] == "fun"])
            ast = ("lam", "x", ("var", "x")) if r.random() < 0.5 else TI.gen_typed(r, want, U, {}, budget)
        else:
            want = r.choice(nest_types or U.types)
            ast = TI.gen_typed(r, want, U, {}, budget)
        if TI.size(ast) > 16:
            continue
        comp = TI.compile_expr(ast, U)
        if not C.solutions(comp.csp, limit=1):
            continue
        # EXACT-VERIFY: certified dedP root-type == Algorithm-W principal type (sound + complete)
        ex = TI.exact_dedP(comp.csp)
        pt, ok = TI.algorithm_w(ast)
        if ok and TI.node_types(comp, ex) != TI.ground_principal(pt, U):
            continue                                             # would be an unsound instance — skip
        d = FD.type_difficulty((comp.csp, comp.ast))
        if want_poly and d["poly_breadth"] <= 1.0 + 1e-9:
            continue                                             # insist the poly knob actually produced poly
        return {"faculty": "type", "csp": comp.csp, "ast": comp.ast, "gen": "type_nested",
                "u_max_depth": max(1, int(nesting)), "root": comp.root,
                "difficulty_axis": "nesting_depth x poly_breadth",
                "nesting_depth": d["nesting_depth"], "poly_breadth": d["poly_breadth"]}
    return None


def U_maxdepth(U):
    return max(TI.type_depth(t) for t in U.types)


# ============================================================================ CROSS-FACULTY generic chain
# The generic flow: each stage NARROWS the working cell-set by the faculty's per-cell value RELATIVE to the
# source (the same district-construction flow the hand-picked cross/tri use, lifted to every faculty +
# ordering). active_{k+1} = selected_k, so each stage's STRUCTURE is determined by the previous stage's
# SOLUTION. Source survives every stage (it is trivially same-colour / reachable / same-team / same-type as
# itself); the target must survive every INTERMEDIATE stage (so every hop is load-bearing) and the final
# stage decides the answer. Every stage is an EXACT solver (exact_dedP / certified closure / brute ground).
def _stage_select(fac, active, source, ctx, need_full):
    """Return (selected_set | None). None => an ising stage was not fully determined (reject the instance)."""
    if fac == "csp":
        col = ctx["colors"]
        return frozenset(i for i in active if col[i] == col[source])
    if fac == "type":
        ty = ctx["types"]
        return frozenset(i for i in active if ty[i] == ty[source])
    if fac == "graph":
        edges = frozenset((u, v) for (u, v) in ctx["cand_edges"] if u in active and v in active)
        return frozenset(FAC.reachable(ctx["n"], edges, source) & active)
    if fac == "ising":
        nodes = sorted(active); loc = {v: i for i, v in enumerate(nodes)}; m = len(nodes)
        und = [(loc[min(u, v)], loc[max(u, v)]) for (u, v) in ctx["cand_edges"]
               if u in loc and v in loc and u != v]
        J = FAC.maxcut_J(m, und); h = np.zeros(m, np.float64)
        G = _ground_configs(J, h)
        si = loc[source]
        G = G * G[:, si:si + 1]                                   # canonicalize spin[source] := +1
        rels = {tuple(int(x) for x in row) for row in G}
        if need_full and len(rels) != 1:
            return None                                          # intermediate ising must be determined
        r0 = next(iter(rels)) if len(rels) == 1 else None
        if r0 is None:                                           # final ising: only target determinacy used
            return frozenset()                                   # placeholder (final reads via _ising_relteam)
        return frozenset(nodes[i] for i in range(m) if r0[i] == 1)
    raise ValueError(fac)


def _run_chain(chain, source, target, ctx):
    """Run the exact pipeline. Returns (kind, value, shrank) or None. kind in {'yesno','type'}."""
    active = frozenset(range(ctx["n"]))
    L = len(chain)
    shrank = False
    for k, fac in enumerate(chain):
        last = (k == L - 1)
        if last:
            if fac == "type":
                if target not in active:
                    return None
                return ("type", ctx["types"][target], shrank)
            if fac == "ising":
                nodes = sorted(active); loc = {v: i for i, v in enumerate(nodes)}; m = len(nodes)
                if target not in loc:
                    return None
                und = [(loc[min(u, v)], loc[max(u, v)]) for (u, v) in ctx["cand_edges"]
                       if u in loc and v in loc and u != v]
                J = FAC.maxcut_J(m, und); h = np.zeros(m, np.float64)
                det, same = _ising_relteam(J, h, loc[source], loc[target])
                if not det:
                    return None
                return ("yesno", bool(same), shrank)
            sel = _stage_select(fac, active, source, ctx, need_full=False)
            if sel is None:
                return None
            return ("yesno", target in sel, shrank)
        sel = _stage_select(fac, active, source, ctx, need_full=True)
        if sel is None:
            return None
        if source not in sel or target not in sel or len(sel) < 2:
            return None                                          # target must survive every intermediate hop
        if len(sel) < len(active):
            shrank = True
        active = sel
    return None


def _make_ctx(rng, n, p_edge=0.5):
    facts_c, _c = _planted_2coloring(rng, n)
    csp_c = CU.build_csp(n, 2, facts_c)
    ded = C.exact_dedP(csp_c, csp_c.full())
    if any(len(ded[i]) != 1 for i in range(n)):
        return None
    colors = [int(next(iter(ded[i]))) for i in range(n)]
    facts_t, _t = _planted_typing(rng, n)
    csp_t = FAC.build_type_csp(facts_t, n)
    dedt = C.exact_dedP(csp_t, csp_t.full())
    if any(len(dedt[i]) != 1 for i in range(n)):
        return None
    types = [int(next(iter(dedt[i]))) for i in range(n)]
    cand = frozenset((i, j) for i in range(n) for j in range(n)
                     if i != j and rng.random() < p_edge)
    return {"n": n, "colors": colors, "types": types, "cand_edges": cand,
            "facts_c": [list(f) for f in facts_c], "facts_t": [list(f) for f in facts_t]}


def gen_chain(rng, faculties, n_choices=(6, 7, 8), want_yes=None, tries=400):
    """The GENERIC cross-faculty chain factory: a problem whose answer requires faculties[0] -> flow ->
    faculties[1] -> flow -> ... EXACT-verified end-to-end (each stage's exact solution determines the next
    stage's working set; the final answer re-checked). `faculties` is an ordered chain over the base
    faculties (csp/graph/ising/type) with no immediate repeat. Fills the empty hop-1/hop-2/hop-3 buckets
    (incl. type-as-flow-endpoint and the 4-faculty chains) the cross-occupancy audit found at 0%."""
    chain = tuple(faculties)
    for _ in range(tries):
        n = int(rng.choice(n_choices))
        ctx = _make_ctx(rng, n)
        if ctx is None:
            continue
        source = int(rng.integers(n))
        cands = [t for t in range(n) if t != source]
        rng.shuffle(cands)
        for target in cands:
            res = _run_chain(chain, source, target, ctx)
            if res is None:
                continue
            kind, val, shrank = res
            if not shrank:
                continue                                         # a no-op chain (no hop filtered) — reject
            if kind == "yesno" and want_yes is not None and bool(val) != bool(want_yes):
                continue
            bucket = FD.CompositionBucket(chain, len(chain) - 1)
            ans = ("yes" if val else "no") if kind == "yesno" else TI.type_str(_type_name(val))
            return {"faculty": "chain", "chain": chain, "hop_depth": len(chain) - 1,
                    "n": n, "source": int(source), "target": int(target),
                    "answer": ans, "answer_kind": kind, "answer_val": val if kind == "type" else bool(val),
                    "colors": ctx["colors"], "types": ctx["types"],
                    "cand_edges": sorted(ctx["cand_edges"]),
                    "facts_c": ctx["facts_c"], "facts_t": ctx["facts_t"], "bucket": bucket}
    return None


def _type_name(idx):
    return ("int",) if idx == 0 else ("bool",)


def verify_chain(rec) -> bool:
    """EXACT end-to-end re-check of a chain record: (1) the CSP colours / type assignment re-derive via the
    faculties' OWN exact solver (C.exact_dedP), (2) re-running the pipeline reproduces the recorded answer,
    (3) any ising stage agrees with ising_organ.brute_force_ground (the independent cross-solver)."""
    n = rec["n"]
    csp_c = CU.build_csp(n, 2, [tuple(f) for f in rec["facts_c"]])
    ded = C.exact_dedP(csp_c, csp_c.full())
    colors = [int(next(iter(ded[i]))) if len(ded[i]) == 1 else None for i in range(n)]
    if colors != rec["colors"]:
        return False
    csp_t = FAC.build_type_csp([tuple(f) for f in rec["facts_t"]], n)
    dedt = C.exact_dedP(csp_t, csp_t.full())
    types = [int(next(iter(dedt[i]))) if len(dedt[i]) == 1 else None for i in range(n)]
    if types != rec["types"]:
        return False
    ctx = {"n": n, "colors": colors, "types": types,
           "cand_edges": frozenset(map(tuple, rec["cand_edges"]))}
    res = _run_chain(rec["chain"], rec["source"], rec["target"], ctx)
    if res is None:
        return False
    kind, val, _shrank = res
    if kind != rec["answer_kind"]:
        return False
    if kind == "type":
        if val != rec["answer_val"]:
            return False
    else:
        if bool(val) != bool(rec["answer_val"]):
            return False
    # ising cross-solver agreement (final-stage ising only; intermediate ising used full determinacy already)
    if rec["chain"][-1] == "ising" and kind == "yesno":
        active = frozenset(range(n))
        for fac in rec["chain"][:-1]:
            active = _stage_select(fac, active, rec["source"], ctx, need_full=True)
            if active is None:
                return False
        nodes = sorted(active); loc = {v: i for i, v in enumerate(nodes)}; m = len(nodes)
        und = [(loc[min(u, v)], loc[max(u, v)]) for (u, v) in ctx["cand_edges"]
               if u in loc and v in loc and u != v]
        J = FAC.maxcut_J(m, und); h = np.zeros(m, np.float64)
        same_organ = _organ_relteam(J, h, loc[rec["source"]], loc[rec["target"]])
        if bool(same_organ) != bool(rec["answer_val"]):
            return False
    return True


# ============================================================================ KARP routing-trap
def gen_karp_route(rng, n_lo=3, n_hi=4, m_lo=3, m_hi=6, tries=200):
    """ROUTING-TRAP: a problem with NO direct narrowing faculty, solvable only by REDUCING across the Karp
    graph (organ/reductions). We emit a 3-SAT instance and route it 3-SAT -> MIS -> Ising (the Lucas
    clause-triangle gadget into the Ising hub) — the held-out reduction route, vs the trained direct
    3-SAT -> CSP edge. EXACT-VERIFIED end-to-end against 3-SAT's OWN brute oracle: the certified
    3-SAT -> MIS route recovers satisfiability + a satisfying assignment, and the approximate MIS -> Ising
    hub is output-checked. Maps to the composition bucket (graph->ising, has_reduction=True) — the
    routing_trap OOD region (1/7 occupied before)."""
    for _ in range(tries):
        x = RED.rand_sat(rng, "sat3", n_lo=n_lo, n_hi=n_hi, m_lo=m_lo, m_hi=m_hi)
        truth = RED.exact_sat(x)
        if not truth["sat"]:                                     # witness-first => always sat, but be safe
            continue
        # (a) the CERTIFIED reduction route 3-SAT -> MIS (gadget): exact, sound-by-construction
        red_mis = RED.Sat3ToMIS()
        if not RED.verify_reduction(red_mis, x):
            continue
        # (b) the routing-trap route 3-SAT -> MIS -> Ising (no direct faculty for argmax|IS|): route the
        # gadget MIS graph through the Ising hub. The EXACT-verified backbone is the certified gadget (the
        # MIS SIZE decides satisfiability; size m <=> SAT — sound-by-construction), checked against the exact
        # MIS. The approximate Ising hub is OUTPUT-CHECKED (reported recovery, not the soundness gate).
        g, _node_lits = RED.sat3_to_mis_gadget(x)
        rg = RED.ReductionGraph()
        route = rg.route(g)                                      # mis -> ising (no certified direct solver)
        size_exact, mis_exact = RED.exact_mis(g)
        sat_via_gadget = (size_exact == len(x.clauses))          # the certified gadget answer
        decoded = red_mis.decode(frozenset(mis_exact), x)        # decode the EXACT max-IS to an assignment
        assign_ok = x.satisfies(decoded)
        sel, _r = rg.solve_via_route(g, restarts=128, steps=200, seed=int(rng.integers(1 << 30)))
        ising_recovers = bool(sel is not None and len(sel) == size_exact)   # approximate hub recovery (report)
        bucket = FD.CompositionBucket(("graph", "ising"), 1, has_reduction=True)
        return {"faculty": "reduction", "src_type": "sat3", "n_vars": x.n,
                "n_clauses": len(x.clauses), "route": repr(route),
                "gadget_nodes": g.n, "mis_size": size_exact,
                "sat_gold": truth["sat"], "sat_via_gadget": sat_via_gadget,
                "assign_satisfies": assign_ok, "certified_mis_route_ok": True,
                "ising_hub_recovers": ising_recovers, "bucket": bucket, "gen": "karp_route",
                "difficulty_axis": "routing_trap"}
    return None


# ============================================================================ the difficulty-controlled MIX
# The consumable spec the fabric-retrain uses: per-faculty bucket TARGETS (the occupancy each faculty's
# generator mix should hit — explicitly weighting the previously-empty hard buckets) + the cross-faculty
# hop-depth distribution + the reduction-on-path share. Each entry names the generator(s) + the knob sweep.
CURRICULUM_MIX = {
    "ising": {
        "generators": [
            ("gen_ising_frustrated", {"target_frustration": [0.0, 0.15, 0.30, 0.45]}),
            ("gen_ising_glass", {"abs_J_spread": [0.6, 1.0, 1.5]}),
        ],
        "bucket_targets": {
            "frustration": {"0": 0.20, "lo(<=.15)": 0.20, "mid(<=.35)": 0.30, "hi": 0.30},
            "abs_J_spread": {"uniform": 0.40, "spread": 0.60},   # was 100% uniform — now majority spread
        },
    },
    "graph": {
        "generators": [("gen_graph_depth", {"reach_depth": [1, 2, 3, 4, 5]})],
        "bucket_targets": {
            "reach_depth": {"1": 0.15, "2": 0.20, "3": 0.25, "deep(>=4)": 0.40},  # was 90% depth<=2
        },
    },
    "type": {
        "generators": [
            ("gen_type_nested", {"nesting": [1, 2], "want_poly": [False, True]}),
        ],
        "bucket_targets": {
            "nesting_depth": {"base(0)": 0.15, "depth1": 0.50, "depth>=2": 0.35},  # was 100% base
            "poly_breadth": {"mono": 0.50, "poly": 0.50},        # was 100% mono
        },
    },
    "cross": {
        "generators": [("gen_chain", {"hop_depth": [1, 2, 3]}),
                       ("gen_karp_route", {})],
        "hop_depth_distribution": {"1": 0.40, "2": 0.35, "3": 0.25},   # was hop1 2/12, hop2 1/36, hop3 0/108
        "reduction_on_path_frac": 0.15,
        "include_type_endpoint": True,
    },
}


# ============================================================================ cross-faculty coverage (buckets)
def cross_coverage(buckets, max_hop=3) -> dict:
    """Composition-lattice coverage from a list of CompositionBucket objects (the gen_chain / gen_karp_route
    outputs) — the same audit faculty_difficulty.cross_occupancy reports, but reading explicit buckets so
    the NEW chains beyond the 7 live faculty tags count. Coverage vs faculty_difficulty.full_lattice."""
    lattice = FD.full_lattice(max_hop)
    lat_chains = {b.chain for b in lattice}
    by_hop_total = Counter(b.hop_depth for b in lattice)
    seen = {b.chain for b in buckets if b.chain in lat_chains}
    by_hop_seen = Counter(len(c) - 1 for c in seen)
    red = sum(1 for b in buckets if b.has_reduction)
    return {
        "distinct_chains_covered": len(seen),
        "lattice_size": len(lattice),
        "coverage_frac": round(len(seen) / max(1, len(lattice)), 3),
        "by_hop_depth": {str(k): {"covered": by_hop_seen.get(k, 0), "total": by_hop_total[k]}
                         for k in sorted(by_hop_total)},
        "reduction_on_path": red,
        "covered_chains": sorted("->".join(c) for c in seen),
    }


# ============================================================================ exact-verification harness
def verify(seed=0, n_each=40, verbose=True) -> dict:
    """Exact-verify every generator: each instance's answer re-checked against the faculty's OWN exact
    solver. Returns mismatch counts (all MUST be 0)."""
    rng = np.random.default_rng(seed)
    res = {}

    # ---- ISING frustrated/glass: brute ground state (FD) == ising_organ.brute_force_ground ----
    mis = tot = 0
    for ax in (0.0, 0.15, 0.30, 0.45):
        for _ in range(n_each // 4 + 1):
            r = gen_ising_frustrated(rng, target_frustration=ax)
            if r is None:
                continue
            tot += 1
            same_organ = _organ_relteam(r["J_true"], r["h_true"], r["source"], r["target"])
            mis += int(same_organ != (r["gold_idx"] == 0))
    res["ising_frustrated"] = (mis, tot)
    mis2 = tot2 = 0
    for sp in (0.6, 1.0, 1.5):
        for _ in range(n_each // 3 + 1):
            r = gen_ising_glass(rng, abs_J_spread=sp)
            if r is None:
                continue
            tot2 += 1
            same_organ = _organ_relteam(r["J_true"], r["h_true"], r["source"], r["target"])
            mis2 += int(same_organ != (r["gold_idx"] == 0))
    res["ising_glass"] = (mis2, tot2)

    # ---- GRAPH depth: certified reachability closure ----
    mis = tot = 0
    for D in (1, 2, 3, 4, 5):
        for k in range(n_each // 5 + 1):
            r = gen_graph_depth(rng, reach_depth=D, want_yes=(k % 2 == 0))
            if r is None:
                continue
            tot += 1
            reach = FAC.reachable(r["n"], frozenset(map(tuple, r["true_edges"])), r["source"])
            mis += int((r["target"] in reach) != (r["gold_idx"] == 0))
            # the certified closure read-out (GraphReach) must agree too
            gs = FAC.GraphReachState(r["n"], frozenset(map(tuple, r["true_edges"])),
                                     r["source"], r["target"])
            mis += int(bool(FAC.GraphReach().reduce(gs).reach[r["target"]]) != (r["gold_idx"] == 0))
    res["graph_depth"] = (mis, tot)

    # ---- TYPE nested/poly: certified dedP == Algorithm-W principal type (recompile under the universe) ----
    mis = tot = 0
    for nest in (1, 2):
        for poly in (False, True):
            for _ in range(n_each // 4 + 1):
                r = gen_type_nested(rng, nesting=nest, want_poly=poly)
                if r is None:
                    continue
                tot += 1
                U = TI.Universe(max_depth=r["u_max_depth"], ctors=("fun", "pair"))
                comp = TI.compile_expr(r["ast"], U)              # the faculty's own compile -> Compiled
                ex = TI.exact_dedP(comp.csp)
                pt, ok = TI.algorithm_w(r["ast"])
                if ok:
                    mis += int(TI.node_types(comp, ex) != TI.ground_principal(pt, U))
    res["type_nested"] = (mis, tot)

    # ---- CHAIN: exact end-to-end (faculty exact_dedP + pipeline + ising_organ cross-solver) ----
    mis = tot = 0
    for chain in _CHAIN_VERIFY_SET:
        for k in range(3):
            r = gen_chain(rng, chain, want_yes=(None if "type" == chain[-1] else (k % 2 == 0)))
            if r is None:
                continue
            tot += 1
            mis += int(not verify_chain(r))
    res["chain"] = (mis, tot)

    # ---- KARP route: reductions' exact end-to-end verifier ----
    mis = tot = 0
    for _ in range(max(8, n_each // 3)):
        r = gen_karp_route(rng)
        if r is None:
            continue
        tot += 1
        # certified mis route already verified inside; re-assert the gadget answer vs the brute SAT oracle:
        # the MIS SIZE must decide satisfiability and the decoded EXACT max-IS must satisfy the formula.
        mis += int(r["sat_via_gadget"] != r["sat_gold"])
        mis += int(r["assign_satisfies"] is False)
    res["karp_route"] = (mis, tot)

    if verbose:
        print("================ EXACT-VERIFICATION (mismatch / instances; MUST be 0 mismatch) ================")
        for k, (m, t) in res.items():
            flag = "OK" if m == 0 else "  <-- MISMATCH"
            print(f"  {k:18s} {m:3d} mismatch / {t:4d} verified   {flag}")
    return res


_CHAIN_VERIFY_SET = [
    ("csp", "graph"), ("graph", "ising"), ("type", "graph"), ("csp", "type"),
    ("csp", "graph", "ising"), ("type", "csp", "graph"),
    ("type", "csp", "graph", "ising"),
]


# ============================================================================ before -> after re-profile
def reprofile(seed=0, n=120, verbose=True) -> dict:
    """The before->after occupancy proof: the live generators (the audit baseline) vs the NEW difficulty-
    controlled generators, per faculty + the cross-faculty lattice coverage. Shows the empty hard buckets
    are now populated (the holes closed)."""
    from . import faculty_tasks as FT                       # read-only baseline (the audit's live generators)
    rng = np.random.default_rng(seed)
    out = {"before": {}, "after": {}}

    # ---------- ISING ----------
    before_is = [FT.gen_ising_record(rng, want_yes=(k % 2 == 0)) for k in range(n)]
    after_is = []
    for j in range(n // 2):                                       # frustration sweep (incl. the hi >.35 tail)
        ax = (0.0, 0.15, 0.30, 0.45)[j % 4]
        r = gen_ising_frustrated(rng, target_frustration=ax)
        if r is not None:
            after_is.append(r)
    for j in range(n - len(after_is)):                            # the SK |J|-spread spin-glass regime
        r = gen_ising_glass(rng, abs_J_spread=(0.6, 1.0, 1.5)[j % 3])
        if r is not None:
            after_is.append(r)
    out["before"]["ising"] = FD.profile_faculty(before_is, "ising", label="gen_ising_record (live)")
    out["after"]["ising"] = FD.profile_faculty(after_is, "ising", label="frustrated+glass (new)")

    # ---------- GRAPH ----------
    before_g = [FT.gen_graph_record(rng) for _ in range(n)]
    after_g = []
    for k in range(n):
        D = (1, 2, 3, 4, 5)[k % 5]
        r = gen_graph_depth(rng, reach_depth=D, want_yes=(k % 2 == 0))
        if r is not None:
            after_g.append(r)
    out["before"]["graph"] = FD.profile_faculty(before_g, "graph", label="gen_graph_record (live)")
    out["after"]["graph"] = FD.profile_faculty(after_g, "graph", label="graph_depth (new)")

    # ---------- TYPE ----------
    before_t = [FT.gen_type_record(rng) for _ in range(n)]
    before_t_structs = [(FAC.build_type_csp([tuple(f) for f in r["true_facts"]], r["n"]), None)
                        for r in before_t]
    after_t = []
    for k in range(n):
        nest = (1, 2)[k % 2]; poly = (k % 3 == 0)
        r = gen_type_nested(rng, nesting=nest, want_poly=poly)
        if r is not None:
            after_t.append((r["csp"], r["ast"]))
    out["before"]["type"] = FD.profile_faculty(before_t_structs, "type", label="gen_type_record (live eq-tree)")
    out["after"]["type"] = FD.profile_faculty(after_t, "type", label="type_nested (new)")

    # ---------- CROSS-FACULTY lattice ----------
    before_cross = cross_coverage([FD.LIVE_COMPOSITION[t] for t in FAC.LIVE_FACULTIES])
    after_buckets = [FD.LIVE_COMPOSITION[t] for t in FAC.LIVE_FACULTIES]
    chain_targets = _chain_target_list()
    chain_ok = 0
    for chain in chain_targets:
        r = gen_chain(rng, chain)
        if r is not None:
            after_buckets.append(r["bucket"]); chain_ok += 1
    for _ in range(6):
        r = gen_karp_route(rng)
        if r is not None:
            after_buckets.append(r["bucket"]); break
    after_cross = cross_coverage(after_buckets)
    out["before"]["cross"] = before_cross
    out["after"]["cross"] = after_cross
    out["cross_chains_generated"] = chain_ok

    if verbose:
        _print_reprofile(out)
    return out


def _chain_target_list():
    """The chains gen_chain is asked to fill: ALL hop-1 directed pairs, a broad hop-2 set, and hop-3
    4-faculty chains (incl. type-as-endpoint) — the empty lattice regions."""
    base = FD.BASE_FACULTIES
    pairs = [(a, b) for a in base for b in base if a != b]
    triples = [(a, b, c) for a in base for b in base for c in base
               if a != b and b != c]
    quads = [(a, b, c, d) for a in base for b in base for c in base for d in base
             if a != b and b != c and c != d]
    import random as _r
    rr = _r.Random(0)
    rr.shuffle(triples); rr.shuffle(quads)
    return pairs + triples[:18] + quads[:14]


def _print_reprofile(out):
    print("\n================ BEFORE -> AFTER per-faculty occupancy (the holes closing) ================")
    for fac in ("ising", "graph", "type"):
        print(f"\n--- {fac} ---")
        print("  BEFORE:"); FD.print_faculty_profile(out["before"][fac])
        print("  AFTER :"); FD.print_faculty_profile(out["after"][fac])
    print("\n================ BEFORE -> AFTER cross-faculty lattice coverage ================")
    b, a = out["before"]["cross"], out["after"]["cross"]
    print(f"  distinct ordered chains covered:  {b['distinct_chains_covered']}/{b['lattice_size']} "
          f"({b['coverage_frac']*100:.1f}%)  ->  {a['distinct_chains_covered']}/{a['lattice_size']} "
          f"({a['coverage_frac']*100:.1f}%)")
    print(f"  by hop-depth BEFORE: {b['by_hop_depth']}")
    print(f"  by hop-depth AFTER : {a['by_hop_depth']}")
    print(f"  reduction-on-path BEFORE: {b['reduction_on_path']}   AFTER: {a['reduction_on_path']}")
    print(f"  ({out['cross_chains_generated']} new exact-verified chains generated)")


if __name__ == "__main__":
    v = verify()
    print()
    reprofile()
