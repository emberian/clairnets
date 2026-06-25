"""Exact finite-CSP harness — the GROUND TRUTH for lattice-deduction experiments.

Pure Python (no torch). Everything here is exact / checkable, so any neural proposer we train
later is measured against THIS, not against a planted solution. Grounded in the Lean LDT files
(~/dev/graphplay/Graphplay/Integrations/LDT*.lean):

  * soundness is free — we either output-verify or dominate the exact transformer dedₚ;
  * COMPLETENESS is the research variable — it depends on the lattice LEVEL (per-cell / pair / …)
    and the problem's WIDTH. Same operator solves the width-1 chain, abstains on affine XOR;
  * the polymorphism DIAGNOSTIC predicts the required level: semilattice ⇒ AC-solvable,
    affine + no-majority ⇒ unbounded width ⇒ per-cell must abstain.

A CSP is (n cells, domain V=0..d-1, constraints). Each constraint is (scope, allowed) with
`scope` a tuple of cell indices and `allowed` a set of value-tuples (extensional relation).
A per-cell lattice STATE is `dom`: a tuple of frozensets, dom[i] ⊆ V = still-alive values at cell i.
"""
from __future__ import annotations

import itertools as it
from dataclasses import dataclass


@dataclass(frozen=True)
class CSP:
    n: int                      # number of cells
    d: int                      # domain size (values 0..d-1)
    cons: tuple                 # tuple of (scope: tuple[int], allowed: frozenset[tuple])

    def full(self):
        V = frozenset(range(self.d))
        return tuple(V for _ in range(self.n))


# --------------------------------------------------------------------- exact solving
def solutions(csp: CSP, dom=None):
    """All assignments s (tuple len n) with s[i] in dom[i] satisfying every constraint.
    Brute force with per-constraint checking — exact, intended for SMALL CSPs (the ground truth)."""
    dom = dom or csp.full()
    sols = []
    # order cells by smallest domain for light pruning
    for s in it.product(*[sorted(dom[i]) for i in range(csp.n)]):
        if all(tuple(s[i] for i in sc) in al for sc, al in csp.cons):
            sols.append(s)
    return sols


def exact_dedP(csp: CSP, dom):
    """The EXACT best per-cell transformer dedₚ(a) = α(γ(a) ∩ solutions): for each cell, the set of
    values used by SOME full solution consistent with the current domains. The strongest SOUND
    per-cell narrowing (requires enumerating solutions). a_next ⊆ a always."""
    sols = solutions(csp, dom)
    if not sols:
        return tuple(frozenset() for _ in range(csp.n))      # ⊥ : unsatisfiable under dom
    return tuple(frozenset(s[i] for s in sols) for i in range(csp.n))


# --------------------------------------------------------------------- arc consistency (level 0, cheap)
def ac_step(csp: CSP, dom):
    """One generalized-arc-consistency pass (the cheap local operator). Keep value v at cell i iff
    every constraint whose scope contains i has a satisfying tuple, consistent with the current
    domains, that places v at i. Sound but LOCAL — weaker than dedₚ."""
    new = list(dom)
    for i in range(csp.n):
        keep = set()
        for v in dom[i]:
            ok = True
            for sc, al in csp.cons:
                if i not in sc:
                    continue
                pos = sc.index(i)
                support = any(t[pos] == v and all(t[k] in dom[c] for k, c in enumerate(sc)) for t in al)
                if not support:
                    ok = False; break
            if ok:
                keep.add(v)
        new[i] = frozenset(keep)
    return tuple(new)


def to_fixpoint(step, csp, dom, max_iters=None):
    """Iterate a reductive per-cell operator to a fixed point. Returns (dom, n_steps)."""
    max_iters = max_iters or csp.n * csp.d + 2
    for t in range(1, max_iters + 1):
        nxt = step(csp, dom)
        if nxt == dom:
            return dom, t
        dom = nxt
    return dom, max_iters


# --------------------------------------------------------------------- outcomes / metrics
def status(dom):
    """'conflict' if any cell empty; 'solved' if every cell a singleton; else 'open'."""
    if any(len(c) == 0 for c in dom):
        return "conflict"
    if all(len(c) == 1 for c in dom):
        return "solved"
    return "open"


def solve(csp: CSP, step=ac_step):
    """Run a deduction operator to fixpoint from the full grid. Returns dict with outcome + the
    exact-alpha completeness facts. SOUND by construction if `step` is dominated by dedₚ."""
    dom, steps = to_fixpoint(step, csp, csp.full())
    st = status(dom)
    sols = solutions(csp)
    # does the operator's narrowing ever drop a value the EXACT transformer would keep? (false elim)
    exact = exact_dedP(csp, csp.full())
    false_elim = sum(len(exact[i] - dom[i]) for i in range(csp.n)) if sols else 0
    return {
        "outcome": st,                                   # solved | conflict | open(=abstain)
        "steps": steps,
        "n_solutions": len(sols),
        "solvable": len(sols) > 0,                       # has >=1 solution
        "uniquely_solvable": len(sols) == 1,
        "false_elim_vs_exact": false_elim,               # MUST be 0 for a sound operator
        "final_alive": tuple(len(c) for c in dom),
        "exact_alive": tuple(len(c) for c in exact),     # what dedₚ (best per-cell) would reach
    }


# --------------------------------------------------------------------- polymorphism diagnostic
def _is_polymorphism(csp: CSP, op, arity):
    """op: callable(*arity values)->value. Polymorphism iff applying op coordinate-wise to any
    `arity` solutions of each constraint yields a tuple still in that constraint. We test on the
    constraint relations directly (closure of each allowed-set under op)."""
    for sc, al in csp.cons:
        al = list(al)
        for combo in it.product(al, repeat=arity):
            res = tuple(op(*[combo[a][p] for a in range(arity)]) for p in range(len(sc)))
            if res not in al:
                return False
    return True


def polymorphism_signature(csp: CSP):
    """Compute the algebraic signature that the bounded-width dichotomy keys on (LDTPolymorphism):
      semilattice (boolean meet ∧)  -> AC-solvable / Horn-like, level 0 suffices
      majority (2-of-3)             -> bounded width (some local-consistency level solves)
      affine   (x⊕y⊕z)             -> the unbounded-width wall; per-cell must abstain
    Only meaningful for boolean domains (d==2); returns flags + a predicted level."""
    if csp.d != 2:
        return {"note": "signature defined for boolean d=2", "predicted_level": "?"}
    meet = _is_polymorphism(csp, lambda a, b: a & b, 2)
    join = _is_polymorphism(csp, lambda a, b: a | b, 2)
    maj = _is_polymorphism(csp, lambda a, b, c: (a & b) | (b & c) | (a & c), 3)
    affine = _is_polymorphism(csp, lambda a, b, c: a ^ b ^ c, 3)
    # dichotomy: bounded width iff omits affine (equivalently has majority/NU); affine+no-majority = hard
    if affine and not maj:
        level = "unbounded (affine wall; per-cell AC must abstain — needs 3-consistency)"
    elif meet or join:
        level = "0 (semilattice/Horn — per-cell AC solves)"
    elif maj:
        level = "bounded (majority — some local-consistency level solves)"
    else:
        level = "unknown"
    return {"semilattice": meet or join, "majority": maj, "affine": affine, "predicted_level": level}


# --------------------------------------------------------------------- generators
def _rel(scope, pred, d):
    """Build an extensional constraint (scope, allowed) from a predicate over a value-tuple."""
    al = frozenset(t for t in it.product(range(d), repeat=len(scope)) if pred(t))
    return (tuple(scope), al)


def chain_eq(n=4):
    """x0 = false ; x_i = x_{i+1}.  Semilattice (boolean meet). Per-cell AC SOLVES (width 1)."""
    cons = [_rel((0,), lambda t: t[0] == 0, 2)]
    cons += [_rel((i, i + 1), lambda t: t[0] == t[1], 2) for i in range(n - 1)]
    return CSP(n, 2, tuple(cons))


def xor_parity():
    """x=y ; y=z ; x⊕y⊕z=0.  Affine, no majority. Unique solution (0,0,0) but per-cell AC ABSTAINS."""
    cons = [_rel((0, 1), lambda t: t[0] == t[1], 2),
            _rel((1, 2), lambda t: t[0] == t[1], 2),
            _rel((0, 1, 2), lambda t: (t[0] ^ t[1] ^ t[2]) == 0, 2)]
    return CSP(3, 2, tuple(cons))


def two_sat(n, clauses):
    """clauses: list of (i, bi, j, bj) meaning (x_i==bi) or (x_j==bj)."""
    cons = [_rel((i, j), (lambda bi, bj: lambda t: t[0] == bi or t[1] == bj)(bi, bj), 2)
            for (i, bi, j, bj) in clauses]
    return CSP(n, 2, tuple(cons))


def three_sat(n, clauses):
    """clauses: list of (i,bi,j,bj,k,bk) meaning the disjunction of the three literals."""
    def mk(bi, bj, bk):
        return lambda t: t[0] == bi or t[1] == bj or t[2] == bk
    cons = [_rel((i, j, k), mk(bi, bj, bk), 2) for (i, bi, j, bj, k, bk) in clauses]
    return CSP(n, 2, tuple(cons))


def coloring(n, edges, k=3):
    """Graph k-coloring: adjacent cells differ. Pair-level (x≠y) constraints."""
    cons = [_rel((u, v), lambda t: t[0] != t[1], k) for (u, v) in edges]
    return CSP(n, k, tuple(cons))


GENERATORS = {"chain_eq": chain_eq, "xor_parity": xor_parity}


# --------------------------------------------------------------------- pair lattice (level 1)
# State P maps each ORDERED pair (i,j), i!=j, to a frozenset of allowed (v_i, v_j). Symmetric by
# construction. This catches 2-cell correlations (e.g. equality) that the per-cell lattice projects
# to ⊤ — the Lean's pair_strictly_richer_than_cell. Still cannot see 3-cell factors (the XOR wall).
def pair_init(csp: CSP, dom=None):
    dom = dom or csp.full()
    P = {}
    for i in range(csp.n):
        for j in range(csp.n):
            if i == j:
                continue
            al = set()
            for vi in dom[i]:
                for vj in dom[j]:
                    a = {i: vi, j: vj}
                    if all(tuple(a[c] for c in sc) in rel for sc, rel in csp.cons if set(sc) <= {i, j}):
                        al.add((vi, vj))
            P[(i, j)] = frozenset(al)
    return P


def pair_step(csp: CSP, P):
    """Path-consistency: keep (v_i,v_j) iff for every other cell k there is a v_k with (v_i,v_k) and
    (v_k,v_j) both allowed. Reductive; sound. Iterating to fixpoint = 3-cell path consistency."""
    new = {}
    for (i, j), al in P.items():
        keep = set()
        for (vi, vj) in al:
            ok = True
            for k in range(csp.n):
                if k in (i, j):
                    continue
                if not any((vi, vk) in P[(i, k)] and (vk, vj) in P[(k, j)] for vk in range(csp.d)):
                    ok = False; break
            if ok:
                keep.add((vi, vj))
        new[(i, j)] = frozenset(keep)
    return new


def pair_cells(csp: CSP, P):
    """Project the pair state back to per-cell domains: v survives at i iff it appears (paired with
    something) for EVERY partner j."""
    if csp.n == 1:
        return (frozenset(range(csp.d)),)
    cells = []
    for i in range(csp.n):
        surv = None
        for j in range(csp.n):
            if j == i:
                continue
            here = frozenset(vi for (vi, vj) in P[(i, j)])
            surv = here if surv is None else (surv & here)
        cells.append(surv if surv is not None else frozenset(range(csp.d)))
    return tuple(cells)


def pair_to_fixpoint(csp: CSP, dom=None):
    P = pair_init(csp, dom)
    for t in range(1, csp.n * csp.d * csp.d + 2):
        nxt = pair_step(csp, P)
        if nxt == P:
            return P, t
        P = nxt
    return P, t


def pair_exact(csp: CSP, dom=None):
    """Exact pair transformer: P[(i,j)] = {(s[i],s[j]) over all solutions}. Ground truth at level 1."""
    sols = solutions(csp, dom)
    return {(i, j): frozenset((s[i], s[j]) for s in sols)
            for i in range(csp.n) for j in range(csp.n) if i != j}


def solve_pair(csp: CSP):
    """Run path-consistency to fixpoint, project to cells, report outcome + soundness vs pair-exact."""
    P, steps = pair_to_fixpoint(csp)
    cells = pair_cells(csp, P)
    sols = solutions(csp)
    ex = pair_exact(csp)
    # false elim at pair level: pairs the operator dropped that some solution actually uses
    fe = sum(len(ex[k] - P[k]) for k in ex) if sols else 0
    return {"outcome": status(cells), "steps": steps, "final_alive": tuple(len(c) for c in cells),
            "false_elim_vs_exact": fe, "solvable": len(sols) > 0}


def completeness_per_level(csps):
    """For a corpus of (name, csp), report solve rate at level 0 (per-cell AC) vs level 1 (pair PC),
    over the SOLVABLE instances. Demonstrates 'completeness = level × width'."""
    lvl0 = lvl1 = tot = 0
    fe0 = fe1 = 0
    for _, c in csps:
        r0, r1 = solve(c, ac_step), solve_pair(c)
        if not r0["solvable"]:
            continue
        tot += 1
        lvl0 += r0["outcome"] == "solved"
        lvl1 += r1["outcome"] == "solved"
        fe0 += r0["false_elim_vs_exact"]; fe1 += r1["false_elim_vs_exact"]
    return {"n_solvable": tot, "level0_solved": lvl0, "level1_solved": lvl1,
            "level0_false_elim": fe0, "level1_false_elim": fe1}


if __name__ == "__main__":
    print("exact-CSP harness self-check (reproduces the proven Lean facts)\n")

    ch = chain_eq(4)
    r = solve(ch, ac_step)
    print(f"chain_eq(4):  AC outcome={r['outcome']}  steps={r['steps']}  sols={r['n_solutions']}  "
          f"false_elim={r['false_elim_vs_exact']}   sig={polymorphism_signature(ch)['predicted_level']}")
    assert r["outcome"] == "solved" and r["false_elim_vs_exact"] == 0, "chain should AC-solve, soundly"

    xr = xor_parity()
    r = solve(xr, ac_step)
    sig = polymorphism_signature(xr)
    print(f"xor_parity:   AC outcome={r['outcome']}  steps={r['steps']}  sols={r['n_solutions']} "
          f"(unique={r['uniquely_solvable']})  false_elim={r['false_elim_vs_exact']}")
    print(f"              exact dedP alive={r['exact_alive']}  AC alive={r['final_alive']}  sig={sig}")
    assert r["outcome"] == "open", "per-cell AC must ABSTAIN on affine XOR (the wall)"
    assert r["uniquely_solvable"], "XOR system has a unique solution (0,0,0)"
    assert r["false_elim_vs_exact"] == 0, "AC stays sound even while incomplete"
    assert sig["affine"] and not sig["majority"], "XOR signature = affine, no majority"

    # the exact transformer dedP solves XOR (global), where local AC abstains — the level gap
    dom, _ = to_fixpoint(exact_dedP, xr, xr.full())
    print(f"              exact dedP outcome={status(dom)} (global enumeration solves where local AC can't)")
    assert status(dom) == "solved"

    # 2-SAT (Horn-ish) and a tiny 3-SAT sanity
    ts = two_sat(3, [(0, 1, 1, 1), (1, 0, 2, 1)])
    print(f"two_sat(3):   sols={solve(ts)['n_solutions']}  sig={polymorphism_signature(ts)['predicted_level']}")
    col = coloring(3, [(0, 1), (1, 2), (0, 2)], k=3)
    print(f"coloring K3:  sols={solve(col)['n_solutions']} (3!=6 proper 3-colorings of a triangle)")
    assert solve(col)["n_solutions"] == 6

    # ---- pair level (Level 1): the representation gap the Lean proves ----
    print("\n-- pair lattice (level 1) --")
    eq = CSP(2, 2, (_rel((0, 1), lambda t: t[0] == t[1], 2),))      # the equality atom {x=y}
    P, _ = pair_to_fixpoint(eq)
    ac_alive = solve(eq, ac_step)["final_alive"]
    print(f"equality {{x=y}}:  per-cell AC alive={ac_alive} (=⊤, forgets the correlation)  "
          f"pair P[(0,1)]={sorted(P[(0,1)])} (the diagonal — correlation kept)")
    assert ac_alive == (2, 2) and P[(0, 1)] == frozenset({(0, 0), (1, 1)}), "pair > cell on equality"

    rp = solve_pair(xr)   # pair on XOR: narrows the equality atoms but still can't see 3-cell parity
    print(f"xor_parity:    pair outcome={rp['outcome']} alive={rp['final_alive']} "
          f"(narrows correlations but abstains — 3-cell parity needs LEVEL 2, honest correction)")
    assert rp["outcome"] == "open" and rp["false_elim_vs_exact"] == 0

    # ---- completeness-per-level over a small corpus ----
    corpus = [("chain", chain_eq(4)), ("xor", xor_parity()),
              ("eq", eq), ("2sat", two_sat(3, [(0, 1, 1, 1), (1, 0, 2, 1)])),
              ("3sat", three_sat(3, [(0, 1, 1, 1, 2, 1), (0, 0, 1, 0, 2, 0)])),
              ("triK3", coloring(3, [(0, 1), (1, 2), (0, 2)], 3))]
    cl = completeness_per_level(corpus)
    print(f"\ncompleteness-per-level over {cl['n_solvable']} solvable CSPs:  "
          f"level0(cell)={cl['level0_solved']} solved  level1(pair)={cl['level1_solved']} solved  "
          f"| false_elim L0={cl['level0_false_elim']} L1={cl['level1_false_elim']} (both must be 0)")
    assert cl["level1_solved"] >= cl["level0_solved"], "pair never solves FEWER than per-cell"
    assert cl["level0_false_elim"] == 0 and cl["level1_false_elim"] == 0, "both levels stay sound"

    print("\nALL CHECKS PASS — harness matches the Lean: soundness free, completeness is level×width.")
