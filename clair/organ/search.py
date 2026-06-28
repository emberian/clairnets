"""clair/organ/search.py — THE DPLL SEARCH LEG GLaDOS dropped from the Lattice Deduction Transformer.

The LDT (notes/ldt_regrounding.md, pdfs/2605.08605, the machine-checked Graphplay theory) is TWO loops,
not one:

  * INNER  = the learned dedₚ deduction (bounded-width narrowing to a fixpoint). We HAVE this — it is
    clair.organ.compose.reduced_product (certified ops + the gated neural organ, met to a fixpoint).
  * OUTER  = a DPLL SEARCH loop: NARROW → if UNDETERMINED, BRANCH on a cell+value → BACKTRACK on
    CONFLICT → OUTPUT-CHECK the assignment. GLaDOS kept only deduction + certified-ops + abstain, so
    it was MISSING this leg — the general way past a deduction stall.

The dichotomy theory (Feder–Vardi / Barto–Kozik, reproduced as `acStep_xor_sound_but_abstains`):
local consistency (dedₚ) solves EXACTLY the bounded-width CSPs; AFFINE / XOR over GF(2) is
UNBOUNDED width — provably NOT solvable by per-cell deduction — and is crossed only by "BRANCHING or
LINEAR ALGEBRA". GLaDOS already had the linear-algebra leg (clair.organ.bank.GF2RowSpace); this file
restores the BRANCHING leg.

Soundness is FREE and training-independent: every returned assignment is OUTPUT-CHECKED against the
exact CSP constraints (clair.csp), so an UNSOUND reduction or an unsound branch heuristic can never
make `solve` return a wrong answer — exactly `checkedSolve_sound` (sound for ANY solve map). The
search is also COMPLETE: full chronological backtracking over the branch order, bounded by `max_nodes`.

This file READS clair.organ.{compose,protocol,bank} and clair.csp and EDITS none of them — the inner
deduction loop is reused verbatim (reduced_product); only the missing outer loop is added here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .. import csp as C
from .protocol import CSPState
from .compose import reduced_product


# ============================================================ the output check (soundness is here)
def check_assignment(csp: C.CSP, assignment) -> bool:
    """The OUTPUT CHECK: does `assignment` (a per-cell value tuple) satisfy EVERY CSP constraint?
    This is the whole soundness argument — `solve` returns an assignment only if this returns True,
    so soundness holds for ANY reductions / ANY branch heuristic (`checkedSolve_sound`)."""
    if assignment is None or any(v is None for v in assignment):
        return False
    return all(tuple(assignment[c] for c in sc) in al for sc, al in csp.cons)


def _singleton_assignment(dom):
    """Extract the per-cell value tuple from an all-singleton domain (caller ensures `solved`)."""
    return tuple(next(iter(dom[i])) for i in range(len(dom)))


# ============================================================ branch heuristics (the pluggable hook)
# A branch heuristic maps an UNDETERMINED (open) CSPState to (cell, ordered_values): which cell to
# split on and in what order to try its still-alive values. It only affects SEARCH ORDER / efficiency
# — never soundness (the output check guards that) and never completeness (every value is tried).
Branch = Callable[[CSPState], "tuple[int, list]"]


def most_constrained_branch(state: CSPState) -> "tuple[int, list]":
    """The sound DEFAULT: most-constrained-cell / smallest-domain-first (the MRV order
    clair.csp.solutions uses), values in ascending order. Picks the open cell with the fewest alive
    values (>1) — the standard fail-first DPLL variable order."""
    dom = state.dom
    cell = min((i for i in range(state.csp.n) if len(dom[i]) > 1), key=lambda i: len(dom[i]))
    return cell, sorted(dom[cell])


def make_guided_branch(organ, fallback: Branch = most_constrained_branch) -> Branch:
    """SCAFFOLD (interface only — NOT trained here). The learned-branch hook: a future ORGAN/LM
    proposes the branch order, and search + the output check keep it SOUND regardless of the
    proposal (the GLaDOS-native "neural proposes, checked" form of branching).

    `organ` is any object exposing `guidance(state) -> {'survival_logits': [n, K], ...}` (the
    clair.organ.protocol.Reduction guidance hook — e.g. CoreNarrowOrgan). We branch on the open cell
    whose alive values the proposer is MOST decisive about and try them in the proposer's preferred
    (descending-logit) order. Soundness/completeness are unchanged — only the order is learned — so
    this can be swapped in (and later trained) without touching the search/soundness core. Falls back
    to `fallback` whenever guidance is unavailable or malformed. No training is done in this file."""
    def _branch(state: CSPState):
        try:
            g = organ.guidance(state)
            logits = g["survival_logits"]                       # [n, K] per-(cell,value) survival logit
        except Exception:
            return fallback(state)
        open_cells = [i for i in range(state.csp.n) if len(state.dom[i]) > 1]
        if not open_cells:
            return fallback(state)

        def decisiveness(i):
            vals = sorted(state.dom[i])
            ls = [float(logits[i][v]) for v in vals if v < len(logits[i])]
            return (max(ls) - min(ls)) if ls else -1.0      # how strongly the proposer prefers a value

        cell = max(open_cells, key=decisiveness)
        vals = sorted(state.dom[cell],
                      key=lambda v: -(float(logits[cell][v]) if v < len(logits[cell]) else 0.0))
        return cell, vals
    return _branch


# ============================================================ stats
@dataclass
class SearchStats:
    nodes: int = 0              # search-tree nodes visited (each = one NARROW + classify)
    branches: int = 0           # BRANCH decisions made (open nodes expanded; 0 ⇒ deduction solved it)
    deduction_calls: int = 0    # reduced_product invocations (== nodes; the inner-loop call count)
    backtracks: int = 0         # subtrees that returned failure (conflict / exhausted / failed check)
    max_depth: int = 0          # deepest branch level reached
    limit_hit: bool = False     # max_nodes bound was reached (search returned without exhausting)

    def as_dict(self):
        return {"nodes": self.nodes, "branches": self.branches,
                "deduction_calls": self.deduction_calls, "backtracks": self.backtracks,
                "max_depth": self.max_depth, "limit_hit": self.limit_hit}


# ============================================================ the DPLL outer loop
def solve(state: CSPState, reductions, branch: Branch = most_constrained_branch,
          max_nodes: int = 20000, gate: str = "exact"):
    """The DPLL SEARCH leg: NARROW (inner dedₚ loop) → BRANCH when undetermined → BACKTRACK on
    conflict → OUTPUT-CHECK the assignment. Restores the LDT's outer loop around the existing inner
    deduction (clair.organ.compose.reduced_product).

    Args:
      state       : the initial CSPState (full grid, optionally carrying a `system` so a certified
                    faculty — GF2 / Modular — can fire inside the narrow step).
      reductions  : the inner-loop reductions (e.g. clair.organ.bank.certified_csp_reductions(bank),
                    or that + a verifier-gated neural organ). Passed straight to reduced_product.
      branch      : the pluggable branch heuristic (default most_constrained_branch; a learned hook
                    can be supplied via make_guided_branch — soundness is independent of it).
      max_nodes   : completeness bound. Full backtracking up to this many nodes; if hit, returns
                    (None, stats with limit_hit=True) — an honest ABSTAIN, never a wrong answer.
      gate        : the per-step verifier gate forwarded to reduced_product (default 'exact').

    Returns (solution | None, stats). SOUND by construction: a non-None return ALWAYS passes
    check_assignment. COMPLETE up to max_nodes: if a solution exists and the tree fits the bound it
    is found. On UNSAT it returns (None, stats) with limit_hit=False (exhausted, certified no-solution
    relative to the input CSP)."""
    stats = SearchStats()

    def narrow(s: CSPState) -> CSPState:
        stats.nodes += 1
        stats.deduction_calls += 1
        # the INNER deduction loop, reused verbatim. verify=False: we OUTPUT-CHECK instead of asserting
        # the exact-oracle invariant per node (and the inner loop may include an unsound neural organ —
        # the gate + the final check keep us sound either way).
        out, _tr = reduced_product(s, reductions, verify=False, gate=gate)
        return out

    def rec(s: CSPState, depth: int) -> Optional[tuple]:
        stats.max_depth = max(stats.max_depth, depth)
        if stats.nodes >= max_nodes:
            stats.limit_hit = True
            return None
        cur = narrow(s)
        st = cur.status()
        if st == "conflict":                                     # empty domain → dead branch
            stats.backtracks += 1
            return None
        if st == "solved":                                       # all singletons → OUTPUT-CHECK
            assign = _singleton_assignment(cur.dom)
            if check_assignment(cur.csp, assign):
                return assign                                    # sound: returned iff it checks
            stats.backtracks += 1                                # guards an unsound reduction/branch
            return None
        # UNDETERMINED: branch (most-constrained-cell / smallest-domain-first by default)
        cell, values = branch(cur)
        stats.branches += 1
        for v in values:
            child = cur.with_dom(tuple(frozenset({v}) if i == cell else cur.dom[i]
                                       for i in range(cur.csp.n)))
            res = rec(child, depth + 1)
            if res is not None:
                return res
        stats.backtracks += 1                                    # all values failed → backtrack
        return None

    sol = rec(state, 0)
    return sol, stats


# ============================================================ a convenience: deduction-only vs +search
def deduction_only(state: CSPState, reductions, gate: str = "exact"):
    """Run JUST the inner deduction loop (no search) and report its outcome — the GLaDOS-as-shipped
    behaviour (deduction + certified-ops + ABSTAIN). Returns (status, narrowed_state)."""
    out, _tr = reduced_product(state, reductions, verify=False, gate=gate)
    return out.status(), out


# ============================================================ self-test (the new search check)
def selftest(verbose: bool = True) -> bool:
    """The restored-leg proof, importable by clair.organ.selftest:
      (1) on the XOR system where DEDUCTION ABSTAINS (acStep_xor_sound_but_abstains), deduction+SEARCH
          SOLVES it — and the returned assignment passes the exact output check (sound);
      (2) the BRANCH-COUNT contrast: bounded-width (chain/coloring) solves with ~0 branches (deduction
          does it, search adds nothing); affine WITHOUT the GF2 faculty needs real branching; affine
          WITH the GF2 faculty solves with ~0 branches (the linear-algebra route) — both LDT routes;
      (3) soundness: on a random instance suite every returned solution checks, and search never
          contradicts the exact solver (solves ⟺ the CSP is satisfiable)."""
    import numpy as np
    from . import bank as B
    from .. import xor_wall as XW

    bank = B.build_bank(load_neural=False)
    certified = B.certified_csp_reductions(bank)                  # AC, factor, modular, GF2, macro
    # the DEDUCTION-ONLY (per-cell/factor) leg: drop the system-keyed linalg/path faculties so the
    # affine coupling is invisible to local deduction (it must abstain — the dichotomy wall).
    ded_local = [r for r in certified if r.name not in ("gf2_rowspace", "modular_snf", "macro_reach")]

    p = lambda *a: (print(*a) if verbose else None)
    p("\n[search] DPLL outer loop — the restored LDT search leg")

    # (1) the canonical contrast: the tiny affine xor_parity (unique sol (0,0,0)) AC abstains on.
    xr = C.xor_parity()
    sx = CSPState.full(xr)
    ded_status, _ = deduction_only(sx, [bank["arc_consistency"]])
    assert ded_status == "open", "per-cell AC must ABSTAIN on affine xor_parity (the wall)"
    sol, stats = solve(sx, [bank["arc_consistency"]])
    assert sol is not None and check_assignment(xr, sol) and sol == (0, 0, 0), \
        f"deduction+search must SOLVE xor_parity the AC leg abstains on (got {sol})"
    p(f"    xor_parity: AC abstains (open) → AC+SEARCH solves {sol} "
      f"(nodes={stats.nodes}, branches={stats.branches}); output-checked sound")

    # (2) a WIDE global-coupling XOR system where even level-2 factor consistency abstains:
    #     deduction-only abstains; deduction+search solves by BRANCHING; deduction+GF2 solves ~0 branches.
    rng = np.random.default_rng(20260628)
    found = None
    for _ in range(200):
        d = XW.gen_xor_system(rng, 8, 8, band=8)                 # global high-treewidth affine
        st = CSPState.full(d["csp"])
        if deduction_only(st, ded_local)[0] == "open":           # an instance the local leg stalls on
            found = d
            break
    assert found is not None, "expected a wide XOR instance where AC+factor abstains"
    st_noxor = CSPState.full(found["csp"])
    st_gf2 = CSPState.full(found["csp"], system=("gf2", found["A"], found["b"]), tags={"xor"})

    sol_s, stats_s = solve(st_noxor, ded_local)                  # the BRANCHING route
    sol_g, stats_g = solve(st_gf2, certified)                    # the LINEAR-ALGEBRA route
    assert sol_s is not None and check_assignment(found["csp"], sol_s), "search must solve the affine wall"
    assert tuple(sol_s) == tuple(int(x) for x in found["s"]), "search must recover the unique witness"
    assert sol_g is not None and check_assignment(found["csp"], sol_g), "GF2+search must solve too"
    assert stats_s.branches > 0, "the affine wall must require REAL branching without GF2"
    assert stats_g.branches == 0, "WITH the GF2 faculty the affine wall solves with ~0 branches (linalg)"
    p(f"    wide XOR (n=8, global): deduction-only ABSTAINS; "
      f"+SEARCH solves with branching (branches={stats_s.branches}, nodes={stats_s.nodes}); "
      f"+GF2 solves with branches={stats_g.branches} (linear-algebra route)")

    # bounded-width control: a determined chain / coloring solves with ~0 branches (deduction does it).
    ch = C.chain_eq(6)
    _, st_ch = deduction_only(CSPState.full(ch), [bank["arc_consistency"]])
    sol_ch, stats_ch = solve(CSPState.full(ch), [bank["arc_consistency"]])
    assert sol_ch is not None and stats_ch.branches == 0, "bounded-width chain solves with 0 branches"
    p(f"    chain_eq(6) [bounded-width]: solves with branches={stats_ch.branches} (deduction alone)")

    # (3) soundness + completeness on a random suite vs the exact solver.
    rng = np.random.default_rng(7)
    bad = 0; checked = 0
    for _ in range(60):
        n = int(rng.integers(5, 9))
        d = XW.gen_xor_system(rng, n, n, band=n)
        st = CSPState.full(d["csp"])
        sol, _stats = solve(st, ded_local)
        sat = len(C.solutions(d["csp"], limit=1)) > 0
        if sol is not None:
            checked += 1
            if not check_assignment(d["csp"], sol):
                bad += 1
        # complete: search solves ⟺ satisfiable (these are constructed satisfiable)
        if sat and sol is None:
            bad += 1
    assert bad == 0, f"search returned an UNSOUND or INCOMPLETE result on {bad}/60 random XOR instances"
    p(f"    random suite: 60/60 XOR instances solved+output-checked sound (0 unsound, 0 missed)")
    p("    [search] PASS — the DPLL outer loop reproduces the LDT two-loop Solve (deduction prunes; "
      "branching crosses the affine wall; the output check keeps it sound)")
    return True


if __name__ == "__main__":
    selftest(verbose=True)
