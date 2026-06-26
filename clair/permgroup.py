"""Exact PERMUTATION-GROUP harness — the GROUND TRUTH for the permutation-group organ.

Pure Python + sympy.combinatorics (Schreier-Sims) for the certified group facts. Everything here is
exact / checkable, so any neural proposer trained later is measured against THIS. This is the
group-theory dual of `clair.csp`: where csp.py narrows per-cell value-domains under arbitrary finite
constraints, here the global structure is the SYMMETRIC GROUP S_n (the unknown is a *permutation*
σ ∈ S_n, i.e. a bijection), and the certified primitives are GROUP / matching algorithms.

THE TASK (a permutation-reasoning problem). An unknown σ ∈ S_n. Constraints:
  * unary      A_i ⊆ {0..n-1} : σ(i) ∈ A_i        (fixed point σ(i)=i ⟺ A_i={i}; forbidden image j ⟺ j∉A_i)
  * ALLDIFFERENT (global)      : σ is a bijection   — THE symmetric-group structure (always on)
  * ordering   (i,j)           : σ(i) < σ(j)        (image-ordering / interval constraint, optional)
  * subgroup   ⟨gens⟩ ≤ S_n    : σ ∈ the generated subgroup (Schreier-Sims membership, optional)

ABSTRACT STATE = per-point candidate-image domains dom[i] ⊆ {0..n-1} (the still-possible images of i).
A point is *solved* at |dom[i]|=1, *conflict* at |dom[i]|=0. We only MEET (narrow); a surviving
singleton is the answer. This is the powerset lattice of csp.py, with the AllDifferent (S_n) global
constraint making it a permutation problem.

CERTIFIED SOUND OPERATORS (sound by construction — the "group algorithm primitives"):
  unary_ac      dom[i] ← dom[i] ∩ A_i.                                            trivially sound
  alldiff_gac   Régin/Hall matching GAC: keep v at i iff some perfect matching     sound & COMPLETE
                of (points→values, edges=dom) uses edge (i,v).                     for one AllDifferent
  ordering_ac   bounds propagation for σ(i)<σ(j).                                  sound
  group_orbit   dom[i] ← dom[i] ∩ orbit_G(i)  (σ(i) ∈ orbit of i under G always).  sound (group fact)

EXACT GROUND TRUTH:
  brute(prob)        all σ ∈ S_n satisfying every constraint (backtracking; n≤9 fine).
  exact_dedP(prob,d) per-point survivors over ALL consistent σ — the strongest SOUND per-point
                     narrowing (the target the neural proposer must DOMINATE, never drop a survivor).

Run:  python -m clair.permgroup        # self-check: certified ops match sympy/brute, soundly
"""
from __future__ import annotations

import itertools as it
from dataclasses import dataclass, field

# sympy only needed for the subgroup (group-membership / orbit) facts; imported lazily so the
# matching/AllDifferent core runs with no sympy installed.


# --------------------------------------------------------------------------- the problem
@dataclass(frozen=True)
class PermProblem:
    n: int
    unary: tuple                  # tuple of frozenset, unary[i] = A_i ⊆ {0..n-1}
    order: tuple = ()             # tuple of (i, j) meaning σ(i) < σ(j)
    gens: tuple = ()              # tuple of permutations (each an n-tuple, images) generating G ≤ S_n
    _group_cache: dict = field(default_factory=dict, compare=False, hash=False)

    def full(self):
        V = frozenset(range(self.n))
        return tuple(V for _ in range(self.n))

    # ----- group facts (sympy Schreier-Sims), memoized -----
    def _group(self):
        if "G" not in self._group_cache:
            from sympy.combinatorics import Permutation, PermutationGroup
            if not self.gens:
                self._group_cache["G"] = None
            else:
                G = PermutationGroup([Permutation(list(g)) for g in self.gens])
                self._group_cache["G"] = G
        return self._group_cache["G"]

    def orbit(self, i):
        """orbit_G(i): the points σ can send i to, for σ ∈ G. Sound upper bound on dom[i]."""
        G = self._group()
        if G is None:
            return frozenset(range(self.n))
        if "orbits" not in self._group_cache:
            self._group_cache["orbits"] = {p: frozenset(G.orbit(p)) for p in range(self.n)}
        return self._group_cache["orbits"][i]

    def in_group(self, perm):
        """Exact Schreier-Sims membership: is `perm` (an n-tuple) an element of G?"""
        G = self._group()
        if G is None:
            return True
        from sympy.combinatorics import Permutation
        return G.contains(Permutation(list(perm)))


# --------------------------------------------------------------------------- bipartite matching (Hall/Régin core)
def _perfect_matching(dom, n):
    """Does a system of distinct representatives exist? i.e. a perfect matching of the bipartite graph
    (points i → values in dom[i]). Returns the matching (point→value) as a list, or None. Augmenting-path
    Kuhn's algorithm — exact. This is the Hall-condition decision the AllDifferent GAC is built on."""
    match_val = [-1] * n          # value -> point
    match_pt = [-1] * n           # point -> value

    def aug(i, seen):
        for v in dom[i]:
            if not seen[v]:
                seen[v] = True
                if match_val[v] == -1 or aug(match_val[v], seen):
                    match_val[v] = i
                    match_pt[i] = v
                    return True
        return False

    for i in range(n):
        if not dom[i]:
            return None
        if not aug(i, [False] * n):
            return None
    return match_pt


# --------------------------------------------------------------------------- certified sound operators
def unary_ac(prob: PermProblem, dom):
    """dom[i] ← dom[i] ∩ A_i. Trivially sound."""
    return tuple(dom[i] & prob.unary[i] for i in range(prob.n))


def group_orbit(prob: PermProblem, dom):
    """dom[i] ← dom[i] ∩ orbit_G(i). Sound: σ(i) ∈ orbit_G(i) for every σ ∈ G."""
    if not prob.gens:
        return dom
    return tuple(dom[i] & prob.orbit(i) for i in range(prob.n))


def alldiff_gac(prob: PermProblem, dom):
    """Régin/Hall GAC for AllDifferent: keep value v at point i iff SOME perfect matching of the
    bipartite (points→values, edges=dom) graph uses edge (i,v). Sound (no valid permutation uses a
    value with no perfect matching) AND complete for a single AllDifferent (== dedₚ restricted to the
    bijection constraint). For small n we test each candidate edge directly by forcing i→v and checking
    a perfect matching still exists — manifestly the Régin characterization, obviously correct."""
    n = prob.n
    # global feasibility first
    if _perfect_matching(dom, n) is None:
        return tuple(frozenset() for _ in range(n))         # ⊥ : no permutation at all
    new = []
    for i in range(n):
        keep = set()
        for v in dom[i]:
            forced = [dom[k] - {v} if k != i else frozenset({v}) for k in range(n)]
            if _perfect_matching(forced, n) is not None:
                keep.add(v)
        new.append(frozenset(keep))
    return tuple(new)


def ordering_ac(prob: PermProblem, dom):
    """Bounds propagation for each σ(i) < σ(j): σ(i) must be < some achievable σ(j), and σ(j) > some
    achievable σ(i). Drop v at i if v ≥ max(dom[j]); drop v at j if v ≤ min(dom[i]). Sound."""
    new = list(dom)
    for (i, j) in prob.order:
        if not new[i] or not new[j]:
            continue
        hi_j = max(new[j]); lo_i = min(new[i])
        new[i] = frozenset(v for v in new[i] if v < hi_j)
        new[j] = frozenset(v for v in new[j] if v > lo_i)
    return tuple(new)


def group_pairs(prob: PermProblem):
    """For each ordered pair (i,j), the set of jointly-realizable image-pairs {(g(i),g(j)) : g ∈ G}.
    This is the PAIRWISE group marginal — the level-1 group structure (analog of csp.py's pair lattice).
    Memoized on the problem."""
    if "pairs" not in prob._group_cache:
        n = prob.n
        if not prob.gens:
            prob._group_cache["pairs"] = None
        else:
            G = close_group(prob.gens, n)
            pr = {}
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    pr[(i, j)] = frozenset((g[i], g[j]) for g in G)
            prob._group_cache["pairs"] = pr
    return prob._group_cache["pairs"]


def group_pair_ac(prob: PermProblem, dom):
    """CERTIFIED level-1 group reduction (pair-consistency on the group): keep value v at point i iff
    for EVERY other point j there is a value w ∈ dom[j] with (v,w) jointly realizable in G. Sound: any
    σ∈G consistent with dom realizes (σ(i),σ(j)) ∈ group_pairs, so a value with no pairwise support is
    used by no valid permutation. Strictly stronger than group_orbit (the unary group marginal); still
    cannot see 3-point group structure (the residual that needs full Schreier-Sims — the KR-irreducible
    part). This is the group analog of csp.path-consistency."""
    pr = group_pairs(prob)
    if pr is None:
        return dom
    n = prob.n
    new = []
    for i in range(n):
        keep = set()
        for v in dom[i]:
            ok = True
            for j in range(n):
                if j == i:
                    continue
                if not any((v, w) in pr[(i, j)] for w in dom[j]):
                    ok = False; break
            if ok:
                keep.add(v)
        new.append(frozenset(keep))
    return tuple(new)


def certified_step(prob: PermProblem, dom):
    """One pass of every LOCAL certified operator (unary ∧ group-orbit ∧ ordering ∧ AllDifferent-GAC).
    Reductive and sound: a_next ⊆ a, and never drops a value used by a valid permutation."""
    dom = unary_ac(prob, dom)
    dom = group_orbit(prob, dom)
    dom = ordering_ac(prob, dom)
    dom = alldiff_gac(prob, dom)
    return dom


def strong_step(prob: PermProblem, dom):
    """certified_step + the level-1 group pair-consistency reduction. Still sound by construction."""
    dom = certified_step(prob, dom)
    dom = group_pair_ac(prob, dom)
    dom = alldiff_gac(prob, dom)             # re-tighten AllDifferent after the group reduction
    return dom


def to_fixpoint(step, prob, dom=None, max_iters=None):
    dom = dom or prob.full()
    max_iters = max_iters or prob.n * prob.n + 2
    for t in range(1, max_iters + 1):
        nxt = step(prob, dom)
        if nxt == dom:
            return dom, t
        dom = nxt
    return dom, max_iters


# --------------------------------------------------------------------------- exact ground truth
class _Stop(Exception):
    pass


def _enumerate(prob: PermProblem, dom, on_solution):
    """Backtracking over points: assign σ(i) ∈ dom[i], distinct (AllDifferent), respecting unary,
    ordering, and (at full assignment) subgroup membership. on_solution(tuple) may raise _Stop."""
    n = prob.n
    order = sorted(range(n), key=lambda i: len(dom[i]))          # smallest domain first
    assign = [None] * n
    used = [False] * n
    # ordering edges grouped by the later-assigned endpoint among (i,j)
    rank = {c: k for k, c in enumerate(order)}
    ord_checks = {k: [] for k in range(n)}
    for (i, j) in prob.order:
        ord_checks[max(rank[i], rank[j])].append((i, j))

    def ok_order(k):
        for (i, j) in ord_checks[k]:
            if assign[i] is not None and assign[j] is not None and not (assign[i] < assign[j]):
                return False
        return True

    def rec(k):
        if k == n:
            perm = tuple(assign)
            if prob.gens and not prob.in_group(perm):
                return
            on_solution(perm)
            return
        cell = order[k]
        for v in sorted(dom[cell]):
            if used[v] or v not in prob.unary[cell]:
                continue
            assign[cell] = v; used[v] = True
            if ok_order(k):
                rec(k + 1)
            assign[cell] = None; used[v] = False

    try:
        rec(0)
    except _Stop:
        pass


def brute(prob: PermProblem, dom=None, limit=None):
    """All σ ∈ S_n consistent with `dom` and every constraint. Exact ground truth (n ≤ 9)."""
    dom = dom or prob.full()
    out = []

    def collect(s):
        out.append(s)
        if limit and len(out) >= limit:
            raise _Stop

    _enumerate(prob, dom, collect)
    return out


def exact_dedP(prob: PermProblem, dom=None):
    """dedₚ[i] = {σ(i) : σ consistent with dom and every constraint}. The strongest SOUND per-point
    narrowing — the target the neural proposer must DOMINATE. ⊥ (all empty) if unsatisfiable."""
    dom = dom or prob.full()
    n = prob.n
    surv = [set() for _ in range(n)]
    need = sum(len(d) for d in dom)
    seen = [0]

    def witness(s):
        for i in range(n):
            if s[i] not in surv[i]:
                surv[i].add(s[i]); seen[0] += 1
        if seen[0] >= need:
            raise _Stop

    _enumerate(prob, dom, witness)
    if seen[0] == 0:
        return tuple(frozenset() for _ in range(n))
    return tuple(frozenset(surv[i]) for i in range(n))


# --------------------------------------------------------------------------- outcomes / metrics
def status(dom):
    if any(len(c) == 0 for c in dom):
        return "conflict"
    if all(len(c) == 1 for c in dom):
        return "solved"
    return "open"


def false_elim(dom, exact):
    """# (i,v) the operator dropped that some valid permutation actually uses. MUST be 0 (soundness)."""
    return sum(len(exact[i] - dom[i]) for i in range(len(dom)))


def solve(prob: PermProblem, step=certified_step):
    """Run a certified operator to fixpoint from the full grid; report outcome + soundness vs exact."""
    dom, steps = to_fixpoint(step, prob)
    ex = exact_dedP(prob)
    sols = brute(prob, limit=1)
    fe = false_elim(dom, ex) if sols else 0
    return {
        "outcome": status(dom), "steps": steps,
        "solvable": len(sols) > 0,
        "false_elim_vs_exact": fe,                       # 0 = sound
        "final_alive": tuple(len(c) for c in dom),
        "exact_alive": tuple(len(c) for c in ex),        # what dedₚ reaches
        "matches_exact": dom == ex,                      # certified fixpoint == strongest sound narrowing?
    }


# --------------------------------------------------------------------------- generators
def _A(n, allowed):
    return frozenset(v for v in range(n) if v in allowed)


def fixed_points(n, fixes):
    """σ(i)=fixes[i] where given; else free. fixes: dict i->j (a partial mapping requirement)."""
    unary = tuple(frozenset({fixes[i]}) if i in fixes else frozenset(range(n)) for i in range(n))
    return PermProblem(n, unary)


def forbidden(n, forb):
    """σ(i) ≠ j for (i,j) in forb. The 'no-fixed-point / derangement-ish' family."""
    bad = {}
    for (i, j) in forb:
        bad.setdefault(i, set()).add(j)
    unary = tuple(frozenset(v for v in range(n) if v not in bad.get(i, set())) for i in range(n))
    return PermProblem(n, unary)


def n_queens_perm(n):
    """The permutation core of N-queens: σ a permutation (one queen per row/col) with the diagonal
    constraints σ(i)-σ(j) ≠ ±(i-j) expressed as forbidden images per pair — but those are BINARY, so
    here we only plant the AllDifferent + a few unary clues; the real n-queens lives in csp/domains."""
    return PermProblem(n, tuple(frozenset(range(n)) for _ in range(n)))


GENERATORS = {"fixed_points": fixed_points, "forbidden": forbidden}


# --------------------------------------------------------------------------- group closure (pure-python BFS)
def close_group(gens, n):
    """Enumerate the subgroup ⟨gens⟩ ≤ S_n by BFS closure under composition (no sympy needed for the
    hot loop). Returns a frozenset of permutation tuples. Small groups only (|G| ≤ a few thousand)."""
    ident = tuple(range(n))
    elems = {ident}
    frontier = [ident]
    gl = list(gens)
    while frontier:
        nxt = []
        for x in frontier:
            for g in gl:
                y = tuple(g[x[i]] for i in range(n))     # x then g
                if y not in elems:
                    elems.add(y); nxt.append(y)
        frontier = nxt
    return frozenset(elems)


def group_prior(gens, n):
    """P[i][v] = fraction of group elements g with g(i)=v — the subgroup's first-order marginal, an
    exactly-computable feature for the neural organ (the certified orbit bound is its support)."""
    G = close_group(gens, n) if gens else None
    if G is None:
        return [[1.0 / n] * n for _ in range(n)]
    m = len(G)
    P = [[0.0] * n for _ in range(n)]
    for g in G:
        for i in range(n):
            P[i][g[i]] += 1.0 / m
    return P


# --------------------------------------------------------------------------- automata-shortcut / Krohn-Rhodes
def compose(g, h):
    """Group/monoid product (g then h):  (h∘g)(i) = h(g(i)).  Returns the composite permutation tuple.
    This is the transition-monoid product — the operator the automata 'shortcut' parallel-scans."""
    return tuple(h[g[i]] for i in range(len(g)))


def prefix_product_sequential(seq, n):
    """State trajectory of a permutation automaton: fold compose left-to-right (the O(T) step-by-step
    simulation). seq = list of permutation tuples; returns the running composite after each step."""
    cur = tuple(range(n))
    out = []
    for g in seq:
        cur = compose(cur, g)
        out.append(cur)
    return out


def prefix_product_scan(seq, n):
    """The SHORTCUT: the same trajectory by an associative parallel scan (Hillis-Steele), O(log T)
    DEPTH instead of O(T). Composition is associative and the group is finite (bounded state), so the
    prefix products can be computed in parallel — this is exactly the Krohn-Rhodes / 'Transformers learn
    shortcuts to automata' fact (a group automaton needs only O(log T) layers, not O(T))."""
    cur = list(seq)
    if not cur:
        return []
    depth = 0
    shift = 1
    T = len(cur)
    while shift < T:
        nxt = list(cur)
        for i in range(T):
            if i - shift >= 0:
                nxt[i] = compose(cur[i - shift], cur[i])    # cur[i] already = product over (i-shift, i]
        cur = nxt
        shift *= 2
        depth += 1
    return cur, depth


# --------------------------------------------------------------------------- self-check
if __name__ == "__main__":
    print("exact permutation-group harness self-check\n")

    # ---- AllDifferent GAC is exact (== dedₚ) for a pure bijection problem ----
    n = 5
    p = n_queens_perm(n)
    dom, steps = to_fixpoint(certified_step, p)
    ex = exact_dedP(p)
    print(f"pure S_{n}:  certified alive={tuple(len(c) for c in dom)}  dedP alive={tuple(len(c) for c in ex)}  "
          f"match={dom == ex}  #perms={len(brute(p))} (=120=5!)")
    assert dom == ex and dom == p.full(), "no constraints -> nothing narrows, dedP == full"
    assert len(brute(p)) == 120

    # ---- a partial mapping forces the rest via AllDifferent (Hall) ----
    p = fixed_points(5, {0: 2, 1: 3})            # σ(0)=2, σ(1)=3
    r = solve(p)
    print(f"\nσ(0)=2,σ(1)=3 in S_5:  outcome={r['outcome']}  alive={r['final_alive']}  "
          f"exact={r['exact_alive']}  FE={r['false_elim_vs_exact']}  #perms={len(brute(p))} (=3!=6)")
    assert r["false_elim_vs_exact"] == 0 and len(brute(p)) == 6
    # points 2,3,4 can only map to {0,1,4} (the unused values) — AllDifferent GAC should reflect that
    dom, _ = to_fixpoint(certified_step, p)
    assert dom[2] == dom[3] == dom[4] == frozenset({0, 1, 4}), dom

    # ---- ordering: σ(0)<σ(1)<σ(2) in S_3 has a unique solution (0,1,2) ----
    p = PermProblem(3, tuple(frozenset(range(3)) for _ in range(3)), order=((0, 1), (1, 2)))
    r = solve(p)
    print(f"\nσ(0)<σ(1)<σ(2) in S_3:  outcome={r['outcome']}  alive={r['final_alive']}  "
          f"matches_dedP={r['matches_exact']}  #perms={len(brute(p))}")
    assert len(brute(p)) == 1 and brute(p)[0] == (0, 1, 2)
    # is the certified local fixpoint COMPLETE here, or does it abstain (the research gap)?
    print(f"   certified outcome={r['outcome']}  (dedP solves: exact_alive={r['exact_alive']})")

    # ---- subgroup: σ ∈ ⟨(0 1 2 3)⟩ (cyclic C_4) — only the 4 rotations ----
    rot = (1, 2, 3, 0)                                       # the 4-cycle 0->1->2->3->0
    p = PermProblem(4, tuple(frozenset(range(4)) for _ in range(4)), gens=(rot,))
    sols = brute(p)
    print(f"\nσ ∈ ⟨(0123)⟩ ≤ S_4:  #perms={len(sols)} (=|C_4|=4)  in_group(id)={p.in_group((0,1,2,3))}  "
          f"in_group(swap01)={p.in_group((1,0,2,3))}")
    assert len(sols) == 4 and p.in_group((0, 1, 2, 3)) and not p.in_group((1, 0, 2, 3))
    # group-orbit narrowing: orbit of 0 under C_4 is all of {0,1,2,3}, so adding σ(0)=0 forces identity
    p2 = PermProblem(4, (frozenset({0}),) + tuple(frozenset(range(4)) for _ in range(3)), gens=(rot,))
    print(f"   ∩ σ(0)=0:  #perms={len(brute(p2))} (=1, only identity in C_4 fixes 0)")
    assert len(brute(p2)) == 1 and brute(p2)[0] == (0, 1, 2, 3)

    # ---- automata shortcut: parallel scan == sequential prefix product ----
    import random
    rng = random.Random(0)
    n = 6
    def randperm():
        p = list(range(n)); rng.shuffle(p); return tuple(p)
    seq = [randperm() for _ in range(13)]
    seq_traj = prefix_product_sequential(seq, n)
    scan_final, depth = prefix_product_scan(seq, n)
    print(f"\nautomata shortcut (S_{n}, T={len(seq)}):  scan depth={depth} (=ceil(log2 T))  "
          f"sequential steps={len(seq)}")
    assert scan_final == seq_traj, "parallel scan must reproduce the exact prefix-product trajectory"
    assert depth == (len(seq) - 1).bit_length(), "scan depth is O(log T) — the shortcut"
    print(f"   final state matches: {scan_final[-1] == seq_traj[-1]}  (group composition is associative)")

    print("\nALL CHECKS PASS — certified group/matching primitives are exact + sound vs sympy/brute.")
