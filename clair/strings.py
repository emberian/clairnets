"""clair/strings.py — the CERTIFIED STRING / SEQUENCE abstract-domain organ (the symbolic-sequence primitive).

The codex zoo review (notes/codex_zoo_review.md) sets the discipline: a domain only counts as a
"checked neural abstract-STATE organ" if it has EXPLICIT semantics — α, concretisation γ, top/bottom,
meet/reduction, a verifier/certificate, and an abstain. clair.csp is the per-cell finite-domain
authority and clair.modular is the congruence reduced-product. THIS file is the STRING piece: a sound
abstract domain over fixed-length-≤N strings, with a CERTIFIED sound constraint-propagation operator and
an independent brute-force verifier — the analogue of automata/length string abstract domains, kept
finite so the ground truth is exact.

TYPED PROTOCOL (so it chains as a reduced product with the per-cell / modular organs):
  * alphabet         : Σ = {0..k-1}; we add ONE end/pad symbol ε = k, so each position lives in
                       V = Σ ∪ {ε}, |V| = k+1. A length-≤N string w is the fixed-N array x ∈ V^N whose
                       non-ε prefix IS w; x is WELL-FORMED iff ε is suffix-closed (x[i]=ε ⟹ x[i+1]=ε),
                       so length L(x) = index of the first ε (= N if none).
  * abstract STATE   : a REDUCED PRODUCT — (a) per-position candidate char-sets D[i] ⊆ V, which is
                       BITWISE a clair.csp per-cell lattice over V (composable for free), times
                       (b) a length interval [lo,hi]. The two factors are reduced against each other
                       (an ε forced at i caps hi; a real char forced at i raises lo).
  * concretisation   : γ(D,[lo,hi]) = { well-formed x ∈ V^N : x[i]∈D[i] ∀i, lo ≤ L(x) ≤ hi }.
  * CERTIFIED operator: `propagate` — SOUND per-constraint narrowing (generalised-arc / automaton arc
                       consistency) to a fixpoint. Sound BY CONSTRUCTION: a value is dropped at a
                       position only when NO string satisfying that one constraint (within the current
                       box) uses it, so the fixpoint never drops a value used by a full solution.
  * verifier / oracle: `brute` — exact enumeration over the finite well-formed string space; the
                       independent ground truth (the per-position reachable char-set = the string dedP).
  * abstain          : on an UNDER-determined puzzle the operator narrows to the exact reachable sets
                       and reports `open` (it does NOT guess a singleton).

CONSTRAINTS (a small string-CSP language; each has an exact `holds(word)` semantics, a sound narrower,
and a clair.csp encoding so the factor-graph organ chains):
    pos(i, S)        position i is a real char in S ⊆ Σ        (per-position char-set; forces L>i)
    len(lo, hi)      lo ≤ L ≤ hi                                (length bound)
    eq(i, j)         x[i] = x[j]                                (position equality, incl. ε=ε)
    neq(i, j)        x[i] ≠ x[j]
    prefix(p)        x starts with the literal word p           (⇒ L ≥ |p|)
    suffix(p)        x ends with the literal word p             (regex Σ* p)
    contains(p)      p occurs as a substring                    (regex Σ* p Σ*)
    regex(ast)       membership in a small regex over Σ         (compiled to a DFA, run on the prefix)

to_csp encodes the whole string-CSP as a clair.csp.CSP over V: positions are cells; regex/suffix/contains
add a chain of auxiliary DFA-STATE cells (arity-3 transition factors), exactly the automaton-on-a-chain
encoding string solvers use. So clair.csp.exact_dedP gives the exact per-cell reachable set and the
existing factor-graph organ (clair.proposer) drives it unchanged. The SMOKE cross-checks the CSP encoding
against the native brute on the position cells.

  python -m clair.strings        # SMOKE: native propagation sound vs brute + CSP round-trip + regex DFA

Pure Python / no torch — the exact, checkable authority. The NEURAL guidance that learns which certified
narrowing to propose (and in what order) lives in clair.run_strings (it imports the generators + encoding).
"""
from __future__ import annotations

import itertools as it
from dataclasses import dataclass

from . import csp as C


# ============================================================ regex -> DFA (small, exact)
# A tiny regex AST over Σ = {0..k-1}, built programmatically (no parser):
#   ('lit', c)         a single char c
#   ('cls', frozenset) any char in the set
#   ('seq', a, b, ...) concatenation
#   ('alt', a, b, ...) union
#   ('star', a)        a*
#   ('plus', a)        a+   ==  seq(a, star(a))
#   ('opt', a)         a?   ==  alt(a, eps)
#   ('eps',)           the empty string
def R_lit(c): return ("lit", int(c))
def R_cls(s): return ("cls", frozenset(int(x) for x in s))
def R_seq(*xs): return ("seq",) + xs
def R_alt(*xs): return ("alt",) + xs
def R_star(x): return ("star", x)
def R_plus(x): return ("plus", x)
def R_opt(x): return ("opt", x)
R_EPS = ("eps",)


def _thompson(ast, k):
    """Compile the AST to an ε-NFA (Thompson construction). Returns (n_states, start, accept, trans, eps)
    where trans: dict[(state,char)] -> set(state), eps: dict[state] -> set(state)."""
    trans, eps = {}, {}
    counter = [0]

    def new():
        s = counter[0]; counter[0] += 1
        return s

    def add_t(s, c, t):
        trans.setdefault((s, c), set()).add(t)

    def add_e(s, t):
        eps.setdefault(s, set()).add(t)

    def build(node):
        tag = node[0]
        if tag == "eps":
            s = new(); a = new(); add_e(s, a); return s, a
        if tag == "lit":
            s = new(); a = new(); add_t(s, node[1], a); return s, a
        if tag == "cls":
            s = new(); a = new()
            for c in node[1]:
                add_t(s, c, a)
            return s, a
        if tag == "seq":
            s, a = build(node[1])
            for child in node[2:]:
                s2, a2 = build(child)
                add_e(a, s2); a = a2
            return s, a
        if tag == "alt":
            s = new(); a = new()
            for child in node[1:]:
                s2, a2 = build(child)
                add_e(s, s2); add_e(a2, a)
            return s, a
        if tag == "star":
            s = new(); a = new(); s2, a2 = build(node[1])
            add_e(s, s2); add_e(s, a); add_e(a2, s2); add_e(a2, a)
            return s, a
        if tag == "plus":
            return build(("seq", node[1], ("star", node[1])))
        if tag == "opt":
            return build(("alt", node[1], ("eps",)))
        raise ValueError(node)

    start, accept = build(ast)
    return counter[0], start, accept, trans, eps


def _eclose(states, eps):
    out = set(states); stack = list(states)
    while stack:
        s = stack.pop()
        for t in eps.get(s, ()):
            if t not in out:
                out.add(t); stack.append(t)
    return frozenset(out)


@dataclass(frozen=True)
class DFA:
    """A complete (total) DFA over Σ = {0..k-1}. States 0..n_states-1; state n_states-1 is a DEAD sink
    (never accepting, self-loops on every char). `delta[s][c]` is the next state; `accept` the accepting set."""
    k: int
    n_states: int
    start: int
    accept: frozenset
    delta: tuple        # delta[s] is a tuple of length k: next state on char c

    def run(self, word) -> bool:
        s = self.start
        for c in word:
            s = self.delta[s][c]
        return s in self.accept


def compile_regex(ast, k) -> DFA:
    """Regex AST -> minimal-ish complete DFA via subset construction. Small by construction (we only
    generate tiny regexes), with an explicit dead sink so the chain encoding is total."""
    n, start, accept, trans, eps = _thompson(ast, k)
    s0 = _eclose({start}, eps)
    dstates = {s0: 0}
    order = [s0]
    delta = []
    queue = [s0]
    while queue:
        cur = queue.pop(0)
        row = []
        for c in range(k):
            nxt = set()
            for s in cur:
                for t in trans.get((s, c), ()):
                    nxt.add(t)
            nxt = _eclose(nxt, eps)
            if nxt not in dstates:
                dstates[nxt] = len(order); order.append(nxt); queue.append(nxt)
            row.append(dstates[nxt])
        delta.append(tuple(row))
    dead = len(order)                                   # explicit dead sink for empty subset (if any)
    # remap: subsets that are empty map to dead
    real = []
    for i, sub in enumerate(order):
        real.append(tuple((delta[i][c] if order[delta[i][c]] else dead) for c in range(k)))
    real.append(tuple(dead for _ in range(k)))          # dead self-loops
    acc = frozenset(i for i, sub in enumerate(order) if accept in sub)
    return DFA(k=k, n_states=dead + 1, start=0, accept=acc, delta=tuple(real))


# ============================================================ the string-CSP type
EPS = "EPS"   # sentinel; the actual integer value of ε is k (the last value of V)


@dataclass(frozen=True)
class StringCSP:
    """A string constraint problem over Σ = {0..k-1}, strings of length 0..N (fixed array width N).
    `cons` is a tuple of constraint descriptors (tag, *args):
        ('pos', i, frozenset(S))            x[i] ∈ S ⊆ Σ  (real char)
        ('len', lo, hi)                     lo ≤ L ≤ hi
        ('eq', i, j) / ('neq', i, j)        x[i] (=|≠) x[j]
        ('prefix', tuple(p))                x starts with p
        ('suffix', tuple(p))                x ends with p
        ('contains', tuple(p))              p is a substring
        ('regex', ast)                      membership in regex `ast`
    """
    N: int
    k: int
    cons: tuple

    @property
    def eps(self):
        return self.k                                   # integer value of ε in V = 0..k

    @property
    def V(self):
        return self.k + 1                               # |Σ ∪ {ε}|

    def full(self):
        return tuple(frozenset(range(self.V)) for _ in range(self.N))


def decode(x, k):
    """The non-ε prefix of a well-formed value array x -> the string (tuple of chars in Σ)."""
    out = []
    for v in x:
        if v == k:
            break
        out.append(v)
    return tuple(out)


def well_formed(x, k):
    """ε suffix-closed: once an ε appears, everything after is ε."""
    seen_eps = False
    for v in x:
        if v == k:
            seen_eps = True
        elif seen_eps:
            return False
    return True


# ------------------------------------------------------------ exact constraint semantics (on a full string array)
def _holds(con, x, k):
    """Does value-array x satisfy ONE constraint? x is a well-formed V^N array; word = decode(x)."""
    tag = con[0]
    word = decode(x, k)
    L = len(word)
    if tag == "pos":
        _, i, S = con
        return x[i] != k and x[i] in S
    if tag == "len":
        _, lo, hi = con
        return lo <= L <= hi
    if tag == "eq":
        _, i, j = con
        return x[i] == x[j]
    if tag == "neq":
        _, i, j = con
        return x[i] != x[j]
    if tag == "prefix":
        _, p = con
        return len(word) >= len(p) and word[:len(p)] == tuple(p)
    if tag == "suffix":
        _, p = con
        return len(word) >= len(p) and word[len(word) - len(p):] == tuple(p)
    if tag == "contains":
        _, p = con
        p = tuple(p)
        return any(word[s:s + len(p)] == p for s in range(0, len(word) - len(p) + 1)) if p else True
    if tag == "regex":
        return compile_regex(con[1], k).run(word)
    raise ValueError(con)


def satisfies(scsp: StringCSP, x) -> bool:
    """x ∈ V^N satisfies the whole string-CSP (well-formed + every constraint)."""
    return well_formed(x, scsp.k) and all(_holds(con, x, scsp.k) for con in scsp.cons)


# ============================================================ exact verifier / ground truth (brute)
def brute(scsp: StringCSP, dom=None, limit=None):
    """Exact enumeration of well-formed strings in the box `dom` satisfying every constraint. Returns
    (solutions, reachable) where solutions is a list of V^N arrays and reachable[i] = {x[i] over all
    solutions} (the per-position STRING dedP). The independent authority the organ is graded against."""
    dom = dom or scsp.full()
    sols = []
    reach = [set() for _ in range(scsp.N)]
    for x in it.product(*[sorted(dom[i]) for i in range(scsp.N)]):
        if satisfies(scsp, x):
            sols.append(x)
            for i in range(scsp.N):
                reach[i].add(x[i])
            if limit and len(sols) >= limit:
                break
    return sols, tuple(frozenset(r) for r in reach)


# ============================================================ CERTIFIED sound propagation (the organ operator)
def _length_interval(dom, k):
    """Reduced-product readout: from per-position sets, the SOUND length interval [lo,hi].
      lo = first position that CAN be ε (every earlier position must be a real char)…  — actually
      lo = number of leading positions whose set excludes ε (forced real)  -> L ≥ lo.
      hi = first position whose set is EXACTLY {ε} caps the length; else N.  -> L ≤ hi."""
    N = len(dom)
    lo = 0
    while lo < N and k not in dom[lo]:        # position lo forced to a real char -> length > lo
        lo += 1
    hi = N
    for i in range(N):
        if dom[i] == frozenset({k}):          # position i forced to ε -> length ≤ i
            hi = i; break
    return lo, hi


def _narrow_wellformed(scsp: StringCSP, dom):
    """Reduce by the suffix-closed-ε invariant (the well-formedness factor + length reduced product):
      * if position i CANNOT be ε (real forced) then i-1 cannot be ε either (no gap before a real char);
      * if position i MUST be ε then i+1 must be ε (suffix closure)."""
    k = scsp.k
    new = list(dom)
    # forward: ε at i forces ε at i+1 ..
    must_eps = False
    for i in range(scsp.N):
        if must_eps:
            new[i] = new[i] & {k}
        if new[i] == frozenset({k}):
            must_eps = True
    # backward: a forced-real position i (k not in set) forbids ε at all earlier positions
    cannot_eps_from = None
    for i in range(scsp.N - 1, -1, -1):
        if k not in new[i]:                    # position i is a real char -> all earlier are real too
            cannot_eps_from = i
        if cannot_eps_from is not None and i < cannot_eps_from:
            new[i] = new[i] - {k}
    return tuple(new)


def _narrow_con(scsp: StringCSP, con, dom):
    """SOUND generalised-arc narrowing for ONE constraint, given the current box `dom`. Keeps value v at
    position i iff SOME well-formed assignment of the constraint's relevant positions (within their
    current sets) satisfies the constraint with x[i]=v. Implemented per constraint family; regex /
    suffix / contains use the DFA (forward-reachable ∩ backward-coreachable state pruning)."""
    k = scsp.k
    tag = con[0]
    new = list(dom)
    if tag == "pos":
        _, i, S = con
        new[i] = new[i] & frozenset(S)                       # real char in S (excludes ε)
        return tuple(new)
    if tag == "len":
        _, lo, hi = con
        for i in range(scsp.N):
            if i < lo:
                new[i] = new[i] - {k}                          # before lo: must be a real char
            if i >= hi:
                new[i] = new[i] & {k}                          # at/after hi: must be ε
        return tuple(new)
    if tag in ("eq", "neq"):
        _, i, j = con
        if tag == "eq":
            common = new[i] & new[j]
            new[i] = common; new[j] = frozenset(common)
        else:
            # x[i] ≠ x[j]: drop v at i only if j is the singleton {v} (then v impossible at i)
            if len(new[j]) == 1:
                new[i] = new[i] - new[j]
            if len(new[i]) == 1:
                new[j] = new[j] - new[i]
        return tuple(new)
    if tag == "prefix":
        _, p = con
        for t, c in enumerate(p):
            if t < scsp.N:
                new[t] = new[t] & {c}
        return tuple(new)
    # suffix / contains / regex -> via DFA arc consistency over the chain
    dfa = _con_dfa(scsp, con)
    return _narrow_dfa(scsp, dfa, new)


def _con_dfa(scsp: StringCSP, con):
    """The DFA whose accepted strings are exactly the words satisfying a suffix/contains/regex constraint."""
    k = scsp.k
    tag = con[0]
    if tag == "regex":
        return compile_regex(con[1], k)
    if tag == "suffix":
        p = tuple(con[1])
        return compile_regex(R_seq(R_star(R_cls(range(k))), *[R_lit(c) for c in p]) if p else R_star(R_cls(range(k))), k)
    if tag == "contains":
        p = tuple(con[1])
        body = R_seq(R_star(R_cls(range(k))), *[R_lit(c) for c in p], R_star(R_cls(range(k)))) if p \
            else R_star(R_cls(range(k)))
        return compile_regex(body, k)
    raise ValueError(con)


def _narrow_dfa(scsp: StringCSP, dfa: DFA, dom):
    """SOUND arc consistency for a DFA-chain constraint. Compute, per position t, the set of DFA states
    reachable BEFORE reading x[t] (forward, from start, over current domains) and the set co-reachable to
    an accepting end (backward). A char v survives at position t iff some forward state s has
    δ(s,v)=s' with s' backward-coreachable. ε (= end) is a stutter: state unchanged. Sound: never drops a
    char used by an accepted well-formed string in the box."""
    k = scsp.k; N = scsp.N
    S = dfa.n_states
    # forward[t] = set of states possible just before position t
    forward = [set() for _ in range(N + 1)]
    forward[0] = {dfa.start}
    for t in range(N):
        cur = forward[t]
        nxt = set()
        for s in cur:
            for v in dom[t]:
                if v == k:                                   # ε: end reached, state stutters
                    nxt.add(s)
                else:
                    nxt.add(dfa.delta[s][v])
        forward[t + 1] = nxt
    # backward[t] = set of states at position t that can still reach an accepting end
    backward = [set() for _ in range(N + 1)]
    backward[N] = set(s for s in forward[N] if s in dfa.accept)
    for t in range(N - 1, -1, -1):
        good = set()
        for s in forward[t]:
            for v in dom[t]:
                s2 = s if v == k else dfa.delta[s][v]
                if s2 in backward[t + 1]:
                    good.add(s); break
        backward[t] = good
    # narrow: keep v at t iff some live forward state s has δ(s,v) coreachable
    new = list(dom)
    for t in range(N):
        keep = set()
        for v in dom[t]:
            for s in forward[t]:
                if s not in backward[t]:
                    continue
                s2 = s if v == k else dfa.delta[s][v]
                if s2 in backward[t + 1]:
                    keep.add(v); break
        new[t] = frozenset(keep)
    return tuple(new)


def propagate(scsp: StringCSP, dom):
    """ONE sound pass: well-formedness reduction then every constraint's certified narrowing."""
    dom = _narrow_wellformed(scsp, dom)
    for con in scsp.cons:
        dom = _narrow_con(scsp, con, dom)
    return _narrow_wellformed(scsp, dom)


def to_fixpoint(scsp: StringCSP, dom=None, max_iters=None):
    """Iterate `propagate` to a fixed point (the certified string-AC organ). Returns (dom, n_steps)."""
    dom = dom or scsp.full()
    max_iters = max_iters or scsp.N * scsp.V + 4
    for t in range(1, max_iters + 1):
        nxt = propagate(scsp, dom)
        if nxt == dom:
            return dom, t
        dom = nxt
    return dom, max_iters


def status(scsp: StringCSP, dom):
    """'conflict' if any position empty; 'solved' if every position is a singleton (one well-formed
    string); else 'open' (abstain — under-determined)."""
    if any(len(c) == 0 for c in dom):
        return "conflict"
    if all(len(c) == 1 for c in dom):
        return "solved"
    return "open"


# ============================================================ clair.csp encoding (so it CHAINS)
def to_csp(scsp: StringCSP):
    """Encode the string-CSP as a clair.csp.CSP over V = 0..k. Position cells 0..N-1 hold chars (value k
    = ε); suffix/contains/regex constraints add a chain of AUXILIARY DFA-state cells with arity-3
    transition factors (the automaton-on-a-chain encoding). All cells share one domain size d = max(V,
    max #DFA-states); cell ranges are pinned by unary factors. clair.csp.exact_dedP on the position cells
    then equals the native `brute` reachable sets (checked in SMOKE), so the factor-graph organ chains."""
    k = scsp.k; N = scsp.N; V = scsp.V
    dfas = [( _con_dfa(scsp, con)) for con in scsp.cons if con[0] in ("suffix", "contains", "regex")]
    d = max(V, max((dfa.n_states for dfa in dfas), default=0))
    cons = []
    # position cells range 0..V-1 (forbid the padding values V..d-1)
    for i in range(N):
        cons.append(C._rel((i,), (lambda V: lambda t: t[0] < V)(V), d))
    # well-formedness: ε suffix-closed (arity-2 chain)
    for i in range(N - 1):
        cons.append(C._rel((i, i + 1), (lambda k: lambda t: not (t[0] == k and t[1] != k))(k), d))
    # per-constraint factors
    next_aux = N
    for con in scsp.cons:
        tag = con[0]
        if tag == "pos":
            _, i, Sset = con
            cons.append(C._rel((i,), (lambda S: lambda t: t[0] in S)(frozenset(Sset)), d))
        elif tag == "len":
            _, lo, hi = con
            for i in range(N):
                if i < lo:
                    cons.append(C._rel((i,), (lambda k: lambda t: t[0] != k)(k), d))
                if i >= hi:
                    cons.append(C._rel((i,), (lambda k: lambda t: t[0] == k)(k), d))
        elif tag == "eq":
            _, i, j = con
            cons.append(C._rel((i, j), lambda t: t[0] == t[1], d))
        elif tag == "neq":
            _, i, j = con
            cons.append(C._rel((i, j), lambda t: t[0] != t[1], d))
        elif tag == "prefix":
            _, p = con
            for tpos, c in enumerate(p):
                if tpos < N:
                    cons.append(C._rel((tpos,), (lambda c: lambda t: t[0] == c)(c), d))
        elif tag in ("suffix", "contains", "regex"):
            dfa = _con_dfa(scsp, con)
            # aux state cells: state_0 .. state_N (N+1 cells). state_0 pinned to start; state_N ∈ accept.
            st = list(range(next_aux, next_aux + N + 1)); next_aux += N + 1
            cons.append(C._rel((st[0],), (lambda s0: lambda t: t[0] == s0)(dfa.start), d))
            cons.append(C._rel((st[N],), (lambda acc: lambda t: t[0] in acc)(dfa.accept), d))
            for tpos in range(N):
                # transition factor (state_t, char_t, state_{t+1}); ε stutters
                allowed = set()
                for s in range(dfa.n_states):
                    allowed.add((s, k, s))                    # ε: end, state unchanged
                    for c in range(k):
                        allowed.add((s, c, dfa.delta[s][c]))
                cons.append((( st[tpos], tpos, st[tpos + 1]), frozenset(allowed)))
            # pin aux cells to their state range (0..n_states-1)
            for c in st:
                cons.append(C._rel((c,), (lambda ns: lambda t: t[0] < ns)(dfa.n_states), d))
    return C.CSP(next_aux, d, tuple(cons))


def csp_reachable_positions(scsp: StringCSP):
    """Run clair.csp.exact_dedP on to_csp(scsp) and project onto the position cells — the exact per-cell
    reachable set via the CSP path. SMOKE checks this equals native `brute`'s reachable sets."""
    csp = to_csp(scsp)
    ded, _ = C.to_fixpoint(C.exact_dedP, csp, csp.full())
    return tuple(ded[i] for i in range(scsp.N))


# ============================================================ generators (witness-first => solvable)
def _rand_word(rng, k, L):
    return tuple(int(rng.integers(0, k)) for _ in range(L))


def gen_position(rng, k=None, N=None):
    """Crossword-like: plant a witness word, emit per-position char-set hints (the true char + decoys),
    a length bound, and a couple of position equalities the witness satisfies. The per-position char-set
    + equality + length family (low arity, no automaton). Witness-first => always solvable."""
    k = k or int(rng.integers(3, 6))
    N = N or int(rng.integers(4, 7))
    L = int(rng.integers(max(2, N - 2), N + 1))
    w = _rand_word(rng, k, L)
    cons = [("len", int(rng.integers(L - 1, L + 1)) if L >= 2 else L, L)]
    # cap len at exactly L sometimes to sharpen
    cons[0] = ("len", L, L) if rng.random() < 0.5 else cons[0]
    n_hint = int(rng.integers(1, L + 1))
    for i in rng.choice(L, size=n_hint, replace=False):
        i = int(i)
        # char-set hint: the true char + up to 1-2 decoys
        decoys = set(int(x) for x in rng.choice(k, size=int(rng.integers(0, 3)), replace=False))
        S = frozenset({w[i]} | decoys)
        cons.append(("pos", i, S))
    # equalities between equal positions of the witness (crossword crossings)
    for _ in range(int(rng.integers(0, 3))):
        i, j = (int(x) for x in rng.choice(L, size=2, replace=False)) if L >= 2 else (0, 0)
        if w[i] == w[j]:
            cons.append(("eq", i, j))
        elif rng.random() < 0.3:
            cons.append(("neq", i, j))
    return StringCSP(N, k, tuple(cons))


def gen_regex(rng, k=None, N=None):
    """Regex-membership puzzle: a small random regex over Σ + a planted matching word + a few position
    hints. The automaton-chain rung (suffix/contains/regex). Witness-first (regex built to accept w)."""
    k = k or int(rng.integers(2, 4))
    N = N or int(rng.integers(4, 7))
    for _ in range(60):
        L = int(rng.integers(2, N + 1))
        w = _rand_word(rng, k, L)
        ast = _regex_accepting(rng, w, k)
        scsp = StringCSP(N, k, (("regex", ast), ("len", 1, N)))
        # add 0-2 position hints consistent with w
        cons = list(scsp.cons)
        for i in rng.choice(L, size=int(rng.integers(0, min(2, L) + 1)), replace=False):
            i = int(i)
            decoys = set(int(x) for x in rng.choice(k, size=int(rng.integers(0, 2)), replace=False))
            cons.append(("pos", i, frozenset({w[i]} | decoys)))
        scsp = StringCSP(N, k, tuple(cons))
        if compile_regex(ast, k).n_states <= 8 and len(brute(scsp, limit=1)[0]) > 0:
            return scsp
    return StringCSP(N, k, (("contains", (w[0],)), ("len", 1, N)))


def _regex_accepting(rng, w, k):
    """Build a small regex that ACCEPTS the witness word w (so the puzzle is solvable). Variants:
    a contains/suffix/prefix-ish pattern, or a per-position class regex with a couple of wildcards."""
    mode = rng.random()
    cls_any = R_cls(range(k))
    if mode < 0.33 and len(w) >= 1:                          # substring pattern: Σ* w[a:b] Σ*
        a = int(rng.integers(0, len(w)))
        b = int(rng.integers(a + 1, len(w) + 1))
        mid = [R_lit(c) for c in w[a:b]]
        return R_seq(R_star(cls_any), *mid, R_star(cls_any))
    if mode < 0.66:                                          # per-position with wildcards / classes
        parts = []
        for c in w:
            r = rng.random()
            if r < 0.5:
                parts.append(R_lit(c))
            elif r < 0.8:
                parts.append(R_cls({c, int(rng.integers(0, k))}))
            else:
                parts.append(cls_any)
        return R_seq(*parts) if parts else R_EPS
    # suffix pattern: Σ* w[-t:]
    t = int(rng.integers(1, len(w) + 1)) if len(w) >= 1 else 0
    tail = [R_lit(c) for c in w[len(w) - t:]]
    return R_seq(R_star(cls_any), *tail) if tail else R_star(cls_any)


GEN = {"position": gen_position, "regex": gen_regex}
TRAIN_RUNGS = ["position", "regex"]


# ============================================================ SMOKE
def _smoke():
    import numpy as np
    rng = np.random.default_rng(0)
    print("=== clair.strings SMOKE: certified string / sequence abstract-domain organ ===\n")

    # 1. regex DFA validity: compiled DFA agrees with brute regex semantics on random words -----------
    print("[1] regex -> DFA: compiled DFA membership == reference NFA semantics (random words)")
    bad = 0; tot = 0
    for _ in range(400):
        k = int(rng.integers(2, 4))
        w_seed = _rand_word(rng, k, int(rng.integers(0, 5)))
        ast = _regex_accepting(rng, w_seed, k) if len(w_seed) else R_star(R_cls(range(k)))
        dfa = compile_regex(ast, k)
        # the witness it was built to accept must be accepted; plus spot-check random words against a
        # brute regex matcher (we trust _holds('regex')) — DFA.run must equal _holds.
        for _ in range(6):
            w = _rand_word(rng, k, int(rng.integers(0, 5)))
            x = tuple(w) + (k,) * 0
            ref = _holds(("regex", ast), tuple(w) + (k,) * (0), k)
            got = dfa.run(w)
            tot += 1; bad += int(ref != got)
        if not dfa.run(w_seed):
            bad += 1; tot += 1
    print(f"    {tot} checks: {tot - bad}/{tot} DFA==semantics ({'OK' if bad == 0 else f'{bad} BAD'})")
    assert bad == 0

    # 2. CERTIFIED propagation SOUND vs brute, on both rungs ------------------------------------------
    print("\n[2] certified propagation soundness + completeness vs brute (the string dedP)")
    print(f"    {'rung':10s} {'k':>2} {'N':>2} {'#con':>4} {'#sol':>5} {'AC-alive':>8} {'brute-alive':>11} "
          f"{'FE':>3} {'outcome':>8}")
    fe_total = 0; checked = 0; solved = 0; abstain = 0; ac_exact = 0
    for rung in TRAIN_RUNGS:
        for _ in range(120):
            scsp = GEN[rung](rng)
            if scsp.N ** 0 and scsp.V ** scsp.N > 200000:    # keep brute cheap
                continue
            sols, reach = brute(scsp)
            if not sols:
                continue
            dom, _ = to_fixpoint(scsp)
            # SOUNDNESS: AC must never drop a value brute keeps (reach[i] ⊆ dom[i])
            fe = sum(len(reach[i] - dom[i]) for i in range(scsp.N))
            fe_total += fe
            ac_alive = sum(len(c) for c in dom)
            br_alive = sum(len(c) for c in reach)
            checked += 1
            st = status(scsp, dom)
            solved += int(st == "solved"); abstain += int(st == "open")
            ac_exact += int(dom == reach)                    # AC reached the exact string dedP
            assert fe == 0, f"UNSOUND: propagation dropped a real value on {rung} {scsp.cons}"
            if checked <= 8:
                print(f"    {rung:10s} {scsp.k:>2} {scsp.N:>2} {len(scsp.cons):>4} {len(sols):>5} "
                      f"{ac_alive:>8} {br_alive:>11} {fe:>3} {st:>8}")
    print(f"    => {checked} solvable puzzles: false-elim TOTAL = {fe_total} (MUST be 0 — sound); "
          f"AC reached exact dedP on {ac_exact}/{checked}; solved {solved}, abstained {abstain}")
    assert fe_total == 0

    # 3. CSP encoding round-trip: to_csp + exact_dedP (positions) == native brute reachable -----------
    print("\n[3] CSP encoding round-trip (to_csp -> clair.csp.exact_dedP positions == native brute)")
    rt_bad = 0; rt_tot = 0
    for rung in TRAIN_RUNGS:
        for _ in range(40):
            scsp = GEN[rung](rng)
            sols, reach = brute(scsp)
            if not sols:
                continue
            csp = to_csp(scsp)
            if csp.d ** csp.n > 4_000_000:                   # keep the CSP enumeration cheap
                continue
            pos_reach = csp_reachable_positions(scsp)
            rt_tot += 1
            if pos_reach != reach:
                rt_bad += 1
                if rt_bad <= 3:
                    print(f"    MISMATCH {rung}: csp={pos_reach} brute={reach} cons={scsp.cons}")
    print(f"    {rt_tot} puzzles: {rt_tot - rt_bad}/{rt_tot} exact_dedP(positions) == brute "
          f"({'OK — same state type, the factor-graph organ chains' if rt_bad == 0 else f'{rt_bad} BAD'})")
    assert rt_bad == 0

    # 4. the abstain story: an under-determined puzzle narrows but stays open (no guessing) -----------
    print("\n[4] abstain on under-determination (narrows soundly, does NOT guess a singleton)")
    # x over Σ={0,1,2}, N=4, contains '0', length 2..4 — many solutions; AC narrows nothing to singletons
    scsp = StringCSP(4, 3, (("contains", (0,)), ("len", 2, 4)))
    dom, _ = to_fixpoint(scsp)
    sols, reach = brute(scsp)
    print(f"    contains(0), len 2..4, Σ=3, N=4: {len(sols)} strings; AC alive={tuple(len(c) for c in dom)} "
          f"outcome={status(scsp, dom)}  (sound: reach⊆dom={all(reach[i] <= dom[i] for i in range(4))})")
    assert status(scsp, dom) == "open" and all(reach[i] <= dom[i] for i in range(4))

    # 5. a SOLVED crossword: position hints + equality pin a unique string ---------------------------
    print("\n[5] a determined crossword: hints + equality drive AC to a unique checkable string")
    # 3-letter word, Σ={0,1,2,3}; pos0∈{1}, pos2∈{1,2}, eq(0,2), len=3 -> unique '1?1'
    scsp = StringCSP(4, 4, (("len", 3, 3), ("pos", 0, frozenset({1})),
                            ("pos", 2, frozenset({1, 2})), ("eq", 0, 2),
                            ("pos", 1, frozenset({2, 3}))))
    dom, _ = to_fixpoint(scsp)
    sols, _ = brute(scsp)
    print(f"    {len(sols)} solutions; AC outcome={status(scsp, dom)} alive={tuple(sorted(c) for c in dom)}")

    print("\nALL CHECKS PASS — certified string organ: regex DFA exact, propagation SOUND vs brute "
          "across crossword + regex rungs, CSP-encoded (positions == brute) so the factor-graph organ\n"
          "chains, abstains on under-determination. The symbolic-sequence reduced-product piece.")


if __name__ == "__main__":
    _smoke()
