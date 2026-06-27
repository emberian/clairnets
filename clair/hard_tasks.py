"""clair/hard_tasks.py — HARD, PROPAGATION-REQUIRED problems for the decisive woven-GLaDOS test.

The failed woven run used easy coloring (N=4..6): the LM could shortcut the answer from the prompt
text + an abstain prior, so the organ was never the only route. These generators remove that escape
hatch. The query's answer is determined ONLY by a multi-step chain the LM cannot do in-context:

  eqchain     a long equality chain  pin --eq-- . --eq-- . ... --eq-- QUERY  (graph distance L from
              the pin). The query's color = the pinned color, but ONLY if you trace the chain. A
              SECOND, differently-coloured decoy component is always present, so "echo the one pinned
              colour" is wrong half the time — you must know WHICH component the query is in.

  forcedcolor a forced 3-colouring cascade: neq(i,i-1) AND neq(i,i-2) for all i, with cells 0,1 pinned
              to two different colours. Every later cell is then FORCED to the third colour of its two
              predecessors => the colours run a,b,c,a,b,c,... The query (far cell) is uniquely
              determined but requires propagating the modular pattern L steps from the two pins.

Anti-shortcut properties (the point — base OLMo / a text-only LoRA must FAIL these as L grows):
  * answer requires L-step transitive / modular propagation, not a local lookup;
  * facts are SHUFFLED and cells are RELABELLED by a random permutation, so neither order nor letter
    range leaks the component / position;
  * TWO pinned colours are present, so the abstain prior and the "echo the pin" heuristic both fail;
  * determinacy + answer are read off the EXACT clair.csp.exact_dedP (never a planted guess).

The organ, by contrast, propagates either chain trivially (it IS message passing to fixpoint), so on
the DETERMINED stratum the organ is the only path to the answer. Chain length L is the OOD axis.

Everything is K=3 (colours), arity<=2, so it drops straight into glados_woven.build_batch + the
proven K=3 ScatterGamma. n cells <= NMAX (=12) is enforced by adaptive decoy sizing.
"""
from __future__ import annotations

import numpy as np

from . import csp as C
from . import curriculum as CU

NMAX = 12          # must match glados_woven.NMAX (cells 0..11, entities A..L)


# ===================================================================== fast EXACT dedP
# The hard chains reach n=12 cells (d**n = 3**12 = 531k), far above clair.csp.solutions' brute-force
# budget. `fast_dedP` ROUTES to clair.csp.exact_dedP — the Rust clair_fast port when the extension is
# built, the single pure-Python fallback (clair.csp._exact_dedP_py) otherwise — so every labeller that
# calls it (the GLaDOS datagen query-picker / verifier / trace builder, the organ co-training loss)
# inherits that speedup. This module's old pure-Python AC-then-search (`_fast_dedP_py` + `_ac`/`_sat`)
# was DELETED after being verified cell-for-cell equal to exact_dedP — the one canonical deductor now.
def fast_dedP(csp, dom=None):
    """EXACT per-cell dedP: value v survives at cell i iff some full solution (consistent with `dom`)
    uses v at i. Thin alias to clair.csp.exact_dedP (the Rust clair_fast port when the extension is
    built, the pure-Python clair.csp._exact_dedP_py fallback otherwise). Same signature + (csp, dom)
    semantics as before; dom=None means the full grid."""
    return C.exact_dedP(csp, csp.full() if dom is None else dom)


class FastExact:
    """Drop-in for run_general.Exact / glados_woven.SHARED using fast_dedP (BOUNDED-LRU cached).

    The module-global instance (run_glados_staged.EX) lives for the whole process, including the
    long organ-pretrain (build_targets calls EX.dedP every step over an on-policy-refreshed pool of
    distinct CSPs + their intermediate domains). An unbounded dict would grow without limit across a
    multi-thousand-step run, so the cache is capped with FIFO eviction (it is a pure-function memo —
    eviction only costs a recompute, never correctness)."""
    def __init__(self, maxsize=200_000):
        from collections import OrderedDict
        self.ded = OrderedDict()
        self.maxsize = int(maxsize)

    def solutions(self, csp, dom):                    # only used by callers that want a count; rare
        return C.solutions(csp, dom)

    def dedP(self, csp, dom):
        key = (csp.cons, csp.d, dom)
        hit = self.ded.get(key)
        if hit is not None:
            self.ded.move_to_end(key)                 # LRU: mark recently used
            return hit
        val = fast_dedP(csp, dom)
        self.ded[key] = val
        if len(self.ded) > self.maxsize:
            self.ded.popitem(last=False)              # evict the oldest
        return val


def _relabel(f, perm):
    t = f[0]
    if t == "pin":
        return ("pin", int(perm[f[1]]), f[2])
    if t in ("eq", "neq", "lt", "le"):
        return (t, int(perm[f[1]]), int(perm[f[2]]))
    if t in ("sum", "xor"):
        return (t, int(perm[f[1]]), int(perm[f[2]]), int(perm[f[3]]))
    if t in ("par", "alldiff"):
        return (t, tuple(int(perm[x]) for x in f[1]))
    if t == "parm":
        return ("parm", tuple(int(perm[x]) for x in f[1]), int(f[2]))
    raise ValueError(t)


def _finalize(rng, n, k, facts, query, relation):
    """Random global cell relabel + fact shuffle, then read the EXACT answer/determinacy off
    exact_dedP. Returns a verified curriculum.Problem (kind='color')."""
    perm = rng.permutation(n)                       # perm[old_cell] = new_cell index (=> new letter)
    rfacts = [_relabel(f, perm) for f in facts]
    rng.shuffle(rfacts)
    q = int(perm[query])
    csp = CU.build_csp(n, k, rfacts)
    exact = fast_dedP(csp)
    det = len(exact[q]) == 1
    ans = int(next(iter(exact[q]))) if det else CU.ABSTAIN
    return CU.Problem(relation, n, k, "color", rfacts, q, ans, det, CU.value_names("color", k))


def _decoy(start, dn, v, facts):
    """A length-dn pinned decoy component (cells start..start+dn-1), pinned to colour v. Injects a
    SECOND pinned colour so 'echo the only pin' / abstain heuristics fail. dn>=1."""
    cells = list(range(start, start + dn))
    for a, b in zip(cells[:-1], cells[1:]):
        facts.append(("eq", a, b))
    facts.append(("pin", cells[0], int(v)))


def gen_eqchain(rng, L, k=3, determined=True):
    """Equality chain of L edges (L+1 cells) 0-1-...-L; query the far end (graph distance L from the
    chain-1 pin). A pinned, differently-coloured decoy component is always present."""
    n1 = L + 1
    dn = max(1, min(3, NMAX - n1))                  # fit the decoy inside NMAX
    n = n1 + dn
    assert n <= NMAX, f"eqchain L={L} needs {n} cells > NMAX={NMAX}"
    facts = []
    for a, b in zip(range(n1)[:-1], range(n1)[1:]):
        facts.append(("eq", a, b))                  # chain-1 path: 0-1-...-(n1-1)
    v1, v2 = (int(x) for x in rng.choice(k, size=2, replace=False))
    if determined:
        facts.append(("pin", 0, v1))               # pin one END
    _decoy(n1, dn, v2, facts)                        # the differently-coloured decoy
    query = n1 - 1                                   # the OTHER end of chain-1
    return _finalize(rng, n, k, facts, query, "eqchain")


def gen_forcedcolor(rng, L, k=3, determined=True):
    """Forced 3-colouring cascade: neq(i,i-1) & neq(i,i-2). Determined: pin cells 0,1 to two colours
    => every later cell forced (a,b,c,a,b,c,...). Abstain: pin only cell 0 (2 completions remain)."""
    n = L + 1
    assert n <= NMAX, f"forcedcolor L={L} needs {n} cells > NMAX={NMAX}"
    facts = [("neq", 1, 0)]
    for i in range(2, n):
        facts.append(("neq", i, i - 1))
        facts.append(("neq", i, i - 2))
    v0, v1 = (int(x) for x in rng.choice(k, size=2, replace=False))
    facts.append(("pin", 0, v0))
    if determined:
        facts.append(("pin", 1, v1))                # two pins => fully forced cascade
    query = n - 1
    return _finalize(rng, n, k, facts, query, "forcedcolor")


def _finalize_g(rng, n, d, kind, facts, query, relation):
    """Generalized `_finalize` for non-coloring chains: random relabel + fact shuffle, exact label off
    fast_dedP, for any domain kind / domain size d."""
    perm = rng.permutation(n)
    rfacts = [_relabel(f, perm) for f in facts]
    rng.shuffle(rfacts)
    q = int(perm[query])
    csp = CU.build_csp(n, d, rfacts)
    exact = fast_dedP(csp)
    det = len(exact[q]) == 1
    ans = int(next(iter(exact[q]))) if det else CU.ABSTAIN
    return CU.Problem(relation, n, d, kind, rfacts, q, ans, det, CU.value_names(kind, d))


def gen_orderchain(rng, L, determined=True):
    """A deep ORDERING propagation chain 0<1<...<L. Determined: a TIGHT chain (d = n) forces position
    i = i for every cell, so the far end is uniquely determined L steps from the start. Abstain: give
    slack (d = n + 3) so the far end is not forced. Query is the far end (graph distance L)."""
    n = L + 1
    if determined:
        d = n                                       # tight: the strict chain has a unique embedding
        facts = [("lt", i, i + 1) for i in range(n - 1)]
        facts.append(("pin", 0, 0))                 # anchor (already forced, makes propagation explicit)
    else:
        d = n + 3                                   # slack: positions not pinned down
        facts = [("lt", i, i + 1) for i in range(n - 1)]
    query = n - 1
    return _finalize_g(rng, n, d, "ordinal", facts, query, "orderchain")


def gen_arithchain(rng, L, d=None, determined=True):
    """A deep ARITHMETIC propagation chain: a unit cell U pinned to 1 and sum(i-1, U, i) for each step,
    so val(i) = val(i-1) + 1 (mod d). Determined: pin the head ⇒ the far end is forced (head + L) mod d,
    L steps away. Abstain: drop the head pin ⇒ the whole chain floats. Query is the far end (cell L)."""
    d = int(d) if d is not None else int(rng.integers(4, 7))
    n = L + 2
    U = n - 1                                        # the constant '+1' cell
    facts = [("pin", U, 1 % d)]
    for i in range(1, L + 1):
        facts.append(("sum", i - 1, U, i))          # val(i) = val(i-1) + val(U) = val(i-1) + 1
    if determined:
        facts.append(("pin", 0, int(rng.integers(0, d))))
    query = L
    return _finalize_g(rng, n, d, "number", facts, query, "arithchain")


FAMILIES = {"eqchain": gen_eqchain, "forcedcolor": gen_forcedcolor,
            "orderchain": gen_orderchain, "arithchain": gen_arithchain}


def make_hard(rng, family, L, determined, tries=60):
    """One verified hard Problem of the requested family/length/determinacy (resamples until the
    exact label matches the requested determinacy; the cascade's determinacy depends on L mod 3)."""
    gen = FAMILIES[family]
    for _ in range(tries):
        p = gen(rng, L, determined=determined)
        if p.determined == determined and p.n <= NMAX:
            return p
    return p                                          # best effort (caller may filter)


def pool(rng, family, L, n, determined=True):
    out = []
    while len(out) < n:
        p = make_hard(rng, family, L, determined)
        if p.determined == determined:
            out.append(p)
    return out


def mixed_pool(rng, n, lengths, families=("eqchain", "forcedcolor"), det_frac=0.5):
    """Training pool: a mix of families x chain-lengths, balanced determined/abstain."""
    out = []
    for i in range(n):
        fam = families[i % len(families)]
        L = int(rng.choice(lengths))
        det = bool(rng.random() < det_frac)
        out.append(make_hard(rng, fam, L, det))
    rng.shuffle(out)
    return out


# CSP-only corpus for the pure-organ BOOTSTRAP (no LM); returns (csp, full_dom) items.
def bootstrap_corpus(rng, n, lengths, families=("eqchain", "forcedcolor"), det_frac=0.6):
    probs = mixed_pool(rng, n, lengths, families, det_frac)
    return [p.csp for p in probs]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("hard-task self-check (exact labels via clair.csp):\n")
    for fam in FAMILIES:
        for L in (3, 5, 8):
            d = make_hard(rng, fam, L, True)
            a = make_hard(rng, fam, L, False)
            print(f"[{fam} L={L}] det: n={d.n} q={d.entity(d.query)} ans={CU.canonical_answer(d)!r}"
                  f"   abs: n={a.n} ans={CU.canonical_answer(a)!r}")
        p = make_hard(rng, fam, 5, True)
        print("   render:", CU.canonical_render(p)[:160], "...\n")
