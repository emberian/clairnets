"""clair/datagen/compose_csp.py — COMPOSED-CSP generator for the organ composition-mixup study.

The broad corpus (datagen/extra.py) composes multi-DOMAIN *text* problems (head-domain -> integer ->
modular tail). But the deduction ORGAN (proposer.FactorGraphProposer) narrows finite-domain per-cell
CSPs, so its "multi-skill composition" is a CSP that combines MULTIPLE constraint FAMILIES over a
SHARED variable set — the CSP-expressible, exact-dedP-gradeable analog of the corpus's domain chains.
This is the ONLY composition for which per-cell recall/soundness is even defined (the corpus's
text-chain compositions are an LLM-stage construct the per-cell organ cannot narrow).

SKILLS (atomic constraint families, all WITNESS-FIRST, shared domain d):
    neq     pairwise differ edges            (coloring)
    eq      equality groups                  (equality)
    order   lt/le chains                     (ordering)
    sum     modular (a+b)%d == c             (arithmetic; xor = sum mod 2)
    alldiff one all-distinct block           (alldiff)

A SINGLE-skill instance emits ONE family + pins. A COMPOSED instance plants ONE shared witness s and
emits TWO families' facts that s satisfies (so it stays solvable) + pins. Train sees a SUBSET of the
C(5,2)=10 skill PAIRINGS; the held-out pairings are the OOD-COMPOSITION split (the 2507.07207 lever:
does coverage of SOME compositions generalize to UNSEEN compositions?). Each held-out skill still
appears individually and in OTHER trained pairings, so OOD-composition isolates the *pairing* as the
only novelty, not the skill.

Everything compiles through clair.curriculum.fact_constraint -> extensional clair.csp.CSP, so the
generated CSPs are picklable (safe for the parallel dedP pool) and graded by the exact dedP harness.
"""
from __future__ import annotations

import itertools as it

import numpy as np

from .. import curriculum as CU
from .. import csp as C
from .. import run_general as RG

N_MAX, D_MAX, M_MAX, A_MAX = RG.N_MAX, RG.D_MAX, RG.M_MAX, RG.A_MAX

SKILLS = ("neq", "eq", "order", "sum", "alldiff")
ALL_PAIRS = tuple(it.combinations(SKILLS, 2))            # the C(5,2)=10 pairings

# Held-out OOD-composition pairings (never co-trained). Each skill still appears in >=2 TRAIN pairs,
# so only the PAIRING is novel at eval, not the constituent skill.
OOD_PAIRS = (("neq", "sum"), ("eq", "alldiff"), ("order", "alldiff"))
TRAIN_PAIRS = tuple(p for p in ALL_PAIRS if p not in OOD_PAIRS)


# --------------------------------------------------------------------------- per-skill fact emitters
def _emit(rng, skill, s, n, d, dens=1.0):
    """Emit fact tuples of ONE skill that the witness s (len n, values 0..d-1) satisfies. `dens`
    scales edge probabilities (lower for composed instances so two families stay under M_MAX)."""
    f = []
    if skill == "neq":
        for i in range(n):
            for j in range(i + 1, n):
                if s[i] != s[j] and rng.random() < 0.5 * dens:
                    f.append(("neq", i, j))
    elif skill == "eq":
        for i in range(n):
            for j in range(i + 1, n):
                if s[i] == s[j] and rng.random() < 0.7 * dens:
                    f.append(("eq", i, j))
                elif s[i] != s[j] and rng.random() < 0.2 * dens:
                    f.append(("neq", i, j))
    elif skill == "order":
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if s[i] < s[j] and rng.random() < 0.35 * dens:
                    f.append(("lt", i, j))
                elif s[i] <= s[j] and rng.random() < 0.2 * dens:
                    f.append(("le", i, j))
    elif skill == "sum":                                  # modular (a+b)%d == c that s satisfies
        tries = int(round(3 * n * dens))
        for _ in range(tries):
            a, b, c = (int(x) for x in rng.choice(n, size=3, replace=False))
            if (s[a] + s[b]) % d == s[c]:
                f.append(("sum", a, b, c))
    elif skill == "alldiff":                              # a distinct block of cells under s
        order = list(rng.permutation(n))
        seen, blk = set(), []
        for i in order:
            if int(s[i]) not in seen:
                seen.add(int(s[i])); blk.append(int(i))
        if len(blk) >= 2:
            k = int(rng.integers(2, len(blk) + 1))
            f.append(("alldiff", tuple(sorted(rng.choice(blk, size=k, replace=False).tolist()))))
    else:
        raise ValueError(skill)
    # dedup
    return list(dict.fromkeys((tuple(x) if not isinstance(x[1], tuple) else x) for x in f))


def _pins(rng, s, n, pin_frac):
    return [("pin", int(i), int(s[i])) for i in range(n) if rng.random() < pin_frac]


def _budget_ok(csp, enum_cap=RG.ENUM_CAP):
    return (csp.n <= N_MAX and csp.d <= D_MAX and len(csp.cons) <= M_MAX
            and csp.d ** csp.n <= enum_cap and len(C.solutions(csp, limit=1)) > 0
            and all(len(sc) <= A_MAX for sc, _ in csp.cons))


def _narrows(csp, min_elim=1):
    """Exact dedP eliminates at least `min_elim` (cell,value) pairs => non-vacuous recall target
    (the graph_color lesson: 100% recall of 0 eliminations is meaningless)."""
    full = csp.full()
    ded = C.exact_dedP(csp, full)
    return sum(len(full[i]) - len(ded[i]) for i in range(csp.n)) >= min_elim


# --------------------------------------------------------------------------- instance samplers
def sample_single(rng, skill, n_lo=4, n_hi=6, d_lo=3, d_hi=5, pin_frac=0.45,
                  enum_cap=RG.ENUM_CAP, require_narrow=False):
    """One SOLVABLE single-skill CSP within budget (witness-first)."""
    for _ in range(300):
        n = int(rng.integers(n_lo, n_hi + 1))
        d = int(rng.integers(d_lo, d_hi + 1))
        if skill == "alldiff":
            d = max(d, min(n, D_MAX))                      # need room for a distinct block
        s = rng.integers(0, d, n)
        facts = _emit(rng, skill, s, n, d) + _pins(rng, s, n, pin_frac)
        if len(facts) < 2:
            continue
        csp = CU.build_csp(n, d, facts)
        if _budget_ok(csp, enum_cap) and (not require_narrow or _narrows(csp)):
            return csp
    return CU.build_csp(3, 3, [("pin", 0, 0), ("pin", 1, 1)])


def sample_composed(rng, pair, n_lo=4, n_hi=6, d_lo=3, d_hi=5, pin_frac=0.45,
                    enum_cap=RG.ENUM_CAP, require_narrow=False):
    """One SOLVABLE composed (two-skill) CSP over a SHARED witness s within budget."""
    a, b = pair
    for _ in range(400):
        n = int(rng.integers(n_lo, n_hi + 1))
        d = int(rng.integers(d_lo, d_hi + 1))
        if "alldiff" in pair:
            d = max(d, min(n, D_MAX))
        s = rng.integers(0, d, n)
        facts = (_emit(rng, a, s, n, d, dens=0.7) + _emit(rng, b, s, n, d, dens=0.7)
                 + _pins(rng, s, n, pin_frac))
        facts = list(dict.fromkeys(facts))
        if len(facts) < 3:
            continue
        csp = CU.build_csp(n, d, facts)
        if _budget_ok(csp, enum_cap) and (not require_narrow or _narrows(csp)):
            return csp
    return CU.build_csp(3, 3, [("pin", 0, 0), ("neq", 0, 1), ("pin", 1, 1)])


def sample_for_spec(rng, spec):
    """Resample a fresh CSP for a corpus spec (used by the on-policy roll). spec is ('S', skill) or
    ('C', (a,b))."""
    kind, arg = spec
    return sample_single(rng, arg) if kind == "S" else sample_composed(rng, arg)


# --------------------------------------------------------------------------- training corpus
def make_corpus(rng, pool, mixup, train_pairs=TRAIN_PAIRS):
    """Build `pool` (csp, dom) items + matching specs, with `mixup` fraction COMPOSED (drawn from
    train_pairs, round-robin) and the rest SINGLE-skill (round-robin over SKILLS). Returns
    (items, specs). specs are the resample tags the overlap loop rolls forward."""
    n_comp = int(round(mixup * pool))
    specs = []
    for i in range(n_comp):
        specs.append(("C", train_pairs[i % len(train_pairs)]))
    for i in range(pool - n_comp):
        specs.append(("S", SKILLS[i % len(SKILLS)]))
    rng.shuffle(specs)
    items, rtags = [], []
    for sp in specs:
        c = sample_for_spec(rng, sp)
        items.append((c, c.full()))
        rtags.append(sp)
    return items, rtags


# --------------------------------------------------------------------------- eval splits
def eval_single(rng, neval, wide=False):
    """IN-DIST single-skill (n 4-6) or OOD-N single-skill (n 7-8, d=3 so backtracking stays cheap).
    Balanced round-robin over the 5 skills; non-vacuous (exact dedP narrows)."""
    cfg = dict(n_lo=7, n_hi=8, d_lo=3, d_hi=3, enum_cap=10000) if wide else dict()
    return [sample_single(rng, SKILLS[i % len(SKILLS)], require_narrow=True, **cfg)
            for i in range(neval)]


def eval_composition(rng, neval, ood_pairs=OOD_PAIRS):
    """OOD-COMPOSITION: held-out skill PAIRINGS (never co-trained), round-robin; non-vacuous."""
    return [sample_composed(rng, ood_pairs[i % len(ood_pairs)], require_narrow=True)
            for i in range(neval)]


def make_eval_sets(neval, seed=12345):
    """The three graded splits for one sweep (fixed seed => identical eval across cells)."""
    rng = np.random.default_rng(seed)
    return {
        "in_dist": eval_single(rng, neval, wide=False),
        "ood_n": eval_single(rng, neval, wide=True),
        "ood_comp": eval_composition(rng, neval),
    }
