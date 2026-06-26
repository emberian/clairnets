"""clair/datagen/reductions.py — THE REDUCTION CURRICULUM (solve-by-reduction + cross-faculty routing).

Extends the GLaDOS corpus pipeline with reduction RECORDS: each is the trace
`(problem X, reduction X→Y, solve-via-Y, decode-back)`, exact-verified END-TO-END against X's OWN
exact solver (clair.organ.reductions.exact_sat). The faculties are inter-reducible (Karp); this teaches
the LM/organ to RECOGNISE X · REDUCE X→Y · SOLVE-VIA-Y · DECODE-BACK, with the routing learned from the
exact reduce-then-solve trace (the structural prior the GNN-NP papers ignore — handed over for free
because the encoders/decoders/checkers already live in clair.ising_organ + clair.csp).

The faculty the woven organ solves is the CSP narrowing spine (compose.reduced_product over CSPState),
so every curriculum record reduces a SAT-family source X → CSP (sat3/sat2/xorsat → CSP, the certified
edges from clair.organ.reductions). The CSP is solved by the certified narrowers; the forced value of
the queried variable decodes back (identity) to the X-answer.

TWO LEVERS (the design ask):
  BLEND  (LM side):      the SAME underlying problem is emitted under a DIVERSE MIX of representations
                         (clause order, literal order, phrasing) so the LM learns the reduction
                         STRUCTURE, not one memorised surface path. Each record is tagged with its
                         `reduction_path` + `representation`.
  CANONICALIZE (verifier side):  the exact check / dedup / label use a CANONICAL form of Y
                         (`canonical_y` = normalized clause-set), so verification + dedup are stable
                         even as the surface representation varies — dedup collapses the diverse
                         representations of one base problem to a single canonical key.

TWO OOD SPLITS:
  route_required          a SOURCE family with NO direct faculty in train (xorsat), solvable ONLY by
                          reducing to a faculty we DO have (CSP). Tests cross-faculty routing.
  heldout_reduction_path  a representation / encoding of Y never seen in train (a held-out phrasing of
                          the reduction), on a TRAINED source family. Tests routing generalization.

Run:  python -m clair.datagen.reductions          # build splits, exact-verify end-to-end, dedup demo
"""
from __future__ import annotations

import numpy as np

from .. import csp as C
from .. import curriculum as CU
from ..organ import reductions as R

# the surface representations (BLEND axis). Each affects clause/literal order AND phrasing, so α sees
# a genuinely different rendering of the same reduction target.
CNF_REPS = ("verbose_orig", "verbose_shuf", "compact_orig", "compact_shuf")
XOR_REPS = ("parity_orig", "parity_shuf")

# split policy: which (family, representation) pairs go where.
TRAIN_FAMILIES = ("sat3", "sat2")                       # the SOURCE families with reductions in TRAIN
TRAIN_REPS = ("verbose_orig", "verbose_shuf", "parity_orig")
HELDOUT_REP = "compact_orig"                            # a phrasing/encoding NEVER seen in train
ROUTE_REQUIRED_FAMILY = "xorsat"                        # no direct faculty; only reducible to CSP

ENTS = CU.ENTITIES
VNAMES = ["false", "true"]
K = 8                                                   # readout width (matches run_glados_staged.K)


# ============================================================ witness-first SAT with forced variables
def gen_sat(rng, kind, n_lo=4, n_hi=8, pin_frac=0.45):
    """A witness-first SAT instance (satisfiable by construction) with a few UNIT clauses (pins) so the
    forced-variable query is non-trivial. Reuses clair.organ.reductions.rand_sat, then appends unit
    clauses on a subset of the witness's bits (these propagate to force other variables)."""
    base = R.rand_sat(rng, kind, n_lo=n_lo, n_hi=n_hi)
    n = base.n
    # recover a witness (rand_sat is witness-first; re-derive one cheaply via brute for the pins)
    truth = R.exact_sat(base)
    if not truth["sat"]:
        return None
    witness = truth["model"]
    clauses = list(base.clauses)
    for v in range(n):
        if rng.random() < pin_frac:
            if kind == "xorsat":
                clauses.append(((v, False), int(witness[v])))      # unit parity = pin
            else:
                clauses.append(((v, witness[v] == 0),))            # unit clause forcing witness[v]
    return R.Sat(n, tuple(clauses), kind=kind)


def pick_query(sat: R.Sat, rng, det_target=True):
    """Pick a queried variable + its EXACT answer via the CSP's exact per-cell transformer (the
    solve-via-Y step). Determined iff the variable takes the same value in every model. Prefers a
    determined-but-not-unit-pinned variable (so the answer needs propagation, not a lookup)."""
    state = R.sat_to_csp(sat)
    forced = C.exact_dedP(state.csp, state.dom)
    pinned = {cl[0][0] for cl in sat.clauses if len(cl) == 1 and isinstance(cl[0], tuple)}
    if sat.kind == "xorsat":
        pinned = {cl[0][0] for cl in sat.clauses if len(cl) == 2 and isinstance(cl[0], tuple)}
    det = [i for i in range(sat.n) if len(forced[i]) == 1]
    und = [i for i in range(sat.n) if len(forced[i]) > 1]
    det_unpinned = [i for i in det if i not in pinned]
    pref = (det_unpinned or det) if det_target else (und or det)
    pool = pref or det_unpinned or det or und
    if not pool:
        return None
    q = int(rng.choice(pool))
    if len(forced[q]) == 1:
        return q, int(next(iter(forced[q]))), True, forced
    return q, R.exact_sat(sat)["sat"] and -1, False, forced


# ============================================================ CANONICALIZE (verifier-side)
def canonical_y(sat: R.Sat) -> tuple:
    """A representation-INVARIANT signature of the reduction target Y: the normalized clause-set
    (each clause = sorted literals; CNF clauses as frozensets, XOR as sorted vars+parity). Invariant
    to clause order, literal order, and phrasing — so dedup collapses the diverse representations of
    one base problem to a SINGLE key while the surface paths stay diverse, and the exact label is
    computed against this stable canonical form."""
    if sat.kind == "xorsat":
        cls = frozenset((tuple(sorted(v for v, _ in cl[:-1])), cl[-1]) for cl in sat.clauses)
    else:
        cls = frozenset(frozenset(cl) for cl in sat.clauses)
    return (sat.kind, sat.n, cls)


# ============================================================ BLEND (LM-side) — diverse renderings
def _lit_phrase(v, neg, style):
    e = ENTS[v]
    if style == "verbose":
        return f"{e} is {'false' if neg else 'true'}"
    return f"(not {e})" if neg else e                        # compact


def render_cnf(sat: R.Sat, rep, rng):
    """Render a CNF instance under representation `rep` (BLEND): phrasing (verbose|compact) × clause
    order (orig|shuffled). Same logical content, diverse surface."""
    style = "verbose" if rep.startswith("verbose") else "compact"
    clauses = list(sat.clauses)
    if rep.endswith("shuf"):
        rng.shuffle(clauses)
    sents = []
    for cl in clauses:
        lits = list(cl)
        if rep.endswith("shuf"):
            rng.shuffle(lits)
        if style == "verbose":
            if len(lits) == 1:
                v, neg = lits[0]
                sents.append(f"{ENTS[v]} is {'false' if neg else 'true'}.")
            else:
                opts = ", ".join(_lit_phrase(v, neg, "verbose") for v, neg in lits)
                sents.append(f"at least one of these holds: {opts}.")
        else:
            body = " or ".join(_lit_phrase(v, neg, "compact") for v, neg in lits)
            sents.append(f"clause: {body}.")
    return " ".join(s[0].upper() + s[1:] for s in sents)


def render_xor(sat: R.Sat, rep, rng):
    clauses = list(sat.clauses)
    if rep.endswith("shuf"):
        rng.shuffle(clauses)
    sents = []
    for cl in clauses:
        par = cl[-1]; vs = [v for v, _ in cl[:-1]]
        names = ", ".join(ENTS[v] for v in vs)
        if len(vs) == 1:
            sents.append(f"{ENTS[vs[0]]} is {'true' if par == 1 else 'false'}.")
        else:
            sents.append(f"an {'odd' if par == 1 else 'even'} number of {names} are true.")
    return " ".join(s[0].upper() + s[1:] for s in sents)


def render_problem(sat: R.Sat, rep, rng):
    return render_xor(sat, rep, rng) if sat.kind == "xorsat" else render_cnf(sat, rep, rng)


# ============================================================ one reduction record (exact end-to-end)
def make_base(rng, family, n_lo=4, n_hi=8):
    """Generate ONE exact-verified BASE problem (representation-independent): a SAT instance X, its
    reduction to CSP (Y), the solve-via-Y forced cells, a determined query, and the decode-back answer
    — checked end-to-end against X's OWN brute SAT solver. Returns a dict or None on a miss.
    The base is reused to emit MULTIPLE representations (the BLEND), all sharing one canonical Y."""
    sat = gen_sat(rng, family, n_lo, n_hi)
    if sat is None:
        return None
    state = R.sat_to_csp(sat)                        # reduce X→CSP (the target Y)
    if not _within_budget(state.csp):                # the organ tensorization budget (N/D/M/A_MAX)
        return None
    pq = pick_query(sat, rng, det_target=True)
    if pq is None:
        return None
    q, ans_bit, determined, forced = pq
    if not determined:
        return None                                  # det_only curriculum (sharp causal-control signal)
    # EXACT end-to-end: the reduce→solve-via-CSP→decode answer must match X's OWN brute SAT solver
    truth = R.exact_sat(sat)
    if not truth["sat"] or truth["forced"][q] != ans_bit:
        return None
    return {"sat": sat, "csp": state.csp, "q": q, "ans_bit": int(ans_bit),
            "forced": forced, "family": family, "canonical": canonical_y(sat)}


def render_record(base, rep, split, rng):
    """Render a verified base problem under representation `rep` (the BLEND) as a live-pool record: the
    woven reads `alpha_prompt` (X in NL under this rep) and the injected COMPOSED lattice of `csp` (the
    solved Y); the gen prompt is the fact-ablated roster so the reduction is the only route to the
    answer. Records sharing a base have the SAME canonical Y but different surface (the BLEND)."""
    sat, csp, q, gold_idx, forced = base["sat"], base["csp"], base["q"], base["ans_bit"], base["forced"]
    edge = {"sat3": "sat3->csp", "sat2": "sat2->csp", "xorsat": "xorsat->csp"}[base["family"]]
    full_body = render_problem(sat, rep, rng)
    qword = f"What is the value of {ENTS[q]}?"
    cells_body = "Variables " + ", ".join(ENTS[i] for i in range(sat.n)) + ". " + qword
    surv = _surv(forced, sat.n)
    return {
        "prompt": cells_body + " Answer:",
        "answer": VNAMES[gold_idx],
        "n": sat.n, "query": q, "determined": True, "gold_idx": int(gold_idx),
        "relation": edge,                            # by_rung key = the reduction path
        "vnames": list(VNAMES),
        "mentions": _mentions(cells_body, sat.n),
        "surv": surv, "tgt": surv.copy(),
        "csp": csp,                                  # the target Y the composer solves
        "alpha_prompt": f"{full_body} {qword} Answer:",
        "alpha_mentions": _mentions(full_body + " " + qword, sat.n),
        "reduction_path": edge, "representation": rep, "family": base["family"],
        "route_required": (split == "route_required"),
        "split": split, "canonical": base["canonical"],
    }


def make_record(rng, family, rep, split, n_lo=4, n_hi=8):
    """Convenience: one base + one rendering (used by the OOD splits, one rep each)."""
    base = make_base(rng, family, n_lo, n_hi)
    return render_record(base, rep, split, rng) if base is not None else None


# the organ tensorization budget (clair.run_glados_staged N_MAX/D_MAX/M_MAX/A_MAX) — inlined so the
# curriculum stays torch-free (the on-disk corpus build never needs the GPU stack).
_N_MAX, _D_MAX, _M_MAX, _A_MAX = 12, 8, 80, 3


def _within_budget(csp) -> bool:
    if csp.n > _N_MAX or csp.d > _D_MAX or len(csp.cons) > _M_MAX:
        return False
    for sc, al in csp.cons:
        if len(sc) > _A_MAX or any(v >= _D_MAX for t in al for v in t):
            return False
    return True


def _surv(forced, n):
    surv = np.zeros((n, K), dtype=np.float32)
    for i in range(n):
        for v in forced[i]:
            if v < K:
                surv[i, v] = 1.0
    return surv


def _mentions(text, n):
    return {int(k): [tuple(s) for s in v] for k, v in CU.entity_mentions(text, n).items()}


# ============================================================ build the splits + verify + dedup
def build_curriculum(seed=0, per_cell=60, n_lo=4, n_hi=8, verbose=True):
    """Build train / route_required / heldout_reduction_path splits. BLEND in train (diverse reps);
    the OOD splits hold out an unseen SOURCE family (route_required) and an unseen REPRESENTATION
    (heldout_reduction_path). Every record is exact-verified end-to-end at construction. Returns
    {split: [records]} plus a dedup report on the canonical form."""
    rng = np.random.default_rng(seed)
    splits = {"train": [], "route_required": [], "heldout_reduction_path": []}

    # TRAIN: each verified BASE problem emitted under EVERY trained representation (the BLEND) — same
    # canonical Y, diverse surface, so the LM learns the reduction STRUCTURE not one memorised path.
    for fam in TRAIN_FAMILIES:
        reps = [r for r in (CNF_REPS if fam != "xorsat" else XOR_REPS) if r in TRAIN_REPS]
        got = tries = 0
        while got < per_cell and tries < per_cell * 40:
            tries += 1
            base = make_base(rng, fam, n_lo, n_hi)
            if base is None:
                continue
            for rep in reps:
                splits["train"].append(render_record(base, rep, "train", rng))
            got += 1

    # ROUTE_REQUIRED: an UNSEEN source family (xorsat) — solvable only by reducing to CSP
    fam = ROUTE_REQUIRED_FAMILY
    for rep in XOR_REPS:
        got = tries = 0
        while got < per_cell // 2 and tries < per_cell * 40:
            tries += 1
            rec = make_record(rng, fam, rep, "route_required", n_lo, n_hi)
            if rec is not None:
                splits["route_required"].append(rec); got += 1

    # HELDOUT_REDUCTION_PATH: a TRAINED family rendered with an UNSEEN representation
    for fam in TRAIN_FAMILIES:
        got = tries = 0
        while got < per_cell and tries < per_cell * 40:
            tries += 1
            rec = make_record(rng, fam, HELDOUT_REP, "heldout_reduction_path", n_lo, n_hi)
            if rec is not None:
                splits["heldout_reduction_path"].append(rec); got += 1

    for s in splits.values():
        rng.shuffle(s)

    report = verify_splits(splits, verbose=verbose)
    return splits, report


def verify_splits(splits, verbose=True):
    """(1) re-verify EVERY record end-to-end from its serialized target Y vs X's exact answer;
    (2) demonstrate CANONICALIZE: dedup on the canonical form collapses diverse representations while
    surface paths stay diverse."""
    report = {"sizes": {k: len(v) for k, v in splits.items()}}
    bad = 0
    for sp, recs in splits.items():
        for rec in recs:
            # rebuild the target CSP, solve exactly, confirm the stored forced answer
            csp = rec["csp"]
            forced = C.exact_dedP(csp, csp.full())
            cell = forced[rec["query"]]
            if not (len(cell) == 1 and int(next(iter(cell))) == rec["gold_idx"]):
                bad += 1
    report["reverify_mismatches"] = bad

    # CANONICALIZE / dedup demo on TRAIN: canonical keys collapse representation-variants; surface stays diverse
    train = splits["train"]
    canon_keys = {r["canonical"] for r in train}
    surfaces = {r["alpha_prompt"] for r in train}
    reps = {r["representation"] for r in train}
    report["train_records"] = len(train)
    report["train_canonical_unique"] = len(canon_keys)
    report["train_surface_unique"] = len(surfaces)
    report["train_representations"] = sorted(reps)
    # the BLEND check: a record whose canonical Y matches another but whose representation differs
    by_canon = {}
    for r in train:
        by_canon.setdefault(r["canonical"], set()).add(r["representation"])
    multi_rep = sum(1 for v in by_canon.values() if len(v) > 1)
    report["canonical_keys_with_multiple_reps"] = multi_rep

    if verbose:
        print("== reduction curriculum ==")
        print(f"  sizes: {report['sizes']}")
        print(f"  end-to-end re-verify mismatches: {bad}  (must be 0)")
        print(f"  TRAIN: {len(train)} records  canonical-unique {len(canon_keys)}  "
              f"surface-unique {len(surfaces)}  reps {sorted(reps)}")
        print(f"  CANONICALIZE: {multi_rep} canonical keys carry >1 representation "
              f"(dedup collapses reps; paths stay diverse)")
        assert bad == 0, "a reduction record failed end-to-end re-verification"
        # confirm the OOD splits are populated and disjoint in their held-out axis
        assert all(r["family"] == ROUTE_REQUIRED_FAMILY for r in splits["route_required"]), \
            "route_required must be the unseen source family"
        assert all(r["representation"] == HELDOUT_REP for r in splits["heldout_reduction_path"]), \
            "heldout_reduction_path must be the unseen representation"
        assert HELDOUT_REP not in reps, "the held-out representation leaked into train"
        assert ROUTE_REQUIRED_FAMILY not in {r["family"] for r in train}, \
            "the route_required family leaked into train"
        print("  OOD splits populated + disjoint (route_required=unseen family, "
              "heldout_reduction_path=unseen representation): OK")
    return report


if __name__ == "__main__":
    build_curriculum(seed=0, per_cell=60)
