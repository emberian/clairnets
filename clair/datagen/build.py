"""clair/datagen/build.py — the seeded, deterministic GLaDOS reasoning-corpus driver.

Pipeline (all rule-based, free, reproducible — NO frontier model on the critical path):

    generators  CSP families (curriculum) + the OOD relation gen_parity_k + hard_tasks chains
        -> render        the non-stilted renderer (skins x naming x phrasing x structure)
        -> trace         organ-grounded, exact-by-construction reasoning trace
        -> QC            exact-verify the label via clair.csp (fast_dedP) + an independent
                         keyword-reader ambiguity gate (parse.py round-trip)
        -> dedup         MinHash/LSH near-duplicate removal, in split order (no train->eval leak)
        -> SPLITS        train / val / ood_n / ood_phrasing / ood_relation
        -> write         parquet + jsonl per split + schema.json + stats.json + sample.jsonl

EXACTNESS GUARANTEE. Every kept record's label is re-derived from the structured CSP by the
exact per-cell deductor (clair.hard_tasks.fast_dedP, verified == clair.csp.exact_dedP) — at
generation (qc.answer_exact) AND again from the SERIALIZED `cons` field (verify_record), so the
on-disk corpus is independently re-checkable with clair.csp alone.

DETERMINISM. Each (split, shard) is generated from a derived seed; shards are concatenated in
index order; dedup is order-deterministic. So `build(seed=s)` reproduces bit-for-bit.

SPLITS / OOD axes (held out so each axis is a clean generalization test):
  train / val    standard CSP families, N in TRAIN_N, TRAIN skins.   (val = disjoint seed)
  ood_n          same families, WIDE N (TRAIN skins) — length generalization.
  ood_phrasing   standard families, TRAIN N, HELD-OUT skins — phrasing/lexical generalization.
  ood_relation   gen_parity_k (arity 3-5 affine 'wall') + hard_tasks propagation chains — a
                 relation FAMILY never seen in train.

Run:  python -m clair.datagen.build --out data/glados_corpus --scale 1.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .. import csp as C
from .. import curriculum as CU
from .. import hard_tasks as HT
from . import qc, render, trace
from . import extra as EX

SCHEMA_VERSION = "2.0"

# --------------------------------------------------------------------------- skin partition
# TRAIN skins dress train / val / ood_n / ood_relation. HELD-OUT skins dress ONLY ood_phrasing,
# so phrasing is a clean OOD axis (no lexical/skin overlap between train and the phrasing eval).
TRAIN_SKINS = frozenset({
    "graph_coloring", "map_regions", "register_alloc",     # color
    "scheduling", "race",                                   # ordinal
    "circuit", "boolean_logic", "arithmetic", "lockbox",   # number
})
HELDOUT_SKINS = frozenset({
    "frequency", "social_teams",                           # color
    "prereqs", "timeline",                                 # ordinal
    "sensors",                                             # number
})

# --------------------------------------------------------------------------- family / N config
# gen_xor (the pure arity-3 affine rung) is now IN train — its omission was the affine-wall-eval-only
# accident the audit flagged. The difficulty-controlled families (affine/bagchain) are appended as
# separate mix components (NEW_FAMILY_GEN) so they carry their own level/treewidth targets.
TRAIN_FAMILIES = ("coloring", "equality", "ordering", "arithmetic", "alldiff", "xor")
# OOD-N uses only arity<=3 families (alldiff's arity-N extensional table is infeasible at wide N).
OODN_FAMILIES = ("coloring", "equality", "ordering", "arithmetic")
# OOD-phrasing uses families that have >=1 held-out skin (coloring/equality/ordering/alldiff).
PHRASING_FAMILIES = ("coloring", "equality", "ordering", "alldiff")
HARD_FAMILIES = ("eqchain", "forcedcolor")
# the family-parametric propagation chains (depth tail) — every family, not just coloring.
CHAIN_FAMILIES = ("eqchain", "forcedcolor", "orderchain", "arithchain")
HARD_LENGTHS = (4, 5, 6, 7, 8, 9)                           # depth ~ L: spans short/med/long thirds
LONG_LENGTHS = (10,)                                        # ood_depth: one bucket past the trained max

# difficulty-CONTROLLED curriculum families: each carries a target (level / treewidth) so the build
# mix can fill the hard buckets the audit found empty (98.7% L0, tw median 2). Generated via
# make_curriculum, rendered + QC'd like any CSP family.
NEW_FAMILY_GEN = {
    "affine_l2": (CU.gen_affine, dict(level=2, n_lo=6, n_hi=11)),   # the affine WALL, ~level 2
    "affine_l3": (CU.gen_affine, dict(level=3, n_lo=7, n_hi=12)),   # wider band, ~level 3+
    "affine_l4": (CU.gen_affine, dict(level=4, n_lo=9, n_hi=12)),   # ood_level: one bucket past train
    "bagchain":  (CU.gen_bagchain, dict()),                        # treewidth 1-5 (family randomized)
}

TRAIN_N = (4, 7)
WIDE_N = (8, 11)
MIN_FACTS = 2                                               # drop trivial problems

# TRAIN CSP-spine mix targeting the audited difficulty distribution (was 98.7% L0 / tw-median-2 /
# depth-3): easy families (L0) + family-parametric chains (DEEP) + bag-chain (treewidth spread) +
# difficulty-controlled affine (the level-2/3 wall) + additive random relations. (kind, mix-cfg, frac).
TRAIN_CSP_MIX = (
    ("curriculum", dict(families=TRAIN_FAMILIES, n=TRAIN_N),  0.27),   # easy spine (L0)
    ("hard",       dict(families=CHAIN_FAMILIES),             0.18),   # family chains (DEEP)
    ("curriculum", dict(families=("bagchain",), n=TRAIN_N),  0.10),   # treewidth 1-5 spread
    ("curriculum", dict(families=("affine_l2",), n=TRAIN_N), 0.25),   # affine wall (level 2)
    ("curriculum", dict(families=("affine_l3",), n=TRAIN_N), 0.12),   # wide affine (level 3+)
    ("randomrel",  dict(),                                   0.08),   # additive novel relations
)

# Per-family density: denser constraints/pins force MORE non-pinned cells, so determined-via-
# propagation queries are available more often (without them, sparse random CSPs are almost all
# "cannot be determined"). Combined with the anti-shortcut query picker this yields a balanced,
# non-shortcutable determined/abstain mix instead of an abstain-dominated one.
DENSITY = {
    "coloring": dict(edge_p=0.6, pin_frac=0.5),
    "equality": dict(pin_frac=0.5),
    "ordering": dict(pin_frac=0.5),
    "arithmetic": dict(pin_frac=0.9),
    "alldiff": dict(pin_frac=0.6),
}

# --------------------------------------------------------------- broadened (non-CSP) domain config
# Each broadened domain ships TWO configs: a `train` distribution and a harder/wider `ood_domain`
# distribution held out for per-domain size/difficulty generalization.
EXTRA_DOMAINS = EX.DOMAINS                  # xor, graph, perm, typeinfer, fol, optimize
EXTRA_TRAIN_CFG = {
    "xor":       {"n": (5, 9)},
    "graph":     {"n": (6, 14)},
    "perm":      {"n": (4, 7)},
    "typeinfer": {"budget": 3, "max_nodes": 14},
    "fol":       {"depth": (1, 5)},
    "optimize":  {},
}
EXTRA_OOD_CFG = {                            # per-domain held-out: wider N / deeper / harder
    "xor":       {"n": (10, 14), "bands": (3, 4, 6, 12)},
    "graph":     {"n": (18, 28), "p": (0.10, 0.20)},
    "perm":      {"n": (8, 9)},
    "typeinfer": {"budget": 4, "max_nodes": 20},
    "fol":       {"depth": (4, 7)},
    "optimize":  {"cut_n": (9, 13)},
}
# COMPOSITION pairs. Train teaches a subset of (head-domain + arith) pairings; the OOD-composition
# split holds out ENTIRELY-UNSEEN domain pairings (xor/ordering heads + a 3-domain graph+perm chain),
# so co-occurrence of domains is a clean generalization axis.
COMPOSE_TRAIN_HEADS = ("graph", "fol", "perm", "typeinfer")
COMPOSE_OOD_HEADS = ("xor", "ordering")
COMPOSE_OOD_TRIPLE = ("graph", "perm")      # head=graph, triple=perm -> graph+perm+arith (3 domains)


# --------------------------------------------------------------------------- problem factories
def _pick_query(csp, facts, rng, det_target):
    """Pick a query cell + its EXACT answer via fast_dedP, biased to det_target. For determined
    cells prefer one that is NOT directly pinned (so the answer needs propagation, not lookup)."""
    exact = HT.fast_dedP(csp)
    pinned = {f[1] for f in facts if f[0] == "pin"}
    det = [i for i in range(csp.n) if len(exact[i]) == 1]
    und = [i for i in range(csp.n) if len(exact[i]) > 1]
    det_unpinned = [i for i in det if i not in pinned]
    # Anti-shortcut: a determined query must require PROPAGATION, never a direct pin lookup. Prefer a
    # determined-but-unpinned cell; if none exists, fall back to an abstain cell rather than a pinned
    # cell (whose answer is read straight off the prompt).
    pref = (det_unpinned or und or det) if det_target else (und or det_unpinned or det)
    pool = pref or det_unpinned or und or det
    if not pool:
        return 0, CU.ABSTAIN, False
    q = int(rng.choice(pool))
    if len(exact[q]) == 1:
        return q, int(next(iter(exact[q]))), True
    return q, CU.ABSTAIN, False


def make_curriculum(rng, family, n_lo, n_hi, det_target):
    if family in NEW_FAMILY_GEN:                     # difficulty-controlled: generator owns its sizing
        gen, kw = NEW_FAMILY_GEN[family]
        n, d, kind, facts, s = gen(rng, **kw)
    else:
        if family == "alldiff":
            n_hi = min(n_hi, 5)     # alldiff's arity-N extensional table is d^n; keep it cheap/small
        gen = CU.GENERATORS[family]
        n, d, kind, facts, s = gen(rng, n_lo=n_lo, n_hi=n_hi, **DENSITY.get(family, {}))
    assert CU.facts_satisfied_by(facts, s, d), "witness violated a fact (generator bug)"
    csp = CU.build_csp(n, d, facts)
    q, ans, det = _pick_query(csp, facts, rng, det_target)
    return CU.Problem(family, n, d, kind, facts, q, ans, det, CU.value_names(kind, d)), csp


def make_randomrel(rng, det_target):
    """A RANDOM allowed-tuple (novel-relation) problem. Built straight from the exact CSP (the prose IS
    the explicit table, so no keyword round-trip is needed) — exactness via clair.csp on `cons`."""
    n, d, kind, facts, s = CU.gen_random_relation(rng)
    assert CU.facts_satisfied_by(facts, s, d), "witness violated a random-relation fact"
    csp = CU.build_csp(n, d, facts)
    q, ans, det = _pick_query(csp, facts, rng, det_target)
    return CU.Problem("random_relation", n, d, kind, facts, q, ans, det, CU.value_names(kind, d)), csp


def make_parity(rng, det_target):
    n, d, kind, facts, s = CU.gen_parity_k(rng, n_lo=5, n_hi=8, kmin=3, kmax=5)
    assert CU.facts_satisfied_by(facts, s, d), "witness violated a parity fact"
    csp = CU.build_csp(n, d, facts)
    q, ans, det = _pick_query(csp, facts, rng, det_target)
    return CU.Problem("parity_k", n, d, kind, facts, q, ans, det, CU.value_names(kind, d)), csp


def make_hard(rng, family, det_target, lengths=HARD_LENGTHS):
    L = int(rng.choice(lengths))
    p = HT.make_hard(rng, family, L, det_target)
    return p, p.csp


# --------------------------------------------------------------------------- serialization
def _clean(o):
    """Recursively convert numpy scalar types to plain Python so the record is JSON/parquet-safe."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(x) for x in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _ser_facts(facts):
    out = []
    for f in facts:
        if f[0] in ("alldiff", "par"):
            out.append([f[0], [int(x) for x in f[1]]])
        elif f[0] == "parm":
            out.append(["parm", [int(x) for x in f[1]], int(f[2])])
        elif f[0] == "rel":
            out.append(["rel", [int(x) for x in f[1]], [[int(x) for x in t] for t in f[2]]])
        else:
            out.append([f[0]] + [int(x) if isinstance(x, (int, np.integer)) else x for x in f[1:]])
    return out


def _record(p, rendering, tr, split, family, seed, idx, csp):
    cons = [[[int(c) for c in sc], sorted([[int(x) for x in t] for t in al])]
            for sc, al in csp.cons]
    ans_idx = int(p.answer) if p.determined else CU.ABSTAIN
    answer = rendering.values[p.answer] if p.determined else "cannot be determined"
    rid = hashlib.blake2b(f"{split}-{seed}-{idx}".encode(), digest_size=8).hexdigest()
    return _clean({
        "id": f"{split}-{rid}",
        "split": split,
        "domain": "csp",
        "family": family,
        "domain_kind": p.kind,
        "skin": rendering.skin,
        "n": int(p.n),
        "d": int(p.d),
        "n_facts": len(p.facts),
        "determined": bool(p.determined),
        "text": rendering.text,
        "question": rendering.question,
        "answer": answer,
        "answer_index": ans_idx,
        "trace": tr,
        "query": int(p.query),
        "entities": list(rendering.entities),
        "values": list(rendering.values),
        "facts": _ser_facts(p.facts),
        "cons": cons,
        "payload": {},
        "naming_scheme": rendering.scheme,
        "structure": rendering.structure,
        "n_domains": 1,
        "compose": "",
        "seed": int(seed),
    })


def verify_record(rec) -> bool:
    """Independent re-verification from the SERIALIZED record (clair.csp only): rebuild the exact
    CSP from `cons`, re-derive the queried cell's exact label, and confirm it matches what we
    stored. This is the corpus's portable exactness guarantee."""
    cons = tuple((tuple(sc), frozenset(tuple(t) for t in al)) for sc, al in rec["cons"])
    csp = C.CSP(rec["n"], rec["d"], cons)
    exact = HT.fast_dedP(csp)
    cell = exact[rec["query"]]
    det = len(cell) == 1
    if det != rec["determined"]:
        return False
    if det:
        return int(next(iter(cell))) == rec["answer_index"]
    return rec["answer_index"] == CU.ABSTAIN


def verify_any(rec) -> bool:
    """Domain-dispatched exact re-verification: CSP records re-check via clair.csp (`cons`); every
    other domain re-runs its own oracle from `payload` (clair.datagen.extra.verify_record)."""
    if rec.get("domain", "csp") == "csp":
        return verify_record(rec)
    return EX.verify_record(rec)


# --------------------------------------------------------------------------- one record
def _finalize_extra(rec, split, seed, idx):
    """Stamp an extra-domain core record with id/split/seed (the keys the driver owns)."""
    rid = hashlib.blake2b(f"{split}-{seed}-{idx}".encode(), digest_size=8).hexdigest()
    rec["id"] = f"{split}-{rid}"
    rec["split"] = split
    rec["seed"] = int(seed)
    return _clean(rec)


def _gen_one(spec, rng, idx):
    """Generate -> render -> trace -> QC one record for split `spec`. Returns (rec, status)."""
    split, family = spec["split"], None

    # ---- broadened (non-CSP) domains: their own generator + EXACT oracle gate ----
    if spec["kind"] == "extra":
        rec = EX.generate(spec["domain"], rng, spec.get("cfg", {}))
        if rec is None:
            return None, "gen_fail"
        rec = _finalize_extra(rec, split, spec["seed"], idx)
        if not EX.verify_record(rec):                      # exactness gate (re-run the domain oracle)
            return None, "label_mismatch"
        return rec, "ok"
    if spec["kind"] == "compose":
        rec = EX.gen_composition(rng, spec["head"], spec.get("cfg", {}), triple=spec.get("triple"))
        if rec is None:
            return None, "gen_fail"
        rec = _finalize_extra(rec, split, spec["seed"], idx)
        if not EX.verify_record(rec):
            return None, "label_mismatch"
        return rec, "ok"

    det_target = bool(rng.random() < 0.6)
    if spec["kind"] == "randomrel":
        # novel-relation records: the prose is an EXPLICIT allowed-table, so the keyword round-trip
        # gate is skipped (exactness is from clair.csp on `cons`, like the reduction records).
        family = "random_relation"
        p, csp = make_randomrel(rng, det_target)
        if len(p.facts) < MIN_FACTS:
            return None, "trivial"
        ents = [CU.ENTITIES[i] for i in range(p.n)]
        tr = trace.build_trace(p, ents, p.vnames, csp=csp)
        r = render.Rendering(CU.canonical_render(p), "random_relation", p.kind, ents, p.vnames,
                             "letters", "prose", CU._question(p))
        return _record(p, r, tr, split, family, spec["seed"], idx, csp), "ok"
    if spec["kind"] == "curriculum":
        family = spec["families"][idx % len(spec["families"])]
        p, csp = make_curriculum(rng, family, spec["n"][0], spec["n"][1], det_target)
    elif spec["kind"] == "parity":
        family = "parity_k"
        p, csp = make_parity(rng, det_target)
    else:                                                  # hard / chain
        fams = spec.get("families") or HARD_FAMILIES
        family = fams[idx % len(fams)]
        p, csp = make_hard(rng, family, det_target, lengths=spec.get("lengths") or HARD_LENGTHS)

    if len(p.facts) < MIN_FACTS:
        return None, "trivial"
    # NOTE: the label is exact BY CONSTRUCTION (make_* derives answer/determined straight from the
    # exact deductor) and is re-verified again from the serialized `cons` in verify_record (the
    # authoritative gate). A redundant qc.answer_exact() here would just double the hot-path dedP.
    r = render.render_problem(p, rng, allowed=spec["skins"])
    if r.skin not in spec["skins"]:                        # skin restriction escaped via fallback
        return None, "skin_fallback"
    ok, _ = qc.unambiguous(p, r)                           # independent-reader ambiguity gate
    if not ok:
        return None, "ambiguous"
    tr = trace.build_trace(p, r.entities, r.values, csp=csp)
    return _record(p, r, tr, split, family, spec["seed"], idx, csp), "ok"


def _gen_shard(args):
    """Generate exactly `quota` kept records for one shard (deterministic from `seed`). Returns
    (records, attempt_stats)."""
    spec, quota = args
    rng = np.random.default_rng(spec["seed"])
    out = []
    stats = {"trivial": 0, "ambiguous": 0, "skin_fallback": 0, "label_mismatch": 0,
             "gen_fail": 0, "attempts": 0}
    idx = 0
    while len(out) < quota:
        rec, status = _gen_one(spec, rng, idx)
        stats["attempts"] += 1
        idx += 1
        if rec is None:
            stats[status] += 1
            continue
        out.append(rec)
    return out, stats


# --------------------------------------------------------------------------- split driver
def _shard_specs(split, kind, count, base_seed, skins, families=None, n=None, nshards=16,
                 domain=None, cfg=None, head=None, triple=None, lengths=None):
    """Split `count` kept records across `nshards` deterministic shards."""
    per = [count // nshards] * nshards
    for i in range(count % nshards):
        per[i] += 1
    specs = []
    for sh in range(nshards):
        if per[sh] == 0:
            continue
        spec = {"split": split, "kind": kind, "seed": base_seed + sh,
                "skins": skins, "families": families, "n": n,
                "domain": domain, "cfg": cfg, "head": head, "triple": triple, "lengths": lengths}
        specs.append((spec, per[sh]))
    return specs


def build_split(split, kind, count, base_seed, skins, families=None, n=None,
                nshards=16, pool=None, domain=None, cfg=None, head=None, triple=None, lengths=None):
    specs = _shard_specs(split, kind, count, base_seed, skins, families, n, nshards,
                         domain=domain, cfg=cfg, head=head, triple=triple, lengths=lengths)
    results = list(pool.map(_gen_shard, specs)) if pool else [_gen_shard(s) for s in specs]
    records, agg = [], {"trivial": 0, "ambiguous": 0, "skin_fallback": 0,
                        "label_mismatch": 0, "gen_fail": 0, "attempts": 0}
    for recs, st in results:
        records.extend(recs)
        for k in agg:
            agg[k] += st[k]
    return records, agg


# --------------------------------------------------------------------------- dedup + write
def dedup_in_order(by_split, rng):
    """Global MinHash/LSH near-dup removal across ALL splits in a fixed order, so later (eval)
    splits never keep a near-duplicate of an earlier (train) record — no train->eval leakage."""
    order = ["train", "val", "ood_n", "ood_phrasing", "ood_relation",
             "ood_level", "ood_depth", "ood_domain", "ood_composition"]
    flat, owners = [], []
    for sp in order:
        for rec in by_split.get(sp, []):
            flat.append(rec["text"])
            owners.append(sp)
    keep = qc.near_dup_mask(flat, rng)
    out = {sp: [] for sp in order}
    dropped = {sp: 0 for sp in order}
    for k, sp, rec in zip(keep, owners, [r for sp2 in order for r in by_split.get(sp2, [])]):
        (out[sp].append(rec) if k else None)
        if not k:
            dropped[sp] += 1
    return out, dropped


def write_split(out_dir, split, records):
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(out_dir, exist_ok=True)
    jsonl = os.path.join(out_dir, f"{split}.jsonl")
    with open(jsonl, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    # parquet: nested structured fields are stored as JSON strings for a flat, portable schema.
    cols = {}
    for r in records:
        for k, v in r.items():
            cols.setdefault(k, []).append(
                json.dumps(v) if k in ("facts", "cons", "entities", "values", "payload") else v)
    pq.write_table(pa.table(cols), os.path.join(out_dir, f"{split}.parquet"))
    return jsonl


def schema_dict():
    return {
        "schema_version": SCHEMA_VERSION,
        "fields": {
            "id": "str — unique record id (split-prefixed content hash)",
            "split": "str — train|val|ood_n|ood_phrasing|ood_relation|ood_domain|ood_composition",
            "domain": "str — reasoning domain: csp|xor|graph|perm|typeinfer|fol|optimize|compose",
            "family": "str — generator family/sub-task within the domain",
            "domain_kind": "str — color|ordinal|number",
            "skin": "str — story-world skin key used to render the problem",
            "n": "int — number of CSP cells (entities)",
            "d": "int — domain size (values 0..d-1)",
            "n_facts": "int — number of ground-truth facts (>= 2)",
            "determined": "bool — is the queried cell uniquely forced",
            "text": "str — the rendered natural-language problem (the model input)",
            "question": "str — the question sentence (also the tail of `text`)",
            "answer": "str — value SURFACE of the answer, or 'cannot be determined'",
            "answer_index": "int — value index 0..d-1, or -1 (ABSTAIN)",
            "trace": "str — exact, organ-grounded reasoning trace (sound AC narrowing + exact closer)",
            "query": "int — queried cell index",
            "entities": "list[str] — surface per entity index (JSON in parquet)",
            "values": "list[str] — surface per value index (JSON in parquet)",
            "facts": "list — structured ground-truth facts, CSP domain only (JSON in parquet)",
            "cons": "list — extensional CSP (scope, allowed-tuples), CSP domain only (JSON in parquet)",
            "payload": "dict — structured problem for the NON-CSP domains, sufficient to re-run the "
                       "domain oracle (JSON in parquet); empty for the CSP domain",
            "naming_scheme": "str — entity naming scheme used",
            "structure": "str — layout structure (prose/semicolon/bullets/numbered)",
            "n_domains": "int — how many distinct reasoning domains the record composes (1 = single)",
            "compose": "str — composition signature (e.g. 'graph+arith'); '' for single-domain records",
            "seed": "int — shard seed that produced the record",
        },
        "exact_verification": {
            "csp": "rebuild clair.csp.CSP from `cons`, run clair.hard_tasks.fast_dedP, confirm "
                   "cell[query] == answer_index (singleton) or ABSTAIN.",
            "xor": "rebuild the GF(2) system from payload.rows, run clair.xor_wall.gf2_forced/gf2_rref.",
            "graph": "rebuild the networkx graph from payload.edges, recompute Dijkstra/reachability.",
            "perm": "rebuild clair.permgroup.PermProblem, run exact_dedP (Régin/Hall GAC + brute).",
            "typeinfer": "rebuild the AST, run clair.typeinfer.algorithm_w (Algorithm W) + ground.",
            "fol": "rebuild rules/facts, run clair.fol.forward_chain (least Herbrand model) + label.",
            "optimize": "min-cost via clair.energy_organ.brute_opt; max-cut via exact 2^n brute force.",
            "compose": "recompute each head's exact result, re-apply the modular-arithmetic tail.",
        },
    }


# --------------------------------------------------------------------------- main
def build(out_dir, seed=0, scale=1.0, nshards=16, workers=None, verify_frac=1.0):
    t0 = time.time()
    # base counts (scale them); val/oods are smaller eval sets.
    counts = {
        "train": int(40000 * scale),
        "val": int(4000 * scale),
        "ood_n": int(4000 * scale),
        "ood_phrasing": int(4000 * scale),
        "ood_relation": int(4000 * scale),
    }
    # val/ood splits. TRAIN itself is generated from TRAIN_CSP_MIX (below) so it occupies the hard
    # difficulty buckets. ood_level / ood_depth are the NEW one-bucket-past-trained-max splits: an
    # affine system a level beyond the trained max, and a propagation chain deeper than the trained max.
    plan = {
        "val": dict(kind="curriculum", families=TRAIN_FAMILIES, n=TRAIN_N, skins=TRAIN_SKINS,
                    base_seed=seed + 2_000_000),
        "ood_n": dict(kind="curriculum", families=OODN_FAMILIES, n=WIDE_N, skins=TRAIN_SKINS,
                      base_seed=seed + 3_000_000),
        "ood_phrasing": dict(kind="curriculum", families=PHRASING_FAMILIES, n=TRAIN_N,
                             skins=HELDOUT_SKINS, base_seed=seed + 4_000_000),
        "ood_relation": dict(kind="hard", families=None, n=None, skins=TRAIN_SKINS,
                             base_seed=seed + 5_000_000),
        "ood_level": dict(kind="curriculum", families=("affine_l4",), n=TRAIN_N, skins=TRAIN_SKINS,
                          base_seed=seed + 6_000_000),
        "ood_depth": dict(kind="hard", families=("eqchain", "forcedcolor"), n=None, skins=TRAIN_SKINS,
                          base_seed=seed + 7_000_000, lengths=LONG_LENGTHS),
    }
    # broadened-domain volumes (scaled): per-domain train share + train compositions, plus the two new
    # eval splits (per-domain held-out + OOD-composition).
    n_extra_each = int(2500 * scale)         # each broadened single-domain in TRAIN
    n_compose_each = int(1500 * scale)       # each train composition pairing in TRAIN
    n_ood_domain_each = int(1000 * scale)    # each broadened single-domain in ood_domain
    n_ood_comp_each = int(1500 * scale)      # each held-out composition pairing in ood_composition
    n_ood_comp_triple = int(1000 * scale)

    workers = workers or os.cpu_count() or 1
    by_split, attempts = {}, {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        # ---- TRAIN: the difficulty-targeted CSP-spine mix (fills the hard buckets) ----
        train_recs, train_agg, ci = [], {k: 0 for k in
            ("trivial", "ambiguous", "skin_fallback", "label_mismatch", "gen_fail", "attempts")}, 0
        for kind, mc, frac in TRAIN_CSP_MIX:
            cnt = int(round(counts["train"] * frac))
            if cnt <= 0:
                continue
            recs, ap = build_split("train", kind, cnt, seed + 1_000_000 + ci * 100_000, TRAIN_SKINS,
                                   families=mc.get("families"), n=mc.get("n"),
                                   nshards=nshards, pool=pool)
            train_recs.extend(recs)
            for k in train_agg:
                train_agg[k] += ap[k]
            tag = (mc.get("families") or (kind,))
            print(f"  [train:{str(tag)[:28]:28s}] {len(recs):6d}  {time.time()-t0:.0f}s", flush=True)
            ci += 1
        by_split["train"], attempts["train"] = train_recs, train_agg
        print(f"  [{'train':13s}] generated {len(train_recs):6d} (CSP-spine mix)  "
              f"{time.time()-t0:.0f}s", flush=True)

        for split, cfg in plan.items():
            # ood_relation is half parity_k, half hard_tasks chains.
            if split == "ood_relation":
                half = counts[split] // 2
                recs_p, ap = build_split(split, "parity", half, cfg["base_seed"],
                                         cfg["skins"], nshards=nshards, pool=pool)
                recs_h, ah = build_split(split, "hard", counts[split] - half,
                                         cfg["base_seed"] + 500_000, cfg["skins"],
                                         nshards=nshards, pool=pool)
                by_split[split] = recs_p + recs_h
                attempts[split] = {k: ap[k] + ah[k] for k in ap}
            else:
                recs, ap = build_split(split, cfg["kind"], counts.get(split, int(4000 * scale)),
                                       cfg["base_seed"], cfg["skins"], families=cfg["families"],
                                       n=cfg["n"], nshards=nshards, pool=pool,
                                       lengths=cfg.get("lengths"))
                by_split[split] = recs
                attempts[split] = ap
            print(f"  [{split:13s}] generated {len(by_split[split]):6d}  "
                  f"(attempts {attempts[split]['attempts']}, ambiguous {attempts[split]['ambiguous']}, "
                  f"trivial {attempts[split]['trivial']})  {time.time()-t0:.0f}s", flush=True)

        # ---- BROADENED single domains: appended to TRAIN + a per-domain held-out (ood_domain) ----
        bs = seed + 10_000_000
        for di, dom in enumerate(EXTRA_DOMAINS):
            recs, ap = build_split("train", "extra", n_extra_each, bs + di * 100_000, None,
                                   nshards=nshards, pool=pool, domain=dom,
                                   cfg=EXTRA_TRAIN_CFG[dom])
            by_split["train"].extend(recs)
            recs_o, _ = build_split("ood_domain", "extra", n_ood_domain_each,
                                    bs + di * 100_000 + 50_000, None, nshards=nshards, pool=pool,
                                    domain=dom, cfg=EXTRA_OOD_CFG[dom])
            by_split.setdefault("ood_domain", []).extend(recs_o)
            print(f"  [extra:{dom:9s}] train {len(recs):5d}  ood_domain {len(recs_o):5d}  "
                  f"(attempts {ap['attempts']}, gen_fail {ap['gen_fail']})  {time.time()-t0:.0f}s",
                  flush=True)

        # ---- COMPOSITIONS: train pairings -> TRAIN; held-out pairings -> ood_composition ----
        cs = seed + 20_000_000
        for hi, head in enumerate(COMPOSE_TRAIN_HEADS):
            recs, ap = build_split("train", "compose", n_compose_each, cs + hi * 100_000, None,
                                   nshards=nshards, pool=pool, head=head, cfg={})
            by_split["train"].extend(recs)
            print(f"  [compose:{head:9s}->train] {len(recs):5d} (gen_fail {ap['gen_fail']})  "
                  f"{time.time()-t0:.0f}s", flush=True)
        cs2 = seed + 30_000_000
        for hi, head in enumerate(COMPOSE_OOD_HEADS):
            recs, ap = build_split("ood_composition", "compose", n_ood_comp_each,
                                   cs2 + hi * 100_000, None, nshards=nshards, pool=pool,
                                   head=head, cfg={})
            by_split.setdefault("ood_composition", []).extend(recs)
            print(f"  [compose:{head:9s}->ood ] {len(recs):5d} (gen_fail {ap['gen_fail']})  "
                  f"{time.time()-t0:.0f}s", flush=True)
        recs, ap = build_split("ood_composition", "compose", n_ood_comp_triple,
                               cs2 + 900_000, None, nshards=nshards, pool=pool,
                               head=COMPOSE_OOD_TRIPLE[0], triple=COMPOSE_OOD_TRIPLE[1], cfg={})
        by_split["ood_composition"].extend(recs)
        print(f"  [compose:triple ->ood ] {len(recs):5d} (gen_fail {ap['gen_fail']})  "
              f"{time.time()-t0:.0f}s", flush=True)

    # global near-dup removal in split order (no train->eval leakage)
    rng = np.random.default_rng(seed)
    by_split, dropped = dedup_in_order(by_split, rng)
    print(f"  dedup dropped: {dropped}", flush=True)

    # independent re-verification from serialized records — PARALLEL: re-deriving a wide-N label is a
    # full UNSAT proof per non-surviving value, so this is run across all cores (serial would dominate
    # wall time on the wide-N split).
    vrng = np.random.default_rng(seed + 99)
    to_check = []
    for sp, recs in by_split.items():
        if verify_frac >= 1.0:
            idxs = range(len(recs))
        else:
            idxs = vrng.choice(len(recs), size=max(1, int(len(recs) * verify_frac)), replace=False)
        to_check.extend(recs[i] for i in idxs)
    with ProcessPoolExecutor(max_workers=workers) as vpool:
        results = list(vpool.map(verify_any, to_check, chunksize=64))
    nver, nbad = len(results), results.count(False)
    print(f"  re-verification: {nver} records checked from serialized payload/cons (domain-dispatched), "
          f"{nbad} mismatches  {time.time()-t0:.0f}s", flush=True)
    assert nbad == 0, "a serialized record failed exact re-verification"

    # write splits + schema + stats + sample
    os.makedirs(out_dir, exist_ok=True)
    sizes = {}
    for sp, recs in by_split.items():
        write_split(out_dir, sp, recs)
        sizes[sp] = len(recs)
    with open(os.path.join(out_dir, "schema.json"), "w") as fh:
        json.dump(schema_dict(), fh, indent=2)

    # corpus-wide per-domain + per-composition tallies (the headline broadened-coverage numbers)
    all_recs = [r for recs in by_split.values() for r in recs]
    by_domain = _count(all_recs, "domain")
    by_compose = _count([r for r in all_recs if r["n_domains"] > 1], "compose")
    by_ndomains = _count(all_recs, "n_domains")

    # per-split diversity + sample
    stats = {"schema_version": SCHEMA_VERSION, "seed": seed, "scale": scale,
             "sizes": sizes, "total": sum(sizes.values()),
             "verified": nver, "verify_mismatches": nbad, "dedup_dropped": dropped,
             "by_domain": by_domain, "by_composition": by_compose,
             "by_n_domains": {str(k): v for k, v in sorted(by_ndomains.items())},
             "splits": {}}
    drng = np.random.default_rng(seed + 7)
    sample = []
    for sp, recs in by_split.items():
        # diversity is measured per-domain WITHIN the split (mixed-domain self-BLEU is meaningless).
        rends = [render.Rendering(r["text"], r["skin"], r["domain_kind"], r["entities"],
                                  r["values"], r["naming_scheme"], r["structure"], r["question"])
                 for r in recs]
        stats["splits"][sp] = qc.diversity_report(rends, drng) if rends else {}
        stats["splits"][sp]["families"] = _count(recs, "family")
        stats["splits"][sp]["domains"] = _count(recs, "domain")
        stats["splits"][sp]["determined_frac"] = round(
            sum(r["determined"] for r in recs) / max(1, len(recs)), 3)
        # per-domain diversity inside this split (non-stilted check per domain, not across domains)
        perdom = {}
        for dom in set(r["domain"] for r in recs):
            drecs = [r for r in recs if r["domain"] == dom]
            drends = [render.Rendering(r["text"], r["skin"], r["domain_kind"], r["entities"],
                                       r["values"], r["naming_scheme"], r["structure"], r["question"])
                      for r in drecs]
            rep = qc.diversity_report(drends, drng)
            perdom[dom] = {"n": rep["n"], "self_bleu": rep["self_bleu"],
                           "distinct_3": rep["distinct_3"], "top_skeleton_share": rep["collapse"]["top_share"]}
        stats["splits"][sp]["per_domain_diversity"] = perdom
        # a sample drawn across domains (first 2 of each domain present)
        for dom in sorted(set(r["domain"] for r in recs)):
            for r in [x for x in recs if x["domain"] == dom][:2]:
                sample.append(r)
    with open(os.path.join(out_dir, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    with open(os.path.join(out_dir, "sample.jsonl"), "w") as fh:
        for r in sample:
            fh.write(json.dumps(r) + "\n")
    print(f"\nDONE in {time.time()-t0:.0f}s — sizes {sizes} total {sum(sizes.values())}", flush=True)
    return stats


def _count(recs, key):
    from collections import Counter
    return dict(Counter(r[key] for r in recs))


def main():
    ap = argparse.ArgumentParser(description="Build the GLaDOS reasoning corpus (rule-based, exact).")
    ap.add_argument("--out", default="data/glados_corpus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all split sizes")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--nshards", type=int, default=16)
    ap.add_argument("--verify-frac", type=float, default=1.0)
    args = ap.parse_args()
    build(args.out, seed=args.seed, scale=args.scale, nshards=args.nshards,
          workers=args.workers, verify_frac=args.verify_frac)


if __name__ == "__main__":
    main()
