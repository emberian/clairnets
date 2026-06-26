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

SCHEMA_VERSION = "1.0"

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
TRAIN_FAMILIES = ("coloring", "equality", "ordering", "arithmetic", "alldiff")
# OOD-N uses only arity<=3 families (alldiff's arity-N extensional table is infeasible at wide N).
OODN_FAMILIES = ("coloring", "equality", "ordering", "arithmetic")
# OOD-phrasing uses families that have >=1 held-out skin (coloring/equality/ordering/alldiff).
PHRASING_FAMILIES = ("coloring", "equality", "ordering", "alldiff")
HARD_FAMILIES = ("eqchain", "forcedcolor")
HARD_LENGTHS = (3, 4, 5, 6, 7, 8)

TRAIN_N = (4, 7)
WIDE_N = (8, 11)
MIN_FACTS = 2                                               # drop trivial problems

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
    if family == "alldiff":
        n_hi = min(n_hi, 5)         # alldiff's arity-N extensional table is d^n; keep it cheap/small
    gen = CU.GENERATORS[family]
    n, d, kind, facts, s = gen(rng, n_lo=n_lo, n_hi=n_hi, **DENSITY.get(family, {}))
    assert CU.facts_satisfied_by(facts, s, d), "witness violated a fact (generator bug)"
    csp = CU.build_csp(n, d, facts)
    q, ans, det = _pick_query(csp, facts, rng, det_target)
    return CU.Problem(family, n, d, kind, facts, q, ans, det, CU.value_names(kind, d)), csp


def make_parity(rng, det_target):
    n, d, kind, facts, s = CU.gen_parity_k(rng, n_lo=5, n_hi=8, kmin=3, kmax=5)
    assert CU.facts_satisfied_by(facts, s, d), "witness violated a parity fact"
    csp = CU.build_csp(n, d, facts)
    q, ans, det = _pick_query(csp, facts, rng, det_target)
    return CU.Problem("parity_k", n, d, kind, facts, q, ans, det, CU.value_names(kind, d)), csp


def make_hard(rng, family, det_target):
    L = int(rng.choice(HARD_LENGTHS))
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
        "naming_scheme": rendering.scheme,
        "structure": rendering.structure,
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


# --------------------------------------------------------------------------- one record
def _gen_one(spec, rng, idx):
    """Generate -> render -> trace -> QC one record for split `spec`. Returns (rec, status)."""
    split, family = spec["split"], None
    det_target = bool(rng.random() < 0.6)
    if spec["kind"] == "curriculum":
        family = spec["families"][idx % len(spec["families"])]
        p, csp = make_curriculum(rng, family, spec["n"][0], spec["n"][1], det_target)
    elif spec["kind"] == "parity":
        family = "parity_k"
        p, csp = make_parity(rng, det_target)
    else:                                                  # hard
        family = HARD_FAMILIES[idx % len(HARD_FAMILIES)]
        p, csp = make_hard(rng, family, det_target)

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
    stats = {"trivial": 0, "ambiguous": 0, "skin_fallback": 0, "label_mismatch": 0, "attempts": 0}
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
def _shard_specs(split, kind, count, base_seed, skins, families=None, n=None, nshards=16):
    """Split `count` kept records across `nshards` deterministic shards."""
    per = [count // nshards] * nshards
    for i in range(count % nshards):
        per[i] += 1
    specs = []
    for sh in range(nshards):
        if per[sh] == 0:
            continue
        spec = {"split": split, "kind": kind, "seed": base_seed + sh,
                "skins": skins, "families": families, "n": n}
        specs.append((spec, per[sh]))
    return specs


def build_split(split, kind, count, base_seed, skins, families=None, n=None,
                nshards=16, pool=None):
    specs = _shard_specs(split, kind, count, base_seed, skins, families, n, nshards)
    results = list(pool.map(_gen_shard, specs)) if pool else [_gen_shard(s) for s in specs]
    records, agg = [], {"trivial": 0, "ambiguous": 0, "skin_fallback": 0,
                        "label_mismatch": 0, "attempts": 0}
    for recs, st in results:
        records.extend(recs)
        for k in agg:
            agg[k] += st[k]
    return records, agg


# --------------------------------------------------------------------------- dedup + write
def dedup_in_order(by_split, rng):
    """Global MinHash/LSH near-dup removal across ALL splits in a fixed order, so later (eval)
    splits never keep a near-duplicate of an earlier (train) record — no train->eval leakage."""
    order = ["train", "val", "ood_n", "ood_phrasing", "ood_relation"]
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
                json.dumps(v) if k in ("facts", "cons", "entities", "values") else v)
    pq.write_table(pa.table(cols), os.path.join(out_dir, f"{split}.parquet"))
    return jsonl


def schema_dict():
    return {
        "schema_version": SCHEMA_VERSION,
        "fields": {
            "id": "str — unique record id (split-prefixed content hash)",
            "split": "str — train|val|ood_n|ood_phrasing|ood_relation",
            "family": "str — generator family (coloring/equality/ordering/arithmetic/alldiff/parity_k/eqchain/forcedcolor)",
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
            "facts": "list — structured ground-truth facts (JSON in parquet)",
            "cons": "list — extensional CSP (scope, allowed-tuples) for re-verification (JSON in parquet)",
            "naming_scheme": "str — entity naming scheme used",
            "structure": "str — layout structure (prose/semicolon/bullets/numbered)",
            "seed": "int — shard seed that produced the record",
        },
        "exact_verification": "rebuild clair.csp.CSP from `cons`, run clair.hard_tasks.fast_dedP, "
                              "confirm cell[query] == answer_index (singleton) or ABSTAIN.",
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
    plan = {
        "train": dict(kind="curriculum", families=TRAIN_FAMILIES, n=TRAIN_N, skins=TRAIN_SKINS,
                      base_seed=seed + 1_000_000),
        "val": dict(kind="curriculum", families=TRAIN_FAMILIES, n=TRAIN_N, skins=TRAIN_SKINS,
                    base_seed=seed + 2_000_000),
        "ood_n": dict(kind="curriculum", families=OODN_FAMILIES, n=WIDE_N, skins=TRAIN_SKINS,
                      base_seed=seed + 3_000_000),
        "ood_phrasing": dict(kind="curriculum", families=PHRASING_FAMILIES, n=TRAIN_N,
                             skins=HELDOUT_SKINS, base_seed=seed + 4_000_000),
        "ood_relation": dict(kind="hard", families=None, n=None, skins=TRAIN_SKINS,
                             base_seed=seed + 5_000_000),
    }
    workers = workers or os.cpu_count() or 1
    by_split, attempts = {}, {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
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
                recs, ap = build_split(split, cfg["kind"], counts[split], cfg["base_seed"],
                                       cfg["skins"], families=cfg["families"], n=cfg["n"],
                                       nshards=nshards, pool=pool)
                by_split[split] = recs
                attempts[split] = ap
            print(f"  [{split:13s}] generated {len(by_split[split]):6d}  "
                  f"(attempts {attempts[split]['attempts']}, ambiguous {attempts[split]['ambiguous']}, "
                  f"trivial {attempts[split]['trivial']})  {time.time()-t0:.0f}s", flush=True)

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
        results = list(vpool.map(verify_record, to_check, chunksize=64))
    nver, nbad = len(results), results.count(False)
    print(f"  re-verification: {nver} records checked from serialized cons, {nbad} mismatches  "
          f"{time.time()-t0:.0f}s", flush=True)
    assert nbad == 0, "a serialized record failed exact re-verification"

    # write splits + schema + stats + sample
    os.makedirs(out_dir, exist_ok=True)
    sizes = {}
    for sp, recs in by_split.items():
        write_split(out_dir, sp, recs)
        sizes[sp] = len(recs)
    with open(os.path.join(out_dir, "schema.json"), "w") as fh:
        json.dump(schema_dict(), fh, indent=2)

    # per-split diversity + sample
    stats = {"schema_version": SCHEMA_VERSION, "seed": seed, "scale": scale,
             "sizes": sizes, "total": sum(sizes.values()),
             "verified": nver, "verify_mismatches": nbad, "dedup_dropped": dropped,
             "splits": {}}
    drng = np.random.default_rng(seed + 7)
    sample = []
    for sp, recs in by_split.items():
        rends = [render.Rendering(r["text"], r["skin"], r["domain_kind"], r["entities"],
                                  r["values"], r["naming_scheme"], r["structure"], r["question"])
                 for r in recs]
        stats["splits"][sp] = qc.diversity_report(rends, drng) if rends else {}
        stats["splits"][sp]["families"] = _count(recs, "family")
        stats["splits"][sp]["determined_frac"] = round(
            sum(r["determined"] for r in recs) / max(1, len(recs)), 3)
        for r in recs[:4]:
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
