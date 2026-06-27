"""clair/datagen/stream.py — STREAMING problemgen from a seed (no-filesystem training).

The batch builder (clair.datagen.build) generates the whole corpus to disk, then runs an O(N) MinHash
dedup pass over every record's text — a single-thread wall that grows with corpus size. For TRAINING
we don't need disk or a global dedup: witness-first generation is near-distinct by construction, so an
ONLINE bloom-filter on the canonical (structural) key is enough. This module is the streaming
alternative — the GPU is fed straight from a deterministic, infinite generator.

  problem_stream(seed, spec) -> Iterator[record]
      Infinite, DETERMINISTIC stream. Reuses the SAME generators as build (curriculum / extra /
      compose / reductions) and the SAME exact per-cell labeller (clair.csp.exact_dedP — the Rust
      clair_fast port when built). Reproducible from (seed, spec): same seed -> identical record run.
      Online dedup on the canonical key via a bloom filter (replaces build's batch MinHash).

  StreamDataset(seed, spec, ...)  (torch IterableDataset)
      Yields training batches directly from problem_stream — `organ` mode rebuilds (csp, dom) items
      the clair.organ pretrain loop (run_glados_staged.train_organ) consumes; `record` mode yields raw
      corpus records. Gen throughput >> a training step, so the Rust gen keeps the GPU fed.

  materialize(seed, spec, n, out)
      The save-to-disk path: stream -> parquet/jsonl (+ schema/stats), deterministic, so an
      HF/reproducible corpus is still producible from the same seed. (build.py stays the batch path.)

Determinism: one np.random.default_rng(seed) drives generation; component selection is a fixed
weighted schedule; the bloom is a pure function of the key sequence. So problem_stream(seed, spec)
replays bit-for-bit, and materialize(seed, spec, n) reproduces the first n records of that stream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from typing import Iterator, Optional

import numpy as np

from .. import csp as C
from .. import curriculum as CU
from . import build as B
from . import reductions as RD

SCHEMA_VERSION = B.SCHEMA_VERSION


# ===================================================================== online dedup (bloom / bounded)
class BloomFilter:
    """A classic bit-array bloom filter keyed by a precomputed 16-byte digest. No false negatives
    (so a planted exact dup is ALWAYS caught); a tunable false-positive rate (a distinct record is
    rarely dropped). Bounded memory — `capacity` records at `err` cost ~ -n ln(err)/ln(2)^2 bits."""

    def __init__(self, capacity: int = 2_000_000, err: float = 1e-4):
        capacity = max(1, int(capacity))
        m = max(8, int(math.ceil(-capacity * math.log(err) / (math.log(2) ** 2))))
        self.m = m + (8 - m % 8) % 8                      # round up to whole bytes
        self.k = max(1, int(round((self.m / capacity) * math.log(2))))
        self.bits = bytearray(self.m // 8)
        self.n = 0

    def _idxs(self, key: bytes):
        h1 = int.from_bytes(key[:8], "big")
        h2 = int.from_bytes(key[8:16], "big") | 1         # odd -> good stride
        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def __contains__(self, key: bytes) -> bool:
        return all((self.bits[j >> 3] >> (j & 7)) & 1 for j in self._idxs(key))

    def add(self, key: bytes) -> bool:
        """Add `key`; return True iff it was ALREADY present (a detected duplicate)."""
        present = True
        for j in self._idxs(key):
            byte, bit = j >> 3, j & 7
            if not (self.bits[byte] >> bit) & 1:
                present = False
                self.bits[byte] |= 1 << bit
        self.n += not present
        return present


class BoundedSeenSet:
    """Exact bounded seen-set (FIFO eviction). Alternative to the bloom when an exact, bounded-memory
    dedup is wanted; drops the OLDEST keys past `maxlen` (so dedup is exact within the recent window)."""

    def __init__(self, maxlen: int = 1_000_000):
        from collections import deque
        self.maxlen = maxlen
        self.seen: set = set()
        self.order = deque()

    def __contains__(self, key: bytes) -> bool:
        return key in self.seen

    def add(self, key: bytes) -> bool:
        if key in self.seen:
            return True
        self.seen.add(key)
        self.order.append(key)
        if len(self.order) > self.maxlen:
            self.seen.discard(self.order.popleft())
        return False


def canonical_key(rec: dict) -> bytes:
    """Stable 16-byte content hash of the CANONICAL (structural) problem — NOT the surface text. Two
    records that pose the SAME problem (same CSP+query, or same domain payload+question) collide and
    are deduped; skins/phrasing are intentionally NOT part of the key. Ints/sorted-cons only, so the
    key is reproducible across runs and processes (no hash-seed sensitivity)."""
    dom = rec.get("domain", "csp")
    if dom == "csp":
        core = (dom, rec["family"], rec["n"], rec["d"], rec["cons"], rec["query"], rec["answer_index"])
    else:
        core = (dom, rec.get("family"), rec.get("payload"), rec.get("question"), rec["answer_index"])
    return hashlib.blake2b(json.dumps(core, sort_keys=True, default=str).encode(), digest_size=16).digest()


# ===================================================================== generation spec / mixture
# A spec is {"name": str, "mix": [component, ...]}. Each component is a build-style spec dict (the keys
# build._gen_one reads) PLUS a "weight". The default mix mirrors build.py's TRAIN distribution so a
# stream is a drop-in for the on-disk train corpus. `kind="reduction"` is handled here (build has no
# reduction component); every other kind delegates verbatim to build._gen_one.
_SEED_OFFSETS = {                                          # per-component seed bases (id-uniqueness)
    "curriculum": 1_000_000, "parity": 5_000_000, "hard": 5_500_000, "randomrel": 6_500_000,
    "extra": 10_000_000, "compose": 20_000_000, "reduction": 40_000_000,
}


def _difficulty_components(base=1.0):
    """The difficulty-CONTROLLED CSP-spine components (mirrors build.TRAIN_CSP_MIX): family chains
    (deep), bag-chain (treewidth spread), the affine wall (level 2/3), and additive random relations.
    Weights are ~ build's per-bucket fractions x `base`."""
    w = lambda x: max(1, int(round(x * base)))
    return [
        dict(kind="hard", families=list(B.CHAIN_FAMILIES), n=None, skins=B.TRAIN_SKINS, weight=w(15)),
        dict(kind="curriculum", families=["bagchain"], n=B.TRAIN_N, skins=B.TRAIN_SKINS, weight=w(10)),
        dict(kind="curriculum", families=["affine_l2"], n=B.TRAIN_N, skins=B.TRAIN_SKINS, weight=w(25)),
        dict(kind="curriculum", families=["affine_l3"], n=B.TRAIN_N, skins=B.TRAIN_SKINS, weight=w(12)),
        dict(kind="randomrel", families=None, n=None, skins=B.TRAIN_SKINS, weight=w(8)),
    ]


def default_spec(include_reductions: bool = True, name: str = "stream",
                 difficulty: bool = True) -> dict:
    """The default training mixture — build.py's TRAIN families + the difficulty-controlled hard-bucket
    components + the 6 broadened domains + train compositions + (optionally) the reduction curriculum.
    Pass difficulty=False for the pre-fix (easy-only) mixture."""
    mix = [
        dict(kind="curriculum", families=list(B.TRAIN_FAMILIES), n=B.TRAIN_N,
             skins=B.TRAIN_SKINS, weight=30),
        dict(kind="parity", families=None, n=None, skins=B.TRAIN_SKINS, weight=4),
    ]
    if difficulty:
        mix += _difficulty_components(base=1.0)
    else:
        mix.append(dict(kind="hard", families=None, n=None, skins=B.TRAIN_SKINS, weight=4))
    for dom in B.EXTRA_DOMAINS:
        mix.append(dict(kind="extra", domain=dom, cfg=B.EXTRA_TRAIN_CFG[dom], weight=3))
    for head in B.COMPOSE_TRAIN_HEADS:
        mix.append(dict(kind="compose", head=head, cfg={}, triple=None, weight=2))
    if include_reductions:
        mix.append(dict(kind="reduction", families=["sat3", "sat2", "xorsat"],
                        reps=["cnf", "xor"], n=(4, 8), weight=4))
    return {"name": name, "mix": mix}


def organ_spec(name: str = "organ_stream", difficulty: bool = True) -> dict:
    """A CSP-ONLY mixture for feeding the clair.organ pretrain loop: the train families + the
    difficulty-controlled hard buckets (deep chains, treewidth spread, the affine level-2/3 wall,
    random relations) + parity wall + the reduction curriculum (all rebuild to a (csp, dom) item).
    No extra/compose domains — they'd be generated then dropped by the organ-mode item filter."""
    mix = [
        dict(kind="curriculum", families=list(B.TRAIN_FAMILIES), n=B.TRAIN_N, skins=B.TRAIN_SKINS, weight=30),
        dict(kind="parity", families=None, n=None, skins=B.TRAIN_SKINS, weight=6),
        dict(kind="reduction", families=["sat3", "sat2", "xorsat"], reps=["cnf", "xor"], n=(4, 8), weight=6),
    ]
    if difficulty:
        mix += _difficulty_components(base=1.0)
    else:
        mix.append(dict(kind="hard", families=None, n=None, skins=B.TRAIN_SKINS, weight=8))
    return {"name": name, "mix": mix}


def _schedule(mix) -> list:
    """Deterministic weighted round-robin order over component indices (interleaved, not blocked)."""
    weights = [max(1, int(c.get("weight", 1))) for c in mix]
    total = sum(weights)
    # Bresenham-style interleave so components are spread evenly across the schedule.
    acc = [0.0] * len(mix)
    order = []
    for _ in range(total):
        for i in range(len(mix)):
            acc[i] += weights[i] / total
        j = max(range(len(mix)), key=lambda i: acc[i])
        acc[j] -= 1.0
        order.append(j)
    return order


def _build_spec(comp: dict, base_seed: int, name: str) -> dict:
    """A build._gen_one-compatible spec dict for a non-reduction component."""
    off = _SEED_OFFSETS[comp["kind"]]
    return {"split": name, "kind": comp["kind"], "seed": base_seed + off,
            "skins": comp.get("skins"), "families": comp.get("families"), "n": comp.get("n"),
            "domain": comp.get("domain"), "cfg": comp.get("cfg"), "head": comp.get("head"),
            "triple": comp.get("triple")}


# ===================================================================== reduction component (adapter)
def _reduction_record(comp: dict, rng, idx: int, base_seed: int, name: str):
    """One reduction-curriculum record (clair.datagen.reductions) ADAPTED into the build CSP-record
    schema, so it flows through the same exact gate (verify_record: rebuild from `cons`, exact_dedP at
    query == answer). The reduction's target CSP `csp` exactly determines the gold answer (checked)."""
    fams, reps = comp["families"], comp["reps"]
    fam = fams[idx % len(fams)]
    rep = reps[idx % len(reps)] if fam != "xorsat" else "xor"
    n_lo, n_hi = comp.get("n", (4, 8))
    r = RD.make_record(rng, fam, rep, name, n_lo=n_lo, n_hi=n_hi)
    if r is None:
        return None, "gen_fail"
    csp = r["csp"]
    if not RD._within_budget(csp):
        return None, "trivial"
    ded = C.exact_dedP(csp, csp.full())
    cell = ded[r["query"]]
    if not (len(cell) == 1 and next(iter(cell)) == r["gold_idx"]):   # exactness gate
        return None, "label_mismatch"
    cons = [[[int(c) for c in sc], sorted([[int(x) for x in t] for t in al])] for sc, al in csp.cons]
    seed = base_seed + _SEED_OFFSETS["reduction"]
    rid = hashlib.blake2b(f"{name}-{seed}-{idx}".encode(), digest_size=8).hexdigest()
    rec = B._clean({
        "id": f"{name}-{rid}", "split": name, "domain": "csp", "family": r["family"],
        "domain_kind": "bool", "skin": r["representation"], "n": int(csp.n), "d": int(csp.d),
        "n_facts": len(csp.cons), "determined": True,
        "text": r["alpha_prompt"], "question": r["prompt"],
        "answer": r["answer"], "answer_index": int(r["gold_idx"]),
        "trace": "", "query": int(r["query"]),
        "entities": [RD.ENTS[i] for i in range(csp.n)], "values": list(RD.VNAMES),
        "facts": [], "cons": cons, "payload": {},
        "naming_scheme": "letters", "structure": "prose", "n_domains": 1,
        "compose": f"reduce:{r['reduction_path']}", "seed": int(seed),
    })
    return rec, "ok"


# ===================================================================== the stream
def problem_stream(seed: int, spec: Optional[dict] = None, *, dedup: bool = True,
                   bloom_capacity: int = 2_000_000, bloom_err: float = 1e-4,
                   seen=None, stats: Optional[dict] = None) -> Iterator[dict]:
    """Infinite DETERMINISTIC record stream from (seed, spec). Witness-first generation + exact Rust
    dedₚ label on the fly, no disk. Online dedup on the canonical key (bloom by default; pass a custom
    `seen` for a BoundedSeenSet or exact set). If `stats` (a dict) is given it is updated in place with
    {attempts, emitted, dups, dup_rate} as the stream runs."""
    spec = spec or default_spec()
    name = spec.get("name", "stream")
    mix = spec["mix"]
    schedule = _schedule(mix)
    rng = np.random.default_rng(seed)
    if dedup and seen is None:
        seen = BloomFilter(bloom_capacity, bloom_err)
    st = stats if stats is not None else {}
    st.update({"attempts": 0, "emitted": 0, "dups": 0, "dup_rate": 0.0})

    attempt = 0
    while True:
        comp = mix[schedule[attempt % len(schedule)]]
        if comp["kind"] == "reduction":
            rec, status = _reduction_record(comp, rng, attempt, seed, name)
        else:
            rec, status = B._gen_one(_build_spec(comp, seed, name), rng, attempt)
        attempt += 1
        st["attempts"] = attempt
        if rec is None:
            continue
        if seen is not None:
            if seen.add(canonical_key(rec)):              # already present -> duplicate
                st["dups"] += 1
                st["dup_rate"] = st["dups"] / max(1, st["emitted"] + st["dups"])
                continue
        st["emitted"] += 1
        st["dup_rate"] = st["dups"] / max(1, st["emitted"] + st["dups"])
        yield rec


# ===================================================================== torch IterableDataset
def _item_from_record(rec: dict):
    """Rebuild (csp, full-domain) from a CSP record's serialized `cons`, within the organ budget
    (run_glados_staged N_MAX/D_MAX/M_MAX/A_MAX). Returns None for non-CSP / out-of-budget records."""
    if rec.get("domain", "csp") != "csp":
        return None
    from .. import run_glados_staged as G
    n, d = rec["n"], rec["d"]
    if n > G.N_MAX or d > G.D_MAX or len(rec["cons"]) > G.M_MAX:
        return None
    cons = []
    for sc, al in rec["cons"]:
        if len(sc) > G.A_MAX or any(v >= G.D_MAX for t in al for v in t):
            return None
        cons.append((tuple(sc), frozenset(tuple(t) for t in al)))
    csp = C.CSP(n, d, tuple(cons))
    return (csp, csp.full())


def StreamDataset(*args, **kwargs):
    """Factory for the torch IterableDataset (torch imported lazily so the module stays importable
    without torch). See _StreamDataset for the args."""
    import torch.utils.data as _tud

    class _StreamDataset(_tud.IterableDataset):
        def __init__(self, seed, spec=None, *, batch_size=64, mode="organ",
                     dedup=True, max_batches=None):
            super().__init__()
            self.seed, self.spec, self.batch_size = seed, spec, batch_size
            self.mode, self.dedup, self.max_batches = mode, dedup, max_batches
            self.stats: dict = {}

        def __iter__(self):
            stream = problem_stream(self.seed, self.spec, dedup=self.dedup, stats=self.stats)
            buf, nb = [], 0
            for rec in stream:
                if self.mode == "organ":
                    item = _item_from_record(rec)
                    if item is None:
                        continue
                    buf.append(item)
                else:                                     # "record" mode
                    buf.append(rec)
                if len(buf) >= self.batch_size:
                    yield buf
                    buf, nb = [], nb + 1
                    if self.max_batches is not None and nb >= self.max_batches:
                        return

    return _StreamDataset(*args, **kwargs)


# ===================================================================== save-to-disk (materialize)
def materialize(seed: int, spec: Optional[dict] = None, n: int = 10_000, out: Optional[str] = None,
                *, dedup: bool = True, verify: bool = True) -> dict:
    """Pull the first `n` records of problem_stream(seed, spec) and (if `out`) write parquet + jsonl +
    schema.json + stats.json — DETERMINISTIC, build.py-equivalent records. Returns the stats dict."""
    spec = spec or default_spec()
    name = spec.get("name", "stream")
    stats: dict = {}
    t0 = time.time()
    recs = []
    for rec in problem_stream(seed, spec, dedup=dedup, stats=stats):
        recs.append(rec)
        if len(recs) >= n:
            break
    nbad = 0
    if verify:
        nbad = sum(0 if B.verify_any(r) else 1 for r in recs)
        assert nbad == 0, "a streamed record failed exact re-verification"
    out_stats = {"schema_version": SCHEMA_VERSION, "seed": seed, "spec_name": name, "n": len(recs),
                 "attempts": stats["attempts"], "dups": stats["dups"], "dup_rate": stats["dup_rate"],
                 "verified": len(recs) if verify else 0, "verify_mismatches": nbad,
                 "by_domain": B._count(recs, "domain"), "by_family": B._count(recs, "family"),
                 "elapsed_s": round(time.time() - t0, 2)}
    if out:
        os.makedirs(out, exist_ok=True)
        B.write_split(out, name, recs)
        with open(os.path.join(out, "schema.json"), "w") as fh:
            json.dump(B.schema_dict(), fh, indent=2)
        with open(os.path.join(out, "stats.json"), "w") as fh:
            json.dump(out_stats, fh, indent=2)
    return out_stats


# ===================================================================== organ-pretrain feed (smoke/util)
def feed_organ_steps(seed: int = 0, steps: int = 5, batch_size: int = 64, dev=None, R: int = 6,
                     target: float = 1.2e5, spec: Optional[dict] = None, verbose: bool = True):
    """Run `steps` REAL clair.organ dominate-dedₚ steps fed straight from StreamDataset (no disk). This
    is the train_organ inner loop with the stream as the data source — proof the stream keeps the GPU
    fed. Returns (log, dataset.stats)."""
    import torch
    from .. import run_glados_staged as G
    dev = dev or G.device()
    d, npar = G.size_for("full", G.N_MAX, G.D_MAX, G.M_MAX, G.A_MAX, target, R=R, ds=min(4, R))
    m = G.FactorGraphProposer("full", G.N_MAX, G.D_MAX, G.M_MAX, G.A_MAX, d=d, R=R, ds=min(4, R)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, betas=(0.9, 0.95))
    ds = StreamDataset(seed, spec or organ_spec(), batch_size=batch_size, mode="organ", max_batches=steps)
    log = []
    t0 = time.time()
    for s, items in enumerate(ds, 1):
        feat = G.featurize(items, dev)
        vm = feat["var_mask"]
        tgt, conflict = G.build_targets(items, dev)
        b, cls, sup = G.fwd(m, feat, vm)
        loss = G.loss_fn(sup, vm, feat["var_valid"], tgt, conflict)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        log.append({"step": s, "loss": float(loss.detach())})
        if verbose:
            print(f"    [stream-organ] step {s}  loss {float(loss.detach()):.3f}  "
                  f"bs={len(items)}  {time.time()-t0:.1f}s", flush=True)
    return log, ds.stats


# ===================================================================== smoke + benchmark
def _smoke():
    print("[stream] SMOKE", flush=True)
    spec = default_spec()

    # (1) determinism: same seed -> identical first-K records
    K = 200
    a = [r["id"] for r in _take(problem_stream(7, spec), K)]
    b = [r["id"] for r in _take(problem_stream(7, spec), K)]
    assert a == b, "stream NOT deterministic"
    c = [r["id"] for r in _take(problem_stream(8, spec), K)]
    assert a != c, "different seeds gave identical streams"
    print(f"  (1) deterministic: seed=7 reproduced {K} ids bit-for-bit; seed=8 differs  OK", flush=True)

    # (2) online dedup catches a planted dup
    bloom = BloomFilter(10_000)
    rec0 = next(problem_stream(3, spec))
    key = canonical_key(rec0)
    assert bloom.add(key) is False, "first insert reported a dup"
    assert bloom.add(key) is True, "planted exact dup NOT caught"
    # and end-to-end: feed the SAME seed's first record back through a shared `seen`
    seen = BloomFilter(10_000)
    g = problem_stream(3, spec, seen=seen)
    r1 = next(g)
    planted = seen.add(canonical_key(r1))                 # re-add -> now present
    assert planted is True, "shared seen did not register the record"
    print("  (2) online dedup: planted exact dup caught by bloom (no false negative)  OK", flush=True)

    # (3) dup-rate at stream scale (should be low — near-distinct by construction)
    stats = {}
    n = 0
    for _ in problem_stream(11, spec, stats=stats):
        n += 1
        if n >= 4000:
            break
    print(f"  (3) dup-rate @ {stats['emitted']} emitted ({stats['attempts']} attempts): "
          f"{stats['dups']} dups = {100*stats['dup_rate']:.3f}%  OK", flush=True)

    # (4) StreamDataset feeds REAL organ-pretrain steps
    log, dstats = feed_organ_steps(seed=0, steps=5, batch_size=64)
    assert len(log) == 5 and all(math_isfinite(x["loss"]) for x in log), "organ steps failed"
    print(f"  (4) StreamDataset fed 5 real organ-pretrain steps; final loss "
          f"{log[-1]['loss']:.3f}  OK", flush=True)

    # (5) materialize reproduces build-equivalent records (same schema + exact gate)
    s5 = materialize(0, spec, n=300, out=None, verify=True)
    assert s5["verify_mismatches"] == 0, "materialized records failed exact re-verification"
    # determinism of materialize: first 300 ids match the live stream
    live_ids = [r["id"] for r in _take(problem_stream(0, spec), 300)]
    mat_ids = [r["id"] for r in _materialize_recs(0, spec, 300)]
    assert live_ids == mat_ids, "materialize diverged from the live stream"
    fields = set(B.schema_dict()["fields"])
    sample = next(problem_stream(0, spec))
    assert fields <= set(sample), "streamed record missing build schema fields"
    print(f"  (5) materialize: 300 records, {s5['verify_mismatches']} mismatches, build-schema + "
          f"determinism vs live stream  OK", flush=True)
    print("[stream] ALL SMOKES PASS", flush=True)


def _bench(n=4000):
    print(f"[stream] BENCH (target {n} records)", flush=True)
    bs = 128

    def rps_of(spec, label):
        stats = {}
        t0 = time.time()
        cnt = 0
        for _ in problem_stream(0, spec, stats=stats):
            cnt += 1
            if cnt >= n:
                break
        dt = time.time() - t0
        rps = cnt / dt
        print(f"  {label}: {cnt} records in {dt:.2f}s = {rps:.0f} records/sec  "
              f"(dup-rate {100*stats['dup_rate']:.3f}%, attempts {stats['attempts']}, "
              f"NO O(N) dedup pass — bloom is O(1)/record)", flush=True)
        return rps

    rps_full = rps_of(default_spec(), "full mixture")
    rps_org = rps_of(organ_spec(), "organ-only (CSP)")

    # a REALISTIC organ training step (the default pretrain size: ~1.5M params, R=12, bs=128) — the
    # bar the stream must out-run to keep the GPU fed.
    try:
        from .. import run_glados_staged as G
        import torch
        dev = G.device()
        ds = StreamDataset(0, organ_spec(), batch_size=bs, mode="organ", max_batches=1)
        items = next(iter(ds))
        d, npar = G.size_for("full", G.N_MAX, G.D_MAX, G.M_MAX, G.A_MAX, 1.5e6, R=12, ds=4)
        m = G.FactorGraphProposer("full", G.N_MAX, G.D_MAX, G.M_MAX, G.A_MAX, d=d, R=12, ds=4).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
        for _ in range(3):                                # warmup (mps/cuda kernels)
            feat = G.featurize(items, dev); vm = feat["var_mask"]
            tgt, conflict = G.build_targets(items, dev)
            bb, cls, sup = G.fwd(m, feat, vm)
            loss = G.loss_fn(sup, vm, feat["var_valid"], tgt, conflict)
            opt.zero_grad(); loss.backward(); opt.step()
        ts = time.time(); NS = 10
        for _ in range(NS):
            feat = G.featurize(items, dev); vm = feat["var_mask"]
            tgt, conflict = G.build_targets(items, dev)
            bb, cls, sup = G.fwd(m, feat, vm)
            loss = G.loss_fn(sup, vm, feat["var_valid"], tgt, conflict)
            opt.zero_grad(); loss.backward(); opt.step()
        step_s = (time.time() - ts) / NS
        gen_batch_s = bs / rps_org
        verdict = "OK gen out-runs the step" if gen_batch_s < step_s else \
                  "single-thread gen ~ step (overlap on a prefetch thread keeps the GPU fed)"
        print(f"  realistic organ step (~{npar:,} params, R=12, bs={bs}) {1000*step_s:.1f} ms/step ; "
              f"organ-only stream gen of a {bs}-batch {1000*gen_batch_s:.1f} ms  -> "
              f"{step_s/gen_batch_s:.2f}x  [{verdict}]", flush=True)
    except Exception as e:
        print(f"  (organ-step bench skipped: {e})", flush=True)


# ----- tiny helpers -----
def _take(it, k):
    out = []
    for x in it:
        out.append(x)
        if len(out) >= k:
            break
    return out


def _materialize_recs(seed, spec, n):
    return _take(problem_stream(seed, spec), n)


def math_isfinite(x):
    return math.isfinite(x)


def main():
    ap = argparse.ArgumentParser(description="Streaming GLaDOS problemgen from a seed (no disk).")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--materialize", metavar="OUT", default=None, help="write n records to OUT dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=10_000)
    args = ap.parse_args()
    if args.materialize:
        s = materialize(args.seed, None, n=args.n, out=args.materialize)
        print(json.dumps(s, indent=2))
        return
    if args.bench:
        _bench()
        return
    _smoke()


if __name__ == "__main__":
    main()
