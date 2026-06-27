"""clair/datagen/stream_prefetch.py — parallel, deterministic prefetch of organ-pretrain items.

WHY. Profiling the organ pretrain loop (clair.organ.train._train_organ_stream) at the real config
(~1.5M params, R=12, difficulty stream, pool=128) showed the on-policy RESAMPLE — pulling the next
fresh CSP item from the difficulty stream when a pool slot terminates/stalls — is ~93% of wall time.
Of that, ~90% is the EXACT dedₚ labelling of the HARD difficulty instances inside problem_stream
(deep chains / affine-wall / treewidth spread): exact backtracking enumeration that is inherently
expensive even on the Rust clair_fast port (~30–90 ms per instance). It runs SERIALLY on ONE core, so
the GPU step (fwd+bwd) starves and the other cores idle — the loop is CPU-bound, not GPU-bound.

WHAT. K 'spawn' worker processes each run an INDEPENDENT, deterministic problem_stream (seeded from
SeedSequence([base_seed, w])), filtered to budget-admitted (csp, full-domain) items, and push them
into PER-WORKER bounded queues. The main loop draws ROUND-ROBIN (item t from worker t % K). So the
expensive exact-dedₚ labelling fans out across all cores, and the bounded queues let each worker run
AHEAD of the GPU step (prefetch/overlap). On an n-core box the resample throughput scales ~min(K, n)x,
which is what moves the loop from CPU-starved to GPU-bound.

DETERMINISM. The global item sequence is a pure function of (base_seed, K, spec): worker w's stream is
deterministic, and the round-robin merge order is fixed. So same (base_seed, K) ⇒ identical item
sequence ⇒ identical pretrain run (reproducible). This is a NEW, parallel reproducibility scheme — it
is NOT byte-identical to the old single-stream order, but nothing committed depends on that order (the
pretrain consumes the LIVE stream; no organ corpus is materialized to disk). K is stamped into the
checkpoint meta. workers<=1 ⇒ the caller keeps the exact single-stream serial path (this module unused).

SAFETY. The gen path is torch-free (problem_stream + the budget filter + clair.csp). Workers are
'spawn'ed clean interpreters that never import torch / touch CUDA, so spawning AFTER the trainer's
CUDA/MPS context is initialised is safe — the same rule clair.datagen.fast.FastGen relies on.
"""
from __future__ import annotations

import multiprocessing as mp
import queue as _queue

import numpy as np

from . import stream as S


def _worker(seed: int, spec: dict, budget, q, stop) -> None:
    """Run one deterministic problem_stream, push budget-admitted (csp, full) items into `q`. Blocks on
    a full queue (backpressure) but wakes periodically to honour `stop` so shutdown can't deadlock."""
    n_max, d_max, m_max, a_max = budget
    for rec in S.problem_stream(seed, spec):
        if stop.is_set():
            return
        item = S.item_from_record_budget(rec, n_max, d_max, m_max, a_max)
        if item is None:
            continue
        while True:
            if stop.is_set():
                return
            try:
                q.put(item, timeout=0.5)
                break
            except _queue.Full:
                continue


def _seed_for(base_seed: int, w: int) -> int:
    """Deterministic, well-distributed, distinct per-worker seed (portable across machines)."""
    return int(np.random.SeedSequence([int(base_seed), int(w)]).generate_state(1)[0])


class ParallelItemStream:
    """Persistent pool of K worker streams + a deterministic round-robin reader. Drop-in replacement
    for `(it for _, it in train._difficulty_items(spec, seed))`: call `.next()` for the next item.

    Args:
      spec, base_seed : the difficulty spec + the run seed (drives every worker's sub-stream).
      budget          : (N_MAX, D_MAX, M_MAX, A_MAX) — passed in so the worker stays torch-free.
      workers         : K (>=1). prefetch : per-worker queue depth (items each worker may run ahead).
    """

    def __init__(self, spec: dict, base_seed: int, budget, workers: int = 8, prefetch: int = 64):
        assert workers >= 1
        self.K = int(workers)
        self.spec = spec
        ctx = mp.get_context("spawn")
        self.stop = ctx.Event()
        self.qs = [ctx.Queue(maxsize=prefetch) for _ in range(self.K)]
        self.procs = [ctx.Process(target=_worker,
                                  args=(_seed_for(base_seed, w), spec, tuple(budget), self.qs[w], self.stop),
                                  daemon=True)
                      for w in range(self.K)]
        for p in self.procs:
            p.start()
        self._t = 0
        self._closed = False

    def next(self):
        """The next budget-admitted (csp, full) item, in the deterministic round-robin order."""
        item = self.qs[self._t % self.K].get()       # blocks until worker (t % K) has produced item t//K
        self._t += 1
        return item

    def fill(self, n: int) -> list:
        return [self.next() for _ in range(n)]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop.set()
        # drain so any worker blocked on a full queue can observe `stop` and exit
        for q in self.qs:
            try:
                while True:
                    q.get_nowait()
            except _queue.Empty:
                pass
        for p in self.procs:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
        for q in self.qs:
            q.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
