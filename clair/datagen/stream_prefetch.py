"""clair/datagen/stream_prefetch.py — parallel, deterministic, DEADLOCK-FREE prefetch of organ items.

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
sequence ⇒ identical pretrain run (reproducible). `serial_reference()` computes that exact sequence in
ONE process (no workers) — the parallel pool must match it bit-for-bit (verify_determinism). K is
stamped into the checkpoint meta. workers<=1 ⇒ the caller keeps the exact single-stream serial path.

ROBUSTNESS (the deadlock fix). The OLD code did a BLOCKING `q.get()` on worker (t % K). If that worker
died (one OOM'd in the real pretrain at RAM 23/30GB) the main blocked FOREVER on its empty pipe — the
run hung at step 800 for 71 min and we fell back to serial gen (GPU idle, ~1.5s/step). The fix keeps
the round-robin but makes the read NON-BLOCKING-WITH-TIMEOUT and adds a DETERMINISM-PRESERVING INLINE
FALLBACK: the main holds, per worker w, a lazily-built fallback `problem_stream(seed_w, spec)` (the
SAME seed, SAME budget filter as worker w). On a timeout/dead worker the main computes item t ITSELF
from worker (t % K)'s deterministic stream, advanced to the exact index the worker would be at — so the
recovered item is BIT-IDENTICAL to what the worker would have produced. The sequence is unchanged
whether an item came from the worker's queue or the inline fallback. Two regimes, both correct:
  • truly-dead worker (process exited): latched to inline fallback forever (and its process reaped to
    free RAM). That worker's 1/K share is then served serially by the main — degraded throughput, NOT a
    hang. One-time catch-up cost: the fallback must regenerate the worker's already-consumed prefix
    (problem_stream has no cheap seek), so the first fallback item after a mid-run death stalls ~O(items
    consumed from that worker). This is bounded and one-time — vs the old unbounded hang.
  • merely-slow worker (alive, queue momentarily empty): ONE item is borrowed from the inline fallback,
    then the worker's still-enqueued copy of that index is discarded (queue stays index-aligned) and the
    worker keeps running in parallel. No permanent demotion from a spurious timeout.
We do NOT respawn dead workers: problem_stream has no cheap seek, so a respawn-from-zero would have to
regenerate the consumed prefix too (same O(N) cost) and, starting with no head-start, could never catch
a continuously-consuming main — it is futile without a cheap state-restore. The principled mitigation
for deaths is PREVENTION (the memory budget below), not resurrection; the inline fallback is the simple,
provably-deterministic safety net.

MEMORY BUDGET. The OOM came from too many spawn interpreters alongside the trainer. AUTO now caps at
min(cores-2, 4) workers (was min(8, cores-1)) and each worker's queue is bounded to `prefetch` items.
Rough budget per worker: ~150 MB (a clean spawn interpreter importing numpy + clair.datagen, NO torch)
+ ~5 MB bloom + prefetch × item(~few KB). K=4, prefetch=32 ⇒ < ~0.7 GB total, which fits beside the
~1.5M-param trainer with margin on a 30 GB box.

SAFETY. The gen path is torch-free (problem_stream + the budget filter + clair.csp). Workers are
'spawn'ed clean interpreters that never import torch / touch CUDA, so spawning AFTER the trainer's
CUDA/MPS context is initialised is safe — the same rule clair.datagen.fast.FastGen relies on.
"""
from __future__ import annotations

import multiprocessing as mp
import queue as _queue
import time

import numpy as np

from . import stream as S


def _seed_for(base_seed: int, w: int) -> int:
    """Deterministic, well-distributed, distinct per-worker seed (portable across machines)."""
    return int(np.random.SeedSequence([int(base_seed), int(w)]).generate_state(1)[0])


def _budget_items(seed: int, spec: dict, budget):
    """The deterministic, infinite stream of BUDGET-ADMITTED (csp, full-domain) items for one worker
    seed — the EXACT sequence worker `seed` pushes onto its queue. Used both by the worker process and
    by the main's inline fallback, so the two are bit-identical by construction. Torch-free."""
    n_max, d_max, m_max, a_max = budget
    for rec in S.problem_stream(seed, spec):
        item = S.item_from_record_budget(rec, n_max, d_max, m_max, a_max)
        if item is not None:
            yield item


def _worker(seed: int, spec: dict, budget, q, stop) -> None:
    """Run one deterministic budget-admitted item stream, push items into `q`. Blocks on a full queue
    (backpressure) but wakes periodically to honour `stop` so shutdown can't deadlock."""
    for item in _budget_items(seed, spec, tuple(budget)):
        if stop.is_set():
            return
        while True:
            if stop.is_set():
                return
            try:
                q.put(item, timeout=0.5)
                break
            except _queue.Full:
                continue


class ParallelItemStream:
    """Persistent pool of K worker streams + a deterministic, deadlock-free round-robin reader. Drop-in
    replacement for `(it for _, it in train._difficulty_items(spec, seed))`: call `.next()` for the next
    item. The produced sequence is identical to `serial_reference(spec, base_seed, budget, K)` regardless
    of worker timing, deaths, or stalls (the inline fallback fills any gap deterministically).

    Args:
      spec, base_seed : the difficulty spec + the run seed (drives every worker's sub-stream).
      budget          : (N_MAX, D_MAX, M_MAX, A_MAX) — passed in so the worker stays torch-free.
      workers         : K (>=1).            prefetch : per-worker queue depth (items each worker runs ahead).
      get_timeout     : seconds the main will WAIT on a LIVE-but-behind worker before borrowing the
                        item inline. Under healthy prefetch the queue is never empty so this never bites;
                        it only matters when the queue drains. Patient enough that a momentarily-behind
                        live worker is waited-for (not wastefully borrowed-then-discarded), which keeps
                        throughput up; DEATH is detected independently (and fast) by polling is_alive.
      poll            : the is_alive / queue poll granularity — a dead worker is caught within ~poll
                        seconds regardless of get_timeout, so the deadlock fix recovers in ~poll, not
                        in get_timeout (and certainly not the old never).
    """

    def __init__(self, spec: dict, base_seed: int, budget, workers: int = 4, prefetch: int = 32,
                 get_timeout: float = 2.0, poll: float = 0.25):
        assert workers >= 1
        self.K = int(workers)
        self.spec = spec
        self.base_seed = int(base_seed)
        self.budget = tuple(budget)
        self.get_timeout = float(get_timeout)
        self.poll = float(poll)
        ctx = mp.get_context("spawn")
        self.stop = ctx.Event()
        self.seeds = [_seed_for(base_seed, w) for w in range(self.K)]
        self.qs = [ctx.Queue(maxsize=prefetch) for _ in range(self.K)]
        self.procs = [ctx.Process(target=_worker,
                                  args=(self.seeds[w], spec, self.budget, self.qs[w], self.stop),
                                  daemon=True)
                      for w in range(self.K)]
        for p in self.procs:
            p.start()
        # per-worker bookkeeping for the deterministic round-robin + inline fallback
        self._consumed = [0] * self.K          # next item-index the main needs from worker w
        self._queue_pos = [0] * self.K         # index of the next item worker w's queue will yield
        self._dead = [False] * self.K          # process confirmed exited -> serve from fallback forever
        self._fb = [None] * self.K             # lazily-built inline fallback generator for worker w
        self._fb_pos = [0] * self.K            # index of the next item fallback w will yield
        self.fallback_items = 0                # diagnostics: items served inline (slow + dead)
        self.dead_workers = 0
        self._t = 0
        self._closed = False

    # ----- inline fallback (deterministic, bit-identical to the worker's stream) -----
    def _fallback_next(self, w: int):
        """Item `self._consumed[w]` of worker w, computed in-process from worker w's deterministic
        stream. The fallback generator is advanced to the needed index (a one-time catch-up the first
        time worker w falls back; O(1) thereafter), so the item is identical to the worker's."""
        if self._fb[w] is None:
            self._fb[w] = _budget_items(self.seeds[w], self.spec, self.budget)
        gen = self._fb[w]
        while self._fb_pos[w] < self._consumed[w]:      # catch up to the consumed frontier (once)
            next(gen)
            self._fb_pos[w] += 1
        item = next(gen)
        self._fb_pos[w] += 1
        self._consumed[w] += 1
        self.fallback_items += 1
        return item

    def _next_from_worker(self, w: int):
        need = self._consumed[w]
        if not self._dead[w]:
            # pull from the queue, discarding any STALE item (index < need) the worker enqueued for a
            # slot the main already served inline during a slow-worker episode. Poll in short slices so
            # a DEAD worker (is_alive False) is caught within ~poll, while a LIVE-but-behind worker is
            # waited-for up to get_timeout before we borrow the item inline.
            deadline = time.monotonic() + self.get_timeout
            while self._queue_pos[w] <= need:
                try:
                    item = self.qs[w].get(timeout=self.poll)
                except _queue.Empty:
                    if not self.procs[w].is_alive():     # truly dead -> latch inline fallback, fast
                        if not self._dead[w]:
                            self._dead[w] = True
                            self.dead_workers += 1
                            self._reap(w)
                        break
                    if time.monotonic() >= deadline:     # alive but too far behind -> borrow inline
                        break
                    continue                             # alive, still within patience -> keep waiting
                got = self._queue_pos[w]
                self._queue_pos[w] += 1
                if got == need:
                    self._consumed[w] += 1
                    return item
                # got < need: a stale duplicate of an already-(inline-)served slot -> drop it
        return self._fallback_next(w)

    def next(self):
        """The next budget-admitted (csp, full) item, in the deterministic round-robin order. Never
        blocks indefinitely: a stalled/dead worker is covered by the inline fallback."""
        w = self._t % self.K
        item = self._next_from_worker(w)
        self._t += 1
        return item

    def fill(self, n: int) -> list:
        return [self.next() for _ in range(n)]

    def _reap(self, w: int) -> None:
        """Terminate + drain a dead worker so its RAM is released and its queue can't wedge shutdown."""
        p = self.procs[w]
        try:
            if p.is_alive():
                p.terminate()
            p.join(timeout=1.0)
        except Exception:
            pass
        try:
            while True:
                self.qs[w].get_nowait()
        except (_queue.Empty, Exception):
            pass

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
            except (_queue.Empty, Exception):
                pass
        for p in self.procs:
            try:
                p.join(timeout=2.0)
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass
        for q in self.qs:
            try:
                q.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# ===================================================================== serial reference (determinism)
def serial_reference(spec: dict, base_seed: int, budget, workers: int):
    """The EXACT deterministic round-robin item sequence the K-worker pool produces, computed in ONE
    process with no workers (item t ← worker (t % K)'s deterministic budget-admitted stream). This is
    the 'serial' oracle the parallel 'fast' pool must match bit-for-bit, and it is also precisely the
    inline-fallback path — so a passing verify proves the fallback preserves the sequence."""
    K = int(workers)
    gens = [_budget_items(_seed_for(base_seed, w), spec, tuple(budget)) for w in range(K)]
    t = 0
    while True:
        yield next(gens[t % K])
        t += 1


def _sig(item) -> tuple:
    """A hashable, picklable signature of a (csp, full-domain) item for bit-identity comparison."""
    csp, dom = item
    return (csp.n, csp.d, csp.cons, dom)


# ===================================================================== verify (determinism + robustness)
def verify_determinism(n: int = 240, seed: int = 0, workers: int = 4) -> bool:
    """(a) The K-worker parallel pool reproduces the serial round-robin reference bit-for-bit."""
    from .. import run_glados_staged as G
    spec = S.organ_spec(difficulty=True)
    budget = (G.N_MAX, G.D_MAX, G.M_MAX, G.A_MAX)
    ref = [_sig(x) for x in _take(serial_reference(spec, seed, budget, workers), n)]
    with ParallelItemStream(spec, seed, budget, workers=workers) as ps:
        par = [_sig(ps.next()) for _ in range(n)]
        fb = ps.fallback_items
    ok = (ref == par)
    print(f"[A] determinism: parallel(K={workers}) == serial round-robin reference for {n} items: "
          f"{'PASS' if ok else 'FAIL'}  (inline-fallback fires={fb})", flush=True)
    if not ok:
        for i, (a, b) in enumerate(zip(ref, par)):
            if a != b:
                print(f"    first divergence at item {i}", flush=True)
                break
    assert ok
    return ok


def verify_killed_worker(n: int = 240, seed: int = 0, workers: int = 4, kill_at: int = 40) -> bool:
    """(b) Robustness: kill a worker mid-run; the main must NOT hang and the sequence must stay
    bit-identical to the serial reference (the inline fallback fires transparently)."""
    import os
    import signal
    import time
    from .. import run_glados_staged as G
    spec = S.organ_spec(difficulty=True)
    budget = (G.N_MAX, G.D_MAX, G.M_MAX, G.A_MAX)
    ref = [_sig(x) for x in _take(serial_reference(spec, seed, budget, workers), n)]
    t0 = time.time()
    with ParallelItemStream(spec, seed, budget, workers=workers) as ps:
        got = []
        victim = 1 % workers
        for i in range(n):
            if i == kill_at:                              # SIGKILL one worker mid-stream
                pid = ps.procs[victim].pid
                try:
                    os.kill(pid, signal.SIGKILL)
                    print(f"    [kill] SIGKILL worker {victim} (pid {pid}) at item {i}", flush=True)
                except ProcessLookupError:
                    pass
            got.append(_sig(ps.next()))
        wall = time.time() - t0
        fb, dead = ps.fallback_items, ps.dead_workers
    ok = (ref == got)
    print(f"[B] killed-worker robustness: sequence still == reference for {n} items: "
          f"{'PASS' if ok else 'FAIL'}  (wall {wall:.1f}s, no hang; inline-fallback={fb}, "
          f"dead_workers={dead})", flush=True)
    if not ok:
        for i, (a, b) in enumerate(zip(ref, got)):
            if a != b:
                print(f"    first divergence at item {i}", flush=True)
                break
    assert ok
    return ok


def verify_throughput(n: int = 360, seed: int = 0, workers: int = 4) -> dict:
    """(d) Quick throughput A/B: serial single-stream vs the K-worker parallel pool (records/sec)."""
    import time
    from .. import run_glados_staged as G
    spec = S.organ_spec(difficulty=True)
    budget = (G.N_MAX, G.D_MAX, G.M_MAX, G.A_MAX)

    t0 = time.time()
    for _ in _take(_budget_items(seed, spec, budget), n):
        pass
    serial_rps = n / max(1e-9, time.time() - t0)

    with ParallelItemStream(spec, seed, budget, workers=workers) as ps:
        ps.fill(workers * 4)                              # warm the spawn pool (one-time start cost)
        t0 = time.time()
        ps.fill(n)
        par_rps = n / max(1e-9, time.time() - t0)
    print(f"[D] throughput: serial {serial_rps:.0f} rec/s  ->  parallel(K={workers}) {par_rps:.0f} "
          f"rec/s  ({par_rps/max(1e-9,serial_rps):.2f}x)", flush=True)
    return {"serial_rps": serial_rps, "parallel_rps": par_rps, "speedup": par_rps / max(1e-9, serial_rps)}


def _take(it, k):
    out = []
    for x in it:
        out.append(x)
        if len(out) >= k:
            break
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="parallel deterministic deadlock-free item prefetch")
    ap.add_argument("--verify", action="store_true", help="determinism + killed-worker robustness")
    ap.add_argument("--bench", action="store_true", help="throughput A/B (serial vs parallel)")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.bench:
        verify_throughput(n=max(args.n, 360), seed=args.seed, workers=args.workers)
        return
    print("[stream_prefetch] VERIFY", flush=True)
    verify_determinism(n=args.n, seed=args.seed, workers=args.workers)
    verify_killed_worker(n=args.n, seed=args.seed, workers=args.workers)
    verify_throughput(n=max(args.n, 360), seed=args.seed, workers=args.workers)
    print("[stream_prefetch] ALL VERIFY PASS", flush=True)


if __name__ == "__main__":
    main()
