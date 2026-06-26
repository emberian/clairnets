"""clair/datagen/fast.py — parallel + overlapped data path for organ training.

WHY. The organ training loop (clair.run_general.train, reused by sweep_organ / run_blade /
run_proposer) is ON-POLICY: each step's batch is the previous step's lattice state rolled forward
by the model's own monotone meet. Profiling that loop on the 16-vCPU T4 box:

    featurize 9.5% | dedP-targets 14% | fwd 26% | bwd 32% | roll+resample 18%   (one core, serial)

i.e. GPU compute (fwd+bwd) ~58% of wall and the CPU data path ~42%, run SERIALLY on a SINGLE core,
so the GPU idles ~40% of the time (~60% util) and 15 of 16 cores sit unused — exactly the
"wastes cores AND starves the GPU" symptom.

WHAT. Same math, two changes, both keeping the result reproducible and bitwise-identical:

  (1) PARALLEL dedP targets. exact-dedP per instance is embarrassingly parallel and a PURE function
      of (csp, dom). We compute it across all cores with a persistent ProcessPoolExecutor. dedP
      returns only the per-cell surviving-value sets (tiny), so IPC stays cheap and the speedup
      scales with instance COST — the future larger-budget organ (bigger d^n) benefits most.

  (2) OVERLAP with the GPU. The on-policy roll (meet -> next domains -> resample) and the next
      batch's featurize+targets depend only on the FORWARD output `b`, NOT on the backward pass.
      So after issuing the (async, non-blocking) CUDA backward+step, we prepare the next batch on
      the CPU while the GPU is still chewing the backward. The data path is hidden behind GPU
      compute -> GPU util goes to ~100% and the wasted cores do the prep.

DETERMINISM. The rng is touched only on the main thread, in the same order as the serial loop
(resampling stays serial); the pool computes a pure function; targets are assembled in batch order.
So fast.train(seed=s) reproduces run_general.train(seed=s) (identical under
torch.use_deterministic_algorithms(True); otherwise within CUDA scatter_add atomic noise, same as
two serial runs differ). See clair/datagen/verify_fast.py for the checks.

Opt-in: run_general.train(fast=True) or pass a FastGen via `gen=`. Enable everywhere with the
--fast flag (run_general / sweep_organ / run_blade).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from .. import run_general as RG
from ..proposer import FactorGraphProposer, size_for
from ._dedp_worker import canonical_key as _key    # disk-cache key (torch-free worker module)
from ._dedp_worker import dedP_one as _dedP_one     # torch-free worker entry (safe to spawn)

N_MAX, D_MAX, M_MAX, A_MAX = RG.N_MAX, RG.D_MAX, RG.M_MAX, RG.A_MAX
C = RG.C  # re-export for callers/tests


# ----------------------------------------------------------------------------- on-disk dedP cache
class DedPCache:
    """Content-addressed sqlite cache of exact-dedP targets, keyed by canonical_key(csp, dom). The
    dedP target is a PURE function of (constraints, domain), so it can be precomputed once and reused
    across epochs / sweep cells / reruns (e.g. a hyperparameter sweep re-deriving the same targets at
    seed s). Reads/writes happen on the main process only (no worker contention)."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS ded (k BLOB PRIMARY KEY, v BLOB)")
        self.db.commit()
        self.hits = self.misses = 0

    def get_many(self, keys):
        out = {}
        cur = self.db.cursor()
        for k in keys:
            row = cur.execute("SELECT v FROM ded WHERE k=?", (k,)).fetchone()
            if row is not None:
                out[k] = pickle.loads(row[0])
        return out

    def put_many(self, pairs):
        self.db.executemany("INSERT OR IGNORE INTO ded(k, v) VALUES (?, ?)",
                            [(k, pickle.dumps(v, protocol=5)) for k, v in pairs])
        self.db.commit()

    def close(self):
        self.db.close()


# ----------------------------------------------------------------------------- parallel dedP targets
class FastGen:
    """Persistent process pool that computes a batch of exact-dedP targets across all cores.

    Reused across steps/sweep-cells (workers + their dedP/solution caches stay warm). Falls back to
    serial for batches below `threshold` (where pool dispatch would cost more than the work). An
    optional on-disk DedPCache (cache_path=) memoises targets across runs.

    Uses the 'spawn' start method (not the default 'fork'): the training process initialises a CUDA
    context and forking after that deadlocks. Spawned workers are clean interpreters that only ever
    import clair.csp (via _dedp_worker) and never touch CUDA, so they're safe and contention-free.
    The pool is persistent (one warm-up paid once, reused across all steps / sweep cells)."""

    def __init__(self, workers=None, threshold=24, chunksize=4, cache_path=None):
        self.workers = workers or os.cpu_count() or 1
        self.threshold = threshold
        self.chunksize = chunksize
        self._pool = None
        self.cache = DedPCache(cache_path) if cache_path else None

    def _ensure(self):
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.workers,
                                             mp_context=mp.get_context("spawn"))
        return self._pool

    def _compute(self, items):
        """Parallel (or serial for small batches) dedP over a list of items, in order.

        When clair_fast is built we use its in-process rayon batch (`dedp_batch`): it releases the
        GIL and fans out across all cores WITHOUT a process pool, so we avoid pickling each CSP
        (frozensets of tuples) over the IPC boundary — which, with the now-tiny Rust compute, was the
        dominant cost. Bitwise-identical to the pure-Python dedP. Falls back to the ProcessPool when
        the extension is absent or an instance is out of the Rust bounds (d>64 / arity>8)."""
        if C.FAST and all(it[0].d <= 64 and all(len(sc) <= 8 for sc, _ in it[0].cons) for it in items):
            marsh = [(csp.n, csp.d, *C._marshal_cons(csp.cons), [sorted(x) for x in dom])
                     for csp, dom in items]
            return [tuple(frozenset(cell) for cell in r) for r in C._CF.dedp_batch(marsh)]
        if len(items) < self.threshold or self.workers <= 1:
            return [_dedP_one(it) for it in items]
        return list(self._ensure().map(_dedP_one, items, chunksize=self.chunksize))

    def dedP_batch(self, items):
        """List of per-cell dedP tuples, one per item, in order. Parallel over cores; disk-cached if
        a cache_path was given (only the misses are dispatched to the pool)."""
        if self.cache is None:
            return self._compute(items)
        keys = [_key(it) for it in items]
        have = self.cache.get_many(set(keys))
        miss_items, miss_pos = [], []
        for i, k in enumerate(keys):
            if k not in have:
                miss_items.append(items[i]); miss_pos.append(i)
        self.cache.hits += len(items) - len(miss_items); self.cache.misses += len(miss_items)
        out = [None] * len(items)
        for i, k in enumerate(keys):
            if k in have:
                out[i] = have[k]
        if miss_items:
            computed = self._compute(miss_items)
            self.cache.put_many([(keys[p], d) for p, d in zip(miss_pos, computed)])
            for p, d in zip(miss_pos, computed):
                out[p] = d
        return out

    def close(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        if self.cache is not None:
            self.cache.close()
            self.cache = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _false_elim_np(vm_before, vm_after, var_valid, tgt):
    """Host-side mirror of run_general.false_elim (so the bookkeeping doesn't sync the GPU and can
    overlap the backward)."""
    elig = tgt * vm_before * var_valid[..., None]
    killed = elig * (vm_after < 0.5)
    return float(killed.sum()), float(elig.sum())


# ----------------------------------------------------------------------------- overlapped training engine
def overlap_loop(m, opt, dev, steps, rng, items, rtags, sample_fn, theta=0.5,
                 gen=None, ex=RG.SHARED, log=None, label="fast"):
    """Model-AGNOSTIC overlapped/parallel training loop, shared by the proposer (run_general /
    sweep_organ) and the blade deductor (run_blade) — any model with the proposer forward signature
    (used via RG.fwd / RG.loss_fn / RG.meet). `sample_fn(rng, rung)` resamples a fresh CSP for a
    rung (kept on the MAIN thread so the rng stream matches the serial path -> reproducible).

    Per step: forward -> meet (sync forward) -> issue ASYNC backward+step on the GPU -> prepare the
    NEXT batch (parallel dedP over cores + featurize) on the CPU while the GPU runs the backward ->
    move it to device. The data path is hidden behind GPU compute."""
    log = log if log is not None else []
    caller_gen = gen is not None       # caller-provided pools are reused across train() calls
    gen = gen or FastGen()

    def prep(items):
        """Pure CPU prep: dedP targets (parallel pool) + featurize (numpy). Host arrays only; the
        H2D move is done later so the copies queue after the backward."""
        deds = gen.dedP_batch(items)
        for (csp, dom), ded in zip(items, deds):       # warm shared in-memory cache (eval reuses)
            ex.ded.setdefault(ex._k(csp, dom), ded)
        tgt_np, conflict_np = RG.targets_np_from_ded(deds)
        return RG.featurize_np(items), tgt_np, conflict_np

    feat_np, tgt_np, conflict_np = prep(items)
    feat = RG.to_device(feat_np, dev)
    vm = feat["var_mask"]
    tgt = torch.as_tensor(tgt_np, device=dev)
    conflict = torch.as_tensor(conflict_np, device=dev)

    fe_k = fe_n = 0
    t0 = time.time()
    for s in range(1, steps + 1):
        b, cls, sup = RG.fwd(m, feat, vm)
        loss = RG.loss_fn(sup, vm, feat["var_valid"], tgt, conflict)
        with torch.no_grad():
            new_vm = RG.meet(vm, b, theta)
        nvc = new_vm.to("cpu").numpy()        # syncs the FORWARD only (backward not issued yet)

        # --- issue backward + step ASYNC on the GPU (returns to Python immediately) ---
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

        # --- CPU prep of the next batch, CONCURRENT with the GPU backward above ---
        k, n = _false_elim_np(feat_np["var_mask"], nvc, feat_np["var_valid"], tgt_np)
        fe_k += k; fe_n += n
        nxt = []
        for bi, (csp, dom) in enumerate(items):
            ndom = RG.dom_from_mask(nvc[bi], csp)
            if C.status(ndom) in ("solved", "conflict") or ndom == dom:   # terminal/stalled -> fresh
                nc = sample_fn(rng, rtags[bi])
                nxt.append((nc, nc.full()))
            else:
                nxt.append((csp, ndom))
        items = nxt
        if s < steps:
            feat_np, tgt_np, conflict_np = prep(items)     # dedP pool overlaps the GPU backward
            feat = RG.to_device(feat_np, dev)              # H2D queues after the backward
            vm = feat["var_mask"]
            tgt = torch.as_tensor(tgt_np, device=dev)
            conflict = torch.as_tensor(conflict_np, device=dev)

        if s % max(1, steps // 12) == 0:
            fer = fe_k / max(1, fe_n)
            log.append({"step": s, "loss": float(loss.detach()), "false_elim": fer})
            print(f"    [{label}] step {s:5d}  loss {float(loss.detach()):.3f}  false_elim {fer:.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            fe_k = fe_n = 0

    if not caller_gen:
        gen.close()
    return log


def train(rungs, dev, target, steps, pool=96, R=8, lr=3e-4, theta=0.5, seed=0,
          ex=RG.SHARED, log=None, gen=None, workers=None):
    """Drop-in for run_general.train with the parallel + GPU-overlapped data path. Same signature,
    same returns (model, d_model, n_params, log), same rng stream -> reproducible."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d, npar = size_for("full", N_MAX, D_MAX, M_MAX, A_MAX, target, R=R, ds=min(4, R))
    m = FactorGraphProposer("full", N_MAX, D_MAX, M_MAX, A_MAX, d=d, R=R, ds=min(4, R)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95))
    caller_gen = gen is not None
    gen = gen or FastGen(workers=workers)
    tagged = RG.sample_corpus(rng, pool, rungs)
    items = [(c, c.full()) for _, c in tagged]
    rtags = [rg for rg, _ in tagged]
    print(f"  [fast] train rungs={rungs} d={d} params={npar:,} pool={pool} R={R} steps={steps} "
          f"workers={gen.workers}", flush=True)
    log = overlap_loop(m, opt, dev, steps, rng, items, rtags, RG.sample_rung_csp,
                       theta=theta, gen=gen, ex=ex, log=log, label="fast")  # gen!=None -> not closed here
    if not caller_gen:
        gen.close()
    return m, d, npar, log
