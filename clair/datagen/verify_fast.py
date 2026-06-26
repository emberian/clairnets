"""clair/datagen/verify_fast.py — correctness + speed checks for the parallel/overlapped data path.

    python -m clair.datagen.verify_fast            # correctness (bitwise) + a short cuda benchmark
    python -m clair.datagen.verify_fast --bench    # longer records/sec + GPU-util A/B only

Three checks:
  1. DATA bitwise-identical: parallel (pool) dedP targets + featurize_np == serial build_targets /
     featurize, exactly, for a fixed batch of items.
  2. TRAINING reproducible: on CPU (deterministic), fast.train(seed) == run_general.train(seed) for
     the FULL loss/false-elim log, bit-for-bit -> the loop reorder + parallelism change nothing.
  3. SPEED + GPU: on cuda, records/sec serial vs fast and (optionally) sampled GPU utilisation.
"""
from __future__ import annotations

import argparse
import subprocess
import threading
import time

import numpy as np
import torch

from .. import run_general as RG
from . import fast as F


def check_data_identical(n=128, seed=3):
    rng = np.random.default_rng(seed)
    tagged = RG.sample_corpus(rng, n, RG.RUNGS)
    items = [(c, c.full()) for _, c in tagged]
    # serial
    tgt_s, conf_s = RG.build_targets(items, RG.Exact(), "cpu")
    feat_s = RG.featurize(items, "cpu")
    # parallel
    with F.FastGen() as gen:
        deds = gen.dedP_batch(items)
    tgt_p_np, conf_p_np = RG.targets_np_from_ded(deds)
    feat_p_np = RG.featurize_np(items)
    ok = True
    ok &= np.array_equal(tgt_s.numpy(), tgt_p_np)
    ok &= np.array_equal(conf_s.numpy(), conf_p_np)
    for k, v in feat_s.items():
        ok &= np.array_equal(v.numpy(), feat_p_np[k])
    print(f"[1] data bitwise-identical (n={n}): {'PASS' if ok else 'FAIL'}")
    assert ok
    return ok


def check_training_reproducible(steps=120, seed=0):
    """On CPU (deterministic) the fast path must reproduce the serial loop bit-for-bit."""
    rungs = ["coloring", "arithmetic", "smt"]
    log_serial = RG.train(rungs, "cpu", 1.2e5, steps, pool=48, R=4, seed=seed, ex=RG.Exact())[3]
    log_fast = F.train(rungs, "cpu", 1.2e5, steps, pool=48, R=4, seed=seed, ex=RG.Exact())[3]
    same = (len(log_serial) == len(log_fast)
            and all(a["step"] == b["step"]
                    and a["loss"] == b["loss"]
                    and a["false_elim"] == b["false_elim"]
                    for a, b in zip(log_serial, log_fast)))
    print(f"[2] training reproducible serial==fast (cpu, {steps} steps): {'PASS' if same else 'FAIL'}")
    if not same:
        for a, b in zip(log_serial, log_fast):
            if a != b:
                print("    first divergence:", a, "vs", b); break
    assert same
    return same


def check_cache(n=96, seed=7):
    """The disk cache must return targets identical to the uncached compute, and hit on rerun."""
    import os
    import tempfile
    rng = np.random.default_rng(seed)
    items = [(c, c.full()) for _, c in RG.sample_corpus(rng, n, RG.RUNGS)]
    with F.FastGen() as g0:
        ref = g0.dedP_batch(items)
    path = os.path.join(tempfile.mkdtemp(), "ded.sqlite")
    with F.FastGen(cache_path=path) as g1:          # cold: all misses
        a = g1.dedP_batch(items); cold_miss = g1.cache.misses
    with F.FastGen(cache_path=path) as g2:          # warm: all hits, no compute
        b = g2.dedP_batch(items); warm_hit, warm_miss = g2.cache.hits, g2.cache.misses
    ok = (a == ref and b == ref and cold_miss == n and warm_hit == n and warm_miss == 0)
    print(f"[1c] disk cache identical + reused (cold miss {cold_miss}/{n}, warm hit {warm_hit}/{n}): "
          f"{'PASS' if ok else 'FAIL'}")
    assert ok
    return ok


class _GpuSampler(threading.Thread):
    """Background nvidia-smi sampler -> mean GPU utilisation over the run."""
    def __init__(self, period=0.1):
        super().__init__(daemon=True); self.period = period; self.stop = False; self.samples = []

    def run(self):
        while not self.stop:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    timeout=2).decode().strip().splitlines()
                self.samples.append(float(out[0]))
            except Exception:
                pass
            time.sleep(self.period)

    def mean(self):
        return float(np.mean(self.samples)) if self.samples else float("nan")


def bench(steps=300, pool=96, target=2.0e5, R=8, seed=0):
    dev = RG.device()
    if dev != "cuda":
        print(f"[3] bench: device={dev} (no GPU; records/sec only)")
    rungs = RG.RUNGS
    if dev == "cuda":
        torch.set_float32_matmul_precision("high")
    gen = F.FastGen()        # persistent spawn pool
    _warm = RG.sample_corpus(np.random.default_rng(1), pool, rungs)   # warm workers (one-time spawn cost)
    gen.dedP_batch([(c, c.full()) for _, c in _warm])

    def run(fn_kw, label):
        s = _GpuSampler() if dev == "cuda" else None
        if s: s.start()
        t0 = time.time()
        RG.train(rungs, dev, target, steps, pool=pool, R=R, seed=seed, **fn_kw)
        if dev == "cuda": torch.cuda.synchronize()
        wall = time.time() - t0
        if s: s.stop = True; s.join()
        rps = steps * pool / wall
        gu = s.mean() if s else float("nan")
        print(f"[3] {label:8s}  wall {wall:6.2f}s  records/sec {rps:7.0f}  GPU-util {gu:5.1f}%")
        return wall, rps, gu

    ws, rs, gs = run({}, "serial")
    wf, rf, gf = run({"gen": gen}, "fast")
    gen.close()
    print(f"[3] SPEEDUP {ws/wf:.2f}x   records/sec {rs:.0f} -> {rf:.0f}   "
          f"GPU-util {gs:.0f}% -> {gf:.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--pool", type=int, default=96)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--target", type=float, default=2.0e5)
    args = ap.parse_args()
    if not args.bench:
        check_data_identical()
        check_cache()
        check_training_reproducible()
    bench(steps=args.steps, pool=args.pool, R=args.R, target=args.target)


if __name__ == "__main__":
    main()
