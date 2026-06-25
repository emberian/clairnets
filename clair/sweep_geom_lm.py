"""Iso-param SCALING SWEEP: for each arm, several param budgets -> a (params -> bpb)
curve, so we can compare scaling slopes across arms. Does geom/wedge sit below swiglu,
and is its log-log slope steeper?

  python -m clair.sweep_geom_lm --steps 4000 --corpus bin
  python -m clair.sweep_geom_lm --arms geom wedge swiglu --budgets 1e6 2e6 4e6

Sequential (one GPU). Each (arm, budget) is one training run via clair.run_geom_lm's
machinery, imported directly (no subprocess) so JSON results accumulate in one file.
Writes runs/geomlm_sweep_<corpus>.json with the full grid + a per-arm log-log slope fit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .geom_lm import GeomLM, size_for
from .run_geom_lm import (LN2, device, load_bin, bin_batch, FineWeb)


def train_one(arm, target, a, corpus_state, dev):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    vocab, get_train, get_val, tpb, stream = corpus_state
    d, h, npar = size_for(arm, target, a.layers, a.heads, a.ratio)
    model = GeomLM(arm, vocab, d, h, a.layers, a.heads, a.ctx).to(dev)
    real = model.n_params()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)

    @torch.no_grad()
    def evaluate():
        model.eval(); ls = []
        for _ in range(a.eval_iters):
            _, l = model(*get_val()); ls.append(l.item())
        model.train(); return float(np.mean(ls))

    def lr_at(s):
        if s < a.warmup:
            return a.lr * s / a.warmup
        p = (s - a.warmup) / max(1, a.steps - a.warmup)
        return 0.1 * a.lr + 0.5 * (a.lr - 0.1 * a.lr) * (1 + math.cos(math.pi * p))

    print(f"  [{arm} @ {real:,}] d={d} h={h}")
    model.train()
    t0 = time.time(); seen = 0
    for s in range(a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        x, y = get_train()
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        seen += x.numel()
    vl = evaluate()
    r = tpb if tpb is not None else stream.tpb()
    bpb = vl / LN2 * r
    toks = seen / max(1e-6, time.time() - t0)
    print(f"  [{arm} @ {real:,}] bpb {bpb:.3f}  val {vl:.3f}  {toks/1e3:.1f}k tok/s")
    return {"arm": arm, "target": target, "params": real, "d_model": d, "h": h,
            "bpb": bpb, "val": vl, "tok_s": toks}


def loglog_slope(points):
    """Fit bpb = c * params^m in log-log; return slope m (bpb-vs-params scaling exponent).
    More-negative = better return on parameters. Pure numpy."""
    if len(points) < 2:
        return None
    x = np.log(np.array([p["params"] for p in points], dtype=float))
    y = np.log(np.array([p["bpb"] for p in points], dtype=float))
    m, b = np.polyfit(x, y, 1)
    return float(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["swiglu", "geom", "dot", "wedge", "geom_g3"])
    ap.add_argument("--budgets", nargs="+", type=float,
                    default=[5e5, 1e6, 2e6, 4e6, 8e6])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ratio", type=float, default=2.67)
    ap.add_argument("--corpus", choices=["bin", "fineweb"], default="bin")
    ap.add_argument("--data_dir", default=os.path.expanduser(
        "~/dev/graphplay/experiments/tinystories_arch/data"))
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = device()

    if a.corpus == "bin":
        tr, va = load_bin(a.data_dir)
        corpus_state = (256, lambda: bin_batch(tr, a.ctx, a.bs, dev),
                        lambda: bin_batch(va, a.ctx, a.bs, dev), 1.0, None)
    else:
        stream = FineWeb(a.tokenizer, a.ctx)
        corpus_state = (stream.vocab, lambda: stream.batch(a.bs, dev),
                        lambda: stream.batch(a.bs, dev), None, stream)

    print(f"dev={dev} corpus={a.corpus} arms={a.arms} budgets={a.budgets} steps={a.steps}")
    grid = []
    for arm in a.arms:
        for target in a.budgets:
            grid.append(train_one(arm, target, a, corpus_state, dev))

    slopes = {}
    for arm in a.arms:
        pts = sorted((g for g in grid if g["arm"] == arm), key=lambda p: p["params"])
        slopes[arm] = {"slope": loglog_slope(pts),
                       "curve": [(p["params"], p["bpb"]) for p in pts]}

    res = {"corpus": a.corpus, "steps": a.steps, "layers": a.layers, "heads": a.heads,
           "ctx": a.ctx, "lr": a.lr, "seed": a.seed, "grid": grid, "slopes": slopes}
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "runs",
                                f"geomlm_sweep_{a.corpus}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("\n=== scaling slopes (log-log bpb vs params; more negative = better) ===")
    for arm in a.arms:
        print(f"  {arm:8s} slope {slopes[arm]['slope']:+.4f}  "
              + "  ".join(f"{p/1e6:.1f}M:{b:.3f}" for p, b in slopes[arm]["curve"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
