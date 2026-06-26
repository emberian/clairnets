"""Train one (arch, task) at a fixed non-embed param budget; report exact-accuracy.

  python -m clair.train --arch geom --task modadd --target 1.0e5 --steps 8000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .model import ClairNet, size_for
from .tasks import GAUNTLET, IGNORE


def device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def exact_acc(model, task, dev, rng, n=2048, bs=512):
    model.eval()
    seqs = tok = 0
    correct_seq = correct_tok = 0.0
    while seqs < n:
        x, y = task.sample(min(bs, n - seqs), dev, rng)
        logits, _ = model(x)
        pred = logits.argmax(-1)
        m = y != IGNORE
        per_tok = (pred == y) & m
        correct_tok += per_tok.sum().item(); tok += m.sum().item()
        # per-example: all scored positions correct
        ok = ((pred == y) | ~m).all(dim=1)
        correct_seq += ok.sum().item(); seqs += x.size(0)
    model.train()
    return correct_seq / seqs, correct_tok / max(1, tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["std", "attn", "geom", "geom_recur"], required=True)
    ap.add_argument("--task", choices=list(GAUNTLET), required=True)
    ap.add_argument("--target", type=float, default=1.0e5)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = device()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    task = GAUNTLET[a.task]()
    d, npar = size_for(a.arch, a.target, task.vocab, a.layers, a.heads, task.ctx, task.causal, a.depth)
    m = ClairNet(a.arch, task.vocab, d, a.layers, a.heads, task.ctx, task.causal, a.depth).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    print(f"dev={dev} arch={a.arch} task={task.name} d_model={d} eff_depth={m.unrolls} "
          f"params={npar:,} (target {a.target:,.0f}) causal={task.causal}")

    def lr_at(s):
        w = a.steps // 20
        if s < w:
            return a.lr * (s + 1) / w
        p = (s - w) / max(1, a.steps - w)
        return a.lr * (0.05 + 0.475 * (1 + math.cos(math.pi * p)))

    log = []; t0 = time.time(); ds = a.arch == "geom_recur"
    for s in range(a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        x, y = task.sample(a.bs, dev, rng)
        _, loss = m(x, y, ignore=IGNORE, deep_supervision=ds)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s % max(1, a.steps // 12) == 0:
            sa, ta = exact_acc(m, task, dev, rng)
            log.append({"step": s, "loss": loss.item(), "seq_acc": sa, "tok_acc": ta})
            print(f"  step {s:6d}  loss {loss.item():.3f}  seq_acc {sa*100:5.1f}%  tok_acc {ta*100:5.1f}%  {time.time()-t0:.0f}s")
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "runs", f"{a.task}_{a.arch}_{int(a.target)}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"arch": a.arch, "task": task.name, "d_model": d, "eff_depth": m.unrolls,
               "target": a.target, "params": npar, "steps": a.steps,
               "final_seq_acc": log[-1]["seq_acc"], "final_tok_acc": log[-1]["tok_acc"],
               "log": log}, open(out, "w"), indent=1)
    print(f"DONE {a.arch}/{task.name}: seq_acc {log[-1]['seq_acc']*100:.1f}%  ({npar:,} params)")


if __name__ == "__main__":
    main()
