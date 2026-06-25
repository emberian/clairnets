"""Train HybridReasoner (host + differentiable sound deductor) vs BaselineReasoner (host + direct
head) on in-context graph-coloring-with-abstention. Headline reads:
  * abstain precision/recall on underdetermined queries (does it KNOW when it can't decide?),
  * SIZE GENERALIZATION: train on small N, test on larger N (the deductor is size-agnostic+sound;
    the pooled head must memorize). If the hybrid abstains correctly AND holds accuracy as N grows
    while the baseline confabulates and decays, the (d) bet is alive.

  python -m clair.run_induce --steps 6000 --train_n 4,6 --test_n 5,8,10,12 --aux 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import induce as I
from . import perf


def device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def build_pool(n_lo, n_hi, k, size, rng, det_frac=0.5):
    """Pre-generate a balanced problem pool ONCE (the CSP solver is the CPU cost; keep it out of
    the per-step hot loop). ~det_frac determined so 'always abstain' isn't a strong trivial baseline."""
    n_det = int(size * det_frac)
    det, ab = [], []
    while len(det) < n_det or len(ab) < size - n_det:
        p = I.gen_problem(int(rng.integers(n_lo, n_hi + 1)), k, rng)
        if p["capped"]:
            continue                                  # determinacy label unreliable -> drop
        if p["determined"] and len(det) < n_det:
            det.append(p)
        elif not p["determined"] and len(ab) < size - n_det:
            ab.append(p)
    pool = det + ab
    rng.shuffle(pool)
    return pool


def pool_batch(pool, bs, k, rng, dev, Nmax):
    idx = rng.integers(0, len(pool), bs)
    return I.make_batch([pool[i] for i in idx], Nmax, k, dev)


def losses(model, ba, aux):
    color_logits, abstain_logit, (cand, E) = model(ba)
    la = F.binary_cross_entropy_with_logits(abstain_logit, ba["abst"])
    det = ba["abst"] < 0.5
    lc = F.cross_entropy(color_logits[det], ba["ans"][det]) if det.any() else color_logits.sum() * 0
    l = la + lc
    if aux > 0 and cand is not None:
        # ground the compiler: predicted exclusion edges vs true DIFF adjacency; pins vs true IS
        vm = ba["vmask"]; pair = vm[:, :, None] * vm[:, None, :]
        le = F.binary_cross_entropy(E.clamp(1e-4, 1 - 1e-4), ba["adj"], reduction="none")
        le = (le * pair).sum() / pair.sum().clamp_min(1)
        lp = F.binary_cross_entropy_with_logits(cand, ba["pins"], reduction="none")
        lp = (lp * ba["pinned"].unsqueeze(-1)).sum() / ba["pinned"].sum().clamp_min(1)
        l = l + aux * (le + lp)
    return l


@torch.no_grad()
def evaluate(model, pool, k, rng, dev, Nmax, n=2048, bs=256):
    model.eval()
    tp = fp = fn = 0          # abstain confusion
    det_tot = det_right = 0
    correct = tot = 0
    seen = 0
    while seen < n:
        ba = pool_batch(pool, min(bs, n - seen), k, rng, dev, Nmax)
        cl, al, _ = model(ba)
        pred_abs = torch.sigmoid(al) > 0.5
        true_abs = ba["abst"] > 0.5
        pred_col = cl.argmax(-1)
        tp += int((pred_abs & true_abs).sum()); fp += int((pred_abs & ~true_abs).sum())
        fn += int((~pred_abs & true_abs).sum())
        det = ~true_abs
        det_tot += int(det.sum()); det_right += int(((pred_col == ba["ans"]) & ~pred_abs & det).sum())
        ok = (true_abs & pred_abs) | (~true_abs & ~pred_abs & (pred_col == ba["ans"]))
        correct += int(ok.sum()); tot += ba["abst"].numel(); seen += ba["abst"].numel()
    model.train()
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return {"overall": correct / tot, "det_acc": det_right / max(1, det_tot),
            "abstain_prec": prec, "abstain_rec": rec}


def train_one(which, args, dev, rng, Nmax, train_pool, eval_pools):
    cls = I.HybridReasoner if which == "hybrid" else I.BaselineReasoner
    m = cls(Nmax, args.k, d=args.d, layers=args.layers, T=args.T).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    print(f"[{which}] params={I.n_params(m):,}  Nmax={Nmax}")
    t0 = time.time(); log = []
    for s in range(1, args.steps + 1):
        ba = pool_batch(train_pool, args.bs, args.k, rng, dev, Nmax)
        loss = losses(m, ba, args.aux)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s % max(1, args.steps // 10) == 0:
            ind = evaluate(m, eval_pools["train"], args.k, rng, dev, Nmax)
            print(f"  [{which}] step {s:5d} loss {loss.item():.3f}  in-dist overall {ind['overall']*100:4.1f}%"
                  f"  abst P/R {ind['abstain_prec']*100:.0f}/{ind['abstain_rec']*100:.0f}  {time.time()-t0:.0f}s")
            log.append({"step": s, "loss": float(loss.detach()), **ind})
    gen = {str(tn): evaluate(m, eval_pools[tn], args.k, rng, dev, Nmax, n=4096)
           for tn in (int(x) for x in args.test_n.split(","))}
    return {"which": which, "params": I.n_params(m), "log": log, "size_gen": gen}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--aux", type=float, default=0.5, help="compiler-grounding aux loss weight (0=pure end-to-end)")
    ap.add_argument("--train_n", default="4,6")
    ap.add_argument("--test_n", default="5,8,10,12")
    ap.add_argument("--pool_size", type=int, default=20000, help="pre-generated train problem pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    perf.setup()
    dev = device()
    Nmax = max(int(x) for x in a.test_n.split(",")) + 1
    n_lo, n_hi = (int(x) for x in a.train_n.split(","))
    prng = np.random.default_rng(a.seed + 999)        # pool-building rng (pools shared by both models)
    t0 = time.time()
    train_pool = build_pool(n_lo, n_hi, a.k, a.pool_size, prng)
    eval_pools = {"train": build_pool(n_lo, n_hi, a.k, 4096, prng)}
    for tn in (int(x) for x in a.test_n.split(",")):
        eval_pools[tn] = build_pool(tn, tn, a.k, 4096, prng)
    print(f"pools built ({a.pool_size} train + eval) in {time.time()-t0:.0f}s")
    res = {"args": vars(a), "models": {}}
    for which in ["hybrid", "baseline"]:
        rng = np.random.default_rng(a.seed)           # same sampling stream for both
        torch.manual_seed(a.seed)
        res["models"][which] = train_one(which, a, dev, rng, Nmax, train_pool, eval_pools)
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "runs", "induce.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)           # SAVE FIRST — never lose results to a print bug
    print("wrote", out)
    print(f"\n=== SIZE GENERALIZATION (overall correct %, train N={a.train_n}) ===")
    ns = [int(x) for x in a.test_n.split(",")]
    print("  N       " + "  ".join(f"{n:>6d}" for n in ns))
    for which in ["hybrid", "baseline"]:
        g = res["models"][which]["size_gen"]
        print(f"  {which:8s}" + "  ".join(f"{g[str(n)]['overall']*100:6.1f}" for n in ns))
    print("  abstain-recall by N:")
    for which in ["hybrid", "baseline"]:
        g = res["models"][which]["size_gen"]
        print(f"  {which:8s}" + "  ".join(f"{g[str(n)]['abstain_rec']*100:6.1f}" for n in ns))


if __name__ == "__main__":
    main()
