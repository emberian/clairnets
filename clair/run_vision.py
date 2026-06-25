"""Train ONE (arm, target_params) ClairVision on CIFAR-100 (or CIFAR-10 with --cifar10) and report
top-1 accuracy, param count, and images/sec. The decisive comparison: at equal non-embed params,
does geom / wedge BEAT conv and mlp on CIFAR, reproducing CliffordNet's vision win?

  python -m clair.run_vision --arm geom  --target 1500000 --epochs 100
  python -m clair.run_vision --arm conv  --target 1500000 --epochs 100 --cifar10

Standard recipe: random-crop(32,pad4)+hflip+normalize, AdamW + cosine LR (linear warmup), label
smoothing, bf16 autocast (perf.amp — full bf16 is fine, this is a vision net, no soundness constraint).
Metrics/JSON-dump structure mirrors clair/run_glados.py & clair/run_geom_lm.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision import ClairVision, size_for
from . import perf


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# CIFAR channel mean/std (the standard normalization)
_STATS = {
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}


def loaders(name, root, bs, workers):
    """CIFAR train/test loaders with standard augmentation. Requires torchvision on the box."""
    import torchvision
    import torchvision.transforms as T
    mean, std = _STATS[name]
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    Ds = torchvision.datasets.CIFAR100 if name == "cifar100" else torchvision.datasets.CIFAR10
    tr = Ds(root=root, train=True, download=True, transform=train_tf)
    te = Ds(root=root, train=False, download=True, transform=test_tf)
    tl = torch.utils.data.DataLoader(tr, batch_size=bs, shuffle=True, num_workers=workers,
                                     pin_memory=True, drop_last=True, persistent_workers=workers > 0)
    el = torch.utils.data.DataLoader(te, batch_size=bs, shuffle=False, num_workers=workers,
                                     pin_memory=True, persistent_workers=workers > 0)
    n_classes = 100 if name == "cifar100" else 10
    return tl, el, n_classes


@torch.no_grad()
def evaluate(model, loader, dev, amp):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with perf.amp(dev, amp):
            logits = model(x)
        correct += (logits.float().argmax(-1) == y).sum().item()
        total += y.numel()
    model.train()
    return correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["geom", "wedge", "conv", "mlp"], required=True)
    ap.add_argument("--target", type=float, default=1.5e6, help="non-embed param budget (iso-param)")
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--patch", type=int, default=4, help="stem stride (32 -> 32/patch token grid)")
    ap.add_argument("--shifts", default="1,2,4,8,15", help="geom cyclic channel shifts S")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=5, help="warmup epochs")
    ap.add_argument("--smooth", type=float, default=0.1, help="label smoothing")
    ap.add_argument("--cifar10", action="store_true", help="fall back to CIFAR-10")
    ap.add_argument("--data_dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "cifar"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", default=True, help="bf16 autocast (default on)")
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    perf.setup()
    dev = device()
    torch.manual_seed(a.seed)

    name = "cifar10" if a.cifar10 else "cifar100"
    shifts = tuple(int(s) for s in a.shifts.split(","))
    tl, el, n_classes = loaders(name, a.data_dir, a.bs, a.workers)

    c, m, npar = size_for(a.arm, a.target, n_blocks=a.blocks, shifts=shifts)
    model = ClairVision(a.arm, c, m, n_blocks=a.blocks, n_classes=n_classes,
                        patch=a.patch).to(dev)
    real = model.n_params(non_embed=True)
    total = model.n_params(non_embed=False)
    print(f"dev={dev} dataset={name} arm={a.arm} c={c} m={m} blocks={a.blocks} "
          f"non_embed={real:,} total={total:,} (target {a.target:,.0f})")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)
    steps_per_epoch = len(tl)
    total_steps = a.epochs * steps_per_epoch
    warmup_steps = a.warmup * steps_per_epoch

    def lr_at(step):
        if step < warmup_steps:
            return a.lr * step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 * a.lr + 0.5 * (a.lr - 0.1 * a.lr) * (1 + math.cos(math.pi * p))

    log = []
    best_acc = 0.0
    gstep = 0
    seen = 0
    t0 = time.time()
    model.train()
    for epoch in range(a.epochs):
        ep_t0 = time.time()
        ep_seen = 0
        for x, y in tl:
            for g in opt.param_groups:
                g["lr"] = lr_at(gstep)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with perf.amp(dev, a.amp):
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y, label_smoothing=a.smooth)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            gstep += 1
            seen += x.size(0)
            ep_seen += x.size(0)
        acc = evaluate(model, el, dev, a.amp)
        best_acc = max(best_acc, acc)
        ep_dt = time.time() - ep_t0
        ips = ep_seen / max(1e-6, ep_dt)                     # images/sec (train throughput)
        print(f"epoch {epoch:3d}  loss {loss.item():.3f}  top1 {acc*100:5.2f}%  "
              f"best {best_acc*100:5.2f}%  {ips:.0f} img/s  {time.time()-t0:.0f}s")
        log.append({"epoch": epoch, "loss": loss.item(), "top1": acc, "best": best_acc,
                    "img_s": ips, "lr": lr_at(gstep)})

    res = {"arm": a.arm, "dataset": name, "target": a.target, "c": c, "m": m,
           "blocks": a.blocks, "patch": a.patch, "shifts": list(shifts),
           "non_embed_params": real, "total_params": total, "n_classes": n_classes,
           "epochs": a.epochs, "bs": a.bs, "lr": a.lr, "wd": a.wd, "seed": a.seed,
           "final_top1": log[-1]["top1"], "best_top1": best_acc,
           "final_img_s": log[-1]["img_s"], "log": log}
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "runs",
                                f"vision_{name}_{a.arm}_{int(a.target)}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(f"DONE {a.arm} @ {real:,} non-embed params: best top1 {best_acc*100:.2f}%  "
          f"({log[-1]['img_s']:.0f} img/s) -> {out}")


if __name__ == "__main__":
    main()
