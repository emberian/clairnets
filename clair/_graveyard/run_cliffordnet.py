"""Train a FAITHFUL CliffordNet (clair/cliffordnet.py) on CIFAR-100 with the paper's recipe and
report top-1 + params + img/s. This is the CORRECTED experiment: the question is no longer the
(wrong) iso-param "does geom beat conv" of run_vision.py, but the PARETO replication:

  Does CliffordNet-Nano (~1.4M, lambda=1) REACH ~76-78% on CIFAR-100 with proper training?
  And does lambda=1 (Differential/Laplacian context) matter vs lambda=0 (the ablation isolating
  the self-energy-suppression mechanism our prior version missed)?

  python -m clair.run_cliffordnet --variant nano                 # the replication target
  python -m clair.run_cliffordnet --variant nano --lam 0         # the decisive ablation arm
  python -m clair.run_cliffordnet --variant nano --mode wedge    # wedge-only ablation (Table 4)
  python -m clair.run_cliffordnet --variant lite                 # the 2.6M SOTA-for-tiny arm

Paper recipe (Sec. 4): 200 epochs, AdamW + cosine annealing, label smoothing, AutoAugment +
RandomCrop(32,pad4) + HFlip + Random Erasing, DropPath (stochastic depth) for CliffordNet, bf16.
JSON-dump structure mirrors clair/run_vision.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from .cliffordnet import build_variant
from . import perf


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_STATS = {
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}


def loaders(name, root, bs, workers, autoaug=True, erase=0.25):
    """CIFAR train/test loaders with the paper's strong augmentation (AutoAugment + Random Erasing
    on top of the CIFAR-standard crop/flip/normalize). Requires torchvision on the box."""
    import torchvision
    import torchvision.transforms as T
    mean, std = _STATS[name]
    aug = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    if autoaug:
        aug.append(T.AutoAugment(T.AutoAugmentPolicy.CIFAR10))
    aug += [T.ToTensor(), T.Normalize(mean, std)]
    if erase > 0:
        aug.append(T.RandomErasing(p=erase))
    train_tf = T.Compose(aug)
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
    ap.add_argument("--variant", default="nano", choices=["nano", "lite", "cn32", "cn64"])
    # ablation overrides (None = use the variant default)
    ap.add_argument("--lam", type=float, default=None, help="self-energy suppression lambda (0 or 1)")
    ap.add_argument("--mode", default=None, choices=["full", "wedge", "inner", "dot"],
                    help="geometric interaction: full | wedge-only | inner(=dot)-only")
    ap.add_argument("--use_global", type=int, default=None, choices=[0, 1],
                    help="override global superposition (gFFN-G) on/off")
    ap.add_argument("--patch", type=int, default=2, help="patch-embed stride (paper uses 2)")
    ap.add_argument("--drop_path", type=float, default=0.1, help="stochastic depth (paper: DropPath)")
    # recipe
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=10, help="warmup epochs")
    ap.add_argument("--smooth", type=float, default=0.1, help="label smoothing")
    ap.add_argument("--no_autoaug", dest="autoaug", action="store_false", default=True)
    ap.add_argument("--erase", type=float, default=0.25, help="Random Erasing prob")
    ap.add_argument("--cifar10", action="store_true")
    ap.add_argument("--data_dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "cifar"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true", default=True, help="bf16 autocast (default on)")
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    perf.setup()
    dev = device()
    torch.manual_seed(a.seed)

    name = "cifar10" if a.cifar10 else "cifar100"
    n_classes = 100 if name == "cifar100" else 10
    tl, el, _ = loaders(name, a.data_dir, a.bs, a.workers, autoaug=a.autoaug, erase=a.erase)

    use_global = None if a.use_global is None else bool(a.use_global)
    model = build_variant(a.variant, patch=a.patch, n_classes=n_classes, lam=a.lam,
                          use_global=use_global, mode=a.mode, drop_path=a.drop_path).to(dev)
    cfg = model.cfg
    if a.compile:
        model = torch.compile(model)
    real = (model._orig_mod if hasattr(model, "_orig_mod") else model).n_params(non_embed=True)
    total = (model._orig_mod if hasattr(model, "_orig_mod") else model).n_params(non_embed=False)
    print(f"dev={dev} dataset={name} variant={a.variant} cfg={cfg} "
          f"non_embed={real:,} total={total:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)
    steps_per_epoch = len(tl)
    total_steps = a.epochs * steps_per_epoch
    warmup_steps = a.warmup * steps_per_epoch

    def lr_at(step):
        if step < warmup_steps:
            return a.lr * step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * a.lr * (1 + math.cos(math.pi * p))            # cosine to 0

    log = []
    best_acc = 0.0
    gstep = 0
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
            ep_seen += x.size(0)
        acc = evaluate(model, el, dev, a.amp)
        best_acc = max(best_acc, acc)
        ips = ep_seen / max(1e-6, time.time() - ep_t0)
        print(f"epoch {epoch:3d}  loss {loss.item():.3f}  top1 {acc*100:5.2f}%  "
              f"best {best_acc*100:5.2f}%  {ips:.0f} img/s  {time.time()-t0:.0f}s")
        log.append({"epoch": epoch, "loss": loss.item(), "top1": acc, "best": best_acc,
                    "img_s": ips, "lr": lr_at(gstep)})

    res = {"variant": a.variant, "cfg": cfg, "dataset": name, "lam_override": a.lam,
           "mode_override": a.mode, "use_global_override": a.use_global, "patch": a.patch,
           "drop_path": a.drop_path, "non_embed_params": real, "total_params": total,
           "epochs": a.epochs, "bs": a.bs, "lr": a.lr, "wd": a.wd, "smooth": a.smooth,
           "autoaug": a.autoaug, "erase": a.erase, "seed": a.seed,
           "final_top1": log[-1]["top1"], "best_top1": best_acc,
           "final_img_s": log[-1]["img_s"], "log": log}
    tag = f"{a.variant}_lam{cfg['lam']}_{cfg['mode']}_glo{int(cfg['use_global'])}"
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "runs", f"cliffordnet_{name}_{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(f"DONE {tag} @ {total:,} total params: best top1 {best_acc*100:.2f}%  "
          f"({log[-1]['img_s']:.0f} img/s) -> {out}")


if __name__ == "__main__":
    main()
