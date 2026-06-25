"""Clifford-graft: can a geometric product carry a real pretrained transformer's FFN?

Take an open-checkpoint model (OLMo-2-1B), replace its SwiGLU MLPs with geometric-product
mixers (GeomFFN), freeze EVERYTHING else, and distill ONLY the new mixers to match the
original model's next-token distribution on a real corpus. Then read off:

  * recovered fraction = how much of the teacher's loss-gap the grafted student closes,
    vs the random-init-graft starting point (1.0 = fully recovers the FFN's function).
  * param delta = FFN params removed vs GeomFFN params added (the efficiency claim).

GeomFFN(d,h): u,w: d->h ; geometric product uv = u·v + u∧v in h ; o: h->d.  ~3·d·h params.
  h = round(2.67 d)  -> iso-param with SwiGLU (~8 d^2).   h = d  -> ~3 d^2 (sub-param test).

Usage (on the GPU box):
  python -m clair.graft --model allenai/OLMo-2-0425-1B --hmult 1.0 --layers all --steps 2000
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class GeomFFN(nn.Module):
    """Geometric-product channel mixer as a drop-in for a SwiGLU MLP."""
    def __init__(self, d, h):
        super().__init__()
        if h % 2:
            h += 1
        self.u = nn.Linear(d, h, bias=False)
        self.w = nn.Linear(d, h, bias=False)
        self.o = nn.Linear(h, d, bias=False)

    def forward(self, x):
        u = self.u(x).unflatten(-1, (-1, 2))
        w = self.w(x).unflatten(-1, (-1, 2))
        inner = (u * w).sum(-1)
        wedge = u[..., 0] * w[..., 1] - u[..., 1] * w[..., 0]
        return self.o(torch.cat([inner, wedge], -1))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def graft_mlps(model, hmult, which, dtype):
    layers = model.model.layers
    d = model.config.hidden_size
    idxs = range(len(layers)) if which == "all" else [int(i) for i in which.split(",")]
    removed = added = 0
    new_mods = []
    for i in idxs:
        old = layers[i].mlp
        removed += n_params(old)
        h = int(round(hmult * d))
        g = GeomFFN(d, h).to(device=next(model.parameters()).device, dtype=dtype)
        added += n_params(g)
        layers[i].mlp = g
        new_mods.append(g)
    return new_mods, removed, added, list(idxs)


@torch.no_grad()
def eval_ppl(model, batches):
    model.eval()
    tot = ntok = 0.0
    for x in batches:
        out = model(x, labels=x)
        tot += out.loss.item() * (x.numel())
        ntok += x.numel()
    model.train()
    return math.exp(tot / ntok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--hmult", type=float, default=1.0, help="GeomFFN width / hidden (1.0=sub-param, 2.67=iso)")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--kl", type=float, default=1.0, help="weight on KL-to-teacher (0=plain LM)")
    ap.add_argument("--corpus", default="HuggingFaceFW/fineweb-edu")
    a = ap.parse_args()
    dev = "cuda"
    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(a.model)

    print(f"loading teacher + student: {a.model}")
    teacher = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(dev)

    base = n_params(student)
    new_mods, removed, added, idxs = graft_mlps(student, a.hmult, a.layers, dtype)
    for p in student.parameters():
        p.requires_grad_(False)
    train_ps = []
    for g in new_mods:
        for p in g.parameters():
            p.requires_grad_(True); train_ps.append(p)
    print(f"grafted {len(idxs)} layers | FFN params removed {removed:,} -> GeomFFN added {added:,} "
          f"({added/removed:.2f}x) | total {base:,} -> {n_params(student):,} | trainable {sum(p.numel() for p in train_ps):,}")

    # corpus stream
    ds = load_dataset(a.corpus, name="sample-10BT", split="train", streaming=True)
    it = iter(ds)

    def get_batch(n):
        xs = []
        while len(xs) < n:
            t = tok(next(it)["text"], return_tensors="pt", truncation=True, max_length=a.ctx)["input_ids"][0]
            if t.numel() >= a.ctx:
                xs.append(t[: a.ctx])
        return torch.stack(xs).to(dev)

    val = [get_batch(a.bs) for _ in range(16)]
    ppl_teacher = eval_ppl(teacher, val)
    ppl_init = eval_ppl(student, val)         # random-graft starting point
    print(f"teacher ppl {ppl_teacher:.2f} | grafted(random-init) ppl {ppl_init:.2f}")

    opt = torch.optim.AdamW(train_ps, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    t0 = time.time()
    for s in range(1, a.steps + 1):
        x = get_batch(a.bs)
        with torch.no_grad():
            tlog = teacher(x).logits
        slog = student(x).logits
        ce = F.cross_entropy(slog[:, :-1].reshape(-1, slog.size(-1)), x[:, 1:].reshape(-1))
        kl = F.kl_div(F.log_softmax(slog[:, :-1], -1), F.log_softmax(tlog[:, :-1], -1),
                      log_target=True, reduction="batchmean")
        loss = ce + a.kl * kl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(train_ps, 1.0); opt.step()
        if s % max(1, a.steps // 12) == 0:
            pv = eval_ppl(student, val)
            rec = (ppl_init - pv) / max(1e-6, ppl_init - ppl_teacher)
            print(f"  step {s:5d}  loss {loss.item():.3f}  ppl {pv:.2f}  recovered {rec*100:5.1f}%  {time.time()-t0:.0f}s")
    pv = eval_ppl(student, val)
    rec = (ppl_init - pv) / max(1e-6, ppl_init - ppl_teacher)
    print(f"\nDONE: teacher {ppl_teacher:.2f} | student {pv:.2f} | recovered {rec*100:.1f}% of FFN function "
          f"| param ratio GeomFFN/FFN = {added/removed:.2f}x")


if __name__ == "__main__":
    main()
