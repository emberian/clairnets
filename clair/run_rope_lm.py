"""Train ONE positional-encoding arm of RopeLM on a real corpus and report bits-per-byte
on a held-out split, plus tokens/sec and param count. The decisive comparison: with the
SwiGLU channel-mixer FIXED, does a RICHER geometric position rotor (rope_learned / rope_nd)
beat plain RoPE at equal compute? If yes there's headroom in geometric position encoding;
if no, RoPE is already optimal — both are clean findings.

  python -m clair.run_rope_lm --arm rope         --target 4000000 --steps 3000
  python -m clair.run_rope_lm --arm rope_learned --target 4000000 --steps 3000 --corpus fineweb

Corpora (identical handling to run_geom_lm.py):
  bin      : byte-level uint8 from --data_dir/{train,val}.bin. vocab=256, bpb = val_CE/ln2.
  fineweb  : stream HuggingFaceFW/fineweb-edu via a HF tokenizer; bpb = val_CE/ln2 * tpb.

Reused VERBATIM from run_geom_lm.py: device(), the byte loader (load_bin/bin_batch), the
FineWeb streamer, the cosine-LR + bpb eval loop, the JSON-dump structure. Only the model
class (RopeLM) + its sizing (rope_lm.size_for, which sizes the shared swiglu backbone) and
the arm/param-delta reporting differ.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import perf
from .rope_lm import ARMS, RopeLM, param_delta, size_for

LN2 = math.log(2)


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----- byte corpus (run_geom_lm.py) -----------------------------------------------------

def load_bin(data_dir):
    tr = np.fromfile(os.path.join(data_dir, "train.bin"), dtype=np.uint8)
    va = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint8)
    return tr, va


def bin_batch(d, ctx, bs, dev):
    ix = np.random.randint(0, len(d) - ctx - 1, size=bs)
    x = np.stack([d[i:i + ctx] for i in ix]).astype(np.int64)
    y = np.stack([d[i + 1:i + 1 + ctx] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)


# ----- fineweb token corpus (run_geom_lm.py) --------------------------------------------

class FineWeb:
    """Stream + tokenize fineweb-edu into fixed (ctx+1) windows; track token/byte ratio so
    token-CE can be rescaled to comparable bits-per-byte."""
    def __init__(self, tokenizer, ctx):
        from datasets import load_dataset
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tokenizer)
        self.vocab = self.tok.vocab_size
        self.ctx = ctx
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        self.it = iter(ds)
        self.ntok = self.nbyte = 0

    def _window(self):
        buf = []
        while len(buf) < self.ctx + 1:
            txt = next(self.it)["text"]
            ids = self.tok(txt)["input_ids"]
            self.ntok += len(ids); self.nbyte += len(txt.encode("utf-8"))
            buf.extend(ids)
        return buf[: self.ctx + 1]

    def batch(self, bs, dev):
        rows = [self._window() for _ in range(bs)]
        a = torch.tensor(rows, dtype=torch.long, device=dev)
        return a[:, :-1], a[:, 1:]

    def tpb(self):
        return self.ntok / max(1, self.nbyte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--target", type=float, default=4e6, help="non-embed param budget (backbone)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ratio", type=float, default=2.67)
    ap.add_argument("--base", type=float, default=10000.0, help="RoPE base for the freq schedule")
    ap.add_argument("--corpus", choices=["bin", "fineweb"], default="bin")
    ap.add_argument("--data_dir", default=os.path.expanduser(
        "~/dev/graphplay/experiments/tinystories_arch/data"))
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    perf.setup()
    dev = device()

    if a.corpus == "bin":
        tr, va = load_bin(a.data_dir)
        vocab = 256
        get_train = lambda: bin_batch(tr, a.ctx, a.bs, dev)
        get_val = lambda: bin_batch(va, a.ctx, a.bs, dev)
        tpb = 1.0
        stream = None
    else:
        stream = FineWeb(a.tokenizer, a.ctx)
        vocab = stream.vocab
        get_train = lambda: stream.batch(a.bs, dev)
        get_val = lambda: stream.batch(a.bs, dev)
        tpb = None

    d, h, npar = size_for(a.target, a.layers, a.heads, a.ratio)     # sizes the shared backbone
    model = RopeLM(a.arm, vocab, d, h, a.layers, a.heads, a.ctx, a.base).to(dev)
    real = model.n_params()
    deltas = param_delta(d, h, a.layers, a.heads)                   # honest per-arm param deltas
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)
    print(f"dev={dev} arm={a.arm} d={d} h={h} L={a.layers} H={a.heads} dh={d//a.heads} "
          f"vocab={vocab} params(non-embed)={real:,} (backbone target {a.target:,.0f}) "
          f"arm_delta=+{deltas[a.arm][1]} corpus={a.corpus} amp={a.amp}")
    print("  param deltas vs rope baseline: " +
          "  ".join(f"{arm}=+{deltas[arm][1]}" for arm in ARMS))

    amp = lambda: perf.amp(dev, a.amp)

    @torch.no_grad()
    def evaluate():
        model.eval(); ls = []
        for _ in range(a.eval_iters):
            with amp():
                _, l = model(*get_val())
            ls.append(l.item())
        model.train(); return float(np.mean(ls))

    def lr_at(step):
        if step < a.warmup:
            return a.lr * step / a.warmup
        p = (step - a.warmup) / max(1, a.steps - a.warmup)
        return 0.1 * a.lr + 0.5 * (a.lr - 0.1 * a.lr) * (1 + math.cos(math.pi * p))

    log = []
    t0 = time.time()
    seen = 0
    model.train()
    for step in range(a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_train()
        with amp():
            _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        seen += x.numel()
        if step % a.eval_every == 0:
            vl = evaluate()
            r = tpb if tpb is not None else stream.tpb()
            bpb = vl / LN2 * r
            dt = time.time() - t0
            toks = seen / max(1e-6, dt)
            print(f"step {step:5d}  train {loss.item():.3f}  val {vl:.3f}  bpb {bpb:.3f}  "
                  f"{toks/1e3:.1f}k tok/s  {dt:.0f}s")
            log.append({"step": step, "train": loss.item(), "val": vl, "bpb": bpb,
                        "tok_s": toks, "tpb": r})

    # final sample (byte corpus only)
    sample = None
    if a.corpus == "bin":
        model.eval()
        prompt = torch.tensor([[ord(c) for c in "Once upon a time"]], dtype=torch.long, device=dev)
        out = model.generate(prompt, 200, temp=0.8)[0].tolist()
        sample = bytes(b & 0xFF for b in out).decode("utf-8", "replace")
        print("\n--- sample ---\n" + sample + "\n")

    res = {"arm": a.arm, "target": a.target, "d_model": d, "h": h, "layers": a.layers,
           "heads": a.heads, "dh": d // a.heads, "vocab": vocab, "params": real,
           "arm_delta": deltas[a.arm][1], "param_deltas": {k: v[1] for k, v in deltas.items()},
           "base": a.base, "corpus": a.corpus, "ctx": a.ctx, "steps": a.steps, "lr": a.lr,
           "seed": a.seed, "amp": a.amp,
           "final_bpb": log[-1]["bpb"], "final_val": log[-1]["val"],
           "final_tok_s": log[-1]["tok_s"], "log": log, "sample": sample}
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "runs",
                                f"ropelm_{a.arm}_{int(a.target)}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(f"DONE {a.arm} @ {real:,} params: final bpb {log[-1]['bpb']:.3f}  "
          f"({log[-1]['tok_s']/1e3:.1f}k tok/s) -> {out}")


if __name__ == "__main__":
    main()
