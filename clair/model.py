"""ClairNet — one configurable transformer, four mixer arms, iso-param-comparable.

Arms (the per-block channel mixer; attention is shared):
  std        : MHA + SwiGLU FFN              baseline
  attn       : MHA only                      attention-only / FFN-light floor
  geom       : MHA + GeomMix (FFN-free)      geometric product uv = u·v + u∧v  (CliffordNet)
  geom_recur : ONE geom block, weight-tied,  recurrent depth + deep supervision (HRM core,
               looped `depth*layers` times    NO puzzle-id / hierarchy hype)

`causal=True` for algorithmic next-token; `causal=False` for full-grid reasoning (Sudoku).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d, ratio=2.67):
        super().__init__()
        h = int(d * ratio)
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GeomMix(nn.Module):
    """Geometric product done blockwise on 2D blades: per pair -> [u·v, u∧v]. ~3 d^2 params."""
    def __init__(self, d):
        super().__init__()
        assert d % 2 == 0
        self.u = nn.Linear(d, d, bias=False)
        self.w = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        u = self.u(x).unflatten(-1, (-1, 2))
        w = self.w(x).unflatten(-1, (-1, 2))
        inner = (u * w).sum(-1)
        wedge = u[..., 0] * w[..., 1] - u[..., 1] * w[..., 0]
        return self.o(torch.cat([inner, wedge], -1))


class Attn(nn.Module):
    def __init__(self, d, heads, causal):
        super().__init__()
        self.h, self.dh, self.causal = heads, d // heads, causal
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = (z.view(B, T, self.h, self.dh).transpose(1, 2) for z in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.proj(o.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d, heads, mixer, causal):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = Attn(d, heads, causal)
        self.mix = {"std": lambda: SwiGLU(d), "geom": lambda: GeomMix(d),
                    "attn": lambda: None}[mixer]()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        if self.mix is not None:
            x = x + self.mix(self.ln2(x))
        return x


class ClairNet(nn.Module):
    def __init__(self, arch, vocab, d, n_layers, heads=4, ctx=64, causal=True, depth=3):
        super().__init__()
        self.arch, self.causal = arch, causal
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        mixer = "geom" if arch.startswith("geom") else arch
        self.recurrent = arch == "geom_recur"
        n_blocks = 1 if self.recurrent else n_layers
        self.blocks = nn.ModuleList([Block(d, heads, mixer, causal) for _ in range(n_blocks)])
        self.unrolls = (n_layers * depth) if self.recurrent else n_layers
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, idx, targets=None, ignore=-1, deep_supervision=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        h = self.tok(idx) + self.pos(pos)
        loss = None
        for u in range(self.unrolls):
            blk = self.blocks[0] if self.recurrent else self.blocks[u]
            h = blk(h)
            if (targets is not None and deep_supervision and self.recurrent
                    and u >= self.unrolls - 3):
                lg = self.head(self.ln_f(h))
                ls = F.cross_entropy(lg.reshape(-1, lg.size(-1)), targets.reshape(-1), ignore_index=ignore)
                loss = ls if loss is None else loss + ls
        logits = self.head(self.ln_f(h))
        if targets is not None and loss is None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=ignore)
        return logits, loss

    def n_params(self, non_embed=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embed:
            n -= self.tok.weight.numel() + self.pos.weight.numel()
        return n


def size_for(arch, target, vocab, n_layers, heads, ctx, causal, depth=3):
    """Binary-search d_model (multiple of 2*heads) for non-embed params ~= target."""
    lo, hi, best = 2 * heads, 1280, None
    nl = 1 if arch == "geom_recur" else n_layers
    while lo <= hi:
        d = max(2 * heads, ((lo + hi) // 2 // (2 * heads)) * (2 * heads))
        np_ = ClairNet(arch, vocab, d, nl, heads, ctx, causal, depth).n_params()
        best = (d, np_)
        if np_ < target:
            lo = d + 2 * heads
        else:
            hi = d - 2 * heads
    return best
