"""Faithful Lattice Deduction Transformer (LDT, arXiv 2605.08605).

A recurrent transformer whose latent state is projected onto an abstract lattice (the
grid-powerset lattice, one V-candidate set per cell) between forward passes. Each forward
pass is one *sound deduction step*: it emits per-candidate SURVIVAL logits b (push toward 0
to eliminate a candidate) and a CLS *conflict* logit c (fire on any unsatisfiable state).

Architecture (paper §4.1):
  * lattice encoding: V alive-bits + a 'given' flag per cell -> d, + learned 2D position.
  * recurrent core: a stack of `n_layers` attention layers (LN->MHA->res, LN->mixer->res),
    weight-TIED, unrolled for `inner` internal iterations; the input lattice encoding is
    RE-INJECTED as a residual at every iteration. Each iteration emits its own (b, c) ->
    DEEP SUPERVISION over all iterations.
  * heads: b_head -> [B,P,V] survival logits ; cls_head -> [B] conflict logit (pooled).

The channel mixer is SWAPPABLE for an eventual wedge ablation:
    ffn      : SwiGLU feed-forward            (DEFAULT / baseline LDT core)
    geom     : Clifford geometric product     (dot + wedge)
    nowedge  : geometric product, dot-only    (kills the antisymmetric branch)

Default d=128, n_layers=4, inner=16 reproduces the paper's ~800K-parameter LDT.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- mixers
class SwiGLU(nn.Module):
    """Baseline LDT channel mixer: gated feed-forward."""
    def __init__(self, d, ratio=2.67):
        super().__init__()
        h = int(d * ratio)
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GeomMix(nn.Module):
    """Shifted geometric product (dot coherence + wedge exclusion). wedge=False -> dot-only."""
    def __init__(self, d, shifts=(1, 2, 4, 8, 15), wedge=True):
        super().__init__()
        self.shifts = tuple(s % d for s in shifts)
        self.wedge = wedge
        self.u = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        per = 2 if wedge else 1
        self.proj = nn.Linear(d * len(self.shifts) * per, d, bias=False)

    def forward(self, x):
        u = self.u(x)
        v = self.v(x)
        outs = []
        for s in self.shifts:
            vr = torch.roll(v, shifts=s, dims=-1)
            outs.append(F.silu(u * vr))                       # dot / coherence
            if self.wedge:
                ur = torch.roll(u, shifts=s, dims=-1)
                outs.append(u * vr - ur * v)                  # wedge / exclusion
        return self.proj(torch.cat(outs, dim=-1))


def make_mixer(mixer, d):
    return {
        "ffn": lambda: SwiGLU(d),
        "geom": lambda: GeomMix(d, wedge=True),
        "nowedge": lambda: GeomMix(d, wedge=False),
    }[mixer]()


# --------------------------------------------------------------------------- transformer layer
class Attn(nn.Module):
    """Full (bidirectional) attention over the fixed set of grid cells = the constraint graph."""
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = (z.view(B, T, self.h, self.dh).transpose(1, 2) for z in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.proj(o.transpose(1, 2).reshape(B, T, D))


class Layer(nn.Module):
    """One transformer layer: LN -> MHA -> residual, LN -> mixer -> residual."""
    def __init__(self, d, heads, mixer):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = Attn(d, heads)
        self.mix = make_mixer(mixer, d)

    def forward(self, h):
        h = h + self.attn(self.ln1(h))
        h = h + self.mix(self.ln2(h))
        return h


# --------------------------------------------------------------------------- the LDT
class LDT(nn.Module):
    """Recurrent lattice-deduction transformer.

    forward(state, given) -> (b_final, c_final, sup) where
        b_final : [B,P,V] survival logits at the last iteration,
        c_final : [B]     conflict logit at the last iteration,
        sup     : list of (b^l, c^l) over the deeply-supervised iterations.
    """
    def __init__(self, P, V, d=128, heads=4, n_layers=4, inner=16, ds=16,
                 mixer="ffn", reinject=0.1):
        super().__init__()
        self.P, self.V = P, V
        self.inner, self.ds, self.reinject = inner, ds, reinject
        self.mixer = mixer
        self.in_proj = nn.Linear(V + 1, d)                 # V alive-bits + 'given' flag
        self.pos = nn.Embedding(P, d)                      # learned 2D position (per cell)
        self.layers = nn.ModuleList([Layer(d, heads, mixer) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.b_head = nn.Linear(d, V)                      # candidate-survival logits
        self.cls_head = nn.Linear(d, 1)                    # conflict (unsat / bottom) logit

    def encode(self, state, given):
        x = torch.cat([state, given.unsqueeze(-1).to(state.dtype)], dim=-1)
        return self.in_proj(x) + self.pos(torch.arange(self.P, device=state.device))

    def core(self, h):
        for layer in self.layers:
            h = layer(h)
        return h

    def forward(self, state, given):
        h0 = self.encode(state, given)
        h = h0
        sup = []
        for r in range(self.inner):
            h = self.core(h) + self.reinject * h0          # re-inject the problem each iteration
            if r >= self.inner - self.ds:
                f = self.ln_f(h)
                sup.append((self.b_head(f), self.cls_head(f.mean(1)).squeeze(-1)))
        b, c = sup[-1]
        return b, c, sup

    def n_params(self, non_embed=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embed:
            n -= self.pos.weight.numel()
        return n
