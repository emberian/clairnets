"""GLaDOS — Geometric Lattice Deduction Over Streams.

A recurrent reasoner whose state is a point in an abstract domain (a per-position
powerset lattice) and whose recurrent cell PROPOSES candidate eliminations via a
Clifford geometric product, while the lattice MEET stays the sole authority on what
may actually be eliminated:

    a_{t+1} = a_t ⊓ Π( f_θ(a_t, problem) )            (the spine)

f_θ is the geometric-product cell (CliffordNet 2601.06793); the meet/narrowing and
on-policy alpha-target supervision are the Lattice Deduction Transformer (2605.08605).

Three iso-param arms, so the ONLY thing that varies is the channel mixer (this is the
decisive core-swap experiment — does the wedge buy lower false-elimination at equal params?):

    ldt     : MHA + SwiGLU FFN          (baseline LDT core)
    glados  : MHA + GeomMix (no FFN)    (geometric product: dot + wedge)
    nowedge : MHA + GeomMix dot-only    (ablation: kill the antisymmetric branch)

The model is domain-blind: it sees (candidate-lattice state, constraint positions),
never "sudoku". Same cell is meant to run on maze / graph-coloring / SAT later.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- mixers
class SwiGLU(nn.Module):
    def __init__(self, d, ratio=2.67):
        super().__init__()
        h = int(d * ratio)
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(h, d, bias=False)

    def forward(self, x, c):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GeomMix(nn.Module):
    """Shifted geometric product between the state stream u and a context stream v.

    Over a fixed set of cyclic channel shifts S (the 'streams'):
        dot_s   = SiLU( u ⊙ roll_s(v) )                 grade-0  (coherence / gating)
        wedge_s = u ⊙ roll_s(v) − roll_s(u) ⊙ v         grade-2  (antisymmetric exclusion)
    concat over S, project back to d. O(N·d·|S|), no FFN.  wedge=False → dot-only ablation.
    """
    def __init__(self, d, shifts=(1, 2, 4, 8, 15), wedge=True):
        super().__init__()
        self.shifts = tuple(s % d for s in shifts)
        self.wedge = wedge
        self.u = nn.Linear(d, d, bias=False)   # state stream
        self.v = nn.Linear(d, d, bias=False)   # context stream (constraint evidence)
        per = 2 if wedge else 1
        self.proj = nn.Linear(d * len(self.shifts) * per, d, bias=False)

    def forward(self, x, c):
        u = self.u(x)
        v = self.v(c)
        outs = []
        for s in self.shifts:
            vr = torch.roll(v, shifts=s, dims=-1)
            outs.append(F.silu(u * vr))                       # dot / coherence
            if self.wedge:
                ur = torch.roll(u, shifts=s, dims=-1)
                outs.append(u * vr - ur * v)                  # wedge / exclusion
        return self.proj(torch.cat(outs, dim=-1))


# --------------------------------------------------------------------------- attention (constraint context)
class Attn(nn.Module):
    """Full attention over the (small, fixed) set of grid positions = the constraint graph,
    learned rather than hard-wired so the same cell ports across domains."""
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


class Cell(nn.Module):
    """One recurrent deduction step: attention builds constraint context c from the
    current state h, then the mixer proposes an update. The arm picks the mixer."""
    def __init__(self, d, heads, arm):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = Attn(d, heads)
        self.mix = {
            "ldt": lambda: SwiGLU(d),
            "glados": lambda: GeomMix(d, wedge=True),
            "nowedge": lambda: GeomMix(d, wedge=False),
        }[arm]()

    def forward(self, h):
        c = self.attn(self.ln1(h))      # constraint evidence
        h = h + c
        h = h + self.mix(self.ln2(h), self.ln2(h))   # propose (geom: state×context)
        return h


# --------------------------------------------------------------------------- the reasoner
class GLaDOS(nn.Module):
    """Recurrent lattice-deduction reasoner.

    state  : per-position candidate logits b ∈ [B, P, V]  (V candidates per position).
    encode : multi-hot lattice membership + givens + learned 2D position.
    recur  : `inner` weight-shared cell iterations per forward pass (LDT uses 16),
             with input re-injection; deep supervision over the last `ds` iterations.
    heads  : b_head → candidate-survival logits;  cls_head → conflict (unsat) logit.
    """
    def __init__(self, arm, P, V, d=128, heads=4, inner=16, ds=16, tied=True, n_layers=1):
        super().__init__()
        self.arm, self.P, self.V = arm, P, V
        self.inner, self.ds, self.tied = inner, ds, tied
        self.in_proj = nn.Linear(V + 1, d)          # V candidate-bits + 1 'given' flag
        self.pos = nn.Embedding(P, d)
        n_cells = 1 if tied else inner
        self.cells = nn.ModuleList([Cell(d, heads, arm) for _ in range(n_cells)])
        self.ln_f = nn.LayerNorm(d)
        self.b_head = nn.Linear(d, V)               # survival logit per candidate
        self.cls_head = nn.Linear(d, 1)             # conflict logit (pooled)

    def encode(self, state, given):
        # state: [B,P,V] in {0,1} alive-mask ; given: [B,P] in {0,1}
        x = torch.cat([state, given.unsqueeze(-1).to(state.dtype)], dim=-1)
        h0 = self.in_proj(x) + self.pos(torch.arange(self.P, device=state.device))
        return h0

    def forward(self, state, given):
        """Returns (b_logits_final, cls_logit_final, list_of_(b,cls)_for_deep_supervision)."""
        h0 = self.encode(state, given)
        h = h0
        sup = []
        for r in range(self.inner):
            cell = self.cells[0] if self.tied else self.cells[r]
            h = cell(h) + 0.1 * h0                      # re-inject the problem every step (LDT)
            if r >= self.inner - self.ds:
                f = self.ln_f(h)
                sup.append((self.b_head(f), self.cls_head(f.mean(1)).squeeze(-1)))
        b, cls = sup[-1]
        return b, cls, sup

    def n_params(self, non_embed=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embed:
            n -= self.pos.weight.numel()
        return n


def size_for(arm, P, V, target, heads=4, inner=16, ds=16, tied=True):
    """Binary-search d (multiple of 2*heads) for non-embed params ~= target. Iso-param arms."""
    lo, hi, best = 2 * heads, 1024, None
    while lo <= hi:
        d = max(2 * heads, ((lo + hi) // 2 // (2 * heads)) * (2 * heads))
        n = GLaDOS(arm, P, V, d, heads, inner, ds, tied).n_params()
        best = (d, n)
        if n < target:
            lo = d + 2 * heads
        else:
            hi = d - 2 * heads
    return best
