"""GLaDOSv — GLaDOS with the LAYERING made into knobs.

Same organ as `clair/glados.py` (attn builds constraint context, a geom/FFN mixer
PROPOSES, the lattice MEET narrows), but the *structural* choices that the baseline
fixed are exposed as flags, each DEFAULTING TO THE CURRENT BEHAVIOR so GLaDOSv with
no knobs reproduces GLaDOS bit-for-bit (same param count, same forward math).

The hypothesis under test: the recurrence should itself be CONTRACTIVE on the lattice.
Baseline GLaDOS only meets BETWEEN forward passes (in `sudoku.step_state`); inside the
`inner` loop nothing narrows — `meet_inside` weaves a differentiable narrowing in.

Knobs (clair/run_glados_variants.py exposes them as CLI flags):
  meet_inside  {none, soft, hard}   narrow INSIDE the loop (differentiable / straight-through)
  order        {attn_then_geom, geom_then_attn, parallel}   Cell composition
  reinject     {add0p1, gated, concat, none}   how problem h0 re-enters each step
  tie          {tied, untied}       one looped cell vs `inner` distinct cells
  couple_state {bool}               feed raw candidate-logits b back into the next iter input
  ds_window    int                  how many trailing iters get deep supervision

The meet/lattice ops still live in clair.sudoku; the narrowing here is the *differentiable*
in-loop analogue (gradients flow through it), distinct from the hard no_grad meet between passes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .glados import SwiGLU, GeomMix, Attn


# --------------------------------------------------------------------------- cell
class CellV(nn.Module):
    """One deduction step, with composition `order` as a knob.

    attn(ln1 h) builds constraint context c; mix(ln2 ·, ln2 ·) proposes. The three orders:
      attn_then_geom : h+=attn(h); h+=mix(h,h)               (baseline)
      geom_then_attn : h+=mix(h,h); h+=attn(h)
      parallel       : h += attn(h) + mix(h,h)               (both off the same input)
    """
    def __init__(self, d, heads, arm, order="attn_then_geom"):
        super().__init__()
        self.order = order
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = Attn(d, heads)
        self.mix = {
            "ldt": lambda: SwiGLU(d),
            "glados": lambda: GeomMix(d, wedge=True),
            "nowedge": lambda: GeomMix(d, wedge=False),
        }[arm]()

    def _geom(self, h):
        g = self.ln2(h)
        return self.mix(g, g)            # geom: state×context off the SAME normed stream (baseline)

    def forward(self, h):
        if self.order == "attn_then_geom":
            h = h + self.attn(self.ln1(h))
            h = h + self._geom(h)
        elif self.order == "geom_then_attn":
            h = h + self._geom(h)
            h = h + self.attn(self.ln1(h))
        elif self.order == "parallel":
            a = self.attn(self.ln1(h))
            g = self._geom(h)
            h = h + a + g
        else:
            raise ValueError(f"order={self.order}")
        return h


# --------------------------------------------------------------------------- reasoner
class GLaDOSv(nn.Module):
    """Recurrent lattice-deduction reasoner with the layering exposed as knobs.

    state  : per-position candidate logits b ∈ [B, P, V].
    encode : multi-hot lattice membership + given flag + learned position -> h0.
    recur  : `inner` cell iterations; per knob: re-inject h0, optionally couple the raw
             candidate-logit state b, optionally MEET (narrow) inside the loop.
    heads  : b_head -> candidate-survival logits ;  cls_head -> conflict (⊥) logit.

    Defaults reproduce GLaDOS(arm,P,V,...) exactly.
    """
    def __init__(self, arm, P, V, d=128, heads=4, inner=16, ds=16, tied=True,
                 meet_inside="none", order="attn_then_geom", reinject="add0p1",
                 couple_state=False, ds_window=None):
        super().__init__()
        self.arm, self.P, self.V = arm, P, V
        self.inner, self.tied = inner, tied
        self.meet_inside, self.order, self.reinject = meet_inside, order, reinject
        self.couple_state = couple_state
        # ds_window defaults to ds (baseline supervises the last `ds` iters; baseline ds==inner).
        self.ds_window = inner if ds_window is None else min(ds_window, inner)

        # input projection. baseline: Linear(V+1, d). couple_state feeds the running candidate
        # logits b ([B,P,V]) back in too -> Linear(V+1 + V, d); zero-init the b-block so step 0
        # (b not yet defined -> zeros) matches the baseline exactly and the coupling is learned.
        in_dim = (V + 1) + (V if couple_state else 0)
        self.in_proj = nn.Linear(in_dim, d)
        if couple_state:
            with torch.no_grad():
                self.in_proj.weight[:, V + 1:].zero_()
        self.pos = nn.Embedding(P, d)

        n_cells = 1 if tied else inner
        self.cells = nn.ModuleList([CellV(d, heads, arm, order) for _ in range(n_cells)])
        self.ln_f = nn.LayerNorm(d)
        self.b_head = nn.Linear(d, V)               # survival logit per candidate
        self.cls_head = nn.Linear(d, 1)             # conflict logit (pooled)

        # reinject machinery -------------------------------------------------
        if reinject == "gated":
            # h <- h + sigmoid(g) * h0 with a learned scalar-per-channel gate, g init s.t.
            # sigmoid(g)=0.1 == baseline's fixed 0.1 coefficient (logit(0.1) ≈ -2.1972246).
            self.reinj_gate = nn.Parameter(torch.full((d,), -2.1972246))
        elif reinject == "concat":
            # h <- proj_r([h ; h0]); init proj_r = [I | 0.1 I] to reproduce the additive baseline.
            self.reinj_proj = nn.Linear(2 * d, d, bias=False)
            with torch.no_grad():
                eye = torch.eye(d)
                self.reinj_proj.weight.copy_(torch.cat([eye, 0.1 * eye], dim=1))

        # meet_inside machinery ---------------------------------------------
        # The differentiable in-loop narrowing acts on a candidate-MEMBERSHIP embedding:
        #   emb(g) = g_membed(g)   where g is a per-candidate survival GATE in [0,1]^[B,P,V].
        # 'soft': g = sigmoid(b_inner / tau)            (a differentiable meet — grads flow through)
        # 'hard': g = STE_threshold(sigmoid(b_inner))   (forward 0/1, backward = soft)
        # The gate narrows a learned membership embedding that is ADDED into h, so the recurrence
        # contracts as candidates die. g_membed is zero-init -> baseline (no in-loop narrowing).
        if meet_inside in ("soft", "hard"):
            self.b_head_inner = nn.Linear(d, V)       # read survival logits inside the loop
            self.g_membed = nn.Linear(V, d, bias=False)
            with torch.no_grad():
                self.g_membed.weight.zero_()          # zero-init: step-0 == baseline, narrowing learned
            self.meet_tau = 1.0
            self.meet_theta = 0.5                     # hard-threshold on the gate (prob space)
        elif meet_inside != "none":
            raise ValueError(f"meet_inside={meet_inside}")

    # ----------------------------------------------------------------- encode
    def encode(self, state, given, b_prev=None):
        feats = [state, given.unsqueeze(-1).to(state.dtype)]
        if self.couple_state:
            if b_prev is None:
                b_prev = torch.zeros_like(state)      # step 0: no candidate logits yet
            feats.append(b_prev.to(state.dtype))
        x = torch.cat(feats, dim=-1)
        h0 = self.in_proj(x) + self.pos(torch.arange(self.P, device=state.device))
        return h0

    # --------------------------------------------------- differentiable meet
    def _narrow(self, h, alive):
        """Compute an in-loop survival gate g and the membership-narrowing to add to h.

        alive [B,P,V] in {0,1}: the (no_grad) lattice mask of currently-alive candidates;
        the gate is only allowed to narrow WITHIN it (dead candidates stay dead), mirroring
        the soundness of the external meet. Returns (delta_h, b_inner) where delta_h is the
        narrowing signal to add to h and b_inner are the in-loop survival logits (for hard STE
        we still backprop through the soft gate)."""
        b_inner = self.b_head_inner(self.ln_f(h))     # [B,P,V] survival logits
        soft = torch.sigmoid(b_inner / self.meet_tau)
        if alive is not None:
            soft = soft * alive                        # never resurrect a dead candidate
        if self.meet_inside == "soft":
            g = soft
        else:  # hard: straight-through threshold (forward hard 0/1, backward soft)
            hard = (soft >= self.meet_theta).to(soft.dtype)
            g = hard + (soft - soft.detach())
        return self.g_membed(g), b_inner

    # ----------------------------------------------------------------- forward
    def forward(self, state, given, alive=None):
        """Returns (b_logits_final, cls_logit_final, sup). `alive` is the optional no_grad lattice
        mask used to keep the in-loop meet sound; defaults to `state` (which IS the alive mask in
        the sudoku loop). Baseline ignores it."""
        if alive is None:
            alive = state
        b_prev = None
        h0 = self.encode(state, given, b_prev)
        h = h0
        sup = []
        start = self.inner - self.ds_window
        for r in range(self.inner):
            cell = self.cells[0] if self.tied else self.cells[r]
            h = cell(h)

            # re-inject the problem ----------------------------------------
            if self.reinject == "add0p1":
                h = h + 0.1 * h0
            elif self.reinject == "gated":
                h = h + torch.sigmoid(self.reinj_gate) * h0
            elif self.reinject == "concat":
                h = self.reinj_proj(torch.cat([h, h0], dim=-1))
            elif self.reinject == "none":
                pass
            else:
                raise ValueError(f"reinject={self.reinject}")

            # in-loop differentiable meet ----------------------------------
            if self.meet_inside != "none":
                delta, b_inner = self._narrow(h, alive)
                h = h + delta
                if self.couple_state:
                    # close propose->narrow->propose: rebuild h0 with the narrowed candidate logits
                    h0 = self.encode(state, given, b_inner)

            if r >= start:
                f = self.ln_f(h)
                sup.append((self.b_head(f), self.cls_head(f.mean(1)).squeeze(-1)))
        b, cls = sup[-1]
        return b, cls, sup

    def n_params(self, non_embed=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embed:
            n -= self.pos.weight.numel()
        return n


def size_for(arm, P, V, target, heads=4, inner=16, ds=16, tied=True, **knobs):
    """Binary-search d (multiple of 2*heads) for non-embed params ~= target, AT the knob config
    (so each variant is sized iso-param to its own structure)."""
    lo, hi, best = 2 * heads, 1024, None
    while lo <= hi:
        d = max(2 * heads, ((lo + hi) // 2 // (2 * heads)) * (2 * heads))
        n = GLaDOSv(arm, P, V, d, heads, inner, ds, tied, **knobs).n_params()
        best = (d, n)
        if n < target:
            lo = d + 2 * heads
        else:
            hi = d - 2 * heads
    return best
