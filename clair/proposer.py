"""Factor-graph lattice-deduction PROPOSER — the crux experiment.

A full geometric product as a CHECKED proposer. The exact finite-CSP harness (`clair.csp`) is the
authority: it computes the exact per-cell transformer dedₚ (every value used by SOME solution
consistent with the current domain). We train this neural proposer to DOMINATE dedₚ — never
eliminate a value dedₚ keeps — so SOUNDNESS is free, and the research variable is COMPLETENESS:
does the proposer drive solvable CSPs to a checkable `solved` state, or does it abstain?

The proposer is domain-blind and sees the PROBLEM PROGRAM (the prior bug was seeing only
alive/given). It encodes a CSP + current per-cell domain as a BIPARTITE FACTOR GRAPH:

    VARIABLE node i  : feature = multi-hot live-candidate mask over d values (+ a 'decided' flag)
    FACTOR   node f  : feature = a learned encoding of the constraint's RELATION TABLE
                       (broadcast to a_max axes) + an arity one-hot
    edge (f, p) -> i : scope position p of factor f points to variable i (position-embedded)

R rounds of variable<->factor message passing, then a per-VARIABLE channel mixer that is the
ABLATABLE PROPOSER:

    arm in {ffn, inner, wedge, full}
      ffn   : SwiGLU baseline (no geometric product)
      inner : shifted dot / coherence only          (CliffordNet grade-0)
      wedge : shifted bivector / exclusion only      (CliffordNet grade-2, antisymmetric)
      full  : dot + wedge  = the FULL geometric product   (the bet)

Heads: per-cell per-value survival logits b[n,d] (sigmoid -> keep prob, used by a MONOTONE meet
that only narrows) and a pooled conflict logit (state is unsat). Padded to (n_max, d_max, m_max,
a_max); everything is masked. Arms are sized iso-param via `size_for`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- channel mixers (arms)
class SwiGLU(nn.Module):
    """ffn arm — the non-geometric baseline proposer."""
    def __init__(self, d, ratio=2.67):
        super().__init__()
        h = int(d * ratio)
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GeomArm(nn.Module):
    """Shifted geometric product (reuses the CliffordNet formulation from clair.glados.GeomMix),
    as a self-mixing per-variable channel mixer with an inner/wedge/full mode switch.

        u = U(x), v = V(x)
        dot_s   = SiLU( u ⊙ roll_s(v) )                 grade-0  (coherence / gating)
        wedge_s = u ⊙ roll_s(v) − roll_s(u) ⊙ v         grade-2  (antisymmetric exclusion)

    inner -> keep dot_s only; wedge -> keep wedge_s only; full -> concat both (the geometric product).
    """
    def __init__(self, d, mode, shifts=(1, 2, 4)):
        super().__init__()
        assert mode in ("inner", "wedge", "full")
        self.mode = mode
        self.shifts = tuple(s % d for s in shifts)
        self.u = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        per = 2 if mode == "full" else 1
        self.proj = nn.Linear(d * len(self.shifts) * per, d, bias=False)

    def forward(self, x):
        u = self.u(x)
        v = self.v(x)
        outs = []
        for s in self.shifts:
            vr = torch.roll(v, shifts=s, dims=-1)
            if self.mode in ("inner", "full"):
                outs.append(F.silu(u * vr))               # dot / coherence
            if self.mode in ("wedge", "full"):
                ur = torch.roll(u, shifts=s, dims=-1)
                outs.append(u * vr - ur * v)              # wedge / exclusion
        return self.proj(torch.cat(outs, dim=-1))


def make_mixer(arm, d):
    if arm == "ffn":
        return SwiGLU(d)
    return GeomArm(d, mode=arm)


# --------------------------------------------------------------------------- the factor-graph proposer
class FactorGraphProposer(nn.Module):
    """Bipartite variable<->factor message-passing proposer; per-variable mixer = the ablatable arm.

    Permutation-equivariant over cells (no absolute cell-position embedding): the factor graph is
    the only source of asymmetry, so the same weights run on any (n, d, cons) up to padding.
    """
    def __init__(self, arm, n_max, d_max, m_max, a_max, d=64, R=8, ds=4):
        super().__init__()
        assert d % 2 == 0
        self.arm = arm
        self.n_max, self.d_max, self.m_max, self.a_max = n_max, d_max, m_max, a_max
        self.R, self.ds = R, ds
        rel_dim = d_max ** a_max
        self.var_in = nn.Linear(d_max + 1, d)             # alive-mask over d values + 'decided' flag
        self.fac_in = nn.Linear(rel_dim + a_max, d)       # broadcast relation table + arity one-hot
        self.pos = nn.Embedding(a_max, d)                 # scope-position embedding (which slot)
        self.vf = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))   # var -> factor message
        self.fv = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))   # factor -> var message
        self.fac_ln = nn.LayerNorm(d)
        self.var_ln = nn.LayerNorm(d)
        self.mix_ln = nn.LayerNorm(d)
        self.mixer = make_mixer(arm, d)                   # THE ABLATABLE PROPOSER
        self.b_head = nn.Linear(d, d_max)                 # per-cell per-value survival logit
        self.cls_head = nn.Linear(d, 1)                   # pooled conflict (unsat) logit
        self.d = d

    def forward(self, var_mask, given, fac_rel, fac_arity, edge_var, edge_valid, var_valid, fac_valid):
        """All tensors batched. Shapes:
          var_mask  [B,n,dv]  alive candidates in {0,1}   given [B,n]   var_valid [B,n]
          fac_rel   [B,m,dv**a] broadcast relation table  fac_arity [B,m,a] one-hot  fac_valid [B,m]
          edge_var  [B,m,a] long (variable index per scope slot; pad slots = n_max)
          edge_valid[B,m,a] in {0,1}
        Returns (b_final [B,n,dv], cls_final [B], sup=list of (b,cls) for deep supervision)."""
        B, n, dv = var_mask.shape
        a, D = self.a_max, self.d
        hv = self.var_in(torch.cat([var_mask, given.unsqueeze(-1)], dim=-1))   # [B,n,D]
        hf = self.fac_in(torch.cat([fac_rel, fac_arity], dim=-1))              # [B,m,D]
        pos = self.pos(torch.arange(a, device=hv.device))                     # [a,D]
        idx = edge_var.reshape(B, -1, 1).expand(-1, -1, D)                    # [B,m*a,D] gather/scatter index
        ev = edge_valid.unsqueeze(-1)                                          # [B,m,a,1]
        fvld = fac_valid.unsqueeze(-1)                                         # [B,m,1]
        sup = []
        for r in range(self.R):
            # ---- variable -> factor ----
            hv_pad = torch.cat([hv, torch.zeros(B, 1, D, device=hv.device, dtype=hv.dtype)], dim=1)  # [B,n+1,D]
            gathered = torch.gather(hv_pad, 1, idx).reshape(B, self.m_max, a, D)                      # [B,m,a,D]
            msg_vf = self.vf(gathered + pos) * ev                                                     # mask pad slots
            hf = self.fac_ln(hf + msg_vf.sum(2))
            # ---- factor -> variable ----
            msg_fv = self.fv(hf.unsqueeze(2).expand(-1, -1, a, -1) + pos) * ev * fvld.unsqueeze(2)
            tgt = torch.zeros(B, self.n_max + 1, D, device=hv.device, dtype=hv.dtype)
            tgt.scatter_add_(1, idx, msg_fv.reshape(B, -1, D))
            hv = self.var_ln(hv + tgt[:, :self.n_max, :])
            # ---- per-variable channel mixer (the ARM) ----
            hv = hv + self.mixer(self.mix_ln(hv))
            if r >= self.R - self.ds:
                b = self.b_head(hv)
                denom = var_valid.sum(1, keepdim=True).clamp(min=1.0)
                pooled = (hv * var_valid.unsqueeze(-1)).sum(1) / denom        # masked mean over real cells
                cls = self.cls_head(pooled).squeeze(-1)
                sup.append((b, cls))
        b, cls = sup[-1]
        return b, cls, sup

    def n_params(self, non_embed=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embed:
            n -= self.pos.weight.numel()
        return n


def size_for(arm, n_max, d_max, m_max, a_max, target, R=8, ds=4):
    """Binary-search d_model (even) so non-embed params ~= target. Gives iso-param arms."""
    lo, hi, best = 2, 1024, None
    while lo <= hi:
        d = max(2, ((lo + hi) // 2 // 2) * 2)
        n = FactorGraphProposer(arm, n_max, d_max, m_max, a_max, d=d, R=R, ds=ds).n_params()
        best = (d, n)
        if n < target:
            lo = d + 2
        else:
            hi = d - 2
    return best
