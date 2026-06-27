"""clair/organ/alpha_struct.py — the MULTI-FACULTY α STRUCTURE COMPILER (the "rack").

This is the architectural core of ALPHA_STRUCT (notes/alpha_struct_design.md). The SHUFFLED diagnostic
(runs/shuffled_diag.json) proved today's woven is 100% metadata-driven: the certified floor runs on
`rec["csp"]` (the true structure), so α contributes nothing. The fix is to make α EMIT the structure.

But α must NOT be a lone CSP factor-graph head — else the woven is forever CSP-narrow-only and the whole
bank (Ising/graph/type/reduction faculties) is wasted. So α is a RACK:

    a shared per-cell ENCODER (mention-pool identity ⊕ CellReader cross-attn read of the prompt)
      → a ROUTER head        : host hidden → faculty {csp, ising, graph, type, reduction}
      → per-faculty STRUCTURE heads, each emitting that faculty's native structure:
          csp       : pin head [B,N,1+K] + typed pair-relation head [B,N,N,R] over {none,eq,neq,lt,le}
          ising     : couplings J [B,N,N] + fields h [B,N]               (scaffold, trainable)
          graph     : edge logits [B,N,N]                                (scaffold)
          type      : per-cell type logits [B,N,Tt]                      (scaffold)
          reduction : chosen reduction-route logits [B, Rroutes]         (scaffold)

Adding a faculty = adding a head to `self.heads` (a ModuleDict) + a route label. Nothing else moves.

The probe (run_alpha_struct_probe.py) trains+evaluates the CSP head + the router on FROZEN OLMo-2-1B
hidden — the make-or-break "can α read problem STRUCTURE off the host hidden?" capability test.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..augmented import CellReader

# typed pair-relation vocabulary for the CSP faculty (notes/alpha_struct_design.md §1.2 minimal set)
PAIR_RELS = ["none", "eq", "neq", "lt", "le"]
R_PAIR = len(PAIR_RELS)
SYMMETRIC = {"eq", "neq"}            # symmetric relations (supervise both orders, dedup at decode)
DIRECTED = {"lt", "le"}             # directed relations (ordered pair a→b)

FACULTIES = ["csp", "ising", "graph", "type", "reduction"]


# ----------------------------------------------------------------- shared per-cell encoder
class CellEncoder(nn.Module):
    """OLMo hidden → per-cell fused feature feat_i = [mention-pool identity ; cross-attn read].
    Identical fusion to DenseLatentProjector (latent_organ.py:50) so the rack reuses the validated
    'copy-from-text + bind' mechanism: mentions give the binding, CellReader reads the relation."""
    def __init__(self, D, dp=384, heads=6):
        super().__init__()
        self.reader = CellReader(D, dp, heads)
        self.din = D + dp

    def forward(self, v_mean, h, attn_mask):
        ctx = self.reader(v_mean, h, attn_mask)        # [B,N,dp]
        return torch.cat([v_mean, ctx], dim=-1)        # [B,N,din]


# ----------------------------------------------------------------- CSP faculty head
class CSPStructureHead(nn.Module):
    """Emit the typed factor graph over the mentioned cells:
       pin head (unary): feat_i → logits over {no-pin, 0..K-1}                       → [B,N,1+K]
       pair head (binary): a bilinear/MLP edge predictor over ORDERED pairs (i,j)    → [B,N,N,R]
    The bind is solved by construction (the pair indices ARE the cell identities); α only recognizes
    the relation off the cross-attn context between the two cells' spans."""
    def __init__(self, din, K, hidden=384, pair_hidden=256):
        super().__init__()
        self.K = K
        self.pin = nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, 1 + K))
        self.pl = nn.Linear(din, pair_hidden)
        self.pr = nn.Linear(din, pair_hidden)
        self.pair_mlp = nn.Sequential(nn.Linear(2 * pair_hidden, pair_hidden), nn.GELU(),
                                      nn.Linear(pair_hidden, R_PAIR))

    def forward(self, feat):
        B, N, _ = feat.shape
        pin_logits = self.pin(feat)                                   # [B,N,1+K]
        li, rj = self.pl(feat), self.pr(feat)                        # [B,N,H]
        H = li.size(-1)
        pij = torch.cat([li[:, :, None, :].expand(B, N, N, H),
                         rj[:, None, :, :].expand(B, N, N, H)], dim=-1)
        pair_logits = self.pair_mlp(pij)                             # [B,N,N,R]
        return pin_logits, pair_logits


# ----------------------------------------------------------------- scaffolded non-CSP faculty heads
class IsingStructureHead(nn.Module):
    """Ising/energy faculty: emit pairwise couplings J [B,N,N] (sign = align/anti) + fields h [B,N].
    Fully trainable (used for the 2nd-faculty viability probe)."""
    def __init__(self, din, hidden=256):
        super().__init__()
        self.jl = nn.Linear(din, hidden)
        self.jr = nn.Linear(din, hidden)
        self.j_mlp = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.h_head = nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, feat):
        B, N, _ = feat.shape
        a, b = self.jl(feat), self.jr(feat)
        H = a.size(-1)
        pij = torch.cat([a[:, :, None, :].expand(B, N, N, H),
                         b[:, None, :, :].expand(B, N, N, H)], dim=-1)
        J = self.j_mlp(pij).squeeze(-1)                              # [B,N,N]
        J = 0.5 * (J + J.transpose(1, 2))                           # symmetric couplings
        h = self.h_head(feat).squeeze(-1)                           # [B,N]
        return J, h


class GraphStructureHead(nn.Module):
    """Graph faculty (scaffold): undirected/directed edge logits [B,N,N]."""
    def __init__(self, din, hidden=256):
        super().__init__()
        self.el = nn.Linear(din, hidden)
        self.er = nn.Linear(din, hidden)
        self.mlp = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, feat):
        B, N, _ = feat.shape
        a, b = self.el(feat), self.er(feat)
        H = a.size(-1)
        pij = torch.cat([a[:, :, None, :].expand(B, N, N, H),
                         b[:, None, :, :].expand(B, N, N, H)], dim=-1)
        return self.mlp(pij).squeeze(-1)                            # [B,N,N]


class TypeStructureHead(nn.Module):
    """Type faculty (scaffold): per-cell type-constraint logits [B,N,Tt]."""
    def __init__(self, din, n_types=8, hidden=256):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, n_types))

    def forward(self, feat):
        return self.head(feat)                                      # [B,N,Tt]


class ReductionHead(nn.Module):
    """Reduction faculty (scaffold): chosen reduction-route logits [B, Rroutes] from the pooled hidden."""
    def __init__(self, din, n_routes=6, hidden=256):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, n_routes))

    def forward(self, pooled):
        return self.head(pooled)                                    # [B,Rroutes]


# ----------------------------------------------------------------- the rack
class StructureRack(nn.Module):
    """Router + per-faculty structure heads on a shared encoder. The make-or-break α-emits-structure
    compiler. Frozen-host probe: only this module trains; OLMo stays frozen."""
    def __init__(self, D, K, faculties=FACULTIES, dp=384, heads=6, hidden=384,
                 n_types=8, n_routes=6):
        super().__init__()
        self.encoder = CellEncoder(D, dp, heads)
        din = self.encoder.din
        self.faculties = list(faculties)
        # ROUTER reads a masked-mean of the host hidden over the prompt (faculty-general; needs no
        # CSP-style mentions) → faculty class.
        self.router = nn.Sequential(nn.Linear(D, hidden), nn.GELU(), nn.Linear(hidden, len(faculties)))
        self.heads = nn.ModuleDict({
            "csp": CSPStructureHead(din, K),
            "ising": IsingStructureHead(din),
            "graph": GraphStructureHead(din),
            "typ": TypeStructureHead(din, n_types=n_types),   # key 'typ' (nn.Module reserves '.type')
            "reduction": ReductionHead(din, n_routes=n_routes),
        })

    def featurize(self, v_mean, h, attn_mask):
        return self.encoder(v_mean, h, attn_mask)                   # [B,N,din]

    def route(self, h, attn_mask):
        """Faculty logits from the masked-mean prompt hidden. h [B,T,D], attn_mask [B,T]."""
        m = attn_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1e-6)          # [B,D]
        return self.router(pooled)

    def csp(self, feat):
        return self.heads["csp"](feat)

    def ising(self, feat):
        return self.heads["ising"](feat)
