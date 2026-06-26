"""Alternative deduction organs — is attention + geometric-product the right inductive bias?

GLaDOS's Cell = full attention (learned constraint graph) + a channel mixer (FFN / geom).
Here we swap the *context-building* mechanism while keeping the exact lattice-deduction spine
(meet authority, on-policy alpha-target, conflict head) untouched. Each arm is a drop-in for
GLaDOS's Cell: hidden h[B,P,d] -> updated h[B,P,d], operating on the embedded lattice state.

Arms (exploration for accidental wins — see notes/roadmap.md task ladder #1):

  gnn       : message-passing over the EXPLICIT constraint graph. Attention is global and has
              to LEARN which positions constrain each other; for a CSP the graph is known and
              fixed (sudoku: each cell's 20 row/col/box peers). Propagate messages only along
              real exclusion edges. This is the natural CSP inductive bias — the question is
              whether handing it the graph beats making attention discover it.

  ssm       : a lightweight state-space / linear-recurrence mixer (Mamba-lite). Per-channel
              gated linear recurrence scanned across the P positions (forward + backward),
              O(P) instead of O(P^2). Long constraint chains (naked/hidden chains in sudoku)
              are sequential dependencies; a linear-recurrence scan may carry them differently
              than attention's one-hop mixing.

  geom_gnn  : gnn message-passing for context + the geometric-product proposer from glados.py
              (the wedge = pairwise-exclusion operator, notes/architecture.md). Combines the
              explicit-graph bias with the antisymmetric exclusion op — both halves now speak
              the constraint-graph algebra: edges ARE the exclusions.

We reuse GLaDOS's SwiGLU / GeomMix / Attn by import (we do NOT edit glados.py). The Organ model
class mirrors GLaDOS exactly (encode / forward / n_params) so clair.run_organs and the existing
clair.sudoku solve loop drive it unchanged.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .glados import GeomMix, SwiGLU


# --------------------------------------------------------------------------- explicit graph
def sudoku_peers():
    """Return peer adjacency for the 9x9 sudoku constraint graph.

    Each cell p in 0..80 shares a row, column, or 3x3 box with exactly 20 other cells (its
    peers = the cells it must differ from). Returns:
      idx [81,20] long : peer cell-indices per cell (the message-passing neighborhood),
      adj [81,81] float: symmetric 0/1 adjacency (self excluded), for dense aggregation.
    Pure-numpy-checkable (see _selfcheck below): exactly 20 peers, symmetric, no self-loops.
    """
    import numpy as np
    adj = np.zeros((81, 81), np.float32)
    for p in range(81):
        r, c = divmod(p, 9)
        br, bc = 3 * (r // 3), 3 * (c // 3)
        for q in range(81):
            if q == p:
                continue
            qr, qc = divmod(q, 9)
            same_row = qr == r
            same_col = qc == c
            same_box = (br <= qr < br + 3) and (bc <= qc < bc + 3)
            if same_row or same_col or same_box:
                adj[p, q] = 1.0
    idx = np.stack([np.nonzero(adj[p])[0] for p in range(81)])   # [81,20]
    return idx.astype(np.int64), adj


def peer_adjacency(P):
    """Constraint graph for a domain of P positions. Sudoku (P==81) is the known case; other
    domains can extend this. Returns (idx[P,K] long, adj[P,P] float) or (None, None) if unknown
    (the gnn arms then fall back to a fully-connected graph minus self — i.e. learn it)."""
    if P == 81:
        return sudoku_peers()
    return None, None


# --------------------------------------------------------------------------- gnn message-passing
class GNNMix(nn.Module):
    """Message-passing over a fixed constraint graph, drop-in for GLaDOS's Attn (context builder).

    For each cell p with peer set N(p) (|N(p)|=K, the 20 sudoku peers), build a message from each
    peer, aggregate, and project:

        m_q       = W_msg h_q                          (per-peer message)
        e_pq      = a( h_p, m_q )                      (optional attention-weighted gate)
        agg_p     = sum_{q in N(p)} softmax_q(e_pq) m_q     (attn)   OR   mean_q m_q  (mean)
        c_p       = W_out [ agg_p ; W_self h_p ]

    Aggregation runs ONLY over real exclusion edges (the natural CSP bias) instead of all P
    positions. If no explicit graph is registered (unknown domain), we fall back to a dense
    all-pairs (self-masked) graph so the cell still runs — degrading to learned global mixing.
    """
    def __init__(self, d, heads=4, agg="attn"):
        super().__init__()
        self.h = heads
        self.dh = d // heads
        self.agg = agg
        self.msg = nn.Linear(d, d, bias=False)
        self.slf = nn.Linear(d, d, bias=False)
        self.proj = nn.Linear(2 * d, d, bias=False)
        if agg == "attn":
            self.q = nn.Linear(d, d, bias=False)
            self.k = nn.Linear(d, d, bias=False)
        self.peer_idx = None        # [P,K] long buffer, set by register_graph
        self.peer_mask = None       # [P,P] float buffer (dense fallback)

    def register_graph(self, idx, adj, device):
        if idx is not None:
            self.peer_idx = torch.as_tensor(idx, dtype=torch.long, device=device)
        else:
            self.peer_mask = None   # dense fallback built lazily per forward

    def forward(self, x):
        B, P, D = x.shape
        msg = self.msg(x)                                       # [B,P,D] messages
        if self.peer_idx is not None:
            K = self.peer_idx.size(1)
            nbr = msg[:, self.peer_idx, :]                      # [B,P,K,D] gather peer messages
            if self.agg == "attn":
                q = self.q(x).view(B, P, self.h, self.dh)                      # [B,P,h,dh]
                kk = self.k(x)[:, self.peer_idx, :].view(B, P, K, self.h, self.dh)
                e = (q.unsqueeze(2) * kk).sum(-1) / math.sqrt(self.dh)         # [B,P,K,h]
                w = torch.softmax(e, dim=2)                                    # over peers
                nbrh = nbr.view(B, P, K, self.h, self.dh)
                agg = (w.unsqueeze(-1) * nbrh).sum(2).reshape(B, P, D)         # [B,P,D]
            else:
                agg = nbr.mean(2)                                             # mean aggregate
        else:
            # dense fallback: all-pairs mean over non-self positions (learn the graph)
            tot = msg.sum(1, keepdim=True)                                    # [B,1,D]
            agg = (tot - msg) / max(1, P - 1)
        return self.proj(torch.cat([agg, self.slf(x)], dim=-1))


# --------------------------------------------------------------------------- state-space scan
class SSMMix(nn.Module):
    """Mamba-lite per-channel gated linear recurrence over positions, drop-in for Attn.

    Treat the P positions as a sequence and run a diagonal (per-channel) first-order linear
    recurrence with input-dependent forget gates (a selective scan, bidirectional so constraint
    information flows both ways along the grid):

        a_t = sigmoid(W_a x_t)                  per-channel forget/retain gate in (0,1)
        b_t = W_b x_t                           per-channel input
        s_t = a_t ⊙ s_{t-1} + (1 - a_t) ⊙ b_t   forward state-space recurrence (convex => stable)
        s'_t = backward scan of the same        (so position 80 can inform position 0)
        c_p  = W_out [ s_p ; s'_p ]

    O(P) sequential. The convex (1-a, a) combination keeps the recurrence contractive (bounded
    state) which suits monotone narrowing. The scan order is just grid index — a fixed but
    domain-blind 1-D ordering of the constraint positions.
    """
    def __init__(self, d, heads=4):
        super().__init__()
        self.ag = nn.Linear(d, d, bias=True)
        self.bp = nn.Linear(d, d, bias=False)
        self.proj = nn.Linear(2 * d, d, bias=False)

    def _scan(self, a, b, reverse=False):
        # a,b: [B,P,D] -> s: [B,P,D], s_t = a_t*s_{t-1} + (1-a_t)*b_t
        B, P, D = a.shape
        rng = range(P - 1, -1, -1) if reverse else range(P)
        s = torch.zeros(B, D, device=a.device, dtype=a.dtype)
        out = []
        for t in rng:
            s = a[:, t] * s + (1.0 - a[:, t]) * b[:, t]
            out.append(s)
        if reverse:
            out = out[::-1]
        return torch.stack(out, dim=1)

    def forward(self, x):
        a = torch.sigmoid(self.ag(x))
        b = self.bp(x)
        sf = self._scan(a, b, reverse=False)
        sb = self._scan(a, b, reverse=True)
        return self.proj(torch.cat([sf, sb], dim=-1))


# --------------------------------------------------------------------------- the swapped cell
class OrganCell(nn.Module):
    """One recurrent deduction step, mirroring glados.Cell but with an alternative context
    builder (gnn / ssm) replacing attention, and a chosen proposer (FFN or geometric product):

        c = ctx(ln1(h)) ;  h = h + c
        h = h + mix(ln2(h), ln2(h))

    arm -> (context builder, proposer):
        gnn      : (GNNMix, SwiGLU)        explicit graph + FFN proposer
        ssm      : (SSMMix, SwiGLU)        state-space context + FFN proposer
        geom_gnn : (GNNMix, GeomMix wedge) explicit graph + geometric-product proposer
    """
    def __init__(self, d, heads, arm, gnn_agg="attn"):
        super().__init__()
        self.arm = arm
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        if arm in ("gnn", "geom_gnn"):
            self.ctx = GNNMix(d, heads, agg=gnn_agg)
        elif arm == "ssm":
            self.ctx = SSMMix(d, heads)
        else:
            raise ValueError(f"unknown organ arm {arm!r}")
        self.mix = (lambda: GeomMix(d, wedge=True)) if arm == "geom_gnn" else (lambda: SwiGLU(d))
        self.mix = self.mix()

    def forward(self, h):
        c = self.ctx(self.ln1(h))                    # constraint evidence (structured)
        h = h + c
        h = h + self.mix(self.ln2(h), self.ln2(h))   # propose update
        return h


# --------------------------------------------------------------------------- the reasoner
class Organ(nn.Module):
    """GLaDOS-compatible recurrent lattice-deduction reasoner with a swappable deduction organ.

    Identical state/encode/forward/heads contract as glados.GLaDOS (state[B,P,V], given[B,P] ->
    (b_logits[B,P,V], cls_logit[B], deep-supervision list)) so clair.sudoku's solve loop and
    clair.run_organs reuse it directly. The only change is the cell (OrganCell, arm in
    {gnn, ssm, geom_gnn}). For gnn arms, call register_graph(P) after construction to wire in the
    explicit constraint graph (run_organs does this from clair.organs.peer_adjacency)."""
    def __init__(self, arm, P, V, d=128, heads=4, inner=16, ds=16, tied=True, n_layers=1,
                 gnn_agg="attn"):
        super().__init__()
        self.arm, self.P, self.V = arm, P, V
        self.inner, self.ds, self.tied = inner, ds, tied
        self.in_proj = nn.Linear(V + 1, d)
        self.pos = nn.Embedding(P, d)
        n_cells = 1 if tied else inner
        self.cells = nn.ModuleList([OrganCell(d, heads, arm, gnn_agg) for _ in range(n_cells)])
        self.ln_f = nn.LayerNorm(d)
        self.b_head = nn.Linear(d, V)
        self.cls_head = nn.Linear(d, 1)

    def register_graph(self, P=None, device="cpu"):
        """Wire the explicit constraint graph into every gnn cell (no-op for ssm arms)."""
        idx, adj = peer_adjacency(P or self.P)
        for cell in self.cells:
            if isinstance(cell.ctx, GNNMix):
                cell.ctx.register_graph(idx, adj, device)

    def encode(self, state, given):
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
            h = cell(h) + 0.1 * h0
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


def size_for(arm, P, V, target, heads=4, inner=16, ds=16, tied=True, gnn_agg="attn"):
    """Binary-search d (multiple of 2*heads) for non-embed params ~= target. Iso-param arms,
    same protocol as glados.size_for so organ arms are budget-matched to ldt/glados/nowedge."""
    lo, hi, best = 2 * heads, 1024, None
    while lo <= hi:
        d = max(2 * heads, ((lo + hi) // 2 // (2 * heads)) * (2 * heads))
        n = Organ(arm, P, V, d, heads, inner, ds, tied, gnn_agg=gnn_agg).n_params()
        best = (d, n)
        if n < target:
            lo = d + 2 * heads
        else:
            hi = d - 2 * heads
    return best


# --------------------------------------------------------------------------- pure-numpy self-check
def _selfcheck():
    """Validate the sudoku peer-adjacency without torch: 20 peers each, symmetric, no self-loop,
    and the gather index matches the dense adjacency. Run: python -m clair.organs"""
    import numpy as np
    idx, adj = sudoku_peers()
    assert adj.shape == (81, 81), adj.shape
    assert np.array_equal(adj, adj.T), "adjacency not symmetric"
    assert np.all(np.diag(adj) == 0), "self-loop present"
    deg = adj.sum(1)
    assert np.all(deg == 20), f"degree not 20 everywhere: {sorted(set(deg.tolist()))}"
    assert idx.shape == (81, 20), idx.shape
    for p in range(81):
        assert np.array_equal(np.sort(idx[p]), np.nonzero(adj[p])[0]), f"idx mismatch at {p}"
    # spot-check known peers of cell 0 (row0: 1-8, col0: 9,18,...,72, box: 1,2,9,10,11,18,19,20)
    peers0 = set(idx[0].tolist())
    assert 1 in peers0 and 9 in peers0 and 10 in peers0 and 80 not in peers0
    assert len(peers0) == 20
    print("organs self-check OK: 81 cells x 20 peers, symmetric, no self-loops, idx==adj")


if __name__ == "__main__":
    _selfcheck()
