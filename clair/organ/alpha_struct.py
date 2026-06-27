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

import torch.nn.functional as F

from ..augmented import CellReader

# typed pair-relation vocabulary for the CSP faculty (notes/alpha_struct_design.md §1.2 minimal set)
PAIR_RELS = ["none", "eq", "neq", "lt", "le"]
R_PAIR = len(PAIR_RELS)
REL_IDX = {r: i for i, r in enumerate(PAIR_RELS)}     # none=0, eq=1, neq=2, lt=3, le=4
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


# ===================================================================== struct-IO (the α↔composer bridge)
# These are the FREE-FUNCTION glue between the rack's emitted CSP-head logits and the certified composer
# (notes/alpha_struct_design.md §1.4 / §5.1): decode the typed factor graph, supervise it against the
# generator's free true facts, and compile it into a clair.csp.CSP the certified floor runs on. They are
# the ONLY place α's emitted structure becomes a CSP — rec["csp"] never appears here (the SHUFFLED fix).

def _expand_alldiff(facts):
    """alldiff(scope) → its pairwise ≠-clique (the rack emits binary relations, not the alldiff macro)."""
    import itertools as it
    out = []
    for f in facts:
        f = tuple(f)
        if f[0] == "alldiff":
            out += [("neq", a, b) for a, b in it.combinations(tuple(f[1]), 2)]
        else:
            out.append(f)
    return out


def decode_structure(pin_logits, pair_logits, vmask, n, d=None):
    """One instance's CSP-head logits → the normalized predicted typed factor graph (a fact list).

    pin_logits [N,1+K] (argmax≠0 ⇒ pin(i, v=argmax-1)); pair_logits [N,N,R] (argmax over
    {none,eq,neq,lt,le} per ordered pair). Identical decode semantics to run_alpha_struct_probe /
    run_alpha_struct_search; returns `list(curriculum.norm_facts(...))` (order-insensitive, de-duped).
    `d` (if given) drops out-of-domain pin values (a pinned value v>=d can never hold)."""
    from .. import curriculum as CU
    pin = pin_logits.argmax(-1)
    pair = pair_logits.argmax(-1)
    facts = []
    for i in range(n):
        if float(vmask[i]) > 0.5 and int(pin[i]) != 0:
            v = int(pin[i]) - 1
            if d is None or v < d:
                facts.append(("pin", i, v))
    for i in range(n):
        if float(vmask[i]) < 0.5:
            continue
        for j in range(n):
            if i == j or float(vmask[j]) < 0.5:
                continue
            cls = PAIR_RELS[int(pair[i, j])]
            if cls in ("eq", "neq", "lt", "le"):
                facts.append((cls, i, j))
    return list(CU.norm_facts(facts))


def build_csp_from_struct(facts, n, d):
    """α's emitted facts → a clair.csp.CSP (thin wrapper over curriculum.build_csp). The composer +
    certified floor run on THIS — never on rec["csp"]. alldiff is already binary (decode emits ≠-pairs)."""
    from .. import curriculum as CU
    return CU.build_csp(n, d, _expand_alldiff(list(facts)))


def facts_to_struct_targets(facts_per_inst, n_per_inst, Nmax):
    """Batched supervision targets from the generator's free true facts (the new J0, §2.1):
       pin_tgt  [B,Nmax]        long, 0=no-pin else value+1
       pair_tgt [B,Nmax,Nmax]   long, REL_IDX of the ordered-pair relation (0=none)
    alldiff is expanded to ≠-pairs; eq/neq are written symmetrically; lt/le keep the ordered pair."""
    import torch as _t
    B = len(facts_per_inst)
    pin = _t.zeros(B, Nmax, dtype=_t.long)
    pair = _t.zeros(B, Nmax, Nmax, dtype=_t.long)
    for bi, facts in enumerate(facts_per_inst):
        for f in _expand_alldiff(list(facts)):
            k = f[0]
            if k == "pin":
                _, a, v = f
                if a < Nmax:
                    pin[bi, a] = int(v) + 1
            elif k in ("eq", "neq"):
                a, b = int(f[1]), int(f[2])
                if a < Nmax and b < Nmax:
                    pair[bi, a, b] = REL_IDX[k]; pair[bi, b, a] = REL_IDX[k]
            elif k in ("lt", "le"):
                a, b = int(f[1]), int(f[2])
                if a < Nmax and b < Nmax:
                    pair[bi, a, b] = REL_IDX[k]
            # ternary/variadic (sum/xor/par/rel) are deferred (design §1.2) — not in the minimal vocab
    return pin, pair


def structure_sup_loss(pin_logits, pair_logits, pin_tgt, pair_tgt, vmask, wpos=2.0, wnone=1.0):
    """Masked, SOUNDNESS-ASYMMETRIC cross-entropy of the CSP-head logits vs the true factor graph
    (design §2.1). `wpos` weights the relation/pin classes (recall — a DROPPED relation under-constrains
    the floor); `wnone` weights the none/no-pin class (precision). Over valid (mentioned) cells/pairs."""
    B, Nm, Kp1 = pin_logits.shape
    dev = pin_logits.device
    pin_w = torch.full((Kp1,), wpos, device=dev); pin_w[0] = wnone
    pair_w = torch.full((R_PAIR,), wpos, device=dev); pair_w[0] = wnone
    cellm = vmask.bool()
    lp = F.cross_entropy(pin_logits[cellm], pin_tgt.to(dev)[cellm], weight=pin_w) \
        if cellm.any() else pin_logits.sum() * 0.0
    pm = (vmask[:, :, None] * vmask[:, None, :]).bool()
    eye = torch.eye(Nm, device=dev, dtype=torch.bool)[None].expand(B, Nm, Nm)
    pm = pm & ~eye
    lpr = F.cross_entropy(pair_logits[pm], pair_tgt.to(dev)[pm], weight=pair_w) \
        if pm.any() else pair_logits.sum() * 0.0
    return lp + lpr


def output_check(answer, csp_true, query):
    """The SOUND eval-time output-checker (design §3, layer 3): accept iff `answer` is a real
    solution-value of the TRUE instance at the query cell — answer ∈ exact_dedP(csp_true)[query].
    This is the ONLY licensed reader of the hidden true structure, and it lives OUTSIDE the forward
    (never threaded into the composer). accept ⇒ correct, regardless of what α emitted."""
    from .. import csp as C
    if answer is None or query is None or query >= csp_true.n:
        return False
    ded = C.exact_dedP(csp_true, csp_true.full())
    return int(answer) in ded[query]
