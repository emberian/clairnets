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

import itertools as _it

import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

from ..augmented import CellReader
from ..latent_organ import NLayer            # masked self-attention over the cell set (FINDING 1)

# typed pair-relation vocabulary for the CSP faculty (notes/alpha_struct_design.md §1.2 minimal set)
PAIR_RELS = ["none", "eq", "neq", "lt", "le"]
R_PAIR = len(PAIR_RELS)
REL_IDX = {r: i for i, r in enumerate(PAIR_RELS)}     # none=0, eq=1, neq=2, lt=3, le=4
SYMMETRIC = {"eq", "neq"}            # symmetric relations (supervise both orders, dedup at decode)
DIRECTED = {"lt", "le"}             # directed relations (ordered pair a→b)

# arity-3 TERNARY factor vocabulary (FINDING 2): the binary {eq,neq,lt,le} vocab CANNOT express a
# parity/XOR constraint (a⊕b⊕c=r), so the certified GF2RowSpace faculty (the affine wall) was
# unreachable from α. We add the minimal symmetric arity-3 parity vocab — enough to make an XOR /
# affine (GF(2)) system EXPRESSIBLE. Decoded to curriculum's ('parm', scope, rhs) parity facts.
#   par_even  ≡  a⊕b⊕c == 0   (== curriculum 'xor' / 'par')
#   par_odd   ≡  a⊕b⊕c == 1   (== curriculum 'parm' rhs=1)
# Deferred (noted, not built): arity>3 parity, directed modular 'sum' (a+b=c mod d>2).
TERN_RELS = ["none", "par_even", "par_odd"]
R_TERN = len(TERN_RELS)
TERN_IDX = {r: i for i, r in enumerate(TERN_RELS)}
TERN_RHS = {"par_even": 0, "par_odd": 1}              # the rhs each parity class decodes to

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


# ----------------------------------------------------------------- cell-self-attention relational backbone
class CellSelfAttention(nn.Module):
    """FINDING 1 — the RELATIONAL BACKBONE. The encoder's per-cell feat reads only the PROMPT (each cell
    independently cross-attends the text); the heads then classify each pair (i,j) from two INDEPENDENT
    per-cell summaries, so nothing makes the emitted [N,N] structure GLOBALLY consistent. On long chains
    independent per-pair errors compound (struct-F1 0.78, "nearly-but-not-exactly").

    Here a small stack of MASKED self-attention layers over the [B,N,din] cell SET contextualizes each
    cell by every OTHER cell BEFORE the heads (reuses latent_organ.NLayer — position-free masked attn,
    runs on any N). The pair head's feat_i/feat_j are then cell-contextualized ⇒ the structure can be
    made globally consistent (transitivity along a chain becomes representable). Project din→dbb, run
    n_layers NLayers, project back; the output proj is ZERO-INIT so the backbone is a BITWISE NO-OP at
    init (woven-graft safe) — it only departs from identity once trained."""
    def __init__(self, din, n_layers=3, dbb=512, heads=8, mixer="ffn", mode="wide"):
        super().__init__()
        self.mode = mode
        if mode == "bottleneck":
            self.proj_in = nn.Linear(din, dbb)
            self.layers = nn.ModuleList([NLayer(dbb, heads, mixer) for _ in range(n_layers)])
            self.ln = nn.LayerNorm(dbb)
            self.proj_out = nn.Linear(dbb, din)
            nn.init.zeros_(self.proj_out.weight); nn.init.zeros_(self.proj_out.bias)   # no-op @ init
        else:                                          # "wide": NLayers at FULL din, per-layer ReZero gate
            hh = heads                                 # pick a head count that divides din
            while din % hh:
                hh -= 1
            self.layers = nn.ModuleList([NLayer(din, hh, mixer) for _ in range(n_layers)])
            self.gates = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(n_layers)])

    def forward(self, feat, vmask):                    # feat [B,N,din], vmask [B,N]
        if self.mode == "bottleneck":
            h = self.proj_in(feat)
            for layer in self.layers:
                h = layer(h, vmask)
            return feat + self.proj_out(self.ln(h))    # gated (zero-init) residual ⇒ identity @ init
        h = feat                                       # wide / ReZero: no bottleneck, identity @ init
        for layer, g in zip(self.layers, self.gates):
            h = h + g * (layer(h, vmask) - h)          # zero-init per-layer gate ⇒ no-op @ init
        return h


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


# ----------------------------------------------------------------- CSP head with EDGE message-passing
class CSPStructureHeadMP(nn.Module):
    """FINDING 1 — the VALIDATED fix. CSP head whose pair predictor runs `mp_rounds` of MESSAGE PASSING
    over the [N,N] edge grid: each edge e_ij is refined from its own hidden PLUS the mean of the edges
    OUT of i and INTO j (one hop → transitivity reach), so the edge decisions stop being independent.

    The circuit diagnosis (run_alpha_struct_circuit) disentangled architecture vs representation on the
    frozen host over the OOD long chains: INDEP (the original independent-pair CSPStructureHead) micro-F1
    0.82 / exact 0.13; a pure-LINEAR probe 0.50 (the info IS in the host hidden, just not linearly); MP
    micro-F1 0.90 / exact 0.34, holding 0.88–0.95 at lengths 8–12 where INDEP falls to 0.73–0.82. So the
    long-chain gap is ARCHITECTURE-bound and the fix is edge-consistency — at the EDGE, via message
    passing; the cell-level self-attention backbone (CellSelfAttention) did NOT close it. Drop-in: same
    (pin_logits, pair_logits) signature as CSPStructureHead, so decode/targets/loss are unchanged.
    Output edge layer is ZERO-INIT (predicts 'none' ⇒ no-op @ init, woven-graft safe)."""
    def __init__(self, din, K, hidden=384, pair_hidden=256, mp_rounds=2):
        super().__init__()
        self.K = K
        self.pin = nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, 1 + K))
        self.pl = nn.Linear(din, pair_hidden)
        self.pr = nn.Linear(din, pair_hidden)
        self.edge_in = nn.Sequential(nn.Linear(2 * pair_hidden, pair_hidden), nn.GELU())
        self.mp = nn.ModuleList([nn.Sequential(nn.Linear(3 * pair_hidden, pair_hidden), nn.GELU())
                                 for _ in range(mp_rounds)])
        self.out = nn.Linear(pair_hidden, R_PAIR)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)        # no-op @ init

    def forward(self, feat, vmask=None):               # feat [B,N,din], vmask [B,N]
        B, N, _ = feat.shape
        if vmask is None:
            vmask = (feat.abs().sum(-1) > 0).float()
        pin_logits = self.pin(feat)
        li, rj = self.pl(feat), self.pr(feat)
        ph = li.size(-1)
        e = self.edge_in(torch.cat([li[:, :, None, :].expand(B, N, N, ph),
                                    rj[:, None, :, :].expand(B, N, N, ph)], dim=-1))    # [B,N,N,ph]
        m = vmask[:, None, :, None]
        for layer in self.mp:
            denom = vmask.sum(-1).clamp_min(1.0)[:, None, None]
            agg_out_i = (e * m).sum(2) / denom         # mean_k e[i,k]  (edges OUT of i)
            agg_in_j = (e * vmask[:, :, None, None]).sum(1) / denom      # mean_k e[k,j]  (edges INTO j)
            ai = agg_out_i[:, :, None, :].expand(B, N, N, ph)
            bj = agg_in_j[:, None, :, :].expand(B, N, N, ph)
            e = e + layer(torch.cat([e, ai, bj], dim=-1))               # residual MP update
        return pin_logits, self.out(e)                                  # [B,N,1+K], [B,N,N,R]


# ----------------------------------------------------------------- ternary (arity-3) factor head
class TernaryFactorHead(nn.Module):
    """FINDING 2 — the HYPEREDGE vocabulary. Per UNORDERED triple of cells (i,j,k), emit a typed-relation
    logit over {none, even-parity, odd-parity}. This makes arity-3 XOR / affine (parity) constraints
    EXPRESSIBLE by α — the pair head's {eq,neq,lt,le} vocabulary provably cannot represent a⊕b⊕c=r, so an
    XOR system could never be emitted and the certified GF2RowSpace faculty was unreachable (the affine
    α-expressivity gap). The factor scorer is SYMMETRIC by construction (sum-pools the triple's 1st- and
    2nd-order symmetric functions) so the parity's full permutation symmetry is built in, not learned.
    The output layer is ZERO-INIT (predicts 'none' ⇒ emits nothing ⇒ no-op @ init, woven-graft safe).

    Scoped to arity 3 (proves expressibility + unlocks GF2); arity>3 parity & directed modular sum are
    deferred. Triples are passed in (a bounded candidate set, e.g. all C(N,3)) so cost stays O(#triples)."""
    def __init__(self, din, hidden=256):
        super().__init__()
        self.proj = nn.Linear(din, hidden)
        self.mlp = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, R_TERN))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)     # no-op @ init

    def forward(self, feat, triples):                  # feat [B,N,din], triples Long [T,3]
        if triples.numel() == 0:
            return feat.new_zeros(feat.size(0), 0, R_TERN)
        p = self.proj(feat)                            # [B,N,hidden]
        pi = p[:, triples[:, 0]]; pj = p[:, triples[:, 1]]; pk = p[:, triples[:, 2]]   # [B,T,hidden]
        s1 = pi + pj + pk                              # symmetric 1st-order
        s2 = pi * pj + pj * pk + pi * pk               # symmetric 2nd-order
        return self.mlp(torch.cat([s1, s2], dim=-1))   # [B,T,R_TERN]


def enum_triples(n, device=None):
    """All unordered triples (i<j<k) over n cells, as a Long [C(n,3),3] tensor (the candidate scope set)."""
    tr = list(_it.combinations(range(int(n)), 3))
    if not tr:
        return torch.zeros(0, 3, dtype=torch.long, device=device)
    return torch.tensor(tr, dtype=torch.long, device=device)


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
                 n_types=8, n_routes=6, backbone=False, backbone_layers=3, backbone_mode="wide",
                 ternary=False, pair_head="indep", mp_rounds=2):
        super().__init__()
        self.encoder = CellEncoder(D, dp, heads)
        din = self.encoder.din
        self.faculties = list(faculties)
        self.pair_head = pair_head
        # FINDING 1 — cell-self-attention relational backbone (no-op @ init). Measured NOT to help the
        # long-chain gap (off by default); the validated fix is the EDGE message-passing CSP head below.
        self.backbone = CellSelfAttention(din, n_layers=backbone_layers, mode=backbone_mode) \
            if backbone else None
        # ROUTER reads a masked-mean of the host hidden over the prompt (faculty-general; needs no
        # CSP-style mentions) → faculty class.
        self.router = nn.Sequential(nn.Linear(D, hidden), nn.GELU(), nn.Linear(hidden, len(faculties)))
        # FINDING 1 — CSP pair head: 'mp' = edge message-passing (the validated long-chain fix, default);
        # 'indep' = the original independent-pair head (the F1-0.82/exact-0.13 baseline).
        csp_head = CSPStructureHeadMP(din, K, mp_rounds=mp_rounds) if pair_head == "mp" \
            else CSPStructureHead(din, K)
        self.heads = nn.ModuleDict({
            "csp": csp_head,
            "ising": IsingStructureHead(din),
            "graph": GraphStructureHead(din),
            "typ": TypeStructureHead(din, n_types=n_types),   # key 'typ' (nn.Module reserves '.type')
            "reduction": ReductionHead(din, n_routes=n_routes),
        })
        # FINDING 2 — arity-3 ternary/parity factor head (no-op @ init; on by default).
        self.heads["ternary"] = TernaryFactorHead(din) if ternary else None

    def featurize(self, v_mean, h, attn_mask, vmask=None):
        feat = self.encoder(v_mean, h, attn_mask)                  # [B,N,din]
        if self.backbone is not None:
            if vmask is None:                                     # derive cell-validity from the pad-zero rows
                vmask = (v_mean.abs().sum(-1) > 0).float()
            feat = self.backbone(feat, vmask)                     # cell-contextualized (FINDING 1)
        return feat

    def route(self, h, attn_mask):
        """Faculty logits from the masked-mean prompt hidden. h [B,T,D], attn_mask [B,T]."""
        m = attn_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1e-6)          # [B,D]
        return self.router(pooled)

    def csp(self, feat, vmask=None):
        head = self.heads["csp"]
        if isinstance(head, CSPStructureHeadMP):           # MP needs vmask to mask pad cells in the agg
            return head(feat, vmask)
        return head(feat)

    def ising(self, feat):
        return self.heads["ising"](feat)

    def ternary(self, feat, triples):
        """Arity-3 parity-factor logits [B,T,R_TERN] over the given candidate `triples` (FINDING 2)."""
        assert self.heads["ternary"] is not None, "rack built with ternary=False"
        return self.heads["ternary"](feat, triples)


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


def decode_structure(pin_logits, pair_logits, vmask, n, d=None, tern_logits=None, triples=None):
    """One instance's CSP-head logits → the normalized predicted typed factor graph (a fact list).

    pin_logits [N,1+K] (argmax≠0 ⇒ pin(i, v=argmax-1)); pair_logits [N,N,R] (argmax over
    {none,eq,neq,lt,le} per ordered pair). Identical decode semantics to run_alpha_struct_probe /
    run_alpha_struct_search; returns `list(curriculum.norm_facts(...))` (order-insensitive, de-duped).
    `d` (if given) drops out-of-domain pin values (a pinned value v>=d can never hold).

    FINDING 2: if `tern_logits` [T,R_TERN] (+ its `triples` [T,3]) are given, the argmax-non-'none'
    parity classes are decoded to ('parm', (i,j,k), rhs) facts (round-tripped by build_csp_from_struct
    → curriculum's 'parm' constraint, and by facts_to_gf2 → the GF2RowSpace faculty)."""
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
    if tern_logits is not None and triples is not None and len(triples) > 0:
        tcls = tern_logits.argmax(-1)
        for t in range(len(triples)):
            i, j, k = (int(x) for x in triples[t])
            if float(vmask[i]) < 0.5 or float(vmask[j]) < 0.5 or float(vmask[k]) < 0.5:
                continue
            cls = TERN_RELS[int(tcls[t])]
            if cls != "none":
                facts.append(("parm", (i, j, k), TERN_RHS[cls]))
    return list(CU.norm_facts(facts))


def facts_to_ternary_targets(facts_per_inst, triples, Nmax):
    """Batched arity-3 parity targets aligned to the candidate `triples` [T,3] (FINDING 2 supervision):
       tern_tgt [B,T] long, TERN_IDX of the triple's parity class (0=none).
    Curriculum parity facts come in three shapes that all denote a⊕b⊕c=rhs over a 3-cell scope:
    ('xor',a,b,c) (rhs 0), ('par',scope) (rhs 0), ('parm',scope,rhs). Each is matched to its triple."""
    B = len(facts_per_inst)
    T = len(triples)
    tgt = torch.zeros(B, T, dtype=torch.long)
    tri_index = {tuple(int(x) for x in triples[t]): t for t in range(T)}
    for bi, facts in enumerate(facts_per_inst):
        for f in facts:
            k = f[0]
            if k == "xor":
                scope, rhs = tuple(sorted(f[1:4])), 0
            elif k == "par":
                scope, rhs = tuple(sorted(f[1])), 0
            elif k == "parm":
                scope, rhs = tuple(sorted(f[1])), int(f[2]) % 2
            else:
                continue
            if len(scope) != 3 or any(c >= Nmax for c in scope):
                continue                                  # arity>3 parity deferred (out of this head's scope)
            t = tri_index.get(scope)
            if t is not None:
                tgt[bi, t] = TERN_IDX["par_odd" if rhs else "par_even"]
    return tgt


def ternary_sup_loss(tern_logits, tern_tgt, tri_valid, wpos=4.0, wnone=0.3):
    """Masked soundness-asymmetric CE of the ternary-head logits vs the true arity-3 parity factors over
    VALID triples (all three cells mentioned). `wpos` weights the parity classes (recall — a dropped
    parity under-constrains the GF2 floor), `wnone` the none class (precision)."""
    dev = tern_logits.device
    w = torch.full((R_TERN,), wpos, device=dev); w[0] = wnone
    m = tri_valid.bool()
    if not m.any():
        return tern_logits.sum() * 0.0
    return F.cross_entropy(tern_logits[m], tern_tgt.to(dev)[m], weight=w)


def facts_to_gf2(facts, n):
    """α's emitted parity+pin facts (d=2) → a GF(2) system (A,b) for the certified GF2RowSpace faculty
    (FINDING 2 routing). Each parity ('parm',scope,rhs)/('xor',a,b,c)/('par',scope) is a row of 1s over
    its scope (rhs; xor/par ⇒ 0); each ('pin',i,v) a unit row. The binary {eq,neq,lt,le} vocab emits
    NO parity rows ⇒ the system stays rank-deficient ⇒ GF2 cannot force the derived cells (the gap)."""
    rows, rhs = [], []
    for f in facts:
        k = f[0]
        if k == "parm":
            row = np.zeros(n, np.uint8)
            for c in f[1]:
                row[int(c)] ^= 1
            rows.append(row); rhs.append(int(f[2]) % 2)
        elif k in ("xor",):
            row = np.zeros(n, np.uint8)
            for c in f[1:4]:
                row[int(c)] ^= 1
            rows.append(row); rhs.append(0)
        elif k == "par":
            row = np.zeros(n, np.uint8)
            for c in f[1]:
                row[int(c)] ^= 1
            rows.append(row); rhs.append(0)
        elif k == "pin":
            row = np.zeros(n, np.uint8); row[int(f[1])] = 1
            rows.append(row); rhs.append(int(f[2]) % 2)
    A = np.array(rows, np.uint8) if rows else np.zeros((0, n), np.uint8)
    b = np.array(rhs, np.uint8) if rows else np.zeros((0,), np.uint8)
    return A, b


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
            # arity-3 parity (xor/par/parm) is now handled by facts_to_ternary_targets (FINDING 2);
            # directed modular 'sum' and arity>3 parity / 'rel' tables remain deferred.
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


# ===================================================================== α-as-search proposer (lever A)
# The candidate-structure proposer + the SOUND certified selector, factored out of
# run_alpha_struct_search.build_candidates / verify_structure so the LIVE woven (AlphaStructWoven.
# propose_structures) reuses the *validated* proposer math (runs/alpha_struct_search_nl.json: verifier-
# selection recovered answer-acc +0.11 over greedy). decode_structure_idx is the sampling-capable
# sibling of decode_structure (which argmaxes); candidate 0 (argmax) is bitwise the greedy decode.

def decode_structure_idx(pin_idx, pair_idx, vmask, n, d=None):
    """One instance's CHOSEN class indices → the normalized predicted typed factor graph (a fact list).
    Like `decode_structure` but over GIVEN indices (argmax for greedy, multinomial sample otherwise),
    so it shares the exact decode semantics + the d-domain pin clamp. pin_idx [N], pair_idx [N,N]."""
    from .. import curriculum as CU
    facts = []
    for i in range(n):
        if float(vmask[i]) > 0.5 and int(pin_idx[i]) != 0:
            v = int(pin_idx[i]) - 1
            if d is None or v < d:
                facts.append(("pin", i, v))
    for i in range(n):
        if float(vmask[i]) < 0.5:
            continue
        for j in range(n):
            if i == j or float(vmask[j]) < 0.5:
                continue
            cls = PAIR_RELS[int(pair_idx[i, j])]
            if cls in ("eq", "neq", "lt", "le"):
                facts.append((cls, i, j))
    return list(CU.norm_facts(facts))


def sample_candidates(pin_l, pair_l, vmask, n, K, temp=1.0, d=None, rng=None):
    """Decode K candidate structures for ONE instance from its CSP-head logits. Candidate 0 = GREEDY
    (argmax — bitwise the single-shot decode); 1..K-1 = temperature samples of the pin & pair-relation
    logits. Returns [{facts, conf}] with conf = the model's joint structure log-prob (its confidence,
    the sound selector's tiebreak). Mirrors run_alpha_struct_search.build_candidates, in-woven."""
    t = max(float(temp), 1e-6)
    pin_lp = F.log_softmax(pin_l / t, dim=-1)              # [N,1+K]
    pair_lp = F.log_softmax(pair_l / t, dim=-1)            # [N,N,R]
    pin_p, pair_p = pin_lp.exp(), pair_lp.exp()
    vb = vmask > 0.5
    cells = [i for i in range(n) if bool(vb[i])]
    pairs = [(i, j) for i in cells for j in cells if i != j]

    def conf(pin_idx, pair_idx):
        s = 0.0
        for i in cells:
            s += float(pin_lp[i, int(pin_idx[i])])
        for (i, j) in pairs:
            s += float(pair_lp[i, j, int(pair_idx[i, j])])
        return s

    out = []
    for k in range(max(1, int(K))):
        if k == 0:
            pin_idx = pin_l.argmax(-1)
            pair_idx = pair_l.argmax(-1)
        else:
            pin_idx = torch.multinomial(pin_p, 1, generator=rng).squeeze(-1)
            pflat = pair_p.reshape(-1, pair_p.size(-1))
            pair_idx = torch.multinomial(pflat, 1, generator=rng).squeeze(-1).reshape(pair_p.shape[:2])
        facts = decode_structure_idx(pin_idx, pair_idx, vmask, n, d)
        out.append({"facts": facts, "conf": conf(pin_idx, pair_idx)})
    return out


def solve_query(facts, n, d, query):
    """The SOUND certified selector for α-as-search: compile csp_α = build_csp_from_struct(facts) and
    read the EXACT solver (clair.csp.exact_dedP). Returns (solvable, determines_query, answer). ⊥ (unsat)
    ⇒ every cell empty ⇒ solvable False. determines ⇔ the query cell is a singleton in some solution.
    Uses ONLY the candidate's own structure + the exact solver — no gold (mirrors verify_structure)."""
    from .. import csp as C
    csp = build_csp_from_struct(facts, n, d)
    ded = C.exact_dedP(csp, csp.full())
    solvable = any(len(c) > 0 for c in ded)
    if not solvable or query is None or query >= n:
        return solvable, False, None
    q = ded[query]
    if len(q) == 1:
        return True, True, int(next(iter(q)))
    return True, False, None
