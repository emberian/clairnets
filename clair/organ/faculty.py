"""clair/organ/faculty.py — the MULTI-FACULTY typed-struct spine + the GRAPH faculty wired END-TO-END.

THE PROBLEM THIS FIXES (the ALPHA_STRUCT CSP-lock). `alpha_struct.StructureRack` has 5 heads
(csp/ising/graph/typ/reduction) + a router, but the live woven (`bank_woven.AlphaStructComposerOrgan`)
only ever calls `build_csp_from_struct` → a literal `CSPState` and reduced-products the CSP bank. The
router never fires; the ising/graph/type heads emit into a void; the composer is hardwired
state_type="csp-domain". So the whole multi-faculty bank is dead weight.

THE FIX (this module). A GENERAL typed `struct_α`:

    build_struct_alpha(faculty, emitted, n, d) -> TypedStruct      # faculty-native, carries state_type

and a DISPATCHING composer (`bank_woven.MultiFacultyComposerOrgan.compose_typed`) that routes by
`state_type` to the right faculty. We wire ONE non-CSP faculty (GRAPH reachability) end-to-end through
its OWN certified reduction (`GraphReach`, sound-by-construction min-plus closure — no checkpoint, like
`clair.graph_organ`'s certified core), with a per-cell readout the SAME zero-init γ channel reads. And
a CROSS-FACULTY composition (`CrossCSPGraph`) whose answer is solvable ONLY by faculty-A→flow→faculty-B
(CSP solves cell colours → those colours decide which candidate edges are ACTIVE → graph reachability
answers), so a CSP-only or graph-only organ provably FAILS it (proven in `prove_multifaculty`).

THE FACULTIES (live, dispatched):
    csp    : CSPState           state_type="csp-domain"     (the existing reduced-product bank)
    graph  : GraphReachState    state_type="graph-reach"    (certified BFS/min-plus closure over cells)
    cross  : CrossState         state_type="cross-csp-graph"(CSP→colours→active-edges→reachability flow)

All graph nodes ARE the cells (the mentioned entities), so α's graph head emits an [N,N] adjacency over
the same cells the CSP head pins, and every faculty reads out as a per-cell [N,K] survival matrix.

Soundness scope is unchanged from notes/soundness.md: the GraphReach closure is sound+complete REL the
emitted edge set (level (A)/(B)); answer-soundness still needs the output check (level (D)). A wrong
emitted graph gives a certified-correct reachability of the WRONG graph — the exact analogue of the
SHUFFLED csp_α gap, now for the graph faculty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import csp as C
from .protocol import Certificate, CSPState, Reduction

LIVE_FACULTIES = ["csp", "graph", "cross"]
FACULTY_IDX = {f: i for i, f in enumerate(LIVE_FACULTIES)}


# ====================================================================== GRAPH faculty (over cells)
@dataclass
class GraphReachState:
    """The GRAPH faculty's native state: a directed graph over the n CELLS + a source cell. The lattice
    element is the per-node REACHABLE flag (a sound UNDER-approximation that GROWS toward the closure,
    exactly clair.graph_organ's certified semantics). `reach` is None until the certified closure runs."""
    n: int
    edges: frozenset             # frozenset[(u, v)] directed edges over cells 0..n-1
    source: int
    target: int = -1             # the queried cell (for the answer / output check); -1 = unset
    reach: tuple = None          # per-node bool tuple after the closure
    state_type: str = field(default="graph-reach", init=False)

    def status(self):
        if self.reach is None:
            return "open"
        return "solved"          # the BFS closure is exact: reachability is always fully decided


def reachable(n, edges, source):
    """Certified BFS/min-plus reachability closure over cells: the set reachable from `source`. Pure set
    arithmetic ⇒ SOUND BY CONSTRUCTION for any edge set (it can only ever mark a node reachable when a
    REAL directed walk source→node exists). Complete: it reaches every truly-reachable node."""
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, []).append(v)
    seen = {source}
    stack = [source]
    while stack:
        x = stack.pop()
        for y in adj.get(x, ()):
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return frozenset(s for s in seen if 0 <= s < n)


class GraphReach(Reduction):
    """The GRAPH faculty as a typed certified reduction: closes a GraphReachState's reachable set from
    the source by the exact min-plus/BFS closure (sound+complete REL the given edge set). Its readout is
    per-cell [n,K]: column 0 = reachable-from-source, column 1 = not — the bit the LM reads at the target
    cell to answer 'can you get from s to t?'. Like clair.graph_organ's certified core, NO neural part is
    needed for soundness; a learned frontier-chooser would only prune work, never change the answer."""
    name = "graph_reach"
    domain = "directed reachability over cells (certified BFS/min-plus closure)"
    state_type = "graph-reach"

    def applies(self, state):
        return isinstance(state, GraphReachState)

    def reduce(self, state: GraphReachState) -> GraphReachState:
        r = reachable(state.n, state.edges, state.source)
        reach = tuple(bool(i in r) for i in range(state.n))
        return GraphReachState(state.n, state.edges, state.source, state.target, reach)

    def certificate(self):
        return Certificate(True, "sound-by-construction",
                           "exact reachability closure REL the given edge set (sound+complete); a wrong "
                           "emitted graph ⇒ certified reach of the WRONG graph (the csp_α-analogue gap)")

    def survival(self, state: GraphReachState, K: int) -> np.ndarray:
        st = state if state.reach is not None else self.reduce(state)
        surv = np.zeros((st.n, K), dtype=np.float32)
        for i in range(st.n):
            surv[i, 0 if st.reach[i] else 1] = 1.0           # col0 = reachable, col1 = not
        return surv


# ====================================================================== CROSS faculty (CSP → graph FLOW)
@dataclass
class CrossState:
    """The CROSS-FACULTY state: a colour-CSP over the cells + a CANDIDATE directed edge set. An edge is
    ACTIVE iff the SOLVED colours of its endpoints match (`edge_rule='eq'`) — so the graph the reachability
    runs on is DETERMINED BY THE CSP SOLUTION. This is the explicit faculty-A→flow→faculty-B coupling that
    makes the routing NECESSARY (a CSP-only organ has the colours but no closure; a graph-only organ has no
    colours so it can't decide which edges are active)."""
    csp: C.CSP                   # the colour CSP (csp_α): unique-solution 2/3-colouring over the cells
    cand_edges: frozenset        # candidate directed edges over cells
    source: int
    target: int = -1
    edge_rule: str = "eq"        # 'eq' ⇒ active iff colour[u]==colour[v]; 'neq' ⇒ active iff differ
    state_type: str = field(default="cross-csp-graph", init=False)


def active_edges(colors, cand_edges, rule="eq"):
    """The FLOW: candidate edges filtered by the CSP solution. `colors[i]` is cell i's solved colour (or
    None if the CSP left it open ⇒ that edge cannot be activated — the honest under-approximation)."""
    out = []
    for (u, v) in cand_edges:
        cu, cv = colors[u], colors[v]
        if cu is None or cv is None:
            continue
        same = (cu == cv)
        if (rule == "eq" and same) or (rule == "neq" and not same):
            out.append((u, v))
    return frozenset(out)


class CrossCSPGraph(Reduction):
    """The CROSS-FACULTY composite reduction: (1) solve the colour-CSP exactly (exact_dedP → singletons
    where determined), (2) FLOW those colours into the active-edge set, (3) run the certified graph
    reachability closure. Sound REL (csp_α, cand_edges): every step is a certified narrowing/closure of
    its own structure; answer-soundness still needs the output check (level (D))."""
    name = "cross_csp_graph"
    domain = "CSP colours → active edges → reachability (faculty-A flow into faculty-B)"
    state_type = "cross-csp-graph"

    def applies(self, state):
        return isinstance(state, CrossState)

    def solve_colors(self, csp: C.CSP):
        ded = C.exact_dedP(csp, csp.full())
        return [(int(next(iter(ded[i]))) if len(ded[i]) == 1 else None) for i in range(csp.n)]

    def reduce(self, state: CrossState) -> GraphReachState:
        colors = self.solve_colors(state.csp)
        edges = active_edges(colors, state.cand_edges, state.edge_rule)
        gs = GraphReachState(state.csp.n, edges, state.source, state.target)
        return GraphReach().reduce(gs)                       # the closed graph state (per-cell reach)

    def certificate(self):
        return Certificate(True, "sound-by-construction",
                           "CSP exact_dedP ∘ colour-gated active edges ∘ certified reachability closure; "
                           "sound REL (csp_α, candidate edges)")

    def survival(self, state: CrossState, K: int) -> np.ndarray:
        return GraphReach().survival(self.reduce(state), K)


# ====================================================================== the general typed struct_α
@dataclass
class TypedStruct:
    """α's emitted structure, GENERAL: a faculty tag + the faculty-native State that carries its own
    `state_type`. The dispatching composer routes on `state.state_type`. This is the de-CSP-lock: csp_α
    was a literal CSPState; struct_α is whichever State the routed faculty consumes."""
    faculty: str
    state: Any                                               # CSPState | GraphReachState | CrossState

    @property
    def state_type(self):
        # CSPState (protocol) carries its state_type on the Reduction, not the dataclass → default it.
        return getattr(self.state, "state_type", "csp-domain")


def build_struct_alpha(faculty: str, emitted, n: int, d: int) -> TypedStruct:
    """The GENERAL typed builder (replaces the CSP-locked build_csp_from_struct in the dispatch path).
    `emitted` is the faculty head's decoded output:
        csp   : a fact list  → CSPState.full(build_csp_from_struct(facts, n, d))    state_type csp-domain
        graph : (edges, source, target) → GraphReachState                            state_type graph-reach
        cross : (facts, cand_edges, source, target[, rule]) → CrossState             state_type cross-csp-graph
    Each carries its native state_type; the composer dispatches on it. No rec['csp'] is read (the emitted
    structure is α's own)."""
    if faculty == "csp":
        from .alpha_struct import build_csp_from_struct
        csp = build_csp_from_struct(list(emitted), n, d)
        return TypedStruct("csp", CSPState.full(csp))
    if faculty == "graph":
        edges, source, target = emitted
        gs = GraphReachState(n, frozenset(map(tuple, edges)), int(source), int(target))
        return TypedStruct("graph", gs)
    if faculty == "cross":
        from .alpha_struct import build_csp_from_struct
        facts, cand_edges, source, target = emitted[0], emitted[1], emitted[2], emitted[3]
        rule = emitted[4] if len(emitted) > 4 else "eq"
        csp = build_csp_from_struct(list(facts), n, d)
        cs = CrossState(csp, frozenset(map(tuple, cand_edges)), int(source), int(target), rule)
        return TypedStruct("cross", cs)
    raise ValueError(f"unknown faculty {faculty!r} (live faculties: {LIVE_FACULTIES})")


# ====================================================================== edge decode (graph head logits → edges)
def decode_edges(edge_logits, vmask, n, thr=0.0):
    """α's graph-head edge logits [N,N] → the emitted directed edge set (logit>thr ⇒ edge i→j), over the
    valid (mentioned) cells, no self-loops. The graph analogue of alpha_struct.decode_structure."""
    import torch
    if torch.is_tensor(edge_logits):
        el = edge_logits.detach().cpu().numpy()
    else:
        el = np.asarray(edge_logits)
    vb = (np.asarray(vmask.detach().cpu()) if hasattr(vmask, "detach") else np.asarray(vmask)) > 0.5
    edges = []
    for i in range(n):
        if not vb[i]:
            continue
        for j in range(n):
            if i == j or not vb[j]:
                continue
            if el[i, j] > thr:
                edges.append((i, j))
    return frozenset(edges)


def edges_to_adj_target(edges, n):
    """Supervision target: dense [n,n] {0,1} adjacency for the graph-head BCE (true edges = 1)."""
    A = np.zeros((n, n), dtype=np.float32)
    for (u, v) in edges:
        if 0 <= u < n and 0 <= v < n:
            A[u, v] = 1.0
    return A
