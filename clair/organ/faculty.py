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

LIVE_FACULTIES = ["csp", "graph", "cross", "ising", "type", "reduction", "tri"]
FACULTY_IDX = {f: i for i, f in enumerate(LIVE_FACULTIES)}

# the type faculty's answer domain (base monotypes) + the small internal type universe (depth<=1 fun
# types so the propagation has real type values, but every QUERY cell is base-typed so the LM answer is a
# single token). Index order matters: cols 0,1 are the base answer types (int,bool).
TYPE_NAMES = ["int", "bool"]                              # the readout answer domain (query cells)
TYPE_UNIVERSE = ["int", "bool", "fun_ii", "fun_ib", "fun_bi", "fun_bb"]   # 6 <= K=8 internal universe


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


# ====================================================================== ISING faculty (optimization)
@dataclass
class IsingState:
    """The ISING/optimization faculty's native state: pairwise couplings J [n,n] (symmetric, zero-diag)
    + fields h [n] over the n CELLS, a reference cell `source` and a queried cell `target`. The lattice
    element is the ±1 spin assignment that MINIMIZES H = −½ sᵀJs − hᵀs (the universal optimization object
    the LM fundamentally cannot do — narrowing+chaining cannot express argmin over an energy). The answer
    the LM reads is RELATIVE: 'is cell t on the same side as cell s in the optimal assignment?' — a
    spin-flip-invariant question (canonicalized so spin[source]=+1), so it is well-posed despite the global
    ±1 symmetry of a zero-field Ising."""
    n: int
    J: Any                       # [n,n] float (numpy or torch-convertible)
    h: Any                       # [n] float
    source: int
    target: int = -1
    spins: tuple = None          # per-cell ±1 (canonical spin[source]=+1) after the anneal; None until solved
    state_type: str = field(default="ising-spin", init=False)


def maxcut_J(n, edges, w=1.0):
    """An undirected weighted edge set → the MAX-CUT Ising coupling J_ij = −w on edges (ising_organ.
    encode_maxcut form, h=0): the ground state of −½sᵀJs is the maximum cut (cells on opposite sides of
    every heavy edge). The cleanest NP→Ising map; the conflict/rivalry graph the LM cannot optimize."""
    J = np.zeros((n, n), dtype=np.float32)
    for (u, v) in edges:
        if u == v:
            continue
        J[u, v] -= w
        J[v, u] -= w
    return J


def anneal_ising(J, h, restarts=96, steps=180, seed=0):
    """Run the certified-reward Ising organ (mean-field annealing + sound greedy local search, then keep
    the lowest EXACT-energy config) on (J,h). Reuses clair.ising_organ — no re-derivation. Returns
    (spins[n] in ±1 int, energy)."""
    import torch
    from .. import ising_organ as IS
    Jt = torch.as_tensor(np.asarray(J, np.float32))
    ht = torch.as_tensor(np.asarray(h, np.float32))
    s, E, _ = IS.mean_field_anneal(Jt, ht, restarts=restarts, steps=steps, seed=seed, device="cpu")
    return s.detach().cpu().numpy().astype(np.int64), float(E)


class IsingMaxcut(Reduction):
    """The ISING faculty as a typed reduction: anneal (J,h) to a low-energy ±1 assignment, CANONICALIZE so
    spin[source]=+1 (kills the global flip symmetry), and read out per-cell [n,K] survival: col0 =
    same-side-as-source (the 'yes' answer), col1 = opposite. NOT sound-by-construction (NP-hard; mean-field
    has local minima) — soundness is the EXACT energy output-check (level (D)); but on the small,
    determined instances we generate the organ hits the ground state, so the readout is the true optimum."""
    name = "ising_maxcut"
    domain = "min-energy ±1 spin assignment / max-cut (mean-field anneal + sound local search)"
    state_type = "ising-spin"

    def __init__(self, restarts=96, steps=180, seed=0):
        self.restarts, self.steps, self.seed = restarts, steps, seed

    def applies(self, state):
        return isinstance(state, IsingState)

    def reduce(self, state: IsingState) -> IsingState:
        sn, _E = anneal_ising(state.J, state.h, self.restarts, self.steps, self.seed)
        if 0 <= state.source < state.n:
            sn = sn * int(sn[state.source])                  # canonicalize: spin[source] := +1
        return IsingState(state.n, state.J, state.h, state.source, state.target,
                          tuple(int(x) for x in sn))

    def certificate(self):
        return Certificate(False, "approximate",
                           "mean-field annealed Ising min-energy; sound only via the exact energy "
                           "output-check (NP-hard, not sound-by-construction)")

    def survival(self, state: IsingState, K: int) -> np.ndarray:
        st = state if state.spins is not None else self.reduce(state)
        surv = np.zeros((st.n, K), dtype=np.float32)
        for i in range(st.n):
            same = (st.spins[i] == 1)                        # canonical: source is +1
            surv[i, 0 if same else 1] = 1.0                  # col0 = same side as source (yes)
        return surv


# ====================================================================== TYPE faculty (type-constraint propagation)
@dataclass
class TypeState:
    """The TYPE-inference faculty's native state: a finite CSP over a small TYPE universe (cells = the
    named program variables/expressions; values = base monotypes {int,bool}). Type declarations are pins,
    'has the same type as' are eq relations; sound type-constraint propagation == clair.typeinfer.
    certified_step == csp arc-consistency over the typing rules (notes §2 rank 2: the type faculty composes
    on the SHARED LATTICE natively, zero bridge). The query cell is always base-typed so the LM answers a
    single type word. state_type 'type-infer' so the dispatch tags it (and a future type→csp flow can meet
    it with the csp faculty on the same lattice)."""
    csp: Any                     # clair.csp.CSP over the type domain (d = len(TYPE_NAMES))
    n: int
    target: int = -1
    state_type: str = field(default="type-infer", init=False)


class TypeInfer(Reduction):
    """The TYPE faculty: exact per-cell type narrowing (clair.csp.exact_dedP over the compiled type-CSP =
    clair.typeinfer's strongest sound deduction). Readout per-cell [n,K] survival: the determined monotype
    one-hot (col v=1 iff cell is forced to type v). Sound-by-construction REL the emitted type facts (the
    csp_α-analogue: a wrong declared type → certified inference of the wrong typing)."""
    name = "type_infer"
    domain = "Hindley-Milner-style type-constraint propagation over a finite monotype universe"
    state_type = "type-infer"

    def applies(self, state):
        return isinstance(state, TypeState)

    def reduce(self, state: TypeState) -> TypeState:
        return state                                         # narrowing happens in survival via exact_dedP

    def certificate(self):
        return Certificate(True, "sound-by-construction",
                           "exact_dedP over the compiled type-CSP (== typeinfer.exact_dedP); sound+"
                           "complete REL the emitted type facts")

    def survival(self, state: TypeState, K: int) -> np.ndarray:
        ded = C.exact_dedP(state.csp, state.csp.full())
        surv = np.zeros((state.n, K), dtype=np.float32)
        for i in range(min(state.n, len(ded))):
            for v in ded[i]:
                if v < K:
                    surv[i, v] = 1.0
        return surv


# ====================================================================== REDUCTION faculty (the Karp glue)
@dataclass
class ReductionState:
    """The REDUCTION-ROUTER faculty's native state: an X-instance of a faculty with NO DIRECT solver,
    carried with the chosen Karp route. We ship MAX-INDEPENDENT-SET (an OPTIMIZATION over a graph: no
    direct narrowing faculty expresses argmax|IS|): α emits the graph; the reduction head picks the route
    `mis → ising` (organ/reductions.MISToIsing); the composer reduces X→Y, solves at the Y-faculty
    (ising_organ), and DECODES the Ising solution back to a per-cell MIS-membership readout. So a problem
    with no direct faculty is solved by routing through the reduction graph (route_required)."""
    n: int
    edges: frozenset             # undirected graph over cells (u<v)
    source: int                  # unused for MIS (kept for spec-uniformity)
    target: int                  # the queried cell ('is t in the maximum independent set?')
    route_src: str = "mis"       # the source problem type the reduction head selected
    sel: frozenset = None        # the decoded MIS (membership) after solving; None until solved
    state_type: str = field(default="reduction-route", init=False)


class ReductionRoute(Reduction):
    """The REDUCTION faculty: route an X-instance with no direct faculty across the Karp graph to a faculty
    that solves it, then decode back. For MIS we route mis→ising (organ/reductions.ReductionGraph), solve
    the Ising (ising_organ), and decode the ±1 config to the selected set. Readout per-cell [n,K]: col0 =
    'in the maximum independent set' (yes), col1 = not. Approximate (the Ising hub is output-checked);
    on our small instances the organ recovers the exact MIS."""
    name = "reduction_route"
    domain = "cross-faculty Karp routing (mis → ising) for problems with no direct faculty"
    state_type = "reduction-route"

    def __init__(self, restarts=128, steps=180, seed=0):
        self.restarts, self.steps, self.seed = restarts, steps, seed

    def applies(self, state):
        return isinstance(state, ReductionState)

    def reduce(self, state: ReductionState) -> ReductionState:
        from . import reductions as RED
        g = RED.Graph(state.n, frozenset((min(u, v), max(u, v)) for (u, v) in state.edges), kind="mis")
        rg = RED.ReductionGraph()
        sel, _route = rg.solve_via_route(g, restarts=self.restarts, steps=self.steps, seed=self.seed)
        sel = frozenset(int(v) for v in sel) if sel else frozenset()
        return ReductionState(state.n, state.edges, state.source, state.target, state.route_src, sel)

    def route(self, state: ReductionState):
        """Expose the Dijkstra route the faculty took (for tracing): the cross-type Karp path mis→…→faculty."""
        from . import reductions as RED
        g = RED.Graph(state.n, frozenset((min(u, v), max(u, v)) for (u, v) in state.edges), kind="mis")
        return RED.ReductionGraph().route(g)

    def certificate(self):
        return Certificate(False, "approximate",
                           "mis → ising Karp route (organ/reductions); sound only via the exact MIS "
                           "output-check; recovers the exact MIS on small instances")

    def survival(self, state: ReductionState, K: int) -> np.ndarray:
        st = state if state.sel is not None else self.reduce(state)
        surv = np.zeros((st.n, K), dtype=np.float32)
        for i in range(st.n):
            surv[i, 0 if i in st.sel else 1] = 1.0           # col0 = in the MIS (yes)
        return surv


# ====================================================================== TRI faculty (3-hop multiorganic flow)
@dataclass
class TriState:
    """The 3-FACULTY state: a colour-CSP over the cells + a CANDIDATE directed edge set + a source. The
    answer needs faculty-A→flow→faculty-B→flow→faculty-C:
        (A) CSP        solve the unique colouring
        (A→B flow)     colours gate the candidate edges (active iff same colour)        — the csp→graph flow
        (B) GRAPH      the cells reachable from source via active edges = the DISTRICT
        (B→C flow)     the active edges INDUCED on the district = a max-cut conflict graph — the graph→ising flow
        (C) ISING      the optimal 2-way split (max-cut) of the district's road network
    Query: 'is t on the same team as source s in the optimal split of the district?' Solvable ONLY by the
    full A→B→C chain — every single/two-faculty sub-wiring computes the max-cut of the WRONG graph."""
    csp: Any                     # clair.csp.CSP colour CSP (unique 2-colouring)
    cand_edges: frozenset        # candidate directed edges over cells
    source: int
    target: int = -1
    spins: tuple = None
    state_type: str = field(default="tri-csp-graph-ising", init=False)


# the FLOW CHANNELS of the tri faculty — cumulative openings of the two bridge flows. Channel c's survival
# is the max-cut readout under that wiring; the learned per-channel FLOW GATES (zero-init, in the woven)
# pick which flows to open. Necessity: only the full channel (both flows) is correct.
TRI_CHANNELS = ["raw", "color", "color_reach"]               # raw=no flows, +csp→graph, +graph→ising(reach)
TRI_N_CHANNELS = len(TRI_CHANNELS)


def _tri_channel_edges(csp, cand_edges, source, n, channel):
    """The undirected conflict-graph edge set the max-cut runs on, under each wiring channel:
      raw         : ALL candidate edges (no colour flow, no reachability)            — ising-only baseline
      color       : colour-ACTIVE edges over ALL cells (csp→graph flow, no closure)  — 2-faculty baseline
      color_reach : colour-active edges INDUCED on the reachable district (both flows)— the true 3-hop
    Returns (undirected_edges, node_set)."""
    if channel == "raw":
        act = frozenset(cand_edges)
        nodes = frozenset(range(n))
    else:
        colors = CrossCSPGraph().solve_colors(csp)
        act = active_edges(colors, cand_edges, "eq")
        if channel == "color":
            nodes = frozenset(range(n))
        else:                                                # color_reach: restrict to the district
            dist = reachable(n, act, source)
            nodes = dist
            act = frozenset((u, v) for (u, v) in act if u in nodes and v in nodes)
    und = frozenset((min(u, v), max(u, v)) for (u, v) in act if u != v)
    return und, nodes


class TriCSPGraphIsing(Reduction):
    """The 3-hop composite: CSP colours → colour-gated active edges → reachable district → max-cut of the
    district's induced road graph (clair.ising_organ). Sound REL (csp_α, candidate edges): every stage is a
    certified narrowing/closure of its own structure, the Ising is output-checked. Readout per-cell [n,K]:
    col0 = same team as source (yes), col1 = opposite; cells outside the district are 'opposite' by
    convention (they are not in the optimized district)."""
    name = "tri_csp_graph_ising"
    domain = "CSP colours → reachable district → max-cut of the district (3 faculties, 2 learned flows)"
    state_type = "tri-csp-graph-ising"

    def __init__(self, restarts=96, steps=180, seed=0):
        self.restarts, self.steps, self.seed = restarts, steps, seed

    def applies(self, state):
        return isinstance(state, TriState)

    def channel_survival(self, state: TriState, K: int, channel: str) -> np.ndarray:
        """The per-cell max-cut readout under ONE wiring channel (for the learned flow gates)."""
        n = state.csp.n
        und, nodes = _tri_channel_edges(state.csp, state.cand_edges, state.source, n, channel)
        surv = np.zeros((n, K), dtype=np.float32)
        if state.source not in nodes:
            nodes = nodes | {state.source}
        node_list = sorted(nodes)
        loc = {v: i for i, v in enumerate(node_list)}
        m = len(node_list)
        J = maxcut_J(m, [(loc[u], loc[v]) for (u, v) in und if u in loc and v in loc])
        h = np.zeros(m, dtype=np.float32)
        sn, _E = anneal_ising(J, h, self.restarts, self.steps, self.seed)
        si = loc[state.source]
        sn = sn * int(sn[si])                                # canonicalize source := +1
        for v in range(n):
            if v in loc:
                surv[v, 0 if sn[loc[v]] == 1 else 1] = 1.0
            else:
                surv[v, 1] = 1.0                             # outside the district → 'opposite' (no)
        return surv

    def reduce(self, state: TriState) -> TriState:
        sv = self.channel_survival(state, 2, "color_reach")
        spins = tuple(1 if sv[i, 0] > 0.5 else -1 for i in range(state.csp.n))
        return TriState(state.csp, state.cand_edges, state.source, state.target, spins)

    def certificate(self):
        return Certificate(False, "approximate",
                           "CSP exact_dedP ∘ colour-gated edges ∘ reachable district ∘ max-cut Ising; "
                           "sound REL (csp_α, candidate edges), Ising output-checked")

    def survival(self, state: TriState, K: int) -> np.ndarray:
        return self.channel_survival(state, K, "color_reach")


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
    if faculty == "ising":
        J, h, source, target = emitted[0], emitted[1], emitted[2], emitted[3]
        return TypedStruct("ising", IsingState(n, np.asarray(J, np.float32), np.asarray(h, np.float32),
                                               int(source), int(target)))
    if faculty == "type":
        facts, target = emitted[0], emitted[1]
        csp = build_type_csp(list(facts), n)
        return TypedStruct("type", TypeState(csp, n, int(target)))
    if faculty == "reduction":
        edges, source, target = emitted[0], emitted[1], emitted[2]
        und = frozenset((min(int(u), int(v)), max(int(u), int(v))) for (u, v) in edges if u != v)
        return TypedStruct("reduction", ReductionState(n, und, int(source), int(target)))
    if faculty == "tri":
        from .alpha_struct import build_csp_from_struct
        facts, cand_edges, source, target = emitted[0], emitted[1], emitted[2], emitted[3]
        csp = build_csp_from_struct(list(facts), n, 2)
        return TypedStruct("tri", TriState(csp, frozenset(map(tuple, cand_edges)),
                                           int(source), int(target)))
    raise ValueError(f"unknown faculty {faculty!r} (live faculties: {LIVE_FACULTIES})")


def build_type_csp(facts, n):
    """α's emitted type facts (pin/eq over the base monotype domain {int,bool}) → a clair.csp.CSP over the
    type domain. This is the type-faculty's compile path (it reuses the CSP spine — typeinfer's
    certified_step IS csp arc-consistency over the typing rules). pin(i, v): cell i has monotype v;
    eq(i, j): i and j share a type (a type variable unified). The same builder the csp faculty uses, with
    the type domain d = len(TYPE_NAMES)."""
    from .alpha_struct import build_csp_from_struct
    return build_csp_from_struct(list(facts), n, len(TYPE_NAMES))


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
