"""clair/organ/reductions.py — THE REDUCTION GRAPH: cross-type ProblemReduction edges + cost routing.

The cheapest generality-multiplier (notes/abstract_machines.md, IDEA 1). The NP faculties we ship are
inter-reducible (Karp); this module encodes those reductions as a TYPED OBJECT (`ProblemReduction`) and
a cost-routed graph, so the organ/LM can learn *solve-by-reduction* + *cross-faculty routing*. NOTHING
here re-derives an encoder: the exact machinery already lives in `clair.ising_organ` (Lucas NP→Ising
encoders/decoders + exact ground-truth checkers + `qubo_to_ising`) and `clair.csp` (`CSPState`,
`exact_dedP`, `solutions`). We only WIRE it as typed cross-type edges.

THE KEY DISTINCTION (vs the composer). `clair.organ.compose.reduced_product` composes reductions
*WITHIN* one state type (CSPState) by meet — it routes inside a problem. A `ProblemReduction` routes
*ACROSS* problem types: it maps an X-instance to a Y-instance of a DIFFERENT faculty, and decodes the
Y-solution back to an X-answer. A solution PATH (X → direct faculty) vs (X → Y → faculty) is a
"modality": same answer, different route. Cross-faculty routing = shortest path over the edges.

THE TYPED OBJECT.
    recognize(x) -> bool        is this edge relevant to instance x? (right src_type + shape)
    reduce(x)    -> y           X-instance → Y-instance (the reduce-X→Y map; reuses ising/CSP encoders)
    decode(y_sol)-> x_answer     Y-solution → X-answer (reuses ising/CSP decoders)
    certificate()-> Certificate  exact / gadget / approximate — what the routing trusts / must gate
    cost(x)      -> float         encoding blow-up (for cheapest-reduction selection)

FIRST EDGES (all EXACT-verified end-to-end against X's OWN exact solver, in `verify_all`):
    MIS ↔ VC ↔ clique     complement set / complement graph (identity blow-up, certified)
    3-SAT → CSP           clause → ternary allowed-tuple over {0,1}   (CSPState, certified)
    2-SAT → CSP           clause → binary allowed-tuple                (CSPState, certified)
    XOR-SAT → CSP         parity clause → allowed-tuple                (CSPState, certified)
    3-SAT → MIS (gadget)  Lucas: m clause-triangles + conflict edges; IS of size m ⇔ SAT
    {MIS, MaxCut, partition, coloring} → Ising   reuse ising_organ.encode_*/decode_* (approximate)
and so 3-SAT → MIS → Ising is a genuine SECOND path to a faculty — the router prefers the certified
CSP edge and falls back to the approximate Ising hub only when no certified route applies.

ROUTING: Dijkstra over the edges to the cheapest faculty that solves the type; tie-break on certificate
strength (a `sound-by-construction` route beats an `approximate` one at equal cost).

Run:  python -m clair.organ.reductions            # exact-verify every edge on N instances + route table
"""
from __future__ import annotations

import heapq
import itertools as it
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import csp as C
from .protocol import Certificate, CSPState


def _IS():
    """Lazy ising_organ import — torch is only needed for the X→Ising edges, so the certified
    CSP / complement-graph edges (and their verification) run with no torch dependency."""
    from .. import ising_organ as IS
    return IS


# ============================================================ X-instance representations
@dataclass(frozen=True)
class Sat:
    """A CNF SAT instance. `clauses` is a tuple of clauses; each clause is a tuple of literals
    (var:int, neg:bool) — neg=True means ¬var. `width` = max literals per clause (3 for 3-SAT,
    2 for 2-SAT). `kind` distinguishes the surface family for routing/recognize."""
    n: int                                   # number of boolean variables
    clauses: tuple                           # tuple[tuple[(var, neg), ...]]
    kind: str = "sat3"                       # "sat3" | "sat2" | "xorsat"

    def satisfies(self, assign) -> bool:
        """Does the boolean assignment (tuple/list over vars) satisfy every clause?"""
        if self.kind == "xorsat":
            for cl in self.clauses:
                par = cl[-1]                 # parity rhs stored as a trailing int (0/1)
                lits = cl[:-1]
                if sum(assign[v] for v, _ in lits) % 2 != par:
                    return False
            return True
        for cl in self.clauses:
            if not any((assign[v] == 1) != neg for v, neg in cl):
                return False
        return True


@dataclass(frozen=True)
class Graph:
    """A simple undirected graph: n vertices 0..n-1, `edges` a frozenset of (u,v) with u<v."""
    n: int
    edges: frozenset                         # frozenset[(u, v)] u < v
    kind: str = "mis"                        # "mis" | "vc" | "clique" — what we are ASKED for

    def complement_edges(self) -> frozenset:
        allp = {(u, v) for u in range(self.n) for v in range(u + 1, self.n)}
        return frozenset(allp - set(self.edges))


@dataclass(frozen=True)
class Partition:
    """A number-partition instance: split the weights into two parts of equal sum (minimise diff)."""
    weights: tuple
    kind: str = "partition"


@dataclass(frozen=True)
class Coloring:
    """A graph k-coloring instance."""
    graph: Graph
    k: int
    kind: str = "coloring"


# ============================================================ the typed cross-type reduction
class ProblemReduction(ABC):
    """One Karp edge as a typed, exact(-or-gated) cross-type map. Distinct from a CSP-domain
    `protocol.Reduction` (which narrows WITHIN one state type): this maps an X-instance of `src_type`
    to a Y-instance of `dst_type` and decodes the Y-solution back to an X-answer."""

    name: str = "reduction"
    src_type: str = ""
    dst_type: str = ""

    @abstractmethod
    def recognize(self, x: Any) -> bool:
        """Is this edge applicable to instance x? (correct src_type + well-formed shape)."""

    @abstractmethod
    def reduce(self, x: Any) -> Any:
        """X-instance → Y-instance (the reduce map). Reuses the ising_organ / csp encoders."""

    @abstractmethod
    def decode(self, y_solution: Any, x: Any = None) -> Any:
        """Y-solution → X-answer. `x` is the source instance (some decoders need its shape)."""

    @abstractmethod
    def certificate(self) -> Certificate:
        """How soundness holds: sound-by-construction (exact/gadget) | approximate (output-checked)."""

    def cost(self, x: Any) -> float:
        """Encoding blow-up (target size / source size), for cheapest-reduction routing. Default 1."""
        return 1.0

    def __repr__(self):
        return f"<{self.name} {self.src_type}->{self.dst_type} {self.certificate().kind}>"


# ============================================================ MIS ↔ VC ↔ clique (complement maps)
class MISToVC(ProblemReduction):
    """MIS → vertex-cover: SAME graph; a maximum independent set S is the complement of a minimum
    vertex cover (VC = V∖S). Identity blow-up, sound by construction."""
    name = "mis_to_vc"; src_type = "mis"; dst_type = "vc"

    def recognize(self, x): return isinstance(x, Graph) and x.kind == "mis"
    def reduce(self, x: Graph) -> Graph: return Graph(x.n, x.edges, kind="vc")

    def decode(self, vc_set, x: Graph = None):
        """A vertex cover C → the independent set V∖C."""
        n = x.n if x is not None else (max(vc_set) + 1 if vc_set else 0)
        return frozenset(range(n)) - frozenset(vc_set)

    def certificate(self):
        return Certificate(True, "sound-by-construction", "VC = V∖IS on the same graph (exact, identity)")


class VCToMIS(ProblemReduction):
    """VC → MIS: SAME graph; min VC ↔ max IS by complement."""
    name = "vc_to_mis"; src_type = "vc"; dst_type = "mis"

    def recognize(self, x): return isinstance(x, Graph) and x.kind == "vc"
    def reduce(self, x: Graph) -> Graph: return Graph(x.n, x.edges, kind="mis")

    def decode(self, is_set, x: Graph = None):
        n = x.n if x is not None else (max(is_set) + 1 if is_set else 0)
        return frozenset(range(n)) - frozenset(is_set)

    def certificate(self):
        return Certificate(True, "sound-by-construction", "min VC = V∖(max IS) (exact, identity)")


class MISToClique(ProblemReduction):
    """MIS → max-clique: an independent set of G is a clique of the COMPLEMENT graph Ḡ. (As used in
    ising_organ.exact_mis.) Complement-graph blow-up, sound by construction."""
    name = "mis_to_clique"; src_type = "mis"; dst_type = "clique"

    def recognize(self, x): return isinstance(x, Graph) and x.kind == "mis"
    def reduce(self, x: Graph) -> Graph: return Graph(x.n, x.complement_edges(), kind="clique")

    def decode(self, clique_set, x: Graph = None):
        """A clique of Ḡ is exactly an independent set of G (same vertex set)."""
        return frozenset(clique_set)

    def certificate(self):
        return Certificate(True, "sound-by-construction", "IS(G) = clique(Ḡ) (exact, complement graph)")


class CliqueToMIS(ProblemReduction):
    """max-clique → MIS: a clique of G is an independent set of Ḡ."""
    name = "clique_to_mis"; src_type = "clique"; dst_type = "mis"

    def recognize(self, x): return isinstance(x, Graph) and x.kind == "clique"
    def reduce(self, x: Graph) -> Graph: return Graph(x.n, x.complement_edges(), kind="mis")

    def decode(self, is_set, x: Graph = None): return frozenset(is_set)

    def certificate(self):
        return Certificate(True, "sound-by-construction", "clique(G) = IS(Ḡ) (exact, complement graph)")


# ============================================================ SAT → CSP (clause → allowed-tuple)
def _clause_scope_allowed(clause, kind):
    """One CNF/XOR clause → an extensional CSP constraint (scope, allowed) over boolean cells {0,1}.
    CNF: scope = the distinct vars; allowed = all assignments EXCEPT the one falsifying every literal.
    XOR: scope = the distinct vars; allowed = assignments whose parity == the clause rhs."""
    if kind == "xorsat":
        par = clause[-1]; lits = clause[:-1]
        vs = tuple(sorted({v for v, _ in lits}))
        pos = {v: i for i, v in enumerate(vs)}
        allowed = frozenset(t for t in it.product((0, 1), repeat=len(vs))
                            if sum(t[pos[v]] for v, _ in lits) % 2 == par)
        return (vs, allowed)
    vs = tuple(sorted({v for v, _ in clause}))
    pos = {v: i for i, v in enumerate(vs)}
    allowed = frozenset(t for t in it.product((0, 1), repeat=len(vs))
                        if any((t[pos[v]] == 1) != neg for v, neg in clause))
    return (vs, allowed)


def sat_to_csp(sat: Sat) -> CSPState:
    """3-SAT / 2-SAT / XOR-SAT → CSPState over n boolean cells, one extensional factor per clause.
    EXACT: the CSP's solution set equals the SAT instance's model set, so the per-cell exact transformer
    (exact_dedP) recovers the forced/free status of every variable. Reuses the shared CSPState spine."""
    cons = tuple(_clause_scope_allowed(cl, sat.kind) for cl in sat.clauses)
    csp = C.CSP(sat.n, 2, cons)
    return CSPState.full(csp)


class Sat3ToCSP(ProblemReduction):
    """3-SAT → CSP. Each clause (a∨b∨c) → a ternary allowed-tuple over {0,1} (all 8 minus the one
    all-false row). Feeds the certified CSP narrowers + the verifier-gated CoreNarrowOrgan."""
    name = "sat3_to_csp"; src_type = "sat3"; dst_type = "csp"

    def recognize(self, x): return isinstance(x, Sat) and x.kind == "sat3"
    def reduce(self, x: Sat) -> CSPState: return sat_to_csp(x)

    def decode(self, assignment, x: Sat = None):
        """A CSP solution (per-cell value tuple) IS the boolean assignment (identity decode)."""
        return tuple(int(v) for v in assignment)

    def certificate(self):
        return Certificate(True, "sound-by-construction",
                           "clause → ternary allowed-tuple; CSP solutions == SAT models (exact)")

    def cost(self, x: Sat) -> float:
        return 1.0 + len(x.clauses) / max(1, x.n)            # m factors over n cells


class Sat2ToCSP(Sat3ToCSP):
    """2-SAT → CSP (binary clauses → pair allowed-tuples). A distinct SOURCE family / representation."""
    name = "sat2_to_csp"; src_type = "sat2"; dst_type = "csp"
    def recognize(self, x): return isinstance(x, Sat) and x.kind == "sat2"


class XorSatToCSP(Sat3ToCSP):
    """XOR-SAT → CSP (parity clauses → allowed-tuples whose XOR matches the rhs). A third source."""
    name = "xorsat_to_csp"; src_type = "xorsat"; dst_type = "csp"
    def recognize(self, x): return isinstance(x, Sat) and x.kind == "xorsat"


# ============================================================ 3-SAT → MIS (Lucas clause-triangle gadget)
def sat3_to_mis_gadget(sat: Sat):
    """Lucas 3-SAT → MIS gadget (arXiv:1302.5843 §4.3, the standard Karp gadget): one node per literal
    occurrence; a TRIANGLE among the (≤3) literals of each clause (pick ≤1 per clause); an edge between
    two literal-nodes in different clauses iff they are CONTRADICTORY (x and ¬x). The formula is
    satisfiable iff this graph has an independent set of size m (= #clauses). Returns (Graph, node_lits)
    where node_lits[i] = (var, neg) for node i."""
    node_lits = []          # (var, neg) per node
    clause_nodes = []       # list of node-index lists, one per clause
    for cl in sat.clauses:
        nodes = []
        for (v, neg) in cl:
            nodes.append(len(node_lits))
            node_lits.append((v, neg))
        clause_nodes.append(nodes)
    edges = set()
    # within-clause triangles
    for nodes in clause_nodes:
        for a, b in it.combinations(nodes, 2):
            edges.add((min(a, b), max(a, b)))
    # cross-clause contradiction edges
    for i in range(len(node_lits)):
        for j in range(i + 1, len(node_lits)):
            vi, ni = node_lits[i]; vj, nj = node_lits[j]
            if vi == vj and ni != nj:
                edges.add((i, j))
    return Graph(len(node_lits), frozenset(edges), kind="mis"), tuple(node_lits)


class Sat3ToMIS(ProblemReduction):
    """3-SAT → MIS via the Lucas clause-triangle gadget. The second path to a faculty: 3-SAT → MIS →
    (Ising). Sound by construction (gadget): IS of size m ⇔ satisfiable, and the chosen literals decode
    to a satisfying assignment (no contradictory pair can be co-selected)."""
    name = "sat3_to_mis"; src_type = "sat3"; dst_type = "mis"

    def recognize(self, x): return isinstance(x, Sat) and x.kind == "sat3"

    def reduce(self, x: Sat) -> Graph:
        g, _ = sat3_to_mis_gadget(x)
        return g

    def decode(self, is_set, x: Sat = None):
        """An IS of size m (one node per clause) → a partial assignment; fill free vars with 0.
        Returns the boolean assignment tuple, or None if the IS does not select one-per-clause."""
        g, node_lits = sat3_to_mis_gadget(x)
        assign = [0] * x.n
        for node in is_set:
            v, neg = node_lits[node]
            assign[v] = 0 if neg else 1          # set the literal TRUE
        return tuple(assign)

    def certificate(self):
        return Certificate(True, "sound-by-construction",
                           "Lucas clause-triangle gadget; IS size m ⇔ SAT (exact gadget)")

    def cost(self, x: Sat) -> float:
        g, _ = sat3_to_mis_gadget(x)
        return 1.0 + g.n / max(1, x.n)           # ~3m nodes vs n vars


# ============================================================ X → Ising (REUSE ising_organ encoders)
class MISToIsing(ProblemReduction):
    """MIS → Ising (Lucas §4.2, ising_organ.encode_mis): H = −Σx_v + B·Σ_edges x_u x_v. Approximate:
    the mean-field annealer is NOT sound-by-construction; soundness is the EXACT output energy-check."""
    name = "mis_to_ising"; src_type = "mis"; dst_type = "ising"

    def recognize(self, x): return isinstance(x, Graph) and x.kind == "mis"

    def reduce(self, x: Graph):
        G = _nx_graph(x)
        J, h, const, meta = _IS().encode_mis(G, B=2.0)
        return {"J": J, "h": h, "const": const, "meta": meta, "G": G}

    def decode(self, spins, x: Graph = None):
        y = self._last
        sel, indep, size = _IS().decode_mis(y["G"], y["meta"], spins)
        return frozenset(sel) if indep else frozenset()

    # decoders need the encoded Y-meta; the router stashes it on reduce. Keep a tiny cache for the
    # standalone path so decode(spins) works after reduce().
    _last = None
    def certificate(self):
        return Certificate(False, "approximate", "Lucas MIS QUBO→Ising; sound only via exact energy-check")
    def cost(self, x: Graph) -> float: return 2.0 + x.n / max(1, x.n)


class MaxCutToIsing(ProblemReduction):
    """MaxCut → Ising (ising_organ.encode_maxcut): the cleanest map, H itself."""
    name = "maxcut_to_ising"; src_type = "maxcut"; dst_type = "ising"
    def recognize(self, x): return isinstance(x, Graph) and x.kind == "maxcut"
    def reduce(self, x: Graph):
        G = _nx_graph(x); J, h, const, meta = _IS().encode_maxcut(G)
        return {"J": J, "h": h, "const": const, "meta": meta, "G": G}
    def decode(self, spins, x: Graph = None):
        y = self._last; return _IS().decode_maxcut(y["G"], y["meta"], spins)
    _last = None
    def certificate(self):
        return Certificate(False, "approximate", "MaxCut→Ising (H itself); sound via exact cut count")
    def cost(self, x): return 2.0


class PartitionToIsing(ProblemReduction):
    """Number-partition → Ising (ising_organ.encode_partition): H=(Σ a_i s_i)²."""
    name = "partition_to_ising"; src_type = "partition"; dst_type = "ising"
    def recognize(self, x): return isinstance(x, Partition)
    def reduce(self, x: Partition):
        J, h, const, meta = _IS().encode_partition(x.weights)
        return {"J": J, "h": h, "const": const, "meta": meta}
    def decode(self, spins, x: Partition = None):
        return _IS().decode_partition(self._last["meta"], spins)
    _last = None
    def certificate(self):
        return Certificate(False, "approximate", "partition→Ising; sound via exact subset-sum diff")
    def cost(self, x): return 2.0


class ColoringToIsing(ProblemReduction):
    """Graph-coloring → Ising (ising_organ.encode_coloring, Lucas §6): k·n spins, H=0 ⇔ proper."""
    name = "coloring_to_ising"; src_type = "coloring"; dst_type = "ising"
    def recognize(self, x): return isinstance(x, Coloring)
    def reduce(self, x: Coloring):
        G = _nx_graph(x.graph); J, h, const, meta = _IS().encode_coloring(G, x.k)
        return {"J": J, "h": h, "const": const, "meta": meta, "G": G, "k": x.k}
    def decode(self, spins, x: Coloring = None):
        y = self._last; coloring, proper, used = _IS().decode_coloring(y["G"], y["meta"], spins)
        return {"coloring": coloring, "proper": proper, "colors": used}
    _last = None
    def certificate(self):
        return Certificate(False, "approximate", "coloring→Ising (k·n spins); sound via proper-check")
    def cost(self, x: Coloring): return 2.0 + x.k


# ============================================================ helpers
def _nx_graph(g: Graph):
    import networkx as nx
    G = nx.Graph(); G.add_nodes_from(range(g.n)); G.add_edges_from(g.edges)
    return G


# ============================================================ Y-solvers (the faculties)
def solve_csp(state: CSPState) -> dict:
    """Solve the CSP faculty: exact per-cell forced values (exact_dedP) + one full solution if any.
    Returns {forced: tuple[frozenset], sat: bool, model: tuple|None}."""
    csp = state.csp
    forced = C.exact_dedP(csp, state.dom)
    sols = C.solutions(csp, state.dom, limit=1)
    sat = len(sols) > 0
    return {"forced": forced, "sat": sat, "model": (sols[0] if sat else None)}


def solve_ising(y: dict, restarts=96, steps=200, seed=0, device="cpu"):
    """Solve the Ising faculty: the mean-field annealing organ → best ±1 config + exact energy."""
    import torch
    J = torch.as_tensor(y["J"], device=device); h = torch.as_tensor(y["h"], device=device)
    s, E, _ = _IS().mean_field_anneal(J, h, restarts=restarts, steps=steps, seed=seed, device=device)
    return s, E + y.get("const", 0.0)


# ============================================================ X's OWN exact solvers (ground truth)
def exact_sat(sat: Sat) -> dict:
    """Independent brute-force SAT oracle over all 2^n assignments: satisfiability, the per-variable
    FORCED bit (same value in every model, else None), and one model. This is X's own solver — the
    reduce→solve→decode loop is checked against THIS, never against the reduction's own target."""
    models = [a for a in it.product((0, 1), repeat=sat.n) if sat.satisfies(a)]
    sat_ok = len(models) > 0
    forced = []
    for v in range(sat.n):
        vals = {a[v] for a in models}
        forced.append(next(iter(vals)) if len(vals) == 1 else None)
    return {"sat": sat_ok, "forced": tuple(forced), "model": (models[0] if sat_ok else None)}


def exact_mis(g: Graph) -> tuple:
    """Independent exact maximum independent set (brute, small n): (size, a max set)."""
    edges = set(g.edges)
    best_size, best = 0, frozenset()
    for r in range(g.n, -1, -1):
        found = False
        for combo in it.combinations(range(g.n), r):
            cs = set(combo)
            if all(not (u in cs and v in cs) for (u, v) in edges):
                if r > best_size:
                    best_size, best = r, frozenset(combo)
                found = True
                break
        if found:
            break
    return best_size, best


def exact_min_vc(g: Graph) -> tuple:
    """Exact minimum vertex cover = V ∖ (max IS)."""
    size, mis = exact_mis(Graph(g.n, g.edges, kind="mis"))
    return g.n - size, frozenset(range(g.n)) - mis


def exact_max_clique(g: Graph) -> tuple:
    """Exact maximum clique = max IS of the complement graph."""
    return exact_mis(Graph(g.n, g.complement_edges(), kind="mis"))


# ============================================================ exact end-to-end verification
def verify_reduction(red: ProblemReduction, x: Any, solver="auto", **skw) -> bool:
    """recognize → reduce → solve-via-Y → decode → compare to X's OWN exact answer. Returns True iff the
    recovered X-answer matches the independent ground truth (the EXACTNESS guarantee per edge)."""
    if not red.recognize(x):
        return False
    y = red.reduce(x)
    # stash Y-meta for the Ising decoders (they need the encode metadata)
    if hasattr(red, "_last"):
        red._last = y if isinstance(y, dict) else None

    if red.dst_type == "csp":
        sol = solve_csp(y)
        # the X-answer we recover: the forced per-variable bits (the determined query is read off these)
        if x.kind == "xorsat":
            truth = exact_xorsat(x)
        else:
            truth = exact_sat(x)
        # satisfiability must match
        if sol["sat"] != truth["sat"]:
            return False
        if not sol["sat"]:
            return True
        # decode a model and check it satisfies X; and the forced bits must match X's forced bits
        model = red.decode(sol["model"], x)
        if not x.satisfies(model):
            return False
        csp_forced = tuple(next(iter(c)) if len(c) == 1 else None for c in sol["forced"])
        return csp_forced == truth["forced"]

    if red.dst_type in ("vc", "mis", "clique"):
        # solve the target graph problem EXACTLY, decode back, check the X-optimum
        if red.dst_type == "vc":
            _, ysol = exact_min_vc(y)
        elif red.dst_type == "clique":
            _, ysol = exact_max_clique(y)
        else:
            _, ysol = exact_mis(y)
        # SAT → MIS (Lucas gadget): X is a SAT instance, not a graph — the MIS SIZE decides
        # satisfiability (size m ⇔ SAT) and the decoded assignment must satisfy the formula.
        if isinstance(x, Sat):
            sat_ok = (len(ysol) == len(x.clauses))
            truth = exact_sat(x)
            if sat_ok != truth["sat"]:
                return False
            if not sat_ok:
                return True
            return x.satisfies(red.decode(ysol, x))
        x_ans = red.decode(ysol, x)
        return _check_graph_answer(x, x_ans)

    if red.dst_type == "ising":
        s, _E = solve_ising(y, **skw)
        x_ans = red.decode(s, x)
        return _check_ising_answer(red, x, x_ans)

    return False


def exact_xorsat(sat: Sat) -> dict:
    return exact_sat(sat)            # Sat.satisfies handles the xorsat kind


def _check_graph_answer(x, x_ans) -> bool:
    """Is the decoded set the correct optimum for X's question (mis/vc/clique)?"""
    if x.kind == "mis":
        size, _ = exact_mis(x)
        edges = set(x.edges)
        indep = all(not (u in x_ans and v in x_ans) for (u, v) in edges)
        return indep and len(x_ans) == size
    if x.kind == "vc":
        size, _ = exact_min_vc(x)
        edges = set(x.edges)
        covers = all(u in x_ans or v in x_ans for (u, v) in edges)
        return covers and len(x_ans) == size
    if x.kind == "clique":
        size, _ = exact_max_clique(x)
        clique = all((min(u, v), max(u, v)) in x.edges for u, v in it.combinations(x_ans, 2))
        return clique and len(x_ans) == size
    return False


def _check_ising_answer(red, x, x_ans) -> bool:
    """Output-check the approximate Ising route against X's exact optimum (the soundness gate)."""
    if isinstance(x, Graph) and x.kind == "mis":
        size, _ = exact_mis(x)
        edges = set(x.edges)
        indep = all(not (u in x_ans and v in x_ans) for (u, v) in edges)
        return indep and len(x_ans) == size
    if isinstance(x, Graph) and x.kind == "maxcut":
        G = _nx_graph(x)
        return abs(x_ans - _IS().exact_maxcut(G)) < 1e-6
    if isinstance(x, Partition):
        return abs(x_ans - _IS().exact_partition(x.weights)) < 1e-6
    if isinstance(x, Coloring):
        return bool(x_ans["proper"]) == exact_colorable(x)
    return False


def exact_colorable(x: Coloring) -> bool:
    G = _nx_graph(x.graph)
    return _IS().exact_chromatic_feasible(G, x.k)


# ============================================================ THE REDUCTION GRAPH + cost routing
# The faculties we ship a SOLVER for (the routing endpoints). CSP = the certified narrowing spine
# (compose.reduced_product); Ising = the universal approximate optimiser (ising_organ).
FACULTIES = ("csp", "ising")

# All edges (the Karp neighbourhood). Order-independent — the router indexes them by src_type.
ALL_EDGES = (
    MISToVC(), VCToMIS(), MISToClique(), CliqueToMIS(),
    Sat3ToCSP(), Sat2ToCSP(), XorSatToCSP(), Sat3ToMIS(),
    MISToIsing(), MaxCutToIsing(), PartitionToIsing(), ColoringToIsing(),
)


@dataclass
class Route:
    path: list                    # list of ProblemReduction edges, X → ... → faculty
    faculty: str                  # the dst_type that solves it
    cost: float                   # summed edge cost
    n_approx: int                 # # of approximate (non-certified) edges on the path (tie-break)

    @property
    def certified(self) -> bool:
        return self.n_approx == 0

    def __repr__(self):
        chain = " → ".join([self.path[0].src_type] + [e.dst_type for e in self.path]) if self.path \
            else self.faculty
        return f"Route[{chain}] cost={self.cost:.2f} {'CERTIFIED' if self.certified else f'{self.n_approx}-approx'}"


class ReductionGraph:
    """The typed reduction graph + Dijkstra cost routing across faculties. `route(x)` finds the CHEAPEST
    path from x's type to a faculty that solves it; ties are broken on certificate strength (a
    sound-by-construction route beats an approximate one). This is cross-faculty routing — the structural
    prior the GNN-NP papers ignore — read straight off the typed edges."""

    def __init__(self, edges=ALL_EDGES, faculties=FACULTIES):
        self.edges = list(edges)
        self.faculties = set(faculties)
        self.by_src = {}
        for e in self.edges:
            self.by_src.setdefault(e.src_type, []).append(e)

    def route(self, x: Any) -> Route | None:
        """Dijkstra over edges from type(x) to the cheapest solving faculty. Priority key =
        (total_cost, total_approx_edges): cost first, certificate strength as the tie-break."""
        start = x.kind
        if start in self.faculties:
            return Route([], start, 0.0, 0)
        # priority queue of (cost, n_approx, counter, node_type, path, current_instance)
        pq = [(0.0, 0, 0, start, [], x)]
        best = {}
        ctr = it.count(1)
        while pq:
            cost, n_approx, _, node, path, inst = heapq.heappop(pq)
            if node in self.faculties:
                return Route(path, node, cost, n_approx)
            key = (node,)
            if key in best and best[key] <= (cost, n_approx):
                continue
            best[key] = (cost, n_approx)
            for e in self.by_src.get(node, []):
                if not e.recognize(inst):
                    continue
                ec = e.cost(inst)
                cert = e.certificate()
                na = n_approx + (0 if cert.sound else 1)
                y = e.reduce(inst)
                # the next "instance" for recognize() on subsequent edges: graph Y-instances carry a
                # kind; CSP/Ising are terminal faculties (no outgoing edges), so a dict placeholder is ok.
                nxt = y if isinstance(y, (Graph, Sat, Partition, Coloring)) else _Terminal(e.dst_type)
                heapq.heappush(pq, (cost + ec, na, next(ctr), e.dst_type, path + [e], nxt))
        return None

    def solve_via_route(self, x: Any, **skw):
        """Route x to a faculty, run the chain reduce·…·solve·decode·…, and return the X-answer."""
        r = self.route(x)
        if r is None:
            return None, None
        inst = x
        encoded = []
        for e in r.path:
            y = e.reduce(inst)
            if hasattr(e, "_last") and isinstance(y, dict):
                e._last = y
            encoded.append((e, y))
            inst = y if isinstance(y, (Graph, Sat, Partition, Coloring)) else inst
        # solve at the faculty
        last_edge, last_y = encoded[-1] if encoded else (None, None)
        if r.faculty == "csp":
            y = encoded[-1][1] if encoded else sat_to_csp(x)
            sol = solve_csp(y)
            ysol = sol["model"]
        elif r.faculty == "ising":
            s, _E = solve_ising(encoded[-1][1], **skw)
            ysol = s
        else:
            return None, r
        # decode back through the path in reverse
        ans = ysol
        for e, _y in reversed(encoded):
            src_inst = x if e is encoded[0][0] else None
            ans = e.decode(ans, _route_src_instance(x, e))
        return ans, r


@dataclass(frozen=True)
class _Terminal:
    kind: str                       # a placeholder "instance" at a faculty node (no outgoing edges)


def _route_src_instance(x, edge):
    """Best-effort source instance for an edge's decode (only the first edge needs the original x)."""
    return x


# ============================================================ exact-verify every edge + route table
def verify_all(n_inst=20, seed=0, verbose=True):
    """Exact end-to-end verification of every reduction edge on N random instances + the routing table.
    Each edge: recognize → reduce → solve-via-Y → decode → recover X's OWN exact answer."""
    rng = np.random.default_rng(seed)
    results = {}

    def report(name, ok, tot):
        results[name] = (ok, tot)
        if verbose:
            print(f"  {name:18s} {ok:3d}/{tot:3d} exact" + ("" if ok == tot else "   <-- MISMATCH"))

    # --- SAT → CSP family (3-SAT, 2-SAT, XOR-SAT) ---
    for kind, red in (("sat3", Sat3ToCSP()), ("sat2", Sat2ToCSP()), ("xorsat", XorSatToCSP())):
        ok = 0
        for _ in range(n_inst):
            x = rand_sat(rng, kind)
            ok += verify_reduction(red, x)
        report(red.name, ok, n_inst)

    # --- 3-SAT → MIS gadget ---
    ok = 0
    red = Sat3ToMIS()
    for _ in range(n_inst):
        x = rand_sat(rng, "sat3", n_lo=3, n_hi=5, m_lo=3, m_hi=7)   # small: gadget MIS is brute-forced
        ok += verify_reduction(red, x)
    report(red.name, ok, n_inst)

    # --- MIS ↔ VC ↔ clique (complement maps) ---
    for red in (MISToVC(), VCToMIS(), MISToClique(), CliqueToMIS()):
        ok = 0
        for _ in range(n_inst):
            g = rand_graph(rng, red.src_type, n_lo=5, n_hi=9)
            ok += verify_reduction(red, g)
        report(red.name, ok, n_inst)

    # --- X → Ising (reuse ising_organ; approximate, output-checked) ---
    try:
        import networkx  # noqa: F401
        for red, gen in ((MISToIsing(), lambda: rand_graph(rng, "mis", 8, 12)),
                         (MaxCutToIsing(), lambda: rand_graph(rng, "maxcut", 8, 11)),
                         (PartitionToIsing(), lambda: rand_partition(rng)),
                         (ColoringToIsing(), lambda: rand_coloring(rng))):
            ok = 0
            for _ in range(max(6, n_inst // 2)):
                ok += verify_reduction(red, gen())
            report(red.name, ok, max(6, n_inst // 2))
    except ImportError:
        if verbose:
            print("  (networkx unavailable — skipping X→Ising edges)")

    # --- routing table ---
    if verbose:
        print("\n  ROUTING (Dijkstra, cheapest faculty; tie-break certified > approximate):")
        G = ReductionGraph()
        for x in (rand_sat(rng, "sat3"), rand_graph(rng, "mis", 7, 9),
                  rand_graph(rng, "vc", 7, 9), rand_graph(rng, "clique", 7, 9),
                  rand_partition(rng), rand_coloring(rng)):
            r = G.route(x)
            print(f"    {x.kind:10s} -> {r}")

    total_ok = sum(o for o, _ in results.values())
    total = sum(t for _, t in results.values())
    if verbose:
        print(f"\n  TOTAL {total_ok}/{total} exact across {len(results)} edges")
    return results


# ============================================================ random instance generators
def rand_sat(rng, kind="sat3", n_lo=4, n_hi=8, m_lo=None, m_hi=None) -> Sat:
    """Random k-SAT / XOR-SAT instance (witness-first → guaranteed satisfiable so the forced-bit check
    is non-trivial). width 3 for sat3/xorsat, 2 for sat2."""
    n = int(rng.integers(n_lo, n_hi + 1))
    width = 2 if kind == "sat2" else 3
    width = min(width, n)
    m = int(rng.integers(m_lo if m_lo else n, (m_hi if m_hi else 2 * n) + 1))
    witness = rng.integers(0, 2, n)
    clauses = []
    for _ in range(m):
        vs = sorted(rng.choice(n, size=width, replace=False).tolist())
        if kind == "xorsat":
            par = int(sum(witness[v] for v in vs) % 2)
            clauses.append(tuple((v, False) for v in vs) + (par,))
        else:
            # build a clause the witness satisfies: pick signs so >=1 literal is true under witness
            lits = []
            sat_one = False
            for v in vs:
                neg = bool(rng.integers(0, 2))
                lits.append((v, neg))
                if (witness[v] == 1) != neg:
                    sat_one = True
            if not sat_one:                       # force the first literal true under the witness
                v0 = vs[0]; lits[0] = (v0, witness[v0] == 0)
            clauses.append(tuple(lits))
    return Sat(n, tuple(clauses), kind=kind)


def rand_graph(rng, kind="mis", n_lo=5, n_hi=9, p=0.4) -> Graph:
    n = int(rng.integers(n_lo, n_hi + 1))
    edges = frozenset((u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p)
    return Graph(n, edges, kind=kind)


def rand_partition(rng, m_lo=8, m_hi=14) -> Partition:
    m = int(rng.integers(m_lo, m_hi + 1))
    return Partition(tuple(int(w) for w in rng.integers(1, 40, size=m)))


def rand_coloring(rng, n_lo=6, n_hi=9, k=3, p=0.35) -> Coloring:
    g = rand_graph(rng, "coloring", n_lo, n_hi, p)
    return Coloring(g, k)


if __name__ == "__main__":
    print("== reductions.py — exact-verify the reduction graph ==\n")
    verify_all(n_inst=24, seed=0)
