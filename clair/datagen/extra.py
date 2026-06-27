"""clair/datagen/extra.py — BROADENED domain generators for the GLaDOS reasoning corpus.

The base pipeline (`build.py` + `curriculum`/`hard_tasks`) only covers narrowing-CSP families
(coloring/equality/ordering/arithmetic/alldiff + parity/eqchain/forcedcolor). This module extends
the corpus across the WHOLE sea of organ domains, each with the SAME discipline as the CSP path:

    witness-first / known-answer generator
        -> a non-stilted, skin-style natural-language renderer (reuses render._assemble)
        -> an EXACT verification via the domain's own oracle
        -> an organ-grounded reasoning trace

Domains (each grounded in a real clair organ + its EXACT oracle):
  xor         GF(2) linear / parity systems          oracle: clair.xor_wall.gf2_forced / gf2_rref
  graph       shortest-path / reachability / connect oracle: networkx (clair.graph_organ)
  perm        permutation constraints (seat/rank)    oracle: clair.permgroup.exact_dedP (Régin/Hall + brute)
  typeinfer   HM type of a small typed expression    oracle: clair.typeinfer.algorithm_w (Algorithm W)
  fol         forward-chaining entailment            oracle: clair.fol.forward_chain (least Herbrand model)
  optimize    min-cost assignment / max-cut          oracle: clair.energy_organ.brute_opt / exact brute

plus COMPOSITIONS (multi-organ chaining, arXiv 2507.07207): a sub-result of one domain feeds the
parameters of a second, exact end-to-end, tagged by which/how-many domains compose.

Every record carries a `payload` (the structured problem) so the on-disk corpus is INDEPENDENTLY
re-checkable from the domain oracle alone — `verify_record(rec)` rebuilds and re-derives the label.

The record schema is kept identical to `build._record` (every key present) so new-domain records
coexist with CSP records in one parquet/jsonl and flow through dedup + diversity + write unchanged.
"""
from __future__ import annotations

import itertools as it
import random

import numpy as np

from .. import xor_wall as XW
from .. import permgroup as PG
from .. import typeinfer as TI
from .. import fol as FOL
from .. import energy_organ as EO
from .. import csp as C
from .. import curriculum as CU
from . import skins as SK
from . import render as RND

ABSTAIN = -1
UNDET = "cannot be determined"

# the new domain tags (single-domain) the corpus teaches
DOMAINS = ("xor", "graph", "perm", "typeinfer", "fol", "optimize")


# =========================================================================== shared rendering helpers
_POOLS = ("people", "cities", "nato", "greek", "letters")


def _names(rng, n, pools=_POOLS):
    """n distinct entity surface names from a randomly-chosen pool (naming diversity axis)."""
    key = str(rng.choice(list(pools)))
    pool = list(SK.POOLS[key])
    if n > len(pool):                                   # fall back to numbered nouns for big n
        return [f"item {i + 1}" for i in range(n)], "noun_num"
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool[int(i)] for i in idx], key


def _and_list(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _assemble(clauses, intros, question, rng):
    """Lay clauses out via the base non-stilted assembler (prose/semicolon/bullets/numbered)."""
    intro = str(rng.choice(intros)) if (intros and rng.random() < 0.6) else ""
    return RND._assemble(list(clauses), intro, question, rng)


def _core(domain, family, kind, text, question, entities, values, answer, answer_index,
          determined, trace, payload, scheme, structure, *, n=0, d=0, query=0,
          n_domains=1, compose=""):
    """Build a schema-complete core record (id/split/seed are added by the driver)."""
    return {
        "domain": domain, "family": family, "domain_kind": kind, "skin": domain,
        "n": int(n), "d": int(d), "n_facts": len(payload.get("clauses", [])) or int(n),
        "determined": bool(determined), "text": text, "question": question,
        "answer": answer, "answer_index": int(answer_index), "trace": trace,
        "query": int(query), "entities": list(entities), "values": list(values),
        "facts": [], "cons": [], "payload": payload,
        "naming_scheme": scheme, "structure": structure,
        "n_domains": int(n_domains), "compose": compose,
    }


# =========================================================================== XOR / parity (GF(2)) domain
_XOR_SKINS = [
    ("switch", "switches on a panel", ("switch", "toggle", "relay"), ("off", "on")),
    ("lamp", "lamps wired to parity gates", ("lamp", "bulb", "indicator"), ("dark", "lit")),
    ("circuit", "logic lines on a board", ("line", "wire", "net"), ("low", "high")),
]
_XOR_INTROS = (
    "A panel of {noun}s is wired through parity gates.",
    "Each {noun} is either {v0} or {v1}; the wiring constrains them.",
    "We must recover the state of every {noun} from the parity rules.",
)


def _rows_from(A, b):
    return [([int(j) for j in np.nonzero(A[i])[0]], int(b[i])) for i in range(A.shape[0])]


def gen_xor(rng, cfg):
    n_lo, n_hi = cfg.get("n", (5, 9))
    n = int(rng.integers(n_lo, n_hi + 1))
    band = int(rng.choice(cfg.get("bands", (3, 4, n))))
    mode = "sat" if rng.random() < cfg.get("sat_frac", 0.22) else "bit"
    skin_key, label, nouns, (v0, v1) = _XOR_SKINS[int(rng.integers(len(_XOR_SKINS)))]
    noun = str(rng.choice(nouns))
    values = [v0, v1]

    if mode == "sat":
        sysd = XW.gen_xor_system(rng, n, n_par=int(1.3 * n), band=band, force_unique=True)
        A, b = sysd["A"].copy(), sysd["b"].copy()
        contradicted = rng.random() < 0.5
        if contradicted and A.shape[0]:                  # flip one rhs -> inconsistent system
            b = b.copy(); b[int(rng.integers(A.shape[0]))] ^= 1
        _, _, _, _, consistent = XW.gf2_rref(A, b)
        ans = "yes" if consistent else "no"
        determined = True
        ai = 1 if consistent else 0
        query = -1
    else:
        det_target = rng.random() < cfg.get("det_frac", 0.6)
        sysd = XW.gen_xor_system(rng, n, n_par=int(1.3 * n), band=band,
                                 force_unique=det_target)
        A, b = sysd["A"], sysd["b"]
        forced, rank, consistent = XW.gf2_forced(A, b, n)
        det_cells = [i for i in range(n) if len(forced[i]) == 1]
        und_cells = [i for i in range(n) if len(forced[i]) > 1]
        pool = (det_cells or und_cells) if det_target else (und_cells or det_cells)
        if not pool:
            return None
        query = int(rng.choice(pool))
        determined = len(forced[query]) == 1
        ai = int(next(iter(forced[query]))) if determined else ABSTAIN
        ans = values[ai] if determined else UNDET

    rows = _rows_from(A, b)
    if not rows:
        return None
    ents, scheme = _names(rng, n)

    # --- non-stilted clauses: pins (arity-1) + even/odd-parity statements ---
    clauses = []
    for vs, p in rows:
        if len(vs) == 1:
            templ = rng.choice([f"{{a}} is {values[p]}", f"{{a}} reads {values[p]}",
                                f"{{a}} is set to {values[p]}"])
            clauses.append(str(templ).format(a=ents[vs[0]]))
        else:
            lst = _and_list([ents[j] for j in vs])
            par = "an odd" if p == 1 else "an even"
            templ = rng.choice([
                f"{par} number of {lst} are {values[1]}",
                f"the number of {lst} that are {values[1]} is {'odd' if p else 'even'}",
                f"{lst} together have {'odd' if p else 'even'} parity",
            ])
            clauses.append(str(templ))
    rng.shuffle(clauses)
    if mode == "sat":
        question = str(rng.choice([
            "Can all of these conditions hold at once?",
            "Is there a consistent setting of every " + noun + "?",
            "Are these parity rules satisfiable together?"]))
    else:
        question = str(rng.choice([
            f"Is {ents[query]} {values[0]} or {values[1]}?",
            f"What state must {ents[query]} take?",
            f"Determine the state of {ents[query]}."]))
    text, structure = _assemble(clauses, [s.format(noun=noun, v0=v0, v1=v1) for s in _XOR_INTROS],
                                question, rng)

    trace = _xor_trace(rows, n, ents, values, query, mode, ans)
    payload = {"n": n, "rows": rows, "mode": mode, "query": query, "values": values}
    return _core("xor", f"xor_{mode}", "number", text, question, ents, values, ans, ai,
                 determined, trace, payload, scheme, structure, n=n, d=2, query=max(0, query),
                 **{"n_domains": 1})


def _xor_trace(rows, n, ents, values, query, mode, ans):
    """Organ-grounded GF(2) trace: unit-propagation (single-unknown rows) then a Gaussian-elim closer."""
    A = np.zeros((len(rows), n), np.uint8)
    b = np.zeros(len(rows), np.uint8)
    for i, (vs, p) in enumerate(rows):
        for v in vs:
            A[i, v] ^= 1
        b[i] = p
    known = {}
    lines, changed = [], True
    while changed:
        changed = False
        for vs, p in rows:
            unk = [v for v in vs if v not in known]
            if len(unk) == 1:
                u = unk[0]
                val = p
                for v in vs:
                    if v != u:
                        val ^= known[v]
                known[u] = val
                lines.append(f"{ents[u]} is forced {values[val]}.")
                changed = True
    if mode == "sat":
        _, _, _, _, consistent = XW.gf2_rref(A, b)
        if consistent:
            lines.append("Reducing the parity equations (Gaussian elimination over GF(2)) leaves no "
                         "contradictory row, so a consistent setting exists. Answer: yes.")
        else:
            lines.append("Reducing the parity equations yields a row 0 = 1 — a contradiction, so no "
                         "consistent setting exists. Answer: no.")
        return " ".join(lines)
    forced, _, _ = XW.gf2_forced(A, b, n)
    if len(forced[query]) == 1:
        bit = int(next(iter(forced[query])))
        if query in known:
            lines.append(f"So {ents[query]} must be {values[bit]}.")
        else:
            lines.append(f"Combining all parity equations by elimination over GF(2), {ents[query]} "
                         f"is forced to {values[bit]}.")
    else:
        lines.append(f"After elimination {ents[query]} is still free (it varies across solutions), "
                     f"so its state cannot be determined.")
    return " ".join(lines)


def verify_xor(rec):
    pl = rec["payload"]
    n, rows, mode = pl["n"], pl["rows"], pl["mode"]
    values = pl["values"]
    A = np.zeros((len(rows), n), np.uint8)
    b = np.zeros(len(rows), np.uint8)
    for i, (vs, p) in enumerate(rows):
        for v in vs:
            A[i, v] ^= 1
        b[i] = p
    if mode == "sat":
        _, _, _, _, consistent = XW.gf2_rref(A, b)
        return rec["answer"] == ("yes" if consistent else "no")
    forced, _, consistent = XW.gf2_forced(A, b, n)
    q = pl["query"]
    if len(forced[q]) == 1:
        return rec["determined"] and rec["answer"] == values[int(next(iter(forced[q])))]
    return (not rec["determined"]) and rec["answer"] == UNDET


# =========================================================================== graph reasoning domain
_GRAPH_SKINS = [
    ("road", "towns connected by roads", "town", "road", "travel time", "hours"),
    ("network", "servers linked by cables", "server", "link", "latency", "ms"),
    ("transit", "stations on a transit map", "station", "line", "ride", "minutes"),
]
_GRAPH_INTROS = (
    "A {ent} map connects several {entp} by {edge}s.",
    "We are routing through a network of {entp}.",
    "Here is a map of {entp} and the {edge}s between them.",
)


def gen_graph(rng, cfg):
    import networkx as nx
    n_lo, n_hi = cfg.get("n", (6, 14))
    n = int(rng.integers(n_lo, n_hi + 1))
    mode = str(rng.choice(["dist", "dist", "reach", "connect"]))
    directed = mode != "connect"
    p = float(rng.uniform(*cfg.get("p", (0.18, 0.34))))
    wmax = 1 if mode == "reach" or rng.random() < 0.25 else int(rng.integers(2, 10))
    edges, G = XW_graph(rng, n, p, wmax, directed)
    skin_key, label, ent, edge, costw, unit = _GRAPH_SKINS[int(rng.integers(len(_GRAPH_SKINS)))]
    src, dst, w = edges
    s = 0
    t = int(rng.integers(1, n))
    ents, scheme = _names(rng, n)

    # clauses: one per edge
    clauses = []
    for u, v, ww in zip(src.tolist(), dst.tolist(), w.tolist()):
        if directed:
            templ = rng.choice([
                f"there is a {edge} from {{u}} to {{v}}",
                f"you can go from {{u}} to {{v}}",
                f"a {edge} runs from {{u}} to {{v}}"])
        else:
            templ = rng.choice([
                f"{{u}} and {{v}} are joined by a {edge}",
                f"there is a {edge} between {{u}} and {{v}}"])
        c = str(templ).format(u=ents[u], v=ents[v])
        if wmax > 1:
            c += rng.choice([f" with a {costw} of {ww} {unit}", f" taking {ww} {unit}"])
        clauses.append(c)
    # dedupe undirected duplicate edges in surface (both directions stored)
    if not directed:
        seen, dd = set(), []
        for c, (u, v, ww) in zip(clauses, zip(src.tolist(), dst.tolist(), w.tolist())):
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            dd.append(c)                  # keep the clause for the FIRST direction of each edge
        clauses = dd
    rng.shuffle(clauses)

    if mode == "dist":
        from .. import graph_organ as GO
        d = XW_dist(G, s, n)
        reach = int(d[t]) < GO.INF
        if reach:
            ans = str(int(d[t])); ai = int(d[t]); determined = True
        else:
            ans = "no route exists"; ai = ABSTAIN; determined = False
        question = str(rng.choice([
            f"What is the shortest {costw} from {ents[s]} to {ents[t]}?" if wmax > 1
            else f"What is the fewest {edge}s from {ents[s]} to {ents[t]}?",
            f"How {'far' if wmax>1 else 'many steps'} is {ents[t]} from {ents[s]}?"]))
        trace = _graph_trace(G, s, t, n, ents, mode, costw, unit if wmax > 1 else "step")
    elif mode == "reach":
        from .. import graph_organ as GO
        d = XW_dist(G, s, n)
        reachable = int(d[t]) < GO.INF
        ans = "yes" if reachable else "no"; ai = 1 if reachable else 0; determined = True
        question = str(rng.choice([
            f"Can you reach {ents[t]} starting from {ents[s]}?",
            f"Is {ents[t]} reachable from {ents[s]}?"]))
        trace = _graph_trace(G, s, t, n, ents, mode, costw, "step")
    else:  # connect (undirected)
        connected = nx.has_path(G, s, t)
        ans = "yes" if connected else "no"; ai = 1 if connected else 0; determined = True
        question = str(rng.choice([
            f"Are {ents[s]} and {ents[t]} connected?",
            f"Is there a path between {ents[s]} and {ents[t]}?"]))
        trace = _graph_trace(G, s, t, n, ents, mode, costw, "step")

    text, structure = _assemble(
        clauses, [x.format(ent=ent, entp=ent + "s", edge=edge) for x in _GRAPH_INTROS],
        question, rng)
    payload = {"n": n, "edges": [[int(u), int(v), int(ww)] for u, v, ww in
                                 zip(src.tolist(), dst.tolist(), w.tolist())],
               "directed": directed, "s": s, "t": t, "mode": mode}
    return _core("graph", f"graph_{mode}", "number", text, question, ents,
                 [str(i) for i in range(n)], ans, ai, determined, trace, payload, scheme,
                 structure, n=n, d=n, query=t)


def XW_graph(rng, n, p, wmax, directed):
    from .. import graph_organ as GO
    return GO.gen_graph(rng, n, p, wmax, directed)


def XW_dist(G, s, n):
    from .. import graph_organ as GO
    return GO.exact_dist(G, s, n)


def _graph_trace(G, s, t, n, ents, mode, costw, unit):
    import networkx as nx
    from .. import graph_organ as GO
    if mode == "connect":
        comp = nx.node_connected_component(G, s)
        if t in comp:
            try:
                path = nx.shortest_path(G, s, t)
                return ("Exploring outward from " + ents[s] + ": " +
                        " -> ".join(ents[i] for i in path) +
                        f". A path connects {ents[s]} and {ents[t]}, so the answer is yes.")
            except Exception:
                return f"{ents[s]} and {ents[t]} lie in the same connected component, so the answer is yes."
        return (f"Exploring outward from {ents[s]} never reaches {ents[t]} (it is in a different "
                f"connected component), so the answer is no.")
    # dist / reach: narrate the min-plus settle order (Dijkstra)
    lengths = nx.single_source_dijkstra_path_length(G, s, weight="weight")
    order = sorted(lengths.items(), key=lambda kv: kv[1])
    settled = ", ".join(f"{ents[v]} at {dv}" for v, dv in order[:8] if v != s)
    head = f"Relaxing edges outward from {ents[s]} (distance 0): {settled}."
    if t in lengths:
        if mode == "reach":
            return head + f" {ents[t]} is reached, so yes."
        return head + f" The shortest distance to {ents[t]} settles at {lengths[t]}."
    return (head + f" {ents[t]} is never reached — no relaxation lowers its distance below infinity, "
            f"so {'no' if mode=='reach' else 'no route exists'}.")


def verify_graph(rec):
    import networkx as nx
    from .. import graph_organ as GO
    pl = rec["payload"]
    n, directed = pl["n"], pl["directed"]
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))
    for u, v, w in pl["edges"]:
        G.add_edge(u, v, weight=w)
    s, t, mode = pl["s"], pl["t"], pl["mode"]
    if mode == "dist":
        d = GO.exact_dist(G, s, n)
        if int(d[t]) < GO.INF:
            return rec["answer"] == str(int(d[t]))
        return rec["answer"] == "no route exists"
    if mode == "reach":
        d = GO.exact_dist(G, s, n)
        return rec["answer"] == ("yes" if int(d[t]) < GO.INF else "no")
    return rec["answer"] == ("yes" if nx.has_path(G, s, t) else "no")


# =========================================================================== permutation domain
_PERM_SKINS = [
    ("seat", "people taking numbered seats", "person", "seat", "sits in"),
    ("rank", "runners and their finishing places", "runner", "place", "finishes in"),
    ("shift", "workers assigned to shifts", "worker", "shift", "works"),
]
_PERM_INTROS = (
    "Each {ent} takes a distinct {slot} (a one-to-one assignment).",
    "We must work out which {slot} each {ent} takes.",
    "{entp} are assigned to {slot}s, no two sharing one.",
)


def gen_perm(rng, cfg):
    n_lo, n_hi = cfg.get("n", (4, 7))
    n = int(rng.integers(n_lo, n_hi + 1))
    s = list(rng.permutation(n))                          # the witness permutation
    skin_key, label, ent, slot, verb = _PERM_SKINS[int(rng.integers(len(_PERM_SKINS)))]
    ents, scheme = _names(rng, n)
    slots = [f"{slot} {i + 1}" for i in range(n)]

    # witness-first constraints s satisfies: forbidden images + a few orderings + a few fixes
    forb = {}
    n_forb = int(rng.integers(n, 2 * n + 1))
    for _ in range(n_forb):
        i = int(rng.integers(n)); j = int(rng.integers(n))
        if j != s[i]:
            forb.setdefault(i, set()).add(j)
    order = []
    for _ in range(int(rng.integers(0, n))):
        i, j = (int(x) for x in rng.choice(n, size=2, replace=False))
        if s[i] < s[j]:
            order.append((i, j))
        elif s[j] < s[i]:
            order.append((j, i))
    fixes = {}
    for i in range(n):
        if rng.random() < cfg.get("fix_frac", 0.18):
            fixes[i] = s[i]

    unary = []
    for i in range(n):
        if i in fixes:
            unary.append(frozenset({fixes[i]}))
        else:
            unary.append(frozenset(v for v in range(n) if v not in forb.get(i, set())))
    prob = PG.PermProblem(n, tuple(unary), order=tuple(order))
    exact = PG.exact_dedP(prob)
    if any(len(c) == 0 for c in exact):                  # over-constrained (witness guards this; skip)
        return None
    det_target = rng.random() < cfg.get("det_frac", 0.55)
    det_cells = [i for i in range(n) if len(exact[i]) == 1]
    und_cells = [i for i in range(n) if len(exact[i]) > 1]
    pool = (det_cells or und_cells) if det_target else (und_cells or det_cells)
    if not pool:
        return None
    q = int(rng.choice(pool))
    determined = len(exact[q]) == 1
    if determined:
        ai = int(next(iter(exact[q]))); ans = slots[ai]
    else:
        ai = ABSTAIN; ans = UNDET

    clauses = []
    for i in range(n):
        if i in fixes:
            clauses.append(str(rng.choice([
                f"{ents[i]} {verb} {slots[fixes[i]]}",
                f"{ents[i]} is in {slots[fixes[i]]}"])))
        for j in sorted(forb.get(i, set())):
            clauses.append(str(rng.choice([
                f"{ents[i]} cannot be in {slots[j]}",
                f"{ents[i]} is not in {slots[j]}",
                f"{ents[i]} refuses {slots[j]}"])))
    for (i, j) in order:
        clauses.append(str(rng.choice([
            f"{ents[i]} comes before {ents[j]}",
            f"{ents[i]}'s {slot} is earlier than {ents[j]}'s",
            f"{ents[i]} is ahead of {ents[j]}"])))
    rng.shuffle(clauses)
    question = str(rng.choice([
        f"Which {slot} does {ents[q]} take?",
        f"Where does {ents[q]} end up?",
        f"Determine the {slot} of {ents[q]}."]))
    text, structure = _assemble(
        clauses, [x.format(ent=ent, entp=ent + "s", slot=slot) for x in _PERM_INTROS],
        question, rng)
    trace = _perm_trace(prob, ents, slots, q, determined, ai)
    payload = {"n": n, "unary": [sorted(int(v) for v in u) for u in unary],
               "order": [[int(i), int(j)] for i, j in order], "query": q, "slots": slots}
    return _core("perm", "perm", "ordinal", text, question, ents, slots, ans, ai,
                 determined, trace, payload, scheme, structure, n=n, d=n, query=q)


def _perm_trace(prob, ents, slots, q, determined, ai):
    n = prob.n
    dom, steps = PG.to_fixpoint(PG.certified_step, prob)
    lines = []
    for i in range(n):
        if len(dom[i]) == 1:
            lines.append(f"{ents[i]} can only take {slots[next(iter(dom[i]))]}.")
    exact = PG.exact_dedP(prob)
    if determined:
        if len(dom[q]) == 1:
            lines.append(f"So {ents[q]} must take {slots[ai]}.")
        else:
            lines.append(f"Taking the one-to-one (all-different) rule together with the clues, "
                         f"{ents[q]} is forced to {slots[ai]}.")
    else:
        opts = ", ".join(slots[v] for v in sorted(exact[q]))
        lines.append(f"{ents[q]} could still take any of {{{opts}}}, so it cannot be determined.")
    return " ".join(lines) if lines else f"{ents[q]} is forced by the all-different rule."


def verify_perm(rec):
    pl = rec["payload"]
    n = pl["n"]
    unary = tuple(frozenset(u) for u in pl["unary"])
    order = tuple((i, j) for i, j in pl["order"])
    prob = PG.PermProblem(n, unary, order=order)
    exact = PG.exact_dedP(prob)
    q = pl["query"]
    if len(exact[q]) == 1:
        return rec["determined"] and rec["answer"] == pl["slots"][int(next(iter(exact[q])))]
    return (not rec["determined"]) and rec["answer"] == UNDET


# =========================================================================== type-inference domain
_TYPE_INTROS = (
    "Consider this small expression in a typed language.",
    "Here is an expression from an ML-style language.",
    "We want the type of the expression below.",
)
_TYPE_NAME = {"int": "int", "bool": "bool"}


def _ast_str(ast):
    """Pretty-print an AST in a small ML-ish concrete syntax."""
    h = ast[0]
    if h == "lit":
        return str(ast[1]).lower() if ast[2] == "bool" else str(ast[1])
    if h == "var":
        return ast[1]
    if h in ("add", "sub"):
        op = "+" if h == "add" else "-"
        return f"({_ast_str(ast[1])} {op} {_ast_str(ast[2])})"
    if h == "cmp":
        return f"({_ast_str(ast[1])} < {_ast_str(ast[2])})"
    if h == "if":
        return f"(if {_ast_str(ast[1])} then {_ast_str(ast[2])} else {_ast_str(ast[3])})"
    if h == "pair":
        return f"({_ast_str(ast[1])}, {_ast_str(ast[2])})"
    if h == "fst":
        return f"fst({_ast_str(ast[1])})"
    if h == "snd":
        return f"snd({_ast_str(ast[1])})"
    if h == "lam":
        return f"(\\{ast[1]} -> {_ast_str(ast[2])})"
    if h == "app":
        return f"({_ast_str(ast[1])} {_ast_str(ast[2])})"
    if h == "let":
        return f"(let {ast[1]} = {_ast_str(ast[2])} in {_ast_str(ast[3])})"
    return "?"


def _ti_universe(cfg):
    return TI.Universe(max_depth=cfg.get("max_depth", 1), ctors=("fun", "pair"))


def gen_typeinfer(rng, cfg):
    U = _ti_universe(cfg)
    pyr = random.Random(int(rng.integers(0, 2**62)))
    comp = TI.rand_problem(pyr, U, budget=cfg.get("budget", 3),
                           max_nodes=cfg.get("max_nodes", 14))
    ast = comp.ast
    ptype, ok = TI.algorithm_w(ast)
    if not ok:
        return None
    ground = TI.ground_principal(ptype, U)
    determined = len(ground) == 1
    if determined:
        ty = next(iter(ground))
        ans = TI.type_str(ty)
        ai = U.idx.get(ty, ABSTAIN)
    else:
        ans = UNDET                                      # polymorphic / free type variable
        ai = ABSTAIN
    code = _ast_str(ast)
    expr_clause = f"the expression {code}"
    question = str(rng.choice([
        f"What type does {code} have?",
        f"Infer the type of {code}.",
        f"What is the type of {code}?"]))
    text, structure = _assemble([expr_clause], _TYPE_INTROS, question, rng)
    trace = _type_trace(ast, ptype, determined, ans)
    payload = {"ast": _ast_to_json(ast), "max_depth": cfg.get("max_depth", 1)}
    vals = [TI.type_str(t) for t in U.types]
    return _core("typeinfer", "typeinfer", "number", text, question,
                 [code], vals, ans, ai, determined, trace, payload, "code", structure,
                 n=TI.size(ast), d=U.T, query=0)


def _type_trace(ast, ptype, determined, ans):
    """A readable typing derivation (the unification narrowing the type organ performs)."""
    lines = []

    def rec(node):
        h = node[0]
        if h == "lit":
            t = node[2]
            art = "an" if t == "int" else "a"
            lines.append(f"{_ast_str(node)} is {art} {t} literal, so it has type {t}.")
            return
        for c in node[1:]:
            if isinstance(c, tuple):
                rec(c)
        if h in ("add", "sub"):
            lines.append(f"{_ast_str(node)}: + and - require int operands and yield int.")
        elif h == "cmp":
            lines.append(f"{_ast_str(node)}: a comparison of ints yields bool.")
        elif h == "if":
            lines.append(f"{_ast_str(node)}: the guard is bool and both branches share a type.")
        elif h == "pair":
            lines.append(f"{_ast_str(node)}: a pair has the product type of its two parts.")
        elif h in ("fst", "snd"):
            lines.append(f"{_ast_str(node)}: projects the {'first' if h=='fst' else 'second'} "
                         f"component of a pair.")
        elif h == "lam":
            lines.append(f"{_ast_str(node)}: a function from its parameter's type to its body's.")
        elif h == "app":
            lines.append(f"{_ast_str(node)}: applying a function unifies its argument type.")
        elif h == "let":
            lines.append(f"{_ast_str(node)}: binds {node[1]} and types the body.")

    rec(ast)
    if determined:
        lines.append(f"Unifying these constraints, the whole expression has type {ans}.")
    else:
        lines.append(f"The type variables are not all pinned down — the expression is polymorphic, "
                     f"so its type cannot be determined to a single monotype.")
    return " ".join(lines)


def _ast_to_json(ast):
    if not isinstance(ast, tuple):
        return ast
    return [ast[0]] + [_ast_to_json(c) for c in ast[1:]]


def _ast_from_json(j):
    if not isinstance(j, list):
        return j
    return tuple([j[0]] + [_ast_from_json(c) for c in j[1:]])


def verify_typeinfer(rec):
    pl = rec["payload"]
    U = TI.Universe(max_depth=pl.get("max_depth", 1), ctors=("fun", "pair"))
    ast = _ast_from_json(pl["ast"])
    ptype, ok = TI.algorithm_w(ast)
    if not ok:
        return False
    ground = TI.ground_principal(ptype, U)
    if len(ground) == 1:
        return rec["determined"] and rec["answer"] == TI.type_str(next(iter(ground)))
    return (not rec["determined"]) and rec["answer"] == UNDET


# =========================================================================== FOL / entailment domain
_FOL_LABEL_ANS = {"entail": "yes", "contradict": "no", "unknown": UNDET}
_FOL_INTROS = (
    "A small knowledge base gives some rules and facts.",
    "From the rules and facts below, decide the question.",
    "Reason forward from these rules and facts.",
    "Here is what we know; deduce the answer.",
)
# per-instance SURFACE synonyms (presentation only — payload keeps the canonical predicate names, so
# verification is unaffected). Rotating these per record is what breaks FOL's boilerplate-rule collapse.
_FOL_BIN_SURF = {
    "link": ["links to", "feeds into", "connects to", "points to", "leads to"],
    "reach": ["can reach", "reaches", "has a route to"],
    "near": ["is near", "is close to", "sits by"],
    "likes": ["likes", "admires", "trusts"],
    "above": ["is above", "is over", "outranks"],
}
_FOL_UN_SURF = {           # unary adjective synonyms
    "red": ["red", "crimson"], "round": ["round", "circular"], "cold": ["cold", "chilly"],
    "kind": ["kind", "gentle"], "heavy": ["heavy", "weighty"], "quiet": ["quiet", "hushed"],
    "calm": ["calm", "at ease"], "trouble": ["in trouble", "troubled"],
    "reach": ["reachable"],
}


def _fol_surface_maps(rng, ents):
    names, _ = _names(rng, len(ents))
    ent_map = {e: names[i] for i, e in enumerate(ents)}
    bin_map = {p: str(rng.choice(v)) for p, v in _FOL_BIN_SURF.items()}
    un_map = {p: str(rng.choice(v)) for p, v in _FOL_UN_SURF.items()}
    return ent_map, bin_map, un_map


def _fol_render(lit, ent_map, bin_map, un_map):
    a = [ent_map.get(x, x) for x in lit.args]
    if len(a) == 1:
        base = f"{a[0]} is {un_map.get(lit.pred, lit.pred)}"
    else:
        verb = bin_map.get(lit.pred, lit.pred)
        base = f"{a[0]} {verb} {a[1]}"
    return ("it is not the case that " if lit.neg else "") + base


def gen_fol(rng, cfg):
    depth = int(rng.integers(*cfg.get("depth", (1, 5))))
    label = str(rng.choice(["entail", "contradict", "unknown"]))
    try:
        p = FOL.gen_problem(rng, label=label, depth=depth if label == "entail" else None)
    except RuntimeError:
        return None
    ents = [str(e) for e in p.entities]
    em, bm, um = _fol_surface_maps(rng, ents)
    rl = lambda l: _fol_render(l, em, bm, um)
    conn = str(rng.choice(["If {b}, then {h}.", "Whenever {b}, {h}.",
                           "{h}, provided {b}.", "Given {b}, it follows that {h}."]))
    rules = " ".join(conn.format(b=" and ".join(rl(b) for b in r.body), h=rl(r.head))
                     for r in p.rules)
    facts = [rl(f).capitalize() + "." for f in p.facts]
    rng.shuffle(facts)
    body_clauses = [rules.strip()] + facts
    qtext = rl(p.query)
    question = str(rng.choice([
        f"Is it true that {qtext}?",
        f"Does it follow that {qtext}?",
        f"Can we conclude that {qtext}?"]))
    text, structure = _assemble([c.rstrip(".") for c in body_clauses], _FOL_INTROS, question, rng)
    ans = _FOL_LABEL_ANS[p.label]
    determined = p.label != "unknown"
    ai = {"yes": 1, "no": 0}.get(ans, ABSTAIN)
    trace = _fol_trace(p, qtext, rl(FOL.neg_of(p.query)))
    payload = {"rules": [_rule_to_json(r) for r in p.rules],
               "facts": [_lit_to_json(f) for f in p.facts],
               "query": _lit_to_json(p.query)}
    surf_ents = [em[e] for e in ents]
    return _core("fol", "fol", "number", text, question, surf_ents, ["no", "yes"], ans, ai,
                 determined, trace, payload, "rules", structure, n=len(ents), d=2, query=0)


def _lit_to_json(l):
    return [bool(l.neg), l.pred, list(l.args)]


def _lit_from_json(j):
    return FOL.Lit(bool(j[0]), j[1], tuple(j[2]))


def _rule_to_json(r):
    return [[_lit_to_json(b) for b in r.body], _lit_to_json(r.head)]


def _rule_from_json(j):
    return FOL.Rule(tuple(_lit_from_json(b) for b in j[0]), _lit_from_json(j[1]))


def _fol_trace(p, qtext, neg_qtext):
    closure = FOL.forward_chain(p.facts, p.rules)
    label = FOL.label_query(closure, p.query)
    if label == "entail":
        depths = FOL.proof_depths(p.facts, p.rules, closure)
        d = depths.get(p.query, 1)
        return (f"Forward-chaining the rules over the facts builds the closure. {qtext} "
                f"is derived after {d} rule application(s), so the answer is yes.")
    if label == "contradict":
        return (f"Forward-chaining derives the negation of the query ({neg_qtext}), so the query is "
                f"false — the answer is no.")
    return (f"Forward-chaining to the least Herbrand model derives neither {qtext} nor its negation, "
            f"so under the open-world reading it cannot be determined.")


def verify_fol(rec):
    pl = rec["payload"]
    rules = [_rule_from_json(r) for r in pl["rules"]]
    facts = [_lit_from_json(f) for f in pl["facts"]]
    query = _lit_from_json(pl["query"])
    closure = FOL.forward_chain(facts, rules)
    label = FOL.label_query(closure, query)
    return rec["answer"] == _FOL_LABEL_ANS[label]


# =========================================================================== optimization domain
_OPT_INTROS = (
    "We want the cheapest valid assignment.",
    "Minimize the total cost subject to the constraints.",
    "Find the best assignment under the rules below.",
)
_CUT_INTROS = (
    "Split the group into two teams to cut as many links as possible.",
    "We partition the items into two sides to maximize crossing links.",
)


def gen_optimize(rng, cfg):
    if rng.random() < cfg.get("maxcut_frac", 0.45):
        return _gen_maxcut(rng, cfg)
    return _gen_mincost(rng, cfg)


def _gen_mincost(rng, cfg):
    rel = str(rng.choice(["coloring", "equality", "ordering"]))
    for _ in range(40):
        cn, cd, kind, facts, _s = CU.GENERATORS[rel](rng)
        csp = CU.build_csp(cn, cd, facts)
        sols = C.solutions(csp, limit=64)
        if len(sols) >= 2:
            break
    else:
        return None
    cost = rng.integers(1, 9, size=(cn, cd)).astype(np.int64)
    opt_cost, _, best = EO.brute_opt(csp, cost)
    ents, scheme = _names(rng, cn)
    vnames = CU.value_names(kind, cd)
    clauses = []
    for f in facts:
        clauses.append(_csp_fact_clause(f, ents, vnames, rng))
    for i in range(cn):
        for v in range(cd):
            if rng.random() < 0.6:
                clauses.append(f"assigning {ents[i]} the value {vnames[v]} costs {int(cost[i, v])}")
    rng.shuffle(clauses)
    question = str(rng.choice([
        "What is the smallest total cost of a valid assignment?",
        "What is the minimum total cost?",
        "What does the cheapest valid assignment cost?"]))
    text, structure = _assemble(clauses, _OPT_INTROS, question, rng)
    ans = str(int(opt_cost))
    trace = (f"Among the valid assignments, the cheapest sets "
             + ", ".join(f"{ents[i]}={vnames[best[i]]}" for i in range(cn))
             + f", for a total cost of {int(opt_cost)}.")
    payload = {"kind": "mincost",
               "cons": [[[int(c) for c in sc], sorted([[int(x) for x in t] for t in al])]
                        for sc, al in csp.cons],
               "n": cn, "d": cd, "cost": cost.tolist()}
    return _core("optimize", "opt_mincost", "number", text, question, ents, vnames,
                 ans, int(opt_cost), True, trace, payload, scheme, structure,
                 n=cn, d=cd, query=0)


def _gen_maxcut(rng, cfg):
    n = int(rng.integers(*cfg.get("cut_n", (5, 11))))
    p = float(rng.uniform(0.3, 0.6))
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
    if not edges:
        edges = [(0, 1)]
    ents, scheme = _names(rng, n)
    best_cut, best_part = _brute_maxcut(n, edges)
    clauses = []
    for (i, j) in edges:
        clauses.append(str(rng.choice([
            f"{ents[i]} and {ents[j]} are linked",
            f"there is a link between {ents[i]} and {ents[j]}",
            f"{ents[i]} connects to {ents[j]}"])))
    rng.shuffle(clauses)
    question = str(rng.choice([
        "Splitting them into two groups, what is the most links you can cut?",
        "What is the maximum number of links between the two groups?"]))
    text, structure = _assemble(clauses, _CUT_INTROS, question, rng)
    sideA = [ents[i] for i in range(n) if best_part[i] == 0]
    sideB = [ents[i] for i in range(n) if best_part[i] == 1]
    trace = (f"Putting {_and_list(sideA) or '(none)'} on one side and {_and_list(sideB) or '(none)'} "
             f"on the other cuts {best_cut} links — the maximum over all 2-way splits.")
    payload = {"kind": "maxcut", "n": n, "edges": [[int(i), int(j)] for i, j in edges]}
    return _core("optimize", "opt_maxcut", "number", text, question, ents,
                 [str(i) for i in range(n)], str(best_cut), int(best_cut), True, trace,
                 payload, scheme, structure, n=n, d=2, query=0)


def _brute_maxcut(n, edges):
    best, bestp = -1, None
    for bits in range(1 << n):
        part = [(bits >> i) & 1 for i in range(n)]
        cut = sum(1 for i, j in edges if part[i] != part[j])
        if cut > best:
            best, bestp = cut, part
    return best, bestp


def _csp_fact_clause(f, ents, vnames, rng):
    k = f[0]
    if k == "pin":
        return f"{ents[f[1]]} is {vnames[f[2]]}"
    if k == "eq":
        return f"{ents[f[1]]} equals {ents[f[2]]}"
    if k == "neq":
        return f"{ents[f[1]]} differs from {ents[f[2]]}"
    if k == "lt":
        return f"{ents[f[1]]} comes before {ents[f[2]]}"
    if k == "le":
        return f"{ents[f[1]]} is no later than {ents[f[2]]}"
    if k == "alldiff":
        return f"{_and_list([ents[i] for i in f[1]])} are all different"
    return ""


def verify_optimize(rec):
    pl = rec["payload"]
    if pl["kind"] == "maxcut":
        best, _ = _brute_maxcut(pl["n"], [(i, j) for i, j in pl["edges"]])
        return rec["answer"] == str(best)
    cons = tuple((tuple(sc), frozenset(tuple(t) for t in al)) for sc, al in pl["cons"])
    csp = C.CSP(pl["n"], pl["d"], cons)
    cost = np.array(pl["cost"], np.int64)
    opt_cost, _, _ = EO.brute_opt(csp, cost)
    return rec["answer"] == str(int(opt_cost))


# =========================================================================== single-domain registry
_GEN = {
    "xor": gen_xor, "graph": gen_graph, "perm": gen_perm,
    "typeinfer": gen_typeinfer, "fol": gen_fol, "optimize": gen_optimize,
}
_VERIFY = {
    "xor": verify_xor, "graph": verify_graph, "perm": verify_perm,
    "typeinfer": verify_typeinfer, "fol": verify_fol, "optimize": verify_optimize,
}


def generate(domain, rng, cfg):
    return _GEN[domain](rng, cfg)


# =========================================================================== COMPOSITIONS (multi-organ)
# Each composition computes an exact integer/bool result r from a "head" domain sub-problem, then a
# modular-arithmetic "tail" combines r into the final answer. Exact end-to-end; tagged by domain pair.
_COMPOSE_INTRO = (
    "Solve the two steps in order; the first result feeds the second.",
    "This problem has two stages — the first answer is used in the second.",
)


# Each _head_* returns (sub_record, R_integer, domain_tag, sub_text, R_instruction). R_integer is the
# number a reader would read off the sub-answer's SURFACE (so the chain is faithful to the prose); the
# R_instruction (referencing {R}) tells the reader how to turn the sub-answer into R.
def _head_graph(rng, cfg):
    rec = gen_graph(rng, {**cfg.get("graph", {}), "n": cfg.get("gn", (6, 11))})
    if rec is None or rec["payload"]["mode"] != "dist" or not rec["determined"]:
        return None
    r = int(rec["answer_index"])                          # the shortest distance the reader computes
    instr = "Let {R} be the answer to that question (a number)."
    return rec, r, "graph", rec["text"], instr


def _head_xor(rng, cfg):
    rec = gen_xor(rng, {**cfg.get("xor", {}), "n": cfg.get("xn", (5, 8)),
                        "sat_frac": 0.0, "det_frac": 1.0})
    if rec is None or not rec["determined"] or rec["payload"]["mode"] != "bit":
        return None
    vals = rec["payload"]["values"]
    r = 1 if rec["answer"] == vals[1] else 0
    instr = "Let {R} be 1 if the answer is %s, otherwise 0." % vals[1]
    return rec, r, "xor", rec["text"], instr


def _head_fol(rng, cfg):
    rec = gen_fol(rng, {**cfg.get("fol", {}), "depth": (1, 4)})
    if rec is None or not rec["determined"]:
        return None
    r = 1 if rec["answer"] == "yes" else 0
    instr = "Let {R} be 1 if the answer is yes, otherwise 0."
    return rec, r, "fol", rec["text"], instr


def _head_perm(rng, cfg):
    rec = gen_perm(rng, {**cfg.get("perm", {}), "n": cfg.get("pn", (4, 6)), "det_frac": 1.0})
    if rec is None or not rec["determined"]:
        return None
    r = int(rec["answer_index"]) + 1                      # the displayed 1-based place number
    instr = "Let {R} be the place number you found above."
    return rec, r, "perm", rec["text"], instr


def _head_type(rng, cfg):
    rec = gen_typeinfer(rng, {**cfg.get("typeinfer", {})})
    if rec is None or not rec["determined"]:
        return None
    r = 1 if "int" in rec["answer"] and "->" not in rec["answer"] else 0
    instr = "Let {R} be 1 if that type is int, otherwise 0."
    return rec, r, "typeinfer", rec["text"], instr


def _head_ordering(rng, cfg):
    """An ordering-CSP head whose forced (1-based) position of the query is the integer result."""
    for _ in range(40):
        cn, cd, kind, facts, _s = CU.gen_ordering(rng, n_lo=4, n_hi=6, pin_frac=0.6)
        csp = CU.build_csp(cn, cd, facts)
        exact = C.exact_dedP(csp, csp.full())
        det = [i for i in range(cn) if len(exact[i]) == 1]
        if det:
            break
    else:
        return None
    q = int(rng.choice(det))
    idx = int(next(iter(exact[q])))
    r = idx + 1                                           # the displayed 1-based position number
    ents, scheme = _names(rng, cn)
    vnames = CU.value_names(kind, cd)
    clauses = [_csp_fact_clause(f, ents, vnames, rng) for f in facts]
    rng.shuffle(clauses)
    question = f"What position does {ents[q]} take?"
    text, structure = _assemble(clauses, ("Order the items from the clues below.",), question, rng)
    rec = _core("ordering", "ordering", "ordinal", text, question, ents, vnames,
                vnames[idx], idx, True, f"Propagating the ordering clues forces {ents[q]} to position "
                f"{vnames[idx]}.", {"cons": [[[int(c) for c in sc],
                sorted([[int(x) for x in t] for t in al])] for sc, al in csp.cons],
                "n": cn, "d": cd, "query": q}, scheme, structure, n=cn, d=cd, query=q)
    instr = f"Let {{R}} be the position number of {ents[q]}."
    return rec, r, "ordering", text, instr


_HEADS = {"graph": _head_graph, "xor": _head_xor, "fol": _head_fol,
          "perm": _head_perm, "typeinfer": _head_type, "ordering": _head_ordering}


def gen_composition(rng, head_domain, cfg, triple=None):
    """Multi-organ composition: one (or two) domain sub-problem(s) -> integer R -> a modular-arithmetic
    tail (final = (R + c) mod m). Exact end-to-end; the prose is self-contained (each sub-answer is
    explicitly turned into R) and reader-faithful (R is the number printed in the sub-answer)."""
    hd = _HEADS[head_domain](rng, cfg)
    if hd is None:
        return None
    rec1, r1, dom1, sub1, instr1 = hd
    m = int(rng.integers(4, 10))
    c = int(rng.integers(0, m))
    domains = [dom1]
    parts = [f"Step 1. {sub1}", instr1.format(R="R1" if triple else "R")]

    if triple is not None:
        hd2 = _HEADS[triple](rng, cfg)
        if hd2 is None:
            return None
        rec2, r2, dom2, sub2, instr2 = hd2
        domains.append(dom2)
        parts += [f"Step 2. {sub2}", instr2.format(R="R2")]
        r = r1 + r2
        tail = f"Step 3. Compute (R1 + R2 + {c}) mod {m}."
        rexpr = f"({r1} + {r2} + {c})"
    else:
        rec2 = None
        r = r1
        tail = f"Step 2. Compute (R + {c}) mod {m}."
        rexpr = f"({r1} + {c})"
    domains.append("arith")
    final = (r + c) % m
    parts.append(tail)
    full_text = "\n\n".join(parts) + "\n\nWhat is the final result?"
    question = f"What is the final result modulo {m}?"

    rline = (f" So R1 = {r1}." if triple else f" So R = {r1}.")
    r2line = f" {rec2['trace']} So R2 = {r2}." if triple is not None else ""
    trace = f"{rec1['trace']}{rline}{r2line} Then {rexpr} mod {m} = {final}."
    payload = {"heads": [_compose_head_payload(rec1)] +
               ([_compose_head_payload(rec2)] if triple is not None else []),
               "m": m, "c": c, "tail": "add_mod"}
    pair = "+".join(domains)
    n_dom = len(set(domains))
    return _core("compose", f"compose_{pair}", "number", full_text, question,
                 rec1["entities"], [str(i) for i in range(m)], str(final), final, True,
                 trace, payload, "compose", "prose", n=rec1["n"], d=m, query=0,
                 n_domains=n_dom, compose=pair)


def _compose_head_payload(rec):
    return {"domain": rec["domain"], "payload": rec["payload"],
            "answer": rec["answer"], "answer_index": rec["answer_index"],
            "determined": rec["determined"]}


def _recompute_head_R(h):
    """Recompute the integer R contributed by one composition head from its payload (exact)."""
    dom = h["domain"]
    sub = {"domain": dom, "payload": h["payload"], "answer": h["answer"],
           "answer_index": h["answer_index"], "determined": h["determined"]}
    if dom == "graph":
        assert verify_graph(sub)
        return int(h["answer_index"])
    if dom == "xor":
        assert verify_xor(sub)
        vals = h["payload"]["values"]
        return 1 if h["answer"] == vals[1] else 0
    if dom == "fol":
        assert verify_fol(sub)
        return 1 if h["answer"] == "yes" else 0
    if dom == "perm":
        assert verify_perm(sub)
        return int(h["answer_index"]) + 1                # 1-based place number (reader-faithful)
    if dom == "typeinfer":
        assert verify_typeinfer(sub)
        return 1 if ("int" in h["answer"] and "->" not in h["answer"]) else 0
    if dom == "ordering":
        cons = tuple((tuple(sc), frozenset(tuple(t) for t in al))
                     for sc, al in h["payload"]["cons"])
        csp = C.CSP(h["payload"]["n"], h["payload"]["d"], cons)
        exact = C.exact_dedP(csp, csp.full())
        q = h["payload"]["query"]
        assert len(exact[q]) == 1
        return int(next(iter(exact[q]))) + 1             # 1-based position number (reader-faithful)
    raise ValueError(dom)


def verify_compose(rec):
    pl = rec["payload"]
    r = sum(_recompute_head_R(h) for h in pl["heads"])
    final = (r + pl["c"]) % pl["m"]
    return rec["answer"] == str(final)


# =========================================================================== unified verification
_ALL_VERIFY = dict(_VERIFY)
_ALL_VERIFY["compose"] = verify_compose


def verify_record(rec):
    """Independent exact re-verification of any extra-domain record from its serialized payload."""
    fn = _ALL_VERIFY.get(rec.get("domain"))
    return bool(fn(rec)) if fn is not None else False


# =========================================================================== smoke self-check
def smoke(n_per=80, seed=0):
    rng = np.random.default_rng(seed)
    cfg = {}
    counts = {}
    for dom in DOMAINS:
        ok = bad = 0
        for _ in range(n_per * 3):
            if ok >= n_per:
                break
            rec = generate(dom, rng, cfg)
            if rec is None:
                continue
            if verify_record(rec):
                ok += 1
            else:
                bad += 1
        counts[dom] = (ok, bad)
        print(f"  {dom:10s} verified {ok:4d}  mismatches {bad}")
        assert bad == 0, f"{dom}: {bad} verification mismatches"
    # compositions
    for head in ("graph", "xor", "fol", "perm", "typeinfer", "ordering"):
        ok = bad = 0
        for _ in range(n_per * 4):
            if ok >= n_per // 2:
                break
            rec = gen_composition(rng, head, cfg)
            if rec is None:
                continue
            (ok := ok + 1) if verify_record(rec) else (bad := bad + 1)
        print(f"  compose[{head:9s}] verified {ok:4d}  mismatches {bad}")
        assert bad == 0, f"compose {head}: {bad} mismatches"
    # a triple
    ok = bad = 0
    for _ in range(n_per * 6):
        if ok >= n_per // 4:
            break
        rec = gen_composition(rng, "graph", cfg, triple="xor")
        if rec is None:
            continue
        (ok := ok + 1) if verify_record(rec) else (bad := bad + 1)
    print(f"  compose[graph+xor triple] verified {ok}  mismatches {bad}")
    assert bad == 0
    print("ALL EXTRA-DOMAIN SMOKE CHECKS PASS — every record exact-verified from its payload.")
    return counts


if __name__ == "__main__":
    smoke()
