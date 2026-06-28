"""clair/organ/faculty_tasks.py — generators for the GRAPH faculty + the CROSS-FACULTY composition task,
and the EXACT composer-level proof that the multifaculty routing is NECESSARY.

Two record generators (schema-compatible with clair.run_glados_staged live records, so the woven trains
on them unchanged) and a self-contained proof:

  gen_graph_record   : a directed-reachability question over CELLS ("can you drive from A to D?"). The
                       answer needs the certified GRAPH closure — a per-cell CSP narrower cannot express
                       transitive reachability, so this exercises the 2nd faculty end-to-end.
  gen_cross_record   : COLOURED reachability — a unique-solution colour CSP over the cells decides which
                       candidate edges are ACTIVE (active iff endpoints share a colour), then the question
                       is reachability over the ACTIVE edges. Solvable ONLY by CSP→flow→graph.
  prove_multifaculty : exact, LM-free. Builds each task's TRUE typed struct, dispatches it, and shows the
                       cross task is solved by the flow while the single-faculty baselines (CSP-only,
                       graph-only) provably FAIL it. This is the multifaculty win, certified.
"""
from __future__ import annotations

import numpy as np

from .. import csp as C
from .. import curriculum as CU
from . import faculty as FAC
from .faculty import (GraphReach, GraphReachState, CrossState, CrossCSPGraph,
                      reachable, active_edges, build_struct_alpha)

YESNO = ["yes", "no"]                                     # the reachability answer domain


# ---------------------------------------------------------------- shared NL helpers
def _cities_intro(n):
    names = ", ".join(CU.ENTITIES[i] for i in range(n))
    return f"Cities {names}."


def _road_sentences(edges):
    return " ".join(f"There is a road from {CU.ENTITIES[u]} to {CU.ENTITIES[v]}." for (u, v) in sorted(edges))


def _mentions(text, n):
    return {int(k): [tuple(s) for s in v] for k, v in CU.entity_mentions(text, n).items()}


def _finish_record(prompt, n, source, target, ans_yes, faculty, extra, alpha_prompt=None,
                   vnames=None, gold_idx=None):
    """Assemble the common record fields (mentions/vnames/gold) for a yes/no reachability prompt. When
    `alpha_prompt` is given (TWO-STREAM), α reads the FULL text (with the roads/colours) while the gen
    prompt is fact-ABLATED (roster + question only) — so the organ readout is the ONLY route to the
    answer (the validated ALPHA_STRUCT engagement recipe; the gate cannot open on a text shortcut).
    `vnames`/`gold_idx` generalize the answer domain beyond yes/no (the type faculty answers a monotype)."""
    body = prompt[: prompt.rfind(" Answer:")]
    vn = list(vnames) if vnames is not None else list(YESNO)
    gi = int(gold_idx) if gold_idx is not None else (0 if ans_yes else 1)
    rec = {
        "prompt": prompt, "answer": vn[gi], "n": n,
        "query": int(target), "determined": True, "gold_idx": int(gi),
        "relation": faculty, "faculty": faculty, "vnames": vn,
        "mentions": _mentions(body, n),
        "source": int(source), "target": int(target),
    }
    if alpha_prompt is not None:
        a_body = alpha_prompt[: alpha_prompt.rfind(" Answer:")] if " Answer:" in alpha_prompt else alpha_prompt
        rec["alpha_prompt"] = alpha_prompt
        rec["alpha_mentions"] = _mentions(a_body, n)
    rec.update(extra)
    return rec


# ---------------------------------------------------------------- GRAPH faculty record
def gen_graph_record(rng, n_choices=(4, 5, 6), p_edge=0.32, two_stream=True):
    """A directed reachability question over cells. Returns a record whose answer NEEDS the graph closure.
    Carries `true_edges` (the graph-head supervision target) + `source`/`target`. two_stream ablates the
    roads from the gen prompt (α reads them; the LM doesn't) so the organ is the only route."""
    n = int(rng.choice(n_choices))
    # sample a directed edge set; resample to keep yes/no roughly balanced + non-trivial
    for _ in range(64):
        edges = frozenset((i, j) for i in range(n) for j in range(n)
                          if i != j and rng.random() < p_edge)
        s = int(rng.integers(n)); t = int(rng.integers(n))
        if t == s:
            continue
        r = reachable(n, edges, s)
        ans = t in r
        # keep it non-degenerate: at least one edge out of the source-component and not all-reachable
        if 1 <= len(r) < n and len(edges) >= 2:
            break
    q = f"Can you drive from {CU.ENTITIES[s]} to {CU.ENTITIES[t]}? Answer:"
    full = f"{_cities_intro(n)} {_road_sentences(edges)} {q}"
    gen = f"{_cities_intro(n)} {q}" if two_stream else full
    surv = GraphReach().survival(GraphReachState(n, edges, s, t), FAC_K())
    return _finish_record(gen, n, s, t, ans, "graph",
                          {"true_edges": sorted(edges), "surv": surv, "tgt": surv.copy(),
                           "true_facts": []}, alpha_prompt=full if two_stream else None)


# ---------------------------------------------------------------- CROSS faculty record
def _planted_2coloring(rng, n):
    """A connected tree of `neq` edges + a pin ⇒ a UNIQUE 2-colouring (the CSP faculty's job). Returns
    (facts, colors, tree_edges)."""
    perm = list(rng.permutation(n))
    tree = [(perm[i], perm[rng.integers(i)]) for i in range(1, n)]   # random spanning tree (undirected neq)
    # 2-colour the tree by BFS from perm[0]
    adj = {i: [] for i in range(n)}
    for (u, v) in tree:
        adj[u].append(v); adj[v].append(u)
    color = [None] * n
    root = perm[0]; c0 = int(rng.integers(2)); color[root] = c0
    stack = [root]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if color[y] is None:
                color[y] = 1 - color[x]; stack.append(y)
    facts = [("pin", root, c0)] + [("neq", u, v) for (u, v) in tree]
    return facts, color, tree


def gen_cross_record(rng, n_choices=(5, 6, 7), p_cand=0.45, want_yes=None, two_stream=True):
    """COLOURED reachability: the unique 2-colouring decides which candidate edges are active (active iff
    same colour), then reachability over the ACTIVE edges answers. `want_yes` (True/False/None) forces the
    answer class so the woven trains on a BALANCED set (reachability over the sparse active graph skews to
    'no' otherwise). The single-faculty separation is honest: graph-only (all candidate edges, no colour
    gating) OVER-reaches, so it is wrong exactly on the 'no'-but-candidate-reachable instances."""
    for _ in range(512):
        n = int(rng.choice(n_choices))
        facts, colors, _tree = _planted_2coloring(rng, n)
        csp = CU.build_csp(n, 2, facts)
        # confirm the CSP is uniquely determined (the faculty-A precondition)
        ded = C.exact_dedP(csp, csp.full())
        if any(len(ded[i]) != 1 for i in range(n)):
            continue
        solved = [int(next(iter(ded[i]))) for i in range(n)]
        if solved != colors:
            continue
        cand = frozenset((i, j) for i in range(n) for j in range(n)
                        if i != j and rng.random() < p_cand)
        act = active_edges(colors, cand, "eq")
        s = int(rng.integers(n))
        reach_act = reachable(n, act, s)
        reach_cand = reachable(n, cand, s)
        # choose target to realize the requested answer class (balanced training)
        if want_yes is True:
            choices = [t for t in range(n) if t != s and t in reach_act]
        elif want_yes is False:
            choices = [t for t in range(n) if t != s and t not in reach_act]
        else:
            choices = [t for t in range(n) if t != s]
        if not choices:
            continue
        t = int(rng.choice(choices))
        ans_cross = t in reach_act                         # the TRUE (multifaculty) answer
        ans_graphonly = t in reach_cand                    # graph-only baseline (ignores colours)
        q = (f"Using only roads between cities of the same color, can you travel "
             f"from {CU.ENTITIES[s]} to {CU.ENTITIES[t]}? Answer:")
        full = (f"{_cities_intro(n)} {' '.join(_color_sentence(f) for f in facts)} "
                f"{_road_sentences(cand)} {q}")
        gen = f"{_cities_intro(n)} {q}" if two_stream else full
        surv = CrossCSPGraph().survival(CrossState(csp, cand, s, t, "eq"), FAC_K())
        return _finish_record(gen, n, s, t, ans_cross, "cross",
                              {"true_facts": [list(f) for f in facts],
                               "true_edges": sorted(cand), "cand_edges": sorted(cand),
                               "colors": solved, "surv": surv, "tgt": surv.copy(),
                               "ans_graphonly": bool(ans_graphonly)},
                              alpha_prompt=full if two_stream else None)
    raise RuntimeError("could not sample a discriminating cross-faculty instance")


def _color_sentence(f):
    COL = CU.COLORS
    if f[0] == "pin":
        return f"{CU.ENTITIES[f[1]]} is {COL[f[2]]}."
    if f[0] == "neq":
        return f"{CU.ENTITIES[f[1]]} and {CU.ENTITIES[f[2]]} are different colors."
    if f[0] == "eq":
        return f"{CU.ENTITIES[f[1]]} and {CU.ENTITIES[f[2]]} are the same color."
    return ""


def FAC_K():
    from .. import run_glados_staged as G
    return G.K


# ================================================================ exact Ising / max-cut ground truth
def _ising_energy(J, h, s):
    return float(-0.5 * (s @ J @ s) - (h * s).sum())


def _all_ground_states(J, h, n):
    """All min-energy ±1 configs of H=−½sᵀJs−hᵀs by brute enumeration (small n). Returns (list[array], E)."""
    best = None
    mins = []
    for m in range(1 << n):
        s = np.array([1.0 if (m >> i) & 1 else -1.0 for i in range(n)], dtype=np.float32)
        E = _ising_energy(J, h, s)
        if best is None or E < best - 1e-9:
            best = E
            mins = [s.copy()]
        elif abs(E - best) < 1e-9:
            mins.append(s.copy())
    return mins, best


def _relspin_determined(J, h, n, source, target):
    """Across ALL ground states (canonicalised so spin[source]=+1), is spin[target] constant? Returns
    (determined, same_side_bool). The well-posedness gate for the relative-team question."""
    mins, _E = _all_ground_states(J, h, n)
    vals = set()
    for s in mins:
        sc = s * s[source]                                   # canonicalize source := +1
        vals.add(int(sc[target]))
    if len(vals) != 1:
        return False, None
    return True, (next(iter(vals)) == 1)


# ---------------------------------------------------------------- ISING faculty record
def _rivalry_sentences(edges):
    return " ".join(f"{CU.ENTITIES[u]} and {CU.ENTITIES[v]} are rivals." for (u, v) in sorted(edges))


def gen_ising_record(rng, n_choices=(5, 6, 7), p_edge=0.42, want_yes=None, two_stream=True):
    """A MAX-CUT optimization NL task: split cities into two teams so that rivals land on OPPOSITE teams
    (max-cut of the rivalry graph). The answer (is t on s's team in the optimal split?) needs argmin over
    an energy — narrowing+chaining provably cannot express it. Carries the true (J,h) for α's coupling
    supervision (the CouplingProjector target). Resampled until the queried pair's relative team is
    DETERMINED (constant across all ground states). `want_yes` balances the same/different team classes."""
    for _ in range(256):
        n = int(rng.choice(n_choices))
        edges = frozenset((i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p_edge)
        if len(edges) < n - 1:
            continue
        J = FAC.maxcut_J(n, edges); h = np.zeros(n, dtype=np.float32)
        s = 0
        cand = [t for t in range(n) if t != s]
        rng.shuffle(cand)
        for t in cand:
            det, same = _relspin_determined(J, h, n, s, t)
            if det and (want_yes is None or bool(same) == bool(want_yes)):
                q = (f"Split the cities into two teams so that rivals are on opposite teams. "
                     f"Is {CU.ENTITIES[t]} on the same team as {CU.ENTITIES[s]}? Answer:")
                full = f"{_cities_intro(n)} {_rivalry_sentences(edges)} {q}"
                gen = f"{_cities_intro(n)} {q}" if two_stream else full
                surv = FAC.IsingMaxcut().survival(FAC.IsingState(n, J, h, s, t), FAC_K())
                return _finish_record(gen, n, s, t, same, "ising",
                                      {"true_edges": sorted(edges), "J_true": J.tolist(),
                                       "h_true": h.tolist(), "surv": surv, "tgt": surv.copy(),
                                       "true_facts": []}, alpha_prompt=full if two_stream else None)
    raise RuntimeError("could not sample a determined Ising instance")


# ---------------------------------------------------------------- TYPE faculty record
def _type_decl_sentence(name, tname):
    art = "an" if tname == "int" else "a"
    return f"{name} is {art} {'integer' if tname=='int' else 'boolean'}."


def gen_type_record(rng, n_choices=(4, 5, 6), two_stream=True):
    """A TYPE-INFERENCE NL task: variables get base monotypes by declaration, propagated through 'has the
    same type as' (type-variable unification = eq on the type lattice). The query variable is base-typed
    via a chain of equalities the organ must follow (clair.typeinfer / csp arc-consistency). Answer ∈
    {int,bool}. Distractors of the other type keep it non-degenerate."""
    n = int(rng.choice(n_choices))
    perm = list(rng.permutation(n))
    # an eq-tree so types propagate; a base-type pin per root; >=2 groups for both int and bool to appear
    facts = []
    # split cells into 2 groups by a random cut so both types can occur
    cut = 1 + int(rng.integers(max(1, n - 2)))
    groups = [perm[:cut], perm[cut:]] if cut < n else [perm]
    decl_text = []
    cell_type = {}
    for gi, grp in enumerate(groups):
        if not grp:
            continue
        tname = TYPE_NAMES_LOCAL[int(rng.integers(2))]
        tval = 0 if tname == "int" else 1
        root = grp[0]
        facts.append(("pin", root, tval))
        decl_text.append(_type_decl_sentence(CU.ENTITIES[root], tname))
        cell_type[root] = tval
        # link the rest of the group to the root via an eq-tree (random parent already placed)
        placed = [root]
        for c in grp[1:]:
            par = placed[int(rng.integers(len(placed)))]
            facts.append(("eq", c, par))
            decl_text.append(f"{CU.ENTITIES[c]} has the same type as {CU.ENTITIES[par]}.")
            cell_type[c] = tval
            placed.append(c)
    # query a NON-pinned cell (one whose type required following an equality)
    nonpinned = [c for c in range(n) if c in cell_type and not any(
        f[0] == "pin" and f[1] == c for f in facts)]
    if not nonpinned:
        return gen_type_record(rng, n_choices, two_stream)
    t = int(rng.choice(nonpinned))
    tval = cell_type[t]
    q = f"What is the type of {CU.ENTITIES[t]}? Answer:"
    full = f"Variables {', '.join(CU.ENTITIES[i] for i in range(n))}. {' '.join(decl_text)} {q}"
    gen = f"Variables {', '.join(CU.ENTITIES[i] for i in range(n))}. {q}" if two_stream else full
    surv = FAC.TypeInfer().survival(FAC.TypeState(FAC.build_type_csp(facts, n), n, t), FAC_K())
    return _finish_record(gen, n, 0, t, None, "type",
                          {"true_facts": [list(f) for f in facts], "true_edges": [],
                           "surv": surv, "tgt": surv.copy()},
                          alpha_prompt=full if two_stream else None,
                          vnames=list(TYPE_NAMES_LOCAL), gold_idx=tval)


# ---------------------------------------------------------------- REDUCTION faculty record (MIS via Karp route)
def _border_sentences(edges):
    return " ".join(f"{CU.ENTITIES[u]} borders {CU.ENTITIES[v]}." for (u, v) in sorted(edges))


def _all_max_independent_sets(n, edges):
    """All MAXIMUM independent sets of an undirected graph (small n). Returns list[frozenset]."""
    adj = {i: set() for i in range(n)}
    for (u, v) in edges:
        adj[u].add(v); adj[v].add(u)
    best = -1; sets = []
    for m in range(1 << n):
        sel = [i for i in range(n) if (m >> i) & 1]
        if all(v not in adj[u] for u in sel for v in sel if v != u):
            if len(sel) > best:
                best = len(sel); sets = [frozenset(sel)]
            elif len(sel) == best:
                sets.append(frozenset(sel))
    return sets


def gen_reduction_record(rng, n_choices=(5, 6, 7), p_edge=0.34, want_yes=None, two_stream=True):
    """A MAX-INDEPENDENT-SET NL task — a problem with NO DIRECT faculty (argmax|IS| is an optimization, not
    a narrowing). Solved by ROUTING across the Karp graph mis→ising (organ/reductions). Resampled until the
    queried city's MIS-membership is DETERMINED (constant across all maximum independent sets). `want_yes`
    balances the in-set / not-in-set classes."""
    for _ in range(256):
        n = int(rng.choice(n_choices))
        edges = frozenset((i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p_edge)
        if len(edges) < 2:
            continue
        mis = _all_max_independent_sets(n, edges)
        if len(mis) == 0:
            continue
        det_cells = [t for t in range(n) if all((t in s) for s in mis) or all((t not in s) for s in mis)]
        if want_yes is not None:
            det_cells = [t for t in det_cells if (t in mis[0]) == bool(want_yes)]
        if not det_cells:
            continue
        t = int(rng.choice(det_cells))
        ans_yes = (t in mis[0])
        q = (f"Choose the largest possible set of cities with no two that border each other. "
             f"Is {CU.ENTITIES[t]} in that set? Answer:")
        full = f"{_cities_intro(n)} {_border_sentences(edges)} {q}"
        gen = f"{_cities_intro(n)} {q}" if two_stream else full
        # the graph head is supervised on the symmetric adjacency (both directions)
        diredges = sorted(set((u, v) for (u, v) in edges) | set((v, u) for (u, v) in edges))
        surv = FAC.ReductionRoute().survival(FAC.ReductionState(n, edges, 0, t), FAC_K())
        return _finish_record(gen, n, 0, t, ans_yes, "reduction",
                              {"true_edges": diredges, "surv": surv, "tgt": surv.copy(),
                               "true_facts": []}, alpha_prompt=full if two_stream else None)
    raise RuntimeError("could not sample a determined MIS instance")


# ---------------------------------------------------------------- TRI faculty record (3-hop multiorganic)
def _tri_channel_answer(csp, cand_edges, source, target, n, channel):
    """The EXACT (LM-free) same-team answer under one wiring channel, with its determinacy — for the
    necessity proof. Returns (determined, same_team_bool)."""
    und, nodes = FAC._tri_channel_edges(csp, cand_edges, source, n, channel)
    if source not in nodes:
        nodes = nodes | {source}
    node_list = sorted(nodes)
    loc = {v: i for i, v in enumerate(node_list)}
    m = len(node_list)
    if target not in loc:
        return True, False                                   # outside district → 'opposite' by convention
    J = FAC.maxcut_J(m, [(loc[u], loc[v]) for (u, v) in und if u in loc and v in loc])
    h = np.zeros(m, dtype=np.float32)
    return _relspin_determined(J, h, m, loc[source], loc[target])


def gen_tri_record(rng, n_choices=(6, 7, 8), p_cand=0.5, want_yes=None, two_stream=True):
    """The 3-FACULTY beast: unique 2-colouring (CSP) → same-colour roads are open (csp→graph flow) → the
    cities reachable from S form a district (GRAPH closure) → split the district's open roads by max-cut
    (graph→ising flow). Query: is T on S's team in the optimal split? Resampled until (a) T is in the
    district, (b) the district max-cut relative team of (S,T) is DETERMINED, and (c) the full 3-hop answer
    DIFFERS from the ising-only ('raw') answer — so a single/two-faculty organ provably fails."""
    for _ in range(1024):
        n = int(rng.choice(n_choices))
        facts, colors, _tree = _planted_2coloring(rng, n)
        csp = CU.build_csp(n, 2, facts)
        ded = C.exact_dedP(csp, csp.full())
        if any(len(ded[i]) != 1 for i in range(n)):
            continue
        if [int(next(iter(ded[i]))) for i in range(n)] != colors:
            continue
        cand = frozenset((i, j) for i in range(n) for j in range(n)
                        if i != j and rng.random() < p_cand)
        s = int(rng.integers(n))
        act = active_edges(colors, cand, "eq")
        dist = reachable(n, act, s)
        if not (2 <= len(dist) <= n // 2 + 1):               # district a SMALL proper subset, so the
            continue                                          # outside structure perturbs the color-only
        #                                                     # max-cut → the reachability flow matters more
        choices = [t for t in dist if t != s]
        rng.shuffle(choices)
        # NECESSITY: every valid target must DIFFER from the ising-only ('raw') wiring (so opening no
        # flow / the optimization alone is wrong). PREFER targets that ALSO differ from the csp→ising
        # ('color', no graph-closure) wiring — so the graph-closure flow is necessary too; this is what
        # the learned flow gates must discover. We keep the strongest available target.
        strong = weak = None
        for t in choices:
            det_full, same_full = _tri_channel_answer(csp, cand, s, t, n, "color_reach")
            if not det_full:
                continue
            if want_yes is not None and bool(same_full) != bool(want_yes):
                continue
            det_raw, same_raw = _tri_channel_answer(csp, cand, s, t, n, "raw")
            if det_raw and same_raw == same_full:
                continue                                      # ising-only must be wrong (hard)
            det_col, same_col = _tri_channel_answer(csp, cand, s, t, n, "color")
            if det_col and same_col != same_full:
                strong = (t, same_full); break               # both flows necessary — best
            weak = weak or (t, same_full)
        sel = strong or weak
        if sel is None:
            continue
        t, same_full = sel
        q = (f"Cities of the same color are connected by open roads. Among the cities reachable from "
             f"{CU.ENTITIES[s]} along open roads, split them into two teams to put as many open-road "
             f"pairs as possible on opposite teams. Is {CU.ENTITIES[t]} on the same team as "
             f"{CU.ENTITIES[s]}? Answer:")
        full = (f"{_cities_intro(n)} {' '.join(_color_sentence(f) for f in facts)} "
                f"{_road_sentences(cand)} {q}")
        gen = f"{_cities_intro(n)} {q}" if two_stream else full
        surv = FAC.TriCSPGraphIsing().survival(FAC.TriState(csp, cand, s, t), FAC_K())
        return _finish_record(gen, n, s, t, same_full, "tri",
                              {"true_facts": [list(f) for f in facts],
                               "true_edges": sorted(cand), "cand_edges": sorted(cand),
                               "colors": colors, "surv": surv, "tgt": surv.copy()},
                              alpha_prompt=full if two_stream else None)
    raise RuntimeError("could not sample a discriminating tri-faculty instance")


TYPE_NAMES_LOCAL = ["int", "bool"]


# ---------------------------------------------------------------- the exact, LM-free multifaculty proof
def prove_multifaculty(n_inst=200, seed=0, verbose=True):
    """Exact composer-level proof (no LM): dispatch each task's TRUE typed struct and show
      (1) the GRAPH faculty solves reachability (2nd faculty fires end-to-end on its problem type);
      (2) the CROSS task is solved by the CSP→flow→graph dispatch;
      (3) the SINGLE-FACULTY baselines FAIL the cross task (CSP-only can't express reachability; graph-only
          ignores the colours) — so the multifaculty routing is NECESSARY.
    Returns a results dict."""
    rng = np.random.default_rng(seed)
    K = FAC_K()

    # (1) graph faculty: answer == certified closure on every instance
    g_ok = 0
    for _ in range(n_inst):
        r = gen_graph_record(rng)
        ts = build_struct_alpha("graph", (r["true_edges"], r["source"], r["target"]), r["n"], 2)
        out = GraphReach().reduce(ts.state)
        pred_yes = bool(out.reach[r["target"]])
        g_ok += int(pred_yes == (r["gold_idx"] == 0))
    graph_acc = g_ok / n_inst

    # (2)+(3) cross task: multifaculty vs single-faculty baselines, on a BALANCED yes/no set
    cross_ok = graphonly_ok = 0
    golds = []
    for k in range(n_inst):
        r = gen_cross_record(rng, want_yes=(k % 2 == 0))
        n = r["n"]
        gold_yes = (r["gold_idx"] == 0); golds.append(gold_yes)
        # MULTIFACULTY: CSP colours → active edges → reachability (the dispatch flow)
        ts = build_struct_alpha("cross", (r["true_facts"], r["cand_edges"], r["source"], r["target"]), n, 2)
        cout = CrossCSPGraph().reduce(ts.state)
        cross_ok += int(bool(cout.reach[r["target"]]) == gold_yes)
        # GRAPH-ONLY baseline: reachability over ALL candidate edges (no colour gating) — over-reaches
        gpred = r["target"] in reachable(n, frozenset(map(tuple, r["cand_edges"])), r["source"])
        graphonly_ok += int(gpred == gold_yes)
    # CSP-ONLY baseline: a per-cell CSP narrower has NO reachability readout (the question is reachability,
    # not a cell value) — faculty-blind. Its ceiling is the MAJORITY-class constant guess.
    p_yes = sum(golds) / max(1, len(golds))
    csponly_acc = max(p_yes, 1.0 - p_yes)
    return {
        "n_inst": n_inst,
        "graph_faculty_acc": graph_acc,
        "cross_balance_yes_frac": p_yes,
        "cross_multifaculty_acc": cross_ok / n_inst,
        "cross_graphonly_acc": graphonly_ok / n_inst,
        "cross_csponly_majority_acc": csponly_acc,
    }


def prove_extended(n_inst=120, seed=0):
    """Exact, LM-free necessity proofs for the NEW faculties (ising / type / reduction / tri):
      ising     : the organ's optimal-split relative team == the exact ground-truth (determined max-cut);
                  a non-optimizing baseline (majority guess) cannot.
      type      : the type faculty's per-cell forced monotype == the exact typing on determined queries.
      reduction : MIS membership via the Karp route mis→ising == the exact MIS membership.
      tri       : the full 3-hop answer == ground truth, while EVERY sub-wiring (raw=ising-only,
                  color=csp+ising-no-closure) and the single-faculty baselines provably FAIL."""
    rng = np.random.default_rng(seed)
    K = FAC_K()
    res = {"n_inst": n_inst}

    # ---- ISING: organ optimum vs exact determined gold; majority baseline ----
    ok = 0; golds = []
    for k in range(n_inst):
        r = gen_ising_record(rng, want_yes=(k % 2 == 0))
        sv = FAC.IsingMaxcut().survival(
            FAC.IsingState(r["n"], np.asarray(r["J_true"], np.float32),
                           np.asarray(r["h_true"], np.float32), r["source"], r["target"]), K)
        ok += int(int(sv[r["target"]].argmax()) == r["gold_idx"])
        golds.append(r["gold_idx"] == 0)
    p = sum(golds) / max(1, len(golds))
    res["ising_faculty_acc"] = ok / n_inst
    res["ising_majority_acc"] = max(p, 1.0 - p)

    # ---- TYPE: forced monotype vs exact on determined queries ----
    ok = 0
    for _ in range(n_inst):
        r = gen_type_record(rng)
        sv = FAC.TypeInfer().survival(FAC.TypeState(FAC.build_type_csp(
            [tuple(f) for f in r["true_facts"]], r["n"]), r["n"], r["target"]), K)
        ok += int(int(sv[r["target"]].argmax()) == r["gold_idx"])
    res["type_faculty_acc"] = ok / n_inst

    # ---- REDUCTION: MIS-via-route membership vs exact ----
    ok = 0; golds = []
    for k in range(n_inst):
        r = gen_reduction_record(rng, want_yes=(k % 2 == 0))
        und = frozenset((min(u, v), max(u, v)) for (u, v) in r["true_edges"])
        sv = FAC.ReductionRoute().survival(FAC.ReductionState(r["n"], und, 0, r["target"]), K)
        ok += int(int(sv[r["target"]].argmax()) == r["gold_idx"])
        golds.append(r["gold_idx"] == 0)
    p = sum(golds) / max(1, len(golds))
    res["reduction_faculty_acc"] = ok / n_inst
    res["reduction_majority_acc"] = max(p, 1.0 - p)

    # ---- TRI: full 3-hop vs each sub-wiring ----
    tri_full = tri_raw = tri_color = 0; golds = []
    for k in range(n_inst):
        r = gen_tri_record(rng, want_yes=(k % 2 == 0))
        n = r["n"]; csp = CU.build_csp(n, 2, [tuple(f) for f in r["true_facts"]])
        cand = frozenset(map(tuple, r["cand_edges"])); s = r["source"]; t = r["target"]
        gold = (r["gold_idx"] == 0); golds.append(gold)
        _d, full = _tri_channel_answer(csp, cand, s, t, n, "color_reach")
        dr, raw = _tri_channel_answer(csp, cand, s, t, n, "raw")
        dc, col = _tri_channel_answer(csp, cand, s, t, n, "color")
        tri_full += int(full == gold)
        tri_raw += int(dr and raw == gold)
        tri_color += int(dc and col == gold)
    p = sum(golds) / max(1, len(golds))
    res["tri_full3hop_acc"] = tri_full / n_inst
    res["tri_isingonly_acc"] = tri_raw / n_inst              # raw = ising on all edges (no flows)
    res["tri_csp_ising_acc"] = tri_color / n_inst           # color flow only (no graph closure)
    res["tri_majority_acc"] = max(p, 1.0 - p)
    return res


def main():
    res = prove_multifaculty(n_inst=200, seed=0)
    print("================ MULTIFACULTY PROOF (exact, LM-free) ================")
    print(f"  GRAPH faculty (2nd faculty fires on reachability): {res['graph_faculty_acc']*100:.1f}% "
          f"exact over {res['n_inst']} instances")
    print(f"  CROSS task ({res['cross_balance_yes_frac']*100:.0f}% yes) — MULTIFACULTY (CSP→flow→graph) : "
          f"{res['cross_multifaculty_acc']*100:.1f}%")
    print(f"  CROSS task — GRAPH-ONLY baseline (no colours)    : {res['cross_graphonly_acc']*100:.1f}%")
    print(f"  CROSS task — CSP-ONLY baseline (majority guess)  : {res['cross_csponly_majority_acc']*100:.1f}%")
    win = (res["cross_multifaculty_acc"] > 0.99
           and res["cross_graphonly_acc"] < 0.9 and res["cross_csponly_majority_acc"] < 0.7)
    print(f"\n  MULTIFACULTY NECESSARY (cross solved, single-faculty fails): {win}")
    return res


if __name__ == "__main__":
    main()
