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


def _finish_record(prompt, n, source, target, ans_yes, faculty, extra, alpha_prompt=None):
    """Assemble the common record fields (mentions/vnames/gold) for a yes/no reachability prompt. When
    `alpha_prompt` is given (TWO-STREAM), α reads the FULL text (with the roads/colours) while the gen
    prompt is fact-ABLATED (roster + question only) — so the organ readout is the ONLY route to the
    answer (the validated ALPHA_STRUCT engagement recipe; the gate cannot open on a text shortcut)."""
    body = prompt[: prompt.rfind(" Answer:")]
    gold_idx = 0 if ans_yes else 1                         # YESNO = ["yes","no"]
    rec = {
        "prompt": prompt, "answer": "yes" if ans_yes else "no", "n": n,
        "query": int(target), "determined": True, "gold_idx": int(gold_idx),
        "relation": faculty, "faculty": faculty, "vnames": list(YESNO),
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
