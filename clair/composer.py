"""clair/composer.py — the MULTI-ORGAN COMPOSER: verifier-gated search over CERTIFIED organ-reductions
on a shared abstract state, then a learned ROUTER distilled from the oracle search routes.

This is the open frontier the codex_zoo_review names: "Composition = REDUCED PRODUCTS (each reducer
with its own soundness rule), not dense fusion ... log oracle routes, train router to imitate." We
realise it GPU-free (the search is exact integer/set arithmetic; only the tiny router uses torch).

THE SHARED ABSTRACT STATE is the per-cell value-domain lattice of clair.csp / clair.permgroup:
    dom = (frozenset_0, ..., frozenset_{n-1}),   dom[i] ⊆ {0..d-1} = still-alive values at cell i.
We only ever MEET (narrow). A cell is SOLVED at |dom[i]|=1, CONFLICT (⊥) at |dom[i]|=0.

THE BANK — CERTIFIED ORGAN-REDUCTIONS (sound-by-construction; each maps dom -> dom' ⊆ dom and never
drops a value used by a true solution).  Each one reads a DIFFERENT constraint TYPE, so no single
reduction sees the whole problem — that orthogonality is what makes composition necessary:
    ac        clair.csp.ac_step                — per-cell GAC over extensional constraints `cons`.
    pair      clair.csp path-consistency       — level-1 (2-cell) narrowing, projected to cells.
    factor    clair.csp factor lattice (k=3)    — level-2 (3-cell: arithmetic/parity), projected.
    alldiff   Régin/Hall matching GAC          — over each AllDifferent group (clair.permgroup core).
    ordering  bounds propagation               — over σ(i)<σ(j) order constraints.
    graph     min-plus reachability            — dom[target] ∩= reachable-from-source value set.
The CSP operators (ac/pair/factor) see ONLY `cons`; alldiff/ordering/graph see ONLY their own typed
constraint. So a problem mixing types is invisible-as-a-whole to every single organ.

WHY COMPOSITION SOLVES WHAT ONE ORGAN CANNOT.  Each reduction is a monotone narrowing operator on the
domain lattice; the REDUCED PRODUCT (run the whole bank to a joint fixpoint) is their meet, which is
≥ as strong as any single operator's fixpoint and — on the engineered mixed-type instances — STRICTLY
stronger: every single organ's fixpoint is `open`, the joint fixpoint is `solved`. (Among pure
narrowing ops the joint fixpoint is order-INDEPENDENT/confluent, so the search's job is least-COST
sequencing + knowing-when-to-branch, not changing the final point — reported honestly.)

VERIFIER-GATED SEARCH (VerMCTS-style).  Best-first over which reduction to apply next; the GATE is
that every reduction is certified-sound, so any path is sound and a CONTRADICTION (empty cell) prunes
its subtree (the monotone narrowing IS the pruning). When propagation stalls with the state still
open, we BRANCH (case-split the smallest open cell) — the search is exact and verifier-checked end to
end against brute-force enumeration of ALL constraint types.

ROUTER DISTILLATION.  We log the oracle's (state-features -> chosen reduction) decisions and train a
small MLP router to predict the next reduction from CHEAP state features (no peeking at reduction
outputs). Cajal soft-branch reading: the router emits w = softmax(logits) over reductions and the
forward-pass executor applies the argmax organ each step — composition with NO search. We measure
route-prediction accuracy, held-out COMPOSITION-TYPE generalisation, and forward-pass solve rate.

Run:  python -m clair.composer --smoke      # bank composes under search to solve a 2-organ problem, soundly
      python -m clair.composer              # composition table (composer vs best single) + router distillation
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import csp as C


# ===================================================================== the unified problem
@dataclass(frozen=True)
class CProblem:
    n: int
    d: int
    cons: tuple = ()        # extensional CSP constraints: tuple of (scope tuple, allowed frozenset[tuple])
    alldiff: tuple = ()     # tuple of cell-index tuples; cells in each group must be pairwise distinct
    order: tuple = ()       # tuple of (i, j) meaning val[i] < val[j]
    graph: tuple = ()       # tuple of (edges tuple[(u,v)], source value, target cell): dom[target] ⊆ reach(source)
    ctype: str = ""         # composition-type tag (bookkeeping)

    def full(self):
        V = frozenset(range(self.d))
        return tuple(V for _ in range(self.n))


def meet(a, b):
    return tuple(a[i] & b[i] for i in range(len(a)))


def status(dom):
    if any(len(c) == 0 for c in dom):
        return "conflict"
    if all(len(c) == 1 for c in dom):
        return "solved"
    return "open"


# ===================================================================== exact ground truth (all types)
def _reach(edges, source, d):
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
    seen = {source}
    stack = [source]
    while stack:
        x = stack.pop()
        for y in adj.get(x, ()):
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return frozenset(s for s in seen if 0 <= s < d)


def initial_domain(prob: CProblem):
    """Full grid tightened by the *unary* facts only (unary cons + graph reachability) — the honest
    starting state. All multi-cell structure is left for the reductions to discover."""
    dom = [set(range(prob.d)) for _ in range(prob.n)]
    for sc, al in prob.cons:
        if len(sc) == 1:
            allowed = {t[0] for t in al}
            dom[sc[0]] &= allowed
    # NB graph reachability is itself a reduction; we do NOT pre-apply it (so `graph` must be routed).
    return tuple(frozenset(c) for c in dom)


class _Stop(Exception):
    pass


def solutions(prob: CProblem, limit=None):
    """All assignments satisfying EVERY constraint type — exact brute-force ground truth (small n)."""
    n, d = prob.n, prob.d
    dom0 = [set(range(d)) for _ in range(n)]
    for sc, al in prob.cons:
        if len(sc) == 1:
            dom0[sc[0]] &= {t[0] for t in al}
    for edges, source, target in prob.graph:
        dom0[target] &= set(_reach(edges, source, d))

    order = sorted(range(n), key=lambda i: len(dom0[i]))
    rank = {c: k for k, c in enumerate(order)}
    # multi-cell cons fire when their last cell is assigned
    con_at = {k: [] for k in range(n)}
    for sc, al in prob.cons:
        if len(sc) >= 2:
            con_at[max(rank[c] for c in sc)].append((sc, al))
    ord_at = {k: [] for k in range(n)}
    for (i, j) in prob.order:
        ord_at[max(rank[i], rank[j])].append((i, j))
    # alldiff groups: track used values per group, check distinctness incrementally
    groups = [set(g) for g in prob.alldiff]
    assign = [None] * n
    out = []

    def rec(k):
        if k == n:
            out.append(tuple(assign))
            if limit and len(out) >= limit:
                raise _Stop
            return
        cell = order[k]
        for v in dom0[cell]:
            # alldiff distinctness
            ok = True
            for g in groups:
                if cell in g and any(assign[c] == v for c in g if c != cell and assign[c] is not None):
                    ok = False
                    break
            if not ok:
                continue
            assign[cell] = v
            if all(tuple(assign[c] for c in sc) in al for sc, al in con_at[k]) and \
               all(assign[i] < assign[j] for (i, j) in ord_at[k]):
                rec(k + 1)
        assign[cell] = None

    try:
        rec(0)
    except _Stop:
        pass
    return out


# ===================================================================== bipartite matching (alldiff core)
def _bipartite_match(domlist):
    """SDR / perfect matching: assign each cell a DISTINCT value from its domain. Returns a match list
    (value per cell) or None. Kuhn augmenting paths — exact. The Hall condition the alldiff GAC needs."""
    match_val = {}                                   # value -> cell index

    def aug(c, seen):
        for v in domlist[c]:
            if v in seen:
                continue
            seen.add(v)
            if v not in match_val or aug(match_val[v], seen):
                match_val[v] = c
                return True
        return False

    for c in range(len(domlist)):
        if not domlist[c] or not aug(c, set()):
            return None
    return match_val


# ===================================================================== the certified reductions (the BANK)
@lru_cache(maxsize=4096)
def _csp_of(prob: CProblem):
    return C.CSP(prob.n, prob.d, prob.cons)


def red_ac(prob, dom):
    """Per-cell GAC over extensional `cons` (clair.csp.ac_step). Sound & cheap. Sees only `cons`."""
    return C.ac_step(_csp_of(prob), dom)


def red_pair(prob, dom):
    """Level-1 path-consistency over `cons`, projected to cells then met into dom. Sound; sees only `cons`."""
    csp = _csp_of(prob)
    P = C.pair_init(csp, dom)
    for _ in range(csp.n * csp.d * csp.d + 2):
        nxt = C.pair_step(csp, P)
        if nxt == P:
            break
        P = nxt
    return meet(dom, C.pair_cells(csp, P))


def red_factor(prob, dom):
    """Level-2 factor (3-cell) consistency over `cons` — sees arithmetic/parity the pair level can't.
    Projected to cells, met into dom. Sound; sees only `cons`."""
    csp = _csp_of(prob)
    factors = C.default_factors(csp, 3)
    st = C.factor_init(csp, factors, dom)
    for _ in range(csp.n * csp.d * csp.d + 2):
        nxt = C.factor_step(csp, st, factors)
        if nxt == st:
            break
        st = nxt
    return meet(dom, C.factor_cells(csp, st, factors))


def red_alldiff(prob, dom):
    """Régin/Hall AllDifferent GAC over each alldiff group: keep v at cell c iff some SDR of the group
    uses c->v. Sound & complete for one AllDifferent; sees only the alldiff structure."""
    new = list(dom)
    for group in prob.alldiff:
        cells = list(group)
        sub = [set(new[c]) for c in cells]
        if _bipartite_match(sub) is None:
            for c in cells:
                new[c] = frozenset()                 # ⊥ : no system of distinct representatives
            continue
        for ci, c in enumerate(cells):
            keep = set()
            for v in sub[ci]:
                forced = [({v} if i == ci else (sub[i] - {v})) for i in range(len(cells))]
                if _bipartite_match(forced) is not None:
                    keep.add(v)
            new[c] = frozenset(keep)
    return tuple(new)


def red_ordering(prob, dom):
    """Bounds propagation for each val[i] < val[j]. Sound; sees only `order`."""
    new = list(dom)
    for (i, j) in prob.order:
        if not new[i] or not new[j]:
            continue
        hi_j, lo_i = max(new[j]), min(new[i])
        new[i] = frozenset(v for v in new[i] if v < hi_j)
        new[j] = frozenset(v for v in new[j] if v > lo_i)
    return tuple(new)


def red_graph(prob, dom):
    """Min-plus reachability: dom[target] ∩= {values reachable from source in the embedded graph}.
    Sound (the planted semantics: cell's value must be reachable); cross-domain (graph -> csp)."""
    new = list(dom)
    for edges, source, target in prob.graph:
        new[target] = new[target] & _reach(edges, source, prob.d)
    return tuple(new)


# (name, fn, cost, applicable predicate)
BANK = [
    ("ac",       red_ac,       1, lambda p: bool(p.cons)),
    ("ordering", red_ordering, 1, lambda p: bool(p.order)),
    ("graph",    red_graph,    2, lambda p: bool(p.graph)),
    ("alldiff",  red_alldiff,  3, lambda p: bool(p.alldiff)),
    ("pair",     red_pair,     3, lambda p: bool(p.cons)),
    ("factor",   red_factor,   5, lambda p: bool(p.cons)),
]
BANK_BY_NAME = {n: (fn, cost) for (n, fn, cost, _) in BANK}
ACTIONS = [n for (n, _, _, _) in BANK] + ["branch", "stop"]
ACTION_ID = {a: i for i, a in enumerate(ACTIONS)}


def applicable(prob):
    return [(n, fn, cost) for (n, fn, cost, ok) in BANK if ok(prob)]


def to_fixpoint(redfn, prob, dom, max_iters=200):
    """Iterate ONE reduction to its fixpoint (a 'single organ run')."""
    for _ in range(max_iters):
        nd = meet(dom, redfn(prob, dom))
        if nd == dom:
            return dom
        dom = nd
        if any(len(c) == 0 for c in dom):
            return dom
    return dom


def reduced_product(prob, dom, max_rounds=400):
    """Run the WHOLE applicable bank to a joint fixpoint (the reduced product / meet of all organs)."""
    apps = applicable(prob)
    for _ in range(max_rounds):
        changed = False
        for (_, fn, _) in apps:
            nd = meet(dom, fn(prob, dom))
            if nd != dom:
                dom = nd
                changed = True
                if any(len(c) == 0 for c in dom):
                    return dom
        if not changed:
            return dom
    return dom


# ===================================================================== verifier-gated oracle search
def featurize(prob, dom, prev_action):
    """CHEAP state features for the router — static problem shape + current domain sizes + last action.
    NO reduction is run here (no peeking), so a router using these is a true forward-pass policy."""
    n, d = prob.n, prob.d
    n_unary = sum(1 for sc, _ in prob.cons if len(sc) == 1)
    n_bin = sum(1 for sc, _ in prob.cons if len(sc) == 2)
    n_tern = sum(1 for sc, _ in prob.cons if len(sc) >= 3)
    sizes = [len(c) for c in dom]
    open_cells = [s for s in sizes if s > 1]
    feat = [
        float(bool(prob.cons)), float(bool(prob.alldiff)), float(bool(prob.order)), float(bool(prob.graph)),
        n / 8.0, d / 8.0,
        n_unary / 4.0, n_bin / 8.0, n_tern / 8.0,
        len(prob.order) / 8.0, len(prob.alldiff) / 2.0, len(prob.graph) / 2.0,
        sum(1 for s in sizes if s > 1) / max(1, n),          # open fraction
        sum(1 for s in sizes if s == 1) / max(1, n),         # singleton fraction
        sum(sizes) / max(1, n * d),                          # alive fraction
        (min(sizes) if sizes else 0) / max(1, d),
        (max(sizes) if sizes else 0) / max(1, d),
        (min(open_cells) if open_cells else 0) / max(1, d),  # smallest open domain
    ]
    onehot = [0.0] * len(ACTIONS)
    onehot[ACTION_ID.get(prev_action, ACTION_ID["stop"])] = 1.0
    return np.asarray(feat + onehot, np.float32)


FEAT_DIM = None  # set on first featurize call below
def _feat_dim(prob, dom):
    global FEAT_DIM
    if FEAT_DIM is None:
        FEAT_DIM = len(featurize(prob, dom, "stop"))
    return FEAT_DIM


def oracle_search(prob, log=None, allow_branch=True, max_steps=5000):
    """Verifier-gated, deterministic least-cost oracle. At each open state apply the CHEAPEST applicable
    reduction that STRICTLY narrows; if none narrows, BRANCH the smallest open cell (verifier prunes
    conflict subtrees). Returns (solved_dom or None, n_decisions). Logs (features, action) at every
    decision when `log` is given. Sound by construction (every action is certified-sound narrowing)."""
    apps = applicable(prob)
    apps.sort(key=lambda t: t[2])                            # cheapest first
    counter = [0]

    def rec(dom, prev):
        counter[0] += 1
        if counter[0] > max_steps:
            return None
        st = status(dom)
        if st == "conflict":
            return None                                      # GATE: contradiction prunes this subtree
        if st == "solved":
            if log is not None:
                log.append((featurize(prob, dom, prev), "stop"))
            return dom
        # cheapest reduction that strictly narrows
        for (name, fn, _cost) in apps:
            nd = meet(dom, fn(prob, dom))
            if nd != dom:
                if log is not None:
                    log.append((featurize(prob, dom, prev), name))
                return rec(nd, name)
        # propagation stalled & still open -> branch
        if not allow_branch:
            return None
        if log is not None:
            log.append((featurize(prob, dom, prev), "branch"))
        cell = min((i for i in range(prob.n) if len(dom[i]) > 1), key=lambda i: len(dom[i]))
        for v in sorted(dom[cell]):
            nd = tuple(frozenset({v}) if i == cell else dom[i] for i in range(prob.n))
            res = rec(nd, "branch")
            if res is not None:
                return res
        return None

    sol = rec(initial_domain(prob), "stop")
    return sol, counter[0]


# ===================================================================== single-organ baselines
def best_single_organ(prob):
    """Run EACH applicable organ alone to its fixpoint from the initial domain. Returns per-organ
    outcome and the best (solved if any single organ solves)."""
    dom0 = initial_domain(prob)
    per = {}
    any_solved = False
    for (name, fn, _cost, ok) in BANK:
        if not ok(prob):
            continue
        dom = to_fixpoint(fn, prob, dom0)
        st = status(dom)
        per[name] = st
        any_solved = any_solved or (st == "solved")
    return per, any_solved


# ===================================================================== problem generators (composition types)
def _rel(scope, pred, d):
    import itertools as it
    al = frozenset(t for t in it.product(range(d), repeat=len(scope)) if pred(t))
    return (tuple(scope), al)


def _verify_composition(prob):
    """Accept iff: exactly one solution, NO single organ solves it, the reduced product DOES, and the
    oracle search recovers that unique solution soundly. This makes 'composition is necessary' true
    BY CONSTRUCTION and exact-verified."""
    sols = solutions(prob, limit=2)
    if len(sols) != 1:
        return None
    sol = sols[0]
    per, any_solved = best_single_organ(prob)
    if any_solved:
        return None                                          # a single organ already solves -> not a comp problem
    rp = reduced_product(prob, initial_domain(prob))
    if status(rp) != "solved":
        return None                                          # need pure-propagation composition (branch-free, clean)
    if tuple(next(iter(c)) for c in rp) != sol:
        return None
    sol_dom, _ = oracle_search(prob)
    if sol_dom is None or tuple(next(iter(c)) for c in sol_dom) != sol:
        return None
    return {"sol": sol, "single": per}


def gen_ac_alldiff(rng):
    """CSP narrowing THEN all-different. A permutation (AllDifferent over all cells) + a few unary clues
    + binary extensional constraints. ac (sees cons, not distinctness) abstains; alldiff (sees only
    distinctness) abstains; ac∘alldiff closes."""
    n = int(rng.choice([5, 6]))
    d = n
    perm = list(rng.permutation(n))
    cons = []
    # 1-2 unary clues: a small candidate set containing the truth
    for c in rng.choice(n, size=int(rng.choice([1, 2])), replace=False):
        distract = [v for v in range(d) if v != perm[c]]
        rng.shuffle(distract)
        allowed = sorted({perm[c], *distract[:2]})
        cons.append((( int(c),), frozenset((v,) for v in allowed)))
    # a few binary order/inequality constraints consistent with the planted perm
    pairs = [(i, j) for i in range(n) for j in range(n) if i < j]
    rng.shuffle(pairs)
    for (i, j) in pairs[: int(rng.choice([2, 3, 4]))]:
        if perm[i] < perm[j]:
            cons.append(_rel((i, j), lambda t: t[0] < t[1], d))
        else:
            cons.append(_rel((i, j), lambda t: t[0] > t[1], d))
    prob = CProblem(n, d, tuple(cons), alldiff=(tuple(range(n)),), ctype="ac_alldiff")
    return prob


def gen_graph_ac(rng):
    """Graph reachability sub-result THEN narrowing. A path of `!=` (2-colouring, d=2) has two
    solutions -> ac abstains; a small graph pins ONE cell to a value via reachability -> ac then
    propagates the colouring. graph alone pins one cell; ac alone abstains; graph∘ac closes."""
    n = int(rng.choice([5, 6, 7]))
    d = 2
    # planted 2-colouring of a path: alternate from a random start colour
    start = int(rng.integers(2))
    sol = [(start + i) % 2 for i in range(n)]
    cons = [_rel((i, i + 1), lambda t: t[0] != t[1], d) for i in range(n - 1)]
    # graph over the value set {0,1}: a directed edge so reach(source) = {sol[target]} exactly.
    target = int(rng.integers(n))
    tv = sol[target]
    # source value s with an edge s->tv only if s != tv would add the other; keep reach a singleton:
    edges = ((tv, tv),)                                      # self-loop: reach(tv) = {tv}
    graph = ((edges, tv, target),)
    prob = CProblem(n, d, tuple(cons), graph=graph, ctype="graph_ac")
    return prob


def gen_order_alldiff(rng):
    """Ordering feeding all-different. A permutation + a PARTIAL set of order constraints + 1 unary
    clue. ordering bounds alone leave ties; alldiff alone abstains; ordering∘alldiff closes."""
    n = int(rng.choice([5, 6]))
    d = n
    perm = list(rng.permutation(n))
    inv = {perm[i]: i for i in range(n)}                     # cell holding each value
    order = []
    # chain a contiguous run of values: cells holding v < v+1 ... (partial, leaves the rest to alldiff)
    lo = int(rng.integers(0, n - 2))
    runlen = int(rng.choice([2, 3]))
    for v in range(lo, min(lo + runlen, n - 1)):
        order.append((inv[v], inv[v + 1]))                  # val[cell_of v] < val[cell_of v+1]
    cons = []
    c = int(rng.integers(n))                                # one unary clue
    distract = [v for v in range(d) if v != perm[c]]
    rng.shuffle(distract)
    cons.append(((c,), frozenset((v,) for v in sorted({perm[c], *distract[:2]}))))
    prob = CProblem(n, d, tuple(cons), alldiff=(tuple(range(n)),), order=tuple(order), ctype="order_alldiff")
    return prob


GEN = {"ac_alldiff": gen_ac_alldiff, "graph_ac": gen_graph_ac, "order_alldiff": gen_order_alldiff}


def gen_single_solvable(rng):
    """CONTROL: problems a SINGLE organ solves (sanity that the table is a fair comparison)."""
    kind = rng.choice(["ac_chain", "alldiff_pin"])
    if kind == "ac_chain":                                  # chain of equalities pinned at one end: ac solves
        n = int(rng.choice([4, 5]))
        d = 2
        cons = [((0,), frozenset({(0,)}))]
        cons += [_rel((i, i + 1), lambda t: t[0] == t[1], d) for i in range(n - 1)]
        return CProblem(n, d, tuple(cons), ctype="ctrl_ac")
    else:                                                   # permutation with all-but-one pinned: alldiff solves
        n = int(rng.choice([5, 6]))
        d = n
        perm = list(rng.permutation(n))
        cons = [((c,), frozenset({(perm[c],)})) for c in range(n - 1)]
        return CProblem(n, d, tuple(cons), alldiff=(tuple(range(n)),), ctype="ctrl_alldiff")


def make_corpus(rng, per_type, types=("ac_alldiff", "graph_ac", "order_alldiff")):
    """Sample composition problems, keeping only those that REQUIRE composition (verified)."""
    corpus = {}
    for t in types:
        kept = []
        tries = 0
        while len(kept) < per_type and tries < per_type * 200:
            tries += 1
            prob = GEN[t](rng)
            info = _verify_composition(prob)
            if info is not None:
                kept.append((prob, info))
        corpus[t] = kept
    return corpus


# ===================================================================== router (distill the oracle routes)
def collect_routes(corpus):
    """Run the oracle on every composition problem, logging (features, action) decisions."""
    data = []                                                # (feat, action_id, ctype)
    for t, items in corpus.items():
        for (prob, _info) in items:
            log = []
            oracle_search(prob, log=log)
            for (feat, action) in log:
                data.append((feat, ACTION_ID[action], t))
    return data


def train_router(train_data, val_data, heldout_data, feat_dim, steps=600, seed=0, quiet=False):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    K = len(ACTIONS)

    def to_xy(data):
        X = torch.as_tensor(np.stack([f for f, _, _ in data]), device=dev)
        y = torch.as_tensor(np.asarray([a for _, a, _ in data], np.int64), device=dev)
        return X, y

    Xtr, ytr = to_xy(train_data)
    Xva, yva = to_xy(val_data)
    model = nn.Sequential(nn.Linear(feat_dim, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
                          nn.Linear(64, K)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    lossf = nn.CrossEntropyLoss()
    best = {"val_acc": -1.0}
    for s in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, Xtr.shape[0], (min(256, Xtr.shape[0]),), device=dev)
        logits = model(Xtr[idx])
        loss = lossf(logits, ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 50 == 0 or s == 1:
            model.eval()
            with torch.no_grad():
                va = float((model(Xva).argmax(-1) == yva).float().mean()) if len(val_data) else float("nan")
            if va >= best["val_acc"]:
                best = {"val_acc": va, "state": {k: v.clone() for k, v in model.state_dict().items()}}
            if not quiet:
                print(f"  router step {s:4d}  loss {float(loss):.3f}  val_acc {va*100:5.1f}%", flush=True)
    if "state" in best:
        model.load_state_dict(best["state"])
    model.eval()

    def acc(data):
        if not data:
            return float("nan")
        X, y = to_xy(data)
        with torch.no_grad():
            return float((model(X).argmax(-1) == y).float().mean())

    return model, {"train_acc": acc(train_data), "val_acc": acc(val_data),
                   "heldout_acc": acc(heldout_data)}


def router_forward_solve(model, prob, max_steps=200):
    """Composition with NO SEARCH: each step the router (forward pass) picks ONE applicable organ; we
    apply it, gated by soundness (skip if it would empty a cell? no — narrowing is always sound; a
    conflict just means unsat). Stops at solved/stall. Returns (solved, steps)."""
    import torch
    dev = next(model.parameters()).device
    app_names = {n for (n, _, _) in applicable(prob)}
    dom = initial_domain(prob)
    prev = "stop"
    for step in range(1, max_steps + 1):
        st = status(dom)
        if st in ("solved", "conflict"):
            return st == "solved", step
        feat = featurize(prob, dom, prev)
        with torch.no_grad():
            logits = model(torch.as_tensor(feat, device=dev).unsqueeze(0))[0].cpu().numpy()
        # mask to applicable reductions (router runs forward; applicability is a hard type-check)
        order = np.argsort(-logits)
        acted = False
        for ai in order:
            name = ACTIONS[ai]
            if name in ("stop", "branch"):
                continue
            if name not in app_names:
                continue
            nd = meet(dom, BANK_BY_NAME[name][0](prob, dom))
            if nd != dom:
                dom = nd
                prev = name
                acted = True
                break
        if not acted:
            return status(dom) == "solved", step             # router stalled (no pick narrowed)
    return status(dom) == "solved", max_steps


def random_forward_solve(rng, prob, max_steps=200):
    """Baseline: pick a RANDOM applicable reduction each step. Confluence means it eventually reaches
    the joint fixpoint, but with wasted steps — the control the router must beat on efficiency."""
    apps = applicable(prob)
    dom = initial_domain(prob)
    for step in range(1, max_steps + 1):
        st = status(dom)
        if st in ("solved", "conflict"):
            return st == "solved", step
        order = list(range(len(apps)))
        rng.shuffle(order)
        acted = False
        for ai in order:
            nd = meet(dom, apps[ai][1](prob, dom))
            if nd != dom:
                dom = nd
                acted = True
                break
        if not acted:
            return status(dom) == "solved", step
    return status(dom) == "solved", max_steps


# ===================================================================== smoke
def smoke():
    print("== SMOKE: the certified bank composes under verifier-gated search to solve a 2-organ problem ==\n")
    rng = np.random.default_rng(0)

    # a graph_ac instance: graph-reachability ∘ csp-AC (cross-domain composition)
    prob = None
    for _ in range(500):
        cand = gen_graph_ac(rng)
        if _verify_composition(cand) is not None:
            prob = cand
            break
    assert prob is not None, "failed to sample a composition instance"
    sols = solutions(prob)
    per, any_solved = best_single_organ(prob)
    sol_dom, ndec = oracle_search(prob)
    rp = reduced_product(prob, initial_domain(prob))
    print(f"  graph_ac  n={prob.n} d={prob.d}  unique-solution={len(sols)==1}  truth={sols[0]}")
    print(f"  single-organ outcomes: {per}   (any single solves: {any_solved})")
    print(f"  reduced-product (all organs): {status(rp)}")
    print(f"  oracle search: solved={sol_dom is not None}  answer={tuple(next(iter(c)) for c in sol_dom)}  "
          f"decisions={ndec}")
    assert not any_solved, "smoke instance must NOT be single-organ solvable"
    assert sol_dom is not None and tuple(next(iter(c)) for c in sol_dom) == sols[0], "composer must solve, exactly"

    # soundness check: the unique solution is preserved at every reduced-product round (never dropped)
    dom = initial_domain(prob)
    truth = sols[0]
    sound = all(truth[i] in dom[i] for i in range(prob.n))
    apps = applicable(prob)
    for _ in range(50):
        nd = dom
        for (_, fn, _) in apps:
            nd = meet(nd, fn(prob, nd))
        sound = sound and all(truth[i] in nd[i] for i in range(prob.n))
        if nd == dom:
            break
        dom = nd
    print(f"  soundness: true solution survives all narrowing = {sound}")
    assert sound, "narrowing must never drop the true solution (certified soundness)"
    print("\n  SMOKE OK — composition is sound, and solves where no single organ does.\n")


# ===================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--per_type", type=int, default=120)
    ap.add_argument("--ctrl", type=int, default=60)
    ap.add_argument("--router_steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    smoke()
    if args.smoke:
        return

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    print("generating composition corpus (filtered to composition-REQUIRING instances) ...", flush=True)
    corpus = make_corpus(rng, args.per_type)
    for t, items in corpus.items():
        print(f"  {t:14s}: {len(items)} instances", flush=True)
    ctrl = [gen_single_solvable(rng) for _ in range(args.ctrl)]
    print(f"  controls (single-organ solvable): {len(ctrl)}   (gen wall {time.time()-t0:.1f}s)\n", flush=True)

    # ---------------- COMPOSITION TABLE: composer vs best-single-organ ----------------
    print("================ COMPOSITION-SOLVING TABLE (composer vs best single organ) ================")
    print(f"  {'type':16s} {'n':>4} {'composer_solved':>16} {'best_single_solved':>19} {'reduced_prod':>13}")
    table = {}
    for t, items in corpus.items():
        comp_ok = single_ok = rp_ok = 0
        for (prob, _info) in items:
            sol_dom, _ = oracle_search(prob)
            comp_ok += int(sol_dom is not None and status(sol_dom) == "solved")
            _, any_solved = best_single_organ(prob)
            single_ok += int(any_solved)
            rp_ok += int(status(reduced_product(prob, initial_domain(prob))) == "solved")
        nN = max(1, len(items))
        table[t] = {"n": len(items), "composer": comp_ok / nN, "best_single": single_ok / nN,
                    "reduced_product": rp_ok / nN}
        print(f"  {t:16s} {len(items):>4} {comp_ok}/{len(items):>4} ({comp_ok/nN*100:5.1f}%)   "
              f"{single_ok}/{len(items):>4} ({single_ok/nN*100:5.1f}%)   {rp_ok/nN*100:5.1f}%", flush=True)
    # controls
    c_comp = sum(int((oracle_search(p)[0] is not None)) for p in ctrl)
    c_single = sum(int(best_single_organ(p)[1]) for p in ctrl)
    print(f"  {'CONTROL(single)':16s} {len(ctrl):>4} {c_comp}/{len(ctrl)} ({c_comp/max(1,len(ctrl))*100:5.1f}%)   "
          f"{c_single}/{len(ctrl)} ({c_single/max(1,len(ctrl))*100:5.1f}%)", flush=True)

    # ---------------- ROUTER DISTILLATION ----------------
    print("\n================ ROUTER DISTILLATION (imitate the oracle routes) ================")
    feat_dim = _feat_dim(*( (corpus['ac_alldiff'][0][0], initial_domain(corpus['ac_alldiff'][0][0])) ))
    types = list(corpus.keys())
    router_results = {}

    # (A) POOLED instance-level generalisation: train on 80% of EVERY type's instances (full action
    #     vocabulary seen), test route-prediction on the held-out 20% instances of every type.
    pooled_train, pooled_val = {}, {}
    for t in types:
        items = corpus[t]
        k = max(1, int(0.8 * len(items)))
        pooled_train[t] = items[:k]
        pooled_val[t] = items[k:]
    ptr, pva = collect_routes(pooled_train), collect_routes(pooled_val)
    _, pooled_acc = train_router(ptr, pva, [], feat_dim, steps=args.router_steps, seed=args.seed, quiet=True)
    print(f"\n  (A) POOLED in-distribution (all types, full action vocab): "
          f"train route-acc {pooled_acc['train_acc']*100:5.1f}%   held-out-instances {pooled_acc['val_acc']*100:5.1f}%")
    router_results["pooled"] = pooled_acc

    # (B) LEAVE-ONE-COMPOSITION-TYPE-OUT: the hard transfer test (held-out type may need an organ-action
    #     ABSENT from the training types' routes — a structural ceiling on cross-type route transfer).
    print("  (B) leave-one-composition-type-out (cross-type transfer):")
    for heldout in types:                                   # leave-one-composition-type-out generalisation
        train_types = [t for t in types if t != heldout]
        # split each train type's instances 80/20 by PROBLEM
        train_corpus, val_corpus = {}, {}
        for t in train_types:
            items = corpus[t]
            k = max(1, int(0.8 * len(items)))
            train_corpus[t] = items[:k]
            val_corpus[t] = items[k:]
        train_data = collect_routes(train_corpus)
        val_data = collect_routes(val_corpus)
        heldout_data = collect_routes({heldout: corpus[heldout]})
        # structural ceiling: fraction of held-out decisions whose ORACLE action was ever seen in training
        train_actions = {a for _, a, _ in train_data}
        vocab_cov = (sum(1 for _, a, _ in heldout_data if a in train_actions) / max(1, len(heldout_data)))
        model, acc = train_router(train_data, val_data, heldout_data, feat_dim,
                                  steps=args.router_steps, seed=args.seed, quiet=True)
        # forward-pass (no search) solve rate on the held-out type, vs oracle and random
        rng_f = np.random.default_rng(args.seed + 1)
        fwd_solved = fwd_steps = orc_solved = rnd_solved = rnd_steps = 0
        ho_items = corpus[heldout]
        for (prob, _info) in ho_items:
            s, st = router_forward_solve(model, prob)
            fwd_solved += int(s); fwd_steps += st
            orc_solved += int(oracle_search(prob)[0] is not None)
            rs, rst = random_forward_solve(rng_f, prob)
            rnd_solved += int(rs); rnd_steps += rst
        nH = max(1, len(ho_items))
        router_results[heldout] = {
            "train_acc": acc["train_acc"], "val_acc": acc["val_acc"], "heldout_route_acc": acc["heldout_acc"],
            "fwd_solve_rate": fwd_solved / nH, "fwd_avg_steps": fwd_steps / nH,
            "oracle_solve_rate": orc_solved / nH,
            "rand_solve_rate": rnd_solved / nH, "rand_avg_steps": rnd_steps / nH,
            "heldout_action_vocab_seen": vocab_cov,
        }
        r = router_results[heldout]
        print(f"\n  held-out type = {heldout}  (train on {train_types})")
        print(f"    route-prediction acc:  train {r['train_acc']*100:5.1f}%  val {r['val_acc']*100:5.1f}%  "
              f"HELD-OUT-TYPE {r['heldout_route_acc']*100:5.1f}%  "
              f"(held-out actions seen in training: {vocab_cov*100:.0f}%)")
        print(f"    forward-pass (no search) on held-out type: solve {r['fwd_solve_rate']*100:5.1f}% "
              f"(avg {r['fwd_avg_steps']:.1f} steps)   oracle {r['oracle_solve_rate']*100:5.1f}%   "
              f"random {r['rand_solve_rate']*100:5.1f}% (avg {r['rand_avg_steps']:.1f} steps)")

    out = Path(args.out or (Path(__file__).resolve().parent.parent / "runs" / "composer.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "table": table,
                               "controls": {"composer": c_comp / max(1, len(ctrl)),
                                            "best_single": c_single / max(1, len(ctrl))},
                               "router": router_results}, indent=2))
    print(f"\nwrote {out}")
    print(f"\ntotal wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
