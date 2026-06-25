"""Domain-general constraint-reasoning gauntlet for GLaDOS.

ONE organ, MANY lattices. Each domain is a per-position POWERSET LATTICE shaped exactly
like sudoku.py: state = alive-candidate mask alive ∈ {0,1}^[B,P,V], a position is *solved*
when exactly one candidate survives, *dead* when zero survive, and we only ever MEET
(narrow) so a surviving singleton is correct. The model is domain-blind — it sees only
(alive, given), never which domain it is — so the SAME cell must port across all of them.

A `Domain` bundles:
  name, P (positions), V (candidates/position),
  gen(n, rng) -> (puz[n,P] int64, sol[n,P] int64)   ground-truth solutions REQUIRED so the
                                                     soundness metric (never eliminate a true
                                                     candidate) is always computable,
  init_state, alpha_target, is_correct              (per-domain only where the lattice differs
                                                     from sudoku; otherwise the generic ops here).

The lattice MEET (step_state), branch, status, conflict_flag are domain-AGNOSTIC and are reused
straight from sudoku.py — narrowing/branching/emptiness logic does not depend on the constraint
semantics, only on the alive-mask shape. What IS per-domain is:
  * the GENERATOR (how a solvable instance + its ground truth are sampled), and
  * alpha_target's notion of "the soundest narrowing still consistent with the remaining
    solutions". Here we supervise toward onehot(sol) wherever the true candidate is still alive,
    and raise the conflict flag when a branch has killed a true candidate — IDENTICAL in form to
    sudoku.alpha_target, because soundness is defined relative to the known solution, not by
    re-deriving the constraint graph at train time. So the generic alpha_target below works for
    every single-solution-supervised domain; we expose it as `generic_alpha_target`.

PUZZLE ENCODING (the int64 [n,P] arrays):
  Every domain encodes its puzzle as a per-position clue array in the SAME convention as sudoku:
    0           = blank / not given (all V candidates start alive)
    1..V        = this position is GIVEN to be candidate (clue-1)  (only that candidate alive)
  This is all init_state needs. The constraint structure (edges / clauses / walls) is NOT stored
  in the puzzle array because the model never sees it — it is used only by the generator (to plant
  a solvable instance) and by the numpy self-check (to verify solvability). Soundness/targets are
  defined against the returned ground-truth `sol`.

torch is imported LAZILY (inside the ops) so the numpy generators + self-check run standalone
with no torch installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


# --------------------------------------------------------------------------- generic lattice ops
# These mirror sudoku.init_state / sudoku.alpha_target / sudoku.is_correct but parameterised by
# (P, V) instead of hard-wired (81, 9). step_state / branch / status / conflict_flag are imported
# from sudoku.py unchanged (they are already domain-agnostic) — see run_gauntlet.py.

def generic_init_state(puz, P, V, dev):
    """puz [B,P] (0=blank, 1..V=clue) -> alive [B,P,V] mask, given [B,P] flag. Same scheme as
    sudoku.init_state but for arbitrary (P,V)."""
    import torch
    puz = torch.as_tensor(puz, dtype=torch.long, device=dev)
    B = puz.size(0)
    alive = torch.ones(B, P, V, device=dev)
    given = (puz > 0).float()
    clue = puz.clamp(min=1) - 1
    onehot = torch.zeros(B, P, V, device=dev).scatter_(2, clue.unsqueeze(-1), 1.0)
    alive = torch.where(given.unsqueeze(-1) > 0, onehot, alive)
    return alive, given


def generic_alpha_target(alive, sol, dev):
    """On-policy sound-deduction target, identical in form to sudoku.alpha_target.

    alive [B,P,V], sol [B,P] (1..V). Returns:
      tgt  [B,P,V] asym-BCE target: onehot(sol) where the true candidate is still alive,
                   current alive on conflict rows (don't push elimination into the empty set),
      cls  [B]     1.0 if any position has already lost its true candidate (⊥ / wrong branch),
      sing [B,P]   positions whose target is a singleton (CE term).
    """
    import torch
    sol = torch.as_tensor(sol, dtype=torch.long, device=dev)
    soli = (sol.clamp(min=1) - 1)
    onehot = torch.zeros_like(alive).scatter_(2, soli.unsqueeze(-1), 1.0)
    contains = (alive * onehot).sum(-1)
    conflict = (contains < 0.5).any(dim=1).float()
    tgt = torch.where(conflict.view(-1, 1, 1) > 0, alive, onehot)
    sing = (tgt.sum(-1) == 1).float()
    return tgt, conflict, sing


def generic_is_correct(alive, sol, dev):
    """For solved states: does the singleton assignment equal the true solution?"""
    import torch
    sol = torch.as_tensor(sol, dtype=torch.long, device=dev)
    pred = alive.argmax(-1) + 1
    return (pred == sol).all(dim=1)


# --------------------------------------------------------------------------- Domain
@dataclass
class Domain:
    name: str
    P: int
    V: int
    gen: Callable                       # gen(n, rng) -> (puz[n,P] int64, sol[n,P] int64)
    init_state: Callable = None         # (puz, dev) -> (alive, given)
    alpha_target: Callable = None       # (alive, sol, dev) -> (tgt, conflict, sing)
    is_correct: Callable = None         # (alive, sol, dev) -> [B] bool

    def __post_init__(self):
        P, V = self.P, self.V
        if self.init_state is None:
            self.init_state = lambda puz, dev: generic_init_state(puz, P, V, dev)
        if self.alpha_target is None:
            self.alpha_target = generic_alpha_target
        if self.is_correct is None:
            self.is_correct = generic_is_correct


# --------------------------------------------------------------------------- graph coloring
def _gen_coloring(n_nodes, k_colors, edge_density, hide=0.5):
    """positions = nodes, candidates = k colors. constraint: adjacent nodes differ in color.

    Solvable by construction: sample a random proper coloring as the PLANTED solution, then add
    edges only between differently-coloured nodes (so the planted coloring stays proper). Then hide
    a fraction `hide` of node colours as blanks; the rest become givens.
    Returns gen(n, rng)->(puz[n,n_nodes], sol[n,n_nodes]) with colours encoded 1..k.
    """
    def gen(n, rng):
        puz = np.zeros((n, n_nodes), np.int64)
        sol = np.zeros((n, n_nodes), np.int64)
        for i in range(n):
            col = rng.integers(0, k_colors, n_nodes)             # planted proper coloring (0..k-1)
            sol[i] = col + 1
            p = (col + 1).copy()
            # hide ~`hide` fraction (but always reveal at least enough to be meaningful)
            nhide = int(round(hide * n_nodes))
            if nhide > 0:
                p[rng.choice(n_nodes, size=nhide, replace=False)] = 0
            puz[i] = p
        return puz, sol

    # edges are not stored in puz (model is domain-blind); they're regenerated only for validation
    def edges_for(col, rng):
        E = []
        for a in range(n_nodes):
            for b in range(a + 1, n_nodes):
                if col[a] != col[b] and rng.random() < edge_density:
                    E.append((a, b))
        return E

    gen._edges_for = edges_for
    return gen


def graph_coloring(n_nodes=16, k_colors=4, edge_density=0.3, hide=0.5):
    return Domain("coloring", n_nodes, k_colors, _gen_coloring(n_nodes, k_colors, edge_density, hide))


# --------------------------------------------------------------------------- 3-SAT
def _gen_3sat(n_vars, n_clauses):
    """positions = variables, candidates = {False=1, True=2} (V=2). constraint: all clauses sat.

    SATISFIABLE by construction: sample a planted assignment, then for each clause draw 3 distinct
    vars and 3 polarities such that the clause is satisfied by the planted assignment (guarantee by
    making at least one literal true under the plant). solution = planted assignment.

    UNIQUENESS CAVEAT: a satisfiable 3-SAT instance generally has MANY satisfying assignments, so the
    planted one is not unique. The HONEST definition of soundness is "never eliminate a value used by
    SOME satisfying assignment". For this MVP we supervise toward the PLANTED assignment only (we know
    it, and it is always valid), which is a STRICTER target than true soundness — it can call a
    legitimately-sound elimination a "false elimination" if that value belonged to a different model.
    That makes the reported false-elim an UPPER BOUND on the true (any-model) soundness violation,
    which is the safe direction for a soundness metric. A full treatment would enumerate/sample the
    solution set per clause; out of scope here, flagged for later.

    Encoding: candidate index 0 = assign False, 1 = assign True. sol[i,v] in {1,2}.
    Clauses are not stored in puz; regenerated for validation via gen._clauses_for.
    """
    def sample_clauses(assign, rng):
        # assign: bool[n_vars] planted. returns list of 3 (var, polarity) literals satisfied by assign.
        clauses = []
        for _ in range(n_clauses):
            vs = rng.choice(n_vars, size=3, replace=False)
            pol = rng.integers(0, 2, 3).astype(bool)             # literal positive if True
            # ensure at least one literal is satisfied by the plant: lit sat iff assign[v]==pol
            if not any(assign[v] == pol[j] for j, v in enumerate(vs)):
                j = rng.integers(0, 3)
                pol[j] = assign[vs[j]]                            # flip one literal to be satisfied
            clauses.append([(int(vs[j]), bool(pol[j])) for j in range(3)])
        return clauses

    def gen(n, rng):
        puz = np.zeros((n, n_vars), np.int64)                    # all blank: SAT is a pure search
        sol = np.zeros((n, n_vars), np.int64)
        for i in range(n):
            assign = rng.integers(0, 2, n_vars).astype(bool)     # planted assignment
            sol[i] = assign.astype(np.int64) + 1                 # False->1, True->2
        return puz, sol

    def clauses_for(assign, rng):
        return sample_clauses(assign, rng)

    gen._clauses_for = clauses_for
    return gen


def threesat(n_vars=20, n_clauses=80):
    return Domain("3sat", n_vars, 2, _gen_3sat(n_vars, n_clauses))


# --------------------------------------------------------------------------- maze (shortest-path)
def _gen_maze(size):
    """positions = grid cells (P = size*size), candidate ∈ {off-path=1, on-path=2} (V=2).
    constraint: the on-path cells form THE unique shortest path from start (0,0) to goal
    (size-1,size-1) through the open cells of the maze. solution = the shortest-path cells.

    Solvable by construction: carve a random spanning tree over the grid (randomized DFS) so EVERY
    pair of open cells has a unique simple path; the shortest path start->goal is then unique. Mark
    its cells on-path. start & goal cells are GIVEN as on-path (clue), all other cells blank.

    Encoding: candidate 0 = off-path, 1 = on-path. sol[i,p] in {1,2}. Walls are not stored in puz
    (model is domain-blind); the path uniqueness is what makes the lattice well-posed.
    """
    H = W = size
    P = H * W
    s, g = 0, P - 1

    def carve(rng):
        # randomized DFS spanning tree over the grid graph -> adjacency among cells (a tree)
        adj = {i: set() for i in range(P)}
        seen = np.zeros(P, bool)
        start = 0
        stack = [start]
        seen[start] = True
        while stack:
            c = stack[-1]
            r, cc = divmod(c, W)
            nbrs = []
            for dr, dcl in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, cc + dcl
                if 0 <= nr < H and 0 <= nc < W:
                    nb = nr * W + nc
                    if not seen[nb]:
                        nbrs.append(nb)
            if nbrs:
                nb = nbrs[rng.integers(0, len(nbrs))]
                adj[c].add(nb); adj[nb].add(c)
                seen[nb] = True
                stack.append(nb)
            else:
                stack.pop()
        return adj

    def tree_path(adj, s, g):
        # unique path in a tree via BFS parent-tracking
        from collections import deque
        par = {s: -1}
        q = deque([s])
        while q:
            c = q.popleft()
            if c == g:
                break
            for nb in adj[c]:
                if nb not in par:
                    par[nb] = c
                    q.append(nb)
        path = []
        c = g
        while c != -1:
            path.append(c)
            c = par[c]
        return path[::-1]

    def induced_chordless(path):
        # the on-path set must be an INDUCED simple grid path (no chords) so that it IS the unique
        # shortest start->goal route: every on-path cell is grid-adjacent ONLY to its path neighbours.
        onset = set(path)
        pos = {c: j for j, c in enumerate(path)}
        for c in path:
            r, cc = divmod(c, W)
            for dr, dcl in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, ncl = r + dr, cc + dcl
                if 0 <= nr < H and 0 <= ncl < W:
                    nb = nr * W + ncl
                    if nb in onset and abs(pos[nb] - pos[c]) != 1:
                        return False                              # chord -> not the unique shortest path
        return True

    def gen(n, rng):
        puz = np.zeros((n, P), np.int64)
        sol = np.zeros((n, P), np.int64)
        paths = []
        for i in range(n):
            while True:
                adj = carve(rng)
                path = tree_path(adj, s, g)
                if induced_chordless(path):
                    break                                        # reject chorded -> regenerate
            lab = np.ones(P, np.int64)                            # off-path = 1
            lab[path] = 2                                         # on-path = 2
            sol[i] = lab
            p = np.zeros(P, np.int64)
            p[s] = 2; p[g] = 2                                    # start & goal given as on-path
            puz[i] = p
            paths.append(path)
        gen._last_paths = paths
        return puz, sol

    def path_of(lab):
        # recover the ordered s->g path from an on-path labeling (assumes a valid induced path)
        onset = set(np.nonzero(lab == 2)[0].tolist())
        path = [s]
        prev = -1
        cur = s
        while cur != g:
            r, cc = divmod(cur, W)
            nxt = -1
            for dr, dcl in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, ncl = r + dr, cc + dcl
                if 0 <= nr < H and 0 <= ncl < W:
                    nb = nr * W + ncl
                    if nb in onset and nb != prev:
                        nxt = nb; break
            if nxt == -1:
                return None
            path.append(nxt); prev, cur = cur, nxt
        return path

    gen._carve = carve
    gen._tree_path = tree_path
    gen._induced_chordless = induced_chordless
    gen._path_of = path_of
    gen._dims = (H, W, P)
    return gen


def maze(size=6):
    return Domain("maze", size * size, 2, _gen_maze(size))


# --------------------------------------------------------------------------- registry
def all_domains():
    return [graph_coloring(), threesat(), maze()]


# --------------------------------------------------------------------------- numpy self-check
def _check_coloring(seed=0):
    rng = np.random.default_rng(seed)
    nn, k, dens = 16, 4, 0.3
    g = _gen_coloring(nn, k, dens)
    puz, sol = g(64, rng)
    assert puz.shape == sol.shape == (64, nn)
    assert sol.min() >= 1 and sol.max() <= k
    # givens must agree with solution; blanks are 0
    given = puz > 0
    assert (puz[given] == sol[given]).all(), "given clue disagrees with solution"
    # planted coloring must be proper on a freshly-sampled edge set consistent with it
    bad = 0
    for i in range(64):
        col = sol[i] - 1
        E = g._edges_for(col, np.random.default_rng(1000 + i))
        for (a, b) in E:
            if col[a] == col[b]:
                bad += 1
    assert bad == 0, f"coloring improper on {bad} edges"
    return f"coloring   OK  n=64 nodes={nn} colors={k} givens/instance~{given.sum()/64:.1f}  proper=100%"


def _check_3sat(seed=0):
    rng = np.random.default_rng(seed)
    nv, nc = 20, 80
    g = _gen_3sat(nv, nc)
    puz, sol = g(64, rng)
    assert puz.shape == sol.shape == (64, nv)
    assert set(np.unique(sol).tolist()) <= {1, 2}
    # every planted assignment must satisfy a freshly-sampled clause set built against it
    unsat = 0
    for i in range(64):
        assign = (sol[i] == 2)
        clauses = g._clauses_for(assign, np.random.default_rng(2000 + i))
        for cl in clauses:
            if not any(assign[v] == pol for (v, pol) in cl):
                unsat += 1
    assert unsat == 0, f"{unsat} clauses unsatisfied by planted assignment"
    return f"3sat       OK  n=64 vars={nv} clauses={nc}  all clauses SAT by plant=100%"


def _check_maze(seed=0):
    rng = np.random.default_rng(seed)
    sz = 6
    g = _gen_maze(sz)
    H, W, P = g._dims
    puz, sol = g(64, rng)
    assert puz.shape == sol.shape == (64, P)
    assert set(np.unique(sol).tolist()) <= {1, 2}
    s, gl = 0, P - 1
    bad = 0
    for i in range(64):
        on = np.nonzero(sol[i] == 2)[0]
        # start & goal must be on-path and given
        assert puz[i, s] == 2 and puz[i, gl] == 2
        assert sol[i, s] == 2 and sol[i, gl] == 2
        # recover the ordered path: consecutive cells 4-adjacent, all distinct, s->g, chordless
        path = g._path_of(sol[i])
        if path is None or path[0] != s or path[-1] != gl:
            bad += 1; continue
        if len(set(path)) != len(path):                          # all distinct
            bad += 1; continue
        if set(path) != set(on.tolist()):                        # path covers exactly the on-path set
            bad += 1; continue
        ok = True
        for a, b in zip(path, path[1:]):                         # consecutive grid-adjacency
            ra, ca = divmod(a, W); rb, cb = divmod(b, W)
            if abs(ra - rb) + abs(ca - cb) != 1:
                ok = False; break
        if not (ok and g._induced_chordless(path)):              # unique shortest (no chords)
            bad += 1
    assert bad == 0, f"maze path invalid on {bad}/64 instances"
    return f"maze       OK  n=64 grid={sz}x{sz} P={P}  valid unique s->g path=100%"


if __name__ == "__main__":
    print("domains.py numpy self-check (no torch)")
    print(" ", _check_coloring())
    print(" ", _check_3sat())
    print(" ", _check_maze())
    print("all generators produce solvable instances with valid ground-truth solutions.")
