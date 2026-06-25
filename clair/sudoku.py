"""Sudoku as a lattice-deduction domain for GLaDOS.

State = per-cell alive-candidate mask  alive ∈ {0,1}^[B,81,9].
A cell is *solved* when exactly one candidate is alive; the grid is *solved* when all
cells are singletons (and, since we only ever narrow soundly, that singleton IS correct).
A cell is *dead* (conflict) when zero candidates are alive.

Two things make this lattice deduction and not answer-memorization:
  * we only ever MEET (alive can lose candidates, never gain) — monotone descent;
  * supervision is the on-policy ALPHA TARGET: from whatever partially-deduced (and possibly
    wrongly-branched) state the model actually reached, the target is the soundest narrowing
    still consistent with the remaining solutions — or a conflict flag if a branch killed the
    true digit. We know the unique solution, so we can compute it exactly.
"""
from __future__ import annotations

import os

import numpy as np
import torch


# --------------------------------------------------------------------------- data
def _solved_grid(rng):
    g = np.zeros((9, 9), int)

    def ok(r, c, v):
        if v in g[r] or v in g[:, c]:
            return False
        br, bc = 3 * (r // 3), 3 * (c // 3)
        return v not in g[br:br + 3, bc:bc + 3]

    def fill(i=0):
        if i == 81:
            return True
        r, c = divmod(i, 9)
        for v in rng.permutation(np.arange(1, 10)):
            if ok(r, c, v):
                g[r, c] = v
                if fill(i + 1):
                    return True
                g[r, c] = 0
        return False
    fill()
    return g


def gen_puzzles(n, rng, holes=51):
    """Fallback generator (NOT uniqueness-checked) for offline smoke tests.
    Real runs use load_sudoku_extreme()."""
    puz = np.zeros((n, 81), np.int64)
    sol = np.zeros((n, 81), np.int64)
    for i in range(n):
        g = _solved_grid(rng).flatten()
        sol[i] = g
        p = g.copy()
        p[rng.choice(81, size=holes, replace=False)] = 0
        puz[i] = p
    return puz, sol


def load_sudoku_extreme(split="train", limit=None):
    """Load the Sudoku-Extreme benchmark (HRM/LDT). Tries common HF sources; each row is an
    81-char puzzle ('0'/'.'=blank) + 81-char solution. Returns (puz[N,81], sol[N,81]) int64."""
    from datasets import load_dataset
    last = None
    for spec in [("sapientinc/sudoku-extreme", None), ("Ritvik19/Sudoku-Extreme", None)]:
        try:
            ds = load_dataset(spec[0], name=spec[1], split=split, streaming=False)
            break
        except Exception as e:  # noqa: BLE001
            last = e; ds = None
    if ds is None:
        raise RuntimeError(f"could not load Sudoku-Extreme: {last}")
    cols = ds.column_names
    pk = next(c for c in cols if c.lower() in ("question", "puzzle", "quizzes", "puzzles"))
    sk = next(c for c in cols if c.lower() in ("answer", "solution", "solutions", "target"))
    rows = ds if limit is None else ds.select(range(min(limit, len(ds))))

    def parse(s):
        return np.array([0 if ch in "0." else int(ch) for ch in s.strip()], np.int64)
    puz = np.stack([parse(r[pk]) for r in rows])
    sol = np.stack([parse(r[sk]) for r in rows])
    return puz, sol


# --------------------------------------------------------------------------- lattice
def init_state(puz, dev):
    """puz [B,81] (0=blank,1-9=clue) -> alive [B,81,9] mask, given [B,81] flag."""
    puz = torch.as_tensor(puz, dtype=torch.long, device=dev)
    B = puz.size(0)
    alive = torch.ones(B, 81, 9, device=dev)
    given = (puz > 0).float()
    clue = puz.clamp(min=1) - 1                       # candidate index of the clue
    onehot = torch.zeros(B, 81, 9, device=dev).scatter_(2, clue.unsqueeze(-1), 1.0)
    alive = torch.where(given.unsqueeze(-1) > 0, onehot, alive)   # clue cells: only the clue alive
    return alive, given


def alpha_target(alive, sol, dev):
    """The on-policy sound-deduction target for the CURRENT (on-policy) alive state.

    alive [B,81,9], sol [B,81] (1-9). Returns:
      tgt   [B,81,9]  asym-BCE target (the soundest narrowing still consistent: onehot(sol)
                       where the state still contains the solution; current alive on conflict rows
                       so we don't push elimination into the empty set),
      cls   [B]       1.0 if any cell has already lost its true candidate (⊥ / branch was wrong),
      sing  [B,81]    cells whose target is a singleton (for the CE term).
    """
    sol = torch.as_tensor(sol, dtype=torch.long, device=dev)
    soli = (sol.clamp(min=1) - 1)
    onehot = torch.zeros_like(alive).scatter_(2, soli.unsqueeze(-1), 1.0)
    contains = (alive * onehot).sum(-1)               # [B,81] 1 if true cand still alive
    conflict = (contains < 0.5).any(dim=1).float()    # any cell lost its true digit -> unsat
    # satisfiable rows: target = onehot(sol). conflict rows: target = current alive (no new pressure).
    tgt = torch.where(conflict.view(-1, 1, 1) > 0, alive, onehot)
    sing = (tgt.sum(-1) == 1).float()
    return tgt, conflict, sing


@torch.no_grad()
def step_state(alive, given, b_logits, theta_elim=0.1):
    """MEET (monotone narrowing): drop currently-alive candidates whose survival prob < theta.
    Protect given clues, and never let thresholding spuriously empty a cell — always keep the
    highest-logit *currently-alive* candidate. Genuine unsatisfiability is reported by the CLS
    conflict head, not by emptiness (see conflict_flag)."""
    keep = (torch.sigmoid(b_logits) >= theta_elim).float()
    new = alive * keep
    new = torch.where(given.unsqueeze(-1) > 0, alive, new)       # protect clues
    # restore the best alive candidate wherever the meet would empty a cell
    masked = b_logits.masked_fill(alive < 0.5, float("-inf"))
    best = torch.zeros_like(alive).scatter_(2, masked.argmax(-1, keepdim=True), 1.0)
    empty = (new.sum(-1, keepdim=True) == 0)
    return torch.where(empty, best, new)


def conflict_flag(cls_logit, theta_cls=0.0):
    """The trained conflict (⊥) head: this state is unsatisfiable (a branch killed a true digit)."""
    return cls_logit > theta_cls


@torch.no_grad()
def branch(alive, b_logits, tau=1.5):
    """Pick, per unsolved example, a multi-candidate cell and PIN one digit sampled from
    softmax(b/tau) among its alive candidates. Fully vectorized (no Python loop over the batch).
    Returns new alive + the (cell,digit) pinned (cell=digit=-1 for examples with no multi cell)."""
    B, P, V = alive.shape
    ar = torch.arange(B, device=alive.device)
    multi = alive.sum(-1) > 1                                 # [B,P]
    has = multi.any(-1)                                       # [B]
    # random multi-cell per example: random score on multi cells, argmax
    score = torch.rand(B, P, device=alive.device).masked_fill(~multi, -1.0)
    cell = score.argmax(-1)                                   # [B]
    bl = b_logits[ar, cell].masked_fill(alive[ar, cell] < 0.5, -1e9)   # [B,V] alive-only logits
    di = torch.multinomial(torch.softmax(bl / tau, -1), 1).squeeze(-1)  # [B]
    onehot = torch.zeros(B, V, device=alive.device).scatter_(1, di[:, None], 1.0)
    out = alive.clone()
    out[ar, cell] = torch.where(has[:, None], onehot, out[ar, cell])   # only pin where a multi cell exists
    neg = torch.full_like(cell, -1)
    pinned = torch.stack([torch.where(has, cell, neg), torch.where(has, di, neg)], -1)
    return out, pinned


def status(alive):
    """Per-example: solved (all singletons), dead (some empty cell)."""
    n = alive.sum(-1)
    solved = (n == 1).all(dim=1)
    dead = (n == 0).any(dim=1)
    return solved, dead


def is_correct(alive, sol, dev):
    """For solved grids: does the singleton assignment equal the true solution?"""
    sol = torch.as_tensor(sol, dtype=torch.long, device=dev)
    pred = alive.argmax(-1) + 1
    return ((pred == sol).all(dim=1))
