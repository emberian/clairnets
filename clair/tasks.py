"""The gauntlet: algorithmic (causal next-token) + reasoning (non-causal grid).

Each Task yields integer batches x,y of shape [B,T]; y uses -1 to mark positions NOT
scored (prompt tokens / given Sudoku cells). exact_acc = fraction of examples with ALL
scored positions correct. This is exact-accuracy, not vibes — the room's rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

IGNORE = -1


@dataclass
class Task:
    name: str
    vocab: int
    ctx: int
    causal: bool
    sample: Callable[[int, torch.device, np.random.Generator], tuple]


def _causal_xy(seqs, ans_lens, device):
    """seqs: [B,T] full sequences (prompt+answer). Score only the last ans_len tokens of each.
    Returns x=[B,T-1], y=[B,T-1] (next-token), with non-answer targets = IGNORE."""
    s = torch.as_tensor(np.asarray(seqs), dtype=torch.long, device=device)
    x, y = s[:, :-1].clone(), s[:, 1:].clone()
    mask = torch.ones_like(y)
    for i, al in enumerate(ans_lens):
        mask[i, : y.size(1) - al] = 0          # only the final `al` next-token preds count
    y[mask == 0] = IGNORE
    return x, y


# ---- algorithmic, causal ----------------------------------------------------
def mod_arith(P=97, op="+"):
    EQ, V = P, P + 1                          # '=' token id; vocab = P+1
    f = {"+": lambda a, b: (a + b) % P, "*": lambda a, b: (a * b) % P,
         "-": lambda a, b: (a - b) % P}[op]

    def sample(B, device, rng):
        a = rng.integers(0, P, B); b = rng.integers(0, P, B)
        seqs = np.stack([a, b, np.full(B, EQ), f(a, b)], 1)   # [a,b,=,c]
        return _causal_xy(seqs, [1] * B, device)
    return Task(f"mod{op}{P}", V, 4, True, sample)


def sort_task(N=12, K=20):
    EQ, V = K, K + 1                          # vocab = symbols 0..K-1 plus '='
    def sample(B, device, rng):
        arr = rng.integers(0, K, (B, N))
        srt = np.sort(arr, 1)
        seqs = np.concatenate([arr, np.full((B, 1), EQ), srt], 1)   # [list, =, sorted]
        return _causal_xy(seqs, [N] * B, device)
    return Task(f"sort{N}", V, 2 * N + 1, True, sample)


def copy_task(N=16, K=20, reverse=False):
    EQ, V = K, K + 1
    def sample(B, device, rng):
        arr = rng.integers(0, K, (B, N))
        out = arr[:, ::-1] if reverse else arr
        seqs = np.concatenate([arr, np.full((B, 1), EQ), out], 1)
        return _causal_xy(seqs, [N] * B, device)
    return Task(("rev" if reverse else "copy") + str(N), V, 2 * N + 1, True, sample)


def dyck(N=24, k=2):
    """Balanced-bracket prediction: given a prefix of a Dyck-k word, predict legal next types.
    Framed as next-token LM over an exactly-balanced word; scores all positions (hard: must
    track the stack). vocab = 2k bracket types + pad."""
    opens = list(range(k)); closes = list(range(k, 2 * k)); V = 2 * k

    def gen_one(rng):
        seq, stack = [], []
        for _ in range(N):
            if stack and (len(stack) >= N - len(seq) or rng.random() < 0.5):
                t = stack.pop(); seq.append(closes[t])
            else:
                t = int(rng.integers(0, k)); stack.append(t); seq.append(opens[t])
        while stack:
            seq.append(closes[stack.pop()])
        return seq[:N]

    def sample(B, device, rng):
        seqs = np.array([gen_one(rng) for _ in range(B)])
        return _causal_xy(seqs, [N - 1] * B, device)   # predict tokens 2..N from 1..N-1
    return Task(f"dyck{k}", V, N, True, sample)


# ---- reasoning, non-causal grid (Sudoku 4x4 by default; 9x9 optional) -------
def sudoku(side=4):
    n = side; box = int(n ** 0.5); V = n + 1; T = n * n   # tokens: 0=blank, 1..n digits

    def solved(rng):
        g = np.zeros((n, n), int)

        def ok(r, c, v):
            if v in g[r] or v in g[:, c]:
                return False
            br, bc = r - r % box, c - c % box
            return v not in g[br:br + box, bc:bc + box]

        def fill(i=0):
            if i == n * n:
                return True
            r, c = divmod(i, n)
            for v in rng.permutation(np.arange(1, n + 1)):
                if ok(r, c, v):
                    g[r, c] = v
                    if fill(i + 1):
                        return True
                    g[r, c] = 0
            return False
        fill()
        return g

    def sample(B, device, rng):
        xs, ys = [], []
        for _ in range(B):
            sol = solved(rng)
            puz = sol.copy()
            ncells = n * n
            holes = rng.choice(ncells, size=int(ncells * 0.55), replace=False)
            puz.flat[holes] = 0
            xs.append(puz.flatten()); ys.append(sol.flatten())
        x = torch.as_tensor(np.array(xs), dtype=torch.long, device=device)
        y = torch.as_tensor(np.array(ys), dtype=torch.long, device=device)
        y = torch.where(x == 0, y, torch.full_like(y, IGNORE))     # score only blanks
        return x, y
    return Task(f"sudoku{n}", V, T, False, sample)


GAUNTLET = {
    "modadd": lambda: mod_arith(97, "+"),
    "modmul": lambda: mod_arith(97, "*"),
    "sort": lambda: sort_task(12),
    "reverse": lambda: copy_task(16, reverse=True),
    "dyck2": lambda: dyck(24, 2),
    "sudoku4": lambda: sudoku(4),
    "sudoku9": lambda: sudoku(9),
}
