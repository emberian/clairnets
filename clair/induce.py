"""The (d) bet, minimally: does a host transformer LEARN to PROGRAM a differentiable sound
deductor — and thereby gain (1) calibrated abstention and (2) size generalization — where a
plain transformer of equal size confabulates and degrades?

Task: in-context graph k-coloring with abstention.
  A problem is a set of facts over N variables:
    IS(v, c)     v must be color c           (a pin)
    DIFF(v, w)   v and w must differ          (an exclusion edge)
  then QUERY(q). The ground-truth answer is:
    - color c   if q takes color c in EVERY proper coloring consistent with the facts (forced),
    - ABSTAIN   if q can take >=2 different colors across solutions (underdetermined) OR there is
                no solution (unsatisfiable).
  We compute this exactly by enumerating solutions (small N), so determinacy is ground-truth.

Two models, equal host:
  HybridReasoner — host encodes the facts, a COMPILE head writes (initial candidates per var,
    exclusion-edge weights) into a differentiable narrowing deductor; a READ head turns the
    queried var's narrowed marginals into (color, abstain).
  BaselineReasoner — same host, mean-pool, a direct (color, abstain) MLP head. No deductor.

The decisive reads: abstain precision/recall on underdetermined queries, and generalization to
LARGER N than seen in training (the deductor is size-agnostic and sound; the pooled head is not).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ABSTAIN = -1  # answer label for underdetermined / unsat


# ----------------------------------------------------------------- task generation
def _solutions(n, k, pins, edges, cap=20000):
    """Enumerate proper k-colorings consistent with pins{var:color} and edges set{(i,j)}.
    Returns a list of color-arrays (capped). Backtracking with forward checking."""
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j); adj[j].append(i)
    out, col = [], [-1] * n

    def ok(v, c):
        if v in pins and pins[v] != c:
            return False
        return all(col[u] != c for u in adj[v])

    order = sorted(range(n), key=lambda v: -len(adj[v]))

    def bt(idx):
        if len(out) >= cap:
            return
        if idx == n:
            out.append(col.copy()); return
        v = order[idx]
        choices = [pins[v]] if v in pins else range(k)
        for c in choices:
            if ok(v, c):
                col[v] = c; bt(idx + 1); col[v] = -1
    bt(0)
    return out


def gen_problem(n, k, rng, edge_p=0.35, pin_frac=0.35):
    """One coloring-with-abstention instance. Returns dict of arrays + the ground-truth answer."""
    # random exclusion edges
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_p:
                edges.add((i, j))
    # pin a subset of vars to random colors (kept consistent-ish; solver handles infeasible too)
    pins = {}
    for v in range(n):
        if rng.random() < pin_frac:
            pins[v] = int(rng.integers(0, k))
    sols = _solutions(n, k, pins, edges)
    q = int(rng.integers(0, n))
    if not sols:
        ans = ABSTAIN                                   # unsat -> abstain
    else:
        qcols = {s[q] for s in sols}
        ans = next(iter(qcols)) if len(qcols) == 1 else ABSTAIN
    # encode facts as triples (type, a, b): IS=0 -> (0, v, color); DIFF=1 -> (1, v, w)
    facts = [(0, v, c) for v, c in pins.items()] + [(1, i, j) for (i, j) in edges]
    rng.shuffle(facts)
    return {"n": n, "k": k, "facts": facts, "query": q, "answer": ans,
            "sat": len(sols) > 0, "determined": ans != ABSTAIN,
            "capped": len(sols) >= 20000}    # if hit, determinacy label is unreliable -> caller skips


def make_batch(probs, Nmax, K, dev):
    """Pad to Nmax vars; build fact-token tensors + the supervision targets + the TRUE compile
    targets (pin matrix, diff adjacency) for optional auxiliary grounding."""
    B = len(probs)
    Fmax = max(len(p["facts"]) for p in probs)
    ftype = torch.zeros(B, Fmax, dtype=torch.long, device=dev)     # 0=IS 1=DIFF 2=PAD
    fa = torch.zeros(B, Fmax, dtype=torch.long, device=dev)
    fb = torch.zeros(B, Fmax, dtype=torch.long, device=dev)
    fmask = torch.zeros(B, Fmax, device=dev)
    pins = torch.full((B, Nmax, K), 0.0, device=dev)               # true init pins (aux)
    pinned = torch.zeros(B, Nmax, device=dev)
    adj = torch.zeros(B, Nmax, Nmax, device=dev)                   # true diff edges (aux)
    vmask = torch.zeros(B, Nmax, device=dev)
    query = torch.zeros(B, dtype=torch.long, device=dev)
    ans = torch.zeros(B, dtype=torch.long, device=dev)            # color id (0..K-1); abstain sep
    abst = torch.zeros(B, device=dev)
    for b, p in enumerate(probs):
        vmask[b, : p["n"]] = 1.0
        for f, (t, a, c) in enumerate(p["facts"]):
            ftype[b, f] = t; fa[b, f] = a; fb[b, f] = c; fmask[b, f] = 1.0
            if t == 0:
                pins[b, a, c] = 1.0; pinned[b, a] = 1.0
            else:
                adj[b, a, c] = 1.0; adj[b, c, a] = 1.0
        query[b] = p["query"]
        abst[b] = 0.0 if p["determined"] else 1.0
        ans[b] = p["answer"] if p["determined"] else 0
    return {"ftype": ftype, "fa": fa, "fb": fb, "fmask": fmask, "pins": pins, "pinned": pinned,
            "adj": adj, "vmask": vmask, "query": query, "ans": ans, "abst": abst,
            "ftype_pad": ftype.masked_fill(fmask < 0.5, 2)}


# ----------------------------------------------------------------- host encoder (shared)
class FactEncoder(nn.Module):
    """Embeds (type, varA, argB) fact-triples + a self-attention stack. Variable and color ids
    share an index space up to max(Nmax,K); a role embedding separates them."""
    def __init__(self, Nmax, K, d=128, layers=3, heads=4):
        super().__init__()
        self.Nmax, self.K, self.d = Nmax, K, d
        idx = max(Nmax, K)
        self.type_emb = nn.Embedding(3, d)                 # IS / DIFF / PAD
        self.a_emb = nn.Embedding(idx, d)                  # arg A is always a var
        self.b_emb = nn.Embedding(idx, d)                  # arg B is color (IS) or var (DIFF)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True), layers)

    def forward(self, ba):
        h = self.type_emb(ba["ftype_pad"]) + self.a_emb(ba["fa"]) + self.b_emb(ba["fb"])
        pad = ba["fmask"] < 0.5
        return self.enc(h, src_key_padding_mask=pad), pad   # [B,F,d]


# ----------------------------------------------------------------- differentiable deductor
class ColorDeductor(nn.Module):
    """Sound-ish narrowing for graph coloring. State alive[B,N,K] in [0,1] (soft membership).
    Edge weights E[B,N,N] (compiled by the host). One step: a color loses viability at var i in
    proportion to how confidently an excluded neighbor IS that color. Multiplicative -> monotone
    narrowing. Iterated T times; gradients flow throughout."""
    def __init__(self, T=8):
        super().__init__()
        self.T = T
        self.gamma = nn.Parameter(torch.tensor(3.0))
        self.bias = nn.Parameter(torch.tensor(2.0))

    def forward(self, alive0, E, vmask):
        alive = alive0
        E = E * (1 - torch.eye(E.size(1), device=E.device))         # no self-edges
        for _ in range(self.T):
            den = alive.sum(-1, keepdim=True).clamp_min(1e-6)
            P = alive / den                                         # color marginals per var
            conf = (P.max(-1, keepdim=True).values - (1 - P.max(-1, keepdim=True).values))
            signal = (P * conf).clamp(0, 1)                         # [B,N,K] confident-pin mass
            pressure = torch.einsum("bij,bjk->bik", E, signal)      # neighbor pins push out color k
            keep = torch.sigmoid(self.bias - self.gamma * pressure)
            alive = alive * keep
            alive = alive * vmask.unsqueeze(-1) + (1 - vmask.unsqueeze(-1)) * alive0  # freeze pads
        return alive


# ----------------------------------------------------------------- the two reasoners
class HybridReasoner(nn.Module):
    def __init__(self, Nmax, K, d=128, layers=3, heads=4, T=8):
        super().__init__()
        self.Nmax, self.K = Nmax, K
        self.enc = FactEncoder(Nmax, K, d, layers, heads)
        self.var_q = nn.Embedding(Nmax, d)                         # var slot queries
        self.cand_head = nn.Linear(d, K)                           # init candidate logits per var
        self.edge_a = nn.Linear(d, d); self.edge_b = nn.Linear(d, d)  # bilinear edge scorer
        self.ded = ColorDeductor(T)
        self.read = nn.Linear(3, 1)                                # marginals stats -> abstain logit

    def compile(self, ba):
        h, pad = self.enc(ba)                                      # [B,F,d]
        slots = self.var_q(torch.arange(self.Nmax, device=h.device))[None].expand(h.size(0), -1, -1)
        att = torch.einsum("bnd,bfd->bnf", slots, h) / h.size(-1) ** 0.5
        att = att.masked_fill(pad[:, None, :], -1e9).softmax(-1)
        v = torch.einsum("bnf,bfd->bnd", att, h)                   # [B,N,d] per-var summary
        cand = self.cand_head(v)                                   # init candidate logits
        ea, eb = self.edge_a(v), self.edge_b(v)
        edge = torch.einsum("bid,bjd->bij", ea, eb) / v.size(-1) ** 0.5
        E = torch.sigmoid(0.5 * (edge + edge.transpose(1, 2)))     # symmetric exclusion weights
        return cand, E, v

    def forward(self, ba):
        cand, E, _ = self.compile(ba)
        alive0 = torch.sigmoid(cand) * ba["vmask"].unsqueeze(-1) + 1e-3
        alive = self.ded(alive0, E, ba["vmask"])
        qa = alive[torch.arange(alive.size(0)), ba["query"]]       # [B,K] queried var candidates
        m = qa / qa.sum(-1, keepdim=True).clamp_min(1e-6)
        color_logits = torch.log(m + 1e-6)
        stats = torch.stack([m.max(-1).values, -(m * (m + 1e-9).log()).sum(-1),
                             (m > 0.1).float().sum(-1)], -1)       # peak, entropy, #alive
        abstain_logit = self.read(stats).squeeze(-1)
        return color_logits, abstain_logit, (cand, E)


class BaselineReasoner(nn.Module):
    """Same encoder; pool the facts + the queried var's slot, predict (color, abstain) directly."""
    def __init__(self, Nmax, K, d=128, layers=3, heads=4, **_):
        super().__init__()
        self.enc = FactEncoder(Nmax, K, d, layers, heads)
        self.var_q = nn.Embedding(Nmax, d)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, K + 1))

    def forward(self, ba):
        h, pad = self.enc(ba)
        pooled = (h * (~pad).unsqueeze(-1)).sum(1) / (~pad).sum(1, keepdim=True).clamp_min(1)
        qslot = self.var_q(ba["query"])
        out = self.head(torch.cat([pooled, qslot], -1))
        return out[:, :-1], out[:, -1], (None, None)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    cnt = {"det": 0, "abs": 0, "unsat": 0}
    for _ in range(400):
        p = gen_problem(int(rng.integers(4, 7)), 3, rng)
        cnt["det" if p["determined"] else "abs"] += 1
        cnt["unsat"] += 0 if p["sat"] else 1
    print("colorcsp self-check (N=4..6, K=3): %d problems  determined=%d abstain=%d (unsat=%d)" % (
        400, cnt["det"], cnt["abs"], cnt["unsat"]))
    print("  -> need a healthy mix of determined vs abstain for the task to be non-trivial")
