"""Problems + targets for the PURE-LATENT deductive organ ("design B").

Design B is the bitter-lesson deductor: the organ is handed NO explicit symbolic program. It reads
only OLMo's dense hidden of the PROBLEM TEXT and is grounded solely on its OUTPUT — its decoded
per-cell candidate sets must dominate the exact transformer dedₚ (clair.csp.exact_dedP). This module
provides the data both designs share, so the comparison is apples-to-apples:

  * generators for COLORING (pins + DIFF) and ORDERING/EQUALITY (pins + LT/SAME/DIFF over an ordered
    domain), each rejection-sampled to be solvable AND non-trivially narrowed;
  * the EXACT per-cell dedₚ target (clair.csp.exact_dedP from the full grid) — the ground truth both
    organs are scored against;
  * English RENDERING with two disjoint phrasing sets ("train" vs "ood") + per-cell mention char-spans
    (the only grounding the latent organ gets: which tokens name which cell);
  * build_batch  -> tokenized text + mention masks + dedₚ targets        (the LATENT-from-text input);
  * featurize_factors -> the bipartite factor-graph tensors + dedₚ targets (the EXPLICIT-from-factors
    upper bound, fed the TRUE constraints — clair.proposer.FactorGraphProposer).

The TRUE factors exist here (we build the CSP to compute dedₚ) but are NEVER handed to the latent
organ — it must recover the constraint structure latently through OLMo + α.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import csp as C

NODES = [chr(ord("A") + i) for i in range(12)]
COLORS = ["red", "green", "blue", "yellow", "orange"]
SIZES = ["tiny", "small", "medium", "large", "huge"]            # ORDERED domain (lt is meaningful)
TASKS = ("coloring", "ordering")


def _valwords(domain):
    return COLORS if domain == "color" else SIZES


@dataclass
class Problem:
    n: int
    K: int
    domain: str                  # "color" | "size"
    cons: list                   # list of (kind, *args): pin/diff/same/lt
    query: int


# --------------------------------------------------------------------------- CSP construction
def build_csp(prob: Problem) -> C.CSP:
    """The TRUE finite CSP (used ONLY to compute the exact dedₚ target + the explicit baseline's
    factors — never shown to the latent organ)."""
    K = prob.K
    cons = []
    for c in prob.cons:
        kind = c[0]
        if kind == "pin":
            i, v = c[1], c[2]
            cons.append(C._rel((i,), (lambda v: lambda t: t[0] == v)(v), K))
        elif kind == "diff":
            i, j = c[1], c[2]
            cons.append(C._rel((i, j), lambda t: t[0] != t[1], K))
        elif kind == "same":
            i, j = c[1], c[2]
            cons.append(C._rel((i, j), lambda t: t[0] == t[1], K))
        elif kind == "lt":
            i, j = c[1], c[2]
            cons.append(C._rel((i, j), lambda t: t[0] < t[1], K))
        else:
            raise ValueError(kind)
    return C.CSP(prob.n, K, tuple(cons))


# --------------------------------------------------------------------------- generation
def gen_problem(kind, n, K, rng, edge_p=0.3, pin_frac=0.3, m_max=40, tries=400):
    """One solvable, non-trivially-narrowed instance. Returns (Problem, CSP, dedP) where dedP is the
    exact per-cell survivor sets (tuple of frozensets) from the FULL grid."""
    domain = "color" if kind == "coloring" else "size"
    for _ in range(tries):
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < edge_p:
                    if kind == "coloring":
                        cons.append(("diff", i, j))
                    else:
                        u = rng.random()
                        if u < 0.55:
                            a, b = (i, j) if rng.random() < 0.5 else (j, i)
                            cons.append(("lt", a, b))
                        elif u < 0.8:
                            cons.append(("same", i, j))
                        else:
                            cons.append(("diff", i, j))
        for v in range(n):
            if rng.random() < pin_frac:
                cons.append(("pin", v, int(rng.integers(K))))
        if len(cons) == 0 or len(cons) > m_max:
            continue
        prob = Problem(n, K, domain, cons, int(rng.integers(n)))
        csp = build_csp(prob)
        ded = C.exact_dedP(csp, csp.full())
        if any(len(d) == 0 for d in ded):
            continue                                            # unsat -> skip
        elim = sum(K - len(ded[i]) for i in range(n))
        if elim < max(1, n // 3):
            continue                                            # require real narrowing structure
        return prob, csp, ded
    # fallback: a single pin (always solvable; narrows one cell by K-1)
    prob = Problem(n, K, domain, [("pin", 0, 0)], 0)
    csp = build_csp(prob)
    return prob, csp, C.exact_dedP(csp, csp.full())


def make_pool(kinds, ns, K, per, rng, **kw):
    """A balanced pool of (Problem, CSP, dedP) over the requested kinds x cell-counts."""
    pool = []
    for n in ns:
        for k in kinds:
            for _ in range(per):
                pool.append(gen_problem(k, n, K, rng, **kw))
    rng.shuffle(pool)
    return pool


# --------------------------------------------------------------------------- rendering (+ mentions)
def render_problem(prob: Problem, pset: str):
    """Render the problem as English; return (text, spans) where spans[cell] = [(char_start,char_end)]
    of that cell's letter mentions. `pset` selects a phrasing family: "train" vs a DISJOINT "ood" set
    (held-out phrasings) — same constraints, different surface form."""
    parts, spans, cur = [], {}, [0]

    def emit(s):
        parts.append(s)
        cur[0] += len(s)

    def emit_node(v):
        emit("node ")
        st = cur[0]
        emit(NODES[v])
        spans.setdefault(v, []).append((st, cur[0]))

    dom = prob.domain
    vw = _valwords(dom)
    word = "color" if dom == "color" else "size"
    if pset == "train":
        emit(f"Assign a {word} to each node. ")
    else:
        emit(f"Here is a {word} puzzle. ")
    for c in prob.cons:
        _render_con(c, dom, pset, vw, word, emit, emit_node)
    emit(f"Question: what {word} is ")
    emit_node(prob.query)
    emit("? Answer:")
    return "".join(parts), spans


def _render_con(c, dom, pset, vw, word, emit, emit_node):
    kind = c[0]
    if kind == "pin":
        i, v = c[1], c[2]
        emit_node(i)
        emit(f" is {vw[v]}. " if pset == "train" else f" has the {word} {vw[v]}. ")
    elif kind == "diff":
        i, j = c[1], c[2]
        emit_node(i)
        if pset == "train":
            emit(" and ")
            emit_node(j)
            emit(f" must be different {word}s. ")
        else:
            emit(" cannot match ")
            emit_node(j)
            emit(f" in {word}. ")
    elif kind == "same":
        i, j = c[1], c[2]
        emit_node(i)
        if pset == "train":
            emit(" and ")
            emit_node(j)
            emit(" are the same. ")
        else:
            emit(" matches ")
            emit_node(j)
            emit(". ")
    elif kind == "lt":
        i, j = c[1], c[2]
        emit_node(i)
        if pset == "train":
            emit(" is smaller than ")
            emit_node(j)
            emit(". ")
        else:
            emit(" comes before ")
            emit_node(j)
            emit(" in size. ")


# --------------------------------------------------------------------------- LATENT input (text)
def build_batch(items, tok, Nmax, K, dev, pset):
    """Tokenize a batch of rendered problems; build per-cell mention masks + the dedₚ targets.
    items: list of (Problem, CSP, dedP). NO factors are placed in the returned dict."""
    texts, all_spans = [], []
    for (prob, csp, ded) in items:
        t, sp = render_problem(prob, pset)
        texts.append(t)
        all_spans.append(sp)
    enc = tok(texts, return_offsets_mapping=True, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(dev)
    attn = enc["attention_mask"].to(dev)
    offsets = enc["offset_mapping"]
    B, T = input_ids.shape
    mention = torch.zeros(B, Nmax, T, device=dev)
    for b, sp in enumerate(all_spans):
        offs = offsets[b].tolist()
        for v, sps in sp.items():
            for (cs, ce) in sps:
                for ti, (a, c) in enumerate(offs):
                    if a == c:
                        continue
                    if a < ce and c > cs:
                        mention[b, v, ti] = 1.0
    tgt = torch.zeros(B, Nmax, K, device=dev)
    vmask = torch.zeros(B, Nmax, device=dev)
    for b, (prob, csp, ded) in enumerate(items):
        vmask[b, : prob.n] = 1.0
        for i in range(prob.n):
            for v in ded[i]:
                tgt[b, i, v] = 1.0
    return {"input_ids": input_ids, "attn": attn, "mention": mention, "vmask": vmask,
            "tgt": tgt, "texts": texts}


# --------------------------------------------------------------------------- EXPLICIT input (factors)
def relation_table(scope, al, K, a_max):
    """Broadcast relation table over K**a_max: T[v0..]=1 iff first |scope| coords are an allowed
    tuple (remaining axes don't-care). Mirrors clair.run_proposer.relation_table."""
    a = len(scope)
    t = np.zeros((K,) * a_max, dtype=np.float32)
    for tup in al:
        sl = [slice(None)] * a_max
        for p in range(a):
            sl[p] = tup[p]
        t[tuple(sl)] = 1.0
    return t.reshape(-1)


def featurize_factors(items, Nmax, K, Mmax, Amax, dev):
    """Bipartite factor-graph tensors for clair.proposer.FactorGraphProposer, from the FULL grid.
    This is the EXPLICIT upper bound: the organ is handed the TRUE constraints as factors."""
    B = len(items)
    rel_dim = K ** Amax
    var_mask = np.zeros((B, Nmax, K), np.float32)
    given = np.zeros((B, Nmax), np.float32)
    var_valid = np.zeros((B, Nmax), np.float32)
    fac_rel = np.zeros((B, Mmax, rel_dim), np.float32)
    fac_arity = np.zeros((B, Mmax, Amax), np.float32)
    fac_valid = np.zeros((B, Mmax), np.float32)
    edge_var = np.full((B, Mmax, Amax), Nmax, np.int64)
    edge_valid = np.zeros((B, Mmax, Amax), np.float32)
    for bi, (prob, csp, ded) in enumerate(items):
        for i in range(csp.n):
            var_valid[bi, i] = 1.0
            var_mask[bi, i, :] = 1.0                            # full grid (every value alive)
        for fi, (sc, al) in enumerate(csp.cons):
            if fi >= Mmax:
                break
            fac_valid[bi, fi] = 1.0
            fac_arity[bi, fi, len(sc) - 1] = 1.0
            fac_rel[bi, fi] = relation_table(sc, al, K, Amax)
            for p, cell in enumerate(sc):
                edge_var[bi, fi, p] = cell
                edge_valid[bi, fi, p] = 1.0
    t = lambda a: torch.as_tensor(a, device=dev)
    return dict(var_mask=t(var_mask), given=t(given), var_valid=t(var_valid),
                fac_rel=t(fac_rel), fac_arity=t(fac_arity), fac_valid=t(fac_valid),
                edge_var=t(edge_var), edge_valid=t(edge_valid))


def targets(items, Nmax, K, dev):
    """dedₚ survivor target [B,Nmax,K] + valid-cell mask [B,Nmax], shared by both designs."""
    B = len(items)
    tgt = np.zeros((B, Nmax, K), np.float32)
    vmask = np.zeros((B, Nmax), np.float32)
    for bi, (prob, csp, ded) in enumerate(items):
        vmask[bi, : prob.n] = 1.0
        for i in range(prob.n):
            for v in ded[i]:
                tgt[bi, i, v] = 1.0
    return torch.as_tensor(tgt, device=dev), torch.as_tensor(vmask, device=dev)


# --------------------------------------------------------------------------- the decisive metric
@torch.no_grad()
def narrowing_stats(b, tgt, vmask, theta=0.5):
    """Decode survival logits b [B,N,K] at threshold theta and compare to dedₚ (tgt). Returns the
    four counts for narrowing-RECALL and FALSE-ELIM:
        recall  = (dedₚ-eliminations the model ALSO makes) / (dedₚ-eliminations)
        false   = (dedₚ-survivors the model WRONGLY eliminates) / (dedₚ-survivors)
    All over valid (cell,value) slots. dedₚ-elimination = valid value NOT in dedₚ survivors."""
    keep = torch.sigmoid(b) >= theta                           # [B,N,K]
    valid = vmask.bool().unsqueeze(-1).expand_as(keep)
    surv = tgt > 0.5
    dedp_elim = valid & (~surv)                                # dedₚ eliminated these
    model_elim = valid & (~keep)
    rec_num = (dedp_elim & model_elim).sum().item()
    rec_den = dedp_elim.sum().item()
    fe_num = (model_elim & surv).sum().item()
    fe_den = (surv & valid).sum().item()
    return rec_num, rec_den, fe_num, fe_den
