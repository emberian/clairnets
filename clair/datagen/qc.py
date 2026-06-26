"""clair/datagen/qc.py — quality control for the rule-based corpus (no frontier model).

HONEST FRAMING (this is NOT the Bedrock faithfulness round-trip). The renderer emits text
DIRECTLY from the exact facts, and the answer comes from clair.csp's exact, self-checked dedₚ,
so **exactness is by construction** — there is nothing to "reconstruct." Re-parsing our own
deterministic output to rediscover a ground truth we already hold would be circular ceremony.

What QC actually buys us, for a rule-based renderer:

  exactness        guaranteed by construction; `answer_exact` only re-asserts the stored label
                   against dedₚ on the ORIGINAL structured CSP (a near-free guard on Problem
                   construction, not a parse round-trip).
  ambiguity gate   an INDEPENDENT keyword reader (parse.py) tries to recover the constraints
                   from the prose. Its real jobs are (a) catching RENDERER BUGS — a flipped
                   "before", a logic-changing aggregation — as a regression test of our own
                   code, and (b) flagging renderings a dumb reader genuinely can't pin down
                   (ambiguous => low quality => drop). Reported as a pass-rate.
  diversity gates  distinct-n n-gram ratios, template-collapse (skeleton frequency), self-BLEU,
                   length sanity, and near-duplicate removal (MinHash/Jaccard). THIS is the
                   point of the exercise.

norm_logical is the canonical comparison key for the ambiguity gate; it expands alldiff ->
pairwise neq and folds xor into parity, so an aggregated rendering still matches its atomic
ground truth instead of tripping a false desync.
"""
from __future__ import annotations

import itertools as it
import re
from collections import Counter

from .. import curriculum as CU
from .. import hard_tasks as HT
from . import parse as P

ABSTAIN = CU.ABSTAIN


# --------------------------------------------------------------------------- canonical logic key
def norm_logical(facts) -> frozenset:
    """Order-insensitive logical key. alldiff -> neq pairs; xor/par -> ('par', sorted scope);
    eq/neq unordered; sum addends sorted; lt/le/pin kept directional. Aggregation-invariant."""
    out = set()
    for f in facts:
        k = f[0]
        if k == "pin":
            out.add(("pin", f[1], f[2]))
        elif k == "eq":
            out.add(("eq", *sorted((f[1], f[2]))))
        elif k == "neq":
            out.add(("neq", *sorted((f[1], f[2]))))
        elif k in ("lt", "le"):
            out.add((k, f[1], f[2]))
        elif k == "sum":
            out.add(("sum", *sorted((f[1], f[2])), f[3]))
        elif k in ("xor", "par"):
            scope = (f[1], f[2], f[3]) if k == "xor" else tuple(f[1])
            out.add(("par", tuple(sorted(scope))))
        elif k == "alldiff":
            for a, b in it.combinations(sorted(f[1]), 2):
                out.add(("neq", a, b))
    return frozenset(out)


# --------------------------------------------------------------------------- per-record checks
def answer_exact(p) -> bool:
    """Re-assert the stored label against dedₚ on the ORIGINAL structured CSP. This is the
    exactness guard — by construction it should always hold; it guards Problem construction,
    NOT the renderer (and involves no parsing of our own output)."""
    exact = HT.fast_dedP(p.csp)
    cell = exact[p.query]
    det = len(cell) == 1
    ans = int(next(iter(cell))) if det else ABSTAIN
    return det == p.determined and ans == p.answer


def unambiguous(p, rendering):
    """Independent-reader ambiguity gate: can a dumb keyword parser recover EXACTLY the
    constraints from the prose? Catches renderer bugs (flipped direction, logic-changing
    aggregation) and genuinely ambiguous text. Returns (ok, parsed_facts_or_None).
    NOT a faithfulness round-trip against an untrusted model — the renderer is exact by
    construction; this tests OUR phrasing for ambiguity/regressions."""
    parsed = P.parse_facts(rendering.text, rendering.entities, rendering.values)
    if parsed is None:
        return False, None
    return (norm_logical(parsed) == norm_logical(p.facts)), parsed


# --------------------------------------------------------------------------- text utilities
_WORD = re.compile(r"[a-z0-9]+")
_NUM = re.compile(r"\b\d+\b")


def _tokens(text: str):
    return _WORD.findall(text.lower())


def skeleton(rendering) -> str:
    """Mask a rendering's entities/values/digits to expose its TEMPLATE shape, for collapse
    detection. Two renderings sharing a skeleton used the same phrasing skeleton."""
    s = P._mask(P._mask(rendering.text, rendering.entities, "E"), rendering.values, "V")
    s = re.sub(r"\x01E\d+\x01", "<E>", s)
    s = re.sub(r"\x01V\d+\x01", "<V>", s)
    return _NUM.sub("<N>", s)


# --------------------------------------------------------------------------- diversity metrics
def _ngrams(toks, n):
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def distinct_n(texts, n):
    """Corpus distinct-n: unique n-grams / total n-grams (higher = more lexically varied)."""
    tot = Counter()
    for t in texts:
        tot.update(_ngrams(_tokens(t), n))
    total = sum(tot.values())
    return (len(tot) / total) if total else 0.0


def self_bleu(texts, rng, sample=200, refs=8):
    """Approximate self-BLEU: for a sample of texts, modified 1-4-gram precision against a few
    random other texts as references, with brevity penalty. LOWER = more diverse. 0..1."""
    import math
    if len(texts) < 3:
        return 0.0
    idx = rng.choice(len(texts), size=min(sample, len(texts)), replace=False)
    scores = []
    for i in idx:
        cand = _tokens(texts[i])
        if len(cand) < 4:
            continue
        others = rng.choice(len(texts), size=min(refs, len(texts)), replace=False)
        ref_ng = {n: Counter() for n in range(1, 5)}
        ref_lens = []
        for j in others:
            if j == i:
                continue
            rt = _tokens(texts[j])
            ref_lens.append(len(rt))
            for n in range(1, 5):
                ref_ng[n].update(_ngrams(rt, n))
        if not ref_lens:
            continue
        precs = []
        for n in range(1, 5):
            cg = Counter(_ngrams(cand, n))
            if not cg:
                precs.append(1e-9)
                continue
            overlap = sum(min(c, ref_ng[n][g]) for g, c in cg.items())
            precs.append(max(overlap, 1e-9) / sum(cg.values()))
        bp = min(1.0, math.exp(1 - (min(ref_lens, key=lambda L: abs(L - len(cand))) / len(cand))))
        scores.append(bp * math.exp(sum(math.log(pp) for pp in precs) / 4))
    return sum(scores) / len(scores) if scores else 0.0


def collapse(renderings, top=5):
    """Template-collapse report: most frequent skeletons and the share of the corpus the single
    most common skeleton covers (high share = the renderer is looping a template)."""
    c = Counter(skeleton(r) for r in renderings)
    n = sum(c.values())
    common = c.most_common(top)
    top_share = (common[0][1] / n) if n else 0.0
    return {"n_skeletons": len(c), "top_share": top_share,
            "top": [(sk[:80], cnt) for sk, cnt in common]}


def diversity_report(renderings, rng):
    texts = [r.text for r in renderings]
    lens = [len(_tokens(t)) for t in texts]
    return {
        "n": len(texts),
        "distinct_1": round(distinct_n(texts, 1), 4),
        "distinct_2": round(distinct_n(texts, 2), 4),
        "distinct_3": round(distinct_n(texts, 3), 4),
        "self_bleu": round(self_bleu(texts, rng), 4),
        "len_mean": round(sum(lens) / len(lens), 1) if lens else 0,
        "len_min": min(lens) if lens else 0,
        "len_max": max(lens) if lens else 0,
        "skins": dict(Counter(r.skin for r in renderings)),
        "structures": dict(Counter(r.structure for r in renderings)),
        "schemes": dict(Counter(r.scheme for r in renderings)),
        "collapse": collapse(renderings),
    }


# --------------------------------------------------------------------------- dedup (MinHash/LSH)
def _shingles(text, k=4):
    toks = _tokens(text)
    return {hash(tuple(toks[i:i + k])) for i in range(max(1, len(toks) - k + 1))}


def _minhash(sh, perms):
    if not sh:
        return tuple([0] * len(perms))
    return tuple(min(((a * h + b) & 0xFFFFFFFF) for h in sh) for a, b in perms)


def near_dup_mask(texts, rng, threshold=0.85, n_perm=24, bands=8):
    """Return a boolean keep-mask removing near-duplicate texts (MinHash LSH + Jaccard verify)."""
    perms = [(int(rng.integers(1, 2**31)), int(rng.integers(0, 2**31))) for _ in range(n_perm)]
    shs = [_shingles(t) for t in texts]
    sigs = [_minhash(s, perms) for s in shs]
    rows = n_perm // bands
    buckets = {}
    keep = [True] * len(texts)
    for i, sig in enumerate(sigs):
        if not keep[i]:
            continue
        cand = set()
        for b in range(bands):
            key = (b, sig[b * rows:(b + 1) * rows])
            cand |= buckets.get(key, set())
        for j in cand:
            if keep[j] and shs[i] and shs[j]:
                jac = len(shs[i] & shs[j]) / len(shs[i] | shs[j])
                if jac >= threshold:
                    keep[i] = False
                    break
        if keep[i]:
            for b in range(bands):
                buckets.setdefault((b, sig[b * rows:(b + 1) * rows]), set()).add(i)
    return keep
