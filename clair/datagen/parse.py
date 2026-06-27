"""clair/datagen/parse.py — an INDEPENDENT heuristic parser: rendered English -> CSP facts.

This is the faithfulness check's reader. It is written deliberately separately from the
renderer (cue-keyword driven, not template-matched) so it can catch a renderer that drops,
adds, garbles or mis-directs a fact. It is given the concrete entity/value SURFACES a record
used (so naming is not the variable under test), masks them to tokens, splits the text into
fact clauses, and classifies each clause by relation cues.

It returns facts in clair.curriculum's tuple format, or ``None`` if any clause fails to parse
(conservative: an unparseable rendering is dropped rather than guessed). Parity (xor) and
all-different (clique) are emitted in a canonical form; qc.norm_logical reconciles them with
the ground truth, so aggregation never causes a false desync.
"""
from __future__ import annotations

import itertools as it
import re

# clause separators the renderer guarantees never appear inside an entity list
_CLAUSE = re.compile(r"\s*;\s*|\s+—\s+|,\s+and\s+|,\s+while\s+|,\s+but\s+")
_SENT = re.compile(r"[.\n]+|^\s*\d+\.\s*|\s*-\s+", re.M)
_TOK = re.compile(r"\x01([EV])(\d+)\x01")

# relation cue keywords (checked in this order: distinct, equal, le, lt)
_DISTINCT = ("differ", "different", "distinct", "not equal", "cannot equal", "can't equal",
             "cannot", "can't", " not ", "no two", "separate", "interfere", "rival", "border",
             "adjacent", "clash", "mutually", "pairwise", "refuse", "overlap", "neighbor")
_EQUAL = ("same", "equal", "match", "agree", "identical", "share", "shar", "tied together",
          "coalesced", "linked", "forced into", "belong to the same", "friend", "insist")
_LE = ("no later", "or in the same", "at or before", "or ties", "or tied", "no worse",
       "before or", "ahead .* or")
_LT = ("before", "precede", "earlier", "ahead", "beat", "higher", "prerequisite",
       "crossed the line", "happened before", "occurred", "ranks ahead", "run before",
       "taken before", "finish")
_PARITY = ("parity", "xor", "exclusive", r"\bodd\b", "even number")
_SUM = ("plus", "sum of", "adding", "add up", "add ")


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def _mask(text: str, surfaces, tag: str) -> str:
    """Replace each known surface with a \x01<tag><idx>\x01 token, longest first. Also matches the
    sentence-capitalized variant (the renderer upper-cases a sentence's first word, so a surface
    like 'tower 3' can appear as 'Tower 3'). We only ADD the capitalized form — never lowercase —
    so a single-letter entity 'A' can't swallow the article 'a'."""
    lut = {}
    for i, s in enumerate(surfaces):
        lut.setdefault(s, i)
        lut.setdefault(_cap(s), i)
    if not lut:
        return text
    alt = "|".join(re.escape(s) for s in sorted(lut, key=len, reverse=True))
    pat = re.compile(r"(?<!\w)(" + alt + r")(?!\w)")
    return pat.sub(lambda m: f"\x01{tag}{lut[m.group(1)]}\x01", text)


def _toks(clause: str):
    es, vs = [], []
    for m in _TOK.finditer(clause):
        (es if m.group(1) == "E" else vs).append(int(m.group(2)))
    return es, vs


def _has(clause: str, cues) -> bool:
    return any(re.search(c, clause, re.I) for c in cues)


def _binary(clause, a, b):
    # LE before EQUAL: "before ... or in the same position" must not be caught by bare "same".
    if _has(clause, _LE):
        return ("le", a, b)
    if _has(clause, _DISTINCT):
        return ("neq", a, b)
    if _has(clause, _EQUAL):
        return ("eq", a, b)
    if _has(clause, _LT):
        return ("lt", a, b)
    return None


def _sum_fact(clause, es):
    """Identify the result entity c vs the two addends from a sum/plus clause."""
    m = re.search(r"\x01E(\d+)\x01\s+plus\s+\x01E(\d+)\x01", clause, re.I)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
    else:
        m = re.search(r"(?:sum of|adding|add up to|add)\s+\x01E(\d+)\x01\s+and\s+\x01E(\d+)\x01",
                      clause, re.I)
        if not m:
            return None
        a, b = int(m.group(1)), int(m.group(2))
    rest = [e for e in es if e != a and e != b]
    if len(rest) != 1:
        return None
    return ("sum", a, b, rest[0])


def _parse_clause(clause: str):
    """Parse one fact clause (already masked) -> a curriculum fact tuple, or None."""
    es, vs = _toks(clause)
    es_u = list(dict.fromkeys(es))
    if len(es_u) >= 3:
        if _has(clause, _PARITY):
            rhs = 1 if (re.search(r"\bodd\b", clause, re.I) or "to one" in clause.lower()) else 0
            return ("parm", tuple(es_u), rhs)
        if _has(clause, _SUM):
            return _sum_fact(clause, es)
        if _has(clause, _DISTINCT):
            return ("alldiff", tuple(es_u))
        if _has(clause, _EQUAL):
            return ("eqclique", tuple(es_u))
        return None
    if len(es_u) == 2 and not vs:
        return _binary(clause, es_u[0], es_u[1])
    if len(es_u) == 2 and vs:                      # rare: two entities + value -> treat as binary
        return _binary(clause, es_u[0], es_u[1])
    if len(es_u) == 1 and vs:
        return ("pin", es_u[0], vs[0])
    return None


def parse_facts(text: str, entities, values):
    """Parse `text` back to curriculum facts using the record's entity/value surfaces.
    Returns a list of fact tuples (eqclique expands to eq pairs), or None on any failure."""
    masked = _mask(_mask(text, entities, "E"), values, "V")
    facts = []
    for sentence in _SENT.split(masked):
        if not sentence or "?" in sentence:
            continue
        if not _TOK.search(sentence):
            continue                              # intro / framing line, no fact
        for clause in _CLAUSE.split(sentence):
            if not clause or not _TOK.search(clause):
                continue
            f = _parse_clause(clause)
            if f is None:
                return None
            if f[0] == "eqclique":
                facts += [("eq", a, b) for a, b in it.combinations(sorted(f[1]), 2)]
            else:
                facts.append(f)
    return facts          # may be [] (a genuinely fact-free problem); None only on a parse failure
