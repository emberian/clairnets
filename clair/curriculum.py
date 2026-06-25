"""clair/curriculum.py — a curriculum data pipeline for constraint-reasoning problems.

THE KEY INVARIANT: the ground truth is generated FIRST and stays EXACT. The verifier
(``clair.csp``) NEVER touches Bedrock. The flow is::

    harness generates a CSP + its exact solution/query/answer   (this file, witness-first)
        -> Bedrock RENDERS it into DIVERSE natural language       (render_diverse)
        -> a faithfulness round-trip drops desynced renderings    (extract_facts + compare)
        -> we cache (diverse_text, exact_CSP, query, answer) pairs (jsonl)

This attacks the augmented-LLM's extraction wall from the INPUT side: the WRITE head
currently learns from one narrow template; diverse renderings force ROBUST extraction.

The harness is extended with more relation types than the existing =/≠:
    pin (unary)  · eq · neq (coloring) · lt/le (ORDERING) · sum (ARITHMETIC mod d) · alldiff.
Every relation is built as an EXTENSIONAL constraint (scope, allowed-tuples), so
``clair.csp.solutions`` / ``exact_dedP`` give exact solutions + exact forced/abstain
labels automatically — no per-relation solver to get wrong.

Solvability is guaranteed by WITNESS-FIRST generation: we sample an assignment ``s`` and
only ever emit facts that ``s`` satisfies, so ``s`` is always a solution. Determinacy of a
queried cell is then read off ``exact_dedP`` (singleton ⇒ forced, else ABSTAIN).

Bedrock is OPTIONAL here: the generators + self-check are pure-Python/numpy and import
without boto3. The renderer functions import the Bedrock wrapper lazily.
"""
from __future__ import annotations

import itertools as it
import json
import re
from dataclasses import dataclass, field

import numpy as np

from . import csp as C

ABSTAIN = -1

# Entity (cell) labels and per-kind value vocabularies. Entities are single capital
# letters (matching clair.induce's "node A" convention); values are named per domain kind.
ENTITIES = [chr(ord("A") + i) for i in range(26)]
COLORS = ["red", "green", "blue", "yellow", "orange", "purple", "cyan", "pink"]


def value_names(kind: str, d: int) -> list[str]:
    """Human value names for a domain of size d under a given kind."""
    if kind == "color":
        return COLORS[:d]
    if kind == "ordinal":
        return [str(v + 1) for v in range(d)]          # 1-based positions
    return [str(v) for v in range(d)]                  # number


# --------------------------------------------------------------------------- facts
# A fact is a small typed tuple. It is BOTH the thing we render to English AND the thing
# we build the CSP constraint from, so CSP <-> facts are exact by construction.
#   ("pin", a, v)           entity a takes value v
#   ("eq",  a, b)           a equals b
#   ("neq", a, b)           a and b differ
#   ("lt",  a, b)           a strictly before b
#   ("le",  a, b)           a at or before b
#   ("sum", a, b, c)        (val(a) + val(b)) mod d == val(c)
#   ("alldiff", (a,b,...))  the listed entities are pairwise distinct


def fact_constraint(fact, d):
    """Compile one fact into an extensional clair.csp constraint (scope, allowed)."""
    k = fact[0]
    if k == "pin":
        _, a, v = fact
        return C._rel((a,), lambda t, v=v: t[0] == v, d)
    if k == "eq":
        _, a, b = fact
        return C._rel((a, b), lambda t: t[0] == t[1], d)
    if k == "neq":
        _, a, b = fact
        return C._rel((a, b), lambda t: t[0] != t[1], d)
    if k == "lt":
        _, a, b = fact
        return C._rel((a, b), lambda t: t[0] < t[1], d)
    if k == "le":
        _, a, b = fact
        return C._rel((a, b), lambda t: t[0] <= t[1], d)
    if k == "sum":
        _, a, b, c = fact
        return C._rel((a, b, c), lambda t, d=d: (t[0] + t[1]) % d == t[2], d)
    if k == "alldiff":
        scope = tuple(fact[1])
        return C._rel(scope, lambda t: len(set(t)) == len(t), d)
    raise ValueError(f"unknown fact kind {k!r}")


def facts_satisfied_by(facts, s, d) -> bool:
    """Witness check: does assignment s satisfy every fact? (numpy-free, exact)."""
    for f in facts:
        sc, al = fact_constraint(f, d)
        if tuple(s[i] for i in sc) not in al:
            return False
    return True


def build_csp(n, d, facts) -> C.CSP:
    return C.CSP(n, d, tuple(fact_constraint(f, d) for f in facts))


# --------------------------------------------------------------------------- problem
@dataclass
class Problem:
    relation: str                 # primary relation tag (coloring/ordering/arithmetic/alldiff/equality)
    n: int
    d: int
    kind: str                     # domain kind: color | ordinal | number
    facts: list                  # list of typed fact tuples (the ground truth source)
    query: int                    # queried entity index
    answer: int                   # forced value, or ABSTAIN
    determined: bool
    vnames: list = field(default_factory=list)

    @property
    def csp(self) -> C.CSP:
        return build_csp(self.n, self.d, self.facts)

    def entity(self, i: int) -> str:
        return ENTITIES[i]


def _query_answer(csp: C.CSP, facts, rng, det_target: bool):
    """Pick a query cell + its EXACT answer from the harness. Prefers a cell matching the
    requested determinacy and, for determined cells, one that is NOT directly pinned (so the
    answer requires propagation, not lookup). Returns (query, answer, determined)."""
    exact = C.exact_dedP(csp, csp.full())          # per-cell values used by SOME solution
    pinned = {f[1] for f in facts if f[0] == "pin"}
    det = [i for i in range(csp.n) if len(exact[i]) == 1]
    und = [i for i in range(csp.n) if len(exact[i]) > 1]
    det_unpinned = [i for i in det if i not in pinned]
    pref = (det_unpinned or det) if det_target else und
    pool = pref or det_unpinned or det or und
    if not pool:                                    # degenerate (shouldn't happen for solvable)
        return 0, ABSTAIN, False
    q = int(rng.choice(pool))
    if len(exact[q]) == 1:
        return q, int(next(iter(exact[q]))), True
    return q, ABSTAIN, False


# --------------------------------------------------------------------------- generators
# All generators are WITNESS-FIRST: sample s, only emit facts s satisfies ⇒ always solvable.

def gen_coloring(rng, k=3, n_lo=4, n_hi=7, edge_p=0.45, pin_frac=0.35):
    n = int(rng.integers(n_lo, n_hi + 1))
    s = rng.integers(0, k, n)
    facts = []
    for i in range(n):
        for j in range(i + 1, n):
            if s[i] != s[j] and rng.random() < edge_p:      # only differ-edges s respects
                facts.append(("neq", i, j))
    for i in range(n):
        if rng.random() < pin_frac:
            facts.append(("pin", i, int(s[i])))
    return n, k, "color", facts, s


def gen_equality(rng, k=3, n_lo=4, n_hi=7, pin_frac=0.3):
    """Groups of equal cells + some cross-group differences. Pins propagate through eq-chains."""
    n = int(rng.integers(n_lo, n_hi + 1))
    s = rng.integers(0, k, n)
    facts = []
    for i in range(n):
        for j in range(i + 1, n):
            if s[i] == s[j] and rng.random() < 0.5:
                facts.append(("eq", i, j))
            elif s[i] != s[j] and rng.random() < 0.3:
                facts.append(("neq", i, j))
    for i in range(n):
        if rng.random() < pin_frac:
            facts.append(("pin", i, int(s[i])))
    return n, k, "color", facts, s


def gen_ordering(rng, n_lo=4, n_hi=6, pin_frac=0.3):
    """Ordering over positions 1..d. Emit a/before-or-equal facts the witness respects."""
    n = int(rng.integers(n_lo, n_hi + 1))
    d = int(rng.integers(n, n + 3))                          # room for a few ties
    s = rng.integers(0, d, n)
    facts = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if s[i] < s[j] and rng.random() < 0.4:
                facts.append(("lt", i, j))
            elif s[i] <= s[j] and rng.random() < 0.25:
                facts.append(("le", i, j))
    for i in range(n):
        if rng.random() < pin_frac:
            facts.append(("pin", i, int(s[i])))
    return n, d, "ordinal", facts, s


def gen_arithmetic(rng, n_lo=3, n_hi=5, d_lo=4, d_hi=7, pin_frac=0.7):
    """Constructive modular sums: later cells are defined as (a+b) mod d of earlier cells.
    Pinning inputs propagates to determine the sums."""
    n = int(rng.integers(n_lo, n_hi + 1))
    d = int(rng.integers(d_lo, d_hi + 1))
    s = rng.integers(0, d, n)
    facts = []
    n_inputs = max(2, n - int(rng.integers(1, max(2, n - 1))))
    for c in range(n_inputs, n):                             # define each non-input as a sum
        a, b = rng.choice(c, size=2, replace=False)
        s[c] = (s[a] + s[b]) % d
        facts.append(("sum", int(a), int(b), c))
    for i in range(n_inputs):                                # pin a subset of inputs
        if rng.random() < pin_frac:
            facts.append(("pin", i, int(s[i])))
    return n, d, "number", facts, s


def gen_alldiff(rng, n_lo=4, n_hi=6, pin_frac=0.4):
    """One all-different block over the cells + some pins (latin-ish forcing)."""
    n = int(rng.integers(n_lo, n_hi + 1))
    d = int(rng.integers(n, n + 2))                          # >= n so a permutation exists
    s = np.array(rng.permutation(d)[:n])                     # distinct witness values
    facts = [("alldiff", tuple(range(n)))]
    for i in range(n):
        if rng.random() < pin_frac:
            facts.append(("pin", i, int(s[i])))
    return n, d, "number", facts, s


GENERATORS = {
    "coloring": gen_coloring,
    "equality": gen_equality,
    "ordering": gen_ordering,
    "arithmetic": gen_arithmetic,
    "alldiff": gen_alldiff,
}


def gen_problem(rng, relation=None, det_target=None) -> Problem:
    """Generate one verified Problem. `relation` selects a generator (random if None);
    `det_target` (default random) biases the query toward a determined vs abstain cell."""
    if relation is None:
        relation = str(rng.choice(list(GENERATORS)))
    if det_target is None:
        det_target = bool(rng.random() < 0.5)
    n, d, kind, facts, s = GENERATORS[relation](rng)
    assert facts_satisfied_by(facts, s, d), "witness violated a fact (generator bug)"
    csp = build_csp(n, d, facts)
    q, ans, det = _query_answer(csp, facts, rng, det_target)
    return Problem(relation, n, d, kind, facts, q, ans, det, value_names(kind, d))


# --------------------------------------------------------------------------- canonical render
def _fact_sentence(p: Problem, f) -> str:
    E, V = p.entity, p.vnames
    k = f[0]
    if k == "pin":
        _, a, v = f
        if p.kind == "color":
            return f"{E(a)} is {V[v]}."
        if p.kind == "ordinal":
            return f"{E(a)} is in position {V[v]}."
        return f"{E(a)} equals {V[v]}."
    if k == "eq":
        return f"{E(f[1])} is the same as {E(f[2])}."
    if k == "neq":
        kind = "different colors" if p.kind == "color" else "different"
        return f"{E(f[1])} and {E(f[2])} are {kind}."
    if k == "lt":
        return f"{E(f[1])} comes before {E(f[2])}."
    if k == "le":
        return f"{E(f[1])} comes before {E(f[2])} or in the same position."
    if k == "sum":
        return f"{E(f[1])} plus {E(f[2])} equals {E(f[3])} (modulo {p.d})."
    if k == "alldiff":
        names = ", ".join(E(i) for i in f[1])
        return f"{names} all take different values."
    raise ValueError(k)


def _question(p: Problem) -> str:
    Q = p.entity(p.query)
    if p.kind == "color":
        return f"What color is {Q}?"
    if p.kind == "ordinal":
        return f"What position is {Q} in?"
    return f"What is the value of {Q}?"


def canonical_render(p: Problem) -> str:
    """Deterministic, minimal ground-truth English for the problem (the reference rendering)."""
    body = " ".join(_fact_sentence(p, f) for f in p.facts)
    return f"{body} {_question(p)}"


def canonical_answer(p: Problem) -> str:
    if not p.determined:
        return "cannot be determined"
    return p.vnames[p.answer]


# --------------------------------------------------------------------------- fact normalization
def norm_facts(facts) -> frozenset:
    """Canonical, order-insensitive set of facts for exact round-trip comparison.
    eq/neq are unordered pairs; alldiff is an unordered set; pin/lt/le keep direction."""
    out = set()
    for f in facts:
        k = f[0]
        if k in ("eq", "neq"):
            a, b = sorted((f[1], f[2]))
            out.add((k, a, b))
        elif k == "alldiff":
            out.add((k, tuple(sorted(f[1]))))
        elif k == "pin":
            out.add(("pin", f[1], f[2]))
        elif k == "sum":
            a, b = sorted((f[1], f[2]))                       # a+b commutes
            out.add(("sum", a, b, f[3]))
        else:                                                 # lt, le directional
            out.add((k, f[1], f[2]))
    return frozenset(out)


# --------------------------------------------------------------------------- Bedrock renderer
DEFAULT_MODEL = "us.amazon.nova-2-lite-v1:0"
DEFAULT_REGION = "us-east-1"

_RENDER_SYS = (
    "You rewrite a list of logical FACTS as a short natural-language constraint puzzle. "
    "You must render EXACTLY the given facts: include EVERY fact, invent NOTHING, drop NOTHING, "
    "and never add information or solve the puzzle. Keep every entity LABEL (single capital "
    "letters like A, B, C) and every value name EXACTLY as written. You may freely vary wording, "
    "sentence order, and light framing/story-skin, but the logical content must be identical.\n"
    "CRITICAL: never SUMMARIZE several facts into a stronger blanket claim. Do NOT say 'all are "
    "different', 'each has a unique value/color', 'no two share', or similar — those add pairs "
    "that were not listed. State ONLY the exact individual facts given, one correspondence each."
)


def _render_user(p: Problem, k: int) -> str:
    legend = {
        "color": "values are colors",
        "ordinal": "values are positions in an ordering (smaller = earlier)",
        "number": f"values are integers 0..{p.d - 1}",
    }[p.kind]
    facts = "\n".join("- " + _fact_sentence(p, f) for f in p.facts)
    return (
        f"DOMAIN: {legend}.\n"
        f"FACTS:\n{facts}\n"
        f"QUESTION (append verbatim at the end of each rendering): {_question(p)}\n\n"
        f"Produce {k} DIFFERENT renderings of this exact puzzle, each as one short paragraph that "
        f"states all the facts then asks the question. Make them genuinely diverse in phrasing and "
        f"order. Return ONLY a JSON array of {k} strings, nothing else."
    )


_EXTRACT_SYS = (
    "You are an information-extraction parser. Read a natural-language constraint puzzle and "
    "extract its facts into a strict JSON schema. Extract ONLY what is stated; do not infer or "
    "solve. Entities are single capital letters."
)


def _extract_user(text: str, kind: str, d: int) -> str:
    return (
        "Extract every fact from the puzzle below into a JSON array. Each fact is one of:\n"
        '  {"t":"pin","a":"A","v":"<value name>"}      (entity A takes that value)\n'
        '  {"t":"eq","a":"A","b":"B"}                   (A is the same as B)\n'
        '  {"t":"neq","a":"A","b":"B"}                  (A and B differ)\n'
        '  {"t":"lt","a":"A","b":"B"}                   (A strictly before B)\n'
        '  {"t":"le","a":"A","b":"B"}                   (A before B or equal)\n'
        '  {"t":"sum","a":"A","b":"B","c":"C"}          (A plus B equals C, modulo d)\n'
        '  {"t":"alldiff","xs":["A","B","C"]}           (all listed are different)\n'
        f"DOMAIN kind: {kind}; modulus d={d}. Ignore the final question sentence.\n"
        "Return ONLY the JSON array.\n\nPUZZLE:\n" + text
    )


def _json_array(text: str):
    """Pull the first JSON array out of a model response (tolerates ``` fences / prose)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    i, j = text.find("["), text.rfind("]")
    if i == -1 or j == -1 or j < i:
        return None
    try:
        return json.loads(text[i : j + 1])
    except json.JSONDecodeError:
        return None


def render_diverse(p: Problem, k=4, model=DEFAULT_MODEL, region=DEFAULT_REGION,
                   temperature=0.9, max_tokens=900):
    """K diverse Bedrock renderings of one problem. Returns (texts, usage). Lazy-imports boto3."""
    from gowexp.bedrock import chat
    in_tok = out_tok = 0
    texts: list[str] = []
    for _ in range(2):                                        # one retry: Nova occasionally returns []
        r = chat(model, _RENDER_SYS, _render_user(p, k), max_tokens=max_tokens,
                 temperature=temperature, region=region, top_p=0.95)
        in_tok += r["in_tokens"]; out_tok += r["out_tokens"]
        arr = _json_array(r["text"])
        texts = [s.strip() for s in arr if isinstance(s, str) and s.strip()] if arr else []
        if texts:
            break
    return texts, (in_tok, out_tok)


def _coerce_fact(j) -> tuple | None:
    """Map one extracted JSON fact + a value->index lookup later. Returns a partly-symbolic
    tuple using value NAMES (resolved against vnames in extract_and_check)."""
    try:
        t = j["t"]
        ei = lambda x: ENTITIES.index(str(x).strip()[:1])
        if t == "pin":
            return ("pin", ei(j["a"]), str(j["v"]).strip())
        if t in ("eq", "neq", "lt", "le"):
            return (t, ei(j["a"]), ei(j["b"]))
        if t == "sum":
            return ("sum", ei(j["a"]), ei(j["b"]), ei(j["c"]))
        if t == "alldiff":
            return ("alldiff", tuple(ei(x) for x in j["xs"]))
    except (KeyError, ValueError, TypeError, IndexError):
        return None
    return None


def extract_and_check(p: Problem, text: str, model=DEFAULT_MODEL, region=DEFAULT_REGION):
    """Faithfulness round-trip: extract facts from `text` via Bedrock, resolve value names, and
    compare to the true facts as canonical sets. Returns (ok, n_extracted, usage)."""
    from gowexp.bedrock import chat
    r = chat(model, _EXTRACT_SYS, _extract_user(text, p.kind, p.d), max_tokens=700,
             temperature=0.0, region=region)
    arr = _json_array(r["text"])
    usage = (r["in_tokens"], r["out_tokens"])
    if not arr:
        return False, 0, usage
    name_to_v = {nm: i for i, nm in enumerate(p.vnames)}
    got = []
    for j in arr:
        f = _coerce_fact(j)
        if f is None:
            return False, len(arr), usage                     # unparseable fact ⇒ reject
        if f[0] == "pin":
            v = name_to_v.get(f[2])
            if v is None:
                return False, len(arr), usage
            f = ("pin", f[1], v)
        got.append(f)
    return norm_facts(got) == norm_facts(p.facts), len(arr), usage


# --------------------------------------------------------------------------- mentions
def entity_mentions(text: str, n: int) -> dict:
    """Best-effort char spans of each entity letter (standalone capital, word-bounded) in `text`.
    Used by a span-grounded WRITE head. Returns {entity_index: [(start,end), ...]}."""
    out = {}
    for i in range(n):
        spans = [(m.start(), m.end()) for m in re.finditer(rf"(?<![A-Za-z0-9]){ENTITIES[i]}(?![A-Za-z0-9])", text)]
        if spans:
            out[i] = spans
    return out


# --------------------------------------------------------------------------- record / loader
def to_record(p: Problem, text: str, source: str) -> dict:
    """Serialize a (diverse text, exact CSP, query, answer) training record to a json dict."""
    csp = p.csp
    return {
        "relation": p.relation, "kind": p.kind, "n": p.n, "d": p.d,
        "facts": [list(f) if f[0] != "alldiff" else ["alldiff", list(f[1])] for f in p.facts],
        "cons": [[list(sc), sorted(list(t) for t in al)] for sc, al in csp.cons],
        "query": p.query, "answer": p.answer, "determined": p.determined,
        "vnames": p.vnames, "text": text, "canonical": canonical_render(p),
        "answer_name": canonical_answer(p),
        "mentions": {str(k): v for k, v in entity_mentions(text, p.n).items()},
        "source": source,
    }


def record_to_csp(rec: dict) -> C.CSP:
    """Rebuild the exact clair.csp.CSP from a stored record (verifier ground truth)."""
    cons = tuple((tuple(sc), frozenset(tuple(t) for t in al)) for sc, al in rec["cons"])
    return C.CSP(rec["n"], rec["d"], cons)


def load_curriculum(path: str) -> list[dict]:
    """Load the jsonl dataset as a list of records (text + exact CSP + query/answer)."""
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def to_induce_problem(rec: dict) -> dict | None:
    """Adapter for the COLORING rung: convert a record into the clair.induce.gen_problem dict
    (facts as (0,v,c)=IS / (1,i,j)=DIFF, plus query/answer/determined/sat), so coloring records
    drop straight into run_augmented's pool. Returns None for non-coloring relations."""
    if rec["relation"] not in ("coloring", "equality") or rec["kind"] != "color":
        return None
    facts = []
    for f in rec["facts"]:
        if f[0] == "pin":
            facts.append((0, f[1], f[2]))
        elif f[0] == "neq":
            facts.append((1, f[1], f[2]))
        else:
            return None                                       # eq has no induce encoding
    return {"n": rec["n"], "k": rec["d"], "facts": facts, "query": rec["query"],
            "answer": rec["answer"], "sat": True,
            "determined": rec["determined"], "capped": False,
            "text": rec["text"], "mentions": rec["mentions"]}


# --------------------------------------------------------------------------- self-check
def self_check(n_per=300, seed=0, verbose=True):
    """numpy self-check: every generated instance is SOLVABLE and its query/answer label matches
    an INDEPENDENT brute-force recount of the solutions. Returns a per-relation stats dict."""
    rng = np.random.default_rng(seed)
    stats = {}
    for rel in GENERATORS:
        det = ab = 0
        for _ in range(n_per):
            p = gen_problem(rng, relation=rel)
            csp = p.csp
            sols = C.solutions(csp)
            assert len(sols) >= 1, f"{rel}: UNSOLVABLE instance (witness-first invariant broken)"
            qvals = {s[p.query] for s in sols}                # independent recount
            if p.determined:
                assert len(qvals) == 1 and next(iter(qvals)) == p.answer, \
                    f"{rel}: determined label wrong (qvals={qvals} answer={p.answer})"
                det += 1
            else:
                assert len(qvals) > 1, f"{rel}: abstain label wrong (qvals={qvals})"
                ab += 1
            # facts <-> CSP round-trip is exact: norm of rebuilt csp solutions unchanged
        stats[rel] = {"determined": det, "abstain": ab, "n": n_per}
        if verbose:
            print(f"  {rel:11s} ok: {det:4d} determined  {ab:4d} abstain  "
                  f"(all {n_per} solvable, labels exact)")
    return stats


if __name__ == "__main__":
    print("clair.curriculum self-check (witness-first; exact labels via clair.csp)\n")
    self_check()
    print("\nexample canonical renderings (one per relation):")
    rng = np.random.default_rng(7)
    for rel in GENERATORS:
        p = gen_problem(rng, relation=rel, det_target=(rel in ("arithmetic", "alldiff")))
        print(f"\n[{rel}]  query={p.entity(p.query)}  answer={canonical_answer(p)!r}")
        print("  " + canonical_render(p))
    print("\nALL SELF-CHECKS PASS — ground truth is exact and Bedrock-independent.")
