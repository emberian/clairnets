"""clair/fol.py — RUNG 3: synthetic FOL / Datalog entailment (ProofWriter-style, but EXACT).

Same discipline as clair.csp / clair.curriculum: the GROUND TRUTH is generated first and is
EXACT/checkable. Here the exact verifier is a FORWARD-CHAINING engine that computes the least
Herbrand model (the fixpoint of rule application over the Herbrand base). A query atom's label is
read straight off that closure:

    ENTAIL      q in closure                    (positively derivable)
    CONTRADICT  (-q) in closure, q not          (the classical negation is derivable)
    UNKNOWN     neither q nor -q derivable       (the native OPEN-WORLD case, ProofWriter's gap)

Closed-world (CWA) collapses UNKNOWN -> NOT-ENTAILED; we keep the 3-way open-world labels because
they exercise abstention (UNKNOWN) the same way clair.curriculum's determined/abstain does.

WITNESS-FIRST: rules + ground facts are sampled, then the forward-chainer COMPUTES the exact entailed
set — the closure *is* the witness, so every emitted label is exact by construction (no separate
solver to get wrong). DIFFICULTY AXIS = proof DEPTH (chain length): a transitive `reach` predicate
over an edge chain of length L gives an entailment at depth exactly L — a built-in size/OOD knob.

Monotone & sound: bodies may contain classical-negation literals (`-p`), but there is NO
negation-as-failure, so the fixpoint stays monotone and the least model is exact. Knowledge bases
are kept CONSISTENT (no atom and its negation both derived) by construction + an explicit check.

Pure Python, no torch, no Bedrock. A diverse-NL surface (Bedrock render_diverse, as in
clair.curriculum) would later wrap `render()` — the symbolic content here is already exact.

Run:  python -m clair.fol      (self-check: brute-force model-enumeration cross-check + label demo)
"""
from __future__ import annotations

import itertools as it
from dataclasses import dataclass

import numpy as np

VARS = frozenset("xyzuvw")                         # rule variables; constants are everything else
UNARY = ["red", "round", "cold", "kind", "heavy", "quiet"]
BINARY = ["near", "likes", "above"]


def is_var(t) -> bool:
    return t in VARS


# --------------------------------------------------------------------------- literals / rules
@dataclass(frozen=True)
class Lit:
    """A (possibly negated) atom. args is a tuple of terms (variables or constants)."""
    neg: bool
    pred: str
    args: tuple

    def __str__(self):
        return f"{'-' if self.neg else ''}{self.pred}({','.join(self.args)})"


@dataclass(frozen=True)
class Rule:
    body: tuple                                     # tuple[Lit] (conjunction)
    head: Lit

    def __str__(self):
        return f"{' & '.join(map(str, self.body))} -> {self.head}"


def atom(pred, *args, neg=False) -> Lit:
    return Lit(neg, pred, tuple(args))


def neg_of(q: Lit) -> Lit:
    return Lit(not q.neg, q.pred, q.args)


# --------------------------------------------------------------------------- the EXACT verifier
def _subst(lit: Lit, b: dict) -> Lit:
    return Lit(lit.neg, lit.pred, tuple(b.get(t, t) for t in lit.args))


def _unify(lit: Lit, fact: Lit, b: dict):
    """Match a (possibly non-ground) body literal against a GROUND fact, extending binding b."""
    if lit.neg != fact.neg or lit.pred != fact.pred or len(lit.args) != len(fact.args):
        return None
    nb = dict(b)
    for t, c in zip(lit.args, fact.args):
        if is_var(t):
            if nb.get(t, c) != c:
                return None
            nb[t] = c
        elif t != c:
            return None
    return nb


def _matches(body, closure, b):
    """Yield every binding under which the whole body is satisfied by ground facts in `closure`."""
    if not body:
        yield b
        return
    head, rest = body[0], body[1:]
    for fact in closure:
        nb = _unify(head, fact, b)
        if nb is not None:
            yield from _matches(rest, closure, nb)


def forward_chain(facts, rules) -> set:
    """THE GROUND-TRUTH VERIFIER. Least Herbrand model = fixpoint of rule application. Exact."""
    closure = set(facts)
    while True:
        snap = frozenset(closure)                   # snapshot: don't mutate during iteration
        new = {_subst(r.head, b) for r in rules for b in _matches(r.body, snap, {})}
        new -= closure
        if not new:
            return closure
        closure |= new


def proof_depths(facts, rules, closure) -> dict:
    """Min proof depth per derived atom: facts depth 0; head = 1 + max(body depths). Fixpoint."""
    INF = 1 << 30
    depth = {f: 0 for f in facts}
    while True:
        changed = False
        snap = frozenset(closure)
        for r in rules:
            for b in _matches(r.body, snap, {}):
                bg = [_subst(l, b) for l in r.body]
                if all(g in closure for g in bg):
                    d = 1 + max((depth.get(g, INF) for g in bg), default=0)
                    h = _subst(r.head, b)
                    if d < depth.get(h, INF):
                        depth[h] = d
                        changed = True
        if not changed:
            return depth


def label_query(closure, q: Lit) -> str:
    """3-way open-world label for a POSITIVE query atom q against the exact closure."""
    if q in closure:
        return "entail"
    if neg_of(q) in closure:
        return "contradict"
    return "unknown"


def consistent(closure) -> bool:
    """No atom and its classical negation both derived."""
    return not any(neg_of(a) in closure for a in closure)


# --------------------------------------------------------------------------- record
@dataclass
class FOLProblem:
    entities: list
    rules: list                                     # list[Rule]
    facts: list                                     # list[Lit] (ground)
    query: Lit                                       # positive ground atom
    label: str                                       # entail | contradict | unknown
    depth: int                                        # proof depth of the query (difficulty knob)

    @property
    def closure(self) -> set:
        return forward_chain(self.facts, self.rules)


# --------------------------------------------------------------------------- generator (witness-first)
def gen_kb(rng, depth: int):
    """Build a KB whose transitive `reach` predicate yields an entailment at depth exactly `depth`,
    plus unary implication chains and a negation gadget (so all three labels are reachable)."""
    L = max(1, int(depth))
    ents = [f"e{i}" for i in range(L + 1 + int(rng.integers(1, 4)))]  # chain + distractors
    rules = [
        Rule((atom("link", "x", "y"),), atom("reach", "x", "y")),
        Rule((atom("link", "x", "y"), atom("reach", "y", "z")), atom("reach", "x", "z")),
    ]
    facts = [atom("link", f"e{i}", f"e{i+1}") for i in range(L)]      # the depth-L chain

    # a unary implication chain p0 -> p1 -> ... (more entail/unknown atoms, extra depth)
    props = list(rng.choice(UNARY, size=min(4, len(UNARY)), replace=False))
    for a, c in zip(props, props[1:]):
        rules.append(Rule((atom(a, "x"),), atom(c, "x")))
    seed_ent = str(rng.choice(ents))
    if props:
        facts.append(atom(props[0], seed_ent))

    # negation gadget: a `trouble` trigger forces a classical-negative conclusion -> CONTRADICT space
    trig = str(rng.choice(ents))
    rules.append(Rule((atom("trouble", "x"),), atom("calm", "x", neg=True)))
    facts.append(atom("trouble", trig))

    # a couple of plain binary facts for surface variety (never collide with reach/calm)
    for _ in range(int(rng.integers(1, 4))):
        r = str(rng.choice(BINARY))
        a, b = (str(x) for x in rng.choice(ents, size=2, replace=True))
        facts.append(atom(r, a, b))
    return ents, rules, list(dict.fromkeys(facts))


def _herbrand_preds(rules, facts):
    """pred -> arity over everything mentioned (for sampling UNKNOWN candidates)."""
    sig = {}
    for f in facts:
        sig[f.pred] = len(f.args)
    for r in rules:
        for l in (*r.body, r.head):
            sig[l.pred] = len(l.args)
    return sig


def gen_problem(rng, label=None, depth=None, max_tries=300) -> FOLProblem:
    """One exact FOL problem. `label` in {entail,contradict,unknown} (random if None);
    `depth` sets the entailment chain length (difficulty). Witness-first: closure is exact."""
    for _ in range(max_tries):
        d = int(depth if depth is not None else rng.integers(1, 5))
        ents, rules, facts = gen_kb(rng, d)
        fset = set(facts)
        closure = forward_chain(facts, rules)
        if not consistent(closure):
            continue
        dep = proof_depths(facts, rules, closure)
        tgt = label or str(rng.choice(["entail", "contradict", "unknown"]))

        # candidate lists are built from `closure` (a set) then index-sampled, so they are sorted by
        # the canonical literal string FIRST — set iteration order is hash-seed dependent, and without
        # this the same (rng, seed) picks different queries across processes (breaks reproducibility).
        if tgt == "entail":
            cand = sorted((a for a in closure if not a.neg and a not in fset), key=str)
            if depth is not None:
                cand = [a for a in cand if dep.get(a, 0) == d] or cand
            if not cand:
                continue
            q = cand[int(rng.integers(len(cand)))]
            return FOLProblem(ents, rules, facts, q, "entail", dep.get(q, 1))

        if tgt == "contradict":
            cand = sorted((neg_of(a) for a in closure if a.neg and neg_of(a) not in closure), key=str)
            if not cand:
                continue
            q = cand[int(rng.integers(len(cand)))]
            return FOLProblem(ents, rules, facts, q, "contradict", dep.get(neg_of(q), 1))

        # unknown: sample a ground positive atom whose atom AND negation are both underivable
        sig = _herbrand_preds(rules, facts)
        preds = sorted(p for p in sig if p != "link")  # link is fully given; skip for fairness
        for _ in range(60):
            p = preds[int(rng.integers(len(preds)))]
            args = tuple(str(x) for x in rng.choice(ents, size=sig[p], replace=True))
            q = atom(p, *args)
            if q not in closure and neg_of(q) not in closure:
                return FOLProblem(ents, rules, facts, q, "unknown", 0)
    raise RuntimeError(f"could not generate FOL problem (label={label}, depth={depth})")


# --------------------------------------------------------------------------- NL surface (canonical)
def render_lit(l: Lit) -> str:
    a = l.args
    base = f"{a[0]} is {l.pred}" if len(a) == 1 else f"{a[0]} {l.pred} {a[1]}"
    return ("it is not the case that " if l.neg else "") + base


def render(p: FOLProblem) -> str:
    """Deterministic English surface. Bedrock diverse-rendering would later wrap this (see
    clair.curriculum.render_diverse) to force robust extraction; the content here is already exact."""
    rules = " ".join(f"If {' and '.join(render_lit(b) for b in r.body)}, "
                     f"then {render_lit(r.head)}." for r in p.rules)
    facts = " ".join(render_lit(f).capitalize() + "." for f in p.facts)
    q = f"Is it true that {render_lit(p.query)}?"
    return f"Rules: {rules} Facts: {facts} {q}"


def to_record(p: FOLProblem) -> dict:
    """Serialize matching the spirit of clair.curriculum.to_record (text + exact label)."""
    return {
        "rung": "fol", "entities": p.entities,
        "rules": [str(r) for r in p.rules], "facts": [str(f) for f in p.facts],
        "query": str(p.query), "label": p.label, "depth": p.depth,
        "text": render(p), "answer_name": p.label,
    }


# --------------------------------------------------------------------------- verify (exact reward)
_LABEL_SYN = {
    "entail": "entail", "entailed": "entail", "yes": "entail", "true": "entail", "proven": "entail",
    "contradict": "contradict", "contradicted": "contradict", "no": "contradict", "false": "contradict",
    "disproven": "contradict", "unknown": "unknown", "unproven": "unknown", "maybe": "unknown",
    "cannot be determined": "unknown", "undetermined": "unknown", "neither": "unknown",
}


def norm_label(s) -> str:
    return _LABEL_SYN.get(str(s).strip().lower(), str(s).strip().lower())


def make_verify(p: FOLProblem):
    """Exact reward closure: recompute the closure (the verifier) and compare candidate to truth."""
    truth = label_query(p.closure, p.query)
    return lambda cand: norm_label(cand) == truth


# --------------------------------------------------------------------------- self-check
def _brute_least_model(facts, rules, herbrand):
    """Independent ground truth: enumerate every rule-closed superset of `facts` over the finite
    Herbrand universe `herbrand`; the least model is their intersection. For tiny KBs only."""
    fset = set(facts)
    others = [h for h in herbrand if h not in fset]
    assert len(others) <= 20, "Herbrand base too big for brute force"
    inter = set(herbrand)
    n_models = 0
    for bits in it.product((0, 1), repeat=len(others)):
        M = fset | {others[i] for i, bit in enumerate(bits) if bit}
        snap = frozenset(M)
        if all(_subst(r.head, b) in M for r in rules for b in _matches(r.body, snap, {})):
            inter &= M
            n_models += 1
    return inter, n_models


def self_check(seed=0, verbose=True) -> bool:
    ok = True

    # (1) brute-force cross-check the forward-chainer on a tiny family/transitivity KB
    rules = [
        Rule((atom("parent", "x", "y"),), atom("ancestor", "x", "y")),
        Rule((atom("parent", "x", "y"), atom("ancestor", "y", "z")), atom("ancestor", "x", "z")),
        Rule((atom("parent", "x", "y"), atom("parent", "y", "z")), atom("grandparent", "x", "z")),
    ]
    ents = ["a", "b", "c"]
    facts = [atom("parent", "a", "b"), atom("parent", "b", "c")]
    closure = forward_chain(facts, rules)
    # search universe = the DERIVED preds (parent atoms are fixed/extensional facts, always present)
    herb = set(facts) | {atom(p, x, y) for p in ("ancestor", "grandparent")
                         for x in ents for y in ents}
    least, nm = _brute_least_model(facts, rules, herb)
    match = least == closure
    ok &= match
    if verbose:
        print(f"  forward-chain vs brute-force least-model: {'MATCH' if match else 'MISMATCH'} "
              f"(|closure|={len(closure)}, {nm} closed models)")
        print(f"    depth-2 entailment ancestor(a,c) entailed? {atom('ancestor','a','c') in closure}; "
              f"depth={proof_depths(facts, rules, closure)[atom('ancestor','a','c')]}")
        print(f"    grandparent(a,c) entailed? {atom('grandparent','a','c') in closure}")
    # UNKNOWN really underivable + brute-force agrees it is absent from the least model
    u = atom("ancestor", "c", "a")
    ok &= (label_query(closure, u) == "unknown") and (u not in least)
    if verbose:
        print(f"    UNKNOWN check ancestor(c,a): label={label_query(closure, u)}, in least model={u in least}")

    # (2) generated labels are exact: re-derive each label via the closure
    rng = np.random.default_rng(seed)
    counts = {"entail": 0, "contradict": 0, "unknown": 0}
    for lab in ("entail", "contradict", "unknown"):
        for _ in range(200):
            p = gen_problem(rng, label=lab)
            if label_query(p.closure, p.query) != p.label:
                ok = False
                break
            if not make_verify(p)(p.label) or make_verify(p)("nonsense_label"):
                ok = False
                break
            counts[lab] += 1
    if verbose:
        print(f"  generated-label exactness: {counts} (each must reach 200)")
    ok &= all(v == 200 for v in counts.values())

    # (3) depth knob: requested depth controls the entailment proof depth
    deep = gen_problem(rng, label="entail", depth=3)
    ok &= deep.depth == 3
    if verbose:
        print(f"  depth knob: requested 3 -> query {deep.query} at proof depth {deep.depth}")
    return ok


if __name__ == "__main__":
    print("clair.fol self-check (forward-chaining entailment; exact 3-way labels)\n")
    rng = np.random.default_rng(7)
    for lab in ("entail", "contradict", "unknown"):
        p = gen_problem(rng, label=lab, depth=3 if lab == "entail" else None)
        print(f"[{lab}] query={p.query}  depth={p.depth}")
        print("  " + render(p)[:300] + ("..." if len(render(p)) > 300 else ""))
    print()
    ok = self_check()
    print("\n" + ("PASS — forward-chainer matches brute force; labels exact; UNKNOWN underivable."
                  if ok else "FAIL"))
