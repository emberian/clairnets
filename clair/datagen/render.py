"""clair/datagen/render.py — a non-stilted, rule-based CSP -> natural-English renderer.

The whole point of this file is QUALITY. Rule-based text usually reads robotic; we fight
that by composing MANY independent randomness axes so no two renderings collapse onto the
same template:

  * SKIN        — a story-world (graph coloring / scheduling / circuit / ... see skins.py),
                  each with its own entity nouns, value words and relation verbs.
  * NAMING      — entities surface as bare letters, proper names, greek/NATO words, cities,
                  or "noun + label"/"noun + number" schemes.
  * PHRASING    — every relation has many surface forms (generic + skin-specific); one is
                  sampled per fact, with natural negation/aggregation.
  * AGGREGATION — cliques of `differ`/`equal` facts are sometimes stated as "pairwise
                  distinct"/"all equal" (logic-preserving: it normalizes to the same pairs).
  * STRUCTURE   — facts are shuffled, chunked into sentences of varying length joined by
                  varied connectives, and laid out as flowing prose, a bulleted list, a
                  numbered list, or a semicolon run. An optional framing intro is prepended.

Everything is seeded/deterministic given an ``rng``. The companion parser (parse.py) reads
the text back to facts for the QC faithfulness gate, so the renderer stays honest.

A future paraphrase/naturalizer step (DiffusionGemma) can post-process ``Rendering.text``;
the rule-based path is designed to read well on its own.
"""
from __future__ import annotations

import itertools as it
from dataclasses import dataclass

from . import skins as SK

# --------------------------------------------------------------------------- generic phrasings
# Placeholders: {a} {b} {c} entity surfaces · {v} value surface · {list} entity list · {d} modulus.
# GENERIC forms are deliberately VALUE-AGNOSTIC (no "color"/"slot" words) so they read correctly for
# every skin; skins inject domain flavour via Skin.verbs, which are PREPENDED to these.
GENERIC = {
    "eq": [
        "{a} equals {b}", "{a} is the same as {b}", "{a} and {b} are equal",
        "{a} agrees with {b}", "{a} and {b} take the same value", "{a} matches {b}",
    ],
    "neq": [
        "{a} differs from {b}", "{a} and {b} are different",
        "{a} and {b} must differ", "{a} cannot equal {b}",
        "{a} and {b} take different values", "{a} is not equal to {b}",
    ],
    "lt": [
        "{a} comes before {b}", "{a} is before {b}", "{a} precedes {b}",
        "{a} is earlier than {b}", "{a} ranks ahead of {b}",
    ],
    "le": [
        "{a} comes before {b} or in the same position",
        "{a} is no later than {b}", "{a} is at or before {b}",
        "{a} is before {b} or ties with {b}",
    ],
    "sum": [
        "{a} plus {b} equals {c}, modulo {d}",
        "the sum of {a} and {b} is {c}, mod {d}",
        "{c} equals {a} plus {b} (mod {d})",
    ],
    "xor": [
        "{a}, {b} and {c} have even parity",
        "{c} is the XOR of {a} and {b}",
        "{a} exclusive-or {b} equals {c}",
        "{a}, {b} and {c} XOR to zero",
    ],
    "par": [
        "{list} have even parity",
        "the parity of {list} is even",
        "an even number of {list} are one",
        "{list} XOR together to zero",
    ],
    "parm_odd": [
        "{list} have odd parity",
        "the parity of {list} is odd",
        "an odd number of {list} are one",
        "{list} XOR together to one",
    ],
    "alldiff": [
        "{list} are all different", "{list} are pairwise distinct",
        "{list} must all take different values", "no two of {list} are the same",
        "{list} are mutually distinct",
    ],
    "eqclique": [
        "{list} are all equal", "{list} all take the same value", "{list} are all identical",
    ],
}

# pin phrasings depend on the VALUE STYLE (what a value "is"), not the relation.
PIN = {
    "colors": ["{a} is {v}", "{a} is colored {v}", "{a} gets the color {v}",
               "{a} must be {v}", "{a} takes the color {v}", "the color of {a} is {v}"],
    "channels": ["{a} is on {v}", "{a} is assigned {v}", "{a} broadcasts on {v}", "{a} uses {v}"],
    "registers": ["{a} lives in {v}", "{a} is assigned {v}", "{a} is allocated to {v}", "{a} uses {v}"],
    "teams": ["{a} joins {v}", "{a} is on {v}", "{a} plays for {v}"],
    "slots": ["{a} runs in {v}", "{a} is scheduled for {v}", "{a} takes {v}", "{a} occupies {v}"],
    "places": ["{a} finished {v}", "{a} came {v}", "{a} placed {v}", "{a} took {v}"],
    "terms": ["{a} is taken in {v}", "{a} is scheduled for {v}", "{a} lands in {v}"],
    "steps": ["{a} happens at {v}", "{a} occurs at {v}", "{a} is at {v}"],
    "digits": ["{a} = {v}", "{a} equals {v}", "{a} is {v}", "{a} has the value {v}", "{a} holds {v}"],
    "bits": ["{a} = {v}", "{a} is {v}", "{a} reads {v}", "{a} is set to {v}"],
}

CONNECT = ["", "Also, ", "Moreover, ", "In addition, ", "Furthermore, ",
           "We also know that ", "Additionally, ", "Besides, ", "On top of that, "]

# questions are chosen by VALUE STYLE so they match the skin's vocabulary.
QUESTIONS = {
    "colors": ["What color is {q}?", "Which color must {q} take?",
               "What color does {q} get?", "Determine the color of {q}."],
    "channels": ["Which channel does {q} use?", "What channel is {q} assigned?",
                 "Determine the channel for {q}."],
    "registers": ["Which register holds {q}?", "Where is {q} allocated?",
                  "What register does {q} get?"],
    "teams": ["Which team is {q} on?", "What team does {q} join?",
              "Which team must {q} play for?"],
    "slots": ["Which slot does {q} run in?", "When is {q} scheduled?",
              "What slot does {q} take?"],
    "places": ["Where did {q} place?", "What position did {q} finish in?",
               "Which place does {q} take?"],
    "terms": ["In which term is {q} taken?", "When is {q} scheduled?",
              "Which term does {q} land in?"],
    "steps": ["When does {q} happen?", "At which step is {q}?", "Where in the order is {q}?"],
    "digits": ["What is the value of {q}?", "What value must {q} take?",
               "Determine {q}.", "What does {q} equal?"],
    "bits": ["What bit is {q}?", "Is {q} a 0 or a 1?", "What value must {q} take?", "Determine {q}."],
}


# --------------------------------------------------------------------------- value surfaces
_PLURAL = {"vertex": "vertices", "class": "classes", "process": "processes"}


def pluralize(noun: str) -> str:
    if not noun:
        return ""
    return _PLURAL.get(noun, noun + ("es" if noun.endswith(("s", "x", "ch", "sh")) else "s"))


def _ordinal_word(i: int) -> str:
    n = i + 1
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def value_surfaces(style: str, d: int) -> list[str]:
    """Surface strings for values 0..d-1 under a skin's value style. Parser is told these."""
    if style == "colors":
        return SK.COLORS[:d]
    if style == "channels":
        return [f"channel {i + 1}" for i in range(d)]
    if style == "registers":
        return [f"R{i}" for i in range(d)]
    if style == "teams":
        return [f"Team {SK.COLORS[i].capitalize()}" for i in range(d)]
    if style == "slots":
        return [f"slot {i + 1}" for i in range(d)]
    if style == "places":
        return [_ordinal_word(i) for i in range(d)]
    if style == "terms":
        return [f"term {i + 1}" for i in range(d)]
    if style == "steps":
        return [f"step {i + 1}" for i in range(d)]
    # bits / digits / fallback
    return [str(i) for i in range(d)]


# --------------------------------------------------------------------------- naming schemes
def _entity_surfaces(skin: SK.Skin, n: int, rng) -> tuple[list[str], str, str]:
    """Return (surfaces[index]->str, scheme_tag, noun)."""
    noun = str(rng.choice(skin.nouns)) if skin.nouns else ""
    schemes = ["bare_pool"]
    if noun:
        schemes += ["noun_pool", "noun_num"]
    if "letters" in skin.pools:
        schemes.append("bare_letters")
    scheme = str(rng.choice(schemes))

    if scheme == "noun_num":
        return [f"{noun} {i + 1}" for i in range(n)], scheme, noun
    if scheme == "bare_letters":
        return SK.LETTERS[:n], scheme, noun

    poolkey = str(rng.choice(skin.pools)) if skin.pools else "letters"
    pool = list(SK.POOLS[poolkey])
    pick = list(rng.choice(len(pool), size=n, replace=False))
    names = [pool[i] for i in pick]
    if scheme == "noun_pool":
        return [f"{noun} {nm}" for nm in names], scheme, noun
    return names, scheme, noun


# --------------------------------------------------------------------------- clique aggregation
def _cliques(pairs: set, min_size: int) -> list[tuple]:
    """Greedy maximal-clique cover of an undirected edge set. Returns vertex-tuples whose every
    internal pair is in `pairs`; each returned clique's edges are fully covered (so removing them
    keeps the normalized fact-set identical). Edges not in any returned clique stay atomic."""
    adj = {}
    for a, b in pairs:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    used = set()
    out = []
    for v in sorted(adj):
        if v in used:
            continue
        clique = {v}
        for u in sorted(adj[v]):
            if u in used:
                continue
            if all((min(u, w), max(u, w)) in pairs for w in clique):
                clique.add(u)
        if len(clique) >= min_size:
            out.append(tuple(sorted(clique)))
            used |= clique
    return out


# --------------------------------------------------------------------------- units
@dataclass
class Unit:
    kind: str               # pin/eq/neq/lt/le/sum/xor/par/alldiff/eqclique
    args: tuple             # entity indices
    value: int = -1         # for pin


def _plan_units(facts, kind, rng, aggregate=True) -> list[Unit]:
    """Turn the raw fact list into render units, optionally aggregating differ/equal cliques."""
    neq = {(min(f[1], f[2]), max(f[1], f[2])) for f in facts if f[0] == "neq"}
    eq = {(min(f[1], f[2]), max(f[1], f[2])) for f in facts if f[0] == "eq"}
    neq_cl = _cliques(neq, 3) if (aggregate and rng.random() < 0.7) else []
    eq_cl = _cliques(eq, 3) if (aggregate and rng.random() < 0.6) else []
    neq_used = {(min(a, b), max(a, b)) for cl in neq_cl for a, b in it.combinations(cl, 2)}
    eq_used = {(min(a, b), max(a, b)) for cl in eq_cl for a, b in it.combinations(cl, 2)}

    units = []
    for f in facts:
        k = f[0]
        if k == "pin":
            units.append(Unit("pin", (f[1],), f[2]))
        elif k == "eq":
            if (min(f[1], f[2]), max(f[1], f[2])) not in eq_used:
                units.append(Unit("eq", (f[1], f[2])))
        elif k == "neq":
            if (min(f[1], f[2]), max(f[1], f[2])) not in neq_used:
                units.append(Unit("neq", (f[1], f[2])))
        elif k in ("lt", "le"):
            units.append(Unit(k, (f[1], f[2])))
        elif k == "sum":
            units.append(Unit("sum", (f[1], f[2], f[3])))
        elif k == "xor":
            units.append(Unit("xor", (f[1], f[2], f[3])))
        elif k == "par":
            units.append(Unit("par", tuple(f[1])))
        elif k == "parm":
            units.append(Unit("parm", tuple(f[1]), int(f[2])))   # value carries the parity rhs
        elif k == "alldiff":
            units.append(Unit("alldiff", tuple(f[1])))
    for cl in neq_cl:
        units.append(Unit("alldiff", cl))
    for cl in eq_cl:
        units.append(Unit("eqclique", cl))
    return units


# --------------------------------------------------------------------------- clause rendering
def _entity_list(surfs, idxs, rng) -> str:
    # non-oxford on purpose: a list never contains ", and ", so the parser can split facts on
    # ", and " / "; " / ", while " without ever slicing a list mid-way.
    names = [surfs[i] for i in idxs]
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _templates(skin: SK.Skin, factkind: str) -> list[str]:
    if factkind == "pin":
        base = PIN.get(skin.value_style or "digits", PIN["digits"])
    else:
        base = GENERIC.get(factkind, [])
    return list(skin.verbs.get(factkind, [])) + list(base)


def _render_unit(u: Unit, skin: SK.Skin, surfs, vals, noun, nouns, d, rng) -> str:
    if u.kind == "parm":                                       # parity with explicit rhs: even/odd
        forms = (skin.verbs.get("par", []) + GENERIC["par"]) if u.value == 0 else GENERIC["parm_odd"]
        return str(rng.choice(forms)).format(list=_entity_list(surfs, u.args, rng),
                                             d=d, noun=noun, nouns=nouns)
    fk = "alldiff" if u.kind == "alldiff" else u.kind
    t = str(rng.choice(_templates(skin, fk)))
    sub = {"d": d, "noun": noun, "nouns": nouns}
    if u.kind == "pin":
        sub["a"] = surfs[u.args[0]]
        sub["v"] = vals[u.value]
    elif u.kind in ("eq", "neq", "lt", "le"):
        sub["a"], sub["b"] = surfs[u.args[0]], surfs[u.args[1]]
    elif u.kind in ("sum", "xor"):
        sub["a"], sub["b"], sub["c"] = (surfs[i] for i in u.args)
    elif u.kind in ("par", "alldiff", "eqclique"):
        sub["list"] = _entity_list(surfs, u.args, rng)
    return t.format(**sub)


# --------------------------------------------------------------------------- assembly
def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def _assemble(clauses, intro, question, rng) -> tuple[str, str]:
    """Lay the rendered clauses out in one of several structures. Returns (text, structure_tag)."""
    structure = str(rng.choice(["prose", "prose", "semicolon", "bullets", "numbered"]))

    if structure in ("bullets", "numbered"):
        lines = []
        for i, c in enumerate(clauses, 1):
            bullet = f"{i}. " if structure == "numbered" else "- "
            lines.append(bullet + _cap(c) + ".")
        head = (intro + "\n") if intro else ""
        return f"{head}" + "\n".join(lines) + f"\n\n{question}", structure

    if structure == "semicolon":
        body = "; ".join(clauses)
        head = (intro + " ") if intro else ""
        return f"{head}{_cap(body)}. {question}", structure

    # prose: chunk clauses into sentences of 1..3, varied connectives
    order = clauses[:]
    sentences = []
    i = 0
    first = True
    while i < len(order):
        take = 1 + int(rng.integers(0, 3)) if len(order) - i >= 2 else 1
        chunk = order[i:i + take]
        i += take
        core = chunk[0]                           # clauses are mid-sentence case; cap at the end
        for c in chunk[1:]:                        # join extra facts with a varied connective
            core += str(rng.choice([", and ", ", while ", "; "])) + c
        if not first and rng.random() < 0.35:
            sent = str(rng.choice(CONNECT[1:])) + core   # connective is already capitalized
        else:
            sent = _cap(core)
        sentences.append(sent.rstrip() + ".")
        first = False
    head = (intro + " ") if intro else ""
    return f"{head}" + " ".join(sentences) + f" {question}", structure


# --------------------------------------------------------------------------- public API
@dataclass
class Rendering:
    text: str
    skin: str
    kind: str
    entities: list      # surface per entity index
    values: list        # surface per value index
    scheme: str
    structure: str
    question: str


def render_problem(p, rng, aggregate=True, allowed=None) -> Rendering:
    """Render one curriculum.Problem (or hard_tasks.Problem) into varied natural English.

    `allowed` (a set of skin keys) restricts which skins may dress this problem — used by the
    dataset driver to keep a fixed pool of skins for train and a DISJOINT pool held out for the
    OOD-phrasing split. The restriction is applied before any rng draw; callers that need a strict
    guarantee should check `rendering.skin in allowed` and drop records that fell back."""
    factkinds = {("alldiff" if f[0] == "alldiff" else f[0]) for f in p.facts}
    cands = SK.skins_for(p.kind, factkinds)
    if allowed is not None:
        cands = [s for s in cands if s.key in allowed]
    if not cands:                                # fall back to any skin of the right kind
        cands = [s for s in SK.SKINS if s.kind == p.kind]
        if allowed is not None:
            cands = [s for s in cands if s.key in allowed] or cands
        cands = cands or SK.SKINS
    skin = cands[int(rng.integers(len(cands)))]

    surfs, scheme, noun = _entity_surfaces(skin, p.n, rng)
    nouns = pluralize(noun)
    vals = value_surfaces(skin.value_style or "digits", p.d)

    units = _plan_units(p.facts, p.kind, rng, aggregate=aggregate)
    rng.shuffle(units)
    clauses = [_render_unit(u, skin, surfs, vals, noun, nouns, p.d, rng) for u in units]

    qstyle = skin.value_style if (skin.value_style in QUESTIONS) else "digits"
    question = str(rng.choice(QUESTIONS[qstyle])).format(q=surfs[p.query])

    intro = ""
    if skin.intros and rng.random() < 0.6:
        intro = str(rng.choice(skin.intros)).format(
            noun=noun or "item", nouns=nouns or "items", d=p.d)

    text, structure = _assemble(clauses, intro, question, rng)
    return Rendering(text, skin.key, p.kind, surfs, vals, scheme, structure, question)
