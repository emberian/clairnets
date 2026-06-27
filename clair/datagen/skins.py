"""clair/datagen/skins.py — thematic vocabulary for the non-stilted renderer.

A *skin* is a story-world that a structured CSP can be dressed in: a domain `kind`
(color / ordinal / number), a set of entity nouns + name pools, a value vocabulary,
some framing/intro sentences, and OPTIONAL skin-specific relation phrasings (`verbs`)
that override the generic templates in ``render.py``.

Variety comes from composing MANY independent axes, and the skin supplies several of
them (framing, entity wordlist, value words, relation lexicon). The renderer adds the
rest (naming scheme, phrasing choice, clause order, aggregation, structure).

Everything here is pure data. The companion parser (``parse.py``) is told the concrete
entity/value surfaces a record used, so any skin parses back faithfully.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- name pools
LETTERS = [chr(ord("A") + i) for i in range(26)]
PEOPLE = [
    "Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Heidi", "Ivan",
    "Judy", "Karl", "Lena", "Mona", "Nate", "Olive", "Pavel", "Quinn", "Rosa",
    "Sven", "Tara", "Umar", "Vera", "Wendy", "Xander", "Yara", "Zane",
]
GREEK = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi",
    "Rho", "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega",
]
NATO = [
    "Alfa", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel",
    "India", "Juliet", "Kilo", "Lima", "Mike", "November", "Oscar", "Papa",
    "Quebec", "Romeo", "Sierra", "Tango", "Uniform", "Victor", "Whiskey", "Xray",
]
CITIES = [
    "Ashford", "Brookline", "Calder", "Dunmore", "Elmridge", "Fairhaven",
    "Glenwood", "Harlow", "Ironvale", "Jorvik", "Kestrel", "Lindholm",
    "Maywood", "Norwell", "Oakdale", "Pinehurst", "Quarry", "Redfield",
    "Stonegate", "Thornwick", "Underhill", "Vexley", "Westmark", "Yarrow",
]

COLORS = [
    "red", "green", "blue", "yellow", "orange", "purple",
    "cyan", "pink", "teal", "magenta", "olive", "navy",
]


# --------------------------------------------------------------------------- the Skin
@dataclass(frozen=True)
class Skin:
    key: str
    kind: str                       # color | ordinal | number
    label: str                      # human description (for the intro)
    nouns: tuple = ()               # entity nouns, e.g. ("node", "vertex"); () = bare names
    pools: tuple = ()               # name-pool keys the skin may draw entity names from
    relations: frozenset | None = None  # fact-kinds the skin is willing to dress (None = any)
    value_style: str = ""           # how to render value surfaces (see render.value_surfaces)
    intros: tuple = ()              # framing sentences (the question/body follow)
    verbs: dict = field(default_factory=dict)   # factkind -> [templates], overrides generic


POOLS = {
    "letters": LETTERS, "people": PEOPLE, "greek": GREEK, "nato": NATO, "cities": CITIES,
}


# --------------------------------------------------------------------------- skin library
# Each skin's `verbs` add domain-flavoured phrasings on top of render.py's generic templates.
# Placeholders: {a} {b} {c} entities · {v} value · {list} an entity list · {d} modulus.
SKINS: list[Skin] = [
    # ---------- color kind ----------
    Skin(
        "graph_coloring", "color", "a graph that must be properly colored",
        nouns=("node", "vertex"), pools=("letters", "greek", "nato"),
        relations=frozenset({"pin", "eq", "neq", "alldiff"}),
        value_style="colors",
        intros=(
            "Consider a graph whose {nouns} each need a color.",
            "We must color the {nouns} of a small graph.",
            "Here is a coloring task over a handful of {nouns}.",
            "A graph needs a proper coloring; the {nouns} and their constraints follow.",
        ),
        verbs={
            "neq": [
                "{a} and {b} are adjacent, so they need different colors",
                "{a} and {b} share an edge and must be colored differently",
                "there is an edge between {a} and {b}, so they cannot match",
                "{a} and {b} are neighbors and must differ in color",
            ],
            "eq": [
                "{a} and {b} are tied together and must share a color",
                "{a} and {b} are forced into the same color",
            ],
            "alldiff": [
                "the {nouns} {list} form a clique, so they all get different colors",
                "{list} are mutually adjacent and must all be colored differently",
            ],
        },
    ),
    Skin(
        "map_regions", "color", "a map whose regions must be shaded",
        nouns=("region", "territory", "province"), pools=("cities", "greek", "letters"),
        relations=frozenset({"pin", "eq", "neq", "alldiff"}),
        value_style="colors",
        intros=(
            "A map must be shaded so that neighboring {nouns} look distinct.",
            "We are shading the {nouns} of a map.",
            "Color the {nouns} of this map under the usual neighbor rule.",
        ),
        verbs={
            "neq": [
                "{a} and {b} share a border, so they must be shaded differently",
                "{a} borders {b}, so the two cannot use the same shade",
                "{a} and {b} are adjacent {nouns} and need distinct colors",
            ],
            "eq": ["{a} and {b} belong to the same zone and share a shade"],
            "alldiff": ["{list} all border one another and must be shaded differently"],
        },
    ),
    Skin(
        "frequency", "color", "transmitters that need clear channels",
        nouns=("tower", "transmitter", "station"), pools=("letters", "nato", "cities"),
        relations=frozenset({"pin", "eq", "neq", "alldiff"}),
        value_style="channels",
        intros=(
            "Several {nouns} must be assigned broadcast channels without interference.",
            "Assign a channel to each {noun} so interfering pairs stay apart.",
            "We allocate frequencies to a set of {nouns}.",
        ),
        verbs={
            "neq": [
                "{a} and {b} would interfere, so they need separate channels",
                "{a} and {b} overlap in range and must use different channels",
                "{a} and {b} cannot broadcast on the same channel",
            ],
            "eq": ["{a} and {b} are linked and must use the same channel"],
            "alldiff": ["{list} all interfere with one another and need distinct channels"],
        },
    ),
    Skin(
        "register_alloc", "color", "program values competing for registers",
        nouns=("value", "variable", "temporary"), pools=("letters", "greek"),
        relations=frozenset({"pin", "eq", "neq", "alldiff"}),
        value_style="registers",
        intros=(
            "A compiler must place each live {noun} into a register.",
            "Allocate registers to the {nouns} below.",
            "We assign machine registers to a set of {nouns}.",
        ),
        verbs={
            "neq": [
                "{a} and {b} are live at the same time, so they cannot share a register",
                "{a} and {b} interfere and must land in different registers",
                "{a} and {b} overlap in lifetime and need separate registers",
            ],
            "eq": ["{a} and {b} are coalesced into the same register"],
            "alldiff": ["{list} are all simultaneously live and need distinct registers"],
        },
    ),
    Skin(
        "social_teams", "color", "people sorted into teams",
        nouns=("player", "person", "member"), pools=("people", "nato"),
        relations=frozenset({"pin", "eq", "neq", "alldiff"}),
        value_style="teams",
        intros=(
            "A group of {nouns} is being split into teams.",
            "We are sorting {nouns} onto teams given their rivalries and alliances.",
            "Assign each {noun} to a team.",
        ),
        verbs={
            "neq": [
                "{a} and {b} are rivals, so they must be on different teams",
                "{a} and {b} refuse to be on the same team",
                "{a} and {b} cannot be teamed together",
            ],
            "eq": [
                "{a} and {b} are friends and want to be on the same team",
                "{a} and {b} insist on sharing a team",
            ],
            "alldiff": ["{list} all dislike each other and must be on different teams"],
            "pin": ["{a} joins {v}", "{a} is placed on {v}", "{a} ends up on {v}"],
        },
    ),

    # ---------- ordinal kind ----------
    Skin(
        "scheduling", "ordinal", "jobs placed into time slots",
        nouns=("job", "task", "operation"), pools=("letters", "nato", "greek"),
        relations=frozenset({"pin", "lt", "le"}),
        value_style="slots",
        intros=(
            "A scheduler must place each {noun} into a time slot.",
            "We are ordering a set of {nouns} in time.",
            "Several {nouns} must run in some order; the constraints follow.",
        ),
        verbs={
            "lt": [
                "{a} must finish before {b} starts",
                "{a} has to run before {b}",
                "{a} is scheduled strictly earlier than {b}",
            ],
            "le": [
                "{a} runs no later than {b}",
                "{a} starts before {b} or in the same slot",
            ],
            "pin": ["{a} runs in {v}", "{a} is scheduled for {v}", "{a} takes {v}"],
        },
    ),
    Skin(
        "race", "ordinal", "runners finishing a race",
        nouns=("runner", "racer", "competitor"), pools=("people", "nato"),
        relations=frozenset({"pin", "lt", "le"}),
        value_style="places",
        intros=(
            "We are reconstructing the finishing order of a race.",
            "A race has just finished; here is what we know about the order.",
            "Work out where each {noun} placed.",
        ),
        verbs={
            "lt": [
                "{a} finished ahead of {b}",
                "{a} crossed the line before {b}",
                "{a} beat {b}",
                "{a} placed higher than {b}",
            ],
            "le": [
                "{a} finished no worse than {b}",
                "{a} placed ahead of {b} or tied",
            ],
            "pin": ["{a} finished {v}", "{a} came {v}", "{a} took {v}"],
        },
    ),
    Skin(
        "prereqs", "ordinal", "courses with prerequisites",
        nouns=("course", "class", "module"), pools=("greek", "letters", "nato"),
        relations=frozenset({"pin", "lt", "le"}),
        value_style="terms",
        intros=(
            "A degree plan orders {nouns} across terms under prerequisite rules.",
            "We are sequencing {nouns} by their prerequisites.",
            "Place each {noun} into a term respecting prerequisites.",
        ),
        verbs={
            "lt": [
                "{a} is a prerequisite of {b}",
                "{a} must be taken before {b}",
                "{a} comes earlier in the plan than {b}",
            ],
            "le": [
                "{a} is taken no later than {b}",
                "{a} comes before {b} or in the same term",
            ],
            "pin": ["{a} is taken in {v}", "{a} is scheduled for {v}", "{a} lands in {v}"],
        },
    ),
    Skin(
        "timeline", "ordinal", "events on a timeline",
        nouns=("event", "milestone", "step"), pools=("letters", "greek", "cities"),
        relations=frozenset({"pin", "lt", "le"}),
        value_style="steps",
        intros=(
            "We are reconstructing the order of several {nouns}.",
            "Put the {nouns} of this story in order.",
            "A timeline must be recovered from the clues below.",
        ),
        verbs={
            "lt": [
                "{a} happened before {b}",
                "{a} occurred earlier than {b}",
                "{a} took place before {b}",
            ],
            "le": [
                "{a} happened no later than {b}",
                "{a} occurred before {b} or at the same time",
            ],
            "pin": ["{a} happened at {v}", "{a} occurred at {v}", "{a} is at {v}"],
        },
    ),

    # ---------- number kind ----------
    Skin(
        "circuit", "number", "logic nets on a board",
        nouns=("net", "wire", "gate"), pools=("letters", "nato"),
        relations=frozenset({"pin", "xor", "par", "parm", "sum"}),
        value_style="bits",
        intros=(
            "A small logic board wires several {nouns} together.",
            "We are reasoning about the bits on a set of {nouns}.",
            "Here is a circuit over a few {nouns}.",
        ),
        verbs={
            "xor": [
                "{c} is the XOR of {a} and {b}",
                "gate {c} drives high exactly when {a} and {b} differ",
                "{a}, {b} and {c} are wired through a parity gate (their XOR is 0)",
            ],
            "par": [
                "{list} feed a parity gate that holds even",
                "an even number of {list} are high",
                "the parity over {list} is even",
            ],
            "pin": ["{a} is held at {v}", "{a} is tied to {v}", "{a} reads {v}"],
        },
    ),
    Skin(
        "boolean_logic", "number", "boolean variables with parity rules",
        nouns=("bit", "flag", "signal"), pools=("letters", "greek"),
        relations=frozenset({"pin", "xor", "par", "parm"}),
        value_style="bits",
        intros=(
            "A handful of {nouns} obey parity rules.",
            "We track boolean {nouns} under XOR constraints.",
            "Some {nouns} are linked by parity; deduce the rest.",
        ),
        verbs={
            "xor": [
                "{a}, {b} and {c} XOR to zero",
                "{a} exclusive-or {b} equals {c}",
                "{c} flips exactly when {a} and {b} disagree",
            ],
            "par": [
                "{list} have even parity",
                "{list} XOR together to zero",
            ],
            "pin": ["{a} is set to {v}", "{a} is {v}", "{a} reads {v}"],
        },
    ),
    Skin(
        "arithmetic", "number", "variables in modular arithmetic",
        nouns=("variable", "cell", "term"), pools=("letters", "greek"),
        relations=frozenset({"pin", "sum", "eq", "neq", "alldiff"}),
        value_style="digits",
        intros=(
            "Several {nouns} satisfy modular equations.",
            "We solve for {nouns} under arithmetic mod {d}.",
            "A few {nouns} are linked by sums modulo {d}.",
        ),
        verbs={
            "sum": [
                "{a} plus {b} equals {c}, modulo {d}",
                "{c} is the sum of {a} and {b} mod {d}",
                "adding {a} and {b} gives {c} (working mod {d})",
            ],
            "pin": ["{a} = {v}", "{a} equals {v}", "{a} holds {v}", "{a} is fixed at {v}"],
        },
    ),
    Skin(
        "lockbox", "number", "dials on a combination lock",
        nouns=("dial", "wheel", "digit"), pools=("letters", "nato"),
        relations=frozenset({"pin", "sum", "alldiff"}),
        value_style="digits",
        intros=(
            "A combination lock has several {nouns} to set.",
            "We are recovering the setting of each {noun}.",
            "The {nouns} of a lock obey the clues below.",
        ),
        verbs={
            "sum": [
                "{a} and {b} add up to {c} on the lock (mod {d})",
                "{c} is {a} plus {b}, modulo {d}",
            ],
            "alldiff": [
                "{list} are all set to different digits",
                "no two of {list} show the same digit",
            ],
            "pin": ["{a} is set to {v}", "{a} shows {v}", "{a} reads {v}"],
        },
    ),
    Skin(
        "sensors", "number", "sensors with distinct readings",
        nouns=("sensor", "probe", "gauge"), pools=("letters", "nato", "cities"),
        relations=frozenset({"pin", "alldiff", "neq"}),
        value_style="digits",
        intros=(
            "An array of {nouns} reports values that must stay distinct.",
            "We calibrate several {nouns} with unique readings.",
            "Each {noun} reports a value under the constraints below.",
        ),
        verbs={
            "alldiff": [
                "{list} all report different values",
                "no two of {list} read the same",
                "{list} are calibrated to distinct readings",
            ],
            "neq": [
                "{a} and {b} must not read the same value",
                "{a} and {b} report different values",
            ],
            "pin": ["{a} reads {v}", "{a} reports {v}", "{a} is calibrated to {v}"],
        },
    ),
]

SKINS_BY_KEY = {s.key: s for s in SKINS}


def skins_for(kind: str, factkinds: set) -> list[Skin]:
    """Skins whose domain matches `kind` and that are willing to dress every fact-kind used."""
    out = []
    for s in SKINS:
        if s.kind != kind:
            continue
        if s.relations is not None and not factkinds <= s.relations:
            continue
        out.append(s)
    return out
