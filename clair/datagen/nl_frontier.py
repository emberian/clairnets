"""clair/datagen/nl_frontier.py — the NL-FRONTIER training set for the α structure-compiler.

The α-probe (runs/alpha_struct_probe.json) showed the make-or-break gap: α reads CANONICAL
structure near-perfectly (micro-F1 0.98) but NATURAL LANGUAGE is the open problem — zero-shot
NL F1 0.62, and only 0.85 even when trained on the existing one-rendering-per-record NL. The
diagnosis (notes/alpha_struct_design.md §2.3): α overfits the template phrasings and can't
robustly read the RELATION off real language. This module attacks that with DATA, on the two
named levers:

  1. HARD-NEGATIVE NL PAIRS — for a problem, a phrasing with the SAME entities but a
     MINIMALLY-CHANGED relation (flip eq↔neq, lt↔le, swap the operands of a directed relation,
     or add/drop one constraint). Anchor and hard-neg share skin/naming/layout and differ in
     EXACTLY one clause, so the surface is near-identical and only the relation-words move — α
     MUST read the relation off the text rather than memorize entity/position patterns. Each
     pair is tagged with its structural delta (the labeled flip).

  2. PHRASING DIVERSITY — the rule-based renderer (render.py: skins × naming × structure ×
     phrasing) widened across the rungs, PLUS an optional Bedrock LLM naturalizer that rewrites
     the rule-based prose into more natural English while preserving the structure. Every
     naturalized rewrite is re-verified (parse.py round-trip) and DROPPED if the recovered
     structure ≠ the original. The rule-based path stands alone when Bedrock is unavailable.

  3. AN NL-OOD SPLIT — held-out skins (disjoint lexicon, mirroring build.HELDOUT_SKINS) and a
     held-out naturalizer STYLE, so α's NL generalization is MEASURED, not just trained on.

EXACT-VERIFIED. Every emitted record's structure label is re-recovered by the independent
keyword parser (parse.parse_facts) and compared to the ground-truth facts (qc.norm_logical);
mismatches are dropped. Records are written in the curriculum-record schema that
run_alpha_struct_probe.py already consumes (relation/n/d/kind/facts/cons/query/answer/
determined/vnames/text/mentions + NL-frontier tags), so the next α-NL-training run just
points the probe at train.jsonl (train) and ood.jsonl (eval).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

from .. import csp as C
from .. import curriculum as CU
from . import parse as P
from . import qc
from . import render as RD
from . import skins as SK

# --------------------------------------------------------------------------- skin partition
# Mirrors build.py: TRAIN skins dress the train split; HELD-OUT skins dress ONLY the NL-OOD
# split, so phrasing/lexicon is a clean generalization axis (no skin overlap train↔ood).
TRAIN_SKINS = frozenset({
    "graph_coloring", "map_regions", "register_alloc",     # color
    "scheduling", "race",                                   # ordinal
    "lockbox", "sensors",                                   # number (alldiff)
})
HELDOUT_SKINS = frozenset({
    "frequency", "social_teams",                           # color
    "prereqs", "timeline",                                 # ordinal
    # (number/alldiff held-out skin: none distinct in skins.py, so alldiff OOD relies on the
    #  Bedrock held-out STYLE; color/ordinal carry the skin-disjoint OOD load.)
})

# the binary+pin rungs the α CSP head supports (run_alpha_struct_probe.BINARY_RUNGS).
RUNGS = ("coloring", "equality", "ordering", "alldiff")
BINARY_KINDS = {"pin", "eq", "neq", "lt", "le"}


# --------------------------------------------------------------------------- entity mentions
def surface_mentions(text: str, entities) -> dict[int, list]:
    """Char spans of each entity SURFACE in `text`, keyed by entity index. Mirrors parse._mask's
    matching (longest-first alternation, word-bounded, + the sentence-capitalized variant) so the
    spans align exactly with what the faithfulness parser saw — which is what the α probe pools
    the host hidden over (oracle_readout._mention_tensor)."""
    lut = {}
    for i, s in enumerate(entities):
        lut.setdefault(s, i)
        lut.setdefault(P._cap(s), i)
    if not lut:
        return {}
    import re
    alt = "|".join(re.escape(s) for s in sorted(lut, key=len, reverse=True))
    pat = re.compile(r"(?<!\w)(" + alt + r")(?!\w)")
    out: dict[int, list] = {}
    for m in pat.finditer(text):
        out.setdefault(lut[m.group(1)], []).append((m.start(), m.end()))
    return out


# --------------------------------------------------------------------------- problem helpers
def _factkinds(facts) -> set:
    return {("alldiff" if f[0] == "alldiff" else f[0]) for f in facts}


def _gen_rung(rng, rung):
    """One verified Problem on a binary+pin rung (denser than the defaults so determined-via-
    propagation queries exist), restricted to facts the α CSP head reads (pin/eq/neq/lt/le/alldiff)."""
    if rung == "coloring":
        n, d, kind, facts, s = CU.gen_coloring(rng, edge_p=0.6, pin_frac=0.5)
    elif rung == "equality":
        n, d, kind, facts, s = CU.gen_equality(rng, pin_frac=0.45)
    elif rung == "ordering":
        n, d, kind, facts, s = CU.gen_ordering(rng, pin_frac=0.45)
    else:                                              # alldiff
        n, d, kind, facts, s = CU.gen_alldiff(rng, pin_frac=0.5)
    assert CU.facts_satisfied_by(facts, s, d)
    csp = CU.build_csp(n, d, facts)
    q, ans, det = CU._query_answer(csp, facts, rng, bool(rng.random() < 0.55))
    return CU.Problem(rung, n, d, kind, facts, q, ans, det, CU.value_names(kind, d))


# --------------------------------------------------------------------------- matched renderer
def _unit_for_fact(f) -> RD.Unit:
    k = f[0]
    if k == "pin":
        return RD.Unit("pin", (f[1],), f[2])
    if k in ("eq", "neq", "lt", "le"):
        return RD.Unit(k, (f[1], f[2]))
    if k == "alldiff":
        return RD.Unit("alldiff", tuple(f[1]))
    raise ValueError(k)


def _pick_skin(kind, factkinds, allowed, rng):
    cands = [s for s in SK.skins_for(kind, factkinds) if s.key in allowed]
    if not cands:
        cands = [s for s in SK.SKINS if s.kind == kind and s.key in allowed]
    if not cands:
        cands = SK.skins_for(kind, factkinds) or [s for s in SK.SKINS if s.kind == kind]
    return cands[int(rng.integers(len(cands)))]


def _question(skin, surfs, query, rng):
    qstyle = skin.value_style if (skin.value_style in RD.QUESTIONS) else "digits"
    return str(rng.choice(RD.QUESTIONS[qstyle])).format(q=surfs[query])


def render_facts(facts, skin, surfs, vals, noun, nouns, d, rng):
    """Render a fact list to per-fact clauses (fact order, NO aggregation/shuffle), so a single
    fact edit maps to a single clause edit — the substrate for matched hard-negative pairs."""
    return [RD._render_unit(_unit_for_fact(f), skin, surfs, vals, noun, nouns, d, rng)
            for f in facts]


# --------------------------------------------------------------------------- hard-neg edits
# Each edit returns (neg_facts, edit_pos, delta_tag): neg_facts differs from facts in EXACTLY
# one fact at position edit_pos (or one appended/removed fact), with a labeled structural delta.
def _binary_positions(facts):
    return [i for i, f in enumerate(facts) if f[0] in ("eq", "neq", "lt", "le")]


def edit_flip_eqneq(facts, rng, n, rung):
    cands = [i for i, f in enumerate(facts) if f[0] in ("eq", "neq")]
    if not cands:
        return None
    i = int(rng.choice(cands))
    k, a, b = facts[i]
    nk = "neq" if k == "eq" else "eq"
    neg = list(facts); neg[i] = (nk, a, b)
    return neg, i, f"{k}->{nk}"


def edit_flip_ltle(facts, rng, n, rung):
    cands = [i for i, f in enumerate(facts) if f[0] in ("lt", "le")]
    if not cands:
        return None
    i = int(rng.choice(cands))
    k, a, b = facts[i]
    nk = "le" if k == "lt" else "lt"
    neg = list(facts); neg[i] = (nk, a, b)
    return neg, i, f"{k}->{nk}"


def edit_swap_operands(facts, rng, n, rung):
    cands = [i for i, f in enumerate(facts) if f[0] in ("lt", "le")]
    if not cands:
        return None
    i = int(rng.choice(cands))
    k, a, b = facts[i]
    neg = list(facts); neg[i] = (k, b, a)
    return neg, i, f"swap_operands({k})"


def edit_drop(facts, rng, n, rung):
    cands = _binary_positions(facts)
    if not cands:
        return None
    i = int(rng.choice(cands))
    neg = [f for j, f in enumerate(facts) if j != i]
    return neg, i, f"drop_{facts[i][0]}"


def edit_add(facts, rng, n, rung):
    """Add one constraint on a currently-unconstrained pair (kind appropriate to the rung)."""
    used = {tuple(sorted((f[1], f[2]))) for f in facts if f[0] in ("eq", "neq", "lt", "le")}
    free = [(a, b) for a in range(n) for b in range(a + 1, n) if (a, b) not in used]
    if not free:
        return None
    a, b = free[int(rng.integers(len(free)))]
    kind = "lt" if rung == "ordering" else "neq"
    neg = list(facts) + [(kind, a, b)]
    return neg, len(facts), f"add_{kind}"


EDITS = {
    "coloring": [edit_flip_eqneq, edit_add, edit_drop],
    "equality": [edit_flip_eqneq, edit_add, edit_drop],
    "ordering": [edit_flip_ltle, edit_swap_operands, edit_add, edit_drop],
    "alldiff":  [edit_add, edit_drop],
}


# --------------------------------------------------------------------------- record builder
def _ser_facts(facts):
    out = []
    for f in facts:
        if f[0] == "alldiff":
            out.append(["alldiff", [int(x) for x in f[1]]])
        else:
            out.append([f[0]] + [int(x) for x in f[1:]])
    return out


def _cons_of(n, d, facts):
    csp = CU.build_csp(n, d, facts)
    return [[[int(c) for c in sc], sorted([[int(x) for x in t] for t in al])] for sc, al in csp.cons]


def make_record(rng, relation, n, d, kind, facts, surfs, vals, text, split, source,
                role, tags=None):
    """A curriculum-schema record (consumable by run_alpha_struct_probe.load_csp_records /
    rec_view) plus NL-frontier tags. query/answer/determined are derived exactly from the facts."""
    csp = CU.build_csp(n, d, facts)
    q, ans, det = CU._query_answer(csp, facts, rng, True)
    ment = surface_mentions(text, surfs)
    rec = {
        "relation": relation, "kind": kind, "n": int(n), "d": int(d),
        "facts": _ser_facts(facts),
        "cons": _cons_of(n, d, facts),
        "query": int(q), "answer": int(ans), "determined": bool(det),
        "vnames": list(vals),
        "text": text,
        "entities": list(surfs), "values": list(vals),
        "mentions": {str(k): [[int(a), int(b)] for (a, b) in v] for k, v in ment.items()},
        "source": source, "split": split, "role": role,
    }
    if tags:
        rec.update(tags)
    return rec


def _mentions_ok(rec) -> bool:
    """Every entity that appears in a fact (or is the query) must have at least one mention span,
    else the α probe pools nothing for that cell and the structure target is unsupervised."""
    need = {rec["query"]}
    for f in rec["facts"]:
        if f[0] == "alldiff":
            need |= set(f[1])
        else:
            need |= set(f[1:])
    have = {int(k) for k in rec["mentions"]}
    return need <= have


def verify_structure(text, surfs, vals, facts) -> bool:
    """Independent re-recovery: the keyword parser reads `text` back to facts; the recovered
    structure must equal the intended facts (aggregation-invariant). Drops any rephrasing /
    naturalization whose recovered factor graph ≠ the original."""
    parsed = P.parse_facts(text, surfs, vals)
    if parsed is None:
        return False
    return qc.norm_logical(parsed) == qc.norm_logical(facts)


# --------------------------------------------------------------------------- generation
def gen_hardneg_pair(rng, rung, allowed, seed):
    """Build a matched (anchor, hard-neg) pair: same skin/surfaces/layout, differing in exactly
    one clause (the edited relation). Returns (anchor_rec_args, neg_rec_args, delta) or None."""
    p = _gen_rung(rng, rung)
    facts = [tuple(f) if f[0] != "alldiff" else ("alldiff", tuple(f[1])) for f in p.facts]
    edits = EDITS[rung]
    res = None
    for ed in rng.permutation(len(edits)):
        out = edits[int(ed)](facts, rng, p.n, rung)
        if out is not None:
            res = out
            break
    if res is None:
        return None
    neg_facts, pos, delta = res
    union_fk = _factkinds(facts) | _factkinds(neg_facts)
    skin = _pick_skin(p.kind, union_fk, allowed, rng)
    surfs, scheme, noun = RD._entity_surfaces(skin, p.n, rng)
    nouns = RD.pluralize(noun)
    vals = RD.value_surfaces(skin.value_style or "digits", p.d)
    rclause_rng = np.random.default_rng(seed ^ 0x9E3779B9)
    anchor_clauses = render_facts(facts, skin, surfs, vals, noun, nouns, p.d, rclause_rng)
    # build the hard-neg clause list as a one-clause edit of the anchor's clauses.
    if delta.startswith("drop_"):
        neg_clauses = [c for j, c in enumerate(anchor_clauses) if j != pos]
    elif delta.startswith("add_"):
        added = render_facts([neg_facts[-1]], skin, surfs, vals, noun, nouns, p.d, rclause_rng)
        neg_clauses = anchor_clauses + added
    else:                                              # flip / swap: re-render the single clause
        new_clause = render_facts([neg_facts[pos]], skin, surfs, vals, noun, nouns, p.d, rclause_rng)
        neg_clauses = list(anchor_clauses); neg_clauses[pos] = new_clause[0]
    intro = ""
    irng = np.random.default_rng(seed ^ 0x85EBCA77)
    if skin.intros and irng.random() < 0.5:
        intro = str(irng.choice(skin.intros)).format(noun=noun or "item", nouns=nouns or "items", d=p.d)
    question = _question(skin, surfs, p.query, np.random.default_rng(seed ^ 0xC2B2AE3D))
    a_text, _ = RD._assemble(anchor_clauses, intro, question, np.random.default_rng(seed ^ 0x27D4EB2F))
    n_text, _ = RD._assemble(neg_clauses, intro, question, np.random.default_rng(seed ^ 0x27D4EB2F))
    if not (verify_structure(a_text, surfs, vals, facts) and
            verify_structure(n_text, surfs, vals, neg_facts)):
        return None
    anchor = dict(relation=rung, n=p.n, d=p.d, kind=p.kind, facts=facts, surfs=surfs,
                  vals=vals, text=a_text, skin=skin.key, scheme=scheme)
    neg = dict(relation=rung, n=p.n, d=p.d, kind=p.kind, facts=neg_facts, surfs=surfs,
               vals=vals, text=n_text, skin=skin.key, scheme=scheme)
    return anchor, neg, delta


def gen_diverse(rng, rung, allowed):
    """One free diverse rendering (full render.py richness) of a fresh problem on the rung."""
    p = _gen_rung(rng, rung)
    r = RD.render_problem(p, rng, allowed=allowed)
    if r.skin not in allowed:
        return None
    facts = [tuple(f) if f[0] != "alldiff" else ("alldiff", tuple(f[1])) for f in p.facts]
    if not verify_structure(r.text, r.entities, r.values, facts):
        return None
    return dict(relation=rung, n=p.n, d=p.d, kind=p.kind, facts=facts, surfs=r.entities,
                vals=r.values, text=r.text, skin=r.skin, scheme=r.scheme)


# --------------------------------------------------------------------------- Bedrock naturalizer
def _resolve_chat():
    try:
        from gowexp.bedrock import chat
        return chat
    except Exception:
        for cand in (os.environ.get("GOWEXP_SRC"), "/Users/ember/dev/gowexp/src"):
            if cand and os.path.isdir(cand) and cand not in sys.path:
                sys.path.insert(0, cand)
        try:
            from gowexp.bedrock import chat
            return chat
        except Exception:
            return None


_NAT_GUIDE = (
    "Keep EVERY entity name and EVERY value name EXACTLY as written (do not rename, abbreviate, "
    "pluralize, or merge them). State exactly the same constraints — add nothing, drop nothing, "
    "never solve the puzzle, and NEVER summarize several facts into a blanket claim (do not say "
    "'all are different' / 'each is unique' / 'no two share'). Keep ONE clause per stated fact and "
    "express each relation EXPLICITLY and unambiguously (e.g. 'is the same as', 'differs from' / "
    "'must be different', 'comes before', 'before or in the same position'). Keep the final "
    "question. Return ONLY the rewritten puzzle text."
)
_NAT_SYS_TRAIN = (
    "You rewrite a constraint puzzle into NATURAL, fluent English — vary the word choice, sentence "
    "flow, connectives, and framing so it reads like prose. " + _NAT_GUIDE
)
# The OOD naturalizer STYLE is held out from train (a clean 'unseen naturalizer' generalization axis).
_NAT_SYS_OOD = (
    "Rewrite this constraint puzzle in a TERSE, clipped, almost legalistic register — short "
    "declarative clauses, minimal connectives, numbered if natural. " + _NAT_GUIDE
)


def naturalize(chat, text, style, model, region, temperature=0.5):
    sys_p = _NAT_SYS_OOD if style == "ood" else _NAT_SYS_TRAIN
    r = chat(model, sys_p, "PUZZLE:\n" + text, max_tokens=500, temperature=temperature,
             region=region, top_p=0.95)
    return r["text"].strip(), (r["in_tokens"], r["out_tokens"])


# --------------------------------------------------------------------------- diversity report
def diversity(texts, rng):
    return {
        "n": len(texts),
        "distinct_1": round(qc.distinct_n(texts, 1), 4),
        "distinct_2": round(qc.distinct_n(texts, 2), 4),
        "distinct_3": round(qc.distinct_n(texts, 3), 4),
        "self_bleu": round(qc.self_bleu(texts, rng), 4),
    }


# --------------------------------------------------------------------------- driver
def build(out_dir, n_problems=1500, hardneg_frac=0.5, seed=0, bedrock=False,
          bedrock_frac=0.25, model=CU.DEFAULT_MODEL, region=CU.DEFAULT_REGION, max_bedrock=400):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    train, ood = [], []
    pairs = []                      # (train) hard-neg pair index: [{pair_id, delta, anchor_id, neg_id}]
    canonical_baseline = []         # canonical_render of the train problems (the "old template")
    drop = Counter()
    chat = _resolve_chat() if bedrock else None
    bedrock_avail = chat is not None
    nat_attempt = nat_kept = 0
    bedrock_tokens = [0, 0]

    splits = [("train", TRAIN_SKINS, train), ("ood", HELDOUT_SKINS, ood)]
    rid = 0
    for split, skins, bucket in splits:
        target = n_problems if split == "train" else max(300, n_problems // 4)
        n_hard = int(target * hardneg_frac)
        # ---- hard-negative matched pairs ----
        made = 0
        guard = 0
        while made < n_hard and guard < n_hard * 20 + 200:
            guard += 1
            rung = RUNGS[int(rng.integers(len(RUNGS)))]
            if rung not in EDITS:
                continue
            res = gen_hardneg_pair(rng, rung, skins, seed=int(rng.integers(1 << 31)))
            if res is None:
                drop["hardneg_unverified"] += 1
                continue
            anchor, neg, delta = res
            pair_id = f"{split}-pair-{made}"
            a_rec = make_record(rng, anchor["relation"], anchor["n"], anchor["d"], anchor["kind"],
                                anchor["facts"], anchor["surfs"], anchor["vals"], anchor["text"],
                                split, "rule:matched", "hardneg_anchor",
                                tags={"skin": anchor["skin"], "scheme": anchor["scheme"],
                                      "pair_id": pair_id, "delta": delta})
            n_rec = make_record(rng, neg["relation"], neg["n"], neg["d"], neg["kind"],
                                neg["facts"], neg["surfs"], neg["vals"], neg["text"],
                                split, "rule:matched", "hardneg_neg",
                                tags={"skin": neg["skin"], "scheme": neg["scheme"],
                                      "pair_id": pair_id, "delta": delta})
            if not (_mentions_ok(a_rec) and _mentions_ok(n_rec)):
                drop["hardneg_mentions"] += 1
                continue
            a_rec["id"] = f"{split}-{rid}"; rid += 1
            n_rec["id"] = f"{split}-{rid}"; rid += 1
            bucket.append(a_rec); bucket.append(n_rec)
            pairs.append({"split": split, "pair_id": pair_id, "delta": delta,
                          "anchor_id": a_rec["id"], "neg_id": n_rec["id"]}) if split == "train" else None
            made += 1
        # ---- diverse single renderings ----
        made = 0
        guard = 0
        n_div = target - n_hard
        while made < n_div and guard < n_div * 20 + 200:
            guard += 1
            rung = RUNGS[int(rng.integers(len(RUNGS)))]
            d = gen_diverse(rng, rung, skins)
            if d is None:
                drop["diverse_unverified"] += 1
                continue
            rec = make_record(rng, d["relation"], d["n"], d["d"], d["kind"], d["facts"],
                              d["surfs"], d["vals"], d["text"], split, "rule:diverse", "diverse",
                              tags={"skin": d["skin"], "scheme": d["scheme"]})
            if not _mentions_ok(rec):
                drop["diverse_mentions"] += 1
                continue
            rec["id"] = f"{split}-{rid}"; rid += 1
            bucket.append(rec)
            if split == "train":
                # canonical "old template" baseline for the SAME structure (diversity comparison).
                p = CU.Problem(d["relation"], d["n"], d["d"], d["kind"], d["facts"], 0, -1, False,
                               CU.value_names(d["kind"], d["d"]))
                canonical_baseline.append(CU.canonical_render(p))
            made += 1
        print(f"  [{split}] {len(bucket)} records ({sum(r['role']!='diverse' for r in bucket)} hardneg, "
              f"{sum(r['role']=='diverse' for r in bucket)} diverse)  {time.time()-t0:.0f}s", flush=True)

    # ---- Bedrock naturalization (optional, structure-preserving, drop on mismatch) ----
    if bedrock and bedrock_avail:
        for split, _skins, bucket in splits:
            style = "ood" if split == "ood" else "train"
            # naturalize a fraction of the rule:diverse records (in place additions).
            cands = [r for r in bucket if r["role"] == "diverse"]
            rng.shuffle(cands)
            k = min(len(cands), int(len(cands) * bedrock_frac), max_bedrock)
            extra = []
            for r in cands[:k]:
                nat_attempt += 1
                try:
                    nt, (ti, to) = naturalize(chat, r["text"], style, model, region)
                    bedrock_tokens[0] += ti; bedrock_tokens[1] += to
                except Exception:
                    drop["bedrock_error"] += 1
                    continue
                facts = [tuple(f) if f[0] != "alldiff" else ("alldiff", tuple(f[1]))
                         for f in r["facts"]]
                if not verify_structure(nt, r["entities"], r["values"], facts):
                    drop["bedrock_struct_mismatch"] += 1
                    continue
                nrec = make_record(rng, r["relation"], r["n"], r["d"], r["kind"], facts,
                                   r["entities"], r["values"], nt, split,
                                   f"bedrock:{model}:{style}", "bedrock",
                                   tags={"skin": r.get("skin", ""), "scheme": r.get("scheme", ""),
                                         "from_id": r["id"]})
                if not _mentions_ok(nrec):
                    drop["bedrock_mentions"] += 1
                    continue
                nrec["id"] = f"{split}-{rid}"; rid += 1
                extra.append(nrec)
                nat_kept += 1
            bucket.extend(extra)
            print(f"  [bedrock:{split}] kept {len(extra)}/{k} naturalized  {time.time()-t0:.0f}s",
                  flush=True)

    # ---- write splits ----
    def _write(name, recs):
        with open(os.path.join(out_dir, name), "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
    _write("train.jsonl", train)
    _write("ood.jsonl", ood)
    with open(os.path.join(out_dir, "hardneg_pairs.jsonl"), "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")

    # ---- diversity: old templates (canonical) vs new train (rule) vs new ood, + bedrock ----
    drng = np.random.default_rng(seed + 7)
    train_rule = [r["text"] for r in train if r["source"].startswith("rule")]
    train_bedrock = [r["text"] for r in train if r["role"] == "bedrock"]
    ood_texts = [r["text"] for r in ood]
    div = {
        "old_template_canonical": diversity(canonical_baseline, drng) if canonical_baseline else {},
        "new_train_rule": diversity(train_rule, drng),
        "new_ood": diversity(ood_texts, drng),
    }
    if train_bedrock:
        div["new_train_bedrock"] = diversity(train_bedrock, drng)

    delta_counts = dict(Counter(p["delta"] for p in pairs))
    stats = {
        "seed": seed,
        "sizes": {"train": len(train), "ood": len(ood),
                  "train_hardneg": sum(r["role"] != "diverse" and r["role"] != "bedrock" for r in train),
                  "train_diverse": sum(r["role"] == "diverse" for r in train),
                  "train_bedrock": len(train_bedrock),
                  "ood_bedrock": sum(r["role"] == "bedrock" for r in ood),
                  "hardneg_pairs_train": len(pairs)},
        "hardneg_deltas": delta_counts,
        "by_rung_train": dict(Counter(r["relation"] for r in train)),
        "by_skin_train": dict(Counter(r.get("skin", "") for r in train)),
        "by_skin_ood": dict(Counter(r.get("skin", "") for r in ood)),
        "diversity": div,
        "bedrock": {"available": bedrock_avail, "attempted": nat_attempt, "kept": nat_kept,
                    "drop_rate": round(1 - nat_kept / max(1, nat_attempt), 4),
                    "in_tokens": bedrock_tokens[0], "out_tokens": bedrock_tokens[1],
                    "model": model if bedrock else None},
        "dropped": dict(drop),
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"\nDONE in {time.time()-t0:.0f}s — train {len(train)} ood {len(ood)} "
          f"pairs {len(pairs)}\n  diversity: {json.dumps(div)}", flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser(description="Build the NL-frontier training set for the α structure-compiler.")
    ap.add_argument("--out", default="data/nl_frontier")
    ap.add_argument("--n", type=int, default=1500, help="train problems (ood is n/4)")
    ap.add_argument("--hardneg-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bedrock", action="store_true", help="run the Bedrock naturalizer (optional)")
    ap.add_argument("--bedrock-frac", type=float, default=0.25)
    ap.add_argument("--max-bedrock", type=int, default=400)
    ap.add_argument("--model", default=CU.DEFAULT_MODEL)
    ap.add_argument("--region", default=CU.DEFAULT_REGION)
    a = ap.parse_args()
    build(a.out, n_problems=a.n, hardneg_frac=a.hardneg_frac, seed=a.seed, bedrock=a.bedrock,
          bedrock_frac=a.bedrock_frac, model=a.model, region=a.region, max_bedrock=a.max_bedrock)


if __name__ == "__main__":
    main()
