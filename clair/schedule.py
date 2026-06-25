"""clair/schedule.py — the VERIFIER-LADDER scheduler: one multi-task loader over all rungs.

The program is "a learned proposer + a ladder of increasingly powerful EXACT verifiers". This module
mixes the rungs into a single training stream under a configurable schedule, behind ONE uniform
problem-record interface so RLVR's reward is exact for every rung:

    Item(rung, prompt, answer, verify, difficulty, label, meta)
        prompt   : the NL/canonical question text (Bedrock diverse-rendering would later swap in here)
        answer   : the canonical EXACT answer string (the label)
        verify   : verify(candidate)->bool, an EXACT reward closure recomputed from ground truth
        difficulty: the rung's size/OOD knob in [0,1]
        label    : the semantic class (forced value / abstain / entail / contradict / unknown / ...)

The rungs (each its own exact verifier):
    csp-coloring  curriculum coloring         (clair.csp enumeration)
    ordering      curriculum ordering         (clair.csp enumeration)
    arithmetic    curriculum modular sums     (clair.csp enumeration)
    curriculum-nl any curriculum relation     (clair.csp; the diverse-NL surface rung)
    fol           Datalog entailment          (clair.fol forward-chainer, 3-way entail/contradict/unknown)
    smt           linear-integer arithmetic   (clair.smt z3 oracle, determined/underdetermined/unsat)

SCHEDULES (make_schedule(spec)):
    uniform     fixed mixture weights + fixed difficulty.
    curriculum  anneal start->end: shifts BOTH the mixture (easy/synthetic rungs early -> harder/
                fuzzier rungs late) AND the difficulty knob, on a step counter.

Pure Python + numpy + z3 (smt). No torch, no Bedrock call. Run:  python -m clair.schedule
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import csp as C
from . import curriculum as CU
from . import fol as FOL
from . import smt as SMT

RUNGS = ["csp-coloring", "ordering", "arithmetic", "curriculum-nl", "fol", "smt"]
_CURRIC_REL = {"csp-coloring": "coloring", "ordering": "ordering", "arithmetic": "arithmetic"}


# --------------------------------------------------------------------------- uniform item interface
@dataclass
class Item:
    rung: str
    prompt: str
    answer: str
    verify: Callable[[object], bool]
    difficulty: float
    label: str = ""
    meta: dict = field(default_factory=dict)


def _norm(s) -> str:
    return str(s).strip().lower()


# --------------------------------------------------------------------------- per-rung generators
def _curric_item(rng, relation, difficulty, rung) -> Item:
    """A curriculum-CSP rung (clair.csp is the exact verifier). difficulty scales the cell count.

    The verifier is BRUTE-FORCE solution enumeration (d^n), so large-domain relations (ordering /
    arithmetic / alldiff) are kept tiny; only the cheap small-domain relations scale with difficulty."""
    if relation in ("ordering", "arithmetic", "alldiff"):
        lo, hi = 4, 5                                     # large d ⇒ keep n small (d^n stays ~<=16k)
    else:
        lo = 4 + int(round(difficulty * 3))              # coloring/equality: small domain, cheap
        hi = lo + 2
    n, d, kind, facts, s = CU.GENERATORS[relation](rng, n_lo=lo, n_hi=hi)
    assert CU.facts_satisfied_by(facts, s, d), "witness violated a fact (generator bug)"
    csp = CU.build_csp(n, d, facts)
    q, ans, det = CU._query_answer(csp, facts, rng, bool(rng.random() < 0.5))
    p = CU.Problem(relation, n, d, kind, facts, q, ans, det, CU.value_names(kind, d))
    answer = CU.canonical_answer(p)                      # exact: forced value name OR "cannot be determined"

    forced = C.exact_dedP(csp, csp.full())[q]            # capture exact ground truth in the closure
    abstain = {"cannot be determined", "unknown", "undetermined"}

    def verify(cand, forced=forced, p=p):
        c = _norm(cand)
        if len(forced) == 1:
            return c == _norm(p.vnames[next(iter(forced))])
        return c in abstain

    # NOTE: Bedrock CU.render_diverse(p) would replace canonical_render here for the NL surface rung.
    return Item(rung, CU.canonical_render(p), answer, verify, float(difficulty),
                label=("forced" if det else "abstain"),
                meta={"relation": relation, "n": n, "d": d, "query": q})


def _fol_item(rng, difficulty, rung="fol") -> Item:
    depth = 1 + int(round(difficulty * 3))               # proof depth 1..4
    p = FOL.gen_problem(rng, depth=(depth if rng.random() < 0.6 else None))
    return Item(rung, FOL.render(p), p.label, FOL.make_verify(p), float(difficulty),
                label=p.label, meta={"depth": p.depth, "query": str(p.query)})


def _smt_item(rng, difficulty, rung="smt") -> Item:
    n = 3 + int(round(difficulty * 2))                   # 3..5 vars
    R = 6 + int(round(difficulty * 12))                  # range 6..18
    p = SMT.gen_problem(rng, n=n, R=R)
    return Item(rung, SMT.render(p), SMT.canonical_answer(p), SMT.make_verify(p),
                float(difficulty), label=p.label, meta={"n": n, "R": R, "query": p.query})


def gen_item(rng, rung, difficulty=0.5) -> Item:
    """Produce one Item for the named rung at the given difficulty. The single dispatch point."""
    if rung in _CURRIC_REL:
        return _curric_item(rng, _CURRIC_REL[rung], difficulty, rung)
    if rung == "curriculum-nl":
        rel = str(rng.choice(list(CU.GENERATORS)))       # any relation; the diverse-NL surface rung
        return _curric_item(rng, rel, difficulty, "curriculum-nl")
    if rung == "fol":
        return _fol_item(rng, difficulty)
    if rung == "smt":
        return _smt_item(rng, difficulty)
    raise ValueError(f"unknown rung {rung!r}")


# --------------------------------------------------------------------------- schedule
def _weights(spec, frac):
    """Mixture weights at progress `frac` in [0,1]. uniform: fixed; curriculum: lerp start->end."""
    rungs = spec["rungs"]
    if spec["mode"] == "uniform":
        w = spec.get("weights") or {r: 1.0 for r in rungs}
        v = np.array([w.get(r, 0.0) for r in rungs], float)
    else:
        a = spec["start"]
        b = spec["end"]
        v = np.array([(1 - frac) * a.get(r, 0.0) + frac * b.get(r, 0.0) for r in rungs], float)
    return v / v.sum()


def _difficulty(spec, frac):
    if spec["mode"] == "uniform":
        return float(spec.get("difficulty", 0.5))
    d0, d1 = spec.get("difficulty", (0.0, 1.0))
    return float((1 - frac) * d0 + frac * d1)


class Schedule:
    """Reproducible iterator of Items under a mixture/difficulty schedule. Infinite unless `steps`."""

    def __init__(self, spec):
        self.spec = spec
        self.rng = np.random.default_rng(spec.get("seed", 0))
        self.steps = spec.get("steps")
        self.horizon = spec.get("anneal_steps") or self.steps or 1000
        self.t = 0

    def _frac(self):
        return min(1.0, self.t / max(1, self.horizon))

    def sample(self) -> Item:
        frac = self._frac()
        w = _weights(self.spec, frac)
        rung = self.spec["rungs"][int(self.rng.choice(len(w), p=w))]
        return gen_item(self.rng, rung, _difficulty(self.spec, frac))

    def __iter__(self):
        return self

    def __next__(self) -> Item:
        if self.steps is not None and self.t >= self.steps:
            raise StopIteration
        item = self.sample()
        self.t += 1
        return item


def make_schedule(spec: dict) -> Schedule:
    """Build a Schedule from a spec. Defaults fill in a sensible curriculum if start/end omitted.

    uniform:    {"mode":"uniform", "weights":{rung:w,...}?, "difficulty":0.5?, "steps":N?, "seed":0?}
    curriculum: {"mode":"curriculum", "start":{...}?, "end":{...}?, "difficulty":(0.0,1.0)?,
                 "steps":N?, "anneal_steps":H?, "seed":0?}
    """
    spec = dict(spec)
    spec.setdefault("rungs", list(RUNGS))
    if spec["mode"] == "curriculum":
        # default anneal: synthetic/easy rungs early -> fuzzier/harder rungs late
        spec.setdefault("start", {"csp-coloring": 3, "ordering": 2, "arithmetic": 3,
                                  "fol": 3, "curriculum-nl": 1, "smt": 1})
        spec.setdefault("end", {"csp-coloring": 1, "ordering": 1, "arithmetic": 1,
                                "fol": 2, "curriculum-nl": 3, "smt": 3})
        spec.setdefault("difficulty", (0.0, 1.0))
    return Schedule(spec)


# --------------------------------------------------------------------------- self-check
def self_check(verbose=True) -> bool:
    ok = True

    # (1) every rung yields a well-formed Item whose verify accepts truth + rejects a wrong answer
    rng = np.random.default_rng(0)
    for rung in RUNGS:
        for _ in range(80):
            it = gen_item(rng, rung, difficulty=float(rng.random()))
            if not it.verify(it.answer):
                ok = False
                if verbose:
                    print(f"  {rung}: verify REJECTED its own exact answer {it.answer!r}")
                break
            if it.verify("__definitely_wrong__"):
                ok = False
                if verbose:
                    print(f"  {rung}: verify ACCEPTED a garbage answer")
                break
        else:
            if verbose:
                print(f"  {rung:13s} ok: 80 items, verify exact (accepts truth, rejects garbage)")
            continue
        break

    # (2) uniform schedule respects the mixture (only listed rungs, roughly its weights)
    sch = make_schedule({"mode": "uniform", "weights": {"fol": 1, "smt": 1},
                         "rungs": ["fol", "smt"], "steps": 200, "seed": 1})
    seen = [it.rung for it in sch]
    ok &= set(seen) == {"fol", "smt"} and len(seen) == 200
    if verbose:
        print(f"  uniform(fol,smt) over 200: rungs={set(seen)} "
              f"fol={seen.count('fol')} smt={seen.count('smt')}")

    # (3) curriculum schedule shifts the mixture early->late and anneals difficulty up
    sch = make_schedule({"mode": "curriculum", "steps": 600, "seed": 2})
    items = list(sch)
    early = items[:150]
    late = items[-150:]
    syn_early = sum(i.rung in ("csp-coloring", "arithmetic", "fol") for i in early)
    syn_late = sum(i.rung in ("csp-coloring", "arithmetic", "fol") for i in late)
    fuzzy_early = sum(i.rung in ("curriculum-nl", "smt") for i in early)
    fuzzy_late = sum(i.rung in ("curriculum-nl", "smt") for i in late)
    d_early = np.mean([i.difficulty for i in early])
    d_late = np.mean([i.difficulty for i in late])
    shift = (syn_early > syn_late) and (fuzzy_late > fuzzy_early) and (d_late > d_early)
    ok &= shift
    if verbose:
        print(f"  curriculum anneal: synthetic {syn_early}->{syn_late}, fuzzy {fuzzy_early}->{fuzzy_late}, "
              f"difficulty {d_early:.2f}->{d_late:.2f}  shift={'OK' if shift else 'NO'}")
        print(f"  every item carries an exact verify(): RLVR reward is exact across all {len(RUNGS)} rungs")
    return ok


if __name__ == "__main__":
    print("clair.schedule self-check (multi-task verifier ladder; exact verify per rung)\n")
    rng = np.random.default_rng(11)
    for rung in RUNGS:
        it = gen_item(rng, rung, difficulty=0.6)
        print(f"[{rung}] answer={it.answer!r} label={it.label} diff={it.difficulty}")
        print("  " + it.prompt[:160] + ("..." if len(it.prompt) > 160 else ""))
    print()
    ok = self_check()
    print("\n" + ("PASS — all rungs share the Item/verify interface; both schedules behave."
                  if ok else "FAIL"))
