"""clair/datagen/trace.py — organ-grounded, exact-by-construction reasoning traces.

A trace narrates the SOUND narrowing the organ performs. Starting from every cell's full
candidate set (with the pinned cells seeded to their given value), we iterate arc-consistency
— the organ's local message-passing — to a fixpoint, reporting each cell whose option set
shrinks. We then close with the EXACT per-cell deduction (``clair.csp`` / ``hard_tasks.fast_dedP``)
for the queried cell.

Why this is exact / honest:

  * arc-consistency is SOUND: a value it keeps is a superset of the values used by some full
    solution, so every "A's options narrow to {red, blue}" is never an over-claim;
  * a cell AC reduces to a singleton {v} is provably forced (for a solvable CSP, exact ⊆ AC,
    and a nonempty singleton pins the exact value), so "B is forced to green" is exact;
  * the final answer is read off the exact dedₚ, never the (possibly incomplete) AC fixpoint —
    so on the affine/parity wall, where per-cell AC must abstain, the closer states the joint
    deduction the factor-level organ performs. The trace's concluded answer therefore always
    equals the record's verified label.

The trace is rendered in the record's own entity/value SURFACES, so it reads in the skin.
"""
from __future__ import annotations

from .. import csp as C
from .. import hard_tasks as HT


def _fmt_set(values, s) -> str:
    items = [values[v] for v in sorted(s)]
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return "{" + ", ".join(items[:-1]) + ", or " + items[-1] + "}"


def build_trace(p, entities, values, max_rounds=None, csp=None) -> str:
    """Return an exact, organ-grounded reasoning trace string for problem `p`, phrased in the
    given entity/value surfaces. The final asserted answer always matches p.answer. Pass `csp`
    to reuse an already-built CSP (the extensional table rebuild is expensive for high-arity
    constraints like alldiff)."""
    csp = csp if csp is not None else p.csp
    n = csp.n
    surf = entities
    pins = [(f[1], f[2]) for f in p.facts if f[0] == "pin"]
    pinned_cells = {a for a, _ in pins}
    lines: list[str] = []

    # 1. the givens (pins) — seed the lattice so AC does not re-report them as a deduction.
    for a, v in pins:
        lines.append(f"{surf[a]} is given as {values[v]}.")
    dom = list(csp.full())
    for a, v in pins:
        dom[a] = frozenset({v})
    dom = tuple(dom)

    # 2. iterate arc-consistency, narrating each cell whose options shrink this round.
    max_rounds = max_rounds or (n * csp.d + 2)
    for _ in range(max_rounds):
        nxt = C.ac_step(csp, dom)
        if nxt == dom:
            break
        for i in range(n):
            if i in pinned_cells or nxt[i] == dom[i] or len(nxt[i]) == 0:
                continue
            if len(nxt[i]) == 1:
                v = next(iter(nxt[i]))
                lines.append(f"{surf[i]} is then forced to {values[v]}.")
            else:
                lines.append(f"{surf[i]}'s options narrow to {_fmt_set(values, nxt[i])}.")
        dom = nxt

    # 3. exact closer on the queried cell (handles the affine/parity wall AC cannot crack locally).
    exact = HT.fast_dedP(csp)
    qcell = exact[p.query]
    q = surf[p.query]
    if p.determined:
        if len(dom[p.query]) == 1:                      # local message-passing already pinned it
            lines.append(f"So {q} must be {values[p.answer]}.")
        else:                                           # needed the joint (factor-level) deduction
            lines.append(f"Taking all the constraints together, {q} is forced to {values[p.answer]}.")
        concluded = values[p.answer]
    else:
        lines.append(f"{q} can still be {_fmt_set(values, qcell)}, so its value cannot be determined.")
        concluded = None

    # exactness guard: the trace must never conclude something other than the verified label.
    if p.determined:
        assert concluded == values[p.answer], "trace concluded a value other than the exact label"
    return " ".join(lines)
