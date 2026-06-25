"""clair/smt.py — RUNG 4: SMT / linear-integer arithmetic, with Z3 as the EXACT oracle.

Same discipline as the lower rungs: ground truth is generated first and is EXACT. Here z3 is BOTH
the generator-check and the verifier. We generate small constraint systems over bounded integers
(pins, ≤, <, sums, differences, modular), then ask z3 the determinacy question for a queried
variable q:

    DETERMINED      q has a UNIQUE value v across all models     (solve, take v, then `q != v` is UNSAT)
    UNDERDETERMINED multiple models disagree on q                (`q != v` is SAT)  -> UNKNOWN / abstain
    UNSAT           the system has no model at all

WITNESS-FIRST: we plant a model m (random integer assignment), then only emit constraints m
satisfies — so the system is SAT by construction and m is always a witness. z3 then decides whether
m's value of q is FORCED (determined) or merely one of several (underdetermined). An explicit
`gen_unsat` adds one contradictory constraint to exercise the UNSAT verdict.

This mirrors clair.curriculum's determined/abstain labeling, but the verifier is a real SMT solver
rather than CSP enumeration, so it scales to arithmetic the extensional CSP can't tabulate.

Self-check: z3's verdicts are internally consistent — DETERMINED ⇒ the unique value equals the
planted m[q] and every model agrees; UNDERDETERMINED ⇒ z3 exhibits ≥2 models disagreeing on q;
UNSAT ⇒ z3 reports unsat. No torch, no Bedrock. A diverse-NL surface would wrap `render()` later.

Run:  python -m clair.smt
"""
from __future__ import annotations

import numpy as np
import z3

# A constraint is a small typed tuple, BOTH rendered to text AND compiled to z3 (exact by construction):
#   ("pin", i, c)             x_i == c
#   ("le",  i, j)             x_i <= x_j
#   ("lt",  i, j)             x_i <  x_j
#   ("sum", i, j, c)          x_i + x_j == c
#   ("diff", i, j, c)         x_i - x_j == c
#   ("mod", i, k, r)          x_i % k == r
#   ("modsum", i, j, k, r)    (x_i + x_j) % k == r
# plus implicit bounds 0 <= x_i <= R on every variable.


def _vars(n):
    return [z3.Int(f"x{i}") for i in range(n)]


def compile_z3(cons, X, R):
    """Compile typed constraints + bounds into a list of z3 expressions over variables X."""
    out = [x >= 0 for x in X] + [x <= R for x in X]
    for c in cons:
        k = c[0]
        if k == "pin":
            out.append(X[c[1]] == c[2])
        elif k == "le":
            out.append(X[c[1]] <= X[c[2]])
        elif k == "lt":
            out.append(X[c[1]] < X[c[2]])
        elif k == "sum":
            out.append(X[c[1]] + X[c[2]] == c[3])
        elif k == "diff":
            out.append(X[c[1]] - X[c[2]] == c[3])
        elif k == "mod":
            out.append(X[c[1]] % c[2] == c[3])
        elif k == "modsum":
            out.append((X[c[1]] + X[c[2]]) % c[3] == c[4])
        else:
            raise ValueError(f"unknown constraint {k!r}")
    return out


def sat_under(m, c, R) -> bool:
    """Does the planted python assignment m satisfy a typed constraint? (witness-first filter)."""
    k = c[0]
    if k == "pin":
        return m[c[1]] == c[2]
    if k == "le":
        return m[c[1]] <= m[c[2]]
    if k == "lt":
        return m[c[1]] < m[c[2]]
    if k == "sum":
        return m[c[1]] + m[c[2]] == c[3]
    if k == "diff":
        return m[c[1]] - m[c[2]] == c[3]
    if k == "mod":
        return m[c[1]] % c[2] == c[3]
    if k == "modsum":
        return (m[c[1]] + m[c[2]]) % c[3] == c[4]
    raise ValueError(k)


# --------------------------------------------------------------------------- THE EXACT ORACLE (z3)
def determinacy(cons, n, R, q):
    """z3 decides the queried variable q. Returns ('determined', v) | ('underdetermined', None)
    | ('unsat', None). This IS the verifier."""
    X = _vars(n)
    base = compile_z3(cons, X, R)
    s = z3.Solver()
    s.add(base)
    if s.check() != z3.sat:
        return "unsat", None
    v = s.model().eval(X[q], model_completion=True).as_long()
    s.push()
    s.add(X[q] != v)
    forced = s.check() == z3.unsat
    s.pop()
    return ("determined", v) if forced else ("underdetermined", None)


def two_models_disagree(cons, n, R, q):
    """For an underdetermined q, return two distinct witness values z3 admits (self-check helper)."""
    X = _vars(n)
    base = compile_z3(cons, X, R)
    s = z3.Solver()
    s.add(base)
    assert s.check() == z3.sat
    v1 = s.model().eval(X[q], model_completion=True).as_long()
    s.add(X[q] != v1)
    assert s.check() == z3.sat
    v2 = s.model().eval(X[q], model_completion=True).as_long()
    return v1, v2


# --------------------------------------------------------------------------- record
class SMTProblem:
    def __init__(self, n, R, cons, query, label, value):
        self.n, self.R, self.cons = n, R, cons
        self.query, self.label, self.value = query, label, value

    def oracle(self):
        return determinacy(self.cons, self.n, self.R, self.query)


# --------------------------------------------------------------------------- generator (witness-first)
def gen_problem(rng, label=None, n=None, R=None, max_tries=200) -> SMTProblem:
    """One verified SMT problem. `label` in {determined,underdetermined,unsat} (random if None).
    Witness-first: plant m, emit only constraints m satisfies; z3 assigns the determinacy label."""
    tgt = label or str(rng.choice(["determined", "underdetermined", "unsat"]))
    if tgt == "unsat":
        return gen_unsat(rng, n=n, R=R)
    for _ in range(max_tries):
        nn = int(n if n is not None else rng.integers(3, 6))
        RR = int(R if R is not None else rng.integers(6, 16))
        m = [int(rng.integers(0, RR + 1)) for _ in range(nn)]
        cons = []
        # pins drive determinacy; relations couple variables; modular adds arithmetic structure
        n_pin = nn - 1 if tgt == "determined" else int(rng.integers(0, max(1, nn - 1)))
        for i in rng.choice(nn, size=min(n_pin, nn), replace=False):
            cons.append(("pin", int(i), m[int(i)]))
        for _ in range(int(rng.integers(nn, 2 * nn))):
            kind = str(rng.choice(["le", "lt", "sum", "diff", "mod", "modsum"]))
            i, j = (int(x) for x in rng.choice(nn, size=2, replace=False))
            if kind == "le" and m[i] <= m[j]:
                cons.append(("le", i, j))
            elif kind == "lt" and m[i] < m[j]:
                cons.append(("lt", i, j))
            elif kind == "sum":
                cons.append(("sum", i, j, m[i] + m[j]))
            elif kind == "diff":
                cons.append(("diff", i, j, m[i] - m[j]))
            elif kind == "mod":
                k = int(rng.integers(2, 5))
                cons.append(("mod", i, k, m[i] % k))
            elif kind == "modsum":
                k = int(rng.integers(2, 5))
                cons.append(("modsum", i, j, k, (m[i] + m[j]) % k))
        cons = list(dict.fromkeys(cons))
        assert all(sat_under(m, c, RR) for c in cons), "planted witness violates a constraint"
        pinned = {c[1] for c in cons if c[0] == "pin"}
        unpinned = [i for i in range(nn) if i not in pinned]
        # for determined, prefer an UNPINNED query (answer needs propagation, not lookup)
        pool = (unpinned or list(range(nn))) if tgt == "determined" else list(range(nn))
        q = int(rng.choice(pool))
        lab, v = determinacy(cons, nn, RR, q)
        if lab == tgt:
            return SMTProblem(nn, RR, cons, q, lab, v)
    # fall back to whatever z3 says for the last system (still exact, just not the requested label)
    return SMTProblem(nn, RR, cons, q, lab, v)


def gen_unsat(rng, n=None, R=None) -> SMTProblem:
    """Plant a SAT system, then add ONE contradictory constraint -> guaranteed UNSAT (z3-confirmed)."""
    nn = int(n if n is not None else rng.integers(3, 6))
    RR = int(R if R is not None else rng.integers(6, 16))
    m = [int(rng.integers(0, RR + 1)) for _ in range(nn)]
    i, j = (int(x) for x in rng.choice(nn, size=2, replace=False))
    cons = [("pin", i, m[i]), ("pin", j, m[j]), ("sum", i, j, m[i] + m[j] + 1)]  # i+j must be one more
    lab, _ = determinacy(cons, nn, RR, i)
    assert lab == "unsat", "gen_unsat produced a satisfiable system"
    return SMTProblem(nn, RR, cons, i, "unsat", None)


# --------------------------------------------------------------------------- NL surface (canonical)
def render_con(c) -> str:
    V = lambda i: f"x{i}"
    k = c[0]
    if k == "pin":
        return f"{V(c[1])} = {c[2]}"
    if k == "le":
        return f"{V(c[1])} <= {V(c[2])}"
    if k == "lt":
        return f"{V(c[1])} < {V(c[2])}"
    if k == "sum":
        return f"{V(c[1])} + {V(c[2])} = {c[3]}"
    if k == "diff":
        return f"{V(c[1])} - {V(c[2])} = {c[3]}"
    if k == "mod":
        return f"{V(c[1])} mod {c[2]} = {c[3]}"
    if k == "modsum":
        return f"({V(c[1])} + {V(c[2])}) mod {c[3]} = {c[4]}"
    raise ValueError(k)


def render(p: SMTProblem) -> str:
    body = "; ".join(render_con(c) for c in p.cons)
    return (f"Integers 0..{p.R}. Given: {body}. "
            f"What is the value of x{p.query} (or is it not uniquely determined)?")


def canonical_answer(p: SMTProblem) -> str:
    if p.label == "determined":
        return str(p.value)
    if p.label == "unsat":
        return "no solution"
    return "cannot be determined"


def to_record(p: SMTProblem) -> dict:
    return {
        "rung": "smt", "n": p.n, "R": p.R, "cons": [list(c) for c in p.cons],
        "query": p.query, "label": p.label, "value": p.value,
        "text": render(p), "answer_name": canonical_answer(p),
    }


# --------------------------------------------------------------------------- verify (exact reward)
def make_verify(p: SMTProblem):
    """Exact reward closure. z3 re-decides; for DETERMINED the candidate must equal the unique value
    (z3-re-checked), else it must name the right abstain/unsat verdict."""
    lab, v = p.oracle()

    def verify(cand) -> bool:
        s = str(cand).strip().lower()
        if lab == "determined":
            try:
                return int(s) == v
            except ValueError:
                return False
        if lab == "unsat":
            return s in ("no solution", "unsat", "none", "no")
        return s in ("cannot be determined", "unknown", "undetermined", "underdetermined")

    return verify


# --------------------------------------------------------------------------- self-check
def self_check(seed=0, verbose=True) -> bool:
    rng = np.random.default_rng(seed)
    ok = True
    counts = {"determined": 0, "underdetermined": 0, "unsat": 0}
    for lab in ("determined", "underdetermined", "unsat"):
        for _ in range(150):
            p = gen_problem(rng, label=lab)
            rlab, rv = p.oracle()
            if rlab != p.label:
                ok = False
                break
            if rlab == "determined":
                # internal consistency: unique value == planted witness's value, and q!=v is unsat
                X = _vars(p.n)
                s = z3.Solver()
                s.add(compile_z3(p.cons, X, p.R) + [X[p.query] != rv])
                if s.check() != z3.unsat:
                    ok = False
                    break
            elif rlab == "underdetermined":
                v1, v2 = two_models_disagree(p.cons, p.n, p.R, p.query)
                if v1 == v2:
                    ok = False
                    break
            if not make_verify(p)(canonical_answer(p)):
                ok = False
                break
            counts[lab] += 1
    if verbose:
        print(f"  z3-verdict consistency: {counts} (each must reach 150)")
    ok &= all(v == 150 for v in counts.values())
    return ok


if __name__ == "__main__":
    print("clair.smt self-check (z3 oracle; determined vs underdetermined vs unsat)\n")
    rng = np.random.default_rng(3)
    for lab in ("determined", "underdetermined", "unsat"):
        p = gen_problem(rng, label=lab)
        print(f"[{lab}] query=x{p.query}  answer={canonical_answer(p)!r}")
        print("  " + render(p))
    print()
    ok = self_check()
    print("\n" + ("PASS — z3 verdicts internally consistent across all three labels."
                  if ok else "FAIL"))
