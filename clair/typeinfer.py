"""Exact TYPE-INFERENCE harness — the GROUND TRUTH for the type-inference organ.

Pure Python (no torch, no sympy). The CODE application of the deductor: the literature's hybrid is
"combine logical constraints (the type system — deterministic) with natural constraints (uncertain,
from text)". THIS file is the logical half — sound type-constraint propagation / Hindley-Milner
unification. A learned organ (clair.type_organ) later supplies the neural guidance (contractor /
propagation-order choice + abstain), measured against THIS.

Following the factory's confirmed pattern (csp.py / permgroup.py): a certified-sound primitive
(soundness free) + an exact ground truth, so completeness is the research variable.

THE LANGUAGE — a small typed lambda-calculus / HM fragment:
  lit int|bool · var · lam λx.e · app (f x) · let x=e1 in e2 (MONOMORPHIC) · add/sub (int)
  · cmp eq/lt (int→int→bool) · if c a b · pair (a,b) · fst · snd            (lists optional)

ABSTRACT STATE = per-(sub)expression-node candidate TYPE set dom[node] ⊆ T, where T is a FINITE type
universe (all type-terms up to a bounded depth). This is EXACTLY the powerset/unification lattice of
csp.py with cells = AST nodes and values = types. A type-variable narrows as its node's candidate set
shrinks; a node is SOLVED at |dom|=1 (a definite type), CONFLICT at |dom|=0 (ill-typed). We only MEET.

THE CERTIFIED SOUND OPERATOR = sound type-constraint propagation. We COMPILE a typed expression to a
finite CSP over T (cells=nodes, constraints = the typing rules as extensional relations), and reuse the
already-self-checked clair.csp machinery:
  certified_step  == csp.ac_step       arc consistency on the typing rules = unification narrowing.   SOUND
  strong_step     == csp pair/path consistency (catches multi-node type correlations AC forgets).      SOUND
  exact_dedP      == csp.exact_dedP    per-node type set over ALL in-universe well-typings.   strongest SOUND

EXACT GROUND TRUTH (two independent oracles, cross-checked in __main__):
  * exact_dedP (above) — the per-node target the organ must DOMINATE (never drop a real type).
  * algorithm_w — a standalone classic HM Algorithm-W (unification + occurs-check) inferring the
    PRINCIPAL type of the root. Grounding the principal type over the universe MUST equal the root's
    exact_dedP — i.e. the finite-CSP deduction agrees with real HM. This validates the compilation.

SCOPE (honest): this is the TYPE-CONSTRAINT sub-problem of coding, not program synthesis/execution —
which needs a separate program/execution primitive. The type universe is depth-bounded; expressions
are generated so their principal monotypes live in-universe. let is MONOMORPHIC (no generalization).

Run:  python -m clair.typeinfer        # self-check: certified propagation sound vs dedP, == Algorithm W
"""
from __future__ import annotations

import itertools as it
import random
from dataclasses import dataclass

from . import csp as C


# =========================================================================== TYPE TERMS + UNIVERSE
# A type term is a nested tuple:  ("int",) ("bool",) ("fun",a,b) ("pair",a,b) ("list",a).
INT = ("int",)
BOOL = ("bool",)


def fun(a, b): return ("fun", a, b)
def pair(a, b): return ("pair", a, b)
def lst(a): return ("list", a)


def type_depth(t):
    if t[0] in ("int", "bool"):
        return 0
    return 1 + max(type_depth(x) for x in t[1:])


def type_str(t):
    h = t[0]
    if h in ("int", "bool"):
        return h
    if h == "fun":
        return f"({type_str(t[1])}->{type_str(t[2])})"
    if h == "pair":
        return f"({type_str(t[1])}*{type_str(t[2])})"
    if h == "list":
        return f"[{type_str(t[1])}]"
    return str(t)


def build_universe(max_depth=1, ctors=("fun", "pair"), bases=("int", "bool")):
    """All type-terms up to `max_depth`, built from `bases` with the structural `ctors`. Closed and
    finite. Default (depth-1, fun+pair over {int,bool}) = 10 types; small enough for dense featurization
    and exact backtracking, rich enough for a real HM fragment (arith / if / pairs / 1st-order lambdas)."""
    cur = [(b,) for b in bases]
    universe = list(cur)
    for _ in range(max_depth):
        nxt = []
        for a, b in it.product(universe, repeat=2):
            if "fun" in ctors:
                nxt.append(fun(a, b))
            if "pair" in ctors:
                nxt.append(pair(a, b))
        if "list" in ctors:
            for a in universe:
                nxt.append(lst(a))
        for t in nxt:
            if t not in universe:
                universe.append(t)
    # de-dup preserving order
    seen, out = set(), []
    for t in universe:
        if t not in seen:
            seen.add(t); out.append(t)
    return out


class Universe:
    """A finite type universe with an index, plus the structural relations the typing rules need."""
    def __init__(self, max_depth=1, ctors=("fun", "pair"), bases=("int", "bool")):
        self.types = build_universe(max_depth, ctors, bases)
        self.idx = {t: i for i, t in enumerate(self.types)}
        self.T = len(self.types)
        self.bases = [(b,) for b in bases]
        self.ctors = ctors

    def has(self, t):
        return t in self.idx

    # ---- extensional relations over type INDICES (for compiling to csp.CSP) ----
    def eq_rel(self):
        return frozenset((i, i) for i in range(self.T))                          # binary diagonal

    def unary(self, t):
        return frozenset({(self.idx[t],)})

    def pair_rel(self):
        """(r,a,b) allowed iff r == pair(a,b)."""
        return frozenset((self.idx[pair(ta, tb)], self.idx[ta], self.idx[tb])
                         for ta in self.types for tb in self.types if pair(ta, tb) in self.idx)

    def fun_rel(self):
        """(r,a,b) allowed iff r == fun(a,b).  (used for lam: r=fun(param,body)) ..."""
        return frozenset((self.idx[fun(ta, tb)], self.idx[ta], self.idx[tb])
                         for ta in self.types for tb in self.types if fun(ta, tb) in self.idx)

    def app_rel(self):
        """(f,x,r) allowed iff f == fun(x,r)."""
        return frozenset((self.idx[fun(tx, tr)], self.idx[tx], self.idx[tr])
                         for tx in self.types for tr in self.types if fun(tx, tr) in self.idx)

    def fst_rel(self):
        """(e,r) allowed iff e == pair(r, _)."""
        return frozenset((self.idx[pair(t1, t2)], self.idx[t1])
                         for t1 in self.types for t2 in self.types if pair(t1, t2) in self.idx)

    def snd_rel(self):
        return frozenset((self.idx[pair(t1, t2)], self.idx[t2])
                         for t1 in self.types for t2 in self.types if pair(t1, t2) in self.idx)


# =========================================================================== EXPRESSION AST
# AST node tuples:
#   ("lit", value, "int"|"bool")  ("var", name)  ("lam", name, body)  ("app", f, x)
#   ("let", name, e1, e2)  ("add"|"sub", a, b)  ("cmp", a, b)  ("if", c, a, b)
#   ("pair", a, b)  ("fst", e)  ("snd", e)
def size(ast):
    if ast[0] in ("lit", "var"):
        return 1
    return 1 + sum(size(c) for c in ast[1:] if isinstance(c, tuple))


def depth(ast):
    kids = [c for c in ast[1:] if isinstance(c, tuple)]
    return 1 + (max(depth(c) for c in kids) if kids else 0)


# =========================================================================== COMPILE expr -> CSP
@dataclass
class Compiled:
    csp: C.CSP
    root: int                 # cell index of the whole expression's type
    n_nodes: int
    universe: Universe
    ast: object


def compile_expr(ast, U: Universe):
    """Compile a typed expression to a finite CSP over the type universe. Each subexpression node and
    each lambda parameter gets one cell; constraints are the typing rules as extensional relations.
    Returns Compiled.  env maps a bound name -> the cell holding its type."""
    cons = []
    counter = [0]
    eq = U.eq_rel()

    def new_cell():
        c = counter[0]; counter[0] += 1; return c

    def base_only(cell):
        # restrict a cell to base types (used for arith/cmp operands & lambda params default-free? no)
        pass

    def rec(node, env):
        h = node[0]
        if h == "lit":
            c = new_cell()
            cons.append(((c,), U.unary((node[2],))))
            return c
        if h == "var":
            # link to the binder's cell (equality); a var node still gets its own cell for the organ
            c = new_cell()
            bc = env[node[1]]
            cons.append(((c, bc), eq))
            return c
        if h in ("add", "sub"):
            ca = rec(node[1], env); cb = rec(node[2], env); c = new_cell()
            cons.append(((ca,), U.unary(INT)))
            cons.append(((cb,), U.unary(INT)))
            cons.append(((c,), U.unary(INT)))
            return c
        if h == "cmp":
            ca = rec(node[1], env); cb = rec(node[2], env); c = new_cell()
            cons.append(((ca,), U.unary(INT)))
            cons.append(((cb,), U.unary(INT)))
            cons.append(((c,), U.unary(BOOL)))
            return c
        if h == "if":
            cc = rec(node[1], env); ca = rec(node[2], env); cb = rec(node[3], env); c = new_cell()
            cons.append(((cc,), U.unary(BOOL)))
            cons.append(((ca, c), eq))       # then-branch == result
            cons.append(((cb, c), eq))       # else-branch == result
            return c
        if h == "pair":
            ca = rec(node[1], env); cb = rec(node[2], env); c = new_cell()
            cons.append(((c, ca, cb), U.pair_rel()))
            return c
        if h == "fst":
            ce = rec(node[1], env); c = new_cell()
            cons.append(((ce, c), U.fst_rel()))
            return c
        if h == "snd":
            ce = rec(node[1], env); c = new_cell()
            cons.append(((ce, c), U.snd_rel()))
            return c
        if h == "lam":
            pc = new_cell()                              # the parameter's type cell
            cbody = rec(node[2], {**env, node[1]: pc})
            c = new_cell()
            cons.append(((c, pc, cbody), U.fun_rel()))   # lam type == fun(param, body)
            return c
        if h == "app":
            cf = rec(node[1], env); cx = rec(node[2], env); c = new_cell()
            cons.append(((cf, cx, c), U.app_rel()))      # f == fun(x, result)
            return c
        if h == "let":                                   # MONOMORPHIC let
            ce1 = rec(node[2], env)
            cbody = rec(node[3], {**env, node[1]: ce1})  # uses of x share e1's cell
            c = new_cell()
            cons.append(((cbody, c), eq))
            return c
        raise ValueError(f"unknown node {h}")

    root = rec(ast, {})
    n = counter[0]
    csp = C.CSP(n, U.T, tuple(cons))
    return Compiled(csp=csp, root=root, n_nodes=n, universe=U, ast=ast)


# =========================================================================== certified operators (reuse csp)
def certified_step(csp, dom):
    """Sound type-constraint propagation = one arc-consistency pass over the typing rules. SOUND by
    construction (clair.csp.ac_step is self-checked: never drops a value used by a solution)."""
    return C.ac_step(csp, dom)


def to_fixpoint(csp, dom=None, step=certified_step):
    return C.to_fixpoint(step, csp, dom or csp.full())


def exact_dedP(csp, dom=None):
    """Per-node type set over ALL in-universe well-typings — the strongest sound narrowing (target)."""
    return C.exact_dedP(csp, dom or csp.full())


# =========================================================================== Algorithm W oracle (independent HM)
class _Fresh:
    def __init__(self): self.k = 0
    def __call__(self):
        self.k += 1; return ("tvar", self.k)


def _walk(t, s):
    while t[0] == "tvar" and t in s:
        t = s[t]
    return t


def _apply(t, s):
    t = _walk(t, s)
    if t[0] in ("int", "bool", "tvar"):
        return t
    return (t[0],) + tuple(_apply(x, s) for x in t[1:])


def _occurs(v, t, s):
    t = _walk(t, s)
    if t == v:
        return True
    if t[0] in ("int", "bool", "tvar"):
        return False
    return any(_occurs(v, x, s) for x in t[1:])


class _UnifyError(Exception):
    pass


def _unify(a, b, s):
    a = _walk(a, s); b = _walk(b, s)
    if a == b:
        return s
    if a[0] == "tvar":
        if _occurs(a, b, s):
            raise _UnifyError(f"occurs {a} in {b}")
        s2 = dict(s); s2[a] = b; return s2
    if b[0] == "tvar":
        return _unify(b, a, s)
    if a[0] != b[0] or len(a) != len(b):
        raise _UnifyError(f"{a} vs {b}")
    for x, y in zip(a[1:], b[1:]):
        s = _unify(x, y, s)
    return s


def algorithm_w(ast):
    """Classic monomorphic HM Algorithm W. Returns (principal_type, ok). ok=False if ill-typed.
    Independent of the CSP compilation — the cross-check oracle."""
    fresh = _Fresh()

    def infer(node, env, s):
        h = node[0]
        if h == "lit":
            return (node[2],), s
        if h == "var":
            return env[node[1]], s
        if h in ("add", "sub"):
            ta, s = infer(node[1], env, s); s = _unify(ta, INT, s)
            tb, s = infer(node[2], env, s); s = _unify(tb, INT, s)
            return INT, s
        if h == "cmp":
            ta, s = infer(node[1], env, s); s = _unify(ta, INT, s)
            tb, s = infer(node[2], env, s); s = _unify(tb, INT, s)
            return BOOL, s
        if h == "if":
            tc, s = infer(node[1], env, s); s = _unify(tc, BOOL, s)
            ta, s = infer(node[2], env, s); tb, s = infer(node[3], env, s)
            s = _unify(ta, tb, s)
            return _apply(ta, s), s
        if h == "pair":
            ta, s = infer(node[1], env, s); tb, s = infer(node[2], env, s)
            return ("pair", _apply(ta, s), _apply(tb, s)), s
        if h == "fst":
            te, s = infer(node[1], env, s); a, b = fresh(), fresh()
            s = _unify(te, ("pair", a, b), s); return _apply(a, s), s
        if h == "snd":
            te, s = infer(node[1], env, s); a, b = fresh(), fresh()
            s = _unify(te, ("pair", a, b), s); return _apply(b, s), s
        if h == "lam":
            a = fresh(); tb, s = infer(node[2], {**env, node[1]: a}, s)
            return ("fun", _apply(a, s), _apply(tb, s)), s
        if h == "app":
            tf, s = infer(node[1], env, s); tx, s = infer(node[2], env, s)
            r = fresh(); s = _unify(tf, ("fun", tx, r), s); return _apply(r, s), s
        if h == "let":                                   # monomorphic
            t1, s = infer(node[2], env, s)
            return infer(node[3], {**env, node[1]: _apply(t1, s)}, s)
        raise ValueError(h)

    try:
        t, s = infer(ast, {}, {})
        return _apply(t, s), True
    except (_UnifyError, KeyError):
        return None, False


def ground_principal(ptype, U: Universe):
    """All in-universe monotypes obtainable by substituting the free type-vars of `ptype` with universe
    types (consistently). The set that the root cell's exact_dedP must equal — the HM cross-check."""
    if ptype is None:
        return frozenset()
    tvars = sorted(_free_tvars(ptype))
    out = set()
    if not tvars:
        return frozenset({ptype}) if U.has(ptype) else frozenset()
    for combo in it.product(U.types, repeat=len(tvars)):
        sub = dict(zip(tvars, combo))
        g = _subst(ptype, sub)
        if U.has(g):
            out.add(g)
    return frozenset(out)


def _free_tvars(t, acc=None):
    acc = set() if acc is None else acc
    if t[0] == "tvar":
        acc.add(t)
    elif t[0] not in ("int", "bool"):
        for x in t[1:]:
            _free_tvars(x, acc)
    return acc


def _subst(t, sub):
    if t[0] == "tvar":
        return sub.get(t, t)
    if t[0] in ("int", "bool"):
        return t
    return (t[0],) + tuple(_subst(x, sub) for x in t[1:])


# =========================================================================== metrics / reporting
def node_types(comp: Compiled, dom):
    """Decode a per-cell domain into root-node candidate type SET (as type terms)."""
    return frozenset(comp.universe.types[i] for i in dom[comp.root])


def status(dom):
    return C.status(dom)


def false_elim(dom, exact):
    return sum(len(exact[i] - dom[i]) for i in range(len(dom)))


def solve(comp: Compiled, step=certified_step):
    """Run certified propagation to fixpoint from the full grid; report soundness + completeness vs the
    exact per-node dedP. false_elim==0 is the soundness guarantee; matches_exact is completeness."""
    csp = comp.csp
    dom, steps = C.to_fixpoint(step, csp, csp.full())
    ex = exact_dedP(csp)
    sols = C.solutions(csp, limit=1)
    fe = false_elim(dom, ex) if sols else 0
    return {
        "outcome": status(dom), "steps": steps,
        "well_typed": len(sols) > 0,
        "false_elim_vs_exact": fe,                           # 0 = sound
        "root_solved": len(dom[comp.root]) == 1,
        "root_alive": len(dom[comp.root]),
        "root_exact": len(ex[comp.root]),
        "matches_exact": dom == ex,                          # certified fixpoint == strongest sound narrowing?
        "root_type": (comp.universe.types[next(iter(dom[comp.root]))]
                      if len(dom[comp.root]) == 1 else None),
    }


# =========================================================================== generators (typed expressions)
def gen_typed(rng: random.Random, want, U: Universe, env, budget):
    """Goal-directed generation: return an AST that has type `want` (in-universe) under `env`, using only
    in-universe intermediate types. `budget` bounds depth so the recursion terminates. By construction
    every subexpression type is in-universe and the whole thing is well-typed."""
    choices = []

    # DIRECT constructors / leaves are ALWAYS available so `want` is always producible (termination):
    if want in (INT, BOOL):
        choices.append("lit")
    if want[0] == "pair":
        choices.append("mkpair")
    if want[0] == "fun":
        choices.append("lam")
    # a variable of the wanted type, if one is in scope
    invars = [x for x, t in env.items() if t == want]
    if invars:
        choices.append("var")

    if budget > 0:                                   # the richer, type-correlating forms
        if want == INT:
            choices += ["add", "sub"]
        if want == BOOL:
            choices += ["cmp"]
        choices += ["if"]
        # fst/snd produce `want` from a pair in-universe
        if any(U.has(pair(want, t2)) for t2 in U.types):
            choices.append("fst")
        if any(U.has(pair(t1, want)) for t1 in U.types):
            choices.append("snd")
        # application producing `want`: build a lambda inline applied to an argument
        choices.append("app")
        choices.append("let")

    k = rng.choice(choices)

    if k == "lit":
        if want == INT:
            return ("lit", rng.randint(0, 9), "int")
        return ("lit", rng.random() < 0.5, "bool")
    if k == "var":
        return ("var", rng.choice(invars))
    if k in ("add", "sub"):
        return (k, gen_typed(rng, INT, U, env, budget - 1), gen_typed(rng, INT, U, env, budget - 1))
    if k == "cmp":
        return ("cmp", gen_typed(rng, INT, U, env, budget - 1), gen_typed(rng, INT, U, env, budget - 1))
    if k == "if":
        return ("if", gen_typed(rng, BOOL, U, env, budget - 1),
                gen_typed(rng, want, U, env, budget - 1), gen_typed(rng, want, U, env, budget - 1))
    if k == "mkpair":
        return ("pair", gen_typed(rng, want[1], U, env, budget - 1),
                gen_typed(rng, want[2], U, env, budget - 1))
    if k == "fst":
        t2 = rng.choice([t for t in U.types if U.has(pair(want, t))])
        return ("fst", gen_typed(rng, pair(want, t2), U, env, budget - 1))
    if k == "snd":
        t1 = rng.choice([t for t in U.types if U.has(pair(t, want))])
        return ("snd", gen_typed(rng, pair(t1, want), U, env, budget - 1))
    if k == "lam":
        _, ta, tb = want
        x = f"v{rng.randint(0, 999)}"
        return ("lam", x, gen_typed(rng, tb, U, {**env, x: ta}, budget - 1))
    if k == "app":
        cand = [t for t in U.types if U.has(fun(t, want))]
        if not cand:
            return gen_typed(rng, want, U, env, 0)        # no in-universe arg type -> fall back to a leaf
        ta = rng.choice(cand)
        f = gen_typed(rng, fun(ta, want), U, env, budget - 1)
        x = gen_typed(rng, ta, U, env, budget - 1)
        return ("app", f, x)
    if k == "let":
        ta = rng.choice([t for t in U.types if type_depth(t) <= 1])
        x = f"w{rng.randint(0, 999)}"
        e1 = gen_typed(rng, ta, U, env, budget - 1)
        e2 = gen_typed(rng, want, U, {**env, x: ta}, budget - 1)
        return ("let", x, e1, e2)
    # fallback
    return ("lit", 0, "int") if want == INT else ("lit", True, "bool")


def rand_problem(rng, U: Universe, budget=3, want=None, max_nodes=14):
    """A random well-typed in-universe expression (and its compilation). Retries to bound node count."""
    for _ in range(60):
        w = want or rng.choice([t for t in U.types if type_depth(t) <= 1])
        ast = gen_typed(rng, w, U, {}, budget)
        if size(ast) <= max_nodes:
            comp = compile_expr(ast, U)
            if C.solutions(comp.csp, limit=1):                 # sanity: in-universe well-typed
                return comp
    return compile_expr(("lit", 0, "int"), U)


# =========================================================================== self-check
if __name__ == "__main__":
    print("exact type-inference harness self-check\n")
    U = Universe(max_depth=1, ctors=("fun", "pair"))
    print(f"type universe (depth<=1, fun+pair over int/bool): T={U.T}")
    print("  " + ", ".join(type_str(t) for t in U.types))

    def check(ast, label):
        comp = compile_expr(ast, U)
        r = solve(comp)
        pt, ok = algorithm_w(ast)
        ex = exact_dedP(comp.csp)
        ground = ground_principal(pt, U)
        root_ex = node_types(comp, ex)
        agree = (root_ex == ground)
        print(f"[{label}]  W-type={type_str(pt) if pt else 'ILL'}  "
              f"certified={r['outcome']}  root_alive={r['root_alive']} dedP={r['root_exact']}  "
              f"FE={r['false_elim_vs_exact']}  match_dedP={r['matches_exact']}  W==dedP:{agree}")
        assert r["false_elim_vs_exact"] == 0, "certified propagation MUST be sound"
        if ok:
            assert agree, f"CSP root dedP {sorted(map(type_str,root_ex))} != grounded W {sorted(map(type_str,ground))}"
        return comp, r

    # ---- basics: arithmetic pins int ----
    check(("add", ("lit", 1, "int"), ("lit", 2, "int")), "1+2")
    # if forces both branches & guard
    check(("if", ("cmp", ("lit", 1, "int"), ("lit", 2, "int")),
           ("lit", 3, "int"), ("lit", 4, "int")), "if 1<2 then 3 else 4")
    # identity lambda applied
    check(("app", ("lam", "x", ("add", ("var", "x"), ("lit", 1, "int"))), ("lit", 5, "int")), "(λx.x+1) 5")
    # pair + fst/snd
    check(("fst", ("pair", ("lit", 1, "int"),
                   ("cmp", ("lit", 0, "int"), ("lit", 1, "int")))), "fst (1, 0<1)")
    # polymorphic-ish identity (free type var) -> root is a SET of monotypes (the gap family)
    comp, r = check(("lam", "x", ("var", "x")), "λx.x  (a->a, polymorphic)")
    assert r["root_alive"] > 1, "λx.x must keep a SET of monotypes (a->a) — undetermined type var"
    # monomorphic let sharing: x used as int forces both uses
    check(("let", "x", ("lit", 3, "int"), ("add", ("var", "x"), ("var", "x"))), "let x=3 in x+x")
    # an ill-typed expression -> conflict, certified detects it soundly
    bad = compile_expr(("add", ("lit", 1, "int"),
                        ("cmp", ("lit", 0, "int"), ("lit", 1, "int"))), U)   # int + bool
    db, _ = C.to_fixpoint(certified_step, bad.csp, bad.csp.full())
    print(f"\n[1 + (0<1)]  ill-typed:  certified outcome={status(db)}  W-ok={algorithm_w(('add',('lit',1,'int'),('cmp',('lit',0,'int'),('lit',1,'int'))))[1]}")
    assert status(db) == "conflict", "type error must be detected (conflict) by sound propagation"

    # ---- corpus: certified soundness + W-agreement over many random expressions ----
    rng = random.Random(0)
    n_fe = n_match = n_total = n_wagree = n_open = 0
    for _ in range(400):
        comp = rand_problem(rng, U, budget=3)
        if not C.solutions(comp.csp, limit=1):
            continue
        n_total += 1
        dom, _ = C.to_fixpoint(certified_step, comp.csp, comp.csp.full())
        ex = exact_dedP(comp.csp)
        n_fe += false_elim(dom, ex)
        n_match += int(dom == ex)
        n_open += int(status(dom) == "open")
        pt, ok = algorithm_w(comp.ast)
        if ok:
            n_wagree += int(node_types(comp, ex) == ground_principal(pt, U))
    print(f"\ncorpus n={n_total}:  certified false_elim TOTAL = {n_fe} (MUST be 0)  "
          f"certified==dedP: {n_match}/{n_total} ({n_match/n_total*100:.1f}%)  "
          f"certified-abstain(open): {n_open}  W==dedP(root): {n_wagree}/{n_total}")
    assert n_fe == 0, "certified type propagation must be SOUND on the whole corpus"
    assert n_wagree == n_total, "finite-CSP deduction must agree with real HM Algorithm W on every well-typed expr"

    print("\nALL CHECKS PASS — certified type-constraint propagation is sound + agrees with Algorithm W.")
