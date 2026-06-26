"""clair/modular.py — the CERTIFIED MODULAR-ARITHMETIC / CONGRUENCE organ (the numeric primitive).

The codex zoo review (notes/codex_zoo_review.md) named the gap precisely: "intervals alone weak for
integers/parity/modular -> need REDUCED PRODUCTS (intervals x sign x parity/congruence x octagons)".
The per-cell lattice (clair.csp) ABSTAINS on affine structure — `xor_parity` is affine over GF(2)=Z_2
and per-cell AC cannot solve it. THIS organ is the congruence reduced-product piece: the same affine
wall, generalised to arbitrary modulus Z_m, attacked with a SOUND-BY-CONSTRUCTION certified operator.

TYPED PROTOCOL (so it chains as a reduced product with the per-cell / blade organs):
  * abstract STATE  : per-variable residue set R_i subset Z_m  (== a per-cell domain over values 0..m-1,
                      so the state is BITWISE the clair.csp per-cell lattice — composable for free).
  * concretisation  : gamma(R) = { x in Z_m^n : x_i in R_i }.
  * CERTIFIED operators (each SOUND by construction — modular arithmetic is exact, no roundoff):
      - narrow_eq     : generalised-arc residue narrowing for ONE linear modular eq (local, incomplete)
      - solve_mod     : COMPLETE solver for a SYSTEM of linear modular eqs via integer Smith Normal Form
                        over Z (handles COMPOSITE moduli / rings, where Gaussian-over-a-field breaks)
      - crt_combine   : Chinese Remainder Theorem — fuse x≡r1 (mod m1), x≡r2 (mod m2)
  * verifier / oracle: brute enumeration over the finite residue space Z_m^n (independent ground truth).
  * abstain         : when the system is UNDER-DETERMINED (free variables) the organ narrows to the exact
                      reachable cosets and reports `multi` — it does NOT guess a singleton.

solve_mod returns, for a consistent system, the EXACT per-variable reachable residue set (the modular
dedP) and the solution count — cheaply (poly-time, no enumeration) — by parametrising the affine
solution lattice from the SNF. This is both the certified complete operator AND a cheap exact ground
truth that scales past brute enumeration. The SMOKE cross-checks it exhaustively against brute.

  python -m clair.modular            # SMOKE: SNF validation + certified-op vs brute cross-check + CRT

Pure Python / no torch — this is the exact, checkable authority. The NEURAL guidance that learns which
certified contraction to propose lives in clair.run_modular (it imports the generators + encoding here).
"""
from __future__ import annotations

import itertools as it
from dataclasses import dataclass
from math import gcd

from . import csp as C


# ============================================================ number theory (all exact)
def egcd(a: int, b: int):
    """Extended Euclid: returns (g, x, y) with a*x + b*y = g = gcd(a,b)."""
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_r, old_s, old_t


def inv_mod(a: int, m: int) -> int:
    """Modular inverse of a mod m (requires gcd(a,m)==1). m==1 -> 0 (trivial ring)."""
    if m == 1:
        return 0
    g, x, _ = egcd(a % m, m)
    if g != 1:
        raise ValueError(f"{a} not invertible mod {m}")
    return x % m


def crt_combine(r1: int, m1: int, r2: int, m2: int):
    """Chinese Remainder Theorem (general moduli, not necessarily coprime). Combine
      x ≡ r1 (mod m1)  AND  x ≡ r2 (mod m2)
    into a single congruence x ≡ r (mod lcm(m1,m2)), or None if INCONSISTENT (no such x).
    Certified: the returned (r, M) is sound (every x satisfying it satisfies both) and complete
    (every common solution is captured) — proven by construction via egcd."""
    g, p, _ = egcd(m1, m2)
    if (r2 - r1) % g != 0:
        return None                                   # inconsistent: r1, r2 disagree mod gcd
    lcm = m1 // g * m2
    r = (r1 + (m1 * ((r2 - r1) // g * p % (m2 // g)))) % lcm
    return r % lcm, lcm


# ============================================================ integer Smith Normal Form
def _ident(k):
    return [[1 if i == j else 0 for j in range(k)] for i in range(k)]


def _snf(A):
    """Integer Smith Normal Form. Returns (D, U, V) with U,V UNIMODULAR (det ±1) over Z and
    U @ A0 @ V == diag(D) (D padded with zeros). D = the invariant factors (each divides the next).
    Row ops accumulate into U, column ops into V, maintaining the invariant  A = U @ A0 @ V  at all
    times (A starts as A0, U=V=I, ends as the diagonal). Classic pivot-and-reduce; fine for the small
    matrices here (n<=~12 vars, <=~48 eqs)."""
    A = [row[:] for row in A]
    r = len(A)
    n = len(A[0]) if r else 0
    U = _ident(r)
    V = _ident(n)

    def swap_rows(i, j):
        A[i], A[j] = A[j], A[i]; U[i], U[j] = U[j], U[i]

    def swap_cols(i, j):
        for row in A:
            row[i], row[j] = row[j], row[i]
        for row in V:
            row[i], row[j] = row[j], row[i]

    def addrow(i, j, k):           # row_i += k * row_j
        for c in range(n):
            A[i][c] += k * A[j][c]
        for c in range(r):
            U[i][c] += k * U[j][c]

    def addcol(i, j, k):           # col_i += k * col_j
        for row in A:
            row[i] += k * row[j]
        for row in V:
            row[i] += k * row[j]

    t = 0
    while t < min(r, n):
        # pick a pivot: any nonzero entry in the active submatrix (rows/cols >= t)
        piv = None
        for i in range(t, r):
            for j in range(t, n):
                if A[i][j] != 0:
                    if piv is None or abs(A[i][j]) < abs(A[piv[0]][piv[1]]):
                        piv = (i, j)
        if piv is None:
            break                                      # rest is all zero
        pi, pj = piv
        if pi != t:
            swap_rows(pi, t)
        if pj != t:
            swap_cols(pj, t)
        # reduce row t and column t until A[t][t] is the only nonzero in its row & col
        while True:
            changed = False
            for i in range(t + 1, r):                  # clear column t below pivot
                if A[i][t]:
                    q = A[i][t] // A[t][t]
                    addrow(i, t, -q)
                    if A[i][t]:                        # remainder nonzero -> bring it up, repeat
                        swap_rows(i, t); changed = True
            for j in range(t + 1, n):                  # clear row t right of pivot
                if A[t][j]:
                    q = A[t][j] // A[t][t]
                    addcol(j, t, -q)
                    if A[t][j]:
                        swap_cols(j, t); changed = True
            if not changed:
                break
        # ensure A[t][t] divides every remaining entry (the | chain); else fold a violating row in
        d = A[t][t]
        bad = None
        for i in range(t + 1, r):
            for j in range(t + 1, n):
                if A[i][j] % d != 0:
                    bad = i; break
            if bad is not None:
                break
        if bad is not None:
            addrow(t, bad, 1)                          # mix the offending row in, re-reduce pivot t
            continue
        if A[t][t] < 0:                                # normalise pivot sign (keep U,V unimodular)
            for c in range(n):
                A[t][c] = -A[t][c]
            for c in range(r):
                U[t][c] = -U[t][c]
        t += 1
    D = [A[i][i] for i in range(min(r, n))]
    return D, U, V


def _matvec_mod(M, v, m):
    return [sum(M[i][k] * v[k] for k in range(len(v))) % m for i in range(len(M))]


# ============================================================ the system type + CERTIFIED solver
@dataclass(frozen=True)
class LinSystem:
    """A system of linear modular equations over Z_m: each eq is (coeffs, rhs) with
    sum_i coeffs[i]*x_i ≡ rhs (mod m). `coeffs` is a full-length n-tuple (sparse = mostly 0)."""
    n: int
    m: int
    eqs: tuple          # tuple of (coeffs: tuple[int]*n, rhs: int)

    def support(self, coeffs):
        return tuple(i for i in range(self.n) if coeffs[i] % self.m != 0)


def solve_mod(sys: LinSystem):
    """CERTIFIED COMPLETE solver for a linear modular system. Returns a dict:
        status     : 'unsat' | 'unique' | 'multi'
        residues   : tuple of frozenset[int] — the EXACT per-variable reachable residues (the modular
                     dedP: residue r is kept at x_i iff SOME solution sets x_i = r). [] each if unsat.
        n_solutions: number of solutions in Z_m^n.
    Method: SNF U A V = D. Sub z = V^{-1} x => D z ≡ U b (mod m). Each z-coordinate is independently a
    coset (or free) -> the solution set is an affine sublattice; x = V z; project per variable. SOUND +
    COMPLETE by construction (every step is an exact equivalence over Z_m), and CHEAP (no enumeration).
    """
    n, m = sys.n, sys.m
    if not sys.eqs:                                            # no constraints -> everything alive
        full = frozenset(range(m))
        return {"status": "multi" if m > 1 else "unique",
                "residues": tuple(full for _ in range(n)), "n_solutions": m ** n}
    A = [list(c) for c, _ in sys.eqs]
    b = [rhs % m for _, rhs in sys.eqs]
    r = len(A)
    D, U, V = _snf(A)
    Ub = _matvec_mod(U, b, m)
    rank_diag = len(D)
    # per-z-coordinate solution: (z0, step, count). count==1 => fixed; count==m & step==1 => free.
    coord = []
    for i in range(n):
        if i < rank_diag:
            d = D[i] % m
            c = Ub[i] % m
            if d == 0:
                if c % m != 0:
                    return {"status": "unsat", "residues": tuple(frozenset() for _ in range(n)),
                            "n_solutions": 0}
                coord.append((0, 1, m))                        # free
            else:
                g = gcd(d, m)
                if c % g != 0:
                    return {"status": "unsat", "residues": tuple(frozenset() for _ in range(n)),
                            "n_solutions": 0}
                mg = m // g
                z0 = (c // g) * inv_mod((d // g) % mg, mg) % mg if mg > 1 else 0
                coord.append((z0, mg, g))                      # coset of size g, step m/g
        else:
            coord.append((0, 1, m))                            # extra column -> free
    # extra rows (more eqs than vars): require U b ≡ 0 there
    for i in range(n, r):
        if Ub[i] % m != 0:
            return {"status": "unsat", "residues": tuple(frozenset() for _ in range(n)),
                    "n_solutions": 0}
    n_sol = 1
    for _, _, cnt in coord:
        n_sol *= cnt
    # project x = V z per variable: x_j = sum_i V[j][i] z_i; reachable_j = x0_j + <generators>·Z_m
    residues = []
    for j in range(n):
        x0 = sum(V[j][i] * coord[i][0] for i in range(n)) % m
        g_j = m
        for i in range(n):
            z0, step, cnt = coord[i]
            if cnt > 1:                                        # this coordinate moves
                g_j = gcd(g_j, (V[j][i] * step) % m)
        g_j = g_j if g_j != 0 else m
        step_j = gcd(g_j, m) if g_j else m
        size = m // step_j if step_j else 1
        residues.append(frozenset((x0 + t * step_j) % m for t in range(size)))
    status = "unique" if n_sol == 1 else "multi"
    return {"status": status, "residues": tuple(residues), "n_solutions": n_sol}


# ============================================================ local certified contractor (sound, incomplete)
def narrow_eq(sys: LinSystem, eq, dom):
    """Generalised-arc residue narrowing for ONE linear modular equation, given the current per-variable
    residue sets `dom`. Keep residue r at x_i iff there EXISTS an assignment of the eq's other support
    vars (within their residue sets) making the equation hold. SOUND (never drops a value some solution
    of THIS eq uses) but LOCAL — it cannot combine equations, so a coupled affine-mod system makes it
    ABSTAIN, exactly like clair.csp.ac_step on xor_parity. This is the modular analogue of AC."""
    coeffs, rhs = eq
    m = sys.m
    sup = sys.support(coeffs)
    new = list(dom)
    if not sup:
        return tuple(new)
    for p, i in enumerate(sup):
        others = [q for q in sup if q != i]
        keep = set()
        for r in dom[i]:
            for combo in it.product(*[sorted(dom[q]) for q in others]):
                tot = coeffs[i] * r + sum(coeffs[others[k]] * combo[k] for k in range(len(others)))
                if tot % m == rhs % m:
                    keep.add(r); break
        new[i] = frozenset(keep)
    return tuple(new)


def ac_fixpoint(sys: LinSystem, dom=None):
    """Iterate narrow_eq over all equations to a fixed point (the local certified residue-AC organ)."""
    dom = dom or tuple(frozenset(range(sys.m)) for _ in range(sys.n))
    for _ in range(sys.n * sys.m + 2):
        nxt = dom
        for eq in sys.eqs:
            nxt = narrow_eq(sys, eq, nxt)
        if nxt == dom:
            return dom
        dom = nxt
    return dom


# ============================================================ independent brute oracle
def brute_solve(sys: LinSystem, dom=None):
    """Exact ground truth by enumerating Z_m^n (or the sub-box `dom`). Returns (residues, n_solutions).
    Only used to CROSS-CHECK solve_mod / narrow_eq on small instances — the proof of soundness."""
    dom = dom or tuple(frozenset(range(sys.m)) for _ in range(sys.n))
    surv = [set() for _ in range(sys.n)]
    cnt = 0
    for x in it.product(*[sorted(dom[i]) for i in range(sys.n)]):
        if all(sum(c[i] * x[i] for i in range(sys.n)) % sys.m == rhs % sys.m for c, rhs in sys.eqs):
            cnt += 1
            for i in range(sys.n):
                surv[i].add(x[i])
    return tuple(frozenset(s) for s in surv), cnt


# ============================================================ LinSystem <-> clair.csp.CSP encoding
def to_csp(sys: LinSystem) -> C.CSP:
    """Encode the modular system as an extensional finite CSP over domain Z_m (values 0..m-1): each
    equation becomes an (scope, allowed-tuple) relation, so the EXISTING factor-graph organ + the exact
    clair.csp.exact_dedP harness apply unchanged. Arity == |support| (kept <= 3 by the generators)."""
    m = sys.m
    cons = []
    for coeffs, rhs in sys.eqs:
        sup = sys.support(coeffs)
        if not sup:
            continue
        cf = tuple(coeffs[i] for i in sup)
        rr = rhs % m
        cons.append(C._rel(sup, (lambda cf, rr: lambda t: sum(cf[k] * t[k] for k in range(len(cf))) % m == rr)(cf, rr), m))
    return C.CSP(sys.n, m, tuple(cons))


# ============================================================ generators (witness-first => solvable)
def gen_lineq(rng, m, n, n_eq=None, max_arity=3, n_pin=None):
    """A sparse linear modular system over Z_m: sample a witness x*, emit `n_eq` random equations
    (each over <=max_arity vars, coeffs in 1..m-1) with rhs computed from x* (so x* is a solution =>
    always solvable), plus `n_pin` unary pins. The COUPLED affine-mod rung the local lattice abstains
    on. Returns LinSystem."""
    x = [int(rng.integers(0, m)) for _ in range(n)]
    n_eq = n_eq if n_eq is not None else max(2, int(rng.integers(n, 2 * n)))
    n_pin = n_pin if n_pin is not None else int(rng.integers(0, max(1, n // 2) + 1))
    eqs = []
    for _ in range(n_eq):
        ar = int(rng.integers(2, max_arity + 1)) if n >= 2 else 1
        ar = min(ar, n)
        vars_ = list(rng.choice(n, size=ar, replace=False))
        coeffs = [0] * n
        for v in vars_:
            coeffs[v] = int(rng.integers(1, m))
        rhs = sum(coeffs[v] * x[v] for v in vars_) % m
        eqs.append((tuple(coeffs), rhs))
    pins = list(rng.choice(n, size=min(n_pin, n), replace=False)) if n_pin else []
    for v in pins:
        coeffs = [0] * n; coeffs[v] = 1
        eqs.append((tuple(coeffs), x[v] % m))
    return LinSystem(n, m, tuple(eqs))


def gen_cyclic(rng, m, n):
    """The MODULAR PARITY WALL (generalises clair.csp.xor_parity to Z_m), kept arity<=3: a connected
    chain of binary EQUALITIES x_i = x_{i+1} (forces all variables equal) + 1-2 ternary equations
    a·x_i + b·x_j + c·x_k ≡ rhs with rhs from a witness. NO pins. Each constraint, taken locally with
    full domains, supports EVERY value — so local residue-AC narrows NOTHING (recall 0) — yet the system
    is globally (near-)determined: all-equal w with (a+b+c)(w−v)≡0. This is the rung the per-cell lattice
    ABSTAINS on; only the certified solve_mod (or a global organ) cracks it. Witness-first => solvable."""
    v = int(rng.integers(0, m))
    eqs = []
    order = list(range(n)); rng.shuffle(order)
    for a, b in zip(order, order[1:]):                        # spanning chain of equalities (connected)
        coeffs = [0] * n; coeffs[a] = 1; coeffs[b] = (m - 1)  # x_a - x_b ≡ 0
        eqs.append((tuple(coeffs), 0))
    for _ in range(int(rng.integers(1, 3))):                  # 1-2 ternary couplers
        k = min(3, n)
        vs = list(rng.choice(n, size=k, replace=False))
        coeffs = [0] * n
        for vv in vs:
            coeffs[vv] = int(rng.integers(1, m))
        rhs = sum(coeffs[vv] * v for vv in vs) % m
        eqs.append((tuple(coeffs), rhs))
    return LinSystem(n, m, tuple(eqs))


def gen_modsum(rng, m, n):
    """Gentler affine rung: ternary modular sums x_i + x_j ≡ x_k (mod m) (== (a+b)%m==c) chained over
    n vars + a couple pins. Witness-first."""
    x = [int(rng.integers(0, m)) for _ in range(n)]
    eqs = []
    for _ in range(max(2, int(rng.integers(n, 2 * n)))):
        i, j, k = (int(t) for t in rng.choice(n, size=3, replace=False)) if n >= 3 else (0, 0, 0)
        coeffs = [0] * n
        coeffs[i] += 1; coeffs[j] += 1; coeffs[k] -= 1
        rhs = (x[i] + x[j] - x[k]) % m
        eqs.append((tuple(c % m for c in coeffs), rhs))
    for v in rng.choice(n, size=max(1, n // 3), replace=False):
        coeffs = [0] * n; coeffs[int(v)] = 1
        eqs.append((tuple(coeffs), x[int(v)] % m))
    return LinSystem(n, m, tuple(eqs))


# ============================================================ SMOKE
def _is_unimodular(M):
    # integer determinant via fraction-free (Bareiss) — small matrices
    import copy
    A = [row[:] for row in M]
    k = len(A)
    if k == 0:
        return True
    prev = 1
    sign = 1
    for i in range(k - 1):
        if A[i][i] == 0:
            sw = next((r for r in range(i + 1, k) if A[r][i] != 0), None)
            if sw is None:
                return False
            A[i], A[sw] = A[sw], A[i]; sign = -sign
        for r in range(i + 1, k):
            for c in range(i + 1, k):
                A[r][c] = (A[r][c] * A[i][i] - A[r][i] * A[i][c]) // prev
        prev = A[i][i]
    det = sign * A[k - 1][k - 1]
    return det in (1, -1)


def _smoke():
    import numpy as np
    rng = np.random.default_rng(0)
    print("=== clair.modular SMOKE: certified congruence organ ===\n")

    # 1. SNF validity: U A V == diag(D), U & V unimodular --------------------------------------------
    print("[1] integer Smith Normal Form  (U A V = D, U,V unimodular)")
    bad = 0
    for _ in range(400):
        r = int(rng.integers(1, 6)); c = int(rng.integers(1, 6))
        A = [[int(rng.integers(-4, 5)) for _ in range(c)] for _ in range(r)]
        D, U, V = _snf(A)
        AV = [[sum(A[a][b] * V[b][j] for b in range(c)) for j in range(c)] for a in range(r)]
        UAV = [[sum(U[i][a] * AV[a][j] for a in range(r)) for j in range(c)] for i in range(r)]
        diag_ok = all(UAV[i][j] == (D[i] if i == j and i < len(D) else 0)
                      for i in range(r) for j in range(c))
        if not (diag_ok and _is_unimodular(U) and _is_unimodular(V)):
            bad += 1
    print(f"    400 random matrices: {400 - bad}/400 valid SNF factorisations "
          f"({'OK' if bad == 0 else f'{bad} BAD'})")
    assert bad == 0

    # 2. CERTIFIED solve_mod vs brute oracle: per-var residues + solution count, all moduli ----------
    print("\n[2] certified solve_mod  vs  brute enumeration  (soundness + completeness of the operator)")
    moduli = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
    mism = 0; trials = 0; unsat_seen = 0; multi_seen = 0
    for _ in range(3000):
        m = int(rng.choice(moduli)); n = int(rng.integers(1, 5))
        # mix solvable (witness-first) and possibly-unsat (random rhs) systems
        if rng.random() < 0.7:
            sysm = gen_lineq(rng, m, n, n_eq=int(rng.integers(1, 2 * n + 1)), n_pin=int(rng.integers(0, n)))
        else:
            n_eq = int(rng.integers(1, 2 * n + 1))
            eqs = []
            for _ in range(n_eq):
                coeffs = [int(rng.integers(0, m)) for _ in range(n)]
                eqs.append((tuple(coeffs), int(rng.integers(0, m))))
            sysm = LinSystem(n, m, tuple(eqs))
        cert = solve_mod(sysm)
        bres, bcnt = brute_solve(sysm)
        trials += 1
        exp_status = "unsat" if bcnt == 0 else ("unique" if bcnt == 1 else "multi")
        if cert["status"] != exp_status or cert["n_solutions"] != bcnt or cert["residues"] != bres:
            mism += 1
            if mism <= 3:
                print(f"    MISMATCH m={m} n={n}: cert={cert} brute=({bres},{bcnt})")
        unsat_seen += (exp_status == "unsat"); multi_seen += (exp_status == "multi")
    print(f"    {trials} random systems (moduli {moduli}, incl {unsat_seen} unsat / {multi_seen} "
          f"under-determined): {trials - mism}/{trials} EXACT match  "
          f"({'OK — operator is sound AND complete' if mism == 0 else f'{mism} MISMATCH'})")
    assert mism == 0

    # 3. CRT certified combine ----------------------------------------------------------------------
    print("\n[3] Chinese Remainder Theorem  crt_combine  (the cross-modulus fuse op)")
    cbad = 0; ctot = 0
    for _ in range(2000):
        m1 = int(rng.integers(2, 13)); m2 = int(rng.integers(2, 13))
        r1 = int(rng.integers(0, m1)); r2 = int(rng.integers(0, m2))
        ctot += 1
        res = crt_combine(r1, m1, r2, m2)
        L = m1 // gcd(m1, m2) * m2
        truth = [x for x in range(L) if x % m1 == r1 and x % m2 == r2]
        if res is None:
            if truth:
                cbad += 1
        else:
            r, M = res
            if M != L or sorted({x for x in range(L) if x % M == r}) != truth:
                cbad += 1
    ex = crt_combine(2, 3, 3, 5)
    print(f"    x≡2(mod 3), x≡3(mod 5) -> x≡{ex[0]}(mod {ex[1]})  (8: 8%3=2, 8%5=3 ✓)")
    print(f"    {ctot} random pairs (coprime + NON-coprime): {ctot - cbad}/{ctot} match brute "
          f"({'OK' if cbad == 0 else f'{cbad} BAD'})")
    assert ex == (8, 15) and cbad == 0

    # 4. the AFFINE-MOD WALL: local residue-AC ABSTAINS where the certified solver SOLVES -------------
    print("\n[4] the affine-mod wall: local certified residue-AC vs the complete certified solver")
    print(f"    {'m':>3} {'n':>3} {'#eq':>4} {'AC-alive(sum)':>13} {'solve-alive(sum)':>16} "
          f"{'AC-FE':>6} {'solver':>8}")
    print("    (cyclic = the modular parity wall: equalities + ternary coupler, NO pins)")
    rng2 = np.random.default_rng(7)
    ac_abstains = 0; solver_solves = 0; checked = 0
    for m in [3, 5, 6, 7]:
        for n in [3, 4, 5]:
            sysm = gen_cyclic(rng2, m, n)
            ac = ac_fixpoint(sysm)
            cert = solve_mod(sysm)
            bres, bcnt = brute_solve(sysm)
            ac_alive = sum(len(s) for s in ac)
            cert_alive = sum(len(s) for s in cert["residues"])
            ac_fe = sum(len(bres[i] - ac[i]) for i in range(n))         # local AC must stay SOUND
            checked += 1
            ac_abstains += int(ac_alive > cert_alive)
            solver_solves += int(cert["residues"] == bres)
            print(f"    {m:>3} {n:>3} {len(sysm.eqs):>4} {ac_alive:>13} {cert_alive:>16} "
                  f"{ac_fe:>6} {'exact' if cert['residues']==bres else 'WRONG':>8}")
            assert ac_fe == 0, "local residue-AC must be SOUND (never drop a real solution value)"
            assert cert["residues"] == bres, "certified solver must match brute exactly"
    print(f"    => local AC leaves strictly more alive (ABSTAINS) on {ac_abstains}/{checked} cyclic "
          f"systems; the certified solver matches brute on {solver_solves}/{checked} (SOUND + COMPLETE).")
    assert ac_abstains >= checked - 1, "cyclic systems should make local AC abstain (the wall)"

    # 5. encoding round-trip: to_csp + clair.csp.exact_dedP == solve_mod residues -------------------
    print("\n[5] CSP encoding round-trip  (to_csp -> clair.csp.exact_dedP == solve_mod, so it chains)")
    rt_bad = 0
    for _ in range(300):
        m = int(rng.choice([2, 3, 4, 5, 6, 7])); n = int(rng.integers(2, 5))
        sysm = gen_lineq(rng, m, n, n_eq=int(rng.integers(2, 2 * n)), n_pin=int(rng.integers(0, n)))
        csp = to_csp(sysm)
        ded, _ = C.to_fixpoint(C.exact_dedP, csp, csp.full())
        cert = solve_mod(sysm)
        if tuple(ded) != cert["residues"]:
            rt_bad += 1
    print(f"    300 systems: {300 - rt_bad}/300 csp.exact_dedP == solve_mod residues "
          f"({'OK — same state type, composable' if rt_bad == 0 else f'{rt_bad} BAD'})")
    assert rt_bad == 0

    print("\nALL CHECKS PASS — certified congruence organ: SNF solver sound+complete vs brute across "
          "moduli (incl composite rings), CRT exact, local AC sound-but-abstains, CSP-encoded for the\n"
          "factor-graph organ to chain. The numeric reduced-product piece the per-cell lattice lacked.")


if __name__ == "__main__":
    _smoke()
