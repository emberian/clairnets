"""clair/rg_csp.py — reasoning-gym COVERAGE for the deduction organ.

The deduction organ narrows per-cell candidate sets over a FINITE-DOMAIN CSP (clair.csp.CSP:
factors = (scope, allowed-tuple relation) over a domain 0..d-1). This module answers the
breadth/generality question for the whole reasoning-gym catalog (106 procedural tasks):

  1. CLASSIFY every task: is it organ-applicable (a finite-domain assignment CSP we can narrow)?
     and if so, can it be expressed as a clair.csp.CSP so the EXACT dedP gives ground truth?
  2. ADAPT the directly-expressible ones (graph_color, n_queens, futoshiki, mini_sudoku,
     knights_knaves) -> clair.csp.CSP.
  3. CHARACTERIZE each adapted task at the TASK level with the exact harness (pure-python, no
     organ): solvability, the dedP narrowing CEILING, factor-lattice fixpoint, polymorphism kind.
     This is the *organ-independent* ceiling: the most a SOUND per-cell organ could ever recall.

Run on the GPU box with the organ via run_general:  python -m clair.rg_csp --coverage
"""
from __future__ import annotations

import itertools as it
from . import csp as C

# =====================================================================================
# (1) CLASSIFICATION of all 106 reasoning-gym tasks
#   category:
#     csp_fits        expressible finite-domain CSP that fits the trained organ budget (N<=8,D<=8,A<=3,M<=28)
#     csp_overbudget  expressible finite-domain CSP but exceeds N/D/A/M (needs a larger-budget organ)
#     csp_boolean     boolean SAT/logic CSP (truth assignment); expressible (arity = #vars in a clause)
#     not_csp         not a finite-domain assignment-narrowing problem (with a sub-reason)
# =====================================================================================
CLASSIFY = {
    # ---- directly-expressible finite-domain CSPs that FIT the trained budget (small configs) ----
    "graph_color":     ("csp_fits", "vertex coloring: pairwise != on edges; d=num_colors, n=num_vertices"),
    "n_queens":        ("csp_fits", "one var/row, domain=column; pairwise !=col & !=diag; pins=preplaced"),
    # ---- expressible CSPs that EXCEED the trained budget (cells/domain/arity) ----
    "mini_sudoku":     ("csp_overbudget", "16 cells>N_MAX=8; alldiff rows/cols/boxes + pins (d=4)"),
    "sudoku":          ("csp_overbudget", "81 cells, d=9; alldiff rows/cols/boxes — far over budget"),
    "futoshiki":       ("csp_overbudget", "board^2 cells; alldiff rows/cols + < pairs + pins (board>=3 over N)"),
    "kakurasu":        ("csp_overbudget", "binary grid; ROW/COL SUM factors are high-arity (>3)"),
    "survo":           ("csp_overbudget", "number grid; row/col SUM high-arity + alldiff of candidates"),
    "cryptarithm":     ("csp_overbudget", "letters->digit d=10>D_MAX; alldiff + per-column carry sums (arity>3)"),
    "zebra_puzzles":   ("csp_overbudget", "logic grid: alldiff per attribute + relational; structure not in metadata"),
    # ---- boolean SAT / logic CSPs ----
    "knights_knaves":  ("csp_boolean", "person truth ↔ eval(statement); boolean CSP, arity=#people in stmt"),
    "propositional_logic": ("csp_boolean", "models of premises = boolean CSP; but task asks ENTAILMENT not assignment"),
    "circuit_logic":   ("csp_boolean", "boolean circuit; deterministic EVALUATION given inputs (CSP-expressible, trivially solved)"),
    "self_reference":  ("csp_boolean", "self-referential truth count; boolean CSP over statements but bespoke"),
    # ---- NOT finite-domain assignment CSPs ----
    "ab":              ("not_csp", "string rewriting / A::B reduction (term rewriting, not narrowing)"),
    "acre":            ("not_csp", "Blicket-detector rule INDUCTION (abduction over hidden rule)"),
    "advanced_geometry": ("not_csp", "continuous geometry (real-valued coords)"),
    "aiw":             ("not_csp", "Alice-in-Wonderland word problem (arithmetic over relations)"),
    "arc_1d":          ("not_csp", "ARC grid-transform INDUCTION (program from examples)"),
    "arc_agi":         ("not_csp", "ARC grid-transform INDUCTION"),
    "rearc":           ("not_csp", "ARC re-generation INDUCTION"),
    "base_conversion": ("not_csp", "radix conversion (arithmetic)"),
    "basic_arithmetic":("not_csp", "arithmetic evaluation"),
    "bf":              ("not_csp", "Brainfuck program EXECUTION"),
    "binary_alternation": ("not_csp", "min-swaps to alternate (optimization over a string)"),
    "binary_matrix":   ("not_csp", "distance-to-nearest-0 (BFS over a grid)"),
    "bitwise_arithmetic": ("not_csp", "hex bitwise arithmetic evaluation"),
    "boxnet":          ("not_csp", "multi-agent box-moving PLANNING"),
    "caesar_cipher":   ("not_csp", "cipher decode (deterministic transform)"),
    "calendar_arithmetic": ("not_csp", "date arithmetic"),
    "chain_sum":       ("not_csp", "arithmetic chain"),
    "codeio":          ("not_csp", "predict program I/O (execution)"),
    "coin_flip":       ("not_csp", "probability"),
    "color_cube_rotation": ("not_csp", "3D cube-rotation state tracking (SIMULATION)"),
    "complex_arithmetic": ("not_csp", "complex-number arithmetic"),
    "composite":       ("not_csp", "meta-dataset wrapper"),
    "count_bits":      ("not_csp", "popcount"),
    "count_primes":    ("not_csp", "counting"),
    "countdown":       ("not_csp", "arithmetic-expression SYNTHESIS to a target"),
    "course_schedule": ("not_csp", "cycle detection / topological feasibility (graph property, not assignment)"),
    "decimal_arithmetic": ("not_csp", "decimal arithmetic"),
    "decimal_chain_sum": ("not_csp", "decimal chain"),
    "dice":            ("not_csp", "probability"),
    "emoji_mystery":   ("not_csp", "steganographic decode"),
    "family_relationships": ("not_csp", "kinship-graph traversal (path, not narrowing)"),
    "figlet_font":     ("not_csp", "ASCII-art font decode"),
    "fraction_simplification": ("not_csp", "gcd reduction"),
    "game_of_life":    ("not_csp", "cellular-automaton SIMULATION"),
    "game_of_life_halting": ("not_csp", "CA halting prediction (simulation)"),
    "gcd":             ("not_csp", "gcd"),
    "lcm":             ("not_csp", "lcm"),
    "group_anagrams":  ("not_csp", "grouping by sorted-key (hashing)"),
    "gsm_symbolic":    ("not_csp", "grade-school math word problem"),
    "intermediate_integration": ("not_csp", "symbolic calculus"),
    "simple_integration": ("not_csp", "symbolic calculus"),
    "isomorphic_strings": ("not_csp", "bijection EXISTENCE check (boolean; could be CSP but trivial)"),
    "jugs":            ("not_csp", "water-jug PLANNING (state search)"),
    "knight_swap":     ("not_csp", "piece-swap PLANNING"),
    "largest_island":  ("not_csp", "connected-component size (graph)"),
    "leg_counting":    ("not_csp", "arithmetic over a bag of animals"),
    "letter_counting": ("not_csp", "string counting"),
    "letter_jumble":   ("not_csp", "anagram unscramble"),
    "list_functions":  ("not_csp", "list-transform INDUCTION"),
    "mahjong_puzzle":  ("not_csp", "game-rule decision (simulation)"),
    "manipulate_matrix": ("not_csp", "apply matrix ops (execution)"),
    "maze":            ("not_csp", "shortest-path search"),
    "modulo_grid":     ("not_csp", "render grid from a modulo rule (deterministic)"),
    "needle_haystack": ("not_csp", "retrieval from distractors"),
    "number_filtering":("not_csp", "filter a list (predicate apply)"),
    "number_format":   ("not_csp", "pick by magnitude (parse/compare)"),
    "number_sequence": ("not_csp", "sequence-rule INDUCTION/extrapolation"),
    "number_sorting":  ("not_csp", "sorting"),
    "palindrome_generation": ("not_csp", "construct a palindrome"),
    "palindrome_partitioning": ("not_csp", "enumerate palindromic partitions (string DP)"),
    "path_star":       ("not_csp", "follow a path in a star graph (traversal)"),
    "polynomial_equations": ("not_csp", "root finding (continuous)"),
    "polynomial_multiplication": ("not_csp", "symbolic algebra"),
    "pool_matrix":     ("not_csp", "max/avg pooling (execution)"),
    "power_function":  ("not_csp", "exponentiation"),
    "prime_factorization": ("not_csp", "factorization"),
    "products":        ("not_csp", "multiplication"),
    "puzzle24":        ("not_csp", "make-24 expression SYNTHESIS"),
    "quantum_lock":    ("not_csp", "state-machine PLANNING to a target value"),
    "ransom_note":     ("not_csp", "multiset-cover boolean check"),
    "rectangle_count": ("not_csp", "count rectangles (combinatorial counting)"),
    "rotate_matrix":   ("not_csp", "matrix rotation (execution)"),
    "rotten_oranges":  ("not_csp", "multi-source BFS time (simulation)"),
    "rubiks_cube":     ("not_csp", "cube-solve PLANNING"),
    "rush_hour":       ("not_csp", "sliding-block PLANNING"),
    "sentence_reordering": ("not_csp", "reorder words"),
    "shortest_path":   ("not_csp", "grid pathfinding"),
    "simple_equations":("not_csp", "single-variable equation solve (arithmetic)"),
    "simple_geometry": ("not_csp", "interior-angle arithmetic"),
    "sokoban":         ("not_csp", "box-push PLANNING"),
    "spell_backward":  ("not_csp", "string reversal"),
    "spiral_matrix":   ("not_csp", "traverse a matrix in spiral order"),
    "string_insertion":("not_csp", "rule-based string insertion (execution)"),
    "string_manipulation": ("not_csp", "string-rewriting to fixpoint (execution)"),
    "string_splitting":("not_csp", "machine-decomposition simulation"),
    "string_synthesis":("not_csp", "block-rule simulation"),
    "syllogism":       ("not_csp", "categorical-logic VALIDITY (entailment, not assignment)"),
    "time_intervals":  ("not_csp", "time arithmetic"),
    "tower_of_hanoi":  ("not_csp", "disk-move PLANNING"),
    "tsumego":         ("not_csp", "go life-and-death (game search)"),
    "word_ladder":     ("not_csp", "word-graph path search"),
    "word_sequence_reversal": ("not_csp", "reverse a word list"),
    "word_sorting":    ("not_csp", "lexicographic sort"),
}


# =====================================================================================
# (2) ADAPTERS: reasoning-gym instance metadata -> clair.csp.CSP
# =====================================================================================
def graph_color_to_csp(md):
    p = md["puzzle"]
    verts = list(p["vertices"]); k = int(p["num_colors"])
    vidx = {v: i for i, v in enumerate(verts)}
    cons = [C._rel((vidx[u], vidx[v]), lambda t: t[0] != t[1], k) for (u, v) in p["edges"]]
    return C.CSP(len(verts), k, tuple(cons))


def n_queens_to_csp(md):
    """One variable per ROW, domain = column index 0..n-1. Pairwise: different column AND not on a
    shared diagonal. Pre-placed queens (md['puzzle']) become pins. Exact harness over n vars / d=n."""
    puzzle = md["puzzle"]; n = len(puzzle)
    cons = []
    for i in range(n):
        for j in range(i + 1, n):
            cons.append(C._rel((i, j), lambda t: t[0] != t[1], n))               # distinct columns
            dij = j - i
            cons.append(C._rel((i, j), (lambda dij: lambda t: abs(t[0] - t[1]) != dij)(dij), n))  # off-diagonal
    for r, row in enumerate(puzzle):
        for c, ch in enumerate(row):
            if ch == "Q":
                cons.append(C._rel((r,), (lambda c: lambda t: t[0] == c)(c), n))  # pin pre-placed queen
    return C.CSP(n, n, tuple(cons))


def _grid_cells(rows, cols):
    return lambda r, c: r * cols + c


def futoshiki_to_csp(md):
    """board x board latin square + inequality pairs + pins. Cell (r,c) -> var r*board+c, domain
    0..board-1 (puzzle stores 1..board with 0=blank; we shift to 0-based)."""
    board = int(md["board_size"]); puzzle = md["puzzle"]; cons = []
    cell = _grid_cells(board, board); n = board * board; d = board
    for r in range(board):                                   # alldiff rows + cols (pairwise !=)
        for a in range(board):
            for b in range(a + 1, board):
                cons.append(C._rel((cell(r, a), cell(r, b)), lambda t: t[0] != t[1], d))
                cons.append(C._rel((cell(a, r), cell(b, r)), lambda t: t[0] != t[1], d))
    for (r1, c1, r2, c2, op) in md["constraints"]:           # < / > between adjacent cells
        if op == "<":
            cons.append(C._rel((cell(r1, c1), cell(r2, c2)), lambda t: t[0] < t[1], d))
        else:
            cons.append(C._rel((cell(r1, c1), cell(r2, c2)), lambda t: t[0] > t[1], d))
    for r in range(board):                                   # pins (given values, shift to 0-based)
        for c in range(board):
            v = puzzle[r][c]
            if v != 0:
                cons.append(C._rel((cell(r, c),), (lambda v: lambda t: t[0] == v - 1)(v), d))
    return C.CSP(n, d, tuple(cons))


def mini_sudoku_to_csp(md):
    """4x4 mini-sudoku: 16 cells, d=4 (values shifted to 0..3). alldiff rows/cols/2x2 boxes + pins."""
    puzzle = md["puzzle"]; n = 16; d = 4; cell = _grid_cells(4, 4); cons = []
    groups = []
    for r in range(4):
        groups.append([cell(r, c) for c in range(4)])       # rows
        groups.append([cell(c, r) for c in range(4)])       # cols
    for br in (0, 2):
        for bc in (0, 2):
            groups.append([cell(br + i, bc + j) for i in range(2) for j in range(2)])  # boxes
    for g in groups:
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                cons.append(C._rel((g[a], g[b]), lambda t: t[0] != t[1], d))
    for r in range(4):
        for c in range(4):
            v = puzzle[r][c]
            if v != 0:
                cons.append(C._rel((cell(r, c),), (lambda v: lambda t: t[0] == v - 1)(v), d))
    return C.CSP(n, d, tuple(cons))


# ---- knights & knaves: compile nested statements to a boolean CSP ----
def _kk_vars(expr, acc):
    """collect person indices referenced by a statement expression."""
    if isinstance(expr, tuple):
        if expr[0] in ("telling-truth", "lying") and isinstance(expr[1], int):
            acc.add(expr[1])
        else:
            for e in expr[1:]:
                _kk_vars(e, acc)
    return acc


def _kk_eval(expr, assign):
    """evaluate a statement under a dict {person_idx: bool}."""
    op = expr[0]
    if op == "telling-truth":
        return assign[expr[1]]
    if op == "lying":
        return not assign[expr[1]]
    if op == "not":
        return not _kk_eval(expr[1], assign)
    if op == "and":
        return _kk_eval(expr[1], assign) and _kk_eval(expr[2], assign)
    if op == "or":
        return _kk_eval(expr[1], assign) or _kk_eval(expr[2], assign)
    if op == "->":
        return (not _kk_eval(expr[1], assign)) or _kk_eval(expr[2], assign)
    if op in ("<=>", "iff"):
        return _kk_eval(expr[1], assign) == _kk_eval(expr[2], assign)
    raise ValueError(op)


def knights_knaves_to_csp(md):
    """Person i is a knight (1) or knave (0). Constraint per person: truth(i) == eval(statement_i).
    Each constraint's scope = {i} ∪ {people referenced by statement_i}; allowed-table enumerated."""
    stmts = md["statements"]; n = len(stmts); d = 2; cons = []
    for i, st in enumerate(stmts):
        vs = sorted(_kk_vars(st, {i}))
        pos = {v: k for k, v in enumerate(vs)}
        allowed = set()
        for combo in it.product((0, 1), repeat=len(vs)):
            assign = {v: bool(combo[pos[v]]) for v in vs}
            if assign[i] == _kk_eval(st, assign):           # knight tells truth / knave lies
                allowed.add(combo)
        cons.append((tuple(vs), frozenset(allowed)))
    return C.CSP(n, d, tuple(cons))


ADAPTERS = {
    "graph_color": graph_color_to_csp,
    "n_queens": n_queens_to_csp,
    "futoshiki": futoshiki_to_csp,
    "mini_sudoku": mini_sudoku_to_csp,
    "knights_knaves": knights_knaves_to_csp,
}


# =====================================================================================
# (3) TASK-LEVEL exact characterization (organ-independent ceiling) — pure clair.csp
# =====================================================================================
def characterize_task(name, adapter, configs, n_inst=30, seed=0, enum_cap=10 ** 10):
    """For n_inst instances: solvability, dedP narrowing CEILING (fraction of full grid the exact
    per-cell transformer eliminates), factor-fixpoint match, % uniquely determined, polymorphism."""
    import numpy as np
    import reasoning_gym as rg
    from . import levels as L
    rng = np.random.default_rng(seed)
    n_ok = n_solv = uniq = ded_solved = fac_solved = 0
    ded_elim = full_cells = 0
    sl_n = maj_n = aff_n = 0
    ncell_sum = d_sum = ar_max = 0
    over_budget = 0
    for _ in range(n_inst):
        ds = rg.create_dataset(name, size=1, seed=int(rng.integers(0, 1 << 30)), **configs)
        md = ds[0]["metadata"]
        try:
            csp = adapter(md)
        except Exception:
            continue
        if csp.d ** csp.n > enum_cap:                       # exact enumeration too costly -> skip
            over_budget += 1
            continue
        n_ok += 1
        ncell_sum += csp.n; d_sum += csp.d
        ar_max = max(ar_max, max((len(sc) for sc, _ in csp.cons), default=0))
        sols = C.solutions(csp, limit=2)
        if not sols:
            continue
        n_solv += 1
        full = csp.full()
        ded, _ = C.to_fixpoint(C.exact_dedP, csp, full)
        ded_elim += sum(len(full[i]) - len(ded[i]) for i in range(csp.n))
        full_cells += sum(len(full[i]) for i in range(csp.n))
        uniq += int(all(len(c) == 1 for c in ded))
        ded_solved += int(C.status(ded) == "solved")
        try:
            fac = C.solve_factor(csp, 3)
            fac_solved += int(fac["outcome"] == "solved")
        except Exception:
            pass
        try:
            sl, maj, aff = L.polymorphism_kind(csp)
            sl_n += sl; maj_n += maj; aff_n += aff
        except Exception:
            pass
    return {
        "task": name, "configs": configs, "n_instances": n_inst, "n_adapted": n_ok,
        "n_overbudget_enum": over_budget,
        "n_solvable": n_solv,
        "avg_cells": ncell_sum / max(1, n_ok), "avg_domain": d_sum / max(1, n_ok), "max_arity": ar_max,
        "dedP_narrow_ceiling": ded_elim / max(1, full_cells),   # fraction of grid dedP eliminates
        "uniq_rate": uniq / max(1, n_solv),                     # fraction uniquely determined by dedP
        "dedP_solved": ded_solved / max(1, n_solv),
        "factor_solved": fac_solved / max(1, n_solv),
        "poly_semilattice": sl_n / max(1, n_solv), "poly_majority": maj_n / max(1, n_solv),
        "poly_affine": aff_n / max(1, n_solv),
    }


def rg_eval_csps(name, adapter, cfg, n, seed=777, budget=(8, 8, 28, 3), require_narrowable=True):
    """Sample up to `n` SOLVABLE, budget-fitting clair.csp.CSPs from a reasoning-gym task, for
    running a trained organ. budget=(N_MAX,D_MAX,M_MAX,A_MAX). require_narrowable drops instances
    the exact dedP can't narrow at all (e.g. under-constrained graph_color -> vacuous recall)."""
    import numpy as np
    import reasoning_gym as rg
    N, D, M, A = budget
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n * 8):
        if len(out) >= n:
            break
        ds = rg.create_dataset(name, size=1, seed=int(rng.integers(0, 1 << 30)), **cfg)
        try:
            csp = adapter(ds[0]["metadata"])
        except Exception:
            continue
        if not (csp.n <= N and csp.d <= D and len(csp.cons) <= M
                and all(len(sc) <= A for sc, _ in csp.cons)):
            continue
        if csp.d ** csp.n > 10 ** 10 or not C.solutions(csp, limit=1):
            continue
        if require_narrowable:
            ded, _ = C.to_fixpoint(C.exact_dedP, csp, csp.full())
            if sum(len(csp.full()[i]) - len(ded[i]) for i in range(csp.n)) == 0:
                continue
        out.append(csp)
    return out


def print_classification():
    cats = {}
    for t, (c, _) in CLASSIFY.items():
        cats.setdefault(c, []).append(t)
    print("=== reasoning-gym coverage classification (106 tasks) ===")
    for c in ("csp_fits", "csp_overbudget", "csp_boolean", "not_csp"):
        ts = sorted(cats.get(c, []))
        print(f"\n[{c}]  ({len(ts)})")
        for t in ts:
            print(f"  {t:24s} {CLASSIFY[t][1]}")


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--classify", action="store_true")
    ap.add_argument("--characterize", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    if args.classify or not args.characterize:
        print_classification()
    out = {}
    if args.characterize:
        TASK_CONFIGS = [
            ("graph_color", graph_color_to_csp, {"min_num_vertices": 8, "max_num_vertices": 8, "num_colors": 3}),
            ("graph_color", graph_color_to_csp, {"min_num_vertices": 6, "max_num_vertices": 6, "num_colors": 3}),
            ("n_queens", n_queens_to_csp, {"n": 5, "min_remove": 1, "max_remove": 4}),
            ("n_queens", n_queens_to_csp, {"n": 6, "min_remove": 1, "max_remove": 5}),
            ("mini_sudoku", mini_sudoku_to_csp, {"min_empty": 8, "max_empty": 10}),
            ("futoshiki", futoshiki_to_csp, {"min_board_size": 4, "max_board_size": 4}),
            ("knights_knaves", knights_knaves_to_csp, {"n_people": 2}),
            ("knights_knaves", knights_knaves_to_csp, {"n_people": 3}),
        ]
        rows = []
        for name, adp, cfg in TASK_CONFIGS:
            r = characterize_task(name, adp, cfg, n_inst=args.n)
            rows.append(r)
            print(f"\n{name} {cfg}")
            print(f"  adapted {r['n_adapted']}/{r['n_instances']} (enum-overbudget {r['n_overbudget_enum']})  "
                  f"avg_cells {r['avg_cells']:.1f} d {r['avg_domain']:.1f} maxarity {r['max_arity']}")
            print(f"  solvable {r['n_solvable']}  dedP-narrow-ceiling {r['dedP_narrow_ceiling']*100:.1f}%  "
                  f"uniq {r['uniq_rate']*100:.1f}%  dedP-solved {r['dedP_solved']*100:.1f}%  "
                  f"factor-solved {r['factor_solved']*100:.1f}%")
            print(f"  poly SL/Maj/Aff = {r['poly_semilattice']*100:.0f}/{r['poly_majority']*100:.0f}/"
                  f"{r['poly_affine']*100:.0f}%")
        out["characterize"] = rows
    if args.out and out:
        json.dump(out, open(args.out, "w"), indent=1)
        print("\nwrote", args.out)
