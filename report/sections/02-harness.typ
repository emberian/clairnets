#import "../helpers.typ": *

= The harness: an exact world to be sound *about*

Before any neural claim, we need ground truth — a generator of constraint problems where the *exactly
correct* deduction is computable, so "the organ eliminated a true candidate" is a measurable event, not a
matter of opinion. The harness (`clair/csp.py`, `clair/levels.py`) is that world, and it is deliberately
unglamorous: its job is to make our soundness claims falsifiable.

== Exact deduction, by construction

For a constraint satisfaction problem $p$, the harness computes the *exact deduction closure* $#raw("ded")_p$:
the per-variable set of values that appear in *at least one* full solution. This is the tightest sound
narrowing possible — anything tighter would kill a value that some solution needs. Two facts about it are
load-bearing and are grounded in the LDT line's Lean formalization (#arxiv("2605.08605")):

- *Soundness is free if you check; completeness is the hard part.* A narrowing that only removes values no
  solution uses is sound by definition. The interesting, *learnable* quantity is #term[completeness]: how
  close to $#raw("ded")_p$ the organ gets — i.e. whether it reaches a checkable answer or honestly abstains.
  Completeness, not soundness, is the research variable.
- *Per-cell arc-consistency equals exact deduction only at the right level.* Cheap local propagation
  (per-cell AC) recovers $#raw("ded")_p$ on binary-structured problems but *not* on problems with
  higher-arity structure. The harness exposes this as a lattice of #term[abstraction levels]: level-0 is
  per-cell/unary, level-1 is pairwise, level-2 carries genuine factor (3-way+) structure.

== The general factor lattice, and the affine wall

The original LDT formulation is binary (pairwise exclusion). Our harness generalizes it to a #term[factor
lattice of any arity]: a factor is a tuple of variables plus an allowed-tuples table, and narrowing
propagates through factors, not just edges. This matters because of one recurring obstacle we name the
#term[affine wall].

#keynote[
*The affine wall = width.* Three-way / modular-arithmetic / XOR-like constraints (the "affine" structure,
e.g. $x + y + z equiv 0$) cannot be captured by *any* amount of pairwise propagation: per-cell AC is
provably stuck. They require a *grade-3 / ternary* primitive in the per-step computation. In the harness,
the factor lattice "breaks XOR's affine wall at level-2" — the factor structure is exactly what supplies
the missing arity. The wall is therefore a *width* (per-step expressivity) limit, not a *depth* one; see
§4 and the macro-deduction result for why depth alone cannot manufacture it.
]

== The polymorphism diagnostic

Constraint tractability has a classical algebraic signature: a CSP is tractable iff its constraint language
admits certain *polymorphisms* (closure operations). The harness ships a polymorphism diagnostic so that,
for any generated instance class, we can tell *in advance* whether per-cell propagation should suffice or
whether factor-level structure is required. This is what lets us aim experiments at the regime a method
targets — a discipline lesson learned the hard way (don't conclude a method failed from a mis-aimed run).

== The verifier ladder

On top of the CSP core sits a ladder of exactly-verifiable tasks that double as the training curriculum and
the generality test (`curriculum.py`, `fol.py`, `smt.py`, `schedule.py`): a forward-chainer giving
entail / contradict / unknown labels, a `z3` SMT oracle, a multi-task scheduler, and Bedrock-rendered
natural-language surface forms so the same logical problem appears in many phrasings. The diversity *is* the
generality measurement: hold out a phrasing or a problem size, and OOD soundness becomes observable.