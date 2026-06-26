# Abstract machines & reductions for GLaDOS's organ bank

Two design studies for the next additions to the bank (`clair/organ/bank.py`, `bank_woven.py`,
`compose.py`; the NP→Ising encodings already in `clair/ising_organ.py`). Both are written against the
typed spine in `protocol.py`: a faculty is a `Reduction` with `applies / reduce / certificate /
survival`, the composer reduced-products within a state type, and soundness is either
by-construction or verifier-gated / output-checked.

The honest one-line frame: **we already ship the expensive parts** — exact Lucas NP→Ising encoders +
decoders + exact ground-truth checkers (`ising_organ.py`), the shared `CSPState`, and an *exact*
Datalog chainer (`unification_chain`). The leverage is in wiring what we have, not in importing a new
differentiable interpreter.

---

## IDEA 1 — reductions between problems as a multimodal faculty

### 1.1 The reduction graph among our faculties

Faculties we ship (bank + ising_organ), as nodes; edges = a problem-to-problem map. We only count an
edge if it is **exact** (a solution of the target decodes to a valid solution of the source) and
**cheap to implement + verify** with code we already have or can write in a few lines.

| edge (source → target) | where | exact? | cost (blow-up) | status |
|---|---|---|---|---|
| graph k-coloring → CSP (≠-constraints) | native `CSPState` | yes | n cells | **have** (curriculum `eq/neq`) |
| k-coloring → Ising | `ising_organ.encode_coloring` (Lucas §6) | yes (H=0 ⇔ proper) | k·n spins | **have** |
| MaxCut → Ising | `encode_maxcut` (H itself) | yes | n spins | **have** |
| number-partition / subset-sum → Ising | `encode_partition` | yes | n spins | **have** |
| max-independent-set → Ising | `encode_mis` (Lucas §4.2) | yes | n spins | **have** |
| QUBO → Ising | `qubo_to_ising` (verified exhaustively in `--smoke`) | yes | identity | **have** |
| XOR system → GF(2) row-space | `gf2_rowspace` | yes (complete) | identity | **have** |
| affine/parity over ℤ_m → SNF | `modular_snf` | yes (complete) | identity | **have** |
| path/chain CSP → reach-doubling | `macro_reach` | yes | O(log L) depth | **have** |
| Datalog/FOL query → least-Herbrand closure | `unification_chain` | yes | identity | **have** |
| **MIS ↔ vertex-cover** (complement set) | trivial (VC = V∖IS) | yes | identity | **ADD** (~10 lines) |
| **MIS ↔ max-clique** (complement graph) | already used in `exact_mis` | yes | identity | **ADD** (~10 lines) |
| **3-SAT → CSP** (clause → ternary allowed-tuple over {0,1}) | new typed edge | yes | m ternary factors | **ADD** (cheap) |
| **3-SAT → Ising** (Lucas via MIS-gadget / clause penalty) | new typed edge | yes (H=0 ⇔ SAT) | O(m) spins | **ADD** |
| k-coloring → 3-coloring (gadget) | classic | yes but gadget-heavy | ×const, messy | **PARK** |
| alldiff → bipartite matching (Hall/Régin) | no matching organ | exact propagator | — | **PARK** (no faculty) |

Two natural **hubs** fall out:

1. **Everything → Ising (Lucas 2014, arXiv:1302.5843).** This is the cleanest universal target *we
   already implement and exhaustively verify* (`qubo_to_ising` max-err < 1e-3 over all 2ⁿ in
   `--smoke`; per-problem decoders + brute/`networkx` ground truth in `run_substrate`). One organ
   (`mean_field_anneal` + sound greedy descent + exact `ising_energy` reward) solves the lot.
2. **Everything → `CSPState`.** coloring / alldiff(≠-clique) / ordering / arithmetic-mod already
   *share* the reduced-product spine — they are "reductions to the common type" by construction, and
   the composer + certified floor already act on them.

Prefer CSP when a direct narrower applies (cheaper, and it can be *certified*); fall back to the
Ising hub (universal, but `approximate`).

### 1.2 Reductions are a NEW typed object (cross-type), distinct from the composer

The composer (`compose.reduced_product`) composes *within* a state type by meet. A reduction routes
*across* types and is its own typed object — propose adding to `protocol.py`:

```python
class ProblemReduction(ABC):
    name: str
    src_type: str            # e.g. "sat", "vertex_cover"
    dst_type: str            # e.g. "csp-domain", "ising"
    def applies(self, x) -> bool: ...
    def encode(self, x) -> Any: ...          # X-instance  → Y-instance  (the reduce-X→Y map)
    def decode(self, y_solution) -> Any: ...  # Y-solution  → X-answer
    def certificate(self) -> Certificate: ... # exact / gadget / approximate
    def cost(self, x) -> float: ...           # encoding blow-up, for routing
```

Each *solution path* through this graph (X → direct faculty) vs (X → Y → faculty) is a **"modality"**
— same answer, different route — which is the multimodal framing: the LM/organ learns
*recognize-X · reduce-X→Y · solve-via-Y · decode-back*, and cross-faculty routing is a
shortest-path over `ProblemReduction` edges.

### 1.3 The curriculum

Witness-first generation already gives us free exact labels; a reduction record just stacks one exact
map on top:

```
rec = {
  "problem":      X,                 # X in its native rendering (e.g. a 3-SAT instance)
  "reduction":    "sat→ising",       # the named ProblemReduction edge (carries its certificate)
  "target":       Y,                 # encode(X): the Ising (J,h) or the CSP
  "solution_Y":   solve_faculty(Y),  # the faculty's answer on the target type
  "answer_X":     decode(solution_Y),# mapped back
  "cost":         edge.cost(X),      # blow-up, for efficient-reduction selection / routing tie-break
  "route_required": bool,            # OOD flag: X has NO direct faculty in train (see §1.5)
}
```

End-to-end **exact-verified**: generate X witness-first (X has a known solution s_X) → `encode` → solve
Y → `decode` → check `answer_X` with X's *own* native checker. This is exactly the loop `ising_organ`
already runs in `smoke`/`run_substrate` (decode + `exact_maxcut/partition/chromatic/mis`); we are only
promoting it from a benchmark harness to a *training record* shape. No new oracle, no human labels.

**Picking the efficient reduction (cost).** Edge weight = encoding blow-up + estimated target solve
cost; route by: (a) if a direct faculty `applies(X)`, use it; (b) else Dijkstra over `ProblemReduction`
edges to the cheapest faculty that applies; (c) tie-break on certificate strength (certified target >
neural/approx). Coloring is the worked example: coloring→CSP costs n cells and is *certifiable*;
coloring→Ising costs k·n spins and is only output-checked — so the router prefers the CSP route unless
the affine/Hall wall makes the narrower abstain.

### 1.4 What the literature actually says (skeptical)

- **Prates et al., "Learning to Solve NP-Complete Problems: A GNN for Decision TSP"** (arXiv:1809.02721,
  AAAI'19). A message-passing net learns the decision-TSP boundary and generalizes across sizes and
  across the cost threshold. **But it learns one problem end-to-end; there is NO cross-problem
  transfer and NO reduction structure.** Evidence a single net *can* absorb an NP-complete decision
  boundary — not evidence for reduction-as-transfer.
- **NeuroSAT** (Selsam et al., arXiv:1802.03685, well-known; not re-read here). The SAT analog, and
  relevant precisely because *everything reduces to SAT* — a SAT faculty is a universal target, the
  discrete-logic twin of our Ising hub. Still single-problem; no reduction curriculum.
- **Curriculum-by-transfer** (Hacohen & Weinshall, arXiv:1802.03796) — generic theory that an easier
  source task can order a curriculum; tangential, not problem-reduction.
- **The gap:** nobody trains on a *reduction curriculum* where supervision is the reduce-then-solve
  trace with exact end-to-end verification. The Karp reduction graph — the one structural prior these
  GNN-NP papers ignore — is the thing we can hand the model nearly for free, because the exact
  encoders/decoders/checkers already live in `ising_organ.py`.

### 1.5 OOD "solve-only-via-reduction" split

Hold a problem type entirely out of the *faculty* set in train, keep only the reduction edge to it.
Concretely: train on `{coloring, Ising}` faculties + the `coloring→Ising` edge; **test on
vertex-cover**, which has no direct faculty and must be solved as `VC → MIS → Ising → decode`. Tag
those records `route_required: true` + the held-out type. If reduce-then-solve transfers, this is the
direct measurement of "one faculty + reductions = many problems."

### 1.6 Verdict (Idea 1)

**Yes — cheapest generality multiplier on the table.** The hard parts (exact Lucas encoders, decoders,
exact ground-truth checkers, the shared `CSPState`) are already shipped and verified. Cost to add:
the `ProblemReduction` type + a routing curriculum + three exact edges. **First reductions to add:**
(1) **MIS ↔ vertex-cover ↔ max-clique** (trivial complement maps — two new problem types for free off
the existing Ising MIS path); (2) **3-SAT → CSP** (clause → ternary allowed-tuple; feeds the certified
narrower + the verifier-gated `core_narrow_organ`); (3) **3-SAT → Ising** (Lucas; routes SAT to the
optimizer, H=0 ⇔ satisfiable). These turn the existing {CSP, Ising} pair into a small Karp
neighborhood — SAT, 3-SAT, coloring, MaxCut, partition, MIS, VC, clique — all exact-verifiable, all
reducible to a faculty we already ship.

---

## IDEA 2 — which abstract-machine faculties to add

### 2.1 NSAM framework — what it actually is

**Neuro-Symbolic Abstract Machines** (Nada Amin's group, Harvard *metareflection*; Retchin, Sanna,
Byrd, Amin): *neural networks structurally equivalent to programming-language interpreters.* Take the
abstract machine for a computational paradigm, make its transition function differentiable, and leave
"holes" (slots) filled by learning from I/O. The public project page is a thin overview; the concrete
artifacts are **Bošnjak, "On Differentiable Interpreters"** (UCL PhD thesis, 2021) and the
**`namin/relaxed-machines`** repo, which implements small machines (`inc_stop`, `dup_add`, `reg_jmp`,
`sub` — counter/register machines) where holes are *whole neural networks* ("hard sketches").

The framework's paradigms map onto our faculties:

| NSAM paradigm / machine | computational class | our faculty |
|---|---|---|
| stack machine / Forth (∂4) | procedural / iterative (Turing) | **MISSING** — "execute a small program" |
| register / counter (relaxed-machines) | imperative loops | partial (`macro_reach` for paths) |
| logic / relational (Datalog, miniKanren) | PTIME fixpoint / entailment | **HAVE, certified** (`unification_chain`) |
| functional / lambda (neurallambda) | Turing-general | none — PARK |

The NSAM pattern *is* the bank's existing pattern (a sound/exact symbolic core + a learned, gated
slot), generalized from CSP narrowing to general computation.

### 2.2 ∂4 — Differentiable Forth (Bošnjak & Rocktäschel et al., arXiv:1605.06640, ICML'17)

**Mechanism (from the paper, §2–3).** Forth's abstract machine is a state `S=(D,R,H,c)`: data stack
`D`, return stack `R`, heap `H`, program counter `c` (Turing-complete, stack-based). ∂4 relaxes it:

- **Differentiable memory** — flat NTM-style buffers `M ∈ {D,R,H}` with read `aᵀM` and write
  `M ← M−(a1ᵀ)⊙M + xaᵀ`; stack pointers `d,r` are vectors shifted by `inc/dec` circular-shift
  matrices; `push/pop` = write/read + pointer shift.
- **Soft program counter** `c` = attention over program positions.
- **Neural Forth words** — `DUP`,`DROP`,… lifted to differentiable ops on the continuous state.
- **Program sketches** `P=w₁…wₙ` — the programmer writes the *known* control structure (loop,
  recursion) in Forth and leaves **slots** `{…}` = trainable transition functions, syntax
  `{encoder → decoder}` with decoders `choose` (weighted pick over Forth words) / `manipulate` /
  `permute`. Holes are learned from I/O of stack start/end states.
- **Execution RNN** `S_{n+1} = Σ_i c_i·w_i(S_n)` — execute *all* words, weight by the PC. Tractable
  only via **symbolic execution** (collapse branch-free word sequences) + **if-branch interpolation**.
- **Training** — cross-entropy on target stack memory + pointer with a care-mask.

**What it buys us.** The only **procedural / iterative "execute-a-small-program" primitive** in the
zoo — sequential algorithms (sort, running aggregate, multi-digit carry-arithmetic, stack-structured
parsing) that the *declarative* faculties (narrowing, chaining, energy) structurally cannot express.
This is the "program-induction / Turing-general" module `future_directions.md §3` flags as the hardest.

**Soundness / protocol fit.** Output-checkable *only* when the task has a cheap verifier (sorting:
sorted + permutation; addition: re-add). So as a `Reduction` it is **`approximate`** (like
`energy_optimise`/Ising): sound *only* via the output-check, never sound-by-construction. It carries
its own `state_type="forth-machine"` (stacks+heap+pc), does **not** reduced-product with `CSPState`,
and **cannot** be verifier-gated by `exact_dedP` (CSP-only) — its gate is a task-specific output
checker. `survival()` = the decoded output one-hot.

**Honest limitations (paper §4, Table 1).** Length generalization is **fragile**: training on
length-4 sequences *collapses* (Permute 19.82% @ test-len 8, 7.81% @ 64; Compare 49.22 / 20.65) — only
short, clean train lengths + **test-time discretization** generalize. The full-RNN-weight-all-words
step is expensive and brittle past tiny programs. The sketch must encode most of the structure — the
net learns the *hole*, not the algorithm. **Verdict: build it SECOND**, scoped to checkable iterative
tasks; it adds a genuinely new capability class but is the brittlest and least-checkable faculty.

### 2.3 DF-εL++ / "Fast and Faithful" — differentiable PTIME logic / Datalog

The faculty = **differentiable Datalog**: logic-program forward-chaining where rule firing stays
*exact symbolic* but predicate/fact weights are differentiable. Exemplars:

- **Scallop** (arXiv:2304.04812, PLDI'23) — differentiable Datalog via **provenance semirings**;
  symbolic semantics exact, only the weights soft.
- **"Fast and Faithful: Scalable Neuro-Symbolic Learning and Reasoning"** (OOPSLA'25, ACM DOI
  10.1145/3770854.3780220) and **Dolphin** (arXiv:2410.03348, ICML'25) — *faithful* = exact symbolic
  reasoning (Dolphin: on CPU) + **vectorized gradient propagation** (GPU), 1.7–62× over Scallop/ISED.
- **Differentiable Logic Machines** (arXiv:2102.11529, TMLR) — weighted-*predicate* relaxation for
  inductive logic programming, learned progressively.

**Key observation:** this is the differentiable *generalization of a faculty we already ship in its
strongest form.* `unification_chain` = `fol.forward_chain` = the **exact** least Herbrand model,
**sound-by-construction**, 3-way entail/contradict/unknown. A "PTIME-Datalog organ" is ~90% already
built — and *certified*. The differentiable versions only trade soundness for *learnable rule
weights*, which we want **only if** the curriculum needs rule *induction*. For solve-by-given-rules
(our entailment tasks) the certified chainer is strictly better.

**Verdict: HIGHEST faculty, but mostly already have it.** Cheap win = extend `unification_chain`
(more rules, stratified negation, recursion depth) and, *only if* we add rule-induction tasks, bolt a
neural rule-proposer **verifier-gated by the exact chainer** — the exact analog of
`core_narrow_organ` gated by `exact_dedP`. Lowest-risk faculty because the certified floor exists.

### 2.4 PARK: full lambda / RISC / Turing-general

**neurallambda / "A Neural Lambda Calculus"** (arXiv:2304.09276) and full differentiable RISC. Judged
by the four criteria they fail the gates: **checked** — no cheap output checker for general functional
programs; **learnable** — the least stable / hardest to train; **composable / protocol-fit** — no
clean lattice or state bridge, no certificate. These are the Turing-general endpoint
`future_directions.md` already flags as "hardest"; defer until ∂4-procedural and the Datalog faculty
are paying off.

### 2.5 Ranked verdict (Idea 2)

| faculty | checked-at-output | learnable | composable | protocol-fit | call |
|---|---|---|---|---|---|
| **PTIME-Datalog** (extend `unification_chain` + gated rule-proposer) | exact (least Herbrand) | certified needs none; soft only for induction | own `fol-closure` state, bridges | perfect (it *is* a faculty) | **1st — cheapest, certified floor exists** |
| **Forth / ∂4 procedural** | only via task checker (`approximate`) | yes but brittle; needs sketches + test-discretization | own `forth-machine` state, new bridge | clean standalone `approximate` (not reduced-product) | **2nd — new capability, scope to checkable tasks** |
| **lambda / RISC / Turing-general** | no cheap checker | poorly | no lattice/state bridge | no certificate | **PARK** |

---

## Single highest-leverage next addition

**Idea 1's reduction graph — not a new abstract machine.** Wire the existing exact encodings as typed
cross-type `ProblemReduction` edges (everything → Ising; MIS ↔ vertex-cover ↔ clique; 3-SAT → CSP /
Ising) plus a routing curriculum with the `route_required` OOD split. It is the cheapest (the exact
encoders, decoders, and exact checkers already live in `ising_organ.py`), it multiplies generality
across problem types with *one faculty + edges*, and it is directly testable via the
solve-only-via-reduction split. Among genuinely *new* abstract machines, build the differentiable
upgrade of `unification_chain` first (its certified floor already exists), then ∂4-procedural (scoped,
output-checked); leave lambda/RISC parked.

PDFs saved to `clair/pdfs/`: `diff_forth_1605.06640.pdf`, `learn_npcomplete_gnn_1809.02721.pdf`.
</content>
</invoke>
