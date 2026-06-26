# Organ excellence — comprehensive characterization of the deduction organ

Dedicated organ-excellence track (us-east-1 organ box, T4). The finding that motivates it:
**OOD is limited by the organ's own recall, not the readout** — so the highest-leverage work is
making the pure organ (no LLM) as flexible and powerful as possible. This note characterizes the
organ comprehensively, finds where it is weak, and pushes it toward excellence.

Everything is measured against the EXACT finite-CSP harness (`clair.csp`): soundness = false-elim
vs the exact per-cell transformer dedₚ (must be 0); completeness = narrowing-recall = fraction of the
(cell,value) eliminations dedₚ makes that the organ also makes. The organ is the general
factor-graph narrower (`clair.proposer.FactorGraphProposer` / `clair.blade_deductor.BladeFactorDeductor`).

New code for this track:
- `clair/rg_csp.py` — reasoning-gym → clair.csp.CSP classification (all 106 tasks) + adapters + exact
  task-level characterization + a budget-filtered organ-eval sampler.
- `clair/sweep_organ.py` — capacity × depth sweep (sizing study).
- `clair/organ_excellence.py` — the integrated "best organ" trainer + checkpoint + zero-shot rg eval.

Artifacts on the box: `runs/rg_coverage.json`, `runs/sweep.json`, `runs/blade_full.json`,
`runs/best_organ.pt`, `runs/best_organ_eval.json`.

---

## 1. Full reasoning-gym coverage (the breadth/generality test)

reasoning-gym registers **106 procedural tasks**. Classification by whether the task is a
finite-domain assignment problem the organ can narrow (and whether it maps to `clair.csp.CSP`, so the
exact dedₚ gives ground truth):

| category | count | meaning |
|---|---|---|
| `csp_fits` | 2 | finite-domain CSP that fits the trained budget (N≤8, D≤8, A≤3, M≤28) |
| `csp_overbudget` | 7 | finite-domain CSP, but exceeds cells/domain/arity budget (needs a larger organ) |
| `csp_boolean` | 4 | boolean SAT/logic CSP (truth assignment) |
| `not_csp` | 93 | not a finite-domain assignment-narrowing problem |

**The honest headline: only ~13 of 106 reasoning-gym tasks (12%) are even expressible as the
finite-domain CSP the organ narrows.** The other 88% are arithmetic evaluation, string
rewriting/transforms, simulation (game-of-life, cube rotation), planning (sokoban, hanoi, rubik,
rush-hour, jugs), graph traversal/path (maze, shortest-path, word-ladder), counting, probability,
symbolic calculus, sequence/grid INDUCTION (ARC, list-functions, number-sequence), and entailment
(syllogism, propositional). None of those are per-cell candidate-set narrowing; the lattice organ
**simply cannot express them** — they need the other organs in the bank (forward-chaining for
entailment/derivation; program-induction for ARC/list-functions; energy/equilibrium or explicit
search for planning; plain compute for arithmetic).

### CSP-shaped tasks (the organ-applicable subset)

| task | category | adapter | encoding |
|---|---|---|---|
| graph_color | csp_fits | ✅ | pairwise ≠ on edges; d=num_colors |
| n_queens | csp_fits | ✅ | one var/row, domain=column; ≠col & off-diagonal pairs; pins=preplaced |
| knights_knaves | csp_boolean | ✅ | person truth ↔ eval(statement); arity = #people in clause |
| mini_sudoku | csp_overbudget | ✅ | 16 cells > N_MAX; alldiff rows/cols/boxes (d=4) |
| futoshiki | csp_overbudget | ✅ | board² cells; alldiff rows/cols + < pairs + pins |
| sudoku | csp_overbudget | (same shape) | 81 cells, d=9 — far over budget |
| cryptarithm | csp_overbudget | — | d=10 > D_MAX; alldiff + per-column carry sums (arity > 3) |
| kakurasu / survo | csp_overbudget | — | row/col SUM factors are high-arity (> A_MAX=3) |
| zebra_puzzles | csp_overbudget | — | logic grid; structure not exposed in metadata |
| propositional_logic | csp_boolean | — | models of premises = boolean CSP, but task asks ENTAILMENT not assignment |
| circuit_logic | csp_boolean | — | deterministic boolean EVALUATION (CSP-expressible, trivially solved) |
| self_reference | csp_boolean | — | self-referential truth count; bespoke |

### Task-level exact ceiling (organ-independent; `clair.csp` only)

The most a SOUND per-cell organ could ever recall, measured on 30 instances/task with the exact
harness. `dedₚ-narrow-ceiling` = fraction of the full grid the exact per-cell transformer eliminates;
`uniq` = fraction the puzzle is uniquely determined to a single solution; `factor-solved` = the
level-2 triple-factor lattice drives it to all-singleton.

| task (config) | cells | d | maxarity | solvable | dedₚ-ceiling | uniq | factor-solved | poly SL/Maj/Aff |
|---|---|---|---|---|---|---|---|---|
| graph_color (v=8, k=3) | 8 | 3 | 2 | 30/30 | **0.0%** | 0% | 0% | 3/3/3 |
| graph_color (v=6, k=3) | 6 | 3 | 2 | 30/30 | **0.0%** | 0% | 0% | 30/30/30 |
| n_queens (n=5) | 5 | 5 | 2 | 30/30 | 75.7% | 73% | 73% | 0/0/0 |
| n_queens (n=6) | 6 | 6 | 2 | 30/30 | 83.3% | 100% | 100% | 0/0/0 |
| mini_sudoku | 16 | 4 | 2 | 30/30 | 75.0% | 100% | 100% | 0/0/0 |
| futoshiki (board=4) | 16 | 4 | 2 | 30/30 | 75.0% | 100% | 100% | 0/0/0 |
| knights_knaves (2 ppl) | 2 | 2 | 2 | 30/30 | 50.0% | 100% | 100% | 57/100/63 |
| knights_knaves (3 ppl) | 3 | 2 | 3 | 30/30 | 50.0% | 100% | 100% | 10/10/17 |

Reads:
- **graph_color is a vacuous narrowing target** — reasoning-gym generates it sparse (few edges, 3
  colors), so the exact dedₚ can eliminate *nothing*: every color survives at every vertex. An organ
  scores 100% "recall" of 0 eliminations — meaningless. (graph_color is the canonical CSP, but only
  *densely-constrained* instances narrow; rg's distribution doesn't.)
- **n_queens / mini_sudoku / futoshiki are excellent narrowing targets** — high ceiling, uniquely
  determined, and the triple-factor lattice (level-2) fully solves them. mini_sudoku/futoshiki need
  16 cells, **over the trained organ's N_MAX=8 budget** — expressible, but the trained organ can't run
  on them without re-budgeting (a prioritized weakness, §5).
- **knights_knaves** is a clean boolean CSP (majority polymorphism at n=2 → bounded width; some
  affine at n=3); uniquely determined, factor-solved.

### Zero-shot organ on reasoning-gym (trained ONLY on the curriculum)

The best organ (§4, blade, R12, 396k params), trained only on the curriculum ladder, run on the
budget-fitting rg adapters (150 instances each). This is the breadth test and it **surfaces the
central weakness**:

| rg task (zero-shot) | n | recall(dedₚ) | match-fix | solved | FALSE-ELIM |
|---|---|---|---|---|---|
| n_queens (n=5) | 150 | 58.9% | 0.0% | 0.0% | **0.0379 (77)** |
| n_queens (n=4) | 150 | 63.2% | 0.0% | 0.0% | **0.0178 (28)** |
| knights_knaves (2) | 150 | 53.3% | 50.7% | 50.7% | 0.0190 (11) |
| knights_knaves (3) | 150 | 23.1% | 13.3% | 18.7% | 0.0117 (15) |
| graph_color (v=8) | 150 | (vacuous: 0 elim) | 100% | 0.0% | 0.0000 (0) |

Compared to the near-ceiling curriculum recall (§4, 90–100% per rung), zero-shot recall **drops to
23–63%** AND soundness BREAKS (false-elim 0.018–0.038 on n_queens — the organ eliminates values some
solution uses). knights_knaves at 2 people the organ actually SOLVES 50.7% (the truth tables overlap
with trained ≠/= structure), but at 3 people (arity-3 implication/iff tables it never saw) recall
collapses to 23%.

The lesson: **soundness is empirical, not guaranteed off-distribution.** The dominate-dedₚ training
makes the organ sound *on the trained relation families*, but on a NOVEL relation table (n_queens'
off-diagonal `|i−j|≠|cᵢ−cⱼ|`, KK's implication/iff truth tables) the learned narrowing both
under-recalls AND eliminates values some solution uses (false-elim > 0). The output-check would
still catch the unsound answer at the boundary, but the *organ itself* is not sound OOD.

---

## 2. Sizing & depth study (capacity × rounds)

`clair/sweep_organ.py`: param-target ∈ {5e4, 1e5, 2e5, 4e5} (→ d_model) × R ∈ {2, 4, 8, 16} rounds,
trained multitask on all 6 curriculum rungs (600 steps), eval per rung (n=120). **R is a pure DEPTH
knob: the message-passing modules are shared across rounds, so d_model and param-count are identical
across R at a fixed target** — this cleanly isolates depth from width.

Mean narrowing-recall (across rungs), vs exact dedₚ:

| params (d_model) | R=2 | R=4 | R=8 | R=16 |
|---|---|---|---|---|
| 50k (d=46) | 67.7% | 66.8% | **79.5%** | 76.8% |
| 97k (d=70) | 67.7% | 77.9% | **85.8%** | 83.5% |
| 205k (d=110) | 79.5% | 85.5% | **90.3%** | 89.0% |
| 403k (d=162) | 87.5% | 90.5% | **93.8%** | 91.2% |

**The depth-for-width law.** (1) R=2 is depth-starved at *every* width (~67% at low width) — too few
rounds to propagate. (2) The recall jump needs **R=8 universally** (best column everywhere); R=16 is
consistently slightly WORSE (harder to optimize at fixed 600 steps, and false-elim creeps up). (3)
**Wider organs amortize depth**: at 50k you NEED R=8 to get the jump (R=4 still 67%); at ≥100k, R=4
already captures most of it. So width and depth substitute — a wider organ needs fewer rounds.
(4) Width helps monotonically; soundness (false-elim) stays ≤0.003 mean and degrades mildly with R.

**Recommended sizing: ≥200k params, R≈8.** Per-rung, the laggard is always `smt` (mixed
le/lt/sum/diff/mod, under-determined → match-fix ~12%) and `ordering` (needs transitive closure over
long chains → depth-limited match-fix ~50%); `coloring`/`alldiff` saturate to ~100%.

---

## 3. Factor generalization (the zero-shot-to-novel-relation gap)

`clair/run_blade.py --mode full`: blade (grade-k arity prior) vs table (generic relation-table MLP),
iso-param. The relation-table representation is general but the LEARNED narrowing fails zero-shot on
NOVEL relation families (the prior result: arithmetic held-out → 13%). Three experiments:
A. multitask (train all, eval per rung); B. zero-shot-3 (train arity≤2 only, eval arithmetic+xor —
NO arity-3 factor ever seen); C. cross-affine (train arity≤2 + xor, eval arithmetic — a DIFFERENT
arity-3 affine relation). Iso-param (table d=108/199k ≥ blade d=96/194k — blade gets no param edge).
1200 steps, n=200 eval.

| experiment | rung | blade recall / FE | table recall / FE |
|---|---|---|---|
| A. multitask (in-distribution) | arithmetic | 98.4% / 0.0004 | 96.2% / 0.0004 |
| A. multitask | ordering | 98.0% / 0.0005 | 95.0% / 0.0005 |
| A. multitask | smt | 90.8% / 0.0032 | 89.1% / 0.0019 |
| A. multitask | (others) | 96–100% / ~0 | 97–100% / ~0 |
| **B. zero-shot arity-3** (never saw arity-3) | arithmetic | 83.5% / **0.0732** | 90.0% / **0.3629** |
| **B. zero-shot arity-3** | xor | 79.7% / **0.1637** | 81.2% / **0.1884** |
| C. cross-affine (xor→arithmetic) | arithmetic | 92.2% / **0.0614** | 88.9% / 0.0842 |

**The blade arity prior buys SOUNDNESS off-distribution, which is the whole game.** In-distribution
blade ≈ table (marginal blade edge on the harder arity-mixed rungs). But zero-shot to a NEVER-SEEN
arity-3 relation, the table's soundness **collapses — false-elim 0.36** (it wrongly eliminates 36% of
the values dedₚ keeps), because its arity-3 relation-table slot was never exercised so it emits
garbage. The blade's grade-3 trivector path is structurally correct *a priori* (it's the
antisymmetric form, exists whether or not an arity-3 factor was trained), so its false-elim stays
**0.07 — 5× more sound**. On cross-affine (saw a *different* arity-3 affine, xor), blade is both more
complete (92.2 vs 88.9%) and more sound. This **closes the zero-shot-to-novel-relation gap** the
prior result flagged (arithmetic FE→13% completeness): training on diverse families lifts zero-shot
*completeness* to 83–92%, and the blade prior is what keeps it *sound* while doing so.

---

## 4. The best organ (integrated config + checkpoint)

`clair/organ_excellence.py --train`: general factor rep + blade arity prior (grade-k == arity-k) +
sufficient depth (R rounds) + monotone-meet set-valued readout, trained on all 7 rungs incl xor.
Checkpoint saved to `runs/best_organ.pt` (state_dict + config for the assembly to bootstrap+freeze).

**Config:** `BladeFactorDeductor(msg="blade")`, d_model=96, **R=12**, ds=4, budget N=8/D=8/M=28/A=3,
**396,441 params**, trained 1500 steps on all 7 rungs (incl xor), pool=128, lr=3e-4, θ=0.5.

**Per-rung completeness (curriculum, n=150), the BEST organ:**

| rung | recall(dedₚ) | match-fix | solved (uniq) | FALSE-ELIM | wrong |
|---|---|---|---|---|---|
| coloring | 98.6% | 97.3% | 2.7% (3.3%) | 0.0000 | 0.0% |
| equality | 98.8% | 95.3% | 8.0% (8.7%) | 0.0000 | 0.0% |
| ordering | 97.9% | 85.3% | 6.0% (6.7%) | 0.0005 (2) | 0.0% |
| arithmetic | 99.1% | 96.7% | 42.0% (42.7%) | 0.0005 (2) | 0.7% |
| xor | 99.5% | 98.7% | 41.3% (41.3%) | 0.0000 | 0.0% |
| alldiff | 100.0% | 100.0% | 18.7% (18.7%) | 0.0000 | 0.0% |
| smt | 90.9% | 44.7% | 30.0% (58.7%) | 0.0013 (5) | 0.7% |

The best organ reaches **97–100% narrowing-recall on 6 of 7 rungs with false-elim ≈ 0** (sound), and
*matches the exact per-cell dedₚ fixpoint* (match-fix) 85–100% on those rungs — it has essentially
learned the exact strongest sound per-cell narrowing. `smt` remains the laggard (90.9% recall, but
match-fix 44.7%: it gets most eliminations but rarely the *complete* fixpoint on the mixed-constraint
instances). `solved`/`uniq` are low because most instances are under-determined (the exact ceiling is
far below all-singleton); on the *determinable* fraction it solves nearly all (e.g. arithmetic
42.0/42.7, alldiff 18.7/18.7). `wrong` (returns a non-solution) ≤0.7% — the rare unsound returns.

This is the checkpoint to bootstrap+freeze for the assembly.

---

## 5. Prioritized remaining weaknesses (what to improve next)

1. **Soundness breaks OOD on novel relation tables** (n_queens off-diagonal, KK implication). The
   dominate-dedₚ guarantee is empirical on trained families only. Fix: train on far MORE diverse
   relation families; add the output-verify gate as a hard mask; or a soundness-calibrated θ per
   relation. *Highest priority — it's the OOD-recall ceiling.*
2. **Budget ceiling: N_MAX=8 cells.** The most valuable rg CSPs (mini_sudoku/futoshiki=16,
   sudoku=81, cryptarithm) need ≥16 cells / d=10 / arity>3. Train a larger-budget organ AND extend
   the curriculum to large-alldiff (the curriculum's alldiff rung is n≤4 → no 16-cell exposure).
3. **alldiff completeness gap (Hall-set propagation).** alldiff is decomposed to pairwise ≠, which
   binary AC cannot fully propagate; the organ inherits this. A native alldiff/global factor would
   close it.
4. **High-arity factors (>3).** kakurasu/survo/cryptarithm have row/col SUM and carry constraints of
   arity 4+. A_MAX=3 caps the blade at the trivector; the exterior-fold (`blade_hi.scan`) handles any
   arity but is untested at scale here.
5. **Only ~12% of reasoning-gym is organ-expressible.** This is not a bug — it's the boundary of the
   lattice organ. Generality at the catalog level needs the rest of the organ bank
   (forward-chaining, program-induction, planning/search), per `notes/future_directions.md §3`.

### Methods / efficiency note
The bottleneck for these sweeps is the exact-dedₚ / featurize **data path, which is single-threaded
Python on CPU** — on the 16-vCPU box it used ~2 cores and left the T4 idle (the models are tiny,
<0.5M params; the GPU is never the limit, the CPU dedₚ ground-truth is). The per-instance exact-dedₚ
generation is embarrassingly parallel — a `concurrent.futures.ProcessPoolExecutor` over instances
(or the parallel+cached dedₚ data path being added to `clair/`) would give ~8× data throughput and
unblock larger sweeps (especially the larger-budget organ for the 16-cell rg CSPs in weakness #2).
The runs in this note all completed (sweep ~13 min, blade ~10 min, best-organ ~4 min), so this is an
optimization for the *next* round, not a re-run of finished work.
