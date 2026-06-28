# Multi-faculty curriculum — per-faculty difficulty metrics + cross-faculty composition-occupancy

**Goal.** The organ is now a multi-faculty fabric (csp · graph · ising · type · reduction-routing + the
cross/tri multi-hop flows, `clair/organ/faculty.py`), but the curriculum is still CSP-difficulty + a few
hand-picked cross tasks. The coverage-hole lesson from the CSP difficulty work (`notes/curriculum_design.md`:
train is 98.7% per-cell-level-0, so the organ pretrains where AC already solves everything) applies *per
faculty* AND *across the composition lattice*: **the model only generalizes where the curriculum spans the
multifaculty complexity.** This note specifies the per-faculty difficulty axes, the current-coverage audit
(the numbers — each faculty's generators cluster easy, like CSP did), the cross-faculty occupancy framework
+ buckets + OOD splits, and the ranked generator specs the expansion should build (gated on the
bank-completion landing, which owns `faculty_tasks.py` + `multifaculty.py`).

Foundation built here: `clair/organ/faculty_difficulty.py` — `difficulty(struct, faculty)`,
`profile_faculty(samples, faculty)`, the `CompositionBucket` lattice + `cross_occupancy`, `OOD_SPLITS`,
`GENERATOR_SPECS`. New file only; `organ.selftest` stays PASS (it imports faculty/ising_organ/typeinfer/
datagen.difficulty read-only and edits nothing). Run: `python -m clair.organ.faculty_difficulty`.

---

## 1. Per-faculty difficulty axes (the intrinsic-hardness vectors)

Each faculty gets its OWN intrinsic-hardness axes, exact/cheap, verified against a brute reference where
feasible. The CSP faculty reuses `datagen/difficulty.py` (level / treewidth / depth) unchanged.

| faculty | axes | what's hard | brute reference |
|---|---|---|---|
| **csp** | `level` (cheapest lattice level reaching exact dedₚ), `treewidth` (min-fill UB), `depth` (AC iters) | affine-wall / wide / deep-propagation | exact_dedP |
| **ising** | `frustration` (Harary index = fraction of bonds unsatisfied at a ground state), `coupling_density`, `ground_degeneracy` (#ground states), `abs_J_spread` (std of \|J\| = spin-glass-hard proxy) | frustrated / glassy / degenerate landscapes (argmin the LM can't express) | brute ground state (n≤8), + frustrated-triangle ⊆ exact |
| **graph** | `reach_depth` (min-plus closure depth from source), `diameter` (directed), `reach_frac`, `edge_density` | deep long-range reachability chains | BFS/min-plus closure (exact) |
| **type** | `prop_depth` (AC iters = unification-chain depth), `poly_breadth` (mean \|dedP\| = residual polymorphism), `nesting_depth` (type-term nesting), `treewidth` | deep eq-chains, nested fun/pair types, free type-vars | exact_dedP + Algorithm-W principal type |

**Verification (in `_verify`):** the cheap ising frustrated-triangle proxy is a SOUND ⊆ of the exact ground
frustration — a frustrated triangle (sign-product −1) can never be satisfied, so it forces
`ground_frustration > 0`. Checked on 60 random instances (29 frustrated witnessed, 0 violations). Every
metric runs on a live generator instance; the composition lattice round-trips.

---

## 2. CURRENT-COVERAGE AUDIT — does each faculty cluster easy? (YES, like CSP)

`run_audit(n_inst=150, seed=0)` over the LIVE generators (`faculty_tasks.gen_*` / the multifaculty
`build_live_pool` for csp). Numbers are the headline of this note.

### 2.1 csp — `build_live_pool` coloring/equality (determined)
```
required-level:  L0=100.0%  L1=0%  L2=0%  L3=0%      (level≥1: 0.0%)
treewidth:       median=1   tw≥4=0.0%   hist 0:3 1:88 2:57 3:2
prop-depth:      short(≤3)=74.7%  med(4-6)=25.3%  long(≥7)=0.0%
```
The multifaculty csp pool is EVEN easier than the curriculum-wide CSP corpus: 100% level-0, treewidth
median 1, no deep propagation. (It draws only determined coloring/equality.)

### 2.2 ising — `gen_ising_record` (max-cut over a rivalry graph)
```
frustration:       0=29.3%  lo(≤.15)=28.7%  mid(≤.35)=42.0%   [no hi(>.35) tail]
coupling_density:  sparse=4.0%  med=92.7%  dense=3.3%
ground_degeneracy: unique(≤2)=53.3%  few(≤4)=25.3%  many(≤16)=21.3%  [no glassy tail]
abs_J_spread:      uniform=100.0%
```
**Clusters easy on the spin-glass axis: `abs_J_spread` is 100% uniform** — every instance is a unit-weight
max-cut (J ∈ {0,−1}), so the SK spin-glass-hard regime (heterogeneous \|J\|, many local minima) is
**entirely unspanned**. Frustration is present (mid-range 42%) but has no hard tail (>0.35); density is
pinned at "med" by the fixed p_edge. The optimization is real but lives in the easy, uniform corner.

### 2.3 graph — `gen_graph_record` (directed reachability)
```
reach_depth:   1=56.7%  2=33.3%  3=8.7%  deep(≥4)=1.3%       [90% depth ≤2]
diameter:      1=16.0%  2=43.3%  3=29.3%  deep(≥4)=11.3%
reach_frac:    lo=27.3%  med=29.3%  hi=43.3%
edge_density:  sparse=20.7%  med=66.0%  dense=13.3%
```
**Clusters shallow: 90% of instances answer at reach-depth ≤2** — the certified min-plus closure barely
iterates, so the long-range-propagation capability (the whole point of an iterate-to-closure organ, and
of the R=12 round budget) is not exercised. Same shape as CSP prop-depth median 3.

### 2.4 type — `gen_type_record` (eq-tree) vs the faculty's own reach
```
gen_type_record (eq-tree):                  typeinfer.rand_problem (fun/pair universe):
  prop_depth:    med(3-4)=91.3% long≥5=8.7%   prop_depth:    short=10.7% med=50.7% long≥5=38.7%
  poly_breadth:  mono=100.0%                  poly_breadth:  mono=49.3%  poly=50.7%
  nesting_depth: base(0)=100.0%               nesting_depth: base=25.3%  depth1=74.7%
  treewidth:     1=100.0%                     treewidth:     1=4.7% 2=94.7% 3=0.7%
```
**The starkest gap. The live type generator is degenerate on three of four axes: 100% monomorphic, 100%
base-typed (nesting 0), 100% treewidth-1** — a flat eq-tree over {int,bool} with no fun/pair, no free
type-vars, no width. The faculty's OWN generator (`typeinfer.rand_problem`, depth-1 fun/pair universe)
shows what it can span: 75% nested types, 51% polymorphic, 39% deep propagation, treewidth-2 dominant. So
the multifaculty type curriculum leaves nearly all of the type faculty's intrinsic complexity unspanned.

**Verdict.** Every faculty's live generator clusters at the easy corner of its OWN intrinsic-hardness
space — the CSP coverage-hole, replicated per faculty: ising is 100% uniform-\|J\| (no glass), graph is
90% depth-≤2 (no long-range), type is 100% mono/base/tw-1 (no nesting/poly/width). The hard tail of each
faculty is under-occupied exactly as the CSP audit predicted.

---

## 3. Cross-faculty composition-occupancy framework

### 3.1 The composition space
A composite task is a CHAIN of faculties with FLOWS between them (`faculty.py`: cross = csp→graph,
tri = csp→graph→ising, reduction = a Karp route mis→ising). The space is combinatorial in
(which faculties × chain-order × hop-depth × reduction-on-path), so we define **principled buckets** rather
than enumerate the product. `CompositionBucket(chain, hop_depth, has_reduction)` is a point; `full_lattice`
enumerates every no-immediate-repeat ordered chain over the four direct-solver faculties (csp/graph/ising/
type) up to a hop-depth bound; the reduction-on-path axis is tracked as an orthogonal bit (a route can
substitute for any direct flow), not a Cartesian blow-up.

### 3.2 The buckets (what to occupy)
- **by hop-depth:** hop-0 (singletons) · hop-1 (directed pairs) · hop-2 (3-chains) · hop-3 (4-chains).
- **by faculty-set:** the unordered combination of faculties on the chain.
- **by reduction-on-path:** is a Karp reduction edge on the path (a route, not a direct flow)?

### 3.3 The audit (`cross_occupancy`, over the 7 live faculty kinds)
```
distinct ordered chains covered: 7 / 160 lattice  (4.4%)
by hop-depth (covered/total):  hop0 4/4 · hop1 2/12 · hop2 1/36 · hop3 0/108
covered chains: csp, graph, ising, type, csp→graph, graph→ising, csp→graph→ising
reduction-on-path share: 14%
```
**The composition lattice is 4.4% covered.** hop-0 is full (all four singletons), but hop-1 is 2/12, hop-2
is 1/36, and hop-3 (4-faculty chains) is **0/108 — entirely empty**. The covered chains are exactly the
hand-picked ones: the four singletons, the two flows inside cross (csp→graph) and reduction (graph→ising),
and the single 3-hop tri (csp→graph→ising). Everything involving the TYPE faculty as a flow endpoint, every
ising-or-type-first chain, and every 4-hop chain is missing. This is the cross-faculty restatement of the
coverage hole: the organ has a multi-faculty *fabric* but a curriculum that spans a 7-point slice of it.

### 3.4 OOD-split definitions (`OOD_SPLITS` — predicates over a CompositionBucket)
The cross-faculty OOD axis is the COMBINATION, not the size (`notes/multifaculty_overhaul §4.4`). Train on
the complement, evaluate on the held-out region:
- **held_out_faculty_combination** — a flow-pair never co-trained (e.g. any `type·*` pair, or a direct
  `graph·ising` pair). Tests whether emergent wiring composes a faculty-combination it never saw.
- **held_out_hop_depth** — train hop ≤ 2 (single / cross / tri), extrapolate to hop ≥ 3 (the 4-faculty
  chain, e.g. type→csp→graph→ising). One flow deeper than trained.
- **routing_trap** — the answer needs a Karp REDUCTION edge (no direct faculty) with the greedy
  single-faculty route a confident WRONG answer; train the direct route, hold out the reduction-routed one
  (sat3→CSP trained vs sat3→MIS→Ising held out).

`classify_ood` confirms the live tasks barely touch these regions (0/7 held-out-combo, 0/7 held-out-hop,
1/7 routing-trap = the reduction faculty) — the splits are clean and almost entirely unoccupied, i.e. the
expansion has the whole held-out region to fill on the train side and probe on the test side.

---

## 4. RANKED generator specs (gated on bank-completion)

The concrete difficulty-controlled generators the expansion should build (`GENERATOR_SPECS` in the module,
consumable as data). Ranked by leverage × cheapness; each gated on the relevant faculty/bank landing.

1. **`gen_ising_frustrated(target_frustration)`** — *axis: frustration; gated on the ising bank (live).*
   Plant a target Harary frustration index: seed a balanced (2-colourable) base, add k frustrating
   odd-cycle bonds, reject-sample on `ground_frustration`. Fills the missing hard-frustration tail.
2. **`gen_ising_glass(abs_J_spread, density)`** — *axis: \|J\|-spread × coupling-density; ising bank.*
   Draw SK-style \|J\| ~ LogNormal(spread) on a density-controlled graph → the spin-glass-hard regime
   (many local minima, high degeneracy). **Closes the 100%-uniform-\|J\| hole — the single biggest ising
   gap.**
3. **`gen_graph_depth(reach_depth=D)`** — *axis: reach-depth / diameter; graph faculty (live).*
   Build a layered DAG of depth D from the source (a reachability chain) + decoy off-path components, so
   the answer needs D closure rounds. Lifts reach-depth from 90%-≤2 to a uniform short/med/deep occupancy
   (and makes the R-round budget earn its keep).
4. **`gen_type_nested(nesting_depth=k, poly_breadth)`** — *axis: nesting × poly-breadth; type faculty.*
   Use `typeinfer.gen_typed` at universe max_depth ≥ 1 (fun/pair) so inferred types NEST, and tune the
   free-tvar count for poly-breadth. Replaces the base-only eq-tree generator (nesting 0, 100% mono) —
   spans the type universe's actual richness (the §2.4 contrast shows the headroom).
5. **`gen_chain(faculty_chain, hop_depth)`** — *axis: composition hop-depth / faculty-combo; bank-completion
   (ALL faculties).* A generic cross-faculty chain factory parametric in the ordered faculty chain (the
   `multifaculty_overhaul §4.3` design): each stage's solution determines the next stage's structure. Fills
   the empty hop-1/hop-2 buckets beyond the two hand-picked (cross, tri) and ENABLES the
   held-out-combination + held-out-hop OOD splits.
6. **`gen_karp_route(src_type, held_out_edge)`** — *axis: routing-trap; reductions hub
   (`organ/reductions.ALL_EDGES`).* Emit instances whose only low-cost route crosses a held-out Karp edge
   (sat3→MIS→Ising vs the trained sat3→CSP); the greedy single-faculty answer is a confident wrong. Trains
   the router + flow gates on routing, exercising the routing-trap split.

**Gating note.** Specs 1-4 are per-faculty and only need the relevant faculty live (all are). Specs 5-6 are
cross-faculty and gated on the bank-completion landing (the concurrent build of `faculty_tasks.py` +
`multifaculty.py`) — they own the generator surface, so this note ships the METRICS + FRAMEWORK + AUDIT
those generators target, not the generators themselves.

---

## 5. TL;DR
Each faculty's live generator clusters at the easy corner of its own intrinsic-hardness space — ising is
100% uniform-\|J\| (no spin-glass), graph is 90% reach-depth-≤2 (no long-range), type is 100% mono/base/
treewidth-1 (no nesting/polymorphism/width), csp is 100% level-0 — the CSP coverage hole replicated per
faculty. The cross-faculty composition lattice is 4.4% covered (hop-3 entirely empty; only csp→graph,
graph→ising, csp→graph→ising among the flows). `faculty_difficulty.py` makes all of this measurable
(per-faculty `difficulty` + `profile_faculty`, the `CompositionBucket` lattice + `cross_occupancy`,
`OOD_SPLITS`) so the curriculum expansion can BUCKET-TARGET each faculty's hard tail and the held-out
composition regions, with the six ranked generators (frustrated/glassy ising, deep graph, nested/poly type,
the generic cross-faculty chain, the Karp routing-trap) the build steps that fill them.
