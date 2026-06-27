# Curriculum design — auditing task breadth + difficulty occupancy, and the difficulty-controlled fix

**Goal.** Make the GLaDOS pretrain corpus ROBUSTLY occupy the difficulty space across diverse kinds,
aligned to the five standing learnings (A-vs-B diversity→OOD-soundness + the UNION-relations fix;
composition-coverage ≈30% with *need-grows-with-scale*, 2507.07207; the affine-wall / treewidth =
required-lattice-level law; the recipe sweep 1.5M/R12/30%-composed; the 60-pt α-on-hard gap; the
reduction graph = faculties × reductions). The crux finding: **the generators occupy *size* (n) well
but leave the genuinely-hard tail — high treewidth, richer-than-per-cell required level, deep
propagation — almost entirely in the held-out / OOD splits, so the organ pretrains where per-cell AC
already solves everything and is evaluated where it doesn't.** That under-occupied hard tail is the
α-gap, restated as a *coverage* gap.

Audit code: `scratchpad/profile_difficulty.py` + `profile_extra.py` (treewidth via min-fill UB on the
primal graph; required-level via AC-fixpoint-vs-exact-dedₚ + the `levels.polymorphism_kind` diagnostic;
propagation depth = AC iterations to fixpoint; solution count capped at 64). N=150–200 samples/kind.

---

## 1. AUDIT — what we have

### 1.1 Kinds (task families)

**CSP-narrowing spine** (the per-cell deduction organ, `clair.curriculum` + `hard_tasks`):
coloring · equality · ordering · arithmetic (modular sum) · alldiff · `gen_xor` (arity-3 affine) ·
`gen_parity_k` (arity-3..5 affine) · `eqchain` / `forcedcolor` (the hard propagation chains).

**Broadened non-CSP domains** (`datagen/extra.py`, each with its own exact oracle): xor/GF(2) ·
graph (dist/reach/connect) · perm · typeinfer (HM) · fol (forward-chaining) · optimize (mincost/maxcut).

**Reductions** (`datagen/reductions.py` + `organ/reductions.py`): sat3 / sat2 / xorsat → CSP
(certified edges), with `route_required` (xorsat, no direct faculty) and `heldout_reduction_path`
splits. The full Karp neighbourhood (`organ/reductions.ALL_EDGES`) also ships MIS↔VC↔clique, sat3→MIS,
and {MIS,maxcut,partition,coloring}→Ising — but **only the sat*→CSP edges are wired into the corpus**.

**Compositions** — two *separate* mechanisms, easy to conflate:
- on-disk LM corpus (`build.py`): a text-chain `head→integer R→(R+c) mod m`, head ∈
  {graph,fol,perm,typeinfer}, **always an arithmetic modular tail**; OOD heads {xor,ordering} + a
  graph+perm triple. ≈10% of train (6k of ~61k).
- organ pretrain (`compose_csp.py`, used by `sweep_organ`/`run_glados_staged`): a CSP combining two of
  {neq,eq,order,sum,alldiff} over a shared witness; 7/10 pairs trained, 3 held out. ≈30% mix.

**Missing kinds** (per `organ_excellence §1`: only ~12% of reasoning-gym is even organ-expressible, and
`abstract_machines`/`future_directions §3`): planning/STRIPS, CLRS graph algorithms beyond
SSSP/reach, **program-induction / ∂4-procedural** (sort, running aggregate, carry-arithmetic),
abduction, constraint-OPTIMIZATION beyond maxcut/mincost, **rule-induction Datalog** (we ship the
*certified* `unification_chain` but no rule-proposer task). These are catalog-level breadth, lower
priority than the difficulty-occupancy fix below.

### 1.2 Variations

- **Relation-family diversity (the A-vs-B / UNION learning).** The UNION fix (named families + random
  allowed-tuple tables, *additive*) lives in `general_relation_organ.py` for the organ trainer, **but the
  on-disk corpus generators emit only NAMED relation families** (eq/neq/lt/le/sum/xor/alldiff). There is
  no random-allowed-tuple generator in `curriculum.py`/`extra.py`. So the corpus does not exercise the
  novel-relation axis that the learning says is the OOD-soundness lever — `ood_relation` is parity_k +
  hard chains (still *named* structure), not a genuinely unseen relation table.
- **Surface/render diversity.** Strong and well-covered: skins × naming-schemes × structure
  (prose/semicolon/bullets/numbered) × per-instance FOL synonym maps; self-BLEU ≈0.11, dedup gate. Not
  a gap.
- **Composition coverage.** CSP skill-pairs 7/10 (good). LM text-chain compositions: only
  `head + modular-arith tail` — the tail is *always* the same operation, and no two *broadened* domains
  ever compose directly (no graph∘fol, perm∘xor, …). Shallow (2-stage) and narrow. 2507.07207 says the
  need grows with scale; current coverage is ~10% and one-shaped.
- **Reduction-path diversity.** Only sat→CSP is in the corpus; the MIS/VC/clique/Ising hub is built and
  verified but unused as training records. `route_required` is a single family (xorsat).

### 1.3 Difficulty occupancy — THE KEY AUDIT (measured)

Per-instance difficulty profiled on 4 axes. Required-level uses the polymorphism law from `levels.py` /
`csp.py`: per-cell AC is complete iff the instance is semilattice/bounded-width; **affine + treewidth w
needs lattice level w+1** (the affine wall). "L0" = AC-fixpoint equals exact-dedₚ (per-cell suffices).

| kind (split) | n (median) | treewidth (median / p90 / max) | required-level (% L0) | prop-depth (median / p90) |
|---|---|---|---|---|
| **TRAIN** coloring | 6 | 2 / 3 / 4 | **95%** | 3 / 4 |
| **TRAIN** equality | 5 | 2 / 2 / 4 | **100%** | 3 / 4 |
| **TRAIN** ordering | 6 | 2 / 3 / 4 | **100%** | 3 / 4 |
| **TRAIN** arithmetic | 5 | 2 / 2 / 3 | **98%** | 3 / 5 |
| **TRAIN** alldiff | 5 | 4 / 4 / 4 | **100%** | 3 / 3 |
| **TRAIN CSP aggregate (1000)** | 5 | **2** / 3 / 4 | **98.7%** | 3 / 4 |
| ood_n coloring (n8-11) | 9 | 3 / 5 / 6 | 90% | 4 / 5 |
| ood_n ordering (n8-11) | 9 | 4 / 6 / 7 | 100% | 4 / 5 |
| ood_n arithmetic (n8-11) | 9 | 2 / 3 / 4 | 96% | 4 / 6 |
| **ood_relation parity_k** | 6 | 4 / 4 / 5 | **57%** | 2 / 5 |
| **ood_relation eqchain** | 9 | 1 / 1 / 1 | 100% | 5 / **9** |
| **ood_relation forcedcolor** | 6 | 2 / 2 / 2 | **65%** | 5 / **9** |
| reduction Y=CSP ⟵ sat3 | 6 | 4 / 5 / 5 | 87% | 2 / 3 |
| reduction Y=CSP ⟵ xorsat | 6 | 4 / 5 / 6 | **81%** | 3 / 5 |
| compose_csp single | 5 | 1 / 2 / 4 | 96% | 3 / 3 |
| compose_csp pair | 5 | 2 / 3 / 4 | 95% | 3 / 4 |

**The four reads (this is the crux, quantified):**

1. **n is the only well-spread axis** — uniform within each split's band (4-7 train, 8-11 ood_n,
   5-12 chains). It is sampled `rng.integers(n_lo, n_hi+1)`. Good.

2. **treewidth clusters LOW in train (median 2) with a sparse, uncontrolled hard tail.** Across the
   1000 train-CSP instances: tw≤1 21%, tw=2 45%, tw=3 22%, tw=4 11% — and the tw=4 mass is almost all
   **alldiff**, whose primal graph is the complete K_n (tw=n-1) yet is GAC-easy (L0=100%), so high
   treewidth there is *not* hard. Excluding alldiff, tw≥3 is only ~17% and tw≥5 is **0**. The genuinely
   high-treewidth region (tw 4-7) lives in the **held-out ood_n split**, not train. Treewidth is an
   emergent byproduct of (n × fixed density), never a target.

3. **Required-lattice-level is the SEVEREST gap. TRAIN is 98.7% level-0** — per-cell AC already reaches
   the exact-dedₚ fixpoint, so the organ never has to use anything richer than per-cell narrowing. The
   level-1+/affine occupancy — the *actual* hardness the affine-wall law cares about — is **1.3% in
   train** but **43% in parity_k, 35% in forcedcolor, 19% in xorsat-reduction, 12-13% in sat reductions**
   — i.e. ~30× denser in the held-out/OOD splits than in train. Worse: `gen_xor` (the pure arity-3
   affine rung) is in `curriculum.GENERATORS` **but omitted from `build.TRAIN_FAMILIES`**, so the
   on-disk train CSP corpus's only affine exposure is `arithmetic`, which as generated is 98% L0. **The
   organ pretrains on a distribution where per-cell AC solves everything, then is evaluated where it
   provably can't.** This is the 60-pt α-on-hard gap, mechanically: not a capacity limit, a coverage
   hole.

4. **Propagation depth is shallow in train (median 3, p90 4).** The deep-chain tail (depth 6-9) exists
   ONLY in `eqchain`/`forcedcolor` — which are `ood_relation`, never trained. So long-range
   propagation, the thing the organ's iterate-to-fixpoint is *for*, is also held out.

**Verdict:** three of the four difficulty axes (treewidth, required-level, propagation-depth) have their
hard tail confined to held-out splits. The pretrain distribution is clustered at easy. Hypothesis from
the α-gap — *the hard tail is under-occupied* — is **confirmed: train is 98.7% level-0, ≤17% tw≥3
(non-alldiff), median prop-depth 3.**

---

## 2. GAPS vs each learning

- **A-vs-B diversity → OOD-soundness + the UNION fix.** Corpus emits only NAMED relations; no
  random-allowed-tuple generator. The *additive* union (named ⊕ random tables) that made the organ sound
  on novel relations (`general_relation_organ`) is **not in the on-disk pipeline**. → add a
  random-relation generator and make it additive in train.
- **Composition-coverage ≈30%, need-grows-with-scale (2507.07207).** On-disk LM corpus is ~10% and
  one-shaped (always arith tail; no broadened∘broadened pairs). compose_csp is 30%/7-of-10 but is the
  *organ-only* path. → raise LM-corpus composition to ~30%, diversify tails + add broadened-domain pairs
  and 3-chains, and **scale the composed fraction with model size** (the need grows).
- **Affine-wall / treewidth = required-lattice-level.** We *have* the diagnostic but **do not use it to
  shape the distribution**: train is 98.7% level-0. The law says required-level = treewidth+1 for affine;
  we never deliberately fill level 2,3,4. → make required-level a generation TARGET (below).
- **Recipe sweep (1.5M / R=12 / 30%-composed).** R=12 buys depth for ~p90 propagation 5; but train's
  propagation depth is median 3, so the organ is not being *asked* to use its depth. The depth budget is
  provisioned for a difficulty the train data doesn't contain. → populate the deep-propagation tail in
  train (chains for every family, not just coloring), so R=12 earns its keep.
- **60-pt α-on-hard gap.** This *is* the difficulty-occupancy hole. Fixing the coverage of high-level /
  high-treewidth / deep-propagation instances in TRAIN is the α-gap fix as a coverage fix.
- **Reduction graph = faculties × reductions.** Only sat→CSP wired; MIS/VC/clique/Ising hub unused as
  records; single route_required family. → wire the rest of `ALL_EDGES` into records and add more
  route_required held-outs (VC, clique, partition via the Ising hub).

---

## 3. DESIGN — difficulty-controlled generation

### 3.1 The difficulty metric (per kind, all already computable, no GPU)

A difficulty vector `(n, treewidth, required_level, prop_depth, n_solutions)`:
- **treewidth** — min-fill UB on the primal graph (`profile_difficulty.treewidth_minfill`, exact enough
  at n≤12).
- **required_level** — the cheapest lattice level that reaches exact-dedₚ: 0 if `ac_step` fixpoint ==
  `exact_dedP`; else 1 if `solve_pair` solves; else 2 if `solve_factor(.,3)`; gated by
  `polymorphism_kind` (affine+no-majority ⇒ ≥ treewidth+1). All in `csp.py`/`levels.py` today.
- **prop_depth** — AC iterations to fixpoint (`to_fixpoint` already returns the step count).
- **n_solutions** — `solutions(csp, limit=k)` for the determined/abstain balance.

### 3.2 A knob per kind that TARGETS a difficulty bucket

The principle: **generate to a target, don't sample-and-hope.** For each kind expose
`target=(level L, treewidth w, depth D)` and construct an instance that hits it, rejection-sampling on
the cheap metric to confirm. Concretely:

- **Required-level L via polymorphism × width.** required-level is set by the relation's polymorphism
  CLASS combined with treewidth. To fill level L: emit an **affine** relation (xor/parity, or modular sum
  over the right modulus) over a constraint graph of **treewidth L−1** (the law). We already have both
  primitives — `gen_xor`/`gen_parity_k` (affine relations) and `xor_wall.gen_xor_system(..., band=w)`
  (the **band parameter IS bandwidth ≈ treewidth** — directly a width knob). Cross them: a
  `gen_affine(level=L)` = parity system at band w=L−1. Verify with `polymorphism_kind` + AC-vs-exact.
  Do the same for majority/bounded-width to fill the *between* levels.

- **Treewidth w directly.** Build the primal graph from a **bag-chain (path/tree decomposition of bag
  size w+1)**: lay cells in bags of size w+1, connect within and across adjacent bags, then drop the
  family's relations onto those edges. Gives exact target pathwidth/treewidth ≤ w for *any* family
  (coloring/ordering/alldiff), decoupling treewidth from n. (Today treewidth only rises by cranking n,
  which is why the tail is in ood_n.)

- **Propagation depth D.** Generalize `hard_tasks`' chain construction (currently coloring-only) to
  every family: plant the determined query at graph-distance D from the pins, with a differently-valued
  decoy component (the anti-shortcut already in `gen_eqchain`). A `chain(family, L)` factory parametric
  in family gives a depth knob across the whole spine.

- **n, d, solution-count** — already knobbed (n_lo/n_hi, d range, pin_frac → determinacy). Keep.

### 3.3 Recommended pretrain breadth / mix / splits

**Fill each bucket, don't cluster.** Target an approximately UNIFORM occupancy over the difficulty
grid `level ∈ {0,1,2,3} × treewidth ∈ {1,2,3,4,5} × depth ∈ {short,med,long}`, per kind, with the
hard tail EXPLICITLY populated. Concrete mix for the CSP-narrowing spine (the organ's training set):

- **By required-level:** ~40% level-0, ~25% level-1 (majority/bounded-width), ~25% level-2 (affine,
  arity-3), ~10% level-≥3 (affine at higher band / arity). (Today: 98.7% / ~1% / ~0 / 0.) This is the
  single highest-leverage shift — it moves the affine/richer-level mass from OOD into train.
- **By treewidth:** roughly flat over tw 1-5 (decoupled from n via the bag-chain), with a deliberate
  tw≥4 tail (~15-20%). (Today: median 2, tw≥5 ≈ 0 in train.)
- **By propagation depth:** ~⅓ short (≤3), ~⅓ medium (4-6), ~⅓ long (7-10), every family. (Today:
  median 3, long tail only in held-out chains.)
- **Relation diversity:** ADD random-allowed-tuple relations *additively* (named ⊕ random ≈ 70/30), per
  the UNION fix — needed for OOD-soundness on novel relations.
- **Composition:** ~30% composed (match the sweep), diversified: keep the modular tail but add
  min/max/compare tails and **broadened∘broadened** pairs + 3-chains; scale the fraction up with base
  size (need-grows-with-scale).
- **Reductions:** wire the full Karp neighbourhood as records (~10-15% of train), multi-path where edges
  exist (sat3→CSP vs sat3→MIS→Ising), so routing is trained not just held out.

**Standing OOD splits (keep clean, but now they test EXTRAPOLATION past a trained tail, not a
never-seen regime):**
- **novel-relation** — random allowed-tuple tables from a family disjoint from train's random pool.
- **novel-composition** — held-out skill-pairs / domain-pairs (compose_csp's 3 OOD pairs + broadened
  pairs never co-seen).
- **route-required** — a source type with no direct faculty (xorsat today; add VC, clique, partition via
  the Ising hub).
- **hard-N + hard-LEVEL + hard-DEPTH** — extrapolate each difficulty knob one bucket past the trained
  max (n beyond train band; required-level beyond the trained max band; propagation depth beyond trained
  max). Splitting the single "hard" axis into its three independent difficulty knobs is itself a
  contribution: it tells us *which* kind of hardness fails to extrapolate.

---

## 4. Ranked concrete generator additions

Ranked by leverage × cheapness (cheap = reuses shipped code).

1. **`gen_affine(level=L)` / difficulty-targeted affine generator — fills the required-level tail.**
   Cross `gen_parity_k` (affine relations) with `xor_wall.gen_xor_system(band=w)` (band ≈ treewidth) →
   instances at a TARGET level L = w+1, verified by `polymorphism_kind`. **Cheapest high-leverage move**
   (both primitives ship). Directly closes the 98.7%-L0 hole — the α-gap fix. Also: **add `gen_xor` back
   into `build.TRAIN_FAMILIES`** (a one-line omission today).
2. **Bag-chain treewidth generator — decouples treewidth from n.** A primal-graph constructor of target
   pathwidth w+1 that any family's relations drop onto; lets us fill tw 1-5 in train without going wide.
   ~30 lines; makes treewidth a knob instead of an n-byproduct.
3. **`chain(family, L)` — generalize the propagation-depth chains to every family.** `hard_tasks` already
   does this for coloring (eqchain/forcedcolor); lift the decoy + distance-L-query construction to a
   family-parametric factory. Fills the deep-propagation tail (median 3 → uniform to ~9) and makes R=12
   earn its depth. Cheap (refactor of existing chain code).
4. **Random-allowed-tuple relation generator, ADDITIVE — the UNION fix in the corpus.** A
   `gen_random_relation(arity, d)` emitting a random satisfiable extensional constraint, mixed 30%
   alongside named families (per `general_relation_organ`). Restores OOD-novel-relation soundness; gives
   a true novel-relation OOD split. Moderate (witness-first sampler over random tables).
5. **Diversify + deepen composition.** Add non-arith tails (min/max/compare) and broadened∘broadened
   pairs + 3-chains to `extra.gen_composition`; raise the LM-corpus composed fraction to ~30% and scale
   with base size. Moderate (extends an existing generator).
6. **Wire the rest of the reduction graph into records.** MIS↔VC↔clique, {MIS,maxcut,partition}→Ising
   are built + verified in `organ/reductions.py`; emit them as `reductions.py` records with multi-path
   routing and add VC/clique/partition `route_required` held-outs. Cheap (the edges + checkers exist).
7. **(catalog breadth, lower priority)** ∂4-procedural and rule-induction-Datalog faculties +
   generators, per `abstract_machines` — genuinely new capability classes, but build the difficulty
   occupancy of the *existing* kinds first.

---

## 5. TL;DR

The generators occupy *size* (n) uniformly but cluster at *easy* on every intrinsic-hardness axis:
**train is 98.7% per-cell-level-0, treewidth median 2 (tw≥5 ≈ 0 outside GAC-easy alldiff), propagation
depth median 3** — while the level-1+/affine, high-treewidth, and deep-propagation mass lives almost
entirely in the held-out splits (parity_k 43% L1+, forcedcolor 35%, chains depth-9). The 60-pt α-on-hard
gap is this coverage hole. Fix = make required-level / treewidth / propagation-depth explicit generation
TARGETS (all three knobs build cheaply on shipped primitives — affine relations × the `band` width param;
a bag-chain graph; family-parametric chains) and fill each difficulty bucket uniformly in TRAIN, moving
the hard tail out of OOD-only and into the training distribution, with OOD redefined as one-bucket
extrapolation past each trained max.
