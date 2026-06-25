# Geometric Product Options for Clairnets

Date: 2026-06-25

This note corrects a narrowing error in the early framing. The Clifford/GA bet should not be
"wedge usefulness" as the main thesis. Wedge is one interpretable component and one useful
ablation. The broader bet is:

> A geometric-product interaction can be a compact learned proposer, mixer, compiler, or fusion
> operator for constraint reasoning, while lattice projection/checking remains the authority for
> reliable deduction.

In CliffordNet terms, the primitive is not just `u ^ v`; it is the full product:

```text
uv = u . v + u ^ v
F(H, C) = P( H . C  concat  H ^ C )
```

The inner component carries coherence, energy, and gating. The wedge component carries
antisymmetric structure, orientation, and contrast. CliffordNet's own ablation says inner-only and
wedge-only are both strong, but the full product is best. So the right question is not "does wedge
solve reasoning?" It is "where does full geometric-product mixing buy the most?"

## What Reliability Can and Cannot Come From

The neural geometric product should not be trusted as a proof system. It should propose.

Reliability has to come from one of these boundaries:

- output verification: return an answer only if it satisfies the constraints;
- lattice meet/projection: the state only narrows, never expands, and eliminations are measured;
- certificate checking: a removed candidate carries a checkable witness that no solution consistent
  with the current state uses it;
- exact alpha targets on small CSPs: train against the best sound abstraction where it is computable.

The Lean LDT files make this distinction sharper than the PyTorch notes did. Soundness is cheap if
outputs or steps are checked. Completeness, meaning whether the organ reaches a checkable answer
instead of abstaining, is the real research variable.

## Candidate Integration Points

### 1. Full Clifford proposer inside LDT

Replace the LDT channel FFN with a full geometric-product mixer:

```text
context c = attention_or_graph_context(h)
proposal = P( dot(h, c) concat wedge(h, c) )
```

Controls:

- `ffn`: baseline LDT/SwiGLU;
- `inner`: dot/coherence only;
- `wedge`: bivector/antisymmetric only;
- `full`: dot plus wedge;
- `full_diff`: dot plus wedge with differential context;
- `grade3`: add a triple interaction for parity/affine-like constraints.

This is the closest "CliffordNet in the organ" experiment.

### 2. CliffordNet-faithful context, not just a small wedge toy

CliffordNet's gains are tied to a package:

- local context `C_loc(H)`;
- self-energy suppression / differential mode `C = C_loc(H) - H`;
- sparse shifted geometric products over selected channel offsets;
- gated geometric residual;
- optional global context path.

For an LDT organ, analogues are:

```text
C_attn = attention_or_graph_context(H)
C_diff = C_attn - H
H_geo  = P( shifted_dot(H, C_diff) concat shifted_wedge(H, C_diff) )
H_next = H + gamma * gate(H, H_geo) * H_geo
```

Hypothesis: the differential context and gating may matter more than wedge alone.

### 3. Geometric attention or compatibility

Standard attention compresses token interaction to scalar `q . k`. CliffordNet explicitly criticizes
scalar-only interaction as lossy. A geometric attention experiment could expose both scalar and
bivector compatibility:

```text
score/value_features = P( q . k concat q ^ k )
```

This is higher risk than replacing the FFN/proposer because attention has normalization and routing
semantics. Treat it as a second wave, not the first implementation target.

### 4. Clifford cortex as constraint compiler

For the full augmented LLM, the geometric product may be more natural in the host/cortex than in
the deductor. The cortex reads text/vision tokens and writes a constraint program:

- candidate universe;
- givens/evidence;
- exclusion edges;
- clauses or higher-arity factors;
- confidence/calibration metadata.

Here wedge has a clean role: pairwise incompatibility or oriented relation proposal. But the deductor
still checks, projects, or abstains.

### 5. Higher-grade products for higher-order constraints

Pairwise wedge is naturally grade-2. But the Lean completeness examples show that per-cell and
pairwise views hit an affine/parity wall. XOR-like structure is not just pairwise exclusion.

Possible probes:

- grade-3 scalar triple product in the proposer for 3-SAT, XOR, and parity;
- vector times bivector interactions for rule composition;
- product lattices that carry pair or triple candidate sets, not only per-cell candidates.

This should be tied to tiny exact CSPs first, not immediately to Sudoku.

### 6. Automata shortcuts for deduction performance

The automata-shortcuts paper is directly relevant to performance, not just theory. A recurrent
deduction loop on a finite lattice is a finite-state dynamical system once the problem instance and
thresholds are fixed:

```text
a_{t+1} = F(a_t)
```

Naively applying `F` for `T` rounds costs `O(T)` sequential steps. The paper's lens says to think in
terms of transformations, not just states:

```text
F^1, F^2, F^4, F^8, ...
```

Then learn or distill macro-deduction operators that approximate composition:

```text
F_2(a)  ~= project(F_1(F_1(a)))
F_4(a)  ~= project(F_2(F_2(a)))
F_8(a)  ~= project(F_4(F_4(a)))
```

Possible uses:

- `O(log T)` inference: run macro steps instead of many one-step recurrent applications.
- Shortcutability diagnostic: if `F_2 ~= F_1 o F_1` is easy to learn and stays sound/checked, the
  domain has compressible deduction dynamics; if not, the task may require real sequential search.
- Training regularizer: randomize unroll depth and require consistency between one-step and macro
  steps.
- Cacheable operator library: train a small family of macro organs, then choose depth adaptively.
- Maze/path tasks: the paper's gridworld result suggests boundary/nearest-obstacle style shortcuts
  may be especially efficient with attention.

There are two levels of this idea, and they should not be conflated:

```text
state macro:
  M_k(problem, a) ~= F_problem^{2^k}(a)

transformation macro:
  T_k(problem, z_operator) ~= compose(T_{k-1}, T_{k-1})
```

The state macro is the near-term implementation target. It takes a lattice state and proposes a
larger narrowing. The transformation macro is closer to the automata theorem: represent the local
transition function itself and compose transition maps in latent space. For LDT this means the
organ should maybe learn an instance-conditioned deduction operator, not only a state update.

Automata-paper motifs worth porting:

- Binary scan/composition: compose short transition maps into long prefix effects.
- Modular counters: learn prefix-sum-like summaries for parity, reachability, and repeated rule
  application.
- Last-write memory: retrieve the most recent relevant event, such as a branch, contradiction,
  boundary hit, or last candidate-support witness.
- Cascade wiring: split the reasoning system into small components with dependency functions
  between them, rather than expecting one monolithic transformer block to discover the whole
  semigroup.
- Gridworld shortcut: for path domains, detect the latest boundary/reset and then compute local
  displacement from that boundary. This is a concrete clue for maze/path tasks.

Runtime policy should be speculative and checked:

```text
try M_4 or M_5
if checked safe, accept
else try M_3, M_2, M_1
else use the base one-step organ/search
```

Important caution: the paper also finds that shortcut solutions can be statistically brittle. They may
generalize in-distribution but fail on shifted counts, lengths, or rare latent variables. For deduction,
that means macro steps should be output-checked or certificate-checked, and evaluations must include
OOD lattice depths, harder branch states, longer chains, and changed puzzle distributions.

### 7. Geometric multimodal fusion

CliffordNet explicitly suggests geometric multimodal fusion: instead of only aligning modalities by
scalar similarity, compose them into cross-modal bivectors. For GLaDOS, this could mean:

```text
image/text hidden state H
constraint workspace context C
fusion = P(H . C concat H ^ C)
```

This is relevant for a later ARC/visual-symbolic stage, especially where grounding is the hard part.

## Concrete Organ Sketch

The LDT organ should be treated as a checked proposal engine:

```text
inputs:
  problem program P      # constraints, topology, factors, givens, optional perception evidence
  lattice state a        # alive candidates or richer pair/triple/factor candidates
  optional latent trace z

neural proposal:
  H0 = embed(P, a, z)
  H1 = recurrent_transformer_or_graph_transformer(H0)
  H2 = clifford_block(H1, context(P, H1))
  proposal logits = read_candidates(H2)
  conflict logits = read_conflict(H2)
  optional certificate logits = read_witnesses(H2)

checked projection:
  a_prop = a meet threshold(proposal logits)
  accept deletions only if they pass exact alpha, verifier, or certificate checks
  otherwise keep a, branch, or abstain
```

For the current codebase, the first concrete target is not "replace every dot product." It is:

```text
existing LDT/GLaDOS block
  -> swap FFN/proposer with CliffordNet-faithful geometric block
  -> keep lattice meet and false-elimination metrics unchanged
  -> add exact finite-CSP targets before trusting multi-domain results
```

The Clifford block should carry separate ablation switches:

```text
context:   attention | factor-graph message passing | local conv | global mean | hybrid
mode:      inner | wedge | full
self:      absolute | differential C-H
topology:  shared shifts | separate dot/wedge shifts | learned sparse shifts
gating:    off | scalar gate | candidate-wise gate
grade:     vector-vector | vector-bivector | grade-3 probe
```

The factor-graph version is probably more important than a pure grid/sequence version for the
multi-domain gauntlet. The problem instance has to be visible to the organ. A SAT instance without
clauses, a graph-coloring instance without edges, or a maze without walls is underdetermined.

## Immediate Experiment Ladder

1. Build exact finite-CSP harness.
   - Small chain, equality, XOR, 2-SAT, 3-SAT, graph-coloring.
   - Compute exact alpha targets by enumerating satisfying assignments.
   - Verify outputs directly.
   - Measure completeness and abstention, not just false elimination against one planted solution.

2. Add a CliffordNet-faithful LDT block.
   - Full product, inner-only, wedge-only, differential context, gated residual.
   - Run on tiny CSPs first, then Sudoku.

3. Fix the multi-domain gauntlet interface.
   - Current gauntlet does not feed instance-specific edges, clauses, or walls to the model.
   - Add a constraint-program input or factor graph representation.
   - Otherwise SAT/coloring/maze are underdetermined from `alive, given` alone.
   - For multi-solution domains, replace planted-solution alpha with exact or sampled alpha.

4. Compare abstraction levels.
   - Per-cell lattice.
   - Pair lattice for equality/path-consistency atoms.
   - Triple/factor lattice for parity and clauses.
   - Use the Lean "kind checker" results as the guide.

5. Add automata-style macro deduction.
   - Distill `F_2`, `F_4`, `F_8` from the checked one-step operator.
   - Track lifting error and checked false-elimination rate.
   - Compare sequential `T` one-step rounds vs logarithmic macro rounds.
   - Try both state macros and latent transformation-map macros.
   - Use speculative fallback from large macro to small macro to base step.

6. Only after that, try the LLM graft.
   - Host writes a constraint program.
   - Organ deduces/checks/abstains.
   - Host reads back the narrowed state.

## Training Regime for Stabilization and Generalization

The goal is not only low loss. The goal is to make the learned step behave like a stable, composable
abstract transformer:

```text
same problem + same lattice state -> same safe narrowing
more rounds -> monotone progress or stable fixed point
changed distribution -> abstain or stay conservative, not hallucinate eliminations
```

Recommended pressures:

- On-policy states: train on states produced by the current model's own meet/branch loop, not only
  clean initial states.
- Random unroll depth: sample the number of inner/recurrent steps during training so the operator
  cannot depend on a fixed clock.
- Deep supervision: every recurrent step predicts candidate survival and conflict, not just the final
  step.
- Monotonicity by construction: apply `alive_next = alive meet proposal`; never let the network
  directly add candidates.
- Exact alpha targets on small domains: where possible, enumerate all remaining solutions and train
  against the true best abstraction.
- Multi-solution alpha: for maze/SAT/coloring, keep every value used by some surviving solution;
  do not train against one arbitrary planted solution unless intentionally measuring a stricter proxy.
- False-elimination-weighted loss: keep asymmetric BCE or an even more direct penalty on deleting
  candidates that exact alpha would keep.
- Conflict calibration: train conflict separately from candidate elimination; over-eager conflict
  causes useless abstention, under-eager conflict causes wrong returns.
- Fixed-point consistency: if `a` is already a fixed point under exact deduction, train the organ to
  leave it unchanged.
- Composition consistency: train `F_2(a)` to match `F_1(F_1(a))`, `F_4(a)` to match
  `F_2(F_2(a))`, etc., with checks after each macro step.
- State-depth coverage: deliberately sample shallow, medium, deep, branched, near-conflict, and
  adversarial lattice states.
- Distribution shift tests during training: hold out puzzle sizes, graph densities, clause densities,
  maze lengths, and branch depths. Select checkpoints by held-out completeness and checked
  soundness, not by train loss alone.
- Symmetry augmentation at the dataset level: useful for Sudoku-like domains, but per-step
  augmentation should be tested carefully because it can diffuse the on-policy trajectory.
- Conservative threshold schedules: start with high retention / low elimination pressure, then anneal
  toward useful narrowing after the conflict head and alpha behavior are calibrated.
- Abstention as a valid outcome: reward "no unsafe progress" on states outside the learned class.

Stability diagnostics to log:

- false-elimination rate against exact alpha;
- checked wrong-return rate;
- abstain rate and reason: conflict, timeout, verifier rejection;
- mean and p90 alive candidates per position over rounds;
- fixed-point drift: how much the organ changes exact fixed points;
- composition/lifting error for `F_2`, `F_4`, `F_8`;
- contraction proxy: distance between two nearby lattice states under repeated organ steps;
- OOD completeness: solve rate on held-out kinds, sizes, densities, and depths.

This regime is where the automata paper becomes practical. We want shortcuts, but not brittle ones.
So the model can learn macro transitions only under a verifier/meet/check boundary, with OOD tests
designed to catch the exact shortcut failures the paper reports.

## Exploiting Shortcuts and Instability Instead of Only Suppressing Them

Some shortcut behavior may be useful if it is sandboxed. The split should be:

```text
unsafe if trusted directly
useful if treated as proposal/search/compression under checks
```

Ways to exploit it:

- Macro proposals: let shortcut organs propose large jumps in the lattice, then meet/check them.
- Speculative deduction: run a cheap `F_8` or `F_16` first; if verification fails, fall back to smaller
  `F_4`, `F_2`, then `F_1`.
- Multi-chain diversity: different shortcut modes may solve different branches. Treat them like
  stochastic search proposals, not one canonical proof trace.
- Abstain-aware routing: if a macro operator is uncertain, route to the slower recurrent solver.
- Shortcut ensembles: train multiple macro operators with different random unrolls, shifts, depths,
  or geometric topologies; accept only checked consensus or verified outputs.
- Lifting error as a scheduler: low `F_2 ~= F_1 o F_1` error means use larger jumps; high error means
  stay local and search.
- Edge-of-stability as exploration: if a checkpoint has lively but bounded trajectory dynamics, it may
  generate useful branch diversity. Keep it only if checked soundness stays intact.
- Geometric-product diversity: inner, wedge, and higher-grade paths may create different proposals;
  use the projection/check boundary to select safe progress rather than forcing one path to dominate.

This suggests a practical runtime policy:

```text
try fast shortcut proposal
check monotone narrowing and verifier/certificate conditions
accept if safe
otherwise shrink step size or branch
```

So the goal is not "make the neural part perfectly reliable." The goal is "make unreliable but
creative proposal dynamics cheap, diverse, and safely filterable."

## PDF Triage

### Core

- `pdfs/2601.06793-cliffordnet-geometric-product.pdf`
  - Most important for the geometric side.
  - The useful object is full geometric-product mixing, not wedge alone.
  - Key mechanisms: differential context `C_loc - H`, shifted sparse product, gated residual,
    optional global context.
  - Direct next action: port the faithful block ideas into an LDT proposer.

- `pdfs/2605.08605-lattice-deduction-transformer.pdf`
  - Most important for the reliable deduction side.
  - LDT defines the per-position powerset lattice, alpha target, recurrent solve loop, conflict
    head, and train/inference on-policy matching.
  - Key correction: use multi-solution alpha where needed. A planted solution target is not the
    same as soundness for multi-solution domains.

- `~/dev/graphplay/Graphplay/Integrations/LDT*.lean`
  - Not in `pdfs/`, but essential.
  - These files clarify that soundness comes from checking/certificates, while completeness is the
    actual research frontier.

### Strongly Useful

- `pdfs/1704.02942-budinich-clifford-sat.pdf`
  - Gives a symbolic Clifford-algebra SAT formulation using idempotents and a necessary/sufficient
    unsatisfiability condition.
  - Useful as conceptual grounding for "Clifford algebra can represent Boolean structure."
  - Not a neural architecture by itself, but it can inspire atom/idempotent encodings.

- `pdfs/2210.10749-transformers-shortcuts-automata.pdf`
  - Gives the semigroup/automata lens.
  - Useful for performance after learning a one-step lattice operator: distill repeated deduction
    into binary-lifted macro steps and measure shortcutability.
  - Key result for us: all finite-state automata admit `O(log T)` transformer shortcuts by composing
    transformations; many solvable semigroups admit constant-depth shortcuts; gridworld-like tasks
    can be even shallower.
  - Key warning: learned shortcuts can be brittle under distribution or length shift, so macro
    deduction must stay checked and be tested OOD.

- `pdfs/1807.03819-universal-transformers.pdf`
  - Supports weight-tied recurrent depth over representations.
  - Useful for organ recurrence and adaptive per-position computation.

- `pdfs/2107.05407-pondernet.pdf`
  - Useful for learned halting/adaptive compute.
  - Lower priority than getting the exact CSP and full-product block right.

- `pdfs/2502.05171-geiping-recurrent-depth.pdf`
  - Useful for the "latent reasoning" framing and LLM graft.
  - It argues recurrent depth can scale test-time compute without emitting chain-of-thought tokens.

- `pdfs/2406.09308-transnar.pdf`
  - Useful for the co-processor architecture.
  - Its pattern is close to the desired host/organ coupling: a Transformer cross-attends to a
    specialized reasoner representation.

### Baselines and Cautions

- `pdfs/1905.12149-satnet-differentiable-maxsat.pdf`
  - Baseline for differentiable solver layers.
  - Useful comparison: differentiable MAXSAT layer vs checked lattice deduction organ.

- `pdfs/2312.11522-satnet-grounding-critique.pdf`
  - Critical warning for any visual/LLM-to-symbol stage.
  - Differentiability alone does not solve symbol grounding. The cortex must be supervised or
    constrained to emit valid symbols/programs.

- `pdfs/2506.21734-hrm-hierarchical-reasoning.pdf`
  - Useful as a recurrent-reasoning baseline and motivation for latent refinement.
  - Do not over-trust the hierarchy story without controls.

- `pdfs/2510.04871-trm-tiny-recursive-model.pdf`
  - Strong baseline. Tiny recurrence plus deep supervision can beat larger mystic architectures.
  - Any organ result should compare against simple recursive/deep-supervised controls.

- `pdfs/2510.00355-arcprize-hrm-analysis.pdf`
  - Useful skepticism: isolates which HRM parts matter and warns about architecture mythology.
  - Supports keeping controls simple and central.

### Diagnostics

- `pdfs/2604.19740-generalization-edge-of-stability.pdf`
  - Useful later for checkpoint and dynamics diagnostics.
  - Not an architecture. Treat as a way to log attractor/trajectory properties and see whether they
    predict held-out-domain completeness or soundness better than validation loss.

## Bottom Line

The right research stance is plural:

- CliffordNet suggests a compact full-product mixer/proposer.
- LDT supplies the checked lattice/recurrent solve loop.
- Lean supplies the soundness/completeness distinction and the kind hierarchy.
- SATNet and its critique warn that differentiability and grounding are separate problems.
- Automata/recurrent-depth papers suggest how to scale or compress repeated deduction.

The next code should therefore not be "more wedge." It should be:

```text
exact CSP harness + full CliffordNet-faithful LDT proposer + abstraction-level tests
```
