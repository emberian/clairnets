# Targets, regime, and what's grounded vs hypothesis

## Task ladder (each gates the next — don't skip)
1. **Sudoku-Extreme core-swap** (the falsifier). iso-param ldt/glados/nowedge. metric: false-elim rate
   + p90 forwards. wedge must lower false-elim or the organ is just an efficiency primitive.
2. **Domain-general gauntlet** (the "generalizes to nonsense" claim, done honestly). ONE organ, many
   lattices: sudoku, maze, graph-coloring (k-col), 3-SAT, nonogram, futoshiki, Latin-square. Shared
   powerset-lattice interface. Hold out a whole domain → measure ZERO-SHOT SOUNDNESS. Cleaner science
   than an ARC number because the abstract domain is explicit.
3. **Rule-induction host** (Layer 1 co-processor) → **ARC-AGI(-2)** as north star.

## ARC-AGI — north star, NOT a Layer-0 target
ARC is NOT a fixed powerset lattice: variable grid size, LATENT constraints (must induce rule from K
demos), output not a clean candidate set. That's why LDT reports no ARC and HRM's ARC is the contested
part (32% not 41%, puzzle-id leakage). The honest factoring:
    ARC = (host induces candidate rules from demos) ∘ (organ executes/verifies soundly, abstains on ⊥)
So ARC is reached through the augmented architecture (host proposes hypotheses → writes constraints →
organ certifies/abstains → resample), not by torturing the bare organ.

## Shortcut-learning-maximizing regime (automata 2210.10749, the GOOD kind of shortcut)
Pressure the model toward the O(log T)-depth compositional (semigroup) solution, and MEASURE it:
- **random unroll depth + weight-tying** → operator must be correct at any depth → ~idempotent on its
  fixpoint → a semigroup element whose powers stabilize (a shortcut can exist).
- **binary-lifting distillation** F₂≈Π(F₁∘F₁), F₄, F₈: literally constructs the shortcut ("over Streams"),
  AND the lifting error is a diagnostic — low ⇒ solvable/shortcuttable semigroup; high ⇒ Krohn-Rhodes says
  genuine depth needed. Report lifting-error curve + effective-depth-to-solve per task.
- **deep supervision every unroll** → steps forced monotone & composable.
- **on-policy alpha-target** → kills transductive/puzzle-id memorization (HRM's ARC inflation).

## Online edge-of-stability (study dynamics the WHOLE way, not post-hoc)
Paper's sharpness-dimension is post-hoc + Hessian-quadratic-memory (intractable). Cheap ONLINE proxies,
logged every N steps (we already built PR + finite-time-Lyapunov in gowexp):
- top-k Hessian eigenvalues via Hutchinson/power-iteration on a minibatch (no full spectrum)
- weight-trajectory participation ratio = effective attractor dimension the optimizer explores
- finite-time Lyapunov of the training map = top singular value of (I − η∇²R̂) by power iteration
Run the SAME suite on the inference-time deduction trajectory (contractive narrowing = gowith "basin-holding").

**Falsifiable bet that is OURS, not the paper's (flag as hypothesis-under-test):** attractor dimension
predicts cross-DOMAIN generalization (held-out-domain soundness) better than val-loss. If true → a
dynamics-aware checkpoint-selection/LR rule: pick lowest-SD checkpoint, it wins on the unseen domain.

## Grounded vs hypothesis (be honest in the writeup)
- GROUNDED: LDT sound deduction @ 800K=100% Sudoku-Extreme; Clifford FFN-redundancy + wedge carries most
  of it; automata O(log T) shortcuts exist for solvable semigroups; HRM core = tied-depth+deep-sup.
- HYPOTHESIS (ours, under test): wedge lowers false-elimination iso-param; one organ generalizes
  zero-shot across domains; attractor-dimension predicts cross-domain soundness; host+organ does ARC.
