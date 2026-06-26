# `clair/` — what's current (the map)

This is the index that says **what to read first, what's a kept result, and what's quarantined**.
Repo housekeeping snapshot: `88 → 59` live `clair/*.py` (+ `organ/`, `datagen/`); 29 stale files moved to
`clair/_graveyard/` (history preserved via `git mv`, still tracked — quarantined, not deleted).

See also `README.md` (the paper-level overview) and `clair/organ/README.md` (the assembled organ).

---

## Canonical pipeline (KEEP — read these first)

The current model is the **staged generative readout**: a learned, checked deduction *organ* woven into
OLMo and trained to be used. Read in this order:

1. **Ground truth & lattice** — `csp.py` (exact finite-CSP harness: solutions, `exact_dedₚ`, the general
   factor lattice — *the verifier*) · `levels.py` (relation × abstraction-level map).
2. **The organ package** — `organ/` IS the assembled organ. `organ/protocol.py` (the typed spine:
   Reduction / CSPState / Certificate + structured-γ readout) · `organ/bank.py` (every validated beast
   registered as a sound reduction) · `organ/compose.py` (verifier-gated reduced product) ·
   `organ/train.py` (the single train + weave entry) · `organ/selftest.py` (`python -m clair.organ.selftest`).
3. **The neural deductors** — `proposer.py` (the core narrow organ: factor-graph lattice proposer) ·
   `blade_deductor.py` (grade-structured factor deductor for affine/modular).
4. **Multitask + woven model** — `run_general.py` (multitask organ training) ·
   ⭐`run_glados_staged.py` (**the working woven model**: live-latent-α, STAGE-1 bootstrap → freeze →
   γ-readout → generate, with the causal controls).
5. **Verifier ladder & corpus** — `curriculum.py` (NL-rendered constraint problems) · `fol.py`
   (forward-chainer) · `smt.py` (z3 oracle) · `schedule.py` (multi-task loader) · `datagen/` (the
   reproducible corpus pipeline: `build · render · skins · extra · qc · parse · trace · fast ·
   verify_fast · compose_csp · _dedp_worker`).

**Load-bearing supports** the pipeline imports transitively (keep): `hard_tasks.py` (propagation-hard
problems) · `oracle_readout.py` / `frozen_readout.py` (the proven readout + causal controls) ·
`latent_organ.py` + `latent_tasks.py` (pure-latent compile) · `induce.py` · `augmented.py` · `ldt.py`
(⚠ see *Flagged* below) · `model.py` (ClairNet, used by `diffusion_organ`).

Verify the tree is intact:
```bash
python -m clair.organ.selftest
python -c "import clair.run_glados_staged, clair.run_general"
```
(Note: `clair.eval_suite` referenced in older notes does **not** exist in the repo.)

---

## Validated beasts & experiments (KEEP — these are the paper's results)

**The organ bank** (each previously proven in its own file, now wired into `organ/bank.py`):

| file | what it proved |
|---|---|
| `csp.py` | exact finite-CSP ground truth + `exact_dedₚ` (the verifier all soundness is measured against) |
| `levels.py` | relation × abstraction-level map for the curriculum |
| `proposer.py` | factor-graph lattice PROPOSER — the crux: soundness-free, completeness the variable |
| `blade_deductor.py` | grade-structured factor deductor (grades ↔ arities) |
| `blade_hi.py` | higher-grade / un-truncated blade folds (grade-4/5 affine, zero-shot arity) |
| `macro_deduct.py` | checked macro-operators → exact dedₚ for path CSPs in O(log L) depth |
| `modular.py` | certified modular-arithmetic / congruence organ (SNF + CRT over ℤ_m) |
| `xor_wall.py` | the affine-wall test — GF(2)-forced narrowing beats the 3-way correlation ceiling |
| `row_space.py` | GF(2) row-space / Gaussian-elimination organ (complete for GF(2)-affine) |
| `fol.py` | synthetic FOL/Datalog entailment, exact (least Herbrand model, 3-way entail/contradict/unknown) |
| `smt.py` | SMT / linear-integer arithmetic with z3 as the exact oracle |
| `strings.py` | certified string/sequence abstract-domain organ |
| `permgroup.py` | exact permutation-group harness (ground truth: where Hall GAC is already exact vs the subgroup abstain-gap) |
| `perm_organ.py` | the learned PERMUTATION-GROUP organ (narrows subgroup-membership where local fixpoint abstains) |
| `typeinfer.py` | exact type-inference harness (CSP over a bounded type universe; Algorithm-W cross-check) |
| `type_organ.py` | the learned TYPE-INFERENCE organ (narrows per-node type sets where AC abstains) |
| `graph_organ.py` | certified single-source shortest-path / reachability organ |
| `ising_organ.py` | optimization beast — Ising / QUBO as a universal substrate |
| `energy_organ.py` | RUNG-3 energy / equilibrium organ (the optimization shape; ~6% infeasible at the affine wall) |
| `chain_organ.py` | RUNG-3 dual — differentiable recurrent neural forward-chainer (sound *or* deep phase transition) |
| `big_organ.py` | larger-budget finite organ (capacity sizing) |
| `general_relation_organ.py` | factor-generalization to OOD-novel relations (the organ's #1 weakness) |

**Beast drivers / studies** (the experiments that produced the tables):

| file | what it ran |
|---|---|
| `run_general.py` | the multitask (7-rung MIX) organ training — produces `general_organ_full.pt` |
| `run_proposer.py` | proposer vs exact dedₚ; the 4-arm geometric ablation (ffn/inner/wedge/full) |
| `run_blade.py` | BLADE vs TABLE factor-deductor expressiveness comparison |
| `run_blade_hi.py` | higher-arity (grade 4/5) & un-truncated blade comparison |
| `run_modular.py` | neural guidance for the certified congruence organ + recall/soundness table |
| `run_strings.py` | neural guidance for the certified string organ + recall/soundness table |
| `sweep_organ.py` | organ-recipe sweep: capacity × depth × composition-mixup |
| `composer.py` | multi-organ composer — verifier-gated search over certified reductions |
| `culmination.py` | the culmination experiment (best-organ integration) |
| `organ_excellence.py` | the best organ + zero-shot reasoning-gym organ eval |
| `rg_csp.py` | reasoning-gym coverage for the deduction organ |
| `run_exploitation.py` | the decisive architecture-vs-trick experiment (does ordinary CPT exploit the organ?) |
| `run_splitbrain.py` | split-brain necessity test (does the LM *wield* the organ, or shortcut?) |
| `oracle_readout.py` / `run_oracle.py` | oracle-readout de-risk + causal control table (true/shuffle/permute/corrupt) |
| `frozen_readout.py` / `run_frozen.py` | frozen-learned-organ readout (the clean complement: co-training distrust vs organ noise) |
| `latent_organ.py` / `latent_tasks.py` / `run_latent.py` | pure-latent organ — narrow from OLMo's hidden alone, no symbolic extraction |
| `interp_probe.py` | interpretability probe on the latent organ's flows |
| `induce.py` / `run_induce.py` | the original design-(d) toy (in-context hybrid; +30 OOD abstain-recall) ⚠ pre-woven, see *Flagged* |
| `augmented.py` | the per-cell CellReader + English renderer (⚠ load-bearing for `latent_organ`, see *Flagged*) |
| `hard_tasks.py` | hard, propagation-required problems for the decisive woven test |
| `diffusion_organ.py` | diffusion-organ graft (organ-narrowing ↔ denoising loop) |
| `diffusion_naturalizer_bench.py` | is a diffusion LM good enough as the datagen NATURALIZER layer? |
| `rlvr_pipeline.py` | RLVR base pipeline (TRL GRPO) de-risk — *needs the external `reasoning_gym` package to import* |
| `schedule.py` | the verifier-ladder multi-task scheduler |
| `gen_curriculum.py` | current curriculum-dataset generator (CSP → Bedrock rendering → faithfulness round-trip → `curriculum.jsonl`) |
| `graft.py` | Clifford-graft experiment: can a geometric product carry a real OLMo's FFN? (GA-on-real-LLM) |

---

## Archived (`clair/_graveyard/` — superseded / dead / pre-pivot)

Moved out of the main tree so they don't mislead about what's canonical. Still tracked (history +
visibility); nothing in the kept tree imports any of these.

**Earlier WOVEN variants** (superseded by `run_glados_staged`'s live-α staged recipe — the priority to
quarantine, since they imply an *old/different* woven approach is current):

| file | why archived |
|---|---|
| `glados_woven.py` | the "corrected-D" woven GLaDOS; predates the live-latent-α staged model |
| `run_glados_woven.py` · `run_glados_hard.py` | train/eval drivers for `glados_woven` |
| `run_alpha.py` | standalone-pretrain-the-α-compiler attempt (symbolic-α, pre-live-α) |
| `gen_augmented.py` · `run_gen.py` | generative hybrid-deductive OLMo via LoRA woven into lm_head (earlier variant) |
| `run_augmented.py` | the augmented OLMo driver (frozen host + zero-init gated adapters) — earlier woven approach |
| `rlvr_augmented.py` | dead RLVR scaffold for the augmented OLMo (current RLVR is `rlvr_pipeline.py`) |

**Pre-pivot GLaDOS/Sudoku monolith** (the powerset-lattice-over-Sudoku era, before the factor-graph
organ + `csp.py` pivot):

| file | why archived |
|---|---|
| `glados.py` | the old monolithic GLaDOS (MHA + GeomMix over a Sudoku powerset lattice) |
| `glados_variants.py` · `run_glados_variants.py` | GLaDOSv layering-knob sweep on Sudoku-Extreme |
| `run_glados.py` | the ldt/glados/nowedge core-swap on Sudoku-Extreme |
| `organs.py` · `run_organs.py` | alternative-context organs (gnn/ssm) for the monolith |
| `sudoku.py` | Sudoku as the original lattice-deduction domain |
| `domains.py` · `run_gauntlet.py` | domain-general gauntlet over sudoku-shaped powerset lattices |
| `tasks.py` · `train.py` | the original ClairNet (arch × task) gauntlet trainer (`organ/train.py` is canonical now) |

**Parked pre-pivot GA / LM prototypes** (README "Parked" — geometric-algebra & LM scaling threads that
predate the deduction-organ direction):

| file | why archived |
|---|---|
| `cliffordnet.py` · `run_cliffordnet.py` | CliffordNet reproduction (CIFAR-100); GA placed elsewhere now |
| `ldt` runner `run_ldt.py` | faithful-LDT-on-Sudoku driver (the `ldt.py` module itself is KEPT — load-bearing, see *Flagged*) |
| `geom_lm.py` · `run_geom_lm.py` · `sweep_geom_lm.py` | GeomLM channel-mixer LM arms + scaling sweep |
| `rope_lm.py` · `run_rope_lm.py` | RopeLM positional-encoding LM arms |
| `vision.py` · `run_vision.py` | ClairVision image-classifier mixer comparison |

---

## ⚠ Flagged (kept despite looking stale — DO NOT archive without decoupling first)

- **`augmented.py`** and **`ldt.py`** look like pre-pivot variants, but `latent_organ.py` (which the
  *canonical* `run_glados_staged.py` imports at top level) pulls `CellReader` from `augmented` and
  `make_mixer` from `ldt`. Archiving either **breaks the canonical import**. Decoupling is non-trivial
  (real `nn.Module` reuse), so they stay. `induce.py` is likewise kept because `augmented` imports it.
- **`latent_tasks.py`** — imported by `run_glados_staged`; not dead.
- **`run_induce.py`** — the earliest design-(d) in-context toy; pre-woven and superseded, but documents a
  real early result (+30 OOD abstain-recall). Kept as a borderline result rather than archived.
- **`graft.py`**, **`gen_curriculum.py`**, **`perm_organ.py`**, **`type_organ.py`** — not in the original
  beast enumeration but each is a real result / current tool; kept.
- **`rlvr_pipeline.py`** — fails to import only because the external `reasoning_gym` package isn't
  installed (pre-existing); it imports no archived module.
