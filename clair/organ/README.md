# `clair/organ/` — THE GLaDOS organ

**This package IS the organ.** One coherent, trainable, LLM-ready artifact that wires every
validated beast (each previously proven in its own file) into a single typed spine — ready to plug
into the full-training run. It does **not** re-prototype anything: every reduction imports + adapts
the already-committed organ code and loads the trained checkpoints.

```
clair/organ/
  protocol.py   the typed spine: Reduction / CSPState / Certificate + the structured-γ readout
  bank.py       the registry of validated beasts as reductions (certified ops + neural guidance)
  compose.py    verifier-gated certified-reduction composition (the sound reduced product)
  train.py      the single training + LLM-weaving entry point
  selftest.py   the consolidated smoke (python -m clair.organ.selftest)
  README.md     this file
```

## The spine (`protocol.py`)

The codex zoo review (`notes/codex_zoo_review.md`) named the precondition for the whole zoo to hold:
*every domain must have explicit semantics — α, γ, ⊤/⊥, meet/reduction, verifier/certificate,
abstain — else "domain × operation" is a taxonomy of vibes.* `protocol.Reduction` is that contract:

- an abstract **STATE** (the lattice element it narrows);
- **`reduce(state) → state′`** with γ(state′) ⊆ γ(state) — a **sound narrowing** (or a no-op = ABSTAIN);
- a **`certificate()`** declaring *how* soundness holds (`sound-by-construction` / `neural-guidance` /
  `approximate`) and the honest completeness caveat;
- an optional **`guidance()`** neural hook (raw proposer logits, for the verifier-gated meet / training);
- the uniform **`survival(state, K) → surv[n,K]`** structured-γ readout — the exact tensor
  `clair.oracle_readout.OracleGamma` projects into OLMo's residual stream (zero-init tanh gate ⇒
  bitwise no-op at init). **One channel the LLM reads every organ through.**

`CSPState` (per-cell domains over a finite `clair.csp.CSP`) is the **reduced-product spine**:
certified narrowing organs over it compose by pointwise meet. Beasts whose native semantics are not
a per-cell lattice (forward-chaining closure, energy simplex) declare their own `state_type` and do
not silently fuse — composing them needs an explicit bridge.

## The bank (`bank.py`) — per-beast status

`build_bank(load_neural=True)` returns `{name: Reduction}`. Pure-certified entries need no torch and
no checkpoints; neural entries lazy-load on first use.

| reduction | source beast | state | status | completeness |
|---|---|---|---|---|
| `arc_consistency` | `csp.ac_step` | CSP-domain | **certified** (sound-by-construction) | level-0; abstains on affine/Hall |
| `factor_consistency` | `csp.factor_step` (k=3) | CSP-domain | **certified** | level-2; solves bounded-width, not unbounded XOR |
| `exact_dedP` | `csp.exact_dedP` | CSP-domain | **certified** (the verifier) | exact per-cell; exponential worst-case |
| `modular_snf` | `modular.solve_mod` (integer SNF) + CRT | CSP-domain (`LinSystem`) | **certified** | complete over ℤ_m incl. composite rings |
| `gf2_rowspace` | `xor_wall.gf2_forced` (GF(2) RREF) | CSP-domain (`('gf2',A,b)`) | **certified** | complete for GF(2)-affine (beats the affine wall) |
| `macro_reach` | `macro_deduct.macro_solve` (reach-doubling) | CSP-domain (path) | **certified** | exact dedP for path CSPs in O(log L) depth |
| `unification_chain` | `fol.forward_chain` (least Herbrand model) | fol-closure | **certified** (standalone) | exact entailment; 3-way entail/contradict/unknown |
| `energy_optimise` | `energy_organ.relax_decode` (mean-field) | simplex | **approximate** (standalone) | sound energy descent, approx decode; exact via `brute_opt` |
| `core_narrow_organ` | `proposer.FactorGraphProposer` + `runs/general_organ_full.pt` | CSP-domain | **neural-guidance** (verifier-gated) | union-trained over 7 rungs; honest affine/ordering gaps |
| `blade_affine_organ` | `blade_deductor.BladeFactorDeductor` + `runs/modular_organ.pt` | CSP-domain | **neural-guidance** (verifier-gated) | blade/grade prior for affine/modular |

**The core narrow organ** is `core_narrow_organ`: the union-trained general deductor
(`runs/general_organ_full.pt` — a `FactorGraphProposer` trained by the STAGE-1 recipe over the 7-rung
MIX coloring/equality/ordering/arithmetic/alldiff/eqchain/forcedcolor). It is the default deductor;
the **blade/grade prior for affine** is `blade_affine_organ` (`runs/modular_organ.pt`).

### Beasts that did **not** cleanly integrate as composable reductions (honest list)

- **graph min-plus / shortest-path**: no dedicated certified min-plus organ exists. The closest are
  `macro_reach` (path-CSP reachability via transition-monoid composition, *wired*) and `fol`
  forward-chaining (a min-plus dual on the rule DAG, wired as `unification_chain`). A standalone
  Floyd–Warshall/Bellman-Ford reduction is **not** in the repo and is not faked here.
- **alldiff / permutation-group**: no Hall-set / matching / Schreier-Sims certified operator exists;
  alldiff is handled by ≠-clique decomposition + `factor_consistency` (sound but incomplete — no Hall
  propagation). Registered behavior is the clique path, not a true permutation organ.
- **`latent_organ`, `chain_organ`, `diffusion_organ`**: neural modules without a checkpoint in
  `runs/` and (chain/diffusion) requiring their own grounded-graph / denoiser state. They are
  validated standalone (their own smokes) but are **not** wired as drop-in `Reduction`s — adding them
  needs a trained checkpoint + a state bridge, deferred rather than half-wired.
- **`energy_optimise`** is registered but is `approximate` and lives on its own simplex state — it
  does **not** reduced-product with the CSP organs.

## The composer (`compose.py`)

`reduced_product(state, reductions, verify=True)` composes CSP-domain reductions by the
**verifier-gated reduced product**. The review's key correction: output-verification makes accepted
*answers* sound, not intermediate *states* — "one unsound reduction poisons the next organ." So:

- a **certified** reduction is trusted (its output is asserted ⊆ the input and met in);
- a **neural/approximate** reduction is **verifier-gated**: only the eliminations the exact per-cell
  verifier (`csp.exact_dedP`) also makes are kept, so a wrong neural proposal can never poison the
  shared state.

The meet of sound narrowings is sound, so the composite stays sound by construction; `verify=True`
asserts false-elim 0 vs the exact oracle as a belt-and-braces check.

## Train it / weave it / RL it (`train.py`, `graft.py`, `eval.py`)

The canonical pipeline is **pretrain_organ → weave → rlvr** (each call delegates to the validated
recipe, nothing re-prototyped). The single front-door map with full prose is [`GLADOS.md`](../../GLADOS.md).

```bash
# STAGE 1 — pretrain the organ (dominate-dedₚ over the 7-rung MIX; recipe ~1.5M params, R=12, ~30% composed)
python -m clair.organ.train pretrain --steps 1500 --out runs/general_organ_full.pt

# STAGE 2 — graft into ANY of the 7 bases (clair.organ.graft) + train the woven readout with the
#           engagement mechanism (two-stream lever + J0 α-supervision + causal-control readout)
python -m clair.organ.train weave --base allenai/OLMo-2-0425-1B --regime hard --steps 2500 --out runs/woven.pt
python -m clair.organ.train weave --smoke         # tiny end-to-end woven smoke (loads OLMo-2-1B)

# STAGE 3 — RLVR (Dr.GRPO, exact-verifier reward; organ-as-process-reward at the insertion point)
python -m clair.organ.train rlvr --task chain_sum --steps 300

# EVAL — the arbiter (Tier-1/2/3 + causal controls + pass@k)
python -m clair.organ.eval --base allenai/OLMo-2-0425-1B --woven_ckpt runs/woven.pt
```

The training corpus is produced on the fly by the witness-first rung generators (so
`data/glados_corpus` is optional); the diverse Bedrock phrasings at
`data/curriculum/curriculum.jsonl` feed the OOD-phrasing eval. `graft_organ` is residual-stream-agnostic
(it splices onto `model.layers` / `gpt_neox.layers` / `language_model.layers` / hybrids), so STAGE 2
hosts any base — frozen + LoRA + latent-α + frozen organ + the zero-init γ readout.

## Test it (`selftest.py`)

```bash
python -m clair.organ.selftest            # core smoke (no OLMo): bank soundness + composer + readout + organ-train
python -m clair.organ.selftest --woven    # + the full OLMo staged woven smoke
```

The unified smoke proves: (1) each certified reduction is sound on its domain (false-elim 0 vs the
exact verifier; specialized ops match their own authority), (2) the composer chains ≥2 domains
(modular + GF(2)) soundly and narrows where local AC abstains, (3) the readout is a bitwise no-op at
init, (4) the woven entry point runs (organ-side training trains + saves + reloads the core organ,
verifier-gated sound; `--woven` runs the full OLMo staged smoke).
