# clairnets — handoff for outside review (GPT-5.5 / codex)

You (codex) reviewed an earlier version of this project. We've since pivoted hard and have real
results. We want a brutally honest review: is this architecture the right bet, what alternatives
should we consider, what training tasks/curricula/schedules would best test+train it, and where are we
fooling ourselves. Be adversarial. We value honest nulls.

## The bet (the spine)
A **learned proposer woven into an LLM + a checked lattice deductor** for reliable reasoning:
> `a_{t+1} = a_t ⊓ Π(f_θ(a_t, problem))` — a neural net PROPOSES candidate eliminations; a lattice MEET
> narrows monotonically; SOUNDNESS is gotten cheaply AT THE OUTPUT (verify the answer), which is a
> *permission to make the deductor loose/learned/differentiable*, not a prescription to make it a verifier.
> COMPLETENESS (does it reach a checkable answer vs abstain) is the research variable.
Grounded in: CliffordNet (geometric product), the Lattice Deduction Transformer (arXiv 2605.08605) + its
**Lean formalization** (soundness is free if checked; completeness is lattice-level × problem-width;
proven), and the automata-shortcuts paper (computation distributes across transformer depth).

## What we built
- **Exact CSP harness (`clair/csp.py`)** — GROUND TRUTH: exact solutions, the exact per-cell transformer
  `dedₚ`, per-cell arc-consistency, a **general factor lattice (level-k, any arity)**, polymorphism
  diagnostic. Reproduces the Lean's proven facts; the factor lattice solves XOR (affine wall) where
  per-cell abstains, soundly. `clair/levels.py` maps relation types onto the hierarchy.
- **Augmented LLM (`clair/augmented.py`)** — frozen OLMo + a **fact-grounded WRITE head** (reads each
  constraint from the host states where its entities co-occur) → a **learned differentiable lattice
  deductor** (currently coloring-specific) → a **set-valued READOUT** (reports the narrowed candidate set;
  `unknown` = a set with >1 element). Trained with SFT and **RLVR** (GRPO on the EXACT verifiable reward).
- **Curriculum (`clair/curriculum.py` + verifier-ladder, building)** — witness-first exact CSPs rendered
  into diverse NL via Bedrock (verified, $0.18 for 2306 records); new relation types ordering/arithmetic/
  all-different; verifier-ladder rungs (FOL/Datalog forward-chainer, Z3/SMT oracle) + a multi-task scheduler.

## The findings (honest chain of diagnoses — each redirected the next)
1. **Geometric product as a channel MIXER**: NULL — ties SwiGLU/MLP at iso-param in language AND vision;
   wins only low-param. (CliffordNet's claim is Pareto efficiency, not iso-param; our vision repro was
   also unfaithful. Regime-caveated, not a clean kill.) Geometry's home is *position/relation* (RoPE is
   already a Cl(2) rotor), not the FFN.
2. **The (d) bet — host compiles a problem into a checked deductor, abstains**: WORKS. Toy version
   (structured facts): calibrated abstention that *survives distribution shift* (abstain-recall gap ~0
   in-dist → +30 OOD). Scaled to a real LLM (OLMo-2-1B reading English): augmented det-acc **88.8% vs base
   OLMo 1.4%** on answerable; real calibration vs base's always-abstain.
3. **RLVR fixes CALIBRATION not capacity**: GRPO on the exact reward → +18–20 abstain-recall, +8–9 overall
   OOD; it learns to abstain rather than confabulate where extraction is unreliable. (Replicates at 1B and 7B.)
4. **Capacity does NOT break the wall**: OLMo-3-7B is *slightly worse* than 1B on extraction, same
   degradation shape. The wall is **architectural**: the WRITE head couldn't bind constraints as N grows.
5. **Fact-grounded WRITE head fixes extraction ROBUSTNESS not exactness**: edge-F1 survives diverse phrasing
   (6→43 held-out) and cross-phrasing OOD (14→65 at N=11), but det-acc is flat — because forcing a unique
   answer needs a *near-exact* program (one bad edge among ~55 flips determinacy). The deductor is BRITTLE.
6. **Set-valued readout makes degradation GRACEFUL + SOUND**: useful-info ~flat OOD (69→67→66 at N=5/8/11)
   while singleton-accuracy collapses (99→79→59); soundness holds ~89% at N=11 (a bad edge widens the set
   instead of flipping the answer). Honest caveats: the *learned* deductor leaks ~11% soundness OOD;
   useful-info drops to 27% on diverse-NL (the hard-extraction frontier).

## The honest current limitation (verified in code)
**The organ has ZERO causal influence on OLMo.** OLMo is frozen (`requires_grad_(False)`), its hiddens are
detached (`.detach()` under `no_grad`), and the organ is TERMINAL (reads the final hidden state, never
writes back into the layers). OLMo computes bit-identically with or without the organ and never learns it
exists. The "woven across layers / solve SAT between layers" language is aspirational; mechanically it's a
read-out head. This is the next thing to fix.

## The staged plan
- **Stage A (done)**: set-valued readout (error-tolerant, native `unknown`).
- **Stage B (building)**: verifier-ladder curriculum + multi-task schedule (FOL/Datalog/Z3 → NL-deduction →
  Lean apex). Exact verifier at every rung so RLVR carries up.
- **Stage C (next)**: **woven inter-layer organ with causal influence** — (1) un-detach + LoRA OLMo
  (optimizer influence); (2) move the organ between layers with a ZERO-INIT gated write-back into the
  residual stream (forward-pass influence); (3) multiple organs across depth (the substrate;
  automata-shortcuts = deduction distributed across depth). Zero-init every addition (no-op at init).
- **Stage D (idea)**: after the architecture-specific curriculum, **re-run OLMo's OWN open recipe (Dolma
  continued-pretrain + instruction-tune + RL-Zero) with the organ in place** to CONSOLIDATE/GENERALIZE the
  organ's use into general behavior (only possible because OLMo is fully open).

## The open questions we most want you to push on
1. **Generalization**: will the LM learn to invoke the deductor during *normal* reasoning, or only on
   constraint-shaped tasks? Is the Stage-D "re-run the general recipe with the organ" the right way to
   generalize a narrow-curriculum skill, or is there a better path?
2. **One general deductor vs a bank of specialists**: we lean toward ONE general factor-lattice deductor over
   a universal constraint representation (the WRITE head compiles any problem into variables/domains/factors;
   matches our exact harness). Alternative: a MoE-of-deductors (theory specialists, run in parallel, the
   verifier selecting the sound result; woven-across-depth organs as different specializations). Which, and why?
2b. Different relations need different lattice LEVELS (per our `levels.py`: arithmetic→factor/triple,
   coloring→search). Should the general organ always run at the factor level + branch, or adapt its level?
3. **The deductor-brittleness problem**: is set-valued readout + error-tolerance the right fix, or is there a
   better one (soft/probabilistic constraints; error-correcting deduction exploiting constraint redundancy;
   abstain-and-resample)? Real CSPs are over-determined — can we exploit that for robustness?
4. **The woven-organ architecture**: is inter-layer zero-init gated write-back + LoRA + multiple organs
   sound? Failure modes? Better ways to give a sub-module causal influence over an LLM's forward pass?
5. **Are we reinventing something?** Neuro-symbolic, SATNet, DeepProbLog, neural theorem proving,
   tool-augmented LLMs, scratchpad/latent-reasoning (Geiping recurrent-depth), TransNAR. What's genuinely new
   here (we think: a learned-but-checked deductor woven into the forward pass + a verifier-ladder curriculum +
   set-valued/unknown native output), and what's a known idea we should just adopt?
6. **Training tasks/schedule**: beyond CSP→FOL→SMT→code→Lean, what verifiable reasoning tasks/curricula would
   best train+test this? What schedule (curriculum anneal vs uniform mix vs self-paced)?

Repo: github.com/emberian/clairnets (branch dev). The README + notes/ have details; csp.py/augmented.py/
rlvr_augmented.py/curriculum.py/levels.py are the live code.
