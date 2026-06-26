# CURSELOG 😈

The running lab notebook for GLaDOS — the *journey*, not the current state (that's the README). Honest,
terse, newest-first. Append new entries at the top. Negatives are results.

---

## 2026-06-26 — toward real training

- **RLVR base pipeline DE-RISKED** (`clair/rlvr_pipeline.py`). TRL-GRPO + reasoning-gym + our exact-verifier
  reward + OLMo-2-1B **runs and learns** on one L40S: reward EMA 0.20→0.72 over 250 steps (chain_sum), no OOM.
  Stack: trl 1.7.0, `use_vllm=False` (organ-compatible rollout), Dr.GRPO, beta=0 (no ref-model), LoRA all-linear.
  **Organ socket = swap `build_model()`'s policy** for the augmented OLMo; reward/dataset/config unchanged. The
  real-training substrate works. Next: wire the organ in + the matched 2×2 + a **process-reward arm** (RLTT/LSRL,
  PDFs in `pdfs/`) so a GRPO null is interpretable.
- **Prior art gathered + positioned** (`notes/reading_list.md`, `nesy_landscape.md`, `prior_art_extract.md`):
  the pieces exist separately — SATNet/DeepProbLog/NLM (neural+logic, pre-LLM), **TransNAR** (LLM↔reasoner
  coupling), **RLTT/LSRL** (process-reward for latent/looped reasoning), reasoning-gym (verifiable fuel) — but
  nobody's assembled our combination: a *checked sound* lattice/factor deductor + the organ bank woven into a
  pretrained LLM, verifier-trained. Novelty is narrow + integrative → lift recipes, don't reinvent.

- **RLVR landscape mapped → the real-training pivot** (`notes/rlvr_landscape.md`). Two findings that reshape
  *how we test the organ*: (1) **outcome-GRPO falsely nulls an organ** — Ouro/RLTT showed terminal-only reward
  never credit-assigns to an organ's internal steps; a process/trajectory-reward arm is *mandatory* for a fair
  test. (2) **OLMo is the honest base** (Qwen RLVR gains are partly spurious/pretraining). Premise *vindicated*:
  Logic-RL — a 7B trained only on procedural Knights-&-Knaves transferred to AIME/MATH. Plan: TRL GRPOTrainer
  (non-vLLM rollout for the custom organ) + LoRA-GRPO on OLMo-1.5-2B + reasoning-gym/ZebraLogic/SATBench, 1 L40S.
- **Strategic zoom-out** (ember): stand out of the way of instabilities (we'd been over-scaffolding); reconsider
  whether the organ is a necessary architecture vs a sample-efficiency crutch RLVR-at-scale obviates; pivot to
  real RLVR on standardized verifiable tasks.
- **Parallel deductor training + new ideas** (5 boxes): depth-vs-affine-wall R-sweep (interim: blade ahead at
  R=2; F₄ macro distills 100% with capacity), general-deductor→excellence, macro-deduction (log-depth shortcut),
  A-vs-B diversity, staged generative assembly.
- **Macro-deduction → YES (with a precise caveat).** Checked macros give sublinear-depth sound reasoning:
  `⌈log₂L⌉` macro-applications replace ~L base steps, identical exact dedₚ fixpoint, **0 false-elim, verified to
  L=256** (32× @ L=256). The compression is the *symbolic transition-monoid composition* (graph-squaring), not
  the neural net (which is a fidelity probe — sound to k=8). Cost is poly in state space `S=k^w`, so it only
  works on the **bounded-width** side — *width is the wall, bounded-width depth is a log-depth shortcut.* Geometry
  gave no edge on these (too-simple) automata transitions (honest null for GA-helps-macro).
- **Code optimizations**: backtracking `exact_dedP` (was `d^n`, hung at n≈5; 14-cell/5-val chain now 0.1ms) →
  unblocks hard/large-N. `infra/box_run.sh` helper. Contributor docs (README rewrite + CONTRIBUTING + GLOSSARY).

## 2026-06-25 — the woven-organ arc (the core question)

**Can a learned deductive organ be *causally used* by an LLM's generation? Answer: yes, with STAGING.**

- **Oracle-readout de-risk → POSITIVE.** Dense structured γ → OLMo's LM head reads the lattice *causally*
  (corrupt→0 even with full text present, OOD-perfect). The "latch gap" was a *bandwidth* problem; dense γ solves it.
- **Co-trained woven (symbolic α, old arch) → NEGATIVE.** LM ignores the organ — settles into a lazy pin-guess
  local optimum. The "airtight negative" bootstrap run confirmed it even with a necessary task + perfect organ +
  forced-open gate — but it was the *old* architecture (don't over-update).
- **Frozen-learned-organ readout → POSITIVE, and it reconciles everything.** A *frozen, standalone-trained*
  learned organ reads through γ essentially as well as the exact oracle (corrupt→0, gate opens, ignores redundant
  text). So the co-trained failure was **co-training distrust of a moving organ**, NOT the coupling. → the **STAGED
  recipe** (bootstrap organ → FREEZE → establish readout) makes a learned organ load-bearing in generation.
  Vindicates "train separately, then combine — don't cold-co-train."
- **Latent organ (B, bitter-lesson) → viable in-dist.** Fed only OLMo's hidden of the text (no factors, no
  extraction supervision), narrows 94% of dedₚ (within 5pts of explicit-from-factors); OOD it degrades (size/
  phrasing-coupled). A-vs-B = "can diversity make latent as invariant as factors are for free?" (running).

**The organ bank** — three inference shapes, each validated with honest limits: **narrow** (proposer; sound,
abstains), **chain** (sound *or* deep — a sharp phase transition), **energy** (uniquely does *optimization*,
~6% infeasible at the affine wall). Portfolio + verifier-selection = complementary, no single organ dominates.

**Geometric algebra, finally placed.** Null as a generic mixer; its home is the factor deductor: grade-k ↔
arity-k (wedge=binary, trivector=ternary affine). Blades help affine *generalization*; the **un-truncated fold**
(one wedge, all arities, K=capacity) is the right abstraction (matches hardcoded grades in-dist, generalizes best
zero-shot). The recurring **affine wall is WIDTH** (beaten by depth/rounds), and GA is *form*, not soundness.

**The exact harness** (ground truth): `csp.py` (exact solutions/dedₚ, general factor lattice), `levels.py`
(per-cell AC == exact dedₚ; the factor lattice breaks XOR's affine wall at level-2), the verifier ladder
(`fol`/`smt`/`schedule` + Bedrock-rendered `curriculum`).

**The augmented LLM (toy → real).** The (d) bet works: a frozen LLM + checked deductor gains calibrated
abstention (det-acc 88.8% vs base 1.4%). RLVR fixes *calibration* (+18-20 abstain-recall OOD), not capacity. 7B
≈ 1B (capacity isn't the wall). Grounded WRITE head fixes extraction *robustness*; set-valued readout makes
degradation graceful + sound.

**Codex review of the "D" weave**: caught the *bypass risk*; α must *compile* not *solve*; full-bandwidth
*structured* not dense; GA is completeness not soundness; organs share a *protocol* not a coupling; don't
backprop through repeated OLMo passes.

**Honest nulls**: geometric product as a channel mixer ties SwiGLU/MLP in language *and* vision (it's a
vision primitive at best, regime-dependent). GLaDOS-on-sudoku — we under-implemented LDT (no backtracking
search) → parked, not a verdict on the idea.

## Meta-lessons (the ones that keep paying off)
- **Stage the training; don't cold-co-train alien substrates.** Organ→excellence, freeze, then establish the readout.
- **Soundness is a *permission* (check the output), not a *prescription*** — the organ can be loose, learned, emergent.
- **Outcome-RL can't credit an organ's internal steps** — a process/trajectory reward is needed to test it fairly.
- **Test in the regime a method targets.** A classification-readout head is *not* the LM reasoning (need generative).
- **Don't conclude from underpowered or mis-aimed runs.** Run a portfolio; let the data, not the enthusiasm, pick next.
