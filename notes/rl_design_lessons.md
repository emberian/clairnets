# RL design lessons for the RLVR + looped-deduction organ test

Extracted from four PDFs in `pdfs/` (full method/experiment reads), plus mid-2026 web context.
Companion to `notes/rlvr_landscape.md` (process-reward arm) and `notes/prior_art_extract.md`
(TransNAR/RLTT/LSRL coupling + per-step reward). Purpose: know what to CONTROL for so a woven-organ
RLVR result on OLMo is *interpretable* — i.e. distinguishable from base-model elicitation, length
bias, or a metric artifact.

Sources:
- **[ELICIT]** 2504.13837 — "Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?" (Yue et al., NeurIPS'25 oral; v5 Nov 2025).
- **[SPUR]** 2506.10947 — "Spurious Rewards: Rethinking Training Signals in RLVR" (Shao, Li et al.; v2 Feb 2026).
- **[LOOP]** 2502.17416 — "Reasoning with Latent Thoughts: On the Power of Looped Transformers" (Saunshi et al., ICLR'25).
- **[2SCALE]** 2509.23314 — "Two-Scale Latent Dynamics for Recurrent-Depth Transformers" (Pappone et al.; v2 Nov 2025).

---

## 0. TL;DR for the experiment design
1. **Report pass@k, not pass@1.** RLVR raises pass@1 and *lowers* pass@k at large k — the curves cross around k ≈ tens–hundreds ([ELICIT] Fig 2). pass@1 alone cannot tell "raised the ceiling" from "sharpened the prior."
2. **OLMo is the correct interpretable base.** Spurious/random rewards recover most of the RLVR gain on *Qwen* but **not** on Llama/OLMo ([SPUR] Fig 1, 3). On OLMo a gain is real signal, not elicitation of a pretrained code-reasoning prior. Use a random-reward arm as a *dummy control*.
3. **Looped depth provably solves our task classes** (addition, p-hop, group composition, i-GSM) with a *narrow* block looped — depth ↔ iteration count, width ↔ per-step expressivity ([LOOP] §2, §5). Directly supports "affine wall = width, depth beatable."
4. **A recurrent narrower self-stabilizes** (step sizes shrink, refinements become orthogonal/spiral) ([2SCALE] §3); use a **second-order / monotone-cardinality exit**, not a norm threshold.
5. **Complications to plan for:** ProRL shows RLVR *can* expand the boundary with prolonged training + KL control + reference resets + task diversity (contradicts the strong [ELICIT] reading); GRPO clipping bias + length bias must be neutralized (Dr.GRPO/DAPO); KL hurts coverage; pass@k=0 prompts give vanishing gradients.

---

## 1. [ELICIT] 2504.13837 — RLVR elicits, rarely expands

**Precise claim.** RLVR improves *sampling efficiency* (pass@1) but does **not** expand the reasoning
*boundary*; the trained model's solvable set is ≈ a subset of the base model's. All reasoning paths the
RLVR model samples already exist in the base distribution.

**Where the metric flips.** pass@k curves of base vs RL **cross**: RL wins at small k (k=1, average-case),
base catches up and **surpasses** as k grows "to the tens or hundreds" ([ELICIT] Fig 2 caption, §3.1).
Concrete crossings:
- Minerva, Qwen2.5-32B: base beats RL by **~9%** at k=128 (≈9% more problems solvable).
- Frontier check (Magistral-Medium, pure-RL on Mistral-Medium-3): RLVR solves ~7 more AIME24 / ~8 more AIME25 at k=1, but the gap **steadily narrows** as k grows (Fig 9) — the effect holds even for near-frontier reasoning models.
- This is at **realistic k (128–1024)**, *not* astronomical k: "the base model already produces correct outputs at realistic values" (§2.2).

**Evidence it's elicitation, not expansion.**
- Accuracy histogram (Fig 5, Minerva): RL raises frequency near acc 1.0 **and** raises frequency at **acc = 0** (more *unsolvable* problems) — net coverage shrinks.
- Coverage table (Table 2): AIME24 — base-solves/RL-fails = **13.3%**, RL-solves/base-fails = **0.0%**. MATH500 — 3.6% vs 1.0%. RL almost never unlocks a new problem.
- Perplexity (Fig 6, §4.1): PPL_base(Y_RL) sits in the *lower tail* of PPL_base(Y_base) — RL responses are high-likelihood samples the base already favors. PPL_base(Y_RL) keeps dropping over training → RL *sharpens within* the prior.
- Manual CoT inspection rules out lucky-guessing: on hardest sub-5%-acc problems, base has ≥1 correct CoT for nearly all solvable items.

**RL-algorithm invariance.** Sampling-Efficiency Gap Δ_SE = (RL pass@1) − (base pass@k, k=256). Across
PPO/GRPO/Reinforce++/RLOO/ReMax/DAPO, Δ_SE varies only slightly and stays **>40 points** (Fig 8, §4.3).
Algorithm choice is second-order; the boundary limit is the binding constraint.

**Conditions under which RLVR does / doesn't expand (control levers):**
- **Distillation expands; RLVR doesn't** (§4.2, Fig 7): DeepSeek-R1-Distill-Qwen pass@k is consistently *above* the base — new reasoning patterns come from a stronger *teacher*, not RL. (Our organ is meant to be that extra source of capacity — so the honest test is whether organ+RL behaves like distillation, lifting pass@k, or like plain RLVR, flattening it.)
- **KL hurts coverage** (§4.4): adding KL coeff 0.001 gives similar pass@1 but **much lower pass@128** than KL-free GRPO. A KL leash keeps you inside the prior.
- **More rollouts barely helps** (§4.4): n 8→32 nudges pass@k up but RL still trails base.
- **Entropy isn't the cause** (§4.5): temperature-matching the RL model to the base's entropy does *not* recover base coverage — the narrowing is structural, not just lower-entropy sampling.

**Eval protocol to copy (so our pass@k is fair):** temp 0.6, top-p 0.95, max 16384 tokens; **no few-shot
prompt for the base model** (few-shot confounds the base's apparent capacity); use an unbiased
low-variance pass@k estimator (Brown et al.); manually verify a CoT subset on math to exclude guessing
(coding pass@k is guess-proof via unit tests).

---

## 2. [SPUR] 2506.10947 — random/wrong rewards work on Qwen, not on OLMo

**Headline numbers** (MATH-500, 300 steps GRPO; [SPUR] Fig 1, §2.2):
- **Qwen2.5-Math-7B:** ground-truth **+29.1**, majority-vote +26.0, format +13.8, **incorrect-label +24.1, random +21.4.** Spurious ≈ within a few points of ground truth.
- **OLMo2-7B:** ground-truth **+15.5**, but **random −6.4, incorrect −2.1, format −6.4, majority −8.3.** Gains come from ground truth *only*.
- **Llama3.1-8B-Instruct:** similar to OLMo — spurious rewards flat or negative.

**Mechanism (why Qwen?).** Two compounding causes:
1. **GRPO clipping bias** (§4, Eq 1–2): the clip term asymmetrically *raises probability of already-high-prior tokens* and suppresses large deviations, even with reward-independent (random) advantage. Ablation (Fig 4): with clipping **removed** (no-clip / single-update variants), random rewards give **no** gain — flat. Only **with clipping** do random rewards improve. So "training signal from noise" is a clipping artifact that amplifies whatever the pretrained prior already does.
2. **Model-specific amplifiable behavior** (§5): for Qwen-Math the amplified prior is **code reasoning** — emitting Python in the CoT (no interpreter), which *correlates with correctness* (Table 1: acc **60.9% with code vs 35.0% without**). Spurious rewards drive code frequency **65% → >90%** within ~15 steps (Fig 6). Llama/OLMo lack this latent prior (No-Code or Bad-Code, Table 1), so there's nothing for clipping to amplify.

**Safe controls.** Spurious-reward elicitation is documented to FAIL on **Llama3.1-8B-Instruct,
Llama3.2-3B-Instruct, OLMo2-7B, OLMo2-7B-SFT** (Fig 1, 3, §3). These are the interpretable bases. **OLMo
(ours) responds only to genuine reward** → a measured organ gain on OLMo is not a hidden-prior artifact.

**Implications for our test:**
- **Add a random-reward (and incorrect-label) dummy arm.** On OLMo it should do ≈ nothing. If our organ-augmented OLMo suddenly *gains* under random reward, that exposes an amplifiable shortcut (e.g. the LM gaming a γ-readout pattern) — exactly the cold-weave shortcut we already fear. This is a cheap, decisive falsification control.
- **Don't over-read Qwen-based RLVR literature.** Much of the field's RLVR "wins" are Qwen code-reasoning elicitation; our OLMo numbers are not directly comparable to Qwen-RLVR headlines.
- **A real organ gain on OLMo is more publishable precisely because OLMo is hard to spuriously move.**

---

## 3. [LOOP] 2502.17416 — looped depth ↔ problem class (the depth/width/macro story)

Notation: **(k⊗L)** = k-layer block looped L times. iso-param baseline **(k⊗1)**; iso-flop baseline
**(kL⊗1)** (same depth, L× more params).

**Provable / empirical depth↔problem map (§2, §5):**

| Problem | Looped result | Depth relation | Maps to our story |
|---|---|---|---|
| **n-ary addition** | (1⊗12) ≈ 99.6–100%, **1/12 the params** of 12-layer; shallow (k⊗1) collapse (1-layer base ~0.1%) (Table 1) | needs **depth**, not params | arithmetic = a *depth* problem if the per-step block has the right primitive |
| **p-hop induction** | looped ~99.5%; **Cor 5.3**: solvable by 1-layer looped **⌊log₂ p⌋+2** times, O(1) width, ≤3 heads (matches lower bound) | depth ↔ **log p** | multi-hop deduction is log-depth — narrow block, many loops |
| **group composition (n elts)** | **Thm 5.1**: 1-layer looped **O(log n)** times, near-optimal depth | depth ↔ **log n** | chain/derive organ: composition is the canonical loop win |
| **i-GSM** (DAG of mod-7 arithmetic, depth 7) | (1⊗8) = **73.2** = iso-flop baseline; looped ≫ iso-param (Table 2) | depth ↔ DAG depth | realistic multi-step math = depth, few params |

**Theory backbone.** Looped L loops can **simulate L steps of CoT** (Thm 5.4 — "latent thoughts"); a
non-looped L-layer model can be simulated by a 1-layer block looped L times with small blowup (Thm 5.2).
CoT = a looped model emitting 1 thought token/iter; looping emits *multiple latent thoughts*/iter.

**LM-scale inductive bias (§3).** Trained on the Pile at 1B: looped models have **worse perplexity** but a
**bias toward reasoning over memorization**:
- % Gap (how much of the iso-param→iso-flop gap looping closes, Eq 1) is **~100%+ on reasoning primitives** (looped *beats* the iso-flop 24-layer baseline at all k, despite 24/k× fewer params) and on math word problems, but only ≈ the perplexity gap on closed-book QA (memorization) (Table 3).
- Downstream acc scales as **α·log(D)**, D = effective depth (Eq 2); relative benefit of loops is higher for reasoning (1.19× for reasoning primitives) (§3.4).
- A **looping-inspired regularizer** (cosine-tie successive blocks) imports the reasoning bias *without hurting perplexity* (§4, Table 4).

**How this informs depth/width/macro:**
- **Decompose the organ's compute as width × depth.** Width = per-step expressivity (one narrowing/derivation step's representational ceiling); depth = number of organ loops (iteration count). [LOOP] says many of *exactly our* tasks (addition, p-hop, composition, i-GSM) are **depth-bound, not width-bound** — a *narrow* block looped suffices. This is direct, published support for **"affine wall = width, depth beatable."**
- **Caveat that sharpens the wall:** depth only beats width *if a single loop of the block can represent one step of the algorithm.* The affine/grade-3 (ternary) structure is about the **per-step width primitive** — looping cannot manufacture a primitive the block lacks. So the right reading is: *the narrowing recurrence buys depth cheaply; the factor/geometric-product width must still encode one sound step.* Depth is beatable; the per-step arity primitive is the width investment that earns its keep.
- **Macro story:** our "organ narrows recurrently *between* OLMo layers, no backprop through repeated OLMo passes" is structurally a looped reasoner welded to a non-looped LM — and [LOOP] is the expressivity license for the looped half. Report an **accuracy-vs-loop-count curve** (their α·log(D) scaling) as the organ's depth signature.

---

## 4. [2SCALE] 2509.23314 — geometry of the recurrent iterates (narrower stability)

**Setup.** GPT-2-scale (12L) with recurrence in three groups (layer 4 self-loop, 5–6 paired, 7
self-loop). Track step Δ^(k)=x^(k+1)−x^(k): step norm ‖Δ^(k)‖ and consecutive-step angle cos∠(Δ^(k),Δ^(k-1)).

**Two-scale picture (§1, §3):**
- **Within a looped block = small-scale refinement.** Step norms **shrink rapidly** (decay within **5–10 loop steps**); consecutive steps become **increasingly orthogonal** (cos∠ rises from noisy/low and settles ≈ **0.5–0.65** in late checkpoints) → updates are *complementary, non-collinear* pushes, tracing a **stable-curvature spiral**, not a repeated push in one direction.
- **Across blocks = coarse drift.** Drift-to-Loop Ratio DLR ≫ 1 at block hand-offs (Fig 2) — the LM layers move the state on a *larger* scale than the in-loop refinements.
- The two scales **separate more over training** (faster norm decay, steadier angles in later checkpoints).

**Early-exit lesson (§4, Algorithm 1).** They compare three halts:
1. **Step-norm** ‖Δ‖<τ (Geiping): cheap O(d) but **mis-halts under the spiral** — norms plateau at a small nonzero value because the direction keeps rotating while magnitude is flat. Sensitive to feature scaling; quality cliff at aggressive τ.
2. **KL on decoded dist** <τ: tied to predictions, scale-invariant, but needs vocab decode O(V) and is slow/calibration-dependent.
3. **Acceleration (theirs)** a^(k)=‖Δ^(k)−Δ^(k-1)‖<τ for **two consecutive steps** (second-order, decoding-free, O(d), Eq 5–6): fires exactly when local curvature stabilizes; best latency-quality trade-off, robust to τ.

**Lessons for our recurrent narrower:**
- **Expect self-stabilization, not divergence.** Step sizes shrink within ~5–10 steps; a recurrent narrower's per-step updates should naturally damp. This is reassuring for stability but also a *warning*: if it converges in too few steps the per-step process-reward signal won't differentiate (mirrors the LSRL r≥8 depth floor in `prior_art_extract.md` flag #3). Budget enough loops *and* check the step-norm decay curve to confirm the organ is actually iterating, not collapsing at step 1.
- **Our exit signal is cleaner than theirs.** Their geometry is a *spiral* (orthogonal refinements) precisely because a generic recurrent block has no monotone scalar to halt on. **Our narrowing is monotone:** candidate-set cardinality only decreases, and the check certifies soundness. So our completeness/cardinality signal is a *better* exit criterion than acceleration — and step-norm halting (which mis-fires under spirals) is a known failure we sidestep by using the discrete monotone quantity. Use **cardinality-plateau (with a two-hit confirm)** as the organ's exit, and keep the second-order acceleration as a cheap geometric sanity cross-check.
- **The two-scale frame validates the weave geometry.** "Organ refines finely *between* OLMo layers; OLMo layers drift coarsely" = exactly the within-loop-refinement / cross-block-drift split, with DLR≫1 expected at the α/γ hand-offs.
- Caveat: their evidence is observational, PCA-2D, single GPT-2-scale model, one recurrence pattern (§5 limitations) — treat as qualitative geometry, not a guarantee.

---

## 5. CONTROL CHECKLIST for the organ-RLVR experiment

**(A) Metrics**
- **Primary: pass@k curve** (k = 1 … 256+), base vs +organ, both SFT and SFT+RLVR. The capability question is *does the curve cross / does the organ lift large-k coverage* ([ELICIT]). pass@1 is necessary but not sufficient.
- Report **per-loop accuracy curve** (acc vs organ loop count) — the α·log(D) depth signature ([LOOP] Eq 2) and the RLTT low-loop-regime check (`rlvr_landscape` finding #1).
- Coverage table (base-solves/organ-fails vs organ-solves/base-fails), à la [ELICIT] Table 2 — the cleanest "did we unlock new problems" statistic.
- Manually verify a CoT subset to exclude guessing on any math slice.

**(B) Base model**
- **Use OLMo** (it moves only under genuine reward — [SPUR]). Do **not** benchmark capability on Qwen-Math (its RLVR gains are partly code-reasoning elicitation).
- **Random-reward + incorrect-label dummy arms** as falsification controls. On OLMo these should be ≈ null; a non-null means the augmented model has an amplifiable shortcut (likely the cold-weave LM-shortcut) — surface it, don't ship it.

**(C) Length / loss-aggregation bias**
- Use **Dr.GRPO or DAPO** token-level loss aggregation (constant/token normalization) to kill GRPO's response-level **length bias** — otherwise pass@1 gains may be longer-output artifacts, not reasoning. (Web: interconnects/GRPO++ — default GRPO inflates length; Dr.GRPO normalizes by a global constant, DAPO by total token count; both in TRL/verl.) Already flagged in `rlvr_landscape.md`.
- Hold **rollout backend, decoding temp, max-tokens identical** across base and +organ arms (the [ELICIT] eval protocol).

**(D) KL**
- **KL trades coverage for stability:** [ELICIT] §4.4 — KL coeff 0.001 keeps pass@1 but tanks pass@128; DAPO/Oat-Zero drop KL entirely. If the *question* is "does the organ raise the ceiling," a KL leash will *hide* a real expansion by pinning the policy to the prior. Run a **KL-free** (or tiny-KL) primary arm; if you need KL for stability, report pass@k *with and without*.

**(E) Sample / credit budget**
- More rollouts (n 8→32) only nudges pass@k ([ELICIT] §4.4) — don't expect n to rescue a null.
- Keep the **process/trajectory-reward arm** (RLTT/LSRL, per `prior_art_extract.md` §4b) so a plain-GRPO null on the organ is *interpretable* (outcome-GRPO credits only the terminal state and falsely nulls a looped organ).
- Beware **pass@k=0 prompts → vanishing GRPO gradients** (web: Berkeley awesome-RLVR-boundary): if no rollout solves a task, all samples are equally bad and the gradient dies. Curriculum / difficulty-filter so the organ sees prompts with non-zero pass@k.

**(F) Depth budget for the organ**
- Give the narrower **enough loops** (≥8-ish, mirroring LSRL r≥8 and [2SCALE] 5–10-step decay) so per-step reward differentiates and the depth-scaling curve is visible. Confirm via the step-norm decay that it's iterating, not collapsing.

---

## 6. CONTRADICTIONS / COMPLICATIONS to our plan

1. **ProRL contradicts the strong "[ELICIT] = RL can't expand" reading** (web: 2505.24864, NeurIPS'25; NVIDIA ProRL-v2). With **prolonged RL + KL-divergence control + reference-policy resetting + diverse verifiable tasks**, both pass@1 *and* pass@k climb over thousands of steps, n-gram overlap with base drops, and the boundary *does* expand — strongest on tasks where the base is weak. The Berkeley awesome-RLVR-boundary survey frames this as the live "pessimist ([ELICIT]) vs optimist (ProRL / 2507.12507)" debate. **Implication for us:** the boundary limit in [ELICIT] is partly an artifact of *short, leashed* RLVR, not a law. Our organ could be the cleaner route to expansion (extra capacity source, like distillation), but the honest comparison must include a *prolonged, KL-controlled, task-diverse* plain-RLVR arm — otherwise we'd be crediting the organ for beating a deliberately weak baseline. Report both the short-RLVR and a ProRL-style long-RLVR control.

2. **The "elicitation" frame can absorb our organ's gain.** If organ+RLVR lifts pass@1 but **not** pass@k, [ELICIT]'s reading says we merely sharpened the base prior. We must show **pass@k expansion** (organ-solves/base-fails > 0, à la Table 2) to claim the organ raised the ceiling — that's the bar, and it's high.

3. **Spurious-reward / clipping bias as a self-deception risk** ([SPUR]). On Qwen we'd fool ourselves; on OLMo we're safer, but the *augmented* model introduces a new amplifiable channel (the γ-readout). The random-reward control is mandatory to prove the organ gain is reward-driven, not clipping-amplified pattern-matching.

4. **Depth helps only if width carries one sound step** ([LOOP] caveat, §3 above). Looping is the cheap half; the affine/ternary primitive is the width we must still pay for. A null at the affine wall under looped RLVR is *expected* if the per-step block lacks grade-3 structure — not evidence against looping.

5. **Convergence vs differentiable credit tension** ([2SCALE] + LSRL floor). The narrower self-stabilizes fast (5–10 steps); too-fast convergence makes per-step process reward un-differentiable (looks like a process-RL null). Budget depth, and read a low-depth process-RL null as expected, not as organ failure.

6. **KL is double-edged across papers.** [ELICIT]: KL suppresses pass@k expansion (drop it to see the ceiling). ProRL: *controlled* KL + reference resets is what makes long RL stable enough to expand. RLTT/LSRL: KL on the terminal step only. Resolution: **KL-free (or tiny) for the capability/pass@k question; controlled-KL + resets only for the long-horizon stability arm** — and never let KL silently hide a coverage change.
