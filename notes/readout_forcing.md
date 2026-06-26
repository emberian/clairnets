# Readout-forcing: making the live-α woven model actually USE the organ

**The blocker (GLaDOS, current):** the live-latent-α woven model
(frozen base + LoRA + latent-α compile + frozen organ + zero-init-γ readout, LM-CE on the
answer) **doesn't engage the readout under naive full-text training.** The LM minimizes
loss via LoRA + prompt text alone, the γ gate stays ≈shut, causal-drop (corrupt/shuffle) ≈ 0.
This is the *reasoning-shortcut theorem* (2305.19951) in the flesh: **the text shortcut is a
genuine optimum of the loss, so by default it WILL be taken.** We need to "shortcut the
shortcutting" — force the hard path — *without* nuking general ability.

**The first principle that orders everything below.** A shut gate over a *useless* organ is
**correct** behavior, not a bug. No forcing technique can succeed if reading the organ
doesn't lower the achievable loss. So the techniques split into two jobs, and they must be
done in this order:
- **(J0) Make reading the organ actually HELP** — i.e. make α's compile good enough that the
  γ-injected state carries answer-relevant information the text path lacks. (Root cause.)
- **(J1) Remove/penalize the easy path so the model is forced onto the (now-useful) hard one.**
  (Symptom.)

Doing J1 before J0 produces a cargo-cult: the model can be *made* organ-sensitive without
being *correct-via-organ*. Every J1 technique below is only genuine when paired with the
causal controls (corrupt/shuffle/permute → accuracy collapses) *and* clean-organ accuracy up.

---

## TL;DR — the three to try first

1. **Supervise α directly against the exact compiled program (J0).** The Topan SATNet *fix*
   (2106.11072): don't let the grounding map learn only from the distal answer loss — give it
   its own grounding objective. We have free exact labels (witness generator + `exact_dedP`),
   so add an α-target loss (α-output ≈ ground-truth typed factor graph) as warmup + a standing
   auxiliary term. **Cheapest high-leverage move; attacks the root cause.** If oracle-γ passes
   (corrupt→0) but live-α doesn't engage, α is the problem by elimination — fix it first.
2. **Fact-ablated generation stream (two-stream input asymmetry, J1).** α reads full text;
   the residual stream feeding the LM-head has the *answer-bearing facts masked*, so the only
   route to the answer is through γ. This is the SATNet output-masking control (2312.11522)
   turned constructive + the "make the shortcut feature unavailable" doctrine
   (Geirhos 2004.07780). Cheap (a forward-pass/data change), and **retention-safe if scoped to
   organ-task batches only**.
3. **Corrupt-consistency auxiliary loss (J1, directly optimizes our metric).** Add
   `λ·max(0, m − D(p_clean ‖ p_corrupt))` on the answer span: punish the model when corrupting
   the injected lattice *fails* to change the output. This is "right for the right reasons"
   (1703.03717) / attribution regularization (1905.09957) applied to the γ channel — it makes
   causal-drop a training target, not just a diagnostic. One extra (corrupted) forward.

Lead bet: **#1 + #2 together** (good organ × unavailable shortcut), with #3 as the
load-bearingness regularizer and the corrupt/shuffle controls gating every claim.

---

## TIER 1 — cheapest + most promising

### T1.1 — Supervise α to standalone grounding competence (the SATNet *fix*) — J0
- **Mechanism.** Topan et al., *Techniques for Symbol Grounding with SATNet* (**arXiv 2106.11072**,
  NeurIPS 2021) is the constructive answer to the Chang grounding critique (2312.11522): SATNet
  only grounds when the perception→symbol map is trained with its *own* signal, not just the
  end-to-end answer loss. Their two tools: (a) **self-supervised pre-training of the
  symbol/perception layer** (cluster inputs into symbols, learn the bijection) to standalone
  competence *before* attaching the solver; (b) a **"proofreading"** pass that re-checks and
  re-grounds the symbol assignment. Result: visual-Sudoku grounding recovered to ~full accuracy
  where naive end-to-end SATNet sits at chance.
- **Maps to us.** α (OLMo hidden → organ input tensors) is exactly SATNet's grounding map, and
  it is the *un-precedented, hardest* half (prior_art_extract flag #1, nesy_lessons). We have
  what SATNet lacked: **free exact labels** for α's target — the witness generator + `exact_dedP`
  produce the ground-truth typed factor graph. So: (1) **pretrain α supervised** (α-output ≈
  ground-truth compiled program) standalone, like Stage-1 for the organ; (2) keep a **standing
  α-target auxiliary loss** during the weave so α can't drift into garbage that the LM then
  rightly ignores; (3) the organ's own soundness check is our free "proofreader." This is
  already gestured at in training.md/prior_art_extract ("pretrain α supervised standalone; do
  NOT expect α to emerge from co-training") — 2106.11072 is the *citation that this is THE fix*
  for the grounding-shortcut, not an optional nicety.
- **Cost/risk.** Low–medium: one extra supervised objective on data we already generate. Risk:
  α's latent compile must be *expressive enough* to hit the symbolic target; if it can't, that's
  a capacity finding, not a forcing failure (and better to learn it now than to chase the gate).
- **Retention.** Neutral-to-positive: α is a new module, base frozen; supervising it doesn't
  touch general LM ability.
- **Genuine vs shift?** **Most genuine of all** — it makes the organ useful, which is the
  precondition for every other technique to be real rather than gamed. Lowest "just shifts the
  shortcut" risk.

### T1.2 — Fact-ablated generation stream (two-stream input asymmetry) — J1
- **Mechanism.** The decisive lesson of the shortcut-learning literature (Geirhos et al.,
  *Shortcut Learning in Deep Neural Networks*, **arXiv 2004.07780**) and the SATNet collapse is:
  **the cleanest way to force the hard feature is to make the easy one UNAVAILABLE.** Chang et al.
  (2312.11522) did this as a *diagnostic* — "output masking" removed the leaked-label channel and
  visual-Sudoku dropped 18.5%→0% (proving the leak); make that *constructive* and it becomes a
  forcing objective. Two-stream: **α reads the full text** (it needs the facts to compile), but
  the residual stream that feeds the **LM-head has the answer-bearing facts ablated/masked**, so
  the only path from problem→answer runs through the γ-injected organ state. This is the same
  doctrine as forcing context-faithfulness over parametric priors via counterfactual/absent
  context (*Context-faithful Prompting*, **2303.11315**; Context-DPO, 2024).
- **Maps to us.** Generalizes the planned **cells-mode curriculum** (training.md: "no facts in
  the prompt → the organ is the ONLY route") from a discrete early phase to a *continuous,
  standing* asymmetry you keep partially on even during full-text training. Concretely: a mask /
  separate ablated copy of the token stream feeding the final LM-head + γ, while α keeps the
  unmasked stream. The split-brain result ("LM wields the organ 99.9% over adversarial text")
  is the same idea already validated on oracle-γ.
- **Cost/risk.** Cheap — a forward-pass plumbing + data-masking change, no new loss, no extra
  forward. **Risk:** if ablation is *leaky* (LoRA reconstructs the masked fact from un-ablated
  tokens / positional cues), the shortcut returns — must verify with shuffle/corrupt that
  accuracy still collapses. This is the main "shifts not forces" trap: the model may learn to
  *infer the ablated fact* rather than read the organ.
- **Retention.** **Safe by construction IF scoped** — apply ablation only on organ-task batches;
  pass general text (Dolma) through untouched. Ablating general text would gut the base skill.
- **Genuine vs shift?** Genuine when ablation is complete; the failure mode is fact-reconstruction
  from residual cues, caught by the controls.

### T1.3 — Curriculum staging (Coconut-style cells→text anneal) — J1
- **Mechanism.** Coconut (**2412.06769**) makes a *latent* reasoning module load-bearing under
  ordinary LM loss **only with a multi-stage curriculum** that progressively replaces text steps
  with latent ones; **remove the curriculum and the module is ignored** (GSM8K 34.1→14.4, ProntoQA
  99.8→52.4). This is the cleanest published analogue of our cold-weave failure: the gate forms
  *before* the LM can camp in the text-shortcut basin.
- **Maps to us.** Exactly the planned readout-forcing schedule (training.md): cells-mode first
  (organ-only route) → anneal in full text. Order matters; the γ-decode must form first.
- **Cost/risk.** Cheap (schedule only). **Risk — important:** the shortcut can **re-emerge when
  text is annealed back in** (gate re-closes). Mitigation: keep T1.2 ablation *partially on
  permanently*, and **monitor corrupt→0 through the entire anneal**, not just at the end. Treat a
  re-opening shortcut as a curriculum-schedule failure, not a model verdict.
- **Retention.** Safe (organ task only).
- **Genuine vs shift?** Genuine early; the anneal phase is where it can silently revert — guard it.

---

## TIER 2 — stronger, more involved

### T2.1 — Distill FROM the proven module-using teacher (oracle-γ → live-α) — J0/J1
- **Mechanism.** Distillation expands the capability boundary where RL cannot
  (**2504.13837**, Fig. 7): supervised next-token on a *better* computation's traces moves
  capacity. We already have a teacher in which the organ is causally load-bearing — the
  **oracle-readout γ checkpoint** (corrupt→0, OOD-perfect). Distill its **output distribution
  and/or γ-injected hidden state** into the live-α student, so the student is supervised to
  *match the organ-conditioned computation*, not merely emit the answer. The supervision target
  is no longer the bare answer (which the text path can fit), so the easy path no longer
  satisfies it.
- **Maps to us.** Teacher = oracle-lattice-γ model; student = latent-α model; add a hidden-state /
  logit KD term aligning the student's post-γ residual to the teacher's.
- **Cost/risk.** Medium (teacher forward per batch). **Risk:** the teacher uses the *ground-truth*
  lattice; the student's α must compile a good-enough lattice or the KD target is unreachable —
  **α-correctness is still the ballgame** (reduces to T1.1). Distilling onto narrow data can
  forget; mix general text.
- **Retention.** KD is a known regularizer (can *help* retention), but narrow-distribution KD
  forgets — interleave Dolma.
- **Genuine vs shift?** Only as genuine as α's compile; a bad α distills a model that *mimes*
  organ-use. Pair with controls.

### T2.2 — Bias-model / Product-of-Experts debiasing (force the residual onto the organ) — J1
- **Mechanism.** Train a deliberate **shortcut-only model** and make the main model fit only what
  the shortcut *can't*. Two recipes: **Product-of-Experts** (Clark et al., *Don't Take the Easy
  Way Out*, **arXiv 1909.03683**; He et al. 1908.10763; Karimi-Mahabadi 1909.06321) — combine
  bias-model and main-model logits in training so the main model is rewarded only for the
  residual; and **Learning-from-Failure** (Nam et al., **arXiv 2007.02561**) — up-weight the
  examples the bias model gets *wrong* (the non-shortcutable ones).
- **Maps to us.** The "bias model" is the **γ-off forward** (LoRA + text only); the "robust model"
  is the full γ-on forward. Penalize/marginalize the γ-off path's easy wins so gradient flows to
  γ on exactly the instances text can't solve (novel numbers, under-specified-without-deduction).
- **Cost/risk.** Medium; PoE/LfF were built for classification and need care under
  autoregressive CE. **Key risk:** the shortcut and the robust path **share the same LoRA/network**
  here (unlike a separate bias net), so penalizing the easy path can just rebalance *within* the
  net rather than route to the organ.
- **Retention.** The shared LoRA carries general ability → penalizing it risks forgetting; scope
  strictly to organ-task batches.
- **Genuine vs shift?** Medium; the shared-substrate issue makes "shift within the net" a real
  hazard. Less clean than T1.2's hard unavailability.

### T2.3 — Roadblocks / adversarial-text augmentation — J1
- **Mechanism.** *Roadblocks for Temporarily Disabling Shortcuts and Learning New Knowledge*
  (NeurIPS 2022, OpenReview **QjurhjyTAb**): **gently modify the task so the learned shortcut is
  insufficient**, forcing the net to discover additional features. Generalizes cells-mode to a
  *continuum* of text perturbations (paraphrase, drop/reorder spans, inject distractor facts,
  per-instance novel numbers) so surface statistics under-determine the answer.
- **Maps to us.** Cheap data augmentation on the organ-task prompts; the adversarial-text
  split-brain regime we already validated.
- **Cost/risk.** Low (augmentation). Risk: too-gentle a perturbation leaves the shortcut intact;
  too-harsh corrupts the problem.
- **Retention.** Safe (organ task only).
- **Genuine vs shift?** Genuine in proportion to how non-shortcutable the perturbed data is.

---

## TIER 3 — supporting / lower priority

- **Randomized positional encoding (TransNAR §6, 2406.09308).** Without it the hybrid is
  *thresholded by the base LM's* OOD score — a candidate cause of cold-weave (the LM dragging the
  organ down). **Cheap, already in the adopt-list; turn it on.** Retention-neutral.
- **Zero-init γ gate + structured dense readback (TransNAR; LLaMA-Adapter 2303.16199).** Already
  in the design — keep it; it's the "reachable but protected" primitive that also *protects base
  knowledge* (retention-positive). Not a *forcing* lever on its own (we observe the gate staying
  shut), but a precondition.
- **IFM / adversarial feature removal (2106.11230).** Adversarially perturb the LM hidden features
  the shortcut uses (closed-form: shift positives away from / negatives toward the anchor) so easy
  features become insufficient — the *continuous* cousin of T1.2 fact-ablation. More exotic to
  port to autoregressive CE; lower priority than hard masking.
- **MDL-probing the shortcut (Voita & Titov, 2003.12298).** Diagnostic, not a forcing lever:
  measure how *cheaply extractable* the answer is from prompt text vs from the γ state. If the
  text answer has low description length, the shortcut is near-irresistible and you *need* T1.2
  unavailability, not just a curriculum.

---

## RETENTION (angle 4) — forcing the skill WITHOUT catastrophic forgetting

The good news: **most J1 forcing above is retention-safe by construction** because it is *scoped
to organ-task batches* — general text flows through the un-ablated, un-penalized path, and the
**frozen base + zero-init γ** structurally protect base knowledge. The retention work is mostly
about the *mix*, not exotic CL machinery:

- **Replay / mix-ratio (the workhorse).** Interleave general text (Dolma) with organ-task
  batches — this is the planned Stage-3 consolidation ("Dolma ⊕ reasoning corpus, tuned ratio").
  Empirically **naive replay beats most sophisticated CL methods** (continual-learning practice;
  survey 2501.13669 *How to Alleviate Catastrophic Forgetting in LLM Finetuning*). Tune the ratio;
  this is the single most reliable retention knob.
- **LoRA-as-skill + frozen base.** LoRA already isolates the new skill from frozen base weights →
  general ability is structurally protected; keep rank modest and never thaw the base. (Caveat:
  shared LoRA across organ-task and general text can still drift — hence scope J1 penalties to
  organ batches so the general-text gradient stays clean.)
- **EWC / regularization-based CL (Kirkpatrick, 1612.00796).** Penalize movement of
  Fisher-important LoRA params. Cite-and-keep-in-reserve: usually *under-performs replay* in LLM
  practice, more knobs. Use only if replay alone is insufficient.
- **Outcome-dominant mixing for any RL phase.** If RLVR/process-reward (RLTT/LSRL) enters, keep
  the outcome reward primary (0.7/0.3) and KL to a frozen ref — standard anti-forgetting for the
  alignment phase.

**Threat assessment of the forcing techniques to retention:**
- T1.1 (α-supervision), T1.2 (fact-ablation, scoped), T1.3 (curriculum), T2.3 (roadblocks):
  **low threat** — new module / organ-task-scoped / base frozen.
- T2.1 (distillation), T2.2 (PoE/LfF): **medium threat** — narrow-distribution training and
  shared-LoRA penalties can forget; **must** interleave replay.

---

## Skeptic's ledger — which techniques genuinely force vs merely shift the shortcut

| Technique | Forces the hard path? | The way it can secretly *shift* the shortcut |
|---|---|---|
| **T1.1 α-supervision (J0)** | **Genuinely** — makes the organ useful (precondition for all else) | Can't shift; worst case reveals α-capacity limit. Lowest risk. |
| **T1.2 fact-ablation (J1)** | Genuinely *iff* ablation is complete | Model **reconstructs the ablated fact** from residual/positional cues → shortcut returns. Catch with shuffle/corrupt. |
| **T1.3 curriculum (J1)** | Genuinely *early* | Shortcut **re-emerges on text-anneal** (gate re-closes). Monitor corrupt→0 throughout; keep some ablation permanent. |
| **T2.1 distillation** | Only as genuine as α | Bad α → student **mimes** organ-use, copying teacher logits via text. |
| **T2.2 PoE/LfF** | Partially | Shared LoRA → **rebalances within the net** instead of routing to γ. |
| **T2.3 roadblocks** | Proportional to non-shortcutability | Too-gentle perturbation leaves the shortcut intact. |

**The discipline (unchanged from nesy_lessons):** *soundness prevents wrong answers; it does NOT
prevent bypass.* Only the **corrupt/shuffle/permute controls collapsing accuracy** prove the organ
is load-bearing. Gate every "it's working" claim behind those controls biting, report over seeds,
and watch for the SATNet two-stage tell (accuracy pinned at floor until LoRA/text quietly solves
it). The reasoning-shortcut theorem guarantees the easy path *exists*; our job is to make it
*unavailable* (T1.2), *insufficient* (T1.3/T2.3), *unrewarding* (T2.2/T3-corrupt-loss), or
*worse than the now-useful organ* (T1.1) — and to *prove* we did with the controls.

---

### Sources (arXiv ids)
- **SATNet** — Wang et al., ICML 2019, **1905.12149** (pdfs/).
- **SATNet grounding critique** — Chang et al., NeurIPS 2020, **2312.11522** (pdfs/).
- **SATNet grounding FIX** — Topan et al., *Techniques for Symbol Grounding with SATNet*,
  NeurIPS 2021, **2106.11072** (self-supervised perception pretraining + proofreading).
- **Reasoning-shortcut theorem** — Marconato et al., NeurIPS 2023, **2305.19951**; continual
  variant **2303.12578**.
- **Shortcut learning (survey/doctrine)** — Geirhos et al., **2004.07780**.
- **Implicit Feature Modification** — Robinson et al., NeurIPS 2021, **2106.11230**
  (adversarial feature removal in InfoNCE).
- **Roadblocks** — *Roadblocks for Temporarily Disabling Shortcuts and Learning New Knowledge*,
  NeurIPS 2022, OpenReview **QjurhjyTAb**.
- **Learning from Failure (LfF)** — Nam et al., NeurIPS 2020, **2007.02561**.
- **Product-of-Experts debiasing** — Clark et al. **1909.03683**; He et al. **1908.10763**;
  Karimi-Mahabadi et al. **1909.06321**.
- **Context-faithfulness forcing** — Zhou et al., *Context-faithful Prompting*, **2303.11315**;
  Context-DPO (2024).
- **Right for the Right Reasons / attribution reg** — Ross et al., IJCAI 2017, **1703.03717**;
  Robust Attribution Regularization, **1905.09957**.
- **MDL probing** — Voita & Titov, **2003.12298**.
- **TransNAR** — **2406.09308** (zero-init gate, frozen organ, randomized PE, dense readback).
- **Coconut** — **2412.06769** (latent module needs curriculum or it's ignored).
- **LLaMA-Adapter** — **2303.16199** (zero-init gated attention).
- **Distillation expands boundary** — Yue et al., NeurIPS 2025, **2504.13837** (Fig. 7).
- **Procedural data / front-loading / get-go** — **2411.12580**, **2510.03264**, **2502.19402**.
- **Retention** — EWC, Kirkpatrick **1612.00796**; LLM-finetuning forgetting survey **2501.13669**;
  spurious-rewards RLVR caution **2506.10947** (pdfs/).
