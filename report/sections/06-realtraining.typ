#import "../helpers.typ": *

= Toward real training, and the decisive open experiment

The toy is built and it works — but it is profoundly *undertrained* (1B host, synthetic CSPs). The honest
next step is real RLVR on standardized verifiable reasoning, with the organ as a *tested* component, on a
base model where a gain *means* something.

== The RLVR pivot

Two literature findings reshape *how* we test the organ before *whether* it helps:

- *Use OLMo as the base.* "Spurious Rewards" (#arxiv("2506.10947")) shows random/wrong rewards recover most
  RLVR gain on Qwen-Math but *not* on Llama/OLMo — Qwen's RLVR partly measures pretraining. On OLMo a gain
  is real signal. We standardize on *OLMo-3-Base-7B* (Base, not Instruct: RL headroom + an honest control;
  our 7B$approx$1B coupling finding means 7B buys LM competence, not better reading). A random-reward and
  incorrect-label *dummy arm* is mandatory — on OLMo it should do ~nothing; if the organ-augmented model
  suddenly gains under random reward, that exposes an amplifiable shortcut.
- *Report pass\@k, not just pass\@1, with a ProRL-style control.* RLVR mostly *elicits* rather than *expands*
  (#arxiv("2504.13837")): it raises pass\@1 and can *lower* pass\@k. The capability question is whether the
  organ behaves like *distillation* (lifts pass\@k, expands the boundary) or like plain RLVR (sharpens the
  prior). A prolonged, KL-controlled, task-diverse RLVR arm (ProRL) is needed so we do not credit the organ
  for beating a deliberately weak baseline.

The substrate is de-risked: the base TRL-GRPO loop (`rlvr_pipeline.py`) *runs and learns* on one L40S
(reward EMA 0.20$arrow.r$0.72 over 250 steps on `chain_sum`), with `use_vllm=False` for an organ-compatible
dynamic-depth rollout, Dr.GRPO, $beta = 0$, LoRA all-linear. The organ slots in by swapping the policy
model; reward, dataset, and config are unchanged. Fuel is reasoning-gym (#arxiv("2505.24760")) plus
ZebraLogic (#arxiv("2502.01100")) / SATBench for CSP transfer and a MATH slice for cross-domain. The premise
is vindicated by Logic-RL (#arxiv("2502.14768")): a 7B trained *only* on procedural Knights-and-Knaves
transferred to AIME/MATH.

== The organ as process reward

Because outcome-only GRPO *falsely nulls* a looped organ (it never credits internal steps), a
process/trajectory-reward arm is not optional — it is what makes a GRPO null *interpretable*. And the organ
*is* the grader: per-step progress = candidate-set cardinality drop, per-step quality = soundness of the
narrowing. We adopt both published arms (RLTT broadcast, parameter-free; LSRL exact-graded), honoring the
depth floor (process supervision is null below ~$r = 8$ recurrent steps) and keeping the outcome reward
primary (0.7 outcome / 0.3 process) so the per-step signal shapes rather than dominates.

== The decisive open experiment: architecture or trick?

#keynote[
*The single most decisive experiment.* The organ supplies *correctness*; next-token CPT optimizes
*likelihood*. On generic prose the two diverge, so the gradient that would route generation through the
organ may be absent (the cold-weave failure). *Does ordinary next-token continued-pretraining exploit the
organ — making it a general architecture — or does only targeted RL open the gate — making it a trick?* No
published work tests next-token CPT on a *checked deductor*.
]

The design is a *2×3 grid*: {next-token CPT, process-RLVR} × {reasoning-dense + non-shortcutable,
reasoning-dense + shortcutable, general prose}. Starting from a checkpoint M★ where the organ is already
causally load-bearing (staged recipe), continue-train with *ordinary next-token loss only* and track the
*exploitation gap* $G(t) = #raw("loss")_#text("off") - #raw("loss")_#text("on")$ — next-token loss with the
organ zeroed vs on, over CPT steps. *Exploited* = $G(t)$ grows monotonically and concentrates on
deduction-determined (low-entropy, verifier-checkable) tokens, the $gamma$ gate opens from ~0, the
corrupt$arrow.r$0 penalty *rises*, and OOD verifiable accuracy improves *with no verifier reward in the
loop*. *Null (RL-only trick)* = $G(t)$ flat/shrinking and the gate stays shut while the process-RLVR arm on
the same data *does* open it. The literature is genuinely split here: the get-go position (#arxiv("2502.19402"),
which proposes our exact reasoner+memory architecture and bets on RL) and SFT-memorizes/RL-generalizes
(#arxiv("2501.17161")) predict the trick; procedural-knowledge (#arxiv("2411.12580")), front-loading
(#arxiv("2510.03264")), and distillation (#arxiv("2504.13837") Fig. 7) predict the architecture — *given the
right (reasoning-dense, non-shortcutable) data.* The 2×3 grid adjudicates it on our own organ.

== Future architectures

Two host shapes marry the organ better than a dense autoregressive stack:

- *DiffusionGemma — organ ↔ denoising.* A diffusion/denoising LM already runs an *iterative refinement* over
  the whole sequence; the organ's monotone narrowing is a natural per-step constraint *projection* inside the
  denoising loop (narrow the candidate set, then denoise toward it). The organ's recurrence and the model's
  iteration become *one* loop rather than two nested ones — and the monotone exit signal gives the denoiser a
  principled, sound stopping criterion.
- *MoE — organ as a routed expert.* In a mixture-of-experts host, the organ is one expert the router can
  dispatch to on reasoning-dense tokens. This makes "use the organ" a *learned routing decision* trained by
  the same load-balancing machinery, and it bounds the organ's cost to the tokens that need it — a clean
  answer to "when should the model reason through the organ" that the router learns rather than the gate.

The consolidation endpoint remains the same in all cases: re-run OLMo's own open recipe
(Dolma continued-pretrain + instruction-tune + RL-Zero) *with the organ in place*, so organ-use folds into
general behavior rather than staying a bolted-on skill. Whether *ordinary* training does that is the
architecture-vs-trick question above — and it is the experiment that decides what GLaDOS actually is.