# Real-training landscape: RLVR tasks + recipes for the organ (research map, mid-2026)

## Two findings that MUST shape the organ's RL test
1. **Outcome-GRPO falsely nulls an organ.** Looped/recurrent models (Ouro arXiv 2510.25741, RLTT 2602.10520) found vanilla
   GRPO failed to beat SFT — GRPO rewards only the FINAL state, so credit never reaches the organ's internal steps. RLTT's
   trajectory/process reward recovered +5.8%(1.4B)/+10.9%(2.6B). => the fair organ test NEEDS a process/trajectory-reward arm;
   a plain-GRPO null is the EXPECTED failure mode, not evidence against the organ.
2. **Use OLMo as base (honest control).** "Spurious Rewards" (2506.10947): random/wrong rewards recover most RLVR gain on
   QWEN-Math but NOT on Llama-3/OLMo => Qwen RLVR partly measures pretraining. OLMo (ours) is the interpretable control.
   Also: RLVR mostly ELICITS not expands (2504.13837 — raises pass@1, lowers pass@k); report pass@1 AND pass@k.

## Premise vindicated
Logic-RL (2502.14768): a 7B trained ONLY on procedural Knights&Knaves transferred to AIME/MATH. Synthetic verifiable
reasoning induces transferable reasoning — our CSP/deduction tasks are legitimate RL fuel.

## Fuel (verifiable, exact-reward)
- **reasoning-gym** (github/open-thought/reasoning-gym, arXiv 2505.24760) ★ — 100+ procedural generators + exact verifiers,
  infinite instances, difficulty/OOD knobs; score_answer() exact. Includes sudoku/zebra/propositional-logic/cryptarithm/
  countdown/maze/shortest_path/arc. RL on it transferred +9.7% MATH, +7.66% BBH. CLOSEST standardized home for our CSPs.
- **ZebraLogic** (2502.01100) — 1000 logic-grid CSPs, SAT-solver unique-solution, grid-size knob. **SATBench** — 2100 from
  random CNF, SAT-solver oracle. **TinyZero/Countdown** — canonical R1-Zero task. **NPHardEval** — monthly-regenerated, P/NP tiers.
- math: DeepScaleR-Preview (~40K, the 1.5B->43% AIME run), Big-Math-RL-Verified (251K), MATH-500/AIME (evals, keep clean),
  Math-Verify for grading. code: CodeContests/TACO/KodCode (+EvalPlus-grade tests), LiveCodeBench (eval only).
- multi-domain ready: **guru-RL-92k** (LLM360, 91.9K across math/code/logic/table/sim/science, exact verifiers) ; Tulu-3 RLVR mix.
- formal (kernel reward): miniF2F, ProofNet, Lean-Workbook (57K+83K), LeanDojo gym. NL-logic: ProofWriter/FOLIO (cheap, shallow).

## Machinery
- **TRL GRPOTrainer** = easiest for OUR case: first-class NON-vLLM rollout (runs custom generate() + custom PreTrainedModel),
  reward = plain Python callable (reward_funcs), LoRA+GRPO on 1-2B fits one 48GB. Dr.GRPO/DAPO to kill length-bias.
- verl (HybridFlow) = scale-out when >1 GPU (needs dtensor_weight_loader or vLLM Transformers backend). OpenRLHF (remote
  reward server). open-instruct (Tulu/OLMo RLVR-native, AI2-stack). NeMo-RL (grpo_math_1B config). PRIME (dense implicit
  process reward — relevant to finding #1).
- Recipe floor: TinyZero — 1.5B LEARNS, 0.5B COLLAPSES (~$30/<5h/2GPU). Tina — LoRA-GRPO 1.5B -> 43% AIME for ~$9.
  LoRA enough for RLVR (Thinking Machines "LoRA Without Regret"; all layers incl MLP, ~10x LR).

## Organ↔rollout integration (the real cost)
vLLM only runs REGISTERED archs. Routes: (A) vLLM Transformers backend (single-pass custom modeling_*.py, _supports_attention_backend);
(B) out-of-tree vLLM registration (rewrite vs paged-attn — heavy); (C) **skip vLLM (use_vllm=False) — the ONLY route for a
latent/dynamic-depth organ loop**, at throughput cost (fine at 1-2B). Shortcut: if organ == LoRA delta -> vLLM multi-LoRA (fast).

## Decisive experiment (recommended)
matched 2x2 {OLMo base, OLMo+organ} x {SFT, SFT+RLVR}, lightly-SFT'd 1.5-2B base (RL headroom), SAME rollout backend both arms,
LoRA-GRPO (Dr.GRPO) single L40S, fuel=reasoning-gym(primary)+ZebraLogic/SATBench(CSP transfer)+MATH slice(cross-domain).
+ ONE process/trajectory-reward arm (RLTT/PRIME-style) so a GRPO null is interpretable. Metric: OOD exact-match pass@1 AND pass@k.
Question: does the organ raise the ceiling under PROPERLY-CREDITED RL, or does outcome-RLVR flatten base and organ together.
