# Reading list / related work

PDFs live in `pdfs/`. Extracted implementable recipes → `notes/prior_art_extract.md` (TransNAR/RLTT/LSRL).
The honest map of *our* space is the neuro-symbolic-LLM survey + awesome-lists at the bottom.

## Our foundation
- **LDT — Lattice Deduction Transformer** `pdfs/2605.08605-...` ([arXiv](https://arxiv.org/abs/2605.08605)) — recurrent
  transformer that projects latent through a candidate-set lattice; 800k params, ~100% Sudoku-Extreme. The architecture we weave.
- **CliffordNet** `pdfs/2601.06793-...` — geometric product; grades↔arities for the factor deductor.
- **Transformers Learn Shortcuts to Automata** `pdfs/2210.10749-...` ([arXiv](https://arxiv.org/abs/2210.10749)) — O(log T)
  depth shortcuts via transition-monoid composition. Grounds macro-deduction + "affine wall = width".

## Closest prior art — coupling an LLM to a reasoner (STUDY)
- **TransNAR — Transformers meet Neural Algorithmic Reasoners** `pdfs/2406.09308-transnar.pdf`
  ([arXiv 2406.09308](https://arxiv.org/abs/2406.09308)) — LLM cross-attends a *pretrained* NAR; big OOD-reasoning gains.
  Our direct cousin; the coupling mechanism is the thing to lift.

## Process-reward for latent/looped reasoning (ADOPT — fixes the organ-RL credit-assignment)
- **RLTT — Rewarding Latent Thought Trajectories** `pdfs/2602.10520-rltt-latent-trajectory-reward.pdf`
  ([arXiv 2602.10520](https://arxiv.org/html/2602.10520v1)) — aligns RLVR with the multi-step latent computation of looped LMs;
  +5.8%(1.4B)/+10.9%(2.6B) over vanilla GRPO. The fix for "outcome-GRPO can't credit the organ's internal steps."
- **LSRL — Process-Supervised GRPO on Latent Recurrent States** `pdfs/lsrl-process-grpo-latent-states.pdf`
  ([EMNLP'25 Findings](https://aclanthology.org/2025.findings-emnlp.669/)) — process supervision on latent recurrent states; the
  concrete recipe to credit internal narrowing steps.
- **Reasoning with Latent Thoughts: Power of Looped Transformers** `pdfs/2502.17416-...` ([arXiv 2502.17416](https://arxiv.org/abs/2502.17416)).
- **Two-Scale Latent Dynamics for Recurrent-Depth Transformers** `pdfs/2509.23314-...`.

## RLVR design — what to control for (so a null is interpretable)
- **RLVR mostly Elicits, doesn't Expand** `pdfs/2504.13837-...` ([arXiv 2504.13837](https://arxiv.org/abs/2504.13837)) —
  RLVR raises pass@1, lowers pass@k. Report both.
- **Spurious Rewards** `pdfs/2506.10947-...` ([arXiv 2506.10947](https://arxiv.org/abs/2506.10947)) — random/wrong rewards
  recover gains on *Qwen-Math* but NOT Llama/OLMo. Use OLMo as the honest base.
- Full RLVR task+recipe map: `notes/rlvr_landscape.md` (reasoning-gym, TRL GRPO, Dr.GRPO/DAPO, the matched 2×2 + process-reward arm).

## The "does next-token training exploit it" question (open; the architecture-vs-trick decider)
- **Procedural knowledge in pretraining drives reasoning in LLMs** — pretraining-data procedural knowledge drives reasoning.
- **General Reasoning Requires Learning to Reason from the Get-go** ([arXiv 2502.19402](https://arxiv.org/abs/2502.19402)).
- Test: after RLVR establishes organ-use, continued next-token training + ablation — does next-token loss on reasoning text rise when the organ is zeroed?

## Maps of our space (CRAWL for novelty-positioning + more reading)
- **Neuro-Symbolic AI: Towards Improving Reasoning of LLMs** (survey) ([arXiv 2508.13678](https://arxiv.org/abs/2508.13678)).
- **Awesome-LLM-Reasoning-with-NeSy** (LAMDA-NeSy) — curated NeSy×LLM list. → crawled subset in `notes/nesy_landscape.md`.
- **reasoning-gym** ([github](https://github.com/open-thought/reasoning-gym), [arXiv 2505.24760](https://arxiv.org/abs/2505.24760)) —
  has a `/training` RL subdir; our primary task source.
