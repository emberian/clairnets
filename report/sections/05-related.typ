#import "../helpers.typ": *

= Related work and the novelty delta

GLaDOS is a *combination*, not a from-nothing primitive — every ingredient has a precedent, and being honest
about which is the whole point of the honesty banner. What no one holds is the *five-way* combination, and
what no one else *ran* is the discipline. We position against the four nearest neighbors.

== TransNAR — the coupling cousin (but soft, and it skips the hard half)

TransNAR (#arxiv("2406.09308")) couples an LLM to a *frozen, pretrained* graph reasoner via zero-init gated
cross-attention, trained with *pure supervised next-token, no RL*, for >20% absolute OOD gains. It validates
our coupling design almost line-for-line: gate init closed, read node *and* edge structure by concat+linear
(pooling loses), reasoner frozen during coupling, and — a fix it handed us — *randomized positional
encoding is load-bearing*: without it the hybrid is thresholded by the base LM's OOD score, a candidate
cause of our cold-weave shortcut. Two deltas: TransNAR's reasoner is *soft* (a CLRS-pretrained net, wrong
OOD — its index/search failures are exactly that), where ours is *output-checked $arrow.r$ sound*; and its
reasoner ingests a clean external graph the LLM never produces. Our $alpha$ (hidden$arrow.r$typed program)
is precisely the input-production step TransNAR sidesteps — the unprecedented half.

== Coconut — our mechanism, emergent and unchecked

Coconut (#arxiv("2412.06769")) makes a *latent* "continuous thought" loop load-bearing under ordinary LM
loss — *but only with a multi-stage curriculum* that progressively replaces text CoT with latent thoughts.
*Remove the curriculum and the module is ignored* (GSM8K 34.1%$arrow.r$14.4%; ProntoQA 99.8%$arrow.r$52.4%).
This is the cleanest external analogue of our cold-co-train failure *and* of our staging fix: LM loss *can*
exploit a reasoning module, but a naive single-stage objective collapses to the shortcut. The delta:
Coconut's latent loop is a *free-form superposition-elimination* with no check — exactly our mechanism, but
*emergent and unverified*. Ours is checked, so its degradation is *abstention*, not silent error.

== SATNet — the cautionary tale, and the discipline lesson

SATNet (#arxiv("1905.12149")) was a differentiable MAXSAT layer that appeared to learn logic end-to-end,
including "visual Sudoku" with no digit labels. The grounding critique (#arxiv("2312.11522")) showed the
"learned logic" was *label leakage*: the loss term on input bits backpropagated the true digit labels
straight into the CNN through an unmasked side channel. With that channel masked, visual Sudoku collapses to
*0.0% ± 0.0%* over 10 seeds; a plain FC net *beat* the SATNet model (72.1%). The reasoning-shortcut theorem
(#arxiv("2305.19951")) then proved the general result: *reasoning shortcuts are optima of the loss* — a NeSy
model can attain perfect accuracy with unintended concept semantics, right for the wrong reasons.

#keynote[
*The discipline lesson.* SATNet is the field's most reproducible result about systems like ours, and it is a
*negative*: a *sound* organ can still be *bypassed*, and end-to-end differentiability *hides* it. The
soundness check is therefore *necessary but not sufficient* — it buys correctness-given-the-organ, never
load-bearingness. Our edge over the whole field is not the organ; it is the *causal controls
(corrupt / shuffle / permute)* — exactly the controls SATNet did not run — gating every claim, plus
isolating $alpha$ with its own grounding test before co-training, and reporting over seeds. The dense-$gamma$
oracle result is the SATNet test *passed*.
]

== RLTT / LSRL — process reward, and the grader they wanted is our organ

The process-reward line fixes a credit-assignment bug we will hit: outcome-only GRPO credits *only the
terminal state* and so *cannot reward an organ's internal steps* (RLTT, #arxiv("2602.10520"); LSRL, Findings
EMNLP 2025). RLTT broadcasts one outcome reward across loop steps parameter-free; LSRL builds a genuine
per-step reward by decoding and grading each latent state — *with a hackable GPT-nano judge*, and it
explicitly asks future work for "a symbol-aware PRM verifying each step." *That is the GLaDOS organ.*
Process-reward = candidate-set cardinality drop; step-quality = soundness of the narrowing. Exact,
unhackable, free. The novelty delta in one line: *TransNAR is soft, RLTT has no verifier, LSRL's judge is
hackable — ours is exact.*

== The five-way combination nobody holds

Surveying the differentiable-symbolic-module cell (NeSy×LLM survey #arxiv("2508.13678"); nearest named
cousins DiLA and the VSA/HRR readout #arxiv("2502.01657"); reward-side RLSF #arxiv("2405.16661")), no single
system holds all five of:

#figure(
  table(columns: (auto, 1fr), inset: 5pt, align: (left, left), stroke: 0.4pt + luma(180),
    table.header([*property*], [*who is missing it*]),
    [(i) woven *in the forward pass*], [DiLA, RLSF, tool-LLMs are external],
    [(ii) differentiable, *dense structured* readback], [tool / abductive / post-hoc-verifier methods are not],
    [(iii) output-checked $arrow.r$ *sound*], [SATNet, TransNAR, AutoCoNN, the VSA readout are soft],
    [(iv) a *per-problem compiled typed program*], [DeepProbLog/LTN/SATNet use fixed/learned-but-fixed rules],
    [(v) *verifier / RLVR-trained*], [only RLSF + the tool-verifier line — and those are not woven],
  ),
  caption: [The honest novelty claim: *first to weave a checked, per-problem-compiled deductor into a
  pretrained LLM and verifier-train it.* The novelty is narrow and integrative — a combination plus the
  soundness twist, not a new primitive. The defensible edge is the discipline.],
)