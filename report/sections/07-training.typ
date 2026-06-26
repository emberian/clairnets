#import "../helpers.typ": *

= Training: the staged, oracle-supervised recipe

The architecture (§3) says *what* the weave is; this section says *how* it is taught. The recipe is
*staged, curriculum-driven, and oracle-supervised* — and that last word matters: it is *not* classic
semisupervised learning. Every label is *free and exact*. Witness-first generation hands us a known
solution, and the exact per-cell deductor (`clair.csp.exact_dedP`) hands us the exact narrowing targets,
the answers, and the verifiable rewards. There are *no human labels and no noisy pseudo-labels* — the
"curriculum" is purely the *staging order*, the readout-forcing schedule, and the domain / composition / $N$
mix. The entry point is `clair/organ/train.py` (`organ` for Stage 1, `weave` for Stages 2–3).

== Stage 1 — pretrain the organ, standalone

The deductor learns to *narrow soundly*, with the base LLM not yet involved. The loss is *dominate-*$#raw("ded")_p$:
a soundness-asymmetric supervised loss against the exact per-cell transition. Eliminating a value that some
solution actually uses (an *unsound* drop) is penalized heavily ($w_"pos" approx 6$); keeping an extra
candidate (*loose but sound*) is penalized lightly ($w_"neg" approx 0.5$). The step is monotone — survival
only ever decreases (the meet) — so the asymmetry trains the organ toward the sound side of the boundary.
Training is *on-policy*: each batch is the previous step's narrowed lattice rolled forward by the organ, with
targets re-computed by the exact deductor every pass (the `given` set is recomputed after each meet).

The data is the *broad multi-domain corpus*: all beast generators (CSP families · xor/GF2 · modular · graph ·
permutation · type · FOL · optimize), plus *union relations* (named families *and* random allowed-tuple
tables, added *additively* — this is the fix for OOD-novel-relation soundness), wide $N$, and *~30% composed
problems* (multi-skill). The sweep recipe is *~1.5M parameters, $R = 12$ recurrent steps* (with an $R = 8$
floor) and the ~30% composition mixup. Composition *recall* is nearly free (the meet composes); composition
*coverage* buys *soundness* on novel compositions — a need that *grows with scale*. The output is a frozen,
general, sound narrower checkpoint (`runs/general_organ_full.pt`).

== Stage 2 — freeze the organ, weave the live readout

Now the LLM learns to *generate through* the frozen organ. The Stage-1 organ is *frozen* (a slow-thaw / EMA
variant is studied; default is freeze), the OLMo base is *frozen*, and the only trained parameters are
*LoRA* + *latent-*$alpha$ + *$gamma$*:

- *latent-$alpha$* is a dense projection from the OLMo hidden state to the organ's input tensors. It is
  *fully latent* — there is *no symbolic extraction step*. The hidden state compiles directly to the typed
  factor-graph program (variables, domains, factors), and never to the answer.
- *$gamma$* is the structured readback, with its gate *zero-initialized* (closed) so step-0 of the hybrid is
  the host exactly.

The forward pass is: OLMo hidden $arrow.r$ latent-$alpha$ $arrow.r$ the frozen organ narrows (under
`no_grad`, so we *never* backprop through repeated OLMo passes) $arrow.r$ $gamma$ injects the narrowed
lattice into the residual $arrow.r$ the LM head generates the answer. The loss is plain *LM cross-entropy on
the answer span only*. $alpha$ never gets a direct target; it learns purely from the gradient flowing
*through the organ's downstream success* — the LM does better when $alpha$ has compiled a usable problem, and
that signal is what teaches the compile.

The curriculum here is *readout-forcing*. Training starts in *cells-mode* — no facts in the prompt, so the
organ is the *only* route to the answer — which forces the $gamma$-decode to form *before* the LM can settle
into a text-shortcut basin. Full-text problems are then mixed back in. The output is a woven model in which
the LM *causally* wields the organ: corrupt the lattice and generation collapses.

== Stage 3 — RLVR and consolidation

Two moves sharpen and stabilize the woven model:

- *RLVR* (GRPO; verifiable reward = the exact checker) with the *organ-as-process-reward* (LSRL-style:
  per-step candidate-set cardinality drop), mixed *0.7 outcome / 0.3 process* so the per-step signal shapes
  rather than dominates. This sharpens *calibration*, not capacity (§4); we report pass\@1 *and* pass\@k with
  ProRL and random-reward controls, on OLMo as the honest base (§6).
- *Consolidation*: continued-pretrain on *Dolma ⊕ a reasoning corpus* (general ⊕
  organ-shaped, tuned ratio) with the organ in place, so organ-use folds into general behavior *without*
  catastrophic forgetting — the preserve-and-generalize endpoint.

== The honest correction: readout-isolation vs live latent-$alpha$

#banner[
*Scope correction.* The woven numbers reported in §4 — woven 98.8%, corrupt$arrow.r$2.7% with full text
present — are *readout-isolation* results: the organ is handed the *true* narrowed lattice (the
ground-truth CSP, compiled outside the model), and what is proven is that the LM *reads back and generates
through* a certified state it cannot get from the prompt. That readout is real and causally controlled, and
it is the hard half TransNAR also enjoys (a clean external graph it never has to produce). The *real* woven
model closes the remaining gap with *live latent-$alpha$*: $alpha$ compiles the organ's input *from the
OLMo hidden state*, with *no symbolic extraction and no ground-truth CSP*. That model is being wired now,
post-codex-preflight. *Live latent-$alpha$ is the critical path and the honest open piece* — readout is
proven; $alpha$ (latent compile) is the hardest part, and a wrong compile means a confidently wrong organ.
]

== The base matrix — one organ, many hosts

The pretrained organ is *base-agnostic*: Stage 1 is done once, and the woven readout (Stages 2–3) is a
weights-only bolt-on, so each base is a cheap, independent paper datapoint. The set now spans
*~100M $arrow.r$ 32B across six model families*, including an SSM-hybrid, to show the technique uplifts
across scale *and* architecture rather than on one lucky host.

#figure(
  table(columns: (auto, auto, 1fr), inset: 5pt, align: (left, left, left),
    stroke: 0.4pt + luma(180),
    table.header([*base*], [*tier / family*], [*why it is in the matrix*]),
    [Pythia-160M / SmolLM2-135M], [~100M · Pythia/SmolLM], [ships *all* training checkpoints =
      interp-gold; the *organ-carries-the-reasoning* extreme (host too small to reason alone)],
    [OLMo-2-1B], [1B · OLMo], [cheap full-recipe datapoint, fully-open base; low end of the scale curve],
    [SmolLM3-3B], [3B · SmolLM], [fully-open small-modern model; a clean small datapoint between 1B and 7B],
    [*OLMo-3-7B (PRIMARY)*], [7B · OLMo], [interp anchor, full causal controls, the honest fully-open
      mechanism — *run \#1, never skipped*],
    [OLMo-3-32B], [32B · OLMo], [does the uplift *hold / grow* with base size (QLoRA @ 4-bit)],
    [Gemma-4], [mid · Gemma], [portability + uplift on an already-strong, code-trained model],
    [Qwen-3.6], [mid · Qwen], [portability across a second strong family],
    [Nemotron-H-8B], [8B hybrid · Nemotron], [the *architecture-diversity* datapoint — Mamba-2 + self-attention
      + MLP hybrid; does the residual-stream coupling survive when the splice point is an SSM block, not
      attention? Also *fully open* (weights + data + recipe), so interp-friendly like OLMo],
  ),
  caption: [The base matrix. ~2.5 orders of magnitude (~100M$arrow.r$32B) over six families
  (OLMo · Gemma · Qwen · Nemotron-H · SmolLM · Pythia), including an SSM-hybrid. Compute: woven = frozen base
  + LoRA + $alpha$ + $gamma$; 7B/8B fit one L40S, 32B fits one A100-80 (or L40S @ 4-bit), the $lt.eq$3B
  tier runs anywhere. OLMo-3-7B is run \#1.],
)

== What each stage proved, and the standing caveats

The evidence ladder (§4, in order): the readout works (corrupt$arrow.r$0, oracle de-risk); a *frozen learned*
organ reads as well as the oracle; *cold co-training fails and staging fixes it*; the staged generative model
uses the organ across seven rung types (corrupt$arrow.r$2.7% with text present); next-token CPT recruits the
organ *with no reward* (architecture, not RLVR trick); and the split-brain control shows the LM wields the
organ 99.9% over adversarial text (necessity, not opportunism). The standing caveats stay on the page:
the organ's soundness is *trained*, not by-construction (the certified operators are the floor, the net is
guidance — *neural proposes, certified guard narrows*); the model trusts the organ *unconditionally*
(split-brain), so *$alpha$-correctness is the whole ballgame*; and the *affine wall* is real — single ternary
constraints need grade-3 and XOR *systems* need the certified row-space organ, neither beaten by depth alone.
