#import "../helpers.typ": *

= Findings — the honest arc

This is the project's spine read in order, negatives included. The shape of the arc is: the *coupling* works
on a frozen LLM; RLVR and scale are *not* the levers people assume; the naive end-to-end version *fails*;
and a staged version *works*, validated by causal controls — which is the only thing that distinguishes it
from an award-winning artifact that was secretly reading the labels.

== The checked-deductor coupling works (and what RLVR / scale actually buy)

A frozen LLM bolted to a checked deductor gains a capability it structurally lacked: *calibrated
abstention*. Deterministic accuracy rises to *88.8% vs a base 1.4%* — the base model essentially cannot say
"I can't conclude that," and the organ gives it that channel. Three sharpening results follow, each a "not
what you'd guess":

- *RLVR fixes calibration, not capacity.* RL adds +18–20 points of abstain-recall OOD — it tunes *when* to
  defer — but does not raise the underlying solvable set. (This matches the broader RLVR-elicits-not-expands
  picture, #arxiv("2504.13837"); see §6.)
- *7B $approx$ 1B on the coupling.* Scaling the host from 1B to 7B does not improve the *reading* of the
  organ. Capacity is not the wall; 7B buys LM competence, not better deduction-use.
- *Grounding fixes extraction robustness, not exactness*, and *set-valued readout makes degradation
  graceful and sound* — when the organ is unsure it returns a *set*, and the system degrades by abstaining
  rather than by confidently erring.

== Cold co-training fails — and staging fixes it

The decisive negative: a *learned* organ co-trained from scratch with the LM is *ignored*. The LM settles
into a lazy local optimum — shortcut the answer from prompt text, fall back on an abstain prior — and the
causal controls go flat (the organ is decorative). An "airtight" bootstrap run confirmed this even with a
necessary task, a perfect organ, and a forced-open gate. This is the same disease as SATNet's grounding
collapse (§5): wherever the loss admits a cheaper non-deductive optimum, it is taken.

The fix is *not* to deny the LM the text. It is to *decouple training*: bootstrap the organ to excellence,
*freeze* it, then establish the readout. Two intermediate results pinned the diagnosis:

- *Oracle readout (de-risk, decisive, positive).* Feed the LM the *true* narrowed lattice through structured
  $gamma$, and its LM head generates the right answer *causally* — corrupt the lattice and accuracy
  goes to 0 *even with the full problem in the prompt* (the LM ignores the text and reads the lattice),
  100% OOD. The "latch gap" was a *bandwidth* problem; structured dense $gamma$ solves it.
- *Frozen-learned-organ readout (positive, reconciling).* A *frozen, standalone-trained learned* organ
  reads through $gamma$ essentially as well as the exact oracle. So the co-training failure was *distrust of
  a moving organ*, not the coupling itself.

== ⭐ The staged generative GLaDOS works

Putting the recipe together yields the project's positive deliverable: a *learned, general* organ causally
woven into the LM's *generation* across seven reasoning types (coloring, equality, ordering, arithmetic,
all-different, equality-chain, forced-color).

#keynote[
*Woven 98.8% · base 48% · text-LoRA 25%* in-distribution; OOD-phrasing *96.3%*; cells-with-no-facts *98.7%*
vs chance. The decisive control: *corrupt$arrow.r$2.7%* with the *full problem text present* — the LM
ignores the text and wields the organ. Per-rung true$arrow.r$corrupt drops of 85–100 points, *including hard
propagation chains the base and text-LoRA cannot shortcut* (woven 100% there). OOD-by-size is limited by the
*organ's own recall*, not the readout. The staged recipe is *demonstrated, not hoped.*
]

#figure(
  table(columns: (auto, auto, auto, auto, 1fr), inset: 5pt, align: (left, center, center, center, left),
    stroke: 0.4pt + luma(180),
    table.header([*setting*], [*woven*], [*base*], [*text-LoRA*], [*reading*]),
    [in-distribution (7 types)], [*98.8%*], [48%], [25%], [the organ generates, not the text],
    [OOD phrasing], [*96.3%*], [—], [—], [robustness from the organ, not surface form],
    [cells, no facts given], [98.7%], [chance], [—], [vs a chance floor],
    [corrupt, full text present], [*2.7%*], [—], [—], [collapses $arrow.r$ the organ is load-bearing],
    [det-acc, frozen-LLM coupling], [88.8%], [1.4%], [—], [calibrated abstention the base lacks],
  ),
  caption: [Headline results. The single most important row is *corrupt, full text present*: accuracy
  collapses to near-zero even though the problem is fully stated in the prompt — the proof that generation
  is routing through the organ rather than reading the text.],
)

== The causal-control table — the actual contribution

Every woven result is gated behind adversarial controls (`oracle_readout.py`). A *sound* organ guarantees a
*correct* answer if the organ produced it; it does *not* prove the organ produced it. These controls do.

#figure(
  table(columns: (auto, 1fr, auto), inset: 5pt, align: (left, left, center),
    stroke: 0.4pt + luma(180),
    table.header([*control*], [*what it injects / what it proves*], [*result*]),
    [`true`], [the real narrowed lattice — the intended path], [pass],
    [`shuffle`], [a *different problem's* lattice (roll along batch); unchanged accuracy $arrow.r$ bypass],
      [collapses],
    [`permute`], [permute candidate/value labels; tests intended semantics, not an alias], [collapses],
    [`corrupt`], [flip *only the query cell's* survivors to a wrong set; must change the answer], [$arrow.r 0$],
  ),
  caption: [The causal controls — exactly the ones SATNet did not run. The dense-$gamma$ oracle result is
  the SATNet grounding test *passed*: the LM reads *this* problem's certified state and ignores the prompt
  text. The open job is keeping it passed once $alpha$ and a learned organ replace the oracle.],
)

== A (explicit) vs B (latent): the bitter-lesson path

We tested a "bitter-lesson" variant B, where the organ is fed *only* the LM's hidden state of the text — no
extracted factors, no extraction supervision. In-distribution it narrows 94% of $#raw("ded")_p$ (within 5
points of explicit-from-factors). The honest verdict from the diversity sweep:

- Wide-$N$ training *collapses the OOD soundness failure* (false-elim 8.8%$arrow.r$0.9%); multi-task
  training *recovers phrasing* robustness (recall 76$arrow.r$88, false-elim 14%$arrow.r$3%). The latent
  narrower is *provably equivariant* (perm/size error = 0), so any residual leak is in $alpha$
  (text$arrow.r$latent), not the architecture.
- What diversity does *not* close: OOD-by-size *recall* stays ~65–74% for B vs explicit's 93%. B is a sound,
  phrasing-robust, credible bitter-lesson path; *explicit retains a ~20-point OOD-completeness edge* (it
  gets size-invariant factors for free).

== Macro-deduction: log-depth on the bounded-width side

Checked *macros* give sublinear-depth sound reasoning: $ceil(log_2 L)$ macro-applications replace ~$L$ base
steps, with an *identical exact $#raw("ded")_p$ fixpoint*, *0 false-elim, verified to $L = 256$* (a 32×
compression at $L = 256$). The compression is *symbolic transition-monoid composition* (graph-squaring,
#arxiv("2210.10749")), not the neural net — which is only a fidelity probe (sound to $k = 8$). Cost is
polynomial in the state space $S = k^w$, so it works *only on the bounded-width side*: *width is the wall;
bounded-width depth is a log-depth shortcut.* Geometry gave no edge here (the automata transitions are too
simple) — an honest null for GA-helps-macro.

== Blades and the recurring null

Blades (the GA grades) help *affine generalization* specifically, consistent with grades$->$arities. And the
recurring signal across all three organs: the *affine wall (3-way / arithmetic correlation) is the level-0
ceiling* — the place where factor / grade-3 structure must earn its keep. The geometric product as a generic
mixer remains a clean null; its value is structural, not magical.