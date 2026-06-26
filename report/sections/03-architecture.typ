#import "../helpers.typ": *

= The woven organ

The architecture is one forward pass up the host's layer stack, with the organ spliced in as a recurrent
side-loop. The temptation — and the trap — is to make $alpha$ powerful enough to *solve* the problem and
$gamma$ wide enough to *dump* an answer. A GPT-5.5 review of an early "dense weave" caught exactly this:
unstructured dense surgery into the residual stream produces a *new* bypass, where the LM encodes the answer
through a lattice-shaped tensor without ever learning stable deduction. The corrections below are not
cosmetic; each closes a bypass.

== $alpha$ compiles a program, it does not solve

$alpha$ maps the host's hidden state to a *typed factor-graph program*: the variables, their domains, and
the initial factors/evidence — and *nothing about the answer*. If $alpha$ were allowed to emit
candidate-survival logits, "the answer posterior shaped like a lattice" would be a bypass: the organ would
have nothing left to do. So $alpha$ states the problem; the organ narrows it. This compile step is also the
part with *no* prior art — TransNAR's reasoner ingests a clean external graph the LLM never has to produce
(§5). $alpha$ is the unprecedented half, and the staged recipe pretrains it on its own.

== The organ narrows between layers — the host is not the recurrence

The organ runs its narrowing recurrence *between* two host layers, not by re-running the host. We
*do not* backprop through repeated full host passes; old lattice states are stop-grad / truncated, and we
supervise every organ step directly. The host drifts the state coarsely across blocks while the organ
refines it finely within the loop — a two-scale split that matches the observed dynamics of recurrent-depth
transformers (#arxiv("2509.23314")). The natural halting signal is *monotone*: candidate-set cardinality
only decreases, so a cardinality-plateau (with a two-step confirm) is a cleaner early-exit than the
norm-threshold halts that misfire on generic recurrent spirals.

== $gamma$ reads back densely *and structurally*, through a zero-init gate

$gamma$ writes the certified lattice state back into the residual stream. Two design choices, both
corroborated by TransNAR (#arxiv("2406.09308")):

- *Structured, not flattened.* Flattening the lattice to a fixed `[N,K]` tensor is anti-equivariant and
  tied to a maximum problem size — it wrecks size generalization. $gamma$ instead uses equivariant adapters
  over the explicit cell-candidate *and* factor-tuple tensors. TransNAR found the same: reading node *and*
  edge structure and combining by concat+linear beats every reductive pooling (sum, MLP, Gram–Schmidt).
- *Zero-init gate.* The readback is gated with the gate initialized *closed* (Flamingo / LLaMA-Adapter
  style), so step-0 of the hybrid is the host *exactly* — no damage to pretrained knowledge — and the model
  learns to open the gate only as the organ proves useful.

#figure(
  block(width: 100%, inset: 9pt, stroke: 0.5pt + luma(160), radius: 4pt)[
    #set text(size: 9pt)
    #set align(left)
    `problem (text)` \
    #h(1em) $arrow.b$ \
    `host layers 1..k` $#h(0.5em) ---->#h(0.5em)$ #raw("(α) compile → typed factor-graph program")\
    #h(1em) $arrow.b$ residual #h(6.5em) $arrow.b$ \
    #h(1em) $arrow.b$ #h(8.2em) `organ: narrow / chain / energy` \
    #h(1em) $arrow.b$ #h(8.2em) (recurrent · checked · differentiable) \
    #h(1em) $arrow.b$ #h(8.2em) $arrow.b$ certified state \
    #raw("⊕  residual add  <----  (γ) structured dense readback · zero-init gate") \
    #h(1em) $arrow.b$ \
    `host layers k+1..N` $arrow.r$ `LM head → answer` $arrow.r$ #raw("exact verifier → reward → (α,γ,LoRA)")
  ],
  caption: [The weave. $alpha$ compiles (does not solve); the organ recurs between layers (the host is not
  the recurrence); $gamma$ reads back structurally through a closed-init gate; the host's own LM head
  generates; an exact verifier closes the loop. Gradient reaches $alpha$, $gamma$, and the host LoRA — the
  organ itself is frozen during weaving (see the staged recipe).],
)

== The organ bank: a shared protocol, three inference shapes

There is not one organ but a bank, sharing a *protocol* (compile $arrow.r$ recurse $arrow.r$ check) rather
than a coupling — each has its own state schema, halting rule, and verifier:

- *narrow* (`proposer.py`) — monotone candidate rule-out; sound, abstains. Best for satisfaction.
- *chain* (`chain_organ.py`) — *grows* derived facts instead of narrowing candidates (the dual); best for
  entailment/derivation. Exhibits a sharp phase transition: sound *or* deep, rarely both at once.
- *energy* (`energy_organ.py`) — soft constraint satisfaction as an equilibrium; uniquely does
  *optimization* (85.5% exact min-cost), with ~6% declared infeasible at the affine wall.

No single organ dominates; a portfolio plus verifier-selection is complementary. The LM router alone is not
trusted to choose — the verifier arbitrates.

== Geometric algebra, placed precisely

Geometric algebra was the project's orphaned opening thread, and as a *generic channel mixer* it is a clean
null (it ties SwiGLU/MLP in language and vision). Its real home is the factor deductor, via a structural
identity: *geometric-product grades correspond to constraint arities.* Grade-0 (inner product) $<->$ unary;
grade-2 (the wedge $u and v$, antisymmetric, "these two cannot coexist") $<->$ binary exclusion; grade-3
(the trivector $u and v and w$) $<->$ the ternary/affine primitive that breaks the level-0 wall. The right
abstraction is the *un-truncated fold* — one wedge spanning all arities up to a capacity $K$ — which matches
hardcoded grades in-distribution and generalizes best zero-shot. Crucially, GA is a
*completeness / representation* bias, *not* a soundness fix: soundness comes only from the output check and
typed symbolic kernels (e.g. exact modular arithmetic), never from the algebra (#arxiv("2601.06793")).

== The staged recipe — the central lesson

#figure(
  block(width: 100%, inset: 9pt, fill: luma(247), radius: 4pt)[
    #set text(size: 9.5pt)
    #set align(left)
    *Stage 1 — organs to excellence, separately.* Train each deductor hard across many tasks until it is a
    general, robust, near-oracle *standalone* reasoner (no LM). Cheap: pure organ.\
    #v(3pt)
    *Stage 2 — pretrain each interface piece on its own sub-task.* $gamma$ on oracle-lattice$arrow.r$answer
    readout (proven; see §4); $alpha$ on text$arrow.r$program extraction (supervised). Each translator must
    be competent *before* it has to cooperate — and each gets its own isolated grounding test, never
    inferred from end-to-end numbers (the SATNet lesson).\
    #v(3pt)
    *Stage 3 — assemble, then consolidate.* Plug the pretrained pieces together, *freeze the organ*, and
    only then run the long, weird training where the LM learns to route its reasoning through the organ —
    closer to re-running real pretraining (Dolma-with-the-organ) so organ-use becomes *native*, not
    bolted-on.
  ],
  caption: [The staged recipe. The central finding (§4) is that you must *not* cold-co-train the LM and the
  organ — two alien substrates, both bad at init, cannot bootstrap together and the LM settles into a
  text-shortcut. Decouple: make the organ *good* (pretrained, frozen) and *necessary* (the task cannot be
  shortcut). The organ becomes load-bearing only when both hold.],
)

== The organ as its own process reward

Because the organ narrows *monotonically* and *checks* each step, it is its own exact per-step grader: the
process-reward "progress" signal is the candidate-set cardinality drop at step $t$, and the "quality" signal
is the soundness of that drop (guaranteed by the check). This is exactly the dense per-step supervision that
the process-reward line (RLTT/LSRL, §5/§6) had to approximate with a hackable LLM judge — and here it is
free, exact, and unhackable.