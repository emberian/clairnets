#import "../helpers.typ": *

= The woven organ

The architecture is one forward pass up a *frozen* host's layer stack, with the organ spliced in as a recurrent
side-loop at the residual stream. The temptation — and the trap — is to make $alpha$ powerful enough to *solve*
the problem and $gamma$ wide enough to *dump* an answer. A GPT-5.5 review of an early "dense weave" caught
exactly this: unstructured dense surgery into the residual stream produces a *new* bypass, where the LM encodes
the answer through a lattice-shaped tensor without ever learning stable deduction. The corrections below are not
cosmetic; each closes a bypass. The coupling is *residual-stream-agnostic* — the same math grafts onto every
base we tried (§"the graft").

== $alpha$ compiles latently — the verifier, not $alpha$, guards soundness

$alpha$ (`latent_organ.DenseLatentProjector`) maps the host's hidden state to the organ's input: a *per-cell
candidate lattice* (`b0` logits over the cells' value sets). It is a *dense, fully-latent compile* — no
symbolic extraction, no parse of the prompt into a hand-specified program. This is the part with *no* prior art
— TransNAR's reasoner ingests a clean external graph the LLM never has to produce (§5); $alpha$ is the
unprecedented half, and the staged recipe gives it its own direct objective.

The early design forbade $alpha$ from emitting survival logits, fearing "the answer posterior shaped like a
lattice" as a bypass. The consolidated organ removes that fear *structurally* rather than by prohibition:
$alpha$'s compile is presented to the composer as *one verifier-gated proposal among many*, sitting on a
*certified floor*. A confidently-wrong $alpha$ can only ever sharpen the composed lattice *toward* the exact
per-cell transformer, never drop a value a real solution uses — so it *cannot poison* the state, and there is
no answer-shaped bypass to camp in. The constraint *structure* the bank consumes comes from the problem
instance (threaded into the records), not from $alpha$; $alpha$ supplies only the latent per-cell lattice.

== The bank + composer narrows between layers — the host is not the recurrence

Between two host layers, the organ slot runs the *bank as a verifier-gated reduced product* (the live
artifact is `organ.bank_woven`). On the problem's true structure it composes:

- the *certified floor* — sound-by-construction CSP-domain reductions: arc-consistency, level-2 factor
  consistency, modular Smith-normal-form, GF(2) row-space, reach-doubling macros (`organ.bank`);
- the *pretrained* neural narrower (`core_narrow_organ`, the union-trained general deductor) and $alpha$'s own
  compile — both *verifier-gated* by the exact per-cell transformer (`csp.exact_dedP`), so a wrong neural
  proposal is intersected back to what the verifier confirms.

The review's correction made this non-optional: output-verification makes accepted *answers* sound, not
intermediate *states* — "one unsound reduction poisons the next organ before the final check." So composition is
a *reduced product* (`compose.reduced_product`), each reducer with its own soundness rule, and the *meet of
sound narrowings is sound by construction*. The result: the injected lattice is *miscompile-robust by
construction*, which *subsumes* the earlier "train the organ to be robust" calibration story.

We *do not* backprop through repeated full host passes; the composer is discrete and its output is detached
($gamma$ + LoRA learn to read it, while $alpha$ is grounded by its own direct loss). The host drifts the state
coarsely across blocks; the organ refines it finely within the loop — a two-scale split matching the dynamics
of recurrent-depth transformers (#arxiv("2509.23314")). Halting is *monotone*: candidate-set cardinality only
decreases, so a cardinality-plateau is a clean early-exit.

== $gamma$ reads back richly *and structurally*, through a zero-init gate

$gamma$ writes the *composed* lattice state back into the residual stream. Three design choices, the first two
corroborated by TransNAR (#arxiv("2406.09308")):

- *Structured, not flattened.* $gamma$ uses equivariant adapters over the explicit cell-candidate tensors, not
  a size-tied flatten (which is anti-equivariant and wrecks size generalization).
- *Zero-init gate.* The readback is gated *closed* (Flamingo / LLaMA-Adapter style), so step-0 of the hybrid is
  the host *exactly* — a *bitwise no-op at init* — and the model learns to open the gate only as the organ
  proves useful.
- *Rich-state.* $gamma$ reads the full candidate *set* per cell *plus* $[|"set"|\/d, "reliability"]$ — the
  alien-consumer finding: the LM mines a partial-but-sound lattice and discounts low-reliability cells, so an
  $alpha$-miscompile degrades *gracefully* instead of dragging the LM off a cliff (rich beats a collapsed
  point-readout by +27.5 pts on partial states).

#figure(
  block(width: 100%, inset: 9pt, stroke: 0.5pt + luma(160), radius: 4pt)[
    #set text(size: 9pt)
    #set align(left)
    `problem (text)` \
    #h(1em) $arrow.b$ \
    `frozen host layers 1..k` $#h(0.5em) ---->#h(0.5em)$ #raw("(α) LATENT compile → per-cell candidate lattice")\
    #h(1em) $arrow.b$ residual #h(6.5em) $arrow.b$ \
    #h(1em) $arrow.b$ #h(6.0em) `composer: certified floor (Arc/Factor/Modular/GF2/Macro)` \
    #h(1em) $arrow.b$ #h(8.2em) `⊕ verifier-gated { core narrower, α }` \
    #h(1em) $arrow.b$ #h(8.2em) $arrow.b$ composed (sound-by-construction) lattice \
    #raw("⊕  residual add  <----  (γ) RICH-STATE structured readback · zero-init gate") \
    #h(1em) $arrow.b$ \
    `frozen host layers k+1..N` $arrow.r$ `LM head → answer` $arrow.r$ #raw("exact verifier → reward → (α,γ,LoRA)")
  ],
  caption: [The weave. $alpha$ compiles latently (it does not solve, and cannot poison); the bank+composer
  narrows between layers on a certified floor (the host is not the recurrence); $gamma$ reads the *composed*
  lattice back richly through a closed-init gate; the host's own LM head generates; an exact verifier closes the
  loop. Only $alpha$, $gamma$, and the host LoRA train — the base and the organ bank are frozen.],
)

== The organ bank: a shared protocol, many inference shapes

There is not one organ but a *bank*, every beast wired to one typed spine (`organ.protocol.Reduction`): an
abstract STATE, a sound `reduce` (or a no-op = ABSTAIN), a `certificate` (`sound-by-construction` /
`neural-guidance` / `approximate`), and the uniform `survival(state, K)` readout $gamma$ projects. The
status partition is honest:

- *certified, CSP-domain* (the reduced-product floor): arc / factor consistency, modular SNF, GF(2) row-space,
  reach-doubling macros, and the exact per-cell transformer as the *verifier*;
- *certified, own state type*: forward-chaining least-Herbrand entailment (`unification_chain`, the narrowing
  dual — grows facts);
- *approximate, own state type*: mean-field energy / equilibrium optimisation (the optimisation shape the
  narrowers cannot express);
- *neural-guidance, verifier-gated*: the union-trained `core_narrow_organ` and the blade/grade-prior affine
  deductor.

No single beast dominates; the verifier — never the LM router alone — arbitrates which narrowing is trusted.

== Reductions: a cross-type faculty (one faculty + edges = many problems)

The composer routes *within* a state type. A second, distinct typed object — `organ.reductions.ProblemReduction`
— routes *across* types: it maps an X-instance to a Y-instance of a different faculty and decodes the
Y-solution back. The shipped Karp neighbourhood (MIS$arrow.l.r$VC$arrow.l.r$clique; 3-SAT / 2-SAT / XOR-SAT
$arrow.r$ CSP; 3-SAT $arrow.r$ MIS $arrow.r$ Ising; {MIS, MaxCut, partition, coloring} $arrow.r$ Ising) reuses
the exact Lucas NP$arrow.r$Ising encoders already in the repo; *every edge is exact-verified end-to-end
against X's own solver*, and `ReductionGraph.route` is Dijkstra cost-routing with a certified-over-approximate
tie-break. The model learns *recognize-X · reduce-X$arrow.r$Y · solve-via-Y · decode* — the structural Karp
prior the GNN-for-NP papers ignore — so one faculty plus edges covers everything reducible to it.

== Geometric algebra, placed precisely

Geometric algebra was the project's orphaned opening thread, and as a *generic channel mixer* it is a clean
null (it ties SwiGLU/MLP in language and vision). Its real home is the factor deductor, via a structural
identity: *geometric-product grades correspond to constraint arities.* Grade-0 (inner) $<->$ unary; grade-2 (the
wedge $u and v$, "these two cannot coexist") $<->$ binary exclusion; grade-3 (the trivector $u and v and w$)
$<->$ the ternary/affine primitive that breaks the level-0 wall. The right abstraction is the *un-truncated
fold* — one wedge spanning all arities up to a capacity $K$. Crucially, GA is a *completeness / representation*
bias, *not* a soundness fix: soundness comes only from the output check and the certified kernels, never from
the algebra (#arxiv("2601.06793")).

== The staged recipe — the central lesson

#figure(
  block(width: 100%, inset: 9pt, fill: luma(247), radius: 4pt)[
    #set text(size: 9.5pt)
    #set align(left)
    *Stage 1 — the organ to excellence, separately.* Train the deductor hard across many tasks on the
    soundness-asymmetric dominate-$#raw("ded")_p$ loss until it is a general, sound, near-oracle *standalone*
    narrower (no LM). Cheap: pure organ. The recipe (from the sweep): ~1.5M params, $R = 12$, ~30% composed.\
    #v(3pt)
    *Stage 2 — freeze it, then weave.* Graft the frozen bank+composer organ into a host; train LoRA + latent-$alpha$
    + zero-init $gamma$ on the answer-span LM cross-entropy, with the *engagement mechanism*: the two-stream
    lever (the LM generates from a fact-ablated prompt so the organ is the only route) + $J_0$ direct
    $alpha$-supervision (the SATNet grounding fix). Each translator competent *before* it must cooperate.\
    #v(3pt)
    *Stage 3 — RLVR, then consolidate.* RL-from-verifiable-rewards (the organ is its own exact process reward),
    then re-run the host's own recipe (Dolma ⊕ reasoning) *with the organ in place* so organ-use
    becomes native, not bolted-on.
  ],
  caption: [The staged recipe. The central finding (§4) is that you must *not* cold-co-train the LM and the
  organ — two alien substrates, both bad at init, settle into a text-shortcut. Decouple: make the organ *good*
  (pretrained, frozen) and *necessary* (the easy path removed). The organ becomes load-bearing only when both
  hold.],
)

== The organ as its own process reward

Because the organ narrows *monotonically* and *checks* each step, it is its own exact per-step grader: the
process-reward "progress" signal is the candidate-set cardinality drop at step $t$, and the "quality" signal is
the soundness of that drop (guaranteed by the check). This is exactly the dense per-step supervision the
process-reward line (RLTT/LSRL, §5/§6) had to approximate with a hackable LLM judge — here it is free, exact,
and unhackable (`organ.process_reward`, mixed $0.7 dot$outcome $+ 0.3 dot$process).
