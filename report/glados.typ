// GLaDOS — main paper file. Sections live in sections/ and are #include-d.
// Preamble + title block carried over from the original clairnets.typ draft.

#set document(title: "GLaDOS: a checked deduction organ woven into a language model")
#set page(numbering: "1", margin: 1in)
#set par(justify: true, leading: 0.62em)
#set text(font: "New Computer Modern", size: 10.5pt)
#show heading: set block(above: 1.2em, below: 0.6em)
#set heading(numbering: "1.1")
#show raw: set text(font: "DejaVu Sans Mono", size: 9pt)

#import "helpers.typ": *

#align(center)[
  #text(18pt, weight: "bold")[GLaDOS]\
  #text(12pt)[Geometric Lattice Deduction Over Streams]\
  #v(2pt)
  #text(10pt, style: "italic")[a sound, abstain-capable deduction organ — and the evidence that an
  LLM can be made to *causally* reason through it]\
  #v(4pt)
  #text(9pt)[ember + Claude (Opus 4.8) · draft #datetime.today().display()]
]

#v(6pt)

#banner[
  *Honesty banner.* We write "empirically sound on this distribution," never "certified." Soundness
  here is a *property of an output check we run*, not a theorem about the network. The borrowed parts are
  named as borrowed: the lattice / abstract-interpretation narrowing + abstain + on-policy targets are the
  Lattice Deduction Transformer (LDT, #arxiv("2605.08605")); the coupling primitives (zero-init gate,
  structured readback, freeze-the-reasoner) are TransNAR (#arxiv("2406.09308")); the process-reward
  recipes are RLTT (#arxiv("2602.10520")) and LSRL (Findings EMNLP 2025). What is *ours* is the
  combination and the discipline: a per-problem-compiled, output-checked deductor woven into a pretrained
  LLM's residual stream and verifier-trained, with adversarial causal controls run on every result.
  Negatives in this paper are results, not omissions.
]

#v(4pt)

#align(center)[#block(width: 94%, inset: 6pt)[#text(9pt, style: "italic")[
*Abstract.* We ask whether a learned, differentiable, output-checked deduction module can be woven into a
pretrained language model so that the model's own generation reasons *through* it — and we hold ourselves
to the standard that broke the closest prior work (SATNet): a sound module can still be *bypassed*, so only
causal controls (shuffle / permute / corrupt the module's state) prove genuine deduction. We build an exact
CSP harness as ground truth, a *bank* of deduction organs wired to one typed spine (certified arc / factor /
modular / GF(2) / macro reductions + a verifier-gated neural narrower + forward-chaining + energy), and a
staged training recipe. Our findings, with the negatives: coupling a frozen LLM to a checked deductor gains
calibrated abstention it lacks (88.8% vs 1.4%); RLVR fixes *calibration*, not *capacity*; 7B behaves like 1B
(capacity is not the wall); *cold co-training fails* — but a staged recipe (organ$arrow.r$excellence
$arrow.r$ freeze $arrow.r$ weave) makes a *learned, general* organ causally load-bearing. The consolidated
live organ — latent $alpha$ $arrow.r$ a verifier-gated reduced product over a *certified floor* $arrow.r$
rich-state $gamma$ — *engages* (corrupt$arrow.r$drop ~97 pts, lift +77), is *miscompile-robust by
construction* (the floor cannot be poisoned), *generalizes by reduction* (a no-direct-faculty family solved
100% via a Karp edge), and is *legible learned-free* ($alpha$-compile read at F1 0.90 / recall 0.98); the
graft is *residual-stream-agnostic* (7/7 bases incl. a Mamba-hybrid). We state the honest scope plainly: in
the easy regime the certified floor carries correctness on its own, so the bottleneck test is saturated — the
open frontier is $alpha$ compiling from *real natural language* with no provided structure. We position this
against TransNAR, Coconut, SATNet, and the process-reward line.
]]]

#v(6pt)

#include "sections/01-intro.typ"
#include "sections/02-harness.typ"
#include "sections/03-architecture.typ"
#include "sections/07-training.typ"
#include "sections/04-findings.typ"
#include "sections/05-related.typ"
#include "sections/06-realtraining.typ"

#v(8pt)
#line(length: 100%, stroke: 0.4pt + gray)
#text(9pt, style: "italic")[
  *Lineage / references (arXiv ids inline above).* LDT 2605.08605 · CliffordNet 2601.06793 · automata-shortcuts
  2210.10749 · TransNAR 2406.09308 · Coconut 2412.06769 · SATNet 1905.12149 · SATNet grounding critique
  2312.11522 · reasoning shortcuts 2305.19951 · RLTT 2602.10520 · LSRL (Findings EMNLP 2025) · looped
  transformers 2502.17416 · two-scale recurrence 2509.23314 · RLVR-elicits 2504.13837 · spurious rewards
  2506.10947 · get-go RPT 2502.19402 · SFT-memorizes-RL-generalizes 2501.17161 · emergent-under-CPT 2506.00288 ·
  procedural-knowledge 2411.12580 · front-loading reasoning 2510.03264 · Logic-RL 2502.14768 · reasoning-gym
  2505.24760 · ZebraLogic 2502.01100 · DiLA (OpenReview uh3ZO2izyr) · VSA/HRR readout 2502.01657 · RLSF
  2405.16661 · NeSy×LLM survey 2508.13678. Full notes + URLs in #raw("clairnets/notes/").
]
