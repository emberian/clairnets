# Prior-art & novelty audit (2026-06-25) — read this before writing ANY claim

Independent web audit of the GLaDOS design. Verdict: **"new combination of known parts,"** novel only
on axes (a)+(d). NOT new science, NOT reinventing SATNet. Be honest about this in the writeup.

## The direct prior art that owns our core: Lattice Deduction Transformers (arXiv 2605.08605)
LDT (Davis, Haller, Alfarano, Santolucito; 9 May 2026) already publishes — three months before now —
our ingredients **(b) sound lattice/abstract-interpretation narrowing + abstain** and **(c) on-policy
training mirroring a search-based constraint solver**. Verbatim from its abstract: "a recurrent
transformer that approximates logically sound deduction by projecting its latent state through a lattice
between forward passes. We train on-policy ... supervise via a domain-agnostic, abstract-interpretation-
based approximation of the set of solution candidates ... returns a correct answer or abstains."
→ **Do not claim (b) or (c) as novel.** Cite LDT head-on. It does NOT use Clifford/wedge and does NOT
touch an LLM — that's our entire novelty seam. READ THE FULL LDT PDF (we only have abstract+HTML so far).

## Per-ingredient novelty verdict
- **(a) Clifford wedge = pairwise-exclusion deduction operator: PARTIALLY NOVEL — our strongest axis.**
  u∧u=0 ⇒ exclusion is textbook (Grassmann/Pauli); GA-for-SAT exists (Budinich 2017, arXiv:1704.02942,
  symbolic not neural); CliffordNet (2601.06793) already contrasts inner-coherence vs wedge-variation in
  a LEARNED layer (but "exclusion" = feature orthogonality, not logical). Unpublished bit: wiring
  u·v+u∧v as a constraint-propagation/deduction operator in a reasoning net. Real but NARROW — high risk
  of "that's just the Pauli analogy renamed." Read Budinich + CliffordNet before claiming.
- **(b) lattice / sound / abstain: NOT NOVEL** (LDT; + DiffAI, Neural Abstract Interpretation precede the
  "learned sound abstract transformer" idea).
- **(c) on-policy alpha-target: NOT NOVEL** (LDT's on-policy constraint-solver mirroring).
- **(d) sound deductor woven INSIDE a pretrained LLM, gradients through the solver, tap-out/tap-in across
  layers: NOVEL as a combination.** Nothing occupies it: TransNAR = frozen-GNN bridge; Geiping
  recurrent-depth (2502.05171) = recurrence in a pretrained LLM but NO symbolic lattice; DiLA/DSR-LM =
  bolted-on, LLM frozen / no gradient into weights; all LLM+solver tool-use = external/discrete text.
  **This is ember's "bolt OLMo on and weave across layers" instinct — it's the genuinely new part.**

## Mandatory controls / baselines (or reviewers kill us)
- **TRM — Tiny Recursive Model** (2510.04871): 7M params, **87.4% Sudoku-Extreme**, strips HRM hierarchy.
  The real neural baseline. LDT's jump to 100% comes from the explicit symbolic lattice, not neural cleverness.
- **ARC-Prize HRM critique** (arcprize.org/blog/hrm-analysis, 2510.00355): HRM's wins ≈ outer refinement
  loop + augmentation + transductive puzzle-id memorization, NOT the architecture. **Demanded control: swap
  the fancy cell for a plain looped transformer at the SAME refinement count.** ← our `ldt` arm in the
  3-way IS exactly this control. Keep it central. Never use puzzle-id embeddings.
- **SATNet** (1905.12149) + Chang grounding critique (2312.11522): unsound, can't abstain, visual-Sudoku
  collapses to ~0% without digit labels. The canonical comparison + cautionary tale on symbol grounding.

## Language discipline
- Never write "sound/certified" unqualified — write **"empirically sound on this distribution."** Even LDT
  breaks soundness on Maze-Hard and plateaus ~36% on ARC. We have NO proof/in-loop verifier (yet).
- Frame the contribution strictly as **(a) wedge-as-exclusion** and/or **(d) in-LLM woven sound deductor** —
  not the lattice/abstain/on-policy machinery.

## Must-reads before building further
LDT 2605.08605 (full PDF) · ARC-Prize HRM analysis + TRM 2510.04871 · CliffordNet 2601.06793 + Budinich
1704.02942 · DiffAI (ICML2018) + Neural Abstract Interpretation (OpenReview WTyyhWhp4m) · SATNet 1905.12149
+ critique 2312.11522 · Geiping recurrent-depth 2502.05171 · TransNAR 2406.09308.
