#set document(title: "GLaDOS: a sound lattice-deduction co-processor for transformers")
#set page(numbering: "1", margin: 1in)
#set par(justify: true, leading: 0.62em)
#set text(font: "New Computer Modern", size: 10.5pt)
#show heading: set block(above: 1.2em, below: 0.6em)
#set heading(numbering: "1.1")

#align(center)[
  #text(17pt, weight: "bold")[GLaDOS]\
  #text(12pt)[Geometric Lattice Deduction Over Streams]\
  #v(2pt)
  #text(10pt, style: "italic")[a sound, abstain-capable deduction organ — and the question of whether it survives being woven into a thinking LLM]\
  #v(4pt)
  #text(9pt)[ember + Claude (Opus 4.8) · draft #datetime.today().display()]
]

#v(6pt)
#align(center)[#block(width: 92%, inset: 8pt, stroke: 0.5pt + gray, radius: 3pt)[
  #text(9pt)[*Honesty banner.* The sound-lattice-deduction + abstain + on-policy machinery is *not ours* — it is
  the Lattice Deduction Transformer (LDT, arXiv 2605.08605, May 2026). Our contribution is narrow and
  specific: *(a)* a Clifford geometric-product cell whose *wedge* is a pairwise-exclusion proposer, and
  *(d)* welding the deductor *inside* a pretrained LLM (OLMo-3), gradients through the solver, woven across
  layers. We write "empirically sound on this distribution," never "certified."]
]]

= The one-sentence architecture
A Clifford transformer (the *cortex*) compiles a problem into a *constraint program* — initial candidate
sets plus a pairwise-exclusion graph — and writes it into a recurrent *deductor* that runs sound powerset
narrowing and returns a certified result or abstains. The cortex *reads it back* and continues. The spine:
$ a_(t+1) = a_t inter Pi_A (f_theta (a_t, p, m)) $
where $a_t$ is a point in a per-position powerset lattice, $Pi_A$ is monotone narrowing (the meet keeps the
result $subset.eq a_t$), and $f_theta$ is the geometric-product cell. Bilinear *proposes*; the lattice meet
is the sole *authority* on elimination.

== Why Clifford is not arbitrary
The wedge $u and v$ is antisymmetric ($u and u = 0$): it discards self-magnitude and keeps only cross-channel
structure — exactly "these two candidates cannot coexist," the primitive a constraint graph is built from.
A Clifford cortex computes, as its native representation op, the same algebra the deductor consumes as
constraints. The host's wedge features become the deductor's exclusion edges. Deep coupling, not vocabulary.

= What is ours vs what is borrowed
#table(columns: (auto, 1fr, auto), inset: 5pt, align: left,
  table.header([*ingredient*], [*status*], [*source*]),
  [(b) lattice / abstract-interpretation narrowing / abstain], [*not novel*], [LDT 2605.08605],
  [(c) on-policy alpha-target on own deduced states], [*not novel*], [LDT 2605.08605],
  [(a) Clifford wedge = pairwise-exclusion deduction op], [*partially novel, narrow*], [ours; cf. CliffordNet 2601.06793, Budinich 1704.02942],
  [(d) sound deductor woven *inside* a pretrained LLM, grad through solver], [*novel as a combination*], [ours],
)
The honest reduction: LDT's leap to 100% on Sudoku-Extreme came from the *symbolic lattice*, not neural
cleverness (cf. TRM, 7M params, 87.4%, arXiv 2510.04871). So our entire bet is two questions — *does the
wedge add anything on top of the lattice?* and *does it survive the LLM graft?*

= The decisive experiment (Layer 0)
Iso-parameter core-swap on Sudoku-Extreme (≈800K params, identical solve loop, only the channel mixer varies):
- *ldt* — attention + SwiGLU FFN. This is *also the control the HRM critics demand* (a plain looped
  transformer at equal refinement count; ARC-Prize showed HRM's gains were the refinement loop +
  augmentation + puzzle-id memorization, not architecture).
- *glados* — attention + geometric product (dot + wedge), no FFN.
- *nowedge* — ablation: dot-only, the antisymmetric branch removed.

== Metric that decides it
*False-elimination rate* — the fraction of (non-given, true-still-alive) cells where the meet just killed the
true candidate. A sound deductor's value is $approx 0$. Plus solve rate, *wrong-return rate* (soundness
violations at output), and p90 forward-passes per solved puzzle. Decision rule: glados *has legs* iff it
matches/beats ldt's solve rate at equal params *and* lowers false-elimination (or cuts p90 forwards $gt.eq 25%$).
Otherwise the wedge is decoration and Clifford is just an efficiency primitive.

#block(width: 100%, inset: 8pt, fill: luma(245), radius: 3pt)[
  *Results — Sudoku-Extreme, iso-param, 2000 steps.* #emph[pending the running 3-way]
  #table(columns: 5, inset: 4pt, align: (left, center, center, center, center),
    table.header([arm], [params], [false-elim], [solve], [wrong-ret], ),
    [ldt], [—], [—], [—], [—],
    [glados], [—], [—], [—], [—],
    [nowedge], [—], [—], [—], [—],
  )
]

= Beyond the organ
+ *Domain-general gauntlet* (built): one organ, many lattices — graph-coloring, 3-SAT, maze through one
  interface; hold out a domain, test *zero-shot soundness*. The honest form of "generalizes to nonsense."
+ *The co-processor* (Layer 1): zero-init gated graft into OLMo-3-Base-7B (Flamingo-style tanh gate → step-0
  hybrid is OLMo exactly); baseline to beat = OLMo-3-Think-7B. Kill-risk: does the cortex *learn to emit
  valid constraint programs*, or route around the organ?
+ *Streams & lifting*: semigroup binary-lifting $F_2 approx Pi(F_1 circle.small F_1)$ → $O(log T)$ deduction;
  the lifting error is itself a diagnostic of how shortcuttable the task's deduction semigroup is.
+ *Online edge-of-stability*: cheap running proxies (participation ratio, finite-time Lyapunov, top-k Hessian)
  the whole way along. Our hypothesis-under-test: *attractor dimension predicts cross-domain soundness better
  than val-loss*.

= Open honesty / risks
- "Sound" is *empirical*, on-distribution. We have no proof and no in-loop verifier yet. Even LDT breaks
  soundness on Maze-Hard.
- (a) is at risk of "that's just the Pauli analogy renamed" — read Budinich + CliffordNet before claiming.
- Must reproduce the plain-looped-transformer control (our ldt arm) and cite LDT/TRM head-on.

#v(4pt)
#text(9pt, style: "italic")[References: see #raw("notes/prior_art.md") for the full lineage and URLs (LDT 2605.08605, TRM 2510.04871, ARC-Prize HRM analysis, CliffordNet 2601.06793, SATNet 1905.12149, …).]
