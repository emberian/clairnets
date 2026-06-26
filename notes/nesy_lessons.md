# NeSy lessons for GLaDOS: cautionary tales + the closest prior work

What this is: a focused crawl of the neuro-symbolic × LLM literature for (a) the failure mode we keep
fearing — a differentiable solver that *appears* to learn the logic but is actually being bypassed /
leaked-to — stated precisely with **the exact controls that detect it**, mapped onto our existing
causal-control harness; and (b) a curated map of who sits closest to GLaDOS, with an honest novelty
verdict.

GLaDOS recap (README.md, notes/prior_art_extract.md): a **checked/sound** lattice-factor deduction organ
woven into a pretrained LLM's forward pass — α compiles the hidden state into a typed factor-graph program
(not the answer), the organ narrows it recurrently *between* OLMo layers, γ writes the certified state
densely back through a zero-init gate, an exact verifier supplies the reward. Soundness is a *permission*
to be loose; completeness (reach a checkable answer vs honestly abstain) is the research variable.

Our causal controls already exist in `clair/oracle_readout.py` (lines 28–34, 255–280):
- **`true`** — inject the real narrowed lattice.
- **`shuffle`** — inject a *different problem's* oracle lattice (roll along batch).
- **`permute`** — permute the candidate (color/value) labels of every survival vector.
- **`corrupt`** — flip *only the query cell's* survivors to a wrong/changed set (the README's "corrupt→0").
- Decisive logic: *if accuracy is unchanged under shuffle, γ is a bypass — the LM is ignoring the lattice
  (or LoRA quietly solved the text).*

The whole point of part (1) below is that **this exact harness is the antidote to the SATNet disease.**

---

## 1. The cautionary tale — SATNet, and the reasoning-shortcut generalization

### 1a. SATNet (Wang, Donti, Wilder, Kolter, ICML 2019, arXiv 1905.12149) — the claim
SATNet is a differentiable (smoothed) MAXSAT solver as a neural layer: it solves an SDP relaxation of
`max_v Σ_j ⋁_i 1{s̄_ij v̄_i > 0}` with the clause matrix `S` as learnable weights, and backprops through the
SDP. The headline claims: it learns the *parity* function from single-bit supervision, learns to play 9×9
Sudoku "without any hand-coded knowledge of the problem structure" (98.3% test), and — the load-bearing
claim for us — **"visual Sudoku"**: a CNN backbone reads MNIST digit images, a SATNet layer solves the
puzzle, trained **end-to-end with no intermediate digit labels** (93.6% train / 63.2% test). This was sold
as combining System-1 perception with System-2 reasoning in a minimally-supervised, end-to-end fashion.
Best-paper honorable mention at ICML 2019.

### 1b. The grounding critique (Chang, Flokas, Lipson, Spranger, NeurIPS 2020, arXiv 2312.11522) — what actually failed
**The apparent "learned logic" was label leakage; the symbolic layer was never grounding the perception.**
Precise mechanism (their §3.1):

- The training loss is a sum of binary cross-entropies over *all* SATNet variables, input bits `I` and
  output bits `O`: `L = Σ_{i∈I} BCE(z_i, l_i) + Σ_{o∈O} BCE(z_o, l_o)`.
- SATNet does **not modify its input variables**. In the *symbolic* case `z_i = l_i`, so the input-bit term
  is identically zero — harmless. But in the *visual* case the input is `z_i′`, the CNN's features for the
  digit image. Now `z_i′ ≠ l_i`, so **the input-bit term is non-zero and backpropagates the true digit
  label straight into the CNN backbone** — the network never had to "ground" symbols; it was handed a
  supervised digit-classification signal through an unmasked side channel. The SATNet layer rode on top of
  an externally-supervised classifier.

**How it was detected (the control that exposed it):** they applied **output masking** — mask the loss on
the input bits so the only learning signal is the genuine end-to-end one — and re-ran over **10 seeds**.

- Result (their Table 1): visual Sudoku collapses from 18.5%/11.9% (already far below the paper's reported
  93.6%/63.2%) to **0.0% ± 0.0% train and test**. Non-visual Sudoku is *unaffected* by masking (99.7/97.6
  either way), proving the leak was specifically the perception→symbol interface. With masking, "the CNN
  does not ever learn to classify the digits better than chance."
- Even *with* the leak, SATNet failed **8 of 10 random seeds** (mean 18.5% train), i.e. the original number
  was a lucky-seed report. The successful runs revealed an accidental **two-stage curriculum**: training
  accuracy sits at 0 until the leaked labels train the CNN to classify digits, *then* SATNet starts solving
  — confirming the symbolic layer only worked once an externally-supervised classifier existed.

**The sanity test they propose for *any* differentiable symbolic solver:** the **MNIST-mapping problem** —
a deliberately *easy* instance of symbol grounding (classify a digit + learn a bijection to output
variables). Both sub-tasks are individually trivial (digits → 99% for a CNN; permutation learnable in poly
time). A solver that genuinely grounds should hit ~99%. SATNet, naively configured, **fails completely**
(chance-level), and worse — a plain baseline (Sudoku-CNN with the SATNet layer replaced by two FC+ReLU
layers) scores **72.1%**. *When a non-reasoning model beats your reasoning model, the reasoning layer's
raison d'être has disappeared.* Proper config recovers 99%, but the lesson stands: end-to-end
differentiability does **not** imply the symbolic component is doing the symbolic work.

### 1c. The generalization — "reasoning shortcuts" (Marconato et al., NeurIPS 2023)
SATNet is the famous instance; the field has since named the disease. **"Not All Neuro-Symbolic Concepts
Are Created Equal: Analysis and Mitigation of Reasoning Shortcuts"** (arXiv 2305.19951, NeurIPS 2023) and
**"Reasoning Shortcuts in NeSy Continual Learning"** (arXiv 2303.12578) prove a **formal link: reasoning
shortcuts are optima of the loss.** A NeSy predictor can attain perfect task accuracy while its learned
concepts carry **unintended semantics** — the constraint is satisfied by a *different* concept assignment
than the intended one, so the model is right for the wrong reasons and breaks under distribution shift,
intervention, or reuse. Mitigations in this line: concept-level supervision/probing, disentanglement,
counting the loss optima, and the **rsbench** benchmark suite for measuring concept quality. NeurIPS 2025
adds **Prototypical Neurosymbolic** architectures (virtual.neurips/2025/poster/116900) that attack
shortcuts "at their root cause."

**Why this is *exactly* our fear.** GLaDOS's analogue of the SATNet leak is the **cold-weave shortcut we
already observed** (README "what we've found"): a learned organ co-trained from scratch is *ignored* — the
LM shortcuts via prompt text + an abstain prior, and the causal controls go flat. SATNet leaked the label
into the *perception* net; our risk is the LM leaking the answer out of the *prompt text* (or a fixed
abstain prior) so the organ is decorative. Same disease, mirror-image interface. The reasoning-shortcut
theorem says this will happen *by default* wherever the loss admits a cheaper non-deductive optimum — which
is precisely why our controls must be adversarial, not confirmatory.

### 1d. The exact controls we must run — mapped to our causal-control table

| SATNet / reasoning-shortcut control | What it proves | GLaDOS instantiation (we mostly already have it) |
|---|---|---|
| **Output masking** (remove the label-leak term from the loss) | The reasoning is genuinely end-to-end, not piggybacking on a leaked supervised channel | **Audit α's input and the prompt for an answer side-channel.** The answer must not be reconstructable from prompt text alone. Our oracle-readout already runs *with the full problem in the prompt* and shows the LM ignores it — keep that adversarial framing in every woven eval. |
| **`corrupt`/mask the solver path → accuracy must collapse** | The symbolic layer is load-bearing, not decorative | **`corrupt` control (corrupt→0):** flip the query cell's survivors; answer must change/collapse. *Already implemented & passing on oracle-γ (100% OOD; corrupt breaks it).* This is the single most important control — run it on every woven checkpoint, not just the oracle one. |
| **`shuffle` — inject a different problem's solution** | The model reads *this* problem's certified state, not a prior/heuristic | **`shuffle` control:** roll the lattice along the batch. *Already implemented.* Decisive logic verbatim in oracle_readout.py: unchanged-under-shuffle ⇒ bypass. |
| **Concept intervention / permute semantics** (reasoning-shortcuts) | The certified state carries the *intended* semantics, not an aliased one | **`permute` control:** permute candidate/color labels. *Already implemented.* Extend: probe the certified state for genuine variable/value semantics (rsbench-style concept-quality check), not just answer accuracy. |
| **MNIST-mapping-style isolation sanity test** | Each interface piece grounds on its own before they cooperate | **Test α and γ standalone.** γ is done: oracle-lattice→answer readout is our MNIST-mapping analogue and it *passes* (100% OOD, causal controls bite). α (text→typed program) needs its **own** isolated grounding test before any co-training — exactly the staged recipe. Do not infer α works from end-to-end numbers. |
| **Multi-seed reporting (SATNet: 8/10 seeds fail)** | The result is not a lucky-seed artifact; no accidental curriculum | Report woven results over seeds; treat a flat-control result as the *expected* cold-weave null (README), and never headline a lucky seed. Watch for the SATNet "two-stage curriculum" tell: accuracy pinned at floor until something *else* (here: LoRA/prompt) quietly solves it. |
| **Beat the no-reasoning baseline** (SATNet lost to 72.1% FC net) | The organ adds capability a plain net of equal size lacks | Always include a **frozen-LM + prompt-only** and **abstain-prior** baseline. If the woven model doesn't beat them *with the controls biting*, the organ isn't earning its keep. |

The throughline: **soundness (our output check) prevents *wrong* answers, but it does NOT prevent
*bypass*.** A sound check guarantees the answer is correct *if produced by the organ* — it says nothing
about whether the organ, vs the prompt/LoRA, produced it. Only the causal controls (corrupt/shuffle/permute
collapsing accuracy) prove genuine deduction. This is why the README's dense-γ oracle result is decisive:
it is the SATNet test *passed* — the LM reads the lattice and ignores the text. The open job is keeping it
passed once α and a *learned* organ replace the oracle.

---

## 2. Closest prior work (method | what it does | how GLaDOS differs)

Ordered roughly nearest → farther. "Woven" = symbolic computation inside the forward pass with gradient/
dense readback; "tool" = LLM emits text, an external solver runs, result is fed back as tokens.

| Method (year, venue) | What it does | How GLaDOS differs |
|---|---|---|
| **SATNet** (2019 ICML, 1905.12149) | Differentiable MAXSAT SDP layer, learnable clause matrix, backprop through the solver; *woven* | SATNet's solver is **soft/unchecked** (SDP relaxation, can be wrong) and grounding-blind (the 2312.11522 disease). GLaDOS's organ is **output-checked → sound**, narrows a **per-problem α-compiled typed factor graph** (not fixed learned clauses), and ships the **causal controls SATNet lacked**. |
| **DiLA — Differential Logic Layer** (2024, openreview uh3ZO2izyr) | LLM parses NL → SAT spec + an initial variable assignment, then a **differentiable MaxSAT logic layer iteratively refines** it; 100% on its NL-constraint benchmark, 65× faster than SATLM. Survey 2508.13678 calls it the canonical "differential symbolic module" | Closest *named* cousin. But DiLA is a **two-stage external refiner** (LLM → spec → logic layer), *not* woven between layers, and is **not verifier-trained** (no RLVR; the LLM weights aren't updated by the solver). GLaDOS weaves the organ *into* the residual stream (α/γ) and trains it with verifiable reward. |
| **Neurosymbolic Representations via VSA/HRR** (2025, 2502.01657, EMNLP) | Trained linear encoder maps LLM hidden states (layer 17) → Vector-Symbolic (Holographic Reduced) vectors; a **symbolic function computes the answer; a trained decoder writes a hidden state back, 50/50 blended** into the residual stream | **Architecturally the nearest analogue of the α/γ weave** — encode hidden→symbolic, compute, decode→hidden, blend (cf. our zero-init γ). But its symbolic step is a **fixed Python arithmetic function, not a learned/differentiable *deductive* organ**, it is **not checked/sound** and **not verifier-trained**, and it targets closed-form arithmetic, not candidate-set narrowing over a compiled factor graph. |
| **TransNAR** (2024, 2406.09308) | LLM + **frozen pretrained NAR** (graph net), one-directional gated cross-attention readback; OOD algorithmic transfer | Our deepest-analyzed cousin (prior_art_extract.md). NAR is **soft** (wrong OOD) and ingests a **clean external graph** — it never has to *produce* the symbolic input. GLaDOS's **α** (hidden→typed program) is the un-precedented half; our organ is **checked**, not soft. |
| **AutoCoNN** (2024; cited in survey 2508.13678 as the other "differential symbolic module") | Integrates compiled neural networks (CoNNs) with rules via artificially-generated attention weights | Fixed compiled rules injected as attention; no per-problem compiled program, no soundness check, no verifiable-reward training. |
| **Abductive Learning family** — ABL (2019 NeurIPS), ABL-Refl / "Abductive Reflection" (AAAI 2025), **Curriculum Abductive Learning** (2505.12275, NeurIPS 2025), ARLC (Abductive Rule Learner w/ Context) | Neural perception + symbolic KB; abduce pseudo-labels consistent with logic, retrain perception; curriculum/reflection variants tame the abductive search | **Reasoning lives in a discrete external KB + abductive search loop**, not a differentiable organ in the forward pass. GLaDOS keeps deduction **differentiable, woven, and recurrent between layers**, with the check (not a KB) supplying soundness. Closest on *spirit* (sound symbolic core constraining a neural net) but far on *mechanism*. |
| **RLSF — RL via Symbolic Feedback** (2024/25, 2405.16661) | Symbolic tools (solvers/provers/CAS) emit **poly-sized certificates** → **token-level RL reward** to fine-tune the LLM | This is GLaDOS's **verifiable-reward arm without the woven organ** — the solver is an *external grader*, never in the forward pass; the LLM stays a pure text policy. We additionally *embed* the deductor and read its certified state densely. RLSF ≈ our reward design, minus the organ. |
| **Neuro-Symbolic Integration for Causal/Reliable Proofs** (2025 NAACL-Findings 2025.findings-naacl.317), **VeriCoT** (2511.04662), **Deductive Verification of CoT** (2023 NeurIPS) | LLM generates a CoT/proof; an external solver/checker *verifies* each step or the whole chain for faithfulness | Verification is **post-hoc on text**, not a differentiable organ producing the answer. GLaDOS's check is at the output of an *embedded* deductor that *generates* (via γ→LM-head), not a checker grading free-text CoT. |
| **Tool/solver-aided LLMs** — Logic-LM (2023 EMNLP), LINC (2023 EMNLP), SATLM (2023 NeurIPS), LLM+P (2023), Logic.py (2026 ICLR), MCP-Solver (2501.00539) | LLM autoformalizes NL → SAT/SMT/ASP/PDDL/DSL; an **external** solver runs; answer fed back as tokens | The dominant paradigm and the **clearest contrast**: the solver is a **black-box external tool**, no gradients flow, the LLM never *reads the solver's internal certified state*, nothing is verifier-trained end-to-end. GLaDOS's bet is that **woven + differentiable + dense-readback** beats the text round-trip bottleneck (TransNAR's readback-bandwidth lesson). |
| **DeepProbLog (2018), Logic Tensor Networks (2022), Logical Neural Networks (2020), Semantic-Loss / MultiplexNet** | Differentiable probabilistic-logic / constraint layers over **fixed, given** logical structure | These **assume the program/rules are given**; they don't *compile a per-problem typed factor graph from an LLM hidden state*, and they are the very systems the reasoning-shortcut theorem (2305.19951) was written about. GLaDOS adds α (learned compilation) + the woven LLM + verifiable reward. |

Survey placement (2508.13678, "Neuro-Symbolic AI: Towards Improving the Reasoning Abilities of LLMs",
IJCAI 2025): three top-level families — **Symbolic→LLM** (data), **LLM→Symbolic** (external modules:
solver-aided / program-aided / tool-aided / search), and **LLM ++ Symbolic** (end-to-end hybrid). GLaDOS
sits squarely in **LLM ++ Symbolic → "Differential Symbolic Module" (§6.2)** with a foot in **"Symbolic
Feedback" (§6.3)**. The survey's *only* exemplars of §6.2 are **DiLA and AutoCoNN** — i.e. the
differentiable-woven-solver cell of the taxonomy is nearly empty, and neither exemplar is checked/sound +
verifier-trained.

---

## 3. Honest novelty verdict

**What is genuinely unoccupied (GLaDOS's claim):** a **per-problem-compiled, checked/sound, differentiable
deduction organ woven into a pretrained LLM's residual stream (α/γ) and trained by verifiable reward.** No
single system in the crawl holds all five of: *(i) woven in the forward pass* (DiLA, RLSF, tool-LLMs are
not), *(ii) differentiable with dense structured readback* (tool/abductive/verification methods are not),
*(iii) output-checked → sound* (SATNet, TransNAR, AutoCoNN, the VSA method are soft), *(iv) a per-problem
compiled typed program rather than fixed rules* (DeepProbLog/LTN/AutoCoNN/SATNet's learned-but-fixed clauses
are not), *(v) verifier/RLVR-trained* (only RLSF + the tool-verifier line, and those aren't woven). The
survey's own §6.2 cell (DiLA, AutoCoNN) is the nearest neighborhood and it is sparse and uniformly
*un-sound + un-verifier-trained*.

**Where the novelty is thinner than it feels.** Each *ingredient* has precedent: differentiable solver
woven (SATNet, DiLA); hidden↔symbolic encode/compute/decode with residual blend (2502.01657 — strikingly
close to α/γ in shape); frozen-symbolic-core + dense readback (TransNAR); verifiable symbolic reward
(RLSF). GLaDOS is a **novel *combination* + the soundness twist**, not a from-nothing primitive. The
honest framing: *"first to weave a **checked** per-problem deductor into an LLM and train it with verifiable
reward,"* and the closest single point of comparison to actively differentiate against in writing is
**DiLA** (named cousin, in the survey) and **2502.01657** (nearest architectural shape).

**The cautionary verdict.** The literature's strongest, most reproducible result about systems like ours is
a **negative** one: SATNet's grounding collapse + the reasoning-shortcut theorem say that a loss admitting a
cheaper non-deductive optimum **will** be exploited, and end-to-end differentiability hides it. Our
already-observed cold-weave shortcut is the same phenomenon. Therefore the soundness check is **necessary
but not sufficient** — it buys correctness-given-the-organ, never load-bearingness. Our edge over the whole
field is not the organ per se but the **discipline**: the corrupt/shuffle/permute controls
(oracle_readout.py) are exactly the controls SATNet didn't run, and the dense-γ oracle result is the SATNet
test *passed*. Keep every woven claim gated behind those controls biting, isolate α with its own grounding
sanity test before co-training, and report over seeds. That is the difference between a real GLaDOS and an
award-winning artifact that turned out to be reading the labels.

---

### Sources
- SATNet — Wang, Donti, Wilder, Kolter. ICML 2019. arXiv 1905.12149 (read: pdfs/).
- Grounding critique — Chang, Flokas, Lipson, Spranger. NeurIPS 2020. arXiv 2312.11522 (read in full: pdfs/).
- Reasoning shortcuts — Marconato et al. NeurIPS 2023, arXiv 2305.19951; continual-learning variant 2303.12578; rsbench; Prototypical NeSy (NeurIPS 2025, poster 116900).
- NeSy×LLM survey — "Towards Improving the Reasoning Abilities of LLMs", IJCAI 2025, arXiv 2508.13678 (taxonomy: §6.2 Differential Symbolic Module = DiLA, AutoCoNN; §6.3 Symbolic Feedback).
- DiLA — openreview uh3ZO2izyr (2024). Neurosymbolic VSA/HRR — arXiv 2502.01657 (EMNLP 2025). TransNAR — 2406.09308 (see prior_art_extract.md). RLSF — 2405.16661. Curriculum Abductive Learning — 2505.12275 (NeurIPS 2025). ABL-Refl — AAAI 2025.
- Awesome-LLM-Reasoning-with-NeSy (LAMDASZ-ML) — crawled for the tool/solver/verifier/abductive entries above.
