# Re-grounding in the LDT — I misread the bounded-width dichotomy as a failure

Read the foundational research (pdfs/2605.08605, graphplay/.../LDT*.lean, research/ldt_theory_for_authors.md,
ldt_assessment.md). My "Finding 3 → the neural narrower is the weak link, GLaDOS is just a certified-bank-compiler"
**reframe was wrong** — it forgot what the LDT actually is and proves. Correcting the record.

## What the LDT actually is (and proves, machine-checked)
- Abstract-interpretation over the per-cell candidate lattice (Galois α ⊣ γ). `dedₚ = α(γ(a) ∩ ‖p‖)` = the best
  abstract transformer = arc-consistency-to-fixpoint. A **looped transformer LEARNS to approximate dedₚ.**
- **TWO loops, not one.** Inner = the learned dedₚ-narrower. **Outer = a DPLL SEARCH loop: narrow → if
  undetermined, BRANCH → backtrack on conflict.** The LDT is deduction **+ branching search**, output-checked.
- **Soundness is FREE + training-independent** (`checkedSolve_sound` holds for ANY solve map; the identity map is
  sound-but-useless). It's a property of the output check / certificate, not the net.
- **Completeness = the bounded-width CSP dichotomy (Feder–Vardi, Barto–Kozik), a THEOREM.** Local consistency
  (dedₚ) solves EXACTLY bounded-width CSPs. Affine/XOR over GF(2) is **unbounded width — provably NOT solvable by
  per-cell deduction** (`acStep_xor_sound_but_abstains`: a sound deduction that ABSTAINS on a uniquely-solvable XOR
  system, because the global linear coupling is invisible per-cell). Affine is "solvable only by **branching or
  linear algebra**."

## How I lost the plot
My Finding-3 experiment found: the neural narrower gets ~2% on xor_global (affine), the certified GF2 floor carries
it, and "the regimes don't overlap." I called that **"the bet isn't realized, drop the narrower."** But that result
is **`acStep_xor_sound_but_abstains` reproduced at scale** — the machine-checked CENTER of the LDT, working exactly
as proven. The deduction organ is SUPPOSED to abstain on affine. That's not a failure; it's the dichotomy.

The narrower is **not the weak link** — it IS the learned bounded-width deduction leg, doing precisely what the
theory says (solves bounded-width, abstains on affine). "exact_dedP is cheap where the organ works" is just
"bounded-width is poly-time" — not evidence the organ is redundant; the organ is the *learned, differentiable,
LM-coupleable, composes-with-search* form of dedₚ, which is the whole LDT research bet.

## The ACTUAL gap (what I should have concluded)
The theory says affine is crossed by **branching OR linear algebra**. GLaDOS:
- ✅ has the **linear-algebra leg** — the certified `GF2RowSpace` faculty, and Finding 2 (ternary vocab) just made
  α able to EMIT xor and ROUTE to it (parity-F1 0.57 → GF2-solve 0.575). So the linalg path to affine is now real.
- ❌ **DROPPED the branching/search outer loop** — the DPLL "branch-when-deduction-stalls + backtrack-on-conflict,
  output-checked" leg that is HALF of the LDT and the general way past a deduction stall. GLaDOS kept only
  deduction + certified-ops + abstain.

So the real next thing is **not "remove the narrower"** — it's **restore the search leg** (and lean on the GF2
linalg leg Finding 2 enabled). Deduction (bounded-width) + branch/search (the rest) + certified linalg (affine) +
output-check (free soundness) = the complete LDT picture. We have three of four; the missing one is search.

## What survives from the recent findings (re-read correctly)
- **Finding 1** (edge-MP for long-chain α-compile) — real α improvement, stands.
- **Finding 2** (ternary vocab → α emits xor → routes to GF2) — this is *the linear-algebra leg the theory names*.
  Even more important than I framed it.
- **Finding 3** — CONFIRMS the bounded-width dichotomy empirically (good!), and points at the missing search leg.
  Retract the "drop the narrower / GLaDOS is just a certified-bank-compiler" conclusion.

In one line: **soundness is in the check; completeness is bounded-width; affine needs branch-or-linalg — and we
dropped the branch leg.** That's the plot.
