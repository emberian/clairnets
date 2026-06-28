# What "sound" means in GLaDOS — the one canonical statement

This is the single place we define the word. Everywhere else in the repo that says
*sound / certified / verified / guarantee* points here. The rule is short:

> **"Soundness is a permission" cashes out ONLY where a boundary check exists** — i.e.
> at training/eval, where the witness generator gives us ground truth, or at inference
> in a verifier-bearing domain (code tests, a Lean kernel). Everywhere else, "sound"
> is *conditional on an unverified compile* — a hope, not a guarantee.

A claim of soundness is meaningless until you say **what it is relative to** and **when
the check that cashes it out is available**. There are four distinct levels, and they
are NOT interchangeable.

## (A) Operator soundness — REAL, but narrow

A certified op (`ArcConsistency`, `FactorConsistency`, `Modular`/SNF, `GF2RowSpace`,
`Macro`, `ExactDedP`) **provably never eliminates a value used by some solution of the
CSP it is given**. `Certificate.sound == True` means exactly this: `reduce(s) ⊆ s` as a
γ-narrowing, drops no real solution. The meet of sound narrowings is a sound narrowing,
so the certified floor composes and stays sound. This is a genuine theorem (the Lean
LDT proof; the Modular/GF2/Macro ops are cross-checked vs brute force).

**The bound:** it is conditional on *the given CSP being the right problem*. The op is
sound **relative to the CSP handed to it** — nothing here says that CSP is your question.

## (B) Composer soundness — REAL, relative to `csp_α`

The verifier gate (`compose._verifier_gate`, against `csp.exact_dedP`) lets the certified
ops override the neural narrower, so the **composed lattice is sound with respect to
`csp_α`** — the instance the composer was actually run on. The neural organ **alone is
NOT sound** (held-out false-elim 0.5–3.3%); it is "sound only under the gate," and only
relative to `csp_α`. So `false-elim == 0` in the self-tests is a (B)-level fact: zero
relative to the exact verifier *on the CSP we composed*.

## (C) Compile soundness — THE GAP, uncertified

`csp_α` is whatever `α` (a `DenseLatentProjector` / `StructureRack`, a **learned net**)
emitted from the host hidden state. **Nothing certifies that `csp_α` matches the actual
question.** So the whole chain is "sound **modulo an unverified compile**": sound *if* `α`
compiled correctly — and **α-correctness carries no certificate**.

This is not hypothetical. The **SHUFFLED diagnostic** (`clair/organ/shuffled_diag.py`)
proves it: feed the composer the *wrong* structure and the certified floor solves the
wrong problem and emits a **certified-correct answer to the WRONG question** —
confidently wrong, full (A)/(B) certification intact. α-correctness is the one
uncertified link in the chain.

## (D) Answer soundness — REAL, but needs an output check

Checking the *final answer against the true instance* is only possible when we **have**
the boundary:

- **train / eval:** the witness generator gives ground truth, so `output_check`
  (`answer ∈ exact_dedP(csp_true)[query]`) is a real, sound reliability gate.
- **inference in a verifier-bearing domain:** a domain-native verifier exists — code
  unit tests, a Lean/proof kernel — so the answer is checkable at inference with no gold.
- **arbitrary NL with no ground truth and no domain verifier:** **NO answer-soundness
  guarantee exists.** Here "sound" collapses to (C): conditional on an unverified compile.

## Why this is the whole thesis, not a footnote

α-correctness is the one uncertified link (C). That is *exactly* why **verifier-bearing
domains — code, theorem-proving — are where the soundness claim becomes real at
inference**: the domain itself supplies the (D) boundary check that the NL setting lacks.
On the "easy regime" benchmarks the *true* structure is handed to the composer, so (D)
holds for free (ground-truth structure = the boundary), and the certified floor carries
answer-correctness — but that is a property of *being given the answer's structure*, not
of α. Move to α-emitting-structure-from-text and (C) is live again.

## How to talk about it (style rule)

- An **operator** is "sound relative to the CSP it is given" (A) — state it as the
  theorem it is, just bound it.
- The **composer / woven lattice** is "sound relative to `csp_α`" / "sound modulo an
  unverified compile (α)" (B/C) — never bare "sound."
- The **answer** is "verified where ground truth or a domain verifier is available
  (train/eval; code/proof at inference); unverified otherwise" (D).
- Never write "certified ⇒ correct answer" on arbitrary input. There is no such arrow.
