# clairnets

A research program on **reasoning architectures that propose freely and stay reliable at the boundary**.
The spine, after a lot of honest correction:

> A **learned geometric-product organ** proposes/compiles/narrows; a **lattice + cheap output check** is
> the authority for reliability. Soundness is *free at the output* (proven, see the Lean files) — which is
> a **permission to make the deductor loose, learned, and differentiable**, not a prescription to make it a
> verifier. **Completeness** (does the learned organ reach a checkable answer vs abstain, and does it
> size-generalize) is the real research variable.

Grounded in: CliffordNet (geometric product, arXiv 2601.06793), the Lattice Deduction Transformer
(arXiv 2605.08605) + its Lean formalization (`~/dev/graphplay/Graphplay/Integrations/LDT*.lean`), and the
automata-shortcuts lens (2210.10749). See `notes/geometric_product_options.md` and `notes/prior_art.md`.

## What we've actually found (honest)

| result | verdict |
|---|---|
| **(d) host compiles → checked deductor → abstains** (`induce`) | **the one real positive** — gains calibrated abstention that survives distribution shift (abstain-recall gap ~0 in-dist → **+30 OOD**); modest on raw accuracy; single-seed |
| geometric product as a **channel mixer** (LM + vision) | **null, regime-caveated** — ties SwiGLU/MLP at iso-param; wins only low-param. It's a *fine* mixer, not a superior one |
| **GLaDOS-on-sudoku** | we *broke* it (under-implemented LDT's backtracking search) → parked; faithful `ldt.py` written |
| **exact CSP harness** (`csp.py`) | **the ground truth** — reproduces the Lean's proven facts (soundness free; completeness = lattice-level × width) |
| **neural proposer vs harness** | **sound** (dominates exact `dedₚ`, beats arc-consistency) but ffn/inner/wedge/full **tie** at the per-cell ceiling — that regime can't separate the mixer |

The throughline: the seductive bets (wedge, geom-as-mixer, GLaDOS) died or need fair retries; the bet that
stands is **neural-proposes / checked-at-the-boundary**, validated weakly by induce.

## Where it's going
**The augmented LLM** (`augmented.py`): frozen **OLMo** + zero-init gated adapters that **write** a constraint
program → a **learned, differentiable, woven** lattice organ narrows it (gradients flow, co-trains) → host
**reads back** → answer/abstain, end-to-end, with the output check as the *only* soundness mechanism.
The make-or-break question: **can the LM learn to program a learned deductor from text and gain reasoning the
base model can't fake?**

## Layout
- `clair/csp.py` — **exact finite-CSP harness** (ground truth: exact solutions, exact `dedₚ`, per-cell AC +
  pair path-consistency, polymorphism diagnostic, completeness/false-elim metrics).
- `clair/proposer.py` / `run_proposer.py` — factor-graph neural proposer, measured against the harness.
- `clair/induce.py` / `run_induce.py` — the toy (d) bet that worked (calibrated abstention + size-gen).
- `clair/augmented.py` / `run_augmented.py` — OLMo + learned woven deductor *(in progress)*.
- `clair/glados.py`, `ldt.py`, `cliffordnet.py`, `rope_lm.py`, `geom_lm.py`, `organs.py` — parked/superseded
  experiments (kept for reference; see commit history + `notes/` for what each found).
- `notes/` — `geometric_product_options.md` (the current design doc), `papers.md`, `prior_art.md`, `roadmap.md`.
- `pdfs/` — the must-read set. `report/clairnets.typ` — living writeup.

## Discipline (learned the hard way today)
- Don't conclude from underpowered or mis-aimed runs; test a method in the regime it targets.
- The soundness proof is a *permission* (be loose, check the output), not a *prescription* (don't build the
  checker into the deductor — that's just "LLM phones a SAT solver").
- Run a portfolio; let the data, not the enthusiasm, pick the next bet.
