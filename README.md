# clairnets

A research program on **param-efficient, sound-reasoning architectures** — mixing recent primitives
(Clifford geometric product · lattice abstract-interpretation · recurrent deduction · automata shortcuts
· edge-of-stability) into things that reason reliably at tiny scale, then asking whether the efficiency
survives a compute-matched instantiation.

> **Honesty first.** The sound-lattice-deduction core is *not ours* — it's the Lattice Deduction
> Transformer (LDT, arXiv 2605.08605). Our seam is narrow: **(a)** the Clifford *wedge* as a
> pairwise-exclusion proposer, and **(d)** welding a sound deductor *inside* a pretrained LLM, woven
> across layers. We say "empirically sound on this distribution," never "certified." See
> [`notes/prior_art.md`](notes/prior_art.md).

## Threads

### 1. GLaDOS — Geometric Lattice Deduction Over Streams  *(the main bet)*
A recurrent reasoner whose state is a point in a per-position powerset lattice; a Clifford
geometric-product cell *proposes* candidate eliminations, the lattice *meet* narrows soundly, and a
conflict head lets it *abstain*.  `a_{t+1} = a_t ⊓ Π(f_θ(a_t, p, m))`.
- `clair/glados.py` — the organ: 3 iso-param arms `ldt` / `glados` / `nowedge`.
- `clair/sudoku.py` — powerset-lattice state, on-policy alpha-target, MEET/branch solve loop.
- `clair/run_glados.py` — the decisive core-swap + the metric that decides it (**false-elimination rate**),
  plus solve-rate / wrong-return / p90-forwards.
- `clair/domains.py`, `clair/run_gauntlet.py` — the domain-general gauntlet (graph-coloring · 3-SAT · maze),
  one organ many lattices, hold-out-a-domain zero-shot soundness.

### 2. Geometric transformers  *(is the Clifford product a viable efficient LM primitive?)*
- `clair/graft.py` — swap a real **OLMo-2/3** model's SwiGLU FFNs for geometric-product mixers (`GeomFFN`),
  freeze everything else, distill to recover; read off *recovered fraction* + *param ratio*.
- `clair/model.py`, `clair/tasks.py`, `clair/train.py` — the 4-arm iso-param toy harness
  (`std`/`attn`/`geom`/`geom_recur`) that reproduced FFN-redundancy (geom mixer solves modadd at ~42%
  fewer params than SwiGLU).

### 3. The augmented LLM  *(Layer 1, future)*
A Clifford *cortex* compiles a problem into a constraint program and writes it into the GLaDOS *deductor*
(zero-init gated graft into OLMo-3, gradients through the solver, tapped across layers); baseline to beat
= OLMo-3-Think. The genuinely novel combination — gated by Thread 1 first.

## Layout
- `notes/` — regrounded paper mechanics (`papers.md`), layered architecture (`architecture.md`),
  task ladder + training regime (`roadmap.md`), and the prior-art / novelty audit (`prior_art.md`).
- `pdfs/` — the full must-read set (LDT, CliffordNet, HRM + ARC-Prize critique, TRM, SATNet, …).
- `report/clairnets.typ` — the living writeup / lab notebook.
- `infra/box.sh`, `infra/threeway.sh` — sync + run on the GPU box.

## Status
Layer-0 decisive 3-way (Sudoku-Extreme, iso-param) running. Gauntlet built. Co-processor graft is next,
gated by whether the wedge lowers false-elimination at equal params.
