# GLaDOS — the layered architecture

ember's framing (2026-06-25): geometric lattice deduction is **one module**, not the whole model.
"Full GLaDOS" is an **augmented transformer** — a host sequence model with a sound deduction organ.

## Layer 0 — the organ (what we build & falsify FIRST)
**Geometric Lattice Deduction core** (`clair/glados.py` + `clair/sudoku.py`).
- state = point in a product powerset lattice (per-position alive-candidate mask).
- recurrent cell PROPOSES via Clifford geometric product (dot+wedge); lattice MEET narrows.
- on-policy alpha-target supervision; asymmetric BCE (never kill a live candidate) = soundness.
- abstain via the conflict (CLS) head.
- **decisive experiment:** iso-param 3-way ldt / glados / nowedge on Sudoku-Extreme.
  metric = false-elimination rate + p90 forward passes. If the wedge doesn't help → the organ is
  just an efficiency primitive and the whole tower below is unjustified. THIS GATES EVERYTHING.

## The one-sentence architecture (ember, 2026-06-25)
> a Clifford transformer PROGRAMS constraints into the lattice deducer and has reliable
> lattice-powerset problem-solving as part of its architecture.

**Cortex** (Clifford host) compiles the problem into a CONSTRAINT PROGRAM (initial candidate sets +
exclusion/clause graph) and writes it into the **Deductor** (GLaDOS organ), which runs sound narrowing
and returns (narrowed candidates, ⊥). The verb is "programs", not "calls": the interface is
differentiable end-to-end AND the solver is learned-but-sound.

Why Clifford host is NOT arbitrary: the wedge u∧v IS the pairwise-exclusion operator ("these two
can't coexist") — the exact primitive a powerset constraint graph is built from. The cortex's wedge
features become the deductor's exclusion edges; both halves speak ONE algebra. Deep coupling.

Beats LLM→external-SAT-solver: discrete text tool-calls are non-differentiable and the solver can't be
learned to tolerate fuzzy perceptual input; here the cortex gets gradient on "did I state the problem
right", and inherits the ⊥ channel (knowing when it doesn't know) that vanilla transformers can't have.

**Host base = OLMo 3** (Ai2, Nov 2025): fully open, checkpoints across the ENTIRE training run (the
released-checkpoint requirement), Dolma 3, + a reasoning variant OLMo 3-Think 7B/32B. Graft target:
OLMo-3-Base-7B (frozen base + organ fits one L40S). Baseline to beat: OLMo-3-Think-7B.

**Layer-1 kill-risk:** does the cortex LEARN to emit valid constraint programs, or route around the
organ? Mitigations: (1) zero-init gated graft (Flamingo tanh-gate → step-0 hybrid == OLMo exactly, no
damage, then learns to use the organ); (2) curriculum where the ONLY path to the answer is the organ;
(3) direct supervision on emitted constraints where ground-truth structure is known.

## Layer 1 — the co-processor (full GLaDOS)
A host transformer (causal, possibly geom-mixer itself) over a token/patch **stream**, augmented with
a **lattice workspace** (a bank of abstract-domain scratchpad cells). Three couplings per host block
(or every k blocks):
- **WRITE / encode:** host hidden states project evidence into the lattice (initialize alive-masks,
  inject soft constraints) via cross-attention `lattice ← host`.
- **DEDUCE:** the organ runs K rounds of sound geometric-lattice narrowing on the workspace,
  independent of the host, producing a narrowed candidate state + a conflict (⊥) signal.
- **READ / decode:** host attends back `host ← lattice` to condition next-token prediction on the
  (certified) narrowed state.
- Gradients flow through all three; the organ keeps its on-policy deduction loss as an auxiliary head.

**Why it's more than a transformer:** it has a reasoning channel that can ABSTAIN. Vanilla
transformers confabulate; the organ is empirically sound, so the host can *trust-or-defer*.

## Layer 2 — streams & lifting (scale + generalization)
- **Over Streams:** the host feeds the lattice over time; deduction is a stream of narrowing steps.
- **Semigroup binary-lifting** (automata 2210.10749): distill F₂≈Π(F₁∘F₁), F₄, F₈ → O(log T) macro
  deductions the host can call → cheaper test-time search, emergent closure rules.
- **Multimodality enters here:** product lattice A_obj × A_rel × A_text × A_answer; image patches
  propose object/relation candidates, text proposes constraints, the lattice narrows the answer.
- **Domain-general training (the "generalizes to nonsense" claim):** one organ, many lattices —
  sudoku/maze/graph-coloring/SAT/ARC-grids through one interface; hold out a domain, test soundness.
- **Edge-of-stability selection:** pick checkpoints by sharpness-dimension / participation-ratio, not
  val-loss (instrument with the gowexp trajectory probes).

## Build order (each gates the next)
1. organ core-swap on Sudoku-Extreme (Layer 0) — IN PROGRESS.
2. organ multi-domain (one cell, sudoku+maze+graph-coloring) — the domain-general claim.
3. co-processor graft into a small host LM (Layer 1) on text+symbolic joint tasks.
4. binary-lifting + multimodal product lattice (Layer 2).
