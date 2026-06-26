# GLaDOS glossary

The project moves fast and invents vocabulary. This is the decoder ring.

### Core bet
- **organ** — a learned, differentiable *reasoning* module woven into the LLM (not a tool it calls). The bank
  has three: narrow, chain, energy.
- **the spine** — *propose loosely, check at the output.* The organ can be loose/learned/general; reliability
  comes from an exact verifier on the final answer, not from the organ being a verifier itself.
- **soundness vs completeness** — *soundness*: never commit a wrong answer (never eliminate a true
  possibility). *Completeness*: actually reach a checkable answer instead of abstaining. Soundness is cheap
  (check the output); **completeness is the research variable.**
- **checked / verifiable reward** — the answer is graded by an *exact* checker (`clair.csp`, FOL
  forward-chainer, z3). Free, un-hackable, the source of all training signal (RLVR).

### The lattice / deduction
- **lattice** — the space of *candidate sets*. A per-cell lattice tracks, per variable, which values are still
  possible; ordered by inclusion (smaller = more narrowed; ⊤ = "anything", ⊥ = contradiction).
- **candidate set / survivors** — the values still possible for a cell given what's known.
- **meet / narrow** — the monotone operation that *removes* possibilities (the NARROW organ). Only shrinks.
- **dedₚ / `exact_dedP`** — the *exact* per-cell transformer: keep exactly the values that appear in *some*
  full solution. The strongest sound per-cell narrowing; the supervision target + ground truth.
- **factor** — a constraint as `(scope, allowed-tuples)`: which combinations of values on which cells are
  legal. Any arity. The general-purpose representation.
- **arity** — how many cells a factor touches (unary pin / binary ≠ / ternary affine …).
- **abstraction level** — per-cell (level-0) / pair (level-1) / triple (level-2) / factor (level-k). Higher
  levels capture more correlation. `levels.py` maps which rung needs which level.
- **the affine wall** — 3-way constraints (XOR, modular arithmetic) that a per-cell lattice *provably* cannot
  represent; needs level-2/triple. The recurring ceiling for all three organs.
- **abstain / unknown** — the honest output when the problem is underdetermined: a candidate *set* with >1
  element, not a guess. The set-valued readout makes this first-class.

### The organ bank (three inference shapes)
- **narrow** (`proposer.py`) — *rule out* possibilities (meet). Sound, abstains. For constraint satisfaction.
- **chain** (`chain_organ.py`) — *derive* new facts (the dual: monotone *growth* / join). For entailment/proof.
- **energy** (`energy_organ.py`) — *relax toward a fixed point / optimum* (mean-field DEQ). Uniquely does
  **optimization** (find the best, not just any). Soft constraints.
- **portfolio + verifier-selection** — run several organs, let the *exact verifier* pick the sound output.

### The weave (LLM ↔ organ)
- **woven / "D"** — the organ inside the LLM's forward pass with high-bandwidth, *causal* influence (vs a
  bolted-on tool result or a terminal read-out head).
- **α (alpha) / compile** — the LLM→organ direction: read OLMo's hidden state and emit a *typed factor-graph
  program* (variables, domains, factors). Must **compile**, not **solve** (emitting the answer = a bypass).
- **γ (gamma) / readout** — the organ→LLM direction: write the narrowed lattice *densely + structurally* back
  into the residual stream (zero-init gate). The *oracle de-risk* proved this works.
- **zero-init gate** — the write-back starts as an exact no-op (tanh(0)=0), so installing the organ doesn't
  damage the pretrained model; it only earns influence through training.
- **causal control / ablation** — the test that matters: shuffle/permute/corrupt the organ's output and see if
  the generated answer changes. If it doesn't, the organ is being *ignored* (a bypass).
- **the latch gap** — "the organ knows the answer but the LM won't say it." Was a *bandwidth* problem; dense γ
  dissolves it.

### Training
- **RLVR** — RL from Verifiable Rewards (GRPO on the exact output-check). Trains the *behavior* (right answer /
  right abstention).
- **dominate-dedₚ** — the soundness-asymmetric loss: heavy penalty for eliminating a value the exact transformer
  keeps (unsound), light penalty for keeping an extra (loose-but-sound).
- **staged recipe** — train organs to excellence *separately* → pretrain α and γ on their sub-tasks → assemble
  → consolidate. The alternative to cold end-to-end co-training (which fails).
- **witness-first** — generate a solution first, emit only facts it satisfies. Solvable by construction, exact labels.
- **rung / verifier ladder** — a task family with an exact checker (CSP → FOL → SMT → … → Lean). The curriculum climbs it.

### Other
- **GA grades / wedge / trivector** — geometric-algebra grade-k = arity-k interaction (wedge=binary,
  trivector=ternary). A *representation/completeness* bias for the factor deductor — **not** a soundness fix.
- **shortcut vs iterate** — a transformer can *shortcut* a T-step deduction to O(log T) depth, but shortcuts
  are brittle OOD. Default to *iterating* the true operation; shortcuts are checked speculative acceleration.
