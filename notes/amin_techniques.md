# Amin-lab techniques → implementable for GLaDOS

Read + extract pass (2026-06-26). PDFs in `pdfs/`, text dumps in `pdfs/txt/`. For each paper:
the concrete mechanism (with quoted constructions), then the actionable takeaway for GLaDOS — a
"sea of checked, learnable, exact-verifiable reasoning organs woven into an LLM that learns to compose them."

**Papers read (all downloaded + read in full or core sections):**
1. Compiling to **Linear** Neurons — Velez-Ginorio, Amin, Kording, Zdancewic, POPL 2026 (arXiv 2511.13769) → `pdfs/compiling_linear_neurons.pdf`
2. Compiling to **Recurrent** Neurons — same authors, PLDI 2026 (arXiv 2511.14953) → `pdfs/compiling_recurrent_neurons.pdf`
3. **VerMCTS** — Brandfonbrener, …, Amin, NeurIPS-MATH-AI 2024 (arXiv 2402.08147) → `pdfs/vermcts_full.pdf`
4. **Multi-stage Relational Programming** (staged-miniKanren) — Ballantyne, Sanna, Hemann, Byrd, Amin, PLDI 2025 → `pdfs/staged_minikanren.pdf`
5. **Reasoning about "Reasoning about Reasoning"** — Zhang & Amin, POPL 2022 → `pdfs/reasoning_about_reasoning.pdf`; and **Collapsing Towers of Interpreters** — Amin & Rompf, POPL 2018 → `pdfs/collapsing_towers.pdf`

---

## HEADLINE VERDICT (the killer question)

> **Can we compile our certified primitives (interval constraint propagation, GF(2) Gaussian
> elimination, HM unification, AC narrowing) directly into neural organs — factory-as-compiler?**

**No — not for those primitives, not with this technology, today.** The Cajal compilers are a *real,
proven* "program → exact differentiable neural net" pipeline, but the **class of programs they cover is
far narrower than our organs need**, and the gap is structural, not incidental:

- **Cajal(2,⊸) covers**: a simply-typed **linear** λ-calculus over **finite/boolean** data → exact **linear
  maps** (matrices). Higher-order → **hypernetworks**. Conditionals → **soft-branch**.
- **Cajal(⊸,2,ℕ) adds**: primitive recursion / iteration over **ℕ** (counters) → **linear recurrent
  neurons = linear dynamical systems**. The iterated body must itself be a **fixed linear map**.
- **It does NOT cover** (the authors say so explicitly): **infinite/algebraic data — lists, trees,
  unbounded terms** ("No language for directly programming neural networks supports infinite data";
  trees are deferred to unproven future "recursive neurons"); **nonlinear/real ops**; **affine** functions
  (needs a different metatheory); and anything whose **state space is large** (dimension blows up:
  `dim(τ₁⊸τ₂)=dim(τ₁)·dim(τ₂)`, finite data with *n* values needs an *n*-dim one-hot space).

Map that onto our four primitives:

| primitive | what it needs | in Cajal's class? |
|---|---|---|
| **Interval constraint propagation** | real-valued domains, `min/max/÷`, unbounded, nonlinear | **No** (nonlinear, real, infinite) |
| **GF(2) Gaussian elimination** | finite field Z₂, but row-pivoting = data-dependent search, XOR over a width-N system | **No in practice** — XOR is linear *over GF(2)*, but Cajal compiles to **ℝ**-vector spaces (the FinVec semantics over finite fields is "the wrong kind"); pivoting is conditional control; one-hot over the configuration space is exponential |
| **HM type unification** | unbounded terms (trees), union-find, occurs-check, *non-linear* variable use (duplication) | **No** (infinite data; duplication violates linearity) |
| **AC narrowing** | term rewriting modulo AC, unbounded, search-heavy | **No** (infinite data + search) |

And even where something *is* finite+linear, the **correctness theorem degrades under linking**: it is
exact only on inputs in the **compiled image** `⟦Δ⟧` (one-hot / well-typed vectors). For arbitrary inputs
piped in from a nonlinear host the authors retreat to *"The compiler is correct if the neurons effectively
learn the task"* — i.e. **empirical, not proven**, and large-norm inputs *"can destabilize learning."*
That is *literally our own lesson* (organ states aren't sound off-distribution; α is the hard part), now
with a formal account: **exactness is a property of the input image, not of arbitrary host activations.**

**But the work is still high-value for GLaDOS** — not as an organ-*factory*, but as (a) a **design-pattern
proof** for the whole weave, and (b) a source of **two drop-in primitives**: the *soft-branching
conditional* (differentiable exact-at-one-hot dispatch) and the *link-exact-against-learned* result. Keep
the heavy numeric kernels as **classical certified contractors** (the codex_zoo_review conclusion stands);
borrow Cajal's compiler for the **finite control / routing / dispatch glue**, not the math kernels.

---

## 1. Compiling to Linear & Recurrent Neurons (THE priority)

### Mechanism — Cajal(2,⊸): discrete programs → exact linear neurons

- **Types → finite ℝ-vector spaces**, dimension = number of *distinct* values:
  `⟦2⟧ = ℝ²`, `⟦τ₁⊸τ₂⟧ = ℝ^(dim τ₂ × dim τ₁)`. Booleans map to a **basis (one-hot)**:
  `⟦tt⟧=[1 0]ᵀ`, `⟦ff⟧=[0 1]ᵀ`. Values must map *injectively to a basis* — that is what makes exact
  linear maps recoverable.
- **Linearity is load-bearing**: a *linear* λ-calculus forbids duplicating/discarding variables
  (no contraction/weakening; contexts are *split* `Δ₁∘Δ₂`, not shared). "Their impermissibility
  establish a remarkable connection between linear programs, linear maps, and linear neurons." A
  program that uses a variable exactly once ⇔ a linear map ⇔ a matrix of connection weights.
- **Functions → matrices**, recovered by running the compiler on each basis input and reading off
  columns. **Higher-order functions → hypernetworks** ("neural networks whose outputs are neural
  networks"), e.g. Church-encoded `eq : 2⊸2⊸2` compiles to a linear map `ℝ²⊸ℝ²ˣ²`.
- ⭐ **THE key trick — the soft-branching conditional.** A hard branch
  `if α₁=1 then ⟦e₂⟧ else ⟦e₃⟧` *"breaks differentiation … the gradient wrt [the condition] is zero
  almost everywhere."* Instead Cajal compiles:

  ```
  ⟦if e₁ then e₂ else e₃⟧(σ) = α₁·⟦e₂⟧ + α₂·⟦e₃⟧     where ⟦e₁⟧ = [α₁ α₂]ᵀ
  ```
  i.e. **weight both branches by the one-hot condition vector**. Exact when `[α₁ α₂]` is one-hot
  (= a real boolean), and **fully differentiable** in the condition so an upstream net can learn to
  produce it. This is a differentiable, exact-at-one-hot *dispatch / multiplexer*.
- **Compiler correctness** (Thm 2): `e⇓v ⟹ ⟦e⟧=⟦v⟧` *and* `e⇓̸v ⟹ ⟦e⟧≠⟦v⟧` (you can predict what the
  neurons will *and won't* do). Note (b) **fails at function types** (`λx.x` and `λy.y` both compile to
  the identity matrix) — exactness is a value-level, ground-type property.
- **Linking** (Thm 3, the practically important and *honest* part): compiled exact-linear neurons compose
  with learned **nonlinear** neurons (receive inputs from / send outputs to them). But the clean theorem
  only covers inputs *"produced by our compiler"*; for arbitrary nonlinear inputs like `[1.3 4.7]` they
  concede the real criterion is empirical (*"correct if the neurons effectively learn the task"*).
- **Empirical payoff**: directly-programmed nets (with the conditional structure compiled-in, the
  perception sub-nets learned) **learn faster + more data-efficiently + are debuggable** (they trace a
  wrong output to a specific compiled sub-path and add a targeted penalty). Strength of benefit tracks
  strength of the correctness theorem the compiler satisfies (`I < T < D = C`).

### Mechanism — Cajal(⊸,2,ℕ): iteration → linear recurrent neurons

- Adds `0`, `succ`, and **`iter{e₁ | y↩→e₂}(e₃)`** (System-T iterator). `ℕ` compiles to an
  **infinite-dim sequence space with finite support** (`n` is **one-hot in time**: `0↦[1 0 0 …]`,
  `1↦[0 1 0 …]`). `succ` = right-shift operator.
- **Iteration compiles to a linear dynamical system** — a sum that selects the *n*-th power of a **fixed**
  recurrence map applied to the base case:
  ```
  ⟦iter{e₁|y↩→e₂}(e₃)⟧(σ) = Σ_{n∈ℕ}  ⟦e₃⟧(n) · (⟦e₂⟧)ⁿ (⟦e₁⟧)
  ```
  "Linear recurrent neurons are discrete-time linear dynamical systems." Differentiability is recovered
  because **"the recurrence, if observed finitely, renders a linear recurrent neuron equivalent to a
  linear map between finite-dim ℝ-vector spaces"** (you differentiate wrt a finite-dim subspace; train
  with a max step count). Example: `iter{tt | x↩→not x}(2)` exactly implements "not, twice."
- **Catch**: the per-step body `e₂` must compile to a **fixed linear map** (iterated by matrix power). A
  data-dependent or nonlinear inner step is out. Real experiment caveat: when the iterated map `vblur`
  has **large norm its repeated application explodes** the output → instability that the spec permits.
- **Next step (unproven)**: lists/trees via "recursive neurons" — explicitly future work.

### Actionable takeaways for GLaDOS

1. **The architecture is now theorem-backed, not just hoped.** "Link an *exactly-compiled* module against
   *learned nonlinear* neurons; the fixed exact part determines part of the function before learning;
   result = faster, more data-efficient, debuggable" **is the GLaDOS weave**, proven (modulo the linking
   caveat). The certified organ = Cajal's compiled-exact part; the LM = the learned nonlinear part. Cite
   this as formal precedent for "soundness is a permission that frees the rest to be learned."
2. ⭐ **Adopt the soft-branching conditional as the differentiable organ-router / dispatch primitive.**
   Routing among the bank (narrow / chain / energy) = `Σ_i route_i · organ_i(x)` with `route` a (near)
   one-hot produced by the LM. It is **exact when the route commits (one-hot) and differentiable
   everywhere** — exactly the property you want for "the LM learns which organ to invoke" without a
   non-differentiable hard switch. This is the in-forward-pass complement to VerMCTS's discrete search.
3. **The honest catch is GLaDOS's own discipline, formalized.** Exactness holds only on the compiled
   image `⟦Δ⟧`; off-image (large-norm, off-distribution) host activations break it. → keep the
   output-check at the boundary; bound/normalize γ-readback and α-compiled vectors; never trust organ
   internals on arbitrary activations. This *is* the "α is the hard part / states aren't sound
   off-distribution" lesson with a proof attached.
4. **Where the compiler *can* help the factory**: finite, linear, one-hot-shaped control — e.g. a
   parity/affine (grade-3) operator over a *fixed small* arity, a finite automaton / monotone narrowing
   step over a *small* candidate set, a bounded-step counter loop. Compile *those* exactly; leave
   interval-CP / Gaussian-elim / unification / AC-narrowing as classical certified contractors.

---

## 2. VerMCTS — verifier + LLM + tree search

### Mechanism (the exact loop)

- **MDP**: state `s` = prompt + partial program; action `a` = one *unit* (a Dafny line / Coq command,
  token-capped); transition = **string concat** `T(s,a)=s+a`; **reward = the verifier**, defined only on
  *complete* programs: `+1` accept, `−1` reject, **`0` for incomplete**.
- **"Evaluate and (maybe) expand"** (Alg. 1) — the heart. From node `s`, repeatedly append LLM
  completions until the verifier returns a *non-zero* score:
  ```
  a ← ""; while Verifier(s+a)=0 and depth<L:  a ← a + LLM(s+a)
  if Verifier(s+a) = −1 or depth=L:  return −1, None         # verified failure → add NOTHING
  else:                              return +1, child s+a    # verified-but-incomplete → add child
  ```
- ⭐ **The exploited property = monotone soundness of the verifier**: *"if a partial program fails the
  verifier, no subsequent completion can yield success"* → a failed prefix **prunes its whole subtree**
  (never added to the tree). The verifier is *"a computationally cheap (relative to the LLM) **upper
  bound on the value function**."*
- **Progressive widening** handles the unbounded action space: each node gets a `0`-valued **"widen"
  child** with prior `p_widen<1`; selecting it adds a *sibling* (grows width on demand). **PUCT**
  selection (`p=1` for real nodes biases toward depth). Backprop standard.
- **Result**: +30% absolute pass@5000 over repeated whole-program sampling; verifier-in-the-loop beats
  verifier-free MCTS-rollout beats whole-sampling beats Reflexion.

### Actionable takeaways for GLaDOS

- **This is a drop-in for the open "multi-organ routing/composition" problem** — and it needs *no GPU /
  no training* to start. Replace "Dafny line + Dafny verifier" with **"one organ invocation (a reduction
  step) + our exact verifier/certificate."** The LM proposes which organ + how to apply it; the certified
  reducer is the cheap intermediate gate; the search composes a *chain* of reductions.
- ⭐ **The monotone-pruning property is GLaDOS's lattice soundness reused as a search algorithm.** Our
  narrowing organ already gives sound monotone reductions: an unsound/over-narrowing step that drops the
  true solution is a "verified failure" → prune. So our organ *is* the VerMCTS verifier, and *"the organ
  is its own exact per-step grader"* (the LSRL/process-reward wish) becomes the tree's value upper bound.
- **Directly implements the codex_zoo_review routing recipe**: (1) verifier-selected **portfolio** =
  expand-and-gate over the bank per step; (2) **log oracle routes** = the search trace of which organ
  verified at each node; (3) train the router to imitate→optimize those traces. VerMCTS *is* steps (1)–(2).
- **pass@T metric**: report **pass@T (per token/compute budget)**, not pass@1 — aligns with the README's
  "report pass@k" discipline and makes the search's compute cost honest.

---

## 3. Multi-stage Relational Programming (staged-miniKanren)

### Mechanism

- **Relational interpreters** run *backwards*: an interpreter written as a miniKanren relation, with
  **holes** (logic variables) for unknown sub-expressions/values, turns evaluation into **synthesis from
  a sketch + examples** (Fig. 1: synthesize `(car xs)` to complete `append`; or *invert* `append`).
  Powerful but crippled by **interpretive overhead**.
- **Staging** = annotate which fragments compute **`later`** (run time) vs. now. *"Compile the known
  parts without interpretive overhead and defer interpretation to run time only for the unknown parts"* →
  turns the interpreter into a **compiler/specializer** (Futamura-style), keeping relational invertibility.
  Erasure property: drop annotations → ordinary relation.
- Novel constructs to reconcile staging with logic programming's three frictions
  (**non-determinism, pervasive partial data, lazily-checked constraints**): **`later`** (defer a goal to
  run time), and **`gather` / `fallback`** to control whether staging-time non-determinism is *specialized
  per branch* (`gather` → emit a run-time branch) or *re-deferred* (`fallback`). Big synthesis speedups.

### Actionable takeaways for GLaDOS

- **For α (text→typed factor-graph program)**: model α as a **staged relational compiler**. The LM fills
  the *holes* (which variables/domains/factors), the relational machinery handles the rest *invertibly*
  and *checkably* — a typed, partially-known program is exactly the "factor-graph with holes" α must emit.
  Staging is how you make that compile *fast enough* to sit inside a forward pass instead of paying
  interpretive overhead each step.
- **For the organs themselves**: `narrow`/`chain` are naturally **relational constraint solving** — a
  narrowing organ *is* a relation run in the "prune candidates" mode; a chaining organ *is* the same
  relation run in the "grow facts" (forward) mode. miniKanren's all-modes nature is the formal version of
  the README's "narrow and chain are duals." `gather`/`fallback` give a principled knob for **when to
  branch (search) vs. when to defer** — i.e. when the organ should fan out vs. abstain/punt to the host.
- Caveat: this is a *symbolic* substrate (exact, but not differentiable). Use it for α-as-compiler and as
  the **oracle / ground-truth organ** (cf. `clair/csp.py` exact `dedₚ`), not as the differentiable organ.

---

## 4. Meta-reasoning: "Reasoning about Reasoning about Reasoning" + Collapsing Towers

### Mechanisms

- **RRR (Zhang & Amin, POPL 2022)**: gives operational + measure-theoretic semantics to **nested inference
  queries** — agents modeled as probabilistic programs that `query` *other* agents who `query` back
  (Schelling coordination; `Alice` samples `query Bob`, `Bob` samples `query Alice`, to fixed depth). The
  technical crux: with **nested queries + general recursion you cannot stratify** the sampling and
  measure semantics — *"the two semantics must be defined mutually recursively."* It establishes
  **contextual-equivalence reasoning principles** (when two metareasoning agents are interchangeable),
  Coq-mechanized. This is the formal handle on *"a reasoner reasoning about which sub-reasoner to invoke."*
- **Collapsing Towers (Amin & Rompf, POPL 2018)**: a **multi-level λ-calculus with staging + stage
  polymorphism** — *"based on runtime parameters, an evaluator either executes source code (interpreter)
  or generates code (compiler)."* **Stage polymorphism** is the key that lets a *tower* of interpreters
  (Lₙ interpreting Lₙ₋₁ … interpreting L₀) **collapse into a single-pass compiler with all interpretive
  overhead removed** — *"exponentially improve runtime performance."* Purple adds **reflect/reify** so the
  semantics itself can change dynamically (a conceptually infinite tower; user programs recompiled under
  user-modified semantics).

### Actionable takeaways for GLaDOS

- **Routing = a 2-level reflective tower.** The LM at the meta-level *reflects* on the problem and
  *reifies* a choice of organ (object-level reasoner); the organ runs; control *reflects* back up to read
  the result. Collapsing-towers says: the meta-level dispatch overhead can be **staged away** so the
  composed (LM-routes → organ-runs → LM-reads) pipeline runs as *one* fused pass rather than an
  interpretive loop — the principled version of "weave the organ into the forward pass, don't bolt on a
  side-loop."
- **RRR gives the soundness vocabulary for routing**: when is invoking organ-A then organ-B *contextually
  equivalent* to some other route? That is exactly the equivalence you need to prove a learned router is
  safe (it may pick different routes but must reach checkably-equal answers). Borrow "contextual
  equivalence under nested queries" as the correctness notion for **router substitution / reduced-product
  composition order**.
- Honest scope: both are *theory* (semantics, equivalences), not algorithms you run tomorrow. They are the
  **conceptual frame + correctness notions** for routing, complementary to VerMCTS (the algorithm) and
  soft-branch (the differentiable primitive).

---

## SINGLE HIGHEST-LEVERAGE TECHNIQUE TO ADOPT

**Build the multi-organ composer as a VerMCTS-style verifier-gated search where each "action" is one
certified organ-reduction and our exact certificate is the `0/±1` gate — and use Cajal's soft-branching
conditional as its differentiable in-network relaxation once you want the router learned.**

Why this one: it attacks GLaDOS's *stated* bottleneck (routing/composition, which the codex review says
won't emerge from plain CPT) with a method that (a) needs **no GPU and no training to prototype**, (b)
**reuses the lattice soundness you already have** — a monotone organ that over-narrows is a "verified
failure" that prunes its subtree, and *the organ is its own exact per-step grader* (the process-reward you
wanted), and (c) **produces the oracle route traces** the codex recipe needs to later train/distill the
learned router (whose differentiable form is the soft-branch dispatcher). It turns "the LM composes
checked organs" from an aspiration into the concrete loop:

```
node = (problem, partial reduction chain)
  → LM proposes (organ, how-to-apply)
  → certified reducer runs; certificate gates: sound-reduction → child; unsound/over-narrow → prune
  → search to a checkable answer or honest abstain;  report pass@T
```

---
## CORRECTION (parent, after a closer read of "Compiling to recurrent neurons" 2511.14953)
The first verdict ("compile-to-neurons not viable for us") was too flat. The RECURRENT paper (Cajal(⊸,2,N))
compiles ITERATION as first-class — and our organs ARE iterative fixpoints, so the *structure* matches. The real
limit: linear recurrent neurons = ℝ-LINEAR dynamical systems (iterated body must be a fixed linear map over ℝ).
KEY NUANCE the first read missed: some of our certified ops are linear, just over a DIFFERENT algebra —
  • GF(2) Gaussian elimination = linear over GF(2)
  • min-plus graph relaxation  = linear over the tropical (min,+) semiring
So a SEMIRING-generalized compile-to-recurrent-neurons would compile THOSE exactly (differentiable-by-construction,
no learned-soundness risk). The paper's own future work points there ("richer recurrence over lists/trees/algebraic").

### Verdict on implementing it: NOT NOW (not super-super-worth-it).
- Only the SEMIRING-LINEAR organs (GF(2), min-plus) could compile — a MINORITY (2 of ~8); the nonlinear ones
  (AC, interval, HM, alldiff-GAC) don't compile regardless of field.
- Even those 2 already work as exact classical certified ops — nothing broken.
- The only payoff is differentiability THROUGH the deduction (end-to-end gradients), but exploitation + split-brain
  proved the LM learns to USE the organ via the OUTPUT-READOUT with no backprop-through-the-op. So it solves a
  problem we don't have.
- TRIGGER to revisit: a downstream task that provably can't learn to use an organ via readout alone (so far: never).
