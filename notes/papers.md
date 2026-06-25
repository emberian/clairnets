# The five primitives — regrounded from the papers (2026-06-25)

Re-fetched arxiv full-text (HTML) after a compaction wiped the details. These are the
*actual* mechanisms, not the congress paraphrase. Sources are the arxiv `/html/<id>v1` pages.

---

## 1. Lattice Deduction Transformer — arXiv 2605.08605  ★ the spine
**The standout. 800K params = 100% Sudoku-Extreme & Snowflake; 1.8M = 99.9% Maze-Hard.
Frontier LLMs (Opus 4.6 / DeepSeek-V4-Pro / GPT-5.4) = 0% on all three.**

- **Abstract domain = grid powerset lattice.** State `a : {1..k} → P(V)` — each position holds the
  set of still-viable candidates. Ordered pointwise by inclusion: `a ⊑ b �iff ∀i a(i) ⊆ b(i)`.
  Represented as multi-hot: |V| binary sigmoids per cell (729 sigmoids for 9×9 sudoku).
- **Galois connection** (this is the soundness backbone):
  - γ(a) = {s | ∀i. s(i) ∈ a(i)}  (concretization → set of full grids consistent with a)
  - α(S') = per-position union of candidates over S'  (abstraction)
  - **deduction operator** `ded_p(a) = α(γ(a) ∩ ‖p‖)` — narrow to the most informative state
    still consistent with the *remaining solutions*. Guarantees `a_{t+1} ⊑ a_t` (monotone descent).
- **Recurrent core:** d=128, 4 attention layers, 4 heads, **L=16 internal iterations per fwd pass**,
  FFN-mult 4.0, learned 2D pos-emb + input residual re-injected every iteration. ~800K params.
  Maze variant d=192, 2D RoPE, ~1.8M.
- **Loss (Eq.1), summed over L deep-supervision steps:** `ℒ = mean_ℓ[ ℒ_BCE^ℓ + λ_cls ℒ_CLS^ℓ + λ_ce ℒ_CE^ℓ ]`
  - **Asymmetric BCE** σ(b) vs target ŷ with **w⁺=4.0, w⁻=0.5** — false eliminations punished ~8× harder
    than false retentions (this is what buys soundness — never kill a live candidate).
  - **CLS conflict BCE** (λ_cls=0.1): head σ(c) predicts "state is unsatisfiable (⊥)".
  - **Singleton softmax-CE** (λ_ce=0.2): where the target cell is a singleton — speeds learning.
  - **alpha target** `ŷ = x ⊓ α({y∈Y : y consistent with x})` — on-policy, state-dependent label.
- **Solve loop (Alg.1, run at BOTH train & inference → states stay in-distribution):**
  fwd pass → **threshold-eliminate** candidates with σ(b) < θ_elim≈0.1 (enforces strict descent) →
  check solved (all singletons) / conflict (empty cell or σ(c)>θ_CLS) → else **stochastic branch**:
  pick a multi-candidate cell, sample digit from softmax(b/τ), τ_decide=1.5, pin it; backtrack on conflict.
  Inference: M=8 slots × K=64 chains, budget R=1000 rounds/puzzle.
- **"Abstain"** = if search times out or CLS fires, return NOTHING rather than guess. Empirically sound
  on Sudoku-Extreme (100/100, never wrong). Maze K=512 is 99.9% but NOT empirically sound (bad paths
  are valid-but-suboptimal length).
- **No ARC numbers in the paper** (the "~36% ARC plateau" was codex's web-search gloss, not this paper —
  flag as unverified). 0.028s/example on Sudoku-Extreme.

## 2. CliffordNet / geometric-product mixing — arXiv 2601.06793  ★ the cheap expressive cell
Vision-only (CIFAR). We already reproduced FFN-redundancy on modadd (~42% fewer params).

- **Geometric product mixer:** `F(H,C) = P( H·C ⊕ H∧C )`. Only **two grades**: grade-0 scalar (inner)
  + grade-2 bivector (wedge). ⊕ is channel concat into ℝ^2D, then learned projection P back to ℝ^D.
- **O(N) via "shifted geometric product"** — don't compute all D² channel pairs; take only cyclic
  channel shifts s ∈ 𝒮 (e.g. {1,2,4,8,15}), i.e. specific diagonals of the full product:
  - Dot:  `D_s = SiLU(H_{i,c} · C_{i,(c+s)%D})`  (coherence / gating / diffusion)
  - Wedge:`W_s = H_{i,c}·C_{i,(c+s)%D} − C_{i,c}·H_{i,(c+s)%D}`  (anti-symmetric; u∧u=0, so it
    THROWS AWAY self-energy and keeps only cross-channel structure — "geometric torque/vorticity").
  - complexity O(N·D·|𝒮|), |𝒮| small fixed.
- **Block (Gated Geometric Residual):** LN → dual stream (linear branch + depthwise-conv context) →
  rolling interaction over 𝒮 → concat+project → gated residual fuse. **No FFN.**
- **Numbers (CIFAR-100):** Nano 1.4M no-FFN 76.41% (> ShuffleNetV2 1.4M 74.6%); Fast 2.6M no-FFN 77.63%;
  Base 3.0M +FFN 78.05% (> ResNet-18 11.2M 76.75%).
- **FFN-redundancy ablation (Table 3):** adding an explicit FFN gives only **+0.42%** → geometric layer
  is the primary driver. **Dot-only 76.91% / Wedge-only 76.35% / both 77.63%** (Table 4) — wedge ALONE,
  with zero energy info, nearly matches. The wedge is doing real discriminative work.

### ★ CORRECTION (2026-06-25, after a hasty mis-test) — the CLAIM is PARETO param-efficiency, NOT iso-param
The headline is "Nano 1.4M ≈ ResNet-18 11.2M (8× fewer params), Lite 2.6M = tiny-model SOTA 79.05%" — i.e.
**geom MATCHES MUCH BIGGER baselines with far fewer params.** Comparing geom to an iso-param MLP-mixer (what
we first did → geom 61% < mlp 64% at 1.5M/30ep) is the WRONG test AND was an unfaithful, undertrained repro.
Mechanisms our first vision repro MISSED (the ones that make it work small):
- **Self-energy suppression `C = C_loc(H) − λ·H`, λ=1 = discrete Laplacian ΔH** (geometric high-pass),
  explicitly "optimal for capacity-constrained models." WE OMITTED THIS — likely the biggest miss.
- **Local context = TWO stacked depthwise 3×3 convs** `Conv3x3(Conv3x3(H))`, not one.
- **Global superposition** `C_glo=GlobalAvgPool`, β-switch (high-perf variants).
- **Gated Geometric Residual** `H_l = H_{l-1} + γ⊙(SiLU(H_{l-1}) + Gate(H_{l-1},H_geo)⊙H_geo)`, not a plain residual.
- isotropic columnar (constant h×w×D), patch-embed conv, shifts S={1,2,4,8,15}.
**Lesson:** test in the regime the method targets (low-param Pareto), with a FAITHFUL build + proper training
(~150 epochs), before concluding. Faithful repro = `clair/cliffordnet.py`; decisive Q = does Nano-1.4M-λ1 reach
~76-78% and does λ=1 beat λ=0. NOTE our geom-LM language sweep ALSO hinted this — geom won at 0.5M, lost at 8M
(low-param regime is geom's home), so that "negative" was likewise regime-dependent, not absolute.

## 3. HRM (Hierarchical Reasoning Model) — arXiv 2506.21734  ⚠ contested
- Two recurrent modules: H (slow/abstract planning) + L (fast/detailed), 27M params, **1000 samples**,
  no pretraining/CoT, "sequential reasoning in a single forward pass."
- **What survives independent scrutiny (ARC-Prize repro):** the driver is the **outer refinement loop +
  deep supervision + data augmentation**, NOT the H/L hierarchy (plain transformer within ~5pp; repro got
  32% not 41%). Puzzle-id embeddings make it transductive (cheating). **Steal: weight-tied looped depth,
  deep supervision, outer refinement. Drop: H/L mysticism, puzzle-id embeddings.**

## 4. Transformers Learn Shortcuts to Automata — arXiv 2210.10749  (lens, not ingredient)
- A low-depth transformer can represent **any** finite-state automaton (any bounded-memory algorithm) by
  hierarchically reparameterizing its recurrent dynamics. **O(log T)-depth shortcuts always exist**;
  O(1)-depth simulators "surprisingly common." Uses **Krohn-Rhodes / semigroup** structure to characterize
  when constant depth suffices (solvable vs unsolvable semigroups). ⇒ recurrence ↔ shallow-structured
  transformer are interchangeable. Use it to **distill / binary-lift** a learned deduction step, not as branding.

## 5. Generalization at the Edge of Stability — arXiv 2604.19740  (diagnostic, not architecture)
- **Sharpness Dimension** (Def 4.2): `dim_s 𝒜 = j* + (Σ_{i≤j*} λ_i)/|λ_{j*+1}|`, where
  `j* = max{i : Σ_{k≤i} λ_k ≥ 0}` and `λ_k` = RDS sharpness of order k =
  `E[ sup_w ln σ_k(I − η∇²R̂(w)) ]` (log singular values of the one-step training Jacobian).
  = a **Kaplan–Yorke / Lyapunov dimension** of the optimizer's fractal attractor: the maximal dim in which
  volumes don't contract. Generalization bound (Thm 4.5) scales with **dim_s, not parameter count**.
- **Practical:** NO prescribed LR schedule. It's a **post-hoc diagnostic** — estimate via stochastic Lanczos
  quadrature on a converged net to check if it sits at EoS. Hessian quadratic-memory → intractable at scale.
  ⇒ for us: pair with our existing inference-time probes (participation ratio, finite-time Lyapunov). Only
  counts as a win if SD predicts OOD solve-rate better than val-loss.

---

### The through-line (what's deep vs cosmetic — both congresses agree)
- **DEEP:** lattice+recurrence (LDT — sound abstract interpretation, can abstain); lattice+semigroup
  (a finite lattice update IS a finite-state transition → automata-shortcut distillation is principled);
  Clifford product as the recurrent cell f_θ (a dense complete bilinear mixer; wedge = "these two
  candidates can't coexist" — naturally suited to exclusion constraints).
- **COSMETIC / weak:** Clifford∧lattice is NOT automatically sound (GP is bilinear, not monotone — the
  lattice projection must remain the sole authority or it hallucinates deductions); EoS is not an
  architecture; HRM's H/L split is not the bet.
- **The spine:** `a_{t+1} = a_t ⊓ Π_A( f_θ(a_t, p, m) )` — recurrent geom cell proposes, lattice meet narrows.
