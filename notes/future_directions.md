# Future directions — toward a general, emergent, checked reasoning substrate

The long arc, beyond the current coloring/CSP organ. The spine ("learned proposer + checked at the
boundary; soundness is a *permission* not a prescription") already licenses all of this.

## 1. The three properties decompose cleanly
- **general** ← the verifier-ladder curriculum (train on every reasoning shape that has an exact checker)
- **sound** ← the output-check (free, exact, un-hackable) — NOT the lattice-by-construction
- **emergent** ← train on the VERIFIABLE OBJECTIVE, not on imitating a hand-specified lattice operator

The lattice (monotone candidate-set narrowing) is the **training wheels**: a strong inductive bias that
makes the organ learnable + sound-by-construction EARLY, so it bootstraps fast. The emergent endpoint
RELAXES the dominate-dedₚ supervision toward "did the answer verify," letting the organ discover its own
inference operations, with the output-check catching errors. RLVR + set-valued readout already lean this
way; fully-emergent is the far end of that slider.

## 2. More tasks (extend the verifier ladder)
Anything with a verifier + inference structure; the diversity IS the generality test:
- planning/scheduling (blocks-world/STRIPS — verify plan reaches goal)
- graph algorithms (CLRS — shortest path/connectivity, verifiable)
- program synthesis/code (verify by running tests)
- math (equation solving/proof — CAS or Lean as checker)
- constraint OPTIMIZATION (verify against an objective, not just satisfaction)
- abduction (find an explanation consistent with observations — the narrow-hypotheses dual)

## 3. Non-lattice reasoning organs (the grand version)
The "deductor" is ONE reasoning organ. The general theory: **a library of differentiable, checked
reasoning modules the LM ROUTES among**, each suited to a reasoning shape:
- **lattice-narrowing** (have) — monotone, sound, abstain-capable; best for constraint satisfaction.
- **forward-chaining** — GROWS facts instead of narrowing candidates (the dual; our FOL chainer made neural).
  Best for derivation/entailment.
- **energy / equilibrium** (DEQ) — soft constraint satisfaction as a fixed point; relaxation + optimization.
- **program-induction** — differentiable interpreter that induces+runs a small program; Turing-general, hardest.
- **memory / scratchpad** — general differentiable working memory; not narrowing at all.
The lattice is the most-structured/most-sound; others trade structure for generality. Long arc: the LM as
a router over a bank of checked organs — narrower for CSPs, chainer for entailment, optimizer for planning —
all woven into the forward pass, all sound at the output.

## 3b. Geometric algebra, finally placed (not as a generic mixer)
GA was the orphaned opening thread — null as a channel mixer (ties SwiGLU/MLP, language & vision). Its REAL
home is the GENERAL FACTOR DEDUCTOR, via the mapping **geometric-product GRADES ↔ lattice abstraction LEVELS**:
- grade-0 (inner u·v, coherence) ↔ per-cell / unary
- grade-2 (wedge u∧v, "these two can't coexist") ↔ pairwise — *literally* the binary-constraint operator (≠,=)
- grade-3 (trivector u∧v∧w = the scalar triple product, already prototyped as geom_lm `geom_g3`) ↔ triple —
  the 3-way antisymmetric / AFFINE structure that breaks the per-cell wall (levels.py: arithmetic=affine=needs L2)
- higher grades ↔ higher-arity factors.
So the factor deductor should compute its arity-k constraint interactions as grade-k geometric products: the
wedge IS the exclusion operator, the trivector IS the parity/affine operator. The grades MEAN the constraint
arities → the antisymmetric structure is load-bearing, not decorative. This is why it failed as a generic
mixer (arbitrary dims vs SwiGLU) but is principled here (structure-matched). Fold into the general-deductor track.

## 4. Consolidation (how the narrow skill becomes general behavior)
After the architecture-specific curriculum installs+trains the organ, RE-RUN OLMo's OWN open recipe
(Dolma continued-pretrain + instruction-tune + RL-Zero) WITH the organ in place, so organ-use folds into
general behavior. Only possible because OLMo is fully open. This is the answer to "will it use the organ
during normal reasoning" — don't hope it transfers; generalize it by re-running the general recipe with
the organ present.

## 5. Future base architectures (the organ is residual-stream-agnostic → grafts onto any transformer)
The dense-γ readback lives at the residual stream, so mechanically the organ ports to any transformer. The
*interesting* question is which non-standard bases are worth the complexity:

- **DiffusionGemma (diffusion LM on Gemma-4 26B-A4B MoE) — the most sympathetic + novel.** It generates by
  ITERATIVE DENOISING of the whole sequence (non-autoregressive). That's the *same shape* as the organ's
  iterative narrowing → couple them **in lockstep**: the lattice narrows AS the sequence denoises, both
  fixed-point processes informing each other. The organ becomes a *parallel refinement*, not an awkward
  autoregressive side-loop. Nobody has woven a checked deductor into a diffusion LM — the "iterative-marriage"
  paper. Honestly the graft I'd most want to try post-OLMo-validation.
- **MoE bases (Laguna M.1 = 225B/23B-active coding MoE; DeepSeek V4 = MoE+MLA+MTP; Gemma-4 A4B) → organ as a
  routed EXPERT.** An MoE already routes tokens to specialists; a **checked-deductor-expert** the router invokes
  on constraint-shaped tokens is a natural addition — and it's the clean framing for our organ BANK
  (narrow/chain/energy = routed deductor-experts). Conceptually beautiful; 225B/V4 scale is compute-prohibitive
  for us (and MLA/MTP add grafting complexity). **Gemma-4-2B/4B** is tractable as a non-OLMo transformer control.
- **Honesty caveat:** Gemma/DeepSeek/Laguna are open-WEIGHTS, not open-RECIPE (unlike OLMo) → RLVR/consolidation
  gains are harder to *attribute* (cf. the spurious-rewards finding). **OLMo-Base-7B stays the research base for
  the paper's claims**; the exotic bases are the scaling/future story, DiffusionGemma the genuinely-novel bet.
