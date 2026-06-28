# GLaDOS forward-research roadmap — Ember's expansive vision, honestly triaged

*Capture note. Ember is generating threads faster than we record them; this is the catch-net. Each thread
is logged verbatim-faithful with: the idea · why it might help · an honest assessment
(likely-helps / fun-but-speculative / dream) · the **money-bar verdict** (box / local-spike / discuss-only) ·
how it connects to what we already have. The ranked **next-5 studies** are at the bottom — that is the
actionable queue; everything above it is the reasoning.*

> The money-bar (from `exotic_losses_and_glitches.md` §discipline): **a box is worth a GPU only when the
> answer is (a) decision-relevant — it changes what we build — AND (b) genuinely uncertain.** Pre-mortem
> each: "if it comes back the other way, does what we build change?" If no → CPU check / back-of-envelope,
> not an L40S. The gold standard was SHUFFLED + α-can-read-structure (decision-relevant AND uncertain); the
> cautionary tales were affine-is-capacity / data-alone-caps / all-views-crowd (re-confirmed known theory,
> should've been cheaper). Three verdict tiers:
> - **box** — fund a study (decision-relevant AND uncertain).
> - **local-spike** — a CPU / small-scale probe or design spike answers it; no GPU sweep.
> - **discuss-only** — a dream / framing / lit-anchor; record it, don't fund it yet.

The single live front door is `GLADOS.md`; the cathedral is `north_star_orchestrator.md` (LM-as-organ-
orchestrator + the search arm). Read those two first — most threads below are bricks in that wall, or
deliberate departures from it.

> **The roadmap's spine (Thread 8, stated up front because it sorts everything else):** there are two
> GLaDOSes and they are **Phase 1 → Phase 2 of one program**, not a fork. **Phase 1 = the PORTABLE GLaDOS**
> (frozen base + bolt-on, the 7-base graft matrix) — its job is to prove the *mechanism generalizes*;
> model-independence is a feature of the demonstration. This is the **science / paper**. **Phase 2 = the
> INTEGRATED GLaDOS** (organ co-designed with the host, woven at multiple layers, host trained to wield it)
> — its job is to *superpower one model*; model-independence is a cost happily paid for performance. This is
> the **product / dream**. *Prove-portable-then-go-deep.* Each thread below is tagged with the phase it
> lives in; the dream threads (Lean/code/HOU-in-a-co-designed-host/broad-uplift) are all Phase 2, gated
> behind Phase-1 completion. The 36% woven-vs-frozen LoRA seam already shows Phase 2 has headroom.

---

## Thread 1 — Non-gradient training methods  ·  *[Phase 1 — α/organ training]*

**The idea (Ember).** Train the organ (or α, or the orchestration policy) with something other than SGD:
evolutionary strategies / CMA-ES, predictive coding, equilibrium propagation, forward-forward,
Hebbian / local learning rules. **KEY INSIGHT: we have an EXACT verifier** → verifier-guided
evolution/search has a *clean, sound fitness function*. This is exactly the regime where ES stops being a
toy: ES is sample-inefficient and noisy when the fitness is a noisy scalar RL return, but our fitness is
"the certified organ + output-check says this candidate solves the held-out instances" — un-hackable, no
false-accepts, low-variance. And **α-as-search is already a non-gradient inference method** (propose
structures, verifier-filter — `alpha_struct_design.md` §5, `north_star_orchestrator.md` "α-as-search"), so
we are *already* doing gradient-free optimization at inference; this thread asks whether it pays at
*training* time too. **Predictive coding** specifically fits the organ's relaxation dynamics — the organ is
an iterate-to-fixpoint contraction (R=12 narrowing steps), and PC is precisely a relaxation-to-equilibrium
local-learning scheme; same structural shape as the energy/equilibrium (DEQ) organ in
`future_directions.md` §3.

**Why it might help.**
- The exact verifier converts ES from "noisy black-box RL" into "sound fitness-guided search" — the one
  setting where neuroevolution is competitive (cf. Salimans et al. 2017, *Evolution Strategies as a
  Scalable Alternative to RL*, arXiv:1703.03864; Such et al., *Deep Neuroevolution*, arXiv:1712.06567;
  Hansen's CMA-ES). It also sidesteps the discrete-bottleneck problem in α (we currently can't backprop
  through `build_csp_from_struct`; `alpha_struct_design.md` §2.2) — ES/PC don't need the gradient through
  the discrete compile at all.
- Predictive coding (Whittington & Bogacz 2017; Millidge et al.; Salvatori et al., *Reverse Differentiation
  via Predictive Coding*, ICML'22) approximates backprop with local updates over a relaxation — a natural
  match for an organ that is *itself* a relaxation, and it gives a biologically-plausible / hardware-local
  story if that ever matters.
- Equilibrium propagation (Scellier & Bengio 2017) and forward-forward (Hinton 2022) round out the
  "no global backward pass" family; Hebbian/local rules are the cheapest-per-step.

**Honest assessment: fun-but-speculative ("fun to study, maybe not better").** The literature is clear
that on standard supervised objectives these methods *match but rarely beat* SGD, and usually at higher
compute. Our differentiator is the exact verifier — which genuinely changes the ES calculus — but SGD +
the existing J0/structure-sup losses already train the organ well (`GLADOS.md` stage 1; the 1.5M/R12 recipe
works). The decision-relevant sub-question is narrow: **does verifier-guided ES/search beat gradient for
the ONE part we can't differentiate — α's discrete structure compile?** That piece is real and connects to
α-as-search; the rest (PC/eqprop/FF/Hebbian as a wholesale SGD replacement) is curiosity.

**Money-bar verdict: mostly discuss-only; ONE local-spike.**
- *discuss-only:* PC / eqprop / forward-forward / Hebbian as a general training-method swap. Re-confirming
  "ES ≈ SGD but costlier" is an affine-is-capacity-class re-confirmation — pre-mortem says we'd build
  nothing differently. Record the lit anchors; don't fund.
- *local-spike:* **verifier-guided ES/CMA-ES vs gradient for α's discrete structure-compile**, on the
  frozen-OLMo structure-faithfulness probe (`alpha_struct_design.md` §5.5, #2 — already CPU/cheap). This
  is decision-relevant (it's the make-or-break α step we can't backprop through) AND uncertain (nobody's
  measured ES-with-a-sound-verifier on factor-graph compilation). Small, runs on the probe we're already
  building. Fold it INTO α-as-search rather than treating it as a separate "non-gradient" research line.

**Connections.** `alpha_struct_design.md` §2.3/§5.3 (discrete bottleneck, α-as-search proposer+filter);
`north_star_orchestrator.md` "The search arm" (a sound verifier *wants* search, RLVR is the weakest tool);
`future_directions.md` §3 (energy/equilibrium DEQ organ ↔ predictive coding's relaxation).

---

## Thread 2 — Higher-order unification (HOU) organ  ⟵ Ember's especially-curious one  ·  *[Phase 1 spike → Phase 2 payoff]*

**The idea (Ember).** Add a **higher-order unification** faculty to the bank. Huet-style HOU is the
substrate of program synthesis, higher-order logic, dependent types, and proof search — i.e. it *is* the
code-task / formal-reasoning faculty. It fits the bank pattern exactly: a **certified Huet unifier (sound
by construction)** + **neural guidance for the branch explosion**. HOU is only semi-decidable, and the
blow-up is precisely at the **flex-flex** pairs and the **imitation-vs-projection** choice in Huet's
pre-unification algorithm — which is *exactly* where "neural proposes, certified checks" earns its keep
(the same shape as `core_narrow_organ` gated by `exact_dedP`, generalized from CSP to λ-terms).

**Sketch of the design (bank-spine fit).**
- **The certified op.** A Huet pre-unification step (Huet 1975, *A Unification Algorithm for Typed
  λ-Calculus*): SIMPL the rigid-rigid pairs (decompose, sound and terminating), and on a flex-rigid pair
  enumerate the **imitation** + **projection** substitutions. The op is sound-by-construction: any
  substitution it returns is a real unifier of the decomposed sub-pairs; it never accepts a non-unifier.
  Termination/completeness is the caveat (semi-decidable) — so its `certificate()` is
  `sound-by-construction` with an explicit completeness/decidability caveat, identical in spirit to how the
  Ising organ is `approximate` (`abstract_machines.md` §2.2 protocol-fit discussion). The **decidable
  sub-fragment** — Miller's higher-order *pattern* unification (Miller 1991; flex args are distinct bound
  variables) — is a clean certified floor that always terminates with a most-general unifier; HOU outside
  the pattern fragment is where neural guidance is needed.
- **The neural-guidance role.** At each flex-rigid node, score the candidate imitation/projection branches
  (and prune flex-flex deferrals) — a learned proposal over Huet's branching, **verifier-gated** by the
  certified SIMPL/occurs-check. The search blows up combinatorially; the neural policy orders/prunes it; a
  wrong neural ranking can only slow the search, never produce an unsound unifier. This is the cleanest
  possible instance of the spine: the soundness is a *permission* (`north_star_orchestrator.md`), the
  neural part is pure search-ordering. Closest live analog in lit: neural premise-selection / tactic
  prediction for proof search (HOList/DeepHOL, GamePad, TacticToe, Holophrasm; Lean/Coq tactic predictors)
  — but those guide a *whole tactic engine*; an HOU organ guides the *unification kernel underneath* them,
  which is finer-grained and more reusable.
- **State type + readout.** Its own `state_type="hou"` (a disagreement-set / substitution-frontier); it
  does NOT reduced-product with `CSPState` (like ∂4-Forth it's a standalone faculty with a survival/readout
  bridge, `readout_bridge.py` pattern). `survival()` = the readout over candidate substitutions /
  resolved-vs-open metavariables.

**What it unlocks (the prize).** This is the faculty that reaches the hard targets the CSP/Ising/Datalog
bank structurally cannot: **code/program synthesis** (synthesis = inhabiting a type = HOU + search),
**proofs** (higher-order logic, tactic unification, Lean as the output-checker), and **higher-rank /
dependent type inference** (elaboration is HOU). It is the missing piece between "logic-grid solver" and
"code faculty" in the Thread-3 path.

**Honest assessment: dream-with-a-feasible-spike.** The full payoff (code/proofs via a woven HOU organ) is
a *dream* — it's a multi-quarter faculty build and the hardest thing in the zoo after lambda/RISC
(`abstract_machines.md` §2.4 parks neurallambda for exactly these reasons). BUT the *first brick* —
"can we ship a certified Huet pre-unifier with a neural branch-ranker, scoped to the decidable pattern
fragment + a shallow flex-rigid tail, and show neural guidance cuts the search?" — is a concrete,
bounded **feasibility spike**. The certified kernel is textbook (Huet '75 is ~one page of rules); the
uncertain part is whether neural guidance meaningfully prunes Huet's branching on our scale and whether the
non-pattern tail stays tractable.

**Money-bar verdict: box (a faculty-build feasibility spike).** Decision-relevant — it's the gate to the
entire code/proof arm of Thread 3, and whether to invest there hinges on this spike. Genuinely uncertain —
we don't know if neural-guided Huet pays off at our scale or drowns in flex-flex. Scope the box tightly:
*certified pattern-unifier + one-step flex-rigid branch-ranker, measured on synthetic HOU problems
(search-nodes-expanded with vs without guidance, soundness=100% by construction), NO LM weave yet.* That is
the cheapest honest test of "is HOU a viable bank faculty?" before any code-eval ambition.

**Connections.** `abstract_machines.md` §2.3 (Datalog faculty — HOU is the higher-order generalization of
the certified `unification_chain`; same "certified core + gated neural proposer" verdict) and §2.4 (the
lambda/RISC park — HOU is the *principled* slice of that Turing-general endpoint, with a real certified
kernel, so it leapfrogs the parked stuff); `north_star_orchestrator.md` (sound verifier wants search — HOU
is search par excellence); `future_directions.md` §2 (program synthesis / math-proof as verifier-ladder
tasks; Lean as checker).

---

## Thread 3 — The Gemma / NL / code superpowering DREAM (the real prize)  ·  *[Phase 1 step 1 → Phase 2 steps 2–3]*

**The idea (Ember).** Stop proving the organ on tiny bases; **tack the organ onto an already-optimized,
strong model (Gemma) and ACTUALLY move the hard evals** — or at least move narrow reasoning domains that
capability-scaling alone left on the table. The cleanest possible demonstration: **frozen-strong-base +
organ > frozen-strong-base** on a domain — i.e. *"the organ granted a capability that parameter-scale did
not."* That is the headline the whole project is for.

**The honest path (narrow → code → broad), staged.**
1. **Narrow reasoning wins FIRST.** Logic-grid / SAT / constraint-math — the organ's home turf, where the
   certified floor is exact and the gap to a strong base is largest. This is `eval_suite` **Tier-1**
   (`GLADOS.md` "HOW it's evaluated"). Frozen-Gemma + organ should beat frozen-Gemma here cleanly because
   these are *exactly* the shapes capability-scale handles worst and the organ handles by construction.
2. **Code SECOND** — needs Thread-2's HOU + the ∂4-procedural faculty (`abstract_machines.md` §2.2). Code
   is `eval_suite` **Tier-2** transfer with a real verifier (run the tests). Don't attempt before the HOU
   spike and a procedural faculty land.
3. **Broad uplift THIRD (stretch / dream).** Move the genuinely hard general evals. This is the
   next-token-helps-broadly bet from `exotic_losses_and_glitches.md` §2 — explicitly flagged there as the
   STRETCH, not the minimum bar. **Tier-3 no-harm** is the honesty gate the whole time (don't degrade
   general ability buying a narrow win).

**The bar, stated precisely.** `frozen-strong-base + organ > frozen-strong-base` on a held-out domain,
with the causal-control table passing (`GLADOS.md`: WOVEN > TEXT-LoRA AND corrupt/shuffle/permute →
collapse), so the win is attributable to the organ and not to LoRA or leakage. That conjunction is the
"capability scale didn't grant" demonstration — and it is *only* meaningful on a strong base (on a 160M
base "the organ helps" is unsurprising; on Gemma-12B/27B it's the result).

**Why it might help.** The 7/7 graft matrix (`GLADOS.md`, Pythia-160M → OLMo-3-32B incl. Gemma-4-12B and
the Nemotron-H Mamba hybrid) already proves the *mechanism* ports to strong bases with bitwise no-op@init.
The open question is whether the *uplift* survives onto a base strong enough that the easy wins are already
saturated — which is precisely Thread 4's MATRIX law.

**Honest assessment: dream, with a fundable narrow first rung.** Step 1 (narrow, Tier-1, on a strong base)
is real, near-term, and the cleanest scientific statement we can make. Steps 2–3 are dreams gated on
Threads 2 + the procedural faculty. The risk is `codex_organ_review.md`'s crux — if α can't compile real
NL structure, "frozen-Gemma + organ" is just a metadata-solver wearing a Gemma coat. So this thread is
**downstream of ALPHA_STRUCT passing** (`alpha_struct_design.md` §4 gate) and `eval_real_nl`
(`GLADOS.md` "Honest scope").

**Money-bar verdict: box for step 1, discuss-only for steps 2–3.**
- *box:* **frozen-strong-base (Gemma) + organ vs frozen-Gemma on Tier-1 narrow reasoning**, full
  causal-control table. Maximally decision-relevant (it's the project's headline claim) AND uncertain (we
  don't know if uplift survives onto a saturated-easy strong base). Gated on ALPHA_STRUCT passing first.
- *discuss-only:* code (gated on Thread-2 HOU + procedural faculty) and broad-uplift (the stretch bet).
  Record the staging; don't fund until step 1 + the prerequisite faculties land.

**Connections.** `GLADOS.md` (7/7 graft matrix, eval Tier-1/2/3, causal controls); `eval_suite`
Tier-2/3; `exotic_losses_and_glitches.md` §2 (next-token-helps = stretch, no-harm = minimum bar);
`alpha_struct_design.md` §4 (the gate this depends on); Thread 4 (the MATRIX law is the quantitative
version of this thread's "does it survive on a strong base?").

---

## Thread 4 — Scaling laws to hypothesize + study  ·  *[Phase 1 (laws a–c, e) · Phase 2 (law d)]*

**The idea (Ember).** Propose concrete, testable scaling laws for the woven organ, and triage which are
cheap to measure on EXISTING artifacts vs which need fresh runs. Candidates:

- **(a) organ-recall vs recurrence-DEPTH R** — does solvable-problem recall scale with the organ's
  iterate-to-fixpoint depth R, and does the needed R track *problem depth* (propagation depth / treewidth)?
  Hypothesis: **compute-to-solve ≈ problem-depth**, the contraction needs R ≥ propagation-depth. (Directly
  the curriculum-design propagation-depth axis: `curriculum_design.md` §1.3 — train median depth 3, the
  R=12 budget is provisioned for a depth the train data doesn't contain.) Lit anchor: Universal
  Transformers' ponder/ACT (Dehghani et al. 2018, arXiv:1807.03819; Graves ACT, arXiv:1603.08983) and the
  recurrent-depth latent-reasoning scaling of Geiping et al. 2025 (*Scaling up Test-Time Compute with
  Latent Reasoning*, arXiv:2502.05171 — "Huginn") show test-time-compute = recurrence-depth scaling; we'd
  show it for a *checked* recurrence with an exact recall metric.
- **(b) recall vs organ-PARAMS** — the capacity law for the organ itself (the 1.5M-param recipe point is
  one sample; sweep it). Cheap-ish.
- **(c) #faculties vs problem-COVERAGE** — coverage grows as we add bank faculties + reduction edges
  (`abstract_machines.md` §1.5 "one faculty + edges = many problems"). Partly measurable on existing
  reduction-graph artifacts (route-required held-outs).
- **(d) composition-DEPTH vs accuracy** — accuracy decay as orchestration depth (N sub-problems chained)
  grows; the `north_star_orchestrator.md` compositional-generalization curve. Needs the orchestrator loop.
- **(e) THE MATRIX law (the prize) — does woven UPLIFT grow (or shrink) with BASE size, 100M → 32B?** Run
  the same organ across the 7/7 graft matrix and fit uplift-vs-base-scale. This is the quantitative form of
  Thread 3's "does it survive on a strong base?" — and the single most decision-relevant law (it tells us
  whether GLaDOS is a small-model crutch or a frontier-model capability-grant).

**Why it might help.** Laws turn "the organ helps" into "the organ helps *here*, scales *this way*" — they
make the paper's claims load-bearing and tell us where to spend compute. (a)+(b)+(c) are partly free on
artifacts we already have; (e) is the expensive, decisive one.

**Honest assessment: mixed — some are local-spikes on existing data, (e) is a real box, (d) is gated.**
Note (per Ember): **the R-depth (a) + faculty-stacking (c) laws are being measured NOW in the live
organ-block study** (see Thread 6) — so don't double-fund them; harvest that study's curves. (b) is a cheap
CPU sweep on the standalone organ. (e) is the genuine multi-base GPU study; (d) needs the orchestrator
(Thread 5/6) to exist first.

**Money-bar verdict:**
- *local-spike:* (b) organ-params capacity sweep (standalone organ, CPU/small-GPU, cheap); (a)+(c) —
  **harvest from the live organ-block study**, don't run separately.
- *box:* (e) **the MATRIX law** — uplift-vs-base-scale across the graft matrix. Maximally
  decision-relevant (small-model-crutch vs frontier-grant is THE question), genuinely uncertain. Expensive
  but this is what compute is for.
- *discuss-only / gated:* (d) composition-depth law — fund it *after* the orchestrator loop exists; it's
  hollow without multi-step orchestration.

**Connections.** `curriculum_design.md` §1.3/§3.2 (propagation-depth & required-level knobs feed law (a));
`abstract_machines.md` §1.5 (coverage law (c)); `north_star_orchestrator.md` (composition-depth law (d));
`GLADOS.md` 7/7 graft matrix (the substrate for the MATRIX law (e)); `roadmap.md` (the dynamics/attractor-
dimension-predicts-generalization bet is an adjacent "law" already logged — cross-reference, don't
duplicate).

---

## Thread 5 — Multifaculty routing + gated routed RECURRENCE through the organ (nested reasoning)  ·  *[Phase 1 spike → Phase 2 box]*

**The idea (Ember).** Two coupled mechanisms:
1. **Multifaculty routing** — the router picks the *faculty* per sub-problem (CSP / Ising / graph / type /
   reduction / [HOU] / [∂4]). This is the α-rack router already designed in `alpha_struct_design.md` §1
   (`StructureRack`: router head → per-faculty structure heads) — Thread 5 is that router promoted from
   "route one problem" to "route each sub-problem of a decomposition."
2. **Gated routed RECURRENCE** — apply the organ-block N times with a **halt gate**, so the LM can do
   *nested* reasoning: solve a sub-problem, feed its result back, route the next. KEY: **the exact fixpoint
   IS a clean halt signal** — unlike ACT's *learned* halting (Graves, arXiv:1603.08983) or Universal
   Transformers' ponder cost (Dehghani et al., arXiv:1807.03819), we don't have to learn when to stop; the
   certified organ tells us when it has reached a fixpoint (no further sound narrowing possible → halt /
   abstain). A *sound* halt, not a learned-and-possibly-wrong one.

**Why it might help.** A learned halt is a notorious source of instability and reward-hacking (ACT's ponder
cost needs careful tuning; halting collapses). Replacing it with the organ's exact fixpoint is the same
move as everywhere else in GLaDOS — *let the certified structure do the job the neural net does badly.* The
router lets one woven model cover many problem types (Thread-4 coverage law (c)); routed recurrence is the
single-forward-pass version of the `north_star_orchestrator.md` multi-step loop (this thread = one organ
call with internal N-step recurrence + sound halt; the north star = the LM emitting many organ calls across
a CoT).

**Honest assessment: likely-helps (the sound-halt insight is genuinely good), but partly already in flight.**
The router exists in the α-rack design; the sound-fixpoint-halt is a clean, defensible improvement over ACT
that's worth stating as a contribution. The open uncertain piece is whether *gated routed recurrence inside
one forward pass* buys nested-reasoning capability over just stacking blocks (Thread 6) or running the
full orchestrator loop (north star) — i.e. is the "single-call N-recurrence with sound halt" a distinct
useful regime, or does it collapse into either Thread 6 (stacking) or the north-star loop?

**Money-bar verdict: local-spike now, box later.**
- *local-spike:* the **sound-fixpoint-halt vs learned-ACT-halt** comparison can be prototyped CPU-side on
  the standalone organ (does halting-at-certified-fixpoint match/beat a learned halt on
  steps-to-solve / accuracy?). Cheap, decision-relevant (validates the "don't learn the halt" thesis),
  uncertain. Good local spike.
- *box (later):* gated routed recurrence woven into a host for nested multi-faculty problems — fund only
  *after* the orchestrator framing (Thread 6 / north star) clarifies whether single-call recurrence is a
  distinct regime worth a separate study. Until then it risks duplicating the organ-block study.

**Connections.** `alpha_struct_design.md` §1 (the `StructureRack` router this builds on);
`north_star_orchestrator.md` (the single-call recurrence is the compressed form of the multi-step
orchestration loop; "gated recurrence" ↔ "organ becomes a tool the LM calls repeatedly"); ACT / Universal
Transformers (the learned-halt baselines this beats with a sound halt); Thread 6 (stacking vs recurrence is
the deeper-vs-wider question).

---

## Thread 6 — Iterating / stacking the organ as a BLOCK ("if one is good, two with flows is wilder")  ·  *[Phase 1 — live study]*

**The idea (Ember).** Treat the woven organ as a reusable **block** and STACK it — heterogeneous organ
stacks (e.g. **narrow → energy → narrow**) with **lattice-flow between** the blocks (each block's narrowed
lattice / readout feeds the next). The analogy is Universal-Transformer / looped-transformer / Ouro
(weight-tied recurrent depth) — **but for CHECKED reasoning**: instead of looping an unconstrained
transformer block, we loop/stack *certified organ blocks*, so every stacked step is sound. This is being
studied NOW (the live organ-block study). The open question Ember flags: **deeper (more recurrence R) vs
wider (more stacked faculties) vs both** — the architectural shape of the checked-reasoning stack.

**Why it might help.** Looped/recurrent-depth models (Universal Transformers, arXiv:1807.03819; *Looped
Transformers as Programmable Computers*, Giannou et al. 2023, arXiv:2301.13196; the Ouro looped-LM and
Geiping et al.'s recurrent-depth latent reasoning, arXiv:2502.05171) show that weight-tied depth buys
test-time compute and algorithmic generalization. GLaDOS adds the thing those lack: **soundness at every
loop**. A heterogeneous stack (narrow→energy→narrow) lets different faculties compose *in depth* — the
energy/DEQ block (`future_directions.md` §3, relaxation/optimization) does soft constraint satisfaction
between two exact-narrowing passes, with lattice-flow carrying the candidate sets across. That's a richer
compositional structure than a single faculty or a single reduced-product layer.

**Honest assessment: likely-helps AND already being measured — the open question is the right one.** Block-
stacking is a natural, well-motivated extension and the deeper-vs-wider-vs-both question is genuinely the
crux of "what shape should the checked-reasoning stack be?" Since it's in flight, the roadmap's job is to
make sure the live study *answers the deeper/wider/both question explicitly* (an ablation grid: R ∈
{R₁..} × #stacked-faculties ∈ {1,2,3}), not just "does stacking help y/n."

**Money-bar verdict: box (already funded — it's the live organ-block study). Roadmap action: sharpen its
design.** Decision-relevant (it sets the architecture) AND uncertain (deeper vs wider is genuinely open).
The spend is justified; ensure the study reports the **R × #faculties grid** (which also harvests Thread-4
laws (a) and (c) for free) and tests at least one **heterogeneous** stack (narrow→energy→narrow with
lattice-flow), not only homogeneous repetition.

**Connections.** Thread 4 (a)+(c) (the R-depth + faculty-stacking laws ARE this study's output);
Thread 5 (stacking ↔ routed recurrence — the wider vs deeper duals of the same question);
`future_directions.md` §3 (the energy/equilibrium DEQ block that makes a heterogeneous narrow→energy→narrow
stack meaningful; also predictive-coding's relaxation, Thread 1); `north_star_orchestrator.md` (a stacked
block is the in-pass primitive the orchestrator loop calls); Universal Transformers / Ouro / looped-
transformer lit (the unchecked analogs this checks).

---

## Thread 7 — Theorem proving / Lean as the natural-home application (the philosophical home, but LATE)  ·  *[Phase 2 — the endpoint]*

**The idea (Ember).** Theorem proving in **Lean** is the deepest fit for the entire GLaDOS bet — but it
lands LATE, because it depends on (i) the woven mechanism actually working, (ii) the HOU organ (Thread 2),
and (iii) a Lean integration. Why it's the *philosophical* home, not just another task:

- **The exact verifier GLaDOS is built around ALREADY EXISTS — it's the Lean kernel.** Everywhere else we
  have to *build and argue for* the output-checker (and defend its soundness — `alpha_struct_design.md` §3
  spends pages making the floor sound w.r.t. α's compiled semantics). In Lean, proofs are
  checked-by-construction by the kernel; soundness is **native, uncontroversial, and someone else's
  problem**. The whole "soundness is a permission, the verifier is exact and un-hackable"
  (`north_star_orchestrator.md`) thesis is *literally instantiated* by the Lean kernel — no checker to
  build, no soundness argument to win.
- **HOU (Thread 2) IS Lean's elaborator.** Higher-order unification is the literal engine of dependent-type
  proof search / elaboration — so the HOU organ is not an *analogy* to theorem-proving machinery, it is a
  *piece of it*. Building the HOU faculty is building a component of a Lean co-processor. This is the
  tightest cross-thread coupling in the whole roadmap.

**The harness + lineage.** LeanDojo (Yang et al., arXiv:2306.15626) for environment + retrieval;
miniF2F (Zheng et al., arXiv:2109.00110) and ProofNet (arXiv:2302.12433) as benchmarks; the prover lineage
is ReProver (LeanDojo) → DeepSeek-Prover / DeepSeek-Prover-V1.5 (arXiv:2405.14333, 2408.08152) → the
huge-RL-search provers (AlphaProof-class).

**THE ANGLE (crucial, honest).** **NOT a standalone prover.** A Gemma-12B is far too small to compete with
the giant RL-search provers on absolute miniF2F/ProofNet SOTA — and pretending otherwise is the exact
overclaim `codex_organ_review.md` warns against. Instead, the organ is a **checked reasoning CO-PROCESSOR**:
it handles the *mechanical / checkable* parts of a proof —
- unification / elaboration subgoals via the **HOU organ** (Thread 2),
- premise & equational-rewrite subgoals,
- arithmetic via the **modular / affine** ops (`bank.py` modular SNF, GF(2)),
- propositional deduction via the **narrower** (CSP/SAT faculty),

so the small LM spends its limited capacity on the **CREATIVE** part — *which* lemma, *what* argument shape,
the high-level proof plan. This is exactly the `north_star_orchestrator.md` division of labor (LM =
meta-reasoner / strategist; organ = certified primitive-solver) instantiated on Lean.

**Measure UPLIFT, not absolute SOTA.** The honest, *more compelling* metric: `small-LM + organ-coprocessor
> small-LM` on miniF2F/ProofNet — because on a small model **scale cannot be the explanation** for the
improvement (the Thread-3 / Thread-8 logic: a win on a small base is attributable to the organ, not to
parameters). Absolute SOTA is the giant provers' game; *organ-attributable uplift on a fixed small base* is
ours.

**Honest assessment: dream / major-future-direction with too many unbuilt dependencies NOW.** It needs the
woven mechanism proven on real NL (ALPHA_STRUCT passing), the HOU organ shipped (Thread 2), AND a Lean
integration built — three serial prerequisites. It is the *philosophical endpoint* of the deep-integrated
GLaDOS (Thread 8, Phase 2), not a near-term study.

**Money-bar verdict: discuss-only NOW; a cheap design-scoping spike once HOU exists.** Funding a Lean study
today fails the pre-mortem on *cost/readiness*, not on relevance — the deps don't exist, so nothing we'd
learn changes what we build next (we'd build HOU regardless). Record it as the major future direction it
is. The one near-ish action: **once the Thread-2 HOU spike lands, a cheap CPU/design spike** scoping the
LeanDojo integration + which subgoal types the bank can actually discharge (map our faculties → Lean tactic
classes) is warranted — and it sharpens the HOU spike's own success criteria.

**Connections.** Thread 2 (HOU IS the elaborator — its strongest justification is that it's a Lean
component); Thread 3 (same uplift-not-SOTA logic, here with a *native* verifier); Thread 8 (Lean lives
squarely in the deep-integrated Phase-2 bucket); `north_star_orchestrator.md` (LM-strategist / organ-
primitive division of labor; sound verifier wants search — Lean's kernel is the ultimate sound verifier);
`future_directions.md` §2 (math-proof / Lean-as-checker already flagged on the verifier ladder).

---

## Thread 8 — Model-independence vs deep-integration: resolve as a SEQUENCE, not a conflict  ·  *[the spine — Phase 1 → Phase 2]*

**The idea (Ember's sharp architectural insight).** There appear to be two GLaDOSes in tension; they are
not in conflict — they are **Phase 1 and Phase 2 of one program.**

- **(a) The PORTABLE GLaDOS** — frozen base + bolt-on organ, weights-only, the 7-base graft matrix
  (`GLADOS.md`). Its job is to prove the **mechanism GENERALIZES** across architectures (Pythia → OLMo-3-
  32B, incl. a Mamba hybrid). Here model-independence is a **feature of the demonstration** — it's *how* you
  show the organ isn't a quirk of one base. **This is the SCIENCE / the paper.**
- **(b) The INTEGRATED GLaDOS** — organ co-designed with the host, woven at *multiple* layers, the host
  trained from the start to wield it (the OLMo full-recipe re-run, `future_directions.md` §4). Its job is to
  **actually superpower one model on hard evals**. Here model-independence is a **cost happily paid for
  performance** — you give up portability to go deep. **This is the PRODUCT / the dream.**

**The resolution: prove-portable-THEN-go-deep.** They are not a fork in the road; (b) is the *maturation*
of (a). Evidence it's a continuum, not a cliff: the **LoRA study already showed the seam** — bidirectional
host-adaptation beat the frozen-host bolt-on by **~36%** (`GLADOS.md` weave stage; the woven-vs-frozen
result). Deep integration is simply *more of that* — more layers adapted, host trained to wield the organ
from the start, organ shape co-designed with the residual stream. **"GLaDOS stops being a bolt-on" is the
maturation of the thesis, not a betrayal of it.**

**Why this framing matters (it reframes the whole roadmap).** It re-casts the entire base-matrix story as
**Phase 1 of 2**, and it sorts every other thread into a phase:
- **Phase 1 (portable / science):** the 7-base graft matrix, the MATRIX scaling law (Thread 4e),
  ALPHA_STRUCT, the causal-control honesty table, the narrow Tier-1 wins (Thread 3 step 1). *Prove the
  mechanism generalizes.*
- **Phase 2 (integrated / product):** deep multi-layer weave, host-trained-to-wield, theorem-proving
  (Thread 7), code (Thread 3 step 2), the HOU organ in service of a co-designed host (Thread 2 → 7), broad
  uplift (Thread 3 step 3), the orchestrator loop (`north_star_orchestrator.md`). *Superpower one model.* —
  All the "dream" threads live here.

**Honest assessment: likely-helps as a FRAMING (it's a paper-narrative + program-sequencing decision, not
an empirical claim).** This isn't a study; it's the organizing principle that keeps us from arguing
"portable vs integrated" as if we had to choose. The one empirically-grounded anchor is the 36% LoRA seam —
which already tells us deep integration has headroom. The uncertain part is *how much further* the
deep-integrated phase goes beyond the bolt-on (that's the MATRIX law + the Phase-2 studies answering it).

**Money-bar verdict: discuss-only (it's a framing decision) — but ADOPT it as the roadmap's spine.** No
study to fund; the action is to *narrate every Phase-1 result as "step 1 of prove-portable-then-go-deep"* in
the paper, and to gate all Phase-2 boxes (Threads 7, 3-code, deep-weave) behind Phase-1 completion. The
36%-seam datapoint is the cheap back-of-envelope that justifies the Phase-2 bet without a new run.

**Connections.** `GLADOS.md` (the 7-base portable matrix = Phase 1; the 36% woven-vs-frozen seam = the
evidence Phase 2 has headroom); `future_directions.md` §4 (the OLMo full-recipe re-run = the canonical
Phase-2 deep-integration move) and §5 (residual-stream-agnostic graft = what *enables* Phase 1's
portability); Thread 3 (narrow = Phase 1, code/broad = Phase 2); Thread 4e (the MATRIX law spans the
Phase-1→2 transition); Thread 7 (Lean = the Phase-2 endpoint); `north_star_orchestrator.md` (the
orchestrator loop is itself a Phase-2 deep-integration: the host trained to *reason with* the organ, not
merely *use* it).

---

## RANKED — the next 5 studies (decision-relevance × uncertainty × cost)

Prioritized research queue. Rank = high decision-relevance × high uncertainty × favorable cost. (The live
organ-block study, Thread 6, is already running and harvests Threads 4a/4c — it's the implicit #0; these
five are what to fund next.)

1. **α-as-search via verifier-guided ES vs gradient, on the frozen-OLMo structure probe**
   *(Thread 1 local-spike + Thread 2 proposer mechanics).* Cheapest of the high-leverage set — runs on the
   structure-faithfulness probe we're already building (`alpha_struct_design.md` §5.5). Decision-relevant:
   it's the make-or-break α step we *cannot backprop through*, and a sound-verifier ES is the one regime
   where neuroevolution should win. Uncertain: nobody's measured ES-with-an-exact-verifier on factor-graph
   compilation. Cost: low (CPU / small-GPU, piggybacks the probe). **Do first — it de-risks ALPHA_STRUCT,
   which gates everything downstream.**

2. **HOU faculty feasibility spike — certified Huet pattern-unifier + neural branch-ranker**
   *(Thread 2 box).* Decision-relevant: it's the gate to the entire code/proof arm (Thread 3 steps 2–3).
   Uncertain: does neural guidance prune Huet's flex-rigid branching at our scale, or drown in flex-flex?
   Cost: moderate, and *bounded* — certified kernel is textbook, metric is search-nodes-expanded with vs
   without guidance, soundness 100% by construction, NO LM weave. **Highest new-capability upside per
   dollar; the one genuinely new faculty worth a box.**

3. **frozen-strong-base (Gemma) + organ vs frozen-Gemma on Tier-1 narrow reasoning + full causal controls**
   *(Thread 3 box, step 1).* THE project headline: "capability scale didn't grant this; the organ did."
   Decision-relevant to the max. Uncertain: does uplift survive onto a base strong enough that easy wins
   are saturated? Cost: real (a strong-base weave + eval) but uses the proven graft matrix. **Gated on
   ALPHA_STRUCT passing (`alpha_struct_design.md` §4) — schedule after #1 succeeds.**

4. **The MATRIX scaling law — woven uplift vs base size across the 7/7 graft matrix**
   *(Thread 4 box, law (e)).* The quantitative form of #3: is GLaDOS a small-model crutch or a frontier-
   model capability-grant? Decision-relevant (it tells us where to aim the whole program). Uncertain
   (uplift could grow, plateau, or vanish with scale). Cost: high (multi-base sweep) — but it's the law the
   paper is built on. **Fund after #3 establishes the single-base uplift is real.**

5. **Sound-fixpoint-halt vs learned-ACT-halt, on the standalone organ**
   *(Thread 5 local-spike).* Validates the clean thesis "don't learn the halt — the certified fixpoint IS
   the halt signal." Decision-relevant (it's a defensible contribution over ACT/Universal-Transformer
   halting and shapes the recurrence design). Uncertain (does sound-halt match/beat learned-halt on
   steps-to-solve and accuracy?). Cost: low (CPU, standalone organ). **Cheap, crisp, paper-ready — slot it
   between the bigger boxes.**

All five are **Phase 1** (prove-the-mechanism) except #2's payoff, which reaches into Phase 2. Note the
**HOU design-scoping spike → Lean integration scoping** (Thread 7) becomes the natural #6 *the moment #2
lands* — cheap, and it sharpens HOU's own success criteria; it's the first concrete Phase-2 brick.

*Below the line (discuss-only / Phase 2, gated): theorem-proving / Lean (Thread 7) — the philosophical
endpoint, gated on ALPHA_STRUCT + HOU + a Lean integration; code + broad-uplift arms of Thread 3 — gated
on #2 + a procedural faculty; composition-depth law 4(d) — gated on the orchestrator loop; non-gradient
training as a wholesale SGD swap (PC/eqprop/FF/Hebbian) — record the lit, don't fund. The model-
independence/deep-integration resolution (Thread 8) is not a study at all — it's the spine: narrate every
Phase-1 result as "step 1 of prove-portable-then-go-deep."*

---

*Money-bar reminder for whoever picks this up: before any box, run the pre-mortem — "if it comes back the
other way, do we build something different?" If the honest answer is no, it's a local-spike or a
back-of-envelope, not a sweep. The discipline is the point. ( ◕‿◕ )*
