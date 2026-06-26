# Interpretability probe — the bank-woven organ, read LEARNED-FREE

*(DRAFT — design + rationale written first; the numbers below are filled from the real run on the
L40S box. Probe code: `clair/interp_probe_bank.py`. Target: the OVERHAULED consolidated organ
`clair/organ/bank_woven.py` — α-compile → verifier-gated composer (bank) → γ readout — NOT the
torn-down stopgap `oracle_readout.LiveLatentWoven` standalone narrower, and NOT the prior probe's
learned decoders.)*

## The constraint that shapes the whole probe: LEARNED-FREE readout

Ember's non-negotiable: **the interpretation must contain no trained decoder.** Every learned layer
stacked onto the readout is a layer of doubt — a probe that "recovers" structure may be recovering it
*from its own training*, not from the organ. So this probe reads only two kinds of signal:

- **DIRECT** — values read straight off the tensors and the composer's own data structures: α's raw
  per-cell compile `b0` thresholded at the *same* θ the live model uses (`σ(b0) ≥ θ`); the
  reduced-product `Trace` object (which reductions fired, which were verifier-gated, cardinality per
  round); the composed lattice's per-cell domains. Faithfulness is a **direct set comparison** to the
  exact per-cell transformer `clair.csp.exact_dedP` (the ground-truth narrowing, == α's J0 target).
- **BEHAVIORAL** — the model's *own* generative answer-correctness (the LM head scoring the legal
  answer strings) under the validated causal controls, including the oracle-override arm.

This is the explicit contrast with the **prior probe** (`clair/interp_probe.py`), which (a) targeted
the standalone `LatentOrgan` — the torn-down single-faculty stopgap, not the consolidated bank — and
(b) **fit MLP decoders** (`train_probe`, an `MLP`) to map α's latent output back to a factor-graph.
That is exactly the learned layer we refuse. Its α-faithfulness numbers measure *decodability under a
trained probe* (an upper bound on what α exposes); ours measure *what α actually compiled*, with
nothing learned in between.

## What α actually compiles in the consolidated organ (an honest architectural point)

In the bank-woven organ the division of labour is sharp, and it matters for what "faithfulness" can
even mean:

- **α compiles the per-cell candidate LATTICE** (`b0 [B,N,K]`) — for each named cell, a soft set over
  the K candidate values. It is trained by the standing **J0 dominate-dedₚ** loss directly on `b0`
  toward `exact_dedP` (the SATNet grounding fix). So α's job is: *from the host's hidden state of the
  problem text, predict the per-cell surviving-value sets.*
- **The relational structure** (A≠B, A<B, …) is **NOT compiled by α**. It is threaded from the problem
  instance into the records (`rec['csp']`) and consumed by the **certified composer** (Arc / Factor /
  Modular / GF2 / Macro), which is sound by construction. α's lattice enters the composition only as a
  *third verifier-gated neural proposal* (`_AlphaProposal`), so it can sharpen the composed lattice but
  can never drop a real survivor.

Therefore **α-compile faithfulness here is per-cell candidate-SET fidelity** (does α's compiled set
match the true dedₚ set per cell), not relational edge-recovery. The relational factor-graph is
legible *by construction* (it is read off the certified instance), and the composer's **routing** is
what exposes which relations got used — that is analysis #2, not a learned reconstruction. This is the
honest read of "read the latent compile directly": a latent dense projection thresholded to a set is
genuinely lossy, and we report exactly where.

## The five reads

1. **What problem did α pose? (DIRECT)** Threshold `b0` → per-cell sets; reverse-render to a readable
   instance (`A=red, B∈{red,green}, …`); set-compare to exact dedₚ → per-(cell,value)
   precision/recall/F1, per-cell exact-set & cardinality accuracy, determined-cell ("pin") detection.
   Lossiness is broken down **by rung** and **by true cardinality** of the cell.
2. **Which faculty did the composer route to? (DIRECT)** Read the reduced-product `Trace`: which bank
   reductions narrowed (arc / factor / modular / gf2 / macro / core_narrow_organ / alpha_compile) on
   which rungs, and which neural proposals were verifier-gated. Legible by construction.
3. **What did it conclude? (DIRECT)** The composed lattice → per-cell sets → answer/abstain, matched
   to the verified dedₚ solution (query-cell match, all-cell match, soundness, determined-query solved).
4. **Is α the bottleneck? (BEHAVIORAL)** Per held-out instance, correlate α-faithfulness (#1) with the
   model's downstream answer-correctness; plus the causal-control table and the **oracle-override** arm
   (inject exact dedₚ instead of α's composed lattice). If oracle-override ≫ true, the readout is
   solved and α's compile fidelity is the wall.
5. **Legible trajectories.** For a spread of examples: posed factor-graph → α-compiled lattice → which
   faculty fired (cardinality per round) → conclusion → answer vs dedₚ.

---

## RESULTS

*(filled from `runs/interp_bank.json` / `runs/probe_run.log` — base OLMo-2-0425-1B, regime `small`
(all curriculum rungs, id split), bank mode, two-stream + J0 + rich-state, organ pretrained 1500
steps, woven 1500 steps (+300 warm). Held-out determined-query eval pool.)*

### Training / engagement (context)
<!-- FILL: controls table (true/zero/corrupt/oracle/gate/lift/drop/engages) from weave -->

### 1. α-compile faithfulness (direct, no learned probe)
<!-- FILL: overall precision/recall/F1, exact-set, cardinality, pin-detect; by-rung table; by-cardinality -->
<!-- NARRATIVE: where is α lossy — which rungs drop survivors (recall<100) vs hallucinate (precision<100) -->

### 2. Composer routing (direct, from the trace)
<!-- FILL: faculty fired-fraction, gated counts, mean rounds, by-rung firing -->

### 3. Conclusion (direct, composed vs dedₚ)
<!-- FILL: query-match, all-match, soundness, determined-query-solved -->

### 4. Is α the bottleneck? (behavioral)
<!-- FILL: answer-acc true vs oracle-override; corr(faithful, correct); acc|faithful vs |unfaithful -->

### 5. Legible trajectories
<!-- FILL: 4-6 worked examples -->

---

## The honest interp story (for the paper)

<!-- FILL after numbers, but the spine: the structured organ exposes its intermediate reasoning —
α's posed per-cell lattice, the composer's faculty routing, and the composed conclusion are all read
DIRECTLY off tensors and the trace, with no learned probe. We report how faithfully α compiles (and
exactly where it is lossy: reading a dense latent compile as a thresholded set is genuinely lossy),
and whether a faithful compile predicts a correct answer (is α the wall). Caveats: faithfulness is
per-cell-set fidelity, not relational-edge recovery (relations are threaded + certified, not
α-compiled); the threshold-to-set read discards α's calibrated uncertainty that the rich-state γ
actually forwards to the LM. -->
