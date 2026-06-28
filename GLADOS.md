# GLADOS — the single front door

**This IS GLaDOS.** One canonical structure: *this is the organ · this is how it's trained · this is
how it's neural-grafted · this is how it's evaluated.* Everything lives under **`clair/organ/`**; the
scattered experiment scripts (`run_glados_staged.py`, `prove_woven_smallbases.py`, `eval_suite.py`,
`rlvr_pipeline.py`) are the validated *mechanism* this front-door wires together — read them only when
you want the internals.

> Paper-level overview: [`README.md`](README.md) · vocabulary: [`GLOSSARY.md`](GLOSSARY.md) ·
> per-beast status table + the typed spine: [`clair/organ/README.md`](clair/organ/README.md) ·
> training detail: [`notes/training.md`](notes/training.md) · readout-forcing:
> [`notes/readout_forcing.md`](notes/readout_forcing.md).

```
clair/organ/
  protocol.py     the typed spine: Reduction / CSPState / Certificate + the structured-γ readout
  bank.py         the registry of validated beasts as sound reductions (certified ops + neural guidance)
  compose.py      verifier-gated certified-reduction composition (the sound reduced product)
  bank_woven.py   ★ THE live woven organ — α → composer(bank) → γ, in the host residual stream
  reductions.py   the cross-type reduction graph (ProblemReduction + Dijkstra cost-routing)
  readout_bridge.py  rich-state survival readouts for the non-CSP faculties (energy / unification)
  process_reward.py  the organ-as-process-reward (LSRL-style cardinality drop, soundness-gated)
  graft.py        THE model-agnostic neural-graft — graft_organ(host, organ, config) onto any of 7 bases
  train.py        THE canonical staged pipeline — pretrain_organ → weave → rlvr
  eval.py         THE arbiter — re-exports clair.eval_suite.run_eval_suite
  smoke.py        the one-command end-to-end smoke (pretrain → graft → weave → eval; needs a GPU)
  selftest.py     the consolidated CPU smoke (python -m clair.organ.selftest)
  README.md       the per-beast status table + the spine contract

clair_fast/        the Rust (PyO3 + rayon) hot path: exact_dedP / solutions / AC, 29–56× CPU, bitwise-identical
clair/eval_suite.py            the external arbiter (Tier-1/2/3 + causal controls + pass@k, base-agnostic)
clair/interp_probe_bank.py     the LEARNED-FREE interp probe on the bank-woven organ → runs/interp_bank.json
```

---

## WHAT the organ is — bank + spine

A **bank** of validated "beasts" (arc/factor consistency, exact dedₚ, modular SNF, GF(2) row-space,
reach-doubling macros, FOL forward-chaining, energy, and the learned `core_narrow_organ` /
`blade_affine_organ`), each wired to ONE typed **spine** (`protocol.Reduction`):

- an abstract **STATE** (the lattice element it narrows; `CSPState` = per-cell domains over a finite CSP);
- **`reduce(state) → state′`** with γ(state′) ⊆ γ(state) — a **sound narrowing OF THAT STATE**, or a no-op = ABSTAIN;
- a **`certificate()`** — `sound-by-construction` / `neural-guidance` / `approximate` + the completeness caveat
  (`sound` = operator-sound *relative to the given CSP* (A), **not** answer-sound (D) — [`notes/soundness.md`](notes/soundness.md));
- the uniform **`survival(state, K) → surv[n,K]`** structured-γ readout — the exact tensor `OracleGamma`
  projects into the host residual stream. **One channel the LM reads every organ through.**

The CSP-domain reductions **compose** as a verifier-gated reduced product (`compose.reduced_product`):
certified ops are trusted; neural/approximate ops are gated by the exact per-cell verifier
(`csp.exact_dedP`), so a wrong neural proposal can never poison the shared state. (Full per-beast table:
[`clair/organ/README.md`](clair/organ/README.md).)

**The LIVE woven organ — `bank_woven.py` (the consolidated artifact).** In the host residual stream the organ
slot IS the bank + composer: **α** (`DenseLatentProjector`) latently compiles the host hidden into a per-cell
candidate lattice; the **`BankComposerOrgan`** runs the verifier-gated reduced product on the problem's true
structure — the **certified floor** (Arc/Factor/Modular/GF2/Macro) + the *pretrained* `CoreNarrowOrgan` + α's
own compile, all gated; **γ** reads the *composed* lattice back through the zero-init gate. The certified
floor makes the injection **robust to a wrong per-cell *value*-compile by construction** (a wrong α value-compile
`b0` can only sharpen toward the exact transformer, never below it — it cannot poison the floor) — but the floor
is sound only *relative to the CSP it composes on*, so a wrong *structure*-compile `csp_α` is NOT covered: the
floor then solves the wrong problem and certifies the wrong answer (the SHUFFLED gap; see
[`notes/soundness.md`](notes/soundness.md) (C)). This replaced the old stopgap standalone
`LatentNarrower`; it **engages** (corrupt→drop ~+97 pts, lift +77, no-op@init 0.0, false-elim 0).

**The reduction graph — `reductions.py` (cross-type, vs the composer's within-type).** A typed
`ProblemReduction` graph encodes the Karp neighbourhood (MIS↔VC↔clique · 3-SAT/2-SAT/XOR-SAT→CSP ·
3-SAT→MIS→Ising · {MIS,MaxCut,partition,coloring}→Ising) — every edge **exact-verified end-to-end against X's
own solver**, reusing the Lucas NP→Ising encoders + `CSPState`; `ReductionGraph.route` is Dijkstra cost-routing
with a certified-over-approx tie-break. The composer routes *within* a state type; a reduction routes *across*
types, so the model learns *recognize-X · reduce-X→Y · solve-via-Y · decode*. Smoke on the real bank-woven: a
route-required family with no direct faculty (XOR-SAT) solved 100% via the reduction; held-out reduction-paths
100%. One faculty + edges = everything reducible to it.

## HOW it's trained — three oracle-supervised stages

Every label is **free and exact** (witness-first generation + `clair.csp.exact_dedP`). No human labels.
Detail: [`notes/training.md`](notes/training.md).

1. **`pretrain_organ`** (STAGE 1) — train the deductor STANDALONE on the soundness-asymmetric
   **dominate-dedₚ** loss (heavy penalty `wpos≈6` for dropping a value some solution uses; light
   `wneg≈0.5` for keeping a loose extra), **on-policy** over the 7-rung MIX, monotone (survival only
   decreases). **Recipe (from the sweep): ~1.5M params, R=12, ~30% composed.** → a frozen, general,
   sound narrower (`runs/general_organ_full.pt`).
2. **`weave`** (STAGE 2, `organ_mode="bank"` DEFAULT) — freeze the organ; train the woven readout: frozen
   base **+ LoRA + latent-α** (dense projection host-hidden → organ inputs, fully latent) **+ the frozen
   bank+composer organ (`bank_woven.BankComposerOrgan`) + zero-init γ readout**, on the answer-span **LM
   cross-entropy**. (`organ_mode="latent"` keeps the legacy single-faculty `LatentNarrower` for ablation.)
   The engagement mechanism (consolidated here, lifted out of the old experiment driver):
   - **TWO-STREAM lever** — α reads the FULL text to compile the constraints; the LM GENERATES from a
     **fact-ablated cells prompt** (roster + question only), so the organ is the *only* route to the answer;
   - **J0 direct α-supervision** — a standing dominate-dedₚ loss on α's RAW compile `b0` (the SATNet
     grounding fix) so α compiles the *right* constraints from hidden, not a text-mimicking surrogate;
   - **rich-state / calibration readout** — the causal-control table at eval.
   → a woven model where the LM **causally wields** the organ (corrupt the lattice → generation collapses).
3. **`rlvr`** (STAGE 3) — RL-from-verifiable-rewards (TRL **Dr.GRPO**, exact-verifier reward); the
   **organ-as-process-reward** (LSRL-style per-step cardinality drop, mixed 0.7·outcome + 0.3·process)
   plugs in at `clair.rlvr_pipeline`'s ORGAN INSERTION POINT. Sharpens *calibration*, not capacity;
   report pass@1 **and** pass@k.

## HOW it's grafted — `graft.py`, the 7/7 matrix

`graft_organ(host, organ, woven_config)` is **residual-stream-agnostic**: a MID hook reads a layer's
hidden; an INJECT hook adds γ(organ(α(hidden))) to a later layer, gated by a **zero-init tanh scalar ⇒
bitwise no-op at init**. `find_decoder_layers` resolves `model.layers` / `gpt_neox.layers` /
`language_model.layers` / `transformer.h` / bare `layers`; mid/inject come from the config or are
auto-chosen and **snap to a full-attention block on hybrids**. The same coupling
`train_live_woven` builds — both are model-agnostic because the decoder-layer lookup is generalized.

**Proven 7/7** (`runs/woven_config_*.json`, no-op@init bitwise + gate-open moves logits, ~100M→32B,
six families incl. an SSM-hybrid):

| base | arch | layers / hidden | mid → inject | no-op@init |
|---|---|---|---|---|
| Pythia-160M | GPT-NeoX | 12 / 768 | 4 → 8 | ✓ 0.0 |
| OLMo-2-1B | Olmo2 | 16 / 2048 | 5 → 10 | ✓ 0.0 |
| SmolLM3-3B | SmolLM3 | 36 / 2048 | 12 → 24 | ✓ 0.0 |
| Gemma-4-12B | Gemma4 (multimodal-wrapped) | 48 / 3840 | 23 → 35 | ✓ 0.0 |
| Qwen-3.6-27B | Qwen3.6 (full/linear-attn) | 64 / 5120 | 31 → 47 | ✓ 0.0 |
| Nemotron-H-8B | **Mamba-2 + attn + MLP hybrid** | 52 / 4096 | 17 → 35 | ✓ 0.0 |
| OLMo-3-32B | Olmo3 (QLoRA 4-bit) | 64 / 5120 | 21 → 32 | ✓ 0.0 |

The Nemotron-H datapoint is decisive: the block boundary is a plain residual add for *every* block type,
so γ-injection is clean even when the inject layer is a Mamba-2 SSM block.

## HOW it's evaluated — `eval.py`, the arbiter

`run_eval_suite(base, woven_ckpt=...)` produces the standardized table over arms
**base · text-LoRA · woven · oracle-override**:

- **Tier 1** — exact-verified logic/CSP (the woven model's home turf) + the **causal-control table**
  (true / shuffle / permute / corrupt / zero) + the reasoning-gym exact ceiling;
- **Tier 2** — reasoning transfer (GSM8K / ARC / …);
- **Tier 3** — no-harm general ability (HellaSwag / WikiText / …);
- **pass@1 and pass@k** throughout.

The causal controls are the honesty gate: WOVEN > TEXT-LoRA *and* corrupt/shuffle/permute → collapse
means the LM genuinely reads the lattice; otherwise the organ is bypassed and we say so.

## HOW it's READ — `interp_probe_bank.py`, learned-free legibility

The interp advantage over a dense bypass is that we can read what the organ posed and concluded **with no
trained decoder anywhere** (`clair.interp_probe_bank`). Every number is either DIRECT (threshold α's raw `b0`
at the live θ → per-cell sets; read the composer's discrete reduced-product trace; set-compare to the exact
dedₚ) or BEHAVIORAL (the model's own generative answer under the causal controls). On the real bank-woven
(n = 300, `runs/interp_bank.json`): **α-compile faithfulness F1 0.90 / recall 0.98** (precision 0.82), by-rung
alldiff 0.97 > coloring 0.90 > arithmetic 0.89 > equality 0.86 > ordering 0.85; candidate-set recovery
0.93–1.0 across cardinalities; query reaches a singleton 100%. The trace shows which faculty fired and which
neural proposals were verifier-gated, legible by construction.

**Honest scope (the open frontier).** In this *easy* regime the certified floor carries answer-correctness on
its own — *because the true structure is handed in* (ground-truth structure = the (D) boundary check; soundness
becomes answer-soundness only where that check exists — see [`notes/soundness.md`](notes/soundness.md)) —
answer-acc 1.0 whether or not α is faithful, corr(faithful, correct) = 0 (no variance), so the
**bottleneck test is saturated / inconclusive** here. The genuinely open piece is **α-on-real-NL** (α compiling
from text with no provided structure, where correctness drops and failures attribute): the `eval_real_nl` hook
is wired; the result is the next frontier.

## The CPU hot path — `clair_fast/` (Rust, PyO3 + rayon)

The pretrain bottleneck was the single-threaded Python exact-dedₚ ground-truth path (not the GPU — the organs
are tiny). `clair_fast/` ports `exact_dedP` / `solutions` / `ac_step` + a batched `dedp_batch` (GIL-released
in-process rayon, replacing the pickling ProcessPool), tuples packed `u64`, domains bitmasks. **29–56× faster**
(per-call 6.7× small → 55× large; full data-path 29× / 56×), **bitwise-identical** (13,590 comparisons, 0
mismatches). `csp.py` dispatches to it when importable; `CLAIR_NO_FAST=1` forces the verified pure-Python path.
This unblocks the real organ-pretrain at the dⁿ-explosion budgets.

---

## One command per stage

```bash
# STAGE 1 — pretrain the organ (the recipe: ~1.5M params, R=12, ~30% composed)
python -m clair.organ.train pretrain --steps 1500 --out runs/general_organ_full.pt

# STAGE 2 — graft + weave into a host (any of the 7 bases; --regime small|hard|large)
python -m clair.organ.train weave --base allenai/OLMo-2-0425-1B --regime hard --steps 2500 --out runs/woven.pt

# STAGE 3 — RLVR (Dr.GRPO, exact-verifier reward; organ-as-process-reward at the insertion point)
python -m clair.organ.train rlvr --task chain_sum --steps 300

# EVAL — the arbiter (Tier-1/2/3 + causal controls + pass@k)
python -m clair.organ.eval --base allenai/OLMo-2-0425-1B --woven_ckpt runs/woven.pt

# SMOKE — the whole flow end-to-end at tiny size on a real host (NEEDS A GPU)
python -m clair.organ.smoke --base allenai/OLMo-2-0425-1B

# SELFTEST — the consolidated CPU proof (bank soundness + composer + readout no-op + organ-train)
python -m clair.organ.selftest
```

The graft itself (model-agnostic + no-op@init proof, no training) on any base:

```python
from transformers import AutoModelForCausalLM
from clair.organ.graft import graft_organ
host  = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")
woven = graft_organ(host, woven_config="runs/woven_config_pythia-160m.json")  # gate=0 ⇒ no-op@init
```
