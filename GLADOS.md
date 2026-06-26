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
  protocol.py   the typed spine: Reduction / CSPState / Certificate + the structured-γ readout
  bank.py       the registry of validated beasts as sound reductions (certified ops + neural guidance)
  compose.py    verifier-gated certified-reduction composition (the sound reduced product)
  graft.py      THE model-agnostic neural-graft — graft_organ(host, organ, config) onto any of 7 bases
  train.py      THE canonical staged pipeline — pretrain_organ → weave → rlvr
  eval.py       THE arbiter — re-exports clair.eval_suite.run_eval_suite
  smoke.py      the one-command end-to-end smoke (pretrain → graft → weave → eval; needs a GPU)
  selftest.py   the consolidated CPU smoke (python -m clair.organ.selftest)
  README.md     the per-beast status table + the spine contract
```

---

## WHAT the organ is — bank + spine

A **bank** of validated "beasts" (arc/factor consistency, exact dedₚ, modular SNF, GF(2) row-space,
reach-doubling macros, FOL forward-chaining, energy, and the learned `core_narrow_organ` /
`blade_affine_organ`), each wired to ONE typed **spine** (`protocol.Reduction`):

- an abstract **STATE** (the lattice element it narrows; `CSPState` = per-cell domains over a finite CSP);
- **`reduce(state) → state′`** with γ(state′) ⊆ γ(state) — a **sound narrowing**, or a no-op = ABSTAIN;
- a **`certificate()`** — `sound-by-construction` / `neural-guidance` / `approximate` + the completeness caveat;
- the uniform **`survival(state, K) → surv[n,K]`** structured-γ readout — the exact tensor `OracleGamma`
  projects into the host residual stream. **One channel the LM reads every organ through.**

The CSP-domain reductions **compose** as a verifier-gated reduced product (`compose.reduced_product`):
certified ops are trusted; neural/approximate ops are gated by the exact per-cell verifier
(`csp.exact_dedP`), so a wrong neural proposal can never poison the shared state. (Full per-beast table:
[`clair/organ/README.md`](clair/organ/README.md).)

## HOW it's trained — three oracle-supervised stages

Every label is **free and exact** (witness-first generation + `clair.csp.exact_dedP`). No human labels.
Detail: [`notes/training.md`](notes/training.md).

1. **`pretrain_organ`** (STAGE 1) — train the deductor STANDALONE on the soundness-asymmetric
   **dominate-dedₚ** loss (heavy penalty `wpos≈6` for dropping a value some solution uses; light
   `wneg≈0.5` for keeping a loose extra), **on-policy** over the 7-rung MIX, monotone (survival only
   decreases). **Recipe (from the sweep): ~1.5M params, R=12, ~30% composed.** → a frozen, general,
   sound narrower (`runs/general_organ_full.pt`).
2. **`weave`** (STAGE 2) — freeze the organ; train the woven readout: frozen base **+ LoRA + latent-α**
   (dense projection host-hidden → organ inputs, fully latent) **+ frozen organ + zero-init γ readout**,
   on the answer-span **LM cross-entropy**. The engagement mechanism (consolidated here, lifted out of
   the old experiment driver):
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
