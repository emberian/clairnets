# How GLaDOS is trained

The training is **staged, curriculum-driven, and oracle-supervised** — *not* classic semisupervised. Every
label is **free and exact**: witness-first generation + the exact deductor (`clair.csp.exact_dedP`) produce the
narrowing targets, the answers, and the verifiable rewards. No human labels. The "curriculum" is the *staging
order* + the readout-forcing schedule + the domain/composition/N mix.

Entry point: `clair/organ/train.py` — `organ` (Stage 1) and `weave` (Stage 2/3).

## Stage 1 — pretrain the organ (standalone)
The deductor learns to **narrow soundly**. Base LLM not involved.
- **Loss: dominate-dedₚ** — soundness-asymmetric supervised loss vs the exact per-cell transformer
  `exact_dedP`: heavy penalty (`wpos≈6`) for eliminating a value some solution uses (UNSOUND), light penalty
  (`wneg≈0.5`) for keeping an extra (loose-but-sound). Monotone: survival only decreases (meet).
- **On-policy**: each batch is the previous step's narrowed lattice rolled forward by the organ; targets
  re-computed by the exact deductor each pass. (Bug fixed: recompute `given` after each meet.)
- **Curriculum / data**: the broad multi-domain corpus — all beast generators (csp families · xor/GF2 · modular ·
  graph · perm · type · fol · optimize) + **union relations** (named families + random allowed-tuple tables,
  *additive* — fixes OOD-novel-relation soundness) + wide-N + **~30% composed problems** (multi-skill).
- **Recipe (from the sweep)**: ~1.5M params, R=12 (R8 floor), ~30% composition mixup. Composition recall is
  ~free (the meet composes); composition coverage buys *soundness* on novel compositions, and that need *grows
  with scale*.
- **Out**: a frozen, general, sound narrower checkpoint (`runs/general_organ_full.pt`).

## Stage 2 — freeze the organ, train the woven readout
The LLM learns to **generate through** the frozen organ.
- **FREEZE** the Stage-1 organ (or slow-thaw / EMA — the thaw study decides; default freeze).
- **Train**: OLMo base FROZEN + **LoRA** + **latent-α** (dense projection OLMo hidden → organ input tensors —
  *fully latent*, no symbolic extraction) + **γ** (structured readback, zero-init gate).
- **Forward**: OLMo hidden → latent-α → frozen organ narrows (under `no_grad`, no backprop through repeated OLMo
  passes) → γ injects the narrowed lattice into the residual → LM head generates the answer.
- **Loss: LM cross-entropy** on the answer span only. α learns from the gradient flowing through the organ's
  downstream success (the LM does better when α compiles a usable problem).
- **Curriculum: readout-forcing** — start cells-mode (no facts in the prompt → the organ is the ONLY route → the
  γ-decode forms before the LM can camp in a text-shortcut basin) → then mix in full-text.
- **Out**: a woven model where the LM causally wields the organ (corrupt the lattice → generation collapses).

## Stage 3 — RLVR + consolidation
- **RLVR** (GRPO, verifiable reward = the exact checker; + the **organ-as-process-reward**, LSRL-style: per-step
  candidate-set cardinality drop, mixed 0.7·outcome + 0.3·process) — sharpens *calibration*, not capacity. Report
  pass@1 AND pass@k; ProRL + random-reward controls; OLMo is the honest base.
- **Consolidation**: continued-pretrain on **Dolma ⊕ reasoning corpus** (general + organ-shaped, tuned ratio)
  with the organ in place → organ-use folds into general behavior without catastrophic forgetting. (The
  preserve-&-generalize corpus.)

## What each stage proved (the evidence)
- readout works (corrupt→0, oracle de-risk) · a *frozen learned* organ reads like the oracle · **cold co-training
  fails → staging fixes it** · the staged generative model uses the organ across 7 rung types (corrupt→2.7% with
  text present) · **next-token CPT recruits the organ with no reward** (it's an architecture, not an RLVR trick) ·
  **split-brain: the LM wields the organ 99.9% over adversarial text** (necessity, not opportunism).
- Honest scope: the "woven" results to date are **readout-isolation** (lattice handed in); the **live latent-α**
  woven model (α compiles from hidden, not the ground-truth CSP) is the real thing — built after the codex
  pre-flight. α (latent compile) is the hardest part and the critical path.

## Honest caveats
- The neural organ's soundness is *trained*, not by-construction — the certified operators are the soundness floor;
  the net is guidance ("neural proposes, certified guard narrows").
- The model trusts the organ *unconditionally* (split-brain) → **α-correctness is the whole ballgame** (a wrong
  compile → confidently wrong organ).
- The affine wall is real (bounded-width): single ternary constraints need grade-3; XOR *systems* need the
  certified row-space organ; neither is beaten by depth alone.

## Base matrix (the bolt-on datapoints — weights-only, so each is cheap)
The pretrained organ is base-agnostic; the woven readout (Stage 2/3) is run on MULTIPLE bases, each a paper datapoint:
- **OLMo-3-7B (PRIMARY)** — interp anchor, full causal controls, the honest mechanism (fully-open recipe). Run #1.
- **OLMo-3-32B** — scale datapoint (does the uplift hold / grow with base size).
- **Gemma-4** — portability + uplift on an already-strong, code-trained model.
- **Qwen-3.6** — portability, a second architecture family.
Goal: show the technique uplifts across TWO scales and THREE model families. 7B is never skipped.
Compute: woven = frozen base + LoRA + α + γ (QLoRA for 32B) → 7B/12B fit one L40S; 32B fits one A100-80 or L40S@4-bit.
