"""clair/organ/graft.py — THE model-agnostic neural-graft for the GLaDOS organ.

ONE function, `graft_organ(hf_model, organ, woven_config) -> woven_model`, that splices the organ
into ANY of the 7 proven bases (OLMo-2-1B / OLMo-3-32B / SmolLM3-3B / Pythia-160M / Gemma-4 /
Qwen-3.6 / Nemotron-H-8B — Transformer AND SSM-hybrid). It is the consolidation of the 7/7-proven
swarm coupling (`clair.prove_woven_smallbases.WovenCoupling` + `find_decoder_layers`) and the live
woven module (`clair.oracle_readout.LiveLatentWoven`) into one residual-stream-agnostic entry.

The coupling (identical math on every base):
  * a MID decoder-layer forward hook captures the host hidden h (read-only);
  * α = `clair.latent_organ.DenseLatentProjector` (sized to THIS host's hidden D) compiles h into the
    organ's input tensors — fully latent, no symbolic extraction;
  * the FROZEN organ (`clair.latent_organ.LatentNarrower`) narrows the per-cell lattice;
  * an INJECT decoder-layer forward hook adds  γ(organ(α(h)))  to the residual stream, gated by a
    zero-init tanh scalar (`clair.oracle_readout.OracleGamma.alpha`) ⇒ **bitwise no-op at init**.

Residual-stream-agnostic by construction: the coupling only ever READS a layer's output hidden and
ADDS to a later layer's output hidden. Every HF decoder block returns the post-residual hidden as the
first output element (a plain residual add), so the injection is a clean residual edit regardless of
the mixer (attention / Mamba-2 SSM / linear-attention / sliding-window). On hybrids the inject layer
is snapped onto a full-attention block when the host advertises its layer kinds (defensive; the
Nemotron-H proof shows even a Mamba-2 inject block is clean).

  from clair.organ.graft import graft_organ
  woven = graft_organ(hf_model, organ=narrower, woven_config="runs/woven_config_olmo2-1b.json")
"""
from __future__ import annotations

import json
import os
from typing import Any

import torch

from ..latent_organ import DenseLatentProjector, LatentNarrower
from ..oracle_readout import LiveLatentWoven, OracleGamma  # noqa: F401 (OracleGamma re-export)


# ============================================================ decoder-layer resolution
def _unwrap(model):
    """Peel a PEFT wrapper to the underlying HF model (LiveLatentWoven hooks the raw decoder layers).
    NB: every HF PreTrainedModel exposes a `.base_model` property, so detect PEFT specifically (it owns
    `peft_config`); a plain HF model is returned untouched."""
    if hasattr(model, "peft_config") and hasattr(model, "base_model"):
        return model.base_model.model
    return model


def find_decoder_layers(model):
    """Locate the residual-stream decoder-block ModuleList on ANY HF causal-LM, returning
    (layers, dotted_path). Resolves the proven matrix:
      model.layers            — Llama-style (OLMo-2/3, SmolLM3, Qwen-3.6 text)
      gpt_neox.layers         — GPT-NeoX parallel-residual (Pythia)
      language_model.layers   — multimodal-wrapped LM (Gemma-4 unified)
      transformer.h           — GPT-2-style fallback
      *.layers                — Nemotron-H NemotronHBlock stack (Mamba-2 / MLP / attention hybrid)
    Works on a raw model or a PEFT-wrapped one (it unwraps first)."""
    base = _unwrap(model)
    probes = [
        ("model.layers",               lambda m: getattr(getattr(m, "model", None), "layers", None)),
        ("gpt_neox.layers",            lambda m: getattr(getattr(m, "gpt_neox", None), "layers", None)),
        ("language_model.layers",      lambda m: getattr(getattr(m, "language_model", None), "layers", None)),
        ("model.language_model.layers", lambda m: getattr(getattr(getattr(m, "model", None),
                                                                  "language_model", None), "layers", None)),
        ("transformer.h",             lambda m: getattr(getattr(m, "transformer", None), "h", None)),
        ("layers",                     lambda m: getattr(m, "layers", None)),
    ]
    for path, fn in probes:
        layers = fn(base)
        if layers is not None and len(layers) > 0:
            return layers, path
    raise RuntimeError(f"could not locate decoder layers on {type(base).__name__}")


def attention_layer_mask(config, nL):
    """Per-layer 'is this a FULL-attention block?' mask (len nL), or None if the host doesn't say.
    Handles HF `layer_types` (Qwen-3.6 full/linear, Gemma-4 full/sliding) and Nemotron-H's
    `hybrid_override_pattern` ('*'=attention, 'M'=Mamba-2, '-'=MLP)."""
    lt = getattr(config, "layer_types", None)
    if isinstance(lt, (list, tuple)) and len(lt) == nL:
        return [("attention" in str(t).lower()) and ("linear" not in str(t).lower())
                and ("sliding" not in str(t).lower()) for t in lt]
    pat = getattr(config, "hybrid_override_pattern", None)
    if isinstance(pat, str) and len(pat) == nL:
        return [c == "*" for c in pat]
    return None


def _snap_to_attention(layer, mask):
    """Snap `layer` to the nearest full-attention block (prefer at-or-below, else above)."""
    if mask is None or not any(mask):
        return layer, False
    if mask[layer]:
        return layer, False
    below = [i for i in range(layer, -1, -1) if mask[i]]
    if below:
        return below[0], True
    above = [i for i in range(layer, len(mask)) if mask[i]]
    return (above[0], True) if above else (layer, False)


# ============================================================ config loading
def load_woven_config(woven_config) -> dict:
    """Accept a dict, a path to one of runs/woven_config_*.json, or None. The proven configs carry
    differing schemas across the swarm; this normalizes the keys graft cares about."""
    if woven_config is None:
        return {}
    if isinstance(woven_config, dict):
        cfg = woven_config
    else:
        with open(woven_config) as f:
            cfg = json.load(f)
    out = dict(cfg)
    # normalize the heterogeneous schemas the 7 swarm runs wrote
    out["mid_layer"] = cfg.get("mid_layer")
    out["inject_layer"] = cfg.get("inject_layer")
    out["K"] = cfg.get("K", cfg.get("K_candidate_dim"))
    out["hidden_dim"] = cfg.get("hidden_dim", cfg.get("hidden_size"))
    return out


# ============================================================ THE graft
def graft_organ(hf_model, organ=None, woven_config=None, *, K=None, alpha=None,
                mid_layer=None, inject_layer=None, dctx=256, alpha_dp=384, alpha_heads=6,
                gamma_hidden=256, organ_kwargs=None, snap_attention=True, freeze_organ=True,
                device=None, dtype=None):
    """Splice `organ` into `hf_model`'s residual stream and return a `LiveLatentWoven` (no-op at init).

    hf_model     : a HF causal-LM, raw OR already PEFT-wrapped (LoRA is the trainer's job, not graft's).
    organ        : a frozen `LatentNarrower` (the residual-stream organ slot). If None, one is built
                   to width K so the no-op@init / forward can be proven structurally.
    woven_config : a dict or a path to runs/woven_config_<tag>.json — supplies mid/inject + K when set.
    K            : per-cell candidate width. Resolution order: arg → organ.K → config → 8.
    mid/inject   : layer indices. Resolution order: arg → config → auto (nL//3, 2·nL//3). On a hybrid
                   the inject layer snaps to the nearest full-attention block when advertised.

    The returned woven model carries the no-op@init guarantee (γ's gate is zero-init ⇒ tanh(0)=0).
    This is the SAME coupling `run_glados_staged.train_live_woven` builds; both are model-agnostic
    because the decoder-layer lookup is generalized (see `find_decoder_layers`)."""
    cfg = load_woven_config(woven_config)
    layers, layer_path = find_decoder_layers(hf_model)
    nL = len(layers)
    base = _unwrap(hf_model)
    config = getattr(base, "config", None)
    D = (getattr(config, "hidden_size", None) or cfg.get("hidden_dim")
         or layers[0].named_parameters().__next__()[1].shape[-1])
    K = int(K or (organ.K if organ is not None else None) or cfg.get("K") or 8)

    mid = mid_layer if mid_layer is not None else cfg.get("mid_layer")
    inj = inject_layer if inject_layer is not None else cfg.get("inject_layer")
    if mid is None:
        mid = max(1, nL // 3)
    if inj is None:
        inj = min(nL - 1, (2 * nL) // 3)
    mid, inj = int(mid), int(inj)

    snapped = False
    if snap_attention:
        amask = attention_layer_mask(config, nL)
        inj2, snapped = _snap_to_attention(inj, amask)
        if snapped:
            inj = max(mid + 1, inj2)
    assert 0 <= mid < inj < nL, f"need 0<=mid<inject<nL, got mid={mid} inject={inj} nL={nL}"

    dev = device or next(base.parameters()).device
    if alpha is None:
        alpha = DenseLatentProjector(D, K, dctx=dctx, dp=alpha_dp, heads=alpha_heads)
    if organ is None:
        ok = dict(d=128, heads=4, n_layers=2, T=12, ds=6, mixer="ffn", monotone=True)
        ok.update(organ_kwargs or {})
        organ = LatentNarrower(K, dctx, **ok)
    if freeze_organ:
        for p in organ.parameters():
            p.requires_grad_(False)
        organ.eval()

    woven = LiveLatentWoven(hf_model, D, K, alpha, organ, mid, inj, gamma_hidden=gamma_hidden).to(dev)
    woven.alpha.float(); woven.organ.float(); woven.gamma.float()
    # the no-op@init guarantee, made explicit: γ's gate is the only thing between the organ and the
    # residual stream, and it is zero-init ⇒ tanh(0)=0 ⇒ the injection is the additive identity.
    assert float(woven.gamma.alpha.detach().abs().max()) == 0.0, "graft must hand back a zero-init (no-op) gate"
    print(f"[graft] {type(base).__name__}  layers={nL} via '{layer_path}'  hidden={D}  K={K}  "
          f"mid={mid} -> inject={inj}{' (snapped to attention)' if snapped else ''}  "
          f"trainable_organ=frozen  gate=0(no-op@init)", flush=True)
    return woven


# ============================================================ no-op@init proof (host-agnostic)
@torch.no_grad()
def noop_report(woven, input_ids, mention, attn, *, open_gate=2.0):
    """Forward the grafted host with the gate SHUT (init) then forced OPEN; returns
    (max|Δlogits|@init, max|Δlogits|@gate-open). The first MUST be ~0 (bitwise no-op), the second >0.
    `mention[B,N,T]` assigns prompt tokens to cells; `attn[B,T]` the attention mask."""
    woven.eval()
    base = woven.model(input_ids=input_ids, attention_mask=attn).logits.float()
    with woven.live(mention, attn, inject=True, capture=True):
        g0 = woven.logits(input_ids, attn).float()
    noop = float((base - g0).abs().max())
    saved = woven.gamma.alpha.data.clone()
    woven.gamma.alpha.data.fill_(open_gate)
    with woven.live(mention, attn, inject=True, capture=True):
        g1 = woven.logits(input_ids, attn).float()
    woven.gamma.alpha.data.copy_(saved)
    return noop, float((base - g1).abs().max())
