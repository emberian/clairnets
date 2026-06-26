#!/usr/bin/env python
"""swarmC: prove the GLaDOS woven-augmentation (LiveLatentWoven generalization) RUNS on small bases.

Per model:  load -> attach the organ coupling at the residual stream -> NO-OP @ init -> forward runs.
NO TRAINING. The coupling mirrors clair.oracle_readout.LiveLatentWoven:
  * a MID decoder-layer forward hook captures the host hidden h (read-only),
  * an INJECT decoder-layer forward hook adds  gamma( organ( alpha(h) ) )  to the residual stream,
    gated by a zero-init tanh scalar (OracleGamma.alpha) => bitwise no-op at init.

alpha = clair.latent_organ.DenseLatentProjector sized to THIS model's hidden dim D.
organ = clair.latent_organ.LatentNarrower (FROZEN; the bitter-lesson latent narrower that fits the
        organ(alpha(hidden)) contract with NO explicit factors). general_organ_full.pt is loaded too
        (load-proof on the box) and supplies K = its candidate dimension (meta D_MAX) so alpha/gamma
        match the real organ's readout width.
gamma = clair.oracle_readout.OracleGamma (K->D projection, zero-init tanh gate).

The arch wrinkle handled here: LiveLatentWoven's _decoder_layers hardcodes the Llama-style
base.model.model.layers; Pythia is GPT-NeoX (gpt_neox.layers, parallel residual). find_decoder_layers
generalizes the lookup.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from clair.latent_organ import DenseLatentProjector, LatentNarrower
from clair.oracle_readout import OracleGamma
from clair.organ.bank import load_core_organ


def find_decoder_layers(model):
    """Generalize clair.oracle_readout._decoder_layers across HF decoder archs.
    Returns (layers_ModuleList, dotted_path). Handles Llama-style (OLMo-2 / SmolLM3) AND
    GPT-NeoX (Pythia) AND GPT-2-style fallbacks."""
    probes = [
        ("model.layers",    lambda m: getattr(getattr(m, "model", None), "layers", None)),
        ("gpt_neox.layers", lambda m: getattr(getattr(m, "gpt_neox", None), "layers", None)),
        ("transformer.h",   lambda m: getattr(getattr(m, "transformer", None), "h", None)),
    ]
    for path, fn in probes:
        layers = fn(model)
        if layers is not None and len(layers) > 0:
            return layers, path
    raise RuntimeError(f"could not locate decoder layers on {type(model).__name__}")


class WovenCoupling(nn.Module):
    """Standalone (no-peft, no-training) generalization of LiveLatentWoven. mid hook captures h;
    inject hook adds tanh(gamma.alpha) * gamma(organ(alpha(h))) to the residual."""

    def __init__(self, model, layers, D, K, mid_layer, inject_layer, dctx=256, theta=0.5):
        super().__init__()
        self.model = model
        self.D, self.K = D, K
        self.mid_layer, self.inject_layer = mid_layer, inject_layer
        self.theta = theta
        self.alpha = DenseLatentProjector(D, K, dctx=dctx)        # sized to THIS model's hidden dim
        self.organ = LatentNarrower(K, dctx)                     # FROZEN latent narrower
        for p in self.organ.parameters():
            p.requires_grad_(False)
        self.gamma = OracleGamma(D, K)                           # zero-init tanh gate => no-op @ init
        # per-forward injection state
        self._mention = self._attn = None
        self._inject = False
        self._h_mid = None
        self._mid_handle = layers[mid_layer].register_forward_hook(self._mid_hook)
        self._inj_handle = layers[inject_layer].register_forward_hook(self._inj_hook)

    def _mid_hook(self, module, args, output):
        self._h_mid = output[0] if isinstance(output, tuple) else output
        return output

    def _run_organ(self, hs):
        h = self._h_mid.float()
        m = self._mention.to(h.device)
        denom = m.sum(-1, keepdim=True).clamp_min(1e-6)
        v_mean = torch.einsum("bnt,btd->bnd", m, h) / denom       # [B,N,D] mention-pool cell identity
        b0, ctx = self.alpha(v_mean, h, self._attn.to(h.device))
        vmask = (m.sum(-1) > 0.5).float()                         # cells with >=1 mention token
        b, _sup = self.organ(b0, ctx, vmask)
        surv = (torch.sigmoid(b) >= self.theta).float() * vmask.unsqueeze(-1)
        return surv

    def _inj_hook(self, module, args, output):
        if not self._inject:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        surv = self._run_organ(hs)
        delta = self.gamma.delta(surv.to(hs.device), self._mention.to(hs.device))
        hs = hs + (torch.tanh(self.gamma.alpha) * delta).to(hs.dtype)
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

    @contextlib.contextmanager
    def live(self, mention, attn, inject):
        old = (self._mention, self._attn, self._inject)
        self._mention, self._attn, self._inject = mention, attn, inject
        try:
            yield
        finally:
            self._mention, self._attn, self._inject = old

    def logits(self, input_ids, attn):
        return self.model(input_ids=input_ids, attention_mask=attn).logits


def build_mention(B, N, T, attn, device):
    """Synthesize N 'cells' by splitting the real (non-pad) token span into N contiguous blocks.
    mention[B,N,T] in {0,1}; every cell gets >=1 token so vmask is all-on (the organ is exercised)."""
    mention = torch.zeros(B, N, T, device=device)
    for b in range(B):
        valid = attn[b].nonzero().flatten().tolist()
        if not valid:
            continue
        L = len(valid)
        for n in range(N):
            lo = (n * L) // N
            hi = max(lo + 1, ((n + 1) * L) // N)
            for ti in valid[lo:hi]:
                mention[b, n, ti] = 1.0
    return mention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--organ_ckpt", default="runs/general_organ_full.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ncells", type=int, default=6)
    ap.add_argument("--prompt", default="The capital of France is")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n==================== {a.tag}  ({a.model_id}) ====================", flush=True)

    # --- load-proof: the frozen general organ checkpoint, for K (candidate dim) ---------------------
    _organ_fgp, meta = load_core_organ(a.organ_ckpt, dev="cpu")
    K = int(meta["D_MAX"])
    print(f"[organ] general_organ_full.pt loaded OK (FactorGraphProposer); meta D_MAX -> K={K}", flush=True)

    # --- load the base model ------------------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(a.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model_id, torch_dtype=torch.bfloat16).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = model.config
    D = cfg.hidden_size
    layers, layer_path = find_decoder_layers(model)
    nL = len(layers)
    mid = max(1, nL // 3)
    inject = min(nL - 1, (2 * nL) // 3)
    assert mid < inject < nL, (mid, inject, nL)
    print(f"[model] {type(model).__name__}  layers={nL} via '{layer_path}'  hidden={D}  "
          f"mid={mid} inject={inject}  dtype=bf16", flush=True)

    # --- attach the coupling ------------------------------------------------------------------------
    coupling = WovenCoupling(model, layers, D, K, mid, inject).to(dev)
    print(f"[attach] WovenCoupling attached: alpha(DenseLatentProjector D={D},K={K}) + "
          f"organ(LatentNarrower, FROZEN) + gamma(OracleGamma, zero-init gate)", flush=True)

    # --- inputs -------------------------------------------------------------------------------------
    enc = tok(a.prompt, return_tensors="pt").to(dev)
    ids, attn = enc["input_ids"], enc["attention_mask"]
    B, T = ids.shape
    mention = build_mention(B, a.ncells, T, attn, dev)

    # --- forward: base (inject disabled) ------------------------------------------------------------
    with torch.no_grad():
        with coupling.live(mention, attn, inject=False):
            logits_base = coupling.logits(ids, attn).float()
        # --- no-op @ init: gate is zero-init -> bitwise identical ----------------------------------
        with coupling.live(mention, attn, inject=True):
            logits_init = coupling.logits(ids, attn).float()
        d_init = (logits_init - logits_base).abs().max().item()
        # --- gate open: force the tanh gate open -> logits MUST move -------------------------------
        with torch.no_grad():
            coupling.gamma.alpha.fill_(3.0)
        with coupling.live(mention, attn, inject=True):
            logits_open = coupling.logits(ids, attn).float()
        d_open = (logits_open - logits_base).abs().max().item()
        with torch.no_grad():
            coupling.gamma.alpha.zero_()

    gate_val = float(torch.tanh(torch.tensor(3.0)))
    noop_ok = (d_init == 0.0)
    open_ok = (d_open > 0.0)
    forward_ok = tuple(logits_base.shape) == (B, T, cfg.vocab_size)
    print(f"[forward] logits shape {tuple(logits_base.shape)} (vocab={cfg.vocab_size})  forward_runs={forward_ok}", flush=True)
    print(f"[NO-OP @init]   gate=tanh(0)=0  ->  max|delta logits| = {d_init:.3e}   bitwise_noop={noop_ok}", flush=True)
    print(f"[GATE OPEN]     gate=tanh(3)={gate_val:.4f}  ->  max|delta logits| = {d_open:.3e}   moves_logits={open_ok}", flush=True)
    verdict = noop_ok and open_ok and forward_ok
    print(f"[VERDICT] {a.tag}: load+attach+noop@init+forward+gate-open  => {'PASS' if verdict else 'FAIL'}", flush=True)

    out = a.out or f"runs/woven_config_{a.tag}.json"
    config = {
        "tag": a.tag,
        "model_id": a.model_id,
        "arch": type(model).__name__,
        "decoder_layer_path": layer_path,
        "n_layers": nL,
        "hidden_dim": D,
        "vocab_size": cfg.vocab_size,
        "K_candidate_dim": K,
        "organ_ckpt": a.organ_ckpt,
        "mid_layer": mid,
        "inject_layer": inject,
        "n_cells": a.ncells,
        "dtype": "bfloat16",
        "noop_at_init_max_abs_delta": d_init,
        "noop_at_init_bitwise": noop_ok,
        "gate_open_tanh": gate_val,
        "gate_open_max_abs_delta": d_open,
        "gate_open_moves_logits": open_ok,
        "forward_runs": forward_ok,
        "verdict": "PASS" if verdict else "FAIL",
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[config] wrote {out}", flush=True)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
