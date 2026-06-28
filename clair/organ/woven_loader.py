"""clair/organ/woven_loader.py — checkpoint-kind-aware reconstruction for the eval-suite arbiter.

eval_suite.save_woven/load_woven only round-trips the legacy LiveLatentWoven (kind="live"): it drops the
ALPHA_STRUCT heads (rack / cand / fb_adapter) and the MULTIFACULTY heads (rack / cand / flow_gate). The
two REAL trained models, though, are saved by their own drivers in different blob shapes:

  * ALPHA_STRUCT  (runs/alpha_struct_woven.pt, run_alpha_struct_gate.py) — a COMPLETE trainable-delta blob
        {kind:"alpha_struct", base_id, cfg{D,K,use_lora,mid_layer,inject_layer,rich},
         rack, cand, fb_adapter, gamma, lora}
    cfg OMITS alpha_dp / alpha_heads / gamma_hidden / cand_hidden / lora_r — they are inferred from the
    stored tensor shapes (alpha_heads falls back to the weave default 6, the only field not shape-derivable).

  * MULTIFACULTY  (runs/multifaculty_full.pt, run_multifaculty.py) — a FULL model.state_dict() (base + LoRA
        + rack/cand/gamma/flow_gate) plus args{model, lora_r, alpha_dp, alpha_heads, gamma_hidden, mid_layer,
        inject_layer, core_ckpt, ...} and flow_gates. Everything needed is in args; D/nL come from the base.

This module reconstructs AlphaStructWoven / MultiFacultyWoven on top of a named base (HF id) and exposes
detect_kind() for the --woven_kind auto path. The build halves (_build_*) are factored so the CPU
structural smoke can exercise the SAME reconstruction logic on a tiny local base with no download.
"""
from __future__ import annotations

import torch

from .. import run_glados_staged as G

_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ============================================================================== kind detection
def detect_kind(blob) -> str:
    """Detect the woven checkpoint kind from a loaded blob (the --woven_kind auto path).
       alpha_struct : the run_alpha_struct_gate delta blob (rack + fb_adapter present, or kind tag).
       multifaculty : a full model.state_dict() blob carrying the flow_gate / flow_gates.
       live         : the legacy eval_suite.save_woven blob (kind in cfg, default 'live')."""
    if not isinstance(blob, dict):
        return "unknown"
    if blob.get("kind") == "alpha_struct" or ("rack" in blob and "fb_adapter" in blob):
        return "alpha_struct"
    sd = blob.get("state_dict")
    if isinstance(sd, dict) and ("flow_gates" in blob or "flow_gate" in sd
                                 or any(k.endswith("flow_gate") for k in sd)):
        return "multifaculty"
    return blob.get("kind") or "live"


def _tok(model_id):
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(model_id)
    t.padding_side = "right"
    if t.pad_token is None:
        t.pad_token = t.eos_token
    return t


def _infer_heads(dp, want=6):
    """alpha_heads is the one rack field not derivable from tensor shapes (it only reshapes attention, not
    params). Prefer the weave default (6) when it divides dp; else fall back to the largest divisor <= 8."""
    if dp % want == 0:
        return want
    for h in (8, 7, 5, 4, 3, 2, 1):
        if dp % h == 0:
            return h
    return 1


def _infer_lora_r(lora_sd):
    for k, v in (lora_sd or {}).items():
        if "lora_A" in k:
            return int(v.shape[0])
    return 16


# ============================================================================== ALPHA_STRUCT
def _build_alpha_struct(peft_model, D, K, *, dp, heads, gamma_hidden, cand_hidden, mid_layer,
                        inject_layer, rich, core_ckpt, use_core, dev):
    """Build a fresh AlphaStructWoven (random heads) over an already-wrapped peft/base — shared by the HF
    loader and the tiny-CPU smoke so both test the same reconstruction."""
    from .alpha_struct import StructureRack
    from .bank_woven import AlphaStructWoven, AlphaStructComposerOrgan
    rack = StructureRack(D, K, dp=dp, heads=heads)
    composer = AlphaStructComposerOrgan(dev="cpu", core_ckpt=core_ckpt, use_core=use_core)
    model = AlphaStructWoven(peft_model, D, K, rack, composer, mid_layer, inject_layer,
                             gamma_hidden=gamma_hidden, rich=rich, cand_hidden=cand_hidden).to(dev)
    model.alpha.float(); model.cand.float(); model.gamma.float()
    return model


def _load_alpha_struct_weights(model, blob):
    model.alpha.load_state_dict(blob["rack"]); model.alpha.float()
    model.cand.load_state_dict(blob["cand"]); model.cand.float()
    model.fb_adapter.load_state_dict(blob["fb_adapter"])
    model.gamma.load_state_dict(blob["gamma"]); model.gamma.float()
    model.eval()
    return model


def _alpha_struct_cfg(blob):
    """Resolve the AlphaStructWoven build config: cfg fields when present, else shape-inferred."""
    cfg = blob["cfg"]
    D, K = int(cfg["D"]), int(cfg["K"])
    dp = int(cfg.get("alpha_dp") or blob["rack"]["encoder.reader.wq.weight"].shape[0])
    heads = int(cfg.get("alpha_heads") or _infer_heads(dp))
    gamma_hidden = int(cfg.get("gamma_hidden") or blob["gamma"]["proj.0.weight"].shape[0])
    cand_hidden = int(cfg.get("cand_hidden") or blob["cand"]["0.weight"].shape[0])
    use_lora = bool(cfg.get("use_lora", blob.get("lora") is not None))
    return dict(D=D, K=K, dp=dp, heads=heads, gamma_hidden=gamma_hidden, cand_hidden=cand_hidden,
                mid_layer=int(cfg["mid_layer"]), inject_layer=int(cfg["inject_layer"]),
                rich=bool(cfg.get("rich", True)), use_lora=use_lora,
                core_ckpt=cfg.get("core_ckpt", "runs/general_organ_full.pt"))


def load_alpha_struct(blob, dev, base_id=None):
    """Reconstruct the ALPHA_STRUCT woven over a named HF base. Returns (model, tok)."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    c = _alpha_struct_cfg(blob)
    mid = base_id or blob["base_id"]
    tok = _tok(mid)
    base = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to(dev).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    if c["use_lora"] and blob.get("lora") is not None:
        r = _infer_lora_r(blob["lora"])
        lconf = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, bias="none",
                           target_modules=_LORA_TARGETS, task_type="CAUSAL_LM")
        peft_model = get_peft_model(base, lconf)
        set_peft_model_state_dict(peft_model, blob["lora"])
    else:
        peft_model = base
    model = _build_alpha_struct(peft_model, c["D"], c["K"], dp=c["dp"], heads=c["heads"],
                                gamma_hidden=c["gamma_hidden"], cand_hidden=c["cand_hidden"],
                                mid_layer=c["mid_layer"], inject_layer=c["inject_layer"], rich=c["rich"],
                                core_ckpt=c["core_ckpt"], use_core=True, dev=dev)
    _load_alpha_struct_weights(model, blob)
    print(f"  loaded ALPHA_STRUCT woven  base={mid}  D={c['D']} K={c['K']} dp={c['dp']} heads={c['heads']} "
          f"lora={c['use_lora']}  faculties={model.composer.faculties()}", flush=True)
    return model, tok


# ============================================================================== MULTIFACULTY
def _build_multifaculty(peft_model, D, K, *, dp, heads, gamma_hidden, mid_layer, inject_layer,
                        core_ckpt, use_core, ising_restarts, ising_steps, dev):
    from .alpha_struct import StructureRack
    from .multifaculty import MultiFacultyWoven, MultiFacultyComposerOrgan
    from .faculty import LIVE_FACULTIES
    rack = StructureRack(D, K, faculties=LIVE_FACULTIES, dp=dp, heads=heads)
    composer = MultiFacultyComposerOrgan(dev="cpu", core_ckpt=core_ckpt, use_core=use_core,
                                         ising_restarts=ising_restarts, ising_steps=ising_steps)
    model = MultiFacultyWoven(peft_model, D, K, rack, composer, mid_layer, inject_layer,
                              gamma_hidden=gamma_hidden).to(dev)
    model.alpha.float(); model.cand.float(); model.gamma.float()
    return model


def load_multifaculty(blob, dev, base_id=None):
    """Reconstruct the MULTIFACULTY woven from its full state_dict + args over a named HF base.
    Returns (model, tok)."""
    from transformers import AutoModelForCausalLM, AutoConfig
    from peft import LoraConfig, get_peft_model
    args = blob["args"]; sd = blob["state_dict"]
    mid = base_id or args["model"]
    cfg = AutoConfig.from_pretrained(mid)
    D = cfg.hidden_size
    nL = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "num_layers", None)
    K = G.K
    tok = _tok(mid)
    olmo = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to(dev).eval()
    for p in olmo.parameters():
        p.requires_grad_(False)
    r = int(args["lora_r"])
    lconf = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, bias="none",
                       target_modules=_LORA_TARGETS, task_type="CAUSAL_LM")
    peft_model = get_peft_model(olmo, lconf)
    mid_l = min(int(args["mid_layer"]), nL - 1); inj_l = min(int(args["inject_layer"]), nL - 1)
    model = _build_multifaculty(peft_model, D, K, dp=int(args["alpha_dp"]), heads=int(args["alpha_heads"]),
                                gamma_hidden=int(args["gamma_hidden"]), mid_layer=mid_l, inject_layer=inj_l,
                                core_ckpt=args.get("core_ckpt", "runs/general_organ_full.pt"), use_core=True,
                                ising_restarts=int(args.get("ising_restarts", 32)),
                                ising_steps=int(args.get("ising_steps", 100)), dev=dev)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    miss = [k for k in missing if not k.startswith("model.")]   # base buffers (rope) may be non-persistent
    if miss:
        print(f"  [multifaculty] WARN: {len(miss)} non-base keys missing on load (e.g. {miss[:4]})", flush=True)
    model.eval()
    print(f"  loaded MULTIFACULTY woven  base={mid}  D={D} K={K}  faculties={model.composer.faculties()}  "
          f"router/flow heads restored", flush=True)
    return model, tok


# ============================================================================== CPU tiny base (smoke)
def tiny_peft_base(D=64, nL=4, V=64, lora_targets=("q_proj", "v_proj"), lora_r=4):
    """A tiny random Llama wrapped in LoRA — the no-download stand-in base for the CPU structural smoke
    (eval_suite.woven_loader_smoke). Returns (peft_model, D, nL)."""
    from transformers import LlamaConfig, LlamaForCausalLM
    from peft import LoraConfig, get_peft_model
    cfg = LlamaConfig(hidden_size=D, intermediate_size=2 * D, num_hidden_layers=nL,
                      num_attention_heads=4, num_key_value_heads=4, vocab_size=V, max_position_embeddings=64)
    base = LlamaForCausalLM(cfg)
    peft = get_peft_model(base, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0,
                                           target_modules=list(lora_targets), task_type="CAUSAL_LM"))
    return peft, D, nL


# Re-export the build halves so the CPU smoke reconstructs through the SAME logic the HF loaders use.
build_alpha_struct = _build_alpha_struct
load_alpha_struct_weights = _load_alpha_struct_weights
build_multifaculty = _build_multifaculty
