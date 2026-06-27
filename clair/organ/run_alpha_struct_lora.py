"""clair/organ/run_alpha_struct_lora.py — the BIDIRECTIONAL-LEARNING lever for the α NL gap.

The frozen-host α probe (run_alpha_struct_probe.py --nl_only) trains the CSP StructureHead on a
FROZEN OLMo-2-1B hidden: a ONE-WAY channel. α reverse-engineers a hidden that was shaped for
next-token-prediction, never for "make the relation legible". Result: NL-OOD micro-F1 0.728 (L9),
canonical ceiling 0.98 — NL is the wall.

THIS run opens the bidirectional channel: LoRA adapters on the HOST (q/k/v/o + mlp) train JOINTLY
with the α StructureHead on the NL-frontier train set. The structure-loss now backprops INTO the
host (the hidden is no longer frozen), so gradient from α's success can shape the host to EXPOSE the
relation. Everything else is held identical to the frozen NL run (same data, same head, same
wpos/wnone/lr_head/steps) so it is a clean A/B:

    NL-OOD F1: LoRA-host vs frozen-host (0.728) — how much of the 0.728→0.98 gap does LoRA close?
    Hard-negative both-exact: frozen 0.14 — does the host now expose the WHOLE relation graph?
    Per-relation NL-OOD F1 (eq/le the weak ones); canonical no-regression check.

Writes runs/alpha_struct_lora_nl.json. Reports the A/B table directly.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..oracle_readout import _mention_tensor
from .alpha_struct import StructureRack, PAIR_RELS
from .run_alpha_struct_probe import (
    K, load_csp_records, rec_view, csp_targets, f1_report, hardneg_report,
)


# ============================================================ live (LoRA-host) forward
def olmo_csp_forward(olmo, tok, rack, dev, chunk_views, L, Nmax):
    """One micro-batch through the (LoRA-adapted) host → CSP pin/pair logits over the mention grid.
    chunk_views: list of (text, mentions, n[, facts]). Returns pin_l[B,Nmax,1+K], pair_l[B,Nmax,Nmax,R],
    vmask[B,Nmax], attn[B,T]. Grad flows through the LoRA adapters (host) + the rack."""
    texts = [c[0] for c in chunk_views]
    enc = tok(texts, return_offsets_mapping=True, padding=True, return_tensors="pt")
    ids = enc["input_ids"].to(dev); attn = enc["attention_mask"].to(dev)
    T = ids.size(1); Bp = len(chunk_views)
    ment = _mention_tensor([c[1] for c in chunk_views], enc["offset_mapping"], Bp, Nmax, T, dev)
    out = olmo(input_ids=ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[L].float()                                # [B,T,D] — LoRA-shaped hidden
    denom = ment.sum(-1, keepdim=True).clamp_min(1e-6)
    vmask = (ment.sum(-1) > 0.5).float()                            # [B,Nmax]
    vmean = torch.einsum("bnt,btd->bnd", ment, h) / denom           # [B,Nmax,D]
    attnf = attn.float()
    feat = rack.featurize(vmean, h, attnf)
    pin_l, pair_l = rack.csp(feat)
    return pin_l, pair_l, vmask, attnf


def eval_views(olmo, tok, rack, dev, views, L, Nmax, micro_bs=12):
    """No-grad eval over `views` → padded [R,Nmax,...] logit/vmask/n tensors for f1_report/hardneg."""
    R = len(views)
    pin_out = torch.zeros(R, Nmax, 1 + K)
    pair_out = torch.zeros(R, Nmax, Nmax, len(PAIR_RELS))
    vmask_out = torch.zeros(R, Nmax)
    n_out = torch.tensor([v[2] for v in views])
    olmo.eval(); rack.eval()
    with torch.no_grad():
        for i in range(0, R, micro_bs):
            sub = views[i:i + micro_bs]
            pin_l, pair_l, vmask, _ = olmo_csp_forward(olmo, tok, rack, dev, sub, L, Nmax)
            b = len(sub)
            pin_out[i:i + b] = pin_l.float().cpu()
            pair_out[i:i + b] = pair_l.float().cpu()
            vmask_out[i:i + b] = vmask.float().cpu()
    return pin_out, pair_out, vmask_out, n_out


# ============================================================ joint LoRA + head training
def train_lora_csp(olmo, tok, rack, dev, train_views, pin_tgt, pair_tgt, Nmax, lora_params,
                   steps, lr_head, lr_lora, bs, micro_bs, wpos, wnone, L, log=print):
    """Train the LoRA host-adapters + CSP head jointly on the NL-frontier train views. Same
    soundness-asymmetric weighting (wpos>wnone) and same head config as the frozen run; the only
    change is the host is no longer frozen (LoRA params train alongside)."""
    head_params = list(rack.encoder.parameters()) + list(rack.heads["csp"].parameters())
    opt = torch.optim.AdamW(
        [{"params": head_params, "lr": lr_head, "weight_decay": 0.0},
         {"params": lora_params, "lr": lr_lora, "weight_decay": 0.0}],
        betas=(0.9, 0.95),
    )
    pin_w = torch.full((1 + K,), wpos, device=dev); pin_w[0] = wnone
    pair_w = torch.full((len(PAIR_RELS),), wpos, device=dev); pair_w[0] = wnone
    rng = np.random.default_rng(0)
    olmo.train(); rack.train()
    idx_all = list(range(len(train_views)))
    t0 = time.time()
    for s in range(1, steps + 1):
        sub = [int(x) for x in rng.choice(idx_all, size=min(bs, len(idx_all)), replace=False)]
        opt.zero_grad()
        n_micro = (len(sub) + micro_bs - 1) // micro_bs
        run_pin = run_pair = 0.0
        for mi in range(0, len(sub), micro_bs):
            msub = sub[mi:mi + micro_bs]
            mviews = [train_views[r] for r in msub]
            pin_l, pair_l, vmask, _ = olmo_csp_forward(olmo, tok, rack, dev, mviews, L, Nmax)
            B, Nm, _ = pin_l.shape
            pt = pin_tgt[msub].to(dev); prt = pair_tgt[msub].to(dev)
            cellm = vmask.bool()
            lp = F.cross_entropy(pin_l[cellm], pt[cellm], weight=pin_w)
            pm = (vmask[:, :, None] * vmask[:, None, :]).bool()
            eye = torch.eye(Nm, device=dev, dtype=torch.bool)[None].expand(B, Nm, Nm)
            pm = pm & ~eye
            lpr = F.cross_entropy(pair_l[pm], prt[pm], weight=pair_w)
            loss = (lp + lpr) / n_micro
            loss.backward()
            run_pin += lp.item() / n_micro; run_pair += lpr.item() / n_micro
        opt.step()
        if s % max(1, steps // 10) == 0 or s == 1:
            log(f"    [lora-csp L{L}] step {s:4d}  pin {run_pin:.3f}  pair {run_pair:.3f}  "
                f"{time.time()-t0:.0f}s")
    return rack


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--layer", type=int, default=9, help="host layer the CSP head reads (frozen best=9)")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr_head", type=float, default=1.5e-3, help="identical to the frozen NL run")
    ap.add_argument("--lr_lora", type=float, default=2e-4)
    ap.add_argument("--wpos", type=float, default=2.0)
    ap.add_argument("--wnone", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=48, help="effective batch (matches frozen train_csp)")
    ap.add_argument("--micro_bs", type=int, default=8, help="grad-accum micro-batch (memory)")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--nl_train", required=True)
    ap.add_argument("--nl_ood", required=True)
    ap.add_argument("--out", default="runs/alpha_struct_lora_nl.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    L = a.layer

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    print(f"loading {a.model} on {dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    olmo = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev)
    D = olmo.config.hidden_size
    cfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
                     bias="none", task_type="CAUSAL_LM")
    olmo = get_peft_model(olmo, cfg)
    if not a.no_grad_ckpt:
        olmo.gradient_checkpointing_enable()
        olmo.enable_input_require_grads()
    lora_params = [p for _, p in olmo.named_parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in lora_params)
    print(f"OLMo hidden {D}, {olmo.config.num_hidden_layers} layers; LoRA trainable params {n_lora:,} "
          f"(r={a.lora_r}); reading layer {L}", flush=True)

    # ---- data: IDENTICAL to the frozen NL run (same train/ood jsonl, NL view) ----
    tr = load_csp_records(a.nl_train)
    od = load_csp_records(a.nl_ood)
    tr_v = [rec_view(r, "nl") for r in tr]
    od_v = [rec_view(r, "nl") for r in od]
    Nmax = max(max(v[2] for v in tr_v), max(v[2] for v in od_v))
    print(f"NL-frontier: {len(tr_v)} train, {len(od_v)} OOD records (binary+pin); Nmax {Nmax}",
          flush=True)
    pin_t, pair_t = csp_targets(tr_v, Nmax, "cpu")               # targets on CPU; sliced per micro

    rack = StructureRack(D, K).to(dev)
    rack.encoder.float(); rack.heads["csp"].float()

    print("\n== JOINT LoRA-host + α-StructureHead training (NL-frontier) ==", flush=True)
    train_lora_csp(olmo, tok, rack, dev, tr_v, pin_t, pair_t, Nmax, lora_params,
                   steps=a.steps, lr_head=a.lr_head, lr_lora=a.lr_lora, bs=a.bs,
                   micro_bs=a.micro_bs, wpos=a.wpos, wnone=a.wnone, L=L)

    # ---- NL-OOD eval (the A/B measurement) ----
    print("\n== NL-OOD eval (LoRA-host) ==", flush=True)
    pin_l, pair_l, vmask, n_arr = eval_views(olmo, tok, rack, dev, od_v, L, Nmax, a.micro_bs)
    idx = list(range(len(od_v)))
    nl_rep = f1_report(rack, None, pin_l, pair_l, vmask, n_arr, od_v, idx)
    nl_rep["hardneg"] = hardneg_report(pin_l, pair_l, vmask, n_arr, od_v, od)
    print(f"   NL-OOD micro-F1 {nl_rep['micro']['F1']:.4f}  exact {nl_rep['exact_factor_graph_match']:.4f}  "
          f"hardneg both-exact {nl_rep['hardneg']['both_exact_rate']:.4f} "
          f"(disc {nl_rep['hardneg']['discriminated_rate']:.4f}, n={nl_rep['hardneg']['n_pairs']})",
          flush=True)

    # ---- canonical no-regression check: canonical renders of the SAME OOD structures ----
    print("\n== canonical no-regression check (canonical render of OOD structures) ==", flush=True)
    can_v = [rec_view(r, "canonical") for r in od]
    cpin, cpair, cvmask, cn = eval_views(olmo, tok, rack, dev, can_v, L, Nmax, a.micro_bs)
    can_rep = f1_report(rack, None, cpin, cpair, cvmask, cn, can_v, list(range(len(can_v))))
    print(f"   canonical-OOD micro-F1 {can_rep['micro']['F1']:.4f}  "
          f"exact {can_rep['exact_factor_graph_match']:.4f}", flush=True)

    FROZEN = {"nl_ood_microF1": 0.7277, "nl_ood_exact": 0.1438, "hardneg_both_exact": 0.1444,
              "per_rel_F1": {"pin": 0.6757, "eq": 0.538, "neq": 0.8115, "lt": 0.677, "le": 0.5399},
              "canonical_ceiling": 0.98}
    out = {
        "model": a.model, "layer": L, "lever": "LoRA host-adaptation (bidirectional)",
        "lora": {"r": a.lora_r, "alpha": a.lora_alpha, "dropout": a.lora_dropout,
                 "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                 "trainable_params": int(n_lora), "grad_ckpt": not a.no_grad_ckpt},
        "train_cfg": {"steps": a.steps, "lr_head": a.lr_head, "lr_lora": a.lr_lora,
                      "wpos": a.wpos, "wnone": a.wnone, "bs": a.bs, "micro_bs": a.micro_bs},
        "data": {"nl_train": a.nl_train, "nl_ood": a.nl_ood, "n_train": len(tr_v), "n_ood": len(od_v)},
        "nl_ood": nl_rep,
        "canonical_check": can_rep,
        "frozen_baseline": FROZEN,
        "ab_delta": {
            "nl_ood_microF1": round(nl_rep["micro"]["F1"] - FROZEN["nl_ood_microF1"], 4),
            "nl_ood_exact": round(nl_rep["exact_factor_graph_match"] - FROZEN["nl_ood_exact"], 4),
            "hardneg_both_exact": round(nl_rep["hardneg"]["both_exact_rate"]
                                        - FROZEN["hardneg_both_exact"], 4),
            "gap_closed_frac": round((nl_rep["micro"]["F1"] - FROZEN["nl_ood_microF1"])
                                     / (FROZEN["canonical_ceiling"] - FROZEN["nl_ood_microF1"]), 4),
        },
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n==== A/B VERDICT (LoRA-host vs frozen-host) ====", flush=True)
    print(f"  NL-OOD micro-F1 : frozen 0.728  →  LoRA {nl_rep['micro']['F1']:.3f}  "
          f"(Δ {out['ab_delta']['nl_ood_microF1']:+.3f}; closes {out['ab_delta']['gap_closed_frac']*100:.0f}% "
          f"of the 0.728→0.98 gap)", flush=True)
    print(f"  exact-graph     : frozen 0.144  →  LoRA {nl_rep['exact_factor_graph_match']:.3f}  "
          f"(Δ {out['ab_delta']['nl_ood_exact']:+.3f})", flush=True)
    print(f"  hardneg both-ex : frozen 0.144  →  LoRA {nl_rep['hardneg']['both_exact_rate']:.3f}  "
          f"(Δ {out['ab_delta']['hardneg_both_exact']:+.3f})", flush=True)
    print(f"  per-rel F1      : "
          + "  ".join(f"{r} {nl_rep[r]['F1']:.3f}(frz {FROZEN['per_rel_F1'][r]:.3f})"
                      for r in ["pin", "eq", "neq", "lt", "le"]), flush=True)
    print(f"  canonical-OOD   : LoRA {can_rep['micro']['F1']:.3f} (no-regression vs ceiling 0.98)",
          flush=True)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
