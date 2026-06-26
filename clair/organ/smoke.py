"""clair/organ/smoke.py — the ONE-COMMAND end-to-end GLaDOS smoke (pretrain → graft → weave → eval).

Exercises the WHOLE canonical pipeline at tiny sizes on a REAL host LM, so the front-door flow is
proven coherent in a single run:

  [1] pretrain_organ  — a tiny union narrow organ (dominate-dedₚ), saved + reloadable;
  [2] graft_organ     — splice it into the host (proves the model-agnostic graft + no-op@init on a
                        real forward of the chosen base);
  [3] weave           — train the live-latent-α woven readout (warmup α+organ, freeze, LoRA+α+γ on
                        LM CE; two-stream + J0), with the causal-control readout;
  [4] run_eval_suite  — the arbiter at a tiny sample (Tier-1 CSP + controls + a light std set).

NEEDS A GPU (it loads a real HF base + LoRA + does forward/backward). Queue it for box 2 AFTER the
calibration study finishes — do NOT launch it while box 2 is busy.

  python -m clair.organ.smoke --base allenai/OLMo-2-0425-1B
  python -m clair.organ.smoke --base EleutherAI/pythia-160m --no_eval   # fastest host
"""
from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description="end-to-end GLaDOS smoke: pretrain → graft → weave → eval")
    ap.add_argument("--base", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--regime", default="small")
    ap.add_argument("--no_eval", action="store_true", help="skip the eval-suite tier (graft+weave only)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from .. import run_glados_staged as G
    from . import train as T
    from . import graft as GR
    dev = G.device()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    print(f"==== GLaDOS end-to-end smoke  base={a.base}  dev={dev} ====", flush=True)

    # [1] pretrain a tiny organ + reload it (load_core_organ format)
    tmp_organ = os.path.join(tempfile.gettempdir(), "glados_smoke_organ.pt")
    T.pretrain_organ(out=tmp_organ, steps=40, target=1.2e5, pool=48, R=6, seed=a.seed, dev=dev)
    from .bank import load_core_organ
    _organ, meta = load_core_organ(tmp_organ, dev="cpu")
    print(f"  [1] organ trained+reloaded: K={meta['D_MAX']} params={meta['params']:,}", flush=True)

    # [2] graft proof on a real forward of the host (model-agnostic + no-op@init)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    host = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16).to(dev).eval()
    for p in host.parameters():
        p.requires_grad_(False)
    woven = GR.graft_organ(host, woven_config=None, K=int(meta["D_MAX"]))
    enc = tok("Cells A, B, C, D. What color is A?", return_tensors="pt").to(dev)
    ids, attn = enc["input_ids"], torch.ones_like(enc["input_ids"])
    B, Tlen = ids.shape
    ncells = 4
    mention = torch.zeros(B, ncells, Tlen, device=dev)
    valid = attn[0].nonzero().flatten().tolist()
    for n in range(ncells):
        lo, hi = (n * len(valid)) // ncells, max((n * len(valid)) // ncells + 1, ((n + 1) * len(valid)) // ncells)
        for ti in valid[lo:hi]:
            mention[0, n, ti] = 1.0
    noop, opened = GR.noop_report(woven, ids, mention, attn)
    print(f"  [2] graft no-op@init max|Δlogits|={noop:.3e} (expect 0)  gate-open moves {opened:.3e}",
          flush=True)
    assert noop == 0.0 and opened > 0.0, "graft must be a bitwise no-op at init and move when opened"
    del woven, host
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # [3] weave the live woven readout (tiny) + the causal-control readout
    model, metrics = T.weave(a.base, regime=a.regime, two_stream=True, smoke=True, dev=dev)
    print(f"  [3] weave done: WOVEN {metrics['true']*100:.1f}%  drop {metrics['drop']*100:+.1f}pts  "
          f"gate {metrics['gate']:+.3f}", flush=True)

    # [4] arbiter at a tiny sample
    if not a.no_eval:
        tmp_ck = os.path.join(tempfile.gettempdir(), "glados_smoke_woven.pt")
        from .. import eval_suite as ES
        wcfg = {"kind": "live", "D": model.D, "K": model.K, "lora_r": 8, "gamma_hidden": 128,
                "inject_layer": model.inject_layer, "mid_layer": model.mid_layer, "dctx": 256,
                "alpha_dp": 384, "alpha_heads": 6, "organ_d": 96, "organ_heads": 4,
                "organ_layers": 2, "organ_T": 8}
        ES.save_woven(model, tmp_ck, base_id=a.base, cfg=wcfg)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from .eval import run_eval_suite
        run_eval_suite(a.base, woven_ckpt=tmp_ck, tiers=("tier1",), arms=("base", "woven", "oracle"),
                       k=(1, 4), limit=6, csp_per_rung=6, controls=True, rg_ceiling=False,
                       out=os.path.join(tempfile.gettempdir(), "glados_smoke_eval.json"), dev=dev)
    print("\n==== GLaDOS end-to-end smoke PASSED (pretrain → graft → weave"
          f"{'' if a.no_eval else ' → eval'}) ====", flush=True)


if __name__ == "__main__":
    main()
