"""clair/organ/run_alpha_struct_gate.py — TRAIN the ALPHA_STRUCT woven + run the acceptance gate.

The culmination driver: one process trains the text->organ reasoner (StructureRack: α EMITS the typed
factor graph; the composer runs the certified floor + pretrained CoreNarrowOrgan on csp_α, NEVER on
rec['csp']) and then runs the SHUFFLED-INVERTED acceptance gate (alpha_struct_diag.run_gate) +
compare_modes (greedy / +search / +loop / +both) on the in-memory model.

Train and gate run in ONE process because the legacy eval_suite.save_woven/load_woven do not round-trip
the AlphaStructWoven heads (cand / fb_adapter) — so the trained model is evaluated live, then a COMPLETE
trainable-delta checkpoint (lora + rack + cand + fb_adapter + γ) is written for scp.

  python -m clair.organ.run_alpha_struct_gate --base allenai/OLMo-2-0425-1B --regime hard \
      --steps 2500 --warm_steps 400 --eval_per_rung 100 --search_K 8 --loop_T 3 \
      --out runs/alpha_struct_woven.pt --json_out runs/alpha_struct_gate.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--regime", default="hard")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--warm_steps", type=int, default=400)
    ap.add_argument("--use_lora", type=int, default=1)
    ap.add_argument("--two_stream", type=int, default=1)
    ap.add_argument("--eval_per_rung", type=int, default=100, help="held-out det-query instances per rung")
    ap.add_argument("--eval_seed", type=int, default=999, help="held-out pool seed (distinct from train)")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--search_K", type=int, default=8, help="compare_modes α-as-search width")
    ap.add_argument("--loop_T", type=int, default=3, help="compare_modes α<->organ loop steps")
    # training-time levers (default 1/1 = the validated single-shot recipe; the gate measures both arms)
    ap.add_argument("--train_search_K", type=int, default=1)
    ap.add_argument("--train_loop_T", type=int, default=1)
    ap.add_argument("--out", default="runs/alpha_struct_woven.pt")
    ap.add_argument("--json_out", default="runs/alpha_struct_gate.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from .. import run_glados_staged as G
    from . import train as T
    from . import alpha_struct_diag as D

    dev = G.device()
    two_stream = bool(a.two_stream)
    print(f"[gate-driver] base={a.base} regime={a.regime} steps={a.steps} warm={a.warm_steps} "
          f"use_lora={a.use_lora} two_stream={two_stream} dev={dev}", flush=True)

    # ---- TRAIN (organ_mode alpha_struct; out=None -> skip the lossy save_woven; we save a full blob) ----
    t0 = time.time()
    model, _ = T.weave(a.base, regime=a.regime, two_stream=two_stream, organ_mode="alpha_struct",
                       out=None, steps=a.steps, warm_steps=a.warm_steps,
                       use_lora=bool(a.use_lora),
                       search_K=a.train_search_K, search_temp=1.0, loop_T=a.train_loop_T)
    print(f"[gate-driver] TRAIN done in {time.time()-t0:.0f}s", flush=True)
    model.eval()

    # ---- HELD-OUT determined-query eval pool (distinct seed from weave's train/eval pools) ----
    tok = AutoTokenizer.from_pretrained(a.base)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rungs, split = G.LIVE_REGIMES[G._REGIME_ALIAS.get(a.regime, a.regime)]
    rng_ev = np.random.default_rng(a.eval_seed)
    recs = G.build_live_pool(rng_ev, rungs, split, a.eval_per_rung, det_only=True, two_stream=two_stream)
    print(f"[gate-driver] held-out gate pool: {len(recs)} det-query recs over {rungs} (split={split})",
          flush=True)

    # ---- THE GATE: the SHUFFLED-INVERTED 4-arm acceptance gate ----
    t1 = time.time()
    gate = D.run_gate(model, recs, tok, dev, bs=a.bs, two_stream=two_stream, verbose=True)
    print(f"[gate-driver] run_gate done in {time.time()-t1:.0f}s", flush=True)

    # ---- compare_modes: greedy vs +search(K) vs +loop(T) vs +both ----
    t2 = time.time()
    modes = D.compare_modes(model, recs, tok, dev, bs=a.bs, two_stream=two_stream,
                            search_K=a.search_K, loop_T=a.loop_T, verbose=True)
    print(f"[gate-driver] compare_modes done in {time.time()-t2:.0f}s", flush=True)

    # ---- save a COMPLETE trainable-delta checkpoint (round-trips the AlphaStructWoven heads) ----
    from peft import get_peft_model_state_dict
    blob = {
        "kind": "alpha_struct",
        "base_id": a.base,
        "cfg": {"D": int(model.D), "K": int(model.K), "use_lora": bool(a.use_lora),
                "mid_layer": int(model.mid_layer), "inject_layer": int(model.inject_layer),
                "rich": bool(model.rich)},
        "rack": model.alpha.state_dict(),
        "cand": model.cand.state_dict(),
        "fb_adapter": model.fb_adapter.state_dict(),
        "gamma": model.gamma.state_dict(),
        "lora": (get_peft_model_state_dict(model.model) if a.use_lora else None),
    }
    import os
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    torch.save(blob, a.out)
    print(f"[gate-driver] saved checkpoint -> {a.out}", flush=True)

    # ---- dump the diagnostic JSON (gate arms + criteria + compare_modes) ----
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, float) and (o != o):
            return None
        return o
    out_json = {"meta": {"base": a.base, "regime": a.regime, "rungs": list(rungs), "split": split,
                         "steps": a.steps, "warm_steps": a.warm_steps, "use_lora": bool(a.use_lora),
                         "n_eval": len(recs), "search_K": a.search_K, "loop_T": a.loop_T},
                "gate": _clean(gate), "compare_modes": _clean(modes)}
    with open(a.json_out, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"[gate-driver] wrote diag JSON -> {a.json_out}", flush=True)
    print(f"[gate-driver] VERDICT PASS={gate['PASS']}", flush=True)


if __name__ == "__main__":
    main()
