"""clair/organ/run_alpha_struct_ab.py — the ADAPTER A/B for ALPHA_STRUCT (capacity vs representation).

ONE arm per process (so each releases its OLMo before the next). Identical data/recipe/steps across arms;
the ONLY change is the host adapter:

    lora16 / lora32 / lora64 : plain LoRA at rank {16,32,64}     (capacity sweep)
    dora16                   : weight-decomposed LoRA (use_dora) at rank 16
    fullft                   : unfreeze the host — the diagnostic UPPER BOUND (host lr lowered)

Train the ALPHA_STRUCT woven on the hard CSP curriculum (regime hard = eqchain/forcedcolor, split ood —
the OOD long propagation chains, L=8/10), then run the SHUFFLED-INVERTED acceptance gate + compare_modes +
the SHARP answer-differing α-corrupt collapse on the SAME held-out determined-query pool. Per arm we record:
structure-F1, exact-factor-graph-match, the α-corrupt collapse (gate + sharp), and answer-acc.

  python -m clair.organ.run_alpha_struct_ab --arm lora16 --steps 2000 --warm_steps 300 \
      --eval_per_rung 100 --eval_seed 999 --json_out runs/ab_lora16.json
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch


ARMS = {
    # arm -> weave hyperparameter overrides (the ONLY thing that differs)
    "lora16": dict(use_lora=1, use_dora=0, full_ft=0, lora_r=16, lora_lr=2e-4),
    "lora32": dict(use_lora=1, use_dora=0, full_ft=0, lora_r=32, lora_lr=2e-4),
    "lora64": dict(use_lora=1, use_dora=0, full_ft=0, lora_r=64, lora_lr=2e-4),
    "dora16": dict(use_lora=1, use_dora=1, full_ft=0, lora_r=16, lora_lr=2e-4),
    "fullft": dict(use_lora=0, use_dora=0, full_ft=1, lora_r=0, lora_lr=1e-5),
}


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


def sharp_corrupt(model, recs, tok, dev, bs=8, seed=999):
    """The SHARP causal α-corrupt collapse (sharp_corrupt_regate logic, in-memory): pair each determined
    instance with a SAME-(relation,n) partner of a DIFFERENT true answer, swap ONLY α's stream to the
    partner's, keep i's gen-stream/query/gold. If α DRIVES correctness, acc collapses toward NO_STRUCT."""
    from .alpha_struct_diag import capture_surv
    from .sharp_corrupt_regate import answer_differing_perm
    from .shuffled_diag import score_preds
    from .. import run_glados_staged as G
    K = G.K
    true_gold = [r["gold_idx"] for r in recs]
    perm, corrupted = answer_differing_perm(recs, np.random.default_rng(seed + 1))
    cidx = [i for i in range(len(recs)) if corrupted[i]]
    corrupt_recs = []
    for i, r in enumerate(recs):
        cr = copy.copy(r); j = perm[i]
        cr["alpha_prompt"] = recs[j].get("alpha_prompt", recs[j]["prompt"])
        cr["alpha_mentions"] = recs[j].get("alpha_mentions", recs[j]["mentions"])
        corrupt_recs.append(cr)

    def cap_and_score(src_recs, inject):
        preds = [None] * len(recs)
        for s in range(0, len(recs), bs):
            chunk = list(range(s, min(s + bs, len(recs))))
            hr = [src_recs[c] for c in chunk]
            Nmax = max(recs[c]["n"] for c in chunk)
            surv = torch.zeros(len(chunk), Nmax, K, device=dev)
            if inject:
                surv_b, _ = capture_surv(model, hr, tok, dev, True)
                for li, c in enumerate(chunk):
                    surv[li, : recs[c]["n"]] = surv_b[li, : recs[c]["n"]]
            pr = score_preds(model, [recs[c] for c in chunk], surv, tok, dev, inject=inject)
            for li, c in enumerate(chunk):
                preds[c] = pr[li]
        return preds

    p_true = cap_and_score(recs, True)
    p_corr = cap_and_score(corrupt_recs, True)
    p_none = cap_and_score(recs, False)

    def acc(preds, ii):
        return sum(preds[i] == true_gold[i] for i in ii) / max(1, len(ii))

    idx = list(range(len(recs)))
    acc_true_ci = acc(p_true, cidx); acc_corr = acc(p_corr, cidx)
    acc_none_ci = acc(p_none, cidx)
    return {
        "n": len(recs), "n_answer_differing": len(cidx),
        "ALPHA_STRUCT_true_on_informative": acc_true_ci,
        "ANSWER_DIFFERING_CORRUPT": acc_corr,
        "NO_STRUCT_on_informative": acc_none_ci,
        "collapse_sharp": acc_true_ci - acc_corr,
        "drops_to_nostruct": acc_corr - acc_none_ci,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--base", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--regime", default="hard")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--warm_steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--eval_per_rung", type=int, default=100)
    ap.add_argument("--eval_seed", type=int, default=999)
    ap.add_argument("--search_K", type=int, default=8)
    ap.add_argument("--loop_T", type=int, default=3)
    ap.add_argument("--json_out", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from .. import run_glados_staged as G
    from . import train as T
    from . import alpha_struct_diag as D

    dev = G.device()
    hp = dict(ARMS[a.arm])
    print(f"[ab] arm={a.arm} hp={hp} steps={a.steps} warm={a.warm_steps} dev={dev}", flush=True)

    # ---- TRAIN (identical recipe; the only change is the adapter hp) ----
    t0 = time.time()
    model, _ = T.weave(a.base, regime=a.regime, two_stream=True, organ_mode="alpha_struct",
                       out=None, steps=a.steps, warm_steps=a.warm_steps, bs=a.bs, **hp)
    train_s = time.time() - t0
    model.eval()
    host_trainable = int(sum(p.numel() for _, p in model.model.named_parameters() if p.requires_grad))
    total_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"[ab] TRAIN done {train_s:.0f}s  host_trainable={host_trainable:,} "
          f"total_trainable={total_trainable:,}", flush=True)

    # ---- held-out determined-query eval pool (distinct seed; the OOD long chains) ----
    tok = AutoTokenizer.from_pretrained(a.base)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rungs, split = G.LIVE_REGIMES[G._REGIME_ALIAS.get(a.regime, a.regime)]
    recs = G.build_live_pool(np.random.default_rng(a.eval_seed), rungs, split, a.eval_per_rung,
                             det_only=True, two_stream=True)
    print(f"[ab] held-out pool: {len(recs)} det-query recs over {rungs} (split={split})", flush=True)

    # ---- the inverted-SHUFFLED gate (struct-F1 + exact-match + corrupt collapse + answer-acc) ----
    gate = D.run_gate(model, recs, tok, dev, bs=a.bs, two_stream=True, verbose=True)
    # ---- greedy vs search vs loop vs both ----
    modes = D.compare_modes(model, recs, tok, dev, bs=a.bs, two_stream=True,
                            search_K=a.search_K, loop_T=a.loop_T, verbose=True)
    # ---- the SHARP answer-differing α-corrupt collapse ----
    sharp = sharp_corrupt(model, recs, tok, dev, bs=a.bs, seed=a.eval_seed)
    print(f"[ab] sharp collapse={sharp['collapse_sharp']:+.3f} "
          f"(true|info {sharp['ALPHA_STRUCT_true_on_informative']:.3f} -> "
          f"corrupt {sharp['ANSWER_DIFFERING_CORRUPT']:.3f}; floor {sharp['NO_STRUCT_on_informative']:.3f})",
          flush=True)

    out = {
        "arm": a.arm, "hp": hp,
        "meta": {"base": a.base, "regime": a.regime, "rungs": list(rungs), "split": split,
                 "steps": a.steps, "warm_steps": a.warm_steps, "bs": a.bs, "n_eval": len(recs),
                 "host_trainable": host_trainable, "total_trainable": total_trainable,
                 "train_seconds": train_s, "search_K": a.search_K, "loop_T": a.loop_T},
        # the headline A/B row:
        "struct_F1": gate["struct_F1"], "exact_match": gate["exact_match"],
        "answer_acc": gate["ALPHA_STRUCT_true"], "no_struct": gate["NO_STRUCT"],
        "collapse_corrupt_gate": gate["criteria"]["collapse_corrupt"],
        "collapse_sharp": sharp["collapse_sharp"],
        "gate": _clean(gate), "compare_modes": _clean(modes), "sharp_corrupt": _clean(sharp),
    }
    import os
    os.makedirs(os.path.dirname(os.path.abspath(a.json_out)) or ".", exist_ok=True)
    with open(a.json_out, "w") as f:
        json.dump(_clean(out), f, indent=2)
    print(f"[ab] wrote {a.json_out}", flush=True)
    print(f"[ab] ARM {a.arm}: F1={out['struct_F1']:.3f} exact={out['exact_match']:.3f} "
          f"ans_acc={out['answer_acc']:.3f} sharp_collapse={out['collapse_sharp']:+.3f} "
          f"host_params={host_trainable:,}", flush=True)


if __name__ == "__main__":
    main()
