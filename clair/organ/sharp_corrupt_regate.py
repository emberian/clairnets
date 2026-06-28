"""clair/organ/sharp_corrupt_regate.py — reload the trained ALPHA_STRUCT woven + a SHARPER causal test.

The acceptance-gate's α-INPUT-CORRUPT arm (alpha_struct_diag.run_gate) deranges α's input WITHIN
(relation, n) groups and applies it only CHUNK-LOCALLY (falls back to self when the partner is not in the
same bs-batch). For OOD chains whose same-shape instances often share the determined answer, that
corruption barely changes the target — so collapse can be under-measured. This driver runs the cleaner
causal test: pair each determined instance i with a partner j of the SAME (relation, n) but a DIFFERENT
true answer, swap ONLY α's stream (alpha_prompt / alpha_mentions) to j's, keep i's gen-stream + query +
gold, and measure accuracy. If α DRIVES correctness, accuracy must collapse toward NO_STRUCT.

  python -m clair.organ.sharp_corrupt_regate --ckpt runs/alpha_struct_woven.pt \
      --base allenai/OLMo-2-0425-1B --regime hard --eval_per_rung 100 --eval_seed 999
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import torch


def reload_model(ckpt, base, dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from .. import run_glados_staged as G
    from .alpha_struct import StructureRack
    from .bank_woven import AlphaStructWoven, AlphaStructComposerOrgan

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = blob["cfg"]
    D, K = int(cfg["D"]), int(cfg["K"])
    use_lora = bool(cfg.get("use_lora", True))
    mid, inj = int(cfg["mid_layer"]), int(cfg["inject_layer"])
    rich = bool(cfg.get("rich", True))

    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    olmo = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to(dev).eval()
    for p in olmo.parameters():
        p.requires_grad_(False)
    if use_lora:
        lconf = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                           "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
        peft_model = get_peft_model(olmo, lconf)
        set_peft_model_state_dict(peft_model, blob["lora"])
    else:
        peft_model = olmo
    rack = StructureRack(D, K, dp=384, heads=6)
    composer = AlphaStructComposerOrgan(dev="cpu", core_ckpt="runs/general_organ_full.pt", use_core=True)
    model = AlphaStructWoven(peft_model, D, K, rack, composer, mid, inj, gamma_hidden=256, rich=rich).to(dev)
    model.alpha.float(); model.cand.float(); model.gamma.float()
    model.alpha.load_state_dict(blob["rack"])
    model.cand.load_state_dict(blob["cand"])
    model.fb_adapter.load_state_dict(blob["fb_adapter"])
    model.gamma.load_state_dict(blob["gamma"])
    model.eval()
    return model, tok


def answer_differing_perm(recs, rng):
    """perm[i]=j with relation[j]==relation[i], n[j]==n[i], gold_idx[j]!=gold_idx[i] (so α still applies
    but the FAITHFUL compile of j yields a different query value). corrupted[i]=False when no such j."""
    groups = {}
    for i, r in enumerate(recs):
        groups.setdefault((r["relation"], r["n"]), []).append(i)
    perm = list(range(len(recs)))
    corrupted = [False] * len(recs)
    for idxs in groups.values():
        for i in idxs:
            cands = [j for j in idxs if recs[j]["gold_idx"] != recs[i]["gold_idx"]]
            if cands:
                perm[i] = int(rng.choice(cands)); corrupted[i] = True
    return perm, corrupted


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/alpha_struct_woven.pt")
    ap.add_argument("--base", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--regime", default="hard")
    ap.add_argument("--eval_per_rung", type=int, default=100)
    ap.add_argument("--eval_seed", type=int, default=999)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--json_out", default="runs/alpha_struct_sharp_corrupt.json")
    a = ap.parse_args()

    from .. import run_glados_staged as G
    from .alpha_struct_diag import capture_surv
    from .shuffled_diag import score_preds
    dev = G.device()
    K = G.K
    model, tok = reload_model(a.ckpt, a.base, dev)

    rungs, split = G.LIVE_REGIMES[G._REGIME_ALIAS.get(a.regime, a.regime)]
    rng = np.random.default_rng(a.eval_seed)
    recs = G.build_live_pool(rng, rungs, split, a.eval_per_rung, det_only=True, two_stream=True)
    true_gold = [r["gold_idx"] for r in recs]
    perm, corrupted = answer_differing_perm(recs, np.random.default_rng(a.eval_seed + 1))
    cidx = [i for i in range(len(recs)) if corrupted[i]]
    print(f"[sharp] pool={len(recs)}  answer-differing-informative={len(cidx)} over {rungs}", flush=True)

    # corrupt recs: swap ONLY α's stream (alpha_prompt / alpha_mentions) to the partner's; keep i's
    # gen-stream prompt/mentions, query, gold, csp, n, vnames.
    corrupt_recs = []
    for i, r in enumerate(recs):
        cr = copy.copy(r)
        j = perm[i]
        cr["alpha_prompt"] = recs[j].get("alpha_prompt", recs[j]["prompt"])
        cr["alpha_mentions"] = recs[j].get("alpha_mentions", recs[j]["mentions"])
        corrupt_recs.append(cr)

    def cap_and_score(src_recs, inject):
        preds = [None] * len(recs)
        for s in range(0, len(recs), a.bs):
            chunk = list(range(s, min(s + a.bs, len(recs))))
            hr = [src_recs[c] for c in chunk]
            Nmax = max(recs[c]["n"] for c in chunk)
            if inject:
                surv_b, _ = capture_surv(model, hr, tok, dev, True)
                surv = torch.zeros(len(chunk), Nmax, K, device=dev)
                for li, c in enumerate(chunk):
                    surv[li, : recs[c]["n"]] = surv_b[li, : recs[c]["n"]]
                pr = score_preds(model, [recs[c] for c in chunk], surv, tok, dev, inject=True)
            else:
                surv = torch.zeros(len(chunk), Nmax, K, device=dev)
                pr = score_preds(model, [recs[c] for c in chunk], surv, tok, dev, inject=False)
            for li, c in enumerate(chunk):
                preds[c] = pr[li]
        return preds

    p_true = cap_and_score(recs, True)
    p_corr = cap_and_score(corrupt_recs, True)
    p_none = cap_and_score(recs, False)

    def acc(preds, ii):
        return sum(preds[i] == true_gold[i] for i in ii) / max(1, len(ii))

    idx = list(range(len(recs)))
    acc_true = acc(p_true, idx); acc_true_ci = acc(p_true, cidx)
    acc_corr = acc(p_corr, cidx); acc_none = acc(p_none, idx); acc_none_ci = acc(p_none, cidx)
    out = {
        "n": len(recs), "n_answer_differing": len(cidx),
        "ALPHA_STRUCT_true_all": acc_true, "ALPHA_STRUCT_true_on_informative": acc_true_ci,
        "ANSWER_DIFFERING_CORRUPT": acc_corr,
        "NO_STRUCT_all": acc_none, "NO_STRUCT_on_informative": acc_none_ci,
        "collapse_sharp": acc_true_ci - acc_corr,
        "drops_to_nostruct": (acc_corr - acc_none_ci),
    }
    print("\n" + "=" * 72, flush=True)
    print("SHARP α-INPUT-CORRUPT (answer-differing partner; the clean causal test)", flush=True)
    print("=" * 72, flush=True)
    print(f"  informative (answer-differing) instances: {len(cidx)} / {len(recs)}", flush=True)
    print(f"  ALPHA_STRUCT(true)  on informative : {acc_true_ci*100:5.1f}%", flush=True)
    print(f"  ANSWER-DIFFERING-CORRUPT           : {acc_corr*100:5.1f}%", flush=True)
    print(f"  NO_STRUCT           on informative : {acc_none_ci*100:5.1f}%", flush=True)
    print(f"  >> sharp collapse (true-corrupt)   : {out['collapse_sharp']*100:+5.1f} pts", flush=True)
    print(f"  >> corrupt vs nostruct (informative): {out['drops_to_nostruct']*100:+5.1f} pts "
          f"(<=+5 ⇒ collapses to the floor ⇒ α DRIVES)", flush=True)
    print("=" * 72 + "\n", flush=True)
    with open(a.json_out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[sharp] wrote {a.json_out}", flush=True)


if __name__ == "__main__":
    main()
