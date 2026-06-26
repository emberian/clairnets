"""The DECISIVE latent-vs-explicit test for design B.

Can a PURE-LATENT deductive organ — fed ONLY OLMo's dense hidden of the problem TEXT, with NO
explicit factors and NO extraction supervision — narrow to match the exact transformer dedₚ? We
measure narrowing-RECALL (fraction of dedₚ's eliminations it makes) + FALSE-ELIM, in-distribution
and OOD (held-out N + held-out phrasings), and compare against the EXPLICIT factor deductor
(clair.proposer.FactorGraphProposer, given the TRUE factors) as the upper bound.

  SMOKE:  python -m clair.run_latent --smoke
  FULL :  python -m clair.run_latent --steps 1500 --lora --train_n 4,5,6 --test_n 5,8,11

Latent organ reaches ~the explicit deductor's recall -> the constraint structure flows LATENTLY from
text through OLMo+α; design B is viable; symbolic extraction is unnecessary.
Latent organ collapses (low recall / high false-elim) -> the dense latent does not carry enough
constraint structure for correct narrowing; explicit conveyance matters. Either is decisive.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from . import latent_tasks as LT
from . import latent_organ as LO
from .proposer import FactorGraphProposer

# global padding budget
NMAX, KVAL, MMAX, AMAX = 12, 4, 40, 2


def device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def eval_latent(organ, pool, tok, dev, pset, bs=32, theta=0.5):
    organ.eval()
    rn = rd = fn = fd = 0
    for i in range(0, len(pool), bs):
        items = pool[i:i + bs]
        ba = LT.build_batch(items, tok, NMAX, KVAL, dev, pset)
        b, _, _ = organ(ba)
        a, c, d, e = LT.narrowing_stats(b, ba["tgt"], ba["vmask"], theta)
        rn += a; rd += c; fn += d; fd += e
    organ.train()
    return {"recall": rn / max(1, rd), "false_elim": fn / max(1, fd),
            "elim_total": rd, "surv_total": fd}


@torch.no_grad()
def eval_explicit(prop, pool, dev, bs=64, theta=0.5):
    prop.eval()
    rn = rd = fn = fd = 0
    for i in range(0, len(pool), bs):
        items = pool[i:i + bs]
        feat = LT.featurize_factors(items, NMAX, KVAL, MMAX, AMAX, dev)
        tgt, vmask = LT.targets(items, NMAX, KVAL, dev)
        b, _, _ = prop(feat["var_mask"], feat["given"], feat["fac_rel"], feat["fac_arity"],
                       feat["edge_var"], feat["edge_valid"], feat["var_valid"], feat["fac_valid"])
        a, c, d, e = LT.narrowing_stats(b, tgt, vmask, theta)
        rn += a; rd += c; fn += d; fd += e
    prop.train()
    return {"recall": rn / max(1, rd), "false_elim": fn / max(1, fd),
            "elim_total": rd, "surv_total": fd}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lora", action="store_true", help="LoRA-adapt OLMo so it can expose constraints latently")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_lr", type=float, default=2e-4)
    ap.add_argument("--mid_layer", type=int, default=8)
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--dctx", type=int, default=256)
    ap.add_argument("--mixer", choices=["ffn", "geom", "nowedge"], default="ffn")
    ap.add_argument("--monotone", type=int, default=1)
    ap.add_argument("--tasks", default="coloring,ordering")
    ap.add_argument("--train_n", default="4,5,6")
    ap.add_argument("--test_n", default="5,8,11")
    ap.add_argument("--pool_per", type=int, default=400, help="problems per (n,kind) in the train pool")
    ap.add_argument("--eval_per", type=int, default=120)
    ap.add_argument("--edge_p", type=float, default=0.3)
    ap.add_argument("--pin_frac", type=float, default=0.3)
    ap.add_argument("--prop_steps", type=int, default=None, help="explicit-baseline steps (default=steps)")
    ap.add_argument("--prop_d", type=int, default=96)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.steps = min(a.steps, 120)
        a.pool_per = 80; a.eval_per = 48; a.prop_steps = a.prop_steps or 300
        a.train_n = "4,5"; a.test_n = "5,7"
    a.prop_steps = a.prop_steps or a.steps

    dev = device()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    kinds = tuple(a.tasks.split(","))
    train_n = [int(x) for x in a.train_n.split(",")]
    test_n = [int(x) for x in a.test_n.split(",")]
    gkw = dict(edge_p=a.edge_p, pin_frac=a.pin_frac, m_max=MMAX)

    print(f"== building pools (kinds={kinds} train_n={train_n} test_n={test_n} K={KVAL}) ==", flush=True)
    t0 = time.time()
    train_pool = LT.make_pool(kinds, train_n, KVAL, a.pool_per, rng, **gkw)
    eval_ind = LT.make_pool(kinds, train_n, KVAL, a.eval_per, np.random.default_rng(a.seed + 1), **gkw)
    eval_oodn = LT.make_pool(kinds, test_n, KVAL, a.eval_per, np.random.default_rng(a.seed + 2), **gkw)
    print(f"   train={len(train_pool)} eval_indist={len(eval_ind)} eval_oodN={len(eval_oodn)}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    # ---- host ----
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("loading", a.host, flush=True)
    tok = AutoTokenizer.from_pretrained(a.host)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    olmo = AutoModelForCausalLM.from_pretrained(a.host, dtype=torch.bfloat16).to(dev).eval()
    nL = olmo.config.num_hidden_layers
    for p in olmo.parameters():
        p.requires_grad_(False)
    host = olmo
    if a.lora:
        from peft import LoraConfig, get_peft_model
        lconf = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, bias="none",
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                           "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
        host = get_peft_model(olmo, lconf)
    mid = min(a.mid_layer, nL)
    print(f"OLMo: {nL} layers, hidden {olmo.config.hidden_size}; mid-layer hidden = {mid}; "
          f"LoRA={'on r=%d' % a.lora_r if a.lora else 'off (frozen)'}", flush=True)

    organ = LO.LatentOrgan(host, NMAX, KVAL, mid, train_host=a.lora, dctx=a.dctx, d=a.d,
                           n_layers=a.n_layers, T=a.T, ds=max(1, a.T // 2), mixer=a.mixer,
                           monotone=bool(a.monotone)).to(dev)
    organ.alpha.float(); organ.narrower.float()
    n_organ = LO.n_params(organ.alpha) + LO.n_params(organ.narrower)
    n_lora = sum(p.numel() for p in organ.lora_parameters())
    print(f"LATENT ORGAN: α(dense projector)+narrower = {n_organ:,} params  | LoRA {n_lora:,} "
          f"| T={a.T} n_layers={a.n_layers} d={a.d} dctx={a.dctx} monotone={bool(a.monotone)}", flush=True)

    # ---- train latent organ ----
    groups = [{"params": organ.organ_parameters(), "lr": a.lr}]
    if a.lora:
        groups.append({"params": organ.lora_parameters(), "lr": a.lora_lr})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    organ.train()
    log = []
    t0 = time.time()
    for s in range(1, a.steps + 1):
        idx = rng.integers(0, len(train_pool), a.bs)
        items = [train_pool[i] for i in idx]
        ba = LT.build_batch(items, tok, NMAX, KVAL, dev, "train")
        b, sup, _ = organ(ba)
        loss = LO.dominate_dedp_loss(sup, ba["tgt"], ba["vmask"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(organ.organ_parameters() + organ.lora_parameters(), 1.0)
        opt.step()
        if s % max(1, a.steps // 20) == 0 or s == 1:
            rn, rd, fn, fd = LT.narrowing_stats(b, ba["tgt"], ba["vmask"], a.theta)
            msg = {"step": s, "loss": float(loss.detach()),
                   "recall": rn / max(1, rd), "false_elim": fn / max(1, fd)}
            log.append(msg)
            print(f"  [latent] step {s:5d}  loss {msg['loss']:.3f}  recall {msg['recall']:.3f}  "
                  f"false_elim {msg['false_elim']:.4f}  {time.time()-t0:.0f}s", flush=True)

    lat = {"indist": eval_latent(organ, eval_ind, tok, dev, "train", theta=a.theta),
           "oodN": eval_latent(organ, eval_oodn, tok, dev, "train", theta=a.theta),
           "oodPhrasing": eval_latent(organ, eval_ind, tok, dev, "ood", theta=a.theta)}

    # ---- train explicit factor deductor (the upper bound; given the TRUE factors) ----
    print(f"\n== explicit factor deductor (proposer, TRUE factors) — upper bound ==", flush=True)
    prop = FactorGraphProposer("full", NMAX, KVAL, MMAX, AMAX, d=a.prop_d, R=8, ds=4).to(dev)
    print(f"EXPLICIT PROPOSER: {prop.n_params():,} params (d={a.prop_d}, R=8)", flush=True)
    popt = torch.optim.AdamW(prop.parameters(), lr=3e-4, betas=(0.9, 0.95))
    prop.train()
    t0 = time.time()
    for s in range(1, a.prop_steps + 1):
        idx = rng.integers(0, len(train_pool), a.bs)
        items = [train_pool[i] for i in idx]
        feat = LT.featurize_factors(items, NMAX, KVAL, MMAX, AMAX, dev)
        tgt, vmask = LT.targets(items, NMAX, KVAL, dev)
        b, cls, sup = prop(feat["var_mask"], feat["given"], feat["fac_rel"], feat["fac_arity"],
                           feat["edge_var"], feat["edge_valid"], feat["var_valid"], feat["fac_valid"])
        sup_b = [bb for (bb, _) in sup]
        loss = LO.dominate_dedp_loss(sup_b, tgt, vmask)
        popt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(prop.parameters(), 1.0); popt.step()
        if s % max(1, a.prop_steps // 12) == 0 or s == 1:
            rn, rd, fn, fd = LT.narrowing_stats(b, tgt, vmask, a.theta)
            print(f"  [explicit] step {s:5d}  loss {float(loss.detach()):.3f}  "
                  f"recall {rn/max(1,rd):.3f}  false_elim {fn/max(1,fd):.4f}  {time.time()-t0:.0f}s", flush=True)

    exp = {"indist": eval_explicit(prop, eval_ind, dev, theta=a.theta),
           "oodN": eval_explicit(prop, eval_oodn, dev, theta=a.theta)}

    # ---- the decisive table ----
    print("\n" + "=" * 78)
    print("NARROWING-RECALL / FALSE-ELIM  (theta=%.2f)   latent-from-text vs explicit-from-factors" % a.theta)
    print("=" * 78)
    hdr = f"{'condition':<16}{'recall':>10}{'false_elim':>14}{'#elims':>10}{'#survivors':>12}"
    print(hdr)
    print("-- LATENT (from TEXT, NO factors) " + "-" * 44)
    for name, r in lat.items():
        print(f"{name:<16}{r['recall']*100:>9.1f}%{r['false_elim']*100:>13.2f}%"
              f"{r['elim_total']:>10}{r['surv_total']:>12}")
    print("-- EXPLICIT (from TRUE factors; upper bound) " + "-" * 33)
    for name, r in exp.items():
        print(f"{name:<16}{r['recall']*100:>9.1f}%{r['false_elim']*100:>13.2f}%"
              f"{r['elim_total']:>10}{r['surv_total']:>12}")
    gap = exp["indist"]["recall"] - lat["indist"]["recall"]
    print("-" * 78)
    print(f"VERDICT  in-dist recall: latent {lat['indist']['recall']*100:.1f}%  vs  "
          f"explicit {exp['indist']['recall']*100:.1f}%  (gap {gap*100:+.1f} pts)")

    results = {"args": vars(a), "latent": lat, "explicit": exp, "train_log": log,
               "n_organ": n_organ, "n_lora": int(n_lora), "n_explicit": prop.n_params()}
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "runs", "latent_organ.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(results, open(out, "w"), indent=1)
    print("\nwrote", out, flush=True)


if __name__ == "__main__":
    main()
