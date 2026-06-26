"""Run the FROZEN-LEARNED-ORGAN readout experiment (clair.frozen_readout) — the clean complement to
the oracle-readout de-risk (clair.run_oracle).

  SMOKE: python -m clair.run_frozen --smoke
  FULL : python -m clair.run_frozen --steps 2500 --organ_steps 1500 --train_n 4,5,6 --test_n 5,8,11

Pipeline:
  1. TRAIN the FactorGraphProposer organ standalone (dominate-dedP, coloring, N in 4..11), FREEZE it,
     report its narrowing-recall vs the EXACT dedP (= oracle) per N.
  2. Extract the FROZEN organ's narrowed lattice (run to fixpoint) -> per-cell surv[n,K]; inject it
     through the SAME structured gamma as the oracle de-risk; train LoRA + gamma generatively.
  3. SAME causal controls (true/shuffle/permute/corrupt) + in-dist + OOD (N=5,8,11), same base
     comparison.  Optionally also run the EXACT-ORACLE readout in the SAME harness (--oracle_baseline)
     for an apples-to-apples side-by-side.

DECISIVE READ (vs the oracle: true=100, shuffle~30, permute=50, corrupt=0):
  frozen organ reads NEARLY as well  -> box1 failure was CO-TRAINING distrust (fix: bootstrap+freeze).
  frozen organ reads POORLY          -> learned-organ noise / lattice mismatch (correlate w/ recall).
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from . import csp as C
from . import curriculum as CU
from . import oracle_readout as O
from . import frozen_readout as Fz


FEWSHOT = (
    "A is red. A and B are different colors. What color is A? Answer: red\n"
    "A is blue. A and B are different colors. What color is B? Answer: cannot be determined\n"
    "A is green. B is green. A and C are different colors. What color is B? Answer: green\n"
    "A is red. B is blue. A and B are different colors. What color is C? Answer: cannot be determined\n"
)


# ===================================================================== organ recall report (vs oracle)
def organ_recall_report(organ, dev, ns, k, n_per=200, seed=999):
    """Per-N narrowing-recall + false-elim of the FROZEN organ vs the EXACT dedP (= oracle lattice),
    plus query-cell exactness rate (the fraction of problems where the organ narrows the QUERY cell
    to exactly the oracle's surviving set — the readout-relevant quantity)."""
    rng = np.random.default_rng(seed)
    dedp = Fz.DedPCache()
    rows = {}
    for N in ns:
        probs = [O.make_problem(rng, N, k=k) for _ in range(n_per)]
        csps = [p.csp for p in probs]
        odoms = Fz.organ_fixpoint(organ, csps, dev)
        rec_num = rec_den = fe = 0
        qhit = match = 0
        recs_inst = []
        for p, od, csp in zip(probs, odoms, csps):
            r, f, oracle = Fz.recall_fe(od, csp, dedp)
            full = csp.full()
            rm_o = Fz._removed(full, od, csp.n); rm_t = Fz._removed(full, oracle, csp.n)
            rec_num += len(rm_o & rm_t); rec_den += len(rm_t); fe += f
            qhit += int(set(od[p.query]) == set(oracle[p.query]))
            match += int(tuple(od) == tuple(oracle))
            recs_inst.append(r)
        rows[N] = {"recall": rec_num / max(1, rec_den), "false_elim_count": int(fe),
                   "query_exact": qhit / n_per, "fixpoint_match": match / n_per,
                   "mean_inst_recall": float(np.mean(recs_inst)), "n": n_per}
    return rows


# ===================================================================== pools (organ OR oracle surv)
def build_pool(rng, N, K, size, source, organ, dev, det_frac=0.5, prompt_mode="full"):
    """Balanced coloring pool. source='oracle' -> exact surv (reproduces run_oracle); source='organ'
    -> the FROZEN organ's fixpoint surv injected instead. Each record also carries the oracle surv,
    the per-instance recall, and query_exact (organ query-cell == oracle query-cell) for correlation."""
    n_det = int(size * det_frac)
    probs_det, probs_ab = [], []
    while len(probs_det) < n_det or len(probs_ab) < size - n_det:
        want = len(probs_det) < n_det
        only = (len(probs_det) < n_det) ^ (len(probs_ab) < size - n_det)
        p = O.make_problem(rng, N, k=K, det_target=want if only else None)
        if p.determined and len(probs_det) < n_det:
            probs_det.append(p)
        elif (not p.determined) and len(probs_ab) < size - n_det:
            probs_ab.append(p)
    probs = probs_det + probs_ab
    rng.shuffle(probs)
    recs = [O.make_record(p, K, prompt_mode) for p in probs]   # oracle surv + prompt/mentions/gold
    if source == "organ":
        dedp = Fz.DedPCache()
        csps = [p.csp for p in probs]
        odoms = Fz.organ_fixpoint(organ, csps, dev)
        for rec, p, od, csp in zip(recs, probs, odoms, csps):
            r, f, oracle = Fz.recall_fe(od, csp, dedp)
            rec["surv_oracle"] = rec["surv"].copy()
            rec["surv"] = Fz.surv_from_dom(od, p.n, K)
            rec["recall"] = float(r); rec["fe_count"] = int(f)
            rec["query_exact"] = bool(set(od[p.query]) == set(oracle[p.query]))
    else:
        for rec in recs:
            rec["surv_oracle"] = rec["surv"].copy()
            rec["recall"] = 1.0; rec["fe_count"] = 0; rec["query_exact"] = True
    return recs


# ===================================================================== no-op-at-init check
def verify_noop(model, tok, dev, K):
    pm = model.model
    text = "A is red. A and B are different colors. What color is B? Answer:"
    ids = tok(text, return_tensors="pt").to(dev)
    with torch.no_grad(), pm.disable_adapter():
        base = pm(input_ids=ids["input_ids"]).logits.float()
    T = ids["input_ids"].size(1)
    surv = (torch.rand(1, 3, K, device=dev) > 0.5).float()
    mention = torch.zeros(1, 3, T, device=dev); mention[0, 0, 2] = 1.0; mention[0, 1, 5] = 1.0
    with torch.no_grad(), model.injection(surv, mention, enabled=True):
        g0 = model.logits(ids["input_ids"], ids["attention_mask"]).float()
    noop = float((base - g0).abs().max())
    with torch.no_grad():
        saved = model.gamma.alpha.data.clone(); model.gamma.alpha.data.fill_(2.0)
        with model.injection(surv, mention, enabled=True):
            g1 = model.logits(ids["input_ids"], ids["attention_mask"]).float()
        model.gamma.alpha.data.copy_(saved)
    return noop, float((base - g1).abs().max())


# ===================================================================== one readout (train + eval)
def train_eval_readout(source, organ, olmo_ids, train_pool, eval_pools, tok, K, colors, dev, a):
    """Build a fresh LoRA+gamma readout over a given surv `source`, train on train_pool, eval with
    accuracy table + causal controls (+ recall correlation for the organ source). Returns a results
    dict. olmo_ids = (mid, D, nL) so we reload a clean base per source (independent LoRA)."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    mid, D, nL = olmo_ids
    olmo = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to(dev).eval()
    for p in olmo.parameters():
        p.requires_grad_(False)
    lconf = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, bias="none",
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
    peft_model = get_peft_model(olmo, lconf)
    inj = min(a.inject_layer, nL - 1)
    model = O.OracleReadout(peft_model, D, K, inj, gamma_hidden=a.gamma_hidden).to(dev)
    model.gamma.float()
    noop, live = verify_noop(model, tok, dev, K)
    print(f"\n######## SOURCE = {source.upper()} ########  inject layer {inj}/{nL}  "
          f"trainable {O.n_trainable(model):,}", flush=True)
    print(f"  NO-OP @ INIT max|base-(LoRA_init+gate0)| = {noop:.3e} (expect ~0) | "
          f"gate-on moves logits {live:.3e}", flush=True)

    lora_params = [p for n, p in model.model.named_parameters() if p.requires_grad]
    gamma_params = list(model.gamma.parameters())
    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": a.lora_lr, "weight_decay": 0.01},
        {"params": gamma_params, "lr": a.gamma_lr, "weight_decay": 0.0},
    ], betas=(0.9, 0.95))
    rng = np.random.default_rng(a.seed)
    ek = list(eval_pools)[0]
    t0 = time.time(); model.train()
    for s in range(1, a.steps + 1):
        idxs = rng.integers(0, len(train_pool), a.bs).tolist()
        ba = O.build_train_batch([train_pool[i] for i in idxs], tok, K, dev)
        with model.injection(ba["surv"], ba["mention"], enabled=True):
            logits = model.logits(ba["input_ids"], ba["attn"]).float()
        lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                             ba["labels"][:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad(); lm.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0); opt.step()
        if s % max(1, a.steps // 10) == 0 or s == 1:
            acc = O.score_closedset(model, eval_pools[ek], tok, K, colors, dev,
                                    control="true", inject=True, bs=a.bs)
            print(f"  [{source}] step {s:5d}  lm {lm.item():.3f}  alpha {float(model.gamma.alpha):.3f}  "
                  f"in-dist acc {acc['overall']*100:4.1f}% (det {acc['det_acc']*100:.0f} "
                  f"abst {acc['abst_acc']*100:.0f})  {time.time()-t0:.0f}s", flush=True)
            model.train()

    print(f"\n  ---- [{source}] GENERATIVE ACCURACY (closed-set LM head) ----", flush=True)
    results = {}
    for name, pool in eval_pools.items():
        readout = O.score_closedset(model, pool, tok, K, colors, dev, control="true", inject=True, bs=a.bs)
        goff = O.score_closedset(model, pool, tok, K, colors, dev, inject=False, use_base=False, bs=a.bs)
        base = O.score_closedset(model, pool, tok, K, colors, dev, inject=False,
                                 fewshot=FEWSHOT, use_base=True, bs=a.bs)
        results[name] = {"readout": readout, "lora_no_lattice": goff, "base_fewshot": base}
        print(f"    {name:16s}  READOUT {readout['overall']*100:5.1f}% (det {readout['det_acc']*100:.0f} "
              f"abst {readout['abst_acc']*100:.0f})  |  LoRA-no-lattice {goff['overall']*100:5.1f}%  |  "
              f"BASE-fewshot {base['overall']*100:5.1f}%  n={readout['n']}", flush=True)

    print(f"\n  ---- [{source}] CAUSAL CONTROL TABLE (true/shuffle/permute/corrupt overall acc %) ----",
          flush=True)
    print("    pool              true     shuffle   permute   corrupt", flush=True)
    controls = {}
    for name, pool in eval_pools.items():
        row = {c: O.score_closedset(model, pool, tok, K, colors, dev, control=c, inject=True,
                                    bs=a.bs)["overall"] for c in ("true", "shuffle", "permute", "corrupt")}
        controls[name] = row
        print(f"    {name:16s}  {row['true']*100:6.1f}   {row['shuffle']*100:6.1f}   "
              f"{row['permute']*100:6.1f}   {row['corrupt']*100:6.1f}", flush=True)

    # ---- recall correlation (organ source only): does readout-acc track organ correctness? ----
    corr = {}
    if source == "organ":
        print(f"\n  ---- [{source}] READOUT-ACC vs ORGAN CORRECTNESS (per pool) ----", flush=True)
        for name, pool in eval_pools.items():
            hit = [r for r in pool if r["query_exact"]]
            miss = [r for r in pool if not r["query_exact"]]
            ah = O.score_closedset(model, hit, tok, K, colors, dev, control="true", inject=True, bs=a.bs) if hit else None
            am = O.score_closedset(model, miss, tok, K, colors, dev, control="true", inject=True, bs=a.bs) if miss else None
            corr[name] = {"query_exact_frac": len(hit) / max(1, len(pool)),
                          "acc_query_exact": (ah["overall"] if ah else None),
                          "acc_query_wrong": (am["overall"] if am else None),
                          "n_exact": len(hit), "n_wrong": len(miss)}
            print(f"    {name:16s}  query-exact {len(hit)/max(1,len(pool))*100:4.0f}% of pool  |  "
                  f"acc|query-exact {('%.1f'%(ah['overall']*100)) if ah else '  - '}%  "
                  f"acc|query-wrong {('%.1f'%(am['overall']*100)) if am else '  - '}%", flush=True)

    gname = list(eval_pools)[0]
    gacc, gex = O.greedy_gen(model, eval_pools[gname][: a.greedy_n], tok, K, colors, dev, control="true")
    gacc_sh, _ = O.greedy_gen(model, eval_pools[gname][: a.greedy_n], tok, K, colors, dev, control="shuffle")
    print(f"\n  ---- [{source}] FREE GREEDY GEN on {gname}: TRUE {gacc*100:.1f}% vs SHUFFLED {gacc_sh*100:.1f}% ----",
          flush=True)

    del model, peft_model, olmo
    torch.cuda.empty_cache()
    return {"noop_gap": noop, "live_gap": live, "accuracy": results, "controls": controls,
            "recall_corr": corr, "greedy": {"true": gacc, "shuffle": gacc_sh}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--organ_steps", type=int, default=1500)
    ap.add_argument("--organ_target", type=float, default=3.0e5)
    ap.add_argument("--organ_R", type=int, default=8)
    ap.add_argument("--organ_lr", type=float, default=3e-4)
    ap.add_argument("--organ_pool", type=int, default=96)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lora_lr", type=float, default=2e-4)
    ap.add_argument("--gamma_lr", type=float, default=1e-3)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--inject_layer", type=int, default=12)
    ap.add_argument("--gamma_hidden", type=int, default=256)
    ap.add_argument("--train_n", default="4,5,6")
    ap.add_argument("--test_n", default="5,8,11")
    ap.add_argument("--pool_per_n", type=int, default=1500)
    ap.add_argument("--eval_n", type=int, default=300)
    ap.add_argument("--greedy_n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt_mode", choices=["full", "cells"], default="full")
    ap.add_argument("--oracle_baseline", action="store_true",
                    help="also run the EXACT-oracle readout in the SAME harness (apples-to-apples)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.steps = 80; a.organ_steps = 200; a.pool_per_n = 250; a.eval_n = 96; a.greedy_n = 16
        a.organ_pool = 64; a.oracle_baseline = True
    colors = O.COLORS
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    K = a.k
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    train_ns = [int(x) for x in a.train_n.split(",")]
    test_ns = [int(x) for x in a.test_n.split(",")]
    organ_ns = sorted(set(range(min(train_ns), max(test_ns) + 1)))   # train organ across 4..11

    # ================= 1. TRAIN + FREEZE the organ; report recall vs oracle =================
    print("================ STEP 1: STANDALONE ORGAN (dominate-dedP, coloring) ================", flush=True)
    print(f"budget N_MAX={Fz.N_MAX} D_MAX={Fz.D_MAX} M_MAX={Fz.M_MAX} A_MAX={Fz.A_MAX}", flush=True)
    organ, od, opar, olog = Fz.train_organ(dev, organ_ns, target=a.organ_target, steps=a.organ_steps,
                                           pool=a.organ_pool, R=a.organ_R, lr=a.organ_lr, seed=a.seed)
    for p in organ.parameters():
        p.requires_grad_(False)
    organ.eval()
    # smoke: assert fast_dedP == csp.exact_dedP on a few instances
    rngc = np.random.default_rng(7)
    for N in organ_ns:
        for _ in range(8):
            csp = Fz.coloring_csp(rngc, N)
            assert Fz.fast_dedP(csp, csp.full()) == C.exact_dedP(csp, csp.full()), f"dedP mismatch N={N}"
    print("  DEDP CHECK: fast_dedP == clair.csp.exact_dedP on all N (exact).", flush=True)
    recall_rows = organ_recall_report(organ, dev, sorted(set(train_ns + test_ns)), K,
                                      n_per=(80 if a.smoke else 300))
    print("\n  FROZEN ORGAN RECALL vs EXACT dedP (= oracle lattice):", flush=True)
    print(f"  {'N':>4s}  {'recall':>8s}  {'false-elim':>11s}  {'query-exact':>12s}  {'fixpt-match':>12s}",
          flush=True)
    for N in sorted(recall_rows):
        r = recall_rows[N]
        print(f"  {N:>4d}  {r['recall']*100:7.1f}%  {r['false_elim_count']:>11d}  "
              f"{r['query_exact']*100:11.1f}%  {r['fixpoint_match']*100:11.1f}%", flush=True)

    # ================= load OLMo / tokenizer (once) =================
    from transformers import AutoTokenizer, AutoModelForCausalLM
    mid = "allenai/OLMo-2-0425-1B"
    print(f"\n================ STEP 2/3: WOVEN READOUT on the FROZEN-ORGAN lattice ================", flush=True)
    print("loading", mid, flush=True)
    tok = AutoTokenizer.from_pretrained(mid)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    probe = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16)
    D = probe.config.hidden_size; nL = probe.config.num_hidden_layers
    del probe
    print(f"OLMo: {nL} layers, hidden {D}", flush=True)
    print(f"PROMPT MODE: {a.prompt_mode}", flush=True)

    # ================= build pools (organ surv; oracle surv if baseline) =================
    sources = ["organ"] + (["oracle"] if a.oracle_baseline else [])
    all_results = {}
    for source in sources:
        prng = np.random.default_rng(a.seed + 7)
        t0 = time.time()
        train_pool = []
        for N in train_ns:
            train_pool += build_pool(prng, N, K, a.pool_per_n, source, organ, dev, prompt_mode=a.prompt_mode)
        prng.shuffle(train_pool)
        mid_n = train_ns[len(train_ns) // 2]
        eval_pools = {f"train(N={min(train_ns)}-{max(train_ns)})":
                      build_pool(prng, mid_n, K, a.eval_n, source, organ, dev, prompt_mode=a.prompt_mode)}
        for N in test_ns:
            eval_pools[f"N={N}"] = build_pool(prng, N, K, a.eval_n, source, organ, dev, prompt_mode=a.prompt_mode)
        print(f"\n[{source}] data: {len(train_pool)} train recs; eval pools "
              f"{ {k: len(v) for k, v in eval_pools.items()} }  built in {time.time()-t0:.0f}s", flush=True)
        res = train_eval_readout(source, organ, (mid, D, nL), train_pool, eval_pools, tok, K, colors, dev, a)
        all_results[source] = res

    # ================= verdict =================
    print("\n================ VERDICT ================", flush=True)
    org = all_results["organ"]
    k0 = list(org["controls"])[0]
    indist = org["controls"][k0]
    drop = indist["true"] - max(indist["shuffle"], indist["permute"], indist["corrupt"])
    print(f"  ORGAN readout in-dist: true {indist['true']*100:.0f}%  worst-control "
          f"{(indist['true']-drop)*100:.0f}%  (drop {drop*100:.0f}pts)", flush=True)
    if a.oracle_baseline:
        orc = all_results["oracle"]["controls"][k0]
        print(f"  ORACLE readout in-dist: true {orc['true']*100:.0f}%  shuffle {orc['shuffle']*100:.0f}% "
              f"permute {orc['permute']*100:.0f}% corrupt {orc['corrupt']*100:.0f}%", flush=True)
    rec_in = recall_rows[sorted(set(train_ns))[len(set(train_ns)) // 2]]["recall"]
    if indist["true"] > 0.7 and drop > 0.25:
        print("  -> FROZEN learned organ READS WELL (high true-acc, controls drop sharply): the readout "
              "works for a good FIXED organ. box1's failure was CO-TRAINING distrust of a noisy/moving "
              "organ; the fix is bootstrap-then-freeze (or a slow organ LR).", flush=True)
    elif indist["true"] <= 0.7:
        print(f"  -> FROZEN learned organ READS POORLY despite recall~{rec_in*100:.0f}%: the issue is "
              "learned-organ noise / lattice distributional mismatch, NOT co-training. See the "
              "READOUT-ACC vs ORGAN-CORRECTNESS table above for the correlation.", flush=True)
    else:
        print("  -> mixed: high true-acc but weak control drop — inspect LoRA-no-lattice / prompt_mode.",
              flush=True)

    out = a.out or os.path.join(os.path.dirname(__file__), "..", "runs",
                                f"frozen_readout_{a.prompt_mode}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    blob = {"args": vars(a), "organ": {"d_model": od, "params": opar, "log": olog,
            "recall": recall_rows}, "results": all_results}
    json.dump(blob, open(out, "w"), indent=1, default=float)
    print("\nwrote", out, flush=True)


if __name__ == "__main__":
    main()
