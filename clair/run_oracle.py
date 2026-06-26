"""Run the ORACLE-READOUT de-risk experiment (clair.oracle_readout).

  SMOKE:  python -m clair.run_oracle --smoke
  FULL :  python -m clair.run_oracle --steps 2500 --train_n 4,5,6 --test_n 5,8,11

Reports: structured-gamma design echo, no-op-at-init confirmation (bitwise vs base OLMo),
oracle generative accuracy (in-dist + OOD) vs base OLMo few-shot, and THE CAUSAL CONTROL table
(true vs shuffle / permute / corrupt).
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np
import torch

from . import csp as C
from . import curriculum as Cu
from . import oracle_readout as O


FEWSHOT = (
    "A is red. A and B are different colors. What color is A? Answer: red\n"
    "A is blue. A and B are different colors. What color is B? Answer: cannot be determined\n"
    "A is green. B is green. A and C are different colors. What color is B? Answer: green\n"
    "A is red. B is blue. A and B are different colors. What color is C? Answer: cannot be determined\n"
)


def verify_noop(model, tok, dev, K):
    """Confirm base+LoRA(init)+gamma(gate=0) logits == base OLMo logits, bitwise. Returns max|diff|.
    Also returns the max|diff| with the gate FORCED on (sanity: the channel is non-trivial).
    base = peft model with adapters DISABLED; the test run has LoRA ENABLED (B=0 init) + gamma gate 0."""
    pm = model.model
    text = "A is red. A and B are different colors. What color is B? Answer:"
    ids = tok(text, return_tensors="pt").to(dev)
    with torch.no_grad():
        with pm.disable_adapter():
            base = pm(input_ids=ids["input_ids"]).logits.float()
    # random oracle lattice on a couple of fake cells, scattered onto the first few tokens
    T = ids["input_ids"].size(1)
    surv = (torch.rand(1, 3, K, device=dev) > 0.5).float()
    mention = torch.zeros(1, 3, T, device=dev); mention[0, 0, 2] = 1.0; mention[0, 1, 5] = 1.0
    with torch.no_grad():
        with model.injection(surv, mention, enabled=True):
            g0 = model.logits(ids["input_ids"], ids["attention_mask"]).float()
    noop_gap = float((base - g0).abs().max())
    # force the gate on to confirm the channel actually moves logits
    with torch.no_grad():
        saved = model.gamma.alpha.data.clone()
        model.gamma.alpha.data.fill_(2.0)
        with model.injection(surv, mention, enabled=True):
            g1 = model.logits(ids["input_ids"], ids["attention_mask"]).float()
        model.gamma.alpha.data.copy_(saved)
    live_gap = float((base - g1).abs().max())
    return noop_gap, live_gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
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
    ap.add_argument("--base_n", type=int, default=200)
    ap.add_argument("--greedy_n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt_mode", choices=["full", "cells"], default="full",
                    help="full=whole problem text (LoRA can deduce -> confound); cells=roster+question "
                         "only, NO facts (text-deduction impossible -> PURE readout isolator)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.steps = 80; a.pool_per_n = 250; a.eval_n = 96; a.base_n = 48; a.greedy_n = 16
    colors = O.COLORS
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    K = a.k

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    mid = "allenai/OLMo-2-0425-1B"
    print("loading", mid, flush=True)
    tok = AutoTokenizer.from_pretrained(mid)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    olmo = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to(dev).eval()
    D = olmo.config.hidden_size; nL = olmo.config.num_hidden_layers
    print(f"OLMo: {nL} layers, hidden {D}", flush=True)

    # base logits BEFORE wrapping (for the bitwise no-op check)
    for p in olmo.parameters():
        p.requires_grad_(False)
    lconf = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, bias="none",
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
    peft_model = get_peft_model(olmo, lconf)
    inj = min(a.inject_layer, nL - 1)
    model = O.OracleReadout(peft_model, D, K, inj, gamma_hidden=a.gamma_hidden).to(dev)
    # keep gamma in fp32 for stable training; OLMo + LoRA stay as loaded
    model.gamma.float()
    print(f"STRUCTURED GAMMA: shared P: R^{K}->R^{D} (hidden {a.gamma_hidden}), zero-init tanh gate, "
          f"scattered onto cell mention tokens at LATE layer {inj}/{nL} (equivariant over cells, "
          f"full-bandwidth per-cell, not a flatten)", flush=True)
    print(f"trainable params {O.n_trainable(model):,} "
          f"(LoRA r={a.lora_r} attn+mlp + gamma)", flush=True)

    # NO-OP at init: base = adapters disabled; test run = LoRA enabled (B=0) + gamma gate 0
    noop_gap, live_gap = verify_noop(model, tok, dev, K)
    print(f"NO-OP @ INIT (max|base - (LoRA_init+gamma_gate0)|): {noop_gap:.3e}  "
          f"(EXACT no-op expected ~0)   | gate-forced-on moves logits by {live_gap:.3e} (channel live)",
          flush=True)

    # ----- data -----
    train_ns = [int(x) for x in a.train_n.split(",")]
    test_ns = [int(x) for x in a.test_n.split(",")]
    prng = np.random.default_rng(a.seed + 7)
    t0 = time.time()
    train_pool = []
    for N in train_ns:
        train_pool += O.build_pool(prng, N, K, a.pool_per_n, prompt_mode=a.prompt_mode)
    prng.shuffle(train_pool)
    eval_pools = {f"train(N={min(train_ns)}-{max(train_ns)})":
                  O.build_pool(prng, train_ns[len(train_ns) // 2], K, a.eval_n, prompt_mode=a.prompt_mode)}
    for N in test_ns:
        eval_pools[f"N={N}"] = O.build_pool(prng, N, K, a.eval_n, prompt_mode=a.prompt_mode)
    print(f"PROMPT MODE: {a.prompt_mode}  ({'whole problem text — LoRA-deduction confound present' if a.prompt_mode=='full' else 'roster+question only, NO facts — pure readout isolator'})", flush=True)
    print(f"data: {len(train_pool)} train recs (N in {train_ns}); "
          f"eval pools { {k: len(v) for k, v in eval_pools.items()} }  built in {time.time()-t0:.0f}s",
          flush=True)

    # smoke: assert the fast oracle equals csp.exact_dedP
    for r_n in train_ns + test_ns:
        rng2 = np.random.default_rng(123 + r_n)
        for _ in range(20):
            pr = O.make_problem(rng2, r_n, k=K)
            fast = O.oracle_survival_fast(pr)
            exact = C.exact_dedP(pr.csp, pr.csp.full())
            assert fast == exact, f"oracle mismatch at N={r_n}: {fast} != {exact}"
    print("ORACLE CHECK: oracle_survival_fast == clair.csp.exact_dedP on all N (exact).", flush=True)

    # ----- optimizer: LoRA + gamma -----
    lora_params = [p for n, p in model.model.named_parameters() if p.requires_grad]
    gamma_params = list(model.gamma.parameters())
    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": a.lora_lr, "weight_decay": 0.01},
        {"params": gamma_params, "lr": a.gamma_lr, "weight_decay": 0.0},
    ], betas=(0.9, 0.95))
    import torch.nn.functional as F
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    model.train()
    ek = list(eval_pools)[0]
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
            print(f"  step {s:5d}  lm {lm.item():.3f}  alpha {float(model.gamma.alpha):.3f}  "
                  f"in-dist oracle-acc {acc['overall']*100:4.1f}% "
                  f"(det {acc['det_acc']*100:.0f} abst {acc['abst_acc']*100:.0f})  "
                  f"{time.time()-t0:.0f}s", flush=True)
            model.train()

    # ===================================================================== DECISIVE MEASUREMENTS
    print("\n================ ORACLE GENERATIVE ACCURACY (closed-set LM-head) ================", flush=True)
    results = {}
    for name, pool in eval_pools.items():
        oracle = O.score_closedset(model, pool, tok, K, colors, dev, control="true", inject=True, bs=a.bs)
        # gamma-OFF ablation: trained LoRA, but NO lattice injected -> isolates text-only solving
        goff = O.score_closedset(model, pool, tok, K, colors, dev, inject=False, use_base=False, bs=a.bs)
        base = O.score_closedset(model, pool, tok, K, colors, dev, inject=False,
                                 fewshot=FEWSHOT, use_base=True, bs=a.bs)
        results[name] = {"oracle": oracle, "lora_no_lattice": goff, "base_fewshot": base}
        print(f"  {name:16s}  ORACLE {oracle['overall']*100:5.1f}% (det {oracle['det_acc']*100:.0f} "
              f"abst {oracle['abst_acc']*100:.0f})  |  LoRA-no-lattice {goff['overall']*100:5.1f}% "
              f"(det {goff['det_acc']*100:.0f})  |  BASE-fewshot {base['overall']*100:5.1f}% "
              f"(det {base['det_acc']*100:.0f})  n={oracle['n']}", flush=True)

    print("\n================ CAUSAL CONTROL TABLE (does generation depend on the lattice?) ========",
          flush=True)
    print("  pool              true     shuffle   permute   corrupt   (overall acc %)", flush=True)
    controls = {}
    for name, pool in eval_pools.items():
        row = {}
        for ctrl in ("true", "shuffle", "permute", "corrupt"):
            row[ctrl] = O.score_closedset(model, pool, tok, K, colors, dev, control=ctrl,
                                          inject=True, bs=a.bs)["overall"]
        controls[name] = row
        print(f"  {name:16s}  {row['true']*100:6.1f}   {row['shuffle']*100:6.1f}   "
              f"{row['permute']*100:6.1f}   {row['corrupt']*100:6.1f}", flush=True)

    print("\n================ FREE GREEDY GENERATION (sanity: real token emission) ============",
          flush=True)
    gname = list(eval_pools)[0]
    gacc, gex = O.greedy_gen(model, eval_pools[gname][: a.greedy_n], tok, K, colors, dev, control="true")
    gacc_sh, _ = O.greedy_gen(model, eval_pools[gname][: a.greedy_n], tok, K, colors, dev, control="shuffle")
    print(f"  greedy gen on {gname}: TRUE-lattice {gacc*100:.1f}%  vs SHUFFLED {gacc_sh*100:.1f}%", flush=True)
    for gold, got in gex:
        print(f"    gold={gold!r:28s} generated={got!r}", flush=True)

    res = {"args": vars(a), "noop_gap": noop_gap, "live_gap": live_gap,
           "accuracy": {k: v for k, v in results.items()},
           "controls": controls, "greedy": {"true": gacc, "shuffle": gacc_sh}}
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "runs",
                                f"oracle_readout_{a.prompt_mode}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1, default=float)
    print("\nwrote", out, flush=True)

    # ----- interpretation -----
    print("\n================ INTERPRETATION ================", flush=True)
    k0 = list(eval_pools)[0]
    indist = controls[k0]
    drop = indist["true"] - max(indist["shuffle"], indist["permute"], indist["corrupt"])
    goff = results[k0]["lora_no_lattice"]["overall"]
    print(f"  in-dist: oracle {indist['true']*100:.0f}%  worst-control "
          f"{(indist['true']-drop)*100:.0f}%  LoRA-no-lattice {goff*100:.0f}%", flush=True)
    if indist["true"] > 0.7 and drop > 0.25:
        print("  -> ORACLE-acc HIGH and shuffle/permute/corrupt DROP it sharply: the LM head genuinely "
              "READS the lattice through gamma. D's residual-stream readout coupling is VIABLE; next "
              "step is replacing the oracle with a learned organ.", flush=True)
    elif indist["true"] > 0.7 and drop < 0.1:
        extra = (" — and LoRA-no-lattice is ALSO high, so the LoRA simply SOLVED THE TEXT itself "
                 "(deduction confound), not a gamma side-channel" if goff > 0.7 else
                 " — gamma is a free side-channel the LM does not read")
        print("  -> ORACLE-acc HIGH but shuffle/permute/corrupt DO NOT drop it: NOT lattice-reading"
              + extra + ". (Use prompt_mode=cells to remove the text-deduction confound.)", flush=True)
    else:
        print("  -> ORACLE-acc LOW even with the TRUE lattice: residual-stream readout/alignment is the "
              "WALL. D needs rethinking.", flush=True)


if __name__ == "__main__":
    main()
