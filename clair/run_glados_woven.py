"""clair/run_glados_woven.py — train/eval the corrected-D woven GLaDOS (clair.glados_woven).

  python -m clair.run_glados_woven --mode smoke
  python -m clair.run_glados_woven --mode full --steps 600 --out runs/glados_woven.json

SMOKE: loads OLMo-2-1B + LoRA, asserts gamma+LoRA is an EXACT no-op at init (woven logits ==
base OLMo logits), trains a few steps and confirms (a) the organ grounds — dominate-ded_P false-elim
drops, (b) alpha grounds — program-recon loss drops, (c) the LM CE drops and generates parseable
answers. FULL: generative accuracy in-dist + OOD (N=5/8/11) vs base OLMo + the causal-control table
(shuffle / candidate-permute the organ state -> does answer accuracy drop?).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from . import glados_woven as G
from . import curriculum as CU


def device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# --------------------------------------------------------------------- model loading
def load(host="allenai/OLMo-2-0425-1B", K=3, lora_r=16, **kw):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(host)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(host, dtype=torch.float32)
    cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
                     task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n)                     # base frozen, LoRA trainable
    gw = G.GladosWoven(model, tok, K=K, **kw)
    return gw, tok


# --------------------------------------------------------------------- fixed-N coloring problems
def fixed_problems(rng, n_each, N, relation="coloring", det_only=False):
    """Coloring problems with EXACTLY N nodes (for OOD-by-size eval). Replicates curriculum.gen_problem
    using the fixed-N generator + exact query/answer labels. det_only keeps only problems whose query
    is FORCED (a specific color) — the cases where the answer depends on the organ's narrowing, so the
    causal controls are meaningful (abstain cases the LM can guess from edge density)."""
    out = []
    tries = 0
    while len(out) < n_each and tries < n_each * 200:
        tries += 1
        nn, d, kind, facts, s = CU.gen_coloring(rng, k=3, n_lo=N, n_hi=N)
        assert CU.facts_satisfied_by(facts, s, d)
        csp = CU.build_csp(nn, d, facts)
        q, ans, det = CU._query_answer(csp, facts, rng, True if det_only else bool(rng.random() < 0.5))
        if det_only and not det:
            continue
        out.append(CU.Problem(relation, nn, d, kind, facts, q, ans, det, CU.value_names(kind, d)))
    return out


# --------------------------------------------------------------------- answer parsing / accuracy
def var_problems(rng, n, lo, hi, want="any", relation="coloring"):
    """n coloring problems with N sampled in [lo,hi], filtered to determined / abstain / any. Used to
    BALANCE training so the LM must emit specific colors (else coloring is mostly abstain and the model
    collapses to always-'cannot be determined', never exercising the organ's narrowing)."""
    out = []; tries = 0
    while len(out) < n and tries < n * 300:
        tries += 1
        N = int(rng.integers(lo, hi + 1))
        nn, d, kind, facts, s = CU.gen_coloring(rng, k=3, n_lo=N, n_hi=N)
        csp = CU.build_csp(nn, d, facts)
        q, ans, det = CU._query_answer(csp, facts, rng, want != "abs")
        if (want == "det" and not det) or (want == "abs" and det):
            continue
        out.append(CU.Problem(relation, nn, d, kind, facts, q, ans, det, CU.value_names(kind, d)))
    return out


def parse_correct(gen, p):
    g = gen.lower()
    if p.determined:
        return CU.canonical_answer(p).lower() in g
    return ("cannot" in g) or ("determined" in g) or ("any" in g)


@torch.no_grad()
def base_generate(gw, ba, max_new_tokens=6):
    """Base OLMo (LoRA disabled, organ detached): the no-organ baseline generator."""
    gw.eval()
    ids = ba["prompt_ids"].clone(); attn = ba["prompt_attn"].clone()
    B = ids.size(0); dev = ids.device; eos = gw.tok.eos_token_id
    new = [[] for _ in range(B)]; fin = torch.zeros(B, dtype=torch.bool, device=dev)
    with gw.model.disable_adapter():
        for _ in range(max_new_tokens):
            out = gw.model(input_ids=ids, attention_mask=attn).logits
            nxt = out[:, -1, :].argmax(-1)
            for b in range(B):
                if not fin[b]:
                    new[b].append(int(nxt[b]))
            fin = fin | (nxt == eos)
            ids = torch.cat([ids, nxt[:, None]], 1)
            attn = torch.cat([attn, torch.ones(B, 1, device=dev, dtype=attn.dtype)], 1)
            if fin.all():
                break
    return [gw.tok.decode(t, skip_special_tokens=True).strip() for t in new]


@torch.no_grad()
def eval_gen(gw, probs, tok, K, dev, control="none", base=False, bs=16):
    gw.eval()
    correct = parseable = total = 0
    for i in range(0, len(probs), bs):
        chunk = probs[i:i + bs]
        ba = G.build_batch(chunk, tok, K, dev)
        if base:
            gens = base_generate(gw, ba)
        else:
            gw.control = control
            B = len(chunk)
            gw._perm = torch.randperm(B, device=dev) if control == "shuffle" else None
            gw._kperm = torch.randperm(K, device=dev) if control == "permute" else None
            gens = gw.generate(ba)
            gw.control = "none"
        for g, p in zip(gens, chunk):
            total += 1
            parseable += int(any(c.isalnum() for c in g))
            correct += int(parse_correct(g, p))
    return {"acc": correct / max(1, total), "parseable": parseable / max(1, total), "n": total}


# --------------------------------------------------------------------- training
def train(gw, tok, K, dev, steps, bs=8, lr=2e-4, lo=4, hi=6, w_organ=1.0, w_prog=0.5,
          balance=True, seed=0, log=None):
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(gw.trainable_parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    log = log if log is not None else []
    gw.train(); t0 = time.time()
    for s in range(1, steps + 1):
        if balance:                                          # 50/50 forced-color vs abstain
            half = bs // 2
            probs = var_problems(rng, half, lo, hi, "det") + var_problems(rng, bs - half, lo, hi, "abs")
        else:
            probs = var_problems(rng, bs, lo, hi, "any")
        ba = G.build_batch(probs, tok, K, dev)
        out = gw(ba)
        loss = out["lm_ce"] + w_organ * out["organ"] + w_prog * out["prog"]
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gw.trainable_parameters(), 1.0); opt.step()
        if s % max(1, steps // 15) == 0 or s == 1:
            rec = {"step": s, "lm_ce": float(out["lm_ce"]), "organ": float(out["organ"]),
                   "prog": float(out["prog"]), "organ_fe": out["organ_fe"]}
            log.append(rec)
            print(f"  step {s:4d}  lm_ce {rec['lm_ce']:.3f}  organ {rec['organ']:.3f}  "
                  f"prog {rec['prog']:.3f}  organ_FE {rec['organ_fe']:.4f}  {time.time()-t0:.0f}s",
                  flush=True)
    return log


# --------------------------------------------------------------------- modes
def run_smoke(args, dev):
    print("\n===== SMOKE: corrected-D woven GLaDOS =====", flush=True)
    gw, tok = load(K=3, compile_layer=args.compile_layer,
                   writeback_layers=tuple(args.writeback))
    gw.to(dev)
    print(f"  organ params={gw.organ_params:,}  trainable={sum(p.numel() for p in gw.trainable_parameters()):,}"
          f"  compile@{gw.compile_layer} writeback@{gw.writeback_layers}", flush=True)
    rng = np.random.default_rng(0)
    probs = [CU.gen_problem(rng, relation="coloring") for _ in range(6)]
    ba = G.build_batch(probs, tok, 3, dev)
    base_fn = lambda ids, attn: _base_logits(gw, ids, attn)
    diff = G.verify_noop(gw, base_fn, ba)
    print(f"  NO-OP @ init: max|woven - base OLMo logits| = {diff:.3e}  "
          f"({'PASS' if diff < 1e-3 else 'FAIL'})", flush=True)
    assert diff < 1e-3, "gamma/LoRA not a no-op at init!"
    print("  training (coloring N=4..6)...", flush=True)
    log = train(gw, tok, 3, dev, steps=args.steps or 120, bs=args.bs)
    fe0, fe1 = log[0]["organ_fe"], log[-1]["organ_fe"]
    pr0, pr1 = log[0]["prog"], log[-1]["prog"]
    ce0, ce1 = log[0]["lm_ce"], log[-1]["lm_ce"]
    print(f"  organ false-elim {fe0:.4f} -> {fe1:.4f} (grounds: {fe1 < fe0})", flush=True)
    print(f"  program-recon    {pr0:.3f} -> {pr1:.3f} (alpha grounds: {pr1 < pr0})", flush=True)
    print(f"  LM CE            {ce0:.3f} -> {ce1:.3f} (learns answer: {ce1 < ce0})", flush=True)
    ev = fixed_problems(np.random.default_rng(99), 16, 5)
    g = eval_gen(gw, ev, tok, 3, dev)
    print(f"  generation: parseable {g['parseable']*100:.0f}%  acc {g['acc']*100:.0f}% (n={g['n']})", flush=True)
    bsmpl = G.build_batch(ev[:4], tok, 3, dev)
    print("  samples:", [(gw.generate(bsmpl)[i], CU.canonical_answer(ev[i])) for i in range(4)], flush=True)
    return {"smoke": {"noop_diff": diff, "log": log, "gen": g}}


def _base_logits(gw, ids, attn):
    with gw.model.disable_adapter():
        return gw.model(input_ids=ids, attention_mask=attn).logits


def run_full(args, dev):
    print("\n===== FULL: corrected-D woven GLaDOS =====", flush=True)
    gw, tok = load(K=3, compile_layer=args.compile_layer, writeback_layers=tuple(args.writeback))
    gw.to(dev)
    print(f"  organ params={gw.organ_params:,}  trainable={sum(p.numel() for p in gw.trainable_parameters()):,}",
          flush=True)
    ba = G.build_batch([CU.gen_problem(np.random.default_rng(0), relation="coloring") for _ in range(4)],
                       tok, 3, dev)
    diff = G.verify_noop(gw, lambda ids, attn: _base_logits(gw, ids, attn), ba)
    print(f"  NO-OP @ init max|diff| = {diff:.3e}", flush=True)
    log = train(gw, tok, 3, dev, steps=args.steps, bs=args.bs, lr=args.lr)
    out = {"noop_diff": diff, "log": log, "eval": {}, "controls": {}}

    print("\n  --- generative accuracy: woven vs base OLMo (overall + FORCED-only) ---", flush=True)
    for N in [5, 8, 11]:
        ev = fixed_problems(np.random.default_rng(1000 + N), args.neval, N)
        evd = fixed_problems(np.random.default_rng(2000 + N), args.neval, N, det_only=True)
        w = eval_gen(gw, ev, tok, 3, dev); b = eval_gen(gw, ev, tok, 3, dev, base=True)
        wd = eval_gen(gw, evd, tok, 3, dev); bd = eval_gen(gw, evd, tok, 3, dev, base=True)
        out["eval"][f"N{N}"] = {"woven": w, "base": b, "woven_forced": wd, "base_forced": bd}
        print(f"  N={N:2d}  overall woven {w['acc']*100:5.1f}% base {b['acc']*100:5.1f}%  |  "
              f"FORCED woven {wd['acc']*100:5.1f}% base {bd['acc']*100:5.1f}%  (n={w['n']})", flush=True)

    print("\n  --- causal controls on FORCED-answer problems (does corrupting the organ break gen?) ---",
          flush=True)
    for N in [5, 8]:
        ev = fixed_problems(np.random.default_rng(7777 + N), args.neval, N, det_only=True)
        out["controls"][f"N{N}"] = {}
        for ctl in ["none", "shuffle", "permute"]:
            c = eval_gen(gw, ev, tok, 3, dev, control=ctl)
            out["controls"][f"N{N}"][ctl] = c
            print(f"  N={N:2d} control={ctl:8s}  acc {c['acc']*100:5.1f}%  (n={c['n']})", flush=True)
        b = eval_gen(gw, ev, tok, 3, dev, base=True)
        out["controls"][f"N{N}"]["base"] = b
        intact = out["controls"][f"N{N}"]["none"]["acc"]
        shuf = out["controls"][f"N{N}"]["shuffle"]["acc"]
        print(f"  N={N:2d}  base(no-organ) acc {b['acc']*100:5.1f}%  | organ load-bearing: "
              f"{intact - shuf > 0.05} (intact {intact*100:.1f}% vs shuffled {shuf*100:.1f}%)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--neval", type=int, default=48)
    ap.add_argument("--compile_layer", type=int, default=5)
    ap.add_argument("--writeback", type=int, nargs="+", default=[7, 9, 11])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.steps:
        args.steps = 120 if args.mode == "smoke" else 600
    dev = device()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    print(f"device={dev}  budget N={G.NMAX} D={G.DMAX} M={G.MMAX}", flush=True)
    out = run_smoke(args, dev) if args.mode == "smoke" else run_full(args, dev)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1, default=float)
        print("\nwrote", args.out, flush=True)


if __name__ == "__main__":
    main()
