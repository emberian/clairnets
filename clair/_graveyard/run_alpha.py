"""clair/run_alpha.py — PRETRAIN THE GLaDOS COMPILER (alpha) STANDALONE.

alpha is the text->program interface piece of the staged GLaDOS recipe. It reads OLMo's hidden
states at a mid layer and emits a TYPED FACTOR-GRAPH PROGRAM: per-cell value DOMAINS (the unary
givens) + per-(unordered)-pair RELATION TABLES (the binary constraints). gamma is de-risked by the
oracle readout and the organ trains fine alone; alpha has only ever been CO-TRAINED (inside
glados_woven, getting gradient from the LM loss through gamma->organ). This file trains alpha ALONE,
supervised DIRECTLY by the true program — no organ, no deductor, no answer generation, no write-back.

  text  --OLMo+LoRA-->  hidden@L  --Compiler-->  (dom_logits [B,N,K], tab_logits [B,N,N,K,K])
  loss  = domain BCE (masked to valid cells x valid values) + relation-table BCE (masked to valid pairs)

We REUSE glados_woven.Compiler verbatim (the grounded CellReader + dom_head + PairTableReader whose
local co-occurrence grounding fixed the earlier phrasing collapse). The only new code is the
program-target construction (facts -> domains + KxK tables) and the assembly-readiness metrics.

RELATIONS that map onto the per-pair binary scaffold: coloring(neq,pin), equality(eq,neq,pin),
ordering(lt,le,pin), alldiff(->pairwise neq, pin). arithmetic/xor are arity-3 (no binary table) and
are EXCLUDED. Domains are scored over the actual value range [0,d) (d is part of the program).

  python -m clair.run_alpha --mode smoke
  python -m clair.run_alpha --mode full --steps 2500 --out runs/alpha.json

MEASURES (in-dist + OOD-size + OOD-phrasing): program-exact (whole program correct), pin/domain
accuracy, relation-table exact, constraint-detect F1 (the 'edge-F1' analogue), relation-type acc.
"""
from __future__ import annotations

import argparse
import itertools as it
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import glados_woven as G
from . import curriculum as CU
from . import csp as C

NMAX, K = G.NMAX, G.DMAX                       # 12 cells, 8 value slots (K = domain budget)
MAPPABLE = ("coloring", "equality", "ordering", "alldiff")   # relations expressible as domains+pair tables


# ===================================================================== program target construction
def _dir_table(kind):
    """The forward KxK allowed-tuple table for a binary relation: T[va,vb]=1 iff (va REL vb)."""
    idx = np.arange(K)
    if kind == "eq":
        return np.eye(K, dtype=np.float32)
    if kind == "neq":
        return (1.0 - np.eye(K)).astype(np.float32)
    if kind == "lt":
        return (idx[:, None] < idx[None, :]).astype(np.float32)
    if kind == "le":
        return (idx[:, None] <= idx[None, :]).astype(np.float32)
    raise ValueError(kind)


def facts_to_program(facts):
    """Split a fact list into (pins {cell:value}, binc [(a,b,kind)]). alldiff -> pairwise neq.
    Tolerates both tuple facts (canonical Problem) and list facts (jsonl records)."""
    pins, binc = {}, []
    for f in facts:
        k = f[0]
        if k == "pin":
            pins[int(f[1])] = int(f[2])
        elif k in ("eq", "neq", "lt", "le"):
            binc.append((int(f[1]), int(f[2]), k))
        elif k == "alldiff":
            sc = list(f[1])
            for a, b in it.combinations(sc, 2):
                binc.append((int(a), int(b), "neq"))
        # sum/xor are arity-3: not representable as a binary table -> caller filters these relations out
    return pins, binc


def encode_batch(samples, tok, dev):
    """samples: list of dicts {text, mentions{cell:[(s,e)]}, facts, n, d}. Build OLMo inputs, mention
    masks, and the program-reconstruction targets/masks (domains over [0,d), tables over [0,d)^2)."""
    B = len(samples)
    texts = [s["text"] for s in samples]
    enc = tok(texts, return_offsets_mapping=True, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(dev); attn = enc["attention_mask"].to(dev)
    offs = enc["offset_mapping"].tolist()
    T = input_ids.size(1)
    mention = torch.zeros(B, NMAX, T, device=dev)
    vmask = torch.zeros(B, NMAX, device=dev)
    dom_t = np.zeros((B, NMAX, K), np.float32); dom_m = np.zeros((B, NMAX, K), np.float32)
    tab_t = np.ones((B, NMAX, NMAX, K, K), np.float32); tab_m = np.zeros((B, NMAX, NMAX, K, K), np.float32)
    ns = [int(s["n"]) for s in samples]; ds = [int(s["d"]) for s in samples]
    for bi, s in enumerate(samples):
        n, d = ns[bi], ds[bi]
        for v, sps in s["mentions"].items():
            v = int(v)
            if v >= NMAX:
                continue
            for (cs, ce) in sps:
                for ti, (a, c) in enumerate(offs[bi]):
                    if a == c:
                        continue
                    if a < ce and c > cs:
                        mention[bi, v, ti] = 1.0
        vmask[bi, :n] = 1.0
        pins, binc = facts_to_program(s["facts"])
        for i in range(n):
            dom_m[bi, i, :d] = 1.0
            if i in pins:
                dom_t[bi, i, :d] = 0.0; dom_t[bi, i, pins[i]] = 1.0
            else:
                dom_t[bi, i, :d] = 1.0
            for j in range(n):
                if j != i:
                    tab_m[bi, i, j, :d, :d] = 1.0
        for (a, b, kind) in binc:
            Tt = _dir_table(kind)
            tab_t[bi, a, b] *= Tt; tab_t[bi, b, a] *= Tt.T          # AND multiple clauses on a pair
    return {"input_ids": input_ids, "attn": attn, "mention": mention, "vmask": vmask,
            "dom_t": torch.as_tensor(dom_t, device=dev), "dom_m": torch.as_tensor(dom_m, device=dev),
            "tab_t": torch.as_tensor(tab_t, device=dev), "tab_m": torch.as_tensor(tab_m, device=dev),
            "ns": ns, "ds": ds}


# ===================================================================== sample sources
def _problem_sample(p):
    text = CU.canonical_render(p)
    return {"text": text, "mentions": CU.entity_mentions(text, p.n),
            "facts": p.facts, "n": p.n, "d": p.d, "relation": p.relation}


# per-relation N caps so the witness domain d stays within the K=8 value budget
#   coloring/equality: d=3 (fixed)         ordering: d<=n+2      alldiff: d<=n+1
NCAP = {"coloring": (4, 8), "equality": (4, 8), "ordering": (4, 6), "alldiff": (4, 6)}


def gen_problem_n(rng, relation, n_lo, n_hi):
    """A verified Problem for `relation` with N in [n_lo,n_hi]; resamples if the witness domain d>K."""
    g = CU.GENERATORS[relation]
    for _ in range(50):
        nn, d, kind, facts, s = g(rng, n_lo=n_lo, n_hi=n_hi)
        if d > K:
            continue
        assert CU.facts_satisfied_by(facts, s, d)
        csp = CU.build_csp(nn, d, facts)
        q, ans, det = CU._query_answer(csp, facts, rng, bool(rng.random() < 0.5))
        return CU.Problem(relation, nn, d, kind, facts, q, ans, det, CU.value_names(kind, d))
    raise RuntimeError(f"could not sample {relation} with d<=K in N=[{n_lo},{n_hi}]")


def sample_canonical(rng, bs, relations=MAPPABLE):
    out = []
    for _ in range(bs):
        rel = str(rng.choice(relations))
        lo, hi = NCAP[rel]
        out.append(_problem_sample(gen_problem_n(rng, rel, lo, hi)))
    return out


def fixed_canonical(rng, n_each, N, relations=("coloring", "equality")):
    """Fixed-N canonical problems for OOD-by-size (coloring/equality keep d=3 at any N)."""
    out = []
    for _ in range(n_each):
        rel = str(rng.choice(relations))
        out.append(_problem_sample(gen_problem_n(rng, rel, N, N)))
    return out


def record_sample(rec):
    """A diverse Bedrock record -> sample (held-out phrasing). None if not mappable / domain too big."""
    if rec["relation"] not in MAPPABLE or int(rec["d"]) > K or int(rec["n"]) > NMAX:
        return None
    ment = {int(k): [tuple(sp) for sp in v] for k, v in rec["mentions"].items()}
    return {"text": rec["text"], "mentions": ment, "facts": rec["facts"],
            "n": int(rec["n"]), "d": int(rec["d"]), "relation": rec["relation"]}


def load_diverse(path):
    recs = CU.load_curriculum(path)
    samples = [s for s in (record_sample(r) for r in recs) if s is not None]
    return samples


# ===================================================================== the standalone alpha module
class Alpha(nn.Module):
    def __init__(self, model, tok, compile_layer=8, dp=512):
        super().__init__()
        self.model = model; self.tok = tok; self.compile_layer = compile_layer
        D = model.config.hidden_size; self.D = D
        self.compiler = G.Compiler(D, K, dp=dp)

    def forward(self, ba):
        out = self.model(input_ids=ba["input_ids"], attention_mask=ba["attn"], output_hidden_states=True)
        h = out.hidden_states[self.compile_layer + 1].float()       # output of decoder layer `compile_layer`
        m = ba["mention"]
        denom = m.sum(-1, keepdim=True).clamp_min(1e-6)
        v_mean = torch.einsum("bnt,btd->bnd", m, h) / denom          # cell identity (mention-mean)
        dom_logits, tab_logits = self.compiler(v_mean, h, m, ba["attn"])
        return dom_logits, tab_logits

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def load(host="allenai/OLMo-2-0425-1B", lora_r=16, compile_layer=8, dp=512):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(host)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(host, dtype=torch.float32)
    cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n)
    return Alpha(model, tok, compile_layer=compile_layer, dp=dp), tok


# ===================================================================== loss + metrics
def prog_loss(dom_logits, tab_logits, ba):
    dl = F.binary_cross_entropy_with_logits(dom_logits, ba["dom_t"], reduction="none")
    dl = (dl * ba["dom_m"]).sum() / ba["dom_m"].sum().clamp_min(1)
    tl = F.binary_cross_entropy_with_logits(tab_logits, ba["tab_t"], reduction="none")
    tl = (tl * ba["tab_m"]).sum() / ba["tab_m"].sum().clamp_min(1)
    return dl, tl


@torch.no_grad()
def accumulate(dom_logits, tab_logits, ba, agg, relations=None):
    """Accumulate assembly-readiness counts into agg (dict of ints). Scores over the valid region."""
    dp = (dom_logits > 0).float(); tp = (tab_logits > 0).float()
    dom_t, tab_t = ba["dom_t"], ba["tab_t"]; ns, ds = ba["ns"], ba["ds"]
    for bi in range(dom_logits.size(0)):
        n, d = ns[bi], ds[bi]
        prog_good = True
        for i in range(n):
            match = bool((dp[bi, i, :d] == dom_t[bi, i, :d]).all())
            agg["cell_ok"] += int(match); agg["cell_tot"] += 1
            if float(dom_t[bi, i, :d].sum()) == 1.0:                 # pinned cell (singleton given)
                agg["pin_ok"] += int(match); agg["pin_tot"] += 1
            prog_good &= match
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                pm = bool((tp[bi, i, j, :d, :d] == tab_t[bi, i, j, :d, :d]).all())
                agg["pair_ok"] += int(pm); agg["pair_tot"] += 1
                true_con = bool((tab_t[bi, i, j, :d, :d] == 0).any())
                pred_con = bool((tp[bi, i, j, :d, :d] == 0).any())
                agg["cdet_tp"] += int(true_con and pred_con)
                agg["cdet_fp"] += int(pred_con and not true_con)
                agg["cdet_fn"] += int(true_con and not pred_con)
                if true_con:
                    agg["rel_ok"] += int(pm); agg["rel_tot"] += 1
                prog_good &= pm
        agg["prog_ok"] += int(prog_good); agg["prog_tot"] += 1
    return agg


def new_agg():
    return dict(cell_ok=0, cell_tot=0, pin_ok=0, pin_tot=0, pair_ok=0, pair_tot=0,
                cdet_tp=0, cdet_fp=0, cdet_fn=0, rel_ok=0, rel_tot=0, prog_ok=0, prog_tot=0)


def finalize(agg):
    tp, fp, fn = agg["cdet_tp"], agg["cdet_fp"], agg["cdet_fn"]
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    r = lambda a, b: agg[a] / max(1, agg[b])
    return {"program_exact": r("prog_ok", "prog_tot"), "domain_acc": r("cell_ok", "cell_tot"),
            "pin_acc": r("pin_ok", "pin_tot"), "table_exact": r("pair_ok", "pair_tot"),
            "constraint_f1": f1, "constraint_prec": prec, "constraint_rec": rec,
            "relation_type_acc": r("rel_ok", "rel_tot"), "n_problems": agg["prog_tot"]}


@torch.no_grad()
def evaluate(alpha, samples, tok, dev, bs=16):
    alpha.eval()
    agg = new_agg()
    for i in range(0, len(samples), bs):
        ba = encode_batch(samples[i:i + bs], tok, dev)
        dom_logits, tab_logits = alpha(ba)
        accumulate(dom_logits, tab_logits, ba, agg)
    return finalize(agg)


@torch.no_grad()
def evaluate_by_relation(alpha, samples, tok, dev, bs=16):
    out = {}
    for rel in MAPPABLE:
        sub = [s for s in samples if s["relation"] == rel]
        if sub:
            out[rel] = evaluate(alpha, sub, tok, dev, bs)
    return out


# ===================================================================== training
def train(alpha, tok, dev, steps, bs=16, lr=2e-4, w_dom=1.0, w_tab=1.0, diverse_frac=0.0,
          diverse_train=None, seed=0, log=None):
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(alpha.trainable_parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    log = log if log is not None else []
    alpha.train(); t0 = time.time()
    for s in range(1, steps + 1):
        nd = int(round(bs * diverse_frac)) if diverse_train else 0
        samples = sample_canonical(rng, bs - nd)
        if nd:
            idx = rng.integers(0, len(diverse_train), nd)
            samples += [diverse_train[i] for i in idx]
        ba = encode_batch(samples, tok, dev)
        dom_logits, tab_logits = alpha(ba)
        dl, tl = prog_loss(dom_logits, tab_logits, ba)
        loss = w_dom * dl + w_tab * tl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(alpha.trainable_parameters(), 1.0); opt.step()
        if s % max(1, steps // 20) == 0 or s == 1:
            agg = accumulate(dom_logits, tab_logits, ba, new_agg())
            m = finalize(agg)
            rec = {"step": s, "dom": float(dl), "tab": float(tl), "prog_exact": m["program_exact"],
                   "pin": m["pin_acc"], "cf1": m["constraint_f1"]}
            log.append(rec)
            print(f"  step {s:4d}  dom {rec['dom']:.4f} tab {rec['tab']:.4f}  "
                  f"prog-exact {rec['prog_exact']*100:5.1f}%  pin {rec['pin']*100:5.1f}%  "
                  f"cF1 {rec['cf1']*100:5.1f}%  {time.time()-t0:.0f}s", flush=True)
    return log


# ===================================================================== modes
def _print_table(name, m):
    print(f"  {name:22s}  prog-exact {m['program_exact']*100:5.1f}%  domain {m['domain_acc']*100:5.1f}%  "
          f"pin {m['pin_acc']*100:5.1f}%  table {m['table_exact']*100:5.1f}%  "
          f"cF1 {m['constraint_f1']*100:5.1f}%  rel-type {m['relation_type_acc']*100:5.1f}%  "
          f"(n={m['n_problems']})", flush=True)


def run_smoke(args, dev):
    print("\n===== SMOKE: standalone alpha compiler =====", flush=True)
    alpha, tok = load(compile_layer=args.compile_layer)
    alpha.to(dev)
    ntr = sum(p.numel() for p in alpha.trainable_parameters())
    print(f"  trainable={ntr:,}  compile@layer {alpha.compile_layer}  K={K} NMAX={NMAX}", flush=True)
    rng = np.random.default_rng(0)
    samples = sample_canonical(rng, 6)
    ba = encode_batch(samples, tok, dev)
    dom_logits, tab_logits = alpha(ba)
    dl, tl = prog_loss(dom_logits, tab_logits, ba)
    print(f"  forward OK: dom_logits {tuple(dom_logits.shape)} tab_logits {tuple(tab_logits.shape)}  "
          f"dom-loss {float(dl):.4f} tab-loss {float(tl):.4f}", flush=True)
    assert torch.isfinite(dl) and torch.isfinite(tl)
    log = train(alpha, tok, dev, steps=args.steps or 80, bs=args.bs)
    d0, d1 = log[0]["dom"], log[-1]["dom"]; t0, t1 = log[0]["tab"], log[-1]["tab"]
    print(f"  dom-loss {d0:.4f} -> {d1:.4f} (drops: {d1 < d0})", flush=True)
    print(f"  tab-loss {t0:.4f} -> {t1:.4f} (drops: {t1 < t0})", flush=True)
    ev = sample_canonical(np.random.default_rng(99), 48)
    m = evaluate(alpha, ev, tok, dev)
    _print_table("in-dist (canonical)", m)
    return {"smoke": {"log": log, "eval": m}}


def run_full(args, dev):
    print("\n===== FULL: standalone alpha compiler =====", flush=True)
    alpha, tok = load(compile_layer=args.compile_layer)
    alpha.to(dev)
    ntr = sum(p.numel() for p in alpha.trainable_parameters())
    print(f"  trainable={ntr:,}  compile@layer {alpha.compile_layer}  K={K} NMAX={NMAX}", flush=True)

    diverse = load_diverse(args.diverse)
    rng_split = np.random.default_rng(1234)
    perm = rng_split.permutation(len(diverse))
    n_test = max(1, int(0.2 * len(diverse)))
    test_idx, train_idx = set(perm[:n_test].tolist()), perm[n_test:].tolist()
    diverse_test = [diverse[i] for i in sorted(test_idx)]
    diverse_train = [diverse[i] for i in train_idx]
    print(f"  diverse records: {len(diverse)} mappable  ({len(diverse_train)} train / {len(diverse_test)} test)  "
          f"diverse_frac={args.diverse_frac}", flush=True)

    log = train(alpha, tok, dev, steps=args.steps, bs=args.bs, lr=args.lr,
                diverse_frac=args.diverse_frac, diverse_train=diverse_train if args.diverse_frac > 0 else None)

    out = {"config": {"steps": args.steps, "bs": args.bs, "lr": args.lr,
                      "compile_layer": alpha.compile_layer, "diverse_frac": args.diverse_frac,
                      "K": K, "NMAX": NMAX}, "log": log, "eval": {}, "by_relation": {}}

    print("\n  --- assembly-readiness table ---", flush=True)
    indist = sample_canonical(np.random.default_rng(7), args.neval)
    out["eval"]["in_dist_canonical"] = evaluate(alpha, indist, tok, dev)
    _print_table("in-dist canonical", out["eval"]["in_dist_canonical"])

    for N in [9, 10, 11]:
        ev = fixed_canonical(np.random.default_rng(1000 + N), args.neval, N)
        out["eval"][f"ood_size_N{N}"] = evaluate(alpha, ev, tok, dev)
        _print_table(f"OOD-size N={N} (col/eq)", out["eval"][f"ood_size_N{N}"])

    out["eval"]["ood_phrasing_diverse"] = evaluate(alpha, diverse_test, tok, dev)
    _print_table("OOD-phrasing (held-out)", out["eval"]["ood_phrasing_diverse"])

    print("\n  --- OOD-phrasing broken down by relation ---", flush=True)
    out["by_relation"]["ood_phrasing"] = evaluate_by_relation(alpha, diverse_test, tok, dev)
    for rel, m in out["by_relation"]["ood_phrasing"].items():
        _print_table(f"diverse:{rel}", m)

    print("\n  --- in-dist broken down by relation ---", flush=True)
    out["by_relation"]["in_dist"] = evaluate_by_relation(alpha, indist, tok, dev)
    for rel, m in out["by_relation"]["in_dist"].items():
        _print_table(f"canon:{rel}", m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--neval", type=int, default=256)
    ap.add_argument("--compile_layer", type=int, default=8)
    ap.add_argument("--diverse_frac", type=float, default=0.0,
                    help="fraction of each batch drawn from the diverse-train split (0 = strict standalone)")
    ap.add_argument("--diverse", default="data/curriculum/curriculum.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.steps:
        args.steps = 80 if args.mode == "smoke" else 2500
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    print(f"device={dev}  mode={args.mode}  steps={args.steps}", flush=True)
    out = run_smoke(args, dev) if args.mode == "smoke" else run_full(args, dev)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1, default=float)
        print("\nwrote", args.out, flush=True)


if __name__ == "__main__":
    main()
