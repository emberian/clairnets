"""clair/organ/run_alpha_struct_circuit.py — MECHANISTIC DIAGNOSIS of α's OOD-long-chain compile gap.

WHY does α compile OOD long chains nearly-but-not-exactly (struct-F1 0.78)? The prime architectural
suspect: CSPStructureHead (alpha_struct.py) emits the [N,N] pair-relation grid as INDEPENDENT per-pair
classifications — nothing enforces global structural consistency (transitivity along a chain, message-
passing among edge decisions). On long chains, independent per-pair errors COMPOUND. "Nearly-but-not-
exactly" is the signature of independent-decision accumulation, NOT necessarily a capacity ceiling — if so
DoRA / bigger-LoRA / even full-FT won't fix it; message-passing / consistency at the edge head will.

This driver runs the clean ARCHITECTURE-vs-REPRESENTATION disentangler on FROZEN OLMo hidden (no LoRA, no
composer, no LM — the purest test), over the hard CSP curriculum incl. the OOD long chains (eqchain/
forcedcolor, L up to 10):

  1. THREE edge heads on the SAME frozen hidden, trained identically:
       - INDEP  : the production CSPStructureHead (independent per-pair MLP) — α's actual architecture.
       - LINEAR : a PURE-LINEAR per-pair probe on the two cells' pooled hidden — "is the edge linearly
                  decodable from the host hidden at all?" (the representation floor; info-present test).
       - MP     : the independent head + 1 round of MESSAGE PASSING over the edge grid (each edge sees its
                  incident edges) — the global-consistency architecture. INDEP vs MP isolates independence.
  2. LENGTH CURVE: micro-F1 + exact-factor-graph-match vs chain length N (monotonic compounding vs cliff),
       both IN-DIST (trained on all lengths) and EXTRAPOLATION (trained ≤6, eval ≥8).
  3. ERROR STRUCTURE: per-relation P/R/F1; recall-vs-gap and false-positive-vs-gap histograms (are missed/
       hallucinated edges RANDOM, or CLUSTERED by chain-gap = the transitive/long-range signature).

VERDICT: if LINEAR recovers the edges INDEP misses AND MP >> INDEP on long chains → ARCHITECTURE-bound
(independent-edge head; fix = edge-head message-passing/consistency, NOT adapters). If LINEAR also fails →
REPRESENTATION-bound (host can't expose it; host-adapt / more layers is the lever).

  python -m clair.organ.run_alpha_struct_circuit --layers 6 --steps 700 --per_rung 600 \
      --json_out runs/alpha_struct_circuit.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import curriculum as CU
from .alpha_struct import CellEncoder, CSPStructureHead, PAIR_RELS, R_PAIR, REL_IDX
from .run_alpha_struct_probe import precompute, csp_targets, decode_facts, f1_report

K = 8  # D_MAX readout width


# ============================================================ edge-head variants
class LinearProbe(nn.Module):
    """PURE-LINEAR per-pair probe on the two cells' MENTION-POOLED hidden (no cross-attn, no MLP, no MP).
    Tests whether the typed edge is LINEARLY decodable from the host hidden — the representation floor."""
    def __init__(self, D, K):
        super().__init__()
        self.pin = nn.Linear(D, 1 + K)
        self.pair = nn.Linear(2 * D, R_PAIR)

    def forward(self, vmean):                       # vmean [B,N,D]
        B, N, D = vmean.shape
        pin_l = self.pin(vmean)                      # [B,N,1+K]
        a = vmean[:, :, None, :].expand(B, N, N, D)
        b = vmean[:, None, :, :].expand(B, N, N, D)
        pair_l = self.pair(torch.cat([a, b], dim=-1))  # [B,N,N,R]
        return pin_l, pair_l


class MPPairHead(nn.Module):
    """Independent edge head + ROUNDS of MESSAGE PASSING over the [N,N] edge grid. Each round refines edge
    e_ij from its hidden PLUS the aggregated edges OUT of i and INTO j (one hop → transitivity reach), so
    edge decisions stop being independent. INDEP (mp_rounds=0 path) vs MP isolates the independence flaw."""
    def __init__(self, din, K, hidden=384, ph=256, mp_rounds=1):
        super().__init__()
        self.pin = nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, 1 + K))
        self.pl = nn.Linear(din, ph)
        self.pr = nn.Linear(din, ph)
        self.edge_in = nn.Sequential(nn.Linear(2 * ph, ph), nn.GELU())     # base edge hidden
        self.mp_rounds = mp_rounds
        self.mp = nn.ModuleList([nn.Sequential(nn.Linear(3 * ph, ph), nn.GELU()) for _ in range(mp_rounds)])
        self.out = nn.Linear(ph, R_PAIR)

    def forward(self, feat, vmask):                 # feat [B,N,din], vmask [B,N]
        B, N, _ = feat.shape
        pin_l = self.pin(feat)
        li, rj = self.pl(feat), self.pr(feat)
        ph = li.size(-1)
        e = self.edge_in(torch.cat([li[:, :, None, :].expand(B, N, N, ph),
                                    rj[:, None, :, :].expand(B, N, N, ph)], dim=-1))   # [B,N,N,ph]
        m = vmask[:, None, :, None]                  # mask over the aggregated index k (valid cells)
        for layer in self.mp:
            denom = vmask.sum(-1).clamp_min(1.0)[:, None, None]          # [B,1,1]
            agg_out_i = (e * m).sum(2) / denom       # mean_k e[i,k]   [B,N,ph]  (edges OUT of i)
            agg_in_j = (e * vmask[:, :, None, None]).sum(1) / denom      # mean_k e[k,j]  (edges INTO j)
            ai = agg_out_i[:, :, None, :].expand(B, N, N, ph)
            bj = agg_in_j[:, None, :, :].expand(B, N, N, ph)
            e = e + layer(torch.cat([e, ai, bj], dim=-1))               # residual MP update
        return pin_l, self.out(e)


# ============================================================ data: hard-curriculum pool spanning lengths
def build_circuit_pool(per_rung, splits=("id", "ood"), seed=0):
    """eqchain+forcedcolor problems spanning lengths (id L∈3..6, ood L∈8,10). Returns views consumable by
    the frozen-host probe: (alpha_text, mentions{int:spans}, n, expanded_true_facts). α reads the FULL
    text (the constraints are lexicalized); the binary+pin facts are the structure-supervision target."""
    from .. import run_glados_staged as G
    rng = np.random.default_rng(seed)
    views = []
    for rg in ("eqchain", "forcedcolor"):
        for split in splits:
            got = tries = 0
            while got < per_rung and tries < per_rung * 60:
                tries += 1
                try:
                    p = G.make_problem(rng, rg, split)
                except Exception:
                    continue
                if not p.determined:
                    # keep a mix; determined not required for structure-reading, but match the gate pool
                    pass
                facts = [tuple(f) if f[0] != "alldiff" else ("alldiff", tuple(f[1])) for f in p.facts]
                exp = []
                import itertools as it
                for f in facts:
                    if f[0] == "alldiff":
                        exp += [("neq", a, b) for a, b in it.combinations(f[1], 2)]
                    else:
                        exp.append(f)
                if any(f[0] not in ("pin", "eq", "neq", "lt", "le") for f in exp):
                    continue
                text = CU.canonical_render(p)
                ment = CU.entity_mentions(text, p.n)
                # need a mention for every cell that appears in a fact or the query
                need = {p.query}
                for f in exp:
                    need |= set(f[1:])
                if not need <= set(ment):
                    continue
                views.append((text, ment, p.n, exp, rg, split))
                got += 1
    rng.shuffle(views)
    return views


# ============================================================ generic frozen-host head trainer
def _targets(views, Nmax, dev):
    return csp_targets([(t, m, n, f) for (t, m, n, f, _, _) in views], Nmax, dev)


def train_head(kind, pre, train_idx, pin_tgt, pair_tgt, D, dev, L, steps=700, lr=1.5e-3,
               wpos=2.0, wnone=1.0, bs=48, mp_rounds=1, log=print):
    """Train one edge head (kind∈{indep,linear,mp}) on frozen hidden. Soundness-asymmetric CE (wpos on
    relation/pin classes = recall; wnone on none/no-pin = precision). Returns the trained modules."""
    Nmax = pre["Nmax"]
    if kind == "linear":
        head = LinearProbe(D, K).to(dev).float(); enc = None
        params = list(head.parameters())
    elif kind == "indep":
        enc = CellEncoder(D).to(dev).float()
        head = CSPStructureHead(enc.din, K).to(dev).float()
        params = list(enc.parameters()) + list(head.parameters())
    else:  # mp
        enc = CellEncoder(D).to(dev).float()
        head = MPPairHead(enc.din, K, mp_rounds=mp_rounds).to(dev).float()
        params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    pin_w = torch.full((1 + K,), wpos, device=dev); pin_w[0] = wnone
    pair_w = torch.full((R_PAIR,), wpos, device=dev); pair_w[0] = wnone
    rng = np.random.default_rng(0)
    t0 = time.time()
    for s in range(1, steps + 1):
        sub = [int(x) for x in rng.choice(train_idx, size=min(bs, len(train_idx)), replace=False)]
        h = pre["H"][L][sub].to(dev); attn = pre["attn"][sub].to(dev)
        vmean = pre["vmean"][L][sub].to(dev); vmask = pre["vmask"][sub].to(dev)
        if kind == "linear":
            pin_l, pair_l = head(vmean)
        else:
            feat = enc(vmean, h, attn)
            pin_l, pair_l = (head(feat) if kind == "indep" else head(feat, vmask))
        B, Nm, _ = pin_l.shape
        cellm = vmask.bool()
        lp = F.cross_entropy(pin_l[cellm], pin_tgt[sub][cellm], weight=pin_w)
        pm = (vmask[:, :, None] * vmask[:, None, :]).bool()
        eye = torch.eye(Nm, device=dev, dtype=torch.bool)[None].expand(B, Nm, Nm)
        pm = pm & ~eye
        lpr = F.cross_entropy(pair_l[pm], pair_tgt[sub][pm], weight=pair_w)
        loss = lp + lpr
        opt.zero_grad(); loss.backward(); opt.step()
        if s % max(1, steps // 5) == 0 or s == 1:
            log(f"    [{kind} L{L}] step {s:4d} pin {lp.item():.3f} pair {lpr.item():.3f} {time.time()-t0:.0f}s")
    return enc, head


@torch.no_grad()
def head_logits(kind, enc, head, pre, idxs, L, dev, bs=64):
    """pin/pair logits over eval records `idxs` for a trained head."""
    Nmax = pre["Nmax"]
    pin_out = torch.zeros(len(idxs), Nmax, 1 + K, device=dev)
    pair_out = torch.zeros(len(idxs), Nmax, Nmax, R_PAIR, device=dev)
    pos = {ri: k for k, ri in enumerate(idxs)}
    for i in range(0, len(idxs), bs):
        sub = idxs[i:i + bs]
        h = pre["H"][L][sub].to(dev); attn = pre["attn"][sub].to(dev)
        vmean = pre["vmean"][L][sub].to(dev); vmask = pre["vmask"][sub].to(dev)
        if kind == "linear":
            pin_l, pair_l = head(vmean)
        else:
            feat = enc(vmean, h, attn)
            pin_l, pair_l = (head(feat) if kind == "indep" else head(feat, vmask))
        for k, ri in enumerate(sub):
            pin_out[pos[ri]] = pin_l[k]; pair_out[pos[ri]] = pair_l[k]
    return pin_out, pair_out


# ============================================================ analyses
def length_curve(pin_l, pair_l, pre, views, idxs):
    """micro-F1 + exact-match binned by chain length n."""
    by_n = defaultdict(list)
    for ri in idxs:
        by_n[int(pre["n"][ri].item())].append(ri)
    out = {}
    for n in sorted(by_n):
        rep = f1_report(None, None, pin_l, pair_l, pre["vmask"], pre["n"],
                        [(t, m, nn, f) for (t, m, nn, f, _, _) in views], by_n[n])
        out[n] = {"micro_F1": rep["micro"]["F1"], "exact_match": rep["exact_factor_graph_match"],
                  "n_recs": len(by_n[n])}
    return out


def error_structure(pin_l, pair_l, pre, views, idxs):
    """Recall-vs-gap and false-positive-vs-gap over BINARY edges (gap=|i-j|). Reveals whether missed/
    hallucinated edges are random or clustered by chain-distance (the transitive/long-range signature)."""
    rec_by_gap = defaultdict(lambda: [0, 0])    # gap -> [recalled, total_true]
    fp_by_gap = defaultdict(int)                # gap -> hallucinated count
    miss_by_pos = defaultdict(lambda: [0, 0])   # normalized chain position bucket -> [missed, total]
    for ri in idxs:
        n = int(pre["n"][ri].item())
        pred = set(decode_facts(pin_l[ri], pair_l[ri], pre["vmask"][ri], n))
        true = set(CU.norm_facts(views[ri][3]))
        tb = {f for f in true if f[0] in ("eq", "neq", "lt", "le")}
        pb = {f for f in pred if f[0] in ("eq", "neq", "lt", "le")}
        for f in tb:
            gap = abs(int(f[1]) - int(f[2]))
            recalled = int(f in pb)
            rec_by_gap[gap][0] += recalled; rec_by_gap[gap][1] += 1
            posb = int(5 * min(int(f[1]), int(f[2])) / max(1, n - 1))   # 0..5 along chain
            miss_by_pos[posb][0] += (1 - recalled); miss_by_pos[posb][1] += 1
        for f in pb - tb:
            fp_by_gap[abs(int(f[1]) - int(f[2]))] += 1
    return {
        "recall_by_gap": {g: {"recall": round(v[0] / max(1, v[1]), 4), "n": v[1]}
                          for g, v in sorted(rec_by_gap.items())},
        "false_pos_by_gap": {g: c for g, c in sorted(fp_by_gap.items())},
        "miss_by_chainpos": {p: {"miss_rate": round(v[0] / max(1, v[1]), 4), "n": v[1]}
                             for p, v in sorted(miss_by_pos.items())},
    }


def full_report(pin_l, pair_l, pre, views, idxs):
    return f1_report(None, None, pin_l, pair_l, pre["vmask"], pre["n"],
                     [(t, m, n, f) for (t, m, n, f, _, _) in views], idxs)


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--layers", default="6")
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--wpos", type=float, default=2.0)
    ap.add_argument("--wnone", type=float, default=1.0)
    ap.add_argument("--per_rung", type=int, default=600, help="problems per (rung,split) for the pool")
    ap.add_argument("--mp_rounds", type=int, default=1)
    ap.add_argument("--json_out", default="runs/alpha_struct_circuit.json")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[circuit] loading {a.model} on {dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    olmo = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    for p in olmo.parameters():
        p.requires_grad_(False)
    D = olmo.config.hidden_size
    print(f"[circuit] OLMo hidden {D}", flush=True)

    # build pools: train+eval span all lengths; a held-out 20% eval; mark short(<=6)/long(>=8)
    pool = build_circuit_pool(a.per_rung, splits=("id", "ood"), seed=0)
    print(f"[circuit] pool {len(pool)} views over eqchain/forcedcolor (id+ood, L=3..10)", flush=True)
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(pool))
    nval = len(pool) // 5
    val_v = [pool[i] for i in perm[:nval]]
    tr_v = [pool[i] for i in perm[nval:]]
    short_tr = [v for v in tr_v if v[2] <= 6]                  # extrapolation arm: train short
    print(f"[circuit] train {len(tr_v)} (short {len(short_tr)})  val {len(val_v)}", flush=True)

    results = {"model": a.model, "D": D, "layers": layers, "per_rung": a.per_rung,
               "n_train": len(tr_v), "n_val": len(val_v), "mp_rounds": a.mp_rounds}

    for L in layers:
        print(f"\n===== layer {L} =====", flush=True)
        pre_tr = precompute(olmo, tok, dev, [(t, m, n) for (t, m, n, _, _, _) in tr_v], [L])
        pre_va = precompute(olmo, tok, dev, [(t, m, n) for (t, m, n, _, _, _) in val_v], [L])
        Nmax = max(pre_tr["Nmax"], pre_va["Nmax"])
        # pad both to common Nmax via re-precompute is unneeded (targets sized to each pre's Nmax);
        # build targets per-pre.
        pin_tr, pair_tr = _targets(tr_v, pre_tr["Nmax"], dev)
        val_idx = list(range(len(val_v)))

        layer_res = {}
        for kind in ("indep", "linear", "mp"):
            print(f"\n  -- training {kind} (trained on ALL lengths) --", flush=True)
            enc, head = train_head(kind, pre_tr, list(range(len(tr_v))), pin_tr, pair_tr, D, dev, L,
                                   steps=a.steps, lr=a.lr, wpos=a.wpos, wnone=a.wnone,
                                   mp_rounds=a.mp_rounds)
            pin_l, pair_l = head_logits(kind, enc, head, pre_va, val_idx, L, dev)
            rep = full_report(pin_l, pair_l, pre_va, val_v, val_idx)
            lc = length_curve(pin_l, pair_l, pre_va, val_v, val_idx)
            es = error_structure(pin_l, pair_l, pre_va, val_v, val_idx)
            layer_res[kind] = {"overall": rep, "length_curve": lc, "error_structure": es}
            print(f"  [{kind}] micro-F1 {rep['micro']['F1']:.3f} exact {rep['exact_factor_graph_match']:.3f} "
                  f"| F1 by len " + " ".join(f"{n}:{lc[n]['micro_F1']:.2f}" for n in sorted(lc)), flush=True)

        # EXTRAPOLATION: train indep + mp on SHORT (<=6), eval on LONG (>=8) — length generalization
        long_idx = [i for i in val_idx if int(pre_va["n"][i].item()) >= 8]
        extra = {}
        if short_tr and long_idx:
            pin_s, pair_s = _targets([v for v in tr_v if v[2] <= 6], pre_tr["Nmax"], dev)
            short_train_idx = [i for i, v in enumerate(tr_v) if v[2] <= 6]
            for kind in ("indep", "mp"):
                print(f"\n  -- training {kind} on SHORT(<=6), eval LONG(>=8) [extrapolation] --", flush=True)
                enc, head = train_head(kind, pre_tr, short_train_idx, pin_s, pair_s, D, dev, L,
                                       steps=a.steps, lr=a.lr, wpos=a.wpos, wnone=a.wnone,
                                       mp_rounds=a.mp_rounds)
                pin_l, pair_l = head_logits(kind, enc, head, pre_va, long_idx, L, dev)
                rep = full_report(pin_l, pair_l, pre_va, val_v, long_idx)
                lc = length_curve(pin_l, pair_l, pre_va, val_v, long_idx)
                extra[kind] = {"long_overall": rep, "length_curve": lc}
                print(f"  [{kind}/extrap] LONG micro-F1 {rep['micro']['F1']:.3f} "
                      f"exact {rep['exact_factor_graph_match']:.3f}", flush=True)
        layer_res["extrapolation_short2long"] = extra
        results[f"layer_{L}"] = layer_res

    # ---- verdict synthesis (on the best/last layer) ----
    L = layers[-1]
    lr_ = results[f"layer_{L}"]
    indep_f1 = lr_["indep"]["overall"]["micro"]["F1"]
    linear_f1 = lr_["linear"]["overall"]["micro"]["F1"]
    mp_f1 = lr_["mp"]["overall"]["micro"]["F1"]
    # long-chain (n>=8) slice
    def long_f1(kind):
        lc = lr_[kind]["overall"]["length_curve"]
        longs = [lc[n]["micro_F1"] for n in lc if n >= 8]
        return float(np.mean(longs)) if longs else float("nan")
    verdict = {
        "indep_micro_F1": indep_f1, "linear_micro_F1": linear_f1, "mp_micro_F1": mp_f1,
        "indep_long_F1": long_f1("indep"), "linear_long_F1": long_f1("linear"), "mp_long_F1": long_f1("mp"),
        "mp_minus_indep_long": long_f1("mp") - long_f1("indep"),
        "note": "MP>>INDEP on long chains ⇒ independence (ARCHITECTURE-bound); LINEAR≈0 on missed edges ⇒ "
                "REPRESENTATION-bound. Compare against the adapter sweep (run_alpha_struct_ab).",
    }
    results["verdict"] = verdict
    import os
    os.makedirs(os.path.dirname(os.path.abspath(a.json_out)) or ".", exist_ok=True)
    with open(a.json_out, "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 78, flush=True)
    print("CIRCUIT VERDICT (frozen host, edge-head architecture test)", flush=True)
    print(f"  INDEP  micro-F1 {indep_f1:.3f}  long(>=8) {verdict['indep_long_F1']:.3f}", flush=True)
    print(f"  LINEAR micro-F1 {linear_f1:.3f}  long(>=8) {verdict['linear_long_F1']:.3f}", flush=True)
    print(f"  MP     micro-F1 {mp_f1:.3f}  long(>=8) {verdict['mp_long_F1']:.3f}", flush=True)
    print(f"  >> MP - INDEP on long chains: {verdict['mp_minus_indep_long']:+.3f}  "
          f"(>0 ⇒ independence is the limiter ⇒ ARCHITECTURE-bound)", flush=True)
    print(f"[circuit] wrote {a.json_out}", flush=True)


if __name__ == "__main__":
    main()
