"""clair/organ/run_alpha_struct_rebuild.py — RETRAIN+MEASURE the two α-structure-compiler fixes.

The α structure compiler (alpha_struct.StructureRack / CSPStructureHead) had two confirmed deficiencies,
read off the source and the OOD-long-chain diagnosis (run_alpha_struct_circuit, struct-F1 0.78):

  FINDING 1  NO cell-to-cell attention. The pair head is a pure dyadic MLP(cat(Wl·feat_i, Wr·feat_j))
             over N² INDEPENDENT pairs; feat reads the PROMPT, never other cells. On long chains the
             independent per-pair errors COMPOUND (transitivity is unrepresentable) ⇒ long-chain
             binding/consistency failure. FIX: a few masked self-attention NLayers over the [B,N,din]
             cell set BEFORE the heads (CellSelfAttention) ⇒ cell-contextualized feat_i/feat_j.

  FINDING 2  BINARY-ONLY relation vocabulary {none,eq,neq,lt,le}. Ternary parity/xor is INEXPRESSIBLE,
             so the certified GF2RowSpace faculty (the affine wall) is unreachable from α — an
             α-EXPRESSIVITY gap, not a solver gap. FIX: an arity-3 TernaryFactorHead emitting
             {none, even-parity, odd-parity} per triple ⇒ XOR/affine systems become emittable and route
             to GF2.

This driver retrains the α structure probe on FROZEN OLMo-2-1B hidden (no LoRA/composer/LM — the purest
architecture test) and MEASURES both fixes against the baseline architecture, with the 2×2 ablation
(backbone off/on × ternary off/on) to attribute the gains.

  ARM 1 (FINDING 1): the hard CSP curriculum incl. OOD long chains (eqchain/forcedcolor, L=3..10).
    base (backbone off, == the production independent-pair head, the F1-0.78 architecture) vs +backbone.
    Per-chain-length struct-F1 + exact-graph-match curve, in-dist AND short→long extrapolation.

  ARM 2 (FINDING 2): xor + affine (GF(2)) instances. binary (ternary off) vs +ternary.
    parity-fact P/R/F1; and the make-or-break: compile α's emitted facts → GF(2) (A,b) → the certified
    GF2RowSpace faculty → does the query get SOLVED? (binary α cannot express parity ⇒ rank-deficient ⇒
    unsolved; ternary α expresses it ⇒ GF2 forces the answer.) 2×2 ablation backbone×ternary.

  python -m clair.organ.run_alpha_struct_rebuild --layers 6 --steps 700 --per_rung 600 \
      --xor_per 700 --out runs/alpha_struct_rebuild.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from .. import curriculum as CU
from .. import xor_wall as XW
from .. import csp as C
from ..organ.protocol import CSPState
from ..organ.bank import GF2RowSpace
from .alpha_struct import (StructureRack, PAIR_RELS, R_PAIR, REL_IDX, TERN_RELS, R_TERN, TERN_IDX,
                           enum_triples, decode_structure, facts_to_ternary_targets,
                           ternary_sup_loss, facts_to_gf2, build_csp_from_struct)
from .run_alpha_struct_probe import precompute, csp_targets, decode_facts, f1_report
from .run_alpha_struct_circuit import build_circuit_pool, length_curve

K = 8


# ====================================================================== ARM-2 data: xor / affine pool
def build_xor_pool(per_kind, seed=0):
    """xor (arity-3 boolean parity, n=3..6) + affine (GF(2) parity SYSTEM, n=5..11) instances rendered
    to canonical English. Returns (views, meta): views=(text, mentions, n, exp_facts) for the shared
    probe machinery; meta carries the witness s, query, true answer, kind. Query is forced DETERMINED so
    a sound GF(2) solve must recover it (the make-or-break for FINDING 2)."""
    rng = np.random.default_rng(seed)
    views, meta = [], []

    def add(n, d, kind_dom, facts, s, tag):
        # normalize parity facts: 'xor'/'par' → 'parm' rhs (the head's canonical arity-3 parity form)
        exp = []
        for f in facts:
            if f[0] == "xor":
                exp.append(("parm", tuple(sorted(f[1:4])), 0))
            elif f[0] == "par":
                exp.append(("parm", tuple(sorted(f[1])), 0))
            else:
                exp.append(tuple(f))
        csp = CU.build_csp(n, d, facts)
        q, ans, det = CU._query_answer(csp, facts, rng, det_target=True)
        if not det:                                     # need a determined query for the GF2 solve test
            return False
        p = CU.Problem(tag, n, d, kind_dom, facts, q, ans, det, CU.value_names(kind_dom, d))
        text = CU.canonical_render(p)
        ment = CU.entity_mentions(text, n)
        need = {q}
        for f in exp:
            need |= set(f[1]) if f[0] == "parm" else set(f[1:])
        if not need <= set(ment):
            return False
        views.append((text, ment, n, exp))
        meta.append({"s": [int(x) for x in s], "query": int(q), "answer": int(ans), "kind": tag, "n": n})
        return True

    got = tries = 0
    while got < per_kind and tries < per_kind * 60:
        tries += 1
        n, d, kd, facts, s = CU.gen_xor(rng, n_lo=3, n_hi=6)
        got += add(n, d, kd, facts, s, "xor")
    got = tries = 0
    while got < per_kind and tries < per_kind * 40:
        tries += 1
        # FAST arity-3 GF(2) parity SYSTEM: feed-forward parity DAG, uniquely solvable, NO exact-solver
        # hardness gate (gen_affine's gate is pure-Python-exponential on the Rust-less box). Longer
        # systems (n up to 11, banded width) ⇒ the affine wall the GF2 faculty owns.
        n = int(rng.integers(6, 12)); band = int(rng.choice([3, 4, 5]))
        sysd = XW.gen_tree_xor_unique(rng, n, band)
        s = np.asarray(sysd["s"]).astype(int)
        facts = []
        for sc, al in sysd["csp"].cons:
            if len(sc) == 3:
                rhs = (int(s[sc[0]]) + int(s[sc[1]]) + int(s[sc[2]])) % 2
                facts.append(("parm", tuple(int(x) for x in sc), rhs))
            elif len(sc) == 1:
                facts.append(("pin", int(sc[0]), int(s[sc[0]])))
        if len(facts) < 2:
            continue
        got += add(n, 2, "number", facts, s, "affine")
    order = rng.permutation(len(views))
    return [views[i] for i in order], [meta[i] for i in order]


def csp_targets_safe(views, Nmax, dev):
    """pin/pair targets like run_alpha_struct_probe.csp_targets, but SKIPS arity-3 parity facts (handled
    by the ternary head) instead of KeyError-ing on them. Pair targets stay all-'none' for a pure-parity
    pool — exactly right: the binary pair head must NOT fire on a parity constraint."""
    R = len(views)
    pin = torch.zeros(R, Nmax, dtype=torch.long)
    pair = torch.zeros(R, Nmax, Nmax, dtype=torch.long)
    for ri, (_, _, n, facts) in enumerate(views):
        for f in facts:
            k = f[0]
            if k == "pin":
                pin[ri, f[1]] = f[2] + 1
            elif k in ("eq", "neq"):
                a, b = f[1], f[2]
                pair[ri, a, b] = REL_IDX[k]; pair[ri, b, a] = REL_IDX[k]
            elif k in ("lt", "le"):
                pair[ri, f[1], f[2]] = REL_IDX[k]
            # parm/xor/par → ternary head; rel/sum → deferred
    return pin.to(dev), pair.to(dev)


# ====================================================================== unified rack trainer / forward
def _params(rack, use_tern):
    ps = list(rack.encoder.parameters()) + list(rack.heads["csp"].parameters())
    if rack.backbone is not None:
        ps += list(rack.backbone.parameters())
    if use_tern and rack.heads["ternary"] is not None:
        ps += list(rack.heads["ternary"].parameters())
    return ps


def train_rack(rack, pre, dev, train_idx, pin_tgt, pair_tgt, L, steps, lr, wpos, wnone,
               tern_tgt=None, triples=None, tri_valid=None, bs=48, bb_lr_mult=1.0, log=print):
    """Train encoder(+backbone)+csp head (+ternary head if tern_tgt given). Soundness-asymmetric CE.
    `bb_lr_mult` lets the deeper backbone learn at a higher LR than the heads (it converges slower)."""
    use_tern = tern_tgt is not None
    if rack.backbone is not None and bb_lr_mult != 1.0:
        bb = set(map(id, rack.backbone.parameters()))
        rest = [p for p in _params(rack, use_tern) if id(p) not in bb]
        groups = [{"params": rest, "lr": lr},
                  {"params": list(rack.backbone.parameters()), "lr": lr * bb_lr_mult}]
        opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW(_params(rack, use_tern), lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    pin_w = torch.full((1 + K,), wpos, device=dev); pin_w[0] = wnone
    pair_w = torch.full((R_PAIR,), wpos, device=dev); pair_w[0] = wnone
    rng = np.random.default_rng(0)
    rack.train(); t0 = time.time()
    train_idx = list(train_idx)
    for s in range(1, steps + 1):
        sub = [int(x) for x in rng.choice(train_idx, size=min(bs, len(train_idx)), replace=False)]
        h = pre["H"][L][sub].to(dev); attn = pre["attn"][sub].to(dev)
        vmean = pre["vmean"][L][sub].to(dev); vmask = pre["vmask"][sub].to(dev)
        feat = rack.featurize(vmean, h, attn, vmask)
        pin_l, pair_l = rack.csp(feat, vmask)
        B, Nm, _ = pin_l.shape
        cellm = vmask.bool()
        lp = F.cross_entropy(pin_l[cellm], pin_tgt[sub][cellm], weight=pin_w)
        pm = (vmask[:, :, None] * vmask[:, None, :]).bool()
        eye = torch.eye(Nm, device=dev, dtype=torch.bool)[None].expand(B, Nm, Nm)
        pm = pm & ~eye
        lpr = F.cross_entropy(pair_l[pm], pair_tgt[sub][pm], weight=pair_w)
        loss = lp + lpr
        lt = torch.tensor(0.0, device=dev)
        if use_tern:
            tern_l = rack.ternary(feat, triples)        # [B,T,R_TERN]
            lt = ternary_sup_loss(tern_l, tern_tgt[sub], tri_valid[sub], wpos=wpos, wnone=wnone)
            loss = loss + lt
        opt.zero_grad(); loss.backward(); opt.step()
        if s % max(1, steps // 5) == 0 or s == 1:
            log(f"      step {s:4d}  pin {lp.item():.3f} pair {lpr.item():.3f} tern {float(lt):.3f}  "
                f"{time.time()-t0:.0f}s")
    return rack


@torch.no_grad()
def forward_all(rack, pre, dev, idxs, L, triples=None, use_tern=False, bs=64):
    """Logits positioned at the ORIGINAL record id (so f1_report/length_curve/gf2_solve, which index by
    ri, work with any idxs subset). Allocates full R-sized tensors and fills the requested rows."""
    Nmax = pre["Nmax"]; R = pre["vmask"].shape[0]
    pin_out = torch.zeros(R, Nmax, 1 + K, device=dev)
    pair_out = torch.zeros(R, Nmax, Nmax, R_PAIR, device=dev)
    tern_out = torch.zeros(R, len(triples) if triples is not None else 0, R_TERN, device=dev)
    rack.eval()
    idxs = list(idxs)
    for i in range(0, len(idxs), bs):
        sub = idxs[i:i + bs]
        h = pre["H"][L][sub].to(dev); attn = pre["attn"][sub].to(dev)
        vmean = pre["vmean"][L][sub].to(dev); vmask = pre["vmask"][sub].to(dev)
        feat = rack.featurize(vmean, h, attn, vmask)
        pin_l, pair_l = rack.csp(feat, vmask)
        for k, ri in enumerate(sub):
            pin_out[ri] = pin_l[k]; pair_out[ri] = pair_l[k]
        if use_tern:
            tern_l = rack.ternary(feat, triples)
            for k, ri in enumerate(sub):
                tern_out[ri] = tern_l[k]
    return pin_out, pair_out, tern_out


# ====================================================================== ARM-2 metrics: parity-F1 + GF2 solve
def parity_f1(tern_l_all, triples, pre, meta, idxs):
    """P/R/F1 over the arity-3 parity facts α emitted vs the true parm facts (FINDING-2 expressibility)."""
    tp = fp = fn = 0
    for ri in idxs:
        n = int(pre["n"][ri].item())
        vmask = pre["vmask"][ri]
        tcls = tern_l_all[ri].argmax(-1)
        pred = set()
        for t in range(len(triples)):
            i, j, k = (int(x) for x in triples[t])
            if max(i, j, k) >= n or vmask[i] < 0.5 or vmask[j] < 0.5 or vmask[k] < 0.5:
                continue
            cls = TERN_RELS[int(tcls[t])]
            if cls != "none":
                rhs = 0 if cls == "par_even" else 1
                pred.add(("parm", (i, j, k), rhs))
        true = {("parm", tuple(sorted(f[1])), int(f[2])) for f in _meta_facts(pre, meta, ri)
                if f[0] == "parm"}
        tp += len(pred & true); fp += len(pred - true); fn += len(true - pred)
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return {"P": round(p, 4), "R": round(r, 4), "F1": round(2 * p * r / max(1e-9, p + r), 4),
            "tp": tp, "fp": fp, "fn": fn}


def _meta_facts(pre, meta, ri):
    return META_FACTS[ri]


def gf2_solve(pin_l_all, pair_l_all, tern_l_all, triples, pre, meta, idxs, use_tern):
    """The make-or-break FINDING-2 test: decode α's structure → GF(2) (A,b) → the certified GF2RowSpace
    faculty → is the (determined) query SOLVED to the true answer? Reports the solve-rate, the number of
    parity rows α emitted (binary α emits 0 ⇒ rank-deficient ⇒ unsolved), and how often α's compiled CSP
    routes to GF2 at all (a parity constraint present)."""
    gf2 = GF2RowSpace()
    n_solved = n_routed = n_parity_emitted = 0
    tot = 0
    for ri in idxs:
        n = int(pre["n"][ri].item()); vmask = pre["vmask"][ri]
        tern_i = tern_l_all[ri] if use_tern else None
        facts = decode_structure(pin_l_all[ri], pair_l_all[ri], vmask, n, d=2,
                                 tern_logits=tern_i, triples=(triples if use_tern else None))
        n_par = sum(1 for f in facts if f[0] == "parm")
        n_parity_emitted += n_par
        n_routed += int(n_par > 0)
        A, b = facts_to_gf2(facts, n)
        # route the compiled parity CSP through the certified GF2RowSpace faculty (system=('gf2',A,b))
        csp = build_csp_from_struct(facts, n, 2)
        st = CSPState.full(csp, system=("gf2", A, b), tags={"xor"})
        out = gf2.reduce(st)
        q, ans = meta[ri]["query"], meta[ri]["answer"]
        solved = (q < out.csp.n and len(out.dom[q]) == 1 and ans in out.dom[q])
        n_solved += int(solved)
        tot += 1
    return {"n": tot, "query_solve_rate": round(n_solved / max(1, tot), 4),
            "gf2_routed_rate": round(n_routed / max(1, tot), 4),
            "parity_facts_emitted": n_parity_emitted}


# ====================================================================== main
META_FACTS = {}    # ri -> expanded true facts (set at pool build); read by parity metrics


def run_arm1(olmo, tok, dev, D, layers, per_rung, steps, lr, wpos, wnone, log,
             backbone_mode="wide", backbone_layers=3, bb_lr_mult=1.0):
    """FINDING 1: long-chain CSP, base (no backbone) vs +backbone, per-length curve + extrapolation."""
    pool = build_circuit_pool(per_rung, splits=("id", "ood"), seed=0)
    log(f"[arm1] circuit pool {len(pool)} (eqchain/forcedcolor, L=3..10)")
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(pool)); nval = len(pool) // 5
    val_v = [pool[i] for i in perm[:nval]]; tr_v = [pool[i] for i in perm[nval:]]
    L = layers[0]
    pre_tr = precompute(olmo, tok, dev, [(t, m, n) for (t, m, n, _, _, _) in tr_v], [L])
    pre_va = precompute(olmo, tok, dev, [(t, m, n) for (t, m, n, _, _, _) in val_v], [L])
    tv = [(t, m, n, f) for (t, m, n, f, _, _) in tr_v]
    vv = [(t, m, n, f) for (t, m, n, f, _, _) in val_v]
    pin_tr, pair_tr = csp_targets(tv, pre_tr["Nmax"], dev)
    val_idx = list(range(len(vv)))
    short_train_idx = [i for i, v in enumerate(tr_v) if v[2] <= 6]
    long_idx = [i for i in val_idx if int(pre_va["n"][i].item()) >= 8]
    # configs: independent-pair baseline; the MP edge-message-passing fix; the cell-attention backbone.
    configs = [("base_indep", dict(backbone=False, pair_head="indep")),
               ("mp", dict(backbone=False, pair_head="mp")),
               ("backbone", dict(backbone=True, pair_head="indep"))]
    out = {}
    for name, cfg in configs:
        log(f"\n  [arm1/{name}] {cfg}  mode={backbone_mode} bb_lr_mult={bb_lr_mult}")
        def mk():
            return StructureRack(D, K, backbone=cfg["backbone"], backbone_mode=backbone_mode,
                                 backbone_layers=backbone_layers, pair_head=cfg["pair_head"],
                                 mp_rounds=2, ternary=False).to(dev).float()
        mult = bb_lr_mult if cfg["backbone"] else 1.0
        rack = mk()
        train_rack(rack, pre_tr, dev, list(range(len(tv))), pin_tr, pair_tr, L, steps, lr, wpos, wnone,
                   bb_lr_mult=mult, log=log)
        pin_l, pair_l, _ = forward_all(rack, pre_va, dev, val_idx, L)
        rep = f1_report(None, None, pin_l, pair_l, pre_va["vmask"], pre_va["n"], vv, val_idx)
        lc = length_curve(pin_l, pair_l, pre_va, val_v, val_idx)
        # extrapolation: train SHORT(<=6), eval LONG(>=8)
        extra = {}
        if short_train_idx and long_idx:
            racks = mk()
            train_rack(racks, pre_tr, dev, short_train_idx, pin_tr, pair_tr, L, steps, lr, wpos, wnone,
                       bb_lr_mult=mult, log=log)
            pl, prl, _ = forward_all(racks, pre_va, dev, val_idx, L)   # full val; eval on long subset
            erep = f1_report(None, None, pl, prl, pre_va["vmask"], pre_va["n"], vv, long_idx)
            elc = length_curve(pl, prl, pre_va, val_v, long_idx)
            extra = {"long_overall": erep, "length_curve": elc}
        out[name] = {"overall": rep, "length_curve": lc, "extrapolation_short2long": extra}
        log(f"  [arm1/{name}] micro-F1 {rep['micro']['F1']:.3f} exact {rep['exact_factor_graph_match']:.3f} "
            f"| len " + " ".join(f"{n}:{lc[n]['micro_F1']:.2f}/{lc[n]['exact_match']:.2f}" for n in sorted(lc)))
    return out


def run_arm2(olmo, tok, dev, D, layers, xor_per, steps, lr, wpos, wnone, log):
    """FINDING 2: xor/affine, 2×2 ablation backbone×ternary, parity-F1 + GF2 query-solve."""
    global META_FACTS
    views, meta = build_xor_pool(xor_per, seed=0)
    log(f"[arm2] xor/affine pool {len(views)} "
        f"(xor {sum(m['kind']=='xor' for m in meta)} / affine {sum(m['kind']=='affine' for m in meta)})")
    rng = np.random.default_rng(2)
    perm = rng.permutation(len(views)); nval = len(views) // 5
    val_idx0 = [int(x) for x in perm[:nval]]; tr_idx0 = [int(x) for x in perm[nval:]]
    L = layers[0]
    pre = precompute(olmo, tok, dev, [(t, m, n) for (t, m, n, _) in views], [L])
    Nmax = pre["Nmax"]
    triples = enum_triples(Nmax, device=dev)
    pin_tgt, pair_tgt = csp_targets_safe(views, Nmax, dev)
    tern_tgt = facts_to_ternary_targets([v[3] for v in views], triples.cpu(), Nmax).to(dev)
    # tri_valid [R,T]: all three cells of the triple are mentioned in that instance
    vmask = pre["vmask"].to(dev)
    ti, tj, tk = triples[:, 0], triples[:, 1], triples[:, 2]
    tri_valid = (vmask[:, ti] * vmask[:, tj] * vmask[:, tk])               # [R,T]
    META_FACTS = {ri: list(views[ri][3]) for ri in range(len(views))}
    # meta keyed by ri for the metric fns
    meta_by_ri = {ri: meta[ri] for ri in range(len(meta))}

    out = {}
    for bb in (False, True):
        for tern in (False, True):
            name = f"bb{int(bb)}_tern{int(tern)}"
            log(f"\n  [arm2/{name}] backbone={bb} ternary={tern}")
            rack = StructureRack(D, K, backbone=bb, ternary=True).to(dev).float()
            tt = tern_tgt if tern else None
            tv_ = tri_valid if tern else None
            train_rack(rack, pre, dev, tr_idx0, pin_tgt, pair_tgt, L, steps, lr, wpos, wnone,
                       tern_tgt=tt, triples=(triples if tern else None), tri_valid=tv_, log=log)
            pin_l, pair_l, tern_l = forward_all(rack, pre, dev, val_idx0, L, triples=triples, use_tern=tern)
            pf = parity_f1(tern_l, triples, pre, meta_by_ri, val_idx0) if tern else \
                {"P": 0.0, "R": 0.0, "F1": 0.0, "tp": 0, "fp": 0, "fn": "n/a (binary vocab)"}
            gf = gf2_solve(pin_l, pair_l, tern_l, triples, pre, meta_by_ri, val_idx0, use_tern=tern)
            out[name] = {"backbone": bb, "ternary": tern, "parity_f1": pf, "gf2": gf}
            log(f"  [arm2/{name}] parity-F1 {pf['F1']:.3f}  GF2 query-solve {gf['query_solve_rate']:.3f}  "
                f"routed {gf['gf2_routed_rate']:.3f}  parity-rows {gf['parity_facts_emitted']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--layers", default="6")
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--wpos", type=float, default=2.0)
    ap.add_argument("--wnone", type=float, default=1.0)
    ap.add_argument("--per_rung", type=int, default=600, help="arm-1 circuit pool per (rung,split)")
    ap.add_argument("--xor_per", type=int, default=700, help="arm-2 per kind (xor / affine)")
    ap.add_argument("--arms", default="1,2")
    ap.add_argument("--backbone_mode", default="wide", choices=["wide", "bottleneck"])
    ap.add_argument("--backbone_layers", type=int, default=3)
    ap.add_argument("--bb_lr_mult", type=float, default=1.0, help="LR multiplier for backbone params")
    ap.add_argument("--out", default="runs/alpha_struct_rebuild.json")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[rebuild] loading {a.model} on {dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    olmo = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    for p in olmo.parameters():
        p.requires_grad_(False)
    D = olmo.config.hidden_size
    print(f"[rebuild] OLMo hidden {D}", flush=True)

    results = {"model": a.model, "layers": layers, "steps": a.steps, "lr": a.lr,
               "wpos": a.wpos, "wnone": a.wnone}
    arms = a.arms.split(",")
    if "1" in arms:
        print("\n========== ARM 1: FINDING 1 (cell-self-attention backbone) ==========", flush=True)
        results["arm1_finding1_longchain"] = run_arm1(olmo, tok, dev, D, layers, a.per_rung,
                                                      a.steps, a.lr, a.wpos, a.wnone, print,
                                                      backbone_mode=a.backbone_mode,
                                                      backbone_layers=a.backbone_layers,
                                                      bb_lr_mult=a.bb_lr_mult)
        results["arm1_config"] = {"backbone_mode": a.backbone_mode, "backbone_layers": a.backbone_layers,
                                  "bb_lr_mult": a.bb_lr_mult, "steps": a.steps}
    if "2" in arms:
        print("\n========== ARM 2: FINDING 2 (ternary parity vocab → GF2) ==========", flush=True)
        results["arm2_finding2_xor"] = run_arm2(olmo, tok, dev, D, layers, a.xor_per,
                                                a.steps, a.lr, a.wpos, a.wnone, print)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[rebuild] wrote {a.out}", flush=True)

    # ---- headline summary ----
    if "1" in arms:
        b = results["arm1_finding1_longchain"]
        def longavg(cfg, key="micro_F1"):
            lc = b[cfg]["length_curve"]; longs = [lc[n][key] for n in lc if int(n) >= 8]
            return float(np.mean(longs)) if longs else float("nan")
        print(f"\nFINDING 1  long-chain(≥8) micro-F1 / exact:", flush=True)
        for cfg in b:
            print(f"    {cfg:11s} micro {longavg(cfg):.3f}  exact {longavg(cfg,'exact_match'):.3f}  "
                  f"(overall micro {b[cfg]['overall']['micro']['F1']:.3f} "
                  f"exact {b[cfg]['overall']['exact_factor_graph_match']:.3f})", flush=True)
    if "2" in arms:
        c = results["arm2_finding2_xor"]
        print(f"FINDING 2  GF2 query-solve:  binary(bb1_tern0) {c['bb1_tern0']['gf2']['query_solve_rate']:.3f}"
              f"  →  +ternary(bb1_tern1) {c['bb1_tern1']['gf2']['query_solve_rate']:.3f}  "
              f"(parity-F1 {c['bb1_tern1']['parity_f1']['F1']:.3f})", flush=True)


if __name__ == "__main__":
    main()
