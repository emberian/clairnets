"""clair/interp_probe.py — INTERPRETABILITY probe on the latent organ's flows.

The paper highlight: the structured organ exposes *legible intermediate reasoning* a black box
cannot. We decode, on held-out problems, BOTH ends of the latent organ:

  α-side ("what problem did the organ POSE?"):
      The latent-α DenseLatentProjector (clair.latent_organ) compiles OLMo's dense hidden of the
      problem TEXT into (b0 = an initial per-cell candidate lattice, context = a latent relational
      tensor) — with NO factors ever given as input. We fit a small read-out PROBE (a learned
      decoder) on a FIT split that maps α's latent output back to the implied FACTOR-GRAPH:
          - per cell: the posed PIN (initial domain), and the task DOMAIN (color/size),
          - per ordered cell-pair: the posed RELATION ∈ {none, ≠ (diff), = (same), < (lt), > (gt)}.
      On a disjoint TEST split we REVERSE-RENDER the MAP decode to a readable problem instance
      ("the organ posed: A≠B, B=red, C<A …") and MEASURE α-compile FAITHFULNESS vs the TRUE
      constraints: per-relation precision/recall/F1, edge-F1, pin accuracy, domain accuracy — and
      characterise WHERE it is lossy. Honest caveat: latent-α is a dense projection; this measures
      *decodability under a learned probe*, an upper bound on what α linearly/simply exposes — what
      the probe recovers, α discovered ON ITS OWN purely from output (dominate-dedₚ) supervision.

  γ-side ("what did it CONCLUDE?"):
      The narrowed lattice → per-cell candidate sets → the answer/abstention. Directly readable; we
      show it matches the exact transformer dedₚ (the verified deduction) and, on uniquely-solvable
      instances, the solution.

  Trajectory: for example problems, the full legible story — posed factor-graph → narrowing
      trajectory (cardinality dropping per round) → conclusion.

  The interp metric: across held-out, the α-faithfulness table + whether a FAITHFUL compile
      predicts a CORRECT answer (is α the bottleneck — codex's blind spot, now MEASURED).

  SMOKE: python -m clair.interp_probe --smoke
  FULL : python -m clair.interp_probe --steps 1400 --lora --train_n 4,5,6 --probe_n 4,5,6 --ood_n 8
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import latent_tasks as LT
from . import latent_organ as LO

NMAX, KVAL, MMAX = 12, 4, 40
REL_NAMES = ["none", "diff", "same", "lt", "gt"]   # 0..4
REL_SYM = {"diff": "≠", "same": "=", "lt": "<", "gt": ">"}


def device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# =========================================================== latent organ: build + train
def build_host(host_name, lora, lora_r, dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(host_name)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    olmo = AutoModelForCausalLM.from_pretrained(host_name, dtype=torch.bfloat16).to(dev).eval()
    for p in olmo.parameters():
        p.requires_grad_(False)
    host = olmo
    if lora:
        from peft import LoraConfig, get_peft_model
        lconf = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0, bias="none",
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                           "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
        host = get_peft_model(olmo, lconf)
    return host, tok, olmo.config.num_hidden_layers


def train_latent_organ(a, dev):
    host, tok, nL = build_host(a.host, a.lora, a.lora_r, dev)
    mid = min(a.mid_layer, nL)
    organ = LO.LatentOrgan(host, NMAX, KVAL, mid, train_host=a.lora, dctx=a.dctx, d=a.d,
                           n_layers=a.n_layers, T=a.T, ds=max(1, a.T // 2), mixer="ffn",
                           monotone=bool(a.monotone)).to(dev)
    organ.alpha.float(); organ.narrower.float()
    n_organ = LO.n_params(organ.alpha) + LO.n_params(organ.narrower)
    n_lora = sum(p.numel() for p in organ.lora_parameters())
    print(f"LATENT ORGAN: α+narrower={n_organ:,} params | LoRA {n_lora:,} | OLMo {nL}L mid={mid} "
          f"T={a.T} d={a.d} dctx={a.dctx}", flush=True)

    rng = np.random.default_rng(a.seed)
    kinds = tuple(a.tasks.split(","))
    train_n = [int(x) for x in a.train_n.split(",")]
    gkw = dict(edge_p=a.edge_p, pin_frac=a.pin_frac, m_max=MMAX)
    train_pool = LT.make_pool(kinds, train_n, KVAL, a.pool_per, rng, **gkw)
    print(f"train pool = {len(train_pool)}", flush=True)

    groups = [{"params": organ.organ_parameters(), "lr": a.lr}]
    if a.lora:
        groups.append({"params": organ.lora_parameters(), "lr": a.lora_lr})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    organ.train()
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
        if s % max(1, a.steps // 12) == 0 or s == 1:
            rn, rd, fn, fd = LT.narrowing_stats(b, ba["tgt"], ba["vmask"], a.theta)
            print(f"  [train] step {s:5d}  loss {float(loss.detach()):.3f}  recall {rn/max(1,rd):.3f}  "
                  f"false_elim {fn/max(1,fd):.4f}  {time.time()-t0:.0f}s", flush=True)
    organ.eval()
    return organ, tok


# =========================================================== α feature extraction
@torch.no_grad()
def alpha_outputs(organ, ba):
    """Replicate LatentOrgan.forward up to α: return (b0 [B,N,K], context [B,N,dctx])."""
    h = organ.host_encode(ba)
    hd = h.float()
    m = ba["mention"]
    denom = m.sum(-1, keepdim=True).clamp_min(1e-6)
    v_mean = torch.einsum("bnt,btd->bnd", m, hd) / denom
    b0, context = organ.alpha(v_mean, hd, ba["attn"])
    return b0, context


def true_factor_graph(prob):
    """Ground-truth pins[cell]->val (or -1), rel[(i,j)]->class id, over the n valid cells."""
    pins = {}
    rels = {}
    for c in prob.cons:
        if c[0] == "pin":
            pins[c[1]] = c[2]
        elif c[0] == "diff":
            i, j = c[1], c[2]; rels[(i, j)] = 1; rels[(j, i)] = 1
        elif c[0] == "same":
            i, j = c[1], c[2]; rels[(i, j)] = 2; rels[(j, i)] = 2
        elif c[0] == "lt":
            i, j = c[1], c[2]; rels[(i, j)] = 3; rels[(j, i)] = 4   # i<j -> lt(i,j), gt(j,i)
    return pins, rels


@torch.no_grad()
def gather_alpha_dataset(organ, tok, pool, dev, pset="train", bs=32):
    """Run α over a pool; return flat per-cell and per-pair feature/label arrays + per-instance index."""
    cell_feat, cell_pin, cell_dom, cell_owner = [], [], [], []
    pair_feat, pair_lab, pair_owner = [], [], []
    inst = []   # per-instance record for γ + correlation
    oid = 0
    for s in range(0, len(pool), bs):
        items = pool[s:s + bs]
        ba = LT.build_batch(items, tok, NMAX, KVAL, dev, pset)
        b0, context = alpha_outputs(organ, ba)
        b0c = b0.float().cpu().numpy(); ctxc = context.float().cpu().numpy()
        for bi, (prob, csp, ded) in enumerate(items):
            n = prob.n
            pins, rels = true_factor_graph(prob)
            dom = 0 if prob.domain == "color" else 1
            for i in range(n):
                cell_feat.append(np.concatenate([b0c[bi, i], ctxc[bi, i]]))
                cell_pin.append(pins.get(i, KVAL))     # KVAL == "no pin" class
                cell_dom.append(dom)
                cell_owner.append(oid)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    f = np.concatenate([ctxc[bi, i], ctxc[bi, j], b0c[bi, i], b0c[bi, j],
                                        ctxc[bi, i] * ctxc[bi, j]])
                    pair_feat.append(f)
                    pair_lab.append(rels.get((i, j), 0))
                    pair_owner.append(oid)
            inst.append({"oid": oid, "prob": prob, "csp": csp, "ded": ded, "pins": pins,
                         "rels": rels, "dom": dom, "n": n})
            oid += 1
    return {
        "cell_feat": np.asarray(cell_feat, np.float32), "cell_pin": np.asarray(cell_pin, np.int64),
        "cell_dom": np.asarray(cell_dom, np.int64), "cell_owner": np.asarray(cell_owner, np.int64),
        "pair_feat": np.asarray(pair_feat, np.float32), "pair_lab": np.asarray(pair_lab, np.int64),
        "pair_owner": np.asarray(pair_owner, np.int64), "inst": inst,
    }


# =========================================================== probes (learned decoders)
class MLP(nn.Module):
    def __init__(self, din, dout, hid=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(),
                                 nn.Linear(hid, dout))

    def forward(self, x):
        return self.net(x)


def train_probe(X, y, dout, dev, steps=1500, bs=2048, lr=2e-3, wclass=None, tag=""):
    Xt = torch.as_tensor(X, device=dev); yt = torch.as_tensor(y, device=dev)
    mu = Xt.mean(0, keepdim=True); sd = Xt.std(0, keepdim=True).clamp_min(1e-5)
    Xn = (Xt - mu) / sd
    net = MLP(X.shape[1], dout).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    w = None if wclass is None else torch.as_tensor(wclass, device=dev, dtype=torch.float32)
    N = Xn.shape[0]
    net.train()
    for s in range(steps):
        idx = torch.randint(0, N, (min(bs, N),), device=dev)
        logit = net(Xn[idx])
        loss = F.cross_entropy(logit, yt[idx], weight=w)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net, (mu, sd)


@torch.no_grad()
def probe_predict(net, norm, X, dev, bs=8192):
    mu, sd = norm
    out = []
    Xt = torch.as_tensor(X, device=dev)
    for s in range(0, Xt.shape[0], bs):
        out.append(net((Xt[s:s + bs] - mu) / sd).argmax(-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.int64)


# =========================================================== metrics
def prf(pred, true, cls):
    tp = int(((pred == cls) & (true == cls)).sum())
    fp = int(((pred == cls) & (true != cls)).sum())
    fn = int(((pred != cls) & (true == cls)).sum())
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    f = 2 * p * r / max(1e-9, p + r)
    return {"p": p, "r": r, "f1": f, "support": int((true == cls).sum())}


def edge_f1(pred, true):
    pe = pred != 0; te = true != 0
    tp = int((pe & te).sum()); fp = int((pe & ~te).sum()); fn = int((~pe & te).sum())
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return {"p": p, "r": r, "f1": 2 * p * r / max(1e-9, p + r),
            "edges_true": int(te.sum()), "edges_pred": int(pe.sum())}


# =========================================================== reverse render
def reverse_render(prob, pin_pred_by_cell, rel_pred):
    """pin_pred_by_cell: {cell: val or KVAL}; rel_pred: {(i,j): class}. Render a readable instance."""
    vw = LT.COLORS if prob.domain == "color" else LT.SIZES
    nd = LT.NODES
    parts = []
    for i in range(prob.n):
        v = pin_pred_by_cell.get(i, KVAL)
        if v != KVAL:
            parts.append(f"{nd[i]}={vw[v]}")
    seen = set()
    for i in range(prob.n):
        for j in range(prob.n):
            if i == j or (i, j) in seen:
                continue
            c = rel_pred.get((i, j), 0)
            if c == 0:
                continue
            seen.add((i, j))
            if c in (1, 2):     # symmetric
                seen.add((j, i))
                parts.append(f"{nd[i]}{REL_SYM[REL_NAMES[c]]}{nd[j]}")
            elif c == 3:        # lt(i,j): i<j
                parts.append(f"{nd[i]}<{nd[j]}")
            elif c == 4:        # gt(i,j): i>j
                parts.append(f"{nd[i]}>{nd[j]}")
    return ", ".join(parts) if parts else "(no constraints decoded)"


def true_render(prob):
    pins, rels = true_factor_graph(prob)
    pin_by = {c: v for c, v in pins.items()}
    return reverse_render(prob, pin_by, rels)


# =========================================================== γ decode + trajectory
@torch.no_grad()
def gamma_decode(organ, tok, items, dev, theta=0.5):
    """Run the full organ; return decoded survivor sets per instance + dedₚ comparison."""
    ba = LT.build_batch(items, tok, NMAX, KVAL, dev, "train")
    b, _, b0 = organ(ba)
    keep = (torch.sigmoid(b) >= theta).cpu().numpy()
    out = []
    for bi, (prob, csp, ded) in enumerate(items):
        dec = [set(int(v) for v in range(KVAL) if keep[bi, i, v]) for i in range(prob.n)]
        ddp = [set(int(v) for v in ded[i]) for i in range(prob.n)]
        allmatch = all(dec[i] == ddp[i] for i in range(prob.n))
        q = prob.query
        qmatch = dec[q] == ddp[q]
        uniq = all(len(d) == 1 for d in ddp)        # dedₚ uniquely solves
        sol_ok = uniq and len(dec[q]) == 1 and dec[q] == ddp[q]
        out.append({"dec": dec, "ddp": ddp, "allmatch": allmatch, "qmatch": qmatch,
                    "uniq": uniq, "sol_ok": sol_ok, "query": q, "prob": prob})
    return out


@torch.no_grad()
def narrow_trajectory(organ, b0, context, vmask, theta=0.5):
    """Replay the narrower recording per-round alive-cardinality. Returns (final b, cards list)."""
    nar = organ.narrower
    ctx = nar.context_proj(context)
    b = b0
    h = nar.lattice_proj(b) + ctx

    def card(bb):
        keep = (torch.sigmoid(bb) >= theta).float() * vmask.unsqueeze(-1)
        return keep.sum().item()

    cards = [card(b0)]
    for t in range(nar.T):
        h = nar.core(h, vmask) + nar.reinject * (nar.lattice_proj(b) + ctx)
        f = nar.ln_f(h)
        b = b - F.softplus(nar.elim_head(f)) if nar.monotone else nar.elim_head(f)
        cards.append(card(b))
    return b, cards


# =========================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1400)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_lr", type=float, default=2e-4)
    ap.add_argument("--mid_layer", type=int, default=8)
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--dctx", type=int, default=256)
    ap.add_argument("--monotone", type=int, default=1)
    ap.add_argument("--tasks", default="coloring,ordering")
    ap.add_argument("--train_n", default="4,5,6")
    ap.add_argument("--probe_n", default="4,5,6")
    ap.add_argument("--ood_n", default="8")
    ap.add_argument("--pool_per", type=int, default=400)
    ap.add_argument("--fit_per", type=int, default=260)
    ap.add_argument("--test_per", type=int, default=160)
    ap.add_argument("--edge_p", type=float, default=0.3)
    ap.add_argument("--pin_frac", type=float, default=0.3)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--probe_steps", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.steps = min(a.steps, 150); a.pool_per = 90; a.fit_per = 80; a.test_per = 60
        a.train_n = "4,5"; a.probe_n = "4,5"; a.ood_n = "7"; a.probe_steps = 800

    dev = device()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(a.seed)
    kinds = tuple(a.tasks.split(","))
    gkw = dict(edge_p=a.edge_p, pin_frac=a.pin_frac, m_max=MMAX)

    # ---- 1. latent organ (train) ----
    print("== TRAIN latent organ (dominate-dedₚ; α compiles from TEXT, no factors) ==", flush=True)
    organ, tok = train_latent_organ(a, dev)
    if a.ckpt:
        os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
        torch.save({"alpha": organ.alpha.state_dict(), "narrower": organ.narrower.state_dict(),
                    "cfg": {"T": a.T, "d": a.d, "dctx": a.dctx, "n_layers": a.n_layers,
                            "mid_layer": a.mid_layer, "lora": a.lora, "host": a.host}}, a.ckpt)
        print("saved organ ->", a.ckpt, flush=True)

    # ---- 2. probe datasets (held-out, disjoint FIT / TEST) ----
    probe_n = [int(x) for x in a.probe_n.split(",")]
    ood_n = [int(x) for x in a.ood_n.split(",")]
    fit_pool = LT.make_pool(kinds, probe_n, KVAL, a.fit_per, np.random.default_rng(a.seed + 11), **gkw)
    test_pool = LT.make_pool(kinds, probe_n, KVAL, a.test_per, np.random.default_rng(a.seed + 23), **gkw)
    ood_pool = LT.make_pool(kinds, ood_n, KVAL, a.test_per, np.random.default_rng(a.seed + 37), **gkw)
    print(f"\nprobe pools: fit={len(fit_pool)} test={len(test_pool)} ood={len(ood_pool)}", flush=True)

    t0 = time.time()
    FIT = gather_alpha_dataset(organ, tok, fit_pool, dev)
    TEST = gather_alpha_dataset(organ, tok, test_pool, dev)
    OOD = gather_alpha_dataset(organ, tok, ood_pool, dev)
    print(f"α features extracted ({time.time()-t0:.0f}s): "
          f"fit cells={len(FIT['cell_pin'])} pairs={len(FIT['pair_lab'])}", flush=True)

    # ---- 3. fit decoders on FIT ----
    # relation probe: class-balance weight (none dominates)
    lab = FIT["pair_lab"]; counts = np.bincount(lab, minlength=5).astype(np.float64)
    wcls = (counts.sum() / np.maximum(counts, 1.0)); wcls = wcls / wcls.mean()
    rel_net, rel_norm = train_probe(FIT["pair_feat"], lab, 5, dev, steps=a.probe_steps,
                                    wclass=wcls, tag="rel")
    pin_net, pin_norm = train_probe(FIT["cell_feat"], FIT["cell_pin"], KVAL + 1, dev,
                                    steps=a.probe_steps, tag="pin")
    dom_net, dom_norm = train_probe(FIT["cell_feat"], FIT["cell_dom"], 2, dev,
                                    steps=max(400, a.probe_steps // 3), tag="dom")

    def evaluate(SET):
        rp = probe_predict(rel_net, rel_norm, SET["pair_feat"], dev)
        pp = probe_predict(pin_net, pin_norm, SET["cell_feat"], dev)
        dp = probe_predict(dom_net, dom_norm, SET["cell_feat"], dev)
        rt = SET["pair_lab"]; pt = SET["cell_pin"]; dt = SET["cell_dom"]
        rel_tbl = {REL_NAMES[c]: prf(rp, rt, c) for c in range(5)}
        # pin: accuracy over cells, and over PINNED cells only (recovering the real pin)
        pinned = pt != KVAL
        res = {
            "relation": rel_tbl,
            "edge_f1": edge_f1(rp, rt),
            "pin_acc_all": float((pp == pt).mean()),
            "pin_acc_pinned": float((pp[pinned] == pt[pinned]).mean()) if pinned.any() else None,
            "pin_detect_f1": prf((pp != KVAL).astype(np.int64), pinned.astype(np.int64), 1),
            "domain_acc": float((dp == dt).mean()),
        }
        return res, rp, pp, dp

    res_test, rp_t, pp_t, dp_t = evaluate(TEST)
    res_ood, rp_o, pp_o, dp_o = evaluate(OOD)

    # ---- 4. per-instance faithfulness + γ correctness + correlation (TEST) ----
    # map flat pair preds back to instances
    def per_instance(SET, rp, pp):
        po = SET["pair_owner"]; co = SET["cell_owner"]
        inst = SET["inst"]
        # build decoded graphs per instance
        rec = {it["oid"]: {"pins": {}, "rels": {}, "prob": it["prob"], "true_pins": it["pins"],
                           "true_rels": it["rels"]} for it in inst}
        # pairs
        # need (i,j) per flat pair: regenerate order identical to gather
        ptr = 0
        for it in inst:
            n = it["n"]; oid = it["oid"]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    c = int(rp[ptr]); ptr += 1
                    if c != 0:
                        rec[oid]["rels"][(i, j)] = c
        ctr = 0
        for it in inst:
            n = it["n"]; oid = it["oid"]
            for i in range(n):
                v = int(pp[ctr]); ctr += 1
                if v != KVAL:
                    rec[oid]["pins"][i] = v
        return rec

    rec_test = per_instance(TEST, rp_t, pp_t)

    # γ decode over TEST instances (batched through full organ)
    test_items = test_pool
    gam = []
    for s in range(0, len(test_items), 64):
        gam.extend(gamma_decode(organ, tok, test_items[s:s + 64], dev, a.theta))

    # per-instance faithfulness F1 (typed edges: pins as ('pin',i,v); rels undirected typed)
    def inst_truth_set(prob):
        pins, rels = true_factor_graph(prob)
        s = {("pin", i, v) for i, v in pins.items()}
        for (i, j), c in rels.items():
            if c in (1, 2):
                s.add(("rel", min(i, j), max(i, j), c))
            elif c == 3:
                s.add(("rel", i, j, 3))
            elif c == 4:
                s.add(("rel", j, i, 3))   # canonicalize gt(i,j) as lt(j,i)
        return s

    def inst_pred_set(rec_i):
        s = {("pin", i, v) for i, v in rec_i["pins"].items()}
        seen = set()
        for (i, j), c in rec_i["rels"].items():
            if (i, j) in seen:
                continue
            if c in (1, 2):
                seen.add((j, i)); s.add(("rel", min(i, j), max(i, j), c))
            elif c == 3:
                s.add(("rel", i, j, 3))
            elif c == 4:
                s.add(("rel", j, i, 3))
        return s

    rows = []
    for it_oid in range(len(test_items)):
        prob = test_items[it_oid][0]
        T = inst_truth_set(prob); P = inst_pred_set(rec_test[it_oid])
        tp = len(T & P); fp = len(P - T); fn = len(T - P)
        prec = tp / max(1, tp + fp); rec_ = tp / max(1, tp + fn)
        f1 = 2 * prec * rec_ / max(1e-9, prec + rec_)
        g = gam[it_oid]
        rows.append({"f1": f1, "prec": prec, "rec": rec_, "qmatch": g["qmatch"],
                     "allmatch": g["allmatch"], "uniq": g["uniq"], "sol_ok": g["sol_ok"]})

    f1s = np.array([r["f1"] for r in rows])
    qm = np.array([1.0 if r["qmatch"] else 0.0 for r in rows])
    am = np.array([1.0 if r["allmatch"] else 0.0 for r in rows])
    # correlation: faithfulness vs answer-correctness
    def corr(x, y):
        if x.std() < 1e-9 or y.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])
    # split by faithful (perfect F1) vs not
    perfect = f1s >= 0.999
    corr_stats = {
        "n": len(rows),
        "mean_f1": float(f1s.mean()),
        "answer_acc_query": float(qm.mean()),
        "answer_acc_all": float(am.mean()),
        "corr_f1_query": corr(f1s, qm),
        "corr_f1_all": corr(f1s, am),
        "query_acc_when_faithful": float(qm[perfect].mean()) if perfect.any() else None,
        "query_acc_when_unfaithful": float(qm[~perfect].mean()) if (~perfect).any() else None,
        "all_acc_when_faithful": float(am[perfect].mean()) if perfect.any() else None,
        "all_acc_when_unfaithful": float(am[~perfect].mean()) if (~perfect).any() else None,
        "frac_faithful": float(perfect.mean()),
        "mean_f1_when_correct": float(f1s[qm > 0.5].mean()) if (qm > 0.5).any() else None,
        "mean_f1_when_wrong": float(f1s[qm < 0.5].mean()) if (qm < 0.5).any() else None,
    }

    # overall γ narrowing quality
    gq = {"query_acc": float(qm.mean()), "allcell_exact": float(am.mean()),
          "n_uniq": int(sum(1 for r in rows if r["uniq"])),
          "sol_acc_on_uniq": (float(np.mean([1.0 if r["sol_ok"] else 0.0 for r in rows if r["uniq"]]))
                              if any(r["uniq"] for r in rows) else None)}

    # ---- 5. example legible trajectories ----
    examples = []
    ex_pool = test_items[:60]
    ba = LT.build_batch(ex_pool, tok, NMAX, KVAL, dev, "train")
    b0e, ctxe = alpha_outputs(organ, ba)
    _, cards_all = narrow_trajectory(organ, b0e, ctxe, ba["vmask"], a.theta)  # cards_all = total alive (whole batch)
    # per-instance trajectory: redo narrow with single-instance batches for the chosen examples
    chosen = []
    # pick: a coloring + an ordering + one with many constraints
    by_kind = {"coloring": None, "ordering": None}
    for oid in range(len(ex_pool)):
        k = "coloring" if ex_pool[oid][0].domain == "color" else "ordering"
        if by_kind[k] is None and len(ex_pool[oid][0].cons) >= 3:
            by_kind[k] = oid
    chosen = [o for o in by_kind.values() if o is not None][:2]
    if not chosen:
        chosen = [0]
    for oid in chosen:
        item = ex_pool[oid]
        prob = item[0]
        bai = LT.build_batch([item], tok, NMAX, KVAL, dev, "train")
        b0i, ctxi = alpha_outputs(organ, bai)
        bfin, cards = narrow_trajectory(organ, b0i, ctxi, bai["vmask"], a.theta)
        # decode posed graph for THIS instance via probes
        ds = gather_alpha_dataset(organ, tok, [item], dev)
        rpi = probe_predict(rel_net, rel_norm, ds["pair_feat"], dev)
        ppi = probe_predict(pin_net, pin_norm, ds["cell_feat"], dev)
        reci = per_instance(ds, rpi, ppi)[0]
        posed = reverse_render(prob, reci["pins"], reci["rels"])
        g = gamma_decode(organ, tok, [item], dev, a.theta)[0]
        vw = LT.COLORS if prob.domain == "color" else LT.SIZES
        concl = []
        for i in range(prob.n):
            d = sorted(g["dec"][i])
            concl.append(f"{LT.NODES[i]}∈{{{','.join(vw[v] for v in d) if d else '⊥'}}}")
        examples.append({
            "domain": prob.domain, "n": prob.n, "query": LT.NODES[prob.query],
            "true": true_render(prob), "posed": posed,
            "trajectory_cardinality": [round(c, 1) for c in cards],
            "conclusion": concl,
            "answer": (vw[sorted(g["dec"][prob.query])[0]] if len(g["dec"][prob.query]) == 1
                       else "ABSTAIN(" + str(len(g["dec"][prob.query])) + " left)"),
            "dedp_answer": (vw[sorted(g["ddp"][prob.query])[0]] if len(g["ddp"][prob.query]) == 1
                            else "open(" + str(len(g["ddp"][prob.query])) + ")"),
            "qmatch": g["qmatch"],
        })

    # ---- report ----
    def print_faith(tag, res):
        print(f"\n===== α-COMPILE FAITHFULNESS ({tag}) =====", flush=True)
        print(f"{'relation':<8}{'precision':>11}{'recall':>10}{'f1':>9}{'support':>10}")
        for nm in REL_NAMES:
            r = res["relation"][nm]
            print(f"{nm:<8}{r['p']*100:>10.1f}%{r['r']*100:>9.1f}%{r['f1']*100:>8.1f}%{r['support']:>10}")
        e = res["edge_f1"]
        print(f"  edge-F1 (any constraint): P {e['p']*100:.1f}%  R {e['r']*100:.1f}%  "
              f"F1 {e['f1']*100:.1f}%  (true edges {e['edges_true']}, pred {e['edges_pred']})")
        print(f"  pin: acc(all cells) {res['pin_acc_all']*100:.1f}%  acc(pinned) "
              f"{(res['pin_acc_pinned'] or 0)*100:.1f}%  detect-F1 {res['pin_detect_f1']['f1']*100:.1f}%")
        print(f"  domain (color/size) acc: {res['domain_acc']*100:.1f}%")

    print_faith("in-dist held-out N=%s" % a.probe_n, res_test)
    print_faith("OOD N=%s" % a.ood_n, res_ood)

    print("\n===== γ-SIDE (conclusion) — decoded narrowed lattice vs exact dedₚ =====", flush=True)
    print(f"  query-cell matches dedₚ:   {gq['query_acc']*100:.1f}%")
    print(f"  ALL cells exact-match dedₚ: {gq['allcell_exact']*100:.1f}%")
    print(f"  uniquely-solvable instances: {gq['n_uniq']}  | answer-on-unique acc: "
          f"{(gq['sol_acc_on_uniq'] or 0)*100:.1f}%")

    print("\n===== IS α THE BOTTLENECK? faithfulness ↔ answer-correctness (TEST) =====", flush=True)
    cs = corr_stats
    print(f"  mean per-instance compile-F1: {cs['mean_f1']*100:.1f}%  ({cs['frac_faithful']*100:.0f}% perfectly faithful)")
    print(f"  corr(compile-F1, query-correct) = {cs['corr_f1_query']:.3f}   "
          f"corr(compile-F1, all-correct) = {cs['corr_f1_all']:.3f}")
    print(f"  query-acc | FAITHFUL compile   = {(cs['query_acc_when_faithful'] or 0)*100:.1f}%")
    print(f"  query-acc | UNFAITHFUL compile = {(cs['query_acc_when_unfaithful'] or 0)*100:.1f}%")
    print(f"  mean compile-F1 | correct ans  = {(cs['mean_f1_when_correct'] or 0)*100:.1f}%")
    print(f"  mean compile-F1 | wrong  ans   = {(cs['mean_f1_when_wrong'] or 0)*100:.1f}%")

    print("\n===== EXAMPLE LEGIBLE TRAJECTORIES (posed → narrowed → concluded) =====", flush=True)
    for ex in examples:
        print(f"\n[{ex['domain']} n={ex['n']} query={ex['query']}]")
        print(f"  TRUE problem:   {ex['true']}")
        print(f"  α POSED (decoded from latent compile): {ex['posed']}")
        print(f"  narrowing cardinality / round: {ex['trajectory_cardinality']}")
        print(f"  γ CONCLUDED:    {' | '.join(ex['conclusion'])}")
        print(f"  answer @{ex['query']}: {ex['answer']}   (dedₚ: {ex['dedp_answer']}; "
              f"{'MATCH' if ex['qmatch'] else 'MISS'})")

    results = {"args": vars(a), "faithfulness_indist": res_test, "faithfulness_ood": res_ood,
               "gamma": gq, "bottleneck": corr_stats, "examples": examples}
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "runs", "interp_probe.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(results, open(out, "w"), indent=1, default=str)
    print("\nwrote", out, flush=True)


if __name__ == "__main__":
    main()
