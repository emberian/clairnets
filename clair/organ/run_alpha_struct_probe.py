"""clair/organ/run_alpha_struct_probe.py — the ALPHA_STRUCT make-or-break capability probe.

Standalone (no composer / no LM / no eval rewiring). Trains the multi-faculty α rack's CSP head + the
router on FROZEN OLMo-2-1B hidden and answers the central question:

    CAN α READ PROBLEM STRUCTURE OFF THE HOST HIDDEN STATE?

1. CSP-capability  — train ONLY the CSP StructureHead to emit the typed factor graph (pin + pair
   relations {eq,neq,lt,le}) supervised against the generator's free true facts (soundness-asymmetric:
   heavier penalty for DROPPED relations). Per-relation P/R/F1 + exact-factor-graph match on held-out
   CANONICAL (the sanity floor → make-or-break) and a few NL phrasings (the generalization frontier).
   F1≈1.0 canonical ⇒ capability exists ⇒ proceed to multi-faculty; F1≪0.8 ⇒ rethink.
2. Router-capability — can α route to the right faculty {csp,ising,graph,type,reduction} from the hidden?
3. (2nd faculty) Ising couplings head — confirm multi-faculty viability beyond CSP.

Writes runs/alpha_struct_probe.json. Reports the F1 numbers directly.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .. import curriculum as CU
from ..oracle_readout import _mention_tensor
from .alpha_struct import StructureRack, PAIR_RELS, FACULTIES

# binary+pin CSP rungs (the pair vocab covers {eq,neq,lt,le}; arithmetic 'sum' is ternary — deferred)
BINARY_RUNGS = {"coloring", "equality", "ordering", "alldiff"}
K = 8                                            # D_MAX readout width (matches run_glados_staged.K)
REL_IDX = {r: i for i, r in enumerate(PAIR_RELS)}   # none=0, eq=1, neq=2, lt=3, le=4


# ============================================================ data
def _expand_alldiff(facts):
    import itertools as it
    out = []
    for f in facts:
        if f[0] == "alldiff":
            out += [("neq", a, b) for a, b in it.combinations(f[1], 2)]
        else:
            out.append(tuple(f))
    return out


def load_csp_records(path, rungs=BINARY_RUNGS, limit=None):
    """Curriculum records whose facts are all binary+pin (after alldiff→neq). Each yields canonical +
    NL views. We keep the raw fields needed for both."""
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["relation"] not in rungs:
                continue
            facts = [tuple(f) if f[0] != "alldiff" else ("alldiff", tuple(f[1])) for f in r["facts"]]
            exp = _expand_alldiff(facts)
            if any(f[0] not in ("pin", "eq", "neq", "lt", "le") for f in exp):
                continue
            recs.append(r)
            if limit and len(recs) >= limit:
                break
    return recs


def rec_view(r, view):
    """(text, mentions{int:spans}, n, expanded_true_facts) for canonical or NL view of a record."""
    n = r["n"]
    facts = [tuple(f) if f[0] != "alldiff" else ("alldiff", tuple(f[1])) for f in r["facts"]]
    exp = _expand_alldiff(facts)
    if view == "canonical":
        # rebuild a Problem-lite for canonical_render + recompute mentions on the canonical string
        p = CU.Problem(r["relation"], n, r["d"], r["kind"], facts, r["query"], r["answer"],
                       r["determined"], r["vnames"])
        text = CU.canonical_render(p)
        ment = CU.entity_mentions(text, n)
    else:
        text = r["text"]
        ment = {int(k): [tuple(s) for s in v] for k, v in r["mentions"].items()}
    return text, ment, n, exp


# ============================================================ frozen-host hidden precompute
@torch.no_grad()
def precompute(olmo, tok, dev, views, layers, bs=16):
    """views: list of (text, mentions, n). Returns per-layer dict of padded CPU tensors:
       H[L][R,Tmax,D], attn[R,Tmax], vmean[L][R,Nmax,D], vmask[R,Nmax], plus n per record.
    One OLMo forward yields all hidden layers; we keep only `layers`."""
    R = len(views)
    Nmax = max(v[2] for v in views)
    out_H = {L: [] for L in layers}
    out_vm = {L: [] for L in layers}
    out_attn, out_vmask, out_n = [], [], []
    Tmaxes = []
    for i in range(0, R, bs):
        chunk = views[i:i + bs]
        texts = [c[0] for c in chunk]
        enc = tok(texts, return_offsets_mapping=True, padding=True, return_tensors="pt")
        ids = enc["input_ids"].to(dev); attn = enc["attention_mask"].to(dev)
        T = ids.size(1); Bp = len(chunk)
        ment = _mention_tensor([c[1] for c in chunk], enc["offset_mapping"], Bp, Nmax, T, dev)
        hs = olmo(input_ids=ids, attention_mask=attn, output_hidden_states=True).hidden_states
        denom = ment.sum(-1, keepdim=True).clamp_min(1e-6)
        vmask = (ment.sum(-1) > 0.5).float()
        for L in layers:
            h = hs[L].float()                                       # [Bp,T,D]
            vmean = torch.einsum("bnt,btd->bnd", ment, h) / denom   # [Bp,Nmax,D]
            out_H[L].append(h.cpu())
            out_vm[L].append(vmean.cpu())
        out_attn.append(attn.float().cpu()); out_vmask.append(vmask.cpu())
        out_n += [c[2] for c in chunk]; Tmaxes.append(T)
    Tmax = max(Tmaxes); D = hs[0].size(-1)
    def padcat(lst, padT):
        full = torch.zeros(R, padT, *lst[0].shape[2:])
        r = 0
        for t in lst:
            b, tt = t.shape[0], t.shape[1]
            full[r:r + b, :tt] = t; r += b
        return full
    H = {L: padcat(out_H[L], Tmax) for L in layers}
    attn = padcat([a.unsqueeze(-1) for a in out_attn], Tmax).squeeze(-1)
    vmean = {L: torch.cat(out_vm[L], 0) for L in layers}
    vmask = torch.cat(out_vmask, 0)
    return {"H": H, "attn": attn, "vmean": vmean, "vmask": vmask,
            "n": torch.tensor(out_n), "R": R, "Nmax": Nmax, "D": D, "Tmax": Tmax}


# ============================================================ CSP targets
def csp_targets(views, Nmax, dev):
    """pin_tgt[R,Nmax] (0=no-pin else v+1), pair_tgt[R,Nmax,Nmax] (REL_IDX), valid masks."""
    R = len(views)
    pin = torch.zeros(R, Nmax, dtype=torch.long)
    pair = torch.zeros(R, Nmax, Nmax, dtype=torch.long)
    for ri, (_, _, n, facts) in enumerate(views):
        for f in facts:
            k = f[0]
            if k == "pin":
                _, a, v = f
                pin[ri, a] = v + 1
            elif k in ("eq", "neq"):
                a, b = f[1], f[2]
                pair[ri, a, b] = REL_IDX[k]; pair[ri, b, a] = REL_IDX[k]
            else:                                  # lt / le directional
                a, b = f[1], f[2]
                pair[ri, a, b] = REL_IDX[k]
    return pin.to(dev), pair.to(dev)


# ============================================================ decode + F1
def decode_facts(pin_logits, pair_logits, vmask, n):
    """One instance → normalized predicted fact set (pin/eq/neq/lt/le) via curriculum.norm_facts."""
    facts = []
    pin = pin_logits.argmax(-1)                                   # [N]
    for i in range(n):
        if vmask[i] > 0.5 and pin[i].item() != 0:
            facts.append(("pin", i, int(pin[i].item()) - 1))
    pr = pair_logits.argmax(-1)                                   # [N,N]
    for i in range(n):
        if vmask[i] < 0.5:
            continue
        for j in range(n):
            if i == j or vmask[j] < 0.5:
                continue
            cls = PAIR_RELS[int(pr[i, j].item())]
            if cls in ("eq", "neq", "lt", "le"):
                facts.append((cls, i, j))
    return CU.norm_facts(facts)


def f1_report(rack, feat_all, pin_l_all, pair_l_all, vmask, n_arr, views, idxs):
    """Per-relation precision/recall/F1 + exact-factor-graph match over `idxs`."""
    rels = ["pin", "eq", "neq", "lt", "le"]
    tp = {r: 0 for r in rels}; fp = {r: 0 for r in rels}; fn = {r: 0 for r in rels}
    exact = 0
    for ri in idxs:
        n = int(n_arr[ri].item())
        pred = decode_facts(pin_l_all[ri], pair_l_all[ri], vmask[ri], n)
        true = CU.norm_facts(views[ri][3])
        if pred == true:
            exact += 1
        for f in pred:
            k = f[0]
            (tp if f in true else fp)[k] += 1
        for f in true:
            if f not in pred:
                fn[f[0]] += 1
    rep = {}
    aTP = aFP = aFN = 0
    for r in rels:
        p = tp[r] / max(1, tp[r] + fp[r]); rc = tp[r] / max(1, tp[r] + fn[r])
        f1 = 2 * p * rc / max(1e-9, p + rc)
        rep[r] = {"P": round(p, 4), "R": round(rc, 4), "F1": round(f1, 4),
                  "tp": tp[r], "fp": fp[r], "fn": fn[r]}
        aTP += tp[r]; aFP += fp[r]; aFN += fn[r]
    P = aTP / max(1, aTP + aFP); Rc = aTP / max(1, aTP + aFN)
    rep["micro"] = {"P": round(P, 4), "R": round(Rc, 4),
                    "F1": round(2 * P * Rc / max(1e-9, P + Rc), 4)}
    rep["exact_factor_graph_match"] = round(exact / max(1, len(idxs)), 4)
    rep["n"] = len(idxs)
    return rep


# ============================================================ CSP training
def forward_csp(rack, pre, dev, idxs, L, bs=64):
    """Run the encoder+CSP head over records `idxs` (batched), return pin/pair logits + feat per record."""
    Nmax = pre["Nmax"]
    pin_out = torch.zeros(len(idxs), Nmax, 1 + K, device=dev)
    pair_out = torch.zeros(len(idxs), Nmax, Nmax, len(PAIR_RELS), device=dev)
    pos = {ri: k for k, ri in enumerate(idxs)}
    for i in range(0, len(idxs), bs):
        sub = idxs[i:i + bs]
        h = pre["H"][L][sub].to(dev); attn = pre["attn"][sub].to(dev)
        vmean = pre["vmean"][L][sub].to(dev)
        feat = rack.featurize(vmean, h, attn)
        pin_l, pair_l = rack.csp(feat)
        for k, ri in enumerate(sub):
            pin_out[pos[ri]] = pin_l[k]; pair_out[pos[ri]] = pair_l[k]
    return pin_out, pair_out


def train_csp(rack, pre, dev, train_idx, pin_tgt, pair_tgt, steps=400, lr=0.0, bs=48,
              wpos=4.0, wnone=0.3, L=6, log=print):
    """Train encoder + CSP head. Soundness-asymmetric weights: missing a relation (predicting none/no-pin
    on a true fact) is penalized heavier (wpos) than hallucinating (none/no-pin class weight wnone)."""
    opt = torch.optim.AdamW(list(rack.encoder.parameters()) + list(rack.heads["csp"].parameters()),
                            lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    pin_w = torch.full((1 + K,), wpos, device=dev); pin_w[0] = wnone
    pair_w = torch.full((len(PAIR_RELS),), wpos, device=dev); pair_w[0] = wnone
    rng = np.random.default_rng(0)
    rack.train(); t0 = time.time()
    train_idx = list(train_idx)
    for s in range(1, steps + 1):
        sub = [int(x) for x in rng.choice(train_idx, size=min(bs, len(train_idx)), replace=False)]
        h = pre["H"][L][sub].to(dev); attn = pre["attn"][sub].to(dev)
        vmean = pre["vmean"][L][sub].to(dev); vmask = pre["vmask"][sub].to(dev)
        feat = rack.featurize(vmean, h, attn)
        pin_l, pair_l = rack.csp(feat)
        B, Nm, _ = pin_l.shape
        pt = pin_tgt[sub]; prt = pair_tgt[sub]
        # pin loss over valid cells
        cellm = vmask.bool()
        lp = F.cross_entropy(pin_l[cellm], pt[cellm], weight=pin_w)
        # pair loss over valid ordered off-diagonal pairs
        pm = (vmask[:, :, None] * vmask[:, None, :]).bool()
        eye = torch.eye(Nm, device=dev, dtype=torch.bool)[None].expand(B, Nm, Nm)
        pm = pm & ~eye
        lpr = F.cross_entropy(pair_l[pm], prt[pm], weight=pair_w)
        loss = lp + lpr
        opt.zero_grad(); loss.backward(); opt.step()
        if s % max(1, steps // 8) == 0 or s == 1:
            log(f"    [csp L{L}] step {s:4d}  pin {lp.item():.3f}  pair {lpr.item():.3f}  "
                f"{time.time()-t0:.0f}s")
    return rack


# ============================================================ router
def gen_router_pool(rng, csp_recs, per=200):
    """(text, mentions, n, faculty_idx). CSP from curriculum; ising/graph/type/reduction templated."""
    F2I = {f: i for i, f in enumerate(FACULTIES)}
    pool = []
    for r in rng.permutation(len(csp_recs))[:per]:
        rr = csp_recs[int(r)]
        text, ment, n, _ = rec_view(rr, "nl")
        pool.append((text, ment, n, F2I["csp"]))
    E = CU.ENTITIES
    for _ in range(per):                                          # ising / energy
        n = int(rng.integers(3, 6)); ms = {}
        parts = []
        for i in range(n):
            ms[i] = []
        for a in range(n):
            for b in range(a + 1, n):
                if rng.random() < 0.6:
                    parts.append(f"{E[a]} and {E[b]} prefer to "
                                 f"{'align' if rng.random()<0.5 else 'point oppositely'}.")
        parts.append("Find the spin configuration that minimizes the total energy.")
        text = " ".join(parts)
        pool.append((text, CU.entity_mentions(text, n), n, F2I["ising"]))
    for _ in range(per):                                          # graph / path
        n = int(rng.integers(4, 7)); edges = []
        for a in range(n):
            for b in range(a + 1, n):
                if rng.random() < 0.4:
                    edges.append(f"{E[a]}-{E[b]}")
        text = (f"Consider a graph on nodes {', '.join(E[:n])}. The edges are "
                f"{', '.join(edges) if edges else 'none'}. Is there a path from {E[0]} to {E[n-1]}?")
        pool.append((text, CU.entity_mentions(text, n), n, F2I["graph"]))
    for _ in range(per):                                          # type inference
        n = int(rng.integers(2, 5))
        lines = []
        for i in range(n):
            rhs = rng.choice(["3", "true", "x + 1", "\"hi\"", "f(y)"])
            lines.append(f"let {E[i]} = {rhs};")
        text = " ".join(lines) + f" What is the type of {E[n-1]}?"
        pool.append((text, CU.entity_mentions(text, n), n, F2I["type"]))
    for _ in range(per):                                          # reduction route
        n = int(rng.integers(3, 6))
        tgt = rng.choice(["2-SAT", "maximum matching", "max-flow", "shortest path", "graph coloring"])
        text = (f"We have a constraint problem over items {', '.join(E[:n])}. "
                f"Which standard problem should we reduce it to in order to solve it? "
                f"Consider {tgt} as a candidate reduction.")
        pool.append((text, CU.entity_mentions(text, n), n, F2I["reduction"]))
    order = rng.permutation(len(pool))
    return [pool[i] for i in order]


def train_eval_router(rack, olmo, tok, dev, pool, layers, log=print):
    views = [(t, m, n) for (t, m, n, _) in pool]
    labels = torch.tensor([lab for (_, _, _, lab) in pool], device=dev)
    pre = precompute(olmo, tok, dev, views, layers)
    best = None
    for L in layers:
        ntot = len(pool); nval = ntot // 5
        idx = np.random.default_rng(1).permutation(ntot)
        val_idx = idx[:nval]; tr_idx = idx[nval:]
        opt = torch.optim.AdamW(rack.router.parameters(), lr=2e-3, weight_decay=0.0)
        Hc = pre["H"][L]; attn = pre["attn"]
        rng = np.random.default_rng(0)
        rack.train()
        for s in range(300):
            sub = [int(x) for x in rng.choice(tr_idx, size=min(64, len(tr_idx)), replace=False)]
            logit = rack.route(Hc[sub].to(dev), attn[sub].to(dev))
            loss = F.cross_entropy(logit, labels[sub])
            opt.zero_grad(); loss.backward(); opt.step()
        rack.eval()
        with torch.no_grad():
            vi = [int(x) for x in val_idx]
            logit = rack.route(Hc[vi].to(dev), attn[vi].to(dev))
            pred = logit.argmax(-1); lab = labels[vi]
            acc = float((pred == lab).float().mean())
            cm = torch.zeros(len(FACULTIES), len(FACULTIES), dtype=torch.long)
            for p, t in zip(pred.tolist(), lab.tolist()):
                cm[t, p] += 1
        log(f"    [router L{L}] val acc {acc*100:.1f}%  (n={len(vi)})")
        per_fac = {FACULTIES[i]: round(float(cm[i, i]) / max(1, int(cm[i].sum())), 4)
                   for i in range(len(FACULTIES))}
        if best is None or acc > best["acc"]:
            best = {"layer": L, "acc": round(acc, 4), "per_faculty_recall": per_fac, "n": len(vi)}
    return best


# ============================================================ Ising 2nd-faculty probe
def gen_ising_pool(rng, n_inst=600):
    """Ising instances with known couplings (sign) + fields. Returns views + targets."""
    E = CU.ENTITIES
    pool = []
    for _ in range(n_inst):
        n = int(rng.integers(3, 6))
        J = np.zeros((n, n), dtype=np.int64); h = np.zeros(n, dtype=np.int64)
        parts = []
        for a in range(n):
            for b in range(a + 1, n):
                if rng.random() < 0.55:
                    s = 1 if rng.random() < 0.5 else -1
                    J[a, b] = J[b, a] = s
                    parts.append(f"{E[a]} and {E[b]} prefer to "
                                 f"{'align' if s>0 else 'point oppositely'}.")
        for a in range(n):
            if rng.random() < 0.5:
                s = 1 if rng.random() < 0.5 else -1
                h[a] = s
                parts.append(f"{E[a]} leans {'up' if s>0 else 'down'}.")
        parts.append("Minimize the total energy.")
        text = " ".join(parts)
        pool.append((text, CU.entity_mentions(text, n), n, (J, h)))
    return pool


def train_eval_ising(rack, olmo, tok, dev, pool, L, log=print):
    views = [(t, m, n) for (t, m, n, _) in pool]
    pre = precompute(olmo, tok, dev, views, [L])
    Nmax = pre["Nmax"]; R = len(pool)
    Jt = torch.zeros(R, Nmax, Nmax); ht = torch.zeros(R, Nmax)
    Jmask = torch.zeros(R, Nmax, Nmax)
    for ri, (_, _, n, (J, h)) in enumerate(pool):
        Jt[ri, :n, :n] = torch.from_numpy(J).float()
        ht[ri, :n] = torch.from_numpy(h).float()
        Jmask[ri, :n, :n] = 1.0
    Jt, ht, Jmask = Jt.to(dev), ht.to(dev), Jmask.to(dev)
    idx = np.random.default_rng(2).permutation(R); nval = R // 5
    val_idx, tr_idx = idx[:nval], idx[nval:]
    opt = torch.optim.AdamW(list(rack.encoder.parameters()) + list(rack.heads["ising"].parameters()),
                            lr=1e-3, weight_decay=0.0)
    rng = np.random.default_rng(0)
    eye = torch.eye(Nmax, device=dev, dtype=torch.bool)
    rack.train()
    for s in range(400):
        sub = [int(x) for x in rng.choice(tr_idx, size=min(48, len(tr_idx)), replace=False)]
        h = pre["H"][L][sub].to(dev); attn = pre["attn"][sub].to(dev)
        vmean = pre["vmean"][L][sub].to(dev); vmask = pre["vmask"][sub].to(dev)
        feat = rack.featurize(vmean, h, attn)
        Jp, hp = rack.ising(feat)
        pm = (vmask[:, :, None] * vmask[:, None, :]); pm = pm * (~eye[None]).float()
        lj = ((Jp - Jt[sub]) ** 2 * pm).sum() / pm.sum().clamp_min(1)
        lh = ((hp - ht[sub]) ** 2 * vmask).sum() / vmask.sum().clamp_min(1)
        loss = lj + lh
        opt.zero_grad(); loss.backward(); opt.step()
    rack.eval()
    with torch.no_grad():
        vi = [int(x) for x in val_idx]
        h = pre["H"][L][vi].to(dev); attn = pre["attn"][vi].to(dev)
        vmean = pre["vmean"][L][vi].to(dev); vmask = pre["vmask"][vi].to(dev)
        feat = rack.featurize(vmean, h, attn)
        Jp, hp = rack.ising(feat)
        # edge detection: |J|>0.5 predicted vs nonzero true; on UPPER triangle valid pairs
        etp = efp = efn = sign_ok = sign_tot = 0
        hsign_ok = hsign_tot = 0
        for k, ri in enumerate(vi):
            n = int(pre["n"][ri].item())
            for a in range(n):
                if vmask[k, a] > 0.5 and ht[ri, a].abs() > 0:
                    hsign_tot += 1
                    hsign_ok += int(np.sign(hp[k, a].item()) == np.sign(ht[ri, a].item()))
                for b in range(a + 1, n):
                    true_e = Jt[ri, a, b].abs() > 0
                    pred_e = Jp[k, a, b].abs() > 0.5
                    if true_e and pred_e:
                        etp += 1
                        sign_tot += 1
                        sign_ok += int(np.sign(Jp[k, a, b].item()) == np.sign(Jt[ri, a, b].item()))
                    elif pred_e and not true_e:
                        efp += 1
                    elif true_e and not pred_e:
                        efn += 1
        p = etp / max(1, etp + efp); rc = etp / max(1, etp + efn)
        f1 = 2 * p * rc / max(1e-9, p + rc)
    res = {"layer": L, "coupling_edge_F1": round(f1, 4), "coupling_P": round(p, 4),
           "coupling_R": round(rc, 4), "coupling_sign_acc": round(sign_ok / max(1, sign_tot), 4),
           "field_sign_acc": round(hsign_ok / max(1, hsign_tot), 4), "n": len(vi)}
    log(f"    [ising L{L}] edge-F1 {f1:.3f}  sign-acc {res['coupling_sign_acc']:.3f}  "
        f"field-sign {res['field_sign_acc']:.3f}")
    return res


# ============================================================ NL-frontier arm (off-template generalization)
def nl_frontier_arm(olmo, tok, dev, D, layers, nl_train_path, nl_ood_path, steps, lr,
                    wpos, wnone, can_train_views=None, log=print):
    """Train the CSP StructureHead on the NL-FRONTIER train set (clair.datagen.nl_frontier:
    hard-negative pairs + widened phrasing + optional Bedrock naturalization) and evaluate on the
    NL-OOD split (held-out skins / held-out naturalizer style). This is the off-template
    generalization test the α-probe's diagnosis calls for — measured, not just trained on.

    Sweeps `layers`, reports per-layer NL-OOD P/R/F1 + exact-match, and the best layer by OOD F1.
    `can_train_views` (optional) mixes canonical-template views into the NL training pool."""
    tr = load_csp_records(nl_train_path)
    od = load_csp_records(nl_ood_path)
    log(f"   NL-frontier: {len(tr)} train, {len(od)} OOD records (binary+pin)")
    tr_v = [rec_view(r, "nl") for r in tr]
    od_v = [rec_view(r, "nl") for r in od]
    train_v = tr_v + list(can_train_views or [])
    pre_tr = precompute(olmo, tok, dev, [(t, m, n) for (t, m, n, _) in train_v], layers)
    pre_od = precompute(olmo, tok, dev, [(t, m, n) for (t, m, n, _) in od_v], layers)
    pin_t, pair_t = csp_targets(train_v, pre_tr["Nmax"], dev)
    per_layer = {}
    for L in layers:
        rack = StructureRack(D, K).to(dev)
        rack.encoder.float(); rack.heads["csp"].float()
        train_csp(rack, pre_tr, dev, list(range(len(train_v))), pin_t, pair_t,
                  steps=steps, lr=lr, L=L, wpos=wpos, wnone=wnone, log=log)
        rack.eval()
        with torch.no_grad():
            idx = list(range(len(od_v)))
            pin_l, pair_l = forward_csp(rack, pre_od, dev, idx, L)
            rep = f1_report(rack, None, pin_l, pair_l, pre_od["vmask"], pre_od["n"], od_v, idx)
        log(f"   [nl-frontier L{L}] NL-OOD micro-F1 {rep['micro']['F1']:.3f}  "
            f"exact {rep['exact_factor_graph_match']:.3f}")
        per_layer[L] = rep
    best_L = max(layers, key=lambda L: per_layer[L]["micro"]["F1"])
    return {"nl_train": nl_train_path, "nl_ood": nl_ood_path,
            "n_nl_train": len(tr_v), "n_canonical_mixed": len(can_train_views or []),
            "n_nl_ood": len(od_v), "per_layer": per_layer, "best_layer": best_L,
            "best_nl_ood": per_layer[best_L]}


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curriculum", default="data/curriculum/curriculum.jsonl")
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--layers", default="6,9,12")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--wpos", type=float, default=2.0, help="weight on relation classes (recall)")
    ap.add_argument("--wnone", type=float, default=1.0, help="weight on none/no-pin class (precision)")
    ap.add_argument("--out", default="runs/alpha_struct_probe.json")
    ap.add_argument("--max_recs", type=int, default=1400)
    ap.add_argument("--nl_train", default=None,
                    help="NL-frontier train jsonl (clair.datagen.nl_frontier out/train.jsonl)")
    ap.add_argument("--nl_ood", default=None,
                    help="NL-OOD eval jsonl (clair.datagen.nl_frontier out/ood.jsonl)")
    ap.add_argument("--nl_only", action="store_true",
                    help="run ONLY the NL-frontier arm (skip the canonical/router/ising sweep)")
    ap.add_argument("--nl_mix_canonical", action="store_true",
                    help="mix canonical-template views into the NL-frontier train pool")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {a.model} on {dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    olmo = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    for p in olmo.parameters():
        p.requires_grad_(False)
    D = olmo.config.hidden_size
    print(f"OLMo hidden {D}, layers {olmo.config.num_hidden_layers}", flush=True)

    # ---- fast standalone NL-frontier run: train on the NL set, eval on NL-OOD, nothing else ----
    if a.nl_only:
        if not (a.nl_train and a.nl_ood):
            raise SystemExit("--nl_only requires --nl_train and --nl_ood")
        can_views = None
        if a.nl_mix_canonical:
            crecs = load_csp_records(a.curriculum, limit=a.max_recs)
            can_views = [rec_view(r, "canonical") for r in crecs]
        print("\n== NL-FRONTIER arm (off-template generalization) ==", flush=True)
        res = nl_frontier_arm(olmo, tok, dev, D, layers, a.nl_train, a.nl_ood, a.steps, a.lr,
                              a.wpos, a.wnone, can_train_views=can_views)
        out = {"model": a.model, "layers": layers, "nl_frontier": res}
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=2)
        b = res["best_nl_ood"]["micro"]["F1"]
        print(f"\n==== NL-FRONTIER: best NL-OOD micro-F1 {b:.3f} (layer {res['best_layer']}) ====",
              flush=True)
        print(f"wrote {a.out}", flush=True)
        return

    recs = load_csp_records(a.curriculum, limit=a.max_recs)
    print(f"{len(recs)} binary+pin CSP records", flush=True)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(recs))
    nval = len(recs) // 5
    val_recs = [recs[i] for i in perm[:nval]]
    tr_recs = [recs[i] for i in perm[nval:]]

    can_train = [rec_view(r, "canonical") for r in tr_recs]
    can_val = [rec_view(r, "canonical") for r in val_recs]
    nl_train = [rec_view(r, "nl") for r in tr_recs]
    nl_val = [rec_view(r, "nl") for r in val_recs]

    print("precomputing frozen hidden (canonical train/val, NL train/val)", flush=True)
    pre_ct = precompute(olmo, tok, dev, [(t, m, n) for (t, m, n, _) in can_train], layers)
    pre_cv = precompute(olmo, tok, dev, [(t, m, n) for (t, m, n, _) in can_val], layers)
    pre_nt = precompute(olmo, tok, dev, [(t, m, n) for (t, m, n, _) in nl_train], layers)
    pre_nv = precompute(olmo, tok, dev, [(t, m, n) for (t, m, n, _) in nl_val], layers)
    Nmax = max(pre_ct["Nmax"], pre_cv["Nmax"], pre_nt["Nmax"], pre_nv["Nmax"])
    for pre in (pre_ct, pre_cv, pre_nt, pre_nv):    # pad Nmax consistent across splits
        pass

    results = {"model": a.model, "layers": layers, "n_train": len(tr_recs), "n_val": len(val_recs)}

    # ---- CSP capability: per-layer sweep, train on CANONICAL, eval canonical-heldout + NL zero-shot ----
    csp_runs = {}
    for L in layers:
        print(f"\n== CSP head, train CANONICAL @ layer {L} ==", flush=True)
        rack = StructureRack(D, K).to(dev)
        rack.encoder.float(); rack.heads["csp"].float()
        pin_t, pair_t = csp_targets(can_train, pre_ct["Nmax"], dev)
        train_csp(rack, pre_ct, dev, list(range(len(can_train))), pin_t, pair_t,
                  steps=a.steps, lr=a.lr, L=L, wpos=a.wpos, wnone=a.wnone)
        # eval canonical heldout
        rack.eval()
        with torch.no_grad():
            idx = list(range(len(can_val)))
            pin_l, pair_l = forward_csp(rack, pre_cv, dev, idx, L)
            can_rep = f1_report(rack, None, pin_l, pair_l, pre_cv["vmask"], pre_cv["n"], can_val, idx)
            idx = list(range(len(nl_val)))
            pin_l, pair_l = forward_csp(rack, pre_nv, dev, idx, L)
            nl_rep = f1_report(rack, None, pin_l, pair_l, pre_nv["vmask"], pre_nv["n"], nl_val, idx)
        print(f"   canonical-heldout micro-F1 {can_rep['micro']['F1']:.3f}  "
              f"exact-match {can_rep['exact_factor_graph_match']:.3f}", flush=True)
        print(f"   NL zero-shot   micro-F1 {nl_rep['micro']['F1']:.3f}  "
              f"exact-match {nl_rep['exact_factor_graph_match']:.3f}", flush=True)
        csp_runs[L] = {"canonical_heldout": can_rep, "nl_zeroshot": nl_rep}

    best_L = max(layers, key=lambda L: csp_runs[L]["canonical_heldout"]["micro"]["F1"])
    results["csp_train_canonical"] = csp_runs
    results["best_layer"] = best_L

    # ---- CSP: train CANONICAL+NL, eval NL-heldout (NL capability when trained on NL) ----
    print(f"\n== CSP head, train CANONICAL+NL @ layer {best_L} ==", flush=True)
    rack = StructureRack(D, K).to(dev)
    rack.encoder.float(); rack.heads["csp"].float()
    # concatenate canonical-train + nl-train precomputes by re-precomputing the union
    union_views = [(t, m, n) for (t, m, n, _) in can_train] + [(t, m, n) for (t, m, n, _) in nl_train]
    union_facts = can_train + nl_train
    pre_u = precompute(olmo, tok, dev, union_views, [best_L])
    pin_t, pair_t = csp_targets(union_facts, pre_u["Nmax"], dev)
    train_csp(rack, pre_u, dev, list(range(len(union_views))), pin_t, pair_t,
              steps=a.steps, lr=a.lr, L=best_L, wpos=a.wpos, wnone=a.wnone)
    rack.eval()
    with torch.no_grad():
        idx = list(range(len(nl_val)))
        pin_l, pair_l = forward_csp(rack, pre_nv, dev, idx, best_L)
        nl_trained = f1_report(rack, None, pin_l, pair_l, pre_nv["vmask"], pre_nv["n"], nl_val, idx)
    print(f"   NL-heldout (trained on NL) micro-F1 {nl_trained['micro']['F1']:.3f}  "
          f"exact-match {nl_trained['exact_factor_graph_match']:.3f}", flush=True)
    results["csp_train_canonical_plus_nl"] = {"layer": best_L, "nl_heldout": nl_trained}

    # ---- Router capability ----
    print(f"\n== ROUTER capability ==", flush=True)
    rack_r = StructureRack(D, K).to(dev)
    pool = gen_router_pool(np.random.default_rng(7), recs, per=220)
    router_res = train_eval_router(rack_r, olmo, tok, dev, pool, layers)
    results["router"] = router_res
    print(f"   router best: layer {router_res['layer']}  acc {router_res['acc']*100:.1f}%", flush=True)

    # ---- 2nd faculty: Ising couplings ----
    print(f"\n== 2nd-FACULTY: Ising couplings @ layer {best_L} ==", flush=True)
    rack_i = StructureRack(D, K).to(dev)
    rack_i.encoder.float(); rack_i.heads["ising"].float()
    ising_pool = gen_ising_pool(np.random.default_rng(11), n_inst=700)
    ising_res = train_eval_ising(rack_i, olmo, tok, dev, ising_pool, best_L)
    results["ising"] = ising_res

    # ---- NL-frontier arm: train on the off-template NL set, eval on NL-OOD (held-out skins/style) ----
    if a.nl_train and a.nl_ood:
        print(f"\n== NL-FRONTIER arm (off-template generalization) ==", flush=True)
        can_views = can_train if a.nl_mix_canonical else None
        results["nl_frontier"] = nl_frontier_arm(olmo, tok, dev, D, layers, a.nl_train, a.nl_ood,
                                                  a.steps, a.lr, a.wpos, a.wnone,
                                                  can_train_views=can_views)

    # ---- verdict ----
    bc = csp_runs[best_L]["canonical_heldout"]["micro"]["F1"]
    verdict = ("PASS (capability exists)" if bc >= 0.95 else
               "MARGINAL" if bc >= 0.8 else "RETHINK (F1 << 0.8)")
    results["verdict"] = {"best_canonical_micro_F1": bc, "decision": verdict}
    print(f"\n==== VERDICT: canonical micro-F1 {bc:.3f} → {verdict} ====", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
