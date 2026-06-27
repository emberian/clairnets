"""clair/organ/shuffled_diag.py — codex's SHUFFLED diagnostic (the GLaDOS crux test).

THE QUESTION (notes/codex_organ_review.md, answer #1): does α (compiled from the host hidden) actually
DRIVE correctness, or does the handed-in metadata structure (rec["csp"], threaded into the composer by
bank_woven.csp_spec_for_record) drive it while α is a verifier-gated passenger?

THE TEST — three arms, SAME held-out determined-query instances, SAME pairing:
  * TRUE_STRUCT  : rec["csp"] is instance i's own correct structure (current path).
  * SHUFFLED_STRUCT: α-prompt / host-hidden for instance i stays CORRECT, but the composer is fed
                     rec["csp"] from a DIFFERENT instance j (same relation/n/d/query so α still applies
                     and j's certified floor determines the SAME query cell). The certified floor now
                     solves the WRONG problem → emits j's answer at the query cell.
  * NO_STRUCT    : control / gate-zero (no injection) — the text-only floor anchor.

THE MEASUREMENT: in SHUFFLED the floor outputs j's answer for the queried cell. Score the model's
generation TWO ways — accuracy-vs-TRUE (i's real answer) AND accuracy-vs-SHUFFLED (j's answer the wrong
floor computed):
  * SHUFFLED output FOLLOWS the metadata (acc-vs-shuffled high, acc-vs-true low) ⇒ α is NOT driving,
    the metadata is → the crux bites, pretrain needs ALPHA_STRUCT.
  * SHUFFLED output FOLLOWS the truth / collapses (acc-vs-true stays up, or it abstains/degrades rather
    than parroting j) ⇒ α / the LM contributes causally beyond the metadata.

This module ONLY reuses the validated bank-woven path (clair.organ.bank_woven): the host hidden for α
is genuinely instance i's; we intervene solely on the (csp, system, tags) list handed to the composer.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch

from .bank_woven import csp_spec_for_record
from ..oracle_readout import _mention_tensor, ABSTAIN_STR


# ---------------------------------------------------------------- structure pairing (the derangement)
def struct_perm(recs, rng, key=("relation", "n", "d", "query")):
    """A within-group derangement: pair each record i with a DIFFERENT record j that shares the grouping
    key, so the swapped structure (a) has the same n (α's per-cell compile still applies) and (b) is the
    SAME relation/query (j's certified floor determines the SAME query cell → a well-defined shuffled
    answer). Returns (perm, shuffleable): perm[i]=j (j=i for singleton groups), shuffleable[i]=bool."""
    groups: dict = {}
    for i, r in enumerate(recs):
        k = tuple(r["csp"].d if f == "d" else r[f] for f in key)
        groups.setdefault(k, []).append(i)
    perm = list(range(len(recs)))
    shuffleable = [False] * len(recs)
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        order = list(idxs)
        rng.shuffle(order)
        roll = order[1:] + order[:1]                       # cyclic shift ⇒ derangement (j != i)
        for i, j in zip(order, roll):
            perm[i] = j
            shuffleable[i] = True
    return perm, shuffleable


# ---------------------------------------------------------------- the composer capture with a struct swap
@torch.no_grad()
def capture_composed(model, host_recs, struct_recs, tok, dev, two_stream=True):
    """Run the capture forward exactly like bank_woven.score_bank_woven, but feed the COMPOSER the
    structure (csp/system/tags + domain size) of `struct_recs` while α reads `host_recs`' host hidden.
    Returns surv [B, Nmax, K] — the composed candidate lattice (DETACHED, uncontrolled)."""
    from .. import run_glados_staged as G
    K = G.K
    B = len(host_recs)
    Nmax = max(r["n"] for r in host_recs)
    cap_p = [r.get("alpha_prompt", r["prompt"]) for r in host_recs] if two_stream \
        else [r["prompt"] for r in host_recs]
    cap_m = [r.get("alpha_mentions", r["mentions"]) for r in host_recs] if two_stream \
        else [r["mentions"] for r in host_recs]
    penc = tok(cap_p, return_offsets_mapping=True, padding=True, return_tensors="pt")
    pids = penc["input_ids"].to(dev); pattn = penc["attention_mask"].to(dev)
    pment = _mention_tensor(cap_m, penc["offset_mapping"], B, Nmax, pids.size(1), dev)
    csps = [csp_spec_for_record(s) for s in struct_recs]               # <-- the intervention
    dvec = torch.tensor([len(s["vnames"]) for s in struct_recs], device=dev)
    with model.live(pment, pattn, inject=False, capture=True, csps=csps, dvec=dvec):
        _ = model.logits(pids, pattn)
    return model._captured_surv[:, :Nmax, :].clone()


def singleton_at_query(surv_row, query):
    """The determined value at the query cell of a composed lattice row [N,K] (or None if not a
    singleton there). surv_row is a numpy/torch [N,K] survival matrix."""
    row = surv_row[query]
    alive = [v for v in range(row.shape[-1]) if float(row[v]) > 0.5]
    return alive[0] if len(alive) == 1 else None


# ---------------------------------------------------------------- generative scoring with a given lattice
@torch.no_grad()
def score_preds(model, host_recs, surv, tok, dev, inject=True):
    """Generative argmax over the legal answers for each host record, with `surv` [B,Nmax,K] injected by
    γ (mirrors bank_woven.score_bank_woven's scoring half). surv is the ALREADY-COMPOSED (and possibly
    structure-swapped) lattice. Returns a list of predicted candidate indices (value index, or d=abstain).
    inject=False ⇒ no lattice (the NO_STRUCT / gate-zero floor)."""
    from .. import run_glados_staged as G
    K = G.K
    Nmax = max(r["n"] for r in host_recs)
    fulls, plens, spans, prob_of = [], [], [], []
    for pi, r in enumerate(host_recs):
        cands = [" " + v for v in r["vnames"]] + [" " + ABSTAIN_STR]
        for cand in cands:
            fulls.append(r["prompt"] + cand); plens.append(len(r["prompt"]))
            spans.append({k: [tuple(s) for s in v] for k, v in r["mentions"].items()})
            prob_of.append(pi)
    enc = tok(fulls, return_offsets_mapping=True, padding=True, return_tensors="pt")
    ids = enc["input_ids"].to(dev); attn = enc["attention_mask"].to(dev)
    offsets = enc["offset_mapping"]; T = ids.size(1)
    if inject:
        mention = _mention_tensor(spans, offsets, len(fulls), Nmax, T, dev)
        prob_of_idx = torch.tensor(prob_of, device=dev)
        surv_exp = surv[prob_of_idx]
        dvec = torch.tensor([len(r["vnames"]) for r in host_recs], device=dev)
        ctx = model.live(mention, attn, inject=True, capture=False, override=surv_exp,
                         dvec=dvec[prob_of_idx])
    else:
        ctx = model.live(None, None, inject=False)
    with ctx:
        logits = model.logits(ids, attn).float()
    lp = torch.log_softmax(logits[:, :-1], -1)
    tok_lp = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    scores = torch.full((len(fulls),), -1e9, device=dev)
    for row in range(len(fulls)):
        offs = offsets[row].tolist(); mask = torch.zeros(T - 1, device=dev)
        for ti in range(1, T):
            lo, hi = offs[ti]
            if lo != hi and lo >= plens[row] and attn[row, ti] > 0.5:
                mask[ti - 1] = 1.0
        denom = mask.sum().clamp_min(1.0)
        scores[row] = (tok_lp[row] * mask).sum() / denom
    prob_of_t = torch.tensor(prob_of, device=dev)
    preds = []
    for pi in range(len(host_recs)):
        rows = (prob_of_t == pi).nonzero().flatten()
        preds.append(int(rows[int(scores[rows].argmax())] - rows[0]))
    return preds


# ---------------------------------------------------------------- the 3-arm diagnostic
@torch.no_grad()
def run_diagnostic(model, recs, tok, dev, bs=8, two_stream=True, seed=0,
                   key=("relation", "n", "d", "query"), verbose=True):
    """SMOKE-able 3-arm SHUFFLED diagnostic over `recs` (held-out determined queries). Returns a dict with
    per-arm acc-vs-true / acc-vs-shuffled over the SHUFFLE-INFORMATIVE subset (instances where the swapped
    floor determines the query cell to a value DIFFERENT from the true answer — the cases that actually
    discriminate metadata-driven from α-driven)."""
    from .. import run_glados_staged as G
    K = G.K
    model.eval()
    rng = np.random.default_rng(seed + 777)
    perm, shuffleable = struct_perm(recs, rng, key=key)

    # capture composed lattices for TRUE and SHUFFLED (host hidden = instance i in BOTH; only the
    # structure handed to the composer differs). batched.
    true_surv = [None] * len(recs)
    shuf_surv = [None] * len(recs)
    for i in range(0, len(recs), bs):
        chunk = list(range(i, min(i + bs, len(recs))))
        hr = [recs[c] for c in chunk]
        ts = capture_composed(model, hr, hr, tok, dev, two_stream)                 # TRUE
        sr = [recs[perm[c]] for c in chunk]
        ss = capture_composed(model, hr, sr, tok, dev, two_stream)                 # SHUFFLED struct
        for li, c in enumerate(chunk):
            true_surv[c] = ts[li, : recs[c]["n"]].cpu().numpy()
            shuf_surv[c] = ss[li, : recs[c]["n"]].cpu().numpy()

    # the labels: true gold (i's answer) and shuffled gold (the value j's floor put at i's query cell)
    true_gold = [r["gold_idx"] for r in recs]
    shuf_gold = [singleton_at_query(shuf_surv[i], recs[i]["query"]) for i in range(len(recs))]
    true_floor_gold = [singleton_at_query(true_surv[i], recs[i]["query"]) for i in range(len(recs))]

    # informative subset: shuffleable, the swapped floor determined the query cell, and it differs from
    # the true answer (so acc-vs-true and acc-vs-shuffled are actually distinguishable).
    info = [i for i in range(len(recs))
            if shuffleable[i] and shuf_gold[i] is not None and shuf_gold[i] != true_gold[i]
            and shuf_gold[i] < len(recs[i]["vnames"])]    # j's answer must be a reachable candidate for i
    # sanity: how often does the TRUE floor itself determine the query cell to the right answer?
    floor_solves = [i for i in range(len(recs))
                    if true_floor_gold[i] is not None and true_floor_gold[i] == true_gold[i]]

    def arm(struct_idx, inject):
        """Score every record with the lattice composed from struct_idx[i]'s structure (or no lattice)."""
        preds = [None] * len(recs)
        for i in range(0, len(recs), bs):
            chunk = list(range(i, min(i + bs, len(recs))))
            hr = [recs[c] for c in chunk]
            if not inject:
                surv = torch.zeros(len(chunk), max(r["n"] for r in hr), K, device=dev)
                pr = score_preds(model, hr, surv, tok, dev, inject=False)
            else:
                Nmax = max(r["n"] for r in hr)
                surv = torch.zeros(len(chunk), Nmax, K, device=dev)
                src = true_surv if struct_idx is True else shuf_surv
                for li, c in enumerate(chunk):
                    s = src[c]
                    surv[li, : s.shape[0]] = torch.from_numpy(s).to(dev)
                pr = score_preds(model, hr, surv, tok, dev, inject=True)
            for li, c in enumerate(chunk):
                preds[c] = pr[li]
        return preds

    preds_true = arm(True, inject=True)
    preds_shuf = arm(False, inject=True)
    preds_none = arm(None, inject=False)

    def rates(preds, idxs):
        n = len(idxs)
        if n == 0:
            return {"acc_true": float("nan"), "acc_shuf": float("nan"), "abstain": float("nan"), "n": 0}
        at = sum(preds[i] == true_gold[i] for i in idxs) / n
        ash = sum(preds[i] == shuf_gold[i] for i in idxs) / n
        ab = sum(preds[i] == len(recs[i]["vnames"]) for i in idxs) / n     # predicted abstain
        return {"acc_true": at, "acc_shuf": ash, "abstain": ab, "n": n}

    out = {
        "n_total": len(recs), "n_shuffleable": sum(shuffleable),
        "n_informative": len(info), "n_floor_solves": len(floor_solves),
        "TRUE_STRUCT": {**rates(preds_true, info)},
        "SHUFFLED_STRUCT": {**rates(preds_shuf, info)},
        "NO_STRUCT": {**rates(preds_none, info)},
        "_info_idx": info,
        "_preds": {"true": preds_true, "shuffled": preds_shuf, "none": preds_none},
        "_true_gold": true_gold, "_shuf_gold": shuf_gold,
    }

    # per-difficulty breakdown (cheap): by relation, and by cell-count n
    def subgroup(idxs_pred, splitter):
        tab = {}
        for i in info:
            g = splitter(recs[i])
            tab.setdefault(g, []).append(i)
        return {g: rates(idxs_pred, ii) for g, ii in sorted(tab.items())}
    out["SHUFFLED_by_relation"] = subgroup(preds_shuf, lambda r: r["relation"])
    out["SHUFFLED_by_n"] = subgroup(preds_shuf, lambda r: r["n"])

    if verbose:
        print_diagnostic(out)
    return out


def print_diagnostic(out):
    print("\n" + "=" * 78, flush=True)
    print("SHUFFLED DIAGNOSTIC — does α drive correctness, or the handed-in rec['csp']?", flush=True)
    print("=" * 78, flush=True)
    print(f"  instances={out['n_total']}  shuffleable={out['n_shuffleable']}  "
          f"informative(swapped-floor determined & != true)={out['n_informative']}  "
          f"true-floor-solves-query={out['n_floor_solves']}", flush=True)
    samp = out.get("_info_idx", [])[:6]
    if samp:
        print("  STRUCT-SWAP SANITY (informative instances): true_gold vs SHUFFLED-floor_gold "
              "(the value j's certified floor put at i's query cell)", flush=True)
        for i in samp:
            print(f"    inst#{i}: true_gold={out['_true_gold'][i]}  shuffled_floor_gold={out['_shuf_gold'][i]}"
                  f"  pred[true]={out['_preds']['true'][i]} pred[shuf]={out['_preds']['shuffled'][i]}",
                  flush=True)
    print(f"  {'arm':16s} {'acc-vs-TRUE':>12s} {'acc-vs-SHUF':>12s} {'pred-abstain':>13s} {'n':>5s}",
          flush=True)
    for a in ("TRUE_STRUCT", "SHUFFLED_STRUCT", "NO_STRUCT"):
        r = out[a]
        print(f"  {a:16s} {r['acc_true']*100:11.1f}% {r['acc_shuf']*100:11.1f}% "
              f"{r['abstain']*100:12.1f}% {r['n']:5d}", flush=True)
    print("  -- SHUFFLED by relation --", flush=True)
    for g, r in out["SHUFFLED_by_relation"].items():
        print(f"    {str(g):14s} acc-true {r['acc_true']*100:5.1f}%  acc-shuf {r['acc_shuf']*100:5.1f}%"
              f"  (n={r['n']})", flush=True)
    print("  -- SHUFFLED by n(cells) --", flush=True)
    for g, r in out["SHUFFLED_by_n"].items():
        print(f"    n={g:<3} acc-true {r['acc_true']*100:5.1f}%  acc-shuf {r['acc_shuf']*100:5.1f}%"
              f"  (n={r['n']})", flush=True)
    s = out["SHUFFLED_STRUCT"]
    print("  VERDICT: ", end="", flush=True)
    if s["n"] == 0:
        print("INCONCLUSIVE — no informative instances (no structurally-matched swaps).", flush=True)
    elif s["acc_shuf"] >= 0.5 and s["acc_shuf"] - s["acc_true"] > 0.2:
        print(f"METADATA-DRIVEN — SHUFFLED follows the swapped floor (acc-shuf {s['acc_shuf']*100:.0f}% "
              f">> acc-true {s['acc_true']*100:.0f}%). α is a passenger → pretrain NEEDS ALPHA_STRUCT.",
              flush=True)
    elif s["acc_true"] >= s["acc_shuf"]:
        print(f"NOT cleanly metadata-driven — SHUFFLED keeps following truth / does not parrot j "
              f"(acc-true {s['acc_true']*100:.0f}% >= acc-shuf {s['acc_shuf']*100:.0f}%).", flush=True)
    else:
        print(f"MIXED — SHUFFLED partly follows the swapped floor (acc-shuf {s['acc_shuf']*100:.0f}% "
              f"vs acc-true {s['acc_true']*100:.0f}%).", flush=True)
    print("=" * 78 + "\n", flush=True)
