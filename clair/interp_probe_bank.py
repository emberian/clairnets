"""clair/interp_probe_bank.py — LEARNED-FREE interpretability probe on the OVERHAULED bank-woven organ.

This is the clean re-do of the interp probe on the RIGHT target: the consolidated woven organ
(`clair.organ.bank_woven.BankWoven` / `BankComposerOrgan`) — α compiles a per-cell lattice, the
COMPOSER runs the certified bank (Arc/Factor/Modular/GF2/Macro) + the pretrained CoreNarrowOrgan +
α (all verifier-gated), and γ reads the COMPOSED lattice. NOT the old stopgap standalone
LatentNarrower, and NOT the prior probe's learned-MLP decoders.

EMBER'S NON-NEGOTIABLE CONSTRAINT — the readout is LEARNED-FREE. There is NO trained decoder anywhere
in this probe. Every number is either:
  * DIRECT  — read straight off the tensors / the composer's own trace (threshold α's b0 logits at the
              SAME θ the live model uses; read the discrete reduced-product trace; set-compare to the
              exact dedₚ), or
  * BEHAVIORAL — the model's own generative answer-correctness under the validated causal controls.
We do not stack a learned layer onto the interpretation, because every learned layer is a layer of
doubt. (The prior probe, clair/interp_probe.py, fit MLP decoders on α's latent output — exactly the
thing we refuse here. It also probed the standalone LatentOrgan, the torn-down stopgap.)

What we read, learned-free:
  1. WHAT PROBLEM α POSED (DIRECT): threshold α's raw compile b0 → per-cell candidate sets; reverse-
     render to a readable instance; set-compare to the EXACT dedₚ lattice → α-compile faithfulness
     (per-(cell,value) precision/recall/F1, per-cell domain/cardinality accuracy, determined-cell /
     "pin" detection). Where lossy: broken down by rung and by true cardinality.
  2. WHICH FACULTY THE COMPOSER ROUTED TO (DIRECT): read the reduced-product Trace — which bank
     reductions actually fired (narrowed) on which rungs, and which neural proposals were verifier-
     gated. Legible by construction.
  3. WHAT IT CONCLUDED (DIRECT): the COMPOSED lattice → per-cell sets → answer/abstain; matched to the
     verified dedₚ solution.
  4. IS α THE BOTTLENECK (BEHAVIORAL): correlate per-instance α-faithfulness (#1) with the model's
     downstream answer-correctness (does a faithful compile predict a correct answer?). Plus the
     standard causal-control / oracle-override table (readout is solved per the oracle-override arm,
     so α's compile fidelity is the wall).
  5. LEGIBLE TRAJECTORIES: posed factor-graph → α-compiled lattice → which faculty fired → narrowing
     cardinality per round → conclusion → answer, for a few examples.

  SMOKE: python -m clair.interp_probe_bank --base allenai/OLMo-2-0425-1B --smoke
  FULL : python -m clair.interp_probe_bank --base allenai/OLMo-2-0425-1B --regime small \
             --steps 1500 --warm_steps 300 --eval_per_rung 60 --out runs/interp_bank.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time

import numpy as np
import torch

from . import csp as C
from . import run_glados_staged as G
from .oracle_readout import _mention_tensor, ABSTAIN_STR
from .organ import bank_woven as BW


# ============================================================ learned-free relation rendering (display only)
def _entity(i: int) -> str:
    """A, B, ..., Z, A1, B1, ... (cell index → readable label). Display-only, not a metric."""
    return chr(ord("A") + i) if i < 26 else f"{chr(ord('A') + i % 26)}{i // 26}"


def classify_constraint(scope, allowed, d, vnames):
    """Pure-logic (NO learned probe) classifier of a CSP factor into a readable relation, for the TRUE
    factor-graph render in the legible trajectories. Recognizes pin / ≠ / = / < / > / generic by
    comparing the allowed-tuple set to the canonical relation; falls back to listing the factor."""
    sc = tuple(scope)
    vn = lambda v: (vnames[v] if v < len(vnames) else str(v))
    if len(sc) == 1:
        vals = sorted({t[0] for t in allowed})
        if len(vals) == 1:
            return f"{_entity(sc[0])}={vn(vals[0])}"
        return f"{_entity(sc[0])}∈{{{','.join(vn(v) for v in vals)}}}"
    if len(sc) == 2:
        i, j = sc
        al = set(allowed)
        neq = {(a, b) for a in range(d) for b in range(d) if a != b}
        eq = {(a, a) for a in range(d)}
        lt = {(a, b) for a in range(d) for b in range(d) if a < b}
        gt = {(a, b) for a in range(d) for b in range(d) if a > b}
        if al == neq:
            return f"{_entity(i)}≠{_entity(j)}"
        if al == eq:
            return f"{_entity(i)}={_entity(j)}"
        if al == lt:
            return f"{_entity(i)}<{_entity(j)}"
        if al == gt:
            return f"{_entity(i)}>{_entity(j)}"
    return f"f({','.join(_entity(c) for c in sc)})"


def render_true_problem(csp: C.CSP, vnames):
    parts = [classify_constraint(sc, al, csp.d, vnames) for sc, al in csp.cons]
    return ", ".join(parts) if parts else "(no constraints)"


def render_sets(dom, n, d, vnames):
    """Per-cell candidate-set render: 'A=red, B∈{red,green}, C=⊥' (singleton = pinned, empty = ⊥)."""
    vn = lambda v: (vnames[v] if v < len(vnames) else str(v))
    out = []
    for i in range(n):
        s = sorted(v for v in dom[i] if v < d)
        if len(s) == 1:
            out.append(f"{_entity(i)}={vn(s[0])}")
        elif len(s) == 0:
            out.append(f"{_entity(i)}=⊥")
        else:
            out.append(f"{_entity(i)}∈{{{','.join(vn(v) for v in s)}}}")
    return ", ".join(out)


# ============================================================ read α + composer, learned-free, per instance
@torch.no_grad()
def read_alpha_and_compose(model, recs, tok, dev, two_stream, bs):
    """For each record, read DIRECTLY (no learned probe):
       * alpha_dom : α's compiled per-cell candidate sets = {v<d : σ(b0[i,v]) ≥ θ}  (θ = the live model's)
       * composed  : the composer's reduced-product output state (the lattice γ reads)
       * trace     : the reduced-product Trace (which reductions fired / were gated)
       * dedp_dom  : the EXACT dedₚ per-cell sets (rec['tgt'], the J0 target == ground-truth lattice)
    Returns a list of per-record dicts."""
    model.eval()
    theta = model.theta
    K = model.K
    out = []
    for s in range(0, len(recs), bs):
        chunk = recs[s:s + bs]
        ba = BW._bank_batch(chunk, tok, dev, two_stream)
        if two_stream:
            ment, attn, ids = ba["a_mention"], ba["a_attn"], ba["a_input_ids"]
        else:
            ment, attn, ids = ba["mention"], ba["attn"], ba["input_ids"]
        # one α-stream forward; capture α's raw compile b0 (composer also runs but we redo per-instance
        # below to recover each instance's discrete trace, which the live hook discards)
        with model.live(ment, attn, inject=False, capture=True, csps=ba["csps"], dvec=ba["dvec"]):
            _ = model.logits(ids, attn)
        b0 = torch.sigmoid(model._last_b0.float()).cpu().numpy()        # [B,Nmax,K] survival prob
        for b, rec in enumerate(chunk):
            n, csp = rec["n"], rec["csp"]
            d = csp.d
            alpha_dom = tuple(frozenset(v for v in range(d) if b0[b, i, v] >= theta) for i in range(n))
            spec = BW.csp_spec_for_record(rec)
            csp_, system, tags = spec
            composed = model.composer.compose_one(csp_, alpha_dom, system=system, tags=tags)
            trace = model.composer.last_trace
            tgt = rec["tgt"]
            dedp_dom = tuple(frozenset(v for v in range(d) if tgt[i, v] > 0.5) for i in range(n))
            out.append({
                "rec": rec, "n": n, "d": d, "csp": csp, "vnames": rec["vnames"],
                "relation": rec["relation"], "query": rec["query"], "gold_idx": rec["gold_idx"],
                "determined": rec["determined"],
                "alpha_dom": alpha_dom, "composed_dom": tuple(composed.dom[:n]),
                "dedp_dom": dedp_dom, "trace": trace,
                "alpha_prob": b0[b, :n].copy(),
            })
    return out


# ============================================================ behavioral: per-instance answer correctness
@torch.no_grad()
def score_per_instance(model, reads, tok, dev, two_stream, bs, control="true", override_oracle=False):
    """The model's OWN generative answer (LM-head scores the legal answer strings) with the COMPOSED
    lattice (#3, the real woven path) injected by γ — mirrors bank_woven.score_bank_woven but returns
    PER-RECORD correctness. control='true' injects the composed lattice; override_oracle injects the
    exact dedₚ (the readout-isolation reference). Behavioral, learned-free (it is the model generating)."""
    model.eval()
    K = model.K
    perm = tuple(list(range(1, K)) + [0])
    results = []
    for s in range(0, len(reads), bs):
        chunk = reads[s:s + bs]
        Bp = len(chunk)
        Nmax = max(r["n"] for r in chunk)
        dvec = torch.tensor([len(r["vnames"]) for r in chunk], device=dev)
        # build the per-cell survival to inject: composed (true) or exact dedₚ (oracle)
        surv = torch.zeros(Bp, Nmax, K, device=dev)
        for b, r in enumerate(chunk):
            dom = r["dedp_dom"] if override_oracle else r["composed_dom"]
            for i in range(r["n"]):
                for v in dom[i]:
                    if v < K:
                        surv[b, i, v] = 1.0
        surv = G.apply_control_ext(surv, [r["rec"] for r in chunk], control, K, perm)
        # expand each problem over its legal candidate continuations
        fulls, plens, spans, prob_of = [], [], [], []
        for pi, r in enumerate(chunk):
            rec = r["rec"]
            cands = [" " + v for v in rec["vnames"]] + [" " + ABSTAIN_STR]
            for cand in cands:
                fulls.append(rec["prompt"] + cand)
                plens.append(len(rec["prompt"]))
                spans.append({k: [tuple(x) for x in v] for k, v in rec["mentions"].items()})
                prob_of.append(pi)
        enc = tok(fulls, return_offsets_mapping=True, padding=True, return_tensors="pt")
        ids = enc["input_ids"].to(dev); attn = enc["attention_mask"].to(dev)
        offsets = enc["offset_mapping"]; T = ids.size(1)
        mention = _mention_tensor(spans, offsets, len(fulls), Nmax, T, dev)
        prob_of_idx = torch.tensor(prob_of, device=dev)
        with model.live(mention, attn, inject=True, capture=False, override=surv[prob_of_idx],
                        dvec=dvec[prob_of_idx]):
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
        for pi, r in enumerate(chunk):
            rows = (prob_of_idx == pi).nonzero().flatten()
            pred = int(rows[int(scores[rows].argmax())] - rows[0])
            results.append(int(pred == r["gold_idx"]))
    return results


# ============================================================ faithfulness metrics (direct set comparison)
def faithfulness_tables(reads):
    """α-compile faithfulness, computed by DIRECT set comparison of α's compiled per-cell sets vs the
    EXACT dedₚ sets (no learned probe). Per-(cell,value) membership P/R/F1, per-cell exact-set and
    cardinality accuracy, determined-cell (pin) detection. Aggregate, per-rung, and per-true-cardinality."""
    def acc(reads_subset):
        TP = FP = FN = 0
        cells = exact_cells = card_cells = 0
        det_cells = det_pinned_right = 0
        for r in reads_subset:
            n, d = r["n"], r["d"]
            for i in range(n):
                A = set(v for v in r["alpha_dom"][i] if v < d)
                Tt = set(v for v in r["dedp_dom"][i] if v < d)
                TP += len(A & Tt); FP += len(A - Tt); FN += len(Tt - A)
                cells += 1
                exact_cells += int(A == Tt)
                card_cells += int(len(A) == len(Tt))
                if len(Tt) == 1:                                   # truly determined cell
                    det_cells += 1
                    det_pinned_right += int(A == Tt)               # α pinned it to the right singleton
        prec = TP / max(1, TP + FP); rec = TP / max(1, TP + FN)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        return {"precision": prec, "recall": rec, "f1": f1,
                "cellvalue_TP": TP, "cellvalue_FP": FP, "cellvalue_FN": FN,
                "cell_exact_set_acc": exact_cells / max(1, cells),
                "cell_cardinality_acc": card_cells / max(1, cells),
                "determined_cells": det_cells,
                "pin_detect_acc": det_pinned_right / max(1, det_cells),
                "n_cells": cells, "n_instances": len(reads_subset)}

    overall = acc(reads)
    by_rung = {}
    for rg in sorted({r["relation"] for r in reads}):
        by_rung[rg] = acc([r for r in reads if r["relation"] == rg])
    # per-true-cardinality of the cell (how lossy is α as a function of how narrowed the truth is)
    by_card = {}
    buckets = {}
    for r in reads:
        d = r["d"]
        for i in range(r["n"]):
            Tt = set(v for v in r["dedp_dom"][i] if v < d)
            A = set(v for v in r["alpha_dom"][i] if v < d)
            c = len(Tt)
            b = buckets.setdefault(c, [0, 0, 0, 0])       # [cells, exact, recall_num, recall_den]
            b[0] += 1; b[1] += int(A == Tt)
            b[2] += len(A & Tt); b[3] += len(Tt)
    for c, b in sorted(buckets.items()):
        by_card[c] = {"n_cells": b[0], "exact_set_acc": b[1] / max(1, b[0]),
                      "recall": b[2] / max(1, b[3])}
    return {"overall": overall, "by_rung": by_rung, "by_true_cardinality": by_card}


# ============================================================ composer routing (direct, from the trace)
def routing_tables(reads):
    """Which bank faculties actually FIRED (narrowed) on which rungs, and which neural proposals were
    verifier-gated — read straight off the reduced-product Trace. Legible by construction."""
    overall = {}
    gated = {}
    by_rung = {}
    rounds = []
    for r in reads:
        tr = r["trace"]
        rg = r["relation"]
        fired = set()
        if tr is not None:
            rounds.append(tr.rounds)
            for (_rnd, name, before, after) in tr.applied:
                if after < before:                          # actually narrowed (not a no-op/abstain)
                    fired.add(name)
            for name in tr.gated:
                gated[name] = gated.get(name, 0) + 1
        for name in fired:
            overall[name] = overall.get(name, 0) + 1
        d = by_rung.setdefault(rg, {})
        for name in fired:
            d[name] = d.get(name, 0) + 1
        d["_n"] = d.get("_n", 0) + 1
    n = len(reads)
    return {
        "faculties": None,                       # filled by caller from model.composer.faculties()
        "fired_fraction": {k: v / max(1, n) for k, v in sorted(overall.items())},
        "fired_count": dict(sorted(overall.items())),
        "gated_count": dict(sorted(gated.items())),
        "mean_rounds": float(np.mean(rounds)) if rounds else 0.0,
        "by_rung_fired_fraction": {
            rg: {k: v / max(1, d["_n"]) for k, v in d.items() if k != "_n"}
            for rg, d in by_rung.items()},
        "by_rung_n": {rg: d["_n"] for rg, d in by_rung.items()},
    }


# ============================================================ conclusion (#3): composed vs dedₚ / gold
def conclusion_table(reads):
    """The COMPOSED lattice → answer/abstain, matched to the verified dedₚ solution. Direct."""
    q_match = all_match = q_singleton = 0
    composed_sound = 0           # composed ⊇ dedₚ (sound: never drops a real survivor)
    det = det_q_solved = 0
    n = len(reads)
    for r in reads:
        nn, d = r["n"], r["d"]
        cm = ad = sound = True
        for i in range(nn):
            ci = set(v for v in r["composed_dom"][i] if v < d)
            ti = set(v for v in r["dedp_dom"][i] if v < d)
            if ci != ti:
                ad = False
            if not (ti <= ci):
                sound = False
        composed_sound += int(sound)
        all_match += int(ad)
        q = r["query"]
        cq = set(v for v in r["composed_dom"][q] if v < d)
        tq = set(v for v in r["dedp_dom"][q] if v < d)
        q_match += int(cq == tq)
        q_singleton += int(len(cq) == 1)
        if r["determined"]:
            det += 1
            det_q_solved += int(len(cq) == 1 and cq == tq)
    return {"query_cell_matches_dedp": q_match / max(1, n),
            "all_cells_match_dedp": all_match / max(1, n),
            "composed_sound_frac": composed_sound / max(1, n),
            "query_singleton_frac": q_singleton / max(1, n),
            "determined_query_solved": det_q_solved / max(1, det) if det else None,
            "n": n}


# ============================================================ #4: is α the bottleneck?
def bottleneck_table(reads, correct_true, correct_oracle):
    """Correlate per-instance α-faithfulness with the model's downstream answer-correctness. If a
    faithful compile predicts a correct answer (and oracle-override ≫ true), α's compile fidelity is
    the wall — the readout is solved."""
    # per-instance compile fidelity: query-cell exact match AND a per-instance cell-value F1
    qfaith = []
    inst_f1 = []
    for r in reads:
        d = r["d"]
        q = r["query"]
        Aq = set(v for v in r["alpha_dom"][q] if v < d)
        Tq = set(v for v in r["dedp_dom"][q] if v < d)
        qfaith.append(1.0 if Aq == Tq else 0.0)
        TP = FP = FN = 0
        for i in range(r["n"]):
            A = set(v for v in r["alpha_dom"][i] if v < d)
            Tt = set(v for v in r["dedp_dom"][i] if v < d)
            TP += len(A & Tt); FP += len(A - Tt); FN += len(Tt - A)
        p = TP / max(1, TP + FP); rc = TP / max(1, TP + FN)
        inst_f1.append(2 * p * rc / max(1e-9, p + rc))
    qfaith = np.array(qfaith); inst_f1 = np.array(inst_f1)
    ct = np.array(correct_true, dtype=float)
    co = np.array(correct_oracle, dtype=float)

    def corr(x, y):
        if x.std() < 1e-9 or y.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    faith = qfaith > 0.5
    return {
        "n": len(reads),
        "answer_acc_true": float(ct.mean()),
        "answer_acc_oracle_override": float(co.mean()),
        "mean_query_faithful": float(qfaith.mean()),
        "mean_instance_compileF1": float(inst_f1.mean()),
        "corr_queryfaithful_correct": corr(qfaith, ct),
        "corr_compileF1_correct": corr(inst_f1, ct),
        "answer_acc_when_query_faithful": float(ct[faith].mean()) if faith.any() else None,
        "answer_acc_when_query_unfaithful": float(ct[~faith].mean()) if (~faith).any() else None,
        "mean_compileF1_when_correct": float(inst_f1[ct > 0.5].mean()) if (ct > 0.5).any() else None,
        "mean_compileF1_when_wrong": float(inst_f1[ct < 0.5].mean()) if (ct < 0.5).any() else None,
        # the oracle-override gap = how much answer-acc the readout already delivers GIVEN a perfect
        # compile; if this ≫ true, the wall is α's compile (#1), not the readout.
        "oracle_minus_true": float(co.mean() - ct.mean()),
    }


# ============================================================ legible trajectories (#5)
def legible_trajectories(reads, correct_true, n_examples=6):
    """For a spread of examples (cover rungs; include a faithful and an UNfaithful one), render the full
    legible story: TRUE factor-graph → α POSED lattice → which faculty fired (cardinality per round) →
    γ CONCLUDED → answer vs dedₚ. All read directly off the tensors / trace."""
    chosen = []
    seen_rungs = set()
    # one per rung first
    for idx, r in enumerate(reads):
        if r["relation"] not in seen_rungs:
            seen_rungs.add(r["relation"]); chosen.append(idx)
    # ensure at least one UNfaithful example (α dropped/added vs dedₚ somewhere)
    for idx, r in enumerate(reads):
        if idx in chosen:
            continue
        d = r["d"]
        unfaithful = any(set(v for v in r["alpha_dom"][i] if v < d) != set(v for v in r["dedp_dom"][i] if v < d)
                         for i in range(r["n"]))
        if unfaithful:
            chosen.append(idx); break
    chosen = chosen[:n_examples]
    exs = []
    for idx in chosen:
        r = reads[idx]
        n, d, vn = r["n"], r["d"], r["vnames"]
        tr = r["trace"]
        steps = []
        init_alive = n * d
        if tr is not None:
            prev = init_alive
            for (rnd, name, before, after) in tr.applied:
                if after < before:
                    steps.append({"round": rnd, "faculty": name, "alive": after, "dropped": before - after})
                    prev = after
        q = r["query"]
        cq = sorted(v for v in r["composed_dom"][q] if v < d)
        tq = sorted(v for v in r["dedp_dom"][q] if v < d)
        ans = (vn[cq[0]] if (len(cq) == 1 and cq[0] < len(vn)) else f"ABSTAIN({len(cq)} left)")
        dedp_ans = (vn[tq[0]] if (len(tq) == 1 and tq[0] < len(vn)) else f"open({len(tq)})")
        exs.append({
            "rung": r["relation"], "n": n, "query": _entity(q),
            "true_problem": render_true_problem(r["csp"], vn),
            "alpha_posed": render_sets(r["alpha_dom"], n, d, vn),
            "init_alive": init_alive,
            "narrowing": steps,
            "final_rounds": tr.rounds if tr is not None else 0,
            "final_status": tr.final_status if tr is not None else "?",
            "gated": list(tr.gated) if tr is not None else [],
            "composed_conclusion": render_sets(r["composed_dom"], n, d, vn),
            "answer_at_query": ans, "dedp_answer": dedp_ans,
            "model_answer_correct": bool(correct_true[idx]),
        })
    return exs


# ============================================================ printing
def _print_report(model, faith, routing, concl, bott, exs, controls):
    P = lambda *a: print(*a, flush=True)
    P("\n" + "=" * 78)
    P("LEARNED-FREE INTERP PROBE — bank-woven organ (α-compile → composer → γ readout)")
    P("=" * 78)
    P("\n# faculties in the live composer:", model.composer.faculties())

    P("\n===== 1. α-COMPILE FAITHFULNESS (DIRECT: σ(b0)≥θ vs exact dedₚ; no learned probe) =====")
    o = faith["overall"]
    P(f"  per-(cell,value): precision {o['precision']*100:.1f}%  recall {o['recall']*100:.1f}%  "
      f"F1 {o['f1']*100:.1f}%   (TP {o['cellvalue_TP']} FP {o['cellvalue_FP']} FN {o['cellvalue_FN']})")
    P(f"  per-cell exact-set acc {o['cell_exact_set_acc']*100:.1f}%  cardinality acc "
      f"{o['cell_cardinality_acc']*100:.1f}%  | pin(determined-cell) detect {o['pin_detect_acc']*100:.1f}% "
      f"over {o['determined_cells']} det cells")
    P(f"  [{o['n_instances']} instances, {o['n_cells']} cells]   RECALL<100% ⇒ α drops a real survivor "
      f"(gated, but a compile gap); PRECISION<100% ⇒ α keeps a dead value")
    P("\n  WHERE LOSSY — by rung:")
    P(f"    {'rung':<13}{'prec':>7}{'recall':>8}{'exact':>8}{'pin':>7}{'inst':>6}")
    for rg, t in faith["by_rung"].items():
        P(f"    {rg:<13}{t['precision']*100:>6.1f}%{t['recall']*100:>7.1f}%{t['cell_exact_set_acc']*100:>7.1f}%"
          f"{t['pin_detect_acc']*100:>6.1f}%{t['n_instances']:>6}")
    P("  WHERE LOSSY — by TRUE cardinality of the cell (how narrowed the truth is):")
    for c, t in faith["by_true_cardinality"].items():
        P(f"    |dedₚ|={c}: exact-set {t['exact_set_acc']*100:5.1f}%  recall {t['recall']*100:5.1f}%  "
          f"({t['n_cells']} cells)")

    P("\n===== 2. COMPOSER ROUTING (DIRECT: read off the reduced-product trace) =====")
    P(f"  mean rounds to fixpoint: {routing['mean_rounds']:.2f}")
    P("  faculty FIRED fraction (narrowed on ≥1 cell):")
    for k, v in routing["fired_fraction"].items():
        P(f"    {k:<20}{v*100:5.1f}%  ({routing['fired_count'][k]} instances)")
    P(f"  verifier-GATED neural proposals (clamped to the exact verifier): {routing['gated_count'] or '{}'}")
    P("  by-rung faculty firing:")
    for rg, d in routing["by_rung_fired_fraction"].items():
        fired = ", ".join(f"{k} {v*100:.0f}%" for k, v in sorted(d.items(), key=lambda x: -x[1]))
        P(f"    {rg:<13} (n={routing['by_rung_n'][rg]}): {fired or '(none fired)'}")

    P("\n===== 3. CONCLUSION (DIRECT: composed lattice vs verified dedₚ) =====")
    P(f"  query-cell matches dedₚ:   {concl['query_cell_matches_dedp']*100:.1f}%")
    P(f"  ALL cells match dedₚ:      {concl['all_cells_match_dedp']*100:.1f}%")
    P(f"  composed is SOUND (⊇dedₚ): {concl['composed_sound_frac']*100:.1f}%  (certified floor ⇒ should be 100%)")
    P(f"  determined-query solved:   {(concl['determined_query_solved'] or 0)*100:.1f}%")

    P("\n===== 4. IS α THE BOTTLENECK? (BEHAVIORAL: faithfulness ↔ answer-correctness) =====")
    P(f"  answer-acc TRUE (α→composer→γ):   {bott['answer_acc_true']*100:.1f}%")
    P(f"  answer-acc ORACLE-override (dedₚ): {bott['answer_acc_oracle_override']*100:.1f}%   "
      f"(oracle−true = +{bott['oracle_minus_true']*100:.1f}pts)")
    P(f"  mean query-faithful: {bott['mean_query_faithful']*100:.1f}%  mean instance compile-F1: "
      f"{bott['mean_instance_compileF1']*100:.1f}%")
    P(f"  corr(query-faithful, correct) = {bott['corr_queryfaithful_correct']:.3f}   "
      f"corr(compileF1, correct) = {bott['corr_compileF1_correct']:.3f}")
    P(f"  answer-acc | query-FAITHFUL   = {(bott['answer_acc_when_query_faithful'] or 0)*100:.1f}%")
    P(f"  answer-acc | query-UNFAITHFUL = {(bott['answer_acc_when_query_unfaithful'] or 0)*100:.1f}%")
    P(f"  mean compileF1 | correct ans  = {(bott['mean_compileF1_when_correct'] or 0)*100:.1f}%   "
      f"| wrong ans = {(bott['mean_compileF1_when_wrong'] or 0)*100:.1f}%")

    if controls:
        P("\n  causal-control table (the engagement honesty gate):")
        for k in ("true", "shuffle", "permute", "corrupt", "zero", "oracle", "base"):
            if k in controls:
                P(f"    {k:<9}{controls[k]*100:5.1f}%")
        P(f"    gate(tanh α) {controls.get('gate', 0):+.3f}  drop {controls.get('drop',0)*100:+.1f}pts  "
          f"lift {controls.get('lift',0)*100:+.1f}pts  engages={controls.get('engages')}")

    P("\n===== 5. LEGIBLE TRAJECTORIES (posed → faculty → narrowing → conclusion) =====")
    for ex in exs:
        P(f"\n  [{ex['rung']} n={ex['n']} query={ex['query']}]")
        P(f"    TRUE problem:    {ex['true_problem']}")
        P(f"    α POSED lattice: {ex['alpha_posed']}")
        traj = f"init {ex['init_alive']}"
        for st in ex["narrowing"]:
            traj += f" --[{st['faculty']} r{st['round']}: -{st['dropped']}]--> {st['alive']}"
        P(f"    NARROWING:       {traj}   ({ex['final_rounds']} rounds, {ex['final_status']})")
        if ex["gated"]:
            P(f"    (verifier-gated: {ex['gated']})")
        P(f"    γ CONCLUDED:     {ex['composed_conclusion']}")
        P(f"    answer @{ex['query']}: {ex['answer_at_query']}   (dedₚ: {ex['dedp_answer']}; "
          f"model-correct={ex['model_answer_correct']})")


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--regime", default="small", help="small | hard | large")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--warm_steps", type=int, default=300)
    ap.add_argument("--two_stream", type=int, default=1)
    ap.add_argument("--eval_per_rung", type=int, default=60)
    ap.add_argument("--core_ckpt", default="runs/general_organ_full.pt")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="runs/interp_bank.json")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.warm_steps, a.eval_per_rung = 60, 40, 12

    from .organ.train import weave
    dev = G.device()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    two_stream = bool(a.two_stream)

    # ---- train the bank-woven model (the validated engagement recipe; bank mode) ----
    print(f"== TRAIN bank-woven [{a.base} | regime={a.regime} | two_stream={two_stream} | "
          f"steps={a.steps} warm={a.warm_steps}] ==", flush=True)
    t0 = time.time()
    model, controls = weave(a.base, regime=a.regime, two_stream=two_stream, organ_mode="bank",
                            smoke=a.smoke, steps=a.steps, warm_steps=a.warm_steps,
                            core_ckpt=a.core_ckpt, seed=a.seed)
    print(f"  trained in {time.time()-t0:.0f}s  controls={ {k: round(v,3) if isinstance(v,float) else v for k,v in controls.items() if k in ('true','zero','corrupt','oracle','gate','engages')} }",
          flush=True)
    tok = model_tok(a.base)

    # ---- held-out eval pool (det_only: sharp right→wrong signal for the bottleneck test) ----
    rungs, split = G.LIVE_REGIMES[G._REGIME_ALIAS.get(a.regime, a.regime)]
    rng = np.random.default_rng(a.seed + 777)
    eval_recs = G.build_live_pool(rng, rungs, split, a.eval_per_rung, det_only=True,
                                  two_stream=two_stream)
    print(f"\n== PROBE on {len(eval_recs)} held-out records (rungs={rungs} split={split}) ==", flush=True)

    # ---- read α + composer DIRECTLY (learned-free) ----
    reads = read_alpha_and_compose(model, eval_recs, tok, dev, two_stream, a.bs)
    # ---- behavioral: per-instance correctness (true composed lattice + oracle override) ----
    correct_true = score_per_instance(model, reads, tok, dev, two_stream, a.bs, control="true")
    correct_oracle = score_per_instance(model, reads, tok, dev, two_stream, a.bs, control="true",
                                        override_oracle=True)

    faith = faithfulness_tables(reads)
    routing = routing_tables(reads); routing["faculties"] = model.composer.faculties()
    concl = conclusion_table(reads)
    bott = bottleneck_table(reads, correct_true, correct_oracle)
    exs = legible_trajectories(reads, correct_true)

    _print_report(model, faith, routing, concl, bott, exs, controls)

    results = {"args": vars(a), "base": a.base, "regime": a.regime, "rungs": rungs, "split": split,
               "n_eval": len(reads), "controls": {k: (float(v) if isinstance(v, (int, float)) else v)
                                                   for k, v in controls.items()},
               "faithfulness": faith, "routing": routing, "conclusion": concl,
               "bottleneck": bott, "trajectories": exs}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=1, default=str)
    print("\nwrote", a.out, flush=True)


def model_tok(base_id):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_id)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


if __name__ == "__main__":
    main()
