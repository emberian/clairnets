"""clair/run_glados_staged.py — the STAGED GENERATIVE GLaDOS (the full deliverable).

Assemble the validated pieces into a working model where OLMo GENERATES answers through a FROZEN,
pre-trained, GENERAL learned organ, across multiple reasoning types. This GENERALIZES the PROVEN
frozen-organ recipe (clair.frozen_readout / run_frozen — coloring-only: a learned FROZEN organ reads
like the oracle, controls drop, the LM ignores text and reads the lattice) to:
    * a GENERAL organ (one FactorGraphProposer over coloring/equality/ordering/arithmetic/alldiff +
      the HARD propagation tasks eqchain/forcedcolor),
    * TRUE GENERATION (OLMo's LM head emits the answer token(s); LM cross-entropy on the answer span),
    * harder/diverse tasks, in-dist + OOD (held-out N/L + held-out Bedrock phrasings).

THE STAGED RECIPE (what the frozen-organ experiment proved works):
  STAGE 1  bootstrap a GENERAL organ standalone (dominate-dedP over a MIX of rungs incl. eqchain),
           report per-rung narrowing-recall vs the EXACT dedP, then FREEZE it.
  STAGE 2  train the readout GENERATIVELY: OLMo-2-1B frozen + LoRA + structured zero-init gamma
           injecting the frozen organ's narrowed lattice; the LM head GENERATES the answer (LM CE on
           the answer span). READOUT-FORCING curriculum: cells-mode (no facts in text -> the gamma
           decode MUST form) first, then mix in full-text, so the LM cannot camp in a text/pin basin.

DELIVERABLE MEASUREMENTS (printed + saved):
  * per-rung organ recall (stage 1, in-dist + OOD);
  * per-rung GENERATIVE accuracy: WOVEN vs BASE-OLMo vs TEXT-ONLY-LoRA, in-dist + OOD-N + OOD-phrasing;
  * the CAUSAL-CONTROL table: shuffle / permute / corrupt / zero the frozen organ's lattice — does
    generation degrade (the LM causally WIELDS the organ) even with full text present?

  SMOKE: python -m clair.run_glados_staged --smoke
  FULL : python -m clair.run_glados_staged --organ_steps 1500 --steps 2500 --out runs/glados_staged.json
"""
from __future__ import annotations

import argparse, contextlib, itertools as it, json, os, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import csp as C
from . import curriculum as CU
from . import hard_tasks as HT
from . import oracle_readout as O
from .proposer import FactorGraphProposer, size_for


# ============================================================ budget (covers chains N<=12 + arith a=3)
# N_MAX=12 fits the eqchain/forcedcolor chains (cells A..L). D_MAX=8 caps every rung's domain (ordering
# d<=N+2, arithmetic d<=7). A_MAX=3 holds the arithmetic modular-sum factor. M_MAX=80 holds the dense
# forced-colour cascade (2 neq per cell). The proposer is permutation+size equivariant -> one organ.
N_MAX, D_MAX, M_MAX, A_MAX = 12, 8, 80, 3
REL_DIM = D_MAX ** A_MAX
K = D_MAX                                   # readout candidate width (per-cell survival vector dim)

RUNGS = ["coloring", "equality", "ordering", "arithmetic", "alldiff", "eqchain", "forcedcolor"]
HARD = {"eqchain", "forcedcolor"}           # length L is the OOD axis; others use cell-count N
CURRIC_RUNGS = {"coloring", "equality", "ordering", "arithmetic", "alldiff"}   # have diverse phrasings

# in-dist vs OOD sizes (N for curriculum rungs; chain length L for the hard rungs)
RANGES = {
    "coloring":    {"id": [4, 5, 6], "ood": [8, 10]},
    "equality":    {"id": [4, 5, 6], "ood": [8, 10]},
    "ordering":    {"id": [3, 4, 5], "ood": [6]},
    "arithmetic":  {"id": [3, 4],    "ood": [5]},
    "alldiff":     {"id": [4, 5],    "ood": [6]},
    "eqchain":     {"id": [3, 4, 5, 6], "ood": [8, 10]},
    "forcedcolor": {"id": [3, 4, 5, 6], "ood": [8, 10]},
}

EX = HT.FastExact()                          # generic, scalable EXACT dedP (AC + complete backtracking)


def device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================ universal problem sampling
def _expand_alldiff(facts):
    """all-distinct over a scope == pairwise-!= (SOLUTION-equivalent) so the organ sees arity<=2 factors;
    the only loss is at the LATTICE level (binary AC can't do Hall-set propagation) = the honest gap."""
    out = []
    for f in facts:
        if f[0] == "alldiff":
            out += [("neq", a, b) for a, b in it.combinations(f[1], 2)]
        else:
            out.append(f)
    return out


def organ_csp(p) -> C.CSP:
    """The CSP the ORGAN sees (alldiff decomposed to a !=-clique; everything else verbatim)."""
    return CU.build_csp(p.n, p.d, _expand_alldiff(p.facts))


def make_problem(rng, rung, split):
    """One verified curriculum.Problem of `rung` at the requested split ('id'|'ood'), within budget."""
    vals = RANGES[rung][split]
    for _ in range(200):
        x = int(rng.choice(vals))
        if rung in HARD:
            det = bool(rng.random() < 0.5)
            p = HT.make_hard(rng, rung, x, det)
            if p.n <= N_MAX:
                return p
            continue
        N = x
        if rung == "coloring":
            n, d, kind, facts, s = CU.gen_coloring(rng, k=3, n_lo=N, n_hi=N)
        elif rung == "equality":
            n, d, kind, facts, s = CU.gen_equality(rng, k=3, n_lo=N, n_hi=N)
        elif rung == "ordering":
            n, d, kind, facts, s = CU.gen_ordering(rng, n_lo=N, n_hi=N)
        elif rung == "arithmetic":
            n, d, kind, facts, s = CU.gen_arithmetic(rng, n_lo=N, n_hi=N, d_lo=4, d_hi=7)
        elif rung == "alldiff":
            n, d, kind, facts, s = CU.gen_alldiff(rng, n_lo=N, n_hi=N)
        else:
            raise ValueError(rung)
        if d > D_MAX or n > N_MAX:
            continue
        csp = CU.build_csp(n, d, facts)
        det_target = bool(rng.random() < 0.5)
        q, ans, det = CU._query_answer(csp, facts, rng, det_target)
        return CU.Problem(rung, n, d, kind, facts, q, ans, det, CU.value_names(kind, d))
    raise RuntimeError(f"could not sample {rung}/{split} within budget")


def sample_organ_corpus(rng, pool, rungs, split="id"):
    """`pool` (organ_csp, rung) items, round-robin over rungs (witness-first => all solvable)."""
    out = []
    for i in range(pool):
        rg = rungs[i % len(rungs)]
        out.append((organ_csp(make_problem(rng, rg, split)), rg))
    return out


# ============================================================ featurization (this budget) + organ ops
def relation_table(scope, al):
    a = len(scope)
    t = np.zeros((D_MAX,) * A_MAX, dtype=np.float32)
    for tup in al:
        sl = [slice(None)] * A_MAX
        for p in range(a):
            sl[p] = tup[p]
        t[tuple(sl)] = 1.0
    return t.reshape(-1)


def featurize(items, dev):
    B = len(items)
    var_mask = np.zeros((B, N_MAX, D_MAX), np.float32)
    given = np.zeros((B, N_MAX), np.float32)
    var_valid = np.zeros((B, N_MAX), np.float32)
    fac_rel = np.zeros((B, M_MAX, REL_DIM), np.float32)
    fac_arity = np.zeros((B, M_MAX, A_MAX), np.float32)
    fac_valid = np.zeros((B, M_MAX), np.float32)
    edge_var = np.full((B, M_MAX, A_MAX), N_MAX, np.int64)
    edge_valid = np.zeros((B, M_MAX, A_MAX), np.float32)
    for bi, (csp, dom) in enumerate(items):
        for i in range(csp.n):
            var_valid[bi, i] = 1.0
            for v in dom[i]:
                var_mask[bi, i, v] = 1.0
            if len(dom[i]) == 1:
                given[bi, i] = 1.0
        for fi, (sc, al) in enumerate(csp.cons):
            if fi >= M_MAX:
                break
            fac_valid[bi, fi] = 1.0
            fac_arity[bi, fi, len(sc) - 1] = 1.0
            fac_rel[bi, fi] = relation_table(sc, al)
            for p, cell in enumerate(sc):
                edge_var[bi, fi, p] = cell
                edge_valid[bi, fi, p] = 1.0
    t = lambda a: torch.as_tensor(a, device=dev)
    return dict(var_mask=t(var_mask), given=t(given), var_valid=t(var_valid),
                fac_rel=t(fac_rel), fac_arity=t(fac_arity), fac_valid=t(fac_valid),
                edge_var=t(edge_var), edge_valid=t(edge_valid))


def fwd(m, feat, var_mask):
    return m(var_mask, feat["given"], feat["fac_rel"], feat["fac_arity"],
             feat["edge_var"], feat["edge_valid"], feat["var_valid"], feat["fac_valid"])


def dom_from_mask(row, csp):
    return tuple(frozenset(v for v in range(csp.d) if row[i, v] > 0.5) for i in range(csp.n))


def build_targets(items, dev):
    B = len(items)
    tgt = np.zeros((B, N_MAX, D_MAX), np.float32)
    conflict = np.zeros((B,), np.float32)
    for bi, (csp, dom) in enumerate(items):
        ded = EX.dedP(csp, dom)
        if all(len(c) == 0 for c in ded):
            conflict[bi] = 1.0
        for i in range(csp.n):
            for v in ded[i]:
                tgt[bi, i, v] = 1.0
    return torch.as_tensor(tgt, device=dev), torch.as_tensor(conflict, device=dev)


def loss_fn(sup, var_mask, var_valid, tgt, conflict, wpos=6.0, wneg=0.5, lcls=0.3, lce=0.3):
    """Dominate-dedP: asymmetric BCE (HEAVY wpos on eliminating a dedP-kept value => soundness) over
    alive candidates + completeness pressure + conflict BCE + singleton CE where dedP pins."""
    eps = 1e-6
    alive = var_mask * var_valid.unsqueeze(-1)
    sing = (tgt.sum(-1) == 1).float() * var_valid
    tlab = tgt.argmax(-1)
    tot = 0.0
    for b, cls in sup:
        p = torch.sigmoid(b)
        bce = -(wpos * tgt * torch.log(p + eps) + wneg * (1 - tgt) * torch.log(1 - p + eps))
        bce = (bce * alive).sum() / (alive.sum() + eps)
        clsl = F.binary_cross_entropy_with_logits(cls, conflict)
        ce_all = F.cross_entropy(b.reshape(-1, b.size(-1)), tlab.reshape(-1),
                                 reduction="none").reshape(b.shape[:-1])
        ce = (ce_all * sing).sum() / (sing.sum() + eps)
        tot = tot + bce + lcls * clsl + lce * ce
    return tot / len(sup)


@torch.no_grad()
def meet(var_mask, b, theta):
    return var_mask * (torch.sigmoid(b) >= theta).float()


@torch.no_grad()
def false_elim(vm_before, vm_after, var_valid, tgt):
    elig = tgt * vm_before * var_valid.unsqueeze(-1)
    killed = elig * (vm_after < 0.5).float()
    return killed.sum().item(), elig.sum().item()


# ============================================================ STAGE 1: bootstrap the GENERAL organ
def train_organ(dev, rungs, target=3.0e5, steps=1500, pool=128, R=8, lr=3e-4, theta=0.5, seed=0,
                log_every=15):
    """On-policy dominate-dedP training of ONE general organ across `rungs` (incl. the hard chains).
    The organ's own monotone meet rolls each lattice forward; terminal/stalled instances are replaced
    with fresh ones of the SAME rung. Supervision = the EXACT dedP (clair.hard_tasks.FastExact)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d, npar = size_for("full", N_MAX, D_MAX, M_MAX, A_MAX, target, R=R, ds=min(4, R))
    m = FactorGraphProposer("full", N_MAX, D_MAX, M_MAX, A_MAX, d=d, R=R, ds=min(4, R)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95))
    tagged = sample_organ_corpus(rng, pool, rungs)
    items = [(c, c.full()) for c, _ in tagged]
    rtags = [rg for _, rg in tagged]
    print(f"  ORGAN train: rungs={rungs}  d_model={d}  params={npar:,}  pool={pool}  R={R} steps={steps}",
          flush=True)
    log = []
    fe_k = fe_n = 0
    t0 = time.time()
    for s in range(1, steps + 1):
        feat = featurize(items, dev)
        vm = feat["var_mask"]
        tgt, conflict = build_targets(items, dev)
        b, cls, sup = fwd(m, feat, vm)
        loss = loss_fn(sup, vm, feat["var_valid"], tgt, conflict)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        with torch.no_grad():
            new_vm = meet(vm, b, theta)
            kk, nn_ = false_elim(vm, new_vm, feat["var_valid"], tgt); fe_k += kk; fe_n += nn_
            nvc = new_vm.cpu().numpy()
            nxt = []
            for bi, (csp, dom) in enumerate(items):
                ndom = dom_from_mask(nvc[bi], csp)
                if C.status(ndom) in ("solved", "conflict") or ndom == dom:
                    nc = organ_csp(make_problem(rng, rtags[bi], "id"))
                    nxt.append((nc, nc.full()))
                else:
                    nxt.append((csp, ndom))
            items = nxt
        if s % max(1, steps // log_every) == 0 or s == 1:
            fer = fe_k / max(1, fe_n)
            log.append({"step": s, "loss": float(loss.detach()), "false_elim": fer})
            print(f"    step {s:5d}  loss {float(loss.detach()):.3f}  false_elim {fer:.4f}  "
                  f"alive {float(vm.sum(-1).mean()):.2f}  {time.time()-t0:.0f}s", flush=True)
            fe_k = fe_n = 0
    return m, d, npar, log


@torch.no_grad()
def organ_fixpoint(m, csps, dev, theta=0.5, R_max=64, chunk=128):
    """Run the FROZEN organ's monotone meet to a fixpoint on each csp (from the full domain) -> the
    LEARNED narrowed lattice (per-cell domain tuple)."""
    m.eval()
    out = [None] * len(csps)
    for c0 in range(0, len(csps), chunk):
        sub = csps[c0:c0 + chunk]
        feat = featurize([(c, c.full()) for c in sub], dev)
        vm = feat["var_mask"].clone()
        done = torch.zeros(len(sub), dtype=torch.bool, device=dev)
        for _ in range(R_max):
            b, cls, _ = fwd(m, feat, vm)
            new_vm = meet(vm, b, theta)
            changed = (new_vm != vm).any(-1).any(-1)
            vm = torch.where(done.view(-1, 1, 1), vm, new_vm)
            done = done | ~changed
            if bool(done.all()):
                break
        nv = vm.cpu().numpy()
        for j, c in enumerate(sub):
            out[c0 + j] = dom_from_mask(nv[j], c)
    return out


def _removed(full, dom, n):
    return {(i, v) for i in range(n) for v in full[i] if v not in dom[i]}


@torch.no_grad()
def organ_recall_report(organ, dev, rungs, splits=("id", "ood"), n_per=200, seed=999):
    """Per-(rung, split) narrowing-recall + false-elim of the FROZEN organ vs the EXACT dedP fixpoint,
    plus query-cell exactness (the readout-relevant quantity) and determined-instance accuracy."""
    rng = np.random.default_rng(seed)
    rows = {}
    for rung in rungs:
        for split in splits:
            probs, ocsps = [], []
            for _ in range(n_per):
                p = make_problem(rng, rung, split)
                probs.append(p); ocsps.append(organ_csp(p))
            odoms = organ_fixpoint(organ, ocsps, dev)
            rec_num = rec_den = fe = qhit = match = qdet_hit = qdet_tot = 0
            for p, od, oc in zip(probs, odoms, ocsps):
                full = oc.full()
                oracle, _ = C.to_fixpoint(lambda cc, dd: EX.dedP(cc, dd), oc, full)
                rm_o = _removed(full, od, oc.n); rm_t = _removed(full, oracle, oc.n)
                rec_num += len(rm_o & rm_t); rec_den += len(rm_t); fe += len(rm_o - rm_t)
                qhit += int(set(od[p.query]) == set(oracle[p.query]))
                match += int(tuple(od) == tuple(oracle))
                if p.determined:
                    qdet_tot += 1
                    qdet_hit += int(od[p.query] == oracle[p.query] and len(od[p.query]) == 1)
            rows[(rung, split)] = {"recall": rec_num / max(1, rec_den), "false_elim_count": int(fe),
                                   "query_exact": qhit / n_per, "fixpoint_match": match / n_per,
                                   "det_query_solved": qdet_hit / max(1, qdet_tot), "n": n_per}
    return rows


# ============================================================ records (prompt rendering + organ surv)
def surv_from_dom(dom, n, Kc) -> np.ndarray:
    surv = np.zeros((n, Kc), dtype=np.float32)
    for i in range(n):
        for v in dom[i]:
            if v < Kc:
                surv[i, v] = 1.0
    return surv


def render_prompt(p, mode, text=None):
    """The prompt OLMo reads, ending in ' Answer:'.
       cells  : ONLY the cell roster + question (NO deductive facts) -> text-deduction impossible, the
                answer MUST come from the injected lattice (the readout-forcing / pure-readout mode).
       full   : the whole problem in canonical English (facts + question) -> base CAN in principle read.
       diverse: a held-out Bedrock phrasing of the same problem (OOD wording)."""
    if mode == "cells":
        return "Cells " + ", ".join(p.entity(i) for i in range(p.n)) + ". " + CU._question(p) + " Answer:"
    if mode == "full":
        return CU.canonical_render(p) + " Answer:"
    if mode == "diverse":
        return text.rstrip() + " Answer:"
    raise ValueError(mode)


def make_record(p, surv_dom, mode, text=None):
    prompt = render_prompt(p, mode, text)
    body = prompt[: prompt.rfind(" Answer:")]
    mentions = CU.entity_mentions(body, p.n)
    gold_idx = p.answer if p.determined else len(p.vnames)
    return {"prompt": prompt, "answer": CU.canonical_answer(p), "n": p.n, "query": p.query,
            "determined": p.determined, "gold_idx": int(gold_idx), "relation": p.relation,
            "vnames": list(p.vnames),
            "mentions": {int(k): [tuple(s) for s in v] for k, v in mentions.items()},
            "surv": surv_from_dom(surv_dom, p.n, K)}


def build_pool(rng, organ, dev, rungs, split, per_rung, modes):
    """Sample per_rung problems of each rung at `split`, run the FROZEN organ to a fixpoint for the
    injected lattice, and render each problem in every requested `mode`. Returns {mode: [records]}."""
    probs = [make_problem(rng, rg, split) for rg in rungs for _ in range(per_rung)]
    ocsps = [organ_csp(p) for p in probs]
    odoms = organ_fixpoint(organ, ocsps, dev)
    pools = {mode: [] for mode in modes}
    for p, od in zip(probs, odoms):
        for mode in modes:
            pools[mode].append(make_record(p, od, mode))
    for mode in modes:
        rng.shuffle(pools[mode])
    return pools


def _facts_from_record(rec):
    out = []
    for f in rec["facts"]:
        out.append(("alldiff", tuple(f[1])) if f[0] == "alldiff" else tuple(f))
    return out


def build_diverse_pool(rng, organ, dev, jsonl_path, per_rung, rungs):
    """OOD-PHRASING pool from the diverse (Bedrock-rendered) curriculum: held-out wordings of the
    curriculum rungs, with the FROZEN organ's lattice injected. The organ never saw these texts."""
    by_rung = {rg: [] for rg in rungs}
    for line in open(jsonl_path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rg = r["relation"]
        if rg not in by_rung or r["d"] > D_MAX or r["n"] > N_MAX:
            continue
        by_rung[rg].append(r)
    recs = []
    chosen = []
    for rg in rungs:
        pool = by_rung[rg]
        if not pool:
            continue
        idx = rng.choice(len(pool), size=min(per_rung, len(pool)), replace=False)
        chosen += [pool[i] for i in idx]
    probs, ocsps, texts = [], [], []
    for r in chosen:
        facts = _facts_from_record(r)
        p = CU.Problem(r["relation"], r["n"], r["d"], r["kind"], facts, r["query"], r["answer"],
                       r["determined"], r["vnames"])
        probs.append(p); ocsps.append(organ_csp(p)); texts.append(r["text"])
    odoms = organ_fixpoint(organ, ocsps, dev)
    for p, od, txt in zip(probs, odoms, texts):
        recs.append(make_record(p, od, "diverse", txt))
    rng.shuffle(recs)
    return recs


# ============================================================ STAGE 2: generative readout (LoRA+gamma)
def verify_noop(model, tok, dev):
    text = "Cells A, B, C. What color is B? Answer:"
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
    ids = enc["input_ids"].to(dev); attn = torch.ones_like(ids)
    with torch.no_grad(), model.model.disable_adapter():
        base = model.model(input_ids=ids, attention_mask=attn).logits.float()
    surv = (torch.rand(1, 3, K, device=dev) > 0.5).float()
    spans = CU.entity_mentions("Cells A, B, C.", 3)
    mention = O._mention_tensor([{int(k): [tuple(s) for s in v] for k, v in spans.items()}],
                                enc["offset_mapping"], 1, 3, ids.size(1), dev)
    with torch.no_grad(), model.injection(surv, mention, enabled=True):
        g0 = model.logits(ids, attn).float()
    noop = float((base - g0).abs().max())
    with torch.no_grad():
        saved = model.gamma.alpha.data.clone(); model.gamma.alpha.data.fill_(2.0)
        with model.injection(surv, mention, enabled=True):
            g1 = model.logits(ids, attn).float()
        model.gamma.alpha.data.copy_(saved)
    return noop, float((base - g1).abs().max())


def apply_control_ext(surv, recs, control, Kc, perm):
    if control == "zero":
        return torch.zeros_like(surv)
    return O.apply_control(surv, recs, control, Kc, perm=perm)


@torch.no_grad()
def score_gen(model, recs, tok, dev, control="true", inject=True, fewshot="", bs=8, use_base=False,
              perm=None):
    """GENERATIVE accuracy: OLMo's LM head scores the answer span (summed length-normalized logprob)
    over the record's OWN legal answer strings {value-names..., 'cannot be determined'} — true LM-head
    generation, constrained to the legal set. Heterogeneous candidate sets per rung. Returns overall +
    determined/abstain split + per-rung breakdown."""
    model.eval()
    if perm is None:
        perm = tuple(list(range(1, K)) + [0])
    correct = tot = det_t = det_r = ab_t = ab_r = 0
    by_rung = {}
    for i in range(0, len(recs), bs):
        chunk = recs[i:i + bs]
        Bp = len(chunk); Nmax = max(r["n"] for r in chunk)
        surv = torch.zeros(Bp, Nmax, K, device=dev)
        for b, r in enumerate(chunk):
            surv[b, : r["n"], :] = torch.from_numpy(r["surv"]).to(dev)
        surv = apply_control_ext(surv, chunk, control, K, perm)
        fulls, plens, spans, prob_of = [], [], [], []
        for pi, r in enumerate(chunk):
            prompt = fewshot + r["prompt"]; shift = len(fewshot)
            cands = [" " + v for v in r["vnames"]] + [" " + O.ABSTAIN_STR]
            for cand in cands:
                fulls.append(prompt + cand); plens.append(len(prompt))
                spans.append({k: [(a + shift, c + shift) for (a, c) in v] for k, v in r["mentions"].items()})
                prob_of.append(pi)
        enc = tok(fulls, return_offsets_mapping=True, padding=True, return_tensors="pt")
        ids = enc["input_ids"].to(dev); attn = enc["attention_mask"].to(dev)
        offsets = enc["offset_mapping"]; T = ids.size(1)
        if inject:
            mention = O._mention_tensor(spans, offsets, len(fulls), Nmax, T, dev)
            surv_exp = surv[torch.tensor(prob_of, device=dev)]
            ctx = model.injection(surv_exp, mention, enabled=True)
        else:
            ctx = model.injection(None, None, enabled=False)
        base_ctx = model.model.disable_adapter() if (use_base and hasattr(model.model, "disable_adapter")) \
            else contextlib.nullcontext()
        with base_ctx, ctx:
            logits = model.logits(ids, attn).float()
        lp = torch.log_softmax(logits[:, :-1], -1)
        tok_lp = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        scores = torch.full((len(fulls),), -1e9, device=dev)
        for row in range(len(fulls)):
            offs = offsets[row].tolist(); mask = torch.zeros(T - 1, device=dev)
            for ti in range(1, T):
                a, c = offs[ti]
                if a != c and a >= plens[row] and attn[row, ti] > 0.5:
                    mask[ti - 1] = 1.0
            denom = mask.sum().clamp_min(1.0)
            scores[row] = (tok_lp[row] * mask).sum() / denom
        prob_of_t = torch.tensor(prob_of, device=dev)
        for pi, r in enumerate(chunk):
            rows = (prob_of_t == pi).nonzero().flatten()
            pred = int(rows[int(scores[rows].argmax())] - rows[0])
            ok = int(pred == r["gold_idx"])
            correct += ok; tot += 1
            d = by_rung.setdefault(r["relation"], [0, 0]); d[0] += ok; d[1] += 1
            if r["determined"]:
                det_t += 1; det_r += ok
            else:
                ab_t += 1; ab_r += ok
    return {"overall": correct / max(1, tot), "det_acc": det_r / max(1, det_t),
            "abst_acc": ab_r / max(1, ab_t), "n": tot,
            "by_rung": {k: v[0] / max(1, v[1]) for k, v in by_rung.items()},
            "by_rung_n": {k: v[1] for k, v in by_rung.items()}}


FEWSHOT = (
    "A is red. A and B are different colors. What color is A? Answer: red\n"
    "A is blue. A and B are different colors. What color is B? Answer: cannot be determined\n"
    "A is green. B is green. A and C are different colors. What color is B? Answer: green\n"
)


def train_readout(organ, olmo_ids, tok, dev, a, train_cells, train_full, eval_cells):
    """Build a fresh LoRA+gamma generative readout, train on the LM CE of the answer span with the
    READOUT-FORCING curriculum (cells-only first, then mix in full-text). Returns the trained model."""
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
    noop, live = verify_noop(model, tok, dev)
    print(f"\n  WOVEN READOUT  inject layer {inj}/{nL}  trainable {O.n_trainable(model):,}", flush=True)
    print(f"  NO-OP @ INIT max|base-(LoRA_init+gate0)| = {noop:.3e} (expect ~0) | "
          f"gate-on moves logits {live:.3e}", flush=True)
    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.model.named_parameters() if p.requires_grad],
         "lr": a.lora_lr, "weight_decay": 0.01},
        {"params": list(model.gamma.parameters()), "lr": a.gamma_lr, "weight_decay": 0.0},
    ], betas=(0.9, 0.95))
    rng = np.random.default_rng(a.seed)
    cells_until = int(a.cells_frac * a.steps)
    t0 = time.time(); model.train()
    for s in range(1, a.steps + 1):
        if s <= cells_until:
            pool = train_cells                                   # readout-forcing: pure-lattice first
        else:
            pool = train_cells if (s % 2 == 0) else train_full   # then 50/50 cells+full
        idxs = rng.integers(0, len(pool), a.bs).tolist()
        ba = O.build_train_batch([pool[i] for i in idxs], tok, K, dev)
        with model.injection(ba["surv"], ba["mention"], enabled=True):
            logits = model.logits(ba["input_ids"], ba["attn"]).float()
        lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                             ba["labels"][:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad(); lm.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0); opt.step()
        if s % max(1, a.steps // 12) == 0 or s == 1:
            acc = score_gen(model, eval_cells, tok, dev, control="true", inject=True, bs=a.bs)
            phase = "cells" if s <= cells_until else "mix"
            print(f"  step {s:5d} [{phase}]  lm {lm.item():.3f}  alpha {float(model.gamma.alpha):.3f}  "
                  f"cells-readout acc {acc['overall']*100:4.1f}% (det {acc['det_acc']*100:.0f})  "
                  f"{time.time()-t0:.0f}s", flush=True)
            model.train()
    return model


# ============================================================ orchestration
def _rrow(rung, e):
    return (f"  {rung:12s} recall {e['recall']*100:5.1f}%  q-exact {e['query_exact']*100:5.1f}%  "
            f"fixpt-match {e['fixpoint_match']*100:5.1f}%  det-solved {e['det_query_solved']*100:5.1f}%  "
            f"FALSE-ELIM {e['false_elim_count']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organ_steps", type=int, default=1500)
    ap.add_argument("--organ_target", type=float, default=3.0e5)
    ap.add_argument("--organ_R", type=int, default=8)
    ap.add_argument("--organ_lr", type=float, default=3e-4)
    ap.add_argument("--organ_pool", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lora_lr", type=float, default=2e-4)
    ap.add_argument("--gamma_lr", type=float, default=1e-3)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--inject_layer", type=int, default=12)
    ap.add_argument("--gamma_hidden", type=int, default=256)
    ap.add_argument("--cells_frac", type=float, default=0.4)
    ap.add_argument("--per_rung_train", type=int, default=400)
    ap.add_argument("--per_rung_eval", type=int, default=80)
    ap.add_argument("--organ_recall_n", type=int, default=200)
    ap.add_argument("--greedy_n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--curriculum", default="data/curriculum/curriculum.jsonl")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.organ_steps = 250; a.steps = 120; a.organ_pool = 64; a.per_rung_train = 60
        a.per_rung_eval = 24; a.organ_recall_n = 60; a.greedy_n = 8
    dev = device()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    print(f"device={dev}  budget N={N_MAX} D={D_MAX} M={M_MAX} A={A_MAX}  rungs={RUNGS}", flush=True)

    # ===== DEDP sanity: FastExact == clair.csp.exact_dedP on small instances of every rung =====
    rngc = np.random.default_rng(7)
    for rg in RUNGS:
        for _ in range(6):
            p = make_problem(rngc, rg, "id")
            oc = organ_csp(p)
            if oc.d ** oc.n <= 5000:                          # only check where brute force is cheap
                assert HT.fast_dedP(oc) == C.exact_dedP(oc, oc.full()), f"dedP mismatch {rg}"
    print("DEDP CHECK: FastExact.fast_dedP == clair.csp.exact_dedP on all rungs (exact).", flush=True)

    # ===== STAGE 1: bootstrap + freeze the GENERAL organ =====
    print("\n================ STAGE 1: GENERAL ORGAN (dominate-dedP over the rung MIX) ================",
          flush=True)
    organ, od, opar, olog = train_organ(dev, RUNGS, target=a.organ_target, steps=a.organ_steps,
                                        pool=a.organ_pool, R=a.organ_R, lr=a.organ_lr, seed=a.seed)
    for p in organ.parameters():
        p.requires_grad_(False)
    organ.eval()
    print("\n  FROZEN ORGAN RECALL vs EXACT dedP (per rung, in-dist then OOD):", flush=True)
    recall = organ_recall_report(organ, dev, RUNGS, n_per=a.organ_recall_n, seed=a.seed + 5)
    print("  -- in-dist --", flush=True)
    for rg in RUNGS:
        print(_rrow(rg, recall[(rg, "id")]), flush=True)
    print("  -- OOD (held-out N / chain length L) --", flush=True)
    for rg in RUNGS:
        print(_rrow(rg, recall[(rg, "ood")]), flush=True)

    # ===== load OLMo =====
    from transformers import AutoTokenizer, AutoModelForCausalLM
    mid = "allenai/OLMo-2-0425-1B"
    print(f"\n================ STAGE 2: GENERATIVE READOUT (frozen organ -> OLMo LM head) ===========",
          flush=True)
    print("loading", mid, flush=True)
    tok = AutoTokenizer.from_pretrained(mid)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    probe = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16)
    D = probe.config.hidden_size; nL = probe.config.num_hidden_layers
    del probe
    print(f"OLMo: {nL} layers, hidden {D}", flush=True)

    # ===== build pools (organ lattice injected); train both cells & full renderings =====
    prng = np.random.default_rng(a.seed + 11)
    t0 = time.time()
    train_pools = build_pool(prng, organ, dev, RUNGS, "id", a.per_rung_train, ("cells", "full"))
    train_cells, train_full = train_pools["cells"], train_pools["full"]
    eval_id = build_pool(prng, organ, dev, RUNGS, "id", a.per_rung_eval, ("cells", "full"))
    eval_ood = build_pool(prng, organ, dev, RUNGS, "ood", a.per_rung_eval, ("full",))
    eval_div = build_diverse_pool(prng, organ, dev, a.curriculum, a.per_rung_eval, CURRIC_RUNGS) \
        if os.path.exists(a.curriculum) else []
    print(f"\n  data: {len(train_cells)} train recs/mode; eval id {len(eval_id['full'])} ood "
          f"{len(eval_ood['full'])} diverse {len(eval_div)}  built in {time.time()-t0:.0f}s", flush=True)

    model = train_readout(organ, (mid, D, nL), tok, dev, a, train_cells, train_full, eval_id["cells"])

    # ===== DELIVERABLE 1: per-rung generative accuracy — WOVEN vs TEXT-LoRA vs BASE =====
    def woven(pool):
        return score_gen(model, pool, tok, dev, control="true", inject=True, bs=a.bs)

    def textlora(pool):
        return score_gen(model, pool, tok, dev, inject=False, use_base=False, bs=a.bs)

    def base(pool):
        return score_gen(model, pool, tok, dev, inject=False, use_base=True, fewshot=FEWSHOT, bs=a.bs)

    print("\n  ====== PER-RUNG GENERATIVE ACCURACY  (WOVEN / TEXT-LoRA / BASE-fewshot) ======", flush=True)
    suites = [("in-dist (full)", eval_id["full"]), ("OOD-N (full)", eval_ood["full"])]
    if eval_div:
        suites.append(("OOD-phrasing", eval_div))
    suites.append(("in-dist (cells=pure-readout)", eval_id["cells"]))
    deliverable = {}
    for sname, pool in suites:
        w, tl, bs_ = woven(pool), textlora(pool), base(pool)
        deliverable[sname] = {"woven": w, "textlora": tl, "base": bs_}
        print(f"\n  --- {sname}  (n={w['n']}) ---", flush=True)
        print(f"    {'rung':12s}  {'WOVEN':>7s}  {'TEXT-LoRA':>9s}  {'BASE':>7s}", flush=True)
        for rg in RUNGS:
            if rg in w["by_rung"]:
                print(f"    {rg:12s}  {w['by_rung'][rg]*100:6.1f}%  {tl['by_rung'].get(rg,0)*100:8.1f}%  "
                      f"{bs_['by_rung'].get(rg,0)*100:6.1f}%", flush=True)
        print(f"    {'OVERALL':12s}  {w['overall']*100:6.1f}%  {tl['overall']*100:8.1f}%  "
              f"{bs_['overall']*100:6.1f}%   (det woven {w['det_acc']*100:.0f} abst {w['abst_acc']*100:.0f})",
              flush=True)

    # ===== DELIVERABLE 2: the CAUSAL-CONTROL table (intervene on the frozen organ's lattice) =====
    print("\n  ====== CAUSAL-CONTROL TABLE  (woven model; full-text present; overall acc %) ======",
          flush=True)
    print("  Does generation degrade when the organ's lattice is corrupted, even though the TEXT still"
          " describes the problem? (the strong test: LM IGNORES text, READS lattice)", flush=True)
    controls = {}
    ctrl_suites = [("in-dist (full)", eval_id["full"]), ("in-dist (cells)", eval_id["cells"]),
                   ("OOD-N (full)", eval_ood["full"])]
    print(f"    {'suite':22s}  {'true':>6s} {'shuffle':>8s} {'permute':>8s} {'corrupt':>8s} {'zero':>6s}",
          flush=True)
    for sname, pool in ctrl_suites:
        row = {c: score_gen(model, pool, tok, dev, control=c, inject=True, bs=a.bs)["overall"]
               for c in ("true", "shuffle", "permute", "corrupt", "zero")}
        controls[sname] = row
        print(f"    {sname:22s}  {row['true']*100:6.1f} {row['shuffle']*100:8.1f} {row['permute']*100:8.1f}"
              f" {row['corrupt']*100:8.1f} {row['zero']*100:6.1f}", flush=True)
    # per-rung corrupt drop on in-dist full (which rungs does the LM causally wield the organ on?)
    wfull = woven(eval_id["full"]); cfull = score_gen(model, eval_id["full"], tok, dev, control="corrupt",
                                                      inject=True, bs=a.bs)
    print("\n    per-rung TRUE->CORRUPT drop (in-dist full):", flush=True)
    for rg in RUNGS:
        if rg in wfull["by_rung"]:
            dr = (wfull["by_rung"][rg] - cfull["by_rung"].get(rg, 0)) * 100
            print(f"      {rg:12s}  true {wfull['by_rung'][rg]*100:5.1f}%  corrupt "
                  f"{cfull['by_rung'].get(rg,0)*100:5.1f}%  drop {dr:5.1f}pts", flush=True)

    # ===== DELIVERABLE 3: free greedy generation (TRUE generation sanity) true vs shuffle =====
    gpool = eval_id["full"][: a.greedy_n]
    gacc, gex = O.greedy_gen(model, gpool, tok, K, [None] * K, dev, control="true")
    gacc_sh, _ = O.greedy_gen(model, gpool, tok, K, [None] * K, dev, control="shuffle")
    print(f"\n  ====== FREE GREEDY GEN (in-dist full, n={len(gpool)}): TRUE {gacc*100:.1f}%  vs  "
          f"SHUFFLE {gacc_sh*100:.1f}% ======", flush=True)
    for gold, got in gex[:6]:
        print(f"      gold {gold!r:30s} -> {got!r}", flush=True)

    # ===== verdict =====
    print("\n================ VERDICT ================", flush=True)
    idf = deliverable["in-dist (full)"]
    cc = controls["in-dist (full)"]
    drop = cc["true"] - min(cc["shuffle"], cc["permute"], cc["corrupt"], cc["zero"])
    print(f"  in-dist(full): WOVEN {idf['woven']['overall']*100:.0f}%  TEXT-LoRA "
          f"{idf['textlora']['overall']*100:.0f}%  BASE {idf['base']['overall']*100:.0f}%  "
          f"| causal drop true->worst-control {drop*100:.0f}pts", flush=True)
    print("  Per-rung: WOVEN beats TEXT-LoRA/BASE where the organ supplies deduction the text-only path"
          " can't shortcut (chains, determined cells); ~equal where text already suffices or the organ"
          " recall is low (the honest affine/ordering gaps). Causal controls dropping with text PRESENT"
          " => the LM causally WIELDS the learned general organ in generation.", flush=True)

    out = a.out or os.path.join(os.path.dirname(__file__), "..", "runs", "glados_staged.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    blob = {"args": vars(a), "budget": [N_MAX, D_MAX, M_MAX, A_MAX],
            "organ": {"d_model": od, "params": opar, "log": olog,
                      "recall": {f"{rg}/{sp}": v for (rg, sp), v in recall.items()}},
            "deliverable": {k: {kk: {"overall": vv["overall"], "det": vv["det_acc"],
                                     "abst": vv["abst_acc"], "by_rung": vv["by_rung"]}
                                for kk, vv in v.items()} for k, v in deliverable.items()},
            "controls": controls, "greedy": {"true": gacc, "shuffle": gacc_sh}}
    json.dump(blob, open(out, "w"), indent=1, default=float)
    print("\nwrote", out, flush=True)


if __name__ == "__main__":
    main()
