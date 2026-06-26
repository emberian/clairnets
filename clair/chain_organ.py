"""clair/chain_organ.py — RUNG-3 dual: a differentiable, recurrent NEURAL FORWARD-CHAINER.

This is the *dual* of the lattice narrower (clair.induce.ColorDeductor / clair.proposer):

    lattice organ   : state alive[cell,value] in [0,1], MONOTONE SHRINK (multiplicative meet) —
                      RULES OUT candidates; soundness = never drop a true survivor (dominate dedₚ).
    chaining organ  : state c[atom] in [0,1] DERIVED-confidence, MONOTONE GROWTH (soft modus ponens) —
                      DERIVES new facts; soundness = never derive an atom outside the true closure.

GROUND TRUTH = clair.fol.forward_chain(facts, rules): the least Herbrand model. We GROUND every rule
over the entity set (the unification the symbolic chainer does for free) to get an AND-OR hypergraph:

    NODE  = a ground atom (pred + args + neg).          state c[node] in [0,1]  (starts 1 on facts).
    FIRING= one ground rule instance  body[] -> head.   a differentiable modus-ponens hyperedge.

Each round, every firing computes a soft conjunction of its body confidences (a learned t-norm: a
soft-min `margin` sharpened by a learnable α,θ plus a neural gate over the body/head embeddings), and
the head's confidence GROWS by a soft-OR over the firings that derive it:

    fire_f      = σ( α·(softmin_b c[b] − θ) + gate_f(emb) )
    agg_h       = 1 − Π_{f:head=h} (1 − fire_f)                  (soft-OR over derivations)
    c_{t}[h]    = c_{t−1}[h] + (1 − c_{t−1}[h])·agg_h            (FILL-UP: monotone non-decreasing)

c only ever increases (the exact dual of the lattice's only-ever-decreasing `alive`). Facts pinned 1.
The neural gate makes the organ *capable* of unsoundness (it can fire on a partial body); SOUNDNESS is
not hard-coded — it is LEARNED from a soundness-ASYMMETRIC loss (heavy penalty for deriving an atom
NOT in the closure; light penalty for missing an entailed one), exactly the dual of dominate-dedₚ.

Readout the query: entail = c[q] high; contradict = c[¬q] high; unknown = neither — supervised against
clair.fol.label_query. Validation: false-derivation rate (soundness), completeness, and PROOF-DEPTH
GENERALIZATION (train shallow depth 1-2, test deep depth 3-4 with extra fixpoint rounds).

Run:  python -m clair.chain_organ --smoke         # quick: chainer trains, false-derivation -> 0
      python -m clair.chain_organ                 # full: soundness + completeness + depth-gen table
"""
from __future__ import annotations

import argparse
import itertools as it
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import fol


# --------------------------------------------------------------------------- predicate vocab (fixed)
# A small closed vocab over everything clair.fol can emit; UNK=0 keeps it robust to new predicates.
PREDS = ["<unk>", "link", "reach", "trouble", "calm", *fol.UNARY, *fol.BINARY]
PRED_ID = {p: i for i, p in enumerate(PREDS)}


def pred_id(p: str) -> int:
    return PRED_ID.get(p, 0)


def rule_vars(r: fol.Rule):
    seen = []
    for lit in (*r.body, r.head):
        for t in lit.args:
            if fol.is_var(t) and t not in seen:
                seen.append(t)
    return seen


# --------------------------------------------------------------------------- grounding (build the AND-OR graph)
def ground(ents, rules, facts, closure, query):
    """Full Datalog grounding -> AND-OR hypergraph. Returns a per-problem dict of node features,
    firing structure, fact/closure masks, per-node depth, and the query/¬query node indices.

    Firings are generated over the FULL entity product (not just satisfied bindings), so the graph
    contains heads OUTSIDE the closure — the spurious modus-ponens steps the organ must learn NOT to
    fire. That is what makes the soundness test real."""
    node_id = {}

    def nid(lit: fol.Lit) -> int:
        i = node_id.get(lit)
        if i is None:
            i = len(node_id)
            node_id[lit] = i
        return i

    # facts + query + ¬query always present (query may be an UNKNOWN atom no firing produces)
    for f in facts:
        nid(f)
    qn, nqn = nid(query), nid(fol.neg_of(query))

    firings = []  # (head_idx, [body_idx,...])
    for r in rules:
        vs = rule_vars(r)
        for assign in it.product(ents, repeat=len(vs)):
            b = dict(zip(vs, assign))
            body_g = [fol._subst(l, b) for l in r.body]
            head_g = fol._subst(r.head, b)
            firings.append((nid(head_g), [nid(x) for x in body_g]))
    for a in closure:                                   # safety: every truth is a node
        nid(a)

    N = len(node_id)
    atoms = [None] * N
    for lit, i in node_id.items():
        atoms[i] = lit
    fset = set(facts)
    depths = fol.proof_depths(facts, rules, closure)    # depth 0 facts; missing -> non-derivable

    pred = np.zeros(N, np.int64)
    neg = np.zeros(N, np.int64)
    arity = np.zeros(N, np.int64)
    fact_mask = np.zeros(N, np.float32)
    clos_mask = np.zeros(N, np.float32)
    depth = np.full(N, -1, np.int64)
    for i, a in enumerate(atoms):
        pred[i] = pred_id(a.pred)
        neg[i] = int(a.neg)
        arity[i] = len(a.args)
        fact_mask[i] = float(a in fset)
        clos_mask[i] = float(a in closure)
        if a in depths:
            depth[i] = depths[a]

    return {
        "N": N, "pred": pred, "neg": neg, "arity": arity,
        "fact_mask": fact_mask, "clos_mask": clos_mask, "depth": depth,
        "firings": firings, "qn": qn, "nqn": nqn,
        "label": 0, "q_depth": 0,                        # set by problem_to_graph from the problem
    }


def problem_to_graph(p: fol.FOLProblem):
    closure = fol.forward_chain(p.facts, p.rules)
    g = ground(p.entities, p.rules, p.facts, closure, p.query)
    g["label"] = {"entail": 0, "contradict": 1, "unknown": 2}[p.label]
    g["q_depth"] = int(p.depth)
    return g


# --------------------------------------------------------------------------- collate (pad a batch)
def collate(graphs, device):
    B = len(graphs)
    Nmax = max(g["N"] for g in graphs)
    Smax = max((max((len(b) for _, b in g["firings"]), default=1) for g in graphs), default=1)
    Fmax = max((len(g["firings"]) for g in graphs), default=1)

    pred = np.zeros((B, Nmax), np.int64)
    neg = np.zeros((B, Nmax), np.int64)
    arity = np.zeros((B, Nmax), np.int64)
    node_valid = np.zeros((B, Nmax), np.float32)
    fact_mask = np.zeros((B, Nmax), np.float32)
    clos_mask = np.zeros((B, Nmax), np.float32)
    depth = np.full((B, Nmax), -1, np.int64)
    f_head = np.zeros((B, Fmax), np.int64)
    f_body = np.zeros((B, Fmax, Smax), np.int64)
    f_bvalid = np.zeros((B, Fmax, Smax), np.float32)
    f_valid = np.zeros((B, Fmax), np.float32)
    qn = np.zeros(B, np.int64); nqn = np.zeros(B, np.int64)
    label = np.zeros(B, np.int64); qdepth = np.zeros(B, np.int64)

    for bi, g in enumerate(graphs):
        N = g["N"]
        pred[bi, :N] = g["pred"]; neg[bi, :N] = g["neg"]; arity[bi, :N] = g["arity"]
        node_valid[bi, :N] = 1.0
        fact_mask[bi, :N] = g["fact_mask"]; clos_mask[bi, :N] = g["clos_mask"]
        depth[bi, :N] = g["depth"]
        for fi, (h, body) in enumerate(g["firings"]):
            f_head[bi, fi] = h; f_valid[bi, fi] = 1.0
            for si, bnode in enumerate(body):
                f_body[bi, fi, si] = bnode; f_bvalid[bi, fi, si] = 1.0
        qn[bi] = g["qn"]; nqn[bi] = g["nqn"]
        label[bi] = g["label"]; qdepth[bi] = g["q_depth"]

    t = lambda a: torch.as_tensor(a, device=device)
    return {
        "pred": t(pred), "neg": t(neg), "arity": t(arity), "node_valid": t(node_valid),
        "fact_mask": t(fact_mask), "clos_mask": t(clos_mask), "depth": t(depth),
        "f_head": t(f_head), "f_body": t(f_body), "f_bvalid": t(f_bvalid), "f_valid": t(f_valid),
        "qn": t(qn), "nqn": t(nqn), "label": t(label), "qdepth": t(qdepth),
    }


# --------------------------------------------------------------------------- the chaining organ
class ChainOrgan(nn.Module):
    """Recurrent differentiable forward-chainer. Monotone GROWTH of per-atom derived-confidence via a
    learned soft modus-ponens over a grounded AND-OR firing hypergraph (the dual of ColorDeductor's
    monotone-shrink soft arc-consistency). Shared weights every round => running more rounds chains
    DEEPER (the recurrent inductive bias the proof-depth-generalization test probes)."""
    def __init__(self, d=48):
        super().__init__()
        self.d = d
        self.pred_emb = nn.Embedding(len(PREDS), d)
        self.neg_emb = nn.Embedding(2, d)
        self.ar_emb = nn.Embedding(3, d)
        # neural modus-ponens gate: per body slot (emb + current conf) -> pooled -> with head emb
        self.slot = nn.Sequential(nn.Linear(d + 1, d), nn.SiLU(), nn.Linear(d, d))
        self.gate = nn.Sequential(nn.Linear(2 * d + 1, d), nn.SiLU(), nn.Linear(d, 1))
        self.gate[-1].bias.data.fill_(-2.0)             # start strict: low fire on a partial body
        self.log_alpha = nn.Parameter(torch.tensor(1.5))   # softmin sharpness of the structural AND
        self.theta = nn.Parameter(torch.tensor(0.6))       # firing threshold on the body margin
        self.log_tau = nn.Parameter(torch.tensor(-1.0))    # softmin temperature
        # 3-way query readout (entail / contradict / unknown) from c[q], c[¬q]
        self.readout = nn.Sequential(nn.Linear(4, d), nn.SiLU(), nn.Linear(d, 3))

    def atom_emb(self, ba):
        return self.pred_emb(ba["pred"]) + self.neg_emb(ba["neg"]) + self.ar_emb(ba["arity"])  # [B,N,d]

    def forward(self, ba, T):
        emb = self.atom_emb(ba)                                          # [B,N,d]
        B, N, d = emb.shape
        f_head, f_body, f_bvalid, f_valid = ba["f_head"], ba["f_body"], ba["f_bvalid"], ba["f_valid"]
        F_, S = f_head.shape[1], f_body.shape[2]
        alpha = F.softplus(self.log_alpha)
        tau = F.softplus(self.log_tau) + 1e-3

        # static per-firing pieces: body-slot embeddings + head embedding (conf is the only dynamic input)
        body_idx = f_body.reshape(B, F_ * S)                            # [B,F*S]
        be = torch.gather(emb, 1, body_idx.unsqueeze(-1).expand(-1, -1, d)).reshape(B, F_, S, d)
        he = torch.gather(emb, 1, f_head.unsqueeze(-1).expand(-1, -1, d))   # [B,F,d]
        c = ba["fact_mask"].clone()                                     # [B,N] start: facts=1, else 0
        history = []
        for _ in range(T):
            cb = torch.gather(c, 1, body_idx).reshape(B, F_, S)         # body confidences [B,F,S]
            # structural soft-AND of the body: softmin over VALID slots. Invalid slots are pushed to a
            # high FINITE confidence so exp(-cb/tau)->0 excludes them (no -inf -> no NaN on pad firings).
            cb_m = torch.where(f_bvalid > 0.5, cb, torch.full_like(cb, 10.0))
            margin = -tau * torch.logsumexp(-cb_m / tau, dim=2)         # [B,F] ~ min_b c[b]
            # neural gate over body/head structure (lets it be unsound -> soundness must be LEARNED)
            sm = self.slot(torch.cat([be, cb.unsqueeze(-1)], -1)) * f_bvalid.unsqueeze(-1)
            body_vec = sm.sum(2) / f_bvalid.sum(2, keepdim=True).clamp_min(1.0)   # [B,F,d]
            g = self.gate(torch.cat([body_vec, he, margin.unsqueeze(-1)], -1)).squeeze(-1)  # [B,F]
            fire = torch.sigmoid(alpha * (margin - self.theta) + g) * f_valid     # [B,F] in [0,1]
            # soft-OR aggregate into heads:  agg_h = 1 - prod(1-fire)
            log1m = torch.log1p(-fire.clamp(max=1 - 1e-6))             # [B,F]
            acc = torch.zeros(B, N, device=c.device).scatter_add(1, f_head, log1m)
            agg = 1.0 - torch.exp(acc)                                  # [B,N] in [0,1]
            c = c + (1.0 - c) * agg                                     # MONOTONE GROWTH (fill-up)
            c = c * (1.0 - ba["fact_mask"]) + ba["fact_mask"]          # re-pin facts to 1
            c = c * ba["node_valid"]
            history.append(c)
        return c, history

    def query_logits(self, c, ba):
        cq = c.gather(1, ba["qn"].unsqueeze(1)).squeeze(1)
        cn = c.gather(1, ba["nqn"].unsqueeze(1)).squeeze(1)
        feats = torch.stack([cq, cn, cq * cn, 1.0 - torch.maximum(cq, cn)], -1)
        return self.readout(feats), cq, cn


# --------------------------------------------------------------------------- loss (soundness-asymmetric)
def chain_loss(model, ba, T, w_unsound=5.0, w_incomplete=1.0, w_query=1.0):
    c, hist = model(ba, T)
    nv = ba["node_valid"]
    clos = ba["clos_mask"]
    fact = ba["fact_mask"]
    # apply the node loss on the last two rounds (deep supervision toward the fixpoint)
    l_uns = l_inc = 0.0
    for ct in hist[-2:]:
        non_clos = nv * (1.0 - clos)                                    # atoms NOT in the closure
        deriv = nv * clos * (1.0 - fact)                                # entailed, non-fact (must derive)
        l_uns = l_uns + (ct * non_clos).sum() / non_clos.sum().clamp_min(1.0)            # -> 0
        l_inc = l_inc + ((1.0 - ct) * deriv).sum() / deriv.sum().clamp_min(1.0)          # -> 0
    l_uns = l_uns / min(2, len(hist)); l_inc = l_inc / min(2, len(hist))
    ql, cq, cn = model.query_logits(c, ba)
    l_q = F.cross_entropy(ql, ba["label"])
    loss = w_unsound * l_uns + w_incomplete * l_inc + w_query * l_q
    parts = {"unsound": float(l_uns), "incomplete": float(l_inc), "query": float(l_q)}
    return loss, parts


# --------------------------------------------------------------------------- metrics
@torch.no_grad()
def evaluate(model, graphs, device, T, bs=64, thr=0.5):
    model.eval()
    fd_num = fd_den = 0           # false-derivation: non-closure nodes derived
    comp_num = comp_den = 0       # completeness: entailed non-fact nodes derived
    by_depth = {}                 # depth -> [derived, total] for entailed atoms
    q_hard = q_tot = 0            # hard query-label accuracy (threshold readout)
    q_mlp = 0                     # learned readout accuracy
    qd_hard = {}                  # per query-depth hard accuracy
    for i in range(0, len(graphs), bs):
        ba = collate(graphs[i:i + bs], device)
        c, _ = model(ba, T)
        ql, cq, cn = model.query_logits(c, ba)
        nv = ba["node_valid"] > 0.5
        clos = ba["clos_mask"] > 0.5
        fact = ba["fact_mask"] > 0.5
        der = c > thr
        non_clos = nv & ~clos
        fd_num += int((der & non_clos).sum()); fd_den += int(non_clos.sum())
        deriv = nv & clos & ~fact
        comp_num += int((der & deriv).sum()); comp_den += int(deriv.sum())
        dep = ba["depth"]
        for dv in torch.unique(dep[deriv]).tolist():
            m = deriv & (dep == dv)
            a, b = by_depth.setdefault(int(dv), [0, 0])
            by_depth[int(dv)] = [a + int((der & m).sum()), b + int(m.sum())]
        # hard query label from derived-confidence thresholds (the honest "did derivation reach q")
        pred_hard = torch.where(
            (cq > thr) & (cq >= cn), torch.zeros_like(ba["label"]),
            torch.where(cn > thr, torch.ones_like(ba["label"]), 2 * torch.ones_like(ba["label"])))
        q_hard += int((pred_hard == ba["label"]).sum()); q_tot += int(ba["label"].numel())
        q_mlp += int((ql.argmax(-1) == ba["label"]).sum())
        for qd in torch.unique(ba["qdepth"]).tolist():
            m = ba["qdepth"] == qd
            a, b = qd_hard.setdefault(int(qd), [0, 0])
            qd_hard[int(qd)] = [a + int((pred_hard[m] == ba["label"][m]).sum()), b + int(m.sum())]
    return {
        "false_deriv_rate": fd_num / max(1, fd_den),
        "completeness": comp_num / max(1, comp_den),
        "query_acc_hard": q_hard / max(1, q_tot),
        "query_acc_mlp": q_mlp / max(1, q_tot),
        "completeness_by_depth": {k: by_depth[k][0] / max(1, by_depth[k][1]) for k in sorted(by_depth)},
        "depth_counts": {k: by_depth[k][1] for k in sorted(by_depth)},
        "query_acc_by_depth": {k: qd_hard[k][0] / max(1, qd_hard[k][1]) for k in sorted(qd_hard)},
    }


# --------------------------------------------------------------------------- data
def gen_set(rng, n, depths):
    """A balanced set of (entail/contradict/unknown) problems whose entailment depth is in `depths`."""
    out = []
    labels = ["entail", "contradict", "unknown"]
    for k in range(n):
        lab = labels[k % 3]
        d = int(rng.choice(depths))
        # depth only constrains the ENTAIL chain; contradict/unknown use it to size the KB
        p = fol.gen_problem(rng, label=lab, depth=d)
        out.append(problem_to_graph(p))
    return out


# --------------------------------------------------------------------------- monotonicity sanity
@torch.no_grad()
def check_monotone(model, ba, T):
    _, hist = model(ba, T)
    worst = 0.0
    for a, b in zip(hist, hist[1:]):
        worst = max(worst, float((a - b).clamp_min(0).max()))   # c must never decrease
    return worst


# --------------------------------------------------------------------------- train
def train(model, device, steps, train_depths, T_train, bs, rng, log_every=50, eval_set=None,
          T_eval=None, w_unsound=5.0, lr=3e-3, quiet=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log = []
    pool = gen_set(rng, max(bs * 8, 256), train_depths)             # a refreshed pool of shallow KBs
    for s in range(1, steps + 1):
        model.train()
        if s % 200 == 0:
            pool = gen_set(rng, max(bs * 8, 256), train_depths)
        idx = rng.choice(len(pool), size=bs, replace=False)
        ba = collate([pool[i] for i in idx], device)
        loss, parts = chain_loss(model, ba, T_train, w_unsound=w_unsound)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if not quiet and (s % log_every == 0 or s == 1):
            ev = evaluate(model, pool[:256], device, T_train)
            mono = check_monotone(model, ba, T_train)
            row = {"step": s, "loss": float(loss), **parts,
                   "fd_rate": ev["false_deriv_rate"], "completeness": ev["completeness"],
                   "q_hard": ev["query_acc_hard"], "mono_viol": mono}
            log.append(row)
            print(f"  step {s:5d} loss {loss:.3f} (uns {parts['unsound']:.4f} inc {parts['incomplete']:.3f} "
                  f"q {parts['query']:.3f})  false-deriv {ev['false_deriv_rate']*100:5.2f}%  "
                  f"complete {ev['completeness']*100:5.1f}%  q-acc {ev['query_acc_hard']*100:5.1f}%  "
                  f"mono-viol {mono:.1e}", flush=True)
    return log


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--T_train", type=int, default=5)
    ap.add_argument("--T_eval", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="runs/chain_organ")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.bs = 300, 32

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    print(f"device={dev}  preds={len(PREDS)}  train depths=[1,2] T_train={args.T_train}  "
          f"test depths=[3,4] T_eval={args.T_eval}")

    model = ChainOrgan().to(dev)
    print(f"chain-organ params: {sum(p.numel() for p in model.parameters())}")

    t0 = time.time()
    log = train(model, dev, args.steps, train_depths=[1, 2], T_train=args.T_train, bs=args.bs, rng=rng)
    print(f"train wall: {time.time()-t0:.1f}s")

    # ---- evaluation sets ----
    print("\nbuilding eval sets ...", flush=True)
    n_eval = 128 if args.smoke else 512
    shallow = gen_set(rng, n_eval, depths=[1, 2])          # in-distribution
    deep = gen_set(rng, n_eval, depths=[3, 4])             # OOD: deeper proofs than trained

    # in-distribution uses T_train rounds; OOD allows extra fixpoint rounds (same shared weights)
    ev_shallow = evaluate(model, shallow, dev, args.T_train)
    ev_deep_same = evaluate(model, deep, dev, args.T_train)   # deep KB but only T_train rounds
    ev_deep_more = evaluate(model, deep, dev, args.T_eval)    # deep KB + extra chaining rounds

    print("\n================ SOUNDNESS / COMPLETENESS / PROOF-DEPTH GENERALIZATION ================")
    def show(name, ev):
        print(f"\n[{name}]")
        print(f"  false-derivation rate (soundness, want ~0) : {ev['false_deriv_rate']*100:6.3f}%")
        print(f"  completeness (entailed atoms derived)      : {ev['completeness']*100:6.2f}%")
        print(f"  completeness by proof depth                : "
              + "  ".join(f"d{k}={v*100:.0f}%(n{ev['depth_counts'][k]})"
                          for k, v in ev["completeness_by_depth"].items()))
        print(f"  query label acc  hard={ev['query_acc_hard']*100:.1f}%  mlp={ev['query_acc_mlp']*100:.1f}%")
        print(f"  query acc by query-depth                   : "
              + "  ".join(f"d{k}={v*100:.0f}%" for k, v in ev["query_acc_by_depth"].items()))
    show(f"IN-DIST  depth 1-2, T={args.T_train}", ev_shallow)
    show(f"OOD DEEP depth 3-4, T={args.T_train} (no extra rounds)", ev_deep_same)
    show(f"OOD DEEP depth 3-4, T={args.T_eval} (extra fixpoint rounds)", ev_deep_more)

    out = Path(args.out + ("_smoke" if args.smoke else "") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "args": vars(args), "train_log": log,
        "in_dist": ev_shallow, "ood_deep_same_T": ev_deep_same, "ood_deep_more_T": ev_deep_more,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
