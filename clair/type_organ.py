"""clair/type_organ.py — the TYPE-INFERENCE organ: a learned, checked proposer that narrows the per-node
candidate TYPE sets of a typed expression, soundly, guiding the unification/propagation where the
certified local fixpoint (arc consistency) ABSTAINS.

THE WEAVE (the code half of the deductor). The exact harness `clair.typeinfer` compiles a typed
expression to a finite CSP over a bounded type universe and gives two exact authorities:
  * exact_dedP — per-node type set over ALL in-universe well-typings (the strongest SOUND narrowing);
  * Algorithm W — the independent HM principal type (cross-checked == dedP in typeinfer's self-check).
The certified primitive (arc consistency = sound type-constraint propagation) is SOUND by construction
but INCOMPLETE on ~12% of expressions: where a type variable's value is fixed only by a MULTI-NODE
correlation that per-cell propagation forgets (e.g. λx.x — the param-body correlation; per-cell AC keeps
all 4 function types, dedP keeps the 2 diagonal a->a). That residual is the organ's job.

THE ORGAN. We REUSE the proven factor-graph proposer (`clair.proposer.FactorGraphProposer`, ablatable
geometric arms, iso-param) over the compiled type-CSP: VARIABLE node = AST node (candidate type mask
over the universe), FACTOR node = a typing-rule relation table, R rounds of message passing + the
ablatable channel mixer, per-(node,type) keep-logits used by a MONOTONE meet. SOUNDNESS is the research
variable: trained with the dominate-dedₚ loss (HEAVY penalty for dropping a type dedₚ keeps) so a sound
organ never false-eliminates while closing the abstain gap.

MEASURE: recall/soundness vs the exact HM oracle (false_elim MUST ~0), completeness (root inferred ==
principal type), exact per-node match vs dedₚ, and gap-closed (of the cells certified keeps-but-dedₚ-kills,
how many the organ correctly kills), across expression sizes. Honest scope: the TYPE-CONSTRAINT
sub-problem of coding (a depth-bounded type universe, monomorphic let) — NOT program synthesis/execution.

  python -m clair.type_organ --arm full --steps 400 --smoke      # smoke: trains, false-elim->0, gap closes
  python -m clair.type_organ --arms ffn,full --steps 2500        # full table + ablation, saves checkpoint
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .proposer import FactorGraphProposer, size_for
from .run_proposer import loss_fn, meet, false_elim, dom_from_mask, relation_table
from . import csp as C
from . import typeinfer as TI

# padding budget (covers depth<=1 universe, budgets 2-4): max_cells=18, max_cons=22, arity=3, T=10
N_MAX, D_MAX, M_MAX, A_MAX = 20, 10, 24, 3
UNIV = TI.Universe(max_depth=1, ctors=("fun", "pair"))
assert UNIV.T == D_MAX, (UNIV.T, D_MAX)


def device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# --------------------------------------------------------------------------- exact-harness cache
class Exact:
    def __init__(self): self.ded = {}

    def dedP(self, csp, dom):
        k = (csp.cons, csp.d, dom)
        if k not in self.ded:
            self.ded[k] = C.exact_dedP(csp, dom)
        return self.ded[k]


# --------------------------------------------------------------------------- samplers
def sample_problem(rng, budget=None, gap_only=False, tries=60):
    """A random well-typed in-universe expression (its compilation). If gap_only, resample until the
    certified AC fixpoint ABSTAINS below dedₚ (the multi-node-correlation cases the organ must earn)."""
    for _ in range(tries):
        b = budget if budget is not None else rng.randint(2, 4)
        comp = TI.rand_problem(rng, UNIV, budget=b, max_nodes=N_MAX - 2)
        if comp.csp.n > N_MAX or len(comp.csp.cons) > M_MAX:
            continue
        if not C.solutions(comp.csp, limit=1):
            continue
        if gap_only:
            dom, _ = C.to_fixpoint(TI.certified_step, comp.csp, comp.csp.full())
            ex = C.exact_dedP(comp.csp, comp.csp.full())
            if dom == ex:
                continue
        return comp
    return TI.compile_expr(("add", ("lit", 1, "int"), ("lit", 2, "int")), UNIV)


def sample_corpus(rng, n, gap_frac=0.0):
    out = []
    for i in range(n):
        out.append(sample_problem(rng, gap_only=(rng.random() < gap_frac)))
    return out


# --------------------------------------------------------------------------- featurization
def featurize(items, dev):
    """items: list of (comp, dom). Returns the static factor-graph tensors + the dynamic var_mask."""
    B = len(items)
    rel_dim = D_MAX ** A_MAX
    var_mask = np.zeros((B, N_MAX, D_MAX), np.float32)
    given = np.zeros((B, N_MAX), np.float32)
    var_valid = np.zeros((B, N_MAX), np.float32)
    fac_rel = np.zeros((B, M_MAX, rel_dim), np.float32)
    fac_arity = np.zeros((B, M_MAX, A_MAX), np.float32)
    fac_valid = np.zeros((B, M_MAX), np.float32)
    edge_var = np.full((B, M_MAX, A_MAX), N_MAX, np.int64)
    edge_valid = np.zeros((B, M_MAX, A_MAX), np.float32)
    for bi, (comp, dom) in enumerate(items):
        csp = comp.csp
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
            fac_rel[bi, fi] = relation_table(sc, al, D_MAX, A_MAX)
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


def build_targets(items, ex: Exact, dev):
    B = len(items)
    tgt = np.zeros((B, N_MAX, D_MAX), np.float32)
    conflict = np.zeros((B,), np.float32)
    for bi, (comp, dom) in enumerate(items):
        ded = ex.dedP(comp.csp, dom)
        if all(len(c) == 0 for c in ded):
            conflict[bi] = 1.0
        for i in range(comp.csp.n):
            for v in ded[i]:
                tgt[bi, i, v] = 1.0
    return torch.as_tensor(tgt, device=dev), torch.as_tensor(conflict, device=dev)


# --------------------------------------------------------------------------- training (on-policy)
def train(arm, dev, target, steps, pool=128, R=8, lr=3e-4, theta=0.5, seed=0, gap_frac=0.5,
          log_every=None):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    ex = Exact()
    d, npar = size_for(arm, N_MAX, D_MAX, M_MAX, A_MAX, target, R=R, ds=min(4, R))
    m = FactorGraphProposer(arm, N_MAX, D_MAX, M_MAX, A_MAX, d=d, R=R, ds=min(4, R)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95))
    log_every = log_every or max(1, steps // 25)
    log = []
    items = [(c, c.csp.full()) for c in sample_corpus(rng, pool, gap_frac=gap_frac)]
    print(f"dev={dev} arm={arm} d={d} params={npar:,} (target {target:,.0f}) pool={pool} R={R} T={D_MAX}")
    fe_k = fe_n = 0
    t0 = time.time()
    for s in range(1, steps + 1):
        feat = featurize(items, dev)
        vm = feat["var_mask"]
        tgt, conflict = build_targets(items, ex, dev)
        b, cls, sup = fwd(m, feat, vm)
        loss = loss_fn(sup, vm, feat["var_valid"], tgt, conflict)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()

        with torch.no_grad():
            new_vm = meet(vm, b, theta)
            k, n = false_elim(vm, new_vm, feat["var_valid"], tgt); fe_k += k; fe_n += n
            new_cpu = new_vm.cpu().numpy()
            nxt = []
            for bi, (comp, dom) in enumerate(items):
                ndom = dom_from_mask(new_cpu[bi], comp.csp)
                st = C.status(ndom)
                if st in ("solved", "conflict") or ndom == dom:        # terminal/stuck -> fresh expr
                    nc = sample_problem(rng, gap_only=(rng.random() < gap_frac))
                    nxt.append((nc, nc.csp.full()))
                else:
                    nxt.append((comp, ndom))
            items = nxt

        if s % log_every == 0 or s == 1:
            fer = fe_k / max(1, fe_n)
            log.append({"step": s, "loss": float(loss.detach()), "false_elim": fer})
            print(f"  step {s:5d}  loss {float(loss):.3f}  false_elim {fer:.4f}  "
                  f"alive {float(vm.sum(-1).mean()):.2f}  {time.time()-t0:.0f}s", flush=True)
            fe_k = fe_n = 0
    return m, d, npar, log


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def run_to_fixpoint(m, comps, dev, theta=0.5, R_max=48):
    items = [(c, c.csp.full()) for c in comps]
    feat = featurize(items, dev)
    vm = feat["var_mask"].clone()
    B = len(comps)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    fe_k = fe_n = 0
    exq = Exact()
    for _ in range(R_max):
        b, cls, _ = fwd(m, feat, vm)
        new_vm = meet(vm, b, theta)
        cur = [(comps[i], dom_from_mask(vm[i].cpu().numpy(), comps[i].csp)) for i in range(B)]
        tgt, _ = build_targets(cur, exq, dev)
        k, n = false_elim(vm, new_vm, feat["var_valid"], tgt); fe_k += k; fe_n += n
        changed = (new_vm != vm).any(-1).any(-1)
        vm = torch.where(done.view(-1, 1, 1), vm, new_vm)
        done = done | ~changed
        if done.all():
            break
    doms = [dom_from_mask(vm[i].cpu().numpy(), comps[i].csp) for i in range(B)]
    return doms, (fe_k, fe_n)


@torch.no_grad()
def evaluate(m, comps, dev, theta=0.5, R_max=48):
    """Soundness (false_elim vs dedₚ) + completeness (root inferred == principal type) + per-node exact
    match + gap-closed vs certified AC, overall and bucketed by expression size."""
    m.eval()
    doms, (fe_k, fe_n) = run_to_fixpoint(m, comps, dev, theta, R_max)
    R = {"n": 0, "fe": 0, "root_recall_num": 0, "root_recall_den": 0,
         "root_solved_correct": 0, "exact_match": 0,
         "ac_root_solved": 0, "ded_root_solved": 0, "gap_cells": 0, "gap_closed": 0,
         "wrong_root": 0}
    by_size = {}
    for comp, dom in zip(comps, doms):
        csp = comp.csp
        ex = C.exact_dedP(csp, csp.full())
        if all(len(c) == 0 for c in ex):                # ill-typed (we don't sample these, skip)
            continue
        root = comp.root
        pt, ok = TI.algorithm_w(comp.ast)
        principal = TI.ground_principal(pt, UNIV) if ok else node_set(ex, root)
        R["n"] += 1
        sz = TI.size(comp.ast)
        bk = "1-4" if sz <= 4 else ("5-7" if sz <= 7 else "8+")
        bs = by_size.setdefault(bk, {"n": 0, "exact": 0, "root_solved": 0, "fe": 0, "gap_cells": 0, "gap_closed": 0})
        bs["n"] += 1
        # soundness: did the organ drop any type dedₚ keeps?
        fe = TI.false_elim(dom, ex); R["fe"] += fe; bs["fe"] += fe
        # root recall vs principal type (== dedₚ root): never drop a real type
        org_root = node_set(dom, root)
        R["root_recall_num"] += len(org_root & principal); R["root_recall_den"] += len(principal)
        # exact per-node match vs dedₚ (full completeness)
        em = int(tuple(dom) == tuple(ex)); R["exact_match"] += em; bs["exact"] += em
        # root solved & correct == principal singleton inferred exactly
        rs = int(len(dom[root]) == 1)
        rs_correct = int(len(dom[root]) == 1 and org_root == principal and len(principal) == 1)
        R["root_solved_correct"] += rs_correct; bs["root_solved"] += rs_correct
        R["wrong_root"] += int(len(dom[root]) == 1 and not (org_root <= principal))   # soundness viol at root
        # baselines
        ac, _ = C.to_fixpoint(TI.certified_step, csp, csp.full())
        R["ac_root_solved"] += int(len(ac[root]) == 1)
        R["ded_root_solved"] += int(len(ex[root]) == 1)
        # gap-closed: cells certified keeps but dedₚ kills -> organ should kill them
        for i in range(csp.n):
            extra = ac[i] - ex[i]
            R["gap_cells"] += len(extra); bs["gap_cells"] += len(extra)
            killed = len(extra - dom[i])
            R["gap_closed"] += killed; bs["gap_closed"] += killed
    m.train()
    R["false_elim_rate"] = fe_k / max(1, fe_n)
    R["by_size"] = by_size
    return R


def node_set(dom, i):
    return frozenset(UNIV.types[v] for v in dom[i])


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ffn", "inner", "wedge", "full"], default=None)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--target", type=float, default=2.5e5)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--pool", type=int, default=128)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--neval", type=int, default=600)
    ap.add_argument("--gap_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.steps = min(a.steps, 400); a.target = min(a.target, 1.5e5); a.neval = min(a.neval, 200)
    dev = device()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    arms = (a.arms.split(",") if a.arms else [a.arm or "full"])

    eval_rng = random.Random(12345)
    eval_all = sample_corpus(eval_rng, a.neval, gap_frac=0.0)               # natural distribution
    eval_gap = [sample_problem(eval_rng, gap_only=True) for _ in range(a.neval // 2)]   # the gap family
    results = {}
    for arm in arms:
        print(f"\n===== ARM {arm} =====")
        m, d, npar, log = train(arm, dev, a.target, a.steps, pool=a.pool, R=a.R,
                                lr=a.lr, theta=a.theta, seed=a.seed, gap_frac=a.gap_frac)
        ev_all = evaluate(m, eval_all, dev, theta=a.theta)
        ev_gap = evaluate(m, eval_gap, dev, theta=a.theta)
        results[arm] = {"d_model": d, "params": npar, "log": log, "all": ev_all, "gap": ev_gap}

        def show(name, ev):
            n = max(1, ev["n"])
            print(f"\n--- ARM {arm}  [{name}]  n={ev['n']} ---")
            print(f"  SOUNDNESS  organ false-elim vs dedₚ (HM oracle) : {ev['fe']:4d}  "
                  f"root-recall {ev['root_recall_num']}/{ev['root_recall_den']} "
                  f"({ev['root_recall_num']/max(1,ev['root_recall_den'])*100:.2f}%)  wrong-root {ev['wrong_root']}")
            print(f"  COMPLETE   organ==dedₚ (all nodes) {ev['exact_match']/n*100:5.1f}%   "
                  f"root inferred==principal {ev['root_solved_correct']/n*100:5.1f}%")
            print(f"  ROOT SOLVED  certified AC {ev['ac_root_solved']/n*100:5.1f}%   "
                  f"organ {ev['root_solved_correct']/n*100:5.1f}%   dedₚ-ceiling {ev['ded_root_solved']/n*100:5.1f}%")
            print(f"  GAP-CLOSED organ kills {ev['gap_closed']}/{ev['gap_cells']} certified-abstain cells "
                  f"({ev['gap_closed']/max(1,ev['gap_cells'])*100:.1f}%)")
            for bk in ("1-4", "5-7", "8+"):
                if bk in ev["by_size"]:
                    s = ev["by_size"][bk]; sn = max(1, s["n"])
                    print(f"     size {bk:4s} n={s['n']:3d}  exact {s['exact']/sn*100:5.1f}%  "
                          f"root {s['root_solved']/sn*100:5.1f}%  fe {s['fe']}  "
                          f"gap {s['gap_closed']}/{s['gap_cells']}")
        show("ALL (natural)", ev_all)
        show("GAP-ONLY (certified abstains)", ev_gap)

    if len(arms) > 1:
        print("\n===== ARM COMPARISON (iso-param) =====")
        for arm in arms:
            e = results[arm]["all"]; g = results[arm]["gap"]
            print(f"  {arm:6s} params {results[arm]['params']:,}  "
                  f"all: exact {e['exact_match']/max(1,e['n'])*100:4.1f}% root {e['root_solved_correct']/max(1,e['n'])*100:4.1f}% fe {e['fe']}  | "
                  f"gap: gap-closed {g['gap_closed']/max(1,g['gap_cells'])*100:4.1f}% fe {g['fe']}")

    out = a.out or os.path.join(os.path.dirname(__file__), "..", "runs",
                                f"type_organ_{'_'.join(arms)}{'_smoke' if a.smoke else ''}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"args": vars(a), "results": results}, open(out, "w"), indent=1)
    # checkpoint the last arm's model
    ckpt = out.replace(".json", ".pt")
    torch.save({"state_dict": m.state_dict(), "arm": arms[-1], "d_model": results[arms[-1]]["d_model"],
                "pad": [N_MAX, D_MAX, M_MAX, A_MAX], "R": a.R}, ckpt)
    print(f"\nwrote {out} and {ckpt}")


if __name__ == "__main__":
    main()
