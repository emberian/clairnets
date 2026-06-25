"""Train + measure the factor-graph proposer against the EXACT finite-CSP harness.

The dominate-dedₚ objective (soundness free, completeness the variable):

  * sample small CSPs (chain / 2sat / 3sat / xor / coloring); enumerate exact solutions + exact_dedP
    for supervision (cached — enumeration is the cost);
  * ON-POLICY: maintain lattice states produced by the model's OWN monotone meet
    (alive_next = alive AND (sigmoid(b) >= theta)); never add candidates; deep supervision each step;
  * LOSS = asymmetric BCE toward exact_dedP with a HEAVY penalty on eliminating a dedₚ-kept value
    (soundness) + conflict BCE (target = current dom unsat) + singleton CE (dominate-dedₚ);
  * METRICS: false_elim_vs_exact (MUST ~0), COMPLETENESS (solvable -> solved vs abstain),
    wrong_return (returns a non-solution — MUST ~0), per CSP kind and per polymorphism signature.

  python -m clair.run_proposer --arm full --steps 800 --smoke
  python -m clair.run_proposer --arms ffn,inner,wedge,full --steps 3000   # the 4-arm comparison
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .proposer import FactorGraphProposer, size_for
from . import csp as C

# global padding budget (covers all sampled kinds)
N_MAX, D_MAX, M_MAX, A_MAX = 8, 3, 14, 3


def device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# --------------------------------------------------------------------------- exact-harness cache
class Exact:
    """Memoize the (expensive) exact enumeration keyed by (cons, dom)."""
    def __init__(self):
        self.sol, self.ded = {}, {}

    def _key(self, csp, dom):
        return (csp.cons, csp.d, dom)

    def solutions(self, csp, dom):
        k = self._key(csp, dom)
        if k not in self.sol:
            self.sol[k] = C.solutions(csp, dom)
        return self.sol[k]

    def dedP(self, csp, dom):
        k = self._key(csp, dom)
        if k not in self.ded:
            sols = self.solutions(csp, dom)
            if not sols:
                self.ded[k] = tuple(frozenset() for _ in range(csp.n))
            else:
                self.ded[k] = tuple(frozenset(s[i] for s in sols) for i in range(csp.n))
        return self.ded[k]


# --------------------------------------------------------------------------- CSP samplers
def _solvable(csp):
    return len(C.solutions(csp)) > 0


def sample_csp(rng, kind):
    """Return a small solvable CSP of the requested kind, within (N_MAX, D_MAX, M_MAX, A_MAX)."""
    for _ in range(200):
        if kind == "chain":
            c = C.chain_eq(int(rng.integers(3, 7)))
        elif kind == "xor":
            c = C.xor_parity()
        elif kind == "2sat":
            n = int(rng.integers(3, 6))
            ncl = int(rng.integers(2, n + 2))
            cl = [(int(rng.integers(n)), int(rng.integers(2)),
                   int(rng.integers(n)), int(rng.integers(2))) for _ in range(ncl)]
            c = C.two_sat(n, cl)
        elif kind == "3sat":
            n = int(rng.integers(3, 6))
            ncl = int(rng.integers(2, n + 1))
            cl = [(int(rng.integers(n)), int(rng.integers(2)), int(rng.integers(n)), int(rng.integers(2)),
                   int(rng.integers(n)), int(rng.integers(2))) for _ in range(ncl)]
            c = C.three_sat(n, cl)
        elif kind == "coloring":
            n = int(rng.integers(3, 6))
            edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.45]
            c = C.coloring(n, edges, k=3)
        else:
            raise ValueError(kind)
        if c.n <= N_MAX and c.d <= D_MAX and len(c.cons) <= M_MAX and _solvable(c):
            return c
    return C.chain_eq(3)  # fallback (always solvable)


KINDS = ["chain", "2sat", "3sat", "xor", "coloring"]


def sample_corpus(rng, n, kinds=KINDS):
    return [(k, sample_csp(rng, k)) for k in (kinds[i % len(kinds)] for i in range(n))]


# --------------------------------------------------------------------------- featurization
def relation_table(scope, al, d_max, a_max):
    """Broadcast relation table T over d_max**a_max: T[v0..]=1 iff first |scope| coords form an
    allowed tuple (remaining axes are don't-care = broadcast). Values >= d are never allowed."""
    a = len(scope)
    t = np.zeros((d_max,) * a_max, dtype=np.float32)
    for tup in al:
        sl = [slice(None)] * a_max
        for p in range(a):
            sl[p] = tup[p]
        t[tuple(sl)] = 1.0
    return t.reshape(-1)


def featurize(items, dev):
    """items: list of (csp, dom). Returns the static factor-graph tensors + the dynamic var_mask.
    (Static parts don't depend on dom; var_mask/given do.)"""
    B = len(items)
    rel_dim = D_MAX ** A_MAX
    var_mask = np.zeros((B, N_MAX, D_MAX), np.float32)
    given = np.zeros((B, N_MAX), np.float32)
    var_valid = np.zeros((B, N_MAX), np.float32)
    fac_rel = np.zeros((B, M_MAX, rel_dim), np.float32)
    fac_arity = np.zeros((B, M_MAX, A_MAX), np.float32)
    fac_valid = np.zeros((B, M_MAX), np.float32)
    edge_var = np.full((B, M_MAX, A_MAX), N_MAX, np.int64)   # pad slots point at the dummy var
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


def dom_from_mask(row, csp):
    return tuple(frozenset(v for v in range(csp.d) if row[i, v] > 0.5) for i in range(csp.n))


# --------------------------------------------------------------------------- targets + loss
def build_targets(items, ex: Exact, dev):
    """Per-item exact_dedP target [B,n,dv] and conflict target [B] (current dom unsat)."""
    B = len(items)
    tgt = np.zeros((B, N_MAX, D_MAX), np.float32)
    conflict = np.zeros((B,), np.float32)
    for bi, (csp, dom) in enumerate(items):
        ded = ex.dedP(csp, dom)
        if all(len(c) == 0 for c in ded):
            conflict[bi] = 1.0
        for i in range(csp.n):
            for v in ded[i]:
                tgt[bi, i, v] = 1.0
    return torch.as_tensor(tgt, device=dev), torch.as_tensor(conflict, device=dev)


def loss_fn(sup, var_mask, var_valid, tgt, conflict, wpos=6.0, wneg=0.5, lcls=0.3, lce=0.3):
    """Dominate-dedₚ: asymmetric BCE (HEAVY wpos on eliminating a dedₚ-kept value) over currently-alive
    candidates, + conflict BCE + singleton CE where dedₚ is a singleton."""
    eps = 1e-6
    alive = var_mask * var_valid.unsqueeze(-1)            # only score real, currently-alive candidates
    sing = (tgt.sum(-1) == 1).float() * var_valid        # cells dedₚ pins to a singleton
    tlab = tgt.argmax(-1)
    tot = 0.0
    for b, cls in sup:
        p = torch.sigmoid(b)
        bce = -(wpos * tgt * torch.log(p + eps) + wneg * (1 - tgt) * torch.log(1 - p + eps))
        bce = (bce * alive).sum() / (alive.sum() + eps)
        clsl = F.binary_cross_entropy_with_logits(cls, conflict)
        ce_all = F.cross_entropy(b.reshape(-1, b.size(-1)), tlab.reshape(-1), reduction="none").reshape(b.shape[:-1])
        ce = (ce_all * sing).sum() / (sing.sum() + eps)
        tot = tot + bce + lcls * clsl + lce * ce
    return tot / len(sup)


@torch.no_grad()
def meet(var_mask, b, theta):
    """Monotone narrowing: keep an alive candidate iff sigmoid(b) >= theta. Never adds."""
    return var_mask * (torch.sigmoid(b) >= theta).float()


@torch.no_grad()
def false_elim(var_mask_before, var_mask_after, var_valid, tgt):
    """Fraction of (alive, dedₚ-kept) candidates the meet wrongly eliminated. Soundness => ~0."""
    elig = tgt * var_mask_before * var_valid.unsqueeze(-1)        # dedₚ keeps it AND it was alive
    killed = elig * (var_mask_after < 0.5).float()
    return killed.sum().item(), elig.sum().item()


# --------------------------------------------------------------------------- training (on-policy)
def train(arm, dev, target, steps, pool=128, R=8, lr=3e-4, theta=0.5, seed=0,
          rng=None, log_every=None, log=None):
    torch.manual_seed(seed)
    rng = rng or np.random.default_rng(seed)
    ex = Exact()
    d, npar = size_for(arm, N_MAX, D_MAX, M_MAX, A_MAX, target, R=R, ds=min(4, R))
    m = FactorGraphProposer(arm, N_MAX, D_MAX, M_MAX, A_MAX, d=d, R=R, ds=min(4, R)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95))
    log_every = log_every or max(1, steps // 25)
    log = log if log is not None else []
    items = sample_corpus(rng, pool)
    items = [(c, c.full()) for _, c in items]              # start every chain from the full grid
    print(f"dev={dev} arm={arm} d={d} params={npar:,} (target {target:,.0f}) pool={pool} R={R}")
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
            new_vm_cpu = new_vm.cpu().numpy()
            nxt = []
            for bi, (csp, dom) in enumerate(items):
                ndom = dom_from_mask(new_vm_cpu[bi], csp)
                st = C.status(ndom)
                stalled = (ndom == dom)
                if st in ("solved", "conflict") or stalled:      # terminal/stuck -> fresh CSP
                    nc = sample_csp(rng, KINDS[(s + bi) % len(KINDS)])
                    nxt.append((nc, nc.full()))
                else:
                    nxt.append((csp, ndom))
            items = nxt

        if s % log_every == 0:
            fer = fe_k / max(1, fe_n)
            msg = {"step": s, "loss": float(loss.detach()), "false_elim": fer,
                   "mean_alive": float(vm.sum(-1).mean())}
            log.append(msg)
            print(f"  step {s:5d}  loss {msg['loss']:.3f}  false_elim {fer:.4f}  "
                  f"alive {msg['mean_alive']:.2f}  {time.time()-t0:.0f}s")
            fe_k = fe_n = 0
    return m, d, npar, log


# --------------------------------------------------------------------------- evaluation (completeness)
@torch.no_grad()
def run_to_fixpoint(m, csps, dev, theta=0.5, R_max=64):
    """Drive each CSP from the full grid with the model's monotone meet until fixpoint/terminal.
    Tensorized: only var_mask changes across rounds. Returns final per-item dom + trajectory false_elim."""
    items = [(c, c.full()) for c in csps]
    feat = featurize(items, dev)                  # static factor graph (fixed across rounds)
    vm = feat["var_mask"].clone()
    B = len(csps)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    fe_k = fe_n = 0
    for r in range(R_max):
        b, cls, _ = m(vm, feat["given"], feat["fac_rel"], feat["fac_arity"],
                      feat["edge_var"], feat["edge_valid"], feat["var_valid"], feat["fac_valid"])
        new_vm = meet(vm, b, theta)
        # soundness check against exact dedP of the CURRENT state
        cur = [(csps[i], dom_from_mask(vm[i].cpu().numpy(), csps[i])) for i in range(B)]
        tgt, _ = build_targets(cur, Exact(), dev)
        k, n = false_elim(vm, new_vm, feat["var_valid"], tgt); fe_k += k; fe_n += n
        changed = (new_vm != vm).any(-1).any(-1)
        vm = torch.where(done.view(-1, 1, 1), vm, new_vm)
        done = done | ~changed
        if done.all():
            break
    doms = [dom_from_mask(vm[i].cpu().numpy(), csps[i]) for i in range(B)]
    return doms, (fe_k, fe_n)


@torch.no_grad()
def evaluate(m, corpus, dev, theta=0.5, R_max=64):
    """Completeness vs the ac_step baseline (+ per-cell exact dedP ceiling), per kind and per
    polymorphism signature, plus wrong_return and trajectory false_elim."""
    m.eval()
    csps = [c for _, c in corpus]
    kinds = [k for k, _ in corpus]
    doms, (fe_k, fe_n) = run_to_fixpoint(m, csps, dev, theta, R_max)

    per = {}
    poly = {}
    n_solv = solved = wrong = ac_solved = ded_solved = 0
    for (kind, csp), dom in zip(corpus, doms):
        sols = C.solutions(csp)
        if not sols:
            continue
        n_solv += 1
        st = C.status(dom)
        is_solved = st == "solved"
        is_correct = is_solved and (tuple(next(iter(dom[i])) for i in range(csp.n)) in sols)
        solved += is_solved
        wrong += int(is_solved and not is_correct)
        ac, _ = C.to_fixpoint(C.ac_step, csp, csp.full())
        ac_solved += C.status(ac) == "solved"
        dd, _ = C.to_fixpoint(C.exact_dedP, csp, csp.full())
        ded_solved += C.status(dd) == "solved"
        d = per.setdefault(kind, {"n": 0, "solved": 0, "ac": 0, "ded": 0, "wrong": 0})
        d["n"] += 1; d["solved"] += is_solved; d["ac"] += C.status(ac) == "solved"
        d["ded"] += C.status(dd) == "solved"; d["wrong"] += int(is_solved and not is_correct)
        if csp.d == 2:
            sig = C.polymorphism_signature(csp)
            key = "affine" if (sig.get("affine") and not sig.get("majority")) else \
                  ("semilattice" if sig.get("semilattice") else "other")
            p = poly.setdefault(key, {"n": 0, "solved": 0, "ac": 0})
            p["n"] += 1; p["solved"] += is_solved; p["ac"] += C.status(ac) == "solved"
    m.train()
    return {
        "n_solvable": n_solv,
        "completeness": solved / max(1, n_solv),
        "ac_completeness": ac_solved / max(1, n_solv),
        "ded_completeness": ded_solved / max(1, n_solv),
        "wrong_return_rate": wrong / max(1, n_solv),
        "eval_false_elim": fe_k / max(1, fe_n),
        "eval_false_elim_count": fe_k,
        "per_kind": per,
        "per_poly": poly,
    }


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ffn", "inner", "wedge", "full"], default=None)
    ap.add_argument("--arms", default=None, help="comma list for the iso-param comparison")
    ap.add_argument("--target", type=float, default=2.5e5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--pool", type=int, default=128)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--neval", type=int, default=300)
    ap.add_argument("--rmax", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny defaults for a quick sanity run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.steps = min(a.steps, 400); a.target = min(a.target, 1.5e5); a.neval = min(a.neval, 150)
    dev = device()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    arms = (a.arms.split(",") if a.arms else [a.arm or "full"])

    eval_rng = np.random.default_rng(12345)
    eval_corpus = sample_corpus(eval_rng, a.neval)        # SAME eval set for every arm
    results = {}
    for arm in arms:
        print(f"\n===== ARM {arm} =====")
        m, d, npar, log = train(arm, dev, a.target, a.steps, pool=a.pool, R=a.R,
                                lr=a.lr, theta=a.theta, seed=a.seed)
        ev = evaluate(m, eval_corpus, dev, theta=a.theta, R_max=a.rmax)
        results[arm] = {"d_model": d, "params": npar, "log": log, "eval": ev}
        print(f"\n--- ARM {arm} eval (n_solvable={ev['n_solvable']}) ---")
        print(f"  COMPLETENESS {ev['completeness']*100:5.1f}%   "
              f"ac_step {ev['ac_completeness']*100:5.1f}%   dedP-ceiling {ev['ded_completeness']*100:5.1f}%")
        print(f"  wrong_return {ev['wrong_return_rate']*100:.2f}%   "
              f"eval_false_elim {ev['eval_false_elim']:.4f} ({ev['eval_false_elim_count']} elims)")
        for k, v in ev["per_kind"].items():
            print(f"    {k:9s} n={v['n']:3d}  model {v['solved']/max(1,v['n'])*100:5.1f}%  "
                  f"ac {v['ac']/max(1,v['n'])*100:5.1f}%  dedP {v['ded']/max(1,v['n'])*100:5.1f}%  "
                  f"wrong {v['wrong']}")
        for k, v in ev["per_poly"].items():
            print(f"    poly[{k:11s}] n={v['n']:3d}  model {v['solved']/max(1,v['n'])*100:5.1f}%  "
                  f"ac {v['ac']/max(1,v['n'])*100:5.1f}%")

    if len(arms) > 1:
        print("\n===== 4-ARM COMPLETENESS (iso-param) =====")
        for arm in arms:
            e = results[arm]["eval"]
            print(f"  {arm:6s}  params {results[arm]['params']:,}  "
                  f"completeness {e['completeness']*100:5.1f}%  "
                  f"false_elim {e['eval_false_elim']:.4f}  wrong {e['wrong_return_rate']*100:.2f}%")
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "runs",
                                f"proposer_{'_'.join(arms)}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(results, open(out, "w"), indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
