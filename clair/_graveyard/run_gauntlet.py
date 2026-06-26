"""Domain-general gauntlet: ONE shared GLaDOS organ over MANY lattices (sudoku-shaped powerset
domains from clair/domains.py), trained on a set of domains, optionally HOLDING OUT one whole
domain to measure ZERO-SHOT SOUNDNESS. Mirrors run_glados.py's on-policy loop and metrics
(false-elimination rate = THE soundness metric; solve_rate; wrong_return_rate; p90_forwards),
just parameterised by Domain and padded across domains.

  # train on coloring+3sat, hold out maze, test zero-shot:
  python -m clair.run_gauntlet --arm glados --target 800000 --steps 4000 --holdout maze
  # train on all three jointly:
  python -m clair.run_gauntlet --arm glados --domains coloring,3sat,maze

----------------------------------------------------------------------------------------------
PADDING / MASKING SCHEME (multi-domain, one shared model)
----------------------------------------------------------------------------------------------
The model has fixed (P, V) = (Pmax, Vmax) = the max positions / max candidates across the
domains it must serve. A batch is always drawn from a SINGLE domain (domains are not mixed within
one [B,P,V] tensor — they are mixed across steps), with that domain's (P_d, V_d) <= (Pmax, Vmax).
We embed a domain's instance into the padded lattice as follows:

  PAD POSITIONS  p in [P_d, Pmax):  marked GIVEN with a singleton alive at candidate 0.
     - given=1   -> protected by the MEET, never branched, excluded from false-elim (sudoku.py
                    only counts non-given cells), and counted as already-solved by status().
     - so pad positions are inert: they cannot create spurious conflicts or solves.

  PAD CANDIDATES c in [V_d, Vmax) at REAL positions p in [0, P_d):  alive=0 from init.
     - the alpha-target's onehot(sol) only ever points at candidates in [0, V_d) (sol in 1..V_d),
       so the target never asks the model to keep a pad candidate; the asym-BCE negative term
       pushes pad-candidate survival toward 0 anyway, and step_state's "restore best ALIVE
       candidate" can never resurrect a candidate that started dead. Pad candidates stay dead.

  sol for pad positions is set to candidate 1 (the singleton we planted) so is_correct/alpha_target
  treat them as trivially satisfied.

This keeps EVERY generic op (init/meet/branch/status/alpha_target/false_elim) correct without
special-casing: padding is expressed entirely through the (given, alive) masks the ops already
respect. The shared cell sees a uniform [B,Pmax,Vmax] interface and is genuinely domain-blind.

torch is required to RUN this (training); domains.py's generators are pure-numpy.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .glados import GLaDOS, size_for
from . import sudoku as S                 # reuse domain-agnostic ops: step_state, branch, status, conflict_flag
from . import domains as D


# --------------------------------------------------------------------------- padded init
def pad_init_state(puz, sol, P_d, V_d, Pmax, Vmax, dev):
    """Embed a single-domain batch (puz,sol [B,P_d], clue/sol in 1..V_d) into the padded
    [B,Pmax,Vmax] lattice per the scheme in the module docstring. Returns (alive, given, sol_pad)."""
    puz = np.asarray(puz)
    sol = np.asarray(sol)
    B = puz.shape[0]
    # pad puzzle: real cols copied, pad positions get clue=1 (given singleton)
    puz_pad = np.ones((B, Pmax), np.int64)            # default clue=1 => given singleton everywhere
    puz_pad[:, :P_d] = puz                            # real positions keep their (0=blank / 1..V_d clue)
    sol_pad = np.ones((B, Pmax), np.int64)            # pad solution = candidate 1
    sol_pad[:, :P_d] = sol

    puz_t = torch.as_tensor(puz_pad, dtype=torch.long, device=dev)
    alive = torch.ones(B, Pmax, Vmax, device=dev)
    # kill pad candidates [V_d, Vmax) at REAL positions; pad POSITIONS handled by given+clue below
    if V_d < Vmax:
        alive[:, :P_d, V_d:] = 0.0
    given = (puz_t > 0).float()                       # blanks(0) at real positions -> not given; all else given
    clue = puz_t.clamp(min=1) - 1
    onehot = torch.zeros(B, Pmax, Vmax, device=dev).scatter_(2, clue.unsqueeze(-1), 1.0)
    alive = torch.where(given.unsqueeze(-1) > 0, onehot, alive)
    return alive, given, sol_pad


# --------------------------------------------------------------------------- loss (generic alpha-target)
def loss_fn(sup, alive, sol, dev, wpos=4.0, wneg=0.5, lcls=0.1, lce=0.2):
    """Deep-supervised LDT loss vs the on-policy alpha target — identical to run_glados.loss_fn but
    using the generic (P,V) alpha-target from domains.py."""
    tgt, conflict, sing = D.generic_alpha_target(alive, sol, dev)
    tot = 0.0
    for b, cls in sup:
        p = torch.sigmoid(b)
        bce = -(wpos * tgt * torch.log(p + 1e-6) + wneg * (1 - tgt) * torch.log(1 - p + 1e-6)).mean()
        clsl = F.binary_cross_entropy_with_logits(cls, conflict)
        tlab = tgt.argmax(-1)
        ce_all = F.cross_entropy(b.reshape(-1, b.size(-1)), tlab.reshape(-1), reduction="none").reshape(b.shape[:-1])
        ce = (ce_all * sing).sum() / (sing.sum() + 1e-6)
        tot = tot + bce + lcls * clsl + lce * ce
    return tot / len(sup), conflict


@torch.no_grad()
def false_elim_rate(alive_before, alive_after, given, sol, dev):
    """Soundness metric (==run_glados): fraction of (non-given, true-still-alive) positions where the
    MEET just killed the true candidate. Pad positions are given -> excluded automatically."""
    sol = torch.as_tensor(sol, dtype=torch.long, device=dev)
    soli = (sol.clamp(min=1) - 1).unsqueeze(-1)
    true_before = alive_before.gather(2, soli).squeeze(-1)
    true_after = alive_after.gather(2, soli).squeeze(-1)
    elig = (given < 0.5) * (true_before > 0.5)
    killed = elig * (true_after < 0.5)
    return killed.sum().item(), elig.sum().item()


# --------------------------------------------------------------------------- data pool per domain
class DomainPool:
    """Holds a generated puzzle bank + an on-policy alive/given pool for one domain, embedded into
    the shared padded (Pmax,Vmax) lattice. One DomainPool per training domain."""
    def __init__(self, dom: D.Domain, n_bank, Pmax, Vmax, pool, dev, rng):
        self.dom, self.Pmax, self.Vmax, self.dev = dom, Pmax, Vmax, dev
        self.puz, self.sol = dom.gen(n_bank, rng)            # [n_bank, P_d] int64
        self.N = len(self.puz)
        self.idx = rng.integers(0, self.N, pool)
        self.alive, self.given, self.solp = pad_init_state(
            self.puz[self.idx], self.sol[self.idx], dom.P, dom.V, Pmax, Vmax, dev)

    def reinit(self, mask, rng):
        """Recycle the masked chains with fresh instances (post solve/conflict)."""
        k = int(mask.sum().item())
        if k == 0:
            return
        ridx = rng.integers(0, self.N, k)
        ra, rg, rs = pad_init_state(self.puz[ridx], self.sol[ridx], self.dom.P, self.dom.V,
                                    self.Pmax, self.Vmax, self.dev)
        self.alive[mask] = ra
        self.given[mask] = rg
        m = mask.cpu().numpy()
        self.idx[m] = ridx
        self.solp[m] = rs


# --------------------------------------------------------------------------- train
def train(arm, train_doms, Pmax, Vmax, dev, target, steps, n_bank=2000, pool=512, lr=3e-4,
          inner=16, log_every=None, seed=0):
    """One shared GLaDOS sized to (Pmax,Vmax) (the FULL gauntlet incl. holdout, so a held-out
    domain is shape-compatible at eval), trained on `train_doms` round-robin, on-policy."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d_model, npar = size_for(arm, Pmax, Vmax, target, inner=inner)
    m = GLaDOS(arm, Pmax, Vmax, d=d_model, inner=inner, ds=inner).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    log_every = log_every or max(1, steps // 20)
    pools = [DomainPool(dom, n_bank, Pmax, Vmax, pool, dev, rng) for dom in train_doms]
    fe_k = {dom.name: 0 for dom in train_doms}; fe_n = {dom.name: 0 for dom in train_doms}
    print(f"dev={dev} arm={arm} d={d_model} params={npar:,} (target {target:,.0f}) "
          f"Pmax={Pmax} Vmax={Vmax} train={[d.name for d in train_doms]} pool={pool}")
    log = []; t0 = time.time()
    for s in range(1, steps + 1):
        dp = pools[(s - 1) % len(pools)]                    # round-robin one domain per step
        b, cls, sup = m(dp.alive, dp.given)
        loss, _ = loss_fn(sup, dp.alive, dp.solp, dev)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        with torch.no_grad():
            conf = S.conflict_flag(cls)
            new = S.step_state(dp.alive, dp.given, b)
            k, n = false_elim_rate(dp.alive, new, dp.given, dp.solp, dev)
            fe_k[dp.dom.name] += k; fe_n[dp.dom.name] += n
            stalled = (new == dp.alive).all(dim=-1).all(dim=-1)
            solved, _ = S.status(new)
            need_branch = stalled & ~solved & ~conf
            if need_branch.any():
                bnew, _ = S.branch(new, b)
                new = torch.where(need_branch.view(-1, 1, 1), bnew, new)
            dp.alive = new
            dp.reinit(solved | conf, rng)
        if s % log_every == 0:
            fers = {nm: fe_k[nm] / max(1, fe_n[nm]) for nm in fe_k}
            log.append({"step": s, "loss": float(loss.detach()), "false_elim": fers,
                        "frac_solved": float(solved.float().mean()), "dom": dp.dom.name})
            fr = " ".join(f"{nm}={v:.4f}" for nm, v in fers.items())
            print(f"  step {s:5d}  loss {loss.item():.3f}  false_elim[{fr}]  "
                  f"solved {solved.float().mean()*100:4.1f}% ({dp.dom.name})  {time.time()-t0:.0f}s")
            fe_k = {nm: 0 for nm in fe_k}; fe_n = {nm: 0 for nm in fe_n}
    return m, {"arm": arm, "d_model": d_model, "params": npar,
               "train_domains": [d.name for d in train_doms], "log": log}


# --------------------------------------------------------------------------- evaluate (per domain)
@torch.no_grad()
def evaluate(m, dom: D.Domain, Pmax, Vmax, dev, n_eval=256, K=16, R_max=64, seed=1):
    """K parallel chains per instance for ONE domain. Reports solve_rate, wrong_return_rate
    (soundness), false_elim, p90 forward-passes. Same chain logic as run_glados.evaluate."""
    m.eval()
    rng = np.random.default_rng(seed)
    puz, sol = dom.gen(n_eval, rng)
    E = len(puz)
    pe = np.repeat(puz, K, axis=0); se = np.repeat(sol, K, axis=0)
    alive, given, solp = pad_init_state(pe, se, dom.P, dom.V, Pmax, Vmax, dev)
    solp_t = torch.as_tensor(solp, dtype=torch.long, device=dev)

    chain_done = torch.zeros(E * K, dtype=torch.bool, device=dev)
    chain_correct = torch.zeros(E * K, dtype=torch.bool, device=dev)
    chain_returned = torch.zeros(E * K, dtype=torch.bool, device=dev)
    chain_fwd = torch.zeros(E * K, dtype=torch.long, device=dev)
    fe_k = fe_n = 0
    for r in range(1, R_max + 1):
        live = ~chain_done
        if not live.any():
            break
        b, cls, _ = m(alive, given)
        chain_fwd[live] = r
        conf = S.conflict_flag(cls)
        new = S.step_state(alive, given, b)
        k, n = false_elim_rate(alive, new, given, solp, dev)
        fe_k += k; fe_n += n
        solved, _ = S.status(new)
        ok = D.generic_is_correct(new, solp, dev)
        fin_solved = solved & live
        fin_abstain = conf & live & ~solved
        chain_returned |= fin_solved
        chain_correct |= (fin_solved & ok)
        chain_done |= fin_solved | fin_abstain
        stalled = (new == alive).all(-1).all(-1) & live & ~solved & ~conf
        if stalled.any():
            bnew, _ = S.branch(new, b)
            new = torch.where(stalled.view(-1, 1, 1), bnew, new)
        alive = new
    cr = chain_returned.view(E, K); cc = chain_correct.view(E, K); cf = chain_fwd.view(E, K)
    returned = cr.any(1); correct = cc.any(1); wrong = returned & ~correct
    big = torch.where(cc, cf, torch.full_like(cf, 10**6))
    fwd_solved = big.min(1).values[correct]
    p90 = float(torch.quantile(fwd_solved.float(), 0.9)) if correct.any() else float("nan")
    m.train()
    return {"domain": dom.name, "solve_rate": float(correct.float().mean()),
            "wrong_return_rate": float(wrong.float().mean()), "n_returned": int(returned.sum()),
            "false_elim": fe_k / max(1, fe_n), "p90_forwards": p90, "n_eval": E}


# --------------------------------------------------------------------------- main
_REG = {"coloring": D.graph_coloring, "3sat": D.threesat, "maze": D.maze}


def _build(names):
    return [_REG[n]() for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ldt", "glados", "nowedge"], required=True)
    ap.add_argument("--target", type=float, default=8e5)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--pool", type=int, default=512)
    ap.add_argument("--inner", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--domains", default="coloring,3sat,maze",
                    help="comma list of domains in the gauntlet")
    ap.add_argument("--holdout", default=None,
                    help="domain to EXCLUDE from training and eval zero-shot (must be in --domains)")
    ap.add_argument("--nbank", type=int, default=2000)
    ap.add_argument("--neval", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    names = [x for x in a.domains.split(",") if x]
    for nm in names:
        assert nm in _REG, f"unknown domain {nm}; known={list(_REG)}"
    all_doms = _build(names)
    # pad to the max across the WHOLE gauntlet (incl. holdout) so the held-out domain fits the model
    Pmax = max(d.P for d in all_doms)
    Vmax = max(d.V for d in all_doms)

    train_names = [n for n in names if n != a.holdout]
    train_doms = _build(train_names)
    # train uses model sized to the full-gauntlet (Pmax,Vmax) so eval on holdout is shape-compatible
    m, res = train(a.arm, train_doms, Pmax, Vmax, dev, a.target, a.steps,
                   n_bank=a.nbank, pool=a.pool, lr=a.lr, inner=a.inner, seed=a.seed)
    res["holdout"] = a.holdout

    print("\n=== per-domain eval ===")
    evals = {}
    for dom in all_doms:
        ev = evaluate(m, dom, Pmax, Vmax, dev, n_eval=a.neval, seed=a.seed + 1)
        zs = " [ZERO-SHOT]" if dom.name == a.holdout else ""
        evals[dom.name] = ev
        print(f"  {dom.name:9s}{zs:11s} solve {ev['solve_rate']*100:5.1f}%  "
              f"wrong-return {ev['wrong_return_rate']*100:5.2f}%  false_elim {ev['false_elim']:.4f}  "
              f"p90-fwd {ev['p90_forwards']:.1f}")
    res["eval"] = evals
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "runs",
                                f"gauntlet_{a.arm}_{'-'.join(train_names)}_ho-{a.holdout}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
