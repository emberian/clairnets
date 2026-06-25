"""The decisive core-swap: train ldt / glados / nowedge at ONE param budget on Sudoku-Extreme,
on-policy, and report the metric that actually decides it — FALSE-ELIMINATION RATE — plus solve
rate, wrong-return rate, and p90 forward-passes.

  python -m clair.run_glados --arm glados --target 800000 --steps 4000 --source extreme

The three arms differ ONLY in the channel mixer (attn+FFN vs geom vs geom-no-wedge), iso-param.
If glados doesn't beat ldt on false-elimination at equal params, the wedge is decoration.
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
from . import sudoku as S


def device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def loss_fn(sup, alive, sol, dev, wpos=4.0, wneg=0.5, lcls=0.1, lce=0.2):
    """Deep-supervised LDT loss over the recurrent steps, vs the on-policy alpha target."""
    tgt, conflict, sing = S.alpha_target(alive, sol, dev)
    tot = 0.0
    for b, cls in sup:
        p = torch.sigmoid(b)
        bce = -(wpos * tgt * torch.log(p + 1e-6) + wneg * (1 - tgt) * torch.log(1 - p + 1e-6)).mean()
        clsl = F.binary_cross_entropy_with_logits(cls, conflict)
        # singleton CE: push the alive distribution toward the singleton target where it exists
        tlab = tgt.argmax(-1)
        ce_all = F.cross_entropy(b.reshape(-1, b.size(-1)), tlab.reshape(-1), reduction="none").reshape(b.shape[:-1])
        ce = (ce_all * sing).sum() / (sing.sum() + 1e-6)
        tot = tot + bce + lcls * clsl + lce * ce
    return tot / len(sup), conflict


@torch.no_grad()
def false_elim_rate(alive_before, alive_after, given, sol, dev):
    """Fraction of (non-given, true-still-alive) cells where the MEET just killed the true digit.
    This is the soundness metric: a sound deductor's value here is ~0."""
    sol = torch.as_tensor(sol, dtype=torch.long, device=dev)
    soli = (sol.clamp(min=1) - 1).unsqueeze(-1)
    true_before = alive_before.gather(2, soli).squeeze(-1)     # [B,81] true cand alive before
    true_after = alive_after.gather(2, soli).squeeze(-1)
    elig = (given < 0.5) * (true_before > 0.5)                 # could have been wrongly killed
    killed = elig * (true_after < 0.5)
    return killed.sum().item(), elig.sum().item()


def train(arm, puz, sol, dev, target, steps, pool=512, lr=3e-4, inner=16, log_every=None, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d, npar = size_for(arm, 81, 9, target, inner=inner)
    m = GLaDOS(arm, 81, 9, d=d, inner=inner, ds=inner).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    log_every = log_every or max(1, steps // 20)
    N = len(puz)

    idx = rng.integers(0, N, pool)
    alive, given = S.init_state(puz[idx], dev)
    fe_k = fe_n = 0
    print(f"dev={dev} arm={arm} d={d} params={npar:,} (target {target:,.0f}) pool={pool}")
    log = []
    t0 = time.time()
    for s in range(1, steps + 1):
        b, cls, sup = m(alive, given)
        loss, _ = loss_fn(sup, alive, sol[idx], dev)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()

        with torch.no_grad():
            conf = S.conflict_flag(cls)                  # ⊥ head on the state we just processed
            new = S.step_state(alive, given, b)
            k, n = false_elim_rate(alive, new, given, sol[idx], dev)
            fe_k += k; fe_n += n
            stalled = (new == alive).all(dim=-1).all(dim=-1)
            solved, _ = S.status(new)
            # branch the stalled-but-unsolved-and-consistent; recycle solved/conflicted as fresh puzzles
            need_branch = stalled & ~solved & ~conf
            if need_branch.any():
                bnew, _ = S.branch(new, b)
                new = torch.where(need_branch.view(-1, 1, 1), bnew, new)
            recycle = solved | conf
            if recycle.any():
                ridx = rng.integers(0, N, int(recycle.sum().item()))
                ra, rg = S.init_state(puz[ridx], dev)
                new[recycle] = ra; given[recycle] = rg
                idx[recycle.cpu().numpy()] = ridx
            alive = new

        if s % log_every == 0:
            fer = fe_k / max(1, fe_n)
            msg = {"step": s, "loss": float(loss.detach()), "false_elim": fer,
                   "frac_solved": float(solved.float().mean()), "mean_alive": float(alive.sum(-1).mean())}
            log.append(msg)
            print(f"  step {s:5d}  loss {loss.item():.3f}  false_elim {fer:.4f}  "
                  f"solved {msg['frac_solved']*100:4.1f}%  alive {msg['mean_alive']:.2f}  {time.time()-t0:.0f}s")
            fe_k = fe_n = 0
    return m, {"arm": arm, "d_model": d, "params": npar, "log": log}


@torch.no_grad()
def evaluate(m, puz, sol, dev, K=16, R_max=512, theta_elim=0.1):
    """K parallel chains per puzzle. A chain abstains on a dead cell; returns on all-singleton.
    Reports solve_rate, wrong_return_rate (soundness!), and p90 forward-passes over solved puzzles."""
    m.eval()
    E = len(puz)
    pe = np.repeat(puz, K, axis=0); se = np.repeat(sol, K, axis=0)
    alive, given = S.init_state(pe, dev)
    chain_done = torch.zeros(E * K, dtype=torch.bool, device=dev)
    chain_correct = torch.zeros(E * K, dtype=torch.bool, device=dev)
    chain_returned = torch.zeros(E * K, dtype=torch.bool, device=dev)
    chain_fwd = torch.zeros(E * K, dtype=torch.long, device=dev)
    for r in range(1, R_max + 1):
        live = ~chain_done
        if not live.any():
            break
        b, cls, _ = m(alive, given)
        chain_fwd[live] = r
        conf = S.conflict_flag(cls)
        new = S.step_state(alive, given, b)
        solved, _ = S.status(new)
        ok = S.is_correct(new, se, dev)
        fin_solved = solved & live
        fin_abstain = conf & live & ~solved             # ⊥ -> abstain (do NOT return an answer)
        chain_returned |= fin_solved
        chain_correct |= (fin_solved & ok)
        chain_done |= fin_solved | fin_abstain
        # branch stalled live chains
        stalled = (new == alive).all(-1).all(-1) & live & ~solved & ~conf
        if stalled.any():
            bnew, _ = S.branch(new, b)
            new = torch.where(stalled.view(-1, 1, 1), bnew, new)
        alive = new
    # reduce K chains -> per puzzle
    cr = chain_returned.view(E, K); cc = chain_correct.view(E, K); cf = chain_fwd.view(E, K)
    returned = cr.any(1)
    correct = cc.any(1)
    wrong = returned & ~correct
    # forwards = min forwards among correct chains (else 0)
    big = torch.where(cc, cf, torch.full_like(cf, 10**6))
    fwd_solved = big.min(1).values[correct]
    p90 = float(torch.quantile(fwd_solved.float(), 0.9)) if correct.any() else float("nan")
    m.train()
    return {"solve_rate": float(correct.float().mean()), "wrong_return_rate": float(wrong.float().mean()),
            "n_returned": int(returned.sum()), "p90_forwards": p90, "n_eval": E}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ldt", "glados", "nowedge"], required=True)
    ap.add_argument("--target", type=float, default=8e5)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--pool", type=int, default=512)
    ap.add_argument("--inner", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--source", choices=["extreme", "gen"], default="extreme")
    ap.add_argument("--dataset", default="sapientinc/sudoku-extreme")
    ap.add_argument("--ntrain", type=int, default=1000)
    ap.add_argument("--neval", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = device()
    rng = np.random.default_rng(a.seed)

    if a.source == "extreme":
        puz, sol = S.load_sudoku_extreme("train", limit=a.ntrain + a.neval)
    else:
        puz, sol = S.gen_puzzles(a.ntrain + a.neval, rng)
    tr_p, tr_s = puz[:a.ntrain], sol[:a.ntrain]
    ev_p, ev_s = puz[a.ntrain:a.ntrain + a.neval], sol[a.ntrain:a.ntrain + a.neval]

    m, res = train(a.arm, tr_p, tr_s, dev, a.target, a.steps, pool=a.pool, lr=a.lr, inner=a.inner, seed=a.seed)
    ev = evaluate(m, ev_p, ev_s, dev)
    res["eval"] = ev
    print(f"\nDONE {a.arm}: solve {ev['solve_rate']*100:.1f}%  wrong-return {ev['wrong_return_rate']*100:.2f}%  "
          f"p90-forwards {ev['p90_forwards']:.1f}  | final false_elim {res['log'][-1]['false_elim']:.4f}")
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "runs", f"glados_{a.arm}_{int(a.target)}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
