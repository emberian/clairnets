"""Sweep the GLaDOS LAYERING knobs on Sudoku-Extreme.

Mirrors clair/run_glados.py (same on-policy pool loop, same metrics: false_elim / solve /
wrong_return / p90) but the model is GLaDOSv and every structural choice is a CLI flag. Defaults
reproduce the baseline GLaDOS run, so:

  python -m clair.run_glados_variants --arm glados --target 800000 --steps 4000   # == baseline

The headline experiment — a differentiable meet woven into the recurrence vs the current
"meet only between forward passes":

  python -m clair.run_glados_variants --arm glados --meet_inside soft --steps 4000
  python -m clair.run_glados_variants --arm glados --meet_inside none --steps 4000   # control

JSON is tagged with the full knob config so a sweep is self-describing.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .glados_variants import GLaDOSv, size_for
from .run_glados import loss_fn, false_elim_rate, device
from . import sudoku as S


def knob_cfg(a):
    return {"arm": a.arm, "meet_inside": a.meet_inside, "order": a.order, "reinject": a.reinject,
            "tie": a.tie, "couple_state": a.couple_state, "ds_window": a.ds_window, "inner": a.inner}


def build(a, dev):
    tied = (a.tie == "tied")
    knobs = dict(meet_inside=a.meet_inside, order=a.order, reinject=a.reinject,
                 couple_state=a.couple_state, ds_window=a.ds_window)
    d, npar = size_for(a.arm, 81, 9, a.target, inner=a.inner, ds=a.inner, tied=tied, **knobs)
    m = GLaDOSv(a.arm, 81, 9, d=d, heads=4, inner=a.inner, ds=a.inner, tied=tied, **knobs).to(dev)
    return m, d, npar


def train(a, puz, sol, dev):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    m, d, npar = build(a, dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    log_every = max(1, a.steps // 20)
    N = len(puz)

    idx = rng.integers(0, N, a.pool)
    alive, given = S.init_state(puz[idx], dev)
    fe_k = fe_n = 0
    print(f"dev={dev} arm={a.arm} d={d} params={npar:,} (target {a.target:,.0f}) "
          f"meet_inside={a.meet_inside} order={a.order} reinject={a.reinject} "
          f"tie={a.tie} couple_state={a.couple_state} ds_window={a.ds_window} pool={a.pool}")
    log = []
    t0 = time.time()
    for s in range(1, a.steps + 1):
        # alive IS the lattice mask; pass it as the soundness floor for the in-loop meet.
        b, cls, sup = m(alive, given, alive=alive)
        loss, _ = loss_fn(sup, alive, sol[idx], dev)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()

        with torch.no_grad():
            conf = S.conflict_flag(cls)
            new = S.step_state(alive, given, b)
            k, n = false_elim_rate(alive, new, given, sol[idx], dev)
            fe_k += k; fe_n += n
            stalled = (new == alive).all(dim=-1).all(dim=-1)
            solved, _ = S.status(new)
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
    return m, {"knobs": knob_cfg(a), "d_model": d, "params": npar, "log": log}


@torch.no_grad()
def evaluate(m, puz, sol, dev, K=16, R_max=512):
    """K parallel chains per puzzle (mirrors run_glados.evaluate); GLaDOSv takes the alive mask."""
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
        b, cls, _ = m(alive, given, alive=alive)
        chain_fwd[live] = r
        conf = S.conflict_flag(cls)
        new = S.step_state(alive, given, b)
        solved, _ = S.status(new)
        ok = S.is_correct(new, se, dev)
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
    returned = cr.any(1)
    correct = cc.any(1)
    wrong = returned & ~correct
    big = torch.where(cc, cf, torch.full_like(cf, 10**6))
    fwd_solved = big.min(1).values[correct]
    p90 = float(torch.quantile(fwd_solved.float(), 0.9)) if correct.any() else float("nan")
    m.train()
    return {"solve_rate": float(correct.float().mean()), "wrong_return_rate": float(wrong.float().mean()),
            "n_returned": int(returned.sum()), "p90_forwards": p90, "n_eval": E}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ldt", "glados", "nowedge"], default="glados")
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
    # --- LAYERING knobs ---
    ap.add_argument("--meet_inside", choices=["none", "soft", "hard"], default="none")
    ap.add_argument("--order", choices=["attn_then_geom", "geom_then_attn", "parallel"], default="attn_then_geom")
    ap.add_argument("--reinject", choices=["add0p1", "gated", "concat", "none"], default="add0p1")
    ap.add_argument("--tie", choices=["tied", "untied"], default="tied")
    ap.add_argument("--couple_state", action="store_true")
    ap.add_argument("--ds_window", type=int, default=None)
    a = ap.parse_args()
    dev = device()
    rng = np.random.default_rng(a.seed)

    if a.source == "extreme":
        puz, sol = S.load_sudoku_extreme("train", limit=a.ntrain + a.neval)
    else:
        puz, sol = S.gen_puzzles(a.ntrain + a.neval, rng)
    tr_p, tr_s = puz[:a.ntrain], sol[:a.ntrain]
    ev_p, ev_s = puz[a.ntrain:a.ntrain + a.neval], sol[a.ntrain:a.ntrain + a.neval]

    m, res = train(a, tr_p, tr_s, dev)
    ev = evaluate(m, ev_p, ev_s, dev)
    res["eval"] = ev
    print(f"\nDONE {a.arm} [{a.meet_inside}/{a.order}/{a.reinject}/{a.tie}"
          f"{'/couple' if a.couple_state else ''}]: solve {ev['solve_rate']*100:.1f}%  "
          f"wrong-return {ev['wrong_return_rate']*100:.2f}%  p90-forwards {ev['p90_forwards']:.1f}  "
          f"| final false_elim {res['log'][-1]['false_elim']:.4f}")

    tag = f"{a.arm}_mi-{a.meet_inside}_ord-{a.order}_ri-{a.reinject}_{a.tie}" \
          f"{'_couple' if a.couple_state else ''}"
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "runs", f"gladosv_{tag}_{int(a.target)}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
