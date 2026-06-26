"""Faithful LDT (arXiv 2605.08605) on Sudoku-Extreme: on-policy training + a REAL search solver.

  python -m clair.run_ldt --mixer ffn --steps 4000 --source extreme

What was broken in the parked attempt (clair/run_glados.py) and is fixed here:

  * The eval ran forward-only chains: step_state RESTORED the best candidate so a cell could
    never go empty, and the CLS conflict head was effectively dead -> a conflict was NEVER
    detected -> every chain "completed" at some singleton state and returned it. With no
    conflict signal there was no backtracking, so the committed (often wrong) branch path was
    returned: 100% completed, ~100% WRONG.

  * The real LDT solver (Algorithm 1 + 2) is a random-restart depth-first SEARCH:
      - deduction LETS a cell go empty  -> structural conflict (bottom),
      - the trained CLS head fires on unsat states -> semantic conflict,
      - a chain that conflicts is a DEAD branch: it is ABANDONED and that search slot RESTARTS
        from a fresh copy of the root puzzle (this is the backtracking — chronological restart,
        not undo-last-pin), and tries a different random branch path,
      - a chain only RETURNS when it reaches all-singletons WITHOUT any conflict ever firing.
    Soundness (wrong_return ~ 0) then follows from: elimination is SOUND (never kills the true
    digit) and conflict detection is COMPLETE (never misses an unsat state). M slots run in
    parallel per puzzle, R rounds deep; if no slot returns within budget the solver ABSTAINS.

Metrics: solve_rate (correct/total), wrong_return_rate (MUST be ~0 = sound), p90 forward-passes
over solved puzzles, false_elim (training-time soundness of the meet).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .ldt import LDT
from . import sudoku as S
from . import perf


def device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# --------------------------------------------------------------------------- lattice meet (solver)
@torch.no_grad()
def meet_eliminate(alive, given, b_logits, theta_elim):
    """Threshold elimination (Algorithm 2, line 3). MEET: drop currently-alive candidates whose
    survival prob < theta. Protect given clues. Crucially, unlike the parked step_state, we DO
    NOT restore a best candidate when a cell empties -> an empty cell IS the structural conflict
    signal the search relies on."""
    keep = (torch.sigmoid(b_logits) >= theta_elim).float()
    new = alive * keep
    new = torch.where(given.unsqueeze(-1) > 0, alive, new)   # clues are immovable
    return new


# --------------------------------------------------------------------------- loss
def loss_fn(sup, alive, sol, dev, wpos=4.0, wneg=0.5, lcls=0.1, lce=0.2):
    """Deep-supervised LDT loss (Eq. 1) over all recurrent iterations vs the on-policy alpha target.
      * asymmetric BCE on survival logits (w+/w- = 8 penalizes false elimination ~8x harder),
      * symmetric BCE on the CLS conflict logit vs the ground-truth conflict label,
      * per-cell softmax CE at singleton-target cells (helps the model converge faster).
    Returns (mean loss, conflict label, last-iter predicted P(conflict))."""
    tgt, conflict, sing = S.alpha_target(alive, sol, dev)
    tot = 0.0
    for b, c in sup:
        p = torch.sigmoid(b)
        bce = -(wpos * tgt * torch.log(p + 1e-6) + wneg * (1 - tgt) * torch.log(1 - p + 1e-6)).mean()
        clsl = F.binary_cross_entropy_with_logits(c, conflict)
        tlab = tgt.argmax(-1)
        ce_all = F.cross_entropy(b.reshape(-1, b.size(-1)), tlab.reshape(-1),
                                 reduction="none").reshape(b.shape[:-1])
        ce = (ce_all * sing).sum() / (sing.sum() + 1e-6)
        tot = tot + bce + lcls * clsl + lce * ce
    return tot / len(sup), conflict, torch.sigmoid(sup[-1][1])


@torch.no_grad()
def false_elim_rate(alive_before, alive_after, given, sol, dev):
    """Fraction of (non-given, true-still-alive) cells where the meet just killed the true digit.
    The soundness metric for deduction: a sound deductor sits at ~0."""
    sol = torch.as_tensor(sol, dtype=torch.long, device=dev)
    soli = (sol.clamp(min=1) - 1).unsqueeze(-1)
    tb = alive_before.gather(2, soli).squeeze(-1)
    ta = alive_after.gather(2, soli).squeeze(-1)
    elig = (given < 0.5) * (tb > 0.5)
    killed = elig * (ta < 0.5)
    return killed.sum().item(), elig.sum().item()


# --------------------------------------------------------------------------- training (parallel-solve pool)
def train(mixer, puz, sol, dev, steps, pool=512, lr=3e-4, d=128, n_layers=4, inner=16,
          wpos=4.0, wneg=0.5, lcls=0.1, lce=0.2, theta_elim=0.1, tau=1.5,
          seed=0, amp=False, log_every=None, eval_every=0, eval_set=None,
          eval_M=32, eval_R=128, theta_cls=0.6):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = LDT(81, 9, d=d, n_layers=n_layers, inner=inner, ds=inner, mixer=mixer).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    log_every = log_every or max(1, steps // 25)
    N = len(puz)

    idx = rng.integers(0, N, pool)                       # which puzzle each pool slot holds
    alive, given = S.init_state(puz[idx], dev)
    fe_k = fe_n = 0
    print(f"dev={dev} mixer={mixer} d={d} layers={n_layers} inner={inner} "
          f"params={model.n_params():,} pool={pool} steps={steps}", flush=True)
    log = []
    t0 = time.time()
    for s in range(1, steps + 1):
        with perf.amp(dev, amp):
            _, _, sup = model(alive, given)
        sup = [(bb.float(), cc.float()) for bb, cc in sup]   # decisions / loss in fp32 (soundness)
        loss, conflict, p_conf = loss_fn(sup, alive, sol[idx], dev, wpos, wneg, lcls, lce)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        with torch.no_grad():
            # --- Step (Algorithm 2): eliminate with the iteration-AVERAGED survival logits
            b_mean = torch.stack([b for b, _ in sup], 0).mean(0)
            new = meet_eliminate(alive, given, b_mean, theta_elim)
            k, n = false_elim_rate(alive, new, given, sol[idx], dev)
            fe_k += k; fe_n += n
            # --- ground-truth terminal status of the stepped state (verified against Y)
            _, conf_new, _ = S.alpha_target(new, sol[idx], dev)     # branch killed a true digit
            empty_new = (new.sum(-1) == 0).any(-1)                  # structural bottom
            solved_new, _ = S.status(new)                           # all singletons
            terminal = conf_new.bool() | empty_new | solved_new
            # --- branch the non-terminal states (sample-pin a multi-candidate cell)
            branchable = ~terminal
            if branchable.any():
                bnew, _ = S.branch(new, b_mean, tau=tau)
                new = torch.where(branchable.view(-1, 1, 1), bnew, new)
            # --- recycle truly-terminal states with fresh puzzles (keeps a depth curriculum)
            if terminal.any():
                ridx = rng.integers(0, N, int(terminal.sum().item()))
                ra, rg = S.init_state(puz[ridx], dev)
                new[terminal] = ra; given[terminal] = rg
                idx[terminal.cpu().numpy()] = ridx
            alive = new

        if s % log_every == 0 or s == steps:
            fer = fe_k / max(1, fe_n)
            pos = float(p_conf[conflict > 0.5].mean()) if (conflict > 0.5).any() else float("nan")
            neg = float(p_conf[conflict < 0.5].mean()) if (conflict < 0.5).any() else float("nan")
            msg = {"step": s, "loss": float(loss.detach()), "false_elim": fer,
                   "cls_on_unsat": pos, "cls_on_sat": neg,
                   "frac_conflict": float((conflict > 0.5).float().mean()),
                   "mean_alive": float(alive.sum(-1).mean())}
            log.append(msg)
            print(f"  step {s:5d}  loss {loss.item():.3f}  false_elim {fer:.5f}  "
                  f"cls[unsat {pos:.2f}/sat {neg:.2f}]  conf {msg['frac_conflict']*100:4.1f}%  "
                  f"alive {msg['mean_alive']:.2f}  {time.time()-t0:.0f}s", flush=True)
            fe_k = fe_n = 0

        if eval_every and eval_set is not None and (s % eval_every == 0):
            ep, es = eval_set
            ev = solve(model, ep, es, dev, M=eval_M, R=eval_R,
                       theta_elim=theta_elim, theta_cls=theta_cls, tau=tau, amp=amp)
            print(f"    [eval @ {s}] solve {ev['solve_rate']*100:5.1f}%  WRONG {ev['wrong_return_rate']*100:5.2f}%  "
                  f"abstain {ev['abstain_rate']*100:5.1f}%  (M={eval_M},R={eval_R})", flush=True)
    return model, {"mixer": mixer, "d_model": d, "params": model.n_params(), "log": log}


# --------------------------------------------------------------------------- the REAL solver (search + backtracking)
@torch.no_grad()
def solve(model, puz, sol, dev, M=64, R=512, theta_elim=0.1, theta_cls=0.6, tau=1.5, amp=False):
    """Random-restart depth-first search (Algorithm 1 + 2 at inference). M parallel slots per
    puzzle, R rounds deep. A slot deduces, then branches; on conflict (empty cell OR CLS>theta)
    the slot is ABANDONED and RESTARTED from the fresh root puzzle (backtracking). A puzzle is
    returned the first round any of its slots reaches all-singletons with NO conflict; otherwise
    the solver ABSTAINS. Returns soundness/efficiency metrics."""
    model.eval()
    E = len(puz)
    sol_t = torch.as_tensor(sol, dtype=torch.long, device=dev)             # [E,81]
    pe = np.repeat(puz, M, axis=0)
    init_alive, given = S.init_state(pe, dev)                              # [E*M,81,9]
    alive = init_alive.clone()
    pid = torch.arange(E, device=dev).repeat_interleave(M)                 # slot -> puzzle id

    solved_puz = torch.zeros(E, dtype=torch.bool, device=dev)
    answer = torch.zeros(E, 81, dtype=torch.long, device=dev)
    fwd_at = torch.zeros(E, dtype=torch.long, device=dev)
    n_restart = torch.zeros(E * M, dtype=torch.long, device=dev)
    rounds_used = 0

    for r in range(1, R + 1):
        active = ~solved_puz[pid]                                          # [E*M]
        if not active.any():
            break
        rounds_used = r
        with perf.amp(dev, amp):
            b, c, _ = model(alive, given)
        b, c = b.float(), c.float()
        new = meet_eliminate(alive, given, b, theta_elim)
        empty = (new.sum(-1) == 0).any(-1)                                # [E*M] structural bottom
        conflict = (torch.sigmoid(c) > theta_cls) | empty                 # CLS head OR empty cell
        nsing = (new.sum(-1) == 1).all(-1)                                # all singletons
        solved = nsing & ~conflict & active                              # a clean solution

        # --- record returns: first slot of each newly-solved puzzle wins ---
        sl = solved.nonzero(as_tuple=False).flatten()
        if sl.numel():
            pids = pid[sl]
            preds = new[sl].argmax(-1) + 1                                # [k,81]
            answer[pids] = preds                                          # any valid slot
            fwd_at[pids] = r
            solved_puz[pids] = True

        # --- backtrack: conflicted slots restart from the fresh root puzzle ---
        restart = conflict & active & ~solved
        if restart.any():
            new = torch.where(restart.view(-1, 1, 1), init_alive, new)
            n_restart += restart.long()
        # --- branch the surviving non-terminal slots ---
        branchable = active & ~solved & ~conflict
        if branchable.any():
            bnew, _ = S.branch(new, b, tau=tau)
            new = torch.where(branchable.view(-1, 1, 1), bnew, new)
        alive = new

    returned = solved_puz
    correct = returned & (answer == sol_t).all(-1)
    wrong = returned & ~correct
    p90 = float(torch.quantile(fwd_at[correct].float(), 0.9)) if correct.any() else float("nan")
    p50 = float(torch.quantile(fwd_at[correct].float(), 0.5)) if correct.any() else float("nan")
    model.train()
    return {
        "solve_rate": float(correct.float().mean()),
        "wrong_return_rate": float(wrong.float().mean()),
        "abstain_rate": float((~returned).float().mean()),
        "n_returned": int(returned.sum()), "n_correct": int(correct.sum()),
        "n_wrong": int(wrong.sum()), "n_eval": E,
        "p50_forwards": p50, "p90_forwards": p90,
        "mean_restarts_per_slot": float(n_restart.float().mean()),
        "rounds_used": rounds_used,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixer", choices=["ffn", "geom", "nowedge"], default="ffn")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--pool", type=int, default=512)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--inner", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--source", choices=["extreme", "gen"], default="extreme")
    ap.add_argument("--ntrain", type=int, default=1000)
    ap.add_argument("--neval", type=int, default=128)
    ap.add_argument("--M", type=int, default=64, help="parallel search slots per puzzle")
    ap.add_argument("--rounds", default="64,128,256,512", help="inference round-budget sweep")
    ap.add_argument("--theta_elim", type=float, default=0.1)
    ap.add_argument("--theta_cls", type=float, default=0.6)
    ap.add_argument("--tau", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=0, help="run a quick held-out solve() every N steps")
    ap.add_argument("--eval_M", type=int, default=32)
    ap.add_argument("--eval_R", type=int, default=128)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    perf.setup()
    dev = device()
    rng = np.random.default_rng(a.seed)

    if a.source == "extreme":
        puz, sol = S.load_sudoku_extreme("train", limit=a.ntrain + a.neval)
    else:
        puz, sol = S.gen_puzzles(a.ntrain + a.neval, rng)
    tr_p, tr_s = puz[:a.ntrain], sol[:a.ntrain]
    ev_p, ev_s = puz[a.ntrain:a.ntrain + a.neval], sol[a.ntrain:a.ntrain + a.neval]

    model, res = train(a.mixer, tr_p, tr_s, dev, a.steps, pool=a.pool, lr=a.lr,
                       d=a.d, n_layers=a.layers, inner=a.inner, seed=a.seed, amp=a.amp,
                       theta_elim=a.theta_elim, tau=a.tau,
                       eval_every=a.eval_every, eval_set=(ev_p, ev_s),
                       eval_M=a.eval_M, eval_R=a.eval_R, theta_cls=a.theta_cls)

    rounds = [int(x) for x in a.rounds.split(",")]
    res["eval_sweep"] = {}
    print(f"\n--- inference search sweep (M={a.M} slots, theta_cls={a.theta_cls}) ---", flush=True)
    for R in rounds:
        ev = solve(model, ev_p, ev_s, dev, M=a.M, R=R,
                   theta_elim=a.theta_elim, theta_cls=a.theta_cls, tau=a.tau, amp=a.amp)
        res["eval_sweep"][str(R)] = ev
        print(f"  R {R:5d}  solve {ev['solve_rate']*100:5.1f}%  WRONG {ev['wrong_return_rate']*100:5.2f}%  "
              f"abstain {ev['abstain_rate']*100:5.1f}%  | p50/p90 fwd {ev['p50_forwards']}/{ev['p90_forwards']}  "
              f"restarts/slot {ev['mean_restarts_per_slot']:.1f}", flush=True)
    res["eval"] = res["eval_sweep"][str(rounds[-1])]
    ev = res["eval"]
    print(f"\nDONE mixer={a.mixer}: solve {ev['solve_rate']*100:.1f}%  wrong-return {ev['wrong_return_rate']*100:.2f}%  "
          f"abstain {ev['abstain_rate']*100:.1f}%  | final false_elim {res['log'][-1]['false_elim']:.5f}", flush=True)
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "runs", f"ldt_{a.mixer}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"state": model.state_dict(), "d_model": a.d, "mixer": a.mixer}, out.replace(".json", ".pt"))
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out, "+ checkpoint", flush=True)


if __name__ == "__main__":
    main()
