"""clair/energy_organ.py — RUNG-3 THIRD organ: ENERGY / EQUILIBRIUM (the soft/optimization shape).

The organ bank so far has two *discrete* shapes (both measured against the exact clair.csp oracle):

    NARROW  (clair.proposer)   : state alive[cell,value] in [0,1], MONOTONE SHRINK — RULES OUT
                                 candidates; soundness = never drop a true survivor (dominate dedₚ).
    CHAIN   (clair.chain_organ): state c[atom] in [0,1], MONOTONE GROWTH — DERIVES new facts;
                                 soundness = never derive an atom outside the true closure.

This is the THIRD, SOFT shape. Where narrowing rules out and chaining derives, ENERGY methods RELAX
toward a fixed point / optimum. It buys two things the discrete organs *cannot* do:
    (a) SOFT / weighted constraints, and
    (b) OPTIMIZATION — find the BEST consistent assignment, not just any (constraint optimisation).

REPRESENTATION.  Relax the discrete per-cell candidate SET to a continuous per-cell PROBABILITY
distribution.  x[i] = softmax(z[i]) over the d values — a point in the product of simplices, the
convex relaxation of the one-hot corners (= concrete assignments).

ENERGY.  E(x) = Σ_f  −log P_f(x)               (soft constraint-violation: 0 iff factor satisfied)
              + w_obj · ⟨cost, x⟩               (optional linear preference, for OPTIMISATION)
              − β · H(x)                        (entropy / decisiveness term; β annealed)
  where, treating the cells as independent, P_f(x) = Σ_{tuple ∈ allowed_f} Π_p x[scope_p, tuple_p]
  is the probability a sampled joint assignment SATISFIES factor f.  −log P_f is 0 iff all mass is on
  consistent tuples and →∞ as forbidden mass →1.  For a binary `≠` factor P_forbidden = Σ_v x_i[v]x_j[v]
  is exactly "mass on equal-value pairs"; for `=` it is the off-diagonal mass; pins are unary factors.

REASONING = descend E to a FIXED POINT.  Two equivalent differentiable solvers:
  • EQUILIBRIUM (DEQ / mean-field):  x_i* = softmax( −∂E/∂x_i / T ),  a damped fixed-point x* = step(x*).
    ∂(−log P_f)/∂x_i is the BP-style message (P_f marginalised over the other cells of f).  Solved by
    DETERMINISTIC ANNEALING: T high→low (a continuation method that dodges local minima).
  • GRADIENT (unrolled):  Adam on the logits z through E.  Used for the differentiable TRAINING demo
    (a soundness-asymmetric loss tunes the global temperature/weights, the dual of dominate-dedₚ).
  Both land on the same fixed point; the equilibrium solver is the analytic fast path for eval.

READOUT.  argmax_v x*[i] = the answer; a flat / high-entropy cell = abstain / underdetermined.

VALIDATION vs clair.csp (exact oracle), the four questions:
  (1) SOLVE   : on DETERMINED CSPs does argmax x* match the unique solution?           (decode mode, T→0)
  (2) SOUND   : does x* avoid CONFIDENT mass on values in NO solution (vs exact_dedP)?  (marginal mode)
  (3) OPTIMISE: with a cost term, does the rounded x* equal the brute-force MIN-cost consistent
                assignment? — the thing narrow/chain structurally cannot express.
  (4) FAILURES: local minima / rounding to an INFEASIBLE corner on a solvable CSP (honest).

Run:  python -m clair.energy_organ --smoke      # quick: descent converges + solves a tiny determined CSP
      python -m clair.energy_organ              # full: solve + soundness + optimisation + failure table
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import csp as C
from . import curriculum as CUR


# ============================================================================= factor tensors
def factor_tensors(csp: C.CSP, device):
    """Compile each constraint (scope, allowed-set) into a DENSE allowed-indicator tensor A of shape
    (d,)*arity with A[tuple]=1 iff that value-tuple is allowed. The energy reads P_f = einsum of the
    per-cell x's against A. Arities are small (<=3 mostly; alldiff up to n) so this stays tiny."""
    facs = []
    d = csp.d
    for sc, al in csp.cons:
        a = len(sc)
        A = torch.zeros((d,) * a, device=device)
        for tup in al:
            A[tup] = 1.0
        facs.append((tuple(sc), A))
    return facs


_LET = "abcdefghijklmn"


def _p_ok(x, scope, A, eps=1e-12):
    """P(factor satisfied) under independent per-cell marginals x: Σ_allowed Π x. A scalar in [0,1]."""
    a = len(scope)
    letters = _LET[:a]
    sub = ",".join(letters) + "," + letters + "->"
    return torch.einsum(sub, *[x[c] for c in scope], A).clamp_min(eps)


def _msg(x, scope, A, p):
    """The BP message ∂P_f/∂x_{scope[p]}: A marginalised over the OTHER cells of f (a length-d vector).
    This is exactly the "support from neighbours" each cell receives — the heart of the equilibrium step."""
    a = len(scope)
    letters = _LET[:a]
    others = [q for q in range(a) if q != p]
    if not others:                                  # unary factor (e.g. a pin): message is A itself
        return A
    sub = ",".join(letters[q] for q in others) + "," + letters + "->" + letters[p]
    return torch.einsum(sub, *[x[scope[q]] for q in others], A)


# ============================================================================= energy (autograd form)
def energy(x, factors, cost=None, w_obj=0.0, beta=0.0, eps=1e-12):
    """E(x) = Σ_f −log P_f  + w_obj·⟨cost,x⟩ − β·H(x).  Differentiable in x (used by the grad solver)."""
    e = x.new_zeros(())
    for sc, A in factors:
        e = e - torch.log(_p_ok(x, sc, A, eps))
    if cost is not None and w_obj:
        e = e + w_obj * (cost * x).sum()
    if beta:
        H = -(x.clamp_min(eps) * x.clamp_min(eps).log()).sum()
        e = e - beta * H
    return e


# ============================================================================= EQUILIBRIUM solver (mean-field)
@torch.no_grad()
def relax_mf(factors, n, d, device, steps=120, temp0=1.0, temp1=0.04, damp=0.5,
             cost=None, w_obj=0.0, x0=None, tol=1e-6, noise=0.0, seed=None):
    """Damped mean-field fixed-point with deterministic annealing — the DEQ/equilibrium solver.

        field_i = ∂E_constraint/∂x_i = Σ_{f∋i} −(1/P_f)·message_{f,i}   (+ w_obj·cost_i)
        x_i*    = softmax( −field_i / T ),   T annealed temp0 → temp1   (continuation / annealing)

    A genuine fixed point x* = step(x*): supported values pull mass in, forbidden ones push it out.
    `noise` perturbs the init logits to BREAK SYMMETRY (a perfectly uniform init is a degenerate mean-
    field fixed point for symmetric problems like coloring — all cells collapse to one value). Returns
    x* and a diagnostics dict (converged residual, #iters, soft constraint residual)."""
    if x0 is not None:
        x = x0.clone()
    elif noise > 0:
        g = torch.Generator(device=device).manual_seed(int(seed) if seed is not None else 0)
        x = F.softmax(noise * torch.randn(n, d, device=device, generator=g), dim=-1)
    else:
        x = torch.full((n, d), 1.0 / d, device=device)
    last_delta = float("nan")
    used = steps
    for t in range(steps):
        frac = t / max(1, steps - 1)
        T = temp0 * (temp1 / temp0) ** frac                      # geometric anneal
        field = torch.zeros(n, d, device=device)
        for sc, A in factors:
            pf = _p_ok(x, sc, A)
            for p, cell in enumerate(sc):
                field[cell] = field[cell] - _msg(x, sc, A, p) / pf
        if cost is not None and w_obj:
            field = field + w_obj * cost
        x_new = F.softmax(-field / T, dim=-1)
        x_new = (1 - damp) * x + damp * x_new
        delta = (x_new - x).abs().max().item()
        x = x_new
        last_delta = delta
        if delta < tol and frac > 0.6:                           # converged after some cooling
            used = t + 1
            break
    # soft constraint residual Σ_f (1 − P_f): 0 ⇔ all factors satisfied in expectation
    resid = float(sum((1.0 - _p_ok(x, sc, A)).item() for sc, A in factors))
    return x, {"residual_delta": last_delta, "iters": used, "soft_resid": resid}


@torch.no_grad()
def relax_decode(factors, n, d, device, steps=140, temp1=0.04, cost=None, w_obj=0.0,
                 restarts=4, noise=0.6):
    """Decode = anneal T->0 from several noisy inits (symmetry-broken random restarts) and keep the
    fixed point with the LOWEST soft constraint residual. Returns (best_x, best_info)."""
    best_x = best_info = None
    for r in range(restarts):
        x, info = relax_mf(factors, n, d, device, steps=steps, temp0=1.0, temp1=temp1,
                           cost=cost, w_obj=w_obj, noise=noise, seed=r)
        if best_info is None or info["soft_resid"] < best_info["soft_resid"]:
            best_x, best_info = x, info
    return best_x, best_info


# ============================================================================= GRADIENT solver (differentiable)
def relax_grad(factors, n, d, device, steps=150, lr=0.2, beta0=0.3, beta1=0.0,
               cost=None, w_obj=0.0, z0=None):
    """Unrolled gradient descent on the logits z (x=softmax(z)) through E — the DIFFERENTIABLE form.
    Backprop-able end-to-end (used by the training demo). Same fixed point as relax_mf."""
    z = (z0.clone() if z0 is not None else torch.zeros(n, d, device=device)).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = beta0 * (1 - frac) + beta1 * frac
        x = F.softmax(z, dim=-1)
        e = energy(x, factors, cost=cost, w_obj=w_obj, beta=beta)
        opt.zero_grad(); e.backward(); opt.step()
    return F.softmax(z, dim=-1)


# ============================================================================= oracle metrics
def feasible(csp: C.CSP, assign) -> bool:
    return all(tuple(assign[i] for i in sc) in al for sc, al in csp.cons)


def eval_csp_solve(csp: C.CSP, x, sols, exact, conf_thr=0.5, mass_eps=0.05):
    """Score one relaxed solution x [n,d] against the exact oracle.
      determined CSP -> accuracy = argmax matches the unique solution.
      soundness      -> confident mass on NEVER-survivor values (values in no solution, per exact_dedP).
      spread         -> for underdetermined cells, do the marginals cover the survivor set?
    Returns a per-CSP dict."""
    n, d = csp.n, csp.d
    xn = x.detach().cpu().numpy()
    argmax = [int(xn[i].argmax()) for i in range(n)]
    is_feasible = feasible(csp, argmax)
    determined = len(sols) == 1
    correct = determined and tuple(argmax) == sols[0]

    # soundness: mass on values that survive in NO solution
    never_mass = 0.0
    n_never_cells = 0
    confident_false = 0           # cells whose ARGMAX is a never-survivor
    any_never = 0
    for i in range(n):
        never = [v for v in range(d) if v not in exact[i]]
        if never:
            any_never += 1
            m = float(xn[i, never].sum())
            never_mass += m
            n_never_cells += 1
            if argmax[i] in never:
                confident_false += 1
    mean_never_mass = never_mass / max(1, n_never_cells)

    # spread on underdetermined cells: fraction of true survivors that get >= mass_eps
    surv_cov_num = surv_cov_den = 0
    und_cells = 0
    for i in range(n):
        if len(exact[i]) > 1:
            und_cells += 1
            for v in exact[i]:
                surv_cov_den += 1
                surv_cov_num += int(xn[i, v] >= mass_eps)
    return {
        "determined": determined, "correct": bool(correct), "feasible": bool(is_feasible),
        "mean_never_mass": mean_never_mass,
        "confident_false_cells": confident_false, "never_cells": n_never_cells,
        "und_cells": und_cells, "surv_cov_num": surv_cov_num, "surv_cov_den": surv_cov_den,
        # confident-false flag for SOLVABLE cells: argmax on a never-survivor with high mass
        "confident_false_event": int(any(
            (argmax[i] not in exact[i]) and (xn[i, argmax[i]] > conf_thr) for i in range(n))),
    }


# ============================================================================= corpus
def sample_corpus(rng, n_per, relations=("coloring", "equality", "ordering", "arithmetic", "alldiff")):
    """Solvable CSPs from the witness-first curriculum generators, tagged by relation type."""
    out = []
    for rel in relations:
        gen = CUR.GENERATORS[rel]
        got = 0
        tries = 0
        while got < n_per and tries < n_per * 40:
            tries += 1
            cn, cd, _kind, facts, _s = gen(rng)
            csp = CUR.build_csp(cn, cd, facts)
            sols = C.solutions(csp)
            if not sols:
                continue                                          # generators are witness-first; guard anyway
            out.append((rel, csp))
            got += 1
    return out


# ============================================================================= optimisation (the new capability)
def brute_opt(csp: C.CSP, cost_np):
    """Brute-force the MIN-cost consistent assignment: argmin_{s ∈ solutions} Σ_i cost[i,s_i].
    Returns (best_cost, worst_cost, best_assign). Exact oracle for the optimisation question."""
    sols = C.solutions(csp)
    costs = [(sum(cost_np[i, s[i]] for i in range(csp.n)), s) for s in sols]
    best = min(costs, key=lambda t: t[0])
    worst = max(c for c, _ in costs)
    return best[0], worst, best[1]


def eval_optimisation(rng, csps, device, w_obj=0.6, steps=200):
    """For each solvable CSP attach a random per-cell linear cost, find the min-cost consistent
    assignment by energy descent (constraints + cost), and compare to the brute-force optimum."""
    hit = feas = tot = 0
    gaps = []
    margins = []
    for csp in csps:
        sols = C.solutions(csp)
        if len(sols) < 2:                                         # need a real choice to optimise over
            continue
        tot += 1
        cost_np = rng.random((csp.n, csp.d)).astype(np.float32)
        cost = torch.as_tensor(cost_np, device=device)
        opt_cost, worst_cost, _ = brute_opt(csp, cost_np)
        factors = factor_tensors(csp, device)
        x, _ = relax_decode(factors, csp.n, csp.d, device, steps=steps,
                            temp1=0.03, cost=cost, w_obj=w_obj, restarts=4)
        assign = [int(x[i].argmax()) for i in range(csp.n)]
        is_feas = feasible(csp, assign)
        feas += int(is_feas)
        if is_feas:
            ac = sum(cost_np[i, assign[i]] for i in range(csp.n))
            rng_span = max(1e-9, worst_cost - opt_cost)
            gaps.append((ac - opt_cost) / rng_span)               # 0 = optimal, 1 = worst feasible
            hit += int(abs(ac - opt_cost) < 1e-6)
            margins.append(rng_span)
    return {
        "n": tot, "feasible_rate": feas / max(1, tot), "optimal_rate": hit / max(1, tot),
        "mean_norm_gap": float(np.mean(gaps)) if gaps else float("nan"),
        "median_norm_gap": float(np.median(gaps)) if gaps else float("nan"),
    }


# ============================================================================= aggregate eval
def evaluate(corpus, device, marg_temp=0.3, decode_temp1=0.04, steps=140):
    """Run the equilibrium solver on every CSP in two modes and aggregate the four questions by
    relation type: DECODE (annealed T→0, for solve accuracy + feasibility) and MARGINAL (fixed
    moderate T, for soundness + survivor spread)."""
    per = {}
    glob = {"n": 0, "det": 0, "det_correct": 0, "feas_solvable": 0, "infeasible_solvable": 0,
            "conf_false_events": 0, "never_mass_sum": 0.0, "never_cells": 0,
            "surv_cov_num": 0, "surv_cov_den": 0, "soft_resid_sum": 0.0, "iters_sum": 0}
    for rel, csp in corpus:
        sols = C.solutions(csp)
        if not sols:
            continue
        exact = C.exact_dedP(csp, csp.full())
        factors = factor_tensors(csp, device)
        # DECODE mode: anneal to near-zero T from symmetry-broken restarts -> decisive corners
        xd, info_d = relax_decode(factors, csp.n, csp.d, device, steps=steps,
                                  temp1=decode_temp1, restarts=4)
        # MARGINAL mode: fixed moderate T -> read survivor spread / soundness
        xm, _ = relax_mf(factors, csp.n, csp.d, device, steps=steps,
                         temp0=marg_temp, temp1=marg_temp, damp=0.4)
        md = eval_csp_solve(csp, xd, sols, exact)
        mm = eval_csp_solve(csp, xm, sols, exact)

        p = per.setdefault(rel, {"n": 0, "det": 0, "det_correct": 0, "feas_solvable": 0,
                                 "infeasible_solvable": 0, "conf_false_events": 0,
                                 "never_mass_sum": 0.0, "never_cells": 0,
                                 "surv_cov_num": 0, "surv_cov_den": 0})
        p["n"] += 1; glob["n"] += 1
        glob["soft_resid_sum"] += info_d["soft_resid"]; glob["iters_sum"] += info_d["iters"]
        if md["determined"]:
            p["det"] += 1; glob["det"] += 1
            p["det_correct"] += int(md["correct"]); glob["det_correct"] += int(md["correct"])
        p["feas_solvable"] += int(md["feasible"]); glob["feas_solvable"] += int(md["feasible"])
        p["infeasible_solvable"] += int(not md["feasible"])
        glob["infeasible_solvable"] += int(not md["feasible"])
        # soundness + spread read from MARGINAL mode (mass spread is meaningful there)
        p["conf_false_events"] += mm["confident_false_event"]
        glob["conf_false_events"] += mm["confident_false_event"]
        p["never_mass_sum"] += mm["mean_never_mass"] * max(1, mm["never_cells"])
        p["never_cells"] += mm["never_cells"]
        glob["never_mass_sum"] += mm["mean_never_mass"] * max(1, mm["never_cells"])
        glob["never_cells"] += mm["never_cells"]
        p["surv_cov_num"] += mm["surv_cov_num"]; p["surv_cov_den"] += mm["surv_cov_den"]
        glob["surv_cov_num"] += mm["surv_cov_num"]; glob["surv_cov_den"] += mm["surv_cov_den"]
    return {"per_kind": per, "global": glob}


# ============================================================================= training demo (soundness-asymmetric)
def train_temperature(corpus, device, steps=120, lr=0.05):
    """Differentiable demo honoring the chain/narrow discipline: tune a GLOBAL log-temperature and a
    constraint scale by backprop through the unrolled gradient solver, under a SOUNDNESS-ASYMMETRIC
    loss (heavy penalty for mass on never-survivors; light penalty for missing survivors) — the dual
    of dominate-dedₚ. Shows the energy organ is end-to-end differentiable and can be tuned sound-ish."""
    log_scale = torch.zeros((), device=device, requires_grad=True)   # softplus -> constraint weight ~1
    log_beta = torch.tensor(-1.0, device=device, requires_grad=True)  # entropy weight
    opt = torch.optim.Adam([log_scale, log_beta], lr=lr)
    # precompute targets
    items = []
    for rel, csp in corpus:
        sols = C.solutions(csp)
        if not sols:
            continue
        exact = C.exact_dedP(csp, csp.full())
        surv = torch.zeros(csp.n, csp.d, device=device)
        for i in range(csp.n):
            for v in exact[i]:
                surv[i, v] = 1.0
        items.append((factor_tensors(csp, device), csp.n, csp.d, surv))
    log = []
    for s in range(1, steps + 1):
        opt.zero_grad()
        scale = F.softplus(log_scale)
        beta = F.softplus(log_beta)
        l_unsound = l_incomplete = 0.0
        for factors, n, d, surv in items:
            z = torch.zeros(n, d, device=device, requires_grad=True)
            zo = torch.optim.SGD([z], lr=0.0)  # placeholder; we do manual unroll for grad flow
            # short unrolled descent (kept tiny for speed), differentiable in scale/beta
            for _t in range(12):
                x = F.softmax(z, dim=-1)
                e = scale * sum(-torch.log(_p_ok(x, sc, A)) for sc, A in factors)
                e = e - beta * (-(x.clamp_min(1e-9) * x.clamp_min(1e-9).log()).sum())
                (g,) = torch.autograd.grad(e, z, create_graph=True)
                z = z - 0.3 * g
            x = F.softmax(z, dim=-1)
            l_unsound = l_unsound + (x * (1 - surv)).sum() / max(1.0, float((1 - surv).sum()))
            l_incomplete = l_incomplete + ((surv - x).clamp_min(0) * surv).sum() / max(1.0, float(surv.sum()))
        l_unsound = l_unsound / len(items); l_incomplete = l_incomplete / len(items)
        loss = 5.0 * l_unsound + 1.0 * l_incomplete
        loss.backward(); opt.step()
        if s % 30 == 0 or s == 1:
            row = {"step": s, "loss": float(loss), "unsound": float(l_unsound),
                   "incomplete": float(l_incomplete), "scale": float(F.softplus(log_scale)),
                   "beta": float(F.softplus(log_beta))}
            log.append(row)
            print(f"  [train] step {s:4d}  loss {loss:.4f}  unsound {float(l_unsound):.4f}  "
                  f"incomplete {float(l_incomplete):.4f}  scale {row['scale']:.3f}  beta {row['beta']:.3f}",
                  flush=True)
    return {"log": log, "scale": float(F.softplus(log_scale)), "beta": float(F.softplus(log_beta))}


# ============================================================================= smoke
def smoke(device):
    print("== SMOKE: energy descent runs, converges, solves a tiny determined CSP ==")
    # chain_eq(4): x0=false, x_i=x_{i+1} -> unique solution all-zero (determined, AC-solvable)
    ch = C.chain_eq(4)
    facs = factor_tensors(ch, device)
    x, info = relax_mf(facs, ch.n, ch.d, device, steps=120, temp0=1.0, temp1=0.04)
    sol = C.solutions(ch)[0]
    argmax = tuple(int(x[i].argmax()) for i in range(ch.n))
    print(f"  chain_eq(4): unique sol={sol}  argmax={argmax}  "
          f"match={argmax == sol}  iters={info['iters']}  soft_resid={info['soft_resid']:.2e}")
    assert argmax == sol, "energy descent must solve the determined chain"
    assert info["soft_resid"] < 1e-2, "must converge to a (near-)feasible fixed point"

    # a determined coloring instance
    col = C.coloring(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], k=3)
    facs = factor_tensors(col, device)
    x, info = relax_decode(facs, col.n, col.d, device, steps=160, temp1=0.03, restarts=4)
    am = [int(x[i].argmax()) for i in range(col.n)]
    print(f"  coloring(5-edge,k=3): argmax={am}  feasible={feasible(col, am)}  resid={info['soft_resid']:.2e}")

    # equality marginal spread: {x=y} underdetermined -> both cells should keep mass on {0,1}
    eq = C.CSP(2, 2, (C._rel((0, 1), lambda t: t[0] == t[1], 2),))
    facs = factor_tensors(eq, device)
    xm, _ = relax_mf(facs, 2, 2, device, steps=120, temp0=0.3, temp1=0.3, damp=0.4)
    print(f"  equality {{x=y}} marginal (underdetermined): x0={xm[0].tolist()}  x1={xm[1].tolist()}")

    # optimisation smoke: cost prefers value -> min-cost consistent assignment matches brute force
    rng = np.random.default_rng(0)
    r = eval_optimisation(rng, [C.coloring(4, [(0, 1), (1, 2), (2, 3)], k=3) for _ in range(8)],
                          device, w_obj=0.6, steps=200)
    print(f"  optimisation smoke (chain-coloring): optimal_rate={r['optimal_rate']:.2f} "
          f"feasible_rate={r['feasible_rate']:.2f} mean_gap={r['mean_norm_gap']:.3f}")
    print("  SMOKE OK\n")


# ============================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n_per", type=int, default=80, help="CSPs per relation type")
    ap.add_argument("--steps", type=int, default=140)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", action="store_true", help="run the differentiable tuning demo")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    print(f"device={dev}  torch={torch.__version__}")

    smoke(dev)
    if a.smoke:
        return

    rng = np.random.default_rng(a.seed)
    corpus = sample_corpus(rng, a.n_per)
    print(f"corpus: {len(corpus)} solvable CSPs across {len(set(k for k, _ in corpus))} relation types\n")

    t0 = time.time()
    ev = evaluate(corpus, dev, steps=a.steps)
    print(f"eval wall: {time.time()-t0:.1f}s\n")

    g = ev["global"]
    print("================ ENERGY / EQUILIBRIUM vs EXACT ORACLE ================\n")
    print("(1) SOLVE  — determined CSPs, argmax x* == unique solution (decode mode, T->0)")
    print(f"    determined accuracy : {g['det_correct']}/{g['det']} "
          f"= {g['det_correct']/max(1,g['det'])*100:5.1f}%")
    print("\n(2) SOUND  — confident mass on NEVER-survivor values (vs exact_dedP, marginal mode)")
    print(f"    mean mass on never-survivor cells : {g['never_mass_sum']/max(1,g['never_cells'])*100:6.3f}%")
    print(f"    confident-false events (>.5 mass on a non-survivor argmax) : "
          f"{g['conf_false_events']}/{g['n']}")
    print(f"    survivor coverage (underdetermined cells, marginals reach survivors) : "
          f"{g['surv_cov_num']}/{g['surv_cov_den']} "
          f"= {g['surv_cov_num']/max(1,g['surv_cov_den'])*100:5.1f}%")
    print("\n(4) FAILURE — rounding to an INFEASIBLE corner on a SOLVABLE CSP (local minima)")
    print(f"    infeasible-decode rate : {g['infeasible_solvable']}/{g['n']} "
          f"= {g['infeasible_solvable']/max(1,g['n'])*100:5.1f}%")
    print(f"    mean soft constraint residual at fixpoint : {g['soft_resid_sum']/max(1,g['n']):.3e}   "
          f"mean iters {g['iters_sum']/max(1,g['n']):.1f}")

    print("\n  per relation type:")
    print(f"  {'relation':11s} {'n':>4} {'det':>4} {'det_acc':>8} {'feas%':>7} {'infeas':>7} "
          f"{'never_mass':>11} {'surv_cov':>9} {'conf_false':>10}")
    for rel, p in sorted(ev["per_kind"].items()):
        det_acc = p["det_correct"] / max(1, p["det"]) * 100
        feasp = p["feas_solvable"] / max(1, p["n"]) * 100
        infp = p["infeasible_solvable"] / max(1, p["n"]) * 100
        nm = p["never_mass_sum"] / max(1, p["never_cells"]) * 100
        sc = p["surv_cov_num"] / max(1, p["surv_cov_den"]) * 100
        print(f"  {rel:11s} {p['n']:>4} {p['det']:>4} {det_acc:>7.1f}% {feasp:>6.1f}% {infp:>6.1f}% "
              f"{nm:>10.3f}% {sc:>8.1f}% {p['conf_false_events']:>10}")

    # (3) OPTIMISATION
    print("\n(3) OPTIMISE — min-cost consistent assignment (brute-force checked) — narrow/chain CANNOT")
    opt_rng = np.random.default_rng(a.seed + 7)
    csps = [c for _, c in corpus]
    ro = eval_optimisation(opt_rng, csps, dev, w_obj=0.6, steps=max(200, a.steps))
    print(f"    instances with >=2 solutions : {ro['n']}")
    print(f"    feasible-rate : {ro['feasible_rate']*100:5.1f}%   "
          f"OPTIMAL-rate (== brute-force min) : {ro['optimal_rate']*100:5.1f}%")
    print(f"    mean normalised cost gap : {ro['mean_norm_gap']:.4f}   "
          f"median : {ro['median_norm_gap']:.4f}   (0 = optimal, 1 = worst feasible)")

    train_out = None
    if a.train:
        print("\n---- differentiable tuning demo (soundness-asymmetric loss; dual of dominate-dedₚ) ----")
        train_out = train_temperature(corpus[: min(60, len(corpus))], dev)

    out = Path(a.out or (Path(__file__).resolve().parent.parent / "runs" / "energy_organ.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(a), "eval": ev, "optimisation": ro, "train": train_out},
                              indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
