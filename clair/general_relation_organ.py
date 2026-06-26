"""clair/general_relation_organ.py — box w2a: fix the organ's #1 weakness (OOD-novel-relation
soundness break).

The documented weakness (notes/organ_excellence.md §5.1 + big_organ): the family-trained organ
BREAKS soundness ZERO-SHOT on a genuinely NOVEL relation table — n_queens off-diagonal
`|cᵢ−cⱼ| ≠ |i−j|` gets false-elim ~0.004 / ~3% wrong even at the LARGER budget, so it is a
RELATION-NOVELTY problem, not a budget problem.

Hypothesis tested here: the organ already takes the relation's allowed-tuple TABLE as an INPUT
(run_general.relation_table -> fac_rel), so it CAN narrow any relation. It breaks OOD only because
it was trained on a handful of named families (=/≠/</+/alldiff). Train it instead on a
RANDOMIZED-RELATION curriculum — constraints drawn from MANY random allowed-tuple relations
(random subsets of value-tuples of arity 1/2/3, witness-first so always solvable) — so it must learn
the GENERAL operation "soundly narrow given this relation table", not memorize specific families.
The blade/grade-k prior (BladeFactorDeductor, msg="blade") keeps the arity-k cell coupling
structurally correct a priori, which §3 shows buys soundness off-distribution.

Then test ZERO-SHOT on HELD-OUT relation families the random-relation organ NEVER saw as such:
n_queens (off-diagonal), futoshiki (<), latin / sudoku4 (alldiff/Hall), and a fresh held-out random
distribution — and compare soundness (false-elim, the key metric) to the family-trained baseline
(big_organ's TRAIN_RUNGS organ). Reuses big_organ's larger budget (N=25,D=6,M=160,A=3), data path,
and exact-dedP evaluator so the head-to-head is apples-to-apples.

  python -m clair.general_relation_organ --smoke
  python -m clair.general_relation_organ --full --steps 1500 --target 4e5 --ckpt runs/genrel_organ.pt
"""
from __future__ import annotations

import argparse, json, os, time
import itertools as it
import numpy as np
import torch

from . import csp as C
from . import run_general as RG
from . import big_organ as BO            # importing BO sets RG's budget to N=25,D=6,M=160,A=3

# adopt big_organ's larger budget (already pushed into RG by the import above)
N_MAX, D_MAX, M_MAX, A_MAX = BO.N_MAX, BO.D_MAX, BO.M_MAX, BO.A_MAX

from .blade_deductor import BladeFactorDeductor, size_for   # noqa: E402


# ============================================================ randomized-relation generator
def gen_random_relation(rng, n_lo=3, n_hi=7, d_lo=2, d_hi=6):
    """Witness-first CSP whose constraints are RANDOM allowed-tuple relations (arity 1/2/3).

    Plant a witness assignment, then emit several constraints, each a random scope + a random
    allowed-tuple TABLE that is forced to contain the witness's projection (so the whole CSP is
    guaranteed solvable). Each non-witness tuple is admitted i.i.d. with a per-constraint density
    p ~ U(0.1,0.6): a huge diversity of relation tables (tight↔loose, any arity) with NO algebraic
    family identity — the organ must learn the general narrowing-given-a-table operation. A random
    subset of cells is revealed as exact pins (sharpens narrowing + matches the eval distribution,
    where n_queens/futoshiki/latin all carry pins)."""
    n = int(rng.integers(n_lo, n_hi + 1))
    d = int(rng.integers(d_lo, d_hi + 1))
    witness = [int(rng.integers(0, d)) for _ in range(n)]
    cons = []
    ncons = int(rng.integers(n, 3 * n + 1))
    for _ in range(ncons):
        a = min(int(rng.choice([1, 2, 3], p=[0.15, 0.5, 0.35])), n)
        scope = tuple(int(x) for x in rng.choice(n, size=a, replace=False))
        wt = tuple(witness[c] for c in scope)
        p = float(rng.uniform(0.1, 0.6))
        allowed = {wt}
        for tup in it.product(range(d), repeat=a):
            if tup != wt and rng.random() < p:
                allowed.add(tup)
        cons.append((scope, frozenset(allowed)))
    # reveal a random subset of cells as exact pins
    nrev = int(rng.integers(0, n))
    if nrev:
        for c in rng.choice(n, size=nrev, replace=False):
            c = int(c)
            cons.append(((c,), frozenset({(witness[c],)})))
    return C.CSP(n, d, tuple(cons))


def sample_random(rng):
    for _ in range(60):
        c = gen_random_relation(rng)
        if BO._within_budget(c):
            return c
    return gen_random_relation(rng, n_lo=3, n_hi=3, d_lo=2, d_hi=3)


def make_combo_sampler(p_random=0.5):
    """Union curriculum: with prob p_random draw a RANDOM relation table, else a NAMED family
    (round-robin over big_organ TRAIN_RUNGS). Tests whether covering BOTH manifolds keeps soundness
    on the structured families AND gains it on arbitrary random tables."""
    state = {"i": 0}

    def _s(rng):
        if rng.random() < p_random:
            return sample_random(rng)
        rg = BO.TRAIN_RUNGS[state["i"] % len(BO.TRAIN_RUNGS)]
        state["i"] += 1
        return BO.sample(rng, rg)
    return _s


# ============================================================ generic trainer (any sampler)
def train_sampler(sampler, dev, d, steps, pool=64, R=12, lr=3e-4, theta=0.5, seed=0, tag=""):
    """Train a blade organ (dominate-dedP) on-policy, drawing fresh instances from `sampler(rng)`.
    Identical machinery to big_organ.train; only the instance distribution differs."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = BladeFactorDeductor("blade", N_MAX, D_MAX, M_MAX, A_MAX, d=d, R=R, ds=min(4, R)).to(dev)
    npar = m.n_params()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95))
    items = []
    while len(items) < pool:
        c = sampler(rng); items.append((c, c.full()))
    print(f"  [{tag}] d={d} params={npar:,} pool={pool} R={R} steps={steps} "
          f"budget N={N_MAX} D={D_MAX} M={M_MAX}", flush=True)
    fe_k = fe_n = 0; t0 = time.time(); log = []
    for s in range(1, steps + 1):
        feat = RG.featurize(items, dev); vm = feat["var_mask"]
        tgt, conflict = RG.build_targets(items, RG.SHARED, dev)
        b, cls, sup = RG.fwd(m, feat, vm)
        loss = RG.loss_fn(sup, vm, feat["var_valid"], tgt, conflict)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        with torch.no_grad():
            new_vm = RG.meet(vm, b, theta)
            k, nn_ = RG.false_elim(vm, new_vm, feat["var_valid"], tgt); fe_k += k; fe_n += nn_
            nvc = new_vm.cpu().numpy(); nxt = []
            for bi, (csp, dom) in enumerate(items):
                ndom = RG.dom_from_mask(nvc[bi], csp)
                if C.status(ndom) in ("solved", "conflict") or ndom == dom:
                    nc = sampler(rng); nxt.append((nc, nc.full()))
                else:
                    nxt.append((csp, ndom))
            items = nxt
        if s % max(1, steps // 15) == 0:
            fer = fe_k / max(1, fe_n)
            log.append({"step": s, "loss": float(loss.detach()), "false_elim": fer})
            print(f"    [{tag}] step {s:5d}  loss {float(loss.detach()):.3f}  false_elim {fer:.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            fe_k = fe_n = 0
    return m, npar, log


# ============================================================ eval sets
# every NAMED family is zero-shot for the random-relation organ; the family organ trains on the
# big_organ TRAIN_RUNGS and sees n_queens/futoshiki/latin5 as OOD (reproducing the 0.004 break).
NAMED_RUNGS = ["coloring", "arithmetic", "ordering", "latin", "sudoku4", "alldiff_big",
               "n_queens", "futoshiki", "latin5"]


def make_named_eval(neval, seed=4321):
    rng = np.random.default_rng(seed)
    out = {}
    for rg in NAMED_RUNGS:
        gen = BO.GEN.get(rg) or BO.OOD_RUNGS[rg]
        cs = []
        tries = 0
        while len(cs) < neval and tries < neval * 80:
            tries += 1
            csp = gen(rng)
            if BO._within_budget(csp):
                cs.append(csp)
        out[rg] = cs
    return out


def make_random_eval(neval, seed=99991):
    rng = np.random.default_rng(seed)        # held-out seed: fresh random relations never trained
    return {"random_heldout": [sample_random(rng) for _ in range(neval)]}


# ============================================================ orchestration
def _evalset(m, sets, dev, theta, rmax):
    return {rg: BO.evaluate_big(m, cs, dev, theta, rmax) for rg, cs in sets.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--combo", action="store_true", help="train ONE organ on named+random UNION; eval both")
    ap.add_argument("--R", type=int, default=12)
    ap.add_argument("--target", type=float, default=4e5)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--neval", type=int, default=120)
    ap.add_argument("--rmax", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="runs/genrel_organ.pt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = RG.device()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    print(f"device={dev}  budget N={N_MAX} D={D_MAX} M={M_MAX} A={A_MAX}", flush=True)

    if args.smoke:
        rng = np.random.default_rng(0)
        print("=== SMOKE: random-relation generator within budget + dedP ground truth ===", flush=True)
        ar_hist = {1: 0, 2: 0, 3: 0}
        for _ in range(200):
            c = sample_random(rng)
            for sc, _ in c.cons:
                ar_hist[len(sc)] = ar_hist.get(len(sc), 0) + 1
        print(f"  arity histogram over 200 random CSPs: {ar_hist}", flush=True)
        for _ in range(5):
            t0 = time.time(); c = sample_random(rng)
            ded, _ = C.to_fixpoint(C.exact_dedP, c, c.full())
            elim = sum(c.d - len(ded[i]) for i in range(c.n))
            print(f"  random n={c.n} d={c.d} m={len(c.cons):2d}  dedP-elim={elim:2d} "
                  f"status={C.status(ded)} ({(time.time()-t0)*1000:.0f}ms)", flush=True)
        d, npar = size_for("blade", N_MAX, D_MAX, M_MAX, A_MAX, args.target, R=args.R, ds=4)
        print(f"  size_for(blade,{args.target:.0e}) -> d={d} params={npar:,}", flush=True)
        m, npar, log = train_sampler(sample_random, dev, d, steps=150, pool=48, R=args.R,
                                     seed=0, tag="rand")
        print(f"  false_elim traj: {[round(x['false_elim'],4) for x in log]}", flush=True)
        ev = _evalset(m, make_random_eval(40), dev, args.theta, args.rmax)
        ev.update(_evalset(m, {k: make_named_eval(40)[k] for k in ("n_queens", "futoshiki")},
                           dev, args.theta, args.rmax))
        for rg in ("random_heldout", "n_queens", "futoshiki"):
            print(RG._row(rg, ev[rg]), flush=True)
        return

    if args.combo:
        d, npar = size_for("blade", N_MAX, D_MAX, M_MAX, A_MAX, args.target, R=args.R, ds=4)
        named = make_named_eval(args.neval)
        rand_ev = make_random_eval(args.neval)
        print("\n##### TRAIN C: UNION curriculum (named families + random tables) #####", flush=True)
        mC, parC, logC = train_sampler(make_combo_sampler(0.5), dev, d, args.steps, pool=args.pool,
                                       R=args.R, lr=args.lr, theta=args.theta, seed=args.seed,
                                       tag="combo")
        print("\n=== UNION organ (C) — held-out random + all named families ===", flush=True)
        evC = _evalset(mC, rand_ev, dev, args.theta, args.rmax)
        evC.update(_evalset(mC, named, dev, args.theta, args.rmax))
        for rg in ["random_heldout"] + NAMED_RUNGS:
            print(RG._row(rg, evC[rg]), flush=True)
        os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
        torch.save({"state_dict": mC.state_dict(),
                    "config": {"msg": "blade", "n_max": N_MAX, "d_max": D_MAX, "m_max": M_MAX,
                               "a_max": A_MAX, "d": d, "R": args.R, "ds": min(4, args.R)},
                    "params": parC, "train": "union(named+random)", "evalC": evC, "logC": logC,
                    "args": vars(args)}, args.ckpt)
        print(f"\nsaved UNION organ -> {args.ckpt} ({parC:,} params)", flush=True)
        if args.out:
            json.dump({"evalC": evC, "logC": logC, "params": parC}, open(args.out, "w"),
                      indent=1, default=str)
            print("wrote", args.out, flush=True)
        return

    if args.full:
        d, npar = size_for("blade", N_MAX, D_MAX, M_MAX, A_MAX, args.target, R=args.R, ds=4)
        named = make_named_eval(args.neval)
        rand_ev = make_random_eval(args.neval)

        print("\n##### TRAIN A: randomized-relation organ #####", flush=True)
        mA, parA, logA = train_sampler(sample_random, dev, d, args.steps, pool=args.pool,
                                       R=args.R, lr=args.lr, theta=args.theta, seed=args.seed,
                                       tag="rand")

        print("\n##### TRAIN B: family-trained baseline (big_organ TRAIN_RUNGS) #####", flush=True)
        mB, parB, logB = BO.train(BO.TRAIN_RUNGS, dev, d, args.steps, pool=args.pool, R=args.R,
                                  lr=args.lr, theta=args.theta, seed=args.seed)

        print("\n=== RANDOMIZED-RELATION organ (A) — held-out random + all named families ZERO-SHOT ===",
              flush=True)
        evA = _evalset(mA, rand_ev, dev, args.theta, args.rmax)
        evA.update(_evalset(mA, named, dev, args.theta, args.rmax))
        for rg in ["random_heldout"] + NAMED_RUNGS:
            print(RG._row(rg, evA[rg]), flush=True)

        print("\n=== FAMILY-TRAINED organ (B) — same eval sets (named families: in-dist vs OOD) ===",
              flush=True)
        evB = _evalset(mB, rand_ev, dev, args.theta, args.rmax)
        evB.update(_evalset(mB, named, dev, args.theta, args.rmax))
        for rg in ["random_heldout"] + NAMED_RUNGS:
            print(RG._row(rg, evB[rg]), flush=True)

        print("\n=== HEAD-TO-HEAD: soundness (FALSE-ELIM) + recall, randomized-trained (A) vs family (B) ===",
              flush=True)
        print(f"  {'rung':14s} {'A-FE':>9s} {'B-FE':>9s} | {'A-recall':>9s} {'B-recall':>9s} | "
              f"{'A-wrong':>8s} {'B-wrong':>8s}", flush=True)
        for rg in ["random_heldout"] + NAMED_RUNGS:
            a, b = evA[rg], evB[rg]
            print(f"  {rg:14s} {a['false_elim']:9.4f} {b['false_elim']:9.4f} | "
                  f"{a['narrowing_recall']*100:8.1f}% {b['narrowing_recall']*100:8.1f}% | "
                  f"{a['wrong_return_rate']*100:7.1f}% {b['wrong_return_rate']*100:7.1f}%", flush=True)

        nq_a = evA["n_queens"]["false_elim"]; nq_b = evB["n_queens"]["false_elim"]
        print(f"\n  >>> n_queens false-elim: randomized-trained={nq_a:.4f}  family-trained={nq_b:.4f}  "
              f"({'FIXED' if nq_a <= nq_b * 0.5 + 1e-9 else 'NOT fixed'})", flush=True)

        os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
        torch.save({"state_dict": mA.state_dict(),
                    "config": {"msg": "blade", "n_max": N_MAX, "d_max": D_MAX, "m_max": M_MAX,
                               "a_max": A_MAX, "d": d, "R": args.R, "ds": min(4, args.R)},
                    "params": parA, "train": "randomized-relation",
                    "evalA": evA, "evalB": evB, "logA": logA, "logB": logB,
                    "args": vars(args)}, args.ckpt)
        print(f"\nsaved randomized-relation organ -> {args.ckpt} ({parA:,} params)", flush=True)
        if args.out:
            json.dump({"evalA": evA, "evalB": evB, "logA": logA, "logB": logB,
                       "params": [parA, parB]}, open(args.out, "w"), indent=1, default=str)
            print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
