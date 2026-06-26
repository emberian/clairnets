"""clair/perm_organ.py — the PERMUTATION-GROUP organ: a learned proposer that narrows the set of
permutations consistent with a SUBGROUP-membership constraint, soundly, where the certified local
fixpoint abstains.

WHY the subgroup family. The exact harness (`clair.permgroup`) shows a clean decomposition of where
completeness comes from:
  * pure S_n (AllDifferent + unary): the certified Régin/Hall matching GAC is ALREADY exact (== dedₚ).
    The symmetric-group structure is solved soundly AND completely by the certified primitive — nothing
    to learn.
  * ordering: bounds-consistency is nearly complete.
  * SUBGROUP  σ ∈ ⟨gens⟩ ≤ S_n: the certified LOCAL narrowing (orbit bound + AllDiff GAC) leaves a
    large ABSTAIN GAP — local lattice propagation cannot see the global group correlation. The exact
    answer needs the GLOBAL group algorithm (Schreier-Sims). This is the Krohn-Rhodes locus: a subgroup
    of S_n is exactly the transition group of a permutation automaton, and its membership structure is
    the part that does NOT reduce to local consistency.

THE ORGAN. STATE = per-(point,value) survival belief alive[B,n,n] ∈ [0,1], MONOTONE meet (only narrows).
Input features per cell (i,v): current alive, the unary-allowed mask A[i,v] (certified-free), the
exactly-computable GROUP PRIOR P[i,v] = frac of g∈G with g(i)=v, the orbit bound 1[v∈orbit_G(i)], and a
`given` flag. The model message-passes over the AllDifferent grid (row = one value per point, col = one
point per value) plus a channel mixer, and emits per-cell keep-logits. The certified unary meet is
applied for free. SOUNDNESS is the research variable: trained with a soundness-ASYMMETRIC loss to
DOMINATE the exact dedₚ — heavy penalty for dropping a value dedₚ keeps (false-elim), light penalty for
keeping one dedₚ drops (incompleteness) — so a sound organ never false-eliminates while closing the gap.

GROUND TRUTH = clair.permgroup.exact_dedP (brute over the subgroup ∩ constraints; sympy/brute checked).

Run:  python -m clair.perm_organ --smoke      # quick: organ trains, false-elim -> ~0, gap closes
      python -m clair.perm_organ              # full: soundness + completeness table vs certified + dedP
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import permgroup as pg


# --------------------------------------------------------------------------- problem generation
def rand_subgroup_problem(rng: random.Random, n_lo=4, n_hi=6, gap_only=False, tries=40):
    """Sample a solvable σ∈⟨gens⟩ problem with a couple of unary clues. If gap_only, keep resampling
    until the certified local fixpoint ABSTAINS below dedₚ (the cases the organ must earn)."""
    for _ in range(tries):
        n = rng.randint(n_lo, n_hi)
        full = tuple(frozenset(range(n)) for _ in range(n))
        gens = []
        for _ in range(rng.randint(1, 2)):
            p = list(range(n)); rng.shuffle(p); gens.append(tuple(p))
        un = list(full)
        for _ in range(rng.randint(1, 2)):           # clues: fix or forbid an image
            i = rng.randrange(n)
            if rng.random() < 0.6:
                un[i] = frozenset({rng.randrange(n)})            # fix σ(i)
            else:
                un[i] = frozenset(v for v in range(n) if v != rng.randrange(n))   # forbid one image
        prob = pg.PermProblem(n, tuple(un), gens=tuple(gens))
        sols = pg.brute(prob, limit=1)
        if not sols:
            continue
        if gap_only:
            dom, _ = pg.to_fixpoint(pg.certified_step, prob)
            ex = pg.exact_dedP(prob)
            if dom == ex:                              # certified already complete -> not interesting
                continue
        return prob
    return prob


def featurize(prob: pg.PermProblem):
    """Per-(point,value) input features + the dedₚ target + masks, for one problem."""
    n = prob.n
    A = np.zeros((n, n), np.float32)                 # unary allowed
    for i in range(n):
        for v in prob.unary[i]:
            A[i, v] = 1.0
    given = np.array([1.0 if len(prob.unary[i]) == 1 else 0.0 for i in range(n)], np.float32)
    P = np.array(pg.group_prior(prob.gens, n), np.float32)            # group marginal
    orbit = (P > 0).astype(np.float32)                                # certified orbit bound
    # pairwise group realizability Q[i,v,j,w]=1 iff (v,w) jointly realizable in G (level-1 group structure)
    Q = np.zeros((n, n, n, n), np.float32)
    if prob.gens:
        Gset = pg.close_group(prob.gens, n)
        for g in Gset:
            for i in range(n):
                for j in range(n):
                    Q[i, g[i], j, g[j]] = 1.0
    else:
        Q[:] = 1.0
    ex = pg.exact_dedP(prob)
    target = np.zeros((n, n), np.float32)            # 1 if v survives in dedₚ
    for i in range(n):
        for v in ex[i]:
            target[i, v] = 1.0
    cert, _ = pg.to_fixpoint(pg.certified_step, prob)
    cert_alive = np.zeros((n, n), np.float32)
    for i in range(n):
        for v in cert[i]:
            cert_alive[i, v] = 1.0
    solvable = float(any(len(c) > 0 for c in ex) and all(len(c) > 0 for c in ex))
    return {"n": n, "A": A, "given": given, "P": P, "orbit": orbit, "Q": Q,
            "target": target, "cert": cert_alive, "solvable": solvable}


def collate(items, device, n_max):
    B = len(items)
    A = np.zeros((B, n_max, n_max), np.float32)
    P = np.zeros((B, n_max, n_max), np.float32)
    orbit = np.zeros((B, n_max, n_max), np.float32)
    given = np.zeros((B, n_max), np.float32)
    target = np.zeros((B, n_max, n_max), np.float32)
    cert = np.zeros((B, n_max, n_max), np.float32)
    Q = np.zeros((B, n_max, n_max, n_max, n_max), np.float32)
    valid = np.zeros((B, n_max, n_max), np.float32)      # real (i,v) cells
    pvalid = np.zeros((B, n_max), np.float32)            # real points
    solvable = np.zeros(B, np.float32)
    for b, it in enumerate(items):
        n = it["n"]
        A[b, :n, :n] = it["A"]; P[b, :n, :n] = it["P"]; orbit[b, :n, :n] = it["orbit"]
        given[b, :n] = it["given"]; target[b, :n, :n] = it["target"]; cert[b, :n, :n] = it["cert"]
        Q[b, :n, :n, :n, :n] = it["Q"]
        valid[b, :n, :n] = 1.0; pvalid[b, :n] = 1.0; solvable[b] = it["solvable"]
    t = lambda a: torch.as_tensor(a, device=device)
    return {"A": t(A), "P": t(P), "orbit": t(orbit), "given": t(given), "target": t(target),
            "cert": t(cert), "Q": t(Q), "valid": t(valid), "pvalid": t(pvalid), "solvable": t(solvable)}


# --------------------------------------------------------------------------- the organ
class SwiGLU(nn.Module):
    def __init__(self, d, ratio=2.0):
        super().__init__()
        h = int(d * ratio)
        self.w1 = nn.Linear(d, h, bias=False); self.w2 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class PermOrgan(nn.Module):
    """Message-passing proposer over the AllDifferent assignment grid (points × values), conditioned on
    the exactly-computable group prior. Permutation-equivariant over both points and values (no absolute
    index embeddings) — the constraints A/P/orbit are the only asymmetry, so the same weights run on any
    (n, gens). Emits per-cell keep-logits (monotone meet) + a pooled conflict logit."""
    def __init__(self, d=64, R=6, fin=5):
        super().__init__()
        self.d, self.R = d, R
        self.cell_in = nn.Linear(fin, d)
        self.row_ln = nn.LayerNorm(d); self.col_ln = nn.LayerNorm(d); self.mix_ln = nn.LayerNorm(d)
        self.gproj = nn.Linear(1, d)                        # inject the group-pair support signal
        # context combiner: [self, row(point) ctx, col(value) ctx] -> update
        self.upd = nn.Sequential(nn.Linear(3 * d, d), nn.SiLU(), nn.Linear(d, d))
        self.mixer = SwiGLU(d)
        self.head = nn.Linear(d, 1)
        self.cls = nn.Linear(d, 1)

    def forward(self, ba):
        A, P, orbit, given, valid = ba["A"], ba["P"], ba["orbit"], ba["given"], ba["valid"]
        Q = ba["Q"]                                            # [B,n,n,n,n] pairwise group realizability
        B, n, _ = A.shape
        alive = valid.clone()                                  # start: full grid (masked to real cells)
        gv = given.unsqueeze(-1).expand(-1, -1, n)             # broadcast given flag over values
        feats = torch.stack([alive, A, P, orbit, gv], dim=-1)  # [B,n,n,fin]
        h = self.cell_in(feats) * valid.unsqueeze(-1)
        vmask = valid.unsqueeze(-1)
        pden = valid.sum(2, keepdim=True).clamp(min=1.0).unsqueeze(-1)   # # values per point
        vden = valid.sum(1, keepdim=True).clamp(min=1.0).unsqueeze(-1)   # # points per value
        eye = torch.eye(n, device=A.device).view(1, n, 1, n, 1)         # to mask j==i in the group message
        sup = []
        belief = (A * valid)                                   # current keep-belief b[j,w] (start: allowed)
        for r in range(self.R):
            # ---- dynamic group-pair support: gsupp[i,v] = mean_{j≠i} max_w Q[i,v,j,w]·belief[j,w] ----
            #   (differentiable group path-consistency: does some still-alive image of j co-occur with v in G?)
            supp = (Q * belief.view(B, 1, 1, n, n)).amax(dim=4)          # [B,n,n,n] over w
            supp = supp * (1.0 - eye.squeeze(-1))                        # drop j==i
            gsupp = supp.sum(3) / (n - 1 if n > 1 else 1)               # [B,n,n] mean over j
            h = h + self.gproj(gsupp.unsqueeze(-1)) * vmask
            row_ctx = self.row_ln((h * vmask).sum(2, keepdim=True) / pden).expand(-1, -1, n, -1)  # point ctx
            col_ctx = self.col_ln((h * vmask).sum(1, keepdim=True) / vden).expand(-1, n, -1, -1)  # value ctx
            h = h + self.upd(torch.cat([h, row_ctx, col_ctx], dim=-1)) * vmask
            h = h + self.mixer(self.mix_ln(h)) * vmask
            logit = self.head(h).squeeze(-1)                   # [B,n,n]
            keep = torch.sigmoid(logit)
            # MONOTONE meet with the certified-free unary mask; pad cells forced to 0
            alive_out = keep * A * valid
            belief = alive_out                                 # feed the narrowed belief into next round
            sup.append(alive_out)
        # pooled conflict logit
        pooled = (h * vmask).sum((1, 2)) / valid.sum((1, 2)).clamp(min=1.0).unsqueeze(-1)
        cls = self.cls(pooled).squeeze(-1)
        return sup[-1], cls, sup


# --------------------------------------------------------------------------- loss (soundness-asymmetric)
def organ_loss(model, ba, w_unsound=6.0, w_incomplete=1.0, w_cls=0.5):
    final, cls, sup = model(ba)
    valid, tgt = ba["valid"], ba["target"]
    surv = valid * tgt                          # cells dedₚ KEEPS (must not drop) -> unsound if alive low
    dead = valid * (1.0 - tgt)                  # cells dedₚ DROPS -> incomplete if alive high
    l_uns = l_inc = 0.0
    for a in sup[-2:]:                           # deep supervision on the last rounds
        a = a.clamp(1e-6, 1 - 1e-6)
        l_uns = l_uns - (surv * torch.log(a)).sum() / surv.sum().clamp(min=1.0)        # push survivors->1
        l_inc = l_inc - (dead * torch.log(1 - a)).sum() / dead.sum().clamp(min=1.0)    # push non-surv->0
    k = min(2, len(sup))
    l_uns, l_inc = l_uns / k, l_inc / k
    l_cls = F.binary_cross_entropy_with_logits(cls, 1.0 - ba["solvable"])              # predict UNSAT
    loss = w_unsound * l_uns + w_incomplete * l_inc + w_cls * l_cls
    return loss, {"unsound": float(l_uns), "incomplete": float(l_inc), "cls": float(l_cls)}


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, items, device, n_max, bs=128, thr=0.5):
    model.eval()
    fe = 0                       # false-elim vs dedₚ (soundness, want 0)
    exact_match = 0              # organ alive == dedₚ exactly (full completeness)
    solved = 0                   # organ drives to a unique permutation
    gap_closed_num = gap_closed_den = 0     # of the cells certified keeps-but-dedₚ-kills, how many organ kills
    cert_solved = ded_solved = tot = 0
    cert_fe = 0
    for s in range(0, len(items), bs):
        batch = items[s:s + bs]
        ba = collate(batch, device, n_max)
        final, cls, _ = model(ba)
        alive = ((final > thr) & (ba["valid"] > 0.5)).float()        # hard organ narrowing
        tgt = ba["target"]; valid = ba["valid"]; cert = ba["cert"]
        for b in range(len(batch)):
            n = batch[b]["n"]
            al = alive[b, :n, :n]; tg = tgt[b, :n, :n]; ce = cert[b, :n, :n]
            tot += 1
            fe += int(((tg > 0.5) & (al < 0.5)).sum())               # dropped a true survivor
            cert_fe += int(((tg > 0.5) & (ce < 0.5)).sum())          # certified is sound by construction (0)
            exact_match += int(torch.equal(al, tg))
            # solved: each point exactly one alive value AND each value once
            rows = al.sum(1); cols = al.sum(0)
            org_solved = bool((rows == 1).all() and (cols == 1).all())
            solved += int(org_solved and torch.equal(al, tg))
            cert_rows = ce.sum(1); cert_cols = ce.sum(0)
            cert_solved += int(bool((cert_rows == 1).all() and (cert_cols == 1).all()))
            ded_rows = tg.sum(1); ded_cols = tg.sum(0)
            ded_solved += int(bool((ded_rows == 1).all() and (ded_cols == 1).all()))
            extra = (ce > 0.5) & (tg < 0.5)                          # cells certified keeps but dedₚ kills
            gap_closed_den += int(extra.sum())
            gap_closed_num += int((extra & (al < 0.5)).sum())        # organ correctly kills them
    return {"n": tot, "false_elim": fe, "cert_false_elim": cert_fe,
            "exact_match_rate": exact_match / max(1, tot),
            "organ_solved": solved, "cert_solved": cert_solved, "ded_solved": ded_solved,
            "gap_cells": gap_closed_den, "gap_closed": gap_closed_num,
            "gap_closed_rate": gap_closed_num / max(1, gap_closed_den)}


# --------------------------------------------------------------------------- data + train
def gen_items(rng, k, n_lo, n_hi, gap_only=False):
    return [featurize(rand_subgroup_problem(rng, n_lo, n_hi, gap_only=gap_only)) for _ in range(k)]


def train(model, device, steps, rng, n_max, n_lo, n_hi, bs=64, lr=2e-3, gap_frac=0.5,
          log_every=100, quiet=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log = []
    pool = gen_items(rng, max(bs * 6, 384), n_lo, n_hi, gap_only=False)
    gap_pool = gen_items(rng, max(bs * 4, 256), n_lo, n_hi, gap_only=True)   # upweight the hard gap cases
    for s in range(1, steps + 1):
        model.train()
        if s % 250 == 0:
            pool = gen_items(rng, max(bs * 6, 384), n_lo, n_hi, gap_only=False)
            gap_pool = gen_items(rng, max(bs * 4, 256), n_lo, n_hi, gap_only=True)
        ng = int(bs * gap_frac)
        batch = [gap_pool[i] for i in rng.sample(range(len(gap_pool)), ng)] + \
                [pool[i] for i in rng.sample(range(len(pool)), bs - ng)]
        ba = collate(batch, device, n_max)
        loss, parts = organ_loss(model, ba)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if not quiet and (s % log_every == 0 or s == 1):
            ev = evaluate(model, gap_pool[:256], device, n_max)
            row = {"step": s, "loss": float(loss), **parts, "fe": ev["false_elim"],
                   "exact": ev["exact_match_rate"], "gap_closed": ev["gap_closed_rate"]}
            log.append(row)
            print(f"  step {s:5d} loss {loss:6.3f} (uns {parts['unsound']:.3f} inc {parts['incomplete']:.3f}) "
                  f"false-elim {ev['false_elim']:4d}  exact-match {ev['exact_match_rate']*100:5.1f}%  "
                  f"gap-closed {ev['gap_closed_rate']*100:5.1f}%", flush=True)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--n_lo", type=int, default=4)
    ap.add_argument("--n_hi", type=int, default=6)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--R", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="runs/perm_organ")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.bs = 500, 48

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(args.seed)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    n_max = args.n_hi
    print(f"device={dev}  n∈[{args.n_lo},{args.n_hi}]  d={args.d} R={args.R}  task=subgroup-membership narrowing")

    model = PermOrgan(d=args.d, R=args.R).to(dev)
    print(f"perm-organ params: {sum(p.numel() for p in model.parameters())}")

    t0 = time.time()
    log = train(model, dev, args.steps, rng, n_max, args.n_lo, args.n_hi, bs=args.bs,
                quiet=False)
    print(f"train wall: {time.time()-t0:.1f}s")

    print("\nbuilding eval sets ...", flush=True)
    n_eval = 256 if args.smoke else 1000
    all_set = gen_items(rng, n_eval, args.n_lo, args.n_hi, gap_only=False)
    gap_set = gen_items(rng, n_eval, args.n_lo, args.n_hi, gap_only=True)
    ood_set = gen_items(rng, n_eval // 2, args.n_hi + 1, args.n_hi + 1, gap_only=True)  # OOD larger n

    ev_all = evaluate(model, all_set, dev, n_max)
    ev_gap = evaluate(model, gap_set, dev, n_max)
    ev_ood = evaluate(model, ood_set, dev, args.n_hi + 1)

    print("\n================ SOUNDNESS / COMPLETENESS vs CERTIFIED + EXACT dedₚ ================")
    def show(name, ev):
        print(f"\n[{name}]  n={ev['n']}")
        print(f"  SOUNDNESS  organ false-elim vs dedₚ : {ev['false_elim']:5d}  (certified: {ev['cert_false_elim']}; both want 0)")
        print(f"  COMPLETE   organ==dedₚ exactly       : {ev['exact_match_rate']*100:6.2f}%")
        print(f"  SOLVED     certified={ev['cert_solved']:4d}  organ={ev['organ_solved']:4d}  dedₚ(max)={ev['ded_solved']:4d}")
        print(f"  GAP-CLOSED organ kills {ev['gap_closed']}/{ev['gap_cells']} of the certified-abstain cells "
              f"({ev['gap_closed_rate']*100:.1f}%)")
    show("ALL solvable subgroup problems", ev_all)
    show("GAP-ONLY (certified abstains)", ev_gap)
    show(f"OOD larger n={args.n_hi+1} (gap-only)", ev_ood)

    out = Path(args.out + ("_smoke" if args.smoke else "") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "train_log": log,
                               "all": ev_all, "gap": ev_gap, "ood": ev_ood}, indent=2))
    ckpt = Path(args.out + ("_smoke" if args.smoke else "") + ".pt")
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, ckpt)
    print(f"\nwrote {out} and {ckpt}")


if __name__ == "__main__":
    main()
