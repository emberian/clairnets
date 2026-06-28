"""clair/organ/run_block_scaling.py — THE ORGAN-AS-A-REPEATABLE-BLOCK SCALING STUDY.

Ember's question: "if one organ is good, two with flows is wilder — how does it SCALE?"

The foundation organ (runs/general_organ_full.pt) is a weight-tied recurrent factor-graph narrower
(R rounds of variable<->factor message passing, trained at R=12). This study asks three scaling
questions, all on the HARD difficulty tail (deep propagation, high treewidth, affine wall), measuring
NARROWING-RECALL (fraction of the exact-dedP eliminations the operator also makes) + FALSE-ELIM
(soundness) per difficulty bucket:

  PART 1  RECURRENCE-DEPTH SCALING (the R-sweep / compute=depth law).
          At EVAL, sweep the number of message-passing rounds R in {1,2,4,8,12,16,24,32} on the
          foundation organ (just run the weight-tied recurrence longer/shorter — a single forward of
          R rounds = R hops of propagation). Does recall rise monotonically with R, where does it
          SATURATE, and does the saturation-R track the problem's propagation DEPTH (a depth-d chain
          needs ~d rounds)?  Plotted as recall-vs-R per depth bucket.

  PART 2  STACKED HETEROGENEOUS ORGANS ("two organs with flows").
          Compose a PIPELINE of bank faculties (neural CoreNarrow <-> certified GF2/Modular) over a
          shared lattice via the verifier-gated reduced product. On affine+propagation MIXED-hard
          instances (neither a single neural narrower NOR a single certified affine op can finish),
          does the STACK solve what no single faculty can?  single-organ recall vs stacked recall.

  PART 3  GATED ADAPTIVE-DEPTH BLOCK (nested reasoning).
          Wrap the organ-step as a block applied N times with a HALT gate = the EXACT lattice fixpoint
          (we KNOW when to stop, cleanly — ACT/Universal-Transformer with a sound halt). Does adaptive
          depth match fixed-max-depth recall at a fraction of the compute, and does the halt-iteration
          count track problem depth (compute=depth again, at the outer loop)?

Writes runs/organ_block_scaling.json. Pure organ + exact clair.csp oracle (no OLMo).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import torch

from .. import csp as C
from .. import xor_wall as XW
from .. import hard_tasks as HT
from . import bank as B
from .. import run_glados_staged as G
from .protocol import CSPState
from .compose import reduced_product

CHAIN_FAMILIES = ("eqchain", "forcedcolor", "orderchain", "arithchain")


# ============================================================ exact-oracle recall helpers
def _removed(full, dom, n):
    return {(i, v) for i in range(n) for v in full[i] if v not in dom[i]}


def dedp_fixpoint(csp):
    """Per-cell domains at the EXACT dedP fixpoint = the strongest sound per-cell ceiling (recall den)."""
    fix, _ = C.to_fixpoint(C.exact_dedP, csp, csp.full())
    return fix


def beyond_factor(csp):
    """True iff the arity-3 FACTOR lattice (level-2, what FactorConsistency(k=3) reaches) is INCOMPLETE
    for `csp` AND the exact dedP actually forces something (nonzero recall denominator). I.e. the
    instance is a genuine level>=3 wall: per-cell AC + pair + factor-3 all fall short of exact, so only
    the row-space (GF2) op can finish it. Cheap (exact_dedP + one factor fixpoint; skips the pair check)."""
    from ..datagen import difficulty as DF
    full = csp.full()
    exact = C.exact_dedP(csp, full)
    if all(len(c) == csp.d for c in exact):          # nothing forced => useless (no recall denom)
        return False
    return DF._factor_cells(csp, 3) != exact


def recall_fe(csp, model_dom, ded_fix=None):
    """(recall, false_elim_cells, n_elim_target, solved) for one instance vs the dedP fixpoint."""
    full = csp.full()
    ded = ded_fix if ded_fix is not None else dedp_fixpoint(csp)
    rm_model = _removed(full, model_dom, csp.n)
    rm_ded = _removed(full, ded, csp.n)
    inter = len(rm_model & rm_ded)
    fe = len(rm_model - rm_ded)                       # eliminations the oracle does NOT make => unsound
    solved = all(len(c) == 1 for c in model_dom)
    return inter, fe, len(rm_ded), int(solved)


# ============================================================ organ inference (variable inner R)
@torch.no_grad()
def organ_singlepass(organ, csps, R, dev, theta=0.5):
    """ONE forward of `R` weight-tied message-passing rounds from the FULL domain, meet once, read the
    per-cell domain. Total propagation budget = R hops. Returns list of model domains (one per csp)."""
    organ.R = int(R)
    organ.ds = max(1, min(getattr(organ, "ds", 4), int(R)))
    items = [(c, c.full()) for c in csps]
    feat = G.featurize(items, dev)
    vm = feat["var_mask"]
    b, _cls, _sup = G.fwd(organ, feat, vm)
    new_vm = G.meet(vm, b, theta)
    nvc = new_vm.cpu().numpy()
    return [G.dom_from_mask(nvc[i], csps[i]) for i in range(len(csps))]


@torch.no_grad()
def organ_fixpoint(organ, csps, R_inner, dev, theta=0.5, max_outer=64, fixed_outer=None):
    """Deployed OUTER loop: repeatedly run a forward of R_inner rounds + meet, re-featurizing `given`
    each pass, until the lattice stops changing (the EXACT fixpoint HALT) or max_outer. Returns
    (model_doms, halt_iter_per_instance). If fixed_outer is set, runs EXACTLY that many passes (no
    early halt) — the fixed-depth baseline for Part 3."""
    organ.R = int(R_inner)
    organ.ds = max(1, min(getattr(organ, "ds", 4), int(R_inner)))
    items = [(c, c.full()) for c in csps]
    feat = G.featurize(items, dev)
    vm = feat["var_mask"].clone()
    Bn = len(csps)
    done = torch.zeros(Bn, dtype=torch.bool, device=dev)
    halt = np.zeros(Bn, dtype=np.int64)
    budget = fixed_outer if fixed_outer is not None else max_outer
    for it in range(budget):
        feat["given"] = (vm.sum(-1) == 1).float() * feat["var_valid"]
        b, _cls, _sup = G.fwd(organ, feat, vm)
        new_vm = G.meet(vm, b, theta)
        changed = (new_vm != vm).any(-1).any(-1)
        if fixed_outer is None:
            vm = torch.where(done.view(-1, 1, 1), vm, new_vm)
            halt[(~done & ~changed).cpu().numpy()] = it + 1     # this instance just reached fixpoint
            done = done | ~changed
            if done.all():
                break
        else:
            vm = new_vm
            halt[:] = it + 1
    halt[halt == 0] = budget                                    # never-halted => hit the budget
    nvc = vm.cpu().numpy()
    return [G.dom_from_mask(nvc[i], csps[i]) for i in range(Bn)], halt


# ============================================================ POOLS
def chain_pool(rng, n_per, Ls=(1, 2, 3, 4, 5, 6, 8, 10), families=CHAIN_FAMILIES):
    """Deep-PROPAGATION instances at controlled chain length L (depth ~= L). Each is a determined hard
    chain; we record the measured prop-depth + treewidth so the recall can be bucketed by real depth."""
    from ..datagen import difficulty as DF
    out = []
    for fam in families:
        for L in Ls:
            got = 0
            tries = 0
            while got < n_per and tries < n_per * 30:
                tries += 1
                try:
                    p = HT.make_hard(rng, fam, int(L), True)
                except Exception:
                    continue
                if not p.determined or p.n > G.N_MAX or p.d > G.D_MAX:
                    continue
                csp = p.csp
                if len(csp.cons) > G.M_MAX or any(len(sc) > G.A_MAX for sc, _ in csp.cons):
                    continue
                df = DF.difficulty(csp)
                out.append({"csp": csp, "family": fam, "L": int(L), "query": int(p.query),
                            "depth": df["depth"], "treewidth": df["treewidth"], "level": df["level"],
                            "n": csp.n, "d": csp.d})
                got += 1
    return out


def affine_pool(rng, n, n_lo=8, n_hi=12, bands=(5, 6, 7), min_level=3):
    """Pure arity-3 XOR/affine instances at the L2/L3 WALL (required_level >= 2, so per-cell AC and the
    arity-3 factor lattice are INCOMPLETE), determined (unique solution => the dedP fixpoint pins every
    cell, a nonzero recall denominator) WITH the GF(2) (A,b) system attached so the certified GF2RowSpace
    can crack them exactly. The neural narrower + arc/factor hit the affine wall here; only GF2 finishes."""
    from ..datagen import difficulty as DF
    out = []
    tries = 0
    while len(out) < n and tries < n * 40:
        tries += 1
        nn = int(rng.integers(n_lo, n_hi + 1))
        d = XW.gen_xor_system(rng, nn, int(round(0.8 * nn)) + 2, nn, force_unique=True)  # global band => max tw
        csp = d["csp"]
        if csp.n > G.N_MAX or len(csp.cons) > G.M_MAX:
            continue
        if not beyond_factor(csp):                        # keep only genuine level>=3 walls (factor incomplete)
            continue
        df = DF.difficulty(csp)
        out.append({"csp": csp, "system": ("gf2", d["A"].astype(np.uint8), d["b"].astype(np.uint8)),
                    "tags": ("xor",), "depth": df["depth"], "treewidth": df["treewidth"],
                    "level": df["level"], "n": csp.n, "d": csp.d, "family": "xor"})
    return out


def mixed_pool(rng, n, chain_L=(2, 3, 4, 5), affine_band=(5, 6, 7), couple=True, min_level=3):
    """MIXED-hard affine+propagation instances, all BINARY (d=2, so the GF(2) op is sound everywhere).

    One instance = an AFFINE xor-wall subsystem (vars 0..nx-1, required_level>=2: GF2-solvable but a
    wall for per-cell AC / arity-3 factor / the neural narrower) PLUS a binary EQUALITY PROPAGATION
    chain (vars nx..nx+L: pin -> eq -> ... -> eq, depth L: trivial for the neural narrower / arc, but
    INVISIBLE to GF2 since the chain is NOT in the (A,b) routed to it). When couple=True the chain's
    far end is identified with an xor variable, so the propagated pin FLOWS into the affine system and
    back (genuine inter-organ information flow); couple=False makes them a disjoint union.

    Neither a single faculty finishes: GF2 misses the chain, arc/factor/neural miss the wide affine.
    Only the STACK (neural+GF2+certified, meeting over the shared lattice) reaches the dedP fixpoint.
    The (A,b) covers ONLY the affine rows (over the full var set; chain columns are zero)."""
    from ..datagen import difficulty as DF
    EQ2 = frozenset({(0, 0), (1, 1)})
    out = []
    tries = 0
    while len(out) < n and tries < n * 40:
        tries += 1
        nx = int(rng.integers(7, 10))                    # bigger affine part => reliably beyond factor
        dd = XW.gen_xor_system(rng, nx, int(round(0.8 * nx)) + 2, nx, force_unique=True)  # global band
        xcsp = dd["csp"]                                  # affine subsystem over vars 0..nx-1, d=2
        if not beyond_factor(xcsp):                       # need level>=3 so arc/factor(k=3) are INCOMPLETE
            continue
        Lmax = G.N_MAX - nx - 1                            # keep n_tot = nx + (L+1) <= NMAX
        choices = [l for l in chain_L if l <= Lmax]
        if not choices:
            continue
        L = int(rng.choice(choices))
        nchain = L + 1                                    # chain cells nx .. nx+L
        n_tot = nx + nchain
        # chain constraints (binary): pin(start) + equality edges; far end optionally == xor var v0
        chain0 = nx
        cons = list(xcsp.cons)
        pinval = int(rng.integers(0, 2))
        cons.append(((chain0,), frozenset({(pinval,)})))                      # pin chain start
        for i in range(L):
            cons.append(((chain0 + i, chain0 + i + 1), EQ2))                  # eq edge
        if couple:
            v0 = int(rng.integers(0, nx))                                     # share the far end with xor var
            cons.append(((chain0 + L, v0), EQ2))
        csp = C.CSP(n_tot, 2, tuple(cons))
        if len(csp.cons) > G.M_MAX or any(len(sc) > G.A_MAX for sc, _ in csp.cons):
            continue
        if not C.solutions(csp):                          # must stay satisfiable
            continue
        # GF(2) system = affine rows only, padded to the full var set (chain columns zero => GF2 blind to them)
        A = dd["A"].astype(np.uint8)
        m = A.shape[0]
        Afull = np.zeros((m, n_tot), np.uint8)
        Afull[:, :nx] = A
        df = DF.difficulty(csp)
        out.append({"csp": csp, "system": ("gf2", Afull, dd["b"].astype(np.uint8)), "tags": ("xor",),
                    "chain_L": L, "nx": nx, "couple": couple,
                    "depth": df["depth"], "treewidth": df["treewidth"], "level": df["level"],
                    "n": csp.n, "d": csp.d, "family": "mixed_coupled" if couple else "mixed_disjoint"})
    return out


# ============================================================ PART 1 — R-sweep
def part1_rsweep(organ, dev, rng, theta, n_per, Rs):
    pool = chain_pool(rng, n_per=n_per)
    # also add affine + high-treewidth for the full picture
    aff = affine_pool(rng, n=120)
    print(f"[part1] chain pool={len(pool)}  affine pool={len(aff)}", flush=True)

    def depth_bucket(d):
        if d <= 2:
            return "d<=2"
        if d <= 4:
            return "d3-4"
        if d <= 6:
            return "d5-6"
        if d <= 8:
            return "d7-8"
        return "d9+"

    # precompute dedP fixpoints once (the recall denominators) for the chain pool
    t0 = time.time()
    ded_chain = [dedp_fixpoint(e["csp"]) for e in pool]
    print(f"[part1] dedP fixpoints for chains: {time.time()-t0:.0f}s", flush=True)

    csps = [e["csp"] for e in pool]
    aff_csps = [e["csp"] for e in aff]
    ded_aff = [dedp_fixpoint(e["csp"]) for e in aff]
    # per-R single-pass recall, bucketed by measured depth (chains) + a flat AFFINE-wall contrast track
    curves = {}                                   # R -> bucket -> {num,den,fe,solved,n}
    perL = {}                                     # R -> family|L -> recall (for the depth-tracking law)
    affine_curve = {}                             # R -> recall on the affine wall (should NOT scale with R)
    for R in Rs:
        doms = organ_singlepass(organ, csps, R, dev, theta)
        bk = defaultdict(lambda: {"num": 0, "den": 0, "fe": 0, "solved": 0, "n": 0})
        lk = defaultdict(lambda: {"num": 0, "den": 0, "n": 0, "depth_sum": 0})
        for e, dom, ded in zip(pool, doms, ded_chain):
            inter, fe, den, solved = recall_fe(e["csp"], dom, ded)
            b = depth_bucket(e["depth"])
            bk[b]["num"] += inter; bk[b]["den"] += den; bk[b]["fe"] += fe
            bk[b]["solved"] += solved; bk[b]["n"] += 1
            key = f"{e['family']}|L{e['L']}"
            lk[key]["num"] += inter; lk[key]["den"] += den; lk[key]["n"] += 1
            lk[key]["depth_sum"] += e["depth"]
        curves[R] = {b: {"recall": v["num"] / max(1, v["den"]), "fe_cells": v["fe"],
                          "solved": v["solved"] / max(1, v["n"]), "n": v["n"],
                          "median_depth": None} for b, v in bk.items()}
        perL[R] = {k: {"recall": v["num"] / max(1, v["den"]), "n": v["n"],
                       "mean_depth": v["depth_sum"] / max(1, v["n"])} for k, v in lk.items()}
        agg = sum(bk[b]["num"] for b in bk) / max(1, sum(bk[b]["den"] for b in bk))
        # affine-wall track (capacity-limited; expect ~flat in R)
        adoms = organ_singlepass(organ, aff_csps, R, dev, theta) if aff_csps else []
        anum = aden = afe = 0
        for e, dom, dx in zip(aff, adoms, ded_aff):
            inter, fe, den, _ = recall_fe(e["csp"], dom, dx)
            anum += inter; aden += den; afe += fe
        affine_curve[R] = {"recall": anum / max(1, aden), "fe_cells": afe, "n": len(aff)}
        print(f"[part1] R={R:2d}  chain recall={agg:.3f}  affine-wall recall={affine_curve[R]['recall']:.3f}  "
              + "  ".join(f"{b}:{curves[R][b]['recall']:.2f}(n{curves[R][b]['n']})" for b in sorted(bk)),
              flush=True)

    # median depth per bucket (depth is fixed per instance, independent of R)
    bktmp = defaultdict(list)
    for e in pool:
        bktmp[depth_bucket(e["depth"])].append(e["depth"])
    median_depth = {b: float(np.median(v)) for b, v in bktmp.items()}

    # SATURATION-R per bucket: smallest R whose recall >= 0.98 * (recall at max R)
    sat = {}
    for b in median_depth:
        rmax = curves[Rs[-1]][b]["recall"]
        sat_R = Rs[-1]
        for R in Rs:
            if curves[R][b]["recall"] >= 0.98 * rmax and rmax > 0.05:
                sat_R = R
                break
        sat[b] = {"saturation_R": sat_R, "recall_at_satR": curves[sat_R][b]["recall"],
                  "recall_at_maxR": rmax, "median_depth": median_depth[b]}

    # SATURATION-R per (family,L): smallest R hitting >=0.98 of the L's max recall — vs the L's depth
    perL_sat = {}
    keys = set()
    for R in Rs:
        keys |= set(perL[R])
    for k in keys:
        recs = {R: perL[R].get(k, {"recall": 0.0})["recall"] for R in Rs}
        rmax = recs[Rs[-1]]
        sat_R = Rs[-1]
        for R in Rs:
            if recs[R] >= 0.98 * rmax and rmax > 0.1:
                sat_R = R
                break
        md = next((perL[R][k]["mean_depth"] for R in Rs if k in perL[R]), 0.0)
        perL_sat[k] = {"saturation_R": sat_R, "mean_depth": md, "max_recall": rmax,
                       "recall_curve": {str(R): round(recs[R], 3) for R in Rs}}

    # correlation: saturation-R vs depth across the (family,L) cells (the compute=depth law)
    xs = [v["mean_depth"] for v in perL_sat.values() if v["max_recall"] > 0.3]
    ys = [v["saturation_R"] for v in perL_sat.values() if v["max_recall"] > 0.3]
    corr = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) >= 3 else None
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 3 else None

    return {"Rs": list(Rs), "curves_by_depth_bucket": curves, "median_depth_by_bucket": median_depth,
            "saturation_by_bucket": sat, "saturation_by_family_L": perL_sat,
            "satR_vs_depth_corr": corr, "satR_vs_depth_slope": slope,
            "affine_wall_curve": affine_curve, "n_chain": len(pool), "n_affine": len(aff)}


# ============================================================ PART 2 — stacked organs
def _state(e):
    """A CSPState carrying the instance's native algebraic system (always attached; ops that don't use
    it ignore state.system, GF2/Modular dispatch on it via applies())."""
    return CSPState.full(e["csp"], system=e.get("system"), tags=frozenset(e.get("tags", ())))


def part2_stack(organ, dev, rng, theta, n_mixed):
    core = B.CoreNarrowOrgan(ckpt="runs/general_organ_full.pt", dev=dev)
    bank = B.build_bank(load_neural=False)
    certified = B.certified_csp_reductions(bank)        # Arc, Factor, Modular, GF2, Macro
    gf2 = bank["gf2_rowspace"]
    arc = bank["arc_consistency"]
    factor = bank["factor_consistency"]

    pools = {
        "mixed_coupled": mixed_pool(rng, n=n_mixed, couple=True),
        "mixed_disjoint": mixed_pool(rng, n=n_mixed, couple=False),
        "affine_pure": affine_pool(rng, n=n_mixed, n_lo=6, n_hi=11),
    }
    print(f"[part2] pools: " + ", ".join(f"{k}={len(v)}" for k, v in pools.items()), flush=True)

    results = {}
    for pname, pool in pools.items():
        if not pool:
            continue
        ded = [dedp_fixpoint(e["csp"]) for e in pool]
        configs = {
            "single_neural_raw": None,                  # organ.run_to_fixpoint, ungated (handled below)
            "single_neural_gated": [core],
            "single_gf2": [gf2],
            "single_arc_factor": [arc, factor],
            "STACK_all": certified + [core],
        }
        acc = {cfg: {"num": 0, "den": 0, "fe": 0, "solved": 0} for cfg in configs}
        # raw neural (run_to_fixpoint at deployed R=12) — the honest STANDALONE single-organ completeness
        csps = [e["csp"] for e in pool]
        raw_doms, _ = organ_fixpoint(organ, csps, R_inner=12, dev=dev, theta=theta)
        for e, dom, dx in zip(pool, raw_doms, ded):
            inter, fe, den, solved = recall_fe(e["csp"], dom, dx)
            acc["single_neural_raw"]["num"] += inter; acc["single_neural_raw"]["den"] += den
            acc["single_neural_raw"]["fe"] += fe; acc["single_neural_raw"]["solved"] += solved
        # composed configs (verifier-gated reduced product; system always present for GF2 dispatch)
        for cfg, reds in configs.items():
            if reds is None:
                continue
            for e, dx in zip(pool, ded):
                st = _state(e)
                try:
                    out, tr = reduced_product(st, reds, verify=False, max_rounds=64)
                except Exception:
                    out = st
                inter, fe, den, solved = recall_fe(e["csp"], tuple(out.dom), dx)
                acc[cfg]["num"] += inter; acc[cfg]["den"] += den
                acc[cfg]["fe"] += fe; acc[cfg]["solved"] += solved
        N = len(pool)
        results[pname] = {
            "n": N,
            "configs": {cfg: {"recall": v["num"] / max(1, v["den"]),
                              "false_elim_cells": v["fe"],
                              "solved_rate": v["solved"] / max(1, N)} for cfg, v in acc.items()},
        }
        print(f"[part2] {pname} (n={N}):", flush=True)
        for cfg in configs:
            r = results[pname]["configs"][cfg]
            print(f"          {cfg:22s} recall={r['recall']*100:5.1f}%  solved={r['solved_rate']*100:5.1f}%"
                  f"  FE={r['false_elim_cells']}", flush=True)
    return results


# ============================================================ PART 3 — gated adaptive depth
def part3_adaptive(organ, dev, rng, theta, n_per):
    pool = chain_pool(rng, n_per=n_per, Ls=(2, 3, 4, 5, 6, 8, 10))
    csps = [e["csp"] for e in pool]
    ded = [dedp_fixpoint(e["csp"]) for e in pool]
    print(f"[part3] adaptive pool={len(pool)}", flush=True)

    # ADAPTIVE: halt at the exact lattice fixpoint (R_inner=12 deployed)
    ad_doms, halt = organ_fixpoint(organ, csps, R_inner=12, dev=dev, theta=theta, max_outer=64)
    ad = {"num": 0, "den": 0, "fe": 0, "solved": 0}
    for e, dom, dx in zip(pool, ad_doms, ded):
        inter, fe, den, solved = recall_fe(e["csp"], dom, dx)
        ad["num"] += inter; ad["den"] += den; ad["fe"] += fe; ad["solved"] += solved
    adaptive = {"recall": ad["num"] / max(1, ad["den"]), "fe_cells": ad["fe"],
                "solved": ad["solved"] / max(1, len(pool)),
                "mean_outer_iters": float(halt.mean()), "max_outer_iters": int(halt.max())}

    # FIXED budgets: run exactly K outer passes (no early halt)
    fixed = {}
    for K in (1, 2, 4, 8, 16):
        fd_doms, _ = organ_fixpoint(organ, csps, R_inner=12, dev=dev, theta=theta, fixed_outer=K)
        f = {"num": 0, "den": 0, "fe": 0, "solved": 0}
        for e, dom, dx in zip(pool, fd_doms, ded):
            inter, fe, den, solved = recall_fe(e["csp"], dom, dx)
            f["num"] += inter; f["den"] += den; f["fe"] += fe; f["solved"] += solved
        fixed[K] = {"recall": f["num"] / max(1, f["den"]), "fe_cells": f["fe"],
                    "solved": f["solved"] / max(1, len(pool)), "outer_iters": K}

    # does halt-iteration track depth? (compute=depth at the OUTER loop)
    depths = np.array([e["depth"] for e in pool], float)
    corr = float(np.corrcoef(depths, halt.astype(float))[0, 1]) if len(pool) >= 3 else None
    # mean halt per depth bucket
    byd = defaultdict(list)
    for e, h in zip(pool, halt):
        byd[int(e["depth"])].append(int(h))
    halt_by_depth = {str(k): {"mean_halt": float(np.mean(v)), "n": len(v)} for k, v in sorted(byd.items())}

    print(f"[part3] adaptive recall={adaptive['recall']*100:.1f}% @ mean {adaptive['mean_outer_iters']:.1f} "
          f"outer iters; fixed K=8 recall={fixed[8]['recall']*100:.1f}% @ 8; "
          f"halt~depth corr={corr}", flush=True)
    return {"adaptive": adaptive, "fixed_budget": fixed,
            "halt_vs_depth_corr": corr, "mean_halt_by_depth": halt_by_depth, "n": len(pool)}


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/general_organ_full.pt")
    ap.add_argument("--out", default="runs/organ_block_scaling.json")
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--n_per", type=int, default=40, help="chain instances per (family,L)")
    ap.add_argument("--n_mixed", type=int, default=150)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_per, args.n_mixed = 8, 24

    dev = G.device()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    Rs = (1, 2, 4, 8, 12, 16, 24, 32) if not args.smoke else (1, 4, 12, 32)

    organ, meta = B.load_core_organ(args.ckpt, dev)
    print(f"[load] foundation organ: budget=({meta['N_MAX']},{meta['D_MAX']},{meta['M_MAX']},"
          f"{meta['A_MAX']}) d={meta['d']} R_train={meta['R']} params={meta.get('params')}  dev={dev}",
          flush=True)

    rng = np.random.default_rng(args.seed)
    out = {"meta": {k: meta[k] for k in ("N_MAX", "D_MAX", "M_MAX", "A_MAX", "d", "R", "params")
                    if k in meta}, "Rs": list(Rs), "theta": args.theta, "seed": args.seed}

    t0 = time.time()
    print("\n===== PART 1: recurrence-depth scaling (R-sweep) =====", flush=True)
    out["part1_rsweep"] = part1_rsweep(organ, dev, rng, args.theta, args.n_per, Rs)

    print("\n===== PART 2: stacked heterogeneous organs =====", flush=True)
    out["part2_stack"] = part2_stack(organ, dev, rng, args.theta, args.n_mixed)

    print("\n===== PART 3: gated adaptive-depth block =====", flush=True)
    out["part3_adaptive"] = part3_adaptive(organ, dev, rng, args.theta, args.n_per)

    out["elapsed_s"] = round(time.time() - t0, 1)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nwrote {args.out}  ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
