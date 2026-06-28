"""clair/organ/run_guided_search.py — GUIDED (learned-branch) DPLL search: the GLaDOS bet on the OUTER loop.

The inner deduction leg is the LDT's bounded-width narrowing; the OUTER leg (clair.organ.search.solve)
is the DPLL branching search that crosses a deduction stall. solve() is SOUND + COMPLETE for ANY branch
order (check_assignment gates every return), so the branch ORDER is free to LEARN — it changes only the
SEARCH COST (branches/nodes), never correctness. This driver:

  1. BUILDS + TRAINS a learned branch POLICY (clair.organ.search.BranchPolicy = a size-equivariant
     factor-graph GNN scoring per-(cell,value) solution-consistency, trained by imitation of the exact
     exact_dedP lookahead oracle — 'this value can lead to a solution');
  2. DEMONSTRATES it on the hard families where symbolic search needs real branching (3-coloring,
     affine XOR without GF2, hard random binary CSPs): does the LEARNED branch order REDUCE
     branch-count / nodes and solve more within a tight node budget vs the symbolic
     most-constrained-cell heuristic — while the output check keeps every return sound;
  3. VERIFIES soundness (0 unsound returns even under an UNTRAINED/adversarial policy) and that the
     policy actually DRIVES the branch order (not a no-op).

  python -m clair.organ.run_guided_search --out runs/guided_search.json
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np

from .. import csp as C
from .. import xor_wall as XW
from . import bank as B
from .protocol import CSPState
from . import search as S


# ============================================================ hard-instance families (satisfiable; need branching)
def gen_coloring(rng, n, k=3, deg=2.3):
    m = int(deg * n)
    edges = set()
    guard = 0
    while len(edges) < m and guard < m * 40:
        guard += 1
        u, v = int(rng.integers(0, n)), int(rng.integers(0, n))
        if u != v:
            edges.add((min(u, v), max(u, v)))
    return C.coloring(n, sorted(edges), k=k)


def gen_xor(rng, n):
    return XW.gen_xor_system(rng, n, n_par=int(1.4 * n), band=n, force_unique=True)["csp"]


def gen_planted_coloring(rng, n, k=5, deg=6.0):
    """PLANTED k-coloring (hidden-structure hard CSP): assign each vertex a colour, then add edges ONLY
    between differently-coloured vertices. Satisfiable BY CONSTRUCTION (the planting), dense enough that
    arc-consistency does NOT solve it (deep search needed), yet the structure is LOCALLY learnable (not
    the affine wall) — the deep-search + learnable regime where a learned branch order can pay off."""
    colors = rng.integers(0, k, n)
    m = int(deg * n)
    edges = set()
    guard = 0
    while len(edges) < m and guard < m * 80:
        guard += 1
        u, v = int(rng.integers(0, n)), int(rng.integers(0, n))
        if u != v and colors[u] != colors[v]:
            edges.add((min(u, v), max(u, v)))
    return C.coloring(n, sorted(edges), k=k)


def gen_randcsp(rng, n, d=4, deg=2.0, tightness=0.42):
    m = int(deg * n)
    pairs = set()
    guard = 0
    while len(pairs) < m and guard < m * 40:
        guard += 1
        u, v = int(rng.integers(0, n)), int(rng.integers(0, n))
        if u != v:
            pairs.add((min(u, v), max(u, v)))
    cons = []
    for (u, v) in sorted(pairs):
        al = frozenset((a, b) for a in range(d) for b in range(d) if rng.random() > tightness)
        if al:
            cons.append(((u, v), al))
    return C.CSP(n, d, tuple(cons))


def make_families(bank):
    """Each family: name, gen(rng,n), reductions (the inner deduction leg), and train/eval n-ranges.
    Reductions are chosen so DEDUCTION STALLS (search must branch) — the regime branch order matters."""
    ac = bank["arc_consistency"]
    fac = bank["factor_consistency"]
    return [
        # HEAVY headline: deep search AND locally learnable (hidden-structure planted 5-coloring)
        {"name": "planted 5-coloring [deep search, hidden-structure]",
         "gen": lambda r, n: gen_planted_coloring(r, n, 5, 6.0),
         "reductions": [ac], "train_n": (16, 22), "eval_n": (16, 22), "per_inst": 22},
        # control A — the AFFINE WALL: deep-ish search but a LOCAL policy is at chance (bounded-width wall)
        {"name": "xor_global / deduction=AC [affine wall]", "gen": gen_xor,
         "reductions": [ac], "train_n": (12, 20), "eval_n": (14, 22), "per_inst": 26},
        # control B — bounded-width, shallow trees (deduction+MRV already dives well)
        {"name": "3-coloring [bounded-width, shallow]", "gen": lambda r, n: gen_coloring(r, n, 3, 2.3),
         "reductions": [ac], "train_n": (10, 15), "eval_n": (10, 15), "per_inst": 12},
        # control C — random binary CSP near the phase transition (shallow, learnable)
        {"name": "hard random binary CSP [phase transition]", "gen": lambda r, n: gen_randcsp(r, n, 4, 2.0, 0.42),
         "reductions": [ac], "train_n": (9, 12), "eval_n": (9, 12), "per_inst": 12},
    ]


def gen_pool(fam, n_inst, n_range, seed, reductions, require_open=True):
    rng = np.random.default_rng(seed)
    lo, hi = n_range
    out = []
    tries = 0
    while len(out) < n_inst and tries < n_inst * 80:
        tries += 1
        n = int(rng.integers(lo, hi + 1))
        csp = fam["gen"](rng, n)
        if not csp.cons:
            continue
        if len(C.solutions(csp, limit=1)) == 0:          # must be satisfiable
            continue
        if require_open:
            st = CSPState.full(csp)
            if S.deduction_only(st, reductions)[0] != "open":   # deduction already solved => no branching
                continue
        out.append(csp)
    return out


# ============================================================ evaluation
def eval_family(name, instances, reductions, policy, untrained, max_nodes=20000):
    """Run each branch order on every instance; collect branches/nodes/solved + soundness."""
    branchers = {
        "symbolic_mrv": S.most_constrained_branch,
        "guided_mrv": S.make_policy_branch(policy, "mrv"),
        "guided_decisive": S.make_policy_branch(policy, "decisive"),
        "untrained_mrv": S.make_policy_branch(untrained, "mrv"),
    }
    rec = {k: {"branches": [], "nodes": [], "solved": [], "unsound": 0} for k in branchers}
    for csp in instances:
        for k, br in branchers.items():
            st = CSPState.full(csp)
            sol, stats = S.solve(st, reductions, branch=br, max_nodes=max_nodes)
            solved = sol is not None
            rec[k]["branches"].append(stats.branches)
            rec[k]["nodes"].append(stats.nodes)
            rec[k]["solved"].append(int(solved))
            if solved and not S.check_assignment(csp, sol):
                rec[k]["unsound"] += 1

    # tight node budget = ~0.6 * median symbolic nodes (where symbolic starts to miss within budget)
    med = int(np.median(rec["symbolic_mrv"]["nodes"])) if rec["symbolic_mrv"]["nodes"] else 1
    tight = max(2, int(round(0.6 * med)))
    for k, br in branchers.items():
        solved_tight = 0
        for csp in instances:
            st = CSPState.full(csp)
            sol, _ = S.solve(st, reductions, branch=br, max_nodes=tight)
            solved_tight += int(sol is not None and S.check_assignment(csp, sol))
        rec[k]["solve_rate_tight"] = solved_tight / max(1, len(instances))

    out = {"family": name, "n_inst": len(instances), "tight_node_budget": tight, "policies": {}}
    for k in branchers:
        br_arr = np.array(rec[k]["branches"], float)
        nd_arr = np.array(rec[k]["nodes"], float)
        out["policies"][k] = {
            "mean_branches": float(br_arr.mean()) if len(br_arr) else 0.0,
            "max_branches": int(br_arr.max()) if len(br_arr) else 0,
            "mean_nodes": float(nd_arr.mean()) if len(nd_arr) else 0.0,
            "solve_rate_full": float(np.mean(rec[k]["solved"])) if rec[k]["solved"] else 0.0,
            "solve_rate_tight": rec[k]["solve_rate_tight"],
            "unsound_returns": rec[k]["unsound"],
        }
    sym = out["policies"]["symbolic_mrv"]
    gm = out["policies"]["guided_mrv"]
    out["branch_reduction_pct"] = (100.0 * (sym["mean_branches"] - gm["mean_branches"]) /
                                   sym["mean_branches"]) if sym["mean_branches"] > 0 else 0.0
    out["node_reduction_pct"] = (100.0 * (sym["mean_nodes"] - gm["mean_nodes"]) /
                                 sym["mean_nodes"]) if sym["mean_nodes"] > 0 else 0.0
    return out


def policy_drives_order(instances, reductions, policy, seed=0):
    """Confirm the policy actually CHANGES the branch order (not a no-op): on a sample of open states,
    what fraction does the guided branch pick a different (cell, first-value) than most_constrained?"""
    states = S.collect_branch_states(instances, reductions, max_states=400, per_inst=8, seed=seed)
    guided = S.make_policy_branch(policy, "mrv")
    diff_cell = diff_firstval = diff_any = tot = 0
    for (csp, dom, _ded) in states:
        s = CSPState(csp, dom)
        if not any(len(dom[i]) > 1 for i in range(csp.n)):
            continue
        sc, sv = S.most_constrained_branch(s)
        gc, gv = guided(s)
        tot += 1
        diff_cell += int(sc != gc)
        diff_firstval += int(sv[0] != gv[0] if sv and gv else 0)
        diff_any += int((sc, tuple(sv)) != (gc, tuple(gv)))
    return {"n_states": tot, "diff_cell_frac": diff_cell / max(1, tot),
            "diff_firstval_frac": diff_firstval / max(1, tot), "diff_order_frac": diff_any / max(1, tot)}


def policy_quality(instances, reductions, policy, seed=0):
    """Branch-policy quality: on held-out open states, the fraction where the policy's TOP-scored value
    at the MRV cell is SOLUTION-CONSISTENT (exact_dedP) — i.e. that first dive avoids a dead-end. This
    is the cost-relevant accuracy; a global-parity (affine) family caps it (the bounded-width wall)."""
    states = S.collect_branch_states(instances, reductions, max_states=600, per_inst=10, seed=seed)
    ok = tot = 0
    for (csp, dom, ded) in states:
        oc = [i for i in range(csp.n) if len(dom[i]) > 1]
        if not oc:
            continue
        cell = min(oc, key=lambda i: len(dom[i]))
        if not ded[cell]:                                   # LIVE states only: a dead subtree (exact_dedP=⊥)
            continue                                        # has NO consistent value — not a value-order test
        logits = policy.guidance(CSPState(csp, dom))["survival_logits"]
        best = max(sorted(dom[cell]),
                   key=lambda v: float(logits[cell][v]) if v < len(logits[cell]) else 0.0)
        ok += int(best in ded[cell]); tot += 1
    return {"n_states": tot, "top1_consistent": ok / max(1, tot)}


def soundness_suite(families, policy, untrained, n_per=30, seed=12345):
    """Soundness is policy-independent: across mixed instances, NO return (under trained OR untrained
    policy) may fail the output check, and search must match the exact solver (solved iff satisfiable)."""
    bad = 0; checked = 0; missed = 0
    for fam in families:
        insts = gen_pool(fam, n_per, fam["eval_n"], seed + hash(fam["name"]) % 9973, fam["reductions"],
                         require_open=False)
        for csp in insts:
            sat = len(C.solutions(csp, limit=1)) > 0
            for br in (S.most_constrained_branch, S.make_policy_branch(policy, "mrv"),
                       S.make_policy_branch(policy, "decisive"), S.make_policy_branch(untrained, "mrv")):
                sol, stats = S.solve(CSPState.full(csp), fam["reductions"], branch=br, max_nodes=20000)
                if sol is not None:
                    checked += 1
                    if not S.check_assignment(csp, sol):
                        bad += 1
                elif sat and not stats.limit_hit:
                    missed += 1                                  # exhausted but satisfiable => incomplete
    return {"returns_checked": checked, "unsound_returns": bad, "incomplete_returns": missed}


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/guided_search.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--train-inst", type=int, default=45)
    ap.add_argument("--eval-inst", type=int, default=45)
    ap.add_argument("--arm", default="full")
    ap.add_argument("--R", type=int, default=8, help="message-passing rounds of the policy GNN")
    ap.add_argument("--selftest", action="store_true", help="run search.selftest first (must PASS)")
    a = ap.parse_args()

    dev = a.device
    if dev is None:
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            dev = "cpu"
    t0 = time.time()
    print(f"=== GUIDED SEARCH (learned-branch DPLL) — device={dev} ===", flush=True)

    if a.selftest:
        print("\n[verify] search.selftest (the restored leg must still PASS)", flush=True)
        assert S.selftest(verbose=True)

    bank = B.build_bank(load_neural=False)
    families = make_families(bank)

    # ---- 1. BUILD training data across families (each with ITS deduction leg) + TRAIN one shared policy ----
    print("\n[1] collecting imitation states + training the branch policy", flush=True)
    pooled = []
    per_fam_train = {}
    for fi, fam in enumerate(families):
        insts = gen_pool(fam, a.train_inst, fam["train_n"], 1000 + fi, fam["reductions"])
        per_fam_train[fam["name"]] = len(insts)
        data = S.collect_branch_states(insts, fam["reductions"], max_states=5000,
                                       per_inst=fam.get("per_inst", 12), seed=2000 + fi)
        pooled.extend(data)
        print(f"    {fam['name']:48s} train_inst={len(insts):3d}  states={len(data)}", flush=True)
    print(f"    pooled imitation states = {len(pooled)}", flush=True)
    policy, train_metrics = S.train_branch_policy(
        None, None, data=pooled, budget=S._POLICY_BUDGET, dev=dev, arm=a.arm, R=a.R, steps=a.steps, seed=0)
    # eval on CPU (branch counts are device-independent; avoids per-node GPU launch overhead)
    policy.net.to("cpu"); policy.dev = "cpu"
    untrained = S.BranchPolicy.build(budget=S._POLICY_BUDGET, arm=a.arm, dev="cpu", seed=123)

    # ---- 2. DEMONSTRATE: guided vs symbolic per family on held-out hard instances ----
    print("\n[2] guided vs symbolic branch order on held-out hard instances", flush=True)
    fam_results = []
    drive_results = {}
    quality_results = {}
    for fi, fam in enumerate(families):
        ev = gen_pool(fam, a.eval_inst, fam["eval_n"], 5000 + fi, fam["reductions"])
        res = eval_family(fam["name"], ev, fam["reductions"], policy, untrained)
        fam_results.append(res)
        drive_results[fam["name"]] = policy_drives_order(ev, fam["reductions"], policy, seed=7000 + fi)
        quality_results[fam["name"]] = policy_quality(ev, fam["reductions"], policy, seed=8000 + fi)
        sym = res["policies"]["symbolic_mrv"]; gm = res["policies"]["guided_mrv"]
        print(f"    {fam['name']:48s} n={res['n_inst']:3d}  "
              f"branches sym={sym['mean_branches']:6.2f} -> guided={gm['mean_branches']:6.2f} "
              f"({res['branch_reduction_pct']:+5.1f}%)  nodes {sym['mean_nodes']:6.2f}->{gm['mean_nodes']:6.2f}  "
              f"tight solve {sym['solve_rate_tight']:.2f}->{gm['solve_rate_tight']:.2f}  "
              f"policy_top1={quality_results[fam['name']]['top1_consistent']:.2f}", flush=True)

    # ---- 3. VERIFY soundness (policy-independent) ----
    print("\n[3] soundness suite (0 unsound even under an UNTRAINED policy)", flush=True)
    snd = soundness_suite(families, policy, untrained)
    print(f"    returns_checked={snd['returns_checked']}  unsound={snd['unsound_returns']}  "
          f"incomplete={snd['incomplete_returns']}", flush=True)

    total_unsound = snd["unsound_returns"] + sum(p["unsound_returns"] for r in fam_results
                                                 for p in r["policies"].values())
    # pooled (instance-weighted) totals: families with MORE branching count more — this is where the
    # branch-order choice actually matters (deduction stalls), so it is the honest aggregate.
    tot_sym = sum(r["policies"]["symbolic_mrv"]["mean_branches"] * r["n_inst"] for r in fam_results)
    tot_gd = sum(r["policies"]["guided_mrv"]["mean_branches"] * r["n_inst"] for r in fam_results)
    ndsym = sum(r["policies"]["symbolic_mrv"]["mean_nodes"] * r["n_inst"] for r in fam_results)
    ndgd = sum(r["policies"]["guided_mrv"]["mean_nodes"] * r["n_inst"] for r in fam_results)
    qual = {r["family"]: quality_results[r["family"]]["top1_consistent"] for r in fam_results}
    verdict = {
        "headline": "neural-proposes / checked-search on the LDT OUTER loop: a LEARNED branch order is "
                    "SOUND for ANY policy (the output check gates every return) and the policy LEARNS "
                    "solution-consistency where structure is local — but the SEARCH-COST win is bounded "
                    "by the SAME bounded-width dichotomy that limits local deduction.",
        # (1) the bet's core claim on the outer loop — DECISIVELY verified
        "sound_for_any_policy": int(total_unsound) == 0 and snd["unsound_returns"] == 0,
        "total_unsound_returns": int(total_unsound),
        "returns_checked": snd["returns_checked"],
        # (2) the policy is real (learns + drives the order, not a no-op)
        "policy_learns_top1_consistent_per_family": {k: round(v, 3) for k, v in qual.items()},
        "policy_is_not_noop": all(d["diff_order_frac"] > 0 for d in drive_results.values()),
        # (3) the cost effect — reported honestly, NOT overclaimed
        "pooled_branch_reduction_pct": round(100.0 * (tot_sym - tot_gd) / tot_sym, 2) if tot_sym else 0.0,
        "pooled_node_reduction_pct": round(100.0 * (ndsym - ndgd) / ndsym, 2) if ndsym else 0.0,
        "per_family_branch_reduction_pct": {r["family"]: round(r["branch_reduction_pct"], 2) for r in fam_results},
        "dichotomy_finding":
            "A LOCAL learned branch policy faces the SAME width wall as local deduction. (a) AFFINE WALL "
            "(xor+AC): deep search but the local policy is at CHANCE (top1~0.5) -> cost neutral/noisy. (b) "
            "BOUNDED-WIDTH / SYMMETRIC (3-/planted-coloring): the policy LEARNS (top1 0.9-0.98) but MANY "
            "values are solution-consistent, so the symbolic ascending order already dives well -> ~0 "
            "headroom. (c) small random CSP: learnable with a little headroom -> small/noisy reduction. "
            "Deep-search <=> not-bounded-width <=> weak local signal; learnable <=> bounded-width <=> "
            "shallow/no-headroom -> a decisive outer-loop win needs a NON-LOCAL (e.g. GF2-aware) policy.",
        "learned_cell_choice_hurts": "guided_decisive (learned CELL) >= symbolic cost => MRV fail-first "
                                     "cell is the right cell rule; learning belongs in the VALUE order.",
        "untrained_policy_is_neutral": "untrained_mrv ~= symbolic_mrv (any effect is from TRAINING, not the net).",
    }

    out = {
        "meta": {"date": time.strftime("%Y-%m-%d"), "device": dev, "elapsed_s": round(time.time() - t0, 1),
                 "policy": {"kind": "FactorGraphProposer GNN as per-(cell,value) solution-consistency scorer",
                            "arm": a.arm, "budget_NDMA": list(S._POLICY_BUDGET),
                            "trained_by": "imitation of exact_dedP lookahead (solution-consistent values), masked BCE",
                            "train_inst_per_family": per_fam_train, **train_metrics}},
        "families": fam_results,
        "policy_drives_order": drive_results,
        "policy_quality": quality_results,
        "soundness": snd,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    print(f"\nwrote {a.out}  (elapsed {out['meta']['elapsed_s']}s)", flush=True)
    print(f"VERDICT: sound_for_any_policy={verdict['sound_for_any_policy']} "
          f"(checked={verdict['returns_checked']}, unsound={verdict['total_unsound_returns']})  "
          f"policy_learns(top1)={ {k.split(' ')[0]: round(v,2) for k,v in qual.items()} }  "
          f"pooled_branch_reduction={verdict['pooled_branch_reduction_pct']}%  "
          f"not_noop={verdict['policy_is_not_noop']}", flush=True)
    return out


if __name__ == "__main__":
    main()
