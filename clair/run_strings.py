"""clair/run_strings.py — NEURAL GUIDANCE for the certified string organ + the recall/soundness table.

clair.strings is the CERTIFIED authority (per-position char-set × length-interval reduced product, a
sound constraint-propagation operator, a regex->DFA chain, and a brute verifier), validated exact vs
brute. THIS file trains the neural part: the factor-graph message-passing narrower (clair.proposer) that
learns to PROPOSE which (position, char) eliminations to make — the learned "which constraint to
propagate / in what order" — while the certified discipline keeps it SOUND:

  * objective = DOMINATE the exact string dedP (clair.csp.exact_dedP on the to_csp encoding, proven equal
    to the native brute reachable sets in strings.py SMOKE step 3). Never eliminate a char some satisfying
    string uses => 0 false-elim. SOUNDNESS is free; COMPLETENESS is the research variable.
  * state = per-position candidate char-set over Σ∪{ε} (== the per-cell lattice; composable reduced
    product), plus the auxiliary DFA-state cells the automaton-chain encoding introduces for regex.
  * the certified string-AC (clair.strings.to_fixpoint) is the SOUND baseline that ABSTAINS where local
    arc consistency is too weak; the exact dedP is the 100% reference ceiling.

The honest research question (notes/codex_zoo_review.md): is a LEARNABLE CHECKED STRING organ viable —
does the neural narrower recover string deductions the local certified operator abstains on, soundly,
across crossword (position/equality/length) and regex-membership puzzles, alphabet sizes, and lengths;
and where (like the affine wall) does it still fall short of the exact dedP, which is exactly why the
certified operator is the necessary soundness floor and the neural part is amortised guidance?

  python -m clair.run_strings --smoke
  python -m clair.run_strings --train --steps 1500 --R 12 --target 4e5 --ckpt runs/string_organ.pt
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np
import torch

from . import csp as C
from . import run_general as RG

# ---- padding budget: regex adds N+1 aux DFA-state cells, so n can reach ~2N+1; d covers V and #states ----
N_MAX, D_MAX, M_MAX, A_MAX = 14, 8, 56, 3
RG.N_MAX, RG.D_MAX, RG.M_MAX, RG.A_MAX = N_MAX, D_MAX, M_MAX, A_MAX
RG.ENUM_CAP = 10 ** 18                                    # we gate via brute solution-count, not d**n

from .proposer import FactorGraphProposer, size_for       # noqa: E402
from . import strings as S                                 # noqa: E402

ALPHABETS = [2, 3, 4, 5]                                   # |Σ| sweep
SOL_CAP = 256                                              # reject puzzles with > this many solutions (cheap dedP)


# ============================================================ generators (witness-first; curated to budget)
def _fits(csp):
    return (csp.n <= N_MAX and csp.d <= D_MAX and len(csp.cons) <= M_MAX
            and all(len(sc) <= A_MAX for sc, _ in csp.cons))


def _curate(rng, rung, k=None, N=None):
    """Sample a (scsp, csp) whose encoding fits the budget, is solvable, and has <= SOL_CAP solutions
    (so the exact dedP target is cheap). Returns (scsp, csp) or None."""
    for _ in range(60):
        scsp = S.GEN[rung](rng, k=k, N=N)
        csp = S.to_csp(scsp)
        if not _fits(csp):
            continue
        sols, _ = S.brute(scsp, limit=SOL_CAP + 1)
        if 0 < len(sols) <= SOL_CAP:
            return scsp, csp
    return None


def sample(rng, rung, k=None, N=None):
    for _ in range(40):
        got = _curate(rng, rung, k=k, N=N)
        if got is not None:
            return got
    # trivial solvable fallback (a single pinned char)
    scsp = S.StringCSP(4, 3, (("len", 1, 4), ("pos", 0, frozenset({0}))))
    return scsp, S.to_csp(scsp)


def sample_corpus(rng, n, rungs):
    return [(rungs[i % len(rungs)],) + sample(rng, rungs[i % len(rungs)]) for i in range(n)]


# ============================================================ baselines (certified, exact)
def _removed(full, dom, n):
    return {(i, v) for i in range(n) for v in full[i] if v not in dom[i]}


def certified_string_ac(scsp: S.StringCSP):
    """Per-POSITION removed-set of the certified native string-AC (clair.strings.to_fixpoint), and whether
    it reached the exact string dedP / a solved singleton state. The sound-but-may-abstain baseline."""
    full = scsp.full()
    dom, _ = S.to_fixpoint(scsp)
    _, reach = S.brute(scsp)
    rm_ac = _removed(full, dom, scsp.N)
    rm_exact = _removed(full, reach, scsp.N)
    rec = len(rm_ac & rm_exact) / max(1, len(rm_exact))
    fe = len(rm_ac - rm_exact)                               # native AC is sound => 0
    return {"ac_recall": rec, "ac_false_elim": fe,
            "ac_solved": int(S.status(scsp, dom) == "solved"),
            "exact_uniq": int(all(len(c) == 1 for c in reach)),
            "rm_exact": rm_exact}


@torch.no_grad()
def evaluate(m, pairs, dev, theta=0.5, R_max=96):
    """Per-set: NEURAL per-position narrowing-recall + soundness vs the exact string dedP, alongside the
    certified native string-AC baseline. Recall/soundness are projected onto the POSITION cells (0..N-1)
    so neural and string-AC are apples-to-apples on the STRING; the exact dedP (= brute reachable) is 1.0."""
    m.eval()
    csps = [csp for _, csp in pairs]
    doms, (fe_k, fe_n) = RG.run_to_fixpoint(m, csps, dev, theta, R_max)
    nv = neu_solved = neu_wrong = ac_solved = uniq = 0
    rec_num = rec_den = 0
    ac_rec_num = ac_rec_den = 0
    for (scsp, csp), dom in zip(pairs, doms):
        sols, reach = S.brute(scsp)
        if not sols:
            continue
        nv += 1
        N = scsp.N
        full = csp.full()
        # exact dedP on position cells == native brute reachable (proven); use brute for clarity
        rm_exact = {(i, v) for i in range(N) for v in full[i] if v not in reach[i]}
        rm_neu = {(i, v) for i in range(N) for v in full[i] if v not in dom[i]}
        rec_num += len(rm_neu & rm_exact); rec_den += len(rm_exact)
        base = certified_string_ac(scsp)
        ac_rec_num += base["ac_recall"] * len(rm_exact)
        ac_rec_den += len(rm_exact)
        ac_solved += base["ac_solved"]; uniq += base["exact_uniq"]
        # neural "solved" = every position cell a singleton; verify the decoded string actually satisfies
        pos_dom = tuple(dom[i] for i in range(N))
        if all(len(c) == 1 for c in pos_dom):
            neu_solved += 1
            x = tuple(next(iter(pos_dom[i])) for i in range(N))
            neu_wrong += int(not S.satisfies(scsp, x))
    m.train()
    ns = max(1, nv)
    return {"n": nv,
            "neural_recall": rec_num / max(1, rec_den),
            "ac_recall": ac_rec_num / max(1, ac_rec_den),
            "exact_recall": 1.0,                             # certified dedP == exact (ceiling)
            "neural_solved": neu_solved / ns, "ac_solved": ac_solved / ns,
            "uniq": uniq / ns,
            "false_elim": fe_k / max(1, fe_n), "false_elim_count": int(fe_k),
            "neural_wrong": neu_wrong / ns}


def _row(tag, e):
    return (f"  {tag:16s} n={e['n']:3d}  recall: neural {e['neural_recall']*100:5.1f}%  "
            f"string-AC {e['ac_recall']*100:5.1f}%  exact {e['exact_recall']*100:5.1f}%  |  "
            f"solved: neural {e['neural_solved']*100:5.1f}% AC {e['ac_solved']*100:5.1f}% "
            f"(uniq {e['uniq']*100:4.1f}%)  |  FALSE-ELIM {e['false_elim']:.4f}({e['false_elim_count']}) "
            f"wrong {e['neural_wrong']*100:.1f}%")


# ============================================================ eval-set construction
def make_eval(rung, neval, k=None, N=None, seed=4321):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < neval:
        out.append(sample(rng, rung, k=k, N=N))
    return out


# ============================================================ training (dominate-dedP, on-policy)
def train(rungs, dev, d, steps, pool=96, R=12, lr=3e-4, theta=0.5, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = FactorGraphProposer("full", N_MAX, D_MAX, M_MAX, A_MAX, d=d, R=R, ds=min(4, R)).to(dev)
    npar = m.n_params()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95))
    tagged = sample_corpus(rng, pool, rungs)
    items = [(csp, csp.full()) for _, _, csp in tagged]
    rtags = [rg for rg, _, _ in tagged]
    print(f"  string organ: full d={d} params={npar:,} pool={pool} R={R} steps={steps} "
          f"budget N={N_MAX} D={D_MAX} M={M_MAX} alphabets={ALPHABETS}", flush=True)
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
                    rg = rtags[bi]; _, nc = sample(rng, rg); nxt.append((nc, nc.full()))
                else:
                    nxt.append((csp, ndom))
            items = nxt
        if s % max(1, steps // 15) == 0:
            fer = fe_k / max(1, fe_n)
            log.append({"step": s, "loss": float(loss.detach()), "false_elim": fer})
            print(f"    step {s:5d}  loss {float(loss.detach()):.3f}  false_elim {fer:.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            fe_k = fe_n = 0
    return m, npar, log


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--R", type=int, default=12)
    ap.add_argument("--target", type=float, default=4e5)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--pool", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--neval", type=int, default=80)
    ap.add_argument("--rmax", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="runs/string_organ.pt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = RG.device()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    print(f"device={dev}  budget N={N_MAX} D={D_MAX} M={M_MAX} A={A_MAX}", flush=True)

    if args.smoke:
        rng = np.random.default_rng(0)
        print("=== run_strings SMOKE: generators within budget + dedP==brute, organ trains ===", flush=True)
        for rg in S.TRAIN_RUNGS:
            t0 = time.time(); scsp, csp = sample(rng, rg)
            base = certified_string_ac(scsp)
            print(f"  {rg:10s} k={scsp.k} N={scsp.N} cells={csp.n:2d} d={csp.d} cons={len(csp.cons):3d} "
                  f"exact-uniq={base['exact_uniq']} AC-solved={base['ac_solved']} "
                  f"AC-recall={base['ac_recall']*100:.0f}% ({(time.time()-t0)*1000:.0f}ms)", flush=True)
        d, npar = size_for("full", N_MAX, D_MAX, M_MAX, A_MAX, args.target, R=args.R, ds=4)
        print(f"  size_for(full,target={args.target:.0e}) -> d={d} params={npar:,}", flush=True)
        m, npar, log = train(S.TRAIN_RUNGS, dev, d, steps=150, pool=48, R=args.R, seed=0)
        for rg in S.TRAIN_RUNGS:
            cs = make_eval(rg, 40, seed=100)
            print(_row(rg, evaluate(m, cs, dev, args.theta, args.rmax)), flush=True)
        return

    if args.train:
        d, npar = size_for("full", N_MAX, D_MAX, M_MAX, A_MAX, args.target, R=args.R, ds=4)
        m, npar, log = train(S.TRAIN_RUNGS, dev, d, args.steps, pool=args.pool, R=args.R,
                             lr=args.lr, theta=args.theta, seed=args.seed)
        out = {"budget": [N_MAX, D_MAX, M_MAX, A_MAX], "alphabets": ALPHABETS, "params": npar,
               "log": log, "args": vars(args)}

        print("\n=== PER-RUNG (mixed alphabets/lengths, in-distribution) ===", flush=True)
        out["per_rung"] = {}
        for rg in S.TRAIN_RUNGS:
            cs = make_eval(rg, args.neval, seed=1000)
            e = evaluate(m, cs, dev, args.theta, args.rmax); out["per_rung"][rg] = e
            print(_row(rg, e), flush=True)

        print("\n=== RECALL/SOUNDNESS ACROSS ALPHABET SIZE |Σ| (rung=position) ===", flush=True)
        out["by_alphabet"] = {}
        for kk in ALPHABETS:
            cs = make_eval("position", args.neval, k=kk, seed=2000 + kk)
            e = evaluate(m, cs, dev, args.theta, args.rmax); out["by_alphabet"][kk] = e
            print(_row(f"position |Σ|={kk}", e), flush=True)

        print("\n=== RECALL/SOUNDNESS ACROSS STRING LENGTH N (rung=regex, |Σ|=3) ===", flush=True)
        out["by_length"] = {}
        for nn in [4, 5, 6]:
            cs = make_eval("regex", args.neval, k=3, N=nn, seed=3000 + nn)
            if not cs:
                continue
            e = evaluate(m, cs, dev, args.theta, args.rmax); out["by_length"][nn] = e
            print(_row(f"regex N={nn}", e), flush=True)

        os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
        torch.save({"state_dict": m.state_dict(),
                    "config": {"arm": "full", "n_max": N_MAX, "d_max": D_MAX, "m_max": M_MAX,
                               "a_max": A_MAX, "d": d, "R": args.R, "ds": min(4, args.R)},
                    "params": npar, "alphabets": ALPHABETS, "train_rungs": S.TRAIN_RUNGS,
                    "per_rung": out["per_rung"], "by_alphabet": out["by_alphabet"],
                    "by_length": out["by_length"], "log": log, "args": vars(args)}, args.ckpt)
        print(f"\nsaved checkpoint -> {args.ckpt} ({npar:,} params)", flush=True)
        if args.out:
            json.dump(out, open(args.out, "w"), indent=1, default=str)
            print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
