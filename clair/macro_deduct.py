"""clair/macro_deduct.py — MACRO-DEDUCTION: checked macro-operators that compress many narrowing
steps into one (the automata-shortcut idea, Liu 2022, applied to lattice narrowing).

THE BET. The deduction fixpoint on deep-propagation problems (long equality / forced-colour chains,
graph diameter L) takes T = O(L) sequential narrowing rounds: information moves one hop per round.
The automata-shortcut theorem says a semiautomaton's T-step computation is simulable in O(log T)
DEPTH (always; O(1) if the transition monoid is solvable), by COMPOSING the per-step transition
MAPS rather than iterating states. Lattice narrowing's per-step operator is a map on the candidate
lattice; composing those maps = a MACRO that does several base steps in one application. If depth is
the cost (the recurring "affine wall = width, beaten by depth" finding), log-depth macros are the win.

WHAT A MACRO IS HERE.  The deep tasks (eqchain, forcedcolor) are PATH-structured: constraints couple
cell i with cells within a window w behind it. Such a CSP is a SEMIAUTOMATON over a sliding-window
state S = V^w, with a per-edge transfer relation M[t] : S -> S (a boolean S x S matrix). Narrowing
to the deduction fixpoint = forward x backward reachability over the M[t] (an HMM-style scan):

    base step F1  = advance the scan ONE edge          -> O(L) sequential steps   (= one ac_step hop)
    MACRO  F_{2^j}= compose two reach-2^{j-1} transfers -> J = ceil(log2 L) rounds (parallel-prefix)

The composition is boolean matrix product = the TRANSITION-MONOID product. For eqchain the per-step
map is idempotent (semilattice: V^w diagonal); for forcedcolor it is a PERMUTATION on the 6 valid
(c_{i-1},c_i) states (a group => solvable => the O(1)/O(log) regime). Both compress.

CHECKED / SOUND.  A macro's narrowed domain is ACCEPTED only if it is DOMINATED by the true
narrowing (never eliminates a value some solution uses), checked against the exact per-cell
transformer dedP (clair.hard_tasks.fast_dedP). The symbolic macro is exact-by-construction (the
composed transfer relation is the true reachability), so it is always accepted; the NEURAL macros
are speculative and the check is what keeps the loop sound — reject -> fall back to a smaller macro
or to F1.

TWO PARTS.
  PART A  (symbolic, pure python; the headline)  reach-doubling scan vs F1-iteration: does the
          macro reach the SAME fixpoint as O(L) base steps in O(log L) macro-applications, soundly?
  PART B  (neural, GPU; the exploratory risk)  distil neural macros N1,N2,N4,N8 (one forward pass
          each) to match the 1/2/4/8-step exact narrowing on the lattice, geometric ("full") vs
          table ("ffn") arm. Measure per-macro recall / false-elim vs its k-step target, and the
          macro-acceptance rate under the dedP check. Does a LEARNED macro compose with fidelity,
          and does the geometry help?

Run:
    python -m clair.macro_deduct --mode smoke            # both parts, tiny
    python -m clair.macro_deduct --mode symbolic         # PART A depth-compression table (CPU)
    python -m clair.macro_deduct --mode neural --steps 600 --out runs/macro_neural.json   # PART B (GPU)
    python -m clair.macro_deduct --mode full  --out runs/macro.json                       # both
"""
from __future__ import annotations

import argparse
import itertools as it
import json
import math
import os
import time

import numpy as np

from . import csp as C
from . import hard_tasks as HT


# =====================================================================================
# deep path-CSP generators (NO NMAX cap — pure-narrowing experiment, so we drop the LM decoy)
# =====================================================================================
def gen_eqchain(L, k=3, rng=None, determined=True):
    """Equality chain of L edges: pin(0)=v0, eq(i,i+1) for i in 0..L-1. n=L+1 cells. Window w=1
    (semilattice / idempotent transition). Determined => every cell forced to v0; needs L base hops."""
    rng = rng or np.random.default_rng(0)
    n = L + 1
    v0 = int(rng.integers(0, k))
    cons = [C._rel((i, i + 1), lambda t: t[0] == t[1], k) for i in range(L)]
    if determined:
        cons = [C._rel((0,), (lambda v: lambda t: t[0] == v)(v0), k)] + cons
    return C.CSP(n, k, tuple(cons)), 1


def gen_forcedcolor(L, k=3, rng=None, determined=True):
    """Forced k-colouring cascade: neq(i,i-1) & neq(i,i-2), pins on cells 0,1. n=L+1 cells. Window
    w=2 (the transition is a PERMUTATION on the 6 valid (c_{i-1},c_i) states => solvable group).
    Determined => fully forced periodic colouring; needs ~L base hops."""
    rng = rng or np.random.default_rng(0)
    assert k >= 3
    n = L + 1
    cons = []
    if n >= 2:
        cons.append(C._rel((1, 0), lambda t: t[0] != t[1], k))
    for i in range(2, n):
        cons.append(C._rel((i, i - 1), lambda t: t[0] != t[1], k))
        cons.append(C._rel((i, i - 2), lambda t: t[0] != t[1], k))
    v0, v1 = (int(x) for x in rng.choice(k, size=2, replace=False))
    cons.append(C._rel((0,), (lambda v: lambda t: t[0] == v)(v0), k))
    if determined:
        cons.append(C._rel((1,), (lambda v: lambda t: t[0] == v)(v1), k))
    return C.CSP(n, k, tuple(cons)), 2


GEN = {"eqchain": gen_eqchain, "forcedcolor": gen_forcedcolor}


# =====================================================================================
# PART A — windowed-automaton transfer relations + reach-doubling macro scan
# =====================================================================================
def _enc(tup, k):
    e = 0
    for v in tup:
        e = e * k + v
    return e


def _dec(e, k, w):
    out = []
    for _ in range(w):
        out.append(e % k)
        e //= k
    return tuple(reversed(out))


def build_automaton(csp: C.CSP, w):
    """Compile a PATH csp (constraints span <= w cells back) into a sliding-window semiautomaton.

      state at position t  = the w-tuple of cell values (cells t-w+1 .. t)        (S = k^w states)
      alive[t][s]          = state s is locally consistent at t (intra-window cons + pins on those cells)
      M[t][s, s']          = transfer t -> t+1 : s' shifts s (drop oldest, append new cell t+1) AND the
                             appended value satisfies every constraint whose scope lies in {t+1-w .. t+1}

    Returns (alive [n,S] bool, M [n-1,S,S] bool, S, w). Exact: a global assignment is consistent iff
    it is a path s_0 -> s_1 -> ... through alive states following M.
    """
    n, k = csp.n, csp.d
    S = k ** w
    states = [_dec(e, k, w) for e in range(S)]
    # constraints indexed by their max cell (where they become checkable) and min cell (window need)
    cons = [(tuple(sc), set(al)) for sc, al in csp.cons]
    pins = {}                                            # cell -> set of allowed values (from unary cons)
    for sc, al in cons:
        if len(sc) == 1:
            vs = {t[0] for t in al}
            pins[sc[0]] = pins.get(sc[0], set(range(k))) & vs

    def cell_val(t, s, cell):
        """value of `cell` in window state s anchored at position t (covers t-w+1..t), or None."""
        off = cell - (t - w + 1)
        return s[off] if 0 <= off < w else None

    def window_ok(t, s):
        """state s at position t satisfies all multi-cell constraints fully inside the window AND pins."""
        lo = t - w + 1
        for cell in range(max(0, lo), t + 1):
            v = cell_val(t, s, cell)
            if cell in pins and v not in pins[cell]:
                return False
        for sc, al in cons:
            if len(sc) < 2:
                continue
            if all(lo <= c <= t for c in sc):
                if tuple(cell_val(t, s, c) for c in sc) not in al:
                    return False
        return True

    def append_ok(t1, s_prev, s_new):
        """transfer (position t1-1, state s_prev) -> (position t1, state s_new): s_new must SHIFT
        s_prev (drop oldest, append the new cell t1), and the newly-revealed cell t1 must satisfy
        every constraint whose scope lies in {t1-w .. t1} (= {oldest cell of s_prev .. t1}). The
        combined (w+1)-cell window = s_prev ++ (new value) gives both endpoints of any span<=w cons."""
        if s_prev[1:] != s_new[:-1]:                      # overlap of w-1 cells must agree
            return False
        cell = t1
        v = s_new[-1]
        if cell in pins and v not in pins[cell]:
            return False
        combined = s_prev + (v,)                          # cells (t1-w .. t1), index off = c-(t1-w)
        for sc, al in cons:
            if len(sc) < 2:
                continue
            if max(sc) == cell and min(sc) >= cell - w:   # checkable exactly when cell t1 is placed
                if tuple(combined[c - (t1 - w)] for c in sc) not in al:
                    return False
        return True

    alive = np.zeros((n, S), dtype=bool)
    for t in range(w - 1, n):
        for e, s in enumerate(states):
            alive[t, e] = window_ok(t, s)
    M = np.zeros((max(0, n - 1), S, S), dtype=bool)
    for t1 in range(w, n):                                # transfer into position t1
        for ep, sp in enumerate(states):
            if not alive[t1 - 1, ep]:
                continue
            for en, sn in enumerate(states):
                if alive[t1, en] and append_ok(t1, sp, sn):
                    M[t1 - 1, ep, en] = True
    return alive, M, S, w


def _bvm(vec, mat):
    """boolean vector @ matrix : reachable next states."""
    return (vec @ mat) > 0


def _bmm(a, b):
    """boolean matrix product."""
    return (a.astype(np.int64) @ b.astype(np.int64)) > 0


def forward_sequential(alive, M, w):
    """O(L) scan: f[t] = window states at t reachable from a valid start. Returns f [n,S], n_steps."""
    n, S = alive.shape
    f = np.zeros((n, S), dtype=bool)
    f[w - 1] = alive[w - 1]
    for t in range(w, n):
        f[t] = _bvm(f[t - 1], M[t - 1]) & alive[t]
    return f, max(0, n - w)


def backward_sequential(alive, M, w):
    n, S = alive.shape
    b = np.zeros((n, S), dtype=bool)
    b[n - 1] = alive[n - 1]
    for t in range(n - 2, w - 2, -1):
        b[t] = (M[t] @ b[t + 1]) & alive[t]               # states at t that can reach a valid end
    return b


def forward_doubling(alive, M, w):
    """O(log L)-DEPTH macro scan: parallel-prefix (Hillis-Steele) product of the per-edge transfer
    matrices — composing the transition MAPS instead of iterating states. Each round COMPOSES a
    reach-r macro with another (boolean matrix product = the transition-monoid product), doubling the
    reach; after ceil(log2 L) rounds every prefix transfer is known. Returns (f [n,S], n_macro_rounds).
    Asserted identical to forward_sequential — same fixpoint, log-depth instead of linear-depth."""
    n, S = alive.shape
    L = n - w                                             # number of edges to traverse
    f = np.zeros((n, S), dtype=bool)
    f[w - 1] = alive[w - 1]
    if L <= 0:
        return f, 0
    base = [M[w - 1 + t] & alive[w + t][None, :] for t in range(L)]   # edge t, target-aliveness baked in
    prefix, rounds = _scan_prefix_products(base, S)       # prefix[p] = base[0]...base[p-1]; depth ~log2 L
    for p in range(1, L + 1):
        f[w - 1 + p] = _bvm(f[w - 1], prefix[p])
    return f, rounds


def _scan_prefix_products(base, S):
    """Hillis-Steele inclusive scan computing all prefix products of a list of SxS boolean transfer
    matrices. Returns (prefix, rounds) where prefix[p] = base[0] . base[1] ... base[p-1]
    (prefix[0] = identity) and rounds = number of doubling rounds = ceil(log2 L) (the MACRO count:
    each round composes reach-r maps into reach-2r maps)."""
    L = len(base)
    I = np.eye(S, dtype=bool)
    arr = [b.copy() for b in base]                        # arr[i] starts = base[i]
    stride, rounds = 1, 0
    while stride < L:
        rounds += 1
        new = [a.copy() for a in arr]
        for i in range(stride, L):
            new[i] = _bmm(arr[i - stride], arr[i])        # left operand = earlier prefix (order matters)
        arr = new
        stride *= 2
    out = [I] + arr                                       # out[p] = product base[0..p-1]
    return out, max(0, rounds)


def per_cell_from_windows(consistent, csp, w):
    """Project globally-consistent window states (forward & backward) to per-cell domains."""
    n, k = csp.n, csp.d
    S = consistent.shape[1]
    states = [_dec(e, k, w) for e in range(S)]
    dom = []
    for cell in range(n):
        t = min(max(cell, w - 1), n - 1)                  # a window position that covers `cell`
        off = cell - (t - w + 1)
        surv = {states[e][off] for e in range(S) if consistent[t, e]}
        dom.append(frozenset(surv))
    return tuple(dom)


def macro_solve(csp: C.CSP, w, use_doubling=True):
    """Reach the deduction fixpoint via the windowed automaton. If use_doubling, the forward pass is
    the O(log L)-DEPTH parallel-prefix macro scan (compose transition maps); else the O(L) sequential
    base-step scan. Returns (dom, effective_depth, n_states)."""
    alive, M, S, w = build_automaton(csp, w)
    if use_doubling:
        f, depth = forward_doubling(alive, M, w)
    else:
        f, depth = forward_sequential(alive, M, w)
    b = backward_sequential(alive, M, w)
    consistent = f & b
    dom = per_cell_from_windows(consistent, csp, w)
    return dom, depth, S


def f1_iterate_depth(csp: C.CSP):
    """Plain F1-iteration baseline: ac_step to fixpoint. Returns (dom, n_rounds). n_rounds ~ L."""
    dom, steps = C.to_fixpoint(C.ac_step, csp, csp.full())
    return dom, steps


def exact_fixpoint(csp: C.CSP):
    """The exact per-cell dedP fixpoint (ground-truth target) and the cheap fast_dedP one-shot."""
    fd = HT.fast_dedP(csp)                                # exact per-cell, value used by some solution
    return fd


def check_dominated(dom, exact):
    """SOUNDNESS check: dom must keep every value the exact transformer keeps (no false elimination),
    and be reductive is not required here (we compare final domains). Returns (ok, false_elim_count)."""
    fe = sum(len(exact[i] - dom[i]) for i in range(len(dom)))
    return fe == 0, fe


# =====================================================================================
# PART A driver — depth-compression table
# =====================================================================================
def run_symbolic(args):
    fams = args.families.split(",")
    Ls = [int(x) for x in args.lengths.split(",")]
    k = args.k
    rng = np.random.default_rng(args.seed)
    print(f"\n===== PART A: symbolic macro-deduction (reach-doubling vs F1-iteration) =====", flush=True)
    print(f"  families={fams} L={Ls} k={k}\n", flush=True)
    rows = []
    hdr = (f"  {'family':12s} {'L':>4s} {'n':>4s} | {'F1-depth':>9s} {'macro-depth':>11s} "
           f"{'ratio':>6s} | {'fixpoint-match':>14s} {'false-elim':>10s} {'accept':>7s}")
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)
    for fam in fams:
        for L in Ls:
            csp, w = GEN[fam](L, k=k, rng=rng, determined=True)
            exact = exact_fixpoint(csp)
            # baseline: F1 iteration (ac_step to fixpoint)
            f1_dom, f1_depth = f1_iterate_depth(csp)
            # macro: reach-doubling scan
            mac_dom, mac_depth, S = macro_solve(csp, w, use_doubling=True)
            # sanity: doubling result must equal sequential result
            seq_dom, seq_depth, _ = macro_solve(csp, w, use_doubling=False)
            assert mac_dom == seq_dom, f"doubling != sequential at {fam} L={L}"
            # checked acceptance: macro domain dominated by exact (sound)?
            ok, fe = check_dominated(mac_dom, exact)
            # does the macro reach the SAME fixpoint as exact dedP?
            match = mac_dom == exact
            # also confirm F1 reaches it (sanity on the baseline)
            f1_match = f1_dom == exact
            ratio = f1_depth / max(1, mac_depth)
            rows.append(dict(family=fam, L=L, n=csp.n, f1_depth=f1_depth, macro_depth=mac_depth,
                             ratio=ratio, fixpoint_match=bool(match), f1_match=bool(f1_match),
                             false_elim=fe, accept=bool(ok), n_states=S))
            print(f"  {fam:12s} {L:>4d} {csp.n:>4d} | {f1_depth:>9d} {mac_depth:>11d} "
                  f"{ratio:>6.1f} | {str(match):>14s} {fe:>10d} {str(ok):>7s}", flush=True)
    print("\n  Interpretation: macro-depth should grow ~log2(L) while F1-depth grows ~L; both must"
          "\n  reach the EXACT dedP fixpoint (fixpoint-match=True) with false-elim=0 (sound).", flush=True)
    return {"symbolic": rows}


# =====================================================================================
# PART B — neural macro distillation (GPU). Import torch lazily.
# =====================================================================================
def k_step_target(csp: C.CSP, dom, k):
    """The k-step narrowing target = the BASE step F1 (one local arc-consistency round, csp.ac_step)
    applied k times from dom. F1 propagates exactly ONE hop, so (F1)^k narrows k hops — this is the
    distillation target for macro Fk (a single application that should match k base steps). Sound:
    ac_step is dominated by dedP, so its k-fold composition is too."""
    d = dom
    for _ in range(k):
        nd = C.ac_step(csp, d)
        if nd == d:
            break
        d = nd
    return d


def run_neural(args, dev=None):
    import torch
    import torch.nn.functional as F
    from .proposer import FactorGraphProposer, size_for
    from . import run_general as RG

    if dev is None:
        dev = RG.device()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    # allow a deeper featuriser budget for the neural part (default run_general caps at 8 cells)
    if getattr(args, "nmax", 0):
        RG.N_MAX = args.nmax
    if getattr(args, "mmax", 0):
        RG.M_MAX = args.mmax
    print(f"\n===== PART B: neural macro distillation  device={dev}  "
          f"budget N={RG.N_MAX} D={RG.D_MAX} M={RG.M_MAX} A={RG.A_MAX} =====", flush=True)

    fams = args.families.split(",")
    ks = [int(x) for x in args.macro_ks.split(",")]       # macro depths to distil (1,2,4,8)
    arms = args.arms.split(",")                            # geometric 'full' vs table 'ffn'
    # training-length pool: must fit run_general's N_MAX budget (<=8 cells). Use short chains for the
    # featurised model (the LATTICE-fidelity question is per-step, not per-L); depth-compression is
    # answered by PART A. We distil each macro on chains up to L_train.
    L_train = args.l_train
    Ls_eval = [int(x) for x in args.neural_eval_L.split(",")]
    Ls_eval = [L for L in Ls_eval if L + 1 <= RG.N_MAX]   # featuriser cap
    rng = np.random.default_rng(args.seed)

    def sample_pool(npool):
        items = []
        for _ in range(npool):
            fam = fams[int(rng.integers(0, len(fams)))]
            L = int(rng.integers(1, L_train + 1))
            det = bool(rng.random() < 0.7)
            csp, w = GEN[fam](L, k=args.k, rng=rng, determined=det)
            if csp.n <= RG.N_MAX and csp.d <= RG.D_MAX and len(csp.cons) <= RG.M_MAX:
                # start from a partially-narrowed state sometimes (on-policy-ish coverage)
                dom = csp.full()
                items.append((csp, dom))
        return items

    out = {"families": fams, "macro_ks": ks, "arms": arms, "L_train": L_train, "results": {}}

    for arm in arms:
        out["results"][arm] = {}
        for k in ks:
            d, npar = size_for(arm, RG.N_MAX, RG.D_MAX, RG.M_MAX, RG.A_MAX, args.target, R=args.R,
                               ds=min(4, args.R))
            m = FactorGraphProposer(arm, RG.N_MAX, RG.D_MAX, RG.M_MAX, RG.A_MAX, d=d, R=args.R,
                                    ds=min(4, args.R)).to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=args.lr, betas=(0.9, 0.95))
            print(f"\n  -- distil macro N{k} arm={arm} d={d} params={npar:,} R={args.R} --", flush=True)
            t0 = time.time()
            pool = sample_pool(args.pool)
            for step in range(1, args.steps + 1):
                # build k-step targets on current pool states
                feat = RG.featurize(pool, dev)
                vm = feat["var_mask"]
                B = len(pool)
                tgt = np.zeros((B, RG.N_MAX, RG.D_MAX), np.float32)
                conflict = np.zeros((B,), np.float32)
                ktarg = []
                for bi, (csp, dom) in enumerate(pool):
                    kd = k_step_target(csp, dom, k)
                    ktarg.append(kd)
                    if all(len(c) == 0 for c in kd):
                        conflict[bi] = 1.0
                    for i in range(csp.n):
                        for v in kd[i]:
                            tgt[bi, i, v] = 1.0
                tgt_t = torch.as_tensor(tgt, device=dev)
                conflict_t = torch.as_tensor(conflict, device=dev)
                b, cls, sup = RG.fwd(m, feat, vm)
                loss = RG.loss_fn(sup, vm, feat["var_valid"], tgt_t, conflict_t)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                # roll pool forward by the model's own meet (on-policy lattice coverage)
                with torch.no_grad():
                    nvm = RG.meet(vm, b, args.theta).cpu().numpy()
                    nxt = []
                    for bi, (csp, dom) in enumerate(pool):
                        ndom = RG.dom_from_mask(nvm[bi], csp)
                        if C.status(ndom) in ("solved", "conflict") or ndom == dom:
                            nxt.append(sample_pool(1)[0] if sample_pool(1) else (csp, csp.full()))
                        else:
                            nxt.append((csp, ndom))
                    pool = nxt
                if step % max(1, args.steps // 6) == 0:
                    print(f"     step {step:5d} loss {float(loss):.3f}  {time.time()-t0:.0f}s", flush=True)
            # ---- evaluate macro fidelity vs its k-step target on fresh deep chains ----
            ev = eval_macro(m, k, fams, Ls_eval, args, dev, RG)
            out["results"][arm][k] = ev
            for fam in fams:
                e = ev[fam]
                print(f"     [{fam}] recall {e['recall']*100:5.1f}%  false-elim {e['false_elim']:.4f} "
                      f"({e['fe_count']})  accept {e['accept_rate']*100:5.1f}%  match {e['match']*100:5.1f}%",
                      flush=True)
    return {"neural": out, "neural_eval_L": Ls_eval}


def eval_macro(m, k, fams, Ls, args, dev, RG):
    """Per-macro fidelity vs the k-step exact narrowing target: micro narrowing-RECALL (of the
    eliminations the k-step target makes, what fraction the macro also makes), FALSE-ELIM (survivors
    wrongly dropped — must be ~0), the dedP-domination ACCEPTANCE rate (the checker), and exact
    fixpoint MATCH. One forward pass per macro application."""
    import torch
    m.eval()
    rng = np.random.default_rng(args.seed + 1)
    res = {}
    for fam in fams:
        rec_num = rec_den = 0
        fe_count = fe_den = 0
        accept = total = 0
        match = 0
        for L in Ls:
            for _ in range(args.neval):
                csp, w = GEN[fam](L, k=args.k, rng=rng, determined=True)
                if csp.n > RG.N_MAX:
                    continue
                dom = csp.full()
                tgt = k_step_target(csp, dom, k)
                feat = RG.featurize([(csp, dom)], dev)
                vm = feat["var_mask"]
                with torch.no_grad():
                    b, cls, _ = RG.fwd(m, feat, vm)
                    nvm = RG.meet(vm, b, args.theta).cpu().numpy()
                mdom = RG.dom_from_mask(nvm[0], csp)
                full = csp.full()
                rm_model = {(i, v) for i in range(csp.n) for v in full[i] if v not in mdom[i]}
                rm_tgt = {(i, v) for i in range(csp.n) for v in full[i] if v not in tgt[i]}
                rec_num += len(rm_model & rm_tgt); rec_den += len(rm_tgt)
                # false elim vs the EXACT one-shot dedP (sound reference): survivors the macro killed
                exact = HT.fast_dedP(csp, dom)
                fe = sum(len(exact[i] - mdom[i]) for i in range(csp.n))
                fe_count += fe; fe_den += sum(len(c) for c in exact)
                total += 1
                accept += int(fe == 0)                    # checker accepts iff dominated (sound)
                match += int(mdom == tgt)
        res[fam] = dict(recall=rec_num / max(1, rec_den), false_elim=fe_count / max(1, fe_den),
                        fe_count=int(fe_count), accept_rate=accept / max(1, total),
                        match=match / max(1, total), n=total)
    m.train()
    return res


# =====================================================================================
# smoke
# =====================================================================================
def run_smoke(args):
    print("\n===== SMOKE =====", flush=True)
    # Part A on tiny chains, with the exact cross-checks
    rng = np.random.default_rng(0)
    for fam in ("eqchain", "forcedcolor"):
        for L in (3, 5, 8):
            csp, w = GEN[fam](L, k=3, rng=rng, determined=True)
            exact = HT.fast_dedP(csp)
            mac_dom, mdepth, S = macro_solve(csp, w, True)
            seq_dom, sdepth, _ = macro_solve(csp, w, False)
            f1_dom, f1d = f1_iterate_depth(csp)
            assert mac_dom == seq_dom, (fam, L, "doubling!=seq")
            assert mac_dom == exact, (fam, L, "macro!=exact", mac_dom, exact)
            assert f1_dom == exact, (fam, L, "f1!=exact")
            print(f"  [{fam} L={L}] n={csp.n} S={S}  F1-depth={f1d}  macro-depth={mdepth}  "
                  f"macro==exact={mac_dom==exact}  sound(fe=0)={check_dominated(mac_dom, exact)[0]}",
                  flush=True)
    print("  PART A smoke OK (doubling==sequential==exact dedP; macro-depth << F1-depth)", flush=True)
    # tiny neural smoke
    if not args.no_neural:
        a = argparse.Namespace(**vars(args))
        a.steps = args.steps or 60
        a.pool = 32
        a.macro_ks = "1,2"
        a.arms = "full"
        a.target = 1.0e5
        a.l_train = 5
        a.neural_eval_L = "4,6"
        a.neval = 20
        try:
            run_neural(a)
            print("  PART B smoke OK (a macro distils; fidelity/accept reported)", flush=True)
        except Exception as e:
            print(f"  PART B smoke SKIPPED/failed: {type(e).__name__}: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "symbolic", "neural", "full"], default="smoke")
    ap.add_argument("--families", default="eqchain,forcedcolor")
    ap.add_argument("--lengths", default="8,16,32,64,128")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    # neural
    ap.add_argument("--macro_ks", default="1,2,4,8")
    ap.add_argument("--arms", default="full,ffn")
    ap.add_argument("--l_train", type=int, default=6)
    ap.add_argument("--neural_eval_L", default="4,6,7")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--pool", type=int, default=96)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--target", type=float, default=2.0e5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--neval", type=int, default=40)
    ap.add_argument("--nmax", type=int, default=0)        # override run_general.N_MAX (deeper chains)
    ap.add_argument("--mmax", type=int, default=0)        # override run_general.M_MAX (more factors)
    ap.add_argument("--no_neural", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = {"args": vars(args)}
    if args.mode == "smoke":
        run_smoke(args)
    elif args.mode == "symbolic":
        out.update(run_symbolic(args))
    elif args.mode == "neural":
        out.update(run_neural(args))
    elif args.mode == "full":
        out.update(run_symbolic(args))
        out.update(run_neural(args))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1, default=str)
        print("\nwrote", args.out, flush=True)


if __name__ == "__main__":
    main()
