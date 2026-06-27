"""clair/run_general.py — a GENERAL lattice-deduction organ across the whole verifier ladder.

The proposer.FactorGraphProposer is ALREADY a general factor-graph message-passing narrower
(variable<->factor messages, monotone meet, arbitrary allowed-tuple relations, arity 1/2/3). What
was missing was a UNIVERSAL DATA PATH: run_proposer.py only ever fed it the toy csp.py kinds
(chain/2sat/3sat/xor/coloring, d<=3). The coloring-only organ is augmented.ColorDeductor; THIS file
trains the general organ on a MIX of the curriculum + SMT rungs, every one expressed in the SAME
universal representation:

    a problem = (n cells, domain 0..d-1, a set of FACTORS = (scope, allowed-tuple relation)).

Rungs (each its own EXACT clair.csp ground truth — solutions / exact_dedP / solve_factor):
    coloring    pairwise != edges + pins                  (arity 1,2 ; d=3)
    equality    = / != groups + pins                      (arity 1,2 ; d=3)
    ordering    < / <= chains + pins                       (arity 1,2 ; d<=6)
    arithmetic  modular (a+b)%d == c sums + pins           (arity 1,3 ; d<=6)   <- the arity-3 workout
    alldiff     all-distinct (as a !=-clique) + pins       (arity 1,2 ; d<=6)
    smt         bounded-int le/lt/sum/diff/mod/modsum+pins (arity 1,2 ; d<=6)

FOL (Datalog entailment) is NOT a finite-domain per-cell assignment CSP — it is derivability in a
Herbrand model, with no fixed set of cells to narrow — so it does NOT map to the factor lattice and
is OUT OF SCOPE for this organ (stated honestly, not silently dropped).

The objective is unchanged from run_proposer: DOMINATE the exact per-cell transformer dedP (never
eliminate a value some solution uses) so SOUNDNESS is free; COMPLETENESS (does the monotone meet
drive a solvable instance to a checkable `solved`, or abstain) is the research variable. We add the
factor-lattice fixpoint (clair.csp.solve_factor) as the level-k completeness reference the prompt
asks for ("reach the exact factor-lattice fixpoint").

THREE experiments (the generalization question):
    multitask   train ONE organ on all rungs; eval per rung  -> does one organ stay sound + complete?
    single      train on rung r, eval rung r                 -> the per-type specialist baseline
    heldout     train on all-but-r, eval r ZERO-SHOT          -> does deduction transfer to an
                                                                 UNSEEN constraint type?

Pure organ (no OLMo). Run:
    python -m clair.run_general --mode smoke
    python -m clair.run_general --mode full --steps 800 --out runs/general.json
"""
from __future__ import annotations

import argparse
import itertools as it
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .proposer import FactorGraphProposer, size_for
from . import csp as C
from . import curriculum as CU

# ----- padding budget (covers every rung after alldiff is decomposed to a !=-clique) -----
N_MAX, D_MAX, M_MAX, A_MAX = 8, 8, 28, 3
ENUM_CAP = 4096               # reject any CSP with d**n above this (keeps exact enumeration cheap)

RUNGS = ["coloring", "equality", "ordering", "arithmetic", "alldiff", "smt"]


def device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# =====================================================================================
# universal rung -> clair.csp.CSP samplers (all witness-first => guaranteed solvable)
# =====================================================================================
def _expand_alldiff(facts):
    """all-distinct over a scope == pairwise-!= over the scope (SOLUTION-equivalent), so an n-ary
    alldiff becomes arity-2 factors the organ can hold. The only lossiness is at the LATTICE level
    (binary AC cannot do Hall-set propagation), which is exactly the honest alldiff completeness gap."""
    out = []
    for f in facts:
        if f[0] == "alldiff":
            out += [("neq", a, b) for a, b in it.combinations(f[1], 2)]
        else:
            out.append(f)
    return out


def _smt_csp(rng, R=5):
    """Bounded-integer system (0..R) as a finite-domain CSP, witness-first (plant m, emit only
    constraints m satisfies). Mirrors clair.smt's constraint kinds; all arity<=2; no z3 needed —
    clair.csp.solutions IS the exact oracle for the deductor."""
    n = int(rng.integers(3, 5))          # 3..4
    d = R + 1
    m = [int(rng.integers(0, d)) for _ in range(n)]
    facts = []                           # reuse curriculum fact tuples where shapes match
    cons = []                            # extensional (scope, allowed) built directly
    # a few pins
    npin = int(rng.integers(1, n))
    for i in rng.choice(n, size=npin, replace=False):
        cons.append(C._rel((int(i),), (lambda v: lambda t: t[0] == v)(m[int(i)]), d))
    for _ in range(int(rng.integers(n, 2 * n))):
        kind = str(rng.choice(["le", "lt", "sum", "diff", "mod", "modsum"]))
        i, j = (int(x) for x in rng.choice(n, size=2, replace=False))
        if kind == "le" and m[i] <= m[j]:
            cons.append(C._rel((i, j), lambda t: t[0] <= t[1], d))
        elif kind == "lt" and m[i] < m[j]:
            cons.append(C._rel((i, j), lambda t: t[0] < t[1], d))
        elif kind == "sum":
            c = m[i] + m[j]
            cons.append(C._rel((i, j), (lambda c: lambda t: t[0] + t[1] == c)(c), d))
        elif kind == "diff":
            c = m[i] - m[j]
            cons.append(C._rel((i, j), (lambda c: lambda t: t[0] - t[1] == c)(c), d))
        elif kind == "mod":
            k = int(rng.integers(2, 5)); r = m[i] % k
            cons.append(C._rel((i,), (lambda k, r: lambda t: t[0] % k == r)(k, r), d))
        elif kind == "modsum":
            k = int(rng.integers(2, 5)); r = (m[i] + m[j]) % k
            cons.append(C._rel((i, j), (lambda k, r: lambda t: (t[0] + t[1]) % k == r)(k, r), d))
    return C.CSP(n, d, tuple(cons))


def sample_rung_csp(rng, rung):
    """Return a SOLVABLE clair.csp.CSP for `rung`, within the padding budget. None of these need
    a planted-solution label — the deductor's supervision is clair.csp.exact_dedP, computed fresh."""
    for _ in range(200):
        if rung == "coloring":
            n, d, _, facts, s = CU.gen_coloring(rng, k=3, n_lo=4, n_hi=6)
        elif rung == "equality":
            n, d, _, facts, s = CU.gen_equality(rng, k=3, n_lo=4, n_hi=6)
        elif rung == "ordering":
            n, d, _, facts, s = CU.gen_ordering(rng, n_lo=3, n_hi=4)
        elif rung == "arithmetic":
            n, d, _, facts, s = CU.gen_arithmetic(rng, n_lo=3, n_hi=4, d_lo=4, d_hi=6)
        elif rung == "alldiff":
            n, d, _, facts, s = CU.gen_alldiff(rng, n_lo=3, n_hi=4)
        elif rung == "smt":
            csp = _smt_csp(rng)
            if csp.n <= N_MAX and csp.d <= D_MAX and len(csp.cons) <= M_MAX \
               and csp.d ** csp.n <= ENUM_CAP and len(C.solutions(csp)) > 0 \
               and all(len(sc) <= A_MAX for sc, _ in csp.cons):
                return csp
            continue
        else:
            raise ValueError(rung)
        csp = CU.build_csp(n, d, _expand_alldiff(facts))
        if csp.n <= N_MAX and csp.d <= D_MAX and len(csp.cons) <= M_MAX \
           and csp.d ** csp.n <= ENUM_CAP and len(C.solutions(csp)) > 0 \
           and all(len(sc) <= A_MAX for sc, _ in csp.cons):
            return csp
    return CU.build_csp(3, 3, [("pin", 0, 0)])   # trivial fallback (always solvable)


def sample_corpus(rng, n, rungs):
    """n (rung, csp) pairs, round-robin over the requested rungs."""
    return [(rungs[i % len(rungs)], sample_rung_csp(rng, rungs[i % len(rungs)])) for i in range(n)]


# =====================================================================================
# exact-harness cache (SHARED across every training run so the d**n enumeration is paid once)
# =====================================================================================
class Exact:
    def __init__(self):
        self.sol, self.ded = {}, {}

    def _k(self, csp, dom):
        return (csp.cons, csp.d, dom)

    def solutions(self, csp, dom):
        k = self._k(csp, dom)
        if k not in self.sol:
            self.sol[k] = C.solutions(csp, dom)
        return self.sol[k]

    def dedP(self, csp, dom):
        # Routes to the ONE canonical deductor (clair.csp.exact_dedP: Rust clair_fast port with the
        # witness early-stop when built, else the pure-Python clair.csp._exact_dedP_py). This used to
        # hold its own union-over-solutions copy; DELETED after verifying it bitwise-equal.
        k = self._k(csp, dom)
        if k not in self.ded:
            self.ded[k] = C.exact_dedP(csp, dom)
        return self.ded[k]


SHARED = Exact()


# =====================================================================================
# featurization (configurable budget; broadcast relation tables of any allowed-tuple set)
# =====================================================================================
def relation_table(scope, al, d_max, a_max):
    a = len(scope)
    t = np.zeros((d_max,) * a_max, dtype=np.float32)
    for tup in al:
        sl = [slice(None)] * a_max
        for p in range(a):
            sl[p] = tup[p]
        t[tuple(sl)] = 1.0
    return t.reshape(-1)


def featurize_np(items):
    """Build the factor-graph feature arrays (host numpy) for a batch of (csp, dom). Pure +
    deterministic — depends only on `items`, so it is safe to compute off the main thread / in a
    worker. `featurize` wraps this and moves the arrays to a device; clair.datagen.fast reuses it
    directly so the parallel/overlapped data path is BITWISE-IDENTICAL to the serial one."""
    B = len(items)
    rel_dim = D_MAX ** A_MAX
    var_mask = np.zeros((B, N_MAX, D_MAX), np.float32)
    given = np.zeros((B, N_MAX), np.float32)
    var_valid = np.zeros((B, N_MAX), np.float32)
    fac_rel = np.zeros((B, M_MAX, rel_dim), np.float32)
    fac_arity = np.zeros((B, M_MAX, A_MAX), np.float32)
    fac_valid = np.zeros((B, M_MAX), np.float32)
    edge_var = np.full((B, M_MAX, A_MAX), N_MAX, np.int64)
    edge_valid = np.zeros((B, M_MAX, A_MAX), np.float32)
    for bi, (csp, dom) in enumerate(items):
        for i in range(csp.n):
            var_valid[bi, i] = 1.0
            for v in dom[i]:
                var_mask[bi, i, v] = 1.0
            if len(dom[i]) == 1:
                given[bi, i] = 1.0
        for fi, (sc, al) in enumerate(csp.cons):
            if fi >= M_MAX:
                break
            fac_valid[bi, fi] = 1.0
            fac_arity[bi, fi, len(sc) - 1] = 1.0
            fac_rel[bi, fi] = relation_table(sc, al, D_MAX, A_MAX)
            for p, cell in enumerate(sc):
                edge_var[bi, fi, p] = cell
                edge_valid[bi, fi, p] = 1.0
    return dict(var_mask=var_mask, given=given, var_valid=var_valid,
                fac_rel=fac_rel, fac_arity=fac_arity, fac_valid=fac_valid,
                edge_var=edge_var, edge_valid=edge_valid)


def to_device(np_feat, dev):
    """Move a featurize_np() dict to `dev` (used by the overlapped fast path)."""
    return {k: torch.as_tensor(v, device=dev) for k, v in np_feat.items()}


def featurize(items, dev):
    return to_device(featurize_np(items), dev)


def fwd(m, feat, var_mask):
    return m(var_mask, feat["given"], feat["fac_rel"], feat["fac_arity"],
             feat["edge_var"], feat["edge_valid"], feat["var_valid"], feat["fac_valid"])


def dom_from_mask(row, csp):
    return tuple(frozenset(v for v in range(csp.d) if row[i, v] > 0.5) for i in range(csp.n))


def targets_np_from_ded(deds):
    """Build the (tgt, conflict) target arrays from precomputed exact-dedP results (a list of
    per-cell frozenset tuples). Factored out so clair.datagen.fast can compute the (expensive) dedP
    in a process pool and assemble the (cheap) arrays on the main thread — bitwise-identical to the
    serial build_targets, since ex.dedP and the worker dedP are the same pure function."""
    B = len(deds)
    tgt = np.zeros((B, N_MAX, D_MAX), np.float32)
    conflict = np.zeros((B,), np.float32)
    for bi, ded in enumerate(deds):
        if all(len(c) == 0 for c in ded):
            conflict[bi] = 1.0
        for i, cell in enumerate(ded):
            for v in cell:
                tgt[bi, i, v] = 1.0
    return tgt, conflict


def build_targets(items, ex, dev):
    deds = [ex.dedP(csp, dom) for csp, dom in items]
    tgt, conflict = targets_np_from_ded(deds)
    return torch.as_tensor(tgt, device=dev), torch.as_tensor(conflict, device=dev)


def loss_fn(sup, var_mask, var_valid, tgt, conflict, wpos=6.0, wneg=0.5, lcls=0.3, lce=0.3):
    """Dominate-dedP: asymmetric BCE (HEAVY wpos on eliminating a dedP-kept value) over alive
    candidates + conflict BCE + singleton CE where dedP pins a singleton."""
    eps = 1e-6
    alive = var_mask * var_valid.unsqueeze(-1)
    sing = (tgt.sum(-1) == 1).float() * var_valid
    tlab = tgt.argmax(-1)
    tot = 0.0
    for b, cls in sup:
        p = torch.sigmoid(b)
        bce = -(wpos * tgt * torch.log(p + eps) + wneg * (1 - tgt) * torch.log(1 - p + eps))
        bce = (bce * alive).sum() / (alive.sum() + eps)
        clsl = F.binary_cross_entropy_with_logits(cls, conflict)
        ce_all = F.cross_entropy(b.reshape(-1, b.size(-1)), tlab.reshape(-1),
                                 reduction="none").reshape(b.shape[:-1])
        ce = (ce_all * sing).sum() / (sing.sum() + eps)
        tot = tot + bce + lcls * clsl + lce * ce
    return tot / len(sup)


@torch.no_grad()
def meet(var_mask, b, theta):
    return var_mask * (torch.sigmoid(b) >= theta).float()


@torch.no_grad()
def false_elim(vm_before, vm_after, var_valid, tgt):
    elig = tgt * vm_before * var_valid.unsqueeze(-1)
    killed = elig * (vm_after < 0.5).float()
    return killed.sum().item(), elig.sum().item()


# =====================================================================================
# training (on-policy; the model's own monotone meet rolls the lattice state forward)
# =====================================================================================
def train(rungs, dev, target, steps, pool=96, R=8, lr=3e-4, theta=0.5, seed=0, ex=SHARED, log=None,
          fast=False, gen=None):
    """Train the general organ on `rungs`. The data path (exact-dedP targets + featurize + resample)
    is single-threaded; pass fast=True (or a clair.datagen.fast.FastGen via `gen`) to run the
    targets across all cores AND overlap the data prep with the GPU backward. Bitwise-identical math,
    same rng stream -> reproducible; opt-in so existing callers are unchanged."""
    if fast or gen is not None:
        from .datagen import fast as _fast
        return _fast.train(rungs, dev, target, steps, pool=pool, R=R, lr=lr, theta=theta,
                           seed=seed, ex=ex, log=log, gen=gen)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d, npar = size_for("full", N_MAX, D_MAX, M_MAX, A_MAX, target, R=R, ds=min(4, R))
    m = FactorGraphProposer("full", N_MAX, D_MAX, M_MAX, A_MAX, d=d, R=R, ds=min(4, R)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95))
    log = log if log is not None else []
    tagged = sample_corpus(rng, pool, rungs)
    items = [(c, c.full()) for _, c in tagged]
    rtags = [rg for rg, _ in tagged]
    print(f"  train rungs={rungs} d={d} params={npar:,} pool={pool} R={R} steps={steps}", flush=True)
    fe_k = fe_n = 0
    t0 = time.time()
    for s in range(1, steps + 1):
        feat = featurize(items, dev)
        vm = feat["var_mask"]
        tgt, conflict = build_targets(items, ex, dev)
        b, cls, sup = fwd(m, feat, vm)
        loss = loss_fn(sup, vm, feat["var_valid"], tgt, conflict)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        with torch.no_grad():
            new_vm = meet(vm, b, theta)
            k, n = false_elim(vm, new_vm, feat["var_valid"], tgt); fe_k += k; fe_n += n
            nvc = new_vm.cpu().numpy()
            nxt = []
            for bi, (csp, dom) in enumerate(items):
                ndom = dom_from_mask(nvc[bi], csp)
                if C.status(ndom) in ("solved", "conflict") or ndom == dom:  # terminal/stalled -> fresh
                    rg = rtags[bi]
                    nc = sample_rung_csp(rng, rg)
                    nxt.append((nc, nc.full()))
                else:
                    nxt.append((csp, ndom))
            items = nxt
        if s % max(1, steps // 12) == 0:
            fer = fe_k / max(1, fe_n)
            log.append({"step": s, "loss": float(loss.detach()), "false_elim": fer})
            print(f"    step {s:5d}  loss {float(loss.detach()):.3f}  false_elim {fer:.4f}  "
                  f"alive {float(vm.sum(-1).mean()):.2f}  {time.time()-t0:.0f}s", flush=True)
            fe_k = fe_n = 0
    return m, d, npar, log


# =====================================================================================
# evaluation (per rung: soundness + completeness vs the exact factor lattice)
# =====================================================================================
@torch.no_grad()
def run_to_fixpoint(m, csps, dev, theta=0.5, R_max=64):
    items = [(c, c.full()) for c in csps]
    feat = featurize(items, dev)
    vm = feat["var_mask"].clone()
    B = len(csps)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    fe_k = fe_n = 0
    for _ in range(R_max):
        # #3: recompute `given` each pass from the CURRENT lattice so inference matches the
        # re-featurized training distribution (newly-singleton cells are marked given).
        feat["given"] = (vm.sum(-1) == 1).float() * feat["var_valid"]
        b, cls, _ = fwd(m, feat, vm)
        new_vm = meet(vm, b, theta)
        cur = [(csps[i], dom_from_mask(vm[i].cpu().numpy(), csps[i])) for i in range(B)]
        tgt, _ = build_targets(cur, SHARED, dev)
        k, n = false_elim(vm, new_vm, feat["var_valid"], tgt); fe_k += k; fe_n += n
        changed = (new_vm != vm).any(-1).any(-1)
        vm = torch.where(done.view(-1, 1, 1), vm, new_vm)
        done = done | ~changed
        if done.all():
            break
    return [dom_from_mask(vm[i].cpu().numpy(), csps[i]) for i in range(B)], (fe_k, fe_n)


def _factor_fixpoint_dom(csp):
    """Per-cell domains at the exact level-k factor-lattice fixpoint (the operator the organ
    approximates). Replicates clair.csp.solve_factor's internal narrowing, returning the domains."""
    factors = C.default_factors(csp, 3)
    st = C.factor_init(csp, factors)
    for _ in range(csp.n * csp.d * csp.d + 2):
        nxt = C.factor_step(csp, st, factors)
        if nxt == st:
            break
        st = nxt
    return C.factor_cells(csp, st, factors)


def _removed(full, dom, valid_n):
    """Set of (cell,value) the operator eliminated from the full grid (over real cells)."""
    return {(i, v) for i in range(valid_n) for v in full[i] if v not in dom[i]}


@torch.no_grad()
def evaluate_rung(m, csps, dev, theta=0.5, R_max=64):
    """Per-rung soundness + GRADED completeness. Because most instances are under-determined
    (the exact ceiling is far below 100% 'all-singleton'), the primary completeness metric is
    NARROWING-RECALL: of the (cell,value) eliminations the EXACT dedP makes, what fraction does the
    organ also make (sound => its eliminations are a subset, modulo the tracked false-elim). 1.0 =
    the organ reaches the strongest sound per-cell fixpoint. We also report exact factor-fixpoint /
    dedP-fixpoint MATCH rates and the (harsher) all-singleton 'solved' rate."""
    m.eval()
    doms, (fe_k, fe_n) = run_to_fixpoint(m, csps, dev, theta, R_max)
    n_solv = solved = wrong = openab = 0
    match_ded = match_fac = uniq = 0
    rec_num = rec_den = 0          # micro narrowing-recall vs dedP
    fac_num = fac_den = 0          # micro narrowing-recall vs factor fixpoint
    for csp, dom in zip(csps, doms):
        sols = C.solutions(csp)
        if not sols:
            continue
        n_solv += 1
        full = csp.full()
        ded, _ = C.to_fixpoint(C.exact_dedP, csp, csp.full())
        fac = _factor_fixpoint_dom(csp)
        rm_model = _removed(full, dom, csp.n)
        rm_ded = _removed(full, ded, csp.n)
        rm_fac = _removed(full, fac, csp.n)
        rec_num += len(rm_model & rm_ded); rec_den += len(rm_ded)
        fac_num += len(rm_model & rm_fac); fac_den += len(rm_fac)
        match_ded += int(dom == ded)
        match_fac += int(dom == fac)
        uniq += int(all(len(c) == 1 for c in ded))      # instance is uniquely determined
        st = C.status(dom)
        is_solved = st == "solved"
        solved += is_solved
        wrong += int(is_solved and tuple(next(iter(dom[i])) for i in range(csp.n)) not in sols)
        openab += int(st == "open")
    m.train()
    ns = max(1, n_solv)
    return {"n_solvable": n_solv,
            "narrowing_recall": rec_num / max(1, rec_den),     # PRIMARY completeness (vs dedP)
            "factor_recall": fac_num / max(1, fac_den),        # completeness vs factor fixpoint
            "match_dedP": match_ded / ns,                      # reaches exact per-cell fixpoint
            "match_factor": match_fac / ns,                    # reaches exact factor fixpoint
            "uniq_rate": uniq / ns,                            # how many instances are determinable
            "solved": solved / ns,                             # all-singleton (only on determinable)
            "abstain_rate": openab / ns,
            "wrong_return_rate": wrong / ns,
            "false_elim": fe_k / max(1, fe_n), "false_elim_count": int(fe_k)}


def eval_all(m, eval_sets, dev, theta, R_max):
    return {rg: evaluate_rung(m, csps, dev, theta, R_max) for rg, csps in eval_sets.items()}


def _row(rg, e):
    return (f"  {rg:11s} n={e['n_solvable']:3d}  recall(dedP) {e['narrowing_recall']*100:5.1f}%  "
            f"recall(factor) {e['factor_recall']*100:5.1f}%  match-fix {e['match_dedP']*100:5.1f}%  "
            f"solved {e['solved']*100:5.1f}%(uniq {e['uniq_rate']*100:4.1f}%)  | "
            f"FALSE-ELIM {e['false_elim']:.4f} ({e['false_elim_count']})  wrong {e['wrong_return_rate']*100:.1f}%")


# =====================================================================================
# experiments
# =====================================================================================
def make_eval_sets(rungs, neval, seed=12345):
    rng = np.random.default_rng(seed)
    return {rg: [sample_rung_csp(rng, rg) for _ in range(neval)] for rg in rungs}


def _maybe_gen(args):
    """One shared FastGen (persistent process pool) reused across every train() call when --fast."""
    if getattr(args, "fast", False):
        from .datagen import fast as _fast
        return _fast.FastGen(workers=getattr(args, "workers", None) or None,
                             cache_path=getattr(args, "cache", None) or None)
    return None


def run_full(args, dev):
    rungs = RUNGS
    eval_sets = make_eval_sets(rungs, args.neval)
    out = {"budget": [N_MAX, D_MAX, M_MAX, A_MAX], "rungs": rungs, "args": vars(args)}
    gen = _maybe_gen(args)

    # ---- MULTITASK: one organ, all rungs ----
    print("\n===== MULTITASK (train all rungs, eval per rung) =====", flush=True)
    m, d, npar, log = train(rungs, dev, args.target, args.steps, pool=args.pool, R=args.R,
                            lr=args.lr, theta=args.theta, seed=args.seed, gen=gen)
    mt = eval_all(m, eval_sets, dev, args.theta, args.rmax)
    out["multitask"] = {"d_model": d, "params": npar, "log": log, "eval": mt}
    for rg in rungs:
        print(_row(rg, mt[rg]), flush=True)

    # ---- SINGLE-TASK: train rung r, eval rung r ----
    print("\n===== SINGLE-TASK (train r, eval r) =====", flush=True)
    out["single"] = {}
    for rg in rungs:
        ms, _, _, _ = train([rg], dev, args.target, args.steps_single, pool=args.pool, R=args.R,
                            lr=args.lr, theta=args.theta, seed=args.seed, gen=gen)
        es = evaluate_rung(ms, eval_sets[rg], dev, args.theta, args.rmax)
        out["single"][rg] = es
        print(_row(rg, es), flush=True)

    # ---- HELD-OUT: train all-but-r, eval r ZERO-SHOT ----
    print("\n===== HELD-OUT (train all-but-r, eval r ZERO-SHOT) =====", flush=True)
    out["heldout"] = {}
    for rg in rungs:
        others = [x for x in rungs if x != rg]
        mh, _, _, _ = train(others, dev, args.target, args.steps, pool=args.pool, R=args.R,
                            lr=args.lr, theta=args.theta, seed=args.seed, gen=gen)
        eh = evaluate_rung(mh, eval_sets[rg], dev, args.theta, args.rmax)
        out["heldout"][rg] = eh
        print(_row(rg, eh), flush=True)

    # ---- transfer summary table (narrowing-recall vs dedP = completeness; FE = soundness) ----
    print("\n===== TRANSFER SUMMARY  (narrowing-recall vs exact dedP; soundness=false-elim) =====", flush=True)
    print(f"  {'rung':11s} {'single':>8s} {'multi':>8s} {'heldout':>9s}   {'multi-FE':>9s} {'held-FE':>9s}", flush=True)
    for rg in rungs:
        s, mu, h = out["single"][rg], mt[rg], out["heldout"][rg]
        print(f"  {rg:11s} {s['narrowing_recall']*100:7.1f}% {mu['narrowing_recall']*100:7.1f}% "
              f"{h['narrowing_recall']*100:8.1f}%   {mu['false_elim']:9.4f} {h['false_elim']:9.4f}", flush=True)
    if gen is not None:
        gen.close()
    return out


def run_smoke(args, dev):
    print("\n===== SMOKE (general deductor trains; false-elim drops on a few rungs) =====", flush=True)
    rungs = ["coloring", "arithmetic", "smt"]
    eval_sets = make_eval_sets(rungs, 80)
    gen = _maybe_gen(args)
    m, d, npar, log = train(rungs, dev, target=1.2e5, steps=args.steps or 250,
                            pool=64, R=args.R, lr=args.lr, theta=args.theta, seed=0, gen=gen)
    if gen is not None:
        gen.close()
    ev = eval_all(m, eval_sets, dev, args.theta, args.rmax)
    print(f"  false_elim trajectory: {[round(x['false_elim'], 4) for x in log]}", flush=True)
    for rg in rungs:
        print(_row(rg, ev[rg]), flush=True)
    return {"smoke": {"log": log, "eval": ev}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--target", type=float, default=2.0e5)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--steps_single", type=int, default=600)
    ap.add_argument("--pool", type=int, default=96)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--neval", type=int, default=200)
    ap.add_argument("--rmax", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true",
                    help="parallel (all-core) dedP targets + GPU-overlapped data path (clair.datagen.fast)")
    ap.add_argument("--workers", type=int, default=0, help="pool workers for --fast (0=os.cpu_count)")
    ap.add_argument("--cache", default=None, help="on-disk dedP target cache (sqlite path), reused across runs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = device()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    print(f"device={dev}  budget N={N_MAX} D={D_MAX} M={M_MAX} A={A_MAX}", flush=True)
    out = run_smoke(args, dev) if args.mode == "smoke" else run_full(args, dev)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
        print("\nwrote", args.out, flush=True)


if __name__ == "__main__":
    main()
