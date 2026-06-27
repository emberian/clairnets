"""FROZEN-LEARNED-ORGAN readout — the clean complement to the oracle-readout de-risk.

The oracle de-risk (clair.oracle_readout / run_oracle) proved OLMo's LM head can READ the EXACT
narrowed lattice (clair.csp.exact_dedP) through a structured zero-init gamma: true=100, controls
drop sharply, OOD-perfect, corrupt->0. The full CO-TRAINED woven model (box1) FAILED (organ ignored).

OPEN QUESTION this isolates: is that failure (a) co-training distrust of a noisy, moving organ, or
(b) learned-organ imperfection itself?  Method: TRAIN a FactorGraphProposer organ STANDALONE to high
recall + low false-elim, FREEZE it, then inject ITS narrowed lattice through the SAME structured
gamma as the oracle de-risk and train LoRA+gamma generatively, with the SAME causal controls + OOD.

  * If the frozen GOOD learned organ reads NEARLY as well as the oracle  -> the original failure was
    CO-TRAINING distrust; the fix is bootstrap-then-freeze (or a slow organ LR).
  * If it reads POORLY despite the organ being accurate                  -> the issue is learned-organ
    noise / distributional mismatch in the lattice the LM must read, not co-training.

This module: (1) the standalone organ trainer (proposer.FactorGraphProposer, dominate-dedP, on-policy,
coloring, budget sized to cover OOD N=11), with a FAST exact coloring dedP so we can supervise up to
N=11 without 3**11 enumeration; (2) frozen-organ lattice extraction (run to fixpoint -> per-cell
surviving candidate set), matched to the oracle's surv[n,K] schema so clair.oracle_readout's gamma /
readout / controls are reused WHOLESALE.  The runner is clair.run_frozen.
"""
from __future__ import annotations

import itertools as it

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import csp as C
from . import curriculum as CU
from .proposer import FactorGraphProposer, size_for


# ===================================================================== budget (coloring, OOD to N=11)
# Coloring is arity<=2 (pin=1, neq=2), k=3 colors. N_MAX=12 covers the OOD N=11 test; M_MAX is a safe
# cap on (edges+pins) at N=11. The proposer is permutation+size equivariant, so one set of weights
# runs at every N up to the pad budget.
N_MAX, D_MAX, M_MAX, A_MAX = 12, 3, 80, 2
REL_DIM = D_MAX ** A_MAX


# ===================================================================== FAST exact coloring dedP
# exact_dedP(csp, dom) = {values used by SOME solution consistent with dom}, per cell. Routed to the
# ONE canonical deductor (clair.csp.exact_dedP: Rust clair_fast port when built, else the pure-Python
# clair.csp._exact_dedP_py). This module's old per-(cell,value)-SAT copy + its `_sat` helper were
# DELETED after being verified cell-for-cell equal to exact_dedP.
def fast_dedP(csp: C.CSP, dom) -> tuple:
    return C.exact_dedP(csp, dom)


class DedPCache:
    """Memoize the exact dedP target (keyed by constraints + current domain). Shared across steps."""
    def __init__(self):
        self.c = {}

    def __call__(self, csp, dom):
        key = (csp.cons, csp.d, dom)
        v = self.c.get(key)
        if v is None:
            v = fast_dedP(csp, dom)
            self.c[key] = v
        return v


# ===================================================================== coloring CSP sampling
def coloring_csp(rng, N, k=3, edge_p=0.45, pin_frac=0.35) -> C.CSP:
    """One solvable coloring CSP of EXACTLY N cells (witness-first via curriculum.gen_coloring)."""
    n, d, _, facts, _ = CU.gen_coloring(rng, k=k, n_lo=N, n_hi=N, edge_p=edge_p, pin_frac=pin_frac)
    return CU.build_csp(n, d, facts)


# ===================================================================== featurization (this budget)
def relation_table(scope, al):
    a = len(scope)
    t = np.zeros((D_MAX,) * A_MAX, dtype=np.float32)
    for tup in al:
        sl = [slice(None)] * A_MAX
        for p in range(a):
            sl[p] = tup[p]
        t[tuple(sl)] = 1.0
    return t.reshape(-1)


def featurize(items, dev):
    """items = list of (csp, dom). Returns the batched proposer input dict (padded to the budget)."""
    B = len(items)
    var_mask = np.zeros((B, N_MAX, D_MAX), np.float32)
    given = np.zeros((B, N_MAX), np.float32)
    var_valid = np.zeros((B, N_MAX), np.float32)
    fac_rel = np.zeros((B, M_MAX, REL_DIM), np.float32)
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
            fac_rel[bi, fi] = relation_table(sc, al)
            for p, cell in enumerate(sc):
                edge_var[bi, fi, p] = cell
                edge_valid[bi, fi, p] = 1.0
    t = lambda a: torch.as_tensor(a, device=dev)
    return dict(var_mask=t(var_mask), given=t(given), var_valid=t(var_valid),
                fac_rel=t(fac_rel), fac_arity=t(fac_arity), fac_valid=t(fac_valid),
                edge_var=t(edge_var), edge_valid=t(edge_valid))


def fwd(m, feat, var_mask):
    return m(var_mask, feat["given"], feat["fac_rel"], feat["fac_arity"],
             feat["edge_var"], feat["edge_valid"], feat["var_valid"], feat["fac_valid"])


def dom_from_mask(row, csp):
    return tuple(frozenset(v for v in range(csp.d) if row[i, v] > 0.5) for i in range(csp.n))


# ===================================================================== targets / loss (dominate-dedP)
def build_targets(items, dedp, dev):
    B = len(items)
    tgt = np.zeros((B, N_MAX, D_MAX), np.float32)
    conflict = np.zeros((B,), np.float32)
    for bi, (csp, dom) in enumerate(items):
        ded = dedp(csp, dom)
        if all(len(c) == 0 for c in ded):
            conflict[bi] = 1.0
        for i in range(csp.n):
            for v in ded[i]:
                tgt[bi, i, v] = 1.0
    return torch.as_tensor(tgt, device=dev), torch.as_tensor(conflict, device=dev)


def loss_fn(sup, var_mask, var_valid, tgt, conflict, wpos=6.0, wneg=0.5, lcls=0.3, lce=0.3):
    """Asymmetric BCE — HEAVY penalty (wpos) on eliminating a dedP-kept value (=> soundness) + light
    pressure to drop the rest (=> completeness) + conflict BCE + singleton CE where dedP pins."""
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


# ===================================================================== standalone organ training
def train_organ(dev, train_ns, target=3.0e5, steps=1500, pool=96, R=8, lr=3e-4, theta=0.5, seed=0,
                edge_p=0.45, pin_frac=0.35, log_every=12):
    """On-policy dominate-dedP training of the coloring organ across N in train_ns. The model's own
    monotone meet rolls the lattice forward; terminal/stalled instances are replaced with fresh ones
    (matched N). Returns (model, d_model, n_params, log)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dedp = DedPCache()
    d, npar = size_for("full", N_MAX, D_MAX, M_MAX, A_MAX, target, R=R, ds=min(4, R))
    m = FactorGraphProposer("full", N_MAX, D_MAX, M_MAX, A_MAX, d=d, R=R, ds=min(4, R)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95))
    log = []
    pool_ns = [int(rng.choice(train_ns)) for _ in range(pool)]
    items = [(coloring_csp(rng, nn_, edge_p=edge_p, pin_frac=pin_frac), None) for nn_ in pool_ns]
    items = [(c, c.full()) for c, _ in items]
    print(f"  ORGAN train: N in {train_ns}  d_model={d}  params={npar:,}  pool={pool}  R={R} "
          f"steps={steps}", flush=True)
    fe_k = fe_n = 0
    import time
    t0 = time.time()
    for s in range(1, steps + 1):
        feat = featurize(items, dev)
        vm = feat["var_mask"]
        tgt, conflict = build_targets(items, dedp, dev)
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
                if C.status(ndom) in ("solved", "conflict") or ndom == dom:
                    nn_ = int(rng.choice(train_ns))
                    nc = coloring_csp(rng, nn_, edge_p=edge_p, pin_frac=pin_frac)
                    nxt.append((nc, nc.full()))
                else:
                    nxt.append((csp, ndom))
            items = nxt
        if s % max(1, steps // log_every) == 0 or s == 1:
            fer = fe_k / max(1, fe_n)
            log.append({"step": s, "loss": float(loss.detach()), "false_elim": fer})
            print(f"    step {s:5d}  loss {float(loss.detach()):.3f}  false_elim {fer:.4f}  "
                  f"alive {float(vm.sum(-1).mean()):.2f}  {time.time()-t0:.0f}s", flush=True)
            fe_k = fe_n = 0
    return m, d, npar, log


# ===================================================================== frozen-organ lattice extraction
@torch.no_grad()
def organ_fixpoint(m, csps, dev, theta=0.5, R_max=64, chunk=128):
    """Run the FROZEN organ's monotone meet to a fixpoint on each csp (from the full domain). Returns
    a list of per-cell domain tuples (the LEARNED narrowed lattice, the analogue of the oracle's
    exact_dedP fixpoint)."""
    m.eval()
    out = [None] * len(csps)
    for c0 in range(0, len(csps), chunk):
        sub = csps[c0:c0 + chunk]
        items = [(c, c.full()) for c in sub]
        feat = featurize(items, dev)
        vm = feat["var_mask"].clone()
        B = len(sub)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        for _ in range(R_max):
            b, cls, _ = fwd(m, feat, vm)
            new_vm = meet(vm, b, theta)
            changed = (new_vm != vm).any(-1).any(-1)
            vm = torch.where(done.view(-1, 1, 1), vm, new_vm)
            done = done | ~changed
            if bool(done.all()):
                break
        nv = vm.cpu().numpy()
        for j, c in enumerate(sub):
            out[c0 + j] = dom_from_mask(nv[j], c)
    return out


def surv_from_dom(dom, n, K) -> np.ndarray:
    surv = np.zeros((n, K), dtype=np.float32)
    for i in range(n):
        for v in dom[i]:
            if v < K:
                surv[i, v] = 1.0
    return surv


def _removed(full, dom, n):
    return {(i, v) for i in range(n) for v in full[i] if v not in dom[i]}


def recall_fe(organ_dom, csp, dedp) -> tuple:
    """Per-instance narrowing recall + false-elim of the organ fixpoint vs the EXACT dedP fixpoint
    (= oracle surv). recall = |organ_removed & oracle_removed| / |oracle_removed|;  fe (unsound)
    eliminations = |organ_removed \\ oracle_removed|."""
    full = csp.full()
    oracle = dedp(csp, full)
    rm_o = _removed(full, organ_dom, csp.n)
    rm_t = _removed(full, oracle, csp.n)
    rec = len(rm_o & rm_t) / max(1, len(rm_t))
    fe = len(rm_o - rm_t)
    return rec, fe, oracle
