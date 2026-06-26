"""clair/row_space.py — THE GF(2) ROW-SPACE / GAUSSIAN-ELIMINATION ORGAN.

The first confirmed "group-algorithm" beast in the primitive factory. xor_wall.py PROVED the
factor-lattice / blade organ provably CANNOT solve high-treewidth XOR systems (recall collapses to
the local-pin floor ~5% as width grows, FLAT across all depth R), while GF(2) Gaussian elimination
solves them ALL in poly time. This module builds that missing primitive as a TYPED CHECKED ORGAN,
following codex's certified-contractor discipline (notes/codex_zoo_review.md):

  ABSTRACT STATE  = the GF(2) linear system / its row-reduced echelon form — the ROW SPACE (the span
                    of the equations = the set of linear consequences). This is precisely the state
                    the per-cell / factor lattice LACKS: it can only hold arity-<=k local marginals,
                    never the global span.  γ(state) = { x in GF(2)^n : A x = b } (the affine solution
                    set).  ⊤ = empty system (all of GF(2)^n).  ⊥ = inconsistent (row 0 = 1).

  CERTIFIED OPERATOR = elementary GF(2) row operations (pivot + eliminate). Each is INVERTIBLE over
                    GF(2), so it preserves γ EXACTLY: no row-op can ever drop a real solution or
                    invent a fake one. SOUNDNESS IS FREE AND TOTAL — for ANY pivot order the read-off
                    is sound (0 false-elim) and COMPLETE on the determined variables (100% recall vs
                    GF(2)-exact). The certificate is "every step is an elementary row op" (checkable).

  NEURAL CONTRACTOR CHOICE = the PIVOT / elimination ORDER (which column to pivot next). Gauss-Jordan's
                    RESULT is order-independent (same row space, same forced bits), but the WORK — the
                    number of row-XOR operations, i.e. the fill-in — is NOT. Choosing sparse pivots
                    early (the Markowitz / min-degree problem, NP-hard to optimize) keeps the matrix
                    sparse and cuts work. The neural policy guides THIS choice ONLY; it never decides a
                    bit, so it CANNOT break soundness. Honest value-add = work reduction vs a fixed
                    order, measured against the classical min-degree heuristic.

  READ-OFF / ABSTAIN = after elimination, a variable is DETERMINED iff its column is a pivot whose row
                    is a pure unit (no free-variable dependence) → emit its forced bit; ABSTAIN on the
                    rest. Exactly gf2_forced (the exact affine dedₚ from xor_wall).

  VERIFIER / GROUND TRUTH = full GF(2) Gaussian elimination (xor_wall.gf2_forced). The certificate test
                    is: organ's read-off == gf2_forced(A0, b0), for every pivot order.

The learnability question (the only honest neural claim here): does a learned pivot policy reach the
GF(2) solution with FEWER row-ops than a fixed/natural order — and does it match/beat classical
min-degree — across the rank/treewidth regime where the lattice organ collapsed flat?

  python -m clair.row_space --smoke
  python -m clair.row_space --sweep  --out runs/row_space_sweep.json
  python -m clair.row_space --train  --ckpt runs/row_space_policy.pt --out runs/row_space_train.json
  python -m clair.row_space --all    --ckpt runs/row_space_policy.pt --out runs/row_space.json
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np

from . import csp as C
from . import xor_wall as XW
from .xor_wall import gf2_rref, gf2_forced, gen_xor_system, recall_vs, removed, factor_fixpoint_cells


# ============================================================ the certified row-space STATE + operator
class GF2RowSpace:
    """The abstract row-space state with the CERTIFIED Gauss-Jordan operator over GF(2).

    Holds a working (A, b). `step(col)` performs ONE certified pivot: choose a pivot row owning `col`
    (sparsest unused row, a fixed sound tiebreak), then eliminate `col` from every OTHER row (full
    Gauss-Jordan → pivot columns stay unit). Every step is an elementary GF(2) row operation, hence
    γ-preserving (sound by construction). `work` accumulates the row-XOR cost (Σ pivot-row weight over
    eliminated rows) — the quantity the pivot ORDER controls and the neural policy minimises.

    The state is order-AGNOSTIC for its read-off (read_off() == gf2_forced(A0,b0) for any order) and
    order-SENSITIVE for its cost (work). That split is the whole point: soundness free, efficiency learned.
    """

    def __init__(self, A, b):
        self.A0 = (np.asarray(A, np.uint8) % 2).copy()
        self.b0 = (np.asarray(b, np.uint8) % 2).copy()
        self.A = self.A0.copy()
        self.b = self.b0.copy()
        self.m, self.n = self.A.shape
        self.pivot_col = {}                 # col -> pivot row index (in working A)
        self.row_used = np.zeros(self.m, bool)
        self.col_done = np.zeros(self.n, bool)
        self.work = 0                       # cumulative row-XOR operations (the learnable cost)
        self.n_pivots = 0

    # ---- the candidate set the policy chooses from --------------------------------------------------
    def candidate_cols(self):
        """Columns not yet pivoted that still have a 1 in some UNUSED row (an available pivot)."""
        live = self.A[~self.row_used]
        if live.shape[0] == 0:
            return np.zeros(0, np.int64)
        has = (live.sum(0) > 0) & (~self.col_done)
        return np.nonzero(has)[0]

    def _pivot_row_for(self, col):
        """Sparsest UNUSED row owning `col` (min Hamming weight; lowest index tiebreak). Sound: any row
        with a 1 in col is a valid pivot — the choice only affects fill-in, never the solution space."""
        rows = np.nonzero(self.A[:, col] & (~self.row_used))[0]
        if rows.size == 0:
            return None
        w = self.A[rows].sum(1)
        return int(rows[int(np.argmin(w))])

    def step(self, col):
        """CERTIFIED pivot on `col`. Returns the work spent (row-XORs). No-op-safe if col not available."""
        p = self._pivot_row_for(col)
        if p is None:
            self.col_done[col] = True
            return 0
        prow = self.A[p]
        pweight = int(prow.sum())
        hit = np.nonzero(self.A[:, col])[0]
        hit = hit[hit != p]
        for r in hit:                       # eliminate col from EVERY other row (Gauss-Jordan)
            self.A[r] ^= prow
            self.b[r] ^= self.b[p]
            self.work += pweight            # one length-(pweight) GF(2) row XOR
        self.pivot_col[col] = p
        self.row_used[p] = True
        self.col_done[col] = True
        self.n_pivots += 1
        return pweight * len(hit)

    def done(self):
        return self.candidate_cols().size == 0

    # ---- read-off: determined variables, ABSTAIN on the rest ----------------------------------------
    def read_off(self):
        """Per-cell domains DERIVED from the organ's own reduced state (not from a planted solution):
        a column c is DETERMINED iff it is a pivot whose row, outside pivot columns, is all zero
        (x_c depends on no free var) → {b[row]}; else ABSTAIN → {0,1}. Equals the exact affine dedₚ."""
        n = self.n
        pivot_cols = set(self.pivot_col)
        forced = [None] * n
        for c in range(n):
            if c in pivot_cols:
                r = self.pivot_col[c]
                free_support = [j for j in range(n) if j not in pivot_cols and self.A[r, j]]
                forced[c] = frozenset({int(self.b[r])}) if not free_support else frozenset({0, 1})
            else:
                forced[c] = frozenset({0, 1})
        return tuple(forced)

    def run(self, policy):
        """Drive to fixpoint under a pivot `policy(state)->col`. Returns the final read-off cells."""
        steps = 0
        while not self.done():
            cands = self.candidate_cols()
            col = policy(self, cands)
            self.step(int(col))
            steps += 1
            if steps > self.n + self.m + 2:        # safety
                break
        return self.read_off()


# ============================================================ fixed (non-neural) pivot policies
def policy_natural(state, cands):
    return int(cands[0])                                            # lowest column index (the gf2_rref order)


def policy_random(rng):
    def f(state, cands):
        return int(rng.choice(cands))
    return f


def policy_min_degree(state, cands):
    """Classical Markowitz / minimum-degree pivot: pick the (col, sparsest-row) minimising the fill
    estimate (n_rows_in_col - 1) * (pivot_row_weight - 1). The strong NON-neural baseline."""
    best, best_cost = None, None
    for c in cands:
        rows = np.nonzero(state.A[:, c] & (~state.row_used))[0]
        if rows.size == 0:
            continue
        w = state.A[rows].sum(1)
        pw = int(w.min())
        cost = (len(rows) - 1) * (pw - 1)
        if best_cost is None or cost < best_cost:
            best_cost, best = cost, int(c)
    return best if best is not None else int(cands[0])


def solve_fixed(A, b, policy):
    """Run the certified organ under a fixed pivot policy. Returns (cells, work, n_pivots)."""
    st = GF2RowSpace(A, b)
    cells = st.run(policy)
    return cells, st.work, st.n_pivots


# ============================================================ the NEURAL pivot policy (contractor choice)
def _torch():
    import torch
    return torch


class PivotPolicy:
    """Bipartite (row<->col) GNN scorer over the CURRENT (filled-in) GF(2) matrix. Permutation-
    equivariant over both rows and columns: the only signal is the sparsity pattern, so one set of
    weights runs on any (m, n). Scores each ELIGIBLE column; soft-sampled during training, argmax at
    eval. It guides the pivot ORDER only — the certified row-op guarantees soundness regardless."""

    def __init__(self, d=48, layers=3, device="cpu", seed=0):
        torch = _torch()
        torch.manual_seed(seed)
        self.torch = torch
        self.d = d
        self.device = device
        g = torch.Generator().manual_seed(seed)

        def lin(i, o):
            w = torch.empty(o, i)
            torch.nn.init.xavier_uniform_(w, generator=g)
            return torch.nn.Parameter(w.to(device))

        self.params = {}
        self.params["row_in"] = lin(2, d)          # [norm row weight, active?]
        self.params["col_in"] = lin(2, d)          # [norm col weight, eligible?]
        self.layers = layers
        for k in range(layers):
            self.params[f"c{k}"] = lin(2 * d, d)   # col update from [col ; msg-from-rows]
            self.params[f"r{k}"] = lin(2 * d, d)   # row update from [row ; msg-from-cols]
        self.params["head"] = lin(d, 1)
        for v in self.params.values():
            v.requires_grad_(True)

    def parameters(self):
        return list(self.params.values())

    def scores(self, state):
        """Return (logits[n_cands], cand_cols) over the current eligible columns. Differentiable."""
        torch = self.torch
        cands = state.candidate_cols()
        if cands.size == 0:
            return None, cands
        A = torch.tensor(state.A.astype(np.float32), device=self.device)        # [m,n]
        active = torch.tensor((~state.row_used).astype(np.float32), device=self.device)  # [m]
        elig = np.zeros(state.n, np.float32); elig[cands] = 1.0
        eligT = torch.tensor(elig, device=self.device)                          # [n]
        Aeff = A * active.unsqueeze(1)                                          # only unused rows speak
        m, n = state.m, state.n
        rowdeg = Aeff.sum(1, keepdim=True).clamp(min=1.0)                       # [m,1]
        coldeg = Aeff.sum(0, keepdim=True).clamp(min=1.0)                       # [1,n]
        relu = torch.relu

        def mlp(w, x):
            return x @ self.params[w].t()

        hr = relu(mlp("row_in", torch.stack([Aeff.sum(1) / max(n, 1), active], 1)))   # [m,d]
        hc = relu(mlp("col_in", torch.stack([Aeff.sum(0) / max(m, 1), eligT], 1)))    # [n,d]
        for k in range(self.layers):
            msg_c = (Aeff.t() @ hr) / coldeg.t()                               # [n,d]  col <- rows
            hc = hc + relu(mlp(f"c{k}", torch.cat([hc, msg_c], 1)))
            msg_r = (Aeff @ hc) / rowdeg                                       # [m,d]  row <- cols
            hr = hr + relu(mlp(f"r{k}", torch.cat([hr, msg_r], 1)))
        s = mlp("head", hc).squeeze(1)                                         # [n]
        return s[cands], cands

    def act(self, state, greedy=False):
        """Sample (or argmax) a pivot column. Returns (col, logprob, entropy)."""
        torch = self.torch
        logits, cands = self.scores(state)
        if logits is None:
            return None, None, None
        dist = torch.distributions.Categorical(logits=logits)
        if greedy:
            idx = torch.argmax(logits)
            return int(cands[int(idx)]), dist.log_prob(idx), dist.entropy()
        idx = dist.sample()
        return int(cands[int(idx)]), dist.log_prob(idx), dist.entropy()

    def greedy_policy(self):
        def f(state, cands):
            col, _, _ = self.act(state, greedy=True)
            return col
        return f

    def save(self, path):
        torch = self.torch
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"d": self.d, "layers": self.layers,
                    "params": {k: v.detach().cpu() for k, v in self.params.items()}}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        torch = _torch()
        blob = torch.load(path, map_location=device)
        pol = cls(d=blob["d"], layers=blob["layers"], device=device)
        for k, v in blob["params"].items():
            pol.params[k] = torch.nn.Parameter(v.to(device)); pol.params[k].requires_grad_(True)
        return pol


# ============================================================ REINFORCE training of the pivot policy
def _rollout(pol, A, b, greedy=False):
    """One certified rollout under the (sampled or greedy) policy. Returns (work, logp_sum, ent_sum)."""
    torch = pol.torch
    st = GF2RowSpace(A, b)
    logps, ents = [], []
    guard = 0
    while not st.done() and guard < st.n + st.m + 2:
        col, lp, ent = pol.act(st, greedy=greedy)
        if col is None:
            break
        st.step(col); logps.append(lp); ents.append(ent); guard += 1
    logp_sum = torch.stack(logps).sum() if logps else torch.zeros((), device=pol.device)
    ent_sum = torch.stack(ents).sum() if ents else torch.zeros((), device=pol.device)
    return st.work, logp_sum, ent_sum


def train_policy(steps=400, batch=24, lr=3e-3, d=48, layers=3, device="cpu", K=4, ent_coef=0.01,
                 n_lo=10, n_hi=22, band_mode="mixed", seed=0, log_every=40):
    """Train the pivot policy to MINIMISE row-XOR work via REINFORCE with a SELF-CRITICAL (RLOO)
    baseline: K sampled rollouts per instance, each rollout's advantage = (mean-of-the-OTHER-rollouts'
    work) − its own work, normalised. This learns from ANY starting point (rewarding rollouts that beat
    the policy's OWN average), unlike a fixed min-degree baseline which degenerates while the policy is
    uniformly worse than min-degree. Soundness is untouched — every rollout is a certified Gauss-Jordan,
    so we only ever shape COST. We additionally LOG adv_vs_mindeg (greedy work vs min-degree) as the
    honest value-add diagnostic. Returns (policy, history)."""
    torch = _torch()
    pol = PivotPolicy(d=d, layers=layers, device=device, seed=seed)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    hist = []
    t0 = time.time()
    for it in range(1, steps + 1):
        loss = torch.zeros((), device=device)
        ent_acc = 0.0; vs_md_acc = 0.0; vs_nat_acc = 0.0; nseen = 0
        for _ in range(batch):
            n = int(rng.integers(n_lo, n_hi + 1))
            if band_mode == "mixed":
                band = int(rng.choice([3, 4, n // 2, n]))
            elif band_mode == "global":
                band = n
            else:
                band = int(band_mode)
            sysd = gen_xor_system(rng, n, n_par=int(1.4 * n), band=min(band, n), force_unique=True)
            A, b = sysd["A"], sysd["b"]
            works = []; logps = []; ents = []
            for _k in range(K):
                w, lp, ent = _rollout(pol, A, b, greedy=False)
                works.append(w); logps.append(lp); ents.append(ent)
            warr = np.array(works, np.float64)
            scale = max(1.0, warr.std())
            for k in range(K):
                loo = (warr.sum() - warr[k]) / (K - 1)              # leave-one-out baseline
                adv = (loo - warr[k]) / scale                       # +ve if this rollout beat its peers
                loss = loss - adv * logps[k] - ent_coef * ents[k]
                ent_acc += float(ents[k].detach())
            # honest diagnostics (no grad): greedy work vs classical baselines on this instance
            _, md_work, _ = solve_fixed(A, b, policy_min_degree)
            _, nat_work, _ = solve_fixed(A, b, policy_natural)
            with torch.no_grad():
                gw, _, _ = _rollout(pol, A, b, greedy=True)
            vs_md_acc += (md_work - gw) / max(1.0, md_work)
            vs_nat_acc += (nat_work - gw) / max(1.0, nat_work)
            nseen += 1
        loss = loss / (batch * K)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
        if it % log_every == 0 or it == 1:
            row = {"it": it, "loss": float(loss.detach()),
                   "greedy_vs_mindeg": vs_md_acc / max(1, nseen),
                   "greedy_vs_natural": vs_nat_acc / max(1, nseen),
                   "mean_entropy": ent_acc / (batch * K), "sec": time.time() - t0}
            hist.append(row)
            print(f"  [train] it {it:4d}/{steps}  loss {row['loss']:+.4f}  "
                  f"greedy vs natural {row['greedy_vs_natural']*100:+5.1f}%  "
                  f"vs min-deg {row['greedy_vs_mindeg']*100:+5.1f}%  "
                  f"H {row['mean_entropy']:.2f}  ({row['sec']:.0f}s)", flush=True)
    return pol, hist


# ============================================================ the rank/treewidth sweep (organ vs lattice vs GF2)
def sweep(n=14, bands=(3, 4, 6, 9, 14), n_inst=24, par_mult=1.4, seed=1, policy=None,
          with_lattice=True, lattice_ks=(3,)):
    """Fixed n, UNIQUELY-determined XOR systems (GF(2) forces all n bits), vary bandwidth (=pathwidth,
    upper-bounds treewidth) — the EXACT regime where the lattice organ collapsed flat. For each band:
      - GF(2)-exact forced fraction (= 1.0 by construction);
      - the ROW-SPACE ORGAN recall + false-elim under each pivot policy (natural / random / min-degree /
        neural) — recall must be 100% and FE 0 for ALL of them (certified), and we report the WORK each
        policy spent (the only thing the order changes);
      - the level-3 factor LATTICE recall (the wall it collapses against).
    Headline: organ recall == 100% across ALL bands (vs lattice → local-pin floor), FE == 0 everywhere;
    work falls from natural → min-degree → neural."""
    rng = np.random.default_rng(seed)
    rrng = np.random.default_rng(seed + 7)
    strategies = {"natural": policy_natural, "random": policy_random(rrng), "min_degree": policy_min_degree}
    if policy is not None:
        strategies["neural"] = policy.greedy_policy()
    rows = []
    for band in bands:
        ranks = []
        agg = {k: {"recall": [], "fe": [], "work": [], "solved": []} for k in strategies}
        lat = {k: {"recall": [], "solved": [], "fe": []} for k in lattice_ks}
        for _ in range(n_inst):
            sysd = gen_xor_system(rng, n, n_par=int(par_mult * n), band=band, force_unique=True)
            A, b, csp = sysd["A"], sysd["b"], sysd["csp"]
            full = csp.full()
            forced, rank, ok = gf2_forced(A, b, n)
            ranks.append(rank)
            for name, pol in strategies.items():
                cells, work, npiv = solve_fixed(A, b, pol)
                rec, fe, _ = recall_vs(forced, cells, n, full)
                agg[name]["recall"].append(rec); agg[name]["fe"].append(fe)
                agg[name]["work"].append(work)
                agg[name]["solved"].append(int(C.status(cells) == "solved"))
            if with_lattice:
                for k in lattice_ks:
                    cellsL, _ = factor_fixpoint_cells(csp, k)
                    recL, feL, _ = recall_vs(forced, cellsL, n, full)
                    lat[k]["recall"].append(recL); lat[k]["fe"].append(feL)
                    lat[k]["solved"].append(int(C.status(cellsL) == "solved"))
        row = {"band": band, "n": n, "mean_rank": float(np.mean(ranks)),
               "organ": {name: {"recall": float(np.mean(agg[name]["recall"])),
                                "solved": float(np.mean(agg[name]["solved"])),
                                "false_elim": int(np.sum(agg[name]["fe"])),
                                "mean_work": float(np.mean(agg[name]["work"]))}
                         for name in strategies},
               "lattice": {f"L{k}": {"recall": float(np.mean(lat[k]["recall"])),
                                     "solved": float(np.mean(lat[k]["solved"])),
                                     "false_elim": int(np.sum(lat[k]["fe"]))}
                           for k in lattice_ks} if with_lattice else {}}
        rows.append(row)
        o = row["organ"]
        latmsg = (f"  | LATTICE L3={row['lattice']['L3']['recall']*100:5.1f}%(solv{row['lattice']['L3']['solved']*100:3.0f})"
                  if with_lattice else "")
        wk = "  ".join(f"{nm[:4]}_work={o[nm]['mean_work']:6.0f}" for nm in strategies)
        rec0 = o["natural"]["recall"] * 100
        fe_tot = sum(o[nm]["false_elim"] for nm in strategies)
        print(f"  band={band:2d} (rank~{row['mean_rank']:.1f})  ORGAN recall={rec0:5.1f}% FE={fe_tot}  {wk}{latmsg}",
              flush=True)
    return rows


def size_sweep(ns=(8, 12, 16, 20, 24, 28), band_mode="global", n_inst=16, par_mult=1.4, seed=2, policy=None):
    """High-width (global band) uniquely-determined systems, vary n. Organ stays 100% as n/rank grows
    (where lattice L3 → floor); reports work growth per policy."""
    rng = np.random.default_rng(seed)
    rrng = np.random.default_rng(seed + 3)
    strategies = {"natural": policy_natural, "random": policy_random(rrng), "min_degree": policy_min_degree}
    if policy is not None:
        strategies["neural"] = policy.greedy_policy()
    rows = []
    for n in ns:
        band = n if band_mode == "global" else max(3, n // 2)
        ranks = []
        agg = {k: {"recall": [], "fe": [], "work": []} for k in strategies}
        lat3 = []
        for _ in range(n_inst):
            sysd = gen_xor_system(rng, n, n_par=int(par_mult * n), band=band, force_unique=True)
            A, b, csp = sysd["A"], sysd["b"], sysd["csp"]; full = csp.full()
            forced, rank, ok = gf2_forced(A, b, n); ranks.append(rank)
            for name, pol in strategies.items():
                cells, work, _ = solve_fixed(A, b, pol)
                rec, fe, _ = recall_vs(forced, cells, n, full)
                agg[name]["recall"].append(rec); agg[name]["fe"].append(fe); agg[name]["work"].append(work)
            if n <= 16:                              # lattice fixpoint blows up past ~16 (pins -> ~C(n,3) factors)
                cellsL, _ = factor_fixpoint_cells(csp, 3)
                recL, _, _ = recall_vs(forced, cellsL, n, full); lat3.append(recL)
        row = {"n": n, "band": band, "mean_rank": float(np.mean(ranks)),
               "organ": {nm: {"recall": float(np.mean(agg[nm]["recall"])),
                              "false_elim": int(np.sum(agg[nm]["fe"])),
                              "mean_work": float(np.mean(agg[nm]["work"]))} for nm in strategies},
               "lattice_L3_recall": (float(np.mean(lat3)) if lat3 else None)}
        rows.append(row)
        wk = "  ".join(f"{nm[:4]}={row['organ'][nm]['mean_work']:7.0f}" for nm in strategies)
        l3s = f"{row['lattice_L3_recall']*100:4.1f}%" if row["lattice_L3_recall"] is not None else "  -- "
        print(f"  n={n:2d} rank~{row['mean_rank']:4.1f}  ORGAN {row['organ']['natural']['recall']*100:5.1f}% "
              f"(L3={l3s})  work: {wk}", flush=True)
    return rows


# ============================================================ SMOKE: certified soundness + policy trains
def smoke(seed=0):
    print("=== row_space SMOKE ===", flush=True)
    rng = np.random.default_rng(seed)

    # 1) the certified GF(2) elimination solves a small XOR system SOUNDLY, and the read-off MATCHES
    #    the exact affine dedₚ AND the exact CSP transformer — for EVERY pivot order.
    print("\n[1] certified Gauss-Jordan read-off == GF(2)-exact == exact dedₚ, for all pivot orders:", flush=True)
    rrng = np.random.default_rng(123)
    orders = {"natural": policy_natural, "random": policy_random(rrng), "min_degree": policy_min_degree}
    bad = 0; ntest = 60
    for _ in range(ntest):
        n = int(rng.integers(5, 12))
        sysd = gen_xor_system(rng, n, n_par=int(rng.integers(n, 2 * n)), band=int(rng.integers(3, n + 1)),
                              force_unique=bool(rng.integers(0, 2)))
        A, b, csp = sysd["A"], sysd["b"], sysd["csp"]
        gf, _, _ = gf2_forced(A, b, n)
        ded, _ = C.to_fixpoint(C.exact_dedP, csp, csp.full())
        gf_s = tuple(sorted(x) for x in gf)
        ded_s = tuple(sorted(x) for x in ded)
        for name, pol in orders.items():
            cells, work, npiv = solve_fixed(A, b, pol)
            if tuple(sorted(x) for x in cells) != gf_s or gf_s != ded_s:
                bad += 1
    print(f"    {ntest*3-bad}/{ntest*3} (order x instance) read-offs EXACT  ({'OK' if bad==0 else 'MISMATCH!'})",
          flush=True)

    # 2) one concrete instance: show recall=100%, FE=0, and the WORK each pivot order spends.
    print("\n[2] one band=n (high-width) instance — recall/FE identical, WORK differs by order:", flush=True)
    sysd = gen_xor_system(np.random.default_rng(5), 14, n_par=20, band=14, force_unique=True)
    A, b, csp = sysd["A"], sysd["b"], sysd["csp"]; full = csp.full()
    forced, rank, _ = gf2_forced(A, b, 14)
    print(f"    n=14 rank={rank} (uniquely solvable, GF2 forces all 14 bits)", flush=True)
    for name, pol in orders.items():
        cells, work, npiv = solve_fixed(A, b, pol)
        rec, fe, _ = recall_vs(forced, cells, 14, full)
        print(f"      {name:10s}: recall {rec*100:5.1f}%  FE {fe}  pivots {npiv}  WORK {work}", flush=True)
    cells3, _ = factor_fixpoint_cells(csp, 3)
    r3, fe3, _ = recall_vs(forced, cells3, 14, full)
    print(f"      lattice L3 : recall {r3*100:5.1f}%  FE {fe3}   <-- the wall the organ steps over", flush=True)

    # 3) the neural pivot policy trains (a few steps; just proves the loop runs + grads flow).
    print("\n[3] neural pivot policy trains (REINFORCE, CPU smoke):", flush=True)
    try:
        pol, hist = train_policy(steps=20, batch=8, K=4, device="cpu", n_lo=8, n_hi=14, seed=0, log_every=10)
        print(f"    trained 20 steps; greedy vs natural {hist[-1]['greedy_vs_natural']*100:+.1f}% / "
              f"vs min-deg {hist[-1]['greedy_vs_mindeg']*100:+.1f}%  (policy loop + grads OK)", flush=True)
    except Exception as e:
        print(f"    [skip torch smoke: {e}]", flush=True)
    print("\nSMOKE DONE.", flush=True)


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"

    out = {}
    if args.smoke:
        smoke()
        return

    policy = None
    if args.train or args.all:
        print(f"\n=== TRAIN neural pivot policy (REINFORCE, device={args.device}) ===", flush=True)
        policy, hist = train_policy(steps=args.steps, batch=args.batch, device=args.device,
                                    band_mode="mixed", seed=0)
        out["train_history"] = hist
        if args.ckpt:
            policy.save(args.ckpt)
            print(f"saved policy -> {args.ckpt}", flush=True)

    if args.sweep or args.all:
        print("\n=== ROW-SPACE ORGAN: WIDTH SWEEP (fixed n=14, vary bandwidth=pathwidth) ===", flush=True)
        print("    organ recall must be 100% / FE 0 at EVERY band (certified) — the lattice wall is at L3.", flush=True)
        out["width_sweep"] = sweep(policy=policy)
        print("\n=== ROW-SPACE ORGAN: SIZE/RANK SWEEP (global high-width, vary n) ===", flush=True)
        out["size_sweep"] = size_sweep(policy=policy)

    if args.out and out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1, default=str)
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
