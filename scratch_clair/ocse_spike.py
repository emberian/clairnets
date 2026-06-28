"""oCSE SPIKE: can optimal Causation Entropy recover OUR CSP factor graphs from solution samples?

Decision-relevant test (CPU, local). We treat each CSP cell as a categorical variable, take the
SOLUTION SET as iid samples, and run a clean discrete-CMI oCSE (greedy forward-add by conditional
mutual information + backward prune, plug-in/empirical discrete CMI, conditional-permutation test).
We compare the inferred undirected dependency graph to the TRUE constraint (primal) graph.

The fork (Caerii/causationentropy) is time-series oriented (max_lag, lagged predictors) and ships
only CONTINUOUS CMI estimators (gaussian/knn/kde/geometric_knn + a correlation-matrix "poisson"),
none a plug-in discrete estimator — so we use our own discrete CMI here, which is the strongest
case for oCSE on categorical data.
"""
from __future__ import annotations
import itertools as it
import math
import numpy as np

import clair.csp as C
import clair.curriculum as Q


# --------------------------------------------------------------------- discrete plug-in CMI
def _cmi(X, Y, Z):
    """Empirical plug-in conditional mutual information I(X;Y|Z) in nats.
    X,Y are 1-D int arrays (N,). Z is (N,k) int array (k>=0 conditioning vars)."""
    N = len(X)
    if Z.shape[1] == 0:
        # marginal MI
        zc = np.zeros(N, dtype=np.int64)
    else:
        # encode the conditioning tuple into a single label
        _, zc = np.unique(Z, axis=0, return_inverse=True)
    total = 0.0
    for zv in np.unique(zc):
        m = zc == zv
        pz = m.mean()
        x, y = X[m], Y[m]
        n = len(x)
        # joint p(x,y), marginals within this z-stratum
        from collections import Counter
        cxy = Counter(zip(x.tolist(), y.tolist()))
        cx = Counter(x.tolist()); cy = Counter(y.tolist())
        mi_z = 0.0
        for (xv, yv), c in cxy.items():
            pxy = c / n
            px = cx[xv] / n; py = cy[yv] / n
            mi_z += pxy * math.log(pxy / (px * py))
        total += pz * mi_z
    return max(0.0, total)


def _cmi_pvalue(X, Y, Z, observed, n_shuffles=200, rng=None):
    """Conditional-permutation test: shuffle Y within strata of Z (preserves H0: X⟂Y|Z)."""
    rng = rng or np.random.default_rng(0)
    N = len(X)
    if Z.shape[1] == 0:
        groups = [np.arange(N)]
    else:
        _, zc = np.unique(Z, axis=0, return_inverse=True)
        groups = [np.where(zc == zv)[0] for zv in np.unique(zc)]
    ge = 0
    for _ in range(n_shuffles):
        Yp = Y.copy()
        for g in groups:
            Yp[g] = Y[g][rng.permutation(len(g))]
        if _cmi(X, Yp, Z) >= observed - 1e-12:
            ge += 1
    return (ge + 1) / (n_shuffles + 1)


# --------------------------------------------------------------------- oCSE (discrete)
def ocse_parents(data, target, alpha=0.05, n_shuffles=200, rng=None, max_parents=None):
    """Greedy forward-add + backward-prune oCSE for one target node. Returns set of parent indices.
    data: (N, n) int array. Conditional dependence is symmetric here (static samples)."""
    rng = rng or np.random.default_rng(0)
    N, n = data.shape
    Y = data[:, target]
    cands = [i for i in range(n) if i != target and data[:, i].std() > 0]
    if Y.std() == 0:
        return set()
    selected = []
    max_parents = max_parents or (n - 1)
    # FORWARD: repeatedly add the candidate with max CMI(X_i; Y | selected) if significant
    while cands and len(selected) < max_parents:
        Z = data[:, selected] if selected else np.zeros((N, 0), dtype=data.dtype)
        scores = [(_cmi(data[:, i], Y, Z), i) for i in cands]
        best, bi = max(scores)
        if best <= 0:
            break
        p = _cmi_pvalue(data[:, bi], Y, Z, best, n_shuffles, rng)
        if p > alpha:
            break
        selected.append(bi)
        cands.remove(bi)
    # BACKWARD: drop any selected node that is insignificant given all the OTHERS
    changed = True
    while changed and selected:
        changed = False
        for s in list(selected):
            others = [x for x in selected if x != s]
            Z = data[:, others] if others else np.zeros((N, 0), dtype=data.dtype)
            obs = _cmi(data[:, s], Y, Z)
            p = _cmi_pvalue(data[:, s], Y, Z, obs, n_shuffles, rng) if obs > 0 else 1.0
            if obs <= 0 or p > alpha:
                selected.remove(s); changed = True; break
    return set(selected)


def infer_graph(data, alpha=0.05, n_shuffles=200, seed=0):
    """Undirected dependency graph: edge(i,j) if j in parents(i) OR i in parents(j) (OR-symmetrize)."""
    n = data.shape[1]
    rng = np.random.default_rng(seed)
    edges = set()
    for t in range(n):
        for p in ocse_parents(data, t, alpha, n_shuffles, rng):
            edges.add((min(t, p), max(t, p)))
    return edges


# --------------------------------------------------------------------- true factor graph
def true_edges(csp: C.CSP, active):
    """Primal graph: edge(i,j) iff some constraint scope contains both i and j. Restrict to `active`
    (non-constant) cells so pinned/constant cells don't count as undetectable misses."""
    E = set()
    for sc, _ in csp.cons:
        for i, j in it.combinations(sorted(set(sc)), 2):
            if i in active and j in active:
                E.add((i, j))
    return E


def prf(inferred, truth):
    tp = len(inferred & truth)
    fp = len(inferred - truth)
    fn = len(truth - inferred)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1, tp, fp, fn


# --------------------------------------------------------------------- families to test
# For the affine families we set pin_frac=0 so the system is FREE (large solution set, no constant
# cells): this is the STRONGEST, fairest test for oCSE on higher-order structure. The binary/ordinal
# families keep light pins (realistic).
FAMILIES = {
    "coloring(neq)":   lambda rng: Q.gen_coloring(rng, k=3, n_lo=4, n_hi=6, pin_frac=0.15),
    "equality(eq/neq)":lambda rng: Q.gen_equality(rng, k=3, n_lo=4, n_hi=6, pin_frac=0.15),
    "ordering(lt/le)": lambda rng: Q.gen_ordering(rng, n_lo=4, n_hi=5, pin_frac=0.15),
    "alldiff":         lambda rng: Q.gen_alldiff(rng, n_lo=4, n_hi=5, pin_frac=0.15),
    "arithmetic(sum)": lambda rng: Q.gen_arithmetic(rng, n_lo=3, n_hi=4, d_lo=4, d_hi=5, pin_frac=0.0),
    "xor(parity-3)":   lambda rng: Q.gen_xor(rng, n_lo=4, n_hi=5, pin_frac=0.0),
    "parity_k":        lambda rng: Q.gen_parity_k(rng, n_lo=5, n_hi=7, kmin=3, kmax=4, pin_frac=0.0),
}

MAX_SOLS = 4000          # cap enumerated solution set (cells small; this is plenty for plug-in CMI)
N_INST = 8               # instances per family
ALPHA = 0.05
N_SHUF = 150


def run():
    print(f"oCSE SPIKE — discrete plug-in CMI, alpha={ALPHA}, {N_SHUF} shuffles, "
          f"{N_INST} instances/family, <= {MAX_SOLS} solution-samples\n")
    header = f"{'family':18} {'avgP':>6} {'avgR':>6} {'avgF1':>6} {'inst':>5} {'meanSols':>9} {'meanCells':>9}"
    print(header); print("-" * len(header))
    for name, gen in FAMILIES.items():
        Ps, Rs, F1s, nsol, ncell = [], [], [], [], []
        used = 0
        for seed in range(40):
            if used >= N_INST:
                break
            rng = np.random.default_rng(1000 + seed)
            try:
                n, d, kind, facts, s = gen(rng)
                csp = Q.build_csp(n, d, facts)
            except Exception:
                continue
            sols = C.solutions(csp, limit=MAX_SOLS)
            if len(sols) < 8:
                continue                       # need samples for CMI
            data = np.array(sols, dtype=np.int64)
            active = {i for i in range(n) if data[:, i].std() > 0}
            T = true_edges(csp, active)
            if not T:
                continue                       # nothing to recover (all pinned / trivial)
            G = infer_graph(data, ALPHA, N_SHUF, seed=seed)
            G = {e for e in G if e[0] in active and e[1] in active}
            p, r, f1, *_ = prf(G, T)
            Ps.append(p); Rs.append(r); F1s.append(f1)
            nsol.append(len(sols)); ncell.append(len(active)); used += 1
        if F1s:
            print(f"{name:18} {np.mean(Ps):6.2f} {np.mean(Rs):6.2f} {np.mean(F1s):6.2f} "
                  f"{used:5d} {np.mean(nsol):9.0f} {np.mean(ncell):9.1f}")
        else:
            print(f"{name:18}  (no usable instances)")

    # ---- one explicit textbook case: the parity blind spot, fully enumerated ----
    print("\n-- diagnostic: xor_parity() (x=y, y=z, x^y^z=0), unique sol (0,0,0) is degenerate; use free parity --")
    # a parity triple with FREE inputs: a,b free, c=a^b -> uniform over 4 solutions
    csp = C.CSP(3, 2, (C._rel((0, 1, 2), lambda t: (t[0] ^ t[1] ^ t[2]) == 0, 2),))
    sols = C.solutions(csp)
    data = np.array(sols, dtype=np.int64)
    print(f"   solutions={sols}")
    print(f"   marginal MI I(0;1)={_cmi(data[:,0], data[:,1], np.zeros((len(data),0),int)):.3f}  "
          f"I(0;2)={_cmi(data[:,0], data[:,2], np.zeros((len(data),0),int)):.3f}  (both 0 -> greedy can't start)")
    z1 = data[:, [1]]
    print(f"   conditional  I(0;2 | 1)={_cmi(data[:,0], data[:,2], z1):.3f}  "
          f"(>0: the edge IS there once you condition on the 3rd — but forward-greedy never gets here)")
    G = infer_graph(data, ALPHA, N_SHUF, seed=0)
    T = true_edges(csp, {0, 1, 2})
    print(f"   true factor edges={sorted(T)}  oCSE inferred={sorted(G)}  -> F1={prf(G,T)[2]:.2f}")


if __name__ == "__main__":
    run()
