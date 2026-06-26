"""clair/ising_organ.py — the OPTIMIZATION BEAST: Ising / QUBO as a UNIVERSAL substrate.

This is the SHARPENED form of clair.energy_organ. The energy organ relaxes a CSP over the product
of simplices and descends a soft constraint-violation energy; here we specialise that machine to the
canonical optimisation object — the ISING HAMILTONIAN — where the energy IS the objective and the
energy IS the exact verifier. (energy_organ's mean-field over a d=2 softmax is literally tanh; this
organ is that, annealed, restarted, and finished with a SOUND local search.)

THE MODEL.
    spins  s ∈ {−1,+1}^n
    H(s)   = − Σ_{i<j} J_ij s_i s_j − Σ_i h_i s_i        (J symmetric, zero diagonal)
           = −½ sᵀ J s − hᵀ s                            (the vectorised form we compute)
    goal   = MINIMISE H (the ground state).  QUBO is the {0,1} twin: x=(s+1)/2, f(x)=xᵀQx.

THE ORGAN = an energy-based PROPOSAL engine that emits low-energy configs:
    (1) MEAN-FIELD ANNEALING over magnetisations m_i=⟨s_i⟩∈[−1,1]:  m_i ← tanh(L_i(m)/T),
        L_i = Σ_j J_ij m_j + h_i, deterministic annealing T:hot→cold, damped fixed point, MANY
        symmetry-broken restarts in parallel on the GPU (a perfectly symmetric init is a degenerate
        mean-field fixed point — noise breaks it, exactly as in energy_organ).
    (2) ROUND  s = sign(m).
    (3) SOUND LOCAL SEARCH: greedy single-spin steepest descent to a 1-flip-stable local minimum.
        ΔH(flip k) = 2 s_k L_k is EXACT; we only ever take ΔH<0 moves, so energy never increases.
    The optimiser is NOT sound-by-construction (NP-hard; mean-field has local minima). SOUNDNESS comes
    from the OUTPUT-CHECK: H(s) of the returned config is computed EXACTLY and cheaply — a verifiable
    reward. We report the config and its true energy; we never claim optimality we didn't check.

THE UNIVERSAL-SUBSTRATE PART (the leverage). A huge class of NP problems COMPILE to Ising (Lucas 2014).
We implement the encodings + decoders for four canonical problems, so: problem → Ising → organ → decode:
    • MAX-CUT            H = Σ_{(i,j)∈E} (s_i s_j − 1)/2     (the cleanest; J_ij=−w_ij on edges)
    • NUMBER-PARTITION   H = (Σ_i a_i s_i)²                  (J_ij=−a_i a_j; ground state 0 ⇔ perfect)
    • GRAPH-COLORING     penalty QUBO (one-hot + edge conflicts) → Ising  (k·n spins)
    • MAX-INDEPENDENT-SET / SET-PACKING  QUBO −Σx_v + B·Σ_{edges} x_u x_v → Ising
  This is α-compile-to-Ising: ONE organ + the encodings solves the ORIGINAL problems.

THE EXACT VERIFIER.
    • H(s) — always, O(n²), the reward.
    • small n: BRUTE-FORCE ground state (all 2^n configs, batched on GPU) → true optimality gap.
    • mapped problems: brute-force / networkx ground truth (true max-cut, min partition diff,
      k-colourability, max independent set) → does solving-via-Ising recover the right answer?

BASELINES.  energy_organ-style plain mean field (no anneal / restarts / local search) and the
canonical SIMULATED ANNEALING (Metropolis). The organ should dominate both, and match brute force
at small n.

Run:  python -m clair.ising_organ --smoke    # tiny Ising ground state vs brute force; max-cut maps→solves
      python -m clair.ising_organ            # optimality-gap tables (raw Ising + mapped) + substrate demo
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None


# ============================================================================= the energy (exact reward)
def ising_energy(J: torch.Tensor, h: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """H(s) = −½ sᵀJs − hᵀs for a BATCH of spin configs S [...,n] (entries ±1). J [n,n] symmetric,
    zero diagonal; h [n]. Returns [...] energies. This is the EXACT objective and the EXACT verifier."""
    quad = (S @ J * S).sum(-1)              # sᵀJs  = Σ_ij J_ij s_i s_j
    lin = (S * h).sum(-1)                   # hᵀs
    return -0.5 * quad - lin


def local_field(J: torch.Tensor, h: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """L_i = Σ_j J_ij s_j + h_i for a batch S [...,n]. The molecular field; ΔH(flip k)=2 s_k L_k."""
    return S @ J + h


# ============================================================================= SOUND local search
@torch.no_grad()
def greedy_descent(J, h, S, max_passes=None):
    """Batched greedy SINGLE-SPIN steepest descent: repeatedly flip, in each config, the one spin with
    the most-negative ΔH=2 s_k L_k, until no improving flip remains (a 1-flip-stable local minimum).
    EXACT and monotone — energy never increases — so this is a SOUND refinement of any proposal."""
    S = S.clone()
    n = S.shape[-1]
    max_passes = max_passes if max_passes is not None else 4 * n
    rows = torch.arange(S.shape[0], device=S.device)
    for _ in range(max_passes):
        L = S @ J + h                       # [B,n]
        dH = 2.0 * S * L                    # ΔH for flipping each spin
        best = dH.argmin(dim=1)             # most-improving spin per config
        gain = dH[rows, best]
        improving = gain < -1e-9
        if not improving.any():
            break
        flip = improving
        S[rows[flip], best[flip]] *= -1.0
    return S


# ============================================================================= THE ORGAN: mean-field annealing
@torch.no_grad()
def mean_field_anneal(J, h, restarts=64, steps=160, temp0=1.5, temp1=0.02, damp=0.4,
                      noise=2.0, sto_noise=1.0, local_search=True, seed=0, device=None, chunk=4096):
    """The optimisation organ. Run `restarts` STOCHASTIC mean-field descents IN PARALLEL on the GPU with
    deterministic annealing T:temp0→temp1, round each to ±1, finish with the sound greedy local search,
    and KEEP THE LOWEST EXACT-ENERGY config across all restarts.

        m_i ← (1−damp)·m_i + damp·tanh( (L_i(m) + sto_noise·T·ξ) / T ),   L_i = Σ_j J_ij m_j + h_i
        T annealed geometrically (a continuation method that dodges local minima)

    Two design choices matter and were both necessary (a pure deterministic anneal from a hot temp melts
    EVERY restart to m≈0 and converges to ONE identical fixed point, so restarts buy nothing): (i) a
    MODERATE temp0 so each restart's symmetry-broken init keeps it in its own basin, and (ii) an annealed
    gaussian field NOISE ξ that keeps the restarts genuinely diverse. The greedy descent then finishes
    each into its local minimum, and we keep the global best by EXACT energy. Returns (best_s, best_E, info)."""
    device = device or J.device
    J = J.to(device); h = h.to(device)
    n = J.shape[0]
    g = torch.Generator(device=device).manual_seed(int(seed))
    best_S = best_E = None
    t0 = time.time()
    done = 0
    while done < restarts:
        b = min(chunk, restarts - done)
        M = torch.tanh(noise * torch.randn(b, n, device=device, generator=g))   # symmetry-broken init
        for t in range(steps):
            frac = t / max(1, steps - 1)
            T = temp0 * (temp1 / temp0) ** frac
            L = M @ J + h
            if sto_noise:
                L = L + sto_noise * T * torch.randn(b, n, device=device, generator=g)
            M = (1 - damp) * M + damp * torch.tanh(L / T)
        S = torch.sign(M)
        S[S == 0] = 1.0
        if local_search:
            S = greedy_descent(J, h, S)
        E = ising_energy(J, h, S)
        k = int(E.argmin())
        if best_E is None or E[k].item() < best_E:
            best_E = E[k].item()
            best_S = S[k].clone()
        done += b
    return best_S, best_E, {"restarts": restarts, "steps": steps, "wall_s": time.time() - t0,
                            "local_search": local_search}


# ============================================================================= baseline: plain mean field
@torch.no_grad()
def mf_baseline(J, h, steps=160, temp=0.2, damp=0.4, device=None):
    """energy_organ-STYLE baseline: a single plain damped mean field at a fixed low T from a near-uniform
    init — NO annealing, NO restarts, NO local search. Isolates what the sharpening (1)-(3) buys."""
    device = device or J.device
    J = J.to(device); h = h.to(device)
    n = J.shape[0]
    M = torch.zeros(n, device=device)
    for _ in range(steps):
        L = M @ J + h
        M = (1 - damp) * M + damp * torch.tanh(L / temp)
    S = torch.sign(M); S[S == 0] = 1.0
    return S, ising_energy(J, h, S).item()


# ============================================================================= baseline: simulated annealing
@torch.no_grad()
def simulated_annealing(J, h, restarts=64, sweeps=400, temp0=3.0, temp1=0.02,
                        seed=0, device=None):
    """Canonical Metropolis SA over spins, `restarts` chains in parallel, geometric cooling. Each sweep
    proposes n single-spin flips; accept ΔH<0 always, else with prob exp(−ΔH/T). The standard yardstick."""
    device = device or J.device
    J = J.to(device); h = h.to(device)
    n = J.shape[0]
    g = torch.Generator(device=device).manual_seed(int(seed))
    S = torch.where(torch.rand(restarts, n, device=device, generator=g) < 0.5, -1.0, 1.0)
    rows = torch.arange(restarts, device=device)
    L = S @ J + h                            # full molecular field, maintained INCREMENTALLY (no per-spin matmul)
    t0 = time.time()
    for sw in range(sweeps):
        frac = sw / max(1, sweeps - 1)
        T = temp0 * (temp1 / temp0) ** frac
        order = torch.argsort(torch.rand(restarts, n, device=device, generator=g), dim=1)
        for j in range(n):
            k = order[:, j]                  # [R] spin chosen by each chain this step
            sk = S[rows, k]                  # [R]
            dH = 2.0 * sk * L[rows, k]       # exact ΔH from the maintained field
            acc = (dH < 0) | (torch.rand(restarts, device=device, generator=g) < torch.exp(-dH / T))
            delta = (-2.0 * sk) * acc.to(S.dtype)            # change in the chosen spin (0 if rejected)
            S[rows, k] = sk + delta
            L = L + delta.unsqueeze(1) * J[k]                # field update: L_i += Δs_k · J[k,i]
    E = ising_energy(J, h, S)
    k = int(E.argmin())
    return S[k].clone(), E[k].item(), {"wall_s": time.time() - t0}


# ============================================================================= EXACT ground state (brute force)
@torch.no_grad()
def brute_force_ground(J, h, device=None, chunk=1 << 16):
    """Exact ground state by enumerating all 2^n spin configs (batched on the GPU). For n ≲ 22. Returns
    (best_s, min_E, max_E). max_E (the worst config) anchors the normalised optimality gap."""
    device = device or J.device
    J = J.to(device); h = h.to(device)
    n = J.shape[0]
    assert n <= 24, "brute force only for small n"
    bits = torch.arange(n, device=device)
    best_s = None; min_E = math.inf; max_E = -math.inf
    total = 1 << n
    for start in range(0, total, chunk):
        idx = torch.arange(start, min(start + chunk, total), device=device)
        S = (((idx[:, None] >> bits[None, :]) & 1) * 2 - 1).float()   # [b,n] ±1
        E = ising_energy(J, h, S)
        mi = int(E.argmin()); ma = float(E.max())
        if E[mi].item() < min_E:
            min_E = E[mi].item(); best_s = S[mi].clone()
        max_E = max(max_E, ma)
    return best_s, min_E, max_E


# ============================================================================= raw-Ising instance generators
def rand_ising(rng, n, kind="sk", device="cpu", field=True):
    """Random Ising instances. kind:
        sk        Sherrington–Kirkpatrick spin glass: J_ij ~ N(0,1/√n), full coupling (the hard one).
        gaussian  dense J_ij ~ N(0,1).
        pm1       ±1 random couplings (Bernoulli) — discrete spin glass.
    h ~ N(0,1) if field else 0. Returns symmetric zero-diagonal (J, h)."""
    A = rng.standard_normal((n, n)).astype(np.float32)
    Jm = np.triu(A, 1)
    Jm = Jm + Jm.T
    if kind == "sk":
        Jm /= math.sqrt(n)
    elif kind == "pm1":
        Jm = np.sign(np.triu(rng.standard_normal((n, n)), 1)).astype(np.float32)
        Jm = Jm + Jm.T
    np.fill_diagonal(Jm, 0.0)
    h = rng.standard_normal(n).astype(np.float32) if field else np.zeros(n, np.float32)
    return (torch.as_tensor(Jm, device=device), torch.as_tensor(h, device=device))


# ============================================================================= QUBO → Ising (Lucas 2014)
def qubo_to_ising(Q: np.ndarray):
    """Exact reduction of QUBO f(x)=xᵀQx (x∈{0,1}, Q symmetric, diagonal carries the linear terms since
    x_i²=x_i) to Ising with x=(1+s)/2:
        J_ij = −Q_ij/2 (i≠j, zero diag),  h_i = −½ Σ_j Q_ij,  const = ¼(Σ_ij Q_ij + Σ_i Q_ii)
    such that f(x) = ising_energy(J,h, 2x−1) + const for EVERY x. (Verified exhaustively in --smoke.)"""
    Q = np.asarray(Q, dtype=np.float64)
    Q = 0.5 * (Q + Q.T)                      # symmetrise
    J = -0.5 * Q.copy()
    np.fill_diagonal(J, 0.0)
    h = -0.5 * Q.sum(axis=1)
    const = 0.25 * (Q.sum() + np.trace(Q))
    return J.astype(np.float32), h.astype(np.float32), float(const)


# ============================================================================= ENCODINGS: NP → Ising + decode
def encode_maxcut(G, device="cpu"):
    """MAX-CUT → Ising. Maximise Σ_{(i,j)∈E} w_ij(1−s_i s_j)/2  ⇔  minimise H=−½ sᵀJs with J_ij=−w_ij on
    edges, h=0. cut(s)=Σ_edges w·[s_i≠s_j]. (The cleanest map: H itself, no penalties.)"""
    nodes = list(G.nodes()); idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    Jm = np.zeros((n, n), np.float32)
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        Jm[idx[u], idx[v]] -= w
        Jm[idx[v], idx[u]] -= w
    h = np.zeros(n, np.float32)
    meta = {"nodes": nodes, "idx": idx}
    return torch.as_tensor(Jm, device=device), torch.as_tensor(h, device=device), 0.0, meta


def decode_maxcut(G, meta, s):
    """Cut value of the ±1 partition. side(v) = sign(s_v)."""
    idx = meta["idx"]
    s = s.cpu().numpy()
    side = {v: (s[idx[v]] > 0) for v in G.nodes()}
    return sum(float(d.get("weight", 1.0)) for u, v, d in G.edges(data=True) if side[u] != side[v])


def encode_partition(a, device="cpu"):
    """NUMBER-PARTITION → Ising. Split {a_i} into ± to minimise (Σ a_i s_i)². J_ij=−a_i a_j (i≠j), h=0.
    H = ½[(Σa_i s_i)² − Σa_i²]; H+½Σa_i² = diff² where diff=|Σa_i s_i|. Ground state diff=0 ⇔ perfect."""
    a = np.asarray(a, np.float64)
    n = len(a)
    Jm = -np.outer(a, a)
    np.fill_diagonal(Jm, 0.0)
    h = np.zeros(n, np.float32)
    const = 0.5 * float((a ** 2).sum())
    return (torch.as_tensor(Jm.astype(np.float32), device=device),
            torch.as_tensor(h, device=device), const, {"a": a})


def decode_partition(meta, s):
    """Subset-sum difference |Σ a_i s_i| of the ±1 split."""
    a = meta["a"]; s = s.cpu().numpy()
    return abs(float((a * s).sum()))


def encode_coloring(G, k, A=None, device="cpu"):
    """GRAPH-COLORING → penalty QUBO → Ising (Lucas §6). Binary x_{v,c}∈{0,1}, spin index = v*k+c:
        H = A·Σ_v (1−Σ_c x_{vc})²  +  Σ_{(u,v)∈E} Σ_c x_{uc} x_{vc}
    one-hot penalty A (default = deg_max+1, so a single edge can never pay for breaking a one-hot).
    H=0 ⇔ a proper k-colouring was found. Returns (J,h,const,meta)."""
    nodes = list(G.nodes()); idx = {v: i for i, v in enumerate(nodes)}
    nv = len(nodes); N = nv * k
    if A is None:
        A = (max((d for _, d in G.degree()), default=0) + 1.0)
    Q = np.zeros((N, N), np.float64)

    def vc(v, c):
        return idx[v] * k + c
    # A·(1 − Σ_c x_vc)² = A·(1 − 2Σ x_vc + (Σ x_vc)²); (Σx)² = Σ x_c + 2Σ_{c<c'} x_c x_c'
    # linear: A·(−2x_vc + x_vc) = −A x_vc   (diagonal);  pair within a vertex: +2A x_vc x_vc'
    for v in nodes:
        for c in range(k):
            Q[vc(v, c), vc(v, c)] += -A
            for c2 in range(c + 1, k):
                Q[vc(v, c), vc(v, c2)] += 2.0 * A
    const_color = A * nv                      # the Σ_v 1 term
    # edge conflict Σ_c x_uc x_vc
    for u, v in G.edges():
        for c in range(k):
            Q[vc(u, c), vc(v, c)] += 1.0
    J, h, const = qubo_to_ising(Q)
    meta = {"nodes": nodes, "idx": idx, "k": k}
    return (torch.as_tensor(J, device=device), torch.as_tensor(h, device=device),
            const + const_color, meta)


def decode_coloring(G, meta, s):
    """Colour(v) = argmax_c x_{vc} (x=(s+1)/2). Returns (coloring dict, is_proper, n_colors_used)."""
    k = meta["k"]; idx = meta["idx"]
    x = ((s.cpu().numpy() + 1) / 2)
    coloring = {}
    for v in G.nodes():
        block = x[idx[v] * k:(idx[v] + 1) * k]
        coloring[v] = int(block.argmax())
    proper = all(coloring[u] != coloring[v] for u, v in G.edges())
    return coloring, proper, len(set(coloring.values()))


def encode_mis(G, B=2.0, device="cpu"):
    """MAX-INDEPENDENT-SET / SET-PACKING → QUBO → Ising (Lucas §4.2). x_v∈{0,1}=v selected:
        H = −Σ_v x_v + B·Σ_{(u,v)∈E} x_u x_v   (B>1 ⇒ no optimal set keeps an edge).
    Maximise |IS|. Returns (J,h,const,meta)."""
    nodes = list(G.nodes()); idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    Q = np.zeros((n, n), np.float64)
    for v in nodes:
        Q[idx[v], idx[v]] += -1.0
    for u, v in G.edges():
        Q[idx[u], idx[v]] += B
    J, h, const = qubo_to_ising(Q)
    return (torch.as_tensor(J, device=device), torch.as_tensor(h, device=device),
            const, {"nodes": nodes, "idx": idx})


def decode_mis(G, meta, s):
    """Selected = {v : x_v=1}. Returns (set, is_independent, size)."""
    idx = meta["idx"]
    x = ((s.cpu().numpy() + 1) / 2)
    sel = {v for v in G.nodes() if x[idx[v]] > 0.5}
    indep = all(not (u in sel and v in sel) for u, v in G.edges())
    return sel, indep, len(sel)


# ============================================================================= exact ground truth for mapped
def exact_maxcut(G, device="cpu"):
    """Exact max cut, GPU-batched. max-cut = (W_total − min_s H(s))/2 where H is the encoded Ising energy
    (cut(s)=(W_total − H(s))/2), so the ground state of the max-cut Ising IS the max cut. O(2^n) on GPU."""
    W_total = sum(float(d.get("weight", 1.0)) for _, _, d in G.edges(data=True))
    J, h, _, _ = encode_maxcut(G, device=device)
    _, min_E, _ = brute_force_ground(J, h, device=device)
    return (W_total - min_E) / 2.0


@torch.no_grad()
def exact_partition(a, device="cpu"):
    """Exact minimum partition difference min_s |Σ a_i s_i|, GPU-batched over all 2^n ± splits."""
    a_t = torch.as_tensor(np.asarray(a, np.float32), device=device)
    n = len(a)
    bits = torch.arange(n, device=device)
    best = math.inf
    for start in range(0, 1 << n, 1 << 16):
        idx = torch.arange(start, min(start + (1 << 16), 1 << n), device=device)
        S = (((idx[:, None] >> bits[None, :]) & 1) * 2 - 1).float()
        best = min(best, float((S @ a_t).abs().min()))
    return best


def exact_chromatic_feasible(G, k):
    """Is G k-colourable? Exact via networkx-free brute force over k^n (tiny n) — ground truth for the
    coloring demo (does the Ising route find a proper k-colouring when one exists?)."""
    nodes = list(G.nodes()); n = len(nodes)
    edges = [(nodes.index(u), nodes.index(v)) for u, v in G.edges()]
    import itertools as it
    for assign in it.product(range(k), repeat=n):
        if all(assign[u] != assign[v] for u, v in edges):
            return True
    return False


def exact_mis(G):
    """Exact maximum independent set size = max clique on the complement (networkx), with a brute-force
    fallback. Ground truth for the MIS demo."""
    if nx is not None:
        comp = nx.complement(G)
        try:
            _, w = nx.max_weight_clique(comp, weight=None)
            return int(w)
        except Exception:
            pass
    nodes = list(G.nodes()); n = len(nodes)
    edges = [(nodes.index(u), nodes.index(v)) for u, v in G.edges()]
    best = 0
    for m in range(1 << n):
        sel = [i for i in range(n) if (m >> i) & 1]
        if all(not (u in sel and v in sel) for u, v in edges):
            best = max(best, len(sel))
    return best


# ============================================================================= SMOKE
def smoke(device):
    print("== SMOKE: Ising organ ==")
    rng = np.random.default_rng(0)

    # (a) QUBO->Ising reduction is EXACT (exhaustive check on a tiny random QUBO)
    n = 6
    Qr = rng.standard_normal((n, n)).astype(np.float64)
    J, h, const = qubo_to_ising(Qr)
    Jt = torch.as_tensor(J); ht = torch.as_tensor(h)
    bits = torch.arange(n)
    idxall = torch.arange(1 << n)
    X = ((idxall[:, None] >> bits[None, :]) & 1).float()
    S = 2 * X - 1
    f_direct = torch.einsum("bi,ij,bj->b", X, torch.as_tensor(0.5 * (Qr + Qr.T)).float(), X)
    f_ising = ising_energy(Jt, ht, S) + const
    max_err = (f_direct - f_ising).abs().max().item()
    print(f"  QUBO->Ising exact reduction: max|f_direct - f_ising| = {max_err:.2e} over all 2^{n} configs")
    assert max_err < 1e-3, "QUBO->Ising reduction must be exact"

    # (b) organ finds the ground state of a tiny Ising vs brute force
    Jt, ht = rand_ising(rng, 12, kind="sk", device=device)
    s_bf, E_bf, E_worst = brute_force_ground(Jt, ht, device=device)
    s_o, E_o, _ = mean_field_anneal(Jt, ht, restarts=64, steps=160, device=device, seed=1)
    print(f"  tiny SK n=12: ground E={E_bf:.4f}  organ E={E_o:.4f}  worst={E_worst:.4f}  "
          f"hit={abs(E_o - E_bf) < 1e-4}")
    assert E_o >= E_bf - 1e-4, "organ energy cannot be below the ground state (verifier sanity)"
    assert abs(E_o - E_bf) < 1e-3, "organ should reach the ground state on a tiny instance"

    # (c) a max-cut maps -> Ising -> solves
    if nx is not None:
        G = nx.gnp_random_graph(10, 0.5, seed=2)
        J, h, _, meta = encode_maxcut(G, device=device)
        s_o, _, _ = mean_field_anneal(J, h, restarts=64, steps=160, device=device, seed=3)
        cut = decode_maxcut(G, meta, s_o)
        opt = exact_maxcut(G, device=device)
        print(f"  max-cut n=10 m={G.number_of_edges()}: organ cut={cut:.0f}  exact max={opt:.0f}  "
              f"hit={abs(cut - opt) < 1e-6}")
        assert cut <= opt + 1e-6
    print("  SMOKE OK\n")


# ============================================================================= raw-Ising optimality table
def run_raw_ising(device, seed=0):
    print("================ RAW ISING / QUBO — optimality gap vs EXACT ground state ================")
    rng = np.random.default_rng(seed)
    rows = []
    small_specs = [("sk", 16), ("sk", 18), ("gaussian", 16), ("pm1", 18)]
    n_inst = 24
    for kind, n in small_specs:
        agg = {"organ": [], "sa": [], "mf": [], "organ_hit": 0, "sa_hit": 0, "mf_hit": 0}
        for r in range(n_inst):
            J, h = rand_ising(rng, n, kind=kind, device=device)
            s_bf, E_bf, E_worst = brute_force_ground(J, h, device=device)
            span = max(1e-9, E_worst - E_bf)
            _, E_o, _ = mean_field_anneal(J, h, restarts=48, steps=140, device=device, seed=r)
            _, E_sa, _ = simulated_annealing(J, h, restarts=48, sweeps=300, device=device, seed=r)
            _, E_mf = mf_baseline(J, h, device=device)
            agg["organ"].append((E_o - E_bf) / span); agg["organ_hit"] += int(abs(E_o - E_bf) < 1e-4)
            agg["sa"].append((E_sa - E_bf) / span);   agg["sa_hit"] += int(abs(E_sa - E_bf) < 1e-4)
            agg["mf"].append((E_mf - E_bf) / span);   agg["mf_hit"] += int(abs(E_mf - E_bf) < 1e-4)
        rows.append({
            "kind": kind, "n": n, "instances": n_inst,
            "organ_gap": float(np.mean(agg["organ"])), "organ_hit": agg["organ_hit"] / n_inst,
            "sa_gap": float(np.mean(agg["sa"])), "sa_hit": agg["sa_hit"] / n_inst,
            "mf_gap": float(np.mean(agg["mf"])), "mf_hit": agg["mf_hit"] / n_inst,
        })
    print(f"  {'kind':9s} {'n':>3} | {'organ gap':>10} {'hit':>5} | {'SA gap':>8} {'hit':>5} | "
          f"{'MFbase gap':>10} {'hit':>5}   (gap: 0=ground,1=worst; over {n_inst} instances)")
    for r in rows:
        print(f"  {r['kind']:9s} {r['n']:>3} | {r['organ_gap']:>10.4f} {r['organ_hit']*100:>4.0f}% | "
              f"{r['sa_gap']:>8.4f} {r['sa_hit']*100:>4.0f}% | {r['mf_gap']:>10.4f} {r['mf_hit']*100:>4.0f}%")

    # larger n: no exact; report energy gap of organ vs SA (organ-SA, negative = organ better)
    print("\n  larger n (no brute force) — best energy found, organ vs SA (lower=better):")
    big_rows = []
    n_big = 6
    for kind, n in [("sk", 60), ("sk", 100), ("gaussian", 80)]:
        do = ds = 0.0
        wins = 0
        for r in range(n_big):
            J, h = rand_ising(rng, n, kind=kind, device=device)
            _, E_o, _ = mean_field_anneal(J, h, restarts=128, steps=220, device=device, seed=r)
            _, E_sa, _ = simulated_annealing(J, h, restarts=128, sweeps=300, device=device, seed=r)
            do += E_o; ds += E_sa; wins += int(E_o <= E_sa + 1e-6)
        big_rows.append({"kind": kind, "n": n, "organ_E": do / n_big, "sa_E": ds / n_big,
                         "organ_wins": wins / n_big})
        print(f"  {kind:9s} n={n:>4}: organ meanE={do/n_big:>12.3f}  SA meanE={ds/n_big:>12.3f}  "
              f"organ<=SA in {wins}/{n_big}")
    return {"small": rows, "large": big_rows}


# ============================================================================= universal-substrate demo
def run_substrate(device, seed=0):
    print("\n================ UNIVERSAL SUBSTRATE — NP problems solved VIA the Ising encoding ================")
    if nx is None:
        print("  networkx unavailable; skipping mapped-problem demo")
        return {}
    rng = np.random.default_rng(seed)
    out = {}

    # ---- MAX-CUT
    hit = 0; gaps = []; tot = 12
    for r in range(tot):
        G = nx.gnp_random_graph(rng.integers(9, 15), 0.5, seed=int(rng.integers(1 << 30)))
        J, h, _, meta = encode_maxcut(G, device=device)
        s, _, _ = mean_field_anneal(J, h, restarts=64, steps=160, device=device, seed=r)
        cut = decode_maxcut(G, meta, s); opt = exact_maxcut(G, device=device)
        hit += int(abs(cut - opt) < 1e-6); gaps.append((opt - cut) / max(1, opt))
    out["maxcut"] = {"instances": tot, "exact_hit_rate": hit / tot, "mean_rel_gap": float(np.mean(gaps))}
    print(f"  MAX-CUT          : {hit}/{tot} hit exact max cut   mean rel-gap {np.mean(gaps)*100:.2f}%")

    # ---- NUMBER PARTITION
    hit = 0; tot = 12; perfect = 0
    for r in range(tot):
        m = int(rng.integers(10, 16))
        a = rng.integers(1, 50, size=m)
        J, h, _, meta = encode_partition(a, device=device)
        s, _, _ = mean_field_anneal(J, h, restarts=64, steps=160, device=device, seed=r)
        diff = decode_partition(meta, s); opt = exact_partition(a, device=device)
        hit += int(abs(diff - opt) < 1e-6); perfect += int(opt < 1e-9 and diff < 1e-9)
    out["partition"] = {"instances": tot, "exact_hit_rate": hit / tot}
    print(f"  NUMBER-PARTITION : {hit}/{tot} matched the minimum subset-sum difference")

    # ---- GRAPH COLORING
    hit = 0; tot = 12; feasible_truth = 0; found = 0
    for r in range(tot):
        nnodes = int(rng.integers(6, 10)); k = 3
        G = nx.gnp_random_graph(nnodes, 0.4, seed=int(rng.integers(1 << 30)))
        truth = exact_chromatic_feasible(G, k)
        feasible_truth += int(truth)
        J, h, _, meta = encode_coloring(G, k, device=device)
        s, _, _ = mean_field_anneal(J, h, restarts=128, steps=220, device=device, seed=r)
        _, proper, used = decode_coloring(G, meta, s)
        found += int(proper)
        if truth:
            hit += int(proper)             # found a proper coloring when one exists
    out["coloring"] = {"instances": tot, "k_colorable_truth": feasible_truth,
                       "proper_found_when_colorable": hit}
    print(f"  GRAPH-COLORING(k=3): {feasible_truth}/{tot} are 3-colorable; organ found a proper "
          f"coloring on {hit}/{feasible_truth} of those")

    # ---- MAX INDEPENDENT SET
    hit = 0; tot = 12; gaps = []
    for r in range(tot):
        nnodes = int(rng.integers(9, 14))
        G = nx.gnp_random_graph(nnodes, 0.35, seed=int(rng.integers(1 << 30)))
        J, h, _, meta = encode_mis(G, B=2.0, device=device)
        s, _, _ = mean_field_anneal(J, h, restarts=96, steps=180, device=device, seed=r)
        sel, indep, size = decode_mis(G, meta, s)
        opt = exact_mis(G)
        size_eff = size if indep else 0
        hit += int(indep and size_eff == opt); gaps.append((opt - size_eff) / max(1, opt))
    out["mis"] = {"instances": tot, "exact_hit_rate": hit / tot, "mean_rel_gap": float(np.mean(gaps))}
    print(f"  MAX-INDEP-SET    : {hit}/{tot} hit exact MIS size   mean rel-gap {np.mean(gaps)*100:.2f}%")
    return out


# ============================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    print(f"device={dev}  torch={torch.__version__}  networkx={'yes' if nx else 'NO'}\n")

    smoke(dev)
    if a.smoke:
        return

    t0 = time.time()
    raw = run_raw_ising(dev, seed=a.seed)
    sub = run_substrate(dev, seed=a.seed + 11)
    print(f"\ntotal wall: {time.time()-t0:.1f}s")

    out = Path(a.out or (Path(__file__).resolve().parent.parent / "runs" / "ising_organ.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(a), "raw_ising": raw, "substrate": sub}, indent=2, default=str))
    print(f"wrote {out}")

    # checkpoint: the (parameter-free) organ config + a solved exemplar instance with its exact ground state
    ckpt = Path(a.ckpt or (Path(__file__).resolve().parent.parent / "runs" / "ising_organ.pt"))
    rng = np.random.default_rng(a.seed)
    Jx, hx = rand_ising(rng, 16, kind="sk", device=dev)
    sx, Ex, Eworst = brute_force_ground(Jx, hx, device=dev)
    so, Eo, _ = mean_field_anneal(Jx, hx, restarts=64, steps=160, device=dev)
    torch.save({
        "organ": "ising_stochastic_mean_field_annealing",
        "solver_config": {"restarts": 64, "steps": 160, "temp0": 1.5, "temp1": 0.02, "damp": 0.4,
                          "noise": 2.0, "sto_noise": 1.0, "local_search": True},
        "exemplar": {"J": Jx.cpu(), "h": hx.cpu(), "ground_state": sx.cpu(),
                     "ground_E": Ex, "worst_E": Eworst, "organ_s": so.cpu(), "organ_E": Eo},
        "results": {"raw_ising": raw, "substrate": sub},
    }, ckpt)
    print(f"wrote checkpoint {ckpt}")


if __name__ == "__main__":
    main()
