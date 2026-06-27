"""clair/organ/readout_bridge.py — RICH readout-bridges for the non-CSP-lattice beasts (blocker 4).

The structured-γ spine reads every CSP-domain organ as a per-(cell, candidate) survival matrix. Two
beasts do NOT narrow a per-cell lattice and so cannot be read that way without throwing information
away:

  * ENERGY / ISING (optimisation) reads out an ASSIGNMENT, its ENERGY, and a CONFIDENCE — the answer
    is the argmax assignment, the energy says how good it is, the marginal sharpness says how sure the
    organ is. The rich feature exposes all three to the LM (reusing the alien_consumer rich-state
    idea: per cell [assignment(K), |spread|, reliability], plus the global energy).
  * UNIFICATION (FOL entailment) reads out a 3-way LABEL (entail / contradict / unknown) + derived
    facts — there is no per-cell domain at all. The rich feature is the label one-hot + how much was
    derived + the proof depth.

Each bridge returns a per-"readout-cell" feature matrix [N, Fin] that clair.oracle_readout.OracleGamma
scatters into the host residual stream through the SAME zero-init-gate channel (bitwise no-op at init).

For ENERGY/ISING the annealer is NON-DIFFERENTIABLE (torch.no_grad mean-field + sound local search), so
α gets NO gradient from the LM-CE through the organ. `CouplingProjector` is α's compile head for this
faculty (host hidden → the Ising couplings (J, h)); it is trained by DIRECT coupling-supervision
(`coupling_loss`: MSE against the true (J, h) of the compiled problem), exactly as J0 grounds the CSP
α. Without it the energy faculty cannot be woven (no gradient path to α).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================ ENERGY / ISING rich readout
def energy_readout(csp, K, steps=140, device="cpu"):
    """Run the energy organ on a CSP and build the RICH per-cell readout feature [n, K+2]:
        [:, :K]   the decoded soft assignment marginals x*[i] (argmax = answer)
        [:, K]    per-cell confidence = max marginal (how peaked the cell is)
        [:, K+1]  the GLOBAL solution quality = exp(-soft_resid) in [0,1] (broadcast to every cell)
    Returns (feat[n, K+2], assignment[n], info)."""
    from .. import energy_organ as EN
    facs = EN.factor_tensors(csp, device)
    x, info = EN.relax_decode(facs, csp.n, csp.d, device, steps=steps)
    xn = x.detach().cpu().numpy()
    assign = [int(xn[i].argmax()) for i in range(csp.n)]
    feat = np.zeros((csp.n, K + 2), dtype=np.float32)
    kd = min(csp.d, K)
    feat[:, :kd] = xn[:, :kd]
    feat[:, K] = xn[:, :csp.d].max(-1)                         # per-cell confidence
    feat[:, K + 1] = float(np.exp(-info["soft_resid"]))        # global quality (1.0 = feasible)
    return feat, assign, info


def ising_readout(J, h, K=2, restarts=64, steps=160, device="cpu", seed=0):
    """Run the Ising organ (mean-field annealing + sound local search) on couplings (J, h) and build the
    rich per-spin readout feature [n, K+2]:
        [:, 0], [:, 1]   the spin one-hot ( -1 -> col 0, +1 -> col 1 )
        [:, K]           per-spin LOCAL-FIELD MARGIN confidence in [0,1]: how strongly this spin is
                         pinned by the rest of the configuration (the flip-cost |f_i| = |h_i + Σ_j J_ij s_j|,
                         normalised by the largest margin in the instance). 1.0 = most-pinned spin;
                         near 0 = a marginal spin the organ is genuinely unsure about. (A relative
                         per-instance confidence, NOT a calibrated probability.)
        [:, K+1]         the GLOBAL energy, min-max normalised to [0,1] (broadcast)
    Returns (feat[n, K+2], spins[n] in ±1, energy)."""
    from .. import ising_organ as IS
    Jt = J if torch.is_tensor(J) else torch.as_tensor(np.asarray(J, np.float32), device=device)
    ht = h if torch.is_tensor(h) else torch.as_tensor(np.asarray(h, np.float32), device=device)
    Jt = Jt.to(device).float(); ht = ht.to(device).float()
    s, E, _info = IS.mean_field_anneal(Jt, ht, restarts=restarts, steps=steps, device=device, seed=seed)
    sn = s.detach().cpu().numpy()
    n = len(sn)
    feat = np.zeros((n, K + 2), dtype=np.float32)
    for i in range(n):
        feat[i, 1 if sn[i] > 0 else 0] = 1.0
    # real per-spin confidence = the local-field margin |h_i + Σ_j J_ij s_j| (the energy cost of
    # flipping spin i), normalised by the largest margin so it lands in [0,1]. Replaces the old
    # hardcoded 1.0 (which falsely claimed every spin was maximally confident).
    field = IS.local_field(Jt, ht, s.float()).abs().detach().cpu().numpy()
    feat[:, K] = field / max(1e-6, float(field.max()))
    # normalise the (negative) energy against a cheap random-config band so the scalar is in [0,1]
    g = torch.Generator(device=device).manual_seed(seed)
    rs = torch.where(torch.rand(32, n, device=device, generator=g) < 0.5, -1.0, 1.0)
    band = IS.ising_energy(Jt, ht, rs)
    lo, hi = float(band.min()), float(band.max())
    feat[:, K + 1] = float(np.clip((hi - E) / max(1e-6, hi - lo), 0.0, 1.0))   # 1.0 = best found
    return feat, sn.astype(np.int64), float(E)


def energy_feat_dim(K):
    return K + 2


# ============================================================ UNIFICATION (FOL entailment) rich readout
_FOL_LABELS = ("entail", "contradict", "unknown")


def unification_readout(facts, rules, query, K, depth_cap=8):
    """Run forward-chaining and build the RICH entailment readout feature [1, K+2]:
        [0, :3]    the 3-way label one-hot (entail / contradict / unknown)
        [0, K]     #derived-facts beyond the givens, normalised (how much the organ entailed)
        [0, K+1]   the query's proof depth / depth_cap (0 if unknown) — the difficulty signal
    Returns (feat[1, K+2], label, closure)."""
    from .. import fol as FOL
    closure = FOL.forward_chain(facts, rules)
    label = FOL.label_query(closure, query)
    feat = np.zeros((1, K + 2), dtype=np.float32)
    li = _FOL_LABELS.index(label)
    if li < K:
        feat[0, li] = 1.0
    feat[0, K] = min(1.0, max(0, len(closure) - len(facts)) / max(1, len(facts)))
    if label != "unknown":
        depths = FOL.proof_depths(facts, rules, closure)
        q = query if label == "entail" else FOL.neg_of(query)
        feat[0, K + 1] = min(1.0, depths.get(q, 0) / depth_cap)
    return feat, label, closure


def unification_feat_dim(K):
    return K + 2


# ============================================================ α → Ising couplings compile head
class CouplingProjector(nn.Module):
    """α's compile head for the ENERGY/ISING faculty: host hidden (per-cell pooled identity v[B,N,D])
    → the Ising couplings (J[B,N,N] symmetric zero-diag, h[B,N]). A pair-MLP gives J_ij from (v_i, v_j)
    symmetrically; a unary MLP gives h_i. Trained by DIRECT coupling-supervision (coupling_loss) because
    the annealer is non-differentiable — this is the (J,h) analogue of the CSP α's J0 grounding."""

    def __init__(self, D, dp=128):
        super().__init__()
        self.pair = nn.Sequential(nn.Linear(2 * D, dp), nn.GELU(), nn.Linear(dp, 1))
        self.field = nn.Sequential(nn.Linear(D, dp), nn.GELU(), nn.Linear(dp, 1))

    def forward(self, v, vmask=None):
        # v [B,N,D]
        B, N, D = v.shape
        vi = v.unsqueeze(2).expand(B, N, N, D)
        vj = v.unsqueeze(1).expand(B, N, N, D)
        pair_in = torch.cat([vi, vj], dim=-1)
        Jraw = self.pair(pair_in).squeeze(-1)                  # [B,N,N]
        J = 0.5 * (Jraw + Jraw.transpose(1, 2))                # symmetrise
        eye = torch.eye(N, device=v.device, dtype=J.dtype).unsqueeze(0)
        J = J * (1 - eye)                                      # zero diagonal
        h = self.field(v).squeeze(-1)                          # [B,N]
        if vmask is not None:
            m2 = vmask.unsqueeze(1) * vmask.unsqueeze(2)
            J = J * m2
            h = h * vmask
        return J, h


def coupling_loss(J_pred, h_pred, J_true, h_true, vmask=None):
    """DIRECT coupling supervision: MSE of α's compiled (J,h) against the true Ising couplings of the
    problem (the annealer gives α no gradient, so this is α's only learning signal for this faculty)."""
    if vmask is not None:
        m2 = vmask.unsqueeze(1) * vmask.unsqueeze(2)
        jl = (((J_pred - J_true) ** 2) * m2).sum() / m2.sum().clamp_min(1.0)
        hl = (((h_pred - h_true) ** 2) * vmask).sum() / vmask.sum().clamp_min(1.0)
    else:
        jl = F.mse_loss(J_pred, J_true)
        hl = F.mse_loss(h_pred, h_true)
    return jl + hl
