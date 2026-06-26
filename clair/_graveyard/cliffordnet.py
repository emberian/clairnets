"""CliffordNet — a FAITHFUL reproduction of "CliffordNet: All You Need is Geometric Algebra"
(Zhongping Ji, arXiv 2601.06793). Supersedes clair/vision.py, which tested the WRONG regime: it
matched arms iso-param around a generic isotropic mixer and MISSED four mechanisms that the paper
says are exactly what make geometry win in the capacity-constrained regime. Those four (which our
prior GeomChan/Block lacked) are, explicitly:

  1. SELF-ENERGY SUPPRESSION (Eq. 5):  C = C_loc(H) - lambda * H, lambda=1 = "Differential Mode".
     The context that enters the geometric product is a discrete Laplacian (high-pass), NOT the
     raw smoothed context. The paper calls lambda=1 "optimal for capacity-constrained models" and
     Table 3 shows diff beats abs by ~1.4%. Our prior version fed the *smoothed* context only.
  2. TWO-STACK DEPTHWISE CONTEXT (Eq. 4):  C_loc(H) = DWConv3x3(DWConv3x3(H)) (factorized 7x7-ish
     receptive field via two cheap depthwise 3x3s + BN + SiLU). Prior version used ONE depthwise.
  3. GATED GEOMETRIC RESIDUAL (Eq. 13 / Alg. 1):  H_l = H_{l-1} + gamma * (SiLU(H_in) +
     Gate(H_in, H_geo) * H_geo), Gate = sigmoid(Linear([H_in; H_geo])), gamma = LayerScale. Prior
     version added the raw geometric output to a plain residual (no SiLU pre-filter, no gate, no
     LayerScale).
  4. GLOBAL SUPERPOSITION (Eqs. 6-7, gFFN-G):  optional second geometric interaction H (.)/(^)
     GlobalAvgPool(H), added with a beta switch. Nano/efficient variants beta=0; high-perf beta=1.
     Prior version had no global path at all.

The geometric interaction itself is the shifted geometric product (Eqs. 9-11, Alg. 1 lines 9-31):
for each cyclic channel shift s in S, with state H=Z_det (a 1x1 projection of the input) and
context C=Z_ctx (the Laplacian context above):
    Dot_s   = SiLU( H_c * C_{(c+s)%D} )                      grade-0 (coherence / gate)   "inner"
    Wedge_s = H_c * C_{(c+s)%D} - C_c * H_{(c+s)%D}          grade-2 (antisymmetric)      "wedge"
concat over S (and over {Dot,Wedge} for the full geometric product) -> 1x1 proj P back to D = H_geo.
NO FFN. Complexity O(N * D * |S|), pointwise over the HxW grid with torch.roll on the channel dim.

Isotropic columnar backbone: a conv patch-embed (kernel 2P-1, stride P) lifts 32x32 -> (32/P)^2 x D,
then L identical blocks at CONSTANT h x w x D (no downsampling), global-avg-pool -> linear head.

Variants (size_for-style D targeting at a fixed depth):
  Nano (~1.4M): depth 12, S={1,2},        full geom, lambda=1, beta=0   (paper's CliffordNet-Nano)
  Lite (~2.6M): depth 12, S={1,2,4,8,16},  full geom, lambda=1, beta=0   (paper's CliffordNet-Lite)
Ablation arms via config: mode in {full, wedge, inner(=dot)} x lambda in {0,1} x use_global in {0,1}.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- helpers
class LN2d(nn.Module):
    """LayerNorm over the channel axis of an [B,C,H,W] map (per-token vision LN)."""
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _shift_diags(z_det, z_ctx, s, mode):
    """The s-diagonal(s) of the geometric product between state z_det (H) and context z_ctx (C).

    Eq. 11 / Alg. 1 lines 12-15, with the cyclic roll T_s on the channel dim (dims=1 for [B,C,H,W]).
    Returns a list of [B,C,H,W] diagonals: Wedge then Dot for 'full'; just one for 'wedge'/'inner'.
    """
    h_roll = torch.roll(z_det, shifts=s, dims=1)         # H_{(c+s)%D}
    c_roll = torch.roll(z_ctx, shifts=s, dims=1)         # C_{(c+s)%D}
    if mode == "inner":                                  # grade-0 only (dot / coherence gate)
        return [F.silu(z_det * c_roll)]
    wedge = z_det * c_roll - z_ctx * h_roll              # grade-2 bivector (antisymmetric)
    if mode == "wedge":                                  # grade-2 only (structure, zero self-energy)
        return [wedge]
    dot = F.silu(z_det * c_roll)                         # grade-0 (coherence)
    return [wedge, dot]                                  # full geometric product


# --------------------------------------------------------------------------- the block
class CliffordBlock(nn.Module):
    """One isotropic CliffordNet block (Algorithm 1, the No-gFFN-G variant, plus the optional
    global superposition path). All sub-mechanisms are present and individually toggleable.

    mode       : 'full' (Dot+Wedge), 'wedge' (Wedge-only), 'inner'/'dot' (Dot-only)   [ablation]
    lam        : self-energy suppression lambda in {0,1}; 1 = Differential Mode (Laplacian context)
    use_global : add the beta-gated global-superposition interaction (gFFN-G, Eq. 7)
    """
    def __init__(self, c, shifts=(1, 2), mode="full", lam=1.0, use_global=False,
                 beta=1.0, ls_init=1e-4, drop_path=0.0):
        super().__init__()
        self.shifts = tuple(shifts)
        self.mode = "inner" if mode == "dot" else mode
        self.lam = float(lam)
        self.use_global = use_global
        self.beta = float(beta)
        self.drop_path = float(drop_path)
        k = 2 if self.mode == "full" else 1              # diagonals produced per shift
        ncat = k * len(self.shifts) * c                  # concat width feeding the projection

        self.norm = LN2d(c)
        # dual-stream generation (Alg. 1 lines 4-5)
        self.det = nn.Conv2d(c, c, 1, bias=True)                       # Z_det = Linear_det(X_in) = H
        self.ctx1 = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)  # C_loc stack: DWConv3x3 ...
        self.ctx2 = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)  #          ... DWConv3x3 (Eq.4)
        self.bn1 = nn.BatchNorm2d(c)
        self.bn2 = nn.BatchNorm2d(c)
        # geometric-interaction projections P (1x1), local and (optional) global
        self.proj = nn.Conv2d(ncat, c, 1, bias=True)                  # P_loc -> H_geo
        if use_global:
            self.proj_glo = nn.Conv2d(ncat, c, 1, bias=True)          # P_glo (gFFN-G)
        # Gated Geometric Residual (Alg. 1 lines 33-39)
        self.gate = nn.Conv2d(2 * c, c, 1, bias=True)                 # Gate = sigmoid(Linear([H_in;H_geo]))
        self.gamma = nn.Parameter(ls_init * torch.ones(1, c, 1, 1))   # LayerScale (Euler step size)

    def _context(self, x_in):
        """C = C_loc(x_in) - lambda * Z_det,  C_loc = SiLU(BN(DWConv(SiLU(BN(DWConv(x)))))) (Eq.4-5).
        Returns (z_det, z_ctx). With lambda=1 this makes z_ctx a discrete-Laplacian high-pass."""
        z_det = self.det(x_in)                                        # H
        h = F.silu(self.bn1(self.ctx1(x_in)))
        c_loc = F.silu(self.bn2(self.ctx2(h)))                        # two-stack depthwise context
        z_ctx = c_loc - self.lam * z_det                             # self-energy suppression (Eq. 5)
        return z_det, z_ctx

    def _interact(self, z_det, z_ctx, proj):
        outs = []
        for s in self.shifts:
            outs.extend(_shift_diags(z_det, z_ctx, s, self.mode))
        return proj(torch.cat(outs, dim=1))                          # concat over S -> P -> H_geo

    def forward(self, x):
        x_in = self.norm(x)
        z_det, z_ctx = self._context(x_in)
        g_feat = self._interact(z_det, z_ctx, self.proj)             # local geometric interaction
        if self.use_global:
            c_glo = z_det.mean(dim=(2, 3), keepdim=True)            # C_glo = GlobalAvgPool(H), Eq. 6
            g_glo = self._interact(z_det, c_glo.expand_as(z_det), self.proj_glo)
            g_feat = g_feat + self.beta * g_glo                      # field superposition (Eq. 7)
        # Gated Geometric Residual (Eq. 13)
        alpha = torch.sigmoid(self.gate(torch.cat([x_in, g_feat], dim=1)))
        h_mix = F.silu(x_in) + alpha * g_feat
        h_mix = self.gamma * h_mix
        if self.training and self.drop_path > 0.0:
            keep = 1.0 - self.drop_path
            mask = torch.empty(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype).bernoulli_(keep)
            h_mix = h_mix * mask / keep
        return x + h_mix


# --------------------------------------------------------------------------- the network
class CliffordNet(nn.Module):
    """Isotropic columnar CliffordNet: patch-embed conv -> L identical CliffordBlocks at constant
    h x w x D -> final LN -> global-avg-pool -> linear head."""
    def __init__(self, depth=12, width=128, patch=2, shifts=(1, 2), mode="full", lam=1.0,
                 use_global=False, beta=1.0, n_classes=100, in_ch=3, ls_init=1e-4, drop_path=0.0):
        super().__init__()
        self.cfg = dict(depth=depth, width=width, patch=patch, shifts=tuple(shifts), mode=mode,
                        lam=lam, use_global=use_global, beta=beta, n_classes=n_classes)
        ks = max(1, 2 * patch - 1)
        self.stem = nn.Conv2d(in_ch, width, kernel_size=ks, stride=patch,
                              padding=ks // 2, bias=True)            # patch-embed (kernel 2P-1, stride P)
        dprs = [drop_path * i / max(1, depth - 1) for i in range(depth)]  # linear stochastic-depth ramp
        self.blocks = nn.ModuleList([
            CliffordBlock(width, shifts=shifts, mode=mode, lam=lam, use_global=use_global,
                          beta=beta, ls_init=ls_init, drop_path=dprs[i]) for i in range(depth)])
        self.norm_f = LN2d(width)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x):
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm_f(h)
        h = h.mean(dim=(2, 3))                                       # global average pool
        return self.head(h)

    def n_params(self, non_embed=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embed:                                                # exclude stem + head ("embedding")
            n -= sum(p.numel() for p in self.stem.parameters())
            n -= sum(p.numel() for p in self.head.parameters())
        return n


# --------------------------------------------------------------------------- pure-python param model
def _block_params(c, n_shifts, mode, use_global):
    """Closed-form CliffordBlock param count (no torch), for size_for / offline checks."""
    k = 2 if mode == "full" else 1
    ncat = k * n_shifts * c
    p = 0
    p += 2 * c                                  # LN (weight+bias)
    p += c * c + c                              # det 1x1 (+bias)
    p += 9 * c + 9 * c                          # two depthwise 3x3 (groups=c, no bias)
    p += 2 * c + 2 * c                          # two BatchNorm2d (weight+bias)
    p += ncat * c + c                           # proj P_loc 1x1 (+bias)
    if use_global:
        p += ncat * c + c                       # proj P_glo 1x1 (+bias)
    p += (2 * c) * c + c                        # gate 1x1: 2c->c (+bias)
    p += c                                      # LayerScale gamma
    return p


def n_params_for(depth, c, patch, n_shifts, mode, use_global, n_classes=100, in_ch=3,
                 non_embed=False):
    ks = max(1, 2 * patch - 1)
    stem = ks * ks * in_ch * c + c
    head = c * n_classes + n_classes
    body = depth * _block_params(c, n_shifts, mode, use_global) + 2 * c   # + final LN
    return body if non_embed else body + stem + head


def size_for(depth, target, patch=2, shifts=(1, 2), mode="full", use_global=False,
             n_classes=100, mult=16):
    """Pick the largest width c (multiple of `mult`, and >= max shift) whose TOTAL params fit
    `target`, at the given depth/shifts/mode. Returns (c, total_params). Pure python (offline)."""
    need = max(shifts) + 1 if shifts else mult
    c = max(mult, (need // mult) * mult)
    best = (c, n_params_for(depth, c, patch, len(shifts), mode, use_global, n_classes))
    while True:
        nc = best[0] + mult
        tot = n_params_for(depth, nc, patch, len(shifts), mode, use_global, n_classes)
        if tot > target or nc > 2048:
            break
        best = (nc, tot)
    return best


# named paper variants: (depth, shifts, mode, lam, use_global, beta, target) -> resolved width
_VARIANTS = {
    "nano": dict(depth=12, shifts=(1, 2), mode="full", lam=1.0, use_global=False, beta=0.0,
                 target=1.45e6),
    "lite": dict(depth=12, shifts=(1, 2, 4, 8, 16), mode="full", lam=1.0, use_global=False,
                 beta=0.0, target=2.65e6),
    # high-performance arms from Table 2 (global superposition on)
    "cn32": dict(depth=16, shifts=(1, 2, 4), mode="inner", lam=1.0, use_global=True, beta=1.0,
                 target=5.0e6),
    "cn64": dict(depth=20, shifts=(1, 2, 4, 8, 16), mode="full", lam=1.0, use_global=True,
                 beta=1.0, target=8.8e6),
}


def build_variant(name, patch=2, n_classes=100, lam=None, use_global=None, mode=None,
                  drop_path=0.0):
    """Construct a named variant, with optional ablation overrides (lam / mode / use_global)."""
    v = dict(_VARIANTS[name])
    if lam is not None:
        v["lam"] = lam
    if mode is not None:
        v["mode"] = mode
    if use_global is not None:
        v["use_global"] = use_global
    c, _ = size_for(v["depth"], v["target"], patch=patch, shifts=v["shifts"], mode=v["mode"],
                    use_global=v["use_global"], n_classes=n_classes)
    return CliffordNet(depth=v["depth"], width=c, patch=patch, shifts=v["shifts"], mode=v["mode"],
                       lam=v["lam"], use_global=v["use_global"], beta=v["beta"],
                       n_classes=n_classes, drop_path=drop_path)


# --------------------------------------------------------------------------- offline self-test
def _selftest():
    import numpy as np

    # 1. wedge antisymmetry + zero self-energy, numerically, on the exact shift formula --------
    rng = np.random.default_rng(0)
    B, C, H, W = 2, 12, 5, 5
    Hs = rng.standard_normal((B, C, H, W)).astype(np.float64)
    Cs = rng.standard_normal((B, C, H, W)).astype(np.float64)
    s = 3

    def wedge(a, b):                                  # numpy mirror of _shift_diags wedge
        return a * np.roll(b, s, axis=1) - b * np.roll(a, s, axis=1)
    assert np.allclose(wedge(Hs, Cs), -wedge(Cs, Hs)), "wedge not antisymmetric under H<->C"
    assert np.allclose(wedge(Hs, Hs), 0.0), "self-wedge (H==H) must vanish (no self-energy)"

    # 2. lambda=1 context == a discrete Laplacian. A normalized 3x3 mean filter M obeys
    #    M(H) - H = (1/9) * (sum_8neighbors - 8*center) = (1/9) * Laplacian_8(H).  -----------
    img = rng.standard_normal((1, 1, 9, 9)).astype(np.float64)

    def mean3x3(a):                                   # reflect-pad 3x3 box blur
        p = np.pad(a, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="reflect")
        out = np.zeros_like(a)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                out += p[:, :, 1 + di:1 + di + a.shape[2], 1 + dj:1 + dj + a.shape[3]]
        return out / 9.0

    def lap8(a):                                      # 8-neighbor discrete Laplacian
        p = np.pad(a, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="reflect")
        out = -8.0 * a
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                out += p[:, :, 1 + di:1 + di + a.shape[2], 1 + dj:1 + dj + a.shape[3]]
        return out
    ctx = mean3x3(img) - 1.0 * img                    # C = C_loc(H) - lambda*H,  lambda=1
    assert np.allclose(ctx, lap8(img) / 9.0), "lambda=1 context != discrete Laplacian"
    # and lambda=0 is the raw smoothed (low-pass) context, NOT a Laplacian
    assert not np.allclose(mean3x3(img) - 0.0 * img, lap8(img) / 9.0), "lambda=0 should differ"

    # 3. closed-form param model is self-consistent; variants land near their targets ----------
    for name, v in _VARIANTS.items():
        c, tot = size_for(v["depth"], v["target"], shifts=v["shifts"], mode=v["mode"],
                          use_global=v["use_global"])
        rel = (v["target"] - tot) / v["target"]
        assert 0.0 <= rel < 0.12, (name, c, tot, v["target"], rel)   # fits under target, not too loose
        assert c % 16 == 0 and c > max(v["shifts"]), (name, c)
    cn, tn = size_for(12, 1.45e6, shifts=(1, 2), mode="full")
    print(f"nano-resolve: width={cn} total~{tn:,}")
    print("cliffordnet _selftest ok")


if __name__ == "__main__":
    _selftest()
