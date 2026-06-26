"""ClairVision — one small image classifier, four pluggable block mixers, iso-param-comparable.

The bet: the geometric product is a SPATIAL/VISION primitive, not a language one. Our language
sweep found geom LOSES to SwiGLU at scale on text; CliffordNet (2601.06793) got its wins on VISION
(CIFAR-100: 76-78% at 1.4-3M params, beating ResNet-18). So we test it on its HOME TURF.

Arms (the per-block CHANNEL mixer; the spatial mix + stem + head are shared, iso-param):
  geom  : CliffordNet-style geometric-product channel mixer, NO FFN          (the home-turf bet)
          shifted geometric product over cyclic channel shifts S: per shift s, a grade-0 DOT
          diagonal D_s = SiLU(H_c · C_{c+s}) and a grade-2 WEDGE diagonal W_s = H_c·C_{c+s} -
          C_c·H_{c+s}; concat the 2|S| diagonals and project back to C. O(N·C·|S|), |S| small.
  wedge : geom, WEDGE-ONLY ablation (drop the dot diagonals) — CliffordNet Table 4 says wedge
          alone (zero self-energy) nearly matches both. Does that hold here?
  conv  : depthwise-separable conv block (depthwise 3x3 + pointwise) — the standard vision baseline.
  mlp   : MLP-mixer-style channel MLP (token MLP lives in the shared spatial slot) — the FFN
          baseline, i.e. the LANGUAGE winner. Does the language result reverse on images?

Each block: LN -> shared SPATIAL mix (depthwise 3x3 conv over the HxW grid, all arms) -> chosen
CHANNEL mixer -> residual. A conv stem (3x3 stride-p patchify) lifts the image to a CxH'xW' token
grid; N blocks; global-avg-pool -> linear head. size_for() iso-param-matches every arm at a target
non-embed budget by binary-searching the geom/mlp width and the conv block's depth-channels.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- channel mixers
class GeomChan(nn.Module):
    """Shifted geometric-product channel mixer (CliffordNet 2601.06793), per spatial token.

    Lift x:[B,C,H,W] -> two streams u,w via 1x1 convs to width `m`. For each cyclic channel
    shift s in `shifts`, take the s-diagonal of the full geometric product:
        dot   D_s = SiLU(u_c * w_{c+s})                 grade-0 (coherence / gating)
        wedge W_s = u_c * w_{c+s} - w_c * u_{c+s}        grade-2 (antisymmetric; u^u=0, no self-energy)
    Concat the (1+wedge_only? :2)*|S| diagonals over channels and project (1x1) back to C. NO FFN.
    Roll is O(C) per shift, so the whole mixer is O(N*C*|S|) — the paper's O(N) claim.
    """
    def __init__(self, c, m, shifts=(1, 2, 4, 8, 15), wedge_only=False):
        super().__init__()
        self.shifts = tuple(shifts)
        self.wedge_only = wedge_only
        self.u = nn.Conv2d(c, m, 1, bias=False)
        self.w = nn.Conv2d(c, m, 1, bias=False)
        grades = 1 if wedge_only else 2                  # diagonals produced per shift
        self.o = nn.Conv2d(grades * m * len(self.shifts), c, 1, bias=False)

    def forward(self, x):
        u = self.u(x)
        w = self.w(x)
        outs = []
        for s in self.shifts:
            us = torch.roll(u, shifts=s, dims=1)         # u_{c+s} along channels (cyclic)
            ws = torch.roll(w, shifts=s, dims=1)
            wedge = u * ws - w * us                       # W_s grade-2 bivector diagonal
            if self.wedge_only:
                outs.append(wedge)
            else:
                dot = F.silu(u * ws)                      # D_s grade-0 scalar diagonal
                outs.append(dot)
                outs.append(wedge)
        return self.o(torch.cat(outs, dim=1))


class ConvChan(nn.Module):
    """Depthwise-separable conv block: depthwise 3x3 (local mixing) + pointwise expand/contract.
    The standard efficient-vision baseline; matched in params via the expansion width `m`."""
    def __init__(self, c, m):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.pw1 = nn.Conv2d(c, m, 1, bias=False)
        self.pw2 = nn.Conv2d(m, c, 1, bias=False)

    def forward(self, x):
        return self.pw2(F.silu(self.pw1(self.dw(x))))


class MlpChan(nn.Module):
    """MLP-mixer channel MLP (the FFN / language winner), applied per spatial token via 1x1 convs."""
    def __init__(self, c, m):
        super().__init__()
        self.fc1 = nn.Conv2d(c, m, 1, bias=False)
        self.fc2 = nn.Conv2d(m, c, 1, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# --------------------------------------------------------------------------- block + net
class LN2d(nn.Module):
    """LayerNorm over channels of an [B,C,H,W] map (per-token, like a vision LN)."""
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def make_chan(arm, c, m):
    if arm == "geom":
        return GeomChan(c, m, wedge_only=False)
    if arm == "wedge":
        return GeomChan(c, m, wedge_only=True)
    if arm == "conv":
        return ConvChan(c, m)
    if arm == "mlp":
        return MlpChan(c, m)
    raise ValueError(arm)


class Block(nn.Module):
    """LN -> shared spatial mix (depthwise 3x3 over the grid) -> channel mixer -> residual.
    The spatial depthwise conv is identical across arms (CliffordNet's depthwise-conv context
    branch); the ONLY thing that varies between arms is the channel mixer."""
    def __init__(self, arm, c, m):
        super().__init__()
        self.ln = LN2d(c)
        self.spatial = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.chan = make_chan(arm, c, m)

    def forward(self, x):
        return x + self.chan(self.spatial(self.ln(x)))


class ClairVision(nn.Module):
    def __init__(self, arm, c, m, n_blocks=6, n_classes=100, in_ch=3, patch=4):
        super().__init__()
        self.arm = arm
        self.stem = nn.Conv2d(in_ch, c, kernel_size=patch * 2 - 1, stride=patch,
                              padding=patch - 1, bias=False)      # 3x3-ish stride-p patchify
        self.blocks = nn.ModuleList([Block(arm, c, m) for _ in range(n_blocks)])
        self.ln_f = LN2d(c)
        self.head = nn.Linear(c, n_classes)

    def forward(self, x):
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        h = h.mean(dim=(2, 3))                                    # global avg pool
        return self.head(h)

    def n_params(self, non_embed=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embed:                                            # exclude stem + head (the "embedding")
            n -= sum(p.numel() for p in self.stem.parameters())
            n -= sum(p.numel() for p in self.head.parameters())
        return n


# --------------------------------------------------------------------------- iso-param sizing
def _mixer_params(arm, c, m, shifts):
    """Closed-form channel-mixer param count (no torch) for the iso-param search."""
    if arm == "geom":
        return 2 * c * m + 2 * m * len(shifts) * c               # u,w 1x1 (c*m each) ; o 1x1 (2|S|m -> c)
    if arm == "wedge":
        return 2 * c * m + m * len(shifts) * c                   # o input is |S|m
    if arm == "conv":
        return 9 * c + c * m + m * c                             # dw 3x3 (9c) + pw1 + pw2
    if arm == "mlp":
        return 2 * c * m                                         # fc1 + fc2
    raise ValueError(arm)


def n_params_for(arm, c, m, n_blocks, shifts):
    """Non-embed params: per block = LN(2c) + spatial dw(9c) + LN_f shared once. Stem/head excluded."""
    per_block = 2 * c + 9 * c + _mixer_params(arm, c, m, shifts)
    return n_blocks * per_block + 2 * c                          # + final ln_f


def size_for(arm, target, n_blocks=6, shifts=(1, 2, 4, 8, 15), c_fixed=None):
    """Iso-param sizing. Two knobs share the budget: channel width c and mixer width m.

    Two-stage so every arm lands tight AND matched: (1) coarse-search c (multiple of 8 so the
    geometric shifts and head pooling stay clean) with m=2c as the natural width, taking the
    LARGEST c whose params fit; (2) hold c and fine-search m to top the budget back up to ~target.
    If c_fixed is given, skip stage 1 (hold the spatial path identical, vary only mixer capacity).

    Returns (c, m, n_params). Pure python — runs without torch for offline iso-param checks.
    """
    if c_fixed is not None:
        c = max(8, (c_fixed // 8) * 8)
        m, np_ = _search_m(arm, target, c, n_blocks, shifts)
        return (c, m, np_)
    # stage 1: largest c (at the natural m=2c) whose params fit the budget
    c = 8
    while True:
        nc = c + 8
        if n_params_for(arm, nc, 2 * nc, n_blocks, shifts) > target or nc > 1024:
            break
        c = nc
    # stage 2: top up m at that c
    m, np_ = _search_m(arm, target, c, n_blocks, shifts)
    return (c, m, np_)


def _search_m(arm, target, c, n_blocks, shifts):
    """Binary-search mixer width m (>=4, multiple of 4) for non-embed params <= target at fixed c."""
    lo, hi, best = 1, 8192, (4, n_params_for(arm, c, 4, n_blocks, shifts))
    while lo <= hi:
        k = (lo + hi) // 2
        m = 4 * k
        np_ = n_params_for(arm, c, m, n_blocks, shifts)
        if np_ <= target:
            best = (m, np_); lo = k + 1
        else:
            hi = k - 1
    return best


# ----- pure-logic self-test (no torch): exercise the param accounting & iso-param search --
def _selftest():
    # 1. closed-form mixer counts match the module construction rules
    assert _mixer_params("geom", 64, 128, (1, 2, 4)) == 2 * 64 * 128 + 2 * 128 * 3 * 64
    assert _mixer_params("wedge", 64, 128, (1, 2, 4)) == 2 * 64 * 128 + 128 * 3 * 64
    assert _mixer_params("conv", 64, 128, (1,)) == 9 * 64 + 64 * 128 + 128 * 64
    assert _mixer_params("mlp", 64, 128, (1,)) == 2 * 64 * 128
    # 2. size_for hits target tightly for every arm, several budgets; arms within a few % of each other
    for target in (1.5e6, 3e6, 6e6):
        sizes = {}
        for arm in ("geom", "wedge", "conv", "mlp"):
            c, m, np_ = size_for(arm, target, n_blocks=6)
            sizes[arm] = np_
            assert c % 8 == 0, (arm, c)
            rel = abs(np_ - target) / target
            assert rel < 0.05, (arm, target, np_, rel)
        lo, hi = min(sizes.values()), max(sizes.values())
        assert (hi - lo) / target < 0.05, (target, sizes)
    print("vision _selftest ok")


if __name__ == "__main__":
    _selftest()
