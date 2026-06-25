"""GeomLM — one small causal byte/token LM, one pluggable channel mixer, five arms,
iso-param-comparable. The scientific question: is the Clifford geometric product
(inner u·v + wedge u∧v) a more param-efficient channel mixer than a SwiGLU FFN for
autoregressive LM, and which part (dot / wedge / grade-3) carries the load?

Attention is SHARED and identical across arms; only the per-block mixer changes:

  swiglu  : SwiGLU FFN, h = round(ratio*d)                         baseline
  geom    : geometric product on 2D blades, inner + wedge          the bet (grades 0,2)
  dot     : inner-only (symmetric / Hadamard coherence)            ablation
  wedge   : wedge-only (antisymmetric area)                        ablation
  geom_g3 : geom (grades 0,2) PLUS a grade-3 triple term across    novel probe
            THREE streams: the scalar triple product on 3D blades.

All FFN-shaped mixers take (d, h); we set h via size_for() binary search so every
arm has ~equal NON-EMBED params at a target. See GRADE-3 note below.

Reused from clairnets/clair/model.py: SwiGLU, GeomMix idea, Attn, Block, size_for
binary-search pattern. Reused from clairnets/clair/graft.py: GeomFFN(d,h) width-h
FFN-shaped geom mixer (the dot+wedge unflatten-to-2D-blades trick). Reused from
graphplay/experiments/tinystories_arch/model.py: tied head, generate(), the
byte-LM GPT skeleton (vocab=256). The byte data-loader + bits-per-byte eval loop
in run_geom_lm.py is lifted from that same train.py.

torch is import-guarded so size_for() / the param-accounting self-test run on CPU
with no torch (numpy-only). The torch path is exercised only on the GPU box.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:                       # pure-logic self-test path (no torch locally)
    torch = nn = F = None
    _HAS_TORCH = False


# ----- the five FFN-shaped mixers; each is (d -> h ... -> d) with a chosen blade op -----

if _HAS_TORCH:

    class SwiGLU(nn.Module):
        """Baseline: gated FFN. params = 3*d*h (w1,w2: d->h ; w3: h->d)."""
        def __init__(self, d, h):
            super().__init__()
            self.w1 = nn.Linear(d, h, bias=False)
            self.w2 = nn.Linear(d, h, bias=False)
            self.w3 = nn.Linear(h, d, bias=False)

        def forward(self, x):
            return self.w3(F.silu(self.w1(x)) * self.w2(x))

    class GeomMix(nn.Module):
        """grades 0 & 2: split h into 2D blades, per-blade inner u·v and wedge u∧v,
        concatenate (back to width h) then project. params = 3*d*h (u,w: d->h ; o: h->d)."""
        def __init__(self, d, h, mode="geom"):
            super().__init__()
            assert mode in ("geom", "dot", "wedge")
            if h % 2:
                h += 1
            self.mode = mode
            self.u = nn.Linear(d, h, bias=False)
            self.w = nn.Linear(d, h, bias=False)
            self.o = nn.Linear(h, d, bias=False)

        def forward(self, x):
            u = self.u(x).unflatten(-1, (-1, 2))          # (..., h/2, 2)
            w = self.w(x).unflatten(-1, (-1, 2))
            inner = (u * w).sum(-1)                        # (..., h/2)  grade 0
            wedge = u[..., 0] * w[..., 1] - u[..., 1] * w[..., 0]   # (..., h/2)  grade 2
            if self.mode == "dot":
                # dot-only: duplicate inner into both slots so the o-projection still sees width h
                z = torch.cat([inner, inner], -1)
            elif self.mode == "wedge":
                z = torch.cat([wedge, wedge], -1)
            else:
                z = torch.cat([inner, wedge], -1)         # width h
            return self.o(z)

    class GeomG3(nn.Module):
        """grades 0, 2 AND a grade-3 triple term across THREE streams.

        Split h into 3D blades (groups of 3 coords). For a blade with streams
        u=(u0,u1,u2), w=(w0,w1,w2) and a third projection t=(t0,t1,t2):
          inner  = u·w                                   (grade 0, sym pairwise)
          wedge  = u0 w1 - u1 w0                          (grade 2, antisym pairwise)
          trip   = det([u;w;t])                           (grade 3, the 3-blade volume)
                 = u0(w1 t2 - w2 t1) - u1(w0 t2 - w2 t0) + u2(w0 t1 - w1 t0)
        This is the scalar triple product u·(w×t): the unique fully-antisymmetric
        trilinear invariant of three 3-vectors, i.e. the coefficient of e1∧e2∧e3 in
        the wedge u∧w∧t of three grade-1 streams in Cl(3). CliffordNet used only
        grades 0 & 2 from a 2-blade product; this adds a genuine grade-3 (pseudoscalar)
        interaction that no pair of streams can produce. params = 4*d*h
        (u,w,t: d->h ; o: h->d), so size_for compensates by shrinking h vs geom.
        """
        def __init__(self, d, h):
            super().__init__()
            if h % 3:
                h += 3 - (h % 3)
            self.u = nn.Linear(d, h, bias=False)
            self.w = nn.Linear(d, h, bias=False)
            self.t = nn.Linear(d, h, bias=False)
            self.o = nn.Linear(h, d, bias=False)

        def forward(self, x):
            u = self.u(x).unflatten(-1, (-1, 3))          # (..., h/3, 3)
            w = self.w(x).unflatten(-1, (-1, 3))
            t = self.t(x).unflatten(-1, (-1, 3))
            u0, u1, u2 = u[..., 0], u[..., 1], u[..., 2]
            w0, w1, w2 = w[..., 0], w[..., 1], w[..., 2]
            t0, t1, t2 = t[..., 0], t[..., 1], t[..., 2]
            inner = u0 * w0 + u1 * w1 + u2 * w2                       # grade 0  (..., h/3)
            wedge = u0 * w1 - u1 * w0                                 # grade 2  (..., h/3)
            trip = (u0 * (w1 * t2 - w2 * t1)
                    - u1 * (w0 * t2 - w2 * t0)
                    + u2 * (w0 * t1 - w1 * t0))                       # grade 3  (..., h/3)
            return self.o(torch.cat([inner, wedge, trip], -1))       # width h

    def make_mixer(arm, d, h):
        if arm == "swiglu":
            return SwiGLU(d, h)
        if arm == "geom":
            return GeomMix(d, h, "geom")
        if arm == "dot":
            return GeomMix(d, h, "dot")
        if arm == "wedge":
            return GeomMix(d, h, "wedge")
        if arm == "geom_g3":
            return GeomG3(d, h)
        raise ValueError(arm)

    class Attn(nn.Module):
        """Shared causal MHA (from model.py), identical across all arms."""
        def __init__(self, d, heads):
            super().__init__()
            self.h, self.dh = heads, d // heads
            self.qkv = nn.Linear(d, 3 * d, bias=False)
            self.proj = nn.Linear(d, d, bias=False)

        def forward(self, x):
            B, T, D = x.shape
            q, k, v = self.qkv(x).split(D, dim=2)
            q, k, v = (z.view(B, T, self.h, self.dh).transpose(1, 2) for z in (q, k, v))
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.proj(o.transpose(1, 2).reshape(B, T, D))

    class Block(nn.Module):
        def __init__(self, arm, d, heads, h):
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.ln2 = nn.LayerNorm(d)
            self.attn = Attn(d, heads)
            self.mix = make_mixer(arm, d, h)

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            return x + self.mix(self.ln2(x))

    class GeomLM(nn.Module):
        def __init__(self, arm, vocab, d, h, n_layers, heads=8, ctx=256):
            super().__init__()
            self.arm, self.ctx = arm, ctx
            self.tok = nn.Embedding(vocab, d)
            self.pos = nn.Embedding(ctx, d)
            self.blocks = nn.ModuleList([Block(arm, d, heads, h) for _ in range(n_layers)])
            self.ln_f = nn.LayerNorm(d)
            self.head = nn.Linear(d, vocab, bias=False)
            self.head.weight = self.tok.weight                 # tied (graphplay GPT)
            self.apply(self._init)

        def _init(self, m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        def forward(self, idx, targets=None):
            B, T = idx.shape
            pos = torch.arange(T, device=idx.device)
            x = self.tok(idx) + self.pos(pos)[None]
            for blk in self.blocks:
                x = blk(x)
            logits = self.head(self.ln_f(x))
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss

        def n_params(self, non_embed=True):
            n = sum(p.numel() for p in self.parameters())
            if non_embed:
                n -= self.tok.weight.numel() + self.pos.weight.numel()
            return n

        @torch.no_grad()
        def generate(self, idx, n, temp=0.8):
            for _ in range(n):
                logits, _ = self(idx[:, -self.ctx:])
                probs = F.softmax(logits[:, -1] / temp, dim=-1)
                idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
            return idx


# ----- iso-param sizing: pure-logic, no torch. Mirrors model.py size_for. ---------------
# Per-arm NON-EMBED param count as a closed form in (d, h, n_layers, heads). The LM head
# is tied to the token embedding, so it is NOT counted (it is part of the embedding tied
# weight). Each block contributes: 2 LayerNorms (2*d each, weight+bias) + attention
# (qkv 3*d*d, proj d*d = 4*d^2) + mixer. ln_f adds a final 2*d.

def _mixer_params(arm, d, h):
    """params in one mixer, given the rounding each mixer does to h."""
    if arm == "swiglu":
        return 3 * d * h                                  # w1,w2: d*h ; w3: h*d
    if arm in ("geom", "dot", "wedge"):
        if h % 2:
            h += 1
        return 3 * d * h                                  # u,w: d*h ; o: h*d
    if arm == "geom_g3":
        if h % 3:
            h += 3 - (h % 3)
        return 4 * d * h                                  # u,w,t: d*h ; o: h*d
    raise ValueError(arm)


def n_params_for(arm, d, h, n_layers, heads):
    per_block = 4 * d + 4 * d * d + _mixer_params(arm, d, h)   # 2 LN (4d) + attn (4d^2) + mixer
    return n_layers * per_block + 2 * d                        # + final ln_f


def _search_h(arm, target, d, n_layers, heads):
    """Binary-search mixer width h (>=2) for non-embed params <= target at fixed d.
    Returns (h, n_params) for the largest h not exceeding target (or h=2 floor)."""
    lo, hi, best = 2, 16384, (2, n_params_for(arm, d, 2, n_layers, heads))
    while lo <= hi:
        h = (lo + hi) // 2
        np_ = n_params_for(arm, d, h, n_layers, heads)
        if np_ <= target:
            best = (h, np_); lo = h + 1
        else:
            hi = h - 1
    return best


def size_for(arm, target, n_layers, heads=8, ratio=2.67, d_fixed=None):
    """Iso-param sizing. Two knobs share the budget: d_model and mixer width h.

    Two-stage so even small budgets land tight: (1) coarse-search d (a multiple of
    lcm(heads,6), so h splits cleanly into 2- and 3-blades) using h=round(ratio*d)
    as the natural transformer shape, picking the LARGEST d whose params still fit;
    (2) hold that d and fine-search h to top the budget back up to ~target. This
    makes every arm land within a few % of target AND of each other (the iso-param
    invariant the sweep relies on). If d_fixed is given, skip stage 1 and just
    fine-search h at that d (hold attention identical, vary only mixer capacity).

    Returns (d, h, n_params). Pure python — runs without torch for offline checks.
    """
    step = _lcm(heads, 6)                                 # d divisible by heads AND 6
    if d_fixed is not None:
        d = (d_fixed // step) * step or step
        h, np_ = _search_h(arm, target, d, n_layers, heads)
        return (d, h, np_)
    # stage 1: largest d (at the natural ratio) whose params fit the budget
    d = step
    while True:
        nd = d + step
        if n_params_for(arm, nd, round(ratio * nd), n_layers, heads) > target or nd > 4096:
            break
        d = nd
    # stage 2: top up h at that d
    h, np_ = _search_h(arm, target, d, n_layers, heads)
    return (d, h, np_)


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _lcm(a, b):
    return a * b // _gcd(a, b)


# ----- pure-logic self-test (no torch): exercise the param accounting & iso-param search --

def _selftest():
    import math
    # 1. closed-form param counts agree with the rounding rules
    assert _mixer_params("swiglu", 60, 160) == 3 * 60 * 160
    assert _mixer_params("geom", 60, 161) == 3 * 60 * 162           # h bumped to even
    assert _mixer_params("geom_g3", 60, 160) == 4 * 60 * 162        # h bumped to mult of 3
    # 2. size_for hits the target reasonably tightly for every arm, several budgets
    for target in (5e5, 1e6, 2e6, 4e6, 8e6):
        sizes = {}
        for arm in ("swiglu", "geom", "dot", "wedge", "geom_g3"):
            d, h, np_ = size_for(arm, target, n_layers=6, heads=8)
            sizes[arm] = np_
            assert d % 8 == 0 and d % 6 == 0, (arm, d)
            # within one search step of the target (binary search granularity)
            rel = abs(np_ - target) / target
            assert rel < 0.05, (arm, target, np_, rel)
        # 3. iso-param: all arms within 3% of each other at the same target
        lo, hi = min(sizes.values()), max(sizes.values())
        assert (hi - lo) / hi < 0.03, (target, sizes)
    # 4. d_fixed path: holds d (snapped to lcm(8,6)=24 grid, leaving room for the mixer),
    # varies h to land just under target
    d, h, np_ = size_for("geom", 2e6, n_layers=6, heads=8, d_fixed=192)
    assert d == 192 and h > 2 and np_ <= 2e6 and abs(np_ - 2e6) / 2e6 < 0.01, (d, h, np_)
    print("geom_lm self-test OK  (param accounting + iso-param search, no torch)")
    for target in (5e5, 1e6, 2e6, 4e6, 8e6):
        row = {arm: size_for(arm, target, 6, 8) for arm in
               ("swiglu", "geom", "dot", "wedge", "geom_g3")}
        line = "  ".join(f"{a}: d={v[0]} h={v[1]} p={v[2]/1e6:.3f}M" for a, v in row.items())
        print(f"  target {target/1e6:.1f}M -> {line}")


if __name__ == "__main__":
    _selftest()
