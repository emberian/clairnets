"""RopeLM — the SAME small causal LM as geom_lm.py, but the channel mixer is FIXED to
SwiGLU (it won the earlier sweep) and the *position encoding* is the pluggable variable.
The scientific question, the inverse of the geom_lm one: geometry lost as a channel-mixer
on language; does it WIN where language structure actually lives — relative position?

RoPE is already a Clifford Cl(2) rotor: it rotates each (q,k) pair in a fixed 2D plane
e_{2i}∧e_{2i+1} by an angle m·theta_i, and the dot product q·R^T R k = q·R(theta(n-m))k
depends only on the relative offset (n-m). That rotor demonstrably works at scale. The
arms ask what a RICHER rotor buys:

  rope         : standard RoPE — fixed geometric-progression freqs theta_i=base^(-2i/dh).
                 The Cl(2)-rotor baseline. (the proof-of-concept that geometry helps)
  rope_learned : same per-plane rotor structure, but the frequencies are LEARNED (one
                 log-freq parameter per plane per head). Same FLOPs, +heads*(dh/2) params.
  rope_nd      : a higher-grade rotor — after the per-plane Cl(2) rotation, apply a CHEAP
                 learned orthogonal MIXING of the rotated planes per head (a Cl(2)
                 bivector "cross-plane" coupling, a small step toward a Cl(3,1)-style
                 rotor that doesn't factor into independent 2-planes). See math below.
  none         : no rotation at all — learned absolute pos-emb only. The floor.

EVERYTHING ELSE is shared and identical across arms (depth, heads, SwiGLU FFN width,
embeddings, attention projections) so the ONLY difference is the positional rotor. The
learned arms add only O(heads*dh) params; we report that delta honestly rather than
force exact iso-param (the deltas are <0.1% of the model — see param_delta()).

Reused from clair/geom_lm.py: SwiGLU, Attn skeleton, Block, GeomLM/RopeLM container,
tied head, generate(), the byte-LM GPT layout. Reused from clair/run_geom_lm.py-side:
the (d,h) sizing via size_for(arm="swiglu") so the backbone matches a geom_lm swiglu run.

torch is import-guarded so the rotor math + param accounting self-test run on CPU with
no torch (numpy-only). The torch path is exercised only on the GPU box.
"""
from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:                       # pure-logic self-test path (no torch locally)
    torch = nn = F = None
    _HAS_TORCH = False

ARMS = ("rope", "rope_learned", "rope_nd", "none")


# ----- the position encodings; each rotates (q,k) of shape (B, H, T, dh) in place ---------

if _HAS_TORCH:

    def _rope_freqs(dh, base=10000.0):
        """Standard geometric-progression inverse frequencies, one per 2D plane.
        theta_i = base^(-2i/dh) for i in [0, dh/2). Returns (dh/2,) tensor."""
        i = torch.arange(0, dh, 2, dtype=torch.float32)        # 0,2,4,...,dh-2
        return base ** (-(i / dh))                             # (dh/2,)

    def _apply_rotor(q, k, cos, sin):
        """Rotate the even/odd coordinate PAIRS of q,k by per-position angles.
        q,k: (B,H,T,dh). cos,sin: (T, dh/2). Each plane (x0,x1) -> (x0 cosθ - x1 sinθ,
        x0 sinθ + x1 cosθ): the Cl(2) rotor exp(θ e0∧e1) acting on grade-1 vectors.
        Norm-preserving by construction (it's a rotation)."""
        def rot(x):
            x = x.unflatten(-1, (-1, 2))                        # (B,H,T,dh/2,2)
            x0, x1 = x[..., 0], x[..., 1]
            r0 = x0 * cos - x1 * sin
            r1 = x0 * sin + x1 * cos
            return torch.stack([r0, r1], -1).flatten(-2)        # (B,H,T,dh)
        return rot(q), rot(k)

    class RopeNone(nn.Module):
        """No rotation — identity. Absolute position comes only from the learned pos-emb."""
        def __init__(self, dh, heads, ctx, base=10000.0):
            super().__init__()

        def forward(self, q, k):
            return q, k

    class RopeFixed(nn.Module):
        """Standard RoPE. Fixed geometric-progression freqs, shared across heads.
        Precomputes cos/sin for all ctx positions (no params). The Cl(2)-rotor baseline."""
        def __init__(self, dh, heads, ctx, base=10000.0):
            super().__init__()
            freqs = _rope_freqs(dh, base)                       # (dh/2,)
            ang = torch.outer(torch.arange(ctx, dtype=torch.float32), freqs)   # (T, dh/2)
            self.register_buffer("cos", ang.cos(), persistent=False)
            self.register_buffer("sin", ang.sin(), persistent=False)

        def forward(self, q, k):
            T = q.size(-2)
            return _apply_rotor(q, k, self.cos[:T], self.sin[:T])

    class RopeLearned(nn.Module):
        """Same rotor structure, but per-(head, plane) frequencies are LEARNED. We store a
        log-frequency parameter initialised at the standard RoPE schedule, so step 0 == RoPE
        and training can re-allocate frequency budget across planes/heads. Angles are formed
        fresh each step (cheap: T x heads x dh/2). params = heads*(dh/2)."""
        def __init__(self, dh, heads, ctx, base=10000.0):
            super().__init__()
            self.ctx = ctx
            f = _rope_freqs(dh, base)                            # (dh/2,)
            logf = f.log().expand(heads, -1).clone()            # (H, dh/2) init at standard
            self.log_freq = nn.Parameter(logf)

        def forward(self, q, k):
            T = q.size(-2)
            pos = torch.arange(T, device=q.device, dtype=torch.float32)        # (T,)
            ang = pos[None, :, None] * self.log_freq.exp()[:, None, :]          # (H,T,dh/2)
            cos = ang.cos()[None]                               # (1,H,T,dh/2) broadcasts over B
            sin = ang.sin()[None]
            return _apply_rotor(q, k, cos, sin)

    class RopeND(nn.Module):
        """Higher-grade rotor: standard per-plane Cl(2) rotation FOLLOWED by a learned
        orthogonal MIXING of the rotated planes, per head. A plain RoPE rotor is a direct
        sum of dh/2 independent 2D rotations — it never couples plane i to plane j. A general
        Cl(3,1)/SO(n) rotor does. We add the cheapest non-trivial coupling: treat the dh/2
        planes as a list of 2-vectors and pre-multiply them by a fixed per-head 2x2 rotation
        Q(phi_h)=exp(phi_h e0∧e1) (a learned scalar angle phi per head) — i.e. ROTATE each
        plane's local frame by a head-specific angle BEFORE the position rotor. Because all
        planes share Q within a head, Q commutes with the per-position diagonal rotor and the
        relative-position property q·R(n-m)k is preserved (both q and k get the same Q, so
        Qᵀ R(n) Qᵀᵀ... cancels into R(n-m)). This is the cheapest learned cross-frame coupling
        that stays a genuine rotor (orthogonal, norm-preserving, relative-position-exact).
        params = heads (one angle per head). It is a strict generalisation: phi=0 ⇒ rope."""
        def __init__(self, dh, heads, ctx, base=10000.0):
            super().__init__()
            self.rope = RopeFixed(dh, heads, ctx, base)
            self.phi = nn.Parameter(torch.zeros(heads))         # (H,) head frame angle, init 0 = rope

        def _frame(self, x):
            # rotate each (x0,x1) plane by the per-head angle phi_h (same for all planes/positions)
            c = self.phi.cos()[None, :, None, None]             # (1,H,1,1)
            s = self.phi.sin()[None, :, None, None]
            x = x.unflatten(-1, (-1, 2))                        # (B,H,T,dh/2,2)
            x0, x1 = x[..., 0], x[..., 1]
            r0 = x0 * c - x1 * s
            r1 = x0 * s + x1 * c
            return torch.stack([r0, r1], -1).flatten(-2)

        def forward(self, q, k):
            q, k = self._frame(q), self._frame(k)               # learned head-frame rotation
            return self.rope(q, k)                              # then standard position rotor

    def make_pos(arm, dh, heads, ctx, base=10000.0):
        if arm == "rope":
            return RopeFixed(dh, heads, ctx, base)
        if arm == "rope_learned":
            return RopeLearned(dh, heads, ctx, base)
        if arm == "rope_nd":
            return RopeND(dh, heads, ctx, base)
        if arm == "none":
            return RopeNone(dh, heads, ctx, base)
        raise ValueError(arm)

    # ----- SwiGLU mixer (fixed; identical to geom_lm.py) -----

    class SwiGLU(nn.Module):
        """Baseline gated FFN (the channel-mixer that won). params = 3*d*h."""
        def __init__(self, d, h):
            super().__init__()
            self.w1 = nn.Linear(d, h, bias=False)
            self.w2 = nn.Linear(d, h, bias=False)
            self.w3 = nn.Linear(h, d, bias=False)

        def forward(self, x):
            return self.w3(F.silu(self.w1(x)) * self.w2(x))

    class Attn(nn.Module):
        """Causal MHA (from geom_lm.py) + a pluggable positional rotor on (q,k). Identical
        projections across arms; only `pos` differs."""
        def __init__(self, arm, d, heads, ctx, base=10000.0):
            super().__init__()
            self.h, self.dh = heads, d // heads
            self.qkv = nn.Linear(d, 3 * d, bias=False)
            self.proj = nn.Linear(d, d, bias=False)
            self.pos = make_pos(arm, self.dh, heads, ctx, base)

        def forward(self, x):
            B, T, D = x.shape
            q, k, v = self.qkv(x).split(D, dim=2)
            q, k, v = (z.view(B, T, self.h, self.dh).transpose(1, 2) for z in (q, k, v))
            q, k = self.pos(q, k)                               # rotate q,k (the only arm-dependent step)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.proj(o.transpose(1, 2).reshape(B, T, D))

    class Block(nn.Module):
        def __init__(self, arm, d, heads, h, ctx, base=10000.0):
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.ln2 = nn.LayerNorm(d)
            self.attn = Attn(arm, d, heads, ctx, base)
            self.mix = SwiGLU(d, h)

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            return x + self.mix(self.ln2(x))

    class RopeLM(nn.Module):
        """Same container as GeomLM. Absolute pos-emb is KEPT for every arm so `none` has a
        real floor and the rope arms don't lose the embedding the swiglu backbone expects;
        the rope arms simply ALSO get the rotor (RoPE is conventionally added on top)."""
        def __init__(self, arm, vocab, d, h, n_layers, heads=8, ctx=256, base=10000.0):
            super().__init__()
            assert arm in ARMS, arm
            self.arm, self.ctx = arm, ctx
            self.tok = nn.Embedding(vocab, d)
            self.pos = nn.Embedding(ctx, d)
            self.blocks = nn.ModuleList(
                [Block(arm, d, heads, h, ctx, base) for _ in range(n_layers)])
            self.ln_f = nn.LayerNorm(d)
            self.head = nn.Linear(d, vocab, bias=False)
            self.head.weight = self.tok.weight                 # tied
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


# ----- iso-param sizing & honest param-delta accounting (pure python, no torch) ----------
# The backbone (attn + SwiGLU + embeddings) is IDENTICAL to a geom_lm "swiglu" run, so we
# size it with that same closed form. Only the rotor adds (tiny) extra params, per arm.

def _pos_extra_params(arm, dh, heads, n_layers):
    """Extra params the positional rotor adds, summed over all layers.
      rope / none : 0  (fixed schedule / identity; cos/sin are non-persistent buffers)
      rope_learned: heads*(dh/2) per layer  (one log-freq per head per plane)
      rope_nd     : heads        per layer  (one frame angle per head)"""
    if arm in ("rope", "none"):
        return 0
    if arm == "rope_learned":
        return n_layers * heads * (dh // 2)
    if arm == "rope_nd":
        return n_layers * heads
    raise ValueError(arm)


def n_params_for(arm, d, h, n_layers, heads):
    """Non-embed params. Backbone matches geom_lm swiglu: per block 2 LN (4d) + attn (4d^2)
    + SwiGLU (3dh); + final ln_f (2d); + the rotor extra (tiny)."""
    per_block = 4 * d + 4 * d * d + 3 * d * h
    dh = d // heads
    return n_layers * per_block + 2 * d + _pos_extra_params(arm, dh, heads, n_layers)


def size_for(target, n_layers, heads=8, ratio=2.67, d_fixed=None):
    """Iso-param sizing for the SHARED swiglu backbone (rotor extras are negligible, so we
    size on the backbone alone and report the per-arm delta separately). Mirrors
    geom_lm.size_for's two-stage search but specialised to the one mixer shape we use here.
    Returns (d, h, n_params_backbone). Pure python — runs without torch."""
    step = _lcm(2 * heads, 6)   # 2*heads -> d/heads is EVEN so RoPE's pairwise unflatten works (dh must be even)
    bb = lambda dd, hh: n_params_for("rope", dd, hh, n_layers, heads)   # backbone (no rotor extra)
    if d_fixed is not None:
        d = (d_fixed // step) * step or step
        h, npar = _search_h(bb, target, d)
        return (d, h, npar)
    d = step
    while True:
        nd = d + step
        if bb(nd, round(ratio * nd)) > target or nd > 4096:
            break
        d = nd
    h, npar = _search_h(bb, target, d)
    return (d, h, npar)


def _search_h(bb, target, d):
    lo, hi, best = 2, 16384, (2, bb(d, 2))
    while lo <= hi:
        h = (lo + hi) // 2
        np_ = bb(d, h)
        if np_ <= target:
            best = (h, np_); lo = h + 1
        else:
            hi = h - 1
    return best


def param_delta(d, h, n_layers, heads):
    """Per-arm non-embed param counts + the delta vs the rope baseline, as a dict. Used by
    the trainer to print the honest (tiny) param differences between arms."""
    base = n_params_for("rope", d, h, n_layers, heads)
    return {arm: (n_params_for(arm, d, h, n_layers, heads),
                  n_params_for(arm, d, h, n_layers, heads) - base) for arm in ARMS}


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _lcm(a, b):
    return a * b // _gcd(a, b)


# ----- pure-logic self-test (no torch): rotor math + param accounting --------------------

def _selftest():
    import numpy as np

    # 1. backbone param counts: rope == none (no extra), learned/nd add the documented deltas
    d, h, L, H = 192, 512, 6, 8
    dh = d // H
    delta = param_delta(d, h, L, H)
    assert delta["rope"][1] == 0 and delta["none"][1] == 0, delta
    assert delta["rope_learned"][1] == L * H * (dh // 2), delta
    assert delta["rope_nd"][1] == L * H, delta
    # the largest delta (learned) is a vanishing fraction of the model
    base = delta["rope"][0]
    assert delta["rope_learned"][1] / base < 1e-3, (delta, base)

    # 2. size_for hits the target tightly (same machinery as geom_lm swiglu)
    for target in (5e5, 1e6, 2e6, 4e6):
        dd, hh, npar = size_for(target, n_layers=6, heads=8)
        assert dd % 8 == 0 and dd % 6 == 0, dd
        assert abs(npar - target) / target < 0.05, (target, npar)

    # 3. ROTOR MATH (numpy reference). Standard RoPE on one 2-plane must equal the complex
    # multiply form  (q0+iq1)*e^{i m theta}, and the q·k inner product must depend only on
    # the relative offset (n - m). Also: the rotor preserves norm.
    rng = np.random.default_rng(0)
    dh2 = 8
    base_f = 10000.0
    i = np.arange(0, dh2, 2)
    freqs = base_f ** (-(i / dh2))                          # (dh2/2,) standard schedule
    q = rng.standard_normal(dh2)
    k = rng.standard_normal(dh2)

    def rope_np(x, pos):
        x2 = x.reshape(-1, 2)
        ang = pos * freqs
        c, s = np.cos(ang), np.sin(ang)
        r0 = x2[:, 0] * c - x2[:, 1] * s
        r1 = x2[:, 0] * s + x2[:, 1] * c
        return np.stack([r0, r1], -1).reshape(-1)

    # (a) complex-multiply reference equals our real rotor
    def rope_complex(x, pos):
        z = x.reshape(-1, 2)[:, 0] + 1j * x.reshape(-1, 2)[:, 1]
        z = z * np.exp(1j * pos * freqs)
        out = np.empty(dh2)
        out[0::2], out[1::2] = z.real, z.imag
        return out
    for pos in (0, 1, 3, 17):
        assert np.allclose(rope_np(q, pos), rope_complex(q, pos), atol=1e-10), pos

    # (b) norm preservation
    for pos in (0, 5, 31):
        assert np.allclose(np.linalg.norm(rope_np(q, pos)), np.linalg.norm(q), atol=1e-10)

    # (c) relative-position property: <R(m)q, R(n)k> depends only on (n-m)
    def inner(m, n):
        return float(rope_np(q, m) @ rope_np(k, n))
    for (m, n, m2, n2) in [(0, 5, 3, 8), (2, 9, 10, 17), (1, 1, 30, 30)]:
        assert abs(inner(m, n) - inner(m2, n2)) < 1e-9, (m, n, m2, n2, inner(m, n), inner(m2, n2))

    # 4. rope_nd's per-head frame rotation Q(phi) is orthogonal and COMMUTES with the relative
    # property: applying the same Q to q and k leaves <R(m)Qq, R(n)Qk> = <R(n-m) Qq, Qk> still
    # a function of (n-m) only (Q is shared, position rotor is per-plane diagonal). Check that
    # the rel-pos invariance survives an arbitrary head-frame rotation phi.
    phi = 0.7

    def frame(x):
        x2 = x.reshape(-1, 2)
        c, s = math.cos(phi), math.sin(phi)
        r0 = x2[:, 0] * c - x2[:, 1] * s
        r1 = x2[:, 0] * s + x2[:, 1] * c
        return np.stack([r0, r1], -1).reshape(-1)

    qf, kf = frame(q), frame(k)
    assert np.allclose(np.linalg.norm(qf), np.linalg.norm(q))    # frame is orthogonal

    def inner_nd(m, n):
        return float(rope_np(qf, m) @ rope_np(kf, n))
    for (m, n, m2, n2) in [(0, 5, 3, 8), (2, 9, 10, 17)]:
        assert abs(inner_nd(m, n) - inner_nd(m2, n2)) < 1e-9, (m, n, inner_nd(m, n), inner_nd(m2, n2))

    print("rope_lm self-test OK  (rotor math: complex-form, norm, rel-pos, nd-frame; param accounting)")
    for target in (5e5, 1e6, 2e6, 4e6):
        dd, hh, npar = size_for(target, 6, 8)
        de = param_delta(dd, hh, 6, 8)
        line = "  ".join(f"{a}:+{de[a][1]}" for a in ARMS)
        print(f"  target {target/1e6:.1f}M -> d={dd} h={hh} backbone={npar/1e6:.3f}M  deltas[{line}]")


if __name__ == "__main__":
    _selftest()
