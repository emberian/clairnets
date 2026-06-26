"""clair/blade_deductor.py — a GRADE-STRUCTURED factor deductor.

Same factor-graph narrowing skeleton as proposer.FactorGraphProposer (bipartite
variable<->factor message passing, monotone meet on per-cell candidate sets, dominate-dedP
training), BUT the factor->variable message for an arity-k factor is a GRADE-k geometric-product
/ blade over the participating cells' candidate-state vectors:

    arity 1 (pin)              -> grade-0  SCALAR        single-cell readout  (no interaction)
    arity 2 (= / != binary)    -> grade-2  BIVECTOR  u∧v antisymmetric pair   (cross product, Cl(3))
    arity 3 (sum / xor / affine)-> grade-3 TRIVECTOR u∧v∧w  scalar triple product / det (Cl(3))

The bet: an arity-k factor's cell<->cell coupling has the algebraic structure of a grade-k blade
(the antisymmetric multilinear form on k vectors). Giving the message exactly that blade — the
trivector u∧v∧w for ternary constraints — is the inductive bias the generic relation-table MLP
lacks, and is hypothesised to (a) raise COMPLETENESS on affine/3-way constraints and (b)
GENERALISE to UNSEEN arity-3 relations (the grade-3 path exists even if no arity-3 factor was
seen in training), where the table version, having never exercised its arity-3 slot, collapses.

Blade primitives are exactly geom_lm.GeomG3 / cliffordnet's wedge: split d into 3-vectors, then
  inner  a·b                                            grade 0  (symmetric)
  cross  a∧b  (3 comps)                                  grade 2  (antisymmetric, vanishes a∥b)
  det    a∧b∧c = a·(b×c)                                 grade 3  (fully antisym, vanishes if dep)

The relation table STILL enters via the factor encoding (`fac_in`), identical to the table
baseline — soundness machinery is unchanged; the blade only adds the arity-structured cell
interaction. `msg in {"blade","table"}` is a config flag so the SAME run harness drives both for a
clean iso-param comparison. The per-variable channel mixer is a SwiGLU in BOTH modes (held fixed),
so the factor->variable message form is the ONLY independent variable.

torch is import-guarded so the numpy blade-validation self-test (`python -m clair.blade_deductor`)
runs on CPU with no torch; the torch model path is exercised only on the GPU box.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:                       # numpy-only blade-validation path (no torch locally)
    torch = nn = F = None
    _HAS_TORCH = False


# --------------------------------------------------------------------------- blade primitives (torch)
if _HAS_TORCH:

    def cross3(a, b):
        """grade-2 bivector a∧b in Cl(3) == cross product (3 comps). a,b: (..., 3).
        Antisymmetric: cross3(a,b) = -cross3(b,a); cross3(a,a)=0 (no self-energy)."""
        a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
        b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
        return torch.stack([a1 * b2 - a2 * b1,
                            a2 * b0 - a0 * b2,
                            a0 * b1 - a1 * b0], dim=-1)

    def det3(a, b, c):
        """grade-3 trivector a∧b∧c in Cl(3): the pseudoscalar coeff = scalar triple product
        a·(b×c) = det([a;b;c]). Fully antisymmetric; vanishes iff a,b,c linearly dependent
        (in particular if any two of the three cells carry equal blade vectors). (..., 3)->(...)."""
        return (a * cross3(b, c)).sum(-1)

    class SwiGLU(nn.Module):
        """ffn per-variable mixer (held identical across blade/table modes)."""
        def __init__(self, d, ratio=2.67):
            super().__init__()
            h = int(d * ratio)
            self.w1 = nn.Linear(d, h, bias=False)
            self.w2 = nn.Linear(d, h, bias=False)
            self.w3 = nn.Linear(h, d, bias=False)

        def forward(self, x):
            return self.w3(F.silu(self.w1(x)) * self.w2(x))

    # --------------------------------------------------------------------- the grade-structured deductor
    class BladeFactorDeductor(nn.Module):
        """Bipartite variable<->factor message passing; factor->variable message = grade-k blade
        (msg='blade') or the generic relation-table MLP broadcast (msg='table', the baseline).

        Permutation-equivariant over cells (no absolute cell position): the factor graph + scope
        slot embedding are the only asymmetry, and the blade uses ONE shared vector-field map across
        slots, so the same weights run on any (n, d, cons) up to padding."""
        def __init__(self, msg, n_max, d_max, m_max, a_max, d=66, R=8, ds=4):
            super().__init__()
            assert msg in ("blade", "table")
            assert d % 3 == 0, "d must split into 3-vectors for the Cl(3) blades (div by 3)"
            assert a_max <= 3, "grade-k blades implemented up to the trivector (Cl(3))"
            self.msg = msg
            self.n_max, self.d_max, self.m_max, self.a_max = n_max, d_max, m_max, a_max
            self.R, self.ds = R, ds
            rel_dim = d_max ** a_max
            self.var_in = nn.Linear(d_max + 1, d)             # alive-mask over d values + 'decided' flag
            self.fac_in = nn.Linear(rel_dim + a_max, d)       # broadcast relation table + arity one-hot
            self.pos = nn.Embedding(a_max, d)                 # scope-position embedding (which slot)
            self.vf = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))   # var -> factor message
            self.fv = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))   # factor -> var message
            self.fac_ln = nn.LayerNorm(d)
            self.var_ln = nn.LayerNorm(d)
            self.mix_ln = nn.LayerNorm(d)
            self.mixer = SwiGLU(d)                            # per-variable mixer (identical both modes)
            self.b_head = nn.Linear(d, d_max)                 # per-cell per-value survival logit
            self.cls_head = nn.Linear(d, 1)                   # pooled conflict (unsat) logit
            self.d, self.G = d, d // 3
            if msg == "blade":                                # grade-k blade message submodules
                self.vec = nn.Linear(d, d, bias=False)        # cell state -> Cl(3) 3-vector field (G groups)
                self.lin0 = nn.Linear(d, d, bias=False)       # grade-0 readout (single-cell / pin)
                self.lin2 = nn.Linear(d, d, bias=False)       # grade-2 bivector (G*3=d) readout
                self.lin3 = nn.Linear(self.G, d, bias=False)  # grade-3 trivector (G scalars) readout

        def _blade(self, gathered, B, a1, a2, a3):
            """Grade-k blade over the scope cells' gathered states. gathered [B,m,a,d].
            Returns a per-FACTOR blade feature [B,m,d], grade-selected by the arity one-hots.
              arity 1: grade-0 scalar  (readout of slot-0 cell)
              arity 2: grade-2 cross(v0,v1)   (the antisymmetric pair / exclusion bivector)
              arity 3: grade-3 det(v0,v1,v2)  (the 3-way antisymmetric trivector / affine coupling)"""
            m, G = self.m_max, self.G
            V = self.vec(gathered).reshape(B, m, self.a_max, G, 3)         # 3-vector field per cell
            v0 = V[:, :, 0]                                                # [B,m,G,3]
            v1 = V[:, :, 1] if self.a_max > 1 else torch.zeros_like(v0)
            v2 = V[:, :, 2] if self.a_max > 2 else torch.zeros_like(v0)
            m0 = self.lin0(v0.reshape(B, m, self.d))                      # grade 0
            m2 = self.lin2(cross3(v0, v1).reshape(B, m, self.d))          # grade 2 (bivector)
            m3 = self.lin3(det3(v0, v1, v2))                              # grade 3 (trivector, [B,m,G]->d)
            return a1 * m0 + a2 * m2 + a3 * m3

        def forward(self, var_mask, given, fac_rel, fac_arity, edge_var, edge_valid, var_valid, fac_valid):
            """Signature-compatible with proposer.FactorGraphProposer (same run harness). Shapes:
              var_mask [B,n,dv]  given [B,n]  var_valid [B,n]
              fac_rel [B,m,dv**a]  fac_arity [B,m,a] one-hot  fac_valid [B,m]
              edge_var [B,m,a] long (pad slots = n_max)  edge_valid [B,m,a] in {0,1}
            Returns (b_final [B,n,dv], cls_final [B], sup=list of (b,cls))."""
            B, n, dv = var_mask.shape
            a, D = self.a_max, self.d
            hv = self.var_in(torch.cat([var_mask, given.unsqueeze(-1)], dim=-1))   # [B,n,D]
            hf = self.fac_in(torch.cat([fac_rel, fac_arity], dim=-1))              # [B,m,D]
            pos = self.pos(torch.arange(a, device=hv.device))                     # [a,D]
            idx = edge_var.reshape(B, -1, 1).expand(-1, -1, D)                    # [B,m*a,D] gather/scatter
            ev = edge_valid.unsqueeze(-1)                                          # [B,m,a,1]
            fvld = fac_valid.unsqueeze(-1)                                         # [B,m,1]
            a1 = fac_arity[..., 0:1]                                               # arity one-hots [B,m,1]
            a2 = fac_arity[..., 1:2]
            a3 = fac_arity[..., 2:3]
            sup = []
            for r in range(self.R):
                # ---- variable -> factor ----
                hv_pad = torch.cat([hv, torch.zeros(B, 1, D, device=hv.device, dtype=hv.dtype)], dim=1)
                gathered = torch.gather(hv_pad, 1, idx).reshape(B, self.m_max, a, D)   # [B,m,a,D]
                msg_vf = self.vf(gathered + pos) * ev
                hf = self.fac_ln(hf + msg_vf.sum(2))
                # ---- factor -> variable (the ABLATABLE message: blade vs table) ----
                base = hf.unsqueeze(2).expand(-1, -1, a, -1) + pos                # relation encoding + slot
                if self.msg == "blade":
                    blade = self._blade(gathered, B, a1, a2, a3)                  # grade-k cell coupling
                    base = base + blade.unsqueeze(2)
                msg_fv = self.fv(base) * ev * fvld.unsqueeze(2)
                tgt = torch.zeros(B, self.n_max + 1, D, device=hv.device, dtype=hv.dtype)
                tgt.scatter_add_(1, idx, msg_fv.reshape(B, -1, D))
                hv = self.var_ln(hv + tgt[:, :self.n_max, :])
                # ---- per-variable channel mixer (held fixed across modes) ----
                hv = hv + self.mixer(self.mix_ln(hv))
                if r >= self.R - self.ds:
                    b = self.b_head(hv)
                    denom = var_valid.sum(1, keepdim=True).clamp(min=1.0)
                    pooled = (hv * var_valid.unsqueeze(-1)).sum(1) / denom        # masked mean over cells
                    cls = self.cls_head(pooled).squeeze(-1)
                    sup.append((b, cls))
            b, cls = sup[-1]
            return b, cls, sup

        def n_params(self, non_embed=True):
            n = sum(p.numel() for p in self.parameters())
            if non_embed:
                n -= self.pos.weight.numel()
            return n


def size_for(msg, n_max, d_max, m_max, a_max, target, R=8, ds=4):
    """Binary-search d_model (multiple of 3, for the Cl(3) split) for the LARGEST d whose non-embed
    params are <= target. Returns (d, n_params). Because the blade mode carries extra submodules,
    sizing both modes to the SAME param budget (rather than the same d) keeps the comparison
    iso-param; the run sizes blade to the table arm's param count so blade gets NO param advantage."""
    if not _HAS_TORCH:
        raise RuntimeError("size_for needs torch (sizes the model by instantiating it)")
    def np_(d):
        return BladeFactorDeductor(msg, n_max, d_max, m_max, a_max, d=d, R=R, ds=ds).n_params()
    lo, hi, best = 3, 1536, (3, np_(3))
    while lo <= hi:
        d = max(3, ((lo + hi) // 2 // 3) * 3)
        n = np_(d)
        if n <= target:
            best = (d, n); lo = d + 3
        else:
            hi = d - 3
    return best


# --------------------------------------------------------------------------- numpy blade validation
def _selftest():
    """NUMPY validation of the blade ops BEFORE any training: wedge antisymmetry / self-energy,
    trivector full antisymmetry, vanishing on linearly-dependent inputs, and the scalar-triple-product
    identity. If torch is present, also check the grade SELECTION in BladeFactorDeductor._blade."""
    import numpy as np
    rng = np.random.default_rng(0)

    def ncross(a, b):                                          # numpy mirror of cross3 (grade-2)
        return np.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                         a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                         a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], axis=-1)

    def ndet(a, b, c):                                         # numpy mirror of det3 (grade-3)
        return (a * ncross(b, c)).sum(-1)

    A = rng.standard_normal((7, 3)); B = rng.standard_normal((7, 3)); Cc = rng.standard_normal((7, 3))

    # 1. grade-2 bivector: antisymmetry + zero self-energy ---------------------------------------
    assert np.allclose(ncross(A, B), -ncross(B, A)), "bivector u∧v not antisymmetric"
    assert np.allclose(ncross(A, A), 0.0), "self-wedge u∧u must vanish (no self-energy)"
    # cross product is orthogonal to both inputs (it IS the bivector's dual axis)
    assert np.allclose((ncross(A, B) * A).sum(-1), 0.0) and np.allclose((ncross(A, B) * B).sum(-1), 0.0)

    # 2. grade-3 trivector: full antisymmetry under ANY transposition ----------------------------
    assert np.allclose(ndet(A, B, Cc), -ndet(B, A, Cc)), "trivector not antisym under (0 1) swap"
    assert np.allclose(ndet(A, B, Cc), -ndet(A, Cc, B)), "trivector not antisym under (1 2) swap"
    assert np.allclose(ndet(A, B, Cc), ndet(B, Cc, A)), "trivector not invariant under cyclic rotation"

    # 3. trivector VANISHES on linearly-dependent inputs (the grade-3 blade is the volume form) ---
    assert np.allclose(ndet(A, A, Cc), 0.0), "trivector must vanish when two inputs equal"
    assert np.allclose(ndet(A, B, A), 0.0), "trivector must vanish when two inputs equal (0,2)"
    dep = 0.4 * A - 1.7 * B                                    # c in span(a,b)
    assert np.allclose(ndet(A, B, dep), 0.0), "trivector must vanish when c ∈ span(a,b)"
    rndvol = ndet(A, B, Cc)
    assert np.abs(rndvol).max() > 1e-3, "generic (independent) triples give nonzero volume"

    # 4. scalar-triple-product identity det([a;b;c]) == a·(b×c) (and == numpy's own det) ----------
    M = np.stack([A, B, Cc], axis=-2)                          # (7,3,3) rows a,b,c
    assert np.allclose(ndet(A, B, Cc), np.linalg.det(M)), "det3 != linalg.det of the row matrix"
    assert np.allclose(ndet(A, B, Cc), (A * ncross(B, Cc)).sum(-1)), "det3 != scalar triple product"

    print("blade ops validated (numpy): grade-2 wedge antisym+zero-self-energy; "
          "grade-3 trivector full-antisym, vanishes on dependent inputs, == scalar triple product")

    if _HAS_TORCH:
        torch.manual_seed(0)
        n_max, d_max, m_max, a_max = 8, 8, 28, 3
        # torch cross3/det3 agree with numpy
        ta, tb, tc = (torch.tensor(x) for x in (A, B, Cc))
        assert torch.allclose(cross3(ta, tb), torch.tensor(ncross(A, B)))
        assert torch.allclose(det3(ta, tb, tc), torch.tensor(ndet(A, B, Cc)))
        # grade SELECTION: a pure arity-2 factor uses ONLY the bivector path (det path masked off),
        # a pure arity-3 factor activates the trivector path. Probe _blade directly.
        d = 66
        net = BladeFactorDeductor("blade", n_max, d_max, m_max, a_max, d=d, R=4, ds=2)
        net.eval()
        B0, m = 1, m_max
        gathered = torch.randn(B0, m, a_max, d)
        for ar, name in [(1, "pin"), (2, "binary"), (3, "ternary")]:
            oh = torch.zeros(B0, m, 1)                          # arity one-hot slices
            a1 = torch.full((B0, m, 1), 1.0 if ar == 1 else 0.0)
            a2 = torch.full((B0, m, 1), 1.0 if ar == 2 else 0.0)
            a3 = torch.full((B0, m, 1), 1.0 if ar == 3 else 0.0)
            out = net._blade(gathered, B0, a1, a2, a3)
            assert out.shape == (B0, m, d), (name, out.shape)
        # arity-3 message genuinely DEPENDS on the third cell (the trivector reads all three) ...
        a1 = torch.zeros(B0, m, 1); a2 = torch.zeros(B0, m, 1); a3 = torch.ones(B0, m, 1)
        g2 = gathered.clone(); g2[:, :, 2] += 1.3              # perturb slot-2 cell only
        o_a = net._blade(gathered, B0, a1, a2, a3)
        o_b = net._blade(g2, B0, a1, a2, a3)
        assert not torch.allclose(o_a, o_b), "arity-3 trivector ignores the third cell"
        # ... whereas the arity-2 message is INVARIANT to the (unused) third cell
        a2 = torch.ones(B0, m, 1); a3 = torch.zeros(B0, m, 1)
        o2_a = net._blade(gathered, B0, torch.zeros(B0, m, 1), a2, a3)
        o2_b = net._blade(g2, B0, torch.zeros(B0, m, 1), a2, a3)
        assert torch.allclose(o2_a, o2_b), "arity-2 bivector should ignore the third cell"
        # iso-param sizing: largest d under budget; size BLADE to the TABLE arm's param count so
        # blade never has MORE params than table (a conservative test of the inductive bias).
        st = size_for("table", n_max, d_max, m_max, a_max, 2.0e5)
        sb = size_for("blade", n_max, d_max, m_max, a_max, st[1])      # blade budget = table params
        assert st[1] <= 2.0e5 and sb[1] <= st[1], (sb, st)
        assert (st[1] - sb[1]) / st[1] < 0.06, ("blade/table param gap too large", sb, st)
        print(f"torch blade model OK: grade selection (pin/binary/ternary) routes correctly; "
              f"arity-3 reads 3 cells, arity-2 reads 2; iso-param: "
              f"table d={st[0]} p={st[1]:,}  >=  blade d={sb[0]} p={sb[1]:,} (blade no param advantage)")
    else:
        print("(torch not present — skipped the model-path grade-selection check; run on the GPU box)")


if __name__ == "__main__":
    _selftest()
