"""The PURE-LATENT deductive organ — "design B" (the bitter-lesson deductor).

NO explicit symbolic program. NO extraction supervision. The organ reads only OLMo's dense hidden of
the problem TEXT and is grounded solely on its OUTPUT (its decoded per-cell candidate sets must
dominate the exact transformer dedₚ). The decisive question: can a latent organ, fed only OLMo's
dense representation of the problem text — with NO explicit factors — learn to narrow correctly?

Two pieces:

  α  (DenseLatentProjector) — read OLMo mid-layer hidden states (per cell via mention-token pooling
     for cell IDENTITY + a per-cell cross-attention READ over the whole prompt for that cell's
     latent context) and project DENSELY into (a) an initial per-cell candidate-logit tensor [n,K]
     and (b) a latent relational/context tensor [n,dctx]. A learned MLP. NO loss on it directly; the
     constraints reach the organ ONLY through this projection — it must learn to expose them.

  organ (LatentNarrower) — an LDT-faithful recurrent narrower: a small masked-attention stack over
     the n per-cell representations, weight-TIED and unrolled for T iterations, RE-INJECTING the
     lattice encoding (a re-projection of the current candidate logits) + the latent context as a
     residual each step ("passes through the lattice itself"). Narrowing is MONOTONE: each step emits
     a non-negative elimination pressure that is SUBTRACTED from the candidate logits, so survival
     probability only decreases. Every late iteration is deeply supervised.

Grounding = OUTPUT supervision only: the dominate-dedₚ loss (soundness-asymmetric BCE, the SAME loss
the explicit proposer uses) on the organ's decoded candidate logits vs clair.csp.exact_dedP — with
NO factors given as input. Host = frozen OLMo + optional LoRA (so OLMo can adapt to expose the
constraints latently to α).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .augmented import CellReader            # per-cell cross-attention over the whole prompt
from .ldt import make_mixer                  # SwiGLU / geom channel mixer (reused, LDT-faithful)


# --------------------------------------------------------------------------- α: dense latent projector
class DenseLatentProjector(nn.Module):
    """OLMo hidden -> (initial candidate logits [B,N,K], latent context [B,N,dctx]). DENSE + learned;
    NO symbolic program, NO per-factor supervision. Each cell's representation fuses its mention-pool
    IDENTITY with a cross-attention READ of the whole prompt (where its constraints live)."""
    def __init__(self, D, K, dctx=256, dp=384, heads=6, hidden=384):
        super().__init__()
        self.reader = CellReader(D, dp, heads)                 # cell queries -> read whole prompt
        din = D + dp                                           # identity (mention-pool) + read context
        self.init_head = nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, K))
        self.ctx_head = nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, dctx))

    def forward(self, v_mean, h, attn_mask):
        # v_mean [B,N,D] mention-pool cell identity ; h [B,T,D] host states ; attn_mask [B,T]
        ctx = self.reader(v_mean, h, attn_mask)                # [B,N,dp]
        feat = torch.cat([v_mean, ctx], dim=-1)
        return self.init_head(feat), self.ctx_head(feat)       # b0 [B,N,K], context [B,N,dctx]


# --------------------------------------------------------------------------- masked attention over cells
class MaskAttn(nn.Module):
    """Bidirectional attention over the cell set, masking padded cells (the constraint graph is the
    only asymmetry — no absolute cell position, so weights run on any n)."""
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x, vmask):
        B, N, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = (z.view(B, N, self.h, self.dh).transpose(1, 2) for z in (q, k, v))
        bias = torch.zeros(B, 1, 1, N, device=x.device, dtype=x.dtype)
        bias = bias.masked_fill(vmask[:, None, None, :] < 0.5, float("-inf"))   # mask pad KEYS
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.proj(o.transpose(1, 2).reshape(B, N, D))


class NLayer(nn.Module):
    """One transformer layer over cells: LN->masked MHA->res, LN->mixer->res."""
    def __init__(self, d, heads, mixer):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MaskAttn(d, heads)
        self.mix = make_mixer(mixer, d)

    def forward(self, h, vmask):
        h = h + self.attn(self.ln1(h), vmask)
        h = h + self.mix(self.ln2(h))
        return h


# --------------------------------------------------------------------------- organ: recurrent narrower
class LatentNarrower(nn.Module):
    """LDT-faithful recurrent latent narrower. Encodes each cell from (current candidate logits, α's
    latent context); a weight-tied attention stack is unrolled for T iterations, re-injecting the
    lattice encoding + context each step. MONOTONE: emits a non-negative elimination pressure that is
    subtracted from the candidate logits (survival prob only decreases). Deep supervision over the
    last `ds` iterations."""
    def __init__(self, K, dctx, d=128, heads=4, n_layers=2, T=12, ds=6, mixer="ffn",
                 reinject=0.5, monotone=True):
        super().__init__()
        self.K, self.T, self.ds = K, T, max(1, min(ds, T))
        self.reinject, self.monotone = reinject, monotone
        self.lattice_proj = nn.Linear(K, d)                    # re-encode the lattice (candidate logits)
        self.context_proj = nn.Linear(dctx, d)                 # α's latent context (constant over steps)
        self.layers = nn.ModuleList([NLayer(d, heads, mixer) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.elim_head = nn.Linear(d, K)                       # elimination pressure per (cell,value)

    def core(self, h, vmask):
        for layer in self.layers:
            h = layer(h, vmask)
        return h

    def forward(self, b0, context, vmask):
        ctx = self.context_proj(context)                       # [B,N,d] re-injected each step
        b = b0
        h = self.lattice_proj(b) + ctx
        sup = []
        for t in range(self.T):
            h = self.core(h, vmask) + self.reinject * (self.lattice_proj(b) + ctx)
            f = self.ln_f(h)
            if self.monotone:
                b = b - F.softplus(self.elim_head(f))          # monotone narrowing (prob decreases)
            else:
                b = self.elim_head(f)                          # free per-step logits
            if t >= self.T - self.ds:
                sup.append(b)
        return b, sup


# --------------------------------------------------------------------------- the latent organ (host + α + narrower)
class LatentOrgan(nn.Module):
    """Frozen OLMo (+ optional LoRA) -> mid-layer hidden -> α (dense latent projection) -> recurrent
    latent narrower -> per-cell candidate logits. The host sees ONLY the problem text; the organ never
    receives explicit factors."""
    def __init__(self, host, Nmax, K, mid_layer, train_host, dctx=256, d=128,
                 heads=4, n_layers=2, T=12, ds=6, dp=384, alpha_heads=6, mixer="ffn",
                 reinject=0.5, monotone=True):
        super().__init__()
        self.host = host
        self.Nmax, self.K = Nmax, K
        self.mid_layer = mid_layer
        self.train_host = train_host                           # True iff LoRA (grad must flow to host)
        D = host.config.hidden_size
        self.D = D
        self.alpha = DenseLatentProjector(D, K, dctx, dp, alpha_heads)
        self.narrower = LatentNarrower(K, dctx, d, heads, n_layers, T, ds, mixer, reinject, monotone)

    def host_encode(self, ba):
        """Mid-layer hidden states [B,T,D]. With LoRA, grad flows through the adapters; otherwise the
        host is run under no_grad (α trains on detached hidden)."""
        kw = dict(input_ids=ba["input_ids"], attention_mask=ba["attn"],
                  output_hidden_states=True, use_cache=False)
        if self.train_host:
            out = self.host(**kw)
        else:
            with torch.no_grad():
                out = self.host(**kw)
        return out.hidden_states[self.mid_layer]

    def forward(self, ba):
        h = self.host_encode(ba)
        hd = h.float()
        m = ba["mention"]
        denom = m.sum(-1, keepdim=True).clamp_min(1e-6)
        v_mean = torch.einsum("bnt,btd->bnd", m, hd) / denom   # [B,N,D] cell identity (mention-pool)
        b0, context = self.alpha(v_mean, hd, ba["attn"])
        b, sup = self.narrower(b0, context, ba["vmask"])
        return b, sup, b0

    def organ_parameters(self):
        return list(self.alpha.parameters()) + list(self.narrower.parameters())

    def lora_parameters(self):
        return [p for n, p in self.host.named_parameters() if p.requires_grad]


# --------------------------------------------------------------------------- dominate-dedₚ loss
def dominate_dedp_loss(sup, tgt, vmask, wpos=6.0, wneg=0.5):
    """Soundness-asymmetric BCE toward dedₚ (the SAME loss the explicit proposer uses, on the organ's
    OUTPUT): a HEAVY wpos penalty on driving a dedₚ-survivor's survival prob to 0 (soundness), a light
    wneg pull on dedₚ-eliminated values. Over all valid (cell,value) slots; deep-supervised."""
    eps = 1e-6
    alive = vmask.unsqueeze(-1)                                # [B,N,1]; every value valid (K fixed)
    denom = alive.expand_as(tgt).sum().clamp_min(1.0)
    tot = 0.0
    for b in sup:
        p = torch.sigmoid(b)
        bce = -(wpos * tgt * torch.log(p + eps) + wneg * (1 - tgt) * torch.log(1 - p + eps))
        tot = tot + (bce * alive).sum() / denom
    return tot / len(sup)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
