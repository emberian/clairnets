"""clair/glados_woven.py — the corrected-D woven GLaDOS model (codex review).

The earlier augmented model (clair.augmented) had ZERO causal influence on OLMo: OLMo was frozen,
its hiddens detached, and the organ was a TERMINAL read-out head. This file builds the corrected
architecture per the outside (codex) review. Four corrections, each a requirement:

  alpha COMPILES, not solves.  From OLMo hidden states at a MID layer, emit a TYPED FACTOR-GRAPH
      PROGRAM: per-cell variables + domains (the givens/evidence) + FACTORS (scopes + relation
      TABLES). NOT a narrowed/answer state. We generalize augmented.PairGroundedEdgeReader (which
      emitted a single neq edge logit per pair) to emit a full per-pair relation TABLE over KxK,
      reusing its locally-grounded co-occurrence extraction. Equivariant over cells.

  the ORGAN narrows (proposer.FactorGraphProposer), running RECURRENTLY BETWEEN OLMo layers.
      A SINGLE OLMo forward pass carries forward hooks: at the compile layer alpha emits the program
      and the organ does an initial narrowing; at each later write-back layer the organ takes one
      more step on the (stop-grad) lattice state -> the organ recurs in the gaps between layers.
      We do NOT backprop through repeated OLMo passes (there is only ONE pass); the lattice
      recurrence is truncated by detaching the carried var-mask state. EVERY organ step is supervised
      against the exact per-cell transformer ded_P (the dominate-ded_P soundness loss, on the TRUE
      program — teacher-forced, exactly the clair.run_general objective).

  gamma STRUCTURED + MODULAR.  Project the narrowed per-(cell,candidate) lattice back into the
      residual stream EQUIVARIANTLY: a SHARED cell projection (not a flatten tied to max N) scattered
      onto each cell's mention token positions, behind a zero-init tanh gate (exact no-op at init).
      gamma is a swappable nn.Module (GammaCoupling) — the oracle-readout de-risk will fix its form.

  generative.  OLMo's LM head GENERATES the answer token(s). loss = LM cross-entropy on the answer
      span (prompt masked) + the per-step organ dominate-ded_P loss + an alpha program-recon loss.
      Base OLMo frozen + LoRA (peft r=16).

The host is NOT the recurrence: alpha gets gradient from the LM loss through gamma -> organ -> alpha,
all within the single OLMo pass; the organ's own recurrence is its loop, truncated.
"""
from __future__ import annotations

import math
import itertools as it

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .proposer import FactorGraphProposer, size_for
from . import csp as C
from . import curriculum as CU
from .run_general import loss_fn as organ_loss_fn, Exact

# ---- organ padding budget (own, larger than run_general's N=8 so we can eval OOD N=11) ----
NMAX, DMAX, AMAX = 12, 8, 3
PAIRS = list(it.combinations(range(NMAX), 2))      # all unordered pairs = the binary factor scaffold
MMAX = len(PAIRS)                                   # C(12,2) = 66 factors
PAIR_I = torch.tensor([i for i, _ in PAIRS])
PAIR_J = torch.tensor([j for _, j in PAIRS])

SHARED = Exact()


# ===================================================================== alpha: the COMPILER
class CellReader(nn.Module):
    """Per-cell cross-attention over the whole prompt (from augmented.CellReader): each cell, keyed by
    the mean of its mention-token states, gathers its facts. The grounded compile context."""
    def __init__(self, D, dp=512, heads=8):
        super().__init__()
        self.h, self.dp, self.dk = heads, dp, dp // heads
        self.wq = nn.Linear(D, dp); self.wk = nn.Linear(D, dp); self.wv = nn.Linear(D, dp)
        self.qln = nn.LayerNorm(D)

    def forward(self, v_mean, h, attn):
        B, N, _ = v_mean.shape; T = h.size(1)
        q = self.wq(self.qln(v_mean)).view(B, N, self.h, self.dk).transpose(1, 2)
        k = self.wk(h).view(B, T, self.h, self.dk).transpose(1, 2)
        vv = self.wv(h).view(B, T, self.h, self.dk).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dk)
        att = att.masked_fill(attn[:, None, None, :] < 0.5, -1e9).softmax(-1)
        return torch.matmul(att, vv).transpose(1, 2).reshape(B, N, self.dp)


class PairTableReader(nn.Module):
    """Locally-grounded per-pair RELATION-TABLE reader — generalizes augmented.PairGroundedEdgeReader
    from a single present/absent edge logit to a full KxK allowed-tuple table. Each ordered pair (i,j)
    forms a symmetric query, cross-attends over the prompt biased toward the union of i & j's mention
    neighborhoods (the local clause that couples them), and a shared head emits a KxK table from that
    local context + a hard co-occurrence feature. One local clause -> one factor, independent of N."""
    def __init__(self, D, K, dp=512, heads=8, window=7):
        super().__init__()
        self.h, self.dp, self.dk, self.K = heads, dp, dp // heads, K
        self.window = window | 1
        self.qln = nn.LayerNorm(2 * D)
        self.wq = nn.Linear(2 * D, dp); self.wk = nn.Linear(D, dp); self.wv = nn.Linear(D, dp)
        self.beta = nn.Parameter(torch.tensor(2.0))
        self.cls = nn.Sequential(nn.Linear(dp + 1, dp), nn.GELU(), nn.Linear(dp, K * K))

    def _prox(self, mention):
        B, N, T = mention.shape; w = self.window
        p = F.max_pool1d(mention.reshape(B * N, 1, T), kernel_size=w, stride=1, padding=w // 2)
        return p.reshape(B, N, T)

    def forward(self, v_mean, h, mention, attn):
        B, N, D = v_mean.shape; T = h.size(1); K = self.K
        prox = self._prox(mention)
        a = v_mean[:, :, None, :].expand(B, N, N, D)
        b = v_mean[:, None, :, :].expand(B, N, N, D)
        pf = torch.cat([a + b, a * b], -1)
        q = self.wq(self.qln(pf)).view(B, N * N, self.h, self.dk).transpose(1, 2)
        k = self.wk(h).view(B, T, self.h, self.dk).transpose(1, 2)
        vv = self.wv(h).view(B, T, self.h, self.dk).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dk)
        union = (prox[:, :, None, :] + prox[:, None, :, :]).reshape(B, N * N, T)
        att = att + F.softplus(self.beta) * union[:, None, :, :]
        att = att.masked_fill(attn[:, None, None, :] < 0.5, -1e9).softmax(-1)
        ctx = torch.matmul(att, vv).transpose(1, 2).reshape(B, N * N, self.dp)
        cooc = (prox[:, :, None, :] * prox[:, None, :, :]).max(-1).values.reshape(B, N * N, 1)
        tab = self.cls(torch.cat([ctx, cooc], -1)).reshape(B, N, N, K, K)
        # symmetric relation: swapping (i,j) and (vi,vj) jointly leaves an unordered factor unchanged
        return 0.5 * (tab + tab.permute(0, 2, 1, 4, 3))


class Compiler(nn.Module):
    """alpha. OLMo mid-layer hidden states -> a typed factor-graph PROGRAM (domains + relation tables).
    Emits ONLY evidence + structure (the givens & the constraints), never an answer/narrowed state —
    the narrowing is the organ's job, and gamma reads the ORGAN, so alpha cannot bypass it."""
    def __init__(self, D, K, dp=512, heads=8, window=7):
        super().__init__()
        self.K = K
        self.reader = CellReader(D, dp, heads)
        self.dom_head = nn.Sequential(nn.Linear(dp, dp), nn.GELU(), nn.Linear(dp, K))
        self.tab = PairTableReader(D, K, dp, heads, window)

    def forward(self, v_mean, h, mention, attn):
        ctx = self.reader(v_mean, h, attn)
        dom_logits = self.dom_head(ctx)                       # [B,N,K]  per-cell domain (givens)
        tab_logits = self.tab(v_mean, h, mention, attn)       # [B,N,N,K,K] per-pair relation tables
        return dom_logits, tab_logits


# ===================================================================== gamma: STRUCTURED + MODULAR
class GammaCoupling(nn.Module):
    """Swappable structured write-back of the narrowed lattice into the residual stream. Contract:
    forward(hs, lattice, mention, vmask) -> residual delta with the SAME shape as hs, EXACTLY zero at
    init (no-op). Subclasses implement the coupling; the oracle-readout de-risk will pick the form."""
    def forward(self, hs, lattice, mention, vmask):                # pragma: no cover - interface
        raise NotImplementedError


class ScatterGamma(GammaCoupling):
    """Equivariant per-cell scatter. A SHARED projection maps each cell's narrowed candidate vector
    (+ determinacy features) to a residual delta, scattered onto that cell's mention token positions.
    Shared across cells and decoupled from N (no flatten); zero-init tanh gate -> exact no-op at init."""
    def __init__(self, D, K, hidden=256, gate_init=0.0):
        super().__init__()
        self.cell_proj = nn.Sequential(nn.Linear(K + 2, hidden), nn.GELU(), nn.Linear(hidden, D))
        # zero-init gate => exact no-op at init. gate_init>0 forces the organ into the residual stream
        # from step 0 (an ABLATION: tests whether the dead-organ is the zero-init trap or a real bypass).
        self.alpha = nn.Parameter(torch.full((1,), float(gate_init)))

    def cell_features(self, lattice, vmask):
        marg = lattice / lattice.sum(-1, keepdim=True).clamp_min(1e-6)
        peak = marg.max(-1, keepdim=True).values
        nal = (lattice > 0.1).float().sum(-1, keepdim=True)
        feat = torch.cat([marg, peak, nal], -1)                 # [B,N,K+2]
        return self.cell_proj(feat) * vmask[..., None]          # [B,N,D]

    def forward(self, hs, lattice, mention, vmask):
        cell = self.cell_features(lattice, vmask)               # [B,N,D]
        denom = mention.sum(1).clamp_min(1.0)                   # [B,T] tokens' mention multiplicity
        delta = torch.einsum("bnt,bnd->btd", mention, cell) / denom[..., None]
        return torch.tanh(self.alpha) * delta


GAMMAS = {"scatter": ScatterGamma}


# ===================================================================== organ feature builders
def featurize_true(items, dev):
    """TRUE program featurization (clair.run_general.featurize, re-typed to OUR N/M budget). Builds the
    organ input tensors from a list of (csp, dom) so the organ is teacher-forced on exact programs."""
    B = len(items); rel_dim = DMAX ** AMAX
    var_mask = np.zeros((B, NMAX, DMAX), np.float32); given = np.zeros((B, NMAX), np.float32)
    var_valid = np.zeros((B, NMAX), np.float32)
    fac_rel = np.zeros((B, MMAX, rel_dim), np.float32); fac_arity = np.zeros((B, MMAX, AMAX), np.float32)
    fac_valid = np.zeros((B, MMAX), np.float32)
    edge_var = np.full((B, MMAX, AMAX), NMAX, np.int64); edge_valid = np.zeros((B, MMAX, AMAX), np.float32)
    for bi, (csp, dom) in enumerate(items):
        for i in range(csp.n):
            var_valid[bi, i] = 1.0
            for v in dom[i]:
                var_mask[bi, i, v] = 1.0
            if len(dom[i]) == 1:
                given[bi, i] = 1.0
        for fi, (sc, al) in enumerate(csp.cons):
            if fi >= MMAX:
                break
            fac_valid[bi, fi] = 1.0; fac_arity[bi, fi, len(sc) - 1] = 1.0
            t = np.zeros((DMAX,) * AMAX, np.float32)
            for tup in al:
                sl = [slice(None)] * AMAX
                for p in range(len(sc)):
                    sl[p] = tup[p]
                t[tuple(sl)] = 1.0
            fac_rel[bi, fi] = t.reshape(-1)
            for p, cell in enumerate(sc):
                edge_var[bi, fi, p] = cell; edge_valid[bi, fi, p] = 1.0
    tt = lambda a: torch.as_tensor(a, device=dev)
    return dict(var_mask=tt(var_mask), given=tt(given), var_valid=tt(var_valid), fac_rel=tt(fac_rel),
                fac_arity=tt(fac_arity), fac_valid=tt(fac_valid), edge_var=tt(edge_var),
                edge_valid=tt(edge_valid))


def build_targets(items, ex, dev):
    """ded_P supervision target + conflict flag (clair.run_general.build_targets at our budget)."""
    B = len(items)
    tgt = np.zeros((B, NMAX, DMAX), np.float32); conflict = np.zeros((B,), np.float32)
    for bi, (csp, dom) in enumerate(items):
        ded = ex.dedP(csp, dom)
        if all(len(c) == 0 for c in ded):
            conflict[bi] = 1.0
        for i in range(csp.n):
            for v in ded[i]:
                tgt[bi, i, v] = 1.0
    return torch.as_tensor(tgt, device=dev), torch.as_tensor(conflict, device=dev)


def emitted_features(dom_prob, pair_tab, vmask):
    """EMITTED program -> organ input tensors (torch, DIFFERENTIABLE). dom_prob [B,N,K] soft domains,
    pair_tab [B,N,N,K,K] soft relation tables, vmask [B,N]. Factors = the all-pairs binary scaffold;
    padded values (>=K) default ALLOWED (var_mask excludes them anyway)."""
    B, N, K = dom_prob.shape
    dev = dom_prob.device
    var_mask = torch.zeros(B, NMAX, DMAX, device=dev)
    var_mask[:, :N, :K] = dom_prob * vmask[..., None]
    given = torch.zeros(B, NMAX, device=dev)
    var_valid = torch.zeros(B, NMAX, device=dev); var_valid[:, :N] = vmask
    ii, jj = PAIR_I.to(dev), PAIR_J.to(dev)
    pt = pair_tab[:, ii, jj]                                    # [B,M,K,K]
    tab2 = torch.ones(B, MMAX, DMAX, DMAX, device=dev)
    tab2[:, :, :K, :K] = pt
    fac_rel = tab2[..., None].expand(B, MMAX, DMAX, DMAX, DMAX).reshape(B, MMAX, DMAX ** AMAX)
    fac_arity = torch.zeros(B, MMAX, AMAX, device=dev); fac_arity[:, :, 1] = 1.0
    pair_valid = vmask[:, ii] * vmask[:, jj]                    # [B,M]
    edge_var = torch.full((B, MMAX, AMAX), NMAX, dtype=torch.long, device=dev)
    edge_var[:, :, 0] = ii[None, :]; edge_var[:, :, 1] = jj[None, :]
    edge_valid = torch.zeros(B, MMAX, AMAX, device=dev)
    edge_valid[:, :, 0] = pair_valid; edge_valid[:, :, 1] = pair_valid
    return dict(var_mask=var_mask, given=given, var_valid=var_valid, fac_rel=fac_rel,
                fac_arity=fac_arity, fac_valid=pair_valid, edge_var=edge_var, edge_valid=edge_valid)


def organ_call(organ, feat):
    return organ(feat["var_mask"], feat["given"], feat["fac_rel"], feat["fac_arity"],
                 feat["edge_var"], feat["edge_valid"], feat["var_valid"], feat["fac_valid"])


# ===================================================================== decoder-layer access (peft-safe)
def get_decoder_layers(model):
    """Find the decoder-layer ModuleList through any peft/HF wrapping."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) and hasattr(mod[0], "self_attn"):
            return mod
    raise RuntimeError("could not locate decoder layers")


# ===================================================================== the woven model
class GladosWoven(nn.Module):
    def __init__(self, model, tok, K=3, compile_layer=5, writeback_layers=(7, 9, 11),
                 organ_target=1.5e5, organ_R=8, gamma="scatter", dp=512, gate_init=0.0):
        super().__init__()
        self.model = model                         # peft-wrapped OLMo (base frozen + LoRA)
        self.tok = tok
        self.K = K
        D = model.config.hidden_size; self.D = D
        nL = model.config.num_hidden_layers
        self.compile_layer = compile_layer
        self.writeback_layers = tuple(w for w in writeback_layers if w < nL)
        self.compiler = Compiler(D, K, dp=dp)
        d, npar = size_for("full", NMAX, DMAX, MMAX, AMAX, organ_target, R=organ_R, ds=min(4, organ_R))
        self.organ = FactorGraphProposer("full", NMAX, DMAX, MMAX, AMAX, d=d, R=organ_R,
                                         ds=min(4, organ_R))
        self.organ_params = npar
        self.gamma = GAMMAS[gamma](D, K, gate_init=gate_init)
        self._layers = get_decoder_layers(model)
        # forward-hook state (set per forward by _run_host)
        self._ba = None; self._lattice = None; self._emit_feat = None
        self.control = "none"; self._perm = None; self._kperm = None

    # ----------------------------------------------------------------- alpha + organ inside hooks
    def _compile_hook(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        ba = self._ba
        h = hs.float()
        m = ba["mention"]                                                   # [B,N,T]
        denom = m.sum(-1, keepdim=True).clamp_min(1e-6)
        v_mean = torch.einsum("bnt,btd->bnd", m, h) / denom                 # cell identity
        dom_logits, tab_logits = self.compiler(v_mean, h, m, ba["attn"])
        self._dom_logits, self._tab_logits = dom_logits, tab_logits
        dom_prob = torch.sigmoid(dom_logits); pair_tab = torch.sigmoid(tab_logits)
        feat = emitted_features(dom_prob, pair_tab, ba["vmask"])
        self._emit_feat = feat
        b, _, _ = organ_call(self.organ, feat)
        self._lattice = feat["var_mask"] * torch.sigmoid(b)                 # [B,NMAX,DMAX] soft meet
        return output

    def _control_lattice(self, lattice):
        if self.control == "shuffle" and self._perm is not None:
            return lattice[self._perm]
        if self.control == "permute" and self._kperm is not None:
            out = lattice.clone()
            out[:, :, :self.K] = lattice[:, :, self._kperm]
            return out
        return lattice

    def _writeback_hook(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        ba = self._ba
        lat = self._control_lattice(self._lattice)
        delta = self.gamma(hs.float(), lat[:, :, :self.K], ba["mention"], ba["vmask"])
        hs = hs + delta.to(hs.dtype)
        # advance the organ for the NEXT write-back: truncated recurrence (stop-grad the carried state)
        vm_next = self._lattice.detach()
        feat2 = dict(self._emit_feat); feat2["var_mask"] = vm_next
        b, _, _ = organ_call(self.organ, feat2)
        self._lattice = vm_next * torch.sigmoid(b)
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs

    def _run_host(self, ba):
        """One OLMo forward pass with the compile + write-back hooks installed."""
        self._ba = ba
        handles = [self._layers[self.compile_layer].register_forward_hook(self._compile_hook)]
        for w in self.writeback_layers:
            handles.append(self._layers[w].register_forward_hook(self._writeback_hook))
        try:
            out = self.model(input_ids=ba["input_ids"], attention_mask=ba["attn"])
        finally:
            for hd in handles:
                hd.remove()
        return out.logits

    # ----------------------------------------------------------------- training forward
    def forward(self, ba):
        logits = self._run_host(ba)
        # LM cross-entropy on the answer span (prompt + pad masked via labels = -100)
        lab = ba["labels"]
        sl = logits[:, :-1, :].reshape(-1, logits.size(-1)).float()
        lm_ce = F.cross_entropy(sl, lab[:, 1:].reshape(-1), ignore_index=-100)
        # organ dominate-ded_P on the TRUE program (teacher-forced; supervises EVERY deep-sup step)
        items = ba["items"]
        tfeat = featurize_true(items, logits.device)
        b, cls, sup = organ_call(self.organ, tfeat)
        tgt, conflict = build_targets(items, SHARED, logits.device)
        organ_loss = organ_loss_fn(sup, tfeat["var_mask"], tfeat["var_valid"], tgt, conflict)
        with torch.no_grad():
            killed = (tgt * tfeat["var_mask"] * (torch.sigmoid(b) < 0.5).float()).sum()
            elig = (tgt * tfeat["var_mask"]).sum()
            organ_fe = (killed / elig.clamp_min(1)).item()
        # alpha program-reconstruction (the grounding signal): emitted domains/tables vs the TRUE program
        prog_loss = self._program_recon(ba)
        return {"lm_ce": lm_ce, "organ": organ_loss, "prog": prog_loss,
                "organ_fe": organ_fe, "logits": logits}

    def _program_recon(self, ba):
        dom_t, dom_m = ba["dom_target"], ba["dom_mask"]                    # [B,N,K], [B,N,1]
        tab_t, tab_m = ba["tab_target"], ba["tab_mask"]                    # [B,N,N,K,K], [B,N,N,1,1]
        dl = F.binary_cross_entropy_with_logits(self._dom_logits, dom_t, reduction="none")
        dl = (dl * dom_m).sum() / dom_m.sum().clamp_min(1)
        tl = F.binary_cross_entropy_with_logits(self._tab_logits, tab_t, reduction="none")
        tl = (tl * tab_m).sum() / tab_m.sum().clamp_min(1)
        return dl + tl

    # ----------------------------------------------------------------- generation + controls
    @torch.no_grad()
    def generate(self, ba, max_new_tokens=4):
        """Greedy decode the answer span. Recomputes the program each step (mention zero-padded for the
        generated tokens; v_mean comes from the prompt region, so the program is effectively frozen)."""
        self.eval()
        ids = ba["prompt_ids"].clone(); attn = ba["prompt_attn"].clone()
        B = ids.size(0); dev = ids.device
        mention = ba["mention"]; vmask = ba["vmask"]
        new = [[] for _ in range(B)]
        eos = self.tok.eos_token_id
        finished = torch.zeros(B, dtype=torch.bool, device=dev)
        for _ in range(max_new_tokens):
            T = ids.size(1)
            men = torch.zeros(B, mention.size(1), T, device=dev)
            men[:, :, :mention.size(2)] = mention[:, :, :min(mention.size(2), T)]
            cur = {"input_ids": ids, "attn": attn, "mention": men, "vmask": vmask}
            logits = self._run_host(cur)
            nxt = logits[:, -1, :].argmax(-1)
            for bcnt in range(B):
                if not finished[bcnt]:
                    new[bcnt].append(int(nxt[bcnt]))
            finished = finished | (nxt == eos)
            ids = torch.cat([ids, nxt[:, None]], 1)
            attn = torch.cat([attn, torch.ones(B, 1, device=dev, dtype=attn.dtype)], 1)
            if finished.all():
                break
        return [self.tok.decode(t, skip_special_tokens=True).strip() for t in new]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# ===================================================================== data
def make_problems(rng, n, relations=("coloring",)):
    return [CU.gen_problem(rng, relation=str(rng.choice(relations))) for _ in range(n)]


def _answer_text(p):
    return CU.canonical_answer(p)                 # "red" / "3" / "cannot be determined"


def build_batch(probs, tok, K, dev, with_answer=True):
    """Tokenize render+answer; build labels (answer span only), mention masks, the TRUE csp items,
    and the alpha program-reconstruction targets. Coloring/equality only (K-valued, arity<=2)."""
    prompts, fulls, items, ns = [], [], [], []
    dom_t = np.ones((len(probs), NMAX, K), np.float32)
    dom_m = np.zeros((len(probs), NMAX, 1), np.float32)
    tab_t = np.ones((len(probs), NMAX, NMAX, K, K), np.float32)
    tab_m = np.zeros((len(probs), NMAX, NMAX, 1, 1), np.float32)
    eyeK = np.eye(K, dtype=np.float32)
    mention_spans = []
    for bi, p in enumerate(probs):
        text = CU.canonical_render(p)
        prompt = text + " Answer:"
        ans = _answer_text(p)
        prompts.append(prompt); fulls.append(prompt + " " + ans)
        items.append((p.csp, p.csp.full())); ns.append(p.n)
        mention_spans.append(CU.entity_mentions(prompt, p.n))
        for i in range(p.n):
            dom_m[bi, i, 0] = 1.0
            for a in range(p.n):
                if a != i:
                    tab_m[bi, i, a, 0, 0] = 1.0
        for f in p.facts:
            if f[0] == "pin":
                _, a, v = f
                dom_t[bi, a, :] = 0.0; dom_t[bi, a, v] = 1.0
            elif f[0] == "neq":
                _, a, b = f
                tab_t[bi, a, b] = 1.0 - eyeK; tab_t[bi, b, a] = 1.0 - eyeK
            elif f[0] == "eq":
                _, a, b = f
                tab_t[bi, a, b] = eyeK.copy(); tab_t[bi, b, a] = eyeK.copy()
    enc = tok(fulls, return_offsets_mapping=True, padding=True, return_tensors="pt")
    penc = tok(prompts, return_offsets_mapping=True, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(dev); attn = enc["attention_mask"].to(dev)
    B, T = input_ids.shape
    labels = torch.full((B, T), -100, device=dev)
    offs = enc["offset_mapping"].tolist()
    for bi in range(B):
        plen = len(prompts[bi])
        for ti, (a, c) in enumerate(offs[bi]):
            if a == c:
                continue
            if a >= plen and attn[bi, ti] > 0:           # token in the answer region
                labels[bi, ti] = input_ids[bi, ti]
    # mention masks over the FULL sequence (mentions live in the prompt region)
    mention = torch.zeros(B, NMAX, T, device=dev)
    for bi, spans in enumerate(mention_spans):
        for v, sps in spans.items():
            for (cs, ce) in sps:
                for ti, (a, c) in enumerate(offs[bi]):
                    if a == c:
                        continue
                    if a < ce and c > cs:
                        mention[bi, v, ti] = 1.0
    vmask = torch.zeros(B, NMAX, device=dev)
    for bi, n in enumerate(ns):
        vmask[bi, :n] = 1.0
    pmention = torch.zeros(B, NMAX, penc["input_ids"].size(1), device=dev)
    poffs = penc["offset_mapping"].tolist()
    for bi, spans in enumerate(mention_spans):
        for v, sps in spans.items():
            for (cs, ce) in sps:
                for ti, (a, c) in enumerate(poffs[bi]):
                    if a == c:
                        continue
                    if a < ce and c > cs:
                        pmention[bi, v, ti] = 1.0
    return {"input_ids": input_ids, "attn": attn, "labels": labels, "mention": mention,
            "vmask": vmask, "items": items, "prompts": prompts,
            "prompt_ids": penc["input_ids"].to(dev), "prompt_attn": penc["attention_mask"].to(dev),
            "pmention": pmention.to(dev),
            "dom_target": torch.as_tensor(dom_t, device=dev),
            "dom_mask": torch.as_tensor(dom_m, device=dev),
            "tab_target": torch.as_tensor(tab_t, device=dev),
            "tab_mask": torch.as_tensor(tab_m, device=dev),
            "answers": [_answer_text(p) for p in probs], "probs": probs}


# ===================================================================== no-op verification
@torch.no_grad()
def verify_noop(gw, base_logits_fn, ba):
    """At init: LoRA delta = 0 (B-init zero) and gamma gate = tanh(0) = 0, so the woven forward must
    equal base OLMo bit-for-bit on the prompt logits. Returns max|diff| over real tokens."""
    woven = gw._run_host({"input_ids": ba["input_ids"], "attn": ba["attn"],
                          "mention": ba["mention"], "vmask": ba["vmask"]}).float()
    base = base_logits_fn(ba["input_ids"], ba["attn"]).float()
    mask = ba["attn"][:, :, None] > 0
    return float(((woven - base).abs() * mask).max())
