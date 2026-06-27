"""clair/organ/problem_interreducer.py — THE ALWAYS-ON PROBLEM INTERREDUCER, as a TUNABLE K-knob.

The idea (Ember): when a problem X is presented, DON'T just route it to one faculty — also reduce it
into OTHER faculties and present the multimodal bank-view through γ, so the LM can mine corroborating /
complementary structure. BUT the NUMBER of views K is a SWEPT KNOB, not hard-wired all-on, because
γ-bandwidth is finite: forcing redundant views might crowd out bandwidth the LM would use for other
things. This module builds the interreducer and the K-sweep that measures parallelism-vs-diversity.

THE K VIEWS (ordered cheap→gadget; per-instance, capped-not-excluded):
  slot 0  "ext"   — the CHEAP, always-available, structure-preserving reduction: SAT → CSP
                    (clause → allowed-tuple), solved by the deployed certified composer (arc/factor
                    consistency + gated neural core). Identity decode. ALWAYS present.
  slot 1  "alg"   — a CHEAP structure-RECOGNIZING reduction that exposes algebra the ext view can't
                    crack with local consistency:
                      xorsat → GF(2) row-space system  (the affine wall — exact where ext abstains)
                      cnf    → pairwise joint-factor CSP (natural-join coupling; sound, identity)
  slot 2  "ising" — the GADGET reduction sat→MIS (Lucas clause-triangle) → Ising, solved by the
                    mean-field optimiser, decoded back. APPROXIMATE (output-checked), so it carries a
                    LOW reliability flag → the LM can down-weight it. CAPPED at gadget_n_max: included
                    when the blow-up (≈3·#clauses nodes) fits the budget, DROPPED per-instance when it
                    doesn't (Ember's point: our instances are small, so don't exclude a priori).

K is the knob:  k1 = [ext]  ·  kfew = [ext, alg]  ·  kall = [ext, alg, ising].

Each view's solved lattice is decoded back to the SOURCE-variable cell space [n_src, K] and presented
as a SEPARATE channel through a multi-view γ (MultiViewGamma): a shared per-view encoder + an
ATTENTION pool whose query is the host hidden, so the LM ALLOCATES the bandwidth (attends / down-
weights) rather than being forced to fuse all views equally. Zero-init gate ⇒ bitwise no-op @ init.

The Rust dedp_batch (clair.csp FAST path) batches the per-view exact-dedP solves inside the composer's
verifier gate — so presenting K views adds no new Rust, only K× the (now-cheap) solve.

THE EXPERIMENT (run_sweep): sweep K, ADEQUATE length (~1500 steps), measure
  · DIVERSITY benefit on route_required (xorsat, no direct faculty — the ext view hits the affine wall;
    the gf2 alg-view cracks it) + heldout_reduction_path: does MORE views help solve-by-reduction?
  · BANDWIDTH cost on DIRECT tasks (in_dist sat3/sat2 the LM has natively): does more views HURT
    (γ-bandwidth crowded by redundant views → worse than k1)?
VERDICT: does more-views help (diversity), hurt (crowding), or net-neutral (mine+ignore)? Optimal K?
"""
from __future__ import annotations

import argparse
import contextlib
import itertools as it
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import csp as C
from ..latent_organ import DenseLatentProjector, dominate_dedp_loss
from ..oracle_readout import _mention_tensor, ABSTAIN_STR
from . import reductions as R
from .bank_woven import BankComposerOrgan
from .protocol import CSPState

K_MODES = {"k1": ["ext"], "kfew": ["ext", "alg"], "kall": ["ext", "alg", "ising"]}
TRUST = {"ext": 1.0, "alg": 1.0, "ising": 0.35}        # sound views trusted; ising approximate → low


# ============================================================ view construction (CPU, numpy)
def _gf2_system(sat: R.Sat):
    """The GF(2) system (A,b) for an XOR-SAT instance (each parity clause = one row). GF2RowSpace
    solves Ax=b EXACTLY — the affine wall the ext view's local consistency abstains on."""
    n = sat.n
    rows, rhs = [], []
    for cl in sat.clauses:
        par = cl[-1]; lits = cl[:-1]
        row = np.zeros(n, np.uint8)
        for v, _ in lits:
            row[v] ^= 1
        rows.append(row); rhs.append(int(par))
    A = np.array(rows, np.uint8) if rows else np.zeros((0, n), np.uint8)
    b = np.array(rhs, np.uint8) if rhs else np.zeros(0, np.uint8)
    return ("gf2", A, b)


def _joint_factor_csp(sat: R.Sat, m_max=80) -> C.CSP:
    """The cnf 'alg' view: the ext CSP AUGMENTED with pairwise natural-joins of clause-factors that
    SHARE a variable (union scope ≤ 3). Each join is the conjunction's projection ⇒ SOUND, identity
    solution set, but a STRONGER local constraint than the two clauses apart, so the composer's
    consistency can narrow more. A genuinely different structural presentation of the same problem."""
    base = R.sat_to_csp(sat).csp
    cons = list(base.cons)
    extra = []
    for i in range(len(base.cons)):
        sc_i, al_i = base.cons[i]
        for j in range(i + 1, len(base.cons)):
            sc_j, al_j = base.cons[j]
            si, sj = set(sc_i), set(sc_j)
            if not (si & sj):
                continue                                       # no shared var ⇒ trivial product, skip
            union = tuple(sorted(si | sj))
            if len(union) > 3 or len(union) <= max(len(sc_i), len(sc_j)):
                continue                                       # too wide, or no new coupling
            pu = {v: k for k, v in enumerate(union)}
            pi = [pu[v] for v in sc_i]; pj = [pu[v] for v in sc_j]
            allowed = frozenset(
                t for t in it.product((0, 1), repeat=len(union))
                if tuple(t[p] for p in pi) in al_i and tuple(t[p] for p in pj) in al_j)
            extra.append((union, allowed))
            if len(cons) + len(extra) >= m_max:
                return C.CSP(sat.n, 2, tuple(cons + extra))
    return C.CSP(sat.n, 2, tuple(cons + extra))


class ProblemInterreducer:
    """Builds the K reduced VIEWS of a source SAT instance and solves each via the bank/composer (or the
    Ising optimiser), decoding every view back to the SOURCE-variable cell space [n_src, K]. K is the
    swept knob (k1 / kfew / kall). The gadget (ising) view is capped at gadget_n_max, dropped per-
    instance when the blow-up doesn't fit (not excluded a priori)."""

    def __init__(self, composer: BankComposerOrgan, k_mode="kall", K=8, theta=0.5,
                 gadget_n_max=27, ising_restarts=24, ising_steps=80, dev="cpu"):
        self.composer = composer
        self.k_mode = k_mode
        self.views = K_MODES[k_mode]
        self.V = len(self.views)
        self.K = K
        self.theta = theta
        self.gadget_n_max = gadget_n_max
        self.ising_restarts = ising_restarts
        self.ising_steps = ising_steps
        self.dev = dev

    # ---- per-view solve → [n_src, K] survival over SOURCE variables -------------------------------
    def _solve_ext(self, sat, alpha_dom):
        csp = R.sat_to_csp(sat).csp
        composed = self.composer.compose_one(csp, alpha_dom, system=None, tags=())
        return self._dom_to_surv(composed.dom, sat.n), True

    def _solve_alg(self, sat, alpha_dom):
        if sat.kind == "xorsat":
            csp = C.CSP(sat.n, 2, ())                          # cells only; the gf2 system does the work
            composed = self.composer.compose_one(csp, None, system=_gf2_system(sat), tags=("xor",))
            return self._dom_to_surv(composed.dom, sat.n), True
        # cnf: the joint-factor re-encoding (sound, identity)
        composed = self.composer.compose_one(_joint_factor_csp(sat), None, system=None, tags=())
        return self._dom_to_surv(composed.dom, sat.n), True

    def _solve_ising(self, sat):
        """Gadget view: sat → MIS (Lucas clause-triangle) → Ising → mean-field → decode the selected
        literals back to a SOURCE assignment (singleton where a selected node asserts a var). APPROX:
        the decoded value can be wrong (low trust). Capped: only when the gadget fits gadget_n_max."""
        if sat.kind == "xorsat":
            return None, False                                 # no clean parity→MIS gadget; drop slot
        g, node_lits = R.sat3_to_mis_gadget(sat)
        if g.n == 0 or g.n > self.gadget_n_max:                # CAP — dropped per-instance, not excluded
            return None, False
        from .. import ising_organ as IS
        G = R._nx_graph(g)
        J, h, const, meta = IS.encode_mis(G, B=2.0)
        s, _E, _ = IS.mean_field_anneal(torch.as_tensor(J), torch.as_tensor(h),
                                        restarts=self.ising_restarts, steps=self.ising_steps, seed=0)
        sel, indep, _size = IS.decode_mis(G, meta, s)
        surv = np.zeros((sat.n, self.K), dtype=np.float32)
        asserted = {}                                          # var -> bit asserted by a selected node
        for node in sel:
            v, neg = node_lits[node]
            asserted[v] = 0 if neg else 1
        for v in range(sat.n):
            if v in asserted:
                surv[v, asserted[v]] = 1.0                     # singleton (approximate)
            else:
                surv[v, 0] = surv[v, 1] = 1.0                  # unconstrained by the selection → full
        return surv, bool(indep)

    def _dom_to_surv(self, dom, n_src):
        surv = np.zeros((n_src, self.K), dtype=np.float32)
        for i in range(n_src):
            for v in dom[i]:
                if v < self.K:
                    surv[i, v] = 1.0
        return surv

    def solve_record(self, sat, alpha_dom):
        """Return (views_surv [V, n_src, K], present [V], trust [V]) for one record's source SAT."""
        n = sat.n
        vs = np.zeros((self.V, n, self.K), np.float32)
        present = np.zeros(self.V, np.float32)
        trust = np.zeros(self.V, np.float32)
        for vi, name in enumerate(self.views):
            if name == "ext":
                surv, ok = self._solve_ext(sat, alpha_dom)
            elif name == "alg":
                surv, ok = self._solve_alg(sat, alpha_dom)
            else:
                surv, ok = self._solve_ising(sat)
            if ok and surv is not None:
                vs[vi] = surv; present[vi] = 1.0; trust[vi] = TRUST[name]
        return vs, present, trust

    @torch.no_grad()
    def __call__(self, sats, alpha_b0, vmask, Nmax):
        """sats: list of B source Sat (or None). alpha_b0 [B,N,K] α's raw compile (for the ext α-gate),
        vmask [B,N]. Returns mv_surv [B,Nmax,V,K], view_mask [B,V], trust [B,V] (torch, on alpha_b0 dev)."""
        B = len(sats)
        alive = (torch.sigmoid(alpha_b0) >= self.theta).float() * vmask.unsqueeze(-1)
        alive_np = alive.cpu().numpy()
        mv = np.zeros((B, Nmax, self.V, self.K), np.float32)
        vmask_out = np.zeros((B, self.V), np.float32)
        trust_out = np.zeros((B, self.V), np.float32)
        for b, sat in enumerate(sats):
            if sat is None:
                continue
            n = sat.n
            adom = tuple(frozenset(v for v in range(2) if alive_np[b, i, v] > 0.5) for i in range(n))
            vs, present, trust = self.solve_record(sat, adom)
            mv[b, :n] = np.transpose(vs, (1, 0, 2))            # [n,V,K]
            vmask_out[b] = present; trust_out[b] = trust
        dev = alpha_b0.device
        return (torch.from_numpy(mv).to(dev), torch.from_numpy(vmask_out).to(dev),
                torch.from_numpy(trust_out).to(dev))


# ============================================================ the multi-view γ (LM allocates bandwidth)
class MultiViewGamma(nn.Module):
    """Reads the K-view bank into a single residual delta. A SHARED per-view encoder embeds each view's
    rich feature [set, |set|/d, reliability·trust]; an ATTENTION pool (query = host hidden at the cell)
    lets the LM ALLOCATE bandwidth across views (attend the informative, down-weight redundant/low-
    trust) instead of being forced to fuse all equally. Zero-init tanh gate ⇒ bitwise no-op @ init,
    regardless of K (so k1/kfew/kall share the no-op guarantee + identical param budget at fixed H)."""

    def __init__(self, D, K, hidden=256):
        super().__init__()
        self.K = K
        self.Fin = K + 2
        self.enc = nn.Sequential(nn.Linear(self.Fin, hidden), nn.GELU())
        self.q = nn.Linear(D, hidden)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, D))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.hidden = hidden

    def _rich(self, mv_surv, dvec, trust):
        """[B,N,V,K] set + dvec[B] domain size + trust[B,V] → [B,N,V,K+2] rich feature."""
        B, N, V, K = mv_surv.shape
        card = mv_surv.sum(-1)                                              # [B,N,V]
        d = (dvec.to(mv_surv.device).float().clamp_min(1.0).view(B, 1, 1)
             if dvec is not None else torch.full((B, 1, 1), float(K), device=mv_surv.device))
        feat = mv_surv.new_zeros(B, N, V, K + 2)
        feat[..., :K] = mv_surv
        feat[..., K] = (card / d).clamp(0.0, 1.0)
        rel = (1.0 - (card - 1).clamp_min(0.0) / (d - 1).clamp_min(1.0)).clamp(0.0, 1.0)
        feat[..., K + 1] = rel * trust.view(B, 1, V)                       # low-trust views down-flagged
        return feat

    def delta(self, mv_surv, view_mask, trust, dvec, cell_h, mention):
        """mv_surv [B,N,V,K], view_mask [B,V], trust [B,V], cell_h [B,N,D], mention [B,N,T] → [B,T,D]."""
        feat = self._rich(mv_surv, dvec, trust)                            # [B,N,V,K+2]
        enc = self.enc(feat)                                               # [B,N,V,H]
        q = self.q(cell_h)                                                 # [B,N,H]
        scores = (enc * q.unsqueeze(2)).sum(-1) / math.sqrt(self.hidden)   # [B,N,V]
        scores = scores.masked_fill(view_mask.unsqueeze(1) < 0.5, -1e9)
        w = torch.softmax(scores, dim=-1)                                  # LM allocates across views
        w = torch.nan_to_num(w)                                            # all-masked row → 0 (no view)
        pooled = (w.unsqueeze(-1) * enc).sum(2)                            # [B,N,H]
        g = self.out(pooled)                                               # [B,N,D]
        return torch.einsum("bnt,bnd->btd", mention, g), w


# ============================================================ the interreducer-woven model
class InterreducerWoven(nn.Module):
    """The live woven GLaDOS with the PROBLEM INTERREDUCER (K views) in the residual path. Mirrors
    bank_woven.BankWoven but presents K reduced views through MultiViewGamma. Trainable: LoRA + α + γ;
    the interreducer's solvers are the frozen/certified bank + Ising optimiser. α is grounded by the
    direct J0 dominate-dedₚ on the ext-view compile (the SATNet grounding fix), exactly as bank_woven."""

    def __init__(self, peft_model, D, K, alpha, interreducer: ProblemInterreducer, mid_layer,
                 inject_layer, gamma_hidden=256, theta=0.5):
        super().__init__()
        from ..oracle_readout import _decoder_layers
        self.model = peft_model
        self.alpha = alpha
        self.ir = interreducer
        self.gamma = MultiViewGamma(D, K, gamma_hidden)
        self.D, self.K, self.V = D, K, interreducer.V
        self.mid_layer, self.inject_layer = mid_layer, inject_layer
        self.theta = theta
        self._mention = self._attn = self._sats = self._dvec = None
        self._inject = self._capture = False
        self._override = self._override_vm = self._override_tr = None
        self._h_mid = None
        self._captured = self._captured_vm = self._captured_tr = None
        self._last_b0 = self._last_vmask = self._last_w = None
        layers = _decoder_layers(peft_model)
        self._mid_handle = layers[mid_layer].register_forward_hook(self._mid_hook)
        self._inj_handle = layers[inject_layer].register_forward_hook(self._inj_hook)

    def _mid_hook(self, module, args, output):
        self._h_mid = output[0] if isinstance(output, tuple) else output
        return output

    def _compile_b0(self):
        h = self._h_mid.float()
        m = self._mention.to(h.device)
        denom = m.sum(-1, keepdim=True).clamp_min(1e-6)
        v_mean = torch.einsum("bnt,btd->bnd", m, h) / denom
        b0, _ = self.alpha(v_mean, h, self._attn.to(h.device))
        vmask = (m.sum(-1) > 0.5).float()
        self._last_b0, self._last_vmask = b0, vmask
        return b0, vmask

    def _inj_hook(self, module, args, output):
        if not (self._inject or self._capture):
            return output
        hs = output[0] if isinstance(output, tuple) else output
        mv, vm, tr = self._override, self._override_vm, self._override_tr
        if mv is None:
            b0, vmask = self._compile_b0()
            Nmax = self._mention.shape[1]
            mv, vm, tr = self.ir(self._sats, b0, vmask, Nmax)
            mv = mv * vmask.unsqueeze(-1).unsqueeze(-1)
            self._captured, self._captured_vm, self._captured_tr = mv.detach(), vm.detach(), tr.detach()
            mv, vm, tr = self._captured, self._captured_vm, self._captured_tr
        if not self._inject:
            return output
        m = self._mention.to(hs.device)
        denom = m.sum(-1, keepdim=True).clamp_min(1e-6)
        cell_h = torch.einsum("bnt,btd->bnd", m, hs.float()) / denom
        delta, w = self.gamma.delta(mv.to(hs.device), vm.to(hs.device), tr.to(hs.device),
                                    self._dvec, cell_h, m)
        self._last_w = w.detach()
        hs = hs + (torch.tanh(self.gamma.alpha) * delta).to(hs.dtype)
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

    @contextlib.contextmanager
    def live(self, mention, attn, *, inject, capture=False, override=None, override_vm=None,
             override_tr=None, sats=None, dvec=None):
        old = (self._mention, self._attn, self._inject, self._capture, self._override, self._override_vm,
               self._override_tr, self._sats, self._dvec)
        self._mention, self._attn = mention, attn
        self._inject, self._capture = inject, capture
        self._override, self._override_vm, self._override_tr = override, override_vm, override_tr
        self._sats, self._dvec = sats, dvec
        try:
            yield
        finally:
            (self._mention, self._attn, self._inject, self._capture, self._override, self._override_vm,
             self._override_tr, self._sats, self._dvec) = old

    def logits(self, input_ids, attn):
        return self.model(input_ids=input_ids, attention_mask=attn).logits

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def remove_hooks(self):
        """Detach both decoder-layer forward hooks (mid + inject) so repeated woven builds over a
        long run don't leak handles. Safe to call more than once."""
        for attr in ("_mid_handle", "_inj_handle"):
            h = getattr(self, attr, None)
            if h is not None:
                h.remove()
                setattr(self, attr, None)

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass


# ============================================================ batching + no-op@init proof
def _ir_batch(recs, tok, dev, two_stream=True):
    from .. import run_glados_staged as G
    ba = G.build_live_batch(recs, tok, dev, two_stream=two_stream)
    ba["sats"] = [r.get("source_sat") for r in recs]
    ba["dvec"] = torch.tensor([len(r["vnames"]) for r in recs], device=dev)
    return ba


def verify_noop_ir(model, tok, dev, rec):
    """Bitwise no-op @ init: the zero-init γ gate ⇒ the K-view injection moves the logits by 0."""
    enc = tok(rec["prompt"], return_offsets_mapping=True, return_tensors="pt")
    ids = enc["input_ids"].to(dev); attn = torch.ones_like(ids)
    mention = _mention_tensor([rec["mentions"]], enc["offset_mapping"], 1, rec["n"], ids.size(1), dev)
    sats = [rec.get("source_sat")]
    dvec = torch.tensor([len(rec["vnames"])], device=dev)
    with torch.no_grad():
        base = model.model(input_ids=ids, attention_mask=attn).logits.float()
    with torch.no_grad(), model.live(mention, attn, inject=True, capture=True, sats=sats, dvec=dvec):
        g0 = model.logits(ids, attn).float()
    noop = float((base - g0).abs().max())
    with torch.no_grad():
        saved = model.gamma.alpha.data.clone(); model.gamma.alpha.data.fill_(2.0)
        with model.live(mention, attn, inject=True, capture=True, sats=sats, dvec=dvec):
            g1 = model.logits(ids, attn).float()
        model.gamma.alpha.data.copy_(saved)
    return noop, float((base - g1).abs().max())


# ============================================================ causal-control transforms (multi-view)
def apply_control_mv(mv, recs, control, K, perm):
    """Transform the K-view bank [B,N,V,K] for a causal control (applied across ALL views)."""
    s = mv.clone()
    if control == "true":
        return s
    if control == "shuffle":
        return torch.roll(s, shifts=1, dims=0)                 # inject a DIFFERENT problem's K-view bank
    if control == "permute":
        idx = torch.tensor(perm[:K], device=s.device)
        return s.index_select(-1, idx)
    if control == "corrupt":
        for b, r in enumerate(recs):
            q = r["query"]
            for vi in range(s.shape[2]):
                row = s[b, q, vi]
                nz = torch.nonzero(row > 0.5).flatten()
                first = int(nz[0]) if nz.numel() else 0
                s[b, q, vi].zero_(); s[b, q, vi, (first + 1) % K] = 1.0
        return s
    raise ValueError(control)


# ============================================================ generative scoring + controls
@torch.no_grad()
def score_interreducer(model, recs, tok, dev, control="true", bs=8, perm=None, use_base=False,
                       fewshot="", two_stream=True):
    """GENERATIVE accuracy for the interreducer path: one capture forward builds the K-view bank from α
    + interreducer; a second forward scores the legal answers with that (optionally controlled) bank
    injected by the multi-view γ. Mirrors bank_woven.score_bank_woven."""
    from .. import run_glados_staged as G
    model.eval()
    K = model.K
    if perm is None:
        perm = tuple(list(range(1, K)) + [0])
    inject = control != "zero"
    correct = tot = det_t = det_r = 0
    by_split = {}
    vsum = vcount = 0.0
    for i in range(0, len(recs), bs):
        chunk = recs[i:i + bs]; Bp = len(chunk); Nmax = max(r["n"] for r in chunk)
        dvec = torch.tensor([len(r["vnames"]) for r in chunk], device=dev)
        mv = vm = tr = None
        if inject:
            cap_p = [r.get("alpha_prompt", r["prompt"]) for r in chunk] if two_stream else \
                [r["prompt"] for r in chunk]
            cap_m = [r.get("alpha_mentions", r["mentions"]) for r in chunk] if two_stream else \
                [r["mentions"] for r in chunk]
            penc = tok(cap_p, return_offsets_mapping=True, padding=True, return_tensors="pt")
            pids = penc["input_ids"].to(dev); pattn = penc["attention_mask"].to(dev)
            pment = _mention_tensor(cap_m, penc["offset_mapping"], Bp, Nmax, pids.size(1), dev)
            sats = [r.get("source_sat") for r in chunk]
            with model.live(pment, pattn, inject=False, capture=True, sats=sats, dvec=dvec):
                _ = model.logits(pids, pattn)
            mv = model._captured[:, :Nmax].clone()
            vm = model._captured_vm.clone(); tr = model._captured_tr.clone()
            vsum += float(vm.sum()); vcount += Bp
            mv = apply_control_mv(mv, chunk, control, K, perm)
        fulls, plens, spans, prob_of = [], [], [], []
        shift = len(fewshot)
        for pi, r in enumerate(chunk):
            cands = [" " + v for v in r["vnames"]] + [" " + ABSTAIN_STR]
            for cand in cands:
                fulls.append(fewshot + r["prompt"] + cand); plens.append(shift + len(r["prompt"]))
                spans.append({k: [(x + shift, y + shift) for (x, y) in v] for k, v in r["mentions"].items()})
                prob_of.append(pi)
        enc = tok(fulls, return_offsets_mapping=True, padding=True, return_tensors="pt")
        ids = enc["input_ids"].to(dev); attn = enc["attention_mask"].to(dev)
        offsets = enc["offset_mapping"]; T = ids.size(1)
        if inject:
            mention = _mention_tensor(spans, offsets, len(fulls), Nmax, T, dev)
            idx = torch.tensor(prob_of, device=dev)
            ctx = model.live(mention, attn, inject=True, capture=False, override=mv[idx],
                             override_vm=vm[idx], override_tr=tr[idx], dvec=dvec[idx])
        else:
            ctx = model.live(None, None, inject=False)
        base_ctx = model.model.disable_adapter() if (use_base and hasattr(model.model, "disable_adapter")) \
            else contextlib.nullcontext()
        with base_ctx, ctx:
            logits = model.logits(ids, attn).float()
        lp = torch.log_softmax(logits[:, :-1], -1)
        tok_lp = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        scores = torch.full((len(fulls),), -1e9, device=dev)
        for row in range(len(fulls)):
            offs = offsets[row].tolist(); mask = torch.zeros(T - 1, device=dev)
            for ti in range(1, T):
                lo, hi = offs[ti]
                if lo != hi and lo >= plens[row] and attn[row, ti] > 0.5:
                    mask[ti - 1] = 1.0
            denom = mask.sum().clamp_min(1.0)
            scores[row] = (tok_lp[row] * mask).sum() / denom
        prob_of_t = torch.tensor(prob_of, device=dev)
        for pi, r in enumerate(chunk):
            rows = (prob_of_t == pi).nonzero().flatten()
            pred = int(rows[int(scores[rows].argmax())] - rows[0])
            ok = int(pred == r["gold_idx"]); correct += ok; tot += 1
            sp = r.get("split", "?"); d = by_split.setdefault(sp, [0, 0]); d[0] += ok; d[1] += 1
            if r["determined"]:
                det_t += 1; det_r += ok
    return {"overall": correct / max(1, tot), "det_acc": det_r / max(1, det_t), "n": tot,
            "mean_views": vsum / max(1.0, vcount),
            "by_split": {k: v[0] / max(1, v[1]) for k, v in by_split.items()}}


@torch.no_grad()
def ir_controls(model, recs, tok, dev, bs, two_stream=True):
    sc = lambda **kw: score_interreducer(model, recs, tok, dev, bs=bs, two_stream=two_stream, **kw)
    full = sc(control="true")
    out = {"true": full["overall"], "det_acc": full["det_acc"], "mean_views": full["mean_views"],
           "n": full["n"]}
    out["zero"] = sc(control="zero")["overall"]
    out["shuffle"] = sc(control="shuffle")["overall"]
    out["corrupt"] = sc(control="corrupt")["overall"]
    from .. import run_glados_staged as G
    out["base"] = sc(control="zero", use_base=True, fewshot=G.FEWSHOT)["overall"]
    out["gate"] = float(torch.tanh(model.gamma.alpha))
    out["lift"] = out["true"] - out["zero"]
    out["drop"] = out["true"] - min(out["shuffle"], out["corrupt"])
    return out


# ============================================================ training the interreducer-woven readout
def build_interreducer_model(olmo_ids, tok, dev, a, k_mode, composer):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from .. import run_glados_staged as G
    mid_id, D, nL = olmo_ids
    olmo = AutoModelForCausalLM.from_pretrained(mid_id, dtype=torch.bfloat16).to(dev).eval()
    for p in olmo.parameters():
        p.requires_grad_(False)
    lconf = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, bias="none",
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
    peft_model = get_peft_model(olmo, lconf)
    mid = min(a.mid_layer, nL - 1); inj = min(a.inject_layer, nL - 1)
    assert mid < inj
    alpha = DenseLatentProjector(D, G.K, dctx=a.dctx, dp=a.alpha_dp, heads=a.alpha_heads)
    ir = ProblemInterreducer(composer, k_mode=k_mode, K=G.K, gadget_n_max=getattr(a, "gadget_n_max", 27))
    model = InterreducerWoven(peft_model, D, G.K, alpha, ir, mid, inj, gamma_hidden=a.gamma_hidden).to(dev)
    model.alpha.float(); model.gamma.float()
    return model


def train_interreducer(model, tok, dev, a, train_recs, eval_recs, two_stream=True):
    """Train the interreducer-woven readout. Phase A: warm α (+LoRA) on the direct J0 dominate-dedₚ
    (the ext-view target). Phase B: LoRA + α + multi-view γ on the answer-span LM CE (γ reads the
    DETACHED K-view bank) + standing J0. Identical recipe to bank_woven; only γ is multi-view."""
    from .. import latent_tasks as LT
    noop, live = verify_noop_ir(model, tok, dev, eval_recs[0])
    from ..oracle_readout import n_trainable
    print(f"\n  INTERREDUCER  K-mode={model.ir.k_mode} views={model.ir.views} V={model.V}  "
          f"trainable {n_trainable(model):,}", flush=True)
    print(f"  NO-OP @ INIT max|base-(LoRA+gate0)| = {noop:.3e} (expect ~0) | gate-on moves {live:.3e}",
          flush=True)
    lora_params = [p for n, p in model.model.named_parameters() if p.requires_grad]
    rng = np.random.default_rng(a.seed + 3)

    def batch(n):
        idxs = rng.integers(0, len(train_recs), n).tolist()
        return _ir_batch([train_recs[i] for i in idxs], tok, dev, two_stream)

    def alpha_capture(ba):
        mm = ("a_mention", "a_attn", "a_input_ids") if two_stream else ("mention", "attn", "input_ids")
        with model.live(ba[mm[0]], ba[mm[1]], inject=False, capture=True, sats=ba["sats"], dvec=ba["dvec"]):
            _ = model.logits(ba[mm[2]], ba[mm[1]])

    optA = torch.optim.AdamW(
        [{"params": lora_params, "lr": a.lora_lr, "weight_decay": 0.01},
         {"params": list(model.alpha.parameters()), "lr": a.alpha_lr, "weight_decay": 0.0}],
        betas=(0.9, 0.95))
    model.train(); t0 = time.time()
    for s in range(1, a.warm_steps + 1):
        ba = batch(a.bs)
        alpha_capture(ba)
        loss = dominate_dedp_loss([model._last_b0], ba["tgt"], ba["vmask"])
        optA.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0); optA.step()
        if s % max(1, a.warm_steps // 5) == 0 or s == 1:
            an, ad, _, _ = LT.narrowing_stats(model._last_b0, ba["tgt"], ba["vmask"])
            print(f"  [warm α/J0] step {s:5d}  dedp {loss.item():.3f}  α-recall {an/max(1,ad):.3f}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    optB = torch.optim.AdamW(
        [{"params": lora_params, "lr": a.lora_lr, "weight_decay": 0.01},
         {"params": list(model.alpha.parameters()), "lr": a.alpha_lr, "weight_decay": 0.0},
         {"params": list(model.gamma.parameters()), "lr": a.gamma_lr, "weight_decay": 0.0}],
        betas=(0.9, 0.95))
    model.train(); t0 = time.time(); seen_open = False
    for s in range(1, a.steps + 1):
        ba = batch(a.bs)
        if two_stream:
            alpha_capture(ba)
            aux = dominate_dedp_loss([model._last_b0], ba["tgt"], ba["vmask"])
            Nm = ba["mention"].shape[1]
            mv = model._captured[:, :Nm]; vm = model._captured_vm; tr = model._captured_tr
            with model.live(ba["mention"], ba["attn"], inject=True, capture=False, override=mv,
                            override_vm=vm, override_tr=tr, dvec=ba["dvec"]):
                logits = model.logits(ba["input_ids"], ba["attn"]).float()
        else:
            with model.live(ba["mention"], ba["attn"], inject=True, capture=True, sats=ba["sats"],
                            dvec=ba["dvec"]):
                logits = model.logits(ba["input_ids"], ba["attn"]).float()
            aux = dominate_dedp_loss([model._last_b0], ba["tgt"], ba["vmask"])
        lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                             ba["labels"][:, 1:].reshape(-1), ignore_index=-100)
        loss = lm + a.alpha_sup_w * aux
        optB.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0); optB.step()
        gate = float(torch.tanh(model.gamma.alpha)); seen_open = seen_open or abs(gate) > 1e-3
        if s % max(1, a.steps // 10) == 0 or s == 1:
            acc = score_interreducer(model, eval_recs, tok, dev, control="true", bs=a.bs,
                                     two_stream=two_stream)
            print(f"  step {s:5d}  lm {lm.item():.3f}  J0 {aux.item():.3f}  gate {gate:+.3f}  "
                  f"acc {acc['overall']*100:4.1f}% (det {acc['det_acc']*100:.0f}, V̄ {acc['mean_views']:.2f})  "
                  f"{time.time()-t0:.0f}s", flush=True)
            model.train()
    if not seen_open:
        print("  [WARN] γ gate never opened.", flush=True)
    return model


# ============================================================ the K-SWEEP driver (the experiment)
def run_sweep(base="allenai/OLMo-2-0425-1B", steps=1500, warm=300, bs=8, per_cell=120, seed=0,
              kmodes=("k1", "kfew", "kall"), out="runs/interreducer_sweep.json"):
    from transformers import AutoTokenizer, AutoConfig
    from .. import run_glados_staged as G
    from . import train as T
    from ..datagen import reductions as RC
    dev = G.device()
    torch.manual_seed(seed); np.random.seed(seed)
    print(f"==== INTERREDUCER K-SWEEP  base={base}  dev={dev}  steps={steps} warm={warm} ====", flush=True)

    splits, crep = RC.build_curriculum(seed=seed, per_cell=per_cell, verbose=True)
    rng = np.random.default_rng(seed); tr = splits["train"]; rng.shuffle(tr)
    cut = int(0.85 * len(tr))
    train_recs = tr[:cut]
    eval_direct = tr[cut:]
    for r in eval_direct:
        r["split"] = "in_dist"
    buckets = {"in_dist_DIRECT": eval_direct, "route_required": splits["route_required"],
               "heldout_reduction_path": splits["heldout_reduction_path"]}
    print(f"  buckets: " + "  ".join(f"{k}={len(v)}" for k, v in buckets.items()), flush=True)

    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(base)
    D = cfg.hidden_size
    nL = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "num_layers", None)
    a = T._weave_args(base, steps=steps, warm_steps=warm, bs=bs, seed=seed)
    a.inject_layer = min(a.inject_layer, nL - 1); a.mid_layer = min(a.mid_layer, a.inject_layer - 1)
    core = "runs/general_organ_full.pt"
    composer = BankComposerOrgan(dev="cpu", core_ckpt=core, use_core=os.path.exists(core))
    print(f"  composer faculties = {composer.faculties()}", flush=True)

    results = {"curriculum": {k: v for k, v in crep.items() if k != "train_representations"},
               "sweep": {}}
    for km in kmodes:
        print(f"\n======== K-mode {km} ({K_MODES[km]}) ========", flush=True)
        t0 = time.time()
        model = build_interreducer_model((base, D, nL), tok, dev, a, km, composer)
        train_interreducer(model, tok, dev, a, train_recs, eval_direct, two_stream=True)
        train_time = time.time() - t0
        km_res = {"views": K_MODES[km], "train_time_s": round(train_time, 1)}
        for bname, recs in buckets.items():
            if not recs:
                continue
            m = ir_controls(model, recs, tok, dev, a.bs, two_stream=True)
            km_res[bname] = m
            print(f"  [{bname:22s}] true {m['true']*100:5.1f}%  zero {m['zero']*100:5.1f}%  "
                  f"base {m['base']*100:5.1f}%  shuf {m['shuffle']*100:5.1f}%  lift {m['lift']*100:+.1f}  "
                  f"drop {m['drop']*100:+.1f}  gate {m['gate']:+.2f}  V̄ {m['mean_views']:.2f}", flush=True)
        results["sweep"][km] = km_res
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"  [{km}] done in {train_time:.0f}s; wrote {out}", flush=True)

    _print_verdict(results)
    return results


def _print_verdict(results):
    sw = results["sweep"]
    print("\n==================== K-SWEEP TABLE (woven-true %) ====================", flush=True)
    buckets = ["in_dist_DIRECT", "route_required", "heldout_reduction_path"]
    print(f"  {'K-mode':8s} {'V̄(direct)':10s} " + " ".join(f"{b[:18]:>18s}" for b in buckets), flush=True)
    for km in sw:
        row = sw[km]
        vbar = row.get("in_dist_DIRECT", {}).get("mean_views", float('nan'))
        cells = []
        for b in buckets:
            v = row.get(b)
            cells.append(f"{v['true']*100:17.1f}%" if v else f"{'--':>18s}")
        print(f"  {km:8s} {vbar:10.2f} " + " ".join(cells), flush=True)
    print("=====================================================================", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--warm", type=int, default=300)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--per_cell", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kmodes", default="k1,kfew,kall")
    ap.add_argument("--out", default="runs/interreducer_sweep.json")
    a = ap.parse_args()
    run_sweep(base=a.base, steps=a.steps, warm=a.warm, bs=a.bs, per_cell=a.per_cell, seed=a.seed,
              kmodes=tuple(a.kmodes.split(",")), out=a.out)


if __name__ == "__main__":
    main()
