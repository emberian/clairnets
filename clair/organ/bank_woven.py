"""clair/organ/bank_woven.py — THE consolidated woven organ: bank + composer in the LIVE path.

This is the fix for GLaDOS blockers 1/2/3 (the "single-faculty standalone narrower" problem). The
prior live woven (`clair.oracle_readout.LiveLatentWoven`) wired α → a *standalone, frozen*
`LatentNarrower` → γ: one neural faculty, disconnected from the bank, the composer, the certified
ops, and the pretrained organ. Here the live organ IS the real consolidated organ:

  α (DenseLatentProjector) compiles the host hidden → a per-cell candidate lattice (b0 logits);
  the COMPOSER (`compose.reduced_product`) runs, on the problem's true constraint structure:
    * the certified ops (Arc/Factor/Modular/GF2/Macro — the SOUNDNESS FLOOR, from the full domain),
    * the pretrained neural narrower (`bank.CoreNarrowOrgan`, loads `runs/general_organ_full.pt`),
      verifier-gated so a bad neural proposal can never poison the shared state,
    * α's own compiled lattice, as a third verifier-gated neural proposal (so α is genuinely IN the
      multi-faculty composition, not a bypass);
  γ (OracleGamma, zero-init gate ⇒ bitwise no-op at init) reads the COMPOSED lattice into the host
  residual stream → the LM head generates the answer.

Result, vs the standalone narrower:
  (1) USES THE PRETRAINED ORGAN — the union-trained `general_organ_full.pt` deductor is the neural
      faculty (resolves the FactorGraphProposer-vs-LatentNarrower mismatch: α compiles only the
      per-cell *lattice*; the *constraint structure* the bank/CoreNarrowOrgan consume comes from the
      problem instance, threaded into the records, and is verifier-gated);
  (2) MULTI-FACULTY — the whole bank is available; the composer routes by each reduction's
      `applies()` (Modular on LinSystems, GF2 on XOR, Macro on path-CSPs, …);
  (3) CERTIFIED-FLOOR-BACKED — the injected lattice is sound by construction RELATIVE TO THE CSP IT
      COMPOSES ON (the certified ops run from the full domain and the meet of sound narrowings is sound;
      the neural proposals only ever sharpen toward the exact per-cell transformer, never below it). Not
      pure-neural. This is (A)/(B) soundness (notes/soundness.md), NOT answer-soundness: when that CSP is
      α's emitted csp_α, the lattice is sound only MODULO the unverified compile — a wrong csp_α gives a
      certified-correct answer to the wrong problem (the SHUFFLED gap, (C)). Answer-soundness needs the
      output check, which needs ground truth (train/eval) or a domain verifier (code/proof) — (D).

α is trained by the DIRECT J0 dominate-dedₚ supervision on its raw compile b0 (the SATNet grounding
fix) — the composer is discrete/non-differentiable, so γ reads the DETACHED composed lattice while α
learns from the standing J0 term (and the LM-CE flows to LoRA + γ). This is the validated engagement
recipe (two-stream + J0 + the structured readout), now with the real organ in the residual path.
"""
from __future__ import annotations

import contextlib
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import csp as C
from ..latent_organ import DenseLatentProjector, dominate_dedp_loss
from ..oracle_readout import OracleGamma, _mention_tensor, ABSTAIN_STR
from .alpha_struct import (StructureRack, build_csp_from_struct, decode_structure,
                           facts_to_struct_targets, structure_sup_loss, output_check,
                           sample_candidates, solve_query)
from .bank import build_bank, certified_csp_reductions
from .compose import reduced_product, reduced_product_batch
from .protocol import Certificate, CSPState, Reduction

# The α↔organ FEEDBACK channel width (lever B): per cell the organ reports back to the rack
#   [narrowed-card-fraction, determined(singleton), conflict(⊥)]  — see compose_and_feedback.
FB_DIM = 3


# ============================================================ α's compile as a gated reduction
class _AlphaProposal(Reduction):
    """α's live-compiled per-cell lattice, presented to the composer as a NEURAL-GUIDANCE reduction so
    it is verifier-gated exactly like CoreNarrowOrgan: it can sharpen the composed lattice (down to the
    exact per-cell transformer) but can NEVER drop a value a real solution uses. This is how α joins the
    multi-faculty composition without ever being able to poison the certified floor."""
    name = "alpha_compile"
    domain = "latent α compile (host hidden → per-cell lattice)"
    state_type = "csp-domain"

    def __init__(self, proposed_dom):
        self._p = proposed_dom                       # tuple[frozenset] over the instance's cells

    def applies(self, state):
        return isinstance(state, CSPState) and len(self._p) == state.csp.n

    def reduce(self, state: CSPState) -> CSPState:
        return state.with_dom(tuple(self._p[i] & state.dom[i] for i in range(state.csp.n)))

    def certificate(self):
        return Certificate(False, "neural-guidance",
                           "latent α compile from host hidden; verifier-gated (cannot poison the floor)")


# ============================================================ the bank+composer organ
class BankComposerOrgan:
    """The live organ slot: given α's compiled per-cell lattice + the instance's true CSP structure,
    run the verifier-gated reduced product (certified floor + pretrained CoreNarrowOrgan + α, all gated)
    and return the composed per-cell survival matrix γ will read. Discrete / non-differentiable: the
    lattice it returns is DETACHED (γ + LoRA learn to read it; α is grounded by the direct J0 term)."""

    def __init__(self, dev="cpu", core_ckpt="runs/general_organ_full.pt", use_core=True,
                 use_alpha_gate=True, max_rounds=16):
        self.bank = build_bank(load_neural=use_core, dev=dev, core_ckpt=core_ckpt)
        self.certified = certified_csp_reductions(self.bank)            # Arc/Factor/Modular/GF2/Macro
        self.core = self.bank.get("core_narrow_organ")                  # pretrained neural narrower (gated)
        self.base_reductions = list(self.certified) + ([self.core] if self.core else [])
        self.use_alpha_gate = use_alpha_gate
        self.max_rounds = max_rounds
        self.last_trace = None

    def faculties(self):
        return [r.name for r in self.base_reductions]

    def compose_one(self, csp, alpha_dom, system=None, tags=()):
        """Compose from the FULL domain on the given structure (`csp`). The certified floor is sound-by-
        construction RELATIVE TO `csp`; CoreNarrowOrgan + α are verifier-gated. Sound rel. `csp`, NOT
        answer-sound (notes/soundness.md (B)): if `csp` is α's emitted csp_α the result is sound only
        modulo that unverified compile. Returns the composed CSPState."""
        full = CSPState.full(csp, system=system, tags=frozenset(tags))
        reds = list(self.base_reductions)
        if self.use_alpha_gate and alpha_dom is not None:
            reds = reds + [_AlphaProposal(alpha_dom)]
        # verify=False: the per-round verifier gate already guarantees soundness; skip the extra
        # full-origin exact_dedP pass (cost). The certified ops still assert subset internally.
        out, tr = reduced_product(full, reds, verify=False, max_rounds=self.max_rounds)
        self.last_trace = tr
        return out

    def compose_batch(self, items):
        """BATCHED compose: `items` is a list of (csp, alpha_dom, system, tags). Runs the SAME verifier-gated
        reduced product as `compose_one` for every item, but fans the gated neural organ's forward across the
        whole batch in one call (reduced_product_batch) — the per-instance composed lattice is BITWISE-
        IDENTICAL to compose_one (asserted in selftest). Returns (list[CSPState], list[Trace])."""
        states, extras = [], []
        for csp, alpha_dom, system, tags in items:
            states.append(CSPState.full(csp, system=system, tags=frozenset(tags)))
            extras.append([_AlphaProposal(alpha_dom)] if (self.use_alpha_gate and alpha_dom is not None) else [])
        outs, traces = reduced_product_batch(states, list(self.base_reductions), extras,
                                             max_rounds=self.max_rounds)
        if traces:
            self.last_trace = traces[-1]
        return outs, traces

    @torch.no_grad()
    def __call__(self, b0, vmask, theta, csps, K):
        """b0 [B,N,K] logits, vmask [B,N], csps a list of B (csp, system, tags) tuples (or None for an
        instance with no structure → fall back to α's thresholded compile). Returns surv [B,N,K] float
        tensor (on b0.device) + a list of composer traces."""
        B, N, _ = b0.shape
        alive = (torch.sigmoid(b0) >= theta).float() * vmask.unsqueeze(-1)   # α's thresholded compile
        alive_np = alive.cpu().numpy()
        out = np.zeros((B, N, K), dtype=np.float32)
        traces = []
        for b in range(B):
            spec = csps[b] if csps is not None else None
            if spec is None:
                out[b] = alive_np[b]                      # no structure available: α-only fallback
                traces.append(None)
                continue
            csp, system, tags = spec
            n = csp.n
            # alive_np is width K (α's output); a value v >= K can never be alive, so clamp the read
            # to K to match the width-guarded write below (csp.d may exceed K).
            adom = tuple(frozenset(v for v in range(min(csp.d, K)) if alive_np[b, i, v] > 0.5)
                         for i in range(n))
            composed = self.compose_one(csp, adom, system=system, tags=tags)
            for i in range(n):
                for v in composed.dom[i]:
                    if v < K:
                        out[b, i, v] = 1.0
            traces.append(self.last_trace)
        return torch.from_numpy(out).to(b0.device), traces


# ============================================================ the bank-woven model
class BankWoven(nn.Module):
    """The live woven GLaDOS with the bank+composer in the residual path. Forward (one host pass) via
    two decoder-layer hooks: MID captures host hidden h; INJECT runs α(h) → composed lattice (bank +
    composer) → γ scatter (zero-init gate ⇒ no-op@init). Trainable: LoRA + α + γ; the organ is the
    frozen/certified bank (no trainable organ params). α is grounded by the direct J0 dominate-dedₚ."""

    def __init__(self, peft_model, D, K, alpha, composer: BankComposerOrgan, mid_layer, inject_layer,
                 gamma_hidden=256, theta=0.5, rich=True):
        super().__init__()
        from ..oracle_readout import _decoder_layers
        self.model = peft_model
        self.alpha = alpha                       # DenseLatentProjector (trainable)
        self.composer = composer                 # BankComposerOrgan (frozen / certified)
        # RICH-STATE readout (DEFAULT, the alien-consumer finding): γ reads the FULL composed candidate
        # SET per cell PLUS [|set|/d, reliability] — reliability = narrowing fraction (1.0 = singleton,
        # 0 = full domain). The LM mines the partial lattice + discounts low-reliability cells, so an
        # α-miscompile degrades gracefully instead of dragging the LM off a cliff. rich=False collapses
        # to the bare set (the ablation).
        self.rich = rich
        self.Fin = K + 2 if rich else K
        self.gamma = OracleGamma(D, self.Fin, gamma_hidden)
        self.D, self.K = D, K
        self.mid_layer, self.inject_layer = mid_layer, inject_layer
        self.theta = theta
        self._mention = self._attn = None
        self._csps = self._dvec = None
        self._inject = False
        self._capture = False
        self._override = None
        self._h_mid = None
        self._captured_surv = None
        self._last_b0 = None
        self._last_vmask = None
        layers = _decoder_layers(peft_model)
        self._mid_handle = layers[mid_layer].register_forward_hook(self._mid_hook)
        self._inj_handle = layers[inject_layer].register_forward_hook(self._inj_hook)

    def _rich_from_set(self, surv_set, dvec):
        """[B,N,K] composed candidate SET (+ per-instance domain sizes dvec[B]) → the rich feature
        [B,N,K+2] = [set(K), |set|/d, reliability]. reliability = 1-(|set|-1)/(d-1) (the alien-consumer
        rich-state encoding; reused so the bridge readouts and the narrow readout share one schema)."""
        B, N, K = surv_set.shape
        card = surv_set.sum(-1)                                       # [B,N]
        if dvec is None:
            d = torch.full((B, 1), float(K), device=surv_set.device)
        else:
            d = dvec.to(surv_set.device).float().clamp_min(1.0).view(B, 1)
        feat = surv_set.new_zeros(B, N, K + 2)
        feat[..., :K] = surv_set
        feat[..., K] = (card / d).clamp(0.0, 1.0)
        feat[..., K + 1] = (1.0 - (card - 1).clamp_min(0.0) / (d - 1).clamp_min(1.0)).clamp(0.0, 1.0)
        return feat

    # ---- hooks ----
    def _mid_hook(self, module, args, output):
        self._h_mid = output[0] if isinstance(output, tuple) else output
        return output

    def _compile_b0(self):
        """α(mid hidden) → raw per-cell compile b0 [B,N,K] + vmask [B,N] (no organ narrowing yet)."""
        h = self._h_mid.float()
        m = self._mention.to(h.device)
        denom = m.sum(-1, keepdim=True).clamp_min(1e-6)
        v_mean = torch.einsum("bnt,btd->bnd", m, h) / denom          # mention-pool cell identity
        b0, _ctx = self.alpha(v_mean, h, self._attn.to(h.device))
        vmask = (m.sum(-1) > 0.5).float()
        self._last_b0, self._last_vmask = b0, vmask
        return b0, vmask

    def _inj_hook(self, module, args, output):
        if not (self._inject or self._capture):
            return output
        hs = output[0] if isinstance(output, tuple) else output
        surv = self._override
        if surv is None:
            b0, vmask = self._compile_b0()
            surv = self.composer(b0, vmask, self.theta, self._csps, self.K)[0]   # detached composed lattice
            surv = surv * vmask.unsqueeze(-1)
            self._captured_surv = surv.detach()
            surv = self._captured_surv
        if not self._inject:
            return output
        feat = self._rich_from_set(surv, self._dvec) if self.rich else surv      # rich-state readout
        delta = self.gamma.delta(feat.to(hs.device), self._mention.to(hs.device))
        hs = hs + (torch.tanh(self.gamma.alpha) * delta).to(hs.dtype)
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

    # ---- context manager ----
    @contextlib.contextmanager
    def live(self, mention, attn, *, inject, capture=False, override=None, csps=None, dvec=None):
        old = (self._mention, self._attn, self._inject, self._capture, self._override, self._csps,
               self._dvec)
        self._mention, self._attn = mention, attn
        self._inject, self._capture, self._override, self._csps, self._dvec = (
            inject, capture, override, csps, dvec)
        try:
            yield
        finally:
            (self._mention, self._attn, self._inject, self._capture, self._override, self._csps,
             self._dvec) = old

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


# ============================================================ record CSP threading
@torch.no_grad()
def eval_real_nl(model, tok, dev, jsonl_path, rungs, per_rung=60, bs=8, seed=0, two_stream=True,
                 engage_thr=5.0):
    """BLOCKER-7 FRONTIER eval hook: measure the bank-woven model on REAL held-out NL phrasings (α must
    compile from natural language it never saw). Returns the same engagement/causal-control metrics as
    bank_controls, on the diverse curriculum. This is a GENERALIZATION STUDY of α's live compile — not a
    wiring fix; the ORACLE-override arm isolates α's NL-compile gap from the readout (which is solved)."""
    import os
    from .. import run_glados_staged as G
    if not os.path.exists(jsonl_path):
        print(f"  [eval_real_nl] no diverse curriculum at {jsonl_path}; skipping NL-frontier eval")
        return None
    rng = np.random.default_rng(seed + 300)
    recs = G.build_diverse_live_pool(rng, jsonl_path, rungs, per_rung, det_only=True, two_stream=two_stream)
    if not recs:
        return None
    print(f"  [eval_real_nl] {len(recs)} real-NL records over {rungs}", flush=True)
    return bank_controls(model, recs, tok, dev, bs, engage_thr, two_stream=two_stream)


# ============================================================ ALPHA_STRUCT no-metadata invariant
# The SHUFFLED diagnostic (runs/shuffled_diag.json) proved today's woven is 100% metadata-driven: the
# certified floor runs on rec["csp"] (the TRUE structure), so α contributes nothing. ALPHA_STRUCT's
# structural fix is that the composer runs on α's EMITTED structure (csp_α) and rec["csp"] survives ONLY
# as the eval-time output-checker. This tripwire makes that a CODE-LEVEL invariant (design §4.1 crit 5):
# csp_spec_for_record (the only reader of rec["csp"]) RAISES if called inside the ALPHA_STRUCT forward.
_FORBID_RECORD_CSP_READ = False


@contextlib.contextmanager
def forbid_record_csp():
    """Inside this context any read of rec['csp'] (via csp_spec_for_record) raises — the structural guard
    the SHUFFLED diagnostic's metadata-leak motivates. The ALPHA_STRUCT forward runs under it; the
    eval-time output_check + the structure-supervision target builder run OUTSIDE it (licensed readers)."""
    global _FORBID_RECORD_CSP_READ
    prev = _FORBID_RECORD_CSP_READ
    _FORBID_RECORD_CSP_READ = True
    try:
        yield
    finally:
        _FORBID_RECORD_CSP_READ = prev


def csp_spec_for_record(rec):
    """(csp, system, tags) for the LEGACY bank composer, from a live record. build_live_pool stashes
    rec['csp'] (the organ_csp) and optional rec['sys']/rec['tags']. None if the record carries no
    structure. RAISES under forbid_record_csp() — rec['csp'] must never enter the ALPHA_STRUCT forward."""
    if _FORBID_RECORD_CSP_READ:
        raise RuntimeError(
            "METADATA LEAK: rec['csp'] read inside the ALPHA_STRUCT composer forward. The composer must "
            "run on α's EMITTED structure (csp_α = build_csp_from_struct(α_facts)); rec['csp'] is the "
            "eval-time output-checker ONLY (notes/alpha_struct_design.md §3-4).")
    csp = rec.get("csp")
    if csp is None:
        return None
    return (csp, rec.get("sys"), tuple(rec.get("tags", ())))


def _bank_batch(recs, tok, dev, two_stream):
    from .. import run_glados_staged as G
    ba = G.build_live_batch(recs, tok, dev, two_stream=two_stream)
    ba["csps"] = [csp_spec_for_record(r) for r in recs]
    ba["dvec"] = torch.tensor([len(r["vnames"]) for r in recs], device=dev)   # domain size per instance
    return ba


# ============================================================ no-op@init proof
def verify_noop_bank(model, tok, dev, rec):
    """Bitwise no-op at init: the zero-init γ gate ⇒ the live bank-composed injection moves logits by 0."""
    enc = tok(rec["prompt"], return_offsets_mapping=True, return_tensors="pt")
    ids = enc["input_ids"].to(dev); attn = torch.ones_like(ids)
    mention = _mention_tensor([rec["mentions"]], enc["offset_mapping"], 1, rec["n"], ids.size(1), dev)
    csps = [csp_spec_for_record(rec)]
    dvec = torch.tensor([len(rec["vnames"])], device=dev)
    with torch.no_grad():
        base = model.model(input_ids=ids, attention_mask=attn).logits.float()
    with torch.no_grad(), model.live(mention, attn, inject=True, capture=True, csps=csps, dvec=dvec):
        g0 = model.logits(ids, attn).float()
    noop = float((base - g0).abs().max())
    with torch.no_grad():
        saved = model.gamma.alpha.data.clone(); model.gamma.alpha.data.fill_(2.0)
        with model.live(mention, attn, inject=True, capture=True, csps=csps, dvec=dvec):
            g1 = model.logits(ids, attn).float()
        model.gamma.alpha.data.copy_(saved)
    return noop, float((base - g1).abs().max())


# ============================================================ generative scoring + causal controls
@torch.no_grad()
def score_bank_woven(model, recs, tok, dev, control="true", bs=8, perm=None, use_base=False,
                     fewshot="", override_oracle=False, two_stream=True):
    """GENERATIVE accuracy for the bank-woven path (mirrors run_glados_staged.score_gen_live): one
    capture forward produces the bank-COMPOSED lattice from host hidden via α+composer; a second forward
    scores the legal answers with that (optionally causally-controlled) lattice injected by γ."""
    from .. import run_glados_staged as G
    model.eval()
    K = G.K
    if perm is None:
        perm = tuple(list(range(1, K)) + [0])
    inject = control != "zero"
    correct = tot = det_t = det_r = ab_t = ab_r = 0
    by_rung = {}
    for i in range(0, len(recs), bs):
        chunk = recs[i:i + bs]; Bp = len(chunk); Nmax = max(r["n"] for r in chunk)
        dvec = torch.tensor([len(r["vnames"]) for r in chunk], device=dev)
        surv = None
        if inject:
            if override_oracle:
                surv = torch.zeros(Bp, Nmax, K, device=dev)
                for b, r in enumerate(chunk):
                    surv[b, : r["n"], :] = torch.from_numpy(r["surv"]).to(dev)
            else:
                cap_p = [r.get("alpha_prompt", r["prompt"]) for r in chunk] if two_stream \
                    else [r["prompt"] for r in chunk]
                cap_m = [r.get("alpha_mentions", r["mentions"]) for r in chunk] if two_stream \
                    else [r["mentions"] for r in chunk]
                penc = tok(cap_p, return_offsets_mapping=True, padding=True, return_tensors="pt")
                pids = penc["input_ids"].to(dev); pattn = penc["attention_mask"].to(dev)
                pment = _mention_tensor(cap_m, penc["offset_mapping"], Bp, Nmax, pids.size(1), dev)
                csps = [csp_spec_for_record(r) for r in chunk]
                with model.live(pment, pattn, inject=False, capture=True, csps=csps, dvec=dvec):
                    _ = model.logits(pids, pattn)
                surv = model._captured_surv[:, :Nmax, :].clone()
            surv = G.apply_control_ext(surv, chunk, control, K, perm)            # control the K-set
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
            prob_of_idx = torch.tensor(prob_of, device=dev)
            surv_exp = surv[prob_of_idx]
            ctx = model.live(mention, attn, inject=True, capture=False, override=surv_exp,
                             dvec=dvec[prob_of_idx])               # rich channels recomputed from controlled set
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
            d = by_rung.setdefault(r["relation"], [0, 0]); d[0] += ok; d[1] += 1
            if r["determined"]:
                det_t += 1; det_r += ok
            else:
                ab_t += 1; ab_r += ok
    return {"overall": correct / max(1, tot), "det_acc": det_r / max(1, det_t),
            "abst_acc": ab_r / max(1, ab_t), "n": tot,
            "by_rung": {k: v[0] / max(1, v[1]) for k, v in by_rung.items()}}


@torch.no_grad()
def bank_controls(model, recs, tok, dev, bs, engage_thr, two_stream=True):
    """Engagement metrics for the bank-woven model (the same causal-control table as the live path)."""
    out = {}
    sc = lambda **kw: score_bank_woven(model, recs, tok, dev, bs=bs, two_stream=two_stream, **kw)
    full = sc(control="true")
    out["true"] = full["overall"]; out["det_acc"] = full["det_acc"]; out["by_rung"] = full["by_rung"]
    for c in ("shuffle", "permute", "corrupt", "zero"):
        out[c] = sc(control=c)["overall"]
    out["oracle"] = sc(control="true", override_oracle=True)["overall"]
    from .. import run_glados_staged as G
    out["base"] = sc(control="zero", use_base=True, fewshot=G.FEWSHOT)["overall"]
    out["gate"] = float(torch.tanh(model.gamma.alpha))
    out["drop"] = out["true"] - min(out["shuffle"], out["permute"], out["corrupt"])
    out["lift"] = out["true"] - out["zero"]
    out["wired"] = bool(out["drop"] * 100 >= engage_thr and abs(out["gate"]) > 1e-2)
    out["necessary"] = bool(out["lift"] * 100 >= engage_thr)
    out["engages"] = bool(out["wired"] and out["necessary"])
    out["n"] = full["n"]
    out["faculties"] = model.composer.faculties()
    return out


# ============================================================ training the bank-woven readout
def train_bank_woven(olmo_ids, tok, dev, a, train_recs, eval_recs, two_stream=True, composer=None,
                     rich=True):
    """Train the bank-woven readout. Phase A: warm α (+LoRA) on the DIRECT J0 dominate-dedₚ (the SATNet
    grounding fix) so α compiles a real per-cell lattice from host hidden. Phase B: train LoRA + α + γ on
    the answer-span LM CE (γ reads the DETACHED bank-COMPOSED lattice) + the standing J0 α-supervision.
    The organ is the frozen/certified bank — no trainable organ params, no Phase-A organ warmup needed."""
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
    assert mid < inj, f"mid_layer ({mid}) must precede inject_layer ({inj})"
    alpha = DenseLatentProjector(D, G.K, dctx=a.dctx, dp=a.alpha_dp, heads=a.alpha_heads)
    if composer is None:
        composer = BankComposerOrgan(dev="cpu",
                                     core_ckpt=getattr(a, "core_ckpt", "runs/general_organ_full.pt"),
                                     use_core=getattr(a, "use_core", True))
    model = BankWoven(peft_model, D, G.K, alpha, composer, mid, inj, gamma_hidden=a.gamma_hidden,
                      rich=rich).to(dev)
    model.alpha.float(); model.gamma.float()
    noop, live = verify_noop_bank(model, tok, dev, eval_recs[0])
    from ..oracle_readout import n_trainable
    print(f"\n  BANK-WOVEN  mid {mid} -> inject {inj}/{nL}  readout={'rich(set+card+reliability)' if rich else 'set'}"
          f"  faculties={composer.faculties()}  trainable {n_trainable(model):,}", flush=True)
    print(f"  NO-OP @ INIT max|base-(LoRA+gate0)| = {noop:.3e} (expect ~0) | gate-on moves {live:.3e}",
          flush=True)
    lora_params = [p for n, p in model.model.named_parameters() if p.requires_grad]
    rng = np.random.default_rng(a.seed + 3)

    def batch(n):
        idxs = rng.integers(0, len(train_recs), n).tolist()
        return _bank_batch([train_recs[i] for i in idxs], tok, dev, two_stream)

    def alpha_capture(ba):
        if two_stream:
            with model.live(ba["a_mention"], ba["a_attn"], inject=False, capture=True,
                            csps=ba["csps"], dvec=ba["dvec"]):
                _ = model.logits(ba["a_input_ids"], ba["a_attn"])
        else:
            with model.live(ba["mention"], ba["attn"], inject=False, capture=True,
                            csps=ba["csps"], dvec=ba["dvec"]):
                _ = model.logits(ba["input_ids"], ba["attn"])

    # ----- Phase A: warm α (+LoRA) on the direct J0 dominate-dedₚ -----
    optA = torch.optim.AdamW(
        [{"params": lora_params, "lr": a.lora_lr, "weight_decay": 0.01},
         {"params": list(model.alpha.parameters()), "lr": a.alpha_lr, "weight_decay": 0.0}],
        betas=(0.9, 0.95))
    model.train(); t0 = time.time()
    for s in range(1, a.warm_steps + 1):
        ba = batch(a.bs)
        alpha_capture(ba)                              # compiles b0 (stashed) — composer runs but γ off
        alpha_loss = dominate_dedp_loss([model._last_b0], ba["tgt"], ba["vmask"])
        optA.zero_grad(); alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0); optA.step()
        if s % max(1, a.warm_steps // 6) == 0 or s == 1:
            from .. import latent_tasks as LT
            an, ad, _, _ = LT.narrowing_stats(model._last_b0, ba["tgt"], ba["vmask"])
            print(f"  [warmup α/J0] step {s:5d}  α_dedp {alpha_loss.item():.3f}  "
                  f"α-recall {an/max(1,ad):.3f}  {time.time()-t0:.0f}s", flush=True)

    # ----- Phase B: LoRA + α + γ on LM CE (γ reads the DETACHED bank-composed lattice) + standing J0 -----
    optB = torch.optim.AdamW(
        [{"params": lora_params, "lr": a.lora_lr, "weight_decay": 0.01},
         {"params": list(model.alpha.parameters()), "lr": a.alpha_lr, "weight_decay": 0.0},
         {"params": list(model.gamma.parameters()), "lr": a.gamma_lr, "weight_decay": 0.0}],
        betas=(0.9, 0.95))
    model.train(); t0 = time.time()
    seen_open = False
    for s in range(1, a.steps + 1):
        ba = batch(a.bs)
        if two_stream:
            alpha_capture(ba)                          # α-stream (full text) → composed lattice + b0
            aux_alpha = dominate_dedp_loss([model._last_b0], ba["tgt"], ba["vmask"])
            surv = model._captured_surv[:, : ba["mention"].shape[1], :]
            with model.live(ba["mention"], ba["attn"], inject=True, capture=False, override=surv,
                            dvec=ba["dvec"]):
                logits = model.logits(ba["input_ids"], ba["attn"]).float()
        else:
            with model.live(ba["mention"], ba["attn"], inject=True, capture=True, csps=ba["csps"],
                            dvec=ba["dvec"]):
                logits = model.logits(ba["input_ids"], ba["attn"]).float()
            aux_alpha = dominate_dedp_loss([model._last_b0], ba["tgt"], ba["vmask"])
        lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                             ba["labels"][:, 1:].reshape(-1), ignore_index=-100)
        loss = lm + a.alpha_sup_w * aux_alpha
        optB.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0); optB.step()
        gate = float(torch.tanh(model.gamma.alpha))
        seen_open = seen_open or (abs(gate) > 1e-3)
        if s % max(1, a.steps // 12) == 0 or s == 1:
            acc = score_bank_woven(model, eval_recs, tok, dev, control="true", bs=a.bs,
                                   two_stream=two_stream)
            print(f"  step {s:5d}  lm {lm.item():.3f}  α_J0 {aux_alpha.item():.3f}  "
                  f"gate(tanh α) {gate:+.3f}  bank-woven acc {acc['overall']*100:4.1f}% "
                  f"(det {acc['det_acc']*100:.0f})  {time.time()-t0:.0f}s", flush=True)
            model.train()
    if not seen_open:
        print("  [WARN] γ gate never opened — readout received no gradient (check inject wiring).",
              flush=True)
    return model


# ============================================================================================
# ALPHA_STRUCT — α EMITS the structure; the composer runs on csp_α (NOT rec["csp"]).            
# ============================================================================================
# This is the Stage-2 architecture the SHUFFLED diagnostic proved is required (runs/shuffled_diag.json:
# correctness was 100% metadata-driven) and the LoRA/search studies proved is learnable (canonical
# structure-F1 0.98; LoRA-host closes 36% of the NL gap; α-as-search refines the rest). Here the live
# organ reads α's COMPILED factor graph, not the handed-in metadata:
#
#   StructureRack(host hidden) → pin + typed-pair logits          (α EMITS the structure)
#     → decode_structure → α_facts                                (the typed factor graph)
#     → build_csp_from_struct(α_facts, n, d) = csp_α              (NO rec["csp"])
#     → AlphaStructComposerOrgan: certified floor + CoreNarrowOrgan + α-lattice on csp_α (verifier-gated)
#     → γ (zero-init gate ⇒ no-op@init) reads the COMPOSED lattice → the LM head generates.
#
# rec["csp"] appears ONLY in the eval-time output_check (alpha_struct.output_check) + the structure-
# supervision target builder — both OUTSIDE the forbid_record_csp() forward guard.


class AlphaStructComposerOrgan(BankComposerOrgan):
    """The ALPHA_STRUCT composer: identical certified machinery as BankComposerOrgan, but it composes on
    csp_α (built from α's EMITTED facts) instead of the handed-in rec['csp']. The pretrained
    CoreNarrowOrgan (runs/general_organ_full.pt) plugs in as the neural faculty when present and falls
    back to a fresh/absent organ for the smoke (build_bank only loads it if the checkpoint exists)."""

    def compose_from_struct(self, facts, n, d, alpha_dom=None):
        """csp_α = build_csp_from_struct(α_facts) → the verifier-gated reduced product. tags/system are
        STRUCTURAL metadata, so they are NOT threaded (the certified AC/factor floor suffices for the
        minimal binary+pin vocab); the soundness guarantee is now relative to csp_α (design §3)."""
        csp_a = build_csp_from_struct(facts, n, d)
        return self.compose_one(csp_a, alpha_dom, system=None, tags=())

    @torch.no_grad()
    def __call__(self, b0, vmask, theta, alpha_facts, nd, K):
        """b0 [B,N,K] α's per-cell candidate logits; alpha_facts a list of B fact-lists (α's emitted
        factor graph, or None ⇒ α-only fallback, or [] ⇒ EMPTY_STRUCT contentless control); nd a list of
        B (n, d). Returns surv [B,N,K] composed from csp_α + a list of composer traces."""
        B, N, _ = b0.shape
        alive = (torch.sigmoid(b0) >= theta).float() * vmask.unsqueeze(-1)
        alive_np = alive.cpu().numpy()
        out = np.zeros((B, N, K), dtype=np.float32)
        traces = [None] * B
        # Phase 1: build each instance's csp_α + α-gate domain (the per-instance prep that can raise on a
        # malformed emitted structure → graceful α-only fallback, exactly as the serial path). Survivors are
        # composed TOGETHER (the gated neural organ — the dominant cost — fans across the batch dim).
        items, slots = [], []
        for b in range(B):
            facts = alpha_facts[b] if alpha_facts is not None else None
            n, d = nd[b]
            if facts is None:
                out[b] = alive_np[b]                       # no structure available: α-only fallback
                continue
            d = min(int(d), K)
            try:
                csp_a = build_csp_from_struct(facts, n, d)
                adom = None
                if self.use_alpha_gate:
                    adom = tuple(frozenset(v for v in range(d) if alive_np[b, i, v] > 0.5)
                                 for i in range(n))         # α's per-cell lattice as a gated passenger
                items.append((csp_a, adom, None, ()))
                slots.append((b, n))
            except Exception:
                out[b] = alive_np[b]                       # malformed csp_α ⇒ graceful α-only fallback
        # Phase 2: one batched verifier-gated reduced product over the survivors (bitwise == per-instance).
        if items:
            composed_list, trace_list = self.compose_batch(items)
            for (b, n), composed, tr in zip(slots, composed_list, trace_list):
                for i in range(n):
                    for v in composed.dom[i]:
                        if v < K:
                            out[b, i, v] = 1.0
                traces[b] = tr
        return torch.from_numpy(out).to(b0.device), traces

    @torch.no_grad()
    def compose_and_feedback(self, b0, vmask, theta, alpha_facts, nd, K, queries=None):
        """Lever B's organ→α channel. Same verifier-gated reduced product as __call__, but ALSO returns
        the per-cell ORGAN FEEDBACK [B,N,FB_DIM] = [narrowed-card-fraction, determined, conflict/⊥] and a
        per-instance verdict {solvable, determines, answer} from the EXACT solver (csp_α's exact_dedP).
        The feedback is what the organ reports back to the rack for the next loop step:
          * card-fraction |composed.dom[i]|/d   — how far the organ narrowed cell i (which cells determined),
          * determined = (|composed.dom[i]|==1) — the solved cells,
          * conflict   = (csp_α UNSAT) broadcast — the 'this structure caused ⊥' (unsat-core) signal.
        Soundness is unchanged: the composed lattice is the certified floor narrowing of csp_α; the
        feedback is a read-only report (it conditions α's NEXT emission, never the floor)."""
        B, N, _ = b0.shape
        alive = (torch.sigmoid(b0) >= theta).float() * vmask.unsqueeze(-1)
        alive_np = alive.cpu().numpy()
        out = np.zeros((B, N, K), dtype=np.float32)
        fb = np.zeros((B, N, FB_DIM), dtype=np.float32)
        verdicts = [None] * B
        # Phase 1: build csp_α + α-gate domain per instance (fallback on a malformed structure). Survivors
        # are composed TOGETHER (the gated neural organ fans across the batch dim — identical per-instance).
        items, slots = [], []
        for b in range(B):
            facts = alpha_facts[b] if alpha_facts is not None else None
            n, d = nd[b]
            if facts is None:
                out[b] = alive_np[b]
                continue
            d = min(int(d), K)
            try:
                csp_a = build_csp_from_struct(facts, n, d)
                adom = None
                if self.use_alpha_gate:
                    adom = tuple(frozenset(v for v in range(d) if alive_np[b, i, v] > 0.5)
                                 for i in range(n))
                items.append((csp_a, adom, None, ()))
                slots.append((b, n, d, csp_a))
            except Exception:
                out[b] = alive_np[b]
        # Phase 2: one batched verifier-gated reduced product over the survivors.
        composed_list = self.compose_batch(items)[0] if items else []
        # Phase 3: per-instance exact verdict (csp_α's exact_dedP) + feedback + scatter (cheap Rust).
        for (b, n, d, csp_a), composed in zip(slots, composed_list):
            q = queries[b] if queries is not None else None
            try:
                ded = C.exact_dedP(csp_a, csp_a.full())          # exact verdict (sound + complete for SAT)
                solvable = any(len(c) > 0 for c in ded)
                for i in range(n):
                    ci = composed.dom[i]
                    for v in ci:
                        if v < K:
                            out[b, i, v] = 1.0
                    if not solvable:                              # ⊥ ⇒ unsat-core signal broadcast
                        fb[b, i, 2] = 1.0
                    else:
                        fb[b, i, 0] = len(ci) / max(1, d)
                        fb[b, i, 1] = 1.0 if len(ci) == 1 else 0.0
                        fb[b, i, 2] = 1.0 if len(ci) == 0 else 0.0
                det, ans = False, None
                if solvable and q is not None and q < n and len(ded[q]) == 1:
                    det, ans = True, int(next(iter(ded[q])))
                verdicts[b] = {"solvable": solvable, "determines": det, "answer": ans}
            except Exception:
                out[b] = alive_np[b]
                verdicts[b] = None
        return (torch.from_numpy(out).to(b0.device),
                torch.from_numpy(fb).to(b0.device), verdicts)


def _rich_from_set(surv_set, dvec):
    """[B,N,K] composed candidate SET (+ per-instance domain sizes dvec[B]) → rich [B,N,K+2] =
    [set(K), |set|/d, reliability] (the alien-consumer encoding; shared schema with BankWoven so the
    LM mines the partial lattice and discounts low-reliability cells — an α-miscompile degrades
    gracefully instead of dragging the LM off a cliff)."""
    B, N, K = surv_set.shape
    card = surv_set.sum(-1)
    if dvec is None:
        d = torch.full((B, 1), float(K), device=surv_set.device)
    else:
        d = dvec.to(surv_set.device).float().clamp_min(1.0).view(B, 1)
    feat = surv_set.new_zeros(B, N, K + 2)
    feat[..., :K] = surv_set
    feat[..., K] = (card / d).clamp(0.0, 1.0)
    feat[..., K + 1] = (1.0 - (card - 1).clamp_min(0.0) / (d - 1).clamp_min(1.0)).clamp(0.0, 1.0)
    return feat


class AlphaStructWoven(nn.Module):
    """The live ALPHA_STRUCT woven GLaDOS. Forward (one host pass) via two decoder-layer hooks: MID
    captures host hidden h; INJECT runs the StructureRack(h) → α_facts → csp_α → composer → γ scatter
    (zero-init gate ⇒ bitwise no-op@init). Trainable: LoRA (host) + the rack + the per-cell candidate
    head + γ; the organ is the frozen/certified bank. The rack is grounded by structure-supervision (the
    new J0) against the witness's free true structure; the per-cell candidate head is grounded by the
    standing dominate-dedₚ J0; the LM-CE flows to LoRA + γ."""

    def __init__(self, peft_model, D, K, rack: StructureRack, composer: AlphaStructComposerOrgan,
                 mid_layer, inject_layer, gamma_hidden=256, theta=0.5, rich=True, cand_hidden=384):
        super().__init__()
        from ..oracle_readout import _decoder_layers
        self.model = peft_model
        self.alpha = rack                          # StructureRack (trainable) — α EMITS the structure
        self.composer = composer                   # AlphaStructComposerOrgan (frozen / certified)
        din = rack.encoder.din
        # per-cell candidate lattice head (b0): the gated α-passenger + the J0 grounding target. Reuses
        # the rack's shared featurization (one encoder pass for both structure + candidate sets).
        self.cand = nn.Sequential(nn.Linear(din, cand_hidden), nn.GELU(), nn.Linear(cand_hidden, K))
        # LEVER B — the α↔organ FEEDBACK adapter: maps the organ's per-cell report [B,N,FB_DIM] into the
        # rack's fused feature so α's NEXT emission can condition on what the organ narrowed/where it hit ⊥.
        # ZERO-INIT (bias-free) ⇒ at t=0 (no feedback) and with the adapter untrained the loop is a bitwise
        # no-op perturbation of the single-shot compile (T=1 == the core); it LEARNS to use feedback, the
        # same "no-op@init then open the gate" discipline as γ. Composable with α-as-search (search runs
        # WITHIN each loop step).
        self.fb_adapter = nn.Linear(FB_DIM, din, bias=False)
        nn.init.zeros_(self.fb_adapter.weight)
        self.rich = rich
        self.Fin = K + 2 if rich else K
        self.gamma = OracleGamma(D, self.Fin, gamma_hidden)
        self.D, self.K = D, K
        self.mid_layer, self.inject_layer = mid_layer, inject_layer
        self.theta = theta
        # live state
        self._mention = self._attn = None
        self._nd = self._dvec = None
        self._inject = self._capture = False
        self._override = None
        self._empty_struct = False
        # lever levers: α-as-search (search_K>1) + iterative α↔organ loop (loop_T>1); both OFF by default
        # (search_K=1, loop_T=1) ⇒ the single-shot core, no behavior change / no-op@init preserved.
        self._search_K = 1
        self._search_temp = 1.0
        self._loop_T = 1
        self._queries = None
        self._search_rng = None
        self._h_mid = None
        self._captured_surv = None
        self._last_pin = self._last_pair = self._last_b0 = self._last_vmask = self._last_facts = None
        self._last_loop_info = None
        layers = _decoder_layers(peft_model)
        self._mid_handle = layers[mid_layer].register_forward_hook(self._mid_hook)
        self._inj_handle = layers[inject_layer].register_forward_hook(self._inj_hook)

    # ---- hooks ----
    def _mid_hook(self, module, args, output):
        self._h_mid = output[0] if isinstance(output, tuple) else output
        return output

    def _compile_struct(self, feedback=None):
        """StructureRack(mid hidden) → (pin_logits, pair_logits, b0, vmask). One featurization feeds both
        the CSP structure head and the per-cell candidate head. `feedback` [B,N,FB_DIM] (lever B) is the
        organ's report from the previous loop step; it conditions α's emission via the zero-init
        fb_adapter (feedback=None ⇒ t=0 ⇒ bitwise the single-shot compile)."""
        h = self._h_mid.float()
        m = self._mention.to(h.device)
        denom = m.sum(-1, keepdim=True).clamp_min(1e-6)
        v_mean = torch.einsum("bnt,btd->bnd", m, h) / denom            # mention-pool cell identity
        feat = self.alpha.featurize(v_mean, h, self._attn.to(h.device))
        if feedback is not None:                                      # α↔organ feedback conditioning
            feat = feat + self.fb_adapter(feedback.to(feat.dtype))
        pin_l, pair_l = self.alpha.csp(feat)                          # α EMITS structure
        b0 = self.cand(feat)                                          # α's per-cell candidate lattice
        vmask = (m.sum(-1) > 0.5).float()
        self._last_pin, self._last_pair, self._last_b0, self._last_vmask = pin_l, pair_l, b0, vmask
        return pin_l, pair_l, b0, vmask

    def _decode_facts(self, pin_l, pair_l, vmask):
        """Per-instance decode of the emitted factor graph. EMPTY_STRUCT ⇒ [] (contentless control)."""
        B = pin_l.shape[0]
        facts = []
        for b in range(B):
            n, d = self._nd[b]
            if self._empty_struct:
                facts.append([])
            else:
                facts.append(decode_structure(pin_l[b], pair_l[b], vmask[b], int(n), int(d)))
        return facts

    def _inj_hook(self, module, args, output):
        if not (self._inject or self._capture):
            return output
        hs = output[0] if isinstance(output, tuple) else output
        surv = self._override
        if surv is None:
            if self._loop_T > 1 or self._search_K > 1:
                # LEVERS A/B: α-as-search and/or the iterative α↔organ loop pick the structure before γ.
                surv = self.iterate_alpha_organ()[0]
            else:
                pin_l, pair_l, b0, vmask = self._compile_struct()
                facts = self._decode_facts(pin_l, pair_l, vmask)
                self._last_facts = facts
                # THE INVARIANT: the composer forward runs under forbid_record_csp() — any read of
                # rec['csp'] (csp_spec_for_record) raises. The composer sees ONLY α's emitted structure.
                with forbid_record_csp():
                    surv = self.composer(b0, vmask, self.theta, facts, self._nd, self.K)[0]
                surv = surv * vmask.unsqueeze(-1)
                self._captured_surv = surv.detach()
            surv = self._captured_surv
        if not self._inject:
            return output
        feat = _rich_from_set(surv, self._dvec) if self.rich else surv
        delta = self.gamma.delta(feat.to(hs.device), self._mention.to(hs.device))
        hs = hs + (torch.tanh(self.gamma.alpha) * delta).to(hs.dtype)
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

    # ---- context manager ----
    @contextlib.contextmanager
    def live(self, mention, attn, *, inject, capture=False, override=None, nd=None, dvec=None,
             empty_struct=False, search_K=1, search_temp=1.0, loop_T=1, queries=None, search_rng=None):
        old = (self._mention, self._attn, self._inject, self._capture, self._override, self._nd,
               self._dvec, self._empty_struct, self._search_K, self._search_temp, self._loop_T,
               self._queries, self._search_rng)
        self._mention, self._attn = mention, attn
        (self._inject, self._capture, self._override, self._nd, self._dvec, self._empty_struct,
         self._search_K, self._search_temp, self._loop_T, self._queries, self._search_rng) = (
            inject, capture, override, nd, dvec, empty_struct, search_K, search_temp, loop_T,
            queries, search_rng)
        try:
            yield
        finally:
            (self._mention, self._attn, self._inject, self._capture, self._override, self._nd,
             self._dvec, self._empty_struct, self._search_K, self._search_temp, self._loop_T,
             self._queries, self._search_rng) = old

    def logits(self, input_ids, attn):
        return self.model(input_ids=input_ids, attention_mask=attn).logits

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def remove_hooks(self):
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

    # ============================================================ LEVER A — α-AS-SEARCH (real)
    def propose_structures(self, pin_l_b, pair_l_b, vmask_b, n, d, query, *, K=1, temp=1.0,
                           rng=None, output_check_fn=None, csp_true=None):
        """α-AS-SEARCH (design §5.3 / runs/alpha_struct_search_nl.json — verifier-selection recovered
        answer-acc +0.11). For ONE instance: decode K candidate factor graphs from its CSP-head logits
        (candidate 0 = greedy argmax; 1..K-1 = temperature samples — `sample_candidates`), compile each
        to csp_α, run the EXACT certified solver (`solve_query` = exact_dedP), KEEP candidates that are
        SOLVABLE (not ⊥) and DETERMINE the query, and SELECT the highest structure-CONFIDENCE survivor.
        The selection is SOUND ONLY RELATIVE TO csp_α — it reads the candidate's own emitted structure +
        the certified solver on THAT structure + the model's confidence, never gold. So a candidate can
        be 'solvable + determines the query' on csp_α while csp_α is the WRONG problem: this filter does
        NOT give answer-soundness (notes/soundness.md (B) vs (D)). Returns the selected
            {facts, conf, solvable, determines, answer, certified, n_pass, K}.
        Answer-soundness arrives ONLY via `output_check_fn` + `csp_true` (the RELIABILITY layer, used
        OUTSIDE the forbid_record_csp forward — e.g. the diagnostic): the survivor must ALSO pass the
        output_check (answer is a real solution-value of the TRUE instance) — the (D) gate that licenses
        the accept, and it needs ground truth (or a domain verifier) to exist. The `certified` flag below
        is set on that gate when present, else on the csp_α-relative filter alone (rel.-csp_α only — read
        it as 'passed the available check', not 'correct'). With none passing we fall back to the greedy
        candidate (uncertified). K=1 ⇒ greedy only
        ⇒ identical to the single-shot decode (no behavior change)."""
        cands = sample_candidates(pin_l_b, pair_l_b, vmask_b, n, K, temp=temp, d=d, rng=rng)
        for c in cands:
            solv, det, ans = solve_query(c["facts"], n, d, query)
            c["solvable"], c["determines"], c["answer"] = solv, det, ans
            ok = bool(solv and det)
            if ok and output_check_fn is not None and csp_true is not None:
                ok = bool(output_check_fn(ans, csp_true, query))     # certified gate (reliability layer)
            c["passes"] = ok
        passing = [c for c in cands if c["passes"]]
        if passing:
            sel = max(passing, key=lambda c: c["conf"])
            sel = {**sel, "certified": True, "n_pass": len(passing)}
        else:
            sel = {**cands[0], "certified": False, "n_pass": 0}      # greedy fallback (uncertified)
        sel["K"] = int(K)
        return sel

    # ============================================================ LEVER B — ITERATIVE α↔organ LOOP (real)
    @torch.no_grad()
    def iterate_alpha_organ(self, *, T=None, search_K=None, search_temp=None, rng=None, queries=None,
                            output_check_fn=None, csps_true=None):
        """The DEEP bidirectional (inference-time) loop (north-star orchestrator on-ramp). Up to T steps:

            t=0: α reads the host hidden → emits structure_0 → composer narrows → organ returns the PARTIAL
                 result (per-cell narrowed cardinalities / which cells determined / the ⊥ unsat-core signal);
            t>0: that feedback CONDITIONS the rack's next read (via the zero-init fb_adapter) → α emits a
                 REVISION structure_t → re-narrow → new feedback;
            stop on FIXPOINT (structure stable vs the previous step) or T.

        α-as-search composes INSIDE each step (search_K>1 ⇒ propose+verifier-select the step's structure).
        Every step's lattice is the certified-floor narrowing of its csp_α (sound REL. csp_α each step —
        the floor never drops a value used by a solution OF csp_α; this is NOT answer-soundness if csp_α
        is the wrong compile); answer-soundness arrives only when the final answer is output-checked
        downstream against ground truth or a domain verifier (notes/soundness.md (B)/(D)). T=1 ⇒ one
        emission, no feedback ever applied ⇒ bitwise the single-shot core. Returns (surv [B,N,K], info).

        Generalizes toward the multi-step orchestrator (notes/north_star_orchestrator.md): here the loop
        iterates α's COMPILE on a fixed host read; the orchestrator additionally re-runs the LM between
        calls — same feedback channel, one level up (the organ-call is the checked anchor, the strategy
        between calls is free)."""
        from .. import curriculum as CU
        T = self._loop_T if T is None else int(T)
        K = self._search_K if search_K is None else int(search_K)
        temp = self._search_temp if search_temp is None else float(search_temp)
        rng = self._search_rng if rng is None else rng
        queries = self._queries if queries is None else queries
        B = self._mention.shape[0]
        feedback = None                                              # zero at t=0 (the single-shot read)
        facts_prev = [None] * B
        surv = None
        info = {"T": T, "search_K": K, "steps": 0, "stop": "T", "facts_per_step": [], "verdicts": None}
        norm = lambda fs: None if fs is None else set(CU.norm_facts([tuple(f) for f in fs]))
        for t in range(max(1, T)):
            pin_l, pair_l, b0, vmask = self._compile_struct(feedback)
            facts = []
            for b in range(B):
                n, d = self._nd[b]
                dd = min(int(d), self.K)
                q = queries[b] if queries is not None else None
                if self._empty_struct:
                    facts.append([])
                elif K > 1:
                    sel = self.propose_structures(
                        pin_l[b], pair_l[b], vmask[b], int(n), dd, q, K=K, temp=temp, rng=rng,
                        output_check_fn=output_check_fn,
                        csp_true=(csps_true[b] if csps_true is not None else None))
                    facts.append(sel["facts"])
                else:
                    facts.append(decode_structure(pin_l[b], pair_l[b], vmask[b], int(n), dd))
            self._last_facts = facts
            with forbid_record_csp():                               # the composer never sees rec['csp']
                surv, feedback_next, verdicts = self.composer.compose_and_feedback(
                    b0, vmask, self.theta, facts, self._nd, self.K, queries)
            surv = surv * vmask.unsqueeze(-1)
            info["facts_per_step"].append(facts)
            info["verdicts"] = verdicts
            info["steps"] = t + 1
            if t > 0 and all(norm(facts[b]) == norm(facts_prev[b]) for b in range(B)):
                info["stop"] = "fixpoint"
                break
            facts_prev = facts
            feedback = feedback_next
        self._captured_surv = surv.detach()
        self._last_loop_info = info
        return self._captured_surv, info


# ============================================================ batch + no-op@init proof
def _alpha_struct_batch(recs, tok, dev, two_stream):
    """build_live_batch + the ALPHA_STRUCT extras: per-instance (n, d), the structure-supervision
    targets (from the witness's free true facts — NOT rec['csp']), and dvec. rec['csp'] is NOT read."""
    from .. import run_glados_staged as G
    ba = G.build_live_batch(recs, tok, dev, two_stream=two_stream)
    Nmax = max(r["n"] for r in recs)
    ba["nd"] = [(r["n"], len(r["vnames"])) for r in recs]
    ba["queries"] = [int(r["query"]) for r in recs]                # lever B "determines-query" selector
    ba["dvec"] = torch.tensor([len(r["vnames"]) for r in recs], device=dev)
    pin_t, pair_t = facts_to_struct_targets([r.get("facts", []) for r in recs],
                                            [r["n"] for r in recs], Nmax)
    ba["pin_tgt"] = pin_t.to(dev); ba["pair_tgt"] = pair_t.to(dev)
    return ba


def verify_noop_alpha_struct(model, tok, dev, rec):
    """Bitwise no-op at init: the zero-init γ gate ⇒ the live α-structure-composed injection moves
    logits by 0 (and gate-on moves them). rec['csp'] is NOT threaded — α emits the structure."""
    enc = tok(rec["prompt"], return_offsets_mapping=True, return_tensors="pt")
    ids = enc["input_ids"].to(dev); attn = torch.ones_like(ids)
    mention = _mention_tensor([rec["mentions"]], enc["offset_mapping"], 1, rec["n"], ids.size(1), dev)
    nd = [(rec["n"], len(rec["vnames"]))]
    dvec = torch.tensor([len(rec["vnames"])], device=dev)
    with torch.no_grad():
        base = model.model(input_ids=ids, attention_mask=attn).logits.float()
    with torch.no_grad(), model.live(mention, attn, inject=True, capture=True, nd=nd, dvec=dvec):
        g0 = model.logits(ids, attn).float()
    noop = float((base - g0).abs().max())
    with torch.no_grad():
        saved = model.gamma.alpha.data.clone(); model.gamma.alpha.data.fill_(2.0)
        with model.live(mention, attn, inject=True, capture=True, nd=nd, dvec=dvec):
            g1 = model.logits(ids, attn).float()
        model.gamma.alpha.data.copy_(saved)
    return noop, float((base - g1).abs().max())


# ============================================================ training the ALPHA_STRUCT woven readout
def train_alpha_struct_woven(olmo_ids, tok, dev, a, train_recs, eval_recs, two_stream=True,
                             composer=None, rich=True, use_lora=True):
    """STAGE-2 ALPHA_STRUCT training (the one-command GPU run once general_organ_full.pt lands).

    Phase A: warm the StructureRack (+ LoRA host, the bidirectional channel the LoRA study validated)
      on the STRUCTURE-SUPERVISION loss (α-emitted vs the witness's free true factor graph, soundness-
      asymmetric) + the per-cell candidate head on the standing dominate-dedₚ J0.
    Phase B: train LoRA + rack + candidate head + γ on the answer-span LM-CE (γ reads the DETACHED
      α-structure-composed lattice) + the standing structure-supervision + the per-cell J0.

    rec['csp'] NEVER enters the forward (the composer runs under forbid_record_csp() on csp_α). The
    organ is the frozen/certified bank — the pretrained CoreNarrowOrgan plugs in when present."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from .. import run_glados_staged as G
    mid_id, D, nL = olmo_ids
    olmo = AutoModelForCausalLM.from_pretrained(mid_id, dtype=torch.bfloat16).to(dev).eval()
    for p in olmo.parameters():
        p.requires_grad_(False)
    # ADAPTER A/B (the capacity-vs-representation study): the ONLY thing that varies across arms.
    #   full_ft  → unfreeze the host (the diagnostic UPPER BOUND; breaks the base-matrix thesis, not prod);
    #   use_dora → weight-decomposed LoRA (use_dora=True; usually beats LoRA at equal rank/params);
    #   else     → plain LoRA at rank a.lora_r. lr for the host group is a.lora_lr (driver lowers it for FT).
    full_ft = bool(getattr(a, "full_ft", False))
    if full_ft:
        for p in olmo.parameters():
            p.requires_grad_(True)
        peft_model = olmo
    elif use_lora:
        lconf = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, bias="none",
                           use_dora=bool(getattr(a, "use_dora", False)),
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                           "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
        peft_model = get_peft_model(olmo, lconf)
    else:
        peft_model = olmo
    mid = min(a.mid_layer, nL - 1); inj = min(a.inject_layer, nL - 1)
    assert mid < inj, f"mid_layer ({mid}) must precede inject_layer ({inj})"
    rack = StructureRack(D, G.K, dp=a.alpha_dp, heads=a.alpha_heads)
    if composer is None:
        composer = AlphaStructComposerOrgan(dev="cpu",
                                            core_ckpt=getattr(a, "core_ckpt", "runs/general_organ_full.pt"),
                                            use_core=getattr(a, "use_core", True))
    model = AlphaStructWoven(peft_model, D, G.K, rack, composer, mid, inj, gamma_hidden=a.gamma_hidden,
                             rich=rich).to(dev)
    model.alpha.float(); model.cand.float(); model.gamma.float()
    noop, live = verify_noop_alpha_struct(model, tok, dev, eval_recs[0])
    from ..oracle_readout import n_trainable
    _adapter = ("full_ft" if full_ft else
                (f"dora_r{a.lora_r}" if (use_lora and bool(getattr(a, "use_dora", False)))
                 else (f"lora_r{a.lora_r}" if use_lora else "frozen_host")))
    print(f"\n  ALPHA_STRUCT-WOVEN  mid {mid} -> inject {inj}/{nL}  adapter={_adapter}  "
          f"readout={'rich' if rich else 'set'}  faculties={composer.faculties()}  "
          f"trainable {n_trainable(model):,}", flush=True)
    print(f"  NO-OP @ INIT max|base-(LoRA+gate0)| = {noop:.3e} (expect ~0) | gate-on moves {live:.3e}",
          flush=True)
    lora_params = [p for n, p in model.model.named_parameters() if p.requires_grad]
    # the rack + the candidate head + the α↔organ feedback adapter (lever B) all train on the α signals.
    rack_cand = list(model.alpha.parameters()) + list(model.cand.parameters()) \
        + list(model.fb_adapter.parameters())
    rng = np.random.default_rng(a.seed + 5)
    struct_w = getattr(a, "struct_sup_w", 1.0)
    # LEVER AUDIT (the composer-cost fix): α-as-search (search_K>1) and the iterative α↔organ loop
    # (loop_T>1) are INFERENCE-time levers (the gate/eval arbiter selects/loops on the certified solver).
    # They do NOT help TRAINING — γ reads the DETACHED composed lattice and the rack/cand/LoRA learn from
    # the standing structure-sup + J0 + LM-CE, none of which need K candidates or T feedback steps. Running
    # them in the training forward is a pure K×T composer tax (the dominant CPU cost) for no training
    # benefit, so TRAINING is hard-pinned to the greedy single-shot pass (search_K=1, loop_T=1); the levers
    # apply ONLY at eval/gate (clair.organ.alpha_struct_diag / run_alpha_struct_gate). With the default
    # config (search_K=1/loop_T=1) this is a no-op; it only strips the tax if the flags were passed to weave.
    _sK = int(getattr(a, "search_K", 1)); _lT = int(getattr(a, "loop_T", 1))
    if _sK > 1 or _lT > 1:
        print(f"  [lever audit] search_K={_sK}/loop_T={_lT} are EVAL/GATE-only — TRAINING runs greedy "
              f"single-pass (K=1,T=1); the levers apply at the gate, not in the training forward.",
              flush=True)

    def batch(n):
        idxs = rng.integers(0, len(train_recs), n).tolist()
        return _alpha_struct_batch([train_recs[i] for i in idxs], tok, dev, two_stream)

    def alpha_capture(ba):
        m, at = (ba["a_mention"], ba["a_attn"]) if two_stream else (ba["mention"], ba["attn"])
        ids = ba["a_input_ids"] if two_stream else ba["input_ids"]
        # greedy single-shot: NO search/loop in TRAINING (levers are eval/gate-only — see the audit note).
        with model.live(m, at, inject=False, capture=True, nd=ba["nd"], dvec=ba["dvec"]):
            _ = model.logits(ids, at)

    def struct_loss(ba):
        return structure_sup_loss(model._last_pin, model._last_pair, ba["pin_tgt"], ba["pair_tgt"],
                                  model._last_vmask, wpos=getattr(a, "wpos", 2.0),
                                  wnone=getattr(a, "wnone", 1.0))

    # ----- Phase A: warm the rack (+LoRA) on structure-supervision + the per-cell J0 -----
    optA = torch.optim.AdamW(
        [{"params": lora_params, "lr": a.lora_lr, "weight_decay": 0.01},
         {"params": rack_cand, "lr": a.alpha_lr, "weight_decay": 0.0}], betas=(0.9, 0.95))
    model.train(); t0 = time.time()
    for s in range(1, a.warm_steps + 1):
        ba = batch(a.bs)
        alpha_capture(ba)                              # emits structure + b0; composer runs, γ off
        l_struct = struct_loss(ba)
        l_j0 = dominate_dedp_loss([model._last_b0], ba["tgt"], ba["vmask"])
        loss = struct_w * l_struct + a.alpha_sup_w * l_j0
        optA.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0); optA.step()
        if s % max(1, a.warm_steps // 6) == 0 or s == 1:
            print(f"  [warmup struct/J0] step {s:5d}  struct {l_struct.item():.3f}  "
                  f"J0 {l_j0.item():.3f}  {time.time()-t0:.0f}s", flush=True)

    # ----- Phase B: LoRA + rack + cand + γ on LM-CE (γ reads DETACHED csp_α lattice) + standing aux ---
    optB = torch.optim.AdamW(
        [{"params": lora_params, "lr": a.lora_lr, "weight_decay": 0.01},
         {"params": rack_cand, "lr": a.alpha_lr, "weight_decay": 0.0},
         {"params": list(model.gamma.parameters()), "lr": a.gamma_lr, "weight_decay": 0.0}],
        betas=(0.9, 0.95))
    model.train(); t0 = time.time(); seen_open = False
    for s in range(1, a.steps + 1):
        ba = batch(a.bs)
        if two_stream:
            alpha_capture(ba)                          # α-stream (full text) → csp_α lattice + structure
            aux_struct = struct_loss(ba)
            aux_j0 = dominate_dedp_loss([model._last_b0], ba["tgt"], ba["vmask"])
            surv = model._captured_surv[:, : ba["mention"].shape[1], :]
            with model.live(ba["mention"], ba["attn"], inject=True, capture=False, override=surv,
                            nd=ba["nd"], dvec=ba["dvec"]):
                logits = model.logits(ba["input_ids"], ba["attn"]).float()
        else:
            # greedy single-shot: NO search/loop in TRAINING (levers are eval/gate-only — see audit note).
            with model.live(ba["mention"], ba["attn"], inject=True, capture=True, nd=ba["nd"],
                            dvec=ba["dvec"]):
                logits = model.logits(ba["input_ids"], ba["attn"]).float()
            aux_struct = struct_loss(ba)
            aux_j0 = dominate_dedp_loss([model._last_b0], ba["tgt"], ba["vmask"])
        lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                             ba["labels"][:, 1:].reshape(-1), ignore_index=-100)
        loss = lm + struct_w * aux_struct + a.alpha_sup_w * aux_j0
        optB.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0); optB.step()
        gate = float(torch.tanh(model.gamma.alpha))
        seen_open = seen_open or (abs(gate) > 1e-3)
        if s % max(1, a.steps // 12) == 0 or s == 1:
            print(f"  step {s:5d}  lm {lm.item():.3f}  struct {aux_struct.item():.3f}  "
                  f"J0 {aux_j0.item():.3f}  gate(tanh α) {gate:+.3f}  {time.time()-t0:.0f}s", flush=True)
            model.train()
    if not seen_open:
        print("  [WARN] γ gate never opened — readout received no gradient (check inject wiring).",
              flush=True)
    return model
