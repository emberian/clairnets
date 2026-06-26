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
  (3) CERTIFIED-FLOOR-BACKED — the injected lattice is sound by construction (the certified ops run
      from the full domain and the meet of sound narrowings is sound; the neural proposals only ever
      sharpen toward the exact per-cell transformer, never below it). Not pure-neural.

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
from .bank import build_bank, certified_csp_reductions
from .compose import reduced_product
from .protocol import Certificate, CSPState, Reduction


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
        """Compose from the FULL domain on the true structure. The certified floor is sound-by-
        construction; CoreNarrowOrgan + α are verifier-gated. Returns the composed CSPState."""
        full = CSPState.full(csp, system=system, tags=frozenset(tags))
        reds = list(self.base_reductions)
        if self.use_alpha_gate and alpha_dom is not None:
            reds = reds + [_AlphaProposal(alpha_dom)]
        # verify=False: the per-round verifier gate already guarantees soundness; skip the extra
        # full-origin exact_dedP pass (cost). The certified ops still assert subset internally.
        out, tr = reduced_product(full, reds, verify=False, max_rounds=self.max_rounds)
        self.last_trace = tr
        return out

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
            n = int(vmask[b].sum().item())
            if spec is None:
                out[b] = alive_np[b]                      # no structure available: α-only fallback
                traces.append(None)
                continue
            csp, system, tags = spec
            n = csp.n
            adom = tuple(frozenset(v for v in range(csp.d) if alive_np[b, i, v] > 0.5)
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
                 gamma_hidden=256, theta=0.5):
        super().__init__()
        from ..oracle_readout import _decoder_layers
        self.model = peft_model
        self.alpha = alpha                       # DenseLatentProjector (trainable)
        self.composer = composer                 # BankComposerOrgan (frozen / certified)
        self.gamma = OracleGamma(D, K, gamma_hidden)
        self.D, self.K = D, K
        self.mid_layer, self.inject_layer = mid_layer, inject_layer
        self.theta = theta
        self._mention = self._attn = None
        self._csps = None
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
        delta = self.gamma.delta(surv.to(hs.device), self._mention.to(hs.device))
        hs = hs + (torch.tanh(self.gamma.alpha) * delta).to(hs.dtype)
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

    # ---- context manager ----
    @contextlib.contextmanager
    def live(self, mention, attn, *, inject, capture=False, override=None, csps=None):
        old = (self._mention, self._attn, self._inject, self._capture, self._override, self._csps)
        self._mention, self._attn = mention, attn
        self._inject, self._capture, self._override, self._csps = inject, capture, override, csps
        try:
            yield
        finally:
            (self._mention, self._attn, self._inject, self._capture, self._override,
             self._csps) = old

    def logits(self, input_ids, attn):
        return self.model(input_ids=input_ids, attention_mask=attn).logits

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


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


def csp_spec_for_record(rec):
    """(csp, system, tags) for the composer, from a live record. build_live_pool stashes rec['csp']
    (the organ_csp) and optional rec['sys']/rec['tags']. None if the record carries no structure."""
    csp = rec.get("csp")
    if csp is None:
        return None
    return (csp, rec.get("sys"), tuple(rec.get("tags", ())))


def _bank_batch(recs, tok, dev, two_stream):
    from .. import run_glados_staged as G
    ba = G.build_live_batch(recs, tok, dev, two_stream=two_stream)
    ba["csps"] = [csp_spec_for_record(r) for r in recs]
    return ba


# ============================================================ no-op@init proof
def verify_noop_bank(model, tok, dev, rec):
    """Bitwise no-op at init: the zero-init γ gate ⇒ the live bank-composed injection moves logits by 0."""
    enc = tok(rec["prompt"], return_offsets_mapping=True, return_tensors="pt")
    ids = enc["input_ids"].to(dev); attn = torch.ones_like(ids)
    mention = _mention_tensor([rec["mentions"]], enc["offset_mapping"], 1, rec["n"], ids.size(1), dev)
    csps = [csp_spec_for_record(rec)]
    with torch.no_grad():
        base = model.model(input_ids=ids, attention_mask=attn).logits.float()
    with torch.no_grad(), model.live(mention, attn, inject=True, capture=True, csps=csps):
        g0 = model.logits(ids, attn).float()
    noop = float((base - g0).abs().max())
    with torch.no_grad():
        saved = model.gamma.alpha.data.clone(); model.gamma.alpha.data.fill_(2.0)
        with model.live(mention, attn, inject=True, capture=True, csps=csps):
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
                with model.live(pment, pattn, inject=False, capture=True, csps=csps):
                    _ = model.logits(pids, pattn)
                surv = model._captured_surv[:, :Nmax, :].clone()
            surv = G.apply_control_ext(surv, chunk, control, K, perm)
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
            surv_exp = surv[torch.tensor(prob_of, device=dev)]
            ctx = model.live(mention, attn, inject=True, capture=False, override=surv_exp)
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
def train_bank_woven(olmo_ids, tok, dev, a, train_recs, eval_recs, two_stream=True, composer=None):
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
    model = BankWoven(peft_model, D, G.K, alpha, composer, mid, inj, gamma_hidden=a.gamma_hidden).to(dev)
    model.alpha.float(); model.gamma.float()
    noop, live = verify_noop_bank(model, tok, dev, eval_recs[0])
    from ..oracle_readout import n_trainable
    print(f"\n  BANK-WOVEN  mid {mid} -> inject {inj}/{nL}  faculties={composer.faculties()}  "
          f"trainable {n_trainable(model):,}", flush=True)
    print(f"  NO-OP @ INIT max|base-(LoRA+gate0)| = {noop:.3e} (expect ~0) | gate-on moves {live:.3e}",
          flush=True)
    lora_params = [p for n, p in model.model.named_parameters() if p.requires_grad]
    rng = np.random.default_rng(a.seed + 3)

    def batch(n):
        idxs = rng.integers(0, len(train_recs), n).tolist()
        return _bank_batch([train_recs[i] for i in idxs], tok, dev, two_stream)

    def alpha_capture(ba):
        if two_stream:
            with model.live(ba["a_mention"], ba["a_attn"], inject=False, capture=True, csps=ba["csps"]):
                _ = model.logits(ba["a_input_ids"], ba["a_attn"])
        else:
            with model.live(ba["mention"], ba["attn"], inject=False, capture=True, csps=ba["csps"]):
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
            with model.live(ba["mention"], ba["attn"], inject=True, capture=False, override=surv):
                logits = model.logits(ba["input_ids"], ba["attn"]).float()
        else:
            with model.live(ba["mention"], ba["attn"], inject=True, capture=True, csps=ba["csps"]):
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
