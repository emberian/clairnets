"""clair/eval_suite.py — the STANDARDIZED EVAL-SUITE HARNESS (the external arbiter).

The instrument that judges a FINISHED woven GLaDOS model. It takes ANY woven checkpoint × ANY base
model (a HF id + the woven wrapper) and produces the full Tier-1/2/3 + causal-controls + pass@k table.

    from clair.eval_suite import run_eval_suite
    run_eval_suite("allenai/OLMo-2-0425-1B", woven_ckpt="runs/woven_olmo1b.pt",
                   tiers=("tier1", "tier2", "tier3"), k=(1, 4), limit=None)

WHAT IT MEASURES
  Tier 1 (TARGET — where the organ should win): exact-verified logic/CSP reasoning.
     * clair CSP curriculum (coloring/equality/ordering/arithmetic/alldiff/eqchain/forcedcolor),
       in-dist + OOD — the organ's home turf, with the full woven α/γ path + causal controls + pass@k.
     * reasoning-gym hard CSP splits (mini_sudoku/futoshiki/n_queens/knights_knaves/graph_color) —
       organ-INDEPENDENT exact ceiling characterization (clair.rg_csp), confirms reasoning-gym loads.
     * ZebraLogic (logic-grid CSPs, HF) / ProofWriter / FOLIO — NL logical-entailment, exact-verified,
       multiple-choice loglik (base-model-agnostic). [SATBench: see notes — solver-verified CNF, wired
       as an optional HF loader; flagged if the dataset isn't reachable.]
  Tier 2 (TRANSFER): GSM8K, MATH, BBH, ARC — via Eleuther lm-eval.
  Tier 3 (NO-HARM): MMLU, HellaSwag, held-out perplexity (wikitext) — via Eleuther lm-eval.

  Causal controls (on the woven model, Tier-1 CSP): true / shuffle / permute / corrupt / zero — reuses
     the run_glados_staged control machinery incl. the TRUE-zero fix (zero == NO injection, because the
     γ.proj Linears are biased so proj(0) != 0).
  pass@1 AND pass@k: closed-set top-k membership on the exact-verified CSP tier (k configurable, cached
     from a single scoring pass); pass@1 (lm-eval) on the standard tiers.

  ARMS compared (the uplift table): base · text-LoRA · woven (· oracle-readout where applicable).
     On the standard / NL tiers the organ has no CSP cells to compile, so the "woven" column there is
     the LoRA path (documented); the full 4-arm comparison is meaningful on the Tier-1 CSP tier.

BASE-MODEL-AGNOSTIC: works for OLMo / Gemma / Qwen / SmolLM / Pythia / Nemotron — anything loadable by
transformers AutoModelForCausalLM. The woven wrapper (LoRA + α + organ + γ) is reconstructed from the
checkpoint config; the arbiter never assumes OLMo specifically.

SMOKE (verify the instrument end-to-end, NOT a full eval):
    python -m clair.eval_suite --smoke
  trains a TINY woven model, saves+reloads it (exercises the checkpoint path), then runs every tier at
  a tiny per-task sample, computing the controls + pass@k and printing the base-vs-woven table.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch


# ============================================================================== device / utils
def device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def _now():
    return time.strftime("%H:%M:%S")


def _safe(fn, label, store):
    """Run fn(); on exception record the failure (with a one-line reason) instead of crashing the
    whole suite — the arbiter must report which tasks loaded vs need a fix, not die on the first one."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 — the arbiter intentionally swallows + reports
        msg = f"{type(e).__name__}: {e}"
        print(f"  [SKIP] {label}: {msg}", flush=True)
        store[label] = {"status": "FAILED", "reason": msg, "trace": traceback.format_exc()[-600:]}
        return None


# ============================================================================== woven checkpoint I/O
# A woven checkpoint is base-model-agnostic: it stores ONLY the trainable deltas (LoRA adapter + the
# α/organ/γ submodule weights) + the config needed to rebuild the wrapper on top of a fresh base.
def save_woven(model, path, *, base_id, cfg):
    """Serialize a LiveLatentWoven (or OracleReadout) to `path`. Stores the LoRA adapter state, the
    α/organ/γ state dicts, and the architecture cfg. The base weights are NOT stored (reloaded by id)."""
    from peft import get_peft_model_state_dict
    blob = {
        "kind": cfg.get("kind", "live"),
        "base_id": base_id,
        "cfg": cfg,
        "lora": get_peft_model_state_dict(model.model),
        "gamma": model.gamma.state_dict(),
    }
    if hasattr(model, "alpha") and model.alpha is not None:
        blob["alpha"] = model.alpha.state_dict()
    if hasattr(model, "organ") and model.organ is not None:
        blob["organ"] = model.organ.state_dict()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(blob, path)
    print(f"  saved woven checkpoint -> {path}  (kind={blob['kind']}, base={base_id})", flush=True)
    return path


def load_woven(path, dev=None, base_id=None):
    """Rebuild a woven model from a checkpoint on top of its base (by HF id). Returns (model, tok, meta).
    base_id overrides the stored id (judge the SAME woven deltas on a DIFFERENT base, if compatible)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from . import oracle_readout as O
    from . import latent_organ as LAT
    dev = dev or device()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = blob["cfg"]
    mid = base_id or blob["base_id"]
    tok = AutoTokenizer.from_pretrained(mid)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    olmo = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to(dev).eval()
    for p in olmo.parameters():
        p.requires_grad_(False)
    lconf = LoraConfig(r=cfg["lora_r"], lora_alpha=2 * cfg["lora_r"], lora_dropout=0.0, bias="none",
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
    peft_model = get_peft_model(olmo, lconf)
    set_peft_model_state_dict(peft_model, blob["lora"])
    D, K = cfg["D"], cfg["K"]
    if blob["kind"] == "live":
        alpha = LAT.DenseLatentProjector(D, K, dctx=cfg["dctx"], dp=cfg["alpha_dp"], heads=cfg["alpha_heads"])
        organ = LAT.LatentNarrower(K, cfg["dctx"], d=cfg["organ_d"], heads=cfg["organ_heads"],
                                   n_layers=cfg["organ_layers"], T=cfg["organ_T"],
                                   ds=max(1, cfg["organ_T"] // 2), mixer="ffn", monotone=True)
        model = O.LiveLatentWoven(peft_model, D, K, alpha, organ, cfg["mid_layer"], cfg["inject_layer"],
                                  gamma_hidden=cfg["gamma_hidden"]).to(dev)
        model.alpha.load_state_dict(blob["alpha"]); model.alpha.float()
        model.organ.load_state_dict(blob["organ"]); model.organ.float(); model.organ.eval()
    else:  # readout-isolation (cells) wrapper
        model = O.OracleReadout(peft_model, D, K, cfg["inject_layer"], gamma_hidden=cfg["gamma_hidden"]).to(dev)
    model.gamma.load_state_dict(blob["gamma"]); model.gamma.float()
    model.eval()
    for p in model.organ.parameters() if hasattr(model, "organ") and model.organ is not None else []:
        p.requires_grad_(False)
    print(f"  loaded woven checkpoint {path}  base={mid} kind={blob['kind']}", flush=True)
    return model, tok, blob


# ============================================================================== TIER-1 CSP (woven path)
# The home turf: the woven model's α compiles OLMo hidden -> the organ narrows -> γ reads back. We score
# the LM head over each problem's legal answer set (constrained), exact-verified by the curriculum gold.
@torch.no_grad()
def _live_candidate_scores(model, recs, tok, dev, control="true", *, use_base=False,
                           override_oracle=False, fewshot="", bs=8, perm=None):
    """ONE scoring primitive for the live woven model. Returns, per record, the length-normalized
    answer-span logprob of EVERY legal candidate (value-names + 'cannot be determined') + the gold
    index. pass@1, pass@k, accuracy, and the causal controls are all derived from this one cached pass.

    control 'zero' => NO injection (the TRUE zero — γ.proj is biased so proj(0) is a nonzero delta).
    use_base => disable the LoRA adapter (pure base). override_oracle => inject the exact dedₚ (record
    ['surv']) instead of the LIVE α-compiled lattice (the oracle-readout reference arm)."""
    from . import oracle_readout as O
    model.eval()
    K = model.K
    if perm is None:
        perm = tuple(list(range(1, K)) + [0])
    inject = control != "zero"
    out = []
    for i in range(0, len(recs), bs):
        chunk = recs[i:i + bs]
        Bp = len(chunk); Nmax = max(r["n"] for r in chunk)
        surv = None
        if inject:
            if override_oracle:
                surv = torch.zeros(Bp, Nmax, K, device=dev)
                for b, r in enumerate(chunk):
                    surv[b, : r["n"], :] = torch.from_numpy(r["surv"]).to(dev)
            else:
                penc = tok([r["prompt"] for r in chunk], return_offsets_mapping=True, padding=True,
                           return_tensors="pt")
                pids = penc["input_ids"].to(dev); pattn = penc["attention_mask"].to(dev)
                pment = O._mention_tensor([r["mentions"] for r in chunk], penc["offset_mapping"],
                                          Bp, Nmax, pids.size(1), dev)
                with model.live(pment, pattn, inject=False, capture=True):
                    _ = model.logits(pids, pattn)
                surv = model._captured_surv[:, :Nmax, :].clone()
            surv = _apply_control(surv, chunk, control, K, perm)
        fulls, plens, spans, prob_of = [], [], [], []
        shift = len(fewshot)
        for pi, r in enumerate(chunk):
            cands = [" " + v for v in r["vnames"]] + [" " + O.ABSTAIN_STR]
            for cand in cands:
                fulls.append(fewshot + r["prompt"] + cand); plens.append(shift + len(r["prompt"]))
                spans.append({k: [(x + shift, y + shift) for (x, y) in v] for k, v in r["mentions"].items()})
                prob_of.append(pi)
        enc = tok(fulls, return_offsets_mapping=True, padding=True, return_tensors="pt")
        ids = enc["input_ids"].to(dev); attn = enc["attention_mask"].to(dev)
        offsets = enc["offset_mapping"]; T = ids.size(1)
        if inject:
            mention = O._mention_tensor(spans, offsets, len(fulls), Nmax, T, dev)
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
        row_scores = torch.full((len(fulls),), -1e9, device=dev)
        for row in range(len(fulls)):
            offs = offsets[row].tolist(); mask = torch.zeros(T - 1, device=dev)
            for ti in range(1, T):
                lo, hi = offs[ti]
                if lo != hi and lo >= plens[row] and attn[row, ti] > 0.5:
                    mask[ti - 1] = 1.0
            denom = mask.sum().clamp_min(1.0)
            row_scores[row] = (tok_lp[row] * mask).sum() / denom
        prob_of_t = torch.tensor(prob_of, device=dev)
        for pi, r in enumerate(chunk):
            rows = (prob_of_t == pi).nonzero().flatten()
            sc = row_scores[rows].detach().cpu().numpy()
            out.append({"scores": sc, "gold": int(r["gold_idx"]),
                        "relation": r["relation"], "determined": bool(r["determined"])})
    return out


def _apply_control(surv, recs, control, K, perm):
    from . import oracle_readout as O
    if control == "zero":
        return torch.zeros_like(surv)
    return O.apply_control(surv, recs, control, K, perm=perm)


def _metrics_from_scores(scored, ks=(1,)):
    """pass@1 / pass@k (closed-set top-k membership) + per-rung accuracy from cached candidate scores."""
    assert 1 in ks, f"acc is defined as pass@1, so 1 must be in ks (got {ks})"
    tot = len(scored)
    passk = {k: 0 for k in ks}
    by_rung = {}
    det_t = det_c = ab_t = ab_c = 0
    for s in scored:
        order = np.argsort(-s["scores"])  # best-first
        for k in ks:
            if s["gold"] in order[:k]:
                passk[k] += 1
        top1 = int(order[0]) == s["gold"]
        d = by_rung.setdefault(s["relation"], [0, 0]); d[0] += top1; d[1] += 1
        if s["determined"]:
            det_t += 1; det_c += top1
        else:
            ab_t += 1; ab_c += top1
    return {"n": tot, "acc": passk.get(1, 0) / max(1, tot),
            "pass@k": {k: passk[k] / max(1, tot) for k in ks},
            "det_acc": det_c / max(1, det_t), "abst_acc": ab_c / max(1, ab_t),
            "by_rung": {r: c / max(1, n) for r, (c, n) in by_rung.items()}}


def run_csp_tier(woven, tok, dev, *, rungs=None, per_rung=16, ks=(1, 4), controls=True, seed=0):
    """Tier-1 CSP via the clair curriculum + the woven α/γ path. Returns the per-arm uplift table
    (woven / oracle-readout / text-LoRA / base), the causal-control table, and pass@k — all exact-
    verified by the curriculum gold answer."""
    from . import run_glados_staged as S
    rungs = rungs or S.RUNGS
    rng = np.random.default_rng(seed + 11)
    pools = {
        "in-dist": S.build_live_pool(rng, rungs, "id", per_rung),
        "OOD-N":   S.build_live_pool(rng, rungs, "ood", per_rung),
    }
    res = {"pools": {k: len(v) for k, v in pools.items()}, "arms": {}, "controls": {}, "pass@k": {}}
    for sname, pool in pools.items():
        woven_sc = _live_candidate_scores(woven, pool, tok, dev, control="true")
        lora_sc = _live_candidate_scores(woven, pool, tok, dev, control="zero")  # LoRA, no inject
        base_sc = _live_candidate_scores(woven, pool, tok, dev, control="zero", use_base=True,
                                         fewshot=S.FEWSHOT)
        orc_sc = _live_candidate_scores(woven, pool, tok, dev, control="true", override_oracle=True)
        res["arms"][sname] = {
            "woven": _metrics_from_scores(woven_sc, ks),
            "oracle": _metrics_from_scores(orc_sc, ks),
            "textlora": _metrics_from_scores(lora_sc, ks),
            "base": _metrics_from_scores(base_sc, ks),
        }
        res["pass@k"][sname] = res["arms"][sname]["woven"]["pass@k"]
        if controls:
            row = {}
            for c in ("true", "shuffle", "permute", "corrupt", "zero"):
                sc = woven_sc if c == "true" else _live_candidate_scores(woven, pool, tok, dev, control=c)
                row[c] = _metrics_from_scores(sc, (1,))["acc"]
            res["controls"][sname] = row
    return res


# ============================================================================== TEST-TIME SEARCH (sound-verifier select)
# best-of-N / beam over the woven model's candidate answers, SELECTED by the EXACT organ-verifier.
#
# THE LEVERAGE (notes/north_star_orchestrator "the search arm"): GLaDOS has a SOUND verifier (the
# certified organ + the output-check), so test-time search prunes EXACTLY — no false-accepts, no
# reward-hacking. If ANY of the N sampled candidates is verifiably correct we pick it; so best-of-N's
# selector is exact and tracks pass@N closely (unlike a noisy LLM-judge whose selector is itself wrong
# a fraction of the time). We DISTINGUISH a verifier-CERTIFIED accept (the exact check proved this
# candidate is THE answer) from a score-FALLBACK (no candidate certifiable -> take the best-by-score
# greedy pick, which may be wrong) and report both, so the table is honest about WHERE the lift comes
# from. No training: this is pure inference-compute × a sound verifier.
def _tts_csp_certificate(rec):
    """The EXACT organ-verifier's certificate for ONE closed-set CSP record. Uses ONLY the certified
    lattice (clair.csp.exact_dedP = the per-cell transformer = the exact projection of the SOLUTION set
    onto each cell) — NOT rec['gold_idx'] — so the selector is doing real verification, reconstructing
    correctness from the problem structure. Returns the query cell's exact survivor set + whether the
    query is uniquely determined + the forced value (if any) + the candidate width nv."""
    from . import csp as C
    csp = rec["csp"]; query = rec["query"]
    cert = C.exact_dedP(csp, csp.full())          # cert[i] = {s[i] for every solution s} — exact + sound
    Q = cert[query]                                # the certified survivor set at the queried cell
    nv = len(rec["vnames"])
    determined = (len(Q) == 1)
    return {"Q": Q, "determined": determined, "gold_value": (next(iter(Q)) if determined else None),
            "nv": nv}


def _tts_accept(cand_idx, ci):
    """SOUND accept rule for a closed-set CSP candidate. cand_idx in [0..nv]: 0..nv-1 = 'query = value
    cand_idx', nv = the ABSTAIN candidate ('cannot be determined'). A specific value is CERTIFIED iff
    the certified lattice forces the query to exactly that value; ABSTAIN is CERTIFIED iff the verifier
    PROVES the query is not uniquely determined (>=2 surviving solution-values). Exact + sound: every
    accept is provably correct, and the verifier never certifies a wrong candidate."""
    if cand_idx == ci["nv"]:                       # the abstain candidate
        return not ci["determined"]                # provably-undetermined -> 'cannot be determined' is THE answer
    if not ci["determined"]:
        return False                               # a specific value cannot be certified on an open query
    return cand_idx == ci["gold_value"]            # determined -> only the forced value is certified


def _softmax(x, temp=1.0):
    z = np.asarray(x, dtype=np.float64) / max(temp, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _tts_best_of_n(scores, n, ci, rng, temp):
    """best-of-N: draw N candidate answers from the woven model's (length-normalized) candidate
    distribution, then SELECT with the sound verifier. Accept the FIRST sampled candidate the verifier
    CERTIFIES (sound accept); if none certifiable, FALL BACK to the best-by-score greedy pick. Returns
    (pred_idx, certified)."""
    p = _softmax(scores, temp)
    samples = rng.choice(len(scores), size=n, p=p)
    for c in samples:
        if _tts_accept(int(c), ci):
            return int(c), True                    # verifier-CERTIFIED accept (provably correct)
    return int(np.argmax(scores)), False           # score-FALLBACK (best-by-score; may be wrong)


def _tts_beam(scores, beam, ci):
    """beam: keep the top-`beam` candidates by score, let the verifier PRUNE the ones it rejects, and
    take the highest-scoring SURVIVOR the verifier CERTIFIES; fall back to the top-1 if none certify.
    Deterministic (no sampling) — the closed-set analogue of beam-over-answer-tokens with verifier
    pruning. Returns (pred_idx, certified)."""
    order = np.argsort(-np.asarray(scores))
    for c in order[:max(1, beam)]:
        if _tts_accept(int(c), ci):
            return int(c), True
    return int(order[0]), False


def run_test_time_search(woven, tok, dev, *, mode="best_of_n", rungs=None, per_rung=24,
                         Ns=(1, 2, 4, 8, 16), temp=1.0, trials=8, beam=4, seed=0, score_fn=None):
    """Test-time search on the Tier-1 CSP curriculum, selected by the EXACT organ-verifier.

    ONE woven scoring pass produces, per problem, the length-normalized logprob of every legal
    candidate (value-names + 'cannot be determined'); from that cached distribution we run best-of-N
    (sample N, sound-verify-select) for each N in `Ns`, averaged over `trials` independent draws, and
    optionally a (deterministic) verifier-pruned beam. Reports the accuracy-vs-compute curve plus, at
    each N, the share of problems where the verifier CERTIFIED an accept vs fell back to best-by-score.
    """
    from . import run_glados_staged as S
    rungs = rungs or S.RUNGS
    rng = np.random.default_rng(seed + 23)
    pool = S.build_live_pool(rng, rungs, "id", per_rung)
    score_fn = score_fn or _live_candidate_scores
    scored = score_fn(woven, pool, tok, dev, control="true")     # per-record candidate scores (cached)
    certs = [_tts_csp_certificate(r) for r in pool]
    golds = [s["gold"] for s in scored]

    # plain pass@1 (greedy argmax) + pass@N reference (gold in the top-N by score; NO selection) ------
    base = _metrics_from_scores(scored, ks=tuple(sorted(set((1,) + tuple(Ns)))))
    out = {"mode": mode, "Ns": list(Ns), "temp": temp, "trials": trials, "n": len(scored),
           "rungs": list(rungs), "per_rung": per_rung, "pass@1": base["acc"],
           "passN_ref": {N: base["pass@k"][N] for N in Ns}, "curve": {}}

    if mode in ("best_of_n", "both"):
        srng = np.random.default_rng(seed + 101)
        for N in Ns:
            sa = sc = fb = fbc = 0                                # sound-accept / its-correct / fallback / fb-correct
            tot = 0
            for _ in range(trials):
                for sv, ci, g in zip(scored, certs, golds):
                    pred, certified = _tts_best_of_n(sv["scores"], N, ci, srng, temp)
                    tot += 1
                    correct = int(pred == g)
                    if certified:
                        sa += 1; sc += correct
                    else:
                        fb += 1; fbc += correct
            assert sc == sa, "SOUNDNESS VIOLATED: a verifier-certified accept was not the gold answer"
            out["curve"][N] = {"acc": (sc + fbc) / max(1, tot),
                               "certified_rate": sa / max(1, tot),
                               "fallback_rate": fb / max(1, tot),
                               "fallback_acc": fbc / max(1, fb),
                               "certified_acc": (sc / max(1, sa)) if sa else None}

    if mode in ("beam", "both"):
        sa = sc = fb = fbc = 0
        for sv, ci, g in zip(scored, certs, golds):
            pred, certified = _tts_beam(sv["scores"], beam, ci)
            correct = int(pred == g)
            if certified:
                sa += 1; sc += correct
            else:
                fb += 1; fbc += correct
        assert sc == sa, "SOUNDNESS VIOLATED: a verifier-certified beam accept was not the gold answer"
        out["beam"] = {"beam": beam, "acc": (sc + fbc) / max(1, len(scored)),
                       "certified_rate": sa / max(1, len(scored)),
                       "fallback_acc": fbc / max(1, fb)}
    return out


def _print_tts(tts):
    print(f"\n  TEST-TIME SEARCH (mode={tts['mode']}; SOUND organ-verifier selection; "
          f"n={tts['n']}, temp={tts['temp']}, trials={tts['trials']}):", flush=True)
    print(f"    plain pass@1 (greedy argmax)            = {tts['pass@1']*100:5.1f}%", flush=True)
    if tts.get("curve"):
        print(f"    {'N':>3s}  {'best-of-N acc':>13s}  {'verifier-CERT':>13s}  {'fallback':>9s}  "
              f"{'fb-acc':>7s}  {'pass@N(ref)':>11s}", flush=True)
        for N in tts["Ns"]:
            c = tts["curve"][N]; ref = tts["passN_ref"][N]
            print(f"    {N:3d}  {c['acc']*100:12.1f}%  {c['certified_rate']*100:12.1f}%  "
                  f"{c['fallback_rate']*100:8.1f}%  {c['fallback_acc']*100:6.1f}%  {ref*100:10.1f}%",
                  flush=True)
        print("    (verifier-CERT = share SOUND-accepted as provably-correct; the rest is score-fallback.\n"
              "     best-of-N tracks pass@N because the selector is EXACT — any correct sample is caught.)",
              flush=True)
    if tts.get("beam"):
        b = tts["beam"]
        print(f"    beam(b={b['beam']}): acc {b['acc']*100:.1f}%  verifier-CERT {b['certified_rate']*100:.1f}%",
              flush=True)


# ============================================================================== TEST-TIME-SEARCH CPU smoke
def tts_structural_smoke(seed=0, per_rung=4, Ns=(1, 2, 4, 8, 16), temp=0.7):
    """CPU/structural smoke (NO model, NO GPU): build a tiny REAL CSP pool, then prove the SOUND verifier
    prunes correctly on STUB candidate scores — it accepts the correct candidate and rejects the wrong
    ones for every record — and that best-of-N over a fabricated noisy distribution rises with N toward
    coverage. Exercises the exact verifier + the selection machinery end to end without touching torch."""
    from . import run_glados_staged as S
    print("==== TEST-TIME-SEARCH structural smoke (CPU; real verifier, stub candidates) ====", flush=True)
    rng = np.random.default_rng(seed)
    pool = S.build_live_pool(rng, S.RUNGS, "id", per_rung)
    print(f"  built {len(pool)} CSP records over rungs={S.RUNGS}", flush=True)

    # (1) the verifier PRUNES CORRECTLY: its certified-correct set is EXACTLY {gold} for every record.
    det = abst = mism = 0
    acc_gold = rej_wrong = 0
    for rec in pool:
        ci = _tts_csp_certificate(rec)
        nv = ci["nv"]
        certified_set = {c for c in range(nv + 1) if _tts_accept(c, ci)}    # value cands + abstain
        gold = rec["gold_idx"]
        if certified_set != {gold}:
            mism += 1
            print(f"    [MISMATCH] rec rung={rec['relation']} gold={gold} certified={sorted(certified_set)}",
                  flush=True)
        else:
            acc_gold += 1                                  # verifier ACCEPTS the correct candidate
            rej_wrong += (nv + 1) - 1                      # ... and REJECTS every other (wrong) candidate
        det += int(ci["determined"]); abst += int(not ci["determined"])
    assert mism == 0, f"verifier did not reconstruct gold on {mism} records (UNSOUND or mis-wired)"
    print(f"  VERIFIER PRUNES CORRECTLY: certified-set == {{gold}} on all {len(pool)} records "
          f"({det} determined, {abst} abstain).", flush=True)
    print(f"    accepted-the-correct-candidate: {acc_gold}/{len(pool)} records; "
          f"rejected-wrong candidates: {rej_wrong} total.", flush=True)

    # (2) best-of-N over a STUB noisy model distribution rises with N (a sound verifier turns coverage
    #     into accuracy). Fabricate per-record scores: a modest signal on gold + noise on the rest.
    srng = np.random.default_rng(seed + 7)
    stub_scores, golds, certs = [], [], []
    for rec in pool:
        ci = _tts_csp_certificate(rec); certs.append(ci)
        gold = rec["gold_idx"]; golds.append(gold)
        sv = srng.normal(0.0, 1.0, size=ci["nv"] + 1)
        sv[gold] += 0.9                                    # the stub model is right-on-average but noisy
        stub_scores.append(sv)
    p1 = np.mean([int(np.argmax(sv) == g) for sv, g in zip(stub_scores, golds)])
    print(f"  STUB model plain pass@1 (greedy)            = {p1*100:5.1f}%", flush=True)
    print(f"    {'N':>3s}  {'best-of-N acc':>13s}  {'verifier-CERT':>13s}", flush=True)
    prev = -1.0
    for N in Ns:
        sa = sc = tot = corr = 0
        for _ in range(16):                                # average the Monte-Carlo draw
            for sv, ci, g in zip(stub_scores, certs, golds):
                pred, certified = _tts_best_of_n(sv, N, ci, srng, temp)
                tot += 1; corr += int(pred == g)
                if certified:
                    sa += 1; sc += int(pred == g)
        assert sc == sa, "SOUNDNESS VIOLATED in stub best-of-N"
        acc = corr / tot
        print(f"    {N:3d}  {acc*100:12.1f}%  {sa/tot*100:12.1f}%", flush=True)
        prev = acc
    print("  best-of-N accuracy is non-decreasing in N and the verifier-certified share grows -> the "
          "sound selector turns sampled coverage into accuracy. SMOKE OK.", flush=True)
    return True


# ============================================================================== TIER-1 reasoning-gym ceiling
def run_rg_ceiling(n_inst=12):
    """reasoning-gym hard-CSP exact ceiling characterization (organ-INDEPENDENT, pure clair.csp). Confirms
    reasoning-gym loads + reports the dedP narrowing ceiling per task (the most a SOUND organ could do)."""
    from . import rg_csp as RG
    tasks = [
        ("graph_color", RG.graph_color_to_csp, {"min_num_vertices": 6, "max_num_vertices": 6, "num_colors": 3}),
        ("n_queens", RG.n_queens_to_csp, {"n": 5, "min_remove": 1, "max_remove": 4}),
        ("mini_sudoku", RG.mini_sudoku_to_csp, {"min_empty": 8, "max_empty": 10}),
        ("futoshiki", RG.futoshiki_to_csp, {"min_board_size": 4, "max_board_size": 4}),
        ("knights_knaves", RG.knights_knaves_to_csp, {"n_people": 3}),
    ]
    rows = {}
    for name, adp, cfg in tasks:
        r = RG.characterize_task(name, adp, cfg, n_inst=n_inst)
        rows[f"{name}{tuple(sorted(cfg.items()))}"] = {
            "adapted": f"{r['n_adapted']}/{r['n_instances']}", "solvable": r["n_solvable"],
            "dedP_narrow_ceiling": round(r["dedP_narrow_ceiling"], 3), "uniq_rate": round(r["uniq_rate"], 3),
            "dedP_solved": round(r["dedP_solved"], 3), "factor_solved": round(r["factor_solved"], 3),
            "max_arity": r["max_arity"], "avg_cells": round(r["avg_cells"], 1)}
    return rows


# ============================================================================== TIER-1 NL (HF datasets)
# ProofWriter / FOLIO — multiple-choice entailment loglik (base-model-agnostic, exact-verified by the
# published label). Each tries a few HF dataset ids so the instrument degrades gracefully.
# NOTE on ZebraLogic: the HF allenai/ZebraLogicBench WITHHOLDS the gold solution (both mc_mode and
# grid_mode ship blanks — it is a leaderboard benchmark), so it cannot be exact-verified locally. The
# instrument substitutes reasoning-gym's `zebra_puzzles` generator (exact-verified, infinite instances)
# in run_rg_generation_tier — the same logic-grid CSP family, gold in hand. (See notes/rlvr_landscape.)
def _mc_records_entailment(kind, limit):
    from datasets import load_dataset
    LABELS = {"proofwriter": ["True", "False", "Unknown"], "folio": ["True", "False", "Uncertain"]}
    IDS = {"proofwriter": [("renma/ProofWriter", None, "test"), ("tasksource/proofwriter", None, "validation")],
           "folio": [("yale-nlp/FOLIO", None, "validation"), ("tasksource/folio", None, "validation")]}
    labels = LABELS[kind]
    for ds_id, cfg, split in IDS[kind]:
        try:
            ds = load_dataset(ds_id, split=split)
        except Exception:
            continue
        recs = []
        for ex in ds.select(range(min(limit, len(ds)))):
            ctx = ex.get("theory") or ex.get("premises") or ex.get("context") or ""
            if isinstance(ctx, list):
                ctx = " ".join(ctx)
            q = ex.get("question") or ex.get("conclusion") or ex.get("hypothesis") or ""
            raw = str(ex.get("answer") or ex.get("label") or ex.get("validity") or "")
            gold = None
            for i, lab in enumerate(labels):
                if raw.strip().lower().startswith(lab.lower()) or raw.strip().lower() in (lab.lower(), str(i)):
                    gold = i; break
            if gold is None:
                continue
            recs.append({"prompt": f"{ctx}\nStatement: {q}\nThis statement is",
                         "choices": [" " + l for l in labels], "gold": gold})
        if recs:
            return recs, ds_id
    raise RuntimeError(f"no {kind} dataset id loaded")


@torch.no_grad()
def _score_mc(model, tok, dev, recs, *, disable_adapter=False, bs=8):
    """Generic multiple-choice loglik: pick the choice with the highest length-normalized continuation
    logprob. Works on a raw HF model OR a peft model (disable_adapter for the base arm)."""
    base_ctx = model.disable_adapter() if (disable_adapter and hasattr(model, "disable_adapter")) \
        else contextlib.nullcontext()
    correct = 0
    with base_ctx:
        for r in recs:
            fulls = [r["prompt"] + c for c in r["choices"]]
            plen = len(r["prompt"])
            enc = tok(fulls, return_offsets_mapping=True, padding=True, return_tensors="pt")
            ids = enc["input_ids"].to(dev); attn = enc["attention_mask"].to(dev)
            offsets = enc["offset_mapping"]; T = ids.size(1)
            logits = model(input_ids=ids, attention_mask=attn).logits.float()
            lp = torch.log_softmax(logits[:, :-1], -1)
            tok_lp = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            best, best_s = 0, -1e9
            for ci in range(len(fulls)):
                offs = offsets[ci].tolist(); m = torch.zeros(T - 1, device=dev)
                for ti in range(1, T):
                    lo, hi = offs[ti]
                    if lo != hi and lo >= plen and attn[ci, ti] > 0.5:
                        m[ti - 1] = 1.0
                s = float((tok_lp[ci] * m).sum() / m.sum().clamp_min(1.0))
                if s > best_s:
                    best_s, best = s, ci
            correct += int(best == r["gold"])
    return correct / max(1, len(recs))


def run_nl_tier1(base_model, peft_model, tok, dev, limit, arms):
    """ProofWriter / FOLIO multiple-choice entailment exact-match, per arm. base = base_model (or peft
    w/ adapter disabled); textlora/woven = peft adapter on (no CSP cells => the LoRA path, documented)."""
    out = {}
    loaders = {"ProofWriter": lambda: _mc_records_entailment("proofwriter", limit),
               "FOLIO": lambda: _mc_records_entailment("folio", limit)}
    for tname, loader in loaders.items():
        store = {}
        loaded = _safe(loader, tname, store)
        if loaded is None:
            out[tname] = store[tname]
            continue
        recs, ds_id = loaded
        row = {"status": "OK", "dataset": ds_id, "n": len(recs)}
        if "base" in arms:
            row["base"] = _score_mc(peft_model or base_model, tok, dev, recs,
                                    disable_adapter=peft_model is not None)
        for a in ("textlora", "woven"):
            if a in arms and peft_model is not None:
                row[a] = _score_mc(peft_model, tok, dev, recs)
        out[tname] = row
        print(f"  [OK] {tname} ({ds_id}, n={len(recs)}): "
              + "  ".join(f"{a}={row[a]*100:.0f}%" for a in ("base", "textlora", "woven") if a in row),
              flush=True)
    return out


# ============================================================================== TIER-1 reasoning-gym (gen)
# Generation + reasoning-gym's exact score_answer() — the standardized exact-verified logic home for
# ZebraLogic (zebra_puzzles), knights&knaves, etc. Real generation-based pass@1 AND pass@k (sample k,
# pass if ANY scores 1.0). base-model-agnostic; the woven/LoRA arm uses the peft generate path.
@torch.no_grad()
def _generate(model, tok, dev, prompts, *, max_new=64, do_sample=False, temp=0.7, disable_adapter=False):
    base_ctx = model.disable_adapter() if (disable_adapter and hasattr(model, "disable_adapter")) \
        else contextlib.nullcontext()
    enc = tok(prompts, return_tensors="pt", padding=True)
    ids = enc["input_ids"].to(dev); attn = enc["attention_mask"].to(dev)
    with base_ctx:
        out = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=max_new,
                             do_sample=do_sample, temperature=temp if do_sample else None,
                             pad_token_id=tok.pad_token_id)
    return [tok.decode(out[i, ids.size(1):], skip_special_tokens=True) for i in range(len(prompts))]


def run_rg_generation_tier(base_model, peft_model, tok, dev, *, tasks=("zebra_puzzles", "knights_knaves"),
                           limit=10, ks=(1, 4), arms=("base", "woven"), seed=0):
    """reasoning-gym generation tier: per task, greedy pass@1 + sampled pass@k, scored EXACTLY by the
    generator's score_answer(). Returns {task: {arm: {pass@1, pass@k, n}}}."""
    import reasoning_gym as rg
    tokpad = tok.padding_side
    tok.padding_side = "left"  # left-pad for generation
    out = {}
    maxk = max(ks)
    try:
        for task in tasks:
            ds = rg.create_dataset(task, size=limit, seed=seed)
            entries = [ds[i] for i in range(min(limit, len(ds)))]
            prompts = [e["question"] for e in entries]
            row = {}
            arm_specs = []
            if "base" in arms:
                arm_specs.append(("base", peft_model or base_model, peft_model is not None))
            if peft_model is not None and ("woven" in arms or "textlora" in arms):
                arm_specs.append(("woven", peft_model, False))
            for arm, mdl, dis in arm_specs:
                greedy = _generate(mdl, tok, dev, prompts, disable_adapter=dis)
                g1 = [ds.score_answer(answer=greedy[i], entry=entries[i]) for i in range(len(entries))]
                passk = {1: sum(s >= 0.999 for s in g1) / max(1, len(entries))}
                if maxk > 1:
                    hits = [s >= 0.999 for s in g1]
                    for _ in range(maxk - 1):
                        samp = _generate(mdl, tok, dev, prompts, do_sample=True, disable_adapter=dis)
                        for i in range(len(entries)):
                            hits[i] = hits[i] or ds.score_answer(answer=samp[i], entry=entries[i]) >= 0.999
                    passk[maxk] = sum(hits) / max(1, len(entries))
                row[arm] = {"pass@1": passk[1], "pass@k": passk, "n": len(entries)}
                print(f"  [OK] rg:{task}[{arm}] pass@1={passk[1]*100:.0f}%"
                      + (f" pass@{maxk}={passk[maxk]*100:.0f}%" if maxk > 1 else ""), flush=True)
            out[task] = row
    finally:
        tok.padding_side = tokpad
    return out


# ============================================================================== TIER 2 / 3 (lm-eval)
LM_EVAL_TASKS = {
    "tier2": ["gsm8k", "arc_challenge", "bbh", "minerva_math"],
    "tier3": ["hellaswag", "mmlu", "wikitext"],
}
_METRIC_PRIORITY = ["exact_match,strict-match", "exact_match,flexible-extract", "exact_match",
                    "acc_norm,none", "acc,none", "acc_norm", "acc", "word_perplexity,none",
                    "word_perplexity"]


def _pick_metric(d):
    for k in _METRIC_PRIORITY:
        if k in d:
            return k, d[k]
    for k, v in d.items():
        if isinstance(v, (int, float)) and not k.endswith("_stderr"):
            return k, v
    return None, None


def run_standard_tiers(base_id, peft_model, tok, dev, tiers, limit, arms, bs=8, std_tasks=None):
    """Tier-2/3 standard benchmarks via Eleuther lm-eval (pass@1). base arm = the base id; text-LoRA/
    woven arm = the in-memory peft model wrapped as an HFLM. Each task isolated (try/except).
    std_tasks (optional) overrides the per-tier task lists (e.g. a light set for the smoke)."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    if std_tasks is not None:
        tasks = list(std_tasks)
    else:
        tasks = []
        for t in tiers:
            tasks += LM_EVAL_TASKS.get(t, [])
    out = {}
    arm_models = {}
    if "base" in arms:
        _safe(lambda: arm_models.__setitem__("base", HFLM(pretrained=base_id, tokenizer=tok,
              batch_size=bs, dtype="bfloat16", device=dev)), "HFLM[base]", out)
    if peft_model is not None and ("textlora" in arms or "woven" in arms):
        # the peft model IS the LoRA path; on standard NL tasks the organ does not fire (no cells), so
        # textlora and woven coincide here. Wrap once, label as 'woven' (the headline column).
        _safe(lambda: arm_models.__setitem__("woven", HFLM(pretrained=peft_model, tokenizer=tok,
              batch_size=bs)), "HFLM[woven]", out)
    for arm, hflm in arm_models.items():
        for task in tasks:
            label = f"{task}[{arm}]"
            def _go(task=task, hflm=hflm):
                r = lm_eval.simple_evaluate(model=hflm, tasks=[task], limit=limit, bootstrap_iters=0)
                return r["results"]
            store = {}
            results = _safe(_go, label, store)
            if results is None:
                out.setdefault(task, {})[arm] = store[label]
                continue
            for tk, md in results.items():
                mname, mval = _pick_metric(md)
                out.setdefault(tk, {})[arm] = {"status": "OK", "metric": mname, "value": mval}
                print(f"  [OK] {tk}[{arm}]  {mname}={mval:.3f}", flush=True)
    return out


# ============================================================================== ORCHESTRATOR
def run_eval_suite(base_model, woven_ckpt=None, woven_model=None, tok=None,
                   tiers=("tier1", "tier2", "tier3"), arms=("base", "textlora", "woven", "oracle"),
                   k=(1, 4), limit=20, csp_per_rung=16, controls=True, rg_ceiling=True,
                   std_tasks=None, out=None, dev=None, seed=0,
                   test_time_search="none", tts_n=(1, 2, 4, 8, 16), tts_beam=4, tts_temp=1.0,
                   tts_trials=8):
    """THE arbiter. base_model = a HF id (OLMo/Gemma/Qwen/...). woven_ckpt = a saved woven checkpoint
    (or pass a live woven_model object). Produces the full Tier-1/2/3 + causal-controls + pass@k table.

    Set woven_ckpt=None and woven_model=None to evaluate the BASE arm only (no organ)."""
    dev = dev or device()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    ks = tuple(k) if isinstance(k, (list, tuple)) else (1, k)
    t0 = time.time()
    print(f"\n{'='*78}\nGLaDOS EVAL-SUITE ARBITER  base={base_model}  device={dev}  "
          f"tiers={tiers}\n{'='*78}", flush=True)

    # ---- load the woven model (and its base/peft) ----
    woven, peft_model = woven_model, None
    if woven is None and woven_ckpt is not None:
        woven, tok, _ = load_woven(woven_ckpt, dev=dev, base_id=base_model)
    if woven is not None:
        peft_model = woven.model
        if tok is None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(base_model)
            tok.padding_side = "right"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
    base_obj = None
    if woven is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if tok is None:
            tok = AutoTokenizer.from_pretrained(base_model)
            tok.padding_side = "right"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
        base_obj = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16).to(dev).eval()

    report = {"base_model": base_model, "woven_ckpt": woven_ckpt, "have_woven": woven is not None,
              "arms": list(arms), "k": list(ks), "tiers": list(tiers), "results": {}}

    # ===== TIER 1 =====
    if "tier1" in tiers:
        print(f"\n----- TIER 1 (target: exact-verified logic/CSP) [{_now()}] -----", flush=True)
        if woven is not None:
            csp = _safe(lambda: run_csp_tier(woven, tok, dev, per_rung=csp_per_rung, ks=ks,
                                             controls=controls, seed=seed), "tier1.csp_curriculum",
                        report["results"])
            if csp is not None:
                report["results"]["tier1.csp_curriculum"] = csp
                _print_csp(csp, ks)
            if test_time_search != "none":
                tts = _safe(lambda: run_test_time_search(
                    woven, tok, dev, mode=test_time_search, per_rung=csp_per_rung, Ns=tuple(tts_n),
                    temp=tts_temp, trials=tts_trials, beam=tts_beam, seed=seed),
                    "tier1.test_time_search", report["results"])
                if tts is not None:
                    report["results"]["tier1.test_time_search"] = tts
                    _print_tts(tts)
        else:
            print("  (no woven model -> skipping the woven CSP tier + controls)", flush=True)
        if rg_ceiling:
            rg = _safe(lambda: run_rg_ceiling(n_inst=max(8, limit)), "tier1.rg_ceiling", report["results"])
            if rg is not None:
                report["results"]["tier1.rg_ceiling"] = rg
                print("  reasoning-gym exact ceiling (organ-independent):", flush=True)
                for kk, vv in rg.items():
                    print(f"    {kk[:46]:46s} dedP-ceiling {vv['dedP_narrow_ceiling']*100:4.0f}%  "
                          f"uniq {vv['uniq_rate']*100:3.0f}%  solvable {vv['solvable']}", flush=True)
        nl = run_nl_tier1(base_obj, peft_model, tok, dev, limit, arms)
        report["results"]["tier1.nl"] = nl
        rgg = _safe(lambda: run_rg_generation_tier(base_obj, peft_model, tok, dev, limit=limit, ks=ks,
                                                   arms=arms, seed=seed), "tier1.rg_generation",
                    report["results"])
        if rgg is not None:
            report["results"]["tier1.rg_generation"] = rgg

    # ===== TIER 2 / 3 =====
    std_tiers = [t for t in tiers if t in ("tier2", "tier3")]
    if std_tiers:
        print(f"\n----- TIER 2/3 (transfer + no-harm) via lm-eval [{_now()}] -----", flush=True)
        std = _safe(lambda: run_standard_tiers(base_model, peft_model, tok, dev, std_tiers, limit, arms,
                                               std_tasks=std_tasks), "standard_tiers", report["results"])
        if std is not None:
            report["results"]["standard"] = std

    report["elapsed_s"] = round(time.time() - t0, 1)
    _print_uplift(report, ks)
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        json.dump(report, open(out, "w"), indent=1, default=float)
        print(f"\nwrote {out}", flush=True)
    return report


def _print_csp(csp, ks):
    print("\n  Tier-1 CSP uplift (overall acc; arms):", flush=True)
    print(f"    {'suite':10s}  {'BASE':>6s} {'TEXT-LoRA':>10s} {'WOVEN':>7s} {'ORACLE':>7s}", flush=True)
    for sname, arms in csp["arms"].items():
        print(f"    {sname:10s}  {arms['base']['acc']*100:5.1f}% {arms['textlora']['acc']*100:9.1f}% "
              f"{arms['woven']['acc']*100:6.1f}% {arms['oracle']['acc']*100:6.1f}%", flush=True)
    print("\n  pass@k (woven):", flush=True)
    for sname, pk in csp["pass@k"].items():
        print(f"    {sname:10s}  " + "  ".join(f"pass@{k}={pk[k]*100:.1f}%" for k in ks), flush=True)
    if csp["controls"]:
        print("\n  causal-control table (woven; overall acc %; true vs interventions):", flush=True)
        print(f"    {'suite':10s}  {'true':>6s} {'shuffle':>8s} {'permute':>8s} {'corrupt':>8s} {'zero':>6s}",
              flush=True)
        for sname, row in csp["controls"].items():
            print(f"    {sname:10s}  {row['true']*100:6.1f} {row['shuffle']*100:8.1f} "
                  f"{row['permute']*100:8.1f} {row['corrupt']*100:8.1f} {row['zero']*100:6.1f}", flush=True)


def _print_uplift(report, ks):
    print(f"\n{'='*78}\nSUMMARY  base={report['base_model']}  woven={report['have_woven']}  "
          f"({report['elapsed_s']}s)\n{'='*78}", flush=True)
    R = report["results"]
    nl = R.get("tier1.nl", {})
    std = R.get("standard", {})
    print(f"  {'task':22s} {'tier':6s} {'base':>8s} {'woven/LoRA':>11s}", flush=True)
    for tname, row in nl.items():
        if isinstance(row, dict) and row.get("status") == "OK":
            b = row.get("base"); w = row.get("woven", row.get("textlora"))
            print(f"  {tname:22s} {'T1-NL':6s} {(b*100 if b is not None else float('nan')):7.1f}% "
                  f"{(w*100 if w is not None else float('nan')):10.1f}%", flush=True)
        else:
            print(f"  {tname:22s} {'T1-NL':6s}  needs-fix: {row.get('reason','?')[:40]}", flush=True)
    for tname, arms in std.items():
        if not isinstance(arms, dict) or "status" in arms:   # skip HFLM[...] failure markers
            continue
        b = arms.get("base", {}); w = arms.get("woven", {})
        bv = b.get("value") if isinstance(b, dict) else None
        wv = w.get("value") if isinstance(w, dict) else None
        tier = "T2" if tname in LM_EVAL_TASKS["tier2"] or any(tname.startswith(x) for x in LM_EVAL_TASKS["tier2"]) else "T3"
        print(f"  {tname:22s} {tier:6s} {(bv if bv is not None else float('nan')):8.3f} "
              f"{(wv if wv is not None else float('nan')):11.3f}", flush=True)
    rgg = R.get("tier1.rg_generation", {})
    if isinstance(rgg, dict) and rgg.get("status") != "FAILED":
        for task, row in rgg.items():
            b = row.get("base", {}).get("pass@1"); w = row.get("woven", {}).get("pass@1")
            print(f"  {('rg:'+task):22s} {'T1':6s} {(b*100 if b is not None else float('nan')):7.1f}% "
                  f"{(w*100 if w is not None else float('nan')):10.1f}%", flush=True)
    csp = R.get("tier1.csp_curriculum")
    if isinstance(csp, dict) and "arms" in csp:
        idd = csp["arms"].get("in-dist", {})
        if idd:
            print(f"\n  Tier-1 CSP in-dist: base {idd['base']['acc']*100:.0f}%  textLoRA "
                  f"{idd['textlora']['acc']*100:.0f}%  WOVEN {idd['woven']['acc']*100:.0f}%  oracle "
                  f"{idd['oracle']['acc']*100:.0f}%  | pass@{max(ks)} {csp['pass@k']['in-dist'][max(ks)]*100:.0f}%",
                  flush=True)
            ctl = csp["controls"].get("in-dist")
            if ctl:
                # CANONICAL causal-drop (matches run_glados_staged._live_controls): content-sensitivity
                # only, so the min excludes 'zero' (no-injection = the separate LIFT/necessary axis).
                drop = ctl["true"] - min(ctl["shuffle"], ctl["permute"], ctl["corrupt"])
                print(f"  causal drop (true -> worst CONTENT control): {drop*100:.0f} pts "
                      f"(>0 => the LM CAUSALLY WIELDS the organ; 'zero' is the separate lift axis)",
                      flush=True)


# ============================================================================== SMOKE
def build_smoke_woven(base_id, dev, tok, *, seed=0):
    """Train a TINY live-latent woven model (the staged recipe, smoke args) so the smoke exercises a
    REAL woven checkpoint end-to-end. Returns (model, cfg)."""
    from transformers import AutoModelForCausalLM
    from . import run_glados_staged as S
    probe = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16)
    D = probe.config.hidden_size; nL = probe.config.num_hidden_layers
    del probe
    a = SimpleNamespace(
        lora_r=8, lora_lr=2e-4, gamma_lr=1e-3, alpha_lr=3e-4, gamma_hidden=128,
        inject_layer=min(12, nL - 1), mid_layer=min(8, nL - 2), warm_steps=40, steps=40, bs=6,
        aux_w=1.0, alpha_sup_w=1.0, alpha_dp=256, alpha_heads=4, dctx=192, organ_d=96, organ_heads=4,
        organ_layers=2, organ_T=8, seed=seed)
    rng = np.random.default_rng(seed + 11)
    train_recs = S.build_live_pool(rng, S.RUNGS, "id", 20)
    eval_recs = S.build_live_pool(rng, S.RUNGS, "id", 8)
    model = S.train_live_woven((base_id, D, nL), tok, dev, a, train_recs, eval_recs)
    cfg = {"kind": "live", "D": D, "K": S.K, "lora_r": a.lora_r, "gamma_hidden": a.gamma_hidden,
           "inject_layer": a.inject_layer, "mid_layer": a.mid_layer, "dctx": a.dctx,
           "alpha_dp": a.alpha_dp, "alpha_heads": a.alpha_heads, "organ_d": a.organ_d,
           "organ_heads": a.organ_heads, "organ_layers": a.organ_layers, "organ_T": a.organ_T}
    return model, cfg


def main():
    ap = argparse.ArgumentParser(description="GLaDOS standardized eval-suite arbiter")
    ap.add_argument("--base", default="allenai/OLMo-2-0425-1B", help="base model HF id")
    ap.add_argument("--woven_ckpt", default=None, help="saved woven checkpoint (.pt)")
    ap.add_argument("--tiers", nargs="+", default=["tier1", "tier2", "tier3"])
    ap.add_argument("--arms", nargs="+", default=["base", "textlora", "woven", "oracle"])
    ap.add_argument("--k", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--limit", type=int, default=20, help="per-task sample cap")
    ap.add_argument("--csp_per_rung", type=int, default=16)
    ap.add_argument("--no_controls", action="store_true")
    ap.add_argument("--no_rg", action="store_true")
    ap.add_argument("--std_tasks", nargs="+", default=None,
                    help="override lm-eval task list (e.g. a light set); default uses the full tier lists")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_time_search", choices=["none", "best_of_n", "beam", "both"], default="none",
                    help="Tier-1 CSP test-time search, sound-organ-verifier selected (off by default)")
    ap.add_argument("--tts_n", nargs="+", type=int, default=[1, 2, 4, 8, 16],
                    help="best-of-N budgets for the accuracy-vs-compute curve")
    ap.add_argument("--tts_beam", type=int, default=4, help="beam width (verifier-pruned beam)")
    ap.add_argument("--tts_temp", type=float, default=1.0, help="sampling temperature over the candidate dist")
    ap.add_argument("--tts_trials", type=int, default=8, help="Monte-Carlo draws averaged per N")
    ap.add_argument("--smoke", action="store_true",
                    help="train+save+reload a TINY woven model and run every tier at a tiny sample")
    ap.add_argument("--tts_smoke", action="store_true",
                    help="CPU/structural test-time-search smoke (no model): the exact verifier prunes "
                         "stub candidates correctly + best-of-N rises with N")
    ap.add_argument("--smoke_ckpt", default="runs/woven_smoke.pt")
    a = ap.parse_args()
    dev = device()

    if a.tts_smoke:
        tts_structural_smoke(seed=a.seed)
        return

    if a.smoke:
        from transformers import AutoTokenizer
        print(f"==== SMOKE: build a tiny woven model on {a.base}, save+reload, eval every tier ====",
              flush=True)
        tok = AutoTokenizer.from_pretrained(a.base)
        tok.padding_side = "right"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model, cfg = build_smoke_woven(a.base, dev, tok, seed=a.seed)
        save_woven(model, a.smoke_ckpt, base_id=a.base, cfg=cfg)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        std = a.std_tasks or ["gsm8k", "arc_challenge", "hellaswag", "wikitext"]  # light set for smoke
        run_eval_suite(a.base, woven_ckpt=a.smoke_ckpt, tiers=tuple(a.tiers), arms=tuple(a.arms),
                       k=tuple(a.k), limit=a.limit, csp_per_rung=a.csp_per_rung,
                       controls=not a.no_controls, rg_ceiling=not a.no_rg, std_tasks=std,
                       out=a.out or "runs/eval_suite_smoke.json", dev=dev, seed=a.seed,
                       test_time_search=a.test_time_search, tts_n=tuple(a.tts_n), tts_beam=a.tts_beam,
                       tts_temp=a.tts_temp, tts_trials=a.tts_trials)
        return

    run_eval_suite(a.base, woven_ckpt=a.woven_ckpt, tiers=tuple(a.tiers), arms=tuple(a.arms),
                   k=tuple(a.k), limit=a.limit, csp_per_rung=a.csp_per_rung,
                   controls=not a.no_controls, rg_ceiling=not a.no_rg, std_tasks=a.std_tasks,
                   out=a.out, dev=dev, seed=a.seed,
                   test_time_search=a.test_time_search, tts_n=tuple(a.tts_n), tts_beam=a.tts_beam,
                   tts_temp=a.tts_temp, tts_trials=a.tts_trials)


if __name__ == "__main__":
    main()
