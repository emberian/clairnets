"""clair/alien_consumer.py — the CALIBRATION / ALIEN-CONSUMER study.

THE QUESTION (Ember's thesis): does the LM extract value from PARTIAL or WRONG organ output, not
just CLEAN? Is the LM sophisticated enough to MINE a partial/imperfect organ signal for its own
purposes — so the α-on-hard-instances WALL becomes a GRACEFUL-DEGRADATION curve rather than a hard
CLIFF? If true, GLaDOS is robust to α-miscompiles (it uses what's good, drops what's not) instead of
the brittle confident-wrong we saw.

DESIGN. Build on the validated two-stream + γ-readout machinery (oracle_readout.OracleReadout: frozen
OLMo + LoRA + a zero-init structured γ that scatters a per-cell lattice into a late residual; LM head
GENERATES the answer). Two changes vs the engagement run:

  (1) RICH-STATE READOUT. Instead of γ reading a CLEANED answer-or-abstain (each cell collapsed to a
      singleton or 'unknown'), expose the FULL partial organ state: the per-cell narrowed candidate
      SET (cardinality, not just singletons) + an honest, organ-derivable RELIABILITY scalar (how far
      did it narrow / did it reach a singleton fixpoint). Feed the mess; let the LM mine it. The γ
      input per cell becomes [set-membership(K), |set|/d, reliability] (Fin = K+2). The POINT-readout
      ablation collapses each cell to a singleton-or-unknown (Fin = K, no set, no reliability) — the
      "cleaned answer-or-abstain" baseline.

  (2) DIVERSE-ORGAN-QUALITY CURRICULUM. Train + eval across a SPECTRUM of organ quality on the SAME
      (easy, in-context-solvable) problems — so the ONLY variable is the ORGAN's output quality, NOT
      task difficulty (this is what isolates the calibration question from the α-compile WALL):
        CLEAN     — the exact dedₚ narrowing (correct, full).
        PARTIAL   — incomplete-but-SOUND: a strict SUPERSET of dedₚ (the organ stopped early; never
                    drops a true survivor, just under-narrowed). The truth is STILL in the set.
        WRONG     — a plausibly-WRONG miscompile: the query cell narrowed to a confident WRONG
                    singleton (the truth is NOT in the set), plus a few other cells flipped. The
                    dangerous "confident-wrong" case.
        CORRUPTED — the control: random meaningless survival bits.
      TWO-STREAM (text-ablated): the LM GENERATES from a CELLS prompt (roster + question, NO facts),
      so the injected lattice is the ONLY route to the answer (no text shortcut, no cross-check).

THE CORE MEASUREMENT. WOVEN-minus-TEXT-LoRA lift as a function of organ quality.
  * GRACEFUL (thesis holds): lift stays positive and degrades smoothly — clean ≫ partial > wrong ≈ 0,
    and PARTIAL is still clearly positive => the LM MINES partial output.
  * CLIFF (thesis fails): lift only on clean; partial ≈ 0 or negative => the LM needs clean output.
  Two sub-questions: (a) does WOVEN beat TEXT-LoRA when the organ is PARTIAL (the key case)? (b) does
  it correctly NOT get dragged into confident-wrong on WRONG (lift ≈ 0, down-weights) or follow it off
  a cliff (lift ≪ 0)? And: does RICH-state help over POINT-readout (esp. on PARTIAL, where point
  collapses the multi-element set to 'unknown')?

  SMOKE: python -m clair.alien_consumer --smoke
  FULL : python -m clair.alien_consumer --steps 2000 --readout both --out runs/alien_consumer.json
"""
from __future__ import annotations

import argparse, contextlib, gc, json, os, time

import numpy as np
import torch
import torch.nn.functional as F

from . import oracle_readout as O
from . import run_glados_staged as G

K = G.K
QUALITIES = ["clean", "partial", "wrong", "corrupted"]


# ===================================================================== quality-spectrum synthesis
def _cell_sets(surv, n, d):
    """Per-cell surviving value SETS from a binary surv[n,K] over the valid domain range(d)."""
    return [set(v for v in range(d) if surv[i, v] > 0.5) for i in range(n)]


def synth_lattice(clean_surv, n, d, query, gold_idx, quality, rng):
    """Take the CLEAN oracle-dedₚ lattice and produce a quality-`quality` variant + an honest
    organ-derivable reliability scalar. Returns (var_surv[n,K] float32, reliability in [0,1])."""
    sets = _cell_sets(clean_surv, n, d)
    var = np.zeros((n, K), dtype=np.float32)
    if quality == "clean":
        out_sets = sets
    elif quality == "partial":
        # incomplete-but-SOUND: keep every true survivor, re-admit some eliminated valid values
        # (the organ stopped early). Strict superset => the truth is still in every cell's set. Cap
        # additions to keep |set| <= d-1 (at least one value still eliminated) so partial stays
        # genuinely NARROWED — never saturating to the full domain (= no organ).
        out_sets = []
        for i, s in enumerate(sets):
            elim = [v for v in range(d) if v not in s]
            rng.shuffle(elim)
            max_add = max(0, (d - 1) - len(s))               # keep >=1 value eliminated
            n_add = min(len(elim), max_add, 1 + int(rng.random() < 0.5))
            out_sets.append(set(s) | set(elim[:n_add]))
    elif quality == "wrong":
        # plausibly-WRONG miscompile: the QUERY cell narrowed to a confident WRONG singleton (truth
        # dropped), plus ~30% of other cells flipped to a random singleton (a plausible-but-wrong
        # lattice). The organ "thinks" it succeeded => reliability is HIGH (it can't know it's wrong).
        out_sets = [set(s) for s in sets]
        wrong_vals = [v for v in range(d) if v != gold_idx]
        if wrong_vals:
            out_sets[query] = {int(rng.choice(wrong_vals))}
        for i in range(n):
            if i != query and rng.random() < 0.3:
                out_sets[i] = {int(rng.integers(0, d))}
    elif quality == "corrupted":
        # the control: random meaningless survival (each valid value alive w.p. 0.5, >=1 alive).
        out_sets = []
        for i in range(n):
            s = set(v for v in range(d) if rng.random() < 0.5)
            if not s:
                s = {int(rng.integers(0, d))}
            out_sets.append(s)
    else:
        raise ValueError(quality)
    for i, s in enumerate(out_sets):
        for v in s:
            if v < K:
                var[i, v] = 1.0
    # honest reliability = mean narrowing fraction over cells (1.0 = all singletons, 0.0 = no
    # narrowing). This is what the organ CAN self-report (progress/fixpoint); it is HIGH for a
    # confident-WRONG lattice too (the organ cannot know it miscompiled) — the realistic adversarial
    # case for the calibration question.
    denom = max(1, d - 1)
    rel = float(np.mean([1.0 - (max(1, len(s)) - 1) / denom for s in out_sets]))
    return var, rel


def encode_feat(var_surv, rel, n, d, mode):
    """Build the per-cell γ input feature.
       rich  : [set-membership(K), |set|/d, reliability]  (Fin = K+2) — the full partial state.
       point : each cell COLLAPSED to a singleton one-hot, else all-zeros 'unknown' (Fin = K) —
               the 'cleaned answer-or-abstain' baseline that throws away the set.
    """
    if mode == "rich":
        feat = np.zeros((n, K + 2), dtype=np.float32)
        feat[:, :K] = var_surv
        card = var_surv[:, :d].sum(-1)
        feat[:, K] = card / max(1, d)
        feat[:, K + 1] = rel
        return feat
    if mode == "point":
        feat = np.zeros((n, K), dtype=np.float32)
        for i in range(n):
            s = [v for v in range(d) if var_surv[i, v] > 0.5]
            if len(s) == 1:                       # only a confident singleton survives the collapse
                feat[i, s[0]] = 1.0
        return feat
    raise ValueError(mode)


def feat_dim(mode):
    return K + 2 if mode == "rich" else K


def precompute_variants(recs, seed):
    """Freeze one quality-variant lattice per (rec, quality) so RICH and POINT models — and the WOVEN
    vs TEXT-LoRA comparisons — all measure on IDENTICAL injected lattices (measurement integrity)."""
    for ri, r in enumerate(recs):
        rng = np.random.default_rng(seed * 1_000_003 + ri)
        d = len(r["vnames"])
        r["variants"] = {q: synth_lattice(r["surv"], r["n"], d, r["query"], r["gold_idx"], q, rng)
                         for q in QUALITIES}
        # sanity: PARTIAL keeps the truth in the query set; WRONG drops it.
        r["_partial_has_gold"] = bool(r["variants"]["partial"][0][r["query"], r["gold_idx"]] > 0.5)
        r["_wrong_has_gold"] = bool(r["variants"]["wrong"][0][r["query"], r["gold_idx"]] > 0.5)


# ===================================================================== batching
def _feat_batch(chunk, qualities, mode, dev, rng=None):
    """Per-chunk γ-feature tensor [B,Nmax,Fin]. If a rec has a frozen variant (eval), use it; else
    synthesize fresh from rng (train augmentation)."""
    Fin = feat_dim(mode)
    Nmax = max(r["n"] for r in chunk)
    feat = torch.zeros(len(chunk), Nmax, Fin, device=dev)
    for b, (r, q) in enumerate(zip(chunk, qualities)):
        d = len(r["vnames"])
        if "variants" in r:
            var, rel = r["variants"][q]
        else:
            var, rel = synth_lattice(r["surv"], r["n"], d, r["query"], r["gold_idx"], q, rng)
        feat[b, : r["n"], :] = torch.from_numpy(encode_feat(var, rel, r["n"], d, mode)).to(dev)
    return feat


def build_train_batch(recs, qualities, tok, mode, dev):
    """Tokenize the (fact-ablated cells) prompt + answer, answer-span LM labels, mentions, and the
    quality-`qualities` γ feature."""
    Nmax = max(r["n"] for r in recs)
    fulls = [r["prompt"] + " " + r["answer"] for r in recs]
    plens = [len(r["prompt"]) for r in recs]
    enc = tok(fulls, return_offsets_mapping=True, padding=True, return_tensors="pt")
    ids = enc["input_ids"].to(dev); attn = enc["attention_mask"].to(dev)
    offsets = enc["offset_mapping"]; T = ids.size(1)
    labels = ids.clone()
    for b in range(len(recs)):
        for ti, (lo, hi) in enumerate(offsets[b].tolist()):
            if not ((lo != hi) and (lo >= plens[b]) and (attn[b, ti] > 0.5)):
                labels[b, ti] = -100
    mention = O._mention_tensor([r["mentions"] for r in recs], offsets, len(recs), Nmax, T, dev)
    return {"input_ids": ids, "attn": attn, "labels": labels, "mention": mention}


# ===================================================================== scoring
@torch.no_grad()
def score_alien(model, recs, tok, dev, quality, mode, inject=True, use_base=False, fewshot="", bs=8):
    """GENERATIVE accuracy: OLMo's LM head scores the answer span over the record's OWN legal answers
    {value-names..., 'cannot be determined'}. inject=True scatters the quality-`quality` lattice via
    γ (WOVEN); inject=False is the TEXT-LoRA path (no lattice — the cells prompt carries NO facts, so
    this is the no-organ floor). use_base disables LoRA (pure base OLMo + fewshot)."""
    model.eval()
    correct = tot = det_t = det_r = 0
    for i in range(0, len(recs), bs):
        chunk = recs[i:i + bs]
        Bp = len(chunk); Nmax = max(r["n"] for r in chunk)
        feat = _feat_batch(chunk, [quality] * Bp, mode, dev) if inject else None
        fulls, plens, spans, prob_of = [], [], [], []
        shift = len(fewshot)
        for pi, r in enumerate(chunk):
            cands = [" " + v for v in r["vnames"]] + [" " + O.ABSTAIN_STR]
            for cand in cands:
                fulls.append(fewshot + r["prompt"] + cand); plens.append(shift + len(r["prompt"]))
                spans.append({k: [(a + shift, c + shift) for (a, c) in v] for k, v in r["mentions"].items()})
                prob_of.append(pi)
        enc = tok(fulls, return_offsets_mapping=True, padding=True, return_tensors="pt")
        ids = enc["input_ids"].to(dev); attn = enc["attention_mask"].to(dev)
        offsets = enc["offset_mapping"]; T = ids.size(1)
        if inject:
            mention = O._mention_tensor(spans, offsets, len(fulls), Nmax, T, dev)
            feat_exp = feat[torch.tensor(prob_of, device=dev)]
            ctx = model.injection(feat_exp, mention, enabled=True)
        else:
            ctx = model.injection(None, None, enabled=False)
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
            if r["determined"]:
                det_t += 1; det_r += ok
    return {"overall": correct / max(1, tot), "det_acc": det_r / max(1, det_t), "n": tot}


# ===================================================================== train one readout
def build_model(olmo_ids, tok, dev, a, mode):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    mid_id, D, nL = olmo_ids
    olmo = AutoModelForCausalLM.from_pretrained(mid_id, dtype=torch.bfloat16).to(dev).eval()
    for p in olmo.parameters():
        p.requires_grad_(False)
    lconf = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, bias="none",
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
    peft_model = get_peft_model(olmo, lconf)
    inj = min(a.inject_layer, nL - 1)
    Fin = feat_dim(mode)
    model = O.OracleReadout(peft_model, D, Fin, inj, gamma_hidden=a.gamma_hidden).to(dev)
    model.gamma.float()
    return model, inj, Fin


def train_readout(model, mode, Fin, tok, dev, a, train_recs, eval_recs):
    """Train LoRA + γ on the answer-span LM CE over the DIVERSE-quality curriculum (each example gets a
    uniformly-random organ quality), two-stream (cells gen prompt). Returns the trained model."""
    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.model.named_parameters() if p.requires_grad],
         "lr": a.lora_lr, "weight_decay": 0.01},
        {"params": list(model.gamma.parameters()), "lr": a.gamma_lr, "weight_decay": 0.0},
    ], betas=(0.9, 0.95))
    rng = np.random.default_rng(a.seed + 7)
    eval_clean = eval_recs  # variants precomputed; pick clean for the in-loop progress probe
    print(f"\n  ALIEN READOUT [{mode}]  Fin={Fin}  inject {model.inject_layer}  "
          f"trainable {O.n_trainable(model):,}", flush=True)
    t0 = time.time(); model.train()
    for s in range(1, a.steps + 1):
        idxs = rng.integers(0, len(train_recs), a.bs).tolist()
        chunk = [train_recs[i] for i in idxs]
        quals = (["clean"] * len(chunk) if getattr(a, "clean_only", False)
                 else [QUALITIES[int(rng.integers(0, len(QUALITIES)))] for _ in chunk])  # diverse-quality mix (clean_only: train CLEAN-only to establish the clean-lift ceiling)
        ba = build_train_batch(chunk, quals, tok, mode, dev)
        feat = _feat_batch(chunk, quals, mode, dev, rng=rng)
        with model.injection(feat, ba["mention"], enabled=True):
            logits = model.logits(ba["input_ids"], ba["attn"]).float()
        lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                             ba["labels"][:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad(); lm.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0); opt.step()
        if s % max(1, a.steps // 12) == 0 or s == 1:
            acc = score_alien(model, eval_clean, tok, dev, "clean", mode, inject=True, bs=a.bs)
            print(f"  step {s:5d}  lm {lm.item():.3f}  alpha(gate tanh) {float(torch.tanh(model.gamma.alpha)):+.3f}"
                  f"  CLEAN-woven acc {acc['overall']*100:4.1f}% (det {acc['det_acc']*100:.0f})"
                  f"  {time.time()-t0:.0f}s", flush=True)
            model.train()
    return model


# ===================================================================== orchestration
def evaluate(model, mode, tok, dev, a, eval_recs):
    """Per-quality WOVEN accuracy + the (quality-independent) TEXT-LoRA / BASE floors => lift curve."""
    textlora = score_alien(model, eval_recs, tok, dev, "clean", mode, inject=False, bs=a.bs)
    base = score_alien(model, eval_recs, tok, dev, "clean", mode, inject=False, use_base=True,
                       fewshot=G.FEWSHOT, bs=a.bs)
    rows = {}
    for q in QUALITIES:
        w = score_alien(model, eval_recs, tok, dev, q, mode, inject=True, bs=a.bs)
        rows[q] = {"woven": w["overall"], "woven_det": w["det_acc"],
                   "lift": w["overall"] - textlora["overall"]}
    return {"textlora": textlora["overall"], "base": base["overall"],
            "per_quality": rows, "n": eval_recs and len(eval_recs)}


def _report(mode, res, gold_in):
    print(f"\n  ===== ALIEN-CONSUMER LIFT-vs-ORGAN-QUALITY  [{mode} readout]  (n={res['n']}) =====",
          flush=True)
    print(f"    TEXT-LoRA floor {res['textlora']*100:5.1f}%   BASE(fewshot) {res['base']*100:5.1f}%",
          flush=True)
    print(f"    {'quality':10s}  {'WOVEN':>7s}  {'det':>5s}  {'lift(W-TL)':>11s}", flush=True)
    for q in QUALITIES:
        r = res["per_quality"][q]
        print(f"    {q:10s}  {r['woven']*100:6.1f}%  {r['woven_det']*100:4.0f}%  {r['lift']*100:+10.1f}",
              flush=True)
    print(f"    [sanity] partial keeps gold in query set: {gold_in['partial']*100:.0f}% | "
          f"wrong keeps gold: {gold_in['wrong']*100:.0f}%", flush=True)


def _verdict(results):
    print("\n================ VERDICT: graceful degradation or cliff? ================", flush=True)
    for mode, res in results.items():
        pq = res["per_quality"]
        lc, lp, lw, lk = (pq[q]["lift"] * 100 for q in QUALITIES)
        partial_pos = lp >= 5.0
        wrong_ok = lw >= -5.0                     # NOT dragged off a cliff by confident-wrong
        wrong_dragged = lw <= -10.0
        graceful = (lc > lp > 0) and partial_pos
        print(f"  [{mode}]  lift  clean {lc:+.1f}  partial {lp:+.1f}  wrong {lw:+.1f}  corrupted {lk:+.1f}",
              flush=True)
        if graceful:
            print(f"    => GRACEFUL DEGRADATION: PARTIAL lift {lp:+.1f} is clearly positive and clean>partial."
                  " The LM MINES incomplete-but-sound organ output (thesis HOLDS).", flush=True)
        elif lc >= 5.0 and not partial_pos:
            print(f"    => CLIFF: lift concentrates on CLEAN ({lc:+.1f}); PARTIAL ({lp:+.1f}) is not"
                  " clearly positive. The LM needs CLEAN output (thesis FAILS).", flush=True)
        else:
            print(f"    => INCONCLUSIVE/OTHER: clean {lc:+.1f} partial {lp:+.1f} (organ may be bypassed"
                  " — check the CLEAN lift cleared threshold at all).", flush=True)
        if wrong_dragged:
            print(f"    WRONG: lift {lw:+.1f} => the LM IS DRAGGED into confident-wrong (follows the"
                  " miscompile off a cliff; brittle in text-ablated mode).", flush=True)
        elif wrong_ok:
            print(f"    WRONG: lift {lw:+.1f} ≈ 0 => the LM does NOT get dragged (down-weights the"
                  " miscompile; robust).", flush=True)
    if len(results) == 2:
        rc, pc = results["rich"]["per_quality"]["partial"]["lift"], results["point"]["per_quality"]["partial"]["lift"]
        d = (rc - pc) * 100
        print(f"\n  RICH vs POINT on PARTIAL (the key case): rich lift {rc*100:+.1f} vs point {pc*100:+.1f}"
              f"  => rich-state {'HELPS' if d > 3 else ('~ties' if d > -3 else 'HURTS')} by {d:+.1f}pts."
              " (point collapses the multi-element partial set to 'unknown'; rich exposes it.)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lora_lr", type=float, default=2e-4)
    ap.add_argument("--gamma_lr", type=float, default=1e-3)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--inject_layer", type=int, default=12)
    ap.add_argument("--gamma_hidden", type=int, default=256)
    ap.add_argument("--per_rung_train", type=int, default=400)
    ap.add_argument("--per_rung_eval", type=int, default=80)
    ap.add_argument("--regime", default="small", help="{small,large,hard} (see run_glados LIVE_REGIMES)")
    ap.add_argument("--readout", default="both", choices=["rich", "point", "both"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean_only", action="store_true",
                    help="CONTROL: train on CLEAN organ output only (no partial/wrong/corrupted) to "
                         "establish the clean-lift CEILING before trusting the quality-mix curve")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.steps = 120; a.per_rung_train = 40; a.per_rung_eval = 20
    dev = G.device()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    from transformers import AutoTokenizer, AutoConfig
    mid = "allenai/OLMo-2-0425-1B"
    print(f"device={dev}  budget N={G.N_MAX} D={G.D_MAX}  K={K}  regime={a.regime}  readout={a.readout}",
          flush=True)
    print("loading tokenizer/config", mid, flush=True)
    tok = AutoTokenizer.from_pretrained(mid)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(mid)
    D, nL = cfg.hidden_size, cfg.num_hidden_layers
    print(f"OLMo: {nL} layers, hidden {D}", flush=True)

    name = G._REGIME_ALIAS.get(a.regime.strip(), a.regime.strip())
    rungs, split = G.LIVE_REGIMES[name]
    print(f"REGIME {name}: rungs={rungs} split={split}", flush=True)
    rng_tr = np.random.default_rng(a.seed + 100)
    rng_ev = np.random.default_rng(a.seed + 200)
    # two_stream=True: gen prompt is the fact-ablated CELLS roster+question; the injected lattice is
    # the ONLY route to the answer. det_only=True: a unique forced answer (sharp right/wrong signal).
    train_recs = G.build_live_pool(rng_tr, rungs, split, a.per_rung_train, det_only=True, two_stream=True)
    eval_recs = G.build_live_pool(rng_ev, rungs, split, a.per_rung_eval, det_only=True, two_stream=True)
    precompute_variants(eval_recs, a.seed + 1)
    gold_in = {"partial": np.mean([r["_partial_has_gold"] for r in eval_recs]),
               "wrong": np.mean([r["_wrong_has_gold"] for r in eval_recs])}
    print(f"data: {len(train_recs)} train / {len(eval_recs)} eval recs (det-only, two-stream cells-gen)",
          flush=True)

    modes = ["rich", "point"] if a.readout == "both" else [a.readout]
    results = {}
    for mode in modes:
        print(f"\n################ READOUT = {mode.upper()} ################", flush=True)
        model, inj, Fin = build_model((mid, D, nL), tok, dev, a, mode)
        model = train_readout(model, mode, Fin, tok, dev, a, train_recs, eval_recs)
        res = evaluate(model, mode, tok, dev, a, eval_recs)
        results[mode] = res
        _report(mode, res, gold_in)
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _verdict(results)
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "runs", "alien_consumer.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"args": vars(a), "gold_in": gold_in, "results": results}, open(out, "w"),
              indent=1, default=float)
    print("\nwrote", out, flush=True)
    return results


if __name__ == "__main__":
    main()
