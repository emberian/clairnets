"""clair/run_splitbrain.py — the SPLIT-BRAIN NECESSITY TEST (does the LM WIELD the organ, or
just consult it OPPORTUNISTICALLY when it already agrees with the text?).

THE GAP THIS CLOSES.  The corrupt-organ control (run_glados_staged DELIVERABLE 2) proves the LM's
generation DEGRADES when the frozen organ's lattice is damaged even with full text present — i.e. the
LM does NOT fall back to reading the text. But corruption CANNOT distinguish two mechanisms:
  (A) the LM genuinely WIELDS the organ — it TRUSTS the organ's answer over the text; or
  (B) the LM uses the organ OPPORTUNISTICALLY — it reads the organ ONLY because (in the in-dist data)
      the organ AGREES with the text; were they to DISAGREE it would override the organ with the text.
Both (A) and (B) predict a corruption drop (corrupting the organ removes the agreeing signal). They
are separated ONLY by CONFLICT: keep the organ CORRECT, make the TEXT adversarially WRONG, and ask
which answer the LM emits. This is the SATNet "opportunistic agreement" failure mode that corruption
is blind to.

THE EXPERIMENT.  Reuse the WORKING staged GLaDOS (run_glados_staged): bootstrap+freeze a general
organ, then train the LoRA+gamma readout with the readout-FORCING (cells-first) curriculum so the LM
causally uses the organ. THEN evaluate four CONFLICT conditions (codex's list). In every condition the
injected lattice is the organ's TRUE, CORRECT fixpoint (verified per-instance: the organ's query-cell
domain is the correct singleton for the determined conditions, genuinely multivalued for the abstain
condition); only the TEXT is corrupted:

  C1  PLAUSIBLE-WRONG    organ pins query=A; text SUGGESTS (hedged) the query is B != A.
  C2  DIRECT-CONTRADICT  organ pins query=A; text ASSERTS (flat pin) the query is B != A.
  C3  ORGAN-ABSTAINS     query genuinely UNDETERMINED (organ keeps >1 value => 'cannot be determined');
                         text CONFIDENTLY asserts a specific (wrong, over-committed) value B.
  C4  WRONG-RATIONALE    organ pins query=A; text gives a FLUENT but incorrect derivation concluding B.

MEASURE per condition (in-dist + OOD): of the LM's emitted answer,
  follow-organ  = it emits the ORGAN's answer  (A, or 'cannot be determined' for C3);
  follow-text   = it emits the TEXT's answer    (B);
  other         = neither.
CALIBRATION rows: the SAME conflict prompts scored by the TEXT-ONLY path (no injection) — which can
ONLY follow the text — and by the woven model with the organ CORRUPTED — to confirm the organ is what
drives any follow-organ.

VERDICT.  high follow-organ under conflict => the LM genuinely WIELDS the organ (trusts it over
misleading text) = the STRONG necessity claim.  high follow-text => the organ is consulted only
OPPORTUNISTICALLY (used when it agrees, overridden when it disagrees) = the WEAKER, brittler claim.
We report HONESTLY whichever it is.

  SMOKE: python -m clair.run_splitbrain --smoke
  FULL : python -m clair.run_splitbrain --organ_steps 1500 --steps 2500 --out runs/splitbrain.json
"""
from __future__ import annotations

import argparse, contextlib, json, os, time

import numpy as np
import torch
import torch.nn.functional as F

from . import curriculum as CU
from . import oracle_readout as O
from . import run_glados_staged as G

K = G.K
CONDITIONS = ["C1_plausible", "C2_contradict", "C3_organ_abstains", "C4_wrong_rationale"]
DET_CONDS = ["C1_plausible", "C2_contradict", "C4_wrong_rationale"]   # need a determined query
AB_CONDS = ["C3_organ_abstains"]                                       # need an undetermined query


# ============================================================ adversarial TEXT construction
def _val_phrase(p, cell, vidx) -> str:
    """'<Entity> <copula> <value-name>' in the rung's kind (mirrors curriculum._fact_sentence pins)."""
    E, X = p.entity(cell), p.vnames[vidx]
    if p.kind == "color":
        return f"{E} is {X}"
    if p.kind == "ordinal":
        return f"{E} is in position {X}"
    return f"{E} equals {X}"


def _adv_sentence(p, cell, B, cond) -> str:
    """The misleading sentence injected into the TEXT body. All four point a TEXT-trusting reader at B;
    they differ in HOW (hedged suggestion / flat assertion / confident over-commit / fake derivation)."""
    base = _val_phrase(p, cell, B)
    E = p.entity(cell)
    if cond == "C1_plausible":
        return f"It looks like {base}."                                  # hedged suggestion
    if cond in ("C2_contradict", "C3_organ_abstains"):
        return f"{base.capitalize()}."                                   # flat assertion / over-commit
    if cond == "C4_wrong_rationale":
        return (f"Following the constraints step by step, {E} is forced to a single value, "
                f"so {base}.")                                           # fluent but WRONG derivation
    raise ValueError(cond)


def _pick_B(p, od, rng) -> int:
    """The TEXT's (wrong) value B.
       determined : a value != A; prefer one that already appears as a pin value (plausible/in-play).
       abstain    : a SURVIVING value (consistent with SOME solution) so the text's over-commitment is
                    the only error — the organ must still say 'cannot be determined'."""
    d = p.d
    if p.determined:
        A = p.answer
        pinvals = [f[2] for f in p.facts if f[0] == "pin" and f[2] != A]
        others = [v for v in range(d) if v != A]
        pool = pinvals or others
        return int(rng.choice(pool)) if pool else (A + 1) % d
    surv = sorted(od[p.query])
    return int(rng.choice(surv)) if surv else int(rng.integers(0, d))


def make_conflict_record(p, od, cond, rng) -> dict:
    """One split-brain record: organ lattice = the TRUE fixpoint `od` (correct); TEXT corrupted to point
    at B. organ_idx = the answer the ORGAN supports (A, or abstain for C3); text_idx = B."""
    B = _pick_B(p, od, rng)
    body_facts = " ".join(CU._fact_sentence(p, f) for f in p.facts)
    adv = _adv_sentence(p, p.query, B, cond)
    body = f"{body_facts} {adv} {CU._question(p)}"
    prompt = body + " Answer:"
    mentions = CU.entity_mentions(body, p.n)
    organ_idx = p.answer if p.determined else len(p.vnames)             # abstain == index len(vnames)
    return {"prompt": prompt, "n": p.n, "query": p.query, "determined": p.determined,
            "relation": p.relation, "vnames": list(p.vnames),
            "organ_idx": int(organ_idx), "text_idx": int(B), "cond": cond,
            "mentions": {int(k): [tuple(s) for s in v] for k, v in mentions.items()},
            "surv": G.surv_from_dom(od, p.n, K)}


# ============================================================ organ-correctness filtering + pools
def _organ_correct(p, od) -> str | None:
    """Classify whether the FROZEN organ is genuinely CORRECT on p's query, and into which family.
       'det' : determined query, organ's query-domain is the correct singleton {answer}, AND the query
               is NOT directly pinned in the text (so the answer needs PROPAGATION, never a text
               lookup — the canonical text does not already state A; the only explicit value claim is
               the adversarial B => the organ-vs-text conflict is clean).
       'ab'  : undetermined query AND organ keeps >1 value (genuinely abstains).
       None  : organ wrong/ambiguous here (excluded — we only test where the organ IS right)."""
    qd = od[p.query]
    pinned = {f[1] for f in p.facts if f[0] == "pin"}
    if p.determined and p.query not in pinned and len(qd) == 1 and next(iter(qd)) == p.answer:
        return "det"
    if (not p.determined) and len(qd) > 1:
        return "ab"
    return None


def build_conflict_pools(rng, organ, dev, rungs, split, per_rung_det, per_rung_ab,
                         max_rounds=40, batch=300, patience=4):
    """For each rung sample problems, run the FROZEN organ to a fixpoint, KEEP only instances where the
    organ is genuinely correct, and emit conflict records: {cond: [records]} (in-dist or OOD split).
    Early-stops a rung/split once `patience` consecutive rounds add NOTHING to either quota (so a rung
    the organ can't solve at this split bails fast instead of burning all max_rounds)."""
    recs = {c: [] for c in CONDITIONS}
    for rung in rungs:
        det_ok, ab_ok = [], []
        stall = 0
        for rd in range(max_rounds):
            if len(det_ok) >= per_rung_det and len(ab_ok) >= per_rung_ab:
                break
            probs = [G.make_problem(rng, rung, split) for _ in range(batch)]
            ocsps = [G.organ_csp(p) for p in probs]
            odoms = G.organ_fixpoint(organ, ocsps, dev)
            added = 0
            for p, od in zip(probs, odoms):
                fam = _organ_correct(p, od)
                if fam == "det" and len(det_ok) < per_rung_det:
                    det_ok.append((p, od)); added += 1
                elif fam == "ab" and len(ab_ok) < per_rung_ab:
                    ab_ok.append((p, od)); added += 1
            stall = stall + 1 if added == 0 else 0
            if stall >= patience:
                break
        print(f"    {rung:12s} organ-correct: det {len(det_ok)}/{per_rung_det}  "
              f"abstain {len(ab_ok)}/{per_rung_ab}", flush=True)
        for (p, od) in det_ok:
            for cond in DET_CONDS:
                recs[cond].append(make_conflict_record(p, od, cond, rng))
        for (p, od) in ab_ok:
            for cond in AB_CONDS:
                recs[cond].append(make_conflict_record(p, od, cond, rng))
    for c in CONDITIONS:
        rng.shuffle(recs[c])
    return recs


# ============================================================ the SPLIT-BRAIN scorer
@torch.no_grad()
def score_conflict(model, recs, tok, dev, inject=True, control="true", use_base=False, fewshot="",
                   bs=8, perm=None):
    """Constrained LM-head generation over {value-names..., 'cannot be determined'} on CONFLICT records.
    Returns follow-organ / follow-text / other rates (overall + per-rung). Mirrors
    run_glados_staged.score_gen's scoring, but tallies which BRAIN the emitted answer matches."""
    model.eval()
    if perm is None:
        perm = tuple(list(range(1, K)) + [0])
    fo = ft = oth = tot = 0
    by = {}                                                            # rung -> [fo, ft, oth, n]
    for i in range(0, len(recs), bs):
        chunk = recs[i:i + bs]
        Bp = len(chunk); Nmax = max(r["n"] for r in chunk)
        surv = torch.zeros(Bp, Nmax, K, device=dev)
        for b, r in enumerate(chunk):
            surv[b, : r["n"], :] = torch.from_numpy(r["surv"]).to(dev)
        if inject:
            surv = G.apply_control_ext(surv, chunk, control, K, perm)
        fulls, plens, spans, prob_of = [], [], [], []
        for pi, r in enumerate(chunk):
            prompt = fewshot + r["prompt"]; shift = len(fewshot)
            cands = [" " + v for v in r["vnames"]] + [" " + O.ABSTAIN_STR]
            for cand in cands:
                fulls.append(prompt + cand); plens.append(len(prompt))
                spans.append({k: [(a + shift, c + shift) for (a, c) in v] for k, v in r["mentions"].items()})
                prob_of.append(pi)
        enc = tok(fulls, return_offsets_mapping=True, padding=True, return_tensors="pt")
        ids = enc["input_ids"].to(dev); attn = enc["attention_mask"].to(dev)
        offsets = enc["offset_mapping"]; T = ids.size(1)
        if inject:
            mention = O._mention_tensor(spans, offsets, len(fulls), Nmax, T, dev)
            surv_exp = surv[torch.tensor(prob_of, device=dev)]
            ctx = model.injection(surv_exp, mention, enabled=True)
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
                a, c = offs[ti]
                if a != c and a >= plens[row] and attn[row, ti] > 0.5:
                    mask[ti - 1] = 1.0
            denom = mask.sum().clamp_min(1.0)
            scores[row] = (tok_lp[row] * mask).sum() / denom
        prob_of_t = torch.tensor(prob_of, device=dev)
        for pi, r in enumerate(chunk):
            rows = (prob_of_t == pi).nonzero().flatten()
            pred = int(rows[int(scores[rows].argmax())] - rows[0])
            is_o = int(pred == r["organ_idx"]); is_t = int(pred == r["text_idx"])
            fo += is_o; ft += is_t; oth += int(not is_o and not is_t); tot += 1
            d = by.setdefault(r["relation"], [0, 0, 0, 0])
            d[0] += is_o; d[1] += is_t; d[2] += int(not is_o and not is_t); d[3] += 1
    n = max(1, tot)
    return {"follow_organ": fo / n, "follow_text": ft / n, "other": oth / n, "n": tot,
            "by_rung": {k: {"organ": v[0] / max(1, v[3]), "text": v[1] / max(1, v[3]),
                            "other": v[2] / max(1, v[3]), "n": v[3]} for k, v in by.items()}}


def _fmt(tag, r):
    return (f"    {tag:34s}  organ {r['follow_organ']*100:5.1f}%   text {r['follow_text']*100:5.1f}%   "
            f"other {r['other']*100:5.1f}%   (n={r['n']})")


# ============================================================ orchestration
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organ_steps", type=int, default=1500)
    ap.add_argument("--organ_target", type=float, default=3.0e5)
    ap.add_argument("--organ_R", type=int, default=8)
    ap.add_argument("--organ_lr", type=float, default=3e-4)
    ap.add_argument("--organ_pool", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lora_lr", type=float, default=2e-4)
    ap.add_argument("--gamma_lr", type=float, default=1e-3)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--inject_layer", type=int, default=12)
    ap.add_argument("--gamma_hidden", type=int, default=256)
    ap.add_argument("--cells_frac", type=float, default=0.4)
    ap.add_argument("--per_rung_train", type=int, default=400)
    ap.add_argument("--per_rung_eval", type=int, default=80)
    ap.add_argument("--sb_per_rung_det", type=int, default=60)   # split-brain determined instances/rung
    ap.add_argument("--sb_per_rung_ab", type=int, default=40)    # split-brain abstain instances/rung
    ap.add_argument("--organ_recall_n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.organ_steps = 250; a.steps = 120; a.organ_pool = 64; a.per_rung_train = 60
        a.per_rung_eval = 24; a.sb_per_rung_det = 10; a.sb_per_rung_ab = 8; a.organ_recall_n = 60
    dev = G.device()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    print(f"device={dev}  SPLIT-BRAIN  budget N={G.N_MAX} D={G.D_MAX}  rungs={G.RUNGS}", flush=True)

    # ===== STAGE 1: bootstrap + freeze the general organ (reuse the staged recipe) =====
    print("\n================ STAGE 1: GENERAL ORGAN ================", flush=True)
    organ, od_dim, opar, olog = G.train_organ(dev, G.RUNGS, target=a.organ_target, steps=a.organ_steps,
                                              pool=a.organ_pool, R=a.organ_R, lr=a.organ_lr, seed=a.seed)
    for pp in organ.parameters():
        pp.requires_grad_(False)
    organ.eval()
    recall = G.organ_recall_report(organ, dev, G.RUNGS, n_per=a.organ_recall_n, seed=a.seed + 5)
    print("\n  FROZEN ORGAN query-exactness vs EXACT dedP (the lattice we will trust under conflict):",
          flush=True)
    for sp in ("id", "ood"):
        print(f"  -- {sp} --", flush=True)
        for rg in G.RUNGS:
            e = recall[(rg, sp)]
            print(f"    {rg:12s}  q-exact {e['query_exact']*100:5.1f}%  det-solved "
                  f"{e['det_query_solved']*100:5.1f}%  FALSE-ELIM {e['false_elim_count']}", flush=True)

    # ===== STAGE 2: generative readout (frozen organ -> OLMo LM head), readout-forcing curriculum =====
    from transformers import AutoTokenizer, AutoModelForCausalLM
    mid = "allenai/OLMo-2-0425-1B"
    print(f"\n================ STAGE 2: GENERATIVE READOUT ({mid}) ================", flush=True)
    tok = AutoTokenizer.from_pretrained(mid)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    probe = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16)
    D = probe.config.hidden_size; nL = probe.config.num_hidden_layers
    del probe
    print(f"OLMo: {nL} layers, hidden {D}", flush=True)

    prng = np.random.default_rng(a.seed + 11)
    t0 = time.time()
    train_pools = G.build_pool(prng, organ, dev, G.RUNGS, "id", a.per_rung_train, ("cells", "full"))
    train_cells, train_full = train_pools["cells"], train_pools["full"]
    eval_id = G.build_pool(prng, organ, dev, G.RUNGS, "id", a.per_rung_eval, ("cells", "full"))
    print(f"  data: {len(train_cells)} train recs/mode  built in {time.time()-t0:.0f}s", flush=True)

    model = G.train_readout(organ, (mid, D, nL), tok, dev, a, train_cells, train_full, eval_id["cells"])

    # ===== PRECONDITION SANITY: the staged result must hold (organ wielded under AGREEMENT) =====
    print("\n================ PRECONDITION: staged organ-use (no conflict) ================", flush=True)
    woven = G.score_gen(model, eval_id["full"], tok, dev, control="true", inject=True, bs=a.bs)
    txtl = G.score_gen(model, eval_id["full"], tok, dev, inject=False, bs=a.bs)
    basef = G.score_gen(model, eval_id["full"], tok, dev, inject=False, use_base=True,
                        fewshot=G.FEWSHOT, bs=a.bs)
    corr = G.score_gen(model, eval_id["full"], tok, dev, control="corrupt", inject=True, bs=a.bs)
    print(f"  in-dist(full): WOVEN {woven['overall']*100:.1f}%  TEXT-LoRA {txtl['overall']*100:.1f}%  "
          f"BASE {basef['overall']*100:.1f}%  | CORRUPT-organ {corr['overall']*100:.1f}%  "
          f"(corrupt drop {(woven['overall']-corr['overall'])*100:.1f}pts)", flush=True)
    print("  (corrupt drop with text PRESENT => the LM does not just read the text; necessary but NOT "
          "sufficient for WIELDS — the conflict test below is the discriminator.)", flush=True)

    # ===== build the SPLIT-BRAIN conflict pools (organ CORRECT, text adversarial), id + OOD =====
    print("\n================ BUILD SPLIT-BRAIN CONFLICT POOLS ================", flush=True)
    sb_id = build_conflict_pools(prng, organ, dev, G.RUNGS, "id", a.sb_per_rung_det, a.sb_per_rung_ab)
    sb_ood = build_conflict_pools(prng, organ, dev, G.RUNGS, "ood", a.sb_per_rung_det, a.sb_per_rung_ab)
    for split, sb in (("in-dist", sb_id), ("OOD", sb_ood)):
        print(f"  {split}: " + "  ".join(f"{c}={len(sb[c])}" for c in CONDITIONS), flush=True)

    # ===== THE SPLIT-BRAIN TABLE: follow-organ vs follow-text x 4 conditions x {id,OOD} =====
    print("\n================ SPLIT-BRAIN TABLE (organ CORRECT, text WRONG) ================", flush=True)
    print("  WOVEN row = the experiment.  TEXT-ONLY / CORRUPT-organ rows = calibration (must follow"
          " text).", flush=True)
    table = {}
    for split, sb in (("in-dist", sb_id), ("OOD", sb_ood)):
        print(f"\n  ===== {split} =====", flush=True)
        table[split] = {}
        for cond in CONDITIONS:
            pool = sb[cond]
            if not pool:
                print(f"  -- {cond}: (no instances) --", flush=True)
                continue
            woven_c = score_conflict(model, pool, tok, dev, inject=True, control="true", bs=a.bs)
            textonly = score_conflict(model, pool, tok, dev, inject=False, bs=a.bs)
            corrupt_c = score_conflict(model, pool, tok, dev, inject=True, control="corrupt", bs=a.bs)
            table[split][cond] = {"woven": woven_c, "text_only": textonly, "corrupt": corrupt_c}
            print(f"  -- {cond}  (n={woven_c['n']}) --", flush=True)
            print(_fmt("WOVEN (organ injected)", woven_c), flush=True)
            print(_fmt("TEXT-ONLY (no organ)", textonly), flush=True)
            print(_fmt("WOVEN, organ CORRUPTED", corrupt_c), flush=True)

    # ===== VERDICT =====
    print("\n================ VERDICT ================", flush=True)
    id_fo = np.mean([table["in-dist"][c]["woven"]["follow_organ"] for c in CONDITIONS
                     if c in table["in-dist"]])
    id_to_woven = np.mean([table["in-dist"][c]["woven"]["follow_text"] for c in CONDITIONS
                           if c in table["in-dist"]])
    id_to_text = np.mean([table["in-dist"][c]["text_only"]["follow_organ"] for c in CONDITIONS
                          if c in table["in-dist"]])
    ood_fo = np.mean([table["OOD"][c]["woven"]["follow_organ"] for c in CONDITIONS
                      if c in table["OOD"]])
    print(f"  mean follow-ORGAN under conflict: in-dist {id_fo*100:.1f}%   OOD {ood_fo*100:.1f}%", flush=True)
    print(f"  mean follow-TEXT  under conflict (woven): in-dist {id_to_woven*100:.1f}%", flush=True)
    print(f"  text-only path's follow-organ (should be ~0, it has no organ): {id_to_text*100:.1f}%",
          flush=True)
    if id_fo >= 0.66:
        print("  => follow-organ DOMINANT: the LM TRUSTS the organ over misleading text => it genuinely"
              " WIELDS the organ. The STRONG necessity claim holds.", flush=True)
    elif id_fo <= 0.34:
        print("  => follow-TEXT DOMINANT: the LM overrides the organ when text disagrees => the organ is"
              " consulted OPPORTUNISTICALLY (used only when it agrees). The WEAKER/brittler claim — the"
              " SATNet opportunistic-agreement failure mode. Reported honestly.", flush=True)
    else:
        print("  => MIXED: the LM partly wields, partly defers to text. Necessity is REAL but BRITTLE"
              " under adversarial text. Reported honestly.", flush=True)

    out = a.out or os.path.join(os.path.dirname(__file__), "..", "runs", "splitbrain.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    blob = {"args": vars(a), "organ_recall": {f"{rg}/{sp}": recall[(rg, sp)] for rg in G.RUNGS
                                               for sp in ("id", "ood")},
            "precondition": {"woven": woven["overall"], "text_lora": txtl["overall"],
                             "base": basef["overall"], "corrupt": corr["overall"]},
            "splitbrain": table,
            "summary": {"id_follow_organ": float(id_fo), "ood_follow_organ": float(ood_fo),
                        "id_follow_text_woven": float(id_to_woven),
                        "text_only_follow_organ": float(id_to_text)}}
    json.dump(blob, open(out, "w"), indent=1, default=float)
    print("\nwrote", out, flush=True)


if __name__ == "__main__":
    main()
