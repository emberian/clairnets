"""clair/organ/run_alpha_struct_search.py — α-AS-SEARCH: the residual NL lever after LoRA.

The bidirectional-LEARNING run (run_alpha_struct_lora.py) closed 36% of the NL gap: frozen-host
NL-OOD micro-F1 0.728 → LoRA-host 0.819, canonical ceiling 0.98. The remaining ~64% is the OTHER
bidirectional lever — bidirectional INFERENCE: α PROPOSES candidate structures, the SOUND certified
verifier (clair.csp.exact_dedP) SELECTS. The premise: when α's top-1 (greedy) compile of a hard NL
problem is wrong, a SAMPLED candidate is often right, and the certified solver can tell which
candidate is SOLVABLE (not ⊥) and DETERMINES the query — so pass@K collapses toward pass@1 through a
SOUND selector (the same test-time-search lever eval_suite applies to answers, here applied to α's
COMPILE).

Pipeline (identical LoRA-host α as run_alpha_struct_lora — same data/head/lr/layer; train fresh or
load --rack_ckpt):
  1. SAMPLE K candidate structures per OOD problem (greedy K=1 + temperature-sampled pin & pair-
     relation logits). K ∈ {1,2,4,8,16}; K=1 = the LoRA greedy baseline.
  2. VERIFIER-SELECT with the certified solver: for each candidate build the CSP and run
     exact_dedP — accept candidates that are SOLVABLE and DETERMINE the query cell; among those
     prefer the model's own structure-confidence (joint head log-prob) [sound: structure+solver
     only, no gold]. Also report a consensus (plurality answer) sound selector.
  3. ORACLE@K (gold breaks ties) — the pass@K ceiling, to bound how much search COULD recover.

Reports structure micro-F1 + exact-graph-match + answer-accuracy on NL-OOD as a function of K for
greedy(K=1) vs verifier-selected vs oracle@K, plus the verifier disambiguation diagnostic (when
multiple solvable+determining candidates disagree on the answer — where the sound selector CAN'T
disambiguate, and whether the gold answer is even in the verifier-passing set). Writes
runs/alpha_struct_search_nl.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from .. import csp as C
from .. import curriculum as CU
from .alpha_struct import StructureRack, PAIR_RELS
from .run_alpha_struct_probe import K, load_csp_records, rec_view, csp_targets
from .run_alpha_struct_lora import eval_views, train_lora_csp

ABSTAIN = CU.ABSTAIN
RELSET = ("pin", "eq", "neq", "lt", "le")


# ============================================================ candidate decode + verifier
def decode_idx(pin_idx, pair_idx, vmask, n):
    """(pin_idx[Nmax], pair_idx[Nmax,Nmax]) chosen classes → normalized predicted fact set.
    Identical decode semantics to run_alpha_struct_probe.decode_facts, but over GIVEN class indices
    (argmax for greedy, multinomial sample otherwise)."""
    facts = []
    for i in range(n):
        if vmask[i] > 0.5 and int(pin_idx[i]) != 0:
            facts.append(("pin", i, int(pin_idx[i]) - 1))
    for i in range(n):
        if vmask[i] < 0.5:
            continue
        for j in range(n):
            if i == j or vmask[j] < 0.5:
                continue
            cls = PAIR_RELS[int(pair_idx[i, j])]
            if cls in ("eq", "neq", "lt", "le"):
                facts.append((cls, i, j))
    return CU.norm_facts(facts)


def verify_structure(facts, n, d, query):
    """Run the SOUND certified verifier (exact_dedP) on the candidate's CSP. Returns
    (solvable, determines_query, answer). exact_dedP[i] = values cell i takes in SOME solution;
    ⊥ (unsat) ⇒ every cell empty. The query is DETERMINED iff its cell is a singleton."""
    csp = CU.build_csp(n, d, list(facts))
    ded = C.exact_dedP(csp, csp.full())
    solvable = any(len(c) > 0 for c in ded)
    if not solvable:
        return False, False, ABSTAIN
    qcell = ded[query]
    if len(qcell) == 1:
        return True, True, int(next(iter(qcell)))
    return True, False, ABSTAIN


def fact_pr_counts(pred, true):
    """Per-relation (tp, fp, fn) Counters for one predicted vs true fact set."""
    tp = Counter(); fp = Counter(); fn = Counter()
    for f in pred:
        (tp if f in true else fp)[f[0]] += 1
    for f in true:
        if f not in pred:
            fn[f[0]] += 1
    return tp, fp, fn


def f1_from(tp, fp, fn):
    rep = {}
    aTP = aFP = aFN = 0
    for r in RELSET:
        p = tp[r] / max(1, tp[r] + fp[r]); rc = tp[r] / max(1, tp[r] + fn[r])
        rep[r] = round(2 * p * rc / max(1e-9, p + rc), 4)
        aTP += tp[r]; aFP += fp[r]; aFN += fn[r]
    P = aTP / max(1, aTP + aFP); Rc = aTP / max(1, aTP + aFN)
    rep["micro_F1"] = round(2 * P * Rc / max(1e-9, P + Rc), 4)
    return rep


# ============================================================ per-record candidate bank
def build_candidates(pin_l, pair_l, vmask, n, Kc, temp, rng):
    """Decode Kc candidate structures for one record from its logits. Candidate 0 = GREEDY (argmax,
    the LoRA baseline); 1..Kc-1 = temperature samples. Returns a list of dicts with the normalized
    facts and the model's joint structure log-prob (confidence)."""
    pin_lp = F.log_softmax(pin_l / temp, dim=-1)          # [Nmax,1+K]
    pair_lp = F.log_softmax(pair_l / temp, dim=-1)        # [Nmax,Nmax,R]
    pin_p = pin_lp.exp(); pair_p = pair_lp.exp()
    vbool = vmask > 0.5
    valid_cells = [i for i in range(n) if vbool[i]]
    valid_pairs = [(i, j) for i in valid_cells for j in valid_cells if i != j]

    def conf(pin_idx, pair_idx):
        s = 0.0
        for i in valid_cells:
            s += float(pin_lp[i, int(pin_idx[i])])
        for (i, j) in valid_pairs:
            s += float(pair_lp[i, j, int(pair_idx[i, j])])
        return s

    cands = []
    seen = {}
    for k in range(Kc):
        if k == 0:
            pin_idx = pin_l.argmax(-1)
            pair_idx = pair_l.argmax(-1)
        else:
            pin_idx = torch.multinomial(pin_p, 1, generator=rng).squeeze(-1)
            pflat = pair_p.reshape(-1, pair_p.size(-1))
            pair_idx = torch.multinomial(pflat, 1, generator=rng).squeeze(-1).reshape(pair_p.shape[:2])
        facts = decode_idx(pin_idx, pair_idx, vmask, n)
        cands.append({"facts": facts, "conf": conf(pin_idx, pair_idx)})
    return cands


# ============================================================ the K-sweep search eval
def search_eval(pin_all, pair_all, vmask_all, n_all, od_v, od, Ks, temp, seed=0, log=print):
    """For each OOD record decode max(Ks) candidates once, verify each with the certified solver, then
    for every K in Ks compute greedy / verifier-selected / oracle@K structure-F1, exact-graph-match,
    and answer-accuracy. Selection is sound ONLY RELATIVE TO csp_α (structure + solver-on-csp_α +
    model-confidence, no gold) — it filters compiles-to-solvable, NOT answer-correctness; a wrong csp_α
    still passes. Answer-soundness needs the output-check against csp_true (notes/soundness.md (B)/(D))."""
    R = len(od_v)
    Kmax = max(Ks)
    rng = torch.Generator().manual_seed(seed)
    # per-record candidate banks (decoded + verified once at Kmax)
    banks = []
    t0 = time.time()
    for ri in range(R):
        n = int(n_all[ri].item())
        d = int(od[ri]["d"]); query = int(od[ri]["query"])
        gold = CU.norm_facts(od_v[ri][3])
        gold_ans = int(od[ri]["answer"]); gold_det = bool(od[ri]["determined"])
        cands = build_candidates(pin_all[ri], pair_all[ri], vmask_all[ri], n, Kmax, temp, rng)
        for cd in cands:
            solv, det, ans = verify_structure(cd["facts"], n, d, query)
            cd.update(solvable=solv, determines=det, ans=ans)
            cd["exact"] = (cd["facts"] == gold)
            tp, fp, fn = fact_pr_counts(cd["facts"], gold)
            cd["tp"], cd["fp"], cd["fn"] = tp, fp, fn
            # per-candidate answer correctness (for oracle)
            if gold_det:
                cd["ans_ok"] = bool(det and ans == gold_ans)
            else:
                cd["ans_ok"] = bool(not det)
        banks.append({"cands": cands, "gold": gold, "gold_ans": gold_ans, "gold_det": gold_det,
                      "query": query, "n": n, "d": d})
        if (ri + 1) % 150 == 0:
            log(f"    decoded+verified {ri+1}/{R} records  {time.time()-t0:.0f}s")

    results = {}
    for Kc in Ks:
        modes = {m: {"tp": Counter(), "fp": Counter(), "fn": Counter(), "exact": 0, "ans_ok": 0}
                 for m in ("greedy", "verifier", "oracle")}
        # consensus (plurality answer among verifier-passing) answer accuracy
        vote_ok = 0
        # disambiguation diagnostic
        multi_solv_det = 0           # records with >=1 verifier-passing candidate
        disagree = 0                 # ... where passing candidates give >1 distinct answer
        gold_in_pass = 0             # ... where gold answer IS among passing answers
        n_with_pass = 0
        for bk in banks:
            cs = bk["cands"][:Kc]
            gold = bk["gold"]; gold_ans = bk["gold_ans"]; gold_det = bk["gold_det"]
            greedy = cs[0]
            # ---- greedy (K=1 baseline = candidate 0) ----
            _acc(modes["greedy"], greedy, gold_det, gold_ans)
            # ---- verifier-selected: solvable & determines query, max model-confidence; else greedy
            passing = [c for c in cs if c["solvable"] and c["determines"]]
            if passing:
                sel = max(passing, key=lambda c: c["conf"])
                n_with_pass += 1
                answers = [c["ans"] for c in passing]
                distinct = set(answers)
                if len(distinct) > 1:
                    disagree += 1
                if gold_det and gold_ans in distinct:
                    gold_in_pass += 1
                # consensus plurality answer (ties → highest-confidence among the plurality)
                cnt = Counter(answers)
                top = max(cnt.values())
                plural = [a for a, c in cnt.items() if c == top]
                if len(plural) == 1:
                    vote_ans, vote_det = plural[0], True
                else:
                    vote_ans = max((c for c in passing if c["ans"] in plural),
                                   key=lambda c: c["conf"])["ans"]
                    vote_det = True
            else:
                sel = greedy
                vote_ans, vote_det = greedy["ans"], greedy["determines"]
            _acc(modes["verifier"], sel, gold_det, gold_ans)
            # vote answer-acc: same correctness rule + denominator as the confidence selector
            vote_ok += int((vote_det and vote_ans == gold_ans) if gold_det else (not vote_det))
            # ---- oracle@K: gold breaks ties ----
            best = max(cs, key=lambda c: (c["exact"], -(_fp_fn(c))))   # exact first, then fewest errors
            o = modes["oracle"]
            o["tp"] += best["tp"]; o["fp"] += best["fp"]; o["fn"] += best["fn"]
            o["exact"] += int(any(c["exact"] for c in cs))
            o["ans_ok"] += int(any(c["ans_ok"] for c in cs))
        out = {}
        for m, acc in modes.items():
            rep = f1_from(acc["tp"], acc["fp"], acc["fn"])
            rep["exact_graph_match"] = round(acc["exact"] / R, 4)
            rep["answer_acc"] = round(acc["ans_ok"] / R, 4)
            out[m] = rep
        out["verifier"]["answer_acc_vote"] = round(vote_ok / R, 4)
        out["diag"] = {
            "records_with_verifier_pass": n_with_pass,
            "pass_rate": round(n_with_pass / R, 4),
            "disagree_among_pass": disagree,
            "disagree_rate_of_passing": round(disagree / max(1, n_with_pass), 4),
            "gold_answer_in_pass_set": gold_in_pass,
            "gold_in_pass_rate_of_passing": round(gold_in_pass / max(1, n_with_pass), 4),
        }
        results[Kc] = out
        log(f"  [K={Kc:2d}] greedy F1 {out['greedy']['micro_F1']:.3f} ans {out['greedy']['answer_acc']:.3f}"
            f" | verifier F1 {out['verifier']['micro_F1']:.3f} ans {out['verifier']['answer_acc']:.3f}"
            f" (vote {out['verifier']['answer_acc_vote']:.3f})"
            f" | oracle F1 {out['oracle']['micro_F1']:.3f} ex {out['oracle']['exact_graph_match']:.3f}"
            f" ans {out['oracle']['answer_acc']:.3f}")
    return results


def _fp_fn(c):
    return sum(c["fp"].values()) + sum(c["fn"].values())


def _acc(slot, cand, gold_det, gold_ans):
    slot["tp"] += cand["tp"]; slot["fp"] += cand["fp"]; slot["fn"] += cand["fn"]
    slot["exact"] += int(cand["exact"])
    slot["ans_ok"] += int(cand["ans_ok"])


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr_head", type=float, default=1.5e-3)
    ap.add_argument("--lr_lora", type=float, default=2e-4)
    ap.add_argument("--wpos", type=float, default=2.0)
    ap.add_argument("--wnone", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--micro_bs", type=int, default=24)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--Ks", default="1,2,4,8,16")
    ap.add_argument("--temps", default="1.0,1.5", help="candidate sampling temperatures to sweep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nl_train", required=True)
    ap.add_argument("--nl_ood", required=True)
    ap.add_argument("--rack_ckpt", default="runs/alpha_struct_search_rack.pt",
                    help="save/load the trained LoRA-host + rack here (load if exists)")
    ap.add_argument("--out", default="runs/alpha_struct_search_nl.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    L = a.layer
    Ks = [int(x) for x in a.Ks.split(",")]
    temps = [float(x) for x in a.temps.split(",")]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    print(f"loading {a.model} on {dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    olmo = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev)
    D = olmo.config.hidden_size
    cfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
                     bias="none", task_type="CAUSAL_LM")
    olmo = get_peft_model(olmo, cfg)
    if not a.no_grad_ckpt:
        olmo.gradient_checkpointing_enable(); olmo.enable_input_require_grads()
    lora_params = [p for _, p in olmo.named_parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in lora_params)
    print(f"OLMo D={D}, layers {olmo.config.num_hidden_layers}; LoRA trainable {n_lora:,} (r={a.lora_r}); "
          f"reading L{L}", flush=True)

    tr = load_csp_records(a.nl_train)
    od = load_csp_records(a.nl_ood)
    tr_v = [rec_view(r, "nl") for r in tr]
    od_v = [rec_view(r, "nl") for r in od]
    Nmax = max(max(v[2] for v in tr_v), max(v[2] for v in od_v))
    print(f"NL-frontier: {len(tr_v)} train, {len(od_v)} OOD; Nmax {Nmax}", flush=True)

    rack = StructureRack(D, K).to(dev)
    rack.encoder.float(); rack.heads["csp"].float()

    if os.path.exists(a.rack_ckpt):
        print(f"loading cached LoRA-host + rack from {a.rack_ckpt}", flush=True)
        blob = torch.load(a.rack_ckpt, map_location=dev)
        rack.load_state_dict(blob["rack"])
        set_peft_model_state_dict(olmo, blob["lora"])
    else:
        print("\n== JOINT LoRA-host + α-StructureHead training (NL-frontier; identical to LoRA A/B) ==",
              flush=True)
        pin_t, pair_t = csp_targets(tr_v, Nmax, "cpu")
        train_lora_csp(olmo, tok, rack, dev, tr_v, pin_t, pair_t, Nmax, lora_params,
                       steps=a.steps, lr_head=a.lr_head, lr_lora=a.lr_lora, bs=a.bs,
                       micro_bs=a.micro_bs, wpos=a.wpos, wnone=a.wnone, L=L)
        os.makedirs(os.path.dirname(a.rack_ckpt) or ".", exist_ok=True)
        torch.save({"rack": rack.state_dict(), "lora": get_peft_model_state_dict(olmo)}, a.rack_ckpt)
        print(f"  saved {a.rack_ckpt}", flush=True)

    # ---- one no-grad forward to logits over the OOD set ----
    print("\n== NL-OOD logits (LoRA-host) ==", flush=True)
    pin_all, pair_all, vmask_all, n_all = eval_views(olmo, tok, rack, dev, od_v, L, Nmax, a.micro_bs)

    LORA_GREEDY = {"nl_ood_microF1": 0.8189, "exact": 0.2437}     # runs/alpha_struct_lora_nl.json
    FROZEN = {"nl_ood_microF1": 0.7277}
    CEILING = 0.98
    all_temps = {}
    for temp in temps:
        print(f"\n== α-as-search K-sweep (temp {temp}) ==", flush=True)
        all_temps[f"T={temp}"] = search_eval(pin_all, pair_all, vmask_all, n_all, od_v, od,
                                             Ks, temp, seed=a.seed)

    out = {
        "model": a.model, "layer": L, "lever": "α-as-search (verifier-gated test-time search on the compile)",
        "lora": {"r": a.lora_r, "alpha": a.lora_alpha, "trainable_params": int(n_lora)},
        "train_cfg": {"steps": a.steps, "lr_head": a.lr_head, "lr_lora": a.lr_lora,
                      "wpos": a.wpos, "wnone": a.wnone, "bs": a.bs, "micro_bs": a.micro_bs},
        "data": {"nl_train": a.nl_train, "nl_ood": a.nl_ood, "n_ood": len(od_v)},
        "Ks": Ks, "temps": temps, "seed": a.seed,
        "baselines": {"frozen": FROZEN, "lora_greedy": LORA_GREEDY, "canonical_ceiling": CEILING},
        "search": all_temps,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)

    # ---- verdict table (best temp by verifier answer-acc at max K) ----
    Kmax = max(Ks)
    best_temp = max(temps, key=lambda t: all_temps[f"T={t}"][Kmax]["verifier"]["answer_acc"])
    sweep = all_temps[f"T={best_temp}"]
    print(f"\n==== α-AS-SEARCH VERDICT (NL-OOD; best temp {best_temp}) ====", flush=True)
    print(f"  frozen 0.728 | LoRA-greedy 0.819 | canonical ceiling 0.98", flush=True)
    print(f"  {'K':>3} | {'greedy-F1':>9} {'verif-F1':>9} {'oracle-F1':>9} | "
          f"{'greedy-ans':>10} {'verif-ans':>9} {'vote-ans':>8} {'oracle-ans':>10} | "
          f"{'gr-exact':>8} {'or-exact':>8} | {'pass':>5} {'disag':>5} {'gold∈':>5}", flush=True)
    for Kc in Ks:
        r = sweep[Kc]; dg = r["diag"]
        print(f"  {Kc:>3} | {r['greedy']['micro_F1']:>9.3f} {r['verifier']['micro_F1']:>9.3f} "
              f"{r['oracle']['micro_F1']:>9.3f} | {r['greedy']['answer_acc']:>10.3f} "
              f"{r['verifier']['answer_acc']:>9.3f} {r['verifier']['answer_acc_vote']:>8.3f} "
              f"{r['oracle']['answer_acc']:>10.3f} | {r['greedy']['exact_graph_match']:>8.3f} "
              f"{r['oracle']['exact_graph_match']:>8.3f} | {dg['pass_rate']:>5.2f} "
              f"{dg['disagree_rate_of_passing']:>5.2f} {dg['gold_in_pass_rate_of_passing']:>5.2f}",
              flush=True)
    vF1 = sweep[Kmax]["verifier"]["micro_F1"]; oF1 = sweep[Kmax]["oracle"]["micro_F1"]
    closed = (vF1 - LORA_GREEDY["nl_ood_microF1"]) / (CEILING - LORA_GREEDY["nl_ood_microF1"])
    print(f"\n  verifier F1 closes {closed*100:.0f}% of the residual LoRA(0.819)→ceiling(0.98) gap "
          f"(verifier {vF1:.3f}, oracle@{Kmax} {oF1:.3f})", flush=True)
    print(f"wrote {a.out}", flush=True)


# peft state-dict helpers imported lazily so the module imports without peft on a CPU box
try:
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
except Exception:
    get_peft_model_state_dict = set_peft_model_state_dict = None


if __name__ == "__main__":
    main()
