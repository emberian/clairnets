"""clair/organ/alpha_struct_diag.py — the ALPHA_STRUCT acceptance gate (the SHUFFLED test, INVERTED).

Under ALPHA_STRUCT there is NO metadata to swap — the structure IS α's output (the composer literally
never receives rec["csp"], enforced by bank_woven.forbid_record_csp). So the SHUFFLED arm is impossible
by construction and the success criterion inverts (notes/alpha_struct_design.md §4): instead of "does the
answer follow swapped metadata?" we ask "does the answer COLLAPSE when α's INPUT is corrupted, and does
accuracy TRACK α-faithfulness?".

Four arms on the SAME held-out determined-query pool:
  * ALPHA_STRUCT (true) : α reads instance i's host hidden → emits csp_α → composer → γ → answer.
  * α-INPUT-CORRUPT     : α reads instance j's text (rolled within an (n,d)-group) while the gen-stream +
                          query stay i's. α emits the WRONG structure; the composer faithfully solves it.
                          Accuracy-vs-true must COLLAPSE toward NO_STRUCT (the exact inversion of SHUFFLED).
  * EMPTY_STRUCT        : α forced to emit no relations (full domains) → composer = identity → no info.
  * NO_STRUCT           : gate-zero / no injection — the text-only floor anchor.

PASS iff (design §4.2): lift >= .30 ; α-corrupt + empty collapse to ~NO_STRUCT ; accuracy tracks
faithfulness (split >= .30 AND corr > .2) ; canonical F1 >= .80 ; and the code-level no-metadata invariant
(csp_spec_for_record RAISES inside the forward). The headline forward never uses the output-check.

  python -m clair.organ.alpha_struct_diag --smoke      # CPU structural smoke (tiny random host, no OLMo)
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from ..oracle_readout import _mention_tensor
from .alpha_struct import output_check
from .bank_woven import forbid_record_csp, csp_spec_for_record


# ---------------------------------------------------------------- α-input derangement (same n,d group)
def alpha_input_perm(recs, rng, key=("relation", "n")):
    """Within-group derangement perm[i]=j (j!=i, same relation+n so α still applies). Singleton groups
    map to themselves (uninformative; dropped from the corrupt arm's denominator)."""
    groups: dict = {}
    for i, r in enumerate(recs):
        k = tuple(r[f] for f in key)
        groups.setdefault(k, []).append(i)
    perm = list(range(len(recs)))
    corrupted = [False] * len(recs)
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        order = list(idxs); rng.shuffle(order)
        roll = order[1:] + order[:1]
        for i, j in zip(order, roll):
            perm[i] = j; corrupted[i] = True
    return perm, corrupted


# ---------------------------------------------------------------- composer-capture (α emits the structure)
@torch.no_grad()
def capture_surv(model, recs, tok, dev, two_stream=True, empty_struct=False, alpha_perm=None):
    """Run the capture forward: α reads the (optionally permuted, for α-INPUT-CORRUPT) α-stream → emits
    csp_α → composer. Returns (surv [B,Nmax,K], emitted_facts). rec['csp'] is NEVER threaded."""
    from .. import run_glados_staged as G
    B = len(recs); Nmax = max(r["n"] for r in recs)
    src = recs if alpha_perm is None else [recs[alpha_perm[i]] for i in range(len(recs))]
    cap_p = [s.get("alpha_prompt", s["prompt"]) for s in src] if two_stream else [s["prompt"] for s in src]
    cap_m = [s.get("alpha_mentions", s["mentions"]) for s in src] if two_stream else [s["mentions"] for s in src]
    penc = tok(cap_p, return_offsets_mapping=True, padding=True, return_tensors="pt")
    pids = penc["input_ids"].to(dev); pattn = penc["attention_mask"].to(dev)
    pment = _mention_tensor(cap_m, penc["offset_mapping"], B, Nmax, pids.size(1), dev)
    nd = [(r["n"], len(r["vnames"])) for r in recs]
    dvec = torch.tensor([len(r["vnames"]) for r in recs], device=dev)
    with model.live(pment, pattn, inject=False, capture=True, nd=nd, dvec=dvec, empty_struct=empty_struct):
        _ = model.logits(pids, pattn)
    return model._captured_surv[:, :Nmax, :].clone(), list(model._last_facts or [])


# ---------------------------------------------------------------- the 4-arm acceptance gate
@torch.no_grad()
def run_gate(model, recs, tok, dev, bs=8, two_stream=True, seed=0, verbose=True):
    """The inverted SHUFFLED acceptance gate over `recs` (held-out determined queries). Returns the
    per-arm accuracies, the structure-F1, the faithfulness split, the invariant-guard result, and the
    PASS/FAIL verdict (design §4.2)."""
    from .. import run_glados_staged as G
    from .. import curriculum as CU
    from .shuffled_diag import score_preds
    K = G.K
    model.eval()
    rng = np.random.default_rng(seed + 909)
    perm, corrupted = alpha_input_perm(recs, rng)
    true_gold = [r["gold_idx"] for r in recs]

    # capture composed lattices for the three injected arms (host hidden = i; only α's INPUT/EMPTY differ)
    true_surv = [None] * len(recs); corr_surv = [None] * len(recs); empty_surv = [None] * len(recs)
    emitted = [None] * len(recs)
    for i in range(0, len(recs), bs):
        chunk = list(range(i, min(i + bs, len(recs)))); hr = [recs[c] for c in chunk]
        ts, tf = capture_surv(model, hr, tok, dev, two_stream)
        sub_perm = [perm[c] - i if (i <= perm[c] < i + len(chunk)) else None for c in chunk]
        # α-corrupt within the chunk where the partner is in-chunk; else self (uninformative)
        cp = [(perm[c] - i) if (i <= perm[c] < i + len(chunk)) else li for li, c in enumerate(chunk)]
        cs, _ = capture_surv(model, hr, tok, dev, two_stream, alpha_perm=cp)
        es, _ = capture_surv(model, hr, tok, dev, two_stream, empty_struct=True)
        for li, c in enumerate(chunk):
            true_surv[c] = ts[li, : recs[c]["n"]].cpu().numpy()
            corr_surv[c] = cs[li, : recs[c]["n"]].cpu().numpy()
            empty_surv[c] = es[li, : recs[c]["n"]].cpu().numpy()
            emitted[c] = tf[li] if li < len(tf) else []

    def arm(src, inject):
        preds = [None] * len(recs)
        for i in range(0, len(recs), bs):
            chunk = list(range(i, min(i + bs, len(recs)))); hr = [recs[c] for c in chunk]
            Nmax = max(r["n"] for r in hr)
            if not inject:
                surv = torch.zeros(len(chunk), Nmax, K, device=dev)
                pr = score_preds(model, hr, surv, tok, dev, inject=False)
            else:
                surv = torch.zeros(len(chunk), Nmax, K, device=dev)
                for li, c in enumerate(chunk):
                    s = src[c]; surv[li, : s.shape[0]] = torch.from_numpy(s).to(dev)
                pr = score_preds(model, hr, surv, tok, dev, inject=True)
            for li, c in enumerate(chunk):
                preds[c] = pr[li]
        return preds

    p_true = arm(true_surv, True); p_corr = arm(corr_surv, True)
    p_empty = arm(empty_surv, True); p_none = arm(None, False)

    idx = list(range(len(recs)))
    cidx = [i for i in idx if corrupted[i]]                     # informative for the α-corrupt arm

    def acc(preds, ii=idx):
        return sum(preds[i] == true_gold[i] for i in ii) / max(1, len(ii))

    acc_true, acc_corr = acc(p_true), acc(p_corr, cidx)
    acc_empty, acc_none = acc(p_empty), acc(p_none)

    # structure-F1 + faithfulness (α-emitted vs the witness's free true facts)
    tp = fp = fn = 0; faithful = [False] * len(recs)
    for i in idx:
        pred = set(CU.norm_facts([tuple(f) for f in (emitted[i] or [])]))
        true = set(CU.norm_facts([tuple(f) for f in recs[i].get("facts", [])]))
        tp += len(pred & true); fp += len(pred - true); fn += len(true - pred)
        faithful[i] = (pred == true)
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
    f1 = 2 * P * R / max(1e-9, P + R)

    correct = np.array([float(p_true[i] == true_gold[i]) for i in idx])
    faith = np.array([float(faithful[i]) for i in idx])
    fa = [i for i in idx if faithful[i]]; nfa = [i for i in idx if not faithful[i]]
    acc_faith = acc(p_true, fa) if fa else float("nan")
    acc_nfaith = acc(p_true, nfa) if nfa else float("nan")
    corr = float(np.corrcoef(faith, correct)[0, 1]) if faith.std() > 0 and correct.std() > 0 else 0.0

    # output-check (reliability layer; NOT in the headline forward): is the LM's answer a real solution?
    oc = 0
    for i in idx:
        ci = recs[i].get("csp"); pi = p_true[i]
        if ci is not None and pi is not None and pi < len(recs[i]["vnames"]):
            oc += int(output_check(pi, ci, recs[i]["query"]))      # licensed reader, OUTSIDE the forward
    output_check_rate = oc / max(1, len(idx))

    # the code-level no-metadata invariant (design §4.2 crit 5)
    guard_fires = _invariant_guard_fires(recs[0] if recs else {"csp": "x"})

    split = (acc_faith - acc_nfaith) if (fa and nfa) else float("nan")
    crit = {
        "lift": acc_true - acc_none,
        "collapse_corrupt": acc_true - acc_corr,
        "collapse_empty": acc_true - acc_empty,
        "faithfulness_split": split,
        "faithfulness_corr": corr,
        "struct_F1": f1,
        "invariant_guard_fires": guard_fires,
    }
    passed = bool(
        crit["lift"] >= 0.30
        and acc_corr <= acc_none + 0.05 and acc_empty <= acc_none + 0.05
        and (np.isnan(split) or split >= 0.30) and corr > 0.2
        and f1 >= 0.80 and guard_fires)
    out = {
        "n": len(recs), "n_corrupt_informative": len(cidx),
        "ALPHA_STRUCT_true": acc_true, "ALPHA_INPUT_CORRUPT": acc_corr,
        "EMPTY_STRUCT": acc_empty, "NO_STRUCT": acc_none,
        "struct_F1": f1, "acc_faithful": acc_faith, "acc_unfaithful": acc_nfaith,
        "output_check_rate": output_check_rate, "criteria": crit, "PASS": passed,
    }
    if verbose:
        print_gate(out)
    return out


def _invariant_guard_fires(rec) -> bool:
    """The code-level no-metadata invariant: csp_spec_for_record MUST raise inside forbid_record_csp()."""
    try:
        with forbid_record_csp():
            csp_spec_for_record(rec)
        return False
    except RuntimeError:
        return True


def print_gate(out):
    print("\n" + "=" * 80, flush=True)
    print("ALPHA_STRUCT ACCEPTANCE GATE — the SHUFFLED test, INVERTED (α DRIVES correctness?)", flush=True)
    print("=" * 80, flush=True)
    print(f"  instances={out['n']}  α-corrupt-informative={out['n_corrupt_informative']}", flush=True)
    print(f"  {'arm':22s} {'acc-vs-TRUE':>12s}", flush=True)
    for a in ("ALPHA_STRUCT_true", "ALPHA_INPUT_CORRUPT", "EMPTY_STRUCT", "NO_STRUCT"):
        print(f"  {a:22s} {out[a]*100:11.1f}%", flush=True)
    c = out["criteria"]
    print(f"  structure-F1={out['struct_F1']:.3f}  acc|faithful={out['acc_faithful']}  "
          f"acc|¬faithful={out['acc_unfaithful']}  output-check={out['output_check_rate']:.3f}", flush=True)
    print(f"  criteria: lift={c['lift']:+.3f} collapse(corrupt)={c['collapse_corrupt']:+.3f} "
          f"collapse(empty)={c['collapse_empty']:+.3f} faith-split={c['faithfulness_split']} "
          f"corr={c['faithfulness_corr']:+.3f} guard={c['invariant_guard_fires']}", flush=True)
    print(f"  VERDICT: {'PASS — α DRIVES correctness (collapses under input-corrupt; tracks faithfulness; '
          'no metadata in the forward).' if out['PASS'] else 'FAIL — see criteria.'}", flush=True)
    print("=" * 80 + "\n", flush=True)


# ================================================================ CPU structural smoke (tiny random host)
def structural_smoke():
    """Build the ALPHA_STRUCT woven on a TINY RANDOM host (no OLMo, CPU): prove (a) it builds, (b) the
    forward runs (α emits → composer on csp_α → γ), (c) bitwise no-op@init / gate-on moves the residual,
    (d) the no-metadata invariant guard fires. No training, no download."""
    from transformers import LlamaConfig, LlamaForCausalLM
    from peft import LoraConfig, get_peft_model
    from .alpha_struct import StructureRack
    from .bank_woven import AlphaStructWoven, AlphaStructComposerOrgan
    from .. import csp as C

    torch.manual_seed(0); np.random.seed(0)
    D, K, nL, V = 64, 8, 4, 64
    cfg = LlamaConfig(hidden_size=D, intermediate_size=128, num_hidden_layers=nL,
                      num_attention_heads=4, num_key_value_heads=4, vocab_size=V,
                      max_position_embeddings=64)
    base = LlamaForCausalLM(cfg)
    peft = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                                           target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    rack = StructureRack(D, K, dp=32, heads=4, hidden=64)
    # no checkpoint -> certified-floor-only composer (the pretrained organ plugs in when present)
    composer = AlphaStructComposerOrgan(dev="cpu", core_ckpt="runs/__nonexistent__.pt", use_core=True)
    model = AlphaStructWoven(peft, D, K, rack, composer, mid_layer=1, inject_layer=3, gamma_hidden=32)
    model.alpha.float(); model.cand.float(); model.gamma.float()
    print(f"[smoke] built AlphaStructWoven  D={D} K={K} layers={nL}  faculties={composer.faculties()}")

    B, N, T = 2, 3, 12
    ids = torch.randint(0, V, (B, T)); attn = torch.ones(B, T)
    mention = torch.zeros(B, N, T)
    for b in range(B):
        for i in range(N):
            mention[b, i, 2 * i + 1] = 1.0
    nd = [(3, 3), (3, 3)]; dvec = torch.tensor([3, 3])

    with torch.no_grad():
        base_logits = model.model(input_ids=ids, attention_mask=attn).logits.float()
    with torch.no_grad(), model.live(mention, attn, inject=True, capture=True, nd=nd, dvec=dvec):
        g0 = model.logits(ids, attn).float()
    noop = float((base_logits - g0).abs().max())
    assert model._last_facts is not None, "forward did not emit/compose a structure"
    print(f"[smoke] forward ran; α emitted facts (inst0)={model._last_facts[0]}  composer produced surv")
    assert noop < 1e-4, f"NOT a no-op@init: max|base-(gate0)|={noop:.3e}"
    print(f"[smoke] NO-OP @ INIT  max|base-(gate0)|={noop:.3e}  (PASS)")

    with torch.no_grad():
        saved = model.gamma.alpha.data.clone(); model.gamma.alpha.data.fill_(2.0)
        with model.live(mention, attn, inject=True, capture=True, nd=nd, dvec=dvec):
            g1 = model.logits(ids, attn).float()
        model.gamma.alpha.data.copy_(saved)
    moved = float((base_logits - g1).abs().max())
    assert moved > 0.0, "gate-on did not move the residual stream"
    print(f"[smoke] gate-on moves residual  max|base-(gate2)|={moved:.3e}  (PASS)")

    # EMPTY_STRUCT control: α forced to emit nothing -> composer = identity -> full lattice
    with torch.no_grad(), model.live(mention, attn, inject=False, capture=True, nd=nd, dvec=dvec,
                                     empty_struct=True):
        _ = model.logits(ids, attn)
    assert all(f == [] for f in model._last_facts), "EMPTY_STRUCT did not force empty structure"
    print(f"[smoke] EMPTY_STRUCT forces contentless structure  (PASS)")

    # the no-metadata invariant: csp_spec_for_record raises inside the forward guard
    rec = {"csp": C.coloring(3, [(0, 1)], k=3), "vnames": ["red", "green", "blue"]}
    fires = _invariant_guard_fires(rec)
    assert fires, "INVARIANT GUARD did not fire — rec['csp'] readable inside the forward!"
    print(f"[smoke] invariant guard FIRES on rec['csp'] read inside forbid_record_csp()  (PASS)")

    model.remove_hooks()
    print("\n[smoke] ALL STRUCTURAL CHECKS PASS — AlphaStructWoven builds, forward runs on csp_α, "
          "no-op@init, gate moves, EMPTY_STRUCT control, invariant guard fires.\n")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU structural smoke (tiny random host, no OLMo)")
    a = ap.parse_args()
    if a.smoke:
        structural_smoke()
    else:
        raise SystemExit("run_gate(model, recs, tok, dev) is the eval-time entry (needs a trained model). "
                         "For the CPU wiring proof: python -m clair.organ.alpha_struct_diag --smoke")


if __name__ == "__main__":
    main()
