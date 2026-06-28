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
def capture_surv(model, recs, tok, dev, two_stream=True, empty_struct=False, alpha_perm=None,
                 search_K=1, search_temp=1.0, loop_T=1, search_rng=None):
    """Run the capture forward: α reads the (optionally permuted, for α-INPUT-CORRUPT) α-stream → emits
    csp_α → composer. Returns (surv [B,Nmax,K], emitted_facts). rec['csp'] is NEVER threaded.

    search_K>1 ⇒ LEVER A (α-as-search) selects the structure; loop_T>1 ⇒ LEVER B (iterative α↔organ
    loop). The query cell (for the verifier 'determines-query' selector) is threaded from the records —
    NOT rec['csp']; the certified output_check stays OUTSIDE this forward (the reliability layer)."""
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
    queries = [int(r["query"]) for r in recs]
    with model.live(pment, pattn, inject=False, capture=True, nd=nd, dvec=dvec, empty_struct=empty_struct,
                    search_K=search_K, search_temp=search_temp, loop_T=loop_T, queries=queries,
                    search_rng=search_rng):
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
        "struct_F1": f1, "exact_match": float(np.mean([1.0 if faithful[i] else 0.0 for i in idx])),
        "acc_faithful": acc_faith, "acc_unfaithful": acc_nfaith,
        "output_check_rate": output_check_rate, "criteria": crit, "PASS": passed,
    }
    if verbose:
        print_gate(out)
    return out


# ---------------------------------------------------------------- greedy vs search vs loop (the levers)
@torch.no_grad()
def compare_modes(model, recs, tok, dev, bs=8, two_stream=True, search_K=8, search_temp=1.0, loop_T=2,
                  seed=0, verbose=True):
    """Measure the TRUE arm under the three inference modes the levers add (the acceptance gate's
    lever readout): greedy (K=1,T=1 — the single-shot core), α-as-search (K>1,T=1), the iterative
    α↔organ loop (K=1,T>1), and search+loop. Per mode: determined-query accuracy + the certified
    output_check rate (the SOUND reliability gate — answer ∈ exact_dedP(csp_true)[query]). The
    output_check runs OUTSIDE the forward (licensed reader of rec['csp'])."""
    from .shuffled_diag import score_preds
    from .. import run_glados_staged as G
    K = G.K
    model.eval()
    true_gold = [r["gold_idx"] for r in recs]
    rng = torch.Generator(device=dev).manual_seed(seed + 4242)   # match the rack's tensor device (CUDA)
    modes = {"greedy": (1, 1), "search": (search_K, 1), "loop": (1, loop_T),
             "search+loop": (search_K, loop_T)}
    out = {}
    for name, (sk, lt) in modes.items():
        preds = [None] * len(recs)
        for i in range(0, len(recs), bs):
            chunk = list(range(i, min(i + bs, len(recs)))); hr = [recs[c] for c in chunk]
            Nmax = max(r["n"] for r in hr)
            surv_b, _ = capture_surv(model, hr, tok, dev, two_stream, search_K=sk, search_temp=search_temp,
                                     loop_T=lt, search_rng=rng)
            surv = torch.zeros(len(chunk), Nmax, K, device=dev)
            for li, c in enumerate(chunk):
                surv[li, : recs[c]["n"]] = surv_b[li, : recs[c]["n"]]
            pr = score_preds(model, hr, surv, tok, dev, inject=True)
            for li, c in enumerate(chunk):
                preds[c] = pr[li]
        acc = sum(preds[i] == true_gold[i] for i in range(len(recs))) / max(1, len(recs))
        oc = 0
        for i in range(len(recs)):
            ci = recs[i].get("csp"); pi = preds[i]
            if ci is not None and pi is not None and pi < len(recs[i]["vnames"]):
                oc += int(output_check(pi, ci, recs[i]["query"]))      # certified gate, OUTSIDE the forward
        out[name] = {"acc": acc, "output_check_rate": oc / max(1, len(recs)), "search_K": sk, "loop_T": lt}
    if verbose:
        print("\n  ALPHA_STRUCT inference levers (TRUE arm; det-query acc + certified output-check):", flush=True)
        print(f"    {'mode':14s} {'K':>3s} {'T':>3s} {'acc':>8s} {'output-check':>13s}", flush=True)
        for name, r in out.items():
            print(f"    {name:14s} {r['search_K']:>3d} {r['loop_T']:>3d} {r['acc']*100:7.1f}% "
                  f"{r['output_check_rate']*100:12.1f}%", flush=True)
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
    from .alpha_struct import StructureRack, solve_query, build_csp_from_struct, output_check, decode_structure
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

    # ============================================ LEVER A — α-as-search (the verifier rejects ⊥)
    print("\n[smoke] --- LEVER A: α-as-search (propose_structures + sound verifier select) ---")
    # solve_query rejects an UNSAT (⊥) structure outright (pin(0,0) ∧ pin(1,1) ∧ eq(0,1) ⇒ 0==1).
    solv, det, _ = solve_query([("pin", 0, 0), ("pin", 1, 1), ("eq", 0, 1)], 2, 2, 1)
    assert (not solv) and (not det), "solve_query must reject the ⊥ (contradictory) structure"
    print(f"[smoke] solve_query rejects ⊥ structure  solvable={solv}  (PASS)")
    # craft per-instance logits: GREEDY decodes the ⊥ above; sampling reaches the SAT determining graph.
    Kc = model.K; n2, d2, q2 = 2, 2, 1
    pin_lb = torch.full((n2, 1 + Kc), -4.0); pair_lb = torch.full((n2, n2, 5), -4.0)
    pin_lb[0, 1] = 5.0                                       # cell0 → pin(0, value0)  (stable)
    pin_lb[1, 2] = 2.3; pin_lb[1, 1] = 2.0                   # cell1 → greedy value1 (⊥); value0 sampled (SAT)
    pair_lb[0, 1, 1] = 5.0; pair_lb[1, 0, 1] = 5.0          # eq(0,1)
    vmb = torch.ones(n2)
    sat_csp = build_csp_from_struct([("pin", 0, 0), ("eq", 0, 1)], n2, d2)
    g = torch.Generator().manual_seed(0)
    sel1 = model.propose_structures(pin_lb, pair_lb, vmb, n2, d2, q2, K=1)
    assert not sel1["certified"], "K=1 greedy must NOT certify (it decodes the ⊥ structure)"
    sel16 = model.propose_structures(pin_lb, pair_lb, vmb, n2, d2, q2, K=16, rng=g,
                                     output_check_fn=output_check, csp_true=sat_csp)
    assert sel16["certified"] and sel16["determines"] and sel16["answer"] == 0, \
        f"search must recover a SOLVABLE+determining candidate that output-checks (got {sel16})"
    s, _, _ = solve_query(sel16["facts"], n2, d2, q2)
    assert s, "the SELECTED structure must be solvable (the verifier kept only ⊥-free candidates)"
    print(f"[smoke] greedy(K=1) certified={sel1['certified']} ⊥; search(K=16) certified={sel16['certified']} "
          f"answer={sel16['answer']} (n_pass={sel16['n_pass']})  — verifier rejects ⊥, selects SAT  (PASS)")

    # ============================================ LEVER B — iterative α↔organ loop
    print("\n[smoke] --- LEVER B: iterative α↔organ loop (feedback channel + T=1 no-op) ---")
    queries = [1, 1]
    # T=1 == the single-shot core: the loop's one emission equals the greedy decode.
    with torch.no_grad(), model.live(mention, attn, inject=False, capture=True, nd=nd, dvec=dvec,
                                     loop_T=1, queries=queries):
        _ = model.logits(ids, attn)
    facts_core = [list(f) for f in model._last_facts]
    # the feedback channel CHANGES α's emission (crafted, like the γ gate-open proof): with the fb_adapter
    # opened, conditioning on a non-zero organ report re-routes the rack's decode.
    with torch.no_grad(), model.live(mention, attn, inject=False, capture=True, nd=nd, dvec=dvec,
                                     loop_T=1, queries=queries):
        _ = model.logits(ids, attn)                          # populate _h_mid for the direct re-compile
        pin0, pair0, _, vm0 = model._compile_struct(None)
        facts_nofb = [decode_structure(pin0[b], pair0[b], vm0[b], nd[b][0], nd[b][1]) for b in range(B)]
        saved_fb = model.fb_adapter.weight.data.clone()
        torch.manual_seed(1); model.fb_adapter.weight.data.copy_(torch.randn_like(saved_fb) * 8.0)
        fb = torch.ones(B, mention.shape[1], 3)              # a non-zero organ report (narrowed/determined)
        pin1, pair1, _, vm1 = model._compile_struct(fb)
        facts_fb = [decode_structure(pin1[b], pair1[b], vm1[b], nd[b][0], nd[b][1]) for b in range(B)]
        model.fb_adapter.weight.data.copy_(saved_fb)         # restore zero-init (no-op preserved)
    assert facts_nofb == facts_core, "T=1 loop emission must equal the single-shot core decode (no-op)"
    changed = any(set(map(tuple, facts_fb[b])) != set(map(tuple, facts_nofb[b])) for b in range(B))
    assert changed, "the feedback channel did not change α's emission (fb_adapter inert)"
    print(f"[smoke] T=1 loop == single-shot core (facts match); feedback FLIPS emission "
          f"inst0: {facts_nofb[0]} -> {facts_fb[0]}  (PASS)")
    # run the multi-step loop end-to-end: T steps, each lattice a sound certified narrowing.
    with torch.no_grad(), model.live(mention, attn, inject=True, capture=True, nd=nd, dvec=dvec,
                                     loop_T=3, queries=queries):
        g3 = model.logits(ids, attn).float()
    info = model._last_loop_info
    assert info["steps"] >= 1 and len(info["facts_per_step"]) == info["steps"], "loop did not run T steps"
    surv = model._captured_surv
    assert ((surv == 0) | (surv == 1)).all(), "composed lattice must be a {0,1} survival set (sound narrowing)"
    print(f"[smoke] loop ran {info['steps']} step(s) (stop={info['stop']}); every step a sound certified "
          f"narrowing; T=1 is the no-op core  (PASS)")

    model.remove_hooks()
    print("\n[smoke] ALL STRUCTURAL CHECKS PASS — AlphaStructWoven builds, forward runs on csp_α, "
          "no-op@init, gate moves, EMPTY_STRUCT control, invariant guard fires; LEVER A (α-as-search) "
          "rejects ⊥ + selects SAT; LEVER B (α↔organ loop) feedback revises emission with T=1 == core.\n")
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
