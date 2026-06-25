"""Train the augmented OLMo (frozen host + zero-init gated adapters + learned differentiable
deductor) on text-rendered graph-coloring-with-abstention, and compare to BASE OLMo (few-shot,
prompted to answer-or-say-cannot-determine).

Decisive reads:
  * accuracy on ANSWERABLE (uniquely-determined) queries,
  * ABSTENTION precision/recall on UNANSWERABLE (underdetermined / unsat) queries,
  * SIZE-GENERALIZATION: train on small graphs, test on larger ones,
  * EXTRACTION fidelity: does the WRITE head actually recover the CSP from English? (the key risk)

  python -m clair.run_augmented --steps 1500 --train_n 4,6 --test_n 5,8,11 --smoke
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from . import augmented as A
from . import induce as I


def build_pool(n_lo, n_hi, k, size, rng, det_frac=0.5):
    n_det = int(size * det_frac)
    det, ab = [], []
    while len(det) < n_det or len(ab) < size - n_det:
        p = I.gen_problem(int(rng.integers(n_lo, n_hi + 1)), k, rng)
        if p["capped"]:
            continue
        if p["determined"] and len(det) < n_det:
            det.append(p)
        elif not p["determined"] and len(ab) < size - n_det:
            ab.append(p)
    pool = det + ab
    rng.shuffle(pool)
    return pool


def batch_from(pool, idxs, tok, Nmax, K, colors, dev):
    return A.build_batch([pool[i] for i in idxs], tok, Nmax, K, colors, dev)


def losses(model, ba, aux):
    color_logits, abstain_logit, (cand, E, alive, elog) = model(ba)
    la = F.binary_cross_entropy_with_logits(abstain_logit, ba["abst"])
    det = ba["abst"] < 0.5
    lc = F.cross_entropy(color_logits[det], ba["ans"][det]) if det.any() else color_logits.sum() * 0
    l = la + lc
    aux_terms = {}
    if aux > 0:
        vm = ba["vmask"]; pair = vm[:, :, None] * vm[:, None, :]
        pw = torch.tensor(6.0, device=elog.device)  # DIFF edges are sparse
        le = F.binary_cross_entropy_with_logits(elog.clamp(-30, 30), ba["adj"], pos_weight=pw, reduction="none")
        le = (le * pair).sum() / pair.sum().clamp_min(1)
        lp = F.binary_cross_entropy_with_logits(cand, ba["pins"], reduction="none")
        lp = (lp * ba["pinned"].unsqueeze(-1)).sum() / ba["pinned"].sum().clamp_min(1)
        l = l + aux * (le + lp)
        aux_terms = {"edge": float(le.detach()), "pin": float(lp.detach())}
    return l, {"la": float(la.detach()), "lc": float(lc.detach() if torch.is_tensor(lc) else 0.0), **aux_terms}


@torch.no_grad()
def extraction_metrics(model, ba, cand, E):
    """How well did WRITE recover the program? pin-color acc on pinned cells, edge F1 vs true DIFF."""
    vm = ba["vmask"]
    pin_pred = cand.argmax(-1)
    pin_true = ba["pins"].argmax(-1)
    pmask = ba["pinned"] > 0.5
    pin_acc = float(((pin_pred == pin_true) & pmask).sum()) / max(1, int(pmask.sum()))
    pair = (vm[:, :, None] * vm[:, None, :]) > 0.5
    epred = (E > 0.5) & pair
    etrue = (ba["adj"] > 0.5) & pair
    tp = float((epred & etrue).sum()); fp = float((epred & ~etrue).sum()); fn = float((~epred & etrue).sum())
    f1 = 2 * tp / max(1.0, 2 * tp + fp + fn)
    # per-problem exact program: all pins right AND all edges right
    exact = 0
    B = vm.size(0)
    for b in range(B):
        ok_p = bool(((pin_pred[b] == pin_true[b]) | ~pmask[b]).all())
        m = pair[b]
        ok_e = bool((epred[b][m] == etrue[b][m]).all())
        exact += int(ok_p and ok_e)
    return {"pin_acc": pin_acc, "edge_f1": f1, "prog_exact": exact / B}


@torch.no_grad()
def evaluate(model, pool, tok, Nmax, K, colors, dev, n=512, bs=64):
    model.eval()
    tp = fp = fn = 0
    det_tot = det_right = 0
    correct = tot = 0
    pin_accs, edge_f1s, prog_exacts = [], [], []
    seen = 0
    i = 0
    while seen < min(n, len(pool)):
        idxs = list(range(i, min(i + bs, len(pool))))
        if not idxs:
            break
        i += bs
        ba = batch_from(pool, idxs, tok, Nmax, K, colors, dev)
        cl, al, (cand, E, alive, elog) = model(ba)
        em = extraction_metrics(model, ba, cand, E)
        pin_accs.append(em["pin_acc"]); edge_f1s.append(em["edge_f1"]); prog_exacts.append(em["prog_exact"])
        pred_abs = torch.sigmoid(al) > 0.5
        true_abs = ba["abst"] > 0.5
        pred_col = cl.argmax(-1)
        tp += int((pred_abs & true_abs).sum()); fp += int((pred_abs & ~true_abs).sum())
        fn += int((~pred_abs & true_abs).sum())
        det = ~true_abs
        det_tot += int(det.sum()); det_right += int(((pred_col == ba["ans"]) & ~pred_abs & det).sum())
        ok = (true_abs & pred_abs) | (~true_abs & ~pred_abs & (pred_col == ba["ans"]))
        correct += int(ok.sum()); tot += ba["abst"].numel(); seen += ba["abst"].numel()
    model.train()
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return {"overall": correct / max(1, tot), "det_acc": det_right / max(1, det_tot),
            "abstain_prec": prec, "abstain_rec": rec, "n": tot,
            "pin_acc": float(np.mean(pin_accs)), "edge_f1": float(np.mean(edge_f1s)),
            "prog_exact": float(np.mean(prog_exacts))}


# ===================================================================== base OLMo few-shot baseline
FEWSHOT = (
    "Node A is red. Node A and node B must be different colors. "
    "Question: what color is node A? Answer: red\n"
    "Node A is blue. Node A and node B must be different colors. "
    "Question: what color is node B? Answer: cannot determine\n"
    "Node A is green. Node B is green. Node A and node B must be different colors. "
    "Question: what color is node A? Answer: cannot determine\n"
)


@torch.no_grad()
def base_eval(olmo, tok, pool, K, colors, dev, n=256):
    """Few-shot base OLMo: score candidate answers {colors..., 'cannot determine'} by LM likelihood."""
    olmo.eval()
    cands = [" " + c for c in colors[:K]] + [" cannot determine"]
    tp = fp = fn = 0
    det_tot = det_right = 0
    correct = tot = 0
    pool = pool[:n]
    for p in pool:
        text, _ = A.render_problem(p, colors)
        prompt = FEWSHOT + A._fix_caps(text)
        scores = []
        for ci, cand in enumerate(cands):
            full = prompt + cand
            ids = tok(full, return_tensors="pt").to(dev)
            pids = tok(prompt, return_tensors="pt")["input_ids"]
            plen = pids.size(1)
            out = olmo(input_ids=ids["input_ids"])
            logits = out.logits[0, :-1]
            tgt = ids["input_ids"][0, 1:]
            lp = torch.log_softmax(logits, -1)
            tok_lp = lp[torch.arange(tgt.size(0)), tgt]
            cand_lp = tok_lp[plen - 1:].sum().item()  # log p of the candidate continuation
            scores.append(cand_lp)
        pred = int(np.argmax(scores))
        pred_abs = pred == K
        pred_col = pred if pred < K else -1
        true_abs = not p["determined"]
        if pred_abs and true_abs: tp += 1
        if pred_abs and not true_abs: fp += 1
        if not pred_abs and true_abs: fn += 1
        if not true_abs:
            det_tot += 1
            if not pred_abs and pred_col == p["answer"]:
                det_right += 1
        ok = (true_abs and pred_abs) or (not true_abs and not pred_abs and pred_col == p["answer"])
        correct += int(ok); tot += 1
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return {"overall": correct / max(1, tot), "det_acc": det_right / max(1, det_tot),
            "abstain_prec": prec, "abstain_rec": rec, "n": tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--aux", type=float, default=1.0)
    ap.add_argument("--adapter_layers", default="5,10,14")
    ap.add_argument("--train_n", default="4,6")
    ap.add_argument("--test_n", default="5,8,11")
    ap.add_argument("--pool_size", type=int, default=6000)
    ap.add_argument("--eval_n", type=int, default=512)
    ap.add_argument("--base_n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no_base", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.steps = min(a.steps, 60); a.pool_size = 1200; a.eval_n = 192; a.base_n = 64
    colors = A.COLORS
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("loading OLMo", A.MODEL_ID, flush=True)
    tok = AutoTokenizer.from_pretrained(A.MODEL_ID)
    tok.padding_side = "right"
    olmo = AutoModelForCausalLM.from_pretrained(A.MODEL_ID, dtype=torch.bfloat16).to(dev).eval()
    Nmax = max(int(x) for x in a.test_n.split(",")) + 1
    model = A.AugmentedOLMo(olmo, tok, Nmax, a.k, T=a.T).to(dev)
    # trainable modules stay fp32 (stable Adam); they cast bf16 host activations internally
    print(f"trainable params {A.n_trainable(model):,}  Nmax={Nmax}", flush=True)

    gap = A.verify_flamingo_noop(olmo, tok, dev)
    print(f"FLAMINGO GATE NO-OP (zero-init gated adapter on OLMo, max|base-gated|, ~0 expected): {gap:.3e}", flush=True)

    n_lo, n_hi = (int(x) for x in a.train_n.split(","))
    prng = np.random.default_rng(a.seed + 999)
    t0 = time.time()
    train_pool = build_pool(n_lo, n_hi, a.k, a.pool_size, prng)
    eval_pools = {"train": build_pool(n_lo, n_hi, a.k, a.eval_n, prng)}
    for tn in (int(x) for x in a.test_n.split(",")):
        eval_pools[tn] = build_pool(tn, tn, a.k, a.eval_n, prng)
    print(f"pools built in {time.time()-t0:.0f}s", flush=True)

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    rng = np.random.default_rng(a.seed)
    log = []
    t0 = time.time()
    model.train()
    for s in range(1, a.steps + 1):
        idxs = rng.integers(0, len(train_pool), a.bs).tolist()
        ba = batch_from(train_pool, idxs, tok, Nmax, a.k, colors, dev)
        loss, parts = losses(model, ba, a.aux)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0); opt.step()
        if s % max(1, a.steps // 10) == 0 or s == 1:
            ind = evaluate(model, eval_pools["train"], tok, Nmax, a.k, colors, dev, n=min(192, a.eval_n))
            print(f"  step {s:5d} loss {loss.item():.3f} ({parts})  in-dist overall {ind['overall']*100:4.1f}%"
                  f"  abstP/R {ind['abstain_prec']*100:.0f}/{ind['abstain_rec']*100:.0f}"
                  f"  pinAcc {ind['pin_acc']*100:.0f} edgeF1 {ind['edge_f1']*100:.0f} progEx {ind['prog_exact']*100:.0f}"
                  f"  {time.time()-t0:.0f}s", flush=True)
            log.append({"step": s, "loss": float(loss.detach()), **ind})

    # ---- size-generalization eval (augmented, end-to-end) ----
    print("\n=== AUGMENTED size-generalization ===", flush=True)
    aug_gen = {}
    for tn in (int(x) for x in a.test_n.split(",")):
        aug_gen[tn] = evaluate(model, eval_pools[tn], tok, Nmax, a.k, colors, dev, n=a.eval_n)
        r = aug_gen[tn]
        print(f"  N={tn:2d}  overall {r['overall']*100:4.1f}  detAcc {r['det_acc']*100:4.1f}"
              f"  abstP/R {r['abstain_prec']*100:.0f}/{r['abstain_rec']*100:.0f}"
              f"  pinAcc {r['pin_acc']*100:.0f} edgeF1 {r['edge_f1']*100:.0f} progEx {r['prog_exact']*100:.0f}", flush=True)

    base_gen = {}
    if not a.no_base:
        print("\n=== BASE OLMo (few-shot) size-generalization ===", flush=True)
        for tn in (int(x) for x in a.test_n.split(",")):
            base_gen[tn] = base_eval(olmo, tok, eval_pools[tn], a.k, colors, dev, n=a.base_n)
            r = base_gen[tn]
            print(f"  N={tn:2d}  overall {r['overall']*100:4.1f}  detAcc {r['det_acc']*100:4.1f}"
                  f"  abstP/R {r['abstain_prec']*100:.0f}/{r['abstain_rec']*100:.0f}  (n={r['n']})", flush=True)

    res = {"args": vars(a), "gate_noop_gap": gap, "log": log,
           "aug_gen": {str(k): v for k, v in aug_gen.items()},
           "base_gen": {str(k): v for k, v in base_gen.items()}}
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "runs", "augmented.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("\nwrote", out, flush=True)


if __name__ == "__main__":
    main()
