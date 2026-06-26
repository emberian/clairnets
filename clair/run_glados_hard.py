"""clair/run_glados_hard.py — THE DECISIVE woven-GLaDOS test (the two-change re-run).

Removes BOTH failure causes of the earlier woven run:
  (1) HARD, propagation-required tasks (clair.hard_tasks): the query answer is determined ONLY through
      an L-step equality / forced-colouring chain the LM cannot do in-context, so base OLMo / a
      text-only LoRA cannot shortcut it — the organ's narrowing is the only route on the DETERMINED
      stratum.  Chain length L is the OOD axis.
  (2) BOOTSTRAP the organ to near-oracle BEFORE co-training: train the FactorGraphProposer alone
      (dominate-dedP, run_general-style) to high narrowing-recall + ~0 false-elim, load those weights
      into the woven model, THEN co-train the LM/LoRA/gamma against a RELIABLE organ from step 0 (organ
      kept sharp via its teacher-forced dominate-dedP loss, optionally at a lower LR).

Decisive measurements (comparable to the failed run + the oracle de-risk):
  * closed-set generative accuracy on DETERMINED hard problems: woven-with-organ vs base OLMo vs
    text-only-LoRA, in-dist + OOD-by-length.  The organ-model should WIN where the others can't
    propagate.
  * THE CAUSAL CONTROLS: shuffle / candidate-permute / corrupt-query-cell / zero the organ output.
    If determined accuracy DROPS sharply (esp. corrupt -> low) and the gate tanh(alpha) has OPENED,
    the LM is causally using the LEARNED organ.  If flat, report honestly + diagnose (organ recall at
    woven inference depth, alpha program-recon).

  python -m clair.run_glados_hard --mode smoke
  python -m clair.run_glados_hard --mode full --out runs/glados_hard.json
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from . import glados_woven as G
from . import hard_tasks as H
from . import curriculum as CU
from . import csp as C
from .proposer import FactorGraphProposer, size_for
from .run_general import loss_fn as organ_loss_fn


def device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ===================================================================== organ BOOTSTRAP (pure organ)
def build_organ(dev, target=1.5e5, R=8):
    d, npar = size_for("full", G.NMAX, G.DMAX, G.MMAX, G.AMAX, target, R=R, ds=min(4, R))
    organ = FactorGraphProposer("full", G.NMAX, G.DMAX, G.MMAX, G.AMAX, d=d, R=R, ds=min(4, R)).to(dev)
    return organ, d, npar


def _dom_from_mask(nv, csp):
    return tuple(frozenset(v for v in range(csp.d) if nv[i, v] > 0.5) for i in range(csp.n))


@torch.no_grad()
def organ_fixpoint(organ, csps, dev, theta=0.5, R_max=24, max_calls=None):
    """Iterate the organ's monotone meet to fixpoint (or to max_calls outer calls = the woven inference
    depth). Returns per-CSP final domains."""
    items = [(c, c.full()) for c in csps]
    calls = 0
    for _ in range(R_max):
        feat = G.featurize_true(items, dev)
        b, _, _ = G.organ_call(organ, feat)
        new_vm = feat["var_mask"] * (torch.sigmoid(b) >= theta).float()
        nv = new_vm.cpu().numpy()
        nxt, changed = [], False
        for bi, (csp, dom) in enumerate(items):
            ndom = _dom_from_mask(nv[bi], csp)
            if ndom != dom:
                changed = True
            nxt.append((csp, ndom))
        items = nxt
        calls += 1
        if (max_calls and calls >= max_calls) or not changed:
            break
    return [dom for _, dom in items]


@torch.no_grad()
def organ_report(organ, probs, dev, tag, max_calls=None):
    """Narrowing-recall vs exact dedP, false-elim, and DETERMINED-query solve-rate (does the organ
    pin the query cell to the exact answer?). max_calls=4 mirrors woven inference (compile + 3 wb)."""
    csps = [p.csp for p in probs]
    doms = organ_fixpoint(organ, csps, dev, max_calls=max_calls)
    rec_n = rec_d = fe = q_solve = q_tot = 0
    for p, dom in zip(probs, doms):
        csp = p.csp
        full = csp.full()
        ded = H.fast_dedP(csp)
        rm_model = {(i, v) for i in range(csp.n) for v in full[i] if v not in dom[i]}
        rm_ded = {(i, v) for i in range(csp.n) for v in full[i] if v not in ded[i]}
        rec_n += len(rm_model & rm_ded); rec_d += len(rm_ded)
        fe += len(rm_model - rm_ded)                        # eliminations dedP would NOT make (unsound)
        if p.determined:
            q_tot += 1
            q_solve += int(dom[p.query] == frozenset({p.answer}))
    return {"recall": rec_n / max(1, rec_d), "false_elim_pairs": int(fe),
            "query_solve": q_solve / max(1, q_tot), "n": len(probs), "ndet": q_tot, "tag": tag}


def bootstrap(dev, args, lengths):
    print("\n===== BOOTSTRAP organ (pure dominate-dedP on hard tasks) =====", flush=True)
    organ, d, npar = build_organ(dev, target=args.organ_target, R=args.organ_R)
    opt = torch.optim.AdamW(organ.parameters(), lr=args.boot_lr, betas=(0.9, 0.95))
    rng = np.random.default_rng(args.seed)
    pool = args.boot_pool
    fams = tuple(getattr(args, "families", ["eqchain", "forcedcolor"]))
    csps = H.bootstrap_corpus(rng, pool, lengths, families=fams)
    items = [(c, c.full()) for c in csps]
    print(f"  organ d={d} params={npar:,} pool={pool} lengths={lengths} steps={args.boot_steps}", flush=True)
    t0 = time.time(); fe_k = fe_n = 0
    ex = H.FastExact()
    for s in range(1, args.boot_steps + 1):
        feat = G.featurize_true(items, dev)
        vm = feat["var_mask"]
        tgt, conflict = G.build_targets(items, ex, dev)
        b, cls, sup = G.organ_call(organ, feat)
        loss = organ_loss_fn(sup, vm, feat["var_valid"], tgt, conflict)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(organ.parameters(), 1.0); opt.step()
        with torch.no_grad():
            new_vm = vm * (torch.sigmoid(b) >= 0.5).float()
            elig = tgt * vm * feat["var_valid"].unsqueeze(-1)
            fe_k += float((elig * (new_vm < 0.5).float()).sum()); fe_n += float(elig.sum())
            nv = new_vm.cpu().numpy()
            nxt = []
            for bi, (csp, dom) in enumerate(items):
                ndom = _dom_from_mask(nv[bi], csp)
                if C.status(ndom) in ("solved", "conflict") or ndom == dom:
                    nc = H.bootstrap_corpus(rng, 1, lengths, families=fams)[0]
                    nxt.append((nc, nc.full()))
                else:
                    nxt.append((csp, ndom))
            items = nxt
        if s % max(1, args.boot_steps // 12) == 0 or s == 1:
            print(f"    step {s:5d}  loss {float(loss):.3f}  false_elim {fe_k/max(1,fe_n):.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            fe_k = fe_n = 0
    # report at fixpoint AND at woven inference depth (4 calls)
    rep = {}
    for fam in fams:
        for L in lengths[:1] + [lengths[-1]] + [args.ood_lengths[-1]]:
            ev = H.pool(np.random.default_rng(7000 + L), fam, L, 64, determined=True)
            rfix = organ_report(organ, ev, dev, f"{fam}L{L}_fix")
            r4 = organ_report(organ, ev, dev, f"{fam}L{L}_4call", max_calls=4)
            rep[f"{fam}_L{L}"] = {"fixpoint": rfix, "woven4": r4}
            print(f"  {fam:11s} L={L:2d}  recall(fix) {rfix['recall']*100:5.1f}%  "
                  f"query-solve(fix) {rfix['query_solve']*100:5.1f}%  query-solve(4call) "
                  f"{r4['query_solve']*100:5.1f}%  FE {rfix['false_elim_pairs']}", flush=True)
    return organ, {"d": d, "params": npar, "report": rep}


# ===================================================================== closed-set scoring
def build_score_batch(probs, cands, tok, dev):
    rows, plen, spans_rows, prob_of, qidx, ns = [], [], [], [], [], []
    for pi, p in enumerate(probs):
        prompt = CU.canonical_render(p) + " Answer:"
        spans = CU.entity_mentions(prompt, p.n)
        for cand in cands:
            rows.append(prompt + cand); plen.append(len(prompt)); spans_rows.append(spans)
            prob_of.append(pi); qidx.append(p.query); ns.append(p.n)
    enc = tok(rows, return_offsets_mapping=True, padding=True, return_tensors="pt")
    ids = enc["input_ids"].to(dev); attn = enc["attention_mask"].to(dev)
    offs = enc["offset_mapping"]; T = ids.size(1); Rn = len(rows)
    mention = torch.zeros(Rn, G.NMAX, T, device=dev)
    ansmask = torch.zeros(Rn, T, device=dev)
    for r in range(Rn):
        o = offs[r].tolist()
        for cell, sps in spans_rows[r].items():
            if cell >= G.NMAX:
                continue
            for (cs, ce) in sps:
                for ti, (a, c) in enumerate(o):
                    if a != c and a < ce and c > cs:
                        mention[r, cell, ti] = 1.0
        for ti, (a, c) in enumerate(o):
            if a != c and a >= plen[r] and attn[r, ti] > 0.5:
                ansmask[r, ti] = 1.0
    vmask = torch.zeros(Rn, G.NMAX, device=dev)
    for r, n in enumerate(ns):
        vmask[r, :n] = 1.0
    return {"input_ids": ids, "attn": attn, "mention": mention, "vmask": vmask,
            "ansmask": ansmask, "qidx": torch.tensor(qidx, device=dev), "Cn": len(cands)}


@torch.no_grad()
def score_closedset(gw, probs, tok, K, dev, control="none", base=False, text_only=False, bs=12):
    """LM head GENERATES the answer, scored by length-normalised answer-span logprob over the legal
    set {colours..., 'cannot be determined'} (constrained decoding). Returns overall/det/abstain acc."""
    gw.eval()
    cands = [" " + c for c in CU.COLORS[:K]] + [" cannot be determined"]
    Cn = len(cands)
    correct = tot = det_t = det_r = ab_t = ab_r = 0
    for i in range(0, len(probs), bs):
        chunk = probs[i:i + bs]
        ba = build_score_batch(chunk, cands, tok, dev)
        Rn = ba["input_ids"].size(0)
        cur = {"input_ids": ba["input_ids"], "attn": ba["attn"],
               "mention": ba["mention"], "vmask": ba["vmask"]}
        gw._qidx = ba["qidx"]
        if base:
            old = gw.organ_writeback; gw.organ_writeback = False
            with gw.model.disable_adapter():
                logits = gw._run_host(cur).float()
            gw.organ_writeback = old
        elif text_only:
            old = gw.organ_writeback; gw.organ_writeback = False
            logits = gw._run_host(cur).float()
            gw.organ_writeback = old
        else:
            gw.control = control
            gw._perm = torch.randperm(Rn, device=dev) if control == "shuffle" else None
            gw._kperm = torch.randperm(K, device=dev) if control == "permute" else None
            logits = gw._run_host(cur).float()
            gw.control = "none"
        lp = torch.log_softmax(logits[:, :-1], -1)
        tgt = ba["input_ids"][:, 1:]
        toklp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        m = ba["ansmask"][:, 1:]
        scores = ((toklp * m).sum(-1) / m.sum(-1).clamp_min(1.0)).view(len(chunk), Cn)
        pred = scores.argmax(-1)
        for b, p in enumerate(chunk):
            gi = p.answer if p.determined else K
            ok = int(pred[b].item()) == gi
            correct += ok; tot += 1
            if p.determined:
                det_t += 1; det_r += ok
            else:
                ab_t += 1; ab_r += ok
    return {"acc": correct / max(1, tot), "det_acc": det_r / max(1, det_t),
            "abst_acc": ab_r / max(1, ab_t), "n": tot, "ndet": det_t}


# ===================================================================== training
def _opt_groups(gw, lr, organ_lr_scale, gamma_lr):
    """LoRA+alpha at lr; organ at a LOWER lr (stay near the bootstrap); gamma/gate at a HIGHER lr
    (the oracle-readout opened its gate at gamma_lr=1e-3 > lora_lr=2e-4 — give the gate the same head
    start so a CLOSED gate can't be blamed on under-training)."""
    organ_ids = {id(p) for p in gw.organ.parameters()}
    gamma_ids = {id(p) for p in gw.gamma.parameters()}
    organ = [p for p in gw.organ.parameters() if p.requires_grad]
    gamma = [p for p in gw.gamma.parameters() if p.requires_grad]
    rest = [p for p in gw.trainable_parameters() if id(p) not in organ_ids and id(p) not in gamma_ids]
    return [{"params": rest, "lr": lr}, {"params": organ, "lr": lr * organ_lr_scale},
            {"params": gamma, "lr": gamma_lr}]


def train_woven(gw, tok, K, dev, steps, lengths, bs=8, lr=2e-4, gamma_lr=1e-3, w_organ=1.0, w_prog=0.5,
                organ_lr_scale=0.25, det_frac=0.5, seed=0, freeze_organ_after=10**9, text_only=False,
                families=("eqchain", "forcedcolor")):
    rng = np.random.default_rng(seed)
    if text_only:
        gw.organ_writeback = False
        params = [p for n, p in gw.model.named_parameters() if p.requires_grad]   # LoRA only
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW(_opt_groups(gw, lr, organ_lr_scale, gamma_lr), betas=(0.9, 0.95))
    gw.train(); t0 = time.time(); log = []
    for s in range(1, steps + 1):
        if not text_only and s == freeze_organ_after:
            for p in gw.organ.parameters():
                p.requires_grad_(False)
        probs = H.mixed_pool(rng, bs, lengths, families=families, det_frac=det_frac)
        ba = G.build_batch(probs, tok, K, dev)
        out = gw(ba)
        loss = out["lm_ce"] + (0 if text_only else w_organ * out["organ"] + w_prog * out["prog"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gw.trainable_parameters(), 1.0); opt.step()
        if s % max(1, steps // 15) == 0 or s == 1:
            gate = float(torch.tanh(gw.gamma.alpha).abs().max()) if not text_only else 0.0
            rec = {"step": s, "lm_ce": float(out["lm_ce"]), "organ": float(out["organ"]),
                   "prog": float(out["prog"]), "organ_fe": out["organ_fe"], "gate": gate}
            log.append(rec)
            print(f"  step {s:4d}  lm_ce {rec['lm_ce']:.3f}  organ {rec['organ']:.3f}  prog "
                  f"{rec['prog']:.3f}  organ_FE {rec['organ_fe']:.4f}  gate {gate:.3f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    return log


# ===================================================================== model loading
def load(dev, host="allenai/OLMo-2-0425-1B", K=3, lora_r=16, **kw):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(host)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(host, dtype=torch.float32)
    cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n)
    gw = G.GladosWoven(model, tok, K=K, **kw).to(dev)
    return gw, tok


def _base_logits(gw, ids, attn):
    with gw.model.disable_adapter():
        return gw.model(input_ids=ids, attention_mask=attn).logits


# ===================================================================== eval-set construction
def make_eval_sets(families, lengths, neval, seed=99):
    sets = {}
    for fam in families:
        for L in lengths:
            sets[(fam, L)] = H.pool(np.random.default_rng(seed + 17 * L + hash(fam) % 1000),
                                    fam, L, neval, determined=True)
    return sets


# ===================================================================== modes
def run_smoke(args, dev):
    print("\n===== SMOKE: hard-task woven GLaDOS =====", flush=True)
    lengths = [3, 4, 5]; ood = [8]
    args.ood_lengths = ood
    # 1) base OLMo must FAIL determined long chains (proof the task is not shortcuttable)
    gw, tok = load(dev, K=3, compile_layer=args.compile_layer, writeback_layers=tuple(args.writeback))
    print(f"  organ params={gw.organ_params:,}  trainable={sum(p.numel() for p in gw.trainable_parameters()):,}",
          flush=True)
    ba = G.build_batch(H.pool(np.random.default_rng(0), "eqchain", 4, 4, True), tok, 3, dev)
    diff = G.verify_noop(gw, lambda ids, attn: _base_logits(gw, ids, attn), ba)
    print(f"  NO-OP @ init max|woven-base| = {diff:.3e} ({'PASS' if diff < 1e-3 else 'FAIL'})", flush=True)
    assert diff < 1e-3
    for fam in ("eqchain", "forcedcolor"):
        for L in (3, 8):
            ev = H.pool(np.random.default_rng(1 + L), fam, L, 32, True)
            b = score_closedset(gw, ev, tok, 3, dev, base=True)
            print(f"  BASE OLMo  {fam:11s} L={L}  det-acc {b['det_acc']*100:5.1f}% (n={b['ndet']})", flush=True)
    # 2) organ bootstraps
    args.boot_steps = args.boot_steps or 300; args.boot_pool = 64
    organ, brep = bootstrap(dev, args, lengths)
    gw.organ.load_state_dict(organ.state_dict())
    # 3) woven trains + gate can move; quick check
    print("  co-training a few steps...", flush=True)
    train_woven(gw, tok, 3, dev, steps=args.steps or 60, lengths=lengths, bs=args.bs)
    ev = H.pool(np.random.default_rng(5), "eqchain", 4, 32, True)
    w = score_closedset(gw, ev, tok, 3, dev); wc = score_closedset(gw, ev, tok, 3, dev, control="corrupt")
    print(f"  woven eqchain L=4 det-acc {w['det_acc']*100:.1f}%  corrupt {wc['det_acc']*100:.1f}%  "
          f"gate {float(torch.tanh(gw.gamma.alpha).abs().max()):.3f}", flush=True)
    return {"smoke": {"noop": diff, "bootstrap": brep}}


def run_full(args, dev):
    print("\n===== FULL: hard-task woven GLaDOS (the decisive run) =====", flush=True)
    families = ["eqchain", "forcedcolor"]
    train_lengths = args.train_lengths
    eval_lengths = args.eval_lengths
    args.ood_lengths = args.eval_lengths
    eval_sets = make_eval_sets(families, eval_lengths, args.neval)
    out = {"train_lengths": train_lengths, "eval_lengths": eval_lengths, "families": families}

    # ---------- text-only-LoRA baseline (no organ) ----------
    print("\n----- text-only-LoRA baseline (organ disabled) -----", flush=True)
    gw_t, tok = load(dev, K=3, compile_layer=args.compile_layer, writeback_layers=tuple(args.writeback))
    train_woven(gw_t, tok, 3, dev, steps=args.steps, lengths=train_lengths, bs=args.bs, lr=args.lr,
                text_only=True, seed=args.seed)
    out["text_lora"] = {}; out["base"] = {}
    for (fam, L), ev in eval_sets.items():
        t = score_closedset(gw_t, ev, tok, 3, dev, text_only=True)
        b = score_closedset(gw_t, ev, tok, 3, dev, base=True)
        out["text_lora"][f"{fam}_L{L}"] = t; out["base"][f"{fam}_L{L}"] = b
        print(f"  {fam:11s} L={L:2d}  base det {b['det_acc']*100:5.1f}%  text-LoRA det {t['det_acc']*100:5.1f}%",
              flush=True)
    del gw_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------- bootstrap + woven ----------
    gw, _ = load(dev, K=3, compile_layer=args.compile_layer, writeback_layers=tuple(args.writeback))
    organ, brep = bootstrap(dev, args, train_lengths)
    gw.organ.load_state_dict(organ.state_dict())
    del organ
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    out["bootstrap"] = brep
    bcheck = G.build_batch(H.pool(np.random.default_rng(0), "eqchain", 4, 4, True), tok, 3, dev)
    out["noop"] = G.verify_noop(gw, lambda ids, attn: _base_logits(gw, ids, attn), bcheck)
    print(f"\n  NO-OP @ init (after organ load) max|diff| = {out['noop']:.3e}", flush=True)

    print("\n----- co-training woven (bootstrapped organ, kept sharp) -----", flush=True)
    out["log"] = train_woven(gw, tok, 3, dev, steps=args.steps, lengths=train_lengths, bs=args.bs,
                             lr=args.lr, gamma_lr=args.gamma_lr, w_organ=args.w_organ, w_prog=args.w_prog,
                             organ_lr_scale=args.organ_lr_scale, seed=args.seed,
                             freeze_organ_after=args.freeze_organ_after)
    out["gate"] = float(torch.tanh(gw.gamma.alpha).abs().max())

    # organ recall AT WOVEN INFERENCE DEPTH after co-training (diagnosis)
    out["organ_after"] = {}
    for fam in families:
        for L in eval_lengths:
            r = organ_report(gw.organ, eval_sets[(fam, L)], dev, f"{fam}L{L}", max_calls=len(gw.writeback_layers) + 1)
            out["organ_after"][f"{fam}_L{L}"] = r

    # ---------- the headline accuracy table ----------
    print("\n===== DETERMINED-QUERY ACCURACY: woven vs base vs text-LoRA =====", flush=True)
    print(f"  {'task':18s} {'base':>7s} {'textLoRA':>9s} {'woven':>7s}   gate={out['gate']:.3f}", flush=True)
    out["woven"] = {}
    for fam in families:
        for L in eval_lengths:
            ev = eval_sets[(fam, L)]
            w = score_closedset(gw, ev, tok, 3, dev)
            out["woven"][f"{fam}_L{L}"] = w
            b = out["base"][f"{fam}_L{L}"]; t = out["text_lora"][f"{fam}_L{L}"]
            print(f"  {fam+' L'+str(L):18s} {b['det_acc']*100:6.1f}% {t['det_acc']*100:8.1f}% "
                  f"{w['det_acc']*100:6.1f}%", flush=True)

    # ---------- THE CAUSAL CONTROL TABLE ----------
    print("\n===== CAUSAL CONTROLS on DETERMINED problems (corrupt organ -> does gen break?) =====",
          flush=True)
    out["controls"] = {}
    for fam in families:
        for L in eval_lengths:
            ev = eval_sets[(fam, L)]
            row = {}
            for ctl in ["none", "shuffle", "permute", "corrupt", "zero"]:
                row[ctl] = score_closedset(gw, ev, tok, 3, dev, control=ctl)["det_acc"]
            out["controls"][f"{fam}_L{L}"] = row
            drop = row["none"] - row["corrupt"]
            print(f"  {fam+' L'+str(L):18s} intact {row['none']*100:5.1f}%  shuffle {row['shuffle']*100:5.1f}%"
                  f"  permute {row['permute']*100:5.1f}%  corrupt {row['corrupt']*100:5.1f}%  zero "
                  f"{row['zero']*100:5.1f}%   | load-bearing(drop>{0.05}): {drop > 0.05}", flush=True)
    return out


def run_ablate(args, dev):
    """GATE-FORCED ablation (the disambiguator). gate_init>0 forces the bootstrapped organ into the
    residual stream from step 0, and the gate (alpha) is FROZEN open so the LM cannot escape the organ
    by closing it — it MUST learn to READ a perfect organ. If determined-eqchain accuracy now jumps
    above the ~47% pin-guess floor and the causal controls (esp. corrupt) bite, the woven READOUT is
    viable and only the gate-opening optimisation failed; if still flat, the readout itself is the wall."""
    print(f"\n===== GATE-FORCED ABLATION (gate_init={args.gate_init}, gate FROZEN open) =====", flush=True)
    families = list(args.families)
    train_lengths = args.train_lengths
    eval_lengths = args.eval_lengths
    args.ood_lengths = args.eval_lengths
    eval_sets = make_eval_sets(families, eval_lengths, args.neval)
    out = {"gate_init": args.gate_init, "train_lengths": train_lengths, "eval_lengths": eval_lengths,
           "families": families, "writeback": args.writeback}
    gw, tok = load(dev, K=3, compile_layer=args.compile_layer,
                   writeback_layers=tuple(args.writeback), gate_init=args.gate_init)
    organ, brep = bootstrap(dev, args, train_lengths)
    gw.organ.load_state_dict(organ.state_dict())
    del organ
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    out["bootstrap"] = brep
    gw.gamma.alpha.requires_grad_(False)               # FREEZE the gate open
    print(f"  gate held at tanh(alpha)={float(torch.tanh(gw.gamma.alpha)):.3f} (frozen)", flush=True)
    out["log"] = train_woven(gw, tok, 3, dev, steps=args.steps, lengths=train_lengths, bs=args.bs,
                             lr=args.lr, gamma_lr=args.gamma_lr, w_organ=args.w_organ, w_prog=args.w_prog,
                             organ_lr_scale=args.organ_lr_scale, seed=args.seed,
                             families=tuple(families))
    out["gate"] = float(torch.tanh(gw.gamma.alpha))
    print("\n===== DETERMINED-QUERY ACCURACY (gate forced open) + CAUSAL CONTROLS =====", flush=True)
    out["woven"] = {}; out["controls"] = {}
    for L in eval_lengths:
        ev = eval_sets[("eqchain", L)]
        row = {}
        for ctl in ["none", "shuffle", "permute", "corrupt", "zero"]:
            row[ctl] = score_closedset(gw, ev, tok, 3, dev, control=ctl)["det_acc"]
        out["controls"][f"eqchain_L{L}"] = row
        drop = row["none"] - row["corrupt"]
        print(f"  eqchain L={L:2d}  intact {row['none']*100:5.1f}%  shuffle {row['shuffle']*100:5.1f}%  "
              f"permute {row['permute']*100:5.1f}%  corrupt {row['corrupt']*100:5.1f}%  zero "
              f"{row['zero']*100:5.1f}%   | READOUT used (drop>.05): {drop > 0.05}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full", "ablate"], default="smoke")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gamma_lr", type=float, default=1e-3)
    ap.add_argument("--w_organ", type=float, default=1.0)
    ap.add_argument("--w_prog", type=float, default=0.5)
    ap.add_argument("--organ_lr_scale", type=float, default=0.25)
    ap.add_argument("--freeze_organ_after", type=int, default=10**9)
    ap.add_argument("--neval", type=int, default=64)
    ap.add_argument("--compile_layer", type=int, default=5)
    ap.add_argument("--writeback", type=int, nargs="+", default=[7, 9, 11])
    ap.add_argument("--train_lengths", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--eval_lengths", type=int, nargs="+", default=[5, 8, 10])
    ap.add_argument("--gate_init", type=float, default=2.0, help="ablate mode: forced-open gate value")
    ap.add_argument("--families", nargs="+", default=["eqchain", "forcedcolor"])
    ap.add_argument("--organ_target", type=float, default=1.5e5)
    ap.add_argument("--organ_R", type=int, default=8)
    ap.add_argument("--boot_steps", type=int, default=1500)
    ap.add_argument("--boot_pool", type=int, default=96)
    ap.add_argument("--boot_lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.steps:
        args.steps = 60 if args.mode == "smoke" else 1500
    dev = device()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    G.SHARED = H.FastExact()          # co-training organ loss uses the fast EXACT dedP (n=12 chains)
    print(f"device={dev}  budget N={G.NMAX} D={G.DMAX} M={G.MMAX}", flush=True)
    out = {"smoke": run_smoke, "full": run_full, "ablate": run_ablate}[args.mode](args, dev)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1, default=float)
        print("\nwrote", args.out, flush=True)


if __name__ == "__main__":
    main()
