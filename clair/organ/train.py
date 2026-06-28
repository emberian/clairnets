"""clair/organ/train.py — THE canonical staged training pipeline for the GLaDOS organ.

Three clean callables, ONE documented flow:  pretrain_organ → weave → rlvr.  Each lifts the
already-validated recipe into a single front-door callable (nothing is re-prototyped — the bodies
delegate to the proven implementations in clair.run_glados_staged / clair.rlvr_pipeline):

  (1) pretrain_organ(...)   STAGE 1 — train the deduction organ STANDALONE on the soundness-asymmetric
                            dominate-dedₚ loss vs the exact per-cell transformer (clair.csp.exact_dedP),
                            on-policy over the DIFFICULTY-CONTROLLED stream (datagen.stream.organ_spec,
                            difficulty=True). A PREFLIGHT OCCUPANCY GATE refuses to run if the data is
                            still the legacy easy distribution (required-level L>=1 < threshold, etc.)
                            and stamps the occupancy into the checkpoint meta. Recipe (from the sweep):
                            ~1.5M params, R=12. Saves the frozen union organ in load_core_organ's format.
                            (--easy/--legacy selects the OLD RUNGS sampler, which the gate then blocks.)

  (2) weave(...)            STAGE 2 — graft the organ into a host LM (clair.organ.graft) and train the
                            woven readout: frozen base + LoRA + latent-α compile + FROZEN organ +
                            zero-init γ readout, on the answer-span LM CE. This is the engagement
                            mechanism, lifted out of run_glados_staged's experiment driver into one
                            callable: the TWO-STREAM lever (α reads full text, the LM generates from a
                            fact-ablated cells prompt), J0 direct α-supervision (dominate-dedₚ on α's
                            raw compile, the SATNet grounding fix), and the rich-state calibration /
                            causal-control readout (true/shuffle/permute/corrupt/zero + oracle-override).

  (3) rlvr(...)             STAGE 3 — RL-from-verifiable-rewards (TRL Dr.GRPO) with the exact checker as
                            reward; the organ-as-process-reward (LSRL-style per-step cardinality drop) is
                            the documented insertion point. Sharpens calibration, not capacity; report
                            pass@1 AND pass@k.

Eval is the arbiter, exposed separately as clair.organ.eval.run_eval_suite.

Usage:
  python -m clair.organ.train pretrain --steps 1500 --out runs/general_organ_full.pt   # difficulty stream
  python -m clair.organ.train pretrain --smoke                          # tiny gated pretrain smoke
  python -m clair.organ.train pretrain --easy                           # legacy RUNGS -> gate BLOCKS
  python -m clair.organ.train weave    --base allenai/OLMo-2-0425-1B --regime hard --steps 2500
  python -m clair.organ.train weave    --smoke                         # tiny end-to-end woven smoke
  python -m clair.organ.train rlvr     --task chain_sum --steps 300
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace


# ============================================================ STAGE 1: pretrain the organ
# ---- PREFLIGHT OCCUPANCY GATE (codex's "assert buckets in run metadata") ----------------------------
# The pretrain must run on the DIFFICULTY-CONTROLLED mix (datagen.stream.organ_spec(difficulty=True)),
# NOT the legacy easy RUNGS distribution the audit measured at ~99% level-0 / treewidth-median-2 / no
# long-depth tail. These thresholds are the design targets (notes/curriculum_design.md): the headline
# is required-level L>=1 >= ~30% (the legacy mix sits at ~9%); the rest are "the bucket is POPULATED"
# floors. Every key gates a minimum percentage; the keys in _GATE_COMPONENT only fire when the spec
# actually carries that mix component. Override any key by passing gate={...} to pretrain_organ.
DEFAULT_OCCUPANCY_GATE = {
    "level_ge1_pct_min": 30.0,        # required-lattice-level L>=1: the affine-wall axis (legacy ~9%)
    "treewidth_ge3_pct_min": 1.0,     # treewidth spread present (tw>=3 fraction > 0)
    "depth_long_pct_min": 1.0,        # long-depth propagation tail present (depth>=7 fraction > 0)
    "randomrel_pct_min": 1.0,         # random-relations present  (only if 'randomrel' in the mix)
    "reduction_pct_min": 1.0,         # reduction-curriculum share (only if 'reduction' in the mix)
    "composition_pct_min": 1.0,       # composition share         (only if 'compose'   in the mix)
}
# which gate keys are conditional on a mix component being configured (vs. always-checked)
_GATE_COMPONENT = {"randomrel_pct_min": "randomrel", "reduction_pct_min": "reduction",
                   "composition_pct_min": "compose"}


def _difficulty_items(spec, seed):
    """Infinite (record, (csp, full)) stream of BUDGET-ADMITTED CSP items from the difficulty spec —
    exactly the distribution the organ pretrain loop tensorizes (out-of-budget / non-CSP records are
    dropped by the organ-mode item filter, so the occupancy reflects what is actually trained on)."""
    from ..datagen import stream as S
    for rec in S.problem_stream(seed, spec):
        item = S._item_from_record(rec)            # (csp, full) or None
        if item is not None:
            yield rec, item


def _difficulty_samples(spec, n, seed=0):
    """The first `n` (record, csp) pairs of the budget-admitted difficulty stream (for the preflight)."""
    cnt = 0
    for rec, item in _difficulty_items(spec, seed):
        yield rec, item[0]
        cnt += 1
        if cnt >= n:
            return


def _legacy_samples(n, seed=0):
    """Yield (pseudo-record, csp) from the OLD RUNGS sampler (run_glados_staged.sample_organ_corpus) —
    the legacy easy distribution codex flagged. The pseudo-record carries only the family tag the
    occupancy profiler reads (RUNGS has no random-relation / reduction / composition components)."""
    import numpy as np
    from .. import run_glados_staged as G
    rng = np.random.default_rng(seed)
    for csp, rung in G.sample_organ_corpus(rng, n, G.RUNGS):
        yield {"family": rung, "compose": "", "n_domains": 1}, csp


def occupancy_profile(samples, label="") -> dict:
    """Single-pass difficulty-occupancy over (record, csp) pairs: per-level / treewidth / depth-third
    histograms (clair.datagen.difficulty) PLUS the family-share signals the gate needs (random-relation,
    reduction, composition). JSON-safe; this dict is BOTH printed and stamped into the checkpoint meta."""
    from collections import Counter
    from ..datagen import difficulty as DF
    lev, tw, dep, fam = Counter(), Counter(), Counter(), Counter()
    n = nrr = nred = ncomp = 0
    for rec, csp in samples:
        d = DF.difficulty(csp)
        n += 1
        lev[min(d["level"], 3)] += 1
        tw[d["treewidth"]] += 1
        dep[DF.depth_bucket(d["depth"])] += 1
        f = rec.get("family", "?")
        fam[f] += 1
        comp = rec.get("compose", "") or ""
        if comp.startswith("reduce:"):
            nred += 1
        if (rec.get("n_domains", 1) or 1) > 1:
            ncomp += 1
        if f == "random_relation":
            nrr += 1
    pct = lambda c: round(100 * c / max(1, n), 1)
    return {
        "label": label, "n": n,
        "level_pct": {str(k): pct(lev.get(k, 0)) for k in DF.LEVEL_BUCKETS},
        "level_ge1_pct": pct(n - lev.get(0, 0)),
        "treewidth_hist": {str(k): tw[k] for k in sorted(tw)},
        "treewidth_median": (sorted(tw.elements())[n // 2] if n else 0),
        "treewidth_ge3_pct": pct(sum(v for k, v in tw.items() if k >= 3)),
        "treewidth_ge4_pct": pct(sum(v for k, v in tw.items() if k >= 4)),
        "depth_thirds_pct": {k: pct(dep.get(k, 0)) for k in DF.DEPTH_THIRDS},
        "randomrel_pct": pct(nrr), "reduction_pct": pct(nred), "composition_pct": pct(ncomp),
        "family_hist": dict(fam),
    }


def print_occupancy(occ: dict):
    print(f"\n=== PREFLIGHT difficulty occupancy: {occ['label']}  (n={occ['n']}) ===", flush=True)
    print("  required-level %:  " + "  ".join(f"L{k}={v}%" for k, v in occ["level_pct"].items())
          + f"   (level>=1: {occ['level_ge1_pct']}%)", flush=True)
    print(f"  treewidth:         median={occ['treewidth_median']}  tw>=3={occ['treewidth_ge3_pct']}%  "
          f"tw>=4={occ['treewidth_ge4_pct']}%  hist={occ['treewidth_hist']}", flush=True)
    print("  prop-depth thirds: "
          + "  ".join(f"{k}={v}%" for k, v in occ["depth_thirds_pct"].items()), flush=True)
    print(f"  component shares:  random-relations={occ['randomrel_pct']}%  "
          f"reduction={occ['reduction_pct']}%  composition={occ['composition_pct']}%", flush=True)


def check_occupancy_gate(occ: dict, mix_kinds: set, gate: dict) -> list:
    """Return a list of human-readable failure strings (empty == PASS). A component-conditional check
    is skipped when its component isn't configured in the mix, so a CSP-only spec isn't failed for
    lacking compositions it never generates."""
    field = {"level_ge1_pct_min": ("required-level L>=1", "level_ge1_pct"),
             "treewidth_ge3_pct_min": ("treewidth>=3 spread", "treewidth_ge3_pct"),
             "depth_long_pct_min": ("long-depth tail (depth>=7)", None),
             "randomrel_pct_min": ("random-relations present", "randomrel_pct"),
             "reduction_pct_min": ("reduction share", "reduction_pct"),
             "composition_pct_min": ("composition share", "composition_pct")}
    fails = []
    for key, thr in gate.items():
        comp = _GATE_COMPONENT.get(key)
        if comp is not None and comp not in mix_kinds:
            continue
        name, fld = field.get(key, (key, None))
        val = occ["depth_thirds_pct"]["long(>=7)"] if key == "depth_long_pct_min" else occ[fld]
        if val < thr:
            fails.append(f"{name}: got {val}% < required {thr}%")
    return fails


def _resolve_gen_workers(gen_workers, steps):
    """Pick the resample parallelism. None => AUTO: serial (1) for tiny runs (smoke / selftest, where
    spawn-pool startup would dominate a sub-50-step run), else min(4, cpu-2) so the real pretrain fans
    the expensive exact-dedₚ stream-gen across cores WITHOUT over-subscribing RAM. The cap is 4 (was 8):
    each spawn worker is a clean ~150 MB interpreter, and an OOM'd worker is what wedged the old blocking
    reader; min(4, cpu-2) keeps K workers + the trainer inside the box's RAM (see stream_prefetch's
    MEMORY BUDGET) while still fanning gen across cores. An explicit int is honoured verbatim (1 = the
    old single-stream serial path; the deterministic-reproducibility unit-of-record)."""
    if gen_workers is not None:
        return max(1, int(gen_workers))
    if steps <= 50:
        return 1
    return max(1, min(4, (os.cpu_count() or 2) - 2))


def _train_organ_stream(dev, spec, *, target, steps, pool, R, lr, theta=0.5, seed=0, log_every=15,
                        gen_workers=None):
    """The VALIDATED dominate-dedₚ on-policy loop (run_glados_staged.train_organ), fed from the
    difficulty-controlled stream instead of the RUNGS sampler: the initial pool and every on-policy
    replacement (terminal/stalled instance -> a fresh one) are drawn from the difficulty stream.

    The resample (the next fresh stream item) is the loop's bottleneck — ~90% of it is the EXACT dedₚ
    labelling of the HARD instances inside problem_stream, which is inherently expensive even on the Rust
    port. With gen_workers>1 it is fanned across cores by a deterministic round-robin prefetch pool
    (datagen.stream_prefetch); gen_workers==1 keeps the exact single-stream serial path. Both are
    reproducible from (seed, gen_workers). Same proposer, same loss, same meet. Returns (m,d,npar,log)."""
    import time
    import torch
    from .. import csp as C
    from .. import run_glados_staged as G
    torch.manual_seed(seed)
    K = _resolve_gen_workers(gen_workers, steps)
    d, npar = G.size_for("full", G.N_MAX, G.D_MAX, G.M_MAX, G.A_MAX, target, R=R, ds=min(4, R))
    m = G.FactorGraphProposer("full", G.N_MAX, G.D_MAX, G.M_MAX, G.A_MAX, d=d, R=R, ds=min(4, R)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95))

    pstream = None
    if K > 1:
        from ..datagen import stream_prefetch as SP
        pstream = SP.ParallelItemStream(spec, seed, (G.N_MAX, G.D_MAX, G.M_MAX, G.A_MAX), workers=K)
        next_item = pstream.next
    else:
        gen = (it for _, it in _difficulty_items(spec, seed))         # infinite (csp, full) stream
        next_item = lambda: next(gen)
    try:
        items = [next_item() for _ in range(pool)]
        print(f"  ORGAN train: difficulty-stream spec={spec['name']}  d_model={d}  params={npar:,}  "
              f"pool={pool}  R={R} steps={steps}  gen_workers={K}", flush=True)
        log = []
        fe_k = fe_n = 0
        t0 = time.time()
        for s in range(1, steps + 1):
            feat = G.featurize(items, dev)
            vm = feat["var_mask"]
            tgt, conflict = G.build_targets(items, dev)
            b, cls, sup = G.fwd(m, feat, vm)
            loss = G.loss_fn(sup, vm, feat["var_valid"], tgt, conflict)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            with torch.no_grad():
                new_vm = G.meet(vm, b, theta)
                kk, nn_ = G.false_elim(vm, new_vm, feat["var_valid"], tgt); fe_k += kk; fe_n += nn_
                nvc = new_vm.cpu().numpy()
                nxt = []
                for bi, (csp, dom) in enumerate(items):
                    ndom = G.dom_from_mask(nvc[bi], csp)
                    if C.status(ndom) in ("solved", "conflict") or ndom == dom:
                        nxt.append(next_item())                       # fresh stream item (on-policy)
                    else:
                        nxt.append((csp, ndom))
                items = nxt
            if s % max(1, steps // log_every) == 0 or s == 1:
                fer = fe_k / max(1, fe_n)
                log.append({"step": s, "loss": float(loss.detach()), "false_elim": fer})
                print(f"    step {s:5d}  loss {float(loss.detach()):.3f}  false_elim {fer:.4f}  "
                      f"alive {float(vm.sum(-1).mean()):.2f}  {time.time()-t0:.0f}s", flush=True)
                fe_k = fe_n = 0
    finally:
        if pstream is not None:
            pstream.close()
    return m, d, npar, log


def pretrain_organ(out="runs/general_organ_full.pt", steps=1500, target=1.5e6, pool=128, R=12,
                   lr=3e-4, seed=0, dev=None, easy=False, force=False, preflight_n=600, gate=None,
                   smoke=False, gen_workers=None):
    """STAGE 1: on-policy dominate-dedₚ training of ONE general narrow organ, saved in the
    {'state','meta'} format clair.organ.bank.load_core_organ reads. Returns (organ, meta).

    DATA SOURCE (the codex highest-leverage fix): by DEFAULT this consumes the DIFFICULTY-CONTROLLED
    streaming mix — datagen.stream.organ_spec(difficulty=True) via _train_organ_stream — NOT the old
    easy RUNGS sampler. A PREFLIGHT OCCUPANCY GATE samples `preflight_n` instances from the configured
    source, runs datagen.difficulty over them, and REFUSES TO RUN if the occupancy is still the legacy
    easy distribution (required-level L>=1 < gate threshold, etc.); the occupancy report is printed and
    stamped into the checkpoint meta regardless. `easy=True` selects the legacy RUNGS source (which the
    gate then BLOCKS — that is the point); `force=True` bypasses the gate so legacy is still reachable.
    Recipe defaults follow the sweep: ~1.5M params, R=12. Reproducible from `seed`."""
    import torch
    from .. import run_glados_staged as G
    from ..datagen import stream as S
    dev = dev or G.device()
    gate = dict(DEFAULT_OCCUPANCY_GATE, **(gate or {}))
    if smoke:
        steps, target, pool, R, preflight_n = 12, 1.2e5, 32, 6, 200

    if easy:
        spec = None                                        # legacy RUNGS (run_glados_staged)
        label = "LEGACY easy RUNGS sampler (run_glados_staged.sample_organ_corpus)"
        samples = _legacy_samples(preflight_n, seed)
        mix_kinds: set = set()                             # RUNGS has no randomrel/reduction/compose
    else:
        spec = S.organ_spec(difficulty=True)               # the difficulty-controlled mix (the fix)
        label = f"difficulty stream organ_spec(difficulty=True) [{spec['name']}]"
        samples = _difficulty_samples(spec, preflight_n, seed)
        mix_kinds = {c["kind"] for c in spec["mix"]}

    # ---- PREFLIGHT: profile the configured source + GATE the run ----
    print(f"[organ] PREFLIGHT occupancy gate: profiling {preflight_n} instances from {label}",
          flush=True)
    occ = occupancy_profile(samples, label)
    print_occupancy(occ)
    fails = check_occupancy_gate(occ, mix_kinds, gate)
    if fails:
        msg = ("pretrain refusing to run: difficulty occupancy is the legacy easy distribution.\n"
               f"  source: {label}\n"
               "  expected the difficulty-controlled mix (datagen.stream.organ_spec(difficulty=True)); "
               "got:\n    " + "\n    ".join(fails)
               + "\n  -> re-run WITHOUT --easy to consume the difficulty stream, "
                 "or pass --force to override the gate.")
        if force:
            print("[organ] WARNING (gate overridden by force):\n  " + msg, flush=True)
        else:
            raise RuntimeError(msg)
    else:
        print(f"[organ] PREFLIGHT PASS — occupancy meets the design thresholds "
              f"(level>=1 {occ['level_ge1_pct']}% >= {gate['level_ge1_pct_min']}%)", flush=True)

    # ---- train (difficulty stream by default; legacy RUNGS only via --easy --force) ----
    if easy:
        print(f"[organ] dominate-dedₚ over LEGACY rungs={G.RUNGS}  steps={steps} target={target:g} "
              f"R={R} dev={dev}", flush=True)
        organ, d, npar, log = G.train_organ(dev, G.RUNGS, target=target, steps=steps, pool=pool,
                                            R=R, lr=lr, seed=seed)
        source = {"mode": "legacy_rungs", "rungs": list(G.RUNGS), "forced": bool(force)}
    else:
        print(f"[organ] dominate-dedₚ over DIFFICULTY stream  steps={steps} target={target:g} "
              f"R={R} dev={dev}", flush=True)
        K = _resolve_gen_workers(gen_workers, steps)
        organ, d, npar, log = _train_organ_stream(dev, spec, target=target, steps=steps, pool=pool,
                                                  R=R, lr=lr, seed=seed, gen_workers=K)
        source = {"mode": "difficulty_stream", "spec": spec["name"], "gen_workers": K}

    for p in organ.parameters():
        p.requires_grad_(False)
    organ.eval()
    meta = {"N_MAX": G.N_MAX, "D_MAX": G.D_MAX, "M_MAX": G.M_MAX, "A_MAX": G.A_MAX,
            "d": d, "params": npar, "R": R, "seed": seed,
            "data_source": source, "difficulty_occupancy": occ, "occupancy_gate": gate}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({"state": organ.state_dict(), "meta": meta}, out)
    print(f"[organ] saved {out}  d={d} params={npar:,} R={R}  source={source['mode']}  "
          f"(occupancy recorded in meta)", flush=True)
    return organ, meta


# back-compat alias (clair.organ.selftest + older notes call train_organ)
train_organ = pretrain_organ


# ============================================================ STAGE 2: weave into a host LM
def _weave_args(base_id, **over):
    """Default woven hyperparameters (the run_glados_staged.main defaults), overridable by kwargs."""
    a = SimpleNamespace(
        base=base_id, steps=2500, warm_steps=400, bs=12, lora_r=16, lora_lr=2e-4, gamma_lr=1e-3,
        alpha_lr=3e-4, gamma_hidden=256, mid_layer=6, inject_layer=12, dctx=256, alpha_dp=384,
        alpha_heads=6, organ_d=128, organ_heads=4, organ_layers=2, organ_T=12, aux_w=0.5,
        alpha_sup_w=1.0, engage_thr=5.0, per_rung_train=400, per_rung_eval=80, det_only=1, seed=0,
        # ALPHA_STRUCT inference levers (train-WITH; default off = the validated single-shot recipe):
        #   search_K>1 → α-as-search (verifier-select the structure); loop_T>1 → iterative α↔organ loop.
        struct_sup_w=1.0, search_K=1, search_temp=1.0, loop_T=1)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def weave(base_id="allenai/OLMo-2-0425-1B", *, regime="hard", two_stream=True, smoke=False,
          organ_mode="bank", out=None, dev=None, **hp):
    """STAGE 2: graft the organ into `base_id` and train the woven readout with the FULL VALIDATED
    ENGAGEMENT RECIPE BY DEFAULT — two-stream lever + J0 direct-α-supervision + the structured (rich-set)
    causal-control readout — lifted here out of run_glados_staged's experiment driver so the canonical
    callable runs the real thing without flags. Returns (woven_model, metrics).

    organ_mode (the blocker-A consolidation, DEFAULT 'bank'):
      'bank'   — the live organ IS the bank + composer: α compiles the per-cell lattice; the composer
                 runs the certified ops (Arc/Factor/Modular/GF2/Macro — the soundness FLOOR) PLUS the
                 pretrained CoreNarrowOrgan (runs/general_organ_full.pt, verifier-gated) PLUS α (gated)
                 on the problem's true structure; γ reads the COMPOSED lattice. Multi-faculty,
                 certified-floor-backed, uses the pretrained organ.
      'latent' — the legacy standalone frozen LatentNarrower path (single-faculty, kept for ablation).

    `regime` selects (rungs, split) from run_glados_staged.LIVE_REGIMES {small, hard, large}; `hp`
    overrides any hyperparameter. The host graft is model-agnostic (generalized decoder-layer lookup),
    so this weaves onto ANY of the 7 proven bases identically."""
    import gc
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoConfig
    from .. import run_glados_staged as G
    from . import bank_woven as BW

    dev = dev or G.device()
    a = _weave_args(base_id, **hp)
    if smoke:
        a.steps, a.warm_steps, a.per_rung_train, a.per_rung_eval, a.bs = 60, 40, 24, 12, 6
        a.lora_r, a.gamma_hidden, a.organ_d, a.organ_T = 8, 128, 96, 8

    print(f"[weave] base={base_id} regime={regime} two_stream={two_stream} organ_mode={organ_mode} "
          f"smoke={smoke}", flush=True)
    tok = AutoTokenizer.from_pretrained(base_id)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(base_id)
    D = cfg.hidden_size
    nL = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "num_layers", None)
    a.inject_layer = min(a.inject_layer, nL - 1)
    a.mid_layer = min(a.mid_layer, a.inject_layer - 1)

    rungs, split = G.LIVE_REGIMES[G._REGIME_ALIAS.get(regime, regime)]
    rng_tr = np.random.default_rng(a.seed + 100)
    rng_ev = np.random.default_rng(a.seed + 200)
    train_recs = G.build_live_pool(rng_tr, rungs, split, a.per_rung_train,
                                   det_only=bool(a.det_only), two_stream=two_stream)
    eval_recs = G.build_live_pool(rng_ev, rungs, split, a.per_rung_eval,
                                  det_only=bool(a.det_only), two_stream=two_stream)
    print(f"[weave] data: {len(train_recs)} train / {len(eval_recs)} eval recs  (D={D}, nL={nL})",
          flush=True)

    if organ_mode == "alpha_struct":
        # STAGE-2 ALPHA_STRUCT: α EMITS the structure; the composer runs on csp_α (NOT rec['csp']). The
        # eval arbiter is the acceptance gate (clair.organ.alpha_struct_diag) — the SHUFFLED-INVERTED
        # test — run separately on the trained model; train returns the woven model.
        model = BW.train_alpha_struct_woven((base_id, D, nL), tok, dev, a, train_recs, eval_recs,
                                            two_stream=two_stream,
                                            use_lora=bool(getattr(a, "use_lora", True)))
        metrics = {"organ_mode": "alpha_struct", "note": "run clair.organ.alpha_struct_diag for the gate"}
    elif organ_mode == "bank":
        model = BW.train_bank_woven((base_id, D, nL), tok, dev, a, train_recs, eval_recs,
                                    two_stream=two_stream)
        metrics = BW.bank_controls(model, eval_recs, tok, dev, a.bs, a.engage_thr,
                                   two_stream=two_stream)
    else:
        model = G.train_live_woven((base_id, D, nL), tok, dev, a, train_recs, eval_recs,
                                   two_stream=two_stream)
        metrics = G._live_controls(model, eval_recs, tok, dev, a.bs, a.engage_thr,
                                   two_stream=two_stream)
    if organ_mode == "alpha_struct":
        print(f"[weave] {regime} | alpha_struct | two_stream={two_stream} — trained; "
              f"run `python -m clair.organ.alpha_struct_diag` for the acceptance gate.", flush=True)
    else:
        G._report_regime(f"{regime} | {organ_mode} | two_stream={two_stream}", metrics, a.engage_thr)

    if out:
        from .. import eval_suite as ES
        wcfg = {"kind": "live", "D": D, "K": G.K, "lora_r": a.lora_r, "gamma_hidden": a.gamma_hidden,
                "inject_layer": a.inject_layer, "mid_layer": a.mid_layer, "dctx": a.dctx,
                "alpha_dp": a.alpha_dp, "alpha_heads": a.alpha_heads, "organ_d": a.organ_d,
                "organ_heads": a.organ_heads, "organ_layers": a.organ_layers, "organ_T": a.organ_T}
        ES.save_woven(model, out, base_id=base_id, cfg=wcfg)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, metrics


# ============================================================ STAGE 3: RLVR (organ-as-process-reward)
def _organ_process_reward_builder(w_outcome=0.7, w_process=0.3, core_ckpt="runs/general_organ_full.pt"):
    """Build the MIXED reward_func (0.7·outcome + 0.3·organ-process), soundness-gated. The organ-process
    term is the candidate-set cardinality drop / survival-depth (clair.organ.process_reward) of the
    emitted answer; it is DENSE even when the outcome is 0. It is active for examples that carry CSP
    structure in their metadata (the GLaDOS rungs — the organ's native domain); for reasoning-gym text
    tasks the organ does not apply, so the process term is 0 and the reward reduces to the outcome (a
    text→CSP compile to extend the process term to NL tasks is the blocker-7 frontier)."""
    import json
    from .. import rlvr_pipeline as RL
    from . import process_reward as PR
    from .. import csp as C
    opr = PR.OrganProcessReward(core_ckpt=core_ckpt, use_core=True)
    base_outcome = RL.make_reward(strict=True)

    def mixed(completions, answer, source, metadata, **kwargs):
        outcomes = base_outcome(completions, answer, source, metadata, **kwargs)
        rewards = []
        for comp, gold, meta_s, oc in zip(completions, answer, metadata, outcomes):
            proc = 0.0
            try:
                meta = json.loads(meta_s) if isinstance(meta_s, str) else (meta_s or {})
                cspd = meta.get("csp")                         # {cons,n,d,query,emitted}: GLaDOS rungs only
                if cspd is not None:
                    csp = C.CSP(cspd["n"], cspd["d"], tuple((tuple(sc), frozenset(map(tuple, al)))
                                                            for sc, al in cspd["cons"]))
                    ea = RL.extract_answer(comp)
                    emitted = int(ea) if str(ea).lstrip("-").isdigit() else cspd.get("emitted", 0)
                    proc = opr.reward(csp, cspd["query"], emitted).process
            except Exception:
                proc = 0.0
            rewards.append(PR.mix_reward(float(oc), proc, w_outcome, w_process))
        return rewards

    mixed.__name__ = "organ_process_mixed_reward"
    return mixed


def rlvr(task="chain_sum", steps=300, *, smoke=False, organ_process=True, out="runs/rlvr_pipeline",
         **over):
    """STAGE 3: RL-from-verifiable-rewards on the woven policy (TRL Dr.GRPO). The ORGAN-AS-PROCESS-REWARD
    (LSRL-style per-step candidate-set cardinality drop / survival-depth, mixed 0.7·outcome + 0.3·process,
    soundness-gated — clair.organ.process_reward) is BUILT HERE and passed into the loop via the pipeline's
    reward_builder hook (no longer just an insertion point). organ_process=False = pure outcome verifier."""
    from .. import rlvr_pipeline as RL
    argv = ["clair.rlvr_pipeline", "--task", str(task), "--steps", str(steps), "--out", str(out)]
    if smoke:
        argv.append("--smoke")
    for k, v in over.items():
        argv += [f"--{k}", str(v)]
    builder = _organ_process_reward_builder if organ_process else None
    old = sys.argv
    try:
        sys.argv = argv
        return RL.main(reward_builder=builder)
    finally:
        sys.argv = old


# ============================================================ CLI
def main():
    ap = argparse.ArgumentParser(description="GLaDOS canonical pipeline: pretrain → weave → rlvr")
    sub = ap.add_subparsers(dest="cmd", required=True)

    po = sub.add_parser("pretrain", aliases=["organ"], help="STAGE 1: train + save the general organ")
    po.add_argument("--out", default="runs/general_organ_full.pt")
    po.add_argument("--steps", type=int, default=1500)
    po.add_argument("--target", type=float, default=1.5e6)
    po.add_argument("--pool", type=int, default=128)
    po.add_argument("--R", type=int, default=12)
    po.add_argument("--lr", type=float, default=3e-4)
    po.add_argument("--seed", type=int, default=0)
    po.add_argument("--preflight_n", type=int, default=600,
                    help="instances sampled for the preflight difficulty-occupancy gate")
    po.add_argument("--easy", "--legacy", action="store_true", dest="easy",
                    help="LEGACY: train on the old easy RUNGS sampler (the occupancy gate will BLOCK it)")
    po.add_argument("--force", action="store_true",
                    help="override the occupancy gate (required to actually run --easy/legacy)")
    po.add_argument("--smoke", action="store_true", help="tiny end-to-end pretrain smoke (CPU-ok)")
    po.add_argument("--gen_workers", type=int, default=None,
                    help="parallel resample workers for the difficulty stream-gen (the on-policy "
                         "bottleneck); None=AUTO (serial for tiny runs, else min(8,cpu-1)); 1=serial")

    pw = sub.add_parser("weave", help="STAGE 2: graft + train the woven readout (engagement mechanism)")
    pw.add_argument("--base", default="allenai/OLMo-2-0425-1B")
    pw.add_argument("--regime", default="hard", help="small | hard | large (LIVE_REGIMES)")
    pw.add_argument("--organ_mode", default="bank", choices=["bank", "latent", "alpha_struct"],
                    help="bank=bank+composer (multi-faculty, certified-floor, pretrained organ; DEFAULT); "
                         "latent=legacy standalone LatentNarrower (ablation); "
                         "alpha_struct=α EMITS the structure, composer runs on csp_α (the SHUFFLED fix)")
    pw.add_argument("--two_stream", type=int, default=1, help="1=two-stream (cells-gen) 0=single-stream")
    pw.add_argument("--steps", type=int, default=2500)
    pw.add_argument("--warm_steps", type=int, default=400)
    pw.add_argument("--search_K", type=int, default=1,
                    help="alpha_struct LEVER A (α-as-search): K candidate structures, verifier-selected "
                         "(1=greedy, the single-shot core)")
    pw.add_argument("--search_temp", type=float, default=1.0, help="α-as-search candidate sampling temperature")
    pw.add_argument("--loop_T", type=int, default=1,
                    help="alpha_struct LEVER B (iterative α↔organ loop): T feedback steps (1=single-shot core)")
    pw.add_argument("--out", default=None)
    pw.add_argument("--smoke", action="store_true")

    pr = sub.add_parser("rlvr", help="STAGE 3: RLVR (Dr.GRPO, exact-verifier reward)")
    pr.add_argument("--task", default="chain_sum")
    pr.add_argument("--steps", type=int, default=300)
    pr.add_argument("--out", default="runs/rlvr_pipeline")
    pr.add_argument("--smoke", action="store_true")

    a = ap.parse_args()
    if a.cmd in ("pretrain", "organ"):
        pretrain_organ(out=a.out, steps=a.steps, target=a.target, pool=a.pool, R=a.R, lr=a.lr,
                       seed=a.seed, easy=a.easy, force=a.force, preflight_n=a.preflight_n,
                       smoke=a.smoke, gen_workers=a.gen_workers)
    elif a.cmd == "weave":
        weave(a.base, regime=a.regime, two_stream=bool(a.two_stream), organ_mode=a.organ_mode,
              smoke=a.smoke, out=a.out, steps=a.steps, warm_steps=a.warm_steps,
              search_K=a.search_K, search_temp=a.search_temp, loop_T=a.loop_T)
    elif a.cmd == "rlvr":
        rlvr(task=a.task, steps=a.steps, out=a.out, smoke=a.smoke)


if __name__ == "__main__":
    main()
