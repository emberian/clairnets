"""clair/organ/run_multifaculty.py — TRAIN + EVAL the multi-faculty woven (the part-4 build).

Trains MultiFacultyWoven on a MIX of single-faculty (CSP determined-coloring/equality + the GRAPH
reachability faculty) and the CROSS-FACULTY composition task, with the validated recipe (structure-sup
+ edge-sup + router-CE + J0 + LM-CE). Watches: does the router learn to route, does the 2nd (graph)
faculty engage, and does the cross-faculty task get solved (vs single-faculty baselines that cannot)?

  python -m clair.organ.run_multifaculty --smoke
  python -m clair.organ.run_multifaculty --steps 1200 --warm 600 --out runs/multifaculty.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import run_glados_staged as G
from ..oracle_readout import _mention_tensor, ABSTAIN_STR, n_trainable
from ..latent_organ import dominate_dedp_loss
from .alpha_struct import StructureRack, structure_sup_loss, facts_to_struct_targets
from . import faculty as FAC
from .faculty import LIVE_FACULTIES, FACULTY_IDX, edges_to_adj_target
from .multifaculty import MultiFacultyComposerOrgan, MultiFacultyWoven
from . import faculty_tasks as FT


# ============================================================ pool
def build_mf_pool(rng, n_csp, n_graph, n_cross, split="id", two_stream=True):
    """Mixed multi-faculty pool: determined CSP (coloring/equality) + graph reachability + cross. With
    two_stream, the gen prompt is fact-ABLATED (α reads the full text) so the organ is the only route."""
    recs = []
    if n_csp:
        csp = G.build_live_pool(rng, ["coloring", "equality"], split,
                                max(1, n_csp // 2), det_only=True, two_stream=two_stream)
        for r in csp[:n_csp]:
            r["faculty"] = "csp"; r["source"] = 0; r["target"] = int(r["query"])
            r["true_facts"] = r["facts"]; r["true_edges"] = []
        recs += csp[:n_csp]
    for _ in range(n_graph):
        recs.append(FT.gen_graph_record(rng, two_stream=two_stream))
    for k in range(n_cross):
        recs.append(FT.gen_cross_record(rng, want_yes=(k % 2 == 0), two_stream=two_stream))
    rng.shuffle(recs)
    return recs


# ============================================================ batch (+ all supervision targets + specs)
def _mf_batch(recs, tok, dev, two_stream=True):
    ba = G.build_live_batch(recs, tok, dev, two_stream=two_stream)
    Nmax = ba["vmask"].shape[1]
    B = len(recs)
    pin_t, pair_t = facts_to_struct_targets([r.get("true_facts", []) for r in recs],
                                            [r["n"] for r in recs], Nmax)
    ba["pin_tgt"] = pin_t.to(dev); ba["pair_tgt"] = pair_t.to(dev)
    edge_t = torch.zeros(B, Nmax, Nmax, device=dev)
    edge_m = torch.zeros(B, device=dev)            # 1 for graph/cross (graph head supervised)
    struct_m = torch.zeros(B, device=dev)          # 1 for csp/cross (csp head supervised)
    csp_m = torch.zeros(B, device=dev)             # 1 for csp (J0 candidate-head grounding)
    router_t = torch.zeros(B, dtype=torch.long, device=dev)
    specs = []
    for b, r in enumerate(recs):
        fac = r["faculty"]; router_t[b] = FACULTY_IDX[fac]
        if fac in ("graph", "cross"):
            edge_m[b] = 1.0
            A = edges_to_adj_target([tuple(e) for e in r.get("true_edges", [])], r["n"])
            edge_t[b, : r["n"], : r["n"]] = torch.from_numpy(A).to(dev)
        if fac in ("csp", "cross"):
            struct_m[b] = 1.0
        if fac == "csp":
            csp_m[b] = 1.0
        specs.append({"faculty": fac, "n": r["n"], "d": len(r["vnames"]) if fac == "csp" else 2,
                      "source": int(r.get("source", 0)), "target": int(r.get("target", r.get("query", 0)))})
    ba["edge_tgt"] = edge_t; ba["edge_mask"] = edge_m; ba["struct_mask"] = struct_m
    ba["csp_mask"] = csp_m; ba["router_tgt"] = router_t; ba["specs"] = specs
    ba["dvec"] = torch.tensor([(len(r["vnames"]) if r["faculty"] == "csp" else r["n"]) for r in recs],
                              device=dev)
    return ba


def _edge_sup_loss(edge_logits, edge_tgt, vmask, edge_mask, wpos=4.0):
    """Masked, recall-weighted BCE of the graph-head edge logits vs the true adjacency, over valid
    cell-pairs of graph/cross instances only (a dropped edge under-reaches the closure → up-weight pos)."""
    B, N, _ = edge_logits.shape
    pm = (vmask[:, :, None] * vmask[:, None, :])
    eye = torch.eye(N, device=edge_logits.device)[None]
    pm = pm * (1 - eye) * edge_mask.view(B, 1, 1)
    if pm.sum() < 1:
        return edge_logits.sum() * 0.0
    w = torch.where(edge_tgt > 0.5, torch.tensor(wpos, device=edge_logits.device),
                    torch.tensor(1.0, device=edge_logits.device))
    bce = F.binary_cross_entropy_with_logits(edge_logits, edge_tgt, weight=w, reduction="none")
    return (bce * pm).sum() / pm.sum().clamp_min(1.0)


# ============================================================ no-op@init
def verify_noop_mf(model, tok, dev, rec):
    enc = tok(rec["prompt"], return_offsets_mapping=True, return_tensors="pt")
    ids = enc["input_ids"].to(dev); attn = torch.ones_like(ids)
    mention = _mention_tensor([rec["mentions"]], enc["offset_mapping"], 1, rec["n"], ids.size(1), dev)
    specs = [{"faculty": rec["faculty"], "n": rec["n"],
              "d": len(rec["vnames"]) if rec["faculty"] == "csp" else 2,
              "source": int(rec.get("source", 0)), "target": int(rec.get("target", rec.get("query", 0)))}]
    dvec = torch.tensor([rec["n"]], device=dev)
    with torch.no_grad():
        base = model.model(input_ids=ids, attention_mask=attn).logits.float()
    with torch.no_grad(), model.live(mention, attn, inject=True, capture=True, specs=specs, dvec=dvec):
        g0 = model.logits(ids, attn).float()
    noop = float((base - g0).abs().max())
    with torch.no_grad():
        saved = model.gamma.alpha.data.clone(); model.gamma.alpha.data.fill_(2.0)
        with model.live(mention, attn, inject=True, capture=True, specs=specs, dvec=dvec):
            g1 = model.logits(ids, attn).float()
        model.gamma.alpha.data.copy_(saved)
    return noop, float((base - g1).abs().max())


# ============================================================ scoring (per-faculty acc + router acc)
@torch.no_grad()
def score_mf(model, recs, tok, dev, bs=8, route_by_model=True, control="true", two_stream=True):
    """Per-faculty answer accuracy + router accuracy, end-to-end through the LM. control='zero' disables
    injection (the no-organ baseline → the cross task should collapse to chance). With two_stream the α
    capture reads the full-text alpha_prompt while the LM scores on the fact-ablated gen prompt."""
    model.eval()
    K = G.K
    inject = control != "zero"
    by_fac = {f: [0, 0] for f in LIVE_FACULTIES}
    router_ok = router_tot = 0
    for i in range(0, len(recs), bs):
        chunk = recs[i:i + bs]; Bp = len(chunk); Nmax = max(r["n"] for r in chunk)
        dvec = torch.tensor([r["n"] for r in chunk], device=dev)
        specs = [{"faculty": r["faculty"], "n": r["n"],
                  "d": len(r["vnames"]) if r["faculty"] == "csp" else 2,
                  "source": int(r.get("source", 0)),
                  "target": int(r.get("target", r.get("query", 0)))} for r in chunk]
        surv = None
        if inject:
            cap_p = [r.get("alpha_prompt", r["prompt"]) if two_stream else r["prompt"] for r in chunk]
            cap_m = [r.get("alpha_mentions", r["mentions"]) if two_stream else r["mentions"] for r in chunk]
            penc = tok(cap_p, return_offsets_mapping=True, padding=True, return_tensors="pt")
            pids = penc["input_ids"].to(dev); pattn = penc["attention_mask"].to(dev)
            pment = _mention_tensor(cap_m, penc["offset_mapping"], Bp, Nmax, pids.size(1), dev)
            with model.live(pment, pattn, inject=False, capture=True, specs=specs, dvec=dvec,
                            route_by_model=route_by_model):
                _ = model.logits(pids, pattn)
            surv = model._captured_surv[:, :Nmax, :].clone()
            rl = model._last["router"]
            for b, r in enumerate(chunk):
                router_ok += int(int(rl[b].argmax()) == FACULTY_IDX[r["faculty"]]); router_tot += 1
        # score candidate answers
        fulls, plens, spans, prob_of = [], [], [], []
        for pi, r in enumerate(chunk):
            for cand in [" " + v for v in r["vnames"]] + [" " + ABSTAIN_STR]:
                fulls.append(r["prompt"] + cand); plens.append(len(r["prompt"]))
                spans.append(r["mentions"]); prob_of.append(pi)
        enc = tok(fulls, return_offsets_mapping=True, padding=True, return_tensors="pt")
        ids = enc["input_ids"].to(dev); attn = enc["attention_mask"].to(dev)
        offsets = enc["offset_mapping"]; T = ids.size(1)
        if inject:
            mention = _mention_tensor(spans, offsets, len(fulls), Nmax, T, dev)
            idx = torch.tensor(prob_of, device=dev)
            ctx = model.live(mention, attn, inject=True, capture=False, override=surv[idx],
                             dvec=dvec[idx])
        else:
            ctx = model.live(None, None, inject=False)
        with ctx:
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
            scores[row] = (tok_lp[row] * mask).sum() / mask.sum().clamp_min(1.0)
        pof = torch.tensor(prob_of, device=dev)
        for pi, r in enumerate(chunk):
            rows = (pof == pi).nonzero().flatten()
            pred = int(rows[int(scores[rows].argmax())] - rows[0])
            d = by_fac[r["faculty"]]; d[0] += int(pred == r["gold_idx"]); d[1] += 1
    return {"by_faculty": {f: (v[0] / max(1, v[1])) for f, v in by_fac.items()},
            "n_by_faculty": {f: v[1] for f, v in by_fac.items()},
            "router_acc": router_ok / max(1, router_tot)}


# ============================================================ train
def train_multifaculty(olmo_ids, tok, dev, a, train_recs, eval_recs):
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
    mid = min(a.mid_layer, nL - 1); inj = min(a.inject_layer, nL - 1)
    assert mid < inj
    rack = StructureRack(D, G.K, faculties=LIVE_FACULTIES, dp=a.alpha_dp, heads=a.alpha_heads)
    composer = MultiFacultyComposerOrgan(dev="cpu", core_ckpt=getattr(a, "core_ckpt",
                                         "runs/general_organ_full.pt"), use_core=getattr(a, "use_core", True))
    model = MultiFacultyWoven(peft_model, D, G.K, rack, composer, mid, inj,
                              gamma_hidden=a.gamma_hidden).to(dev)
    model.alpha.float(); model.cand.float(); model.gamma.float()
    noop, live = verify_noop_mf(model, tok, dev, eval_recs[0])
    print(f"\n  MULTIFACULTY-WOVEN  mid {mid} -> inject {inj}/{nL}  faculties={composer.faculties()}  "
          f"trainable {n_trainable(model):,}", flush=True)
    print(f"  NO-OP @ INIT max|base-(LoRA+gate0)| = {noop:.3e} (expect ~0) | gate-on moves {live:.3e}",
          flush=True)
    lora_params = [p for n, p in model.model.named_parameters() if p.requires_grad]
    rack_cand = list(model.alpha.parameters()) + list(model.cand.parameters())
    rng = np.random.default_rng(a.seed + 7)

    two_stream = getattr(a, "two_stream", True)

    def batch(n):
        idxs = rng.integers(0, len(train_recs), n).tolist()
        return _mf_batch([train_recs[i] for i in idxs], tok, dev, two_stream=two_stream)

    def capture(ba):
        m, at, ids = ((ba["a_mention"], ba["a_attn"], ba["a_input_ids"]) if two_stream
                      else (ba["mention"], ba["attn"], ba["input_ids"]))
        with model.live(m, at, inject=False, capture=True, specs=ba["specs"], dvec=ba["dvec"]):
            _ = model.logits(ids, at)

    def aux_losses(ba):
        L = model._last
        struct_vmask = L["vmask"] * ba["struct_mask"].unsqueeze(-1)
        l_struct = structure_sup_loss(L["pin"], L["pair"], ba["pin_tgt"], ba["pair_tgt"], struct_vmask,
                                      wpos=getattr(a, "wpos", 2.0), wnone=getattr(a, "wnone", 1.0))
        l_edge = _edge_sup_loss(L["edge"], ba["edge_tgt"], L["vmask"], ba["edge_mask"])
        l_router = F.cross_entropy(L["router"], ba["router_tgt"])
        l_j0 = dominate_dedp_loss([L["b0"]], ba["tgt"], L["vmask"] * ba["csp_mask"].unsqueeze(-1))
        return l_struct, l_edge, l_router, l_j0

    # ---- Phase A: warm rack heads + router + candidate head (no LM-CE) ----
    optA = torch.optim.AdamW([{"params": lora_params, "lr": a.lora_lr, "weight_decay": 0.01},
                              {"params": rack_cand, "lr": a.alpha_lr, "weight_decay": 0.0}],
                             betas=(0.9, 0.95))
    model.train(); t0 = time.time()
    for s in range(1, a.warm_steps + 1):
        ba = batch(a.bs); capture(ba)
        ls, le, lr, lj = aux_losses(ba)
        loss = a.struct_sup_w * ls + a.edge_sup_w * le + a.router_w * lr + a.alpha_sup_w * lj
        optA.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0); optA.step()
        if s % max(1, a.warm_steps // 6) == 0 or s == 1:
            print(f"  [warmA] step {s:5d}  struct {ls.item():.3f}  edge {le.item():.3f}  "
                  f"router {lr.item():.3f}  J0 {lj.item():.3f}  {time.time()-t0:.0f}s", flush=True)

    # ---- Phase B: LM-CE (γ reads the DETACHED dispatched lattice) + standing aux ----
    optB = torch.optim.AdamW([{"params": lora_params, "lr": a.lora_lr, "weight_decay": 0.01},
                              {"params": rack_cand, "lr": a.alpha_lr, "weight_decay": 0.0},
                              {"params": list(model.gamma.parameters()), "lr": a.gamma_lr, "weight_decay": 0.0}],
                             betas=(0.9, 0.95))
    model.train(); t0 = time.time(); seen_open = False
    for s in range(1, a.steps + 1):
        ba = batch(a.bs)
        capture(ba)                                          # dispatched lattice (γ off) + heads
        ls, le, lr, lj = aux_losses(ba)
        surv = model._captured_surv[:, : ba["mention"].shape[1], :]
        with model.live(ba["mention"], ba["attn"], inject=True, capture=False, override=surv,
                        dvec=ba["dvec"]):
            logits = model.logits(ba["input_ids"], ba["attn"]).float()
        lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                             ba["labels"][:, 1:].reshape(-1), ignore_index=-100)
        loss = lm + a.struct_sup_w * ls + a.edge_sup_w * le + a.router_w * lr + a.alpha_sup_w * lj
        optB.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0); optB.step()
        gate = float(torch.tanh(model.gamma.alpha)); seen_open = seen_open or abs(gate) > 1e-3
        if s % max(1, a.steps // 12) == 0 or s == 1:
            sc = score_mf(model, eval_recs, tok, dev, bs=a.bs, route_by_model=True, two_stream=two_stream)
            bf = sc["by_faculty"]
            print(f"  step {s:5d}  lm {lm.item():.3f}  edge {le.item():.3f}  router {lr.item():.3f}  "
                  f"gate {gate:+.3f}  route-acc {sc['router_acc']*100:.0f}%  "
                  f"csp {bf['csp']*100:.0f} graph {bf['graph']*100:.0f} cross {bf['cross']*100:.0f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            model.train()
    if not seen_open:
        print("  [WARN] γ gate never opened.", flush=True)
    return model


# ============================================================ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--warm", type=int, dest="warm_steps", default=800)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_lr", type=float, default=1e-4)
    ap.add_argument("--alpha_lr", type=float, default=3e-4)
    ap.add_argument("--gamma_lr", type=float, default=1e-3)
    ap.add_argument("--gamma_hidden", type=int, default=256)
    ap.add_argument("--alpha_dp", type=int, default=384)
    ap.add_argument("--alpha_heads", type=int, default=6)
    ap.add_argument("--mid_layer", type=int, default=6)
    ap.add_argument("--inject_layer", type=int, default=12)
    ap.add_argument("--struct_sup_w", type=float, default=1.0)
    ap.add_argument("--edge_sup_w", type=float, default=1.0)
    ap.add_argument("--router_w", type=float, default=1.0)
    ap.add_argument("--alpha_sup_w", type=float, default=1.0)
    ap.add_argument("--wpos", type=float, default=2.0)
    ap.add_argument("--wnone", type=float, default=1.0)
    ap.add_argument("--n_csp", type=int, default=300)
    ap.add_argument("--n_graph", type=int, default=300)
    ap.add_argument("--n_cross", type=int, default=300)
    ap.add_argument("--n_eval", type=int, default=80)
    ap.add_argument("--core_ckpt", default="runs/general_organ_full.pt")
    ap.add_argument("--single_stream", action="store_true",
                    help="disable two-stream (default two-stream: gen prompt fact-ablated → organ is the only route)")
    ap.add_argument("--out", default="runs/multifaculty.json")
    ap.add_argument("--ckpt", default="runs/multifaculty.pt")
    a = ap.parse_args()
    a.two_stream = not a.single_stream
    if a.smoke:
        a.steps, a.warm_steps, a.bs = 40, 30, 6
        a.n_csp = a.n_graph = a.n_cross = 40; a.n_eval = 24

    dev = G.device()
    tok = _tok(a.model)
    rng = np.random.default_rng(a.seed)
    print(f"device={dev}  building multi-faculty pool "
          f"(csp={a.n_csp} graph={a.n_graph} cross={a.n_cross}) ...", flush=True)
    train = build_mf_pool(rng, a.n_csp, a.n_graph, a.n_cross, two_stream=a.two_stream)
    ev = build_mf_pool(np.random.default_rng(a.seed + 1), a.n_eval, a.n_eval, a.n_eval,
                       two_stream=a.two_stream)
    olmo_ids = _olmo_ids(a.model, dev)
    model = train_multifaculty(olmo_ids, tok, dev, a, train, ev)

    # final eval: router argmax dispatch + the no-organ control (cross should collapse)
    sc = score_mf(model, ev, tok, dev, bs=a.bs, route_by_model=True, control="true", two_stream=a.two_stream)
    sc0 = score_mf(model, ev, tok, dev, bs=a.bs, route_by_model=True, control="zero", two_stream=a.two_stream)
    proof = FT.prove_multifaculty(n_inst=200, seed=123)
    print("\n================ MULTIFACULTY WOVEN — FINAL ================")
    print(f"  router accuracy (end-to-end, router argmax dispatch): {sc['router_acc']*100:.1f}%")
    for f in LIVE_FACULTIES:
        print(f"  {f:6s} answer acc: {sc['by_faculty'][f]*100:5.1f}%  (no-organ {sc0['by_faculty'][f]*100:.1f}%)  "
              f"n={sc['n_by_faculty'][f]}")
    out = {"args": vars(a), "final": sc, "no_organ": sc0, "exact_proof": proof}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    torch.save({"state_dict": model.state_dict(), "args": vars(a), "final": sc}, a.ckpt)
    print(f"\nwrote {a.out} + {a.ckpt}", flush=True)
    return out


def _tok(model_id):
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(model_id)
    t.padding_side = "right"
    if t.pad_token is None:
        t.pad_token = t.eos_token
    return t


def _olmo_ids(model_id, dev):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    return (model_id, cfg.hidden_size, cfg.num_hidden_layers)


if __name__ == "__main__":
    main()
