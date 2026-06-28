"""clair/organ/multifaculty.py — the MULTI-FACULTY woven: α's router picks a faculty, the matching head
emits THAT faculty's native struct_α, and a DISPATCHING composer routes by state_type to the right
faculty (CSP reduced-product / certified graph reachability / the CSP→flow→graph cross composite). γ
(zero-init gate ⇒ no-op@init) reads the dispatched per-cell lattice → the LM head answers.

This de-CSP-locks bank_woven.AlphaStructWoven (which hardwired build_csp_from_struct → a literal CSPState
+ state_type='csp-domain'). Here:
  * the StructureRack ROUTER fires (3-way over faculty.LIVE_FACULTIES = csp/graph/cross),
  * the GRAPH head's [N,N] edge logits become a real GraphReachState (the 2nd faculty, end-to-end),
  * the CROSS dispatch threads the CSP solution into the active-edge set (the faculty-A→flow→faculty-B),
  * every faculty reads out through the SAME per-cell [N,K] γ channel (no-op@init preserved).

Training (train_multifaculty): the dispatch is TEACHER-FORCED to each record's true faculty (so the LM
readout always sees the right faculty while the router learns); the router is trained by a separate CE on
the faculty label; the CSP head by structure-supervision; the GRAPH head by edge BCE; γ + LoRA by the
answer-span LM-CE. At eval the dispatch uses the ROUTER's own argmax, so router accuracy is a real,
end-to-end-tested number. The composer NEVER reads rec['csp'] (it runs on α's emitted struct_α)."""
from __future__ import annotations

import contextlib
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import csp as C
from ..latent_organ import dominate_dedp_loss
from ..oracle_readout import OracleGamma, _mention_tensor
from .alpha_struct import (StructureRack, decode_structure, structure_sup_loss,
                           facts_to_struct_targets, build_csp_from_struct)
from . import faculty as FAC
from .faculty import (LIVE_FACULTIES, FACULTY_IDX, GraphReach, GraphReachState,
                      CrossState, CrossCSPGraph, decode_edges, edges_to_adj_target,
                      IsingState, IsingMaxcut, TypeState, TypeInfer, ReductionState, ReductionRoute,
                      TriState, TriCSPGraphIsing, TRI_CHANNELS, TRI_N_CHANNELS, build_type_csp,
                      TYPE_NAMES)
from .bank_woven import AlphaStructComposerOrgan, forbid_record_csp


# ============================================================ the dispatching composer
class MultiFacultyComposerOrgan:
    """Dispatch by faculty/state_type. CSP → the certified reduced-product bank (reuses
    AlphaStructComposerOrgan, verifier-gated, α-gate from b0); GRAPH → the certified reachability closure;
    CROSS → the CSP→colours→active-edges→reachability flow. Returns a per-cell [B,N,K] survival the γ
    channel reads, plus the per-instance faculty actually dispatched (for tracing/eval)."""

    def __init__(self, dev="cpu", core_ckpt="runs/general_organ_full.pt", use_core=True, max_rounds=16,
                 ising_restarts=48, ising_steps=120):
        self.csp = AlphaStructComposerOrgan(dev=dev, core_ckpt=core_ckpt, use_core=use_core,
                                            max_rounds=max_rounds)
        self.graph = GraphReach()
        self.cross = CrossCSPGraph()
        # the optimization faculties (modest anneal budget — n<=8 hits the ground state cheaply on CPU)
        self.ising = IsingMaxcut(restarts=ising_restarts, steps=ising_steps)
        self.typ = TypeInfer()
        self.reduction = ReductionRoute(restarts=max(64, ising_restarts), steps=ising_steps)
        self.tri = TriCSPGraphIsing(restarts=ising_restarts, steps=ising_steps)

    def faculties(self):
        return list(LIVE_FACULTIES)

    @torch.no_grad()
    def dispatch(self, b0, vmask, theta, edge_logits, specs, K, J_logits=None, h_logits=None):
        """b0 [B,N,K] α's per-cell candidate logits (CSP α-gate); edge_logits [B,N,N] α's graph-head logits;
        J_logits [B,N,N]/h_logits [B,N] α's Ising-head couplings; `specs` a list of B dicts
        {faculty, facts, n, d, source, target} (α's EMITTED structure — NOT rec metadata). Returns
        (surv [B,N,K], dispatched_faculties, tri_surv [B,TRI_N_CHANNELS,N,K]). tri_surv holds the per-WIRING
        survivals of the 3-hop faculty (raw/color/color_reach) for the learned flow gates; zeros elsewhere."""
        B, N, _ = b0.shape
        alive = (torch.sigmoid(b0) >= theta).float() * vmask.unsqueeze(-1)
        alive_np = alive.cpu().numpy()
        out = np.zeros((B, N, K), dtype=np.float32)
        tri_out = np.zeros((B, TRI_N_CHANNELS, N, K), dtype=np.float32)
        dispatched = [None] * B
        csp_items, csp_slots = [], []
        for b in range(B):
            sp = specs[b]
            fac = sp["faculty"]; n = int(sp["n"]); d = min(int(sp["d"]), K)
            dispatched[b] = fac
            try:
                if fac == "csp":
                    csp_a = build_csp_from_struct(sp["facts"], n, d)
                    adom = tuple(frozenset(v for v in range(d) if alive_np[b, i, v] > 0.5)
                                 for i in range(n))
                    csp_items.append((csp_a, adom, None, ())); csp_slots.append((b, n))
                elif fac == "graph":
                    edges = decode_edges(edge_logits[b], vmask[b], n)
                    gs = self.graph.reduce(GraphReachState(n, edges, int(sp["source"]), int(sp["target"])))
                    out[b, :n, :] = self.graph.survival(gs, K)[:n, :]
                elif fac == "cross":
                    edges = decode_edges(edge_logits[b], vmask[b], n)
                    csp_a = build_csp_from_struct(sp["facts"], n, d)
                    cs = CrossState(csp_a, edges, int(sp["source"]), int(sp["target"]), "eq")
                    out[b, :n, :] = self.cross.survival(cs, K)[:n, :]
                elif fac == "ising":
                    Jb = J_logits[b, :n, :n].cpu().numpy()
                    Jb = 0.5 * (Jb + Jb.T); np.fill_diagonal(Jb, 0.0)
                    hb = h_logits[b, :n].cpu().numpy()
                    ist = IsingState(n, Jb, hb, int(sp["source"]), int(sp["target"]))
                    out[b, :n, :] = self.ising.survival(ist, K)[:n, :]
                elif fac == "type":
                    csp_t = build_type_csp(sp["facts"], n)
                    out[b, :n, :] = self.typ.survival(TypeState(csp_t, n, int(sp["target"])), K)[:n, :]
                elif fac == "reduction":
                    edges = decode_edges(edge_logits[b], vmask[b], n)
                    und = frozenset((min(u, v), max(u, v)) for (u, v) in edges if u != v)
                    rst = ReductionState(n, und, int(sp["source"]), int(sp["target"]))
                    out[b, :n, :] = self.reduction.survival(rst, K)[:n, :]
                elif fac == "tri":
                    edges = decode_edges(edge_logits[b], vmask[b], n)
                    csp_a = build_csp_from_struct(sp["facts"], n, 2)
                    tst = TriState(csp_a, edges, int(sp["source"]), int(sp["target"]))
                    for c, ch in enumerate(TRI_CHANNELS):
                        tri_out[b, c, :n, :] = self.tri.channel_survival(tst, K, ch)[:n, :]
                    out[b, :n, :] = tri_out[b, TRI_N_CHANNELS - 1, :n, :]   # full wiring = default
                else:
                    out[b] = alive_np[b]
            except Exception:
                out[b] = alive_np[b]                              # malformed struct ⇒ graceful α-only fallback
        if csp_items:
            composed_list, _ = self.csp.compose_batch(csp_items)
            for (b, n), composed in zip(csp_slots, composed_list):
                for i in range(n):
                    for v in composed.dom[i]:
                        if v < K:
                            out[b, i, v] = 1.0
        dev = b0.device
        return (torch.from_numpy(out).to(dev), dispatched, torch.from_numpy(tri_out).to(dev))


# ============================================================ the multi-faculty woven model
def _rich_from_set(surv_set, dvec):
    B, N, K = surv_set.shape
    card = surv_set.sum(-1)
    d = (torch.full((B, 1), float(K), device=surv_set.device) if dvec is None
         else dvec.to(surv_set.device).float().clamp_min(1.0).view(B, 1))
    feat = surv_set.new_zeros(B, N, K + 2)
    feat[..., :K] = surv_set
    feat[..., K] = (card / d).clamp(0.0, 1.0)
    feat[..., K + 1] = (1.0 - (card - 1).clamp_min(0.0) / (d - 1).clamp_min(1.0)).clamp(0.0, 1.0)
    return feat


class MultiFacultyWoven(nn.Module):
    """The live multi-faculty woven GLaDOS. MID hook captures host hidden; INJECT hook runs
    StructureRack(h) → {router, csp head, graph head, candidate head} → dispatch by faculty → γ scatter
    (zero-init gate ⇒ bitwise no-op@init). Trainable: LoRA + rack + candidate head + γ; the composer is
    the frozen/certified bank + the parameter-free graph/cross closures."""

    def __init__(self, peft_model, D, K, rack: StructureRack, composer: MultiFacultyComposerOrgan,
                 mid_layer, inject_layer, gamma_hidden=256, theta=0.5, rich=True, cand_hidden=384):
        super().__init__()
        from ..oracle_readout import _decoder_layers
        self.model = peft_model
        self.alpha = rack
        self.composer = composer
        din = rack.encoder.din
        self.cand = nn.Sequential(nn.Linear(din, cand_hidden), nn.GELU(), nn.Linear(cand_hidden, K))
        self.rich = rich
        self.Fin = K + 2 if rich else K
        self.gamma = OracleGamma(D, self.Fin, gamma_hidden)
        # the LEARNED FLOW GATES (zero-init, like γ's tanh gate): one per wiring channel of the 3-hop
        # faculty. tanh(0)=0 ⇒ the fabric starts DISCONNECTED (no flow contributes) and LEARNS which flows
        # to open from the answer-CE. The connectivity is the learned object (notes §3.2).
        self.flow_gate = nn.Parameter(torch.zeros(TRI_N_CHANNELS))
        self.D, self.K = D, K
        self.mid_layer, self.inject_layer = mid_layer, inject_layer
        self.theta = theta
        self._mention = self._attn = None
        self._specs = self._dvec = None
        self._inject = self._capture = False
        self._override = None
        self._tri_override = self._tri_mask = None
        self._route_by_model = False                          # eval: dispatch by the router's own argmax
        self._h_mid = None
        self._captured_surv = None
        self._captured_tri = None
        self._captured_tri_mask = None
        self._last = {}                                       # router_logits, pin, pair, edge, b0, vmask, faculties
        layers = _decoder_layers(peft_model)
        self._mid_handle = layers[mid_layer].register_forward_hook(self._mid_hook)
        self._inj_handle = layers[inject_layer].register_forward_hook(self._inj_hook)

    def _mid_hook(self, module, args, output):
        self._h_mid = output[0] if isinstance(output, tuple) else output
        return output

    def _compile(self):
        """StructureRack(mid hidden) → router logits + csp head (pin,pair) + graph head (edges) + cand b0."""
        h = self._h_mid.float()
        m = self._mention.to(h.device)
        denom = m.sum(-1, keepdim=True).clamp_min(1e-6)
        v_mean = torch.einsum("bnt,btd->bnd", m, h) / denom
        feat = self.alpha.featurize(v_mean, h, self._attn.to(h.device))
        router_logits = self.alpha.route(h, self._attn.to(h.device))      # [B, n_faculties]
        pin_l, pair_l = self.alpha.csp(feat)
        edge_l = self.alpha.heads["graph"](feat)                          # [B,N,N] graph-head edges
        J_l, h_l = self.alpha.heads["ising"](feat)                        # [B,N,N],[B,N] Ising couplings
        b0 = self.cand(feat)
        vmask = (m.sum(-1) > 0.5).float()
        self._last = {"router": router_logits, "pin": pin_l, "pair": pair_l,
                      "edge": edge_l, "J": J_l, "h": h_l, "b0": b0, "vmask": vmask}
        return router_logits, pin_l, pair_l, edge_l, b0, vmask

    def _build_specs(self, router_logits, pin_l, pair_l, edge_l, vmask):
        """Per-instance dispatch spec from the emitted heads. Faculty = the record's true faculty (teacher-
        forced) at train, or the ROUTER's argmax at eval (_route_by_model). decode the csp facts the chosen
        faculty needs (csp/cross/type/tri); graph/ising/reduction read edges/(J,h) directly off their heads."""
        B = pin_l.shape[0]
        specs = []
        route_pred = router_logits.argmax(-1).tolist()
        facs = []
        for b in range(B):
            base = self._specs[b]
            fac = LIVE_FACULTIES[route_pred[b]] if self._route_by_model else base["faculty"]
            facs.append(fac)
            n, d = int(base["n"]), int(base["d"])
            facts = []
            if fac in ("csp", "cross", "type", "tri"):
                dd = 2 if fac in ("type", "tri") else d
                facts = decode_structure(pin_l[b], pair_l[b], vmask[b], n, dd)
            specs.append({"faculty": fac, "facts": facts, "n": n, "d": d,
                          "source": base.get("source", 0), "target": base.get("target", base.get("query", 0))})
        self._last["faculties"] = facs
        return specs

    def _feat_for_inject(self, surv, tri, tri_mask, dvec):
        """Build the γ-input feature. Non-tri rows: rich(surv). TRI rows: the FLOW-GATED mixture
        Σ_c tanh(gate_c)·rich(channel_c) over the wiring channels (raw / +csp→graph / +graph→ising). The
        gates are zero-init ⇒ tri rows start at feat=0 (disconnected) and the answer-CE LEARNS which flows
        to open (the gates are the differentiable, learnable wiring; the channel survivals are constants)."""
        feat = _rich_from_set(surv, dvec) if self.rich else surv          # [B,N,Fin]
        if tri is not None and tri_mask is not None and bool(tri_mask.any()):
            gates = torch.tanh(self.flow_gate)                            # [C]
            mixed = feat.new_zeros(feat.shape)
            for c in range(tri.shape[1]):
                fc = _rich_from_set(tri[:, c], dvec) if self.rich else tri[:, c]
                mixed = mixed + gates[c] * fc
            mrow = tri_mask.to(feat.dtype).view(-1, 1, 1)
            feat = feat * (1.0 - mrow) + mixed * mrow
        return feat

    def _inj_hook(self, module, args, output):
        if not (self._inject or self._capture):
            return output
        hs = output[0] if isinstance(output, tuple) else output
        surv = self._override
        tri = self._tri_override
        if surv is None:
            router_logits, pin_l, pair_l, edge_l, b0, vmask = self._compile()
            specs = self._build_specs(router_logits, pin_l, pair_l, edge_l, vmask)
            with forbid_record_csp():
                surv, _disp, tri = self.composer.dispatch(b0, vmask, self.theta, edge_l, specs, self.K,
                                                          J_logits=self._last["J"], h_logits=self._last["h"])
            surv = surv * vmask.unsqueeze(-1)
            tri = tri * vmask.unsqueeze(1).unsqueeze(-1)
            self._captured_surv = surv.detach()
            self._captured_tri = tri.detach()
            surv = self._captured_surv
            tri = self._captured_tri
            facs = self._last.get("faculties", [])
            self._tri_mask = torch.tensor([f == "tri" for f in facs], device=surv.device)
            self._captured_tri_mask = self._tri_mask                # persists past the live() ctx restore
        if not self._inject:
            return output
        feat = self._feat_for_inject(surv, tri, self._tri_mask, self._dvec)
        delta = self.gamma.delta(feat.to(hs.device), self._mention.to(hs.device))
        hs = hs + (torch.tanh(self.gamma.alpha) * delta).to(hs.dtype)
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

    @contextlib.contextmanager
    def live(self, mention, attn, *, inject, capture=False, override=None, specs=None, dvec=None,
             route_by_model=False, tri_override=None, tri_mask=None):
        old = (self._mention, self._attn, self._inject, self._capture, self._override, self._specs,
               self._dvec, self._route_by_model, self._tri_override, self._tri_mask)
        self._mention, self._attn = mention, attn
        (self._inject, self._capture, self._override, self._specs, self._dvec, self._route_by_model) = (
            inject, capture, override, specs, dvec, route_by_model)
        self._tri_override, self._tri_mask = tri_override, tri_mask
        try:
            yield
        finally:
            (self._mention, self._attn, self._inject, self._capture, self._override, self._specs,
             self._dvec, self._route_by_model, self._tri_override, self._tri_mask) = old

    def logits(self, input_ids, attn):
        return self.model(input_ids=input_ids, attention_mask=attn).logits

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def remove_hooks(self):
        for attr in ("_mid_handle", "_inj_handle"):
            h = getattr(self, attr, None)
            if h is not None:
                h.remove(); setattr(self, attr, None)

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass
