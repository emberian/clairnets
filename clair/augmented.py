"""Augmented-LLM prototype: a frozen open-checkpoint LLM (OLMo-2-1B) that learns to PROGRAM a
LEARNED, DIFFERENTIABLE lattice-deduction co-processor from NATURAL-LANGUAGE problems, and read
the narrowed state back — end-to-end.

This scales the toy `clair.induce` HybridReasoner (which got +30 OOD abstain-recall) from a
structured fact-encoder to a REAL language model reading English constraint problems.

Pipeline (all differentiable, co-trained):
  HOST   = OLMo-2-0425-1B, FROZEN. Reads the English problem.
  WRITE  = a cell-grounded read-out head: each graph node is grounded at its mention tokens in the
           prompt; from the host's hidden states it emits the CONSTRAINT PROGRAM — per-cell
           candidate-color masks (from "Node A is red") + pairwise exclusion weights (from "A and B
           must differ"). Supervised by the TRUE CSP (clair.csp / induce ground truth).
  DEDUCE = a LEARNED differentiable narrowing organ (ColorDeductor): monotone multiplicative meet,
           iterated T steps, gradients flow throughout. NOT an exact oracle — it co-trains, and we
           measure whether it reaches a checkable answer vs abstains, and whether that size-generalizes.
  READ   = the host RE-READS the prompt with ZERO-INIT tanh-GATED cross-attention adapters
           (Flamingo-style) that attend to the narrowed lattice workspace; a thin classification head
           on the answer position emits a color or the ABSTAIN token. At init the gates are 0 so the
           hybrid == base OLMo exactly (no damage to the pretrained model). Train ONLY the adapters +
           WRITE head + deductor + read head; OLMo stays frozen.

SOUNDNESS is checked cheaply AT THE OUTPUT (verify the answer against the constraints / score vs the
exact harness). COMPLETENESS + calibrated abstention + size-generalization are the measured variables.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import induce as I  # reuse gen_problem / _solutions ground truth + ColorDeductor

ABSTAIN = I.ABSTAIN
COLORS = ["red", "green", "blue", "yellow", "orange", "purple", "cyan", "pink"]
NODES = [chr(ord("A") + i) for i in range(26)]

# ---- HOST registry: the frozen open-checkpoint LLM. Everything downstream is handled GENERICALLY
# from the loaded model's config (hidden_size feeds the WRITE head + adapters; num_hidden_layers
# picks the gate-noop attach depth as a fraction; the tokenizer comes from the same id), so adding a
# host is just an entry here. Default stays olmo2-1b -> the validated 1B baseline path is unchanged.
HOSTS = {
    "olmo2-1b": "allenai/OLMo-2-0425-1B",
    "olmo3-7b": "allenai/Olmo-3-1025-7B",
}
MODEL_ID = HOSTS["olmo2-1b"]


def resolve_host(name):
    """Map a --host config name (e.g. 'olmo3-7b') to a HF model id. A raw HF id passes through."""
    return HOSTS.get(name, name)


# ===================================================================== text rendering
def render_problem(p, colors=COLORS):
    """Render a coloring-with-abstention problem (from induce.gen_problem) as English. Returns
    (text, spans) where spans[node_id] = list of (char_start, char_end) of that node's letter mentions.
    Facts: (0, v, c) = IS(v, color c) ; (1, i, j) = DIFF(i, j)."""
    spans = {}
    parts = []
    cur = 0

    def emit(s):
        nonlocal cur
        parts.append(s)
        cur += len(s)

    def emit_node(v):
        nonlocal cur
        emit("node ")
        st = cur
        emit(NODES[v])
        spans.setdefault(v, []).append((st, cur))

    for (t, a, b) in p["facts"]:
        if t == 0:  # IS(a, color b)
            emit_node(a)
            emit(" is " + colors[b] + ". ")
        else:  # DIFF(a, b)
            emit_node(a)
            emit(" and ")
            emit_node(b)
            emit(" must be different colors. ")
    emit("Question: what color is ")
    emit_node(p["query"])
    emit("? Answer:")
    text = "".join(parts)
    # Capitalize sentence starts: our node-emit lowercases "node"; fix leading "node" of each sentence.
    return text, spans


def _fix_caps(text):
    # capitalize the first letter of each sentence (after ". " or at start)
    out = []
    cap = True
    for ch in text:
        if cap and ch.isalpha():
            out.append(ch.upper()); cap = False
        else:
            out.append(ch)
        if ch in ".?":
            cap = True
    return "".join(out)


def surviving_set(p, K):
    """The EXACT surviving set for the query cell q: {v : SOME full solution consistent with the
    givens has q=v}. This is the q-th component of clair.csp.exact_dedP (per-cell projection of the
    solution set), computed here directly from the problem's facts via the induce backtracking solver
    so it is exact for BOTH templated (induce.gen_problem) and diverse (curriculum coloring) records
    — both reach here as (0,v,c)=IS / (1,i,j)=DIFF facts. Empty set == unsatisfiable (bottom)."""
    pins, edges = {}, set()
    for (t, a, b) in p["facts"]:
        if t == 0:
            pins[a] = b
        else:
            edges.add((a, b))
    sols = I._solutions(p["n"], K, pins, edges)
    if not sols:
        return set()                                   # unsat -> the sound answer is the empty set
    return {s[p["query"]] for s in sols}


def build_batch(probs, tok, Nmax, K, colors, device):
    """Tokenize a batch of rendered problems and build cell-mention masks + the ground-truth program
    and answer targets. Returns a dict of tensors.

    A problem may carry a PRE-RENDERED `text` + `mentions` (diverse curriculum records, where the
    natural language and entity char-spans come from Bedrock + the curriculum loader). In that case
    we use them verbatim; otherwise we render the canonical template from `facts`."""
    texts, all_spans = [], []
    for p in probs:
        if p.get("text") is not None and p.get("mentions") is not None:
            # diverse curriculum record: text + char-span mentions already provided
            texts.append(p["text"])
            all_spans.append({int(k): [tuple(s) for s in v] for k, v in p["mentions"].items()})
            continue
        t, spans = render_problem(p, colors)
        # capitalization shouldn't move char offsets (same length), apply after computing spans
        texts.append(_fix_caps(t))
        all_spans.append(spans)
    enc = tok(texts, return_offsets_mapping=True, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    offsets = enc["offset_mapping"]  # [B,T,2] on cpu
    B, T = input_ids.shape
    mention = torch.zeros(B, Nmax, T, device=device)
    for b, spans in enumerate(all_spans):
        offs = offsets[b].tolist()
        for v, sps in spans.items():
            for (cs, ce) in sps:
                for ti, (a, c) in enumerate(offs):
                    if a == c:  # special/pad token
                        continue
                    if a < ce and c > cs:  # overlap with node-letter span
                        mention[b, v, ti] = 1.0
    last_idx = attn.sum(1) - 1  # answer position = last real token (right padding)

    vmask = torch.zeros(B, Nmax, device=device)
    pins = torch.zeros(B, Nmax, K, device=device)
    pinned = torch.zeros(B, Nmax, device=device)
    adj = torch.zeros(B, Nmax, Nmax, device=device)
    query = torch.zeros(B, dtype=torch.long, device=device)
    ans = torch.zeros(B, dtype=torch.long, device=device)
    abst = torch.zeros(B, device=device)
    surv = torch.zeros(B, K, device=device)          # SET-VALUED target: 1 if v survives at query cell
    for b, p in enumerate(probs):
        vmask[b, : p["n"]] = 1.0
        for (t, a, c) in p["facts"]:
            if t == 0:
                pins[b, a, c] = 1.0; pinned[b, a] = 1.0
            else:
                adj[b, a, c] = 1.0; adj[b, c, a] = 1.0
        query[b] = p["query"]
        abst[b] = 0.0 if p["determined"] else 1.0
        ans[b] = p["answer"] if p["determined"] else 0
        for v in surviving_set(p, K):
            surv[b, v] = 1.0
    return {"input_ids": input_ids, "attn": attn, "mention": mention, "last_idx": last_idx,
            "vmask": vmask, "pins": pins, "pinned": pinned, "adj": adj, "query": query,
            "ans": ans, "abst": abst, "surv": surv, "texts": texts}


# ===================================================================== WRITE: cell cross-attn pooler
class CellReader(nn.Module):
    """Per-cell cross-attention over the WHOLE prompt. Each graph node, identified by the mean of its
    mention-token hidden states, issues a query that attends over all host hidden states to gather
    that node's facts (its pin color, its exclusion partners). This is the toy compile() head, but
    reading a frozen OLMo instead of a small fact-encoder."""
    def __init__(self, D, dp=512, heads=8):
        super().__init__()
        self.h, self.dp, self.dk = heads, dp, dp // heads
        self.wq = nn.Linear(D, dp)
        self.wk = nn.Linear(D, dp)
        self.wv = nn.Linear(D, dp)
        self.qln = nn.LayerNorm(D)

    def forward(self, v_mean, h, attn_mask):
        # v_mean [B,N,D] (cell identity), h [B,T,D] (host states), attn_mask [B,T]
        B, N, _ = v_mean.shape
        T = h.size(1)
        q = self.wq(self.qln(v_mean)).view(B, N, self.h, self.dk).transpose(1, 2)
        k = self.wk(h).view(B, T, self.h, self.dk).transpose(1, 2)
        vv = self.wv(h).view(B, T, self.h, self.dk).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dk)
        att = att.masked_fill(attn_mask[:, None, None, :] < 0.5, -1e9)
        att = att.softmax(-1)
        ctx = torch.matmul(att, vv).transpose(1, 2).reshape(B, N, self.dp)
        return ctx


# ===================================================================== WRITE: pair-grounded edge reader
class PairGroundedEdgeReader(nn.Module):
    """LOCALLY-GROUNDED DIFF-edge extractor (the fix for the extraction wall).

    The pool baseline scores every (i,j) pair BILINEARLY from two INDEPENDENT per-cell summaries
    (edge_a/edge_b on each cell's globally-pooled context). That summary throws away WHICH clause
    couples i to j, so as N grows the bilinear scorer can no longer bind the right DIFF-edges
    (edge-F1 63->48, det-acc 84->26 as N 5->11).

    Here each edge decision (i,j) is read from the host hidden states WHERE i AND j CO-OCCUR. For
    every ordered pair we form a SYMMETRIC query from both cells' mention-mean states and
    cross-attend over the prompt, with the attention BIASED toward the union of i's and j's mention
    neighborhoods (the local fact-clause spanning both mentions). We then classify edge present/
    absent from that LOCAL context plus a hard co-occurrence feature (max overlap of the two
    proximity fields). Grounding is what makes it scale: a DIFF fact is ONE local clause -> ONE
    edge, independent of N; pairs that never co-occur locally get no evidence and default off.
    """
    def __init__(self, D, dp=512, heads=8, window=7):
        super().__init__()
        self.h, self.dp, self.dk = heads, dp, dp // heads
        self.window = window | 1                      # force odd so padding keeps T fixed
        self.qln = nn.LayerNorm(2 * D)
        self.wq = nn.Linear(2 * D, dp)
        self.wk = nn.Linear(D, dp)
        self.wv = nn.Linear(D, dp)
        self.beta = nn.Parameter(torch.tensor(2.0))   # locality-bias strength (>=0 via softplus)
        self.cls = nn.Sequential(nn.Linear(dp + 1, dp), nn.GELU(), nn.Linear(dp, 1))

    def _prox(self, mention):
        """mention [B,N,T] binary -> proximity field [B,N,T] in {0,1}: 1 within +-window//2 tokens
        of any mention of that cell (a dilation along the token axis)."""
        B, N, T = mention.shape
        w = self.window
        p = F.max_pool1d(mention.reshape(B * N, 1, T), kernel_size=w, stride=1, padding=w // 2)
        return p.reshape(B, N, T)

    def forward(self, v_mean, h, mention, attn_mask):
        # v_mean [B,N,D] cell identity, h [B,T,D] host states, mention [B,N,T], attn_mask [B,T]
        B, N, D = v_mean.shape
        T = h.size(1)
        prox = self._prox(mention)                                          # [B,N,T]
        a = v_mean[:, :, None, :].expand(B, N, N, D)
        b = v_mean[:, None, :, :].expand(B, N, N, D)
        pf = torch.cat([a + b, a * b], -1)                                  # symmetric pair feat [B,N,N,2D]
        q = self.wq(self.qln(pf)).view(B, N * N, self.h, self.dk).transpose(1, 2)  # [B,h,N*N,dk]
        k = self.wk(h).view(B, T, self.h, self.dk).transpose(1, 2)
        vv = self.wv(h).view(B, T, self.h, self.dk).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dk)     # [B,h,N*N,T]
        # locality bias: attend near EITHER mention so the clause spanning both is read
        union = (prox[:, :, None, :] + prox[:, None, :, :]).reshape(B, N * N, T)   # [B,N*N,T]
        att = att + F.softplus(self.beta) * union[:, None, :, :]
        att = att.masked_fill(attn_mask[:, None, None, :] < 0.5, -1e9)
        att = att.softmax(-1)
        ctx = torch.matmul(att, vv).transpose(1, 2).reshape(B, N * N, self.dp)    # [B,N*N,dp]
        # hard co-occurrence feature: do i and j share a local neighborhood at all?
        cooc = (prox[:, :, None, :] * prox[:, None, :, :]).max(-1).values.reshape(B, N * N, 1)
        elog = self.cls(torch.cat([ctx, cooc], -1)).reshape(B, N, N)
        return 0.5 * (elog + elog.transpose(1, 2))                          # symmetric raw edge logits


# ===================================================================== gated lattice readback (abstain)
class ReadBack(nn.Module):
    """Zero-init tanh-gated cross-attention: a host-derived query reads the NARROWED lattice workspace
    (values = lattice marginals only -> the host cannot bypass the deductor). Output feeds the abstain
    decision; alpha=0 at init so abstention starts as a pure function of the certified deductor state."""
    def __init__(self, Dq, Dw, dp=128, heads=4):
        super().__init__()
        self.h, self.dp, self.dk = heads, dp, dp // heads
        self.wq = nn.Linear(Dq, dp)
        self.wk = nn.Linear(Dw, dp)
        self.wv = nn.Linear(Dw, dp)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, qvec, ws, ws_mask):
        B = qvec.size(0); M = ws.size(1)
        q = self.wq(qvec).view(B, 1, self.h, self.dk).transpose(1, 2)
        k = self.wk(ws).view(B, M, self.h, self.dk).transpose(1, 2)
        v = self.wv(ws).view(B, M, self.h, self.dk).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dk)
        att = att.masked_fill(ws_mask[:, None, None, :] < 0.5, -1e9)
        att = att.softmax(-1)
        ctx = torch.matmul(att, v).transpose(1, 2).reshape(B, self.dp)
        return torch.tanh(self.alpha) * ctx


# ===================================================================== SET-VALUED lattice readout
class SetReadout(nn.Module):
    """Per-candidate-value KEEP/DROP logits for the QUERY cell, read from the deductor's narrowed
    marginal. The reported answer is the SET {v : keep}. A singleton set = a determinate answer; a
    larger set = an honest PARTIAL answer; the empty set = unsatisfiable. This replaces the brittle
    answer-or-abstain head: there is NO separate abstain token, and a single extraction error widens
    the set GENTLY (one more value survives) instead of flipping a binary answer.

    Each value's keep decision is a SHARED tiny MLP on that value's own narrowed mass + the cell's
    determinacy context (so it is permutation-equivariant over values and generalizes across K). The
    inductive bias is monotone: more surviving mass -> more likely kept. Soundness (never drop a true
    survivor) is enforced by the asymmetric training loss, not hard-coded here."""
    def __init__(self, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(5, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, qa):
        # qa [B,K] = narrowed alive-mass at the query cell (in (0,1]); larger = more viable
        qa = qa.clamp_min(1e-6)
        m = qa / qa.sum(-1, keepdim=True).clamp_min(1e-6)            # normalized marginal
        peak = m.max(-1, keepdim=True).values.expand_as(m)          # cell determinacy context
        feats = torch.stack([qa.clamp(0, 1), m, qa.log(), m.log(), peak], -1)  # [B,K,5]
        return self.mlp(feats).squeeze(-1)                           # [B,K] keep/drop logits


# ===================================================================== woven host write-back
class WovenWriteBack(nn.Module):
    """Zero-init tanh-gated in-stream workspace adapter.

    The terminal augmented model reads the lattice after OLMo is done. This module gives the lattice
    a causal path back into the host: hidden tokens query the narrowed lattice workspace, and the
    resulting update is added to the residual stream. Since alpha starts at 0, installing the adapter
    is an exact no-op at initialization.
    """
    def __init__(self, D, Dw, heads=8, dh=512):
        super().__init__()
        self.h, self.dh, self.dk = heads, dh, dh // heads
        self.q = nn.Linear(D, dh, bias=False); self.k = nn.Linear(Dw, dh, bias=False)
        self.v = nn.Linear(Dw, dh, bias=False); self.o = nn.Linear(dh, D, bias=False)
        self.ln = nn.LayerNorm(D); self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, hs, ws, ws_mask):
        dt = hs.dtype; hs = hs.float(); ws = ws.float()
        B, T, _ = hs.shape; M = ws.size(1)
        q = self.q(self.ln(hs)).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.k(ws).view(B, M, self.h, self.dk).transpose(1, 2)
        v = self.v(ws).view(B, M, self.h, self.dk).transpose(1, 2)
        att = (torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dk))
        att = att.masked_fill(ws_mask[:, None, None, :] < 0.5, -1e9).softmax(-1)
        out = torch.matmul(att, v).transpose(1, 2).reshape(B, T, self.dh)
        return (torch.tanh(self.alpha) * self.o(out)).to(dt)


def _parse_layer_spec(spec, n_layers):
    """Parse comma-separated layer indices. Negative indices are relative to the end."""
    if spec is None or spec == "":
        return ()
    if isinstance(spec, (list, tuple)):
        vals = spec
    else:
        vals = [x.strip() for x in str(spec).split(",") if x.strip()]
    out = []
    for x in vals:
        i = int(x)
        if i < 0:
            i = n_layers + i
        if not 0 <= i < n_layers:
            raise ValueError(f"woven layer {x!r} resolves to {i}, outside [0,{n_layers})")
        if i not in out:
            out.append(i)
    return tuple(out)


# ===================================================================== the augmented model
class AugmentedOLMo(nn.Module):
    """Frozen OLMo host + learned WRITE (cell cross-attn) + learned differentiable DEDUCTOR + checked
    READ (answer = the deductor's narrowed query-cell marginals; abstain = deductor stats + gated
    lattice readback). The answer is FORCED through the deductor, so the host cannot bypass it and the
    deductor is load-bearing -> checked reasoning + size-agnostic abstention."""
    def __init__(self, olmo, tok, Nmax, K, T=10, dp=512, edge_dim=256, Dw=128,
                 write_head="pool", edge_window=7, readout="single",
                 read_source="terminal", woven_layers=(), woven_heads=8, woven_dh=512):
        super().__init__()
        self.olmo = olmo; self.tok = tok
        self.Nmax, self.K, self.T = Nmax, K, T
        self.write_head = write_head
        self.readout = readout
        self.read_source = read_source
        D = olmo.config.hidden_size; self.D = D
        if read_source not in {"terminal", "woven", "fused"}:
            raise ValueError("read_source must be terminal, woven, or fused")
        for p in self.olmo.parameters():
            p.requires_grad_(False)
        # WRITE: per-cell pooler (pins always; edges in pool mode)
        self.reader = CellReader(D, dp)
        self.cand_head = nn.Sequential(nn.Linear(dp, dp), nn.GELU(), nn.Linear(dp, K))
        self.edge_bias = nn.Parameter(torch.tensor(-2.0))  # sparse-edge prior (shared by both heads)
        if write_head == "grounded":
            # LOCALLY-GROUNDED edge extractor: read each (i,j) edge from where i & j co-occur
            self.edge_reader = PairGroundedEdgeReader(D, dp, window=edge_window)
        else:                                              # pool baseline: bilinear on per-cell ctx
            self.edge_a = nn.Linear(dp, edge_dim)
            self.edge_b = nn.Linear(dp, edge_dim)
        # DEDUCE (learned differentiable organ; co-trains)
        self.ded = I.ColorDeductor(T)
        # workspace + gated readback (abstain)
        self.ws_enc = nn.Linear(K, Dw); self.ws_query = nn.Parameter(torch.randn(Dw) * 0.02)
        self.ws_ln = nn.LayerNorm(Dw)
        self.readback = ReadBack(D, Dw)
        self.abstain_head = nn.Sequential(nn.Linear(3 + self.readback.dp, 64), nn.GELU(), nn.Linear(64, 1))
        # SET-VALUED readout (config-gated): per-value keep/drop on the query cell's narrowed marginal.
        # Only built in setvalued mode so the single-answer baseline is byte-for-byte unchanged.
        self.set_readout = SetReadout() if readout == "setvalued" else None
        # WOVEN readout path: after the lattice is built, inject it into selected host layers and read
        # the modified final hidden state. `terminal` keeps the old path exactly; `woven` replaces the
        # terminal decision; `fused` adds the woven logits to the terminal logits.
        self.woven_layers = _parse_layer_spec(woven_layers, olmo.config.num_hidden_layers)
        self.woven = nn.ModuleDict({
            str(i): WovenWriteBack(D, Dw, heads=woven_heads, dh=woven_dh) for i in self.woven_layers
        })
        self.woven_single = nn.Sequential(nn.Linear(D + 3, 128), nn.GELU(), nn.Linear(128, K + 1))
        self.woven_keep = nn.Sequential(nn.Linear(D + 3, 128), nn.GELU(), nn.Linear(128, K))

    @torch.no_grad()
    def host_encode(self, ba):
        out = self.olmo.model(input_ids=ba["input_ids"], attention_mask=ba["attn"])
        return out.last_hidden_state  # [B,T,D] bf16

    def host_encode_woven(self, ba, ws, ws_mask):
        """Run a second host pass with lattice workspace adapters installed.

        OLMo parameters stay frozen. The pass is intentionally not wrapped in no_grad: gradients must
        flow through the woven adapters and the frozen layers after each injection site.
        """
        handles = []

        def make_hook(layer_id):
            gate = self.woven[str(layer_id)]

            def hook(module, args, output):
                hs = output[0] if isinstance(output, tuple) else output
                hs = hs + gate(hs, ws, ws_mask)
                return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

            return hook

        try:
            for layer_id in self.woven_layers:
                handles.append(self.olmo.model.layers[layer_id].register_forward_hook(make_hook(layer_id)))
            out = self.olmo.model(input_ids=ba["input_ids"], attention_mask=ba["attn"])
            return out.last_hidden_state
        finally:
            for h in handles:
                h.remove()

    def write(self, h, ba):
        m = ba["mention"]
        hd = h.float().detach()
        denom = m.sum(-1, keepdim=True).clamp_min(1e-6)
        v_mean = (torch.einsum("bnt,btd->bnd", m, hd) / denom).detach()         # cell identity [B,N,D]
        ctx = self.reader(v_mean, hd, ba["attn"])                              # [B,N,dp]
        cand = self.cand_head(ctx)                                              # pin logits [B,N,K] (per-cell)
        if self.write_head == "grounded":
            elog = self.edge_reader(v_mean, hd, m, ba["attn"])                  # locally-grounded raw edge logits
        else:
            ea, eb = self.edge_a(ctx), self.edge_b(ctx)                         # pool: bilinear on per-cell ctx
            elog = torch.einsum("bid,bjd->bij", ea, eb) / ea.size(-1) ** 0.5
            elog = 0.5 * (elog + elog.transpose(1, 2))
        elog = elog + self.edge_bias
        vm = ba["vmask"]; pair = vm[:, :, None] * vm[:, None, :]
        elog = elog.masked_fill(pair < 0.5, -1e4)
        E = torch.sigmoid(elog) * pair
        return cand, elog, E

    def deduce(self, cand, E, ba):
        vm = ba["vmask"]
        alive0 = torch.sigmoid(cand) * vm.unsqueeze(-1) + 1e-3
        return self.ded(alive0, E, vm)                                          # [B,N,K]

    def workspace(self, alive, ba):
        marg = alive / alive.sum(-1, keepdim=True).clamp_min(1e-6)
        ws = self.ws_enc(marg)
        qoh = F.one_hot(ba["query"], self.Nmax).float()
        ws = self.ws_ln(ws + qoh.unsqueeze(-1) * self.ws_query[None, None, :])
        return ws, ba["vmask"]

    def read(self, alive, h, ba, h_woven=None):
        B = alive.size(0)
        qa = alive[torch.arange(B, device=alive.device), ba["query"]]           # [B,K]
        term_keep = self.set_readout(qa) if self.set_readout is not None else None
        qm = qa / qa.sum(-1, keepdim=True).clamp_min(1e-6)
        term_color = torch.log(qm + 1e-6)                                       # answer FROM deductor
        peak = qm.max(-1).values
        ent = -(qm * (qm + 1e-9).log()).sum(-1)
        nalive = (qm > 0.1).float().sum(-1)
        stats = torch.stack([peak, ent, nalive], -1)                           # [B,3]
        # gated lattice readback for abstain (values = lattice only -> no bypass)
        ws, ws_mask = self.workspace(alive, ba)
        qvec = h[torch.arange(B, device=h.device), ba["last_idx"]].float().detach()
        rb = self.readback(qvec, ws, ws_mask)                                  # gated; 0 at init
        term_abstain = self.abstain_head(torch.cat([stats, rb], -1)).squeeze(-1)

        color_logits, abstain_logit, keep_logits = term_color, term_abstain, term_keep
        if h_woven is not None:
            wvec = h_woven[torch.arange(B, device=h_woven.device), ba["last_idx"]].float()
            wfeat = torch.cat([wvec, stats], -1)
            wsingle = self.woven_single(wfeat)
            wcolor, wabstain = wsingle[:, :self.K], wsingle[:, self.K]
            wkeep = self.woven_keep(wfeat)
            if self.read_source == "woven":
                color_logits, abstain_logit = wcolor, wabstain
                keep_logits = wkeep if self.set_readout is not None else keep_logits
            elif self.read_source == "fused":
                color_logits, abstain_logit = term_color + wcolor, term_abstain + wabstain
                keep_logits = (term_keep + wkeep) if self.set_readout is not None else keep_logits
        return color_logits, abstain_logit, keep_logits

    def forward(self, ba):
        h = self.host_encode(ba)
        cand, elog, E = self.write(h, ba)
        alive = self.deduce(cand, E, ba)
        h_woven = None
        if self.read_source in {"woven", "fused"}:
            ws, ws_mask = self.workspace(alive, ba)
            h_woven = self.host_encode_woven(ba, ws, ws_mask)
        color_logits, abstain_logit, keep_logits = self.read(alive, h, ba, h_woven=h_woven)
        return color_logits, abstain_logit, (cand, E, alive, elog, keep_logits)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


@torch.no_grad()
def verify_flamingo_noop(olmo, tok, dev, layer=None, Dw=128):
    """Confirm the Flamingo coupling is an exact no-op at init: install a zero-init gated adapter on an
    OLMo layer, feed a random workspace, and check the logits equal base OLMo's. Returns max|diff|.
    `layer` defaults to ~1/3 depth, computed from the host's num_hidden_layers so it is valid for any
    host size (OLMo-2-1B has 16 layers, OLMo-3-7B has 32)."""
    nL = olmo.config.num_hidden_layers
    if layer is None:
        layer = nL // 3
    layer = min(layer, nL - 1)
    text = "Node A is red. Node A and node B must be different colors. Question: what color is node B? Answer:"
    ids = tok(text, return_tensors="pt").to(dev)
    base = olmo(input_ids=ids["input_ids"]).logits.float()
    gate = WovenWriteBack(olmo.config.hidden_size, Dw).to(dev)
    ws = torch.randn(1, 6, Dw, device=dev); wm = torch.ones(1, 6, device=dev)
    def hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        hs = hs + gate(hs, ws, wm)
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
    hd = olmo.model.layers[layer].register_forward_hook(hook)
    gated = olmo(input_ids=ids["input_ids"]).logits.float()
    hd.remove()
    return float((base - gated).abs().max())


def n_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
