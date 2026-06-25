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
    return {"input_ids": input_ids, "attn": attn, "mention": mention, "last_idx": last_idx,
            "vmask": vmask, "pins": pins, "pinned": pinned, "adj": adj, "query": query,
            "ans": ans, "abst": abst, "texts": texts}


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


# ===================================================================== Flamingo no-op demonstrator
class _GateNoOp(nn.Module):
    """Standalone zero-init tanh-gated in-stream adapter, used ONLY to confirm the Flamingo property:
    a gated cross-attn added to an OLMo layer is an EXACT no-op at init (tanh(0)=0), so installing the
    coupling does not damage the frozen pretrained model. Not used for the decision (an in-stream
    answer head lets OLMo bypass the deductor — see report)."""
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


# ===================================================================== the augmented model
class AugmentedOLMo(nn.Module):
    """Frozen OLMo host + learned WRITE (cell cross-attn) + learned differentiable DEDUCTOR + checked
    READ (answer = the deductor's narrowed query-cell marginals; abstain = deductor stats + gated
    lattice readback). The answer is FORCED through the deductor, so the host cannot bypass it and the
    deductor is load-bearing -> checked reasoning + size-agnostic abstention."""
    def __init__(self, olmo, tok, Nmax, K, T=10, dp=512, edge_dim=256, Dw=128,
                 write_head="pool", edge_window=7):
        super().__init__()
        self.olmo = olmo; self.tok = tok
        self.Nmax, self.K, self.T = Nmax, K, T
        self.write_head = write_head
        D = olmo.config.hidden_size; self.D = D
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

    @torch.no_grad()
    def host_encode(self, ba):
        out = self.olmo.model(input_ids=ba["input_ids"], attention_mask=ba["attn"])
        return out.last_hidden_state  # [B,T,D] bf16

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

    def read(self, alive, h, ba):
        B = alive.size(0)
        qa = alive[torch.arange(B, device=alive.device), ba["query"]]           # [B,K]
        qm = qa / qa.sum(-1, keepdim=True).clamp_min(1e-6)
        color_logits = torch.log(qm + 1e-6)                                     # answer FROM deductor
        peak = qm.max(-1).values
        ent = -(qm * (qm + 1e-9).log()).sum(-1)
        nalive = (qm > 0.1).float().sum(-1)
        stats = torch.stack([peak, ent, nalive], -1)                           # [B,3]
        # gated lattice readback for abstain (values = lattice only -> no bypass)
        marg = alive / alive.sum(-1, keepdim=True).clamp_min(1e-6)
        ws = self.ws_enc(marg)
        qoh = F.one_hot(ba["query"], self.Nmax).float()
        ws = self.ws_ln(ws + qoh.unsqueeze(-1) * self.ws_query[None, None, :])
        qvec = h[torch.arange(B, device=h.device), ba["last_idx"]].float().detach()
        rb = self.readback(qvec, ws, ba["vmask"])                              # gated; 0 at init
        abstain_logit = self.abstain_head(torch.cat([stats, rb], -1)).squeeze(-1)
        return color_logits, abstain_logit

    def forward(self, ba):
        h = self.host_encode(ba)
        cand, elog, E = self.write(h, ba)
        alive = self.deduce(cand, E, ba)
        color_logits, abstain_logit = self.read(alive, h, ba)
        return color_logits, abstain_logit, (cand, E, alive, elog)

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
    gate = _GateNoOp(olmo.config.hidden_size, Dw).to(dev)
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
