"""GENERATIVE hybrid-deductive LLM: OLMo emits its answer through its OWN LM head, with the
lattice-deduction organ WOVEN into the late layers that drive generation, and OLMo LoRA-retrained
to lean on the organ.

This is the experiment that actually tests the "hybrid deductive LLM" claim. Everything in
`clair.augmented` reads the organ with a SEPARATE classification head -- which never tests whether
the organ makes the LANGUAGE MODEL reason better. Here:

  * the prompt is the English constraint problem ending in "Answer:" (clair.augmented.render_problem);
  * OLMo GENERATES the answer as TEXT via lm_head -- it must emit a color word ("red") or
    "cannot be determined". The supervised target is the answer string's tokens; loss is plain LM
    cross-entropy on the ANSWER span only (prompt masked). NO classification head decides anything.
  * the organ (grounded WRITE head -> learned ColorDeductor -> narrowed lattice workspace) is built
    from the PROMPT, then injected into chosen LATE OLMo layers via zero-init gated cross-attention
    (WovenWriteBack). The certified deduction shapes the final hidden states that feed lm_head, so it
    is in the CAUSAL PATH to the generated answer tokens. Gate=0 at init -> exact no-op (logits
    bit-identical to base OLMo).
  * LoRA (peft) on OLMo attn+mlp adapts the frozen host to generate the answer + use the organ.
    LoRA + organ (WRITE head, deductor, woven adapters) train jointly; base OLMo frozen.

The decisive measurements live in run_gen.py: generative accuracy vs base OLMo (in-dist + OOD), and
THE CAUSAL ABLATION -- generate twice, once with the organ gate live, once forced to 0; if accuracy
drops when the organ is zeroed, the LM's token selection causally depends on the organ.
"""
from __future__ import annotations

import math
import contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import augmented as A
from . import induce as I

COLORS = A.COLORS
NODES = A.NODES
ABSTAIN_STR = "cannot be determined"


# ===================================================================== the LATCH
class GatedVocabLatch(nn.Module):
    """Latch the deductor's verdict into OLMo's OWN token-embedding space.

    The woven cross-attn must learn, from scratch, to decode an abstract lattice slot into a vocab
    token -- which the baseline run showed is unstable (the LM thrashes and falls into always-abstain
    even though the organ KNOWS the answer). This module hands the answer to the LM pre-translated:
    the query cell's narrowed marginal selects a CERTAINTY-WEIGHTED blend of the color WORDS' input
    embeddings (anchor = qmarg @ color_embeds), and that vector is gated into the residual at the late
    layers. When the deductor is confident the anchor ~= the embedding of the right color word, so the
    residual is nudged straight toward that token's direction -- a soft, differentiable latch.

    It stays LEARNED, not hard-wired: the projection (init identity), the gate (init 0 -> exact no-op
    at init), and whether the LM uses the channel are all trained by the generative LM loss. Soundness
    still rests entirely on the organ producing the right marginal; this only eases the readout.
    """
    def __init__(self, D):
        super().__init__()
        self.ln = nn.LayerNorm(D)
        self.proj = nn.Linear(D, D)
        nn.init.eye_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hs, anchor):
        # hs [B,T,D] (dtype of host), anchor [B,D] in embedding space
        dt = hs.dtype
        a = self.proj(self.ln(anchor.float()))
        return (torch.tanh(self.gate) * a[:, None, :]).to(dt)


# ===================================================================== peft / module navigation
def _unwrap_causal(m):
    """Return the underlying *ForCausalLM whether or not `m` is a peft PeftModel wrapper."""
    return m.base_model.model if hasattr(m, "base_model") else m


def decoder_layers(m):
    """The decoder layer ModuleList (for forward hooks), peft-wrapping-agnostic."""
    return _unwrap_causal(m).model.layers


def trunk(m):
    """The transformer trunk (returns last_hidden_state), LoRA active if present."""
    return _unwrap_causal(m).model


# ===================================================================== answer strings
def answer_string(p, colors=COLORS):
    """The supervised generation target (no leading space; caller adds the separator)."""
    if not p.get("determined", p.get("answer", -1) != I.ABSTAIN):
        return ABSTAIN_STR
    name = p.get("answer_name")
    return name if name else colors[p["answer"]]


def parse_answer(text, colors=COLORS, K=3):
    """Map a generated continuation to a prediction:
        None  -> abstain ("cannot" / "determine" / "unknown" present)
        int   -> color id of the first color word found
        -2    -> unparseable (counts as wrong)."""
    t = text.lower()
    for kw in ("cannot", "can not", "determine", "undetermined", "unknown", "depends", "any color"):
        if kw in t:
            return None
    for ci, c in enumerate(colors[:K]):
        if c in t:
            return ci
    return -2


# ===================================================================== batch construction
def _build_prompt_tensors(probs, tok, Nmax, K, colors, device):
    """Tokenize the PROMPTS only (the constraint problem ending in 'Answer:'), with cell-mention
    masks + the ground-truth program (pins/edges) for the WRITE head + organ. Returns (tensors,
    prompts) where prompts is the list of prompt strings (for the full generative tokenization)."""
    prompts, all_spans = [], []
    for p in probs:
        if p.get("text") is not None and p.get("mentions") is not None:
            prompts.append(p["text"])
            all_spans.append({int(k): [tuple(s) for s in v] for k, v in p["mentions"].items()})
        else:
            t, spans = A.render_problem(p, colors)
            prompts.append(A._fix_caps(t))
            all_spans.append(spans)
    enc = tok(prompts, return_offsets_mapping=True, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    offsets = enc["offset_mapping"]
    B, T = input_ids.shape
    mention = torch.zeros(B, Nmax, T, device=device)
    for b, spans in enumerate(all_spans):
        offs = offsets[b].tolist()
        for v, sps in spans.items():
            for (cs, ce) in sps:
                for ti, (a, c) in enumerate(offs):
                    if a == c:
                        continue
                    if a < ce and c > cs:
                        mention[b, v, ti] = 1.0
    last_idx = attn.sum(1) - 1
    vmask = torch.zeros(B, Nmax, device=device)
    pins = torch.zeros(B, Nmax, K, device=device)
    pinned = torch.zeros(B, Nmax, device=device)
    adj = torch.zeros(B, Nmax, Nmax, device=device)
    query = torch.zeros(B, dtype=torch.long, device=device)
    for b, p in enumerate(probs):
        vmask[b, : p["n"]] = 1.0
        for (t, a, c) in p["facts"]:
            if t == 0:
                pins[b, a, c] = 1.0; pinned[b, a] = 1.0
            else:
                adj[b, a, c] = 1.0; adj[b, c, a] = 1.0
        query[b] = p["query"]
    tensors = {"input_ids": input_ids, "attn": attn, "mention": mention, "last_idx": last_idx,
               "vmask": vmask, "pins": pins, "pinned": pinned, "adj": adj, "query": query}
    return tensors, prompts


def build_gen_batch(probs, tok, Nmax, K, colors, device):
    """Organ tensors (prompt-only) + the generative-LM tensors (prompt + ' ' + answer with the
    answer-span CE mask) + exact labels."""
    org, prompts = _build_prompt_tensors(probs, tok, Nmax, K, colors, device)
    answers = [answer_string(p, colors) for p in probs]
    plens = [len(pt) for pt in prompts]
    fulls = [pt + " " + an for pt, an in zip(prompts, answers)]
    fenc = tok(fulls, return_offsets_mapping=True, padding=True, return_tensors="pt")
    f_ids = fenc["input_ids"].to(device)
    f_attn = fenc["attention_mask"].to(device)
    f_off = fenc["offset_mapping"]
    B, T = f_ids.shape
    labels = f_ids.clone()
    ans_tok = torch.zeros(B, T, device=device)
    for b in range(B):
        offs = f_off[b].tolist()
        for ti, (a, c) in enumerate(offs):
            is_ans = (a != c) and (a >= plens[b]) and (f_attn[b, ti] > 0.5)
            ans_tok[b, ti] = 1.0 if is_ans else 0.0
    labels[ans_tok < 0.5] = -100
    # exact labels for scoring
    determined = torch.tensor([1.0 if p["determined"] else 0.0 for p in probs], device=device)
    ans_id = torch.tensor([p["answer"] if p["determined"] else -1 for p in probs],
                          dtype=torch.long, device=device)
    surv = torch.zeros(B, K, device=device)
    for b, p in enumerate(probs):
        for v in A.surviving_set(p, K):
            surv[b, v] = 1.0
    return {"org": org, "f_ids": f_ids, "f_attn": f_attn, "labels": labels, "ans_tok": ans_tok,
            "determined": determined, "ans_id": ans_id, "surv": surv,
            "prompts": prompts, "answers": answers}


# ===================================================================== the generative model
class GenAugmentedOLMo(nn.Module):
    """LoRA-adapted OLMo whose lm_head generates the answer, with the lattice organ woven into its
    late layers. Reuses clair.augmented pieces: PairGroundedEdgeReader (WRITE), ColorDeductor
    (DEDUCE), WovenWriteBack (the zero-init in-stream injection)."""

    def __init__(self, olmo, tok, cfg, Nmax, K, T=10, dp=512, edge_dim=256, Dw=128,
                 write_head="grounded", edge_window=7, woven_layers=(11, 13, 15),
                 woven_heads=8, woven_dh=512, latch=True):
        super().__init__()
        self.olmo = olmo
        self.tok = tok
        self.Nmax, self.K, self.T = Nmax, K, T
        self.write_head = write_head
        self.use_latch = latch
        D = cfg.hidden_size
        self.D = D
        nL = cfg.num_hidden_layers
        # WRITE: per-cell pooler (pins) + grounded edge reader (diffs)
        self.reader = A.CellReader(D, dp)
        self.cand_head = nn.Sequential(nn.Linear(dp, dp), nn.GELU(), nn.Linear(dp, K))
        self.edge_bias = nn.Parameter(torch.tensor(-2.0))
        if write_head == "grounded":
            self.edge_reader = A.PairGroundedEdgeReader(D, dp, window=edge_window)
        else:
            self.edge_a = nn.Linear(dp, edge_dim)
            self.edge_b = nn.Linear(dp, edge_dim)
        # DEDUCE: the learned differentiable narrowing organ
        self.ded = I.ColorDeductor(T)
        # workspace + woven write-back into late layers
        self.ws_enc = nn.Linear(K, Dw)
        self.ws_query = nn.Parameter(torch.randn(Dw) * 0.02)
        self.ws_ln = nn.LayerNorm(Dw)
        self.woven_layers = A._parse_layer_spec(woven_layers, nL)
        self.woven = nn.ModuleDict({
            str(i): A.WovenWriteBack(D, Dw, heads=woven_heads, dh=woven_dh) for i in self.woven_layers
        })
        # LATCH: per-layer vocab-anchored injection (filled by set_color_embeds before training)
        self.latch_mods = nn.ModuleDict({str(i): GatedVocabLatch(D) for i in self.woven_layers}) \
            if latch else nn.ModuleDict()
        self.register_buffer("color_embeds", torch.zeros(K, D), persistent=False)

    def set_color_embeds(self, embed_weight, colors=COLORS):
        """Cache the OLMo input embedding of each color WORD (mean over its tokens) for the latch."""
        rows = []
        for c in colors[:self.K]:
            ids = self.tok(" " + c, add_special_tokens=False)["input_ids"]
            rows.append(embed_weight[ids].float().mean(0))
        self.color_embeds = torch.stack(rows).to(embed_weight.device)

    # ---- WRITE / DEDUCE / workspace (read frozen, detached host states) ----
    def write(self, h, org):
        m = org["mention"]
        hd = h.float().detach()
        denom = m.sum(-1, keepdim=True).clamp_min(1e-6)
        v_mean = (torch.einsum("bnt,btd->bnd", m, hd) / denom).detach()
        ctx = self.reader(v_mean, hd, org["attn"])
        cand = self.cand_head(ctx)
        if self.write_head == "grounded":
            elog = self.edge_reader(v_mean, hd, m, org["attn"])
        else:
            ea, eb = self.edge_a(ctx), self.edge_b(ctx)
            elog = torch.einsum("bid,bjd->bij", ea, eb) / ea.size(-1) ** 0.5
            elog = 0.5 * (elog + elog.transpose(1, 2))
        elog = elog + self.edge_bias
        vm = org["vmask"]; pair = vm[:, :, None] * vm[:, None, :]
        elog = elog.masked_fill(pair < 0.5, -1e4)
        E = torch.sigmoid(elog) * pair
        return cand, elog, E

    def deduce(self, cand, E, org):
        vm = org["vmask"]
        alive0 = torch.sigmoid(cand) * vm.unsqueeze(-1) + 1e-3
        return self.ded(alive0, E, vm)

    def workspace(self, alive, org):
        marg = alive / alive.sum(-1, keepdim=True).clamp_min(1e-6)
        ws = self.ws_enc(marg)
        qoh = F.one_hot(org["query"], self.Nmax).float()
        ws = self.ws_ln(ws + qoh.unsqueeze(-1) * self.ws_query[None, None, :])
        return ws, org["vmask"]

    def build_organ(self, org):
        """Encode the PROMPT (frozen host, no grad) and run WRITE + DEDUCE -> lattice workspace.
        The OLMo pass is no-grad (host frozen); the WRITE head + deductor still train (they read
        detached host states but their own weights receive gradient through the woven path)."""
        with torch.no_grad():
            h = trunk(self.olmo)(input_ids=org["input_ids"], attention_mask=org["attn"]).last_hidden_state
        cand, elog, E = self.write(h, org)
        alive = self.deduce(cand, E, org)
        ws, ws_mask = self.workspace(alive, org)
        # latch anchor: the QUERY cell's narrowed marginal, blended over color-word embeddings
        B = alive.size(0)
        qa = alive[torch.arange(B, device=alive.device), org["query"]]          # [B,K]
        qm = qa / qa.sum(-1, keepdim=True).clamp_min(1e-6)
        anchor = qm @ self.color_embeds.to(qm.dtype)                            # [B,D] in embed space
        return ws, ws_mask, anchor, {"cand": cand, "elog": elog, "E": E, "alive": alive}

    # ---- woven hooks ----
    @contextlib.contextmanager
    def organ_hooks(self, ws, ws_mask, anchor):
        handles = []

        def make_hook(layer_id):
            gate = self.woven[str(layer_id)]
            latch = self.latch_mods[str(layer_id)] if self.use_latch else None

            def hook(module, args, output):
                hs = output[0] if isinstance(output, tuple) else output
                hs = hs + gate(hs, ws, ws_mask)
                if latch is not None:
                    hs = hs + latch(hs, anchor)
                return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

            return hook

        layers = decoder_layers(self.olmo)
        try:
            for lid in self.woven_layers:
                handles.append(layers[lid].register_forward_hook(make_hook(lid)))
            yield
        finally:
            for hd in handles:
                hd.remove()

    def gen_logits(self, f_ids, f_attn, ws, ws_mask, anchor, organ_on=True):
        """LM logits over prompt+answer with the organ woven into the late layers (organ_on)."""
        if organ_on:
            with self.organ_hooks(ws, ws_mask, anchor):
                out = self.olmo(input_ids=f_ids, attention_mask=f_attn)
        else:
            out = self.olmo(input_ids=f_ids, attention_mask=f_attn)
        return out.logits

    # ---- training loss ----
    def loss(self, batch, aux=0.0):
        org = batch["org"]
        ws, ws_mask, anchor, diag = self.build_organ(org)
        logits = self.gen_logits(batch["f_ids"], batch["f_attn"], ws, ws_mask, anchor, organ_on=True).float()
        labels = batch["labels"]
        lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                             labels[:, 1:].reshape(-1), ignore_index=-100)
        total = lm
        parts = {"lm": float(lm.detach())}
        if aux > 0:
            vm = org["vmask"]; pair = vm[:, :, None] * vm[:, None, :]
            pw = torch.tensor(6.0, device=logits.device)
            le = F.binary_cross_entropy_with_logits(diag["elog"].clamp(-30, 30), org["adj"],
                                                    pos_weight=pw, reduction="none")
            le = (le * pair).sum() / pair.sum().clamp_min(1)
            lp = F.binary_cross_entropy_with_logits(diag["cand"], org["pins"], reduction="none")
            lp = (lp * org["pinned"].unsqueeze(-1)).sum() / org["pinned"].sum().clamp_min(1)
            total = total + aux * (le + lp)
            parts.update({"edge": float(le.detach()), "pin": float(lp.detach())})
        return total, parts, diag

    # ---- generation (eval) ----
    @torch.no_grad()
    def generate(self, prob, tok, Nmax, K, colors, device, max_new=8, organ_on=True, use_base=False):
        """Greedy-generate the answer for ONE problem and parse it. organ_on=False is the causal
        ablation (organ removed); use_base disables LoRA (base OLMo)."""
        org, prompts = _build_prompt_tensors([prob], tok, Nmax, K, colors, device)
        ws, ws_mask, anchor, _ = self.build_organ(org)
        ptext = prompts[0]
        ids = tok(ptext, return_tensors="pt").to(device)
        plen = ids["input_ids"].size(1)
        base_ctx = self.olmo.disable_adapter() if (use_base and hasattr(self.olmo, "disable_adapter")) \
            else contextlib.nullcontext()
        organ_ctx = self.organ_hooks(ws, ws_mask, anchor) if (organ_on and not use_base) \
            else contextlib.nullcontext()
        with base_ctx, organ_ctx:
            out = self.olmo.generate(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"],
                                     max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
        cont = tok.decode(out[0, plen:], skip_special_tokens=True)
        return cont, parse_answer(cont, colors, K)

    @torch.no_grad()
    def generate_fewshot(self, prob, tok, Nmax, K, colors, device, fewshot, max_new=8):
        """Base OLMo (LoRA disabled, no organ), few-shot prompted, greedy generation."""
        _, prompts = _build_prompt_tensors([prob], tok, Nmax, K, colors, device)
        full = fewshot + prompts[0]
        ids = tok(full, return_tensors="pt").to(device)
        plen = ids["input_ids"].size(1)
        ctx = self.olmo.disable_adapter() if hasattr(self.olmo, "disable_adapter") \
            else contextlib.nullcontext()
        with ctx:
            out = self.olmo.generate(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"],
                                     max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
        cont = tok.decode(out[0, plen:], skip_special_tokens=True)
        return cont, parse_answer(cont, colors, K)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def n_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
