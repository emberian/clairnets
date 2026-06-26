"""ORACLE-READOUT de-risk experiment (the highest-leverage GLaDOS isolation test).

Question: can a STRUCTURED, full-bandwidth projection of the EXACT narrowed lattice (the ORACLE
`clair.csp.exact_dedP` — the true surviving candidate set per cell) injected into OLMo's residual
stream make OLMo's OWN LM head GENERATE the right answer — and does that answer CAUSALLY depend on
the lattice? This isolates the residual-stream READOUT/alignment question from the (separately hard)
DEDUCTION question. There is NO learned organ and NO learned alpha here: we feed the EXACT lattice.

Contrast with `clair.gen_augmented` (the FAILED path): that wove a LEARNED organ (WRITE head +
ColorDeductor) plus a learned vocab "latch" into late layers. Here we delete all of that and hand
OLMo the ground-truth lattice through a deliberately simple, equivariant channel, then ask the only
question that matters for the architecture: *can the LM head read a lattice out of its residual
stream at all?*

DESIGN (structured, equivariant, full-bandwidth gamma — NOT a flatten):
  * For each problem compute the EXACT per-cell candidate-survival sets via clair.csp.exact_dedP
    -> a per-(cell, candidate) binary tensor surv[B, N, K]  (the ORACLE lattice we inject).
  * A SHARED tiny projection P: R^K -> R^D maps each cell's K-dim survival vector to a residual
    delta g[B, N, D].  P is shared across cells => the injection is EQUIVARIANT over cells (permuting
    cells permutes injections); it is per-cell/structured, not a global flatten tied to max N.
  * Scatter g onto each cell's MENTION-token positions at a LATE OLMo layer via a zero-init tanh
    gate:  hs <- hs + tanh(alpha) * sum_i mention[b,i,t] * g[b,i,:].  alpha=0 at init => EXACT no-op
    (verified bitwise vs base OLMo logits).  Late layer => the lattice shapes the hidden states that
    feed lm_head; causal attention routes each cell's injected survival to the "Answer:" position.
  * HOST = OLMo-2-1B, base FROZEN + LoRA (peft, attn+mlp, r=16).  Train LoRA + P + alpha on the
    GENERATIVE LM cross-entropy of the answer span only ("red" / "cannot be determined").

THE DECISIVE CAUSAL CONTROLS (run at eval, intervening on surv before injection):
  (a) BATCH-SHUFFLED   inject a DIFFERENT problem's oracle lattice;
  (b) CANDIDATE-PERMUTED  permute the value(color) labels of the survival vectors;
  (c) ONE-CELL CORRUPTED  flip the QUERY cell's survivors to a wrong/changed set.
If accuracy is high with the TRUE lattice and DROPS sharply under (a)/(b)/(c), the LM head is
genuinely READING the lattice through gamma -> the dense-readout coupling WORKS.  If accuracy is
unchanged under shuffle, gamma is a bypass / the LM ignores the lattice (or LoRA solved the text
itself) -> readout is the wall.  If accuracy is low even with the TRUE lattice, residual-stream
readout/alignment is itself the wall.
"""
from __future__ import annotations

import math
import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import csp as C
from . import curriculum as Cu
from . import induce as I

COLORS = Cu.COLORS
ABSTAIN_STR = "cannot be determined"


# ===================================================================== data / oracle
def make_problem(rng, N, k=3, edge_p=0.45, pin_frac=0.35, det_target=None) -> Cu.Problem:
    """One graph-coloring problem of EXACTLY N cells (witness-first, so always solvable). Reuses the
    curriculum coloring generator + its exact exact_dedP-based query/answer labelling."""
    n, d, kind, facts, s = Cu.gen_coloring(rng, k=k, n_lo=N, n_hi=N, edge_p=edge_p, pin_frac=pin_frac)
    csp = Cu.build_csp(n, d, facts)
    if det_target is None:
        det_target = bool(rng.random() < 0.5)
    q, ans, det = Cu._query_answer(csp, facts, rng, det_target)
    return Cu.Problem("coloring", n, d, kind, facts, q, ans, det, Cu.value_names(kind, d))


def oracle_survival_fast(p: Cu.Problem) -> tuple:
    """EXACT per-cell surviving candidate sets via per-(cell,value) SAT (each check stops at the first
    solution — fast & exact even at N=11). v survives at cell i iff the coloring with cell i pinned to
    v is still satisfiable. Equivalent to clair.csp.exact_dedP (asserted in the smoke); coloring only."""
    pins, edges = {}, set()
    for f in p.facts:
        if f[0] == "pin":
            pins[f[1]] = f[2]
        elif f[0] == "neq":
            edges.add((f[1], f[2]))
        else:
            raise ValueError(f"oracle_survival_fast handles coloring (pin/neq) only, got {f[0]!r}")
    out = []
    for i in range(p.n):
        surv = set()
        for v in range(p.d):
            if i in pins and pins[i] != v:
                continue
            test = dict(pins); test[i] = v
            if I._solutions(p.n, p.d, test, edges, cap=1):
                surv.add(v)
        out.append(frozenset(surv))
    return tuple(out)


def render_prompt(p: Cu.Problem, prompt_mode: str) -> str:
    """The prompt OLMo reads, ending in 'Answer:'.
       full  : the whole coloring problem in English (facts + question) — as specified by the review.
               The model CAN in principle deduce from this text (the LoRA-bypass confound).
       cells : ONLY the cell roster + the question, NO deductive facts. Text-deduction is impossible,
               so any correct answer MUST come from the injected lattice — the PURE readout isolator."""
    if prompt_mode == "full":
        return Cu.canonical_render(p) + " Answer:"
    if prompt_mode == "cells":
        roster = "Cells " + ", ".join(p.entity(i) for i in range(p.n)) + ". "
        return roster + Cu._question(p) + " Answer:"
    raise ValueError(prompt_mode)


def make_record(p: Cu.Problem, K: int, prompt_mode: str = "full") -> dict:
    """Freeze one problem into a training/eval record: prompt (ending 'Answer:'), gold answer string,
    entity char-span mentions, and the EXACT per-cell survival matrix surv[n,K] (the oracle lattice)."""
    prompt = render_prompt(p, prompt_mode)
    answer = Cu.canonical_answer(p)
    body = prompt[: prompt.rfind(" Answer:")]                   # mentions live in the pre-'Answer:' body
    mentions = Cu.entity_mentions(body, p.n)                    # spans valid in `prompt` (a prefix)
    surv_sets = oracle_survival_fast(p)
    surv = np.zeros((p.n, K), dtype=np.float32)
    for i, ss in enumerate(surv_sets):
        for v in ss:
            if v < K:
                surv[i, v] = 1.0
    gold_idx = p.answer if p.determined else K                   # index into candidate list (K = abstain)
    return {"prompt": prompt, "answer": answer, "n": p.n, "query": p.query,
            "determined": p.determined, "gold_idx": int(gold_idx),
            "mentions": {int(k): [tuple(s) for s in v] for k, v in mentions.items()},
            "surv": surv}


def build_pool(rng, N, K, size, det_frac=0.5, prompt_mode="full"):
    """A balanced pool of `size` coloring records at fixed cell-count N (50/50 determined/abstain)."""
    n_det = int(size * det_frac)
    det, ab = [], []
    while len(det) < n_det or len(ab) < size - n_det:
        want_det = len(det) < n_det
        p = make_problem(rng, N, k=K, det_target=want_det if (len(det) < n_det) ^ (len(ab) < size - n_det) else None)
        rec = make_record(p, K, prompt_mode)
        if rec["determined"] and len(det) < n_det:
            det.append(rec)
        elif (not rec["determined"]) and len(ab) < size - n_det:
            ab.append(rec)
    pool = det + ab
    rng.shuffle(pool)
    return pool


# ===================================================================== tensor batching
def _mention_tensor(all_spans, offsets, B, Nmax, T, device, base_shift=0):
    """Build mention[B,Nmax,T] from per-row {cell: [(cs,ce)]} char spans + the row's token offsets.
    base_shift shifts the spans (used when a fewshot prefix is prepended)."""
    mention = torch.zeros(B, Nmax, T, device=device)
    for b, spans in enumerate(all_spans):
        offs = offsets[b].tolist()
        for cell, sps in spans.items():
            if cell >= Nmax:
                continue
            for (cs, ce) in sps:
                cs += base_shift; ce += base_shift
                for ti, (a, c) in enumerate(offs):
                    if a == c:            # special / pad token
                        continue
                    if a < ce and c > cs:
                        mention[b, cell, ti] = 1.0
    return mention


def build_train_batch(recs, tok, K, device):
    """Tokenize prompt + ' ' + answer; answer-span CE mask; mention[B,Nmax,T] + surv[B,Nmax,K]."""
    Nmax = max(r["n"] for r in recs)
    B = len(recs)
    prompts = [r["prompt"] for r in recs]
    fulls = [r["prompt"] + " " + r["answer"] for r in recs]
    plens = [len(pt) for pt in prompts]
    enc = tok(fulls, return_offsets_mapping=True, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    offsets = enc["offset_mapping"]
    T = input_ids.size(1)
    labels = input_ids.clone()
    for b in range(B):
        offs = offsets[b].tolist()
        for ti, (a, c) in enumerate(offs):
            is_ans = (a != c) and (a >= plens[b]) and (attn[b, ti] > 0.5)
            if not is_ans:
                labels[b, ti] = -100
    mention = _mention_tensor([r["mentions"] for r in recs], offsets, B, Nmax, T, device)
    surv = torch.zeros(B, Nmax, K, device=device)
    for b, r in enumerate(recs):
        surv[b, : r["n"], :] = torch.from_numpy(r["surv"]).to(device)
    return {"input_ids": input_ids, "attn": attn, "labels": labels,
            "mention": mention, "surv": surv, "Nmax": Nmax}


# ===================================================================== structured oracle gamma
class OracleGamma(nn.Module):
    """SHARED per-cell projection of the K-dim candidate-survival vector into a D-dim residual delta,
    gated by a zero-init tanh scalar (exact no-op at init). Equivariant over cells (one projection
    applied to every cell), full-bandwidth (the whole survival vector is projected, not a scalar)."""
    def __init__(self, D, K, hidden=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(K, hidden), nn.GELU(), nn.Linear(hidden, D))
        self.alpha = nn.Parameter(torch.zeros(1))

    def delta(self, surv, mention):
        # surv [B,N,K] -> g [B,N,D]; scatter onto mention positions -> [B,T,D]
        g = self.proj(surv)                                        # equivariant per-cell projection
        return torch.einsum("bnt,bnd->btd", mention, g)            # [B,T,D]


def _decoder_layers(peft_model):
    base = peft_model.base_model.model if hasattr(peft_model, "base_model") else peft_model
    return base.model.layers


class OracleReadout(nn.Module):
    """LoRA-adapted OLMo whose lm_head GENERATES the answer, with the EXACT oracle lattice scattered
    into a late layer's residual stream via OracleGamma (zero-init gate). Base OLMo frozen; train LoRA
    + gamma on generative LM CE."""
    def __init__(self, peft_model, D, K, inject_layer, gamma_hidden=256):
        super().__init__()
        self.model = peft_model
        self.gamma = OracleGamma(D, K, gamma_hidden)
        self.inject_layer = inject_layer
        self._surv = None
        self._mention = None
        self._enabled = True
        layers = _decoder_layers(peft_model)
        self._handle = layers[inject_layer].register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        if (not self._enabled) or (self._surv is None):
            return output
        hs = output[0] if isinstance(output, tuple) else output
        delta = self.gamma.delta(self._surv.to(hs.device), self._mention.to(hs.device))
        hs = hs + (torch.tanh(self.gamma.alpha) * delta).to(hs.dtype)
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

    @contextlib.contextmanager
    def injection(self, surv, mention, enabled=True):
        old = (self._surv, self._mention, self._enabled)
        self._surv, self._mention, self._enabled = surv, mention, enabled
        try:
            yield
        finally:
            self._surv, self._mention, self._enabled = old

    def logits(self, input_ids, attn):
        return self.model(input_ids=input_ids, attention_mask=attn).logits

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def n_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ===================================================================== causal-control transforms
def apply_control(surv, recs, control, K, perm=(1, 2, 0)):
    """Return a transformed copy of surv[B,Nmax,K] for a causal control.
       true      : identity (the real oracle lattice)
       shuffle   : inject a DIFFERENT problem's lattice (roll along batch)
       permute   : permute the candidate(color) labels of every survival vector
       corrupt   : flip ONLY the query cell's survivors to a single wrong/changed color
    """
    s = surv.clone()
    if control == "true":
        return s
    if control == "shuffle":
        return torch.roll(s, shifts=1, dims=0)
    if control == "permute":
        idx = torch.tensor(perm[:K], device=s.device)
        return s.index_select(-1, idx)                              # s2[...,k] = s[...,perm[k]]
    if control == "corrupt":
        for b, r in enumerate(recs):
            q = r["query"]
            row = s[b, q]
            nz = torch.nonzero(row > 0.5).flatten()
            first = int(nz[0]) if nz.numel() else 0
            s[b, q].zero_()
            s[b, q, (first + 1) % K] = 1.0
        return s
    raise ValueError(control)


# ===================================================================== closed-set LM-head scoring
@torch.no_grad()
def score_closedset(model, recs, tok, K, colors, device, control="true", inject=True,
                    fewshot="", bs=16, use_base=False):
    """Primary metric: the LM head GENERATES the answer, scored by summed answer-token logprob over
    the LEGAL answer strings {colors..., 'cannot be determined'} (constrained decoding). Returns
    overall exact-match accuracy + determined-only color accuracy + abstain accuracy.

    inject=False  -> no lattice (base-OLMo / text-only path); fewshot is prepended.
    use_base=True -> disable LoRA adapters (pure base OLMo)."""
    model.eval()
    cands = [" " + c for c in colors[:K]] + [" " + ABSTAIN_STR]
    Cn = len(cands)
    correct = tot = 0
    det_tot = det_right = 0
    abst_tot = abst_right = 0
    for i in range(0, len(recs), bs):
        chunk = recs[i:i + bs]
        Bp = len(chunk)
        Nmax = max(r["n"] for r in chunk)
        surv = torch.zeros(Bp, Nmax, K, device=device)
        for b, r in enumerate(chunk):
            surv[b, : r["n"], :] = torch.from_numpy(r["surv"]).to(device)
        surv = apply_control(surv, chunk, control, K)
        # expand each problem over the Cn candidate continuations
        fulls, plens, spans_rows, prob_of = [], [], [], []
        for pi, r in enumerate(chunk):
            prompt = fewshot + r["prompt"]
            shift = len(fewshot)
            for cand in cands:
                fulls.append(prompt + cand)
                plens.append(len(prompt))
                spans_rows.append({k: [(a + shift, c + shift) for (a, c) in v] for k, v in r["mentions"].items()})
                prob_of.append(pi)
        enc = tok(fulls, return_offsets_mapping=True, padding=True, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        offsets = enc["offset_mapping"]
        T = ids.size(1)
        if inject:
            mention = _mention_tensor(spans_rows, offsets, Bp * Cn, Nmax, T, device)
            surv_exp = surv[torch.tensor(prob_of, device=device)]   # [Bp*Cn, Nmax, K]
            ctx = model.injection(surv_exp, mention, enabled=True)
        else:
            ctx = model.injection(None, None, enabled=False)
        base_ctx = model.model.disable_adapter() if (use_base and hasattr(model.model, "disable_adapter")) \
            else contextlib.nullcontext()
        with base_ctx, ctx:
            logits = model.logits(ids, attn).float()
        lp = torch.log_softmax(logits[:, :-1], -1)
        tgt = ids[:, 1:]
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)        # [B*Cn, T-1]
        # answer-span mask per row
        scores = torch.full((Bp * Cn,), -1e9, device=device)
        for row in range(Bp * Cn):
            offs = offsets[row].tolist()
            mask = torch.zeros(T - 1, device=device)
            for ti in range(1, T):
                a, c = offs[ti]
                if a != c and a >= plens[row] and attn[row, ti] > 0.5:
                    mask[ti - 1] = 1.0
            denom = mask.sum().clamp_min(1.0)
            scores[row] = (tok_lp[row] * mask).sum() / denom        # length-normalized logprob
        scores = scores.view(Bp, Cn)
        pred = scores.argmax(-1)
        for b, r in enumerate(chunk):
            gi = r["gold_idx"]
            ok = int(pred[b].item()) == gi
            correct += ok; tot += 1
            if r["determined"]:
                det_tot += 1; det_right += ok
            else:
                abst_tot += 1; abst_right += ok
    return {"overall": correct / max(1, tot), "det_acc": det_right / max(1, det_tot),
            "abst_acc": abst_right / max(1, abst_tot), "n": tot}


# ===================================================================== free greedy generation (sanity)
@torch.no_grad()
def greedy_gen(model, recs, tok, K, colors, device, max_new=6, control="true"):
    """TRUE unconstrained generation (use_cache=False so the late-layer injection re-applies each
    step): greedily emit the answer tokens and parse. Returns (accuracy, examples)."""
    model.eval()
    correct = 0
    examples = []
    stop_ids = {tok.eos_token_id}
    nl = tok("\n", add_special_tokens=False)["input_ids"]
    if nl:
        stop_ids.add(nl[-1])
    for ri, r in enumerate(recs):
        Nmax = r["n"]
        surv = torch.zeros(1, Nmax, K, device=device)
        if control == "shuffle":
            # bs=1: borrow the NEXT record's lattice (roll-by-1 is identity on a singleton)
            src = recs[(ri + 1) % len(recs)]["surv"]
            m = min(Nmax, src.shape[0])
            surv[0, :m, :] = torch.from_numpy(src[:m]).to(device)
        else:
            surv[0, : r["n"], :] = torch.from_numpy(r["surv"]).to(device)
            if control in ("permute", "corrupt"):
                surv = apply_control(surv, [r], control, K)
        penc = tok(r["prompt"], return_offsets_mapping=True, return_tensors="pt")
        ids = penc["input_ids"].to(device)
        plen = ids.size(1)
        m_prompt = _mention_tensor([r["mentions"]], penc["offset_mapping"], 1, Nmax, plen, device)
        gen = []
        for _ in range(max_new):
            T = ids.size(1)
            if T > plen:
                mention = torch.cat([m_prompt, torch.zeros(1, Nmax, T - plen, device=device)], dim=2)
            else:
                mention = m_prompt
            attn = torch.ones_like(ids)
            with model.injection(surv, mention, enabled=True):
                out = model.model(input_ids=ids, attention_mask=attn, use_cache=False)
            nxt = int(out.logits[0, -1].argmax())
            if nxt in stop_ids:
                break
            gen.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
        cont = tok.decode(gen, skip_special_tokens=True).strip()
        gold = r["answer"]
        ok = cont.lower().startswith(gold.lower()) or (gold == ABSTAIN_STR and "cannot" in cont.lower())
        correct += int(ok)
        if len(examples) < 8:
            examples.append((gold, cont))
    return correct / max(1, len(recs)), examples
