"""DIFFUSION-organ graft — the "iterative marriage" (notes/future_directions.md §5).

Weave the lattice-deduction organ into a masked-diffusion LM's DENOISING loop so the sequence
denoises and the lattice narrows IN LOCKSTEP. Both are fixed-point iterations; we make them share
ONE iteration and inform each other:

    at each denoising step t:
      (alpha)  compile a lattice state from the CURRENT, PARTLY-MASKED sequence  (the new wrinkle:
               alpha reads a NOISY sequence, not clean text -> it reads the committed answer tokens
               as unit constraints on top of the clean problem constraints, and refines a PERSISTENT
               lattice rather than recompiling from scratch);
      (organ)  narrow the lattice ONE step (clair.csp.ac_step, the local deductor; or exact_dedP,
               the oracle deductor — a learned organ slots in at exactly this seam);
      (gamma)  write the narrowed per-cell survival sets back into the denoiser's hidden state at the
               answer-token positions, through a SHARED per-cell projection gated by a zero-init tanh
               scalar  ->  EXACT no-op at init (bitwise), and a biasing channel once trained.

So denoising-step and narrowing-step advance together: each commit feeds alpha; each narrowing feeds
gamma; the denoiser's next commit is biased by the deduction. The organ is a PARALLEL refinement, not
an autoregressive side-loop. The answer is read from the FINAL denoised answer region (now organ-shaped).

WHICH DIFFUSION LM (honest feasibility, mid-2026):
  * DiffusionGemma (google/diffusiongemma-26B-A4B-it) is the named target but is a 25.2B-total MoE:
    ~50 GB bf16 weights (all experts resident) > one L40S (46 GB). bf16 does NOT fit; only 4-bit/NVFP4
    (~13-18 GB) fits, at the cost of quant + remote-code complexity. Out of scope for a mechanism proto.
  * LLaDA-8B (GSAI-ML/LLaDA-8B-Instruct) is a real open masked-diffusion LM that DOES fit (~16 GB bf16).
    Its generate() is the canonical iterative-unmask loop; `LLaDAAdapter` below wraps it to our hook
    interface and we run the no-op-at-init check on it (proves the graft ports to a real diffusion LM).
  * The PRIMARY vehicle here is `SyntheticMDLM` — a small bidirectional absorbing-state masked-diffusion
    transformer we implement and train in seconds, so the FULL coupling (alpha-on-noisy + organ + gamma
    + lockstep) is proved cleanly, cheaply and reproducibly on a graph-coloring CSP rendered for a
    diffusion LM. Model quality is not the claim; the MECHANISM is.

Run:  python -m clair.diffusion_organ            # synthetic smoke (no-op + lockstep narrowing + tiny train)
      python -m clair.diffusion_organ --llada    # additionally attempt the LLaDA-8B no-op-at-init check
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import csp as C
from . import curriculum as Cu
from .model import Block


# ===================================================================== sequence rendering for a diffusion LM
# A coloring CSP rendered as ONE sequence the diffusion LM denoises. Layout (fixed for given N,k):
#   [ pin region : N ] [ adjacency region : N*N ] [ answer region : N ]
# pin/adjacency are CLEAN conditioning (never masked); the answer region is what diffuses (mask -> color).
# Vocab:  MASK=0  NO=1  YES=2  COLOR0=3..3+k-1
MASK, NO, YES, COLOR0 = 0, 1, 2, 3


def vocab_size(k: int) -> int:
    return COLOR0 + k


def render(p: Cu.Problem, N: int, k: int):
    """Render a coloring Problem into (tokens, answer_pos, pins, edges).
       tokens : LongTensor[T] with the answer region all-MASK (the t=0 fully-noised state).
       answer_pos : LongTensor[N] absolute positions of each cell's answer token.
       pins : dict cell->value ; edges : set of (i,j)  (what alpha would read off the CLEAN regions)."""
    n = p.n
    pin_base, adj_base, ans_base = 0, N, N + N * N
    T = N + N * N + N
    x = torch.full((T,), NO, dtype=torch.long)
    pins, edges = {}, set()
    for f in p.facts:
        if f[0] == "pin":
            pins[f[1]] = f[2]
            x[pin_base + f[1]] = COLOR0 + f[2]
        elif f[0] == "neq":
            i, j = f[1], f[2]
            edges.add((i, j))
            x[adj_base + i * N + j] = YES
            x[adj_base + j * N + i] = YES
    ans_pos = torch.tensor([ans_base + i for i in range(N)], dtype=torch.long)
    x[ans_pos[:n]] = MASK                                   # cells 0..n-1 diffuse; padding cells stay NO
    return x, ans_pos, pins, edges


def base_csp(n: int, k: int, pins: dict, edges: set) -> C.CSP:
    facts = [("pin", i, v) for i, v in pins.items()] + [("neq", i, j) for (i, j) in edges]
    return Cu.build_csp(n, k, facts)


# ===================================================================== alpha-on-noisy + the organ (lattice side)
class CoupledLattice:
    """The lattice half of the marriage. Holds a PERSISTENT per-cell domain `dom` (tuple of frozensets)
    and advances it in lockstep with the denoiser:

      observe(tokens) : alpha reads the CURRENT (partly-denoised) answer region -> every COMMITTED cell
                        becomes a unit constraint; intersect it into `dom` (monotone, refine-not-recompile).
      narrow(steps)   : the ORGAN narrows `dom` `steps` local deduction passes (ac_step) — or one exact
                        deduction (exact_dedP) if organ='exact'.

    `dom` is monotone non-increasing across the whole loop => cardinality only ever drops: that IS the
    lockstep narrowing. Soundness is free (ac_step / exact_dedP are both dominated by the exact transformer).
    """

    def __init__(self, n: int, k: int, ans_pos, pins, edges, organ: str = "ac"):
        self.n, self.k, self.organ = n, k, organ
        self.ans_pos = ans_pos
        self.csp = base_csp(n, k, pins, edges)
        self.dom = list(self.csp.full())
        for i, v in pins.items():                          # pins are immediate unit constraints
            self.dom[i] = frozenset({v})
        self.dom = tuple(self.dom)

    def observe(self, tokens: torch.Tensor):
        """alpha on a NOISY sequence: committed (non-MASK) answer tokens -> unit constraints; intersect."""
        new = list(self.dom)
        for i in range(self.n):
            tid = int(tokens[self.ans_pos[i]])
            if tid >= COLOR0:                              # this cell has been denoised/committed
                v = tid - COLOR0
                new[i] = self.dom[i] & frozenset({v})     # intersect (stays sound even if model erred -> {})
        self.dom = tuple(new)

    def narrow(self, steps: int = 1):
        if self.organ == "exact":
            self.dom = C.exact_dedP(self.csp, self.dom)
        else:
            for _ in range(steps):
                nxt = C.ac_step(self.csp, self.dom)
                if nxt == self.dom:
                    break
                self.dom = nxt

    def cardinality(self) -> int:
        return sum(len(self.dom[i]) for i in range(self.n))

    def status(self) -> str:
        return C.status(self.dom[: self.n])

    def survival(self) -> torch.Tensor:
        """[n,k] binary survival matrix — the gamma payload."""
        s = torch.zeros(self.n, self.k)
        for i in range(self.n):
            for v in self.dom[i]:
                if v < self.k:
                    s[i, v] = 1.0
        return s


# ===================================================================== gamma (lattice -> denoiser hidden state)
class OrganGamma(nn.Module):
    """SHARED per-cell projection of the k-dim survival vector into a D-dim residual delta, scattered onto
    each cell's ANSWER-token position, gated by a zero-init tanh scalar => EXACT no-op at init. This is the
    proven dense-gamma readback (clair.oracle_readout.OracleGamma) re-pointed at a diffusion denoiser, and
    invoked INSIDE the denoising loop instead of once per autoregressive forward."""

    def __init__(self, D, k, hidden=128):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(k, hidden), nn.GELU(), nn.Linear(hidden, D))
        self.alpha = nn.Parameter(torch.zeros(1))

    def delta(self, surv, ans_pos, T):
        """surv [B,n,k] -> add proj(surv) at answer positions -> [B,T,D] (zeros elsewhere)."""
        B, n, _ = surv.shape
        g = self.proj(surv)                                # [B,n,D] equivariant per-cell
        D = g.size(-1)
        out = surv.new_zeros(B, T, D)
        idx = ans_pos[:n].view(1, n, 1).expand(B, n, D)
        out.scatter_(1, idx, g)
        return out

    def gate(self):
        return torch.tanh(self.alpha)


# ===================================================================== the synthetic masked-diffusion LM
class SyntheticMDLM(nn.Module):
    """Small bidirectional (non-causal) absorbing-state masked-diffusion transformer. Trained to predict
    masked answer tokens from the clean pin/adjacency conditioning + the partially-unmasked answer region.
    Exposes a forward hook seam (a named late block) for OrganGamma — the same residual-stream graft point
    the organ uses on any transformer."""

    def __init__(self, k, ctx, d=128, n_layers=4, heads=4):
        super().__init__()
        V = vocab_size(k)
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, heads, "std", causal=False) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.tok.weight
        self.inject_layer = n_layers - 2                    # a late block (the residual-stream readout site)
        self._gamma = None
        self._surv = None
        self._ans_pos = None
        self._enabled = False

    def attach_gamma(self, gamma: OrganGamma, ans_pos):
        self._gamma, self._ans_pos = gamma, ans_pos

    def set_injection(self, surv, enabled=True):
        self._surv, self._enabled = surv, enabled

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        h = self.tok(idx) + self.pos(pos)
        for li, blk in enumerate(self.blocks):
            h = blk(h)
            if (li == self.inject_layer and self._enabled and self._gamma is not None
                    and self._surv is not None):
                delta = self._gamma.delta(self._surv.to(h.dtype), self._ans_pos.to(h.device), T)
                h = h + (self._gamma.gate() * delta).to(h.dtype)   # gate=0 at init -> bitwise no-op
        return self.head(self.ln_f(h))


# ===================================================================== the coupled denoising loop (the marriage)
@torch.no_grad()
def coupled_denoise(model: SyntheticMDLM, gamma: OrganGamma, p: Cu.Problem, N, k,
                    steps=None, organ="ac", inject=True, device="cpu", temperature=0.0):
    """ONE iterative marriage. Returns dict with the cardinality trajectory + final answer region.

    Each iteration is BOTH a denoising step and a narrowing step:
      1. gamma writes the CURRENT lattice survival into the denoiser hidden state (if inject);
      2. the denoiser scores all positions; we COMMIT the highest-confidence still-masked answer cells
         (LLaDA-style low-confidence remasking schedule: ~1 cell/step);
      3. alpha observes the new commits; the organ narrows one step; we log cardinality.
    """
    model.eval()
    x, ans_pos, pins, edges = render(p, N, k)
    x = x.to(device)
    ans_pos = ans_pos.to(device)
    lat = CoupledLattice(p.n, k, ans_pos, pins, edges, organ=organ)
    model.attach_gamma(gamma, ans_pos)
    n = p.n
    steps = steps or n
    cells = list(range(n))
    traj = [lat.cardinality()]                              # cardinality BEFORE any denoising
    # even unmask schedule across `steps`
    per = [0] * steps
    remaining = n
    for s in range(steps):
        per[s] = remaining // (steps - s)
        remaining -= per[s]
    for s in range(steps):
        masked = [i for i in cells if int(x[ans_pos[i]]) == MASK]
        if not masked:
            traj.append(lat.cardinality())
            continue
        surv = lat.survival().unsqueeze(0).to(device)      # current narrowed lattice -> gamma payload
        model.set_injection(surv, enabled=inject)
        logits = model(x.unsqueeze(0))[0]                  # [T,V]
        model.set_injection(None, enabled=False)
        # confidence per masked answer cell over the COLOR tokens only
        ans_logits = logits[ans_pos[masked]][:, COLOR0:COLOR0 + k]      # [m,k]
        prob = ans_logits.softmax(-1)
        conf, choice = prob.max(-1)                        # [m]
        order = torch.argsort(conf, descending=True)
        budget = max(1, per[s])
        for r in order[:budget]:
            cell = masked[int(r)]
            x[ans_pos[cell]] = COLOR0 + int(choice[int(r)])
        lat.observe(x)                                     # alpha reads the noisy/partly-denoised seq
        lat.narrow(steps=1)                                # the organ narrows ONE step (lockstep)
        traj.append(lat.cardinality())
    answer = [int(x[ans_pos[i]]) - COLOR0 for i in range(n)]
    return {"traj": traj, "answer": answer, "lattice": lat, "tokens": x.cpu(),
            "status": lat.status(), "query": p.query, "gold": p.answer, "determined": p.determined}


# ===================================================================== tiny training (bonus: learn the coupling)
def gen_problem(rng, N, k, det_target=True):
    n, d, kind, facts, s = Cu.gen_coloring(rng, k=k, n_lo=N, n_hi=N, edge_p=0.5, pin_frac=0.4)
    csp = Cu.build_csp(n, d, facts)
    q, ans, det = Cu._query_answer(csp, facts, rng, det_target)
    p = Cu.Problem("coloring", n, d, kind, facts, q, ans, det, Cu.value_names(kind, d))
    return p, s                                            # s = a valid witness coloring


def train_coupling(model, gamma, N, k, device, steps=600, bs=32, lr=3e-3, seed=0):
    """Absorbing-state masked-diffusion training WITH the organ in the loop, so the denoiser learns to
    read gamma. For each example: render with the witness coloring, mask a random subset of answer cells,
    feed the organ's lattice computed from the UNMASKED cells, predict the masked cells."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(list(model.parameters()) + list(gamma.parameters()), lr=lr)
    model.train()
    T = N + N * N + N
    for it in range(steps):
        batch_x, batch_tgt, batch_surv, batch_anspos, batch_lossmask = [], [], [], [], []
        for _ in range(bs):
            p, s = gen_problem(rng, N, k)
            x, ans_pos, pins, edges = render(p, N, k)
            n = p.n
            sol = [int(s[i]) for i in range(n)]
            x[ans_pos[:n]] = torch.tensor([COLOR0 + v for v in sol])   # start from the solved region
            mask_ratio = float(rng.uniform(0.2, 1.0))
            kmask = max(1, int(round(mask_ratio * n)))
            mcells = rng.choice(n, size=kmask, replace=False)
            lossmask = torch.zeros(T, dtype=torch.bool)
            tgt = torch.full((T,), -100, dtype=torch.long)
            for c in mcells:
                tgt[ans_pos[c]] = COLOR0 + sol[c]
                lossmask[ans_pos[c]] = True
                x[ans_pos[c]] = MASK
            # organ reads the UNMASKED cells -> lattice (alpha-on-partly-denoised, exactly as at inference)
            lat = CoupledLattice(n, k, ans_pos, pins, edges, organ="ac")
            lat.observe(x)
            lat.narrow(steps=1)
            surv = torch.zeros(N, k)
            surv[:n] = lat.survival()
            batch_x.append(x); batch_tgt.append(tgt); batch_surv.append(surv)
            batch_anspos.append(ans_pos); batch_lossmask.append(lossmask)
        X = torch.stack(batch_x).to(device)
        TGT = torch.stack(batch_tgt).to(device)
        SURV = torch.stack(batch_surv).to(device)
        AP = batch_anspos[0].to(device)                    # same layout across batch (fixed N)
        model.attach_gamma(gamma, AP)
        model.set_injection(SURV, enabled=True)
        logits = model(X)
        model.set_injection(None, enabled=False)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), TGT.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0 or it == steps - 1:
            print(f"  [train] step {it:4d}  loss {loss.item():.4f}  gate {gamma.gate().item():+.4f}")
    return model, gamma


# ===================================================================== LLaDA adapter (real diffusion LM, port check)
class LLaDAAdapter:
    """Wrap GSAI-ML/LLaDA-8B-Instruct to our hook interface and verify the zero-init gamma is a bitwise
    no-op on a REAL masked-diffusion LM. Proves the graft ports beyond the synthetic model. Best-effort:
    needs the weights + trust_remote_code; we don't train it here (8B, mechanism check only)."""

    def __init__(self, model_id="GSAI-ML/LLaDA-8B-Instruct", device="cuda", dtype=torch.bfloat16):
        import transformers
        import transformers.modeling_utils as mu
        from transformers import AutoModel, AutoTokenizer
        # compat shims: LLaDA's remote modeling code predates transformers 5.x. The new loader
        # (a) reads `all_tied_weights_keys` (a dict in 5.x) and (b) calls tie_weights(missing_keys=...,
        # recompute_mapping=...) with kwargs the old custom class rejects. Patch the (class-identity-
        # independent) caller `_finalize_model_loading` so the REAL diffusion LM loads on this box.
        # The obstacle is transformers<->remote-code API churn, NOT the coupling.
        if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):
            transformers.PreTrainedModel.all_tied_weights_keys = property(lambda self: {})
        def _retie(model):
            if getattr(model.config, "tie_word_embeddings", True):
                o, i = model.get_output_embeddings(), model.get_input_embeddings()
                if o is not None and i is not None:
                    o.weight = i.weight
        raw = mu.PreTrainedModel.__dict__.get("_finalize_model_loading")
        if raw is not None and not getattr(raw, "_organ_patched", False):
            if isinstance(raw, classmethod):
                func = raw.__func__

                def _p(cls, model, *a, **k):
                    model.tie_weights = lambda *aa, **kk: None
                    r = func(cls, model, *a, **k); _retie(model); return r
                _p._organ_patched = True
                mu.PreTrainedModel._finalize_model_loading = classmethod(_p)
            else:                                              # staticmethod / plain function
                func = raw.__func__ if isinstance(raw, staticmethod) else raw

                def _p(model, *a, **k):
                    model.tie_weights = lambda *aa, **kk: None
                    r = func(model, *a, **k); _retie(model); return r
                _p._organ_patched = True
                mu.PreTrainedModel._finalize_model_loading = staticmethod(_p)
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True,
                                               torch_dtype=dtype).to(device).eval()
        self.model.config.use_cache = False                # LLaDA forward reads this; absent on old config
        self.device = device
        self.MASK_ID = 126336                              # LLaDA absorbing/mask token id

    def _blocks(self):
        m = self.model
        for path in ("model.transformer.blocks", "transformer.blocks", "model.layers", "model.model.layers"):
            obj = m
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return list(obj), path
            except AttributeError:
                continue
        raise RuntimeError("could not locate LLaDA transformer blocks; inspect model.named_modules()")

    @torch.no_grad()
    def noop_at_init_check(self, prompt="What color is node A? Constraints: A != B."):
        """Register a zero-init gamma hook on a late block; confirm logits are bitwise-identical to base."""
        blocks, path = self._blocks()
        D = self.model.config.hidden_size
        gamma = OrganGamma(D, k=8).to(self.device, dtype=next(self.model.parameters()).dtype)
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        gen = torch.full((1, 16), self.MASK_ID, dtype=torch.long, device=self.device)
        x = torch.cat([ids, gen], dim=1)
        base = self.model(x).logits.clone()
        layer = max(0, len(blocks) - 3)
        ans_pos = torch.arange(ids.size(1), x.size(1), device=self.device)

        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            surv = torch.ones(1, ans_pos.numel(), 8, device=hs.device, dtype=hs.dtype)
            delta = gamma.delta(surv, ans_pos, hs.size(1))
            hs = hs + (gamma.gate() * delta).to(hs.dtype)
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

        h = blocks[layer].register_forward_hook(hook)
        try:
            inj = self.model(x).logits
        finally:
            h.remove()
        identical = torch.equal(base, inj)
        return {"blocks_path": path, "n_blocks": len(blocks), "hidden": D,
                "inject_layer": layer, "noop_bitwise": bool(identical),
                "max_abs_diff": float((base - inj).abs().max())}


# ===================================================================== smoke
def smoke(N=6, k=3, device=None, train_steps=600, n_eval=24, seed=1):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ctx = N + N * N + N
    print(f"=== DIFFUSION-organ smoke  (synthetic masked-diffusion LM)  N={N} k={k} dev={device} ===")
    model = SyntheticMDLM(k, ctx).to(device)
    D = model.head.in_features
    gamma = OrganGamma(D, k).to(device)

    # ---- (1) no-op-at-init: bitwise-identical denoiser logits with gate on vs off ----
    rng = np.random.default_rng(seed)
    p0, _ = gen_problem(rng, N, k)
    x0, ans_pos0, pins0, edges0 = render(p0, N, k)
    x0 = x0.unsqueeze(0).to(device)
    lat0 = CoupledLattice(p0.n, k, ans_pos0.to(device), pins0, edges0)
    lat0.narrow(1)
    surv0 = torch.zeros(1, N, k, device=device); surv0[0, :p0.n] = lat0.survival()
    model.attach_gamma(gamma, ans_pos0.to(device))
    model.eval()
    with torch.no_grad():
        model.set_injection(None, enabled=False)
        base = model(x0).clone()
        model.set_injection(surv0, enabled=True)          # alpha=0 -> tanh(0)=0 -> exact no-op
        inj = model(x0)
        model.set_injection(None, enabled=False)
    noop = torch.equal(base, inj)
    print(f"[no-op@init] gate={gamma.gate().item():.1e}  bitwise-identical={noop}  "
          f"max|diff|={(base-inj).abs().max().item():.3e}")
    assert noop, "zero-init gamma must be a bitwise no-op at init!"

    # ---- (2) lockstep narrowing BEFORE training (organ narrows even with an untrained denoiser) ----
    print("\n[lockstep] cardinality trajectory across denoising steps (organ='ac'):")
    res = coupled_denoise(model, gamma, p0, N, k, organ="ac", inject=False, device=device)
    print(f"  untrained denoiser, gamma OFF: card {res['traj']}  status={res['status']}")
    res_ex = coupled_denoise(model, gamma, p0, N, k, organ="exact", inject=False, device=device)
    print(f"  untrained denoiser, organ=exact: card {res_ex['traj']}  status={res_ex['status']}")

    # ---- (3) tiny train so the denoiser READS the organ; then show clean lockstep to a solved lattice ----
    print(f"\n[train] coupling for {train_steps} steps (denoiser + gamma)...")
    train_coupling(model, gamma, N, k, device, steps=train_steps, seed=seed + 7)
    print(f"[train] final gate (tanh(alpha)) = {gamma.gate().item():+.4f}  (nonzero => the denoiser uses gamma)")

    # re-verify no-op identity property is structural (set alpha back to 0 -> still exact no-op)
    with torch.no_grad():
        saved = gamma.alpha.clone()
        gamma.alpha.zero_()
        model.set_injection(surv0, enabled=True); a = model(x0)
        model.set_injection(None, enabled=False); b = model(x0)
        gamma.alpha.copy_(saved)
    print(f"[no-op@alpha=0 post-train] bitwise-identical={torch.equal(a,b)} (gate is a true off-switch)")

    # ---- (4) evaluation: does the coupled loop narrow + solve, and does gamma matter? ----
    print(f"\n[eval] {n_eval} problems — coupled denoising with the organ ON vs OFF:")
    rng = np.random.default_rng(seed + 100)
    drops_on, solved_on, correct_on = [], 0, 0
    solved_off, correct_off = 0, 0
    monotone = True
    for ei in range(n_eval):
        p, _ = gen_problem(rng, N, k)
        r_on = coupled_denoise(model, gamma, p, N, k, organ="ac", inject=True, device=device)
        r_off = coupled_denoise(model, gamma, p, N, k, organ="ac", inject=False, device=device)
        t = r_on["traj"]
        if ei < 3:
            print(f"  sample {ei}: trained+ON card-traj {t}  status={r_on['status']}  "
                  f"(OFF ends {r_off['traj'][-1]}, status={r_off['status']})")
        monotone &= all(t[i + 1] <= t[i] for i in range(len(t) - 1))
        drops_on.append(t[0] - t[-1])
        if r_on["status"] == "solved":
            solved_on += 1
        if r_off["status"] == "solved":
            solved_off += 1
        if p.determined:
            if r_on["lattice"].dom[p.query] == frozenset({p.answer}):
                correct_on += 1
            if r_off["lattice"].dom[p.query] == frozenset({p.answer}):
                correct_off += 1
    print(f"  lattice cardinality monotone-nonincreasing across all loops : {monotone}")
    print(f"  mean cardinality DROP (start-end), organ ON                 : {np.mean(drops_on):.2f}")
    print(f"  lattice SOLVED (all singletons) at loop end  ON / OFF       : {solved_on}/{n_eval}  vs  {solved_off}/{n_eval}")
    print(f"  query cell forced to GOLD at loop end        ON / OFF       : {correct_on}/{n_eval}  vs  {correct_off}/{n_eval}")
    print("\n=== synthetic smoke complete ===")
    return model, gamma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--train-steps", type=int, default=600)
    ap.add_argument("--llada", action="store_true", help="also run the LLaDA-8B no-op-at-init port check")
    ap.add_argument("--llada-id", default="GSAI-ML/LLaDA-8B-Instruct")
    args = ap.parse_args()
    smoke(N=args.N, k=args.k, train_steps=args.train_steps)
    if args.llada:
        print("\n=== LLaDA-8B port check (real masked-diffusion LM) ===")
        try:
            ad = LLaDAAdapter(args.llada_id)
            print("  loaded:", args.llada_id)
            r = ad.noop_at_init_check()
            print("  ", r)
        except Exception as e:
            print(f"  LLaDA check skipped/failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
