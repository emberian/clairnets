"""RLVR (RL from Verifiable Rewards) for the augmented OLMo (frozen OLMo-2-1B + learned WRITE head +
differentiable ColorDeductor + abstain head). The verifier is FREE and EXACT — the answer is checked
against the true CSP (enumerated solutions) — so this is the ideal RLVR setup: we train the model on
the thing we actually want (correct answer OR correct abstention, verified), not the brittle
supervised edge-extraction proxy that saturates out-of-distribution.

WHAT IS THE POLICY
  The WRITE head emits a *constraint program*: per-cell candidate-color logits `cand` [B,N,K] and
  symmetric DIFF-edge logits `elog` [B,N,N]. We treat this as a DISTRIBUTION over discrete programs
  and SAMPLE one per rollout:
    * candidate membership  C[b,n,k] ~ Bernoulli(sigmoid(cand))     (per-color alive/dead; the model
      is supervised in SFT with per-color BCE on `cand`, so per-color Bernoulli is the native action
      space, and crucially it keeps cells UNPINNED/multi-candidate so the deductor stays load-bearing.
      A per-cell Categorical pin — `--pin_dist categorical` — is also implemented, but it samples a
      FULL assignment per cell and makes the deductor vestigial; bernoulli is the default & correct.)
    * DIFF edges            E[b,i,j] ~ Bernoulli(sigmoid(elog))      (upper triangle, symmetrized)
  The sampled program is fed to the (still differentiable) ColorDeductor, which narrows it; the query
  cell's marginal is read back. We then sample the final decision:
    * abstain  a ~ Bernoulli(sigmoid(abstain_logit))                (the calibrated-abstention head)
    * if committing, color  y ~ Categorical(query_marginal)         (the readback answer)

WHY THIS GETS GRADIENT TO EVERYTHING TRAINABLE (OLMo frozen)
  log p(action) = log p(C) + log p(E) + log p(a) + (1-a)*log p(y)
  - log p(C), log p(E) depend on `cand`,`elog`  -> grad to the WRITE head (reader, cand_head, edge_*)
  - the deductor params (gamma,bias) + readback + abstain_head enter through `abstain_logit` and the
    query marginal that feeds log p(a) and log p(y) (the sampled program inputs are detached, but the
    deductor's narrowing of them is differentiable) -> grad to the deductor + abstain stack.

ALGORITHM — GRPO (no value model)
  For each problem sample a GROUP of G rollouts, compute each rollout's EXACT reward, normalize within
  the group (advantage = (r-mean)/std), and do the score-function (REINFORCE) update on the sampled
  program's log-prob, + an entropy bonus (anti-collapse) + an optional KL to the SFT-init proposer.
  OLMo's forward is shared across the group: hidden states are computed ONCE per problem; only the
  head/deductor are resampled per rollout (cheap).

REWARD — exact output verification, asymmetric (soundness > completeness):
    answerable & answer==true color  -> +1
    answerable & wrong color         -> -1   (confabulation)
    unanswerable & abstain           -> +1
    unanswerable & commits a color   -> -1   (the unsound case we most want to kill)
    answerable & abstain             ->  0   (incomplete, not unsound)
  "answerable / true color / unanswerable" come from clair.induce's exact backtracking solver (cached
  in each problem dict). `verify_csp` below reconstructs the SAME facts as a clair.csp.CSP and checks
  the answer via csp.solutions, and `--check_verifier` asserts the two agree — so the reward is exact
  and independently auditable against clair.csp.

  python -m clair.rlvr_augmented --smoke                          # tiny end-to-end smoke
  python -m clair.rlvr_augmented --init runs/sft.ckpt --steps 600 # RLVR from an SFT checkpoint
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical

from . import augmented as A
from . import induce as I
from . import run_augmented as R
from . import csp as C   # exact verifier (independent audit path)


# ===================================================================== EXACT verifier (clair.csp)
def problem_to_csp(p, K):
    """Reconstruct the generated problem as a clair.csp.CSP: DIFF facts -> binary !=, IS facts ->
    unary pin constraints. solutions(csp) then gives the exact answerability/true-color, independent
    of induce's own solver (used to AUDIT the cached labels we train on)."""
    cons = []
    for (t, a, b) in p["facts"]:
        if t == 0:  # IS(a, color b) -> unary pin
            cons.append(C._rel((a,), (lambda col: lambda tup: tup[0] == col)(b), K))
        else:       # DIFF(a, b) -> a != b
            cons.append(C._rel((a, b), lambda tup: tup[0] != tup[1], K))
    return C.CSP(p["n"], K, tuple(cons))


def verify_csp(p, K):
    """EXACT verification via clair.csp.solutions: returns (answerable, true_color|None)."""
    sols = C.solutions(problem_to_csp(p, K))
    if not sols:
        return False, None                       # unsat -> unanswerable (must abstain)
    qcols = {s[p["query"]] for s in sols}
    if len(qcols) == 1:
        return True, next(iter(qcols))           # forced -> answerable
    return False, None                           # >=2 colors -> underdetermined -> unanswerable


def check_verifier(k, n_lo, n_hi, n=400, seed=0):
    """Assert the clair.csp verifier reproduces induce's cached (determined, answer) labels exactly."""
    rng = np.random.default_rng(seed)
    n_ok = 0
    while n_ok < n:
        p = I.gen_problem(int(rng.integers(n_lo, n_hi + 1)), k, rng)
        if p["capped"]:
            continue
        det, col = verify_csp(p, k)
        assert det == p["determined"], (det, p["determined"], p["facts"])
        if det:
            assert col == p["answer"], (col, p["answer"], p["facts"])
        n_ok += 1
    return n_ok


# ===================================================================== reward (exact, asymmetric)
def compute_reward(ab, y, abst, ans):
    """Vectorized exact reward over a flat [BG] batch.
       ab: sampled abstain decision (1=abstain). y: sampled color. abst: 1=unanswerable. ans: true color."""
    one = torch.ones_like(ab); zero = torch.zeros_like(ab)
    abst_b = abst > 0.5                                   # unanswerable / must abstain
    abstained = ab > 0.5
    correct = (y == ans)
    commit_r = torch.where(abst_b, -one,                  # commit on unanswerable -> -1 (unsound)
                           torch.where(correct, one, -one))  # commit on answerable -> +1 / -1
    abstain_r = torch.where(abst_b, one, zero)            # abstain: +1 if unanswerable else 0 (incomplete)
    return torch.where(abstained, abstain_r, commit_r)


# ===================================================================== one GRPO group of rollouts
def rollout_group(model, ba, G, pin_dist="bernoulli", sample_color=True):
    """Compute OLMo states ONCE, then sample G discrete programs + decisions per problem. Returns
    flat [BG] tensors: reward, logprob, entropy, abstain decision, plus (h, cand, elog) for the KL."""
    dev = ba["input_ids"].device
    h = model.host_encode(ba)                                 # [B,T,D]  (no grad; OLMo frozen)
    cand, elog, E = model.write(h, ba)                        # [B,N,K],[B,N,N]  grad -> WRITE head
    B, N, K = cand.shape
    qvec = h[torch.arange(B, device=dev), ba["last_idx"]].float().detach()   # [B,D]

    rep = lambda t: t.repeat_interleave(G, 0)
    candG, elogG = rep(cand), rep(elog)
    vmG, qG, qvecG = rep(ba["vmask"]), rep(ba["query"]), rep(qvec)
    abstG, ansG = rep(ba["abst"]), rep(ba["ans"])

    # ---- sample candidate program ----
    if pin_dist == "bernoulli":
        cd = Bernoulli(logits=candG)
        c_samp = cd.sample()                                  # [BG,N,K] in {0,1} (detached)
        alive0 = c_samp * vmG.unsqueeze(-1) + 1e-3
        lp_cand = (cd.log_prob(c_samp) * vmG.unsqueeze(-1)).sum((1, 2))
        ent_cand = (cd.entropy() * vmG.unsqueeze(-1)).sum((1, 2))
    else:  # categorical: one pinned color per cell (deductor becomes ~vestigial; here for ablation)
        cd = Categorical(logits=candG)
        c_idx = cd.sample()                                   # [BG,N]
        alive0 = F.one_hot(c_idx, K).float() * vmG.unsqueeze(-1) + 1e-3
        lp_cand = (cd.log_prob(c_idx) * vmG).sum(1)
        ent_cand = (cd.entropy() * vmG).sum(1)

    # ---- sample DIFF edges (upper triangle, symmetrized) ----
    triu = torch.triu(torch.ones(N, N, device=dev), 1)
    pair = vmG[:, :, None] * vmG[:, None, :]
    upair = triu[None] * pair                                 # valid upper pairs
    ed = Bernoulli(logits=elogG)
    e_full = ed.sample()                                      # [BG,N,N]
    e_up = e_full * triu[None]
    E_samp = (e_up + e_up.transpose(-1, -2)) * pair           # symmetric, padded out
    lp_edge = (ed.log_prob(e_full) * upair).sum((1, 2))
    ent_edge = (ed.entropy() * upair).sum((1, 2))

    # ---- deduce (differentiable in gamma/bias; sampled program is the detached input) ----
    alive = model.ded(alive0, E_samp, vmG)                    # [BG,N,K]

    # ---- decision readback ----
    qa = alive[torch.arange(alive.size(0), device=dev), qG]
    qm = qa / qa.sum(-1, keepdim=True).clamp_min(1e-6)
    peak = qm.max(-1).values
    cent = -(qm * (qm + 1e-9).log()).sum(-1)
    nalive = (qm > 0.1).float().sum(-1)
    stats = torch.stack([peak, cent, nalive], -1)
    marg = alive / alive.sum(-1, keepdim=True).clamp_min(1e-6)
    ws = model.ws_enc(marg)
    qoh = F.one_hot(qG, model.Nmax).float()
    ws = model.ws_ln(ws + qoh.unsqueeze(-1) * model.ws_query[None, None, :])
    rb = model.readback(qvecG, ws, vmG)
    abstain_logit = model.abstain_head(torch.cat([stats, rb], -1)).squeeze(-1)

    # ---- sample decisions ----
    abd = Bernoulli(logits=abstain_logit)
    ab = abd.sample()                                         # 1 = abstain
    lp_ab, ent_ab = abd.log_prob(ab), abd.entropy()
    cold = Categorical(probs=qm.clamp_min(1e-9))
    y = cold.sample() if sample_color else qm.argmax(-1)
    lp_color = cold.log_prob(y)

    r = compute_reward(ab, y.float(), abstG, ansG.float())
    logprob = lp_cand + lp_edge + lp_ab + (1.0 - ab) * lp_color
    entropy = ent_cand + ent_edge + ent_ab
    return {"r": r, "logprob": logprob, "entropy": entropy, "ab": ab,
            "h": h, "cand": cand, "elog": elog}


# ===================================================================== KL to SFT-init proposer
def bernoulli_kl(logit_p, logit_q):
    """KL( Bernoulli(sigmoid(logit_p)) || Bernoulli(sigmoid(logit_q)) ), elementwise (stable)."""
    p = torch.sigmoid(logit_p)
    return (p * (F.logsigmoid(logit_p) - F.logsigmoid(logit_q))
            + (1 - p) * (F.logsigmoid(-logit_p) - F.logsigmoid(-logit_q)))


def proposer_kl(model, ref, h, ba):
    """Mean KL of the current program-proposal distributions to the SFT-init reference, over valid
    cells/colors (cand) and valid upper-triangle pairs (elog)."""
    cand, elog, _ = model.write(h, ba)
    with torch.no_grad():
        cand_r, elog_r, _ = ref.write(h, ba)
    vm = ba["vmask"]; B, N, _ = cand.shape
    triu = torch.triu(torch.ones(N, N, device=cand.device), 1)
    upair = triu[None] * vm[:, :, None] * vm[:, None, :]
    kl_c = (bernoulli_kl(cand, cand_r) * vm.unsqueeze(-1)).sum() / vm.sum().clamp_min(1) / cand.size(-1)
    kl_e = (bernoulli_kl(elog.clamp(-30, 30), elog_r.clamp(-30, 30)) * upair).sum() / upair.sum().clamp_min(1)
    return kl_c + kl_e


# ===================================================================== SFT warmup (reuses run_augmented)
def sft_warmup(model, train_pool, tok, Nmax, k, colors, dev, steps, lr, aux, bs, seed):
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    rng = np.random.default_rng(seed)
    model.train()
    t0 = time.time()
    for s in range(1, steps + 1):
        idxs = rng.integers(0, len(train_pool), bs).tolist()
        ba = R.batch_from(train_pool, idxs, tok, Nmax, k, colors, dev)
        loss, parts = R.losses(model, ba, aux)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0); opt.step()
        if s % max(1, steps // 8) == 0 or s == 1:
            print(f"  [sft] step {s:4d} loss {loss.item():.3f} ({parts})  {time.time()-t0:.0f}s", flush=True)


def head_state(model):
    """state_dict of everything EXCEPT the frozen OLMo host (the trainable heads + deductor)."""
    return {k: v for k, v in model.state_dict().items() if not k.startswith("olmo.")}


# ===================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--G", type=int, default=8)                  # rollouts per problem (group size)
    ap.add_argument("--bs", type=int, default=16)                # problems per GRPO step
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--host", default="olmo2-1b",
                    help="frozen host config: olmo2-1b (default, baseline) | olmo3-7b | raw HF id")
    ap.add_argument("--write_head", choices=["pool", "grounded"], default="pool")
    ap.add_argument("--edge_window", type=int, default=7)
    ap.add_argument("--lr", type=float, default=1e-4)            # lower than SFT; RL is higher-variance
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--kl_coef", type=float, default=0.0)        # KL to SFT init (0 = off)
    ap.add_argument("--adv_eps", type=float, default=1e-4)
    ap.add_argument("--pin_dist", choices=["bernoulli", "categorical"], default="bernoulli")
    ap.add_argument("--sample_color", type=int, default=1)
    ap.add_argument("--init", default=None)                     # SFT checkpoint (head_state) to warm-start
    ap.add_argument("--warmup_steps", type=int, default=400)    # brief SFT if no --init
    ap.add_argument("--warmup_aux", type=float, default=1.0)
    ap.add_argument("--train_n", default="4,6")
    ap.add_argument("--test_n", default="5,8,11")
    ap.add_argument("--pool_size", type=int, default=6000)
    ap.add_argument("--eval_n", type=int, default=512)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check_verifier", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.steps = min(a.steps, 30); a.bs = 6; a.G = 4; a.pool_size = 800
        a.eval_n = 192; a.warmup_steps = min(a.warmup_steps, 40); a.eval_every = 15

    colors = A.COLORS
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    n_lo, n_hi = (int(x) for x in a.train_n.split(","))

    if a.check_verifier or a.smoke:
        nck = check_verifier(a.k, n_lo, n_hi, n=200 if a.smoke else 600)
        print(f"VERIFIER AUDIT: clair.csp reproduces induce labels on {nck} problems (exact). OK", flush=True)
        if a.check_verifier:
            return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = A.resolve_host(a.host)
    print("loading host", a.host, "->", model_id, flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "right"
    olmo = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(dev).eval()
    print(f"host config: {olmo.config.num_hidden_layers} layers, hidden_size {olmo.config.hidden_size}, "
          f"{sum(p.numel() for p in olmo.parameters())/1e9:.2f}B params", flush=True)
    Nmax = max(int(x) for x in a.test_n.split(",")) + 1
    model = A.AugmentedOLMo(olmo, tok, Nmax, a.k, T=a.T,
                            write_head=a.write_head, edge_window=a.edge_window).to(dev)
    print(f"trainable params {A.n_trainable(model):,}  Nmax={Nmax}  dev={dev}  write_head={a.write_head}", flush=True)

    gap = A.verify_flamingo_noop(olmo, tok, dev)
    print(f"FLAMINGO GATE NO-OP (zero-init gated adapter, max|base-gated|, ~0 expected): {gap:.3e}", flush=True)

    # ---- build pools (same recipe as run_augmented for comparability) ----
    prng = np.random.default_rng(a.seed + 999)
    t0 = time.time()
    train_pool = R.build_pool(n_lo, n_hi, a.k, a.pool_size, prng)
    eval_pools = {"train": R.build_pool(n_lo, n_hi, a.k, a.eval_n, prng)}
    for tn in (int(x) for x in a.test_n.split(",")):
        eval_pools[tn] = R.build_pool(tn, tn, a.k, a.eval_n, prng)
    print(f"pools built in {time.time()-t0:.0f}s", flush=True)

    # ---- warm-start: load SFT ckpt or run a brief SFT warmup ----
    if a.init:
        sd = torch.load(a.init, map_location=dev)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded init {a.init}  (missing={len(missing)} unexpected={len(unexpected)})", flush=True)
    else:
        print(f"no --init: running {a.warmup_steps} SFT warmup steps", flush=True)
        sft_warmup(model, train_pool, tok, Nmax, a.k, colors, dev,
                   a.warmup_steps, lr=3e-4, aux=a.warmup_aux, bs=24, seed=a.seed)

    # ---- reference for KL (snapshot of the SFT-init heads; shares the frozen OLMo) ----
    ref = None
    if a.kl_coef > 0:
        ref = A.AugmentedOLMo(olmo, tok, Nmax, a.k, T=a.T,
                              write_head=a.write_head, edge_window=a.edge_window).to(dev)
        ref.load_state_dict(model.state_dict())
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()

    def eval_all(tag):
        out = {}
        for key in ["train"] + [int(x) for x in a.test_n.split(",")]:
            out[str(key)] = R.evaluate(model, eval_pools[key], tok, Nmax, a.k, colors, dev, n=a.eval_n)
        print(f"\n=== {tag} (greedy forward) ===", flush=True)
        for key in ["train"] + [int(x) for x in a.test_n.split(",")]:
            r = out[str(key)]
            print(f"  N={str(key):>5}  overall {r['overall']*100:4.1f}  detAcc {r['det_acc']*100:4.1f}"
                  f"  abstP/R {r['abstain_prec']*100:3.0f}/{r['abstain_rec']*100:3.0f}"
                  f"  progEx {r['prog_exact']*100:3.0f}", flush=True)
        return out

    start_eval = eval_all("SFT-START (pre-RLVR)")

    # ===================================== GRPO loop =====================================
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    rng = np.random.default_rng(a.seed + 7)
    log = []
    model.train()
    t0 = time.time()
    run_rew = run_wrong = None
    for s in range(1, a.steps + 1):
        idxs = rng.integers(0, len(train_pool), a.bs).tolist()
        ba = R.batch_from(train_pool, idxs, tok, Nmax, a.k, colors, dev)
        out = rollout_group(model, ba, a.G, pin_dist=a.pin_dist, sample_color=bool(a.sample_color))
        r, lp, ent, ab = out["r"], out["logprob"], out["entropy"], out["ab"]
        r2 = r.view(a.bs, a.G); lp2 = lp.view(a.bs, a.G); ent2 = ent.view(a.bs, a.G)
        adv = (r2 - r2.mean(1, keepdim=True)) / (r2.std(1, keepdim=True) + a.adv_eps)
        pg = -(adv.detach() * lp2).mean()
        ent_loss = -a.ent_coef * ent2.mean()
        loss = pg + ent_loss
        kl_val = 0.0
        if ref is not None:
            kl = proposer_kl(model, ref, out["h"], ba)
            loss = loss + a.kl_coef * kl
            kl_val = float(kl.detach())
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        opt.step()

        mr = float(r.mean()); wr = float((r < -0.5).float().mean()); abr = float((ab > 0.5).float().mean())
        run_rew = mr if run_rew is None else 0.9 * run_rew + 0.1 * mr
        run_wrong = wr if run_wrong is None else 0.9 * run_wrong + 0.1 * wr
        if s % max(1, a.steps // 20) == 0 or s == 1:
            print(f"  step {s:4d}  reward {mr:+.3f} (ema {run_rew:+.3f})  wrong-return {wr*100:4.1f}%"
                  f" (ema {run_wrong*100:4.1f})  abstain {abr*100:3.0f}%  ent {float(ent.mean()):.2f}"
                  f"  kl {kl_val:.3f}  gnorm {float(gn):.2f}  {time.time()-t0:.0f}s", flush=True)
            log.append({"step": s, "reward": mr, "wrong_return": wr, "abstain_rate": abr,
                        "entropy": float(ent.mean()), "kl": kl_val, "grad_norm": float(gn)})
        if a.eval_every and s % a.eval_every == 0:
            ind = R.evaluate(model, eval_pools["train"], tok, Nmax, a.k, colors, dev, n=min(192, a.eval_n))
            print(f"    [eval@{s}] in-dist overall {ind['overall']*100:4.1f}  detAcc {ind['det_acc']*100:4.1f}"
                  f"  abstP/R {ind['abstain_prec']*100:.0f}/{ind['abstain_rec']*100:.0f}", flush=True)
            model.train()

    final_eval = eval_all("RLVR-FINAL (post-RLVR)")

    # ---- decisive deltas: did training-on-the-verifiable-objective break the extraction wall? ----
    print("\n=== RLVR vs SFT-START delta (OOD is the decisive read) ===", flush=True)
    for key in ["train"] + [int(x) for x in a.test_n.split(",")]:
        s0, s1 = start_eval[str(key)], final_eval[str(key)]
        print(f"  N={str(key):>5}  detAcc {s0['det_acc']*100:4.1f}->{s1['det_acc']*100:4.1f}"
              f"  ({(s1['det_acc']-s0['det_acc'])*100:+4.1f})   "
              f"abstRec {s0['abstain_rec']*100:4.1f}->{s1['abstain_rec']*100:4.1f}"
              f"  ({(s1['abstain_rec']-s0['abstain_rec'])*100:+4.1f})   "
              f"overall {s0['overall']*100:4.1f}->{s1['overall']*100:4.1f}"
              f"  ({(s1['overall']-s0['overall'])*100:+4.1f})", flush=True)

    out_path = a.out or os.path.join(os.path.dirname(__file__), "..", "runs", "rlvr_augmented.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({"args": vars(a), "host": model_id, "gate_noop_gap": gap, "log": log,
               "start_eval": {k: v for k, v in start_eval.items()},
               "final_eval": {k: v for k, v in final_eval.items()}},
              open(out_path, "w"), indent=1)
    torch.save(head_state(model), out_path.replace(".json", ".ckpt"))
    print(f"\nwrote {out_path}  (+ .ckpt)", flush=True)


if __name__ == "__main__":
    main()
