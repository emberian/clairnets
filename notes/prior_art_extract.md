# Prior-art extract: TransNAR, RLTT, LSRL → implementable recipes for GLaDOS

Source PDFs in `pdfs/`. This is an *implementation* extract (mechanisms + equations + section refs),
not a lit review. GLaDOS context: README.md (the "D" weave: α compiles hidden→typed factor-graph
program; organ narrows recurrently *between* OLMo layers; γ writes certified state densely back
through a zero-init gate; exact verifier → reward) + notes/rlvr_landscape.md (organ-RL needs a
process/trajectory-reward arm or GRPO falsely nulls it).

---

## 1. TransNAR — "Transformers meet Neural Algorithmic Reasoners" (2406.09308)

Our closest architectural cousin: an LLM coupled to a *pretrained, frozen* GNN reasoner. Decoder-only
Chinchilla Transformer (6 layers, 70M, ctx 2048) + a Triplet-GMPNN NAR pretrained on CLRS-30 graphs.

### Coupling mechanism (§3, Fig. 3) — the part we lift
Two parallel streams updated layer-by-layer:
- **Transformer self-attention** (Eq. 1): standard layer over token reps `T^(t)`, producing `Θ^(t+1)`.
- **NAR step** (Eq. 2): a *shared-weight* max-MPNN (really Triplet-GMPNN w/ triplet interactions +
  gating), `g_u^(t+1) = φ(g_u^(t), max_v ψ(g_u^(t), g_v^(t)))`. No timestep index on the learnable
  params — iterative, same function each step, mirroring algorithmic computation.
- **Cross-attention readback** (Eq. 3): the *only* coupling, and it is **one-directional** — tokens
  query, graph supplies keys/values:
  `T^(t+1) = FFN( softmax( (Θ^(t) Q_l^×)ᵀ G^(t) K_l^× / √d_k ) G^(t) V_l^× )`.
  Q from Transformer tokens; K,V from NAR **node** embeddings `G`. **No transformation is applied to
  G**, and **the graph never attends back to the tokens** ("no cross-attention is performed by the
  graph embeddings") — precisely *because* the NAR is frozen and back-gradients would wreck its
  robustness.
- **Nodes AND edges** (§4 "Combining cross-attention contributions"): they cross-attend over node
  feats `G` *and* edge feats `E ∈ R^{N×N×k}` (flatten first two axes), then **combine by
  concatenation + a linear layer**. They tried sum, a 2-layer MLP, and Gram–Schmidt orthogonalisation
  of the two contributions — **none beat plain concat+linear.**
- **Gated, Flamingo-style** (§3 caption): cross-attention layers are *interleaved* through the stack,
  and the gate is **initialised closed** "to fully preserve the language model's internal knowledge at
  the beginning of training." This is a **zero-init gate**.

### Pretrain + freeze (§3 end, §4 "Pre-training the NAR")
- NAR pretrained standalone on CLRS-30 graphs, multi-task, problem sizes up to 16 (train [4,8],12),
  following Ibarz et al. — yields OOD generalisation to **~4× larger** inputs *in graph space*.
- **The NAR is kept FROZEN during TransNAR fine-tuning.** Explicit rationale: "additional gradients
  would eliminate the model's original robustness properties." → direct vindication of bootstrap→freeze.
- Transformer init ablated two ways: **"Pretrained"** (LM-pretrained, = fine-tuning) and **"Untrained"**
  (random) — "we recover the same experimental findings even if the LM is randomly initialised."

### Objective + training (§4)
- **Pure supervised next-token prediction** on output text (prediction head off final layer). *No RL.*
  7 epochs, batch 256, Adam 1e-4, **randomized positional encoding** (max len 8192) on top of RoPE.
- Data: CLRS-Text (text rendering of CLRS-30), 2.4M points, 70/30 split. Train sizes [4,8] and 12;
  eval **10 (OOD interpolation), 12 (ID), 14 (OOD extrapolation)**. Metrics: shape / parse / CLRS
  (=% output elements matching ground truth) scores.

### OOD results + what drives them (§4.1, Fig. 1/4/5)
- "**over 20% absolute improvement** in several algorithm classes" OOD; TransNAR beats the Transformer
  on most algorithms ID and OOD, and **causes emergence where the baseline is at zero/near-zero.**
- Driver: the NAR's *graph-space* OOD robustness transferred into text via cross-attention; biggest
  visible win is **shape correctness** (grounding fixes the input-size→output-size dependency).
- **Randomized PE is load-bearing** (§6): *before* it, the hybrid's OOD perf was **thresholded by the
  base LM** — "if the base LLM achieved near-zero performance, the hybrid architecture would fatally
  share the same fate." With randomized PE the hybrid is good *even when the base LM is near-zero.*
- **Failure modes:** searching-for-an-index tasks (binary search, find-max-subarray, minimum,
  quickselect) — cross-attention can't generalise to *novel index boundaries* (index-hints suggested as
  fix). Parse score also dips on a few algorithms in extrapolation due to **insufficient cross-attention
  capacity to decode the NAR's outputs** (§4.1, App. 7) — a *readback-bandwidth* bottleneck. They even
  float **progressive decoding**: early cross-attn layers decode from *earlier* message-passing steps,
  later layers from *later* steps.
- **Limitation (§4.2):** requires *both* text and a ground-truth graph representation. Future = distill
  TransNAR into a unimodal Transformer.

### Lift vs. diverge for GLaDOS
| TransNAR | GLaDOS analogue | verdict |
|---|---|---|
| Gated cross-attn, **gate init closed** | zero-init γ gate | **adopt — direct precedent** |
| Read **node + edge** structure, **concat+linear** (pooling/sum lose) | structured dense-γ readback | **adopt — supports dense-readback finding over a pooled summary** |
| **NAR frozen** during coupling | bootstrap organ → freeze → weave | **adopt — vindicates staged recipe** |
| **Randomized PE** decouples hybrid from base-LM OOD ceiling | mitigates LM-shortcut in cold-weave | **adopt** |
| Pure **supervised next-token** suffices for OOD transfer | γ-readout warmup (already proven on oracle lattice) | **adopt as warmup; RL only for calibration** |
| Coupling is **one-directional** (LLM reads organ). NAR's *input* is a clean external symbolic graph — **the LLM never has to produce it.** | GLaDOS's **α** (LLM hidden → typed factor-graph program) is exactly the input-production step TransNAR sidesteps | **KEY DIVERGENCE — see flag #1** |

---

## 2. RLTT — "Reward Latent Thought Trajectories" (2602.10520)

Process/trajectory reward for **looped** LMs (base = Ouro LoopLM, the only open one). A LoopLM applies
shared-weight blocks `T_max` times before emitting each token. Drop-in **GRPO replacement, no external
verifier, negligible compute.**

### The mechanism (§3)
- `h_j^(t)` = hidden for token `j` after loop `t`; `g` = the **shared** LM head.
- Standard policy uses *only* the terminal distribution `P^(T_max)_θ(y_j) = softmax(g(h_j^(T_max)))`
  (Eq. 1). GRPO/REINFORCE therefore credits **only the last latent state** (Eq. 3) → the credit-
  assignment mismatch (Fig. 1 left).
- **Key trick:** decode the *same* LM head at *every* loop to get an intermediate "latent thought
  distribution" `P^(t)_θ(y_j) = softmax(g(h_j^(t)))`, `t=1…T_max-1` (Eq. 2).
- **RLTT policy gradient** (Eq. 4/6): replace the single terminal log-prob with a weighted sum over
  loops:
  `∇J = E[ (1/g) Σ_i (1/|y_i|) Σ_j Σ_t ω_t ∇log P^(t)_θ(y_{i,j}) · Â_i ]`,
  with `ω_t ≥ 0`, `Σ_t ω_t = 1`. **Â_i is the *group-normalized outcome* advantage** (Eq. 3 form:
  `(r_i − mean)/std`) — i.e. **the same scalar outcome reward is distributed across loop steps by ω_t.**
  No process reward model. Reward is just binary correctness `r∈{0,1}`.
- KL to a frozen ref, **computed on the terminal loop only** (Eq. 5/7): `J = J_PG + β·D_KL(π_θ‖π_ref)`.

### Loop-weighting strategies (§4.1)
- **Exit-PDF:** `ω_t = p_exit(t|x)` — use the model's own learned **early-exit head probability** as the
  per-loop weight (used in the main experiments: "leverage Ouro's halting signal as a proxy for internal
  confidence — loops with lower exit probability are less reliable and get proportionally less credit").
- **Progressive:** `ω_t = t^α / Σ s^α` (later loops weighted more).
- **Uniform:** `ω_t = 1/T_max`.
- **App. A.3: results are largely *insensitive* to the weighting choice** — the gain comes from exposing
  the *whole* trajectory, not from a tuned schedule. → don't over-engineer ω.

### Algorithm 1 + cost (§4.2)
Sample `g` rollouts → outcome rewards → group advantages → for each rollout/token/loop record per-loop
log-prob `ℓ = log P^(t)` and the terminal KL → assemble `J` → optimizer step. **Compute overhead ≈ 0**
(per-loop logits already exist; the weighted sum is linear in `T_max`). **Cost is MEMORY** — must retain
per-loop log-probs; they halved `ppo_max_token_len_per_gpu` to 8192 and added mini-steps.

### Results (§5, Table 2)
Trained on **MATH only.** Gains over GRPO — Ouro-1.4B: **+7.0 MATH-500, +16.6 AIME24, +10.0 AIME26,
+10.0 BeyondAIME** (avg math 39.6 base → 41.7 GRPO → **46.0 RLTT**). Ouro-2.6B: avg math **42.5 GRPO →
51.2 RLTT**. Transfers zero-shot to non-math (ARC-C, MMLU-ST, GPQA, MBPP); GPQA nearly doubles. Stat-sig
(p<0.05) on 7/9 (1.4B) and 8/9 (2.6B) benchmarks.

### Why (§6) + actionable takeaway
RLTT converges to **shorter** responses (Fig. 3) with controlled entropy decrease; richer **GSNR**
especially on the hardest benchmarks; **App. A.6: RLTT beats GRPO at every loop count, with the largest
margins in the 1–2-loop regime** — i.e. it improves *early* latent reasoning. Theory A.5/A.10:
trajectory credit ⇒ weakly shorter optimal decode + robustness under tight budgets.

**Takeaway for GLaDOS:** the organ's recurrent narrowing *between* OLMo layers is exactly a latent loop.
To credit its internal steps under GRPO **without a verifier and at ~zero compute**: decode an answer
distribution via γ→LM-head at *each* narrowing step, weight each step's log-prob by `ω_t`, and broadcast
the *single* exact-verifier outcome reward across steps. Natural `ω_t` = the organ's own
**completeness/abstain (exit) signal** — GLaDOS's completeness variable *is* Ouro's exit head.

---

## 3. LSRL — "Process-Supervised GRPO on Latent Recurrent States" (Findings EMNLP 2025)

Process supervision for **recurrent-depth** LMs (base = Huginn-3.5B; Prelude–Core–Coda, Core looped
`r` times, params fixed). Where RLTT broadcasts one outcome reward, LSRL builds a **genuine per-step
reward by decoding and grading each latent state.**

### Mechanism (§3)
- Huginn recurrence (Eq. 1): `s_{k+1} = R_θ([e; s_k])`, `k=0…r-1`; LM head `p(y) = softmax(W_o s_r)` (Eq. 2).
- **Full-depth decoding** (Eq. 3): each intermediate state `s_k` is **autoregressively decoded into a
  textual snapshot** `ŷ_k = AutoregressiveDecode(s_k)` using the Coda + shared LM head — exposing `r`
  text snapshots (complete sentences/paragraphs, not token fragments).
- **Per-step reward** (Eq. 4): a lightweight **GPT-4.1-nano grader** scores each snapshot for **Internal
  Quality (IQS)** = logical consistency/clarity, and **Math Progress (PS)** = does the step meaningfully
  advance (reduce unknowns, apply an op, simplify). `R_k = w_IQS·q̂_k + w_PS·p̂_k`, `w=0.5/0.5`,
  min–max-normalized within the GRPO group.
- **Process reward** = discounted sum (Eq. 5): `R_proc = Σ_{k=1}^r γ^{k-1} R_k`, **γ=0.99**.
- **Total reward** (Eq. 6): `R_tot = w_f·1[ŷ_final=y*] + w_p·R_proc`, **`(w_f,w_p)=(0.7,0.3)`** (outcome-dominant).
- Plug into **standard GRPO** (Eq. 7/8): `A_i = R_i − R̄` (group mean, critic-free), clipped PPO surrogate
  ε=0.2, adaptive KL β→0.1. `G=8` rollouts/prompt.

### Implementation specifics (§3.3, App. A/B)
- **One-pass hidden-state cache:** cache `{s_k}` in a single forward unrolling of Core, decode all depths
  from the cache → **avoids r-fold re-execution, ~50% FLOP saving** vs naive recompute.
- **LoRA rank-8 (α=16) on the 16 Core projection matrices only** (Q/K/V/O + MLP up/down); Prelude/Coda
  frozen → **0.17% trainable params**, single L40S, int8/QLoRA. lr 2e-6 const, AdamW, grad-accum 4.
- Graders A/B (quality/progress) called **twice and averaged** per trajectory; Grader C checks final
  answer (full prompts in App. B; temp 0, JSON output).

### Results (§5, Table 3)
SFT-r8 baseline → **LSRL: GSM-8K 13.49→17.76 (+4.27), MATH 5.61→6.94 (+1.33), MathQA 24.07→26.13
(+2.06).** Recovers **~75% of depth-32's accuracy at 25% of the recurrent compute (1.0× vs 4.0× FLOPs).**
**Source decomposition:** outcome-only RL gives **+1.05pp**; process supervision adds **+3.22pp ≈ 75% of
the total lift.** Qualitatively (Fig. 1) LSRL has correct computations as early as depth 1 while SFT
drifts off-topic.

### Critical caveats
- **Shallow-recurrence null (§5.3, Table 4):** at **r=4** process supervision has *no* effect (flat ~8%
  GSM) — "a minimum threshold of recurrent depth is necessary before process supervision can take effect."
  Needs **r≥8.**
- **Residual reward-hacking** persists (§5.2): prompt-echo, irrelevant "fluff", unit/type mix-ups — partly
  because the **GPT-nano judge over-estimates unit-mismatched arithmetic, letting the policy game the
  reward early.** Future work explicitly asks for a **symbol-aware PRM** that *verifies* each sub-step.

**Takeaway for GLaDOS:** LSRL is GLaDOS's organ-RL test with the wrong grader. Swap the hackable,
expensive GPT-nano judge for GLaDOS's **exact checked deductor**: per-step **PS** = candidate-set
cardinality drop (narrowing progress); per-step **IQS** = soundness of the narrowing (the check). GLaDOS
gets LSRL's dense per-step reward *for free, exact, and unhackable* — and LSRL's stated future-work
("symbol-aware PRM verifying each step") **is literally the GLaDOS deductor.**

---

## 4. What we adopt

### (a) Coupling design for α/γ (from TransNAR)
1. **Zero-init gated cross-attention readback** is the right γ primitive — TransNAR inits the gate closed
   to protect LM knowledge; matches our zero-init γ. **Supports the dense-readback finding directly:**
   reading the organ's **structured** state (nodes *and* edges) via concat+linear **beats** every
   reductive pooling (sum / MLP / orthogonalisation) they tried. Don't summarize the lattice — read it
   densely and structurally.
2. **Freeze the organ during weaving.** TransNAR keeps the NAR frozen and *recovers the same findings
   even with a randomly-initialised LM* — strong precedent for "organ to excellence → freeze → weave,"
   and for our finding that **RLVR fixes calibration, not capacity** (their OOD jump comes with *no* RL).
3. **Randomized positional encoding** — without it the hybrid is *thresholded by the base LM's* OOD
   score. This is a candidate cause of our cold-weave shortcut (the LM dragging the organ down); cheap to
   add.
4. **Multi-depth readback** (their "progressive decoding"): let earlier cross-attn read earlier
   narrowing steps, later read later — dovetails with γ being queryable at *every* organ step (needed for
   (b) anyway).
5. **But α is ours alone.** TransNAR's organ ingests a *clean external symbolic graph*; the LLM never
   compiles it. Our α (hidden→typed factor-graph program) is the harder, un-precedented half. TransNAR
   validates γ strongly and says ~nothing about α → **pretrain α supervised (text→program) standalone**,
   exactly per the staged recipe; do **not** expect α to emerge from co-training.

### (b) Process-reward recipe for the organ-RL test (RLTT ⊕ LSRL)
Use **both arms**, in increasing strength — and *use the published recipe instead of hand-rolling*:

- **Arm 1 — RLTT-style, parameter-free (cheap baseline process arm).** Decode an answer distribution via
  γ→LM-head at each organ-narrowing step `t`; RLTT Eq. 6 gradient
  `Σ_t ω_t ∇log P^(t)(answer)·Â`, Â = group-normalized **exact-verifier** outcome reward, KL on terminal
  step only. `ω_t` = **organ completeness/abstain signal** (= RLTT's Exit-PDF; our completeness *is* the
  exit head). Insensitive to ω (A.3), so don't tune it. Cost: memory for per-step logprobs (halve token
  packing, add mini-steps).
- **Arm 2 — LSRL-style, exact-graded (the strong, novel arm).** Decode each narrowing step to a
  snapshot, but **grade with the checked deductor, not an LLM**: per-step `R_k = w_IQS·(soundness) +
  w_PS·(Δ candidate-set cardinality)`, discounted `R_proc = Σ γ^{k-1} R_k` (γ=0.99),
  `R_tot = 0.7·1[correct] + 0.3·R_proc`, into standard GRPO. Reuse LSRL's **one-pass hidden-state cache**
  (decode all depths from a single forward — no re-execution) and **rank-8 LoRA on the organ/interface
  projections only** (their 0.17%-params, single-L40S recipe transfers).
- **Concrete credit for the organ's internal narrowing:** the per-step signal = *monotone candidate-set
  narrowing* — cardinality reduction at step `t` is "progress," and the lattice/check guarantees the
  reduction is sound ("quality"). That is an **exact** version of LSRL's PS/IQS with no judge.
- **Honor the depth floor:** LSRL is *dead at r=4*, alive at r≥8; RLTT's biggest wins are the low-loop
  regime. Give the organ **enough narrowing steps** for the per-step signal to differentiate, and report
  the loop-level curve (RLTT App. A.6).
- This is exactly the "+1 process/trajectory-reward arm" rlvr_landscape.md prescribes so a plain-GRPO
  null stays interpretable.

### (c) Honest novelty delta — what GLaDOS does that none of them do
- **A sound, checked deductor.** TransNAR's NAR is a *soft* CLRS-pretrained net (can be wrong OOD; their
  index/search failures are exactly that). RLTT has *no* verifier. LSRL's grader is a *hackable* LLM
  (documented reward-hacking: prompt-echo, fluff, unit/type errors). GLaDOS's **output check makes the
  organ sound** — soundness as a *permission* to be loose, which none of these have.
- **Exact verifiable reward + bank.** Our verifier ladder replaces LSRL's GPT-nano grader (no overhead,
  no hacking) *and* supplies the exact per-step progress RLTT lacks. LSRL's own future-work ask — a
  "symbol-aware PRM verifying each sub-step" — **is the GLaDOS deductor.**
- **A per-problem compiled typed program**, not a fixed pretrained executor (TransNAR) nor just deeper
  looping of the *same* weights (Ouro/Huginn). α compiles a problem-specific factor graph; the organ
  narrows *that*. Closest structural novelty.

---

## 5. Flags — where these CONTRADICT or COMPLICATE our plan

1. **TransNAR succeeds with pure supervised next-token and a *one-directional, clean-input* organ — no
   RL, no α.** Our cold-weave failure may be over-determined: we demand α (LLM produces the organ's
   input) *and* RL *and* a learned organ, all at once. TransNAR says: a supervised readback through a
   *frozen, clean-input* organ already transfers OOD. **Implication:** lead with a supervised γ-readback
   warmup (already proven on the oracle lattice) and use RL only for calibration — consistent with our
   staged recipe and our "RLVR fixes calibration not capacity" finding. Don't assume RL is what unlocks
   the organ; the *interface*, frozen-organ, and randomized-PE may be.

2. **γ must be readable mid-narrowing, not just terminally.** RLTT/LSRL both rely on decoding the LM head
   at *every* latent step. GLaDOS's organ runs *between* OLMo layers, so we must wire γ→LM-head to emit a
   well-formed answer distribution at *each* narrowing depth (RLTT) and/or a gradable snapshot (LSRL).
   This is an **architecture requirement**, not a free add-on — design γ to emit per-step from the start.

3. **Minimum-depth threshold (LSRL §5.3): process supervision is *null* at r=4.** If the organ converges
   in too few narrowing steps, per-step reward won't differentiate and the process arm will (correctly)
   look like a null. Budget enough narrowing depth; treat a low-depth process-RL null as expected, not as
   evidence against the organ (mirrors rlvr_landscape finding #1).

4. **RLTT memory cost.** Retaining per-step log-probs caps token packing (they halved it). With our
   non-vLLM, dynamic-depth organ loop (rlvr_landscape route C), budget for this; it compounds.

5. **Readback bandwidth is a real bottleneck (TransNAR §4.1, App. 7).** Their parse score dips in
   extrapolation from "insufficient cross-attention capacity to decode the NAR's outputs," and
   index/search tasks fail on *novel index boundaries*. We already found structured dense-γ solves the
   latch-gap — TransNAR corroborates that the readback is where capacity must go — **but warns that
   pointer/index-style answers may stay hard even with a correct organ.** Watch arithmetic/index tasks
   (our "affine wall" rhymes with their index-boundary wall).

6. **Outcome-dominant mixing.** Both LSRL (0.7 outcome / 0.3 process) and RLTT (terminal-only KL) keep
   the *outcome* primary and the process signal a *shaping* term. Don't let an exact per-step organ reward
   dominate and induce step-reward-hacking of its own — keep the verifiable final-answer reward in charge.
