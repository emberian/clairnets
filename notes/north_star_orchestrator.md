# North star: the LM as organ-orchestrator

The endgame for GLaDOS is not "a checked organ the LM *uses*" (single-shot α→organ→γ→answer) but
**"a checked organ the LM *reasons with*"** — the LM runs a chain-of-thought that strategically *interacts*
with the organ across many steps: pose a sub-problem → read the lattice → reduce / compose / decompose →
call again → … → answer. The organ becomes a tool the LM calls repeatedly inside a learned strategic loop;
the *strategy* (which sub-problems, how to chain, when to reduce) is what's learned. Single-shot training
cannot teach this — it needs the loop.

## The spine resolves neuralese-vs-NL
- **NL CoT** → the strategy is interpretable (we read the plan). Good for the paper + the discipline.
- **Neuralese (Coconut-latent) CoT** → more expressive (superposed plans, continuous strategy) but un-checked.
- **Synthesis (our spine, one level up):** the organ-CALLS are the structured, checked anchors (every α/γ +
  the final answer is verified), so **the strategy *between* calls can be as loose / latent / neuralese as it
  wants** — soundness stays a *permission*. The LM thinks however it likes between calls; honesty is enforced
  *at the calls and the answer*. The CoT medium is a knob, not a commitment.

## Training: RLVR over the orchestration
- Verifiable reward on the final answer → the LM *explores* organ-interaction strategies; RL rewards the ones
  that reach truth. "Learn to orchestrate."
- The **organ-as-process-reward** (per-step candidate-set cardinality drop) shapes the *intermediate* calls
  (each call should make progress) — dense signal, not sparse-outcome-only.
- The **reductions + composer ARE the action space**: the CoT picks from {compile-to-faculty, reduce X→Y,
  compose, decode}. Codex's *decode-from-route* (recognize→reduce→solve→decode) is the SINGLE-STEP version;
  the multi-step CoT generalizes it to a strategic sequence.

## Chaining to higher problems (compositional generalization)
A problem too hard for one organ-call is *decomposed* by the LM into sub-problems the organ CAN do (each a
faculty / reduction), solved, and recomposed. LM = meta-reasoner · organ = primitive-solver · reductions = glue.
The payoff: solve N-step problems from 1-step organ-primitives.

## Two prerequisites (or the orchestrator is hollow)
1. **α must actually DRIVE correctness** (the ALPHA_STRUCT crux) — else the multi-step LM is just orchestrating
   a *metadata*-solver (the structure handed in). The SHUFFLED diagnostic measures how much this bites today.
2. **The single-shot organ genuinely good** — the difficulty-coverage fix + the real pretrain. Walk → run.

## Staging
single-shot reasoner (α-driven, hard-occupancy pretrain) → **then** the orchestrator loop:
interactive CoT ↔ organ-calls · a **compositional curriculum** (problems that *require* orchestration, can't be
one-shot) · RLVR over the strategy + organ-process-reward.

## Open research questions
- The CoT medium (NL vs neuralese — interpretability vs expressivity; can be per-experiment).
- The **interface**: how the LM emits an organ-call mid-generation and reads the result back (a tool-call channel
  / the α/γ ports invoked iteratively).
- **Credit assignment**: RLVR over a multi-step organ-interaction (the process-reward helps; ProRL/pass@k controls).
- The **compositional curriculum**: generating problems that *demand* multi-step organ-use (decompose→sub-solve→
  recompose), with held-out depth/composition splits.
- Provenance: a route/provenance channel (source_type · chosen_path · cert_strength · cost · decode_map) so the
  LM's choices are legible + the strategy is auditable (codex).

Everything we're building now — reductions, composer, organ-as-process-reward, decode-from-route, the rich-state
readout — is a brick in THIS. The single-shot pretrain is the foundation; this is the cathedral.
