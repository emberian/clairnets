# Stage C Woven-Organ Experiments

Stage C adds a selectable causal path from the narrowed lattice workspace back into the frozen host
LLM. The old terminal path is still available.

## What Changed

`clair.augmented.AugmentedOLMo` now supports:

```text
read_source=terminal   # old path: host -> write -> deductor -> terminal readout
read_source=woven      # host -> write -> deductor -> second host pass with workspace write-back
read_source=fused      # terminal logits + woven-host logits
```

The woven path installs zero-init tanh-gated cross-attention adapters at selected host layers:

```text
hidden_tokens <- hidden_tokens + tanh(alpha) * CrossAttn(hidden_tokens, lattice_workspace)
```

At initialization `alpha=0`, so installing the adapters is an exact no-op. OLMo remains frozen in
this first build; only the write head, deductor, read heads, and woven adapters train.

## Controls

Run the old terminal baseline:

```bash
python -m clair.run_augmented --read_source terminal --readout setvalued --write_head grounded
```

Run the prompt-only frozen-host readout control. This uses a second host pass but no lattice
write-back layers, so it tests whether a trainable head on frozen OLMo hidden states can do the task
without causal organ influence:

```bash
python -m clair.run_augmented --read_source woven --woven_layers "" --readout setvalued --write_head grounded
```

Run the causal woven readout:

```bash
python -m clair.run_augmented --read_source woven --woven_layers 5,10,14 --readout setvalued --write_head grounded
```

Run the fused causal readout:

```bash
python -m clair.run_augmented --read_source fused --woven_layers 5,10,14 --readout setvalued --write_head grounded
```

For a quick smoke:

```bash
python -m clair.run_augmented --smoke --no_base --read_source fused --woven_layers 5,10,14 --readout setvalued --write_head grounded
```

## Decisive Read

The useful comparison is not only absolute accuracy. Compare:

- `terminal` vs `woven_layers ""`: does a prompt-only frozen-host head outperform the deductor path?
- `woven_layers ""` vs `woven_layers 5,10,14`: does the lattice workspace causally improve host-state
  readout?
- `terminal` vs `fused`: does woven host feedback add useful information without degrading
  soundness?

Primary metrics:

- `SND`: reported set contains every true survivor;
- `USE`: sound and informative, not just full-domain unknown;
- `setDet`: exact singleton on determined queries;
- `edgeF1` / `progEx`: whether gains come from extraction or from downstream readout.

## Not Yet Done

RLVR still uses the terminal rollout path. A woven RLVR rollout needs a second host pass per sampled
program, because each sampled program gives a different workspace. That is implementable, but it is
a separate cost/performance experiment.
