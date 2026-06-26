# Codex (GPT-5.5) architecture review — D (dense LDT-faithful weave)

Verdict: **D is directionally right on the bandwidth problem, but not approvable as stated.**
"Unstructured dense alpha/gamma surgery into pretrained OLMo is a high-risk overcorrection that can
produce a NEW bypass: the model encodes the answer through the lattice-shaped tensor without learning
stable deduction."

Key corrections:
1. **Full-bandwidth STRUCTURED, not full-dense.** Flatten `H->[N,K]` is anti-equivariant + max-shape-tied →
   wrecks size-gen. Keep factor-graph/equivariant adapters over explicit cell-candidate AND factor-tuple tensors.
2. **alpha must COMPILE, not SOLVE.** alpha→candidate-survival-logits = "answer posterior as lattice" = a bypass.
   alpha emits a TYPED PROBLEM PROGRAM + initial evidence; the ORGAN narrows.
3. **Factors/relations first-class state** — one sigmoid/candidate isn't enough for modular arithmetic/XOR/clauses.
4. **GA grades are NOT a soundness fix** — completeness/representation bias only. Soundness only from
   output-verification + elimination CERTIFICATES + typed symbolic kernels (e.g. modular arithmetic).
5. **Organ bank shares a PROTOCOL, not a coupling.** narrow(shrink)/chain(grow)/energy(non-monotone) need
   distinct state schemas+halting+verifiers. Don't trust the LM router alone — portfolio + verifier-selection.
6. **Don't backprop through repeated full OLMo passes.** Organ recurrence BETWEEN layers, supervise every step,
   truncated/stop-grad on old lattice states. "The host should not be the recurrence."
- Shortcut vs iterate: ITERATE is primary; shortcuts = checked speculative acceleration only.
- Genuinely-new claim (narrow): a checked, recurrent lattice/factor reasoning workspace causally woven into a
  pretrained open LM so the normal LM head generates from it, verifier-trained + causal-ablation-validated.

**Highest-leverage change (DE-RISK BEFORE building organs):**
oracle narrowed lattice -> structured gamma -> OLMo+LoRA -> LM-head answer, with batch-shuffle /
candidate-permute / corrupt-cell controls. If ablation still doesn't bite → readout/alignment is the wall, not
deduction. If it bites → coupling works, then add the learned organ. (Running now: clair/oracle_readout.py.)

Prior art it's adjacent to: LDT (lattice projection/recurrent narrowing/alpha targets/abstention), SATNet
(differentiable solver layer + grounding caution), TransNAR (closest: Transformer reads specialist reasoner
embeddings), Universal Transformers/PonderNet/DEQ (recurrence/adaptive/equilibrium), DeepProbLog/NTP.
