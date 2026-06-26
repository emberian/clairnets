# Contributing to GLaDOS

Read [`README.md`](README.md) for the bet and [`GLOSSARY.md`](GLOSSARY.md) for the vocabulary first.

## The repo has two halves
1. **The exact harness — pure Python, the ground truth.** `csp.py`, `levels.py`, `fol.py`, `smt.py`,
   `schedule.py`, `curriculum.py`. Runs anywhere (no GPU; `smt` needs `z3-solver`, `curriculum` needs an AWS
   Bedrock key only to *regenerate* data). **Every one is self-checking** — `python -m clair.csp` etc. prints
   PASS. These define what is true; the neural side is *measured against them*.
2. **The neural organs + woven LLM — needs a GPU.** `proposer.py`/`chain_organ.py`/`energy_organ.py` (the
   bank), `augmented.py`/`oracle_readout.py`/`glados_woven.py`/`gen_augmented.py` (the woven LLM). Built and
   run on rented L40S/T4 boxes (OLMo-2-1B is the default host).

## Setup
```bash
git clone https://github.com/emberian/clairnets && cd clairnets
python -m clair.csp        # pure-python ground truth, should print "ALL CHECKS PASS"
python -m clair.levels     # relations × abstraction levels
```
GPU work runs on a Deep-Learning-AMI box (PyTorch prebaked; `pip install peft z3-solver datasets transformers`).
The launch pattern (tmux+inline quoting is flaky — use a script file):
```bash
# write /home/ubuntu/run_X.sh with:  exec >/tmp/X.log 2>&1   then the python call
tmux new-session -d -s X "bash /home/ubuntu/run_X.sh"
```

## Non-negotiable conventions
- **Exact verifier, always.** Every task/rung is graded by an *exact* checker (`clair.csp`, the FOL
  forward-chainer, or z3) — never a learned reward model, never a planted-solution proxy. Soundness lives at
  the check.
- **Witness-first generation.** Sample a solution *first*, emit only facts it satisfies → instances are
  solvable by construction and labels are exact. (See `curriculum.py`, `fol.py`, `smt.py`.)
- **Honest nulls.** A clean negative is a result. Report what *didn't* work and why; the commit history and
  `notes/` are a ledger of honest findings, not a highlight reel.
- **Measure in the right regime.** A classification-readout head is *not* the LM reasoning; the capability test
  is generative + retrained, with **causal ablations** (shuffle/permute/corrupt the organ — does the answer change?).
- **Don't `git stash` on a shared box** (multiple agents share the working tree). Commit conventions: terse,
  factual, say what was found including negatives.

## How to add things
- **A new organ** (4th inference shape, etc.): match the bank pattern — operate on a state tensor derived from
  `clair.csp` factors, train with a *soundness-asymmetric* loss against the exact transformer
  (`exact_dedP` / `forward_chain`), report soundness + completeness + an honest failure mode. Add it to the
  `portfolio.py` verifier-selection harness.
- **A verifier-ladder rung** (new task family): a witness-first generator + an exact `verify(answer)->bool`
  closure + the `Item` record schema, wired into `schedule.py`. Keep a difficulty knob (proof depth, N, range).
- **A new abstraction level / lattice op**: extend `csp.py` (see `factor_*` for the general level-k lattice)
  and add it to the `levels.py` analysis so we know which rungs need it.

## What's open right now
See the README "What we've found" + `notes/future_directions.md` + `notes/codex_arch_review.md`. The live
question: does a *good (pretrained) + necessary (non-shortcuttable)* learned organ become load-bearing for the
LM's generation — i.e. is the **staged** training recipe the way through the co-training wall. Causal-control
tables are the evidence that counts.
