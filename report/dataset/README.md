---
license: mit
language:
- en
task_categories:
- question-answering
- text-generation
tags:
- reasoning
- constraint-satisfaction
- synthetic
- verifiable
- neuro-symbolic
pretty_name: GLaDOS Reasoning Corpus
configs:
- config_name: default
  data_files:
  - split: train
    path: train.parquet
  - split: val
    path: val.parquet
  - split: ood_n
    path: ood_n.parquet
  - split: ood_phrasing
    path: ood_phrasing.parquet
  - split: ood_relation
    path: ood_relation.parquet
---

# GLaDOS Reasoning Corpus

A reproducible, **exact-verified** corpus of natural-language constraint-reasoning problems for
training and evaluating reasoning systems — in particular the *organ* (a lattice-deduction module
woven between an LM's layers). Every problem is a finite constraint-satisfaction problem (CSP)
generated **witness-first** (a solution is sampled before any fact is emitted, so it is always
solvable), then **rendered into varied natural English** by a rule-based, non-stilted renderer, and
the answer is the cell's value forced by the **exact per-cell deductor** of `clair.csp`.

The corpus is the fuel for the "general organ at scale" line: the exploitation experiment showed the
organ only becomes load-bearing on **non-shortcutable, verifiable** data, which is exactly what this
is — answers require multi-step propagation, are read off an exact solver (never a planted guess),
and every record is re-checkable from the structured CSP it ships with.

## Why it's trustworthy

- **Exact by construction.** The ground-truth solution exists *before* the prose; the rendered text
  is emitted directly from the structured facts. There is nothing to "reconstruct."
- **Re-verified, twice.** At generation each label is re-derived by the exact deductor
  (`clair.hard_tasks.fast_dedP`, verified equal to `clair.csp.exact_dedP`). Then **every record is
  re-verified again from its serialized `cons` field** — rebuild `clair.csp.CSP`, run the exact
  deductor, confirm `cell[query] == answer_index` (or `ABSTAIN`). The published corpus reports **0
  mismatches**. Consumers can re-run this check with `clair.csp` alone.
- **Honest abstention.** ~60% of records are *undetermined* (the query is genuinely not forced); their
  answer is `"cannot be determined"` (`answer_index = -1`). The rest are *determined* but require
  multi-step **propagation** — the query is never a directly-pinned cell (anti-shortcut), so the answer
  cannot be read off the prompt. A model must learn to deduce, and to abstain rather than guess.
- **No frontier model on the critical path.** 100% rule-based: no API keys, no network, fully
  deterministic given a seed. (A Bedrock naturalizer *could* post-process `text` for extra wording
  variety — see "Optional naturalizer" — but the rule-based path stands alone.)

## Splits (each an OOD axis held out from `train`)

| split | what's held out vs train | purpose |
|-------|--------------------------|---------|
| `train` | — | the training distribution |
| `val` | disjoint seeds, same distribution | in-distribution generalization |
| `ood_n` | **wider N** (more cells/entities) | length / problem-size generalization |
| `ood_phrasing` | **disjoint skins** (story-worlds/vocabulary never seen in train) | phrasing / lexical generalization |
| `ood_relation` | **unseen relation families** — arity 3–5 parity (`gen_parity_k`, the affine "wall") + propagation chains (`eqchain`, `forcedcolor`) | relational / structural generalization |

`val`/`ood_n`/`ood_phrasing` draw from the standard CSP families (`coloring`, `equality`, `ordering`,
`arithmetic`, `alldiff`); `ood_relation` uses families absent from train. Near-duplicate texts are
removed globally with MinHash/LSH **in split order**, so no eval record is a near-duplicate of a
train record (no leakage).

### Sizes & per-split statistics (seed 0, scale 1.0)

| split | records | determined | mean tokens | distinct-3 | self-BLEU | unique skeletons |
|-------|--------:|-----------:|------------:|-----------:|----------:|-----------------:|
| `train`        | 40,000 | 38% | 60 | 0.101 | 0.110 | 39,971 / 40,000 |
| `val`          | 4,000  | 39% | 60 | 0.291 | 0.118 | 4,000 / 4,000 |
| `ood_n`        | 4,000  | 56% | 155 | 0.216 | 0.190 | 4,000 / 4,000 |
| `ood_phrasing` | 4,000  | 32% | 62 | 0.233 | 0.168 | 4,000 / 4,000 |
| `ood_relation` | 4,000  | 64% | 68 | 0.217 | 0.190 | 4,000 / 4,000 |
| **total**      | **56,000** | **41%** | — | — | — | — |

**56,000 records, 0 exact-verification mismatches, 0 near-duplicates** (MinHash/LSH, global, split-order).
The `train` set's per-corpus `distinct-3` (0.10) is lower than the smaller splits' simply because
n-gram density saturates as a corpus grows; the **scale-invariant** diversity signals are what matter
and they are strong everywhere: **self-BLEU ≈ 0.11–0.19 (low ⇒ diverse)** and the **most-common phrasing
skeleton covers < 0.03% of any split** (≈ one unique skeleton per record — no template collapse). Full
metrics per split are in `stats.json`.



## Schema

One JSON object per record (jsonl); the parquet mirror stores the nested fields (`facts`, `cons`,
`entities`, `values`) as JSON strings. Full field docs in `schema.json`.

| field | type | meaning |
|-------|------|---------|
| `id` | str | unique id (`<split>-<hash>`) |
| `split` | str | `train` / `val` / `ood_n` / `ood_phrasing` / `ood_relation` |
| `family` | str | generator family (`coloring`/`equality`/`ordering`/`arithmetic`/`alldiff`/`parity_k`/`eqchain`/`forcedcolor`) |
| `domain_kind` | str | `color` / `ordinal` / `number` |
| `skin` | str | story-world used to render the problem |
| `n`, `d` | int | number of cells; domain size (values `0..d-1`) |
| `n_facts` | int | number of ground-truth facts (≥ 2; trivial problems filtered) |
| `determined` | bool | is the queried cell uniquely forced |
| `text` | str | **the rendered NL problem (model input)** |
| `question` | str | the question sentence (also the tail of `text`) |
| `answer` | str | the answer's **value surface** (e.g. `"green"`, `"slot 3"`) or `"cannot be determined"` |
| `answer_index` | int | value index `0..d-1`, or `-1` (ABSTAIN) |
| `trace` | str | **exact, organ-grounded reasoning trace** (sound arc-consistency narrowing + an exact closer) |
| `query` | int | queried cell index |
| `entities`, `values` | list[str] | surface strings per entity / value index |
| `facts` | list | structured ground-truth facts |
| `cons` | list | extensional CSP `(scope, allowed-tuples)` for re-verification |
| `naming_scheme`, `structure` | str | renderer axes (entity naming; prose/semicolon/bullets/numbered) |
| `seed` | int | shard seed that produced the record |

The `answer` is the answer mapped into the skin's **value surface** (so it reads in-world); the raw
index is `answer_index`. The `trace` field is **sound and exact**: every "options narrow to …" is a
superset of the true solution set (arc-consistency never over-prunes), every "forced to v" is the
proven value, and the final answer always equals `answer`/`answer_index`.

## Generation method

`clair/datagen/` (rule-based, no frontier model):

1. **Generators** — witness-first CSP families (`clair.curriculum`), the OOD arity-k parity relation
   (`gen_parity_k`), and propagation chains (`clair.hard_tasks`). Trivial problems (`n_facts < 2`)
   are dropped.
2. **Render** (`render.py`) — composes independent randomness axes (skin × entity-naming × per-relation
   phrasing × clique-aggregation × layout) so no two renderings collapse to one template.
3. **Trace** (`trace.py`) — narrates the organ's sound narrowing to a fixpoint, closed by the exact
   per-cell deduction.
4. **QC** (`qc.py`) — (a) exact label re-check via `clair.csp`; (b) an *independent keyword reader*
   (`parse.py`) recovers the constraints from the prose — renderings it can't pin down unambiguously
   are dropped (this also regression-tests the renderer for flipped/garbled facts).
5. **Dedup** — MinHash/LSH near-duplicate removal in split order.
6. **Write** — parquet + jsonl per split + `schema.json` + `stats.json` + `sample.jsonl`.

## Intended use

- Train / fine-tune reasoning models on **non-shortcutable, exactly-verifiable** CSP reasoning.
- Evaluate **out-of-distribution generalization** along three independent axes (size, phrasing,
  relation family) with the held-out splits.
- Supervise latent/looped reasoning with the exact `trace` field (process-reward / step supervision).

## Honesty notes & limitations

- Problems are **small finite CSPs** dressed in story-worlds; this is structured deductive reasoning,
  not open-domain or commonsense reasoning.
- Text diversity comes from the rule-based renderer (near-unique phrasing skeletons, self-BLEU ≈ 0.11 —
  see `stats.json`), not a language model. It reads naturally but is narrower than human writing.
- The `ood_phrasing` split holds out *skins/vocabulary*; it does not guarantee held-out *syntax*.
- Dropped-as-ambiguous records (the QC gate) are renderings a deliberately-dumb keyword parser can't
  disambiguate; the kept set is biased toward parser-recoverable phrasings. This is a quality choice,
  not a faithfulness claim about a model.

## Optional naturalizer (not on the critical path)

For extra wording variety a Bedrock LLM could post-process `text` (a paraphrase pass that must
preserve every entity/value token and add/drop no fact), re-gated by the same `qc.unambiguous`
round-trip. The hook is `clair.curriculum.render_diverse` / `extract_and_check`. The rule-based path
is the published corpus and stands alone.

## License

MIT (ours). See repository root.

## Reproduce

```bash
SCALE=1.0 SEED=0 bash report/dataset/reproduce.sh
```

Deterministic given the seed. Publishing to the Hugging Face Hub is **not** automated (needs Ember's
token); the command is documented at the bottom of `reproduce.sh`.
