#!/usr/bin/env bash
# report/dataset/reproduce.sh — regenerate the GLaDOS reasoning corpus from scratch, deterministically.
#
# The corpus is 100% rule-based: no API keys, no frontier model, no network. Given the same seed it
# reproduces bit-for-bit. Generation is CPU-only and parallel (one shard per core).
set -euo pipefail

SEED="${SEED:-0}"
SCALE="${SCALE:-1.0}"                       # multiply all split sizes (1.0 = the published corpus)
OUT="${OUT:-data/glados_corpus}"
PY="${PY:-python}"                          # on the organ box: /opt/pytorch/bin/python

cd "$(dirname "$0")/../.."                   # repo root (this file lives in report/dataset/)

# Deps: numpy (always), pyarrow (parquet). datasets only needed to publish to the Hub.
$PY -c "import numpy, pyarrow" 2>/dev/null || $PY -m pip install -q numpy pyarrow

echo "== self-checks (exact ground truth, Bedrock-independent) =="
$PY -m clair.csp            # Lean-grounded harness checks
$PY -m clair.curriculum     # witness-first generators: every label exact vs brute force

echo "== building corpus (seed=$SEED scale=$SCALE -> $OUT) =="
$PY -m clair.datagen.build --out "$OUT" --seed "$SEED" --scale "$SCALE" --verify-frac 1.0

echo "== outputs =="
ls -lh "$OUT"
echo "Per-split sizes + diversity are in $OUT/stats.json ; schema in $OUT/schema.json"

# ---------------------------------------------------------------------------------------------------
# PUBLISH TO HUGGING FACE (manual; needs Ember's token — NOT run automatically).
#
#   export HF_TOKEN=...                       # Ember's write token
#   python - <<'PY'
#   from datasets import load_dataset
#   import glob, os
#   out = os.environ.get("OUT", "data/glados_corpus")
#   files = {os.path.basename(f).split(".")[0]: f for f in glob.glob(f"{out}/*.jsonl")}
#   ds = load_dataset("json", data_files=files)          # splits: train/val/ood_n/ood_phrasing/ood_relation
#   ds.push_to_hub("clairnets/glados-reasoning", private=True, token=os.environ["HF_TOKEN"])
#   PY
#
# The card in report/dataset/README.md is HF-ready (front-matter + schema + honesty notes).
# ---------------------------------------------------------------------------------------------------
