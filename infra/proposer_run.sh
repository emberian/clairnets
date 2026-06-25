#!/usr/bin/env bash
# 4-arm iso-param comparison of the factor-graph lattice-deduction proposer against the EXACT
# CSP harness. Run on the box from ~/clairnets:
#   tmux new -s prop
#   PYTHONUNBUFFERED=1 bash infra/proposer_run.sh 2>&1 | tee runs/proposer.log
#
# Decisive question: does `full` (dot+wedge geometric product) reach higher COMPLETENESS than
# `ffn` at equal params, while every arm stays sound (false_elim ~ 0)?
set -o pipefail
cd ~/clairnets
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/pytorch/bin/python
mkdir -p runs

STEPS="${STEPS:-3000}"
TARGET="${TARGET:-250000}"
NEVAL="${NEVAL:-400}"
POOL="${POOL:-128}"

echo "===== SMOKE ($(date -u +%H:%M:%S)) ====="
$PY -m clair.run_proposer --arm full --smoke --steps 400 --neval 150 --out runs/proposer_smoke.json

echo "===== 4-ARM ($(date -u +%H:%M:%S)) ====="
$PY -m clair.run_proposer --arms ffn,inner,wedge,full \
    --steps "$STEPS" --target "$TARGET" --neval "$NEVAL" --pool "$POOL" \
    --out runs/proposer_4arm.json
echo "ALL DONE $(date -u)"
