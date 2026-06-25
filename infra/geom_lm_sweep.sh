#!/usr/bin/env bash
# Iso-param scaling sweep for the geometric-transformer LM arms. Run on the box from
# ~/clairnets (after `infra/box.sh sync`):
#   nohup bash infra/geom_lm_sweep.sh > runs/geom_lm_sweep.log 2>&1 &
# Override knobs via env: ARMS, BUDGETS, STEPS, CORPUS, DATA_DIR.
set -o pipefail
source ~/clairnets-venv/bin/activate
cd ~/clairnets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p runs

ARMS=${ARMS:-"swiglu geom dot wedge geom_g3"}
BUDGETS=${BUDGETS:-"5e5 1e6 2e6 4e6 8e6"}
STEPS=${STEPS:-4000}
CORPUS=${CORPUS:-bin}            # bin (byte-level) | fineweb (streamed, gpt2 tokenizer)

echo "===== geom_lm sweep begin $(date -u) | arms=[$ARMS] budgets=[$BUDGETS] corpus=$CORPUS ====="
python -m clair.sweep_geom_lm --arms $ARMS --budgets $BUDGETS --steps "$STEPS" --corpus "$CORPUS"
echo "===== geom_lm sweep done  $(date -u) ====="
