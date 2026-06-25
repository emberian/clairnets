#!/usr/bin/env bash
# FAITHFUL CliffordNet (arXiv 2601.06793) replication + the decisive lambda ablation on CIFAR-100.
# The corrected, Pareto-framed question:
#   (1) Does CliffordNet-Nano (~1.4M, lambda=1) REACH ~76-78% with the paper's recipe? (replicate?)
#   (2) Does lambda=1 (Differential / Laplacian context) matter vs lambda=0 (the self-energy-
#       suppression mechanism our prior clair/vision.py MISSED)?
# Run on the box from ~/clairnets (after infra/box.sh sync):
#   nohup bash infra/cliffordnet_run.sh > runs/cliffordnet_run.log 2>&1 &
# Override knobs via env: EPOCHS, BS, VARIANT, EXTRA.
set -o pipefail
source ~/clairnets-venv/bin/activate
cd ~/clairnets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p runs

EPOCHS=${EPOCHS:-200}
BS=${BS:-128}
VARIANT=${VARIANT:-nano}
EXTRA=${EXTRA:-}

echo "===== cliffordnet replication begin $(date -u) | variant=$VARIANT epochs=$EPOCHS bs=$BS ====="

# (1) the replication target: faithful Nano, lambda=1 (Differential Mode, full geometric product)
echo "----- $VARIANT lambda=1 (replication target) $(date -u +%H:%M:%S) -----"
python -m clair.run_cliffordnet --variant "$VARIANT" --lam 1 --epochs "$EPOCHS" --bs "$BS" $EXTRA

# (2) the decisive ablation: SAME model, lambda=0 (Absolute Mode) — isolates self-energy suppression
echo "----- $VARIANT lambda=0 (ablation: no Laplacian high-pass) $(date -u +%H:%M:%S) -----"
python -m clair.run_cliffordnet --variant "$VARIANT" --lam 0 --epochs "$EPOCHS" --bs "$BS" $EXTRA

echo "ALL DONE $(date -u)"
