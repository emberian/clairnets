#!/usr/bin/env bash
# Iso-param vision sweep on CIFAR-100: geom vs wedge vs conv vs mlp at CliffordNet's scale.
# The decisive question: does the geometric product BEAT conv & mlp on its home turf (vision)
# at equal non-embed params, reproducing CliffordNet (CIFAR-100 76-78% @ 1.4-3M)?
# Run on the box from ~/clairnets:
#   nohup bash infra/vision_sweep.sh > runs/vision_sweep.log 2>&1 &
set -o pipefail
source ~/clairnets-venv/bin/activate
cd ~/clairnets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p runs
for target in 1500000 3000000; do
  for arm in geom wedge conv mlp; do
    echo "===== $arm @ $target begin $(date -u +%H:%M:%S) ====="
    python -m clair.run_vision --arm "$arm" --target "$target" --epochs 100 --bs 128
    echo "===== $arm @ $target done  $(date -u +%H:%M:%S) ====="
  done
done
echo "ALL DONE $(date -u)"
