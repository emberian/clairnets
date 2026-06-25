#!/usr/bin/env bash
# Sequential iso-param 3-way on Sudoku-Extreme. Run on the box from ~/clairnets:
#   nohup bash infra/threeway.sh > runs/threeway.log 2>&1 &
set -o pipefail
source ~/clairnets-venv/bin/activate
cd ~/clairnets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p runs
for arm in glados ldt nowedge; do
  echo "===== $arm begin $(date -u +%H:%M:%S) ====="
  python -m clair.run_glados --arm "$arm" --target 800000 --steps 2000 --source extreme \
      --ntrain 1000 --neval 256 --pool 256
  echo "===== $arm done  $(date -u +%H:%M:%S) ====="
done
echo "ALL DONE $(date -u)"
