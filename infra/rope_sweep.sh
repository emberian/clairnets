#!/usr/bin/env bash
# Head-to-head positional-encoding sweep: all 4 RopeLM arms at ONE model size (~4M backbone
# params, ~3000 steps) for a clean bpb comparison. The channel-mixer is FIXED to SwiGLU; the
# ONLY variable is the position rotor. Run on the box from ~/clairnets (after box.sh sync):
#   nohup bash infra/rope_sweep.sh > runs/rope_sweep.log 2>&1 &
# Override knobs via env: ARMS, TARGET, STEPS, CORPUS, DATA_DIR.
set -o pipefail
source ~/clairnets-venv/bin/activate
cd ~/clairnets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p runs

ARMS=${ARMS:-"rope rope_learned rope_nd none"}
TARGET=${TARGET:-4e6}
STEPS=${STEPS:-3000}
CORPUS=${CORPUS:-bin}            # bin (byte-level enwik8/tinystories) | fineweb (gpt2 tokenizer)

echo "===== rope sweep begin $(date -u) | arms=[$ARMS] target=$TARGET steps=$STEPS corpus=$CORPUS ====="
for arm in $ARMS; do
  echo "===== $arm begin $(date -u +%H:%M:%S) ====="
  python -m clair.run_rope_lm --arm "$arm" --target "$TARGET" --steps "$STEPS" --corpus "$CORPUS"
  echo "===== $arm done  $(date -u +%H:%M:%S) ====="
done
echo "ALL DONE $(date -u)"
