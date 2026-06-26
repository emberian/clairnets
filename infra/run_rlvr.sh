#!/usr/bin/env bash
# Launch the base RLVR (TRL GRPO) de-risk run on box4 (one L40S).
# Usage: bash infra/run_rlvr.sh [--smoke | --steps N ...]   (extra args pass through)
# Run inside tmux; logs to /tmp/rlvr.log.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/opt/pytorch/bin/python
exec "$PY" -m clair.rlvr_pipeline "$@" 2>&1 | tee /tmp/rlvr.log
