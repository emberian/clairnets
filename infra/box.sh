#!/usr/bin/env bash
# Sync the clairnets repo to the live L40S box and run a command inside the venv.
#   infra/box.sh sync                      # rsync clair/ + notes/ to ~/clairnets on the box
#   infra/box.sh run -m clair.run_glados --arm glados --target 800000 --steps 200 --source gen
#   infra/box.sh ssh                       # interactive shell
set -euo pipefail
KEY=~/dev/gowexp/infra/gowexp-key.pem
HOST=ubuntu@32.185.233.51
VENV=~/clairnets-venv
REMOTE=~/clairnets
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
cd "$(dirname "$0")/.."

case "${1:-}" in
  sync)
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
      --exclude '__pycache__' --exclude runs --exclude '.git' \
      clair notes "$HOST:$REMOTE/"
    echo "synced -> $HOST:$REMOTE" ;;
  run)
    shift
    "${SSH[@]}" "$HOST" "cd $REMOTE && source $VENV/bin/activate && PYTHONPATH=. python $*" ;;
  ssh)
    exec "${SSH[@]}" "$HOST" ;;
  *)
    echo "usage: infra/box.sh {sync|run <args>|ssh}"; exit 1 ;;
esac
