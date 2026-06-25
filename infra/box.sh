#!/usr/bin/env bash
# Sync the clairnets repo to the live L40S box and run a command inside the venv.
# Paths are REMOTE-RELATIVE (no local ~ expansion): rsync/cd resolve against the box's $HOME.
#   infra/box.sh sync
#   infra/box.sh run -m clair.run_glados --arm glados --source gen --steps 20 --target 200000
#   infra/box.sh ssh
set -euo pipefail
KEY=~/dev/gowexp/infra/gowexp-key.pem
HOST=ubuntu@32.185.233.51
REMOTE_DIR=clairnets        # resolves to ~/clairnets on the box
VENV_DIR=clairnets-venv     # resolves to ~/clairnets-venv on the box
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
cd "$(dirname "$0")/.."

case "${1:-}" in
  sync)
    "${SSH[@]}" "$HOST" "mkdir -p $REMOTE_DIR"
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
      --exclude '__pycache__' --exclude runs --exclude '.git' \
      clair notes "$HOST:$REMOTE_DIR/"
    echo "synced -> $HOST:~/$REMOTE_DIR" ;;
  run)
    shift
    "${SSH[@]}" "$HOST" "source $VENV_DIR/bin/activate && cd $REMOTE_DIR && PYTHONPATH=. python $*" ;;
  ssh)
    exec "${SSH[@]}" "$HOST" ;;
  *)
    echo "usage: infra/box.sh {sync|run <args>|ssh}"; exit 1 ;;
esac
