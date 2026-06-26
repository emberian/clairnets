#!/bin/bash
# box_run.sh <key.pem> <user@ip> <session> <remote-command...>
# Launch a long remote command in tmux, logging to /tmp/<session>.log, robust against the
# inline-tmux-quoting flakiness (writes a script file first, then `tmux new-session bash <script>`).
set -euo pipefail
KEY="$1"; HOST="$2"; SESS="$3"; shift 3; CMD="$*"
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new"
TMP="/tmp/run_${SESS}.sh"
$SSH "$HOST" "cat > $TMP" <<RUN
#!/bin/bash
cd ~/clairnets
exec > /tmp/${SESS}.log 2>&1
export PYTHONUNBUFFERED=1
echo "[box_run] ${SESS} start \$(date)"
${CMD}
echo "[box_run] ${SESS} EXIT=\$?"
RUN
$SSH "$HOST" "tmux kill-session -t ${SESS} 2>/dev/null; tmux new-session -d -s ${SESS} 'bash $TMP'; sleep 2; tmux ls"
echo "launched ${SESS} on ${HOST}; log: /tmp/${SESS}.log"
