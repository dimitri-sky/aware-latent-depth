#!/usr/bin/env bash
# Wait for Gate 8 + EXP-001 to finish, archive logs/results, then halt the pod.
# RunPod stops GPU billing when the pod shuts down; /workspace data persists on Stop.
set -euo pipefail

LOG=/workspace/auto_shutdown.log
exec >>"$LOG" 2>&1

echo "=== auto-shutdown watcher started $(date -u -Iseconds) ==="

gate8_done() {
  grep -qE 'GATE6 VERDICT' /workspace/gate8.log 2>/dev/null
}

exp001_done() {
  grep -qE 'EXP COMPLETE' /workspace/exp001.log 2>/dev/null
}

gate8_failed() {
  grep -qiE 'Traceback|Error' /workspace/gate8.log 2>/dev/null \
    && ! pgrep -f 'gate6_depth.py' >/dev/null
}

exp001_failed() {
  grep -qiE 'Traceback|Error' /workspace/exp001.log 2>/dev/null \
    && ! pgrep -f 'run_exp.py' >/dev/null
}

wait_job() {
  local name=$1
  local done_fn=$2
  local fail_fn=$3
  local deadline=$(( $(date +%s) + 7200 ))  # 2h safety cap

  while true; do
    if "$done_fn"; then
      echo "[$name] complete $(date -u -Iseconds)"
      return 0
    fi
    if "$fail_fn"; then
      echo "[$name] process exited with error — continuing shutdown anyway"
      return 0
    fi
    if ! pgrep -f 'train_single.py' >/dev/null \
       && ! pgrep -f 'gate6_depth.py' >/dev/null \
       && ! pgrep -f 'run_exp.py' >/dev/null; then
      echo "[$name] no training processes left; checking log..."
      sleep 30
      if "$done_fn" || "$fail_fn"; then
        return 0
      fi
      echo "[$name] processes gone but no completion marker — continuing anyway"
      return 0
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "[$name] 2h deadline reached — shutting down anyway"
      return 0
    fi
    sleep 60
  done
}

wait_job gate8 gate8_done gate8_failed
wait_job exp001 exp001_done exp001_failed

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ARCHIVE=/workspace/night_results_${STAMP}.tar.gz
tar czf "$ARCHIVE" \
  /workspace/gate8.log \
  /workspace/exp001.log \
  /workspace/aware/experiments/results.csv \
  /workspace/aware/experiments/results_EXP-001_*.csv \
  /workspace/aware/experiments/results_EXP-000C_*.csv \
  2>/dev/null || true

cp /workspace/aware/experiments/results.csv /workspace/results_final.csv 2>/dev/null || true
echo "Archived to $ARCHIVE"
echo "READY_FOR_PICKUP $(date -u -Iseconds)" | tee /workspace/DONE.txt

echo "Halting pod in 15s..."
sleep 15
/sbin/shutdown -h now
