#!/usr/bin/env bash
# Wait for the EXP-001B chain (algo -> seeds -> flopmatch) to finish, archive
# logs/results, then halt the pod so GPU billing stops. /workspace persists on Stop.
set -euo pipefail

LOG=/workspace/auto_shutdown_b.log
exec >>"$LOG" 2>&1

echo "=== EXP-001B auto-shutdown watcher started $(date -u -Iseconds) ==="

DEADLINE=$(( $(date +%s) + 21600 ))  # 6h hard cap

while true; do
  if grep -q 'EXP001B_ALL_DONE' /workspace/exp001b.log 2>/dev/null; then
    echo "EXP-001B chain complete $(date -u -Iseconds)"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "6h deadline reached — shutting down anyway"
    break
  fi
  # The runner script is the single process to track; if it is gone and the
  # sentinel never appeared, it crashed — grace period, then shut down.
  if ! pgrep -f 'run_exp001b.sh' >/dev/null; then
    sleep 120
    if ! pgrep -f 'run_exp001b.sh' >/dev/null \
       && ! grep -q 'EXP001B_ALL_DONE' /workspace/exp001b.log 2>/dev/null; then
      echo "runner process gone and no completion sentinel — assuming crash, shutting down"
      break
    fi
  fi
  sleep 60
done

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
tar czf /workspace/exp001b_results_${STAMP}.tar.gz \
  /workspace/exp001b.log \
  /workspace/aware/experiments/results.csv \
  /workspace/aware/experiments/results_EXP-001B*.csv \
  2>/dev/null || true

cp /workspace/aware/experiments/results.csv /workspace/results_final_b.csv 2>/dev/null || true
echo "EXP001B_READY_FOR_PICKUP $(date -u -Iseconds)" | tee /workspace/DONE_B.txt

echo "Halting pod in 15s..."
sleep 15
/sbin/shutdown -h now
