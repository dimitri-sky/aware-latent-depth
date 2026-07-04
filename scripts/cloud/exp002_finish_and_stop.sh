#!/usr/bin/env bash
# Wait for the EXP-002 chain to finish, archive logs/results, then stop the pod
# via the RunPod API (lesson from EXP-001B: /sbin/shutdown is unreliable in some
# containers; runpodctl uses the pod-scoped RUNPOD_API_KEY and actually stops
# billing). shutdown remains as a last-resort fallback.
set -uo pipefail

LOG=/workspace/auto_stop_exp002.log
exec >>"$LOG" 2>&1

echo "=== EXP-002 auto-stop watcher started $(date -u -Iseconds) ==="

DEADLINE=$(( $(date +%s) + 21600 ))  # 6h hard cap

while true; do
  if grep -q 'EXP002_ALL_DONE' /workspace/exp002.log 2>/dev/null; then
    echo "EXP-002 chain complete $(date -u -Iseconds)"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "6h deadline reached — stopping anyway"
    break
  fi
  if ! pgrep -f 'run_exp002.sh' >/dev/null; then
    sleep 120
    if ! pgrep -f 'run_exp002.sh' >/dev/null \
       && ! grep -q 'EXP002_ALL_DONE' /workspace/exp002.log 2>/dev/null; then
      echo "runner gone with no completion sentinel — assuming crash, stopping"
      break
    fi
  fi
  sleep 60
done

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
tar czf /workspace/exp002_results_${STAMP}.tar.gz \
  /workspace/exp002.log \
  /workspace/aware/experiments/results.csv \
  /workspace/aware/experiments/results_EXP-002*.csv \
  2>/dev/null || true
cp /workspace/aware/experiments/results.csv /workspace/results_final_exp002.csv 2>/dev/null || true
echo "EXP002_READY_FOR_PICKUP $(date -u -Iseconds)" | tee /workspace/DONE_EXP002.txt

echo "Stopping pod via RunPod API in 15s..."
sleep 15
if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
  runpodctl stop pod "$RUNPOD_POD_ID" && echo "runpodctl stop issued" && exit 0
fi
echo "runpodctl unavailable — falling back to shutdown"
/sbin/shutdown -h now || echo "shutdown also failed; pod must be stopped externally"
