#!/usr/bin/env bash
# Wait for EXP-004/009 session C, archive results, wait for pickup, stop pod.
set -uo pipefail

LOG=/workspace/auto_stop_exp004c.log
exec >>"$LOG" 2>&1

echo "=== exp004c auto-stop watcher started $(date -u -Iseconds) ==="

DEADLINE=$(( $(date +%s) + 32400 ))  # 9h hard cap

while true; do
  if grep -q 'EXP004C_ALL_DONE' /workspace/exp004c.log 2>/dev/null; then
    echo "exp004c chain complete $(date -u -Iseconds)"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "9h deadline reached — stopping anyway"
    break
  fi
  if ! pgrep -f 'run_exp004c.sh' >/dev/null; then
    sleep 120
    if ! pgrep -f 'run_exp004c.sh' >/dev/null \
       && ! grep -q 'EXP004C_ALL_DONE' /workspace/exp004c.log 2>/dev/null; then
      echo "runner gone with no completion sentinel — assuming crash, stopping"
      break
    fi
  fi
  sleep 60
done

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
tar czf /workspace/exp004c_results_${STAMP}.tar.gz \
  /workspace/exp004c.log \
  /workspace/exp004_gpu_lane.log \
  /workspace/aware/experiments/results.csv \
  /workspace/aware/checkpoints/diag/*.jsonl \
  2>/dev/null || true
echo "EXP004C_READY_FOR_PICKUP $(date -u -Iseconds)" | tee /workspace/EXP004C_FINAL_DONE

# Session-A lesson (2026-07-07): do NOT stop before the local side confirms pickup.
# Wait up to 60 min for /workspace/PULLED (~$0.69 worst case, cheap insurance).
PICKUP_DEADLINE=$(( $(date +%s) + 3600 ))
while [ ! -f /workspace/PULLED ] && [ "$(date +%s)" -lt "$PICKUP_DEADLINE" ]; do
  sleep 30
done
echo "pickup marker: $([ -f /workspace/PULLED ] && echo found || echo TIMEOUT) $(date -u -Iseconds)"
sleep 5
if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
  runpodctl stop pod "$RUNPOD_POD_ID" && echo "runpodctl stop issued" && exit 0
fi
echo "runpodctl unavailable — falling back to shutdown"
/sbin/shutdown -h now || echo "shutdown also failed; pod must be stopped externally"
