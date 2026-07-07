#!/usr/bin/env bash
# Wait for EXP-004/009 session A, archive results, stop pod via RunPod API.
set -uo pipefail

LOG=/workspace/auto_stop_exp004a.log
exec >>"$LOG" 2>&1

echo "=== exp004a auto-stop watcher started $(date -u -Iseconds) ==="

DEADLINE=$(( $(date +%s) + 32400 ))  # 9h hard cap

while true; do
  if grep -q 'EXP004A_ALL_DONE' /workspace/exp004a.log 2>/dev/null; then
    echo "exp004a chain complete $(date -u -Iseconds)"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "9h deadline reached — stopping anyway"
    break
  fi
  if ! pgrep -f 'run_exp004a.sh' >/dev/null; then
    sleep 120
    if ! pgrep -f 'run_exp004a.sh' >/dev/null \
       && ! grep -q 'EXP004A_ALL_DONE' /workspace/exp004a.log 2>/dev/null; then
      echo "runner gone with no completion sentinel — assuming crash, stopping"
      break
    fi
  fi
  sleep 60
done

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
tar czf /workspace/exp004a_results_${STAMP}.tar.gz \
  /workspace/exp004a.log \
  /workspace/exp004_gpu_lane.log \
  /workspace/probe49.log /workspace/probe37.log \
  /workspace/aware/experiments/results.csv \
  /workspace/aware/checkpoints/diag/*.jsonl \
  2>/dev/null || true
echo "EXP004A_READY_FOR_PICKUP $(date -u -Iseconds)" | tee /workspace/EXP004A_FINAL_DONE

# Session-A lesson (2026-07-07): stopping right after archiving stranded results on
# an unrestartable community pod. Wait up to 60 min for a /workspace/PULLED marker.
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
