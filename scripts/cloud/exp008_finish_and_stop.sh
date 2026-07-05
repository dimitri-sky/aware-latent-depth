#!/usr/bin/env bash
# Wait for EXP-008, archive, stop pod via RunPod API.
set -uo pipefail

LOG=/workspace/auto_stop_exp008.log
exec >>"$LOG" 2>&1

echo "=== EXP-008 auto-stop watcher started $(date -u -Iseconds) ==="

DEADLINE=$(( $(date +%s) + 72000 ))  # 20h hard cap

while true; do
  if grep -q 'EXP008_ALL_DONE' /workspace/exp008.log 2>/dev/null; then
    echo "EXP-008 chain complete $(date -u -Iseconds)"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "20h deadline reached — stopping anyway"
    break
  fi
  if ! pgrep -f 'run_exp008.sh' >/dev/null; then
    sleep 120
    if ! pgrep -f 'run_exp008.sh' >/dev/null \
       && ! grep -q 'EXP008_ALL_DONE' /workspace/exp008.log 2>/dev/null; then
      echo "runner gone with no completion sentinel — assuming crash, stopping"
      break
    fi
  fi
  sleep 120
done

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
tar czf /workspace/exp008_results_${STAMP}.tar.gz \
  /workspace/exp008.log \
  /workspace/aware/experiments/results.csv \
  2>/dev/null || true
echo "EXP008_READY_FOR_PICKUP $(date -u -Iseconds)" | tee /workspace/EXP008_FINAL_DONE

echo "Stopping pod via RunPod API in 15s..."
sleep 15
if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
  runpodctl stop pod "$RUNPOD_POD_ID" && echo "runpodctl stop issued" && exit 0
fi
echo "runpodctl unavailable — falling back to shutdown"
/sbin/shutdown -h now || echo "shutdown also failed; pod must be stopped externally"
