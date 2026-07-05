#!/usr/bin/env bash
# Wait for the Stage-1/2 chain, archive results, stop pod via RunPod API.
set -uo pipefail

LOG=/workspace/auto_stop_stage12.log
exec >>"$LOG" 2>&1

echo "=== Stage-1/2 auto-stop watcher started $(date -u -Iseconds) ==="

DEADLINE=$(( $(date +%s) + 21600 ))  # 6h hard cap

while true; do
  if grep -q 'STAGE12_ALL_DONE' /workspace/stage12.log 2>/dev/null; then
    echo "Stage-1/2 chain complete $(date -u -Iseconds)"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "6h deadline reached — stopping anyway"
    break
  fi
  if ! pgrep -f 'run_stage12_podB.sh' >/dev/null; then
    sleep 120
    if ! pgrep -f 'run_stage12_podB.sh' >/dev/null \
       && ! grep -q 'STAGE12_ALL_DONE' /workspace/stage12.log 2>/dev/null; then
      echo "runner gone with no completion sentinel — assuming crash, stopping"
      break
    fi
  fi
  sleep 60
done

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
tar czf /workspace/stage12_results_${STAMP}.tar.gz \
  /workspace/stage12.log \
  /workspace/exp006_v2full.log \
  /workspace/aware/experiments/results.csv \
  2>/dev/null || true
echo "STAGE12_READY_FOR_PICKUP $(date -u -Iseconds)" | tee /workspace/STAGE12_FINAL_DONE

echo "Stopping pod via RunPod API in 15s..."
sleep 15
if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
  runpodctl stop pod "$RUNPOD_POD_ID" && echo "runpodctl stop issued" && exit 0
fi
echo "runpodctl unavailable — falling back to shutdown"
/sbin/shutdown -h now || echo "shutdown also failed; pod must be stopped externally"
