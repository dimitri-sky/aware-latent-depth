#!/usr/bin/env bash
# Wait for session 3, archive results, stop pod via RunPod API.
set -uo pipefail

LOG=/workspace/auto_stop_session3.log
exec >>"$LOG" 2>&1

echo "=== session3 auto-stop watcher started $(date -u -Iseconds) ==="

DEADLINE=$(( $(date +%s) + 32400 ))  # 9h hard cap

while true; do
  if grep -q 'SESSION3_ALL_DONE' /workspace/session3.log 2>/dev/null; then
    echo "session3 chain complete $(date -u -Iseconds)"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "9h deadline reached — stopping anyway"
    break
  fi
  if ! pgrep -f 'run_session3.sh' >/dev/null; then
    sleep 120
    if ! pgrep -f 'run_session3.sh' >/dev/null \
       && ! grep -q 'SESSION3_ALL_DONE' /workspace/session3.log 2>/dev/null; then
      echo "runner gone with no completion sentinel — assuming crash, stopping"
      break
    fi
  fi
  sleep 60
done

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
tar czf /workspace/session3_results_${STAMP}.tar.gz \
  /workspace/session3.log \
  /workspace/exp007_v2.log \
  /workspace/aware/experiments/results.csv \
  2>/dev/null || true
echo "SESSION3_READY_FOR_PICKUP $(date -u -Iseconds)" | tee /workspace/SESSION3_FINAL_DONE

echo "Stopping pod via RunPod API in 15s..."
sleep 15
if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
  runpodctl stop pod "$RUNPOD_POD_ID" && echo "runpodctl stop issued" && exit 0
fi
echo "runpodctl unavailable — falling back to shutdown"
/sbin/shutdown -h now || echo "shutdown also failed; pod must be stopped externally"
