#!/usr/bin/env bash
# Wait for the EXP-003 + EXP-002-AX chain, archive results, stop pod via RunPod
# API (runpodctl with pod-scoped key; shutdown fallback is unreliable here).
set -uo pipefail

LOG=/workspace/auto_stop_exp003.log
exec >>"$LOG" 2>&1

echo "=== EXP-003 auto-stop watcher started $(date -u -Iseconds) ==="

DEADLINE=$(( $(date +%s) + 28800 ))  # 8h hard cap

while true; do
  if grep -q 'EXP003_ALL_DONE' /workspace/exp003.log 2>/dev/null; then
    echo "EXP-003 chain complete $(date -u -Iseconds)"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "8h deadline reached — stopping anyway"
    break
  fi
  if ! pgrep -f 'run_exp003_podB.sh' >/dev/null; then
    sleep 120
    if ! pgrep -f 'run_exp003_podB.sh' >/dev/null \
       && ! grep -q 'EXP003_ALL_DONE' /workspace/exp003.log 2>/dev/null; then
      echo "runner gone with no completion sentinel — assuming crash, stopping"
      break
    fi
  fi
  sleep 60
done

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
tar czf /workspace/exp003_results_${STAMP}.tar.gz \
  /workspace/exp003.log \
  /workspace/exp002_ax.log \
  /workspace/aware/experiments/results.csv \
  2>/dev/null || true
cp /workspace/aware/experiments/results.csv /workspace/results_final_exp003.csv 2>/dev/null || true
echo "EXP003_READY_FOR_PICKUP $(date -u -Iseconds)" | tee /workspace/EXP003_FINAL_DONE

echo "Stopping pod via RunPod API in 15s..."
sleep 15
if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
  runpodctl stop pod "$RUNPOD_POD_ID" && echo "runpodctl stop issued" && exit 0
fi
echo "runpodctl unavailable — falling back to shutdown"
/sbin/shutdown -h now || echo "shutdown also failed; pod must be stopped externally"
