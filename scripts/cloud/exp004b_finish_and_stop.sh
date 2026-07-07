#!/usr/bin/env bash
# Wait for EXP-004/009 session B, archive results, stop pod via RunPod API.
set -uo pipefail

LOG=/workspace/auto_stop_exp004b.log
exec >>"$LOG" 2>&1

echo "=== exp004b auto-stop watcher started $(date -u -Iseconds) ==="

DEADLINE=$(( $(date +%s) + 32400 ))  # 9h hard cap

while true; do
  if grep -q 'EXP004B_ALL_DONE' /workspace/exp004b.log 2>/dev/null; then
    echo "exp004b chain complete $(date -u -Iseconds)"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "9h deadline reached — stopping anyway"
    break
  fi
  if ! pgrep -f 'run_exp004b.sh' >/dev/null; then
    sleep 120
    if ! pgrep -f 'run_exp004b.sh' >/dev/null \
       && ! grep -q 'EXP004B_ALL_DONE' /workspace/exp004b.log 2>/dev/null; then
      echo "runner gone with no completion sentinel — assuming crash, stopping"
      break
    fi
  fi
  sleep 60
done

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
tar czf /workspace/exp004b_results_${STAMP}.tar.gz \
  /workspace/exp004b.log \
  /workspace/exp004_v3cot.log \
  /workspace/aware/experiments/results.csv \
  /workspace/aware/checkpoints/diag/*.jsonl \
  2>/dev/null || true
echo "EXP004B_READY_FOR_PICKUP $(date -u -Iseconds)" | tee /workspace/EXP004B_FINAL_DONE

echo "Stopping pod via RunPod API in 15s..."
sleep 15
if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
  runpodctl stop pod "$RUNPOD_POD_ID" && echo "runpodctl stop issued" && exit 0
fi
echo "runpodctl unavailable — falling back to shutdown"
/sbin/shutdown -h now || echo "shutdown also failed; pod must be stopped externally"
