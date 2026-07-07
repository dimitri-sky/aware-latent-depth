#!/usr/bin/env bash
# Replacement watcher for session C + supplement (the original exp004c watcher is
# killed when this one starts, since the supplement extends the session past the
# original sentinel). Archives, waits for pickup marker, stops pod.
set -uo pipefail

LOG=/workspace/auto_stop_exp004cs.log
exec >>"$LOG" 2>&1

echo "=== exp004cs auto-stop watcher started $(date -u -Iseconds) ==="

DEADLINE=$(( $(date +%s) + 32400 ))  # 9h hard cap from watcher start

while true; do
  if grep -q 'EXP004CS_ALL_DONE' /workspace/exp004c_supp.log 2>/dev/null; then
    echo "supplement complete $(date -u -Iseconds)"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "9h deadline reached — stopping anyway"
    break
  fi
  if ! pgrep -f 'run_exp004c' >/dev/null && ! pgrep -f 'exp004c_supplement' >/dev/null; then
    sleep 120
    if ! pgrep -f 'run_exp004c' >/dev/null && ! pgrep -f 'exp004c_supplement' >/dev/null \
       && ! grep -q 'EXP004CS_ALL_DONE' /workspace/exp004c_supp.log 2>/dev/null; then
      echo "runners gone with no completion sentinel — assuming crash, stopping"
      break
    fi
  fi
  sleep 60
done

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
tar czf /workspace/exp004c_results_${STAMP}.tar.gz \
  /workspace/exp004c.log \
  /workspace/exp004c_supp.log \
  /workspace/exp004_gpu_lane.log \
  /workspace/aware/experiments/results.csv \
  /workspace/aware/checkpoints/diag/*.jsonl \
  2>/dev/null || true
echo "EXP004C_READY_FOR_PICKUP $(date -u -Iseconds)" | tee /workspace/EXP004C_FINAL_DONE

# Do NOT stop before the local side confirms pickup (session-A lesson).
PICKUP_DEADLINE=$(( $(date +%s) + 3600 ))
while [ ! -f /workspace/PULLED ] && [ "$(date +%s)" -lt "$PICKUP_DEADLINE" ]; do
  sleep 30
done
echo "pickup marker: $([ -f /workspace/PULLED ] && echo found || echo TIMEOUT) $(date -u -Iseconds)"
sleep 5
# STOP only — never terminate from automation. A stopped pod's volume can still
# be recovered (CPU-only start via website UI); terminate happens manually after
# the results are adjudicated (lessons.md, owner directive 2026-07-07).
if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
  runpodctl stop pod "$RUNPOD_POD_ID" && echo "runpodctl stop issued" && exit 0
fi
echo "runpodctl unavailable — falling back to shutdown"
/sbin/shutdown -h now || echo "shutdown also failed; pod must be stopped externally"
