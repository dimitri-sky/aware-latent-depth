#!/usr/bin/env bash
# Auto-stop for the demo-checkpoint mini-session (4h cap).
set -uo pipefail
exec >>/workspace/auto_stop_demo.log 2>&1
DEADLINE=$(( $(date +%s) + 14400 ))
while true; do
  grep -q 'DEMOCKPT_ALL_DONE' /workspace/demo_ckpts.log 2>/dev/null && break
  [ "$(date +%s)" -ge "$DEADLINE" ] && break
  if ! pgrep -f 'run_demo_ckpts.sh' >/dev/null; then
    sleep 120
    pgrep -f 'run_demo_ckpts.sh' >/dev/null \
      || grep -q 'DEMOCKPT_ALL_DONE' /workspace/demo_ckpts.log 2>/dev/null || break
  fi
  sleep 60
done
echo DONE | tee /workspace/DEMOCKPT_FINAL_DONE
sleep 15
if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
  runpodctl stop pod "$RUNPOD_POD_ID" && exit 0
fi
/sbin/shutdown -h now || true
