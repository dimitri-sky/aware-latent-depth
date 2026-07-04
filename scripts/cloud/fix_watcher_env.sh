#!/usr/bin/env bash
# Restart the auto-stop watcher with RunPod env (pod id + API key) imported from
# PID 1, since SSH sessions don't inherit the container environment.
pkill -f exp002_finish_and_stop.sh
sleep 1
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | xargs)
echo "POD_ID=${RUNPOD_POD_ID:-STILL_UNSET}"
echo "API_KEY_SET=$([ -n "${RUNPOD_API_KEY:-}" ] && echo yes || echo no)"
nohup bash /workspace/exp002_finish_and_stop.sh >/dev/null 2>&1 &
sleep 2
pgrep -af finish_and_stop
