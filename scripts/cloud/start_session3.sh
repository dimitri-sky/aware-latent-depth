#!/usr/bin/env bash
# Detached launcher for session 3.
export $(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD' | xargs)
nohup bash /workspace/run_session3.sh > /workspace/session3.log 2>&1 &
sleep 2
nohup bash /workspace/session3_finish_and_stop.sh >/dev/null 2>&1 &
sleep 1
pgrep -af 'session3' | grep -v start_session3
