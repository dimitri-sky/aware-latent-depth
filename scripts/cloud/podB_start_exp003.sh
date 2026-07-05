#!/usr/bin/env bash
# Detached launcher for EXP-003 on pod B (EXP-002 confirmed complete).
# import RUNPOD env from PID 1 for runpodctl (lesson from EXP-001B auto-stop)
export $(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD' | xargs)
nohup bash /workspace/run_exp003_podB.sh > /workspace/exp003.log 2>&1 &
sleep 2
nohup bash /workspace/exp003_finish_and_stop.sh >/dev/null 2>&1 &
sleep 1
pgrep -af 'exp003' | grep -v start_exp003
