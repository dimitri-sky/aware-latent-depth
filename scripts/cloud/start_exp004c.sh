#!/usr/bin/env bash
# Detached launcher for EXP-004/009 session C.
export $(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD' | xargs)
nohup bash /workspace/run_exp004c.sh > /workspace/exp004c.log 2>&1 &
sleep 2
nohup bash /workspace/exp004c_finish_and_stop.sh >/dev/null 2>&1 &
sleep 1
pgrep -af 'exp004c' | grep -v start_exp004c
