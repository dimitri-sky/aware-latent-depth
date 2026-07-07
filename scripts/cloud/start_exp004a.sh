#!/usr/bin/env bash
# Detached launcher for EXP-004/009 session A (RUNPOD env imported from PID 1,
# since SSH sessions don't inherit the container environment).
export $(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD' | xargs)
nohup bash /workspace/run_exp004a.sh > /workspace/exp004a.log 2>&1 &
sleep 2
nohup bash /workspace/exp004a_finish_and_stop.sh >/dev/null 2>&1 &
sleep 1
pgrep -af 'exp004a' | grep -v start_exp004a
