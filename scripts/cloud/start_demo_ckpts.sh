#!/usr/bin/env bash
# Detached launcher for the demo-checkpoint mini-session.
export $(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD' | xargs)
nohup bash /workspace/run_demo_ckpts.sh > /workspace/demo_ckpts.log 2>&1 &
sleep 2
nohup bash /workspace/demo_ckpts_stop.sh >/dev/null 2>&1 &
sleep 1
pgrep -af 'demo_ckpts' | grep -v start_demo
