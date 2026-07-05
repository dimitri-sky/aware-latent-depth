#!/usr/bin/env bash
# EXP-008 bootstrap + feasibility probe (run in FOREGROUND over SSH, ~20 min):
# env setup, data gen, then 200-step V3-50M throughput/VRAM probe on algo_exec
# and 100-step on state_guard. Prints PROBE_METRICS lines; the gate decision
# stays with the operator (>=5k tok/s, <30GB).
set -u

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AWARE_THROTTLE=0

cd /workspace
if [ ! -d /workspace/aware ]; then
  mkdir aware && cd aware
  unzip -q /workspace/aware_src.zip
  pip install -q --break-system-packages numpy pyyaml pytest tqdm matplotlib
else
  cd /workspace/aware
  unzip -qo /workspace/aware_src.zip
fi

for fam in algo_exec state_guard; do
  if [ ! -f "data/sage/train/${fam}.jsonl" ]; then
    python scripts/make_data.py --split train --per-family 20000 --families "$fam"
    python scripts/make_data.py --split eval --per-family 400 --families "$fam"
  fi
done
python -m sage.contamination.audit --train-dir data/sage/train --eval-dir data/sage/eval
python -m pytest tests/ -q || echo "WARN: tests failed"

VJSON='{"arch":"delta","d_model":768,"n_heads":12,"n_kv_heads":4,"d_ff":2048,"n_layers":8,"delta_every":2,"window":128,"d_k":512,"d_v":512,"max_seq_len":1024}'

probe () {
  fam=$1; steps=$2
  (python scripts/train_single.py --exp-id EXP-008-PROBE --model-id V3-50M \
    --model-json "$VJSON" --families "$fam" --steps "$steps" --seed 0 --lr 4.2e-4 \
    > "/workspace/probe_${fam}.log" 2>&1) &
  TPID=$!
  PEAK=0
  while kill -0 $TPID 2>/dev/null; do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$M" -gt "$PEAK" ] && PEAK=$M
    sleep 5
  done
  wait $TPID; RC=$?
  TOKS=$(grep -oE '[0-9]+ tok/s' "/workspace/probe_${fam}.log" | tail -1)
  echo "PROBE_METRICS family=${fam} rc=${RC} peak_vram_mb=${PEAK} last_${TOKS:-tok/s=NA}"
}

probe algo_exec 200
probe state_guard 100
echo PROBE_DONE
