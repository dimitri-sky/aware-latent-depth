#!/usr/bin/env bash
# EXP-004/EXP-009 session A: EXP-004 arms (GPU lane, 2 workers) + first 8 grokking
# seeds (CPU-bound delta lane, 2 workers). Memory plan (arc-1 lesson): 2 delta
# workers + 2 small B2 GPU jobs fits 32GB.
# No `set -e`: the sentinel must appear even after partial failure.
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

if [ ! -f data/sage/train/algo_exec.jsonl ]; then
  python scripts/make_data.py --split train --per-family 20000 --families algo_exec,rule_shift
  python scripts/make_data.py --split eval --per-family 400 --families algo_exec,rule_shift
fi
python -m sage.contamination.audit --train-dir data/sage/train --eval-dir data/sage/eval \
  || { echo "AUDIT FAILED - aborting"; echo EXP004A_ALL_DONE; exit 1; }

python -m pytest tests/ -q || echo "WARN: tests failed, continuing (results flagged)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
free -g | head -2

echo "=== config-hash record (pre-launch check: all arms distinct) ==="
python - <<'EOF'
import yaml
from pathlib import Path
from models import ModelConfig
for f in ["exp004_algo_exec", "exp004_rule_shift", "exp009_grok"]:
    spec = yaml.safe_load(Path(f"experiments/configs/{f}.yaml").read_text())
    for m in spec["models"]:
        mkw = {k: v for k, v in m.items() if k != "id"}
        print(f"{spec['exp_id']:12s} {m['id']:16s} {ModelConfig(**mkw).config_hash()}")
EOF

echo "=== B2-wide lr probe (pre-eval protocol decision, EXP-004.md) ==="
WIDE='{"arch":"tf_pp","d_model":672,"n_heads":8,"n_kv_heads":4,"d_ff":1856,"n_layers":6,"max_seq_len":1024}'
AWARE_RESULTS_CSV=/workspace/probe_shard.csv python scripts/train_single.py \
  --exp-id EXP-004-PROBE --model-id B2-wide-lr49 --model-json "$WIDE" \
  --families algo_exec --steps 500 --seed 0 --lr 4.9e-4 \
  > /workspace/probe49.log 2>&1
AWARE_RESULTS_CSV=/workspace/probe_shard.csv python scripts/train_single.py \
  --exp-id EXP-004-PROBE --model-id B2-wide-lr37 --model-json "$WIDE" \
  --families algo_exec --steps 500 --seed 0 --lr 3.7e-4 \
  > /workspace/probe37.log 2>&1
L49=$(grep -oE 'loss [0-9.]+' /workspace/probe49.log | tail -1 | cut -d' ' -f2)
L37=$(grep -oE 'loss [0-9.]+' /workspace/probe37.log | tail -1 | cut -d' ' -f2)
WIDE_LR=$(python -c "print('4.9e-4' if float('${L49:-9}') <= float('${L37:-9}') else '3.7e-4')")
echo "WIDE_LR_DECISION lr49_loss=${L49:-NA} lr37_loss=${L37:-NA} chosen=$WIDE_LR"

echo "=== GPU lane (background): CoT arms -> fillers -> B2-wide ==="
(
  python scripts/run_exp.py --config experiments/configs/exp004_algo_exec.yaml \
    --models B2-CoT-short,B2-CoT-med,B2-CoT-long --workers 2
  python scripts/run_exp.py --config experiments/configs/exp004_rule_shift.yaml \
    --models B2-CoT-med,B2-CoT-long --workers 2
  python scripts/run_exp.py --config experiments/configs/exp004_algo_exec.yaml \
    --models B2-filler-long --seeds 0,1 --workers 2
  python scripts/run_exp.py --config experiments/configs/exp004_rule_shift.yaml \
    --models B2-filler-long --seeds 0,1 --workers 2
  python scripts/run_exp.py --config experiments/configs/exp004_algo_exec.yaml \
    --models B2-wide --lr "$WIDE_LR" --workers 2
  python scripts/run_exp.py --config experiments/configs/exp004_rule_shift.yaml \
    --models B2-wide --lr "$WIDE_LR" --workers 2
  echo GPULANE_DONE
) > /workspace/exp004_gpu_lane.log 2>&1 &
GPU_PID=$!

echo "=== CPU lane (foreground): EXP-009 grok seeds 6-13, 2 workers ==="
python scripts/run_exp.py --config experiments/configs/exp009_grok.yaml \
  --seeds 6,7,8,9,10,11,12,13 --workers 2
echo CPULANE_DONE

wait $GPU_PID
tail -5 /workspace/exp004_gpu_lane.log

echo EXP004A_ALL_DONE
