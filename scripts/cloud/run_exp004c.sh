#!/usr/bin/env bash
# EXP-004/EXP-009 session C (canonical re-run after session A's results were
# stranded — EXP-004.md deviation log): all EXP-004 arms (GPU lane, 2 workers)
# + grok seeds 14-21 (CPU lane, 3 workers). B2-wide lr = 3.7e-4 (probed in
# session A, protocol decision logged in EXP-004.md — not re-probed).
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
  || { echo "AUDIT FAILED - aborting"; echo EXP004C_ALL_DONE; exit 1; }

python -m pytest tests/ -q || echo "WARN: tests failed, continuing (results flagged)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

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

echo "=== GPU lane (background, 2 workers): CoT arms -> fillers -> B2-wide ==="
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
    --models B2-wide --lr 3.7e-4 --workers 2
  python scripts/run_exp.py --config experiments/configs/exp004_rule_shift.yaml \
    --models B2-wide --lr 3.7e-4 --workers 2
  echo GPULANE_DONE
) > /workspace/exp004_gpu_lane.log 2>&1 &
GPU_PID=$!

echo "=== CPU lane (foreground): EXP-009 grok seeds 14-21, 3 workers ==="
python scripts/run_exp.py --config experiments/configs/exp009_grok.yaml \
  --seeds 14,15,16,17,18,19,20,21 --workers 3
echo CPULANE_DONE

wait $GPU_PID
tail -5 /workspace/exp004_gpu_lane.log

echo EXP004C_ALL_DONE
