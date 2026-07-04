#!/usr/bin/env bash
# RunPod bootstrap: unpack repo, install deps, regenerate SAGE data, run the
# validity gate, and (only on gate PASS - guardrail 5) run EXP-001.
# Usage on pod:  bash bootstrap.sh > /workspace/bootstrap.log 2>&1
set -euo pipefail

cd /workspace
rm -rf aware && mkdir aware && cd aware
unzip -q /workspace/aware_src.zip

pip install -q numpy pyyaml pytest tqdm matplotlib

export AWARE_THROTTLE=0
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/make_data.py --split train --per-family 20000
python scripts/make_data.py --split eval --per-family 400
python -m pytest tests/ -q
python -m sage.contamination.audit --train-dir data/sage/train --eval-dir data/sage/eval

echo "=== VALIDITY GATE (attempt 5, cloud) ==="
if python scripts/validity_gate.py 2>&1 | tee gate5.log; then
    echo "=== GATE PASSED -> EXP-001 falsifier ==="
    python scripts/train_tiny.py --config experiments/configs/exp001_loop_falsifier.yaml 2>&1 | tee exp001.log
    echo "=== EXP-001 COMPLETE ==="
else
    echo "=== GATE FAILED - stopping per benchmark-first rule (no EXP-001) ==="
fi
echo "=== BOOTSTRAP DONE ==="
