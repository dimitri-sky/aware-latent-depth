#!/usr/bin/env bash
# Stage 1+2 session on pod B (reusing EXP-002/003 volume): EXP-006 attribution
# 2x2 + EXP-003B recipe retry + H2 EXTEND (rule_shift seeds 3-5).
# Schedule: CPU-bound V2 jobs background, GPU-bound chain foreground; V2
# rule_shift EXTEND runs alone at 3 workers last (3 x ~9GB; 2.3h wave).
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

# generate any missing families (fresh volume has none; seeded generation makes
# data identical across pods)
MISSING=""
for fam in rewrite algo_exec rule_shift compress state_guard; do
  [ -f "data/sage/train/${fam}.jsonl" ] || MISSING="${MISSING}${MISSING:+,}${fam}"
done
if [ -n "$MISSING" ]; then
  echo "generating missing families: $MISSING"
  python scripts/make_data.py --split train --per-family 20000 --families "$MISSING"
  python scripts/make_data.py --split eval --per-family 400 --families "$MISSING"
  python -m sage.contamination.audit --train-dir data/sage/train --eval-dir data/sage/eval
fi

python -m pytest tests/ -q || echo "WARN: tests failed, continuing (results flagged)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

echo "=== BG: EXP-006-AX V2-full (CPU-bound, 2 workers) ==="
python scripts/run_exp.py --config experiments/configs/exp006_algo_exec.yaml \
  --models V2-full --workers 2 > /workspace/exp006_v2full.log 2>&1 &
V2FULL_PID=$!

echo "=== FG GPU chain (1 worker while V2-full active) ==="
python scripts/run_exp.py --config experiments/configs/exp006_algo_exec.yaml --models B2-SWA --workers 1
python scripts/run_exp.py --config experiments/configs/exp006_compress.yaml --workers 1
python scripts/run_exp.py --config experiments/configs/exp006_state_guard.yaml --workers 1

echo "=== waiting for V2-full before widening GPU concurrency ==="
wait $V2FULL_PID

echo "=== EXP-003B recipe (2 workers) + B2 rule_shift EXTEND (2 workers) ==="
python scripts/run_exp.py --config experiments/configs/exp003b_recipe.yaml --workers 2
python scripts/run_exp.py --config experiments/configs/exp002_rule_shift.yaml \
  --models B2-6L --seeds 3,4,5 --workers 2

echo "=== H2 EXTEND: V2 rule_shift seeds 3-5 alone (3 workers) ==="
python scripts/run_exp.py --config experiments/configs/exp002_rule_shift.yaml \
  --models V2-delta --seeds 3,4,5 --workers 3

echo STAGE12_ALL_DONE
