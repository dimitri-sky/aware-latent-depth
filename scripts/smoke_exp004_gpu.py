"""Local GPU efficacy check for the CoT instrument (pre-pod de-risk, not a result):
a small-but-real tf_pp (4L d256) trained on algo_exec CoT-long for 1200 steps must
(a) drive format-failure rate near 0 and (b) score above 0 — i.e. the traced
format is learnable end-to-end through our train+eval path. Uses the canonical
seeded train/eval data regenerated with trace variants into data/sage_smoke_gpu.

    python scripts/smoke_exp004_gpu.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SMOKE = Path("data/sage_smoke_gpu")
SHARD = Path("experiments/results_SMOKE_exp004_gpu.csv")


def main() -> None:
    if not (SMOKE / "train" / "algo_exec.jsonl").exists():
        for split, n in (("train", 4000), ("eval", 400)):
            subprocess.run([sys.executable, "scripts/make_data.py", "--split", split,
                            "--per-family", str(n), "--families", "algo_exec",
                            "--out-dir", str(SMOKE)], check=True)

    header = Path("experiments/results.csv").open(encoding="utf-8").readline()
    SHARD.write_text(header, encoding="utf-8")
    os.environ["AWARE_RESULTS_CSV"] = str(SHARD)

    from models import ModelConfig  # noqa: E402
    from train.train import TrainConfig, train_one  # noqa: E402

    mcfg = ModelConfig(arch="tf_pp", d_model=256, n_layers=4, n_heads=4,
                       n_kv_heads=2, d_ff=704, max_seq_len=1024,
                       extra={"trace_level": "long"})
    tc = TrainConfig(families=["algo_exec"], steps=1200, batch_size=32, seq_len=768,
                     lr=6e-4, warmup=100, seed=0, traced=True, trace_level="long",
                     eval_limit=100, data_dir=str(SMOKE / "train"),
                     eval_dir=str(SMOKE / "eval"))
    out = train_one(mcfg, tc, exp_id="EXP-SMOKE4G", model_id="tf4L-cot-long",
                    device="cuda", notes="local gpu cot efficacy smoke")
    res = out["results"]
    ffr = res["_cot"]["format_failure_rate"]
    n = sum(v[0] for f, s in res.items() if not f.startswith("_") for v in s.values())
    c = sum(v[1] for f, s in res.items() if not f.startswith("_") for v in s.values())
    print(f"[gpu-smoke] acc={c / max(1, n):.3f} format_failure_rate={ffr:.3f}")
    assert ffr < 0.2, "model failed to learn the THINK/ANSWER format"
    assert c > 0, "zero correct answers after 1200 steps"
    print("[gpu-smoke] COT INSTRUMENT VALIDATED")


if __name__ == "__main__":
    main()
