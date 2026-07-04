"""Evaluate a saved checkpoint on SAGE eval splits (optionally at a different loop
count for test-time compute scaling).

    python scripts/eval.py --ckpt checkpoints/EXP-001-V1-loop4-s0.pt \
        --families algo_exec,rewrite [--loop-count 8]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from eval.harness import evaluate_model, summarize  # noqa: E402
from models import ModelConfig, build_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--families", required=True)
    ap.add_argument("--loop-count", type=int, default=None)
    ap.add_argument("--eval-dir", type=Path, default=Path("data/sage/eval"))
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    blob = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ModelConfig(**blob["config"])
    model = build_model(cfg).to(args.device)
    model.load_state_dict(blob["model"])

    results = evaluate_model(model, cfg, args.eval_dir, args.families.split(","),
                             args.device, max_seq=cfg.max_seq_len,
                             loop_count=args.loop_count, limit=args.limit)
    print(summarize(results))


if __name__ == "__main__":
    main()
