"""Run a multi-model, multi-seed experiment from a YAML config (see
experiments/configs/exp001_loop_falsifier.yaml).

    python scripts/train_tiny.py --config experiments/configs/exp001_loop_falsifier.yaml
    python scripts/train_tiny.py --config ... --seeds 0 --models V1-loop1,V1-loop4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import yaml  # noqa: E402

from models import ModelConfig  # noqa: E402
from train.train import TrainConfig, train_one  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--seeds", default=None, help="comma-separated override")
    ap.add_argument("--models", default=None, help="comma-separated model id filter")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    spec = yaml.safe_load(args.config.read_text())
    exp_id = spec["exp_id"]
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else spec.get("seeds", [0]))
    model_filter = set(args.models.split(",")) if args.models else None

    for seed in seeds:
        for m in spec["models"]:
            mid = m["id"]
            if model_filter and mid not in model_filter:
                continue
            mkw = {k: v for k, v in m.items() if k != "id"}
            mcfg = ModelConfig(**mkw)
            tc = TrainConfig(seed=seed, **spec["train"])
            train_one(mcfg, tc, exp_id=exp_id, model_id=mid, device=args.device)


if __name__ == "__main__":
    main()
