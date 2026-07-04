"""Minimal CLI around train_one for parallel workers.

    python scripts/train_single.py --exp-id EXP-000B --model-id probe-2L \
        --arch tf_pp --n-layers 2 --families rewrite --steps 4000 --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from models import ModelConfig  # noqa: E402
from train.train import TrainConfig, train_one  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--arch", default="tf_pp")
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-kv-heads", type=int, default=2)
    ap.add_argument("--d-ff", type=int, default=704)
    ap.add_argument("--model-json", default=None,
                    help="full ModelConfig as JSON; overrides the flags above")
    ap.add_argument("--families", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--notes", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.model_json:
        import json
        mcfg = ModelConfig(**json.loads(args.model_json))
    else:
        mcfg = ModelConfig(arch=args.arch, n_layers=args.n_layers, d_model=args.d_model,
                           n_heads=args.n_heads, n_kv_heads=args.n_kv_heads,
                           d_ff=args.d_ff, max_seq_len=1024)
    tc = TrainConfig(families=args.families.split(","), steps=args.steps,
                     batch_size=32, seq_len=768, seed=args.seed, lr=args.lr,
                     warmup=300, eval_limit=200)
    train_one(mcfg, tc, exp_id=args.exp_id, model_id=args.model_id,
              device=args.device, notes=args.notes)


if __name__ == "__main__":
    main()
