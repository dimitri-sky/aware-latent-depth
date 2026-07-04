"""Debug utility: print model generations vs expected answers for a checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from eval.harness import greedy_generate
from models import ModelConfig, build_model
from train.data import load_sage_records
from train.tokenizer import BOS, decode, encode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--families", default="rewrite,compress,dsl_learn")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    blob = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ModelConfig(**blob["config"])
    model = build_model(cfg).to(args.device)
    model.load_state_dict(blob["model"])

    for fam in args.families.split(","):
        recs = load_sage_records(Path(f"data/sage/eval/{fam}.jsonl"), expect_train=False)[: args.n]
        prompts = [[BOS] + encode(r["prompt"]) for r in recs]
        gens = greedy_generate(model, prompts, args.device)
        print("=====", fam)
        for r, g in zip(recs, gens):
            print(f"  d={r['difficulty']} want={r['answer']!r:32s} got={decode(g)!r}")


if __name__ == "__main__":
    main()
