"""A/A placebo: two identical baseline configs, different torch data-order seeds only,
through the full train+eval pipeline. If the compare step calls the difference
'significant' relative to the pre-registered margin, the harness or stats are broken.

    python scripts/aa_test.py [--steps 800]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from models import ModelConfig  # noqa: E402
from train.train import TrainConfig, train_one  # noqa: E402

MARGIN = 0.03  # pre-registered decision margin (agent/decision_policy.md)


def fam_mean(results: dict) -> float:
    vals = []
    for fam, stats in results.items():
        if fam.startswith("_"):
            continue
        n = sum(v[0] for v in stats.values())
        c = sum(v[1] for v in stats.values())
        vals.append(c / max(1, n))
    return statistics.mean(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    mcfg = ModelConfig(arch="tf_pp", d_model=256, n_layers=4, n_heads=4, n_kv_heads=2,
                       d_ff=704, max_seq_len=1024)
    means = []
    for arm, seed in [("AA-armA", 100), ("AA-armB", 200)]:
        tc = TrainConfig(families=["algo_exec", "rewrite"], steps=args.steps,
                         seq_len=768, seed=seed, eval_limit=200)
        r = train_one(mcfg, tc, exp_id="EXP-AA", model_id=arm, device=args.device,
                      notes="A/A placebo")
        means.append(fam_mean(r["results"]))

    diff = abs(means[0] - means[1])
    print(f"A/A family-mean difference: {diff:.4f} (margin {MARGIN})")
    if diff >= MARGIN:
        print("A/A WARNING: identical models differ by more than the decision margin; "
              "seed noise at this budget swamps the margin — raise seeds/steps before "
              "trusting single-seed comparisons.")
        sys.exit(1)
    print("A/A OK")


if __name__ == "__main__":
    main()
