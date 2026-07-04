"""Summarize experiments/results.csv: family-mean accuracy per run, seed-aggregated
comparison per experiment. Usage: python scripts/report.py [--exp EXP-001]
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path("experiments/results.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None)
    args = ap.parse_args()

    rows = list(csv.DictReader(RESULTS.open(encoding="utf-8")))
    rows = [r for r in rows if r["difficulty"] == "all" and r["metric"] == "accuracy"]
    if args.exp:
        rows = [r for r in rows if r["exp_id"] == args.exp]

    # (exp, model, family) -> list of (seed, value)
    acc: dict = defaultdict(list)
    fpc: dict = {}
    for r in rows:
        acc[(r["exp_id"], r["model_id"], r["family"])].append(float(r["value"]))
        fpc[(r["exp_id"], r["model_id"])] = float(r["infer_flops_per_correct"])

    by_exp: dict = defaultdict(lambda: defaultdict(dict))
    for (exp, mid, fam), vals in acc.items():
        by_exp[exp][mid][fam] = vals

    for exp in sorted(by_exp):
        print(f"\n== {exp} ==")
        fams = sorted({f for m in by_exp[exp].values() for f in m})
        header = f"{'model':14s}" + "".join(f"{f[:11]:>13s}" for f in fams) + f"{'fam-mean':>10s}{'flops/corr':>12s}"
        print(header)
        for mid in sorted(by_exp[exp]):
            cells, means = [], []
            for f in fams:
                vals = by_exp[exp][mid].get(f, [])
                if vals:
                    mu = statistics.mean(vals)
                    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                    cells.append(f"{mu:.3f}±{sd:.2f}")
                    means.append(mu)
                else:
                    cells.append("-")
            fam_mean = statistics.mean(means) if means else 0.0
            print(f"{mid:14s}" + "".join(f"{c:>13s}" for c in cells)
                  + f"{fam_mean:>10.3f}{fpc.get((exp, mid), 0):>12.3g}")


if __name__ == "__main__":
    main()
