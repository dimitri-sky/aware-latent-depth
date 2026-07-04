"""Status of the per-family validity gate from results.csv (survives crashes)."""
from __future__ import annotations

import csv

SINCE = "2026-07-04T00:01"
FAMS = ["dsl_learn", "rewrite", "algo_exec", "rule_shift", "compress"]

rows = [r for r in csv.DictReader(open("experiments/results.csv", encoding="utf-8"))
        if r["timestamp"] > SINCE and r["difficulty"] == "all"]
acc = {r["model_id"]: float(r["value"]) for r in rows}

print(f"{'family':12s} {'4L':>7s} {'8L':>7s} {'delta':>8s}")
separated = 0
missing = []
for f in FAMS:
    a = acc.get(f"gate-tfpp-4L-{f}")
    b = acc.get(f"gate-tfpp-8L-{f}")
    aa = f"{a:.3f}" if a is not None else "MISS"
    bb = f"{b:.3f}" if b is not None else "MISS"
    d = f"{b - a:+.3f}" if a is not None and b is not None else ""
    if a is not None and b is not None and b - a >= 0.05:
        separated += 1
        d += " *"
    if a is None or b is None:
        missing.append(f)
    print(f"{f:12s} {aa:>7s} {bb:>7s} {d:>8s}")
print(f"\nseparated (>= +0.05): {separated}/5   missing runs: {missing or 'none'}")
