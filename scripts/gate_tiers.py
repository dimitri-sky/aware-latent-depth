"""Per-tier 4L vs 8L deltas for the per-family gate runs (aggregates can mask
separation that exists only on hard tiers)."""
from __future__ import annotations

import csv

SINCE = "2026-07-04T00:01"
FAMS = ["dsl_learn", "rewrite", "algo_exec", "rule_shift", "compress"]

rows = [r for r in csv.DictReader(open("experiments/results.csv", encoding="utf-8"))
        if r["timestamp"] > SINCE and r["difficulty"] != "all"]
acc = {(r["model_id"], r["difficulty"]): float(r["value"]) for r in rows}

print(f"{'family':12s} {'tier':4s} {'4L':>7s} {'8L':>7s} {'delta':>8s}")
for f in FAMS:
    for d in "12345":
        a = acc.get((f"gate-tfpp-4L-{f}", d))
        b = acc.get((f"gate-tfpp-8L-{f}", d))
        if a is None and b is None:
            continue
        aa = f"{a:.3f}" if a is not None else "MISS"
        bb = f"{b:.3f}" if b is not None else "MISS"
        dd = f"{b - a:+.3f}" if a is not None and b is not None else ""
        print(f"{f:12s} {d:4s} {aa:>7s} {bb:>7s} {dd:>8s}")
