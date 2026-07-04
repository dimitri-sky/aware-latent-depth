"""Per-difficulty breakdown of the latest validity-gate runs."""
from __future__ import annotations

import csv
from collections import defaultdict

rows = [r for r in csv.DictReader(open("experiments/results.csv", encoding="utf-8"))
        if r["exp_id"] == "EXP-000" and r["difficulty"] != "all"]

# keep only the most recent run per model_id
latest_ts: dict[str, str] = {}
for r in rows:
    if r["timestamp"] > latest_ts.get(r["model_id"], ""):
        latest_ts[r["model_id"]] = r["timestamp"]
rows = [r for r in rows if r["timestamp"] == latest_ts[r["model_id"]]]

table: dict = defaultdict(dict)
for r in rows:
    table[(r["family"], r["difficulty"])][r["model_id"]] = float(r["value"])

fams = ["dsl_learn", "rewrite", "algo_exec", "rule_shift", "compress",
        "state_guard", "fresh_dsl"]
print(f"{'family':12s} {'tier':4s} {'4L':>7s} {'8L':>7s} {'delta':>7s}")
for f in fams:
    for d in "12345":
        row = table.get((f, d), {})
        a = row.get("gate-tfpp-4L")
        b = row.get("gate-tfpp-8L")
        if a is None and b is None:
            continue
        print(f"{f:12s} {d:4s} {a if a is not None else -1:7.3f} "
              f"{b if b is not None else -1:7.3f} {(b or 0) - (a or 0):+7.3f}")
