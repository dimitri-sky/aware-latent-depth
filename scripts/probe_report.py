"""EXP-000B report: A/A seed-noise floor + 2L-vs-16L max-contrast depth deltas."""
from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

src = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/results.csv")
rows = [r for r in csv.DictReader(src.open(encoding="utf-8"))
        if r["exp_id"] == "EXP-000B" and r["difficulty"] == "all"]

acc: dict = defaultdict(dict)
for r in rows:
    acc[r["model_id"]].setdefault(r["family"], {})[int(r["seed"])] = float(r["value"])

print("== raw rows ==")
for mid in sorted(acc):
    for fam, seeds in sorted(acc[mid].items()):
        cells = " ".join(f"s{s}={v:.3f}" for s, v in sorted(seeds.items()))
        print(f"  {mid:24s} {fam:10s} {cells}")

print("\n== A/A noise floor ==")
for fam in ("algo_exec", "rewrite"):
    a = acc.get("AA-armA", {}).get(fam, {})
    b = acc.get("AA-armB", {}).get(fam, {})
    if a and b:
        va, vb = list(a.values())[0], list(b.values())[0]
        print(f"  {fam:10s} armA={va:.3f} armB={vb:.3f} |diff|={abs(va - vb):.3f}")

print("\n== 2L vs 16L ==")
for fam in ("rewrite", "dsl_learn", "algo_exec"):
    lo = acc.get(f"probe-2L-{fam}", {}).get(fam, {})
    hi = acc.get(f"probe-16L-{fam}", {}).get(fam, {})
    for s in sorted(lo):
        if s in hi:
            print(f"  {fam:10s} s{s}: 2L={lo[s]:.3f} 16L={hi[s]:.3f} delta={hi[s] - lo[s]:+.3f}")
    if lo and hi:
        print(f"  {fam:10s} MEAN delta={statistics.mean(hi.values()) - statistics.mean(lo.values()):+.3f}")
