"""Adjudicate H1 against the EXP-001B pre-registration (agent/log/EXP-001B.md).

Pools EXP-001 (s0-2) with EXP-001B-SEEDS (s3-5), reports EXP-001B-ALGO, and
checks the matched-FLOP control EXP-001B-FM against EXP-001 loop4.

    python scripts/adjudicate_exp001b.py --csv experiments/results_pod_b.csv
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict

MARGIN = 0.03  # +3.0 accuracy points, family mean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="experiments/results_pod_b.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    acc = defaultdict(list)
    for r in rows:
        if r["metric"] == "accuracy" and r["split"] == "eval":
            acc[(r["exp_id"], r["model_id"], r["family"], int(r["seed"]))].append(
                float(r["value"]))
    mean = {k: st.mean(v) for k, v in acc.items()}

    def get(exp, model, fam, seed):
        return mean.get((exp, model, fam, seed))

    def gate(deltas):
        m = st.mean(deltas)
        sd = st.stdev(deltas)
        ok = m >= MARGIN and m > 2 * sd
        return f"mean={m:+.3f} sd={sd:.3f} 2xSD={2*sd:.3f} -> {'PASS' if ok else 'FAIL'}"

    print("=== 1) EXP-001B-ALGO: loop4 vs loop1 on algo_exec (3 seeds) ===")
    for m in ("V1-loop1", "V1-loop4"):
        vals = [get("EXP-001B-ALGO", m, "algo_exec", s) for s in (0, 1, 2)]
        print(f"  {m}: {[round(v, 3) for v in vals]} median={st.median(vals):.3f}")
    d = [get("EXP-001B-ALGO", "V1-loop4", "algo_exec", s)
         - get("EXP-001B-ALGO", "V1-loop1", "algo_exec", s) for s in (0, 1, 2)]
    print(f"  deltas={[round(x, 3) for x in d]}  {gate(d)}")

    print()
    print("=== 2) Pooled 6 seeds (EXP-001 s0-2 + EXP-001B-SEEDS s3-5) ===")
    pooled = {}
    for fam in ("rewrite", "dsl_learn"):
        deltas = []
        for s in (0, 1, 2):
            deltas.append(get("EXP-001", "V1-loop4", fam, s)
                          - get("EXP-001", "V1-loop1", fam, s))
        for s in (3, 4, 5):
            deltas.append(get("EXP-001B-SEEDS", "V1-loop4", fam, s)
                          - get("EXP-001B-SEEDS", "V1-loop1", fam, s))
        pooled[fam] = deltas
        print(f"  {fam}: deltas={[round(x, 3) for x in deltas]}")
        print(f"    {gate(deltas)}")

    print()
    print("=== 3) EXP-001B-FM: loop1 @ 15000 steps vs EXP-001 loop4 @ 6000 ===")
    for fam in ("rewrite", "dsl_learn"):
        fm = [get("EXP-001B-FM", "V1-loop1-fm", fam, s) for s in (0, 1, 2)]
        l4 = [get("EXP-001", "V1-loop4", fam, s) for s in (0, 1, 2)]
        print(f"  {fam}: loop1-fm={[round(v, 3) for v in fm]} med={st.median(fm):.3f}"
              f" | loop4={[round(v, 3) for v in l4]} med={st.median(l4):.3f}"
              f" | loop4 - fm = {st.median(l4) - st.median(fm):+.3f}")


if __name__ == "__main__":
    main()
