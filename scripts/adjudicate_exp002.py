"""Adjudicate H2 against the EXP-002 pre-registration (agent/log/EXP-002.md).

Headline: V2-delta beats param-matched B2-6L by >= 3.0 points on the
rule_shift + compress + state_guard family mean, 3 seeds, mean diff > 2x pooled
seed SD. Dissociation: rule_shift gain must exceed algo_exec gain (EXP-002-AX).

    python scripts/adjudicate_exp002.py --csv experiments/results.csv
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict

MARGIN = 0.03  # +3.0 accuracy points, family mean

FAMILIES = [
    ("EXP-002-RS", "rule_shift"),
    ("EXP-002-CP", "compress"),
    ("EXP-002-SG", "state_guard"),
]
SEEDS = (0, 1, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="experiments/results.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    acc = {}
    for r in rows:
        if (r["metric"] == "accuracy" and r["split"] == "eval"
                and r["difficulty"] == "all"):
            acc[(r["exp_id"], r["model_id"], r["family"], int(r["seed"]))] = \
                float(r["value"])

    def get(exp, model, fam, seed):
        return acc.get((exp, model, fam, seed))

    print("=== EXP-002 headline: V2-delta vs B2-6L, 3 memory families ===")
    all_deltas = []
    fam_means = []
    for exp, fam in FAMILIES:
        v2 = [get(exp, "V2-delta", fam, s) for s in SEEDS]
        b2 = [get(exp, "B2-6L", fam, s) for s in SEEDS]
        if None in v2 or None in b2:
            print(f"  {fam}: INCOMPLETE v2={v2} b2={b2}")
            continue
        d = [a - b for a, b in zip(v2, b2)]
        all_deltas.extend(d)
        fam_means.append(st.mean(d))
        print(f"  {fam}: V2={[round(x,3) for x in v2]} (med {st.median(v2):.3f})"
              f"  B2={[round(x,3) for x in b2]} (med {st.median(b2):.3f})")
        print(f"    deltas={[round(x,3) for x in d]}  mean={st.mean(d):+.3f}")

    pooled_mean = st.mean(fam_means)
    pooled_sd = st.stdev(all_deltas)
    print(f"\n  pooled family-mean delta = {pooled_mean:+.4f} (margin {MARGIN:+.3f})")
    print(f"  pooled seed SD = {pooled_sd:.4f}  2xSD = {2*pooled_sd:.4f}")
    margin_ok = pooled_mean >= MARGIN
    stable_ok = pooled_mean > 2 * pooled_sd
    print(f"  margin gate:    {'PASS' if margin_ok else 'FAIL'}")
    print(f"  stability gate: {'PASS' if stable_ok else 'FAIL'}"
          f"  (mean {'>' if stable_ok else '<='} 2x pooled SD)")

    print("\n=== Dissociation: rule_shift gain vs algo_exec gain (EXP-002-AX) ===")
    rs_gain = fam_means[0] if fam_means else None
    ax_v2 = [get("EXP-002-AX", "V2-delta", "algo_exec", s) for s in SEEDS]
    ax_b2 = [get("EXP-002-AX", "B2-6L", "algo_exec", s) for s in SEEDS]
    if None in ax_v2 or None in ax_b2:
        print(f"  PENDING (EXP-002-AX incomplete) v2={ax_v2} b2={ax_b2}")
    else:
        ax_gain = st.mean([a - b for a, b in zip(ax_v2, ax_b2)])
        print(f"  algo_exec gain = {ax_gain:+.3f}  rule_shift gain = {rs_gain:+.3f}")
        print(f"  dissociation: {'PASS' if rs_gain > ax_gain else 'FAIL'}")

    print("\n=== Verdict inputs ===")
    print(f"  headline = margin {'PASS' if margin_ok else 'FAIL'}"
          f" + stability {'PASS' if stable_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
