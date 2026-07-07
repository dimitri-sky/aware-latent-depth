"""Merge a pulled pod results.csv into the local ledger (row-level dedupe on the
run_exp.py key: run_id, family, difficulty, metric). Append-only; never rewrites.

    python scripts/merge_results.py --incoming experiments/session_c_pull/workspace/aware/experiments/results.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

LOCAL = Path("experiments/results.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incoming", type=Path, required=True)
    args = ap.parse_args()

    def key(row):
        return (row[0], row[7], row[8], row[11])

    existing = set()
    with LOCAL.open(encoding="utf-8") as fh:
        fh.readline()
        for row in csv.reader(fh):
            if row:
                existing.add(key(row))

    added = 0
    with LOCAL.open("a", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        with args.incoming.open(encoding="utf-8") as fh:
            fh.readline()
            for row in csv.reader(fh):
                if row and key(row) not in existing:
                    w.writerow(row)
                    existing.add(key(row))
                    added += 1
    print(f"merged {added} new rows into {LOCAL}")


if __name__ == "__main__":
    main()
