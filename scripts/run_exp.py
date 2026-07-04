"""Parallel experiment runner: YAML config (models x seeds) -> N workers with CSV
shards -> merged into experiments/results.csv (row-level dedup).

    python scripts/run_exp.py --config experiments/configs/exp001_loop_falsifier.yaml --workers 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

RESULTS = Path("experiments/results.csv")


def merge_shards(shards: list[Path]) -> None:
    def key(row):
        return (row[0], row[7], row[8], row[11])
    existing = set()
    with RESULTS.open(encoding="utf-8") as fh:
        fh.readline()
        for row in csv.reader(fh):
            if row:
                existing.add(key(row))
    with RESULTS.open("a", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        for shard in shards:
            if not shard.exists():
                continue
            with shard.open(encoding="utf-8") as fh:
                fh.readline()
                for row in csv.reader(fh):
                    if row and key(row) not in existing:
                        w.writerow(row)
                        existing.add(key(row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seeds", default=None)
    args = ap.parse_args()

    spec = yaml.safe_load(args.config.read_text())
    exp_id = spec["exp_id"]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else spec.get("seeds", [0])
    train = spec["train"]
    header = RESULTS.open(encoding="utf-8").readline()

    def run_job(m: dict, seed: int, shard: Path) -> int:
        env = dict(os.environ, AWARE_RESULTS_CSV=str(shard))
        mkw = {k: v for k, v in m.items() if k != "id"}
        cmd = [sys.executable, "scripts/train_single.py",
               "--exp-id", exp_id, "--model-id", m["id"],
               "--model-json", json.dumps(mkw),
               "--families", ",".join(train["families"]),
               "--steps", str(train.get("steps", 4000)),
               "--seed", str(seed),
               "--lr", str(train.get("lr", 3e-4))]
        print(f"[launch] {m['id']} s{seed}", flush=True)
        rc = subprocess.run(cmd, env=env).returncode
        print(f"[done rc={rc}] {m['id']} s{seed}", flush=True)
        return rc

    shards, futures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for seed in seeds:
            for m in spec["models"]:
                shard = Path(f"experiments/results_{exp_id}_{m['id']}_{seed}.csv")
                shard.write_text(header, encoding="utf-8")
                shards.append(shard)
                futures.append(pool.submit(run_job, m, seed, shard))
        rcs = [f.result() for f in futures]

    merge_shards(shards)
    failed = sum(1 for rc in rcs if rc)
    print(f"EXP COMPLETE: {len(rcs) - failed}/{len(rcs)} jobs ok")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
