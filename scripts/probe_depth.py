"""EXP-000B: seed-noise (A/A) + maximum-contrast depth probe, parallel on one GPU.

Jobs:
  - A/A: two identical tf_pp-4L runs, seeds 100/200, algo_exec+rewrite mix
    -> quantifies the noise floor of the gate protocol itself.
  - Depth: tf_pp 2L vs 16L (max contrast, ~8x depth), per family in
    {rewrite, dsl_learn, algo_exec}, seeds {0, 1} -> 12 runs.

Runs N workers concurrently (tiny models leave a big GPU mostly idle); each worker
writes to its own results CSV shard (AWARE_RESULTS_CSV), merged into
experiments/results.csv at the end (append, dedup by run_id).

    python scripts/probe_depth.py --workers 3
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path("experiments/results.csv")


def job_list() -> list[dict]:
    jobs = []
    for arm, seed in [("AA-armA", 100), ("AA-armB", 200)]:
        jobs.append(dict(exp_id="EXP-000B", model_id=arm, n_layers=4,
                         families="algo_exec,rewrite", seed=seed,
                         notes="A/A seed-noise floor"))
    for fam in ["rewrite", "dsl_learn", "algo_exec"]:
        for n_layers in [2, 16]:
            for seed in [0, 1]:
                jobs.append(dict(exp_id="EXP-000B", model_id=f"probe-{n_layers}L-{fam}",
                                 n_layers=n_layers, families=fam, seed=seed,
                                 notes="max-contrast depth probe"))
    return jobs


def run_job(job: dict, shard: Path) -> int:
    env = dict(os.environ, AWARE_RESULTS_CSV=str(shard))
    cmd = [sys.executable, "scripts/train_single.py",
           "--exp-id", job["exp_id"], "--model-id", job["model_id"],
           "--n-layers", str(job["n_layers"]), "--families", job["families"],
           "--seed", str(job["seed"]), "--notes", job["notes"]]
    print(f"[launch] {job['model_id']} s{job['seed']}", flush=True)
    r = subprocess.run(cmd, env=env)
    print(f"[done rc={r.returncode}] {job['model_id']} s{job['seed']}", flush=True)
    return r.returncode


def merge_shards(shards: list[Path]) -> None:
    existing = set()
    with RESULTS.open(encoding="utf-8") as fh:
        header = fh.readline()
        for row in csv.reader(fh):
            if row:
                existing.add(row[0])
    with RESULTS.open("a", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        for shard in shards:
            if not shard.exists():
                continue
            with shard.open(encoding="utf-8") as fh:
                fh.readline()  # skip shard header
                for row in csv.reader(fh):
                    if row and row[0] not in existing:
                        w.writerow(row)
                        existing.add(row[0])
            shard.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    jobs = job_list()
    shards = []
    rcs = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for i, job in enumerate(jobs):
            shard = Path(f"experiments/results_shard_{i}.csv")
            # seed shard with header so merge logic is uniform
            shard.write_text(RESULTS.open(encoding="utf-8").readline(), encoding="utf-8")
            shards.append(shard)
            futures.append(pool.submit(run_job, job, shard))
        rcs = [f.result() for f in futures]

    merge_shards(shards)
    failed = sum(1 for rc in rcs if rc != 0)
    print(f"PROBE COMPLETE: {len(rcs) - failed}/{len(rcs)} jobs ok, merged into {RESULTS}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
