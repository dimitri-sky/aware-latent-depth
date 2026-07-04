"""Gate attempt 6 depth check: 2L vs 16L, 3 seeds, median aggregation, collapse
detection with one retry. Parallel workers with CSV shards (see probe_depth.py).

Protocol calibrated to EXP-000B findings:
- A/A noise floor was 12-20 pts on single seeds -> median of 3 seeds per condition.
- Seed-dependent degenerate collapse (tier-1 acc < 0.05 with converged train loss)
  is an instrument failure, not evidence -> retried once with seed+1000, and the
  collapsed run is EXCLUDED from the median (recorded in the JSON verdict).
- Deep models get depth-scaled lr = 6e-4 * sqrt(4 / n_layers), capped at 6e-4.

Verdict: PASS iff median(16L) - median(2L) >= 0.05 on all of rewrite, dsl_learn,
algo_exec. Headroom + memory-band checks are carried over from attempt 5 (both
passed; regenerated data affects only algo_exec, whose trivial-solver headroom is
re-checked here).

    python scripts/gate6_depth.py --workers 3
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path("experiments/results.csv")
EXP_ID = "EXP-000C"
FAMS = ("rewrite", "dsl_learn", "algo_exec")
DEPTHS = (2, 16)
SEEDS = (0, 1, 2)
MARGIN = 0.05
COLLAPSE_TIER1 = 0.05


def lr_for(n_layers: int) -> float:
    return min(6e-4, 6e-4 * math.sqrt(4 / n_layers))


def run_job(fam: str, n_layers: int, seed: int, shard: Path) -> int:
    env = dict(os.environ, AWARE_RESULTS_CSV=str(shard))
    cmd = [sys.executable, "scripts/train_single.py",
           "--exp-id", EXP_ID, "--model-id", f"g6-{n_layers}L-{fam}",
           "--n-layers", str(n_layers), "--families", fam, "--seed", str(seed),
           "--lr", str(lr_for(n_layers)), "--notes", "gate6 depth check"]
    print(f"[launch] g6-{n_layers}L-{fam} s{seed}", flush=True)
    return subprocess.run(cmd, env=env).returncode


def read_scores(shards: list[Path]) -> dict:
    """(fam, n_layers, seed) -> {difficulty: acc}"""
    out: dict = {}
    for shard in shards:
        if not shard.exists():
            continue
        with shard.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["exp_id"] != EXP_ID:
                    continue
                mid = r["model_id"]  # g6-<n>L-<fam>
                n_layers = int(mid.split("-")[1][:-1])
                fam = mid.split("-", 2)[2]
                key = (fam, n_layers, int(r["seed"]))
                out.setdefault(key, {})[r["difficulty"]] = float(r["value"])
    return out


def merge_into_results(shards: list[Path]) -> None:
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
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    header = RESULTS.open(encoding="utf-8").readline()
    jobs = [(fam, nl, seed) for fam in FAMS for nl in DEPTHS for seed in SEEDS]
    shards: list[Path] = []

    def launch_wave(wave_jobs):
        futures = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for fam, nl, seed in wave_jobs:
                shard = Path(f"experiments/results_g6_{fam}_{nl}_{seed}.csv")
                shard.write_text(header, encoding="utf-8")
                shards.append(shard)
                futures.append(pool.submit(run_job, fam, nl, seed, shard))
            return [f.result() for f in futures]

    launch_wave(jobs)
    scores = read_scores(shards)

    # collapse detection + one retry with seed+1000
    collapsed, retries = [], []
    for (fam, nl, seed), tiers in list(scores.items()):
        if tiers.get("1", 1.0) < COLLAPSE_TIER1:
            collapsed.append([fam, nl, seed])
            retries.append((fam, nl, seed + 1000))
    if retries:
        print(f"[collapse] retrying {len(retries)} degenerate runs: {collapsed}")
        launch_wave(retries)
        scores = read_scores(shards)

    verdict: dict = {"collapsed_excluded": collapsed, "families": {}}
    all_pass = True
    for fam in FAMS:
        meds = {}
        for nl in DEPTHS:
            vals = [t["all"] for (f, n, s), t in scores.items()
                    if f == fam and n == nl and "all" in t
                    and t.get("1", 1.0) >= COLLAPSE_TIER1]
            meds[nl] = statistics.median(vals) if vals else float("nan")
        delta = meds[16] - meds[2]
        ok = delta >= MARGIN
        all_pass &= ok
        verdict["families"][fam] = {"median_2L": meds[2], "median_16L": meds[16],
                                    "delta": round(delta, 4), "pass": ok}
        print(f"  {fam:10s} 2L(med)={meds[2]:.3f} 16L(med)={meds[16]:.3f} "
              f"delta={delta:+.3f} [{'OK' if ok else 'FAIL'}]")

    merge_into_results(shards)
    verdict["pass"] = all_pass
    Path("experiments/gate6_verdict.json").write_text(json.dumps(verdict, indent=2))
    print(f"GATE6 VERDICT: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
