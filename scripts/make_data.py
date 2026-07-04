"""Generate SAGE splits as JSONL. Train/eval seed ranges are disjoint by construction;
this writer refuses to emit an instance whose seed belongs to the other split.

    python scripts/make_data.py --split train --per-family 4000
    python scripts/make_data.py --split eval  --per-family 400
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sage.generators import FAMILIES  # noqa: E402
from sage.generators.base import (  # noqa: E402
    DIFFICULTIES, EVAL_SEED_LO, TRAIN_SEED_LO, assert_split,
)
from sage.contamination.audit import CANARY_PREFIX  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "eval"], required=True)
    ap.add_argument("--per-family", type=int, default=1000)
    ap.add_argument("--families", default="all")
    ap.add_argument("--out-dir", type=Path, default=Path("data/sage"))
    args = ap.parse_args()

    families = list(FAMILIES) if args.families == "all" else args.families.split(",")
    base_seed = TRAIN_SEED_LO if args.split == "train" else EVAL_SEED_LO
    out_dir = args.out_dir / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    for fam in families:
        gen = FAMILIES[fam]
        path = out_dir / f"{fam}.jsonl"
        n_written = 0
        with path.open("w", encoding="utf-8") as fh:
            if args.split == "eval":
                fh.write(json.dumps({"kind": "canary",
                                     "text": f"{CANARY_PREFIX}{uuid.uuid4()}"}) + "\n")
            i = 0
            while n_written < args.per_family:
                seed = base_seed + i
                i += 1
                assert_split(seed, args.split)
                difficulty = DIFFICULTIES[n_written % len(DIFFICULTIES)]
                try:
                    inst = gen(seed, difficulty)
                except Exception as e:  # rare degenerate draws are skipped, logged
                    print(f"[skip] {fam} seed={seed} d={difficulty}: {e}", file=sys.stderr)
                    continue
                fh.write(json.dumps(inst.to_dict()) + "\n")
                n_written += 1
        print(f"{fam}: wrote {n_written} -> {path}")


if __name__ == "__main__":
    main()
