"""Ablation driver (Phase 5): expands one experiment config into the named ablation
rows from docs/PLAN.md that apply to the current variant set, and runs them.

Placeholder until Phase 3 selects a surviving variant; refuses to run before EXP-001
has logged results (guardrail 5: no large run without small-run signal).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path("experiments/results.csv")


def main() -> None:
    rows = list(csv.DictReader(RESULTS.open(encoding="utf-8")))
    if not any(r["exp_id"] == "EXP-001" for r in rows):
        print("Refusing: EXP-001 has no logged results yet (no large run without "
              "small-run signal). Run scripts/train_tiny.py first.")
        sys.exit(1)
    print("Ablation grid is defined in docs/PLAN.md; per-variant ablation configs are "
          "added under experiments/configs/ once a variant survives Phase 3.")


if __name__ == "__main__":
    main()
