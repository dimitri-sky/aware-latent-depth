"""Rebuild aware_src.zip for pod upload: source + configs + results.csv header
carrier, no paper archives / checkpoints / data (pods regenerate data from seeds).

    python scripts/build_src_zip.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRS = ["sage", "models", "train", "eval", "tests",
                "experiments/configs", "scripts"]
INCLUDE_FILES = ["experiments/results.csv", "requirements.txt", "Makefile"]
EXCLUDE_SUFFIXES = {".pt", ".pyc", ".zip", ".tar.gz", ".png", ".webp", ".pdf"}


def main() -> None:
    out = ROOT / "aware_src.zip"
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for d in INCLUDE_DIRS:
            for p in sorted((ROOT / d).rglob("*")):
                if not p.is_file() or p.suffix in EXCLUDE_SUFFIXES \
                        or "__pycache__" in p.parts:
                    continue
                z.write(p, p.relative_to(ROOT).as_posix())
                n += 1
        for f in INCLUDE_FILES:
            z.write(ROOT / f, f)
            n += 1
    print(f"wrote {out} ({n} files, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
