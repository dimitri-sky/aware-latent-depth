"""Download TinyStories and write a plain-text corpus for the LM data mix.

    python scripts/get_tinystories.py --max-docs 200000
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-docs", type=int, default=200_000)
    ap.add_argument("--out", type=Path, default=Path("data/tinystories/train.txt"))
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for rec in ds:
            fh.write(rec["text"].strip() + "\n\n")
            n += 1
            if n >= args.max_docs:
                break
    print(f"wrote {n} stories -> {args.out}")


if __name__ == "__main__":
    main()
