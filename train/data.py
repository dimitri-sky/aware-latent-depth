"""Data pipeline: SAGE JSONL -> fixed-length training batches (+ optional TinyStories).

Sequence layout per instance: BOS + prompt_bytes + ' ' + answer_bytes + EOS, padded.
Loss is computed over the full sequence (identical treatment for every model; PAD
positions masked with -100). `traced=True` uses the traced form (CoT baseline
training). Instances that do not fit `seq_len` are skipped and counted.

The loader refuses eval-split files for training (guardrail 8): it checks record seeds
against the training range at load time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from sage.generators.base import TRAIN_SEED_HI, TRAIN_SEED_LO

from .tokenizer import BOS, EOS, PAD, encode


def load_sage_records(path: Path, expect_train: bool) -> list[dict]:
    records = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        rec = json.loads(ln)
        if rec.get("kind") == "canary":
            continue
        if expect_train and not (TRAIN_SEED_LO <= rec["seed"] < TRAIN_SEED_HI):
            raise ValueError(f"eval-split seed {rec['seed']} in training data: {path}")
        records.append(rec)
    return records


def record_to_ids(rec: dict, traced: bool = False) -> list[int]:
    if traced:
        text = rec["prompt"].replace("ANSWER:", "THINK:\n" + rec["trace"] + "\nANSWER:")
    else:
        text = rec["prompt"]
    return [BOS] + encode(text + " " + rec["answer"]) + [EOS]


class SageDataset:
    def __init__(self, data_dir: Path, families: list[str], seq_len: int,
                 traced: bool = False, expect_train: bool = True):
        self.seq_len = seq_len
        self.sequences: list[list[int]] = []
        self.skipped = 0
        for fam in families:
            for rec in load_sage_records(data_dir / f"{fam}.jsonl", expect_train):
                ids = record_to_ids(rec, traced)
                if len(ids) > seq_len:
                    self.skipped += 1
                    continue
                self.sequences.append(ids)

    def batches(self, batch_size: int, rng: np.random.Generator, device):
        order = rng.permutation(len(self.sequences))
        for i in range(0, len(order) - batch_size + 1, batch_size):
            chunk = [self.sequences[j] for j in order[i : i + batch_size]]
            width = max(len(s) for s in chunk)
            x = torch.full((batch_size, width), PAD, dtype=torch.long)
            y = torch.full((batch_size, width), -100, dtype=torch.long)
            for r, seq in enumerate(chunk):
                t = torch.tensor(seq, dtype=torch.long)
                x[r, : len(seq)] = t
                y[r, : len(seq) - 1] = t[1:]
            yield x.to(device), y.to(device)

    def tokens_per_epoch(self) -> int:
        return sum(len(s) for s in self.sequences)


def load_tinystories(path: Path, seq_len: int, max_docs: int | None = None) -> list[list[int]]:
    """Plain-text TinyStories (one story per blank-line-separated block) -> sequences."""
    out: list[list[int]] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        ids = [BOS] + encode(block)[: seq_len - 2] + [EOS]
        out.append(ids)
        if max_docs and len(out) >= max_docs:
            break
    return out
