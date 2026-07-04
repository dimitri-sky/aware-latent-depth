"""Byte-level tokenizer: 256 byte values + BOS/EOS/PAD.

Chosen so no tokenizer advantage exists between variants (docs/BASELINES.md): SAGE
tasks and TinyStories are ASCII-dominant, and byte-level removes vocabulary size as a
confound at the 10-50M scale.
"""
from __future__ import annotations

BOS, EOS, PAD = 256, 257, 258
VOCAB_SIZE = 259


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8", errors="replace"))


def decode(ids: list[int]) -> str:
    return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")
