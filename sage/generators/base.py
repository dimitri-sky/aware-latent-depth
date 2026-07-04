"""Shared generator infrastructure: seed discipline, instance schema, RNG."""
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict

# Disjoint by construction (guardrail 8). The training data writer refuses eval seeds.
TRAIN_SEED_LO, TRAIN_SEED_HI = 0, 1_000_000
EVAL_SEED_LO, EVAL_SEED_HI = 2_000_000, 2_100_000

DIFFICULTIES = (1, 2, 3, 4, 5)


def split_of_seed(seed: int) -> str:
    if TRAIN_SEED_LO <= seed < TRAIN_SEED_HI:
        return "train"
    if EVAL_SEED_LO <= seed < EVAL_SEED_HI:
        return "eval"
    raise ValueError(f"seed {seed} outside both split ranges")


def assert_split(seed: int, split: str) -> None:
    actual = split_of_seed(seed)
    if actual != split:
        raise ValueError(f"seed {seed} belongs to split '{actual}', not '{split}'")


def rng_for(family: str, difficulty: int, seed: int) -> random.Random:
    # Family/difficulty folded into the stream so identical seeds do not produce
    # correlated instances across families.
    return random.Random(f"{family}|{difficulty}|{seed}")


@dataclass
class Instance:
    family: str
    difficulty: int
    seed: int
    prompt: str          # plain form, ends with "ANSWER:"
    answer: str          # constrained short string
    trace: str           # ground-truth step trace (for CoT baseline training)
    scoring: dict        # {"type": "exact"} | {"type": "numeric"} | {"type": "dsl_exec", ...}
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def plain_text(self) -> str:
        return self.prompt + " " + self.answer

    def traced_text(self) -> str:
        return self.prompt.replace("ANSWER:", "THINK:\n" + self.trace + "\nANSWER:") + " " + self.answer
