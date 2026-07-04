"""Generator correctness: determinism, seed discipline, answer verifiability.

A scorer/generator bug is the classic source of fake architectural gains; these run
before any experiment is logged (guardrail: scorer correctness tests).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from sage.generators import FAMILIES
from sage.generators.base import DIFFICULTIES, assert_split, split_of_seed
from sage.generators import dsl_core as dc
from sage.scoring import score_output

SAMPLE_SEEDS_TRAIN = [0, 7, 123, 999_999]
SAMPLE_SEEDS_EVAL = [2_000_000, 2_000_017, 2_099_999]


@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_deterministic(family, difficulty):
    a = FAMILIES[family](42, difficulty)
    b = FAMILIES[family](42, difficulty)
    assert a.prompt == b.prompt and a.answer == b.answer and a.trace == b.trace


@pytest.mark.parametrize("family", list(FAMILIES))
def test_own_answer_scores_correct(family):
    """The ground-truth answer must pass its own scoring spec (known-answer test)."""
    for seed in SAMPLE_SEEDS_TRAIN + SAMPLE_SEEDS_EVAL:
        for difficulty in DIFFICULTIES:
            inst = FAMILIES[family](seed, difficulty)
            assert score_output(inst.answer, inst.answer, inst.scoring), (
                f"{family} seed={seed} d={difficulty}: own answer rejected")


@pytest.mark.parametrize("family", list(FAMILIES))
def test_wrong_answer_scores_incorrect(family):
    for seed in SAMPLE_SEEDS_TRAIN:
        inst = FAMILIES[family](seed, 3)
        wrong = "999" if inst.scoring["type"] != "exact" else inst.answer + " zzz"
        assert not score_output(wrong, inst.answer, inst.scoring)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_prompt_ends_with_answer_marker(family):
    inst = FAMILIES[family](5, 2)
    assert inst.prompt.rstrip().endswith("ANSWER:")
    assert inst.trace, "trace required for CoT baseline training"


def test_seed_discipline():
    assert split_of_seed(0) == "train"
    assert split_of_seed(2_000_000) == "eval"
    with pytest.raises(ValueError):
        split_of_seed(1_500_000)
    with pytest.raises(ValueError):
        assert_split(2_000_001, "train")


def test_dsl_exec_scoring_by_execution_not_string():
    """A differently-written but semantically correct program must be accepted."""
    name_to_op = {"foo": "add", "bar": "mul"}
    scoring = {"type": "dsl_exec", "name_to_op": name_to_op, "target": 7, "require_op": True}
    assert score_output("(foo 3 4)", "(foo 4 3)", scoring)     # different surface form
    assert score_output("(foo (bar 1 3) 4)", "?", scoring)      # nested alternative
    assert not score_output("7", "?", scoring)                  # bare literal rejected
    assert not score_output("(foo 3 5)", "?", scoring)          # wrong value


def test_traced_form_contains_trace():
    inst = FAMILIES["algo_exec"](11, 2)
    assert "THINK:" in inst.traced_text()
    assert inst.trace.splitlines()[0] in inst.traced_text()
