"""Scoring: dispatch on the instance's machine-checkable scoring spec.

- exact:    normalized string equality
- numeric:  integer equality (tolerant of surrounding text on the first line)
- dsl_exec: parse the model's output as a DSL expression, execute it with the
            instance's private interpreter, compare to target. Never string match.
"""
from __future__ import annotations

import re

from ..generators import dsl_core as dc


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def _normalize(s: str) -> str:
    return " ".join(s.strip().lower().split())


def score_output(output: str, answer: str, scoring: dict) -> bool:
    kind = scoring["type"]
    line = _first_line(output)

    if kind == "exact":
        return _normalize(line) == _normalize(answer)

    if kind == "numeric":
        m = re.search(r"-?\d+", line)
        return m is not None and int(m.group()) == int(answer)

    if kind == "dsl_exec":
        name_to_op = scoring["name_to_op"]
        target = scoring["target"]
        try:
            expr = dc.parse_program(line)
        except dc.DslError:
            return False
        if scoring.get("require_op") and not isinstance(expr, list):
            return False  # bare literal rejected
        try:
            val = dc.evaluate(expr, name_to_op)
        except dc.DslError:
            return False
        return val == target

    raise ValueError(f"unknown scoring type {kind}")
