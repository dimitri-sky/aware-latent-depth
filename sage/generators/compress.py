"""Compress: long context of variable assignments with updates and distractor prose;
query the final value of one variable (or an aggregate). Tests compression of long
context into reusable state. Difficulty scales context length and update count.
"""
from __future__ import annotations

from .base import Instance, rng_for

FAMILY = "compress"
_N_VARS = {1: 2, 2: 3, 3: 5, 4: 8, 5: 12}
_N_EVENTS = {1: 4, 2: 8, 3: 16, 4: 28, 5: 44}
_DISTRACTOR_RATE = {1: 0.15, 2: 0.25, 3: 0.35, 4: 0.45, 5: 0.5}
# Single-knob ramp (gate attempt 3 postmortem): tiers 1-2 are pure set/overwrite
# tracking (the actual compression skill); mod-100 increment/decrement arithmetic
# enters at tier 3; two-variable aggregates only at tier 4+.
_ARITH_TIER = 3

_NAMES = ["kap", "rud", "mel", "tov", "sil", "nar", "bex", "fum", "gid", "hol",
          "jyn", "wex", "pia", "quz", "vor", "yem"]
_DISTRACTORS = [
    "the lamp in the hall is on", "it rained briefly at noon", "a door closed somewhere",
    "the meter hummed quietly", "nothing else changed", "a cart rolled past outside",
    "the clock chimed once", "someone stacked the crates",
]


def generate(seed: int, difficulty: int) -> Instance:
    rng = rng_for(FAMILY, difficulty, seed)
    var_names = rng.sample(_NAMES, _N_VARS[difficulty])
    state: dict[str, int] = {}
    lines: list[str] = []
    trace_lines: list[str] = []

    n_events = _N_EVENTS[difficulty]
    for _ in range(n_events):
        if state and rng.random() < _DISTRACTOR_RATE[difficulty]:
            lines.append(rng.choice(_DISTRACTORS) + ".")
            continue
        v = rng.choice(var_names)
        kind = rng.random() if difficulty >= _ARITH_TIER else 0.0
        if v not in state or kind < 0.5:
            val = rng.randint(0, 99)
            state[v] = val
            lines.append(f"set {v} = {val}.")
            trace_lines.append(f"{v} := {val}")
        elif kind < 0.75:
            delta = rng.randint(1, 9)
            state[v] = (state[v] + delta) % 100
            lines.append(f"increase {v} by {delta}.")
            trace_lines.append(f"{v} := {state[v]}")
        else:
            delta = rng.randint(1, 9)
            state[v] = (state[v] - delta) % 100
            lines.append(f"decrease {v} by {delta}.")
            trace_lines.append(f"{v} := {state[v]}")

    assigned = sorted(state)
    if difficulty >= 4 and rng.random() < 0.4 and len(assigned) >= 2:
        a, b = rng.sample(assigned, 2)
        answer = (state[a] + state[b]) % 100
        question = f"What is ({a} + {b}) mod 100 now?"
        trace_lines.append(f"{a}={state[a]}, {b}={state[b]}, sum mod 100 = {answer}")
    else:
        a = rng.choice(assigned)
        answer = state[a]
        question = f"What is {a} now?"
        trace_lines.append(f"final {a} = {answer}")

    prompt = (
        "Track the values through the events. Values are mod 100.\n"
        + " ".join(lines)
        + f"\n{question}\nANSWER:"
    )
    return Instance(
        family=FAMILY, difficulty=difficulty, seed=seed,
        prompt=prompt, answer=str(answer), trace="\n".join(trace_lines),
        scoring={"type": "numeric"},
        meta={"n_events": n_events, "context_chars": len(prompt)},
    )
