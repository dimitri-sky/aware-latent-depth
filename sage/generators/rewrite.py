"""Rewrite: infer hidden term-rewrite rules from before/after pairs, apply to a query.

Rules map symbol bigrams to a single symbol (e.g. "A B -> C"). Rewriting is leftmost,
repeated until fixpoint (bounded). Difficulty scales rule count and string length.
"""
from __future__ import annotations

from .base import Instance, rng_for

FAMILY = "rewrite"
_ALPHABET = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
_N_RULES = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
_STR_LEN = {1: 3, 2: 5, 3: 7, 4: 9, 5: 12}
_N_EXAMPLES = {1: 4, 2: 4, 3: 5, 4: 5, 5: 6}
# serial depth ramp: max rewrite applications per tier (fixpoint if reached earlier)
_MAX_STEPS = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16}


def _apply_fixpoint(rules: dict[tuple[str, str], str], s: list[str],
                    max_steps: int) -> tuple[list[str], list[str]]:
    trace = []
    for _ in range(max_steps):
        for i in range(len(s) - 1):
            key = (s[i], s[i + 1])
            if key in rules:
                s = s[:i] + [rules[key]] + s[i + 2:]
                trace.append(" ".join(s))
                break
        else:
            break
    return s, trace


def _make_rules(rng, n_rules: int) -> dict[tuple[str, str], str]:
    rules: dict[tuple[str, str], str] = {}
    while len(rules) < n_rules:
        a, b, c = rng.choice(_ALPHABET), rng.choice(_ALPHABET), rng.choice(_ALPHABET)
        # avoid trivial self-loops that never terminate: forbid c == a with a == b
        if (a, b) in rules or (a == b == c):
            continue
        rules[(a, b)] = c
    return rules


def _random_string(rng, rules, length: int) -> list[str]:
    # ensure at least one rule applies
    for _ in range(200):
        s = [rng.choice(_ALPHABET) for _ in range(length)]
        if any((s[i], s[i + 1]) in rules for i in range(len(s) - 1)):
            return s
    # force one applicable site
    s = [rng.choice(_ALPHABET) for _ in range(length)]
    (a, b) = next(iter(rules))
    s[0], s[1] = a, b
    return s


def generate(seed: int, difficulty: int) -> Instance:
    rng = rng_for(FAMILY, difficulty, seed)
    rules = _make_rules(rng, _N_RULES[difficulty])

    max_steps = _MAX_STEPS[difficulty]
    examples = []
    for _ in range(_N_EXAMPLES[difficulty]):
        s = _random_string(rng, rules, _STR_LEN[difficulty])
        out, _ = _apply_fixpoint(rules, list(s), max_steps)
        examples.append((" ".join(s), " ".join(out)))

    query = _random_string(rng, rules, _STR_LEN[difficulty])
    answer, trace = _apply_fixpoint(rules, list(query), max_steps)

    ex_text = "\n".join(f"{a} => {b}" for a, b in examples)
    prompt = (
        "Hidden rewrite rules transform the left string into the right string.\n"
        f"{ex_text}\n"
        f"Apply the same hidden rules:\n{' '.join(query)} =>\nANSWER:"
    )
    return Instance(
        family=FAMILY, difficulty=difficulty, seed=seed,
        prompt=prompt, answer=" ".join(answer),
        trace="\n".join(trace) if trace else " ".join(query),
        scoring={"type": "exact"},
        meta={"n_rules": len(rules)},
    )
