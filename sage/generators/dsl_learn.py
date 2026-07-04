"""DSL-Learn: learn a fresh symbolic mini-language from K examples, evaluate a query.

Op names are randomized per seed (fresh language every instance). Difficulty scales
number of ops, nesting depth, and example count.
"""
from __future__ import annotations

from .base import Instance, rng_for
from . import dsl_core as dc

FAMILY = "dsl_learn"
_N_OPS = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
_DEPTH = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3}
_N_EXAMPLES = {1: 4, 2: 5, 3: 6, 4: 8, 5: 10}


def generate(seed: int, difficulty: int) -> Instance:
    rng = rng_for(FAMILY, difficulty, seed)
    ops = rng.sample(list(dc.INT_OPS), _N_OPS[difficulty])
    op_to_name = dc.fresh_names(rng, tuple(ops))
    name_to_op = {v: k for k, v in op_to_name.items()}

    # Examples: at least one flat example per op so the language is learnable,
    # then random (possibly nested) fillers.
    examples = []
    for op in ops:
        n_args = 1 if op in ("inc", "dbl") else 2
        expr = [op] + [rng.randint(0, 20) for _ in range(n_args)]
        examples.append(expr)
    while len(examples) < _N_EXAMPLES[difficulty]:
        examples.append(dc.random_int_expr(rng, ops, _DEPTH[difficulty]))
    rng.shuffle(examples)

    ex_lines = []
    for expr in examples:
        if isinstance(expr, int):
            continue
        val = dc.eval_internal(expr)
        ex_lines.append(f"{dc.render_expr(expr, op_to_name)} = {val}")

    # Query must be a nested expression at tier >= 3 (harder than any single example)
    while True:
        query = dc.random_int_expr(rng, ops, _DEPTH[difficulty])
        if isinstance(query, list) and (difficulty < 3 or any(isinstance(a, list) for a in query[1:])):
            break
    trace: list[str] = []
    answer = dc.eval_internal(query, trace, op_to_name)

    prompt = (
        "A new language. Numbers are mod 100. Learn the operators from the examples.\n"
        + "\n".join(ex_lines)
        + f"\nEvaluate:\n{dc.render_expr(query, op_to_name)} =\nANSWER:"
    )
    return Instance(
        family=FAMILY, difficulty=difficulty, seed=seed,
        prompt=prompt, answer=str(answer), trace="\n".join(trace),
        scoring={"type": "numeric"},
        meta={"ops": ops},
    )
