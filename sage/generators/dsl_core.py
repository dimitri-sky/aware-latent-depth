"""Shared tiny DSL: s-expressions over ints (mod 100) and small int lists.

Op surface names are randomized per seed, so every instance family built on this DSL is
a *fresh* language the model has never seen. Semantics keys are stable internal ids.
"""
from __future__ import annotations

import random

MOD = 100

INT_OPS = ("add", "sub", "mul", "max", "min", "inc", "dbl")
LIST_OPS = ("seq", "suml", "maxl", "lenl", "headl", "lastl", "revl")
ALL_OPS = INT_OPS + LIST_OPS

_SYLLABLES = ["ba", "ke", "zi", "mo", "ru", "ta", "vu", "li", "so", "ne", "fi", "gu",
              "pa", "de", "wo", "ju", "xa", "che", "qui", "bro"]


def fresh_names(rng: random.Random, ops: tuple[str, ...]) -> dict[str, str]:
    """Map internal op id -> random pronounceable surface name (unique)."""
    names: set[str] = set()
    out: dict[str, str] = {}
    for op in ops:
        while True:
            name = "".join(rng.choice(_SYLLABLES) for _ in range(2))
            if name not in names:
                names.add(name)
                out[op] = name
                break
    return out


class DslError(Exception):
    pass


def tokenize(text: str) -> list[str]:
    return text.replace("(", " ( ").replace(")", " ) ").split()


def parse(tokens: list[str]):
    if not tokens:
        raise DslError("empty")
    tok = tokens.pop(0)
    if tok == "(":
        expr = []
        while tokens and tokens[0] != ")":
            expr.append(parse(tokens))
        if not tokens:
            raise DslError("unbalanced")
        tokens.pop(0)  # ')'
        return expr
    if tok == ")":
        raise DslError("unexpected )")
    try:
        return int(tok)
    except ValueError:
        return tok  # op surface name


def parse_program(text: str):
    tokens = tokenize(text.strip())
    expr = parse(tokens)
    if tokens:
        raise DslError("trailing tokens")
    return expr


def evaluate(expr, name_to_op: dict[str, str], depth: int = 0):
    """Evaluate parsed expression. name_to_op maps surface name -> internal op id."""
    if depth > 16:
        raise DslError("too deep")
    if isinstance(expr, int):
        return expr % MOD
    if isinstance(expr, str):
        raise DslError(f"bare symbol {expr}")
    if not expr or not isinstance(expr[0], str):
        raise DslError("head must be an op")
    op = name_to_op.get(expr[0])
    if op is None:
        raise DslError(f"unknown op {expr[0]}")
    args = [evaluate(a, name_to_op, depth + 1) for a in expr[1:]]

    def ints(n):
        if len(args) != n or not all(isinstance(a, int) for a in args):
            raise DslError(f"{op} wants {n} ints")
        return args

    if op == "add":
        a, b = ints(2); return (a + b) % MOD
    if op == "sub":
        a, b = ints(2); return (a - b) % MOD
    if op == "mul":
        a, b = ints(2); return (a * b) % MOD
    if op == "max":
        a, b = ints(2); return max(a, b)
    if op == "min":
        a, b = ints(2); return min(a, b)
    if op == "inc":
        (a,) = ints(1); return (a + 1) % MOD
    if op == "dbl":
        (a,) = ints(1); return (a * 2) % MOD
    if op == "seq":
        a, b = ints(2)
        if b < a or b - a > 7:
            raise DslError("bad seq range")
        return list(range(a, b + 1))
    # list-consuming ops
    if len(args) != 1 or not isinstance(args[0], list):
        raise DslError(f"{op} wants a list")
    lst = args[0]
    if not lst:
        raise DslError("empty list")
    if op == "suml":
        return sum(lst) % MOD
    if op == "maxl":
        return max(lst)
    if op == "lenl":
        return len(lst)
    if op == "headl":
        return lst[0]
    if op == "lastl":
        return lst[-1]
    if op == "revl":
        return list(reversed(lst))
    raise DslError(f"unhandled op {op}")


def render_value(v) -> str:
    if isinstance(v, list):
        return "[" + " ".join(str(x) for x in v) + "]"
    return str(v)


def render_expr(expr, op_to_name: dict[str, str]) -> str:
    """Render internal-form expression (lists with internal op ids at head)."""
    if isinstance(expr, int):
        return str(expr)
    head, *args = expr
    return "(" + " ".join([op_to_name[head]] + [render_expr(a, op_to_name) for a in args]) + ")"


def random_int_expr(rng: random.Random, ops: list[str], depth: int):
    """Random int-valued expression in internal form."""
    if depth <= 0 or rng.random() < 0.3:
        return rng.randint(0, 20)
    op = rng.choice(ops)
    n_args = 1 if op in ("inc", "dbl") else 2
    return [op] + [random_int_expr(rng, ops, depth - 1) for _ in range(n_args)]


def eval_internal(expr, trace: list[str] | None = None, op_to_name: dict[str, str] | None = None):
    """Evaluate internal-form expression, optionally collecting a per-node trace."""
    identity = {op: op for op in ALL_OPS}
    if isinstance(expr, int):
        return expr % MOD
    head, *args = expr
    vals = [eval_internal(a, trace, op_to_name) for a in args]
    surface = [head] + vals
    result = evaluate([head] + [v for v in vals], identity)
    if trace is not None and op_to_name is not None:
        shown = "(" + " ".join([op_to_name[head]] + [render_value(v) for v in vals]) + ")"
        trace.append(f"{shown} = {render_value(result)}")
    return result
