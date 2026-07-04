"""Algo-Exec: execute a small stack-VM program. Deterministic, serial-depth bound.

Difficulty scales program length; ops enter gradually. Arithmetic SATURATES at
[0, 99] (no mod wrap-around): wrap is a separate hard skill that floored tiny models
(attempt 4), while ADD-only tiers were order-independent and needed no depth at all
(EXP-000B). Saturating SUB/SWAP restore genuine serial order-dependence from tier 2
without smuggling in modular arithmetic.
"""
from __future__ import annotations

from .base import Instance, rng_for

FAMILY = "algo_exec"
LO, HI = 0, 99
_OPS_BY_TIER = {
    1: ["PUSH", "ADD"],
    2: ["PUSH", "ADD", "SUB", "SWAP"],
    3: ["PUSH", "ADD", "SUB", "SWAP", "DUP"],
    4: ["PUSH", "ADD", "SUB", "SWAP", "DUP"],
    5: ["PUSH", "ADD", "SUB", "SWAP", "DUP", "MUL", "POP"],
}
_LEN_BY_TIER = {1: 3, 2: 5, 3: 7, 4: 10, 5: 14}


def _sat(v: int) -> int:
    return max(LO, min(HI, v))


def _step(op: str, arg: int | None, stack: list[int]) -> None:
    if op == "PUSH":
        stack.append(_sat(arg))
    elif op == "ADD":
        b, a = stack.pop(), stack.pop()
        stack.append(_sat(a + b))
    elif op == "SUB":
        b, a = stack.pop(), stack.pop()
        stack.append(_sat(a - b))
    elif op == "MUL":
        b, a = stack.pop(), stack.pop()
        stack.append(_sat(a * b))
    elif op == "DUP":
        stack.append(stack[-1])
    elif op == "SWAP":
        stack[-1], stack[-2] = stack[-2], stack[-1]
    elif op == "POP":
        stack.pop()
    else:
        raise ValueError(op)


def _arity(op: str) -> int:
    return {"PUSH": 0, "ADD": 2, "SUB": 2, "MUL": 2, "DUP": 1, "SWAP": 2, "POP": 1}[op]


def generate(seed: int, difficulty: int) -> Instance:
    rng = rng_for(FAMILY, difficulty, seed)
    ops_pool = _OPS_BY_TIER[difficulty]
    n_ops = _LEN_BY_TIER[difficulty]

    program: list[tuple[str, int | None]] = []
    stack: list[int] = []
    trace_lines: list[str] = []

    def emit(op: str) -> None:
        arg = rng.randint(0, 20) if op == "PUSH" else None
        _step(op, arg, stack)
        program.append((op, arg))
        shown = f"{op} {arg}" if arg is not None else op
        trace_lines.append(f"{shown} -> stack {stack}")

    while len(program) < n_ops - 1:
        candidates = [o for o in ops_pool if _arity(o) <= len(stack)]
        if len(stack) < 2:
            candidates = ["PUSH"]
        op = rng.choice(candidates)
        if op == "POP" and len(stack) == 1:
            op = "PUSH"
        emit(op)
    # final op must CONSUME the stack so the answer depends on the computation chain,
    # never on a trailing PUSH literal (difficulty must bind)
    combiners = [o for o in ops_pool if _arity(o) == 2 and o != "SWAP"]
    while len(stack) < 2:
        emit("PUSH")
    emit(rng.choice(combiners) if combiners else "DUP")
    answer = stack[-1]

    prog_text = "\n".join(f"{op} {arg}" if arg is not None else op for op, arg in program)
    prompt = (
        "Execute this stack program. Results clamp to the range 0..99. "
        "Report the final top of stack.\n"
        f"{prog_text}\nANSWER:"
    )
    return Instance(
        family=FAMILY, difficulty=difficulty, seed=seed,
        prompt=prompt, answer=str(answer), trace="\n".join(trace_lines),
        scoring={"type": "numeric"},
        meta={"program_len": n_ops},
    )
