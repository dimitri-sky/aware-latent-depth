"""Algo-Exec: execute a small stack-VM program. Deterministic, serial-depth bound.

Difficulty scales program length. Values are kept in [0, 99] (mod 100) so answers are
short and tokenization is not a confound.
"""
from __future__ import annotations

from .base import Instance, rng_for

FAMILY = "algo_exec"
MOD = 100
# ops available per difficulty tier
_OPS_BY_TIER = {
    1: ["PUSH", "ADD"],
    2: ["PUSH", "ADD", "SUB", "DUP"],
    3: ["PUSH", "ADD", "SUB", "MUL", "DUP", "SWAP"],
    4: ["PUSH", "ADD", "SUB", "MUL", "DUP", "SWAP", "POP"],
    5: ["PUSH", "ADD", "SUB", "MUL", "DUP", "SWAP", "POP"],
}
_LEN_BY_TIER = {1: 4, 2: 7, 3: 11, 4: 16, 5: 24}


def _step(op: str, arg: int | None, stack: list[int]) -> None:
    if op == "PUSH":
        stack.append(arg % MOD)
    elif op == "ADD":
        b, a = stack.pop(), stack.pop()
        stack.append((a + b) % MOD)
    elif op == "SUB":
        b, a = stack.pop(), stack.pop()
        stack.append((a - b) % MOD)
    elif op == "MUL":
        b, a = stack.pop(), stack.pop()
        stack.append((a * b) % MOD)
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
    while len(program) < n_ops:
        # POP that would empty the stack right at the end is pointless; keep >=1 at end
        candidates = [o for o in ops_pool if _arity(o) <= len(stack)]
        # bias toward PUSH early so the stack grows
        if len(stack) < 2:
            candidates = ["PUSH"]
        op = rng.choice(candidates)
        if op == "POP" and len(stack) == 1:
            op = "PUSH"
        arg = rng.randint(0, 20) if op == "PUSH" else None
        _step(op, arg, stack)
        program.append((op, arg))
        shown = f"{op} {arg}" if arg is not None else op
        trace_lines.append(f"{shown} -> stack {stack}")
    answer = stack[-1]

    prog_text = "\n".join(f"{op} {arg}" if arg is not None else op for op, arg in program)
    prompt = (
        "Execute this stack program. Values are mod 100. "
        "Report the final top of stack.\n"
        f"{prog_text}\nANSWER:"
    )
    return Instance(
        family=FAMILY, difficulty=difficulty, seed=seed,
        prompt=prompt, answer=str(answer), trace="\n".join(trace_lines),
        scoring={"type": "numeric"},
        meta={"program_len": n_ops},
    )
