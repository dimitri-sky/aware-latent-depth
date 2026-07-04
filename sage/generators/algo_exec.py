"""Algo-Exec: execute a small stack-VM program. Deterministic, serial-depth bound.

Difficulty scales program length. Values are kept in [0, 99] (mod 100) so answers are
short and tokenization is not a confound.
"""
from __future__ import annotations

from .base import Instance, rng_for

FAMILY = "algo_exec"
MOD = 100
# Single-knob ramp (gate attempt 3 postmortem): tiers 1-2 ramp only program length
# with ADD-only arithmetic; harder ops enter one at a time from tier 3. Mod-wrap SUB
# and MUL are disproportionately hard for byte-level models and used to gate tier 2.
_OPS_BY_TIER = {
    1: ["PUSH", "ADD"],
    2: ["PUSH", "ADD"],
    3: ["PUSH", "ADD", "SUB", "DUP"],
    4: ["PUSH", "ADD", "SUB", "DUP", "SWAP"],
    5: ["PUSH", "ADD", "SUB", "MUL", "DUP", "SWAP", "POP"],
}
_LEN_BY_TIER = {1: 3, 2: 5, 3: 7, 4: 10, 5: 14}


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

    def would_wrap(op: str) -> bool:
        if op == "SUB" and len(stack) >= 2:
            return stack[-2] < stack[-1]
        if op == "MUL" and len(stack) >= 2:
            return stack[-2] * stack[-1] >= MOD
        return False

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
        # mod-wrap arithmetic is its own skill; it enters only at tier 5
        # (gate attempt 4: wrap-SUB at tier 3 floored both depths to 0.10)
        if difficulty < 5 and would_wrap(op):
            op = "PUSH" if len(stack) < 2 else "DUP"
        emit(op)
    # final op must CONSUME the stack so the answer depends on the computation chain,
    # never on a trailing PUSH literal (difficulty must bind)
    combiners = [o for o in ops_pool if _arity(o) == 2 and o != "SWAP"
                 and not (difficulty < 5 and would_wrap(o))]
    while len(stack) < 2:
        emit("PUSH")
    emit(rng.choice(combiners) if combiners else "DUP")
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
