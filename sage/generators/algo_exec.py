"""Algo-Exec v2: execute a single-accumulator register program. Pure serial depth.

Postmortem of v1 (stack VM, gate attempts 1-6): stack programs contain dead code —
the answer depends only on values that survive to the stack top, so the live data
path is far shorter than the program, and attention retrieves it shallowly (2L
matched 16L within noise across every formulation). An accumulator machine has no
dead code by construction: every instruction transforms the single register, so the
serial dependency chain length EQUALS the program length. Difficulty ramps exactly
one knob: chain length. Arithmetic saturates at [0, 99] (mod-wrap is a separate hard
skill, per attempt-4 postmortem); argument choice biases away from the saturation
rails so answer entropy stays high.
"""
from __future__ import annotations

from .base import Instance, rng_for

FAMILY = "algo_exec"
# v3 (final revision before park, per timebox): SINGLE-DIGIT register, values clamp
# to 0..9. v2's two-digit arithmetic made per-step accuracy (~55%) the bottleneck —
# the task measured arithmetic skill, not chain-following. Single-digit steps are
# trivially learnable, so accuracy isolates serial composition; frequent clamping at
# the 0/9 rails keeps the chain order-dependent (not a commutative sum).
LO, HI = 0, 9
_OPS_BY_TIER = {
    1: ["ADD", "SUB"],
    2: ["ADD", "SUB", "DBL"],
    3: ["ADD", "SUB", "DBL", "HALF"],
    4: ["ADD", "SUB", "DBL", "HALF"],
    5: ["ADD", "SUB", "DBL", "HALF"],
}
_LEN_BY_TIER = {1: 3, 2: 5, 3: 8, 4: 12, 5: 18}


def _sat(v: int) -> int:
    return max(LO, min(HI, v))


def _apply(op: str, arg: int | None, acc: int) -> int:
    if op == "SET":
        return _sat(arg)
    if op == "ADD":
        return _sat(acc + arg)
    if op == "SUB":
        return _sat(acc - arg)
    if op == "DBL":
        return _sat(acc * 2)
    if op == "HALF":
        return acc // 2
    raise ValueError(op)


def generate(seed: int, difficulty: int) -> Instance:
    rng = rng_for(FAMILY, difficulty, seed)
    ops_pool = _OPS_BY_TIER[difficulty]
    n_ops = _LEN_BY_TIER[difficulty]

    acc = rng.randint(2, 7)
    program: list[str] = [f"SET {acc}"]
    trace_lines: list[str] = [f"SET {acc} -> {acc}"]
    for _ in range(n_ops):
        op = rng.choice(ops_pool)
        if op in ("ADD", "SUB"):
            arg = rng.randint(1, 4)  # small operands: clamps stay occasional, not dominant
            # steer away from pinning: don't saturate twice in a row
            if op == "ADD" and acc >= HI:
                op = "SUB"
            elif op == "SUB" and acc <= LO:
                op = "ADD"
            acc = _apply(op, arg, acc)
            program.append(f"{op} {arg}")
            trace_lines.append(f"{op} {arg} -> {acc}")
        else:
            if op == "DBL" and acc >= HI:
                op = "HALF"
            if op == "HALF" and acc <= 1:
                op = "DBL"
            acc = _apply(op, None, acc)
            program.append(op)
            trace_lines.append(f"{op} -> {acc}")
    answer = acc

    prog_text = "\n".join(program)
    prompt = (
        "Execute this register program. The register starts with SET. "
        "Results clamp to the range 0..9; HALF rounds down. "
        "Report the final register value.\n"
        f"{prog_text}\nANSWER:"
    )
    return Instance(
        family=FAMILY, difficulty=difficulty, seed=seed,
        prompt=prompt, answer=str(answer), trace="\n".join(trace_lines),
        scoring={"type": "numeric"},
        meta={"chain_len": n_ops + 1},
    )
