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
LO, HI = 0, 99
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

    acc = rng.randint(5, 60)
    program: list[str] = [f"SET {acc}"]
    trace_lines: list[str] = [f"SET {acc} -> {acc}"]
    for _ in range(n_ops):
        op = rng.choice(ops_pool)
        if op in ("ADD", "SUB"):
            # bias arguments to keep the value off the 0/99 saturation rails,
            # so answers stay high-entropy and priors stay unlearnable
            if op == "ADD":
                hi_room = max(1, (HI - 10) - acc)
                arg = rng.randint(1, min(20, hi_room)) if hi_room > 1 else rng.randint(1, 9)
            else:
                lo_room = max(1, acc - (LO + 5))
                arg = rng.randint(1, min(20, lo_room)) if lo_room > 1 else rng.randint(1, 9)
            acc = _apply(op, arg, acc)
            program.append(f"{op} {arg}")
            trace_lines.append(f"{op} {arg} -> {acc}")
        else:
            # keep the value off the rails: DBL would saturate high, repeated HALF
            # pins to 0 (tier-5 answer prior concentrated at 11% on '0')
            if op == "DBL" and acc > 49:
                op = "HALF"
            if op == "HALF" and acc < 8:
                op = "DBL"
            acc = _apply(op, None, acc)
            program.append(op)
            trace_lines.append(f"{op} -> {acc}")
    answer = acc

    prog_text = "\n".join(program)
    prompt = (
        "Execute this register program. The register starts with SET. "
        "Results clamp to 0..99; HALF rounds down. Report the final register value.\n"
        f"{prog_text}\nANSWER:"
    )
    return Instance(
        family=FAMILY, difficulty=difficulty, seed=seed,
        prompt=prompt, answer=str(answer), trace="\n".join(trace_lines),
        scoring={"type": "numeric"},
        meta={"chain_len": n_ops + 1},
    )
