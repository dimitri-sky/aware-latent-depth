"""FreshDSL-CodeBench: a private-seed tiny DSL; execute, write, and debug programs,
plus adapt to a revised DSL and transfer to a sibling DSL.

Subtasks (rotated by seed):
  execute  - evaluate a program (numeric answer)
  write    - produce a program that evaluates to a target (scored by execution;
             the bare target literal is rejected)
  debug    - one token of a program was corrupted; output the fixed program
             (scored by execution against the expected value)
  revise   - one operator's semantics changed mid-prompt (shown by new examples);
             evaluate a query under the revised DSL
  transfer - same DSL, new surface names (sibling); evaluate after a few examples
"""
from __future__ import annotations

from .base import Instance, rng_for
from . import dsl_core as dc

FAMILY = "fresh_dsl"
_N_OPS = {1: 3, 2: 4, 3: 5, 4: 6, 5: 7}
_DEPTH = {1: 1, 2: 2, 3: 2, 4: 3, 5: 3}
_SUBTASKS = ["execute", "write", "debug", "revise", "transfer"]


def _example_lines(rng, ops, op_to_name, n):
    lines = []
    for op in ops:
        n_args = 1 if op in ("inc", "dbl") else 2
        expr = [op] + [rng.randint(0, 20) for _ in range(n_args)]
        lines.append(f"{dc.render_expr(expr, op_to_name)} = {dc.eval_internal(expr)}")
    while len(lines) < n:
        expr = dc.random_int_expr(rng, list(ops), 2)
        if isinstance(expr, list):
            lines.append(f"{dc.render_expr(expr, op_to_name)} = {dc.eval_internal(expr)}")
    rng.shuffle(lines)
    return lines


def generate(seed: int, difficulty: int) -> Instance:
    rng = rng_for(FAMILY, difficulty, seed)
    subtask = _SUBTASKS[seed % len(_SUBTASKS)]
    ops = tuple(rng.sample(list(dc.INT_OPS), _N_OPS[difficulty]))
    op_to_name = dc.fresh_names(rng, ops)
    name_to_op = {v: k for k, v in op_to_name.items()}
    header = "A private mini-language. Numbers are mod 100. Examples:\n"
    examples = _example_lines(rng, ops, op_to_name, _N_OPS[difficulty] + 3)

    if subtask == "execute":
        query = dc.random_int_expr(rng, list(ops), _DEPTH[difficulty])
        while not isinstance(query, list):
            query = dc.random_int_expr(rng, list(ops), _DEPTH[difficulty])
        trace: list[str] = []
        answer = dc.eval_internal(query, trace, op_to_name)
        prompt = (header + "\n".join(examples)
                  + f"\nEvaluate:\n{dc.render_expr(query, op_to_name)} =\nANSWER:")
        scoring = {"type": "numeric"}
        ans_str, trace_str = str(answer), "\n".join(trace)

    elif subtask == "write":
        expr = dc.random_int_expr(rng, list(ops), _DEPTH[difficulty])
        while not isinstance(expr, list):
            expr = dc.random_int_expr(rng, list(ops), _DEPTH[difficulty])
        target = dc.eval_internal(expr)
        prompt = (header + "\n".join(examples)
                  + f"\nWrite one expression in this language that evaluates to {target}. "
                  + "It must use at least one operator.\nANSWER:")
        scoring = {"type": "dsl_exec", "name_to_op": name_to_op, "target": target,
                   "require_op": True}
        ans_str = dc.render_expr(expr, op_to_name)  # a reference solution
        trace_str = f"target {target}; one solution: {ans_str}"

    elif subtask == "debug":
        expr = dc.random_int_expr(rng, list(ops), max(2, _DEPTH[difficulty]))
        while not isinstance(expr, list):
            expr = dc.random_int_expr(rng, list(ops), max(2, _DEPTH[difficulty]))
        expected = dc.eval_internal(expr)
        good = dc.render_expr(expr, op_to_name)
        toks = good.replace("(", " ( ").replace(")", " ) ").split()
        idx = rng.choice([i for i, t in enumerate(toks) if t.isdigit()])
        toks[idx] = str((int(toks[idx]) + rng.randint(1, 9)) % 21)
        broken = " ".join(toks).replace("( ", "(").replace(" )", ")")
        prompt = (header + "\n".join(examples)
                  + f"\nThis program should evaluate to {expected} but is broken:\n{broken}\n"
                  + "Output a corrected program.\nANSWER:")
        scoring = {"type": "dsl_exec", "name_to_op": name_to_op, "target": expected,
                   "require_op": True}
        ans_str, trace_str = good, f"expected {expected}; original: {good}"

    elif subtask == "revise":
        changed = rng.choice([o for o in ops if o in ("add", "mul", "max", "min", "sub")])
        pool = [o for o in dc.INT_OPS if o not in ops and o not in ("inc", "dbl")]
        new_sem = rng.choice(pool) if pool else ("max" if changed != "max" else "min")
        revised = {op: (new_sem if op == changed else op) for op in ops}
        upd_lines = []
        for _ in range(3):
            a, b = rng.randint(0, 20), rng.randint(0, 20)
            val = dc.eval_internal([new_sem, a, b])
            upd_lines.append(f"({op_to_name[changed]} {a} {b}) = {val}")
        query = dc.random_int_expr(rng, list(ops), _DEPTH[difficulty])
        while not (isinstance(query, list) and _uses(query, changed)):
            query = dc.random_int_expr(rng, list(ops), _DEPTH[difficulty])
        rev_query = _substitute(query, changed, new_sem)
        answer = dc.eval_internal(rev_query)
        prompt = (header + "\n".join(examples)
                  + "\nUPDATE - the language changed; new examples:\n" + "\n".join(upd_lines)
                  + f"\nEvaluate under the updated language:\n{dc.render_expr(query, op_to_name)} =\nANSWER:")
        scoring = {"type": "numeric"}
        ans_str, trace_str = str(answer), f"{op_to_name[changed]} now behaves as {new_sem}"

    else:  # transfer
        sibling_names = dc.fresh_names(rng, ops)
        sib_examples = _example_lines(rng, ops, sibling_names, len(ops) + 1)
        query = dc.random_int_expr(rng, list(ops), _DEPTH[difficulty])
        while not isinstance(query, list):
            query = dc.random_int_expr(rng, list(ops), _DEPTH[difficulty])
        trace2: list[str] = []
        answer = dc.eval_internal(query, trace2, sibling_names)
        prompt = (header + "\n".join(examples)
                  + "\nA sibling language with the same structure but new names:\n"
                  + "\n".join(sib_examples)
                  + f"\nEvaluate in the sibling language:\n{dc.render_expr(query, sibling_names)} =\nANSWER:")
        scoring = {"type": "numeric"}
        ans_str, trace_str = str(answer), "\n".join(trace2)

    return Instance(
        family=FAMILY, difficulty=difficulty, seed=seed,
        prompt=prompt, answer=ans_str, trace=trace_str, scoring=scoring,
        meta={"subtask": subtask},
    )


def _uses(expr, op) -> bool:
    return isinstance(expr, list) and (expr[0] == op or any(_uses(a, op) for a in expr[1:]))


def _substitute(expr, old, new):
    if not isinstance(expr, list):
        return expr
    return [new if expr[0] == old else expr[0]] + [_substitute(a, old, new) for a in expr[1:]]
