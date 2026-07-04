from . import algo_exec, compress, dsl_learn, fresh_dsl, rewrite, rule_shift, state_guard

FAMILIES = {
    "dsl_learn": dsl_learn.generate,
    "rewrite": rewrite.generate,
    "algo_exec": algo_exec.generate,
    "rule_shift": rule_shift.generate,
    "compress": compress.generate,
    "state_guard": state_guard.generate,
    "fresh_dsl": fresh_dsl.generate,
}

# The five families that count for the 160M gate (docs/BENCHMARK.md)
CORE_FAMILIES = ("dsl_learn", "rewrite", "algo_exec", "rule_shift", "compress")
