"""EXP-004 trace-emitter contract tests.

1. INVARIANCE (the reuse contract): adding trace variants must not consume extra
   RNG draws — prompt/answer for every (family, difficulty, seed) must remain
   byte-identical to the pre-EXP-004 generators. Golden hashes below were captured
   from the generators at commit time BEFORE the emitter change (2026-07-06).
   If this test fails, reuse of the EXP-002 runs as EXP-004 arms is INVALID.
2. Variant well-formedness: budgets ordered, filler contentless and length-matched.
"""
from __future__ import annotations

import hashlib

from sage.generators import algo_exec, rule_shift

# sha256(prompt + "|" + answer)[:16] from the pre-EXP-004 generators.
GOLDEN = {
    ("algo_exec", 1, 0): "ee50568ed4214a36",
    ("algo_exec", 1, 7): "6362a60e64ba3ae1",
    ("algo_exec", 1, 123): "e7bd40b1f81b84ca",
    ("algo_exec", 1, 2000000): "c3ccbe7b44d25e47",
    ("algo_exec", 1, 2000055): "1712085672ae48c8",
    ("algo_exec", 2, 0): "cc23510ed6aa792d",
    ("algo_exec", 2, 7): "feb856b972111014",
    ("algo_exec", 2, 123): "b648106f5ec282b8",
    ("algo_exec", 2, 2000000): "216decabc6a55807",
    ("algo_exec", 2, 2000055): "215657fc0e4a44c1",
    ("algo_exec", 3, 0): "3addc4a64537ac65",
    ("algo_exec", 3, 7): "a40d7d2948dda362",
    ("algo_exec", 3, 123): "f102b9d3401e3c60",
    ("algo_exec", 3, 2000000): "05a79cb198af4456",
    ("algo_exec", 3, 2000055): "6893059546ed389c",
    ("algo_exec", 4, 0): "e60191b379bfa121",
    ("algo_exec", 4, 7): "27ea7e03dd307afc",
    ("algo_exec", 4, 123): "96cc031017448da7",
    ("algo_exec", 4, 2000000): "7de3e2182ac19c97",
    ("algo_exec", 4, 2000055): "3561eafe886176ff",
    ("algo_exec", 5, 0): "ff0a4f92014ac523",
    ("algo_exec", 5, 7): "3fa1a8b0b83a10ea",
    ("algo_exec", 5, 123): "be123f2c662d8b01",
    ("algo_exec", 5, 2000000): "d1b0967889aa28c2",
    ("algo_exec", 5, 2000055): "87617c3a482910b0",
    ("rule_shift", 1, 0): "08f5d6a92bb54f03",
    ("rule_shift", 1, 7): "0f78e8586a80f390",
    ("rule_shift", 1, 123): "2a41462c653c48cc",
    ("rule_shift", 1, 2000000): "acefb3e3cfc9c198",
    ("rule_shift", 1, 2000055): "af3c0d35d28fc648",
    ("rule_shift", 2, 0): "90cce85275f42b25",
    ("rule_shift", 2, 7): "7ad7b2c79a29274e",
    ("rule_shift", 2, 123): "91b74056898206c1",
    ("rule_shift", 2, 2000000): "fa68f413f2418178",
    ("rule_shift", 2, 2000055): "27a1e273cddee3d4",
    ("rule_shift", 3, 0): "237b3f8cb1c00cfa",
    ("rule_shift", 3, 7): "682d7007ba3c5d2a",
    ("rule_shift", 3, 123): "4a2d9442d51933a1",
    ("rule_shift", 3, 2000000): "2ccb065f84903ca3",
    ("rule_shift", 3, 2000055): "5f6b2d22158a59db",
    ("rule_shift", 4, 0): "de60b263b97eef8d",
    ("rule_shift", 4, 7): "d9e00ac95de86c60",
    ("rule_shift", 4, 123): "f769e6ff8bebb4c9",
    ("rule_shift", 4, 2000000): "497321adcd6351d5",
    ("rule_shift", 4, 2000055): "d705aa60fad9ae14",
    ("rule_shift", 5, 0): "5e5e00da6f7de373",
    ("rule_shift", 5, 7): "d69612249693dbaa",
    ("rule_shift", 5, 123): "e62ef883a4d543e3",
    ("rule_shift", 5, 2000000): "23443ff37ab59945",
    ("rule_shift", 5, 2000055): "0746209c60a30e97",
}

MODS = {"algo_exec": algo_exec, "rule_shift": rule_shift}


def _h(inst) -> str:
    return hashlib.sha256((inst.prompt + "|" + inst.answer).encode()).hexdigest()[:16]


def test_prompt_answer_invariance():
    for (fam, d, seed), want in GOLDEN.items():
        inst = MODS[fam].generate(seed, d)
        assert _h(inst) == want, (
            f"{fam} d{d} seed{seed}: prompt/answer changed — EXP-002 arm reuse invalid")


def test_algo_exec_variants():
    for seed in (0, 7, 123):
        for d in (1, 3, 5):
            inst = algo_exec.generate(seed, d)
            tv = inst.meta["trace_variants"]
            assert set(tv) == {"med", "short", "filler"}
            # budget ordering: short <= med <= long
            assert len(tv["short"]) <= len(tv["med"]) <= len(inst.trace)
            # med = one value per program line (SET + n_ops)
            assert len(tv["med"].split()) == inst.meta["chain_len"]
            # med/short end with the answer (the model must still emit ANSWER: x,
            # but the value chain resolves to it)
            assert tv["med"].split()[-1] == inst.answer
            assert tv["short"].split()[-1] == inst.answer
            # filler: contentless, byte-length-matched to long within 2 chars
            assert set(tv["filler"]) <= {".", " "}
            assert abs(len(tv["filler"]) - len(inst.trace)) <= 2


def test_rule_shift_variants():
    for seed in (0, 7, 123):
        for d in (1, 3, 5):
            inst = rule_shift.generate(seed, d)
            tv = inst.meta["trace_variants"]
            assert set(tv) == {"med", "filler"}
            assert len(tv["med"]) < len(inst.trace)
            # long trace contains the full current table and the per-char resolution
            assert inst.trace.startswith("rule:")
            assert tv["med"] in inst.trace
            # every mapped char in med agrees with the answer characters
            q_chars = [ln.split(" -> ")[1] for ln in tv["med"].splitlines()]
            assert "".join(q_chars) == inst.answer
            assert set(tv["filler"]) <= {".", " "}
            assert abs(len(tv["filler"]) - len(inst.trace)) <= 2
