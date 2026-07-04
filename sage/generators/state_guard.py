"""StateGuard: retention, forgetting, and state-corruption probes.

Facts are introduced early, followed by a long interference stretch (facts about other
keys), and optionally explicit retractions ("forget X"). The probe targets either an
early fact (retention), a retracted fact (the answer must be 'unknown'), or an
overwritten fact (must be the latest value). meta records probe kind and interference
distance for the diagnostic report.
"""
from __future__ import annotations

from .base import Instance, rng_for

FAMILY = "state_guard"
_N_FACTS = {1: 3, 2: 4, 3: 6, 4: 8, 5: 10}
_INTERFERENCE = {1: 3, 2: 6, 3: 12, 4: 20, 5: 32}

_KEYS = ["door", "vent", "gate", "dial", "lamp", "fuse", "belt", "pump", "tank", "coil",
         "fan", "lock"]
# Large value vocabulary keeps trivial-prior solvers under the headroom threshold.
_VALUES = ["red", "blue", "green", "gold", "gray", "pink", "teal", "ivory", "amber",
           "plum", "rust", "jade", "coral", "navy", "olive", "white", "black", "cyan",
           "beige", "maroon"]


def generate(seed: int, difficulty: int) -> Instance:
    rng = rng_for(FAMILY, difficulty, seed)
    keys = rng.sample(_KEYS, min(_N_FACTS[difficulty] + 2, len(_KEYS)))
    early_keys = keys[: _N_FACTS[difficulty]]
    noise_keys = keys[_N_FACTS[difficulty]:]

    state: dict[str, str | None] = {}
    lines: list[str] = []
    for k in early_keys:
        v = rng.choice(_VALUES)
        state[k] = v
        lines.append(f"the {k} is {v}.")

    # retract kept rare: its fixed answer 'unknown' is what trivial priors exploit
    probe_kind = rng.choice(["retention", "retention", "retention",
                             "overwrite", "overwrite", "retract"])
    target = rng.choice(early_keys)

    if probe_kind == "retract":
        lines.append(f"forget the {target}.")
        state[target] = None
    elif probe_kind == "overwrite":
        v2 = rng.choice([v for v in _VALUES if v != state[target]])
        lines.append(f"update: the {target} is now {v2}.")
        state[target] = v2

    # interference stretch: churn on non-target keys
    for _ in range(_INTERFERENCE[difficulty]):
        k = rng.choice([k for k in early_keys + noise_keys if k != target])
        v = rng.choice(_VALUES)
        state[k] = v
        lines.append(f"the {k} is {v}.")

    answer = state[target] if state[target] is not None else "unknown"
    trace = f"last statement about {target}: {answer}"

    prompt = (
        "Remember the facts. 'forget X' means X becomes unknown. "
        "Always report the latest state.\n"
        + " ".join(lines)
        + f"\nWhat is the {target}?\nANSWER:"
    )
    return Instance(
        family=FAMILY, difficulty=difficulty, seed=seed,
        prompt=prompt, answer=answer, trace=trace,
        scoring={"type": "exact"},
        meta={"probe": probe_kind, "interference": _INTERFERENCE[difficulty]},
    )
