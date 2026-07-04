"""Rule-Shift: a symbol->symbol mapping rule changes mid-session; adapt from examples.

The prompt shows a session of input->output pairs. The mapping table silently changes
at a shift point; some post-shift examples are shown, then a query. meta records
`post_shift_examples` so the harness can compute adaptation speed (accuracy as a
function of how many post-shift examples were available).
"""
from __future__ import annotations

from .base import Instance, rng_for

FAMILY = "rule_shift"
_SYMBOLS = list("abcdefghij")
_TABLE_SIZE = {1: 3, 2: 4, 3: 5, 4: 6, 5: 8}
_PRE_EXAMPLES = {1: 4, 2: 5, 3: 6, 4: 8, 5: 10}
_WORD_LEN = {1: 2, 2: 2, 3: 3, 4: 3, 5: 4}


def _make_table(rng, symbols: list[str]) -> dict[str, str]:
    targets = symbols.copy()
    while True:
        rng.shuffle(targets)
        if all(a != b for a, b in zip(symbols, targets)):
            return dict(zip(symbols, targets))


def _shifted_table(rng, table: dict[str, str]) -> dict[str, str]:
    """Change the images of ~half the keys (a real rule shift, not a fresh table)."""
    symbols = list(table)
    n_change = max(1, len(symbols) // 2)
    changed = rng.sample(symbols, n_change)
    new = dict(table)
    for k in changed:
        choices = [s for s in symbols if s != new[k] and s != k]
        new[k] = rng.choice(choices)
    return new


def _apply(table: dict[str, str], word: str) -> str:
    return "".join(table[c] for c in word)


def generate(seed: int, difficulty: int) -> Instance:
    rng = rng_for(FAMILY, difficulty, seed)
    symbols = _SYMBOLS[: _TABLE_SIZE[difficulty]]
    table1 = _make_table(rng, symbols)
    table2 = _shifted_table(rng, table1)
    wlen = _WORD_LEN[difficulty]

    def word():
        return "".join(rng.choice(symbols) for _ in range(wlen))

    lines = []
    for _ in range(_PRE_EXAMPLES[difficulty]):
        w = word()
        lines.append(f"{w} -> {_apply(table1, w)}")
    # 0..4 post-shift examples; adaptation speed = accuracy vs this count
    n_post = rng.randint(0, 4)
    post_words = []
    for _ in range(n_post):
        w = word()
        post_words.append(w)
        lines.append(f"{w} -> {_apply(table2, w)}")

    # Query must involve at least one changed symbol, else the shift is unobservable
    changed_syms = [k for k in table1 if table1[k] != table2[k]]
    while True:
        q = word()
        if any(c in changed_syms for c in q):
            break
    answer = _apply(table2, q)
    trace = "\n".join(f"{c} -> {table2[c]}" for c in q)

    prompt = (
        "Each line maps a word through a hidden letter-substitution rule. "
        "The rule may change during the session; always use the latest rule.\n"
        + "\n".join(lines)
        + f"\n{q} ->\nANSWER:"
    )
    return Instance(
        family=FAMILY, difficulty=difficulty, seed=seed,
        prompt=prompt, answer=answer, trace=trace,
        scoring={"type": "exact"},
        meta={"post_shift_examples": n_post,
              "query_changed_syms": sum(c in changed_syms for c in q)},
    )
