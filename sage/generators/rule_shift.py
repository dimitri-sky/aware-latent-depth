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
# Harder ramp (gate attempt 4: a 3M-param 4L transformer saturated the old ramp at
# 99.5%): longer words and larger tables create headroom; this family probes MEMORY
# (latest-rule retrieval across a long session), not depth.
_TABLE_SIZE = {1: 4, 2: 5, 3: 6, 4: 8, 5: 10}
_PRE_EXAMPLES = {1: 6, 2: 7, 3: 8, 4: 10, 5: 12}
_WORD_LEN = {1: 2, 2: 3, 3: 3, 4: 4, 5: 5}


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
    seen_pre: set[str] = set()
    for _ in range(_PRE_EXAMPLES[difficulty]):
        w = word()
        seen_pre.update(w)
        lines.append(f"{w} -> {_apply(table1, w)}")
    # every symbol must be demonstrated at least once pre-shift
    for s in symbols:
        if s not in seen_pre:
            w = s + "".join(rng.choice(symbols) for _ in range(wlen - 1))
            seen_pre.update(w)
            lines.append(f"{w} -> {_apply(table1, w)}")
    # 1..4 post-shift examples; adaptation speed = accuracy vs this count
    n_post = rng.randint(1, 4)
    covered: set[str] = set()
    for _ in range(n_post):
        w = word()
        covered.update(w)
        lines.append(f"{w} -> {_apply(table2, w)}")

    # INFERABILITY GUARANTEE (gate attempt 3 postmortem): the answer must be
    # derivable from the prompt. Every changed symbol in the query must appear in a
    # post-shift example; unchanged symbols are known from pre-shift examples.
    changed_syms = [k for k in table1 if table1[k] != table2[k]]
    observable = [c for c in changed_syms if c in covered]
    if not observable:
        # force coverage: add one post-shift example containing a changed symbol
        c = rng.choice(changed_syms)
        w = c + "".join(rng.choice(symbols) for _ in range(wlen - 1))
        covered.update(w)
        lines.append(f"{w} -> {_apply(table2, w)}")
        n_post += 1
        observable = [c for c in changed_syms if c in covered]
    # query pool: unchanged symbols (known from pre-shift) + post-covered changed ones
    pool = [s for s in symbols if s not in changed_syms or s in covered]
    for _ in range(400):
        q = word()
        q_changed = [c for c in q if c in changed_syms]
        if q_changed and all(c in pool for c in q):
            break
    else:
        q = rng.choice(observable) + "".join(rng.choice(pool) for _ in range(wlen - 1))
    answer = _apply(table2, q)

    # EXP-004 trace budgets, derived deterministically from already-drawn state
    # (ZERO extra RNG calls — prompt/answer identity enforced by
    # tests/test_trace_emitters.py). No "short" tier: it would be
    # FLOP-indistinguishable from med (logged design change #4).
    #   long (the `trace` field): reconstruct the full current rule table, then
    #     resolve the query char by char
    #   med:  per-char query mapping only (the pre-EXP-004 trace)
    #   filler: contentless '.' tokens, length-matched to the long trace
    med = "\n".join(f"{c} -> {table2[c]}" for c in q)
    trace = ("rule:\n" + "\n".join(f"{k} -> {table2[k]}" for k in symbols)
             + "\nquery:\n" + med)
    filler = ". " * (len(trace) // 2)
    trace_variants = {"med": med, "filler": filler.strip()}

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
              "query_changed_syms": sum(c in changed_syms for c in q),
              "trace_variants": trace_variants},
    )
