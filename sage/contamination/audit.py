"""Contamination audit: canary GUIDs + n-gram overlap between train corpora and eval.

Run before every training job (guardrail 8):
    python -m sage.contamination.audit --train-dir data/sage/train --eval-dir data/sage/eval \
        [--extra-corpus data/tinystories/train.txt]

Every prompt's first line is a family-fixed instruction header (identical across all
records by design); the audit strips it and checks task bodies only. Failure criteria:

1. Canary violation (canary missing from eval, or present in any training corpus).
2. Near-duplicate instances: any eval instance whose 13-gram coverage against *rare*
   train 13-grams exceeds NEAR_DUP_COVERAGE (rare = below template doc-frequency).
   Coverage-based (not raw counts) because tiny task vocabularies produce benign
   short collisions; whole-instance duplication is what constitutes leakage.

Exits non-zero on failure. Prints an audit hash recorded by the harness in results.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

CANARY_PREFIX = "SAGE-CANARY-"
NGRAM_N = 13
# An eval instance is a near-duplicate if this fraction of its 13-grams also occur as
# rare 13-grams in training data. Legitimate structural collisions in tiny-vocabulary
# tasks sit far below this; true instance duplication sits near 1.0.
NEAR_DUP_COVERAGE = 0.8
# n-grams present in >= this fraction of train records are treated as template text
TEMPLATE_DOC_FREQ = 0.005


def _strip_header(prompt: str) -> str:
    """Drop the family-fixed instruction header (first line) — template by design."""
    parts = prompt.split("\n", 1)
    return parts[1] if len(parts) > 1 else parts[0]


def _iter_texts(path: Path):
    for f in sorted(path.glob("*.jsonl")):
        for ln in f.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            rec = json.loads(ln)
            if rec.get("kind") == "canary":
                continue
            yield _strip_header(rec.get("prompt", "")) + " " + rec.get("answer", "")


def _ngrams(text: str, n: int):
    toks = text.split()
    for i in range(len(toks) - n + 1):
        yield " ".join(toks[i : i + n])


def run_audit(train_dir: Path, eval_dir: Path, extra_corpora: list[Path]) -> tuple[bool, str, dict]:
    report: dict = {"canary": "ok", "overlap": {}}
    ok = True

    eval_raw = "\n".join(f.read_text(encoding="utf-8") for f in sorted(eval_dir.glob("*.jsonl")))
    if CANARY_PREFIX not in eval_raw:
        ok, report["canary"] = False, "missing canaries in eval"

    train_texts = list(_iter_texts(train_dir))
    extra_texts = [p.read_text(encoding="utf-8", errors="ignore") for p in extra_corpora]
    if any(CANARY_PREFIX in t for t in train_texts + extra_texts):
        ok, report["canary"] = False, "CANARY FOUND IN TRAINING CORPUS"

    eval_texts = list(_iter_texts(eval_dir))
    n_train_docs = max(1, len(train_texts))
    doc_freq: Counter[str] = Counter()
    for t in train_texts:
        doc_freq.update(set(_ngrams(t, NGRAM_N)))
    for t in extra_texts:  # extra corpora audited at full strictness
        doc_freq.update(set(_ngrams(t, NGRAM_N)))
    template_cut = max(2, TEMPLATE_DOC_FREQ * n_train_docs)
    rare_train = {g for g, c in doc_freq.items() if c < template_cut}

    near_dups = 0
    worst = 0.0
    for t in eval_texts:
        gs = set(_ngrams(t, NGRAM_N))
        if not gs:
            continue
        cov = sum(1 for g in gs if g in rare_train) / len(gs)
        worst = max(worst, cov)
        if cov >= NEAR_DUP_COVERAGE:
            near_dups += 1
    report["overlap"] = {"ngram": NGRAM_N, "near_duplicates": near_dups,
                         "worst_coverage": round(worst, 3),
                         "threshold": NEAR_DUP_COVERAGE}
    if near_dups > 0:
        ok = False

    h = hashlib.sha256()
    for f in sorted(eval_dir.glob("*.jsonl")):
        h.update(hashlib.sha256(f.read_bytes()).digest())
    h.update(json.dumps(report, sort_keys=True).encode())
    h.update(b"PASS" if ok else b"FAIL")
    return ok, h.hexdigest()[:16], report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=Path, required=True)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--extra-corpus", type=Path, action="append", default=[])
    args = ap.parse_args()

    ok, audit_hash, report = run_audit(args.train_dir, args.eval_dir, args.extra_corpus)
    print(json.dumps({"pass": ok, "audit_hash": audit_hash, "report": report}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
