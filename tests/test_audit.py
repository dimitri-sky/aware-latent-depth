"""The audit must pass on clean disjoint splits and fail on planted leakage."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sage.contamination.audit import CANARY_PREFIX, run_audit
from sage.generators import FAMILIES


def _write(dirpath: Path, records: list[dict], canary: bool):
    dirpath.mkdir(parents=True, exist_ok=True)
    with (dirpath / "fam.jsonl").open("w", encoding="utf-8") as fh:
        if canary:
            fh.write(json.dumps({"kind": "canary", "text": f"{CANARY_PREFIX}test-guid"}) + "\n")
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _instances(split_lo: int, n: int) -> list[dict]:
    return [FAMILIES["compress"](split_lo + i, 3).to_dict() for i in range(n)]


def test_clean_split_passes(tmp_path):
    _write(tmp_path / "train", _instances(0, 40), canary=False)
    _write(tmp_path / "eval", _instances(2_000_000, 10), canary=True)
    ok, _, report = run_audit(tmp_path / "train", tmp_path / "eval", [])
    assert ok, report


def test_planted_duplicate_fails(tmp_path):
    evals = _instances(2_000_000, 10)
    train = _instances(0, 40) + [evals[0]]  # leak one eval instance into train
    _write(tmp_path / "train", train, canary=False)
    _write(tmp_path / "eval", evals, canary=True)
    ok, _, report = run_audit(tmp_path / "train", tmp_path / "eval", [])
    assert not ok
    assert report["overlap"]["near_duplicates"] >= 1


def test_canary_in_train_fails(tmp_path):
    train = _instances(0, 20)
    train.append({"prompt": f"stray {CANARY_PREFIX}leaked-guid", "answer": "x"})
    _write(tmp_path / "train", train, canary=False)
    _write(tmp_path / "eval", _instances(2_000_000, 5), canary=True)
    ok, _, report = run_audit(tmp_path / "train", tmp_path / "eval", [])
    assert not ok


def test_missing_canary_fails(tmp_path):
    _write(tmp_path / "train", _instances(0, 20), canary=False)
    _write(tmp_path / "eval", _instances(2_000_000, 5), canary=False)
    ok, _, _ = run_audit(tmp_path / "train", tmp_path / "eval", [])
    assert not ok
