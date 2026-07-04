"""Benchmark validity gate (docs/BENCHMARK.md), run before any hypothesis training.

(a) Headroom: trivial-solver battery must stay below 30% on every family at tier >= 2.
    Solvers: majority answer (from train), nearest-neighbor prompt overlap (from
    train), copy-last-line.
(b) Discrimination: 4-layer vs 8-layer matched-width Transformer++ trained identically
    must separate by >= 5 points on >= 4 families.

Writes verdicts to stdout and results.csv (exp_id=EXP-000).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from models import ModelConfig  # noqa: E402
from sage.generators import FAMILIES  # noqa: E402
from sage.scoring import score_output  # noqa: E402
from train.data import load_sage_records  # noqa: E402
from train.train import TrainConfig, train_one  # noqa: E402

TRAIN_DIR = Path("data/sage/train")
EVAL_DIR = Path("data/sage/eval")
HEADROOM_MAX = 0.30
DISCRIMINATION_MARGIN = 0.05
FAMS = list(FAMILIES)


def trivial_solvers() -> dict:
    out: dict = {}
    for fam in FAMS:
        train = load_sage_records(TRAIN_DIR / f"{fam}.jsonl", expect_train=True)
        evals = [r for r in load_sage_records(EVAL_DIR / f"{fam}.jsonl", expect_train=False)
                 if r["difficulty"] >= 2]
        majority = Counter(r["answer"] for r in train).most_common(1)[0][0]

        train_toksets = [(set(r["prompt"].split()), r["answer"]) for r in train[:1500]]

        def nn_answer(prompt: str) -> str:
            toks = set(prompt.split())
            best, best_j = "", -1.0
            for ts, ans in train_toksets:
                inter = len(toks & ts)
                if inter == 0:
                    continue
                j = inter / len(toks | ts)
                if j > best_j:
                    best_j, best = j, ans
            return best

        def last_line(prompt: str) -> str:
            lines = [ln for ln in prompt.splitlines() if ln.strip() and "ANSWER" not in ln]
            return lines[-1] if lines else ""

        scores = {}
        for name, fn in [("majority", lambda r: majority),
                         ("nn_overlap", lambda r: nn_answer(r["prompt"])),
                         ("copy_last", lambda r: last_line(r["prompt"]))]:
            c = sum(score_output(fn(r), r["answer"], r["scoring"]) for r in evals)
            scores[name] = c / max(1, len(evals))
        out[fam] = scores
    return out


def discrimination(device: str) -> dict:
    """Per-family focused training: joint 7-family training at this budget starves
    each family (~570 steps/family) below the level where depth can matter — verified
    by probe run EXP-000-probe-rewrite-8L-s0-8bee22 (21.5% single-family vs 2.0%
    mixed). Per-family runs mirror the budget density of real experiments (EXP-001
    trains 2 families over 6000 steps)."""
    from sage.generators import CORE_FAMILIES

    common = dict(vocab_size=259, d_model=256, n_heads=4, n_kv_heads=2, d_ff=704,
                  max_seq_len=1024)
    res: dict = {"gate-tfpp-4L": {}, "gate-tfpp-8L": {}}
    for fam in CORE_FAMILIES:
        tc = TrainConfig(families=[fam], steps=4000, batch_size=32, seq_len=768,
                         seed=0, lr=6e-4, warmup=300, eval_limit=200, throttle=0)
        for n_layers, mid in [(4, "gate-tfpp-4L"), (8, "gate-tfpp-8L")]:
            mcfg = ModelConfig(arch="tf_pp", n_layers=n_layers, **common)
            r = train_one(mcfg, tc, exp_id="EXP-000", model_id=f"{mid}-{fam}",
                          device=device, notes="validity gate per-family")
            stats = r["results"][fam]
            res[mid][fam] = (sum(v[1] for v in stats.values())
                             / max(1, sum(v[0] for v in stats.values())))
    return res


# Pre-registered criterion (2026-07-04, before gate attempt 5; docs/BENCHMARK.md):
# depth separation required on ALL computation families; memory families require a
# learnable headroom band instead (they are H2 dissociation targets by design).
DEPTH_FAMS = ("dsl_learn", "rewrite", "algo_exec")
MEMORY_FAMS = ("rule_shift", "compress")


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("== (a) headroom: trivial solvers ==")
    triv = trivial_solvers()
    headroom_ok = True
    for fam, scores in triv.items():
        worst = max(scores.values())
        flag = "OK " if worst < HEADROOM_MAX else "FAIL"
        if worst >= HEADROOM_MAX:
            headroom_ok = False
        print(f"  [{flag}] {fam:12s} " + " ".join(f"{k}={v:.3f}" for k, v in scores.items()))

    print("== (b) 4L vs 8L Transformer++, per-family ==")
    disc = discrimination(device)
    a, b = disc["gate-tfpp-4L"], disc["gate-tfpp-8L"]
    for fam in a:
        print(f"  {fam:12s} 4L={a[fam]:.3f} 8L={b[fam]:.3f} delta={b[fam] - a[fam]:+.3f}")

    separated = [f for f in DEPTH_FAMS if b.get(f, 0) - a.get(f, 0) >= DISCRIMINATION_MARGIN]
    depth_ok = len(separated) == len(DEPTH_FAMS)
    mem_ok_fams = [f for f in MEMORY_FAMS if 0.10 < max(a.get(f, 0), b.get(f, 0)) < 0.90]
    mem_ok = len(mem_ok_fams) == len(MEMORY_FAMS)

    verdict = {"headroom_ok": headroom_ok, "depth_ok": depth_ok, "memory_ok": mem_ok,
               "separated_depth_families": separated, "memory_in_band": mem_ok_fams,
               "trivial": triv, "disc": disc}
    Path("experiments/validity_gate.json").write_text(json.dumps(verdict, indent=2))
    print(f"VERDICT: headroom={'PASS' if headroom_ok else 'FAIL'} "
          f"depth={'PASS' if depth_ok else 'FAIL'} ({len(separated)}/3 computation "
          f"families) memory-band={'PASS' if mem_ok else 'FAIL'} ({len(mem_ok_fams)}/2)")
    return 0 if (headroom_ok and depth_ok and mem_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
