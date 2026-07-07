"""Local CPU smoke for the EXP-004/EXP-009 pipeline (mechanical validation only).

Runs against a throwaway data dir + results shard (real data/sage and
results.csv untouched). Checks:
  1. traced training (long + filler levels) trains and CoT-evals end to end,
  2. CoT eval logs fpa/fpa35/cotfail/maxnew in notes,
  3. EXP-009 diagnostics sidecar has light + heavy records (delta-state stats),
  4. per-arm config hashes distinct (trace_level in extra).

    python scripts/smoke_exp004.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SMOKE = Path("data/sage_smoke")
SHARD = Path("experiments/results_SMOKE_exp004.csv")


def main() -> None:
    # fresh smoke data (tiny; seeded, so deterministic)
    for split, n in (("train", 150), ("eval", 50)):
        subprocess.run([sys.executable, "scripts/make_data.py", "--split", split,
                        "--per-family", str(n), "--families", "algo_exec,rule_shift",
                        "--out-dir", str(SMOKE)], check=True)

    header = Path("experiments/results.csv").open(encoding="utf-8").readline()
    SHARD.write_text(header, encoding="utf-8")
    os.environ["AWARE_RESULTS_CSV"] = str(SHARD)

    from eval.harness import extract_final_answer  # noqa: E402
    from models import ModelConfig  # noqa: E402
    from sage.scoring import score_output  # noqa: E402
    from train.train import TrainConfig, train_one  # noqa: E402

    # mechanical check of the CoT scoring contract
    ans, ok = extract_final_answer("THINK:\n5 8 4\nANSWER: 4\n")
    assert ok and score_output(ans, "4", {"type": "numeric"})
    ans, ok = extract_final_answer("5 8 4 never answers")
    assert not ok
    print("[smoke] extract_final_answer contract ok")

    tiny_tf = dict(arch="tf_pp", d_model=64, n_layers=2, n_heads=2, n_kv_heads=1,
                   d_ff=128, max_seq_len=1024)
    tiny_delta = dict(arch="delta", d_model=64, n_layers=2, n_heads=2, n_kv_heads=1,
                      d_ff=128, delta_every=2, window=32, d_k=32, d_v=32,
                      max_seq_len=1024)

    # arm-hash distinctness (trace_level rides in extra)
    hashes = {lvl: ModelConfig(**tiny_tf, extra={"trace_level": lvl}).config_hash()
              for lvl in ("short", "med", "long", "filler")}
    hashes["direct"] = ModelConfig(**tiny_tf).config_hash()
    assert len(set(hashes.values())) == len(hashes), f"hash collision: {hashes}"
    print(f"[smoke] config hashes distinct: {hashes}")

    common = dict(batch_size=8, seq_len=768, warmup=10, eval_limit=30,
                  data_dir=str(SMOKE / "train"), eval_dir=str(SMOKE / "eval"))

    # 1) CoT-long traced training on algo_exec
    train_one(ModelConfig(**tiny_tf, extra={"trace_level": "long"}),
              TrainConfig(families=["algo_exec"], steps=12, lr=3e-4, seed=0,
                          traced=True, trace_level="long", **common),
              exp_id="EXP-SMOKE4", model_id="tf-cot-long", device="cpu",
              notes="exp004 smoke")

    # 2) filler traced training on rule_shift
    train_one(ModelConfig(**tiny_tf, extra={"trace_level": "filler"}),
              TrainConfig(families=["rule_shift"], steps=12, lr=3e-4, seed=0,
                          traced=True, trace_level="filler", **common),
              exp_id="EXP-SMOKE4", model_id="tf-filler", device="cpu",
              notes="exp004 smoke")

    # 3) EXP-009 diagnostics on a tiny delta model (direct training; >= 3 light
    # records needed for the curvature second-difference)
    train_one(ModelConfig(**tiny_delta),
              TrainConfig(families=["rule_shift"], steps=80, lr=3e-4, seed=0,
                          diag=True, **common),
              exp_id="EXP-SMOKE9", model_id="delta-diag", device="cpu",
              notes="exp009 smoke")

    # --- assertions -----------------------------------------------------------
    rows = SHARD.read_text(encoding="utf-8").splitlines()[1:]
    cot_rows = [r for r in rows if "tf-cot-long" in r]
    assert cot_rows and all("fpa=" in r and "cotfail=" in r and "maxnew=" in r
                            for r in cot_rows), "CoT notes missing extras"
    assert any("fpa35=" in r for r in cot_rows), "fpa35 missing"
    fill_rows = [r for r in rows if "tf-filler" in r]
    assert fill_rows and all("cotfail=" in r for r in fill_rows)
    print(f"[smoke] results shard ok: {len(rows)} rows, CoT notes present")

    diag_path = Path("checkpoints/diag/EXP-SMOKE9-delta-diag-s0.jsonl")
    recs = [json.loads(ln) for ln in diag_path.read_text(encoding="utf-8").splitlines()]
    kinds = {r["kind"] for r in recs}
    assert kinds == {"light", "heavy"}, f"diag kinds: {kinds}"
    heavy = [r for r in recs if r["kind"] == "heavy"]
    assert all("probe_acc" in r and "w_norm" in r and "delta_state" in r
               for r in heavy), "heavy record incomplete"
    ds = heavy[-1]["delta_state"]
    assert ds and all(k in ds[0] for k in ("s_fro", "s_erank", "alpha_mean", "beta_mean"))
    lights = [r for r in recs if r["kind"] == "light"]
    assert any(r["curvature"] is not None for r in lights[2:]), "curvature never computed"
    print(f"[smoke] diag sidecar ok: {len(lights)} light + {len(heavy)} heavy records")
    print("[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
