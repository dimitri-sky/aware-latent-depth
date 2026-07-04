"""Re-evaluate saved checkpoints and log full result rows (recovery path when
training succeeded but result rows were lost; evaluation is cheap, training is not).

    python scripts/reeval_checkpoints.py --glob "checkpoints/EXP-000B-*.pt" --exp-id EXP-000B
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from eval.harness import evaluate_model, log_results, summarize  # noqa: E402
from models import ModelConfig, build_model  # noqa: E402
from models.zoo import n_params  # noqa: E402
from sage.contamination.audit import run_audit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--eval-dir", type=Path, default=Path("data/sage/eval"))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ok, audit_hash, report = run_audit(Path("data/sage/train"), args.eval_dir, [])
    if not ok:
        raise RuntimeError(f"audit failed: {report}")

    ckpts = sorted(Path(".").glob(args.glob))
    print(f"re-evaluating {len(ckpts)} checkpoints")
    for ckpt in ckpts:
        m = re.match(rf"{re.escape(args.exp_id)}-(.+)-s(\d+)$", ckpt.stem)
        if not m:
            print(f"[skip] {ckpt.name}")
            continue
        model_id, seed = m.group(1), int(m.group(2))
        blob = torch.load(ckpt, map_location=args.device, weights_only=False)
        cfg = ModelConfig(**blob["config"])
        model = build_model(cfg).to(args.device)
        model.load_state_dict(blob["model"])
        # families: last token of model_id matches a family for probe runs; A/A uses both
        fam = model_id.split("-")[-1]
        families = [fam] if fam in ("rewrite", "dsl_learn", "algo_exec", "rule_shift",
                                    "compress", "state_guard", "fresh_dsl") \
            else ["algo_exec", "rewrite"]
        results = evaluate_model(model, cfg, args.eval_dir, families, args.device,
                                 max_seq=cfg.max_seq_len, limit=args.limit)
        run_id = log_results(results, exp_id=args.exp_id, model_id=model_id,
                             params=n_params(model), config_hash=cfg.config_hash(),
                             seed=seed, train_tokens=0, train_flops=0.0,
                             audit_hash=audit_hash, notes="reeval from checkpoint")
        print(f"[{model_id} s{seed}] {run_id}\n{summarize(results)}")


if __name__ == "__main__":
    main()
