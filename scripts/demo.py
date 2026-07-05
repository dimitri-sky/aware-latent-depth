"""`aware demo` — V3 (delta memory) vs B2 (Transformer++) on freshly generated
puzzles, with live FLOPs-per-correct accounting.

Each puzzle is generated on the fly with a seed far outside the training range
(fresh language / fresh rules every time), so neither model has seen it.

    python scripts/demo.py --family dsl_learn --n 5 --difficulty 3 \
        --ckpt-a checkpoints/EXP-002-DL-V2-delta-s0.pt \
        --ckpt-b checkpoints/EXP-002-DL-B2-6L-s0.pt
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from eval.harness import greedy_generate  # noqa: E402
from models import ModelConfig, build_model  # noqa: E402
from sage.flops.accounting import generation_flops  # noqa: E402
from sage.scoring import score_output  # noqa: E402
from train.tokenizer import BOS, decode, encode  # noqa: E402

DEMO_SEED_BASE = 10_000_000  # far outside any training seed range


def load(ckpt: Path, device: str):
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = ModelConfig(**blob["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    return model, cfg, n


def main() -> None:
    ap = argparse.ArgumentParser(description="Aware demo: memory vs attention, priced honestly")
    ap.add_argument("--family", default="dsl_learn")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--difficulty", type=int, default=3, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--ckpt-a", type=Path, default=Path("checkpoints/EXP-002-DL-V2-delta-s0.pt"),
                    help="the memory model (V3/'Aware')")
    ap.add_argument("--ckpt-b", type=Path, default=Path("checkpoints/EXP-002-DL-B2-6L-s0.pt"),
                    help="the baseline (B2 Transformer++)")
    ap.add_argument("--show-prompt", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    gen_mod = importlib.import_module(f"sage.generators.{args.family}")

    (model_a, cfg_a, n_a) = load(args.ckpt_a, args.device)
    (model_b, cfg_b, n_b) = load(args.ckpt_b, args.device)
    name_a = f"AWARE (delta memory, {n_a/1e6:.1f}M)"
    name_b = f"B2 (Transformer++, {n_b/1e6:.1f}M)"

    print(f"\n=== aware demo: {args.family}, difficulty {args.difficulty}, "
          f"{args.n} fresh puzzles ===")
    print(f"  A: {name_a}\n  B: {name_b}\n")

    stats = {"a": [0, 0.0], "b": [0, 0.0]}  # correct, flops
    for i in range(args.n):
        inst = gen_mod.generate(DEMO_SEED_BASE + i, args.difficulty)
        prompt_ids = [BOS] + encode(inst.prompt)
        if args.show_prompt:
            print("-" * 60)
            print(inst.prompt)

        row = {}
        for key, model, cfg in (("a", model_a, cfg_a), ("b", model_b, cfg_b)):
            t0 = time.perf_counter()
            gen = greedy_generate(model, [prompt_ids], args.device)[0]
            dt = time.perf_counter() - t0
            text = decode(gen)
            ok = score_output(text, inst.answer, inst.scoring)
            fl = generation_flops(cfg.flops_cfg(), prompt_len=len(prompt_ids),
                                  gen_len=max(1, len(gen)))
            stats[key][0] += int(ok)
            stats[key][1] += fl
            row[key] = (text.strip(), ok, fl, dt)

        print(f"puzzle {i+1}: expected {inst.answer!r}")
        for key, name in (("a", "AWARE"), ("b", "B2   ")):
            text, ok, fl, dt = row[key]
            mark = "OK " if ok else "MISS"
            print(f"  {name} [{mark}] {text!r:24s} {fl/1e9:6.2f} GFLOPs  {dt*1e3:5.0f} ms")

    print("\n=== scoreboard ===")
    for key, name in (("a", name_a), ("b", name_b)):
        c, fl = stats[key]
        fpc = fl / max(1, c)
        print(f"  {name}")
        print(f"    correct: {c}/{args.n}   total: {fl/1e9:.1f} GFLOPs"
              f"   FLOPs/correct: {fpc/1e9:.1f}G" + ("  (no correct answers)" if c == 0 else ""))
    ca, cb = stats["a"][0], stats["b"][0]
    if ca and cb:
        ratio = (stats["b"][1] / cb) / (stats["a"][1] / ca)
        print(f"\n  => each correct answer costs B2 {ratio:.2f}x what it costs AWARE")


if __name__ == "__main__":
    main()
