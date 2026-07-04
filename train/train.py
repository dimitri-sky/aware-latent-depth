"""Shared training loop for all architectures. Config-driven; identical treatment for
every model (guardrail: fairness). Runs the contamination audit first, trains, then
evaluates and appends to results.csv.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.harness import evaluate_model, log_results, summarize  # noqa: E402
from models import ModelConfig, build_model  # noqa: E402
from models.zoo import n_params  # noqa: E402
from sage.contamination.audit import run_audit  # noqa: E402
from sage.flops.accounting import training_flops  # noqa: E402
from train.data import SageDataset  # noqa: E402


@dataclass
class TrainConfig:
    families: list
    steps: int = 3000
    batch_size: int = 32
    seq_len: int = 768
    lr: float = 3e-4
    warmup: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 0
    traced: bool = False           # CoT-format training (B2-CoT baseline)
    data_dir: str = "data/sage/train"
    eval_dir: str = "data/sage/eval"
    eval_limit: int | None = 300
    log_every: int = 100


def lr_at(step: int, tc: TrainConfig) -> float:
    if step < tc.warmup:
        return tc.lr * step / max(1, tc.warmup)
    p = (step - tc.warmup) / max(1, tc.steps - tc.warmup)
    return tc.lr * 0.5 * (1 + math.cos(math.pi * p))


def train_one(mcfg: ModelConfig, tc: TrainConfig, exp_id: str, model_id: str,
              device: str = "cuda", eval_loop_count: int | None = None,
              notes: str = "") -> dict:
    torch.manual_seed(tc.seed)
    np_rng = np.random.default_rng(tc.seed)

    ok, audit_hash, report = run_audit(Path(tc.data_dir), Path(tc.eval_dir), [])
    if not ok:
        raise RuntimeError(f"contamination audit FAILED: {report}")

    ds = SageDataset(Path(tc.data_dir), tc.families, tc.seq_len, traced=tc.traced)
    print(f"[{model_id} s{tc.seed}] {len(ds.sequences)} train sequences "
          f"({ds.skipped} skipped), ~{ds.tokens_per_epoch():,} tokens/epoch")

    model = build_model(mcfg).to(device)
    params = n_params(model)
    print(f"[{model_id}] params={params / 1e6:.2f}M hash={mcfg.config_hash()}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": tc.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=tc.lr, betas=(0.9, 0.95))

    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" \
        else torch.autocast(device_type="cpu", enabled=False)

    step, tokens_seen, t0 = 0, 0, time.time()
    losses = []
    while step < tc.steps:
        for x, y in ds.batches(tc.batch_size, np_rng, device):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, tc)
            with amp:
                _, loss = model(x, targets=y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            tokens_seen += int((y != -100).sum())
            losses.append(loss.item())
            step += 1
            if step % tc.log_every == 0:
                dt = time.time() - t0
                print(f"[{model_id} s{tc.seed}] step {step}/{tc.steps} "
                      f"loss {np.mean(losses[-tc.log_every:]):.4f} "
                      f"{tokens_seen / dt:.0f} tok/s", flush=True)
            if step >= tc.steps:
                break

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt = ckpt_dir / f"{exp_id}-{model_id}-s{tc.seed}.pt"
    torch.save({"model": model.state_dict(), "config": mcfg.to_dict()}, ckpt)

    tf = training_flops(mcfg.flops_cfg(), tokens_seen, tc.seq_len // 2,
                        loop_count=getattr(mcfg, "loop_count", None)
                        if mcfg.arch == "loop" else None)
    results = evaluate_model(model, mcfg, Path(tc.eval_dir), tc.families, device,
                             max_seq=mcfg.max_seq_len, loop_count=eval_loop_count,
                             limit=tc.eval_limit)
    run_id = log_results(results, exp_id=exp_id, model_id=model_id, params=params,
                         config_hash=mcfg.config_hash(), seed=tc.seed,
                         train_tokens=tokens_seen, train_flops=tf,
                         audit_hash=audit_hash, notes=notes)
    print(f"[{model_id} s{tc.seed}] run_id={run_id}\n{summarize(results)}", flush=True)
    return {"run_id": run_id, "results": results, "params": params,
            "train_tokens": tokens_seen, "model": model}
