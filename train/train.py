"""Shared training loop for all architectures. Config-driven; identical treatment for
every model (guardrail: fairness). Runs the contamination audit first, trains, then
evaluates and appends to results.csv.
"""
from __future__ import annotations

import math
import os
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
from train.data import SageDataset, cot_decode_budget  # noqa: E402


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
    trace_level: str = "long"      # long | med | short | filler (EXP-004 budget knob)
    diag: bool = False             # EXP-009 grokking diagnostics (JSONL sidecar)
    data_dir: str = "data/sage/train"
    eval_dir: str = "data/sage/eval"
    eval_limit: int | None = 300
    log_every: int = 100
    # GPU duty-cycle throttle: sleep this fraction of each step's duration.
    # Default 0.25 locally — protects consumer PSUs from the transient-spike shutdowns
    # observed on this box. Cloud pods set AWARE_THROTTLE=0 for full speed.
    throttle: float = float(os.environ.get("AWARE_THROTTLE", "0.25"))


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

    ds = SageDataset(Path(tc.data_dir), tc.families, tc.seq_len, traced=tc.traced,
                     trace_level=tc.trace_level)
    print(f"[{model_id} s{tc.seed}] {len(ds.sequences)} train sequences "
          f"({ds.skipped} skipped), ~{ds.tokens_per_epoch():,} tokens/epoch"
          + (f", traced={tc.trace_level}" if tc.traced else ""))

    model = build_model(mcfg).to(device)
    params = n_params(model)
    print(f"[{model_id}] params={params / 1e6:.2f}M hash={mcfg.config_hash()}")

    diag = None
    if tc.diag:
        from train.diagnostics import DiagRecorder
        diag = DiagRecorder(exp_id, model_id, tc.seed, tc.families, device,
                            total_steps=tc.steps)
        print(f"[{model_id} s{tc.seed}] diagnostics on -> {diag.path}", flush=True)

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
                logits, loss = model(x, targets=y)
                # z-loss: standard small-LM logit stabilizer; EXP-000B showed
                # seed-dependent degenerate collapse (train loss fine, eval ~0)
                mask = y != -100
                if mask.any():
                    z = logits.float().logsumexp(-1)[mask]
                    loss = loss + 1e-4 * (z ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            # FLOPs are paid on every processed token, not just supervised ones
            tokens_seen += int(x.numel())
            if tc.throttle > 0 and device == "cuda":
                torch.cuda.synchronize()
                step_dt = (time.time() - t0) / max(1, step + 1)
                time.sleep(min(0.2, step_dt * tc.throttle))
            losses.append(loss.item())
            step += 1
            if diag is not None:
                diag.log_step(step, losses[-1], float(grad_norm))
                diag.heavy(step, model, mcfg)
                diag.maybe_ckpt(step, model, mcfg)
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
    # CoT arms (EXP-004): per-arm decode budget = p99 of the trained suffix length
    # over the eval split + 16 slack. Self-adjusting per trace level; charged
    # honestly by generation_flops either way.
    cot_max_new = None
    if tc.traced:
        cot_max_new = cot_decode_budget(Path(tc.eval_dir), tc.families,
                                        tc.trace_level)
        print(f"[{model_id} s{tc.seed}] CoT eval: trace_level={tc.trace_level} "
              f"max_new={cot_max_new}", flush=True)
    results = evaluate_model(model, mcfg, Path(tc.eval_dir), tc.families, device,
                             max_seq=mcfg.max_seq_len, loop_count=eval_loop_count,
                             limit=tc.eval_limit, cot=tc.traced, max_new=cot_max_new)
    run_id = log_results(results, exp_id=exp_id, model_id=model_id, params=params,
                         config_hash=mcfg.config_hash(), seed=tc.seed,
                         train_tokens=tokens_seen, train_flops=tf,
                         audit_hash=audit_hash, notes=notes)
    print(f"[{model_id} s{tc.seed}] run_id={run_id}\n{summarize(results)}", flush=True)

    # K-gap diagnostic (readout blind spot, arXiv:2606.24898): a loop model whose
    # accuracy is identical at K=1 and K=trained is not using its iterations, no
    # matter how healthy the loss looks. Logged as a separate run with K1diag notes.
    if mcfg.arch == "loop" and mcfg.loop_count > 1 and eval_loop_count is None:
        k1 = evaluate_model(model, mcfg, Path(tc.eval_dir), tc.families, device,
                            max_seq=mcfg.max_seq_len, loop_count=1,
                            limit=tc.eval_limit)
        k1_id = log_results(k1, exp_id=exp_id, model_id=model_id, params=params,
                            config_hash=mcfg.config_hash(), seed=tc.seed,
                            train_tokens=tokens_seen, train_flops=tf,
                            audit_hash=audit_hash,
                            notes=(notes + "|" if notes else "") + "K1diag")
        def _acc(res):
            n = sum(v[0] for f, s in res.items() if not f.startswith("_") for v in s.values())
            c = sum(v[1] for f, s in res.items() if not f.startswith("_") for v in s.values())
            return c / max(1, n)
        print(f"[{model_id} s{tc.seed}] K-gap: K={mcfg.loop_count} acc={_acc(results):.3f} "
              f"vs K=1 acc={_acc(k1):.3f} (diag run {k1_id})", flush=True)

    return {"run_id": run_id, "results": results, "params": params,
            "train_tokens": tokens_seen, "model": model}
