"""Evaluation harness: greedy generation on SAGE eval splits, execution-checked
scoring, FLOPs-per-correct accounting, append-only CSV logging.

CSV rows are appended BEFORE any verdict is written into an EXP log (guardrail 4).
"""
from __future__ import annotations

import csv
import datetime
import json
import subprocess
import uuid
from pathlib import Path

import torch

from sage.flops.accounting import generation_flops
from sage.scoring import score_output
from train.data import load_sage_records
from train.tokenizer import BOS, EOS, PAD, decode, encode

RESULTS_CSV = Path("experiments/results.csv")
MAX_NEW_TOKENS = 48


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "nogit"


@torch.no_grad()
def greedy_generate(model, prompts: list[list[int]], device, max_new: int = MAX_NEW_TOKENS,
                    loop_count: int | None = None) -> list[list[int]]:
    """Batched greedy decoding without KV cache (correctness over speed at this scale).

    Rows are right-padded; each row's next token is read at its own last real
    position, so PAD never sits inside any row's causal context (it only appears to
    the right of the gathered position, where causal masking makes it irrelevant).
    """
    model.eval()
    b = len(prompts)
    lens = [len(p) for p in prompts]
    width = max(lens) + max_new
    x = torch.full((b, width), PAD, dtype=torch.long, device=device)
    for r, p in enumerate(prompts):
        x[r, : len(p)] = torch.tensor(p, device=device)
    pos = torch.tensor(lens, device=device)          # next write position per row
    done = torch.zeros(b, dtype=torch.bool, device=device)
    out_ids = [[] for _ in range(b)]
    kwargs = {"loop_count": loop_count} if loop_count is not None else {}
    for _ in range(max_new):
        upto = int(pos.max().item())
        logits, _ = model(x[:, :upto], **kwargs)
        nxt = logits[torch.arange(b, device=device), pos - 1].argmax(-1)
        for r in range(b):
            if not done[r]:
                tok = int(nxt[r])
                if tok == EOS or tok == ord("\n"):
                    done[r] = True
                else:
                    out_ids[r].append(tok)
                    x[r, pos[r]] = tok
        pos = pos + (~done).long()
        if bool(done.all()):
            break
    return out_ids


def evaluate_model(model, cfg, eval_dir: Path, families: list[str], device,
                   batch_size: int = 32, max_seq: int = 1024,
                   loop_count: int | None = None, limit: int | None = None) -> dict:
    """Returns {family: {difficulty: (n, n_correct)}, ...} plus FLOP totals."""
    results: dict = {}
    total_infer_flops = 0.0
    total_correct = 0
    fcfg = cfg.flops_cfg()

    for fam in families:
        recs = load_sage_records(eval_dir / f"{fam}.jsonl", expect_train=False)
        if limit:
            recs = recs[:limit]
        recs = [r for r in recs if len(r["prompt"]) + 2 <= max_seq - MAX_NEW_TOKENS]
        fam_stats: dict[int, list[int]] = {}
        for i in range(0, len(recs), batch_size):
            chunk = recs[i : i + batch_size]
            prompts = [[BOS] + encode(r["prompt"]) for r in chunk]
            gens = greedy_generate(model, prompts, device, loop_count=loop_count)
            for rec, prompt, gen in zip(chunk, prompts, gens):
                text = decode(gen)
                ok = score_output(text, rec["answer"], rec["scoring"])
                d = rec["difficulty"]
                fam_stats.setdefault(d, [0, 0])
                fam_stats[d][0] += 1
                fam_stats[d][1] += int(ok)
                total_correct += int(ok)
                total_infer_flops += generation_flops(
                    fcfg, prompt_len=len(prompt), gen_len=max(1, len(gen)),
                    loop_count=loop_count)
        results[fam] = fam_stats

    results["_flops"] = {
        "total_infer_flops": total_infer_flops,
        "flops_per_correct": total_infer_flops / max(1, total_correct),
    }
    return results


def log_results(results: dict, *, exp_id: str, model_id: str, params: int,
                config_hash: str, seed: int, train_tokens: int, train_flops: float,
                audit_hash: str, notes: str = "") -> str:
    run_id = f"{exp_id}-{model_id}-s{seed}-{uuid.uuid4().hex[:6]}"
    sha = git_sha()
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    fpc = results["_flops"]["flops_per_correct"]
    rows = []
    for fam, stats in results.items():
        if fam.startswith("_"):
            continue
        for d, (n, c) in sorted(stats.items()):
            rows.append([run_id, ts, sha, config_hash, exp_id, model_id, params, fam,
                         d, "eval", seed, "accuracy", round(c / max(1, n), 4),
                         train_tokens, f"{train_flops:.4g}", f"{fpc:.4g}", audit_hash, notes])
        n_all = sum(v[0] for v in stats.values())
        c_all = sum(v[1] for v in stats.values())
        rows.append([run_id, ts, sha, config_hash, exp_id, model_id, params, fam,
                     "all", "eval", seed, "accuracy", round(c_all / max(1, n_all), 4),
                     train_tokens, f"{train_flops:.4g}", f"{fpc:.4g}", audit_hash, notes])
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    return run_id


def summarize(results: dict) -> str:
    lines = []
    for fam, stats in results.items():
        if fam.startswith("_"):
            continue
        n = sum(v[0] for v in stats.values())
        c = sum(v[1] for v in stats.values())
        lines.append(f"  {fam:12s} {c}/{n} = {c / max(1, n):.3f}")
    lines.append(f"  flops/correct = {results['_flops']['flops_per_correct']:.3g}")
    return "\n".join(lines)
