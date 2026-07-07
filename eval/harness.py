"""Evaluation harness: greedy generation on SAGE eval splits, execution-checked
scoring, FLOPs-per-correct accounting, append-only CSV logging.

CSV rows are appended BEFORE any verdict is written into an EXP log (guardrail 4).
"""
from __future__ import annotations

import csv
import datetime
import json
import os
import subprocess
import uuid
from pathlib import Path

import torch

from sage.flops.accounting import generation_flops
from sage.scoring import score_output
from train.data import load_sage_records
from train.tokenizer import BOS, EOS, PAD, decode, encode

# Overridable so parallel shard workers write separate files (merged afterward);
# avoids concurrent-append interleaving on one CSV.
RESULTS_CSV = Path(os.environ.get("AWARE_RESULTS_CSV", "experiments/results.csv"))
MAX_NEW_TOKENS = 48


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "nogit"


@torch.no_grad()
def greedy_generate(model, prompts: list[list[int]], device, max_new: int = MAX_NEW_TOKENS,
                    loop_count: int | None = None,
                    stop_at_newline: bool = True) -> list[list[int]]:
    """Batched greedy decoding without KV cache (correctness over speed at this scale).

    Rows are right-padded; each row's next token is read at its own last real
    position, so PAD never sits inside any row's causal context (it only appears to
    the right of the gathered position, where causal masking makes it irrelevant).

    `stop_at_newline=False` is the CoT decode mode (EXP-004): rationales are
    multi-line, so only EOS (or the budget) terminates a row.
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
                if tok == EOS or (stop_at_newline and tok == ord("\n")):
                    done[r] = True
                else:
                    out_ids[r].append(tok)
                    x[r, pos[r]] = tok
        pos = pos + (~done).long()
        if bool(done.all()):
            break
    return out_ids


def extract_final_answer(text: str) -> tuple[str, bool]:
    """CoT scoring contract (EXP-004): grade ONLY the final answer. Returns
    (answer_text_after_last_ANSWER:, format_ok). A generation that never emits
    'ANSWER:' is a format failure and scores wrong."""
    idx = text.rfind("ANSWER:")
    if idx < 0:
        return "", False
    return text[idx + len("ANSWER:"):], True


def evaluate_model(model, cfg, eval_dir: Path, families: list[str], device,
                   batch_size: int = 32, max_seq: int = 1024,
                   loop_count: int | None = None, limit: int | None = None,
                   cot: bool = False, max_new: int | None = None) -> dict:
    """Returns {family: {difficulty: (n, n_correct)}, ...} plus FLOP totals.

    `cot=True` (EXP-004): the model was trained on the traced form, so the eval
    prompt is the instance prompt WITHOUT the trailing 'ANSWER:' — the model must
    generate 'THINK:\\n<trace>\\nANSWER: <x>' itself. Decoding stops at EOS only,
    within the per-arm `max_new` budget; ONLY the text after the last 'ANSWER:' is
    scored. Every generated token (trace included) is charged a full forward pass
    at its context length by `generation_flops` — CoT is never free.

    The instance filter is IDENTICAL to the direct protocol (MAX_NEW_TOKENS
    constant) so every arm is evaluated on the same instances.
    """
    results: dict = {}
    total_infer_flops = 0.0
    total_correct = 0
    n_total = 0
    format_failures = 0
    tier_flops: dict[int, list[float]] = {}   # difficulty -> [n, flops_sum]
    fcfg = cfg.flops_cfg()
    budget = max_new if max_new is not None else MAX_NEW_TOKENS

    for fam in families:
        recs = load_sage_records(eval_dir / f"{fam}.jsonl", expect_train=False)
        if limit:
            recs = recs[:limit]
        recs = [r for r in recs if len(r["prompt"]) + 2 <= max_seq - MAX_NEW_TOKENS]
        fam_stats: dict[int, list[int]] = {}
        for i in range(0, len(recs), batch_size):
            chunk = recs[i : i + batch_size]
            if cot:
                prompt_texts = [r["prompt"][: r["prompt"].rindex("ANSWER:")] for r in chunk]
            else:
                prompt_texts = [r["prompt"] for r in chunk]
            prompts = [[BOS] + encode(t) for t in prompt_texts]
            gens = greedy_generate(model, prompts, device, max_new=budget,
                                   loop_count=loop_count, stop_at_newline=not cot)
            for rec, prompt, gen in zip(chunk, prompts, gens):
                text = decode(gen)
                if cot:
                    answer_text, format_ok = extract_final_answer(text)
                    ok = format_ok and score_output(answer_text, rec["answer"],
                                                    rec["scoring"])
                    format_failures += int(not format_ok)
                else:
                    ok = score_output(text, rec["answer"], rec["scoring"])
                d = rec["difficulty"]
                fam_stats.setdefault(d, [0, 0])
                fam_stats[d][0] += 1
                fam_stats[d][1] += int(ok)
                n_total += 1
                total_correct += int(ok)
                gf = generation_flops(fcfg, prompt_len=len(prompt),
                                      gen_len=max(1, len(gen)),
                                      loop_count=loop_count)
                total_infer_flops += gf
                t = tier_flops.setdefault(d, [0, 0.0])
                t[0] += 1
                t[1] += gf
        results[fam] = fam_stats

    f35 = [tier_flops[d] for d in (3, 4, 5) if d in tier_flops]
    results["_flops"] = {
        "total_infer_flops": total_infer_flops,
        "flops_per_correct": total_infer_flops / max(1, total_correct),
        "flops_per_answer": total_infer_flops / max(1, n_total),
        # tier 3-5 mean cost per answer: the x-axis of the EXP-004 curves
        "flops_per_answer_t35": (sum(f for _, f in f35)
                                 / max(1, sum(n for n, _ in f35))) if f35 else None,
    }
    if cot:
        results["_cot"] = {"format_failures": format_failures,
                           "format_failure_rate": format_failures / max(1, n_total),
                           "max_new": budget}
    return results


def log_results(results: dict, *, exp_id: str, model_id: str, params: int,
                config_hash: str, seed: int, train_tokens: int, train_flops: float,
                audit_hash: str, notes: str = "") -> str:
    run_id = f"{exp_id}-{model_id}-s{seed}-{uuid.uuid4().hex[:6]}"
    sha = git_sha()
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    fpc = results["_flops"]["flops_per_correct"]
    # EXP-004 curve inputs ride in notes (CSV schema is append-only, columns fixed):
    # FLOPs/answer always; CoT format-failure rate when in CoT mode.
    extras = [f"fpa={results['_flops']['flops_per_answer']:.4g}"] \
        if "flops_per_answer" in results["_flops"] else []
    if results["_flops"].get("flops_per_answer_t35"):
        extras.append(f"fpa35={results['_flops']['flops_per_answer_t35']:.4g}")
    if "_cot" in results:
        extras.append(f"cotfail={results['_cot']['format_failure_rate']:.4f}")
        extras.append(f"maxnew={results['_cot']['max_new']}")
    if extras:
        notes = (notes + "|" if notes else "") + ";".join(extras)
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
    if "flops_per_answer" in results["_flops"]:
        lines.append(f"  flops/answer  = {results['_flops']['flops_per_answer']:.3g}")
    if "_cot" in results:
        lines.append(f"  cot format-failure rate = "
                     f"{results['_cot']['format_failure_rate']:.3f}")
    return "\n".join(lines)
