"""EXP-009 adjudication: grokking reproduction rate (Q1) + pre-registered
first-1000-step predictor signals (Q2). See agent/log/EXP-009.md for definitions.

Inputs: results.csv EXP-009 rows (final eval, rule_shift 'all' difficulty) and
checkpoints/diag/EXP-009-V2-delta-s*.jsonl sidecars.

    python scripts/exp009_grok_analysis.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RATE_SEEDS = set(range(6, 22))          # fresh seeds; s2 re-run excluded from Q1
GROK_AT, FLAT_AT = 0.9, 0.3             # final-eval all-tier thresholds
TRANSITION_AT = 0.5                     # probe all-tier crossing
WINDOW_END = 1000

SIGNALS = {                              # name -> (record kind, extractor)
    "loss_curvature": ("light", lambda r: r.get("curvature")),
    "grad_norm_mean": ("light", lambda r: r.get("grad_norm_mean")),
    "grad_norm_max": ("light", lambda r: r.get("grad_norm_max")),
    "s_fro_mean": ("heavy", lambda r: _ds_mean(r, "s_fro")),
    "s_erank_mean": ("heavy", lambda r: _ds_mean(r, "s_erank")),
    "alpha_mean": ("heavy", lambda r: _ds_mean(r, "alpha_mean")),
    "beta_mean": ("heavy", lambda r: _ds_mean(r, "beta_mean")),
    "w_norm_ratio": ("heavy", lambda r: (r["w_norm_delta"] / r["w_norm_attn"])
                     if r.get("w_norm_attn") else None),
    "tier1_probe": ("heavy", lambda r: (r.get("probe_acc") or {}).get("1")),
}


def _ds_mean(rec: dict, key: str):
    ds = rec.get("delta_state")
    if not ds:
        return None
    return sum(layer[key] for layer in ds) / len(ds)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def final_accs(csv_path: Path) -> dict[int, float]:
    out = {}
    with csv_path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if (r["exp_id"] == "EXP-009" and r["family"] == "rule_shift"
                    and r["difficulty"] == "all"):
                out[int(r["seed"])] = float(r["value"])
    return out


def load_diag(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def transition_step(recs: list[dict]) -> int | None:
    for r in recs:
        if r["kind"] == "heavy" and r.get("probe_acc_all", 0) >= TRANSITION_AT:
            return r["step"]
    return None


def window_values(recs: list[dict], kind: str, extract, end: int) -> list[float]:
    vals = []
    for r in recs:
        if r["kind"] != kind or r["step"] > end:
            continue
        v = extract(r)
        if v is not None:
            vals.append(float(v))
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("experiments/results.csv"))
    ap.add_argument("--diag-dir", type=Path, default=Path("checkpoints/diag"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/figs"))
    args = ap.parse_args()

    accs = final_accs(args.csv)
    groups: dict[str, list[int]] = {"grokked": [], "flat": [], "intermediate": []}
    for s, a in sorted(accs.items()):
        if s not in RATE_SEEDS:
            continue
        g = "grokked" if a >= GROK_AT else ("flat" if a <= FLAT_AT else "intermediate")
        groups[g].append(s)
    n = sum(len(v) for v in groups.values())
    k = len(groups["grokked"])
    lo, hi = wilson(k, n)
    print(f"=== Q1: reproduction rate ===")
    print(f"  grokked {k}/{n} = {k / max(1, n):.3f}  Wilson95 [{lo:.3f}, {hi:.3f}] "
          f"(prior point estimate 1/6 = .167)")
    for g, seeds in groups.items():
        print(f"  {g:12s}: {seeds} " +
              str([round(accs[s], 3) for s in seeds]))
    if 2 in accs:
        print(f"  seed-2 re-run (excluded from rate): final acc {accs[2]:.3f}")

    diags = {}
    for s in list(accs):
        p = args.diag_dir / f"EXP-009-V2-delta-s{s}.jsonl"
        if p.exists():
            diags[s] = load_diag(p)
    trans = {s: transition_step(r) for s, r in diags.items()}
    tsteps = {s: t for s, t in trans.items() if t is not None}
    if tsteps:
        print(f"  transition steps (probe >= {TRANSITION_AT}): {tsteps}")

    print(f"\n=== Q2: predictor-window signals (steps <= {WINDOW_END}, "
          f"pre-transition) ===")
    gr, fl = groups["grokked"], groups["flat"]
    if not gr:
        print("  k=0 grokked -> Q2 void per pre-registration; distributions below "
              "are the null reference.")
    rows = []
    for name, (kind, extract) in SIGNALS.items():
        stats = {}
        for label, seeds in (("grokked", gr), ("flat", fl)):
            per_seed = []
            for s in seeds:
                if s not in diags:
                    continue
                end = min(WINDOW_END, (trans.get(s) or WINDOW_END + 1) - 1)
                vals = window_values(diags[s], kind, extract, end)
                if vals:
                    per_seed.append(statistics.mean(vals))
            stats[label] = per_seed
        g_, f_ = stats["grokked"], stats["flat"]
        line = f"  {name:16s} grok(n={len(g_)})=" + \
               (f"{statistics.mean(g_):8.4f}" if g_ else "     n/a") + \
               f"  flat(n={len(f_)})=" + \
               (f"{statistics.mean(f_):8.4f}" if f_ else "     n/a")
        verdict = ""
        if len(g_) >= 2 and len(f_) >= 2:
            fsd = statistics.stdev(f_)
            gap = abs(statistics.mean(g_) - statistics.mean(f_))
            separated = (min(g_) > max(f_)) or (max(g_) < min(f_))
            if separated and fsd > 0 and gap > 2 * fsd:
                verdict = "  ** CANDIDATE PREDICTOR (perfect separation, >2x flat SD)"
            elif separated:
                verdict = "  separated but < 2x flat SD"
        print(line + verdict)
        rows.append((name, g_, f_))

    # trajectory figure: probe accuracy + a couple of signals over steps
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [("probe_acc_all", "heavy", lambda r: r.get("probe_acc_all")),
              ("loss (light)", "light", lambda r: r.get("loss")),
              ("s_fro_mean", "heavy", lambda r: _ds_mean(r, "s_fro")),
              ("beta_mean", "heavy", lambda r: _ds_mean(r, "beta_mean"))]
    for ax, (title, kind, ex) in zip(axes.flat, panels):
        for s, recs in sorted(diags.items()):
            xs = [r["step"] for r in recs if r["kind"] == kind and ex(r) is not None]
            ys = [ex(r) for r in recs if r["kind"] == kind and ex(r) is not None]
            grokked = s in groups["grokked"] or (s == 2 and accs.get(2, 0) >= GROK_AT)
            ax.plot(xs, ys, color="tab:red" if grokked else "tab:gray",
                    alpha=.9 if grokked else .35, lw=1.5 if grokked else .8)
        ax.set_title(title)
        ax.grid(alpha=.3)
        ax.set_xlabel("step")
    fig.suptitle("EXP-009 diagnostics (red = grokked, gray = others)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "exp009_grok_signals.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
