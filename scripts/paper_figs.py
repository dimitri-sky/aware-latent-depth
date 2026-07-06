"""Paper-grade figures for the fast-weight-memory paper (vector PDF + PNG).

    python scripts/paper_figs.py --out paper/figs
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = Path("experiments/results.csv")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.6,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})

C_AWARE = "#1f6fb2"
C_B2 = "#8a8a8a"
FAM_LABEL = {"algo_exec": "algo\\_exec", "state_guard": "state\\_guard",
             "compress": "compress", "rule_shift": "rule\\_shift",
             "dsl_learn": "dsl\\_learn"}


def load():
    acc, fpc = {}, {}
    for r in csv.DictReader(RESULTS.open(encoding="utf-8")):
        if (r["metric"] != "accuracy" or r["split"] != "eval"
                or "K1diag" in (r.get("notes") or "")):
            continue
        k = (r["exp_id"], r["model_id"], r["family"], r["difficulty"], int(r["seed"]))
        acc[k] = float(r["value"])
        if r["difficulty"] == "all":
            fpc[(r["exp_id"], r["model_id"], r["family"], int(r["seed"]))] = \
                float(r["infer_flops_per_correct"])
    return acc, fpc


def seeds_of(acc, exp, model, fam):
    out = []
    s = 0
    while (exp, model, fam, "all", s) in acc:
        out.append(acc[(exp, model, fam, "all", s)])
        s += 1
    return out


PAIRS = [("algo_exec", "EXP-002-AX"), ("state_guard", "EXP-002-SG"),
         ("compress", "EXP-002-CP"), ("rule_shift", "EXP-002-RS"),
         ("dsl_learn", "EXP-002-DL")]


def fig_hero(acc, fpc, out: Path):
    """Pareto scatter, paper-styled."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    fams = ["algo_exec", "state_guard", "compress", "rule_shift"]
    colors = {"algo_exec": "#1f6fb2", "state_guard": "#2e8b57",
              "compress": "#e08214", "rule_shift": "#c0392b"}
    for fam in fams:
        exp = dict(PAIRS)[fam]
        for model, marker in (("V2-delta", "o"), ("B2-6L", "s")):
            a = seeds_of(acc, exp, model, fam)
            f = [fpc[(exp, model, fam, s)] for s in range(len(a))]
            ax.scatter(st.mean(f), st.mean(a), c=colors[fam], marker=marker,
                       s=110, edgecolors="black", linewidths=0.6, zorder=3)
        pa = (st.mean([fpc[(exp, "V2-delta", fam, s)] for s in
                       range(len(seeds_of(acc, exp, "V2-delta", fam)))]),
              st.mean(seeds_of(acc, exp, "V2-delta", fam)))
        pb = (st.mean([fpc[(exp, "B2-6L", fam, s)] for s in
                       range(len(seeds_of(acc, exp, "B2-6L", fam)))]),
              st.mean(seeds_of(acc, exp, "B2-6L", fam)))
        ax.annotate("", xy=pa, xytext=pb,
                    arrowprops=dict(arrowstyle="-|>", color=colors[fam],
                                    lw=1.6, alpha=0.85))
        ax.annotate(fam.replace("_", "\u2009"), xy=pb,
                    xytext=(6, -11), textcoords="offset points",
                    fontsize=8.5, color=colors[fam])
    h = [plt.Line2D([], [], marker="o", ls="", mec="black", mfc="white",
                    label="memory hybrid (17.86M)"),
         plt.Line2D([], [], marker="s", ls="", mec="black", mfc="white",
                    label="Transformer++ (17.83M)")]
    ax.legend(handles=h, loc="lower left", fontsize=9, frameon=False)
    ax.set_xscale("log")
    ax.set_xlabel("inference FLOPs per correct answer")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.05, 1.02)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"hero_pareto.{ext}", dpi=300)
    plt.close(fig)


def fig_families(acc, out: Path):
    """Per-family accuracy bars with individual seed dots."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    width = 0.36
    xs = np.arange(len(PAIRS))
    for i, (model, color, label) in enumerate(
            (("B2-6L", C_B2, "Transformer++"),
             ("V2-delta", C_AWARE, "memory hybrid"))):
        means, allseeds = [], []
        for fam, exp in PAIRS:
            v = seeds_of(acc, exp, model, fam)
            means.append(st.mean(v))
            allseeds.append(v)
        pos = xs + (i - 0.5) * width
        ax.bar(pos, means, width * 0.92, color=color, label=label, zorder=2)
        for p, v in zip(pos, allseeds):
            jitter = (np.random.RandomState(0).rand(len(v)) - 0.5) * width * 0.5
            ax.scatter(p + jitter, v, s=14, c="black", alpha=0.75, zorder=3,
                       linewidths=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([f.replace("_", "\u2009") for f, _ in PAIRS], fontsize=9)
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.set_ylim(0, 1.05)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"families.{ext}", dpi=300)
    plt.close(fig)


def fig_attribution(acc, out: Path):
    """2x2 ablation on algo_exec."""
    cells = [("Transformer++", ("EXP-002-AX", "B2-6L")),
             ("+ SWA only", ("EXP-006-AX", "B2-SWA")),
             ("+ delta only", ("EXP-006-AX", "V2-full")),
             ("+ delta + SWA", ("EXP-002-AX", "V2-delta"))]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for i, (label, (exp, model)) in enumerate(cells):
        v = seeds_of(acc, exp, model, "algo_exec")
        color = C_AWARE if "delta" in label else C_B2
        ax.bar(i, st.mean(v), 0.62, color=color, zorder=2)
        ax.scatter([i] * len(v), v, s=16, c="black", alpha=0.8, zorder=3,
                   linewidths=0)
        ax.text(i, st.mean(v) + 0.035, f"{st.mean(v):.3f}", ha="center",
                fontsize=8.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([c[0] for c in cells], fontsize=8.5)
    ax.set_ylabel("algo\u2009exec accuracy")
    ax.set_ylim(0, 1.1)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"attribution.{ext}", dpi=300)
    plt.close(fig)


def fig_budget(acc, out: Path):
    """Training-budget robustness."""
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for model, color, label, marker in (
            ("B2-6L", C_B2, "Transformer++", "s"),
            ("V2-delta", C_AWARE, "memory hybrid", "o")):
        y4 = seeds_of(acc, "EXP-002-AX", model, "algo_exec")
        y8 = seeds_of(acc, "EXP-007", model, "algo_exec")
        ax.plot([4000, 8000], [st.mean(y4), st.mean(y8)], marker=marker,
                color=color, label=label, lw=1.8, ms=7)
        ax.scatter([4000] * len(y4) + [8000] * len(y8), y4 + y8, s=16,
                   c="black", alpha=0.75, zorder=3, linewidths=0)
    ax.set_xticks([4000, 8000])
    ax.set_xlabel("training steps (identical batch and sequence length)")
    ax.set_ylabel("algo\u2009exec accuracy")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.set_ylim(0.55, 1.06)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"budget.{ext}", dpi=300)
    plt.close(fig)


def fig_scale(acc, out: Path):
    """Gap at 18M vs 50M for the two scaled families."""
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    data = {
        "algo\u2009exec": [
            st.mean(seeds_of(acc, "EXP-002-AX", "V2-delta", "algo_exec"))
            - st.mean(seeds_of(acc, "EXP-002-AX", "B2-6L", "algo_exec")),
            st.mean(seeds_of(acc, "EXP-008-AX", "V3-50M", "algo_exec"))
            - st.mean(seeds_of(acc, "EXP-008-AX", "B2-50M", "algo_exec"))],
        "state\u2009guard": [
            st.mean(seeds_of(acc, "EXP-002-SG", "V2-delta", "state_guard"))
            - st.mean(seeds_of(acc, "EXP-002-SG", "B2-6L", "state_guard")),
            st.mean(seeds_of(acc, "EXP-008-SG", "V3-50M", "state_guard"))
            - st.mean(seeds_of(acc, "EXP-008-SG", "B2-50M", "state_guard"))],
    }
    xs = np.arange(2)
    for i, (fam, gaps) in enumerate(data.items()):
        color = "#1f6fb2" if i == 0 else "#2e8b57"
        ax.plot(xs, gaps, marker="o", lw=1.8, ms=7, color=color, label=fam)
        for x, g in zip(xs, gaps):
            ax.text(x, g + 0.015, f"{g*100:+.1f}", ha="center", fontsize=8.5,
                    color=color)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["18M parameters", "50M parameters"])
    ax.set_ylabel("accuracy gap (hybrid \u2212 TF++)")
    ax.legend(fontsize=9, frameon=False)
    ax.set_ylim(-0.12, 0.40)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"scale.{ext}", dpi=300)
    plt.close(fig)


def fig_grok(acc, out: Path):
    """Bimodal seed outcomes on rule_shift."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    v2 = seeds_of(acc, "EXP-002-RS", "V2-delta", "rule_shift")
    b2 = seeds_of(acc, "EXP-002-RS", "B2-6L", "rule_shift")
    for i, (vals, color, label) in enumerate(
            ((b2, C_B2, "Transformer++"), (v2, C_AWARE, "memory hybrid"))):
        ax.scatter([i] * len(vals), vals, s=60, c=color, alpha=0.85,
                   edgecolors="black", linewidths=0.5, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Transformer++\n(6 seeds)", "memory hybrid\n(6 seeds)"])
    ax.set_ylabel("rule\u2009shift accuracy")
    ax.set_ylim(0, 1.05)
    ax.annotate("1/6 seeds transitions\nto 100%", xy=(1, 1.0),
                xytext=(0.35, 0.82), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=1.0))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"grok.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("paper/figs"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    acc, fpc = load()
    fig_hero(acc, fpc, args.out)
    fig_families(acc, args.out)
    fig_attribution(acc, args.out)
    fig_budget(acc, args.out)
    fig_scale(acc, args.out)
    fig_grok(acc, args.out)
    print(f"wrote 6 figures (pdf+png) to {args.out}")


if __name__ == "__main__":
    main()
