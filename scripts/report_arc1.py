"""Arc-1 report artifacts: claim-table numbers + Pareto figure from results.csv.

    python scripts/report_arc1.py          # prints tables, writes reports/figs/
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path("experiments/results.csv")
FIGS = Path("reports/figs")

# (exp_id, model_id) -> plot label; the 18M head-to-head suite
PAIRS = {
    "rule_shift": ("EXP-002-RS", 6),
    "compress": ("EXP-002-CP", 3),
    "state_guard": ("EXP-002-SG", 3),
    "algo_exec": ("EXP-002-AX", 3),
}
MODELS = {"V2-delta": "AWARE (delta)", "B2-6L": "B2 (Transformer++)",
          "B2-SWA": "B2-SWA", "V2-full": "V2-full"}


def load():
    acc, fpc = {}, {}
    for r in csv.DictReader(RESULTS.open(encoding="utf-8")):
        if (r["metric"] != "accuracy" or r["split"] != "eval"
                or "K1diag" in (r.get("notes") or "")):
            continue
        k = (r["exp_id"], r["model_id"], r["family"], r["difficulty"], int(r["seed"]))
        acc[k] = float(r["value"])
        if r["difficulty"] == "all":
            fpc[k[:3] + (k[4],)] = float(r["infer_flops_per_correct"])
    return acc, fpc


def main() -> None:
    acc, fpc = load()
    FIGS.mkdir(parents=True, exist_ok=True)

    print("=== 18M head-to-head (all-tier mean over seeds; flops/correct) ===")
    rows = []
    for fam, (exp, nseeds) in PAIRS.items():
        for m in ("V2-delta", "B2-6L"):
            a = [acc.get((exp, m, fam, "all", s)) for s in range(nseeds)]
            a = [x for x in a if x is not None]
            f = [fpc.get((exp, m, fam, s)) for s in range(nseeds)]
            f = [x for x in f if x is not None]
            rows.append((fam, m, st.mean(a), st.mean(f), len(a)))
            print(f"  {fam:12s} {m:9s} acc={st.mean(a):.3f} (n={len(a)})"
                  f"  fpc={st.mean(f):.3g}")

    # Pareto figure: x = inference FLOPs/correct, y = accuracy, one marker per
    # (family, model); delta model should sit up-left of B2 everywhere.
    fig, ax = plt.subplots(figsize=(7, 5))
    fam_colors = {"rule_shift": "tab:red", "compress": "tab:orange",
                  "state_guard": "tab:green", "algo_exec": "tab:blue"}
    for fam, m, a, f, _ in rows:
        marker = "o" if m == "V2-delta" else "s"
        ax.scatter(f, a, c=fam_colors[fam], marker=marker, s=90,
                   edgecolors="black", linewidths=0.5, zorder=3)
    for fam, (exp, nseeds) in PAIRS.items():
        pts = {m: (st.mean([fpc[(exp, m, fam, s)] for s in range(nseeds)
                            if (exp, m, fam, s) in fpc]),
                   st.mean([acc[(exp, m, fam, "all", s)] for s in range(nseeds)
                            if (exp, m, fam, "all", s) in acc]))
               for m in ("V2-delta", "B2-6L")}
        ax.annotate("", xy=pts["V2-delta"], xytext=pts["B2-6L"],
                    arrowprops=dict(arrowstyle="->", color=fam_colors[fam],
                                    alpha=0.6, lw=1.4))
    handles = [plt.Line2D([], [], marker="o", ls="", mec="black", mfc="gray",
                          label="AWARE (delta memory)"),
               plt.Line2D([], [], marker="s", ls="", mec="black", mfc="gray",
                          label="B2 (Transformer++)")]
    handles += [plt.Line2D([], [], marker="o", ls="", mfc=c, mec="none", label=f)
                for f, c in fam_colors.items()]
    ax.legend(handles=handles, fontsize=8, loc="lower left")
    ax.set_xscale("log")
    ax.set_xlabel("inference FLOPs per correct answer (log)")
    ax.set_ylabel("accuracy (all tiers)")
    ax.set_title("Same params, same training budget: accuracy vs cost per correct answer\n"
                 "(arrows: B2 \u2192 AWARE, per SAGE family; up-left is better)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = FIGS / "pareto_18m.png"
    fig.savefig(out, dpi=160)
    print(f"\nwrote {out}")

    # Budget-robustness mini-figure (EXP-007)
    fig2, ax2 = plt.subplots(figsize=(5.4, 4))
    b4 = st.mean([acc[("EXP-002-AX", "B2-6L", "algo_exec", "all", s)] for s in range(3)])
    v4 = st.mean([acc[("EXP-002-AX", "V2-delta", "algo_exec", "all", s)] for s in range(3)])
    b8 = st.mean([acc[("EXP-007", "B2-6L", "algo_exec", "all", s)] for s in range(2)])
    v8 = st.mean([acc[("EXP-007", "V2-delta", "algo_exec", "all", s)] for s in range(2)])
    ax2.plot([4000, 8000], [b4, b8], "s-", color="tab:gray", label="B2 (Transformer++)")
    ax2.plot([4000, 8000], [v4, v8], "o-", color="tab:blue", label="AWARE (delta)")
    ax2.set_xticks([4000, 8000])
    ax2.set_xlabel("training steps (same batch/seq)")
    ax2.set_ylabel("algo_exec accuracy (all tiers)")
    ax2.set_title("Doubling the training budget does not close the gap")
    ax2.legend()
    ax2.grid(alpha=0.25)
    fig2.tight_layout()
    out2 = FIGS / "budget_robustness.png"
    fig2.savefig(out2, dpi=160)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
