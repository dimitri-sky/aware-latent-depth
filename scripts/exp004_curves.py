"""EXP-004 adjudication: exchange-rate curves (tier 3-5 accuracy vs FLOPs/answer)
+ pre-registered gates G1-G4 (agent/log/EXP-004.md).

Data sources: results.csv rows for EXP-004-AX / EXP-004-RS (new arms; measured
fpa35 parsed from run notes) and the REUSED arms declared in the pre-registration
(EXP-002-AX / EXP-002-RS B2-6L + V2-delta; fpa35 from the analytic FLOP-match
table since their notes predate the fpa fields — flagged on the plot).

    python scripts/exp004_curves.py [--csv experiments/results.csv]
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PRIMARY_TIERS = ("3", "4", "5")
MARGIN = 3.0  # points, program convention

# Reused direct arms (pre-registered): exp_id, model_id -> curve arm name.
REUSED = {
    ("EXP-002-AX", "B2-6L"): ("algo_exec", "B2-direct"),
    ("EXP-002-AX", "V2-delta"): ("algo_exec", "V3-direct"),
    ("EXP-002-RS", "B2-6L"): ("rule_shift", "B2-direct"),
    ("EXP-002-RS", "V2-delta"): ("rule_shift", "V3-direct"),
}
# Analytic fpa35 for reused arms (EXP-004.md FLOP-match table).
ANALYTIC_FPA35 = {
    ("algo_exec", "B2-direct"): 8.73e9,
    ("algo_exec", "V3-direct"): 8.36e9,
    ("rule_shift", "B2-direct"): 1.26e10,
    ("rule_shift", "V3-direct"): 1.18e10,
}
NEW_EXPS = {"EXP-004-AX": "algo_exec", "EXP-004-RS": "rule_shift"}
# Curve ordering for the CoT budget line (per family).
COT_ORDER = {"algo_exec": ["B2-direct", "B2-CoT-short", "B2-CoT-med", "B2-CoT-long"],
             "rule_shift": ["B2-direct", "B2-CoT-med", "B2-CoT-long"]}


def load(csv_path: Path):
    """-> arms[(family, arm)] = {seed: {"t35": acc, "all": acc, "fpa35": float|None,
    "cotfail": float|None, "run_id": str}}"""
    arms: dict = defaultdict(dict)
    with csv_path.open(encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            exp, mid = r["exp_id"], r["model_id"]
            if (exp, mid) in REUSED:
                fam, arm = REUSED[(exp, mid)]
            elif exp in NEW_EXPS and r["family"] in ("algo_exec", "rule_shift"):
                fam, arm = NEW_EXPS[exp], mid
            else:
                continue
            if r["family"] != fam or "K1diag" in r["notes"]:
                continue
            seed = int(r["seed"])
            rec = arms[(fam, arm)].setdefault(
                seed, {"tiers": {}, "run_id": r["run_id"], "fpa35": None,
                       "cotfail": None})
            if rec["run_id"] != r["run_id"]:
                # keep the newest run per (arm, seed) — dedupe safety, should not
                # trigger given the append-only + dedupe-merge ledger discipline
                if r["run_id"] > rec["run_id"]:
                    rec.update({"tiers": {}, "run_id": r["run_id"]})
                else:
                    continue
            rec["tiers"][r["difficulty"]] = float(r["value"])
            m = re.search(r"fpa35=([\d.e+-]+)", r["notes"])
            if m:
                rec["fpa35"] = float(m.group(1))
            m = re.search(r"cotfail=([\d.]+)", r["notes"])
            if m:
                rec["cotfail"] = float(m.group(1))
    out = {}
    for key, seeds in arms.items():
        rows = {}
        for s, rec in seeds.items():
            if not all(t in rec["tiers"] for t in PRIMARY_TIERS):
                continue
            rows[s] = {
                "t35": 100 * sum(rec["tiers"][t] for t in PRIMARY_TIERS) / 3,
                "all": 100 * rec["tiers"].get("all", 0.0),
                "fpa35": rec["fpa35"] if rec["fpa35"] is not None
                else ANALYTIC_FPA35.get(key),
                "cotfail": rec["cotfail"], "run_id": rec["run_id"],
            }
        if rows:
            out[key] = rows
    return out


def agg(rows: dict, stat: str = "mean"):
    vals = [v["t35"] for v in rows.values()]
    acc = statistics.median(vals) if stat == "median" else statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    fpas = [v["fpa35"] for v in rows.values() if v["fpa35"]]
    return acc, sd, (statistics.mean(fpas) if fpas else None), len(vals)


def pooled_sd(a: dict, b: dict) -> float:
    va = [v["t35"] for v in a.values()]
    vb = [v["t35"] for v in b.values()]
    allv = [x - statistics.mean(va) for x in va] + [x - statistics.mean(vb) for x in vb]
    return statistics.stdev(allv) if len(allv) > 2 else 0.0


def gate(name: str, diff: float, psd: float, note: str = "") -> str:
    verdict = "PASS" if (diff >= MARGIN and diff > 2 * psd) else \
              ("NEAR-MISS (EXTEND policy)" if diff >= MARGIN else "no")
    return f"  {name}: delta={diff:+.1f} pts (2x pooled SD={2 * psd:.1f}) -> {verdict} {note}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("experiments/results.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/figs"))
    args = ap.parse_args()
    arms = load(args.csv)

    for fam in ("algo_exec", "rule_shift"):
        fam_arms = {a: r for (f, a), r in arms.items() if f == fam}
        if not fam_arms:
            continue
        print(f"\n=== {fam} (tier 3-5 mean accuracy, pts) ===")
        stat = "median" if fam == "rule_shift" else "mean"
        table = {}
        for a, rows in sorted(fam_arms.items()):
            acc, sd, fpa, n = agg(rows, "mean")
            med = statistics.median([v["t35"] for v in rows.values()])
            table[a] = (acc, sd, fpa, n, med)
            ff = [v["cotfail"] for v in rows.values() if v["cotfail"] is not None]
            ffs = f" cotfail={statistics.mean(ff):.2f}" if ff else ""
            print(f"  {a:16s} n={n} mean={acc:5.1f} median={med:5.1f} sd={sd:4.1f} "
                  f"fpa35={fpa:.3g}{ffs}" if fpa else
                  f"  {a:16s} n={n} mean={acc:5.1f} median={med:5.1f} sd={sd:4.1f}")

        # ---- gates ----
        print(f"-- gates ({'median' if stat == 'median' else 'mean'} primary) --")
        direct = fam_arms.get("B2-direct")
        cots = [a for a in fam_arms if a.startswith("B2-CoT")]
        if direct and cots:
            best = max(cots, key=lambda a: table[a][0])
            print(gate(f"G1 CoT pays ({best})",
                       table[best][0] - table["B2-direct"][0],
                       pooled_sd(fam_arms[best], direct)))
        if "B2-CoT-long" in fam_arms and "B2-filler-long" in fam_arms:
            print(gate("G2 content-vs-compute (CoT-long vs filler)",
                       table["B2-CoT-long"][0] - table["B2-filler-long"][0],
                       pooled_sd(fam_arms["B2-CoT-long"], fam_arms["B2-filler-long"])))
            if direct:
                print(gate("G2b filler vs direct",
                           table["B2-filler-long"][0] - table["B2-direct"][0],
                           pooled_sd(fam_arms["B2-filler-long"], direct)))
        if "B2-CoT-long" in fam_arms and "B2-wide" in fam_arms:
            d = table["B2-CoT-long"][0] - table["B2-wide"][0]
            psd = pooled_sd(fam_arms["B2-CoT-long"], fam_arms["B2-wide"])
            tie = "TIE" if abs(d) < MARGIN else ("CoT wins" if d > 0 else "params win")
            print(f"  G3 tokens-vs-params: delta={d:+.1f} (2xSD={2 * psd:.1f}) -> {tie}")
        if "V3-direct" in fam_arms and "B2-CoT-long" in fam_arms:
            v3 = (statistics.median([v['t35'] for v in fam_arms['V3-direct'].values()])
                  if stat == "median" else table["V3-direct"][0])
            cot_acc, _, cot_fpa, _, _ = table["B2-CoT-long"]
            v3_fpa = table["V3-direct"][2]
            ratio = v3_fpa / cot_fpa if (v3_fpa and cot_fpa) else float("nan")
            if v3 >= cot_acc - MARGIN and ratio <= 0.75:
                verdict = "V3 DOMINATES"
            elif cot_acc >= v3 + MARGIN:
                verdict = "CoT DOMINATES on accuracy (spends more)"
            else:
                verdict = "no dominance"
            print(f"  G4 latent-vs-visible: V3({stat})={v3:.1f} vs CoT-long="
                  f"{cot_acc:.1f}, FLOP ratio V3/CoT={ratio:.2f} -> {verdict}")

        # ceiling guard check
        top2 = sorted((t[0] for t in table.values()), reverse=True)[:2]
        if len(top2) == 2 and all(t > 95 for t in top2):
            print("  CEILING GUARD TRIGGERED: two arms > .95 — re-read on tier 5 only")

        # ---- plot ----
        fig, ax = plt.subplots(figsize=(7, 5))
        for a, rows in sorted(fam_arms.items()):
            acc, sd, fpa, n, med = table[a]
            if fpa is None:
                continue
            y = med if (stat == "median" and a == "V3-direct") else acc
            marker = "s" if "direct" in a else ("^" if "wide" in a else
                                                ("x" if "filler" in a else "o"))
            ax.errorbar(fpa, y, yerr=sd, marker=marker, ms=8, capsize=3,
                        label=f"{a} (n={n})")
            if a == "V3-direct" and fam == "rule_shift":
                for v in rows.values():   # show bimodality honestly
                    ax.plot(v["fpa35"] or fpa, v["t35"], ".", color="gray", alpha=.5)
        cot_line = [(table[a][2], table[a][0]) for a in COT_ORDER[fam]
                    if a in table and table[a][2]]
        if len(cot_line) > 1:
            xs, ys = zip(*cot_line)
            ax.plot(xs, ys, "--", color="tab:blue", alpha=.6, lw=1)
        ax.set_xlabel("inference FLOPs per answer (tier 3-5 mean)")
        ax.set_ylabel("tier 3-5 accuracy (pts)")
        ax.set_title(f"EXP-004 exchange rate — {fam}\n"
                     "(reused direct arms use analytic FLOPs; V3 rule_shift: "
                     "median marker, gray dots = seeds)")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out = args.out_dir / f"exp004_curves_{fam}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
