"""EXP-004 FLOP-matching lock: expected inference FLOPs per ANSWER for every arm,
computed from the real eval distribution (same generator seeds/protocol as the
harness: eval seeds from 2,000,000, 400/family, first 200 used, tiers 3-5 primary).

Selects the B2-wide width whose direct FLOPs/answer matches B2-CoT-long on
algo_exec within +-10%. The printed table goes verbatim into agent/log/EXP-004.md
BEFORE any training (pre-registration lock).

    python scripts/exp004_flopmatch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import ModelConfig, build_model  # noqa: E402
from models.zoo import n_params  # noqa: E402
from sage.flops.accounting import generation_flops  # noqa: E402
from sage.generators import FAMILIES  # noqa: E402
from sage.generators.base import DIFFICULTIES, EVAL_SEED_LO  # noqa: E402
from train.data import trace_text  # noqa: E402
from train.tokenizer import encode  # noqa: E402

EVAL_PER_FAMILY = 400
EVAL_LIMIT = 200          # matches TrainConfig.eval_limit in train_single.py
PRIMARY_TIERS = (3, 4, 5)

B2 = dict(arch="tf_pp", d_model=512, n_heads=8, n_kv_heads=4, d_ff=1408,
          n_layers=6, max_seq_len=1024)
V3 = dict(arch="delta", d_model=512, n_heads=8, n_kv_heads=4, d_ff=1408,
          n_layers=6, delta_every=2, window=128, d_k=384, d_v=384, max_seq_len=1024)


def eval_records(fam: str) -> list[dict]:
    """Mirror make_data + harness: seeds from EVAL_SEED_LO, difficulty cycling,
    skip degenerate draws, first EVAL_LIMIT records."""
    gen = FAMILIES[fam]
    recs, i = [], 0
    while len(recs) < EVAL_PER_FAMILY:
        seed = EVAL_SEED_LO + i
        i += 1
        d = DIFFICULTIES[len(recs) % len(DIFFICULTIES)]
        try:
            inst = gen(seed, d)
        except Exception:
            continue
        recs.append(inst.to_dict())
    return recs[:EVAL_LIMIT]


def arm_flops(cfg: dict, recs: list[dict], trace_level: str | None,
              tiers=PRIMARY_TIERS) -> tuple[float, float, float]:
    """(mean FLOPs/answer on `tiers`, mean over all tiers, mean gen tokens on tiers).

    Convention identical to eval/harness.py: prompt includes BOS; gen_len counts the
    emitted suffix tokens (EOS-producing forward not charged) — every CoT token is a
    full forward pass at its context length.
    """
    fcfg = ModelConfig(**cfg).flops_cfg()
    per_tier: dict[int, list[float]] = {}
    gen_lens: list[int] = []
    for r in recs:
        if trace_level is None:
            prompt_len = 1 + len(encode(r["prompt"]))
            gen_len = len(encode(" " + r["answer"]))
        else:
            prefix = r["prompt"][: r["prompt"].rindex("ANSWER:")]
            prompt_len = 1 + len(encode(prefix))
            sup = "THINK:\n" + trace_text(r, trace_level) + "\nANSWER: " + r["answer"]
            gen_len = len(encode(sup))
        f = generation_flops(fcfg, prompt_len=prompt_len, gen_len=gen_len)
        per_tier.setdefault(r["difficulty"], []).append(f)
        if r["difficulty"] in tiers:
            gen_lens.append(gen_len)
    primary = [f for d in tiers for f in per_tier.get(d, [])]
    allt = [f for fs in per_tier.values() for f in fs]
    return (sum(primary) / len(primary), sum(allt) / len(allt),
            sum(gen_lens) / len(gen_lens))


def wide_candidates() -> list[dict]:
    """6L tf_pp widths; d_model % 16 == 0 keeps head_dim even for RoPE;
    d_ff = 2.75x d_model rounded to /16 (the B2 ratio)."""
    out = []
    for d in range(576, 897, 32):
        d_ff = int(round(2.75 * d / 16) * 16)
        out.append(dict(B2, d_model=d, d_ff=d_ff))
    return out


def main() -> None:
    recs = {fam: eval_records(fam) for fam in ("algo_exec", "rule_shift")}

    arms: list[tuple[str, dict, str | None]] = [
        ("B2 direct", B2, None),
        ("V3 direct", V3, None),
        ("B2-CoT-short", B2, "short"),      # algo_exec only
        ("B2-CoT-med", B2, "med"),
        ("B2-CoT-long", B2, "long"),
        ("B2-filler-long", B2, "filler"),
    ]

    print("=== expected inference FLOPs per answer (analytic, eval distribution) ===")
    print(f"{'arm':22s} {'family':12s} {'tier3-5':>10s} {'all-tier':>10s} "
          f"{'x B2':>6s} {'gen@3-5':>8s}")
    ref: dict[str, float] = {}
    target: dict[str, float] = {}
    for name, cfg, lvl in arms:
        for fam in recs:
            if lvl == "short" and fam == "rule_shift":
                continue        # no short tier for rule_shift (design change #4)
            f35, fall, glen = arm_flops(cfg, recs[fam], lvl)
            if name == "B2 direct":
                ref[fam] = f35
            if name == "B2-CoT-long":
                target[fam] = f35
            print(f"{name:22s} {fam:12s} {f35:10.3g} {fall:10.3g} "
                  f"{f35 / ref[fam]:6.2f} {glen:8.1f}")

    print("\n=== B2-wide selection: match B2-CoT-long on algo_exec within +-10% ===")
    best, best_err = None, 1e9
    for cand in wide_candidates():
        f35, _, _ = arm_flops(cand, recs["algo_exec"], None)
        err = abs(f35 - target["algo_exec"]) / target["algo_exec"]
        p = n_params(build_model(ModelConfig(**cand)))
        print(f"  d_model={cand['d_model']:4d} d_ff={cand['d_ff']:4d} "
              f"params={p / 1e6:6.2f}M  fpa(3-5)={f35:.3g}  err={err * 100:5.1f}%")
        if err < best_err:
            best, best_err = (cand, p), err

    cand, p = best
    f35_ax, fall_ax, _ = arm_flops(cand, recs["algo_exec"], None)
    f35_rs, fall_rs, _ = arm_flops(cand, recs["rule_shift"], None)
    status = "OK (within +-10%)" if best_err <= 0.10 else "FAIL (outside +-10%)"
    print(f"\nLOCKED B2-wide: d_model={cand['d_model']} d_ff={cand['d_ff']} "
          f"n_layers=6 n_heads=8 n_kv_heads=4  params={p / 1e6:.2f}M")
    print(f"  algo_exec  fpa(3-5)={f35_ax:.3g} vs target {target['algo_exec']:.3g} "
          f"-> err {best_err * 100:.1f}%  [{status}]")
    print(f"  rule_shift fpa(3-5)={f35_rs:.3g} (= {f35_rs / target['rule_shift']:.2f}x "
          f"that family's CoT-long; single shared config, logged design change #2)")


if __name__ == "__main__":
    main()
