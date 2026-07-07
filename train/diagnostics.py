"""Opt-in early-training diagnostics for the grokking side-study (EXP-009).

Design constraints (pre-registered in agent/log/EXP-009.md):
- NEVER influences training: no lr changes, no early stopping, no model selection.
  Probe instances are pre-generated at init (no RNG draws during training), decoding
  is greedy (no torch RNG), so the training trajectory is identical to an
  uninstrumented run up to hardware nondeterminism.
- Mini-eval uses a PROBE set drawn from unused train-range seeds (900k+): disjoint
  from the training files (seeds 0..~20k) AND from the eval range (2M+), so the
  final eval split is never touched during training.
- Output is a JSONL sidecar per run under checkpoints/diag/ — results.csv stays an
  eval-rows-only ledger.

Cadence: light records (loss, curvature, grad norm) every 25 steps for the first
1000 steps (the pre-registered predictor window), every 100 after; heavy records
(per-tier mini-eval, delta-state stats, weight norms) every 200 steps; checkpoints
every 1000 steps.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from eval.harness import greedy_generate
from sage.generators import FAMILIES
from sage.generators.base import DIFFICULTIES, TRAIN_SEED_HI
from sage.scoring import score_output
from train.tokenizer import BOS, PAD, encode

PROBE_SEED_BASE = 900_000        # inside train range, outside any training file
PROBE_PER_TIER = 10
LIGHT_EVERY_EARLY, EARLY_UNTIL, LIGHT_EVERY_LATE = 25, 1000, 100
HEAVY_EVERY = 200
CKPT_EVERY = 1000


def _effective_rank(S: torch.Tensor) -> float:
    """exp(entropy of normalized squared singular values), averaged over batch x
    heads. S: (b, h, d_k, d_v)."""
    b, h, dk, dv = S.shape
    flat = S.reshape(b * h, dk, dv).float()
    sv = torch.linalg.svdvals(flat)                      # (b*h, min(dk,dv))
    p = sv.pow(2)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    ent = -(p * p.clamp_min(1e-12).log()).sum(dim=-1)
    return float(ent.exp().mean())


class DiagRecorder:
    def __init__(self, exp_id: str, model_id: str, seed: int, families: list[str],
                 device: str, total_steps: int, out_dir: Path = Path("checkpoints/diag")):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / f"{exp_id}-{model_id}-s{seed}.jsonl"
        self.path.write_text("", encoding="utf-8")       # fresh sidecar per run
        self.ckpt_stem = out_dir / f"{exp_id}-{model_id}-s{seed}"
        self.device = device
        self.total_steps = total_steps
        self.t0 = time.time()

        # Probe instances: PROBE_PER_TIER per difficulty tier per family,
        # pre-generated NOW so training consumes zero extra RNG draws.
        self.probe = []
        for fam in families:
            gen = FAMILIES[fam]
            i = 0
            for d in DIFFICULTIES:
                n = 0
                while n < PROBE_PER_TIER:
                    s = PROBE_SEED_BASE + i
                    i += 1
                    assert s < TRAIN_SEED_HI
                    try:
                        inst = gen(s, d)
                    except Exception:
                        continue
                    self.probe.append(inst.to_dict())
                    n += 1
        # Fixed forward batch for delta-state statistics (first 8 probe prompts).
        ids = [[BOS] + encode(r["prompt"]) for r in self.probe[:8]]
        width = max(len(p) for p in ids)
        xb = torch.full((len(ids), width), PAD, dtype=torch.long)
        for r, p in enumerate(ids):
            xb[r, : len(p)] = torch.tensor(p, dtype=torch.long)
        self.state_batch = xb

        self._loss_win: list[float] = []
        self._gn_win: list[float] = []
        self._smooth: list[tuple[int, float]] = []       # (step, windowed mean loss)

    def _write(self, rec: dict) -> None:
        rec["t"] = round(time.time() - self.t0, 1)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    # ---- light path: every training step -----------------------------------
    def log_step(self, step: int, loss: float, grad_norm: float) -> None:
        self._loss_win.append(loss)
        self._gn_win.append(grad_norm)
        every = LIGHT_EVERY_EARLY if step <= EARLY_UNTIL else LIGHT_EVERY_LATE
        if step % every != 0 and step != self.total_steps:
            return
        mean_loss = sum(self._loss_win) / max(1, len(self._loss_win))
        self._smooth.append((step, mean_loss))
        curv = None
        if len(self._smooth) >= 3:
            (_, a), (_, b), (_, c) = self._smooth[-3], self._smooth[-2], self._smooth[-1]
            curv = c - 2 * b + a                          # 2nd difference of smoothed loss
        self._write({"kind": "light", "step": step,
                     "loss": round(mean_loss, 6),
                     "loss_last": round(loss, 6),
                     "curvature": None if curv is None else round(curv, 6),
                     "grad_norm_mean": round(sum(self._gn_win) / len(self._gn_win), 4),
                     "grad_norm_max": round(max(self._gn_win), 4)})
        self._loss_win, self._gn_win = [], []

    # ---- heavy path: mini-eval + state stats + weight norms ----------------
    @torch.no_grad()
    def heavy(self, step: int, model, mcfg) -> None:
        if step % HEAVY_EVERY != 0 and step != self.total_steps:
            return
        rec: dict = {"kind": "heavy", "step": step}

        # per-tier mini-eval on the probe set (greedy, direct form)
        prompts = [[BOS] + encode(r["prompt"]) for r in self.probe]
        gens = greedy_generate(model, prompts, self.device, max_new=16)
        tiers: dict[int, list[int]] = {}
        from train.tokenizer import decode  # local import avoids cycle at module load
        for r, gen in zip(self.probe, gens):
            ok = score_output(decode(gen), r["answer"], r["scoring"])
            t = tiers.setdefault(r["difficulty"], [0, 0])
            t[0] += 1
            t[1] += int(ok)
        rec["probe_acc"] = {str(d): round(c / max(1, n), 3)
                            for d, (n, c) in sorted(tiers.items())}
        rec["probe_acc_all"] = round(sum(c for _, c in tiers.values())
                                     / max(1, sum(n for n, _ in tiers.values())), 4)

        # weight norms (global + delta-vs-attention split)
        total_sq = delta_sq = attn_sq = 0.0
        for name, p in model.named_parameters():
            sq = float(p.detach().float().pow(2).sum())
            total_sq += sq
            if ".mem." in name:
                delta_sq += sq
            elif ".attn." in name:
                attn_sq += sq
        rec["w_norm"] = round(total_sq ** 0.5, 3)
        rec["w_norm_delta"] = round(delta_sq ** 0.5, 3)
        rec["w_norm_attn"] = round(attn_sq ** 0.5, 3)

        # delta-state statistics on the fixed probe batch
        if mcfg.arch == "delta":
            from models.delta_memory import GatedDeltaLayer
            captured: list[dict] = []
            hooks = []
            for name, mod in model.named_modules():
                if isinstance(mod, GatedDeltaLayer):
                    def make_hook(nm, m_ref):
                        def hook(module, args, output):
                            _, S = output
                            alpha = torch.sigmoid(m_ref.w_alpha(args[0]))
                            beta = torch.sigmoid(m_ref.w_beta(args[0]))
                            captured.append({
                                "layer": nm,
                                "s_fro": round(float(S.float().norm(dim=(-2, -1)).mean()), 4),
                                "s_erank": round(_effective_rank(S), 3),
                                "alpha_mean": round(float(alpha.mean()), 4),
                                "beta_mean": round(float(beta.mean()), 4),
                            })
                        return hook
                    hooks.append(mod.register_forward_hook(make_hook(name, mod)))
            model(self.state_batch.to(self.device))
            for h in hooks:
                h.remove()
            rec["delta_state"] = captured

        model.train()
        self._write(rec)

    def maybe_ckpt(self, step: int, model, mcfg) -> None:
        if step % CKPT_EVERY != 0:
            return
        torch.save({"model": model.state_dict(), "config": mcfg.to_dict()},
                   f"{self.ckpt_stem}-step{step}.pt")
