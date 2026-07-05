"""V1: shared-weight recurrent-depth LM (prelude -> looped core -> coda).

Design follows the simplest defensible form of the recurrent-depth hypothesis
(Huginn arXiv:2502.05171, Ouro arXiv:2510.25741, looped-TF arXiv:2502.17416):
- prelude blocks encode the input once,
- a small core block stack is applied `loop_count` times with shared weights,
- input injection adds the prelude output at each loop entry (stabilizes iteration),
- truncated BPTT: gradients flow through only the last `bptt_loops` iterations,
- optional deep supervision: logits at every loop step (V3; off for V1),
- `loop_count` can be overridden at inference for test-time compute scaling.

EXP-003B revision (readout blind spot, arXiv:2606.24898 — EXP-003 produced a
loop-invariant model, K-gap < 1 pt):
- loop-index conditioning: a learned per-step embedding is added at each loop
  entry so iterations are functionally distinguishable (loop_step_embed=True),
- auxiliary (non-final) per-loop losses run through a DETACHED hidden state so
  early-step CE cannot pull the core toward a fixed point; only the final step
  backprops through the loop chain,
- per-step losses are linearly weighted (step k gets weight k/K) instead of
  equal-weight mean, making later iterations strictly more valuable.

Halting is deliberately absent (H4 is tested later as an efficiency add-on).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import RMSNorm, TFBlock, init_weights, rope_cache


class LoopLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.prelude = nn.ModuleList(TFBlock(cfg) for _ in range(cfg.n_prelude))
        self.core = nn.ModuleList(TFBlock(cfg) for _ in range(cfg.n_core))
        self.coda = nn.ModuleList(TFBlock(cfg) for _ in range(cfg.n_coda))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.loop_step_embed:
            # one embedding per possible loop step (randomized K can reach 2x)
            self.step_embed = nn.Embedding(2 * cfg.loop_count, cfg.d_model)
        init_weights(self)
        if cfg.loop_step_embed:
            nn.init.normal_(self.step_embed.weight, std=0.02)
        self.head.weight = self.tok.weight
        self._rope = None

    def _rope_cs(self, device, t: int):
        if self._rope is None or self._rope[0].device != device or self._rope[0].shape[0] < t:
            self._rope = rope_cache(self.cfg.d_model // self.cfg.n_heads,
                                    max(self.cfg.max_seq_len, t), device)
        return self._rope

    def forward(self, idx, targets=None, loop_count: int | None = None):
        cfg = self.cfg
        if loop_count is not None:
            loops = loop_count
        elif self.training and cfg.loop_randomize:
            # clipped Poisson, mean = loop_count (RD-VLA-style depth randomization;
            # expected training FLOPs equal the fixed-depth model's)
            k = int(torch.poisson(torch.tensor(float(cfg.loop_count))).item())
            loops = max(1, min(2 * cfg.loop_count, k))
        else:
            loops = cfg.loop_count
        rope_cs = self._rope_cs(idx.device, idx.shape[1])

        x = self.tok(idx)
        for blk in self.prelude:
            x = blk(x, rope_cs)
        h0 = x

        step_logits = []
        h = h0
        for step in range(loops):
            # truncated BPTT: detach the state entering early iterations
            if self.training and step < loops - cfg.bptt_loops:
                h = h.detach()
            if cfg.input_injection:
                h = h + h0
            if cfg.loop_step_embed:
                h = h + self.step_embed.weight[min(step, self.step_embed.num_embeddings - 1)]
            for blk in self.core:
                h = blk(h, rope_cs)
            if cfg.deep_supervision and targets is not None:
                # auxiliary (non-final) readouts see a DETACHED state: they train
                # the coda/head to decode intermediate states without pulling the
                # core toward a step-invariant fixed point (readout blind spot fix)
                step_logits.append(self._decode(h if step == loops - 1 else h.detach(),
                                                rope_cs))

        logits = step_logits[-1] if step_logits else self._decode(h, rope_cs)

        loss = None
        if targets is not None:
            if cfg.deep_supervision and step_logits:
                # linearly increasing weights: step k of K gets weight (k+1)/K,
                # so later iterations are strictly more valuable than earlier ones
                losses = [F.cross_entropy(lg.view(-1, lg.size(-1)),
                                          targets.reshape(-1), ignore_index=-100)
                          for lg in step_logits]
                k = len(losses)
                w = torch.arange(1, k + 1, device=losses[0].device, dtype=losses[0].dtype)
                w = w / w.sum()
                loss = (torch.stack(losses) * w).sum()
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def _decode(self, h, rope_cs):
        x = h
        for blk in self.coda:
            x = blk(x, rope_cs)
        return self.head(self.norm(x))
