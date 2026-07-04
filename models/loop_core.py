"""V1: shared-weight recurrent-depth LM (prelude -> looped core -> coda).

Design follows the simplest defensible form of the recurrent-depth hypothesis
(Huginn arXiv:2502.05171, Ouro arXiv:2510.25741, looped-TF arXiv:2502.17416):
- prelude blocks encode the input once,
- a small core block stack is applied `loop_count` times with shared weights,
- input injection adds the prelude output at each loop entry (stabilizes iteration),
- truncated BPTT: gradients flow through only the last `bptt_loops` iterations,
- optional deep supervision: logits at every loop step (V3; off for V1),
- `loop_count` can be overridden at inference for test-time compute scaling.

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
        init_weights(self)
        self.head.weight = self.tok.weight
        self._rope = None

    def _rope_cs(self, device):
        if self._rope is None or self._rope[0].device != device:
            self._rope = rope_cache(self.cfg.d_model // self.cfg.n_heads,
                                    self.cfg.max_seq_len, device)
        return self._rope

    def forward(self, idx, targets=None, loop_count: int | None = None):
        cfg = self.cfg
        loops = loop_count if loop_count is not None else cfg.loop_count
        rope_cs = self._rope_cs(idx.device)

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
            for blk in self.core:
                h = blk(h, rope_cs)
            if cfg.deep_supervision and targets is not None:
                step_logits.append(self._decode(h, rope_cs))

        logits = step_logits[-1] if step_logits else self._decode(h, rope_cs)

        loss = None
        if targets is not None:
            if cfg.deep_supervision and step_logits:
                # equal-weight CE over loop steps (TRM-style deep supervision)
                losses = [F.cross_entropy(lg.view(-1, lg.size(-1)),
                                          targets.reshape(-1), ignore_index=-100)
                          for lg in step_logits]
                loss = torch.stack(losses).mean()
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def _decode(self, h, rope_cs):
        x = h
        for blk in self.coda:
            x = blk(x, rope_cs)
        return self.head(self.norm(x))
