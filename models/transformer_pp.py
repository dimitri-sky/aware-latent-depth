"""B2: Transformer++ (RoPE, SwiGLU, RMSNorm, GQA). The reference opponent."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import RMSNorm, TFBlock, rope_cache


class TransformerPP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(TFBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight
        self._rope = None

    def _rope_cs(self, device):
        if self._rope is None or self._rope[0].device != device:
            self._rope = rope_cache(self.cfg.d_model // self.cfg.n_heads,
                                    self.cfg.max_seq_len, device)
        return self._rope

    def forward(self, idx, targets=None):
        x = self.tok(idx)
        rope_cs = self._rope_cs(idx.device)
        for blk in self.blocks:
            x = blk(x, rope_cs)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
        return logits, loss
