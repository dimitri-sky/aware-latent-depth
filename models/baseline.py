"""B1: GPT-2-style baseline (LayerNorm, GELU, learned positional embeddings, full MHA).

Attention is implemented explicitly (not nn.MultiheadAttention) so the FLOP profiler
sees every matmul; the fused fast path is invisible to FlopCounterMode.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Attention


class Gpt2Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.n1 = nn.LayerNorm(d)
        self.attn = Attention(d, cfg.n_heads, cfg.n_heads, window=None, rope=False)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class Gpt2LM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Gpt2Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tied

    def forward(self, idx, targets=None):
        b, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
        return logits, loss
