"""V2: Transformer++ with gated delta-rule fast-weight layers interleaved with
sliding-window attention (Gated DeltaNet hybrid, arXiv:2412.06464).

State per head: S in R^{d_k x d_v}, updated per token t:
    S_t = alpha_t * S_{t-1} + beta_t * (v_t - S_{t-1}^T-read k_t) k_t^T
    o_t = S_t^T q_t
alpha in (0,1) is the forget gate (rapid memory clearing), beta in (0,1) the write
strength (targeted delta update).

Implementation note: this is the straightforward chunk-sequential scan in pure
PyTorch (correct, slow). Fine for 10-50M local runs; swap in the chunked-parallel
kernel (flash-linear-attention) for cloud runs on Linux. Recorded in the H2 backlog
entry's hardware-compat note.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import RMSNorm, SwiGLU, TFBlock, rope_cache


class GatedDeltaLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.dk = cfg.d_k // cfg.n_heads
        self.dv = cfg.d_v // cfg.n_heads
        self.wq = nn.Linear(d, cfg.d_k, bias=False)
        self.wk = nn.Linear(d, cfg.d_k, bias=False)
        self.wv = nn.Linear(d, cfg.d_v, bias=False)
        self.w_alpha = nn.Linear(d, cfg.n_heads, bias=True)
        self.w_beta = nn.Linear(d, cfg.n_heads, bias=True)
        self.wo = nn.Linear(cfg.d_v, d, bias=False)
        nn.init.constant_(self.w_alpha.bias, 4.0)   # start with long retention
        nn.init.constant_(self.w_beta.bias, -1.0)   # start with gentle writes

    def forward(self, x, state=None):
        b, t, _ = x.shape
        h, dk, dv = self.n_heads, self.dk, self.dv
        q = F.normalize(self.wq(x).view(b, t, h, dk), dim=-1)
        k = F.normalize(self.wk(x).view(b, t, h, dk), dim=-1)
        v = self.wv(x).view(b, t, h, dv)
        alpha = torch.sigmoid(self.w_alpha(x))  # (b,t,h)
        beta = torch.sigmoid(self.w_beta(x))

        S = state if state is not None else x.new_zeros(b, h, dk, dv)
        outs = []
        for i in range(t):
            ki, qi, vi = k[:, i], q[:, i], v[:, i]          # (b,h,dk/dv)
            ai = alpha[:, i].unsqueeze(-1).unsqueeze(-1)     # (b,h,1,1)
            bi = beta[:, i].unsqueeze(-1)                    # (b,h,1)
            pred = torch.einsum("bhkv,bhk->bhv", S, ki)      # S^T k
            delta = (vi - pred) * bi                         # write strength
            S = ai * S + torch.einsum("bhk,bhv->bhkv", ki, delta)
            outs.append(torch.einsum("bhkv,bhk->bhv", S, qi))
        out = torch.stack(outs, dim=1).reshape(b, t, h * dv)
        return self.wo(out), S


class DeltaBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.mem = GatedDeltaLayer(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, rope_cs=None):
        y, _ = self.mem(self.n1(x))
        x = x + y
        return x + self.mlp(self.n2(x))


class DeltaLM(nn.Module):
    """Interleaved: every `delta_every`-th block is a delta-memory block, the rest are
    sliding-window Transformer++ blocks (window from cfg; defaults to 128 if unset)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.window is None:
            cfg.window = 128
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        blocks = []
        for i in range(cfg.n_layers):
            if i % cfg.delta_every == 0:
                blocks.append(DeltaBlock(cfg))
            else:
                blocks.append(TFBlock(cfg))
        self.blocks = nn.ModuleList(blocks)
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
