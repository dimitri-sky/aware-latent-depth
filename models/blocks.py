"""Shared building blocks: RMSNorm, RoPE, GQA attention (optional sliding window),
SwiGLU. Used by Transformer++, loop, and delta variants."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module: nn.Module) -> None:
    """GPT-style init: N(0, 0.02) for Linear/Embedding. Essential with tied
    embeddings, whose default N(0,1) init produces ~10x inflated initial loss."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(x.dtype)


def rope_cache(head_dim: int, max_len: int, device, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, H, T, Dh)
    t = x.shape[2]
    cos, sin = cos[:t].to(x.dtype), sin[:t].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def sliding_window_mask(t: int, window: int, device) -> torch.Tensor:
    i = torch.arange(t, device=device)
    causal = i[:, None] >= i[None, :]
    near = (i[:, None] - i[None, :]) < window
    return causal & near


class Attention(nn.Module):
    """Causal attention with RoPE and GQA; optional sliding window."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 window: int | None = None, rope: bool = True):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv = n_heads, n_kv_heads
        self.dh = d_model // n_heads
        self.window, self.rope = window, rope
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, self.n_kv * self.dh, bias=False)
        self.wv = nn.Linear(d_model, self.n_kv * self.dh, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, rope_cs=None):
        b, t, d = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.dh).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv, self.dh).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv, self.dh).transpose(1, 2)
        if self.rope and rope_cs is not None:
            q, k = apply_rope(q, *rope_cs), apply_rope(k, *rope_cs)
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        if self.window is not None and t > self.window:
            mask = sliding_window_mask(t, self.window, x.device)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).reshape(b, t, d))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TFBlock(nn.Module):
    """Pre-norm Transformer++ block."""

    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.window)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, rope_cs=None):
        x = x + self.attn(self.n1(x), rope_cs)
        return x + self.mlp(self.n2(x))
