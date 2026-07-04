"""Analytic FLOP accounting (MAC = 2 FLOPs), per docs/BASELINES.md.

Works off a plain config dict so it stays independent of model code:
  arch: gpt2 | tf_pp | loop | delta
  d_model, n_layers, n_heads, n_kv_heads, d_ff, vocab_size
  loop: n_prelude, n_core, n_coda, loop_count  (n_layers ignored)
  delta: delta_every (every k-th layer is a gated-delta layer), d_k, d_v
  window: sliding-window size (None = full causal)

Cross-checked against torch.utils.flop_counter in tests/test_flops.py (<=10% error
required; the analytic count includes matmuls only, as does the profiler's default).
"""
from __future__ import annotations


def _attn_layer_flops(cfg: dict, ctx: int) -> float:
    d = cfg["d_model"]
    n_heads = cfg["n_heads"]
    n_kv = cfg.get("n_kv_heads", n_heads)
    t_eff = min(ctx, cfg["window"]) if cfg.get("window") else ctx
    proj = 2 * d * d * (1 + 1)          # Q and output projections
    proj += 2 * d * d * (n_kv / n_heads) * 2  # K and V (GQA-adjusted)
    scores = 4 * d * t_eff              # QK^T and AV, both 2*d*t_eff
    return proj + scores


def _mlp_layer_flops(cfg: dict) -> float:
    d, d_ff = cfg["d_model"], cfg["d_ff"]
    if cfg.get("mlp", "swiglu") == "gelu":   # GPT-2 style: two matmuls
        return 2 * d * d_ff * 2
    return 3 * (2 * d * d_ff)                # SwiGLU: gate, up, down


def _delta_layer_flops(cfg: dict) -> float:
    """Gated delta rule per token: q/k/v/alpha/beta projections + state update/readout.

    State S is (d_k/h) x (d_v/h) per head. Three matmul-like ops per token per head
    (prediction S^T k, outer-product write, readout S^T q), each 2*(d_k/h)*(d_v/h)*h
    = 2*d_k*d_v/h FLOPs. Elementwise decay/gating excluded per the matmul-only
    convention (<1%). See arXiv:2412.06464.
    """
    d, h = cfg["d_model"], cfg["n_heads"]
    d_k = cfg.get("d_k", d)
    d_v = cfg.get("d_v", d)
    proj = 2 * d * (d_k + d_k + d_v + 2 * h)  # q, k, v, alpha, beta heads
    proj += 2 * d_v * d                       # output projection
    state = 3 * (2 * d_k * d_v / h)           # predict, write, readout
    return proj + state


def _block_flops(cfg: dict, ctx: int, layer_idx: int) -> float:
    if cfg["arch"] == "delta" and layer_idx % cfg.get("delta_every", 2) == 0:
        return _delta_layer_flops(cfg) + _mlp_layer_flops(cfg)
    return _attn_layer_flops(cfg, ctx) + _mlp_layer_flops(cfg)


def model_flops_per_token(cfg: dict, ctx: int, loop_count: int | None = None) -> float:
    """Forward FLOPs for one token at context length ctx."""
    d, vocab = cfg["d_model"], cfg["vocab_size"]
    head = 2 * d * vocab
    emb = 0  # lookup, negligible

    if cfg["arch"] == "loop":
        loops = loop_count if loop_count is not None else cfg["loop_count"]
        n_layers_effective = (cfg["n_prelude"] + cfg["n_core"] * loops + cfg["n_coda"])
        body = sum(_block_flops(cfg, ctx, i) for i in range(n_layers_effective))
    else:
        body = sum(_block_flops(cfg, ctx, i) for i in range(cfg["n_layers"]))
    return emb + body + head


def training_flops(cfg: dict, n_tokens: int, avg_ctx: int, loop_count: int | None = None) -> float:
    """Total training FLOPs: backward counted as 2x forward (standard convention)."""
    return 3.0 * model_flops_per_token(cfg, avg_ctx, loop_count) * n_tokens


def generation_flops(cfg: dict, prompt_len: int, gen_len: int,
                     loop_count: int | None = None) -> float:
    """Inference FLOPs: prefill over the prompt + one full forward per generated token
    at growing context. Every CoT token is charged a full forward pass (never free)."""
    total = prompt_len * model_flops_per_token(cfg, prompt_len, loop_count)
    for i in range(gen_len):
        total += model_flops_per_token(cfg, prompt_len + i, loop_count)
    return total


def params_estimate(cfg: dict) -> float:
    """Approximate parameter count (embeddings tied with LM head)."""
    d, vocab, d_ff = cfg["d_model"], cfg["vocab_size"], cfg["d_ff"]
    n_heads = cfg["n_heads"]
    n_kv = cfg.get("n_kv_heads", n_heads)

    attn = d * d * (2 + 2 * n_kv / n_heads)
    mlp = 2 * d * d_ff if cfg.get("mlp", "swiglu") == "gelu" else 3 * d * d_ff
    delta = d * (cfg.get("d_k", d) * 2 + cfg.get("d_v", d) + 2) + cfg.get("d_v", d) * d

    def block(i: int) -> float:
        if cfg["arch"] == "delta" and i % cfg.get("delta_every", 2) == 0:
            return delta + mlp
        return attn + mlp

    if cfg["arch"] == "loop":
        n_unique = cfg["n_prelude"] + cfg["n_core"] + cfg["n_coda"]
        body = sum(block(i) for i in range(n_unique))
    else:
        body = sum(block(i) for i in range(cfg["n_layers"]))
    return body + vocab * d
