"""Analytic FLOP counts must agree with torch's FlopCounterMode within 10%
(docs/BASELINES.md).

The comparison forces the MATH sdpa backend (the flash-CPU kernel is invisible to
FlopCounterMode) and uses the full-quadratic analytic attention term, because the
profiler counts the full T x T matmuls without a causal discount. Production
accounting uses the causal-averaged context; only the cross-check uses full T.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from models import ModelConfig, build_model
from models.zoo import n_params
from sage.flops.accounting import generation_flops, model_flops_per_token, params_estimate

TINY = dict(vocab_size=259, d_model=128, n_layers=4, n_heads=4, n_kv_heads=2,
            d_ff=256, max_seq_len=256)


def _measured_forward_flops(model, ctx: int) -> float:
    idx = torch.randint(0, 259, (1, ctx))
    model.eval()
    with sdpa_kernel(SDPBackend.MATH):
        with FlopCounterMode(display=False) as fc:
            with torch.no_grad():
                model(idx)
    return fc.get_total_flops()


@pytest.mark.parametrize("arch", ["gpt2", "tf_pp", "loop", "delta"])
def test_analytic_vs_profiler(arch):
    torch.manual_seed(0)
    cfg = ModelConfig(arch=arch, **TINY)
    model = build_model(cfg)
    ctx = 128
    measured = _measured_forward_flops(model, ctx)
    # full-T attention term to mirror the profiler (see module docstring)
    analytic = model_flops_per_token(cfg.flops_cfg(), ctx) * ctx
    err = abs(measured - analytic) / measured
    assert err < 0.10, f"{arch}: analytic {analytic:.3e} vs measured {measured:.3e} ({err:.1%})"


def test_loop_flops_linear_in_loop_count():
    """Exact invariant: each extra loop adds exactly the same core cost."""
    cfg = ModelConfig(arch="loop", **TINY)
    f = [model_flops_per_token(cfg.flops_cfg(), 128, loop_count=k) for k in (1, 2, 4)]
    per_loop = f[1] - f[0]
    assert per_loop > 0
    assert abs((f[2] - f[0]) - 3 * per_loop) < 1e-6 * f[2]


def test_cot_tokens_are_charged():
    """Exact invariant: each generated token costs at least one full forward pass."""
    cfg = ModelConfig(arch="tf_pp", **TINY)
    short = generation_flops(cfg.flops_cfg(), prompt_len=100, gen_len=4)
    cot = generation_flops(cfg.flops_cfg(), prompt_len=100, gen_len=64)
    per_tok_min = model_flops_per_token(cfg.flops_cfg(), 104)
    assert cot - short >= 60 * per_tok_min


@pytest.mark.parametrize("arch", ["tf_pp", "loop", "delta"])
def test_param_estimate_close(arch):
    cfg = ModelConfig(arch=arch, **TINY)
    est = params_estimate(cfg.flops_cfg())
    actual = n_params(build_model(cfg))
    assert abs(est - actual) / actual < 0.05, f"{arch}: est {est} vs actual {actual}"
