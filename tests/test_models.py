"""Model zoo shape/loss sanity + loop/delta behavior checks (CPU, tiny)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from models import ModelConfig, build_model
from models.zoo import n_params

TINY = dict(vocab_size=259, d_model=64, n_layers=4, n_heads=4, n_kv_heads=2,
            d_ff=128, max_seq_len=128)


def _cfg(**kw):
    return ModelConfig(**{**TINY, **kw})


@pytest.mark.parametrize("arch", ["gpt2", "tf_pp", "loop", "delta"])
def test_forward_and_loss(arch):
    torch.manual_seed(0)
    cfg = _cfg(arch=arch)
    model = build_model(cfg)
    idx = torch.randint(0, 259, (2, 32))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 32, 259)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()  # gradients must flow
    assert n_params(model) > 0


def test_loop_count_changes_output():
    torch.manual_seed(0)
    model = build_model(_cfg(arch="loop", loop_count=4))
    model.eval()
    idx = torch.randint(0, 259, (1, 16))
    with torch.no_grad():
        l1, _ = model(idx, loop_count=1)
        l4, _ = model(idx, loop_count=4)
    assert not torch.allclose(l1, l4), "loop count must change computation"


def test_loop_params_independent_of_loop_count():
    a = build_model(_cfg(arch="loop", loop_count=1))
    b = build_model(_cfg(arch="loop", loop_count=8))
    assert n_params(a) == n_params(b)


def test_deep_supervision_loss_differs():
    torch.manual_seed(0)
    idx = torch.randint(0, 259, (2, 17))
    inputs, targets = idx[:, :-1], idx[:, 1:]  # proper next-token shift
    m1 = build_model(_cfg(arch="loop", deep_supervision=False))
    m2 = build_model(_cfg(arch="loop", deep_supervision=True))
    m2.load_state_dict(m1.state_dict())
    m1.train(); m2.train()
    _, l_last = m1(inputs, targets=targets)
    _, l_deep = m2(inputs, targets=targets)
    assert torch.isfinite(l_deep) and not torch.isclose(l_last, l_deep)


def test_loop_randomize_samples_depths():
    torch.manual_seed(0)
    model = build_model(_cfg(arch="loop", loop_count=4, loop_randomize=True,
                             deep_supervision=True))
    idx = torch.randint(0, 259, (2, 17))
    inputs, targets = idx[:, :-1], idx[:, 1:]

    model.train()
    losses = []
    for _ in range(8):
        _, loss = model(inputs, targets=targets)
        assert torch.isfinite(loss)
        losses.append(loss)
    losses[-1].backward()  # gradients flow through sampled-depth unroll

    # eval must be deterministic at the configured depth (no sampling)
    model.eval()
    with torch.no_grad():
        a, _ = model(inputs)
        b, _ = model(inputs)
    assert torch.allclose(a, b), "eval forward must not sample loop counts"


def test_delta_state_carries_information():
    torch.manual_seed(0)
    from models.delta_memory import GatedDeltaLayer
    cfg = _cfg(arch="delta")
    layer = GatedDeltaLayer(cfg)
    x = torch.randn(1, 8, cfg.d_model)
    out1, S = layer(x)
    out2, _ = layer(x, state=S)   # warm state must change the output
    assert not torch.allclose(out1, out2)


def test_sliding_window_masks_far_context():
    torch.manual_seed(0)
    cfg = _cfg(arch="tf_pp", window=4)
    model = build_model(cfg)
    model.eval()
    idx = torch.randint(0, 259, (1, 32))
    idx2 = idx.clone()
    idx2[0, 0] = (idx2[0, 0] + 1) % 259  # perturb a token far outside the window
    with torch.no_grad():
        la, _ = model(idx)
        lb, _ = model(idx2)
    assert torch.allclose(la[0, -1], lb[0, -1], atol=1e-4), \
        "token outside every window must not affect the last position"
