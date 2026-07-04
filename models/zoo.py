from __future__ import annotations

from .baseline import Gpt2LM
from .config import ModelConfig
from .delta_memory import DeltaLM
from .loop_core import LoopLM
from .transformer_pp import TransformerPP

_ARCHS = {"gpt2": Gpt2LM, "tf_pp": TransformerPP, "loop": LoopLM, "delta": DeltaLM}


def build_model(cfg: ModelConfig):
    if cfg.arch not in _ARCHS:
        raise ValueError(f"unknown arch {cfg.arch}")
    return _ARCHS[cfg.arch](cfg)


def n_params(model) -> int:
    # tied head/embedding counted once (PyTorch shares the tensor, so this is exact)
    return sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
