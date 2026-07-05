from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    arch: str = "tf_pp"          # gpt2 | tf_pp | loop | delta
    vocab_size: int = 259        # byte-level: 256 bytes + BOS/EOS/PAD
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    n_kv_heads: int = 2          # GQA (ignored by gpt2)
    d_ff: int = 1024             # SwiGLU hidden (gpt2 uses 4*d_model regardless)
    max_seq_len: int = 1024
    dropout: float = 0.0
    window: int | None = None    # sliding-window attention (None = full causal)

    # loop arch
    n_prelude: int = 1
    n_core: int = 2
    n_coda: int = 1
    loop_count: int = 4
    bptt_loops: int = 2          # backprop through only the last k loops
    input_injection: bool = True
    deep_supervision: bool = False
    # H3 recipe (2026 lit scan): sample training loop count from a clipped Poisson
    # with mean loop_count (min 1, max 2*loop_count). Expected FLOPs unchanged.
    loop_randomize: bool = False
    # EXP-003B fix (readout blind spot): learned per-loop-step embedding added at
    # each loop entry so iterations are functionally distinguishable.
    loop_step_embed: bool = False

    # delta arch
    delta_every: int = 2         # every k-th layer is a gated-delta layer
    d_k: int | None = None       # default: d_model
    d_v: int | None = None

    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.d_k is None:
            self.d_k = self.d_model
        if self.d_v is None:
            self.d_v = self.d_model

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]

    def flops_cfg(self) -> dict:
        """View consumed by sage.flops.accounting."""
        d = self.to_dict()
        d["mlp"] = "gelu" if self.arch == "gpt2" else "swiglu"
        if self.arch == "gpt2":
            d["d_ff"] = 4 * self.d_model
            d["n_kv_heads"] = self.n_heads
        return d
