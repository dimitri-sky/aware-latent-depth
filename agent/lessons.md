# Distilled Lessons

Transferable insights from every kill/revise/scale decision. Later hypotheses inherit
these; check before designing any new experiment.

- (build) Tied embeddings need GPT-style N(0,0.02) init; PyTorch defaults produced
  ~10x inflated initial loss (58 vs 5.6) and would have silently wasted every early
  run. Caught by the CPU smoke test before any GPU time was spent.
- (build) nn.MultiheadAttention's fused path is invisible to FlopCounterMode; explicit
  attention matmuls are required for honest FLOP cross-checks.
- (build) Template text (family-fixed instruction headers) makes raw n-gram overlap
  audits useless; audit rare n-grams / near-duplicate coverage instead.
- (EXP-000) Multi-task dilution: 4000 steps over 7 families (~570/family) starves
  every family below learnability at 3-6M params; the same model on one family
  reaches 21.5% (probe 8bee22). Design all tiny experiments with >= ~2500 focused
  steps per family; validity-gate discrimination now runs per-family.
- (EXP-000) Full-sequence LM loss on template-heavy prompts converges 4L and 8L to
  identical loss while both fail the task; supervise the answer suffix only.
- (hardware) Sustained 100% GPU duty cycle hard-crashed the local box (PSU transient
  trip). All local runs now use the 25% duty-cycle throttle + 250W power cap
  (nvidia-smi -pl 250, needs admin, resets on reboot).
