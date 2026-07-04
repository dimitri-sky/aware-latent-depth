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
