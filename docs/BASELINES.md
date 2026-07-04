# Baselines, Fairness Rules, and FLOP Accounting

## Baselines

| ID | Model | Purpose |
|----|-------|---------|
| B1 | GPT-2-style transformer (LayerNorm, GELU, learned pos-emb, MHA) | historical reference |
| B2 | Transformer++ (RoPE, SwiGLU, RMSNorm, GQA) | **the reference opponent** |
| B2-P | B2 at matched parameters to the variant | param fairness |
| B2-TF | B2 at matched training FLOPs (may differ in size/tokens) | training-FLOP fairness |
| B2-IF | B2 evaluated at matched inference FLOPs (via CoT token budget) | inference-FLOP fairness |
| B2-CoT | B2 trained on traced (rationale) data, evaluated with CoT decoding | explicit-CoT control |
| B2-PM | B2 with in-context example prompt memory, no state update | prompt-memory control for H2 |
| B2-NM | B2 with bolt-on neural memory module (if H2 scales) | bolt-on-vs-integrated control |
| V* | Agent-discovered variants (V1 loop, V2 delta-memory, V3 deep-sup, ...) | hypotheses |

## Fairness requirements (all enforced by the harness)

- Same data files (hash-recorded), same tokenizer (byte-level, vocab 259 — chosen so no
  tokenizer advantage exists between variants; justification: symbolic tasks +
  TinyStories are ASCII-dominant, and byte-level removes vocab-size as a confound at
  10-50M params), same training-token budget, same parameter budget (+-2%), same
  training-FLOP budget where the comparison claims it, same benchmark splits, disjoint
  train/eval seeds, contamination audit pass recorded per run.
- All results in `experiments/results.csv`; all claims backed by ablations.

## FLOP accounting method (`sage/flops/accounting.py`)

Analytic per-module counts (multiply-accumulate = 2 FLOPs), forward:

- Linear layer `(m,n)`: `2*m*n` per token.
- Attention (per token, causal, seq len T, model dim d, window W):
  QKV/out projections `8*d*d` (adjusted for GQA), score+value `4*d*min(T,W)`.
- SwiGLU MLP: `6*d*d_ff` per token (3 matmuls) + elementwise (ignored, <1%).
- RMSNorm/embedding lookups: counted, negligible.
- **Recurrence:** per-step core cost x number of steps actually executed
  (loop count or halting-determined; the executed-step histogram is logged).
- **Delta-rule scan:** per token `O(d_k * d_v)` state update + readout, counted per
  the chunked-scan algorithm's arithmetic (arXiv:2412.06464).
- **CoT decode:** every generated rationale token is charged a FULL forward pass at
  current context length. CoT is never free.
- Backward = 2x forward (standard approximation, cross-checked).
- Training FLOPs ≈ 6*N*D sanity check for plain transformers (Kaplan convention).

Cross-check: `torch.utils.flop_counter.FlopCounterMode` on tiny shapes must agree with
the analytic count within 10%, or CI fails (`tests/test_flops.py`).

Every eval report includes: FLOPs-per-correct-answer, params, tokens/s, peak VRAM.
