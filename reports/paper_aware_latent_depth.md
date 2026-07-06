# When Does Latent Depth Pay? An Honest-FLOP Study of Loops and Fast-Weight Memory at 15–50M Parameters

*dimitri-sky · aware-research project · July 2026 · preprint draft v1*

## Abstract

Architectural claims for latent computation in language models — recurrent
depth (weight-tied loops) and fast-weight memory (delta-rule layers) — are
typically published without matched-compute controls. We run the controls. On
a contamination-audited procedural benchmark (SAGE, 5 task families, byte-level
models trained from scratch), with parameters matched to ≤0.3%, identical data,
and pre-registered decision margins, we find: **(1)** weight-tied loops never
pay their training-FLOP cost — vanilla loops lose to a step-matched control,
and the 2026 training recipe (per-loop supervision, randomized loop counts,
truncated BPTT) twice produced models that are *loop-invariant*, a silent
failure detected only by a pre-registered K-gap diagnostic; **(2)** gated
delta-rule fast-weight layers pay broadly — a hybrid replacing every 2nd
attention layer beats a parameter-matched Transformer++ on 5/5 families at 18M
while spending 20–40% fewer inference FLOPs per correct answer; **(3)** a 2×2
ablation attributes both the accuracy and the efficiency advantage to the delta
mechanism (+22.8 points) rather than its sliding-window component (+3.5), and
the advantage survives doubling the baseline's training budget (+25.8 on hard
tiers); **(4)** at 50M under a fixed recipe the picture is task-dependent: the
computation-family gap grows (+31.7) while the state-tracking gap reverses
(−5.2). We also observe bimodal skill acquisition (1/6 seeds jumping
discontinuously to 100%). Total cost of the study: ~$38 of commodity cloud
compute. All pre-registrations, run-level results, and code are public.

## 1. Introduction

Two families of architectural mechanisms promise language models "more thinking
per parameter": *recurrent latent depth* — running shared layers multiple times
(Huginn, Ouro, looped Transformers) — and *fast-weight memory* — layers that
write to a recurrent state during the forward pass (DeltaNet lineage, Gated
DeltaNet, production hybrids such as Kimi Linear). Both are active frontier
threads; neither is usually evaluated under the accounting that matters:
**matched parameters *and* matched training FLOPs *and* per-answer inference
FLOPs**, with margins fixed before results exist.

This gap is not an accident. Frontier labs run such controls internally and
rarely publish them; academic papers at small scale rarely hold the FLOP axis
fixed. The result is a literature where "X beats the Transformer" claims are
abundant and rarely comparable. Our contribution is deliberately narrow: a
small, fully audited data point that answers *when latent depth pays* under
honest accounting, including the negative results.

We study three hypotheses at 15–50M parameters on procedural reasoning tasks:
**H1** — weight-tied loops improve reasoning per FLOP; **H3** — loops pay when
trained with the 2026 recipe (per-loop readout supervision, randomized loop
counts, truncated BPTT, input injection); **H2** — gated delta-rule fast-weight
memory improves in-session rule learning. H1 and H3 die; H2 survives every
attack we could afford, with an honest scale caveat.

## 2. Related work

**Loops.** Weight-tied recurrent depth is parameter-efficient at scale (Ouro
1.4B ≈ 4B dense; Huginn). Independent 2026 iso-FLOP studies (Sparse-Layers,
SpiralFormer's anchor control, LoopFormer) agree with our H1 result: dense
vanilla loops are never FLOP-efficient. Ingredients reported to make loops pay:
per-iteration functional diversity (MoE routing, multi-resolution), step-aligned
supervision at loop readouts (LOTUS, +6.7), gated cross-iteration state (MELT),
randomized loop counts with TBPTT (RD-VLA). The *readout blind spot*
(arXiv:2606.24898) — per-loop CE through a shared normalized readout yielding
loop-invariant models — is the trap our K-gap diagnostic pre-registered
against; it fired twice.

**Fast-weight memory.** Delta-rule layers (fast-weight programmers; DeltaNet;
Gated DeltaNet, ICLR'25) interleaved with sliding-window attention are validated
at production scale (Kimi Linear) — but without public per-FLOP accounting at
matched training budgets. Our study supplies the controlled small-scale
evidence chain: attribution, budget robustness, density, scale trend.

**Latent vs explicit reasoning.** Coconut, latent-VLM work (MCOUT, LanteRn) and
RD-VLA argue latent iteration substitutes for chain-of-thought outside pure
text. Our benchmark is symbolic text — CoT's home turf and latent depth's
hardest venue; scope claims accordingly.

## 3. Method

**Benchmark.** SAGE: five procedural families (`algo_exec` program execution,
`rule_shift` mid-session rule changes, `compress` in-session codebook learning,
`state_guard` state tracking, `dsl_learn` fresh-language induction), generated
instances with difficulty tiers 1–5, execution-checked scoring, n-gram +
near-duplicate contamination audit before every training job, disjoint
train/eval seed ranges. A validity gate (trivial-solver battery; 4L-vs-8L
discrimination) precedes all experiments.

**Models.** Byte-level (vocab 259), trained from scratch per family.
Baseline **B2** = Transformer++ (RoPE, SwiGLU, RMSNorm, GQA), 17.83M at 6
layers. **V1/V1R** = prelude(1) → weight-tied core(2)×K → coda(1), input
injection, TBPTT; recipe adds per-loop supervision + clipped-Poisson loop
counts (+ step embeddings and detached weighted readouts in the retry).
**AWARE (V2)** = B2 with every 2nd layer replaced by a gated delta-rule
fast-weight layer, remaining layers sliding-window (w=128); 17.86M (+0.1%).
50M versions: d768/8L, matched to −0.05%.

**Accounting.** Analytic FLOP counts (2 FLOPs/MAC) cross-checked against
`torch.utils.flop_counter` in CI (≤10% or CI fails); loops charged per executed
iteration; every generated token charged a full forward pass at current context
(CoT is never free); backward = 2× forward. Primary efficiency metric:
**inference FLOPs per correct answer**.

**Protocol.** 4000 steps × batch 32 × seq 768 (per-family), lr depth-scaled,
3 seeds (2 for trend reads, stated in each pre-registration), margins fixed in
`agent/log/EXP-*.md` before results; adjudication scripts in `scripts/`.
Deviations logged at decision time (two occurred: one arm dropped for cost
before its results existed; one config bug caught by a feasibility probe).

## 4. Results

### 4.1 Loops never pay (H1, H3: killed / parked)

- **EXP-001/001B** (matched-FLOP control): loop4's equal-params gain on
  `rewrite` (+9.8, 5/6 seeds) vanishes when loop1 trains 2.5× longer; loop4
  costs 2.5× inference FLOPs. H1 killed. Matches three independent 2026
  iso-FLOP studies.
- **EXP-003** (recipe): K-gap < 1 pt — evaluated at K=1 vs K=4, the same
  checkpoint answers identically. The model routes around its recurrence.
  Verdict: instrument-fail, not a kill.
- **EXP-003B** (fixes: loop-index embeddings, detached auxiliary readouts with
  linearly increasing weights, full-depth TBPTT): K-gap ~+1.5 pts, still far
  below the 5-pt gate; also loses to its own loop1 and FLOP-matched controls.
  Timebox exhausted → **H3 parked permanently.**

The methodological point: without the pre-registered K-gap gate we would have
published loop results from models that never used their loops — both times.

### 4.2 Fast-weight memory pays broadly (H2)

18M head-to-head, all-tier accuracy (seeds pooled) and FLOPs/correct:

| family | B2 | AWARE | Δ | B2 F/c | AWARE F/c |
|---|---|---|---|---|---|
| algo_exec | .747 | **.965** | +21.8 | 1.07e10 | **8.0e9** |
| state_guard | .563 | **.612** | +4.8 | 3.51e10 | **2.79e10** |
| compress | .184 | **.263** | +7.9 | 9.40e10 | **5.73e10** |
| rule_shift (6s) | .130 | **.288** | +15.8 | 9.18e10 | **6.3e10** |
| dsl_learn (2s) | .165 | **.205** | +4.0 | — | 4.18e10 |

Pre-registered headline (≥+3.0 pooled family mean): margin passes at +13.5
(4.5×); the 2×-pooled-SD stability gate fails due to rule_shift's bimodality
(§4.5) — reported as such.

### 4.3 Attribution: the delta mechanism, not the window (EXP-006)

2×2 on algo_exec: {full attention, SWA-128} × {delta, no delta}.

| | full attn | SWA-128 |
|---|---|---|
| no delta | .747 | .782 |
| delta | **.975** | .965 |

Delta effect +22.8; window effect +3.5; interaction −4.5. B2-SWA reproduces
none of the memory-family gains (state_guard −4.1). A pre-registered prediction
was falsified: we expected SWA to capture most of the FLOP discount; in fact
the efficiency edge is also delta-driven (window trims ~7% of FLOPs; the rest
is the accuracy denominator).

### 4.4 Budget robustness (EXP-007) and density (EXP-005)

Doubling training to 8000 steps: B2 improves (.747→.795 — the arm
discriminates) yet the tier-3–5 gap remains **+25.8** (bar: retain half of
+21.8). The advantage is not a convergence-speed artifact at this scale.
Density sweep: delta on every 2nd layer beats every-3rd (.965 vs .938, also on
F/c); every-layer was dropped pre-results as cost-dominated (deviation logged).

### 4.5 Bimodal skill acquisition

rule_shift, 6 seeds: AWARE = .140/.130/**1.000**/.145/.170/.145. Five seeds
show a small consistent edge; one transitions discontinuously to perfect
accuracy (train loss 0.0014). B2: 0/6. Same data, same config, different init.
Checkpoints retained for a follow-up predictor study.

### 4.6 Scale trend at 50M (EXP-008): task-dependent, honestly mixed

| family | 18M gap | 50M gap |
|---|---|---|
| algo_exec | +21.8 | **+31.7** (tiers 3–5: +33.4) |
| state_guard | +4.8 | **−5.2** |

Caveat cutting both ways: B2-50M underperforms B2-18M on algo_exec and both
models drop on state_guard — the fixed 50M recipe is likely undertrained, so
part of the algo_exec growth is baseline weakness and the state_guard reversal
may be a tuning artifact. **No uniform-scaling claim is made.**

## 5. Discussion

At this scale, on procedural text, the answer to "when does latent depth pay?"
is: **when the latent computation carries writable state through the sequence
(delta-rule fast weights), and not when it merely re-applies the same
transformation in depth (loops)** — even with the field's best training recipe
for the latter. The delta result aligns with the direction production hybrids
have already taken; our contribution is the controlled, matched, pre-registered
evidence chain (attribution, budget robustness, density, scale) that public
literature skips. The loops result is a cautionary tale with a reusable tool:
the K-gap diagnostic should be standard practice for any looped-model claim.

## 6. Limitations

15–50M scale; procedural byte-level benchmark; per-family specialists (no
multi-task control); one window size; 2–6 seeds; the 50M recipe untuned; the
delta scan implementation is sequential (wall-clock favors B2 — visible in the
demo — but FLOP counts are implementation-independent and chunked kernels
exist). rule_shift's mean gap is seed-distribution-sensitive (median +2.5).
Nothing here licenses frontier-scale claims; the mechanism family is already
validated at scale by others — what transfers from this study is the
accounting methodology and the controlled comparisons.

## 7. Reproducibility

Everything is public: pre-registrations (`agent/log/EXP-*.md`), 1300+ row
results ledger (`experiments/results.csv`, run_id-addressable), frozen configs,
adjudication scripts, pod runners, demo (`scripts/demo.py`), and this report's
figures (`scripts/report_arc1.py`). Total compute: one RTX 3080 (local) + ~55
GPU-hours of commodity RTX 5090 (~$38).

## References

(Bracketed arXiv IDs in text; full list to be formatted for submission:
Huginn 2502.05171; Ouro 2510.25741; looped-TF 2502.17416; TRM 2510.04871;
LOTUS 2606.31779; RD-VLA 2602.07845; readout blind spot 2606.24898;
Sparse-Layers 2605.09165; SpiralFormer 2602.11698; LoopFormer 2602.11451;
CART 2606.01495; MELT 2605.07721; probing 2601.10242; FWP 2102.11174;
Gated DeltaNet 2412.06464; Kimi Linear 2510.26692; Coconut/latent-reasoning
and multimodal latent lines as cited in agent/lit_scan_2026-07.md.)
