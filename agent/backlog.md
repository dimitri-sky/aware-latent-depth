# Hypothesis Backlog

Statuses: proposed | testing | killed | revised | scaled | parked.
Rules: pre-registered margins only; every claim links results.csv run_ids; timebox
forces kill-or-park; hardware-compat note is a design-time check (see docs/AGENT.md).

## Program map (updated 2026-07-05)

Two ingredients, tested separately, combined if positive:

    DELTA LAYERS (H2 + EXP-006 attribution: delta-driven, broad, cheaper)
      └──> V3 "Aware" (H6, EXP-005) = delta-centric, NO loops
             -> 50M scale check -> demo + report ("when does latent depth pay?")

- H1 killed; H3 PARKED PERMANENTLY (EXP-003 instrument-fail, EXP-003B retry still
  loop-invariant + loses to own controls; timebox exhausted). Loops out of V3.
- EXP-006 2x2: V2's win is the delta layers (+22.8), not the SWA window (+3.5);
  efficiency edge also delta-driven. B2-SWA fails to reproduce memory-family gains.
- H2 EXTEND (6 rule_shift seeds): 5/6 positive small deltas + 1/6 grokked to 100%
  (0/6 for B2) — bimodal, checkpoints kept for the grokking side-study.
- Parked/support: H4 halting (efficiency ablation later), H5 Titans (replication debt),
  H0a/H0b nulls (B2 opponent in every run; CoT control after verdicts).

## Naming system

Three namespaces, three lifecycles (a killed hypothesis does not kill its model —
V1 outlived H1 and is retried under H3):

| tag | kind | examples | lives in |
|---|---|---|---|
| H# | hypothesis (claim; killable) | H2 memory, H3 recipe-loops, H0a/b nulls | this file |
| EXP-### | experiment (pre-registered test of one H; suffix = arm) | EXP-001B (H1 control), EXP-002-RS/-CP/-SG/-AX (H2 arms) | agent/log/EXP-*.md |
| V# / B# | model architecture (V=variant, B=baseline; suffix = config) | V1-loop4, V2-delta, V3 combo (planned), B2-6L | models/ + configs |

Run IDs compose all three: `EXP-002-RS-V2-delta-s0-2581d6` =
experiment-arm + model + seed + config hash (traceable in results.csv).
H#-to-EXP-### numbering is NOT aligned (H6 -> EXP-005); check the map above.

---

### H1: Shared-weight recurrent latent depth improves reasoning-per-FLOP
- Status: **killed** (EXP-001B, 2026-07-04; agent/log/EXP-001B.md). The equal-params
  gain vanished under the matched-training-FLOP control (loop1 @ 2.5x steps ties or
  beats loop4), and loop4 costs 2.5x inference FLOPs — per-FLOP claim dead. Residual:
  consistent equal-params rewrite advantage (+9.8 mean, 5/6 seeds) = parameter
  efficiency, not FLOP efficiency; re-propose only for memory-bound deployment cases.
- Kill discipline: per owner directive, a near-miss with promising signal gets
  EXTEND (seeds/steps/regime) before any kill decision — see decision_policy.md.
- Expected advantage: V1-loop4 beats V1-loop1 by >= 3.0 accuracy points (family mean)
  on algo_exec + rewrite at equal params, 3 seeds, mean diff > 2x pooled seed SD.
  Secondary: loop4 closes >= 50% of the gap toward depth-matched B2-10L.
- Minimum falsifier: EXP-001 (15M loop model, loop 1 vs 4 vs 10L Transformer++,
  identical data/tokens). If loop count changes nothing at equal params, H1 is in
  serious trouble.
- Evidence: Ouro arXiv:2510.25741; Huginn arXiv:2502.05171; looped-TF arXiv:2502.17416;
  TRM arXiv:2510.04871. Formal caveat: loops lose to CoT on stochastic-sampling tasks
  (arXiv:2505.19245) — SAGE contains both types deliberately.
- Hardware-compat note: KV-cache sharing / last-step reuse exists at scale (Ouro);
  truncated BPTT (bptt_loops=2) avoids full unroll; core is plain TF blocks (no
  kernel risk).
- Est. cost: ~6 GPU-h local (9 runs) | Actual: ~2.2 GPU-h (RunPod, 9/9 ok)
- Timebox: 2 revisions max, then kill-or-park.

### H2: Gated delta-rule fast-weight memory + SWA improves in-session rule learning
- Status: **testing — headline adjudicated** (EXP-002, 2026-07-05; agent/log/EXP-002.md).
  Pooled family-mean delta +13.5 pts (margin PASS by 4.5x) but stability gate
  FAIL on rule_shift seed bimodality (one seed 100%, two ~13%). compress cleanest
  (3/3 seeds, +8.0). Dissociation (EXP-002-AX, 2026-07-05): passes on the letter
  (+.278 rule_shift > +.218 algo_exec) but V2 also beats B2 by +21.8 on algo_exec
  (a non-memory family) at 23% fewer inference FLOPs/correct — advantage is
  BROAD, not memory-specific. EXTEND (6 rule_shift seeds): 5/6 positive, 1/6
  grokked to 100% (0/6 B2). Attribution (EXP-006 2x2): **delta-driven** — delta
  effect +22.8 vs window +3.5; SWA alone reproduces nothing. Final verdict:
  the gated delta-rule layers are a genuinely better generic component at 18M
  on SAGE, for accuracy AND per-correct cost. H6/EXP-005 proceeds delta-centric.
- Expected advantage: V2 beats param-matched B2 by >= 3.0 points on rule_shift +
  compress + state_guard family mean at matched training FLOPs; dissociation check:
  gain on rule_shift must exceed gain on algo_exec (memory, not generic capacity).
- Minimum falsifier: 15M V2 (delta every 2nd layer, window 128) vs param-matched B2,
  families rule_shift + compress + state_guard, 3 seeds.
- Evidence: Gated DeltaNet arXiv:2412.06464 (ICLR 2025); fast-weight programmers
  arXiv:2102.11174; Kimi Linear arXiv:2510.26692.
- Hardware-compat note: current impl is a sequential scan (slow, correct); the
  chunked-parallel form exists (flash-linear-attention) for cloud Linux runs —
  training parallelism is proven, not speculative.
- Est. cost: ~5 GPU-h local | Actual: TBD
- Timebox: 2 revisions max.

### H3: Loops pay their FLOP cost when trained with the 2026 recipe
- Status: **parked permanently** (EXP-003 instrument-fail + EXP-003B retry,
  2026-07-05; agent/log/EXP-003.md). The fixed recipe (step embedding, detached
  weighted readouts, bptt 4) still produced a loop-invariant model (K-gap ~+1.5
  vs 5-pt gate) that loses to its own loop1/FM controls. Timebox exhausted
  (one retry). Residual: EXP-001B param-efficiency finding stands; latent
  iteration remains plausible in continuous-output domains (RD-VLA) — out of
  scope for SAGE. (Revised 2026-07-04 after owner-directed literature
  re-scan; agent/lit_scan_2026-07.md). Owner directive: H1's kill is verdict on
  vanilla loops only; loops get their evidence-backed second chance here.
- Recipe (from lit scan): per-loop readout supervision (LOTUS +6.7 pts), randomized
  loop counts + truncated BPTT (RD-VLA), input injection (already have), middle-cycle
  allocation (already have: 1+2+1).
- Expected advantage: recipe-loop4 beats the matched-TRAINING-FLOP loop1 control by
  >= 3.0 pts family mean on rewrite + algo_exec, 3 seeds, > 2x pooled SD. The control
  ships in the same batch (EXP-001B lesson).
- Mandatory diagnostic: K-gap (same checkpoint evaluated at loop 1 vs 4) must exceed
  noise — else the readout blind spot (arXiv:2606.24898) voids the run.
- Evidence: TRM arXiv:2510.04871; LOTUS arXiv:2606.31779; RD-VLA arXiv:2602.07845;
  blind-spot caveat arXiv:2606.24898.
- Est. cost: ~3 GPU-h cloud | Timebox: 1 revision.

### H6: V3 "Aware" candidate — delta-centric (loops excluded per H3 park)
- Status: proposed (EXP-005, redesigned 2026-07-05 after EXP-006 attribution).
- Design: delta-density sweep (delta_every 1 vs 2 vs 3) at fixed params on the
  full family suite (memory + computation + generalization), vs param-matched
  AND training-FLOP-matched B2 in the same batch. The winning density = V3.
- Expected advantage: >= 3.0 pts on >= 3 of 5 families at matched training
  FLOPs, plus flops/correct <= B2 on those families.
- Est. cost: ~4 GPU-h cloud | Timebox: 2 revisions.

### H4: Learned halting improves compute allocation (efficiency only)
- Status: parked (test after a 160M-bound variant exists)
- Expected advantage: matches fixed-depth accuracy at >= 25% lower mean inference
  FLOPs. Never expected to improve accuracy.
- Evidence: ACT arXiv:1603.08983 unstable; TRM ablation: ACT slightly hurts (86.1 vs
  87.4 Sudoku-Extreme); Ouro's entropy-regularized exit works as efficiency feature.
- Timebox: 1 revision; kill on any accuracy regression > 1 point.

### H5: Titans-style test-time neural memory
- Status: parked (deprioritized)
- Reason: replication debt — arXiv:2510.09551 shows chunking sensitivity and that
  memory-only test-time updates with a frozen backbone barely help. H2 tests the
  better-evidenced fast-weight mechanism first. Revisit only if H2 shows memory
  signal but insufficient capacity.

### H0a (null): modern Transformer++ is already hard to beat
- Status: live — B2 is the reference opponent in every experiment.

### H0b (null): explicit CoT beats latent computation at matched inference FLOPs
- Status: proposed (EXP-004; after EXP-001 verdict)
- Minimum falsifier: B2 trained on traced data (CoT decoding, rationale budget
  counted at full FLOPs) vs V1-loop4 at matched inference FLOPs on algo_exec +
  dsl_learn (deterministic, loop-favored) AND a sampling-flavored family if added.
