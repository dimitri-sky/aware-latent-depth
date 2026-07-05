# Hypothesis Backlog

Statuses: proposed | testing | killed | revised | scaled | parked.
Rules: pre-registered margins only; every claim links results.csv run_ids; timebox
forces kill-or-park; hardware-compat note is a design-time check (see docs/AGENT.md).

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
  (3/3 seeds, +8.0). Verdict: PROMISING-BUT-UNSTABLE -> EXTEND policy; dissociation
  arm EXP-002-AX (algo_exec) running on pod B alongside EXP-003.
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
- Status: **testing** (EXP-003 launched 2026-07-05 on pod B, interleaved with the
  EXP-002-AX dissociation arm; revised 2026-07-04 after owner-directed literature
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

### H6: Looped delta+SWA hybrid (V3 "Aware" candidate) — H2xH3 combination
- Status: proposed (EXP-005; runs only if H2 and H3 each show >= promising signal;
  MELT arXiv:2605.07721 validates the topology at 1.6B without per-FLOP accounting —
  our niche is the honest-FLOP procedural-reasoning version at 15M).
- Design: loop the {gated-delta + SWA} hybrid block with the H3 training recipe;
  memory + computation families; vs param-matched AND training-FLOP-matched B2.
- Expected advantage: >= 3.0 pts on >= 2 families at matched training FLOPs.
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
