# Hypothesis Backlog

Statuses: proposed | testing | killed | revised | scaled | parked.
Rules: pre-registered margins only; every claim links results.csv run_ids; timebox
forces kill-or-park; hardware-compat note is a design-time check (see docs/AGENT.md).

---

### H1: Shared-weight recurrent latent depth improves reasoning-per-FLOP
- Status: testing (EXP-001)
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
- Est. cost: ~6 GPU-h local (9 runs) | Actual: TBD
- Timebox: 2 revisions max, then kill-or-park.

### H2: Gated delta-rule fast-weight memory + SWA improves in-session rule learning
- Status: proposed (EXP-002, runs only after EXP-001 verdict)
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

### H3: Deep supervision at recurrent steps (+ iteration distillation) improves latent reasoning
- Status: proposed (EXP-003; only if H1 survives)
- Expected advantage: V1-loop4 + deep supervision beats plain V1-loop4 by >= 2.0
  points at identical FLOPs-per-token budget (deep supervision changes loss, not
  inference cost).
- Minimum falsifier: flip `deep_supervision: true` on the EXP-001 winner config, 3 seeds.
- Evidence: TRM arXiv:2510.04871 (deep supervision is its key ingredient);
  caveat arXiv:2512.11847 (augmentation/supervision drive much of TRM's gain).
- Hardware-compat note: extra coda decodes during training only; no inference cost.
- Est. cost: ~2 GPU-h | Timebox: 1 revision.

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
