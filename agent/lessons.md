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
- (EXP-000 attempt 4) Mod-wrap arithmetic (3-7=96) is a distinct skill that floors
  3-6M models at ~10%; it gated algo_exec tier 3+ for BOTH depths. Difficulty knobs
  must never smuggle in an unrelated hard skill. Wrap ops now enter only at tier 5.
- (EXP-000 attempt 4) Gate criterion corrected BEFORE attempt 5 (pre-registered):
  depth separation required on all 3 computation families; memory families
  (rule_shift, compress) instead require a learnable headroom band — they are H2
  dissociation targets and expecting depth gains there contradicted our own H2
  design. Recorded here to make the criterion change auditable and non-cherry-picked.
- (EXP-000 attempt 4) An 8L model collapsed to degenerate repeats on rule_shift
  (train loss smooth, eval 0%) while 4L scored 100% — optimization instability at
  depth on easy saturated tasks. If it recurs: lower gate lr 6e-4 -> 4e-4.
- (EXP-000 attempt 5, cloud) dsl_learn separation flipped sign vs attempt 4
  (+0.070 -> -0.025) across minor generator changes; rewrite separated (+0.060) but
  single-seed deltas near the margin are suspect. STOP tweaking ramps against a
  possibly-noisy instrument: EXP-000B measures the A/A seed-noise floor and runs a
  max-contrast 2L-vs-16L depth probe (2 seeds) before any further gate attempt.
- (infra) Tiny models use ~10% of a 5090 (kernel-launch bound, 79% util, 3GB VRAM):
  run 3+ training workers per GPU with per-worker results-CSV shards
  (AWARE_RESULTS_CSV) merged after; parallelism-across-experiments beats
  speed-per-experiment at this scale.
- (EXP-001B, H1 kill) Equal-params comparisons flatter recurrence: extra loops are
  extra training AND inference compute. loop1 trained at matched training FLOPs
  (2.5x steps) tied or beat loop4; the "loop gain" was a training-compute artifact.
  Any compute-adding mechanism must ship its matched-FLOP control in the SAME batch.
- (EXP-001B) A 3-seed positive that matters must replicate before adjudication:
  dsl_learn's +4.0 (seeds 0-2) became -0.1 pooled over 6 seeds. Family-level deltas
  under ~5 pts at this budget are noise until >= 6 seeds agree.
- (infra) /sbin/shutdown works in some RunPod containers and not others (systemd
  shim, "Host is down"); pod self-stop must use the RunPod API (runpodctl or MCP
  stop-pod) with shutdown only as a fallback. Cost of the failure: ~9h idle 4090.
- (infra, EXP-004 session A) **NEVER delete/terminate a RunPod pod until its
  results are verified locally.** A STOPPED community pod whose GPU was re-rented
  can often still be started CPU-ONLY (0 GPUs) — usually only from the RunPod
  website UI; the API/MCP `start-pod` just errors "not enough free GPUs" — which
  is enough to SSH in and pull /workspace. Terminating destroys the volume and
  everything on it. Owner directive (2026-07-07): stopped-not-terminated is the
  default end state; terminate only after adjudication. Cost of the failure:
  ~$4.5 of session-A results stranded then destroyed.
- (infra, EXP-004 session A) Auto-stop watchers must not stop the pod right
  after archiving: wait for a /workspace/PULLED marker dropped by the local side
  (60-min window, ~$0.69 worst case) so results are pulled from a LIVE pod.
- (infra, EXP-004 session C) The 32GB memory plan is per-GPU global, not
  per-lane: 3 delta workers + 2 B2 GPU-lane jobs = ~31.9GB -> startup OOM killed
  6 jobs in step 1. Recheck the arc-1 plan (max 2 delta + 1 GPU job, or 3 delta
  ALONE) whenever lanes overlap, including transient overlaps at session start.
