# Part 2: Expert Pruning for Laguna XS — Design

## Goal

Find how many of Laguna XS 2.1's routed experts can be removed, one-shot and
without retraining, and what it costs in agent behaviour, measured by
teacher-forced replay of Laguna's own SWE-bench sessions. Deliver:

1. A quality-vs-pruning curve for several criteria (the research result).
2. One materialized pruned checkpoint at a chosen operating point, with its
   memory and speed on the 24 GB M4 Air (resident and streaming).

Target story: *Laguna XS pruned to fit a 16 GB Mac resident, and smaller/faster
when streamed, at a measured quality cost.* 4-bit weights are ~18.6 GB and the
routed experts are ~17.4 GB of that, so pruning 50% of experts takes the model to
~9.9 GB.

## Starting point (from Part 1)

- `laguna_mlx/laguna.py`: `SparseMoeBlock.route(x) -> (inds, w)` (sigmoid router,
  selection-only `e_score_correction_bias`, `norm_topk_prob`, routed scale 2.5);
  `switch_mlp(x, inds) -> [..., k, hidden]` (per-selection expert outputs, before
  gate weighting); 39 MoE layers (1–39) × 256 experts, top-8.
- `laguna_mlx/hooks.py`: `RoutingRecorder` (counts, gate mass).
- `replay/replay.py`: `build_ref`, `score` (top-64 truncated KL, top-1 agreement).
- `traces/`: 11 captured pool sessions from the streaming model — 6 astropy,
  5 django, 57–78k tokens each.
- Measured: resident decode 18.6 tok/s at 19.8 GB peak; streaming 10 tok/s at 4.7 GB.
  Resident prefill speed on real prompts is not yet measured (Task 10), and it
  sets the cost of every replay pass.

## Decisions

1. **Criteria compared:** REAP (Cerebras 2025, arXiv 2510.13999), where the
   saliency of expert j is the mean over tokens routed to j of gate_j(x)·‖f_j(x)‖;
   routing frequency; gate mass; and a random baseline. REAP is the expected winner
   (it reported ~97% retention at 50% pruning on agentic SWE-bench for Qwen3-Coder);
   frequency and random show how much the criterion matters.
2. **Allocation:** uniform (same fraction pruned in every layer) as the main sweep;
   global (one budget, layers compete on normalized saliency, at least 16 experts
   kept per layer) at two ratios, to test whether some layers tolerate more pruning.
3. **Split — no tuning on the test set:** calibrate (collect saliency) on the
   6 astropy sessions; evaluate on the 5 django sessions (a different repository);
   also score 2 astropy sessions to measure the calibration–held-out gap.
4. **Evaluation path:** resident 4-bit model with pruning applied as a router
   mask (pruned experts' selection bias = −inf, so they are never chosen and the
   remaining top-8 weights renormalize as usual). The reference and every pruned
   run use the same kernels, so an unpruned run scores KL ≈ 0 exactly. Masking needs
   no new checkpoints for the sweep.
5. **Eval budget:** each session capped at 16,384 tokens, so the held-out set is
   ≈ 80k tokens, ≈ 30k of them assistant tokens. Increase the cap if resident
   prefill turns out fast (decided from Task P2-1 timings).
6. **Metrics per config:** held-out KL (mean, p99), top-1 agreement, calibration-set
   KL, and experts kept per layer; for the operating point, also greedy action
   match on held-out turns, resident and streaming memory and speed, and streaming
   hit rate.
7. **Materialization:** uniform pruning only (the same expert count in every layer
   keeps `num_experts` a scalar); the operating point is written as a normal MLX
   checkpoint (sliced `switch_mlp`, router rows, bias) plus a repacked streaming
   directory. If global allocation clearly wins, per-layer counts are follow-up work.
8. **Operating point:** chosen by the human from the curve. The default suggestion
   is the largest uniform ratio whose held-out top-1 agreement is ≥ 0.90 and whose
   KL mean is ≤ 0.10 nats.

## Components

```
laguna_mlx/pruning.py      ExpertSaliencyRecorder, keep_sets, apply_mask, prune_checkpoint
scripts/prune_sweep.py     calibrate → refs → sweep → results/pruning.json (resumable)
scripts/prune_point.py     materialize + measure the chosen operating point
tests/test_pruning.py
results/expert_stats.npz   per-layer per-expert count, gate mass, REAP sum (calibration set)
results/pruning.json       one row per (criterion, allocation, ratio)
results/prune_point.json   operating-point numbers
```

## Testing

Tiny config only (the real model is exercised by the scripts):
- The saliency accumulator matches hand-computed values; the recorder is
  transparent (logits unchanged) and its counts agree with `RoutingRecorder`.
- The mask never selects pruned experts; masking experts never routed gives
  identical logits; `apply_mask` can be undone.
- A materialized pruned checkpoint loads with mlx-lm and matches the masked
  model's logits.
- `keep_sets`: exact counts, the per-layer minimum respected, and deterministic
  results.

## Risks

- **Slow resident prefill** makes the sweep take hours. Mitigation: measure first,
  resume the sweep across runs, and shrink the cap or ratios.
- **Calibration domain is narrow** (astropy only). Mitigation: the django held-out
  set measures transfer, and the gap is reported.
- **Top-64 truncated KL** understates damage when the pruned model moves mass
  outside the reference top-64; top-1 agreement and action match cover that.
- **bf16 ties:** greedy action match can diverge on ties; per-token metrics are
  the primary signal.

## Out of scope

Expert merging (REAP showed pruning beats merging for generative tasks),
post-pruning distillation or fine-tuning, per-layer counts in the checkpoint
format, and SWE-bench end-to-end solve rate (Laguna solved few tasks on this
machine; replay is the metric).
