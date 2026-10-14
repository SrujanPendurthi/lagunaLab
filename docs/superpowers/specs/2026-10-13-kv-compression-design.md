# Part 3: KV Cache Compression for Laguna XS — Design

## Goal

Run the longest captured agent session (~78k tokens) on the pruned Laguna XS 2.1
(reap, uniform, 0.5) inside the memory a 16 GB Mac would give it, at a measured
quality cost. This closes the target story from Part 1: *Laguna XS on a 16 GB
laptop with long agent sessions.*

Success is "the session fits and quality holds", not a research curve and not
maximum context. Stop at the first setting that passes the gate.

## Starting point

- Full-attention layers (0, 4, …, 36): 10 layers × 8 KV heads × head_dim 128 ×
  2 (K and V) × 2 bytes (bf16) = 40,960 bytes per token. The captured sessions
  are 57–78k tokens, so 2.3–3.2 GB of KV.
- Sliding layers hold a fixed 512-token window (~60 MB total). Not touched.
- Pruned model, from `results/prune_point.json`: resident peak 10.1 GB,
  streaming (slots=32) peak 3.97 GB. Both measured on short prompts.
- `Model.make_cache()` returns a plain list, `KVCache` for full layers and
  `RotatingKVCache` for sliding ones. Any cache class can be substituted there.
- mlx-lm 0.31.3 ships `QuantizedKVCache(group_size=64, bits=8)`. Stored size per
  128-dim vector: 136 bytes at 8-bit, 72 bytes at 4-bit, against 256 for bf16
  (1.9× and 3.6× smaller).
- `replay.replay.build_ref` / `score` take any model; `score` builds its cache
  with `model.make_cache()`.

Not yet measured, and assumed below: the peak during prefill of a long session
(Part 1 only extrapolated KV linearly), and the real GPU limit of a 16 GB Mac.

## Decisions

1. **Budget:** peak GPU memory ≤ 12 GB, which is 75% of 16 GB (the macOS default
   recommended working set). This is an assumption standing in for a 16 GB
   machine; the result is reported as "under 12 GB on a 24 GB M4 Air".
2. **Measure before building.** Step 0 replays the longest session at full
   length with the plain cache on pruned resident and pruned streaming, and
   records the peak. Estimates: resident ≈ 13.3 GB (over budget), streaming ≈
   7.2 GB (fits). A target that already fits needs no compression, and the
   writeup says so.
3. **Method:** `QuantizedKVCache` on the 10 full-attention layers, at 8 and
   4 bits. No new quantizer.
4. **Reference:** the pruned resident model with the plain cache. Part 3
   measures the damage KV compression adds on top of pruning, not the total
   against the unpruned model.
5. **Evaluation set:** the 5 django sessions at full length (no 16k cap), since
   KV error can only show at long range. Metrics as in Part 2: KL mean and p99,
   top-1 agreement on assistant tokens.
6. **Gate:** the least compression that keeps the longest session under the
   budget with top-1 ≥ 0.98 and KL mean ≤ 0.02 nats against the plain cache.
7. **Speed is reported, not gated:** prefill and decode tok/s per setting.
   mlx-lm's quantized attention (`quantized_scaled_dot_product_attention`)
   materializes the full [heads, chunk, context] score matrix: ~16 GB per layer
   at 82k context. The first P3-4 run hit this on the 8-bit setting, hung the
   machine and caused a watchdog reboot (2026-10-13 20:44). The cache therefore
   stores K/V quantized but dequantizes one layer at a time and attends with
   mx.fast's fused kernel (`DequantKVCache`). Decode re-dequantizes each layer
   per token; if that makes decode unusable, it becomes its own task.

## Components

```
laguna_mlx/cache.py     make_quantized_cache(model, bits): QuantizedKVCache on full
                        layers, RotatingKVCache on sliding layers
replay/replay.py        score() and assistant_logprobs() accept an optional cache
                        factory (default: model.make_cache)
scripts/bench_kv.py     step 0, then plain / 8-bit / 4-bit → results/kv.json (resumable)
tests/test_cache.py     tiny-config tests
results/kv.json         one row per (target, bits): peak GPU, KV bytes, prefill and
                        decode tok/s, KL mean/p99, top-1
```

## Testing

Tiny config only:
- With `make_quantized_cache`, logits stay close to the plain cache, and 8-bit
  is closer than 4-bit.
- Sliding layers get `RotatingKVCache` and full layers get `QuantizedKVCache`,
  matched by `layer_types`.
- Chunked prefill followed by single-token steps gives the same logits as one
  pass with the same quantized cache (cache growth across the 256-step
  boundary).

## Constraints carried over

- The Mac must not crash: never hold two models at once, and stop a run if swap
  passes ~10 GB.
- No new dependencies; don't upgrade mlx or mlx-lm.
- Every number in a writeup comes from a file in `results/`.
- Never tune on anything but the stated gate; django was already used to pick
  the pruning operating point, so Part 3 numbers are post-selection too.

## Risks

- **Prefill activation spike** may dominate the peak, so compressing KV alone
  might not reach the budget. Step 0 shows this; the fallback is a smaller
  prefill chunk, then a higher pruning ratio.
- **Full-length replay is slow.** Five sessions of 57–78k tokens per setting.
  The bench is resumable per (target, bits, session).
- **Pruned base in question:** greedy action match on the pruned model is 0.3
  (`results/prune_point.json`). `scripts/diag_actions.py` is checking whether
  that is the exact-match metric being harsh or real damage. If it is real
  damage, the base model for Part 3 changes before any of this runs.

## Out of scope

Token eviction, compressing sliding layers, context beyond the captured
sessions (toward the 262k limit), and SWE-bench end-to-end solve rate.
