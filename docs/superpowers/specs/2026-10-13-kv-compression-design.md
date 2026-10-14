# Part 3: KV Cache Compression for Laguna XS — Design

## Goal

Run the longest captured agent session (82,671 tokens) on the pruned Laguna XS 2.1
(reap, uniform, 0.5) inside the memory a 16 GB Mac would give it, at a measured
quality cost. This closes the target story from Part 1: *Laguna XS on a 16 GB
laptop with long agent sessions.*

Success is "the session fits and quality holds", not a research curve and not
maximum context. Stop at the first setting that passes the gate.

## Starting point

- Full-attention layers (0, 4, …, 36): 10 layers × 8 KV heads × head_dim 128 ×
  2 (K and V) × 2 bytes (bf16) = 40,960 bytes per token. The captured sessions
  are 42.8k–82.7k tokens, so 1.8–3.4 GB of KV.
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
3. **Method:** `QuantizedKVCache` storage on the 10 full-attention layers, at
   8 and 4 bits, wrapped in `DequantKVCache` for attention (decision 7). No new
   quantizer.
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
laguna_mlx/cache.py     make_quantized_cache(model, bits): DequantKVCache on full
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
- Sliding layers get `RotatingKVCache` and full layers get `DequantKVCache`,
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
- **Full-length replay is slow.** Five sessions of 42.8k–82.7k tokens per setting.
  The bench is resumable per (target, bits, session).
- **Pruned base in question:** greedy action match on the pruned model is 0.3
  (`results/prune_point.json`). `scripts/diag_actions.py` is checking whether
  that is the exact-match metric being harsh or real damage. If it is real
  damage, the base model for Part 3 changes before any of this runs.

## Out of scope

Token eviction, compressing sliding layers, context beyond the captured
sessions (toward the 262k limit), and SWE-bench end-to-end solve rate.

## Result (P3-4, 2026-10-14)

All numbers from `results/kv.json` (pruned-50, 5 django sessions at full length,
longest 82,671 tokens; peak = max of prefill and decode peak).

| setting           | peak GB | KV GB | top-1  | KL mean | KL p99 max | prefill tok/s | decode tok/s | gate |
|-------------------|---------|-------|--------|---------|------------|---------------|--------------|------|
| resident plain    | 14.60   | 3.39  | 1.0    | 0       | 0          | 95.8          | 7.42         | fail: budget |
| resident 8-bit    | 13.16   | 1.80  | 0.9963 | 0.0020  | 0.050      | 109.5         | 4.10         | fail: budget |
| resident 4-bit    | 12.29   | 0.95  | 0.9944 | 0.0038  | 0.120      | 114.5         | 4.99         | fail: budget (+0.29 GB) |
| streaming plain   | 8.04    | 3.39  | 1.0    | 0       | 0          | 69.6          | 4.69         | pass |

prefill tok/s: resident rows are the mean over 5 sessions (includes the 42.8k one);
streaming is the longest session only. Longest session alone, resident: 92.6 plain,
104.0 8-bit, 102.4 4-bit.

- **No resident setting passes.** Quality holds at both 8 and 4 bits (every row
  top-1 ≥ 0.9878, KL mean ≤ 0.0142); the budget is what fails. 4-bit misses by
  0.29 GB at 82k (12.19 GB already at 78k).
- **Streaming needs no compression.** Pruned streaming (slots=32) with the plain
  cache runs the longest session at 8.04 GB, 4 GB under budget, identical to
  resident (top-1 1.0, KL 0).
- Peak minus KV is ~11.2–11.4 GB in every resident setting (short-prompt resident peak 10.1 GB, `results/prune_point.json`, plus
  ~1.4 GB prefill working set). Shrinking KV past 4-bit cannot reach 12 GB; the
  working set is what's left.
- Quantized KV decodes slower than plain (4.1 / 5.0 vs 7.4 tok/s at 82k):
  `DequantKVCache` re-dequantizes the whole layer per token (decision 7).
  Prefill is faster with quantized KV (110–115 vs 96 tok/s), not measured why.

Options for the human (none started):
1. Accept streaming/plain as the 16 GB answer: fits with no KV compression.
2. Raise the budget to ~12.3 GB and accept resident 4-bit.
3. Spec fallback 1: smaller prefill chunk (1024) for resident 4-bit on the
   longest session only (~25 min GPU), to recover the 0.29 GB.
4. Spec fallback 2: higher pruning ratio (new base, redo quality work).

**Decision (human, 2026-10-14): option 1.** Pruned streaming with the plain cache is
the 16 GB answer for the longest captured session. KV quantization stays in the
codebase (`make_quantized_cache`) but is not the operating point.
