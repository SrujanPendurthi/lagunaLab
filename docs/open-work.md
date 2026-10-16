# Open work and known limitations

What is left, what is unmeasured, and where the current code stops. Nothing
here blocks the Part 3 answer (`serve_streaming --preset laptop16`: 10.16 GB
peak, 6.87 tok/s, exact at 82k; [part3-answer.md](part3-answer.md)).

## Decisions waiting on the user

| Question | Context | Default if nobody decides |
|---|---|---|
| Turn `fused_decode` on by default anywhere? | Decided for `laptop16` (off: plain is faster and exact). `make_quantized_cache` and `DequantKVCache` keep `fused_decode=False` as their default, and moeStream's imports depend on that default (rule 7) | off |

Decided:

- **Prompt lookup (user, 2026-10-16): keep, unused.** `generate_lookup` gives
  1.22× decode but 1.00× end to end resident (`lookup.json`); it is exact and
  tested, so keeping it costs nothing.
- **Drop-on-miss simulation code (user, 2026-10-16): removed.** It reproduced
  a failed gate (KL 0.146, top-1 0.937; `drop_sim.json`); removed 2026-10-16 (user's decision); the last version is in `12e008c` (`git show 12e008c:scripts/sim_drop.py`).

## Engineering, optional

### Exact fix for the streaming sync floor (not pursued)

At 32k, 128 slots (nothing loaded) is still 22 ms per step slower than
resident (68.9 against 46.8 ms; `stream_profile.json`): 39 host round-trips
per token. Every Python-level fix was measured or assessed and dropped
([no-gos.md](no-gos.md)). The exact fix would let the GPU read expert pages
straight from an mmap of the pack, so routing never leaves the device. That
needs a C++ MLX extension (wrap file-backed memory as a Metal buffer, handle
non-resident pages), which means adding build tooling. **The user chose not to
pursue it (2026-10-15).** A smaller first step, if it is ever revisited: a
spike that only checks whether a Metal buffer over a file-backed mmap can be
read by a kernel when its pages aren't resident.

### Tune the fused 8-bit decode kernel

`laguna_mlx/q8_attn.py` runs at ~15 GB/s effective in isolation (0.18 GB in
12.24 ms at 82k, `kv_decode_micro_fused.json`), against ~47 GB/s for MLX's
fused SDPA on bf16. Its real-model gain (1.69×) comes mostly from not
allocating bf16 temporaries, not from kernel speed. Likely causes, not
profiled:

- each simdgroup walks its 64 keys serially, with G = 6 `simd_sum`
  reductions and 12 `exp`s per key, so it is latency-bound;
- only one 32-bit load per lane per row, so memory-level parallelism is low.

Ideas, in rough order of effort: process several keys per simdgroup step and
reduce once (amortize `simd_sum`); give each lane whole keys instead of 4 dims
(no cross-lane reduction for scores); tune `BLOCK` and `SIMDS` for the 10-core
M4 GPU. If it reached plain SDPA's bandwidth on half the bytes, 8-bit decode
would beat the plain cache, and `--fused-decode` could become the faster
choice as well as the smaller one. Any change must keep
`tests/test_q8_attn.py` passing and should rerun the `--qdecode fused` rows.

### Smaller items

- `decode_peak_gpu_gb` in `bench_kv` covers prefill plus decode, not decode
  alone; the name overstates it.
- `DequantKVCache.state` has no setter, so mlx-lm's prompt-cache save/load to
  disk would fail. The in-memory LRU prompt cache only deepcopies and trims, so
  serving works.
- References built before `2fb0bcc` (`refs/prune`, `refs/prune_actions`,
  `refs/kv`) don't record their model; `score` returns `ref_model: None` for
  them. Which model built each set is documented in
  [parity-harness.md](parity-harness.md).

## Not measured

| What | Why it matters | How to measure |
|---|---|---|
| **An actual 16 GB Mac** | the 12 GB budget is an assumption (75% of 16 GB); a smaller page cache would serve more of the ~21.6 misses per step at 64 slots from the SSD (≈ 38 MB of reads per token, estimate) | run `bench_kv --target streaming --slots 64 --cache-limit-gb 1.0` there |
| **`laptop16` served end to end** | a `pool` session through `serve_streaming`, including the prompt cache's memory (up to 4 GiB allowed) | `serve_streaming --preset laptop16` + `gen_traces --n 1`, watch footprint |
| **32 slots with the cap at 82k** | ~2 GB more headroom than 64 slots; capped 32k profile suggests ~12 against 13.9 tok/s | `bench_kv --target streaming --slots 32 --cache-limit-gb 1.0` |
| **Action match on 100 actions** | the pruned model's 6/20 exact, 18/20 same tool is a small sample | `prune_point`-style refs with more actions, then `diag_actions` (~1 h GPU, estimate) |
| **Speed variance** | every row is a single run; some ran with video playing or the laptop in use (plain `laptop16` measured 6.87 and 6.07 tok/s in two runs) | repeat runs on an idle machine |
| **Why quantized-KV prefill is faster than plain** | 109.5 / 114.5 against 95.8 tok/s resident (`kv.json`) | profile one chunk |
| **Why pruned decode is faster than full** | 38.7 against 34.2 tok/s on short prompts (`prune_point.json`); smaller expert tensors is the guess | profile |
| **The 128-slot, 256-context outlier** | sync 67.6 ms against ~38 for 32–64 slots (`sweep_capped`); peak 12.8 GB, video playing | rerun idle |
| **Whether streaming's greedy differences are bf16 ties** | `identical_frac` < 1 against resident while logits match to 1e-5; the tie hypothesis is unverified for streaming (verified for DFlash in S1) | `diag_dflash`-style divergence check on streaming |
| **Cold page cache** | streaming numbers right after packing may have had a warm page cache; purging needs sudo | `sudo purge` before a run |

## Known limitations of the current code

- **Fused kernel:** 8-bit, group size 64, head_dim 128 only, and at most 6
  query heads per KV head (threadgroup memory). Laguna's full-attention layers
  have 48/8 = 6. Single-token steps only; prefill uses the dequantize path.
- **`--fused-decode` serving** is one request at a time: the cache has no
  `merge`, so mlx-lm skips its batch generator.
- **Streaming prefill** runs several masked passes when a chunk needs more
  experts than there are slots (a `ponytail` note in `streaming.py`). At 64
  slots of 128 that is common for 2,048-token chunks; prefill is ~71 tok/s.
- **Streaming is exact per forward, not always per greedy sequence:**
  summation order across experts differs from resident, so exact bf16 ties can
  break differently.
- **Pruning:** only uniform allocation can be materialized, since
  `num_experts` must be one number.
- **Parity evidence against llama.cpp** is one valid row (top-1 0.969); the
  port's correctness rests mainly on the tiny-config tests against Poolside's
  PyTorch code.
- **Pinned versions:** MLX 0.32.0 and mlx-lm 0.31.3. The pinned mlx-lm has no
  Laguna of its own; upstream does now, and an upgrade would need the port,
  the caches and the kernel rechecked.
