# Part 3 answer: a 16 GB laptop running long agent sessions

**Recommendation:** run the pruned-50 model (REAP, 128 experts per layer)
streamed from SSD:

- **64 expert slots per layer**;
- **plain KV cache**;
- **MLX buffer pool capped at 1 GB**: `load_streaming(pack, 64, cache_limit_gb=1.0)`.

On the longest captured agent session (82,671 tokens) this peaks at
**10.16 GB**, under the 12 GB budget. It decodes at **6.87 tok/s** and matches
the pruned resident model exactly (top-1 1.0, KL 0). That is 93% of resident
decode speed at 70% of resident peak memory.

Every number below comes from a file in `results/`, named next to it. The
12 GB budget is a stand-in, 75% of 16 GB. Nothing here was run on an actual
16 GB machine.

## The candidates at 82k tokens

Source: `results/kv.json` → `summary`. Quality is measured against the pruned
model with a plain cache. Resident rows cover all 5 django sessions; streaming
rows cover the longest session. Peak is `mx.get_peak_memory`, the maximum over
prefill and decode.

| Setting | Peak GB | KV GB | Top-1 | KL mean | Decode tok/s @82k | Prefill tok/s | 12 GB budget |
|---|---|---|---|---|---|---|---|
| Resident, plain KV | 14.60 | 3.39 | 1.0 | 0 | 7.42 | 95.8 | fail |
| Resident, 8-bit KV | 13.16 | 1.80 | 0.9963 | 0.0020 | 4.10 | 109.5 | fail |
| Resident, 4-bit KV | 12.29 | 0.95 | 0.9944 | 0.0038 | 4.99 | 114.5 | fail (+0.29 GB) |
| Streaming, 32 slots, uncapped pool (superseded) | 8.04 | 3.39 | 1.0 | 0 | 4.69 | 69.6 | pass |
| **Streaming, 64 slots, 1 GB pool cap** | **10.16** | 3.39 | 1.0 | 0 | **6.87** | 71.0 | **pass** |

- **KV quantization doesn't get the resident model under budget.** Peak minus
  KV is 11.2–11.4 GB in every resident row. That is the weights plus the
  long-prefill working set, so even 4-bit KV ends 0.29 GB over. Quantized KV
  also decodes slower at 82k, because `DequantKVCache` re-dequantizes each layer
  every token.
- **Streaming is exact** (top-1 1.0, KL 0). It uses the same weights and
  kernels, so no compression is needed.
- **The pool cap makes 64 slots worth it.** With the cap, the old 32-slot
  number (4.69 tok/s) is superseded: that run was slowed by MLX's uncapped
  free-buffer pool pushing the expert files out of the page cache. 32 slots with
  the cap at 82k hasn't been measured.

## Why 64 slots: the capped slot sweep

Source: `results/stream_profile.json` → `sweep_capped`. 1 GB pool cap, 32
decode steps after a 32,768-token prefix of the longest session. Two YouTube
tabs may have been playing.

| Slots | Step ms @32k | tok/s @32k | Hit rate | Misses/step | Peak GB @32k |
|---|---|---|---|---|---|
| 16 | 97.5 | 10.3 | 0.47 | 165.3 | 4.95 |
| 32 | 83.5 | 12.0 | 0.70 | 94.6 | 5.91 |
| 64 | 72.2 | 13.9 | 0.93 | 21.6 | 7.88 |
| 128 (all experts) | 68.9 | 14.5 | 1.0 | 0 | 12.22 |

- **64 slots gets most of the gain:** 64 to 128 saves only 3.3 ms per step but
  costs 4.3 GB more.
- **128 slots doesn't fit:** at 82k it would add the same ~2.3 GB of KV
  growth seen at 64 slots, landing near 14.5 GB. That's an estimate, not a
  measurement.

## Where streaming decode time goes (S5)

Source: `results/stream_profile.json`.

| Run | Step ms @256 | Step ms @32k |
|---|---|---|
| Resident pruned (`resident`) | 23.1 | 46.8 |
| Streaming, 32 slots, uncapped pool (`rows`) | 54.9 | 169.9 |
| Streaming, 32 slots, 1 GB cap (`sweep_capped`) | 58.7 | 83.5 |
| Streaming, 64 slots, 1 GB cap (`sweep_capped`) | 50.8 | 72.2 |

- **The uncapped pool was the long-context problem.** At 32k, capping it cut
  expert-load time from 65.4 to 15.0 ms per step at 32 slots.
- **What remains is the per-layer sync,** 63–65 ms per step at 32k for every
  slot count. Even with every expert resident (128 slots), the step is 22 ms
  slower than the resident model. That's the cost of 39 host round-trips per
  token. results.md ("The sync floor") lists the candidates.
- **Three measured dead ends:**
  - starting each layer's expert compute early with `mx.async_eval` was about
    1.5× slower, with identical tokens (`async_ab`);
  - batched slot writes were slower (`ab`);
  - prefetching the next layer's experts was about break-even in the sibling
    moeStream project, by its own numbers, not this repo's.

## How to run it

```python
from laguna_mlx.streaming import load_streaming
model, tok = load_streaming("models/Laguna-XS-2.1-pruned-50-packed", slots=64, cache_limit_gb=1.0)
```

`scripts/serve_streaming.py` passes `cache_limit_gb=1.0`. Set its slot count
with `--slots 64`; its default is unchanged.

## What is not measured

- **An actual 16 GB Mac.** The 12 GB budget is an assumption.
  - On a 16 GB machine the page cache is smaller, so more of the ~21.6 misses
    per step at 64 slots (32k) would come from the SSD instead of memory.
  - Rough worst case: 21.6 × 1.77 MB ≈ 38 MB of reads per token. That's an
    estimate; its cost wasn't measured.
- **Process footprint at 82k with 64 slots.** `mx.get_peak_memory` (10.16 GB)
  doesn't count the free-buffer pool, which is capped at 1 GB. So top's
  footprint should stay at or below about 11.2 GB, but that wasn't recorded.
  The run never tripped the 14 GB watchdog.
- **32 slots with the cap, at 82k.** It would trade speed for about 2 GB more
  headroom. The capped 32k profile suggests about 12 tok/s against 13.9 at
  64 slots.
- **Lowering the sync floor further** would need a redesign that routes on the
  GPU. `mx.compile` was assessed and ruled out (results.md, "The sync floor").
  The lossy drop-on-miss variant failed the quality gate (KL 0.146, top-1
  0.937; `results/drop_sim.json`).
- **The serving path end to end** with the cap and 64 slots: a `pool` agent
  session through `serve_streaming`.
- **Speed variance.** Each row is a single run, and some ran alongside video
  playback.
