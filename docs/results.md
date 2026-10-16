# Results

All findings, in the order they were produced. Every number comes from the
`results/` file named next to it; nothing here is extrapolated unless it says
so. Hardware for everything: 24 GB M4 MacBook Air, MLX 0.32.0, mlx-lm 0.31.3.

**Comparing speeds.** Decode tok/s is only comparable *within one results
file*. Different scripts use different prompts, lengths, decode loops, and
machine states. For example, the full resident model measured 18.6 tok/s in
`streaming.json`, 40.3 in `dflash.json` and 34.2 in `prune_point.json`.

**Streaming numbers before S5.** Every streaming result before the S5 option-2
rerun was measured with MLX's free-buffer pool uncapped. That pool grows during
long prefills and pushes the expert files out of the page cache. With a 1 GB cap,
the 32k streaming decode step halved (S5 section). Those older streaming
numbers are marked below. They hold as measured, but for streaming speed and
for slot-count conclusions the capped numbers supersede them.

---

## The session corpus

Source: `results/lookup_acceptance.json` → `corpus`.

`pool` (Poolside's agent) ran SWE-bench Lite tasks against this repo's
streaming MLX model, and every request was captured.

| | Sessions | Tokens | Assistant tokens |
|---|---|---|---|
| astropy (calibration) | 6 | 397,400 | 145,768 |
| django (held-out evaluation) | 5 | 366,290 | 155,975 |
| **Total** | **11** | **763,690** | **301,743** |

Session lengths range from 42,771 to 82,671 tokens. The longest,
`django__django-11039` at 82,671 tokens, is the Part 3 target.

---

## Part 1: harness, parity, baseline

### Parity against llama.cpp

Source: `results/parity.json`.

The plan: teacher-force our MLX model on llama.cpp's greedy continuations of 21
prompts, then require top-1 agreement ≥ 0.95 and in-top-5 ≥ 0.99.

What happened: llama.cpp build 10964 has a Laguna bug on short prompts. For
prompts of roughly 36–512 tokens it returns a constant token with null
logprobs. As a result, **20 of 21 reference rows are degenerate**, and the gate
was evaluated on the one valid row:

| Row | Prompt tokens | Top-1 | In top-5 |
|---|---|---|---|
| `long_doc` | 9,782 | 0.969 | 1.0 |

This is weak evidence on its own. Two other checks carry most of the
confidence in the port:

- the tiny-config tests against Poolside's own PyTorch `modeling_laguna.py`;
- replay self-checks scoring KL = 0 exactly.

### Baseline, and streaming across slot budgets

Source: `results/streaming.json` (Task 18; full, unpruned 4-bit model).

> **Uncapped MLX buffer pool.** Streaming numbers here were measured with an uncapped MLX buffer pool. For streaming, the capped numbers in "Streaming decode profile (Task S5)" supersede them. The flat decode across slot counts below is confounded by it. A capped
re-sweep on the pruned model is in the S5 section.

| Target | Decode tok/s | Prefill tok/s | Peak GPU GB | Expert hit rate |
|---|---|---|---|---|
| Resident | 18.6 | 40.6 | 19.81 | n/a |
| Streaming, 16 slots | 10.7 | 52.2 | 3.70 | 0.314 |
| Streaming, 32 slots | 10.1 | 71.0 | 4.66 | 0.405 |
| Streaming, 64 slots | 9.9 | 90.1 | 7.06 | 0.527 |
| Streaming, 128 slots | 9.7 | 115.4 | 12.80 | 0.735 |

At 16k context, streaming decode drops to 4.4 tok/s at both 16 and 32 slots.

- **Decode speed is flat across slot budgets** even though the hit rate more
  than doubles. *(Uncapped pool: the conclusion below doesn't hold for the
  capped pool; see S5.)* So SSD reads are not the decode bottleneck. The suspect is the
  per-layer GPU→CPU sync of router indices (39 per token), which Task S5
  profiles.
- **Prefill does scale with slots**: more experts fit per chunk, so there are
  fewer masked passes.
- `identical_frac` (greedy outputs equal to resident) is 0 at 16–64 slots and
  0.67 at 128 slots. The per-forward logits match to 1e-5. The recorded
  hypothesis is that bf16 exact ties break differently because the summation
  order across experts differs. This hasn't been verified directly.

### DFlash speculative decoding: negative result

Sources: `results/dflash.json`, `results/dflash_diag.json`.

This is the first MLX port of Poolside's DFlash draft model.

| Target | Greedy baseline | DFlash range | Mean accepted tokens per step |
|---|---|---|---|
| Resident | 40.3 tok/s | 0.36×–0.86× (9 settings: draft bits 16/8/4 × n_draft 3/7/15) | 1.82–2.51 |
| Streaming, 32 slots | 11.4 tok/s | 1.06× (n=3), 0.65× (n=7), 0.32× (n=15) | 1.84–2.75 |

> **Uncapped MLX buffer pool.** Streaming numbers here were measured with an uncapped MLX buffer pool. For streaming, the capped numbers in "Streaming decode profile (Task S5)" supersede them.

Is the port wrong? The S1 diagnostic says no:

- In every prompt and on both targets, plain greedy and greedy on the
  trimmable cache produce identical outputs.
- Wherever DFlash diverges, the top-2 logit gap is 0–4 bf16 ulps, i.e. near-ties.

So the low speedup comes from the hardware's verify cost, not a bug.

Median verify time per block (`dflash_diag.json`):

| Block length | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|
| Resident (ms) | 25.3 | 32.7 | 40.6 | 49.8 | 83.3 | 96.4 | 118.8 | 132.0 |
| Streaming (ms) | 37.8 | 45.0 | 52.9 | 61.9 | 212.0 | 306.0 | 451.5 | 574.1 |

- **The MoE verify cost** grows with block length because each extra token
  routes to more experts.
- **Streaming hits a cliff** above 4 tokens: the block needs more experts than
  the 32 slots hold, which forces multiple masked passes.
- **Acceptance is low**: 1.8–2.5 per step, against the 3.5–4.6 Poolside reports
  on GPU. Neither problem alone would make DFlash pay off here.

### Prompt-lookup speculative decoding: marginal

Sources: `results/lookup_acceptance.json` (S2, offline, CPU), `results/lookup.json` (S4, real weights).

S2 simulated lookup drafting over all 301,743 assistant tokens, priced with
S1's measured verify costs. Best predicted speedup: **g=4, n=3, fixed** at
2.35 tokens per forward, giving 1.48× resident and 1.70× streaming. Longer
drafts produce more tokens per forward (3.91 at g=2, n=15) but are predicted
slower, because verify is expensive.

S4 measured on real weights: 256 generated tokens after real 8k-token session
prefixes (5 sessions).

> **Uncapped MLX buffer pool.** Streaming numbers here were measured with an uncapped MLX buffer pool. For streaming, the capped numbers in "Streaming decode profile (Task S5)" supersede them.

| Target | Setting | Decode speedup | End-to-end speedup | Mean accepted |
|---|---|---|---|---|
| Resident | g=4, n=3, fixed | 1.22× | 1.00× | 1.23 |
| Resident | g=4, n=3, adaptive | 1.22× | 0.98× | 0.95 |
| Resident | g=4, n=7, fixed | 0.94× | 0.97× | 1.87 |
| Streaming | g=4, n=3, fixed | 1.14× | 1.02× | 1.46 |
| Streaming | g=4, n=3, adaptive | 1.10× | 1.02× | 1.13 |
| Streaming | g=4, n=2, fixed | 1.18× | 1.03× | 1.01 |

On the 10 short prompts, every setting is between 1.01× and 1.06×.

The decode-only speedup clears the plan's 1.1× keep threshold, but end-to-end
time is unchanged, presumably because the 8k-token prefill dominates a
256-token generation. The real-machine gain is well below S2's prediction. The
repo records the numbers but no keep/drop decision.

---

## Part 2: expert pruning

Source: `results/pruning.json` (sweep), `results/prune_point.json` (operating
point), `results/action_diag.json` (action match).

- **Setup:** calibrate on 6 astropy sessions; evaluate on 5 django sessions.
  Each session is capped at 16,384 tokens for the sweep.
- **Reference:** the unpruned 4-bit model (its self-check row scores KL 0,
  top-1 1.0).
- **Metrics:** held-out top-1 agreement and KL mean, in nats.

### Quality vs. fraction of experts pruned (uniform allocation)

| Pruned | REAP | Gate mass | Frequency | Random |
|---|---|---|---|---|
| 12.5% | **0.985** / 0.005 | 0.978 / 0.014 | 0.977 / 0.017 | 0.950 / 0.095 |
| 25% | **0.967** / 0.025 | 0.968 / 0.036 | 0.964 / 0.041 | 0.924 / 0.176 |
| 37.5% | **0.952** / 0.052 | 0.948 / 0.076 | 0.946 / 0.086 | 0.883 / 0.331 |
| 50% | **0.945** / 0.081 | 0.930 / 0.118 | 0.917 / 0.154 | 0.842 / 0.490 |
| 62.5% | **0.909** / 0.187 | 0.890 / 0.265 | 0.881 / 0.343 | 0.782 / 0.697 |

Each cell is top-1 / KL mean.

- **REAP has the lowest KL at every ratio** and the best or tied top-1 (gate mass edges it on top-1 at
  25%, 0.968 vs 0.967). The criterion matters a lot: random pruning of 12.5%
  hurts more than REAP pruning of 37.5%.
- **Global allocation is worse than uniform.** REAP global scores 0.952 / 0.061
  at 25% and 0.928 / 0.125 at 50%. The per-layer flexibility doesn't help, so
  uniform is also the only allocation that can be materialized as a checkpoint.
- **The calibration–held-out gap is real but modest.** REAP at 50% scores KL
  0.047 on calibration sessions and 0.081 on held-out ones.

### Operating point: REAP, uniform, 50%

The spec's default rule picks the largest uniform ratio with held-out top-1 ≥
0.90 and KL ≤ 0.10. That is REAP 50%; 62.5% fails on KL at 0.187. The human
confirmed it.

Materialized checkpoint (`prune_point.json`, short prompts):

| | Full 4-bit | Pruned 50% |
|---|---|---|
| Disk | 18.39 GB | 9.80 GB |
| Resident peak GPU | 18.90 GB | 10.10 GB |
| Resident decode | 34.2 tok/s | 38.7 tok/s |
| Held-out top-1 / KL | (reference) | 0.945 / 0.081 |

> **Uncapped MLX buffer pool.** Streaming numbers here were measured with an uncapped MLX buffer pool. For streaming, the capped numbers in "Streaming decode profile (Task S5)" supersede them. This applies to the streaming row below.
| Streaming, 32 slots: peak / decode / hit rate | | 3.97 GB / 12.8 tok/s / 0.618 |

Why pruned decode is faster was not profiled; smaller expert tensors per layer is the likely reason.
Streaming's hit rate rises because 32 slots now cover 25% of each layer's
experts instead of 12.5%.

### Action match: is the pruned model still a usable agent?

Greedy action match was 0.3 at the operating point, low enough to question the
base. `scripts/diag_actions.py` separated metric harshness from real damage
(`action_diag.json`, 20 tool calls):

| | Exact match | Same tool |
|---|---|---|
| Full model vs. its own references (noise floor) | 20/20 | 20/20 |
| Pruned 50% | 6/20 | 18/20 |

The noise floor is perfect, so the 0.3 is real divergence, not metric noise.

A manual reading of the 14 mismatches (a judgment, not stored in `results/`)
found:

- 2 hard failures: a degenerate repetition loop, and a corrupted file path;
- 3 that differ only in a line-range argument;
- 9 plausible alternative commands.

The human accepted pruned-50 as the base for Part 3 on that basis. n = 20 is
small; a 100-action sample would tighten it (about 1 hour of GPU time, not run).

---

## Part 3: KV cache compression

Source: `results/kv.json`. Spec: `docs/superpowers/specs/2026-10-13-kv-compression-design.md`.

The goal was to run the longest session, 82,671 tokens, on the pruned model
within **12 GB peak GPU**, assumed as 75% of a 16 GB Mac. The quality gate,
against the pruned model with a plain cache, was top-1 ≥ 0.98 and KL mean ≤ 0.02.

Method: quantize K/V on the 10 full-attention layers to 8 or 4 bits. All 5
django sessions were run at full length, and peak is the maximum over prefill
and decode.

The streaming row was measured with an uncapped MLX buffer pool, so its
decode and prefill speed are superseded for streaming. Its quality, and peak
from `mx.get_peak_memory`, which doesn't count the pool, still stand.

| Setting | Peak GB | KV GB | Top-1 | KL mean | KL p99 max | Prefill tok/s | Decode tok/s @82k | Gate |
|---|---|---|---|---|---|---|---|---|
| Resident, plain | 14.60 | 3.39 | 1.0 | 0 | 0 | 95.8 | 7.42 | fail: budget |
| Resident, 8-bit | 13.16 | 1.80 | 0.9963 | 0.0020 | 0.050 | 109.5 | 4.10 | fail: budget |
| Resident, 4-bit | 12.29 | 0.95 | 0.9944 | 0.0038 | 0.120 | 114.5 | 4.99 | fail: budget (+0.29 GB) |
| Streaming, plain (32 slots, uncapped pool) | 8.04 | 3.39 | 1.0 | 0 | 0 | 69.6 | 4.69 | pass |
| **Streaming, plain, 64 slots, 1 GB pool cap** | **10.16** | 3.39 | 1.0 | 0 | 0 | 71.0 | **6.87** | **pass** |

In this table, resident prefill is the mean over the 5 sessions. Streaming
prefill is from the longest session only. On the longest session alone,
resident prefill was 92.6 (plain), 104.0 (8-bit) and 102.4 (4-bit) tok/s.

- **Quality holds at both bit widths.** Every row has top-1 ≥ 0.988 and KL
  mean ≤ 0.015. The budget is what fails.
- **KV compression alone can't reach 12 GB resident.** Peak minus KV is
  11.2–11.4 GB in every resident setting: the 10.1 GB short-prompt footprint
  plus ~1.1–1.3 GB of long-prefill working set. 4-bit already shrinks KV to 0.95 GB.
- **Streaming needs no compression.** At 8.04 GB it is about 4 GB under budget,
  and identical to resident.
- **Quantized KV decodes slower** at 82k (4.1 / 5.0 vs. 7.4 tok/s), because
  `DequantKVCache` re-dequantizes each layer per token (~58 ms per token at
  82k; see below). Prefill is faster with quantized KV; why was not investigated.

**Decision (human, 2026-10-14):** pruned streaming with the plain cache is the
16 GB answer. KV quantization stays in the codebase but is not the operating
point.

**Update (2026-10-15):** with the MLX pool capped at 1 GB, 64 slots fits at
10.16 GB and decodes at 6.87 tok/s at 82k (row `streaming-64slots-cap1.0gb`).
That's the recommended configuration. The full answer, with what remains
unmeasured, is in [part3-answer.md](part3-answer.md).

**Incident during this part:** mlx-lm's quantized attention path materialized
full score matrices (about 16 GB per layer at 82k). The first 8-bit run hung
the machine into a watchdog reboot. `DequantKVCache` replaced that path, and
`bench_kv` now aborts if any row peaks above 16 GB. See runbook.md.

### Why quantized KV decodes slower, and the `quantized_decode` flag

Source: `results/kv_decode_micro.json` (`scripts/micro_kv_decode.py`). It times
one full-attention layer's decode attention on synthetic K/V with the real
shapes: 48 query heads, 8 KV heads, head_dim 128, 8-bit, group size 64, no
weights. The model has 10 such layers. The run finished before moeStream's
GPU job started.

| Per layer, one token | 8k | 32k | 82,671 |
|---|---|---|---|
| Plain bf16 K/V, fused SDPA | 1.97 ms | 2.42 ms | 7.19 ms |
| 8-bit, dequantize then fused SDPA (today) | 3.43 ms | 4.97 ms | 12.98 ms |
| 8-bit, mlx-lm quantized SDPA (`quantized_decode=True`) | 1.04 ms | 3.79 ms | 12.05 ms |
| K/V size, bf16 / 8-bit | 0.034 / 0.018 GB | 0.134 / 0.071 GB | 0.339 / 0.180 GB |

- **The dequantize is the cost.** At 82k, each token reads 0.180 GB of
  packed K/V and writes 0.339 GB of bf16 per layer, then SDPA reads that
  back: about 0.86 GB moved per layer, against 0.339 GB for plain.
  - Measured, it adds 5.8 ms per layer, or **~58 ms per token** over 10 layers.
  - At 32k it adds ~26 ms per token.
- **That explains about half the resident gap.** Resident 8-bit against plain
  at 82k is 244 against 135 ms per token (4.10 against 7.42 tok/s), a 109 ms
  gap.
  - The rest is not measured. The likely cause is buffer churn: the 0.68 GB
    of bf16 temporaries per layer per token must come from MLX's buffer pool,
    and they can't be recycled when the pool is capped or full.
  - moeStream's gap (3.1–3.3 against 6.87 tok/s) comes from a different
    engine, and its plain-cache figure is its own measurement.
- **`quantized_decode=True` (off by default)** is on `DequantKVCache` and
  `make_quantized_cache`.
  - On single-token steps it hands mlx-lm's quantized SDPA the packed K/V
    directly. The cache exposes `bits` for that step only, so it doesn't fall
    into the prefill score-matrix trap above.
  - Prefill is unchanged and bit-identical. Decode matches the dequantize
    path within 1e-3 and gives the same argmax (tiny-model tests in
    `tests/test_cache.py`).
- **Expected gain from the flag:**
  - **82k: small, ~9 ms per token** (0.9 ms per layer). Pinned MLX's quantized
    matmuls at this shape run at ~15 GB/s effective, against ~47 GB/s for
    fused SDPA on bf16. Reading half the bytes doesn't pay off.
  - **32k: ~12 ms per token.**
  - **8k: faster than plain.**
  - It also removes the 0.68 GB of temporaries per layer. So if buffer churn
    is the unmeasured half of the gap, the flag could win more than these
    figures. Only a GPU A/B can tell.
  - Memory: no change.
- **A persistent bf16 copy of the prefix is never worth it.** It would hold
  8-bit plus bf16, 1.80 + 3.39 = 5.19 GB of KV at 82k. The plain cache is
  3.39 GB and already decodes at full speed. Any budget that fits the copy
  fits the plain cache with 1.8 GB to spare.
- **Superseded by the real-model A/B below:** the flag gains far more than
  this single-layer estimate.

### 8-bit decode on the real model: dequantize vs `quantized_decode` vs fused kernel

Sources: `results/kv_qdecode_off.json`, `kv_qdecode_on.json`,
`kv_qdecode_fused.json` (`python3 -m scripts.bench_kv --qdecode off|on|fused`).
Resident pruned-50 model, 8-bit KV, longest django session (82,671 tokens),
64 greedy decode steps after the prefill. One run each, back to back in one
overnight GPU slot.

| Decode path | Decode tok/s @82k | vs dequantize | Decode tokens vs dequantize |
|---|---|---|---|
| Dequantize then fused SDPA (default) | 4.49 | 1.0× | — |
| mlx-lm quantized SDPA (`quantized_decode=True`) | 6.64 | 1.48× | 64/64 identical |
| **Fused 8-bit kernel (`fused_decode=True`)** | **7.59** | **1.69×** | **64/64 identical** |

- **All three have the same peak (13.09 GB) and KV (1.80 GB).** Peak is set by
  the prefill, which is identical in all three, so replay quality is the 8-bit
  row's (top-1 0.9963, KL 0.0020, `kv.json`).
- **The fused kernel** is `laguna_mlx/q8_attn.py`: one `mx.fast.metal_kernel`
  pass per layer reads the packed K/V once, dequantizes in registers and keeps
  an online softmax; small mx ops merge the per-block partials. Nothing
  cache-sized is written. It matches dequantize + SDPA within bf16 rounding
  (`tests/test_q8_attn.py`).
- **At 7.59 tok/s, 8-bit KV with the fused kernel is about as fast as the plain
  cache** (7.42 tok/s in `kv.json`, a different run), with 1.6 GB less KV.
- **The single-layer microbench does not predict this.** Rerun with the kernel
  (`results/kv_decode_micro_fused.json`), the fused kernel alone takes 12.24 ms
  per layer at 82k, about the same as dequantize + SDPA (12.46 ms), and
  slower than mlx-lm's quantized SDPA (10.93 ms). Yet in the model it saves
  ~91 ms per token over the dequantize path (222.7 against 131.8 ms). So most
  of the real gap was not the arithmetic but the 0.68 GB of bf16 temporaries
  per layer per token, as suspected above. That cause is inferred, not
  measured directly.
- **The kernel itself is slow:** 0.18 GB in 12.24 ms is ~15 GB/s effective.
  Each simdgroup walks its 64 keys in sequence with 6 `simd_sum` reductions
  and 12 `exp`s per key, so it is likely latency-bound, not bandwidth-bound
  (not profiled). Tuning it (more keys per lane, fewer
  reductions) could make 8-bit decode faster than plain. Not attempted.
- **Recommendation:**
  - If 1.6 GB more KV fits, the plain cache stays the simplest exact choice.
  - If it doesn't, use 8-bit KV with `fused_decode=True`. Both flags are still
    off by default; flipping the default is the user's call.
  - Resident 8-bit still fails the 12 GB budget (13.09 GB peak, set by
    prefill), so the Part 3 answer (streaming, plain KV) is unchanged.
    For streaming with 8-bit fused KV, see the next section.

### 8-bit fused KV on the streaming path (`serve_streaming --fused-decode`)

Sources: `results/kv_stream_q8.json`, `kv_stream_q8fused.json`,
`kv_stream_plain.json` (`python3 -m scripts.bench_kv --target streaming
--slots 64 --cache-limit-gb 1.0 --qdecode off|fused|plain`). The `laptop16`
configuration (pruned-50 pack, 64 slots, 1 GB pool cap), longest django
session (82,671 tokens), 64 greedy decode steps. One run each, back to back in
one GPU slot, while the laptop was in use. Footprint is the watchdog's peak
process footprint (`results/sf_*.log`).

| KV cache | Decode tok/s @82k | ms/token | Peak GB | Footprint GB | KV GB | Decode tokens |
|---|---|---|---|---|---|---|
| 8-bit, dequantize | 3.87 | 258.4 | 8.72 | 9.62 | 1.80 | — |
| 8-bit, fused kernel | 5.70 | 175.5 | 8.72 | 9.54 | 1.80 | identical to dequantize, 64/64 |
| **Plain (`laptop16` default)** | **6.07** | **164.6** | **10.17** | **11.26** | 3.39 | first differs from 8-bit at step 10 |

- **The fused kernel is 1.47× faster than dequantize on the streaming path,**
  with identical tokens. That's less than resident's 1.69×, as expected: the
  streaming sync floor is the same in every row.
- **Plain is still faster, by 11 ms per token (6%), and exact.** 8-bit KV is
  not exact: greedy decode leaves plain's path after 10 tokens on this
  session. On replay it passes the gate (top-1 0.9963, KL 0.0020 resident,
  `kv.json`), but the Part 3 answer is exact (top-1 1.0, KL 0).
- **8-bit fused saves 1.45 GB of peak** (8.72 against 10.17 GB) and 1.7 GB of
  footprint. Both fit the 12 GB budget.
- **Decision: `laptop16` keeps the plain cache;** `--fused-decode` is the
  opt-in for a machine where those 1.45 GB matter. The rule set beforehand
  was: turn it on only if it matches its own unfused tokens and is at least
  as fast as plain. It meets the first but not the second, and plain is exact.
- Plain streaming here is 6.07 tok/s against 6.87 in `kv.json`. Different
  runs, and the laptop was in use; single runs, no variance measured.

---

## Streaming decode profile (Task S5)

Source: `results/stream_profile.json`. Pruned-50 pack, 32 slots, 32 greedy
decode steps after a prefix of the longest session. All times are ms per
decode step.

> **Uncapped MLX buffer pool.** Streaming numbers here were measured with an uncapped MLX buffer pool. For streaming, the capped numbers in "Streaming decode profile (Task S5)" supersede them. The first table and the resident-control comparison are uncapped. The
option-2 rerun further down is capped.

| Context | tok/s | Step | Router sync | Expert load (wall) | pread (thread-summed) | Other | Hit rate | Misses/step |
|---|---|---|---|---|---|---|---|---|
| 256 | 18.2 | 54.9 | 34.5 (63%) | 16.2 (30%) | 26.0 | 4.1 | 0.73 | 84 |
| 32,768 | 5.9 | 169.9 | 98.6 (58%) | 65.4 (38%) | 154.4 | 5.9 | 0.70 | 95 |

How to read the columns:

- **Router sync** is time blocked in `np.array(indices)`. It includes the GPU
  finishing all queued work (attention and the previous layers), so it bundles
  real compute with round-trip stalls. That's why it triples at 32k: the
  full-attention layers have more context to attend over.
- **Expert load** covers slot bookkeeping, preads and slot writes. pread time is
  summed across 8 threads, so it can exceed wall time.

Readings:

- **Sync is the largest bucket in both runs.** The brief's rule therefore points
  at overlapping the host round-trip.
- **Loading costs 4× more per step at 32k, for similar miss counts.** Per-miss
  pread time rose about 5×. Not investigated; one possibility is expert pages
  falling out of the macOS page cache as the KV cache grows.

### Resident control run

Source: `results/stream_profile.json` → `resident`. Same script, same prefixes,
pruned-50 fully in memory, so there's no sync and no loading.

| Context | Resident step | Streaming step | Streaming overhead | of which: expert load | of which: everything else |
|---|---|---|---|---|---|
| 256 | 23.1 ms (43.2 tok/s) | 54.9 ms | 31.7 ms | 16.2 ms (51%) | 15.6 ms (49%) |
| 32,768 | 46.8 ms (21.4 tok/s) | 169.9 ms | 123.1 ms | 65.4 ms (53%) | 57.7 ms (47%) |

Here "everything else" means streaming step − resident step − expert load.

- **The streaming overhead splits roughly half and half.** About half is expert
  loading. The other half is the extra time inside the sync bucket beyond
  resident compute: the GPU sitting idle while the host routes, plus any
  GPU-side cost streaming adds, such as slot writes. This profile can't
  separate those two.
- **Both halves grow with context** (×4 and ×3.7 from 256 to 32k), while
  resident compute only doubles.
- **Skipping the sync on steps with no misses won't help.** At about 84–95
  misses per step across 39 layers, almost every layer misses on almost every
  step.
- **Prefetch was already tried in moeStream** (sibling project: next-layer
  routing guess, background preads). Its own team reports it about breaks even:
  ~10 ms/token of reads hidden, ~18 ms/token of extra sync added, ~30% wrong
  guesses, rejected by its tuner in 6 of 8 A/B pairs. These are moeStream's
  numbers, not in this repo's `results/`.

### Fix options (for a human pick; nothing run)

Streaming adds 31.7 ms/step at 256 context and 123.1 ms at 32k, about half
expert loading and half everything else. Already ruled out: prefetch (about
break-even in moeStream) and skipping the sync (almost every layer misses).

One more measured fact matters here. In the full-model slot sweep
(`results/streaming.json`, short prompts), going from 16 to 128 slots cut
misses 2.6× (110,906 to 42,928) but decode didn't improve: 10.65, 10.05, 9.92,
9.68 tok/s. So per-step cost isn't driven by miss count alone. One explanation
that fits: each slot write (`param[slot] = arr`, 9 writes per miss) may copy
the whole slot bank, which grows with slot count (32 slots × 1.77 MB per
expert ≈ 57 MB per layer). That is a hypothesis, not a measurement.

| # | Option | Expected gain | Cost / risk |
|---|---|---|---|
| 1 | **Split the load and other buckets** (time preads apart from slot writes; force-eval the writes; compare a cold run with a pre-warmed 32k run) | None directly; it tells which of 2–4 is real | ~1 h code, ~15 min GPU. No risk |
| 2 | **Batch slot writes**: one scatter per tensor per layer, not 9 per miss | If the copy hypothesis holds: part of the load bucket (16.2 / 65.4 ms) and part of "everything else" (15.6 / 57.7 ms). Zero if it doesn't | ~10 lines in `ExpertStore.load`, lossless, covered by the tiny streaming tests. Low risk |
| 3 | **More slots** (64 of the pruned model's 128) | Fewer misses, but the full-model sweep says ~0 decode gain unless 2 lands first | No code; more memory per layer |
| 4 | **Long-context growth** (load ×4 per step for similar misses; other ×3.7) | Up to ~90 ms/step at 32k if it fell back to the 256 cost | Cause unknown (page cache eviction is the suspect); can't size it without 1 |

**Recommendation:** do 1, then 2 if the slot-write cost shows up. They're one
small job. Hold 3 until after 2. Size 4 from 1's warm/cold comparison.

**Option 1 result (partial).** Source: `results/stream_profile.log`, last two
rows (`write_ms`). The JSON wasn't written: the run was stopped by hand during
its third row, the pre-read 32k row, because the machine was starving the
user's other apps (footprint ~17 GB, swap rising).

| Context | Step | Sync | Load (CPU side) | Slot writes (GPU) | Other |
|---|---|---|---|---|---|
| 256 | 61.6 | 30.3 | 15.2 | **11.7** | 4.4 |
| 32,768 | 181.4 | 80.8 | 66.9 | **27.7** | 6.0 |

- **Slot writes are a real cost.** They're about 37% of the streaming overhead
  at 256 context and about 22% at 32k. Forcing them to finish adds a wait per
  layer, so steps here run a little slower than the unforced baseline (54.9 and
  169.9 ms).
- **Option 2 (batched writes) is worth running;** best case it saves about
  these amounts.
- **Option 4 is still unsized,** because the pre-read row was lost.
- **Changes for the rerun:**
  - the script now saves after every row;
  - the wired limit is set only for `--resident`;
  - MLX's buffer cache is cleared before each row;
  - the watchdog footprint cap is 14 GB.

**Option 2 run (`--ab`): stopped by the watchdog.**
- **What happened.** The run hit the 14 GB footprint cap during the first
  model's 32k prefill. Only one row was saved: `stream_profile.json` →
  `ab.per_expert`, 256 context.
  - That row was 72.5 ms/step against 54.9 in the earlier baseline, with sync
    51 ms against 34.5.
  - Two YouTube tabs were playing during this run, so its speed numbers can't
    be compared with earlier runs.
- **Suspected cause (not verified).** MLX keeps freed buffers in a pool, and
  top's footprint counts that pool while `mx.get_peak_memory` doesn't.
  `bench_kv` measured an 8.04 GB peak for the whole 82k streaming session.
- **Changes for the next run:**
  - streaming runs now cap the pool at 1 GB (`mx.set_cache_limit`);
  - each row records `peak_gb` and `cache_after_prefill_gb`.

**Option 2 rerun: batching doesn't help; capping the buffer pool does.**
Source: `results/stream_profile.json` → `ab`. Fresh model per mode, one at a
time, 1 GB pool cap, two YouTube tabs playing throughout.

| Context | Writes | Step | Sync | Load | pread (thread-summed) | Peak GB |
|---|---|---|---|---|---|---|
| 256 | per expert | 55.5 | 37.0 | 14.2 | 19.5 | 4.04 |
| 256 | batched | 72.4 | 48.9 | 17.8 | 23.3 | 4.16 |
| 32,768 | per expert | **84.7** | 64.9 | 15.5 | 22.5 | 5.91 |
| 32,768 | batched | 92.5 | 71.4 | 16.3 | 23.2 | 5.87 |

Generated tokens were identical between the two modes (`tokens_identical:
true`).

- **Batched writes aren't faster:** slower in both rows, though video playback
  adds noise. The batched path was removed and the per-expert writes stay.
- **The 1 GB pool cap alone halved the 32k step:**
  - step 169.9 to 84.7 ms (5.9 to 11.8 tok/s), against the uncapped profile
    above;
  - load 65.4 to 15.5 ms per step;
  - thread-summed pread 154.4 to 22.5 ms.
  The 256 row barely moved (54.9 to 55.5). So the long-context slowdown
  (option 4) was mostly the uncapped MLX pool squeezing the expert files out of
  the page cache, not context itself. Per-step load at 32k is now about the
  same as at 256.
- **What's left at 32k** is mostly sync (64.9 ms): GPU compute plus the
  per-layer round-trip. The resident model takes 46.8 ms per step here.
- **Follow-up done (9c35d6e):** `load_streaming(..., cache_limit_gb=None)` is
  an opt-in cap. `serve_streaming` and `profile_stream` pass 1.0.

### Capped slot sweep

Source: `results/stream_profile.json` → `sweep_capped`. Pruned-50 pack, 1 GB
pool cap, a fresh model per slot count, 32 decode steps. The pruned model has
128 experts per layer, so 128 slots holds every expert. Two YouTube tabs may
have been playing.

| Slots | Context | Step ms | tok/s | Sync | Load | Misses/step | Hit rate | Peak GB |
|---|---|---|---|---|---|---|---|---|
| 16 | 256 | 68.6 | 14.6 | 43.3 | 20.5 | 128.3 | 0.59 | 2.76 |
| 32 | 256 | 58.7 | 17.0 | 38.7 | 15.2 | 84.3 | 0.73 | 4.04 |
| 64 | 256 | 50.8 | 19.7 | 37.7 | 8.5 | 41.6 | 0.87 | 7.09 |
| 128 | 256 | 78.6 | 12.7 | 67.6 | 3.4 | 1.8 | 0.99 | 12.82 |
| 16 | 32,768 | 97.5 | 10.3 | 67.0 | 25.6 | 165.3 | 0.47 | 4.95 |
| 32 | 32,768 | 83.5 | 12.0 | 64.2 | 15.0 | 94.6 | 0.70 | 5.91 |
| 64 | 32,768 | 72.2 | 13.9 | 63.1 | 4.7 | 21.6 | 0.93 | 7.88 |
| 128 | 32,768 | 68.9 | 14.5 | 64.7 | 0.2 | 0 | 1.0 | 12.22 |

- **With the cap, more slots do help.** At 32k the step falls from 97.5 ms at
  16 slots to 68.9 ms at 128, and load falls from 25.6 ms to almost nothing.
  The old flat sweep was an artifact of the uncapped pool.
- **What's left is the sync floor.** Sync stays at 63–65 ms at 32k whatever
  the slot count. At 128 slots nothing is loaded, yet the step is still 22 ms
  slower than the resident model (46.8 ms). That gap is the cost of the
  per-layer host round-trip itself.
- **The 128-slot row at 256 context is an outlier** (sync 67.6 ms). Not
  investigated; the peak was 12.8 GB, the highest of the sweep, and video was
  playing.
- **64 slots is the sweet spot:**
  - 13.9 tok/s at 32k, against 12.0 at 32 slots;
  - 7.9 GB peak, against 5.9 GB at 32 slots;
  - at 82k it would add about 2 GB of full-attention KV, so roughly 10 GB
    peak. That's an estimate from 40,960 bytes/token, not measured. It would
    still be under the 12 GB budget, but with less headroom than 32 slots.
    Worth measuring before switching the 16 GB answer to 64 slots.

### The sync floor (off-GPU analysis)

At 32k the sync bucket stays at 63–65 ms whatever the slot count. At 128 slots
(`sweep_capped`), nothing is loaded and the step is 68.9 ms, against 46.8 ms
for the resident model. Those 22 ms (about 0.56 ms per MoE layer) are the most
any change to the per-layer handoff can win back.

Where those 0.56 ms per layer go, read from `StreamingSwitchGLU.__call__`:

1. `np.array(indices)` blocks until the GPU has finished everything queued so
   far: the attention and router for this layer.
2. Host bookkeeping runs: `np.unique`, `ExpertStore.load`, building the slot
   table.
   - Measured on the CPU alone, with numpy only, as a 128-expert all-hit
     replica of the code: **9.8 µs per layer, 0.38 ms per token.**
   - It's negligible.
3. Python builds the graph for the rest of this layer and the next layer's
   attention and router, then reaches the next sync. **The GPU sits idle
   through all of this,** since nothing is submitted until the next sync. On
   top of that is Metal's per-commit latency.

Candidates:

| # | Change | Expected gain | Risk |
|---|---|---|---|
| A | **`mx.async_eval(out)` at the end of each streamed MoE layer**, so the expert compute runs while Python builds the next layer's graph | Part of the 22 ms: the graph-build time that overlaps. Not sizable off-GPU | Lowest. Same graph, same kernels; bit-exact in the tiny test. **Implemented, off by default:** `StreamingSwitchGLU.async_dispatch` |
| B | `mx.compile` the attention and router blocks, so less Python graph building per layer | Same bucket as A, overlapping with it | Medium: compile has to handle mutable KV caches and shapes that change every step |
| C | Remap expert to slot on the GPU, and sync only an "any miss" flag | ~0: the sync is a data dependency, not a cost of its size | Not worth it |
| D | Skip the sync entirely when slots = experts (static table) | Removes the floor, but needs 12.2 GB, more than the resident pruned model's 10.1 GB, which is faster anyway | Pointless |
| E | Route one layer ahead (prefetch) | Break-even in moeStream's measurements | Already rejected |

**A, measured: slower, and removed.** Source: `results/stream_profile.json` →
`async_ab`. 64 slots, 1 GB pool cap, one process, with async off and then on,
on a fresh model each.

| Context | Async | Step ms | Sync | Load | Other |
|---|---|---|---|---|---|
| 256 | off | 46.8 | 34.5 | 8.2 | 4.0 |
| 256 | on | 77.4 | 46.9 | 17.9 | 12.7 |
| 32,768 | off | 68.7 | 59.8 | 4.7 | 4.2 |
| 32,768 | on | 104.6 | 76.8 | 13.3 | 14.5 |

Tokens were identical (`async_ab_tokens_identical: true`).

- **Async is about 1.5× slower.** Every bucket grows, and "other" roughly
  triples: each extra `async_eval` costs a command-buffer commit.
- **Not checked:** whether load doubles because the GPU is still reading the
  slot buffers that the next miss overwrites.
- **Ruled out by measurement:**
  - candidate A (removed);
  - batched writes.
- **Left:** B (`mx.compile`) was the only remaining candidate, at medium
  risk. The 22 ms gap to resident at 128 slots is its ceiling.

**B, assessed off-GPU: not worth building, not run.** Between two syncs, the
work is layer L's experts, residual, layer L+1's norm, attention and router.

- **The attention part can't be compiled without a retrace every token.**
  mlx-lm's `KVCache.update_and_fetch` writes and slices with Python-int
  offsets (`keys[..., prev:offset, :]`), so each token's graph has different
  constants. `RotatingKVCache` on the sliding layers works the same way. A
  shapeless compile doesn't remove that, because the offsets are graph
  constants, not input shapes.
- **The fixed-shape part is too small to matter.** The decode expert call has
  fixed shapes at any miss count: `x` [1,1,H], slot indices [1,1,8], slot
  weights [slots,…]. It could compile once, with the slot weights passed as
  `inputs=` so slot writes are seen. But it is 3 `gather_qmm` kernels, which
  compile can't fuse, plus SwiGLU. mlx-lm already compiles SwiGLU
  (`activations.swiglu`, `mx.compile(shapeless=True)`).
  - Compiling it saves no kernel launches, only Python dispatch for about 5 ops
    per layer.
  - That's on the order of the 0.38 ms/token measured for all host
    bookkeeping, against the 22 ms gap.
- **The rest of the floor is out of compile's reach.** It is per-sync GPU idle
  time and Metal commit latency, 39 times per token.

**Drop-on-miss (stage 1 of the GPU-side routing note), measured: fails.** At
64 slots, KL 0.146 and top-1 0.937 against the exact path, with 14% of picks
dropped (`results/drop_sim.json`; details in
[gpu-side-routing.md](gpu-side-routing.md)).

**Conclusion:** the sync floor is structural to per-layer host routing. The
next real lever would remove syncs altogether: routing on the GPU with a
device-side expert table and on-GPU miss handling. That is a redesign of the
streaming loader, not a tweak.

### Pending slot writes during prefill (fixed)

Found by moeStream and checked here on the tiny model. mlx-lm's prefill
(`generate_step`, which `serve_streaming` uses) evaluates only the KV caches
after each chunk. Nothing reads the last MoE layer's output. So that layer's
slot writes, and the loaded expert arrays they reference, stayed pending and
piled up chunk after chunk.

- **Tiny model:** active memory grew with every chunk, and evaluating the last
  layer's slot tensors freed 130,864 bytes. Real-weight size wasn't measured;
  moeStream reports ~0.4 GB per chunk in its engine.
- **Fix:** `StreamingSwitchGLU` now calls `mx.async_eval` on its slot tensors
  after every multi-token call. Test:
  `test_prefill_leaves_no_pending_slot_writes`, which failed before the fix.
- **Decode is unchanged,** since its output is always read.
- **This repo's benchmark numbers aren't affected.** `replay.score`,
  `bench_kv`, `profile_stream` and `sim_drop` evaluate the hidden state after
  every chunk. The serving path (`serve_streaming`) was the exposed one.

## Open items

Moved to [open-work.md](open-work.md), which tracks what is still open. Closed
since this list was first written:

- **64 slots at 82k with the pool cap:** measured, 10.16 GB peak and 6.87
  tok/s (`kv.json` → `streaming-64slots-cap1.0gb`); footprint 11.26 GB in a
  later run (`sf_3.log`).
- **The MLX pool cap is opt-in:** `load_streaming(pack_dir, slots,
  cache_limit_gb=None)`; `serve_streaming`, `profile_stream`, `sim_drop` and
  `bench_kv --cache-limit-gb` use 1.0. Older streaming rows stay uncapped, as
  recorded.
- **Task 11, the mlx-lm upstream PR:** moot. Upstream mlx-lm already has
  Laguna (#1334, #1704).
- **`refs/kv` not tied to a model path:** new references record `model`
  (`2fb0bcc`); existing sets are documented in parity-harness.md.
