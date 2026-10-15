# Results

All findings, in the order they were produced. Every number comes from the
`results/` file named next to it; nothing here is extrapolated unless it says
so. Hardware for everything: 24 GB M4 MacBook Air, MLX 0.32.0, mlx-lm 0.31.3.

**Comparing speeds.** Decode tok/s is only comparable *within one results
file*. Different scripts use different prompts, lengths, decode loops, and
machine states. For example, the full resident model measured 18.6 tok/s in
`streaming.json`, 40.3 in `dflash.json` and 34.2 in `prune_point.json`.

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

| Target | Decode tok/s | Prefill tok/s | Peak GPU GB | Expert hit rate |
|---|---|---|---|---|
| Resident | 18.6 | 40.6 | 19.81 | n/a |
| Streaming, 16 slots | 10.7 | 52.2 | 3.70 | 0.314 |
| Streaming, 32 slots | 10.1 | 71.0 | 4.66 | 0.405 |
| Streaming, 64 slots | 9.9 | 90.1 | 7.06 | 0.527 |
| Streaming, 128 slots | 9.7 | 115.4 | 12.80 | 0.735 |

At 16k context, streaming decode drops to 4.4 tok/s at both 16 and 32 slots.

- **Decode speed is flat across slot budgets** even though the hit rate more
  than doubles. So SSD reads are not the decode bottleneck. The suspect is the
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

| Setting | Peak GB | KV GB | Top-1 | KL mean | KL p99 max | Prefill tok/s | Decode tok/s @82k | Gate |
|---|---|---|---|---|---|---|---|---|
| Resident, plain | 14.60 | 3.39 | 1.0 | 0 | 0 | 95.8 | 7.42 | fail: budget |
| Resident, 8-bit | 13.16 | 1.80 | 0.9963 | 0.0020 | 0.050 | 109.5 | 4.10 | fail: budget |
| Resident, 4-bit | 12.29 | 0.95 | 0.9944 | 0.0038 | 0.120 | 114.5 | 4.99 | fail: budget (+0.29 GB) |
| **Streaming, plain** | **8.04** | 3.39 | 1.0 | 0 | 0 | 69.6 | 4.69 | **pass** |

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
  `DequantKVCache` re-dequantizes each layer per token. Prefill is faster with
  quantized KV; why was not investigated.

**Decision (human, 2026-10-14):** pruned streaming with the plain cache is the
16 GB answer. KV quantization stays in the codebase but is not the operating
point.

**Incident during this part:** mlx-lm's quantized attention path materialized
full score matrices (about 16 GB per layer at 82k). The first 8-bit run hung
the machine into a watchdog reboot. `DequantKVCache` replaced that path, and
`bench_kv` now aborts if any row peaks above 16 GB. See runbook.md.

---

## Streaming decode profile (Task S5)

Source: `results/stream_profile.json`. Pruned-50 pack, 32 slots, 32 greedy
decode steps after a prefix of the longest session. All times are ms per
decode step.

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

## Open items

- **S5 follow-up: no fix picked yet.** See "Fix options" above; recommendation
  is to measure first (option 1), then batch slot writes (option 2).
- **Task 11: mlx-lm upstream PR** adding Laguna. Not filed; it is
  outward-facing, so it needs a human go-ahead.
- **Prompt lookup:** keep/drop decision not recorded (see above).
- **Action match** on a larger sample (100 actions).
- **Deferred minors** from the Part 3 review:
  - `refs/kv` references aren't tied to a model path;
  - `decode_peak_gpu_gb` actually covers prefill plus decode;
  - `DequantKVCache.state` has no setter, so prompt-cache save/load would fail
    (unused today).
