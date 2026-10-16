# No-gos: what was tried or assessed and not adopted

Every idea that was measured or analysed and then dropped, with the numbers
and the reason. "Measured" means a `results/` file holds the evidence;
"assessed" means it was ruled out by analysis without a GPU run. Code that was
removed is gone from the tree; code that stays (off by default) says so.

## Speed: streaming sync floor

At 32k context a streaming decode step at 128 slots (nothing to load) takes
68.9 ms, against 46.8 ms resident (`stream_profile.json` → `sweep_capped`,
`resident`). Those 22 ms are 39 host round-trips per token, one per MoE layer:
`np.array(indices)` blocks until the GPU finishes, and the GPU idles while
Python routes and builds the next layer's graph.

| Idea | Status | Numbers | Why not |
|---|---|---|---|
| **`mx.async_eval(out)` per streamed MoE layer**, so expert compute overlaps graph building | measured, code removed (`b35250e`) | 64 slots, 1 GB cap: 46.8 → 77.4 ms per step at 256 context, 68.7 → 104.6 at 32k; tokens identical (`stream_profile.json` → `async_ab`) | Every bucket grew and "other" roughly tripled: each extra `async_eval` costs a command-buffer commit. Whether load doubled because the GPU was still reading slot buffers the next miss overwrote wasn't checked |
| **Batched slot writes** (one scatter per tensor per layer instead of 9 writes per miss) | measured, code removed (`2c20ec0`) | 32 slots, 1 GB cap: 55.5 → 72.4 ms at 256, 84.7 → 92.5 at 32k; tokens identical (`ab`) | Slower in both rows. Video playback added noise, but the direction was consistent |
| **`mx.compile` of the decode step** (attention + router) | assessed (`2a68ae1`) | ceiling: the 22 ms floor | mlx-lm's `KVCache` and `RotatingKVCache` write and slice with Python-int offsets, which are graph constants, so the graph changes every token and recompiles. The fixed-shape part (the expert call) is 3 `gather_qmm` kernels compile can't fuse, plus SwiGLU, which mlx-lm already compiles; that would save Python dispatch on the order of the 0.38 ms/token measured for all host bookkeeping |
| **Remap expert to slot on the GPU, sync only an "any miss" flag** | assessed | ~0 | The sync is a data dependency: learning that a miss happened is itself the stall. Its cost is the wait, not the 8 integers |
| **Sync only on steps with no misses** | assessed | 84–95 misses per step at 32 slots; ~21.6 at 64 slots | Almost every layer misses on almost every step |
| **Run optimistically, check once per token, redo on a miss** | assessed ([gpu-side-routing.md](gpu-side-routing.md)) | 21.6 misses in 312 picks per token at 64 slots, 32k | P(no miss in a token) ≈ 0.93^312, effectively zero, so nearly every token would run twice |
| **Static slot table when slots = experts** (no sync at all) | assessed | 128 slots peaks at 12.22 GB at 32k | Needs more memory than the resident pruned model (10.1 GB on short prompts), which is faster anyway |
| **Prefetch: route one layer ahead, read in the background** | measured in moeStream, not here | moeStream's numbers: ~10 ms/token of reads hidden, ~18 ms/token of extra sync, ~30% wrong guesses, rejected by its tuner in 6 of 8 A/B pairs | About break-even |
| **Drop-on-miss**: skip non-resident experts at decode, renormalize, one sync per token | measured by simulation, gate failed (`0505303`); simulation code removed 2026-10-16 | 64 slots, 1 GB cap, 5 sessions to 32k: KL 0.146, top-1 0.937, 14% of picks dropped, 11–25% per layer (`drop_sim.json`) | Fails the quality gate (KL ≤ 0.02, top-1 ≥ 0.98) by a wide margin. The real sync-free path (stage 2) was not built. The simulation (`StreamingSwitchGLU.drop_misses`, `scripts/sim_drop.py`) was removed 2026-10-16 (user's decision); the last version is in `12e008c` (`git show 12e008c:scripts/sim_drop.py`) |
| **Drop on miss but keep the top-1 pick exact** | assessed | — | Checking the top-1 on the host needs the per-layer sync again, so it saves nothing unless the check runs on the device |
| **GPU reads expert pages from an mmap of the pack** (the exact fix) | assessed; not pursued by the user's choice (2026-10-15) | would remove the 22 ms floor | Needs MLX to wrap file-backed memory as a Metal buffer, and GPU-side handling of non-resident pages. Pinned MLX 0.32.0 can't do it from Python; it would take a C++ MLX extension (nanobind and cmake build tooling) |

## Speed: speculative decoding

| Idea | Status | Numbers | Why not |
|---|---|---|---|
| **DFlash** (Poolside's 5-layer draft, first MLX port) | measured (`dflash.json`, `dflash_diag.json`) | resident 0.36×–0.86× of 40.3 tok/s greedy over draft bits 16/8/4 × n_draft 3/7/15; 1.82–2.51 accepted per step (Poolside reports 3.5–4.6 on GPU). Streaming 32 slots: 1.06× at n=3, 0.32× at n=15 | Verify cost on an MoE grows with block length because each token routes to more experts: resident 25.3 ms at 1 token, 132.0 at 16. Under streaming it jumps past 4 tokens (61.9 → 212.0 ms) once a block needs more experts than the slots hold. Not a bug: greedy on the trimmable cache is identical, and DFlash diverges only at 0–4 bf16-ulp near-ties |
| **Prompt lookup** (copy what followed the last g-gram) | measured (`lookup.json`); code stays (`generate_lookup`) | resident g=4 n=3: 1.22× decode, 1.00× end to end; streaming 1.14× / 1.02× (uncapped pool). Offline prediction was 1.48× / 1.70× (`lookup_acceptance.json`) | The decode-only gain doesn't survive end to end on 8k prefixes + 256 generated tokens, where prefill dominates. User decision (2026-10-16): keep the code, unused. It is exact and tested, so it costs nothing to keep in case long generations after short contexts ever matter |
| **Decode pipelining / compile / prefill tuning** (plan Task 15) | not run | — | Replaced by the S1–S5 plan once DFlash failed; `mx.compile` was assessed under S5 (above) |

## Memory: KV cache (Part 3)

| Idea | Status | Numbers | Why not |
|---|---|---|---|
| **mlx-lm's quantized attention for quantized KV** (any cache with `bits`) | measured the hard way: machine hang and GPU watchdog reboot | builds a `[heads, chunk, context]` score matrix: 1.64 GB at 2048 × 8192, ~16 GB per layer at 82k | Unusable for prefill at long context. `DequantKVCache` deliberately has no `bits`; `bench_kv` aborts any row above 16 GB |
| **Resident pruned model with quantized KV** for the 12 GB budget | measured (`kv.json`) | 8-bit 13.16 GB, 4-bit 12.29 GB peak at 82k (quality passes: top-1 0.9963 / 0.9944) | Peak minus KV is 11.2–11.4 GB in every resident row (weights plus long-prefill working set), so even 4-bit KV ends 0.29 GB over |
| **4-bit KV** | measured | 0.95 GB KV; top-1 0.9944, KL 0.0038 | Never needed: resident still fails the budget, and streaming fits without compression |
| **Persistent bf16 shadow copy of the prefix next to the 8-bit cache** | assessed | 1.80 + 3.39 = 5.19 GB of KV at 82k | Any budget that fits the copy fits the plain cache (3.39 GB) with 1.8 GB to spare, and plain is exact and fast |
| **`quantized_decode=True` as the 8-bit decode path** | measured; flag stays, off | resident 82k: 6.64 tok/s vs 4.49 dequantize, identical tokens (`kv_qdecode_*.json`) | Superseded by `fused_decode` (7.59 tok/s) |
| **`fused_decode` (8-bit KV) as the `laptop16` default** | measured; flag stays, off (`--fused-decode`) | streaming `laptop16`: 5.70 tok/s, 8.72 GB vs plain 6.07 tok/s, 10.17 GB; greedy decode leaves plain's path after 10 tokens | Plain is faster and exact; the rule set before the run required at least plain's speed |

## Expert pruning (Part 2)

| Idea | Status | Numbers | Why not |
|---|---|---|---|
| **Global allocation** (layers compete on normalized saliency, ≥ 16 kept) | measured (`pruning.json`) | REAP global 25%: 0.952 / 0.061; 50%: 0.928 / 0.125, against uniform 0.967 / 0.025 and 0.945 / 0.080 | Worse than uniform at both ratios, and can't be materialized (one `num_experts` for all layers). It also kept only the 16-expert minimum in layers 38–39 |
| **Frequency, gate mass, random criteria** | measured | at 50%: frequency 0.917 / 0.154, gate 0.930 / 0.118, random 0.842 / 0.490 | REAP has the lowest KL at every ratio |
| **62.5% pruning** | measured | REAP 0.909 / 0.187 | Fails the KL ≤ 0.10 operating-point rule |

## Other

| Idea | Status | Why not |
|---|---|---|
| **Upstream mlx-lm PR adding Laguna** (plan Task 11) | not filed (2026-10-15) | Upstream already has Laguna (`mlx_lm/models/laguna.py`, #1334 on 2026-07-26 and #1704, with tests), so it would be a duplicate. The pinned 0.31.3 predates it, so this repo keeps its own port |
| **ollama as the capture backend** (Task 8) | replaced by Task 8b | Sessions are captured from this repo's own streaming MLX model through mlx-lm's server, so the captures come from the model under study; one ollama capture is kept in `traces_ollama/` |
| **`--prompt-cache-size 2`** | replaced by 6 + a 4 GiB byte cap | Under mlx-lm's assistant/user/system LRU split, 2 evicted the conversation's own entry before the next turn, forcing a full re-prefill every turn |
| **llama.cpp as the parity reference on short prompts** | worked around | Build 10964 returns a constant token with null logprobs for Laguna prompts of ~36–512 tokens; 20 of 21 rows are degenerate and gated out |
| **`fix_mistral_regex=True` on the tokenizer** | rejected | transformers' warning is a false positive for Laguna; the flag changes tokenization |
