# Architecture

How the pieces fit together, and the reasons behind the non-obvious choices.
For measured outcomes see [results.md](results.md); for commands see
[runbook.md](runbook.md); for the full doc index see [README.md](README.md).

## The model

Laguna XS 2.1 facts the code depends on (from the checkpoint's `config.json`):

| | |
|---|---|
| Parameters | 33B total, ~3B active per token |
| Layers | 40. Layer 0 has a dense MLP; layers 1–39 are MoE |
| MoE | 256 routed experts, top-8, plus 1 shared expert; sigmoid router |
| Attention | 48 query heads, 8 KV heads, head_dim 128, per-head output gate |
| Layer types | Full attention every 4th layer (0, 4, …, 36) = 10 layers; the other 30 use a 512-token sliding window |
| RoPE | Full layers: YaRN, θ = 5e5, factor 32, partial rotary 0.5. Sliding layers: θ = 1e4, full rotary |
| KV cost | Full layers: 10 × 8 heads × 128 × 2 (K, V) × 2 bytes = **40,960 bytes per token**. Sliding layers: fixed, ~60 MB |

At 4-bit the weights are ~18.4 GB on disk, and the routed experts are ~17.4 GB of
that (Part 2 spec). That's why pruning (Part 2) and streaming target the experts, and KV
compression (Part 3) targets only the 10 full-attention layers.

## Repo map

Every tracked file outside `docs/` and `results/`, what it does, and what uses
it. "Real weights" means the script loads a real checkpoint and must run under
`../.gpu/gpu-run laguna` (runbook.md).

### `laguna_mlx/`: the library

| File | What it is | Used by |
|---|---|---|
| `__init__.py` | Importing it registers `model_type: "laguna"` with mlx-lm, so stock `mlx_lm.load` / `convert` / `server` work. Also `wire_resident(resident)`, which raises MLX's wired limit only for a fully resident model, and `mx_without_wired_limit()`, an `mlx.core` stand-in whose `set_wired_limit` is a no-op (see "Wired limit") | every script; moeStream |
| `laguna.py` | The model port: `ModelArgs`, `Attention`, `MLP`, `SparseMoeBlock` (with `route()`), `DecoderLayer`, `LagunaModel` (with layer-output `capture`), `Model` (`make_cache`, `sanitize`, `quant_predicate`) | everything; moeStream (rule 3) |
| `streaming.py` | SSD expert streaming: `ExpertStore` (slot bank per layer), `StreamingSwitchGLU` (replaces each MoE layer's `switch_mlp`), `load_streaming(pack_dir, slots, cache_limit_gb=None)`, `stream_stats(model)` | `serve_streaming`, `bench_*`, `profile_stream`, `diag_dflash`; moeStream (rules 3, 9) |
| `cache.py` | `TrimmableWindowCache` / `make_trimmable_cache` (speculative decoding), `DequantKVCache` / `make_quantized_cache` (Part 3, with the `quantized_decode` and `fused_decode` switches) | `dflash.py`, `bench_kv`, `micro_kv_decode`, `serve_streaming`; moeStream (rule 7) |
| `q8_attn.py` | `q8_decode_attention(q, keys, values, n, scale)`: the fused 8-bit KV single-token attention kernel (`mx.fast.metal_kernel`) | `DequantKVCache.attend`, `micro_kv_decode` |
| `pruning.py` | Part 2: `ExpertSaliencyRecorder`, `accumulate`, `load_stats`, `scores`, `keep_sets`, `apply_mask`, `prune_checkpoint` | `prune_sweep`, `prune_point` |
| `dflash.py` | Speculative decoding: `DFlashDraft`, `load_draft`, `generate_greedy`, `generate_dflash`, `NgramIndex`, `generate_lookup` | `bench_dflash`, `diag_dflash`, `bench_lookup`; `generate_greedy` by moeStream (rule 3) |
| `hooks.py` | `RoutingRecorder`: opt-in per-layer expert-selection counter | tests; Part 1 instrumentation |

### `replay/`: the quality metric

| File | What it is |
|---|---|
| `session.py` | `normalize` (OpenAI wire format to Laguna chat-template messages), `load_session(path)` (messages and tools of a capture's last request), `tokenize(tokenizer, messages, tools, max_tokens)` (ids, assistant mask, truncated flag) |
| `replay.py` | `assistant_logprobs`, `build_ref` (store the reference's top-64 logprobs per assistant position as `.npz`), `score` (KL top-64, top-1, peak, tok/s), `greedy_actions` / `score_actions` (action match), CLI `build-ref` / `score` |

Both are stable for moeStream (rules 8, 11); see [interfaces.md](interfaces.md)
and [parity-harness.md](parity-harness.md).

### `scripts/`: entry points

Run every one as `python3 -m scripts.<name>` from the repo root.

| Script | Part | Real weights | What it does | Writes |
|---|---|---|---|---|
| `convert.py` | 1 | yes | bf16 HF checkpoint to MLX 4-bit, router gates 8-bit | `models/Laguna-XS-2.1-mlx-4bit` |
| `pack_experts.py` | 1 | yes (disk only) | split an MLX checkpoint into resident weights plus one expert file per MoE layer, byte for byte | `models/*-packed` |
| `check_parity.py` | 1 | yes | teacher-force MLX on llama.cpp greedy continuations (`collect`, then `score`); `PROMPTS` is imported by moeStream | `results/llama_ref.json`, `parity.json` |
| `proxy.py` | 1 | no | logging reverse proxy for an OpenAI-compatible server; streaming preserved | `traces/*.jsonl` |
| `gen_traces.py` | 1 | via server | runs `pool` on SWE-bench Lite tasks through the proxy | `traces/` |
| `serve_streaming.py` | 1, 3 | yes | mlx-lm's OpenAI server on the streaming model; `--slots`, `--preset laptop16`, `--fused-decode`; drops SSE keepalives, adds the `</assistant>` EOS, keeps the wired limit down | — |
| `bench_stream.py` | 1 | yes | decode/prefill speed, hit rate, peak across slot budgets | `results/streaming.json` |
| `bench_dflash.py` | 1 (S) | yes | greedy vs DFlash over draft sizes and precisions | `results/dflash.json` |
| `diag_dflash.py` | S1 | yes | DFlash equivalence check and verify-cost timing | `results/dflash_diag.json` |
| `lookup_acceptance.py` | S2 | no (CPU) | offline prompt-lookup acceptance and predicted speedup on the captures | `results/lookup_acceptance.json` |
| `bench_lookup.py` | S4 | yes | prompt-lookup vs a fair greedy baseline on real weights | `results/lookup.json` |
| `overnight.sh` | 1–2 | yes | unattended queue: S1, Part 2 calibrate/refs/sweep, S4 | `results/overnight/` |
| `prune_sweep.py` | 2 | yes | `calibrate` (saliency), `refs` (references), `sweep` (criteria × ratios) | `results/expert_stats.npz`, `expert_stats_meta.json`, `refs/prune/`, `results/pruning.json` |
| `prune_point.py` | 2 | yes | materialize one pruned checkpoint, pack it, benchmark resident and streaming | `models/*-pruned-50[-packed]`, `results/prune_point.json`, `refs/prune_actions/` |
| `diag_actions.py` | 2 | yes | action-match noise floor (base vs its own refs) and pruned divergence | `results/action_diag.json` |
| `bench_kv.py` | 3 | yes | KV quantization sweep (resident/streaming), `--qdecode plain/off/on/fused` decode A/B | `results/kv.json`, `kv_qdecode_*.json`, `kv_stream_*.json`, `refs/kv/` |
| `micro_kv_decode.py` | 3 | no (synthetic, GPU) | one layer's decode attention: plain, dequantize, quantized SDPA, fused kernel | `results/kv_decode_micro_fused.json` (older run: `kv_decode_micro.json`) |
| `profile_stream.py` | S5 | yes | where a streaming decode step goes: sync, load, pread, writes; slot sweep; async A/B | `results/stream_profile.json` |

### `tests/`

`python3 -m pytest -q` collects 183 tests (160 test functions, some
parametrized) and runs in ~25 s on tiny random-weight models. No real weights.

| File | Covers |
|---|---|
| `tiny.py` | Tiny Laguna config (`TINY`), `torch_model` (Poolside's PyTorch reference), `mlx_model`, `mlx_only_model(**overrides)`, `quantized_model_dir` (a 4-bit tiny checkpoint on disk) |
| `reference/` | Poolside's `modeling_laguna.py` and `configuration_laguna.py`, vendored, the ground truth for the port |
| `test_laguna.py` | port matches the PyTorch reference past the sliding window; `sanitize` idempotent; unsupported configs raise; registered with mlx-lm |
| `test_cache.py` | cached decode equals full forward; quantized cache layout, error vs bits, chunking, no score-matrix materialization, `quantized_decode` |
| `test_q8_attn.py` | fused kernel matches dequantize + SDPA at 1 to 1,636 keys; model decode with `fused_decode` matches the dequantize path; 4-bit refused |
| `test_spec_cache.py` | trimmable cache: decode equals full forward, trim after the window wraps, layer capture |
| `test_streaming.py` | pack round-trips bit-exact; streaming equals resident (fewer slots than experts, cached decode, bf16, sorted path); pool cap; prefill leaves no pending slot writes |
| `test_pruning.py` | saliency arithmetic, recorder transparency, keep-sets (uniform, global), masking and undo, materialization |
| `test_prune_sweep.py` | calibrate → refs → sweep end to end on the tiny model; resumability guards |
| `test_replay.py` | self-score is perfect and chunking-invariant; perturbation scores worse; truncation; cache factory; `forward=`; public `greedy_actions` |
| `test_session.py` | wire-format normalization, last-request selection, assistant mask, truncation flag, tokenizer wrapper |
| `test_dflash.py` | DFlash and lookup are lossless (random and oracle drafts, EOS and max-tokens mid-block); `NgramIndex` |
| `test_bench_kv.py`, `test_bench_lookup.py`, `test_profile_stream.py`, `test_diag_dflash.py`, `test_diag_actions.py`, `test_lookup_acceptance.py`, `test_overnight.py` | dry runs of each real-weight script on the tiny model, resume logic, arithmetic |
| `test_proxy.py` | proxy forwards streams, logs errors, survives client disconnects |
| `test_serve_preset.py` | `--preset laptop16` resolution, explicit flags win, `--fused-decode` swaps the cache |
| `test_wired.py` | wired limit only for resident models; `serve_streaming` keeps mlx-lm's server off it |
| `test_hooks.py` | `RoutingRecorder` is transparent and counts every selection |

### Data and model directories (not in git)

| Path | Contents | Made by |
|---|---|---|
| `models/Laguna-XS-2.1` | Poolside's bf16 HF checkpoint (config, tokenizer; shards deleted after conversion) | `hf download` |
| `models/Laguna-XS-2.1-mlx-4bit` | full 4-bit MLX model, 18.39 GB | `convert.py` |
| `models/Laguna-XS-2.1-packed` | its streaming pack | `pack_experts.py` |
| `models/Laguna-XS-2.1-pruned-50` | REAP 50% uniform checkpoint, 128 experts per layer, 9.80 GB | `prune_point.py` |
| `models/Laguna-XS-2.1-pruned-50-packed` | its streaming pack (the `laptop16` model) | `prune_point.py` |
| `models/Laguna-XS-2.1-DFlash` | Poolside's DFlash draft model | `hf download` |
| `traces/` | 11 captured `pool` sessions (`*.jsonl`) plus pool's own logs (`*.pool.json`) | `gen_traces.py` |
| `traces_ollama/` (in git) | one early capture served by ollama, kept for reference | Part 1 |
| `refs/prune` | references from the 4-bit base, 16k-truncated: 2 astropy + 5 django, no actions | `prune_sweep refs` |
| `refs/prune_actions` | 4-bit base, 16k, 5 django, 4 actions each | `prune_point` |
| `refs/kv` | pruned-50 resident, plain cache, full length, 5 django, no actions | `bench_kv` |

`models/` and `traces/` are read by moeStream; don't edit or move them (rule 4).

## Data flow

### Resident (full or pruned)

```
mlx_lm.load(models/<4-bit dir>)            # import laguna_mlx registers the model type
  -> Model (all weights in unified memory; wire_resident(True) raises the wired limit)
  -> model.make_cache(): KVCache on 10 full layers, RotatingKVCache(512) on 30 sliding
  -> forward: embed -> 40 x [attention (+cache) -> MLP or SparseMoeBlock] -> norm -> lm_head
```

No host round-trips inside a forward. Pruned differs only in `num_experts`
(128 instead of 256); the code path is the same.

### Streaming

```
load_streaming(models/<pack>, slots, cache_limit_gb=1.0)
  -> Model with resident.safetensors loaded (attention, router, shared expert, embeddings)
  -> each MoE layer's switch_mlp replaced by StreamingSwitchGLU(ExpertStore(slots))
  -> per MoE layer per forward:
       route on GPU -> np.array(indices)  [GPU->CPU sync, 39 per token]
       np.unique -> ExpertStore.load: hits stay, misses pread from layer_<i>.bin (8 threads),
                    LFU + recency eviction, slot writes
       remap expert ids to slots -> quantized SwitchGLU on the slot bank
```

Output equals the resident model of the same checkpoint (same bytes, same
kernels). Speed is set by the 39 syncs and, below ~64 slots, by misses
(results.md, S5).

### Pruned + streaming + serving (the `laptop16` answer)

```
python3 -m scripts.serve_streaming --preset laptop16
  apply_preset: pack=models/Laguna-XS-2.1-pruned-50-packed, slots=64, cache_limit_gb=1.0,
                fused_decode=False, server flags --max-tokens 4096 --prompt-cache-size 6
                --prompt-cache-bytes 4294967296 (explicit flags win)
  server.load = _load: load_streaming(...) + add EOS 24 ("</assistant>")
                       [+ model.make_cache -> make_quantized_cache(model, 8, fused_decode=True)
                        with --fused-decode]
  server.mx = mx_without_wired_limit()   # mlx_lm.server would otherwise pin the max working set
  APIHandler.setup wraps wfile to drop SSE keepalive comments (pool can't parse them)
  -> mlx_lm.server.main(): OpenAI-compatible HTTP, LRU prompt cache, GLM-4.7 tool parser
```

### Replay scoring (every quality number)

```
traces/<session>.jsonl -> load_session -> tokenize -> ids, assistant mask
reference model -> build_ref -> refs/<set>/<session>.npz (ids, top-64 ids/logprobs, actions, model)
candidate model (+ make_cache= or forward=) -> score -> KL top-64, top-1, peak, tok/s
```

## Model port (`laguna_mlx/laguna.py`)

Written in the shape of an mlx-lm model file (`ModelArgs`, `Model`,
`sanitize`, `quant_predicate`), so `import laguna_mlx` registers
`model_type: "laguna"` and the stock `mlx_lm.load` / `mlx_lm.convert` /
`mlx_lm.server` work unchanged. Plan Task 11 was to upstream it to mlx-lm;
that turned out to be moot: upstream mlx-lm already has its own Laguna
support (mlx-lm #1334, 2026-07-26, and #1704). The pinned mlx-lm 0.31.3 doesn't
include it, so this repo keeps its own port.

Where Laguna differs from Qwen-style MoE, and what the port does:

- **Per-head softplus output gate** (`Attention`): `out * softplus(g_proj(x))`,
  computed in float32.
- **QK RMSNorm** on queries and keys before RoPE.
- **Router** (`SparseMoeBlock.route`): sigmoid scores in float32. The
  `e_score_correction_bias` is added for *selection only*; the gate weights
  come from the unbiased scores. Then `norm_topk_prob`, and the routed output
  is scaled by `moe_routed_scaling_factor` before the shared expert is added.
  `route()` is a separate method on purpose: pruning masks experts by editing
  the bias, and the routing recorder reads its output.
- **Two RoPE configs**: `ModelArgs.rope_for(layer_type)` picks the sliding or
  full one per layer.
- **Caches**: `Model.make_cache()` returns `RotatingKVCache(512)` for sliding
  layers and `KVCache` for full layers. Any cache class can be substituted
  here, which is how both speculative decoding and Part 3 plug in.
- **Hidden-state capture**: `LagunaModel.__call__(..., capture={layer: None})`
  fills in the outputs of chosen layers. DFlash uses it.
- **Quantization**: `quant_predicate` keeps the router gates at 8-bit while
  everything else is 4-bit (group size 64). `sanitize` moves the checkpoint's
  `experts.e_score_correction_bias` onto the MoE block.

Correctness: the port is checked against Poolside's own PyTorch
`modeling_laguna.py` (vendored in `tests/reference/`) on tiny random-weight
configs, and against llama.cpp on the real weights (see results.md for the
caveat on that comparison).

## Replay metric (`replay/`)

The question every experiment asks is: *how much did this change move the
model's behaviour on real agent work?* SWE-bench solve rate is too slow and too
noisy on one laptop, so the metric is **teacher-forced replay of captured agent
sessions**.

1. **Capture** (`scripts/proxy.py`, `scripts/gen_traces.py`,
   `scripts/serve_streaming.py`). Poolside's `pool` agent runs SWE-bench Lite
   tasks against our own streaming MLX model, served through mlx-lm's
   OpenAI-compatible server. A logging reverse proxy records every request as
   JSONL. The last request of a session contains the whole conversation.
2. **Tokenize** (`replay/session.py`). `load_session` takes the final request's
   messages and tools and normalizes OpenAI wire format to what Laguna's chat
   template expects. `tokenize` returns token ids plus an assistant-token mask.
3. **Reference** (`replay.replay.build_ref`). Run the *unmodified* model over the
   session once and store, at each assistant position, the top-64 logprobs and
   the token ids, as an `.npz` under `refs/`.
4. **Score** (`replay.replay.score`). Run the modified model over the same ids,
   in chunks of 2,048 tokens with a KV cache, and report on assistant tokens:
   - KL(reference ‖ modified), truncated to the reference's top-64 (mean and
     p99). This is the primary metric.
   - Top-1 agreement.
   - Peak GPU memory and tokens per second.
   `score(..., make_cache=...)` accepts any cache factory. Part 3 uses that.
5. **Action match** (`score_actions`, `scripts/diag_actions.py`). Greedily
   regenerate a tool call from a session prefix and compare it to the
   reference's. This is a coarser, behaviour-level check.

Why the reference is our own model and not the trace tokens: the traces come
from a sampler and a serving stack, but the reference must use the same
kernels as the modified run so that "no change" scores KL = 0 exactly. Every
sweep includes that self-check row.

## SSD expert streaming (`laguna_mlx/streaming.py`, `scripts/pack_experts.py`)

Only ~3B of 33B parameters are active per token, so most expert weights can
live on the SSD.

- **Pack** (`scripts/pack_experts.py`). Split an MLX checkpoint into
  `resident.safetensors` (attention, router, shared experts, embeddings) plus
  one file per MoE layer with every expert's quantized bytes at a fixed stride
  (`layout.json` describes it). Bytes are copied exactly; nothing is
  re-quantized.
- **Load** (`load_streaming(pack_dir, slots, cache_limit_gb=None)`). Builds the model, swaps each
  layer's `switch_mlp` for a `StreamingSwitchGLU` backed by an `ExpertStore`
  holding `slots` experts, and loads the resident weights.
- **Per forward** (`StreamingSwitchGLU.__call__`): sync the router indices to
  the host, find the unique experts, have `ExpertStore.load` make them
  resident, then run the quantized `SwitchGLU` with indices remapped to slots.
  - Eviction is LFU with recency tie-break.
  - Misses are read with parallel `os.pread` from an 8-thread pool.
  - A prefill chunk needing more experts than there are slots runs several
    masked passes (a `ponytail` note marks this ceiling).
- **Buffer-pool cap.** Pass `cache_limit_gb=1.0` for real runs. Uncapped,
  MLX's free-buffer pool grows during long prefills and pushes the expert files
  out of the macOS page cache, which doubled the 32k decode step (results.md,
  S5).
- **Lossless.** Same bits, same weights, same kernels: streamed logits match
  resident within 1e-5 in the tiny tests. On real weights, greedy outputs can
  still differ at exact bf16 ties, because the summation order across experts
  differs.

Design after TurboFieldfare (drumih/turbo-fieldfare).

Cost: the router indices must reach the host before the experts can be chosen,
so there is one GPU→CPU sync per MoE layer per step (39 per token). Task S5
(`scripts/profile_stream.py`) measured the split. With the pool capped, more
slots do help (fewer misses), and what remains is a sync floor: at 128 slots,
with nothing to load, a 32k step is still 22 ms slower than resident (68.9
against 46.8 ms). Removing it exactly would need GPU-side routing with
GPU-readable expert pages, which pinned MLX can't do from Python
([gpu-side-routing.md](gpu-side-routing.md), [no-gos.md](no-gos.md)).

Other details of `StreamingSwitchGLU`:

- **Prefill slot writes are evaluated** (`mx.async_eval` on the slot tensors
  after every multi-token call). mlx-lm's `generate_step` evaluates only the
  caches after each prefill chunk, so the last layer's writes used to pile up.
- **No drop-on-miss path.** The simulation flag `drop_misses` and
  `scripts/sim_drop.py` were removed 2026-10-16 (user's decision); the last version is in `12e008c` (`git show 12e008c:scripts/sim_drop.py`).

## Expert pruning (`laguna_mlx/pruning.py`)

One-shot, no retraining:

1. **Saliency** (`ExpertSaliencyRecorder`): per layer and expert, routed count,
   gate mass, and the REAP sum `gate_j(x) · ‖f_j(x)‖` over calibration tokens.
   REAP (Cerebras 2025, arXiv 2510.13999) is the mean of that over the tokens
   routed to the expert.
2. **Keep-sets** (`keep_sets`): uniform (same fraction pruned per layer) or
   global (layers compete on normalized saliency, at least 16 kept per layer).
3. **Evaluate by masking** (`apply_mask`): set pruned experts' selection bias to
   −inf, so they are never chosen and the remaining top-8 renormalize. No new
   checkpoint is needed per sweep point, and an unpruned run is bit-identical to
   the reference. `apply_mask` returns an undo function.
4. **Materialize** (`prune_checkpoint`): slice `switch_mlp`, router rows and the
   bias down to the kept experts and write a normal MLX checkpoint with a
   smaller `num_experts`. It loads with plain `mlx_lm.load`, and
   `pack_experts.py` packs it for streaming. Only uniform pruning can be
   materialized, because `num_experts` must be one number.

Calibration uses the 6 astropy sessions; evaluation uses the 5 django
sessions, a different repository. No setting was tuned on the evaluation set.

## Caches (`laguna_mlx/cache.py`)

- **`TrimmableWindowCache` / `make_trimmable_cache`**: a sliding-window cache
  that keeps `window + slack` keys, so speculative decoding can roll back
  rejected draft tokens. mlx-lm's `RotatingKVCache` can't trim after it wraps.
- **`DequantKVCache` / `make_quantized_cache(model, bits, group_size=64,
  quantized_decode=False, fused_decode=False)`**: full-attention layers store
  K/V quantized (8- or 4-bit, group 64) using mlx-lm's `QuantizedKVCache`.
  Sliding layers keep `RotatingKVCache`. Three ways to attend, chosen per call:
  - **default:** dequantize that one layer and hand bf16 K/V to `mx.fast`'s
    fused attention (prefill and decode);
  - **`quantized_decode=True`:** on single-token steps, return the packed
    tuples and expose `bits` for that step only, so mlx-lm's quantized SDPA
    reads them (its score matrix is one row per head at L = 1, so the trap
    below doesn't apply);
  - **`fused_decode=True`** (8-bit, group 64 only): on single-token steps,
    `update_and_fetch` sets `fused_step` and Laguna's `Attention` calls
    `cache.attend(q, scale)` instead of SDPA. That runs
    `laguna_mlx.q8_attn.q8_decode_attention` on the full-capacity packed
    buffers (no slicing, so `metal_kernel` doesn't copy them). Prefill always
    uses the default path.
  - **Why not mlx-lm's quantized attention:** it builds the full
    `[heads, chunk, context]` score matrix, about 16 GB per layer at 82k
    context. That hung the machine and caused a watchdog reboot. The class
    deliberately has no `bits` attribute, because mlx-lm routes any cache that
    has one to that path. The quantization width is exposed as `kv_bits`.
  - **Measured cost of the default path:** decode re-dequantizes the whole
    layer every token: 4.49 tok/s at 82k resident, against 7.59 with
    `fused_decode` (identical tokens; results.md).
  - **No `merge`:** mlx-lm's server treats this cache as non-batchable and
    serves one request at a time. **No `state` setter:** prompt-cache
    save/load to disk would fail (unused).

## Fused 8-bit KV decode kernel (`laguna_mlx/q8_attn.py`)

Flash-decoding over the packed cache, in one `mx.fast.metal_kernel` plus a few
mx ops:

- **Layout it assumes** (mlx `quantize`, 8-bit, group 64, head_dim 128): each
  key or value row is 32 `uint32`, four 8-bit values each, low byte first, with
  2 scales and 2 biases per row. Lane *i* of a simdgroup owns `uint32` *i*,
  i.e. dims 4i–4i+3, which all fall in group *i* / 16.
- **Grid:** one threadgroup per (block of 512 keys, KV head, batch), 8
  simdgroups of 32 lanes each. Each simdgroup walks every 8th key of its block
  and keeps, for each of the G query heads that share the KV head (G = 6 for
  Laguna's 48/8), a running max, sum and 4-dim accumulator: an online softmax
  in float32. Each key costs G `simd_sum` reductions.
- **Merge:** the 8 simdgroups merge through threadgroup memory (8 × G × 128
  floats, guarded to stay under 28 KB), and each block writes (max, sum,
  accumulator). mx ops then merge the blocks: `M = max`, weights
  `exp(m - M)`, output = weighted accumulators / weighted sums.
- **Inputs:** queries pre-scaled in float32; the cache's full-capacity buffers
  plus the valid length `n` as a runtime int32 array, so the kernel doesn't
  recompile as the context grows.
- **Measured:** resident 7.59 tok/s at 82k, against 4.49 for the default
  path; in isolation it runs at only ~15 GB/s effective, so it is probably
  latency-bound (not profiled). See [open-work.md](open-work.md).

## Speculative decoding (`laguna_mlx/dflash.py`)

All variants are greedy and lossless. A verify forward checks
`[token] + drafts` in one pass, accepts the longest matching prefix, and trims
the rejected tail from every cache. Output equals `generate_greedy` up to bf16
ties.

- **DFlash**: Poolside's 5-layer draft model, conditioned on target hidden
  states from layers [1, 13, 25, 33, 39]. This is the first MLX port.
- **Prompt lookup**: `NgramIndex` drafts by copying what followed the most
  recent earlier occurrence of the last g tokens. There is no draft model.
  `adaptive=True` varies the draft length.

Both were measured and documented as negative or marginal results (results.md).
The core reason: verifying k tokens on an MoE touches up to ~k× more experts,
so verify cost rises steeply. Under streaming it rises sharply once a block
needs more experts than the slots hold.

## Instrumentation (`laguna_mlx/hooks.py`)

`RoutingRecorder` counts expert selections per MoE layer while active. Nothing
in the model imports it, so it costs nothing when off.

## Wired limit

`mx.set_wired_limit` pins memory so macOS can't page it out. The full resident
model (18.4 GB) needs it raised, but streaming doesn't, and a raised limit
starved the rest of the machine (it stalled video playback).

- `wire_resident(resident)` raises it to the max recommended working set only
  when `resident` is true. Every script that loads weights goes through it
  (`tests/test_wired.py` checks that).
- `mlx_lm.server.main()` raises it on its own. `serve_streaming` replaces the
  server module's `mx` with `mx_without_wired_limit()`, which ignores only
  that call.
- The limit is per process and released on exit.

## Stable interfaces

A sibling project (`../moeStream`) imports from this repo. The full list, with
signatures and what is safe to change, is in [interfaces.md](interfaces.md).
The shared rules are in `../.gpu/RULES.md`.
