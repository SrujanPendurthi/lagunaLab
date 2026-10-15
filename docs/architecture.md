# Architecture

How the pieces fit together, and the reasons behind the non-obvious choices.
For measured outcomes see [results.md](results.md); for commands see
[runbook.md](runbook.md).

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

## Model port (`laguna_mlx/laguna.py`)

Written in the shape of an mlx-lm model file (`ModelArgs`, `Model`,
`sanitize`, `quant_predicate`), so `import laguna_mlx` registers
`model_type: "laguna"` and the stock `mlx_lm.load` / `mlx_lm.convert` /
`mlx_lm.server` work unchanged. It is also meant to be upstreamed to mlx-lm as
is (plan Task 11, not yet filed).

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
so there is one GPU→CPU sync per MoE layer per step (39 per token). Measured
decode speed is flat across slot budgets (results.md), which points at this
sync rather than SSD reads. Task S5 (`scripts/profile_stream.py`) measures the
split.

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
- **`DequantKVCache` / `make_quantized_cache(model, bits)`**: full-attention
  layers store K/V quantized (8- or 4-bit, group 64) using mlx-lm's
  `QuantizedKVCache`. Each call dequantizes that one layer and hands bf16 K/V to
  `mx.fast`'s fused attention. Sliding layers keep `RotatingKVCache`.
  - **Why not mlx-lm's quantized attention:** it builds the full
    `[heads, chunk, context]` score matrix, about 16 GB per layer at 82k
    context. That hung the machine and caused a watchdog reboot. The class
    deliberately has no `bits` attribute, because mlx-lm routes any cache that
    has one to that path. The quantization width is exposed as `kv_bits`.
  - **Known ceiling:** decode re-dequantizes the whole layer every token, so
    quantized-KV decode is slower than plain decode at long context.

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

## Stable interfaces

A sibling project (`../moeStream`) imports from this repo. Keep these
signatures stable; adding to them is fine:

- `laguna_mlx.laguna`: `Model`, `ModelArgs`, `SparseMoeBlock.route`, `MLP`
- `laguna_mlx.streaming`: `load_streaming`, `stream_stats`
- `laguna_mlx.dflash.generate_greedy`
- `laguna_mlx.cache`: `make_quantized_cache`, `DequantKVCache`
- `scripts.pack_experts`: `pack`, `expert_from_bytes`
- `scripts.check_parity.PROMPTS`, `scripts.serve_streaming`
- `replay.session.load_session`

The shared rules are in `../.gpu/RULES.md`.
