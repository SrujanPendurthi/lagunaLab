# Interfaces other repos rely on

The sibling project `../moeStream` imports this repo read-only and reads its
models, packs, traces and references. The rules are in `../.gpu/RULES.md`;
this page lists every interface they freeze, with its current signature and
the behaviour callers depend on, and says what is safe to change.

**The rule for all of them:** additive changes only. Add a keyword argument
with a default that keeps today's behaviour, add a function, or add a field;
don't rename, remove, reorder positional parameters or change what an
existing call returns. Anything else needs a message to the coordinator
session first (rule 3). Every addition so far followed this: `cache_limit_gb`,
`quantized_decode`, `fused_decode`, `forward=`, `model_id=`, the
`diag_actions` flags.

## Frozen Python interfaces

| Interface | Signature today | Callers rely on | Rule |
|---|---|---|---|
| `laguna_mlx` (import) | — | importing it registers `model_type: "laguna"` with mlx-lm | 3 |
| `laguna_mlx.laguna.ModelArgs` | dataclass; `from_dict(...)`, `rope_for(layer_type)` | config field names as in Poolside's `config.json` (`num_attention_heads_per_layer`, `layer_types`, `rope_parameters`, `moe_routed_scaling_factor`, …) | 3 |
| `laguna_mlx.laguna.Model` | `Model(args)`; `.model` (`LagunaModel`), `.lm_head`, `.layers`, `.args`, `make_cache()`, `sanitize(weights)`, `quant_predicate` | `make_cache()` returns `RotatingKVCache(sliding_window)` for sliding layers and `KVCache` for full ones; `model.model(x, cache=...)` returns hidden states before `lm_head` | 3 |
| `laguna_mlx.laguna.SparseMoeBlock.route` | `route(self, x) -> (inds, w)` | sigmoid router in float32; `e_score_correction_bias` used for selection only; weights normalized (`norm_topk_prob`) | 3 |
| `laguna_mlx.laguna.MLP` | `MLP(dim, hidden)` | SwiGLU, used for layer 0 and the shared expert | 3 |
| `laguna_mlx.streaming.load_streaming` | `load_streaming(pack_dir, slots, cache_limit_gb=None) -> (model, tokenizer)` | output identical to the resident model of the same checkpoint; `cache_limit_gb=None` leaves MLX's cache limit alone (old behaviour) | 3, 9 |
| `laguna_mlx.streaming.stream_stats` | `stream_stats(model) -> dict` | keys `hits`, `misses`, `hit_rate`, `bytes_read`, summed over streamed layers | 3 |
| `laguna_mlx.dflash.generate_greedy` | `generate_greedy(target, prompt, max_tokens, eos=())` | plain greedy decode, the reference for every speculative method | 3 |
| `laguna_mlx.cache.make_quantized_cache` | `make_quantized_cache(model, bits=8, group_size=64, quantized_decode=False, fused_decode=False)` | quantized K/V on full-attention layers only; sliding layers get `RotatingKVCache`; defaults keep the dequantize path | 7 |
| `laguna_mlx.cache.DequantKVCache` | `DequantKVCache(group_size=64, bits=8, quantized_decode=False, fused_decode=False)`; `update_and_fetch`, `offset`, `kv_bits`, `group_size`, `make_mask`, `state`, `is_trimmable`, `trim`, `empty`, `nbytes`; `attend(queries, scale)` with `fused_decode` | **`hasattr(cache, "bits")` is False except on single-token steps with `quantized_decode`**, so mlx-lm never routes prefill to its quantized SDPA (the score-matrix crash). `kv_bits` holds the width | 7 |
| `scripts.pack_experts.pack` | `pack(model_dir, out_dir)` | byte-exact split; output layout below | 3 |
| `scripts.pack_experts.expert_from_bytes` | `expert_from_bytes(buf, layout)` | decodes one expert's bytes from a layer file using `layout.json` | 3 |
| `scripts.check_parity.PROMPTS` | list of the 10 base prompts | content and order | 3 |
| `scripts.serve_streaming` | module; `_DropKeepalive`, `_setup` (patches `mlx_lm.server.APIHandler.setup` at import) | importing it installs the SSE keepalive filter; moeStream imports only that patch | 3, 7, 9 |
| `replay.session.load_session` | `load_session(path) -> (messages, tools)` | messages and tools of a capture's last chat request, normalized | 8, 11 |
| `replay.session.tokenize` | `tokenize(tokenizer, messages, tools, max_tokens=None) -> (ids, mask, truncated)` | ids through Laguna's chat template; `mask` marks assistant tokens | 11 |
| `replay.replay.assistant_logprobs` | `assistant_logprobs(model, ids, mask, chunk=2048, make_cache=None, forward=None)` | yields `(positions, logprobs[n, V] float32)`; `forward(ids[1,L], cache) -> logits[1,L,V]` for other engines, with `make_cache`, `model` may then be None | 11 |
| `replay.replay.build_ref` | `build_ref(model, tokenizer, session_path, out_path, max_tokens=65536, n_actions=2, max_new=1024, chunk=2048, seed=0, model_id=None)` | writes the `.npz` format below | 11 |
| `replay.replay.score` | `score(model, ref_path, chunk=2048, make_cache=None, forward=None) -> dict` | keys `tokens`, `assistant_tokens`, `truncated`, `kl_top64_mean`, `kl_top64_p99`, `top1_agree`, `peak_mem_gb`, `tok_per_s`, `ref_model` (None for references built before `model_id`) | 11 |
| `replay.replay.score_actions` | `score_actions(model, tokenizer, ref_path, max_new=None) -> float` | fraction of the reference's greedy actions reproduced exactly; NaN if the reference has none | 11 |
| `replay.replay.greedy_actions` (alias `_greedy_actions`) | `greedy_actions(model, tokenizer, ids, starts, max_new) -> [str]` | the tool call if there is one, else the whole reply, at each start | 11 |
| `scripts.diag_actions` | `run(path, refs=AREFS)`, `main(argv=None)`, CLI `--pruned`, `--base`, `--refs`, `--out` | defaults unchanged: base `models/Laguna-XS-2.1-mlx-4bit`, refs `refs/prune_actions`, out `results/action_diag.json`; other repos pass `--out` so nothing is written here | 11 |

## Frozen data formats

| Data | Format | Read by |
|---|---|---|
| `models/`, `traces/` | don't edit or move anything (rule 4) | moeStream |
| Pack directory (`*-packed`) | `resident.safetensors` (everything except routed experts); `layout.json` (keys `num_experts`, `stride`, `layers`, `arrays`); `experts/`, one file per MoE layer with every expert's quantized bytes at a fixed stride; plus `config.json`, `generation_config.json`, tokenizer files and `chat_template.jinja` | `load_streaming`, moeStream's engine |
| Reference `.npz` (`refs/*/`) | `ids`, `mask`, `truncated`, `pos`, `top_ids` and `top_lp` (top-64 per assistant position, sorted), `action_starts`, `actions` (JSON list), `max_new`, and since `2fb0bcc` `model` (the path that built it) | `score`, `score_actions`, moeStream's parity eval |
| Capture JSONL (`traces/*.jsonl`) | one proxy record per line: `path`, `request` (the OpenAI chat request), `status`, `response` | `load_session` |

Reference sets, the models that built them and the quality gates are
described in [parity-harness.md](parity-harness.md).

## Safe to change

- Anything not in the tables above: script internals and CLIs other than
  `diag_actions`, private helpers (except `_greedy_actions`, `_DropKeepalive`
  and `_setup`), `bench_*`, `profile_stream`, `sim_drop`, `micro_kv_decode`,
  `laguna_mlx/q8_attn.py` (internal to `DequantKVCache`), `pruning.py`,
  `hooks.py`, tests and docs.
- New files in `results/` and new reference directories. Don't overwrite or
  delete an existing results file another doc cites; write a new one (every
  number in the docs points at a file).
- Defaults of scripts that only this repo runs. `serve_streaming`'s defaults
  without `--preset` have been kept unchanged on purpose, since moeStream
  imports the module.

## Version pins

MLX 0.32.0 and mlx-lm 0.31.3. The port, the cache classes, the
quantized-SDPA workaround and the fused kernel's packed layout are written
against them. Upgrading either is a coordinated change, not a local one.
