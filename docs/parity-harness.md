# Reusing lagunaLab's replay harness (read-only)

This guide is for another repo, moeStream first, that wants to score its own
engine against lagunaLab's references on the captured agent sessions. It
imports from lagunaLab, reads `traces/`, `refs/` and `models/`, and writes
nothing here (cross-repo rule 2). Numbers come from the `results/` file named
next to them; the rest is labelled as an estimate.

## What the harness measures

1. **Teacher-forced next-token quality** (`replay.replay.score`). It feeds a
   session's exact token ids through the model under test and compares the
   log-probs at every assistant position with a stored reference:
   - KL(reference ‖ test), truncated to the reference's top-64 tokens;
   - top-1 agreement.
2. **Greedy action match** (`replay.replay.score_actions` and
   `scripts/diag_actions.py`). At sampled assistant-turn starts, the model
   generates greedily, and the first tool call (or the whole reply, if there's
   no tool call) is compared with the reference's string.

## API

### `replay.session` (stable, rule 8)

```python
load_session(path) -> (messages: list[dict], tools: list | None)
```

- Reads a proxy capture (`traces/*.jsonl`) and returns the last
  `/chat/completions` request, which holds the whole conversation.
- Messages are normalized for Laguna's chat template: list contents are
  joined, tool-call arguments are parsed into dicts (malformed JSON is kept as
  `{"_raw": text}`), and `None` contents become `""`.

```python
tokenize(tokenizer, messages, tools, max_tokens=None) -> (ids int32[n], mask bool[n], truncated: bool)
```

- It applies the chat template with `enable_thinking=True` and
  `return_assistant_tokens_mask=True`.
- It accepts an mlx-lm `TokenizerWrapper` (it unwraps it) or a plain
  Hugging Face tokenizer.
- `mask[i]` is True when token `i` is an assistant token.

### `replay.replay` (stable, rule 11)

```python
assistant_logprobs(model, ids, mask, chunk=2048, make_cache=None, forward=None)
    -> yields (positions int[k], logprobs mx.array float32 [k, V])
```

- It yields the positions whose *next* token is an assistant token: logits
  at `t` predict `t+1`.
- Prefill runs in `chunk`-token pieces on one cache from
  `make_cache or model.make_cache`.
- Log-softmax runs over at most `SUB = 256` positions at a time, about 100 MB
  each.
- **Engine contract, default:** `model.model(ids[None], cache=cache)`
  returns hidden states `[1, L, H]`, and `model.lm_head(h)` returns logits.
  Any mlx-lm `Model` fits, and so does `load_streaming`'s model.
- **Engine contract, other engines:** pass `forward(ids [1, L], cache) ->
  logits [1, L, V]` together with `make_cache`. `model` may then be `None`.
  - This materializes a whole chunk's logits, ~410 MB in bf16 at 2,048 ×
    100k, where the default runs the head only on assistant positions.
  - With `forward=None` the path is unchanged: a self-score is still exactly
    KL 0 and top-1 1.0 (`test_score_with_forward_callable`).

```python
build_ref(model, tokenizer, session_path, out_path, max_tokens=65536, n_actions=2,
          max_new=1024, chunk=2048, seed=0, model_id=None)
```

It writes an `.npz` with these fields:

| Field | Contents |
|---|---|
| `ids`, `mask`, `truncated` | the tokenized session |
| `pos`, `top_ids`, `top_lp` | assistant positions, and the top-64 ids and log-probs at each, sorted |
| `action_starts` | `n_actions` assistant-turn starts, sampled with `seed` |
| `actions` | JSON list of the reference's greedy actions, each up to `max_new` tokens |
| `max_new` | the generation cap used for `actions` |
| `model` | `model_id` (the model's path), `""` if none was passed; refs built before this field have no `model` key |

**`max_tokens=65536` truncates the longest sessions.** Pass 131072 for full
length, as `scripts/bench_kv.build_refs` does; it also refuses a truncated
reference.

```python
score(model, ref_path, chunk=2048, make_cache=None, forward=None) -> dict
```

It returns:

| Key | Meaning |
|---|---|
| `tokens`, `assistant_tokens`, `truncated` | session size, and whether the reference was truncated |
| `kl_top64_mean`, `kl_top64_p99` | top-64 KL against the reference |
| `top1_agree` | top-1 agreement |
| `peak_mem_gb` | `mx.get_peak_memory()` over the run, reset at the start |
| `tok_per_s` | prefill throughput over the whole session |
| `ref_model` | the reference's `model` field, or `None` for refs that predate it |

- It asserts that the positions match the reference, so the ids must be
  identical.
- Summaries in this repo weight by `assistant_tokens` across sessions.

```python
score_actions(model, tokenizer, ref_path, max_new=None) -> float
```

- It returns the exact-match fraction over the reference's actions, or NaN
  if the reference has none.
- It uses `greedy_actions(model, tokenizer, ids, starts, max_new)` (public;
  `_greedy_actions` is kept as an alias). That calls `mlx_lm.generate` with `prompt=ids[:start]`, which re-prefills the
  whole prefix for every action, using `model.make_cache()`.

### Cache factories

The `make_cache` argument takes a zero-argument function returning a list
with one cache per layer:

| Factory | Use |
|---|---|
| `model.make_cache` (default) | plain KVCache on full-attention layers, rotating window on sliding layers |
| `lambda: laguna_mlx.cache.make_trimmable_cache(model, slack=64)` | sliding layers trimmable (speculative decoding) |
| `lambda: laguna_mlx.cache.make_quantized_cache(model, bits, group_size=64, quantized_decode=False, fused_decode=False)` | 8- or 4-bit K/V on full-attention layers (rule 7); `fused_decode` (8-bit only) uses the fused decode kernel |

### `scripts/diag_actions.py`

`python3 -m scripts.diag_actions --pruned <model dir> [--base <model dir>] [--refs <dir>] [--out <json>]`

The defaults are unchanged: `--base models/Laguna-XS-2.1-mlx-4bit`,
`--refs refs/prune_actions`, `--out results/action_diag.json`.

- **What it measures:** greedy action match on `refs/prune_actions` for two
  models: the 4-bit base `models/Laguna-XS-2.1-mlx-4bit` (the noise floor,
  since it built those refs) and the model named by `--pruned`.
- **What it records:** for each action, exact match, same tool name (the
  text between `<tool_call>` and the next `<`), common-prefix length, and
  both strings.
- **Result** (`results/action_diag.json`, 20 actions): base 20/20 exact and
  20/20 same tool; pruned-50 6/20 exact and 18/20 same tool.
- **From another repo, always pass `--out` to a path in your own repo.** The
  default writes `results/action_diag.json` inside lagunaLab (rule 2). Any
  model already in the output file is skipped. The pieces can also be
  imported: `replay.replay.greedy_actions` and `run(path, refs)`.
- **Pointing it at another engine:** `run()` calls `mlx_lm.load(path)`.
  - A packed streaming model needs `load_streaming(pack, slots, cache_limit_gb=1.0)`
    instead.
  - Any other engine has to work with `mlx_lm.generate`, which means
    `model(ids, cache=...)` returns logits and `model.make_cache()` exists.
  - `main()` calls `wire_resident(True)`. Drop that for anything that isn't
    the full resident model.
- **Runtime per session (not recorded; estimate):** each of the 4 actions
  re-prefills up to 16,384 tokens and generates up to 1,024. Resident, that's
  roughly 3–4 minutes per action, about 15 minutes per session per model, so
  ~2.5 h for both models over all 5 sessions. Streaming would be slower.

## Captured sessions

Source: `results/lookup_acceptance.json` → `corpus`. Proxy captures of
Poolside's `pool` agent on SWE-bench Lite tasks are in `traces/*.jsonl`.
`traces/*.pool.json` files are pool's own run records, not inputs to the
harness.

| Session | Role | Tokens | Assistant tokens |
|---|---|---|---|
| astropy__astropy-12907 | calibration | 71,072 | 37,766 |
| astropy__astropy-14182 | calibration | 75,001 | 24,780 |
| astropy__astropy-14365 | calibration | 56,961 | 21,187 |
| astropy__astropy-14995 | calibration | 77,556 | 30,060 |
| astropy__astropy-6938 | calibration | 50,750 | 10,638 |
| astropy__astropy-7746 | calibration | 66,060 | 21,337 |
| django__django-10914 | held-out | 78,112 | 21,740 |
| django__django-10924 | held-out | 82,196 | 28,551 |
| django__django-11001 | held-out | 80,540 | 43,083 |
| django__django-11019 | held-out | 42,771 | 18,397 |
| django__django-11039 | held-out | 82,671 | 44,204 |

The astropy sessions were used to calibrate pruning, so **evaluate on the
django five**.

### References on disk

| Directory | Built from | Sessions | Length | Actions per session |
|---|---|---|---|---|
| `refs/prune/` | 4-bit base (`prune_sweep`) | astropy 12907, 14182; all 5 django | truncated at 16,384 | 0 |
| `refs/prune_actions/` | 4-bit base (`prune_point`) | 5 django | truncated at 16,384 | 4 (`max_new` 1024): 20 in total |
| `refs/kv/` | pruned-50 resident, plain cache (`bench_kv`) | 5 django | full (42,771–82,671), not truncated | 0 |

- **For parity of a pruned-50 engine,** `refs/kv` is the full-length
  reference.
  - Its assistant positions number 21,740 to 44,204 per session.
  - Streaming pruned-50 scores top-1 1.0 and KL 0 against it on
    django__django-11039 (`results/kv.json` → `streaming-64slots-cap1.0gb`).
- **For action match against the full model,** use `refs/prune_actions`
  (16k prefix).
- **Nothing on disk** has full-length action references, or action references
  from pruned-50. Building them takes `build_ref` with `n_actions > 0` and
  GPU time.
- **The references on disk predate the `model` field,** so `score` returns
  `ref_model: None` for them. The table above is their only record. New
  references built through `build_ref(..., model_id=...)` record it; the CLI,
  `bench_kv`, `prune_sweep` and `prune_point` pass it.

## Models on disk (`models/`; read-only, rule 4)

| Directory | What | Size | Load with |
|---|---|---|---|
| `Laguna-XS-2.1` | config and tokenizer only; the bf16 shards were deleted after conversion | 9.7 MB | — |
| `Laguna-XS-2.1-mlx-4bit` | full model, 4-bit, router gates 8-bit | 18 GB | `mlx_lm.load` (needs `import laguna_mlx`) |
| `Laguna-XS-2.1-packed` | the same model, expert-packed for streaming | 18 GB | `load_streaming(pack, slots, cache_limit_gb=1.0)` |
| `Laguna-XS-2.1-pruned-50` | REAP, 50% of experts pruned (128 per layer), 4-bit | 9.3 GB | `mlx_lm.load` |
| `Laguna-XS-2.1-pruned-50-packed` | pruned-50, expert-packed | 9.3 GB | `load_streaming` |
| `Laguna-XS-2.1-DFlash` | DFlash draft model | 881 MB | `laguna_mlx.dflash` |

## Quality gates used here

| Comparison | Gate | Source |
|---|---|---|
| A change that should be lossless or near-lossless (KV quantization, streaming, drop-on-miss), against its own unmodified model | KL top-64 mean ≤ 0.02 **and** top-1 ≥ 0.98, weighted by assistant tokens over the 5 django sessions | `bench_kv` (drop-on-miss used `sim_drop`, removed 2026-10-16) |
| The pruned model against the full model | no fixed gate; the operating point was chosen at KL 0.081, top-1 0.945, action match 0.3 | `results/prune_point.json` |
| Action match | judged against the noise floor (base vs its own references: 20/20) and by same-tool rate, then a human read the mismatches | `results/action_diag.json` |

Measured so far:

- Streaming is **exact**: top-1 1.0, KL 0.
- KV 8-bit and 4-bit pass, at worst top-1 0.9944 and KL 0.0038.
- Drop-on-miss **fails**: KL 0.146, top-1 0.937. Its KL is full-vocabulary,
  not top-64 (`results/drop_sim.json`).

## Pitfalls

- **Tokenizer warning.** transformers warns about `fix_mistral_regex`. It's
  a verified false positive. **Never set `fix_mistral_regex=True`**: it
  changes the token ids, and every `score` call then fails the position
  assertion, or worse, compares different text.
- **bf16 near-ties.** Any change in floating-point summation order
  (streaming, a different kernel, batching, chunk size) flips greedy argmax at
  exact or near bf16 ties (0–4 ulps measured; runbook). Per-forward logits
  still match to 1e-5.
  - Expect top-1 slightly below 1.0 and greedy generations that diverge
    after a tie, even for an exact engine.
  - Treat action exact-match as noisy: run your engine's own noise floor
    (engine against references it built itself) and report same-tool
    alongside it.
- **Use the same chunk size.** References used `chunk=2048`. Prefill chunking
  changes summation order, so a different chunk size moves top-1 slightly.
- **Memory per run** (`mx.get_peak_memory`; it doesn't count MLX's buffer
  pool):
  - Pruned-50 resident at 82,671 tokens: 14.60 GB with a plain cache, 13.16
    with 8-bit KV, 12.29 with 4-bit (`results/kv.json`).
  - Streaming pruned-50, 64 slots, pool capped at 1 GB: 10.16 GB. The
    uncapped 32-slot run was 8.04 GB, but that pool grows unbounded.
  - Full 4-bit base: 18.9 GB on short prompts (`results/prune_point.json`).
    Full-length scoring of the base was never run here; it would add about
    3.4 GB of KV at 82k on top of that (estimate), which doesn't fit safely
    on 24 GB.
  - `score` keeps the whole cache for the session plus one 2,048-token
    chunk's activations. The `SUB` slices add about 100 MB.
- **Rule 6.** Before loading real weights, set a hard cap and run a per-PID
  watchdog:
  - poll with `kill -0 <pid>`, never `pgrep -f`;
  - kill on footprint (`top -l 1 -pid <pid> -stats mem`, which counts Metal
    buffers) and on swap;
  - lagunaLab's wrappers kill at 14 GB footprint or 4 GB swap.
  - `mx.set_memory_limit` is **not** a hard cap: a 2 GB allocation succeeded
    under a 1 GB limit.
- **Rule 9.** For streaming, pass `cache_limit_gb=1.0` to `load_streaming`.
  The default `None` leaves MLX's pool uncapped. During long prefills the
  pool pushes expert pages out of the page cache, which halves decode speed at
  32k, and it inflates the footprint.
- **Never leave the wired limit raised.** `mx.set_wired_limit` lasts for the
  life of the process and pins memory the rest of the machine needs; it
  stalled video playback here.
  - Call `laguna_mlx.wire_resident(resident)`: it raises the limit only when
    `resident` is True, for the 18.4 GB full model.
  - Don't raise it for pruned or streaming runs, and don't keep a process
    alive after its run.
  - A test (`tests/test_wired.py`) blocks raw `set_wired_limit` calls in this
    repo's scripts.
- **One model at a time.** Never hold two models in one process, or run two
  weight-loading jobs together. For a base-vs-engine comparison, build the
  references, free the model (`del model; mx.clear_cache()`), then load the
  next. Go through `../.gpu/gpu-run`. Don't run tests or benchmarks while
  another session holds the lock: it skews their timings.
- **Pending slot writes (fixed in `c237343`).** mlx-lm's prefill evaluates
  only the caches, so a streaming MoE layer's slot writes could stay pending
  across calls. `StreamingSwitchGLU` now runs `async_eval` on them after
  every multi-token call. An engine with its own slot store needs the same
  fix.
- **Prefill speed in `score` isn't a decode speed.** `tok_per_s` is whole
  session over wall time with chunked prefill. Use your own decode loop for
  tok/s (see `bench_kv.decode_at_ctx`).

## Interface changes (2026-10-15, all additive)

1. Rule 11 in `../.gpu/RULES.md` lists `replay.replay`, `replay.session` and
   `scripts.diag_actions` as stable.
2. `replay.replay.greedy_actions` is public; `_greedy_actions` remains.
3. `forward=` on `assistant_logprobs` and `score` (default `None` is today's
   path).
4. `--base`, `--refs` and `--out` on `scripts/diag_actions.py` (defaults
   unchanged).
5. References record `model`; `score` returns `ref_model` and tolerates its
   absence.

Still open: full-length action references for pruned-50 on the 5 django
sessions (`refs/kv` has none). That costs GPU time and needs the
coordinator's go.
