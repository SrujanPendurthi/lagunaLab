# Runbook

How to rebuild the models, reproduce every result, and keep a 24 GB Mac alive
while doing it. Run every script from the repo root as `python3 -m scripts.<name>`.

## Machine-safety rules (read first)

This machine has rebooted once from a GPU watchdog timeout. These rules come
from that:

1. **One real model in memory at a time.** Never run two weight-loading jobs
   together, and never run a benchmark while `llama-server`, `ollama` or
   `scripts.serve_streaming` is up.
2. **Go through the shared GPU lock.** Any job that loads real weights runs as
   `../.gpu/gpu-run laguna <cmd...>`. It is a mkdir lock, shared with the
   sibling `moeStream` project; lagunaLab has priority. Tiny-model tests don't
   need it. Rules: `../.gpu/RULES.md`.
3. **Watch swap.** Check `sysctl vm.swapusage` before starting. Stop if used
   swap is over ~10 GB.
4. **Use a per-PID watchdog for long runs.**
   - Kill the job if swap exceeds 10 GB, or if its memory footprint exceeds
     20 GB (`top -l 1 -pid <pid> -stats mem`, which counts Metal buffers).
   - Poll with `kill -0 <pid>`. Never use `pgrep -f`: it matches the wrapper
     shell that contains the pattern.
5. **`mx.set_memory_limit` is not a hard cap.** A 2 GB allocation succeeded
   under a 1 GB limit, so don't rely on it for safety.
6. **Resumable scripts save after every row.** Stop at a row boundary, and
   re-run the same command to continue. `bench_kv` also re-checks finished
   rows: if a saved row's peak is over 16 GB, a resume stops instead of moving
   on to longer sessions.
7. **Raise the wired limit for the full resident model** (needs sudo, resets on
   reboot): `sudo sysctl iogpu.wired_limit_mb=22528`. The default working set
   (~19 GB) is barely above the 18.4 GB weights.

## Setup

```bash
pip install mlx==0.32.0 mlx-lm==0.31.3 numpy transformers torch pandas pytest
python3 -m pytest -q        # 188 tests, ~25 s, tiny random-weight models only
```

`torch` is only needed by the tests (Poolside's PyTorch reference); `pandas`
by `lookup_acceptance`. `pool` (Poolside's agent CLI) and SWE-bench Lite are
only needed to capture new sessions.

Don't upgrade mlx or mlx-lm: the port, the cache classes and the quantized-SDPA
workaround are written against these versions.

## Building the models

| Step | Command | Output | Notes |
|---|---|---|---|
| 1. Download bf16 | `hf download poolside/Laguna-XS-2.1 --include "*.safetensors" --local-dir models/Laguna-XS-2.1` | ~66 GB | Needs > 90 GB free disk. Fetch config/tokenizer files too |
| 2. Convert to 4-bit | `python3 scripts/convert.py` | `models/Laguna-XS-2.1-mlx-4bit` (18 GB) | Lazy, shard at a time. Router gates stay 8-bit. Delete the bf16 shards afterwards |
| 3. Pack for streaming | `python3 scripts/pack_experts.py` | `models/Laguna-XS-2.1-packed` | Byte-exact split into resident and per-layer expert files |
| 4. DFlash draft (optional) | `hf download poolside/Laguna-XS-2.1-DFlash --local-dir models/Laguna-XS-2.1-DFlash` | 0.9 GB | Only needed for the DFlash benchmarks |
| 5. Pruned checkpoint | `python3 -m scripts.prune_point --criterion reap --ratio 0.5` | `models/Laguna-XS-2.1-pruned-50` (9.3 GB) and its `-packed` twin | Needs `results/expert_stats.npz` from calibration (below). Also benchmarks the result |

## Serving the model (`scripts/serve_streaming.py`)

An OpenAI-compatible HTTP server: mlx-lm's own `mlx_lm.server`, with the model
swapped for the streaming one. Not a real-weight benchmark, but it loads real
weights, so don't run it alongside a benchmark.

```bash
python3 -m scripts.serve_streaming --preset laptop16 --port 8081                 # the 16 GB answer
python3 -m scripts.serve_streaming --preset laptop16 --fused-decode --port 8081  # 1.45 GB less peak, 6% slower, not exact
python3 -m scripts.serve_streaming --slots 32 --port 8081 \
    --prompt-cache-size 6 --prompt-cache-bytes 4294967296 --max-tokens 4096   # the Task 8b capture setup (full model)
```

Flags this script handles (everything else goes to `mlx_lm.server`):

| Flag | Default | Effect |
|---|---|---|
| `--preset laptop16` | none | pack `models/Laguna-XS-2.1-pruned-50-packed`, 64 slots, `cache_limit_gb` 1.0, plain KV; adds `--max-tokens 4096 --prompt-cache-size 6 --prompt-cache-bytes 4294967296` unless given. Prints the resolved config at start |
| `--slots N` | 32 | expert slots per MoE layer; overrides the preset |
| `--fused-decode` | off | 8-bit K/V on the full-attention layers, decoded by the fused kernel. mlx-lm then serves one request at a time (the cache has no `merge`) |
| `--model PATH` | the pack | added automatically; mlx-lm needs it for the tokenizer and alias |

Without `--preset`, the pack is the full model's `models/Laguna-XS-2.1-packed`
and `cache_limit_gb` is still 1.0.

Server flags that matter (`mlx_lm.server`):

- `--max-tokens 4096`: mlx-lm's default is 512; pool never sends its own
  `max_tokens` and aborts the run on a truncated reply.
- `--prompt-cache-size 6` and `--prompt-cache-bytes 4294967296`: the LRU
  prompt cache keeps the growing conversation's KV between turns; 2 evicted it.
  Its memory on a 16 GB Mac isn't measured.
- `--port`, `--host`: as usual. The model alias is `default_model`.

What the wrapper also does: drops SSE keepalive comment lines (pool's client
fails on them), adds EOS id 24 (`</assistant>`) from `generation_config.json`,
and keeps `mlx_lm.server` from raising the wired limit.

Expected, on the 82,671-token session (`kv.json`, `kv_stream_*.json`, 24 GB
M4 Air): `laptop16` peaks at 10.16–10.17 GB (`mx.get_peak_memory`; process
footprint 11.26 GB) and decodes at 6.07–6.87 tok/s depending on the run;
prefill runs at ~71 tok/s, so a cold 82k prefill takes ~19 minutes. With
`--fused-decode`: 8.72 GB peak, 9.54 GB footprint, 5.70 tok/s. Short-prompt
streaming is much faster (19.7 tok/s at 256 context, 64 slots,
`stream_profile.json` → `sweep_capped`). The first load takes ~85 s.

## Memory budgets

| Configuration | Peak GB (`mx.get_peak_memory`) | Source | Fits 12 GB? |
|---|---|---|---|
| Full 4-bit, resident, short prompts | 18.90 | `prune_point.json` | no |
| Pruned-50, resident, short prompts | 10.10 | `prune_point.json` | yes |
| Pruned-50, resident, 82k, plain / 8-bit / 4-bit KV | 14.60 / 13.16 / 12.29 | `kv.json` | no |
| Pruned-50, streaming 64 slots + 1 GB cap, 82k, plain | 10.16 | `kv.json` | yes |
| Same, 8-bit fused KV | 8.72 | `kv_stream_q8fused.json` | yes |
| Pruned-50, streaming 16 / 32 / 64 / 128 slots, 32k | 4.95 / 5.91 / 7.88 / 12.22 | `stream_profile.json` → `sweep_capped` | yes / yes / yes / no |

- Full-attention KV costs 40,960 bytes per token in bf16 (3.39 GB at 82k),
  21,760 at 8-bit (1.80 GB) and 11,520 at 4-bit (0.95 GB), counting scales and biases. Sliding layers are a
  fixed ~60 MB.
- `mx.get_peak_memory` doesn't count MLX's free-buffer pool; `top`'s footprint
  does. With the 1 GB cap, footprint ran 0.8–1.1 GB above peak (`sf_*.log`).
- The 12 GB budget stands in for a 16 GB Mac (75%); nothing was run on one.

## Capturing agent sessions

Serve the streaming model, put the logging proxy in front of it, then point
`pool` at the proxy:

```bash
python3 -m scripts.serve_streaming --slots 32 --port 8081 \
    --prompt-cache-size 6 --prompt-cache-bytes 4294967296 --max-tokens 4096 > results/serve.log 2>&1 &
python3 -m scripts.gen_traces --n 1     # smoke test: one SWE-bench Lite task
nohup python3 -m scripts.gen_traces --n 12 > traces/run.log 2>&1 &
```

`gen_traces` starts the proxy (listening on 11435, upstream 8081) and runs
`pool` with:

- `POOLSIDE_STANDALONE_BASE_URL=http://127.0.0.1:11435`
- `POOLSIDE_API_KEY=EMPTY`
- `POOLSIDE_STANDALONE_MODEL=default_model` (mlx-lm's alias for the loaded model)
- a 32k context
- a 7,200 s timeout per task

Output is one JSONL per task in `traces/`; reruns skip finished tasks. The first
server load takes ~85 s.

The server flags matter:
- `--prompt-cache-size 6`: a size of 2 evicts the conversation's own KV entry,
  forcing a full re-prefill every turn.
- `--max-tokens 4096`: pool never sets max tokens, mlx-lm's default of 512
  truncates tool calls, and pool aborts the whole run on a truncated
  completion.

mlx-lm infers the tool-call parser from Laguna's chat template.

## Reproducing each result

All of these load real weights, so wrap each in `../.gpu/gpu-run laguna ...`.

| Result file | Command | Time (measured where noted) |
|---|---|---|
| `parity.json` | `python3 -m scripts.check_parity collect` with `llama-server` running the official GGUF, then stop it, then `python3 -m scripts.check_parity score` | not recorded |
| `streaming.json` | `python3 -m scripts.bench_stream --slots 16 32 64 128 --resident` | not recorded |
| `dflash.json` | `python3 -m scripts.bench_dflash --target resident`, then `--target streaming` | not recorded |
| `dflash_diag.json` | `python3 -m scripts.diag_dflash` | not recorded |
| `lookup_acceptance.json` | `python3 -m scripts.lookup_acceptance` | CPU only, no model |
| `lookup.json` | `python3 -m scripts.bench_lookup` | overnight run |
| `expert_stats.npz` | `python3 -m scripts.prune_sweep calibrate` | ~7 min (98,304 tokens at 233 tok/s, measured) |
| `pruning.json` | `python3 -m scripts.prune_sweep refs`, then `python3 -m scripts.prune_sweep sweep` | 525–590 s per row, 23 rows (measured) |
| `prune_point.json` | `python3 -m scripts.prune_point --criterion reap --ratio 0.5` | not recorded |
| `action_diag.json` | `python3 -m scripts.diag_actions --pruned models/Laguna-XS-2.1-pruned-50` | not recorded |
| `kv.json` | `python3 -m scripts.bench_kv`, then `python3 -m scripts.bench_kv --target streaming` | several hours resident (builds `refs/kv` first; the last 3 rows took 51 min); streaming 40 min (measured) |
| `stream_profile.json` → `rows` / `resident` / `sweep_capped` | `python3 -m scripts.profile_stream`; `--resident`; `--slots 16 32 64 128` (the `ab` and `async_ab` modes were removed with their code) | ~10–15 min per mode (estimate) |
| `kv.json` → `streaming-64slots-cap1.0gb` | `python3 -m scripts.bench_kv --target streaming --slots 64 --cache-limit-gb 1.0` | ~20 min (estimate) |
| `drop_sim.json` | `python3 -m scripts.sim_drop` (takes no flags: even `--help` runs it) | not recorded |
| `kv_decode_micro_fused.json` | `python3 -m scripts.micro_kv_decode` (synthetic, < 1 GB, still a GPU job) | ~10 s (measured: step 3 of the 2026-10-16 queue) |
| `kv_qdecode_{off,on,fused}.json` | `python3 -m scripts.bench_kv --qdecode off\|on\|fused --out results/kv_qdecode_<mode>.json` | 11.5–13 min each (measured, `overnight_chain.log`) |
| `kv_stream_{q8,q8fused,plain}.json` | `python3 -m scripts.bench_kv --target streaming --slots 64 --cache-limit-gb 1.0 --qdecode off\|fused\|plain --out results/kv_stream_<name>.json` | ~16–19 min each (measured, `sf_chain.log`) |

`scripts/overnight.sh` chains S1, the Part 2 sweep and S4 unattended. It stops
on a `results/overnight/STOP` file or on high swap. It predates the GPU lock,
so wrap it too: `../.gpu/gpu-run laguna bash scripts/overnight.sh`.

### Ad-hoc replay of any model

```bash
python3 -m replay.replay build-ref --model models/Laguna-XS-2.1-pruned-50 --traces traces --refs refs/mine
python3 -m replay.replay score --model <modified model> --refs refs/mine --out results/mine.json [--actions]
```

From Python, pass a custom cache with `score(model, ref, make_cache=lambda: make_quantized_cache(model, 4))`,
or a non-mlx-lm engine with `score(None, ref, make_cache=..., forward=lambda ids, cache: logits)`.
Details: [parity-harness.md](parity-harness.md).

### Script flags

| Script | Flags (defaults) |
|---|---|
| `replay.replay` | `build-ref\|score`, `--model` (required), `--traces traces`, `--refs refs`, `--out`, `--max-tokens 65536`, `--actions` |
| `bench_kv` | `--target resident\|streaming` (resident), `--slots 32`, `--cache-limit-gb` (none), `--qdecode plain\|off\|on\|fused` (decode A/B on the longest session only), `--out results/kv_qdecode.json` (`--qdecode` only). Without `--qdecode`, resumable rows go to `results/kv.json` |
| `profile_stream` | `--resident`, `--split` (force slot writes inside load; adds a pre-read 32k row), `--slots N [N ...]` (fresh model per count, 1 GB cap) |
| `prune_sweep` | `calibrate\|refs\|sweep`, `--cap 16384` |
| `prune_point` | `--criterion reap`, `--ratio` (required), `--cap 16384`, `--actions 4` |
| `diag_actions` | `--pruned` (required), `--base models/Laguna-XS-2.1-mlx-4bit`, `--refs refs/prune_actions`, `--out results/action_diag.json` |
| `bench_stream` | `--slots 16 32 64 128`, `--resident`, `--max-tokens 128` |
| `bench_dflash` | `--target resident\|streaming`, `--max-tokens 256`, `--slots 32`, `--max-swap-gb 10` |
| `diag_dflash` | `--target resident\|streaming\|both`, `--skip-existing` |
| `bench_lookup` | `--target resident\|streaming\|both`, `--slots 32`, `--max-tokens`, `--max-swap-gb`, `--skip-existing` |
| `check_parity` | `collect\|score` |
| `gen_traces` | `--n 50`, `--work ~/…`, `--traces traces`, `--timeout 7200`, `--proxy-port 11435`, `--server-port 8081`, `--model default_model`, `--context-length 32768` |
| `proxy` | `--listen`, `--upstream`, `--out` (all required) |
| `convert` | `--hf models/Laguna-XS-2.1`, `--out models/Laguna-XS-2.1-mlx-4bit` |
| `pack_experts` | `--model models/Laguna-XS-2.1-mlx-4bit`, `--out models/Laguna-XS-2.1-packed` |
| `sim_drop`, `micro_kv_decode`, `lookup_acceptance` | none |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Other apps stall (video stutters) while a model is loaded | the wired limit was raised: pinned memory can't be paged out | streaming never needs it; scripts raise it only for resident models (`wire_resident`), and `serve_streaming` blocks `mlx_lm.server`'s own raise. Check other tools you run. It resets when the process exits |
| Full resident model swaps heavily or fails to load | default working set (~19 GB) barely above the 18.4 GB weights | `sudo sysctl iogpu.wired_limit_mb=22528` (resets on reboot); or use the pruned model (10.1 GB) |
| Streaming decode slows down sharply at long context | MLX's free-buffer pool grows during prefill and pushes the expert files out of the page cache | pass `cache_limit_gb=1.0` to `load_streaming` (`serve_streaming` does). It halved the 32k step (169.9 to 84.7 ms) |
| `top` shows a footprint well above `mx.get_peak_memory` | `top` counts MLX's buffer pool and other process memory; peak doesn't | judge safety by footprint; cap the pool |
| Memory grows chunk after chunk during a long prefill in a server | the last MoE layer's slot writes stayed pending (mlx-lm evaluates only the caches) | fixed in `c237343`; make sure you're on it |
| Swap climbs past a few GB | two models in memory, or a long resident prefill on 24 GB | stop at a row boundary (scripts are resumable); never run two weight-loading jobs; check `sysctl vm.swapusage` first |
| Machine hangs during a quantized-KV prefill | a cache with a `bits` attribute sent mlx-lm to its quantized SDPA, which builds a ~16 GB score matrix per layer at 82k | use `make_quantized_cache`; never give a cache class `bits` outside single-token steps |
| pool fails with "unexpected end of JSON input" and re-sends the request | mlx-lm's SSE keepalive comments | use `serve_streaming`, which drops them |
| The model repeats `</assistant>` until max tokens | mlx-lm only inferred EOS id 2 | `serve_streaming` adds id 24 |
| pool aborts the run mid-session | a reply truncated at mlx-lm's default 512 tokens | `--max-tokens 4096` (in the preset) |
| Every turn re-prefills the whole conversation | the prompt cache evicted the conversation's own entry | `--prompt-cache-size 6` (in the preset) |
| A plain-cache replay row shows KL ≠ 0 | stale references from a different base model | delete that `refs/` directory and rebuild; new references record their model (`ref_model` in `score`'s output) |
| `--fused-decode` raises "fused_decode needs 8-bit, group_size 64" | the kernel only supports that layout and head_dim 128 | use 8-bit, group 64 |

## Known gotchas

- **Tokenizer warning about `fix_mistral_regex`.** transformers warns that
  Laguna's tokenizer has an incorrect regex pattern. It was checked and is a
  false positive for this tokenizer. **Do not set `fix_mistral_regex=True`**:
  that changes tokenization.
- **llama.cpp short-prompt bug.** Build 10964 returns a constant token with
  null logprobs for Laguna prompts of ~36–512 tokens. `check_parity` flags
  those rows as `ref_degenerate` and gates only on valid rows.
- **mlx-lm quantized attention runs out of memory.** Any cache that has a
  `bits` attribute is routed to `quantized_scaled_dot_product_attention`, which
  materializes `[heads, chunk, context]` scores. That measured 1.64 GB at
  2048 × 8192, and is ~16 GB per layer at 82k. Use
  `laguna_mlx.cache.make_quantized_cache`; its cache class deliberately has no
  `bits` attribute.
- **`max_rss_gb` is meaningless on Apple Silicon.** `resource.getrusage`
  doesn't count Metal buffers. Use `mx.get_peak_memory()` (`peak_gpu_gb`), or
  `top`'s MEM column, for safety decisions.
- **`identical_frac` < 1 doesn't mean a bug.** Streaming, the trimmable cache
  and batched verification all change floating-point summation order. Greedy
  outputs then diverge at exact or near bf16 ties (0–4 ulps measured). The
  per-forward logits still match to 1e-5.
- **Reference truncation.** `build_ref`'s default cap is 65,536 tokens, and the
  longest session is 82,671. `bench_kv` passes `cap=131072` and refuses a
  truncated reference. Do the same in any new full-length experiment.
- **Stale references.** The files under `refs/` aren't tied to the model that
  produced them. After changing the base model, delete the relevant `refs/`
  directory, or the plain-cache row will show KL ≠ 0.
- **Warm page cache.** Streaming numbers measured right after packing may
  benefit from a warm macOS page cache. Purging needs sudo and was not done.

## Tests

`python3 -m pytest -q` runs 188 tests in about 25 s.

- They use tiny random-weight Laguna configs (`tests/tiny.py`), checked against
  Poolside's PyTorch reference in `tests/reference/`.
- `mx.quantize` needs a group size of 32, 64 or 128, so quantized-cache tests
  build the tiny model with `head_dim=64`
  (`mlx_only_model(head_dim=64)`).
- New behaviour gets a tiny test first. Real-weight scripts get a dry-run test
  on the tiny model (see `tests/test_bench_kv.py`, `tests/test_profile_stream.py`).
