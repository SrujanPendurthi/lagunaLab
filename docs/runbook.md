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
python3 -m pytest -q        # ~30 s, tiny random-weight models only
```

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
| `stream_profile.json` | `python3 -m scripts.profile_stream` | ~10–15 min (estimate) |

`scripts/overnight.sh` chains S1, the Part 2 sweep and S4 unattended. It stops
on a `results/overnight/STOP` file or on high swap. It predates the GPU lock,
so wrap it too: `../.gpu/gpu-run laguna bash scripts/overnight.sh`.

### Ad-hoc replay of any model

```bash
python3 -m replay.replay build-ref --model models/Laguna-XS-2.1-pruned-50 --traces traces --refs refs/mine
python3 -m replay.replay score --model <modified model> --refs refs/mine --out results/mine.json [--actions]
```

From Python, pass a custom cache with `score(model, ref, make_cache=lambda: make_quantized_cache(model, 4))`.

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

`python3 -m pytest -q` runs 152 tests in about 30 s.

- They use tiny random-weight Laguna configs (`tests/tiny.py`), checked against
  Poolside's PyTorch reference in `tests/reference/`.
- `mx.quantize` needs a group size of 32, 64 or 128, so quantized-cache tests
  build the tiny model with `head_dim=64`
  (`mlx_only_model(head_dim=64)`).
- New behaviour gets a tiny test first. Real-weight scripts get a dry-run test
  on the tiny model (see `tests/test_bench_kv.py`, `tests/test_profile_stream.py`).
