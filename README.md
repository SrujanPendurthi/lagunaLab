# lagunaLab

Poolside's **Laguna XS 2.1** (33B-total / ~3B-active MoE coding model) running on
Apple Silicon with MLX, plus the measurement harness used to shrink it for a
**16 GB laptop running long agent sessions**.

Everything was built and measured on a 24 GB M4 MacBook Air (MLX 0.32.0,
mlx-lm 0.31.3, Python 3.13). Every number in these docs comes from a file in
[`results/`](results/); the source file is named next to each number.

## Summary

- **What:** an MLX port of Laguna XS 2.1, SSD expert streaming, expert pruning
  (REAP), KV-cache compression, and a teacher-forced replay harness. The
  harness scores any change against the unmodified model on 11 captured
  `pool` agent sessions (42,771–82,671 tokens each).
- **Goal (Parts 1–3):**
  - Part 1: port the model and set a baseline.
  - Part 2: prune experts.
  - Part 3: run the longest captured session on a **16 GB Mac**, assumed to
    give a 12 GB peak budget.
- **The answer:** the pruned model, SSD-streamed, 64 expert slots per layer,
  MLX buffer pool capped at 1 GB, plain KV cache. On the 82,671-token session
  it peaks at **10.16 GB**, decodes at **6.87 tok/s** (93% of resident), and
  its output is identical to the resident pruned model. One command:
  `python3 -m scripts.serve_streaming --preset laptop16`.

## Results, Parts 1–3

| Configuration | Peak GB | Decode tok/s | Quality vs. its reference | 12 GB budget | Source |
|---|---|---|---|---|---|
| **Short prompts** | | | | | |
| Full 4-bit model, resident (Part 1 baseline) | 18.9 | 34.2 | reference | — | `prune_point.json` |
| Pruned 50% (REAP), resident (Part 2) | 10.1 | 38.7 | top-1 0.945, KL 0.081 vs. full | — | `prune_point.json` |
| Pruned, streamed, 32 slots, uncapped pool | 3.97 | 12.8 | identical to pruned resident | — | `prune_point.json` |
| **82,671-token session (Part 3), pruned model** | | | | | |
| Resident, plain KV | 14.60 | 7.42 | reference | fail | `kv.json` |
| Resident, 8-bit KV | 13.16 | 4.10 | top-1 0.9963, KL 0.0020 | fail | `kv.json` |
| Resident, 4-bit KV | 12.29 | 4.99 | top-1 0.9944, KL 0.0038 | fail (+0.29 GB) | `kv.json` |
| Streamed, 32 slots, uncapped pool (superseded) | 8.04 | 4.69 | top-1 1.0, KL 0 | pass | `kv.json` |
| **Streamed, 64 slots, 1 GB pool cap (`laptop16`)** | **10.16** | **6.87** | **top-1 1.0, KL 0** | **pass** | `kv.json` |

- **KV gate:** KL top-64 mean ≤ 0.02 and top-1 ≥ 0.98 against the pruned model
  with a plain cache, over the 5 held-out django sessions. Both KV widths pass
  on quality. Resident fails on memory: peak minus KV is 11.2–11.4 GB.
- **Quantized KV decode:** the default path re-dequantizes each layer every
  token (4.49 tok/s at 82k). The fused 8-bit kernel (`fused_decode=True`, off
  by default) decodes at 7.59 tok/s with identical tokens, about plain-cache
  speed with 1.6 GB less KV (`kv_qdecode_*.json`; results.md).
- **Pruned-model actions:** greedy action match is 6/20 exact and 18/20 same
  tool, against a noise floor of 20/20 (`action_diag.json`).
- **Parity:** against llama.cpp, the one valid row scored top-1 0.969
  (`parity.json`). Tiny-model tests against Poolside's PyTorch reference
  carry the rest.
- Speeds are comparable only within one results file. Full findings:
  [docs/results.md](docs/results.md). Recommendation and what is unmeasured:
  [docs/part3-answer.md](docs/part3-answer.md).

## Run the 16 GB configuration

```bash
python3 -m scripts.serve_streaming --preset laptop16 --port 8081
```

That is an OpenAI-compatible server on mlx-lm's.

- **The preset:** it resolves to the `models/Laguna-XS-2.1-pruned-50-packed`
  pack, 64 slots, `cache_limit_gb=1.0` and the plain KV cache, and prints them
  at start.
- **Server flags it adds:** `--max-tokens 4096` (pool aborts on a truncated
  reply), `--prompt-cache-size 6` and `--prompt-cache-bytes 4294967296`.
  Flags you pass explicitly win.
- **Wired limit:** the server never raises it.
- **Unmeasured:** serving end to end on an actual 16 GB Mac, including the
  prompt cache's memory, has not been measured. See part3-answer.md.

From Python:

```python
from laguna_mlx.streaming import load_streaming
model, tok = load_streaming("models/Laguna-XS-2.1-pruned-50-packed", slots=64, cache_limit_gb=1.0)
```

## Reuse by moeStream

The sibling `moeStream` project imports this repo read-only: the model, the
streaming loader, the KV caches, and the replay harness for its parity eval.

- How to reuse the harness, references, models and gates:
  [docs/parity-harness.md](docs/parity-harness.md).
- The stable interfaces are listed in `../.gpu/RULES.md`. Changes to them
  are additive only.

## Ruled out (measured)

| Idea | Result | Source |
|---|---|---|
| `mx.async_eval` dispatch per streamed layer | about 1.5× slower (64 slots: 46.8 to 77.4 ms per step at 256 context, 68.7 to 104.6 at 32k), same tokens | `stream_profile.json` → `async_ab` |
| Batched expert-slot writes | slower than per-expert writes | `stream_profile.json` → `ab` |
| `mx.compile` for the decode step | ruled out off-GPU: KV cache offsets retrace every token | results.md, "The sync floor" |
| Drop-on-miss (skip non-resident experts, sync once per token) | fails the gate at 64 slots: KL 0.146, top-1 0.937, 14% of picks dropped | `drop_sim.json` |
| DFlash speculative decoding | 0.36×–0.86× of greedy, resident | `dflash.json` |
| Prompt-lookup speculative decoding | 1.22× decode, 1.00× end to end, resident (8k prefixes); no keep/drop decision | `lookup.json` |

**Drop-on-miss simulation, reproduce:** `python3 -m scripts.sim_drop`. It
uses `StreamingSwitchGLU.drop_misses`, a class flag that is off by default
and changes nothing unless set. Whether to remove the code is the user's
call; it stays for reproducibility. Design and verdict:
[docs/gpu-side-routing.md](docs/gpu-side-routing.md).

Every no-go with its numbers: [docs/no-gos.md](docs/no-gos.md).

## Open, optional work

Full list, with what is unmeasured and the code's known limits:
[docs/open-work.md](docs/open-work.md).

- **Fused 8-bit KV decode kernel: built and measured** (`laguna_mlx/q8_attn.py`).
  1.69× faster decode than dequantize at 82k, identical tokens. Open:
  - the kernel runs at ~15 GB/s effective, so tuning it could beat plain KV;
  - on the streaming path (`serve_streaming --fused-decode`) it decodes at
    5.70 tok/s against plain's 6.07, with 1.45 GB less peak (8.72 against
    10.17 GB). `laptop16` keeps plain: faster and exact (results.md).
- **An exact fix for the streaming sync floor** (not pursued, the user's call:
  it needs a C++ MLX extension). With every expert resident
  (128 slots), streaming is still 22 ms per step slower than resident at 32k
  (68.9 against 46.8 ms; `stream_profile.json`). That's the cost of 39
  host round-trips per token.
- **User decisions pending:** remove the drop-on-miss simulation code; keep or
  drop prompt lookup.
- **Not yet measured:**
  - a real 16 GB Mac;
  - 32 slots with the cap at 82k;
  - a `serve_streaming` end-to-end session;
  - single-run speed variance;
  - a 100-action match sample.

## What's in here

| Path | What |
|---|---|
| `laguna_mlx/laguna.py` | MLX port of Laguna, written as an mlx-lm model file (`import laguna_mlx` registers `model_type: laguna`) |
| `laguna_mlx/streaming.py` | SSD expert streaming: resident common weights, per-layer expert files, LFU slot cache |
| `laguna_mlx/pruning.py` | Expert saliency (REAP, frequency, gate mass), keep-sets, router masking, checkpoint slicing |
| `laguna_mlx/cache.py` | Trimmable sliding-window cache (speculative decoding); quantized KV cache (`DequantKVCache`, with `quantized_decode` and `fused_decode`) |
| `laguna_mlx/q8_attn.py` | Fused 8-bit KV single-token attention kernel (`mx.fast.metal_kernel`) |
| `laguna_mlx/dflash.py` | Greedy decoding, DFlash and prompt-lookup speculative decoding (lossless) |
| `laguna_mlx/hooks.py` | Opt-in routing recorder |
| `replay/` | Captured-session loading and the teacher-forced replay metric (KL, top-1, action match) |
| `scripts/` | Conversion, packing, parity, serving, trace capture, every benchmark and sweep |
| `tests/` | Tiny random-weight tests (no real weights needed), PyTorch reference in `tests/reference/` |
| `results/` | Every measured number, as JSON plus run logs |
| `docs/` | This documentation, plus the design specs and implementation plans under `docs/superpowers/` |

Not in git (see `.gitignore`): `models/` (weights, ~56 GB), `traces/` (captured
sessions), `refs/` (cached reference logprobs), `data/` (SWE-bench Lite).

## Quick start

```bash
# dependencies (pinned versions used here; do not upgrade mlx / mlx-lm)
pip install mlx==0.32.0 mlx-lm==0.31.3 numpy transformers torch pandas pytest

# tests: 188 tests on tiny random-weight models, ~25 s, no weights or GPU budget needed
python3 -m pytest -q
```

Run with real weights (full steps in [docs/runbook.md](docs/runbook.md)):

```python
import laguna_mlx                      # registers the "laguna" model type with mlx-lm
from mlx_lm import load, generate

model, tok = load("models/Laguna-XS-2.1-pruned-50")
prompt = tok.apply_chat_template([{"role": "user", "content": "Reverse a linked list in Python."}],
                                 add_generation_prompt=True, enable_thinking=False)
print(generate(model, tok, prompt=prompt, max_tokens=200))
```

Low-memory streaming version: see "Run the 16 GB configuration" above.

## Documentation

Index with reading order: [docs/README.md](docs/README.md).

- [docs/architecture.md](docs/architecture.md): repo map (every module,
  script and test), data flows, and how the port, streaming, pruning, caches,
  the fused kernel and the replay metric work, and why.
- [docs/history.md](docs/history.md): chronological record of Parts 1–3:
  goals, methods, commands, results, decisions and commits.
- [docs/results-catalog.md](docs/results-catalog.md): every results file with
  its command and commit, and every run's numbers.
- [docs/no-gos.md](docs/no-gos.md): everything tried or assessed and dropped,
  with numbers.
- [docs/interfaces.md](docs/interfaces.md): what moeStream relies on, and what
  is safe to change.
- [docs/open-work.md](docs/open-work.md): open decisions, optional
  engineering, unmeasured items, known limitations.
- [docs/results.md](docs/results.md): all findings for Parts 1–3, with sources.
- [docs/part3-answer.md](docs/part3-answer.md): the 16 GB recommendation and what is unmeasured.
- [docs/gpu-side-routing.md](docs/gpu-side-routing.md): design note for removing the per-layer sync; stage 1 (drop-on-miss) failed the quality gate.
- [docs/parity-harness.md](docs/parity-harness.md): how another repo can reuse the replay harness, references and gates read-only.
- [docs/runbook.md](docs/runbook.md): reproducing everything, machine-safety rules,
  and known gotchas.
- `docs/superpowers/specs/` and `docs/superpowers/plans/`: the original design
  specs and task-by-task implementation plans (historical record, including
  decisions made along the way).
