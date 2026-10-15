# lagunaLab

Poolside's **Laguna XS 2.1** (33B-total / ~3B-active MoE coding model) running on
Apple Silicon with MLX, plus the measurement harness used to shrink it for a
**16 GB laptop running long agent sessions**.

Everything was built and measured on a 24 GB M4 MacBook Air (MLX 0.32.0,
mlx-lm 0.31.3, Python 3.13). Every number in these docs comes from a file in
[`results/`](results/); the source file is named next to each number.

## Where it ended up

| Configuration | Disk | Peak GPU memory | Decode | Quality vs. its reference |
|---|---|---|---|---|
| Full 4-bit model, resident | 18.4 GB | 18.9 GB | 34.2 tok/s | reference |
| Pruned 50% of experts (REAP), resident | 9.8 GB | 10.1 GB | 38.7 tok/s | top-1 0.945, KL 0.081 vs. full model |
| Pruned, SSD-streamed (32 expert slots/layer) | 9.8 GB | 3.97 GB | 12.8 tok/s | identical to pruned resident |
| Pruned, streamed, **82,671-token session** | 9.8 GB | **8.04 GB** | 4.7 tok/s | identical (top-1 1.0, KL 0) |

Sources: `results/prune_point.json` (short prompts), `results/kv.json` (82k session).

The streaming speeds in this table were measured before the MLX buffer-pool
cap. With `load_streaming(..., cache_limit_gb=1.0)`, the 32k-context streaming
decode step halved (169.9 to 84.7 ms; `results/stream_profile.json`, S5 in
[docs/results.md](docs/results.md)). The memory and quality numbers stand.

**The 16 GB answer:** the pruned model, streamed from SSD, runs the longest
captured agent session (82,671 tokens) at 8.04 GB peak, under the 12 GB budget
assumed for a 16 GB Mac, with no KV-cache compression needed. KV quantization
works for quality (4-bit: top-1 0.994, KL 0.004) but the resident pruned model
still peaks at 12.29 GB, just over that budget.

See [docs/results.md](docs/results.md) for the full findings, including the
negative ones (DFlash and prompt-lookup speculative decoding).

## What's in here

| Path | What |
|---|---|
| `laguna_mlx/laguna.py` | MLX port of Laguna, written as an mlx-lm model file (`import laguna_mlx` registers `model_type: laguna`) |
| `laguna_mlx/streaming.py` | SSD expert streaming: resident common weights, per-layer expert files, LFU slot cache |
| `laguna_mlx/pruning.py` | Expert saliency (REAP, frequency, gate mass), keep-sets, router masking, checkpoint slicing |
| `laguna_mlx/cache.py` | Trimmable sliding-window cache (speculative decoding); quantized KV cache (`DequantKVCache`) |
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

# tests: tiny random-weight models, ~30 s, no weights or GPU budget needed
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

Low-memory streaming version of the same model:

```python
from laguna_mlx.streaming import load_streaming
model, tok = load_streaming("models/Laguna-XS-2.1-pruned-50-packed", slots=32)
```

## Documentation

- [docs/architecture.md](docs/architecture.md): how the port, streaming, pruning,
  caches and the replay metric work, and why they are built that way.
- [docs/results.md](docs/results.md): all findings for Parts 1–3, with sources.
- [docs/runbook.md](docs/runbook.md): reproducing everything, machine-safety rules,
  and known gotchas.
- `docs/superpowers/specs/` and `docs/superpowers/plans/`: the original design
  specs and task-by-task implementation plans (historical record, including
  decisions made along the way).
