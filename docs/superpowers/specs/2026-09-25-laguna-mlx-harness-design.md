# Laguna XS 2.1 on MLX: Harness Design (Part 1 of 3)

## Goal

Run Poolside's Laguna XS 2.1 on a 24 GB M4 Mac with full visibility into MoE
routing and the KV cache, plus a replay-based metric that scores any model
modification. This harness is the foundation for:

- **Part 2 (B):** expert pruning — shrink weights (experts are ~95% of params).
- **Part 3 (A):** KV cache compression on the 10 full-attention layers.

Combined target story: *Laguna XS on a 16 GB laptop with long agent sessions.*

Parts 2 and 3 get their own specs. This spec covers Part 1 only.

## Facts the design relies on (from `config.json`, verified 2026-09-25)

| | |
|---|---|
| Params | 33B total, ~3B active |
| Layers | 40; layer 0 dense MLP, 1–39 MoE |
| MoE | 256 experts, top-8, `norm_topk_prob`, 1 shared expert (512 inter) |
| Attention | 48 q heads, 8 kv heads, head_dim 128, per-head gating |
| Layer types | full attention every 4th layer (0,4,…,36) = 10; others sliding window 512 |
| RoPE | full: YaRN θ=5e5 factor 32, partial_rotary 0.5; sliding: θ=1e4, full rotary |
| KV cost | full layers 40 KB/token total; sliding layers fixed ~60 MB |

At 4-bit weights ≈ 19 GB, leaving ~4 GB ≈ 100k tokens of full-layer KV. Memory
pressure is real, which is what Parts 2 and 3 attack.

`mlx-lm` 0.31.3 (latest) has no Laguna support. Reference implementation:
`modeling_laguna.py` in the HF repo.

## Components

```
lagunaLab/
  laguna_mlx/model.py      # MLX port of LagunaForCausalLM, mlx-lm model-file shape
  laguna_mlx/hooks.py      # routing recorder + KV cache access
  scripts/convert.py       # bf16 HF shards -> MLX 4-bit, shard-at-a-time
  scripts/check_parity.py  # our logits vs llama.cpp GGUF on fixed prompts
  scripts/gen_traces.sh    # ollama + pool on SWE-bench Lite, log raw prompts
  replay/replay.py         # teacher-forced replay, emits metrics JSON
  results/                 # every number in any writeup is generated from here
```

1. **Model port (`model.py`).** Written as an mlx-lm model file
   (`ModelArgs`, `Model`, `sanitize`) so it can be upstreamed unchanged.
   Covers: dense layer 0, MoE with shared expert via `SwitchGLU`, per-head
   gated attention, two RoPE configs, mlx-lm `RotatingKVCache` for sliding
   layers and `KVCache` for full layers.
2. **Conversion (`convert.py`).** Download bf16 (~66 GB; disk has 106 GB free),
   quantize one shard at a time with lazy loading to stay under 24 GB RAM,
   delete bf16 afterwards. Output ~19 GB 4-bit, group size 64.
3. **Parity check.** 20 fixed prompts (code + chat template). Compare
   next-token top-1 and top-20 overlap against `llama-server` running the
   official Q4 GGUF. Pass: top-1 agreement ≥ 95%, no NaNs, coherent greedy
   generation. (Exact logits will differ: different quantizers.)
4. **Hooks (`hooks.py`).** Opt-in, zero cost when off.
   - Routing recorder: per layer, per token, top-8 expert ids + weights, and
     token role (system / user / tool-output / assistant). Aggregated to
     counts in-memory; raw dump optional.
   - KV access: the per-layer cache objects are already exposed by mlx-lm;
     hooks only add a policy slot on full-attention layers (used in Part 3).
5. **Trace generation.** `ollama launch pool --model laguna-xs-2.1` against
   ~50 SWE-bench Lite tasks, overnight. Capture every raw request ollama
   receives (messages + tools) so replay rebuilds identical token sequences
   via `chat_template.jinja`. Store as JSONL, one session per file.
6. **Replay metric (`replay.py`).** Reference = *our unmodified MLX 4-bit
   model*, not the trace tokens (ollama's quant differs, so trace ≠ reference).
   The trace supplies context only. For each session, teacher-force the full
   token sequence through reference and modified models and report, on
   assistant tokens only:
   - **KL(ref ‖ mod)** mean and p99 — primary.
   - **Top-1 agreement** — secondary.
   - **Action exact-match** on a sample of 100 assistant turns: greedy-decode
     the tool call from the prefix with each model, compare after JSON
     normalisation.
   - Peak memory and tokens/sec.
   Reference logits are cached to disk (top-64 logprobs per token) so each
   experiment runs only the modified model.

## Testing

- Unit: tiny random-weight Laguna config (2 layers, 8 experts) — forward pass
  with and without cache gives identical logits; sliding window masks
  correctly past 512 tokens; hooks on/off give identical logits.
- Parity: `check_parity.py` gate above.
- Replay sanity: reference vs itself → KL 0, agreement 100%.

## Out of scope for Part 1

Pruning, KV policies, SWE-bench end-to-end scoring (final check after Parts
2–3), DFlash speculative decoding, the mlx-lm PR itself (filed once parity
passes).

## Checklist

- [ ] Read `modeling_laguna.py`; note gating and RoPE details the config hides
- [ ] Port model to MLX with tiny-config unit tests
- [ ] Download bf16, convert to 4-bit, delete bf16
- [ ] Parity vs GGUF passes
- [ ] Routing + KV hooks, on/off identity test
- [ ] Install pool, run 1 task end-to-end, confirm prompt capture
- [ ] Overnight trace run, ~50 tasks
- [ ] Replay: reference cache + self-check (KL 0)
- [ ] Baseline numbers in `results/`: memory, tok/s, max context
- [ ] Open mlx-lm PR adding Laguna

## Risks

- **Port subtlety** (gating formula, partial rotary, YaRN): mitigated by the
  parity gate and reading reference code before writing.
- **Conversion OOM:** shard-at-a-time; fallback is converting on a rented
  box for ~$2 once.
- **pool prompt capture:** if pool/ollama don't expose raw requests, run a
  tiny logging proxy on the OpenAI-compatible endpoint.
