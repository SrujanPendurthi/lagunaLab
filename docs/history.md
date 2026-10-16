# History: what was done, in order, and why

A chronological record of the project, part by part: the goal, the method and
its parameters, the exact commands, the results, and what was decided. Dates
and commit hashes come from `git log`. Every number comes from the `results/`
file named next to it; [results.md](results.md) has the full tables, and
[results-catalog.md](results-catalog.md) maps every file to the command and
commit that made it.

Hardware throughout: 24 GB M4 MacBook Air, MLX 0.32.0, mlx-lm 0.31.3,
Python 3.13. Specs and plans live in `docs/superpowers/specs/` and
`docs/superpowers/plans/`.

**Target story** (Part 1 spec): *Laguna XS 2.1 on a 16 GB laptop running long
agent sessions.* Part 1 builds the harness, Part 2 shrinks the weights, Part 3
fits the KV cache.

---

## Part 1: harness, port and baseline (2026-09-25 to 2026-09-27)

Spec `2026-09-25-laguna-mlx-harness-design.md`, plan
`2026-09-25-laguna-mlx-harness.md` (Tasks 1–18, 8b, S1–S5).

**Goal.** Run Laguna XS 2.1 on the 24 GB Mac with full visibility into MoE
routing and the KV cache, plus a replay metric that scores any model change.
mlx-lm 0.31.3 had no Laguna support.

### The port (Tasks 1–4)

| Task | Commit | What |
|---|---|---|
| 1 | `73b5276` | `laguna_mlx/laguna.py`, in mlx-lm model-file shape. Exact against Poolside's PyTorch `modeling_laguna.py` (vendored in `tests/reference/`) on a tiny random-weight config |
| 2 | `e95ed35` | cached decode equals the full forward across the 512-token sliding window |
| 3 | `5e359af` | `RoutingRecorder` hook |
| 4 | `c8686ab` | `scripts/convert.py`: bf16 HF shards to MLX 4-bit, group 64, router gates 8-bit (`quant_predicate`) |

Commands:

```bash
hf download poolside/Laguna-XS-2.1 --include "*.safetensors" --local-dir models/Laguna-XS-2.1
python3 scripts/convert.py            # -> models/Laguna-XS-2.1-mlx-4bit (18.39 GB on disk)
```

### Parity against llama.cpp (Task 5, `af6bc4c`, `6a9d9c4`)

- **Method:** teacher-force the MLX model on llama.cpp's greedy continuations
  of 21 prompts (10 base prompts × thinking on/off, plus one long document).
  Gate: top-1 ≥ 0.95 and in-top-5 ≥ 0.99.
- **Commands:** `python3 -m scripts.check_parity collect` with
  `llama-server -m <gguf> --port 8080 -c 16384` running, then stop it and run
  `python3 -m scripts.check_parity score`.
- **Result** (`parity.json`): llama.cpp build 10964 returns a constant token
  with null logprobs for Laguna prompts of ~36–512 tokens, so 20 of 21
  references are degenerate. The one valid row (`long_doc`, 9,782 tokens)
  scored top-1 0.969, in-top-5 1.0.
- **Decision:** the script flags degenerate rows (`ref_degenerate`) and gates
  only on valid ones. Confidence in the port rests mostly on the PyTorch tiny
  tests and on replay self-checks scoring KL = 0.

### Session capture (Tasks 6–8, 8b)

- **Task 6** (`913f1fe` … `6e80a74`): `scripts/proxy.py`, a logging reverse
  proxy that preserves streaming and logs every request as JSONL. Four fixes
  for exception handling, timeouts and client disconnects.
- **Task 7** (`42544ed`, `0fc4934`, `c45cad6`): `replay/session.py` turns a
  capture into Laguna chat-template ids plus an assistant mask.
- **Task 8** (`584a40d`): `scripts/gen_traces.py` runs Poolside's `pool` agent
  on SWE-bench Lite. First backend: ollama (one capture kept in
  `traces_ollama/`).
- **Task 8b** (`a87a76f`, `d7cdc0b`): the backend became this repo's own
  streaming MLX model through `scripts/serve_streaming.py`. Problems found and
  fixed on the way:
  - mlx-lm's server sends SSE keepalive comments during long prefills, and
    pool's client fails on them and retries from scratch. The server's
    `wfile` is wrapped to drop them.
  - `generation_config.json` lists EOS ids [2, 24]; mlx-lm only inferred 2,
    so the model repeated `</assistant>` (24). `_load` adds 24.
  - `--prompt-cache-size 2` evicted the conversation's own cache entry every
    turn; 6, capped at 4 GiB, keeps it.
  - `--max-tokens` defaults to 512 in mlx-lm's server; pool never sends one
    and aborts on a truncated reply, so it is raised to 4096.
- **Corpus** (`lookup_acceptance.json` → `corpus`): 11 sessions, 763,690
  tokens, 301,743 assistant tokens. 6 astropy (397,400 tokens) for
  calibration, 5 django (366,290) held out. Sessions run 42,771–82,671 tokens;
  the longest, `django__django-11039`, became the Part 3 target.

### Replay metric (Task 9, `58ce7b8`)

`replay/replay.py`: build a reference `.npz` from the unmodified model (top-64
logprobs at every assistant position), then score any model on the same ids:
KL truncated to the reference's top-64, top-1 agreement, peak memory, tok/s,
and optionally greedy action match. The reference is this repo's own model, not
the trace tokens, so "no change" scores KL = 0 exactly; every sweep carries
that self-check row.

### SSD expert streaming (Tasks 16–18)

- **Design** after TurboFieldfare: resident common weights, one file per MoE
  layer holding every expert at a fixed stride, an LFU slot cache with
  recency tie-break, parallel `pread`s.
- **Task 16** (`f056385`, `fe2d168`): `scripts/pack_experts.py`, byte-exact.
- **Task 17** (`1fa9ee0`, `28b572c`): `laguna_mlx/streaming.py`. Streamed
  logits match resident within 1e-5 on the tiny model.
- **Task 18** (`ac013c3`, `9744d0e`): `python3 -m scripts.bench_stream
  --slots 16 32 64 128 --resident` → `streaming.json` (full 4-bit model,
  uncapped MLX pool):

  | Target | Decode tok/s | Prefill tok/s | Peak GB | Hit rate |
  |---|---|---|---|---|
  | Resident | 18.6 | 40.6 | 19.81 | — |
  | 16 slots | 10.7 | 52.2 | 3.70 | 0.314 |
  | 32 slots | 10.1 | 71.0 | 4.66 | 0.405 |
  | 64 slots | 9.9 | 90.1 | 7.06 | 0.527 |
  | 128 slots | 9.7 | 115.4 | 12.80 | 0.735 |

  Decode looked flat across slots. That reading was later shown to be an
  artifact of the uncapped MLX buffer pool (S5, below).
- **Task 10** (baseline script) was not built as planned; Task 18's resident
  row and Part 2's `prune_point.json` base row serve as the baseline.

### Speculative decoding (Tasks 12–14, S1–S4)

- **Task 12** (`6b5ca38`): `TrimmableWindowCache`, because mlx-lm's
  `RotatingKVCache` can't roll back after it wraps; plus layer-output capture.
- **Task 13** (`51cd4a5`): first MLX port of Poolside's DFlash draft model
  (5 layers, conditioned on target layers 1, 13, 25, 33, 39), lossless greedy
  verify loop.
- **Task 14** (`920a345`, `253eb76`): `python3 -m scripts.bench_dflash
  --target resident`, then `--target streaming` → `dflash.json`. Resident:
  0.36×–0.86× of a 40.3 tok/s greedy baseline over 9 settings (draft bits
  16/8/4 × n_draft 3/7/15), 1.82–2.51 accepted tokens per step. Streaming (32
  slots): 1.06× at n = 3, 0.32× at n = 15.
- **S1** (`45f0a8e`, `13dda24`): `python3 -m scripts.diag_dflash` →
  `dflash_diag.json`. Greedy on the plain and trimmable caches is identical;
  DFlash diverges only at 0–4 bf16-ulp near-ties. Verify cost grows with block
  length (resident 25.3 ms at 1 token, 132.0 ms at 16) and, under streaming,
  jumps past 4 tokens (61.9 to 212.0 ms) once a block needs more experts than
  32 slots hold. **Not a bug: the hardware's MoE verify cost.**
- **S2** (`4606a85`, `7b1277b`, `22f84f1`): `python3 -m
  scripts.lookup_acceptance` (CPU only) → `lookup_acceptance.json`. Best
  predicted: g = 4, n = 3, fixed, 2.35 tokens per forward, 1.48× resident and
  1.70× streaming.
- **S3** (`bfbe7e7`, `bdc40c5`): `generate_lookup` with an O(1) `NgramIndex`;
  prefill computes last-position logits only.
- **S4** (`ffb1ae4` … `5659be6`): `python3 -m scripts.bench_lookup` overnight →
  `lookup.json`. Resident g = 4, n = 3: 1.22× decode, 1.00× end to end, after
  8k-token real session prefixes. The script's own `summary.keep` is true (its
  decode-only 1.1× rule). Decision (user, 2026-10-16): keep the code,
  unused; it is exact but doesn't pay off end to end here.
- **Task 15** (decode pipelining, `mx.compile`, prefill tuning) was not run as
  planned; the S1–S5 addendum replaced it, and `mx.compile` was assessed later
  under S5.
- `scripts/overnight.sh` (`99a3676`, `7a4ae65`) ran S1, the Part 2 chain and
  S4 unattended: `caffeinate -dims bash scripts/overnight.sh`.

**Decision:** neither speculative method is used. DFlash is a measured
negative; prompt lookup is marginal end to end. See [no-gos.md](no-gos.md).

---

## Part 2: expert pruning (2026-09-26 to 2026-10-13)

Spec `2026-09-26-expert-pruning-design.md`, plan `2026-09-26-expert-pruning.md`
(P2-1 to P2-5).

**Goal.** Find how many of the 256 routed experts per layer can be removed
one-shot, without retraining, and what it costs in agent behaviour; then
materialize one checkpoint. Routed experts are ~17.4 GB of the ~18.4 GB 4-bit
weights.

| Task | Commits | What |
|---|---|---|
| P2-1 | `d4bbf2c` | `ExpertSaliencyRecorder`: per layer and expert, routed count, gate mass, REAP sum `gate_j(x)·‖f_j(x)‖` |
| P2-2 | `0ac5305`, `e6748a7` | `keep_sets` (uniform, global with ≥ 16 kept per layer) and `apply_mask` (pruned experts' selection bias set to −inf; returns undo) |
| P2-3 | `acd8cd0` | `prune_checkpoint`: slice experts, router rows and bias; write a normal MLX checkpoint |
| P2-4 | `7e4cfc8`, `883ce2b`, `b37d935` | `scripts/prune_sweep.py` and its tiny dry run |
| P2-5 | `8e22adb`, `15bd4c8` | `scripts/prune_point.py`, the real runs and results |

**Method and parameters.**

- Calibration on the 6 astropy sessions, each capped at 16,384 tokens: 98,304
  tokens at 233 tok/s (`expert_stats_meta.json`).
- Evaluation on the 5 django sessions, 16,384-token cap, against references
  from the unpruned 4-bit model (`refs/prune`).
- Criteria: REAP, gate mass, routing frequency, random. Ratios 12.5–62.5%.
  Allocation: uniform (main), global (comparison).
- Operating-point rule (spec): the largest uniform ratio with held-out top-1 ≥
  0.90 and KL ≤ 0.10.

**Commands.**

```bash
python3 -m scripts.prune_sweep calibrate    # -> results/expert_stats.npz (~7 min)
python3 -m scripts.prune_sweep refs         # -> refs/prune/
python3 -m scripts.prune_sweep sweep        # -> results/pruning.json (23 rows, 525–590 s each)
python3 -m scripts.prune_point --criterion reap --ratio 0.5
python3 -m scripts.diag_actions --pruned models/Laguna-XS-2.1-pruned-50
```

**Results** (`pruning.json`, held-out top-1 / KL mean, uniform):

| Pruned | REAP | Gate mass | Frequency | Random |
|---|---|---|---|---|
| 12.5% | 0.985 / 0.005 | 0.978 / 0.014 | 0.977 / 0.017 | 0.950 / 0.095 |
| 25% | 0.967 / 0.025 | 0.968 / 0.036 | 0.964 / 0.041 | 0.924 / 0.176 |
| 37.5% | 0.952 / 0.052 | 0.948 / 0.076 | 0.946 / 0.086 | 0.883 / 0.331 |
| 50% | 0.945 / 0.081 | 0.930 / 0.118 | 0.917 / 0.154 | 0.842 / 0.490 |
| 62.5% | 0.909 / 0.187 | 0.890 / 0.265 | 0.881 / 0.343 | 0.782 / 0.697 |

- REAP has the lowest KL at every ratio. Global allocation is worse (REAP
  global 50%: 0.928 / 0.125) and can't be materialized anyway, since
  `num_experts` must be one number.
- Operating point: **REAP, uniform, 50%**. 62.5% fails on KL (0.187).

`prune_point.json`, short prompts:

| | Full 4-bit | Pruned 50% |
|---|---|---|
| Disk | 18.39 GB | 9.80 GB |
| Resident peak | 18.90 GB | 10.10 GB |
| Resident decode | 34.2 tok/s | 38.7 tok/s |
| Held-out top-1 / KL | reference | 0.945 / 0.081 |
| Streaming, 32 slots (uncapped pool) | — | 3.97 GB, 12.8 tok/s, hit rate 0.618 |

**Action match** (`17d8c84`, `action_diag.json`): greedy action match at the
operating point was 0.3. The noise floor (the base model against its own
references) is 20/20 exact, so the gap is real: the pruned model gets 6/20
exact and 18/20 same tool. A manual read of the 14 mismatches found 2 hard
failures (a repetition loop, a corrupted path), 3 that differ only in a
line-range argument and 9 plausible alternatives.

**Decision (human, 2026-10-13, `9c6f6d9`):** pruned-50 is the base for Part 3.
n = 20 is small; a 100-action sample is open.

---

## Part 3: KV cache compression and the 16 GB answer (2026-10-13 to 2026-10-16)

Spec `2026-10-13-kv-compression-design.md`, plan `2026-10-13-kv-compression.md`
(P3-1 to P3-4), then the S5 streaming work and the 8-bit decode follow-ups.

**Goal.** Run the 82,671-token session on pruned-50 within **12 GB peak**
(75% of 16 GB, a stand-in for the macOS working set), with quality gate
top-1 ≥ 0.98 and KL mean ≤ 0.02 against pruned-50 with a plain cache. Stop at
the first setting that passes.

### P3-1 to P3-4: quantized KV

| Task | Commits | What |
|---|---|---|
| P3-1 | `19b90f1` | `make_quantized_cache(model, bits)`: quantized K/V on the 10 full-attention layers only |
| P3-2 | `745a6a6` | `score(..., make_cache=)` |
| P3-3 | `3a43e84`, `bf8a5ab`, `7b0e66e` | `scripts/bench_kv.py`, with a 16 GB peak abort that also re-checks finished rows |
| P3-4 | `2327f08`, `5d16a5a` | real runs → `kv.json` |

**Incident** (`d60dd8e`): the first 8-bit run went through mlx-lm's quantized
attention, which materializes a `[heads, chunk, context]` score matrix (~16 GB
per layer at 82k). The machine hung into a GPU watchdog reboot that killed
every job. `DequantKVCache` replaced that path: it stores K/V quantized and
dequantizes one layer at a time for `mx.fast`'s fused SDPA, and deliberately
has no `bits` attribute so mlx-lm never routes it to the quantized path.

**Commands:** `python3 -m scripts.bench_kv` (resident: builds `refs/kv`, then
plain, 8-bit, 4-bit over 5 sessions; several hours), then `python3 -m
scripts.bench_kv --target streaming` (40 min).

**Results** (`kv.json`, 82k session for decode and peak):

| Setting | Peak GB | KV GB | Top-1 | KL mean | Decode tok/s | Gate |
|---|---|---|---|---|---|---|
| Resident, plain | 14.60 | 3.39 | 1.0 | 0 | 7.42 | fail: budget |
| Resident, 8-bit | 13.16 | 1.80 | 0.9963 | 0.0020 | 4.10 | fail: budget |
| Resident, 4-bit | 12.29 | 0.95 | 0.9944 | 0.0038 | 4.99 | fail: budget (+0.29) |
| Streaming, plain, 32 slots, uncapped | 8.04 | 3.39 | 1.0 | 0 | 4.69 | pass |

- Quality passes at both widths; memory is what fails. Peak minus KV is
  11.2–11.4 GB in every resident row (weights plus long-prefill working set),
  so KV compression alone can't reach 12 GB resident.
- **Decision (human, 2026-10-14, `5d16a5a`):** pruned streaming with the plain
  cache is the 16 GB answer. Quantized KV stays in the code, off.

### S5: where streaming decode time goes (2026-10-14 to 2026-10-15)

`scripts/profile_stream.py` → `stream_profile.json`. Pruned-50 pack, 32 decode
steps after a prefix of the longest session.

1. **Profile** (`01c52b0`, `e0c5deb`), 32 slots, uncapped: 54.9 ms per step at
   256 context, 169.9 ms at 32k. Router sync 63% / 58%, expert load 30% / 38%.
2. **Resident control** (`83b71aa`, `711499c`): 23.1 ms at 256, 46.8 ms at
   32k. Streaming overhead splits about half loading, half everything else.
3. **Option 1, split timing** (`c085680`, `7477ffb`): slot writes are a real
   cost (11.7 ms at 256, 27.7 at 32k). The run was stopped by hand: footprint
   ~17 GB, swap rising.
4. **Option 2, batched slot writes** (`a1ca51e`, `2c20ec0`): first attempt
   stopped by the 14 GB watchdog; rerun with a 1 GB MLX pool cap. Batched
   writes were slower (84.7 → 92.5 ms at 32k) and were removed. **The pool cap
   alone halved the 32k step** (169.9 → 84.7 ms; load 65.4 → 15.5 ms): the
   uncapped free-buffer pool had been pushing the expert files out of the page
   cache.
5. **Opt-in cap** (`9c35d6e`, `ee78006`): `load_streaming(...,
   cache_limit_gb=None)`; `serve_streaming` and `profile_stream` pass 1.0. The
   wired limit is now raised only for resident models.
6. **Capped slot sweep** (`ea24d15`): `python3 -m scripts.profile_stream
   --slots 16 32 64 128`. At 32k: 97.5, 83.5, 72.2, 68.9 ms per step. More
   slots help once the pool is capped. Sync stays at 63–65 ms; at 128 slots
   nothing loads, yet the step is 22 ms slower than resident. That is the
   **sync floor**.
7. **64 slots at 82k** (`da2f36b`, `5798f52`): `python3 -m scripts.bench_kv
   --target streaming --slots 64 --cache-limit-gb 1.0` → `kv.json` row
   `streaming-64slots-cap1.0gb`: **10.16 GB peak, 6.87 tok/s, top-1 1.0, KL 0**.
   This became the Part 3 answer ([part3-answer.md](part3-answer.md)).
8. **Sync floor attempts:**
   - `mx.async_eval` per streamed layer (`f381e67`, `965c83d`, `b35250e`):
     1.5× slower, same tokens; removed.
   - `mx.compile` (`2a68ae1`): assessed off-GPU, ruled out (KV offsets are
     graph constants, so every token retraces).
   - Drop-on-miss (`ffe1d74`, `0be0788`, `0505303`): design note
     [gpu-side-routing.md](gpu-side-routing.md), stage 1 simulated with
     `python3 -m scripts.sim_drop` → `drop_sim.json`: KL 0.146, top-1 0.937,
     14% of picks dropped. Fails the gate.
   - An exact fix needs GPU-readable expert pages: a C++ MLX extension. Not
     pursued, by the user's choice (2026-10-15).
9. **Prefill slot-write leak** (`c237343`), found by moeStream:
   `generate_step` evaluates only the caches after each prefill chunk, so the
   last MoE layer's slot writes stayed pending. `StreamingSwitchGLU` now
   `async_eval`s its slot tensors after multi-token calls.

### Serving the answer (2026-10-15)

- `743ffa6`: `mlx_lm.server.main()` raised the wired limit to the max working
  set on every start. `serve_streaming` now hands it
  `mx_without_wired_limit()`.
- `dbcf9b2`: `serve_streaming --preset laptop16` (pruned-50 pack, 64 slots,
  1 GB pool cap, plain KV, plus `--max-tokens 4096`, `--prompt-cache-size 6`,
  `--prompt-cache-bytes 4294967296`). Explicit flags win.

### Interfaces for moeStream (2026-10-15)

`14e63a2` (parity-harness guide), `fd0e82f` (public `greedy_actions`),
`3393d33` (`forward=` on `assistant_logprobs` and `score`), `2fb0bcc`
(references record `model`; `score` returns `ref_model`), `28a494f`
(`diag_actions --base/--refs/--out`). All additive. See
[interfaces.md](interfaces.md).

### 8-bit KV decode: `quantized_decode` and the fused kernel (2026-10-15 to 2026-10-16)

1. **Microbenchmark** (`fa828e5`): `python3 -m scripts.micro_kv_decode` →
   `kv_decode_micro.json`, one layer, synthetic K/V. At 82k: plain 7.19 ms,
   dequantize + SDPA 12.98 ms, mlx-lm quantized SDPA 12.05 ms. It predicted only
   ~9 ms per token from `quantized_decode`.
2. **`quantized_decode` flag** (`fa828e5`): single-token steps hand mlx-lm's
   quantized SDPA the packed cache.
3. **Fused kernel** (`dc0667f`): `laguna_mlx/q8_attn.py`, `fused_decode`.
4. **Resident A/B** (`ef9403e`, results `2128b04`), one overnight GPU slot:

   ```bash
   python3 -m scripts.bench_kv --qdecode off   --out results/kv_qdecode_off.json
   python3 -m scripts.bench_kv --qdecode on    --out results/kv_qdecode_on.json
   python3 -m scripts.micro_kv_decode
   python3 -m scripts.bench_kv --qdecode fused --out results/kv_qdecode_fused.json
   ```

   82k session, 8-bit KV: dequantize 4.49 tok/s, `quantized_decode` 6.64
   (1.48×), fused 7.59 (1.69×). Tokens identical to dequantize over 64 steps in
   both, same 13.09 GB peak. The microbenchmark had under-predicted the gain,
   because it doesn't capture the cost of allocating ~0.68 GB of bf16
   temporaries per layer per token in the real model (inferred, not measured
   directly).
5. **Streaming A/B** (`033e059`, results `06cea1d`), `laptop16` config:

   ```bash
   python3 -m scripts.bench_kv --target streaming --slots 64 --cache-limit-gb 1.0 --qdecode off   --out results/kv_stream_q8.json
   python3 -m scripts.bench_kv --target streaming --slots 64 --cache-limit-gb 1.0 --qdecode fused --out results/kv_stream_q8fused.json
   python3 -m scripts.bench_kv --target streaming --slots 64 --cache-limit-gb 1.0 --qdecode plain --out results/kv_stream_plain.json
   ```

   8-bit dequantize 3.87 tok/s / 8.72 GB; 8-bit fused 5.70 / 8.72 (identical
   tokens to dequantize); plain 6.07 / 10.17.
6. **Decision (2026-10-16):** `laptop16` keeps the plain cache, because it is
   faster and exact. `--fused-decode` is opt-in, for when 1.45 GB of peak
   matters more. The rule was set before the run: on only if fused matches its
   own unfused tokens and is at least as fast as plain.

### Task 11: upstream PR (2026-10-15)

Not filed. A fresh clone of mlx-lm showed upstream already has Laguna
(`mlx_lm/models/laguna.py`, #1334 on 2026-07-26 and #1704), so the PR would be
a duplicate.
