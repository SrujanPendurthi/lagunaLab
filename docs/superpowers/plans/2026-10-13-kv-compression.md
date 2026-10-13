# Part 3: KV Cache Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the longest captured agent session (~78k tokens) running on the pruned Laguna XS 2.1 with peak GPU memory ≤ 12 GB, at a measured quality cost, by quantizing the KV cache of the 10 full-attention layers.

**Architecture:** A cache factory puts mlx-lm's `QuantizedKVCache` on full-attention layers and leaves sliding layers alone. `replay.score` accepts that factory, so the existing replay metric scores it. One bench script measures the plain cache first (step 0), then 8-bit and 4-bit, and writes `results/kv.json`.

**Tech Stack:** Python 3.13, mlx 0.32.0, mlx-lm 0.31.3, numpy, pytest; the existing `laguna_mlx`, `replay` and `scripts` packages.

**Spec:** `docs/superpowers/specs/2026-10-13-kv-compression-design.md`

## Checklist

- [ ] **Gate:** `results/action_diag.json` exists and the human confirmed the pruned model (`models/Laguna-XS-2.1-pruned-50`) is the base for Part 3
- [ ] P3-1 `make_quantized_cache`, tiny tests
- [ ] P3-2 `score` / `assistant_logprobs` accept a cache factory, tiny test
- [ ] P3-3 `scripts/bench_kv.py`, tiny dry-run test
- [ ] P3-4 Real-model run, writing `results/kv.json`
- [ ] **Human decision:** accept the setting the gate picks, or change the budget/thresholds
- [ ] Final review for Part 3 (whole-diff reviewer)

## Global Constraints

- Machine: 24 GB M4 Air. The Mac must not crash. Never hold two models at once. Check `sysctl vm.swapusage` before and after each long run and stop if swap exceeds ~10 GB.
- No new dependencies; don't upgrade mlx or mlx-lm.
- Run entry points as `python3 -m scripts.<name>` from the repo root; tests as `python3 -m pytest`.
- Every number in any writeup comes from a file in `results/`.
- Never set `fix_mistral_regex` on the tokenizer: the transformers warning about it is a verified false positive.
- Budget: peak GPU ≤ 12.0 GB. Gate: top-1 ≥ 0.98 and KL mean ≤ 0.02 nats against the pruned resident model with the plain cache.
- Evaluation: the 5 django sessions at full length. The reference is the pruned resident model with the plain cache.
- `mx.quantize` supports group sizes 32, 64 and 128 only. The real model's head_dim is 128; the shared tiny test config has head_dim 16, so quantized-cache tests use a tiny model with head_dim 64.

## Review Focus

1. **head_dim not divisible by the group size** would make `QuantizedKVCache` allocate the wrong shape. `make_quantized_cache` must raise `ValueError`. Pinned in P3-1.
2. **Cache growth across the 256-step allocation boundary, mixed with single-token steps:** chunked decoding must give the same logits as one pass with the same quantized cache. Pinned in P3-1.
3. **Default behaviour of `score` unchanged:** with no factory it uses `model.make_cache()`, and passing `model.make_cache` explicitly gives the same numbers. Pinned in P3-2.
4. **A session longer than `build_ref`'s default 65,536-token cap is silently truncated,** which would report a peak for a shorter session than claimed. The bench passes a higher cap and refuses a truncated reference. Pinned in P3-3.
5. **Interrupted bench:** a restart must not redo finished rows or leave `results/kv.json` unparseable. Pinned in P3-3.

## File Map

```
laguna_mlx/cache.py     + make_quantized_cache(model, bits=8, group_size=64)
replay/replay.py        assistant_logprobs(..., make_cache=None), score(..., make_cache=None)
scripts/bench_kv.py     new: refs, sweep, decode_at_ctx, summarize
tests/tiny.py           + mlx_only_model(**overrides)
tests/test_cache.py     + quantized cache tests
tests/test_replay.py    + cache factory test
tests/test_bench_kv.py  new
results/kv.json         output
refs/kv/                full-length references from the pruned resident model (gitignored like refs/prune)
```

---

### Task P3-1: Quantized cache factory

**Files:**
- Modify: `laguna_mlx/cache.py` (append), `tests/tiny.py` (append)
- Test: `tests/test_cache.py` (append)

**Interfaces:**
- Produces: `make_quantized_cache(model, bits=8, group_size=64) -> list` (one cache per layer, `QuantizedKVCache` where `model.args.layer_types[i] == "full_attention"`, else `RotatingKVCache(max_size=model.args.sliding_window)`); `tests.tiny.mlx_only_model(**overrides) -> laguna.Model` (random MLX weights, no PyTorch reference).

- [ ] **Step 1: Add the tiny-model helper to `tests/tiny.py`**

```python
def mlx_only_model(seed=0, **overrides):
    """Random-weight tiny model with config overrides (no PyTorch twin), e.g. head_dim=64."""
    mx.random.seed(seed)
    m = laguna.Model(laguna.ModelArgs.from_dict({**copy.deepcopy(TINY), **overrides}))
    mx.eval(m.parameters())
    return m
```

- [ ] **Step 2: Write the failing tests in `tests/test_cache.py`**

Add to the imports at the top:

```python
from mlx_lm.models.cache import QuantizedKVCache, RotatingKVCache

from laguna_mlx.cache import make_quantized_cache
from tests.tiny import mlx_only_model
```

Append:

```python
@pytest.fixture(scope="module")
def model64():
    return mlx_only_model(head_dim=64)  # mx.quantize needs a group size of 32, 64 or 128


def _ids(n):
    return mx.array(np.random.default_rng(1).integers(0, TINY["vocab_size"], size=(1, n)))


def test_quantized_cache_layout_follows_layer_types(model64):
    cache = make_quantized_cache(model64, bits=4)
    for c, lt in zip(cache, model64.args.layer_types):
        if lt == "full_attention":
            assert isinstance(c, QuantizedKVCache) and c.bits == 4 and c.group_size == 64
        else:
            assert isinstance(c, RotatingKVCache) and c.max_size == TINY["sliding_window"]


def test_quantized_cache_error_shrinks_with_bits_and_cache_is_smaller(model64):
    ids = _ids(300)
    full = np.array(model64(ids))
    plain = model64.make_cache()
    mx.eval(model64(ids, cache=plain))
    err, size = {}, {}
    for bits in (8, 4):
        cache = make_quantized_cache(model64, bits)
        # mean, not max: a small KV error can flip an expert's top-k routing at a few positions
        err[bits] = float(np.abs(np.array(model64(ids, cache=cache)) - full).mean())
        size[bits] = cache[0].nbytes
    assert err[8] < 0.05 and err[8] < err[4]
    assert size[4] < size[8] < plain[0].nbytes


def test_quantized_cache_chunked_matches_one_pass(model64):
    ids = _ids(300)  # crosses QuantizedKVCache.step (256) mid-chunk, then single-token steps
    one = np.array(model64(ids, cache=make_quantized_cache(model64, 8)))
    cache = make_quantized_cache(model64, 8)
    splits = [(0, 200), (200, 260)] + [(i, i + 1) for i in range(260, 300)]
    outs = [np.array(model64(ids[:, a:b], cache=cache)) for a, b in splits]
    np.testing.assert_allclose(np.concatenate(outs, axis=1), one, atol=1e-4, rtol=1e-3)


def test_quantized_cache_rejects_indivisible_head_dim(model):
    with pytest.raises(ValueError, match="head_dim"):
        make_quantized_cache(model, 8)  # shared tiny model: head_dim 16, group size 64
```

- [ ] **Step 3: Run them to see them fail**

Run: `python3 -m pytest tests/test_cache.py -q -k quantized`
Expected: collection error, `ImportError: cannot import name 'make_quantized_cache'`.

- [ ] **Step 4: Implement in `laguna_mlx/cache.py`**

Change the import line to:

```python
from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache, _BaseCache
```

Append:

```python
def make_quantized_cache(model, bits=8, group_size=64):
    """mlx-lm's quantized KV cache on full-attention layers; sliding layers keep their window."""
    a = model.args
    if a.head_dim % group_size:
        raise ValueError(f"head_dim {a.head_dim} is not a multiple of group_size {group_size}")
    return [
        RotatingKVCache(max_size=a.sliding_window)
        if lt == "sliding_attention" else QuantizedKVCache(group_size=group_size, bits=bits)
        for lt in a.layer_types
    ]
```

- [ ] **Step 5: Run the tests**

Run: `python3 -m pytest tests/test_cache.py -q`
Expected: all pass (the 3 existing split tests plus 4 new ones).

- [ ] **Step 6: Commit**

```bash
git add laguna_mlx/cache.py tests/tiny.py tests/test_cache.py
git commit -m "feat: quantized KV cache factory for full-attention layers (P3-1)"
```

---

### Task P3-2: Replay accepts a cache factory

**Files:**
- Modify: `replay/replay.py` (`assistant_logprobs`, `score`)
- Test: `tests/test_replay.py` (append)

**Interfaces:**
- Consumes: `make_quantized_cache` (P3-1), `mlx_only_model` (P3-1).
- Produces: `assistant_logprobs(model, ids, mask, chunk=2048, make_cache=None)` and `score(model, ref_path, chunk=2048, make_cache=None)`. `make_cache` is a zero-argument callable returning the per-layer cache list; `None` means `model.make_cache`.

- [ ] **Step 1: Write the failing test in `tests/test_replay.py`**

Add to the imports:

```python
from laguna_mlx.cache import make_quantized_cache
from tests.tiny import mlx_only_model
```

Append:

```python
def test_score_with_cache_factory(tok, session, tmp_path):
    m = mlx_only_model(head_dim=64)
    ref = tmp_path / "s.npz"
    build_ref(m, tok, session, ref, n_actions=0)
    default = score(m, ref)
    explicit = score(m, ref, make_cache=m.make_cache)
    assert explicit["kl_top64_mean"] == default["kl_top64_mean"] and explicit["top1_agree"] == 1.0
    q4 = score(m, ref, make_cache=lambda: make_quantized_cache(m, 4))
    assert q4["kl_top64_mean"] > abs(default["kl_top64_mean"]) + 1e-6  # the factory is really used
```

- [ ] **Step 2: Run it to see it fail**

Run: `python3 -m pytest tests/test_replay.py -q -k cache_factory`
Expected: FAIL, `TypeError: score() got an unexpected keyword argument 'make_cache'`.

- [ ] **Step 3: Implement in `replay/replay.py`**

In `assistant_logprobs`, change the signature and the cache line:

```python
def assistant_logprobs(model, ids, mask, chunk=2048, make_cache=None):
```
```python
    cache = (make_cache or model.make_cache)()
```

In `score`, change the signature and the loop header:

```python
def score(model, ref_path, chunk=2048, make_cache=None):
```
```python
    for p, lp in assistant_logprobs(model, ids, mask, chunk, make_cache):
```

- [ ] **Step 4: Run the replay tests**

Run: `python3 -m pytest tests/test_replay.py -q`
Expected: all pass (3 existing, 1 new).

- [ ] **Step 5: Commit**

```bash
git add replay/replay.py tests/test_replay.py
git commit -m "feat: replay score accepts a cache factory (P3-2)"
```

---

### Task P3-3: KV bench script

**Files:**
- Create: `scripts/bench_kv.py`
- Test: `tests/test_bench_kv.py`

**Interfaces:**
- Consumes: `make_quantized_cache` (P3-1); `score(..., make_cache=)` (P3-2); `build_ref` (`replay.replay`); `load_streaming` (`laguna_mlx.streaming`); `mlx_only_model` (P3-1).
- Produces: `build_refs(model, tok, traces, refs_dir, cap)`, `decode_at_ctx(model, ids, make_cache, n=64, chunk=2048) -> dict`, `sweep(model, target, settings, refs, out, decode_n=64)`, `summarize(res) -> dict`, and `results/kv.json` with shape `{"rows": {"<target>/<plain|8|4>/<session>": {...}}, "summary": {...}}`.

- [ ] **Step 1: Write the failing test `tests/test_bench_kv.py`**

```python
import json

import pytest
from transformers import AutoTokenizer

from scripts import bench_kv
from tests.tiny import mlx_only_model

MESSAGES = [
    {"role": "user", "content": "list files"},
    {"role": "assistant", "content": "There are two files, a.py and b.py."},
]


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("models/Laguna-XS-2.1")


@pytest.fixture
def traces(tmp_path):
    out = []
    for name, extra in (("short", ""), ("long", " Please also describe each of them in detail.")):
        f = tmp_path / f"{name}.jsonl"
        msgs = [{**MESSAGES[0], "content": MESSAGES[0]["content"] + extra}, MESSAGES[1]]
        f.write_text(json.dumps({"path": "/v1/chat/completions", "request": {"messages": msgs},
                                 "status": 200, "response": ""}))
        out.append(f)
    return out


def test_sweep_rows_summary_and_resume(tok, traces, tmp_path, monkeypatch):
    m = mlx_only_model(head_dim=64)
    refs_dir, out = tmp_path / "refs", tmp_path / "kv.json"
    refs = bench_kv.build_refs(m, tok, traces, refs_dir, cap=4096)
    bench_kv.sweep(m, "resident", (None, 8, 4), refs, out, decode_n=4)
    res = json.loads(out.read_text())
    assert set(res["rows"]) == {f"resident/{b}/{s}" for b in ("plain", "8", "4") for s in ("short", "long")}
    # decode numbers only on the longest session
    assert "decode_tok_s" in res["rows"]["resident/8/long"] and "decode_tok_s" not in res["rows"]["resident/8/short"]
    s = res["summary"]
    assert abs(s["resident/plain"]["kl_mean"]) < 1e-5 and s["resident/plain"]["top1"] == 1.0
    assert s["resident/4"]["kv_gb"] < s["resident/8"]["kv_gb"] < s["resident/plain"]["kv_gb"]
    assert s["resident/plain"]["fits"] and s["resident/plain"]["passes"]

    # resume: nothing is recomputed, the file is unchanged
    def boom(*a, **k):
        raise AssertionError("recomputed a finished row")

    monkeypatch.setattr(bench_kv, "score", boom)
    before = out.read_text()
    bench_kv.sweep(m, "resident", (None, 8, 4), refs, out, decode_n=4)
    assert out.read_text() == before


def test_truncated_reference_is_refused(tok, traces, tmp_path):
    m = mlx_only_model(head_dim=64)
    with pytest.raises(ValueError, match="truncated"):
        bench_kv.build_refs(m, tok, traces, tmp_path / "refs", cap=8)
```

- [ ] **Step 2: Run it to see it fail**

Run: `python3 -m pytest tests/test_bench_kv.py -q`
Expected: collection error, `ImportError: cannot import name 'bench_kv'`.

- [ ] **Step 3: Write `scripts/bench_kv.py`**

```python
"""KV cache quantization on the pruned model: memory, speed and replay quality.

python3 -m scripts.bench_kv                    # resident: plain, 8-bit, 4-bit on 5 django sessions
python3 -m scripts.bench_kv --target streaming # step 0 only: plain cache, longest session
Resumable: finished rows in results/kv.json are skipped.
"""
import argparse
import glob
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import laguna_mlx  # noqa: F401
from laguna_mlx.cache import make_quantized_cache
from replay.replay import build_ref, score

MODEL = "models/Laguna-XS-2.1-pruned-50"
PACK = "models/Laguna-XS-2.1-pruned-50-packed"
HELDOUT = sorted(glob.glob("traces/django__*.jsonl"))
REFS = Path("refs/kv")
OUT = Path("results/kv.json")
CAP = 131072  # above the longest captured session (~78k); build_ref's default would truncate
BUDGET_GB = 12.0  # 75% of 16 GB: stand-in for a 16 GB Mac's recommended working set
MIN_TOP1, MAX_KL = 0.98, 0.02


def build_refs(model, tok, traces, refs_dir, cap=CAP):
    """Full-length references from `model` with the plain cache. Returns ref paths, longest last."""
    refs_dir = Path(refs_dir)
    refs_dir.mkdir(parents=True, exist_ok=True)
    refs = []
    for f in traces:
        r = refs_dir / f"{Path(f).stem}.npz"
        if not r.exists():
            build_ref(model, tok, f, r, max_tokens=cap, n_actions=0)
            print("ref", r.stem, flush=True)
        if bool(np.load(r)["truncated"]):
            r.unlink()
            raise ValueError(f"{r.stem}: truncated at cap={cap}; raise the cap")
        refs.append(r)
    return sorted(refs, key=lambda r: len(np.load(r)["ids"]))


def decode_at_ctx(model, ids, make_cache, n=64, chunk=2048):
    """Prefill `ids`, then greedy-decode n tokens: KV size, decode speed and peak at full context."""
    cache = make_cache()
    mx.reset_peak_memory()
    for a in range(0, len(ids), chunk):
        h = model.model(mx.array(ids[None, a : a + chunk]), cache=cache)
        mx.eval(h)
    kv = sum(c.nbytes for c, lt in zip(cache, model.args.layer_types) if lt == "full_attention")
    tok = mx.argmax(model.lm_head(h[:, -1]), axis=-1)
    mx.eval(tok)
    t0 = time.perf_counter()
    for _ in range(n):
        tok = mx.argmax(model.lm_head(model.model(tok[None], cache=cache)[:, -1]), axis=-1)
        mx.eval(tok)
    return {"kv_gb": kv / 1e9, "decode_tok_s": n / (time.perf_counter() - t0),
            "decode_peak_gpu_gb": mx.get_peak_memory() / 1e9}


def summarize(res):
    out = {}
    for key in sorted({k.rsplit("/", 1)[0] for k in res["rows"]}):
        rows = [v for k, v in res["rows"].items() if k.rsplit("/", 1)[0] == key]
        w = np.array([r["assistant_tokens"] for r in rows])
        s = {"sessions": len(rows),
             "kl_mean": float(np.average([r["kl_top64_mean"] for r in rows], weights=w)),
             "kl_p99_max": max(r["kl_top64_p99"] for r in rows),
             "top1": float(np.average([r["top1_agree"] for r in rows], weights=w)),
             "prefill_tok_s": float(np.mean([r["tok_per_s"] for r in rows])),
             "peak_gpu_gb": max(max(r["peak_mem_gb"], r.get("decode_peak_gpu_gb", 0)) for r in rows)}
        for r in rows:
            if "kv_gb" in r:
                s.update(kv_gb=r["kv_gb"], decode_tok_s=r["decode_tok_s"], longest_tokens=r["tokens"])
        s["fits"] = s["peak_gpu_gb"] <= BUDGET_GB
        s["passes"] = bool(s["fits"] and s["top1"] >= MIN_TOP1 and s["kl_mean"] <= MAX_KL)
        out[key] = s
    return out


def sweep(model, target, settings, refs, out=OUT, decode_n=64):
    """One row per (setting, session); `refs` is ordered longest last. Saves after every row."""
    out = Path(out)
    res = json.loads(out.read_text()) if out.exists() else {"rows": {}}
    res.setdefault("budget_gb", BUDGET_GB)
    for bits in settings:
        mk = model.make_cache if bits is None else (lambda b=bits: make_quantized_cache(model, b))
        for r in refs:
            key = f"{target}/{bits or 'plain'}/{r.stem}"
            if key in res["rows"]:
                continue
            row = score(model, r, make_cache=mk)
            if r == refs[-1]:
                row.update(decode_at_ctx(model, np.load(r)["ids"], mk, n=decode_n))
            res["rows"][key] = row
            res["summary"] = summarize(res)
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(res, indent=1))
            tmp.replace(out)  # atomic: an interrupt never leaves a half-written file
            print(key, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    return res


def main():
    from mlx_lm import load

    from laguna_mlx.streaming import load_streaming

    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["resident", "streaming"], default="resident")
    p.add_argument("--slots", type=int, default=32)
    a = p.parse_args()
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    if a.target == "resident":
        model, tok = load(MODEL)
        refs = build_refs(model, tok, HELDOUT, REFS)
        res = sweep(model, "resident", (None, 8, 4), refs)
    else:
        refs = sorted(REFS.glob("*.npz"), key=lambda r: len(np.load(r)["ids"]))
        if not refs:
            raise SystemExit("run the resident target first: it builds refs/kv")
        model, _ = load_streaming(PACK, a.slots)
        res = sweep(model, "streaming", (None,), refs[-1:])
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_bench_kv.py -q`
Expected: 2 passed.

- [ ] **Step 5: Run the whole suite**

Run: `python3 -m pytest -q`
Expected: no failures.

- [ ] **Step 6: Commit**

```bash
git add scripts/bench_kv.py tests/test_bench_kv.py
git commit -m "feat: KV quantization bench script + tiny dry-run test (P3-3)"
```

---

### Task P3-4: Real-model run

Run only when the human is not using the Mac for anything heavy: the resident run is expected to peak near 13 GB with the plain cache.

**Files:**
- Output: `results/kv.json`, `results/kv_resident.log`, `results/kv_streaming.log`, `refs/kv/*.npz`

**Interfaces:**
- Consumes: `python3 -m scripts.bench_kv` (P3-3).

- [ ] **Step 1: Check memory before starting**

Run: `sysctl vm.swapusage`
Expected: `used` well under 10000M. If not, stop and tell the human.

- [ ] **Step 2: Resident run (builds references, then plain, 8-bit, 4-bit)**

Run: `python3 -m scripts.bench_kv > results/kv_resident.log 2>&1`
Expected: 5 `ref` lines, then 15 row lines, then the summary. `resident/plain` has `kl_mean` ≈ 0 and `top1` = 1.0 (same model, same cache as the reference); if it does not, stop: the measurement is broken, not the cache. Duration is unknown (full-length resident prefill was never timed); the run is resumable, so an interrupt loses at most one row.

- [ ] **Step 3: Check swap, then the streaming step-0 run**

Run: `sysctl vm.swapusage && python3 -m scripts.bench_kv --target streaming > results/kv_streaming.log 2>&1`
Expected: one row, `streaming/plain/<longest session>`. Its KL and top-1 measure streaming against resident (kernel-path noise), not KV damage.

- [ ] **Step 4: Read the gate off the summary**

Run: `python3 -c "import json; [print(k, v) for k, v in json.load(open('results/kv.json'))['summary'].items()]"`

The operating point is the first of `resident/plain`, `resident/8`, `resident/4` with `passes: true`. Report to the human: that setting, its peak, KV size, prefill and decode speed, and whether `streaming/plain` fits with no compression. If nothing passes, report which condition failed (budget or quality) and stop; the spec's fallbacks (smaller prefill chunk, higher pruning ratio) are a new decision for the human.

- [ ] **Step 5: Commit the results**

```bash
git add results/kv.json
git commit -m "feat: KV quantization results on the pruned model (P3-4)"
```
