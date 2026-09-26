# Part 2: Expert Pruning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure quality-vs-pruning curves for Laguna XS 2.1's routed experts (REAP, frequency, gate mass, random; uniform and global allocation) by replay on held-out agent sessions, then materialize and benchmark one operating point.

**Architecture:** A recorder collects per-expert saliency on calibration sessions. Pruning is applied during the sweep as a router mask (pruned experts' selection bias set to −inf), so every config runs through the same kernels as the reference. Only the chosen operating point is written out as a smaller checkpoint, plus a streaming pack.

**Tech Stack:** Python 3.13, mlx 0.32.0, mlx-lm 0.31.3, numpy, pytest; the existing `laguna_mlx` (model, hooks, streaming), `replay` and `scripts` packages.

**Spec:** `docs/superpowers/specs/2026-09-26-expert-pruning-design.md`

## Checklist

- [ ] **Gate:** Part 1 Task 14 (DFlash on real weights) concluded and reviewed. The human confirmed Part 2 starts.
- [ ] P2-1 Saliency recorder (`ExpertSaliencyRecorder`, `accumulate`, `scores`), tiny tests
- [ ] P2-2 Keep-set selection and router masking (`keep_sets`, `apply_mask`), tiny tests
- [ ] P2-3 Pruned checkpoint materialization (`prune_checkpoint`), tiny tests
- [ ] P2-4 Real-model sweep: calibrate on astropy, then build refs, then sweep, writing `results/expert_stats.npz` and `results/pruning.json`
- [ ] **Human decision:** pick the operating point from the curve (default: the largest uniform ratio with held-out top-1 ≥ 0.90 and KL ≤ 0.10)
- [ ] P2-5 Materialize and benchmark the operating point, writing `results/prune_point.json`
- [ ] Final review for Part 2 (whole-diff reviewer)

## Global Constraints

- Machine: 24 GB M4 Air with a GPU wired limit of 22.5 GB. The human's constraint is that the Mac must not crash. Never hold two full models at once. Check `sysctl vm.swapusage` between long runs and stop if swap exceeds ~10 GB.
- No new dependencies; don't upgrade mlx or mlx-lm.
- Run entry points as `python3 -m scripts.<name>` from the repo root.
- Every number in any writeup comes from a file in `results/`.
- Never set `fix_mistral_regex` on the tokenizer: the transformers warning about it is a verified false positive.
- Split: calibrate on the 6 astropy sessions only; held-out evaluation is the 5 django sessions. Never tune on django.
- The reference and all pruned configs run on the resident 4-bit model (`models/Laguna-XS-2.1-mlx-4bit`) through the same code path.

## Review Focus

1. **Recorder transparency and cleanup**, including when the forward pass raises: the model must be unchanged after `__exit__`. Pinned in P2-1.
2. **Masked routing still selects exactly top-k from kept experts,** with weights renormalized over the kept set. A layer whose keep-set is smaller than top_k must raise, not silently select −inf experts. Pinned in P2-2.
3. **Global allocation respects the per-layer minimum and the exact budget.** Pinned in P2-2.
4. **Materialized checkpoint:** router rows, `e_score_correction_bias` and every quantized part (`weight`, `scales`, `biases`) are sliced with the same indices, in the same order. Pinned in P2-3.
5. **Sweep resumability:** an interrupted sweep restarts without redoing finished configs or corrupting `results/pruning.json`. Pinned in P2-4.

---

### Task P2-1: Saliency recorder

**Files:**
- Create: `laguna_mlx/pruning.py`
- Test: `tests/test_pruning.py`

**Interfaces:**
- Consumes: `SparseMoeBlock.route(x) -> (inds, w)` and `switch_mlp(x, inds) -> [..., k, hidden]` (Part 1, Task 1); `laguna_mlx.hooks.RoutingRecorder` (Part 1, Task 3, for test comparison).
- Produces: `accumulate(stats, inds, w, norms, mask=None)`; `ExpertSaliencyRecorder(model)` context manager with `.stats: dict[int, {"count": int64[E], "gate": float64[E], "reap": float64[E]}]`, `.token_mask`, `.save(path)`; `load_stats(path)`; `scores(stats, criterion, seed=0) -> dict[int, float64[E]]` for `criterion in {"reap", "frequency", "gate", "random"}`.

- [ ] **Step 1: Write failing tests `tests/test_pruning.py`**

```python
import mlx.core as mx
import numpy as np
import pytest

from laguna_mlx.hooks import RoutingRecorder
from laguna_mlx.pruning import ExpertSaliencyRecorder, accumulate, load_stats, scores
from tests.tiny import TINY, mlx_model, torch_model

E, K = TINY["num_experts"], TINY["num_experts_per_tok"]


def _ids(n=12, seed=11):
    return mx.array(np.random.default_rng(seed).integers(0, TINY["vocab_size"], size=(1, n)))


def _empty():
    return {"count": np.zeros(E, np.int64), "gate": np.zeros(E), "reap": np.zeros(E)}


def test_accumulate_matches_hand_computation():
    s = _empty()
    inds = np.array([[0, 3], [3, 5]])
    w = np.array([[0.6, 0.4], [0.7, 0.3]])
    norms = np.array([[2.0, 1.0], [3.0, 10.0]])
    accumulate(s, inds, w, norms)
    assert s["count"].tolist() == [1, 0, 0, 2, 0, 1, 0, 0]
    np.testing.assert_allclose(s["gate"][[0, 3, 5]], [0.6, 1.1, 0.3])
    np.testing.assert_allclose(s["reap"][[0, 3, 5]], [1.2, 0.4 + 2.1, 3.0])
    accumulate(s, inds, w, norms, mask=np.array([False, True]))
    assert s["count"][3] == 3 and s["count"][0] == 1


def test_recorder_is_transparent_restores_and_matches_routing_counts():
    m = mlx_model(torch_model())
    ids = _ids()
    before = np.array(m(ids))
    with RoutingRecorder(m) as rr:
        m(ids)
    with ExpertSaliencyRecorder(m) as rec:
        during = np.array(m(ids))
    np.testing.assert_array_equal(before, during)
    np.testing.assert_array_equal(before, np.array(m(ids)))
    for i in rr.counts:
        np.testing.assert_array_equal(rec.stats[i]["count"], rr.counts[i])
        np.testing.assert_allclose(rec.stats[i]["gate"], rr.mass[i], rtol=1e-5)
        assert (rec.stats[i]["reap"] >= 0).all() and rec.stats[i]["reap"].sum() > 0
    assert all("route" not in vars(l.mlp) for l in m.layers)
    assert all(type(l.mlp.switch_mlp).__name__ == "SwitchGLU" for l in m.layers if hasattr(l.mlp, "route"))


def test_recorder_restores_model_when_forward_raises():
    m = mlx_model(torch_model())
    with pytest.raises(ValueError):
        with ExpertSaliencyRecorder(m):
            raise ValueError("boom")
    assert all(type(l.mlp.switch_mlp).__name__ == "SwitchGLU" for l in m.layers if hasattr(l.mlp, "route"))


def test_save_load_and_scores(tmp_path):
    m = mlx_model(torch_model())
    with ExpertSaliencyRecorder(m) as rec:
        m(_ids(20))
    rec.save(tmp_path / "s.npz")
    st = load_stats(tmp_path / "s.npz")
    assert sorted(st) == sorted(rec.stats)
    r = scores(st, "reap")
    i = sorted(st)[0]
    c = st[i]["count"]
    np.testing.assert_allclose(r[i][c > 0], st[i]["reap"][c > 0] / c[c > 0])
    assert (r[i][c == 0] == 0).all()
    np.testing.assert_array_equal(scores(st, "random", seed=3)[i], scores(st, "random", seed=3)[i])
    with pytest.raises(ValueError):
        scores(st, "nope")
```

- [ ] **Step 2: Run to see it fail**

Run: `python3 -m pytest tests/test_pruning.py -q`
Expected: `ModuleNotFoundError: No module named 'laguna_mlx.pruning'`

- [ ] **Step 3: Implement the recorder part of `laguna_mlx/pruning.py`**

```python
"""One-shot expert pruning for Laguna MoE layers: saliency, keep-sets, masking, materialization.

REAP (Cerebras 2025, arXiv 2510.13999): saliency of expert j = mean over tokens routed to j
of gate_j(x) * ||f_j(x)||, where f_j is the expert's output before gate weighting.
"""
import glob
import json
import re
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def accumulate(stats, inds, w, norms, mask=None):
    """Add one forward's routing to stats. inds/w/norms: [N, k] numpy; mask: [N] bool."""
    if mask is not None:
        inds, w, norms = inds[mask], w[mask], norms[mask]
    ii = inds.ravel()
    np.add.at(stats["count"], ii, 1)
    np.add.at(stats["gate"], ii, w.ravel())
    np.add.at(stats["reap"], ii, (w * norms).ravel())


class _Tap(nn.Module):
    """Wraps a SwitchGLU to report per-selection expert outputs to the recorder."""

    def __init__(self, inner, rec, layer):
        super().__init__()
        self.inner = inner
        self._ctx = [rec, layer]  # list: keeps the recorder out of the parameter tree

    def __call__(self, x, indices):
        y = self.inner(x, indices)
        rec, layer = self._ctx
        rec._observe(layer, indices, y)
        return y


class ExpertSaliencyRecorder:
    """Per MoE layer and expert: routed count, gate mass, REAP sum (gate * ||expert output||).

        with ExpertSaliencyRecorder(model) as rec:
            model(ids)
        rec.save("results/expert_stats.npz")

    Set rec.token_mask (bool over B*L of the next forward) to count only some tokens.
    Syncs to numpy per layer: slow, use for calibration only.
    """

    def __init__(self, model):
        self.moe = {i: l.mlp for i, l in enumerate(model.layers) if hasattr(l.mlp, "route")}
        E = model.args.num_experts
        self.stats = {i: {"count": np.zeros(E, np.int64), "gate": np.zeros(E), "reap": np.zeros(E)}
                      for i in self.moe}
        self.token_mask = None
        self._w = {}

    def __enter__(self):
        for i, mlp in self.moe.items():
            mlp.route = self._wrap_route(i, mlp)
            mlp.switch_mlp = _Tap(mlp.switch_mlp, self, i)
        return self

    def __exit__(self, *exc):
        for mlp in self.moe.values():
            if "route" in vars(mlp):
                del mlp.route
            if isinstance(mlp.switch_mlp, _Tap):
                mlp.switch_mlp = mlp.switch_mlp.inner

    def _wrap_route(self, i, mlp):
        orig = type(mlp).route

        def route(x):
            inds, w = orig(mlp, x)
            self._w[i] = w
            return inds, w

        return route

    def _observe(self, i, inds, y):
        k = inds.shape[-1]
        w = self._w.pop(i)
        norms = mx.linalg.norm(y.astype(mx.float32), axis=-1)
        accumulate(self.stats[i], np.array(inds).reshape(-1, k),
                   np.array(w.astype(mx.float32)).reshape(-1, k),
                   np.array(norms).reshape(-1, k), self.token_mask)

    def save(self, path):
        np.savez(path, **{f"{i}_{n}": a for i, s in self.stats.items() for n, a in s.items()})


def load_stats(path):
    z = np.load(path)
    out = {}
    for key in z.files:
        i, name = key.split("_", 1)
        out.setdefault(int(i), {})[name] = z[key]
    return out


def scores(stats, criterion, seed=0):
    """Per-layer saliency, higher = more important."""
    rng = np.random.default_rng(seed)
    out = {}
    for i in sorted(stats):
        s = stats[i]
        if criterion == "reap":
            out[i] = s["reap"] / np.maximum(s["count"], 1)
        elif criterion == "frequency":
            out[i] = s["count"].astype(np.float64)
        elif criterion == "gate":
            out[i] = s["gate"].copy()
        elif criterion == "random":
            out[i] = rng.random(len(s["count"]))
        else:
            raise ValueError(f"unknown criterion {criterion!r}")
    return out
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_pruning.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add laguna_mlx/pruning.py tests/test_pruning.py
git commit -m "feat: REAP expert saliency recorder"
```

---

### Task P2-2: Keep-sets and router masking

**Files:**
- Modify: `laguna_mlx/pruning.py`
- Modify: `tests/test_pruning.py`

**Interfaces:**
- Consumes: `scores(...)` output (P2-1); `SparseMoeBlock.e_score_correction_bias`, `.top_k`.
- Produces: `keep_sets(sal, ratio, allocation="uniform", min_keep=16) -> dict[int, int64 array (sorted)]`, where `ratio` is the fraction of experts pruned; `apply_mask(model, keep) -> undo()`.

- [ ] **Step 1: Add failing tests**

```python
from laguna_mlx.pruning import apply_mask, keep_sets


def _fake_sal(layers=(1, 2, 3), E=8, seed=0):
    rng = np.random.default_rng(seed)
    return {i: rng.random(E) for i in layers}


def test_keep_sets_uniform_counts_and_order():
    sal = _fake_sal()
    keep = keep_sets(sal, 0.25)
    for i, v in sal.items():
        assert len(keep[i]) == 6 and list(keep[i]) == sorted(keep[i])
        assert set(keep[i]) == set(np.argsort(-v)[:6])


def test_keep_sets_global_budget_and_minimum():
    sal = _fake_sal(E=8)
    keep = keep_sets(sal, 0.5, allocation="global", min_keep=3)
    assert sum(len(v) for v in keep.values()) == 12
    assert all(len(v) >= 3 for v in keep.values())
    # global allocation normalizes per layer, so a layer's raw scale must not matter
    scaled = keep_sets({**sal, 1: sal[1] * 100}, 0.5, allocation="global", min_keep=3)
    assert {i: v.tolist() for i, v in scaled.items()} == {i: v.tolist() for i, v in keep.items()}
    with pytest.raises(ValueError):
        keep_sets(sal, 0.9, allocation="global", min_keep=3)


def test_mask_routes_only_kept_experts_and_undoes():
    m = mlx_model(torch_model())
    ids = _ids(16)
    before = np.array(m(ids))
    keep = {1: np.array([0, 2, 5]), 2: np.array([1, 3, 4, 7]), 3: np.array([0, 1])}
    undo = apply_mask(m, keep)
    with RoutingRecorder(m) as rr:
        m(ids)
    for i, k in keep.items():
        assert set(np.nonzero(rr.counts[i])[0]) <= set(k.tolist())
        assert rr.counts[i].sum() == 16 * K
    undo()
    np.testing.assert_array_equal(np.array(m(ids)), before)


def test_masking_unused_experts_is_exact():
    m = mlx_model(torch_model())
    ids = _ids(16)
    with RoutingRecorder(m) as rr:
        before = np.array(m(ids))
    keep = {i: np.nonzero(c)[0] for i, c in rr.counts.items()}
    keep = {i: k if len(k) >= K else np.arange(K) for i, k in keep.items()}
    undo = apply_mask(m, keep)
    np.testing.assert_array_equal(np.array(m(ids)), before)
    undo()


def test_mask_smaller_than_top_k_raises():
    m = mlx_model(torch_model())
    with pytest.raises(ValueError):
        apply_mask(m, {1: np.array([0])})
```

- [ ] **Step 2: Run to see the new tests fail**

Run: `python3 -m pytest tests/test_pruning.py -q`
Expected: `ImportError: cannot import name 'apply_mask'`

- [ ] **Step 3: Implement in `laguna_mlx/pruning.py`**

```python
def keep_sets(sal, ratio, allocation="uniform", min_keep=16):
    """Experts to keep per layer (sorted). ratio = fraction of experts pruned."""
    if allocation == "uniform":
        out = {}
        for i, v in sal.items():
            n = len(v) - int(round(ratio * len(v)))
            out[i] = np.sort(np.argsort(-v, kind="stable")[:n])
        return out
    if allocation == "global":
        total = sum(len(v) for v in sal.values())
        budget = total - int(round(ratio * total))
        if budget < min_keep * len(sal):
            raise ValueError(f"budget {budget} below per-layer minimum {min_keep} x {len(sal)} layers")
        norm = {i: v / max(float(v.sum()), 1e-12) for i, v in sal.items()}
        order = {i: np.argsort(-norm[i], kind="stable") for i in sal}
        keep = {i: list(order[i][:min_keep]) for i in sal}
        rest = sorted(((norm[i][e], -i, -e) for i in sal for e in order[i][min_keep:]), reverse=True)
        for _, ni, ne in rest[: budget - min_keep * len(sal)]:
            keep[-ni].append(-ne)
        return {i: np.sort(np.array(v, dtype=np.int64)) for i, v in keep.items()}
    raise ValueError(f"unknown allocation {allocation!r}")


def apply_mask(model, keep):
    """Route only to kept experts: pruned experts' selection bias becomes -inf. Returns undo()."""
    saved = {}
    for i, k in keep.items():
        mlp = model.layers[i].mlp
        if len(k) < mlp.top_k:
            raise ValueError(f"layer {i}: keeping {len(k)} experts < top_k {mlp.top_k}")
        b = mlp.e_score_correction_bias
        saved[i] = b
        allowed = np.zeros(b.shape[0], bool)
        allowed[np.asarray(k)] = True
        mlp.e_score_correction_bias = mx.where(mx.array(allowed), b.astype(mx.float32), -mx.inf)

    def undo():
        for i, b in saved.items():
            model.layers[i].mlp.e_score_correction_bias = b

    return undo
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_pruning.py -q`
Expected: 9 passed. If `test_masking_unused_experts_is_exact` fails by a tiny margin, the cause is the bias dtype change (bf16 → f32) altering the tie-break among unmasked experts. On the tiny float32 model the bias is already float32, so any mismatch there is a real bug.

- [ ] **Step 5: Commit**

```bash
git add laguna_mlx/pruning.py tests/test_pruning.py
git commit -m "feat: keep-set selection and router-mask pruning"
```

---

### Task P2-3: Materialize a pruned checkpoint

**Files:**
- Modify: `laguna_mlx/pruning.py`
- Modify: `tests/test_pruning.py`

**Interfaces:**
- Consumes: an MLX model dir (config with `quantization`, `*.safetensors`); `keep_sets` uniform output.
- Produces: `prune_checkpoint(model_dir, keep, out_dir)` writes an MLX model dir with `num_experts = len(keep[i])` and every per-expert tensor sliced. Uniform keep-sets only (raises otherwise).

- [ ] **Step 1: Add failing tests**

```python
from mlx_lm.utils import load_model

import laguna_mlx  # noqa: F401
from laguna_mlx.pruning import prune_checkpoint
from tests.tiny import quantized_model_dir


def test_materialized_checkpoint_matches_masked_model(tmp_path):
    d, m = quantized_model_dir(tmp_path)
    keep = {i: np.array([0, 2, 3, 6, 7]) for i in (1, 2, 3)}
    out = tmp_path / "pruned"
    prune_checkpoint(d, keep, out)
    pruned, _ = load_model(out)
    ids = _ids(20, seed=5)
    undo = apply_mask(m, keep)
    want = np.array(m(ids))
    undo()
    np.testing.assert_allclose(np.array(pruned(ids)), want, atol=1e-5, rtol=1e-5)
    assert pruned.args.num_experts == 5


def test_prune_checkpoint_rejects_non_uniform(tmp_path):
    d, _ = quantized_model_dir(tmp_path)
    with pytest.raises(ValueError):
        prune_checkpoint(d, {1: np.arange(5), 2: np.arange(6), 3: np.arange(5)}, tmp_path / "x")
```

- [ ] **Step 2: Run to see them fail**

Run: `python3 -m pytest tests/test_pruning.py -q`
Expected: `ImportError: cannot import name 'prune_checkpoint'`

- [ ] **Step 3: Implement in `laguna_mlx/pruning.py`**

```python
_EXPERT_KEY = re.compile(
    r"model\.layers\.(\d+)\.mlp\.(?:switch_mlp\.\w+\.(?:weight|scales|biases)"
    r"|gate\.(?:weight|scales|biases)|e_score_correction_bias)$"
)


def prune_checkpoint(model_dir, keep, out_dir):
    """Write a smaller MLX checkpoint keeping only `keep[layer]` experts (uniform count)."""
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    counts = {len(v) for v in keep.values()}
    if len(counts) != 1:
        raise ValueError("prune_checkpoint supports uniform keep-sets only")
    out_dir.mkdir(parents=True)
    idx = {i: mx.array(np.asarray(v)) for i, v in keep.items()}
    for f in sorted(glob.glob(str(model_dir / "*.safetensors"))):
        out = {}
        for key, a in mx.load(f).items():
            m = _EXPERT_KEY.match(key)
            out[key] = a[idx[int(m.group(1))]] if m and int(m.group(1)) in idx else a
        mx.save_safetensors(str(out_dir / Path(f).name), out, metadata={"format": "mlx"})
    config = json.loads((model_dir / "config.json").read_text())
    config["num_experts"] = counts.pop()
    (out_dir / "config.json").write_text(json.dumps(config, indent=1))
    for f in model_dir.iterdir():
        if f.suffix in (".json", ".jinja") and f.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(f, out_dir / f.name)
```

Notes for the implementer:
- `model.safetensors.index.json` is left out on purpose: its `total_size` would be wrong. Confirm mlx-lm 0.31.3 loads a directory without it; the loader globs `*.safetensors`.
- Real-model memory: `mx.load` is lazy (mmap). Each shard's sliced arrays are evaluated at `save_safetensors`, so peak memory is about one shard (~5 GB).

- [ ] **Step 4: Run tests, full suite, commit**

```bash
python3 -m pytest -q
git add laguna_mlx/pruning.py tests/test_pruning.py
git commit -m "feat: materialize pruned MLX checkpoints"
```

---

### Task P2-4: Real-model calibration, references and sweep

**Files:**
- Create: `scripts/prune_sweep.py`
- Output: `results/expert_stats.npz`, `results/expert_stats_meta.json`, `results/pruning.json` (committed); `refs/prune/*.npz` (gitignored)

**Interfaces:**
- Consumes: `ExpertSaliencyRecorder`, `load_stats`, `scores`, `keep_sets`, `apply_mask` (P2-1/2); `replay.session.load_session`, `tokenize`; `replay.replay.build_ref`, `score` (Part 1, Task 9).
- Produces: CLI `python3 -m scripts.prune_sweep {calibrate,refs,sweep} [--cap 16384]`.

- [ ] **Step 1: Write `scripts/prune_sweep.py`**

```python
"""Expert-pruning sweep on the resident 4-bit model.

python3 -m scripts.prune_sweep calibrate   # astropy sessions -> results/expert_stats.npz
python3 -m scripts.prune_sweep refs        # reference logprobs for eval sessions -> refs/prune/
python3 -m scripts.prune_sweep sweep       # criteria x ratios -> results/pruning.json (resumable)
"""
import argparse
import glob
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import laguna_mlx  # noqa: F401
from laguna_mlx.pruning import ExpertSaliencyRecorder, apply_mask, keep_sets, load_stats, scores
from mlx_lm import load
from replay.replay import build_ref, score
from replay.session import load_session, tokenize

MODEL = "models/Laguna-XS-2.1-mlx-4bit"
CALIB = sorted(glob.glob("traces/astropy__*.jsonl"))
HELDOUT = sorted(glob.glob("traces/django__*.jsonl"))
CALIB_EVAL = CALIB[:2]  # in-sample, to measure the calibration/held-out gap
REFS = Path("refs/prune")
OUT = Path("results/pruning.json")
GRID = ([(c, "uniform", r) for c in ("reap", "frequency", "gate", "random")
         for r in (0.125, 0.25, 0.375, 0.5, 0.625)]
        + [("reap", "global", r) for r in (0.25, 0.5)])


def calibrate(model, tok, cap, chunk=2048):
    t0, n = time.perf_counter(), 0
    with ExpertSaliencyRecorder(model) as rec:
        for f in CALIB:
            msgs, tools = load_session(f)
            ids, _, _ = tokenize(tok, msgs, tools, cap)
            cache = model.make_cache()
            for a in range(0, len(ids), chunk):
                mx.eval(model.model(mx.array(ids[None, a : a + chunk]), cache=cache))
            n += len(ids)
            print(f"calibrated {Path(f).stem}: {len(ids)} tokens", flush=True)
    rec.save("results/expert_stats.npz")
    meta = {"sessions": [Path(f).stem for f in CALIB], "tokens": n, "cap": cap,
            "tok_per_s": n / (time.perf_counter() - t0), "note": "recorder syncs per layer; slower than plain prefill"}
    Path("results/expert_stats_meta.json").write_text(json.dumps(meta, indent=1))
    print(meta)


def refs(model, tok, cap):
    REFS.mkdir(parents=True, exist_ok=True)
    for f in HELDOUT + CALIB_EVAL:
        out = REFS / f"{Path(f).stem}.npz"
        if not out.exists():
            t0 = time.perf_counter()
            build_ref(model, tok, f, out, max_tokens=cap, n_actions=0)
            print(f"ref {out.name} in {time.perf_counter() - t0:.0f}s", flush=True)


def _aggregate(rows):
    w = np.array([r["assistant_tokens"] for r in rows], float)
    return {"kl_mean": float(np.average([r["kl_top64_mean"] for r in rows], weights=w)),
            "kl_p99_max": float(max(r["kl_top64_p99"] for r in rows)),
            "top1": float(np.average([r["top1_agree"] for r in rows], weights=w)),
            "assistant_tokens": int(w.sum())}


def sweep(model):
    stats = load_stats("results/expert_stats.npz")
    done = json.loads(OUT.read_text())["rows"] if OUT.exists() else []
    have = {(r["criterion"], r["allocation"], r["ratio"]) for r in done}
    held = [REFS / f"{Path(f).stem}.npz" for f in HELDOUT]
    cal = [REFS / f"{Path(f).stem}.npz" for f in CALIB_EVAL]
    for crit, alloc, ratio in [("none", "uniform", 0.0)] + GRID:
        if (crit, alloc, ratio) in have:
            continue
        t0 = time.perf_counter()
        undo = lambda: None
        kept = None
        if ratio > 0:
            keep = keep_sets(scores(stats, crit), ratio, alloc)
            kept = {str(i): int(len(v)) for i, v in keep.items()}
            undo = apply_mask(model, keep)
        try:
            row = {"criterion": crit, "allocation": alloc, "ratio": ratio, "kept_per_layer": kept,
                   "heldout": _aggregate([score(model, r) for r in held]),
                   "calib": _aggregate([score(model, r) for r in cal])}
        finally:
            undo()
        row["seconds"] = time.perf_counter() - t0
        done.append(row)
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps({"model": MODEL, "rows": done}, indent=1))
        tmp.replace(OUT)  # atomic: an interrupt never leaves a half-written file
        print(crit, alloc, ratio, row["heldout"], f"{row['seconds']:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["calibrate", "refs", "sweep"])
    p.add_argument("--cap", type=int, default=16384)
    a = p.parse_args()
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    model, tok = load(MODEL)
    {"calibrate": lambda: calibrate(model, tok, a.cap), "refs": lambda: refs(model, tok, a.cap),
     "sweep": lambda: sweep(model)}[a.stage]()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Calibrate, and read the timing before committing to the grid**

Run: `python3 -m scripts.prune_sweep calibrate`
Expected: 6 sessions calibrated and `results/expert_stats.npz` written. From `tok_per_s`, estimate the sweep: (1 + 22 configs) × 7 refs × ~16k tokens ÷ plain prefill speed. Plain prefill is faster than calibration because it has no per-layer sync; time one `refs` session to measure it. If the sweep would take more than ~10 hours, lower `--cap` for the sweep (refs and sweep must use the same cap) or trim the grid to reap, frequency and random. Record the decision in the task report.

- [ ] **Step 3: Build references**

Run: `python3 -m scripts.prune_sweep refs`
Expected: 7 `.npz` files in `refs/prune/`.

- [ ] **Step 4: Sweep, resumable, in the background with blocking polls**

Run: `nohup python3 -m scripts.prune_sweep sweep > results/prune_sweep.log 2>&1 &`
Expected: the first row `("none", "uniform", 0.0)` has held-out `kl_mean` < 1e-5 and `top1` = 1.0. Anything else means the evaluation path is nondeterministic, so stop and investigate. Kill and restart once to confirm that already-finished rows are skipped.

- [ ] **Step 5: Commit**

```bash
git add scripts/prune_sweep.py results/expert_stats.npz results/expert_stats_meta.json results/pruning.json
git commit -m "feat: expert pruning sweep on held-out agent sessions"
```

---

### Task P2-5: Materialize and benchmark the operating point

Starts after the human picks `(criterion, ratio)` from `results/pruning.json`.

**Files:**
- Create: `scripts/prune_point.py`
- Output: `results/prune_point.json`; `models/Laguna-XS-2.1-pruned-<pct>/` and `models/Laguna-XS-2.1-pruned-<pct>-packed/` (gitignored)

**Interfaces:**
- Consumes: `prune_checkpoint`, `keep_sets`, `scores`, `load_stats` (P2); `scripts.pack_experts.pack`, `load_streaming`, `stream_stats` (Part 1, Tasks 16–17); `generate_greedy` (Part 1, Task 13); `build_ref`, `score`, `score_actions` (Part 1, Task 9); `PROMPTS` (Part 1, Task 5).

- [ ] **Step 1: Write `scripts/prune_point.py`**

```python
"""Materialize + benchmark one pruning operating point.

python3 -m scripts.prune_point --criterion reap --ratio 0.5
"""
import argparse
import glob
import json
import subprocess
import time
from pathlib import Path

import mlx.core as mx

import laguna_mlx  # noqa: F401
from laguna_mlx.dflash import generate_greedy
from laguna_mlx.pruning import keep_sets, load_stats, prune_checkpoint, scores
from laguna_mlx.streaming import load_streaming, stream_stats
from mlx_lm import load
from replay.replay import build_ref, score, score_actions
from scripts.check_parity import PROMPTS
from scripts.pack_experts import pack

BASE = "models/Laguna-XS-2.1-mlx-4bit"
HELDOUT = sorted(glob.glob("traces/django__*.jsonl"))
AREFS = Path("refs/prune_actions")


def du_gb(p):
    return int(subprocess.check_output(["du", "-sk", str(p)]).split()[0]) / 1e6


def decode_bench(model, tok):
    prs = [tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                   enable_thinking=False) for q in PROMPTS[:3]]
    mx.reset_peak_memory()
    t0, n = time.perf_counter(), 0
    for pr in prs:
        n += len(generate_greedy(model, pr, 128, tuple(tok.eos_token_ids)))
    return {"decode_tok_s": n / (time.perf_counter() - t0), "peak_gpu_gb": mx.get_peak_memory() / 1e9}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--criterion", default="reap")
    p.add_argument("--ratio", type=float, required=True)
    p.add_argument("--cap", type=int, default=16384)
    p.add_argument("--actions", type=int, default=4)
    a = p.parse_args()
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    pct = int(round(a.ratio * 100))
    out_dir = Path(f"models/Laguna-XS-2.1-pruned-{pct}")
    pack_dir = Path(f"models/Laguna-XS-2.1-pruned-{pct}-packed")
    res = {"criterion": a.criterion, "ratio": a.ratio}

    # 1. references with greedy actions from the UNPRUNED resident model
    AREFS.mkdir(parents=True, exist_ok=True)
    base, tok = load(BASE)
    for f in HELDOUT:
        r = AREFS / f"{Path(f).stem}.npz"
        if not r.exists():
            build_ref(base, tok, f, r, max_tokens=a.cap, n_actions=a.actions)
    res["base"] = decode_bench(base, tok)
    res["base"]["disk_gb"] = du_gb(BASE)
    del base
    mx.clear_cache()

    # 2. materialize + pack
    keep = keep_sets(scores(load_stats("results/expert_stats.npz"), a.criterion), a.ratio)
    if not out_dir.exists():
        prune_checkpoint(BASE, keep, out_dir)
    if not pack_dir.exists():
        pack(out_dir, pack_dir)

    # 3. pruned resident: memory, speed, replay, actions
    model, tok = load(str(out_dir))
    res["resident"] = decode_bench(model, tok)
    res["resident"]["disk_gb"] = du_gb(out_dir)
    rows = [score(model, r) for r in sorted(AREFS.glob("*.npz"))]
    w = [r["assistant_tokens"] for r in rows]
    res["resident"]["kl_mean"] = sum(r["kl_top64_mean"] * x for r, x in zip(rows, w)) / sum(w)
    res["resident"]["top1"] = sum(r["top1_agree"] * x for r, x in zip(rows, w)) / sum(w)
    res["resident"]["action_match"] = sum(score_actions(model, tok, r) for r in sorted(AREFS.glob("*.npz"))) / len(rows)
    del model
    mx.clear_cache()

    # 4. pruned streaming
    model, tok = load_streaming(pack_dir, 32)
    res["streaming_slots32"] = {**decode_bench(model, tok), **stream_stats(model), "disk_gb": du_gb(pack_dir)}
    Path("results/prune_point.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python3 -m scripts.prune_point --criterion <picked> --ratio <picked>`
Expected: `results/prune_point.json` with base, resident and streaming sections. The pruned resident `kl_mean` and `top1` should match the sweep's masked row for the same config: same experts, a different tensor layout, so bf16 rounding differences are tiny. A large gap means a slicing bug, so stop. Disk and `peak_gpu_gb` should shrink roughly in proportion to the pruned fraction of expert weights.

- [ ] **Step 3: Commit**

```bash
git add scripts/prune_point.py results/prune_point.json
git commit -m "feat: materialize and benchmark pruned Laguna operating point"
```
