# Laguna XS MLX Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Laguna XS 2.1 as a 4-bit MLX model on a 24 GB M4 with expert-routing hooks, capture ~50 pool agent sessions, and score any model modification by teacher-forced replay against the unmodified model.

**Architecture:** A single mlx-lm-shaped model file (`laguna_mlx/laguna.py`) proven exact against Poolside's own PyTorch code on a tiny random config, then converted with stock `mlx_lm.convert`. Sessions are captured by a logging proxy between `pool` and ollama. Replay reuses the reference's token ids, so modified models are compared on identical inputs.

**Tech Stack:** Python 3.13, mlx 0.32.0, mlx-lm 0.31.3, transformers 5.14.1 + torch 2.8.0 (reference tests only), numpy, pandas, pytest 9, llama.cpp `llama-server`, ollama 0.30.10, pool.

**Spec:** `docs/superpowers/specs/2026-09-25-laguna-mlx-harness-design.md`

## Global Constraints

- Hardware: Apple M4, 24 GB unified memory. Default GPU working set is 19.07 GB; 4-bit weights are ~18.6 GB, so Task 4 raises it (`iogpu.wired_limit_mb`).
- Disk: 106 GB free. Never hold bf16 (66 GB) and the GGUF (~19 GB) at the same time.
- Do not upgrade mlx, mlx-lm, transformers or torch mid-plan; versions above are what the tests were designed against.
- No new dependencies. pandas + pyarrow are already installed.
- `laguna_mlx/laguna.py` imports only `mlx` and `mlx_lm` — it must stay copy-pasteable into mlx-lm.
- Every number in any writeup is generated from a file in `results/`.
- `models/`, `data/`, `traces/`, `refs/` are gitignored (large or third-party).
- Reference numerics come from Poolside's `modeling_laguna.py`, vendored unchanged into `tests/reference/`.

## Review Focus

1. **Chunked prefill across the 512-token sliding window** (chunk boundaries not aligned to the window) — logits must equal a single full forward. Pinned in Task 2.
2. **Checkpoint features we don't implement** (router softcapping, attention sinks, router weight on input, no gating) — must raise at load, never silently diverge. Pinned in Task 1.
3. **Loading an already-converted checkpoint** — mlx-lm calls `sanitize` again on MLX weights; it must be a no-op. Pinned in Task 1.
4. **Real OpenAI-format messages** — tool-call `arguments` as a JSON string (or malformed), `content` as a list of parts or `null` — must tokenize, not crash or drop text. Pinned in Task 7.
5. **Sessions longer than memory allows** and **upstream errors during capture** — replay truncates with a recorded flag instead of OOM; the proxy logs and forwards 5xx/unreachable upstreams instead of hanging pool. Pinned in Tasks 9 and 6.

## Deviations from spec (decided while planning)

- **KV hook dropped.** mlx-lm caches are already a plain list the caller creates via `model.make_cache()`; Part 3 substitutes cache classes there. No code needed now.
- **`convert.py` is a 10-line wrapper** around `mlx_lm.convert`, which already loads lazily and applies `model.quant_predicate`.
- **Parity is two-tier:** exact (1e-4) against Poolside's PyTorch code on a tiny random config (Task 1), then top-1 agreement against llama.cpp on real weights (Task 5).
- **Assistant-token mask comes from the chat template's `{% generation %}` block** via `return_assistant_tokens_mask`, not from prefix re-rendering.

## File Map

```
lagunaLab/
  pyproject.toml                 pytest config (pythonpath = ".")
  .gitignore
  laguna_mlx/__init__.py         registers laguna with mlx-lm on import
  laguna_mlx/laguna.py           the model (upstreamable)
  laguna_mlx/hooks.py            RoutingRecorder
  replay/__init__.py
  replay/session.py              proxy log -> messages -> ids + assistant mask
  replay/replay.py               build-ref / score / actions, CLI
  scripts/__init__.py
  scripts/convert.py             bf16 -> MLX 4-bit
  scripts/check_parity.py        MLX vs llama-server on real weights
  scripts/proxy.py               logging reverse proxy
  scripts/gen_traces.py          pool x SWE-bench Lite driver
  scripts/baseline.py            memory / speed / max-context numbers
  tests/__init__.py
  tests/reference/{__init__,modeling_laguna,configuration_laguna}.py   vendored
  tests/tiny.py                  tiny config + torch/MLX model builders
  tests/test_laguna.py  tests/test_cache.py  tests/test_hooks.py
  tests/test_proxy.py   tests/test_session.py tests/test_replay.py
  results/                       committed JSON outputs
```

---

### Task 1: Model port, exact against Poolside's PyTorch reference

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `laguna_mlx/__init__.py`, `laguna_mlx/laguna.py`
- Create: `tests/__init__.py`, `tests/reference/__init__.py`, `tests/reference/modeling_laguna.py`, `tests/reference/configuration_laguna.py`, `tests/tiny.py`
- Test: `tests/test_laguna.py`

**Interfaces:**
- Produces: `laguna.ModelArgs`, `laguna.Model(args)` with `__call__(inputs: mx.array[B,L], cache=None) -> logits[B,L,V]`, `.model` (inner `LagunaModel`, returns final hidden states), `.lm_head`, `.layers`, `.args`, `make_cache() -> list`, `sanitize(dict) -> dict`, `quant_predicate`, `cast_predicate(k) -> bool`. MoE layers expose `mlp.route(x) -> (inds[..., k] int, weights[..., k])`.
- Produces (tests): `tests.tiny.TINY: dict`, `torch_model(seed=0)`, `mlx_model(torch_model)`.

- [ ] **Step 1: Scaffold and vendor the reference**

```bash
cd ~/Projects/Current/lagunaLab
mkdir -p laguna_mlx replay scripts tests/reference results
touch tests/__init__.py tests/reference/__init__.py replay/__init__.py scripts/__init__.py
for f in modeling_laguna.py configuration_laguna.py; do
  curl -sL -o tests/reference/$f https://huggingface.co/poolside/Laguna-XS-2.1/raw/main/$f
done
hf download poolside/Laguna-XS-2.1 tokenizer.json tokenizer_config.json special_tokens_map.json \
  chat_template.jinja config.json generation_config.json --local-dir models/Laguna-XS-2.1
```

`pyproject.toml`:
```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

`.gitignore`:
```
models/
data/
traces/
refs/
__pycache__/
.DS_Store
```

- [ ] **Step 2: Write `tests/tiny.py`**

```python
"""Tiny random Laguna config shared by tests: same structure as XS 2.1, toy sizes."""
import copy

import mlx.core as mx
import torch

from laguna_mlx import laguna
from tests.reference.configuration_laguna import LagunaConfig
from tests.reference.modeling_laguna import LagunaForCausalLM

ROPE = {
    "full_attention": {
        "rope_theta": 500000.0, "rope_type": "yarn", "factor": 32.0,
        "original_max_position_embeddings": 8192, "beta_slow": 1.0, "beta_fast": 64.0,
        "attention_factor": 1.3465735902799727, "partial_rotary_factor": 0.5,
    },
    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
}

TINY = dict(
    model_type="laguna", vocab_size=100352, hidden_size=64, intermediate_size=128,
    num_hidden_layers=4, num_attention_heads=4, num_attention_heads_per_layer=[4, 8, 8, 8],
    num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
    num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
    shared_expert_intermediate_size=32, norm_topk_prob=True, mlp_only_layers=[0],
    decoder_sparse_step=1, gating="per-head", sliding_window=8,
    layer_types=["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
    moe_routed_scaling_factor=2.5, max_position_embeddings=262144, rope_parameters=ROPE,
    tie_word_embeddings=False,
)


def torch_model(seed=0):
    torch.manual_seed(seed)
    kw = copy.deepcopy({k: v for k, v in TINY.items() if k != "model_type"})
    cfg = LagunaConfig(**kw)
    cfg._attn_implementation = "eager"
    m = LagunaForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith("norm.weight"):
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:  # includes e_score_correction_bias, so biased selection is exercised
                p.copy_(0.05 * torch.randn_like(p))
    return m


def to_disk_names(model):
    """HF in-memory tensors -> the per-expert layout Laguna checkpoints ship on disk."""
    E, I = TINY["num_experts"], TINY["moe_intermediate_size"]
    out = {}
    for k, v in model.state_dict().items():
        v = v.detach().float().numpy()
        if k.endswith("mlp.experts.gate_up_proj"):
            p = k[: -len("gate_up_proj")]
            for e in range(E):
                out[f"{p}{e}.gate_proj.weight"] = v[e, :I]
                out[f"{p}{e}.up_proj.weight"] = v[e, I:]
        elif k.endswith("mlp.experts.down_proj"):
            p = k[: -len("down_proj")]
            for e in range(E):
                out[f"{p}{e}.down_proj.weight"] = v[e]
        elif k.endswith("mlp.gate.e_score_correction_bias"):
            out[k.replace("gate.e_score", "experts.e_score")] = v
        else:
            out[k] = v
    return out


def mlx_model(tmodel):
    m = laguna.Model(laguna.ModelArgs.from_dict(copy.deepcopy(TINY)))
    w = m.sanitize({k: mx.array(v) for k, v in to_disk_names(tmodel).items()})
    m.load_weights(list(w.items()))
    mx.eval(m.parameters())
    return m
```

- [ ] **Step 3: Write the failing tests `tests/test_laguna.py`**

```python
import copy

import mlx.core as mx
import numpy as np
import pytest
import torch

from laguna_mlx import laguna
from tests.tiny import TINY, mlx_model, to_disk_names, torch_model


def test_matches_hf_reference_past_sliding_window():
    t = torch_model()
    m = mlx_model(t)
    ids = np.random.default_rng(0).integers(0, TINY["vocab_size"], size=(1, 24))
    with torch.no_grad():
        ref = t(torch.tensor(ids), use_cache=False).logits.numpy()
    out = np.array(m(mx.array(ids)))
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-3)


def test_sanitize_is_idempotent_on_converted_weights():
    t = torch_model()
    m = laguna.Model(laguna.ModelArgs.from_dict(copy.deepcopy(TINY)))
    once = m.sanitize({k: mx.array(v) for k, v in to_disk_names(t).items()})
    twice = m.sanitize(dict(once))
    assert sorted(once) == sorted(twice)


@pytest.mark.parametrize("bad", [
    {"moe_router_logit_softcapping": 30.0},
    {"swa_attention_sink_enabled": True},
    {"moe_apply_router_weight_on_input": True},
    {"gating": False},
])
def test_unsupported_config_raises(bad):
    with pytest.raises(ValueError):
        laguna.ModelArgs.from_dict({**copy.deepcopy(TINY), **bad})


def test_registered_with_mlx_lm():
    import laguna_mlx  # noqa: F401
    from mlx_lm.utils import _get_classes

    assert _get_classes({"model_type": "laguna"})[0] is laguna.Model
```

- [ ] **Step 4: Run to see them fail**

Run: `python3 -m pytest tests/test_laguna.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'laguna_mlx'`.

- [ ] **Step 5: Implement `laguna_mlx/laguna.py`**

```python
"""Poolside Laguna (XS 2.1 / S 2.1 / M.1) for mlx-lm.

Reference: modeling_laguna.py in poolside/Laguna-XS-2.1. Differences from Qwen-MoE:
per-head softplus output gate, QK RMSNorm, sigmoid router with selection-only bias
and routed scaling, per-layer head counts, and separate RoPE for sliding layers.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    rope_parameters: Dict[str, Any]
    layer_types: Optional[List[str]] = None
    sliding_window: Optional[int] = None
    swa_rope_parameters: Optional[Dict[str, Any]] = None
    num_attention_heads_per_layer: Optional[List[int]] = None
    mlp_only_layers: List[int] = field(default_factory=lambda: [0])
    norm_topk_prob: bool = True
    moe_routed_scaling_factor: float = 1.0
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = False
    gating: Any = True
    moe_router_logit_softcapping: float = 0.0
    swa_attention_sink_enabled: bool = False
    moe_apply_router_weight_on_input: bool = False

    def __post_init__(self):
        if self.moe_router_logit_softcapping:
            raise ValueError("Laguna: moe_router_logit_softcapping is not supported")
        if self.swa_attention_sink_enabled:
            raise ValueError("Laguna: swa_attention_sink_enabled is not supported")
        if self.moe_apply_router_weight_on_input:
            raise ValueError("Laguna: moe_apply_router_weight_on_input is not supported")
        if self.gating not in ("per-head", True):
            raise ValueError(f"Laguna: gating={self.gating!r} is not supported")
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.num_attention_heads_per_layer is None:
            self.num_attention_heads_per_layer = [self.num_attention_heads] * self.num_hidden_layers

    def rope_for(self, layer_type):
        rp = self.rope_parameters
        if layer_type == "sliding_attention":
            return self.swa_rope_parameters or rp.get("sliding_attention", rp)
        return rp.get("full_attention", rp)


def _make_rope(p, head_dim, max_pos):
    dims = int(head_dim * p.get("partial_rotary_factor", 1.0))
    scaling = p if p.get("rope_type", "default") != "default" else None
    return initialize_rope(dims, p["rope_theta"], False, scaling, max_pos)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.n_heads = args.num_attention_heads_per_layer[layer_idx]
        self.n_kv_heads = args.num_key_value_heads
        hd = args.head_dim
        self.scale = hd**-0.5
        h = args.hidden_size
        self.q_proj = nn.Linear(h, self.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(h, self.n_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(h, self.n_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * hd, h, bias=False)
        self.per_head_gate = args.gating == "per-head"
        self.g_proj = nn.Linear(h, self.n_heads if self.per_head_gate else self.n_heads * hd, bias=False)
        self.q_norm = nn.RMSNorm(hd, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(hd, eps=args.rms_norm_eps)
        lt = args.layer_types[layer_idx]
        self.rope = _make_rope(args.rope_for(lt), hd, args.max_position_embeddings)

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3)  # B, L, H, D
        gate = nn.softplus(self.g_proj(x).astype(mx.float32)).astype(out.dtype)
        gate = gate[..., None] if self.per_head_gate else gate.reshape(B, L, self.n_heads, -1)
        return self.o_proj((out * gate).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        h = args.hidden_size
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.routed_scaling_factor = args.moe_routed_scaling_factor
        self.gate = nn.Linear(h, args.num_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((args.num_experts,))
        self.switch_mlp = SwitchGLU(h, args.moe_intermediate_size, args.num_experts)
        self.shared_expert = MLP(h, args.shared_expert_intermediate_size)

    def route(self, x):
        scores = mx.sigmoid(self.gate(x).astype(mx.float32))
        k = self.top_k
        biased = scores + self.e_score_correction_bias.astype(mx.float32)
        inds = mx.argpartition(-biased, kth=k - 1, axis=-1)[..., :k]
        w = mx.take_along_axis(scores, inds, axis=-1)
        if self.norm_topk_prob:
            w = w / w.sum(axis=-1, keepdims=True)
        return inds, w.astype(x.dtype)

    def __call__(self, x):
        inds, w = self.route(x)
        y = (self.switch_mlp(x, inds) * w[..., None]).sum(axis=-2)
        return y * self.routed_scaling_factor + self.shared_expert(x)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        if layer_idx in args.mlp_only_layers:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)
        else:
            self.mlp = SparseMoeBlock(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x, mask=None, cache=None):
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class LagunaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layer_types = args.layer_types
        self.sliding_window = args.sliding_window

    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        masks = {}
        for lt, c in zip(self.layer_types, cache):
            if lt not in masks:
                window = self.sliding_window if lt == "sliding_attention" else None
                masks[lt] = create_attention_mask(h, c, window_size=window)
        for layer, lt, c in zip(self.layers, self.layer_types, cache):
            h = layer(h, masks[lt], c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LagunaModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        return self.lm_head(self.model(inputs, cache))

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            RotatingKVCache(max_size=self.args.sliding_window)
            if lt == "sliding_attention" else KVCache()
            for lt in self.args.layer_types
        ]

    def sanitize(self, weights):
        E = self.args.num_experts
        for l in range(self.args.num_hidden_layers):
            p = f"model.layers.{l}.mlp"
            if f"{p}.experts.0.gate_proj.weight" in weights:
                for n in ("gate_proj", "up_proj", "down_proj"):
                    weights[f"{p}.switch_mlp.{n}.weight"] = mx.stack(
                        [weights.pop(f"{p}.experts.{e}.{n}.weight") for e in range(E)]
                    )
            bias = weights.pop(f"{p}.experts.e_score_correction_bias", None)
            if bias is not None:
                weights[f"{p}.e_score_correction_bias"] = bias
        return weights

    def cast_predicate(self, key):
        # keep the router's selection bias in fp32: bf16 rounding flips near-tie expert picks
        return "e_score_correction_bias" not in key

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
```

`laguna_mlx/__init__.py`:
```python
"""Importing this package registers model_type "laguna" with mlx-lm (load / convert)."""
import sys

from . import laguna

sys.modules.setdefault("mlx_lm.models.laguna", laguna)
```

- [ ] **Step 6: Run tests**

Run: `python3 -m pytest tests/test_laguna.py -q`
Expected: 7 passed. If the parity test fails, bisect by comparing per-layer hidden states (`t.model.layers[i]` forward hooks vs MLX layer outputs) — the likeliest culprits are YaRN mscale on partial rotary and the per-head gate reshape. Fix the MLX side; never edit `tests/reference/`.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: Laguna MLX model, exact vs Poolside PyTorch reference on tiny config"
```

---

### Task 2: Incremental decoding with sliding-window caches

**Files:**
- Test: `tests/test_cache.py`
- Modify: `laguna_mlx/laguna.py` only if the test fails

**Interfaces:**
- Consumes: `Model.make_cache()`, `Model.__call__(inputs, cache)`, `tests.tiny.*`

- [ ] **Step 1: Write the test**

```python
import mlx.core as mx
import numpy as np
import pytest

from tests.tiny import TINY, mlx_model, torch_model

# window is 8: splits cross it mid-chunk, at chunk edges, and in one big prefill
SPLITS = {
    "ragged": [(0, 5), (5, 12), (12, 13)] + [(i, i + 1) for i in range(13, 30)],
    "big_prefill": [(0, 20)] + [(i, i + 1) for i in range(20, 30)],
    "window_aligned": [(0, 8), (8, 16), (16, 24), (24, 30)],
}


@pytest.fixture(scope="module")
def model():
    return mlx_model(torch_model())


@pytest.mark.parametrize("split", SPLITS)
def test_cached_forward_matches_full_forward(model, split):
    ids = mx.array(np.random.default_rng(1).integers(0, TINY["vocab_size"], size=(1, 30)))
    full = np.array(model(ids))
    cache = model.make_cache()
    outs = [np.array(model(ids[:, a:b], cache=cache)) for a, b in SPLITS[split]]
    np.testing.assert_allclose(np.concatenate(outs, axis=1), full, atol=1e-4, rtol=1e-3)
```

- [ ] **Step 2: Run**

Run: `python3 -m pytest tests/test_cache.py -q`
Expected: 3 passed. mlx-lm's `RotatingKVCache.make_mask` handles multi-token updates; if `ragged` fails while `window_aligned` passes, the bug is mask construction for chunks that overflow the window — confirm `create_attention_mask(h, c, window_size=...)` receives the sliding layer's cache (not the full-attention one).

- [ ] **Step 3: Commit**

```bash
git add tests/test_cache.py && git commit -m "test: cached decoding matches full forward across sliding window"
```

---

### Task 3: Routing recorder

**Files:**
- Create: `laguna_mlx/hooks.py`
- Test: `tests/test_hooks.py`

**Interfaces:**
- Consumes: `mlp.route(x)` on MoE layers; `model.layers`; `model.args.num_experts`.
- Produces: `RoutingRecorder(model)` context manager; `.counts: dict[int, np.ndarray[int64, E]]`, `.mass: dict[int, np.ndarray[float64, E]]`, `.token_mask: np.ndarray[bool] | None` (flattened over B*L of the next forward).

- [ ] **Step 1: Write the failing test**

```python
import mlx.core as mx
import numpy as np

from laguna_mlx.hooks import RoutingRecorder
from tests.tiny import TINY, mlx_model, torch_model

K = TINY["num_experts_per_tok"]


def test_recorder_is_transparent_and_counts_every_selection():
    m = mlx_model(torch_model())
    ids = mx.array(np.random.default_rng(2).integers(0, TINY["vocab_size"], size=(1, 10)))
    before = np.array(m(ids))
    with RoutingRecorder(m) as rec:
        during = np.array(m(ids))
    after = np.array(m(ids))
    np.testing.assert_array_equal(before, during)
    np.testing.assert_array_equal(before, after)
    assert sorted(rec.counts) == [1, 2, 3]  # layer 0 is dense
    assert all(c.sum() == 10 * K for c in rec.counts.values())
    assert all(abs(rec.mass[i].sum() - 10) < 1e-3 for i in rec.mass)  # norm_topk_prob
    assert not any("route" in vars(l.mlp) for l in m.layers)  # uninstalled


def test_token_mask_limits_counting():
    m = mlx_model(torch_model())
    ids = mx.array(np.random.default_rng(3).integers(0, TINY["vocab_size"], size=(1, 10)))
    with RoutingRecorder(m) as rec:
        rec.token_mask = np.array([True] * 4 + [False] * 6)
        m(ids)
    assert all(c.sum() == 4 * K for c in rec.counts.values())
```

- [ ] **Step 2: Run to see it fail**

Run: `python3 -m pytest tests/test_hooks.py -q`
Expected: `ModuleNotFoundError: No module named 'laguna_mlx.hooks'`

- [ ] **Step 3: Implement `laguna_mlx/hooks.py`**

```python
"""Opt-in instrumentation. Nothing here is imported by the model itself."""
import mlx.core as mx
import numpy as np


class RoutingRecorder:
    """Count expert selections per MoE layer while active.

        with RoutingRecorder(model) as rec:
            model(ids)
        rec.counts[layer]  # int64[num_experts]
        rec.mass[layer]    # float64[num_experts], summed routing weights

    Set rec.token_mask (bool over B*L of the next forward) to count only some tokens.
    Recording syncs to numpy per layer, so it is slow; never leave it on for timing runs.
    """

    def __init__(self, model):
        self.moe = {i: l.mlp for i, l in enumerate(model.layers) if hasattr(l.mlp, "route")}
        E = model.args.num_experts
        self.counts = {i: np.zeros(E, np.int64) for i in self.moe}
        self.mass = {i: np.zeros(E, np.float64) for i in self.moe}
        self.token_mask = None

    def __enter__(self):
        for i, mlp in self.moe.items():
            mlp.route = self._wrap(i, mlp)
        return self

    def __exit__(self, *exc):
        for mlp in self.moe.values():
            del mlp.route

    def _wrap(self, i, mlp):
        orig = type(mlp).route

        def route(x):
            inds, w = orig(mlp, x)
            k = inds.shape[-1]
            ii = np.array(inds).reshape(-1, k)
            ww = np.array(w.astype(mx.float32)).reshape(-1, k)
            if self.token_mask is not None:
                ii, ww = ii[self.token_mask], ww[self.token_mask]
            np.add.at(self.counts[i], ii.ravel(), 1)
            np.add.at(self.mass[i], ii.ravel(), ww.ravel())
            return inds, w

        return route
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/ -q`
Expected: all pass (7 + 3 + 2 = 12).

- [ ] **Step 5: Commit**

```bash
git add laguna_mlx/hooks.py tests/test_hooks.py && git commit -m "feat: routing recorder hook"
```

---

### Task 4: Convert real weights to MLX 4-bit

**Files:**
- Create: `scripts/convert.py`

**Interfaces:**
- Produces: `models/Laguna-XS-2.1-mlx-4bit/` loadable via `import laguna_mlx; mlx_lm.load(path)`.

- [ ] **Step 1: Write `scripts/convert.py`**

```python
"""bf16 HF checkpoint -> MLX 4-bit (router gates 8-bit via Model.quant_predicate).

python scripts/convert.py [--hf models/Laguna-XS-2.1] [--out models/Laguna-XS-2.1-mlx-4bit]
"""
import argparse

import laguna_mlx  # noqa: F401  registers model_type "laguna"
from mlx_lm import convert

p = argparse.ArgumentParser()
p.add_argument("--hf", default="models/Laguna-XS-2.1")
p.add_argument("--out", default="models/Laguna-XS-2.1-mlx-4bit")
a = p.parse_args()
convert(a.hf, mlx_path=a.out, quantize=True, q_bits=4, q_group_size=64)
```

- [ ] **Step 2: Download bf16 weights (~66 GB, hours)**

```bash
df -h ~ | tail -1   # need > 90 GB free
hf download poolside/Laguna-XS-2.1 --include "*.safetensors" "model.safetensors.index.json" \
  --local-dir models/Laguna-XS-2.1
```

- [ ] **Step 3: Convert, watching peak RSS**

Run: `/usr/bin/time -l python3 scripts/convert.py 2>&1 | tail -20`
Expected: finishes; `maximum resident set size` well under 24 GB (lazy loading). If it gets OOM-killed, stop and report — fallback is converting once on a rented Linux box with mlx[cpu] (spec risk section).

- [ ] **Step 4: Verify output**

```bash
du -sh models/Laguna-XS-2.1-mlx-4bit   # expect ~18-19 GB
python3 - <<'EOF'
import json
q = json.load(open("models/Laguna-XS-2.1-mlx-4bit/config.json"))["quantization"]
assert q["bits"] == 4 and q["group_size"] == 64
assert q["model.layers.1.mlp.gate"]["bits"] == 8, "router must be 8-bit"
print("quantization config ok")
EOF
```

- [ ] **Step 5: Raise the GPU wired-memory limit (needs sudo; resets on reboot)**

The default working set is 19.07 GB and the weights are ~18.6 GB. Ask the human to run:
`! sudo sysctl iogpu.wired_limit_mb=22528`
Then confirm: `python3 -c "import mlx.core as mx; print(mx.device_info()['max_recommended_working_set_size']/1e9)"`

- [ ] **Step 6: Smoke test generation**

```bash
python3 - <<'EOF'
import laguna_mlx
from mlx_lm import load, generate
m, t = load("models/Laguna-XS-2.1-mlx-4bit")
msgs = [{"role": "user", "content": "Write a Python function that reverses a linked list."}]
prompt = t.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
print(generate(m, t, prompt=prompt, max_tokens=200, verbose=True))
EOF
```
Expected: coherent Python, decode speed printed (record it in the task report).

- [ ] **Step 7: Delete bf16 shards (keeps tokenizer/config files) and commit**

```bash
rm models/Laguna-XS-2.1/*.safetensors
git add scripts/convert.py && git commit -m "feat: convert script for MLX 4-bit"
```

---

### Task 5: Real-weight parity against llama.cpp

**Files:**
- Create: `scripts/check_parity.py`
- Output: `results/parity.json`

**Interfaces:**
- Consumes: converted model from Task 4; `llama-server` on :8080.

- [ ] **Step 1: Download the GGUF (~19 GB; bf16 must already be deleted)**

```bash
hf download poolside/Laguna-XS-2.1-GGUF --include "*Q4_K_M*" --local-dir models/gguf
ls models/gguf
```
If no `Q4_K_M` file exists, `hf download poolside/Laguna-XS-2.1-GGUF --dry-run` (or list files on the model page) and pick the closest 4-bit quant.

- [ ] **Step 2: Start llama-server and inspect one response's shape**

```bash
llama-server -m models/gguf/*.gguf --port 8080 -c 16384 &
sleep 30
curl -s localhost:8080/completion -d '{"prompt":[2,100,200],"n_predict":2,"temperature":0,"n_probs":5,"return_tokens":true}' | python3 -m json.tool | head -40
```
Confirm the response contains `tokens` (generated ids) and `completion_probabilities[i].top_logprobs[j].id`. If field names differ in this build, adjust the two lines marked `# llama-server fields` below.

- [ ] **Step 3: Write `scripts/check_parity.py`**

```python
"""Teacher-force MLX on llama.cpp's greedy continuations of 21 prompts.

Pass: MLX argmax == llama token >= 95%, and MLX argmax in llama top-5 >= 99%.
Needs: llama-server -m <gguf> --port 8080 -c 16384
"""
import json
import urllib.request
from pathlib import Path

import mlx.core as mx
import numpy as np

import laguna_mlx  # noqa: F401
from mlx_lm import load

MODEL = "models/Laguna-XS-2.1-mlx-4bit"
N_GEN = 32
PROMPTS = [
    "Write a Python function that merges two sorted lists.",
    "Explain what a race condition is, with a short Go example.",
    "Fix the bug: def avg(xs): return sum(xs) / len(xs)  # crashes on empty list",
    "Write a SQL query returning the top 3 customers by total order value.",
    "What does `git rebase --onto` do? Give an example.",
    "Implement an LRU cache in TypeScript.",
    "Write a bash one-liner that counts lines of Python code in a repo, excluding tests.",
    "Refactor this to avoid the nested loop: for a in xs:\n  for b in ys:\n    if a == b: out.append(a)",
    "Write a Rust function that parses '12:34' into (hours, minutes) with error handling.",
    "Write pytest tests for a function slugify(title: str) -> str.",
]


def llama_greedy(ids):
    body = {"prompt": ids, "n_predict": N_GEN, "temperature": 0, "n_probs": 5,
            "return_tokens": True, "cache_prompt": False}
    req = urllib.request.Request("http://127.0.0.1:8080/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    gen = r["tokens"]  # llama-server fields
    top5 = [[t["id"] for t in p["top_logprobs"]] for p in r["completion_probabilities"]]  # llama-server fields
    return gen, top5


def main():
    model, tok = load(MODEL)
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                enable_thinking=think)
        for p in PROMPTS for think in (False, True)
    ]
    long_doc = Path("tests/reference/modeling_laguna.py").read_text()  # ~9k tokens, crosses 8192
    prompts.append(tok.apply_chat_template(
        [{"role": "user", "content": "Summarize this file:\n\n" + long_doc}],
        add_generation_prompt=True, enable_thinking=False))
    rows = []
    for ids in prompts:
        gen, top5 = llama_greedy(ids)
        logits = model(mx.array([ids + gen]))[0, len(ids) - 1 : len(ids) - 1 + len(gen)]
        pred = np.array(mx.argmax(logits, axis=-1)).tolist()
        rows.append({
            "prompt_tokens": len(ids),
            "top1": float(np.mean([a == b for a, b in zip(pred, gen)])),
            "in_top5": float(np.mean([a in t for a, t in zip(pred, top5)])),
        })
        print(rows[-1])
    summary = {"top1": float(np.mean([r["top1"] for r in rows])),
               "in_top5": float(np.mean([r["in_top5"] for r in rows])), "rows": rows}
    Path("results/parity.json").write_text(json.dumps(summary, indent=1))
    print("top1", summary["top1"], "in_top5", summary["in_top5"])
    assert summary["top1"] >= 0.95 and summary["in_top5"] >= 0.99, "parity gate failed"


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run**

Run: `python3 scripts/check_parity.py`
Expected: prints `top1 >= 0.95`, `in_top5 >= 0.99`, exits 0. If only the long prompt fails, suspect YaRN (positions > 8192 use the scaled frequencies); if all fail, suspect weight loading (`sanitize`) — Task 1 already rules out architecture.

- [ ] **Step 5: Stop server, delete GGUF, commit**

```bash
kill %1; rm -rf models/gguf
git add scripts/check_parity.py results/parity.json && git commit -m "feat: real-weight parity vs llama.cpp"
```

---

### Task 6: Logging proxy

**Files:**
- Create: `scripts/proxy.py`
- Test: `tests/test_proxy.py`

**Interfaces:**
- Produces: `scripts.proxy.make_handler(upstream: str, out: str) -> type[BaseHTTPRequestHandler]`; CLI `python3 scripts/proxy.py --listen PORT --upstream URL --out FILE.jsonl`. Each POST appends `{"path": str, "request": dict|str, "status": int, "response": str}`.

- [ ] **Step 1: Write the failing test**

```python
import http.server
import json
import threading
import urllib.error
import urllib.request

from scripts.proxy import make_handler


class Upstream(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(500 if self.path == "/fail" else 200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for part in (b'data: {"a":1}\n\n', b"data: [DONE]\n\n"):
            self.wfile.write(part)
            self.wfile.flush()

    def log_message(self, *a):
        pass


def serve(handler):
    s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


def post(url, body):
    return urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}))


def test_forwards_stream_logs_requests_and_errors(tmp_path):
    up = serve(Upstream)
    log = tmp_path / "t.jsonl"
    px = serve(make_handler(f"http://127.0.0.1:{up.server_port}", str(log)))
    base = f"http://127.0.0.1:{px.server_port}"
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()

    assert post(base + "/v1/chat/completions", body).read() == b'data: {"a":1}\n\ndata: [DONE]\n\n'
    try:
        post(base + "/fail", body)
        raise AssertionError("expected HTTPError")
    except urllib.error.HTTPError as e:
        assert e.code == 500

    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert [l["status"] for l in lines] == [200, 500]
    assert lines[0]["request"]["messages"][0]["content"] == "hi"
    assert "[DONE]" in lines[0]["response"]


def test_unreachable_upstream_returns_502_and_logs(tmp_path):
    log = tmp_path / "t.jsonl"
    px = serve(make_handler("http://127.0.0.1:9", str(log)))  # port 9: nothing listening
    try:
        post(f"http://127.0.0.1:{px.server_port}/v1/chat/completions", b"{}")
        raise AssertionError("expected HTTPError")
    except urllib.error.HTTPError as e:
        assert e.code == 502
    assert json.loads(log.read_text())["status"] == 502
```

- [ ] **Step 2: Run to see it fail**

Run: `python3 -m pytest tests/test_proxy.py -q`
Expected: `ModuleNotFoundError: No module named 'scripts.proxy'`

- [ ] **Step 3: Implement `scripts/proxy.py`**

```python
"""Logging reverse proxy for an OpenAI-compatible server (streaming preserved).

python3 scripts/proxy.py --listen 11434 --upstream http://127.0.0.1:11435 --out traces/x.jsonl
"""
import argparse
import http.server
import json
import urllib.error
import urllib.request

HOP = {"host", "content-length", "transfer-encoding", "connection"}


def make_handler(upstream, out):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _relay(self, body):
            """Forward, stream the body back as chunks, return (status, raw bytes). No terminator."""
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            req = urllib.request.Request(upstream + self.path, data=body, method=self.command, headers=headers)
            try:
                resp = urllib.request.urlopen(req, timeout=3600)
            except urllib.error.HTTPError as e:
                resp = e
            except urllib.error.URLError as e:
                msg = f"proxy: upstream unreachable: {e.reason}".encode()
                self.send_response(502)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(msg), msg))
                return 502, msg
            self.send_response(resp.status)
            for k, v in resp.headers.items():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            read = getattr(resp, "read1", resp.read)
            chunks = []
            while chunk := read(65536):
                chunks.append(chunk)
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            return resp.status, b"".join(chunks)

        def _finish(self):
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            status, raw = self._relay(body)
            try:
                request = json.loads(body)
            except ValueError:
                request = body.decode(errors="replace")
            with open(out, "a") as f:  # log before finishing, so a reader never races the write
                f.write(json.dumps({"path": self.path, "request": request, "status": status,
                                    "response": raw.decode(errors="replace")}) + "\n")
            self._finish()

        def do_GET(self):
            self._relay(None)
            self._finish()

        def log_message(self, *a):
            pass

    return Handler


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--listen", type=int, required=True)
    p.add_argument("--upstream", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    http.server.ThreadingHTTPServer(("127.0.0.1", a.listen), make_handler(a.upstream, a.out)).serve_forever()
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_proxy.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/proxy.py tests/test_proxy.py && git commit -m "feat: logging proxy for session capture"
```

---

### Task 7: Session loading and tokenization

**Files:**
- Create: `replay/session.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Consumes: proxy JSONL format (Task 6); tokenizer files in `models/Laguna-XS-2.1` (Task 1).
- Produces: `load_session(path) -> (messages: list[dict], tools: list|None)`, `normalize(message) -> dict`, `tokenize(tokenizer, messages, tools, max_tokens=None) -> (ids int32[N], mask bool[N], truncated bool)`. `mask[t]` is True when token t belongs to an assistant turn.

- [ ] **Step 1: Write the failing tests**

```python
import json

import pytest
from transformers import AutoTokenizer

from replay.session import load_session, normalize, tokenize

TOK = "models/Laguna-XS-2.1"
TOOL_CALL = {"id": "c1", "type": "function",
             "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}
MESSAGES = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": [{"type": "text", "text": "list files"}]},
    {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
    {"role": "tool", "tool_call_id": "c1", "content": "a.py\nb.py"},
    {"role": "assistant", "content": "There are two files."},
]


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(TOK)


def test_normalize_handles_openai_shapes():
    assert normalize(MESSAGES[1])["content"] == "list files"
    a = normalize(MESSAGES[2])
    assert a["content"] == "" and a["tool_calls"][0]["function"]["arguments"] == {"cmd": "ls"}
    bad = {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "bash", "arguments": "{not json"}}]}
    assert normalize(bad)["tool_calls"][0]["function"]["arguments"] == {"_raw": "{not json"}


def test_load_session_uses_last_chat_request(tmp_path):
    f = tmp_path / "s.jsonl"
    lines = [
        {"path": "/v1/chat/completions", "request": {"messages": MESSAGES[:2]}, "status": 200, "response": ""},
        {"path": "/v1/chat/completions", "request": {"messages": MESSAGES, "tools": [{"x": 1}]}, "status": 200, "response": ""},
        {"path": "/api/show", "request": "not json", "status": 200, "response": ""},
    ]
    f.write_text("\n".join(json.dumps(l) for l in lines))
    msgs, tools = load_session(f)
    assert len(msgs) == 5 and tools == [{"x": 1}]


def test_mask_covers_assistant_turns_only(tok):
    msgs = [normalize(m) for m in MESSAGES]
    ids, mask, truncated = tokenize(tok, msgs, None)
    assert not truncated and len(ids) == len(mask)
    inside, outside = tok.decode(ids[mask]), tok.decode(ids[~mask])
    assert "<tool_call>bash" in inside and "There are two files." in inside
    assert "list files" in outside and "a.py" in outside and "list files" not in inside


def test_truncation_is_flagged(tok):
    msgs = [normalize(m) for m in MESSAGES]
    ids, mask, truncated = tokenize(tok, msgs, None, max_tokens=10)
    assert truncated and len(ids) == 10 and len(mask) == 10
```

- [ ] **Step 2: Run to see them fail**

Run: `python3 -m pytest tests/test_session.py -q`
Expected: `ModuleNotFoundError: No module named 'replay.session'`

- [ ] **Step 3: Implement `replay/session.py`**

```python
"""Captured pool session (proxy JSONL) -> Laguna chat-template tokens + assistant mask."""
import json
from pathlib import Path

import numpy as np


def normalize(m):
    """OpenAI wire format -> what Laguna's chat template expects."""
    m = dict(m)
    c = m.get("content")
    if isinstance(c, list):
        m["content"] = "".join(p.get("text", "") for p in c if isinstance(p, dict))
    elif c is None:
        m["content"] = ""
    if m.get("tool_calls"):
        calls = []
        for tc in m["tool_calls"]:
            args = tc["function"]["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"_raw": args}  # model emitted malformed JSON; keep the text
            calls.append({**tc, "function": {**tc["function"], "arguments": args}})
        m["tool_calls"] = calls
    return m


def load_session(path):
    """(messages, tools) of the last chat request: it holds the whole conversation so far."""
    reqs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    reqs = [r for r in reqs if r["path"].endswith("/chat/completions") and isinstance(r["request"], dict)]
    if not reqs:
        raise ValueError(f"{path}: no chat requests")
    last = reqs[-1]["request"]
    return [normalize(m) for m in last["messages"]], last.get("tools")


def tokenize(tokenizer, messages, tools, max_tokens=None):
    enc = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=True, return_dict=True,
        return_assistant_tokens_mask=True, enable_thinking=True,
    )
    ids = np.array(enc["input_ids"], np.int32)
    mask = np.array(enc["assistant_masks"], bool)
    truncated = max_tokens is not None and len(ids) > max_tokens
    if truncated:
        ids, mask = ids[:max_tokens], mask[:max_tokens]
    return ids, mask, truncated
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_session.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add replay/session.py tests/test_session.py && git commit -m "feat: session loading and assistant-masked tokenization"
```

---

### Task 8: Capture ~50 pool sessions on SWE-bench Lite

**Files:**
- Create: `scripts/gen_traces.py`
- Output: `traces/<instance_id>.jsonl` (gitignored), `results/traces_summary.json` (committed)

**Interfaces:**
- Consumes: `scripts/proxy.py` CLI; `replay.session.load_session`.

> **Safety:** `pool exec --unsafe-auto-allow` runs shell commands the model chooses, on this Mac, without asking. Workspaces live in `~/laguna-work`. Ask the human before the first run; recommend running under a separate macOS user account.

- [ ] **Step 1: Get data, pool, and the ollama model**

```bash
hf download princeton-nlp/SWE-bench_Lite --repo-type dataset --local-dir data/swe-bench-lite
ls data/swe-bench-lite/data/test-*.parquet
curl -fsSL https://raw.githubusercontent.com/poolsideai/pool/main/install.sh | sh   # confirm command on github.com/poolsideai/pool README first
ollama pull laguna-xs-2.1   # if the tag differs, find it on ollama.com/library
ollama launch pool --model laguna-xs-2.1   # configures pool for ollama at :11434; quit once it starts
```

- [ ] **Step 2: Write `scripts/gen_traces.py`**

```python
"""Run pool on SWE-bench Lite tasks; the proxy captures every model request.

Prereqs (separate terminal): OLLAMA_HOST=127.0.0.1:11435 ollama serve
pool is configured for :11434, where this script puts the proxy.
python3 scripts/gen_traces.py --n 50      # resumable: skips finished tasks
"""
import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from replay.session import load_session

PROMPT = ("Resolve the following GitHub issue in the repository in the current directory. "
          "Make the minimal code change needed.\n\n{issue}")

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=50)
p.add_argument("--work", default=str(Path.home() / "laguna-work"))
p.add_argument("--traces", default="traces")
p.add_argument("--timeout", type=int, default=3600)
a = p.parse_args()

work, traces = Path(a.work), Path(a.traces)
traces.mkdir(exist_ok=True)
df = pd.read_parquet(glob.glob("data/swe-bench-lite/data/test-*.parquet")[0])
df = df.sort_values("instance_id").head(a.n)

for row in df.itertuples():
    out = traces / f"{row.instance_id}.jsonl"
    if out.exists():
        continue
    mirror = work / "_repos" / row.repo.replace("/", "__")
    if not mirror.exists():
        subprocess.run(["git", "clone", "--quiet", f"https://github.com/{row.repo}.git", str(mirror)], check=True)
    wt = work / row.instance_id
    if not wt.exists():
        subprocess.run(["git", "-C", str(mirror), "worktree", "add", "--detach", str(wt), row.base_commit], check=True)
    partial = out.with_suffix(".partial")
    proxy = subprocess.Popen([sys.executable, "scripts/proxy.py", "--listen", "11434",
                              "--upstream", "http://127.0.0.1:11435", "--out", str(partial)])
    time.sleep(1)
    t0 = time.time()
    try:
        r = subprocess.run(["pool", "exec", "-p", PROMPT.format(issue=row.problem_statement),
                            "-o", "json", "--unsafe-auto-allow"],
                           cwd=wt, timeout=a.timeout, capture_output=True, text=True)
        (traces / f"{row.instance_id}.pool.json").write_text(r.stdout)
        status = f"exit {r.returncode}"
    except subprocess.TimeoutExpired:
        status = "timeout"
    finally:
        proxy.terminate()
        proxy.wait()
    if partial.exists():
        partial.rename(out)
    print(f"{row.instance_id}: {status} in {time.time() - t0:.0f}s", flush=True)

summary = []
for f in sorted(traces.glob("*.jsonl")):
    try:
        msgs, _ = load_session(f)
        summary.append({"session": f.stem, "messages": len(msgs),
                        "assistant_turns": sum(m["role"] == "assistant" for m in msgs)})
    except ValueError as e:
        summary.append({"session": f.stem, "error": str(e)})
Path("results/traces_summary.json").write_text(json.dumps(summary, indent=1))
```

- [ ] **Step 3: One-task smoke run**

```bash
OLLAMA_HOST=127.0.0.1:11435 ollama serve &   # or in another terminal
python3 scripts/gen_traces.py --n 1
python3 -c "
from replay.session import load_session, tokenize
from transformers import AutoTokenizer
import glob
f = sorted(glob.glob('traces/*.jsonl'))[0]
m, t = load_session(f)
ids, mask, _ = tokenize(AutoTokenizer.from_pretrained('models/Laguna-XS-2.1'), m, t)
print(len(m), 'messages', len(ids), 'tokens', mask.sum(), 'assistant tokens')"
```
Expected: more than 2 messages, some thousands of tokens, nonzero assistant tokens. If `traces/` is empty, pool is not talking to :11434 — check the pool config that `ollama launch` wrote and point its base URL at `http://127.0.0.1:11434/v1`.

- [ ] **Step 4: Overnight run**

Run: `nohup python3 scripts/gen_traces.py --n 50 > traces/run.log 2>&1 &`
Next morning: `tail traces/run.log; cat results/traces_summary.json | python3 -m json.tool | head -30`
Expected: ≥ 40 sessions with no `error` field.

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_traces.py results/traces_summary.json && git commit -m "feat: pool session capture on SWE-bench Lite"
```

---

### Task 9: Replay metric

**Files:**
- Create: `replay/replay.py`
- Test: `tests/test_replay.py`

**Interfaces:**
- Consumes: `load_session`, `tokenize` (Task 7); `model.model`, `model.lm_head`, `model.make_cache()` (Task 1).
- Produces: `build_ref(model, tokenizer, session_path, out_path, max_tokens=65536, n_actions=2, max_new=1024, chunk=2048, seed=0)` writes `.npz`; `score(model, ref_path, chunk=2048) -> dict`; `score_actions(model, tokenizer, ref_path, max_new=None) -> float` (nan when the ref has no actions); CLI `python3 -m replay.replay {build-ref,score} ...`. `score` dict keys: `tokens, assistant_tokens, truncated, kl_top64_mean, kl_top64_p99, top1_agree, peak_mem_gb, tok_per_s`.

- [ ] **Step 1: Write the failing tests**

```python
import json

import mlx.core as mx
import numpy as np
import pytest
from transformers import AutoTokenizer

from replay.replay import build_ref, score, score_actions
from tests.tiny import mlx_model, torch_model

MESSAGES = [
    {"role": "user", "content": "list files"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}]},
    {"role": "tool", "content": "a.py\nb.py"},
    {"role": "assistant", "content": "There are two files, a.py and b.py."},
]


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("models/Laguna-XS-2.1")


@pytest.fixture
def session(tmp_path):
    f = tmp_path / "s.jsonl"
    f.write_text(json.dumps({"path": "/v1/chat/completions", "request": {"messages": MESSAGES},
                             "status": 200, "response": ""}))
    return f


def test_self_score_is_perfect_and_chunking_invariant(tok, session, tmp_path):
    m = mlx_model(torch_model())
    ref = tmp_path / "s.npz"
    build_ref(m, tok, session, ref, n_actions=1, max_new=8)
    for chunk in (2048, 7):
        s = score(m, ref, chunk=chunk)
        assert abs(s["kl_top64_mean"]) < 1e-5 and s["top1_agree"] == 1.0
        assert s["assistant_tokens"] > 0 and not s["truncated"]
    assert score_actions(m, tok, ref, max_new=8) == 1.0


def test_perturbed_model_scores_worse(tok, session, tmp_path):
    ref = tmp_path / "s.npz"
    build_ref(mlx_model(torch_model()), tok, session, ref, n_actions=0)
    s = score(mlx_model(torch_model(seed=1)), ref)
    # tiny random models are near-uniform, so truncated KL is small; argmax flips are not
    assert s["top1_agree"] < 0.5 and s["kl_top64_mean"] != 0


def test_long_session_is_truncated_not_fatal(tok, session, tmp_path):
    ref = tmp_path / "s.npz"
    build_ref(mlx_model(torch_model()), tok, session, ref, max_tokens=30, n_actions=0)
    s = score(mlx_model(torch_model()), ref)
    assert s["truncated"] and s["tokens"] == 30
```

- [ ] **Step 2: Run to see them fail**

Run: `python3 -m pytest tests/test_replay.py -q`
Expected: `ModuleNotFoundError: No module named 'replay.replay'`

- [ ] **Step 3: Implement `replay/replay.py`**

```python
"""Teacher-forced replay: compare any model to the unmodified reference on captured sessions.

The reference .npz stores the token ids, so every scored model sees identical inputs.
KL is truncated to the reference's top-64 tokens (>99.9% of mass in practice).

python3 -m replay.replay build-ref --model M --traces traces --refs refs
python3 -m replay.replay score --model M --refs refs --out results/x.json [--actions]
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from replay.session import load_session, tokenize

TOPK = 64
SUB = 256  # positions per log-softmax batch: 256 x 100k vocab x 4 B = 100 MB


def assistant_logprobs(model, ids, mask, chunk=2048):
    """Yield (positions, logprobs[n, V] float32) for positions whose next token is assistant."""
    target = np.zeros_like(mask)
    target[:-1] = mask[1:]
    cache = model.make_cache()
    for a in range(0, len(ids), chunk):
        b = min(a + chunk, len(ids))
        h = model.model(mx.array(ids[None, a:b]), cache=cache)[0]
        sel = np.nonzero(target[a:b])[0]
        if len(sel) == 0:
            mx.eval(h)
            continue
        for s in range(0, len(sel), SUB):
            part = sel[s : s + SUB]
            lp = model.lm_head(h[mx.array(part)]).astype(mx.float32)
            lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
            mx.eval(lp)
            yield part + a, lp


def _action(text):
    """The tool call if there is one, else the whole reply: what an agent would execute."""
    i, j = text.find("<tool_call>"), text.find("</tool_call>")
    return text[i : j + len("</tool_call>")] if 0 <= i < j else text.strip()


def _greedy_actions(model, tokenizer, ids, starts, max_new):
    from mlx_lm import generate

    # ponytail: re-prefills the whole prefix per turn; reuse a trimmed cache if this is too slow
    return [_action(generate(model, tokenizer, prompt=ids[:s].tolist(), max_tokens=max_new))
            for s in starts]


def build_ref(model, tokenizer, session_path, out_path, max_tokens=65536, n_actions=2,
              max_new=1024, chunk=2048, seed=0):
    msgs, tools = load_session(session_path)
    ids, mask, truncated = tokenize(tokenizer, msgs, tools, max_tokens)
    pos, top_ids, top_lp = [], [], []
    for p, lp in assistant_logprobs(model, ids, mask, chunk):
        idx = mx.argpartition(-lp, kth=TOPK - 1, axis=-1)[:, :TOPK]
        vals = mx.take_along_axis(lp, idx, axis=-1)
        order = mx.argsort(-vals, axis=-1)
        pos.append(p)
        top_ids.append(np.array(mx.take_along_axis(idx, order, axis=-1)))
        top_lp.append(np.array(mx.take_along_axis(vals, order, axis=-1)))
    if not pos:
        raise ValueError(f"{session_path}: no assistant tokens")
    starts = np.nonzero(mask & ~np.r_[False, mask[:-1]])[0]
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.choice(starts, min(n_actions, len(starts)), replace=False))
    actions = _greedy_actions(model, tokenizer, ids, starts, max_new)
    np.savez_compressed(out_path, ids=ids, mask=mask, truncated=truncated,
                        pos=np.concatenate(pos), top_ids=np.concatenate(top_ids),
                        top_lp=np.concatenate(top_lp), action_starts=starts,
                        actions=json.dumps(actions), max_new=max_new)


def score(model, ref_path, chunk=2048):
    ref = np.load(ref_path)
    ids, mask = ref["ids"], ref["mask"]
    kls, agree, i = [], [], 0
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    for p, lp in assistant_logprobs(model, ids, mask, chunk):
        n = len(p)
        assert (p == ref["pos"][i : i + n]).all(), "position mismatch vs reference"
        tid = mx.array(ref["top_ids"][i : i + n])
        rlp = mx.array(ref["top_lp"][i : i + n])
        i += n
        mlp_ = mx.take_along_axis(lp, tid, axis=-1)
        kls.append(np.array((mx.exp(rlp) * (rlp - mlp_)).sum(axis=-1)))
        agree.append(np.array(mx.argmax(lp, axis=-1) == tid[:, 0]))
    dt = time.perf_counter() - t0
    kl, ag = np.concatenate(kls), np.concatenate(agree)
    return {"tokens": int(len(ids)), "assistant_tokens": int(len(kl)),
            "truncated": bool(ref["truncated"]), "kl_top64_mean": float(kl.mean()),
            "kl_top64_p99": float(np.percentile(kl, 99)), "top1_agree": float(ag.mean()),
            "peak_mem_gb": mx.get_peak_memory() / 1e9, "tok_per_s": len(ids) / dt}


def score_actions(model, tokenizer, ref_path, max_new=None):
    ref = np.load(ref_path)
    want = json.loads(str(ref["actions"]))
    if not want:
        return float("nan")
    got = _greedy_actions(model, tokenizer, ref["ids"], ref["action_starts"],
                          max_new or int(ref["max_new"]))
    return float(np.mean([a == b for a, b in zip(got, want)]))


def main():
    import laguna_mlx  # noqa: F401
    from mlx_lm import load

    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build-ref", "score"])
    p.add_argument("--model", required=True)
    p.add_argument("--traces", default="traces")
    p.add_argument("--refs", default="refs")
    p.add_argument("--out")
    p.add_argument("--max-tokens", type=int, default=65536)
    p.add_argument("--actions", action="store_true")
    a = p.parse_args()
    model, tok = load(a.model)
    refs = Path(a.refs)
    if a.cmd == "build-ref":
        refs.mkdir(exist_ok=True)
        for f in sorted(Path(a.traces).glob("*.jsonl")):
            out = refs / f"{f.stem}.npz"
            if not out.exists():
                build_ref(model, tok, f, out, a.max_tokens)
                print("ref", f.stem, flush=True)
        return
    rows = []
    for r in sorted(refs.glob("*.npz")):
        row = {"session": r.stem, **score(model, r)}
        if a.actions:
            row["action_match"] = score_actions(model, tok, r)
        rows.append(row)
        print(row, flush=True)
    w = np.array([x["assistant_tokens"] for x in rows])
    summary = {k: float(np.average([x[k] for x in rows], weights=w))
               for k in ("kl_top64_mean", "top1_agree")}
    summary["peak_mem_gb"] = max(x["peak_mem_gb"] for x in rows)
    if a.actions:
        summary["action_match"] = float(np.nanmean([x["action_match"] for x in rows]))
    Path(a.out).write_text(json.dumps({"summary": summary, "sessions": rows}, indent=1))
    print(summary)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Build reference and self-check on real sessions**

```bash
python3 -m replay.replay build-ref --model models/Laguna-XS-2.1-mlx-4bit      # hours; resumable
python3 -m replay.replay score --model models/Laguna-XS-2.1-mlx-4bit --out results/self-check.json --actions
```
Expected summary: `kl_top64_mean` < 1e-5, `top1_agree` == 1.0, `action_match` == 1.0. Anything else means nondeterminism in the forward pass: stop and investigate before any Part 2/3 experiment.

- [ ] **Step 6: Commit**

```bash
git add replay/replay.py tests/test_replay.py results/self-check.json && git commit -m "feat: teacher-forced replay metric with reference self-check"
```

---

### Task 10: Baseline memory and speed numbers

**Files:**
- Create: `scripts/baseline.py`
- Output: `results/baseline.json`

- [ ] **Step 1: Write `scripts/baseline.py`**

```python
"""Weights, KV bytes/token, prefill/decode speed, projected max context -> results/baseline.json"""
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

import laguna_mlx  # noqa: F401
from mlx_lm import load

MODEL = "models/Laguna-XS-2.1-mlx-4bit"
model, _ = load(MODEL)
weights_gb = sum(v.nbytes for _, v in tree_flatten(model.parameters())) / 1e9
limit_gb = mx.device_info()["max_recommended_working_set_size"] / 1e9

rows = []
for ctx in (1024, 8192, 32768):
    mx.clear_cache()
    mx.reset_peak_memory()
    cache = model.make_cache()
    ids = mx.array(np.random.default_rng(0).integers(1000, 50000, size=(1, ctx)))
    t0 = time.perf_counter()
    for a in range(0, ctx, 2048):
        mx.eval(model.model(ids[:, a : a + 2048], cache=cache))
    prefill = ctx / (time.perf_counter() - t0)
    tok = mx.array([[1000]])
    t0 = time.perf_counter()
    for _ in range(64):
        mx.eval(model(tok, cache=cache))
    decode = 64 / (time.perf_counter() - t0)
    rows.append({"ctx": ctx, "prefill_tok_s": prefill, "decode_tok_s": decode,
                 "kv_gb": sum(c.nbytes for c in cache) / 1e9,
                 "peak_gb": mx.get_peak_memory() / 1e9})
    print(rows[-1], flush=True)

kv_per_tok = (rows[-1]["kv_gb"] - rows[0]["kv_gb"]) / (rows[-1]["ctx"] - rows[0]["ctx"])
# ponytail: linear KV extrapolation ignores activation spikes at prefill; measure directly in Part 3
max_ctx = int((limit_gb - weights_gb - 0.5) / kv_per_tok)
out = {"weights_gb": weights_gb, "gpu_limit_gb": limit_gb, "kv_kb_per_token": kv_per_tok * 1e6,
       "projected_max_context": max_ctx, "rows": rows}
Path("results/baseline.json").write_text(json.dumps(out, indent=1))
print(out)
```

- [ ] **Step 2: Run**

Run: `python3 scripts/baseline.py`
Expected: `kv_kb_per_token` ≈ 41 (10 full layers × 8 kv heads × 128 dims × 2 (K,V) × 2 bytes), `weights_gb` ≈ 18.6. A KV figure far from 41 means sliding layers are not using `RotatingKVCache` — fix before continuing.

- [ ] **Step 3: Commit**

```bash
git add scripts/baseline.py results/baseline.json && git commit -m "feat: baseline memory and speed numbers"
```

---

### Task 11: Prepare the mlx-lm upstream PR

**Files (in a separate clone, not this repo):**
- Create: `mlx-lm/mlx_lm/models/laguna.py`
- Modify: `mlx-lm/tests/test_models.py`

> Opening the PR is public and outward-facing: prepare the branch, then stop and ask the human before `gh pr create`.

- [ ] **Step 1: Branch and copy the model with relative imports**

```bash
cd ~/Projects/Current && gh repo fork ml-explore/mlx-lm --clone && cd mlx-lm
git checkout -b add-laguna
sed -e 's/from mlx_lm\.models\./from ./' ../lagunaLab/laguna_mlx/laguna.py > mlx_lm/models/laguna.py
grep -n "^from" mlx_lm/models/laguna.py   # expect: from .base / .cache / .rope_utils / .switch_layers
```

- [ ] **Step 2: Add a test next to the other model tests in `tests/test_models.py`**

```python
    def test_laguna(self):
        from mlx_lm.models import laguna

        args = laguna.ModelArgs.from_dict({
            "model_type": "laguna", "vocab_size": 1000, "hidden_size": 64,
            "intermediate_size": 128, "num_hidden_layers": 4, "num_attention_heads": 4,
            "num_attention_heads_per_layer": [4, 8, 8, 8], "num_key_value_heads": 2,
            "head_dim": 16, "rms_norm_eps": 1e-6, "num_experts": 8, "num_experts_per_tok": 2,
            "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
            "gating": "per-head", "sliding_window": 8, "moe_routed_scaling_factor": 2.5,
            "layer_types": ["full_attention"] + ["sliding_attention"] * 3,
            "rope_parameters": {
                "full_attention": {"rope_type": "yarn", "rope_theta": 500000.0, "factor": 32.0,
                                   "original_max_position_embeddings": 8192, "beta_fast": 64.0,
                                   "beta_slow": 1.0, "partial_rotary_factor": 0.5},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            },
        })
        model = laguna.Model(args)
        self.model_test_runner(model, args.model_type, args.vocab_size, args.num_hidden_layers)
```

- [ ] **Step 3: Run their tests**

Run: `python3 -m pytest tests/test_models.py -q -k laguna`
Expected: 1 passed.

- [ ] **Step 4: Commit on the fork branch; stop for approval**

```bash
git add mlx_lm/models/laguna.py tests/test_models.py
git commit -m "Add Poolside Laguna (XS 2.1 / S 2.1 / M.1)"
```
Then ask the human to approve pushing and opening the PR. The PR body cites `results/parity.json` numbers (exact on tiny config vs Poolside's PyTorch code; top-1 agreement vs llama.cpp on real weights).

---

## Addendum: inference optimization (Tasks 12–15)

Spec: "Addendum (2026-09-25): inference optimization" in the spec file. Execute after Task 10.

Additional Review Focus for this addendum:
- **Rejected drafts after the sliding window has wrapped** (context > 512) — trimming must leave caches exactly as if the rejected tokens were never fed. Pinned in Task 12.
- **EOS inside an accepted draft run, and `max_tokens` landing mid-block** — generation must stop at the same token plain greedy stops at. Pinned in Task 13.

### Task 12: Trimmable sliding-window cache and hidden-state capture

**Files:**
- Create: `laguna_mlx/cache.py`
- Modify: `laguna_mlx/laguna.py` (`LagunaModel.__call__` gains `capture=None`)
- Test: `tests/test_spec_cache.py`

**Interfaces:**
- Produces: `TrimmableWindowCache(window: int, slack: int = 64)` with `update_and_fetch`, `make_mask`, `offset`, `is_trimmable() -> True`, `trim(n) -> int`, `nbytes`; `make_trimmable_cache(model) -> list` (KVCache for full layers, TrimmableWindowCache for sliding); `LagunaModel.__call__(inputs, cache=None, capture: dict[int, Any] | None = None)` fills `capture[i]` with the output of layer `i` for each key present.

- [ ] **Step 1: Write the failing tests `tests/test_spec_cache.py`**

```python
import mlx.core as mx
import numpy as np

from laguna_mlx.cache import TrimmableWindowCache, make_trimmable_cache
from tests.tiny import TINY, mlx_model, torch_model


def _ids(n, seed=4):
    return mx.array(np.random.default_rng(seed).integers(0, TINY["vocab_size"], size=(1, n)))


def _cache(model):
    # slack 4 with window 8, so truncation happens constantly in a 30-token test
    return [TrimmableWindowCache(8, slack=4) if isinstance(c, TrimmableWindowCache) else c
            for c in make_trimmable_cache(model)]


def test_decode_matches_full_forward_with_truncation():
    m = mlx_model(torch_model())
    ids = _ids(30)
    full = np.array(m(ids))
    cache = _cache(m)
    outs = [np.array(m(ids[:, a:b], cache=cache))
            for a, b in [(0, 5), (5, 12), (12, 13)] + [(i, i + 1) for i in range(13, 30)]]
    np.testing.assert_allclose(np.concatenate(outs, axis=1), full, atol=1e-4, rtol=1e-3)


def test_trim_after_window_wrap_forgets_rejected_tokens():
    m = mlx_model(torch_model())
    ids = _ids(30)
    full = np.array(m(ids))
    cache = _cache(m)
    m(ids[:, :20], cache=cache)
    m(_ids(5, seed=99), cache=cache)  # "rejected draft"
    for c in cache:
        assert c.trim(5) == 5
    out = np.array(m(ids[:, 20:], cache=cache))
    np.testing.assert_allclose(out, full[:, 20:], atol=1e-4, rtol=1e-3)


def test_capture_returns_requested_layer_outputs():
    m = mlx_model(torch_model())
    cap = dict.fromkeys([1, 3])
    h = m.model(_ids(6), capture=cap)
    assert cap[1].shape == (1, 6, TINY["hidden_size"])
    assert mx.allclose(m.model.norm(cap[3]), h).item()
```

- [ ] **Step 2: Run to see them fail**

Run: `python3 -m pytest tests/test_spec_cache.py -q`
Expected: `ModuleNotFoundError: No module named 'laguna_mlx.cache'`

- [ ] **Step 3: Implement `laguna_mlx/cache.py`**

```python
"""Sliding-window KV cache that can be trimmed, for speculative decoding."""
import mlx.core as mx
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import KVCache, _BaseCache


class TrimmableWindowCache(_BaseCache):
    """Keeps the last `window + slack` keys; `offset` stays the absolute position.

    mlx-lm's RotatingKVCache can't trim once it wraps, and speculative decoding
    trims after every rejected draft. Trimming up to `slack` tokens beyond the
    latest update is always safe here.
    """

    def __init__(self, window, slack=64):
        self.window, self.slack = window, slack
        self.keys = self.values = None
        self.offset = 0

    def _stored(self):
        return 0 if self.keys is None else min(self.keys.shape[2], self.window + self.slack)

    def update_and_fetch(self, keys, values):
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            keep = self.window + self.slack
            self.keys = mx.concatenate([self.keys[..., -keep:, :], keys], axis=2)
            self.values = mx.concatenate([self.values[..., -keep:, :], values], axis=2)
        self.offset += keys.shape[2]
        return self.keys, self.values

    def make_mask(self, N, return_array=False, window_size=None):
        return create_causal_mask(N, offset=self._stored(), window_size=window_size or self.window)

    @property
    def state(self):
        return self.keys, self.values

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(n, 0 if self.keys is None else self.keys.shape[2])
        if n:
            self.keys, self.values = self.keys[..., :-n, :], self.values[..., :-n, :]
            self.offset -= n
        return n

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        return 0 if self.keys is None else self.keys.nbytes + self.values.nbytes


def make_trimmable_cache(model, slack=64):
    return [
        TrimmableWindowCache(model.args.sliding_window, slack)
        if lt == "sliding_attention" else KVCache()
        for lt in model.args.layer_types
    ]
```

- [ ] **Step 4: Add `capture` to `LagunaModel.__call__` in `laguna_mlx/laguna.py`**

Replace the method with:
```python
    def __call__(self, inputs, cache=None, capture=None):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        masks = {}
        for lt, c in zip(self.layer_types, cache):
            if lt not in masks:
                window = self.sliding_window if lt == "sliding_attention" else None
                masks[lt] = create_attention_mask(h, c, window_size=window)
        for i, (layer, lt, c) in enumerate(zip(self.layers, self.layer_types, cache)):
            h = layer(h, masks[lt], c)
            if capture is not None and i in capture:
                capture[i] = h
        return self.norm(h)
```

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest -q`
Expected: all pass (earlier tasks' tests unchanged). If the trim test fails but the decode test passes, `make_mask` and `update_and_fetch` disagree on how many keys are stored — both must use `window + slack` for old keys.

- [ ] **Step 6: Commit**

```bash
git add laguna_mlx/cache.py laguna_mlx/laguna.py tests/test_spec_cache.py
git commit -m "feat: trimmable sliding-window cache and layer-output capture"
```

---

### Task 13: DFlash draft model and lossless speculative loop

**Files:**
- Create: `laguna_mlx/dflash.py`
- Test: `tests/test_dflash.py`

**Interfaces:**
- Consumes: `TrimmableWindowCache`, `make_trimmable_cache`, `LagunaModel.__call__(..., capture=)` (Task 12); `laguna.ModelArgs`, `laguna.DecoderLayer` (Task 1).
- Produces: `DFlashDraft(config: dict)` with `.target_layer_ids`, `.block_size`, `.mask_token_id`, `make_cache()`, `context_states(aux: list) -> mx.array`, `add_context(ctx, cache)`, `propose(target, token: int, cache, n_draft: int) -> list[int]`, `sanitize(weights)`; `load_draft(path, bits=None) -> DFlashDraft`; `generate_greedy(target, prompt, max_tokens, eos=()) -> list[int]`; `generate_dflash(target, draft, prompt, max_tokens, eos=(), n_draft=None, prefill_step=2048) -> (list[int], dict)` with stats keys `steps, drafted, accepted, mean_accepted`.

- [ ] **Step 1: Write the failing tests `tests/test_dflash.py`**

```python
import copy

import mlx.core as mx
import numpy as np
import pytest

from laguna_mlx.dflash import DFlashDraft, generate_dflash, generate_greedy
from tests.tiny import TINY, mlx_model, torch_model

DRAFT_CFG = {
    "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
    "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 16,
    "rms_norm_eps": 1e-6, "rope_theta": 500000.0, "sliding_window": 8,
    "vocab_size": TINY["vocab_size"], "gating": "per-head",
    "layer_types": ["sliding_attention"] * 2,
    "dflash_config": {"block_size": 6, "mask_token_id": 12, "target_layer_ids": [1, 3], "causal": True},
}
PROMPT = list(np.random.default_rng(5).integers(0, TINY["vocab_size"], size=13))


@pytest.fixture(scope="module")
def target():
    m = mlx_model(torch_model())
    m.lm_head.weight = m.lm_head.weight * 50  # peaky logits: no float near-ties in argmax
    return m


@pytest.fixture(scope="module")
def draft():
    d = DFlashDraft(copy.deepcopy(DRAFT_CFG))
    mx.eval(d.parameters())
    return d


class OracleDraft:
    """Proposes the true greedy continuation, corrupted from slot `wrong` on."""

    def __init__(self, real, truth, wrong=None):
        self.real, self.truth, self.wrong = real, truth, wrong
        self.target_layer_ids, self.block_size = real.target_layer_ids, real.block_size

    def make_cache(self):
        return self.real.make_cache()

    def context_states(self, aux):
        return self.real.context_states(aux)

    def add_context(self, ctx, cache):
        self.real.add_context(ctx, cache)

    def propose(self, target, token, cache, n_draft):
        pos = cache[0].offset  # position of `token`
        out = [self.truth[pos + i] if pos + i < len(self.truth) else 0 for i in range(1, n_draft + 1)]
        if self.wrong is not None:
            out[self.wrong:] = [(t + 1) % TINY["vocab_size"] for t in out[self.wrong:]]
        return out


def test_random_draft_is_lossless(target, draft):
    want = generate_greedy(target, PROMPT, 25)
    got, stats = generate_dflash(target, draft, PROMPT, 25)
    assert got == want and stats["steps"] > 0


@pytest.mark.parametrize("wrong,expect_accept", [(None, 5), (2, 2)])
def test_oracle_draft_accepts_and_stays_lossless(target, draft, wrong, expect_accept):
    want = generate_greedy(target, PROMPT, 40)
    truth = PROMPT + want
    got, stats = generate_dflash(target, OracleDraft(draft, truth, wrong), PROMPT, 40)
    assert got == want
    assert stats["mean_accepted"] >= expect_accept - 0.5


def test_eos_inside_accepted_run_stops_like_greedy(target, draft):
    free = generate_greedy(target, PROMPT, 40)
    eos = (free[17],)  # a token that will arrive mid-block with the oracle
    want = generate_greedy(target, PROMPT, 40, eos=eos)
    got, _ = generate_dflash(target, OracleDraft(draft, PROMPT + free), PROMPT, 40, eos=eos)
    assert got == want and got[-1] == eos[0]


def test_max_tokens_mid_block(target, draft):
    want = generate_greedy(target, PROMPT, 23)
    free = generate_greedy(target, PROMPT, 40)
    got, _ = generate_dflash(target, OracleDraft(draft, PROMPT + free), PROMPT, 23)
    assert got == want
```

- [ ] **Step 2: Run to see them fail**

Run: `python3 -m pytest tests/test_dflash.py -q`
Expected: `ModuleNotFoundError: No module named 'laguna_mlx.dflash'`

- [ ] **Step 3: Implement `laguna_mlx/dflash.py`**

```python
"""DFlash speculative decoding for Laguna on MLX (greedy, lossless).

Draft: dense Laguna layers conditioned on target hidden states from
`target_layer_ids` (per-slice RMSNorm -> fc -> hidden_norm), injected as extra
K/V in every draft layer. One draft pass over [token, mask x n] proposes n tokens;
block slot i predicts the token at position(token) + i.
References: vLLM PR #46853 (laguna_dflash.py, qwen3_dflash.py);
speculators/models/dflash/core.py (training-time alignment).
"""
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask

from laguna_mlx import laguna
from laguna_mlx.cache import TrimmableWindowCache, make_trimmable_cache


class DFlashDraft(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        d = config["dflash_config"]
        self.target_layer_ids = list(d["target_layer_ids"])
        self.block_size = d["block_size"]
        self.mask_token_id = d["mask_token_id"]
        n = config["num_hidden_layers"]
        self.args = laguna.ModelArgs.from_dict({
            **config, "model_type": "laguna", "num_experts": 0, "num_experts_per_tok": 0,
            "moe_intermediate_size": 0, "shared_expert_intermediate_size": 0,
            "mlp_only_layers": list(range(n)),
            # draft config ships only rope_theta: default RoPE, full rotary (as vLLM reads it)
            "rope_parameters": config.get("rope_parameters")
            or {"rope_type": "default", "rope_theta": config["rope_theta"]},
        })
        h = self.args.hidden_size
        self.layers = [laguna.DecoderLayer(self.args, i) for i in range(n)]
        self.aux_hidden_norms = [nn.RMSNorm(h, eps=self.args.rms_norm_eps) for _ in self.target_layer_ids]
        self.fc = nn.Linear(h * len(self.target_layer_ids), h, bias=False)
        self.hidden_norm = nn.RMSNorm(h, eps=self.args.rms_norm_eps)
        self.norm = nn.RMSNorm(h, eps=self.args.rms_norm_eps)

    def make_cache(self):
        return [TrimmableWindowCache(self.args.sliding_window, slack=self.block_size) for _ in self.layers]

    def context_states(self, aux):
        x = mx.concatenate([norm(a) for norm, a in zip(self.aux_hidden_norms, aux)], axis=-1)
        return self.hidden_norm(self.fc(x))

    def add_context(self, ctx, cache):
        """Project context states to K/V in every draft layer and append them."""
        B, L, _ = ctx.shape
        for layer, c in zip(self.layers, cache):
            a = layer.self_attn
            x = layer.input_layernorm(ctx)
            k = a.k_norm(a.k_proj(x).reshape(B, L, a.n_kv_heads, -1)).transpose(0, 2, 1, 3)
            v = a.v_proj(x).reshape(B, L, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
            c.update_and_fetch(a.rope(k, offset=c.offset), v)

    def propose(self, target, token, cache, n_draft):
        ids = mx.array([[token] + [self.mask_token_id] * n_draft])
        h = target.model.embed_tokens(ids)
        mask = create_attention_mask(h, cache[0], window_size=self.args.sliding_window)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        drafts = mx.argmax(target.lm_head(self.norm(h[:, 1:])), axis=-1)[0].tolist()
        for c in cache:
            c.trim(n_draft + 1)  # only context K/V persists between steps
        return drafts

    def sanitize(self, weights):
        q = self.args.num_attention_heads * self.args.head_dim
        kv = self.args.num_key_value_heads * self.args.head_dim
        for i in range(len(self.layers)):
            p = f"layers.{i}.self_attn."
            qkv = weights.pop(p + "qkv_proj.weight", None)
            if qkv is not None:
                weights[p + "q_proj.weight"] = qkv[:q]
                weights[p + "k_proj.weight"] = qkv[q : q + kv]
                weights[p + "v_proj.weight"] = qkv[q + kv :]
        return weights


def load_draft(path, bits=None):
    path = Path(path)
    d = DFlashDraft(json.loads((path / "config.json").read_text()))
    d.load_weights(list(d.sanitize(mx.load(str(path / "model.safetensors"))).items()))
    if bits:
        nn.quantize(d, group_size=64, bits=bits)
    mx.eval(d.parameters())
    return d


def _target_step(target, ids, cache, layer_ids):
    cap = dict.fromkeys(layer_ids)
    h = target.model(mx.array([ids]), cache=cache, capture=cap)
    return target.lm_head(h)[0], [cap[i] for i in layer_ids]


def generate_greedy(target, prompt, max_tokens, eos=()):
    cache = target.make_cache()
    logits = target(mx.array([prompt]), cache=cache)[0, -1]
    out = []
    while True:
        t = int(mx.argmax(logits))
        out.append(t)
        if t in eos or len(out) >= max_tokens:
            return out
        logits = target(mx.array([[t]]), cache=cache)[0, -1]


def generate_dflash(target, draft, prompt, max_tokens, eos=(), n_draft=None, prefill_step=2048):
    """Greedy speculative decoding; output equals generate_greedy."""
    n_draft = n_draft or draft.block_size - 1
    ids_layers = draft.target_layer_ids
    tcache, dcache = make_trimmable_cache(target, slack=n_draft + 1), draft.make_cache()
    for a in range(0, len(prompt), prefill_step):
        logits, aux = _target_step(target, prompt[a : a + prefill_step], tcache, ids_layers)
        draft.add_context(draft.context_states(aux), dcache)
    token = int(mx.argmax(logits[-1]))
    out, steps, accepted = [], 0, 0
    while True:
        drafts = draft.propose(target, token, dcache, n_draft)
        logits, aux = _target_step(target, [token] + drafts, tcache, ids_layers)
        g = mx.argmax(logits, axis=-1).tolist()  # g[j] = greedy token at position(token) + j + 1
        k = 0
        while k < n_draft and drafts[k] == g[k]:
            k += 1
        steps, accepted = steps + 1, accepted + k
        for t in [token] + drafts[:k]:
            out.append(t)
            if t in eos or len(out) >= max_tokens:
                return out, {"steps": steps, "drafted": steps * n_draft, "accepted": accepted,
                             "mean_accepted": accepted / steps}
        for c in tcache:
            c.trim(n_draft - k)
        draft.add_context(draft.context_states([a[:, : k + 1] for a in aux]), dcache)
        token = g[k]
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_dflash.py -q`
Expected: 5 passed. A lossless failure with the oracle but not the random draft means the trim/context bookkeeping is off by one: after a step, target and draft caches must both have `offset == position(token)`.

- [ ] **Step 5: Full suite, then commit**

```bash
python3 -m pytest -q
git add laguna_mlx/dflash.py tests/test_dflash.py
git commit -m "feat: DFlash draft model and lossless speculative decoding"
```

---

### Task 14: DFlash on real weights — acceptance and speed

**Files:**
- Create: `scripts/bench_dflash.py`
- Output: `results/dflash.json`

**Interfaces:**
- Consumes: `load_draft`, `generate_greedy`, `generate_dflash` (Task 13); `PROMPTS` from `scripts/check_parity.py` (Task 5); converted model (Task 4).

- [ ] **Step 1: Download the draft (0.9 GB)**

```bash
hf download poolside/Laguna-XS-2.1-DFlash --local-dir models/Laguna-XS-2.1-DFlash
```

- [ ] **Step 2: Write `scripts/bench_dflash.py`**

```python
"""Greedy baseline vs DFlash on 10 coding prompts x draft sizes x draft precisions.

python3 scripts/bench_dflash.py [--max-tokens 256]
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

import laguna_mlx  # noqa: F401
from laguna_mlx.dflash import generate_dflash, generate_greedy, load_draft
from mlx_lm import load
from scripts.check_parity import PROMPTS

p = argparse.ArgumentParser()
p.add_argument("--max-tokens", type=int, default=256)
a = p.parse_args()

mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
target, tok = load("models/Laguna-XS-2.1-mlx-4bit")
eos = tuple(tok.eos_token_ids)
prompts = [tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                   enable_thinking=False) for q in PROMPTS]


def timed(fn):
    t0 = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t0


generate_greedy(target, prompts[0], 8)  # warm-up
base = []
for pr in prompts:
    out, dt = timed(lambda: generate_greedy(target, pr, a.max_tokens, eos))
    base.append({"tokens": out, "tok_s": len(out) / dt})
base_tps = sum(b["tok_s"] for b in base) / len(base)
print("baseline tok/s", base_tps, flush=True)

runs = []
for bits in (None, 8, 4):
    draft = load_draft("models/Laguna-XS-2.1-DFlash", bits=bits)
    for n_draft in (3, 7, 15):
        rows = []
        for pr, b in zip(prompts, base):
            (out, st), dt = timed(lambda: generate_dflash(target, draft, pr, a.max_tokens, eos, n_draft))
            rows.append({"tok_s": len(out) / dt, "mean_accepted": st["mean_accepted"],
                         "identical": out == b["tokens"]})
        run = {"draft_bits": bits or 16, "n_draft": n_draft,
               "tok_s": sum(r["tok_s"] for r in rows) / len(rows),
               "mean_accepted": sum(r["mean_accepted"] for r in rows) / len(rows),
               "identical_frac": sum(r["identical"] for r in rows) / len(rows)}
        run["speedup"] = run["tok_s"] / base_tps
        runs.append(run)
        print(run, flush=True)
    del draft

Path("results/dflash.json").write_text(json.dumps(
    {"baseline_tok_s": base_tps, "max_tokens": a.max_tokens, "runs": runs}, indent=1))
```

- [ ] **Step 3: Sanity-check acceptance before the full sweep**

```bash
python3 - <<'EOF'
import laguna_mlx
from laguna_mlx.dflash import generate_dflash, load_draft
from mlx_lm import load
from scripts.check_parity import PROMPTS
t, tok = load("models/Laguna-XS-2.1-mlx-4bit")
d = load_draft("models/Laguna-XS-2.1-DFlash")
pr = tok.apply_chat_template([{"role": "user", "content": PROMPTS[0]}], add_generation_prompt=True, enable_thinking=False)
out, st = generate_dflash(t, d, pr, 128, tuple(tok.eos_token_ids))
print(st); print(tok.decode(out))
EOF
```
Expected: coherent code and `mean_accepted` ≥ 2. If it is below 0.5 the draft is mis-wired. Check in this order, re-running this step after each: (1) draft RoPE — try `{"rope_type": "default", "rope_theta": 500000.0, "partial_rotary_factor": 0.5}`; (2) captured layers — try `target_layer_ids` shifted by +1 (the config also lists `eagle_aux_hidden_state_layer_ids` `[2,14,26,34,40]`, a hidden-states-tuple index, i.e. output of layer id−1, which is what the code captures); (3) block causality — `propose` currently uses a causal mask within the block. Record what fixed it in the task report.

- [ ] **Step 4: Run the sweep**

Run: `python3 scripts/bench_dflash.py`
Expected: `results/dflash.json` with 9 runs. `identical_frac` should be ≥ 0.9 (bf16 batched verification can flip rare near-ties; below 0.9 means a bug). Report the best speedup and its setting in the task report.

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_dflash.py results/dflash.json && git commit -m "feat: DFlash benchmark on real weights"
```

---

### Task 15: Decode pipelining, compile, and prefill tuning

**Files:**
- Create: `scripts/bench_decode.py`
- Modify (only if the measurement says so): `laguna_mlx/laguna.py`, `replay/replay.py` (`chunk` default), `laguna_mlx/dflash.py` (`prefill_step` default)
- Output: `results/inference_opt.json`

- [ ] **Step 1: Write `scripts/bench_decode.py`**

```python
"""Decode and prefill measurements -> results/inference_opt.json

python3 scripts/bench_decode.py [--compile]
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import laguna_mlx  # noqa: F401
from mlx_lm import load, stream_generate

p = argparse.ArgumentParser()
p.add_argument("--compile", action="store_true", help="label this run as the compiled variant")
a = p.parse_args()
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
model, tok = load("models/Laguna-XS-2.1-mlx-4bit")
prompt = tok.apply_chat_template([{"role": "user", "content": "Write a JSON parser in Python."}],
                                 add_generation_prompt=True, enable_thinking=False)


def naive_decode(n=128):
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache=cache)[0, -1]
    t0 = time.perf_counter()
    for _ in range(n):
        logits = model(mx.array([[int(mx.argmax(logits))]]), cache=cache)[0, -1]
    return n / (time.perf_counter() - t0)


def pipelined_decode(n=128):
    last = None
    for last in stream_generate(model, tok, prompt, max_tokens=n):
        pass
    return last.generation_tps


def prefill(ctx, step):
    ids = np.random.default_rng(0).integers(1000, 50000, size=ctx).tolist()
    mx.clear_cache()
    mx.reset_peak_memory()
    last = None
    for last in stream_generate(model, tok, ids, max_tokens=1, prefill_step_size=step):
        pass
    return {"ctx": ctx, "step": step, "prompt_tps": last.prompt_tps, "peak_gb": mx.get_peak_memory() / 1e9}


naive_decode(8)  # warm-up
out = {"variant": "compiled" if a.compile else "baseline",
       "naive_decode_tok_s": naive_decode(), "pipelined_decode_tok_s": pipelined_decode(),
       "prefill": [prefill(32768, s) for s in (512, 1024, 2048, 4096)]}
print(json.dumps(out, indent=1))
path = Path("results/inference_opt.json")
runs = json.loads(path.read_text())["runs"] if path.exists() else []
path.write_text(json.dumps({"runs": [r for r in runs if r["variant"] != out["variant"]] + [out]}, indent=1))
```

- [ ] **Step 2: Baseline run**

Run: `python3 scripts/bench_decode.py`
Record: naive vs pipelined decode tok/s, and the prefill step with the highest `prompt_tps` whose `peak_gb` stays under the GPU limit.

- [ ] **Step 3: Compile candidates in `laguna_mlx/laguna.py`**

Add at module level and use them in place of the inline expressions:
```python
@mx.compile
def _gated(out, g, per_head):
    gate = nn.softplus(g.astype(mx.float32)).astype(out.dtype)
    if per_head:
        return out * gate[..., None]
    B, L, H, _ = out.shape
    return out * gate.reshape(B, L, H, -1)


@mx.compile
def _topk_weights(scores, inds, norm):
    w = mx.take_along_axis(scores, inds, axis=-1)
    return w / w.sum(axis=-1, keepdims=True) if norm else w
```
In `Attention.__call__`: `return self.o_proj(_gated(out, self.g_proj(x), self.per_head_gate).reshape(B, L, -1))`.
In `SparseMoeBlock.route`: `w = _topk_weights(scores, inds, self.norm_topk_prob)` then `return inds, w.astype(x.dtype)`.

Run: `python3 -m pytest -q` (all pass: compile must not change numerics beyond the existing tolerances), then `python3 scripts/bench_decode.py --compile`.

- [ ] **Step 4: Decide and record**

Keep the compile change only if `pipelined_decode_tok_s` improves ≥3% over baseline; otherwise `git checkout laguna_mlx/laguna.py` and note the negative result in the task report. Set the best prefill step as the default `chunk` in `replay/replay.py` (`assistant_logprobs`, `build_ref`, `score`) and `prefill_step` in `laguna_mlx/dflash.py`, and add `mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])` at the top of `replay.replay.main`.

- [ ] **Step 5: Commit**

```bash
python3 -m pytest -q
git add -A scripts/bench_decode.py results/inference_opt.json laguna_mlx replay
git commit -m "perf: decode/prefill measurements and tuned defaults"
```
