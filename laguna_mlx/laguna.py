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
