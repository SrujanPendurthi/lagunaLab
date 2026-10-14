"""Sliding-window KV cache that can be trimmed, for speculative decoding."""
import mlx.core as mx
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache, _BaseCache


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


class DequantKVCache(_BaseCache):
    """Stores K/V quantized (mlx-lm's QuantizedKVCache) and hands attention bf16 K/V, one layer
    at a time, so mx.fast's fused SDPA runs. mlx-lm's quantized SDPA would materialize the full
    [heads, L, context] score matrix: ~16 GB per layer at 82k context, which hung the machine.

    No `bits` attribute on purpose: mlx-lm routes any cache with one to its quantized SDPA.
    """

    def __init__(self, group_size=64, bits=8):
        self.q = QuantizedKVCache(group_size=group_size, bits=bits)

    def update_and_fetch(self, keys, values):
        # ponytail: decode re-dequantizes the whole layer every token (~0.3 GB per layer at
        # 82k); keep a bf16 copy of the prefix or use quantized SDPA for L == 1 if decode is slow
        qk, qv = self.q.update_and_fetch(keys, values)
        g, b = self.q.group_size, self.q.bits
        return mx.dequantize(*qk, group_size=g, bits=b), mx.dequantize(*qv, group_size=g, bits=b)

    @property
    def offset(self):
        return self.q.offset

    @property
    def kv_bits(self):
        return self.q.bits

    @property
    def group_size(self):
        return self.q.group_size

    def make_mask(self, *args, **kwargs):
        return self.q.make_mask(*args, **kwargs)

    @property
    def state(self):
        return self.q.state

    def is_trimmable(self):
        return True

    def trim(self, n):
        return self.q.trim(n)

    def empty(self):
        return self.q.empty()

    @property
    def nbytes(self):
        return self.q.nbytes


def make_quantized_cache(model, bits=8, group_size=64):
    """Quantized KV storage on full-attention layers; sliding layers keep their window."""
    a = model.args
    if a.head_dim % group_size:
        raise ValueError(f"head_dim {a.head_dim} is not a multiple of group_size {group_size}")
    return [
        RotatingKVCache(max_size=a.sliding_window)
        if lt == "sliding_attention" else DequantKVCache(group_size=group_size, bits=bits)
        for lt in a.layer_types
    ]
