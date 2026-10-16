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

    `quantized_decode=True` (off by default): single-token steps skip the dequantize and return
    the quantized K/V tuples, and `bits` exists for that step only, so mlx-lm's quantized SDPA
    reads the packed cache directly. At L == 1 its score matrix is one row per head, so the
    materialization problem above doesn't apply. Without it, decode dequantizes the whole layer
    every token (~0.34 GB written per full-attention layer at 82k).
    """

    def __init__(self, group_size=64, bits=8, quantized_decode=False, fused_decode=False):
        if fused_decode and (bits, group_size) != (8, 64):
            raise ValueError("fused_decode needs 8-bit, group_size 64")
        self.q = QuantizedKVCache(group_size=group_size, bits=bits)
        self.quantized_decode = quantized_decode
        self.fused_decode = fused_decode
        self._quantized_step = False
        self.fused_step = False

    def update_and_fetch(self, keys, values):
        qk, qv = self.q.update_and_fetch(keys, values)
        # Attention checks `fused_step` and then calls attend(); the K/V returned are unused
        self.fused_step = self.fused_decode and keys.shape[2] == 1
        if self.fused_step:
            return qk, qv
        # the attention call reads `bits` right after this, on the same layer's cache
        self._quantized_step = self.quantized_decode and keys.shape[2] == 1
        if self._quantized_step:
            return qk, qv
        g, b = self.q.group_size, self.q.bits
        return mx.dequantize(*qk, group_size=g, bits=b), mx.dequantize(*qv, group_size=g, bits=b)

    def attend(self, queries, scale):
        """Single-token attention straight off the packed cache (laguna_mlx.q8_attn)."""
        from laguna_mlx.q8_attn import q8_decode_attention

        return q8_decode_attention(queries, self.q.keys, self.q.values, self.q.offset, scale)

    @property
    def bits(self):
        if not self._quantized_step:
            raise AttributeError("bits")  # hasattr(cache, "bits") is False: fused bf16 SDPA
        return self.q.bits

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


def make_quantized_cache(model, bits=8, group_size=64, quantized_decode=False, fused_decode=False):
    """Quantized KV storage on full-attention layers; sliding layers keep their window."""
    a = model.args
    if a.head_dim % group_size:
        raise ValueError(f"head_dim {a.head_dim} is not a multiple of group_size {group_size}")
    return [
        RotatingKVCache(max_size=a.sliding_window)
        if lt == "sliding_attention" else DequantKVCache(group_size=group_size, bits=bits, quantized_decode=quantized_decode,
                                                       fused_decode=fused_decode)
        for lt in a.layer_types
    ]
