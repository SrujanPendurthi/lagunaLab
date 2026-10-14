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
