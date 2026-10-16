import mlx.core as mx
import pytest
from mlx_lm.models.cache import QuantizedKVCache

from laguna_mlx.cache import make_quantized_cache
from laguna_mlx.q8_attn import BLOCK, q8_decode_attention
from tests.tiny import mlx_only_model


def _filled(n, hkv=8, d=128, seed=0):
    mx.random.seed(seed)
    c = QuantizedKVCache(group_size=64, bits=8)
    k = mx.random.normal((1, hkv, n, d)).astype(mx.bfloat16)
    v = mx.random.normal((1, hkv, n, d)).astype(mx.bfloat16)
    c.update_and_fetch(k, v)
    return c


@pytest.mark.parametrize("n", [1, 7, BLOCK, BLOCK + 1, 3 * BLOCK + 100])
def test_matches_dequantize_then_sdpa(n):
    c = _filled(n)
    q = mx.random.normal((1, 48, 1, 128)).astype(mx.bfloat16)
    scale = 128 ** -0.5
    # the cache's buffers are longer than n (256-row steps): the kernel must stop at n
    assert c.keys[0].shape[2] >= n
    got = q8_decode_attention(q, c.keys, c.values, n, scale)
    k, v = (mx.dequantize(*t, group_size=64, bits=8) for t in c.state)
    want = mx.fast.scaled_dot_product_attention(q.astype(mx.float32), k.astype(mx.float32),
                                                 v.astype(mx.float32), scale=scale)
    assert got.shape == q.shape and got.dtype == q.dtype
    assert mx.abs(got.astype(mx.float32) - want).max().item() < 2e-2  # bf16 output rounding


def test_model_decode_matches_dequant_cache():
    # fused decode inside the model: same logits as the dequantize path, prefill unchanged
    m = mlx_only_model(head_dim=128)
    ids = mx.array([[5, 9, 2, 7, 1, 3, 8, 4] * 40])
    out = []
    for fused in (False, True):
        cache = make_quantized_cache(m, 8, fused_decode=fused)
        m(ids, cache=cache)
        tok = mx.array([[6]])
        logits = [m(tok, cache=cache) for _ in range(3)]
        out.append(mx.concatenate(logits, axis=1).astype(mx.float32))
    assert mx.abs(out[0] - out[1]).max().item() < 5e-2
    assert (mx.argmax(out[0], -1) == mx.argmax(out[1], -1)).all().item()


def test_fused_decode_needs_8bit():
    m = mlx_only_model(head_dim=128)
    with pytest.raises(ValueError, match="8-bit"):
        make_quantized_cache(m, 4, fused_decode=True)
