import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import RotatingKVCache

from laguna_mlx.cache import DequantKVCache, make_quantized_cache
from tests.tiny import TINY, mlx_model, mlx_only_model, torch_model

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


@pytest.fixture(scope="module")
def model64():
    return mlx_only_model(head_dim=64)  # mx.quantize needs a group size of 32, 64 or 128


def _ids(n):
    return mx.array(np.random.default_rng(1).integers(0, TINY["vocab_size"], size=(1, n)))


def test_quantized_cache_layout_follows_layer_types(model64):
    cache = make_quantized_cache(model64, bits=4)
    for c, lt in zip(cache, model64.args.layer_types):
        if lt == "full_attention":
            assert isinstance(c, DequantKVCache) and c.kv_bits == 4 and c.group_size == 64
            assert not hasattr(c, "bits")  # else mlx-lm routes it to its quantized SDPA
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


def test_quantized_cache_prefill_never_materializes_scores(model64):
    # mlx-lm's quantized attention builds the full [heads, L, kL] score matrix: 4 x 4096 x 4096
    # x 4 B = 268 MB here, ~16 GB per layer on the real model at 82k context (it hung the Mac).
    ids = _ids(4096)

    def prefill_peak(cache):
        mx.eval(ids)
        mx.reset_peak_memory()
        base = mx.get_active_memory()
        mx.eval(model64.model(ids, cache=cache))
        return mx.get_peak_memory() - base

    plain = prefill_peak(model64.make_cache())
    quant = prefill_peak(make_quantized_cache(model64, 8))
    assert quant < plain + 100e6, (plain, quant)
