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
