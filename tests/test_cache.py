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
