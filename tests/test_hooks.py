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
