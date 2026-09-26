import mlx.core as mx
import numpy as np
import pytest

from laguna_mlx.hooks import RoutingRecorder
from laguna_mlx.pruning import ExpertSaliencyRecorder, accumulate, apply_mask, keep_sets, load_stats, scores
from tests.tiny import TINY, mlx_model, torch_model

E, K = TINY["num_experts"], TINY["num_experts_per_tok"]


def _ids(n=12, seed=11):
    return mx.array(np.random.default_rng(seed).integers(0, TINY["vocab_size"], size=(1, n)))


def _empty():
    return {"count": np.zeros(E, np.int64), "gate": np.zeros(E), "reap": np.zeros(E)}


def test_accumulate_matches_hand_computation():
    s = _empty()
    inds = np.array([[0, 3], [3, 5]])
    w = np.array([[0.6, 0.4], [0.7, 0.3]])
    norms = np.array([[2.0, 1.0], [3.0, 10.0]])
    accumulate(s, inds, w, norms)
    assert s["count"].tolist() == [1, 0, 0, 2, 0, 1, 0, 0]
    np.testing.assert_allclose(s["gate"][[0, 3, 5]], [0.6, 1.1, 0.3])
    np.testing.assert_allclose(s["reap"][[0, 3, 5]], [1.2, 0.4 + 2.1, 3.0])
    accumulate(s, inds, w, norms, mask=np.array([False, True]))
    assert s["count"][3] == 3 and s["count"][0] == 1


def test_recorder_is_transparent_restores_and_matches_routing_counts():
    m = mlx_model(torch_model())
    ids = _ids()
    before = np.array(m(ids))
    with RoutingRecorder(m) as rr:
        m(ids)
    with ExpertSaliencyRecorder(m) as rec:
        during = np.array(m(ids))
    np.testing.assert_array_equal(before, during)
    np.testing.assert_array_equal(before, np.array(m(ids)))
    for i in rr.counts:
        np.testing.assert_array_equal(rec.stats[i]["count"], rr.counts[i])
        np.testing.assert_allclose(rec.stats[i]["gate"], rr.mass[i], rtol=1e-5)
        assert (rec.stats[i]["reap"] >= 0).all() and rec.stats[i]["reap"].sum() > 0
    assert all("route" not in vars(l.mlp) for l in m.layers)
    assert all(type(l.mlp.switch_mlp).__name__ == "SwitchGLU" for l in m.layers if hasattr(l.mlp, "route"))


def test_recorder_restores_model_when_forward_raises():
    m = mlx_model(torch_model())
    with pytest.raises(ValueError):
        with ExpertSaliencyRecorder(m):
            raise ValueError("boom")
    assert all(type(l.mlp.switch_mlp).__name__ == "SwitchGLU" for l in m.layers if hasattr(l.mlp, "route"))


def test_save_load_and_scores(tmp_path):
    m = mlx_model(torch_model())
    with ExpertSaliencyRecorder(m) as rec:
        m(_ids(20))
    rec.save(tmp_path / "s.npz")
    st = load_stats(tmp_path / "s.npz")
    assert sorted(st) == sorted(rec.stats)
    r = scores(st, "reap")
    i = sorted(st)[0]
    c = st[i]["count"]
    np.testing.assert_allclose(r[i][c > 0], st[i]["reap"][c > 0] / c[c > 0])
    assert (r[i][c == 0] == 0).all()
    np.testing.assert_array_equal(scores(st, "random", seed=3)[i], scores(st, "random", seed=3)[i])
    with pytest.raises(ValueError):
        scores(st, "nope")


def _fake_sal(layers=(1, 2, 3), E=8, seed=0):
    rng = np.random.default_rng(seed)
    return {i: rng.random(E) for i in layers}


def test_keep_sets_uniform_counts_and_order():
    sal = _fake_sal()
    keep = keep_sets(sal, 0.25)
    for i, v in sal.items():
        assert len(keep[i]) == 6 and list(keep[i]) == sorted(keep[i])
        assert set(keep[i]) == set(np.argsort(-v)[:6])


def test_keep_sets_global_budget_and_minimum():
    sal = _fake_sal(E=8)
    keep = keep_sets(sal, 0.5, allocation="global", min_keep=3)
    assert sum(len(v) for v in keep.values()) == 12
    assert all(len(v) >= 3 for v in keep.values())
    # global allocation normalizes per layer, so a layer's raw scale must not matter
    scaled = keep_sets({**sal, 1: sal[1] * 100}, 0.5, allocation="global", min_keep=3)
    assert {i: v.tolist() for i, v in scaled.items()} == {i: v.tolist() for i, v in keep.items()}
    with pytest.raises(ValueError):
        keep_sets(sal, 0.9, allocation="global", min_keep=3)


def test_mask_routes_only_kept_experts_and_undoes():
    m = mlx_model(torch_model())
    ids = _ids(16)
    before = np.array(m(ids))
    keep = {1: np.array([0, 2, 5]), 2: np.array([1, 3, 4, 7]), 3: np.array([0, 1])}
    undo = apply_mask(m, keep)
    with RoutingRecorder(m) as rr:
        m(ids)
    for i, k in keep.items():
        assert set(np.nonzero(rr.counts[i])[0]) <= set(k.tolist())
        assert rr.counts[i].sum() == 16 * K
    undo()
    np.testing.assert_array_equal(np.array(m(ids)), before)


def test_masking_unused_experts_is_exact():
    m = mlx_model(torch_model())
    ids = _ids(16)
    with RoutingRecorder(m) as rr:
        before = np.array(m(ids))
    keep = {i: np.nonzero(c)[0] for i, c in rr.counts.items()}
    keep = {i: k if len(k) >= K else np.arange(K) for i, k in keep.items()}
    undo = apply_mask(m, keep)
    np.testing.assert_array_equal(np.array(m(ids)), before)
    undo()


def test_mask_smaller_than_top_k_raises():
    m = mlx_model(torch_model())
    with pytest.raises(ValueError):
        apply_mask(m, {1: np.array([0])})
