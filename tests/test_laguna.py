import copy

import mlx.core as mx
import numpy as np
import pytest
import torch

from laguna_mlx import laguna
from tests.tiny import TINY, mlx_model, to_disk_names, torch_model


def test_matches_hf_reference_past_sliding_window():
    t = torch_model()
    m = mlx_model(t)
    ids = np.random.default_rng(0).integers(0, TINY["vocab_size"], size=(1, 24))
    with torch.no_grad():
        ref = t(torch.tensor(ids), use_cache=False).logits.numpy()
    out = np.array(m(mx.array(ids)))
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-3)


def test_sanitize_is_idempotent_on_converted_weights():
    t = torch_model()
    m = laguna.Model(laguna.ModelArgs.from_dict(copy.deepcopy(TINY)))
    once = m.sanitize({k: mx.array(v) for k, v in to_disk_names(t).items()})
    twice = m.sanitize(dict(once))
    assert sorted(once) == sorted(twice)


@pytest.mark.parametrize("bad", [
    {"moe_router_logit_softcapping": 30.0},
    {"swa_attention_sink_enabled": True},
    {"moe_apply_router_weight_on_input": True},
    {"gating": False},
])
def test_unsupported_config_raises(bad):
    with pytest.raises(ValueError):
        laguna.ModelArgs.from_dict({**copy.deepcopy(TINY), **bad})


def test_registered_with_mlx_lm():
    import laguna_mlx  # noqa: F401
    from mlx_lm.utils import _get_classes

    assert _get_classes({"model_type": "laguna"})[0] is laguna.Model
