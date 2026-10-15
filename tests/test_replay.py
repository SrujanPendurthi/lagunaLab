import json

import mlx.core as mx
import numpy as np
import pytest
from transformers import AutoTokenizer

from laguna_mlx.cache import make_quantized_cache
from replay.replay import build_ref, score, score_actions
from tests.tiny import mlx_model, mlx_only_model, torch_model

MESSAGES = [
    {"role": "user", "content": "list files"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}]},
    {"role": "tool", "content": "a.py\nb.py"},
    {"role": "assistant", "content": "There are two files, a.py and b.py."},
]


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("models/Laguna-XS-2.1")


@pytest.fixture
def session(tmp_path):
    f = tmp_path / "s.jsonl"
    f.write_text(json.dumps({"path": "/v1/chat/completions", "request": {"messages": MESSAGES},
                             "status": 200, "response": ""}))
    return f


def test_self_score_is_perfect_and_chunking_invariant(tok, session, tmp_path):
    m = mlx_model(torch_model())
    ref = tmp_path / "s.npz"
    build_ref(m, tok, session, ref, n_actions=1, max_new=8)
    for chunk in (2048, 7):
        s = score(m, ref, chunk=chunk)
        assert abs(s["kl_top64_mean"]) < 1e-5 and s["top1_agree"] == 1.0
        assert s["assistant_tokens"] > 0 and not s["truncated"]
    assert score_actions(m, tok, ref, max_new=8) == 1.0


def test_perturbed_model_scores_worse(tok, session, tmp_path):
    ref = tmp_path / "s.npz"
    build_ref(mlx_model(torch_model()), tok, session, ref, n_actions=0)
    s = score(mlx_model(torch_model(seed=1)), ref)
    # tiny random models are near-uniform, so truncated KL is small; argmax flips are not
    assert s["top1_agree"] < 0.5 and s["kl_top64_mean"] != 0


def test_long_session_is_truncated_not_fatal(tok, session, tmp_path):
    ref = tmp_path / "s.npz"
    build_ref(mlx_model(torch_model()), tok, session, ref, max_tokens=50, n_actions=0)
    s = score(mlx_model(torch_model()), ref)
    assert s["truncated"] and s["tokens"] == 50


def test_score_with_cache_factory(tok, session, tmp_path):
    m = mlx_only_model(head_dim=64)
    ref = tmp_path / "s.npz"
    build_ref(m, tok, session, ref, n_actions=0)
    default = score(m, ref)
    explicit = score(m, ref, make_cache=m.make_cache)
    assert explicit["kl_top64_mean"] == default["kl_top64_mean"] and explicit["top1_agree"] == 1.0
    q4 = score(m, ref, make_cache=lambda: make_quantized_cache(m, 4))
    assert q4["kl_top64_mean"] > abs(default["kl_top64_mean"]) + 1e-6  # the factory is really used


def test_greedy_actions_is_public_and_keeps_old_name():
    from replay import replay
    assert replay.greedy_actions is replay._greedy_actions


def test_score_with_forward_callable(tok, session, tmp_path):
    m = mlx_model(torch_model())
    ref = tmp_path / "s.npz"
    build_ref(m, tok, session, ref, n_actions=0)
    default = score(m, ref)
    assert default["kl_top64_mean"] == 0.0 and default["top1_agree"] == 1.0  # default path unchanged
    calls = []

    def fwd(ids, cache):  # what a non-mlx-lm engine supplies: [1, L] ids -> [1, L, V] logits
        calls.append(ids.shape[1])
        return m.lm_head(m.model(ids, cache=cache))

    s = score(None, ref, forward=fwd, make_cache=m.make_cache)
    assert calls and s["top1_agree"] == 1.0 and abs(s["kl_top64_mean"]) < 1e-5


def test_ref_records_its_model_and_old_refs_still_score(tok, session, tmp_path):
    m = mlx_model(torch_model())
    ref = tmp_path / "s.npz"
    build_ref(m, tok, session, ref, n_actions=0, model_id="models/x")
    assert str(np.load(ref)["model"]) == "models/x" and score(m, ref)["ref_model"] == "models/x"
    old = tmp_path / "old.npz"  # written before the field existed
    np.savez_compressed(old, **{k: v for k, v in np.load(ref).items() if k != "model"})
    s = score(m, old)
    assert s["ref_model"] is None and s["top1_agree"] == 1.0
