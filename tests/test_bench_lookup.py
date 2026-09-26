"""Unit tests for scripts/bench_lookup.py -- CPU/tiny model only, no real weights."""
import json

import numpy as np
import pytest

import scripts.bench_lookup as bl
from laguna_mlx.dflash import generate_greedy
from scripts.bench_lookup import greedy_fast, session_prefix
from tests.tiny import mlx_model, torch_model

PROMPT = list(np.random.default_rng(7).integers(0, 100352, size=17))


@pytest.fixture(scope="module")
def target():
    m = mlx_model(torch_model())
    m.lm_head.weight = m.lm_head.weight * 50  # peaky logits: no float near-ties in argmax
    return m


# ---------------------------------------------------------------- greedy_fast

@pytest.mark.parametrize("max_tokens,prefill_step", [(30, 2048), (30, 5), (1, 2048), (17, 4)])
def test_greedy_fast_matches_generate_greedy(target, max_tokens, prefill_step):
    want = generate_greedy(target, PROMPT, max_tokens)
    got = greedy_fast(target, PROMPT, max_tokens, prefill_step=prefill_step)
    assert got == want


def test_greedy_fast_respects_eos(target):
    free = generate_greedy(target, PROMPT, 30)
    eos = (free[5],)
    want = generate_greedy(target, PROMPT, 30, eos=eos)
    got = greedy_fast(target, PROMPT, 30, eos=eos)
    assert got == want and got[-1] == eos[0]


# ------------------------------------------------------------- session_prefix

def _mask(*spans, n):
    m = np.zeros(n, bool)
    for a, b in spans:
        m[a:b] = True
    return m


def test_session_prefix_ends_right_before_last_assistant_turn():
    ids = np.arange(30)
    mask = _mask((5, 8), (20, 25), n=30)  # assistant turns at [5,8) and [20,25)
    prefix = session_prefix(ids, mask, max_tokens=8192)
    assert list(prefix) == list(range(20))  # everything up to (not including) the last turn's start


def test_session_prefix_caps_at_max_tokens():
    ids = np.arange(100)
    mask = _mask((90, 95), n=100)
    prefix = session_prefix(ids, mask, max_tokens=10)
    assert len(prefix) == 10
    assert list(prefix) == list(range(80, 90))


def test_session_prefix_no_assistant_turn_raises():
    ids = np.arange(10)
    mask = np.zeros(10, bool)
    with pytest.raises(ValueError):
        session_prefix(ids, mask, max_tokens=8192)


# ------------------------------------------------------- JSON write/skip helpers

def test_atomic_write_and_load_roundtrip(tmp_path, monkeypatch):
    out = tmp_path / "lookup.json"
    monkeypatch.setattr(bl, "OUT", out)
    bl._atomic_write_json(out, {"a": 1})
    assert json.loads(out.read_text()) == {"a": 1}
    assert not out.with_name(out.name + ".tmp").exists()


def test_save_target_merges_and_target_complete(tmp_path, monkeypatch):
    out = tmp_path / "lookup.json"
    monkeypatch.setattr(bl, "OUT", out)
    assert bl._target_complete("resident") is False
    bl._save_target("resident", {"prompts": {"rows": [], "aggregate": []},
                                  "sessions": {"rows": [], "aggregate": []}, "stopped": None})
    assert bl._target_complete("resident") is True
    assert bl._target_complete("streaming") is False
    res = bl._load_results()
    assert res["resident"]["prompts"] == {"rows": [], "aggregate": []}
    assert "summary" in res and "notes" in res


def test_target_complete_false_when_stopped(tmp_path, monkeypatch):
    out = tmp_path / "lookup.json"
    monkeypatch.setattr(bl, "OUT", out)
    bl._save_target("resident", {"prompts": {"rows": [], "aggregate": []},
                                  "sessions": {"rows": [], "aggregate": []},
                                  "stopped": "swap 12.0 GB > 10.0 GB (loaded)"})
    assert bl._target_complete("resident") is False


def test_summarize_picks_best_setting_and_keep_threshold():
    res = {
        "resident": {"stopped": None, "sessions": {"aggregate": [
            {"g": 4, "n_draft": 7, "adaptive": False, "mean_speedup_decode": 1.05},
            {"g": 4, "n_draft": 15, "adaptive": True, "mean_speedup_decode": 1.4},
        ]}},
        "streaming": {"stopped": None, "sessions": {"aggregate": [
            {"g": 3, "n_draft": 7, "adaptive": False, "mean_speedup_decode": 1.2},
        ]}},
    }
    s = bl._summarize(res)
    assert s["best"]["target"] == "resident" and s["best"]["mean_speedup_decode"] == 1.4
    assert s["keep"] is True


def test_summarize_no_keep_when_below_threshold():
    res = {"resident": {"stopped": None, "sessions": {"aggregate": [
        {"g": 4, "n_draft": 7, "adaptive": False, "mean_speedup_decode": 1.02},
    ]}}}
    s = bl._summarize(res)
    assert s["keep"] is False


def test_summarize_handles_missing_sessions():
    res = {"resident": {"stopped": "swap high"}}
    s = bl._summarize(res)
    assert s == {"by_target_setting": [], "best": None, "keep": False}
