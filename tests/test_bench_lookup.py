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
    got, prefill_s = greedy_fast(target, PROMPT, max_tokens, prefill_step=prefill_step)
    assert got == want
    assert prefill_s > 0


def test_greedy_fast_respects_eos(target):
    free = generate_greedy(target, PROMPT, 30)
    eos = (free[5],)
    want = generate_greedy(target, PROMPT, 30, eos=eos)
    got, _prefill_s = greedy_fast(target, PROMPT, 30, eos=eos)
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
    settings = bl.SETTINGS["resident"]
    bl._save_target("resident", {"prompts": {"rows": [], "aggregate": [], "settings": settings},
                                  "sessions": {"rows": [], "aggregate": [], "settings": settings},
                                  "stopped": None})
    assert bl._target_complete("resident") is True
    assert bl._target_complete("streaming") is False
    res = bl._load_results()
    assert res["resident"]["prompts"] == {"rows": [], "aggregate": [], "settings": settings}
    assert "summary" in res and "notes" in res


def test_target_complete_false_when_stopped(tmp_path, monkeypatch):
    out = tmp_path / "lookup.json"
    monkeypatch.setattr(bl, "OUT", out)
    settings = bl.SETTINGS["resident"]
    bl._save_target("resident", {"prompts": {"rows": [], "aggregate": [], "settings": settings},
                                  "sessions": {"rows": [], "aggregate": [], "settings": settings},
                                  "stopped": "swap 12.0 GB > 10.0 GB (loaded)"})
    assert bl._target_complete("resident") is False


def test_target_complete_false_when_settings_dont_match_current(tmp_path, monkeypatch):
    """A settings change (e.g. dropping streaming's n=7/15 settings for the verify-cost-cliff
    follow-up) must force a redo, not let --skip-existing treat stale-settings data as done."""
    out = tmp_path / "lookup.json"
    monkeypatch.setattr(bl, "OUT", out)
    old_settings = [{"g": 4, "n_draft": 7, "adaptive": False}]  # stale: not bl.SETTINGS["streaming"]
    assert old_settings != bl.SETTINGS["streaming"]  # sanity: genuinely different from current
    bl._save_target("streaming", {"prompts": {"rows": [], "aggregate": [], "settings": old_settings},
                                   "sessions": {"rows": [], "aggregate": [], "settings": old_settings},
                                   "stopped": None})
    assert bl._target_complete("streaming") is False


# --------------------------------------------------------- SETTINGS per target

def test_settings_is_per_target_and_streaming_stays_below_the_cliff():
    assert set(bl.SETTINGS) == {"resident", "streaming"}
    resident_n = {s["n_draft"] for s in bl.SETTINGS["resident"]}
    streaming_n = {s["n_draft"] for s in bl.SETTINGS["streaming"]}
    assert resident_n == {7, 15}  # unchanged: g=4/n=7 fixed, g=4/n=15 adaptive, g=3/n=7 fixed
    # streaming: every setting's verify-block length (1 + n_draft) stays <= 4 tokens, i.e.
    # n_draft <= 3, below the measured verify-cost cliff (results/dflash_diag.json: 61.9ms at
    # block=4 vs 212.0ms at block=6).
    assert streaming_n == {3, 2}
    assert all(s["n_draft"] <= 3 for s in bl.SETTINGS["streaming"])
    assert bl.SETTINGS["resident"] != bl.SETTINGS["streaming"]


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


# ------------------------------------------------------- _run_item arithmetic

def _fake_perf_counter(seq):
    it = iter(seq)
    return lambda: next(it)


def test_run_item_timing_and_speedup_arithmetic(monkeypatch):
    # 8 perf_counter() calls: (t0,t1) for the baseline, then (t2,t3) per of the 3 settings.
    monkeypatch.setattr(bl.time, "perf_counter",
                         _fake_perf_counter([0.0, 12.0, 100.0, 108.0, 200.0, 210.0, 300.0, 306.0]))
    # baseline: 5 tokens, prefill_s=2.0, total=12-0=12 -> decode_s=10, decode_tok_s=0.5, e2e=5/12
    monkeypatch.setattr(bl, "greedy_fast", lambda target, ids, max_tokens, eos: (list(range(5)), 2.0))

    lookup_returns = iter([
        (list(range(8)), {"mean_accepted": 1.0, "steps": 4, "drafted": 12, "accepted": 4,
                           "steps_with_draft": 3, "prefill_s": 3.0}),   # total=8-0=8 -> decode_s=5
        (list(range(10)), {"mean_accepted": 2.0, "steps": 5, "drafted": 20, "accepted": 8,
                            "steps_with_draft": 4, "prefill_s": 4.0}),  # total=10 -> decode_s=6
        (list(range(5)), {"mean_accepted": 0.5, "steps": 5, "drafted": 10, "accepted": 2,
                           "steps_with_draft": 2, "prefill_s": 5.0}),   # total=6 -> decode_s=1
    ])
    monkeypatch.setattr(bl, "generate_lookup",
                         lambda target, ids, max_tokens, eos, g, n_draft, adaptive, prefill_step:
                             next(lookup_returns))

    settings = bl.SETTINGS["resident"]  # 3 settings, matching the 3 fake lookup_returns above
    base, rows = bl._run_item(target=None, ids=[1, 2, 3], max_tokens=999, eos=(), settings=settings)

    assert base["tokens"] == 5 and base["prefill_s"] == 2.0
    assert base["total_s"] == pytest.approx(12.0) and base["decode_s"] == pytest.approx(10.0)
    assert base["decode_tok_s"] == pytest.approx(0.5) and base["e2e_tok_s"] == pytest.approx(5 / 12)

    r0, r1, r2 = rows
    assert r0["total_s"] == pytest.approx(8.0) and r0["prefill_s"] == 3.0
    assert r0["decode_s"] == pytest.approx(5.0) and r0["decode_tok_s"] == pytest.approx(8 / 5)
    assert r0["speedup_decode"] == pytest.approx((8 / 5) / 0.5)
    assert r0["e2e_tok_s"] == pytest.approx(8 / 8)
    assert r0["speedup_e2e"] == pytest.approx((8 / 8) / (5 / 12))
    assert r0["identical"] is False  # base_out has 5 tokens, r0's out has 8

    assert r1["decode_s"] == pytest.approx(6.0) and r1["decode_tok_s"] == pytest.approx(10 / 6)

    assert r2["decode_s"] == pytest.approx(1.0) and r2["decode_tok_s"] == pytest.approx(5.0)
    assert r2["identical"] is True  # both base_out and r2's out are range(5)
    assert r2["first_divergence"] is None


def test_run_item_floors_decode_when_prefill_exceeds_total(monkeypatch):
    settings = [{"g": 4, "n_draft": 7, "adaptive": False}]  # a single setting is enough here
    monkeypatch.setattr(bl.time, "perf_counter", _fake_perf_counter([0.0, 4.0, 10.0, 10.5]))
    monkeypatch.setattr(bl, "greedy_fast", lambda target, ids, max_tokens, eos: (list(range(3)), 1.0))
    monkeypatch.setattr(bl, "generate_lookup", lambda *a, **k: (
        list(range(3)),
        {"mean_accepted": 0.0, "steps": 1, "drafted": 0, "accepted": 0, "steps_with_draft": 0,
         "prefill_s": 3.0},  # total = 10.5 - 10 = 0.5 < prefill_s=3.0 -> would-be-negative decode_s
    ))
    base, rows = bl._run_item(target=None, ids=[1], max_tokens=10, eos=(), settings=settings)
    assert rows[0]["decode_s"] == pytest.approx(1e-9)
    assert rows[0]["decode_tok_s"] == pytest.approx(3 / 1e-9)


# ------------------------------------------------------- _run_workload resume

def _fake_setting_row(**over):
    return {"g": 4, "n_draft": 3, "adaptive": False, "speedup_decode": 1.0, "speedup_e2e": 1.0,
            "mean_accepted": 1.0, "identical": True, **over}


def test_run_workload_resumes_and_skips_already_done_items(tmp_path, monkeypatch):
    out = tmp_path / "lookup.json"
    monkeypatch.setattr(bl, "OUT", out)
    monkeypatch.setattr(bl, "swap_gb", lambda: 0.0)
    settings = bl.SETTINGS["streaming"]
    seen = []

    def fake_run_item(target, ids, max_tokens, eos, settings):
        seen.append(ids)
        return {"tokens": len(ids)}, [_fake_setting_row()]

    monkeypatch.setattr(bl, "_run_item", fake_run_item)
    result = {"target": "streaming", "prompts": {
        "rows": [{"name": "p0", "prompt_tokens": 3, "baseline": {}, "settings": [_fake_setting_row()]}],
        "aggregate": [], "settings": settings}}
    items = [{"name": "p0", "ids": [1, 2, 3]}, {"name": "p1", "ids": [4, 5, 6]}]
    swap = [("start", 0.0)]
    workload = bl._run_workload("streaming", result, target=None, items=items, max_tokens=10,
                                 eos=(), swap=swap, over=lambda stage: False, label="prompts",
                                 settings=settings)
    assert seen == [[4, 5, 6]]  # p0 already done under the *same* settings, so only p1 reruns
    assert [r["name"] for r in workload["rows"]] == ["p0", "p1"]
    assert workload["settings"] == settings
    # the checkpoint written after p1 must be on disk (atomic save happened inside the loop)
    on_disk = bl._load_results()["streaming"]["prompts"]
    assert [r["name"] for r in on_disk["rows"]] == ["p0", "p1"] and on_disk["settings"] == settings


def test_run_workload_recomputes_when_settings_changed(tmp_path, monkeypatch):
    """The resume key includes the settings list itself (see _existing_rows): a settings
    change (this streaming-cliff follow-up dropped n=7/15 for streaming) must not resume p0's
    row under the new settings as if it had been measured with them."""
    out = tmp_path / "lookup.json"
    monkeypatch.setattr(bl, "OUT", out)
    monkeypatch.setattr(bl, "swap_gb", lambda: 0.0)
    old_settings = [{"g": 4, "n_draft": 7, "adaptive": False}]  # stale
    new_settings = bl.SETTINGS["streaming"]
    seen = []

    def fake_run_item(target, ids, max_tokens, eos, settings):
        seen.append(ids)
        assert settings == new_settings  # _run_item always gets the *current* settings
        return {"tokens": len(ids)}, [_fake_setting_row()]

    monkeypatch.setattr(bl, "_run_item", fake_run_item)
    result = {"target": "streaming", "prompts": {
        "rows": [{"name": "p0", "prompt_tokens": 3, "baseline": {}, "settings": [_fake_setting_row()]}],
        "aggregate": [], "settings": old_settings}}
    items = [{"name": "p0", "ids": [1, 2, 3]}]
    swap = [("start", 0.0)]
    workload = bl._run_workload("streaming", result, target=None, items=items, max_tokens=10,
                                 eos=(), swap=swap, over=lambda stage: False, label="prompts",
                                 settings=new_settings)
    assert seen == [[1, 2, 3]]  # p0 recomputed: old row's settings don't match new_settings
    assert workload["settings"] == new_settings
