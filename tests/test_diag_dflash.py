"""Unit tests for scripts/diag_dflash.py's pure helpers -- no model loading."""
import json
import sys

import mlx.core as mx
import numpy as np
import pytest

import scripts.diag_dflash as dd
from scripts.diag_dflash import bf16_ulp_gap, first_divergence, linear_fit, median_iqr


def _next_bf16_above(x):
    bits = int(np.array(mx.array([x], dtype=mx.bfloat16).view(mx.uint16))[0])
    nxt = np.array([np.uint16(bits + 1)])
    return float(mx.array(nxt).view(mx.bfloat16).astype(mx.float32)[0])


# ---------------------------------------------------------------- first_divergence

def test_first_divergence_identical():
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None


def test_first_divergence_at_index():
    assert first_divergence([1, 2, 3, 4], [1, 2, 9, 4]) == 2


def test_first_divergence_at_start():
    assert first_divergence([9, 2], [1, 2]) == 0


def test_first_divergence_prefix_of_other():
    # agree everywhere they overlap, but one is shorter: diverges where it ends
    assert first_divergence([1, 2, 3], [1, 2, 3, 4]) == 3
    assert first_divergence([1, 2, 3, 4], [1, 2, 3]) == 3


def test_first_divergence_empty():
    assert first_divergence([], []) is None
    assert first_divergence([], [1]) == 0


# ---------------------------------------------------------------- bf16_ulp_gap

def test_bf16_ulp_gap_self_is_zero():
    for x in (0.0, 1.0, -3.5, 1e30, -1e-30):
        assert bf16_ulp_gap(x, x) == 0


def test_bf16_ulp_gap_adjacent_bf16_values_is_one():
    assert bf16_ulp_gap(1.0, _next_bf16_above(1.0)) == 1
    assert bf16_ulp_gap(2.0, _next_bf16_above(2.0)) == 1
    assert bf16_ulp_gap(-8.0, _next_bf16_above(-8.0)) == 1


def test_bf16_ulp_gap_symmetric():
    assert bf16_ulp_gap(1.0, 2.0) == bf16_ulp_gap(2.0, 1.0)


def test_bf16_ulp_gap_across_zero_is_small():
    # 0.0 and -0.0 are adjacent bf16 representations: the gap should be tiny,
    # not the huge integer distance a naive signed-bit-pattern diff would give.
    assert bf16_ulp_gap(0.0, -0.0) <= 2


def test_bf16_ulp_gap_monotonic_with_value():
    # gaps should grow as values move further apart in bf16 space
    near = bf16_ulp_gap(1.0, _next_bf16_above(1.0))
    far = bf16_ulp_gap(1.0, 2.0)
    assert far > near


def test_bf16_ulp_gap_ordering_matches_value_ordering():
    vals = sorted([-2.0, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 2.0, 100.0, -100.0, 3.4e38, -3.4e38, 1.2e-38])
    # gap(vals[0], vals[i]) must be non-decreasing as i increases, since the
    # bf16-order keys are monotonic in value
    gaps = [bf16_ulp_gap(vals[0], v) for v in vals]
    assert gaps == sorted(gaps)


# ---------------------------------------------------------------- linear_fit

def test_linear_fit_exact_line():
    xs = [0, 1, 2, 3, 4, 5]
    ys = [2 + 3 * x for x in xs]
    fit = linear_fit(xs, ys)
    assert fit["a"] == pytest.approx(2.0, abs=1e-9)
    assert fit["b"] == pytest.approx(3.0, abs=1e-9)


def test_linear_fit_noisy_recovers_slope_sign():
    xs = list(range(10))
    ys = [24 + 6.5 * x + n for x, n in zip(xs, [0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.3, -0.3, 0.0, 0.0])]
    fit = linear_fit(xs, ys)
    assert fit["a"] == pytest.approx(24.0, abs=0.5)
    assert fit["b"] == pytest.approx(6.5, abs=0.2)


# ---------------------------------------------------------------- median_iqr

def test_median_iqr_basic():
    r = median_iqr([1, 2, 3, 4, 5])
    assert r["median"] == 3
    assert r["iqr"] >= 0


# ---------------------------------------------------------------- persistence helpers

def test_atomic_write_produces_valid_json_and_no_leftover_tmp(tmp_path):
    path = tmp_path / "out.json"
    dd._atomic_write_json(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    assert not path.with_name(path.name + ".tmp").exists()


def test_atomic_write_overwrites_existing_file(tmp_path):
    path = tmp_path / "out.json"
    dd._atomic_write_json(path, {"a": 1})
    dd._atomic_write_json(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 2}


def test_save_target_merges_with_other_targets_already_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "OUT", tmp_path / "out.json")
    dd._save_target("resident", {"equivalence": [1]})
    dd._save_target("streaming", {"equivalence": [2]})
    on_disk = dd._load_results()
    assert on_disk["targets"]["resident"] == {"equivalence": [1]}
    assert on_disk["targets"]["streaming"] == {"equivalence": [2]}
    assert on_disk["notes"] == dd.NOTES


def test_save_target_overwrites_only_its_own_key(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "OUT", tmp_path / "out.json")
    dd._save_target("resident", {"equivalence": [1]})
    dd._save_target("resident", {"equivalence": [1], "verify_ms": {"1": {}}})
    dd._save_target("streaming", {"equivalence": [2]})
    on_disk = dd._load_results()
    assert on_disk["targets"]["resident"] == {"equivalence": [1], "verify_ms": {"1": {}}}
    assert on_disk["targets"]["streaming"] == {"equivalence": [2]}  # untouched by the resident write


def test_target_complete_requires_equivalence_and_verify_ms(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "OUT", tmp_path / "out.json")
    assert dd._target_complete("resident") is False
    dd._save_target("resident", {"equivalence": [1]})
    assert dd._target_complete("resident") is False  # Part A only, no verify_ms yet
    dd._save_target("resident", {"equivalence": [1], "verify_ms": {"1": {}}})
    assert dd._target_complete("resident") is True
    assert dd._target_complete("streaming") is False  # never written


# ---------------------------------------------------------------- --skip-existing / main()

class _FakeTok:
    eos_token_ids = [999]

    def apply_chat_template(self, *a, **k):
        return [1, 2, 3]


def test_main_skips_only_the_already_complete_target(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "OUT", tmp_path / "out.json")
    dd._save_target("resident", {"equivalence": [1], "verify_ms": {"1": {}}})  # already complete

    calls = []
    monkeypatch.setattr(dd, "_run_target", lambda name: calls.append(name) or
                         {"equivalence": [], "verify_ms": {}})
    monkeypatch.setattr(sys, "argv", ["diag_dflash", "--target", "both", "--skip-existing"])
    dd.main()
    assert calls == ["streaming"]


def test_main_without_skip_existing_reruns_every_target(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "OUT", tmp_path / "out.json")
    dd._save_target("resident", {"equivalence": [1], "verify_ms": {"1": {}}})

    calls = []
    monkeypatch.setattr(dd, "_run_target", lambda name: calls.append(name) or
                         {"equivalence": [], "verify_ms": {}})
    monkeypatch.setattr(sys, "argv", ["diag_dflash", "--target", "both"])
    dd.main()
    assert calls == ["resident", "streaming"]


# ---------------------------------------------------------------- _run_target Part B error path

def test_run_target_records_part_b_exception_and_keeps_part_a(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "OUT", tmp_path / "out.json")
    monkeypatch.setattr(dd, "swap_gb", lambda: 0.1)
    monkeypatch.setattr(dd, "_load_target", lambda name, slots=32: ("FAKE_TARGET", _FakeTok()))
    monkeypatch.setattr(dd, "generate_greedy", lambda *a, **k: [0])
    monkeypatch.setattr(dd, "load_draft", lambda path: "FAKE_DRAFT")
    fake_equivalence = [{"prompt_index": 0, "pairs": {}, "verdict": None}]
    monkeypatch.setattr(dd, "_part_a", lambda target, draft, tok: fake_equivalence)

    def boom(target, draft, tok):
        raise ValueError("boom")

    monkeypatch.setattr(dd, "_part_b", boom)

    result = dd._run_target("resident")

    assert result["equivalence"] == fake_equivalence
    assert result["stopped"] == "partB error"
    assert "boom" in result["error"]
    assert "verify_ms" not in result
    # Part A was persisted before Part B ran (and blew up)
    on_disk = dd._load_results()
    assert on_disk["targets"]["resident"]["equivalence"] == fake_equivalence


def test_run_target_part_b_success_has_no_stopped_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "OUT", tmp_path / "out.json")
    monkeypatch.setattr(dd, "swap_gb", lambda: 0.1)
    monkeypatch.setattr(dd, "_load_target", lambda name, slots=32: ("FAKE_TARGET", _FakeTok()))
    monkeypatch.setattr(dd, "generate_greedy", lambda *a, **k: [0])
    monkeypatch.setattr(dd, "load_draft", lambda path: "FAKE_DRAFT")
    monkeypatch.setattr(dd, "_part_a", lambda target, draft, tok: [])
    monkeypatch.setattr(dd, "_part_b", lambda target, draft, tok: {
        "verify_ms": {"1": {"median": 1.0, "iqr": 0.0}}, "fit": {"a": 1.0, "b": 1.0}, "propose_ms": {}})

    result = dd._run_target("resident")

    assert result["stopped"] is None
    assert "error" not in result
    assert result["verify_ms"] == {"1": {"median": 1.0, "iqr": 0.0}}
