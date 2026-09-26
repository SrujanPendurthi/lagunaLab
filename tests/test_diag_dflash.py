"""Unit tests for scripts/diag_dflash.py's pure helpers -- no model loading."""
import mlx.core as mx
import numpy as np
import pytest

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
