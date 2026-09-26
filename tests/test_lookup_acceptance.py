import pytest

from scripts.lookup_acceptance import GramIndex, build_measured_cost_fn, common_prefix_len, lookup, simulate

# Hand-made ids: prefix [0, 1], then "2 3 4 5" repeated, then "2 3 4" followed by a divergent 9.
#   idx:  0  1  2  3  4  5  6  7  8  9  10 11 12 13
IDS = [0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 9]


def test_lookup_reference_hand_traced():
    # t < g -> no proposal
    assert lookup(IDS, 1, 2, 15) == []
    # at t=6/7 the 2-gram key has no earlier occurrence yet
    assert lookup(IDS, 6, 2, 15) == []
    assert lookup(IDS, 7, 2, 15) == []
    # at t=8, key=(2,3) last occurred at s=2; continuation ids[4:8]
    assert lookup(IDS, 8, 2, 15) == [4, 5, 2, 3]
    # at t=13, key=(3,4) last occurred at s=7 (more recent than s=3); continuation ids[9:13]
    assert lookup(IDS, 13, 2, 15) == [5, 2, 3, 4]


def test_gram_index_matches_reference_exhaustively():
    for g in (1, 2, 3, 4):
        idx = GramIndex(IDS, g)
        for t in range(len(IDS) + 1):
            assert idx.propose(t, 15) == lookup(IDS, t, g, 15)


def test_common_prefix_len():
    assert common_prefix_len([], [1, 2]) == 0
    assert common_prefix_len([1, 2, 3], [1, 2, 9]) == 2
    assert common_prefix_len([1, 2], [1, 2]) == 2


def test_simulate_fixed_matches_hand_derivation():
    # assistant span is ids[6:14]; steps at t=6,7 (no proposal), t=8 (full 4-token accept),
    # t=13 (proposal but first token diverges from the trailing 9)
    r = simulate(IDS, [(6, 14)], g=2, n=15, adaptive=False)
    assert r == {"steps": 4, "hits": 2, "accepted_sum": 4, "tokens_generated": 8, "total_cost_ms": 148.0}
    assert r["tokens_generated"] == 14 - 6  # exact span length here, no overshoot
    speedup = (r["tokens_generated"] / r["total_cost_ms"]) / (1 / 24.0)
    assert abs(speedup - 8 * 24 / 148.0) < 1e-9


def test_simulate_adaptive_caps_draft_length():
    # same span; adaptive shrinks the draft after low-acceptance steps, taking 5 steps
    # instead of 4 but still covering the same 8 tokens
    r = simulate(IDS, [(6, 14)], g=2, n=15, adaptive=True)
    assert r == {"steps": 5, "hits": 3, "accepted_sum": 3, "tokens_generated": 8, "total_cost_ms": 159.0}


def test_no_span_is_a_no_op():
    r = simulate(IDS, [], g=2, n=15, adaptive=False)
    assert r == {"steps": 0, "hits": 0, "accepted_sum": 0, "tokens_generated": 0, "total_cost_ms": 0.0}


# Second span begins with "1 2" (matching s=0's "1 2"), so at t=10 the draft copies the whole
# repeat: [3,4,5,6,7,8, 1,2] (8 tokens). But the assistant span itself only covers ids[10:14]
# (values 3,4,5,6) -- ids[14:17] = 7,8,9 is the *next* message, which coincidentally also
# starts with 7,8. An unclamped comparison would accept k=6 (matching through idx 15) and
# advance t by 7, overshooting the 4-token span by 3. The clamp must cap both the accept
# window and the credited advance at the span's own end.
IDS_BOUNDARY = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_simulate_clamps_accept_window_and_advance_to_span_end():
    r = simulate(IDS_BOUNDARY, [(10, 14)], g=2, n=15, adaptive=False)
    # clamped: k is capped at the 4 remaining span tokens (all match), advance capped at 4
    # (not k+1=5, which would run past the span's own end)
    assert r == {"steps": 1, "hits": 1, "accepted_sum": 4, "tokens_generated": 4, "total_cost_ms": 76.0}
    assert r["tokens_generated"] == 14 - 10  # exact span length, no overshoot

    # confirm the bug this guards against: uncapped, the draft's tail ("...,1,2") diverges
    # from the true continuation only after matching 6 tokens into the next message
    idx = GramIndex(IDS_BOUNDARY, 2)
    d = idx.propose(10, 15)
    unclamped_window = IDS_BOUNDARY[10 : 10 + len(d)]
    assert common_prefix_len(d, unclamped_window) == 6  # would overshoot the 4-token span by 3


# Hand-made measured-cost fixture: L=1,3,5 measured (median only; iqr isn't read by the cost fn).
VERIFY_MS = {"1": {"median": 10.0, "iqr": 0.0}, "3": {"median": 20.0, "iqr": 0.0}, "5": {"median": 44.0, "iqr": 0.0}}


def test_measured_cost_fn_exact_hit():
    cost = build_measured_cost_fn(VERIFY_MS)
    assert cost.measured_L == [1, 3, 5]
    assert cost(0) == 10.0  # L=1, measured directly
    assert cost(2) == 20.0  # L=3, measured directly
    assert cost(4) == 44.0  # L=5, measured directly


def test_measured_cost_fn_interpolates_between_nearest_measured_l():
    cost = build_measured_cost_fn(VERIFY_MS)
    # L=2 (m=1): halfway between L=1 (10.0) and L=3 (20.0)
    assert cost(1) == pytest.approx(15.0)
    # L=4 (m=3): halfway between L=3 (20.0) and L=5 (44.0)
    assert cost(3) == pytest.approx(32.0)


def test_measured_cost_fn_refuses_to_extrapolate():
    cost = build_measured_cost_fn(VERIFY_MS)
    with pytest.raises(ValueError):
        cost(-1)  # L=0, below the lowest measured L=1
    with pytest.raises(ValueError):
        cost(5)  # L=6, above the highest measured L=5
