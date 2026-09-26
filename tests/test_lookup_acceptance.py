from scripts.lookup_acceptance import GramIndex, common_prefix_len, lookup, simulate

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
