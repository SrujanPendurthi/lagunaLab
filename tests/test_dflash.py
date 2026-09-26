import copy

import mlx.core as mx
import numpy as np
import pytest

from laguna_mlx import dflash
from laguna_mlx.dflash import DFlashDraft, NgramIndex, generate_dflash, generate_greedy, generate_lookup
from tests.tiny import TINY, mlx_model, torch_model

DRAFT_CFG = {
    "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
    "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 16,
    "rms_norm_eps": 1e-6, "rope_theta": 500000.0, "sliding_window": 8,
    "vocab_size": TINY["vocab_size"], "gating": "per-head",
    "layer_types": ["sliding_attention"] * 2,
    "dflash_config": {"block_size": 6, "mask_token_id": 12, "target_layer_ids": [1, 3], "causal": True},
}
PROMPT = list(np.random.default_rng(5).integers(0, TINY["vocab_size"], size=13))


@pytest.fixture(scope="module")
def target():
    m = mlx_model(torch_model())
    m.lm_head.weight = m.lm_head.weight * 50  # peaky logits: no float near-ties in argmax
    return m


@pytest.fixture(scope="module")
def draft():
    d = DFlashDraft(copy.deepcopy(DRAFT_CFG))
    mx.eval(d.parameters())
    return d


class OracleDraft:
    """Proposes the true greedy continuation, corrupted from slot `wrong` on."""

    def __init__(self, real, truth, wrong=None):
        self.real, self.truth, self.wrong = real, truth, wrong
        self.target_layer_ids, self.block_size = real.target_layer_ids, real.block_size

    def make_cache(self):
        return self.real.make_cache()

    def context_states(self, aux):
        return self.real.context_states(aux)

    def add_context(self, ctx, cache):
        self.real.add_context(ctx, cache)

    def propose(self, target, token, cache, n_draft):
        pos = cache[0].offset  # position of `token`
        out = [self.truth[pos + i] if pos + i < len(self.truth) else 0 for i in range(1, n_draft + 1)]
        if self.wrong is not None:
            out[self.wrong:] = [(t + 1) % TINY["vocab_size"] for t in out[self.wrong:]]
        return out


def test_random_draft_is_lossless(target, draft):
    want = generate_greedy(target, PROMPT, 25)
    got, stats = generate_dflash(target, draft, PROMPT, 25)
    assert got == want and stats["steps"] > 0


@pytest.mark.parametrize("wrong,expect_accept", [(None, 5), (2, 2)])
def test_oracle_draft_accepts_and_stays_lossless(target, draft, wrong, expect_accept):
    want = generate_greedy(target, PROMPT, 40)
    truth = PROMPT + want
    got, stats = generate_dflash(target, OracleDraft(draft, truth, wrong), PROMPT, 40)
    assert got == want
    assert stats["mean_accepted"] >= expect_accept - 0.5


def test_eos_inside_accepted_run_stops_like_greedy(target, draft):
    free = generate_greedy(target, PROMPT, 40)
    eos = (free[17],)  # a token that will arrive mid-block with the oracle
    want = generate_greedy(target, PROMPT, 40, eos=eos)
    got, _ = generate_dflash(target, OracleDraft(draft, PROMPT + free), PROMPT, 40, eos=eos)
    assert got == want and got[-1] == eos[0]


def test_max_tokens_mid_block(target, draft):
    want = generate_greedy(target, PROMPT, 23)
    free = generate_greedy(target, PROMPT, 40)
    got, _ = generate_dflash(target, OracleDraft(draft, PROMPT + free), PROMPT, 23)
    assert got == want


def lookup(ids, t, g, n):
    """Reference: tokens that followed the most recent earlier occurrence of ids[t-g:t], up to n."""
    if t < g:
        return []
    key = tuple(ids[t - g : t])
    for s in range(t - g - 1, -1, -1):
        if tuple(ids[s : s + g]) == key:
            return list(ids[s + g : min(s + g + n, t)])
    return []


@pytest.mark.parametrize("seq", [
    list(np.random.default_rng(1).integers(0, 5, size=300)),  # random, small alphabet: many hits
    list(np.random.default_rng(2).integers(0, 1000, size=300)),  # random: few hits
    [1, 2, 3, 4, 5] * 20 + [1, 2, 9] * 10 + [7] * 30,  # repetitive, overlapping runs
])
@pytest.mark.parametrize("g", [1, 3, 4])
def test_ngram_index_matches_reference_lookup(seq, g):
    seq = [int(t) for t in seq]
    idx = NgramIndex(g)
    for t in range(len(seq) + 1):
        for n in (1, 7, 15):
            assert idx.propose(n) == lookup(seq, t, g, n), (t, n)
        if t < len(seq):
            idx.extend(seq[t : t + 1])


def test_ngram_index_bulk_extend_matches_incremental():
    seq = [int(t) for t in np.random.default_rng(3).integers(0, 4, size=200)]
    a, b = NgramIndex(3), NgramIndex(3)
    a.extend(seq)
    for i in range(0, 200, 7):
        b.extend(seq[i : i + 7])
    assert a.last == b.last and a.propose(15) == b.propose(15) == lookup(seq, 200, 3, 15)


# The tiny random model almost never repeats itself, so natural lookup hits need a
# prompt that echoes its own prefix: PROMPT + greedy(PROMPT) + PROMPT hits at g=2.
def _echo_prompt(target):
    return PROMPT + generate_greedy(target, PROMPT, 60) + PROMPT


def test_lookup_natural_hits_are_lossless(target):
    prompt = _echo_prompt(target)
    want = generate_greedy(target, prompt, 60)
    got, stats = generate_lookup(target, prompt, 60, g=2)
    assert got == want
    assert stats["accepted"] > 0 and stats["steps_with_draft"] > 0
    assert stats["steps"] < 60


def test_lookup_random_prompt_is_lossless(target):
    want = generate_greedy(target, PROMPT, 30)
    got, stats = generate_lookup(target, PROMPT, 30)
    assert got == want and stats["steps"] > 0


@pytest.fixture
def oracle(monkeypatch):
    """Make NgramIndex propose the true greedy continuation (corrupted from slot `wrong` on)."""

    def install(truth, wrong=None):
        def propose(self, n):
            t = len(self.ids)  # ids = prompt + emitted + pending token
            out = [int(x) for x in truth[t : t + n]]
            if wrong is not None:
                out[wrong:] = [(x + 1) % TINY["vocab_size"] for x in out[wrong:]]
            return out

        monkeypatch.setattr(dflash.NgramIndex, "propose", propose)

    return install


@pytest.mark.parametrize("wrong,expect_accept", [(None, 7), (2, 2)])
def test_lookup_oracle_accepts_and_stays_lossless(target, oracle, wrong, expect_accept):
    want = generate_greedy(target, PROMPT, 40)
    oracle(PROMPT + want, wrong)
    got, stats = generate_lookup(target, PROMPT, 40, n_draft=7)
    assert got == want
    assert stats["mean_accepted"] >= expect_accept - 1.5


def test_lookup_eos_inside_accepted_run(target, oracle):
    free = generate_greedy(target, PROMPT, 40)
    eos = (free[17],)
    want = generate_greedy(target, PROMPT, 40, eos=eos)
    oracle(PROMPT + free)
    got, _ = generate_lookup(target, PROMPT, 40, eos=eos)
    assert got == want and got[-1] == eos[0]


def test_lookup_max_tokens_mid_block(target, oracle):
    free = generate_greedy(target, PROMPT, 40)
    want = generate_greedy(target, PROMPT, 23)
    oracle(PROMPT + free)
    got, _ = generate_lookup(target, PROMPT, 23)
    assert got == want


@pytest.mark.parametrize("wrong", [None, 1])
def test_lookup_adaptive_is_lossless(target, oracle, wrong):
    want = generate_greedy(target, PROMPT, 40)
    oracle(PROMPT + want, wrong)
    got, stats = generate_lookup(target, PROMPT, 40, n_draft=15, adaptive=True)
    assert got == want
    if wrong == 1:  # cap is 1 + previous accepted = 2 after the first step
        assert stats["drafted"] <= 15 + 2 * (stats["steps"] - 1)


def test_lookup_adaptive_natural_is_lossless(target):
    prompt = _echo_prompt(target)
    want = generate_greedy(target, prompt, 60)
    got, _ = generate_lookup(target, prompt, 60, g=2, n_draft=15, adaptive=True)
    assert got == want
