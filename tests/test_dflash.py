import copy

import mlx.core as mx
import numpy as np
import pytest

from laguna_mlx.dflash import DFlashDraft, generate_dflash, generate_greedy
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
