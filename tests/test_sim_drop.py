import numpy as np

from laguna_mlx.streaming import StreamingSwitchGLU, load_streaming
from scripts.pack_experts import pack
from scripts.sim_drop import run_session, summarize, window_starts
from tests.tiny import TINY, quantized_model_dir


def test_window_starts_snap_to_assistant_tokens():
    mask = np.zeros(200, bool)
    mask[50:60] = True
    s = window_starts(200, ctx=200, w=8, k=4, mask=mask)
    assert s[0] == 49  # 47 snaps forward so its first target (50) is assistant
    assert s[1:] == window_starts(200, ctx=200, w=8, k=4)[1:]  # no assistant tokens nearby: unchanged


def test_window_starts_stay_inside_context():
    s = window_starts(100, ctx=60, w=8, k=4)
    assert len(s) == 4 and s[-1] + 8 <= 60 and all(b - a >= 8 for a, b in zip(s, s[1:]))


def test_run_session_measures_drops_and_restores_the_flag(tmp_path):
    d, _ = quantized_model_dir(tmp_path)
    pack(d, tmp_path / "packed")
    ids = np.random.default_rng(0).integers(0, TINY["vocab_size"], size=80)
    mask = np.ones(80, bool)
    few = run_session(load_streaming(tmp_path / "packed", slots=2)[0], ids, mask, ctx=64, w=6, k=3, chunk=16)
    assert few["assistant_tokens"] == 18 and few["kl_mean"] >= 0 and 0 <= few["top1"] <= 1
    assert few["drop_rate"] > 0  # 2 of 8 slots: decode tokens must miss
    assert StreamingSwitchGLU.drop_misses is False
    # every expert fits: nothing to drop once warm, and drop changes nothing
    all_ = run_session(load_streaming(tmp_path / "packed", slots=8)[0], ids, mask, ctx=64, w=6, k=3, chunk=16)
    assert all_["drop_rate"] < few["drop_rate"]
    s = summarize({"a": few, "b": all_})
    assert s["sessions"] == 2 and isinstance(s["passes"], bool)
