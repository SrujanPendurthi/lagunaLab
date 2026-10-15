import numpy as np

from laguna_mlx import streaming
from scripts.pack_experts import pack
from scripts.profile_stream import profile
from tests.tiny import TINY, quantized_model_dir


def test_profile_times_each_part_and_restores_streaming(tmp_path):
    d, _ = quantized_model_dir(tmp_path)
    pack(d, tmp_path / "packed")
    model, _ = streaming.load_streaming(tmp_path / "packed", slots=3)
    load, read = streaming.ExpertStore.load, streaming.ExpertStore._read
    ids = np.random.default_rng(0).integers(0, TINY["vocab_size"], size=40)
    r = profile(model, ids, n=6, chunk=16)
    assert r["context"] == 40 and r["steps"] == 6
    assert r["sync_ms"] > 0 and r["load_ms"] > 0 and r["step_ms"] >= r["sync_ms"] + r["load_ms"]
    assert r["misses_per_step"] > 0 and r["pread_ms"] > 0  # 3 slots < 8 experts: decode misses
    assert streaming.np is np  # patches undone
    assert streaming.ExpertStore.load is load and streaming.ExpertStore._read is read


def test_profile_runs_on_a_resident_model():
    # resident control: nothing streams, so sync and load stay zero and the step is pure compute
    from tests.tiny import mlx_only_model

    m = mlx_only_model()
    r = profile(m, np.arange(20) % TINY["vocab_size"], n=3, chunk=8)
    assert r["sync_ms"] == 0 and r["load_ms"] == 0 and r["misses_per_step"] == 0 and r["step_ms"] > 0
