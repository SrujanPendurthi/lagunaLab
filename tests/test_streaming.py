import json

import mlx.core as mx
import numpy as np

from scripts.pack_experts import _to_bytes, expert_from_bytes, pack, read_expert
from laguna_mlx.streaming import load_streaming, stream_stats
from tests.tiny import TINY, quantized_model_dir


def test_pack_roundtrips_every_expert_bit_exact(tmp_path):
    d, m = quantized_model_dir(tmp_path)
    out = tmp_path / "packed"
    pack(d, out)
    layout = json.loads((out / "layout.json").read_text())
    assert layout["layers"] == [1, 2, 3] and layout["stride"] % 16384 == 0
    for l in layout["layers"]:
        sw = m.layers[l].mlp.switch_mlp
        for e in range(TINY["num_experts"]):
            got = read_expert(out, layout, l, e)
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for part in ("weight", "scales", "biases"):
                    want = getattr(getattr(sw, proj), part)[e]
                    have = got[f"{proj}.{part}"]
                    assert have.dtype == want.dtype
                    assert mx.array_equal(have, want).item(), (l, e, proj, part)
    resident = mx.load(str(out / "resident.safetensors"))
    assert not any("switch_mlp" in k for k in resident)
    assert "model.layers.1.mlp.gate.weight" in resident


def test_bfloat16_round_trips_bit_exact():
    """The real model's scales/biases are bfloat16 (the tiny test model is float32,
    so the main test above never exercises NP_VIEW["bfloat16"] / the uint16 view
    reinterpretation). Round-trip edge values + random values through the exact
    _to_bytes -> expert_from_bytes path used by pack()/read_expert()."""
    edge = [0.0, -0.0, 1.0, -1.0, 3.38e38, 1.18e-38, float("inf"), float("-inf")]
    rng = np.random.default_rng(0)
    vals = edge + list(rng.standard_normal(24))
    a = mx.array(vals, dtype=mx.bfloat16)

    raw = _to_bytes(a)
    layout = {"arrays": {"x": {"offset": 0, "shape": list(a.shape), "dtype": "bfloat16"}}}
    got = expert_from_bytes(raw, layout)["x"]

    assert got.dtype == mx.bfloat16
    assert _to_bytes(got) == raw  # bit-exact, including the -0.0 sign bit
    assert mx.array_equal(got, a).item()


def _ids(n, seed=6):
    return mx.array(np.random.default_rng(seed).integers(0, TINY["vocab_size"], size=(1, n)))


def test_streaming_matches_resident_with_fewer_slots_than_experts(tmp_path):
    d, resident = quantized_model_dir(tmp_path)
    out = tmp_path / "packed"
    pack(d, out)
    streamed, _ = load_streaming(out, slots=3)  # 3 < 8 experts: prefill must group
    ids = _ids(20)
    np.testing.assert_allclose(np.array(streamed(ids)), np.array(resident(ids)), atol=1e-5, rtol=1e-5)
    s = stream_stats(streamed)
    assert s["misses"] > 0 and s["bytes_read"] > 0


def test_streaming_decode_with_cache_matches_resident(tmp_path):
    d, resident = quantized_model_dir(tmp_path)
    out = tmp_path / "packed"
    pack(d, out)
    streamed, _ = load_streaming(out, slots=4)
    ids = _ids(16, seed=7)
    want = np.array(resident(ids))
    cache = streamed.make_cache()
    got = [np.array(streamed(ids[:, :10], cache=cache))]
    got += [np.array(streamed(ids[:, i : i + 1], cache=cache)) for i in range(10, 16)]
    np.testing.assert_allclose(np.concatenate(got, axis=1), want, atol=1e-5, rtol=1e-5)
    assert stream_stats(streamed)["hits"] > 0  # decode reuses cached experts


def test_streaming_bfloat16_is_exact_and_stays_bfloat16(tmp_path):
    d, resident = quantized_model_dir(tmp_path, dtype=mx.bfloat16)
    out = tmp_path / "packed"
    pack(d, out)
    streamed, _ = load_streaming(out, slots=3)
    ids = _ids(20)
    got, want = streamed(ids), resident(ids)
    assert want.dtype == mx.bfloat16 and got.dtype == mx.bfloat16
    assert mx.array_equal(got, want).item()


def test_streaming_matches_resident_on_sorted_switchglu_path(tmp_path):
    d, resident = quantized_model_dir(tmp_path)
    out = tmp_path / "packed"
    pack(d, out)
    streamed, _ = load_streaming(out, slots=3)
    ids = _ids(48, seed=8)  # 48 tokens * top-2 = 96 >= 64: SwitchGLU sorts indices
    np.testing.assert_allclose(np.array(streamed(ids)), np.array(resident(ids)), atol=1e-5, rtol=1e-5)


def test_cache_limit_gb_is_opt_in(tmp_path):
    # default None leaves MLX's buffer-pool limit alone (moeStream's numbers keep their meaning)
    d, _ = quantized_model_dir(tmp_path)
    out = tmp_path / "packed"
    pack(d, out)
    before = mx.set_cache_limit(123 << 20)
    try:
        load_streaming(out, slots=3)
        assert mx.set_cache_limit(123 << 20) == 123 << 20
        load_streaming(out, slots=3, cache_limit_gb=0.5)
        assert mx.set_cache_limit(123 << 20) == int(0.5 * (1 << 30))
    finally:
        mx.set_cache_limit(before)
