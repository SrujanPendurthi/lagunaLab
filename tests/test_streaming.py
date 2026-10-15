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


def _drop_setup(tmp_path):
    d, resident = quantized_model_dir(tmp_path)
    out = tmp_path / "packed"
    pack(d, out)
    streamed, _ = load_streaming(out, slots=4)
    l = json.loads((out / "layout.json").read_text())["layers"][0]
    x = mx.random.normal((1, 1, TINY["hidden_size"]), key=mx.random.key(3))
    inds, w = resident.layers[l].mlp.route(x)
    return resident.layers[l].mlp, streamed.layers[l].mlp, x, inds, w


def test_drop_misses_off_by_default_and_exact_when_nothing_misses(tmp_path, monkeypatch):
    from laguna_mlx.streaming import StreamingSwitchGLU

    assert StreamingSwitchGLU.drop_misses is False
    res_block, st_block, x, inds, w = _drop_setup(tmp_path)
    store = st_block.switch_mlp._store[0]
    store.load(np.unique(np.array(inds)).tolist())  # everything this token needs is resident
    monkeypatch.setattr(StreamingSwitchGLU, "drop_misses", True)
    got = st_block.switch_mlp(x, inds)
    np.testing.assert_allclose(np.array(got), np.array(res_block.switch_mlp(x, inds)), atol=1e-5, rtol=1e-5)
    assert store.dropped == 0


def test_drop_misses_zeroes_missing_experts_and_renormalizes(tmp_path, monkeypatch):
    from laguna_mlx.streaming import StreamingSwitchGLU

    res_block, st_block, x, inds, w = _drop_setup(tmp_path)
    store = st_block.switch_mlp._store[0]
    picked = np.array(inds)[0, 0].tolist()  # top-2 in the tiny config
    store.load([picked[0]])  # first pick resident, second misses
    monkeypatch.setattr(StreamingSwitchGLU, "drop_misses", True)
    got = (st_block.switch_mlp(x, inds) * w[..., None]).sum(axis=-2)
    present = mx.array([[[1.0, 0.0]]])
    full = res_block.switch_mlp(x, inds)
    want = (full * (w * present / (w * present).sum(-1, keepdims=True))[..., None]).sum(axis=-2)
    np.testing.assert_allclose(np.array(got), np.array(want), atol=1e-5, rtol=1e-5)
    assert store.dropped == 1 and store.decode_picks == 2
    assert picked[1] in store.expert_slot  # the miss is still loaded for later tokens


def test_prefill_leaves_no_pending_slot_writes(tmp_path):
    # mlx-lm's prefill evals only the caches; nothing reads the last MoE layer's output, so its
    # slot writes would stay pending and pile up across chunks (found by moeStream)
    d, _ = quantized_model_dir(tmp_path)
    out = tmp_path / "packed"
    pack(d, out)
    streamed, _ = load_streaming(out, slots=2)
    cache = streamed.make_cache()
    ids = _ids(64, seed=4)
    for a in range(0, 64, 16):
        streamed(ids[:, a : a + 16], cache=cache)
        mx.eval([c.state for c in cache])
    last = [l for l in streamed.layers if hasattr(l.mlp, "switch_mlp")][-1].mlp.switch_mlp._store[0].glu
    mx.synchronize()
    before = mx.get_active_memory()
    mx.eval(last.parameters())
    assert before - mx.get_active_memory() == 0
