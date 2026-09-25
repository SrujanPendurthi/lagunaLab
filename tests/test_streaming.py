import json

import mlx.core as mx
import numpy as np

from scripts.pack_experts import _to_bytes, expert_from_bytes, pack, read_expert
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
