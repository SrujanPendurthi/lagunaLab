import json

import mlx.core as mx
import numpy as np

from scripts.pack_experts import pack, read_expert
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
