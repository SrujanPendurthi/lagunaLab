"""Split an MLX Laguna checkpoint into resident weights + per-layer expert files.

python3 scripts/pack_experts.py [--model models/Laguna-XS-2.1-mlx-4bit] [--out models/Laguna-XS-2.1-packed]
Values are copied byte for byte; nothing is dequantized or requantized.
"""
import argparse
import glob
import json
import re
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np

PAGE = 16384
PROJS = ("gate_proj", "up_proj", "down_proj")
PARTS = ("weight", "scales", "biases")
NP_VIEW = {"uint32": (mx.uint32, np.uint32), "bfloat16": (mx.uint16, np.uint16),
           "float16": (mx.float16, np.float16), "float32": (mx.float32, np.float32)}
KEY = re.compile(r"model\.layers\.(\d+)\.mlp\.switch_mlp\.(\w+)\.(weight|scales|biases)$")


def _dtype_name(a):
    return str(a.dtype).replace("mlx.core.", "")


def _to_bytes(a):
    view, _ = NP_VIEW[_dtype_name(a)]
    return np.array(a.view(view)).tobytes()


def pack(model_dir, out_dir):
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    (out_dir / "experts").mkdir(parents=True, exist_ok=True)
    weights = {}
    for f in sorted(glob.glob(str(model_dir / "*.safetensors"))):
        weights.update(mx.load(f))
    layers = sorted({int(m.group(1)) for k in weights if (m := KEY.match(k))})
    first = f"model.layers.{layers[0]}.mlp.switch_mlp"
    E = weights[f"{first}.gate_proj.weight"].shape[0]
    arrays, offset = {}, 0
    for proj in PROJS:
        for part in PARTS:
            a = weights[f"{first}.{proj}.{part}"][0]
            arrays[f"{proj}.{part}"] = {"offset": offset, "shape": list(a.shape), "dtype": _dtype_name(a)}
            offset += a.nbytes
    stride = -(-offset // PAGE) * PAGE
    for l in layers:
        p = f"model.layers.{l}.mlp.switch_mlp"
        stacked = {n: weights.pop(f"{p}.{n}") for n in arrays}
        with open(out_dir / "experts" / f"layer_{l:02d}.bin", "wb") as f:
            for e in range(E):
                blob = b"".join(_to_bytes(stacked[n][e]) for n in arrays)
                f.write(blob + b"\0" * (stride - len(blob)))
        del stacked
    mx.save_safetensors(str(out_dir / "resident.safetensors"), weights)
    layout = {"num_experts": E, "stride": stride, "layers": layers, "arrays": arrays}
    (out_dir / "layout.json").write_text(json.dumps(layout, indent=1))
    for f in model_dir.iterdir():
        if f.suffix in (".json", ".jinja") and f.name != "model.safetensors.index.json":
            shutil.copy(f, out_dir / f.name)


def expert_from_bytes(buf, layout):
    out = {}
    for name, a in layout["arrays"].items():
        mxv, npv = NP_VIEW[a["dtype"]]
        count = int(np.prod(a["shape"]))
        arr = np.frombuffer(buf, dtype=npv, count=count, offset=a["offset"]).reshape(a["shape"])
        out[name] = mx.array(arr).view(getattr(mx, a["dtype"]))
    return out


def read_expert(out_dir, layout, layer, e):
    with open(Path(out_dir) / "experts" / f"layer_{layer:02d}.bin", "rb") as f:
        f.seek(e * layout["stride"])
        return expert_from_bytes(f.read(layout["stride"]), layout)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Laguna-XS-2.1-mlx-4bit")
    p.add_argument("--out", default="models/Laguna-XS-2.1-packed")
    a = p.parse_args()
    pack(a.model, a.out)
