"""bf16 HF checkpoint -> MLX 4-bit (router gates 8-bit via Model.quant_predicate).

python scripts/convert.py [--hf models/Laguna-XS-2.1] [--out models/Laguna-XS-2.1-mlx-4bit]
"""
import argparse

import laguna_mlx  # noqa: F401  registers model_type "laguna"
from mlx_lm import convert

p = argparse.ArgumentParser()
p.add_argument("--hf", default="models/Laguna-XS-2.1")
p.add_argument("--out", default="models/Laguna-XS-2.1-mlx-4bit")
a = p.parse_args()
convert(a.hf, mlx_path=a.out, quantize=True, q_bits=4, q_group_size=64)
