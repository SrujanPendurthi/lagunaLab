"""One full-attention layer's decode attention on synthetic K/V with the real model's shapes.

python3 -m scripts.micro_kv_decode
Compares, per token: plain bf16 K/V + fused SDPA; 8-bit cache dequantized then fused SDPA
(DequantKVCache today); 8-bit cache read by mlx-lm's quantized SDPA (quantized_decode=True);
8-bit cache read by the fused Metal kernel (fused_decode=True, laguna_mlx.q8_attn).
No weights; under 1 GB at the longest context.
"""
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.base import quantized_scaled_dot_product_attention

from laguna_mlx.q8_attn import q8_decode_attention

OUT = Path("results/kv_decode_micro_fused.json")  # kv_decode_micro.json is the run before the fused kernel
HQ, HKV, D, G, BITS, N = 48, 8, 128, 64, 8, 50


def bench(f):
    for _ in range(5):
        mx.eval(f())
    t0 = time.perf_counter()
    for _ in range(N):
        mx.eval(f())
    return (time.perf_counter() - t0) / N * 1e3


def main():
    rows = {}
    for ctx in (8192, 32768, 82671):
        q = mx.random.normal((1, HQ, 1, D)).astype(mx.bfloat16)
        k, v = (mx.random.normal((1, HKV, ctx, D)).astype(mx.bfloat16) for _ in range(2))
        qk, qv = mx.quantize(k, group_size=G, bits=BITS), mx.quantize(v, group_size=G, bits=BITS)
        mx.eval(q, k, v, qk, qv)
        s = D**-0.5
        deq = lambda t: mx.dequantize(*t, group_size=G, bits=BITS)
        rows[ctx] = {
            "plain_ms": bench(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=s, mask=None)),
            "dequant_sdpa_ms": bench(lambda: mx.fast.scaled_dot_product_attention(q, deq(qk), deq(qv), scale=s, mask=None)),
            "dequant_only_ms": bench(lambda: (deq(qk), deq(qv))),
            "quantized_sdpa_ms": bench(lambda: quantized_scaled_dot_product_attention(q, qk, qv, scale=s, mask=None, group_size=G, bits=BITS)),
            "fused_q8_ms": bench(lambda: q8_decode_attention(q, qk, qv, ctx, s)),
            "bf16_kv_gb": (k.nbytes + v.nbytes) / 1e9,
            "q8_kv_gb": sum(a.nbytes for a in (*qk, *qv)) / 1e9,
        }
        print(ctx, {a: round(b, 3) for a, b in rows[ctx].items()}, flush=True)
        del q, k, v, qk, qv
        mx.clear_cache()
    OUT.write_text(json.dumps({"shapes": {"q_heads": HQ, "kv_heads": HKV, "head_dim": D, "group_size": G,
                                          "bits": BITS, "iters": N}, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
