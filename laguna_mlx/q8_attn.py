"""Fused single-token attention over an 8-bit quantized KV cache (flash-decoding style).

One Metal pass reads the packed K/V once, dequantizes in registers and keeps an online softmax;
a few small mx ops merge the per-block partials. Nothing the size of the cache is written, unlike
dequantize + SDPA (two bf16 copies per layer per token) or mlx-lm's quantized SDPA (separate
quantized matmuls with a score matrix in between).

Layout (mlx.quantize, bits=8, group_size=64, head_dim 128): a key row is 32 uint32, each holding
4 consecutive values, low byte first; 2 scales and 2 biases per row. Lane i of a simdgroup owns
uint32 i, i.e. dims 4i..4i+3, which all sit in group i // 16.
"""
import mlx.core as mx

BLOCK = 512  # keys per threadgroup; 82k context -> 162 blocks x 8 kv heads
SIMDS = 8    # simdgroups per threadgroup

_SOURCE = """
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint blk = threadgroup_position_in_grid.x;
    uint hk = threadgroup_position_in_grid.y;
    uint b = threadgroup_position_in_grid.z;
    uint nblk = threadgroups_per_grid.x;
    int n = n_keys[0];
    uint cap = kq_shape[2];
    uint hkv = kq_shape[1];

    // queries of the G heads that share kv head hk; this lane's 4 dims, already scaled
    float qv[G][4];
    for (int g = 0; g < G; g++) {
        uint h = hk * G + g;
        for (int j = 0; j < 4; j++) qv[g][j] = q[((b * hkv * G + h) * D) + lane * 4 + j];
    }
    float m[G], l[G], acc[G][4];
    for (int g = 0; g < G; g++) {
        m[g] = -INFINITY; l[g] = 0.0f;
        for (int j = 0; j < 4; j++) acc[g][j] = 0.0f;
    }

    size_t row0 = (size_t)(b * hkv + hk) * cap;
    uint grp = lane / (64 / 4);
    int start = blk * BLOCK;
    int stop = min(start + BLOCK, n);
    for (int t = start + sg; t < stop; t += SIMDS) {
        size_t r = row0 + t;
        uint kw = kq[r * (D / 4) + lane];
        float ks = float(kscale[r * (D / 64) + grp]);
        float kb = float(kbias[r * (D / 64) + grp]);
        float kf[4];
        for (int j = 0; j < 4; j++) kf[j] = ks * float((kw >> (8 * j)) & 0xff) + kb;
        uint vw = vq[r * (D / 4) + lane];
        float vs = float(vscale[r * (D / 64) + grp]);
        float vb = float(vbias[r * (D / 64) + grp]);
        float vf[4];
        for (int j = 0; j < 4; j++) vf[j] = vs * float((vw >> (8 * j)) & 0xff) + vb;
        for (int g = 0; g < G; g++) {
            float s = 0.0f;
            for (int j = 0; j < 4; j++) s += qv[g][j] * kf[j];
            s = simd_sum(s);
            float mn = max(m[g], s);
            float c = fast::exp(m[g] - mn);
            float p = fast::exp(s - mn);
            l[g] = l[g] * c + p;
            for (int j = 0; j < 4; j++) acc[g][j] = acc[g][j] * c + p * vf[j];
            m[g] = mn;
        }
    }

    // merge the SIMDS simdgroups of this threadgroup
    threadgroup float tm[SIMDS][G];
    threadgroup float tl[SIMDS][G];
    threadgroup float ta[SIMDS][G][D];
    for (int g = 0; g < G; g++) {
        if (lane == 0) { tm[sg][g] = m[g]; tl[sg][g] = l[g]; }
        for (int j = 0; j < 4; j++) ta[sg][g][lane * 4 + j] = acc[g][j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // simdgroup sg writes heads g == sg (mod SIMDS)
    for (int g = sg; g < G; g += SIMDS) {
        float M = -INFINITY;
        for (int s2 = 0; s2 < SIMDS; s2++) M = max(M, tm[s2][g]);
        float L = 0.0f, A[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int s2 = 0; s2 < SIMDS; s2++) {
            float w = (tm[s2][g] == -INFINITY) ? 0.0f : fast::exp(tm[s2][g] - M);
            L += w * tl[s2][g];
            for (int j = 0; j < 4; j++) A[j] += w * ta[s2][g][lane * 4 + j];
        }
        size_t o = ((size_t)(b * hkv * G + hk * G + g) * nblk + blk);
        if (lane == 0) { pm[o] = M; pl[o] = L; }
        for (int j = 0; j < 4; j++) pa[o * D + lane * 4 + j] = A[j];
    }
"""

_kernel = mx.fast.metal_kernel(
    name="laguna_q8_decode_attn",
    input_names=["q", "kq", "kscale", "kbias", "vq", "vscale", "vbias", "n_keys"],
    output_names=["pm", "pl", "pa"],
    source=_SOURCE,
)


def q8_decode_attention(q, keys, values, n, scale):
    """q [B, Hq, 1, D]; keys/values: QuantizedKVCache's full-capacity (packed, scales, biases)
    tuples, of which the first `n` rows are valid. Returns [B, Hq, 1, D] in q's dtype.

    Passing the full buffers (not the [:n] views) keeps them row-contiguous, so metal_kernel
    doesn't copy the cache every token.
    """
    B, Hq, L, D = q.shape
    assert L == 1 and D == 128, "decode only, head_dim 128"
    assert keys[0].shape[-1] == D // 4, "8-bit, group_size 64 only"
    hkv = keys[0].shape[1]
    G = Hq // hkv
    # threadgroup memory: SIMDS x G x D floats of partial outputs, under Metal's 32 KB
    assert SIMDS * G * D * 4 <= 28 * 1024, f"{G} q heads per kv head is too many for one threadgroup"
    nblk = max(1, -(-n // BLOCK))
    qs = (q.astype(mx.float32) * scale).reshape(B, Hq, D)
    pm, pl, pa = _kernel(
        inputs=[qs, *keys, *values, mx.array([n], dtype=mx.int32)],
        template=[("G", G), ("D", D), ("BLOCK", BLOCK), ("SIMDS", SIMDS)],
        grid=(nblk * 32 * SIMDS, hkv, B),
        threadgroup=(32 * SIMDS, 1, 1),
        output_shapes=[(B, Hq, nblk), (B, Hq, nblk), (B, Hq, nblk, D)],
        output_dtypes=[mx.float32] * 3,
    )
    # merge blocks; every block has >= 1 key, so no block is all -inf
    M = pm.max(axis=-1, keepdims=True)
    w = mx.exp(pm - M)
    out = (w[..., None] * pa).sum(axis=2) / (w * pl).sum(axis=-1)[..., None]
    return out[:, :, None, :].astype(q.dtype)
