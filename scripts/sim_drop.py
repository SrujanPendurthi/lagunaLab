"""Stage 1 of docs/gpu-side-routing.md: the quality cost of drop-on-miss, simulated.

python3 -m scripts.sim_drop      # pruned-50 pack, 64 slots, 1 GB pool cap, the 5 django sessions

Replay scores in 2,048-token chunks, but drop-on-miss is a per-token decode behaviour, so each
session runs an exact chunked prefill up to CTX tokens with N_WIN teacher-forced, token-by-token
windows of W tokens at evenly spaced points. Each window runs once with drop on, is trimmed off
the KV cache, then runs again exactly. On assistant target tokens inside the windows we record
KL(exact || drop) over the full vocabulary and top-1 agreement, plus the drop rate per layer.
No speed number: the simulation still syncs and loads at every layer, so its timing says
nothing about a real sync-free path.
Resumable: finished sessions in results/drop_sim.json are skipped.
"""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

import laguna_mlx  # noqa: F401
from laguna_mlx.cache import make_trimmable_cache
from laguna_mlx.streaming import StreamingSwitchGLU

PACK = "models/Laguna-XS-2.1-pruned-50-packed"
REFS = Path("refs/kv")  # only the token ids and assistant masks are used
OUT = Path("results/drop_sim.json")
SLOTS, CTX, W, N_WIN = 64, 32768, 128, 8
MIN_TOP1, MAX_KL = 0.98, 0.02  # the Part 3 gate


def window_starts(n, ctx=CTX, w=W, k=N_WIN, mask=None):
    """k evenly spaced, non-overlapping windows; with `mask`, each snaps forward to an assistant span."""
    end = min(n - 1, ctx) - w  # the last window still has a target token after it
    gap = end // k
    starts = [end * (i + 1) // k for i in range(k)]
    if mask is not None:
        for i, t in enumerate(starts):
            hits = np.flatnonzero(mask[t + 1 : t + 1 + max(0, gap - w)])
            if len(hits) and t + hits[0] <= end:
                starts[i] = int(t + hits[0])
    return starts


def _window(model, ids, s, w, cache, drop):
    StreamingSwitchGLU.drop_misses = drop
    try:
        out = []
        for t in range(s, s + w):
            lp = model(mx.array(ids[None, t : t + 1]), cache=cache)[0, -1].astype(mx.float32)
            lp = lp - mx.logsumexp(lp)
            mx.eval(lp)
            out.append(lp)
        return out
    finally:
        StreamingSwitchGLU.drop_misses = False


def run_session(model, ids, mask, ctx=CTX, w=W, k=N_WIN, chunk=2048):
    stores = [l.mlp.switch_mlp._store[0] for l in model.layers
              if isinstance(getattr(l.mlp, "switch_mlp", None), StreamingSwitchGLU)]
    for st in stores:
        st.dropped = st.decode_picks = 0
    cache = make_trimmable_cache(model, slack=w)
    pos, kls, agree = 0, [], []
    for s in window_starts(len(ids), ctx, w, k, mask):
        for a in range(pos, s, chunk):
            mx.eval(model(mx.array(ids[None, a : min(a + chunk, s)]), cache=cache))
        on = _window(model, ids, s, w, cache, drop=True)
        for c in cache:
            c.trim(w)
        off = _window(model, ids, s, w, cache, drop=False)
        for j in range(w):
            if mask[s + j + 1]:  # logits at position t predict token t+1
                kls.append((mx.exp(off[j]) * (off[j] - on[j])).sum().item())
                agree.append(mx.argmax(off[j]).item() == mx.argmax(on[j]).item())
        pos = s + w
    picks = [st.decode_picks for st in stores]
    return {"tokens": int(len(ids)), "ctx": int(min(len(ids), ctx)), "windows": k, "window": w,
            "assistant_tokens": len(kls), "kl_mean": float(np.mean(kls)) if kls else 0.0,
            "kl_p99": float(np.percentile(kls, 99)) if kls else 0.0,
            "top1": float(np.mean(agree)) if agree else 1.0,
            "drop_rate": sum(st.dropped for st in stores) / max(1, sum(picks)),
            "drop_rate_per_layer": [st.dropped / max(1, p) for st, p in zip(stores, picks)]}


def summarize(rows):
    w = np.array([r["assistant_tokens"] for r in rows.values()])
    s = {"sessions": len(rows),
         "kl_mean": float(np.average([r["kl_mean"] for r in rows.values()], weights=w)),
         "kl_p99_max": max(r["kl_p99"] for r in rows.values()),
         "top1": float(np.average([r["top1"] for r in rows.values()], weights=w)),
         "drop_rate": float(np.mean([r["drop_rate"] for r in rows.values()]))}
    s["passes"] = bool(s["top1"] >= MIN_TOP1 and s["kl_mean"] <= MAX_KL)
    return s


def main():
    from laguna_mlx.streaming import load_streaming

    res = json.loads(OUT.read_text()) if OUT.exists() else {"rows": {}}
    res.update(pack=PACK, slots=SLOTS, ctx=CTX, window=W, windows=N_WIN, cache_limit_gb=1.0,
               note="KL is full-vocab KL(exact streaming || drop), not the top-64 truncated KL of kv.json")
    model = load_streaming(PACK, SLOTS, cache_limit_gb=1.0)[0]
    for r in sorted(REFS.glob("*.npz")):
        if r.stem in res["rows"]:
            continue
        ref = np.load(r)
        row = run_session(model, ref["ids"], ref["mask"])
        res["rows"][r.stem] = row
        res["summary"] = summarize(res["rows"])
        OUT.write_text(json.dumps(res, indent=1))
        print(r.stem, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()
                       if k != "drop_rate_per_layer"}, flush=True)
        mx.clear_cache()
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
