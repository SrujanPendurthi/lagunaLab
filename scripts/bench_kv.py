"""KV cache quantization on the pruned model: memory, speed and replay quality.

python3 -m scripts.bench_kv                    # resident: plain, 8-bit, 4-bit on 5 django sessions
python3 -m scripts.bench_kv --target streaming # step 0 only: plain cache, longest session
Resumable: finished rows in results/kv.json are skipped.
"""
import argparse
import glob
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import laguna_mlx  # noqa: F401
from laguna_mlx.cache import make_quantized_cache
from replay.replay import build_ref, score

MODEL = "models/Laguna-XS-2.1-pruned-50"
PACK = "models/Laguna-XS-2.1-pruned-50-packed"
HELDOUT = sorted(glob.glob("traces/django__*.jsonl"))
REFS = Path("refs/kv")
OUT = Path("results/kv.json")
CAP = 131072  # above the longest captured session (~78k); build_ref's default would truncate
BUDGET_GB = 12.0  # 75% of 16 GB: stand-in for a 16 GB Mac's recommended working set
MIN_TOP1, MAX_KL = 0.98, 0.02
# Plain cache peaks at 14.6 GB at 82k tokens. A row above this means the attention path is
# allocating something it shouldn't; stop before the longer sessions (sessions run shortest first).
PEAK_ABORT_GB = 16.0


def build_refs(model, tok, traces, refs_dir, cap=CAP):
    """Full-length references from `model` with the plain cache. Returns ref paths, longest last."""
    refs_dir = Path(refs_dir)
    refs_dir.mkdir(parents=True, exist_ok=True)
    refs = []
    for f in traces:
        r = refs_dir / f"{Path(f).stem}.npz"
        if not r.exists():
            build_ref(model, tok, f, r, max_tokens=cap, n_actions=0)
            print("ref", r.stem, flush=True)
        if bool(np.load(r)["truncated"]):
            r.unlink()
            raise ValueError(f"{r.stem}: truncated at cap={cap}; raise the cap")
        refs.append(r)
    return sorted(refs, key=lambda r: len(np.load(r)["ids"]))


def decode_at_ctx(model, ids, make_cache, n=64, chunk=2048):
    """Prefill `ids`, then greedy-decode n tokens: KV size, decode speed and peak at full context."""
    cache = make_cache()
    mx.reset_peak_memory()
    for a in range(0, len(ids), chunk):
        h = model.model(mx.array(ids[None, a : a + chunk]), cache=cache)
        mx.eval(h)
    kv = sum(c.nbytes for c, lt in zip(cache, model.args.layer_types) if lt == "full_attention")
    tok = mx.argmax(model.lm_head(h[:, -1]), axis=-1)
    mx.eval(tok)
    t0 = time.perf_counter()
    for _ in range(n):
        tok = mx.argmax(model.lm_head(model.model(tok[None], cache=cache)[:, -1]), axis=-1)
        mx.eval(tok)
    return {"kv_gb": kv / 1e9, "decode_tok_s": n / (time.perf_counter() - t0),
            "decode_peak_gpu_gb": mx.get_peak_memory() / 1e9}


def summarize(res):
    out = {}
    for key in sorted({k.rsplit("/", 1)[0] for k in res["rows"]}):
        rows = [v for k, v in res["rows"].items() if k.rsplit("/", 1)[0] == key]
        w = np.array([r["assistant_tokens"] for r in rows])
        s = {"sessions": len(rows),
             "kl_mean": float(np.average([r["kl_top64_mean"] for r in rows], weights=w)),
             "kl_p99_max": max(r["kl_top64_p99"] for r in rows),
             "top1": float(np.average([r["top1_agree"] for r in rows], weights=w)),
             "prefill_tok_s": float(np.mean([r["tok_per_s"] for r in rows])),
             "peak_gpu_gb": max(max(r["peak_mem_gb"], r.get("decode_peak_gpu_gb", 0)) for r in rows)}
        for r in rows:
            if "kv_gb" in r:
                s.update(kv_gb=r["kv_gb"], decode_tok_s=r["decode_tok_s"], longest_tokens=r["tokens"])
        s["fits"] = s["peak_gpu_gb"] <= BUDGET_GB
        s["passes"] = bool(s["fits"] and s["top1"] >= MIN_TOP1 and s["kl_mean"] <= MAX_KL)
        out[key] = s
    return out


def sweep(model, target, settings, refs, out=OUT, decode_n=64):
    """One row per (setting, session); `refs` is ordered longest last. Saves after every row."""
    out = Path(out)
    res = json.loads(out.read_text()) if out.exists() else {"rows": {}}
    res.setdefault("budget_gb", BUDGET_GB)
    for bits in settings:
        mk = model.make_cache if bits is None else (lambda b=bits: make_quantized_cache(model, b))
        for r in refs:
            key = f"{target}/{bits or 'plain'}/{r.stem}"
            if key in res["rows"]:
                continue
            row = score(model, r, make_cache=mk)
            if r == refs[-1]:
                row.update(decode_at_ctx(model, np.load(r)["ids"], mk, n=decode_n))
            res["rows"][key] = row
            res["summary"] = summarize(res)
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(res, indent=1))
            tmp.replace(out)  # atomic: an interrupt never leaves a half-written file
            print(key, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
            peak = max(row["peak_mem_gb"], row.get("decode_peak_gpu_gb", 0))
            if peak > PEAK_ABORT_GB:
                raise SystemExit(f"{key}: peak {peak:.2f} GB > {PEAK_ABORT_GB} GB; stopping before longer sessions")
    return res


def main():
    from mlx_lm import load

    from laguna_mlx.streaming import load_streaming

    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["resident", "streaming"], default="resident")
    p.add_argument("--slots", type=int, default=32)
    a = p.parse_args()
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    if a.target == "resident":
        model, tok = load(MODEL)
        refs = build_refs(model, tok, HELDOUT, REFS)
        res = sweep(model, "resident", (None, 8, 4), refs)
    else:
        refs = sorted(REFS.glob("*.npz"), key=lambda r: len(np.load(r)["ids"]))
        if not refs:
            raise SystemExit("run the resident target first: it builds refs/kv")
        model, _ = load_streaming(PACK, a.slots)
        res = sweep(model, "streaming", (None,), refs[-1:])
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
