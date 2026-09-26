"""Decode/prefill speed, hit rate and memory across expert-slot budgets.

python3 -m scripts.bench_stream [--slots 16 32 64 128] [--resident]

Controller rulings (Task 18) vs. the original brief:
  - `--resident` (full 18 GB model) runs LAST, after all streaming budgets,
    and is skipped with a note if swap is already > ~6 GB at that point.
  - identical_frac is therefore computed after the fact: each streaming
    budget's outputs are kept and compared to the resident reference once
    it's available.
  - swap is sampled after every run; if it exceeds ~12 GB we stop early.
  - the best 2 streaming budgets (by short-prompt decode_tok_s) get one
    extra measurement: decode tok/s after a ~16k-token prefill built from
    repeated real text (tests/reference/modeling_laguna.py).
"""
import argparse
import json
import re
import resource
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import laguna_mlx  # noqa: F401
from laguna_mlx.dflash import generate_greedy
from laguna_mlx.streaming import load_streaming, stream_stats
from scripts.check_parity import PROMPTS

p = argparse.ArgumentParser()
p.add_argument("--slots", type=int, nargs="+", default=[16, 32, 64, 128])
p.add_argument("--resident", action="store_true", help="also run the fully resident model")
p.add_argument("--max-tokens", type=int, default=128)
a = p.parse_args()
PACK = "models/Laguna-XS-2.1-packed"
REF_TEXT = Path("tests/reference/modeling_laguna.py").read_text()


def swap_gb():
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    if not m:  # fail closed: an unparsed reading must not look like "no swap pressure"
        raise RuntimeError(f"could not parse `sysctl vm.swapusage` output: {out!r}")
    return float(m.group(1)) / 1024


def run(model, tok, label):
    prompts = [tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                       enable_thinking=False) for q in PROMPTS[:3]]
    mx.reset_peak_memory()
    outs, t0 = [], time.perf_counter()
    for pr in prompts:
        outs.append(generate_greedy(model, pr, a.max_tokens, tuple(tok.eos_token_ids)))
    decode = sum(map(len, outs)) / (time.perf_counter() - t0)
    ids = np.random.default_rng(0).integers(1000, 50000, size=4096).tolist()
    cache = model.make_cache()
    t0 = time.perf_counter()
    for i in range(0, len(ids), 1024):
        mx.eval(model(mx.array([ids[i : i + 1024]]), cache=cache))
    prefill = len(ids) / (time.perf_counter() - t0)
    row = {"label": label, "decode_tok_s": decode, "prefill_tok_s": prefill,
           "peak_gpu_gb": mx.get_peak_memory() / 1e9,
           "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
           "swap_gb_after": swap_gb()}
    print(row, flush=True)
    return row, outs


def long_context_decode(model, tok, ctx_tokens=16000, gen_tokens=64):
    """Ruling 3: decode tok/s after a realistic ~16k-token prefill."""
    ids = tok.encode(REF_TEXT)
    reps = -(-ctx_tokens // max(1, len(ids))) + 1
    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": REF_TEXT * reps}],
        add_generation_prompt=True, enable_thinking=False)[:ctx_tokens]
    cache = model.make_cache()
    t0 = time.perf_counter()
    logits = model(mx.array([prompt_ids]), cache=cache)[0, -1]
    mx.eval(logits)
    prefill_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    tkn = int(mx.argmax(logits))
    for _ in range(gen_tokens):
        logits = model(mx.array([[tkn]]), cache=cache)[0, -1]
        mx.eval(logits)
        tkn = int(mx.argmax(logits))
    decode_s = time.perf_counter() - t0
    return {"prefill_16k_tok_s": len(prompt_ids) / prefill_s, "decode_16k_tok_s": gen_tokens / decode_s,
            "ctx_tokens": len(prompt_ids)}


rows, outs_by_label = [], {}
notes = [
    "identical_frac < 1 is hypothesized to come from bf16 exact logit ties broken differently by "
    "streaming's vs. resident's floating-point summation order across expert contributions "
    "(per-forward-pass logits still agree to atol/rtol=1e-5); pending direct verification.",
    "max_rss_gb (resource.getrusage) is unreliable on Apple Silicon unified memory -- it does not "
    "track MLX/Metal GPU buffer allocation. Use peak_gpu_gb for memory-safety judgments instead.",
    "hit_rate/bytes_read on each row come from that row's short-prompt run only; the 16k-context "
    "measurement (prefill_16k_tok_s/decode_16k_tok_s) reloads a fresh model instance with its own "
    "empty expert-slot cache, so its hits/misses are not folded into the row's hit_rate/bytes_read.",
]
first_streaming = True

for s in a.slots:
    sg = swap_gb()
    if sg > 12:
        notes.append(f"stopped early before slots={s}: swap at {sg:.1f} GB > 12 GB budget")
        print(notes[-1], flush=True)
        break
    m, tok = load_streaming(PACK, s)
    if first_streaming:
        notes.append("first streaming run (slots=%d): packing just wrote the 18 GB pack dir "
                      "seconds earlier, so macOS page cache was very likely already warm for it; "
                      "no cache purge was performed (would need sudo)." % s)
        first_streaming = False
    row, outs = run(m, tok, f"slots={s}")
    row.update(slots=s, **stream_stats(m))
    rows.append(row)
    outs_by_label[row["label"]] = outs
    del m
    mx.clear_cache()

# Extra measurement (ruling 3): best 2 streaming budgets by short-prompt decode speed.
best = sorted(rows, key=lambda r: r["decode_tok_s"], reverse=True)[:2]
for row in best:
    sg = swap_gb()
    if sg > 12:
        notes.append(f"skipped 16k-context measurement for {row['label']}: swap {sg:.1f} GB > 12 GB")
        continue
    m, tok = load_streaming(PACK, row["slots"])
    row.update(long_context_decode(m, tok))
    print({"label": row["label"], **{k: row[k] for k in ("decode_16k_tok_s", "prefill_16k_tok_s", "ctx_tokens")}},
          flush=True)
    del m
    mx.clear_cache()

ref = None
if a.resident:
    sg = swap_gb()
    if sg > 6:
        notes.append(f"skipped --resident: swap already {sg:.1f} GB > 6 GB budget before the 18 GB load")
        print(notes[-1], flush=True)
    else:
        from mlx_lm import load
        m, tok = load("models/Laguna-XS-2.1-mlx-4bit")
        row, ref = run(m, tok, "resident")
        rows.append(row)
        del m
        mx.clear_cache()

if ref is not None:
    for row in rows:
        if row["label"] in outs_by_label:
            row["identical_frac"] = float(np.mean(
                [x == y for x, y in zip(outs_by_label[row["label"]], ref)]))

Path("results/streaming.json").write_text(json.dumps({"rows": rows, "notes": notes}, indent=1))
