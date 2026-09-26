"""Greedy baseline vs DFlash on 10 coding prompts x draft sizes x draft precisions.

python3 -m scripts.bench_dflash --target resident|streaming [--max-tokens 256]

One target per process (never both in memory). Results for each target are
merged into results/dflash.json under their own key. Swap is checked between
runs; the sweep stops early above --max-swap-gb.
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import mlx.core as mx

import laguna_mlx  # noqa: F401
from laguna_mlx.dflash import generate_dflash, generate_greedy, load_draft
from scripts.check_parity import PROMPTS

p = argparse.ArgumentParser()
p.add_argument("--target", choices=["resident", "streaming"], default="resident")
p.add_argument("--max-tokens", type=int, default=256)
p.add_argument("--slots", type=int, default=32)
p.add_argument("--max-swap-gb", type=float, default=10.0)
a = p.parse_args()
OUT = Path("results/dflash.json")
DRAFT = "models/Laguna-XS-2.1-DFlash"

mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
if a.target == "resident":
    from mlx_lm import load
    target, tok = load("models/Laguna-XS-2.1-mlx-4bit")
    bits_list = (None, 8, 4)
else:
    from laguna_mlx.streaming import load_streaming
    target, tok = load_streaming("models/Laguna-XS-2.1-packed", a.slots)
    bits_list = (None,)
eos = tuple(tok.eos_token_ids)
prompts = [tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                   enable_thinking=False) for q in PROMPTS]


def swap_gb():
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    if not m:  # fail closed
        raise RuntimeError(f"could not parse `sysctl vm.swapusage` output: {out!r}")
    return float(m.group(1)) / 1024


def timed(fn):
    t0 = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t0


swaps = [("start", swap_gb())]
generate_greedy(target, prompts[0], 8)  # warm-up
base = []
for pr in prompts:
    out, dt = timed(lambda: generate_greedy(target, pr, a.max_tokens, eos))
    base.append({"tokens": out, "tok_s": len(out) / dt})
base_tps = sum(b["tok_s"] for b in base) / len(base)
swaps.append(("baseline", swap_gb()))
print("baseline tok/s", base_tps, "swap", swaps[-1], flush=True)

runs, stopped = [], None
for bits in bits_list:
    draft = load_draft(DRAFT, bits=bits)
    for n_draft in (3, 7, 15):
        if swaps[-1][1] > a.max_swap_gb:
            stopped = f"swap {swaps[-1][1]:.1f} GB > {a.max_swap_gb} GB"
            break
        rows = []
        for pr, b in zip(prompts, base):
            (out, st), dt = timed(lambda: generate_dflash(target, draft, pr, a.max_tokens, eos, n_draft))
            rows.append({"tok_s": len(out) / dt, "mean_accepted": st["mean_accepted"],
                         "identical": out == b["tokens"]})
        run = {"target": a.target, "draft_bits": bits or 16, "n_draft": n_draft,
               "tok_s": sum(r["tok_s"] for r in rows) / len(rows),
               "mean_accepted": sum(r["mean_accepted"] for r in rows) / len(rows),
               "identical_frac": sum(r["identical"] for r in rows) / len(rows),
               "per_prompt": rows}
        run["speedup"] = run["tok_s"] / base_tps
        runs.append(run)
        swaps.append((f"bits{bits or 16}_n{n_draft}", swap_gb()))
        print({k: v for k, v in run.items() if k != "per_prompt"}, "swap", swaps[-1], flush=True)
    del draft
    mx.clear_cache()
    if stopped:
        print("STOPPED:", stopped, flush=True)
        break

res = json.loads(OUT.read_text()) if OUT.exists() else {}
res[a.target] = {"target": a.target, "slots": a.slots if a.target == "streaming" else None,
                 "baseline_tok_s": base_tps, "baseline_per_prompt_tok_s": [b["tok_s"] for b in base],
                 "max_tokens": a.max_tokens, "peak_gpu_gb": mx.get_peak_memory() / 1e9,
                 "swap_gb": swaps, "stopped": stopped, "runs": runs}
OUT.write_text(json.dumps(res, indent=1))
