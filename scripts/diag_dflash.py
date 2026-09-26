"""DFlash equivalence diagnostic + verify-cost timing (Task S1, extended).

Part A: for each target (resident/streaming) x PROMPTS[:3], compares
generate_greedy (plain cache), the same greedy loop on a trimmable cache, and
generate_dflash. Records the first divergence per pair and, at each
divergence, re-runs both paths' own execution style (single-token decode vs
an (n_draft+1)-token verify batch) over prompt+prefix to capture the top-2
logit gap there -- this is the "bf16 near-tie" hypothesis for why DFlash's
lossless decoding isn't always bit-identical to greedy.

Part B: forward latency vs verify block length L, and draft.propose latency
vs n_draft, timed on an (assumed) idle machine, with a linear fit
ms ~= a + b*(L-1) over the verify-length medians.

python3 -m scripts.diag_dflash [--target resident|streaming|both]
Writes results/dflash_diag.json (merged in, one target's key at a time).
Never holds both targets resident at once: each target is loaded, run, then
deleted + mx.clear_cache()'d before the next.
"""
import argparse
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from laguna_mlx.cache import make_trimmable_cache
from laguna_mlx.dflash import _target_step, generate_dflash, generate_greedy, load_draft
from scripts.check_parity import PROMPTS

OUT = Path("results/dflash_diag.json")
DRAFT = "models/Laguna-XS-2.1-DFlash"
N_DRAFT = 3
MAX_TOKENS = 256
PREFIX_LEN = 2048
BLOCK_LENGTHS = [1, 2, 3, 4, 6, 8, 12, 16]
PROPOSE_N = [3, 7, 15]
WARMUP, REPEATS = 3, 10
MAX_SWAP_GB = 10.0

NOTES = [
    "Timings (verify_ms, propose_ms) assume the machine is otherwise idle (run overnight); "
    "if something else was running on the machine they may be noisy.",
    "bf16 ulp gap: round both values to bfloat16, reinterpret the 16-bit pattern as an "
    "order-preserving unsigned integer (the standard radix-sort-floats sign-flip trick), and "
    "take the absolute difference of those integers. gap(1.0, next bf16 above 1.0) == 1; "
    "gap(x, x) == 0.",
    "Equivalence recompute at a divergence: for a divergence at the very first generated token "
    "(index 0), all three paths' first token came from a single one-shot forward over the whole "
    "prompt, so the recompute uses that same one-shot batch for every path. For a later "
    "divergence, each path's own steady-state batching is replayed instead: a single-token decode "
    "step (batch size 1) for std/trim, an (n_draft+1)-token block for dflash. Both sides use the "
    "same target weights and the same token history, so any gap tests whether batched-vs-single "
    "matmul reduction order changes the result, not a model difference.",
    "Single run per target; no repeats for Part A (equivalence is exact-match, not a noisy metric).",
]


# --------------------------------------------------------------------------
# Pure helpers -- unit-tested in tests/test_diag_dflash.py, no model/no I/O.
# --------------------------------------------------------------------------

def first_divergence(a, b):
    """First index where token sequences a, b differ. If one is a prefix of
    the other, they diverge where the shorter one ends. None if identical."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _bf16_order_key(bits):
    """Map a bf16 bit pattern (uint16, same layout as the top 16 bits of
    float32) to a monotonic integer: flip all bits if negative, else set the
    sign bit -- the usual trick for radix-sorting floats."""
    bits = int(bits)
    return (~bits) & 0xFFFF if bits & 0x8000 else bits | 0x8000


def bf16_ulp_gap(x, y):
    """Gap between two float values, each rounded to bf16, in bf16 ulps."""
    bits = np.array(mx.array([float(x), float(y)], dtype=mx.bfloat16).view(mx.uint16))
    return abs(_bf16_order_key(bits[0]) - _bf16_order_key(bits[1]))


def linear_fit(xs, ys):
    """Least-squares ms ~= a + b*x."""
    b, a = np.polyfit(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), 1)
    return {"a": float(a), "b": float(b)}


def median_iqr(vals):
    vals = list(vals)
    q1, q3 = np.percentile(vals, [25, 75])
    return {"median": float(statistics.median(vals)), "iqr": float(q3 - q1)}


# --------------------------------------------------------------------------
# Overnight-only: swap guard, model I/O, timing. Nothing here runs on import.
# --------------------------------------------------------------------------

def swap_gb():
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    if not m:  # fail closed
        raise RuntimeError(f"could not parse `sysctl vm.swapusage` output: {out!r}")
    return float(m.group(1)) / 1024


def _timed_repeats(fn, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return median_iqr(times)


def _greedy_trim(target, prompt, max_tokens, eos):
    """generate_greedy's loop, copied here with a trimmable cache instead
    (per the brief: don't modify laguna_mlx)."""
    prompt = [int(t) for t in prompt]
    cache = make_trimmable_cache(target)
    logits = target(mx.array([prompt]), cache=cache)[0, -1]
    out = []
    while True:
        t = int(mx.argmax(logits))
        out.append(t)
        if t in eos or len(out) >= max_tokens:
            return out
        logits = target(mx.array([[t]]), cache=cache)[0, -1]


def _recompute_logits(target, prefix_ids, make_cache, block):
    """Replay prefix_ids through a fresh cache, forwarding the last `block`
    tokens together in one call (and everything before that as a single
    prefill) -- reproduces a path's own batching granularity at its last
    step, so its logits can be compared against another path's."""
    cache = make_cache()
    a = max(0, len(prefix_ids) - block)
    if a:
        target(mx.array([prefix_ids[:a]]), cache=cache)
    return target(mx.array([prefix_ids[a:]]), cache=cache)[0, -1]


def _top2_gap(logits_row):
    order = mx.argsort(logits_row)
    top2 = [int(i) for i in order[-2:].tolist()][::-1]  # [best, second]
    v0, v1 = float(logits_row[top2[0]]), float(logits_row[top2[1]])
    return {"top2_ids": top2, "gap_f32": v0 - v1, "gap_bf16_ulps": bf16_ulp_gap(v0, v1)}


def _equivalence_for_prompt(target, draft, prompt_ids, eos, n_draft=N_DRAFT, max_tokens=MAX_TOKENS):
    outs = {
        "std": generate_greedy(target, prompt_ids, max_tokens, eos),
        "trim": _greedy_trim(target, prompt_ids, max_tokens, eos),
        "dflash": generate_dflash(target, draft, prompt_ids, max_tokens, eos, n_draft=n_draft)[0],
    }
    specs = {
        "std": (target.make_cache, 1),
        "trim": (lambda: make_trimmable_cache(target), 1),
        "dflash": (lambda: make_trimmable_cache(target), n_draft + 1),
    }
    pairs, verdicts = {}, []
    for name_a, name_b in (("std", "trim"), ("trim", "dflash"), ("std", "dflash")):
        idx = first_divergence(outs[name_a], outs[name_b])
        if idx is None:
            pairs[f"{name_a}_vs_{name_b}"] = {"first_divergence": None}
            continue
        prefix = list(prompt_ids) + list(outs[name_a][:idx])
        rows = {}
        for name in (name_a, name_b):
            make_cache, block = specs[name]
            # at the very first token, every path's own first step was one
            # batched forward over the whole prompt -- reuse that batch size
            block = len(prefix) if idx == 0 else block
            rows[name] = _top2_gap(_recompute_logits(target, prefix, make_cache, block))
        gap = min(rows[name_a]["gap_bf16_ulps"], rows[name_b]["gap_bf16_ulps"])
        verdict = "tie" if gap <= 2 else "non-tie"
        verdicts.append(verdict)
        pairs[f"{name_a}_vs_{name_b}"] = {"first_divergence": idx, "logits": rows, "verdict": verdict}
    overall = None if not verdicts else ("tie" if all(v == "tie" for v in verdicts) else "non-tie")
    return {"pairs": pairs, "verdict": overall}


def _part_a(target, draft, tok):
    eos = tuple(tok.eos_token_ids)
    prompts = [tok.apply_chat_template([{"role": "user", "content": q}],
                                        add_generation_prompt=True, enable_thinking=False)
               for q in PROMPTS[:3]]
    return [dict(prompt_index=i, **_equivalence_for_prompt(target, draft, p, eos))
            for i, p in enumerate(prompts)]


def _prefix_2k(tok):
    """A ~2k-token chat-templated prompt: repeat a real prompt until the
    templated length reaches PREFIX_LEN, then trim to exactly that."""
    msg, text = PROMPTS[0], PROMPTS[0]
    ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                   add_generation_prompt=True, enable_thinking=False)
    while len(ids) < PREFIX_LEN:
        text += " " + msg
        ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                       add_generation_prompt=True, enable_thinking=False)
    return ids[:PREFIX_LEN]


def _verify_timing(target, prefix_ids):
    cache = make_trimmable_cache(target)
    mx.eval(target(mx.array([prefix_ids]), cache=cache))  # prefill once
    out = {}
    for L in BLOCK_LENGTHS:
        block = mx.array([prefix_ids[:L]])

        def step(block=block, L=L):
            mx.eval(target(block, cache=cache))
            for c in cache:
                c.trim(L)

        out[str(L)] = _timed_repeats(step)
    return out


def _propose_timing(target, draft, prefix_ids):
    tcache, dcache = make_trimmable_cache(target), draft.make_cache()
    logits = None
    for a in range(0, len(prefix_ids), 2048):
        logits, aux = _target_step(target, prefix_ids[a : a + 2048], tcache, draft.target_layer_ids)
        draft.add_context(draft.context_states(aux), dcache)
    token = int(mx.argmax(logits[-1]))
    return {str(n): _timed_repeats(lambda n=n: draft.propose(target, token, dcache, n))
            for n in PROPOSE_N}


def _part_b(target, draft, tok):
    prefix_ids = _prefix_2k(tok)
    verify_ms = _verify_timing(target, prefix_ids)
    xs = [L - 1 for L in BLOCK_LENGTHS]
    ys = [verify_ms[str(L)]["median"] for L in BLOCK_LENGTHS]
    return {"verify_ms": verify_ms, "fit": linear_fit(xs, ys),
            "propose_ms": _propose_timing(target, draft, prefix_ids)}


def _load_target(name, slots=32):
    if name == "resident":
        from mlx_lm import load
        return load("models/Laguna-XS-2.1-mlx-4bit")
    from laguna_mlx.streaming import load_streaming
    return load_streaming("models/Laguna-XS-2.1-packed", slots)


def _run_target(name):
    """Stop-and-record (not crash) if swap crosses MAX_SWAP_GB at a checkpoint;
    swap_gb() itself still fails closed (raises) if `sysctl` is unparseable."""
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    swap = [("start", swap_gb())]
    result = {"swap_gb": swap, "stopped": None}

    def over(stage):
        if swap[-1][1] > MAX_SWAP_GB:
            result["stopped"] = f"swap {swap[-1][1]:.1f} GB > {MAX_SWAP_GB} GB ({stage})"
            return True
        return False

    if over("start"):
        return result
    target, tok = _load_target(name)
    eos = tuple(tok.eos_token_ids)
    warm = tok.apply_chat_template([{"role": "user", "content": PROMPTS[0]}],
                                    add_generation_prompt=True, enable_thinking=False)
    generate_greedy(target, warm, 8, eos)  # warm-up
    draft = load_draft(DRAFT)
    swap.append(("loaded", swap_gb()))
    if over("loaded"):
        del target, draft
        mx.clear_cache()
        return result
    result["equivalence"] = _part_a(target, draft, tok)
    swap.append(("after_part_a", swap_gb()))
    if over("after_part_a"):
        del target, draft
        mx.clear_cache()
        return result
    part_b = _part_b(target, draft, tok)
    swap.append(("after_part_b", swap_gb()))
    result.update({"verify_ms": part_b["verify_ms"], "fit": part_b["fit"],
                   "propose_ms": part_b["propose_ms"]})
    del target, draft
    mx.clear_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["resident", "streaming", "both"], default="both")
    args = p.parse_args()
    targets = ["resident", "streaming"] if args.target == "both" else [args.target]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res.setdefault("targets", {})
    res["notes"] = NOTES
    for name in targets:
        print(f"=== {name} ===", flush=True)
        res["targets"][name] = _run_target(name)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, indent=1))
        print(f"wrote {OUT} ({name})", flush=True)


if __name__ == "__main__":
    main()
