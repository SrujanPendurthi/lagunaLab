"""Prompt-lookup speculative decoding on real weights (Task S4, overnight).

Times `generate_lookup` (laguna_mlx.dflash) against a *fair* greedy baseline on
both the resident and streaming (slots=32) targets, over (i) the 10
`check_parity.PROMPTS` and (ii) 5 real session prefixes -- the last 8,192
tokens before the start of an assistant turn in a captured pool session,
since long agent contexts are where copying a prior span pays off most.

Fair baseline (controller ruling, 2026-09-26): `generate_lookup` prefills with
last-position logits only, evaluated per chunk, so its prefill is cheaper than
`generate_greedy`'s single all-position forward. Comparing lookup's decode
speed against unmodified `generate_greedy` would credit it for a prefill
optimization it happens to share with a fair baseline, not with drafting
itself. `greedy_fast` below prefills exactly like `generate_lookup` (same
chunking, same last-position-only logits, same `make_trimmable_cache`) and
then decodes one token at a time -- see test_greedy_fast_matches_generate_greedy
in tests/test_bench_lookup.py for the CPU/tiny-model equivalence check.

The headline number is DECODE tok/s, lookup vs greedy_fast; end-to-end
(prefill+decode) tok/s is also reported. Settings swept: g=4/n=7 fixed,
g=4/n=15 adaptive (S2's best), and g=3/n=7 fixed as a sensitivity check.

python3 -m scripts.bench_lookup [--target resident|streaming|both]
                                 [--max-tokens 256] [--slots 32]
                                 [--max-swap-gb 10] [--skip-existing]

Writes results/lookup.json incrementally (one target's key at a time, via an
atomic tmp-file-then-replace write) so a crash never loses a finished target,
or even a finished workload (prompts vs sessions) within a target. Never
holds both targets resident at once: each target is loaded, run, then
deleted + mx.clear_cache()'d before the next. Swap is checked after every
workload item and fails closed (an unparseable `sysctl` reading raises,
rather than looking like "no swap pressure"); crossing --max-swap-gb stops
and records rather than crashing or silently truncating.

Nothing above the `if __name__ == "__main__":` guard loads a model, a
tokenizer, or a trace file -- `python3 -c "import scripts.bench_lookup"` is
safe to run while the real model is in use elsewhere.
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import mlx.core as mx

import laguna_mlx  # noqa: F401
from laguna_mlx.cache import make_trimmable_cache
from laguna_mlx.dflash import generate_greedy, generate_lookup
from replay.session import load_session, tokenize
from scripts.check_parity import PROMPTS
from scripts.diag_dflash import first_divergence
from scripts.lookup_acceptance import spans

OUT = Path("results/lookup.json")
MAX_TOKENS = 256
MAX_SWAP_GB = 10.0
PREFILL_STEP = 2048
PREFIX_TOKENS = 8192
TOK_DIR = "models/Laguna-XS-2.1"
RESIDENT_DIR = "models/Laguna-XS-2.1-mlx-4bit"
PACK_DIR = "models/Laguna-XS-2.1-packed"
# g=4/n=7 fixed and g=4/n=15 adaptive (S2's best offline setting), plus g=3/n=7
# fixed as a sensitivity check (brief, controller additions 2026-09-26).
SETTINGS = [
    {"g": 4, "n_draft": 7, "adaptive": False},
    {"g": 4, "n_draft": 15, "adaptive": True},
    {"g": 3, "n_draft": 7, "adaptive": False},
]
# 5 sessions, mixing astropy/django, fixed so the run is reproducible.
SESSION_FILES = [
    "traces/astropy__astropy-12907.jsonl",
    "traces/astropy__astropy-14182.jsonl",
    "traces/astropy__astropy-14365.jsonl",
    "traces/django__django-10914.jsonl",
    "traces/django__django-10924.jsonl",
]

NOTES = [
    "greedy_fast, not raw generate_greedy, is the baseline: generate_lookup's prefill computes "
    "last-position logits only, evaluated per chunk, which is already cheaper than "
    "generate_greedy's single all-position forward. greedy_fast prefills exactly like "
    "generate_lookup (same chunking, same last-position-only logits, same "
    "make_trimmable_cache) so the comparison isolates drafting, not the prefill scheme "
    "(controller ruling, 2026-09-26).",
    "Each row's prefill_s is measured once (from the item's greedy_fast baseline) and reused "
    "as the prefill estimate for all 3 lookup settings on that item, rather than re-timing a "
    "throwaway prefill per setting: generate_lookup's prefill is architecturally identical "
    "(same chunking/last-position-logits/cache type), and its own make_trimmable_cache slack "
    "differs by at most 16 tokens of retained cache buffer -- attention itself is still masked "
    "to the model's sliding_window regardless of slack, so this is a negligible compute delta, "
    "not a masked one. decode_s = (end-to-end wall time of the full generate_lookup call) - "
    "prefill_s; on a very fast, very short generation this can occasionally go slightly negative "
    "from timing noise, in which case decode_s is floored rather than reported as infinite tok/s.",
    "identical (vs greedy_fast, not generate_greedy) < 1 is hypothesized to come from bf16 exact "
    "logit near-ties broken differently between greedy_fast's single-token decode steps and "
    "generate_lookup's batched-verify steps (same hypothesis as results/dflash.json); "
    "first_divergence records where, in tokens.",
    "Session prefixes are the last up-to-8192 tokens of the session's tokenized transcript, "
    "ending right before the start of the *last* assistant turn (replay.session tokenize's "
    "assistant mask) -- the context an agent had just before deciding what to write next, "
    "which is where prompt-lookup drafting is expected to pay off most.",
    "Timings may include noise from concurrent human activity on the machine.",
    "Single run per setting; no repeats.",
]


# --------------------------------------------------------------------------
# Pure helpers -- unit-tested in tests/test_bench_lookup.py, no model/no I/O.
# --------------------------------------------------------------------------

def _lookup_prefill(target, prompt, prefill_step=PREFILL_STEP, slack=64):
    """Chunked prefill identical to generate_lookup's own: last-position logits
    only, mx.eval'd per chunk so the compute graph never spans the whole
    prompt. Copied here (not imported) per the brief: don't modify laguna_mlx."""
    prompt = [int(t) for t in prompt]  # accept numpy ints
    cache = make_trimmable_cache(target, slack=slack)
    logits = None
    for a in range(0, len(prompt), prefill_step):
        h = target.model(mx.array([prompt[a : a + prefill_step]]), cache=cache)
        logits = target.lm_head(h[0, -1])
        mx.eval(logits)
    return cache, logits


def _decode_greedy(target, cache, logits, max_tokens, eos=()):
    out = []
    while True:
        t = int(mx.argmax(logits))
        out.append(t)
        if t in eos or len(out) >= max_tokens:
            return out
        logits = target(mx.array([[t]]), cache=cache)[0, -1]


def greedy_fast(target, prompt, max_tokens, eos=(), prefill_step=PREFILL_STEP):
    """generate_greedy, but prefills exactly like generate_lookup (chunked,
    last-position logits only, eval per chunk, make_trimmable_cache) instead
    of one all-position forward, then decodes one token at a time. Output
    equals generate_greedy -- see test_greedy_fast_matches_generate_greedy."""
    cache, logits = _lookup_prefill(target, prompt, prefill_step)
    return _decode_greedy(target, cache, logits, max_tokens, eos)


def session_prefix(ids, mask, max_tokens=PREFIX_TOKENS):
    """Up to the last `max_tokens` tokens of `ids`, ending right before the
    start of the last assistant turn in `mask` (a bool array from
    replay.session.tokenize). Raises if `mask` has no assistant turn."""
    turns = spans(mask)
    if not turns:
        raise ValueError("no assistant turn in this session")
    start = turns[-1][0]
    a = max(0, start - max_tokens)
    return ids[a:start]


def _aggregate(rows):
    """Per-setting means across a workload's items."""
    out = []
    for i, s in enumerate(SETTINGS):
        vals = [r["settings"][i] for r in rows if len(r["settings"]) > i]
        if not vals:
            continue
        out.append({
            **s, "n_items": len(vals),
            "mean_speedup_decode": sum(v["speedup_decode"] for v in vals) / len(vals),
            "mean_speedup_e2e": sum(v["speedup_e2e"] for v in vals) / len(vals),
            "mean_accepted": sum(v["mean_accepted"] for v in vals) / len(vals),
            "identical_frac": sum(v["identical"] for v in vals) / len(vals),
        })
    return out


def _summarize(res):
    """Per (target, setting) mean session-workload decode speedup, and the go/no-go verdict:
    keep prompt-lookup if the best setting's mean speedup on (ii) session prefixes is >= 1.1x."""
    rows = []
    for name in ("resident", "streaming"):
        t = res.get(name)
        sessions = t.get("sessions") if t else None
        if not sessions or not sessions.get("aggregate"):
            continue
        for agg in sessions["aggregate"]:
            rows.append({"target": name, **agg})
    best = max(rows, key=lambda r: r["mean_speedup_decode"], default=None)
    return {"by_target_setting": rows, "best": best,
            "keep": bool(best and best["mean_speedup_decode"] >= 1.1)}


def _atomic_write_json(path, data):
    """Write `data` as JSON to `path` without ever leaving a half-written
    file there: write to a sibling .tmp file, then atomically replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def _load_results():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def _save_target(name, result):
    """Merge `result` into targets[name] of the on-disk results file, refresh
    the cross-target summary, and atomically replace it. Safe to call mid-run
    (e.g. right after the prompts workload, before sessions starts) so a
    later crash never loses an already-finished workload."""
    res = _load_results()
    res[name] = result
    res["settings"] = SETTINGS
    res["notes"] = NOTES
    res["summary"] = _summarize(res)
    _atomic_write_json(OUT, res)
    return res


def _target_complete(name):
    """True if targets[name] already ran to completion (both workloads, no stop)."""
    t = _load_results().get(name)
    return bool(t) and t.get("stopped") is None and "prompts" in t and "sessions" in t


# --------------------------------------------------------------------------
# Overnight-only: swap guard, model I/O, timing. Nothing here runs on import.
# --------------------------------------------------------------------------

def swap_gb():
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    if not m:  # fail closed
        raise RuntimeError(f"could not parse `sysctl vm.swapusage` output: {out!r}")
    return float(m.group(1)) / 1024


def _load_target(name, slots):
    if name == "resident":
        from mlx_lm import load
        return load(RESIDENT_DIR)
    from laguna_mlx.streaming import load_streaming
    return load_streaming(PACK_DIR, slots)


def _prompt_items(tok):
    return [{"name": f"p{i}", "ids": tok.apply_chat_template(
                [{"role": "user", "content": q}], add_generation_prompt=True, enable_thinking=False)}
            for i, q in enumerate(PROMPTS)]


def _session_items():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    items = []
    for f in SESSION_FILES:
        msgs, tools = load_session(f)
        ids, mask, _truncated = tokenize(tok, msgs, tools)
        prefix = session_prefix(ids, mask, PREFIX_TOKENS)
        items.append({"name": Path(f).stem, "ids": [int(t) for t in prefix]})
    return items


def _run_item(target, ids, max_tokens, eos):
    t0 = time.perf_counter()
    cache, logits = _lookup_prefill(target, ids)
    prefill_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    base_out = _decode_greedy(target, cache, logits, max_tokens, eos)
    decode_s = time.perf_counter() - t1
    base = {"tokens": len(base_out), "prefill_s": prefill_s, "decode_s": decode_s,
            "decode_tok_s": len(base_out) / decode_s,
            "e2e_tok_s": len(base_out) / (prefill_s + decode_s)}

    rows = []
    for s in SETTINGS:
        t2 = time.perf_counter()
        out, stats = generate_lookup(target, ids, max_tokens, eos, g=s["g"], n_draft=s["n_draft"],
                                      adaptive=s["adaptive"], prefill_step=PREFILL_STEP)
        total_s = time.perf_counter() - t2
        decode_only_s = max(total_s - prefill_s, 1e-9)  # see NOTES: prefill_s is shared/estimated
        identical = out == base_out
        rows.append({
            **s, "tokens": len(out), "total_s": total_s, "decode_s": decode_only_s,
            "decode_tok_s": len(out) / decode_only_s,
            "e2e_tok_s": len(out) / (prefill_s + decode_only_s),
            "speedup_decode": (len(out) / decode_only_s) / base["decode_tok_s"],
            "speedup_e2e": (len(out) / (prefill_s + decode_only_s)) / base["e2e_tok_s"],
            "mean_accepted": stats["mean_accepted"], "steps": stats["steps"],
            "drafted": stats["drafted"], "accepted": stats["accepted"],
            "steps_with_draft": stats["steps_with_draft"],
            "identical": identical,
            "first_divergence": None if identical else first_divergence(out, base_out),
        })
    return base, rows


def _run_workload(target, items, max_tokens, eos, swap, over, label):
    rows = []
    for it in items:
        base, setting_rows = _run_item(target, it["ids"], max_tokens, eos)
        rows.append({"name": it["name"], "prompt_tokens": len(it["ids"]),
                     "baseline": base, "settings": setting_rows})
        swap.append((f"{label}:{it['name']}", swap_gb()))
        if over(f"{label}:{it['name']}"):
            break
    return {"rows": rows, "aggregate": _aggregate(rows)}


def _run_target(name, args):
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    swap = [("start", swap_gb())]
    result = {"target": name, "slots": args.slots if name == "streaming" else None,
              "max_tokens": args.max_tokens, "swap_gb": swap, "stopped": None}

    def over(stage):
        if swap[-1][1] > args.max_swap_gb:
            result["stopped"] = f"swap {swap[-1][1]:.1f} GB > {args.max_swap_gb} GB ({stage})"
            return True
        return False

    if over("start"):
        return result
    target, tok = _load_target(name, args.slots)
    eos = tuple(tok.eos_token_ids)
    prompt_items = _prompt_items(tok)
    generate_greedy(target, prompt_items[0]["ids"], 8, eos)  # warm-up
    swap.append(("loaded", swap_gb()))
    if over("loaded"):
        del target
        mx.clear_cache()
        return result

    result["prompts"] = _run_workload(target, prompt_items, args.max_tokens, eos, swap, over, "prompts")
    result["peak_gpu_gb"] = mx.get_peak_memory() / 1e9
    _save_target(name, result)  # persist the prompts workload before sessions can crash it away
    if result["stopped"]:
        del target
        mx.clear_cache()
        return result

    session_items = _session_items()
    result["sessions"] = _run_workload(target, session_items, args.max_tokens, eos, swap, over, "sessions")
    result["peak_gpu_gb"] = mx.get_peak_memory() / 1e9
    del target
    mx.clear_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["resident", "streaming", "both"], default="both")
    p.add_argument("--slots", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--max-swap-gb", type=float, default=MAX_SWAP_GB)
    p.add_argument("--skip-existing", action="store_true",
                    help="skip a target already fully present (both workloads, no stop) in "
                         "results/lookup.json")
    args = p.parse_args()
    targets = ["resident", "streaming"] if args.target == "both" else [args.target]
    for name in targets:
        if args.skip_existing and _target_complete(name):
            print(f"=== {name} === skip (already complete)", flush=True)
            continue
        print(f"=== {name} ===", flush=True)
        result = _run_target(name, args)
        res = _save_target(name, result)
        print(f"wrote {OUT} ({name})", flush=True)
        print("summary so far:", res["summary"], flush=True)


if __name__ == "__main__":
    main()
