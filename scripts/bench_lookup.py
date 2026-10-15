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
(prefill+decode) tok/s is also reported. Settings are swept PER TARGET (see
SETTINGS below): resident sweeps g=4/n=3 fixed, g=4/n=3 adaptive (S2's best
per results/lookup_acceptance.json's measured-cost re-run), g=4/n=7 fixed as
a comparison point; streaming is restricted to short drafts (block <= 4
tokens) to stay below its verify-cost cliff (see NOTES / results/dflash_diag.json).

python3 -m scripts.bench_lookup [--target resident|streaming|both]
                                 [--max-tokens 256] [--slots 32]
                                 [--max-swap-gb 10] [--skip-existing]

Writes results/lookup.json incrementally: after every single workload item
(one PROMPTS entry or one session prefix), not just once per target or once
per workload, via an atomic tmp-file-then-replace write, so a crash loses at
most the one item in flight. Re-running (with or without --skip-existing)
resumes a target from whichever items are already on disk for it -- the
model is still loaded only once per target. Never holds both targets
resident at once: each target is loaded, run, then deleted + mx.clear_cache()'d
before the next. Swap is checked after every workload item and fails closed
(an unparseable `sysctl` reading raises, rather than looking like "no swap
pressure"); crossing --max-swap-gb stops and records rather than crashing or
silently truncating.

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
from laguna_mlx import wire_resident
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
# Settings are per target (single source of truth), not one shared list. resident sweeps
# g=4/n=3 fixed, g=4/n=3 adaptive, g=4/n=7 fixed (brief, controller additions 2026-09-26;
# updated 2026-09-26 follow-up): S2's re-run with S1's MEASURED verify costs
# (results/lookup_acceptance.json, commit 22f84f1, build_measured_cost_fn) predicts g=4/n=3
# fixed as resident's best expected speedup (1.483x), not g=4/n=7 (1.297x) -- the earlier
# provisional linear cost model had favored longer drafts. n=7 is kept as the comparison
# point (n=15 adaptive and g=3/n=7 are dropped: g=4/n=3 adaptive is the stronger second
# setting at 1.478x, per the same measured-cost sweep). streaming is restricted to short
# drafts -- g=4/n=3 fixed, g=4/n=3 adaptive, g=4/n=2 fixed -- all with verify-block length
# (1 + n_draft) <= 4 tokens, staying below the streaming verify-cost cliff measured in
# results/dflash_diag.json (S1 follow-up, 2026-09-26): streaming's median verify_ms was
# 37.8@block=1, 61.9@block=4, then 212.0@block=6 and 574.1@block=16 -- a cliff, not a curve,
# because a block of >4 newly-drafted tokens routes up to 8 experts/token through a 32-slot
# expert cache per layer, and once that exceeds 32 slots StreamingSwitchGLU needs multiple
# masked passes (laguna_mlx/streaming.py's "a prefill chunk needing > slots experts runs
# several masked passes" note) instead of one. Benchmarking streaming with n_draft=7/15
# would only measure that cliff, not prompt-lookup's drafting benefit -- and the same
# measured-cost sweep shows it: streaming g=4/n=7 fixed is 0.692x (slower than no drafting).
SETTINGS = {
    "resident": [
        {"g": 4, "n_draft": 3, "adaptive": False},
        {"g": 4, "n_draft": 3, "adaptive": True},
        {"g": 4, "n_draft": 7, "adaptive": False},
    ],
    "streaming": [
        {"g": 4, "n_draft": 3, "adaptive": False},
        {"g": 4, "n_draft": 3, "adaptive": True},
        {"g": 4, "n_draft": 2, "adaptive": False},
    ],
}
# 5 sessions (3 astropy + 2 django), fixed rather than "first 5 of sorted(glob(...))" so the
# mix isn't accidentally all-astropy (astropy sorts before django); reproducible either way.
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
    "prefill_s is each method's OWN prefill wall time: greedy_fast returns its own (from the "
    "same _lookup_prefill call it decodes from), and generate_lookup returns its own via "
    "stats['prefill_s']. decode_s = (that call's own end-to-end wall time) - (that same call's "
    "own prefill_s); on a very fast, very short generation this can occasionally go slightly "
    "negative from timing noise, in which case decode_s is floored rather than reported as "
    "infinite tok/s. Earlier drafts of this script subtracted greedy_fast's prefill_s from "
    "generate_lookup's total (a different cache slack, n_draft+1 vs greedy_fast's 64) -- fixed "
    "per S4 review round 1, since that over-corrected generate_lookup's decode_s in its own "
    "favour on exactly the long-prefix workload the 1.1x keep decision is gated on.",
    "identical (vs greedy_fast, not generate_greedy) < 1 is hypothesized to come from bf16 exact "
    "logit near-ties broken differently between greedy_fast's single-token decode steps and "
    "generate_lookup's batched-verify steps (same hypothesis as results/dflash.json); "
    "first_divergence records where, in tokens.",
    "Session prefixes are the last up-to-8192 tokens of the session's tokenized transcript, "
    "ending right before the start of the *last* assistant turn (replay.session tokenize's "
    "assistant mask) -- the context an agent had just before deciding what to write next, "
    "which is where prompt-lookup drafting is expected to pay off most.",
    "Settings differ per target (see the top-level `settings` key). resident sweeps g=4/n=3 "
    "fixed, g=4/n=3 adaptive, g=4/n=7 fixed: results/lookup_acceptance.json (commit 22f84f1) "
    "re-ran S2's offline simulation against S1's MEASURED verify costs (not the earlier "
    "provisional linear model) and predicts g=4/n=3 fixed as resident's best expected "
    "speedup (1.483x) and g=4/n=3 adaptive close behind (1.478x) -- both ahead of g=4/n=7 "
    "fixed (1.297x, kept here only as the comparison point against the earlier settings). "
    "streaming is restricted to short drafts (verify-block length <= 4 tokens: g=4/n=3 "
    "fixed, g=4/n=3 adaptive, g=4/n=2 fixed), dropping the n=7/n=15 settings resident uses, "
    "because results/dflash_diag.json measured a streaming verify-cost cliff, not a curve: "
    "median verify_ms was 37.8@block=1, 61.9@block=4, then 212.0@block=6 and 574.1@block=16. "
    "A block of drafted tokens > 4 can route more experts/token than the 32-slot expert "
    "cache holds at once, forcing StreamingSwitchGLU into multiple masked passes per layer; "
    "the same measured-cost sweep in results/lookup_acceptance.json confirms it directly -- "
    "streaming g=4/n=7 fixed predicts 0.692x (slower than no drafting at all), vs 1.697x for "
    "g=4/n=3 fixed (S1 follow-up, 2026-09-26).",
    "Timings may include noise from concurrent human activity on the machine.",
    "Single run per setting; no repeats.",
]


# --------------------------------------------------------------------------
# Pure helpers -- unit-tested in tests/test_bench_lookup.py, no model/no I/O.
# --------------------------------------------------------------------------

def _lookup_prefill(target, prompt, prefill_step=PREFILL_STEP, slack=64):
    """Chunked prefill identical to generate_lookup's own: last-position logits
    only, mx.eval'd per chunk so the compute graph never spans the whole
    prompt. Copied here (not imported) per the brief: don't modify laguna_mlx
    beyond generate_lookup's additive prefill_s stat. Returns (cache, logits,
    prefill_s) -- prefill_s is this call's own wall time, timed the same way
    as generate_lookup's stats["prefill_s"]."""
    prompt = [int(t) for t in prompt]  # accept numpy ints
    cache = make_trimmable_cache(target, slack=slack)
    logits = None
    t0 = time.perf_counter()
    for a in range(0, len(prompt), prefill_step):
        h = target.model(mx.array([prompt[a : a + prefill_step]]), cache=cache)
        logits = target.lm_head(h[0, -1])
        mx.eval(logits)
    prefill_s = time.perf_counter() - t0
    return cache, logits, prefill_s


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
    equals generate_greedy -- see test_greedy_fast_matches_generate_greedy.
    Returns (tokens, prefill_s): prefill_s is this call's own prefill wall
    time (see _lookup_prefill), so a caller comparing against generate_lookup
    subtracts each call's own prefill_s from its own total, never one call's
    prefill from another's total (Task S4 review, round 1)."""
    cache, logits, prefill_s = _lookup_prefill(target, prompt, prefill_step)
    out = _decode_greedy(target, cache, logits, max_tokens, eos)
    return out, prefill_s


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


def _aggregate(rows, settings):
    """Per-setting means across a workload's items. `settings` is that target's own
    SETTINGS[name] list -- row["settings"][i] must line up with settings[i]."""
    out = []
    for i, s in enumerate(settings):
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
    """True if targets[name] already ran to completion (both workloads, no stop), using the
    current SETTINGS[name] -- a settings change (e.g. this streaming-cliff follow-up) forces
    a redo instead of --skip-existing silently treating stale-settings data as current."""
    t = _load_results().get(name)
    if not t or t.get("stopped") is not None:
        return False
    settings = SETTINGS[name]
    return all(t.get(label, {}).get("settings") == settings for label in ("prompts", "sessions"))


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


def _run_item(target, ids, max_tokens, eos, settings):
    """Time greedy_fast and each of `settings` (that target's own SETTINGS[name] list) on
    the same item, each against its own prefill_s (see NOTES) -- not a shared or cross-call
    estimate."""
    t0 = time.perf_counter()
    base_out, base_prefill_s = greedy_fast(target, ids, max_tokens, eos)
    base_total_s = time.perf_counter() - t0
    base_decode_s = max(base_total_s - base_prefill_s, 1e-9)
    base = {"tokens": len(base_out), "prefill_s": base_prefill_s, "decode_s": base_decode_s,
            "total_s": base_total_s, "decode_tok_s": len(base_out) / base_decode_s,
            "e2e_tok_s": len(base_out) / base_total_s}

    rows = []
    for s in settings:
        t2 = time.perf_counter()
        out, stats = generate_lookup(target, ids, max_tokens, eos, g=s["g"], n_draft=s["n_draft"],
                                      adaptive=s["adaptive"], prefill_step=PREFILL_STEP)
        total_s = time.perf_counter() - t2
        decode_s = max(total_s - stats["prefill_s"], 1e-9)
        identical = out == base_out
        rows.append({
            **s, "tokens": len(out), "total_s": total_s, "prefill_s": stats["prefill_s"],
            "decode_s": decode_s, "decode_tok_s": len(out) / decode_s,
            "e2e_tok_s": len(out) / total_s,
            "speedup_decode": (len(out) / decode_s) / base["decode_tok_s"],
            "speedup_e2e": (len(out) / total_s) / base["e2e_tok_s"],
            "mean_accepted": stats["mean_accepted"], "steps": stats["steps"],
            "drafted": stats["drafted"], "accepted": stats["accepted"],
            "steps_with_draft": stats["steps_with_draft"],
            "identical": identical,
            "first_divergence": None if identical else first_divergence(out, base_out),
        })
    return base, rows


def _existing_rows(result, label, settings):
    """{item_name: row} already completed for this (target, workload), from a result dict
    possibly loaded from a previous, interrupted run -- but only if that run used the same
    `settings` list. Settings can change between runs (e.g. this streaming-cliff follow-up),
    and a row's "settings" list is aligned by *index*, not by name; resuming under a
    different settings list would silently pair old data at index i with a new, different
    setting at index i. So a settings mismatch is treated as no prior data at all, not as
    data to reuse."""
    w = result.get(label)
    if not w or w.get("settings") != settings:
        return {}
    return {r["name"]: r for r in w["rows"]}


def _run_workload(name, result, target, items, max_tokens, eos, swap, over, label, settings):
    """Runs `items` not already present in result[label] (resumed from disk, keyed against
    `settings` -- see _existing_rows), saving to results/lookup.json (atomic) after every
    item so a crash mid-workload only costs the items not yet finished, not the whole target."""
    done = _existing_rows(result, label, settings)
    rows = list(done.values())
    for it in items:
        if it["name"] in done:
            continue
        base, setting_rows = _run_item(target, it["ids"], max_tokens, eos, settings)
        rows.append({"name": it["name"], "prompt_tokens": len(it["ids"]),
                     "baseline": base, "settings": setting_rows})
        result[label] = {"rows": rows, "aggregate": _aggregate(rows, settings), "settings": settings}
        _save_target(name, result)  # checkpoint this item immediately
        swap.append((f"{label}:{it['name']}", swap_gb()))
        if over(f"{label}:{it['name']}"):
            break
    result[label] = {"rows": rows, "aggregate": _aggregate(rows, settings), "settings": settings}
    return result[label]


def _run_target(name, args):
    """Loads the model once, resumes any already-completed items for this target from
    results/lookup.json (per-item checkpointing: see _run_workload), then runs the rest."""
    wire_resident(name == "resident")
    swap = [("start", swap_gb())]
    settings = SETTINGS[name]
    prior = _load_results().get(name) or {}
    result = {"target": name, "slots": args.slots if name == "streaming" else None,
              "max_tokens": args.max_tokens, "settings": settings, "swap_gb": swap, "stopped": None,
              **{k: prior[k] for k in ("prompts", "sessions") if k in prior}}

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

    _run_workload(name, result, target, prompt_items, args.max_tokens, eos, swap, over, "prompts", settings)
    result["peak_gpu_gb"] = mx.get_peak_memory() / 1e9
    _save_target(name, result)
    if result["stopped"]:
        del target
        mx.clear_cache()
        return result

    session_items = _session_items()
    _run_workload(name, result, target, session_items, args.max_tokens, eos, swap, over, "sessions", settings)
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
