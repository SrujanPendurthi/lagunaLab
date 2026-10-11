"""Expert-pruning sweep on the resident 4-bit model.

python3 -m scripts.prune_sweep calibrate   # astropy sessions -> results/expert_stats.npz
python3 -m scripts.prune_sweep refs        # reference logprobs for eval sessions -> refs/prune/
python3 -m scripts.prune_sweep sweep       # criteria x ratios -> results/pruning.json (resumable)
"""
import argparse
import glob
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import laguna_mlx  # noqa: F401
from laguna_mlx.pruning import ExpertSaliencyRecorder, apply_mask, keep_sets, load_stats, scores
from mlx_lm import load
from replay.replay import build_ref, score
from replay.session import load_session, tokenize

MODEL = "models/Laguna-XS-2.1-mlx-4bit"
CALIB = sorted(glob.glob("traces/astropy__*.jsonl"))
HELDOUT = sorted(glob.glob("traces/django__*.jsonl"))
CALIB_EVAL = CALIB[:2]  # in-sample, to measure the calibration/held-out gap
REFS = Path("refs/prune")
OUT = Path("results/pruning.json")
STATS_PATH = Path("results/expert_stats.npz")
STATS_META_PATH = Path("results/expert_stats_meta.json")
# keep_sets(..., allocation="global") per-layer floor. 16 is sized for the real model's expert
# count; a model with fewer experts per layer (e.g. the tiny test model) needs a smaller value.
GLOBAL_MIN_KEEP = 16
# Priority order (controller ruling): baseline, then reap/random/frequency uniform across all
# ratios, then reap global, then gate uniform last -- gate is the weakest prior of the four.
RATIOS = (0.125, 0.25, 0.375, 0.5, 0.625)
GRID = ([(c, "uniform", r) for c in ("reap", "random", "frequency") for r in RATIOS]
        + [("reap", "global", r) for r in (0.25, 0.5)]
        + [("gate", "uniform", r) for r in RATIOS])


def calibrate(model, tok, cap, chunk=2048):
    # n counts every token in each session (system prompt, tools, user, assistant) -- routing,
    # and so saliency, happens on every position, not only on assistant-generated tokens.
    t0, n = time.perf_counter(), 0
    with ExpertSaliencyRecorder(model) as rec:
        for f in CALIB:
            msgs, tools = load_session(f)
            ids, _, _ = tokenize(tok, msgs, tools, cap)
            cache = model.make_cache()
            for a in range(0, len(ids), chunk):
                mx.eval(model.model(mx.array(ids[None, a : a + chunk]), cache=cache))
            n += len(ids)
            print(f"calibrated {Path(f).stem}: {len(ids)} tokens", flush=True)
    rec.save(STATS_PATH)
    meta = {"sessions": [Path(f).stem for f in CALIB], "tokens": n, "cap": cap,
            "tok_per_s": n / (time.perf_counter() - t0), "note": "recorder syncs per layer; slower than plain prefill"}
    STATS_META_PATH.write_text(json.dumps(meta, indent=1))
    print(meta)


def refs(model, tok, cap):
    REFS.mkdir(parents=True, exist_ok=True)
    for f in HELDOUT + CALIB_EVAL:
        out = REFS / f"{Path(f).stem}.npz"
        if not out.exists():
            t0 = time.perf_counter()
            build_ref(model, tok, f, out, max_tokens=cap, n_actions=0)
            print(f"ref {out.name} in {time.perf_counter() - t0:.0f}s", flush=True)


def _aggregate(rows):
    w = np.array([r["assistant_tokens"] for r in rows], float)
    return {"kl_mean": float(np.average([r["kl_top64_mean"] for r in rows], weights=w)),
            # max over sessions of each session's p99 (not a pooled p99 over all tokens)
            "kl_p99_max": float(max(r["kl_top64_p99"] for r in rows)),
            "top1": float(np.average([r["top1_agree"] for r in rows], weights=w)),
            "assistant_tokens": int(w.sum())}


def _stamp(ref_paths):
    """Identity of the stats/refs a sweep run was built on: catches stale resumes."""
    meta = json.loads(STATS_META_PATH.read_text()) if STATS_META_PATH.exists() else {}
    refs_info = sorted(({"name": p.name, "size": p.stat().st_size} for p in ref_paths),
                        key=lambda d: d["name"])
    return {"stats_sha256": hashlib.sha256(STATS_PATH.read_bytes()).hexdigest(),
            "refs": refs_info, "cap": meta.get("cap")}


def _check_baseline(row):
    if row["heldout"]["kl_mean"] >= 1e-5 or row["heldout"]["top1"] != 1.0:
        raise SystemExit(
            f"baseline sanity check failed: heldout kl_mean={row['heldout']['kl_mean']!r}, "
            f"top1={row['heldout']['top1']!r} (want < 1e-5 and == 1.0) -- the evaluation path "
            f"is nondeterministic. The row was saved to {OUT}; investigate before sweeping further.")


def sweep(model):
    held = [REFS / f"{Path(f).stem}.npz" for f in HELDOUT]
    cal = [REFS / f"{Path(f).stem}.npz" for f in CALIB_EVAL]
    missing = [str(p) for p in [STATS_PATH, *held, *cal] if not p.exists()]
    if missing:
        raise SystemExit(
            "sweep can't run: missing " + ", ".join(missing) + ". Run "
            "`python3 -m scripts.prune_sweep calibrate` if results/expert_stats.npz is missing, "
            "then `python3 -m scripts.prune_sweep refs` for any missing refs/prune/*.npz.")
    stats = load_stats(STATS_PATH)
    stamp = _stamp(held + cal)
    done = []
    if OUT.exists():
        prev = json.loads(OUT.read_text())
        if prev.get("stamp") != stamp:
            raise SystemExit(
                f"{OUT} was built from different stats/refs than what's on disk now (stats hash, "
                "ref file sizes, or calibration cap changed) -- move it aside, e.g. "
                f"`mv {OUT} {OUT}.bak`, and rerun the sweep from scratch, or restore the stats/refs "
                "it was built from.")
        done = prev["rows"]
        for r in done:
            if r["criterion"] == "none":
                _check_baseline(r)
    have = {(r["criterion"], r["allocation"], r["ratio"]) for r in done}
    for crit, alloc, ratio in [("none", "uniform", 0.0)] + GRID:
        if (crit, alloc, ratio) in have:
            continue
        t0 = time.perf_counter()
        undo = lambda: None
        kept = None
        if ratio > 0:
            keep = keep_sets(scores(stats, crit), ratio, alloc, min_keep=GLOBAL_MIN_KEEP)
            kept = {str(i): int(len(v)) for i, v in keep.items()}
            undo = apply_mask(model, keep)
        try:
            row = {"criterion": crit, "allocation": alloc, "ratio": ratio, "kept_per_layer": kept,
                   "heldout": _aggregate([score(model, r) for r in held]),
                   "calib": _aggregate([score(model, r) for r in cal])}
        finally:
            undo()
        row["seconds"] = time.perf_counter() - t0
        done.append(row)
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps({"model": MODEL, "stamp": stamp, "rows": done}, indent=1))
        tmp.replace(OUT)  # atomic: an interrupt never leaves a half-written file
        print(crit, alloc, ratio, row["heldout"], f"{row['seconds']:.0f}s", flush=True)
        if crit == "none":
            _check_baseline(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["calibrate", "refs", "sweep"])
    p.add_argument("--cap", type=int, default=16384)
    a = p.parse_args()
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    model, tok = load(MODEL)
    {"calibrate": lambda: calibrate(model, tok, a.cap), "refs": lambda: refs(model, tok, a.cap),
     "sweep": lambda: sweep(model)}[a.stage]()


if __name__ == "__main__":
    main()
