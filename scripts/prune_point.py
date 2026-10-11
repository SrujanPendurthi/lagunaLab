"""Materialize + benchmark one pruning operating point.

python3 -m scripts.prune_point --criterion reap --ratio 0.5
"""
import argparse
import glob
import json
import subprocess
import time
from pathlib import Path

import mlx.core as mx

import laguna_mlx  # noqa: F401
from laguna_mlx.dflash import generate_greedy
from laguna_mlx.pruning import keep_sets, load_stats, prune_checkpoint, scores
from laguna_mlx.streaming import load_streaming, stream_stats
from mlx_lm import load
from replay.replay import build_ref, score, score_actions
from scripts.check_parity import PROMPTS
from scripts.pack_experts import pack

BASE = "models/Laguna-XS-2.1-mlx-4bit"
HELDOUT = sorted(glob.glob("traces/django__*.jsonl"))
AREFS = Path("refs/prune_actions")


def du_gb(p):
    return int(subprocess.check_output(["du", "-sk", str(p)]).split()[0]) / 1e6


def decode_bench(model, tok):
    prs = [tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                   enable_thinking=False) for q in PROMPTS[:3]]
    mx.reset_peak_memory()
    t0, n = time.perf_counter(), 0
    for pr in prs:
        n += len(generate_greedy(model, pr, 128, tuple(tok.eos_token_ids)))
    return {"decode_tok_s": n / (time.perf_counter() - t0), "peak_gpu_gb": mx.get_peak_memory() / 1e9}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--criterion", default="reap")
    p.add_argument("--ratio", type=float, required=True)
    p.add_argument("--cap", type=int, default=16384)
    p.add_argument("--actions", type=int, default=4)
    a = p.parse_args()
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    pct = int(round(a.ratio * 100))
    out_dir = Path(f"models/Laguna-XS-2.1-pruned-{a.criterion}-{pct}")
    pack_dir = Path(f"models/Laguna-XS-2.1-pruned-{a.criterion}-{pct}-packed")
    res = {"criterion": a.criterion, "ratio": a.ratio, "cap": a.cap, "actions": a.actions,
           "note": "operating point was picked on the django held-out curve in results/pruning.json, "
                   "so kl/top1 here are post-selection, not an untouched estimate"}

    # 1. references with greedy actions from the UNPRUNED resident model
    AREFS.mkdir(parents=True, exist_ok=True)
    base, tok = load(BASE)
    for f in HELDOUT:
        r = AREFS / f"{Path(f).stem}.npz"
        if not r.exists():
            build_ref(base, tok, f, r, max_tokens=a.cap, n_actions=a.actions)
    res["base"] = decode_bench(base, tok)
    res["base"]["disk_gb"] = du_gb(BASE)
    del base
    mx.clear_cache()

    # 2. materialize + pack
    keep = keep_sets(scores(load_stats("results/expert_stats.npz"), a.criterion), a.ratio)
    if not out_dir.exists():
        prune_checkpoint(BASE, keep, out_dir)
    if not pack_dir.exists():
        pack(out_dir, pack_dir)

    # 3. pruned resident: memory, speed, replay, actions
    model, tok = load(str(out_dir))
    res["resident"] = decode_bench(model, tok)
    res["resident"]["disk_gb"] = du_gb(out_dir)
    rows = [score(model, r) for r in sorted(AREFS.glob("*.npz"))]
    w = [r["assistant_tokens"] for r in rows]
    res["resident"]["kl_mean"] = sum(r["kl_top64_mean"] * x for r, x in zip(rows, w)) / sum(w)
    res["resident"]["top1"] = sum(r["top1_agree"] * x for r, x in zip(rows, w)) / sum(w)
    am = [m for m in (score_actions(model, tok, r) for r in sorted(AREFS.glob("*.npz"))) if m == m]
    res["resident"]["action_match"] = sum(am) / len(am) if am else None
    res["resident"]["action_match_refs"] = len(am)
    del model
    mx.clear_cache()

    # 4. pruned streaming
    model, tok = load_streaming(pack_dir, 32)
    res["streaming_slots32"] = {**decode_bench(model, tok), **stream_stats(model), "disk_gb": du_gb(pack_dir)}
    Path("results/prune_point.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
