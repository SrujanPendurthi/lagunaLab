"""Why is greedy action match low? Noise floor (base vs its own refs) and where pruned diverges.

python3 -m scripts.diag_actions --pruned models/Laguna-XS-2.1-pruned-50
From another repo, pass --out (and --refs/--base as needed) so nothing is written here.
"""
import argparse
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np

import laguna_mlx  # noqa: F401
from laguna_mlx import wire_resident
from mlx_lm import load
from replay.replay import greedy_actions

BASE = "models/Laguna-XS-2.1-mlx-4bit"
AREFS = Path("refs/prune_actions")


def tool(a):
    return a[len("<tool_call>"):].split("<", 1)[0] if a.startswith("<tool_call>") else None


def run(path, refs=AREFS):
    model, tok = load(path)
    rows = []
    for r in sorted(Path(refs).glob("*.npz")):
        ref = np.load(r)
        want = json.loads(str(ref["actions"]))
        got = greedy_actions(model, tok, ref["ids"], ref["action_starts"], int(ref["max_new"]))
        for s, w, g in zip(ref["action_starts"].tolist(), want, got):
            rows.append({"session": r.stem, "start": s, "exact": w == g, "same_tool": tool(w) == tool(g),
                         "common_prefix": len(os.path.commonprefix([w, g])), "ref_len": len(w),
                         "ref": w, "got": g})
            print(path, r.stem, s, rows[-1]["exact"], rows[-1]["same_tool"], flush=True)
    del model
    mx.clear_cache()
    return {"exact": float(np.mean([x["exact"] for x in rows])),
            "same_tool": float(np.mean([x["same_tool"] for x in rows])), "rows": rows}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pruned", required=True)
    p.add_argument("--base", default=BASE, help="model that built --refs: the noise floor")
    p.add_argument("--refs", default=str(AREFS))
    p.add_argument("--out", default="results/action_diag.json")
    a = p.parse_args(argv)
    wire_resident(True)
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    for name, path in (("base", a.base), ("pruned", a.pruned)):
        if name not in res:
            res[name] = run(path, a.refs)
            out.write_text(json.dumps(res, indent=1))
    print({k: (v["exact"], v["same_tool"]) for k, v in res.items()})


if __name__ == "__main__":
    main()
