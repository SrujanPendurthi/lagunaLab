"""Teacher-force MLX on llama.cpp's greedy continuations of 21 prompts.

Pass: MLX argmax == llama token >= 95%, and MLX argmax in llama top-5 >= 99%.

Two phases so the MLX model and llama-server are never resident together:

  python3 -m scripts.check_parity collect   # needs llama-server -m <gguf> --port 8080 -c 16384
  python3 -m scripts.check_parity score      # needs llama-server stopped; loads the MLX model
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

MODEL = "models/Laguna-XS-2.1-mlx-4bit"
N_GEN = 32
LLAMA_REF = Path("results/llama_ref.json")
PARITY_OUT = Path("results/parity.json")

PROMPTS = [
    "Write a Python function that merges two sorted lists.",
    "Explain what a race condition is, with a short Go example.",
    "Fix the bug: def avg(xs): return sum(xs) / len(xs)  # crashes on empty list",
    "Write a SQL query returning the top 3 customers by total order value.",
    "What does `git rebase --onto` do? Give an example.",
    "Implement an LRU cache in TypeScript.",
    "Write a bash one-liner that counts lines of Python code in a repo, excluding tests.",
    "Refactor this to avoid the nested loop: for a in xs:\n  for b in ys:\n    if a == b: out.append(a)",
    "Write a Rust function that parses '12:34' into (hours, minutes) with error handling.",
    "Write pytest tests for a function slugify(title: str) -> str.",
]


def llama_greedy(ids):
    body = {"prompt": ids, "n_predict": N_GEN, "temperature": 0, "n_probs": 5,
            "return_tokens": True, "cache_prompt": False}
    req = urllib.request.Request("http://127.0.0.1:8080/completion", json.dumps(body).encode(),
                                  {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    gen = r["tokens"]  # llama-server fields
    top5 = [[t["id"] for t in p["top_logprobs"]] for p in r["completion_probabilities"]]  # llama-server fields
    return gen, top5


def build_prompts(tok):
    """Return [(label, ids), ...] for the 21 prompts (10 base x thinking on/off + 1 long doc)."""
    labeled = [
        (f"p{i}_{'think' if think else 'plain'}",
         tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                  enable_thinking=think))
        for i, p in enumerate(PROMPTS) for think in (False, True)
    ]
    long_doc = Path("tests/reference/modeling_laguna.py").read_text()  # ~9k tokens, crosses 8192
    labeled.append(("long_doc", tok.apply_chat_template(
        [{"role": "user", "content": "Summarize this file:\n\n" + long_doc}],
        add_generation_prompt=True, enable_thinking=False)))
    return labeled


def collect():
    from mlx_lm.utils import load_tokenizer  # tokenizer only -- never load the MLX model here
    tok = load_tokenizer(MODEL)
    labeled = build_prompts(tok)
    rows = []
    for label, ids in labeled:
        gen, top5 = llama_greedy(ids)
        rows.append({"label": label, "prompt_ids": ids, "gen": gen, "top5": top5})
        print(label, "prompt_tokens", len(ids), "gen", len(gen))
    LLAMA_REF.parent.mkdir(parents=True, exist_ok=True)
    LLAMA_REF.write_text(json.dumps(rows))
    print(f"wrote {LLAMA_REF} ({len(rows)} rows)")


def score():
    import mlx.core as mx
    import numpy as np

    import laguna_mlx  # noqa: F401
    from mlx_lm import load

    rows_in = json.loads(LLAMA_REF.read_text())
    model, _ = load(MODEL)
    rows = []
    for row in rows_in:
        ids, gen, top5 = row["prompt_ids"], row["gen"], row["top5"]
        logits = model(mx.array([ids + gen]))[0, len(ids) - 1: len(ids) - 1 + len(gen)]
        pred = np.array(mx.argmax(logits, axis=-1)).tolist()
        # llama-server build 10964 has a Laguna short-prompt bug: for ~36-512 token
        # prompts it returns a constant token (id 31) with a fixed top5 at every
        # position, regardless of context. Flag that so a broken reference isn't
        # read as an MLX failure.
        ref_degenerate = len(set(gen)) == 1 and len({tuple(t) for t in top5}) == 1
        out = {
            "label": row["label"],
            "prompt_tokens": len(ids),
            "top1": float(np.mean([a == b for a, b in zip(pred, gen)])),
            "in_top5": float(np.mean([a in t for a, t in zip(pred, top5)])),
            "ref_degenerate": ref_degenerate,
        }
        rows.append(out)
        print(out)
    valid = [r for r in rows if not r["ref_degenerate"]]
    summary = {
        "top1": float(np.mean([r["top1"] for r in rows])),
        "in_top5": float(np.mean([r["in_top5"] for r in rows])),
        "valid_rows": len(valid),
        "top1_valid": float(np.mean([r["top1"] for r in valid])) if valid else None,
        "in_top5_valid": float(np.mean([r["in_top5"] for r in valid])) if valid else None,
        "note": ("Rows with ref_degenerate=true come from llama.cpp build 10964's Laguna "
                 "short-prompt bug: the reference /completion response returns a constant "
                 "token (id 31) with null logprobs and an identical top5 at every position "
                 "for prompts in roughly the 36-512 token range, independent of prompt "
                 "content. top1/in_top5 are over all rows (raw); top1_valid/in_top5_valid "
                 "exclude those degenerate-reference rows and are the metrics the 0.95/0.99 "
                 "gate is asserted against."),
        "rows": rows,
    }
    PARITY_OUT.parent.mkdir(parents=True, exist_ok=True)
    PARITY_OUT.write_text(json.dumps(summary, indent=1))
    print("raw: top1", summary["top1"], "in_top5", summary["in_top5"])
    print("valid_rows", summary["valid_rows"], "top1_valid", summary["top1_valid"],
          "in_top5_valid", summary["in_top5_valid"])
    assert summary["valid_rows"] >= 1, "parity gate failed: no rows with a non-degenerate reference"
    assert summary["top1_valid"] >= 0.95 and summary["in_top5_valid"] >= 0.99, "parity gate failed"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["collect", "score"])
    args = p.parse_args()
    {"collect": collect, "score": score}[args.phase]()


if __name__ == "__main__":
    sys.exit(main())
