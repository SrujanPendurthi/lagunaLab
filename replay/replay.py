"""Teacher-forced replay: compare any model to the unmodified reference on captured sessions.

The reference .npz stores the token ids, so every scored model sees identical inputs.
KL is truncated to the reference's top-64 tokens (>99.9% of mass in practice).

python3 -m replay.replay build-ref --model M --traces traces --refs refs
python3 -m replay.replay score --model M --refs refs --out results/x.json [--actions]
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from replay.session import load_session, tokenize

TOPK = 64
SUB = 256  # positions per log-softmax batch: 256 x 100k vocab x 4 B = 100 MB


def assistant_logprobs(model, ids, mask, chunk=2048):
    """Yield (positions, logprobs[n, V] float32) for positions whose next token is assistant."""
    target = np.zeros_like(mask)
    target[:-1] = mask[1:]
    cache = model.make_cache()
    for a in range(0, len(ids), chunk):
        b = min(a + chunk, len(ids))
        h = model.model(mx.array(ids[None, a:b]), cache=cache)[0]
        sel = np.nonzero(target[a:b])[0]
        if len(sel) == 0:
            mx.eval(h)
            continue
        for s in range(0, len(sel), SUB):
            part = sel[s : s + SUB]
            lp = model.lm_head(h[mx.array(part)]).astype(mx.float32)
            lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
            mx.eval(lp)
            yield part + a, lp


def _action(text):
    """The tool call if there is one, else the whole reply: what an agent would execute."""
    i, j = text.find("<tool_call>"), text.find("</tool_call>")
    return text[i : j + len("</tool_call>")] if 0 <= i < j else text.strip()


def _greedy_actions(model, tokenizer, ids, starts, max_new):
    from mlx_lm import generate

    # ponytail: re-prefills the whole prefix per turn; reuse a trimmed cache if this is too slow
    return [_action(generate(model, tokenizer, prompt=ids[:s].tolist(), max_tokens=max_new))
            for s in starts]


def build_ref(model, tokenizer, session_path, out_path, max_tokens=65536, n_actions=2,
              max_new=1024, chunk=2048, seed=0):
    msgs, tools = load_session(session_path)
    ids, mask, truncated = tokenize(tokenizer, msgs, tools, max_tokens)
    pos, top_ids, top_lp = [], [], []
    for p, lp in assistant_logprobs(model, ids, mask, chunk):
        idx = mx.argpartition(-lp, kth=TOPK - 1, axis=-1)[:, :TOPK]
        vals = mx.take_along_axis(lp, idx, axis=-1)
        order = mx.argsort(-vals, axis=-1)
        pos.append(p)
        top_ids.append(np.array(mx.take_along_axis(idx, order, axis=-1)))
        top_lp.append(np.array(mx.take_along_axis(vals, order, axis=-1)))
    if not pos:
        raise ValueError(f"{session_path}: no assistant tokens")
    starts = np.nonzero(mask & ~np.r_[False, mask[:-1]])[0]
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.choice(starts, min(n_actions, len(starts)), replace=False))
    actions = _greedy_actions(model, tokenizer, ids, starts, max_new)
    np.savez_compressed(out_path, ids=ids, mask=mask, truncated=truncated,
                        pos=np.concatenate(pos), top_ids=np.concatenate(top_ids),
                        top_lp=np.concatenate(top_lp), action_starts=starts,
                        actions=json.dumps(actions), max_new=max_new)


def score(model, ref_path, chunk=2048):
    ref = np.load(ref_path)
    ids, mask = ref["ids"], ref["mask"]
    kls, agree, i = [], [], 0
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    for p, lp in assistant_logprobs(model, ids, mask, chunk):
        n = len(p)
        assert (p == ref["pos"][i : i + n]).all(), "position mismatch vs reference"
        tid = mx.array(ref["top_ids"][i : i + n])
        rlp = mx.array(ref["top_lp"][i : i + n])
        i += n
        mlp_ = mx.take_along_axis(lp, tid, axis=-1)
        kls.append(np.array((mx.exp(rlp) * (rlp - mlp_)).sum(axis=-1)))
        agree.append(np.array(mx.argmax(lp, axis=-1) == tid[:, 0]))
    dt = time.perf_counter() - t0
    kl, ag = np.concatenate(kls), np.concatenate(agree)
    return {"tokens": int(len(ids)), "assistant_tokens": int(len(kl)),
            "truncated": bool(ref["truncated"]), "kl_top64_mean": float(kl.mean()),
            "kl_top64_p99": float(np.percentile(kl, 99)), "top1_agree": float(ag.mean()),
            "peak_mem_gb": mx.get_peak_memory() / 1e9, "tok_per_s": len(ids) / dt}


def score_actions(model, tokenizer, ref_path, max_new=None):
    ref = np.load(ref_path)
    want = json.loads(str(ref["actions"]))
    if not want:
        return float("nan")
    got = _greedy_actions(model, tokenizer, ref["ids"], ref["action_starts"],
                          max_new or int(ref["max_new"]))
    return float(np.mean([a == b for a, b in zip(got, want)]))


def main():
    import laguna_mlx  # noqa: F401
    from mlx_lm import load

    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build-ref", "score"])
    p.add_argument("--model", required=True)
    p.add_argument("--traces", default="traces")
    p.add_argument("--refs", default="refs")
    p.add_argument("--out")
    p.add_argument("--max-tokens", type=int, default=65536)
    p.add_argument("--actions", action="store_true")
    a = p.parse_args()
    model, tok = load(a.model)
    refs = Path(a.refs)
    if a.cmd == "build-ref":
        refs.mkdir(exist_ok=True)
        for f in sorted(Path(a.traces).glob("*.jsonl")):
            out = refs / f"{f.stem}.npz"
            if not out.exists():
                build_ref(model, tok, f, out, a.max_tokens)
                print("ref", f.stem, flush=True)
        return
    rows = []
    for r in sorted(refs.glob("*.npz")):
        row = {"session": r.stem, **score(model, r)}
        if a.actions:
            row["action_match"] = score_actions(model, tok, r)
        rows.append(row)
        print(row, flush=True)
    w = np.array([x["assistant_tokens"] for x in rows])
    summary = {k: float(np.average([x[k] for x in rows], weights=w))
               for k in ("kl_top64_mean", "top1_agree")}
    summary["peak_mem_gb"] = max(x["peak_mem_gb"] for x in rows)
    if a.actions:
        summary["action_match"] = float(np.nanmean([x["action_match"] for x in rows]))
    Path(a.out).write_text(json.dumps({"summary": summary, "sessions": rows}, indent=1))
    print(summary)


if __name__ == "__main__":
    main()
