"""Offline prompt-lookup acceptance / expected-speedup on captured pool sessions (CPU only, no model).

For every assistant span in every traces/*.jsonl session, simulate greedy speculative decoding
where the "draft" is prompt-lookup (copy the continuation of the most recent earlier occurrence
of the last g tokens) instead of a learned draft model. Acceptance is exact because we compare
against the model's own recorded output (see task-S2-brief.md).

python3 -m scripts.lookup_acceptance
"""
import glob
import json
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from replay.session import load_session, tokenize

TOK = "models/Laguna-XS-2.1"
TRACES = "traces/*.jsonl"
OUT = Path("results/lookup_acceptance.json")
G_VALUES = (2, 3, 4)
N_VALUES = (3, 7, 15)
BASE_MS, PER_TOKEN_MS = 24.0, 6.5  # C(m) = BASE_MS + PER_TOKEN_MS * m, m = drafted tokens


def cost_ms(m):
    return BASE_MS + PER_TOKEN_MS * m


def lookup(ids, t, g, n):
    """Reference from the brief: tokens that followed the most recent earlier occurrence of
    ids[t-g:t], up to n. O(T) per call -- kept only for tests; the sweep below uses GramIndex,
    which is the dict-based equivalent the brief's ponytail note asks for once a linear scan
    would make a full run too slow (it would here: O(T^2) over ~65k-token sessions)."""
    if t < g:
        return []
    key = tuple(ids[t - g : t])
    for s in range(t - g - 1, -1, -1):
        if tuple(ids[s : s + g]) == key:
            return list(ids[s + g : min(s + g + n, t)])
    return []


class GramIndex:
    """Incremental dict from g-gram -> most recent starting position, built lazily as the
    query position t advances (t must be non-decreasing across calls, as it is in simulate()).
    propose(t, n) is equivalent to lookup(ids, t, g, n) but O(1) amortized per call."""

    def __init__(self, ids, g):
        self.ids, self.g, self.d, self.next_s = ids, g, {}, 0

    def propose(self, t, n):
        g, ids, d = self.g, self.ids, self.d
        if t < g:
            return []
        limit = t - g  # dict must hold grams for s in [0, limit-1] before this query
        if limit > self.next_s:
            for s in range(self.next_s, limit):
                d[tuple(ids[s : s + g])] = s
            self.next_s = limit
        s = d.get(tuple(ids[t - g : t]))
        return [] if s is None else ids[s + g : min(s + g + n, t)]


def common_prefix_len(a, b):
    m = min(len(a), len(b))
    i = 0
    while i < m and a[i] == b[i]:
        i += 1
    return i


def spans(mask):
    """Contiguous (start, end) runs of True in a bool mask, end exclusive."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]])) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def simulate(ids, span_list, g, n, adaptive):
    """Greedy speculative decoding over each assistant span using prompt-lookup drafts.

    At t, propose d = lookup(ids, t, g, n) (adaptive: n capped at min(n, 1 + previous
    step's accepted count); a span's first step uses the full n). Accepted k = matching
    prefix length of d against the true continuation, clamped to the span's own end (a
    draft can't be credited with tokens past the assistant span -- those positions belong
    to a different message/role and were never something this decode step could have
    produced). One verify forward then advances k+1 tokens (also clamped to not run past
    the span end: if the entire remaining span matched, k+1 would overshoot by exactly 1,
    since there's no room left in the span for the target's own resampled token) at cost
    C(len(d)) -- the drafted length itself is not clamped, since that's the cost actually
    paid for the forward regardless of how much of it turns out to be creditable.
    """
    idx = GramIndex(ids, g)
    steps = hits = accepted_sum = tokens_generated = 0
    total_cost = 0.0
    for start, end in span_list:
        t, prev_k = start, n - 1
        while t < end:
            n_cur = n if not adaptive else min(n, 1 + prev_k)
            d = idx.propose(t, n_cur)
            window = ids[t : min(t + len(d), end)]
            k = common_prefix_len(d, window)
            c = cost_ms(len(d))
            advance = min(k + 1, end - t)
            steps += 1
            hits += 1 if d else 0
            accepted_sum += k
            tokens_generated += advance
            total_cost += c
            prev_k = k
            t += advance
    return {
        "steps": steps, "hits": hits, "accepted_sum": accepted_sum,
        "tokens_generated": tokens_generated, "total_cost_ms": total_cost,
    }


def derive(r):
    steps = r["steps"] or 1
    speedup = (r["tokens_generated"] / r["total_cost_ms"]) / (1.0 / BASE_MS) if r["total_cost_ms"] else 0.0
    return {
        **r,
        "mean_accepted": r["accepted_sum"] / steps,
        "hit_rate": r["hits"] / steps,
        "tokens_per_forward": r["tokens_generated"] / steps,
        "expected_speedup": speedup,
    }


def merge(rs):
    keys = ("steps", "hits", "accepted_sum", "tokens_generated", "total_cost_ms")
    return {k: sum(r[k] for r in rs) for k in keys}


def repo_of(path):
    return Path(path).stem.split("__")[0]


def main():
    t_start = time.time()
    tok = AutoTokenizer.from_pretrained(TOK)
    files = sorted(glob.glob(TRACES))
    sessions = []
    for f in files:
        msgs, tools = load_session(f)
        ids, mask, truncated = tokenize(tok, msgs, tools)
        sessions.append({
            "name": Path(f).stem, "repo": repo_of(f), "ids": ids.tolist(),
            "spans": spans(mask), "truncated": truncated,
        })
    print(f"loaded {len(sessions)} sessions in {time.time()-t_start:.1f}s", flush=True)

    corpus_by_session = {
        s["name"]: {
            "repo": s["repo"], "total_tokens": len(s["ids"]),
            "assistant_tokens": sum(e - st for st, e in s["spans"]),
            "truncated": s["truncated"],
        }
        for s in sessions
    }
    corpus_by_repo = {}
    for repo in sorted({s["repo"] for s in sessions}):
        rows = [corpus_by_session[s["name"]] for s in sessions if s["repo"] == repo]
        corpus_by_repo[repo] = {
            "num_sessions": len(rows),
            "total_tokens": sum(r["total_tokens"] for r in rows),
            "assistant_tokens": sum(r["assistant_tokens"] for r in rows),
        }
    corpus = {
        "num_sessions": len(sessions),
        "total_tokens": sum(v["total_tokens"] for v in corpus_by_session.values()),
        "total_assistant_tokens": sum(v["assistant_tokens"] for v in corpus_by_session.values()),
        "by_repo": corpus_by_repo,
        "by_session": corpus_by_session,
    }

    sweep = []
    for adaptive in (False, True):
        for g in G_VALUES:
            for n in N_VALUES:
                by_session = {}
                for s in sessions:
                    r = simulate(s["ids"], s["spans"], g, n, adaptive)
                    by_session[s["name"]] = {"repo": s["repo"], **derive(r)}
                by_repo = {}
                for repo in sorted({s["repo"] for s in sessions}):
                    rs = [by_session[s["name"]] for s in sessions if s["repo"] == repo]
                    by_repo[repo] = derive(merge(rs))
                overall = derive(merge(list(by_session.values())))
                sweep.append({
                    "g": g, "n": n, "variant": "adaptive" if adaptive else "fixed",
                    "overall": overall, "by_repo": by_repo, "by_session": by_session,
                })

    fixed_settings = [r for r in sweep if r["variant"] == "fixed"]
    best_tpf = max(sweep, key=lambda r: r["overall"]["tokens_per_forward"])
    best_speedup = max(sweep, key=lambda r: r["overall"]["expected_speedup"])
    best_fixed_speedup = max(fixed_settings, key=lambda r: r["overall"]["expected_speedup"])

    go_tpf = best_tpf["overall"]["tokens_per_forward"] >= 2.0
    go_speedup = best_speedup["overall"]["expected_speedup"] >= 1.1

    result = {
        "cost_model": {
            "base_ms": BASE_MS, "per_token_ms": PER_TOKEN_MS,
            "formula": "C(m) = base_ms + per_token_ms * m, where m = number of drafted "
                       "(proposed) tokens; m=0 is a plain 1-token decode at base_ms.",
            "expected_speedup_formula": "(tokens_generated / total_cost_ms) / (1 / base_ms)",
            "status": "provisional",
            "source": "Two points (m=0, m=15) from Task 14's profiling of the resident "
                      "Laguna-XS-2.1 target (~24ms for a 1-token forward, ~121ms for a "
                      "16-token block); that profiling is not itself checked into results/ "
                      "and is treated here as provisional pending S1's "
                      "results/dflash_diag.json, which should supersede these constants. "
                      "A linear model is assumed between and beyond the two measured "
                      "points; not independently verified at intermediate m.",
        },
        "g_values": list(G_VALUES), "n_values": list(N_VALUES),
        "corpus": corpus,
        "num_sessions": len(sessions), "session_names": [s["name"] for s in sessions],
        "sweep": sweep,
        "best_tokens_per_forward": {"g": best_tpf["g"], "n": best_tpf["n"], "variant": best_tpf["variant"],
                                     "tokens_per_forward": best_tpf["overall"]["tokens_per_forward"]},
        "best_expected_speedup": {"g": best_speedup["g"], "n": best_speedup["n"], "variant": best_speedup["variant"],
                                   "expected_speedup": best_speedup["overall"]["expected_speedup"]},
        "best_fixed_expected_speedup": {"g": best_fixed_speedup["g"], "n": best_fixed_speedup["n"],
                                         "expected_speedup": best_fixed_speedup["overall"]["expected_speedup"]},
        "go_no_go": {
            "tokens_per_forward_threshold": 2.0, "tokens_per_forward_pass": go_tpf,
            "expected_speedup_threshold": 1.1, "expected_speedup_pass": go_speedup,
            "proceed_to_s3": go_tpf and go_speedup,
        },
        "runtime_seconds": time.time() - t_start,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))
    print(f"wrote {OUT} in {result['runtime_seconds']:.1f}s total", flush=True)
    print("best tokens/forward:", result["best_tokens_per_forward"], flush=True)
    print("best expected speedup:", result["best_expected_speedup"], flush=True)
    print("go/no-go:", result["go_no_go"], flush=True)


if __name__ == "__main__":
    main()
