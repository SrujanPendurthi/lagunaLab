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
DIAG = Path("results/dflash_diag.json")
OUT = Path("results/lookup_acceptance.json")
G_VALUES = (2, 3, 4)
N_VALUES = (1, 2, 3, 7, 15)
N_VALUES_LEGACY = (3, 7, 15)  # the original provisional-linear-model sweep, kept for comparison
TARGETS = ("resident", "streaming")
BASE_MS, PER_TOKEN_MS = 24.0, 6.5  # provisional linear model: C(m) = BASE_MS + PER_TOKEN_MS * m


def provisional_linear_cost_ms(m):
    """The original (now-superseded) cost model: linear extrapolation from just two points."""
    return BASE_MS + PER_TOKEN_MS * m


def build_measured_cost_fn(verify_ms):
    """Cost model measured by S1 (results/dflash_diag.json -> targets.<target>.verify_ms):
    a dict from block length L (int, as a JSON string key) -> {"median": ms, "iqr": ms}, with
    L = m + 1 (m = drafted tokens; the "+1" is the target's own resampled token). Returns
    cost_ms(m) reading the median directly for a measured L, linearly interpolating between the
    two nearest measured L for one that wasn't measured. Never extrapolates outside the measured
    range -- the sweep never needs to (max n=15 -> max L=16, which is itself measured).
    """
    medians = {int(L): v["median"] for L, v in verify_ms.items()}
    measured_L = sorted(medians)
    lo_L, hi_L = measured_L[0], measured_L[-1]

    def cost_ms(m):
        L = m + 1
        if L < lo_L or L > hi_L:
            raise ValueError(f"L={L} (m={m}) outside measured range [{lo_L}, {hi_L}]")
        if L in medians:
            return medians[L]
        below = max(x for x in measured_L if x < L)
        above = min(x for x in measured_L if x > L)
        frac = (L - below) / (above - below)
        return medians[below] + frac * (medians[above] - medians[below])

    cost_ms.measured_L = measured_L
    cost_ms.medians_by_L = medians
    return cost_ms


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


def simulate(ids, span_list, g, n, adaptive, cost_fn=provisional_linear_cost_ms):
    """Greedy speculative decoding over each assistant span using prompt-lookup drafts.

    At t, propose d = lookup(ids, t, g, n) (adaptive: n capped at min(n, 1 + previous
    step's accepted count); a span's first step uses the full n). Accepted k = matching
    prefix length of d against the true continuation, clamped to the span's own end (a
    draft can't be credited with tokens past the assistant span -- those positions belong
    to a different message/role and were never something this decode step could have
    produced). One verify forward then advances k+1 tokens (also clamped to not run past
    the span end: if the entire remaining span matched, k+1 would overshoot by exactly 1,
    since there's no room left in the span for the target's own resampled token) at cost
    cost_fn(len(d)) -- the drafted length itself is not clamped, since that's the cost
    actually paid for the forward regardless of how much of it turns out to be creditable.
    Defaults to the (superseded) provisional linear cost model for backward compatibility.
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
            c = cost_fn(len(d))
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


def derive(r, base_ms):
    """base_ms = cost_fn(0), i.e. a plain 1-token decode -- the speedup baseline. Differs per
    target/cost model, so it's always passed explicitly rather than assumed as a constant."""
    steps = r["steps"] or 1
    speedup = (r["tokens_generated"] / r["total_cost_ms"]) / (1.0 / base_ms) if r["total_cost_ms"] else 0.0
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

    def run_sweep(g_values, n_values, cost_fn, base_ms):
        sweep = []
        for adaptive in (False, True):
            for g in g_values:
                for n in n_values:
                    by_session = {}
                    for s in sessions:
                        r = simulate(s["ids"], s["spans"], g, n, adaptive, cost_fn)
                        by_session[s["name"]] = {"repo": s["repo"], **derive(r, base_ms)}
                    by_repo = {}
                    for repo in sorted({s["repo"] for s in sessions}):
                        rs = [by_session[s["name"]] for s in sessions if s["repo"] == repo]
                        by_repo[repo] = derive(merge(rs), base_ms)
                    overall = derive(merge(list(by_session.values())), base_ms)
                    sweep.append({
                        "g": g, "n": n, "variant": "adaptive" if adaptive else "fixed",
                        "overall": overall, "by_repo": by_repo, "by_session": by_session,
                    })
        return sweep

    def summarize(sweep):
        fixed_settings = [r for r in sweep if r["variant"] == "fixed"]
        adaptive_settings = [r for r in sweep if r["variant"] == "adaptive"]
        best_tpf = max(sweep, key=lambda r: r["overall"]["tokens_per_forward"])
        best_speedup = max(sweep, key=lambda r: r["overall"]["expected_speedup"])
        best_fixed = max(fixed_settings, key=lambda r: r["overall"]["expected_speedup"])
        best_adaptive = max(adaptive_settings, key=lambda r: r["overall"]["expected_speedup"])
        go_tpf = best_tpf["overall"]["tokens_per_forward"] >= 2.0
        go_speedup = best_speedup["overall"]["expected_speedup"] >= 1.1

        def pick(row):
            return {"g": row["g"], "n": row["n"], "variant": row["variant"],
                    "tokens_per_forward": row["overall"]["tokens_per_forward"],
                    "expected_speedup": row["overall"]["expected_speedup"],
                    "clears_1.1x": row["overall"]["expected_speedup"] >= 1.1}

        return {
            "best_tokens_per_forward": pick(best_tpf),
            "best_expected_speedup": pick(best_speedup),
            "best_fixed": pick(best_fixed),
            "best_adaptive": pick(best_adaptive),
            "go_no_go": {
                "tokens_per_forward_threshold": 2.0, "tokens_per_forward_pass": go_tpf,
                "expected_speedup_threshold": 1.1, "expected_speedup_pass": go_speedup,
                "proceed_to_s3": go_tpf and go_speedup,
            },
        }

    # --- measured cost models (S1, results/dflash_diag.json), per target ---
    diag = json.loads(DIAG.read_text())
    measured_cost_fns, cost_models_measured = {}, {}
    for target in TARGETS:
        verify_ms = diag["targets"][target]["verify_ms"]
        fn = build_measured_cost_fn(verify_ms)
        measured_cost_fns[target] = fn
        cost_models_measured[target] = {
            "status": "measured",
            "source": str(DIAG),
            "measured_L": fn.measured_L,
            "verify_ms_median_by_L": fn.medians_by_L,
            "baseline_ms": fn.medians_by_L[1],
            "formula": "C(m) = verify_ms_median_by_L[m+1]; for an L=m+1 not directly measured, "
                       "linearly interpolated between the two nearest measured L (e.g. L=5, 7, "
                       "9-11, 13-15 for measured_L={1,2,3,4,6,8,12,16}); never extrapolated "
                       "outside the measured range.",
            "expected_speedup_formula": "(tokens_generated / total_cost_ms) / (1 / baseline_ms)",
        }

    sweep_by_target, best_per_target, go_no_go_per_target = {}, {}, {}
    for target in TARGETS:
        sw = run_sweep(G_VALUES, N_VALUES, measured_cost_fns[target], measured_cost_fns[target](0))
        sweep_by_target[target] = sw
        summary = summarize(sw)
        best_per_target[target] = {k: summary[k] for k in
                                    ("best_tokens_per_forward", "best_expected_speedup", "best_fixed", "best_adaptive")}
        go_no_go_per_target[target] = summary["go_no_go"]

    # --- legacy provisional linear model (superseded; kept for before/after comparison) ---
    legacy_sweep = run_sweep(G_VALUES, N_VALUES_LEGACY, provisional_linear_cost_ms, BASE_MS)
    legacy_summary = summarize(legacy_sweep)

    result = {
        "g_values": list(G_VALUES), "n_values": list(N_VALUES),
        "corpus": corpus,
        "num_sessions": len(sessions), "session_names": [s["name"] for s in sessions],
        "cost_models": {
            "measured": cost_models_measured,
            "provisional_linear_legacy": {
                "status": "superseded",
                "base_ms": BASE_MS, "per_token_ms": PER_TOKEN_MS,
                "formula": "C(m) = base_ms + per_token_ms * m (superseded by cost_models.measured; "
                           "kept only so the before/after is visible)",
                "note": "Originally sourced from two points (m=0, m=15) in Task 14's profiling "
                        "of the resident target, not itself checked into results/. Replaced by "
                        "S1's results/dflash_diag.json, which measures verify cost at 8 block "
                        "lengths per target instead of extrapolating a line through 2 points.",
            },
        },
        "sweep": [{"target": target, **row} for target in TARGETS for row in sweep_by_target[target]],
        "best_per_target": best_per_target,
        "go_no_go_per_target": go_no_go_per_target,
        "legacy_provisional_linear_model": {
            "n_values": list(N_VALUES_LEGACY),
            "sweep": legacy_sweep,
            "best_tokens_per_forward": legacy_summary["best_tokens_per_forward"],
            "best_expected_speedup": legacy_summary["best_expected_speedup"],
            "best_fixed": legacy_summary["best_fixed"],
            "best_adaptive": legacy_summary["best_adaptive"],
            "go_no_go": legacy_summary["go_no_go"],
        },
        "runtime_seconds": time.time() - t_start,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))
    print(f"wrote {OUT} in {result['runtime_seconds']:.1f}s total", flush=True)
    for target in TARGETS:
        print(target, "best fixed:", best_per_target[target]["best_fixed"], flush=True)
        print(target, "best adaptive:", best_per_target[target]["best_adaptive"], flush=True)
        print(target, "go/no-go:", go_no_go_per_target[target], flush=True)


if __name__ == "__main__":
    main()
