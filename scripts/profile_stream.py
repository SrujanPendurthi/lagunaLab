"""Where streaming decode time goes (Task S5): per decode step, summed over the MoE layers.

python3 -m scripts.profile_stream   # pruned-50 pack, slots=32, at short and 32k context

  sync_ms   time blocked in np.array(indices): the GPU finishing everything queued
            since the last layer (attention + previous layers' compute) plus the round-trip
  load_ms   wall time in ExpertStore.load (slot bookkeeping + preads + slot writes)
  pread_ms  time inside os.pread, summed over pool threads (an upper bound on its wall share)
  other_ms  step - sync - load: graph building, lm_head, argmax
"""
import json
import time
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import numpy as np

import laguna_mlx  # noqa: F401
from laguna_mlx import streaming

PACK = "models/Laguna-XS-2.1-pruned-50-packed"
SESSION = Path("refs/kv/django__django-11039.npz")  # longest captured session (82,671 tokens)
OUT = Path("results/stream_profile.json")


class _TimedNumpy:
    """Stands in for `np` inside laguna_mlx.streaming: times np.array, forwards the rest."""

    def __init__(self, t):
        self._t = t

    def array(self, *a, **k):
        t0 = time.perf_counter()
        r = np.array(*a, **k)
        self._t["sync"] += time.perf_counter() - t0
        return r

    def __getattr__(self, name):
        return getattr(np, name)


@contextmanager
def timers():
    t = {"sync": 0.0, "load": 0.0, "pread": 0.0}
    S = streaming.ExpertStore
    load, read = S.load, S._read

    def timed_load(self, experts):
        t0 = time.perf_counter()
        r = load(self, experts)
        t["load"] += time.perf_counter() - t0
        return r

    def timed_read(self, e):
        t0 = time.perf_counter()
        r = read(self, e)
        t["pread"] += time.perf_counter() - t0  # ponytail: unlocked += from pool threads, may drop a few samples
        return r

    S.load, S._read, streaming.np = timed_load, timed_read, _TimedNumpy(t)
    try:
        yield t
    finally:
        S.load, S._read, streaming.np = load, read, np


def profile(model, ids, n=32, chunk=2048):
    """Prefill `ids`, then time n greedy decode steps. Returns per-step ms."""
    cache = model.make_cache()
    for a in range(0, len(ids), chunk):
        h = model.model(mx.array(ids[None, a : a + chunk]), cache=cache)
        mx.eval(h)
    tok = mx.argmax(model.lm_head(h[:, -1]), axis=-1)
    mx.eval(tok)
    before = streaming.stream_stats(model)
    with timers() as t:
        t0 = time.perf_counter()
        for _ in range(n):
            tok = mx.argmax(model.lm_head(model.model(tok[None], cache=cache)[:, -1]), axis=-1)
            mx.eval(tok)
        step = time.perf_counter() - t0
    after = streaming.stream_stats(model)
    ms = lambda s: 1000 * s / n
    hits, misses = after["hits"] - before["hits"], after["misses"] - before["misses"]
    return {"context": len(ids), "steps": n, "tok_s": n / step, "step_ms": ms(step),
            "sync_ms": ms(t["sync"]), "load_ms": ms(t["load"]), "pread_ms": ms(t["pread"]),
            "other_ms": ms(step - t["sync"] - t["load"]),
            "hit_rate": hits / max(1, hits + misses), "misses_per_step": misses / n}


def main():
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    model, _ = streaming.load_streaming(PACK, 32)
    ids = np.load(SESSION)["ids"]
    rows = []
    for ctx in (256, 32768):
        r = profile(model, ids[:ctx])
        rows.append(r)
        print({k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    OUT.write_text(json.dumps({"pack": PACK, "slots": 32, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
