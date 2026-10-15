"""Where streaming decode time goes (Task S5): per decode step, summed over the MoE layers.

python3 -m scripts.profile_stream              # pruned-50 pack, slots=32, at short and 32k context
python3 -m scripts.profile_stream --resident   # same, pruned-50 fully resident: no sync, no loads
python3 -m scripts.profile_stream --split      # force slot writes to finish inside load (write_ms); adds 32k with pre-read expert files
python3 -m scripts.profile_stream --ab         # per-expert vs batched slot writes, fresh model each, tokens must match

  sync_ms   time blocked in np.array(indices): the GPU finishing everything queued
            since the last layer (attention + previous layers' compute) plus the round-trip
  load_ms   wall time in ExpertStore.load (slot bookkeeping + preads + slot writes)
  pread_ms  time inside os.pread, summed over pool threads (an upper bound on its wall share)
  write_ms  (--split only) GPU time to finish the slot writes, evaluated right after each load
  other_ms  step - sync - load - write: graph building, lm_head, argmax
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
MODEL = "models/Laguna-XS-2.1-pruned-50"
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
def timers(split=False):
    t = {"sync": 0.0, "load": 0.0, "pread": 0.0, "write": 0.0}
    S = streaming.ExpertStore
    load, read = S.load, S._read

    def timed_load(self, experts):
        t0 = time.perf_counter()
        r = load(self, experts)
        t1 = time.perf_counter()
        t["load"] += t1 - t0
        if split:
            mx.eval(self.glu.parameters())
            t["write"] += time.perf_counter() - t1
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


def warm(pack):
    """Read every expert file once so the decode steps start from a warm page cache."""
    for f in sorted(Path(pack, "experts").glob("*.bin")):
        with open(f, "rb") as fh:
            while fh.read(64 << 20):
                pass


def profile(model, ids, n=32, chunk=2048, split=False, warm_pack=None):
    """Prefill `ids`, then time n greedy decode steps. Returns per-step ms."""
    cache = model.make_cache()
    for a in range(0, len(ids), chunk):
        h = model.model(mx.array(ids[None, a : a + chunk]), cache=cache)
        mx.eval(h)
    tok = mx.argmax(model.lm_head(h[:, -1]), axis=-1)
    mx.eval(tok)
    if warm_pack:
        warm(warm_pack)
    before = streaming.stream_stats(model)
    toks = []
    with timers(split) as t:
        t0 = time.perf_counter()
        for _ in range(n):
            tok = mx.argmax(model.lm_head(model.model(tok[None], cache=cache)[:, -1]), axis=-1)
            mx.eval(tok)
            toks.append(tok.item())
        step = time.perf_counter() - t0
    after = streaming.stream_stats(model)
    ms = lambda s: 1000 * s / n
    hits, misses = after["hits"] - before["hits"], after["misses"] - before["misses"]
    return {"context": len(ids), "steps": n, "tok_s": n / step, "step_ms": ms(step),
            "sync_ms": ms(t["sync"]), "load_ms": ms(t["load"]), "pread_ms": ms(t["pread"]),
            "write_ms": ms(t["write"]), "other_ms": ms(step - t["sync"] - t["load"] - t["write"]),
            "hit_rate": hits / max(1, hits + misses), "misses_per_step": misses / n,
            "warm": bool(warm_pack), "tokens": toks}


def _show(r):
    print({k: round(v, 2) if isinstance(v, float) else v for k, v in r.items() if k != "tokens"}, flush=True)


def main():
    import argparse
    import gc

    from mlx_lm import load

    p = argparse.ArgumentParser()
    p.add_argument("--resident", action="store_true")
    p.add_argument("--split", action="store_true")
    p.add_argument("--ab", action="store_true")
    a = p.parse_args()
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    ids = np.load(SESSION)["ids"]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    if a.ab:
        res["ab"] = {}
        for batched in (False, True):
            streaming.ExpertStore.batch_writes = batched
            model = streaming.load_streaming(PACK, 32)[0]
            rows = [profile(model, ids[:ctx]) for ctx in (256, 32768)]
            for r in rows:
                _show(r)
            res["ab"]["batched" if batched else "per_expert"] = rows
            del model
            gc.collect()
            mx.clear_cache()  # one model at a time
        res["ab"]["tokens_identical"] = all(
            x["tokens"] == y["tokens"] for x, y in zip(res["ab"]["per_expert"], res["ab"]["batched"]))
        print("tokens identical:", res["ab"]["tokens_identical"], flush=True)
    else:
        model = load(MODEL)[0] if a.resident else streaming.load_streaming(PACK, 32)[0]
        runs = [(256, None), (32768, None)] + ([(32768, PACK)] if a.split else [])
        rows = []
        for ctx, w in runs:
            rows.append(profile(model, ids[:ctx], split=a.split, warm_pack=w))
            _show(rows[-1])
        if a.resident:
            res["resident"] = {"model": MODEL, "rows": rows}
        elif a.split:
            res["split"] = {"pack": PACK, "slots": 32, "rows": rows}
        else:
            res.update(pack=PACK, slots=32, rows=rows)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
