"""One-shot expert pruning for Laguna MoE layers: saliency, keep-sets, masking, materialization.

REAP (Cerebras 2025, arXiv 2510.13999): saliency of expert j = mean over tokens routed to j
of gate_j(x) * ||f_j(x)||, where f_j is the expert's output before gate weighting.
"""
import glob
import json
import re
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def accumulate(stats, inds, w, norms, mask=None):
    """Add one forward's routing to stats. inds/w/norms: [N, k] numpy; mask: [N] bool."""
    if mask is not None:
        inds, w, norms = inds[mask], w[mask], norms[mask]
    ii = inds.ravel()
    np.add.at(stats["count"], ii, 1)
    np.add.at(stats["gate"], ii, w.ravel())
    np.add.at(stats["reap"], ii, (w * norms).ravel())


class _Tap(nn.Module):
    """Wraps a SwitchGLU to report per-selection expert outputs to the recorder."""

    def __init__(self, inner, rec, layer):
        super().__init__()
        self.inner = inner
        self._ctx = [rec, layer]  # list: keeps the recorder out of the parameter tree

    def __call__(self, x, indices):
        y = self.inner(x, indices)
        rec, layer = self._ctx
        rec._observe(layer, indices, y)
        return y


class ExpertSaliencyRecorder:
    """Per MoE layer and expert: routed count, gate mass, REAP sum (gate * ||expert output||).

        with ExpertSaliencyRecorder(model) as rec:
            model(ids)
        rec.save("results/expert_stats.npz")

    Set rec.token_mask (bool over B*L of the next forward) to count only some tokens.
    Syncs to numpy per layer: slow, use for calibration only.
    """

    def __init__(self, model):
        self.moe = {i: l.mlp for i, l in enumerate(model.layers) if hasattr(l.mlp, "route")}
        E = model.args.num_experts
        self.stats = {i: {"count": np.zeros(E, np.int64), "gate": np.zeros(E), "reap": np.zeros(E)}
                      for i in self.moe}
        self.token_mask = None
        self._w = {}

    def __enter__(self):
        for i, mlp in self.moe.items():
            mlp.route = self._wrap_route(i, mlp)
            mlp.switch_mlp = _Tap(mlp.switch_mlp, self, i)
        return self

    def __exit__(self, *exc):
        for mlp in self.moe.values():
            if "route" in vars(mlp):
                del mlp.route
            if isinstance(mlp.switch_mlp, _Tap):
                mlp.switch_mlp = mlp.switch_mlp.inner

    def _wrap_route(self, i, mlp):
        orig = type(mlp).route

        def route(x):
            inds, w = orig(mlp, x)
            self._w[i] = w
            return inds, w

        return route

    def _observe(self, i, inds, y):
        k = inds.shape[-1]
        w = self._w.pop(i)
        norms = mx.linalg.norm(y.astype(mx.float32), axis=-1)
        accumulate(self.stats[i], np.array(inds).reshape(-1, k),
                   np.array(w.astype(mx.float32)).reshape(-1, k),
                   np.array(norms).reshape(-1, k), self.token_mask)

    def save(self, path):
        np.savez(path, **{f"{i}_{n}": a for i, s in self.stats.items() for n, a in s.items()})


def load_stats(path):
    z = np.load(path)
    out = {}
    for key in z.files:
        i, name = key.split("_", 1)
        out.setdefault(int(i), {})[name] = z[key]
    return out


def scores(stats, criterion, seed=0):
    """Per-layer saliency, higher = more important."""
    rng = np.random.default_rng(seed)
    out = {}
    for i in sorted(stats):
        s = stats[i]
        if criterion == "reap":
            out[i] = s["reap"] / np.maximum(s["count"], 1)
        elif criterion == "frequency":
            out[i] = s["count"].astype(np.float64)
        elif criterion == "gate":
            out[i] = s["gate"].copy()
        elif criterion == "random":
            out[i] = rng.random(len(s["count"]))
        else:
            raise ValueError(f"unknown criterion {criterion!r}")
    return out


def keep_sets(sal, ratio, allocation="uniform", min_keep=16):
    """Experts to keep per layer (sorted). ratio = fraction of experts pruned.

    min_keep applies to allocation="global" only; uniform ignores it.
    """
    if allocation == "uniform":
        out = {}
        for i, v in sal.items():
            n = len(v) - int(round(ratio * len(v)))
            out[i] = np.sort(np.argsort(-v, kind="stable")[:n])
        return out
    if allocation == "global":
        total = sum(len(v) for v in sal.values())
        budget = total - int(round(ratio * total))
        if budget < min_keep * len(sal):
            raise ValueError(f"budget {budget} below per-layer minimum {min_keep} x {len(sal)} layers")
        norm = {i: v / max(float(v.sum()), 1e-12) for i, v in sal.items()}
        order = {i: np.argsort(-norm[i], kind="stable") for i in sal}
        keep = {i: list(order[i][:min_keep]) for i in sal}
        rest = sorted(((norm[i][e], -i, -e) for i in sal for e in order[i][min_keep:]), reverse=True)
        for _, ni, ne in rest[: budget - min_keep * len(sal)]:
            keep[-ni].append(-ne)
        return {i: np.sort(np.array(v, dtype=np.int64)) for i, v in keep.items()}
    raise ValueError(f"unknown allocation {allocation!r}")


def apply_mask(model, keep):
    """Route only to kept experts: pruned experts' selection bias becomes -inf. Returns undo().

    Validates every layer's keep-set (size >= top_k, indices in range and unique) before
    mutating any layer, so a bad keep-set anywhere raises without partially masking the model.
    """
    checked = {}
    for i, k in keep.items():
        mlp = model.layers[i].mlp
        k = np.asarray(k)
        E = mlp.e_score_correction_bias.shape[0]
        if len(k) < mlp.top_k:
            raise ValueError(f"layer {i}: keeping {len(k)} experts < top_k {mlp.top_k}")
        if k.size and (k.min() < 0 or k.max() >= E):
            raise ValueError(f"layer {i}: keep indices out of range [0, {E})")
        if len(set(k.tolist())) != len(k):
            raise ValueError(f"layer {i}: keep indices must be unique")
        checked[i] = (mlp, k)

    saved = {}
    for i, (mlp, k) in checked.items():
        b = mlp.e_score_correction_bias
        saved[i] = b
        allowed = np.zeros(b.shape[0], bool)
        allowed[k] = True
        mlp.e_score_correction_bias = mx.where(mx.array(allowed), b.astype(mx.float32), -mx.inf)

    def undo():
        for i, b in saved.items():
            model.layers[i].mlp.e_score_correction_bias = b

    return undo
