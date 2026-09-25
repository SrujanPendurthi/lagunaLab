"""Opt-in instrumentation. Nothing here is imported by the model itself."""
import mlx.core as mx
import numpy as np


class RoutingRecorder:
    """Count expert selections per MoE layer while active.

        with RoutingRecorder(model) as rec:
            model(ids)
        rec.counts[layer]  # int64[num_experts]
        rec.mass[layer]    # float64[num_experts], summed routing weights

    Set rec.token_mask (bool over B*L of the next forward) to count only some tokens.
    Recording syncs to numpy per layer, so it is slow; never leave it on for timing runs.
    """

    def __init__(self, model):
        self.moe = {i: l.mlp for i, l in enumerate(model.layers) if hasattr(l.mlp, "route")}
        E = model.args.num_experts
        self.counts = {i: np.zeros(E, np.int64) for i in self.moe}
        self.mass = {i: np.zeros(E, np.float64) for i in self.moe}
        self.token_mask = None

    def __enter__(self):
        for i, mlp in self.moe.items():
            mlp.route = self._wrap(i, mlp)
        return self

    def __exit__(self, *exc):
        for mlp in self.moe.values():
            del mlp.route

    def _wrap(self, i, mlp):
        orig = type(mlp).route

        def route(x):
            inds, w = orig(mlp, x)
            k = inds.shape[-1]
            ii = np.array(inds).reshape(-1, k)
            ww = np.array(w.astype(mx.float32)).reshape(-1, k)
            if self.token_mask is not None:
                ii, ww = ii[self.token_mask], ww[self.token_mask]
            np.add.at(self.counts[i], ii.ravel(), 1)
            np.add.at(self.mass[i], ii.ravel(), ww.ravel())
            return inds, w

        return route
