"""Stream routed experts from SSD into a fixed number of in-memory slots per layer.

Same bits, same weights as the resident model: output is identical.
Design after TurboFieldfare (drumih/turbo-fieldfare): resident common weights,
per-layer expert files, LFU slot cache with recency tie-break, parallel preads.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.switch_layers import SwitchGLU

from laguna_mlx import laguna
from scripts.pack_experts import expert_from_bytes

_POOL = ThreadPoolExecutor(8)


class ExpertStore:
    """Slots for one layer: a quantized SwitchGLU with `slots` experts, refilled from disk."""

    # ponytail: A/B switch for Task S5 (one scatter per tensor vs one per expert); drop the loser once measured
    batch_writes = False

    def __init__(self, pack_dir, layout, layer, slots, group_size, bits, hidden, inter):
        self.layout, self.slots = layout, slots
        self.fd = os.open(Path(pack_dir) / "experts" / f"layer_{layer:02d}.bin", os.O_RDONLY)
        glu = SwitchGLU(hidden, inter, slots)
        # match the packed scales dtype (bf16 on the real model): no cast on write, no f32 promotion
        glu.set_dtype(getattr(mx, layout["arrays"]["gate_proj.scales"]["dtype"]))
        nn.quantize(glu, group_size=group_size, bits=bits)
        self.glu = glu
        self.slot_expert = [-1] * slots
        self.expert_slot = {}
        E = layout["num_experts"]
        self.freq, self.last = np.zeros(E, np.int64), np.zeros(E, np.int64)
        self.clock = 0
        self.hits = self.misses = self.bytes_read = 0

    def _read(self, e):
        s = self.layout["stride"]
        return e, os.pread(self.fd, s, e * s)

    def _victims(self, n, keep):
        free = [i for i, e in enumerate(self.slot_expert) if e == -1]
        used = [i for i, e in enumerate(self.slot_expert) if e != -1 and e not in keep]
        used.sort(key=lambda i: (self.freq[self.slot_expert[i]], self.last[self.slot_expert[i]]))
        return (free + used)[:n]

    def load(self, experts):
        """Make every expert in `experts` (len <= slots) resident; return {expert: slot}."""
        self.clock += 1
        keep = set(experts)
        missing = [e for e in experts if e not in self.expert_slot]
        self.hits += len(experts) - len(missing)
        self.misses += len(missing)
        self.freq[list(experts)] += 1
        self.last[list(experts)] = self.clock
        if missing:
            slots, parts = [], []
            for slot, (e, buf) in zip(self._victims(len(missing), keep), _POOL.map(self._read, missing)):
                old = self.slot_expert[slot]
                if old != -1:
                    del self.expert_slot[old]
                self.slot_expert[slot], self.expert_slot[e] = e, slot
                self.bytes_read += len(buf)
                slots.append(slot)
                parts.append(expert_from_bytes(buf, self.layout))
            for name in parts[0]:
                proj, part = name.split(".")
                param = getattr(getattr(self.glu, proj), part)
                if self.batch_writes:
                    param[mx.array(slots)] = mx.stack([p[name] for p in parts])
                else:
                    for slot, p in zip(slots, parts):
                        param[slot] = p[name]
        return {e: self.expert_slot[e] for e in experts}


class StreamingSwitchGLU(nn.Module):
    def __init__(self, store):
        super().__init__()
        self._store = [store]  # list: keeps the store out of the parameter tree

    def __call__(self, x, indices):
        store = self._store[0]
        idx = np.array(indices)
        experts = np.unique(idx).tolist()
        out = None
        # ponytail: a prefill chunk needing > slots experts runs several masked passes
        for g in range(0, len(experts), store.slots):
            group = experts[g : g + store.slots]
            table = np.zeros(store.layout["num_experts"], np.int32)
            member = np.zeros(store.layout["num_experts"], bool)
            for e, s in store.load(group).items():
                table[e], member[e] = s, True
            y = store.glu(x, mx.array(table[idx]))
            if len(experts) > store.slots:
                y = mx.where(mx.array(member[idx])[..., None], y, 0)
            out = y if out is None else out + y
        return out


def load_streaming(pack_dir, slots):
    from mlx_lm.utils import load_tokenizer

    pack_dir = Path(pack_dir)
    config = json.loads((pack_dir / "config.json").read_text())
    layout = json.loads((pack_dir / "layout.json").read_text())
    q = config["quantization"]
    model = laguna.Model(laguna.ModelArgs.from_dict(config))  # lazy: never evaluated before the swap

    def predicate(p, m):
        if p in q:
            return q[p]
        return hasattr(m, "to_quantized") and "switch_mlp" not in p

    nn.quantize(model, group_size=q["group_size"], bits=q["bits"], class_predicate=predicate)
    a = model.args
    for l in layout["layers"]:
        mlp = model.layers[l].mlp
        mlp.switch_mlp = StreamingSwitchGLU(ExpertStore(
            pack_dir, layout, l, slots, q["group_size"], q["bits"], a.hidden_size, a.moe_intermediate_size))
    model.load_weights(str(pack_dir / "resident.safetensors"), strict=True)
    mx.eval(model.parameters())
    tok = load_tokenizer(pack_dir) if (pack_dir / "tokenizer.json").exists() else None
    return model, tok


def stream_stats(model):
    stores = [l.mlp.switch_mlp._store[0] for l in model.layers
              if isinstance(getattr(l.mlp, "switch_mlp", None), StreamingSwitchGLU)]
    hits = sum(s.hits for s in stores)
    misses = sum(s.misses for s in stores)
    return {"hits": hits, "misses": misses, "hit_rate": hits / max(1, hits + misses),
            "bytes_read": sum(s.bytes_read for s in stores)}
