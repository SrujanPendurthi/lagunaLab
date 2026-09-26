"""DFlash speculative decoding for Laguna on MLX (greedy, lossless).

Draft: dense Laguna layers conditioned on target hidden states from
`target_layer_ids` (per-slice RMSNorm -> fc -> hidden_norm), injected as extra
K/V in every draft layer. One draft pass over [token, mask x n] proposes n tokens;
block slot i predicts the token at position(token) + i.
References: vLLM PR #46853 (laguna_dflash.py, qwen3_dflash.py);
speculators/models/dflash/core.py (training-time alignment).
"""
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask

from laguna_mlx import laguna
from laguna_mlx.cache import TrimmableWindowCache, make_trimmable_cache


class DFlashDraft(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        d = config["dflash_config"]
        self.target_layer_ids = list(d["target_layer_ids"])
        self.block_size = d["block_size"]
        self.mask_token_id = d["mask_token_id"]
        n = config["num_hidden_layers"]
        self.args = laguna.ModelArgs.from_dict({
            **config, "model_type": "laguna", "num_experts": 0, "num_experts_per_tok": 0,
            "moe_intermediate_size": 0, "shared_expert_intermediate_size": 0,
            "mlp_only_layers": list(range(n)),
            # draft config ships only rope_theta: default RoPE, full rotary (as vLLM reads it)
            "rope_parameters": config.get("rope_parameters")
            or {"rope_type": "default", "rope_theta": config["rope_theta"]},
        })
        h = self.args.hidden_size
        self.layers = [laguna.DecoderLayer(self.args, i) for i in range(n)]
        self.aux_hidden_norms = [nn.RMSNorm(h, eps=self.args.rms_norm_eps) for _ in self.target_layer_ids]
        self.fc = nn.Linear(h * len(self.target_layer_ids), h, bias=False)
        self.hidden_norm = nn.RMSNorm(h, eps=self.args.rms_norm_eps)
        self.norm = nn.RMSNorm(h, eps=self.args.rms_norm_eps)

    def make_cache(self):
        return [TrimmableWindowCache(self.args.sliding_window, slack=self.block_size) for _ in self.layers]

    def context_states(self, aux):
        x = mx.concatenate([norm(a) for norm, a in zip(self.aux_hidden_norms, aux)], axis=-1)
        return self.hidden_norm(self.fc(x))

    def add_context(self, ctx, cache):
        """Project context states to K/V in every draft layer and append them."""
        B, L, _ = ctx.shape
        for layer, c in zip(self.layers, cache):
            a = layer.self_attn
            x = layer.input_layernorm(ctx)
            k = a.k_norm(a.k_proj(x).reshape(B, L, a.n_kv_heads, -1)).transpose(0, 2, 1, 3)
            v = a.v_proj(x).reshape(B, L, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
            c.update_and_fetch(a.rope(k, offset=c.offset), v)

    def propose(self, target, token, cache, n_draft):
        ids = mx.array([[token] + [self.mask_token_id] * n_draft])
        h = target.model.embed_tokens(ids)
        mask = create_attention_mask(h, cache[0], window_size=self.args.sliding_window)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        drafts = mx.argmax(target.lm_head(self.norm(h[:, 1:])), axis=-1)[0].tolist()
        for c in cache:
            c.trim(n_draft + 1)  # only context K/V persists between steps
        return drafts

    def sanitize(self, weights):
        q = self.args.num_attention_heads * self.args.head_dim
        kv = self.args.num_key_value_heads * self.args.head_dim
        for i in range(len(self.layers)):
            p = f"layers.{i}.self_attn."
            qkv = weights.pop(p + "qkv_proj.weight", None)
            if qkv is not None:
                weights[p + "q_proj.weight"] = qkv[:q]
                weights[p + "k_proj.weight"] = qkv[q : q + kv]
                weights[p + "v_proj.weight"] = qkv[q + kv :]
        return weights


def load_draft(path, bits=None):
    path = Path(path)
    d = DFlashDraft(json.loads((path / "config.json").read_text()))
    d.load_weights(list(d.sanitize(mx.load(str(path / "model.safetensors"))).items()))
    if bits:
        nn.quantize(d, group_size=64, bits=bits)
    mx.eval(d.parameters())
    return d


def _target_step(target, ids, cache, layer_ids):
    cap = dict.fromkeys(layer_ids)
    h = target.model(mx.array([ids]), cache=cache, capture=cap)
    return target.lm_head(h)[0], [cap[i] for i in layer_ids]


def generate_greedy(target, prompt, max_tokens, eos=()):
    prompt = [int(t) for t in prompt]  # accept numpy ints
    cache = target.make_cache()
    logits = target(mx.array([prompt]), cache=cache)[0, -1]
    out = []
    while True:
        t = int(mx.argmax(logits))
        out.append(t)
        if t in eos or len(out) >= max_tokens:
            return out
        logits = target(mx.array([[t]]), cache=cache)[0, -1]


def generate_dflash(target, draft, prompt, max_tokens, eos=(), n_draft=None, prefill_step=2048):
    """Greedy speculative decoding; output equals generate_greedy."""
    n_draft = n_draft or draft.block_size - 1
    prompt = [int(t) for t in prompt]  # accept numpy ints
    ids_layers = draft.target_layer_ids
    tcache, dcache = make_trimmable_cache(target, slack=n_draft + 1), draft.make_cache()
    for a in range(0, len(prompt), prefill_step):
        logits, aux = _target_step(target, prompt[a : a + prefill_step], tcache, ids_layers)
        draft.add_context(draft.context_states(aux), dcache)
    token = int(mx.argmax(logits[-1]))
    out, steps, accepted = [], 0, 0
    while True:
        drafts = draft.propose(target, token, dcache, n_draft)
        logits, aux = _target_step(target, [token] + drafts, tcache, ids_layers)
        g = mx.argmax(logits, axis=-1).tolist()  # g[j] = greedy token at position(token) + j + 1
        k = 0
        while k < n_draft and drafts[k] == g[k]:
            k += 1
        steps, accepted = steps + 1, accepted + k
        for t in [token] + drafts[:k]:
            out.append(t)
            if t in eos or len(out) >= max_tokens:
                return out, {"steps": steps, "drafted": steps * n_draft, "accepted": accepted,
                             "mean_accepted": accepted / steps}
        for c in tcache:
            c.trim(n_draft - k)
        draft.add_context(draft.context_states([a[:, : k + 1] for a in aux]), dcache)
        token = g[k]


class NgramIndex:
    """Prompt-lookup drafter: g-gram -> most recent start, updated as tokens arrive.

    `propose(n)` equals scanning back for the latest earlier occurrence of the last
    g tokens and copying up to n tokens after it (never past the current end).
    Only starts s <= len(ids) - g - 1 are indexed, so the key never matches itself.
    """

    def __init__(self, g):
        self.g, self.ids, self.last = g, [], {}

    def extend(self, tokens):
        g, ids = self.g, self.ids
        for t in tokens:
            ids.append(t)
            s = len(ids) - g - 1
            if s >= 0:
                self.last[tuple(ids[s : s + g])] = s

    def propose(self, n):
        t, g = len(self.ids), self.g
        s = self.last.get(tuple(self.ids[t - g :])) if t >= g else None
        return [] if s is None else self.ids[s + g : min(s + g + n, t)]


def generate_lookup(target, prompt, max_tokens, eos=(), g=4, n_draft=7, adaptive=False, prefill_step=2048):
    """Greedy prompt-lookup speculative decoding; output equals generate_greedy.

    adaptive: cap each draft at min(n_draft, 1 + previous step's accepted count).
    """
    prompt = [int(t) for t in prompt]  # accept numpy ints
    cache = make_trimmable_cache(target, slack=n_draft + 1)
    for a in range(0, len(prompt), prefill_step):
        logits = target(mx.array([prompt[a : a + prefill_step]]), cache=cache)[0]
    token = int(mx.argmax(logits[-1]))
    index = NgramIndex(g)
    index.extend(prompt + [token])  # ids = prompt + emitted + pending token
    out, cap = [], n_draft
    stats = {"steps": 0, "drafted": 0, "accepted": 0, "steps_with_draft": 0}
    while True:
        drafts = index.propose(cap)  # [] -> plain 1-token step
        pred = mx.argmax(target(mx.array([[token] + drafts]), cache=cache)[0], axis=-1).tolist()
        k = 0
        while k < len(drafts) and drafts[k] == pred[k]:
            k += 1
        stats["steps"] += 1
        stats["drafted"] += len(drafts)
        stats["accepted"] += k
        stats["steps_with_draft"] += bool(drafts)
        for t in [token] + drafts[:k]:
            out.append(t)
            if t in eos or len(out) >= max_tokens:
                return out, {**stats, "mean_accepted": stats["accepted"] / stats["steps"]}
        for c in cache:
            c.trim(len(drafts) - k)
        token = pred[k]
        index.extend(drafts[:k] + [token])
        if adaptive:
            cap = min(n_draft, 1 + k)
