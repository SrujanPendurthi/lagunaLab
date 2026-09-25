"""Tiny random Laguna config shared by tests: same structure as XS 2.1, toy sizes."""
import copy

import mlx.core as mx
import torch

from laguna_mlx import laguna
from tests.reference.configuration_laguna import LagunaConfig
from tests.reference.modeling_laguna import LagunaForCausalLM

ROPE = {
    "full_attention": {
        "rope_theta": 500000.0, "rope_type": "yarn", "factor": 32.0,
        "original_max_position_embeddings": 8192, "beta_slow": 1.0, "beta_fast": 64.0,
        "attention_factor": 1.3465735902799727, "partial_rotary_factor": 0.5,
    },
    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
}

TINY = dict(
    model_type="laguna", vocab_size=100352, hidden_size=64, intermediate_size=128,
    num_hidden_layers=4, num_attention_heads=4, num_attention_heads_per_layer=[4, 8, 8, 8],
    num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
    num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
    shared_expert_intermediate_size=32, norm_topk_prob=True, mlp_only_layers=[0],
    decoder_sparse_step=1, gating="per-head", sliding_window=8,
    layer_types=["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
    moe_routed_scaling_factor=2.5, max_position_embeddings=262144, rope_parameters=ROPE,
    tie_word_embeddings=False,
)


def torch_model(seed=0):
    torch.manual_seed(seed)
    kw = copy.deepcopy({k: v for k, v in TINY.items() if k != "model_type"})
    cfg = LagunaConfig(**kw)
    cfg._attn_implementation = "eager"
    m = LagunaForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith("norm.weight"):
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:  # includes e_score_correction_bias, so biased selection is exercised
                p.copy_(0.05 * torch.randn_like(p))
    return m


def to_disk_names(model):
    """HF in-memory tensors -> the per-expert layout Laguna checkpoints ship on disk."""
    E, I = TINY["num_experts"], TINY["moe_intermediate_size"]
    out = {}
    for k, v in model.state_dict().items():
        v = v.detach().float().numpy()
        if k.endswith("mlp.experts.gate_up_proj"):
            p = k[: -len("gate_up_proj")]
            for e in range(E):
                out[f"{p}{e}.gate_proj.weight"] = v[e, :I]
                out[f"{p}{e}.up_proj.weight"] = v[e, I:]
        elif k.endswith("mlp.experts.down_proj"):
            p = k[: -len("down_proj")]
            for e in range(E):
                out[f"{p}{e}.down_proj.weight"] = v[e]
        elif k.endswith("mlp.gate.e_score_correction_bias"):
            out[k.replace("gate.e_score", "experts.e_score")] = v
        else:
            out[k] = v
    return out


def mlx_model(tmodel):
    m = laguna.Model(laguna.ModelArgs.from_dict(copy.deepcopy(TINY)))
    w = m.sanitize({k: mx.array(v) for k, v in to_disk_names(tmodel).items()})
    m.load_weights(list(w.items()))
    mx.eval(m.parameters())
    return m
