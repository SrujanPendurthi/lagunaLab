"""Importing this package registers model_type "laguna" with mlx-lm (load / convert)."""
import sys

from . import laguna

sys.modules.setdefault("mlx_lm.models.laguna", laguna)


def wire_resident(resident):
    """Raise MLX's wired limit to the max working set, only for a fully resident model.

    The 18.4 GB full model needs it (runbook rule 7). Streaming fits without it, and a max
    wired limit pins memory the rest of the machine needs (it stalled video playback)."""
    if resident:
        import mlx.core as mx

        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
