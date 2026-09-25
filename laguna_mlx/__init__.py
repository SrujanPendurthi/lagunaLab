"""Importing this package registers model_type "laguna" with mlx-lm (load / convert)."""
import sys

from . import laguna

sys.modules.setdefault("mlx_lm.models.laguna", laguna)
