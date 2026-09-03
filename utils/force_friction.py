"""
Friction coefficient from GS Mini 6D force: shear = dims 0,1, normal = dim 2.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

Array = Union[np.ndarray, "torch.Tensor"]


def friction_coefficient_from_shear_normal(
    shear_x: Array,
    shear_y: Array,
    normal: Array,
    eps: float = 1e-3,
) -> Array:
    """μ = sqrt(shear_x² + shear_y²) / (|normal| + eps)."""
    _is_torch = torch is not None and isinstance(shear_x, torch.Tensor)
    if _is_torch:
        smag = torch.sqrt(shear_x * shear_x + shear_y * shear_y + 1e-30)
        denom = torch.abs(normal) + eps
        return smag / denom
    smag = np.sqrt(np.square(shear_x) + np.square(shear_y))
    denom = np.abs(normal) + eps
    return smag / denom
