from __future__ import annotations

from typing import Any

import torch

from skyrmion_cfm.cfm.bridges import (
    CartBridge,
    RFMBridge,
    RotationVector2DBridge,
    RotationVectorBridge,
)


def spatial_weight(
    mask_or_cond: torch.Tensor | dict[str, Any] | None,
    ref: torch.Tensor,
) -> torch.Tensor | None:
    if mask_or_cond is None:
        return None
    mask = (
        mask_or_cond.get("defect_field")
        if isinstance(mask_or_cond, dict)
        else mask_or_cond
    )
    if mask is None:
        return None
    weight = mask.to(device=ref.device, dtype=ref.dtype)
    if weight.ndim == ref.ndim + 1 and weight.shape[1] == 1:
        weight = weight[:, 0]
    if weight.ndim == ref.ndim:
        if ref.ndim >= 4:
            weight = weight[:, 0] if weight.shape[1] == 1 else weight.mean(dim=1)
    if weight.ndim == ref.ndim:
        pass
    elif weight.ndim == ref.ndim - 1:
        if (
            weight.shape[0] != ref.shape[0]
            and tuple(weight.shape[-2:]) == tuple(ref.shape[-2:])
        ):
            weight = weight.unsqueeze(0).expand(ref.shape[0], -1, -1)
    elif weight.ndim == ref.ndim - 2:
        weight = weight.unsqueeze(0).expand(ref.shape[0], -1, -1)
    else:
        raise ValueError(
            f"spatial mask shape {tuple(mask.shape)} is incompatible with {tuple(ref.shape)}"
        )
    spatial_shape_ok = tuple(weight.shape[-2:]) == tuple(ref.shape[-2:])
    if weight.shape[0] != ref.shape[0] or not spatial_shape_ok:
        if weight.shape[0] == 1 and ref.shape[0] > 1 and spatial_shape_ok:
            weight = weight.expand(ref.shape[0], -1, -1)
        else:
            raise ValueError(
                f"spatial mask shape {tuple(mask.shape)} is incompatible with {tuple(ref.shape)}"
            )
    return weight.clamp(0.0, 1.0)


def expand_spatial_weight(weight: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    while weight.ndim < ref.ndim:
        weight = weight.unsqueeze(1)
    return weight


def apply_spatial_constraint(
    x: torch.Tensor | None,
    mask_or_cond: torch.Tensor | dict[str, Any] | None,
    outside: torch.Tensor | float | None = None,
) -> torch.Tensor | None:
    if x is None:
        return None
    weight = spatial_weight(mask_or_cond, x)
    if weight is None:
        return x
    view = expand_spatial_weight(weight, x)
    if outside is None:
        return x * view
    if not torch.is_tensor(outside):
        outside = x.new_full((), float(outside))
    return x * view + outside.to(device=x.device, dtype=x.dtype) * (1.0 - view)


def masked_site_mean(
    values: torch.Tensor,
    mask_or_cond: torch.Tensor | dict[str, Any] | None,
) -> torch.Tensor:
    weight = spatial_weight(mask_or_cond, values)
    if weight is None:
        return values.mean(dim=tuple(range(1, values.ndim)))
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(1)
    reduce_dims = tuple(range(1, values.ndim))
    denom = weight.sum(dim=reduce_dims).clamp_min(1.0)
    return (values * weight).sum(dim=reduce_dims) / denom


def weighted_site_mean(
    values: torch.Tensor,
    mask_or_cond: torch.Tensor | dict[str, Any] | None,
    void_weight: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    magnetic = spatial_weight(mask_or_cond, values)
    denom_weight: torch.Tensor | None = None
    if magnetic is None:
        if valid_mask is None:
            return values.mean(dim=tuple(range(1, values.ndim)))
        weight = spatial_weight(valid_mask, values)
        denom_weight = weight
    else:
        weight = magnetic + float(void_weight) * (1.0 - magnetic)
        denom_weight = magnetic
        if valid_mask is not None:
            valid = spatial_weight(valid_mask, values)
            if valid is not None:
                weight = magnetic * valid + float(void_weight) * (1.0 - magnetic)
                denom_weight = magnetic * valid
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(1)
    if denom_weight is None:
        denom_weight = weight
    while denom_weight.ndim < values.ndim:
        denom_weight = denom_weight.unsqueeze(1)
    reduce_dims = tuple(range(1, values.ndim))
    denom = denom_weight.sum(dim=reduce_dims).clamp_min(1.0)
    return (values * weight).sum(dim=reduce_dims) / denom


def bridge_state_background(bridge, m_init: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if isinstance(bridge, (RotationVectorBridge, RotationVector2DBridge)):
        return torch.zeros_like(like)
    if isinstance(bridge, CartBridge) and getattr(bridge, "objective", "endpoint") == "residual":
        return torch.zeros_like(like)
    if isinstance(bridge, (CartBridge, RFMBridge)) and getattr(bridge, "source", "latent") in {
        "anchored",
        "identity",
    }:
        if like.shape[1] == m_init.shape[1]:
            return m_init.to(device=like.device, dtype=like.dtype)
    return torch.zeros_like(like)


def bridge_output_background(bridge, m_init: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if getattr(bridge, "source", "latent") in {"anchored", "identity"}:
        return m_init.to(device=like.device, dtype=like.dtype)
    return torch.zeros_like(like)
