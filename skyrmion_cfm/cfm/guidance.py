"""Condition masking utilities for classifier-free guidance."""

from __future__ import annotations

from typing import Any

import torch


TIME_KEYS = {"t_end_index", "t_end_ns", "t_end_s", "dt_index", "dt_s", "dt_scale"}
SPATIAL_KEYS = {"defect_field", "j_field", "control_grid"}
CONTROL_KEYS = {"b_t", "current_a_m2"}
MATERIAL_KEYS = {
    "temp_k",
    "alpha",
    "aex_j_per_m",
    "dind_j_per_m2",
    "ku1_j_per_m3",
    "msat_a_per_m",
    "pol_eff",
    "epsilon_prime",
    "fixed_layer_x",
    "fixed_layer_y",
    "fixed_layer_z",
}


def _batch_size(cond: dict[str, torch.Tensor]) -> int | None:
    for value in cond.values():
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
    return None


def _zero_like_condition(value: torch.Tensor, key: str, keep_time: bool) -> torch.Tensor:
    if keep_time and key in TIME_KEYS:
        return value
    if key in {"t_end_index", "dt_index", "dt_scale"}:
        return torch.zeros_like(value)
    return torch.zeros_like(value)


def make_unconditional_cond(
    cond: dict[str, torch.Tensor],
    cfg: dict[str, Any] | None = None,
    *,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return a condition dict with selected conditioning channels zeroed.

    ``mask`` optionally selects which batch rows become unconditional. When it
    is omitted the whole batch is converted.
    """
    cfg = cfg or {}
    keep_time = bool(cfg.get("keep_time", True))
    drop_spatial = bool(cfg.get("drop_spatial", True))
    drop_controls = bool(cfg.get("drop_controls", True))
    drop_material = bool(cfg.get("drop_material", False))
    extra_drop = set(cfg.get("drop_keys", []) or [])
    bsz = _batch_size(cond)
    if mask is not None:
        mask = mask.bool()
        if bsz is not None and mask.shape[:1] != (bsz,):
            raise ValueError("classifier-free dropout mask must match batch size")
    out: dict[str, torch.Tensor] = {}
    for key, value in cond.items():
        if value is None or not torch.is_tensor(value):
            out[key] = value
            continue
        should_drop = key in extra_drop
        should_drop = should_drop or (drop_spatial and key in SPATIAL_KEYS)
        should_drop = should_drop or (drop_controls and key in CONTROL_KEYS)
        should_drop = should_drop or (drop_material and key in MATERIAL_KEYS)
        should_drop = should_drop or ((not keep_time) and key in TIME_KEYS)
        if not should_drop:
            out[key] = value
            continue
        dropped = _zero_like_condition(value, key, keep_time)
        if mask is None or value.ndim == 0:
            out[key] = dropped
            continue
        view = mask.reshape(mask.shape[0], *([1] * (value.ndim - 1)))
        out[key] = torch.where(view, dropped, value)
    return out


def apply_cfg_dropout(
    cond: dict[str, torch.Tensor],
    cfg: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    cfg = cfg or {}
    if not bool(cfg.get("enabled", False)):
        return cond
    p = float(cfg.get("p", 0.0))
    if p <= 0.0:
        return cond
    bsz = _batch_size(cond)
    if bsz is None:
        return cond
    device = next(v.device for v in cond.values() if torch.is_tensor(v))
    mask = torch.rand(bsz, device=device) < p
    if not bool(mask.any()):
        return cond
    return make_unconditional_cond(cond, cfg, mask=mask)
