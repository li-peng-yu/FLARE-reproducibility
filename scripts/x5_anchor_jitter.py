#!/usr/bin/env python3
"""Deterministic, geometry-preserving test-time jitter for x5 anchors."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F


def _gaussian_kernel(
    sigma_px: float, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma_px)))
    coordinate = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-0.5 * (coordinate / sigma_px).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d[None, None].repeat(3, 1, 1, 1)


def apply_anchor_jitter(
    fields: torch.Tensor,
    *,
    rms_degrees: float,
    correlation_px: float,
    seeds: Sequence[int],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Apply a smooth tangent perturbation with an exact valid-cell RMS angle.

    The perturbation is sampled independently for every batch item, projected
    into the pointwise tangent plane of S2, globally rescaled to the requested
    valid-cell RMS angle, and mapped back with the spherical exponential map.
    Invalid geometry cells remain exactly zero.
    """
    if fields.ndim != 4 or fields.shape[1] != 3:
        raise ValueError(f"expected [B,3,H,W] fields, found {tuple(fields.shape)}")
    if len(seeds) != len(fields):
        raise ValueError(f"expected {len(fields)} seeds, found {len(seeds)}")
    if rms_degrees < 0.0 or correlation_px < 0.0:
        raise ValueError("jitter RMS and correlation length must be non-negative")

    work = fields.float()
    valid = work.square().sum(dim=1, keepdim=True).sqrt().gt(0.5)
    base = F.normalize(work, dim=1, eps=1.0e-8) * valid
    if rms_degrees == 0.0:
        reports = [
            {
                "seed": int(seed),
                "requested_rms_degrees": 0.0,
                "realized_rms_degrees": 0.0,
                "maximum_degrees": 0.0,
            }
            for seed in seeds
        ]
        return base.to(dtype=fields.dtype), reports

    kernel = (
        _gaussian_kernel(
            correlation_px, device=work.device, dtype=work.dtype
        )
        if correlation_px > 0.0
        else None
    )
    padding = kernel.shape[-1] // 2 if kernel is not None else 0
    target_radians = math.radians(rms_degrees)
    outputs: list[torch.Tensor] = []
    reports: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        generator = torch.Generator(device=work.device)
        generator.manual_seed(int(seed))
        noise = torch.randn(
            (1, 3, work.shape[-2], work.shape[-1]),
            generator=generator,
            device=work.device,
            dtype=work.dtype,
        )
        if kernel is not None:
            noise = F.conv2d(
                F.pad(noise, (padding,) * 4, mode="reflect"),
                kernel,
                groups=3,
            )
        point = base[index : index + 1]
        mask = valid[index : index + 1]
        tangent = noise - (noise * point).sum(dim=1, keepdim=True) * point
        tangent = tangent * mask
        squared_angle = tangent.square().sum(dim=1, keepdim=True)
        denominator = mask.sum().clamp_min(1).to(dtype=work.dtype)
        current_rms = torch.sqrt(squared_angle.sum() / denominator)
        if not torch.isfinite(current_rms) or float(current_rms) <= 1.0e-12:
            raise RuntimeError("sampled anchor jitter has zero or non-finite norm")
        tangent = tangent * (target_radians / current_rms)
        angle = tangent.square().sum(dim=1, keepdim=True).sqrt()
        direction = tangent / angle.clamp_min(1.0e-12)
        perturbed = torch.cos(angle) * point + torch.sin(angle) * direction
        perturbed = F.normalize(perturbed, dim=1, eps=1.0e-8) * mask
        realized_rms = torch.sqrt((angle.square() * mask).sum() / denominator)
        reports.append(
            {
                "seed": int(seed),
                "requested_rms_degrees": float(rms_degrees),
                "realized_rms_degrees": float(torch.rad2deg(realized_rms)),
                "maximum_degrees": float(torch.rad2deg(angle[mask]).max()),
            }
        )
        outputs.append(perturbed)
    return torch.cat(outputs, dim=0).to(dtype=fields.dtype), reports

