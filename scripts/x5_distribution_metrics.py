#!/usr/bin/env python3
"""Complementary metrics for x5 same-condition sample sets."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _angular_distance_matrix_deg(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor,
    *,
    identical_sets: bool = False,
) -> torch.Tensor:
    """Return pairwise masked mean full-vector angles for two field ensembles."""
    if left.ndim != 4 or right.ndim != 4 or left.shape[1:] != right.shape[1:]:
        raise ValueError(f"incompatible field ensembles: {left.shape}, {right.shape}")
    if mask.ndim != 2 or tuple(mask.shape) != tuple(left.shape[-2:]):
        raise ValueError(f"incompatible geometry mask: {mask.shape}, {left.shape}")
    valid = mask.to(dtype=torch.bool)
    valid_count = int(valid.sum().item())
    if valid_count <= 0:
        raise ValueError("geometry mask contains no valid cells")

    left = F.normalize(left.float(), dim=1, eps=1.0e-8)
    right = F.normalize(right.float(), dim=1, eps=1.0e-8)
    left_pairs = left[:, None].expand(-1, right.shape[0], -1, -1, -1)
    right_pairs = right[None, :].expand(left.shape[0], -1, -1, -1, -1)
    dot = (left_pairs * right_pairs).sum(dim=2).clamp(-1.0, 1.0)
    cross_norm = torch.linalg.vector_norm(
        torch.cross(left_pairs, right_pairs, dim=2),
        dim=2,
    )
    angles = torch.rad2deg(torch.atan2(cross_norm, dot))
    distances = angles[..., valid].mean(dim=-1)
    if identical_sets:
        if distances.shape[0] != distances.shape[1]:
            raise ValueError("identical-set distance matrix must be square")
        distances.fill_diagonal_(0.0)
    return distances


@torch.inference_mode()
def angular_energy_distance_from_files(
    model_samples_path: Path,
    mumax_targets_path: Path,
    geometry_mask_path: Path,
    *,
    num_model: int = 5,
    device: str | torch.device = "cpu",
) -> dict[str, float | int | str]:
    """Compute the biased angular energy distance between two empirical sets.

    The V-statistic includes zero self-distances.  All paper-facing comparisons
    use five model outputs and five MuMax repeats, so finite-ensemble bias is
    matched across methods.
    """
    if num_model < 2:
        raise ValueError("angular energy distance requires at least two model samples")
    model_array = np.load(model_samples_path, mmap_mode="r")
    target_array = np.load(mumax_targets_path, mmap_mode="r")
    mask_array = np.load(geometry_mask_path, mmap_mode="r")
    if model_array.shape[0] < num_model:
        raise ValueError(
            f"requested {num_model} model samples but only {model_array.shape[0]} exist"
        )
    if target_array.shape[0] < 2:
        raise ValueError("angular energy distance requires at least two MuMax repeats")

    compute_device = torch.device(device)
    model = torch.from_numpy(
        np.asarray(model_array[:num_model], dtype=np.float32)
    ).to(compute_device)
    target = torch.from_numpy(
        np.asarray(target_array, dtype=np.float32)
    ).to(compute_device)
    # ``mmap_mode='r'`` yields a read-only array.  Copy the small mask so
    # torch never receives a non-writable NumPy view.
    mask = torch.from_numpy(
        np.array(mask_array, dtype=np.bool_, copy=True)
    ).to(compute_device)
    mask = mask.squeeze()

    cross = _angular_distance_matrix_deg(model, target, mask)
    within_model = _angular_distance_matrix_deg(
        model, model, mask, identical_sets=True
    )
    within_target = _angular_distance_matrix_deg(
        target, target, mask, identical_sets=True
    )
    cross_mean = float(cross.mean().item())
    within_model_mean = float(within_model.mean().item())
    within_target_mean = float(within_target.mean().item())
    raw = 2.0 * cross_mean - within_model_mean - within_target_mean
    result: dict[str, float | int | str] = {
        "angular_energy_distance_deg": max(raw, 0.0),
        "cross_model_mumax_mean_deg": cross_mean,
        "within_model_mean_deg": within_model_mean,
        "within_mumax_mean_deg": within_target_mean,
        "model_samples": int(num_model),
        "mumax_samples": int(target.shape[0]),
        "estimator": "biased_v_statistic_with_zero_self_distances",
        "field_distance": "valid-cell mean full-vector angle in degrees",
    }
    if cross.shape[0] == cross.shape[1]:
        result["paired_model_mumax_mean_deg"] = float(
            torch.diagonal(cross).mean().item()
        )
    return result


def aggregate_clustered_mean(
    rows: list[dict[str, Any]],
    key: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Aggregate a condition metric with base-group clustered bootstrap CIs."""
    if not rows:
        raise ValueError("cannot aggregate an empty row collection")
    rng = np.random.default_rng(seed)
    segments = sorted({int(row["control_segment_index"]) for row in rows})

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in selected:
            grouped[str(row["base_id"])].append(float(row[key]))
        group_ids = sorted(grouped)
        values = np.asarray(
            [value for group_id in group_ids for value in grouped[group_id]],
            dtype=np.float64,
        )
        bootstrap = np.empty(iterations, dtype=np.float64)
        for index in range(iterations):
            sampled = rng.integers(0, len(group_ids), size=len(group_ids))
            draw = np.asarray(
                [
                    value
                    for group_index in sampled
                    for value in grouped[group_ids[int(group_index)]]
                ],
                dtype=np.float64,
            )
            bootstrap[index] = draw.mean()
        return {
            "conditions": int(len(values)),
            "base_groups": int(len(group_ids)),
            "mean": float(values.mean()),
            "bootstrap_95ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
            "bootstrap_unit": "base group; segments kept together",
        }

    return {
        "metric": key,
        "combined": summarize(rows),
        "by_segment": {
            str(segment): summarize(
                [
                    row
                    for row in rows
                    if int(row["control_segment_index"]) == segment
                ]
            )
            for segment in segments
        },
    }
