from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from skyrmion_cfm.data.conditions import ConditionStats


STATS_SCHEMA_VERSION = 4


@dataclass
class TrainingStats:
    condition: ConditionStats
    dt_scales: list[int]
    omega_std: torch.Tensor
    omega_count: torch.Tensor
    dataset_root: str | None = None
    physics_kappa: float | None = None
    physics_kappa_count: int = 0
    schema_version: int = STATS_SCHEMA_VERSION

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "condition_mean": self.condition.mean.tolist(),
            "condition_std": self.condition.std.tolist(),
            "condition_keys": list(self.condition.keys),
            "dt_scales": self.dt_scales,
            "omega_std": self.omega_std.tolist(),
            "omega_count": self.omega_count.tolist(),
            "dataset_root": self.dataset_root,
            "physics_kappa": self.physics_kappa,
            "physics_kappa_count": self.physics_kappa_count,
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "TrainingStats":
        return cls(
            condition=ConditionStats(
                mean=torch.tensor(payload["condition_mean"], dtype=torch.float32),
                std=torch.tensor(payload["condition_std"], dtype=torch.float32),
                keys=tuple(payload.get("condition_keys", ())),
            ),
            dt_scales=[int(x) for x in payload["dt_scales"]],
            omega_std=torch.tensor(payload["omega_std"], dtype=torch.float32),
            omega_count=torch.tensor(payload["omega_count"], dtype=torch.long),
            dataset_root=payload.get("dataset_root"),
            physics_kappa=payload.get("physics_kappa"),
            physics_kappa_count=int(payload.get("physics_kappa_count", 0)),
            schema_version=int(payload.get("schema_version", 1)),
        )


def save_training_stats(path: str | Path, stats: TrainingStats) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats.to_json_dict(), f, indent=2)


def load_training_stats(path: str | Path) -> TrainingStats:
    with Path(path).open("r", encoding="utf-8") as f:
        return TrainingStats.from_json_dict(json.load(f))


def estimate_omega_stats(dataset, dt_scales: list[int], max_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate per-component omega std for each configured time scale."""
    n_scales = len(dt_scales)
    sums = torch.zeros(n_scales, 3, dtype=torch.float64)
    sq_sums = torch.zeros(n_scales, 3, dtype=torch.float64)
    counts = torch.zeros(n_scales, dtype=torch.long)
    n = min(int(max_samples), len(dataset))
    for idx in range(n):
        sample = dataset[idx]
        dt_index = sample["dt_index"]
        omega_target = sample["omega_target"]
        if torch.is_tensor(dt_index) and dt_index.ndim > 0:
            for hop in range(int(dt_index.numel())):
                bucket = int(dt_index.flatten()[hop])
                omega = omega_target[hop].double().flatten(1)
                sums[bucket] += omega.sum(dim=1)
                sq_sums[bucket] += omega.square().sum(dim=1)
                counts[bucket] += omega.shape[1]
        else:
            bucket = int(dt_index)
            omega = omega_target.double().flatten(1)
            sums[bucket] += omega.sum(dim=1)
            sq_sums[bucket] += omega.square().sum(dim=1)
            counts[bucket] += omega.shape[1]
    std = torch.ones(n_scales, 3, dtype=torch.float32)
    valid = counts > 1
    if valid.any():
        denom = counts[valid].double().unsqueeze(1)
        mean = sums[valid] / denom
        var = (sq_sums[valid] / denom - mean.square()).clamp_min(1e-12)
        std[valid] = var.sqrt().float()
    if (~valid).any() and valid.any():
        std[~valid] = std[valid].mean(dim=0)
    return std, counts


def fit_physics_kappa(dataset, max_samples: int, min_temp_k: float = 1.0e-9) -> tuple[float | None, int]:
    """Fit sigma_omega ~= kappa * sqrt(dt_s * T) through the origin.

    Each sampled training pair contributes one scalar target sigma, measured as
    the mean per-component spatial standard deviation of its rotation-vector
    target. Zero-temperature samples do not constrain the thermal scale and are
    skipped.
    """
    numerator = 0.0
    denominator = 0.0
    used = 0
    n = min(int(max_samples), len(dataset))
    for idx in range(n):
        sample = dataset[idx]
        temp_k = sample["temp_k"]
        dt_s = sample["dt_s"]
        omega_target = sample["omega_target"]
        if torch.is_tensor(temp_k) and temp_k.ndim > 0:
            temp_flat = temp_k.flatten()
            dt_flat = dt_s.flatten() if torch.is_tensor(dt_s) else torch.full_like(temp_flat, float(dt_s))
            for hop in range(int(temp_flat.numel())):
                temp_val = float(temp_flat[hop])
                dt_val = float(dt_flat[hop])
                x = (dt_val * max(temp_val, 0.0)) ** 0.5
                if x <= 0.0 or temp_val <= min_temp_k:
                    continue
                omega = omega_target[hop].double().flatten(1)
                sigma = float(omega.std(dim=1, unbiased=False).mean())
                if not math.isfinite(sigma) or sigma <= 0.0:
                    continue
                numerator += x * sigma
                denominator += x * x
                used += 1
        else:
            temp_val = float(temp_k)
            dt_val = float(dt_s)
            x = (dt_val * max(temp_val, 0.0)) ** 0.5
            if x <= 0.0 or temp_val <= min_temp_k:
                continue
            omega = omega_target.double().flatten(1)
            sigma = float(omega.std(dim=1, unbiased=False).mean())
            if not math.isfinite(sigma) or sigma <= 0.0:
                continue
            numerator += x * sigma
            denominator += x * x
            used += 1
    if denominator <= 0.0 or used == 0:
        return None, used
    return numerator / denominator, used
