from __future__ import annotations

import torch

from skyrmion_cfm.train import make_training_stats
from skyrmion_cfm.data.fixed_time import _fixed_time_omega_settings


class _Record:
    run_id = "r0"

    def condition_row(self, dt_s: float) -> dict[str, float]:
        return {
            "t_end_s": float(dt_s),
            "temp_k": 30.0,
            "current_a_m2": 1.0,
        }

    def material_row(self) -> dict[str, float]:
        return {
            "alpha": 0.3,
            "aex_j_per_m": 1.5e-11,
            "dind_j_per_m2": 3.25e-3,
            "ku1_j_per_m3": 8.0e5,
            "msat_a_per_m": 5.8e5,
        }


class _FixedTimeDataset:
    def __init__(self) -> None:
        self.records = [_Record()]
        self._t_end_s = [0.25e-9, 0.5e-9, 1.0e-9]

    def __len__(self) -> int:
        return 12

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        bucket = idx % len(self._t_end_s)
        base = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
        omega = 0.001 * base + 0.01 * (bucket + 1)
        return {
            "dt_index": torch.tensor(bucket, dtype=torch.long),
            "t_end_index": torch.tensor(bucket, dtype=torch.long),
            "dt_s": torch.tensor(self._t_end_s[bucket], dtype=torch.float32),
            "t_end_s": torch.tensor(self._t_end_s[bucket], dtype=torch.float32),
            "temp_k": torch.tensor(30.0, dtype=torch.float32),
            "omega_target": omega,
        }


def test_make_training_stats_accepts_fixed_time_dataset_without_dt_scales():
    cfg = {
        "data": {
            "dataset_root": "dummy-fixed-time",
            "t_end_ns": [0.25, 0.5, 1.0],
            "stats_max_samples": 12,
            "kappa_fit_max_samples": 12,
        },
        "prior": {"fit_kappa": True},
    }

    stats = make_training_stats(_FixedTimeDataset(), cfg)

    assert stats.dt_scales == [0, 1, 2]
    assert stats.omega_std.shape == (3, 3)
    assert stats.omega_count.tolist() == [16, 16, 16]
    assert stats.physics_kappa_count == 12


def test_fixed_time_fit_kappa_forces_uncapped_omega_targets():
    cfg = {
        "data": {},
        "bridge": {"target_repr": "cart"},
        "prior": {"fit_kappa": True},
        "train": {"loss": {"alpha_t_end_ns_max": 1.0, "alpha_mix": 0.0}},
    }

    compute_omega, omega_cap = _fixed_time_omega_settings(cfg)

    assert compute_omega is True
    assert omega_cap == float("inf")
