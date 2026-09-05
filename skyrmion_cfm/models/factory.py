from __future__ import annotations

from skyrmion_cfm.data.conditions import ConditionStats
from skyrmion_cfm.models.dit import DiTVelocity
from skyrmion_cfm.models.fno import FNOVelocity
from skyrmion_cfm.models.mlp import MLPVelocity
from skyrmion_cfm.models.unet import UNetVelocity


def build_model(cfg: dict, stats: ConditionStats | None = None):
    arch = cfg["model"].get("arch", "unet").lower()
    if arch == "mlp":
        model = MLPVelocity(cfg, stats)
    elif arch == "unet":
        model = UNetVelocity(cfg, stats)
    elif arch == "dit":
        model = DiTVelocity(cfg, stats)
    elif arch in ("fno", "fourier", "fourier_neural_operator"):
        model = FNOVelocity(cfg, stats)
    elif arch in ("esteer", "steerable", "track_c"):
        # Track C — lazy import keeps the global module load free of the
        # heavy ``escnn`` dependency for users who do not need it.
        from skyrmion_cfm.models.esteer import EquivariantUNetVelocity

        model = EquivariantUNetVelocity(cfg, stats)
    else:
        raise ValueError(f"Unknown model architecture: {arch}")
    return model
