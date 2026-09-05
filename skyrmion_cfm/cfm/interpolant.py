from __future__ import annotations

import torch


def linear_interpolate(
    omega0: torch.Tensor,
    omega1: torch.Tensor,
    tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    while tau.ndim < omega0.ndim:
        tau = tau[..., None]
    omega_tau = (1.0 - tau) * omega0 + tau * omega1
    velocity = omega1 - omega0
    return omega_tau, velocity
