from __future__ import annotations

import os

import torch


def normalize_spin(m: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return m / m.norm(dim=-1, keepdim=True).clamp_min(eps)


def log_map_s2(m0: torch.Tensor, m1: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pointwise minimal rotation vector omega with R(omega) m0 = m1.

    The last dimension must contain xyz spin components. The returned omega has
    the same shape as the inputs.

    Near antipodal pairs (``m0 · m1 → -1``) the rotation axis is genuinely
    ambiguous; we cap ``theta`` at ``π − eps`` and pick an arbitrary tangent
    direction so callers never see NaN / Inf.
    """
    m0 = normalize_spin(m0)
    m1 = normalize_spin(m1)
    cross = torch.cross(m0, m1, dim=-1)
    sin_theta = cross.norm(dim=-1, keepdim=True)
    cos_theta = (m0 * m1).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.atan2(sin_theta, cos_theta)
    small = sin_theta < eps
    axis = cross / sin_theta.clamp_min(eps)
    omega = theta * axis
    # For near-antipodal points (sin_theta ~ 0 but cos_theta ~ -1) fall back
    # to ``π · ê`` along an arbitrary tangent direction picked from the
    # cross product of m0 with the most off-axis canonical vector.
    near_anti = cos_theta < (-1.0 + 1e-3)
    # Build a deterministic tangent direction at m0: project +z (or +x when
    # m0 is collinear with z) onto T_{m0} S^2. Keep this branchless so the
    # fused torch.compile path does not graph-break on Tensor.item()/bool().
    canon = torch.zeros_like(m0)
    canon[..., 2] = 1.0
    parallel = (m0[..., 2:3].abs() > 0.9).expand_as(canon)
    canon_x = torch.zeros_like(m0)
    canon_x[..., 0] = 1.0
    canon = torch.where(parallel, canon_x, canon)
    tangent = canon - (canon * m0).sum(dim=-1, keepdim=True) * m0
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(eps)
    omega_anti = torch.pi * tangent
    omega = torch.where(near_anti.expand_as(omega), omega_anti, omega)
    near_same = small & ~near_anti
    return torch.where(near_same.expand_as(omega), cross, omega)


def rodrigues(omega: torch.Tensor, m: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Apply pointwise Rodrigues rotations to spins.

    Both tensors use xyz as their last dimension.
    """
    theta = omega.norm(dim=-1, keepdim=True)
    axis = omega / theta.clamp_min(eps)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    rotated = (
        m * cos_t
        + torch.cross(axis, m, dim=-1) * sin_t
        + axis * (axis * m).sum(dim=-1, keepdim=True) * (1.0 - cos_t)
    )
    first_order = m + torch.cross(omega, m, dim=-1)
    return normalize_spin(torch.where(theta < eps, first_order, rotated))


def chw_to_last(x: torch.Tensor) -> torch.Tensor:
    return x.movedim(1, -1)


def last_to_chw(x: torch.Tensor) -> torch.Tensor:
    return x.movedim(-1, 1)


_COMPILED_LOG_MAP_CHW = None
_COMPILED_RODRIGUES_CHW = None


def _use_fused_cuda(x: torch.Tensor) -> bool:
    return (
        os.environ.get("SKYRMION_CFM_FUSED_OPS", "0") == "1"
        and x.is_cuda
        and hasattr(torch, "compile")
    )


def _log_map_chw_impl(m0: torch.Tensor, m1: torch.Tensor) -> torch.Tensor:
    return last_to_chw(log_map_s2(chw_to_last(m0), chw_to_last(m1)))


def _rodrigues_chw_impl(omega: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return last_to_chw(rodrigues(chw_to_last(omega), chw_to_last(m)))


def log_map_chw(m0: torch.Tensor, m1: torch.Tensor) -> torch.Tensor:
    global _COMPILED_LOG_MAP_CHW
    if _use_fused_cuda(m0):
        if _COMPILED_LOG_MAP_CHW is None:
            _COMPILED_LOG_MAP_CHW = torch.compile(_log_map_chw_impl, mode="reduce-overhead")
        return _COMPILED_LOG_MAP_CHW(m0, m1).clone()
    return _log_map_chw_impl(m0, m1)


def rodrigues_chw(omega: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    global _COMPILED_RODRIGUES_CHW
    if _use_fused_cuda(omega):
        if _COMPILED_RODRIGUES_CHW is None:
            _COMPILED_RODRIGUES_CHW = torch.compile(_rodrigues_chw_impl, mode="reduce-overhead")
        return _COMPILED_RODRIGUES_CHW(omega, m).clone()
    return _rodrigues_chw_impl(omega, m)
