"""Pointwise S^2 geometry primitives for Riemannian Flow Matching.

Throughout, tensors store the xyz spin components along the channel dim
(``dim=1``) for shape ``(B, 3, ..., H, W)`` or along the last dim for
``(..., 3)``. Helpers in this module accept the *last-axis* layout; thin
``_chw`` wrappers convert from / to the channel layout used by the dataset.
"""

from __future__ import annotations

import torch


def _normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def project_to_tangent(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Project an ambient vector ``v`` onto ``T_p S^2``: ``v - (v·p) p``."""
    return v - (v * p).sum(dim=-1, keepdim=True) * p


def exp_map(p: torch.Tensor, v: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Pointwise S^2 exponential map: ``Exp_p(v) = cos(|v|) p + sin(|v|) v/|v|``.

    ``v`` is assumed to already lie in the tangent space at ``p``. We do not
    re-project for speed; callers that want safety should run
    :func:`project_to_tangent` first.
    """
    norm = v.norm(dim=-1, keepdim=True)
    cos = torch.cos(norm)
    sin_over_norm = torch.where(
        norm > eps, torch.sin(norm) / norm.clamp_min(eps), torch.ones_like(norm)
    )
    out = cos * p + sin_over_norm * v
    return _normalize(out)


def log_map(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pointwise S^2 log map: ``Log_p(q) = θ · ê`` where ``ê`` is in ``T_p``.

    Implements the numerically-stable form from plan2:
    ``θ = arccos(clip(p·q))``, ``ê = (q − cosθ · p) / sinθ``.

    Near antipodal (θ → π) the direction ê is geometrically ill-defined
    (any tangent direction is a valid Log) but the formula stays finite
    because ``q − cosθ p`` and ``sinθ`` both go to zero. We bump ``eps`` to
    1e-6 (was 1e-7) and project ``e_hat`` to the tangent space at the end
    so floating-point drift cannot push the magnitude past θ ≈ π.
    """
    p = _normalize(p)
    q = _normalize(q)
    dot = (p * q).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta).clamp_min(eps)
    e_hat = (q - dot * p) / sin_theta
    # Project ê back onto T_p — the formula is analytically orthogonal to p
    # but rounding can violate that near antipodal.
    e_hat = project_to_tangent(p, e_hat)
    # Renormalise: |ê| = 1 holds exactly only when ``dot`` was the true
    # cosine; the clamp shifted ``dot`` near antipodal so the magnitude
    # drifts from 1. Forcing unit norm preserves the geometric identity
    # ``Log_p(q) = θ ê`` where ``ê`` is a unit tangent vector.
    e_hat = e_hat / e_hat.norm(dim=-1, keepdim=True).clamp_min(eps)
    return theta * e_hat


def slerp(p: torch.Tensor, q: torch.Tensor, tau: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Spherical linear interpolation ``μ_τ = Exp_p(τ Log_p q)`` between unit p, q."""
    u = log_map(p, q, eps=eps)
    while tau.ndim < u.ndim:
        tau = tau[..., None]
    return exp_map(p, tau * u)


def slerp_velocity(
    p: torch.Tensor,
    q: torch.Tensor,
    tau: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(μ_τ, dot μ_τ)``.

    ``dot μ_τ = θ · (â × μ_τ)`` where ``â = (p × q) / sinθ``. Equivalently,
    ``dot μ_τ = ParallelTransport_{p → μ_τ}(Log_p q)`` (no τ dependence in the
    magnitude because the geodesic has constant speed).

    Near antipodal (θ → π) and near identical (θ → 0) the axis ``p × q`` is
    degenerate; we use the same closed form but project the resulting
    velocity back onto T_{μ_τ} so floating-point drift in either limit does
    not push it off the tangent plane. The "use Log_p q" identity ensures the
    answer remains a valid tangent vector even when the axis cross product is
    rounded to zero.
    """
    p = _normalize(p)
    q = _normalize(q)
    dot = (p * q).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta).clamp_min(eps)
    e_hat = (q - dot * p) / sin_theta
    e_hat = project_to_tangent(p, e_hat)
    e_hat = e_hat / e_hat.norm(dim=-1, keepdim=True).clamp_min(eps)
    while tau.ndim < e_hat.ndim:
        tau = tau[..., None]
    # μ_τ = cos(τθ) p + sin(τθ) ê
    cos_t = torch.cos(tau * theta)
    sin_t = torch.sin(tau * theta)
    mu = _normalize(cos_t * p + sin_t * e_hat)
    # Parallel transport of the log_map vector to μ_τ. Geometrically:
    # u(τ) = θ · (-sin(τθ) p + cos(τθ) ê). This formulation avoids the
    # ill-conditioned ``(p × q) / sinθ`` axis in the antipodal limit.
    target_v = theta * (-sin_t * p + cos_t * e_hat)
    # Final safety: project back onto T_{μ_τ}.
    target_v = project_to_tangent(mu, target_v)
    # When p and q coincide, log_map returns 0 so velocity is 0 already.
    target_v = torch.where(theta < eps, torch.zeros_like(target_v), target_v)
    return mu, target_v


def sample_tangent_noise(p: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> torch.Tensor:
    """Draw an iid ambient Gaussian and project onto ``T_p S^2``.

    Returned vectors are scaled by ``sigma`` (which may broadcast against
    ``p``). ``sigma`` is a scalar or has the leading-batch shape; the function
    will right-pad it to ``p.ndim``.
    """
    noise = torch.randn_like(p)
    tangent = project_to_tangent(p, noise)
    if not isinstance(sigma, torch.Tensor):
        sigma = p.new_tensor(float(sigma))
    while sigma.ndim < tangent.ndim:
        sigma = sigma[..., None]
    return sigma * tangent


# ---------------------------------------------------------------------------
# CHW (B, 3, H, W) convenience wrappers.
# ---------------------------------------------------------------------------


def _to_last(x: torch.Tensor) -> torch.Tensor:
    return x.movedim(1, -1)


def _from_last(x: torch.Tensor) -> torch.Tensor:
    return x.movedim(-1, 1)


def project_to_tangent_chw(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _from_last(project_to_tangent(_to_last(p), _to_last(v)))


def exp_map_chw(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _from_last(exp_map(_to_last(p), _to_last(v)))


def log_map_chw(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return _from_last(log_map(_to_last(p), _to_last(q)))


def slerp_chw(p: torch.Tensor, q: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    return _from_last(slerp(_to_last(p), _to_last(q), tau))


def slerp_velocity_chw(
    p: torch.Tensor, q: torch.Tensor, tau: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    mu, v = slerp_velocity(_to_last(p), _to_last(q), tau)
    return _from_last(mu), _from_last(v)


def sample_tangent_noise_chw(p: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> torch.Tensor:
    return _from_last(sample_tangent_noise(_to_last(p), sigma))
