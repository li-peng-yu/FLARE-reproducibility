r"""Plan2 forward / inverse CFM bridges.

Three state representations live behind a uniform interface:

- ``cart`` (β-Cart): straight-line interpolant in the ambient ``R^{3N}`` with
  target velocity ``v_τ = P_T_{ψ_τ}(M_t − x_0) / ‖z_τ‖``. The source ``x_0``
  is either anchored at ``M_init + σ_0 ε`` (default) or sampled from a
  latent ``σ_0 ε`` distribution (``source=latent``).
- ``rfm`` (β-RFM): SLERP between an endpoint-noise source on ``S^{2N}`` and
  ``M_t``; target velocity is the geodesic velocity ``θ â × μ_τ``.
- ``alpha`` (Mode α): legacy rotation-vector parameterisation.
- ``alpha2d``: rotation-vector parameterisation in a fixed 2D tangent basis at
  ``M_init``.

Bridge configuration separates three axes:

- ``state_repr`` selects the coordinate system of ``x_tau``.
- ``source_mode`` selects how the source endpoint ``x_0`` is sampled.
- ``objective`` describes the tau=1 endpoint semantics. The model is still
  trained only by velocity matching: ``endpoint`` means ``x_1 = M_t``;
  ``residual`` means ``x_1`` is the displacement from ``M_init`` to ``M_t`` in
  the bridge's state coordinates.

Every bridge exposes the same shape contract so the training loop and sampler
can dispatch without branching:

``build(x_init, x_target, tau, source_noise=None, cond=None)`` returns a dict
with ``state`` (the network input field), ``target`` (the velocity / Ω target
the network should match), and ``extra`` (bridge-specific quantities used by
the unit / topo regularisers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from skyrmion_cfm.cfm.rfm import (
    exp_map_chw,
    log_map_chw,
    project_to_tangent_chw,
    sample_tangent_noise_chw,
    slerp_velocity_chw,
)
from skyrmion_cfm.data.log_map import log_map_chw as rot_log_map_chw, normalize_spin
from skyrmion_cfm.data.log_map import rodrigues_chw


STATE_REPRS = {"alpha", "alpha2d", "cart", "rfm"}
SOURCE_MODES = {"anchored", "latent", "identity"}
OBJECTIVES = {"endpoint", "residual"}
DEFAULT_OBJECTIVE = {"alpha": "residual", "alpha2d": "residual", "cart": "endpoint", "rfm": "endpoint"}
SUPPORTED_OBJECTIVES = {
    "alpha": {"residual"},
    "alpha2d": {"residual"},
    "cart": {"endpoint", "residual"},
    "rfm": {"endpoint"},
}


def _normalise_repr(value: str, *, field: str = "state_repr") -> str:
    mode = str(value).lower()
    aliases = {
        "α": "alpha",
        "rotation_vector": "alpha",
        "rotation_vector_2d": "alpha2d",
        "alpha_2d": "alpha2d",
        "beta_cart": "cart",
        "β-cart": "cart",
        "beta_rfm": "rfm",
        "β-rfm": "rfm",
    }
    mode = aliases.get(mode, mode)
    if mode not in STATE_REPRS:
        raise ValueError(f"Unknown {field}: {value!r}; expected one of {sorted(STATE_REPRS)}")
    return mode


def _source_mode(value: str, *, default: str) -> str:
    mode = str(value or default).lower()
    if mode not in SOURCE_MODES:
        raise ValueError(
            f"Unknown source mode: {mode!r}; expected one of {sorted(SOURCE_MODES)}"
        )
    return mode


def bridge_state_repr(cfg: dict[str, Any]) -> str:
    """Return the bridge state coordinate system.

    ``state_repr`` is the new name. ``target_repr`` is accepted as a legacy
    alias because existing configs still use it.
    """
    return _normalise_repr(cfg.get("state_repr", cfg.get("target_repr", "cart")))


def bridge_source_mode(cfg: dict[str, Any], state_repr: str | None = None) -> str:
    """Return the source distribution mode for ``x_0``."""
    state = bridge_state_repr(cfg) if state_repr is None else _normalise_repr(state_repr)
    default = "latent" if state in {"alpha", "alpha2d"} else "anchored"
    per_state = cfg.get(state, {}) if isinstance(cfg.get(state, {}), dict) else {}
    legacy_source = per_state.get("source", cfg.get("source", default))
    return _source_mode(
        cfg.get("source_mode", legacy_source),
        default=default,
    )


def bridge_objective(cfg: dict[str, Any], state_repr: str | None = None) -> str:
    """Return tau=1 label semantics for the bridge."""
    state = bridge_state_repr(cfg) if state_repr is None else _normalise_repr(state_repr)
    objective = str(cfg.get("objective", DEFAULT_OBJECTIVE[state])).lower()
    if objective not in OBJECTIVES:
        raise ValueError(
            f"Unknown bridge objective: {objective!r}; expected one of {sorted(OBJECTIVES)}"
        )
    supported = SUPPORTED_OBJECTIVES[state]
    if objective not in supported:
        raise NotImplementedError(
            f"bridge.state_repr={state!r} currently supports objective={sorted(supported)!r}; "
            f"got {objective!r}"
        )
    return objective


def _broadcast_tau(tau: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    while tau.ndim < like.ndim:
        tau = tau[..., None]
    return tau


def _normalize_field(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


def tangent_basis_chw(m_init: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic orthonormal basis of ``T_m S^2`` for CHW spin fields."""
    m = _normalize_field(m_init, eps=eps)
    canon = torch.zeros_like(m)
    canon[:, 2:3] = 1.0
    canon_x = torch.zeros_like(m)
    canon_x[:, 0:1] = 1.0
    canon = torch.where((m[:, 2:3].abs() > 0.9).expand_as(m), canon_x, canon)
    e1 = canon - (canon * m).sum(dim=1, keepdim=True) * m
    e1 = _normalize_field(e1, eps=eps)
    e2 = torch.cross(m, e1, dim=1)
    e2 = _normalize_field(e2, eps=eps)
    return e1, e2


def tangent_coords_from_omega_chw(m_init: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    e1, e2 = tangent_basis_chw(m_init)
    c1 = (omega * e1).sum(dim=1, keepdim=True)
    c2 = (omega * e2).sum(dim=1, keepdim=True)
    return torch.cat([c1, c2], dim=1)


def omega_from_tangent_coords_chw(m_init: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    if coords.shape[1] != 2:
        raise ValueError(f"alpha2d coordinates must have 2 channels, got {coords.shape[1]}")
    e1, e2 = tangent_basis_chw(m_init)
    return coords[:, 0:1] * e1 + coords[:, 1:2] * e2


@dataclass
class BridgeOutput:
    state: torch.Tensor  # network input field
    target_velocity: torch.Tensor  # what the model should regress on
    source: torch.Tensor  # x_0 used for the bridge (Cart only; RFM source for diagnostics)
    extra: dict[str, torch.Tensor]


class CartBridge:
    """β-Cart bridge in either projected-sphere or raw affine coordinates.

    With the legacy default ``normalize_path=True``, the affine interpolation
    ``z_τ = (1-τ) x_0 + τ x_1`` is normalised site-wise and the model regresses
    its tangent velocity
    ``P_T_{ψ_τ}(x_1 − x_0) / ‖z_τ‖`` for ``ψ_τ = normalize(z_τ)``.

    ``normalize_path=False`` selects canonical Cartesian conditional flow
    matching: the network state is the *raw* affine interpolation ``z_τ`` and
    the target is the constant ambient velocity ``x_1 - x_0``.  The sampler
    likewise integrates in ``R^{3N}`` and only projects the learned endpoint
    back to unit spins when producing the final physical field.  Keeping this
    behind an explicit flag preserves every existing Cart checkpoint/config.

    Setting ``ignore_norm_factor=True`` only affects the legacy normalised
    path. It drops the ``1/‖z_τ‖`` factor as an ablation, which no longer
    matches that normalised sampler path exactly.
    """

    mode = "cart"

    def __init__(
        self,
        source: str = "anchored",
        objective: str = "endpoint",
        unit_eps: float = 1e-8,
        ignore_norm_factor: bool = False,
        min_norm_z: float = 0.1,
        normalize_path: bool = True,
    ) -> None:
        self.state_repr = self.mode
        self.source_mode = _source_mode(source, default="anchored")
        self.objective = str(objective).lower()
        if self.objective not in {"endpoint", "residual"}:
            raise NotImplementedError(
                "CartBridge currently supports objective='endpoint' or 'residual'"
            )
        # Legacy attribute used throughout the loss / sampler.
        self.source = self.source_mode
        self.unit_eps = float(unit_eps)
        self.ignore_norm_factor = bool(ignore_norm_factor)
        self.normalize_path = bool(normalize_path)
        # Velocity-denominator floor. ``1/‖z_τ‖`` diverges where z_τ passes
        # near the origin (near-antipodal endpoints around τ≈0.5), producing
        # exploding velocity labels. Floored separately from ``unit_eps`` so
        # the state direction ψ keeps its faithful (tiny-eps) normalisation.
        self.min_norm_z = float(min_norm_z)

    def build(
        self,
        x_init: torch.Tensor,
        x_target: torch.Tensor,
        tau: torch.Tensor,
        source_noise: torch.Tensor | None = None,
        sigma_anchor: torch.Tensor | None = None,
    ) -> BridgeOutput:
        if source_noise is None:
            source_noise = torch.zeros_like(x_init)
        if self.objective == "residual":
            if self.source == "identity":
                x0 = torch.zeros_like(x_target)
            else:
                x0 = source_noise
            tau_b = _broadcast_tau(tau, x_target)
            state = (1.0 - tau_b) * x0 + tau_b * x_target
            target_velocity = x_target - x0
            return BridgeOutput(
                state=state,
                target_velocity=target_velocity,
                source=x0,
                extra={},
            )
        if self.source == "identity":
            x0 = x_init
        elif self.source == "anchored":
            x0 = x_init + source_noise
        else:  # latent
            x0 = source_noise
        tau_b = _broadcast_tau(tau, x_init)
        z = (1.0 - tau_b) * x0 + tau_b * x_target
        diff = x_target - x0
        if not self.normalize_path:
            return BridgeOutput(
                state=z,
                target_velocity=diff,
                source=x0,
                extra={"z": z},
            )
        raw_norm = z.norm(dim=1, keepdim=True)
        psi = z / raw_norm.clamp_min(self.unit_eps)
        tangent_diff = diff - (diff * psi).sum(dim=1, keepdim=True) * psi
        if self.ignore_norm_factor:
            target_velocity = tangent_diff
        else:
            target_velocity = tangent_diff / raw_norm.clamp_min(self.min_norm_z)
        return BridgeOutput(
            state=psi,
            target_velocity=target_velocity,
            source=x0,
            extra={"z": z, "norm_z": raw_norm.clamp_min(self.unit_eps).squeeze(1)},
        )

    def reconstruct(
        self,
        state: torch.Tensor,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Apply normalisation; used by unit / topology regularisers."""
        return _normalize_field(state + velocity, eps=self.unit_eps)


class RFMBridge:
    r"""β-RFM SLERP bridge on ``S^{2N}``.

    Two source distributions (parallel to :class:`CartBridge`):

    - ``anchored``: ``x_0 = Exp_{M_init}(σ_0 ε^⊥)`` — default, learns flows
      from a tangent-noised true initial state.
    - ``latent``: the bridge expects ``base_point`` to be passed in via
      ``source_noise = base_point_tangent_pair`` semantics — concretely the
      caller passes (base_point, tangent_noise) packed as
      ``source_noise = base_point + tangent`` style; to keep the simple
      interface, :class:`RFMBridge` is told the ``source`` mode at construction
      and the loss-side ``_sample_source`` passes the right tensor for each
      mode (see ``cfm.loss._sample_source``).

    The signature still accepts a single ``source_noise`` tensor: for
    ``anchored`` it is the tangent kick at M_init; for ``latent`` it is the
    *full* x_0 already on S^2 (loss / sampler hand it in that way).
    """

    mode = "rfm"

    def __init__(
        self,
        source: str = "anchored",
        objective: str = "endpoint",
        unit_eps: float = 1e-8,
    ) -> None:
        self.state_repr = self.mode
        self.source_mode = _source_mode(source, default="anchored")
        self.objective = str(objective).lower()
        if self.objective != "endpoint":
            raise NotImplementedError("RFMBridge currently supports objective='endpoint'")
        self.source = self.source_mode
        self.unit_eps = float(unit_eps)

    def build(
        self,
        x_init: torch.Tensor,
        x_target: torch.Tensor,
        tau: torch.Tensor,
        source_noise: torch.Tensor | None = None,
        sigma_anchor: torch.Tensor | None = None,
    ) -> BridgeOutput:
        m_init = normalize_spin(x_init.movedim(1, -1)).movedim(-1, 1)
        m_t = normalize_spin(x_target.movedim(1, -1)).movedim(-1, 1)
        if self.source == "identity":
            x0 = m_init
        elif self.source == "anchored":
            if source_noise is None:
                x0 = m_init
            else:
                # tangent displacement at M_init
                x0 = exp_map_chw(m_init, source_noise)
        else:  # latent
            if source_noise is None:
                # Without a noise tensor, fall back to a fresh per-call uniform
                # point — keeps the build() contract safe in tests.
                raw = torch.randn_like(m_init)
                x0 = raw / raw.norm(dim=1, keepdim=True).clamp_min(1e-12)
            else:
                # source_noise IS the x_0 on S^2 (caller already constructed it).
                x0 = normalize_spin(source_noise.movedim(1, -1)).movedim(-1, 1)
        mu, velocity = slerp_velocity_chw(x0, m_t, tau)
        return BridgeOutput(
            state=mu,
            target_velocity=velocity,
            source=x0,
            extra={"u0": log_map_chw(x0, m_t)},
        )

    def reconstruct(self, state: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        tangent = project_to_tangent_chw(state, velocity)
        return exp_map_chw(state, tangent)


class RotationVectorBridge:
    """Mode α legacy bridge: straight line in rotation-vector space.

    ``x_target`` for this bridge is the rotation vector ``Ω = log_S2(M_init,
    M_t)`` (computed once on dataset construction). The state lives in
    ``R^{3N}`` (no manifold constraint) and the target velocity is the simple
    finite-difference ``Ω - Ω_0``.

    """

    mode = "alpha"

    def __init__(self, source: str = "latent", objective: str = "residual") -> None:
        self.state_repr = self.mode
        self.source_mode = _source_mode(source, default="latent")
        self.objective = str(objective).lower()
        if self.objective != "residual":
            raise NotImplementedError(
                "RotationVectorBridge currently supports objective='residual'"
            )
        self.source = self.source_mode

    def build(
        self,
        x_init: torch.Tensor,  # this is M_init (3, H, W) — used as context only
        x_target: torch.Tensor,  # this is Ω_target = log_S2(M_init, M_t)
        tau: torch.Tensor,
        source_noise: torch.Tensor | None = None,
        sigma_anchor: torch.Tensor | None = None,
    ) -> BridgeOutput:
        if source_noise is None:
            source_noise = torch.zeros_like(x_target)
        omega_0 = source_noise
        omega_target = x_target
        tau_b = _broadcast_tau(tau, omega_target)
        omega_tau = (1.0 - tau_b) * omega_0 + tau_b * omega_target
        velocity = omega_target - omega_0
        return BridgeOutput(
            state=omega_tau,
            target_velocity=velocity,
            source=omega_0,
            extra={"omega_target": omega_target},
        )

    def reconstruct(
        self,
        m_init: torch.Tensor,
        omega: torch.Tensor,
    ) -> torch.Tensor:
        return rodrigues_chw(omega, m_init)


class RotationVector2DBridge:
    """Mode α in a fixed 2D tangent coordinate system at ``M_init``."""

    mode = "alpha2d"

    def __init__(self, source: str = "latent", objective: str = "residual") -> None:
        self.state_repr = self.mode
        self.source_mode = _source_mode(source, default="latent")
        self.objective = str(objective).lower()
        if self.objective != "residual":
            raise NotImplementedError(
                "RotationVector2DBridge currently supports objective='residual'"
            )
        self.source = self.source_mode

    def target_from_omega(self, m_init: torch.Tensor, omega_target: torch.Tensor) -> torch.Tensor:
        return tangent_coords_from_omega_chw(m_init, omega_target)

    def omega_from_state(self, m_init: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return omega_from_tangent_coords_chw(m_init, coords)

    def build(
        self,
        x_init: torch.Tensor,
        x_target: torch.Tensor,
        tau: torch.Tensor,
        source_noise: torch.Tensor | None = None,
        sigma_anchor: torch.Tensor | None = None,
    ) -> BridgeOutput:
        coords_target = self.target_from_omega(x_init, x_target)
        if source_noise is None:
            source_noise = torch.zeros_like(coords_target)
        coords_0 = source_noise
        tau_b = _broadcast_tau(tau, coords_target)
        coords_tau = (1.0 - tau_b) * coords_0 + tau_b * coords_target
        velocity = coords_target - coords_0
        return BridgeOutput(
            state=coords_tau,
            target_velocity=velocity,
            source=coords_0,
            extra={"omega_target": x_target, "coords_target": coords_target},
        )

    def reconstruct(self, m_init: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return rodrigues_chw(self.omega_from_state(m_init, coords), m_init)


def make_bridge(cfg: dict[str, Any]):
    """Instantiate the bridge described by bridge state/source/objective config."""
    target = bridge_state_repr(cfg)
    source = bridge_source_mode(cfg, target)
    objective = bridge_objective(cfg, target)
    if target == "cart":
        cart_cfg = cfg.get("cart", {})
        return CartBridge(
            source=source,
            objective=objective,
            ignore_norm_factor=bool(cart_cfg.get("ignore_norm_factor", False)),
            min_norm_z=float(cart_cfg.get("min_norm_z", 0.1)),
            normalize_path=bool(cart_cfg.get("normalize_path", True)),
        )
    if target == "rfm":
        return RFMBridge(source=source, objective=objective)
    if target == "alpha":
        return RotationVectorBridge(source=source, objective=objective)
    if target == "alpha2d":
        return RotationVector2DBridge(source=source, objective=objective)
    raise ValueError(f"Unknown state_repr: {target}")


def project_velocity_to_state(bridge, state: torch.Tensor, raw_velocity: torch.Tensor) -> torch.Tensor:
    """Project the network's raw output to the bridge's tangent space.

    - Cart normalised endpoint: project onto ``T_{ψ_τ} S^2`` per lattice site.
    - Cart raw affine / residual: no projection (Euclidean coordinates).
    - RFM:  same projection (state lives on S^2).
    - α/α2d: no projection (Euclidean target coordinates).
    """
    if isinstance(bridge, CartBridge):
        if (
            getattr(bridge, "objective", "endpoint") == "residual"
            or not getattr(bridge, "normalize_path", True)
        ):
            return raw_velocity
        return project_to_tangent_chw(state, raw_velocity)
    if isinstance(bridge, RFMBridge):
        return project_to_tangent_chw(state, raw_velocity)
    return raw_velocity
