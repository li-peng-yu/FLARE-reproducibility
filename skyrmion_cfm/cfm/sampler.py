"""Bridge-aware ODE samplers.

Plan-1 used a single Heun integrator on the rotation-vector field; plan-2
needs distinct flows for Cart / RFM / α. The :class:`BridgeSampler` provides
the dispatch and exposes the same ``sample`` and ``sample_state`` entry points
the legacy training loop expects.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from skyrmion_cfm.cfm.bridges import (
    CartBridge,
    RFMBridge,
    RotationVector2DBridge,
    RotationVectorBridge,
    tangent_coords_from_omega_chw,
    project_velocity_to_state,
)
from skyrmion_cfm.cfm.guidance import make_unconditional_cond
from skyrmion_cfm.cfm.prior import CartSourcePrior, RFMSourcePrior, RotationPrior
from skyrmion_cfm.cfm.rfm import exp_map_chw, project_to_tangent_chw, sample_tangent_noise_chw
from skyrmion_cfm.cfm.spatial_mask import (
    apply_spatial_constraint,
    bridge_output_background,
    bridge_state_background,
)
from skyrmion_cfm.data.log_map import log_map_chw as rot_log_map_chw
from skyrmion_cfm.data.log_map import rodrigues_chw


class BridgeSampler:
    """Heun integrator for Cart / RFM / α bridges."""

    def __init__(
        self,
        bridge,
        rotation_prior: RotationPrior,
        cart_prior: CartSourcePrior | None = None,
        rfm_prior: RFMSourcePrior | None = None,
        ode_steps: int = 20,
        method: str = "heun",
        classifier_free_guidance: dict | None = None,
        stochastic_sampler: dict | None = None,
    ) -> None:
        self.bridge = bridge
        self.rotation_prior = rotation_prior
        self.cart_prior = cart_prior
        self.rfm_prior = rfm_prior
        self.ode_steps = int(ode_steps)
        self.method = method
        self.classifier_free_guidance = classifier_free_guidance or {}
        self.stochastic_sampler = stochastic_sampler or {}
        self.grad_checkpoint = False

    # ------------------------------------------------------------------
    def _init_state(
        self,
        m_init: torch.Tensor,
        cond: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample ``x_0`` for the configured bridge.

        Returns ``(state, alpha_omega0)``. ``alpha_omega0`` is only set for
        Mode α and is needed by :meth:`_finalize`.
        """
        if isinstance(self.bridge, CartBridge):
            if getattr(self.bridge, "objective", "endpoint") == "residual":
                if self.bridge.source == "identity":
                    return torch.zeros_like(m_init), None
                sigma = self.rotation_prior.sigma(cond, m_init)
                return sigma * torch.randn_like(m_init), None
            if self.cart_prior is None:
                raise ValueError("Cart sampling requires cart_prior")
            if self.bridge.source == "identity":
                return m_init.clone(), None
            sigma = self.rotation_prior.sigma(cond, m_init)
            x0, _ = self.cart_prior.sample(m_init, sigma)
            if not getattr(self.bridge, "normalize_path", True):
                return x0, None
            # Train/sample consistency: CartBridge.build at τ=0 exposes
            # ``ψ_0 = z_0 / ‖z_0‖ = x_0 / ‖x_0‖`` as the network input, with
            # ``unit_eps = 1e-8``. The sampler MUST start from the same
            # normalised state so the first ODE step lands on the trained
            # trajectory. We align the clamp here to the bridge's eps; the
            # legacy 1e-6 caused a tiny offset only in the latent-source
            # regime where ‖x_0‖ can be small.
            norm = x0.norm(dim=1, keepdim=True).clamp_min(self.bridge.unit_eps)
            return x0 / norm, None
        if isinstance(self.bridge, RFMBridge):
            if self.rfm_prior is None:
                return m_init.clone(), None
            if self.bridge.source == "identity":
                return m_init.clone(), None
            sigma = self.rotation_prior.sigma(cond, m_init)
            if self.rfm_prior.mode == "anchored":
                if not self.rfm_prior.enabled:
                    return m_init.clone(), None
                tangent = self.rfm_prior.sample_tangent_noise(m_init, sigma)
                return exp_map_chw(m_init, tangent), None
            # latent: random base on S^2 + optional small tangent kick
            base = self.rfm_prior.base_point(m_init)
            tangent = self.rfm_prior.sample_tangent_noise(base, sigma)
            if tangent is None:
                return base, None
            return exp_map_chw(base, tangent), None
        if isinstance(self.bridge, RotationVectorBridge):
            source_mode = getattr(self.bridge, "source", "latent")
            if source_mode == "identity":
                omega0 = torch.zeros_like(m_init)
            elif source_mode == "anchored":
                sigma = self.rotation_prior.sigma(cond, m_init)
                tangent = sample_tangent_noise_chw(m_init, sigma)
                source_m = exp_map_chw(m_init, tangent)
                omega0 = rot_log_map_chw(m_init, source_m)
            elif source_mode == "latent":
                omega0 = self.rotation_prior.sample(m_init.shape, cond)
            else:
                raise ValueError(f"Unknown alpha source mode: {source_mode}")
            return omega0, omega0
        if isinstance(self.bridge, RotationVector2DBridge):
            source_mode = getattr(self.bridge, "source", "latent")
            if source_mode == "identity":
                coords0 = m_init.new_zeros(m_init.shape[0], 2, *m_init.shape[2:])
            elif source_mode == "anchored":
                sigma = self.rotation_prior.sigma(cond, m_init)
                tangent = sample_tangent_noise_chw(m_init, sigma)
                source_m = exp_map_chw(m_init, tangent)
                omega0 = rot_log_map_chw(m_init, source_m)
                coords0 = tangent_coords_from_omega_chw(m_init, omega0)
            elif source_mode == "latent":
                omega0 = self.rotation_prior.sample(m_init.shape, cond)
                coords0 = tangent_coords_from_omega_chw(m_init, omega0)
            else:
                raise ValueError(f"Unknown alpha2d source mode: {source_mode}")
            return coords0, coords0
        raise TypeError(f"Unknown bridge type: {type(self.bridge)}")

    def _clamp_state(
        self,
        state: torch.Tensor,
        m_init: torch.Tensor,
        cond: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return apply_spatial_constraint(
            state,
            cond,
            outside=bridge_state_background(self.bridge, m_init, state),
        )

    def _mask_velocity(
        self,
        velocity: torch.Tensor,
        cond: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return apply_spatial_constraint(velocity, cond)

    def _step(
        self,
        model: nn.Module,
        state: torch.Tensor,
        m_init: torch.Tensor,
        tau0: torch.Tensor,
        tau1: torch.Tensor,
        cond: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        v0 = self._velocity(model, state, m_init, tau0, cond)
        h = (tau1 - tau0).reshape(-1, *([1] * (state.ndim - 1)))
        if self.method == "euler":
            return self._clamp_state(self._advance(state, h * v0), m_init, cond)
        candidate = self._clamp_state(self._advance(state, h * v0), m_init, cond)
        v1 = self._velocity(model, candidate, m_init, tau1, cond)
        return self._clamp_state(self._advance(state, 0.5 * h * (v0 + v1)), m_init, cond)

    def _guidance_scale(self, tau: torch.Tensor) -> float:
        cfg = self.classifier_free_guidance
        if not bool(cfg.get("enabled", False)):
            return 1.0
        scale = float(cfg.get("scale", 1.0))
        tau_min = float(cfg.get("tau_min", 0.0))
        tau_max = float(cfg.get("tau_max", 1.0))
        tau_mean = float(tau.detach().float().mean().cpu())
        if tau_mean < tau_min or tau_mean > tau_max:
            return 1.0
        return scale

    def _velocity(
        self,
        model: nn.Module,
        state: torch.Tensor,
        m_init: torch.Tensor,
        tau: torch.Tensor,
        cond: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        scale = self._guidance_scale(tau)
        if bool(getattr(self, "grad_checkpoint", False)) and torch.is_grad_enabled():
            def run_model(m_init_arg, state_arg, tau_arg):
                return model(m_init_arg, state_arg, tau_arg, cond)

            v_cond_raw = checkpoint(run_model, m_init, state, tau, use_reentrant=False)
        else:
            v_cond_raw = model(m_init, state, tau, cond)
        if scale == 1.0:
            return self._mask_velocity(project_velocity_to_state(self.bridge, state, v_cond_raw), cond)
        uncond = make_unconditional_cond(cond, self.classifier_free_guidance)
        if bool(getattr(self, "grad_checkpoint", False)) and torch.is_grad_enabled():
            def run_uncond_model(m_init_arg, state_arg, tau_arg):
                return model(m_init_arg, state_arg, tau_arg, uncond)

            v_uncond_raw = checkpoint(run_uncond_model, m_init, state, tau, use_reentrant=False)
        else:
            v_uncond_raw = model(m_init, state, tau, uncond)
        guided = v_uncond_raw + scale * (v_cond_raw - v_uncond_raw)
        return self._mask_velocity(project_velocity_to_state(self.bridge, state, guided), cond)

    def _churn_state(
        self,
        state: torch.Tensor,
        tau: torch.Tensor,
        h: float,
    ) -> torch.Tensor:
        cfg = self.stochastic_sampler
        if not bool(cfg.get("enabled", False)):
            return state
        tau_min = float(cfg.get("tau_min", 0.0))
        tau_max = float(cfg.get("tau_max", 1.0))
        tau_mean = float(tau.detach().float().mean().cpu())
        if tau_mean < tau_min or tau_mean > tau_max:
            return state
        gamma = float(cfg.get("gamma", 0.0))
        noise_scale = float(cfg.get("noise_scale", 1.0))
        if gamma <= 0.0 or noise_scale <= 0.0:
            return state
        std = gamma * noise_scale * (max(float(h), 1e-12) ** 0.5)
        noise = torch.randn_like(state) * std
        if isinstance(self.bridge, RFMBridge):
            tangent = project_to_tangent_chw(state, noise)
            return exp_map_chw(state, tangent)
        if (
            isinstance(self.bridge, CartBridge)
            and getattr(self.bridge, "objective", "endpoint") != "residual"
            and getattr(self.bridge, "normalize_path", True)
        ):
            out = state + noise
            return out / out.norm(dim=1, keepdim=True).clamp_min(self.bridge.unit_eps)
        return state + noise

    def _advance(self, state: torch.Tensor, increment: torch.Tensor) -> torch.Tensor:
        if isinstance(self.bridge, CartBridge):
            if (
                getattr(self.bridge, "objective", "endpoint") == "residual"
                or not getattr(self.bridge, "normalize_path", True)
            ):
                return state + increment
            new_state = state + increment
            return new_state / new_state.norm(dim=1, keepdim=True).clamp_min(
                self.bridge.unit_eps
            )
        if isinstance(self.bridge, RFMBridge):
            tangent = project_to_tangent_chw(state, increment)
            return exp_map_chw(state, tangent)
        return state + increment

    def _finalize(
        self,
        state: torch.Tensor,
        m_init: torch.Tensor,
        alpha_omega0: torch.Tensor | None,
        cond: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Convert the τ=1 state into a unit-norm ``M_t`` field."""
        if isinstance(self.bridge, CartBridge):
            if getattr(self.bridge, "objective", "endpoint") == "residual":
                out = m_init + state
                out = out / out.norm(dim=1, keepdim=True).clamp_min(self.bridge.unit_eps)
            else:
                out = state
                if not getattr(self.bridge, "normalize_path", True):
                    out = out / out.norm(dim=1, keepdim=True).clamp_min(
                        self.bridge.unit_eps
                    )
            return apply_spatial_constraint(
                out,
                cond,
                outside=bridge_output_background(self.bridge, m_init, out),
            )
        if isinstance(self.bridge, RFMBridge):
            return apply_spatial_constraint(
                state,
                cond,
                outside=bridge_output_background(self.bridge, m_init, state),
            )
        if isinstance(self.bridge, RotationVectorBridge):
            out = rodrigues_chw(state, m_init)
            return apply_spatial_constraint(
                out,
                cond,
                outside=bridge_output_background(self.bridge, m_init, out),
            )
        if isinstance(self.bridge, RotationVector2DBridge):
            out = self.bridge.reconstruct(m_init, state)
            return apply_spatial_constraint(
                out,
                cond,
                outside=bridge_output_background(self.bridge, m_init, out),
            )
        raise TypeError(f"Unknown bridge type: {type(self.bridge)}")

    @torch.no_grad()
    def sample_state(
        self,
        model: nn.Module,
        m_init: torch.Tensor,
        cond: dict[str, torch.Tensor],
        state0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._sample_state_impl(model, m_init, cond, state0=state0)

    def sample_state_with_grad(
        self,
        model: nn.Module,
        m_init: torch.Tensor,
        cond: dict[str, torch.Tensor],
        state0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._sample_state_impl(model, m_init, cond, state0=state0)

    def _sample_state_impl(
        self,
        model: nn.Module,
        m_init: torch.Tensor,
        cond: dict[str, torch.Tensor],
        state0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if state0 is None:
            state, alpha_omega0 = self._init_state(m_init, cond)
            state = self._clamp_state(state, m_init, cond)
            if alpha_omega0 is not None:
                alpha_omega0 = self._mask_velocity(alpha_omega0, cond)
        else:
            state, alpha_omega0 = self._clamp_state(state0, m_init, cond), None
        bsz = m_init.shape[0]
        device = m_init.device
        steps = self.ode_steps
        for s in range(steps):
            tau0 = torch.full((bsz,), s / steps, device=device)
            tau1 = torch.full((bsz,), (s + 1) / steps, device=device)
            state = self._churn_state(state, tau0, 1.0 / float(steps))
            state = self._clamp_state(state, m_init, cond)
            state = self._step(model, state, m_init, tau0, tau1, cond)
        return state, alpha_omega0

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        m_init: torch.Tensor,
        cond: dict[str, torch.Tensor],
        state0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state, alpha_omega0 = self.sample_state(model, m_init, cond, state0=state0)
        m_t = self._finalize(state, m_init, alpha_omega0, cond)
        return m_t, state

    def sample_with_grad(
        self,
        model: nn.Module,
        m_init: torch.Tensor,
        cond: dict[str, torch.Tensor],
        state0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state, alpha_omega0 = self.sample_state_with_grad(model, m_init, cond, state0=state0)
        m_t = self._finalize(state, m_init, alpha_omega0, cond)
        return m_t, state


# ---------------------------------------------------------------------------
# Legacy HeunSampler kept as a thin alias so plan-1 train / eval code paths
# continue to work without changes.
# ---------------------------------------------------------------------------
class HeunSampler:
    """Thin wrapper that emulates the plan-1 rotation-vector Heun sampler."""

    def __init__(self, prior: RotationPrior, ode_steps: int = 20) -> None:
        self.prior = prior
        self.ode_steps = int(ode_steps)
        # Internal BridgeSampler that handles the actual integration.
        self._inner = BridgeSampler(
            bridge=RotationVectorBridge(),
            rotation_prior=prior,
            cart_prior=None,
            rfm_prior=None,
            ode_steps=ode_steps,
        )

    @torch.no_grad()
    def sample_omega(
        self,
        model: nn.Module,
        m0: torch.Tensor,
        cond: dict[str, torch.Tensor],
        omega0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        omega, _ = self._inner.sample_state(model, m0, cond, state0=omega0)
        return omega

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        m0: torch.Tensor,
        cond: dict[str, torch.Tensor],
        omega0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._inner.sample(model, m0, cond, state0=omega0)
