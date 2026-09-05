"""Source distributions for plan-1 (rotation-vector) and plan-2 (Cart / RFM)."""

from __future__ import annotations

import torch

from skyrmion_cfm.cfm.bridges import bridge_source_mode, bridge_state_repr
from skyrmion_cfm.cfm.rfm import sample_tangent_noise_chw


class RotationPrior:
    r"""Plan-1 / Mode α prior on the rotation-vector field.

    The full ``sigma_0 = kappa * sqrt(dt * T)`` law lives here so the same
    object can serve as the Mode α prior *and* as the Cart anchor-noise sigma
    (plan-2 §forward) and as the RFM source-noise sigma. Callers select the
    prior type via ``prior_type``:

    - ``standard_gaussian``: ``N(0, I)`` — used as the latent source in
      β-Cart and as the legacy noisy CFM source baseline.
    - ``physics_scaled``: thermal scaling ``sigma_0(dt, T)``.
    - ``empirical`` / ``empirical_scaled``: per-time-bucket empirical std
      multiplied by ``kappa``.
    """

    def __init__(
        self,
        prior_type: str = "physics_scaled",
        kappa: float = 1.25,
        min_sigma: float = 1e-4,
        dt_scales: list[int] | None = None,
        omega_std: torch.Tensor | None = None,
    ) -> None:
        self.prior_type = prior_type
        self.kappa = float(kappa)
        self.min_sigma = float(min_sigma)
        self.dt_scales = list(dt_scales) if dt_scales is not None else None
        self.omega_std = omega_std.float() if omega_std is not None else None

    def _time_seconds(self, cond: dict[str, torch.Tensor]) -> torch.Tensor:
        if "t_end_s" in cond:
            return cond["t_end_s"].float()
        return cond["dt_s"].float()

    def sigma(self, cond: dict[str, torch.Tensor], like: torch.Tensor | None = None) -> torch.Tensor:
        if self.prior_type == "standard_gaussian":
            dt = self._time_seconds(cond)
            sigma = torch.ones_like(dt)
        elif self.prior_type in {"empirical", "empirical_scaled"}:
            if self.omega_std is None:
                raise ValueError("empirical prior requires omega_std")
            idx = cond.get("dt_index", cond.get("t_end_index"))
            if idx is None:
                raise ValueError("empirical prior requires dt_index / t_end_index in cond")
            std = self.omega_std.to(idx.device)
            sigma = std[idx.long()] * self.kappa
        elif self.prior_type == "physics_scaled":
            dt = self._time_seconds(cond)
            temp = cond["temp_k"].float().clamp_min(0.0)
            sigma = self.kappa * torch.sqrt((dt * temp).clamp_min(0.0))
            sigma = sigma.clamp_min(self.min_sigma)
        else:
            raise ValueError(f"Unknown prior type: {self.prior_type}")
        sigma = sigma.clamp_min(self.min_sigma)
        if like is not None:
            while sigma.ndim < like.ndim:
                sigma = sigma[..., None]
        return sigma

    def sample(self, shape: torch.Size | tuple[int, ...], cond: dict[str, torch.Tensor]) -> torch.Tensor:
        device = self._time_seconds(cond).device
        noise = torch.randn(shape, device=device)
        return noise * self.sigma(cond, noise)


class CartSourcePrior:
    """Plan-2 β-Cart source ``x_0``.

    Two modes:

    - ``anchored`` (default): ``x_0 = M_init + σ_0 ε`` where ``σ_0`` follows
      the physics-scaled law. Drives the learned dynamics away from the true
      initial state without losing anchor information.
    - ``latent``: ``x_0 = σ_0 ε`` (no anchor); learns the conditional latent →
      M_target distribution.

    The caller passes ``M_init`` and ``σ_0`` (already broadcast).
    """

    def __init__(self, mode: str = "anchored", min_sigma: float = 1e-4) -> None:
        if mode not in {"anchored", "latent", "identity"}:
            raise ValueError(f"Unknown CartSourcePrior mode: {mode}")
        self.mode = mode
        self.min_sigma = float(min_sigma)

    def sample(
        self,
        m_init: torch.Tensor,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x_0, source_noise). source_noise is ``σ_0 ε`` for bookkeeping."""
        epsilon = torch.randn_like(m_init)
        while sigma.ndim < epsilon.ndim:
            sigma = sigma[..., None]
        noise = sigma.clamp_min(self.min_sigma) * epsilon
        if self.mode == "identity":
            return m_init, torch.zeros_like(m_init)
        if self.mode == "anchored":
            return m_init + noise, noise
        return noise, noise


class RFMSourcePrior:
    r"""Plan-2 β-RFM source.

    Two source modes, parallel to :class:`CartSourcePrior`:

    - ``anchored`` (default): ``x_0 = Exp_{M_init}(σ_0 ε^\perp)`` — small
      tangent kick from the true initial state, enabling the network to learn
      "M_init-conditioned generation around the anchor."
    - ``latent``: ``x_0 = Exp_{p}(σ_uniform ε^\perp)`` where ``p`` is a uniform
      draw on ``S^2`` per site (independent of ``M_init``); the bridge then
      learns to flow from a latent RFM prior to the conditional target.

    Passing ``enabled=False`` keeps ``anchored`` semantics but collapses the
    tangent noise to zero (``x_0 = M_init``), reproducing the
    source-noise-off ablation called out in plan-2.
    """

    def __init__(
        self,
        mode: str = "anchored",
        min_sigma: float = 1e-4,
        enabled: bool = True,
        latent_kappa: float = 1.0,
    ) -> None:
        if mode not in {"anchored", "latent", "identity"}:
            raise ValueError(f"Unknown RFMSourcePrior mode: {mode}")
        self.mode = mode
        self.min_sigma = float(min_sigma)
        self.enabled = bool(enabled)
        # vMF-ish concentration controlling how spread the latent base point is
        # around the north pole before being randomly rotated; large κ → near
        # the north pole, κ → 0 → uniform on the sphere. ``1.0`` is "fairly
        # uniform" and matches the spirit of "latent" prior.
        self.latent_kappa = float(latent_kappa)

    def base_point(self, m_init: torch.Tensor) -> torch.Tensor:
        """Return the centre point the tangent noise is taken at.

        anchored → ``M_init`` itself.
        latent   → a random unit-vector field iid per site (lattice-uniform).
        """
        if self.mode in {"anchored", "identity"}:
            return m_init
        # Per-site iid uniform direction on S^2. We pull a 3D Gaussian and
        # normalise; this is exact uniform on the sphere.
        raw = torch.randn_like(m_init)
        return raw / raw.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def sample_tangent_noise(
        self,
        base_point: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return the tangent noise vector at ``base_point``.

        Returns ``None`` only in the ``anchored`` + ``enabled=False`` ablation.
        In ``latent`` mode the noise is unconditionally drawn since the base
        point itself is already random and we want a *small* tangent kick on
        top of it (using the same physics-scaled σ as anchored).
        """
        if self.mode == "identity":
            return None
        if self.mode == "anchored" and not self.enabled:
            return None
        while sigma.ndim < base_point.ndim:
            sigma = sigma[..., None]
        return sample_tangent_noise_chw(base_point, sigma.clamp_min(self.min_sigma))


def make_priors(cfg: dict, stats=None) -> dict[str, object]:
    """Build the bridge-specific priors required by the chosen state_repr."""
    prior_cfg = cfg.get("prior", {})
    bridge_cfg = cfg.get("bridge", {})
    state_repr = bridge_state_repr(bridge_cfg or {"state_repr": "cart"})
    source_mode = bridge_source_mode(bridge_cfg, state_repr)
    rotation_prior = RotationPrior(
        prior_type=str(prior_cfg.get("type", "physics_scaled")),
        kappa=float(prior_cfg.get("kappa", 1.25)),
        min_sigma=float(prior_cfg.get("min_sigma", 1e-4)),
        dt_scales=list(cfg["data"].get("dt_scales", [])),
        omega_std=(stats.omega_std if stats is not None and hasattr(stats, "omega_std") else None),
    )
    cart_prior = CartSourcePrior(
        mode=(
            source_mode
            if state_repr == "cart"
            else str(bridge_cfg.get("cart", {}).get("source", prior_cfg.get("cart_source", "anchored")))
        ),
        min_sigma=float(prior_cfg.get("min_sigma", 1e-4)),
    )
    rfm_prior = RFMSourcePrior(
        mode=(
            source_mode
            if state_repr == "rfm"
            else str(bridge_cfg.get("rfm", {}).get("source", prior_cfg.get("rfm_source", "anchored")))
        ),
        min_sigma=float(prior_cfg.get("min_sigma", 1e-4)),
        enabled=bool(prior_cfg.get("rfm_source_noise", True)),
    )
    return {
        "rotation": rotation_prior,
        "cart": cart_prior,
        "rfm": rfm_prior,
    }
