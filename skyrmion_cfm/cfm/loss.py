"""Plan-2 CFM loss with bridge dispatch.

The model interface is uniform across modes: the velocity network receives the
8-channel image (``M_init`` ‖ state ‖ defect_field ‖ j_field, or the legacy
6-channel input when spatial conditioning is disabled), the conditioning
vector, and τ. Its raw 3D output is projected to the bridge's tangent space
before the L2 / weighted loss is taken.

Auxiliary losses (unit / LLG / topology-charge) are kept from plan-1 and only
apply where the bridge state lives in S^2; α-bridge skips the surface ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from skyrmion_cfm.cfm.bridges import (
    BridgeOutput,
    CartBridge,
    RFMBridge,
    RotationVector2DBridge,
    RotationVectorBridge,
    tangent_coords_from_omega_chw,
    project_velocity_to_state,
)
from skyrmion_cfm.cfm.guidance import apply_cfg_dropout
from skyrmion_cfm.cfm.prior import CartSourcePrior, RFMSourcePrior, RotationPrior
from skyrmion_cfm.cfm.rfm import exp_map_chw, project_to_tangent_chw, sample_tangent_noise_chw
from skyrmion_cfm.cfm.spatial_mask import (
    apply_spatial_constraint,
    bridge_output_background,
    bridge_state_background,
    masked_site_mean,
    spatial_weight,
    weighted_site_mean,
)
from skyrmion_cfm.data.log_map import log_map_chw as rot_log_map_chw
from skyrmion_cfm.data.log_map import rodrigues_chw
from skyrmion_cfm.eval.metrics import topological_charge
from skyrmion_cfm.models.common import central_diff, normalize_boundary, pad_2d


@dataclass
class LossOutput:
    total: torch.Tensor
    cfm: torch.Tensor
    unit: torch.Tensor
    llg: torch.Tensor
    topo: torch.Tensor
    endpoint: torch.Tensor
    action: torch.Tensor | None = None
    action_valid_fraction: torch.Tensor | None = None


def _build_spatial_cond(cond: dict[str, torch.Tensor]) -> torch.Tensor | None:
    parts: list[torch.Tensor] = []
    for key in ("defect_field", "j_field"):
        if key in cond and cond[key] is not None:
            parts.append(cond[key].float())
    if not parts:
        return None
    return torch.cat(parts, dim=1)


def _bridge_target_repr(bridge) -> str:
    if isinstance(bridge, CartBridge):
        return "cart"
    if isinstance(bridge, RFMBridge):
        return "rfm"
    if isinstance(bridge, RotationVectorBridge):
        return "alpha"
    if isinstance(bridge, RotationVector2DBridge):
        return "alpha2d"
    raise TypeError(f"Unknown bridge type: {type(bridge)}")


def _is_alpha_like_bridge(bridge) -> bool:
    return isinstance(bridge, (RotationVectorBridge, RotationVector2DBridge))


class CFMLoss(nn.Module):
    """Universal CFM loss for plan-2 forward / inverse training.

    Bridge / prior / source are explicit dependencies; the loss does not
    instantiate them itself, so the same object trains forward and inverse.
    """

    def __init__(
        self,
        bridge=None,
        rotation_prior: RotationPrior | None = None,
        cart_prior: CartSourcePrior | None = None,
        rfm_prior: RFMSourcePrior | None = None,
        cfm_weight: float = 1.0,
        unit_weight: float = 0.0,
        llg_weight: float = 0.0,
        llg_max_scale: int = 5,
        topo_weight: float = 0.0,
        alpha_t_end_ns_max: float = 1.0,
        alpha_mix: float = 0.0,
        tau_sampling: str = "uniform",
        weight_mode: str = "uniform",
        target_rms_reweight: dict[str, Any] | None = None,
        drive_family_reweight: dict[str, Any] | None = None,
        segment_role_reweight: dict[str, Any] | None = None,
        change_aware_reweight: dict[str, Any] | None = None,
        minibatch_ot: dict[str, Any] | None = None,
        cfg_dropout: dict[str, Any] | None = None,
        action_matching: dict[str, Any] | None = None,
        boundary: str = "open",
        antipodal_eps: float = 0.0,
        void_loss_weight: float = 1.0e-2,
        prior: RotationPrior | None = None,  # legacy positional alias
    ) -> None:
        super().__init__()
        # Legacy plan-1 callers passed ``CFMLoss(prior, ...)`` — interpret a
        # ``RotationPrior`` as the first positional argument and default the
        # bridge to the rotation-vector form so old test rigs keep working.
        if isinstance(bridge, RotationPrior) and rotation_prior is None:
            rotation_prior = bridge
            bridge = RotationVectorBridge()
        if prior is not None and rotation_prior is None:
            rotation_prior = prior
        if rotation_prior is None:
            raise ValueError("CFMLoss requires a rotation_prior (or a positional RotationPrior)")
        if bridge is None:
            bridge = RotationVectorBridge()
        self.bridge = bridge
        self.rotation_prior = rotation_prior
        self.cart_prior = cart_prior
        self.rfm_prior = rfm_prior
        self.cfm_weight = float(cfm_weight)
        self.unit_weight = float(unit_weight)
        self.llg_weight = float(llg_weight)
        self.llg_max_scale = int(llg_max_scale)
        self.topo_weight = float(topo_weight)
        self.alpha_t_end_ns_max = float(alpha_t_end_ns_max)
        self.alpha_mix = float(alpha_mix)
        self.tau_sampling = tau_sampling
        self.weight_mode = weight_mode
        self.target_rms_reweight = target_rms_reweight or {}
        self.drive_family_reweight = drive_family_reweight or {}
        self.segment_role_reweight = segment_role_reweight or {}
        self.change_aware_reweight = change_aware_reweight or {}
        self.minibatch_ot = minibatch_ot or {}
        self.cfg_dropout = cfg_dropout or {}
        self.action_matching = action_matching or {}
        self.boundary = normalize_boundary(boundary)
        self.void_loss_weight = max(0.0, float(void_loss_weight))
        # plan-2 §forward (line 87-89): sites with θ_i > 5π/6 are marked
        # unreliable and excluded from the RFM velocity supervision because
        # the SLERP / Log_p label divides by sin θ. β-Cart's target lives in
        # R^3 (no sin θ in the denominator), so this mask is only auto-applied
        # when the bridge is :class:`RFMBridge`.
        # The threshold ``cos θ < -(1 - antipodal_eps)``
        # reproduces 5π/6 at ``antipodal_eps ≈ 0.134`` (= 1 − cos(5π/6) =
        # 1 − √3/2). Default 0 keeps the legacy "no masking" behaviour.
        self.antipodal_eps = float(antipodal_eps)
        # plan-2 line 34: L = (1-λ_α) L_β + λ_α L_α. When the primary bridge
        # is β-Cart / β-RFM and alpha_mix > 0, we run a *second* forward pass
        # with the rotation-vector bridge state Ω_τ so the network is jointly
        # supervised on both targets. The α residual still respects
        # alpha_t_end_ns_max when t_end_ns is provided.
        self._alpha_aux_bridge = (
            RotationVectorBridge()
            if (self.alpha_mix > 0.0 and not _is_alpha_like_bridge(self.bridge))
            else None
        )

    # ------------------------------------------------------------------
    # LLG residual identical to plan-1; reused only when the bridge state
    # represents M_t (Cart / RFM).
    # ------------------------------------------------------------------
    def _llg_residual(
        self,
        m_init: torch.Tensor,
        m_t_hat: torch.Tensor,
        cond: dict[str, torch.Tensor],
        dt_seconds: torch.Tensor,
    ) -> torch.Tensor:
        mu0 = 4.0 * torch.pi * 1.0e-7
        gamma_ll = 2.211e5
        hbar = 1.054571817e-34
        electron_charge = 1.602176634e-19

        def scalar(name: str, default: float) -> torch.Tensor:
            value = cond.get(name)
            if value is None:
                value = m_init.new_full((m_init.shape[0],), default)
            value = value.to(device=m_init.device, dtype=m_init.dtype)
            while value.ndim < m_init.ndim:
                value = value[..., None]
            return value

        alpha = scalar("alpha", 0.1).clamp_min(0.0)
        aex = scalar("aex_j_per_m", 1.0e-11)
        dind = scalar("dind_j_per_m2", 0.0)
        ku1 = scalar("ku1_j_per_m3", 0.0)
        msat = scalar("msat_a_per_m", 580000.0).clamp_min(1.0)
        dx = scalar("dx_m", 1.0).clamp_min(1.0e-12)
        dy = scalar("dy_m", 1.0).clamp_min(1.0e-12)
        dz = scalar("dz_m", 1.0).clamp_min(1.0e-12)

        bnd = self.boundary
        padded_x = pad_2d(m_init, (1, 1, 0, 0), bnd)
        padded_y = pad_2d(m_init, (0, 0, 1, 1), bnd)
        lap = (
            (padded_x[..., :, 2:] + padded_x[..., :, :-2] - 2.0 * m_init) / dx.square()
            + (padded_y[..., 2:, :] + padded_y[..., :-2, :] - 2.0 * m_init) / dy.square()
        )
        h_ex = 2.0 * aex * lap / (mu0 * msat)
        dmx_dx = central_diff(m_init[:, 0:1], dim=-1, boundary=bnd) / dx
        dmy_dy = central_diff(m_init[:, 1:2], dim=-2, boundary=bnd) / dy
        dmz_dx = central_diff(m_init[:, 2:3], dim=-1, boundary=bnd) / dx
        dmz_dy = central_diff(m_init[:, 2:3], dim=-2, boundary=bnd) / dy
        h_dmi = torch.cat([-dmz_dx, -dmz_dy, dmx_dx + dmy_dy], dim=1) * (2.0 * dind / (mu0 * msat))
        h_anis = torch.zeros_like(m_init)
        h_anis[:, 2:3] = 2.0 * ku1 * m_init[:, 2:3] / (mu0 * msat)
        b_t = cond["b_t"].to(device=m_init.device, dtype=m_init.dtype)
        while b_t.ndim < m_init.ndim:
            b_t = b_t[..., None]
        h_eff = h_ex + h_dmi + h_anis + b_t / mu0
        mxh = torch.cross(m_init, h_eff, dim=1)
        mxmxh = torch.cross(m_init, mxh, dim=1)
        rhs = -gamma_ll / (1.0 + alpha.square()) * (mxh + alpha * mxmxh)
        if "current_a_m2" in cond:
            current = scalar("current_a_m2", 0.0)
            pol = scalar("pol_eff", 0.0)
            eps_prime = scalar("epsilon_prime", 0.0)
            p = torch.stack(
                [
                    cond.get("fixed_layer_x", m_init.new_zeros(m_init.shape[0])).to(m_init.device, m_init.dtype),
                    cond.get("fixed_layer_y", m_init.new_ones(m_init.shape[0])).to(m_init.device, m_init.dtype),
                    cond.get("fixed_layer_z", m_init.new_zeros(m_init.shape[0])).to(m_init.device, m_init.dtype),
                ],
                dim=1,
            )
            p = F.normalize(p, dim=1)
            while p.ndim < m_init.ndim:
                p = p[..., None]
            beta = gamma_ll * hbar * current * pol / (2.0 * electron_charge * msat * dz)
            damping_like = torch.cross(m_init, torch.cross(m_init, p.expand_as(m_init), dim=1), dim=1)
            field_like = torch.cross(m_init, p.expand_as(m_init), dim=1)
            rhs = rhs + beta / (1.0 + alpha.square()) * (damping_like + eps_prime * field_like)
        dt = dt_seconds.clamp_min(1e-15)
        while dt.ndim < m_init.ndim:
            dt = dt[..., None]
        return (m_t_hat - m_init) / dt - rhs

    # ------------------------------------------------------------------
    # τ sampling and weighting
    # ------------------------------------------------------------------
    def _sample_tau(self, bsz: int, device: torch.device) -> torch.Tensor:
        if self.tau_sampling == "zero":
            # Explicit direct-regression baseline.  With an identity-source
            # residual bridge, tau=0 exposes a zero residual state.  The
            # velocity target is M_target - M_init for Cartesian coordinates
            # and Log_{M_init}(M_target) for local rotation coordinates.
            # Euler-1 inference is one deterministic network forward pass.
            return torch.zeros(bsz, device=device)
        if self.tau_sampling == "logit_normal":
            return torch.sigmoid(torch.randn(bsz, device=device))
        if self.tau_sampling == "uniform":
            return torch.rand(bsz, device=device)
        raise ValueError(
            "train.loss.tau_sampling must be uniform, logit_normal, or zero; "
            f"got {self.tau_sampling!r}"
        )

    def _antipodal_site_mask(
        self,
        m_init: torch.Tensor,
        m_target: torch.Tensor,
    ) -> torch.Tensor:
        """Per-site validity mask: 1 where ``cos(m_init · m_target) > -(1-eps)``.

        Symmetric in the two arguments, so the inverse loss (which swaps
        endpoints before calling ``forward``) gets the same mask. Shape
        ``(B, H, W)``.
        """
        cos_pair = (m_init * m_target).sum(dim=1)
        threshold = -(1.0 - self.antipodal_eps)
        return (cos_pair > threshold).to(m_init.dtype)

    def _tau_weight(self, tau: torch.Tensor) -> torch.Tensor:
        if self.weight_mode == "mid_reweight":
            # w(τ) = max(1-τ, τ)^{-1}, bounded in [1, 2]; mild mid-segment
            # reweighting only — not SNR weighting (see plan2).
            return 1.0 / torch.maximum(1.0 - tau, tau).clamp_min(1e-3)
        return torch.ones_like(tau)

    def _target_rms_weight(
        self,
        target_velocity: torch.Tensor,
        cond: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cfg = self.target_rms_reweight
        if not bool(cfg.get("enabled", False)):
            return target_velocity.new_ones((target_velocity.shape[0],))
        mode = str(cfg.get("mode", "target_rms")).lower()
        eps = float(cfg.get("eps", 1e-4))
        power = float(cfg.get("power", 2.0))
        per_site_sq = target_velocity.detach().square().mean(dim=1)
        rms = masked_site_mean(per_site_sq, cond).sqrt()
        if mode in {"dt_rms", "per_dt", "per_dt_rms"}:
            bucket = cond.get("t_end_index", cond.get("dt_index", cond.get("dt_scale")))
            if torch.is_tensor(bucket) and bucket.shape[:1] == rms.shape[:1]:
                bucket = bucket.detach()
                grouped = rms.clone()
                for value in bucket.unique():
                    mask = bucket == value
                    grouped[mask] = rms[mask].mean()
                rms = grouped
        elif mode not in {"target_rms", "per_target", "sample_rms"}:
            raise ValueError(
                "train.loss.target_rms_reweight.mode must be target_rms or dt_rms, "
                f"got {mode!r}"
            )
        weight = (rms + eps).pow(-power)
        clamp = cfg.get("clamp")
        if clamp is not None:
            lo, hi = float(clamp[0]), float(clamp[1])
            weight = weight.clamp(lo, hi)
        if bool(cfg.get("normalize", True)):
            weight = weight / weight.mean().clamp_min(1e-12)
        return weight

    def _drive_family_weight(
        self,
        cond: dict[str, torch.Tensor],
        bsz: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Optional per-sample drive-family weighting for controlled probes."""
        cfg = self.drive_family_reweight
        if not bool(cfg.get("enabled", False)):
            return torch.ones(bsz, device=device, dtype=dtype)
        configured = cfg.get("weights", {}) or {}
        weights = {
            "none": float(configured.get("none", 1.0)),
            "field": float(configured.get("field", 1.0)),
            "zhang_li": float(configured.get("zhang_li", 1.0)),
            "slonczewski": float(configured.get("slonczewski", 1.0)),
            "sot": float(configured.get("sot", 1.0)),
        }
        if any(value < 0.0 for value in weights.values()):
            raise ValueError("train.loss.drive_family_reweight weights must be >= 0")

        out = torch.full((bsz,), weights["none"], device=device, dtype=dtype)

        def gate(key: str) -> torch.Tensor:
            value = cond.get(key)
            if not torch.is_tensor(value) or value.shape[:1] != (bsz,):
                return torch.zeros(bsz, device=device, dtype=torch.bool)
            return value.to(device=device).reshape(bsz, -1)[:, 0] > 0.5

        # The v4 generator chooses one drive family per condition. Applying
        # these in order also gives deterministic behaviour for legacy rows
        # that happen to expose more than one compatibility flag.
        out = torch.where(gate("has_field"), out.new_tensor(weights["field"]), out)
        out = torch.where(gate("has_zhang_li"), out.new_tensor(weights["zhang_li"]), out)
        out = torch.where(
            gate("has_slonczewski"), out.new_tensor(weights["slonczewski"]), out
        )
        out = torch.where(gate("has_sot"), out.new_tensor(weights["sot"]), out)
        if bool(cfg.get("normalize", True)):
            out = out / out.mean().clamp_min(1.0e-12)
        return out

    def _segment_role_weight(
        self,
        cond: dict[str, torch.Tensor],
        bsz: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Optionally emphasize underperforming control stages without test-ID leakage.

        Weights are keyed by the metadata-defined ``control_segment_index``.
        Normalizing within each batch keeps the global CFM loss scale stable.
        """

        cfg = self.segment_role_reweight
        if not bool(cfg.get("enabled", False)):
            return torch.ones(bsz, device=device, dtype=dtype)
        segment = cond.get("control_segment_index")
        if not torch.is_tensor(segment) or segment.shape[:1] != (bsz,):
            raise KeyError(
                "train.loss.segment_role_reweight requires control_segment_index "
                "in the fixed-time condition batch"
            )
        configured = cfg.get("weights", {}) or {}
        default = float(cfg.get("default_weight", 1.0))
        if default < 0.0:
            raise ValueError("segment_role_reweight.default_weight must be >= 0")
        out = torch.full((bsz,), default, device=device, dtype=dtype)
        flat = segment.to(device=device).reshape(bsz, -1)[:, 0].long()
        for raw_index, raw_weight in configured.items():
            index = int(raw_index)
            weight = float(raw_weight)
            if weight < 0.0:
                raise ValueError("segment_role_reweight weights must be >= 0")
            out = torch.where(flat == index, out.new_tensor(weight), out)
        if bool(cfg.get("normalize", True)):
            out = out / out.mean().clamp_min(1.0e-12)
        return out

    def _change_aware_site_weight(
        self,
        m_init: torch.Tensor,
        m_target: torch.Tensor,
        cond: dict[str, torch.Tensor],
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Weight CFM sites by endpoint angular change without changing its target.

        The raw weight is ``1 + (max_weight - 1) * min(d / saturation, 1)^power``
        for ``d = 1 - cos(M_init, M_target)``. Batch normalization preserves the
        average CFM scale while retaining a larger mean weight for high-change
        samples; per-sample normalization is also available for spatial-only
        reweighting.
        """

        cfg = self.change_aware_reweight
        if not bool(cfg.get("enabled", False)):
            return None

        max_weight = float(cfg.get("max_weight", 2.0))
        saturation = float(cfg.get("saturation", 0.5))
        power = float(cfg.get("power", 1.0))
        if max_weight < 1.0:
            raise ValueError("train.loss.change_aware_reweight.max_weight must be >= 1")
        if saturation <= 0.0:
            raise ValueError("train.loss.change_aware_reweight.saturation must be > 0")
        if power <= 0.0:
            raise ValueError("train.loss.change_aware_reweight.power must be > 0")

        normalize_cfg = cfg.get("normalize", "batch")
        if isinstance(normalize_cfg, bool):
            normalize_mode = "batch" if normalize_cfg else "none"
        else:
            normalize_mode = str(normalize_cfg).lower().replace("-", "_")
        if normalize_mode not in {"none", "batch", "sample", "per_sample"}:
            raise ValueError(
                "train.loss.change_aware_reweight.normalize must be "
                "none, batch, or sample"
            )

        init_unit = m_init.detach() / m_init.detach().norm(
            dim=1, keepdim=True
        ).clamp_min(1.0e-8)
        target_unit = m_target.detach() / m_target.detach().norm(
            dim=1, keepdim=True
        ).clamp_min(1.0e-8)
        cosine = (init_unit * target_unit).sum(dim=1).clamp(-1.0, 1.0)
        change = (1.0 - cosine).clamp(0.0, 2.0)
        progress = (change / saturation).clamp(0.0, 1.0).pow(power)
        weight = 1.0 + (max_weight - 1.0) * progress

        geometry = spatial_weight(cond, weight)
        active = torch.ones_like(weight) if geometry is None else geometry
        if valid_mask is not None:
            valid = spatial_weight(valid_mask, weight)
            if valid is not None:
                active = active * valid

        if normalize_mode == "batch":
            active_count = active.sum()
            active_mean = (weight * active).sum() / active_count.clamp_min(1.0)
            active_mean = torch.where(
                active_count > 0.0,
                active_mean,
                active_mean.new_ones(()),
            )
            weight = weight / active_mean.clamp_min(1.0e-12)
        elif normalize_mode in {"sample", "per_sample"}:
            reduce_dims = tuple(range(1, weight.ndim))
            active_count = active.sum(dim=reduce_dims, keepdim=True)
            active_mean = (weight * active).sum(
                dim=reduce_dims, keepdim=True
            ) / active_count.clamp_min(1.0)
            active_mean = torch.where(
                active_count > 0.0,
                active_mean,
                torch.ones_like(active_mean),
            )
            weight = weight / active_mean.clamp_min(1.0e-12)

        # The non-magnetic padding already has its own void_loss_weight. Keep
        # it neutral instead of treating its zero vectors as a large change.
        if geometry is not None:
            weight = weight * geometry + (1.0 - geometry)
        return weight.detach()

    def _group_labels(
        self,
        cond: dict[str, torch.Tensor],
        bsz: int,
    ) -> list[tuple[int | float, ...]]:
        keys = list(self.minibatch_ot.get("group_keys", ["t_end_index"]))
        float_tol = float(self.minibatch_ot.get("float_tolerance", 1e-6))
        labels: list[list[int | float]] = [[] for _ in range(bsz)]
        for key in keys:
            value = cond.get(key)
            if not torch.is_tensor(value) or value.shape[:1] != (bsz,):
                continue
            flat = value.detach().flatten(1)[:, 0] if value.ndim > 1 else value.detach()
            flat_cpu = flat.float().cpu() if torch.is_floating_point(flat) else flat.cpu()
            for i in range(bsz):
                if torch.is_floating_point(flat):
                    labels[i].append(round(float(flat_cpu[i]) / float_tol))
                else:
                    labels[i].append(int(flat_cpu[i]))
        return [tuple(x) if x else (0,) for x in labels]

    def _downsample_for_ot(self, x: torch.Tensor) -> torch.Tensor:
        stride = int(self.minibatch_ot.get("cost_downsample", 8))
        if stride > 1 and x.ndim == 4:
            x = F.avg_pool2d(x.float(), kernel_size=stride, stride=stride)
        return x.float().flatten(1)

    @staticmethod
    def _greedy_assignment(cost: torch.Tensor) -> torch.Tensor:
        n = cost.shape[0]
        order = cost.min(dim=1).values.argsort()
        used: set[int] = set()
        assignment = torch.empty(n, dtype=torch.long)
        for row in order.tolist():
            row_cost = cost[row].clone()
            if used:
                row_cost[list(used)] = float("inf")
            col = int(row_cost.argmin().item())
            used.add(col)
            assignment[row] = col
        return assignment

    def _linear_assignment(self, cost: torch.Tensor) -> torch.Tensor:
        try:
            from scipy.optimize import linear_sum_assignment  # type: ignore

            rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
            out = torch.empty(cost.shape[0], dtype=torch.long)
            out[torch.as_tensor(rows, dtype=torch.long)] = torch.as_tensor(cols, dtype=torch.long)
            return out
        except Exception:
            return self._greedy_assignment(cost.detach().cpu())

    def _source_cost_state(
        self,
        m_init: torch.Tensor,
        x_target: torch.Tensor,
        source_noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.bridge, RotationVectorBridge):
            return source_noise, x_target
        if isinstance(self.bridge, RotationVector2DBridge):
            return source_noise, self.bridge.target_from_omega(m_init, x_target)
        if isinstance(self.bridge, CartBridge):
            if getattr(self.bridge, "objective", "endpoint") == "residual":
                return source_noise, x_target
            if self.bridge.source == "anchored":
                return m_init + source_noise, x_target
            return source_noise, x_target
        if isinstance(self.bridge, RFMBridge):
            target = F.normalize(x_target, dim=1)
            if self.bridge.source == "anchored":
                tangent = project_to_tangent_chw(m_init, source_noise)
                return exp_map_chw(m_init, tangent), target
            return F.normalize(source_noise, dim=1), target
        raise TypeError(f"Unknown bridge type: {type(self.bridge)}")

    def _match_source_noise(
        self,
        m_init: torch.Tensor,
        x_target: torch.Tensor,
        source_noise: torch.Tensor | None,
        cond: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        cfg = self.minibatch_ot
        if source_noise is None or not bool(cfg.get("enabled", False)):
            return source_noise
        source_noise = apply_spatial_constraint(source_noise, cond)
        bsz = int(source_noise.shape[0])
        min_group = int(cfg.get("min_group_size", 2))
        max_group = int(cfg.get("max_group_size", 128))
        labels = self._group_labels(cond, bsz)
        groups: dict[tuple[int | float, ...], list[int]] = {}
        for i, label in enumerate(labels):
            groups.setdefault(label, []).append(i)
        matched = source_noise.clone()
        source_state, target_state = self._source_cost_state(m_init, x_target, source_noise)
        source_flat = self._downsample_for_ot(source_state)
        target_flat = self._downsample_for_ot(target_state)
        for idxs in groups.values():
            if len(idxs) < min_group or len(idxs) > max_group:
                continue
            idx = torch.as_tensor(idxs, device=source_noise.device, dtype=torch.long)
            cost = torch.cdist(target_flat[idx], source_flat[idx], p=2).square()
            cols_cpu = self._linear_assignment(cost)
            cols = cols_cpu.to(device=source_noise.device)
            matched[idx] = source_noise[idx[cols]]
        if isinstance(self.bridge, RFMBridge) and self.bridge.source == "anchored":
            matched = project_to_tangent_chw(m_init, matched)
        return matched

    # ------------------------------------------------------------------
    # Bridge-dependent source sampling.
    # ------------------------------------------------------------------
    def _sample_source(
        self,
        m_init: torch.Tensor,
        x_target: torch.Tensor,
        cond: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return (source_noise, alpha_omega0) for the current bridge."""
        if isinstance(self.bridge, CartBridge):
            if self.bridge.source == "identity":
                return None, None
            sigma = self.rotation_prior.sigma(cond, m_init)
            if getattr(self.bridge, "objective", "endpoint") == "residual":
                return sigma * torch.randn_like(x_target), None
            if self.cart_prior is None:
                raise ValueError("Cart bridge requires cart_prior")
            # ``sample`` returns (x_0, noise); CartBridge.build re-derives x_0
            # from (x_init, noise) so we forward only ``noise``.
            x0, source_noise = self.cart_prior.sample(m_init, sigma)
            # In ``anchored`` mode CartBridge expects source_noise = ``σε``
            # (added to m_init internally). In ``latent`` mode it expects
            # source_noise to *be* x_0 directly — encode that by passing the
            # full x_0 as source_noise: build() will then form
            # ``x_init + source_noise`` if anchored — which is incorrect. So
            # we override CartBridge.build with the canonical pattern: pass
            # ``x0 - x_init`` for anchored, and x0 for latent.
            if self.cart_prior.mode == "anchored":
                return x0 - m_init, None
            return x0, None
        if isinstance(self.bridge, RFMBridge):
            if self.bridge.source == "identity":
                return None, None
            sigma = self.rotation_prior.sigma(cond, m_init)
            if self.rfm_prior is None:
                return None, None
            if self.rfm_prior.mode == "anchored":
                # tangent kick at M_init — bridge will exp_map at M_init.
                tangent_noise = self.rfm_prior.sample_tangent_noise(m_init, sigma)
                return tangent_noise, None
            # latent: sample a random base point and a tangent kick, return x_0.
            base = self.rfm_prior.base_point(m_init)
            tangent = self.rfm_prior.sample_tangent_noise(base, sigma)
            if tangent is None:
                return base, None
            x0 = exp_map_chw(base, tangent)
            return x0, None
        if isinstance(self.bridge, RotationVectorBridge):
            source_mode = getattr(self.bridge, "source", "latent")
            if source_mode == "identity":
                omega0 = torch.zeros_like(x_target)
            elif source_mode == "anchored":
                sigma = self.rotation_prior.sigma(cond, m_init)
                tangent = sample_tangent_noise_chw(m_init, sigma)
                source_m = exp_map_chw(m_init, tangent)
                omega0 = rot_log_map_chw(m_init, source_m)
            elif source_mode == "latent":
                omega0 = self.rotation_prior.sample(x_target.shape, cond)
            else:
                raise ValueError(f"Unknown alpha source mode: {source_mode}")
            return omega0, omega0
        if isinstance(self.bridge, RotationVector2DBridge):
            source_mode = getattr(self.bridge, "source", "latent")
            if source_mode == "identity":
                coords0 = x_target.new_zeros(x_target.shape[0], 2, *x_target.shape[2:])
            elif source_mode == "anchored":
                sigma = self.rotation_prior.sigma(cond, m_init)
                tangent = sample_tangent_noise_chw(m_init, sigma)
                source_m = exp_map_chw(m_init, tangent)
                omega0 = rot_log_map_chw(m_init, source_m)
                coords0 = tangent_coords_from_omega_chw(m_init, omega0)
            elif source_mode == "latent":
                omega0 = self.rotation_prior.sample(x_target.shape, cond)
                coords0 = tangent_coords_from_omega_chw(m_init, omega0)
            else:
                raise ValueError(f"Unknown alpha2d source mode: {source_mode}")
            return coords0, coords0
        raise TypeError(f"Unknown bridge type: {type(self.bridge)}")

    # ------------------------------------------------------------------
    # Main entry.
    # ------------------------------------------------------------------
    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        cond: dict[str, torch.Tensor],
    ) -> LossOutput:
        m_init = batch.get("m_init", batch.get("m0"))
        m_target = batch.get("m_t", batch.get("m1"))
        omega_target = batch.get("omega_target")
        if m_init is None or m_target is None:
            raise KeyError("batch is missing m_init / m_t (or legacy m0 / m1)")
        bsz = m_init.shape[0]
        tau = self._sample_tau(bsz, m_init.device)
        w = self._tau_weight(tau)
        model_cond = apply_cfg_dropout(cond, self.cfg_dropout)

        # α-bridge loss is masked by alpha_t_end_ns_max when t_end_ns is provided.
        alpha_active = _is_alpha_like_bridge(self.bridge)
        if alpha_active and "t_end_ns" in cond:
            mask = cond["t_end_ns"] <= self.alpha_t_end_ns_max
            if not torch.all(mask):
                # We still compute the bridge but mask the loss contribution
                # for samples that exceed the cap; CFMLoss returns zero on
                # those entries.
                pass

        # Build the bridge endpoints and target.
        if alpha_active:
            if omega_target is None:
                raise KeyError("Mode α bridge requires omega_target in the batch")
            source_noise, _ = self._sample_source(m_init, omega_target, cond)
            source_noise = self._match_source_noise(m_init, omega_target, source_noise, cond)
            source_noise = apply_spatial_constraint(source_noise, cond)
            omega_target = apply_spatial_constraint(omega_target, cond)
            bridge_out = self.bridge.build(m_init, omega_target, tau, source_noise=source_noise)
        else:
            x_target = (
                m_target - m_init
                if isinstance(self.bridge, CartBridge)
                and getattr(self.bridge, "objective", "endpoint") == "residual"
                else m_target
            )
            if (
                isinstance(self.bridge, CartBridge)
                and getattr(self.bridge, "objective", "endpoint") == "residual"
            ):
                x_target = apply_spatial_constraint(x_target, cond)
            else:
                x_target = apply_spatial_constraint(
                    x_target,
                    cond,
                    outside=bridge_output_background(self.bridge, m_init, x_target),
                )
            source_noise, _ = self._sample_source(m_init, x_target, cond)
            source_noise = self._match_source_noise(m_init, x_target, source_noise, cond)
            source_noise = apply_spatial_constraint(source_noise, cond)
            bridge_out = self.bridge.build(m_init, x_target, tau, source_noise=source_noise)
        bridge_out.target_velocity = apply_spatial_constraint(bridge_out.target_velocity, cond)
        bridge_out.state = apply_spatial_constraint(
            bridge_out.state,
            cond,
            outside=bridge_state_background(self.bridge, m_init, bridge_out.state),
        )

        # Velocity network forward. For Cart / RFM the network receives ψ_τ /
        # μ_τ as the "state" channel; for α it receives Ω_τ.
        am_cfg = self.action_matching
        action_enabled = bool(am_cfg.get("enabled", False)) and float(am_cfg.get("weight", 0.0)) > 0.0
        pred_raw = model(m_init, bridge_out.state, tau, model_cond)
        pred = project_velocity_to_state(self.bridge, bridge_out.state, pred_raw)

        residual = pred - bridge_out.target_velocity
        # Optional per-site validity mask (plan-2 §forward line 87-89 + §inverse
        # antipodal handling). Only β-RFM needs this: its target velocity
        # ``θ (â × μ_τ)`` and its training label ``Log_p(q)`` divide by
        # ``sin θ``, which vanishes at antipodal sites. β-Cart's target lives
        # in R^3 (no sin θ in the denominator). We therefore gate
        # auto-injection on RFM specifically; the caller can still inject a
        # mask explicitly via ``batch["site_valid_mask"]`` for any bridge.
        site_mask = batch.get("site_valid_mask")
        if (
            site_mask is None
            and self.antipodal_eps > 0.0
            and isinstance(self.bridge, RFMBridge)
        ):
            site_mask = self._antipodal_site_mask(m_init, m_target)
        per_site_sq = residual.square().mean(dim=1)
        change_weight = self._change_aware_site_weight(
            m_init,
            m_target,
            cond,
            valid_mask=site_mask,
        )
        if change_weight is not None:
            per_site_sq = per_site_sq * change_weight.to(
                device=per_site_sq.device,
                dtype=per_site_sq.dtype,
            )
        per_sample_sq = weighted_site_mean(
            per_site_sq,
            cond,
            self.void_loss_weight,
            valid_mask=site_mask,
        )
        rms_weight = self._target_rms_weight(bridge_out.target_velocity, cond)
        drive_weight = self._drive_family_weight(
            cond,
            bsz,
            device=per_sample_sq.device,
            dtype=per_sample_sq.dtype,
        )
        segment_weight = self._segment_role_weight(
            cond,
            bsz,
            device=per_sample_sq.device,
            dtype=per_sample_sq.dtype,
        )
        per_sample_sq = per_sample_sq * rms_weight * drive_weight * segment_weight
        if alpha_active and "t_end_ns" in cond:
            cap_mask = (cond["t_end_ns"] <= self.alpha_t_end_ns_max).to(per_sample_sq.dtype)
            cfm = (w * cap_mask * per_sample_sq).sum() / cap_mask.sum().clamp_min(1.0)
        else:
            cfm = (w * per_sample_sq).mean()

        # plan-2 line 34 convex blend: L = (1-λ_α) L_β + λ_α L_α. The β-loss
        # we just computed is in ``cfm``; if alpha_mix > 0 we now compute the
        # α-loss using a *second* forward pass with the rotation-vector state
        # Ω_τ. Skipping the branch when alpha_mix == 0 keeps the default path
        # exactly as before.
        if self._alpha_aux_bridge is not None and not alpha_active:
            if omega_target is None:
                raise KeyError(
                    "alpha_mix > 0 requires omega_target in the batch; set "
                    "data.force_omega_target=true or train.loss.alpha_mix>0 "
                    "to make the fixed-time loader compute it."
                )
            alpha_source_noise = self.rotation_prior.sample(omega_target.shape, cond)
            alpha_source_noise = apply_spatial_constraint(alpha_source_noise, cond)
            omega_target = apply_spatial_constraint(omega_target, cond)
            alpha_out = self._alpha_aux_bridge.build(
                m_init, omega_target, tau, source_noise=alpha_source_noise
            )
            alpha_out.state = apply_spatial_constraint(
                alpha_out.state,
                cond,
                outside=0.0,
            )
            alpha_out.target_velocity = apply_spatial_constraint(alpha_out.target_velocity, cond)
            pred_alpha_raw = model(m_init, alpha_out.state, tau, model_cond)
            pred_alpha = project_velocity_to_state(self._alpha_aux_bridge, alpha_out.state, pred_alpha_raw)
            residual_alpha = pred_alpha - alpha_out.target_velocity
            per_site_sq_alpha = residual_alpha.square().mean(dim=1)
            if change_weight is not None:
                per_site_sq_alpha = per_site_sq_alpha * change_weight.to(
                    device=per_site_sq_alpha.device,
                    dtype=per_site_sq_alpha.dtype,
                )
            per_sample_sq_alpha = weighted_site_mean(
                per_site_sq_alpha,
                cond,
                self.void_loss_weight,
            )
            per_sample_sq_alpha = per_sample_sq_alpha * drive_weight * segment_weight
            if "t_end_ns" in cond:
                cap_mask = (cond["t_end_ns"] <= self.alpha_t_end_ns_max).to(per_sample_sq_alpha.dtype)
                cfm_alpha = (w * cap_mask * per_sample_sq_alpha).sum() / cap_mask.sum().clamp_min(1.0)
            else:
                cfm_alpha = (w * per_sample_sq_alpha).mean()
            cfm = (1.0 - self.alpha_mix) * cfm + self.alpha_mix * cfm_alpha

        # Auxiliary losses are only meaningful when the predicted state lives
        # on / near S^2.
        unit = m_init.new_tensor(0.0)
        llg = m_init.new_tensor(0.0)
        topo = m_init.new_tensor(0.0)

        if (self.unit_weight > 0.0 or self.llg_weight > 0.0 or self.topo_weight > 0.0):
            if isinstance(self.bridge, CartBridge):
                # For Euclidean Cart paths the velocity target is the full
                # endpoint displacement, so the one-pass endpoint estimate is
                # source + v_hat.  Using the intermediate x_tau here would
                # double count tau portions of that displacement.  The legacy
                # normalized-sphere path instead uses a local next-state
                # proxy, for which state + v_hat remains intentional.
                if (
                    getattr(self.bridge, "objective", "endpoint") == "residual"
                    or not getattr(self.bridge, "normalize_path", True)
                ):
                    step = bridge_out.source + pred
                else:
                    step = bridge_out.state + pred
                if getattr(self.bridge, "objective", "endpoint") == "residual":
                    m_t_hat = m_init + step
                    norm = m_t_hat.norm(dim=1)
                    if self.unit_weight > 0.0:
                        unit = masked_site_mean((norm - 1.0).square(), cond).mean()
                    m_t_hat = m_t_hat / norm.clamp_min(1e-6).unsqueeze(1)
                else:
                    # ψ_τ + h · v (small-step proxy) is the next-step state
                    # under the integrated ODE; for the unit term we instead
                    # push the predicted velocity onto the surface and check
                    # that an extrapolated step still lies near S^2.
                    norm = step.norm(dim=1)
                    if self.unit_weight > 0.0:
                        unit = masked_site_mean((norm - 1.0).square(), cond).mean()
                    m_t_hat = step / norm.clamp_min(1e-6).unsqueeze(1)
            elif isinstance(self.bridge, RFMBridge):
                # RFM stays on S^2 by construction; the unit residual is
                # functionally zero but we report it for symmetry.
                m_t_hat = self.bridge.reconstruct(bridge_out.state, pred)
                if self.unit_weight > 0.0:
                    unit = masked_site_mean((m_t_hat.norm(dim=1) - 1.0).square(), cond).mean()
            else:  # alpha / alpha2d bridge
                if isinstance(self.bridge, RotationVector2DBridge):
                    m_t_hat = self.bridge.reconstruct(m_init, bridge_out.source + pred)
                else:
                    omega_hat = bridge_out.source + pred
                    m_t_hat = rodrigues_chw(omega_hat, m_init)
                if self.unit_weight > 0.0:
                    unit = masked_site_mean((m_t_hat.norm(dim=1) - 1.0).square(), cond).mean()
            if self.llg_weight > 0.0:
                dt_seconds = cond.get("t_end_s", cond.get("dt_s"))
                # plan-1 only applied LLG for small dt; we generalise to
                # ``dt_scale <= llg_max_scale`` so legacy configs still work.
                if "dt_scale" in cond:
                    mask = cond["dt_scale"] <= self.llg_max_scale
                else:
                    mask = torch.ones(bsz, dtype=torch.bool, device=m_init.device)
                if mask.any():
                    sub = {k: (v[mask] if isinstance(v, torch.Tensor) and v.shape[:1] == (bsz,) else v) for k, v in cond.items()}
                    sub_dt = dt_seconds[mask] if dt_seconds is not None else m_init.new_full((int(mask.sum()),), 1e-12)
                    residual_llg = self._llg_residual(m_init[mask], m_t_hat[mask], sub, sub_dt)
                    weight = 1.0 / (1.0 + 1e-2 * sub["temp_k"])
                    while weight.ndim < residual_llg.ndim:
                        weight = weight[..., None]
                    llg_per_site = (weight * residual_llg.square()).mean(dim=1)
                    llg = masked_site_mean(llg_per_site, sub).mean()
            if self.topo_weight > 0.0:
                topo = (
                    topological_charge(m_t_hat, boundary=self.boundary)
                    - topological_charge(m_target, boundary=self.boundary)
                ).abs().mean()

        action = m_init.new_tensor(0.0)
        action_valid_fraction = m_init.new_tensor(0.0)
        if action_enabled:
            if not isinstance(self.bridge, RotationVectorBridge):
                raise NotImplementedError("action_matching currently supports bridge.state_repr=alpha only")
            action_state = batch.get("action_state")
            action_velocity = batch.get("action_velocity")
            action_tau = batch.get("action_tau")
            action_valid = batch.get("action_valid")
            if (
                torch.is_tensor(action_state)
                and torch.is_tensor(action_velocity)
                and torch.is_tensor(action_tau)
                and torch.is_tensor(action_valid)
            ):
                max_batch = int(am_cfg.get("max_batch_per_rank", 0) or 0)
                am_bsz = min(bsz, max_batch) if max_batch > 0 else bsz
                action_state = action_state[:am_bsz]
                action_velocity = action_velocity[:am_bsz]
                action_tau = action_tau[:am_bsz]
                action_valid = action_valid[:am_bsz]
                bsz_am, k_am = int(action_state.shape[0]), int(action_state.shape[1])
                real_state = action_state.reshape(bsz_am * k_am, *action_state.shape[2:])
                real_velocity = action_velocity.reshape(bsz_am * k_am, *action_velocity.shape[2:])
                flat_tau = action_tau.reshape(bsz_am * k_am).to(device=m_init.device, dtype=m_init.dtype)
                flat_valid = action_valid.reshape(bsz_am * k_am).to(device=m_init.device, dtype=torch.bool)
                source = bridge_out.source[:bsz_am].unsqueeze(1).expand(
                    -1,
                    k_am,
                    *bridge_out.source.shape[1:],
                ).reshape(bsz_am * k_am, *bridge_out.source.shape[1:])
                tau_b = flat_tau.reshape(-1, *([1] * (real_state.ndim - 1)))
                flat_state = ((1.0 - tau_b) * source + real_state).detach()
                flat_target = (real_velocity - source).detach()
                flat_m_init = m_init[:bsz_am].unsqueeze(1).expand(-1, k_am, *m_init.shape[1:]).reshape(
                    bsz_am * k_am,
                    *m_init.shape[1:],
                )
                am_cond: dict[str, torch.Tensor] = {}
                for key, value in model_cond.items():
                    if torch.is_tensor(value) and value.shape[:1] == (bsz,):
                        value_am = value[:bsz_am]
                        am_cond[key] = value_am.unsqueeze(1).expand(-1, k_am, *value_am.shape[1:]).reshape(
                            bsz_am * k_am,
                            *value.shape[1:],
                        )
                    else:
                        am_cond[key] = value
                path_pred_raw = model(flat_m_init, flat_state, flat_tau, am_cond)
                path_pred = project_velocity_to_state(self.bridge, flat_state, path_pred_raw)
                residual_action = path_pred - flat_target
                per_item = residual_action.square().mean(dim=tuple(range(1, residual_action.ndim)))
                if bool(flat_valid.any().item()):
                    action = per_item[flat_valid].mean()
                else:
                    action = per_item.sum() * 0.0
                action_valid_fraction = flat_valid.float().mean()

        total = (
            self.cfm_weight * cfm
            + self.unit_weight * unit
            + self.llg_weight * llg
            + self.topo_weight * topo
            + float(am_cfg.get("weight", 0.0)) * action
        )
        return LossOutput(
            total=total,
            cfm=cfm.detach(),
            unit=unit.detach(),
            llg=llg.detach(),
            topo=topo.detach(),
            endpoint=m_init.new_tensor(0.0),
            action=action.detach(),
            action_valid_fraction=action_valid_fraction.detach(),
        )
