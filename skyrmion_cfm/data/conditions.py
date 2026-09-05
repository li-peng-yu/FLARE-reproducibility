from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
from torch import nn


# Per-trajectory scalar fields that may carry variation across the dataset.
# ``FullConditionEmbedder`` will only encode the ones whose training-set std
# exceeds ``min_std`` (see :func:`audit_scalar_conditions`).
SCALAR_CONDITION_KEYS: tuple[str, ...] = (
    "t_end_s",
    "temp_k",
    "b_x_t",
    "b_y_t",
    "b_z_t",
    "current_a_m2",
    "alpha",
    "aex_j_per_m",
    "dind_j_per_m2",
    "ku1_j_per_m3",
    "msat_a_per_m",
    "pol_eff",
    "epsilon_prime",
    "fixed_layer_x",
    "fixed_layer_y",
    "fixed_layer_z",
    "anis_u_x",
    "anis_u_y",
    "anis_u_z",
    "cubic_axis_1_x",
    "cubic_axis_1_y",
    "cubic_axis_1_z",
    "cubic_axis_2_x",
    "cubic_axis_2_y",
    "cubic_axis_2_z",
    "cubic_axis_3_x",
    "cubic_axis_3_y",
    "cubic_axis_3_z",
    "anchor_offset_s",
    "anchor_offset_norm",
    "target_offset_norm",
    "temperature_start_k",
    "temperature_segment_midpoint_k",
    "temperature_over_tc",
    "m_reduced",
    "ms_t_a_per_m",
    "a_t_j_per_m",
    "ku_t_j_per_m3",
    "d_t_j_per_m2",
    "alpha_t",
    "kc1_t_j_per_m3",
    "kc2_t_j_per_m3",
    "b1_t_j_per_m3",
    "b2_t_j_per_m3",
    "ms_ref_a_per_m",
    "a_ref_j_per_m",
    "ku_ref_j_per_m3",
    "d_ref_j_per_m2",
    "alpha_ref",
    "thickness_m",
    "tc_k",
    "theta_t",
    "qk_ref",
    "kappa_d_ref",
    "d_d_over_dc_ref",
    "ld_cells_ref",
    "rex",
    "r_min_ref",
    "delta_dw_cells_ref",
    "domain_wall_width_cells_ref",
    "rex_t",
    "r_min_t",
    "ld_cells_t",
    "delta_dw_cells_t",
    "domain_wall_width_cells_t",
    "b_ext_x_t",
    "b_ext_y_t",
    "b_ext_z_t",
    "j_vector_x_a_per_m2",
    "j_vector_y_a_per_m2",
    "j_vector_z_a_per_m2",
    "j_abs_a_per_m2",
    "charge_current_x_a_per_m2",
    "charge_current_y_a_per_m2",
    "charge_current_z_a_per_m2",
    "polarization_x",
    "polarization_y",
    "polarization_z",
    "lambda_sl",
    "theta_dl_eff",
    "r_fl_dl",
    "sot_b_dl_t",
    "sot_b_fl_t",
    "sot_explicit_b_dl_t",
    "sot_explicit_b_fl_t",
    "pre_relaxation_flag",
    "drive_active_flag",
    "has_field",
    "has_zhang_li",
    "has_slonczewski",
    "has_sot",
    "has_sot_like_slonczewski_proxy",
    "theta_T_T_a_K",
    "theta_T_T_b_K",
    "theta_T_s_time",
    "theta_T_n_cycle",
    "theta_T_delta_T_K",
    "theta_T_x0_norm",
    "theta_T_y0_norm",
    "theta_T_sigma_norm",
    "theta_T_u0_norm",
    "theta_T_w_norm",
    "temp_param_beta_ms",
    "temp_param_m_red_min",
    "temp_param_m_red_max",
    "temp_param_p_a",
    "temp_param_p_ku",
    "temp_param_p_d",
    "temp_param_p_kc",
    "temp_param_p_kc2",
    "temp_param_p_b1",
    "temp_param_p_b2",
    "temp_param_c_alpha",
    "temp_param_alpha_min",
    "temp_param_alpha_max",
)


DEFAULT_EMBEDDED_SCALAR_KEYS: tuple[str, ...] = (
    "alpha",
    "aex_j_per_m",
    "dind_j_per_m2",
    "ku1_j_per_m3",
    "msat_a_per_m",
    "pol_eff",
    "epsilon_prime",
    "fixed_layer_x",
    "fixed_layer_y",
    "fixed_layer_z",
)


V4_EMBEDDED_SCALAR_KEYS: tuple[str, ...] = DEFAULT_EMBEDDED_SCALAR_KEYS + (
    "anis_u_x",
    "anis_u_y",
    "anis_u_z",
    "cubic_axis_1_x",
    "cubic_axis_1_y",
    "cubic_axis_1_z",
    "cubic_axis_2_x",
    "cubic_axis_2_y",
    "cubic_axis_2_z",
    "cubic_axis_3_x",
    "cubic_axis_3_y",
    "cubic_axis_3_z",
    "anchor_offset_s",
    "anchor_offset_norm",
    "target_offset_norm",
    "temperature_start_k",
    "temperature_segment_midpoint_k",
    "temperature_over_tc",
    "m_reduced",
    "ms_t_a_per_m",
    "a_t_j_per_m",
    "ku_t_j_per_m3",
    "d_t_j_per_m2",
    "alpha_t",
    "kc1_t_j_per_m3",
    "kc2_t_j_per_m3",
    "b1_t_j_per_m3",
    "b2_t_j_per_m3",
    "ms_ref_a_per_m",
    "a_ref_j_per_m",
    "ku_ref_j_per_m3",
    "d_ref_j_per_m2",
    "alpha_ref",
    "thickness_m",
    "tc_k",
    "theta_t",
    "qk_ref",
    "kappa_d_ref",
    "d_d_over_dc_ref",
    "ld_cells_ref",
    "rex",
    "r_min_ref",
    "delta_dw_cells_ref",
    "domain_wall_width_cells_ref",
    "rex_t",
    "r_min_t",
    "ld_cells_t",
    "delta_dw_cells_t",
    "domain_wall_width_cells_t",
    "b_ext_x_t",
    "b_ext_y_t",
    "b_ext_z_t",
    "j_vector_x_a_per_m2",
    "j_vector_y_a_per_m2",
    "j_vector_z_a_per_m2",
    "j_abs_a_per_m2",
    "polarization_x",
    "polarization_y",
    "polarization_z",
    "lambda_sl",
    "pre_relaxation_flag",
    "drive_active_flag",
    "has_field",
    "has_zhang_li",
    "has_slonczewski",
    "has_sot",
    "has_sot_like_slonczewski_proxy",
    "temp_param_beta_ms",
    "temp_param_m_red_min",
    "temp_param_m_red_max",
    "temp_param_p_a",
    "temp_param_p_ku",
    "temp_param_p_d",
    "temp_param_p_kc",
    "temp_param_p_kc2",
    "temp_param_p_b1",
    "temp_param_p_b2",
    "temp_param_c_alpha",
    "temp_param_alpha_min",
    "temp_param_alpha_max",
)


V4_T_THETA_KEYS: tuple[str, ...] = (
    "T_a_K",
    "T_b_K",
    "s_time",
    "n_cycle",
    "delta_T_K",
    "x0_norm",
    "y0_norm",
    "sigma_norm",
    "u0_norm",
    "w_norm",
)


V4_CATEGORICAL_ORDERS: dict[str, tuple[str, ...]] = {
    "dataset_profile": ("core_relax", "driven_segment", "spinwave_fast", "long_final", "edge_rare"),
    "material_family": ("inplane_soft", "pma_no_dmi", "pma_idmi", "q1_stripe_competition", "bulk_dmi_2d"),
    "geometry_mode": (
        "full_rectangle",
        "nanostrip",
        "disk",
        "ellipse",
        "notched_strip",
        "ring",
        "antidot_or_holes",
        "smooth_polygon",
    ),
    "boundary_mode": ("open", "pbc_x", "pbc_y", "pbc_xy", "absorbing_edge"),
    "init_family": (
        "uniform_near_uniform",
        "smooth_random",
        "domain_wall",
        "stripe_labyrinth",
        "bubble_skyrmion",
        "vortex_antivortex",
        "spinwave_fmr_seed",
    ),
    "DMI_TYPE": ("none", "interfacial", "bulk"),
    "TEMP_PARAM_MODE": ("temp_params_off", "simple_power_law", "family_power_law", "strong_temp_drift_edge"),
    "T_schedule_mode": (
        "isothermal",
        "warmup_hold",
        "anneal_hold",
        "warmup_anneal_cycle",
        "quench_relax",
        "local_temperature_spot",
    ),
    "CUBIC_ANISOTROPY_MODE": (
        "cubic_off",
        "weak_cubic",
        "moderate_cubic",
        "mixed_uniaxial_cubic",
        "cubic_dominant_edge",
    ),
    "MAGNETOELASTIC_MODE": (
        "magnetoelastic_off",
        "uniform_strain_weak",
        "uniform_strain_moderate",
        "strain_gradient",
        "local_strain_defect",
    ),
    "DEFECT_MODE": ("uniform_material", "grain_like_disorder", "discrete_defect_or_interface"),
    "TIME_REGIME": ("fast", "standard", "long", "final_relax"),
    "drive_type": (
        "none",
        "static_field",
        "field_quench",
        "field_pulse_or_local_field",
        "zhang_li_stt",
        "slonczewski_stt",
        "sot",
        "initial_spinwave_only_then_free",
        "weak_current",
        "strong_field_pulse",
        "strong_current",
        "local_write_delete",
        "topology_collision_setup",
    ),
    "drive_type_rendered": (
        "none",
        "static_field",
        "field_quench",
        "field_pulse_or_local_field",
        "zhang_li_stt",
        "slonczewski_stt",
        "sot",
        "initial_spinwave_only_then_free",
        "weak_current",
        "strong_field_pulse",
        "strong_current",
        "local_write_delete",
        "topology_collision_setup",
    ),
    "rendered_torque_model": (
        "none",
        "field",
        "zhang_li",
        "slonczewski",
        "slonczewski_fixed_layer_proxy",
        "effective_spin_hall_sot_via_slonczewski_lambda1",
        "mixed_or_none",
    ),
    "drive_active_kind": ("none", "field", "zhang_li", "slonczewski", "sot"),
    "segment_role": ("pre_relaxation", "condition"),
}


V4_CATEGORICAL_KEYS: tuple[str, ...] = tuple(V4_CATEGORICAL_ORDERS)


@dataclass
class ConditionStats:
    """Per-key mean/std of the scalar conditions, used for z-scoring."""

    mean: torch.Tensor
    std: torch.Tensor
    keys: tuple[str, ...] = field(default_factory=lambda: SCALAR_CONDITION_KEYS)

    @classmethod
    def from_records(
        cls,
        rows: Iterable[dict[str, float]],
        keys: tuple[str, ...] | None = None,
    ) -> "ConditionStats":
        keys = tuple(keys) if keys is not None else SCALAR_CONDITION_KEYS
        values: list[list[float]] = []
        for row in rows:
            values.append([float(row.get(k, 0.0)) for k in keys])
        if not values:
            mean = torch.zeros(len(keys))
            std = torch.ones(len(keys))
        else:
            x = torch.tensor(values, dtype=torch.float32)
            mean = x.mean(dim=0)
            std = x.std(dim=0).clamp_min(1e-6)
        return cls(mean=mean, std=std, keys=keys)


@dataclass
class AuditedCondition:
    """Outcome of label_audit for a single scalar key.

    ``enabled`` indicates whether the field varies enough across the training
    set to be worth encoding. Disabled fields are still recorded so the
    embedder knows to skip them (and Mode B knows to drop them from the
    inverse output head).
    """

    key: str
    mean: float
    std: float
    enabled: bool
    coverage: float = 1.0


def audit_scalar_conditions(
    rows: Iterable[dict[str, float]],
    keys: tuple[str, ...] = SCALAR_CONDITION_KEYS,
    min_std: float = 1e-6,
    coverage_threshold: float = 0.9,
) -> dict[str, AuditedCondition]:
    """Return per-key audit decisions.

    ``enabled = True`` iff ``coverage >= coverage_threshold`` *and*
    ``empirical_std > min_std``. ``coverage`` is the fraction of rows that
    provided a *finite* value for that key (NaN / None counted as missing).
    """
    rows = list(rows)
    n = max(1, len(rows))
    out: dict[str, AuditedCondition] = {}
    for k in keys:
        values: list[float] = []
        for row in rows:
            v = row.get(k)
            if v is None:
                continue
            fv = float(v)
            if fv != fv:  # NaN
                continue
            values.append(fv)
        coverage = len(values) / n
        if values:
            t = torch.tensor(values, dtype=torch.float32)
            mean = float(t.mean())
            std = float(t.std().clamp_min(0.0))
        else:
            mean = 0.0
            std = 0.0
        enabled = (coverage >= coverage_threshold) and (std > min_std)
        out[k] = AuditedCondition(key=k, mean=mean, std=std, enabled=enabled, coverage=coverage)
    return out


def sinusoidal_embedding(x: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal positional encoding of a scalar feature."""
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, device=x.device, dtype=x.dtype))
        * torch.arange(half, device=x.device, dtype=x.dtype)
        / max(half, 1)
    )
    args = x[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class FullConditionEmbedder(nn.Module):
    """Build a 512-d condition vector and a 2-channel spatial conditioning map.

    Encodes:
      - ``t_end`` index (lookup, 128) + ``log t_end`` sinusoidal (64)
      - temperature (64)
      - magnetic field B (sin per component, fused → 64)
      - SOT current (64)
      - τ flow time (128)

    Material scalars from ``SCALAR_CONDITION_KEYS`` are also encoded when the
    audit marks them enabled. Output dim is fixed at 512: we project the
    concatenated raw embedding through a final Linear so adding / removing
    audited fields does not break shape contracts downstream.

    Spatial conditioning (``defect_field``, ``j_field``) is passed straight
    through when enabled in the velocity network. Optionally, the 8x8
    ``control_grid`` can also be encoded as 64 scalar J values and fused into
    the 512-d condition vector.
    """

    def __init__(
        self,
        t_end_buckets: list[float] | tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0),
        cond_dim: int = 512,
        stats: ConditionStats | None = None,
        audit: dict[str, AuditedCondition] | None = None,
        dt_scales: list[int] | None = None,
        disable_dt_condition: bool = False,
        mask_target_time: bool = False,
        pulse_condition: bool = False,
        j_scalar_embedding: bool = False,
        j_scalar_grid_shape: tuple[int, int] = (8, 8),
        j_scalar_scale: float = 1.0,
        scalar_condition_keys: tuple[str, ...] | list[str] | None = None,
        categorical_condition_dims: dict[str, int] | None = None,
        force_scalar_condition_keys: tuple[str, ...] | list[str] | None = None,
        scalar_condition_scales: dict[str, float] | None = None,
        physics_phase_condition: bool = False,
        physics_phase_gamma_hz_per_t: float = 28.0e9,
        drive_phase_condition: bool = False,
        drive_phase_gamma_hz_per_t: float = 28.0e9,
    ) -> None:
        super().__init__()
        if cond_dim != 512:
            raise ValueError("FullConditionEmbedder implements the 512-d condition vector")
        self.cond_dim = int(cond_dim)
        self.t_end_buckets = tuple(float(t) for t in t_end_buckets)
        # Lookup table works equally well for the plan-2 fixed-time buckets and
        # legacy plan-1 dt_scales when the latter is supplied.
        self.dt_scales = list(dt_scales) if dt_scales is not None else None
        n_buckets = len(self.dt_scales) if self.dt_scales is not None else max(len(self.t_end_buckets), 1)
        self.t_lookup = nn.Embedding(n_buckets, 128)
        self.b_linear = nn.Linear(3 * 64, 64)
        self.material_linear: nn.Linear | None = None
        self.categorical_linear: nn.Linear | None = None
        self.j_grid_linear: nn.Linear | None = None
        self.disable_dt_condition = bool(disable_dt_condition)
        self.mask_target_time = bool(mask_target_time)
        self.pulse_condition = bool(pulse_condition)
        self.j_scalar_embedding = bool(j_scalar_embedding)
        self.j_scalar_grid_shape = (int(j_scalar_grid_shape[0]), int(j_scalar_grid_shape[1]))
        if self.j_scalar_grid_shape[0] <= 0 or self.j_scalar_grid_shape[1] <= 0:
            raise ValueError("j_scalar_grid_shape dimensions must be positive")
        self.j_scalar_scale = float(j_scalar_scale)
        if self.j_scalar_scale <= 0.0:
            raise ValueError("j_scalar_scale must be positive")
        self.audit = audit or {}
        # Scalar controls that survived the audit are flattened into a single
        # sinusoidal embedding stack and projected to 64-d.
        requested_scalar_keys = (
            tuple(str(k) for k in scalar_condition_keys)
            if scalar_condition_keys is not None
            else DEFAULT_EMBEDDED_SCALAR_KEYS
        )
        self.force_scalar_condition_keys = frozenset(
            str(k) for k in (force_scalar_condition_keys or ())
        )
        self.scalar_condition_scales = {
            str(k): float(v)
            for k, v in (scalar_condition_scales or {}).items()
            if float(v) > 0.0
        }
        self.enabled_material_keys = tuple(
            k for k in requested_scalar_keys
            if k in self.force_scalar_condition_keys
            or (k not in self.audit)
            or self.audit[k].enabled
        )
        if self.enabled_material_keys:
            self.material_linear = nn.Linear(64 * len(self.enabled_material_keys), 64)
        self.categorical_condition_dims = {
            str(k): int(v)
            for k, v in (categorical_condition_dims or {}).items()
            if int(v) > 0
        }
        if self.categorical_condition_dims:
            self.categorical_linear = nn.Linear(sum(self.categorical_condition_dims.values()), 64)
        if self.j_scalar_embedding:
            self.j_grid_linear = nn.Linear(64 * self.j_scalar_grid_shape[0] * self.j_scalar_grid_shape[1], 64)
        self.physics_phase_condition = bool(physics_phase_condition)
        self.physics_phase_gamma_hz_per_t = float(physics_phase_gamma_hz_per_t)
        self.physics_phase_linear: nn.Linear | None = None
        if self.physics_phase_condition:
            if self.physics_phase_gamma_hz_per_t <= 0.0:
                raise ValueError("physics_phase_gamma_hz_per_t must be positive")
            # Explicitly expose the otherwise multiplicative/periodic
            # B-times-dt dependence.  Existing embeddings see B and log(dt)
            # separately, which is unnecessarily hard in the multi-cycle
            # precession regime.
            self.physics_phase_linear = nn.Linear(6, 64)
        self.drive_phase_condition = bool(drive_phase_condition)
        self.drive_phase_gamma_hz_per_t = float(drive_phase_gamma_hz_per_t)
        self.drive_phase_linear: nn.Linear | None = None
        if self.drive_phase_condition:
            if self.drive_phase_gamma_hz_per_t <= 0.0:
                raise ValueError("drive_phase_gamma_hz_per_t must be positive")
            # 8 Slonczewski/SOT phase features, 4 Zhang-Li displacement
            # features, three torque-family gates, and beta_ZL.
            self.drive_phase_linear = nn.Linear(16, 64)
        if stats is None:
            mean = torch.zeros(len(SCALAR_CONDITION_KEYS))
            std = torch.ones(len(SCALAR_CONDITION_KEYS))
            keys = SCALAR_CONDITION_KEYS
        else:
            mean, std = stats.mean.float(), stats.std.float()
            keys = stats.keys
        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp_min(1e-6))
        self._key_to_index = {k: i for i, k in enumerate(keys)}
        raw_dim = 128 + 64 + 64 + 64 + 64 + 128  # = 512 baseline
        if self.material_linear is not None:
            raw_dim += 64
        if self.categorical_linear is not None:
            raw_dim += 64
        if self.pulse_condition:
            raw_dim += 64 * 3
        if self.j_grid_linear is not None:
            raw_dim += 64
        if self.physics_phase_linear is not None:
            raw_dim += 64
        if self.drive_phase_linear is not None:
            raw_dim += 64
        self.project = nn.Linear(raw_dim, self.cond_dim)

    def _scalar(self, cond: dict[str, torch.Tensor], key: str, default: float = 0.0) -> torch.Tensor:
        if key in cond:
            return cond[key].float()
        device = next(iter(cond.values())).device if cond else torch.device("cpu")
        return torch.zeros(1, dtype=torch.float32, device=device).fill_(default)

    def _zscore(self, key: str, value: torch.Tensor) -> torch.Tensor:
        idx = self._key_to_index.get(key)
        explicit_scale = self.scalar_condition_scales.get(key)
        if explicit_scale is not None:
            center = 0.0 if idx is None or idx >= self.mean.numel() else self.mean[idx]
            return (value - center) / float(explicit_scale)
        if idx is None or idx >= self.mean.numel():
            return value
        return (value - self.mean[idx]) / self.std[idx]

    def _control_grid(self, cond: dict[str, torch.Tensor], batch: int, device: torch.device) -> torch.Tensor:
        grid = cond.get("control_grid")
        if grid is None:
            return torch.zeros(batch, *self.j_scalar_grid_shape, dtype=torch.float32, device=device)
        grid = grid.float().to(device=device)
        if grid.ndim == 4 and grid.shape[1] == 1:
            grid = grid[:, 0]
        elif grid.ndim == 2:
            grid = grid.unsqueeze(0)
        if grid.ndim != 3:
            raise ValueError(f"control_grid must have shape (B, H, W), got {tuple(grid.shape)}")
        if tuple(grid.shape[-2:]) != self.j_scalar_grid_shape:
            raise ValueError(
                f"control_grid shape {tuple(grid.shape[-2:])} does not match "
                f"j_scalar_grid_shape {self.j_scalar_grid_shape}"
            )
        if grid.shape[0] != batch:
            raise ValueError(f"control_grid batch {grid.shape[0]} does not match tau batch {batch}")
        return grid

    def _categorical_features(
        self,
        cond: dict[str, torch.Tensor],
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for key, dim in self.categorical_condition_dims.items():
            x = cond.get(f"{key}_onehot", cond.get(key))
            if x is None:
                parts.append(torch.zeros(batch, dim, dtype=torch.float32, device=device))
                continue
            x = x.to(device=device)
            if x.ndim == 1:
                if x.dtype.is_floating_point:
                    if dim != 1:
                        raise ValueError(f"{key} one-hot must have dim {dim}, got scalar")
                    x = x.float().reshape(batch, 1)
                else:
                    x = torch.nn.functional.one_hot(x.long().clamp_min(0).clamp_max(dim - 1), num_classes=dim)
            elif x.ndim != 2:
                raise ValueError(f"{key} one-hot must have shape (B, {dim}), got {tuple(x.shape)}")
            if x.shape[0] != batch or x.shape[1] != dim:
                raise ValueError(f"{key} one-hot must have shape ({batch}, {dim}), got {tuple(x.shape)}")
            parts.append(x.float())
        return torch.cat(parts, dim=-1)

    def forward(self, tau: torch.Tensor, cond: dict[str, torch.Tensor]) -> torch.Tensor:
        device = tau.device
        # Time-endpoint index. We pull from ``t_end_index`` for the plan-2 path
        # and fall back to legacy ``dt_index`` for plan-1.
        if "t_end_index" in cond:
            idx = cond["t_end_index"].long().clamp_min(0)
            idx = idx.clamp_max(self.t_lookup.num_embeddings - 1)
            t_log = torch.log(cond.get("t_end_s", cond.get("dt_s", torch.ones_like(idx, dtype=torch.float32))).clamp_min(1e-15))
        else:
            idx = cond["dt_index"].long().clamp_max(self.t_lookup.num_embeddings - 1)
            t_log = torch.log(cond["dt_s"].clamp_min(1e-15))
        t_lookup = self.t_lookup(idx)
        t_sin = sinusoidal_embedding(t_log, 64)
        if self.disable_dt_condition or self.mask_target_time:
            # Preserve the embedding parameter in the autograd graph so DDP
            # still reduces its (zero) gradient for the masked-time ablation.
            t_lookup = t_lookup * 0.0
            t_sin = torch.zeros_like(t_sin)
        temp = cond.get("temp_k", torch.zeros(idx.shape[0], device=device))
        temp_z = self._zscore("temp_k", temp.float())
        temp_sin = sinusoidal_embedding(temp_z, 64)
        b = cond.get("b_t", torch.zeros(idx.shape[0], 3, device=device)).float()
        b_components: list[torch.Tensor] = []
        for ci, key in enumerate(("b_x_t", "b_y_t", "b_z_t")):
            b_components.append(sinusoidal_embedding(self._zscore(key, b[:, ci]), 64))
        b_emb = self.b_linear(torch.cat(b_components, dim=-1))
        current = cond.get("current_a_m2", torch.zeros(idx.shape[0], device=device)).float()
        current_z = self._zscore("current_a_m2", current)
        current_sin = sinusoidal_embedding(current_z, 64)
        tau_emb = sinusoidal_embedding(tau.float(), 128)
        feats = [t_lookup, t_sin, temp_sin, b_emb, current_sin, tau_emb]
        if self.physics_phase_linear is not None:
            # gamma/(2*pi) * |B| * dt gives precession cycles.  LLG damping
            # changes the angular frequency by 1/(1+alpha^2) and accumulates
            # an alpha-scaled relaxation phase, so expose both phases.
            t_s = cond.get("t_end_s", cond.get("dt_s", torch.zeros_like(temp))).float()
            b_mag = b.norm(dim=1)
            cycles = self.physics_phase_gamma_hz_per_t * b_mag * t_s
            alpha = cond.get("alpha_t", cond.get("alpha", torch.zeros_like(cycles))).float()
            phase = (2.0 * torch.pi * cycles) / (1.0 + alpha.square())
            damping_phase = alpha * phase
            phase_features = torch.stack(
                [
                    phase.sin(),
                    phase.cos(),
                    damping_phase.sin(),
                    damping_phase.cos(),
                    torch.log1p(cycles.clamp_min(0.0)),
                    cycles / (1.0 + cycles.clamp_min(0.0)),
                ],
                dim=1,
            )
            feats.append(self.physics_phase_linear(phase_features))
        if self.drive_phase_linear is not None:
            # Convert the raw Slonczewski/SOT controls to the two integrated
            # Gilbert-basis torque phases actually accumulated over this
            # prediction horizon. This mirrors the V4 generator convention.
            t_s = cond.get("t_end_s", cond.get("dt_s", torch.zeros_like(temp))).float()
            alpha = cond.get("alpha_t", cond.get("alpha", torch.zeros_like(t_s))).float()
            ms = cond.get("ms_t_a_per_m", cond.get("msat_a_per_m", torch.ones_like(t_s))).float()
            thickness = cond.get("thickness_m", cond.get("dz_m", torch.ones_like(t_s))).float()
            pol = cond.get("pol_eff", torch.zeros_like(t_s)).float()
            eps_prime = cond.get("epsilon_prime", torch.zeros_like(t_s)).float()
            current = cond.get("current_a_m2", torch.zeros_like(t_s)).float()
            has_zl = cond.get("has_zhang_li", torch.zeros_like(t_s)).float().clamp(0.0, 1.0)
            has_sl = cond.get("has_slonczewski", torch.zeros_like(t_s)).float().clamp(0.0, 1.0)
            has_sot = cond.get("has_sot", torch.zeros_like(t_s)).float().clamp(0.0, 1.0)
            sl_gate = torch.maximum(has_sl, has_sot)
            hbar_over_e = 1.054_571_817e-34 / 1.602_176_634e-19
            beta_t = hbar_over_e * current / (
                thickness.clamp_min(1.0e-12) * ms.clamp_min(1.0)
            )
            b_dl = beta_t * (0.5 * pol)
            b_fl = beta_t * eps_prime
            gilbert = 1.0 / (1.0 + alpha.square())
            explicit_dl = gilbert * (b_dl + alpha * b_fl) * sl_gate
            explicit_fl = gilbert * (b_fl - alpha * b_dl) * sl_gate
            phase_scale = 2.0 * torch.pi * self.drive_phase_gamma_hz_per_t * t_s
            phase_dl = (phase_scale * explicit_dl).clamp(-1.0e6, 1.0e6)
            phase_fl = (phase_scale * explicit_fl).clamp(-1.0e6, 1.0e6)

            # For Zhang-Li torque the natural integrated control is an
            # advection distance measured in lattice cells, u*dt/dx.
            jx = cond.get("j_vector_x_a_per_m2", torch.zeros_like(t_s)).float()
            jy = cond.get("j_vector_y_a_per_m2", torch.zeros_like(t_s)).float()
            dx = cond.get("dx_m", torch.ones_like(t_s)).float().clamp_min(1.0e-12)
            dy = cond.get("dy_m", torch.ones_like(t_s)).float().clamp_min(1.0e-12)
            mu_b_over_e = 9.274_010_0783e-24 / 1.602_176_634e-19
            zl_x = (mu_b_over_e * pol * jx * t_s / (ms.clamp_min(1.0) * dx) * has_zl).clamp(-1.0e6, 1.0e6)
            zl_y = (mu_b_over_e * pol * jy * t_s / (ms.clamp_min(1.0) * dy) * has_zl).clamp(-1.0e6, 1.0e6)
            beta_zl = cond.get("beta_zl", torch.zeros_like(t_s)).float() * has_zl

            def integrated_features(value: torch.Tensor) -> list[torch.Tensor]:
                return [
                    value.sin(),
                    value.cos() - 1.0,
                    value.sign() * torch.log1p(value.abs()),
                    value / (1.0 + value.abs()),
                ]

            drive_features = torch.stack(
                [
                    *integrated_features(phase_dl),
                    *integrated_features(phase_fl),
                    zl_x.sign() * torch.log1p(zl_x.abs()),
                    zl_x / (1.0 + zl_x.abs()),
                    zl_y.sign() * torch.log1p(zl_y.abs()),
                    zl_y / (1.0 + zl_y.abs()),
                    has_zl,
                    has_sl,
                    has_sot,
                    beta_zl,
                ],
                dim=1,
            )
            feats.append(self.drive_phase_linear(drive_features))
        if self.j_grid_linear is not None:
            j_grid = self._control_grid(cond, idx.shape[0], device) / self.j_scalar_scale
            j_flat = j_grid.reshape(idx.shape[0], -1)
            j_emb = sinusoidal_embedding(j_flat.reshape(-1), 64).reshape(idx.shape[0], -1)
            feats.append(self.j_grid_linear(j_emb))
        if self.categorical_linear is not None:
            feats.append(self.categorical_linear(self._categorical_features(cond, idx.shape[0], device)))
        if self.pulse_condition:
            drive_time = cond.get("drive_time_s", torch.zeros(idx.shape[0], device=device)).float()
            relax_time = cond.get("relax_time_s", torch.zeros(idx.shape[0], device=device)).float()
            drive_fraction = cond.get("drive_fraction", torch.zeros(idx.shape[0], device=device)).float()
            if self.mask_target_time:
                drive_time = torch.zeros_like(drive_time)
                relax_time = torch.zeros_like(relax_time)
                drive_fraction = torch.zeros_like(drive_fraction)
            feats.extend(
                [
                    sinusoidal_embedding(torch.log(drive_time.clamp_min(1e-15)), 64),
                    sinusoidal_embedding(torch.log(relax_time.clamp_min(1e-15)), 64),
                    sinusoidal_embedding(drive_fraction.clamp(0.0, 1.0), 64),
                ]
            )
        if self.material_linear is not None:
            mat_components = []
            for key in self.enabled_material_keys:
                v = cond.get(key, torch.zeros(idx.shape[0], device=device)).float()
                if self.mask_target_time and key in {
                    "anchor_offset_norm",
                    "target_offset_norm",
                }:
                    v = torch.zeros_like(v)
                mat_components.append(sinusoidal_embedding(self._zscore(key, v), 64))
            feats.append(self.material_linear(torch.cat(mat_components, dim=-1)))
        raw = torch.cat(feats, dim=-1)
        return self.project(raw)


# Backwards-compatible alias used throughout the v1 code paths.
ConditionEmbedder = FullConditionEmbedder
