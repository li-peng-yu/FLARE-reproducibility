from __future__ import annotations

import math

import torch
from torch import nn

from skyrmion_cfm.data.conditions import (
    ConditionEmbedder,
    ConditionStats,
    V4_CATEGORICAL_ORDERS,
    V4_EMBEDDED_SCALAR_KEYS,
    V4_T_THETA_KEYS,
)


DEFAULT_SPATIAL_COND_FIELDS: tuple[str, ...] = ("defect_field", "j_field")
V4_BOUNDARY_MODE_ORDER: tuple[str, ...] = ("open", "pbc_x", "pbc_y", "pbc_xy", "absorbing_edge")


def make_condition_embedder(cfg: dict, stats: ConditionStats | None = None) -> ConditionEmbedder:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    t_end_ns = data_cfg.get("t_end_ns")
    dt_scales = data_cfg.get("dt_scales")
    audit = cfg.get("condition_audit")
    j_scalar_cfg = model_cfg.get("j_scalar_embedding", False)
    if isinstance(j_scalar_cfg, dict):
        j_scalar_enabled = bool(j_scalar_cfg.get("enabled", False))
        j_scalar_grid_shape = tuple(j_scalar_cfg.get("grid_shape", data_cfg.get("control_grid", (8, 8))))
        j_scalar_scale = float(
            j_scalar_cfg.get(
                "scale",
                model_cfg.get("spatial_cond_j_scale", 1.0),
            )
        )
    else:
        j_scalar_enabled = bool(j_scalar_cfg)
        j_scalar_grid_shape = tuple(model_cfg.get("j_scalar_grid_shape", data_cfg.get("control_grid", (8, 8))))
        j_scalar_scale = float(model_cfg.get("j_scalar_scale", model_cfg.get("spatial_cond_j_scale", 1.0)))
    scalar_keys = model_cfg.get("scalar_condition_keys")
    if scalar_keys == "v4_full" or (
        scalar_keys is None and bool(model_cfg.get("v4_full_conditioning", False))
    ):
        scalar_keys = list(V4_EMBEDDED_SCALAR_KEYS + tuple(f"theta_T_{k}" for k in V4_T_THETA_KEYS))
    excluded_scalar_keys = {
        str(key) for key in model_cfg.get("exclude_scalar_condition_keys", ())
    }
    if scalar_keys is not None and excluded_scalar_keys:
        scalar_keys = [
            str(key) for key in scalar_keys if str(key) not in excluded_scalar_keys
        ]
    categorical_dims: dict[str, int] = {}
    categorical_cfg = model_cfg.get("categorical_condition_dims")
    if categorical_cfg == "v4_full" or bool(model_cfg.get("v4_full_conditioning", False)):
        categorical_dims.update({k: len(v) for k, v in V4_CATEGORICAL_ORDERS.items()})
    elif isinstance(categorical_cfg, dict):
        categorical_dims.update({str(k): int(v) for k, v in categorical_cfg.items()})
    phase_cfg = model_cfg.get("physics_phase_condition", False)
    if isinstance(phase_cfg, dict):
        phase_enabled = bool(phase_cfg.get("enabled", False))
        phase_gamma_hz_per_t = float(phase_cfg.get("gamma_hz_per_t", 28.0e9))
    else:
        phase_enabled = bool(phase_cfg)
        phase_gamma_hz_per_t = 28.0e9
    drive_phase_cfg = model_cfg.get("drive_phase_condition", False)
    if isinstance(drive_phase_cfg, dict):
        drive_phase_enabled = bool(drive_phase_cfg.get("enabled", False))
        drive_phase_gamma_hz_per_t = float(
            drive_phase_cfg.get("gamma_hz_per_t", 28.0e9)
        )
    else:
        drive_phase_enabled = bool(drive_phase_cfg)
        drive_phase_gamma_hz_per_t = 28.0e9
    kwargs = {
        "cond_dim": int(model_cfg.get("cond_dim", 512)),
        "stats": stats,
        "disable_dt_condition": bool(model_cfg.get("disable_dt_condition", False)),
        "mask_target_time": bool(model_cfg.get("mask_target_time", False)),
        "pulse_condition": bool(model_cfg.get("pulse_condition", {}).get("enabled", False)),
        "j_scalar_embedding": j_scalar_enabled,
        "j_scalar_grid_shape": j_scalar_grid_shape,
        "j_scalar_scale": j_scalar_scale,
        "scalar_condition_keys": scalar_keys,
        "categorical_condition_dims": categorical_dims,
        "force_scalar_condition_keys": model_cfg.get("force_scalar_condition_keys"),
        "scalar_condition_scales": model_cfg.get("scalar_condition_scales"),
        "physics_phase_condition": phase_enabled,
        "physics_phase_gamma_hz_per_t": phase_gamma_hz_per_t,
        "drive_phase_condition": drive_phase_enabled,
        "drive_phase_gamma_hz_per_t": drive_phase_gamma_hz_per_t,
    }
    if t_end_ns is not None:
        kwargs["t_end_buckets"] = list(t_end_ns)
    if dt_scales is not None:
        kwargs["dt_scales"] = list(dt_scales)
    if audit is not None:
        kwargs["audit"] = audit
    return ConditionEmbedder(**kwargs)


def spatial_condition_fields_from_cfg(cfg: dict) -> tuple[str, ...]:
    model_cfg = cfg.get("model", {})
    fields = model_cfg.get("spatial_cond_fields")
    if fields == "v4_full":
        return (
            "defect_field",
            "j_x_field",
            "j_y_field",
            "j_z_field",
            "b_x_field",
            "b_y_field",
            "b_z_field",
            "temperature_field",
            "msat_field",
            "aex_field",
            "ku1_field",
            "dind_field",
            "alpha_field",
        )
    if fields is None:
        return DEFAULT_SPATIAL_COND_FIELDS
    return tuple(str(x) for x in fields)


def spatial_condition_scales_from_cfg(cfg: dict) -> dict[str, float]:
    model_cfg = cfg.get("model", {})
    scales = {str(k): float(v) for k, v in model_cfg.get("spatial_cond_scales", {}).items()}
    j_scale = float(model_cfg.get("spatial_cond_j_scale", 1.0))
    if j_scale != 1.0:
        for key in ("j_field", "j_x_field", "j_y_field", "j_z_field"):
            scales.setdefault(key, j_scale)
    return scales


def state_channels_from_cfg(cfg: dict | None = None) -> int:
    if cfg is None:
        return 3
    bridge_cfg = cfg.get("bridge", {}) if isinstance(cfg, dict) else {}
    state_repr = str(bridge_cfg.get("state_repr", bridge_cfg.get("target_repr", ""))).lower()
    return 2 if state_repr in {"alpha2d", "alpha_2d", "rotation_vector_2d"} else 3


def input_image_channels(
    input_repr: str,
    include_spatial_cond: bool = False,
    state_channels: int = 3,
    spatial_channels: int | None = None,
) -> int:
    """Number of conv-input channels accepted by velocity networks.

    plan-1 used 6 (M0, Ω_τ) or 9 (spherical). plan-2 prepends a 2-channel
    spatial conditioning map (defect_field + j_field) for an 8/11-channel
    input. The flag is wired through ``cfg['model']['use_spatial_cond']`` so
    legacy configs continue to instantiate the 6/9-channel form.
    """
    base = input_channels(input_repr, state_channels=state_channels)
    if include_spatial_cond:
        return base + (2 if spatial_channels is None else int(spatial_channels))
    return base


def build_image_input(
    m_init: torch.Tensor,
    state: torch.Tensor,
    input_repr: str,
    spatial_cond: torch.Tensor | None = None,
) -> torch.Tensor:
    """Concatenate (M_init, ψ_τ / Ω_τ) with optional (defect, j_field).

    ``state`` is whatever the bridge feeds the network at time τ — ``ψ_τ`` for
    Cart, ``μ_τ`` for RFM, or ``Ω_τ`` for Mode α. The function selects between
    the cartesian and spherical input layouts already supported by plan-1.
    """
    base = build_input(m_init, state, input_repr)
    if spatial_cond is None:
        return base
    return torch.cat([base, spatial_cond], dim=1)


def build_spatial_condition_image(
    cond: dict[str, torch.Tensor],
    ref: torch.Tensor,
    *,
    j_field_scale: float = 1.0,
    fields: tuple[str, ...] | list[str] | None = None,
    scales: dict[str, float] | None = None,
) -> torch.Tensor:
    """Return configured spatial condition image channels aligned to ``ref``.

    ``j_field`` is stored in physical A/m^2 units. The velocity networks can
    optionally divide it before concatenating it with spin-valued channels.
    """
    if j_field_scale <= 0.0:
        raise ValueError("j_field_scale must be positive")
    fields = DEFAULT_SPATIAL_COND_FIELDS if fields is None else tuple(str(x) for x in fields)
    scales = dict(scales or {})
    if "j_field" not in scales and j_field_scale != 1.0:
        scales["j_field"] = float(j_field_scale)
    parts: list[torch.Tensor] = []
    for key in fields:
        x = cond.get(key)
        if x is None:
            x = torch.zeros(ref.shape[0], 1, *ref.shape[-2:], device=ref.device, dtype=ref.dtype)
        else:
            x = x.to(device=ref.device, dtype=ref.dtype)
            if x.ndim == 3:
                x = x.unsqueeze(1)
            if x.ndim != 4:
                raise ValueError(f"{key} must have shape (B, C, H, W) or (B, H, W), got {tuple(x.shape)}")
        scale = scales.get(key)
        if scale is not None:
            if scale <= 0.0:
                raise ValueError(f"spatial condition scale for {key} must be positive")
            x = x / float(scale)
        parts.append(x)
    return torch.cat(parts, dim=1)


def input_channels(input_repr: str, state_channels: int = 3) -> int:
    if input_repr == "cartesian":
        return 3 + int(state_channels)
    if input_repr == "spherical":
        return 6 + int(state_channels)
    raise ValueError(f"Unknown input_repr: {input_repr}")


def build_input(m0: torch.Tensor, omega: torch.Tensor, input_repr: str) -> torch.Tensor:
    if input_repr == "cartesian":
        return torch.cat([m0, omega], dim=1)
    if input_repr == "spherical":
        mx, my, mz = m0[:, 0], m0[:, 1], m0[:, 2].clamp(-1.0, 1.0)
        theta = torch.acos(mz)
        phi = torch.atan2(my, mx)
        sph = torch.stack(
            [theta, phi, phi.sin(), phi.cos(), theta.sin(), theta.cos()],
            dim=1,
        )
        return torch.cat([sph, omega], dim=1)
    raise ValueError(f"Unknown input_repr: {input_repr}")


class AdaLayerNorm(nn.Module):
    def __init__(self, hidden: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * hidden))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.mod(cond).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class AdaGroupNorm(nn.Module):
    def __init__(self, channels: int, cond_dim: int, groups: int = 8) -> None:
        super().__init__()
        groups = min(groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * channels))

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        extra_modulation: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        scale, shift = self.mod(cond).chunk(2, dim=1)
        if extra_modulation is not None:
            extra_scale, extra_shift = extra_modulation
            scale = scale + extra_scale
            shift = shift + extra_shift
        return self.norm(x) * (1 + scale[..., None, None]) + shift[..., None, None]


def normalize_boundary(mode: str | None) -> str:
    """Canonicalize a boundary spec.

    ``open`` (alias ``free`` / ``neumann``) follows mumax3's free-boundary
    behaviour, which is Neumann (∂m/∂n = 0) for the exchange Laplacian and the
    DMI gradient. We implement Neumann by replicate padding, so a uniform field
    has zero Laplacian everywhere including the edges. ``zeros`` (alias
    ``dirichlet``) is the older "magnetization vanishes outside the domain"
    convention and is kept as an explicit, opt-in option.
    """
    if mode is None:
        return "open"
    m = str(mode).lower()
    if m in ("periodic", "circular", "pbc", "pbc_xy"):
        return "periodic"
    if m == "pbc_x":
        return "pbc_x"
    if m == "pbc_y":
        return "pbc_y"
    if m in ("open", "free", "neumann"):
        return "open"
    if m == "absorbing_edge":
        return "open"
    if m in ("zeros", "zero", "constant", "dirichlet"):
        return "zeros"
    if m == "replicate":
        return "open"
    if m == "reflect":
        return "reflect"
    raise ValueError(f"Unknown boundary mode: {mode}")


_BOUNDARY_TO_PAD_MODE = {
    "periodic": "circular",
    # Neumann ≡ ghost cell equals the boundary cell → replicate padding.
    "open": "replicate",
    # Dirichlet zero outside the domain (legacy; explicit opt-in).
    "zeros": "constant",
    "reflect": "reflect",
}


def pad_2d(x: torch.Tensor, pad: tuple[int, int, int, int], boundary: str) -> torch.Tensor:
    """Pad a 4D tensor according to ``boundary`` (left, right, top, bottom)."""
    boundary = normalize_boundary(boundary)
    if boundary == "pbc_x":
        return _pad_2d_axiswise(x, pad, x_mode="circular", y_mode="replicate")
    if boundary == "pbc_y":
        return _pad_2d_axiswise(x, pad, x_mode="replicate", y_mode="circular")
    mode = _BOUNDARY_TO_PAD_MODE[boundary]
    if mode == "constant":
        return torch.nn.functional.pad(x, pad, mode="constant", value=0.0)
    return torch.nn.functional.pad(x, pad, mode=mode)


def _pad_axis(x: torch.Tensor, pad: tuple[int, int, int, int], mode: str) -> torch.Tensor:
    if not any(pad):
        return x
    if mode == "constant":
        return torch.nn.functional.pad(x, pad, mode="constant", value=0.0)
    return torch.nn.functional.pad(x, pad, mode=mode)


def _pad_2d_axiswise(
    x: torch.Tensor,
    pad: tuple[int, int, int, int],
    *,
    x_mode: str,
    y_mode: str,
) -> torch.Tensor:
    left, right, top, bottom = pad
    x = _pad_axis(x, (left, right, 0, 0), x_mode)
    return _pad_axis(x, (0, 0, top, bottom), y_mode)


def boundary_mode_indices_from_cond(
    cond: dict[str, torch.Tensor],
    batch: int,
    device: torch.device,
) -> torch.Tensor | None:
    onehot = cond.get("boundary_mode_onehot")
    if onehot is None:
        return None
    onehot = onehot.to(device=device)
    if onehot.ndim == 1:
        if onehot.numel() == len(V4_BOUNDARY_MODE_ORDER) and batch == 1 and onehot.dtype.is_floating_point:
            onehot = onehot.unsqueeze(0)
        else:
            return onehot.long().reshape(batch)
    if onehot.ndim != 2 or onehot.shape[0] != batch:
        raise ValueError(f"boundary_mode_onehot must have shape (B, {len(V4_BOUNDARY_MODE_ORDER)}), got {tuple(onehot.shape)}")
    return onehot.float().argmax(dim=-1).long()


def boundary_periodicity_from_modes(
    boundary_modes: torch.Tensor,
    batch: int,
    device: torch.device,
) -> torch.Tensor:
    """Return per-sample ``(periodic_x, periodic_y)`` flags.

    ``boundary_modes`` normally contains indices into
    :data:`V4_BOUNDARY_MODE_ORDER`.  A boolean ``(B, 2)`` tensor is accepted
    as an already-normalized fast path so a model can decode the condition
    once and reuse the flags in every convolution.
    """
    modes = boundary_modes.to(device=device)
    if modes.dtype == torch.bool and modes.ndim == 2 and modes.shape == (batch, 2):
        return modes
    modes = modes.long().reshape(-1)
    if modes.numel() != batch:
        raise ValueError(f"boundary_modes batch {modes.numel()} does not match input batch {batch}")
    periodic_x = (modes == 1) | (modes == 3)
    periodic_y = (modes == 2) | (modes == 3)
    return torch.stack((periodic_x, periodic_y), dim=-1)


def boundary_periodicity_from_cond(
    cond: dict[str, torch.Tensor],
    batch: int,
    device: torch.device,
) -> torch.Tensor | None:
    modes = boundary_mode_indices_from_cond(cond, batch, device)
    if modes is None:
        return None
    return boundary_periodicity_from_modes(modes, batch, device)


def _mixed_axis_pad(
    x: torch.Tensor,
    before: int,
    after: int,
    *,
    dim: int,
    periodic: torch.Tensor,
) -> torch.Tensor:
    """Pad one spatial axis with per-sample circular/replicate semantics."""
    if before < 0 or after < 0:
        raise ValueError("padding values must be non-negative")
    if before == 0 and after == 0:
        return x
    size = int(x.shape[dim])
    if size <= 0:
        raise ValueError("cannot pad an empty spatial dimension")
    mask_shape = [int(x.shape[0])] + [1] * (x.ndim - 1)
    mask = periodic.reshape(mask_shape)
    pieces: list[torch.Tensor] = []
    if before:
        circular_index = torch.arange(-before, 0, device=x.device).remainder(size)
        replicate_index = torch.zeros(before, device=x.device, dtype=torch.long)
        circular = torch.index_select(x, dim, circular_index)
        replicate = torch.index_select(x, dim, replicate_index)
        pieces.append(torch.where(mask, circular, replicate))
    pieces.append(x)
    if after:
        circular_index = torch.arange(after, device=x.device).remainder(size)
        replicate_index = torch.full(
            (after,),
            size - 1,
            device=x.device,
            dtype=torch.long,
        )
        circular = torch.index_select(x, dim, circular_index)
        replicate = torch.index_select(x, dim, replicate_index)
        pieces.append(torch.where(mask, circular, replicate))
    return torch.cat(pieces, dim=dim)


def pad_2d_per_sample(
    x: torch.Tensor,
    pad: tuple[int, int, int, int],
    boundary_periodicity: torch.Tensor,
) -> torch.Tensor:
    """Pad a batch with mixed open/periodic axes without host synchronization.

    Open and absorbing-edge samples use replicate padding, matching
    :func:`pad_2d`; PBC samples wrap independently along x and y.  Only the
    halo tensors are selected per sample, after which the convolution can run
    once over the full batch.
    """
    if x.ndim != 4:
        raise ValueError(f"mixed boundary padding expects BCHW input, got {tuple(x.shape)}")
    flags = boundary_periodicity_from_modes(
        boundary_periodicity,
        int(x.shape[0]),
        x.device,
    )
    left, right, top, bottom = pad
    x = _mixed_axis_pad(
        x,
        left,
        right,
        dim=3,
        periodic=flags[:, 0],
    )
    return _mixed_axis_pad(
        x,
        top,
        bottom,
        dim=2,
        periodic=flags[:, 1],
    )


class BoundaryConv2d(nn.Conv2d):
    """Conv2d whose spatial padding follows a configurable boundary mode.

    The padding is applied manually with :func:`pad_2d` so the same code path
    works for ``circular`` (periodic), ``replicate`` (open / Neumann-like),
    ``constant`` (explicit Dirichlet zero), and ``reflect``.
    """

    def __init__(self, *args, boundary: str = "open", **kwargs) -> None:
        padding = kwargs.pop("padding", 0)
        super().__init__(*args, padding=0, **kwargs)
        if isinstance(padding, tuple):
            self.boundary_padding = padding
        else:
            self.boundary_padding = (padding, padding)
        self.boundary = normalize_boundary(boundary)

    def _forward_with_boundary(self, x: torch.Tensor, boundary: str) -> torch.Tensor:
        py, px = self.boundary_padding
        if px or py:
            x = pad_2d(x, (px, px, py, py), boundary)
        return super().forward(x)

    def forward(self, x: torch.Tensor, boundary_modes: torch.Tensor | None = None) -> torch.Tensor:
        if boundary_modes is None or not any(self.boundary_padding):
            return self._forward_with_boundary(x, self.boundary)
        py, px = self.boundary_padding
        x = pad_2d_per_sample(x, (px, px, py, py), boundary_modes)
        return super().forward(x)


# Backwards-compat alias; behaves identically to the original CircularConv2d.
class CircularConv2d(BoundaryConv2d):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("boundary", "periodic")
        super().__init__(*args, **kwargs)


def periodic_2d_embedding(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError("2D periodic embedding dim must be divisible by 4")
    quarter = dim // 4
    y = torch.arange(h, device=device, dtype=torch.float32) / h
    x = torch.arange(w, device=device, dtype=torch.float32) / w
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    freqs = torch.arange(1, quarter + 1, device=device, dtype=torch.float32)
    xargs = 2 * math.pi * xx[..., None] * freqs
    yargs = 2 * math.pi * yy[..., None] * freqs
    emb = torch.cat([xargs.sin(), xargs.cos(), yargs.sin(), yargs.cos()], dim=-1)
    return emb.reshape(1, h * w, dim)


def absolute_2d_embedding(
    h: int,
    w: int,
    dim: int,
    device: torch.device,
    base: float = 10000.0,
) -> torch.Tensor:
    """Non-periodic 2D sinusoidal positional embedding (standard transformer style).

    Half the dimensions encode the y-coordinate, half encode x; within each half
    we use the usual log-spaced frequencies, so wavelengths span a wide range
    and the encoding does not wrap around the grid like the periodic version.
    """
    if dim % 4 != 0:
        raise ValueError("absolute 2d embedding dim must be divisible by 4")
    quarter = dim // 4
    y = torch.arange(h, device=device, dtype=torch.float32)
    x = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    inv_freq = torch.exp(
        -math.log(base) * torch.arange(quarter, device=device, dtype=torch.float32) / max(quarter, 1)
    )
    xargs = xx[..., None] * inv_freq
    yargs = yy[..., None] * inv_freq
    emb = torch.cat([xargs.sin(), xargs.cos(), yargs.sin(), yargs.cos()], dim=-1)
    return emb.reshape(1, h * w, dim)


def central_diff(m: torch.Tensor, dim: int, boundary: str) -> torch.Tensor:
    """Central finite difference along ``dim`` respecting ``boundary``.

    ``periodic`` uses ``torch.roll``; non-periodic modes pad with the same rule
    used by the boundary (``open`` → replicate / Neumann, ``zeros`` → constant
    zero, ``reflect`` → mirrored edge) and then take a centered difference.
    """
    mode = normalize_boundary(boundary)
    if mode == "periodic":
        return 0.5 * (torch.roll(m, shifts=-1, dims=dim) - torch.roll(m, shifts=1, dims=dim))
    if dim == -1 or dim == m.ndim - 1:
        pad = (1, 1, 0, 0)
    elif dim == -2 or dim == m.ndim - 2:
        pad = (0, 0, 1, 1)
    else:
        raise ValueError(f"central_diff only supports the last two dims, got {dim}")
    padded = pad_2d(m, pad, mode)
    if dim == -1 or dim == m.ndim - 1:
        return 0.5 * (padded[..., 2:] - padded[..., :-2])
    return 0.5 * (padded[..., 2:, :] - padded[..., :-2, :])


def shift_neighbor(m: torch.Tensor, dim: int, boundary: str) -> torch.Tensor:
    """Return ``m`` shifted by one step toward the start of ``dim`` (i.e. ``m_{i+1}``).

    Used for nearest-neighbor terms like the Heisenberg sum. ``periodic`` wraps;
    non-periodic pads with the configured rule and callers can mask invalid
    hard-boundary bonds when they need missing-neighbor energy semantics.
    """
    mode = normalize_boundary(boundary)
    if mode == "periodic":
        return torch.roll(m, shifts=-1, dims=dim)
    if dim == -1 or dim == m.ndim - 1:
        pad = (0, 1, 0, 0)
    elif dim == -2 or dim == m.ndim - 2:
        pad = (0, 0, 0, 1)
    else:
        raise ValueError(f"shift_neighbor only supports the last two dims, got {dim}")
    padded = pad_2d(m, pad, mode)
    if dim == -1 or dim == m.ndim - 1:
        return padded[..., 1:]
    return padded[..., 1:, :]
