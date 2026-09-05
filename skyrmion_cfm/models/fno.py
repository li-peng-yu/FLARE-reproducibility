from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from skyrmion_cfm.data.conditions import ConditionStats
from skyrmion_cfm.models.common import (
    build_image_input,
    build_spatial_condition_image,
    input_image_channels,
    make_condition_embedder,
    spatial_condition_fields_from_cfg,
    spatial_condition_scales_from_cfg,
    state_channels_from_cfg,
)


class SpectralConv2d(nn.Module):
    """Low-frequency 2-D Fourier convolution with independent +/- y modes."""

    def __init__(self, in_channels: int, out_channels: int, modes_y: int, modes_x: int) -> None:
        super().__init__()
        if min(in_channels, out_channels, modes_y, modes_x) <= 0:
            raise ValueError("spectral convolution dimensions must be positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_y = int(modes_y)
        self.modes_x = int(modes_x)
        scale = (self.in_channels * self.out_channels) ** -0.5
        shape = (self.in_channels, self.out_channels, self.modes_y, self.modes_x, 2)
        self.weight_positive = nn.Parameter(scale * torch.randn(*shape))
        self.weight_negative = nn.Parameter(scale * torch.randn(*shape))

    @staticmethod
    def _multiply(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        # CUDA FFT support for bf16 varies across PyTorch/cuFFT versions.  The
        # formal baseline therefore performs only the spectral operation in
        # fp32 while the surrounding network remains AMP-compatible.
        x_ft = torch.fft.rfft2(x.float(), norm="ortho")
        height, width_half = x_ft.shape[-2:]
        modes_y = min(self.modes_y, height // 2)
        modes_x = min(self.modes_x, width_half)
        out_ft = torch.zeros(
            x.shape[0],
            self.out_channels,
            height,
            width_half,
            device=x.device,
            dtype=x_ft.dtype,
        )
        positive = torch.view_as_complex(
            self.weight_positive[:, :, :modes_y, :modes_x].float().contiguous()
        )
        negative = torch.view_as_complex(
            self.weight_negative[:, :, :modes_y, :modes_x].float().contiguous()
        )
        out_ft[:, :, :modes_y, :modes_x] = self._multiply(
            x_ft[:, :, :modes_y, :modes_x], positive
        )
        out_ft[:, :, -modes_y:, :modes_x] = self._multiply(
            x_ft[:, :, -modes_y:, :modes_x], negative
        )
        out = torch.fft.irfft2(out_ft, s=x.shape[-2:], norm="ortho")
        return out.to(dtype=input_dtype)


class FNOBlock2d(nn.Module):
    def __init__(self, width: int, modes_y: int, modes_x: int, cond_dim: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_y, modes_x)
        self.local = nn.Conv2d(width, width, kernel_size=1)
        groups = min(8, width)
        while width % groups:
            groups -= 1
        self.norm = nn.GroupNorm(groups, width)
        self.film = nn.Linear(cond_dim, 2 * width)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        residual = self.spectral(x) + self.local(x)
        residual = self.norm(residual)
        scale, shift = self.film(condition).chunk(2, dim=-1)
        residual = residual * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        return F.gelu(residual)


class FNOVelocity(nn.Module):
    """Conditioned Fourier neural operator compatible with the CFM sampler.

    The same bridge and training loop can use this class for either stochastic
    flow matching or the deterministic identity-source endpoint baseline.  It
    is intentionally condition-aware so the comparison tests the spatial
    operator rather than withholding physical metadata from the baseline.
    """

    def __init__(self, cfg: dict, stats: ConditionStats | None = None) -> None:
        super().__init__()
        model_cfg = cfg.get("model", {})
        fno_cfg = model_cfg.get("fno", {}) or {}
        self.input_repr = str(model_cfg.get("input_repr", "cartesian"))
        self.use_spatial_cond = bool(model_cfg.get("use_spatial_cond", False))
        self.spatial_cond_fields = spatial_condition_fields_from_cfg(cfg)
        self.spatial_cond_scales = spatial_condition_scales_from_cfg(cfg)
        self.spatial_cond_j_scale = float(model_cfg.get("spatial_cond_j_scale", 1.0))
        self.state_channels = state_channels_from_cfg(cfg)
        self.padding = int(fno_cfg.get("padding", 8))
        width = int(fno_cfg.get("width", 96))
        depth = int(fno_cfg.get("depth", 4))
        modes_y = int(fno_cfg.get("modes_y", fno_cfg.get("modes", 20)))
        modes_x = int(fno_cfg.get("modes_x", fno_cfg.get("modes", 20)))
        cond_dim = int(model_cfg.get("cond_dim", 512))
        if min(width, depth, modes_y, modes_x) <= 0 or self.padding < 0:
            raise ValueError("invalid model.fno dimensions")

        channels = input_image_channels(
            self.input_repr,
            self.use_spatial_cond,
            state_channels=self.state_channels,
            spatial_channels=len(self.spatial_cond_fields),
        )
        # Normalized coordinates let the operator represent open-boundary
        # effects instead of imposing translation invariance at the edges.
        self.input_projection = nn.Conv2d(channels + 2, width, kernel_size=1)
        self.condition = make_condition_embedder(cfg, stats)
        self.blocks = nn.ModuleList(
            FNOBlock2d(width, modes_y, modes_x, cond_dim) for _ in range(depth)
        )
        self.output_projection = nn.Sequential(
            nn.Conv2d(width, 2 * width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * width, self.state_channels, kernel_size=1),
        )

    def _spatial(self, cond: dict[str, torch.Tensor], ref: torch.Tensor) -> torch.Tensor | None:
        if not self.use_spatial_cond:
            return None
        return build_spatial_condition_image(
            cond,
            ref,
            j_field_scale=self.spatial_cond_j_scale,
            fields=self.spatial_cond_fields,
            scales=self.spatial_cond_scales,
        )

    @staticmethod
    def _coordinates(ref: torch.Tensor) -> torch.Tensor:
        height, width = ref.shape[-2:]
        y = torch.linspace(-1.0, 1.0, height, device=ref.device, dtype=ref.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=ref.device, dtype=ref.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=0).unsqueeze(0).expand(ref.shape[0], -1, -1, -1)

    def forward(
        self,
        m_init: torch.Tensor,
        state: torch.Tensor,
        tau: torch.Tensor,
        cond: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        image = build_image_input(
            m_init,
            state,
            self.input_repr,
            spatial_cond=self._spatial(cond, m_init),
        )
        image = torch.cat((image, self._coordinates(image)), dim=1)
        x = self.input_projection(image)
        if self.padding:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        condition = self.condition(tau, cond).to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, condition)
        if self.padding:
            x = x[..., self.padding : -self.padding, self.padding : -self.padding]
        return self.output_projection(x)

