"""Track C: E(2)-steerable U-Net (C4 lattice rotations + synchronous in-plane
spin xy rotations).

Built on :mod:`escnn`. The lattice 2D space is acted on by ``rot2dOnR2(N=4)``
(``p4``: discrete 90° rotations × translations), and within each lattice site
the magnetisation splits as

- ``(m_x, m_y)``: standard rotation rep ``irrep(1)`` (rotated with the lattice);
- ``m_z``: trivial rep (invariant under C4).

The condition vector is global and not equivariant; we inject it only on the
trivial subset of each layer's field type so the lattice symmetry of the
``irrep(1)`` channels is preserved. ``defect_field`` and ``j_field`` are
trivial scalars on the lattice.

If ``escnn`` is missing at instantiation time we raise a clear ImportError.
The rest of the package functions independently of this module.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from skyrmion_cfm.data.conditions import ConditionStats
from skyrmion_cfm.models.common import (
    build_spatial_condition_image,
    make_condition_embedder,
    normalize_boundary,
    spatial_condition_fields_from_cfg,
    spatial_condition_scales_from_cfg,
)


def _require_escnn():
    try:
        import escnn  # type: ignore
        from escnn import gspaces, nn as enn  # type: ignore

        return escnn, gspaces, enn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Track C (EquivariantUNetVelocity) requires `escnn`. "
            "Install with `pip install escnn`."
        ) from exc


class _CondAffineTrivial(nn.Module):
    """Apply scale/shift from the condition vector to the trivial channels only.

    Equivariance under C4 is preserved because only invariant channels are
    affected. ``trivial_slice`` indicates where the trivial chunk lives in the
    channel dim (escnn FieldType orders reprs in the order they are added).
    """

    def __init__(self, n_trivial: int, cond_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * int(n_trivial)))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, trivial_slice: slice) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        out = x.clone()
        out[:, trivial_slice] = x[:, trivial_slice] * (1 + scale) + shift
        return out


class EquivariantUNetVelocity(nn.Module):
    """C4-equivariant U-Net for forward / inverse velocity prediction.

    Each U-Net level holds two reprs:
      ``width`` copies of ``irrep(1)`` (each 2 channels for ``xy`` components)
      ``width`` copies of ``trivial`` (1 channel each for scalars)

    so the per-level channel count is ``3 × width``. Encoder uses
    ``R2Conv`` + ``GNormBatchNorm`` + ``NormNonLinearity``; downsampling is a stride-2
    ``PointwiseAvgPool``. Decoder mirrors and uses ``R2Upsampling``.
    """

    def __init__(self, cfg: dict, stats: ConditionStats | None = None) -> None:
        super().__init__()
        escnn, gspaces, enn = _require_escnn()
        self._enn = enn
        model_cfg = cfg["model"]
        es_cfg = model_cfg.get("esteer", {})
        self.input_repr = model_cfg.get("input_repr", "cartesian")
        self.boundary = normalize_boundary(cfg["data"].get("boundary", "open"))
        self.use_spatial_cond = bool(model_cfg.get("use_spatial_cond", True))
        self.spatial_cond_j_scale = float(model_cfg.get("spatial_cond_j_scale", 1.0))
        self.spatial_cond_fields = spatial_condition_fields_from_cfg(cfg)
        self.spatial_cond_scales = spatial_condition_scales_from_cfg(cfg)
        self.r2_act = gspaces.rot2dOnR2(N=4)
        cond_dim = int(model_cfg.get("cond_dim", 512))
        self.cond_embed = make_condition_embedder(cfg, stats)

        trivial = self.r2_act.trivial_repr
        irrep1 = self.r2_act.irrep(1)

        def _ft(n_vec: int, n_scalar: int):
            return enn.FieldType(self.r2_act, [irrep1] * n_vec + [trivial] * n_scalar)

        n_vec_in = 2  # m_init xy + state xy
        n_scalar_in = 2 + (len(self.spatial_cond_fields) if self.use_spatial_cond else 0)
        self.in_type = _ft(n_vec_in, n_scalar_in)
        self.out_type = _ft(1, 1)

        base = int(es_cfg.get("base_channels", 32))
        mults = list(es_cfg.get("channel_mults", [1, 2, 4, 8]))
        widths = [base * int(m) for m in mults]
        self.widths = widths

        self.in_conv = enn.R2Conv(self.in_type, _ft(widths[0], widths[0]), kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.encoder_cond = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()
        prev = widths[0]
        for level, width in enumerate(widths):
            ft_in = _ft(prev, prev)
            ft_out = _ft(width, width)
            self.encoders.append(
                enn.SequentialModule(
                    enn.R2Conv(ft_in, ft_out, kernel_size=3, padding=1),
                    enn.GNormBatchNorm(ft_out),
                    enn.NormNonLinearity(ft_out),
                    enn.R2Conv(ft_out, ft_out, kernel_size=3, padding=1),
                    enn.GNormBatchNorm(ft_out),
                    enn.NormNonLinearity(ft_out),
                )
            )
            self.encoder_cond.append(_CondAffineTrivial(width, cond_dim))
            if level < len(widths) - 1:
                self.encoder_pools.append(enn.PointwiseAvgPool(ft_out, kernel_size=2, stride=2))
            else:
                self.encoder_pools.append(nn.Identity())
            prev = width

        ft_bn = _ft(widths[-1], widths[-1])
        self.bottleneck = enn.SequentialModule(
            enn.R2Conv(ft_bn, ft_bn, kernel_size=3, padding=1),
            enn.GNormBatchNorm(ft_bn),
            enn.NormNonLinearity(ft_bn),
            enn.R2Conv(ft_bn, ft_bn, kernel_size=3, padding=1),
            enn.GNormBatchNorm(ft_bn),
            enn.NormNonLinearity(ft_bn),
        )
        self.bottleneck_cond = _CondAffineTrivial(widths[-1], cond_dim)

        self.decoders = nn.ModuleList()
        self.decoder_cond = nn.ModuleList()
        self.decoder_ups = nn.ModuleList()
        prev = widths[-1]
        rev_widths = list(reversed(widths))
        for di, target in enumerate(rev_widths):
            ft_in = _ft(prev + target, prev + target)
            ft_out = _ft(target, target)
            self.decoders.append(
                enn.SequentialModule(
                    enn.R2Conv(ft_in, ft_out, kernel_size=3, padding=1),
                    enn.GNormBatchNorm(ft_out),
                    enn.NormNonLinearity(ft_out),
                    enn.R2Conv(ft_out, ft_out, kernel_size=3, padding=1),
                    enn.GNormBatchNorm(ft_out),
                    enn.NormNonLinearity(ft_out),
                )
            )
            self.decoder_cond.append(_CondAffineTrivial(target, cond_dim))
            if di < len(rev_widths) - 1:
                self.decoder_ups.append(enn.R2Upsampling(ft_out, scale_factor=2, mode="nearest"))
            else:
                self.decoder_ups.append(nn.Identity())
            prev = target

        self.head = enn.R2Conv(_ft(widths[0], widths[0]), self.out_type, kernel_size=1)

    @staticmethod
    def _trivial_slice(width: int) -> slice:
        # Channels are laid out as [irrep1 × width = 2*width chans, trivial × width = width chans].
        return slice(2 * width, 3 * width)

    @staticmethod
    def _concat_by_rep(x: torch.Tensor, x_width: int, skip: torch.Tensor, skip_width: int) -> torch.Tensor:
        x_vec, x_scalar = x[:, : 2 * x_width], x[:, 2 * x_width : 3 * x_width]
        skip_vec = skip[:, : 2 * skip_width]
        skip_scalar = skip[:, 2 * skip_width : 3 * skip_width]
        return torch.cat([x_vec, skip_vec, x_scalar, skip_scalar], dim=1)

    def _spatial_cond(self, cond: dict[str, torch.Tensor], m0: torch.Tensor) -> torch.Tensor | None:
        if not self.use_spatial_cond:
            return None
        return build_spatial_condition_image(
            cond,
            m0,
            j_field_scale=self.spatial_cond_j_scale,
            fields=self.spatial_cond_fields,
            scales=self.spatial_cond_scales,
        )

    def _pack_input(self, m_init: torch.Tensor, state: torch.Tensor, spatial: torch.Tensor | None) -> torch.Tensor:
        # Order matches FieldType: [irrep1 × N_vec, trivial × N_scalar].
        irrep1_packed = torch.cat([m_init[:, :2], state[:, :2]], dim=1)
        trivial_parts = [m_init[:, 2:3], state[:, 2:3]]
        if spatial is not None:
            trivial_parts.append(spatial)
        return torch.cat([irrep1_packed, torch.cat(trivial_parts, dim=1)], dim=1)

    def _unpack_output(self, x: torch.Tensor) -> torch.Tensor:
        # Output layout: [irrep1 × 1 = 2 chans (vx, vy), trivial × 1 = 1 chan (vz)].
        return torch.cat([x[:, :2], x[:, 2:3]], dim=1)

    def forward(
        self,
        m_init: torch.Tensor,
        state: torch.Tensor,
        tau: torch.Tensor,
        cond: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        enn = self._enn
        cond_vec = self.cond_embed(tau, cond)
        spatial = self._spatial_cond(cond, m_init)
        packed = self._pack_input(m_init, state, spatial)
        x = enn.GeometricTensor(packed, self.in_type)
        x = self.in_conv(x)
        skips: list[Any] = []
        for level, (block, cond_layer, pool) in enumerate(
            zip(self.encoders, self.encoder_cond, self.encoder_pools)
        ):
            x = block(x)
            x = enn.GeometricTensor(
                cond_layer(x.tensor, cond_vec, self._trivial_slice(self.widths[level])),
                x.type,
            )
            skips.append(x)
            if not isinstance(pool, nn.Identity):
                x = pool(x)
        x = self.bottleneck(x)
        x = enn.GeometricTensor(
            self.bottleneck_cond(x.tensor, cond_vec, self._trivial_slice(self.widths[-1])),
            x.type,
        )
        rev_widths = list(reversed(self.widths))
        current = self.widths[-1]
        for di, (block, cond_layer, up) in enumerate(
            zip(self.decoders, self.decoder_cond, self.decoder_ups)
        ):
            target = rev_widths[di]
            skip = skips.pop()
            if x.tensor.shape[-2:] != skip.tensor.shape[-2:]:
                # Align spatial size — nearest upsampling via the equivariant
                # ``R2Upsampling`` would belong here, but the encoder-stage
                # output of the deepest level matches the bottleneck shape, so
                # we only need to upsample at the transition between levels.
                # All such transitions are handled by the per-decoder ``up``.
                pass
            x_cat = self._concat_by_rep(x.tensor, current, skip.tensor, target)
            ft_cat = enn.FieldType(
                self.r2_act,
                [self.r2_act.irrep(1)] * (current + target)
                + [self.r2_act.trivial_repr] * (current + target),
            )
            x = enn.GeometricTensor(x_cat, ft_cat)
            x = block(x)
            x = enn.GeometricTensor(
                cond_layer(x.tensor, cond_vec, self._trivial_slice(target)),
                x.type,
            )
            if not isinstance(up, nn.Identity):
                x = up(x)
            current = target
        out = self.head(x).tensor
        return self._unpack_output(out)
