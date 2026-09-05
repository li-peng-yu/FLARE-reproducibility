from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from skyrmion_cfm.data.conditions import ConditionStats
from skyrmion_cfm.models.common import (
    absolute_2d_embedding,
    build_image_input,
    build_input,
    build_spatial_condition_image,
    input_channels,
    input_image_channels,
    make_condition_embedder,
    normalize_boundary,
    spatial_condition_fields_from_cfg,
    spatial_condition_scales_from_cfg,
    state_channels_from_cfg,
)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden: int,
        heads: int,
        cond_dim: int,
        mlp_ratio: float,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        if use_rope and (hidden // heads) % 4 != 0:
            raise ValueError("2D RoPE requires per-head dimension divisible by 4")
        self.use_rope = bool(use_rope)
        self.heads = int(heads)
        self.head_dim = hidden // heads
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.attn_out = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, int(hidden * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden * mlp_ratio), hidden),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * hidden))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    @staticmethod
    def _apply_2d_rope(x: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        bsz, heads, tokens, head_dim = x.shape
        gh, gw = grid
        if tokens != gh * gw:
            raise ValueError("Token count does not match DiT RoPE grid")
        quarter = head_dim // 4
        device = x.device
        dtype = x.dtype
        y = torch.arange(gh, device=device, dtype=torch.float32)
        xcoord = torch.arange(gw, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, xcoord, indexing="ij")
        freqs = torch.arange(1, quarter + 1, device=device, dtype=torch.float32)
        phase_x = (2.0 * torch.pi * xx.reshape(-1, 1) * freqs.reshape(1, -1) / gw).to(dtype)
        phase_y = (2.0 * torch.pi * yy.reshape(-1, 1) * freqs.reshape(1, -1) / gh).to(dtype)

        def rotate_pairs(v: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
            v = v.reshape(bsz, heads, tokens, quarter, 2)
            cos = phase.cos()[None, None, :, :, None]
            sin = phase.sin()[None, None, :, :, None]
            first, second = v[..., 0:1], v[..., 1:2]
            return torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1).flatten(-2)

        x_x = rotate_pairs(x[..., : 2 * quarter], phase_x)
        x_y = rotate_pairs(x[..., 2 * quarter : 4 * quarter], phase_y)
        return torch.cat([x_x, x_y, x[..., 4 * quarter :]], dim=-1)

    def _attention(self, h: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        bsz, tokens, hidden = h.shape
        qkv = self.qkv(h).view(bsz, tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.use_rope:
            q = self._apply_2d_rope(q, grid)
            k = self._apply_2d_rope(k, grid)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(bsz, tokens, hidden)
        return self.attn_out(out)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        s1, b1, g1, s2, b2, g2 = self.ada(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + s1[:, None]) + b1[:, None]
        x = x + g1[:, None] * self._attention(h, grid)
        h = self.norm2(x) * (1 + s2[:, None]) + b2[:, None]
        x = x + g2[:, None] * self.mlp(h)
        return x


class DiTVelocity(nn.Module):
    def __init__(self, cfg: dict, stats: ConditionStats | None = None) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        dit_cfg = model_cfg["dit"]
        self.input_repr = model_cfg.get("input_repr", "cartesian")
        self.state_channels = state_channels_from_cfg(cfg)
        self.lattice_size = int(cfg["data"]["lattice_size"])
        self.boundary = normalize_boundary(cfg["data"].get("boundary", "open"))
        self.patch_size = int(dit_cfg.get("patch_size", 16))
        if self.lattice_size % self.patch_size != 0:
            raise ValueError("lattice_size must be divisible by patch_size")
        hidden = int(dit_cfg.get("hidden", 768))
        heads = int(dit_cfg.get("heads", 12))
        depth = int(dit_cfg.get("depth", 12))
        mlp_ratio = float(dit_cfg.get("mlp_ratio", 4.0))
        cond_dim = int(model_cfg.get("cond_dim", 512))
        self.use_spatial_cond = bool(model_cfg.get("use_spatial_cond", False))
        self.spatial_cond_j_scale = float(model_cfg.get("spatial_cond_j_scale", 1.0))
        self.spatial_cond_fields = spatial_condition_fields_from_cfg(cfg)
        self.spatial_cond_scales = spatial_condition_scales_from_cfg(cfg)
        # 2D RoPE assumes a torus, so we only use it when the lattice is
        # periodic; otherwise inject an absolute (non-wrapping) sinusoidal
        # positional embedding once after patchify.
        self.use_rope = self.boundary == "periodic"
        self.cond_embed = make_condition_embedder(cfg, stats)
        self.patch = nn.Conv2d(
            input_image_channels(
                self.input_repr,
                self.use_spatial_cond,
                state_channels=self.state_channels,
                spatial_channels=len(self.spatial_cond_fields),
            ),
            hidden,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden, heads, cond_dim, mlp_ratio, use_rope=self.use_rope) for _ in range(depth)]
        )
        if not self.use_rope:
            grid = self.lattice_size // self.patch_size
            pos = absolute_2d_embedding(grid, grid, hidden, device=torch.device("cpu"))
            self.register_buffer("pos_embed", pos, persistent=False)
        else:
            self.pos_embed = None
        self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * hidden))
        self.out = nn.Linear(hidden, self.patch_size * self.patch_size * self.state_channels)
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz = tokens.shape[0]
        p = self.patch_size
        gh = gw = self.lattice_size // p
        x = self.out(tokens).view(bsz, gh, gw, self.state_channels, p, p)
        return x.permute(0, 3, 1, 4, 2, 5).reshape(
            bsz,
            self.state_channels,
            self.lattice_size,
            self.lattice_size,
        )

    def _spatial_cond_image(self, cond: dict[str, torch.Tensor], m0: torch.Tensor) -> torch.Tensor | None:
        if not self.use_spatial_cond:
            return None
        return build_spatial_condition_image(
            cond,
            m0,
            j_field_scale=self.spatial_cond_j_scale,
            fields=self.spatial_cond_fields,
            scales=self.spatial_cond_scales,
        )

    def forward(
        self,
        m0: torch.Tensor,
        omega_tau: torch.Tensor,
        tau: torch.Tensor,
        cond: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        c = self.cond_embed(tau, cond)
        spatial = self._spatial_cond_image(cond, m0)
        x = build_image_input(m0, omega_tau, self.input_repr, spatial)
        h = self.patch(x)
        grid_h, grid_w = h.shape[-2:]
        h = h.flatten(2).transpose(1, 2)
        if self.pos_embed is not None:
            h = h + self.pos_embed.to(dtype=h.dtype, device=h.device)
        for block in self.blocks:
            h = block(h, c, (grid_h, grid_w))
        scale, shift = self.final_ada(c).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + scale[:, None]) + shift[:, None]
        return self._unpatchify(h)
