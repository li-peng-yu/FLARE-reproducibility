from __future__ import annotations

import torch
from torch import nn

from skyrmion_cfm.data.conditions import ConditionStats
from skyrmion_cfm.models.common import AdaLayerNorm, build_input, input_channels, make_condition_embedder


class MLPBlock(nn.Module):
    def __init__(self, hidden: int, cond_dim: int) -> None:
        super().__init__()
        self.adaln = AdaLayerNorm(hidden, cond_dim)
        self.ff = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.adaln(x, cond))


class MLPVelocity(nn.Module):
    def __init__(self, cfg: dict, stats: ConditionStats | None = None) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        mlp_cfg = model_cfg["mlp"]
        self.input_repr = model_cfg.get("input_repr", "cartesian")
        self.lattice_size = int(cfg["data"]["lattice_size"])
        self.spatial_size = int(mlp_cfg.get("spatial_size", self.lattice_size))
        in_dim = input_channels(self.input_repr) * self.spatial_size * self.spatial_size
        out_dim = 3 * self.spatial_size * self.spatial_size
        hidden = int(mlp_cfg.get("hidden", 2048))
        bottleneck = int(mlp_cfg.get("bottleneck", min(4096, hidden)))
        depth = int(mlp_cfg.get("depth", 8))
        cond_dim = int(model_cfg.get("cond_dim", 512))
        self.cond_embed = make_condition_embedder(cfg, stats)
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, hidden),
        )
        self.blocks = nn.ModuleList([MLPBlock(hidden, cond_dim) for _ in range(depth)])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, out_dim),
        )

    def forward(
        self,
        m0: torch.Tensor,
        omega_tau: torch.Tensor,
        tau: torch.Tensor,
        cond: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        bsz = m0.shape[0]
        x_img = build_input(m0, omega_tau, self.input_repr)
        if self.spatial_size != self.lattice_size:
            x_img = torch.nn.functional.adaptive_avg_pool2d(
                x_img, (self.spatial_size, self.spatial_size)
            )
        x = x_img.flatten(1)
        c = self.cond_embed(tau, cond)
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, c)
        out = self.out_proj(h).view(bsz, 3, self.spatial_size, self.spatial_size)
        if self.spatial_size != self.lattice_size:
            out = torch.nn.functional.interpolate(
                out,
                size=(self.lattice_size, self.lattice_size),
                mode="bilinear",
                align_corners=False,
            )
        return out
