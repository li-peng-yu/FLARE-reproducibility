from __future__ import annotations

import warnings

import torch
from torch import nn
import torch.nn.functional as F

from skyrmion_cfm.data.conditions import ConditionStats, sinusoidal_embedding
from skyrmion_cfm.models.common import (
    AdaGroupNorm,
    BoundaryConv2d,
    absolute_2d_embedding,
    boundary_periodicity_from_cond,
    build_image_input,
    build_spatial_condition_image,
    input_channels,
    input_image_channels,
    make_condition_embedder,
    normalize_boundary,
    periodic_2d_embedding,
    spatial_condition_fields_from_cfg,
    spatial_condition_scales_from_cfg,
    state_channels_from_cfg,
)


def _valid_groups(channels: int, groups: int) -> int:
    groups = min(groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class EndpointTimeConditioner(nn.Module):
    """Independent embedding path for the physical prediction horizon.

    This intentionally does not share parameters with ``FullConditionEmbedder``:
    endpoint time gets its own bucket lookup and log-time Fourier features before
    being injected through dedicated per-block AdaGN modulation layers.
    """

    def __init__(
        self,
        n_buckets: int,
        output_dim: int,
        *,
        bucket_dim: int = 128,
        fourier_dim: int = 128,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        if n_buckets <= 0:
            raise ValueError("EndpointTimeConditioner requires at least one time bucket")
        if min(bucket_dim, fourier_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("EndpointTimeConditioner dimensions must be positive")
        self.bucket_lookup = nn.Embedding(int(n_buckets), int(bucket_dim))
        self.fourier_dim = int(fourier_dim)
        self.mlp = nn.Sequential(
            nn.Linear(int(bucket_dim) + self.fourier_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, cond: dict[str, torch.Tensor]) -> torch.Tensor:
        if "t_end_index" in cond:
            index = cond["t_end_index"].long()
            time_s = cond.get("t_end_s", cond.get("dt_s"))
        elif "dt_index" in cond:
            index = cond["dt_index"].long()
            time_s = cond.get("dt_s")
        else:
            raise KeyError("Time conditioning requires t_end_index or dt_index")
        if time_s is None:
            raise KeyError("Time conditioning requires t_end_s or dt_s")
        index = index.clamp(0, self.bucket_lookup.num_embeddings - 1)
        log_time = torch.log(time_s.float().clamp_min(1.0e-15))
        features = torch.cat(
            [self.bucket_lookup(index), sinusoidal_embedding(log_time, self.fourier_dim)],
            dim=-1,
        )
        return self.mlp(features)


def _zero_initialized_time_modulation(input_dim: int, channels: int) -> nn.Sequential:
    modulation = nn.Sequential(nn.SiLU(), nn.Linear(input_dim, 2 * channels))
    nn.init.zeros_(modulation[-1].weight)
    nn.init.zeros_(modulation[-1].bias)
    return modulation


class ResBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        groups: int,
        boundary: str = "open",
        *,
        time_cond_dim: int | None = None,
        time_condition_scale: float = 1.0,
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = AdaGroupNorm(in_ch, cond_dim, groups)
        self.conv1 = BoundaryConv2d(in_ch, out_ch, kernel_size=3, padding=1, boundary=boundary)
        self.norm2 = AdaGroupNorm(out_ch, cond_dim, groups)
        self.conv2 = BoundaryConv2d(out_ch, out_ch, kernel_size=3, padding=1, boundary=boundary)
        self.time_condition_scale = float(time_condition_scale)
        self.time_mod1 = (
            _zero_initialized_time_modulation(int(time_cond_dim), in_ch)
            if time_cond_dim is not None
            else None
        )
        self.time_mod2 = (
            _zero_initialized_time_modulation(int(time_cond_dim), out_ch)
            if time_cond_dim is not None
            else None
        )
        if zero_init_residual:
            nn.init.zeros_(self.conv2.weight)
            if self.conv2.bias is not None:
                nn.init.zeros_(self.conv2.bias)
        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else BoundaryConv2d(in_ch, out_ch, 1, boundary=boundary)
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        boundary_modes: torch.Tensor | None = None,
        time_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        time_mod1 = self._time_modulation(self.time_mod1, time_cond)
        time_mod2 = self._time_modulation(self.time_mod2, time_cond)
        h = self.conv1(F.silu(self.norm1(x, cond, time_mod1)), boundary_modes)
        h = self.conv2(F.silu(self.norm2(h, cond, time_mod2)), boundary_modes)
        skip = self.skip(x, boundary_modes) if isinstance(self.skip, BoundaryConv2d) else self.skip(x)
        return h + skip

    def _time_modulation(
        self,
        module: nn.Sequential | None,
        time_cond: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if module is None:
            return None
        if time_cond is None:
            raise ValueError("time_cond is required when independent time conditioning is enabled")
        scale, shift = module(time_cond).chunk(2, dim=1)
        return self.time_condition_scale * scale, self.time_condition_scale * shift


class BottleneckAttention(nn.Module):
    """Single self-attention block at the U-Net bottleneck (plan-1 default)."""

    def __init__(
        self,
        channels: int,
        heads: int = 8,
        *,
        active_channels: int | None = None,
    ) -> None:
        super().__init__()
        active_channels = int(channels if active_channels is None else active_channels)
        if active_channels <= 0 or active_channels > channels:
            raise ValueError("Legacy attention active_channels must be in [1, channels]")
        self.active_channels = active_channels
        self.norm = nn.GroupNorm(_valid_groups(active_channels, 8), active_channels)
        self.attn = nn.MultiheadAttention(active_channels, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, h, w = x.shape
        active = x[:, : self.active_channels]
        tokens = self.norm(active).flatten(2).transpose(1, 2)
        out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        active = active + out.transpose(1, 2).reshape(
            bsz,
            self.active_channels,
            h,
            w,
        )
        if self.active_channels == x.shape[1]:
            return active
        return torch.cat([active, x[:, self.active_channels :]], dim=1)


class DiTBottleneckBlock(nn.Module):
    """One AdaLN-Zero DiT block specialised for the U-Net bottleneck.

    Plan-2 (Track A) stacks 8 of these blocks at the 16×16 bottleneck:
    flatten the spatial map to 256 tokens (per-channel hidden = ``channels``),
    apply attention + MLP with AdaLN-Zero on the condition vector, then
    unpatchify back to ``(C, H, W)``. 2D RoPE is used under the periodic
    boundary; an absolute 2D sinusoidal embedding under the open boundary so
    we do not falsely identify opposite edges.
    """

    def __init__(
        self,
        hidden: int,
        heads: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        if use_rope and (hidden // heads) % 4 != 0:
            raise ValueError("2D RoPE requires per-head dim divisible by 4")
        self.heads = int(heads)
        self.head_dim = hidden // heads
        self.use_rope = bool(use_rope)
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
            raise ValueError("Token count does not match RoPE grid")
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

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        grid: tuple[int, int],
        pos_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        s1, b1, g1, s2, b2, g2 = self.ada(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + s1[:, None]) + b1[:, None]
        if pos_embed is not None:
            h = h + pos_embed
        x = x + g1[:, None] * self._attention(h, grid)
        h = self.norm2(x) * (1 + s2[:, None]) + b2[:, None]
        x = x + g2[:, None] * self.mlp(h)
        return x


class TransformerBottleneck(nn.Module):
    """Stack of N DiT blocks on the U-Net bottleneck."""

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        depth: int = 8,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        boundary: str = "open",
        identity_init: bool = False,
        hidden: int | None = None,
    ) -> None:
        super().__init__()
        hidden = int(channels if hidden is None else hidden)
        if hidden <= 0:
            raise ValueError("Transformer bottleneck hidden size must be positive")
        self.use_rope = normalize_boundary(boundary) == "periodic"
        self.identity_init = bool(identity_init)
        if hidden == channels:
            self.input_proj: nn.Module = nn.Identity()
            self.output_proj: nn.Module = nn.Identity()
        else:
            if self.identity_init and hidden < channels:
                raise ValueError(
                    "Checkpoint-compatible Transformer widening requires hidden >= channels"
                )
            self.input_proj = nn.Linear(channels, hidden)
            self.output_proj = nn.Linear(hidden, channels)
            if self.identity_init:
                # Embed the original channels into the leading hidden
                # dimensions and project them back exactly. Together with
                # AdaLN-Zero blocks this keeps the complete extension equal to
                # the identity before the first optimizer step.
                nn.init.zeros_(self.input_proj.weight)
                nn.init.zeros_(self.input_proj.bias)
                nn.init.zeros_(self.output_proj.weight)
                nn.init.zeros_(self.output_proj.bias)
                eye = torch.eye(channels)
                with torch.no_grad():
                    self.input_proj.weight[:channels].copy_(eye)
                    self.output_proj.weight[:, :channels].copy_(eye)
        self.blocks = nn.ModuleList(
            [
                DiTBottleneckBlock(hidden, heads, cond_dim, mlp_ratio, use_rope=self.use_rope)
                for _ in range(depth)
            ]
        )
        # Absolute pos-emb is registered lazily once we know (h, w).
        self.pos_embed: torch.Tensor | None = None
        self._hidden = hidden

    def _maybe_build_pos_embed(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.use_rope:
            return None
        if (
            self.pos_embed is None
            or self.pos_embed.shape[1] != h * w
            or self.pos_embed.shape[-1] != self._hidden
        ):
            self.pos_embed = absolute_2d_embedding(h, w, self._hidden, device=device)
        return self.pos_embed.to(dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        bsz, ch, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.input_proj(tokens)
        pe = self._maybe_build_pos_embed(h, w, tokens.device, tokens.dtype)
        block_pos_embed = None
        if pe is not None and not self.identity_init:
            tokens = tokens + pe
        elif pe is not None:
            # In a checkpoint-compatible extension the new bottleneck must be
            # an exact identity at step 0.  Feed position information only to
            # the zero-gated attention branch instead of adding it to the
            # residual stream directly.
            block_pos_embed = pe
        for block in self.blocks:
            tokens = block(tokens, cond, (h, w), pos_embed=block_pos_embed)
        tokens = self.output_proj(tokens)
        return tokens.transpose(1, 2).reshape(bsz, ch, h, w)


class UNetVelocity(nn.Module):
    """Track A: U-Net with a switchable attention/Transformer bottleneck.

    Input channels are decided at construction time:
      - legacy plan-1 build: 6 / 9 channels (M_init ‖ Ω_τ)
      - plan-2 build: 8 / 11 channels (adds defect_field + j_field)

    The output stays at 3 channels (the bridge target velocity). Bridge-side
    tangent projection (Cart / RFM) is the loss's responsibility.
    """

    def __init__(self, cfg: dict, stats: ConditionStats | None = None) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        unet_cfg = model_cfg["unet"]
        self.input_repr = model_cfg.get("input_repr", "cartesian")
        self.state_channels = state_channels_from_cfg(cfg)
        self.boundary = normalize_boundary(cfg["data"].get("boundary", "open"))
        self.use_spatial_cond = bool(model_cfg.get("use_spatial_cond", False))
        self.per_sample_boundary = bool(model_cfg.get("per_sample_boundary", False))
        self.spatial_cond_j_scale = float(model_cfg.get("spatial_cond_j_scale", 1.0))
        self.spatial_cond_fields = spatial_condition_fields_from_cfg(cfg)
        self.spatial_cond_scales = spatial_condition_scales_from_cfg(cfg)
        cond_dim = int(model_cfg.get("cond_dim", 512))
        base = int(unet_cfg.get("base_channels", 64))
        mults = list(unet_cfg.get("channel_mults", [1, 2, 4, 8]))
        groups = int(unet_cfg.get("groups", 8))
        num_res_blocks = int(unet_cfg.get("num_res_blocks", 2))
        zero_init_residual = bool(unet_cfg.get("zero_init_residual", False))
        self.checkpoint_channel_expansion = bool(
            unet_cfg.get("checkpoint_channel_expansion", False)
        )
        self.checkpoint_source_base_channels = int(
            unet_cfg.get("checkpoint_source_base_channels", base)
        )
        self.checkpoint_source_groups = int(
            unet_cfg.get("checkpoint_source_groups", groups)
        )
        if self.checkpoint_channel_expansion:
            if self.checkpoint_source_base_channels >= base:
                raise ValueError(
                    "checkpoint_channel_expansion requires source base_channels < target base_channels"
                )
            if (
                base * self.checkpoint_source_groups
                != self.checkpoint_source_base_channels * groups
            ):
                raise ValueError(
                    "Checkpoint-compatible widening must preserve GroupNorm group size: "
                    "target_base/source_base must equal target_groups/source_groups"
                )
        bottleneck_downsample = bool(unet_cfg.get("bottleneck_downsample", True))
        bottleneck_mode = str(unet_cfg.get("bottleneck_mode", "attn")).lower()
        bottleneck_depth = int(unet_cfg.get("bottleneck_depth", 1))
        if bottleneck_depth <= 0:
            raise ValueError("model.unet.bottleneck_depth must be positive")
        valid_bottleneck_modes = {"attn", "transformer", "attn_transformer"}
        if bottleneck_mode not in valid_bottleneck_modes:
            raise ValueError(
                "model.unet.bottleneck_mode must be one of "
                f"{sorted(valid_bottleneck_modes)}, got {bottleneck_mode!r}"
            )
        if (
            bottleneck_mode == "attn"
            and "bottleneck_depth" in unet_cfg
            and bottleneck_depth != 1
        ):
            warnings.warn(
                "model.unet.bottleneck_depth is ignored by legacy "
                "bottleneck_mode='attn', which always has one attention block; "
                "use bottleneck_mode='transformer' or 'attn_transformer' to "
                "activate the requested depth",
                UserWarning,
                stacklevel=2,
            )
        time_cfg = model_cfg.get("time_conditioning", {}) or {}
        if isinstance(time_cfg, bool):
            time_cfg = {"enabled": time_cfg}
        self.use_time_conditioning = bool(time_cfg.get("enabled", False))
        self.mask_target_time = bool(model_cfg.get("mask_target_time", False))
        if self.use_time_conditioning and bool(model_cfg.get("disable_dt_condition", False)):
            raise ValueError("time_conditioning cannot be enabled with disable_dt_condition")
        time_cond_dim = int(time_cfg.get("embedding_dim", cond_dim))
        time_condition_scale = float(time_cfg.get("scale", 1.0))
        if time_condition_scale <= 0.0:
            raise ValueError("time_conditioning.scale must be positive")
        bnd = self.boundary
        self.cond_embed = make_condition_embedder(cfg, stats)
        if self.use_time_conditioning:
            time_buckets = cfg["data"].get("t_end_ns", cfg["data"].get("dt_scales", (1,)))
            self.time_conditioner: EndpointTimeConditioner | None = EndpointTimeConditioner(
                len(time_buckets),
                time_cond_dim,
                bucket_dim=int(time_cfg.get("bucket_dim", 128)),
                fourier_dim=int(time_cfg.get("fourier_dim", 128)),
                hidden_dim=int(time_cfg.get("hidden_dim", cond_dim)),
            )
        else:
            self.time_conditioner = None
        in_channels = input_image_channels(
            self.input_repr,
            self.use_spatial_cond,
            state_channels=self.state_channels,
            spatial_channels=len(self.spatial_cond_fields),
        )
        self.in_conv = BoundaryConv2d(in_channels, base, 3, padding=1, boundary=bnd)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base
        channels = []
        for i, mult in enumerate(mults):
            out_ch = base * int(mult)
            blocks = []
            for block_idx in range(num_res_blocks):
                blocks.append(
                    ResBlock(
                        ch if block_idx == 0 else out_ch,
                        out_ch,
                        cond_dim,
                        groups,
                        bnd,
                        time_cond_dim=time_cond_dim if self.use_time_conditioning else None,
                        time_condition_scale=time_condition_scale,
                        zero_init_residual=zero_init_residual,
                    )
                )
            self.down_blocks.append(nn.ModuleList(blocks))
            channels.append(out_ch)
            ch = out_ch
            if i < len(mults) - 1 or bottleneck_downsample:
                self.downsamples.append(
                    BoundaryConv2d(ch, ch, 3, stride=2, padding=1, boundary=bnd)
                )

        bottleneck_hidden = int(unet_cfg.get("bottleneck_hidden", ch))
        legacy_attention_channels = int(unet_cfg.get("bottleneck_legacy_channels", ch))
        legacy_attention_heads = int(unet_cfg.get("bottleneck_legacy_heads", 8))
        self.mid1 = ResBlock(
            ch,
            ch,
            cond_dim,
            groups,
            bnd,
            time_cond_dim=time_cond_dim if self.use_time_conditioning else None,
            time_condition_scale=time_condition_scale,
            zero_init_residual=zero_init_residual,
        )
        self.bottleneck_transformer: TransformerBottleneck | None = None
        if bottleneck_mode == "transformer":
            self.attn = TransformerBottleneck(
                ch,
                cond_dim,
                depth=bottleneck_depth,
                heads=int(unet_cfg.get("bottleneck_heads", 8)),
                mlp_ratio=float(unet_cfg.get("bottleneck_mlp_ratio", 4.0)),
                boundary=self.boundary,
                hidden=bottleneck_hidden,
            )
        elif bottleneck_mode == "attn_transformer":
            # Keep the legacy attention module (and therefore its checkpoint
            # keys) while adding a zero-gated deep transformer extension.
            self.attn = BottleneckAttention(
                ch,
                legacy_attention_heads,
                active_channels=legacy_attention_channels,
            )
            self.bottleneck_transformer = TransformerBottleneck(
                ch,
                cond_dim,
                depth=bottleneck_depth,
                heads=int(unet_cfg.get("bottleneck_heads", 8)),
                mlp_ratio=float(unet_cfg.get("bottleneck_mlp_ratio", 4.0)),
                boundary=self.boundary,
                identity_init=True,
                hidden=bottleneck_hidden,
            )
        elif bool(unet_cfg.get("attention", True)):
            self.attn = BottleneckAttention(
                ch,
                legacy_attention_heads,
                active_channels=legacy_attention_channels,
            )
        else:
            self.attn = nn.Identity()
        self.mid2 = ResBlock(
            ch,
            ch,
            cond_dim,
            groups,
            bnd,
            time_cond_dim=time_cond_dim if self.use_time_conditioning else None,
            time_condition_scale=time_condition_scale,
            zero_init_residual=zero_init_residual,
        )

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, skip_ch in reversed(list(enumerate(channels))):
            blocks = [
                ResBlock(
                    ch + skip_ch,
                    skip_ch,
                    cond_dim,
                    groups,
                    bnd,
                    time_cond_dim=time_cond_dim if self.use_time_conditioning else None,
                    time_condition_scale=time_condition_scale,
                    zero_init_residual=zero_init_residual,
                )
            ]
            for _ in range(num_res_blocks - 1):
                blocks.append(
                    ResBlock(
                        skip_ch,
                        skip_ch,
                        cond_dim,
                        groups,
                        bnd,
                        time_cond_dim=time_cond_dim if self.use_time_conditioning else None,
                        time_condition_scale=time_condition_scale,
                        zero_init_residual=zero_init_residual,
                    )
                )
            self.up_blocks.append(nn.ModuleList(blocks))
            ch = skip_ch
            if i > 0:
                self.upsamples.append(
                    BoundaryConv2d(ch, channels[i - 1], 3, padding=1, boundary=bnd)
                )
                ch = channels[i - 1]

        self.out_norm = nn.GroupNorm(_valid_groups(ch, groups), ch)
        self.out_conv = BoundaryConv2d(ch, self.state_channels, 3, padding=1, boundary=bnd)

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

    def _boundary_periodicity(
        self,
        cond: dict[str, torch.Tensor],
        batch: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not self.per_sample_boundary:
            return None
        return boundary_periodicity_from_cond(cond, batch, device)

    def forward(
        self,
        m_init: torch.Tensor,
        state: torch.Tensor,
        tau: torch.Tensor,
        cond: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        c = self.cond_embed(tau, cond)
        time_cond = self.time_conditioner(cond) if self.time_conditioner is not None else None
        if time_cond is not None and self.mask_target_time:
            # Keep the exact parameter graph for a capacity-matched ablation,
            # but remove all requested-time information from the forward pass.
            time_cond = time_cond * 0.0
        spatial = self._spatial_cond_image(cond, m_init)
        boundary_periodicity = self._boundary_periodicity(cond, int(m_init.shape[0]), m_init.device)
        h = self.in_conv(build_image_input(m_init, state, self.input_repr, spatial), boundary_periodicity)
        skips = []
        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, c, boundary_periodicity, time_cond)
            skips.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h, boundary_periodicity)
        mid = self.mid1(h, c, boundary_periodicity, time_cond)
        if isinstance(self.attn, TransformerBottleneck):
            mid = self.attn(mid, c)
        else:
            mid = self.attn(mid)
        if self.bottleneck_transformer is not None:
            mid = self.bottleneck_transformer(mid, c)
        h = self.mid2(mid, c, boundary_periodicity, time_cond)
        upsample_idx = 0
        for blocks in self.up_blocks:
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, c, boundary_periodicity, time_cond)
            if upsample_idx < len(self.upsamples) and skips:
                h = F.interpolate(h, size=skips[-1].shape[-2:], mode="nearest")
                h = self.upsamples[upsample_idx](h, boundary_periodicity)
                upsample_idx += 1
        return self.out_conv(F.silu(self.out_norm(h)), boundary_periodicity)
