"""Adapters for author-released PDE foundation models on the SKX x5 task.

This module deliberately contains no project-owned learned layers.  Each class
only translates the common ``[B, 37, H, W]`` x5 condition tensor into the
layout/resolution expected by an author implementation and translates its
output back to ``[B, 3, H, W]``.  Where an author checkpoint has a different
number of physical variables, all shape-compatible backbone tensors are
loaded and the author input/output heads remain newly initialized for x5.
"""

from __future__ import annotations

import json
import collections
import collections.abc
import importlib.machinery
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn


EXTERNAL_ROOT = Path(
    os.environ.get("X5_AUTHOR_REPO_ROOT", str(Path(__file__).resolve().parents[1] / "third_party"))
).resolve()
PRETRAINED_ROOT = Path(os.environ.get("FLARE_CHECKPOINT_ROOT", str(Path(__file__).resolve().parents[2] / "FLARE_checkpoints"))) / "pretrained"


def _prepend(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def _unwrap_state(payload: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint payload must be a mapping, got {type(payload)!r}")
    for key in ("model", "model_state", "state_dict", "model_state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            payload = candidate
            break
    return payload


def _load_matching_state(module: nn.Module, state: Mapping[str, Any]) -> dict[str, Any]:
    """Load only exact-shape tensors and return an auditable transfer summary."""
    target = module.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for original_key, value in state.items():
        if not torch.is_tensor(value):
            continue
        key = original_key
        for prefix in ("module.", "model."):
            if key.startswith(prefix) and key[len(prefix) :] in target:
                key = key[len(prefix) :]
                break
        if key in target and target[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(original_key)
    missing, unexpected = module.load_state_dict(compatible, strict=False)
    transferred = sum(int(value.numel()) for value in compatible.values())
    total = sum(int(value.numel()) for value in target.values() if torch.is_tensor(value))
    return {
        "policy": "author checkpoint: exact-name and exact-shape tensors",
        "loaded_tensor_count": len(compatible),
        "skipped_tensor_count": len(skipped),
        "missing_tensor_count": len(missing),
        "unexpected_tensor_count": len(unexpected),
        "transferred_parameter_values": transferred,
        "model_parameter_values": total,
        "transferred_fraction": transferred / max(total, 1),
        "representative_skipped_tensors": skipped[:12],
    }


class AuthorAdapter(nn.Module):
    algorithm: dict[str, Any]
    initialization: dict[str, Any]


class PoseidonTAdapter(AuthorAdapter):
    """Poseidon-T / ScOT with its released PDE foundation-model weights."""

    def __init__(self, condition_channels: int, initialize_pretrained: bool) -> None:
        super().__init__()
        repo = EXTERNAL_ROOT / "poseidon"
        _prepend(repo)
        from scOT.model import ScOT, ScOTConfig

        config_path = Path(__file__).resolve().parents[1] / "configs/pretrained/poseidon_t/config.json"
        config_dict = json.loads(config_path.read_text(encoding="utf-8"))
        config_dict["num_channels"] = condition_channels
        config_dict["num_out_channels"] = 3
        config_dict["channel_slice_list_normalized_loss"] = None
        self.core = ScOT(ScOTConfig(**config_dict))
        self.time_channel = 16  # 3 state + 13 spatial channels, then t_end_ns / 4.
        self.initialization = {"policy": "random x5 initialization"}
        if initialize_pretrained:
            from safetensors.torch import load_file

            checkpoint = PRETRAINED_ROOT / "poseidon_t" / "model.safetensors"
            self.initialization = _load_matching_state(self.core, load_file(str(checkpoint)))
            self.initialization["checkpoint"] = str(checkpoint)
        self.algorithm = {
            "architecture": "poseidon.scOT.model.ScOT (Poseidon-T)",
            "pretraining": "Poseidon multiphysics PDE foundation checkpoint",
            "native_resolution": 128,
            "patch_size": 4,
            "embed_dim": 48,
            "depths": [4, 4, 4, 4],
            "conditioning": "official continuous-time conditional layer normalization",
            "network_calls_per_prediction": 1,
            "adapter": "official Fourier resize 256->128->256; time=t_end_ns/4",
        }

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        time_value = condition[:, self.time_channel, 0, 0]
        return self.core(
            pixel_values=condition,
            time=time_value,
            return_dict=True,
        ).output


class DPOTTinyAdapter(AuthorAdapter):
    """DPOT-Ti with the authors' released 2-D pretraining checkpoint."""

    def __init__(self, condition_channels: int, initialize_pretrained: bool) -> None:
        super().__init__()
        repo = EXTERNAL_ROOT / "dpot"
        _prepend(repo)
        from models.dpot import DPOTNet

        self.core = DPOTNet(
            img_size=128,
            patch_size=8,
            mixing_type="afno",
            in_channels=condition_channels,
            out_channels=3,
            in_timesteps=10,
            out_timesteps=1,
            n_blocks=4,
            embed_dim=512,
            out_layer_dim=32,
            depth=4,
            modes=32,
            mlp_ratio=1.0,
            n_cls=12,
            normalize=False,
            act="gelu",
            time_agg="exp_mlp",
        )
        self.initialization = {"policy": "random x5 initialization"}
        if initialize_pretrained:
            checkpoint = PRETRAINED_ROOT / "dpot_ti" / "model_Ti.pth"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.initialization = _load_matching_state(self.core, _unwrap_state(payload))
            self.initialization["checkpoint"] = str(checkpoint)
        self.algorithm = {
            "architecture": "dpot.models.dpot.DPOTNet (DPOT-Ti)",
            "pretraining": "DPOT 12-dataset 2-D checkpoint",
            "native_resolution": 128,
            "patch_size": 8,
            "embed_dim": 512,
            "depth": 4,
            "fourier_modes": 32,
            "network_calls_per_prediction": 1,
            "history_steps": 10,
            "adapter": (
                "official 10-frame input history; bilinear resize "
                "256->128->256; one output time step"
            ),
        }

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim == 4:
            history = history[:, None].expand(-1, 10, -1, -1, -1)
        if history.ndim != 5 or history.shape[1] != 10:
            raise ValueError(
                "DPOT-Ti expects [batch, 10, channels, height, width], got "
                f"{tuple(history.shape)}"
            )
        batch, steps, channels, height, width = history.shape
        target_size = (height, width)
        x = F.interpolate(
            history.reshape(batch * steps, channels, height, width),
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )
        x = x.reshape(batch, steps, channels, 128, 128).permute(0, 3, 4, 1, 2)
        prediction, _ = self.core(x)
        prediction = prediction[:, :, :, 0].permute(0, 3, 1, 2)
        return F.interpolate(prediction, size=target_size, mode="bilinear", align_corners=False)


def _install_timm_droppath_compatibility_shim() -> None:
    """Provide the one timm primitive MPP imports without importing torchvision."""
    try:
        from timm.layers import DropPath as _DropPath  # noqa: F401

        return
    except Exception:
        for name in tuple(sys.modules):
            if name == "timm" or name.startswith("timm."):
                del sys.modules[name]

    def drop_path(
        x: torch.Tensor,
        drop_prob: float = 0.0,
        training: bool = False,
        scale_by_keep: bool = True,
    ) -> torch.Tensor:
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1.0 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    class DropPath(nn.Module):
        def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
            super().__init__()
            self.drop_prob = drop_prob
            self.scale_by_keep = scale_by_keep

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    timm_module = types.ModuleType("timm")
    timm_module.__path__ = []
    layers_module = types.ModuleType("timm.layers")
    layers_module.DropPath = DropPath
    layers_module.drop_path = drop_path
    timm_module.layers = layers_module
    sys.modules["timm"] = timm_module
    sys.modules["timm.layers"] = layers_module


class MPPAViTTinyAdapter(AuthorAdapter):
    """MPP AViT-Ti using the released multiphysics pretraining weights."""

    def __init__(self, condition_channels: int, initialize_pretrained: bool) -> None:
        super().__init__()
        _install_timm_droppath_compatibility_shim()
        repo = EXTERNAL_ROOT / "mpp"
        _prepend(repo)
        from models.avit import build_avit

        pretrained_state_slots = 12
        params = SimpleNamespace(
            block_type="axial",
            space_type="axial_attention",
            time_type="attention",
            embed_dim=192,
            num_heads=3,
            processor_blocks=12,
            patch_size=(16, 16),
            n_states=pretrained_state_slots,
            bias_type="rel",
            gradient_checkpointing=False,
        )
        self.core = build_avit(params)
        self.condition_channels = condition_channels
        self.initialization = {"policy": "random x5 initialization"}
        if initialize_pretrained:
            checkpoint = PRETRAINED_ROOT / "mpp_avit_ti" / "MPP_AViT_Ti.zip"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.initialization = _load_matching_state(self.core, _unwrap_state(payload))
            self.initialization["checkpoint"] = str(checkpoint)
        # This is the authors' released out-of-context fine-tuning operation:
        # preserve all 12 pretrained variable slots and append new projections
        # for the 37 x5 inputs/outputs.  x5 labels use only those new slots.
        self.core.expand_projections(condition_channels)
        self.condition_offset = pretrained_state_slots
        self.initialization["author_projection_expansion"] = {
            "pretrained_state_slots": pretrained_state_slots,
            "appended_x5_state_slots": condition_channels,
            "x5_label_offset": self.condition_offset,
        }
        self.algorithm = {
            "architecture": "mpp.models.avit.AViT (AViT-Ti)",
            "pretraining": "MPP heterogeneous multiphysics checkpoint",
            "patch_size": 16,
            "embed_dim": 192,
            "processor_blocks": 12,
            "attention": "axial space attention interleaved with time attention",
            "network_calls_per_prediction": 1,
            "history_steps": 16,
            "adapter": (
                "official 16-frame history; author expand_projections appends 37 x5 "
                "variable slots after the 12 pretrained slots"
            ),
        }

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim == 4:
            history = history[:, None].expand(-1, 16, -1, -1, -1)
        if history.ndim != 5 or history.shape[1] != 16:
            raise ValueError(
                "MPP AViT-Ti expects [batch, 16, channels, height, width], got "
                f"{tuple(history.shape)}"
            )
        history = history.permute(1, 0, 2, 3, 4)
        labels = torch.arange(
            self.condition_offset,
            self.condition_offset + self.condition_channels,
            device=history.device,
            dtype=torch.long,
        )[None].expand(history.shape[1], -1)
        boundary_conditions = torch.zeros(
            (history.shape[1], 2),
            device=history.device,
            dtype=history.dtype,
        )
        return self.core(history, labels, boundary_conditions)[:, :3]


def _install_cno_import_shims() -> None:
    """Expose only the Lightning/data-loader names needed to import CNO-FM."""
    # CNO's StyleGAN-derived loader discards the module returned by modern
    # ``torch.utils.cpp_extension.load`` and then imports it by name.  Put any
    # already-built author plugin directory on sys.path so that second import
    # remains compatible with current PyTorch.
    extension_root = Path(
        os.environ.get(
            "TORCH_EXTENSIONS_DIR",
            str(Path(__file__).resolve().parent / "torch_extensions"),
        )
    )
    for plugin in extension_root.glob(
        "filtered_lrelu_plugin/*/filtered_lrelu_plugin.so"
    ):
        _prepend(plugin.parent)
    if "pytorch_lightning" not in sys.modules:
        lightning = types.ModuleType("pytorch_lightning")
        lightning.__spec__ = importlib.machinery.ModuleSpec(
            "pytorch_lightning", loader=None
        )
        lightning.LightningModule = nn.Module
        sys.modules["pytorch_lightning"] = lightning
    name = "DataLoaders.load_utils"
    if name not in sys.modules:
        load_utils = types.ModuleType(name)
        load_utils.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)

        def unavailable(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("CNO-FM's repository data loader is not used by the x5 adapter")

        load_utils._load_dataset = unavailable
        sys.modules[name] = load_utils


def _find_cno_checkpoint() -> Path:
    extracted = PRETRAINED_ROOT / "cno_fm" / "extracted"
    candidates: list[Path] = []
    for suffix in ("*.ckpt", "*.pt", "*.pth"):
        candidates.extend(extracted.rglob(suffix))
    if not candidates:
        raise FileNotFoundError(
            f"no extracted CNO-FM checkpoint under {extracted}; finish and extract "
            "CNO-FM-checkpoint.zip first"
        )
    return max(candidates, key=lambda path: path.stat().st_size)


class CNOFMAdapter(AuthorAdapter):
    """109M-parameter CNO-FM with the authors' out-of-context fine-tuning heads."""

    def __init__(self, condition_channels: int, initialize_pretrained: bool) -> None:
        super().__init__()
        repo = EXTERNAL_ROOT / "cno" / "CNO2d_temporal"
        _prepend(repo)
        _install_cno_import_shims()
        from CNO_timeModule_CIN import CNO_time

        base = CNO_time(
            in_dim=5,
            in_size=128,
            N_layers=4,
            N_res=8,
            N_res_neck=8,
            channel_multiplier=82,
            batch_norm=True,
            out_dim=4,
            activation="cno_lrelu",
            time_steps=10,
            is_time=True,
            nl_dim=[2, 3],
            p_loss=1,
            lr=5.0e-4,
            batch_size=32,
            weight_decay=1.0e-6,
            loader_dictionary={},
            is_att=False,
            patch_size=2,
            dim_multiplier=1.0,
            depth=2,
            heads=2,
            dim_head_multiplier=0.5,
            mlp_dim_multiplier=1.0,
            emb_dropout=0.05,
        )
        self.initialization = {"policy": "random x5 initialization"}
        if initialize_pretrained:
            checkpoint = _find_cno_checkpoint()
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.initialization = _load_matching_state(base, _unwrap_state(payload))
            self.initialization["checkpoint"] = str(checkpoint)
            if self.initialization["transferred_fraction"] < 0.99:
                raise RuntimeError(
                    "CNO-FM checkpoint does not match the published 109M architecture: "
                    f"{self.initialization}"
                )

        # This is the authors' own out-of-context fine-tuning mechanism: a
        # learned pre-lift map into the four PDE variables plus time, and a new
        # three-channel projection head.  No project-owned layer is inserted.
        from test_and_fine_tune_utils.fine_tune_lift import initialize_FT

        self.core = initialize_FT(
            model=base,
            old_in_dim=5,
            new_in_dim=condition_channels,
            new_out_dim=3,
            old_out_dim=4,
        )
        self.time_channel = 16
        self.algorithm = {
            "architecture": "cno.CNO_time (CNO-FM 109M)",
            "pretraining": "six Poseidon fluid-PDE datasets",
            "native_resolution": 128,
            "lifting_dimension": 82,
            "up_downsampling_layers": 4,
            "residual_blocks_middle": 8,
            "residual_blocks_bottleneck": 8,
            "continuous_time_conditioning": "FiLM conditioned instance normalization",
            "network_calls_per_prediction": 1,
            "adapter": (
                "authors' initialize_FT out-of-context pre-lift and projection; "
                "bilinear resize 256->128->256; time=t_end_ns/4"
            ),
        }

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        target_size = condition.shape[-2:]
        time_value = condition[:, self.time_channel, 0, 0]
        x = F.interpolate(condition, size=(128, 128), mode="bilinear", align_corners=False)
        prediction = self.core(x, time_value)
        return F.interpolate(prediction, size=target_size, mode="bilinear", align_corners=False)


class PDEArenaUNetAdapter(AuthorAdapter):
    """The official PDEArena U-Net-2015 baseline implementation."""

    def __init__(self, condition_channels: int, initialize_pretrained: bool) -> None:
        super().__init__()
        repo = EXTERNAL_ROOT / "pdearena"
        _prepend(repo)
        from pdearena.modules.twod_unet2015 import Unet2015

        # Input condition fields are exogenous; only m_x/m_y/m_z are dynamic
        # output variables on x5.  The released forward's final reshape assumes
        # equal input/output component counts, so forward() below supplies the
        # exact same flattened tensor with singleton component dimension and
        # then removes that layout-only dimension.
        self.core = Unet2015(
            n_input_scalar_components=condition_channels,
            n_input_vector_components=0,
            n_output_scalar_components=3,
            n_output_vector_components=0,
            time_history=1,
            time_future=1,
            hidden_channels=64,
            activation="gelu",
        )
        self.initialization = {
            "policy": "trained from scratch",
            "reason": "PDEArena does not publish a transferable x5/micromagnetics checkpoint",
        }
        self.algorithm = {
            "architecture": "pdearena.modules.twod_unet2015.Unet2015",
            "hidden_channels": 64,
            "encoder_decoder_levels": 4,
            "history_steps": 1,
            "history_choice": (
                "author-supported one-step conditioned configuration; full magnetization "
                "is the Markov state of the x5 LLG task"
            ),
            "network_calls_per_prediction": 1,
            "adapter": "one history step; 37 exogenous/state inputs and 3 dynamic outputs",
        }

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.core(condition[:, :, None])[:, :, 0]


def _install_le_pde_dataset_import_shim() -> None:
    """Avoid importing DeepSnap dataset loaders that the model never calls."""
    # The 2022 author code used the pre-Python-3.10 import location.
    if not hasattr(collections, "Iterable"):
        collections.Iterable = collections.abc.Iterable
    if "IPython" not in sys.modules:
        ipython = types.ModuleType("IPython")
        display_module = types.ModuleType("IPython.display")
        ipython.__spec__ = importlib.machinery.ModuleSpec("IPython", loader=None)
        display_module.__spec__ = importlib.machinery.ModuleSpec("IPython.display", loader=None)

        class Image:  # Compatibility placeholder for an unused plotting helper.
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                pass

        def display(*_args: Any, **_kwargs: Any) -> None:
            return None

        ipython.get_ipython = lambda: None
        ipython.version_info = (0, 0)
        display_module.Image = Image
        display_module.display = display
        ipython.display = display_module
        sys.modules["IPython"] = ipython
        sys.modules["IPython.display"] = display_module
    if "sklearn" not in sys.modules:
        sklearn = types.ModuleType("sklearn")
        sklearn.__path__ = []
        cluster = types.ModuleType("sklearn.cluster")
        model_selection = types.ModuleType("sklearn.model_selection")
        sklearn.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None, is_package=True)
        cluster.__spec__ = importlib.machinery.ModuleSpec("sklearn.cluster", loader=None)
        model_selection.__spec__ = importlib.machinery.ModuleSpec(
            "sklearn.model_selection", loader=None
        )

        class SpectralClustering:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError("LE-PDE clustering utilities are not used by the x5 model")

        def train_test_split(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("LE-PDE split utilities are not used by the x5 model")

        cluster.SpectralClustering = SpectralClustering
        model_selection.train_test_split = train_test_split
        sklearn.cluster = cluster
        sklearn.model_selection = model_selection
        sys.modules["sklearn"] = sklearn
        sys.modules["sklearn.cluster"] = cluster
        sys.modules["sklearn.model_selection"] = model_selection
    if "termcolor" not in sys.modules:
        termcolor = types.ModuleType("termcolor")
        termcolor.__spec__ = importlib.machinery.ModuleSpec("termcolor", loader=None)
        termcolor.colored = lambda value, *_args, **_kwargs: str(value)
        sys.modules["termcolor"] = termcolor
    name = "le_pde.datasets.load_dataset"
    if name in sys.modules:
        return
    module = types.ModuleType(name)

    def unavailable(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("LE-PDE's repository dataset loader is not used by the x5 adapter")

    module.load_data = unavailable
    sys.modules[name] = module


class _LEPDEData:
    def __init__(self, condition: torch.Tensor) -> None:
        batch, channels, height, width = condition.shape
        self.node_feature = {
            "n0": condition.permute(0, 2, 3, 1).reshape(
                batch * height * width, 1, channels
            )
        }
        self.original_shape = (("n0", (height, width)),)
        self.grid_keys = ("n0",)
        self.part_keys = ()
        self.dyn_dims = (("n0", 3),)
        self.compute_func = (("n0", (0, None)),)
        self.mask = None

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


class LEPDEAdapter(AuthorAdapter):
    """LE-PDE's released CNN encoder, latent evolution MLP and decoder."""

    def __init__(self, condition_channels: int, initialize_pretrained: bool) -> None:
        super().__init__()
        repo_parent = EXTERNAL_ROOT
        _prepend(repo_parent)
        _prepend(repo_parent / "le_pde")
        _install_le_pde_dataset_import_shim()
        from le_pde.models import Contrastive

        self.core = Contrastive(
            input_size={"n0": condition_channels},
            output_size={"n0": 3},
            latent_size=256,
            encoder_type="cnn-s",
            evolution_type="mlp-3-elu-2",
            decoder_type="cnn-tr",
            input_shape=(("n0", (256, 256)),),
            grid_keys=("n0",),
            part_keys=(),
            no_latent_evo=False,
            temporal_bundle_steps=1,
            forward_type="Euler",
            channel_mode="exp-16",
            kernel_size=4,
            stride=2,
            padding=1,
            padding_mode="zeros",
            output_padding_str="None",
            encoder_mode="dense",
            encoder_n_linear_layers=0,
            act_name="elu",
            decoder_last_act_name="linear",
            is_pos_transform=False,
            normalization_type="gn",
            cnn_n_conv_layers=2,
            is_latent_flatten=True,
            reg_type="None",
            n_conv_blocks=4,
            n_latent_levs=1,
            n_conv_layers_latent=3,
            evo_conv_type="cnn",
            evo_pos_dims=-1,
            evo_inte_dims=-1,
            evo_groups=1,
            loss_type="mse",
            static_latent_size=0,
            static_encoder_type="None",
            static_input_size={"n0": 0},
            decoder_act_name="linear",
            is_prioritized_dropout=False,
            vae_mode="None",
        )
        self.initialization = {
            "policy": "trained from scratch",
            "reason": "LE-PDE publishes code/results but no compatible x5 checkpoint",
        }
        self.algorithm = {
            "architecture": "le_pde.models.Contrastive",
            "encoder": "cnn-s, 4 convolutional blocks",
            "latent_size": 256,
            "evolution": "mlp-3-elu-2 with Euler residual update",
            "decoder": "cnn-tr",
            "network_calls_per_prediction": 1,
            "adapter": "author graph-grid container with one 256x256 grid node type",
        }

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = condition.shape
        if (height, width) != (256, 256):
            raise ValueError("the selected LE-PDE author architecture is fixed to 256x256")
        predictions, _ = self.core(
            _LEPDEData(condition),
            pred_steps=1,
            use_grads=False,
            use_pos=False,
        )
        return predictions["n0"][:, 0].reshape(batch, height, width, 3).permute(0, 3, 1, 2)


def build_author_adapter(
    method: str,
    condition_channels: int,
    initialize_pretrained: bool,
) -> AuthorAdapter:
    builders = {
        "poseidon_t": PoseidonTAdapter,
        "dpot_ti": DPOTTinyAdapter,
        "mpp_avit_ti": MPPAViTTinyAdapter,
        "cno_fm": CNOFMAdapter,
        "pdearena_unet": PDEArenaUNetAdapter,
        "le_pde": LEPDEAdapter,
    }
    if method not in builders:
        raise ValueError(f"no model-zoo adapter registered for {method!r}")
    return builders[method](condition_channels, initialize_pretrained)
