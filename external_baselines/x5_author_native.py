#!/usr/bin/env python3
"""Train and benchmark third-party models with their native time semantics.

The models use three different temporal protocols:

* Poseidon and temporal CNO are lead-time-conditioned endpoint operators.
* DPOT, MPP-AViT, and PDEArena U-Net are fixed-step autoregressive models.
* LE-PDE encodes once, evolves a global latent repeatedly, and decodes only at
  the requested horizon.

Only x5 data parsing, physical-condition channels, author-supported variable
heads, and the unit-magnetization output projection are task adapters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, default_collate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTHOR_ROOT = Path(os.environ.get("X5_AUTHOR_REPO_ROOT", str(PROJECT_ROOT / "third_party")))
os.environ["X5_AUTHOR_REPO_ROOT"] = str(AUTHOR_ROOT)

if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from external_baselines.x5_external_baseline import (  # noqa: E402
    CONDITION_CHANNELS as LEGACY_CONDITION_CHANNELS,
    _atomic_checkpoint,
    _atomic_json,
    _autocast,
    _checkpoint_model_state,
    _configure_cuda,
    _dataset_config,
    _loader,
    build_condition_tensor,
    normalize_magnetization,
)
from external_baselines.x5_model_zoo import (  # noqa: E402
    LEPDEAdapter,
    _LEPDEData,
    build_author_adapter,
)
from graph.x5_author_native_rollout_dataset import (  # noqa: E402
    X5AuthorNativeRolloutCases,
    X5AuthorNativeTrainingDataset,
    attach_native_schedule_fields,
    first_legal_segment_sample,
)
from graph.x5_initial_horizon_dataset import X5InitialStateHorizonDataset  # noqa: E402
from skyrmion_cfm.config import seed_everything  # noqa: E402
from skyrmion_cfm.data.fixed_time import build_fixed_time_datasets  # noqa: E402
from skyrmion_cfm.eval.metrics import angular_error_deg, mse_m  # noqa: E402
from skyrmion_cfm.train import move_batch  # noqa: E402
from scripts.evaluate_skx_x5_same_condition_distribution import (  # noqa: E402
    DEFAULT_REPEAT_DATASET,
    _complete_test_groups,
    _override_multisegment_timing_durations,
    _prepare_multisegment_rollout_condition,
)


METHODS = (
    "poseidon_t",
    "cno_fm",
    "dpot_ti",
    "mpp_avit_ti",
    "pdearena_unet",
    "le_pde",
)
LEAD_TIME_METHODS = {"poseidon_t", "cno_fm"}
AUTOREGRESSIVE_METHODS = {"dpot_ti", "mpp_avit_ti", "pdearena_unet"}
LATENT_METHODS = {"le_pde"}
FP32_ONLY_METHODS = {"poseidon_t", "cno_fm", "dpot_ti", "mpp_avit_ti"}
PRETRAINED_METHODS = {"poseidon_t", "cno_fm", "dpot_ti", "mpp_avit_ti"}
LABELS = {
    "poseidon_t": "Poseidon-T (author-native lead-time endpoint)",
    "cno_fm": "CNO-FM 109M (author-native lead-time endpoint)",
    "dpot_ti": "DPOT-Ti (author-native 10-frame, 0.25 ns AR)",
    "mpp_avit_ti": "MPP AViT-Ti (author-native 16-frame, 0.25 ns AR)",
    "pdearena_unet": "PDEArena U-Net (author-native 0.25 ns AR)",
    "le_pde": "LE-PDE (author-native latent evolution)",
}
REPOSITORIES = {
    "poseidon_t": AUTHOR_ROOT / "poseidon",
    "cno_fm": AUTHOR_ROOT / "cno",
    "dpot_ti": AUTHOR_ROOT / "dpot",
    "mpp_avit_ti": AUTHOR_ROOT / "mpp",
    "pdearena_unet": AUTHOR_ROOT / "pdearena",
    "le_pde": AUTHOR_ROOT / "le_pde",
}
HORIZON_STEPS = (1, 2, 4, 8, 16, 20)
STEP_NS = 0.25
HISTORY_STEPS = {
    "dpot_ti": 10,
    "mpp_avit_ti": 16,
}
NATIVE_EXTRA_SPATIAL_FIELDS: tuple[tuple[str, float], ...] = (
    ("planned_j_x_field", 1.0e12),
    ("planned_j_y_field", 1.0e12),
    ("planned_j_z_field", 1.0e12),
)
NATIVE_EXTRA_SCALAR_FIELDS: tuple[tuple[str, float], ...] = (
    ("current_a_m2", 1.0e12),
    ("frame_init_time_ns", 5.0),
    ("frame_target_time_ns", 5.0),
    ("pulse_start_ns", 5.0),
    ("pulse_end_ns", 5.0),
)
NATIVE_CONDITION_CHANNELS = (
    LEGACY_CONDITION_CHANNELS
    + len(NATIVE_EXTRA_SPATIAL_FIELDS)
    + len(NATIVE_EXTRA_SCALAR_FIELDS)
)


def _native_single_channel(
    value: torch.Tensor,
    *,
    batch_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    if value.ndim == 3:
        value = value[:, None]
    if value.shape != (batch_size, 1, height, width):
        raise ValueError(f"native spatial condition shape mismatch: {tuple(value.shape)}")
    return value.float()


def build_native_condition_tensor(batch: dict[str, Any]) -> torch.Tensor:
    """Build the auditable x5 input, appending the future pulse schedule.

    The time channel is at index 16 in the first 29 channels. The appended
    fields encode the future-current schedule, including pulses beginning
    at 1 ns.
    """
    base = build_condition_tensor(batch)
    batch_size, _, height, width = base.shape
    extras: list[torch.Tensor] = []
    for key, scale in NATIVE_EXTRA_SPATIAL_FIELDS:
        value = batch.get(key)
        if value is None:
            field = base.new_zeros((batch_size, 1, height, width))
        else:
            field = _native_single_channel(
                value,
                batch_size=batch_size,
                height=height,
                width=width,
            ).to(device=base.device)
        extras.append(field / scale)
    for key, scale in NATIVE_EXTRA_SCALAR_FIELDS:
        value = batch.get(key)
        if value is None:
            scalar = base.new_zeros(batch_size)
        else:
            scalar = value.float().reshape(batch_size, -1)[:, 0].to(device=base.device)
        extras.append((scalar / scale)[:, None, None, None].expand(-1, 1, height, width))
    condition = torch.cat((base, *extras), dim=1)
    return torch.nan_to_num(condition, nan=0.0, posinf=16.0, neginf=-16.0).clamp_(-16.0, 16.0)


def _git_commit(path: Path) -> str:
    frozen_revision = path / "SOURCE_COMMIT"
    if frozen_revision.is_file():
        return frozen_revision.read_text(encoding="utf-8").strip()
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _relative_channel_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(2, prediction.ndim))
    numerator = (prediction - target).square().sum(dim=dims).sqrt()
    denominator = target.square().sum(dim=dims).sqrt().clamp_min(1.0e-8)
    return (numerator / denominator).mean()


def _normalized_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(2, prediction.ndim))
    residual = (prediction - target).square().mean(dim=dims)
    scale = target.square().mean(dim=dims).clamp_min(1.0e-7)
    return (residual / scale).mean()


class AuthorNativeX5Model(nn.Module):
    """Thin protocol wrapper around a model imported from a clean author clone."""

    def __init__(self, method: str, *, initialize_pretrained: bool = False) -> None:
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.method = method
        self.history_steps = HISTORY_STEPS.get(method, 1)
        self.adapter = build_author_adapter(
            method,
            NATIVE_CONDITION_CHANNELS,
            initialize_pretrained=initialize_pretrained,
        )
        self.initialization = self.adapter.initialization
        self.algorithm = dict(self.adapter.algorithm)
        self.algorithm.update(self._native_protocol_metadata())
        self.algorithm["x5_condition_adapter"] = (
            "state, instantaneous physics/control fields, complete planned-current map, "
            "absolute phase, and pulse start/end; no learned project-owned layer"
        )
        if method == "mpp_avit_ti":
            self.algorithm["adapter"] = (
                "official 16-frame author history containing the full x5 condition; "
                "author expand_projections and offset x5 field labels"
            )
        if method in HISTORY_STEPS:
            self.algorithm["history_steps"] = HISTORY_STEPS[method]
            self.algorithm["history_initialization"] = (
                "causal left padding with the physical t=0 state until the author window fills"
            )

    def _native_protocol_metadata(self) -> dict[str, Any]:
        if self.method in LEAD_TIME_METHODS:
            return {
                "native_temporal_protocol": "lead-time-conditioned endpoint operator",
                "physical_step_ns": None,
                "rollout_feedback": False,
                "stochastic": False,
            }
        if self.method in AUTOREGRESSIVE_METHODS:
            return {
                "native_temporal_protocol": "fixed-step autoregressive rollout",
                "physical_step_ns": STEP_NS,
                "rollout_feedback": True,
                "stochastic": False,
            }
        return {
            "native_temporal_protocol": "encode once, repeated global-latent evolution, decode on request",
            "physical_step_ns": STEP_NS,
            "rollout_feedback": "latent state",
            "stochastic": False,
        }

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _mpp_prediction_all_channels(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 5 or history.shape[1] != HISTORY_STEPS["mpp_avit_ti"]:
            raise ValueError(f"invalid MPP history shape: {tuple(history.shape)}")
        labels = torch.arange(
            self.adapter.condition_offset,
            self.adapter.condition_offset + NATIVE_CONDITION_CHANNELS,
            device=history.device,
            dtype=torch.long,
        )[None].expand(history.shape[0], -1)
        boundary_conditions = torch.zeros(
            (history.shape[0], 2),
            device=history.device,
            dtype=history.dtype,
        )
        return self.adapter.core(
            history.permute(1, 0, 2, 3, 4), labels, boundary_conditions
        )

    def raw_prediction(self, condition: torch.Tensor) -> torch.Tensor:
        if self.method == "mpp_avit_ti":
            return self._mpp_prediction_all_channels(condition)[:, :3]
        return self.adapter(condition)

    def training_loss(
        self,
        condition: torch.Tensor,
        target: torch.Tensor,
        *,
        future_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.method in {"poseidon_t", "cno_fm"}:
            # Both released lead-time models use p=1 in their author configs.
            return F.l1_loss(self.raw_prediction(condition), target)
        if self.method == "dpot_ti":
            return _relative_channel_l2(self.raw_prediction(condition), target)
        if self.method == "mpp_avit_ti":
            prediction = self._mpp_prediction_all_channels(condition)
            # The condition tensor contains exogenous controls and material
            # descriptors.  MPP's normalized state loss therefore applies to
            # the three x5 dynamic variables, not to those supplied inputs.
            return _normalized_mse(prediction[:, :3], target)
        if self.method == "pdearena_unet":
            return F.mse_loss(self.raw_prediction(condition), target)
        if self.method == "le_pde":
            if future_condition is None:
                raise ValueError("LE-PDE native latent loss requires the next-step encoded target")
            core = self.adapter.core
            data = _LEPDEData(condition)
            predictions, info = core(
                data,
                pred_steps=1,
                latent_pred_steps=1,
                is_recons=True,
                use_grads=False,
                use_pos=False,
            )
            batch, _, height, width = condition.shape
            prediction = predictions["n0"][:, 0].reshape(
                batch, height, width, 3
            ).permute(0, 3, 1, 2)
            reconstruction = info["recons"]["n0"][:, 0].reshape(
                batch, height, width, 3
            ).permute(0, 3, 1, 2)
            latent_prediction = info["latent_preds"][:, 0]
            latent_target = core.encoder(
                _LEPDEData(future_condition.to(dtype=latent_prediction.dtype)),
                use_grads=False,
                use_pos=False,
            )
            prediction_loss = F.mse_loss(prediction, target)
            reconstruction_loss = F.mse_loss(reconstruction, condition[:, :3])
            consistency_loss = F.mse_loss(latent_prediction, latent_target)
            return prediction_loss + reconstruction_loss + consistency_loss
        raise AssertionError(self.method)

    def predict(self, condition: torch.Tensor) -> torch.Tensor:
        return normalize_magnetization(self.raw_prediction(condition).float())

    def latent_rollout(self, condition: torch.Tensor, steps: int) -> torch.Tensor:
        if self.method != "le_pde":
            raise RuntimeError("latent_rollout is only valid for LE-PDE")
        if steps <= 0:
            raise ValueError("steps must be positive")
        adapter = self.adapter
        if not isinstance(adapter, LEPDEAdapter):
            raise TypeError(type(adapter))
        core = adapter.core
        batch, _, height, width = condition.shape
        latent = core.encoder(_LEPDEData(condition), use_grads=False, use_pos=False)
        for _ in range(steps):
            latent = core.evolve_latent(latent)
        decoded = core.decoder_n0(latent)
        prediction = decoded[:, 0].reshape(batch, height, width, 3).permute(0, 3, 1, 2)
        return normalize_magnetization(prediction.float())


def _mpp_author_parameter_groups(
    model: nn.Module, weight_decay: float = 1.0e-3
) -> list[dict[str, Any]]:
    """Mirror MPP ``train_basic.add_weight_decay`` for its released recipe."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if len(parameter.squeeze().shape) <= 1:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def _optimizer_for(
    model: nn.Module, method: str, lr: float, optimizer_recipe: str
) -> torch.optim.Optimizer:
    if optimizer_recipe == "author_mpp":
        if method != "mpp_avit_ti":
            raise ValueError("author_mpp optimizer recipe is only defined for mpp_avit_ti")
        from dadaptation import DAdaptAdan

        return DAdaptAdan(
            _mpp_author_parameter_groups(model),
            lr=1.0,
            growth_rate=1.05,
            log_every=100,
        )
    if method == "le_pde":
        return torch.optim.Adam(model.parameters(), lr=lr)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-6)


def _scheduler_for(
    optimizer: torch.optim.Optimizer,
    *,
    method: str,
    optimizer_recipe: str,
    start_step: int,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    if optimizer_recipe == "author_mpp":
        # Exact schedule constants from mpp/config/mpp_avit_ti_config.yaml:
        # max_epochs=500 and epoch_size=2000.  The released trainer interprets
        # learning_rate=-1 as D-Adaptation and nevertheless sets eta_min=-0.01.
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=500 * 2000,
            eta_min=-0.01,
            last_epoch=start_step - 1,
        )

    def multiplier(step: int) -> float:
        absolute = start_step + step
        if warmup_steps > 0 and absolute < warmup_steps:
            return max(absolute, 1) / warmup_steps
        progress = (absolute - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _validate_precision(method: str, precision: str) -> None:
    if method in FP32_ONLY_METHODS and precision != "fp32":
        raise ValueError(f"{method} must use fp32 with its released FFT/normalization path")


def _training_metadata(args: argparse.Namespace, model: AuthorNativeX5Model) -> dict[str, Any]:
    repository = REPOSITORIES[args.method]
    metadata = {
        "schema": "x5_author_native_training_v1",
        "method": args.method,
        "label": LABELS[args.method],
        "author_repository": str(repository),
        "author_commit": _git_commit(repository),
        "algorithm": model.algorithm,
        "initialization": model.initialization,
        "parameter_count": model.parameter_count,
        "condition_channels": NATIVE_CONDITION_CHANNELS,
        "x5_adaptation_scope": [
            "dataset parsing and unit normalization",
            "author-supported input/output variable projection",
            "instantaneous x5 physical/control condition channels",
            "complete planned-current map plus absolute phase and pulse window",
            "unit-magnetization output projection",
        ],
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "gradient_accumulation": int(args.grad_accum),
        "effective_batch_size": int(args.batch_size * args.grad_accum),
        "precision": args.precision,
        "learning_rate": 1.0 if args.optimizer_recipe == "author_mpp" else float(args.lr),
        "optimizer": (
            "DAdaptAdan"
            if args.optimizer_recipe == "author_mpp"
            else "Adam"
            if args.method == "le_pde"
            else "AdamW"
        ),
        "optimizer_recipe": args.optimizer_recipe,
        "weight_decay": (
            1.0e-3
            if args.optimizer_recipe == "author_mpp"
            else 0.0
            if args.method == "le_pde"
            else 1.0e-6
        ),
        "warmup_steps": 0 if args.optimizer_recipe == "author_mpp" else int(args.warmup_steps),
        "seed": int(args.seed),
        "x5_training_pair_view": (
            "the unchanged SCFM fixed-time, control-segmented training pairs"
            if args.method in LEAD_TIME_METHODS
            else (
                f"native 0.25 ns autoregression with {HISTORY_STEPS[args.method]} consecutive frames"
                if args.method in HISTORY_STEPS
                else (
                    "native 0.25 ns one-step pairs"
                    if args.method == "pdearena_unet"
                    else "native 0.25 ns pairs plus one following pair for latent consistency"
                )
            )
        ),
        "checkpoint_use": (
            "one checkpoint is used for both the formal single-segment accuracy "
            "evaluation and the complete 5 ns native-execution timing"
        ),
    }
    if args.optimizer_recipe == "author_mpp":
        metadata["author_batching_equivalence"] = {
            "released_batch_size": 1,
            "released_gradient_accumulation": 5,
            "released_effective_batch_size": 5,
            "device_vectorized_batch_size": int(args.batch_size),
            "device_gradient_accumulation": int(args.grad_accum),
            "effective_batch_preserved": int(args.batch_size * args.grad_accum) == 5,
            "semantics": "the five independent microbatches are evaluated as one vectorized batch",
        }
    return metadata


def _build_normal_training_dataset(
    cfg: dict[str, Any], method: str
) -> tuple[Any, int]:
    """Build the method's normal x5 task, never a special 0--5 ns training task.

    Lead-time operators use exactly the same random control-segment pairs as
    SCFM.  Native recurrent methods learn a 0.25 ns advance.  DPOT/MPP receive
    their released consecutive history length, while LE-PDE receives two
    consecutive advances solely for its author latent-consistency loss.
    """
    if method in LEAD_TIME_METHODS:
        train_dataset, _, _ = build_fixed_time_datasets(
            deepcopy(cfg), build_splits={"train"}
        )
        if train_dataset is None:
            raise RuntimeError("failed to build x5 training split")
        return train_dataset, 1

    base_train_dataset, _, _ = build_fixed_time_datasets(
        deepcopy(cfg), build_splits={"train"}
    )
    if base_train_dataset is None:
        raise RuntimeError("failed to build x5 training split")
    protocol = "latent" if method in LATENT_METHODS else "autoregressive"
    train_dataset = X5AuthorNativeTrainingDataset(
        base_train_dataset,
        protocol=protocol,
        seed=int(cfg.get("seed", 78)),
        history_steps=HISTORY_STEPS.get(method, 1),
    )
    # Sentinel zero means the wrapper already returns the author-native
    # history/future structures rather than FixedTimePairDataset clip tensors.
    return train_dataset, 0


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("author-native training requires CUDA")
    _validate_precision(args.method, args.precision)
    _configure_cuda()
    seed_everything(args.seed)
    device = torch.device("cuda")
    cfg = _dataset_config(args.config)
    # Training-seed ensembles must vary both initialization/shuffling and the
    # deterministic per-index sampling stream used by the native wrappers.
    cfg["seed"] = int(args.seed)
    train_ds, _ = _build_normal_training_dataset(cfg, args.method)
    loader = _loader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    iterator = iter(loader)
    model = AuthorNativeX5Model(
        args.method,
        initialize_pretrained=bool(args.pretrained and args.resume is None),
    ).to(device)
    optimizer = _optimizer_for(model, args.method, args.lr, args.optimizer_recipe)
    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("method") != args.method:
            raise ValueError("resume checkpoint method mismatch")
        model.load_state_dict(_checkpoint_model_state(checkpoint["model"]), strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        model.initialization = checkpoint.get("metadata", {}).get(
            "initialization", model.initialization
        )
    scheduler = _scheduler_for(
        optimizer,
        method=args.method,
        optimizer_recipe=args.optimizer_recipe,
        start_step=start_step,
        total_steps=args.steps,
        warmup_steps=args.warmup_steps,
    )
    metadata = _training_metadata(args, model)
    if args.resume is not None:
        metadata["resume_checkpoint"] = str(args.resume)
        metadata["resume_step"] = start_step
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output_dir / "protocol.json", metadata)
    log_path = args.output_dir / "train_metrics.jsonl"
    started = time.perf_counter()
    interval_started = started
    interval_losses: list[float] = []

    for step in range(start_step + 1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        update_losses: list[float] = []
        for _ in range(args.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = move_batch(batch, device)
            if args.method in HISTORY_STEPS:
                history_batches = batch.get("history_conditions")
                if not isinstance(history_batches, list):
                    raise TypeError("collated history_conditions must be a list of batches")
                condition = torch.stack(
                    [
                        build_native_condition_tensor(move_batch(item, device))
                        for item in history_batches
                    ],
                    dim=1,
                )
                target = batch["m_t"].float()
                future_condition = None
            elif args.method in LATENT_METHODS:
                condition = build_native_condition_tensor(batch)
                target = batch["m_t"].float()
                future_batch = move_batch(batch["future_condition"], device)
                future_condition = build_native_condition_tensor(future_batch)
            else:
                condition = build_native_condition_tensor(batch)
                target = batch["m_t"].float()
                future_condition = None
            with _autocast(device, args.precision):
                micro_loss = model.training_loss(
                    condition,
                    target,
                    future_condition=future_condition,
                )
                loss = micro_loss / args.grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {float(loss)}")
            loss.backward()
            update_losses.append(float(micro_loss.detach()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        interval_losses.append(statistics.fmean(update_losses))

        if step % args.log_every == 0 or step == args.steps:
            torch.cuda.synchronize(device)
            now = time.perf_counter()
            row = {
                "step": step,
                "loss": statistics.fmean(interval_losses),
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_s": now - started,
                "steps_per_s": len(interval_losses) / max(now - interval_started, 1.0e-9),
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                "device": torch.cuda.get_device_name(device),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(row, flush=True)
            interval_losses.clear()
            interval_started = now

        if (not args.no_checkpoint) and (
            step % args.save_every == 0 or step == args.steps
        ):
            payload = {
                "schema": "x5_author_native_checkpoint_v1",
                "method": args.method,
                "step": step,
                "model": _checkpoint_model_state(model),
                "optimizer": optimizer.state_dict(),
                "metadata": metadata,
            }
            _atomic_checkpoint(args.output_dir / f"checkpoint_{step:07d}.pt", payload)
            _atomic_checkpoint(args.output_dir / "checkpoint_latest.pt", payload)


def _batched_indices(size: int, batch_size: int) -> Iterable[range]:
    for start in range(0, size, batch_size):
        yield range(start, min(start + batch_size, size))


def _collated_condition(
    samples: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = default_collate(samples)
    return (
        build_native_condition_tensor(batch).half(),
        batch["m_t"].half(),
        batch["defect_field"].half(),
    )


def _prepare_direct_cache(
    cases: X5AuthorNativeRolloutCases,
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for horizon_step in HORIZON_STEPS:
        conditions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for indices in _batched_indices(len(cases), 8):
            samples = [cases.direct_sample(index, horizon_step) for index in indices]
            condition, target, mask = _collated_condition(samples)
            conditions.append(condition)
            targets.append(target)
            masks.append(mask)
        cache[horizon_step] = (
            torch.cat(conditions),
            torch.cat(targets),
            torch.cat(masks),
        )
    return cache


def _prepare_step_cache(
    cases: X5AuthorNativeRolloutCases,
) -> tuple[list[torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    conditions_by_step: list[torch.Tensor] = []
    targets: dict[int, torch.Tensor] = {}
    masks: dict[int, torch.Tensor] = {}
    for step_index in range(cases.num_steps):
        conditions: list[torch.Tensor] = []
        step_targets: list[torch.Tensor] = []
        step_masks: list[torch.Tensor] = []
        for indices in _batched_indices(len(cases), 8):
            samples = [cases.step_sample(index, step_index) for index in indices]
            condition, target, mask = _collated_condition(samples)
            conditions.append(condition)
            step_targets.append(target)
            step_masks.append(mask)
        conditions_by_step.append(torch.cat(conditions))
        horizon_step = step_index + 1
        if horizon_step in HORIZON_STEPS:
            targets[horizon_step] = torch.cat(step_targets)
            masks[horizon_step] = torch.cat(step_masks)
    return conditions_by_step, targets, masks


def _quality_sums(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float, int]:
    prediction = normalize_magnetization(prediction.float())
    target = normalize_magnetization(target.float())
    mask = mask.float()
    batch = prediction.shape[0]
    # Match the project's formal paper metric: average only over magnetic
    # material.  Without this mask, every zero-vector cell outside the sample
    # contributes acos(0) = 90 degrees and badly inflates small geometries.
    mse_per_sample = mse_m(prediction, target, mask=mask)
    angle_per_sample = angular_error_deg(prediction, target, mask=mask)
    return float(mse_per_sample.sum()), float(angle_per_sample.sum()), batch


def _device_tensor(value: torch.Tensor, device: torch.device, precision: str) -> torch.Tensor:
    if precision == "fp32":
        return value.to(device=device, dtype=torch.float32, non_blocking=True)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return value.to(device=device, dtype=dtype, non_blocking=True)


def _causal_history_input(
    history: list[torch.Tensor],
    condition: torch.Tensor,
    history_steps: int,
) -> torch.Tensor:
    """Append one rolled condition and form the author's fixed causal window."""
    if history_steps <= 1:
        return condition
    history.append(condition)
    if len(history) > history_steps:
        del history[0]
    padded = [history[0]] * (history_steps - len(history)) + history
    return torch.stack(padded, dim=1)


@torch.no_grad()
def _benchmark_direct(
    model: AuthorNativeX5Model,
    cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    precision: str,
    batch_size: int,
    repeats: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    timings: dict[str, Any] = {}
    quality: dict[str, Any] = {}
    for horizon_step in HORIZON_STEPS:
        condition_cpu, target_cpu, mask_cpu = cache[horizon_step]
        # Warm up on one complete native endpoint query.
        warm = _device_tensor(condition_cpu[:batch_size], device, precision)
        with _autocast(device, precision):
            model.predict(warm)
        torch.cuda.synchronize(device)
        runs: list[float] = []
        for _ in range(repeats):
            elapsed = 0.0
            count = 0
            for indices in _batched_indices(len(condition_cpu), batch_size):
                condition = _device_tensor(condition_cpu[indices], device, precision)
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                with _autocast(device, precision):
                    model.predict(condition)
                torch.cuda.synchronize(device)
                elapsed += time.perf_counter() - started
                count += len(indices)
            runs.append(1000.0 * elapsed / count)

        mse_sum = angle_sum = 0.0
        count = 0
        for indices in _batched_indices(len(condition_cpu), batch_size):
            condition = _device_tensor(condition_cpu[indices], device, precision)
            target = target_cpu[indices].to(device, dtype=torch.float32)
            mask = mask_cpu[indices].to(device, dtype=torch.float32)
            with _autocast(device, precision):
                prediction = model.predict(condition)
            mse, angle, n = _quality_sums(prediction, target, mask)
            mse_sum += mse
            angle_sum += angle
            count += n
        key = f"{horizon_step * STEP_NS:g}"
        timings[key] = {
            "mean_ms_per_sample": statistics.fmean(runs),
            "std_ms_per_sample": statistics.stdev(runs) if len(runs) > 1 else 0.0,
            "runs": runs,
        }
        quality[key] = {"mse": mse_sum / count, "mean_angle_deg": angle_sum / count}
    return timings, quality


@torch.no_grad()
def _benchmark_autoregressive(
    model: AuthorNativeX5Model,
    conditions_by_step: list[torch.Tensor],
    targets: dict[int, torch.Tensor],
    masks: dict[int, torch.Tensor],
    *,
    device: torch.device,
    precision: str,
    batch_size: int,
    repeats: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    samples = conditions_by_step[0].shape[0]
    warm_indices = range(0, min(batch_size, samples))
    warm_prediction: torch.Tensor | None = None
    warm_history: list[torch.Tensor] = []
    for condition_cpu in conditions_by_step:
        warm_condition = _device_tensor(condition_cpu[warm_indices], device, precision)
        if warm_prediction is not None:
            warm_condition[:, :3] = warm_prediction.to(dtype=warm_condition.dtype)
        warm_input = _causal_history_input(
            warm_history, warm_condition, model.history_steps
        )
        with _autocast(device, precision):
            warm_prediction = model.predict(warm_input)
    torch.cuda.synchronize(device)

    run_values = {step: [] for step in HORIZON_STEPS}
    for _ in range(repeats):
        elapsed_by_horizon = {step: 0.0 for step in HORIZON_STEPS}
        for indices in _batched_indices(samples, batch_size):
            prediction: torch.Tensor | None = None
            history: list[torch.Tensor] = []
            cumulative = 0.0
            for step_index, condition_cpu in enumerate(conditions_by_step):
                condition = _device_tensor(condition_cpu[indices], device, precision)
                if prediction is not None:
                    condition[:, :3] = prediction.to(dtype=condition.dtype)
                model_input = _causal_history_input(
                    history, condition, model.history_steps
                )
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                with _autocast(device, precision):
                    prediction = model.predict(model_input)
                torch.cuda.synchronize(device)
                cumulative += time.perf_counter() - started
                horizon_step = step_index + 1
                if horizon_step in elapsed_by_horizon:
                    elapsed_by_horizon[horizon_step] += cumulative
        for horizon_step in HORIZON_STEPS:
            run_values[horizon_step].append(
                1000.0 * elapsed_by_horizon[horizon_step] / samples
            )

    mse_sums = {step: 0.0 for step in HORIZON_STEPS}
    angle_sums = {step: 0.0 for step in HORIZON_STEPS}
    counts = {step: 0 for step in HORIZON_STEPS}
    for indices in _batched_indices(samples, batch_size):
        prediction = None
        history = []
        for step_index, condition_cpu in enumerate(conditions_by_step):
            condition = _device_tensor(condition_cpu[indices], device, precision)
            if prediction is not None:
                condition[:, :3] = prediction.to(dtype=condition.dtype)
            model_input = _causal_history_input(history, condition, model.history_steps)
            with _autocast(device, precision):
                prediction = model.predict(model_input)
            horizon_step = step_index + 1
            if horizon_step in targets:
                target = targets[horizon_step][indices].to(device, dtype=torch.float32)
                mask = masks[horizon_step][indices].to(device, dtype=torch.float32)
                mse, angle, n = _quality_sums(prediction, target, mask)
                mse_sums[horizon_step] += mse
                angle_sums[horizon_step] += angle
                counts[horizon_step] += n

    timings: dict[str, Any] = {}
    quality: dict[str, Any] = {}
    for horizon_step in HORIZON_STEPS:
        key = f"{horizon_step * STEP_NS:g}"
        runs = run_values[horizon_step]
        timings[key] = {
            "mean_ms_per_sample": statistics.fmean(runs),
            "std_ms_per_sample": statistics.stdev(runs) if len(runs) > 1 else 0.0,
            "runs": runs,
        }
        quality[key] = {
            "mse": mse_sums[horizon_step] / counts[horizon_step],
            "mean_angle_deg": angle_sums[horizon_step] / counts[horizon_step],
        }
    return timings, quality


@torch.no_grad()
def _benchmark_latent(
    model: AuthorNativeX5Model,
    conditions_by_step: list[torch.Tensor],
    targets: dict[int, torch.Tensor],
    masks: dict[int, torch.Tensor],
    *,
    device: torch.device,
    precision: str,
    batch_size: int,
    repeats: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # LE-PDE receives the initial state and the complete schedule encoding, then
    # evolves only its global latent.  It does not consume teacher-forced fields.
    initial = conditions_by_step[0]
    timings: dict[str, Any] = {}
    quality: dict[str, Any] = {}
    for horizon_step in HORIZON_STEPS:
        warm = _device_tensor(initial[:batch_size], device, precision)
        with _autocast(device, precision):
            model.latent_rollout(warm, horizon_step)
        torch.cuda.synchronize(device)
        runs: list[float] = []
        for _ in range(repeats):
            elapsed = 0.0
            count = 0
            for indices in _batched_indices(len(initial), batch_size):
                condition = _device_tensor(initial[indices], device, precision)
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                with _autocast(device, precision):
                    model.latent_rollout(condition, horizon_step)
                torch.cuda.synchronize(device)
                elapsed += time.perf_counter() - started
                count += len(indices)
            runs.append(1000.0 * elapsed / count)

        mse_sum = angle_sum = 0.0
        count = 0
        for indices in _batched_indices(len(initial), batch_size):
            condition = _device_tensor(initial[indices], device, precision)
            target = targets[horizon_step][indices].to(device, dtype=torch.float32)
            mask = masks[horizon_step][indices].to(device, dtype=torch.float32)
            with _autocast(device, precision):
                prediction = model.latent_rollout(condition, horizon_step)
            mse, angle, n = _quality_sums(prediction, target, mask)
            mse_sum += mse
            angle_sum += angle
            count += n
        key = f"{horizon_step * STEP_NS:g}"
        timings[key] = {
            "mean_ms_per_sample": statistics.fmean(runs),
            "std_ms_per_sample": statistics.stdev(runs) if len(runs) > 1 else 0.0,
            "runs": runs,
        }
        quality[key] = {"mse": mse_sum / count, "mean_angle_deg": angle_sum / count}
    return timings, quality


@torch.no_grad()
def benchmark(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("author-native benchmark requires CUDA")
    _validate_precision(args.method, args.precision)
    _configure_cuda()
    seed_everything(args.seed)
    device = torch.device("cuda")
    cfg = _dataset_config(args.config)
    _, _, test_ds = build_fixed_time_datasets(cfg, build_splits={"test"})
    if test_ds is None:
        raise RuntimeError("failed to build x5 test split")
    cases = X5AuthorNativeRolloutCases(test_ds, limit=args.samples)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != args.method:
        raise ValueError("checkpoint method mismatch")
    model = AuthorNativeX5Model(args.method).to(device)
    model.load_state_dict(_checkpoint_model_state(checkpoint["model"]), strict=True)
    model.eval()

    if args.method in LEAD_TIME_METHODS:
        cache: Any = _prepare_direct_cache(cases)
    else:
        cache = _prepare_step_cache(cases)

    results_by_batch: dict[str, Any] = {}
    quality_reference: dict[str, Any] | None = None
    for batch_size in args.batch_sizes:
        if args.method in LEAD_TIME_METHODS:
            timings, quality = _benchmark_direct(
                model,
                cache,
                device=device,
                precision=args.precision,
                batch_size=batch_size,
                repeats=args.repeats,
            )
        elif args.method in AUTOREGRESSIVE_METHODS:
            timings, quality = _benchmark_autoregressive(
                model,
                *cache,
                device=device,
                precision=args.precision,
                batch_size=batch_size,
                repeats=args.repeats,
            )
        else:
            timings, quality = _benchmark_latent(
                model,
                *cache,
                device=device,
                precision=args.precision,
                batch_size=batch_size,
                repeats=args.repeats,
            )
        results_by_batch[str(batch_size)] = {"timing": timings, "quality": quality}
        if quality_reference is None:
            quality_reference = quality

    payload = {
        "schema": "x5_author_native_benchmark_v1",
        "method": args.method,
        "label": LABELS[args.method],
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "training_metadata": checkpoint.get("metadata"),
        "author_repository": str(REPOSITORIES[args.method]),
        "author_commit": _git_commit(REPOSITORIES[args.method]),
        "algorithm": model.algorithm,
        "parameter_count": model.parameter_count,
        "precision": args.precision,
        "device": torch.cuda.get_device_name(device),
        "samples": args.samples,
        "repeats": args.repeats,
        "horizons_ns": [step * STEP_NS for step in HORIZON_STEPS],
        "selection": cases.selection_metadata(),
        "results_by_batch_size": results_by_batch,
        "timing_scope": (
            "model execution plus unit-magnetization projection; excludes data loading, "
            "host-to-device transfer, and x5 condition construction"
        ),
        "quality_metric": {
            "angle": "mean per-cell angular error over defect_field magnetic mask",
            "mse": "mean squared vector error over defect_field magnetic mask",
            "magnetization_normalization": "per-cell unit length before both metrics",
        },
        "quality": quality_reference,
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("evaluation produced no rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _formal_segment_cases(dataset: Any, max_cases: int = 0) -> list[dict[str, int]]:
    """Return the exact deterministic cases used by evaluate_x5_single_segment."""
    cases: list[dict[str, int]] = []
    for record_index in dataset._valid_record_idx:
        for choice_index, start_frame in dataset._segment_boundary_specs(
            int(record_index)
        ):
            cases.append(
                {
                    "record_index": int(record_index),
                    "choice_index": int(choice_index),
                    "start_frame": int(start_frame),
                }
            )
    if max_cases > 0:
        cases = cases[: int(max_cases)]
    return cases


def _formal_sample(dataset: Any, case: dict[str, int], seed: int) -> dict[str, Any]:
    return dataset._build_sample(
        case["record_index"],
        case["choice_index"],
        np.random.default_rng(seed + 104_729 * case["record_index"]),
        frame_init=case["start_frame"],
        apply_augment=False,
    )


def _native_substeps(
    dataset: Any,
    case: dict[str, int],
    sample: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    start_frame = int(sample["frame_init"].item())
    target_frame = int(sample["frame_target"].item())
    duration_ns = float(sample["t_end_ns"].item())
    steps = max(1, int(round(duration_ns / STEP_NS)))
    frames = [
        int(round(start_frame + (target_frame - start_frame) * index / steps))
        for index in range(steps + 1)
    ]
    frames[0] = start_frame
    frames[-1] = target_frame
    if any(right <= left for left, right in zip(frames, frames[1:])):
        raise RuntimeError(
            f"non-increasing native rollout frames for {sample['run_id']}: {frames}"
        )
    substeps: list[dict[str, Any]] = []
    for step, (left, right) in enumerate(zip(frames, frames[1:])):
        hop = dataset._build_visual_sample_for_frames(
            case["record_index"],
            left,
            right,
            np.random.default_rng(
                seed + 104_729 * case["record_index"] + 1_009 * step
            ),
        )
        substeps.append(
            attach_native_schedule_fields(dataset, case["record_index"], hop)
        )
    return substeps


# Close every physical segment with
# a fractional tail instead of advancing across the control switch.
def _exact_control_substeps(
    dataset: Any,
    record_index: int,
    sample: dict[str, Any],
) -> list[dict[str, Any]]:
    """Split one exact control interval into native steps with no time gap.

    Fixed-step baselines use 0.25-ns calls plus one final fractional call when
    the protocol boundary is off grid.  The fractional call is explicit in
    the condition tensor; this is an extrapolation for models trained only on
    the native 0.25-ns step, but it preserves the physical switch time and the
    exact final timestamp instead of silently dropping an interval.
    """
    duration_ns = float(sample["t_end_ns"])
    start_ns = float(sample["frame_init_time_ns"])
    end_ns = float(sample["frame_target_time_ns"])
    if duration_ns <= 0.0:
        raise ValueError("exact control segment duration must be positive")
    if not math.isclose(
        end_ns - start_ns,
        duration_ns,
        rel_tol=0.0,
        abs_tol=2.0e-4,
    ):
        raise RuntimeError(
            f"exact segment time mismatch for {sample['run_id']}: "
            f"{end_ns - start_ns} != {duration_ns} ns"
        )

    full_steps = int(math.floor(duration_ns / STEP_NS + 1.0e-10))
    durations = [STEP_NS] * full_steps
    remainder_ns = duration_ns - sum(durations)
    if remainder_ns > 1.0e-7:
        durations.append(remainder_ns)
    elif durations:
        durations[-1] += remainder_ns
    else:
        durations = [duration_ns]
    if any(value <= 0.0 or value > STEP_NS + 1.0e-7 for value in durations):
        raise RuntimeError(
            f"invalid exact native steps for {sample['run_id']}: {durations}"
        )
    if not math.isclose(
        sum(durations), duration_ns, rel_tol=0.0, abs_tol=2.0e-7
    ):
        raise RuntimeError(
            f"native step time does not close for {sample['run_id']}: "
            f"{sum(durations)} != {duration_ns} ns"
        )

    driven = float(sample.get("drive_fraction", 0.0)) > 0.5
    substeps: list[dict[str, Any]] = []
    cursor_ns = start_ns
    for step_duration_ns in durations:
        hop = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in sample.items()
        }
        next_ns = cursor_ns + step_duration_ns
        hop["t_end_ns"] = torch.tensor(step_duration_ns, dtype=torch.float32)
        hop["t_end_s"] = torch.tensor(
            step_duration_ns * 1.0e-9, dtype=torch.float32
        )
        hop["t_end_index"] = torch.tensor(
            dataset._t_end_bucket_index(step_duration_ns), dtype=torch.long
        )
        hop["dt_s"] = hop["t_end_s"].clone()
        hop["dt_index"] = hop["t_end_index"].clone()
        hop["frame_init_time_ns"] = torch.tensor(cursor_ns, dtype=torch.float32)
        hop["frame_target_time_ns"] = torch.tensor(next_ns, dtype=torch.float32)
        hop["control_time_ns"] = torch.tensor(cursor_ns, dtype=torch.float32)
        hop["drive_time_s"] = torch.tensor(
            step_duration_ns * 1.0e-9 if driven else 0.0,
            dtype=torch.float32,
        )
        hop["relax_time_s"] = torch.tensor(
            0.0 if driven else step_duration_ns * 1.0e-9,
            dtype=torch.float32,
        )
        hop["drive_fraction"] = torch.tensor(
            1.0 if driven else 0.0, dtype=torch.float32
        )
        substeps.append(
            attach_native_schedule_fields(dataset, int(record_index), hop)
        )
        cursor_ns = next_ns
    if not math.isclose(cursor_ns, end_ns, rel_tol=0.0, abs_tol=2.0e-4):
        raise RuntimeError(
            f"native substeps end at {cursor_ns}, expected {end_ns} ns for "
            f"{sample['run_id']}"
        )
    return substeps


def _one_batch(sample: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return move_batch(default_collate([sample]), device)


@torch.no_grad()
def _formal_prediction(
    model: AuthorNativeX5Model,
    dataset: Any,
    case: dict[str, int],
    sample: dict[str, Any],
    *,
    device: torch.device,
    precision: str,
    seed: int,
) -> torch.Tensor:
    if model.method in LEAD_TIME_METHODS:
        condition = build_native_condition_tensor(_one_batch(sample, device))
        with _autocast(device, precision):
            return model.predict(condition)

    substeps = _native_substeps(dataset, case, sample, seed)
    if model.method in LATENT_METHODS:
        condition = build_native_condition_tensor(_one_batch(substeps[0], device))
        with _autocast(device, precision):
            return model.latent_rollout(condition, len(substeps))

    prediction: torch.Tensor | None = None
    history: list[torch.Tensor] = []
    for hop in substeps:
        condition = build_native_condition_tensor(_one_batch(hop, device))
        if prediction is not None:
            condition[:, :3] = prediction.to(dtype=condition.dtype)
        model_input = _causal_history_input(history, condition, model.history_steps)
        with _autocast(device, precision):
            prediction = model.predict(model_input)
    if prediction is None:
        raise RuntimeError("native rollout produced no prediction")
    return prediction


@torch.no_grad()
def evaluate_formal(args: argparse.Namespace) -> None:
    """Evaluate normal x5 accuracy on the exact SCFM single-segment cases."""
    if not torch.cuda.is_available():
        raise RuntimeError("formal native evaluation requires CUDA")
    _validate_precision(args.method, args.precision)
    _configure_cuda()
    seed_everything(args.seed)
    device = torch.device("cuda")
    cfg = _dataset_config(args.config)
    _, _, test_dataset = build_fixed_time_datasets(cfg, build_splits={"test"})
    if test_dataset is None:
        raise RuntimeError("failed to build x5 test split")
    cases = _formal_segment_cases(test_dataset, args.max_cases)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != args.method:
        raise ValueError("checkpoint method mismatch")
    model = AuthorNativeX5Model(args.method).to(device)
    model.load_state_dict(_checkpoint_model_state(checkpoint["model"]), strict=True)
    model.eval()

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        sample = _formal_sample(test_dataset, case, args.seed)
        prediction = _formal_prediction(
            model,
            test_dataset,
            case,
            sample,
            device=device,
            precision=args.precision,
            seed=args.seed,
        ).float()
        batch = _one_batch(sample, device)
        target = batch["m_t"].float()
        mask = batch.get("defect_field")
        angle = angular_error_deg(prediction, target, mask=mask)
        mse = mse_m(prediction, target, mask=mask)
        run_id = str(sample["run_id"])
        rows.append(
            {
                "method": args.method,
                "record_index": case["record_index"],
                "base_id": run_id.split("_base", 1)[1].split("_", 1)[0]
                if "_base" in run_id
                else run_id,
                "run_id": run_id,
                "control_segment_index": int(
                    sample["control_segment_index"].item()
                ),
                "start_time_ns": float(sample["frame_init_time_ns"].item()),
                "end_time_ns": float(sample["frame_target_time_ns"].item()),
                "duration_ns": float(sample["t_end_ns"].item()),
                "native_steps": (
                    1
                    if args.method in LEAD_TIME_METHODS
                    else max(1, int(round(float(sample["t_end_ns"].item()) / STEP_NS)))
                ),
                "ang": float(angle[0].detach().cpu()),
                "mse": float(mse[0].detach().cpu()),
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(cases):
            print(
                json.dumps({"completed_cases": index + 1, "total_cases": len(cases)}),
                flush=True,
            )

    _atomic_csv(args.output, rows)
    payload = {
        "schema": "x5_native_formal_single_segment_v1",
        "status": "complete",
        "method": args.method,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "author_commit": _git_commit(REPOSITORIES[args.method]),
        "cases": len(rows),
        "case_protocol": "exact SCFM formal control-segment cases",
        "temporal_execution": model.algorithm["native_temporal_protocol"],
        "precision": args.precision,
        "device": torch.cuda.get_device_name(device),
        "quality_metric": "formal defect_field-masked angular error and vector MSE",
    }
    _atomic_json(args.output.with_suffix(".json"), payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def _timed_rollout(
    model: AuthorNativeX5Model,
    condition: torch.Tensor,
    *,
    horizon_ns: float,
    precision: str,
) -> torch.Tensor:
    native_steps = int(round(float(horizon_ns) / STEP_NS))
    if model.method in LEAD_TIME_METHODS:
        with _autocast(condition.device, precision):
            return model.predict(condition)
    if model.method in LATENT_METHODS:
        with _autocast(condition.device, precision):
            return model.latent_rollout(condition, native_steps)
    prediction: torch.Tensor | None = None
    history: list[torch.Tensor] = []
    for _ in range(native_steps):
        step_condition = condition.clone()
        if prediction is not None:
            step_condition[:, :3] = prediction.to(dtype=step_condition.dtype)
        model_input = _causal_history_input(
            history, step_condition, model.history_steps
        )
        with _autocast(condition.device, precision):
            prediction = model.predict(model_input)
    if prediction is None:
        raise RuntimeError(f"{horizon_ns:g} ns rollout produced no prediction")
    return prediction


def _first_multisegment_timing_path(
    dataset: Any,
    seed: int,
    condition_dir: Path,
    timing_segment_durations_ns: list[float] | None = None,
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    groups = _complete_test_groups(
        dataset,
        repeat_dataset=DEFAULT_REPEAT_DATASET,
        repeats_per_group=5,
    )
    base, record_indices = next(iter(groups.items()))
    record_index = int(record_indices[0])
    prepared = _prepare_multisegment_rollout_condition(
        dataset,
        base=base,
        record_indices=record_indices,
        condition_dir=condition_dir,
    )
    if timing_segment_durations_ns is not None:
        prepared = _override_multisegment_timing_durations(
            prepared, dataset, timing_segment_durations_ns
        )
    path = list(prepared["segment_samples"][0])
    repeat = prepared["metadata"]["repeats"][0]
    return record_index, path, {
        "base_id": base,
        "run_id": str(path[0]["run_id"]),
        "control_segment_indices": prepared["metadata"][
            "control_segment_indices"
        ],
        "segments": repeat["segments"],
        "handoff_count": prepared["metadata"]["handoff_count"],
        "absolute_span_ns": repeat["absolute_span_ns"],
        "composed_model_horizon_ns": repeat["composed_model_horizon_ns"],
        "handoffs": repeat["handoffs"],
        "timing_segment_durations_ns": prepared["metadata"].get(
            "timing_segment_durations_ns"
        ),
        "timing_role": (
            "standardized fixed-duration exact-control-segment rollout"
            if timing_segment_durations_ns is not None
            else "complete exact-control-segment rollout"
        ),
    }


# Diagnostic timing path only; the
# paper retains the frozen representative 5-ns speed workload.
def _multisegment_timing_conditions(
    model: AuthorNativeX5Model,
    dataset: Any,
    record_index: int,
    path: list[dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[list[torch.Tensor], int, list[int]]:
    if model.method in LEAD_TIME_METHODS:
        return [
            build_native_condition_tensor(
                move_batch(
                    default_collate(
                        [
                            attach_native_schedule_fields(
                                dataset,
                                int(record_index),
                                {
                                    key: value.clone()
                                    if torch.is_tensor(value)
                                    else value
                                    for key, value in sample.items()
                                },
                            )
                        ]
                        * batch_size
                    ),
                    device,
                )
            )
            for sample in path
        ], len(path), [1] * len(path)

    hops_by_segment = [
        _exact_control_substeps(
            dataset,
            int(record_index),
            sample,
        )
        for sample in path
    ]
    # Keep one timing template per control
    # segment and execute it for the exact 8+12 native-step counts below.
    # Materializing 20 duplicated full-resolution condition tensors would add
    # artificial resident-memory pressure even though condition construction
    # and transfer are outside the reported model-side timing scope.
    segment_step_counts = [len(segment) for segment in hops_by_segment]
    conditions = [
        build_native_condition_tensor(
            move_batch(default_collate([segment[0]] * batch_size), device)
        )
        for segment in hops_by_segment
    ]
    return conditions, sum(segment_step_counts), segment_step_counts


def _timed_multisegment_rollout(
    model: AuthorNativeX5Model,
    conditions: list[torch.Tensor],
    *,
    precision: str,
    segment_step_counts: list[int],
) -> torch.Tensor:
    if not conditions:
        raise RuntimeError("empty multisegment timing conditions")
    device = conditions[0].device
    if model.method in LEAD_TIME_METHODS:
        prediction: torch.Tensor | None = None
        for original in conditions:
            condition = original.clone()
            if prediction is not None:
                condition[:, :3] = prediction.to(dtype=condition.dtype)
            with _autocast(device, precision):
                prediction = model.predict(condition)
        if prediction is None:
            raise RuntimeError("direct multisegment timing produced no prediction")
        return prediction
    if model.method in LATENT_METHODS:
        if len(conditions) != len(segment_step_counts):
            raise RuntimeError("latent timing conditions are misaligned")
        prediction: torch.Tensor | None = None
        for original, step_count in zip(conditions, segment_step_counts):
            condition = original.clone()
            if prediction is not None:
                condition[:, :3] = prediction.to(dtype=condition.dtype)
            with _autocast(device, precision):
                prediction = model.latent_rollout(condition, step_count)
        if prediction is None:
            raise RuntimeError("latent exact-control timing path is inconsistent")
        return prediction

    if len(conditions) != len(segment_step_counts):
        raise RuntimeError("autoregressive timing conditions are misaligned")
    prediction = None
    history: list[torch.Tensor] = []
    for original, step_count in zip(conditions, segment_step_counts):
        for _ in range(step_count):
            condition = original.clone()
            if prediction is not None:
                condition[:, :3] = prediction.to(dtype=condition.dtype)
            model_input = _causal_history_input(
                history, condition, model.history_steps
            )
            with _autocast(device, precision):
                prediction = model.predict(model_input)
    if prediction is None:
        raise RuntimeError("autoregressive multisegment timing produced no prediction")
    return prediction


@torch.no_grad()
def benchmark_horizon(args: argparse.Namespace) -> None:
    """Time complete native execution using the formal accuracy checkpoint."""
    horizon_ns = float(args.horizon_ns)
    if args.timing_segment_durations_ns is not None and not args.multisegment_rollout:
        raise ValueError(
            "--timing-segment-durations-ns requires --multisegment-rollout"
        )
    native_steps = int(round(horizon_ns / STEP_NS))
    if (not args.multisegment_rollout) and (
        horizon_ns <= 0.0
        or not math.isclose(
            native_steps * STEP_NS,
            horizon_ns,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        raise ValueError(f"horizon_ns must be a positive multiple of {STEP_NS:g} ns")
    if not torch.cuda.is_available():
        raise RuntimeError("native timing benchmark requires CUDA")
    _validate_precision(args.method, args.precision)
    _configure_cuda()
    seed_everything(args.seed)
    device = torch.device("cuda")
    cfg = _dataset_config(args.config)
    _, _, test_dataset = build_fixed_time_datasets(cfg, build_splits={"test"})
    if test_dataset is None:
        raise RuntimeError("failed to build x5 test split")
    multisegment_path: list[dict[str, Any]] | None = None
    multisegment_record_index: int | None = None
    if args.multisegment_rollout:
        if args.matched_segment_index is not None:
            raise ValueError(
                "--multisegment-rollout and --matched-segment-index are mutually exclusive"
            )
        (
            multisegment_record_index,
            multisegment_path,
            selection,
        ) = _first_multisegment_timing_path(
            test_dataset,
            args.seed,
            args.output.parent / "timing_condition",
            args.timing_segment_durations_ns,
        )
        horizon_ns = float(selection["composed_model_horizon_ns"])
        native_steps = int(round(horizon_ns / STEP_NS))
    elif args.matched_segment_index is not None:
        matched_sample, matched_metadata = first_legal_segment_sample(
            test_dataset,
            segment_index=int(args.matched_segment_index),
            horizon_ns=horizon_ns,
            seed=args.seed,
            preferred_run_id=args.matched_run_id,
        )
        if args.method in LEAD_TIME_METHODS:
            sample = matched_sample
        else:
            # Recurrent and latent interfaces consume a native 0.25-ns step
            # condition, then execute the requested number of legal steps.
            sample, _step_metadata = first_legal_segment_sample(
                test_dataset,
                segment_index=int(args.matched_segment_index),
                horizon_ns=STEP_NS,
                seed=args.seed,
                preferred_run_id=args.matched_run_id,
            )
        selection = {
            **matched_metadata,
            "timing_role": "matched legal single-segment query",
        }
    else:
        cases = X5AuthorNativeRolloutCases(test_dataset, limit=1)
        if args.method in LEAD_TIME_METHODS:
            if horizon_ns <= cases.max_horizon_ns:
                sample = cases.direct_sample(0, native_steps)
            else:
                sample = X5InitialStateHorizonDataset(test_dataset, horizon_ns, 1)[0]
            for key, _ in NATIVE_EXTRA_SPATIAL_FIELDS:
                sample.pop(key, None)
            for key, _ in NATIVE_EXTRA_SCALAR_FIELDS:
                sample.pop(key, None)
        else:
            sample = cases.step_sample(0, 0)
        selection = {
            **cases.selection_metadata(),
            "timing_role": "legacy initial-state timing view",
        }
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != args.method:
        raise ValueError("checkpoint method mismatch")
    model = AuthorNativeX5Model(args.method).to(device)
    model.load_state_dict(_checkpoint_model_state(checkpoint["model"]), strict=True)
    model.eval()

    results: dict[str, Any] = {}
    realized_network_calls: int | None = None
    for batch_size in args.batch_sizes:
        if multisegment_path is not None:
            assert multisegment_record_index is not None
            (
                conditions,
                execution_steps,
                segment_step_counts,
            ) = _multisegment_timing_conditions(
                model,
                test_dataset,
                multisegment_record_index,
                multisegment_path,
                batch_size=batch_size,
                device=device,
                seed=args.seed,
            )
            realized_network_calls = execution_steps
        else:
            batch = move_batch(default_collate([sample] * batch_size), device)
            condition = build_native_condition_tensor(batch)
        for _ in range(args.warmup):
            if multisegment_path is not None:
                _timed_multisegment_rollout(
                    model,
                    conditions,
                    precision=args.precision,
                    segment_step_counts=segment_step_counts,
                )
            else:
                _timed_rollout(
                    model,
                    condition,
                    horizon_ns=horizon_ns,
                    precision=args.precision,
                )
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        runs: list[float] = []
        for _ in range(args.repeats):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            if multisegment_path is not None:
                _timed_multisegment_rollout(
                    model,
                    conditions,
                    precision=args.precision,
                    segment_step_counts=segment_step_counts,
                )
            else:
                _timed_rollout(
                    model,
                    condition,
                    horizon_ns=horizon_ns,
                    precision=args.precision,
                )
            torch.cuda.synchronize(device)
            runs.append(1000.0 * (time.perf_counter() - started) / batch_size)
        results[str(batch_size)] = {
            "mean_ms_per_sample": statistics.fmean(runs),
            "std_ms_per_sample": statistics.stdev(runs) if len(runs) > 1 else 0.0,
            "runs_ms_per_sample": runs,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        }

    network_calls = (
        realized_network_calls
        if realized_network_calls is not None
        else (1 if args.method in LEAD_TIME_METHODS else native_steps)
    )
    payload = {
        "schema": "x5_native_complete_horizon_timing_v1",
        "parameter_count": model.parameter_count,
        "status": "complete",
        "method": args.method,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "author_commit": _git_commit(REPOSITORIES[args.method]),
        "horizon_ns": horizon_ns,
        "multisegment_rollout": args.multisegment_rollout,
        "reference_mode": (
            "exact_control_multisegment_rollout_endpoint"
            if args.multisegment_rollout
            else "timing_horizon_query"
        ),
        "native_step_ns": None if args.method in LEAD_TIME_METHODS else STEP_NS,
        "network_calls": network_calls,
        "execution": model.algorithm["native_temporal_protocol"],
        "exact_control_temporal_adapter": (
            (
                "one exact-duration endpoint call per control segment"
                if args.method in LEAD_TIME_METHODS
                else (
                    "per-segment re-encode with nearest enclosing integer "
                    "latent-step count; LE-PDE cannot represent a fractional step"
                    if args.method in LATENT_METHODS
                    else (
                        "continuous autoregressive history with 0.25-ns native "
                        "steps and one fractional tail interval per segment"
                    )
                )
            )
            if args.multisegment_rollout
            else None
        ),
        "batch_sizes": args.batch_sizes,
        "repeats": args.repeats,
        "precision": args.precision,
        "device": torch.cuda.get_device_name(device),
        "results_by_batch_size": results,
        "selection": selection,
        "timing_scope": (
            (
                "complete model-side execution over the exact protocol control "
                "segments with prediction handoff; "
                if args.multisegment_rollout
                else f"complete model-side native execution through {horizon_ns:g} ns; "
            )
            + "data loading and host transfer excluded"
        ),
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def inspect_model(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    model = AuthorNativeX5Model(args.method, initialize_pretrained=args.pretrained)
    print(
        json.dumps(
            {
                "method": args.method,
                "label": LABELS[args.method],
                "parameter_count": model.parameter_count,
                "algorithm": model.algorithm,
                "initialization": model.initialization,
                "author_repository": str(REPOSITORIES[args.method]),
                "author_commit": _git_commit(REPOSITORIES[args.method]),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=78)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser("inspect")
    _common(inspect_parser)
    inspect_parser.add_argument("--pretrained", action="store_true")
    inspect_parser.set_defaults(func=inspect_model)

    train_parser = commands.add_parser("train")
    _common(train_parser)
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--steps", type=int, default=10_000)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--grad-accum", type=int, default=2)
    train_parser.add_argument("--num-workers", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=1.0e-4)
    train_parser.add_argument("--warmup-steps", type=int, default=500)
    train_parser.add_argument(
        "--optimizer-recipe",
        choices=("standardized", "author_mpp"),
        default="standardized",
    )
    train_parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    train_parser.add_argument("--log-every", type=int, default=50)
    train_parser.add_argument("--save-every", type=int, default=2500)
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--pretrained", action="store_true")
    train_parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Run a training smoke test without writing large checkpoint files.",
    )
    train_parser.set_defaults(func=train)

    benchmark_parser = commands.add_parser("benchmark")
    _common(benchmark_parser)
    benchmark_parser.add_argument("--config", type=Path, required=True)
    benchmark_parser.add_argument("--checkpoint", type=Path, required=True)
    benchmark_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser.add_argument("--samples", type=int, default=128)
    benchmark_parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32])
    benchmark_parser.add_argument("--repeats", type=int, default=10)
    benchmark_parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    benchmark_parser.set_defaults(func=benchmark)

    formal_parser = commands.add_parser("formal-evaluate")
    _common(formal_parser)
    formal_parser.add_argument("--config", type=Path, required=True)
    formal_parser.add_argument("--checkpoint", type=Path, required=True)
    formal_parser.add_argument("--output", type=Path, required=True)
    formal_parser.add_argument("--max-cases", type=int, default=0)
    formal_parser.add_argument(
        "--precision", choices=("fp32", "bf16", "fp16"), default="bf16"
    )
    formal_parser.set_defaults(func=evaluate_formal)

    # Use a horizon-neutral command name;
    # retain time-5ns only as a backward-compatible alias for frozen scripts.
    timing_parser = commands.add_parser("time-horizon", aliases=["time-5ns"])
    _common(timing_parser)
    timing_parser.add_argument("--config", type=Path, required=True)
    timing_parser.add_argument("--checkpoint", type=Path, required=True)
    timing_parser.add_argument("--output", type=Path, required=True)
    timing_parser.add_argument("--horizon-ns", type=float, default=5.0)
    timing_parser.add_argument(
        "--multisegment-rollout",
        action="store_true",
        help=(
            "Ignore --horizon-ns and time the exact-control drive->post-relax "
            "path with the method's native temporal interface."
        ),
    )
    timing_parser.add_argument(
        "--timing-segment-durations-ns",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Timing-only durations for --multisegment-rollout, for example "
            "2 3 for a standardized two-segment 5-ns workload."
        ),
    )
    timing_parser.add_argument(
        "--matched-segment-index",
        type=int,
        default=None,
        help=(
            "Require an exact target in this constant-control segment; use 2 "
            "for the matched 3-ns post-relaxation benchmark."
        ),
    )
    timing_parser.add_argument(
        "--matched-run-id",
        default=None,
        help="Optionally require the exact held-out run used by the quality set.",
    )
    timing_parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32])
    timing_parser.add_argument("--repeats", type=int, default=10)
    timing_parser.add_argument("--warmup", type=int, default=2)
    timing_parser.add_argument(
        "--precision", choices=("fp32", "bf16", "fp16"), default="bf16"
    )
    timing_parser.set_defaults(func=benchmark_horizon)
    return parser


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
