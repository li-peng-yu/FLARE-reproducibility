#!/usr/bin/env python3
"""Train and benchmark unmodified third-party neural solvers on SKX x5.

The only project-owned component is the data/condition adapter.  Model
architectures and stochastic/refinement algorithms are imported directly from
the pinned third-party implementations under ``third_party/``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skyrmion_cfm.config import load_config, seed_everything
from skyrmion_cfm.data.fixed_time import build_fixed_time_datasets
from skyrmion_cfm.train import move_batch
from graph.x5_initial_horizon_dataset import X5InitialStateHorizonDataset


EXTERNAL_ROOT = Path(os.environ.get("X5_AUTHOR_REPO_ROOT", str(PROJECT_ROOT / "third_party")))
REPOSITORIES = {
    "neuralop_fno": EXTERNAL_ROOT / "neuraloperator",
    "acdm": EXTERNAL_ROOT / "autoreg-pde-diffusion",
    "pde_refiner": EXTERNAL_ROOT / "autoreg-pde-diffusion",
    "poseidon_t": EXTERNAL_ROOT / "poseidon",
    "dpot_ti": EXTERNAL_ROOT / "dpot",
    "mpp_avit_ti": EXTERNAL_ROOT / "mpp",
    "cno_fm": EXTERNAL_ROOT / "cno",
    "pdearena_unet": EXTERNAL_ROOT / "pdearena",
    "le_pde": EXTERNAL_ROOT / "le_pde",
}
METHOD_LABELS = {
    "neuralop_fno": "Official FNO (NeuralOperator) fp32 b32",
    "acdm": "ACDM-r20 (Kohl et al.) bf16 b32",
    "pde_refiner": "PDE-Refiner-r4 (Kohl et al.) bf16 b32",
    "poseidon_t": "Poseidon-T pretrained + x5 fine-tune",
    "dpot_ti": "DPOT-Ti pretrained + x5 fine-tune",
    "mpp_avit_ti": "MPP AViT-Ti pretrained + x5 fine-tune",
    "cno_fm": "CNO-FM 109M pretrained + x5 fine-tune",
    "pdearena_unet": "PDEArena U-Net trained on x5",
    "le_pde": "LE-PDE latent evolution trained on x5",
}
HORIZONS_NS = (0.25, 0.5, 1.0, 2.0, 4.0)
MODEL_ZOO_METHODS = {
    "poseidon_t",
    "dpot_ti",
    "mpp_avit_ti",
    "cno_fm",
    "pdearena_unet",
    "le_pde",
}
DIRECT_METHODS = MODEL_ZOO_METHODS | {"neuralop_fno"}
FP32_ONLY_METHODS = {
    "neuralop_fno",
    "poseidon_t",
    "dpot_ti",
    "mpp_avit_ti",
    "cno_fm",
}


# These are the x5 spatial fields already constructed by FixedTimePairDataset.
# Divisors are the same physical scales used by the SCFM spatial conditioner.
SPATIAL_FIELDS: tuple[tuple[str, float], ...] = (
    ("defect_field", 1.0),
    ("j_x_field", 1.0e11),
    ("j_y_field", 1.0e11),
    ("j_z_field", 1.0e11),
    ("b_x_field", 1.0e-1),
    ("b_y_field", 1.0e-1),
    ("b_z_field", 1.0e-1),
    ("temperature_field", 3.0e1),
    ("msat_field", 1.0e5),
    ("aex_field", 1.0e-12),
    ("ku1_field", 1.0e7),
    ("dind_field", 1.0e-4),
    ("alpha_field", 1.0),
)

# Scalar maps retain controls that cannot be reconstructed from the spatial
# fields alone.  Broadcasting is the conventional adapter used by unconditioned
# field operators; no learned project-owned encoder is inserted.
SCALAR_FIELDS: tuple[tuple[str, float], ...] = (
    ("t_end_ns", 4.0),
    ("drive_time_s", 4.0e-9),
    ("relax_time_s", 4.0e-9),
    ("drive_fraction", 1.0),
    ("target_offset_norm", 1.0),
    ("temp_k", 30.0),
    ("pol_eff", 1.0),
    ("epsilon_prime", 1.0),
    ("fixed_layer_x", 1.0),
    ("fixed_layer_y", 1.0),
    ("fixed_layer_z", 1.0),
    ("theta_dl_eff", 1.0),
    ("r_fl_dl", 1.0),
)
CONDITION_CHANNELS = 3 + len(SPATIAL_FIELDS) + len(SCALAR_FIELDS)


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _checkpoint_model_state(model_or_state: nn.Module | dict[str, Any]) -> dict[str, Any]:
    """Return a plain PyTorch state dict without TensorLy's sentinel entry.

    NeuralOperator's factorized-tensor state dict currently exposes
    ``_metadata`` as a regular mapping item.  PyTorch's recursive loader then
    treats it as a parameter name and rejects an otherwise valid checkpoint.
    The factor tensors themselves contain all values needed for restoration.
    """
    state = model_or_state.state_dict() if isinstance(model_or_state, nn.Module) else model_or_state
    return {
        key: value
        for key, value in state.items()
        if key != "_metadata" and not key.endswith("._metadata")
    }


def _as_single_channel(
    value: torch.Tensor,
    *,
    batch_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    if value.ndim == 2 and value.shape == (height, width):
        value = value[None, None].expand(batch_size, -1, -1, -1)
    elif value.ndim == 3:
        if value.shape[0] != batch_size:
            raise ValueError(f"unexpected unbatched spatial field shape: {tuple(value.shape)}")
        value = value[:, None]
    elif value.ndim == 4:
        if value.shape[1] != 1:
            raise ValueError(f"spatial field must have one channel: {tuple(value.shape)}")
    else:
        raise ValueError(f"unsupported spatial field shape: {tuple(value.shape)}")
    if value.shape != (batch_size, 1, height, width):
        raise ValueError(f"spatial field shape mismatch: {tuple(value.shape)}")
    return value


def build_condition_tensor(batch: dict[str, Any]) -> torch.Tensor:
    """Build the common, non-learned conditioning tensor for all baselines."""
    m_init = batch["m_init"].float()
    if m_init.ndim != 4 or m_init.shape[1] != 3:
        raise ValueError(f"m_init must be [B,3,H,W], got {tuple(m_init.shape)}")
    batch_size, _, height, width = m_init.shape
    zero_map = torch.zeros(
        (batch_size, 1, height, width),
        device=m_init.device,
        dtype=m_init.dtype,
    )
    channels = [m_init]
    for key, scale in SPATIAL_FIELDS:
        value = batch.get(key)
        if value is None:
            field = zero_map
        else:
            field = _as_single_channel(
                value.float(),
                batch_size=batch_size,
                height=height,
                width=width,
            )
        channels.append(field / scale)
    for key, scale in SCALAR_FIELDS:
        value = batch.get(key)
        if value is None:
            scalar = torch.zeros(batch_size, device=m_init.device, dtype=m_init.dtype)
        else:
            scalar = value.float().reshape(batch_size, -1)[:, 0]
        channels.append((scalar / scale)[:, None, None, None].expand(-1, 1, height, width))
    condition = torch.cat(channels, dim=1)
    condition = torch.nan_to_num(condition, nan=0.0, posinf=16.0, neginf=-16.0)
    return condition.clamp_(-16.0, 16.0)


def normalize_magnetization(m: torch.Tensor) -> torch.Tensor:
    return m / m.square().sum(dim=1, keepdim=True).clamp_min(1.0e-12).sqrt()


class ThirdPartyX5Model(nn.Module):
    """Thin shape adapter around models imported from author repositories."""

    def __init__(self, method: str, *, initialize_pretrained: bool = False) -> None:
        super().__init__()
        self.method = method
        self.initialization: dict[str, Any] = {"policy": "trained from scratch"}
        if method == "neuralop_fno":
            repo = REPOSITORIES[method]
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            from neuralop.models import FNO

            self.core = FNO(
                n_modes=(20, 20),
                in_channels=CONDITION_CHANNELS,
                out_channels=3,
                hidden_channels=96,
                n_layers=4,
                domain_padding=8.0 / 256.0,
                positional_embedding="grid",
            )
            self.algorithm = {
                "architecture": "neuralop.models.FNO",
                "hidden_channels": 96,
                "n_layers": 4,
                "n_modes": [20, 20],
                "domain_padding": 8.0 / 256.0,
                "network_calls_per_prediction": 1,
            }
        elif method in {"acdm", "pde_refiner"}:
            repo_src = REPOSITORIES[method] / "src"
            if str(repo_src) not in sys.path:
                sys.path.insert(0, str(repo_src))
            from turbpred.params import DataParams, ModelParamsDecoder

            # The authors' published 2-D Inc/Tra configurations use width 128.
            p_d = DataParams(dataSize=[128, 128], dimension=3)
            if method == "acdm":
                from turbpred.model_diffusion import DiffusionModel

                p_md = ModelParamsDecoder(
                    arch="direct-ddpm+Prev",
                    diffSteps=20,
                    diffSchedule="linear",
                    diffCondIntegration="noisy",
                )
                self.core = DiffusionModel(
                    p_d,
                    p_md,
                    dimension=2,
                    condChannels=CONDITION_CHANNELS,
                )
                self.algorithm = {
                    "architecture": "turbpred.model_diffusion.DiffusionModel",
                    "base_width": 128,
                    "diffusion_steps": 20,
                    "diffusion_schedule": "linear",
                    "conditioning_integration": "noisy",
                    "sampling": "DDPM",
                    "network_calls_per_prediction": 20,
                }
            else:
                from turbpred.model_refiner import PDERefiner

                p_md = ModelParamsDecoder(
                    arch="refiner",
                    diffSteps=4,
                    refinerStd=1.0e-6,
                )
                self.core = PDERefiner(p_d, p_md, condChannels=CONDITION_CHANNELS)
                self.algorithm = {
                    "architecture": "turbpred.model_refiner.PDERefiner",
                    "base_width": 128,
                    "refinement_steps": 4,
                    "minimum_noise_std": 1.0e-6,
                    # The released implementation makes one direct call followed
                    # by four refinement calls.
                    "network_calls_per_prediction": 5,
                }
        elif method in MODEL_ZOO_METHODS:
            from external_baselines.x5_model_zoo import build_author_adapter

            self.core = build_author_adapter(
                method,
                CONDITION_CHANNELS,
                initialize_pretrained=initialize_pretrained,
            )
            self.algorithm = self.core.algorithm
            self.initialization = self.core.initialization
        else:
            raise ValueError(f"unknown method: {method}")

    def training_loss(self, condition: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.method in DIRECT_METHODS:
            prediction = self.core(condition)
            data_loss = F.mse_loss(prediction, target)
            unit_loss = (prediction.square().sum(dim=1).sqrt() - 1.0).square().mean()
            return data_loss + 1.0e-3 * unit_loss

        prediction, objective = self.core(condition[:, None], target[:, None])
        # This is the loss used by the ACDM authors' released trainer for both
        # diffusion noise prediction and PDE-Refiner objectives.
        return F.smooth_l1_loss(prediction, objective)

    def predict(self, condition: torch.Tensor) -> torch.Tensor:
        if self.method in DIRECT_METHODS:
            prediction = self.core(condition)
        else:
            shape_only = torch.zeros(
                (condition.shape[0], 1, 3, condition.shape[-2], condition.shape[-1]),
                device=condition.device,
                dtype=condition.dtype,
            )
            prediction = self.core(condition[:, None], shape_only)[:, 0]
        return normalize_magnetization(prediction.float())

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return torch.autocast(device_type="cuda", dtype=dtype)


def _configure_cuda() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def _dataset_config(config_path: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    cfg.setdefault("data", {}).setdefault("memmap", {})["auto_build"] = False
    cfg["data"].setdefault("zeeman_precondition", {})["enabled"] = False
    # Request all spatial fields used by the common external adapter.
    cfg.setdefault("model", {})["use_spatial_cond"] = True
    cfg["model"]["spatial_cond_fields"] = "v4_full"
    return cfg


def _loader(dataset: Any, *, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=shuffle,
    )


def _training_metadata(args: argparse.Namespace, model: ThirdPartyX5Model) -> dict[str, Any]:
    repo = REPOSITORIES[args.method]
    return {
        "schema": "x5_external_baseline_v1",
        "method": args.method,
        "label": METHOD_LABELS[args.method],
        "author_repository": str(repo),
        "author_commit": _git_commit(repo),
        "algorithm": model.algorithm,
        "initialization": model.initialization,
        "parameter_count": model.parameter_count,
        "condition_channels": CONDITION_CHANNELS,
        "condition_schema": {
            "state": ["m_x", "m_y", "m_z"],
            "spatial": [key for key, _ in SPATIAL_FIELDS],
            "scalar_broadcast": [key for key, _ in SCALAR_FIELDS],
        },
        "dataset_config": str(args.config),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "gradient_accumulation": int(args.grad_accum),
        "effective_batch_size": int(args.batch_size * args.grad_accum),
        "precision": args.precision,
        "learning_rate": float(args.lr),
        "seed": int(args.seed),
        "adapter_policy": "channel concatenation and scalar broadcasting only",
    }


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("formal external-baseline training requires CUDA")
    if args.steps <= 0 or args.batch_size <= 0 or args.grad_accum <= 0:
        raise ValueError("steps, batch size, and gradient accumulation must be positive")
    if args.method in FP32_ONLY_METHODS and args.precision != "fp32":
        raise ValueError(
            f"{args.method} uses an author FFT or low-variance normalization path "
            "that is run in fp32 for numerical compatibility"
        )
    _configure_cuda()
    seed_everything(args.seed)
    device = torch.device("cuda")
    cfg = _dataset_config(args.config)
    train_ds, _, _ = build_fixed_time_datasets(cfg, build_splits={"train"})
    if train_ds is None:
        raise RuntimeError("failed to build the x5 training split")
    loader = _loader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    iterator = iter(loader)
    model = ThirdPartyX5Model(
        args.method,
        initialize_pretrained=bool(args.pretrained and args.resume is None),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0,
    )
    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(_checkpoint_model_state(checkpoint["model"]), strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        previous_metadata = checkpoint.get("metadata", {})
        if isinstance(previous_metadata.get("initialization"), dict):
            model.initialization = previous_metadata["initialization"]

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
    peak_memory = 0

    for step in range(start_step + 1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        update_losses: list[float] = []
        for _micro_step in range(args.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = move_batch(batch, device)
            condition = build_condition_tensor(batch)
            target = batch["m_t"].float()
            with _autocast(device, args.precision):
                micro_loss = model.training_loss(condition, target)
                loss = micro_loss / args.grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at step {step}: {float(loss)}")
            loss.backward()
            update_losses.append(float(micro_loss.detach()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        interval_losses.append(statistics.fmean(update_losses))
        if device.type == "cuda":
            peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated(device)))

        if step % args.log_every == 0 or step == args.steps:
            torch.cuda.synchronize(device)
            now = time.perf_counter()
            row = {
                "step": step,
                "loss": statistics.fmean(interval_losses),
                "elapsed_s": now - started,
                "steps_per_s": len(interval_losses) / max(now - interval_started, 1.0e-9),
                "peak_gpu_memory_bytes": peak_memory,
                "device": torch.cuda.get_device_name(device),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(row, flush=True)
            interval_losses.clear()
            interval_started = now

        if step % args.save_every == 0 or step == args.steps:
            payload = {
                "schema": "x5_external_baseline_checkpoint_v1",
                "method": args.method,
                "step": step,
                "model": _checkpoint_model_state(model),
                "optimizer": optimizer.state_dict(),
                "metadata": metadata,
            }
            _atomic_checkpoint(args.output_dir / f"checkpoint_{step:07d}.pt", payload)
            _atomic_checkpoint(args.output_dir / "checkpoint_latest.pt", payload)


def _quality(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    target = normalize_magnetization(target.float())
    mse = F.mse_loss(prediction, target).item()
    dot = (prediction * target).sum(dim=1).clamp(-1.0, 1.0)
    angle_deg = torch.rad2deg(torch.acos(dot)).mean().item()
    return mse, angle_deg


@torch.no_grad()
def benchmark(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("formal external-baseline benchmark requires CUDA")
    if args.method in FP32_ONLY_METHODS and args.precision != "fp32":
        raise ValueError(
            f"{args.method} uses an author FFT or low-variance normalization path "
            "that is run in fp32 for numerical compatibility"
        )
    _configure_cuda()
    seed_everything(args.seed)
    device = torch.device("cuda")
    cfg = _dataset_config(args.config)
    _, _, test_ds = build_fixed_time_datasets(cfg, build_splits={"test"})
    if test_ds is None:
        raise RuntimeError("failed to build the x5 test split")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != args.method:
        raise ValueError(
            f"checkpoint method {checkpoint.get('method')!r} does not match {args.method!r}"
        )
    model = ThirdPartyX5Model(args.method).to(device)
    model.load_state_dict(_checkpoint_model_state(checkpoint["model"]), strict=True)
    model.eval()
    timings: dict[str, Any] = {}
    quality: dict[str, Any] = {}
    selections: dict[str, Any] = {}

    for horizon in HORIZONS_NS:
        subset = X5InitialStateHorizonDataset(
            test_ds,
            horizon,
            args.samples_per_horizon,
        )
        loader = _loader(subset, batch_size=args.batch_size, num_workers=0, shuffle=False)
        cached: list[tuple[torch.Tensor, torch.Tensor]] = []
        for batch in loader:
            batch = move_batch(batch, device)
            observed_horizons = batch["t_end_ns"].float()
            if not torch.allclose(
                observed_horizons,
                torch.full_like(observed_horizons, horizon),
                atol=subset.tolerance_ns,
                rtol=0.0,
            ):
                raise RuntimeError(
                    f"selected samples do not match {horizon:g} ns: "
                    f"{observed_horizons.detach().cpu().tolist()}"
                )
            cached.append((build_condition_tensor(batch), batch["m_t"].float()))
        if not cached:
            raise RuntimeError(f"no test samples found for {horizon:g} ns")

        for condition, _ in cached[: args.warmup_batches]:
            with _autocast(device, args.precision):
                model.predict(condition)
        torch.cuda.synchronize(device)

        runs: list[float] = []
        for repeat in range(args.repeats):
            total_s = 0.0
            total_n = 0
            for condition, _ in cached:
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                with _autocast(device, args.precision):
                    model.predict(condition)
                torch.cuda.synchronize(device)
                total_s += time.perf_counter() - start
                total_n += condition.shape[0]
            runs.append(1000.0 * total_s / total_n)
            print(
                {
                    "method": args.method,
                    "horizon_ns": horizon,
                    "repeat": repeat + 1,
                    "ms_per_sample": runs[-1],
                },
                flush=True,
            )

        mse_values: list[float] = []
        angle_values: list[float] = []
        for condition, target in cached:
            with _autocast(device, args.precision):
                prediction = model.predict(condition)
            mse, angle = _quality(prediction, target)
            mse_values.append(mse)
            angle_values.append(angle)
        key = f"{horizon:g}"
        timings[key] = {
            "mean_ms": statistics.fmean(runs),
            "std_ms": statistics.stdev(runs) if len(runs) > 1 else 0.0,
            "samples": len(subset),
            "runs": runs,
        }
        quality[key] = {
            "mse": statistics.fmean(mse_values),
            "mean_angle_deg": statistics.fmean(angle_values),
            "batches": len(mse_values),
        }
        selections[key] = subset.selection_metadata()

    payload = {
        "schema": "x5_external_baseline_benchmark_v1",
        "unit": "ms / sample",
        "method": args.method,
        "label": METHOD_LABELS[args.method],
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "split": "test",
        "batch_size": args.batch_size,
        "precision": args.precision,
        "samples_per_horizon": args.samples_per_horizon,
        "repeats": args.repeats,
        "device": torch.cuda.get_device_name(device),
        "parameter_count": model.parameter_count,
        "algorithm": model.algorithm,
        "author_repository": str(REPOSITORIES[args.method]),
        "author_commit": _git_commit(REPOSITORIES[args.method]),
        "timing_excludes": [
            "DataLoader iteration",
            "CPU-to-GPU transfer",
            "condition construction",
        ],
        "evaluation_dataset_view": {
            "t_end_ns": list(HORIZONS_NS),
            "control_segmented": False,
            "current_time_mode": "t0",
            "meaning": "full pulse/relax trajectory from the x5 initial state",
        },
        "selection": selections,
        "horizons": timings,
        "quality": quality,
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def inspect_model(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    model = ThirdPartyX5Model(args.method, initialize_pretrained=args.pretrained)
    payload = {
        "method": args.method,
        "label": METHOD_LABELS[args.method],
        "parameter_count": model.parameter_count,
        "condition_channels": CONDITION_CHANNELS,
        "algorithm": model.algorithm,
        "initialization": model.initialization,
        "author_repository": str(REPOSITORIES[args.method]),
        "author_commit": _git_commit(REPOSITORIES[args.method]),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", choices=tuple(REPOSITORIES), required=True)
    parser.add_argument("--seed", type=int, default=78)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser("inspect", help="Instantiate a third-party model on CPU.")
    _common(inspect_parser)
    inspect_parser.add_argument("--pretrained", action="store_true")
    inspect_parser.set_defaults(func=inspect_model)

    train_parser = commands.add_parser("train", help="Train on the SKX x5 training split.")
    _common(train_parser)
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--steps", type=int, default=50_000)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--grad-accum", type=int, default=8)
    train_parser.add_argument("--num-workers", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=1.0e-4)
    train_parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    train_parser.add_argument("--log-every", type=int, default=50)
    train_parser.add_argument("--save-every", type=int, default=2500)
    train_parser.add_argument("--resume", type=Path, default=None)
    train_parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Initialize from the author-released checkpoint when one is available.",
    )
    train_parser.set_defaults(func=train)

    bench_parser = commands.add_parser("benchmark", help="Benchmark a trained checkpoint.")
    _common(bench_parser)
    bench_parser.add_argument("--config", type=Path, required=True)
    bench_parser.add_argument("--checkpoint", type=Path, required=True)
    bench_parser.add_argument("--output", type=Path, required=True)
    bench_parser.add_argument("--samples-per-horizon", type=int, default=128)
    bench_parser.add_argument("--batch-size", type=int, default=32)
    bench_parser.add_argument("--warmup-batches", type=int, default=2)
    bench_parser.add_argument("--repeats", type=int, default=10)
    bench_parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    bench_parser.set_defaults(func=benchmark)
    return parser


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
