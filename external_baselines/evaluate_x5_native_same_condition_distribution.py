#!/usr/bin/env python3
"""Score deterministic author-native x5 baselines on five-repeat distributions.

Each physical base group contains five MuMax trajectories with matching material,
geometry, and control parameters.  A deterministic model is evaluated once from
each matching repeat anchor, yielding its honest five-point empirical output
distribution.  No artificial noise or ground-truth candidate selection is used.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_ROOT = Path(
    str(Path(__file__).resolve().parents[1] / "third_party/distribution_score")
)
for path in (PROJECT_ROOT, DIST_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from distribution_score.distance import (  # noqa: E402
    FORMAL_BLOCKS,
    FORMAL_SHIFT_RADIUS_PX,
)
from distribution_score.same_condition import run as run_same_condition  # noqa: E402
from external_baselines.x5_author_native import (  # noqa: E402
    LATENT_METHODS,
    LEAD_TIME_METHODS,
    AuthorNativeX5Model,
    FP32_ONLY_METHODS,
    METHODS,
    REPOSITORIES,
    _autocast,
    _causal_history_input,
    _exact_control_substeps,
    _formal_prediction,
    _git_commit,
    _native_substeps,
    _validate_precision,
    build_native_condition_tensor,
)
from external_baselines.x5_external_baseline import (  # noqa: E402
    _checkpoint_model_state,
    _configure_cuda,
    _dataset_config,
)
from graph.x5_author_native_rollout_dataset import (  # noqa: E402
    attach_native_schedule_fields,
)
from scripts.evaluate_skx_x5_same_condition_distribution import (  # noqa: E402
    DEFAULT_REPEAT_DATASET,
    SEGMENT_LABELS,
    _aggregate,
    _complete_test_groups,
    _condition_supports_duration,
    _prepare_condition,
    _prepare_multisegment_rollout_condition,
    _prepare_rollout_condition,
    _sha256,
    _write_json,
)
from skyrmion_cfm.config import seed_everything  # noqa: E402
from skyrmion_cfm.data.fixed_time import build_fixed_time_datasets  # noqa: E402
from skyrmion_cfm.train import move_batch  # noqa: E402
from scripts.x5_distribution_metrics import (  # noqa: E402
    aggregate_clustered_mean,
    angular_energy_distance_from_files,
)
from scripts.x5_anchor_jitter import apply_anchor_jitter  # noqa: E402


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("no condition scores were produced")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _clone_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in sample.items()
    }


def _one_native_condition(
    sample: dict[str, Any], device: torch.device
) -> torch.Tensor:
    batch = move_batch(default_collate([sample]), device)
    return build_native_condition_tensor(batch)


def _zero_future_native_schedule(sample: dict[str, Any]) -> dict[str, Any]:
    """Remove the original pulse descriptor from an extended relax query."""
    # ``attach_native_schedule_fields`` normally
    # describes the complete source trajectory.  Here the anchor is already
    # post-drive, so the requested future interval contains no pulse at all.
    for axis in ("x", "y", "z"):
        key = f"planned_j_{axis}_field"
        if key in sample:
            sample[key] = torch.zeros_like(sample[key])
    start_ns = float(sample["frame_init_time_ns"])
    sample["pulse_start_ns"] = torch.tensor(start_ns, dtype=torch.float32)
    sample["pulse_end_ns"] = torch.tensor(start_ns, dtype=torch.float32)
    return sample


@torch.inference_mode()
def _constant_control_rollout_prediction(
    model: AuthorNativeX5Model,
    dataset: Any,
    record_index: int,
    sample: dict[str, Any],
    *,
    device: torch.device,
    precision: str,
) -> torch.Tensor:
    """Execute the released method on the matched 5-ns continuation.

    [5NS-ROLLOUT FIX 2026-08-29] Direct endpoint operators receive one 5-ns
    condition. Recurrent methods receive twenty causal 0.25-ns zero-current
    steps, and LE-PDE receives one encode plus twenty latent advances. This is
    the same execution graph used by the paper's existing 5-ns timing table.
    """
    direct = _zero_future_native_schedule(
        attach_native_schedule_fields(
            dataset, int(record_index), _clone_sample(sample)
        )
    )
    if model.method in LEAD_TIME_METHODS:
        condition = _one_native_condition(direct, device)
        with _autocast(device, precision):
            return model.predict(condition)

    native_steps = 20
    hop = _clone_sample(direct)
    start_ns = float(direct["frame_init_time_ns"])
    hop["t_end_ns"] = torch.tensor(0.25, dtype=torch.float32)
    hop["t_end_s"] = torch.tensor(0.25e-9, dtype=torch.float32)
    hop["t_end_index"] = torch.tensor(
        dataset._t_end_bucket_index(0.25), dtype=torch.long
    )
    hop["drive_time_s"] = torch.tensor(0.0, dtype=torch.float32)
    hop["relax_time_s"] = torch.tensor(0.25e-9, dtype=torch.float32)
    hop["drive_fraction"] = torch.tensor(0.0, dtype=torch.float32)

    if model.method in LATENT_METHODS:
        condition = _one_native_condition(hop, device)
        with _autocast(device, precision):
            return model.latent_rollout(condition, native_steps)

    prediction: torch.Tensor | None = None
    history: list[torch.Tensor] = []
    for step in range(native_steps):
        step_sample = _clone_sample(hop)
        step_sample["frame_init_time_ns"] = torch.tensor(
            start_ns + 0.25 * step, dtype=torch.float32
        )
        step_sample["frame_target_time_ns"] = torch.tensor(
            start_ns + 0.25 * (step + 1), dtype=torch.float32
        )
        condition = _one_native_condition(step_sample, device)
        if prediction is not None:
            condition[:, :3] = prediction.to(dtype=condition.dtype)
        model_input = _causal_history_input(
            history, condition, model.history_steps
        )
        with _autocast(device, precision):
            prediction = model.predict(model_input)
    if prediction is None:
        raise RuntimeError("5-ns native rollout produced no prediction")
    return prediction


@torch.inference_mode()
# Apply the same predicted-state
# handoff and exact control switch to every learned baseline.
def _multisegment_rollout_prediction(
    model: AuthorNativeX5Model,
    dataset: Any,
    record_index: int,
    path: list[dict[str, Any]],
    *,
    device: torch.device,
    precision: str,
    seed: int,
) -> torch.Tensor:
    """Run one baseline through the same physical segment path as FLARE.

    Lead-time endpoint operators receive one query per physical segment.
    Fixed-step autoregressive methods keep both their predicted state and
    causal history across the control boundary.  LE-PDE retains its released
    encode-once temporal interface and receives the complete future pulse
    schedule at the first native step.
    """
    if len(path) < 2:
        raise RuntimeError("multisegment baseline rollout needs at least two segments")

    if model.method in LEAD_TIME_METHODS:
        prediction: torch.Tensor | None = None
        for segment_position, original in enumerate(path):
            query = attach_native_schedule_fields(
                dataset, int(record_index), _clone_sample(original)
            )
            if prediction is not None:
                predicted_start = prediction[0].detach().float().cpu()
                query["m_init"] = predicted_start
                query["m0"] = predicted_start
            prediction = _formal_prediction(
                model,
                dataset,
                {"record_index": int(record_index)},
                query,
                device=device,
                precision=precision,
                seed=seed + 100_003 * segment_position,
            ).float()
        if prediction is None:
            raise RuntimeError("lead-time multisegment rollout produced no prediction")
        return prediction

    substeps_by_segment = [
        _exact_control_substeps(
            dataset,
            int(record_index),
            _clone_sample(original),
        )
        for original in path
    ]
    if not substeps_by_segment or any(not steps for steps in substeps_by_segment):
        raise RuntimeError("native multisegment rollout produced no substeps")

    if model.method in LATENT_METHODS:
        prediction: torch.Tensor | None = None
        for segment_steps in substeps_by_segment:
            condition = _one_native_condition(segment_steps[0], device)
            if prediction is not None:
                condition[:, :3] = prediction.to(dtype=condition.dtype)
            with _autocast(device, precision):
                prediction = model.latent_rollout(
                    condition, len(segment_steps)
                ).float()
        if prediction is None:
            raise RuntimeError("latent multisegment rollout produced no prediction")
        return prediction

    prediction = None
    history: list[torch.Tensor] = []
    for segment_steps in substeps_by_segment:
        for hop in segment_steps:
            condition = _one_native_condition(hop, device)
            if prediction is not None:
                condition[:, :3] = prediction.to(dtype=condition.dtype)
            model_input = _causal_history_input(
                history, condition, model.history_steps
            )
            with _autocast(device, precision):
                prediction = model.predict(model_input).float()
    if prediction is None:
        raise RuntimeError("autoregressive multisegment rollout produced no prediction")
    return prediction


@torch.inference_mode()
def _unique_predictions(
    prepared: dict[str, Any],
    dataset: Any,
    record_indices: list[int],
    model: AuthorNativeX5Model,
    *,
    device: torch.device,
    precision: str,
    seed: int,
    draws_per_anchor: int = 1,
    anchor_jitter_rms_deg: float = 0.0,
    anchor_jitter_correlation_px: float = 4.0,
    jitter_reports: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    is_multisegment = (
        prepared["metadata"].get("reference_mode")
        == "exact_control_multisegment_rollout_endpoint"
    )
    samples = prepared.get("segment_samples") if is_multisegment else prepared["samples"]
    if len(samples) != len(record_indices):
        raise RuntimeError("repeat samples and record indices are misaligned")
    predictions: list[np.ndarray] = []
    for repeat_position, (original_sample, record_index) in enumerate(
        zip(samples, record_indices)
    ):
        for draw_index in range(draws_per_anchor):
            draw_seed = (
                seed
                + repeat_position * 1_000_003
                + draw_index * 10_000_019
            )
            if is_multisegment:
                sample = [_clone_sample(item) for item in original_sample]
                jitter_sample = sample[0]
            else:
                sample = _clone_sample(original_sample)
                jitter_sample = sample
            if anchor_jitter_rms_deg > 0.0:
                jittered, report = apply_anchor_jitter(
                    jitter_sample["m_init"][None].to(device),
                    rms_degrees=anchor_jitter_rms_deg,
                    correlation_px=anchor_jitter_correlation_px,
                    seeds=[draw_seed + 701],
                )
                jittered_cpu = jittered[0].float().cpu()
                jitter_sample["m_init"] = jittered_cpu
                if "m0" in jitter_sample:
                    jitter_sample["m0"] = jittered_cpu.clone()
                if jitter_reports is not None:
                    jitter_reports.extend(report)
            if is_multisegment:
                prediction = _multisegment_rollout_prediction(
                    model,
                    dataset,
                    int(record_index),
                    sample,
                    device=device,
                    precision=precision,
                    seed=draw_seed,
                ).float()
                mask_sample = sample[-1]
            elif prepared["metadata"].get("reference_mode") == "mumax3_rollout_extension":
                prediction = _constant_control_rollout_prediction(
                    model,
                    dataset,
                    int(record_index),
                    sample,
                    device=device,
                    precision=precision,
                ).float()
                mask_sample = sample
            else:
                sample = attach_native_schedule_fields(dataset, int(record_index), sample)
                prediction = _formal_prediction(
                    model,
                    dataset,
                    {"record_index": int(record_index)},
                    sample,
                    device=device,
                    precision=precision,
                    seed=draw_seed,
                ).float()
                mask_sample = sample
            observed = mask_sample.get(
                "m_observed_init", mask_sample["m_init"]
            ).float()
            mask = observed.square().sum(dim=0, keepdim=True).sqrt().gt(0.5)
            prediction = F.normalize(prediction, dim=1, eps=1.0e-8)
            prediction = prediction * mask[None].to(device=prediction.device)
            predictions.append(prediction[0].cpu().numpy().astype(np.float16))
    return np.stack(predictions)


def _generate_condition(
    prepared: dict[str, Any],
    dataset: Any,
    record_indices: list[int],
    model: AuthorNativeX5Model,
    *,
    method: str,
    checkpoint: Path,
    checkpoint_digest: str,
    checkpoint_step: int,
    device: torch.device,
    precision: str,
    seed: int,
    draws_per_anchor: int,
    anchor_jitter_rms_deg: float,
    anchor_jitter_correlation_px: float,
    force: bool,
) -> Path:
    model_dir = Path(prepared["condition_dir"]) / "model_samples"
    model_path = model_dir / "model_samples_f16.npy"
    manifest_path = model_dir / "manifest.json"
    expected_shape = [len(record_indices) * draws_per_anchor, 3, 256, 256]
    if model_path.is_file() and manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") == "complete"
            and manifest.get("method") == method
            and manifest.get("checkpoint_sha256") == checkpoint_digest
            and manifest.get("shape") == expected_shape
            and manifest.get("requested_horizon_ns")
            == prepared["metadata"].get("requested_horizon_ns")
            and manifest.get("reference_mode")
            == prepared["metadata"].get("reference_mode", "saved_segment_endpoint")
            and int(manifest.get("draws_per_anchor", 1)) == draws_per_anchor
            and float(manifest.get("anchor_jitter_rms_deg", 0.0))
            == float(anchor_jitter_rms_deg)
            and (
                anchor_jitter_rms_deg == 0.0
                or float(manifest.get("anchor_jitter_correlation_px", 0.0))
                == float(anchor_jitter_correlation_px)
            )
        ):
            return model_path

    model_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    jitter_reports: list[dict[str, Any]] = []
    values = _unique_predictions(
        prepared,
        dataset,
        record_indices,
        model,
        device=device,
        precision=precision,
        seed=seed,
        draws_per_anchor=draws_per_anchor,
        anchor_jitter_rms_deg=anchor_jitter_rms_deg,
        anchor_jitter_correlation_px=anchor_jitter_correlation_px,
        jitter_reports=jitter_reports,
    )
    if list(values.shape) != expected_shape:
        raise RuntimeError(f"wrong deterministic sample shape: {values.shape}")
    temporary = model_path.with_suffix(model_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    temporary.replace(model_path)
    assignment = np.repeat(
        np.arange(len(record_indices), dtype=np.int16), draws_per_anchor
    )
    np.save(model_dir / "anchor_repeat_index.npy", assignment)
    _write_json(
        manifest_path,
        {
            "status": "complete",
            "method": method,
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_digest,
            "checkpoint_step": checkpoint_step,
            "condition_id": prepared["metadata"]["condition_id"],
            "requested_horizon_ns": prepared["metadata"].get(
                "requested_horizon_ns"
            ),
            "reference_mode": prepared["metadata"].get(
                "reference_mode", "saved_segment_endpoint"
            ),
            "realized_horizon_ns": prepared["metadata"]["horizon_ns"],
            "distribution_semantics": (
                "independent tangent-jitter forecasts for each matching MuMax anchor"
                if anchor_jitter_rms_deg > 0.0
                else "one deterministic prediction for each matching MuMax repeat anchor"
            ),
            "draws_per_anchor": draws_per_anchor,
            "anchor_repeat_index_file": model_dir / "anchor_repeat_index.npy",
            "anchor_jitter_rms_deg": anchor_jitter_rms_deg,
            "anchor_jitter_correlation_px": anchor_jitter_correlation_px,
            "anchor_jitter_application": (
                "one tangent-plane perturbation of the exact drive-start anchor only"
                if anchor_jitter_rms_deg > 0.0
                else "none"
            ),
            "anchor_jitter_reports": jitter_reports,
            "unique_model_predictions": len(values),
            "shape": expected_shape,
            "dtype": "float16",
            "elapsed_seconds": time.time() - started,
        },
    )
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--segments", type=int, nargs="+", default=(1, 2))
    # Permit a quality evaluation at the
    # exact same legal within-segment horizon used by the timing benchmark.
    parser.add_argument(
        "--duration-ns",
        type=float,
        default=None,
        help=(
            "Override each segment endpoint by this duration from its exact "
            "boundary; conditions without a legal target are skipped."
        ),
    )
    parser.add_argument(
        "--rollout-target-root",
        type=Path,
        default=None,
        help=(
            "[5NS-ROLLOUT FIX 2026-08-29] Score the native 5-ns execution "
            "against MuMax3 continuations stored under TARGET_ROOT/targets."
        ),
    )
    parser.add_argument(
        "--multisegment-rollout",
        action="store_true",
        help=(
            "Score the complete drive->post-relax path using exact protocol "
            "control boundaries. Direct endpoint methods hand off there; "
            "recurrent methods preserve their native causal history."
        ),
    )
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--bootstrap", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=208_160_700)
    parser.add_argument("--draws-per-anchor", type=int, default=1)
    parser.add_argument("--anchor-jitter-rms-deg", type=float, default=0.0)
    parser.add_argument("--anchor-jitter-correlation-px", type=float, default=4.0)
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate samples (including repeated jitter draws) without macro scoring.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    _validate_precision(args.method, args.precision)
    if args.duration_ns is not None and args.duration_ns <= 0.0:
        raise ValueError("duration-ns must be positive")
    if args.draws_per_anchor <= 0:
        raise ValueError("draws-per-anchor must be positive")
    if args.anchor_jitter_rms_deg < 0.0 or args.anchor_jitter_correlation_px < 0.0:
        raise ValueError("anchor jitter parameters must be non-negative")
    if args.draws_per_anchor != 1 and not args.generate_only:
        raise ValueError("multiple draws per anchor require --generate-only")
    selected_task_modes = sum(
        (
            args.duration_ns is not None,
            args.rollout_target_root is not None,
            bool(args.multisegment_rollout),
        )
    )
    if selected_task_modes > 1:
        raise ValueError(
            "--duration-ns, --rollout-target-root, and --multisegment-rollout "
            "are mutually exclusive"
        )
    if args.rollout_target_root is not None:
        args.rollout_target_root = args.rollout_target_root.resolve()
        if not (args.rollout_target_root / "manifest.json").is_file():
            raise FileNotFoundError(args.rollout_target_root / "manifest.json")
        args.segments = (2,)
    if args.multisegment_rollout:
        args.segments = (-1,)
    if not DIST_ROOT.is_dir():
        raise FileNotFoundError(DIST_ROOT)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _configure_cuda()

    cfg = _dataset_config(args.config)
    cfg.setdefault("data", {}).setdefault("augment", {})["enabled"] = False
    _, _, test_dataset = build_fixed_time_datasets(cfg, build_splits={"test"})
    if test_dataset is None:
        raise RuntimeError("failed to build x5 test split")
    groups = _complete_test_groups(
        test_dataset,
        repeat_dataset=DEFAULT_REPEAT_DATASET,
        repeats_per_group=5,
    )
    if args.max_groups is not None:
        if args.max_groups <= 0:
            raise ValueError("max-groups must be positive")
        groups = dict(list(groups.items())[: args.max_groups])

    checkpoint_digest = _sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != args.method:
        raise ValueError("checkpoint method mismatch")
    checkpoint_step = int(checkpoint["step"])
    model = AuthorNativeX5Model(args.method).to(device)
    model.load_state_dict(_checkpoint_model_state(checkpoint["model"]), strict=True)
    model.eval()
    del checkpoint

    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped_conditions: list[dict[str, Any]] = []
    started = time.time()
    total = len(groups) * len(args.segments)
    position = 0
    for base, record_indices in groups.items():
        for segment_index in args.segments:
            position += 1
            condition_id = (
                f"base{base}_exact_control_drive_to_post_relax"
                if args.multisegment_rollout
                else (
                    f"base{base}_segment{segment_index:03d}_"
                    f"{SEGMENT_LABELS.get(segment_index, 'condition')}"
                )
            )
            if args.duration_ns is not None and not _condition_supports_duration(
                test_dataset,
                record_indices=record_indices,
                segment_index=segment_index,
                duration_ns=float(args.duration_ns),
            ):
                skipped_conditions.append(
                    {
                        "condition_id": condition_id,
                        "base_id": base,
                        "control_segment_index": segment_index,
                        "requested_horizon_ns": float(args.duration_ns),
                        "reason": "requested target lies outside this control segment",
                    }
                )
                print(
                    json.dumps(
                        {
                            "position": position,
                            "total": total,
                            "condition_id": condition_id,
                            "status": "unsupported_horizon",
                        }
                    ),
                    flush=True,
                )
                continue
            condition_dir = args.output_root / "conditions" / condition_id
            condition_dir.mkdir(parents=True, exist_ok=True)
            if args.multisegment_rollout:
                prepared = _prepare_multisegment_rollout_condition(
                    test_dataset,
                    base=base,
                    record_indices=record_indices,
                    condition_dir=condition_dir,
                )
            elif args.rollout_target_root is not None:
                prepared = _prepare_rollout_condition(
                    test_dataset,
                    base=base,
                    record_indices=record_indices,
                    condition_dir=condition_dir,
                    target_root=args.rollout_target_root,
                    horizon_ns=5.0,
                )
            else:
                prepared = _prepare_condition(
                    test_dataset,
                    base=base,
                    record_indices=record_indices,
                    segment_index=segment_index,
                    condition_dir=condition_dir,
                    duration_ns=args.duration_ns,
                )
            model_path = _generate_condition(
                prepared,
                test_dataset,
                record_indices,
                model,
                method=args.method,
                checkpoint=args.checkpoint,
                checkpoint_digest=checkpoint_digest,
                checkpoint_step=checkpoint_step,
                device=device,
                precision=args.precision,
                seed=args.seed + int(base) * 100 + segment_index * 10_000,
                draws_per_anchor=args.draws_per_anchor,
                anchor_jitter_rms_deg=args.anchor_jitter_rms_deg,
                anchor_jitter_correlation_px=args.anchor_jitter_correlation_px,
                force=args.force,
            )
            if args.generate_only:
                print(
                    json.dumps(
                        {
                            "position": position,
                            "total": total,
                            "condition_id": condition_id,
                            "status": "samples_complete",
                        }
                    ),
                    flush=True,
                )
                continue
            score_dir = condition_dir / "same_condition_score_4x4_shift16"
            result = run_same_condition(
                argparse.Namespace(
                    condition_dir=condition_dir,
                    output_dir=score_dir,
                    model_samples=model_path,
                    num_model=len(record_indices) * args.draws_per_anchor,
                    num_mumax=len(record_indices),
                    blocks=FORMAL_BLOCKS,
                    shift_radius=FORMAL_SHIFT_RADIUS_PX,
                    reference_chunk=len(record_indices),
                    bootstrap=args.bootstrap,
                    seed=args.seed + int(base) * 1_000 + segment_index,
                    device=args.device,
                    progress=False,
                    skip_auxiliary_texture=True,
                )
            )
            primary = result["primary_patch_shift"]
            energy = angular_energy_distance_from_files(
                model_path,
                score_dir / "mumax_targets_f16.npy",
                score_dir / "geometry_mask.npy",
                num_model=len(record_indices) * args.draws_per_anchor,
                device=args.device,
            )
            rows.append(
                {
                    "condition_id": condition_id,
                    "base_id": base,
                    "control_segment_index": segment_index,
                    "segment_role": prepared["metadata"].get(
                        "segment_role", SEGMENT_LABELS.get(segment_index, "condition")
                    ),
                    "absolute_start_ns": prepared["metadata"]["absolute_start_ns"],
                    "absolute_end_ns": prepared["metadata"]["absolute_end_ns"],
                    "horizon_ns": prepared["metadata"]["horizon_ns"],
                    "symmetric_ratio_score": primary["symmetric_ratio_score"],
                    "symmetric_log_ratio_mae": primary["symmetric_log_ratio_mae"],
                    "kernel_sigma": primary["sigma"],
                    "score_bootstrap_low": result[
                        "primary_symmetric_ratio_score_bootstrap_95ci"
                    ][0],
                    "score_bootstrap_high": result[
                        "primary_symmetric_ratio_score_bootstrap_95ci"
                    ][1],
                    **energy,
                }
            )
            print(
                json.dumps(
                    {
                        "position": position,
                        "total": total,
                        "condition_id": condition_id,
                        "score": primary["symmetric_ratio_score"],
                    }
                ),
                flush=True,
            )
            torch.cuda.empty_cache()

    summary_dir = args.output_root / "summary"
    _write_json(summary_dir / "skipped_conditions.json", skipped_conditions)
    if args.generate_only:
        summary = {
            "status": "complete",
            "mode": "exact_anchor_sample_generation_only",
            "method": args.method,
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": checkpoint_digest,
            "checkpoint_step": checkpoint_step,
            "reference_mode": (
                "exact_control_multisegment_rollout_endpoint"
                if args.multisegment_rollout
                else "saved_segment_endpoint"
            ),
            "conditions": total,
            "mumax_repeats_per_condition": 5,
            "draws_per_anchor": args.draws_per_anchor,
            "model_predictions_per_condition": 5 * args.draws_per_anchor,
            "anchor_jitter_rms_deg": args.anchor_jitter_rms_deg,
            "anchor_jitter_correlation_px": args.anchor_jitter_correlation_px,
            "elapsed_seconds": time.time() - started,
        }
        _write_json(summary_dir / "run_summary.json", summary)
        print(json.dumps(summary, indent=2, default=str), flush=True)
        return
    _atomic_csv(summary_dir / "condition_scores.csv", rows)
    algorithm_version = json.loads(
        (DIST_ROOT / "ALGORITHM_VERSION.json").read_text(encoding="utf-8")
    )
    summary = {
        "status": "complete",
        "method": args.method,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_step": checkpoint_step,
        "author_repository": REPOSITORIES[args.method],
        "author_commit": _git_commit(REPOSITORIES[args.method]),
        "implementation": DIST_ROOT,
        "algorithm_version": algorithm_version,
        "evaluation_split": "metadata test split",
        "complete_x5_base_groups": len(groups),
        "segments": list(args.segments),
        "conditions": len(rows),
        "candidate_conditions": total,
        "skipped_conditions": len(skipped_conditions),
        "requested_horizon_ns": (
            5.0 if args.rollout_target_root is not None else args.duration_ns
        ),
        "rollout_target_root": args.rollout_target_root,
        "multisegment_rollout": args.multisegment_rollout,
        "reference_mode": (
            "exact_control_multisegment_rollout_endpoint"
            if args.multisegment_rollout
            else (
                "mumax3_rollout_extension"
                if args.rollout_target_root is not None
                else "saved_segment_endpoint"
            )
        ),
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
        "mumax_repeats_per_condition": 5,
        "model_predictions_per_condition": 5 * args.draws_per_anchor,
        "draws_per_anchor": args.draws_per_anchor,
        "anchor_jitter_rms_deg": args.anchor_jitter_rms_deg,
        "anchor_jitter_correlation_px": args.anchor_jitter_correlation_px,
        "distribution_semantics": (
            "deterministic empirical distribution over five matching repeat anchors"
        ),
        "score": _aggregate(rows, iterations=args.bootstrap, seed=args.seed + 999),
        "angular_energy_distance": aggregate_clustered_mean(
            rows,
            "angular_energy_distance_deg",
            iterations=args.bootstrap,
            seed=args.seed + 1_999,
        ),
        "angular_energy_distance_model_samples_per_condition": 5,
        "paired_angular_error": aggregate_clustered_mean(
            rows,
            "paired_model_mumax_mean_deg",
            iterations=args.bootstrap,
            seed=args.seed + 2_999,
        ),
        "elapsed_seconds": time.time() - started,
    }
    _write_json(summary_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str), flush=True)

    del model, test_dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
