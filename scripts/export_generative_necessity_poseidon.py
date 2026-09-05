#!/usr/bin/env python3
"""Export fresh FLARE-30 draws and audited Poseidon-T outputs.

This is the paper-facing Generative Necessity inference entry point.  It
deliberately does not read the older 128-draw cache: each FLARE outcome is a
fresh ODE-10 call from an explicit per-draw seed.  It also verifies the
released compact checkpoints tensor-by-tensor against the original Table-1
checkpoint files available on the GPU evaluation node.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from external_baselines.evaluate_x5_native_same_condition_distribution import (  # noqa: E402
    _clone_sample,
    _unique_predictions,
)
from external_baselines.x5_author_native import (  # noqa: E402
    AuthorNativeX5Model,
    NATIVE_EXTRA_SCALAR_FIELDS,
    NATIVE_EXTRA_SPATIAL_FIELDS,
    _formal_prediction,
    build_native_condition_tensor,
)
from external_baselines.x5_external_baseline import (  # noqa: E402
    SCALAR_FIELDS,
    SPATIAL_FIELDS,
    _checkpoint_model_state,
    _configure_cuda,
    _dataset_config,
)
from graph.x5_author_native_rollout_dataset import (  # noqa: E402
    attach_native_schedule_fields,
)
from scripts.evaluate_skx_x5_same_condition_distribution import (  # noqa: E402
    _complete_test_groups,
)
from scripts.analyze_x5_distribution_sensitivity_observables import (  # noqa: E402
    _observables as _paper_observables,
)
from skyrmion_cfm.config import load_config, seed_everything  # noqa: E402
from skyrmion_cfm.data.fixed_time import (  # noqa: E402
    build_fixed_time_datasets,
    collate_fixed_time_conditions,
)
from skyrmion_cfm.data.ovf import read_ovf  # noqa: E402
from skyrmion_cfm.data.stats import load_training_stats  # noqa: E402
from skyrmion_cfm.eval.conditional_algorithm_diagnostics import (  # noqa: E402
    _autocast as _flare_autocast,
    _build_sampler,
    _load_state_verified,
)
from skyrmion_cfm.models import build_model  # noqa: E402
from skyrmion_cfm.train import move_batch  # noqa: E402


DEFAULT_SELECTION = (
    PROJECT / "outputs/paper_artifacts/generative_necessity/selection_manifest.json"
)
DEFAULT_OUTPUT = PROJECT / "outputs/paper_artifacts/generative_necessity"
DEFAULT_CONFIG = PROJECT / "configs/data/x30_evaluation.yaml"
DEFAULT_POSEIDON_CHECKPOINT = (
    Path(os.environ.get("FLARE_CHECKPOINT_ROOT", str(PROJECT.parent / "FLARE_checkpoints"))) / "baselines/poseidon_t.pt"
)
DEFAULT_FLARE_CHECKPOINT = (
    Path(os.environ.get("FLARE_CHECKPOINT_ROOT", str(PROJECT.parent / "FLARE_checkpoints"))) / "flare/core_seed78.pt"
)
DEFAULT_STATS = (
    PROJECT / "configs/stats/dataset_stats_both.json"
)
TABLE1_POSEIDON_CHECKPOINT = Path(os.environ.get("FLARE_CHECKPOINT_ROOT", str(PROJECT.parent / "FLARE_checkpoints"))) / "original/poseidon_t.pt"
TABLE1_FLARE_CHECKPOINT = Path(os.environ.get("FLARE_CHECKPOINT_ROOT", str(PROJECT.parent / "FLARE_checkpoints"))) / "original/core_seed78.pt"
TABLE1_POSEIDON_SHA256 = "e2f00c0f19e867d55f5f157551e215819b931467560d92c0985d02136ada836a"
TABLE1_FLARE_SHA256 = "ab658f5750b3208c69b0fb5a681ef77d59441294cd60c37f545f85ee1b7128ac"
REPEAT_DATASET = "skx_bt_165base_x30_sharedrelax_20260813"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_sha256(function: Any) -> str:
    source = inspect.getsource(function).encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def _canonical_state(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    result: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        canonical = key[len(prefix) :] if key.startswith(prefix) else key
        result[canonical] = value
    return result


def _compare_tensor_states(
    released: dict[str, Any], original: dict[str, Any], *, label: str
) -> dict[str, Any]:
    left = _canonical_state(released)
    right = _canonical_state(original)
    missing = sorted(set(right) - set(left))
    unexpected = sorted(set(left) - set(right))
    mismatched: list[str] = []
    maximum = 0.0
    compared_numel = 0
    for key in sorted(set(left) & set(right)):
        a = left[key]
        b = right[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            mismatched.append(key)
            continue
        compared_numel += int(a.numel())
        if not torch.equal(a, b):
            mismatched.append(key)
            if a.is_floating_point() or a.is_complex():
                maximum = max(
                    maximum,
                    float((a.detach().cpu() - b.detach().cpu()).abs().max()),
                )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"{label} compact/Table-1 tensor mismatch: missing={missing[:4]}, "
            f"unexpected={unexpected[:4]}, mismatched={mismatched[:4]}, "
            f"max_abs={maximum}"
        )
    return {
        "label": label,
        "tensor_keys": len(left),
        "tensor_numel": compared_numel,
        "missing_keys": 0,
        "unexpected_keys": 0,
        "mismatched_tensors": 0,
        "maximum_absolute_tensor_difference": maximum,
        "bitwise_tensor_equality": "passed",
    }


def _checkpoint_equivalence_audit(
    *,
    released_poseidon: Path,
    released_flare: Path,
    original_poseidon: Path,
    original_flare: Path,
) -> dict[str, Any]:
    for path in (released_poseidon, released_flare, original_poseidon, original_flare):
        if not path.is_file():
            raise FileNotFoundError(path)
    original_poseidon_digest = _sha256(original_poseidon)
    original_flare_digest = _sha256(original_flare)
    if original_poseidon_digest != TABLE1_POSEIDON_SHA256:
        raise RuntimeError("the original Poseidon file is not the Table-1 checkpoint")
    if original_flare_digest != TABLE1_FLARE_SHA256:
        raise RuntimeError("the original FLARE file is not the Table-1 checkpoint")

    compact_poseidon = torch.load(
        released_poseidon, map_location="cpu", weights_only=False, mmap=True
    )
    table_poseidon = torch.load(
        original_poseidon, map_location="cpu", weights_only=False, mmap=True
    )
    poseidon_report = _compare_tensor_states(
        _checkpoint_model_state(compact_poseidon["model"]),
        _checkpoint_model_state(table_poseidon["model"]),
        label="Poseidon-T",
    )
    del compact_poseidon, table_poseidon
    gc.collect()

    compact_flare = torch.load(
        released_flare, map_location="cpu", weights_only=False, mmap=True
    )
    table_flare = torch.load(
        original_flare, map_location="cpu", weights_only=False, mmap=True
    )
    flare_report = _compare_tensor_states(
        compact_flare["ema"], table_flare["ema"], label="FLARE Stage-1 EMA"
    )
    del compact_flare, table_flare
    gc.collect()
    return {
        "status": "passed",
        "poseidon": {
            **poseidon_report,
            "released_file": str(released_poseidon.resolve()),
            "released_file_sha256": _sha256(released_poseidon),
            "table1_file": str(original_poseidon.resolve()),
            "table1_file_sha256": original_poseidon_digest,
        },
        "flare": {
            **flare_report,
            "released_file": str(released_flare.resolve()),
            "released_file_sha256": _sha256(released_flare),
            "table1_file": str(original_flare.resolve()),
            "table1_file_sha256": original_flare_digest,
        },
    }


def _verify_exact_anchor(metadata: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    anchors = [Path(repeat["anchor_ovf"]) for repeat in metadata["repeats"]]
    if len(anchors) != 30:
        raise RuntimeError(f"expected 30 x30 anchors, found {len(anchors)}")
    missing = [str(path) for path in anchors if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} x30 anchors, first={missing[0]}")
    reference = np.asarray(read_ovf(anchors[0]), dtype=np.float32)
    maximum = 0.0
    mean_maximum = 0.0
    hashes: list[str] = []
    for path in anchors:
        value = np.asarray(read_ovf(path), dtype=np.float32)
        maximum = max(maximum, float(np.max(np.abs(value - reference))))
        mean_maximum = max(mean_maximum, float(np.mean(np.abs(value - reference))))
        hashes.append(_sha256(path))
    # The audit protocol promises one common relaxed checkpoint.  Stop instead
    # of silently relabeling merely similar anchors as exact.
    if maximum > 1.0e-7:
        raise RuntimeError(
            f"x30 anchors are not bitwise-equivalent after OVF parsing: max_abs={maximum}"
        )
    return reference, {
        "anchor_count": len(anchors),
        "parsed_max_abs_difference": maximum,
        "parsed_max_mean_abs_difference": mean_maximum,
        "unique_ovf_sha256": len(set(hashes)),
        "exact_anchor_test": "passed",
        "reference_anchor_ovf": str(anchors[0]),
    }


def _sample_for_condition(
    dataset: Any,
    record_index: int,
    segment_index: int,
    expected_horizon_ns: float,
    seed: int,
) -> dict[str, Any]:
    specs = {
        int(spec[0]): spec
        for spec in dataset._trajectory_visual_segment_specs(record_index)
    }
    if segment_index not in specs:
        raise RuntimeError(f"record {record_index} has no segment {segment_index}")
    _, start_frame, end_frame, _start_ns, _end_ns, _step_ps = specs[segment_index]
    sample = dataset._build_visual_sample_for_frames(
        record_index,
        int(start_frame),
        int(end_frame),
        np.random.default_rng(seed),
    )
    realized = float(sample["t_end_ns"])
    if not math.isclose(realized, expected_horizon_ns, rel_tol=0.0, abs_tol=2.0e-4):
        raise RuntimeError(
            f"condition horizon mismatch: dataset={realized}, audit={expected_horizon_ns}"
        )
    return _clone_sample(sample)


def _postprocess_prediction(
    prediction: torch.Tensor, sample: dict[str, Any]
) -> torch.Tensor:
    """The exact normalize-then-mask policy used by the formal evaluator."""
    observed = sample.get("m_observed_init", sample["m_init"]).float()
    mask = observed.square().sum(dim=0, keepdim=True).sqrt().gt(0.5)
    normalized = F.normalize(prediction.float(), dim=1, eps=1.0e-8)
    return normalized * mask[None].to(device=normalized.device)


def _native_condition(
    dataset: Any, record_index: int, sample: dict[str, Any]
) -> torch.Tensor:
    attached = attach_native_schedule_fields(
        dataset, int(record_index), _clone_sample(sample)
    )
    return build_native_condition_tensor(default_collate([attached])).float()


def _audit_repeat_inputs(
    dataset: Any,
    record_indices: list[int],
    *,
    segment_index: int,
    expected_horizon_ns: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify all 30 loader inputs encode one state and one physical query."""
    reference_sample: dict[str, Any] | None = None
    reference_condition: torch.Tensor | None = None
    maximum_state = 0.0
    maximum_physics = 0.0
    channel_names = (
        ["m_x", "m_y", "m_z"]
        + [key for key, _scale in SPATIAL_FIELDS]
        + [key for key, _scale in SCALAR_FIELDS]
        + [key for key, _scale in NATIVE_EXTRA_SPATIAL_FIELDS]
        + [key for key, _scale in NATIVE_EXTRA_SCALAR_FIELDS]
    )
    channel_scales = (
        [1.0, 1.0, 1.0]
        + [float(scale) for _key, scale in SPATIAL_FIELDS]
        + [float(scale) for _key, scale in SCALAR_FIELDS]
        + [float(scale) for _key, scale in NATIVE_EXTRA_SPATIAL_FIELDS]
        + [float(scale) for _key, scale in NATIVE_EXTRA_SCALAR_FIELDS]
    )
    timestamp_units = {
        "t_end_ns": "ns",
        "drive_time_s": "s",
        "relax_time_s": "s",
        "frame_init_time_ns": "ns",
        "frame_target_time_ns": "ns",
        "pulse_start_ns": "ns",
        "pulse_end_ns": "ns",
    }
    maximum_by_channel: torch.Tensor | None = None
    horizons: list[float] = []
    for position, record_index in enumerate(record_indices):
        sample = _sample_for_condition(
            dataset,
            int(record_index),
            int(segment_index),
            float(expected_horizon_ns),
            seed + position,
        )
        condition = _native_condition(dataset, int(record_index), sample)
        horizons.append(float(sample["t_end_ns"]))
        if reference_sample is None:
            reference_sample = sample
            reference_condition = condition
            continue
        assert reference_condition is not None
        channel_difference = (
            condition - reference_condition
        ).abs().amax(dim=(0, 2, 3))
        if maximum_by_channel is None:
            maximum_by_channel = channel_difference
        else:
            maximum_by_channel = torch.maximum(maximum_by_channel, channel_difference)
        maximum_state = max(
            maximum_state,
            float((condition[:, :3] - reference_condition[:, :3]).abs().max()),
        )
        maximum_physics = max(
            maximum_physics,
            float((condition[:, 3:] - reference_condition[:, 3:]).abs().max()),
        )
    if reference_sample is None or reference_condition is None:
        raise RuntimeError("empty repeat input audit")
    horizon_spread = max(horizons) - min(horizons)
    if maximum_by_channel is None:
        maximum_by_channel = torch.zeros(len(channel_names))
    differing_channels = [
        {"index": index, "name": name, "maximum_absolute_difference": float(value)}
        for index, (name, value) in enumerate(
            zip(channel_names, maximum_by_channel.tolist(), strict=True)
        )
        if value > 0.0
    ]
    timestamp_jitter: list[dict[str, Any]] = []
    maximum_non_timestamp_physics = 0.0
    maximum_timestamp_jitter_ns = 0.0
    for item in differing_channels:
        index = int(item["index"])
        name = str(item["name"])
        normalized_difference = float(item["maximum_absolute_difference"])
        if name in timestamp_units:
            physical_difference = normalized_difference * channel_scales[index]
            difference_ns = (
                physical_difference * 1.0e9
                if timestamp_units[name] == "s"
                else physical_difference
            )
            maximum_timestamp_jitter_ns = max(
                maximum_timestamp_jitter_ns, difference_ns
            )
            timestamp_jitter.append(
                {
                    **item,
                    "maximum_physical_difference_ns": difference_ns,
                }
            )
        elif index >= 3:
            maximum_non_timestamp_physics = max(
                maximum_non_timestamp_physics, normalized_difference
            )
    # MuMax3's adaptive solver may write nominally identical frames a few
    # femtoseconds apart.  Keep that numerical timestamp jitter explicit, but
    # require every state/material/field/current/schedule-value channel to be
    # identical and the realized-time jitter to remain below the same strict
    # frame-matching tolerance used by the x30 evaluator.
    if maximum_state > 1.0e-7 or maximum_non_timestamp_physics > 1.0e-7:
        raise RuntimeError(
            "the 30 repeat inputs do not share an exact state/physical condition: "
            f"state={maximum_state}, non_timestamp_physics="
            f"{maximum_non_timestamp_physics}, "
            f"differing_channels={differing_channels}"
        )
    if horizon_spread > 2.0e-4 or maximum_timestamp_jitter_ns > 2.0e-4:
        raise RuntimeError(
            "repeat target-time jitter exceeds the frame-matching tolerance: "
            f"horizon_spread={horizon_spread} ns, "
            f"channel_jitter={maximum_timestamp_jitter_ns} ns"
        )
    return reference_sample, {
        "repeat_input_count": len(record_indices),
        "native_condition_channels": int(reference_condition.shape[1]),
        "native_condition_state_max_abs_across_repeats": maximum_state,
        "native_condition_physics_max_abs_across_repeats": maximum_physics,
        "native_condition_non_timestamp_physics_max_abs_across_repeats": (
            maximum_non_timestamp_physics
        ),
        "native_condition_differing_channels": differing_channels,
        "adaptive_solver_timestamp_jitter": timestamp_jitter,
        "adaptive_solver_timestamp_jitter_max_ns": maximum_timestamp_jitter_ns,
        "target_horizon_min_ns": min(horizons),
        "target_horizon_max_ns": max(horizons),
        "target_horizon_spread_ns": horizon_spread,
        "exact_state_and_physical_condition_test": "passed",
        "same_nominal_target_horizon_test": "passed",
        "model_query_policy": (
            "one untouched reference sample is reused for Poseidon-T and all "
            "30 FLARE draws"
        ),
    }


@torch.inference_mode()
def _one_flare_draw(
    model: torch.nn.Module,
    sampler: Any,
    sample: dict[str, Any],
    *,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    batch = move_batch(default_collate([_clone_sample(sample)]), device)
    condition = collate_fixed_time_conditions(batch)
    model_input = batch["m_init"].float()
    with _flare_autocast(device):
        prediction, _ = sampler.sample(model, model_input, condition)
    prediction = _postprocess_prediction(prediction, sample)
    return prediction[0].detach().cpu().numpy().astype(np.float16)


def _fresh_flare_ensemble(
    model: torch.nn.Module,
    sampler: Any,
    sample: dict[str, Any],
    *,
    device: torch.device,
    master_seed: int,
    count: int = 30,
) -> tuple[np.ndarray, dict[str, Any]]:
    draw_seeds = [int(master_seed + index) for index in range(count)]
    draws = np.stack(
        [
            _one_flare_draw(model, sampler, sample, device=device, seed=draw_seed)
            for draw_seed in draw_seeds
        ]
    )
    replay = _one_flare_draw(
        model, sampler, sample, device=device, seed=draw_seeds[0]
    )
    replay_max_abs = float(
        np.max(np.abs(replay.astype(np.float32) - draws[0].astype(np.float32)))
    )
    if replay_max_abs != 0.0:
        raise RuntimeError(
            f"fixed-seed FLARE replay is not bitwise reproducible: {replay_max_abs}"
        )
    return draws, {
        "draw_count": count,
        "master_seed": int(master_seed),
        "draw_seeds": draw_seeds,
        "generation_calls": count,
        "batch_size": 1,
        "ode_steps": 10,
        "integrator": "Heun",
        "cached_128_draw_array_read": False,
        "first_draw_replay_max_abs_after_float16": replay_max_abs,
        "fixed_seed_replay_test": "passed",
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--poseidon-checkpoint", type=Path, default=DEFAULT_POSEIDON_CHECKPOINT
    )
    parser.add_argument(
        "--flare-checkpoint", type=Path, default=DEFAULT_FLARE_CHECKPOINT
    )
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument(
        "--table1-poseidon-checkpoint", type=Path, default=TABLE1_POSEIDON_CHECKPOINT
    )
    parser.add_argument(
        "--table1-flare-checkpoint", type=Path, default=TABLE1_FLARE_CHECKPOINT
    )
    parser.add_argument("--audit-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32",), default="fp32")
    parser.add_argument("--seed", type=int, default=208_310_700)
    parser.add_argument("--flare-seed", type=int, default=2_026_083_100)
    parser.add_argument("--skip-table1-equivalence-audit", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Poseidon-T export requires a CUDA device")
    os.environ.setdefault("SKYRMION_CFM_FUSED_OPS", "1")
    seed_everything(args.seed)
    _configure_cuda()
    device = torch.device(args.device)
    selection = _json(args.selection)
    audit_root = Path(args.audit_root or selection["audit_root"])

    cfg = _dataset_config(args.config)
    cfg.setdefault("data", {}).setdefault("augment", {})["enabled"] = False
    cfg.setdefault("data", {}).setdefault("memmap", {})["auto_build"] = False
    # The existing x30 trajectory-index pickle was written by Python 3.13,
    # whereas the pinned Poseidon/Transformers environment is Python 3.11.
    # Rebuild this metadata-only cache once with an explicit versioned name.
    cfg["data"]["trajectory_index_cache"] = str(
        args.output_dir / "x30_trajectory_index_py311.pkl"
    )
    cfg.setdefault("train", {})["num_workers"] = 0
    _, _, dataset = build_fixed_time_datasets(cfg, build_splits={"test"})
    if dataset is None:
        raise RuntimeError("failed to build the x30 held-out dataset")
    groups = _complete_test_groups(
        dataset, repeat_dataset=REPEAT_DATASET, repeats_per_group=30
    )

    checkpoint_audit = None
    if not args.skip_table1_equivalence_audit:
        checkpoint_audit = _checkpoint_equivalence_audit(
            released_poseidon=args.poseidon_checkpoint,
            released_flare=args.flare_checkpoint,
            original_poseidon=args.table1_poseidon_checkpoint,
            original_flare=args.table1_flare_checkpoint,
        )

    poseidon_digest = _sha256(args.poseidon_checkpoint)
    expected_digest = selection["poseidon_source"]["checkpoint_sha256"]
    if poseidon_digest != expected_digest:
        raise RuntimeError(
            f"Poseidon checkpoint hash mismatch: {poseidon_digest} != {expected_digest}"
        )
    poseidon_checkpoint = torch.load(
        args.poseidon_checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    if (
        poseidon_checkpoint.get("method") != "poseidon_t"
        or int(poseidon_checkpoint["step"]) != 50000
    ):
        raise RuntimeError(
            "not the released Poseidon-T final checkpoint: "
            f"method={poseidon_checkpoint.get('method')}, "
            f"step={poseidon_checkpoint.get('step')}"
        )
    poseidon_model = AuthorNativeX5Model("poseidon_t").to(device)
    poseidon_model.load_state_dict(
        _checkpoint_model_state(poseidon_checkpoint["model"]), strict=True
    )
    poseidon_model.eval()
    poseidon_step = int(poseidon_checkpoint["step"])
    del poseidon_checkpoint

    flare_digest = _sha256(args.flare_checkpoint)
    expected_flare_digest = selection["flare_source"]["released_checkpoint_sha256"]
    if flare_digest != expected_flare_digest:
        raise RuntimeError(
            f"FLARE checkpoint hash mismatch: {flare_digest} != {expected_flare_digest}"
        )
    flare_checkpoint = torch.load(
        args.flare_checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    if int(flare_checkpoint["step"]) != 50000 or "ema" not in flare_checkpoint:
        raise RuntimeError("not the released Stage-1 final EMA checkpoint")
    flare_cfg = flare_checkpoint["config"]
    evaluation_cfg = load_config(args.config)
    flare_cfg["data"] = evaluation_cfg["data"]
    flare_cfg["performance"] = {
        **(flare_cfg.get("performance", {}) or {}),
        **(evaluation_cfg.get("performance", {}) or {}),
    }
    stats = load_training_stats(args.stats)
    flare_model = build_model(flare_cfg, stats.condition).to(device)
    flare_load_report = _load_state_verified(flare_model, flare_checkpoint, "ema")
    flare_model.eval()
    flare_sampler = _build_sampler(flare_cfg, stats, ode_steps=10)
    flare_step = int(flare_checkpoint["step"])
    del flare_checkpoint

    reports: list[dict[str, Any]] = []
    started = time.time()
    for case_index, case in enumerate(selection["selected_cases"]):
        condition_id = str(case["condition_id"])
        base = str(case["base_id"]).zfill(4)
        segment = int(case["control_segment_index"])
        metadata = _json(audit_root / "conditions" / condition_id / "condition_metadata.json")
        anchor, exact_report = _verify_exact_anchor(metadata)
        record_indices = groups.get(base)
        if record_indices is None or len(record_indices) != 30:
            raise RuntimeError(f"selected x30 base {base} is not in the held-out split")
        record_index = int(record_indices[0])
        sample, condition_report = _audit_repeat_inputs(
            dataset,
            [int(index) for index in record_indices],
            segment_index=segment,
            expected_horizon_ns=float(case["horizon_ns"]),
            seed=args.seed + int(base) * 100 + segment,
        )
        sample_anchor_tensor = sample["m_init"]
        loader_anchor_dtype = str(getattr(sample_anchor_tensor, "dtype", "unknown"))
        if isinstance(sample_anchor_tensor, torch.Tensor):
            sample_anchor = sample_anchor_tensor.detach().cpu().numpy().astype(
                np.float32, copy=False
            )
        else:
            sample_anchor = np.asarray(sample_anchor_tensor, dtype=np.float32)
        sample_anchor = sample_anchor / np.maximum(
            np.linalg.norm(sample_anchor, axis=0, keepdims=True), 1.0e-8
        )
        parsed_anchor = anchor / np.maximum(
            np.linalg.norm(anchor, axis=0, keepdims=True), 1.0e-8
        )
        loader_max_abs = float(np.max(np.abs(sample_anchor - parsed_anchor)))
        # The trajectory loader stores magnetization frames as float16.  The
        # exact-anchor claim is established above from all 30 raw float32 OVFs;
        # this second check only guards against loading the wrong frame after
        # the expected half-precision quantization.
        if loader_max_abs > 5.0e-4:
            raise RuntimeError(
                f"dataset sample is not the audited exact anchor for {condition_id}: "
                f"max_abs={loader_max_abs}"
            )

        poseidon_sample = attach_native_schedule_fields(
            dataset, record_index, _clone_sample(sample)
        )
        manual_prediction = _formal_prediction(
            poseidon_model,
            dataset,
            {"record_index": record_index},
            poseidon_sample,
            device=device,
            precision=args.precision,
            seed=args.seed + case_index * 1_000_003,
        )
        manual_prediction = _postprocess_prediction(
            manual_prediction, poseidon_sample
        )[0].detach().cpu().numpy().astype(np.float16)
        official_prediction = _unique_predictions(
            {
                "metadata": {"reference_mode": "saved_segment_endpoint"},
                "samples": [sample],
            },
            dataset,
            [record_index],
            poseidon_model,
            device=device,
            precision=args.precision,
            seed=args.seed + case_index * 1_000_003,
        )[0]
        adapter_replay_max_abs = float(
            np.max(
                np.abs(
                    manual_prediction.astype(np.float32)
                    - official_prediction.astype(np.float32)
                )
            )
        )
        if adapter_replay_max_abs != 0.0:
            raise RuntimeError(
                f"manual/formal Poseidon adapter mismatch: {adapter_replay_max_abs}"
            )

        flare_master_seed = int(
            args.flare_seed + int(base) * 1_000 + segment * 100
        )
        flare_draws, flare_report = _fresh_flare_ensemble(
            flare_model,
            flare_sampler,
            sample,
            device=device,
            master_seed=flare_master_seed,
            count=30,
        )

        raw_dir = args.output_dir / "raw" / condition_id
        _save_npy(raw_dir / "anchor_f16.npy", parsed_anchor.astype(np.float16))
        _save_npy(raw_dir / "poseidon_t_f16.npy", official_prediction)
        _save_npy(raw_dir / "flare_30_f16.npy", flare_draws)
        report = {
            "condition_id": condition_id,
            "base_id": base,
            "control_segment_index": segment,
            "horizon_ns": float(case["horizon_ns"]),
            "poseidon_checkpoint": str(args.poseidon_checkpoint.resolve()),
            "poseidon_checkpoint_sha256": poseidon_digest,
            "poseidon_checkpoint_step": poseidon_step,
            "poseidon_precision": args.precision,
            "poseidon_output_shape": list(official_prediction.shape),
            "poseidon_temporal_adapter": "one lead-time-conditioned endpoint call",
            "poseidon_manual_vs_formal_adapter_max_abs_after_float16": adapter_replay_max_abs,
            "poseidon_formal_adapter_replay_test": "passed",
            "poseidon_postprocessing": "F.normalize(eps=1e-8), then exact geometry mask",
            "flare_checkpoint": str(args.flare_checkpoint.resolve()),
            "flare_checkpoint_sha256": flare_digest,
            "flare_checkpoint_step": flare_step,
            "flare_checkpoint_state": "ema",
            "flare_output_shape": list(flare_draws.shape),
            "flare_generation": flare_report,
            "dataset_anchor_dtype": loader_anchor_dtype,
            "dataset_anchor_quantization_tolerance": 5.0e-4,
            "dataset_sample_vs_reference_anchor_max_abs": loader_max_abs,
            **condition_report,
            **exact_report,
        }
        _write_json(raw_dir / "export_manifest.json", report)
        reports.append(report)
        print(json.dumps({"exported": condition_id, "exact_anchor": exact_report}), flush=True)

    payload = {
        "schema": "flare_generative_necessity_inference_export_v2",
        "status": "complete",
        "selection": str(args.selection.resolve()),
        "fresh_flare_draws_per_case": 30,
        "cached_flare_draws_used": False,
        "poseidon_checkpoint": str(args.poseidon_checkpoint.resolve()),
        "poseidon_checkpoint_sha256": poseidon_digest,
        "poseidon_checkpoint_step": poseidon_step,
        "flare_checkpoint": str(args.flare_checkpoint.resolve()),
        "flare_checkpoint_sha256": flare_digest,
        "flare_checkpoint_step": flare_step,
        "flare_checkpoint_state": "ema",
        "flare_state_load": flare_load_report,
        "checkpoint_tensor_equivalence": checkpoint_audit,
        "code_audit": {
            "poseidon_formal_prediction_source_sha256": _source_sha256(
                _formal_prediction
            ),
            "poseidon_official_unique_predictions_source_sha256": _source_sha256(
                _unique_predictions
            ),
            "poseidon_condition_adapter_source_sha256": _source_sha256(
                build_native_condition_tensor
            ),
            "poseidon_predict_source_sha256": _source_sha256(
                AuthorNativeX5Model.predict
            ),
            "paper_observable_source_sha256": _source_sha256(_paper_observables),
        },
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "elapsed_seconds": time.time() - started,
        "cases": reports,
    }
    _write_json(args.output_dir / "raw_export_manifest.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
