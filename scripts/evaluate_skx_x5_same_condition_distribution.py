#!/usr/bin/env python3
"""Evaluate SKX checkpoints on complete held-out MuMax repeat groups.

The defaults preserve the original x5 protocol.  ``--repeat-dataset`` and
``--repeats-per-group`` allow the same published metric to evaluate deeper
repeat sets (for example, the 165-base x30 dataset) without changing the
statistical algorithm.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_ROOT = Path(
    str(Path(__file__).resolve().parents[1] / "third_party/distribution_score")
)
DIST_SRC = DIST_ROOT / "src"
for path in (PROJECT_ROOT, DIST_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from distribution_score.same_condition import run as run_same_condition  # noqa: E402
from distribution_score.distance import (  # noqa: E402
    FORMAL_BLOCKS,
    FORMAL_SHIFT_RADIUS_PX,
)
from skyrmion_cfm.config import release_checkpoint_config
from skyrmion_cfm.config import load_config, seed_everything  # noqa: E402
from skyrmion_cfm.data.fixed_time import (  # noqa: E402
    build_fixed_time_datasets,
    collate_fixed_time_conditions,
)
from skyrmion_cfm.data.stats import load_training_stats  # noqa: E402
from skyrmion_cfm.eval.conditional_algorithm_diagnostics import (  # noqa: E402
    _autocast,
    _build_sampler,
    _load_state_verified,
)
from skyrmion_cfm.models import build_model  # noqa: E402
from skyrmion_cfm.train import move_batch  # noqa: E402
from scripts.x5_distribution_metrics import (  # noqa: E402
    aggregate_clustered_mean,
    angular_energy_distance_from_files,
)
from scripts.x5_anchor_jitter import apply_anchor_jitter  # noqa: E402


RUN_PATTERN = re.compile(r"_base(?P<base>\d+)_tr(?P<tr>\d+)_")
DEFAULT_REPEAT_DATASET = "skx_bt_1000base_x5_20260803"
SEGMENT_LABELS = {1: "drive", 2: "post_relax"}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() != target:
            raise RuntimeError(f"existing symlink {link} points to the wrong target")
        return
    if link.exists():
        raise RuntimeError(f"refusing to replace existing path: {link}")
    link.symlink_to(target, target_is_directory=directory)


def _frame_path(record: Any, frame_index: int) -> Path:
    if 0 <= int(frame_index) < len(record.frames):
        candidate = Path(record.frames[int(frame_index)])
        if candidate.is_file():
            return candidate
    suffix = f"{int(frame_index):06d}"
    matches = [Path(path) for path in record.frames if suffix in Path(path).stem]
    if len(matches) != 1:
        raise RuntimeError(
            f"cannot resolve frame {frame_index} for {record.run_id}: {matches}"
        )
    return matches[0]


def _complete_test_groups(
    dataset: Any,
    *,
    repeat_dataset: str = DEFAULT_REPEAT_DATASET,
    repeats_per_group: int = 5,
) -> dict[str, list[int]]:
    if repeats_per_group < 2:
        raise ValueError("repeats-per-group must be at least 2")
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for record_index, record in enumerate(dataset.records):
        if record.path.parent.name != repeat_dataset:
            continue
        match = RUN_PATTERN.search(record.run_id)
        if match is None:
            continue
        groups[match.group("base")].append((int(match.group("tr")), record_index))
    complete: dict[str, list[int]] = {}
    for base, entries in groups.items():
        entries.sort()
        if [trajectory for trajectory, _ in entries] == list(range(repeats_per_group)):
            complete[base] = [record_index for _, record_index in entries]
    if not complete:
        raise RuntimeError(
            "held-out split contains no complete repeat groups for "
            f"dataset={repeat_dataset!r}, repeats_per_group={repeats_per_group}"
        )
    return dict(sorted(complete.items()))


def _condition_supports_duration(
    dataset: Any,
    *,
    record_indices: list[int],
    segment_index: int,
    duration_ns: float,
) -> bool:
    """Return whether every repeat has the requested within-segment target."""
    for record_index in record_indices:
        record = dataset.records[record_index]
        specs = {
            int(spec[0]): spec
            for spec in dataset._trajectory_visual_segment_specs(record_index)
        }
        if segment_index not in specs:
            return False
        _, start_frame, _end_frame, _start_ns, _end_ns, save_step_ps = specs[
            segment_index
        ]
        target = dataset._target_frame_for_duration(
            int(record_index),
            record,
            float(save_step_ps),
            int(start_frame),
            float(duration_ns),
        )
        if target is None:
            return False
    return True


def _prepare_condition(
    dataset: Any,
    *,
    base: str,
    record_indices: list[int],
    segment_index: int,
    condition_dir: Path,
    duration_ns: float | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(20260811 + int(base) * 10 + segment_index)
    samples: list[dict[str, Any]] = []
    repeats: list[dict[str, Any]] = []
    starts: list[float] = []
    ends: list[float] = []
    for repeat_position, record_index in enumerate(record_indices):
        record = dataset.records[record_index]
        specs = {
            int(spec[0]): spec
            for spec in dataset._trajectory_visual_segment_specs(record_index)
        }
        if segment_index not in specs:
            raise RuntimeError(
                f"{record.run_id} has no control segment {segment_index}: {sorted(specs)}"
            )
        _, start_frame, segment_end_frame, start_ns, segment_end_ns, save_step_ps = specs[
            segment_index
        ]
        if duration_ns is None:
            end_frame = int(segment_end_frame)
            end_ns = float(segment_end_ns)
        else:
            target_frame = dataset._target_frame_for_duration(
                int(record_index),
                record,
                float(save_step_ps),
                int(start_frame),
                float(duration_ns),
            )
            if target_frame is None:
                raise RuntimeError(
                    f"{record.run_id} segment {segment_index} has no legal "
                    f"{duration_ns:g}-ns target"
                )
            end_frame = int(target_frame)
            end_ns = float(
                dataset._frame_time_ns(record, end_frame, float(save_step_ps))
            )
        sample = dataset._build_visual_sample_for_frames(
            record_index,
            int(start_frame),
            int(end_frame),
            rng,
        )
        if int(sample["frame_init"]) != int(start_frame):
            raise RuntimeError("dataset changed the requested segment start frame")
        if int(sample["frame_target"]) != int(end_frame):
            raise RuntimeError("dataset changed the requested segment target frame")
        samples.append(sample)
        starts.append(float(start_ns))
        ends.append(float(end_ns))

        repeat_dir = condition_dir / "repeats" / f"rep_{repeat_position:03d}"
        run_out = repeat_dir / "run.out"
        run_out.mkdir(parents=True, exist_ok=True)
        anchor_path = _frame_path(record, int(start_frame))
        truth_path = _frame_path(record, int(end_frame))
        _ensure_symlink(run_out / "segment_000_end.ovf", anchor_path)
        _ensure_symlink(run_out / "segment_001_end.ovf", truth_path)
        params_path = record.path / "params.json"
        if params_path.is_file():
            _ensure_symlink(repeat_dir / "params.json", params_path)
        repeats.append(
            {
                "repeat_position": repeat_position,
                "run_id": record.run_id,
                "trajectory_path": record.path,
                "frame_init": int(start_frame),
                "frame_target": int(end_frame),
                "absolute_start_ns": float(start_ns),
                "absolute_end_ns": float(end_ns),
                "horizon_ns": float(sample["t_end_ns"]),
                "anchor_ovf": anchor_path,
                "truth_ovf": truth_path,
            }
        )
    if max(starts) - min(starts) > 1.0e-4 or max(ends) - min(ends) > 1.0e-4:
        raise RuntimeError(
            f"base {base} segment {segment_index} has inconsistent repeat boundaries: "
            f"starts={starts}, ends={ends}"
        )
    metadata = {
        "condition_id": condition_dir.name,
        "base_id": base,
        "control_segment_index": segment_index,
        "segment_role": SEGMENT_LABELS.get(segment_index, f"segment_{segment_index}"),
        "num_mumax_repeats": len(repeats),
        "absolute_start_ns": float(np.mean(starts)),
        "absolute_end_ns": float(np.mean(ends)),
        "horizon_ns": float(np.mean([float(sample["t_end_ns"]) for sample in samples])),
        "requested_horizon_ns": duration_ns,
        "model_input": "matching MuMax segment start for each assigned repeat",
        "repeats": repeats,
    }
    _write_json(condition_dir / "condition_metadata.json", metadata)
    return {"samples": samples, "metadata": metadata, "condition_dir": condition_dir}


def _retarget_zero_current_rollout_sample(
    dataset: Any,
    sample: dict[str, Any],
    *,
    absolute_start_ns: float,
    horizon_ns: float,
) -> dict[str, Any]:
    """Describe a solver-generated constant-control continuation.

    [5NS-ROLLOUT FIX 2026-08-29] The reference endpoint need not already be a
    saved frame in the original finite trajectory.  MuMax3 rolls forward from
    the exact saved post-drive anchor; this helper gives the learned model the
    identical zero-current horizon without pretending that an original frame
    index exists.
    """
    horizon_s = float(horizon_ns) * 1.0e-9
    sample["t_end_ns"] = torch.tensor(horizon_ns, dtype=torch.float32)
    sample["t_end_s"] = torch.tensor(horizon_s, dtype=torch.float32)
    sample["t_end_index"] = torch.tensor(
        dataset._t_end_bucket_index(float(horizon_ns)), dtype=torch.long
    )
    # Keep backwards-compatible aliases consistent with the primary fields.
    sample["dt_s"] = sample["t_end_s"].clone()
    sample["dt_index"] = sample["t_end_index"].clone()
    sample["frame_target_time_ns"] = torch.tensor(
        float(absolute_start_ns) + float(horizon_ns), dtype=torch.float32
    )
    sample["drive_time_s"] = torch.tensor(0.0, dtype=torch.float32)
    sample["relax_time_s"] = torch.tensor(horizon_s, dtype=torch.float32)
    sample["drive_fraction"] = torch.tensor(0.0, dtype=torch.float32)
    sample["current_a_m2"] = torch.tensor(0.0, dtype=torch.float32)
    sample["control_segment_index"] = torch.tensor(2, dtype=torch.long)
    for key in (
        "j_field",
        "control_grid",
        "j_x_field",
        "j_y_field",
        "j_z_field",
    ):
        value = sample.get(key)
        if torch.is_tensor(value):
            sample[key] = torch.zeros_like(value)
    return sample


def _prepare_rollout_condition(
    dataset: Any,
    *,
    base: str,
    record_indices: list[int],
    condition_dir: Path,
    target_root: Path,
    horizon_ns: float = 5.0,
) -> dict[str, Any]:
    """Prepare five exact anchors and their new MuMax3 rollout endpoints."""
    rng = np.random.default_rng(20260829 + int(base) * 10)
    samples: list[dict[str, Any]] = []
    repeats: list[dict[str, Any]] = []
    starts: list[float] = []
    ends: list[float] = []
    target_dir = target_root / "targets" / f"base{base}_segment002_post_relax"
    for repeat_position, record_index in enumerate(record_indices):
        record = dataset.records[record_index]
        specs = {
            int(spec[0]): spec
            for spec in dataset._trajectory_visual_segment_specs(record_index)
        }
        if 2 not in specs:
            raise RuntimeError(f"{record.run_id} has no post-drive segment")
        _, start_frame, saved_end_frame, start_ns, _saved_end_ns, _save_step_ps = specs[2]
        target_path = target_dir / f"rep_{repeat_position:03d}_m_final.ovf"
        target_metadata_path = target_dir / f"rep_{repeat_position:03d}_target.json"
        if not target_path.is_file() or not target_metadata_path.is_file():
            raise FileNotFoundError(
                f"missing 5-ns rollout target for base {base}, repeat "
                f"{repeat_position}: {target_path}"
            )
        target_metadata = json.loads(target_metadata_path.read_text(encoding="utf-8"))
        if target_metadata.get("status") != "complete":
            raise RuntimeError(f"incomplete rollout target: {target_metadata_path}")
        if str(target_metadata.get("run_id")) != str(record.run_id):
            raise RuntimeError(
                f"rollout target run mismatch: {target_metadata.get('run_id')} "
                f"!= {record.run_id}"
            )

        sample = dataset._build_visual_sample_for_frames(
            int(record_index), int(start_frame), int(saved_end_frame), rng
        )
        # ``start_ns`` is the nominal pulse
        # boundary, whereas the selected OVF is the first saved frame after
        # that boundary.  Condition the model on the OVF's actual timestamp.
        actual_start_ns = float(sample["frame_init_time_ns"])
        if not math.isclose(
            float(target_metadata.get("anchor_absolute_time_ns", float("nan"))),
            actual_start_ns,
            rel_tol=0.0,
            abs_tol=5.0e-4,
        ):
            raise RuntimeError(
                f"rollout target anchor time mismatch for {record.run_id}: "
                f"{target_metadata.get('anchor_absolute_time_ns')} vs "
                f"{actual_start_ns} ns"
            )
        if not math.isclose(
            float(target_metadata.get("horizon_ns", float("nan"))),
            float(horizon_ns),
            rel_tol=0.0,
            abs_tol=1.0e-8,
        ) or target_metadata.get("crosses_control_boundary") is not False:
            raise RuntimeError(f"wrong rollout task metadata: {target_metadata_path}")
        sample = _retarget_zero_current_rollout_sample(
            dataset,
            sample,
            absolute_start_ns=actual_start_ns,
            horizon_ns=float(horizon_ns),
        )
        samples.append(sample)
        starts.append(actual_start_ns)
        ends.append(actual_start_ns + float(horizon_ns))

        repeat_dir = condition_dir / "repeats" / f"rep_{repeat_position:03d}"
        run_out = repeat_dir / "run.out"
        run_out.mkdir(parents=True, exist_ok=True)
        anchor_path = _frame_path(record, int(start_frame))
        _ensure_symlink(run_out / "segment_000_end.ovf", anchor_path)
        _ensure_symlink(run_out / "segment_001_end.ovf", target_path)
        params_path = record.path / "params.json"
        if params_path.is_file():
            _ensure_symlink(repeat_dir / "params.json", params_path)
        repeats.append(
            {
                "repeat_position": repeat_position,
                "run_id": record.run_id,
                "trajectory_path": record.path,
                "frame_init": int(start_frame),
                "frame_target": None,
                "absolute_start_ns": actual_start_ns,
                "absolute_end_ns": actual_start_ns + float(horizon_ns),
                "nominal_post_drive_boundary_ns": float(start_ns),
                "horizon_ns": float(horizon_ns),
                "anchor_ovf": anchor_path,
                "truth_ovf": target_path,
                "truth_metadata": target_metadata_path,
            }
        )
    if max(starts) - min(starts) > 1.0e-4 or max(ends) - min(ends) > 1.0e-4:
        raise RuntimeError(
            f"base {base} has inconsistent rollout boundaries: "
            f"starts={starts}, ends={ends}"
        )
    metadata = {
        "condition_id": condition_dir.name,
        "base_id": base,
        "control_segment_index": 2,
        "segment_role": "post_relax",
        "reference_mode": "mumax3_rollout_extension",
        "change_marker": "[5NS-ROLLOUT FIX 2026-08-29]",
        "num_mumax_repeats": len(repeats),
        "absolute_start_ns": float(np.mean(starts)),
        "absolute_end_ns": float(np.mean(ends)),
        "horizon_ns": float(horizon_ns),
        "requested_horizon_ns": float(horizon_ns),
        "saved_in_original_trajectory": False,
        "crosses_control_boundary": False,
        "control": "constant zero current for the full continuation",
        "model_input": "matching exact MuMax post-drive anchor for each repeat",
        "rollout_target_root": target_root,
        "repeats": repeats,
    }
    _write_json(condition_dir / "condition_metadata.json", metadata)
    return {"samples": samples, "metadata": metadata, "condition_dir": condition_dir}


# Build the physical two-segment
# path from protocol boundaries rather than treating 5 ns as one transition.
def _prepare_multisegment_rollout_condition(
    dataset: Any,
    *,
    base: str,
    record_indices: list[int],
    condition_dir: Path,
) -> dict[str, Any]:
    """Prepare an exact-control-boundary drive->relax rollout.

    Saved frames are used only for the initial state and final reference.  The
    internal handoff time comes directly from the control protocol.  Carrier
    samples on either side of the boundary provide the appropriate constant
    control tensors, but their saved-frame durations are replaced by the exact
    protocol intervals.  Consequently no saved-frame gap is skipped and no
    ground-truth state is inserted at the handoff.
    """
    paths: list[list[dict[str, Any]]] = []
    repeats: list[dict[str, Any]] = []
    absolute_starts: list[float] = []
    absolute_ends: list[float] = []
    composed_horizons: list[float] = []
    segment_counts: set[int] = set()
    segment_indices: set[tuple[int, ...]] = set()

    for repeat_position, record_index in enumerate(record_indices):
        record = dataset.records[int(record_index)]
        visual_specs = dataset._trajectory_visual_segment_specs(int(record_index))
        if len(visual_specs) < 2:
            raise RuntimeError(
                f"{record.run_id} has fewer than two physical control segments"
            )
        exact_intervals = [
            (int(index), float(start_ns), float(end_ns))
            for index, _start_frame, _end_frame, start_ns, end_ns, _save_step_ps
            in visual_specs
        ]
        if any(
            not math.isclose(left[2], right[1], rel_tol=0.0, abs_tol=2.0e-4)
            for left, right in zip(exact_intervals, exact_intervals[1:])
        ):
            raise RuntimeError(
                f"non-contiguous control protocol for {record.run_id}: "
                f"{exact_intervals}"
            )

        first_spec = visual_specs[0]
        last_spec = visual_specs[-1]
        first_saved_start_ns = float(
            dataset._frame_time_ns(record, int(first_spec[1]), float(first_spec[5]))
        )
        final_saved_end_ns = float(
            dataset._frame_time_ns(record, int(last_spec[2]), float(last_spec[5]))
        )
        # Adaptive MuMax stepping can put a saved header a few femtoseconds
        # beyond the nominal protocol start/end.  The rollout starts from the
        # actual saved anchor and ends at the actual saved reference, while
        # every internal switch remains the exact protocol boundary.
        exact_boundaries = [
            first_saved_start_ns,
            *[float(interval[2]) for interval in exact_intervals[:-1]],
            final_saved_end_ns,
        ]
        if any(
            right <= left
            for left, right in zip(exact_boundaries, exact_boundaries[1:])
        ):
            raise RuntimeError(
                f"non-increasing exact rollout boundaries for {record.run_id}: "
                f"{exact_boundaries}"
            )

        sources = dataset._sources_for(int(record_index), record)
        pulse_window = dataset._pulse_window_for(int(record_index), sources)
        samples: list[dict[str, Any]] = []
        segment_metadata: list[dict[str, Any]] = []
        for segment_position, spec in enumerate(visual_specs):
            (
                control_segment_index,
                carrier_start_frame,
                carrier_end_frame,
                protocol_start_ns,
                protocol_end_ns,
                save_step_ps,
            ) = spec
            rng = np.random.default_rng(
                20260829
                + int(base) * 10_000
                + repeat_position * 101
                + segment_position
            )
            sample = dataset._build_visual_sample_for_frames(
                int(record_index),
                int(carrier_start_frame),
                int(carrier_end_frame),
                rng,
            )
            exact_start_ns = float(exact_boundaries[segment_position])
            exact_end_ns = float(exact_boundaries[segment_position + 1])
            exact_duration_ns = exact_end_ns - exact_start_ns
            drive_time_ns = min(
                dataset._pulse_overlap_ns(
                    exact_start_ns,
                    exact_end_ns,
                    pulse_window,
                ),
                exact_duration_ns,
            )
            relax_time_ns = exact_duration_ns - drive_time_ns
            # Each model call must see one constant-control segment.  A partial
            # pulse overlap here would mean the protocol was segmented wrong.
            overlap_tol_ns = 2.0e-4
            if (
                drive_time_ns > overlap_tol_ns
                and relax_time_ns > overlap_tol_ns
            ):
                raise RuntimeError(
                    f"control changes inside exact segment {segment_position} "
                    f"for {record.run_id}: drive={drive_time_ns}, "
                    f"relax={relax_time_ns} ns"
                )

            sample["t_end_ns"] = torch.tensor(
                exact_duration_ns, dtype=torch.float32
            )
            sample["t_end_s"] = torch.tensor(
                exact_duration_ns * 1.0e-9, dtype=torch.float32
            )
            sample["t_end_index"] = torch.tensor(
                dataset._t_end_bucket_index(exact_duration_ns), dtype=torch.long
            )
            sample["dt_s"] = sample["t_end_s"].clone()
            sample["dt_index"] = sample["t_end_index"].clone()
            sample["frame_init_time_ns"] = torch.tensor(
                exact_start_ns, dtype=torch.float32
            )
            sample["frame_target_time_ns"] = torch.tensor(
                exact_end_ns, dtype=torch.float32
            )
            sample["control_time_ns"] = torch.tensor(
                exact_start_ns, dtype=torch.float32
            )
            sample["control_segment_index"] = torch.tensor(
                int(control_segment_index), dtype=torch.long
            )
            sample["drive_time_s"] = torch.tensor(
                drive_time_ns * 1.0e-9, dtype=torch.float32
            )
            sample["relax_time_s"] = torch.tensor(
                relax_time_ns * 1.0e-9, dtype=torch.float32
            )
            sample["drive_fraction"] = torch.tensor(
                drive_time_ns / exact_duration_ns, dtype=torch.float32
            )
            j_field_max_abs = float(sample["j_field"].abs().max())
            has_drive = (
                abs(float(sample["current_a_m2"])) > 1.0
                or j_field_max_abs > 1.0
            )
            expects_drive = drive_time_ns > overlap_tol_ns
            if has_drive != expects_drive:
                raise RuntimeError(
                    f"control tensor/protocol mismatch in exact segment "
                    f"{segment_position} for {record.run_id}: "
                    f"expects_drive={expects_drive}, "
                    f"current={float(sample['current_a_m2'])}, "
                    f"j_field_max={j_field_max_abs}"
                )
            samples.append(sample)
            segment_metadata.append(
                {
                    "segment_position": segment_position,
                    "control_segment_index": int(control_segment_index),
                    "exact_start_ns": exact_start_ns,
                    "exact_end_ns": exact_end_ns,
                    "duration_ns": exact_duration_ns,
                    "drive_time_ns": drive_time_ns,
                    "relax_time_ns": relax_time_ns,
                    "current_a_m2": float(sample["current_a_m2"]),
                    "j_field_max_abs_a_m2": j_field_max_abs,
                    "drive_fraction": drive_time_ns / exact_duration_ns,
                    "carrier_frame_init": int(carrier_start_frame),
                    "carrier_frame_target": int(carrier_end_frame),
                    "carrier_saved_start_ns": float(
                        dataset._frame_time_ns(
                            record, int(carrier_start_frame), float(save_step_ps)
                        )
                    ),
                    "carrier_saved_end_ns": float(
                        dataset._frame_time_ns(
                            record, int(carrier_end_frame), float(save_step_ps)
                        )
                    ),
                    "protocol_start_ns": float(protocol_start_ns),
                    "protocol_end_ns": float(protocol_end_ns),
                }
            )

        indices = tuple(
            int(sample["control_segment_index"]) for sample in samples
        )
        if indices != tuple(sorted(indices)) or len(set(indices)) != len(indices):
            raise RuntimeError(
                f"unordered/repeated segment indices for {record.run_id}: {indices}"
            )
        segment_counts.add(len(samples))
        segment_indices.add(indices)
        paths.append(samples)

        first = samples[0]
        last = samples[-1]
        absolute_start_ns = float(exact_boundaries[0])
        absolute_end_ns = float(exact_boundaries[-1])
        composed_horizon_ns = float(
            sum(float(sample["t_end_ns"]) for sample in samples)
        )
        absolute_span_ns = absolute_end_ns - absolute_start_ns
        if not math.isclose(
            composed_horizon_ns,
            absolute_span_ns,
            rel_tol=0.0,
            abs_tol=2.0e-4,
        ):
            raise RuntimeError(
                f"exact rollout time does not close for {record.run_id}: "
                f"sum(segments)={composed_horizon_ns}, "
                f"final-anchor={absolute_span_ns} ns"
            )
        absolute_starts.append(absolute_start_ns)
        absolute_ends.append(absolute_end_ns)
        composed_horizons.append(composed_horizon_ns)

        handoffs: list[dict[str, Any]] = []
        for left, right in zip(segment_metadata, segment_metadata[1:]):
            boundary_gap_ns = right["exact_start_ns"] - left["exact_end_ns"]
            if not math.isclose(
                boundary_gap_ns, 0.0, rel_tol=0.0, abs_tol=2.0e-4
            ):
                raise RuntimeError(
                    f"nonzero exact handoff gap for {record.run_id}: "
                    f"{boundary_gap_ns} ns"
                )
            handoffs.append(
                {
                    "from_control_segment_index": left["control_segment_index"],
                    "to_control_segment_index": right["control_segment_index"],
                    "exact_boundary_ns": left["exact_end_ns"],
                    "exact_boundary_gap_ns": boundary_gap_ns,
                    "left_carrier_saved_end_ns": left["carrier_saved_end_ns"],
                    "right_carrier_saved_start_ns": right["carrier_saved_start_ns"],
                    "carrier_saved_frame_gap_ns": (
                        right["carrier_saved_start_ns"]
                        - left["carrier_saved_end_ns"]
                    ),
                    "state_source": "previous model prediction",
                }
            )

        repeat_dir = condition_dir / "repeats" / f"rep_{repeat_position:03d}"
        run_out = repeat_dir / "run.out"
        run_out.mkdir(parents=True, exist_ok=True)
        anchor_path = _frame_path(record, int(first["frame_init"]))
        truth_path = _frame_path(record, int(last["frame_target"]))
        _ensure_symlink(run_out / "segment_000_end.ovf", anchor_path)
        _ensure_symlink(run_out / "segment_001_end.ovf", truth_path)
        params_path = record.path / "params.json"
        if params_path.is_file():
            _ensure_symlink(repeat_dir / "params.json", params_path)
        repeats.append(
            {
                "repeat_position": repeat_position,
                "run_id": record.run_id,
                "trajectory_path": record.path,
                "absolute_start_ns": absolute_start_ns,
                "absolute_end_ns": absolute_end_ns,
                "absolute_span_ns": absolute_span_ns,
                "composed_model_horizon_ns": composed_horizon_ns,
                "anchor_ovf": anchor_path,
                "truth_ovf": truth_path,
                "segments": segment_metadata,
                "handoffs": handoffs,
            }
        )

    if len(segment_counts) != 1 or len(segment_indices) != 1:
        raise RuntimeError(
            "inconsistent multisegment paths within one repeat group: "
            f"counts={segment_counts}, indices={segment_indices}"
        )
    # Repeats of one base share the same nominal schedule.  Permit only the
    # sub-ps header/float roundoff already tolerated by the segment evaluator.
    if (
        max(absolute_starts) - min(absolute_starts) > 1.0e-4
        or max(absolute_ends) - min(absolute_ends) > 1.0e-4
        or max(composed_horizons) - min(composed_horizons) > 1.0e-4
    ):
        raise RuntimeError(
            f"base {base} has inconsistent repeat timelines: "
            f"starts={absolute_starts}, ends={absolute_ends}, "
            f"composed={composed_horizons}"
        )

    ordered_indices = list(next(iter(segment_indices)))
    metadata = {
        "condition_id": condition_dir.name,
        "base_id": base,
        "control_segment_index": -1,
        "control_segment_indices": ordered_indices,
        "segment_role": "drive_to_post_relax",
        "reference_mode": "exact_control_multisegment_rollout_endpoint",
        "protocol": "autoregressive composition over exact protocol control segments",
        "num_mumax_repeats": len(repeats),
        "segments_per_trajectory": len(ordered_indices),
        "handoff_count": len(ordered_indices) - 1,
        "absolute_start_ns": float(np.mean(absolute_starts)),
        "absolute_end_ns": float(np.mean(absolute_ends)),
        "absolute_span_ns": float(
            np.mean(np.asarray(absolute_ends) - np.asarray(absolute_starts))
        ),
        "horizon_ns": float(np.mean(composed_horizons)),
        "composed_model_horizon_ns": float(np.mean(composed_horizons)),
        "requested_horizon_ns": None,
        "saved_in_original_trajectory": True,
        "crosses_control_boundary": True,
        "internal_boundary_source": "control protocol; never snapped to saved frames",
        "time_closure_invariant": (
            "sum(segment durations) == final timestamp - anchor timestamp"
        ),
        "teacher_forcing_after_first_segment": False,
        "model_input": (
            "matching MuMax drive-start anchor for each repeat; every later "
            "exact control segment consumes only the previous model prediction"
        ),
        "repeats": repeats,
    }
    _write_json(condition_dir / "condition_metadata.json", metadata)
    return {
        "segment_samples": paths,
        "metadata": metadata,
        "condition_dir": condition_dir,
    }


# Timing-only retiming keeps the two exact
# control segments and prediction handoff while standardizing their total span.
def _override_multisegment_timing_durations(
    prepared: dict[str, Any],
    dataset: Any,
    durations_ns: list[float] | tuple[float, ...],
) -> dict[str, Any]:
    """Retimestamp a prepared path for a fixed-duration timing workload.

    This helper is intentionally restricted to benchmark preparation.  It does
    not alter saved states or references and must never be used for quality.
    """
    durations = tuple(float(value) for value in durations_ns)
    if not durations or any(value <= 0.0 for value in durations):
        raise ValueError(f"invalid timing segment durations: {durations}")
    paths = prepared.get("segment_samples")
    metadata = prepared.get("metadata")
    if not isinstance(paths, list) or not paths or not isinstance(metadata, dict):
        raise ValueError("invalid prepared multisegment timing condition")
    if any(len(path) != len(durations) for path in paths):
        raise ValueError(
            f"timing duration count {len(durations)} does not match path shape"
        )
    repeats = metadata.get("repeats")
    if not isinstance(repeats, list) or len(repeats) != len(paths):
        raise ValueError("timing repeat metadata is misaligned")

    absolute_starts: list[float] = []
    absolute_ends: list[float] = []
    for path, repeat in zip(paths, repeats):
        segments = repeat.get("segments")
        if not isinstance(segments, list) or len(segments) != len(durations):
            raise ValueError("timing segment metadata is misaligned")
        cursor_ns = float(path[0]["frame_init_time_ns"])
        absolute_start_ns = cursor_ns
        boundaries = [cursor_ns]
        for sample, segment, duration_ns in zip(path, segments, durations):
            driven = float(sample.get("drive_fraction", 0.0)) > 0.5
            next_ns = cursor_ns + duration_ns
            sample["t_end_ns"] = torch.tensor(duration_ns, dtype=torch.float32)
            sample["t_end_s"] = torch.tensor(
                duration_ns * 1.0e-9, dtype=torch.float32
            )
            sample["t_end_index"] = torch.tensor(
                dataset._t_end_bucket_index(duration_ns), dtype=torch.long
            )
            sample["dt_s"] = sample["t_end_s"].clone()
            sample["dt_index"] = sample["t_end_index"].clone()
            sample["frame_init_time_ns"] = torch.tensor(
                cursor_ns, dtype=torch.float32
            )
            sample["frame_target_time_ns"] = torch.tensor(
                next_ns, dtype=torch.float32
            )
            sample["control_time_ns"] = torch.tensor(
                cursor_ns, dtype=torch.float32
            )
            sample["drive_time_s"] = torch.tensor(
                duration_ns * 1.0e-9 if driven else 0.0,
                dtype=torch.float32,
            )
            sample["relax_time_s"] = torch.tensor(
                0.0 if driven else duration_ns * 1.0e-9,
                dtype=torch.float32,
            )
            sample["drive_fraction"] = torch.tensor(
                1.0 if driven else 0.0, dtype=torch.float32
            )

            segment["exact_start_ns"] = cursor_ns
            segment["exact_end_ns"] = next_ns
            segment["duration_ns"] = duration_ns
            segment["drive_time_ns"] = duration_ns if driven else 0.0
            segment["relax_time_ns"] = 0.0 if driven else duration_ns
            segment["drive_fraction"] = 1.0 if driven else 0.0
            segment["protocol_start_ns"] = cursor_ns
            segment["protocol_end_ns"] = next_ns
            cursor_ns = next_ns
            boundaries.append(cursor_ns)

        repeat["absolute_start_ns"] = absolute_start_ns
        repeat["absolute_end_ns"] = cursor_ns
        repeat["absolute_span_ns"] = cursor_ns - absolute_start_ns
        repeat["composed_model_horizon_ns"] = sum(durations)
        handoffs = repeat.get("handoffs", [])
        for position, handoff in enumerate(handoffs):
            handoff["exact_boundary_ns"] = boundaries[position + 1]
            handoff["exact_boundary_gap_ns"] = 0.0
        absolute_starts.append(absolute_start_ns)
        absolute_ends.append(cursor_ns)

    total_ns = sum(durations)
    metadata["protocol"] = (
        "standardized fixed-duration composition over exact control segments"
    )
    metadata["timing_only_duration_override"] = True
    metadata["timing_segment_durations_ns"] = list(durations)
    metadata["absolute_start_ns"] = float(np.mean(absolute_starts))
    metadata["absolute_end_ns"] = float(np.mean(absolute_ends))
    metadata["absolute_span_ns"] = total_ns
    metadata["horizon_ns"] = total_ns
    metadata["composed_model_horizon_ns"] = total_ns
    metadata["requested_horizon_ns"] = total_ns
    condition_dir = Path(prepared["condition_dir"])
    _write_json(condition_dir / "condition_metadata.json", metadata)
    return prepared


@torch.inference_mode()
def _generate_condition(
    prepared: dict[str, Any],
    model: torch.nn.Module,
    sampler: Any,
    *,
    checkpoint: Path,
    checkpoint_digest: str,
    checkpoint_step: int,
    state: str,
    num_model: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    anchor_jitter_rms_deg: float,
    anchor_jitter_correlation_px: float,
    force: bool,
) -> Path:
    condition_dir = Path(prepared["condition_dir"])
    model_dir = condition_dir / "model_samples"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model_samples_f16.npy"
    manifest_path = model_dir / "manifest.json"
    expected_shape = [num_model, 3, 256, 256]
    if model_path.is_file() and manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") == "complete"
            and manifest.get("checkpoint_sha256") == checkpoint_digest
            and manifest.get("shape") == expected_shape
            and manifest.get("requested_horizon_ns")
            == prepared["metadata"].get("requested_horizon_ns")
            and manifest.get("reference_mode")
            == prepared["metadata"].get("reference_mode", "saved_segment_endpoint")
            and float(manifest.get("anchor_jitter_rms_deg", 0.0))
            == float(anchor_jitter_rms_deg)
            and (
                anchor_jitter_rms_deg == 0.0
                or float(manifest.get("anchor_jitter_correlation_px", 0.0))
                == float(anchor_jitter_correlation_px)
            )
        ):
            return model_path

    samples = prepared["samples"]
    assignment = np.arange(num_model, dtype=np.int64) % len(samples)
    memmap = np.lib.format.open_memmap(
        model_path,
        mode="w+",
        dtype=np.float16,
        shape=tuple(expected_shape),
    )
    generated = 0
    started = time.time()
    jitter_reports: list[dict[str, Any]] = []
    while generated < num_model:
        end = min(num_model, generated + batch_size)
        selected = [samples[int(index)] for index in assignment[generated:end]]
        torch.manual_seed(seed + generated)
        torch.cuda.manual_seed_all(seed + generated)
        batch = move_batch(default_collate(selected), device)
        condition = collate_fixed_time_conditions(batch)
        model_input = batch["m_init"].float()
        if anchor_jitter_rms_deg > 0.0:
            model_input, batch_reports = apply_anchor_jitter(
                model_input,
                rms_degrees=anchor_jitter_rms_deg,
                correlation_px=anchor_jitter_correlation_px,
                seeds=[
                    seed + 1_000_003 * sample_index + 701
                    for sample_index in range(generated, end)
                ],
            )
            jitter_reports.extend(batch_reports)
        with _autocast(device):
            prediction, _ = sampler.sample(model, model_input, condition)
        mask = (
            batch.get("m_observed_init", model_input)
            .float()
            .square()
            .sum(dim=1, keepdim=True)
            .sqrt()
            > 0.5
        ).float()
        prediction = F.normalize(prediction.float(), dim=1, eps=1.0e-8) * mask
        memmap[generated:end] = prediction.cpu().numpy().astype(np.float16)
        generated = end
        memmap.flush()
    del memmap
    manifest = {
        "status": "complete",
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_step": checkpoint_step,
        "checkpoint_state": state,
        "condition_id": prepared["metadata"]["condition_id"],
        "model_input": prepared["metadata"]["model_input"],
        "absolute_start_ns": prepared["metadata"]["absolute_start_ns"],
        "absolute_end_ns": prepared["metadata"]["absolute_end_ns"],
        # Make cached samples task-specific.
        "requested_horizon_ns": prepared["metadata"].get("requested_horizon_ns"),
        "reference_mode": prepared["metadata"].get(
            "reference_mode", "saved_segment_endpoint"
        ),
        "num_samples": num_model,
        "repeat_assignment": (
            f"round_robin_over_{len(samples)}_matching_MuMax_anchors"
        ),
        "repeat_index_file": model_dir / "anchor_repeat_index.npy",
        "seed": seed,
        "anchor_jitter_rms_deg": anchor_jitter_rms_deg,
        "anchor_jitter_correlation_px": anchor_jitter_correlation_px,
        "anchor_jitter_application": (
            "one tangent-plane perturbation of the initial anchor before the model call"
            if anchor_jitter_rms_deg > 0.0
            else "none"
        ),
        "anchor_jitter_reports": jitter_reports,
        "shape": expected_shape,
        "dtype": "float16",
        "elapsed_seconds": time.time() - started,
    }
    np.save(model_dir / "anchor_repeat_index.npy", assignment.astype(np.int16))
    _write_json(manifest_path, manifest)
    return model_path


@torch.inference_mode()
def _generate_multisegment_condition(
    prepared: dict[str, Any],
    model: torch.nn.Module,
    sampler: Any,
    *,
    checkpoint: Path,
    checkpoint_digest: str,
    checkpoint_step: int,
    state: str,
    num_model: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    anchor_jitter_rms_deg: float,
    anchor_jitter_correlation_px: float,
    force: bool,
) -> Path:
    """Generate final states after recursive FLARE segment composition."""
    condition_dir = Path(prepared["condition_dir"])
    model_dir = condition_dir / "model_samples"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model_samples_f16.npy"
    manifest_path = model_dir / "manifest.json"
    paths = prepared["segment_samples"]
    if not paths or not paths[0]:
        raise RuntimeError("empty multisegment rollout task")
    segment_count = len(paths[0])
    if any(len(path) != segment_count for path in paths):
        raise RuntimeError("inconsistent multisegment sample paths")
    expected_shape = [num_model, 3, 256, 256]
    if model_path.is_file() and manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") == "complete"
            and manifest.get("checkpoint_sha256") == checkpoint_digest
            and manifest.get("shape") == expected_shape
            and manifest.get("reference_mode")
            == prepared["metadata"]["reference_mode"]
            and manifest.get("segment_count") == segment_count
            and manifest.get("handoff_count") == segment_count - 1
            and float(manifest.get("anchor_jitter_rms_deg", 0.0))
            == float(anchor_jitter_rms_deg)
            and (
                anchor_jitter_rms_deg == 0.0
                or float(manifest.get("anchor_jitter_correlation_px", 0.0))
                == float(anchor_jitter_correlation_px)
            )
        ):
            return model_path

    assignment = np.arange(num_model, dtype=np.int64) % len(paths)
    memmap = np.lib.format.open_memmap(
        model_path,
        mode="w+",
        dtype=np.float16,
        shape=tuple(expected_shape),
    )
    generated = 0
    started = time.time()
    jitter_reports: list[dict[str, Any]] = []
    while generated < num_model:
        end = min(num_model, generated + batch_size)
        selected_paths = [paths[int(index)] for index in assignment[generated:end]]
        prediction: torch.Tensor | None = None
        final_batch: dict[str, Any] | None = None
        for segment_position in range(segment_count):
            selected = [path[segment_position] for path in selected_paths]
            torch.manual_seed(seed + generated + 100_003 * segment_position)
            torch.cuda.manual_seed_all(seed + generated + 100_003 * segment_position)
            batch = move_batch(default_collate(selected), device)
            condition = collate_fixed_time_conditions(batch)
            model_input = batch["m_init"].float() if prediction is None else prediction
            if segment_position == 0 and anchor_jitter_rms_deg > 0.0:
                model_input, batch_reports = apply_anchor_jitter(
                    model_input,
                    rms_degrees=anchor_jitter_rms_deg,
                    correlation_px=anchor_jitter_correlation_px,
                    seeds=[
                        seed + 1_000_003 * sample_index + 701
                        for sample_index in range(generated, end)
                    ],
                )
                jitter_reports.extend(batch_reports)
            with _autocast(device):
                prediction, _ = sampler.sample(model, model_input, condition)
            # Deliberately do not teacher-force or project at the handoff.  This
            # matches the established stage-rollout evaluator exactly.
            final_batch = batch
        if prediction is None or final_batch is None:
            raise RuntimeError("multisegment rollout produced no prediction")
        observed = final_batch.get(
            "m_observed_init", final_batch["m_init"]
        ).float()
        mask = observed.square().sum(dim=1, keepdim=True).sqrt().gt(0.5).float()
        prediction = F.normalize(prediction.float(), dim=1, eps=1.0e-8) * mask
        memmap[generated:end] = prediction.cpu().numpy().astype(np.float16)
        generated = end
        memmap.flush()
    del memmap
    np.save(model_dir / "anchor_repeat_index.npy", assignment.astype(np.int16))
    _write_json(
        manifest_path,
        {
            "status": "complete",
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_digest,
            "checkpoint_step": checkpoint_step,
            "checkpoint_state": state,
            "condition_id": prepared["metadata"]["condition_id"],
            "reference_mode": prepared["metadata"]["reference_mode"],
            "protocol": prepared["metadata"]["protocol"],
            "control_segment_indices": prepared["metadata"][
                "control_segment_indices"
            ],
            "segment_count": segment_count,
            "handoff_count": segment_count - 1,
            "teacher_forcing_after_first_segment": False,
            "absolute_start_ns": prepared["metadata"]["absolute_start_ns"],
            "absolute_end_ns": prepared["metadata"]["absolute_end_ns"],
            "absolute_span_ns": prepared["metadata"]["absolute_span_ns"],
            "composed_model_horizon_ns": prepared["metadata"][
                "composed_model_horizon_ns"
            ],
            "num_samples": num_model,
            "repeat_assignment": (
                f"round_robin_over_{len(paths)}_matching_MuMax_pre_drive_anchors"
            ),
            "repeat_index_file": model_dir / "anchor_repeat_index.npy",
            "seed": seed,
            "anchor_jitter_rms_deg": anchor_jitter_rms_deg,
            "anchor_jitter_correlation_px": anchor_jitter_correlation_px,
            "anchor_jitter_application": (
                "one tangent-plane perturbation of the exact drive-start anchor only"
                if anchor_jitter_rms_deg > 0.0
                else "none"
            ),
            "anchor_jitter_reports": jitter_reports,
            "shape": expected_shape,
            "dtype": "float16",
            "elapsed_seconds": time.time() - started,
        },
    )
    return model_path


def _geometric_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.exp(np.log(values.clip(1.0e-300, 1.0)).mean()))


def _aggregate(
    rows: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    segments = sorted({int(row["control_segment_index"]) for row in rows})

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in selected:
            grouped[str(row["base_id"])].append(
                float(row["symmetric_ratio_score"])
            )
        group_ids = sorted(grouped)
        values = np.asarray(
            [value for group_id in group_ids for value in grouped[group_id]],
            dtype=np.float64,
        )
        bootstrap = np.empty(iterations, dtype=np.float64)
        for index in range(iterations):
            sampled = rng.integers(0, len(group_ids), size=len(group_ids))
            draw = np.asarray(
                [
                    value
                    for group_index in sampled
                    for value in grouped[group_ids[int(group_index)]]
                ],
                dtype=np.float64,
            )
            bootstrap[index] = _geometric_mean(draw)
        return {
            "conditions": len(values),
            "base_groups": len(group_ids),
            "geometric_mean_score": _geometric_mean(values),
            "geometric_mean_bootstrap_95ci": np.quantile(
                bootstrap, [0.025, 0.975]
            ).tolist(),
            "arithmetic_mean_score": float(values.mean()),
            "median_score": float(np.median(values)),
            "min_score": float(values.min()),
            "max_score": float(values.max()),
            "bootstrap_unit": "base group; segments kept together",
        }

    return {
        "aggregation_note": (
            "The published same-condition algorithm defines one score per physical "
            "condition. The geometric means below are an equal-condition diagnostic, "
            "not an additional score defined by the publication package. Confidence "
            "intervals resample base groups and keep their segments together."
        ),
        "combined": summarize(rows),
        "by_segment": {
            str(segment): summarize(
                [row for row in rows if int(row["control_segment_index"]) == segment]
            )
            for segment in segments
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=None,
        help=(
            "Optional config whose data section defines the evaluation dataset. "
            "The checkpoint model/prior config and the supplied training stats are retained."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", choices=("ema", "raw"), default="ema")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--segments", type=int, nargs="+", default=(1, 2))
    parser.add_argument(
        "--duration-ns",
        type=float,
        default=None,
        help=(
            "Override each segment endpoint by this duration from the exact "
            "segment boundary. The requested target must remain inside the segment."
        ),
    )
    parser.add_argument(
        "--rollout-target-root",
        type=Path,
        default=None,
        help=(
            "[5NS-ROLLOUT FIX 2026-08-29] Evaluate one 5-ns zero-current "
            "query from each saved post-drive anchor against MuMax3 endpoints "
            "generated under TARGET_ROOT/targets."
        ),
    )
    parser.add_argument(
        "--multisegment-rollout",
        action="store_true",
        help=(
            "Run the complete drive->post-relax trajectory by composing one "
            "model transition per exact protocol control segment. Saved frames "
            "are used only for the initial state and final reference."
        ),
    )
    parser.add_argument("--repeat-dataset", default=DEFAULT_REPEAT_DATASET)
    parser.add_argument("--repeats-per-group", type=int, default=5)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--num-model", type=int, default=128)
    parser.add_argument(
        "--score-num-model",
        type=int,
        default=None,
        help=(
            "Use only the first N generated samples for Distribution Score and "
            "angular energy distance. This permits 25 generated samples (five per "
            "exact anchor) while retaining the matched five-versus-five score."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ode-steps", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=208110700)
    parser.add_argument(
        "--anchor-jitter-rms-deg",
        type=float,
        default=0.0,
        help=(
            "Apply a fixed-RMS tangent perturbation to each exact input anchor. "
            "Zero preserves the original evaluation bit-for-bit."
        ),
    )
    parser.add_argument(
        "--anchor-jitter-correlation-px",
        type=float,
        default=4.0,
        help="Gaussian correlation length for nonzero anchor jitter.",
    )
    parser.add_argument(
        "--disable-cudnn-sdp",
        action="store_true",
        help="Disable the cuDNN SDPA backend when its GPU execution plan is unavailable.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help=(
            "Export anchor-conditioned model samples and metadata without running "
            "the legacy macro-condition density/energy diagnostics."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    score_num_model = (
        int(args.num_model)
        if args.score_num_model is None
        else int(args.score_num_model)
    )
    if args.num_model <= 0 or score_num_model <= 0:
        raise ValueError("num-model and score-num-model must be positive")
    if args.anchor_jitter_rms_deg < 0.0:
        raise ValueError("anchor-jitter-rms-deg must be non-negative")
    if args.anchor_jitter_correlation_px < 0.0:
        raise ValueError("anchor-jitter-correlation-px must be non-negative")
    if score_num_model > args.num_model:
        raise ValueError("score-num-model cannot exceed num-model")
    if args.duration_ns is not None and args.duration_ns <= 0.0:
        raise ValueError("duration-ns must be positive")
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
        # One scored condition per base.  ``-1`` denotes the composed path,
        # not an additional physical segment.
        args.segments = (-1,)

    if not DIST_ROOT.is_dir():
        raise FileNotFoundError(DIST_ROOT)
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SKYRMION_CFM_FUSED_OPS", "1")
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    checkpoint_digest = _sha256(args.checkpoint)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    cfg = release_checkpoint_config(checkpoint["config"])
    checkpoint_training_dataset_root = cfg.get("data", {}).get("dataset_root")
    if args.evaluation_config is not None:
        evaluation_cfg = load_config(args.evaluation_config)
        cfg["data"] = evaluation_cfg["data"]
        cfg["performance"] = {
            **(cfg.get("performance", {}) or {}),
            **(evaluation_cfg.get("performance", {}) or {}),
        }
    performance_cfg = cfg.get("performance", {}) or {}
    if args.disable_cudnn_sdp or bool(performance_cfg.get("disable_cudnn_sdp", False)):
        torch.backends.cuda.enable_cudnn_sdp(False)
        print(
            {
                "attention_backends": {
                    "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled(),
                    "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
                    "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
                    "math_sdp": torch.backends.cuda.math_sdp_enabled(),
                }
            },
            flush=True,
        )
    cfg.setdefault("data", {}).setdefault("memmap", {})["auto_build"] = False
    cfg["data"]["memmap"]["force_rebuild"] = False
    cfg.setdefault("train", {})["num_workers"] = 0
    cfg["train"]["persistent_workers"] = False
    _, _, test_dataset = build_fixed_time_datasets(cfg, build_splits={"test"})
    if test_dataset is None:
        raise RuntimeError("failed to build held-out dataset")
    groups = _complete_test_groups(
        test_dataset,
        repeat_dataset=args.repeat_dataset,
        repeats_per_group=args.repeats_per_group,
    )
    if args.max_groups is not None:
        if args.max_groups <= 0:
            raise ValueError("max-groups must be positive")
        groups = dict(list(groups.items())[: args.max_groups])

    stats = load_training_stats(args.stats)
    model = build_model(cfg, stats.condition).to(device)
    load_report = _load_state_verified(model, checkpoint, args.state)
    checkpoint_step = int(checkpoint.get("step", -1))
    del checkpoint
    model.eval()
    sampler = _build_sampler(cfg, stats, ode_steps=args.ode_steps)

    rows: list[dict[str, Any]] = []
    started = time.time()
    total = len(groups) * len(args.segments)
    position = 0
    skipped_conditions: list[dict[str, Any]] = []
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
            generator = (
                _generate_multisegment_condition
                if args.multisegment_rollout
                else _generate_condition
            )
            model_path = generator(
                prepared,
                model,
                sampler,
                checkpoint=args.checkpoint,
                checkpoint_digest=checkpoint_digest,
                checkpoint_step=checkpoint_step,
                state=args.state,
                num_model=args.num_model,
                batch_size=args.batch_size,
                seed=args.seed + int(base) * 100 + segment_index * 10_000,
                device=device,
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
                            "model_samples": str(model_path),
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
                    num_model=score_num_model,
                    num_mumax=args.repeats_per_group,
                    blocks=FORMAL_BLOCKS,
                    shift_radius=FORMAL_SHIFT_RADIUS_PX,
                    reference_chunk=128,
                    bootstrap=args.bootstrap,
                    seed=args.seed + int(base) * 1000 + segment_index,
                    device=args.device,
                    progress=False,
                    skip_auxiliary_texture=True,
                )
            )
            primary = result["primary_patch_shift"]
            energy = None
            if args.repeats_per_group == 5 and score_num_model >= 5:
                energy = angular_energy_distance_from_files(
                    model_path,
                    score_dir / "mumax_targets_f16.npy",
                    score_dir / "geometry_mask.npy",
                    num_model=score_num_model,
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
                    **(energy or {}),
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

    del model, sampler, stats, test_dataset
    gc.collect()
    torch.cuda.empty_cache()
    summary_dir = args.output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    _write_json(summary_dir / "skipped_conditions.json", skipped_conditions)
    if args.generate_only:
        summary = {
            "status": "complete",
            "mode": "exact_anchor_sample_generation_only",
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": checkpoint_digest,
            "checkpoint_step": checkpoint_step,
            "checkpoint_state": args.state,
            "checkpoint_load": load_report,
            "evaluation_split": "metadata test split",
            "evaluation_config": args.evaluation_config,
            "checkpoint_training_dataset_root": checkpoint_training_dataset_root,
            "evaluation_dataset_root": cfg.get("data", {}).get("dataset_root"),
            "repeat_dataset": args.repeat_dataset,
            "complete_repeat_base_groups": len(groups),
            "segments": list(args.segments),
            "conditions": total,
            "mumax_repeats_per_condition": args.repeats_per_group,
            "model_samples_per_condition": args.num_model,
            "score_model_samples_per_condition": score_num_model,
            "requested_horizon_ns": (
                5.0 if args.rollout_target_root is not None else args.duration_ns
            ),
            "rollout_target_root": args.rollout_target_root,
            "multisegment_rollout": args.multisegment_rollout,
            "model_draws_per_exact_anchor": (
                args.num_model // args.repeats_per_group
                if args.num_model % args.repeats_per_group == 0
                else None
            ),
            # Generate-only summaries
            # must describe the same prediction-only handoff recorded in each
            # condition manifest; the previous fixed string described only the
            # legacy single-segment path.
            "model_input": (
                "matching pre-drive MuMax start followed by model-prediction handoff"
                if args.multisegment_rollout
                else "matching real MuMax start for each evaluated segment"
            ),
            "rollout": args.rollout_target_root is not None or args.multisegment_rollout,
            "reference_mode": (
                "exact_control_multisegment_rollout_endpoint"
                if args.multisegment_rollout
                else (
                    "mumax3_rollout_extension"
                    if args.rollout_target_root is not None
                    else "saved_segment_endpoint"
                )
            ),
            "sampler_ode_steps": args.ode_steps,
            "anchor_jitter_rms_deg": args.anchor_jitter_rms_deg,
            "anchor_jitter_correlation_px": args.anchor_jitter_correlation_px,
            "elapsed_seconds": time.time() - started,
        }
        _write_json(summary_dir / "run_summary.json", summary)
        print(json.dumps(summary, indent=2, default=_json_default), flush=True)
        return
    if not rows:
        summary = {
            "status": "no_legal_conditions",
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": checkpoint_digest,
            "checkpoint_step": checkpoint_step,
            "checkpoint_state": args.state,
            "checkpoint_load": load_report,
            "evaluation_split": "metadata test split",
            "repeat_dataset": args.repeat_dataset,
            "complete_repeat_base_groups": len(groups),
            "segments": list(args.segments),
            "conditions": 0,
            "candidate_conditions": total,
            "skipped_conditions": len(skipped_conditions),
            "requested_horizon_ns": (
                5.0 if args.rollout_target_root is not None else args.duration_ns
            ),
            "rollout_target_root": args.rollout_target_root,
            "multisegment_rollout": args.multisegment_rollout,
            "model_samples_per_condition": args.num_model,
            "score_model_samples_per_condition": score_num_model,
            "support_statement": "no requested target remains inside a complete control segment",
            "elapsed_seconds": time.time() - started,
        }
        _write_json(summary_dir / "run_summary.json", summary)
        print(json.dumps(summary, indent=2, default=_json_default), flush=True)
        return
    with (summary_dir / "condition_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    algorithm_version = json.loads(
        (DIST_ROOT / "ALGORITHM_VERSION.json").read_text(encoding="utf-8")
    )
    summary = {
        "status": "complete",
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_step": checkpoint_step,
        "checkpoint_state": args.state,
        "checkpoint_load": load_report,
        "implementation": DIST_ROOT,
        "algorithm_version": algorithm_version,
        "evaluation_split": "metadata test split",
        "evaluation_config": args.evaluation_config,
        "checkpoint_training_dataset_root": checkpoint_training_dataset_root,
        "evaluation_dataset_root": cfg.get("data", {}).get("dataset_root"),
        "repeat_dataset": args.repeat_dataset,
        "complete_repeat_base_groups": len(groups),
        "segments": list(args.segments),
        "conditions": len(rows),
        "candidate_conditions": total,
        "skipped_conditions": len(skipped_conditions),
        "mumax_repeats_per_condition": args.repeats_per_group,
        "model_samples_per_condition": args.num_model,
        "score_model_samples_per_condition": score_num_model,
        "requested_horizon_ns": (
            5.0 if args.rollout_target_root is not None else args.duration_ns
        ),
        "rollout_target_root": args.rollout_target_root,
        "multisegment_rollout": args.multisegment_rollout,
        "model_input": (
            "matching pre-drive MuMax start followed by model-prediction handoff"
            if args.multisegment_rollout
            else "matching real MuMax start for each evaluated segment"
        ),
        "rollout": args.rollout_target_root is not None or args.multisegment_rollout,
        "reference_mode": (
            "exact_control_multisegment_rollout_endpoint"
            if args.multisegment_rollout
            else (
                "mumax3_rollout_extension"
                if args.rollout_target_root is not None
                else "saved_segment_endpoint"
            )
        ),
        "sampler_ode_steps": args.ode_steps,
        "anchor_jitter_rms_deg": args.anchor_jitter_rms_deg,
        "anchor_jitter_correlation_px": args.anchor_jitter_correlation_px,
        "score": _aggregate(rows, iterations=args.bootstrap, seed=args.seed + 999),
        "elapsed_seconds": time.time() - started,
    }
    if rows and "angular_energy_distance_deg" in rows[0]:
        summary["angular_energy_distance"] = aggregate_clustered_mean(
            rows,
            "angular_energy_distance_deg",
            iterations=args.bootstrap,
            seed=args.seed + 1_999,
        )
        summary["angular_energy_distance_model_samples_per_condition"] = score_num_model
        summary["paired_angular_error"] = aggregate_clustered_mean(
            rows,
            "paired_model_mumax_mean_deg",
            iterations=args.bootstrap,
            seed=args.seed + 2_999,
        )
    _write_json(summary_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
