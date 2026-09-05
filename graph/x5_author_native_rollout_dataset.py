"""Deterministic x5 cases and local-step training for native rollout baselines.

This module selects trajectories once and exposes two views of exactly the
same cases:

The evaluation view can expose a direct ``frame(0) -> frame(t)`` pair for
timing lead-time operators.  Training in this module is restricted to
consecutive fixed-duration pairs for autoregressive/latent evolution.  The
lead-time models train on the project's ordinary control-segment dataset in
``external_baselines/x5_author_native.py``; there is deliberately no special
0-to-5-ns endpoint-training mode here.

It deliberately contains no model logic.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


SELECTION_SEED = 20_260_815
TIME_TOLERANCE_NS = 2.0e-4
STEP_NS = 0.25
MAX_HORIZON_NS = 5.0


def first_legal_segment_sample(
    base_dataset: Any,
    *,
    segment_index: int,
    horizon_ns: float,
    seed: int = SELECTION_SEED,
    preferred_run_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return one exact within-segment sample at ``horizon_ns``.

    [MATCHED-HORIZON FIX 2026-08-29] Timing must use a query that belongs to
    the same legal task as fidelity evaluation.  This helper deliberately
    rejects cross-boundary initial-to-endpoint views.
    """
    if horizon_ns <= 0.0:
        raise ValueError("horizon_ns must be positive")
    required = (
        "records",
        "_valid_record_idx",
        "_trajectory_visual_segment_specs",
        "_target_frame_for_duration",
        "_build_visual_sample_for_frames",
        "_frame_time_ns",
    )
    missing = [name for name in required if not hasattr(base_dataset, name)]
    if missing:
        raise TypeError(f"unsupported x5 dataset; missing internals: {missing}")

    candidates = sorted(
        (int(index) for index in base_dataset._valid_record_idx),
        key=lambda index: str(base_dataset.records[index].run_id),
    )
    if preferred_run_id is not None:
        candidates = [
            index
            for index in candidates
            if str(base_dataset.records[index].run_id) == preferred_run_id
        ]
        if not candidates:
            raise RuntimeError(
                f"requested matched timing run is not in the held-out split: "
                f"{preferred_run_id}"
            )
    for record_index in candidates:
        record = base_dataset.records[record_index]
        specs = {
            int(spec[0]): spec
            for spec in base_dataset._trajectory_visual_segment_specs(record_index)
        }
        if segment_index not in specs:
            continue
        _, start_frame, _end_frame, start_ns, _end_ns, save_step_ps = specs[
            segment_index
        ]
        target_frame = base_dataset._target_frame_for_duration(
            record_index,
            record,
            float(save_step_ps),
            int(start_frame),
            float(horizon_ns),
        )
        if target_frame is None:
            continue
        sample = base_dataset._build_visual_sample_for_frames(
            record_index,
            int(start_frame),
            int(target_frame),
            np.random.default_rng(seed + 104_729 * record_index),
        )
        observed = float(sample["t_end_ns"])
        if not math.isclose(observed, horizon_ns, rel_tol=0.0, abs_tol=TIME_TOLERANCE_NS):
            continue
        sample = attach_native_schedule_fields(base_dataset, record_index, sample)
        target_ns = base_dataset._frame_time_ns(
            record,
            int(target_frame),
            float(save_step_ps),
        )
        metadata = {
            "selection": (
                "explicit paper-quality held-out run with an exact legal target"
                if preferred_run_id is not None
                else "first lexicographic held-out run with an exact legal target"
            ),
            "run_id": str(record.run_id),
            "record_index": record_index,
            "control_segment_index": int(segment_index),
            "frame_init": int(start_frame),
            "frame_target": int(target_frame),
            "absolute_start_ns": float(start_ns),
            "absolute_end_ns": float(target_ns),
            "requested_horizon_ns": float(horizon_ns),
            "realized_horizon_ns": observed,
            "crosses_control_boundary": False,
        }
        return sample, metadata
    raise RuntimeError(
        f"no held-out control segment {segment_index} has an exact legal "
        f"{horizon_ns:g}-ns target"
    )


def _planned_control_fields(
    base_dataset: Any,
    record_index: int,
    sample: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Return the complete-pulse current map aligned with sample augmentation.

    The x5 pulse begins after the recorded initial state.  Consequently the
    instantaneous current map at t=0 is correctly zero but is not a complete
    description of the prescribed future control.  Native endpoint operators
    and LE-PDE's encode-once rollout need this separate schedule field.
    """
    record = base_dataset.records[record_index]
    sources = base_dataset._sources_for(record_index, record)
    pulse_window = base_dataset._pulse_window_for(record_index, sources)
    zero = torch.zeros_like(sample["defect_field"], dtype=torch.float32)
    planned = {axis: zero.clone() for axis in ("x", "y", "z")}
    if pulse_window is not None:
        pulse_start_ns, pulse_end_ns = pulse_window
        active_time_ns = 0.5 * (float(pulse_start_ns) + float(pulse_end_ns))
        fields = base_dataset._extra_spatial_fields_for(
            record_index,
            sources,
            active_time_ns,
        )
        for axis in ("x", "y", "z"):
            value = fields.get(f"j_{axis}_field")
            if torch.is_tensor(value):
                planned[axis] = value.float().clone()

    rot_k = int(sample.get("augment_rot_k", torch.tensor(0)).item()) % 4
    if rot_k:
        for axis in planned:
            planned[axis] = torch.rot90(planned[axis], k=rot_k, dims=(-2, -1))
        planned["x"], planned["y"] = base_dataset._rotate_xy_values(
            planned["x"], planned["y"], rot_k
        )
    return planned


def attach_native_schedule_fields(
    base_dataset: Any,
    record_index: int,
    sample: dict[str, Any],
) -> dict[str, Any]:
    """Attach non-learned absolute-time and complete-control descriptors."""
    record = base_dataset.records[record_index]
    sources = base_dataset._sources_for(record_index, record)
    pulse_window = base_dataset._pulse_window_for(record_index, sources)
    if pulse_window is None:
        pulse_start_ns = pulse_end_ns = 0.0
    else:
        pulse_start_ns, pulse_end_ns = map(float, pulse_window)
    planned = _planned_control_fields(base_dataset, record_index, sample)
    for axis in ("x", "y", "z"):
        sample[f"planned_j_{axis}_field"] = planned[axis]
    sample["pulse_start_ns"] = torch.tensor(pulse_start_ns, dtype=torch.float32)
    sample["pulse_end_ns"] = torch.tensor(pulse_end_ns, dtype=torch.float32)
    return sample


def _complete_rollout_cases(
    base_dataset: Any,
    *,
    step_ns: float,
    max_horizon_ns: float,
    tolerance_ns: float,
) -> list[NativeRolloutCase]:
    num_steps = int(round(max_horizon_ns / step_ns))
    candidates: list[NativeRolloutCase] = []
    for rec_idx in base_dataset._valid_record_idx:
        record = base_dataset.records[rec_idx]
        max_frame = base_dataset._n_frames(record) - 1
        if max_frame < num_steps:
            continue
        save_step_ps = 25.0
        initial_time = base_dataset._frame_time_ns(record, 0, save_step_ps)
        frames = [0]
        times = [float(initial_time)]
        lower = 1
        for step in range(1, num_steps + 1):
            requested = initial_time + step * step_ns
            frame = base_dataset._frame_nearest_in_range(
                record,
                requested,
                lower,
                max_frame,
                save_step_ps,
            )
            if frame is None:
                break
            observed = base_dataset._frame_time_ns(record, int(frame), save_step_ps)
            if abs(float(observed) - float(requested)) > tolerance_ns:
                break
            frames.append(int(frame))
            times.append(float(observed))
            lower = int(frame) + 1
        if len(frames) == num_steps + 1:
            candidates.append(
                NativeRolloutCase(
                    record_index=int(rec_idx),
                    frames=tuple(frames),
                    times_ns=tuple(times),
                    run_id=str(record.run_id),
                )
            )
    return candidates


@dataclass(frozen=True)
class NativeRolloutCase:
    record_index: int
    frames: tuple[int, ...]
    times_ns: tuple[float, ...]
    run_id: str


class X5AuthorNativeRolloutCases:
    """Shared held-out cases for a fixed-step rollout through ``max_horizon``."""

    def __init__(
        self,
        base_dataset: Any,
        *,
        step_ns: float = 0.25,
        max_horizon_ns: float = 5.0,
        limit: int = 128,
        selection_seed: int = SELECTION_SEED,
        tolerance_ns: float = TIME_TOLERANCE_NS,
    ) -> None:
        if step_ns <= 0.0 or max_horizon_ns <= 0.0:
            raise ValueError("step_ns and max_horizon_ns must be positive")
        ratio = max_horizon_ns / step_ns
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("max_horizon_ns must be an integer multiple of step_ns")
        if limit <= 0:
            raise ValueError("limit must be positive")
        required = (
            "records",
            "_valid_record_idx",
            "_n_frames",
            "_frame_time_ns",
            "_frame_nearest_in_range",
            "_build_visual_sample_for_frames",
        )
        missing = [name for name in required if not hasattr(base_dataset, name)]
        if missing:
            raise TypeError(f"unsupported x5 dataset; missing internals: {missing}")

        self.base_dataset = base_dataset
        self.step_ns = float(step_ns)
        self.max_horizon_ns = float(max_horizon_ns)
        self.num_steps = int(round(ratio))
        self.limit = int(limit)
        self.selection_seed = int(selection_seed)
        self.tolerance_ns = float(tolerance_ns)

        candidates = _complete_rollout_cases(
            self.base_dataset,
            step_ns=self.step_ns,
            max_horizon_ns=self.max_horizon_ns,
            tolerance_ns=self.tolerance_ns,
        )

        candidates.sort(key=lambda item: item.run_id)
        random.Random(self.selection_seed).shuffle(candidates)
        if len(candidates) < self.limit:
            raise RuntimeError(
                f"only {len(candidates)} held-out x5 records support a complete "
                f"0--{self.max_horizon_ns:g} ns rollout; requested {self.limit}"
            )
        self.cases = tuple(candidates[: self.limit])

    def __len__(self) -> int:
        return len(self.cases)

    def _rng(self, case_index: int, step_index: int, *, direct: bool) -> np.random.Generator:
        salt = 1_000_003 if direct else 0
        return np.random.default_rng(
            self.selection_seed + salt + 10_007 * int(case_index) + 101 * int(step_index)
        )

    def step_sample(self, case_index: int, step_index: int) -> dict[str, Any]:
        """Return the native adjacent-step sample, with controls at its start."""
        if not 0 <= step_index < self.num_steps:
            raise IndexError(step_index)
        case = self.cases[case_index]
        self.base_dataset.segment_policy = "none"
        self.base_dataset.current_time_mode = "anchor"
        sample = self.base_dataset._build_visual_sample_for_frames(
            case.record_index,
            case.frames[step_index],
            case.frames[step_index + 1],
            self._rng(case_index, step_index, direct=False),
        )
        observed = float(sample["t_end_ns"])
        if abs(observed - self.step_ns) > self.tolerance_ns:
            raise RuntimeError(
                f"constructed {observed:.9g} ns native step; expected {self.step_ns:g} ns"
            )
        return attach_native_schedule_fields(
            self.base_dataset,
            case.record_index,
            sample,
        )

    def direct_sample(self, case_index: int, step_index: int) -> dict[str, Any]:
        """Return the initial-to-endpoint view for a lead-time operator."""
        if not 1 <= step_index <= self.num_steps:
            raise IndexError(step_index)
        case = self.cases[case_index]
        self.base_dataset.segment_policy = "none"
        self.base_dataset.current_time_mode = "t0"
        sample = self.base_dataset._build_visual_sample_for_frames(
            case.record_index,
            case.frames[0],
            case.frames[step_index],
            self._rng(case_index, step_index, direct=True),
        )
        expected = step_index * self.step_ns
        observed = float(sample["t_end_ns"])
        if abs(observed - expected) > self.tolerance_ns:
            raise RuntimeError(
                f"constructed {observed:.9g} ns direct pair; expected {expected:g} ns"
            )
        return attach_native_schedule_fields(
            self.base_dataset,
            case.record_index,
            sample,
        )

    def selection_metadata(self) -> dict[str, Any]:
        return {
            "selection_seed": self.selection_seed,
            "step_ns": self.step_ns,
            "max_horizon_ns": self.max_horizon_ns,
            "time_tolerance_ns": self.tolerance_ns,
            "samples": len(self.cases),
            "run_ids": [case.run_id for case in self.cases],
            "frame_indices": [list(case.frames) for case in self.cases],
            "frame_times_ns": [list(case.times_ns) for case in self.cases],
        }


class X5AuthorNativeTrainingDataset(Dataset):
    """Random local history-to-step or latent pairs from x5 trajectories."""

    def __init__(
        self,
        base_dataset: Any,
        *,
        protocol: str,
        length: int | None = None,
        seed: int = 78,
        step_ns: float = STEP_NS,
        max_horizon_ns: float = MAX_HORIZON_NS,
        tolerance_ns: float = TIME_TOLERANCE_NS,
        history_steps: int = 1,
    ) -> None:
        if protocol not in {"autoregressive", "latent"}:
            raise ValueError(f"unsupported native training protocol: {protocol!r}")
        self.base_dataset = base_dataset
        self.protocol = protocol
        self.seed = int(seed)
        self.step_ns = float(step_ns)
        self.max_horizon_ns = float(max_horizon_ns)
        self.num_steps = int(round(self.max_horizon_ns / self.step_ns))
        self.history_steps = int(history_steps)
        if self.history_steps <= 0:
            raise ValueError("history_steps must be positive")
        if self.protocol != "autoregressive" and self.history_steps != 1:
            raise ValueError("history_steps > 1 is only valid for autoregressive training")
        self.length = int(length if length is not None else len(base_dataset))
        self.cases = tuple(
            _complete_rollout_cases(
                base_dataset,
                step_ns=self.step_ns,
                max_horizon_ns=self.max_horizon_ns,
                tolerance_ns=float(tolerance_ns),
            )
        )
        if not self.cases:
            raise RuntimeError("no training trajectories support a complete 0--5 ns rollout")
        # Visual-pair construction supplies explicit frame overrides.  These
        # settings make condition lookup describe that full span rather than
        # silently truncating it at an SCFM control-stage boundary.
        self.base_dataset.segment_policy = "none"
        self.base_dataset.current_time_mode = "anchor"

    def __len__(self) -> int:
        return self.length

    def _augment_pair(
        self,
        sample: dict[str, Any],
        *,
        rot_k: int,
        spin_flip: bool,
    ) -> dict[str, Any]:
        if self.base_dataset.augment:
            self.base_dataset._augment_sample_in_place(
                sample,
                rot_k=rot_k,
                spin_flip=spin_flip,
            )
        return sample

    def _sample_for_frames(
        self,
        case: NativeRolloutCase,
        frame_start: int,
        frame_end: int,
        rng: np.random.Generator,
        *,
        rot_k: int,
        spin_flip: bool,
    ) -> dict[str, Any]:
        sample = self.base_dataset._build_visual_sample_for_frames(
            case.record_index,
            case.frames[frame_start],
            case.frames[frame_end],
            rng,
        )
        self._augment_pair(sample, rot_k=rot_k, spin_flip=spin_flip)
        return attach_native_schedule_fields(
            self.base_dataset,
            case.record_index,
            sample,
        )

    @staticmethod
    def _condition_view(sample: dict[str, Any]) -> dict[str, Any]:
        """Keep tensor condition entries needed to encode an LE-PDE target."""
        excluded = {
            "m_t",
            "m1",
            "omega_target",
            "aux_m_t",
            "aux_omega_target",
            "run_id",
        }
        return {
            key: value
            for key, value in sample.items()
            if key not in excluded and (torch.is_tensor(value) or isinstance(value, (int, float)))
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = np.random.default_rng(self.seed + 104_729 * int(index))
        case = self.cases[int(rng.integers(0, len(self.cases)))]
        rot_k = int(rng.integers(0, 4)) if self.base_dataset.augment_rot90 else 0
        spin_flip = (
            self.base_dataset.augment_spin_flip_prob > 0.0
            and rng.random() < self.base_dataset.augment_spin_flip_prob
        )

        # LE-PDE's latent-consistency loss needs a condition tensor anchored at
        # the following state, hence leave one extra adjacent pair available.
        needs_future = self.protocol == "latent"
        last_start = self.num_steps - (2 if needs_future else 1)
        start_step = int(rng.integers(0, last_start + 1))

        history_start = max(0, start_step - self.history_steps + 1)
        history_samples = [
            self._sample_for_frames(
                case,
                history_step,
                history_step + 1,
                rng,
                rot_k=rot_k,
                spin_flip=spin_flip,
            )
            for history_step in range(history_start, start_step + 1)
        ]
        sample = history_samples[-1]
        if self.history_steps > 1:
            history_conditions = [self._condition_view(item) for item in history_samples]
            # A benchmark trajectory begins at the task's physical initial
            # state and has no earlier observations.  Repeat that state on the
            # left, then use a causal sliding window after predictions begin.
            history_conditions = [history_conditions[0]] * (
                self.history_steps - len(history_conditions)
            ) + history_conditions
            sample["history_conditions"] = history_conditions

        if needs_future:
            future = self._sample_for_frames(
                case,
                start_step + 1,
                start_step + 2,
                rng,
                rot_k=rot_k,
                spin_flip=spin_flip,
            )
            sample["future_condition"] = self._condition_view(future)
        return sample
