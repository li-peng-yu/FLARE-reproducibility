"""Fixed-time-endpoint pair dataset for plan2 forward training.

Each draw pairs an anchor frame with one of a fixed set of target durations
``t_end \\in {0.25, 0.5, 1, 2, 3, 4} ns``. ``save_step_ps`` is auto-detected
from each run's ``params.json`` so 5-ps and 25-ps datasets co-exist
transparently.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from skyrmion_cfm.cfm.bridges import bridge_state_repr
from skyrmion_cfm.data.log_map import log_map_chw, normalize_spin
from skyrmion_cfm.data.ovf import read_ovf
from skyrmion_cfm.data.spatial_fields import (
    SpatialFieldSources,
    build_j_field,
    control_segments_ns,
    load_defect_field,
    pulse_window_ns,
    spatial_condition_fields_at_time,
)
from skyrmion_cfm.data.conditions import SCALAR_CONDITION_KEYS, V4_CATEGORICAL_KEYS
from skyrmion_cfm.data.quality_sampling import (
    PAIR_CATEGORY_TO_INDEX,
    SEGMENT_CATEGORY_TO_INDEX,
    build_segment_quality_cache,
    classify_pair_quality,
    load_segment_quality_cache,
    normalize_quality_sampling_config,
    pair_keep_probability,
    pair_quality_metrics,
    segment_keep_probability,
)
from skyrmion_cfm.data.trajectory import TrajectoryRecord
from skyrmion_cfm.data import v4_metadata


DEFAULT_T_END_NS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0)
_FRAME_RE = re.compile(r"m(\d+)\.ovf$")


def _split_trajectory_index_from_manifest(
    index: Any,
    manifest_path: str | Path,
) -> tuple[Any, Any, Any]:
    """Apply a predeclared run-id split without modifying dataset metadata."""
    from skyrmion_cfm.data.trajectory import TrajectoryIndex

    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    split_ids = payload.get("splits", payload)
    if not isinstance(split_ids, dict):
        raise ValueError(f"split manifest must contain an object: {path}")
    expected = ("train", "val", "test")
    missing = [name for name in expected if name not in split_ids]
    if missing:
        raise ValueError(f"split manifest is missing {missing}: {path}")
    records = {str(record.run_id): record for record in index.records}
    if len(records) != len(index.records):
        raise ValueError("trajectory index contains duplicate run_id values")
    used: set[str] = set()
    buckets: dict[str, list[Any]] = {}
    for name in expected:
        values = split_ids[name]
        if not isinstance(values, list) or not values:
            raise ValueError(f"split manifest {name!r} must be a non-empty list")
        ids = [str(value) for value in values]
        duplicates = used.intersection(ids)
        if duplicates:
            raise ValueError(
                f"split manifest reuses run ids across splits: {sorted(duplicates)[:5]}"
            )
        unknown = [run_id for run_id in ids if run_id not in records]
        if unknown:
            raise ValueError(
                f"split manifest references unknown run ids: {unknown[:5]}"
            )
        used.update(ids)
        buckets[name] = [records[run_id] for run_id in ids]
    return tuple(TrajectoryIndex(buckets[name]) for name in expected)


def _analytic_zeeman_reference(
    m_init: torch.Tensor,
    b_field: torch.Tensor,
    *,
    alpha: float,
    dt_s: float,
    gamma_hz_per_t: float,
    sign: float,
) -> torch.Tensor:
    """Exact constant-field Gilbert-LLG update, independently per site.

    This is an opt-in reference operator for the dataset, not a replacement
    for the learned dynamics.  Exchange, DMI, demagnetization, current torque,
    spatial coupling, and thermal noise remain for the CFM to learn as the
    correction from this reference state to the simulated target.
    """
    if b_field.ndim == 2:
        b_field = b_field.unsqueeze(0)
    if b_field.shape[0] != 3 or b_field.shape[-2:] != m_init.shape[-2:]:
        raise ValueError(
            "Zeeman reference field must have shape (3, H, W), got "
            f"{tuple(b_field.shape)} for m_init={tuple(m_init.shape)}"
        )
    m0 = m_init.float()
    field = b_field.float()
    b_mag = field.norm(dim=0, keepdim=True)
    active = b_mag > 1.0e-12
    b_hat = field / b_mag.clamp_min(1.0e-12)
    u0 = (m0 * b_hat).sum(dim=0, keepdim=True).clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
    transverse = m0 - u0 * b_hat
    transverse_norm = transverse.norm(dim=0, keepdim=True)
    e1 = transverse / transverse_norm.clamp_min(1.0e-8)

    # A deterministic perpendicular fallback keeps the formula well-defined
    # when a spin starts almost exactly parallel to the field.
    axis_x = torch.zeros_like(b_hat)
    axis_x[0] = 1.0
    axis_y = torch.zeros_like(b_hat)
    axis_y[1] = 1.0
    fallback_axis = torch.where((b_hat[0:1].abs() < 0.9).expand_as(b_hat), axis_x, axis_y)
    fallback = torch.cross(b_hat, fallback_axis, dim=0)
    fallback = fallback / fallback.norm(dim=0, keepdim=True).clamp_min(1.0e-8)
    e1 = torch.where((transverse_norm > 1.0e-8).expand_as(e1), e1, fallback)
    e2 = torch.cross(b_hat, e1, dim=0)

    phase = (
        2.0
        * math.pi
        * float(gamma_hz_per_t)
        * b_mag
        * float(dt_s)
        / (1.0 + float(alpha) ** 2)
    )
    damp = float(alpha) * phase
    u = torch.tanh(torch.atanh(u0) + damp)
    radius = (1.0 - u.square()).clamp_min(0.0).sqrt()
    updated = u * b_hat + radius * (
        torch.cos(phase) * e1 + float(sign) * torch.sin(phase) * e2
    )
    updated = normalize_spin(updated.movedim(0, -1)).movedim(-1, 0)
    return torch.where(active.expand_as(updated), updated, m0)


def _normalize_anchor_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode == "first":
        mode = "start"
    if mode not in {"start", "random"}:
        raise ValueError(f"anchor_mode must be start|random, got {mode!r}")
    return mode


def _normalize_condition_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in {"v2", "legacy_alpha"}:
        raise ValueError(f"condition_mode must be v2|legacy_alpha, got {mode!r}")
    return mode


def _normalize_segment_time_range_ns(
    value: list[float] | tuple[float, float] | None,
) -> tuple[float, float] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError("segment_time_range_ns must be [start_ns, end_ns]")
    start_ns = float(value[0])
    end_ns = float(value[1])
    if not math.isfinite(start_ns) or not math.isfinite(end_ns):
        raise ValueError("segment_time_range_ns values must be finite")
    if start_ns < 0.0 or end_ns <= start_ns:
        raise ValueError("segment_time_range_ns must satisfy 0 <= start_ns < end_ns")
    return start_ns, end_ns


def _normalize_segment_policy(value: str | None) -> str:
    mode = "none" if value is None else str(value).lower()
    aliases = {
        "off": "none",
        "disabled": "none",
        "false": "none",
        "no_cross_control": "control",
        "control_change": "control",
        "control_changes": "control",
        "drive": "control",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"none", "control"}:
        raise ValueError(
            "segment_policy must be none|control "
            f"(or alias no_cross_control), got {value!r}"
        )
    return mode


def _normalize_segment_drive_filter(value: str | None) -> str:
    mode = "all" if value is None else str(value).lower()
    aliases = {
        "none": "all",
        "off": "all",
        "disabled": "all",
        "zero_j": "no_j",
        "zero_current": "no_j",
        "no_current": "no_j",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"all", "no_j"}:
        raise ValueError(
            "segment drive_filter must be all|no_j "
            f"(or alias zero_j), got {value!r}"
        )
    return mode


_SEGMENT_J_KEYS: tuple[str, ...] = (
    "J_A_per_m2",
    "J_vector_A_per_m2",
    "charge_current_vector_A_per_m2",
    "solver_J_vector_A_per_m2",
    "current_a_m2",
    "j_sot_eff_a_per_m2",
    "j_amp_a_per_m2",
)


def _segment_j_max_abs_a_m2(segment: dict[str, Any]) -> float:
    """Return the largest actual current amplitude encoded by a v4 segment.

    Both global and per-region values are checked. Torque-family intent flags
    are deliberately ignored: a sampled SOT/STT family with an inactive,
    exactly-zero current is a valid no-J segment.
    """

    drive = segment.get("drive") or {}
    if not isinstance(drive, dict):
        return 0.0
    values: list[float] = []
    sources = [drive, *(drive.get("region_values") or [])]
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in _SEGMENT_J_KEYS:
            value = source.get(key)
            if value is None:
                continue
            components = value if isinstance(value, (list, tuple)) else (value,)
            for component in components:
                scalar = float(component)
                if not math.isfinite(scalar):
                    return float("inf")
                values.append(abs(scalar))
    return max(values, default=0.0)


def _save_step_ps_from_params(params: dict[str, Any], default: float = 25.0) -> float:
    """Resolve ``save_step_ps`` from per-run params, tolerating multiple key names."""
    for key in ("save_step_ps", "save_ps", "save_dt_ps", "dt_save_ps"):
        v = params.get(key)
        if v is not None and float(v) > 0.0:
            return float(v)
    time_cfg = params.get("time")
    if isinstance(time_cfg, dict):
        v = time_cfg.get("save_dt_s")
        if v is not None and float(v) > 0.0:
            return float(v) * 1.0e12
    return float(default)


def _frame_index(path: Path) -> int | None:
    m = _FRAME_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1))


@dataclass(frozen=True)
class FixedTimeSample:
    record: TrajectoryRecord
    frame_init: int
    frame_target: int
    save_step_ps: float
    t_end_ns: float


def _ranges_from_values(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    ranges: list[tuple[int, int]] = []
    lo = hi = int(values[0])
    for value in values[1:]:
        value = int(value)
        if value == hi + 1:
            hi = value
        else:
            ranges.append((lo, hi))
            lo = hi = value
    ranges.append((lo, hi))
    return ranges


class FixedTimePairDataset(Dataset):
    """Sample (M_init, M_t_end) pairs at the fixed-time targets of plan2.

    Each ``__getitem__`` draws a random trajectory and a random ``t_end`` from
    those whose target frame is present in the run. ``anchor_mode='start'``
    anchors at the first valid frame; ``anchor_mode='random'`` draws any frame
    whose fixed-time target remains within the same trajectory. Optional
    ``segment_time_range_ns=[start, end]`` restricts the whole sampled segment
    to that time window: ``frame_init >= start`` and ``frame_target <= end``.
    ``condition_mode`` selects either the v2 vector field condition or the
    legacy alpha scalar ``b_z`` construction.
    """

    def __init__(
        self,
        records: list[TrajectoryRecord],
        t_end_ns: list[float] | tuple[float, ...] = DEFAULT_T_END_NS,
        lattice_size: int = 256,
        samples_per_epoch: int = 100_000,
        seed: int = 0,
        memmap_manifest: str | Path | None = None,
        augment: bool = False,
        augment_rot90: bool = True,
        augment_spin_flip_prob: float = 0.0,
        compute_omega_target: bool = False,
        alpha_t_end_ns_max: float = 1.0,
        anchor_mode: str = "start",
        condition_mode: str = "v2",
        current_time_mode: str = "t0",
        segment_time_range_ns: list[float] | tuple[float, float] | None = None,
        rollout_steps: int = 0,
        rollout_dt_ns: float | None = None,
        segment_pair_mode: bool = False,
        segment_pair_include_first: bool = False,
        segment_pair_predicted_inits: int = 1,
        segment_pair_predicted_probability: float = 0.5,
        segment_pair_sampling_mode: str = "legacy",
        t_end_probs: list[float] | tuple[float, ...] | None = None,
        aux_t_end_ns: list[float] | tuple[float, ...] | None = None,
        action_matching: dict[str, Any] | None = None,
        segment_policy: str | None = None,
        segment_drive_filter: str | None = None,
        segment_j_atol_a_m2: float = 0.0,
        field_cache_size: int = 512,
        spatial_cond_fields: list[str] | tuple[str, ...] | None = None,
        zeeman_precondition: dict[str, Any] | None = None,
        quality_sampling: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not records:
            raise ValueError("FixedTimePairDataset requires at least one record")
        self.records = list(records)
        self.t_end_ns = tuple(float(t) for t in t_end_ns)
        self.aux_t_end_ns = tuple(float(t) for t in (aux_t_end_ns or ()))
        self.t_end_probs = None
        if t_end_probs is not None:
            probs = np.asarray(t_end_probs, dtype=np.float64)
            if probs.shape != (len(self.t_end_ns),):
                raise ValueError("t_end_probs must have the same length as t_end_ns")
            if np.any(probs < 0.0) or float(probs.sum()) <= 0.0:
                raise ValueError("t_end_probs must be non-negative and sum to > 0")
            self.t_end_probs = (probs / probs.sum()).astype(np.float64)
        self.lattice_size = int(lattice_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.augment_rot90 = bool(augment_rot90)
        self.augment_spin_flip_prob = float(augment_spin_flip_prob)
        self.anchor_mode = _normalize_anchor_mode(anchor_mode)
        self.condition_mode = _normalize_condition_mode(condition_mode)
        # ``t0`` (default, legacy) always reads the control map at the canonical
        # t=0 state. ``anchor`` reads it at the anchor's own absolute time.
        # ``segment`` is the control-segmented mode: every pair is constrained
        # to a single drive stage and uses that stage's boundary time as its
        # parameter lookup, so the condition means "stage parameters" rather
        # than a continuously queried current-time current.
        self.current_time_mode = str(current_time_mode).lower()
        if self.current_time_mode not in {"t0", "anchor", "segment"}:
            raise ValueError(
                f"current_time_mode must be t0|anchor|segment, got {current_time_mode!r}"
            )
        self.segment_policy = _normalize_segment_policy(segment_policy)
        self.segment_drive_filter = _normalize_segment_drive_filter(segment_drive_filter)
        self.segment_j_atol_a_m2 = float(segment_j_atol_a_m2)
        if not math.isfinite(self.segment_j_atol_a_m2) or self.segment_j_atol_a_m2 < 0.0:
            raise ValueError("segment j_atol_a_m2 must be finite and >= 0")
        if self.segment_policy != "none" and self.current_time_mode not in {"anchor", "segment"}:
            raise ValueError(
                "segment_policy='control' requires current_time_mode='segment' "
                "or 'anchor' so each segment uses its own control parameters"
            )
        if self.current_time_mode == "segment" and self.segment_policy == "none":
            raise ValueError("current_time_mode='segment' requires segment_policy='control'")
        if self.segment_drive_filter != "all" and self.segment_policy == "none":
            raise ValueError("segment drive_filter requires segment_policy='control'")
        self.segment_time_range_ns = _normalize_segment_time_range_ns(segment_time_range_ns)
        # Mode α rotation-vector target is expensive (per-site arccos / atan2);
        # computing it for all buckets in non-α training wastes ~20% CPU time.
        self.compute_omega_target = bool(compute_omega_target)
        self.alpha_t_end_ns_max = float(alpha_t_end_ns_max)
        self.field_cache_size = max(0, int(field_cache_size))
        self.spatial_cond_fields = tuple(str(x) for x in (spatial_cond_fields or ()))
        zeeman_cfg = zeeman_precondition or {}
        self.zeeman_precondition_enabled = bool(zeeman_cfg.get("enabled", False))
        self.zeeman_precondition_min_cycles = float(zeeman_cfg.get("min_cycles", 2.0))
        self.zeeman_precondition_gamma_hz_per_t = float(
            zeeman_cfg.get("gamma_hz_per_t", 28.0e9)
        )
        self.zeeman_precondition_spatial_field = bool(
            zeeman_cfg.get("use_spatial_field", True)
        )
        self.zeeman_precondition_sign = float(zeeman_cfg.get("sign", 1.0))
        if self.zeeman_precondition_min_cycles < 0.0:
            raise ValueError("data.zeeman_precondition.min_cycles must be >= 0")
        if self.zeeman_precondition_gamma_hz_per_t <= 0.0:
            raise ValueError("data.zeeman_precondition.gamma_hz_per_t must be > 0")
        if self.zeeman_precondition_sign not in {-1.0, 1.0}:
            raise ValueError("data.zeeman_precondition.sign must be -1 or +1")
        if self.zeeman_precondition_enabled and self.aux_t_end_ns:
            raise ValueError(
                "data.zeeman_precondition is not yet compatible with data.aux_t_end_ns"
            )
        am_cfg = action_matching or {}
        self.action_matching_enabled = bool(am_cfg.get("enabled", False))
        if self.zeeman_precondition_enabled and self.action_matching_enabled:
            raise ValueError(
                "data.zeeman_precondition is not yet compatible with action_matching"
            )
        self.quality_sampling = normalize_quality_sampling_config(quality_sampling)
        self.quality_sampling_enabled = bool(self.quality_sampling["enabled"])
        self._quality_segment_entries: dict[tuple[str, int], dict[str, Any]] = {}
        self._quality_cache_summary: dict[str, Any] = {}
        self.action_min_t_end_ns = (
            None
            if am_cfg.get("min_t_end_ns") is None
            else float(am_cfg.get("min_t_end_ns"))
        )
        self.action_mid_fractions = tuple(float(x) for x in am_cfg.get("mid_fractions", (1.0 / 3.0, 2.0 / 3.0)))
        if not self.action_mid_fractions:
            self.action_mid_fractions = (0.5,)
        self.action_diff_radius_frames = max(1, int(am_cfg.get("diff_radius_frames", 1)))
        self._sources_cache: OrderedDict[int, SpatialFieldSources] = OrderedDict()
        self._defect_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._pulse_cache: OrderedDict[int, tuple[float, float] | None] = OrderedDict()
        self._control_segment_cache: OrderedDict[tuple[int, float], tuple[tuple[float, float], ...]] = OrderedDict()
        self._j_field_cache: OrderedDict[tuple[int, float], torch.Tensor] = OrderedDict()
        self._control_grid_cache: OrderedDict[tuple[int, float], torch.Tensor] = OrderedDict()
        self._spatial_field_cache: OrderedDict[tuple[int, float, tuple[str, ...]], dict[str, torch.Tensor]] = OrderedDict()
        self._memmap = None
        if memmap_manifest is not None:
            from skyrmion_cfm.data.memmap import OVFMemmapStore

            self._memmap = OVFMemmapStore(memmap_manifest)
        # Per-record cache: list of (t_end_ns, frame_offset, save_step_ps).
        # ``_choice_start_ranges`` is parallel to ``_record_choices`` and pins
        # segment-boundary targets to their own stage start when the boundary
        # duration is not one of the configured YAML buckets.
        self._record_choices: list[list[tuple[float, int, float]]] = []
        self._choice_start_ranges: list[list[list[tuple[int, int]]]] = []
        for rec_idx, rec in enumerate(self.records):
            choices, ranges = self._enumerate_targets(rec_idx, rec)
            self._record_choices.append(choices)
            self._choice_start_ranges.append(ranges)
        valid = [i for i, choices in enumerate(self._record_choices) if choices]
        if not valid:
            raise ValueError("No trajectory has frames at any of the requested t_end values")
        self._valid_record_idx = valid
        self._segment_boundary_choice_cache: dict[int, list[int]] = {}
        self._segment_boundary_spec_cache: dict[int, list[tuple[int, int]]] = {}
        # Rollout (pushforward) clip sampling. ``rollout_steps=0`` (default)
        # leaves the single-pair behaviour untouched. When >0 each draw returns
        # a stack of K consecutive hops at the fixed ``rollout_dt_ns`` bucket so
        # the training loop can roll the model forward K-1 steps (no-grad) and
        # supervise the last hop against the true frame.
        self.rollout_steps = int(rollout_steps)
        self.rollout_dt_ns = None if rollout_dt_ns is None else float(rollout_dt_ns)
        self.segment_pair_mode = bool(segment_pair_mode)
        self.segment_pair_include_first = bool(segment_pair_include_first)
        self.segment_pair_predicted_inits = int(segment_pair_predicted_inits)
        if self.segment_pair_predicted_inits < 1:
            raise ValueError("segment_pair_predicted_inits must be >= 1")
        self.segment_pair_predicted_probability = float(
            segment_pair_predicted_probability
        )
        if not 0.0 <= self.segment_pair_predicted_probability <= 1.0:
            raise ValueError(
                "segment_pair_predicted_probability must be between 0 and 1"
            )
        self.segment_pair_sampling_mode = str(segment_pair_sampling_mode).lower()
        if self.segment_pair_sampling_mode not in {"legacy", "dataset_mixture"}:
            raise ValueError(
                "segment_pair_sampling_mode must be legacy|dataset_mixture"
            )
        self._segment_pair_indices: list[tuple[int, int]] = []
        self._segment_pair_replay_indices: list[tuple[int, int]] = []
        if self.segment_pair_mode:
            if self.segment_policy == "none":
                raise ValueError("segment_pair_mode requires segment_policy='control'")
            for i in self._valid_record_idx:
                n_segments = len(self._segment_boundary_specs(i))
                start = 0 if (self.segment_pair_include_first or self.segment_time_range_ns is not None) else 1
                self._segment_pair_indices.extend((i, seg_idx) for seg_idx in range(start, n_segments))
                self._segment_pair_replay_indices.extend(
                    (i, seg_idx) for seg_idx in range(max(1, start), n_segments)
                )
            if not self._segment_pair_indices:
                raise ValueError("No valid control-segment pairs found for segment_pair_mode")
            if (
                self.segment_pair_sampling_mode == "dataset_mixture"
                and self.segment_pair_predicted_probability > 0.0
                and not self._segment_pair_replay_indices
            ):
                raise ValueError(
                    "dataset_mixture segment sampling has no non-first segments"
                )
        self._rollout_valid_idx: list[int] = []
        if self.rollout_steps > 0 and not self.segment_pair_mode:
            for i in self._valid_record_idx:
                ci = self._rollout_choice_idx(i)
                if ci is None:
                    continue
                offset = self._record_choices[i][ci][1]
                save_step_ps = self._record_choices[i][ci][2]
                if self._valid_rollout_start_ranges(i, offset, save_step_ps):
                    self._rollout_valid_idx.append(i)
            if not self._rollout_valid_idx:
                raise ValueError(
                    "No trajectory is long enough for "
                    f"rollout_steps={self.rollout_steps} at rollout_dt_ns={self.rollout_dt_ns}"
                )
        if self.quality_sampling_enabled and (
            self.segment_pair_mode or self.rollout_steps > 0
        ):
            raise ValueError(
                "data.quality_sampling currently supports ordinary fixed-time pair "
                "training, not rollout or segment_pushforward batches"
            )

    def _rollout_choice_idx(self, rec_idx: int) -> int | None:
        """Index into ``_record_choices[rec_idx]`` for the rollout dt bucket."""
        choices = self._record_choices[rec_idx]
        if not choices:
            return None
        if self.rollout_dt_ns is None:
            # Default: the smallest available bucket for this record.
            return min(range(len(choices)), key=lambda i: choices[i][0])
        for i, c in enumerate(choices):
            if abs(c[0] - self.rollout_dt_ns) < 1e-9:
                return i
        return None

    def _build_clip(self, rec_idx: int, rng: np.random.Generator) -> dict[str, Any]:
        """K consecutive hops at the rollout dt, stacked on a leading hop axis."""
        ci = self._rollout_choice_idx(rec_idx)
        if ci is None:
            raise ValueError(f"record {rec_idx} has no rollout dt bucket")
        offset = self._record_choices[rec_idx][ci][1]
        save_step_ps = self._record_choices[rec_idx][ci][2]
        ranges = self._valid_rollout_start_ranges(rec_idx, offset, save_step_ps)
        if not ranges:
            raise ValueError(f"record {rec_idx} has no valid rollout segment")
        base = self._choose_start_from_ranges(ranges, rng)
        if self.anchor_mode == "start":
            base = ranges[0][0]
        samples = [
            self._build_sample(rec_idx, ci, rng, frame_init=base + i * offset, apply_augment=False)
            for i in range(self.rollout_steps)
        ]
        if self.augment:
            rot_k = int(rng.integers(0, 4)) if self.augment_rot90 else 0
            spin_flip = self.augment_spin_flip_prob > 0.0 and rng.random() < self.augment_spin_flip_prob
            for sample in samples:
                self._augment_sample_in_place(sample, rot_k=rot_k, spin_flip=spin_flip)
        out: dict[str, Any] = {}
        for key, v0 in samples[0].items():
            if torch.is_tensor(v0):
                out[key] = torch.stack([s[key] for s in samples], dim=0)
            else:
                out[key] = v0
        return out

    def _segment_in_time_range(self, start_ns: float, end_ns: float) -> bool:
        if self.segment_time_range_ns is None:
            return True
        range_start, range_end = self.segment_time_range_ns
        eps = 1e-9
        return float(start_ns) + eps >= float(range_start) and float(end_ns) <= float(range_end) + eps

    def _segment_boundary_specs(self, rec_idx: int) -> list[tuple[int, int]]:
        cached = self._segment_boundary_spec_cache.get(rec_idx)
        if cached is not None:
            return cached
        rec = self.records[rec_idx]
        save_step_ps = _save_step_ps_from_params(rec.params, default=25.0)
        max_frame = self._n_frames(rec) - 1
        out: list[tuple[int, int]] = []
        for start_ns, end_ns in self._control_segments_for(rec_idx, rec, save_step_ps):
            if not self._segment_in_time_range(float(start_ns), float(end_ns)):
                continue
            start_frame = self._frame_at_or_after(rec, float(start_ns), save_step_ps)
            end_frame = self._frame_at_or_before(rec, float(end_ns), save_step_ps)
            if start_frame is None or end_frame is None:
                continue
            frame_offset = end_frame - start_frame
            if frame_offset <= 0 or end_frame > max_frame:
                continue
            fallback: int | None = None
            for choice_idx, choice in enumerate(self._record_choices[rec_idx]):
                _, offset, _ = choice
                ranges = self._choice_start_ranges[rec_idx][choice_idx]
                if int(offset) != int(frame_offset):
                    continue
                if ranges == [(start_frame, start_frame)]:
                    out.append((choice_idx, start_frame))
                    break
                if (start_frame, start_frame) in ranges and fallback is None:
                    fallback = choice_idx
            else:
                if fallback is not None:
                    out.append((fallback, start_frame))
        self._segment_boundary_spec_cache[rec_idx] = out
        return out

    def _segment_boundary_choice_indices(self, rec_idx: int) -> list[int]:
        cached = self._segment_boundary_choice_cache.get(rec_idx)
        if cached is not None:
            return cached
        out = [choice_idx for choice_idx, _start_frame in self._segment_boundary_specs(rec_idx)]
        self._segment_boundary_choice_cache[rec_idx] = out
        return out

    def _set_augment_metadata(self, sample: dict[str, Any], *, rot_k: int, spin_flip: bool) -> None:
        sample["augment_rot_k"] = torch.tensor(int(rot_k) % 4, dtype=torch.long)
        sample["augment_spin_flip"] = torch.tensor(bool(spin_flip), dtype=torch.bool)

    @staticmethod
    def _rotate_xy_values(x: torch.Tensor, y: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate an xy vector using the same convention as spin/B augmentation."""
        k = int(k) % 4
        x = x.clone()
        y = y.clone()
        if k == 1:
            return -y, x
        if k == 2:
            return -x, -y
        if k == 3:
            return y, -x
        return x, y

    @classmethod
    def _rotate_condition_metadata_in_place(cls, sample: dict[str, Any], *, rot_k: int) -> None:
        """Keep vector/scalar/categorical conditions aligned with rot90 data."""
        k = int(rot_k) % 4
        if not k:
            return
        xy_pairs = (
            ("b_x_t", "b_y_t"),
            ("b_ext_x_t", "b_ext_y_t"),
            ("j_vector_x_a_per_m2", "j_vector_y_a_per_m2"),
            ("charge_current_x_a_per_m2", "charge_current_y_a_per_m2"),
            ("polarization_x", "polarization_y"),
            ("fixed_layer_x", "fixed_layer_y"),
            ("anis_u_x", "anis_u_y"),
            ("cubic_axis_1_x", "cubic_axis_1_y"),
            ("cubic_axis_2_x", "cubic_axis_2_y"),
            ("cubic_axis_3_x", "cubic_axis_3_y"),
        )
        for x_key, y_key in xy_pairs:
            if x_key not in sample or y_key not in sample:
                continue
            sample[x_key], sample[y_key] = cls._rotate_xy_values(
                sample[x_key], sample[y_key], k
            )

        # Local-temperature spot coordinates live in [0, 1]^2 rather than
        # being vectors about the origin, so rotate them around the grid centre.
        x_key = "theta_T_x0_norm"
        y_key = "theta_T_y0_norm"
        if x_key in sample and y_key in sample:
            x_centered = sample[x_key] - 0.5
            y_centered = sample[y_key] - 0.5
            # torch.rot90 acts on (row=y, column=x); normalized map
            # coordinates therefore use the inverse sign of an xy vector.
            x_rot, y_rot = cls._rotate_xy_values(x_centered, y_centered, -k)
            sample[x_key] = x_rot + 0.5
            sample[y_key] = y_rot + 0.5

        # A 90/270-degree lattice rotation exchanges x-only and y-only PBC.
        boundary = sample.get("boundary_mode_onehot")
        if k % 2 and torch.is_tensor(boundary) and boundary.ndim >= 1 and boundary.shape[-1] >= 3:
            boundary = boundary.clone()
            pbc_x = boundary[..., 1].clone()
            boundary[..., 1] = boundary[..., 2]
            boundary[..., 2] = pbc_x
            sample["boundary_mode_onehot"] = boundary

        # Cell dimensions are directional conditions for the optional LLG
        # residual, even though the current v4 data use square in-plane cells.
        if k % 2 and "dx_m" in sample and "dy_m" in sample:
            sample["dx_m"], sample["dy_m"] = sample["dy_m"].clone(), sample["dx_m"].clone()

    @classmethod
    def _rotate_spatial_conditions_in_place(cls, sample: dict[str, Any], *, rot_k: int) -> None:
        """Rotate spatial condition maps not handled by _augment_with_params."""
        k = int(rot_k) % 4
        if not k:
            return
        scalar_fields = (
            "control_grid",
            "j_z_field",
            "b_z_field",
            "temperature_field",
            "msat_field",
            "aex_field",
            "ku1_field",
            "dind_field",
            "alpha_field",
        )
        for key in scalar_fields:
            value = sample.get(key)
            if torch.is_tensor(value):
                sample[key] = torch.rot90(value, k=k, dims=(-2, -1))
        for x_key, y_key in (("j_x_field", "j_y_field"), ("b_x_field", "b_y_field")):
            x_value = sample.get(x_key)
            y_value = sample.get(y_key)
            if not torch.is_tensor(x_value) or not torch.is_tensor(y_value):
                continue
            x_spatial = torch.rot90(x_value, k=k, dims=(-2, -1))
            y_spatial = torch.rot90(y_value, k=k, dims=(-2, -1))
            sample[x_key], sample[y_key] = cls._rotate_xy_values(x_spatial, y_spatial, k)

    def _augment_sample_in_place(self, sample: dict[str, Any], *, rot_k: int, spin_flip: bool) -> None:
        (
            sample["m_init"],
            sample["m_t"],
            sample["defect_field"],
            sample["j_field"],
            sample["b_t"],
            sample["omega_target"],
        ) = self._augment_with_params(
            sample["m_init"],
            sample["m_t"],
            sample["defect_field"],
            sample["j_field"],
            sample["b_t"],
            sample["omega_target"],
            rot_k=rot_k,
            spin_flip=spin_flip,
        )
        self._rotate_spatial_conditions_in_place(sample, rot_k=rot_k)
        self._rotate_condition_metadata_in_place(sample, rot_k=rot_k)
        sample["m0"] = sample["m_init"]
        sample["m1"] = sample["m_t"]
        self._set_augment_metadata(sample, rot_k=rot_k, spin_flip=spin_flip)

    def _build_segment_pair(
        self,
        rec_idx: int,
        segment_idx: int,
        rng: np.random.Generator,
        *,
        apply_augment: bool = True,
        force_predicted: bool | None = None,
    ) -> dict[str, Any]:
        specs = self._segment_boundary_specs(rec_idx)
        if segment_idx < 0 or segment_idx >= len(specs):
            raise ValueError(f"segment index {segment_idx} out of range for record {rec_idx}")
        curr_choice_idx, curr_start_frame = specs[segment_idx]
        curr = self._build_sample(
            rec_idx,
            curr_choice_idx,
            rng,
            frame_init=curr_start_frame,
            apply_augment=False,
        )
        prev_idx = max(0, int(segment_idx) - 1)
        prev_choice_idx, prev_start_frame = specs[prev_idx]
        prev = self._build_sample(
            rec_idx,
            prev_choice_idx,
            rng,
            frame_init=prev_start_frame,
            apply_augment=False,
        )
        if apply_augment and self.augment:
            rot_k = int(rng.integers(0, 4)) if self.augment_rot90 else 0
            spin_flip = self.augment_spin_flip_prob > 0.0 and rng.random() < self.augment_spin_flip_prob
            self._augment_sample_in_place(curr, rot_k=rot_k, spin_flip=spin_flip)
            self._augment_sample_in_place(prev, rot_k=rot_k, spin_flip=spin_flip)
        else:
            self._set_augment_metadata(curr, rot_k=0, spin_flip=False)
            self._set_augment_metadata(prev, rot_k=0, spin_flip=False)
        out = dict(curr)
        out["segment_index"] = torch.tensor(int(segment_idx), dtype=torch.long)
        has_prev = int(segment_idx) > 0
        if force_predicted is None:
            use_predicted = (
                has_prev
                and rng.random() < self.segment_pair_predicted_probability
            )
        else:
            use_predicted = has_prev and bool(force_predicted)
        out["has_prev_segment"] = torch.tensor(has_prev, dtype=torch.bool)
        out["use_predicted_init"] = torch.tensor(use_predicted, dtype=torch.bool)
        prediction_idx = (
            int(rng.integers(0, self.segment_pair_predicted_inits))
            if use_predicted
            else 0
        )
        out["predicted_init_index"] = torch.tensor(prediction_idx, dtype=torch.long)
        for key, value in prev.items():
            if torch.is_tensor(value):
                out[f"prev_{key}"] = value
        return out

    def _enumerate_targets(
        self,
        rec_idx: int | TrajectoryRecord,
        rec: TrajectoryRecord | None = None,
    ) -> tuple[list[tuple[float, int, float]], list[list[tuple[int, int]]]] | list[tuple[float, int, float]]:
        legacy_call = rec is None
        if rec is None:
            rec = rec_idx  # type: ignore[assignment]
            rec_idx = -1
        save_step_ps = _save_step_ps_from_params(rec.params, default=25.0)
        max_frame = self._n_frames(rec) - 1
        choices: list[tuple[float, int, float]] = []
        ranges_by_choice: list[list[tuple[int, int]]] = []
        seen: set[tuple[int, tuple[tuple[int, int], ...]]] = set()
        if not legacy_call and self.segment_policy != "none":
            eps = 1.0e-9
            def _append_choice(choice_t_ns: float, frame: int, ranges: list[tuple[int, int]]) -> None:
                if frame <= 0 or not ranges:
                    return
                if (
                    self.action_matching_enabled
                    and self.action_min_t_end_ns is not None
                    and float(choice_t_ns) < self.action_min_t_end_ns
                ):
                    return
                key = (int(frame), tuple(ranges))
                if key in seen:
                    return
                seen.add(key)
                choices.append((float(choice_t_ns), int(frame), save_step_ps))
                ranges_by_choice.append(ranges)

            if self.anchor_mode == "random":
                for t_ns in self.t_end_ns:
                    frame = int(round(float(t_ns) * 1000.0 / save_step_ps))
                    ranges = self._valid_time_start_ranges(int(rec_idx), float(t_ns), save_step_ps)
                    _append_choice(float(t_ns), max(1, frame), ranges)

            for start_ns, end_ns in self._control_segments_for(int(rec_idx), rec, save_step_ps):
                if not self._segment_in_time_range(float(start_ns), float(end_ns)):
                    continue
                start_frame = self._frame_at_or_after(rec, float(start_ns), save_step_ps)
                end_frame = self._frame_at_or_before(rec, float(end_ns), save_step_ps)
                if start_frame is None or end_frame is None:
                    continue
                if end_frame <= start_frame or end_frame > max_frame:
                    continue
                segment_start_ns = self._frame_time_ns(rec, start_frame, save_step_ps)
                segment_end_ns = self._frame_time_ns(rec, end_frame, save_step_ps)
                target_frames: list[int] = []
                if self.anchor_mode != "random":
                    for t_ns in self.t_end_ns:
                        target_time_ns = segment_start_ns + float(t_ns)
                        if target_time_ns > segment_end_ns + eps:
                            continue
                        target_frame = self._frame_nearest_in_range(
                            rec,
                            target_time_ns,
                            start_frame + 1,
                            end_frame,
                            save_step_ps,
                        )
                        if target_frame is not None and target_frame > start_frame:
                            target_frames.append(int(target_frame))
                target_frames.append(int(end_frame))
                for target_frame in sorted(set(target_frames)):
                    frame = int(target_frame) - int(start_frame)
                    if frame <= 0:
                        continue
                    t_ns = self._duration_ns(rec, start_frame, target_frame, save_step_ps)
                    ranges = [(int(start_frame), int(start_frame))]
                    _append_choice(float(t_ns), frame, ranges)
            return choices, ranges_by_choice
        for t_ns in self.t_end_ns:
            frame = int(round(t_ns * 1000.0 / save_step_ps))
            if legacy_call:
                base = self._base_frame_init_bounds(rec, frame, save_step_ps)
                ranges = [] if base is None else [base]
            else:
                ranges = self._valid_frame_start_ranges(int(rec_idx), frame, save_step_ps)
            if (
                getattr(self, "action_matching_enabled", False)
                and getattr(self, "action_min_t_end_ns", None) is not None
                and float(t_ns) < self.action_min_t_end_ns
            ):
                continue
            if (
                0 < frame <= max_frame
                and ranges
            ):
                key = (int(frame), tuple(ranges))
                if key in seen:
                    continue
                seen.add(key)
                choices.append((float(t_ns), frame, save_step_ps))
                ranges_by_choice.append(ranges)
        if legacy_call:
            return choices
        if self.segment_policy != "none":
            for start_ns, end_ns in self._control_segments_for(rec_idx, rec, save_step_ps):
                if not self._segment_in_time_range(float(start_ns), float(end_ns)):
                    continue
                start_frame = self._frame_at_or_after(rec, float(start_ns), save_step_ps)
                end_frame = self._frame_at_or_before(rec, float(end_ns), save_step_ps)
                if start_frame is None or end_frame is None:
                    continue
                frame = end_frame - start_frame
                if frame <= 0 or end_frame > max_frame:
                    continue
                t_ns = self._duration_ns(rec, start_frame, end_frame, save_step_ps)
                if (
                    self.action_matching_enabled
                    and self.action_min_t_end_ns is not None
                    and t_ns < self.action_min_t_end_ns
                ):
                    continue
                ranges = [(start_frame, start_frame)]
                key = (int(frame), tuple(ranges))
                if key in seen:
                    continue
                seen.add(key)
                choices.append((t_ns, frame, save_step_ps))
                ranges_by_choice.append(ranges)
        return choices, ranges_by_choice

    def _n_frames(self, rec: TrajectoryRecord) -> int:
        if self._memmap is None:
            return rec.n_frames
        try:
            return max(int(self._memmap.n_frames(rec.run_id)), int(rec.n_frames))
        except KeyError:
            return rec.n_frames

    def _frame_time_ns(
        self,
        rec: TrajectoryRecord,
        frame_idx: int,
        save_step_ps: float,
    ) -> float:
        times = getattr(rec, "frame_times_s", None)
        if times is not None and 0 <= int(frame_idx) < len(times):
            return float(times[int(frame_idx)]) * 1.0e9
        return float(frame_idx) * float(save_step_ps) / 1000.0

    def _duration_ns(
        self,
        rec: TrajectoryRecord,
        frame_init: int,
        frame_target: int,
        save_step_ps: float,
    ) -> float:
        return max(
            0.0,
            self._frame_time_ns(rec, frame_target, save_step_ps)
            - self._frame_time_ns(rec, frame_init, save_step_ps),
        )

    def _frame_at_or_after(
        self,
        rec: TrajectoryRecord,
        t_ns: float,
        save_step_ps: float,
    ) -> int | None:
        eps = 1.0e-9
        for idx in range(self._n_frames(rec)):
            if self._frame_time_ns(rec, idx, save_step_ps) + eps >= float(t_ns):
                return idx
        return None

    def _frame_at_or_before(
        self,
        rec: TrajectoryRecord,
        t_ns: float,
        save_step_ps: float,
    ) -> int | None:
        eps = 1.0e-9
        out: int | None = None
        for idx in range(self._n_frames(rec)):
            if self._frame_time_ns(rec, idx, save_step_ps) <= float(t_ns) + eps:
                out = idx
            else:
                break
        return out

    def _frame_nearest_in_range(
        self,
        rec: TrajectoryRecord,
        t_ns: float,
        start_frame: int,
        end_frame: int,
        save_step_ps: float,
    ) -> int | None:
        best: tuple[float, int] | None = None
        for idx in range(max(0, int(start_frame)), min(self._n_frames(rec) - 1, int(end_frame)) + 1):
            dt = abs(self._frame_time_ns(rec, idx, save_step_ps) - float(t_ns))
            if best is None or dt < best[0]:
                best = (dt, idx)
        return None if best is None else best[1]

    def _control_segment_for_start_frame(
        self,
        rec_idx: int,
        rec: TrajectoryRecord,
        save_step_ps: float,
        frame_init: int,
    ) -> tuple[int, float, float, int, int] | None:
        start_time_ns = self._frame_time_ns(rec, frame_init, save_step_ps)
        eps = 1.0e-9
        fallback: tuple[int, float, float, int, int] | None = None
        for segment_idx, (seg_start, seg_end) in enumerate(
            self._control_segments_for(rec_idx, rec, save_step_ps)
        ):
            seg_start_frame = self._frame_at_or_after(rec, float(seg_start), save_step_ps)
            seg_end_frame = self._frame_at_or_before(rec, float(seg_end), save_step_ps)
            if seg_start_frame is None or seg_end_frame is None:
                continue
            item = (int(segment_idx), float(seg_start), float(seg_end), int(seg_start_frame), int(seg_end_frame))
            if abs(start_time_ns - float(seg_start)) <= eps:
                return item
            if float(seg_start) - eps <= start_time_ns < float(seg_end) - eps:
                fallback = item
        return fallback

    def _target_frame_for_duration(
        self,
        rec_idx: int,
        rec: TrajectoryRecord,
        save_step_ps: float,
        frame_init: int,
        duration_ns: float,
    ) -> int | None:
        if float(duration_ns) <= 0.0:
            return None
        start_time_ns = self._frame_time_ns(rec, frame_init, save_step_ps)
        target_time_ns = start_time_ns + float(duration_ns)
        max_frame = self._n_frames(rec) - 1
        if self.segment_policy != "none":
            segment = self._control_segment_for_start_frame(rec_idx, rec, save_step_ps, frame_init)
            if segment is None:
                return None
            _segment_idx, _seg_start, _seg_end, _seg_start_frame, seg_end_frame = segment
            max_frame = min(max_frame, seg_end_frame)
        if target_time_ns > self._frame_time_ns(rec, max_frame, save_step_ps) + 1.0e-9:
            return None
        target_frame = self._frame_nearest_in_range(
            rec,
            target_time_ns,
            int(frame_init) + 1,
            max_frame,
            save_step_ps,
        )
        if target_frame is None or target_frame <= frame_init:
            return None
        return int(target_frame)

    def _valid_time_start_ranges(
        self,
        rec_idx: int,
        duration_ns: float,
        save_step_ps: float,
    ) -> list[tuple[int, int]]:
        rec = self.records[rec_idx]
        valid: list[int] = []
        range_start = range_end = None
        if self.segment_time_range_ns is not None:
            range_start, range_end = self.segment_time_range_ns
        for start in range(0, max(0, self._n_frames(rec) - 1)):
            start_ns = self._frame_time_ns(rec, start, save_step_ps)
            if range_start is not None and start_ns + 1.0e-9 < float(range_start):
                continue
            target = self._target_frame_for_duration(rec_idx, rec, save_step_ps, start, duration_ns)
            if target is None:
                continue
            if range_end is not None and self._frame_time_ns(rec, target, save_step_ps) > float(range_end) + 1.0e-9:
                continue
            valid.append(int(start))
        return _ranges_from_values(valid)

    def _base_frame_init_bounds(
        self,
        rec: TrajectoryRecord,
        frame_offset: int,
        save_step_ps: float,
    ) -> tuple[int, int] | None:
        max_start = self._n_frames(rec) - 1 - int(frame_offset)
        valid = list(range(0, max_start + 1))
        if self.segment_time_range_ns is not None:
            start_ns, end_ns = self.segment_time_range_ns
            valid = [
                start
                for start in valid
                if self._frame_time_ns(rec, start, save_step_ps) + 1.0e-9 >= float(start_ns)
                and self._frame_time_ns(rec, start + int(frame_offset), save_step_ps)
                <= float(end_ns) + 1.0e-9
            ]
        if not valid:
            return None
        return min(valid), max(valid)

    def _control_segments_for(
        self,
        rec_idx: int,
        rec: TrajectoryRecord,
        save_step_ps: float,
    ) -> tuple[tuple[float, float], ...]:
        key = (rec_idx, round(float(save_step_ps), 12))
        total_ns = max(0.0, self._frame_time_ns(rec, self._n_frames(rec) - 1, save_step_ps))

        def build_segments() -> tuple[tuple[float, float], ...]:
            segments = control_segments_ns(self._sources_for(rec_idx, rec), total_ns)
            if self.segment_drive_filter == "all":
                return segments
            return tuple(
                (float(start_ns), float(end_ns))
                for start_ns, end_ns in segments
                if self._segment_j_max_abs_for_span(
                    rec_idx,
                    rec,
                    float(start_ns),
                    float(end_ns),
                )
                <= self.segment_j_atol_a_m2
            )

        return self._cache_get_or_put(self._control_segment_cache, key, build_segments)

    def _segment_j_max_abs_for_span(
        self,
        rec_idx: int,
        rec: TrajectoryRecord,
        start_ns: float,
        end_ns: float,
    ) -> float:
        if rec.is_v4:
            segment = v4_metadata.segment_at_time(rec.params, float(start_ns) * 1.0e-9)
            if segment:
                return _segment_j_max_abs_a_m2(segment)
            row = rec.v4_condition_row(float(start_ns) * 1.0e-9, float(end_ns) * 1.0e-9)
            return max(
                abs(float(row.get(key, 0.0)))
                for key in (
                    "current_a_m2",
                    "j_vector_x_a_per_m2",
                    "j_vector_y_a_per_m2",
                    "j_vector_z_a_per_m2",
                    "charge_current_x_a_per_m2",
                    "charge_current_y_a_per_m2",
                    "charge_current_z_a_per_m2",
                )
            )

        current = abs(float(rec.condition_row(dt_s=0.0).get("current_a_m2", 0.0)))
        sources = self._sources_for(rec_idx, rec)
        j_field = self._j_field_for(rec_idx, sources, float(start_ns))
        spatial = float(j_field.abs().max()) if j_field.numel() else 0.0
        return max(current, spatial)

    def _valid_frame_start_ranges(
        self,
        rec_idx: int,
        frame_offset: int,
        save_step_ps: float,
    ) -> list[tuple[int, int]]:
        rec = self.records[rec_idx]
        base = self._base_frame_init_bounds(rec, frame_offset, save_step_ps)
        if base is None:
            return []
        if self.segment_policy == "none":
            return [base]
        base_min, base_max = base
        valid: list[int] = []
        for start_ns, end_ns in self._control_segments_for(rec_idx, rec, save_step_ps):
            for start in range(base_min, base_max + 1):
                target = start + int(frame_offset)
                if (
                    self._frame_time_ns(rec, start, save_step_ps) + 1.0e-9 >= float(start_ns)
                    and self._frame_time_ns(rec, target, save_step_ps) <= float(end_ns) + 1.0e-9
                ):
                    valid.append(start)
        return _ranges_from_values(sorted(set(valid)))

    def _valid_choice_start_ranges(self, rec_idx: int, choice_idx: int) -> list[tuple[int, int]]:
        return list(self._choice_start_ranges[rec_idx][choice_idx])

    def _segment_control_time_ns(
        self,
        rec_idx: int,
        rec: TrajectoryRecord,
        save_step_ps: float,
        frame_init: int,
        frame_target: int,
    ) -> float:
        """Return the control-stage boundary time for a sampled pair.

        Segment intervals are treated as frame-overlapping at the boundary:
        the boundary frame may be the target of the previous stage and the
        initial state of the next stage. Parameter lookup for the next stage
        therefore uses the segment start when ``frame_init`` lands exactly on a
        boundary.
        """
        start_ns = self._frame_time_ns(rec, frame_init, save_step_ps)
        end_ns = self._frame_time_ns(rec, frame_target, save_step_ps)
        eps = 1e-9
        chosen = start_ns
        for seg_start, seg_end in self._control_segments_for(rec_idx, rec, save_step_ps):
            if start_ns + eps >= seg_start and end_ns <= seg_end + eps:
                # Prefer the segment whose start exactly matches a boundary
                # frame, so the shared state becomes the next stage's init.
                if abs(start_ns - seg_start) <= eps:
                    return float(seg_start)
                chosen = float(seg_start)
        return chosen

    def _control_segment_index_for_span(
        self,
        rec_idx: int,
        rec: TrajectoryRecord,
        save_step_ps: float,
        frame_init: int,
        frame_target: int,
    ) -> int:
        start_ns = self._frame_time_ns(rec, frame_init, save_step_ps)
        end_ns = self._frame_time_ns(rec, frame_target, save_step_ps)
        eps = 1e-9
        chosen = 0
        for idx, (seg_start, seg_end) in enumerate(self._control_segments_for(rec_idx, rec, save_step_ps)):
            if start_ns + eps >= seg_start and end_ns <= seg_end + eps:
                if abs(start_ns - seg_start) <= eps:
                    return int(idx)
                chosen = int(idx)
        return chosen

    def _frame_init_bounds(
        self,
        rec: TrajectoryRecord,
        frame_offset: int,
        save_step_ps: float,
    ) -> tuple[int, int] | None:
        """Legacy compatibility wrapper returning the broad envelope."""
        try:
            rec_idx = self.records.index(rec)
        except ValueError:
            return self._base_frame_init_bounds(rec, frame_offset, save_step_ps)
        ranges = self._valid_frame_start_ranges(rec_idx, frame_offset, save_step_ps)
        if not ranges:
            return None
        return ranges[0][0], ranges[-1][1]

    @staticmethod
    def _choose_start_from_ranges(
        ranges: list[tuple[int, int]],
        rng: np.random.Generator,
    ) -> int:
        if not ranges:
            raise ValueError("No valid frame-start ranges")
        sizes = np.asarray([hi - lo + 1 for lo, hi in ranges], dtype=np.int64)
        pick = int(rng.integers(0, int(sizes.sum())))
        cursor = 0
        for (lo, hi), size in zip(ranges, sizes, strict=True):
            if pick < cursor + int(size):
                return int(lo + (pick - cursor))
            cursor += int(size)
        return int(ranges[-1][1])

    def _valid_rollout_start_ranges(
        self,
        rec_idx: int,
        frame_offset: int,
        save_step_ps: float,
    ) -> list[tuple[int, int]]:
        rec = self.records[rec_idx]
        base = self._base_frame_init_bounds(
            rec,
            self.rollout_steps * int(frame_offset),
            save_step_ps,
        )
        if base is None:
            return []
        if self.segment_policy == "none":
            return [base]
        hop_ranges = self._valid_frame_start_ranges(rec_idx, frame_offset, save_step_ps)
        if not hop_ranges:
            return []

        def _hop_ok(start: int) -> bool:
            for lo, hi in hop_ranges:
                if lo <= start <= hi:
                    return True
            return False

        valid: list[int] = []
        for start in range(base[0], base[1] + 1):
            if all(_hop_ok(start + i * int(frame_offset)) for i in range(self.rollout_steps)):
                valid.append(start)
        if not valid:
            return []
        ranges: list[tuple[int, int]] = []
        lo = hi = valid[0]
        for value in valid[1:]:
            if value == hi + 1:
                hi = value
            else:
                ranges.append((lo, hi))
                lo = hi = value
        ranges.append((lo, hi))
        return ranges

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _rng(self, idx: int) -> np.random.Generator:
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        return np.random.default_rng(self.seed + idx * 9973 + worker_id * 104729)

    def _read_frame(self, rec: TrajectoryRecord, frame_idx: int) -> np.ndarray:
        if self._memmap is not None:
            try:
                return self._memmap.get(rec.run_id, frame_idx)
            except (KeyError, IndexError):
                pass
        if 0 <= int(frame_idx) < len(rec.frames):
            return read_ovf(rec.frames[int(frame_idx)])
        # Look up frame path by index within the record. Memmap-built records
        # always have a contiguous prefix so this fallback is rarely hit.
        for path in rec.frames:
            if _frame_index(path) == frame_idx:
                return read_ovf(path)
        # Fall back to positional indexing.
        return read_ovf(rec.frames[frame_idx])

    def _choose_frame_init(
        self,
        rec_idx: int | TrajectoryRecord,
        rec: TrajectoryRecord | None = None,
        frame_offset: int | None = None,
        save_step_ps: float | None = None,
        rng: np.random.Generator | None = None,
        choice_idx: int | None = None,
    ) -> int:
        if rec is None:
            rec = rec_idx  # type: ignore[assignment]
            if frame_offset is None or save_step_ps is None or rng is None:
                raise TypeError("frame_offset, save_step_ps, and rng are required")
            base = self._base_frame_init_bounds(rec, int(frame_offset), float(save_step_ps))
            ranges = [] if base is None else [base]
        else:
            if frame_offset is None or save_step_ps is None or rng is None:
                raise TypeError("frame_offset, save_step_ps, and rng are required")
            ranges = (
                self._valid_choice_start_ranges(int(rec_idx), choice_idx)
                if choice_idx is not None
                else self._valid_frame_start_ranges(int(rec_idx), int(frame_offset), float(save_step_ps))
            )
        if not ranges:
            raise ValueError("No valid segment for the configured segment_time_range_ns")
        if self.anchor_mode == "start":
            return ranges[0][0]
        return self._choose_start_from_ranges(ranges, rng)

    def _t_end_bucket_index(self, t_end_ns: float) -> int:
        if not self.t_end_ns:
            return 0
        return int(min(range(len(self.t_end_ns)), key=lambda i: abs(float(self.t_end_ns[i]) - float(t_end_ns))))

    def _condition_b_t(self, cond_row: dict[str, float]) -> torch.Tensor:
        if self.condition_mode == "legacy_alpha":
            values = [0.0, 0.0, cond_row["b_z_t"]]
        else:
            values = [cond_row["b_x_t"], cond_row["b_y_t"], cond_row["b_z_t"]]
        return torch.tensor(values, dtype=torch.float32)

    def _cache_get_or_put(
        self,
        cache: OrderedDict[Any, Any],
        key: Any,
        build,
    ) -> Any:
        if self.field_cache_size <= 0:
            return build()
        try:
            value = cache.pop(key)
        except KeyError:
            value = build()
            if len(cache) >= self.field_cache_size:
                cache.popitem(last=False)
        cache[key] = value
        return value

    def _sources_for(self, rec_idx: int, rec: TrajectoryRecord) -> SpatialFieldSources:
        return self._cache_get_or_put(
            self._sources_cache,
            rec_idx,
            lambda: SpatialFieldSources.from_run_dir(rec.path),
        )

    def _defect_for(self, rec_idx: int, sources: SpatialFieldSources) -> torch.Tensor:
        return self._cache_get_or_put(
            self._defect_cache,
            rec_idx,
            lambda: torch.from_numpy(load_defect_field(sources, self.lattice_size)).float(),
        )

    def _pulse_window_for(
        self,
        rec_idx: int,
        sources: SpatialFieldSources,
    ) -> tuple[float, float] | None:
        return self._cache_get_or_put(
            self._pulse_cache,
            rec_idx,
            lambda: pulse_window_ns(sources),
        )

    @staticmethod
    def _pulse_overlap_ns(
        start_ns: float,
        end_ns: float,
        pulse_window: tuple[float, float] | None,
    ) -> float:
        if pulse_window is None:
            return 0.0
        pulse_start_ns, pulse_end_ns = pulse_window
        return max(
            0.0,
            min(float(end_ns), pulse_end_ns) - max(float(start_ns), pulse_start_ns),
        )

    def _j_field_for(
        self,
        rec_idx: int,
        sources: SpatialFieldSources,
        t_ns: float,
    ) -> torch.Tensor:
        key = (rec_idx, round(float(t_ns), 12))
        return self._cache_get_or_put(
            self._j_field_cache,
            key,
            lambda: torch.from_numpy(
                build_j_field(sources, self.lattice_size, t_ns=t_ns)
            ).float(),
        )

    def _control_grid_for(
        self,
        rec_idx: int,
        sources: SpatialFieldSources,
        t_ns: float,
    ) -> torch.Tensor:
        from skyrmion_cfm.data.spatial_fields import control_grid_at_time

        key = (rec_idx, round(float(t_ns), 12))
        return self._cache_get_or_put(
            self._control_grid_cache,
            key,
            lambda: torch.from_numpy(control_grid_at_time(sources, t_ns=t_ns)).float(),
        )

    def _extra_spatial_fields_for(
        self,
        rec_idx: int,
        sources: SpatialFieldSources,
        t_ns: float,
    ) -> dict[str, torch.Tensor]:
        fields = tuple(
            key
            for key in self.spatial_cond_fields
            if key not in {"defect_field", "j_field", "control_grid"}
        )
        if not fields:
            return {}
        key = (rec_idx, round(float(t_ns), 12), fields)
        return self._cache_get_or_put(
            self._spatial_field_cache,
            key,
            lambda: {
                name: torch.from_numpy(value).float()
                for name, value in spatial_condition_fields_at_time(
                    sources,
                    self.lattice_size,
                    t_ns,
                    fields,
                ).items()
            },
        )

    def _apply_zeeman_precondition(self, sample: dict[str, Any]) -> None:
        """Replace the observed start by a gated analytic Zeeman reference."""
        if not self.zeeman_precondition_enabled:
            return
        observed = sample["m_init"]
        sample["m_observed_init"] = observed
        b_t = sample["b_t"].float()
        dt_s = float(sample["t_end_s"].item())
        cycles = self.zeeman_precondition_gamma_hz_per_t * float(b_t.norm().item()) * dt_s
        apply_reference = (
            cycles >= self.zeeman_precondition_min_cycles
            and float(b_t.norm().item()) > 1.0e-12
        )
        sample["zeeman_reference_cycles"] = torch.tensor(cycles, dtype=torch.float32)
        sample["zeeman_reference_applied"] = torch.tensor(apply_reference, dtype=torch.bool)
        if not apply_reference:
            return

        b_field: torch.Tensor | None = None
        if self.zeeman_precondition_spatial_field:
            parts: list[torch.Tensor] = []
            for key in ("b_x_field", "b_y_field", "b_z_field"):
                value = sample.get(key)
                if not torch.is_tensor(value):
                    parts = []
                    break
                if value.ndim == 2:
                    value = value.unsqueeze(0)
                if value.ndim != 3 or value.shape[0] != 1:
                    parts = []
                    break
                parts.append(value)
            if len(parts) == 3:
                b_field = torch.cat(parts, dim=0)
        if b_field is None:
            b_field = b_t[:, None, None].expand(3, *observed.shape[-2:])
        alpha_value = sample.get("alpha_t", sample.get("alpha"))
        alpha = float(alpha_value.item()) if torch.is_tensor(alpha_value) else 0.0
        reference = _analytic_zeeman_reference(
            observed,
            b_field,
            alpha=alpha,
            dt_s=dt_s,
            gamma_hz_per_t=self.zeeman_precondition_gamma_hz_per_t,
            sign=self.zeeman_precondition_sign,
        )
        sample["m_init"] = reference
        if self.compute_omega_target and float(sample["t_end_ns"].item()) <= self.alpha_t_end_ns_max:
            sample["omega_target"] = log_map_chw(
                reference.unsqueeze(0), sample["m_t"].unsqueeze(0)
            ).squeeze(0)

    def _build_sample(
        self,
        rec_idx: int,
        choice_idx: int,
        rng: np.random.Generator,
        frame_init: int | None = None,
        apply_augment: bool = True,
        frame_target_override: int | None = None,
    ) -> dict[str, Any]:
        rec = self.records[rec_idx]
        choices = self._record_choices[rec_idx]
        t_end_ns, frame_offset, save_step_ps = choices[choice_idx]
        if frame_init is None:
            frame_init = self._choose_frame_init(
                rec_idx,
                rec,
                frame_offset,
                save_step_ps,
                rng,
                choice_idx=choice_idx,
            )
        if frame_target_override is not None:
            frame_target = int(frame_target_override)
        elif self.segment_policy != "none":
            frame_target = self._target_frame_for_duration(
                rec_idx,
                rec,
                save_step_ps,
                int(frame_init),
                float(t_end_ns),
            )
            if frame_target is None:
                frame_target = int(frame_init) + int(frame_offset)
        else:
            frame_target = int(frame_init) + int(frame_offset)
        segment_start_ns = self._frame_time_ns(rec, frame_init, save_step_ps)
        segment_end_ns = self._frame_time_ns(rec, frame_target, save_step_ps)
        actual_t_end_ns = max(0.0, segment_end_ns - segment_start_ns)
        t_end_ns = actual_t_end_ns
        # Time (ns) at which to sample the control current. ``t0`` keeps the
        # legacy canonical-state behaviour; ``anchor`` uses the anchor frame's
        # own time so post-pulse anchors see zero drive.
        if self.current_time_mode == "segment":
            current_t_ns = self._segment_control_time_ns(
                rec_idx,
                rec,
                save_step_ps,
                frame_init,
                frame_target,
            )
        elif self.current_time_mode == "anchor":
            current_t_ns = segment_start_ns
        else:
            current_t_ns = 0.0
        control_segment_index = (
            self._control_segment_index_for_span(
                rec_idx,
                rec,
                save_step_ps,
                frame_init,
                frame_target,
            )
            if self.segment_policy != "none"
            else 0
        )
        m_init_np = self._read_frame(rec, frame_init)
        m_t_np = self._read_frame(rec, frame_target)
        m_init = torch.from_numpy(np.ascontiguousarray(m_init_np)).float()
        m_t = torch.from_numpy(np.ascontiguousarray(m_t_np)).float()
        m_init = normalize_spin(m_init.movedim(0, -1)).movedim(-1, 0)
        m_t = normalize_spin(m_t.movedim(0, -1)).movedim(-1, 0)
        sources = self._sources_for(rec_idx, rec)
        pulse_window = self._pulse_window_for(rec_idx, sources)
        drive_time_ns = min(
            self._pulse_overlap_ns(segment_start_ns, segment_end_ns, pulse_window),
            float(t_end_ns),
        )
        relax_time_ns = max(0.0, float(t_end_ns) - drive_time_ns)
        drive_fraction = drive_time_ns / max(float(t_end_ns), 1e-12)
        defect = self._defect_for(rec_idx, sources)
        # j_field is the SOT region-current map projected onto the lattice,
        # sampled at ``current_t_ns``. In ``segment`` mode this is the segment
        # boundary time, i.e. the stage's fixed control parameter.
        j_field = self._j_field_for(rec_idx, sources, current_t_ns)
        # 8x8 control grid (yet-unrasterized) is returned for condition-vector
        # J embeddings; keep it aligned with j_field under data augmentation.
        control_grid = self._control_grid_for(rec_idx, sources, current_t_ns)
        if rec.is_v4:
            cond_row = rec.v4_condition_row(segment_start_ns * 1e-9, segment_end_ns * 1e-9)
            drive_time_ns = float(t_end_ns) if float(cond_row.get("drive_active_flag", 0.0)) > 0.0 else 0.0
            relax_time_ns = max(0.0, float(t_end_ns) - drive_time_ns)
            drive_fraction = drive_time_ns / max(float(t_end_ns), 1e-12)
        else:
            cond_row = rec.condition_row(dt_s=t_end_ns * 1e-9)
        b_t = self._condition_b_t(cond_row)
        # Keep the scalar current consistent with the spatial map: when the
        # selected stage is past the pulse the spatial j_field is all-zero, so
        # the scalar amplitude is zeroed too (only outside legacy ``t0`` mode).
        current_amp = float(cond_row["current_a_m2"])
        if (not rec.is_v4) and self.current_time_mode != "t0" and not bool(j_field.any()):
            current_amp = 0.0
        extra_spatial = self._extra_spatial_fields_for(rec_idx, sources, current_t_ns)
        # Optional Mode α target. We skip the per-site log map entirely when α
        # is disabled (default) or the bucket is masked by alpha_t_end_ns_max to
        # save ~20% CPU per worker.
        if self.compute_omega_target and t_end_ns <= self.alpha_t_end_ns_max:
            omega_target = log_map_chw(m_init.unsqueeze(0), m_t.unsqueeze(0)).squeeze(0)
        else:
            omega_target = torch.zeros_like(m_init)
        aux_targets: list[torch.Tensor] = []
        aux_omegas: list[torch.Tensor] = []
        aux_tensors: dict[str, list[torch.Tensor]] = {
            "t_end_ns": [],
            "t_end_s": [],
            "t_end_index": [],
            "frame_target": [],
            "drive_time_s": [],
            "relax_time_s": [],
            "drive_fraction": [],
        }
        for aux_ns in self.aux_t_end_ns:
            aux_offset = int(round(float(aux_ns) * 1000.0 / save_step_ps))
            aux_target_frame = (
                self._target_frame_for_duration(
                    rec_idx,
                    rec,
                    save_step_ps,
                    int(frame_init),
                    float(aux_ns),
                )
                if self.segment_policy != "none"
                else int(frame_init) + aux_offset
            )
            if aux_offset <= 0 or aux_target_frame is None or aux_target_frame >= self._n_frames(rec):
                continue
            aux_t_np = self._read_frame(rec, aux_target_frame)
            aux_t = torch.from_numpy(np.ascontiguousarray(aux_t_np)).float()
            aux_t = normalize_spin(aux_t.movedim(0, -1)).movedim(-1, 0)
            if self.compute_omega_target and aux_ns <= self.alpha_t_end_ns_max:
                aux_omega = log_map_chw(m_init.unsqueeze(0), aux_t.unsqueeze(0)).squeeze(0)
            else:
                aux_omega = torch.zeros_like(m_init)
            aux_start_ns = segment_start_ns
            aux_end_ns = self._frame_time_ns(rec, aux_target_frame, save_step_ps)
            aux_actual_ns = max(0.0, aux_end_ns - aux_start_ns)
            aux_drive_ns = min(
                self._pulse_overlap_ns(aux_start_ns, aux_end_ns, pulse_window),
                float(aux_actual_ns),
            )
            aux_relax_ns = max(0.0, float(aux_actual_ns) - aux_drive_ns)
            aux_targets.append(aux_t)
            aux_omegas.append(aux_omega)
            aux_tensors["t_end_ns"].append(torch.tensor(aux_actual_ns, dtype=torch.float32))
            aux_tensors["t_end_s"].append(torch.tensor(aux_actual_ns * 1e-9, dtype=torch.float32))
            aux_tensors["t_end_index"].append(
                torch.tensor(
                    self._t_end_bucket_index(aux_actual_ns),
                    dtype=torch.long,
                )
            )
            aux_tensors["frame_target"].append(torch.tensor(aux_target_frame, dtype=torch.long))
            aux_tensors["drive_time_s"].append(torch.tensor(aux_drive_ns * 1e-9, dtype=torch.float32))
            aux_tensors["relax_time_s"].append(torch.tensor(aux_relax_ns * 1e-9, dtype=torch.float32))
            aux_tensors["drive_fraction"].append(
                torch.tensor(aux_drive_ns / max(float(aux_actual_ns), 1e-12), dtype=torch.float32)
            )
        action_states: list[torch.Tensor] = []
        action_velocities: list[torch.Tensor] = []
        action_taus: list[torch.Tensor] = []
        action_valid: list[torch.Tensor] = []
        if self.action_matching_enabled:
            for frac in self.action_mid_fractions:
                tau_mid = max(0.0, min(1.0, float(frac)))
                mid_offset = int(round(tau_mid * frame_offset))
                diff_radius = self.action_diff_radius_frames
                prev_offset = mid_offset - diff_radius
                next_offset = mid_offset + diff_radius
                valid = (
                    mid_offset > 0
                    and mid_offset < frame_offset
                    and prev_offset >= 0
                    and next_offset <= frame_offset
                )
                if valid:
                    mid_np = self._read_frame(rec, frame_init + mid_offset)
                    prev_np = self._read_frame(rec, frame_init + prev_offset)
                    next_np = self._read_frame(rec, frame_init + next_offset)
                    mid_m = torch.from_numpy(np.ascontiguousarray(mid_np)).float()
                    prev_m = torch.from_numpy(np.ascontiguousarray(prev_np)).float()
                    next_m = torch.from_numpy(np.ascontiguousarray(next_np)).float()
                    mid_m = normalize_spin(mid_m.movedim(0, -1)).movedim(-1, 0)
                    prev_m = normalize_spin(prev_m.movedim(0, -1)).movedim(-1, 0)
                    next_m = normalize_spin(next_m.movedim(0, -1)).movedim(-1, 0)
                    omega_mid = log_map_chw(m_init.unsqueeze(0), mid_m.unsqueeze(0)).squeeze(0)
                    omega_prev = log_map_chw(m_init.unsqueeze(0), prev_m.unsqueeze(0)).squeeze(0)
                    omega_next = log_map_chw(m_init.unsqueeze(0), next_m.unsqueeze(0)).squeeze(0)
                    velocity = (omega_next - omega_prev) * (float(frame_offset) / float(2 * diff_radius))
                else:
                    omega_mid = torch.zeros_like(m_init)
                    velocity = torch.zeros_like(m_init)
                action_states.append(omega_mid)
                action_velocities.append(velocity)
                action_taus.append(torch.tensor(tau_mid, dtype=torch.float32))
                action_valid.append(torch.tensor(valid, dtype=torch.bool))
        if apply_augment and self.augment:
            rot_k = int(rng.integers(0, 4)) if self.augment_rot90 else 0
            spin_flip = self.augment_spin_flip_prob > 0.0 and rng.random() < self.augment_spin_flip_prob
            m_init, m_t, defect, j_field, b_t, omega_target = self._augment_with_params(
                m_init,
                m_t,
                defect,
                j_field,
                b_t,
                omega_target,
                rot_k=rot_k,
                spin_flip=spin_flip,
            )
            if rot_k:
                control_grid = torch.rot90(control_grid, k=rot_k, dims=(-2, -1))
                extra_spatial = {
                    key: torch.rot90(value, k=rot_k, dims=(-2, -1))
                    for key, value in extra_spatial.items()
                }
                for x_key, y_key in (("b_x_field", "b_y_field"), ("j_x_field", "j_y_field")):
                    if x_key in extra_spatial and y_key in extra_spatial:
                        vx = extra_spatial[x_key].clone()
                        vy = extra_spatial[y_key].clone()
                        if rot_k == 1:
                            extra_spatial[x_key], extra_spatial[y_key] = -vy, vx
                        elif rot_k == 2:
                            extra_spatial[x_key], extra_spatial[y_key] = -vx, -vy
                        elif rot_k == 3:
                            extra_spatial[x_key], extra_spatial[y_key] = vy, -vx
            if spin_flip:
                for key in ("b_x_field", "b_y_field", "b_z_field"):
                    if key in extra_spatial:
                        extra_spatial[key] = -extra_spatial[key]
            aux_aug: list[torch.Tensor] = []
            aux_omega_aug: list[torch.Tensor] = []
            for aux_t, aux_omega in zip(aux_targets, aux_omegas, strict=True):
                aux_t_aug = aux_t
                aux_omega_aug_i = aux_omega
                if rot_k:
                    aux_t_aug = self._rotate_spin_xy(torch.rot90(aux_t_aug, k=rot_k, dims=(-2, -1)), rot_k)
                    aux_omega_aug_i = self._rotate_spin_xy(
                        torch.rot90(aux_omega_aug_i, k=rot_k, dims=(-2, -1)),
                        rot_k,
                    )
                if spin_flip:
                    aux_t_aug = -aux_t_aug
                    aux_omega_aug_i = -aux_omega_aug_i
                aux_aug.append(aux_t_aug)
                aux_omega_aug.append(aux_omega_aug_i)
            aux_targets = aux_aug
            aux_omegas = aux_omega_aug
            action_states = [
                self._rotate_spin_xy(torch.rot90(x, k=rot_k, dims=(-2, -1)), rot_k) if rot_k else x
                for x in action_states
            ]
            action_velocities = [
                self._rotate_spin_xy(torch.rot90(x, k=rot_k, dims=(-2, -1)), rot_k) if rot_k else x
                for x in action_velocities
            ]
        sample = {
            "m_init": m_init,
            "m_t": m_t,
            "defect_field": defect,
            "j_field": j_field,
            "control_grid": control_grid,
            "omega_target": omega_target,
            "t_end_ns": torch.tensor(t_end_ns, dtype=torch.float32),
            "t_end_s": torch.tensor(t_end_ns * 1e-9, dtype=torch.float32),
            "t_end_index": torch.tensor(
                self._t_end_bucket_index(t_end_ns),
                dtype=torch.long,
            ),
            "save_step_ps": torch.tensor(save_step_ps, dtype=torch.float32),
            "frame_init": torch.tensor(frame_init, dtype=torch.long),
            "frame_target": torch.tensor(frame_target, dtype=torch.long),
            "frame_init_time_ns": torch.tensor(segment_start_ns, dtype=torch.float32),
            "frame_target_time_ns": torch.tensor(segment_end_ns, dtype=torch.float32),
            "control_time_ns": torch.tensor(current_t_ns, dtype=torch.float32),
            "control_segment_index": torch.tensor(control_segment_index, dtype=torch.long),
            "temp_k": torch.tensor(cond_row["temp_k"], dtype=torch.float32),
            "b_t": b_t,
            "current_a_m2": torch.tensor(current_amp, dtype=torch.float32),
            "drive_time_s": torch.tensor(drive_time_ns * 1e-9, dtype=torch.float32),
            "relax_time_s": torch.tensor(relax_time_ns * 1e-9, dtype=torch.float32),
            "drive_fraction": torch.tensor(drive_fraction, dtype=torch.float32),
            "run_id": rec.run_id,
        }
        sample.update(extra_spatial)
        if rec.is_v4:
            sample.update(rec.v4_categorical_tensors(segment_start_ns * 1e-9))
            for key in SCALAR_CONDITION_KEYS:
                if key in cond_row and key not in sample:
                    sample[key] = torch.tensor(float(cond_row[key]), dtype=torch.float32)
            # Auxiliary drive scalars used only by opt-in physics features.
            # Keep them outside SCALAR_CONDITION_KEYS so adding one does not
            # resize the persistent ConditionStats buffers of old checkpoints.
            for key in ("beta_zl",):
                if key in cond_row and key not in sample:
                    sample[key] = torch.tensor(float(cond_row[key]), dtype=torch.float32)
        for key, value in rec.material_row().items():
            if key not in sample:
                sample[key] = torch.tensor(value, dtype=torch.float32)
        if apply_augment and self.augment:
            self._rotate_condition_metadata_in_place(sample, rot_k=rot_k)
        self._apply_zeeman_precondition(sample)
        # Backwards-compat aliases so legacy code that consumed
        # ``SkyrmionPairDataset`` outputs keeps working.
        sample["m0"] = sample["m_init"]
        sample["m1"] = sample["m_t"]
        sample["dt_s"] = sample["t_end_s"]
        sample["dt_scale"] = sample["frame_target"]
        sample["dt_index"] = sample["t_end_index"]
        self._set_augment_metadata(
            sample,
            rot_k=rot_k if apply_augment and self.augment else 0,
            spin_flip=spin_flip if apply_augment and self.augment else False,
        )
        if aux_targets:
            sample["aux_m_t"] = torch.stack(aux_targets, dim=0)
            sample["aux_omega_target"] = torch.stack(aux_omegas, dim=0)
            for key, values in aux_tensors.items():
                sample[f"aux_{key}"] = torch.stack(values, dim=0)
        if self.action_matching_enabled:
            sample["action_state"] = torch.stack(action_states, dim=0)
            sample["action_velocity"] = torch.stack(action_velocities, dim=0)
            sample["action_tau"] = torch.stack(action_taus, dim=0)
            sample["action_valid"] = torch.stack(action_valid, dim=0)
        return sample

    def visual_rows(self, max_rows: int = 5) -> list[list[dict[str, Any]]]:
        """Return rows of one trajectory across all available fixed-time targets."""
        rows: list[list[dict[str, Any]]] = []
        full_rows = [
            idx for idx in self._valid_record_idx if len(self._record_choices[idx]) == len(self.t_end_ns)
        ]
        candidates = full_rows or self._valid_record_idx
        for row_idx, rec_idx in enumerate(candidates[:max_rows]):
            rng = np.random.default_rng(self.seed + 7_919 * (row_idx + 1))
            frame_init = None
            if self.anchor_mode == "random":
                rec = self.records[rec_idx]
                _, max_offset, save_step_ps = max(
                    self._record_choices[rec_idx],
                    key=lambda choice: choice[1],
                )
                frame_init = self._choose_frame_init(rec_idx, rec, max_offset, save_step_ps, rng)
            rows.append(
                [
                    self._build_sample(rec_idx, choice_idx, rng, frame_init=frame_init)
                    for choice_idx in range(len(self._record_choices[rec_idx]))
                ]
            )
        return rows

    def _build_visual_sample_for_frames(
        self,
        rec_idx: int,
        frame_init: int,
        frame_target: int,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        rec = self.records[rec_idx]
        save_step_ps = _save_step_ps_from_params(rec.params, default=25.0)
        frame_offset = int(frame_target) - int(frame_init)
        if frame_offset <= 0:
            raise ValueError("visual sample requires frame_target > frame_init")
        for choice_idx, (_, offset, _) in enumerate(self._record_choices[rec_idx]):
            if int(offset) == frame_offset:
                return self._build_sample(
                    rec_idx,
                    choice_idx,
                    rng,
                    frame_init=int(frame_init),
                    apply_augment=False,
                    frame_target_override=int(frame_target),
                )
        t_end_ns = self._duration_ns(rec, int(frame_init), int(frame_target), save_step_ps)
        choice_idx = len(self._record_choices[rec_idx])
        self._record_choices[rec_idx].append((float(t_end_ns), frame_offset, save_step_ps))
        self._choice_start_ranges[rec_idx].append([(int(frame_init), int(frame_init))])
        try:
            return self._build_sample(
                rec_idx,
                choice_idx,
                rng,
                frame_init=int(frame_init),
                apply_augment=False,
                frame_target_override=int(frame_target),
            )
        finally:
            self._record_choices[rec_idx].pop()
            self._choice_start_ranges[rec_idx].pop()

    def _trajectory_visual_segment_specs(
        self,
        rec_idx: int,
    ) -> list[tuple[int, int, int, float, float, float]]:
        rec = self.records[rec_idx]
        save_step_ps = _save_step_ps_from_params(rec.params, default=25.0)
        max_frame = self._n_frames(rec) - 1
        specs: list[tuple[int, int, int, float, float, float]] = []
        for segment_idx, (start_ns, end_ns) in enumerate(
            self._control_segments_for(rec_idx, rec, save_step_ps)
        ):
            if not self._segment_in_time_range(float(start_ns), float(end_ns)):
                continue
            start_frame = self._frame_at_or_after(rec, float(start_ns), save_step_ps)
            end_frame = self._frame_at_or_before(rec, float(end_ns), save_step_ps)
            if start_frame is None or end_frame is None:
                continue
            if end_frame <= start_frame or end_frame > max_frame:
                continue
            specs.append(
                (
                    int(segment_idx),
                    int(start_frame),
                    int(end_frame),
                    float(start_ns),
                    float(end_ns),
                    float(save_step_ps),
                )
            )
        return specs

    def trajectory_visual_rows(self, max_rows: int = 5) -> list[list[dict[str, Any]]]:
        """Return multi-segment visual rows with horizons reset at each segment.

        Each row is ordered by control segment. Within a segment, cells are
        fixed-time horizons from that segment's start plus the segment end.
        The validation visualizer chains the segment-end prediction into the
        next segment.
        """
        if self.segment_policy == "none":
            return self.visual_rows(max_rows)
        scored: list[tuple[int, int, int]] = []
        specs_by_rec: dict[int, list[tuple[int, int, int, float, float, float]]] = {}
        for rec_idx in self._valid_record_idx:
            specs = self._trajectory_visual_segment_specs(rec_idx)
            if not specs:
                continue
            specs_by_rec[rec_idx] = specs
            scored.append((0 if len(specs) > 1 else 1, -len(specs), rec_idx))
        rows: list[list[dict[str, Any]]] = []
        eps = 1.0e-9
        for row_idx, (_, _, rec_idx) in enumerate(sorted(scored)):
            if len(rows) >= max_rows:
                break
            rec = self.records[rec_idx]
            rng = np.random.default_rng(self.seed + 13_193 * (row_idx + 1))
            row: list[dict[str, Any]] = []
            for segment_idx, start_frame, end_frame, _start_ns, _end_ns, save_step_ps in specs_by_rec[rec_idx]:
                segment_start_time_ns = self._frame_time_ns(rec, start_frame, save_step_ps)
                segment_end_time_ns = self._frame_time_ns(rec, end_frame, save_step_ps)
                targets: list[int] = []
                for t_end_ns in sorted(self.t_end_ns):
                    target_time_ns = segment_start_time_ns + float(t_end_ns)
                    if target_time_ns > segment_end_time_ns + eps:
                        continue
                    target_frame = self._frame_nearest_in_range(
                        rec,
                        target_time_ns,
                        int(start_frame) + 1,
                        end_frame,
                        save_step_ps,
                    )
                    if target_frame is None:
                        continue
                    if self._frame_time_ns(rec, target_frame, save_step_ps) > segment_start_time_ns + eps:
                        targets.append(int(target_frame))
                targets.append(int(end_frame))
                seen: set[int] = set()
                for target_frame in sorted(targets):
                    if target_frame in seen:
                        continue
                    seen.add(target_frame)
                    sample = self._build_visual_sample_for_frames(
                        rec_idx,
                        start_frame,
                        target_frame,
                        rng,
                    )
                    segment_elapsed_ns = self._duration_ns(
                        rec,
                        start_frame,
                        target_frame,
                        save_step_ps,
                    )
                    sample["visual_segment_start_frame"] = torch.tensor(start_frame, dtype=torch.long)
                    sample["visual_segment_end_frame"] = torch.tensor(end_frame, dtype=torch.long)
                    sample["visual_segment_elapsed_ns"] = torch.tensor(segment_elapsed_ns, dtype=torch.float32)
                    sample["visual_segment_start_time_ns"] = torch.tensor(segment_start_time_ns, dtype=torch.float32)
                    sample["visual_is_segment_end"] = torch.tensor(target_frame == end_frame, dtype=torch.bool)
                    sample["visual_control_segment_index"] = torch.tensor(segment_idx, dtype=torch.long)
                    row.append(sample)
            if row:
                rows.append(row)
        return rows

    def rollout_visual_rows(self, max_rows: int = 5) -> list[list[dict[str, Any]]]:
        """Return chronological one-hop samples for autoregressive validation.

        Unlike :meth:`trajectory_visual_rows`, every cell starts at the
        previous cell's target frame.  The visualizer can therefore replace
        that input with the previous prediction at *every* step.  When an
        off-grid control boundary leaves a gap between adjacent segment frame
        ranges, an explicit boundary-transition hop is inserted so rollout
        time remains continuous instead of silently lagging by one frame.
        """
        if self.segment_policy == "none":
            return self.visual_rows(max_rows)
        scored: list[tuple[int, int, int]] = []
        specs_by_rec: dict[int, list[tuple[int, int, int, float, float, float]]] = {}
        for rec_idx in self._valid_record_idx:
            specs = self._trajectory_visual_segment_specs(rec_idx)
            if not specs:
                continue
            specs_by_rec[rec_idx] = specs
            scored.append((0 if len(specs) > 1 else 1, -len(specs), rec_idx))

        rows: list[list[dict[str, Any]]] = []
        eps = 1.0e-9
        for row_idx, (_, _, rec_idx) in enumerate(sorted(scored)):
            if len(rows) >= max_rows:
                break
            rec = self.records[rec_idx]
            rng = np.random.default_rng(self.seed + 17_389 * (row_idx + 1))
            row: list[dict[str, Any]] = []
            rollout_frame: int | None = None
            previous_segment_idx: int | None = None
            for segment_idx, start_frame, end_frame, _start_ns, _end_ns, save_step_ps in specs_by_rec[rec_idx]:
                segment_start_time_ns = self._frame_time_ns(rec, start_frame, save_step_ps)
                segment_end_time_ns = self._frame_time_ns(rec, end_frame, save_step_ps)
                if rollout_frame is None:
                    rollout_frame = int(start_frame)
                targets: list[tuple[int, bool]] = []
                # The previous segment may end before the first saved frame in
                # this segment when a physical boundary falls between frames.
                if rollout_frame < start_frame:
                    targets.append((int(start_frame), True))
                for horizon_ns in sorted(self.t_end_ns):
                    target_time_ns = segment_start_time_ns + float(horizon_ns)
                    if target_time_ns > segment_end_time_ns + eps:
                        continue
                    target_frame = self._frame_nearest_in_range(
                        rec,
                        target_time_ns,
                        int(start_frame) + 1,
                        end_frame,
                        save_step_ps,
                    )
                    if target_frame is not None:
                        targets.append((int(target_frame), False))
                targets.append((int(end_frame), False))

                seen: set[int] = set()
                for target_frame, is_boundary_transition in sorted(targets):
                    if target_frame in seen or target_frame <= int(rollout_frame):
                        continue
                    seen.add(target_frame)
                    hop_start_frame = int(rollout_frame)
                    sample = self._build_visual_sample_for_frames(
                        rec_idx,
                        hop_start_frame,
                        target_frame,
                        rng,
                    )
                    sample["visual_segment_start_frame"] = torch.tensor(start_frame, dtype=torch.long)
                    sample["visual_segment_end_frame"] = torch.tensor(end_frame, dtype=torch.long)
                    sample["visual_segment_elapsed_ns"] = torch.tensor(
                        self._duration_ns(rec, start_frame, target_frame, save_step_ps),
                        dtype=torch.float32,
                    )
                    sample["visual_segment_start_time_ns"] = torch.tensor(
                        segment_start_time_ns,
                        dtype=torch.float32,
                    )
                    sample["visual_is_segment_end"] = torch.tensor(
                        target_frame == end_frame,
                        dtype=torch.bool,
                    )
                    sample["visual_boundary_transition"] = torch.tensor(
                        is_boundary_transition,
                        dtype=torch.bool,
                    )
                    sample["visual_rollout_step"] = torch.tensor(len(row), dtype=torch.long)
                    visual_segment_idx = (
                        previous_segment_idx
                        if is_boundary_transition and previous_segment_idx is not None
                        else segment_idx
                    )
                    sample["visual_control_segment_index"] = torch.tensor(
                        visual_segment_idx,
                        dtype=torch.long,
                    )
                    row.append(sample)
                    rollout_frame = int(target_frame)
                previous_segment_idx = int(segment_idx)
            if row:
                rows.append(row)
        return rows

    @staticmethod
    def _rotate_spin_xy(m: torch.Tensor, k: int) -> torch.Tensor:
        k = int(k) % 4
        if k == 0:
            return m
        out = m.clone()
        mx, my = m[0].clone(), m[1].clone()
        if k == 1:
            out[0], out[1] = -my, mx
        elif k == 2:
            out[0], out[1] = -mx, -my
        else:
            out[0], out[1] = my, -mx
        return out

    @staticmethod
    def _rotate_b_xy(b_t: torch.Tensor, k: int) -> torch.Tensor:
        k = int(k) % 4
        if k == 0:
            return b_t
        out = b_t.clone()
        bx, by = b_t[0].clone(), b_t[1].clone()
        if k == 1:
            out[0], out[1] = -by, bx
        elif k == 2:
            out[0], out[1] = -bx, -by
        else:
            out[0], out[1] = by, -bx
        return out

    def _augment(
        self,
        m_init: torch.Tensor,
        m_t: torch.Tensor,
        defect: torch.Tensor,
        j_field: torch.Tensor,
        b_t: torch.Tensor,
        omega_target: torch.Tensor | np.random.Generator,
        rng: np.random.Generator | None = None,
    ) -> tuple[torch.Tensor, ...]:
        legacy = rng is None
        if legacy:
            rng = omega_target  # type: ignore[assignment]
            omega_target = torch.zeros_like(m_init)
        if not self.augment:
            out = (m_init, m_t, defect, j_field, b_t, omega_target)
            return out[:5] if legacy else out
        rot_k = int(rng.integers(0, 4)) if self.augment_rot90 else 0
        spin_flip = self.augment_spin_flip_prob > 0.0 and rng.random() < self.augment_spin_flip_prob
        out = self._augment_with_params(
            m_init,
            m_t,
            defect,
            j_field,
            b_t,
            omega_target,
            rot_k=rot_k,
            spin_flip=spin_flip,
        )
        return out[:5] if legacy else out

    def _augment_with_params(
        self,
        m_init: torch.Tensor,
        m_t: torch.Tensor,
        defect: torch.Tensor,
        j_field: torch.Tensor,
        b_t: torch.Tensor,
        omega_target: torch.Tensor,
        *,
        rot_k: int,
        spin_flip: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        k = int(rot_k) % 4
        if k:
            m_init = self._rotate_spin_xy(torch.rot90(m_init, k=k, dims=(-2, -1)), k)
            m_t = self._rotate_spin_xy(torch.rot90(m_t, k=k, dims=(-2, -1)), k)
            omega_target = self._rotate_spin_xy(torch.rot90(omega_target, k=k, dims=(-2, -1)), k)
            defect = torch.rot90(defect, k=k, dims=(-2, -1))
            j_field = torch.rot90(j_field, k=k, dims=(-2, -1))
            b_t = self._rotate_b_xy(b_t, k)
        if spin_flip:
            m_init = -m_init
            m_t = -m_t
            b_t = -b_t
        return m_init, m_t, defect, j_field, b_t, omega_target

    def _quality_cache_path(self) -> Path:
        raw = self.quality_sampling.get("cache_path")
        if raw is None:
            raise ValueError(
                "data.quality_sampling.enabled=true requires "
                "data.quality_sampling.cache_path"
            )
        return Path(str(raw))

    def _install_quality_cache(self, payload: dict[str, Any]) -> None:
        entries: dict[tuple[str, int], dict[str, Any]] = {}
        for run_id, by_segment in (payload.get("segments") or {}).items():
            if not isinstance(by_segment, dict):
                continue
            for segment_idx, entry in by_segment.items():
                if isinstance(entry, dict):
                    entries[(str(run_id), int(segment_idx))] = dict(entry)
        self._quality_segment_entries = entries
        self._quality_cache_summary = dict(payload.get("summary") or {})

    def load_quality_sampling_cache(self) -> dict[str, Any]:
        """Load the preflight segment cache into this dataset instance."""

        if not self.quality_sampling_enabled:
            return {}
        payload = load_segment_quality_cache(
            self,
            self.quality_sampling,
            self._quality_cache_path(),
        )
        self._install_quality_cache(payload)
        return payload

    def prepare_quality_sampling_cache(self) -> dict[str, Any]:
        """Build or validate the segment cache before training starts.

        In DDP this method is called only on rank zero; the other ranks load
        the completed cache after a barrier.
        """

        if not self.quality_sampling_enabled:
            return {}
        path = self._quality_cache_path()
        rebuild = bool(self.quality_sampling.get("rebuild_cache", False))
        auto_build = bool(self.quality_sampling.get("auto_build", True))
        if path.is_file() and not rebuild:
            try:
                return self.load_quality_sampling_cache()
            except (ValueError, json.JSONDecodeError):
                if not auto_build:
                    raise
                print(
                    {"quality_cache": "stale_or_invalid", "path": str(path)},
                    flush=True,
                )
        elif not path.is_file() and not auto_build:
            raise FileNotFoundError(
                f"quality sampling cache does not exist and auto_build=false: {path}"
            )
        payload = build_segment_quality_cache(self, self.quality_sampling, path)
        self._install_quality_cache(payload)
        return payload

    def _draw_standard_pair_spec(
        self,
        rng: np.random.Generator,
    ) -> tuple[int, int, int]:
        rec_idx = self._valid_record_idx[
            int(rng.integers(0, len(self._valid_record_idx)))
        ]
        choices = self._record_choices[rec_idx]
        if self.t_end_probs is None:
            choice_idx = int(rng.integers(0, len(choices)))
        else:
            weights = np.asarray(
                [
                    self.t_end_probs[self._t_end_bucket_index(choice[0])]
                    for choice in choices
                ],
                dtype=np.float64,
            )
            weights = weights / weights.sum()
            choice_idx = int(rng.choice(len(choices), p=weights))
        _t_end_ns, frame_offset, save_step_ps = choices[choice_idx]
        frame_init = self._choose_frame_init(
            rec_idx,
            self.records[rec_idx],
            frame_offset,
            save_step_ps,
            rng,
            choice_idx=choice_idx,
        )
        return int(rec_idx), int(choice_idx), int(frame_init)

    @staticmethod
    def _quality_segment_category(entry: dict[str, Any] | None) -> str:
        if not entry:
            return "informative"
        category = str(entry.get("category", "informative"))
        return category if category in SEGMENT_CATEGORY_TO_INDEX else "informative"

    def _annotate_quality_sample(
        self,
        sample: dict[str, Any],
        *,
        metrics: dict[str, float] | None,
        pair_category: str | None,
        segment_category: str | None,
        attempts: int,
        original_mix: bool,
        forced_accept: bool,
    ) -> None:
        sample["quality_original_mix"] = torch.tensor(original_mix, dtype=torch.bool)
        sample["quality_forced_accept"] = torch.tensor(forced_accept, dtype=torch.bool)
        sample["quality_sampling_attempts"] = torch.tensor(int(attempts), dtype=torch.long)
        sample["quality_pair_category_index"] = torch.tensor(
            -1 if pair_category is None else PAIR_CATEGORY_TO_INDEX[pair_category],
            dtype=torch.long,
        )
        sample["quality_segment_category_index"] = torch.tensor(
            -1
            if segment_category is None
            else SEGMENT_CATEGORY_TO_INDEX[segment_category],
            dtype=torch.long,
        )
        for key in (
            "raw_angle_deg",
            "coherent_angle_deg",
            "coherent_retention",
            "lowfreq_mz_change",
            "source_block_resultant",
            "target_block_resultant",
            "target_neighbor_angle_deg",
        ):
            sample[f"quality_{key}"] = torch.tensor(
                float("nan") if metrics is None else float(metrics[key]),
                dtype=torch.float32,
            )

    def _build_quality_candidate(
        self,
        rec_idx: int,
        choice_idx: int,
        frame_init: int,
        rng: np.random.Generator,
    ) -> tuple[dict[str, Any], bool]:
        """Build a candidate without paying the log-map cost before admission."""

        needs_omega = bool(getattr(self, "compute_omega_target", False))
        self.compute_omega_target = False
        try:
            sample = self._build_sample(
                rec_idx,
                choice_idx,
                rng,
                frame_init=frame_init,
            )
        finally:
            self.compute_omega_target = needs_omega
        return sample, needs_omega

    def _finish_quality_candidate(
        self,
        sample: dict[str, Any],
        *,
        needs_omega: bool,
    ) -> dict[str, Any]:
        if not needs_omega:
            return sample
        if float(sample["t_end_ns"].item()) <= self.alpha_t_end_ns_max:
            sample["omega_target"] = log_map_chw(
                sample["m_init"].unsqueeze(0), sample["m_t"].unsqueeze(0)
            ).squeeze(0)
        else:
            sample["omega_target"] = torch.zeros_like(sample["m_init"])
        if "aux_m_t" in sample:
            aux_target = sample["aux_m_t"]
            source = sample["m_init"].unsqueeze(0).expand_as(aux_target)
            aux_omega = log_map_chw(source, aux_target)
            if "aux_t_end_ns" in sample:
                valid = sample["aux_t_end_ns"] <= self.alpha_t_end_ns_max
                aux_omega = aux_omega * valid[:, None, None, None].to(aux_omega.dtype)
            sample["aux_omega_target"] = aux_omega
        return sample

    def _build_quality_sample(
        self,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        cfg = self.quality_sampling

        # Keep an explicit unbiased component.  It is selected before looking
        # at any target-derived score, so the original conditional support is
        # never erased by the quality policy.
        if rng.random() < float(cfg["original_mix_probability"]):
            rec_idx, choice_idx, frame_init = self._draw_standard_pair_spec(rng)
            sample = self._build_sample(
                rec_idx,
                choice_idx,
                rng,
                frame_init=frame_init,
            )
            self._annotate_quality_sample(
                sample,
                metrics=None,
                pair_category=None,
                segment_category=None,
                attempts=1,
                original_mix=True,
                forced_accept=False,
            )
            return sample

        best: tuple[float, dict[str, Any], dict[str, float], str, str] | None = None
        best_needs_omega = False
        max_attempts = int(cfg["max_attempts"])
        for attempt in range(1, max_attempts + 1):
            rec_idx, choice_idx, frame_init = self._draw_standard_pair_spec(rng)
            sample, needs_omega = self._build_quality_candidate(
                rec_idx,
                choice_idx,
                frame_init,
                rng,
            )
            metrics = pair_quality_metrics(
                sample,
                block_factor=int(cfg["block_factor"]),
            )
            pair_category = classify_pair_quality(metrics, cfg)
            segment_idx = int(sample["control_segment_index"].item())
            entry = self._quality_segment_entries.get(
                (str(sample["run_id"]), segment_idx)
            )
            segment_category = self._quality_segment_category(entry)
            pair_probability = pair_keep_probability(pair_category, cfg)
            segment_probability = segment_keep_probability(entry, cfg)
            effective_probability = pair_probability * segment_probability
            if best is None or effective_probability > best[0]:
                best = (
                    effective_probability,
                    sample,
                    metrics,
                    pair_category,
                    segment_category,
                )
                best_needs_omega = needs_omega
            if (
                rng.random() < segment_probability
                and rng.random() < pair_probability
            ):
                self._annotate_quality_sample(
                    sample,
                    metrics=metrics,
                    pair_category=pair_category,
                    segment_category=segment_category,
                    attempts=attempt,
                    original_mix=False,
                    forced_accept=False,
                )
                return self._finish_quality_candidate(
                    sample,
                    needs_omega=needs_omega,
                )

        assert best is not None
        _probability, sample, metrics, pair_category, segment_category = best
        self._annotate_quality_sample(
            sample,
            metrics=metrics,
            pair_category=pair_category,
            segment_category=segment_category,
            attempts=max_attempts,
            original_mix=False,
            forced_accept=True,
        )
        return self._finish_quality_candidate(
            sample,
            needs_omega=best_needs_omega,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = self._rng(idx)
        if self.segment_pair_mode:
            force_predicted: bool | None = None
            pool = self._segment_pair_indices
            if self.segment_pair_sampling_mode == "dataset_mixture":
                probability = self.segment_pair_predicted_probability
                if abs(probability - 0.5) <= 1.0e-12:
                    # An even-length epoch has an exact global 1:1 split even
                    # after DistributedSampler shuffles its indices.
                    force_predicted = bool(int(idx) % 2)
                else:
                    force_predicted = bool(rng.random() < probability)
                if force_predicted:
                    pool = self._segment_pair_replay_indices
            rec_idx, segment_idx = pool[int(rng.integers(0, len(pool)))]
            return self._build_segment_pair(
                rec_idx,
                segment_idx,
                rng,
                force_predicted=force_predicted,
            )
        if self.rollout_steps > 0:
            pool = self._rollout_valid_idx
            rec_idx = pool[int(rng.integers(0, len(pool)))]
            return self._build_clip(rec_idx, rng)
        if self.quality_sampling_enabled:
            return self._build_quality_sample(rng)
        rec_idx, choice_idx, frame_init = self._draw_standard_pair_spec(rng)
        return self._build_sample(
            rec_idx,
            choice_idx,
            rng,
            frame_init=frame_init,
        )


def _fixed_time_omega_settings(cfg: dict[str, Any]) -> tuple[bool, float]:
    data_cfg = cfg["data"]
    bridge_cfg = cfg.get("bridge", {})
    bridge_repr = bridge_state_repr(bridge_cfg)
    loss_cfg = cfg.get("train", {}).get("loss", {})
    alpha_mix = float(loss_cfg.get("alpha_mix", 0.0))
    fit_kappa = bool(cfg.get("prior", {}).get("fit_kappa", False))
    force_omega = bool(data_cfg.get("force_omega_target", False))
    compute_omega = (
        bridge_repr in {"alpha", "alpha2d"}
        or alpha_mix > 0.0
        or fit_kappa
        or force_omega
    )
    alpha_cap = float(loss_cfg.get("alpha_t_end_ns_max", 1.0))
    omega_cap = float("inf") if (fit_kappa or force_omega) else alpha_cap
    return compute_omega, omega_cap


def build_fixed_time_datasets(
    cfg: dict[str, Any],
    *,
    build_splits: set[str] | None = None,
) -> tuple[
    FixedTimePairDataset | None,
    FixedTimePairDataset | None,
    FixedTimePairDataset | None,
]:
    """Build train/val/test fixed-time datasets following ``cfg['data']``."""
    from skyrmion_cfm.data.trajectory import TrajectoryIndex

    data_cfg = cfg["data"]
    requested_splits = {"train", "val", "test"} if build_splits is None else set(build_splits)
    invalid_splits = requested_splits - {"train", "val", "test"}
    if invalid_splits:
        raise ValueError(f"Unknown fixed-time dataset splits: {sorted(invalid_splits)}")
    t_end_ns = list(data_cfg.get("t_end_ns", DEFAULT_T_END_NS))
    lattice_size = int(data_cfg.get("lattice_size", 256))
    index = TrajectoryIndex.from_root(
        data_cfg["dataset_root"],
        frame_glob=data_cfg.get("frame_glob", "run.out/m*.ovf"),
        include_before_drive=bool(data_cfg.get("include_before_drive", True)),
        cache_path=data_cfg.get("trajectory_index_cache"),
        rebuild_cache=bool(data_cfg.get("rebuild_trajectory_index_cache", False)),
    )
    split_manifest = data_cfg.get("split_manifest")
    if split_manifest:
        train_idx, val_idx, test_idx = _split_trajectory_index_from_manifest(
            index, split_manifest
        )
    else:
        train_idx, val_idx, test_idx = index.split(
            int(data_cfg.get("train_trajectories", 1600)),
            int(data_cfg.get("val_trajectories", 200)),
            int(data_cfg.get("test_trajectories", 200)),
            int(data_cfg.get("split_seed", 1234)),
            str(data_cfg.get("split_mode", "random")),
        )

    memmap_cfg = data_cfg.get("memmap", {})
    memmap_manifest = None
    if bool(memmap_cfg.get("enabled", False)):
        from skyrmion_cfm.data.memmap import ensure_memmap

        memmap_manifest = ensure_memmap(
            data_cfg["dataset_root"],
            memmap_cfg.get("path", "outputs/skyrmion_cfm/memmap"),
            frame_glob=data_cfg.get("frame_glob", "run.out/m*.ovf"),
            dtype=memmap_cfg.get("dtype", "float16"),
            include_before_drive=bool(data_cfg.get("include_before_drive", True)),
            auto_build=bool(memmap_cfg.get("auto_build", False)),
            force=bool(memmap_cfg.get("force_rebuild", False)),
        )
    aug_cfg = data_cfg.get("augment", {})
    # plan-2 line 34 mixed-α ablation: the convex blend
    # ``L = (1-λ_α) L_β + λ_α L_α`` needs Ω_target even when the *primary*
    # bridge is β. The physics-scaled prior also needs real Ω_target samples
    # to fit kappa; otherwise fixed-time Cart/RFM would hand the fitter the
    # loader's all-zero placeholder for buckets past the alpha cap.
    compute_omega, omega_cap = _fixed_time_omega_settings(cfg)

    train_anchor_mode = _normalize_anchor_mode(data_cfg.get("anchor_mode", "start"))
    val_anchor_mode = _normalize_anchor_mode(data_cfg.get("val_start_frame_mode", "start"))
    test_anchor_mode = _normalize_anchor_mode(
        data_cfg.get("test_start_frame_mode", data_cfg.get("val_start_frame_mode", "start"))
    )
    condition_mode = _normalize_condition_mode(data_cfg.get("condition_mode", "v2"))
    current_time_mode = str(data_cfg.get("current_time_mode", "t0")).lower()
    control_segmented_cfg = data_cfg.get("control_segmented", {}) or {}
    if bool(control_segmented_cfg.get("enabled", False)):
        segment_policy = str(control_segmented_cfg.get("policy", "control"))
        if "current_time_mode" not in data_cfg:
            current_time_mode = "segment"
    else:
        segment_policy = str(data_cfg.get("segment_policy", "none"))
    segment_policy = _normalize_segment_policy(segment_policy)
    segment_drive_filter = _normalize_segment_drive_filter(
        control_segmented_cfg.get("drive_filter")
    )
    segment_j_atol_a_m2 = float(control_segmented_cfg.get("j_atol_a_m2", 0.0))
    if segment_policy != "none" and current_time_mode not in {"anchor", "segment"}:
        raise ValueError(
            "data.control_segmented requires data.current_time_mode: segment "
            "or anchor (or omit current_time_mode and let the builder select segment)"
        )
    segment_time_range_ns = data_cfg.get("segment_time_range_ns")
    t_end_probs = data_cfg.get("t_end_probs")
    field_cache_size = int(data_cfg.get("field_cache_size", memmap_cfg.get("field_cache_size", 512)))
    if bool(cfg.get("model", {}).get("use_spatial_cond", False)):
        from skyrmion_cfm.models.common import spatial_condition_fields_from_cfg

        spatial_cond_fields = spatial_condition_fields_from_cfg(cfg)
    else:
        spatial_cond_fields = ()
    # Pushforward rollout clips apply to TRAINING only; val/test stay single-pair
    # so the val metrics remain a clean single-step comparison. Default off.
    rollout_cfg = cfg.get("train", {}).get("rollout", {}) or {}
    segment_pf_cfg = cfg.get("train", {}).get("segment_pushforward", {}) or {}
    segment_pair_mode = bool(segment_pf_cfg.get("enabled", False))
    rollout_on = bool(rollout_cfg.get("enabled", True)) and int(rollout_cfg.get("steps", 0)) >= 2
    if segment_pair_mode and rollout_on:
        raise ValueError("train.segment_pushforward and train.rollout are mutually exclusive")
    segment_pair_predicted_probability = 0.0
    if segment_pair_mode:
        true_weight = float(segment_pf_cfg.get("true_weight", 1.0))
        pred_weight = float(segment_pf_cfg.get("pred_weight", 1.0))
        if true_weight < 0.0 or pred_weight < 0.0:
            raise ValueError("segment_pushforward true_weight and pred_weight must be >= 0")
        total_init_weight = true_weight + pred_weight
        if total_init_weight <= 0.0:
            raise ValueError(
                "segment_pushforward true_weight and pred_weight cannot both be zero"
            )
        segment_pair_predicted_probability = pred_weight / total_init_weight
    rollout_steps = int(rollout_cfg.get("steps", 0)) if rollout_on else 0
    rollout_dt_ns = rollout_cfg.get("dt_ns")
    rollout_dt_ns = None if rollout_dt_ns is None else float(rollout_dt_ns)

    def _make(
        records,
        n: int,
        seed_offset: int,
        augment: bool,
        anchor_mode: str,
        rollout_steps: int = 0,
        segment_pair_mode: bool = False,
    ):
        return FixedTimePairDataset(
            records=records,
            t_end_ns=t_end_ns,
            lattice_size=lattice_size,
            samples_per_epoch=n,
            seed=int(cfg.get("seed", 0)) + seed_offset,
            memmap_manifest=memmap_manifest,
            augment=augment,
            augment_rot90=bool(aug_cfg.get("rot90", True)),
            augment_spin_flip_prob=float(aug_cfg.get("spin_flip_prob", 0.0)),
            compute_omega_target=compute_omega,
            alpha_t_end_ns_max=omega_cap,
            anchor_mode=anchor_mode,
            condition_mode=condition_mode,
            current_time_mode=current_time_mode,
            segment_time_range_ns=segment_time_range_ns,
            rollout_steps=rollout_steps,
            rollout_dt_ns=rollout_dt_ns,
            segment_pair_mode=segment_pair_mode,
            segment_pair_include_first=bool(segment_pf_cfg.get("include_first_segment", False)),
            segment_pair_predicted_inits=int(
                segment_pf_cfg.get("predicted_inits_per_segment", 1)
            ),
            segment_pair_predicted_probability=(
                segment_pair_predicted_probability
            ),
            segment_pair_sampling_mode=str(
                segment_pf_cfg.get("sampling_mode", "legacy")
            ),
            t_end_probs=t_end_probs,
            aux_t_end_ns=data_cfg.get("aux_t_end_ns") if seed_offset == 0 else None,
            action_matching=cfg.get("train", {}).get("loss", {}).get("action_matching") if seed_offset == 0 else None,
            segment_policy=segment_policy,
            segment_drive_filter=segment_drive_filter,
            segment_j_atol_a_m2=segment_j_atol_a_m2,
            field_cache_size=field_cache_size,
            spatial_cond_fields=spatial_cond_fields,
            zeeman_precondition=data_cfg.get("zeeman_precondition"),
            quality_sampling=(
                data_cfg.get("quality_sampling") if seed_offset == 0 else None
            ),
        )

    train_ds = (
        _make(
            train_idx.records,
            int(data_cfg.get("train_samples_per_epoch", 100_000)),
            0,
            augment=bool(aug_cfg.get("enabled", False)),
            anchor_mode=train_anchor_mode,
            rollout_steps=rollout_steps,
            segment_pair_mode=segment_pair_mode,
        )
        if "train" in requested_splits
        else None
    )
    val_ds = (
        _make(
            val_idx.records,
            int(data_cfg.get("val_samples_per_epoch", 2_000)),
            1,
            False,
            anchor_mode=val_anchor_mode,
        )
        if "val" in requested_splits
        else None
    )
    test_ds = (
        _make(
            test_idx.records,
            int(data_cfg.get("test_samples_per_epoch", 2_000)),
            2,
            False,
            anchor_mode=test_anchor_mode,
        )
        if "test" in requested_splits
        else None
    )
    return train_ds, val_ds, test_ds


def collate_fixed_time_conditions(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Collate the per-sample dict into a condition dict the embedder consumes."""
    cond = {
        "t_end_index": batch["t_end_index"],
        "t_end_ns": batch["t_end_ns"],
        "t_end_s": batch["t_end_s"],
        "dt_index": batch.get("dt_index", batch["t_end_index"]),
        "dt_s": batch.get("dt_s", batch["t_end_s"]),
        "temp_k": batch["temp_k"],
        "b_t": batch["b_t"],
        "current_a_m2": batch["current_a_m2"],
        "defect_field": batch["defect_field"],
        "j_field": batch["j_field"],
        "control_grid": batch.get("control_grid"),
    }
    for key in ("drive_time_s", "relax_time_s", "drive_fraction"):
        if key in batch:
            cond[key] = batch[key]
    if "control_segment_index" in batch:
        cond["control_segment_index"] = batch["control_segment_index"]
    for key in (
        "alpha",
        "aex_j_per_m",
        "dind_j_per_m2",
        "ku1_j_per_m3",
        "msat_a_per_m",
        "dx_m",
        "dy_m",
        "dz_m",
        "pol_eff",
        "epsilon_prime",
        "fixed_layer_x",
        "fixed_layer_y",
        "fixed_layer_z",
    ):
        if key in batch:
            cond[key] = batch[key]
    for key in SCALAR_CONDITION_KEYS:
        if key in batch and key not in cond:
            cond[key] = batch[key]
    for key in ("beta_zl",):
        if key in batch:
            cond[key] = batch[key]
    for key in V4_CATEGORICAL_KEYS:
        onehot_key = f"{key}_onehot"
        if onehot_key in batch:
            cond[onehot_key] = batch[onehot_key]
    for key in (
        "j_x_field",
        "j_y_field",
        "j_z_field",
        "b_x_field",
        "b_y_field",
        "b_z_field",
        "temperature_field",
        "msat_field",
        "aex_field",
        "ku1_field",
        "dind_field",
        "alpha_field",
    ):
        if key in batch:
            cond[key] = batch[key]
    return cond
