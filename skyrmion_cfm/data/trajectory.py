from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from skyrmion_cfm.config import load_config
from skyrmion_cfm.data.log_map import log_map_chw, normalize_spin
from skyrmion_cfm.data.ovf import read_ovf
from skyrmion_cfm.data import v4_metadata


_FRAME_RE = re.compile(r"m(\d+)\.ovf$")
_SEGMENT_END_RE = re.compile(r"segment_(\d+)_end\.ovf$")
_FINAL_FRAME_NAME = "m_final.ovf"
_OVF_TIME_RE = re.compile(r"Total simulation time:\s*([^\s]+)\s*s", re.IGNORECASE)
_TRAJECTORY_INDEX_CACHE_SCHEMA = 1


def _param_float(params: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = params.get(key)
        if value is not None:
            return float(value)
    protocol = params.get("current_protocol")
    if isinstance(protocol, dict):
        for key in keys:
            value = protocol.get(key)
            if value is not None:
                return float(value)
    return None


def _save_step_ps_from_params(params: dict[str, Any]) -> float | None:
    value = _param_float(params, ("save_step_ps", "save_ps", "save_dt_ps", "dt_save_ps"))
    if value is not None:
        return value
    time_cfg = params.get("time")
    if isinstance(time_cfg, dict):
        save_dt_s = time_cfg.get("save_dt_s")
        if save_dt_s is not None:
            return float(save_dt_s) * 1.0e12
    return None


def _run_time_s_from_params(params: dict[str, Any]) -> float | None:
    value = _param_float(params, ("run_time_s", "total_run_s", "total_time_s", "duration_s"))
    if value is not None:
        return value
    time_cfg = params.get("time")
    if isinstance(time_cfg, dict):
        for key in ("actual_final_time_s", "T_end_s", "condition_end_s"):
            v = time_cfg.get(key)
            if v is not None:
                return float(v)
    return None


def _final_frame_index_from_params(params: dict[str, Any]) -> int | None:
    run_time_s = _run_time_s_from_params(params)
    save_step_ps = _save_step_ps_from_params(params)
    if run_time_s is None or save_step_ps is None or save_step_ps <= 0.0:
        return None
    frame = float(run_time_s) * 1.0e12 / float(save_step_ps)
    rounded = int(round(frame))
    if rounded < 0 or not math.isclose(frame, rounded, rel_tol=0.0, abs_tol=1.0e-3):
        return None
    return rounded


@dataclass(frozen=True)
class TrajectoryRecord:
    run_id: str
    path: Path
    frames: tuple[Path, ...]
    params: dict[str, Any]
    frame_times_s: tuple[float, ...] | None = None

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def _param_float(self, keys: tuple[str, ...], default: float = 0.0) -> float:
        containers: list[dict[str, Any]] = [self.params]
        protocol = self.params.get("current_protocol")
        if isinstance(protocol, dict):
            containers.append(protocol)
        for container in tuple(containers):
            for key in keys:
                value = container.get(key)
                if value is not None:
                    return float(value)
            shape = container.get("shape")
            if isinstance(shape, dict):
                for key in keys:
                    value = shape.get(key)
                    if value is not None:
                        return float(value)
        return float(default)

    @property
    def is_v4(self) -> bool:
        return v4_metadata.is_v4_metadata(self.params)

    def condition_row(self, dt_s: float = 0.0) -> dict[str, float]:
        if self.is_v4:
            return self.v4_condition_row(0.0, float(dt_s))
        return {
            "dt_s": dt_s,
            "t_end_s": dt_s,
            "temp_k": self._param_float(("temp_k", "fixed_temp_k"), 0.0),
            "b_x_t": 0.0,
            "b_y_t": 0.0,
            "b_z_t": self._param_float(("bias_bz_t", "fixed_bz_t", "b_z_t"), 0.0),
            "current_a_m2": self._param_float(
                ("j_sot_eff_a_per_m2", "j_amp_a_per_m2", "current_a_m2"), 0.0
            ),
        }

    def v4_condition_row(self, t_i_s: float, t_j_s: float) -> dict[str, float]:
        return v4_metadata.pair_condition_row(self.params, t_i_s, t_j_s)

    def v4_categorical_tensors(self, t_s: float) -> dict[str, torch.Tensor]:
        seg = v4_metadata.segment_at_time(self.params, t_s)
        return v4_metadata.categorical_tensors(self.params, seg)

    def material_row(self) -> dict[str, float]:
        if self.is_v4:
            row = self.v4_condition_row(0.0, 0.0)
            return {
                "alpha": row["alpha"],
                "aex_j_per_m": row["aex_j_per_m"],
                "dind_j_per_m2": row["dind_j_per_m2"],
                "ku1_j_per_m3": row["ku1_j_per_m3"],
                "msat_a_per_m": row["msat_a_per_m"],
                "dx_m": row["dx_m"],
                "dy_m": row["dy_m"],
                "dz_m": row["dz_m"],
                "pol_eff": row["pol_eff"],
                "epsilon_prime": row["epsilon_prime"],
                "fixed_layer_x": row["fixed_layer_x"],
                "fixed_layer_y": row["fixed_layer_y"],
                "fixed_layer_z": row["fixed_layer_z"],
            }
        fixed_layer = self.params.get("fixed_layer", [0.0, 1.0, 0.0])
        return {
            "alpha": float(self.params.get("alpha", 0.1)),
            "aex_j_per_m": float(self.params.get("aex_j_per_m", 1.0e-11)),
            "dind_j_per_m2": float(self.params.get("dind_j_per_m2", 0.0)),
            "ku1_j_per_m3": float(self.params.get("ku1_j_per_m3", 0.0)),
            "msat_a_per_m": float(self.params.get("msat_a_per_m", 580000.0)),
            "dx_m": float(self.params.get("dx_m", 1.0)),
            "dy_m": float(self.params.get("dy_m", 1.0)),
            "dz_m": float(self.params.get("dz_m", 1.0)),
            "pol_eff": float(self.params.get("pol_eff", 0.0)),
            "epsilon_prime": float(self.params.get("epsilon_prime", 0.0)),
            "fixed_layer_x": float(fixed_layer[0]),
            "fixed_layer_y": float(fixed_layer[1]),
            "fixed_layer_z": float(fixed_layer[2]),
        }


def _frame_sort_key(path: Path) -> tuple[int, str]:
    match = _FRAME_RE.search(path.name)
    if match:
        return int(match.group(1)), path.name
    return -1, path.name


def _sort_frames(frames: list[Path], params: dict[str, Any]) -> list[Path]:
    timed: list[tuple[float, str, Path]] = []
    for frame in frames:
        value = _frame_time_s(frame, params)
        if value is None:
            return sorted(frames, key=_frame_sort_key)
        timed.append((float(value), frame.name, frame))
    return [frame for _, _, frame in sorted(timed)]


def _append_final_frame_if_needed(
    frames: list[Path],
    final_frame: Path | None,
    params: dict[str, Any],
) -> list[Path]:
    if final_frame is None:
        return frames
    final_idx = _final_frame_index_from_params(params)
    if final_idx is None:
        return frames
    present = []
    for frame in frames:
        match = _FRAME_RE.search(frame.name)
        if match:
            present.append(int(match.group(1)))
    if final_idx in present:
        return frames
    if sorted(present) != list(range(len(present))):
        return frames
    if final_idx != len(present):
        return frames
    return [*frames, final_frame]


def _ovf_time_s(path: Path) -> float | None:
    try:
        with path.open("rb") as f:
            for _ in range(128):
                raw = f.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "ignore")
                match = _OVF_TIME_RE.search(line)
                if match:
                    return float(match.group(1))
                if "Begin: Data" in line:
                    break
    except OSError:
        return None
    return None


def _fallback_frame_time_s(path: Path, params: dict[str, Any]) -> float | None:
    match = _FRAME_RE.search(path.name)
    save_step_ps = _save_step_ps_from_params(params)
    if match and save_step_ps is not None:
        return int(match.group(1)) * float(save_step_ps) * 1.0e-12
    if path.name == _FINAL_FRAME_NAME:
        return _run_time_s_from_params(params)
    return None


def _frame_time_s(path: Path, params: dict[str, Any]) -> float | None:
    value = _ovf_time_s(path)
    if value is not None:
        return value
    return _fallback_frame_time_s(path, params)


def _append_final_frame_by_time(
    frames: list[Path],
    final_frame: Path | None,
    params: dict[str, Any],
) -> list[Path]:
    if final_frame is None:
        return frames
    if not frames:
        return [final_frame]
    final_time = _frame_time_s(final_frame, params)
    last_time = _frame_time_s(frames[-1], params)
    if final_time is not None and last_time is not None:
        if final_time > last_time + 1.0e-15:
            return [*frames, final_frame]
        return frames
    return _append_final_frame_if_needed(frames, final_frame, params)


def _frame_times_for(frames: list[Path], params: dict[str, Any]) -> tuple[float, ...] | None:
    times: list[float] = []
    for frame in frames:
        value = _frame_time_s(frame, params)
        if value is None:
            return None
        times.append(float(value))
    if any(b < a for a, b in zip(times, times[1:], strict=False)):
        return None
    return tuple(times)


class TrajectoryIndex:
    def __init__(self, records: list[TrajectoryRecord]) -> None:
        self.records = records

    @classmethod
    def from_root(
        cls,
        root: str | Path | list[str | Path] | tuple[str | Path, ...],
        frame_glob: str = "run.out/m*.ovf",
        include_before_drive: bool = True,
        cache_path: str | Path | None = None,
        rebuild_cache: bool = False,
    ) -> "TrajectoryIndex":
        # plan-2 anchors at the *pre-drive* magnetisation. For the
        # ``skx_concentrated_sot_pairs_*`` datasets ``m000000.ovf`` is already
        # the pre-drive frame because ``AutoSave`` fires before the first J
        # pulse, so this default has no effect there; the flag governs newer
        # datasets that emit a separate ``m_before_drive.ovf``.
        #
        # ``root`` may be a single directory or a list of directories — the
        # latter supports batched generation runs whose output landed in
        # separate folders (different save_step_ps schedules, etc). Records
        # carry their own ``save_step_ps`` via ``params.json`` so heterogeneous
        # roots compose without further configuration.
        if isinstance(root, (str, Path)):
            roots = [Path(root)]
        else:
            roots = [Path(r) for r in root]
        cache_identity = {
            "schema": _TRAJECTORY_INDEX_CACHE_SCHEMA,
            "roots": [str(path.resolve()) for path in roots],
            "root_mtime_ns": [int(path.stat().st_mtime_ns) for path in roots],
            "frame_glob": str(frame_glob),
            "include_before_drive": bool(include_before_drive),
        }
        cache_file = Path(cache_path) if cache_path is not None else None
        if cache_file is not None and cache_file.is_file() and not rebuild_cache:
            try:
                with cache_file.open("rb") as handle:
                    cached = pickle.load(handle)
                cached_records = cached.get("records") if isinstance(cached, dict) else None
                if (
                    isinstance(cached, dict)
                    and cached.get("identity") == cache_identity
                    and isinstance(cached_records, list)
                    and cached_records
                    and all(isinstance(record, TrajectoryRecord) for record in cached_records)
                ):
                    print(
                        {
                            "trajectory_index_cache": "loaded",
                            "path": str(cache_file),
                            "records": len(cached_records),
                        },
                        flush=True,
                    )
                    return cls(cached_records)
            except (OSError, EOFError, pickle.PickleError, AttributeError, ValueError) as exc:
                print(
                    {
                        "trajectory_index_cache": "ignored",
                        "path": str(cache_file),
                        "reason": f"{type(exc).__name__}: {exc}",
                    },
                    flush=True,
                )
        seen_run_ids: set[str] = set()
        records: list[TrajectoryRecord] = []
        for r in roots:
            for params_path in sorted(r.glob("*/params.json")):
                run_dir = params_path.parent
                with params_path.open("r", encoding="utf-8") as f:
                    params = json.load(f)
                frames = []
                final_frame = None
                for frame in run_dir.glob(frame_glob):
                    if frame.name == "m_before_drive.ovf" and not include_before_drive:
                        continue
                    if _FRAME_RE.search(frame.name) or _SEGMENT_END_RE.search(frame.name):
                        frames.append(frame)
                    elif frame.name == _FINAL_FRAME_NAME:
                        final_frame = frame
                frames = _sort_frames(frames, params)
                frames = _append_final_frame_by_time(frames, final_frame, params)
                frames = _sort_frames(frames, params)
                frame_times_s = _frame_times_for(frames, params)
                if len(frames) >= 2:
                    run_id = str(params.get("run_id", run_dir.name))
                    # Disambiguate identical run_ids coming from different
                    # roots; otherwise the memmap manifest would clash.
                    if run_id in seen_run_ids:
                        run_id = f"{r.name}/{run_id}"
                    seen_run_ids.add(run_id)
                    records.append(
                        TrajectoryRecord(
                            run_id=run_id,
                            path=run_dir,
                            frames=tuple(frames),
                            params=params,
                            frame_times_s=frame_times_s,
                        )
                    )
        if not records:
            raise FileNotFoundError(
                f"No trajectories with OVF frames found below {[str(p) for p in roots]}"
            )
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_file.with_name(f".{cache_file.name}.{os.getpid()}.tmp")
            with temporary.open("wb") as handle:
                pickle.dump(
                    {"identity": cache_identity, "records": records},
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            os.replace(temporary, cache_file)
            print(
                {
                    "trajectory_index_cache": "written",
                    "path": str(cache_file),
                    "records": len(records),
                },
                flush=True,
            )
        return cls(records)

    def split(
        self,
        train: int,
        val: int,
        test: int,
        seed: int,
        mode: str = "random",
    ) -> tuple["TrajectoryIndex", "TrajectoryIndex", "TrajectoryIndex"]:
        split_mode = str(mode).strip().lower().replace("-", "_")
        if split_mode in {"metadata", "predefined", "params"}:
            requested = {"train": int(train), "val": int(val), "test": int(test)}
            if any(count < 0 for count in requested.values()):
                raise ValueError("metadata split counts must be >= 0")
            buckets: dict[str, list[TrajectoryRecord]] = {
                "train": [],
                "val": [],
                "test": [],
            }
            invalid: list[tuple[str, object]] = []
            for record in self.records:
                label = str(record.params.get("split", "")).strip().lower()
                if label not in buckets:
                    invalid.append((record.run_id, record.params.get("split")))
                    continue
                buckets[label].append(record)
            if invalid:
                preview = ", ".join(f"{run_id}={label!r}" for run_id, label in invalid[:5])
                raise ValueError(
                    "metadata split mode requires params.split=train|val|test for every "
                    f"trajectory; invalid examples: {preview}"
                )
            for label, count in requested.items():
                available = len(buckets[label])
                if count > available:
                    raise ValueError(
                        f"metadata split requested {count} {label} trajectories, "
                        f"but only {available} are available"
                    )
            rng = random.Random(seed)
            for records in buckets.values():
                rng.shuffle(records)
            return (
                TrajectoryIndex(buckets["train"][: requested["train"]]),
                TrajectoryIndex(buckets["val"][: requested["val"]]),
                TrajectoryIndex(buckets["test"][: requested["test"]]),
            )
        if split_mode not in {"random", "shuffle", "shuffled"}:
            raise ValueError(
                "split mode must be random or metadata, "
                f"got {mode!r}"
            )
        records = list(self.records)
        rng = random.Random(seed)
        rng.shuffle(records)
        total = len(records)
        if train + val + test > total:
            train = min(train, total)
            val = min(val, max(0, total - train))
            test = max(0, total - train - val)
        return (
            TrajectoryIndex(records[:train]),
            TrajectoryIndex(records[train : train + val]),
            TrajectoryIndex(records[train + val : train + val + test]),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "n_trajectories": len(self.records),
            "min_frames": min(r.n_frames for r in self.records),
            "max_frames": max(r.n_frames for r in self.records),
            "mean_frames": sum(r.n_frames for r in self.records) / len(self.records),
        }


class FrameCache:
    def __init__(self, preload: bool = False, memmap_manifest: str | Path | None = None) -> None:
        self.preload = preload
        self._cache: dict[Path, np.ndarray] = {}
        self.memmap = None
        if memmap_manifest is not None:
            from skyrmion_cfm.data.memmap import OVFMemmapStore

            self.memmap = OVFMemmapStore(memmap_manifest)

    def get(self, path: Path, rec: TrajectoryRecord | None = None, frame_index: int | None = None) -> np.ndarray:
        if self.memmap is not None and rec is not None and frame_index is not None:
            return self.memmap.get(rec.run_id, frame_index)
        if self.preload and path in self._cache:
            return self._cache[path]
        arr = read_ovf(path)
        if self.preload:
            self._cache[path] = arr
        return arr

    def n_frames(self, rec: TrajectoryRecord) -> int:
        if self.memmap is None:
            return rec.n_frames
        try:
            return self.memmap.n_frames(rec.run_id)
        except KeyError:
            return rec.n_frames


class SkyrmionPairDataset(Dataset):
    """Stochastic multi-scale pair sampler over whole trajectories."""

    def __init__(
        self,
        index: TrajectoryIndex,
        dt_scales: list[int],
        scale_probs: list[float] | None,
        dt_raw_ps: float,
        samples_per_epoch: int = 100_000,
        preload: bool = False,
        seed: int = 0,
        allowed_scales: list[int] | None = None,
        augment: bool = False,
        augment_shift: bool = True,
        augment_rot90: bool = True,
        augment_spin_flip_prob: float = 0.0,
        memmap_manifest: str | Path | None = None,
        start_frame_mode: str = "random",
    ) -> None:
        super().__init__()
        self.index = index
        self.dt_scales = list(dt_scales)
        self.scale_probs = scale_probs
        if self.scale_probs is not None:
            probs = np.asarray(self.scale_probs, dtype=np.float64)
            self.scale_probs = (probs / probs.sum()).tolist()
        self.dt_raw_ps = float(dt_raw_ps)
        self.samples_per_epoch = int(samples_per_epoch)
        self.cache = FrameCache(preload=preload, memmap_manifest=memmap_manifest)
        self.seed = int(seed)
        self.allowed_scales = set(allowed_scales) if allowed_scales is not None else None
        self.augment = bool(augment)
        self.augment_shift = bool(augment_shift)
        self.augment_rot90 = bool(augment_rot90)
        self.augment_spin_flip_prob = float(augment_spin_flip_prob)
        if start_frame_mode not in {"random", "first"}:
            raise ValueError(f"Unknown start_frame_mode: {start_frame_mode}")
        self.start_frame_mode = start_frame_mode
        self.valid_records = [
            r
            for r in self.index.records
            if any(self.cache.n_frames(r) > s for s in self.dt_scales if self._scale_allowed(s))
        ]
        if not self.valid_records:
            raise ValueError("No trajectory has enough frames for the configured dt_scales")

    def _scale_allowed(self, scale: int) -> bool:
        return self.allowed_scales is None or scale in self.allowed_scales

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _rng(self, idx: int) -> np.random.Generator:
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        return np.random.default_rng(self.seed + idx * 9973 + worker_id * 104729)

    def _choose(self, rng: np.random.Generator) -> tuple[TrajectoryRecord, int, int, int]:
        rec = self.valid_records[int(rng.integers(0, len(self.valid_records)))]
        n_frames = self.cache.n_frames(rec)
        scales = [s for s in self.dt_scales if self._scale_allowed(s) and n_frames > s]
        if self.scale_probs is None:
            scale = int(scales[int(rng.integers(0, len(scales)))])
        else:
            probs = np.asarray(
                [self.scale_probs[self.dt_scales.index(s)] for s in scales], dtype=np.float64
            )
            probs = probs / probs.sum()
            scale = int(rng.choice(scales, p=probs))
        frame0 = 0 if self.start_frame_mode == "first" else int(rng.integers(0, n_frames - scale))
        dt_index = self.dt_scales.index(scale)
        return rec, frame0, scale, dt_index

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

    def _augment_pair(
        self,
        m0: torch.Tensor,
        m1: torch.Tensor,
        b_t: torch.Tensor,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.augment:
            return m0, m1, b_t
        if self.augment_shift:
            shift_y = int(rng.integers(0, m0.shape[-2]))
            shift_x = int(rng.integers(0, m0.shape[-1]))
            m0 = torch.roll(m0, shifts=(shift_y, shift_x), dims=(-2, -1))
            m1 = torch.roll(m1, shifts=(shift_y, shift_x), dims=(-2, -1))
        if self.augment_rot90:
            k = int(rng.integers(0, 4))
            if k:
                m0 = self._rotate_spin_xy(torch.rot90(m0, k=k, dims=(-2, -1)), k)
                m1 = self._rotate_spin_xy(torch.rot90(m1, k=k, dims=(-2, -1)), k)
                b_t = self._rotate_b_xy(b_t, k)
        if self.augment_spin_flip_prob > 0.0 and rng.random() < self.augment_spin_flip_prob:
            m0 = -m0
            m1 = -m1
            b_t = -b_t
        return m0, m1, b_t

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = self._rng(idx)
        rec, frame0, scale, dt_index = self._choose(rng)
        m0_np = self.cache.get(rec.frames[frame0], rec, frame0)
        m1_np = self.cache.get(rec.frames[frame0 + scale], rec, frame0 + scale)
        m0 = torch.from_numpy(m0_np.copy()).float()
        m1 = torch.from_numpy(m1_np.copy()).float()
        m0 = normalize_spin(m0.movedim(0, -1)).movedim(-1, 0)
        m1 = normalize_spin(m1.movedim(0, -1)).movedim(-1, 0)
        cond_row = rec.condition_row(dt_s=0.0)
        b_t = torch.tensor([0.0, 0.0, cond_row["b_z_t"]], dtype=torch.float32)
        m0, m1, b_t = self._augment_pair(m0, m1, b_t, rng)
        omega = log_map_chw(m0.unsqueeze(0), m1.unsqueeze(0)).squeeze(0)
        dt_s = scale * self.dt_raw_ps * 1e-12
        cond_row = rec.condition_row(dt_s=dt_s)
        sample = {
            "m0": m0,
            "m1": m1,
            "omega_target": omega,
            "dt_scale": torch.tensor(scale, dtype=torch.long),
            "dt_index": torch.tensor(dt_index, dtype=torch.long),
            "dt_s": torch.tensor(dt_s, dtype=torch.float32),
            "temp_k": torch.tensor(cond_row["temp_k"], dtype=torch.float32),
            "b_t": b_t,
            "current_a_m2": torch.tensor(cond_row["current_a_m2"], dtype=torch.float32),
            "run_id": rec.run_id,
            "frame0": torch.tensor(frame0, dtype=torch.long),
            "frame_init": torch.tensor(frame0, dtype=torch.long),
            "frame_target": torch.tensor(frame0 + scale, dtype=torch.long),
        }
        for key, value in rec.material_row().items():
            sample[key] = torch.tensor(value, dtype=torch.float32)
        return sample


def build_datasets(cfg: dict[str, Any]) -> tuple[SkyrmionPairDataset, SkyrmionPairDataset, SkyrmionPairDataset]:
    data_cfg = cfg["data"]
    index = TrajectoryIndex.from_root(
        data_cfg["dataset_root"],
        frame_glob=data_cfg.get("frame_glob", "run.out/m*.ovf"),
        include_before_drive=bool(data_cfg.get("include_before_drive", True)),
    )
    train_idx, val_idx, test_idx = index.split(
        int(data_cfg.get("train_trajectories", 800)),
        int(data_cfg.get("val_trajectories", 50)),
        int(data_cfg.get("test_trajectories", 150)),
        int(data_cfg.get("split_seed", 1234)),
    )
    common = {
        "dt_scales": list(data_cfg["dt_scales"]),
        "scale_probs": data_cfg.get("scale_probs"),
        "dt_raw_ps": float(data_cfg.get("dt_raw_ps", 5.0)),
        "preload": bool(data_cfg.get("preload", False)),
    }
    memmap_cfg = data_cfg.get("memmap", {})
    if bool(memmap_cfg.get("enabled", False)):
        from skyrmion_cfm.data.memmap import ensure_memmap

        common["memmap_manifest"] = ensure_memmap(
            data_cfg["dataset_root"],
            memmap_cfg.get("path", "outputs/skyrmion_cfm/memmap"),
            frame_glob=data_cfg.get("frame_glob", "run.out/m*.ovf"),
            dtype=memmap_cfg.get("dtype", "float16"),
            include_before_drive=bool(data_cfg.get("include_before_drive", True)),
            auto_build=bool(memmap_cfg.get("auto_build", False)),
            force=bool(memmap_cfg.get("force_rebuild", False)),
        )
    aug_cfg = data_cfg.get("augment", {})
    boundary = str(data_cfg.get("boundary", "open")).lower()
    augment_enabled = bool(aug_cfg.get("enabled", False))
    requested_shift = bool(aug_cfg.get("shift", True))
    if augment_enabled and requested_shift and boundary not in ("periodic", "circular", "pbc"):
        # Random toroidal translations wrap the lattice and only preserve the
        # data distribution under periodic BCs; under open/free BCs they leak
        # defects and edge structure across the opposite boundary.
        import warnings

        warnings.warn(
            f"augment.shift was requested but data.boundary='{boundary}' is non-periodic; "
            "disabling shift augmentation to stay aligned with the simulation domain.",
            stacklevel=2,
        )
        requested_shift = False
    train_ds = SkyrmionPairDataset(
        train_idx,
        **common,
        samples_per_epoch=int(data_cfg.get("train_samples_per_epoch", 100_000)),
        seed=int(cfg.get("seed", 0)),
        augment=augment_enabled,
        augment_shift=requested_shift,
        augment_rot90=bool(aug_cfg.get("rot90", True)),
        augment_spin_flip_prob=float(aug_cfg.get("spin_flip_prob", 0.0)),
    )
    val_ds = SkyrmionPairDataset(
        val_idx,
        **common,
        samples_per_epoch=int(data_cfg.get("val_samples_per_epoch", 2_000)),
        seed=int(cfg.get("seed", 0)) + 1,
        start_frame_mode=str(data_cfg.get("val_start_frame_mode", "random")),
    )
    test_ds = SkyrmionPairDataset(
        test_idx,
        **common,
        samples_per_epoch=int(data_cfg.get("test_samples_per_epoch", 2_000)),
        seed=int(cfg.get("seed", 0)) + 2,
    )
    return train_ds, val_ds, test_ds


def collate_conditions(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cond = {
        "dt_index": batch["dt_index"],
        "dt_s": batch["dt_s"],
        "temp_k": batch["temp_k"],
        "b_t": batch["b_t"],
        "current_a_m2": batch["current_a_m2"],
    }
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
    return cond


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the skyrmion dataset index.")
    parser.add_argument("--config", default="skyrmion_cfm/configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    index = TrajectoryIndex.from_root(
        cfg["data"]["dataset_root"],
        frame_glob=cfg["data"].get("frame_glob", "run.out/m*.ovf"),
        include_before_drive=cfg["data"].get("include_before_drive", False),
    )
    print(json.dumps(index.summary(), indent=2))


if __name__ == "__main__":
    main()
