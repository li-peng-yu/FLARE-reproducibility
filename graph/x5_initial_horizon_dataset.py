"""Deterministic x5 initial-state evaluation pairs for compute-cost plots."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from torch.utils.data import Dataset


SELECTION_SEED = 20_260_814
TIME_TOLERANCE_NS = 2.0e-4


@dataclass(frozen=True)
class HorizonPair:
    record_index: int
    frame_init: int
    frame_target: int
    actual_horizon_ns: float
    run_id: str


class X5InitialStateHorizonDataset(Dataset):
    """One initial-to-target pair per held-out x5 trajectory.

    Stage-1 training draws pairs inside individual control segments.  The
    simulator cost benchmark instead starts from ``m_initial`` and executes the
    complete pulse/relax schedule.  This wrapper creates that same view using
    the trajectory's recorded frame timestamps and the existing project's
    condition/spatial-field construction.  It contains no learned component.
    """

    def __init__(
        self,
        base_dataset: Any,
        horizon_ns: float,
        limit: int,
        *,
        selection_seed: int = SELECTION_SEED,
        tolerance_ns: float = TIME_TOLERANCE_NS,
    ) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        required = (
            "records",
            "_valid_record_idx",
            "_n_frames",
            "_frame_time_ns",
            "_frame_nearest_in_range",
            "_duration_ns",
            "_build_visual_sample_for_frames",
        )
        missing = [name for name in required if not hasattr(base_dataset, name)]
        if missing:
            raise TypeError(f"unsupported x5 dataset; missing internals: {missing}")

        self.base_dataset = base_dataset
        self.horizon_ns = float(horizon_ns)
        self.tolerance_ns = float(tolerance_ns)
        self.selection_seed = int(selection_seed)
        # Full-pulse pairs may cross the training loader's control boundaries.
        # The current map is sampled at the initial state; drive/relax duration
        # and fraction remain explicit condition channels.
        self.base_dataset.segment_policy = "none"
        self.base_dataset.current_time_mode = "t0"

        candidates: list[HorizonPair] = []
        for rec_idx in self.base_dataset._valid_record_idx:
            record = self.base_dataset.records[rec_idx]
            frame_init = 0
            max_frame = self.base_dataset._n_frames(record) - 1
            if max_frame <= frame_init:
                continue
            save_step_ps = 25.0
            start_ns = self.base_dataset._frame_time_ns(record, frame_init, save_step_ps)
            frame_target = self.base_dataset._frame_nearest_in_range(
                record,
                start_ns + self.horizon_ns,
                frame_init + 1,
                max_frame,
                save_step_ps,
            )
            if frame_target is None:
                continue
            actual = self.base_dataset._duration_ns(
                record,
                frame_init,
                int(frame_target),
                save_step_ps,
            )
            if abs(float(actual) - self.horizon_ns) > self.tolerance_ns:
                continue
            candidates.append(
                HorizonPair(
                    record_index=int(rec_idx),
                    frame_init=frame_init,
                    frame_target=int(frame_target),
                    actual_horizon_ns=float(actual),
                    run_id=str(record.run_id),
                )
            )

        # Sorting removes dependence on index construction order.  With the
        # same candidate records, every horizon receives the same shuffled
        # trajectory order and every model sees identical examples.
        candidates.sort(key=lambda item: item.run_id)
        random.Random(self.selection_seed).shuffle(candidates)
        if len(candidates) < limit:
            raise RuntimeError(
                f"only {len(candidates)} held-out x5 records support "
                f"{self.horizon_ns:g} ns; requested {limit}"
            )
        self.pairs = tuple(candidates[:limit])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        rng = np.random.default_rng(
            self.selection_seed + 10_007 * int(index) + round(self.horizon_ns * 1_000)
        )
        sample = self.base_dataset._build_visual_sample_for_frames(
            pair.record_index,
            pair.frame_init,
            pair.frame_target,
            rng,
        )
        observed = float(sample["t_end_ns"])
        if abs(observed - self.horizon_ns) > self.tolerance_ns:
            raise RuntimeError(
                f"constructed {observed:.9g} ns pair for requested "
                f"{self.horizon_ns:g} ns"
            )
        return sample

    def selection_metadata(self) -> dict[str, Any]:
        return {
            "selection_seed": self.selection_seed,
            "requested_horizon_ns": self.horizon_ns,
            "time_tolerance_ns": self.tolerance_ns,
            "samples": len(self.pairs),
            "run_ids": [pair.run_id for pair in self.pairs],
            "actual_horizons_ns": [pair.actual_horizon_ns for pair in self.pairs],
        }
