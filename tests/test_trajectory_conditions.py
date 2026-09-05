from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from skyrmion_cfm.data.trajectory import SkyrmionPairDataset, TrajectoryIndex, TrajectoryRecord


class _FakeCache:
    def n_frames(self, rec: TrajectoryRecord) -> int:
        return rec.n_frames

    def get(self, path: Path, rec: TrajectoryRecord, frame_index: int) -> np.ndarray:
        field = np.zeros((3, 2, 2), dtype=np.float32)
        field[2] = 1.0
        if frame_index:
            field[0] = 0.1
        return field


def test_pair_dataset_uses_fixed_environment_condition_aliases():
    rec = TrajectoryRecord(
        run_id="r0",
        path=Path("/tmp/r0"),
        frames=(Path("m000000.ovf"), Path("m000001.ovf")),
        params={
            "fixed_temp_k": 30.0,
            "fixed_bz_t": 0.016,
            "j_amp_a_per_m2": -1.2e12,
        },
    )
    ds = SkyrmionPairDataset(
        TrajectoryIndex([rec]),
        dt_scales=[1],
        scale_probs=None,
        dt_raw_ps=25.0,
        samples_per_epoch=1,
    )
    ds.cache = _FakeCache()

    sample = ds[0]

    assert float(sample["temp_k"]) == pytest.approx(30.0)
    assert float(sample["b_t"][2]) == pytest.approx(0.016)
    assert float(sample["current_a_m2"]) == pytest.approx(-1.2e12)


def test_condition_row_reads_current_amplitude_from_nested_shape():
    rec = TrajectoryRecord(
        run_id="nested-current",
        path=Path("/tmp/nested-current"),
        frames=(),
        params={
            "shape": {"j_amp_a_per_m2": 2.5e12},
            "current_protocol": {
                "shape": {"j_amp_a_per_m2": 3.5e12},
            },
        },
    )

    # The run-level shape is the canonical generator metadata; protocol shape
    # remains a fallback for datasets that only embed current_protocol.
    assert rec.condition_row()["current_a_m2"] == pytest.approx(2.5e12)

    protocol_only = TrajectoryRecord(
        run_id="protocol-current",
        path=Path("/tmp/protocol-current"),
        frames=(),
        params={"current_protocol": {"shape": {"j_amp_a_per_m2": 3.5e12}}},
    )
    assert protocol_only.condition_row()["current_a_m2"] == pytest.approx(3.5e12)


def _write_run(root: Path, names: list[str]) -> Path:
    run = root / "r0"
    out = run / "run.out"
    out.mkdir(parents=True)
    for name in names:
        (out / name).write_text("", encoding="utf-8")
    (run / "params.json").write_text(
        json.dumps({"run_id": "r0", "run_time_s": 4.0e-9, "save_step_ps": 250.0}),
        encoding="utf-8",
    )
    return run


def test_trajectory_index_uses_m_final_when_numbered_final_is_missing(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        [*(f"m{i:06d}.ovf" for i in range(16)), "m_final.ovf", "m_initial.ovf"],
    )

    index = TrajectoryIndex.from_root(tmp_path)
    rec = index.records[0]

    assert rec.n_frames == 17
    assert rec.frames[-1].name == "m_final.ovf"


def test_trajectory_index_does_not_duplicate_existing_numbered_final(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        [*(f"m{i:06d}.ovf" for i in range(17)), "m_final.ovf"],
    )

    index = TrajectoryIndex.from_root(tmp_path)
    rec = index.records[0]

    assert rec.n_frames == 17
    assert rec.frames[-1].name == "m000016.ovf"
