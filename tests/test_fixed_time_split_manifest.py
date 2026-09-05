from __future__ import annotations

import json
from pathlib import Path

import pytest

from skyrmion_cfm.data.fixed_time import _split_trajectory_index_from_manifest
from skyrmion_cfm.data.trajectory import TrajectoryIndex, TrajectoryRecord


def _index() -> TrajectoryIndex:
    records = [
        TrajectoryRecord(
            run_id=f"run_{index}",
            path=Path(f"/tmp/run_{index}"),
            frames=(Path("m0.ovf"), Path("m1.ovf")),
            params={},
        )
        for index in range(4)
    ]
    return TrajectoryIndex(records)


def test_split_manifest_preserves_declared_order(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            {
                "splits": {
                    "train": ["run_2", "run_0"],
                    "val": ["run_1"],
                    "test": ["run_3"],
                }
            }
        ),
        encoding="utf-8",
    )
    train, val, test = _split_trajectory_index_from_manifest(_index(), path)
    assert [record.run_id for record in train.records] == ["run_2", "run_0"]
    assert [record.run_id for record in val.records] == ["run_1"]
    assert [record.run_id for record in test.records] == ["run_3"]


def test_split_manifest_rejects_cross_split_leakage(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            {
                "train": ["run_0", "run_1"],
                "val": ["run_1"],
                "test": ["run_2"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reuses run ids"):
        _split_trajectory_index_from_manifest(_index(), path)
