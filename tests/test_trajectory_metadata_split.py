from pathlib import Path

import pytest

from skyrmion_cfm.data.trajectory import TrajectoryIndex, TrajectoryRecord


def _record(run_id: str, split: object) -> TrajectoryRecord:
    return TrajectoryRecord(
        run_id=run_id,
        path=Path(run_id),
        frames=(Path(f"{run_id}/m000000.ovf"), Path(f"{run_id}/m000001.ovf")),
        params={"split": split},
    )


def test_metadata_split_respects_predefined_partitions() -> None:
    index = TrajectoryIndex(
        [
            _record("train_0", "train"),
            _record("train_1", "train"),
            _record("train_2", "train"),
            _record("val_0", "val"),
            _record("val_1", "val"),
            _record("test_0", "test"),
        ]
    )

    train, val, test = index.split(3, 2, 1, seed=1234, mode="metadata")

    assert {record.run_id for record in train.records} == {
        "train_0",
        "train_1",
        "train_2",
    }
    assert {record.run_id for record in val.records} == {"val_0", "val_1"}
    assert {record.run_id for record in test.records} == {"test_0"}


def test_metadata_split_rejects_missing_or_oversubscribed_partitions() -> None:
    with pytest.raises(ValueError, match="params.split"):
        TrajectoryIndex([_record("missing", None)]).split(
            1, 0, 0, seed=0, mode="metadata"
        )

    with pytest.raises(ValueError, match="only 1 are available"):
        TrajectoryIndex([_record("train_0", "train")]).split(
            2, 0, 0, seed=0, mode="metadata"
        )
