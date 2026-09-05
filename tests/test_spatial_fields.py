"""Tests for the defect / j_field rasterisation."""

from __future__ import annotations

import json

import numpy as np

from skyrmion_cfm.data.spatial_fields import (
    SpatialFieldSources,
    control_change_times_ns,
    control_grid_at_time,
    control_segments_ns,
    pulse_duration_ns,
    pulse_window_ns,
    rasterize_j_field,
)


def test_rasterize_constant_grid_matches_repeat():
    grid = np.ones((8, 8), dtype=np.float32) * 5.0
    raster = rasterize_j_field(grid, lattice_size=256)
    assert raster.shape == (1, 256, 256)
    assert np.all(raster == 5.0)


def test_rasterize_distinct_regions_form_blocks():
    grid = np.arange(64, dtype=np.float32).reshape(8, 8)
    raster = rasterize_j_field(grid, lattice_size=256)
    # Each 32x32 patch should hold a single value matching the grid.
    for iy in range(8):
        for ix in range(8):
            patch = raster[0, iy * 32:(iy + 1) * 32, ix * 32:(ix + 1) * 32]
            assert np.all(patch == grid[iy, ix])


def test_delayed_pulse_region_values_schema_uses_absolute_window(tmp_path):
    protocol = {
        "control_grid": [2, 3],
        "pulse_start_time_ns": 1.0,
        "pulse_end_time_ns": 1.75,
        "shape": {"pulse_duration_ns": 0.75},
        "region_values": [
            {"region_id": rid, "ix": (rid - 1) % 2, "iy": (rid - 1) // 2}
            for rid in range(1, 7)
        ],
    }
    (tmp_path / "current_protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    (tmp_path / "current_control.csv").write_text(
        "t_start_ns,duration_ps,"
        + ",".join(f"J_region_{rid:03d}_A_per_m2" for rid in range(1, 7))
        + "\n1.0,750,1,2,3,4,5,6\n",
        encoding="utf-8",
    )
    sources = SpatialFieldSources.from_run_dir(tmp_path)

    assert pulse_window_ns(sources) == (1.0, 1.75)
    assert pulse_duration_ns(sources) == 0.75
    assert control_change_times_ns(sources) == (0.0, 1.0, 1.75)
    assert control_segments_ns(sources, 3.0) == (
        (0.0, 1.0),
        (1.0, 1.75),
        (1.75, 3.0),
    )
    assert np.array_equal(control_grid_at_time(sources, 0.5), np.zeros((3, 2)))
    assert np.array_equal(
        control_grid_at_time(sources, 1.25),
        np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
    )
    assert np.array_equal(control_grid_at_time(sources, 1.75), np.zeros((3, 2)))


def test_nested_duration_and_csv_start_define_pulse_window(tmp_path):
    (tmp_path / "current_protocol.json").write_text(
        json.dumps({"control_grid": [1, 1], "shape": {"pulse_duration_ns": 0.5}}),
        encoding="utf-8",
    )
    (tmp_path / "current_control.csv").write_text(
        "t_start_ns,duration_ps,J_region_001_A_per_m2\n2.0,500,7\n",
        encoding="utf-8",
    )
    sources = SpatialFieldSources.from_run_dir(tmp_path)

    assert pulse_window_ns(sources) == (2.0, 2.5)
    assert control_grid_at_time(sources, 1.0).item() == 0.0
    assert control_grid_at_time(sources, 2.25).item() == 7.0
