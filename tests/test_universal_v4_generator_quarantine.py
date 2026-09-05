from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from unittest.mock import patch


GENERATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "dataset_generation"
    / "generate_universal_2d_micromagnetic_dynamics_v4-2.py"
)
SPEC = importlib.util.spec_from_file_location("universal_v4_generator_quarantine_test", GENERATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


def _mask_metadata(*, encoding: str | None = None) -> dict:
    geometry = {
        "geometry_mode": "nanostrip",
        "parameters": {
            "length_px": 6.0,
            "width_px": 4.0,
            "edge_rounding_px": 0.0,
            "orientation_deg": 0.0,
        },
    }
    if encoding is not None:
        geometry["mask_image_encoding"] = encoding
    return {
        "trajectory_id": "mask_run",
        "geometry_mode": "nanostrip",
        "geometry": geometry,
        "grid": {"Nx": 8, "Ny": 8},
        "drive_type": "none",
    }


def test_geometry_mask_uses_mumax_dark_inside_polarity() -> None:
    rows = GENERATOR.geometry_mask_rows(_mask_metadata())

    assert rows[4][4] == 0
    assert rows[0][0] == 255


def test_fixed_mask_encoding_is_not_classified_as_legacy_affected() -> None:
    metadata = _mask_metadata(encoding=GENERATOR.MASK_IMAGE_ENCODING)

    assert GENERATOR.affected_reasons_for_metadata(metadata) == []


def _write_params(root: Path, name: str, metadata: dict) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "params.json").write_text(json.dumps(metadata), encoding="utf-8")
    return run_dir


def test_quarantine_dry_run_then_apply_moves_only_known_affected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    quarantine_root = tmp_path / "quarantine"
    dataset_root.mkdir()
    _write_params(dataset_root, "bad_mask", _mask_metadata())
    _write_params(
        dataset_root,
        "bad_proxy",
        {
            "trajectory_id": "bad_proxy",
            "geometry_mode": "disk",
            "geometry": {"geometry_mode": "disk", "parameters": {}},
            "drive_type": GENERATOR.SOT_PROXY_RENDERED_DRIVE_TYPE,
        },
    )
    _write_params(
        dataset_root,
        "clean",
        {
            "trajectory_id": "clean",
            "geometry_mode": "disk",
            "geometry": {"geometry_mode": "disk", "parameters": {}},
            "drive_type": "none",
        },
    )

    dry_run = GENERATOR.quarantine_known_affected_trajectories(
        dataset_root,
        quarantine_root,
    )

    assert dry_run["trajectory_count"] == 3
    assert dry_run["affected_trajectory_count"] == 2
    assert dry_run["clean_trajectory_count"] == 1
    assert dry_run["split_counts"] == {"unspecified": 3}
    assert dry_run["affected_split_counts"] == {"unspecified": 2}
    assert dry_run["clean_split_counts"] == {"unspecified": 1}
    assert not quarantine_root.exists()
    assert (dataset_root / "bad_mask").is_dir()
    assert (dataset_root / "bad_proxy").is_dir()

    applied = GENERATOR.quarantine_known_affected_trajectories(
        dataset_root,
        quarantine_root,
        apply=True,
        progress_every=0,
    )

    assert applied["moved_trajectory_count"] == 2
    assert (dataset_root / "clean").is_dir()
    assert not (dataset_root / "bad_mask").exists()
    assert not (dataset_root / "bad_proxy").exists()
    assert (quarantine_root / "bad_mask").is_dir()
    assert (quarantine_root / "bad_proxy").is_dir()
    assert len((quarantine_root / "quarantine_manifest.jsonl").read_text().splitlines()) == 2
    summary = json.loads((quarantine_root / "quarantine_summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["moved_trajectory_count"] == 2


def test_quarantine_apply_resumes_after_interrupted_move(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    quarantine_root = tmp_path / "quarantine"
    dataset_root.mkdir()
    _write_params(dataset_root, "bad_mask", _mask_metadata())
    _write_params(
        dataset_root,
        "bad_proxy",
        {
            "trajectory_id": "bad_proxy",
            "geometry_mode": "disk",
            "geometry": {"geometry_mode": "disk", "parameters": {}},
            "drive_type": GENERATOR.SOT_PROXY_RENDERED_DRIVE_TYPE,
        },
    )

    original_rename = Path.rename
    moved_once = False

    def interrupted_rename(source: Path, target: Path) -> Path:
        nonlocal moved_once
        if source.parent == dataset_root:
            if moved_once:
                raise OSError("simulated interruption")
            moved_once = True
        return original_rename(source, target)

    with patch.object(Path, "rename", interrupted_rename):
        try:
            GENERATOR.quarantine_known_affected_trajectories(
                dataset_root,
                quarantine_root,
                apply=True,
                progress_every=1,
            )
        except OSError as exc:
            assert str(exc) == "simulated interruption"
        else:
            raise AssertionError("simulated interrupted quarantine unexpectedly completed")

    partial_summary = json.loads((quarantine_root / "quarantine_summary.json").read_text())
    assert partial_summary["status"] == "moving"
    assert partial_summary["moved_trajectory_count"] == 1

    resumed = GENERATOR.quarantine_known_affected_trajectories(
        dataset_root,
        quarantine_root,
        apply=True,
        progress_every=0,
    )

    assert resumed["resumed"] is True
    assert resumed["moved_trajectory_count"] == 2
    assert not (dataset_root / "bad_mask").exists()
    assert not (dataset_root / "bad_proxy").exists()
    assert (quarantine_root / "bad_mask").is_dir()
    assert (quarantine_root / "bad_proxy").is_dir()
    completed_summary = json.loads((quarantine_root / "quarantine_summary.json").read_text())
    assert completed_summary["status"] == "complete"
    assert completed_summary["moved_trajectory_count"] == 2


def test_strict_sot_mapping_matches_mumax_lambda_one_basis() -> None:
    pol = 0.24
    r_fl_dl = -0.5
    epsilon_prime = GENERATOR.sot_epsilon_prime(pol, r_fl_dl)

    assert epsilon_prime == -0.06
    coeffs = GENERATOR.sot_field_coefficients_T(
        8.0e11,
        pol,
        epsilon_prime,
        6.0e5,
        1.2e-9,
        0.08,
    )
    assert coeffs["sot_B_FL_T"] / coeffs["sot_B_DL_T"] == r_fl_dl
    expected_beta = (
        GENERATOR.HBAR_J_S
        / GENERATOR.ELEMENTARY_CHARGE_C
        * 8.0e11
        / (6.0e5 * 1.2e-9)
    )
    assert math.isclose(coeffs["sot_beta_T"], expected_beta, rel_tol=1.0e-12)


def test_strict_sot_metadata_is_not_quarantined_and_uses_safe_timestep() -> None:
    cfg = dict(GENERATOR.CONFIG)
    drive = {
        "has_sot": True,
        "has_sot_like_slonczewski_proxy": False,
        "rendered_torque_model": GENERATOR.SOT_RENDERED_TORQUE_MODEL,
        "J_A_per_m2": 5.0e11,
        "J_vector_A_per_m2": [0.0, 0.0, 5.0e11],
        "Pol": 0.2,
        "Lambda": 1.0,
        "EpsilonPrime": GENERATOR.sot_epsilon_prime(0.2, 0.4),
        "polarization": [0.0, 1.0, 0.0],
    }
    segment = {
        "drive": drive,
        "instantaneous_material": {"Ms_T_A_per_m": 5.8e5, "alpha_T": 0.1},
        "region_material_values": [],
    }
    summary: dict = {}

    GENERATOR.annotate_sot_segments(
        [segment],
        summary,
        {"thickness_m": 1.0e-9},
        cfg,
    )

    assert summary["has_sot"] is True
    assert summary["has_sot_like_slonczewski_proxy"] is False
    assert summary["rendered_torque_model"] == GENERATOR.SOT_RENDERED_TORQUE_MODEL
    assert 0.0 < drive["sot_recommended_max_dt_s"] <= cfg["max_dt_s"]
    assert GENERATOR.sot_segment_rejection_reasons([segment], cfg) == []
    rendered = GENERATOR.render_drive_assignment(segment, cfg)
    assert "Effective spin-Hall SOT" in rendered
    assert "Lambda = 1" in rendered
    assert "EpsilonPrime = 0.04" in rendered
    assert "proxy" not in rendered.lower()

    metadata = {
        "geometry_mode": "disk",
        "geometry": {"geometry_mode": "disk", "parameters": {}},
        "drive_type": "sot",
        "drive": summary,
        "segments": [segment],
    }
    assert GENERATOR.affected_reasons_for_metadata(metadata) == []


def test_sot_effective_field_limit_rejects_overdriven_proposal() -> None:
    cfg = dict(GENERATOR.CONFIG)
    cfg["sot_max_effective_field_T"] = 1.0e-3
    drive = {
        "has_sot": True,
        "J_A_per_m2": 3.0e12,
        "Pol": 0.6,
        "EpsilonPrime": 0.3,
    }
    segment = {
        "drive": drive,
        "instantaneous_material": {"Ms_T_A_per_m": 2.0e5, "alpha_T": 0.02},
        "region_material_values": [],
    }
    GENERATOR.annotate_sot_segments(
        [segment],
        {},
        {"thickness_m": 5.0e-10},
        cfg,
    )

    assert "sot_effective_field_exceeds_limit" in GENERATOR.sot_segment_rejection_reasons([segment], cfg)
