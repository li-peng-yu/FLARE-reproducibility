#!/usr/bin/env python3
"""Summarize the matched rotation-target MSE flow-necessity control."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    PROJECT
    / "outputs/skx_bt_1000base_x5/paper_revision_20260902/"
    "flow_necessity_rotation_mse"
)
DEFAULT_CORE = (
    PROJECT
    / "outputs/skx_bt_1000base_x5/paper_revision_20260830/"
    "exact_control_seed_geometry/summary/summary.json"
)
REFERENCE_MODE = "exact_control_multisegment_rollout_endpoint"
LABELS = ("rotmse_s78", "rotmse_s79", "rotmse_s80")
METRICS = (
    "distribution_score",
    "angular_energy_distance_deg",
    "paired_angular_error_deg",
    "fair_energy_score",
)


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"expected three finite seed values, got {values}")
    return {
        "seeds": 3,
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _validate_exact_anchor(root: Path, label: str) -> None:
    summary = _json(root / "exact_anchor" / label / "summary/run_summary.json")
    if summary.get("multisegment_rollout") is not True:
        raise RuntimeError(f"{label}: exact-anchor run is not a multisegment rollout")
    if int(summary.get("conditions", -1)) != 33:
        raise RuntimeError(f"{label}: expected 33 exact-anchor conditions")
    if int(summary.get("sampler_ode_steps", -1)) != 1:
        raise RuntimeError(f"{label}: deterministic control was not evaluated with Euler-1")
    condition_dirs = sorted(
        path
        for path in (root / "exact_anchor" / label / "conditions").glob("*")
        if path.is_dir()
    )
    if len(condition_dirs) != 33:
        raise RuntimeError(f"{label}: expected 33 exact-anchor condition directories")
    for condition_dir in condition_dirs:
        metadata = _json(condition_dir / "condition_metadata.json")
        manifest = _json(condition_dir / "model_samples/manifest.json")
        if metadata.get("reference_mode") != REFERENCE_MODE:
            raise RuntimeError(f"{label}: wrong reference mode in {condition_dir}")
        if metadata.get("teacher_forcing_after_first_segment") is not False:
            raise RuntimeError(f"{label}: teacher-forced handoff in {condition_dir}")
        if int(manifest.get("num_samples", -1)) != 5:
            raise RuntimeError(f"{label}: expected one forecast for each of five anchors")
        assignment = np.load(condition_dir / "model_samples/anchor_repeat_index.npy")
        if not np.array_equal(assignment, np.arange(5, dtype=assignment.dtype)):
            raise RuntimeError(f"{label}: forecasts are not one-to-one with repeat anchors")


def _load_direct(root: Path, fair: dict[str, Any], label: str) -> dict[str, Any]:
    seed = int(label[-2:])
    distribution_path = root / "distribution" / label / "summary/run_summary.json"
    distribution = _json(distribution_path)
    if distribution.get("reference_mode") != REFERENCE_MODE:
        raise RuntimeError(f"{label}: distribution run is not exact-control rollout")
    if int(distribution.get("conditions", -1)) != 33:
        raise RuntimeError(f"{label}: expected 33 complete paths")
    if int(distribution.get("sampler_ode_steps", -1)) != 1:
        raise RuntimeError(f"{label}: paired metric was not evaluated with Euler-1")
    if int(distribution.get("model_samples_per_condition", -1)) != 5:
        raise RuntimeError(f"{label}: expected five anchor-matched forecasts")
    _validate_exact_anchor(root, label)

    method = fair["methods"][label]
    forecast_types = method.get("forecast_type", [])
    if "deterministic" not in forecast_types:
        raise RuntimeError(f"{label}: fair-energy evaluator did not detect point forecasts")
    fair_metric = method["combined"]["fair_energy_score"]
    if int(fair_metric.get("anchor_cases", -1)) != 165:
        raise RuntimeError(f"{label}: fair energy does not contain 165 exact anchors")
    return {
        "label": label,
        "seed": seed,
        "distribution_score": float(
            distribution["score"]["combined"]["geometric_mean_score"]
        ),
        "angular_energy_distance_deg": float(
            distribution["angular_energy_distance"]["combined"]["mean"]
        ),
        "paired_angular_error_deg": float(
            distribution["paired_angular_error"]["combined"]["mean"]
        ),
        "fair_energy_score": float(fair_metric["mean"]),
        "checkpoint": str(distribution["checkpoint"]),
        "checkpoint_sha256": str(distribution["checkpoint_sha256"]),
        "distribution_source": str(distribution_path),
        "fair_energy_source": str(root / "fair_energy/run_summary.json"),
    }


def _fmt(stats: dict[str, float | int], digits: int) -> str:
    return f"{float(stats['mean']):.{digits}f} ± {float(stats['sample_std']):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--core-summary", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cno-paired-angle-deg", type=float)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = (args.output or root / "summary").expanduser().resolve()

    fair = _json(root / "fair_energy/run_summary.json")
    if int(fair.get("bootstrap_iterations", -1)) != 50_000:
        raise RuntimeError("fair-energy evaluation did not use 50,000 bootstrap draws")
    direct_rows = [_load_direct(root, fair, label) for label in LABELS]
    direct_stats = {
        metric: _stats([float(row[metric]) for row in direct_rows])
        for metric in METRICS
    }

    core = _json(args.core_summary.expanduser().resolve())
    core_rows = {
        int(row["seed"]): row
        for row in core["per_seed"]
        if row["family"] == "core"
    }
    if set(core_rows) != {78, 79, 80}:
        raise RuntimeError("core reference does not contain seeds 78/79/80")
    core_stats = core["families"]["core"]["metrics"]
    matched_rows: list[dict[str, Any]] = []
    for direct in direct_rows:
        seed = int(direct["seed"])
        reference = core_rows[seed]
        matched_rows.append(
            {
                "seed": seed,
                **{
                    f"{metric}_rotation_mse_minus_core": (
                        float(direct[metric]) - float(reference[metric])
                    )
                    for metric in METRICS
                },
            }
        )
    difference_stats = {
        metric: _stats(
            [
                float(row[f"{metric}_rotation_mse_minus_core"])
                for row in matched_rows
            ]
        )
        for metric in METRICS
    }

    paired_mean = float(direct_stats["paired_angular_error_deg"]["mean"])
    cno_delta = (
        None
        if args.cno_paired_angle_deg is None
        else paired_mean - float(args.cno_paired_angle_deg)
    )
    payload = {
        "status": "complete",
        "control": (
            "same U-Net and local axis-angle target as core; direct masked MSE "
            "at tau=0; identity zero source; deterministic Euler-1 inference"
        ),
        "protocol": {
            "reference_mode": REFERENCE_MODE,
            "base_groups": 33,
            "complete_paths_per_seed": 165,
            "training_seeds": [78, 79, 80],
            "evaluation_seed": 209020160,
            "deterministic_forecast_handling": "one point forecast per exact repeat anchor",
        },
        "core": core_stats,
        "rotation_mse": direct_stats,
        "rotation_mse_minus_core_matched_seed": difference_stats,
        "per_seed": direct_rows,
        "matched_seed_differences": matched_rows,
        "table1_cno_fm_paired_angle_deg": args.cno_paired_angle_deg,
        "rotation_mse_minus_cno_fm_paired_angle_deg": cno_delta,
    }
    _atomic_json(output / "summary.json", payload)
    _atomic_csv(output / "per_seed.csv", direct_rows)
    _atomic_csv(output / "matched_seed_differences.csv", matched_rows)

    rows = []
    for label, stats in (
        ("Core local axis--angle CFM", core_stats),
        ("Rotation-target MSE (ours control)", direct_stats),
        ("MSE minus CFM (matched seeds)", difference_stats),
    ):
        rows.append(
            "| "
            + label
            + " | "
            + " | ".join(
                _fmt(stats[metric], 4 if metric in {"distribution_score", "fair_energy_score"} else 3)
                for metric in METRICS
            )
            + " |"
        )
    report = [
        "# Flow-necessity control: direct local rotation regression",
        "",
        "All learned rows use the same U-Net, data split, conditioning, optimizer,",
        "50k-update budget, and local axis--angle endpoint target. The control",
        "replaces flow matching with direct masked MSE regression at tau=0 and",
        "uses one deterministic Euler step at inference.",
        "",
        "| Method | Distribution Score ↑ | Angular ED ↓ | Paired angle ↓ | Fair ES ↓ |",
        "|---|---:|---:|---:|---:|",
        *rows,
        "",
    ]
    paired_delta = float(difference_stats["paired_angular_error_deg"]["mean"])
    score_delta = float(difference_stats["distribution_score"]["mean"])
    report.append(
        f"The direct regressor changes paired angle by {paired_delta:+.3f}° and "
        f"Distribution Score by {score_delta:+.4f} relative to matched CFM."
    )
    if cno_delta is not None:
        report.append(
            f"Against the supplied Table-1 CNO-FM paired angle "
            f"({float(args.cno_paired_angle_deg):.2f}°), its difference is {cno_delta:+.3f}°."
        )
    report.append("")
    _atomic_text(output / "REPORT.md", "\n".join(report))
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
