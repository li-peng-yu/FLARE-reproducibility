#!/usr/bin/env python3
"""Aggregate complete-path geometry/source results over seeds 78/79/80."""

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
    / "outputs/skx_bt_1000base_x5/paper_revision_20260830/"
    "exact_control_seed_geometry"
)
REFERENCE_MODE = "exact_control_multisegment_rollout_endpoint"
FAMILIES = {
    "core": {
        "display": "Core local axis--angle",
        "labels": ("core_s78", "core_s79", "core_s80"),
    },
    "cartesian_cfm": {
        "display": "Cartesian CFM",
        "labels": ("cart_s78", "cart_s79", "cart_s80"),
    },
    "tangent_projected_prior": {
        "display": "Tangent-projected source",
        "labels": ("tan_s78", "tan_s79", "tan_s80"),
    },
    "fixed_2d_tangent_basis": {
        "display": "Fixed 2D tangent basis",
        "labels": ("a2d_s78", "a2d_s79", "a2d_s80"),
    },
    "riemannian_fm": {
        "display": "Riemannian FM",
        "labels": ("rfm_s78", "rfm_s79", "rfm_s80"),
    },
}
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


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


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


def _seed(label: str) -> int:
    value = int(label[-2:])
    if value not in (78, 79, 80):
        raise ValueError(f"unexpected seed label: {label}")
    return value


def _load_seed(root: Path, fair: dict[str, Any], label: str) -> dict[str, Any]:
    distribution_path = root / "distribution" / label / "summary/run_summary.json"
    distribution = _json(distribution_path)
    if distribution.get("reference_mode") != REFERENCE_MODE:
        raise RuntimeError(f"{label}: not an exact-control rollout")
    if int(distribution.get("conditions", -1)) != 33:
        raise RuntimeError(f"{label}: expected 33 complete paths")
    if distribution.get("rollout") is not True:
        raise RuntimeError(f"{label}: rollout flag is not true")
    model_input = str(distribution.get("model_input", ""))
    if (
        "previous model prediction" not in model_input
        and "model-prediction handoff" not in model_input
    ):
        raise RuntimeError(f"{label}: prediction-only handoff is not documented")

    # Validate the exact-anchor
    # samples used by fair energy from their per-condition metadata and
    # manifests.  This prevents a single-segment sample pool from being paired
    # with a correctly labelled distribution summary.
    exact_root = root / "exact_anchor" / label
    exact_summary = _json(exact_root / "summary/run_summary.json")
    if exact_summary.get("multisegment_rollout") is not True:
        raise RuntimeError(f"{label}: exact-anchor generation is not a multisegment rollout")
    if int(exact_summary.get("conditions", -1)) != 33:
        raise RuntimeError(f"{label}: expected 33 exact-anchor rollout conditions")
    condition_dirs = sorted(
        path for path in (exact_root / "conditions").glob("*") if path.is_dir()
    )
    if len(condition_dirs) != 33:
        raise RuntimeError(f"{label}: expected 33 exact-anchor condition directories")
    for condition_dir in condition_dirs:
        metadata = _json(condition_dir / "condition_metadata.json")
        manifest = _json(condition_dir / "model_samples/manifest.json")
        if metadata.get("reference_mode") != REFERENCE_MODE:
            raise RuntimeError(f"{label}: wrong exact-anchor reference mode in {condition_dir}")
        if int(metadata.get("segments_per_trajectory", -1)) != 2:
            raise RuntimeError(f"{label}: exact-anchor path is not two segments in {condition_dir}")
        if int(metadata.get("handoff_count", -1)) != 1:
            raise RuntimeError(f"{label}: exact-anchor path lacks one handoff in {condition_dir}")
        if metadata.get("teacher_forcing_after_first_segment") is not False:
            raise RuntimeError(f"{label}: exact-anchor handoff is teacher-forced in {condition_dir}")
        if "previous model prediction" not in str(metadata.get("model_input", "")):
            raise RuntimeError(f"{label}: exact-anchor model input is not prediction-only in {condition_dir}")
        if manifest.get("reference_mode") != REFERENCE_MODE:
            raise RuntimeError(f"{label}: wrong sample-manifest reference mode in {condition_dir}")
        if int(manifest.get("segment_count", -1)) != 2:
            raise RuntimeError(f"{label}: sample manifest is not two segments in {condition_dir}")
        if int(manifest.get("handoff_count", -1)) != 1:
            raise RuntimeError(f"{label}: sample manifest lacks one handoff in {condition_dir}")
        if manifest.get("teacher_forcing_after_first_segment") is not False:
            raise RuntimeError(f"{label}: sample manifest records teacher forcing in {condition_dir}")

    fair_method = fair["methods"][label]
    fair_combined = fair_method["combined"]["fair_energy_score"]
    if int(fair_combined.get("base_groups", -1)) != 33:
        raise RuntimeError(f"{label}: fair score does not contain 33 base groups")
    if int(fair_combined.get("anchor_cases", -1)) != 165:
        raise RuntimeError(f"{label}: fair score does not contain 165 anchors")
    return {
        "label": label,
        "seed": _seed(label),
        "distribution_score": float(
            distribution["score"]["combined"]["geometric_mean_score"]
        ),
        "angular_energy_distance_deg": float(
            distribution["angular_energy_distance"]["combined"]["mean"]
        ),
        "paired_angular_error_deg": float(
            distribution["paired_angular_error"]["combined"]["mean"]
        ),
        "fair_energy_score": float(fair_combined["mean"]),
        "checkpoint": str(distribution["checkpoint"]),
        "checkpoint_sha256": str(distribution["checkpoint_sha256"]),
        "distribution_source": str(distribution_path),
        "fair_energy_source": str(root / "fair_energy/run_summary.json"),
    }


def _stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"expected three finite values, got {values}")
    return {
        "seeds": 3,
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _latex_value(metric: str, stats: dict[str, float | int]) -> str:
    digits = 4 if metric in ("distribution_score", "fair_energy_score") else 3
    mean = float(stats["mean"])
    std = float(stats["sample_std"])
    return f"${mean:.{digits}f}\\pm{std:.{digits}f}$"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = (args.output or (root / "summary")).expanduser().resolve()
    fair_path = root / "fair_energy/run_summary.json"
    fair = _json(fair_path)
    if int(fair.get("bootstrap_iterations", -1)) != 50000:
        raise RuntimeError("fair-energy merge did not use 50,000 bootstrap draws")

    seed_rows: list[dict[str, Any]] = []
    for family, spec in FAMILIES.items():
        for label in spec["labels"]:
            seed_rows.append(
                {
                    "family": family,
                    "display": spec["display"],
                    **_load_seed(root, fair, label),
                }
            )

    family_payload: dict[str, Any] = {}
    family_rows: list[dict[str, Any]] = []
    latex_rows: list[str] = []
    for family, spec in FAMILIES.items():
        selected = [row for row in seed_rows if row["family"] == family]
        metrics = {
            metric: _stats([float(row[metric]) for row in selected])
            for metric in METRICS
        }
        family_payload[family] = {
            "display": spec["display"],
            "metrics": metrics,
        }
        flat: dict[str, Any] = {
            "family": family,
            "display": spec["display"],
            "seeds": 3,
        }
        for metric, stats in metrics.items():
            flat[f"{metric}_mean"] = stats["mean"]
            flat[f"{metric}_sample_std"] = stats["sample_std"]
        family_rows.append(flat)
        latex_rows.append(
            " & ".join(
                [
                    str(spec["display"]),
                    _latex_value("distribution_score", metrics["distribution_score"]),
                    _latex_value(
                        "angular_energy_distance_deg",
                        metrics["angular_energy_distance_deg"],
                    ),
                    _latex_value(
                        "paired_angular_error_deg",
                        metrics["paired_angular_error_deg"],
                    ),
                    _latex_value("fair_energy_score", metrics["fair_energy_score"]),
                ]
            )
            + r" \\"
        )

    by_family_seed = {
        (str(row["family"]), int(row["seed"])): row for row in seed_rows
    }
    difference_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        if family == "core":
            continue
        for seed in (78, 79, 80):
            core = by_family_seed[("core", seed)]
            candidate = by_family_seed[(family, seed)]
            difference_rows.append(
                {
                    "family": family,
                    "seed": seed,
                    **{
                        f"{metric}_ablation_minus_core": (
                            float(candidate[metric]) - float(core[metric])
                        )
                        for metric in METRICS
                    },
                }
            )

    payload = {
        "status": "complete",
        "protocol": {
            "reference_mode": REFERENCE_MODE,
            "path": (
                "saved drive-start state -> exact driven-control segment -> "
                "prediction-only handoff -> exact 3.5-ns zero-current relaxation"
            ),
            "base_groups": 33,
            "complete_paths_per_checkpoint": 165,
            "fair_energy_anchors_per_checkpoint": 165,
            "training_seeds": [78, 79, 80],
            "evaluation_seed": 208300160,
        },
        "families": family_payload,
        "per_seed": seed_rows,
        "matched_seed_ablation_minus_core": difference_rows,
    }
    _atomic_json(output / "summary.json", payload)
    _atomic_csv(output / "per_seed.csv", seed_rows)
    _atomic_csv(output / "family_summary.csv", family_rows)
    _atomic_csv(output / "matched_seed_differences.csv", difference_rows)
    _atomic_text(output / "table_rows.tex", "\n".join(latex_rows) + "\n")

    report = [
        "# Complete-path three-seed geometry/source audit",
        "",
        "All rows use an exact-control prediction-only handoff and are reported "
        "as mean ± sample standard deviation over seeds 78/79/80.",
        "",
        "| Family | Distribution Score | Angular ED | Paired angle | Fair ES |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        item = family_payload[family]
        metrics = item["metrics"]
        report.append(
            f"| {item['display']} | "
            + " | ".join(
                f"{float(metrics[metric]['mean']):.4f} ± "
                f"{float(metrics[metric]['sample_std']):.4f}"
                for metric in METRICS
            )
            + " |"
        )
    report.append("")
    _atomic_text(output / "REPORT.md", "\n".join(report))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
