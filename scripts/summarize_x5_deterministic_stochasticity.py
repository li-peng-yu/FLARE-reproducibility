#!/usr/bin/env python3
"""Consolidate deep-ensemble and anchor-jitter exact-path experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
METHODS = ("direct_unet", "cno_fm", "dpot_ti", "poseidon_t", "pdearena_unet")
JITTERS = (("0p25", 0.25), ("0p5", 0.5), ("1p0", 1.0))


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        raise RuntimeError(f"incomplete result: {path}")
    return value


def _quality(summary: dict[str, Any]) -> dict[str, Any]:
    score = summary["score"]["combined"]
    angular = summary["angular_energy_distance"]["combined"]
    paired = summary["paired_angular_error"]["combined"]
    return {
        "distribution_score": float(score["geometric_mean_score"]),
        "distribution_score_ci": score["geometric_mean_bootstrap_95ci"],
        "angular_energy_distance_deg": float(angular["mean"]),
        "angular_energy_distance_deg_ci": angular["bootstrap_95ci"],
        "paired_angle_deg": float(paired["mean"]),
        "paired_angle_deg_ci": paired["bootstrap_95ci"],
    }


def _fair(summary: dict[str, Any], label: str) -> dict[str, Any]:
    combined = summary["methods"][label]["combined"]
    score = combined["fair_energy_score"]
    spread = combined["ensemble_spread_rms_chord"]
    return {
        "fair_energy_score": float(score["mean"]),
        "fair_energy_score_ci": score["bootstrap_95ci"],
        "ensemble_spread_rms_chord": float(spread["mean"]),
        "ensemble_spread_rms_chord_ci": spread["bootstrap_95ci"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT / "outputs/x5_deterministic_stochasticity_20260903",
    )
    parser.add_argument(
        "--table1-summary",
        type=Path,
        default=PROJECT / "reports/x5_exact_control_rollout_seed78_20260903.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "reports/x5_deterministic_stochasticity_20260903.json",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    table1 = _json(args.table1_summary.resolve())
    base_rows = {str(row["method"]): row for row in table1["rows"]}

    rows: list[dict[str, Any]] = []
    for method in METHODS:
        base = base_rows[method]
        rows.append(
            {
                "method": method,
                "strategy": "deterministic_table1_seed78",
                "jitter_rms_deg": None,
                "distribution_score": base["distribution_score"],
                "distribution_score_ci": base["distribution_score_ci"],
                "angular_energy_distance_deg": base["angular_energy_distance_deg"],
                "angular_energy_distance_deg_ci": base[
                    "angular_energy_distance_deg_ci"
                ],
                "paired_angle_deg": base["paired_angle_deg"],
                "paired_angle_deg_ci": base["paired_angle_deg_ci"],
                "fair_energy_score": base["fair_energy_score"],
                "fair_energy_score_ci": base["fair_energy_score_ci"],
                "ensemble_spread_rms_chord": 0.0,
                "ensemble_spread_rms_chord_ci": [0.0, 0.0],
            }
        )

        deep_root = root / "deep_ensemble" / method
        deep_label = f"{method}_deep3"
        deep = {
            **_quality(_json(deep_root / "distribution/summary/run_summary.json")),
            **_fair(_json(deep_root / "fair_energy/run_summary.json"), deep_label),
        }
        rows.append(
            {
                "method": method,
                "strategy": "deep_ensemble_3seed",
                "jitter_rms_deg": None,
                **deep,
            }
        )

        for tag, rms_deg in JITTERS:
            jitter_root = root / "anchor_jitter" / method / f"rms_{tag}"
            label = f"{method}_jitter_{tag}"
            jitter = {
                **_quality(
                    _json(jitter_root / "distribution/summary/run_summary.json")
                ),
                **_fair(_json(jitter_root / "fair_energy/run_summary.json"), label),
            }
            rows.append(
                {
                    "method": method,
                    "strategy": "anchor_jitter_5draw",
                    "jitter_rms_deg": rms_deg,
                    **jitter,
                }
            )

    by_method: dict[str, Any] = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        baseline = selected[0]
        augmented = []
        for row in selected[1:]:
            candidate = dict(row)
            candidate["delta_vs_deterministic"] = {
                "distribution_score": (
                    float(row["distribution_score"])
                    - float(baseline["distribution_score"])
                ),
                "angular_energy_distance_deg": (
                    float(row["angular_energy_distance_deg"])
                    - float(baseline["angular_energy_distance_deg"])
                ),
                "paired_angle_deg": (
                    float(row["paired_angle_deg"])
                    - float(baseline["paired_angle_deg"])
                ),
                "fair_energy_score": (
                    float(row["fair_energy_score"])
                    - float(baseline["fair_energy_score"])
                ),
            }
            augmented.append(candidate)
        by_method[method] = {"deterministic": baseline, "stochasticized": augmented}

    payload = {
        "status": "complete",
        "protocol": {
            "quality": "Table-1 exact-control complete path with five model outputs",
            "deep_ensemble": (
                "three independently trained seeds; five-output quality budget uses a "
                "target-independent balanced member assignment; fair energy uses all "
                "three unique members per exact anchor"
            ),
            "anchor_jitter": (
                "five smooth tangent-plane perturbations per exact anchor, Gaussian "
                "correlation length 4 px, followed by S2 exponential-map projection"
            ),
            "jitter_grid_rms_deg": [value for _tag, value in JITTERS],
        },
        "methods": by_method,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    csv_path = args.output.with_suffix(".csv")
    flat_rows = []
    for row in rows:
        flat_rows.append(
            {
                key: (
                    ";".join(str(item) for item in value)
                    if isinstance(value, list)
                    else value
                )
                for key, value in row.items()
            }
        )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

