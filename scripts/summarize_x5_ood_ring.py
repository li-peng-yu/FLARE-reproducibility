#!/usr/bin/env python3
"""Aggregate ID-versus-ring-OOD single-step and fully-AR results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty OOD input: {path}")
    return rows


def _base_id(run_id: str) -> str:
    match = re.search(r"(?:^|_)base(\d+)(?:_|$)", run_id)
    return f"base{match.group(1)}" if match else run_id


def _estimate(rows: list[dict[str, Any]], key: str, iterations: int, seed: int) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[_base_id(str(row["run_id"]))].append(float(row[key]))
    clusters = sorted(grouped)
    if not clusters:
        raise RuntimeError(f"empty OOD cluster set for {key}")
    array = np.asarray([np.mean(grouped[item]) for item in clusters], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        chosen = rng.integers(0, len(array), len(array))
        samples[index] = float(array[chosen].mean())
    return {
        "mean": float(array.mean()),
        "95ci": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "base_condition_clusters": len(clusters),
        "rows": len(rows),
    }


def _case_means(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["run_id"], row["control_segment_index"], row["start_time_ns"], row["end_time_ns"])].append(row)
    return [
        {
            "run_id": values[0]["run_id"],
            **{
                metric: float(np.mean([float(row[metric]) for row in values]))
                for metric in ("ang", "mse", "q_abs")
            },
        }
        for values in grouped.values()
    ]


def _rollout_means(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["run_id"]].append(row)
    result = []
    for run_id, values in grouped.items():
        final_position = max(int(row["segment_position"]) for row in values)
        final = [row for row in values if int(row["segment_position"]) == final_position]
        result.append(
            {
                "run_id": run_id,
                "tf_ang": float(np.mean([float(row["tf_ang"]) for row in values])),
                "ar_ang": float(np.mean([float(row["ar_ang"]) for row in values])),
                "gap_ang": float(np.mean([float(row["gap_ang"]) for row in values])),
                "final_ar_ang": float(np.mean([float(row["ar_ang"]) for row in final])),
                "final_gap_ang": float(np.mean([float(row["gap_ang"]) for row in final])),
            }
        )
    return result


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty table: {path}")
    # Single-segment and rollout rows intentionally expose different metrics.
    # Preserve a stable first-seen ordering while allowing both schemas in the
    # paper-facing comparison table.
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _estimate_cell(estimate: dict[str, Any]) -> str:
    return (
        f"{estimate['mean']:.4f} "
        f"[{estimate['95ci'][0]:.4f}, {estimate['95ci'][1]:.4f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    strata = {
        run_id: stratum
        for stratum, run_ids in manifest["test_strata"].items()
        for run_id in run_ids
    }

    single_paths = {
        "scfm": args.root / "single_segment" / "ood_scfm_ode10.csv",
        "fno": args.root / "single_segment" / "ood_fno_ode1.csv",
    }
    single = {label: _case_means(_read(path)) for label, path in single_paths.items()}
    rollout_paths = {
        "scfm": sorted((args.root / "rollout" / "raw").glob("ood_scfm_draw*.csv")),
        "fno": sorted((args.root / "rollout" / "raw").glob("ood_fno_draw*.csv")),
    }
    rollout = {
        label: _rollout_means([row for path in paths for row in _read(path)])
        for label, paths in rollout_paths.items()
    }
    table: list[dict[str, Any]] = []
    payload: dict[str, Any] = {"status": "complete", "single_segment": {}, "fully_autoregressive": {}}
    for phase_index, (phase, models, metrics) in enumerate(
        (
            ("single_segment", single, ("ang", "mse", "q_abs")),
            ("fully_autoregressive", rollout, ("tf_ang", "ar_ang", "gap_ang", "final_ar_ang", "final_gap_ang")),
        )
    ):
        for model_index, (model, rows) in enumerate(models.items()):
            payload[phase][model] = {}
            unknown = [row["run_id"] for row in rows if row["run_id"] not in strata]
            if unknown:
                raise RuntimeError(f"OOD rows absent from split manifest: {unknown[:3]}")
            for stratum_index, stratum in enumerate(("id_unseen_base", "ood_geometry")):
                selected = [row for row in rows if strata[row["run_id"]] == stratum]
                estimates = {
                    metric: _estimate(
                        selected,
                        metric,
                        args.bootstrap,
                        args.seed + 1_000_003 * phase_index + 100_003 * model_index + 7_919 * stratum_index + metric_index,
                    )
                    for metric_index, metric in enumerate(metrics)
                }
                payload[phase][model][stratum] = estimates
                table.append(
                    {
                        "phase": phase,
                        "model": model,
                        "stratum": stratum,
                        **{metric: estimates[metric]["mean"] for metric in metrics},
                        "base_condition_clusters": estimates[metrics[0]]["base_condition_clusters"],
                        "rows_or_trajectories": estimates[metrics[0]]["rows"],
                    }
                )
    _write(args.output_dir / "ood_id_comparison.csv", table)
    payload["protocol"] = {
        "held_out_geometry": manifest["holdout"],
        "counts": manifest["counts"],
        "split_seed": manifest["seed"],
        "leakage_unit": "base condition",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    markdown = [
        "# Held-out ring-geometry generalization",
        "",
        "All records from the same base condition remain in one split. "
        "Intervals are 95% base-condition-cluster bootstrap intervals.",
        "",
        "## Single-segment prediction",
        "",
        "| Model | Test stratum | Angle (95% CI) | MSE (95% CI) | |Delta Q| (95% CI) |",
        "|---|---|---:|---:|---:|",
    ]
    for model in ("scfm", "fno"):
        for stratum in ("id_unseen_base", "ood_geometry"):
            values = payload["single_segment"][model][stratum]
            markdown.append(
                f"| {model} | {stratum} | {_estimate_cell(values['ang'])} | "
                f"{_estimate_cell(values['mse'])} | {_estimate_cell(values['q_abs'])} |"
            )
    markdown.extend(
        [
            "",
            "## Fully autoregressive rollout",
            "",
            "| Model | Test stratum | TF angle (95% CI) | Fully-AR angle (95% CI) | ARG (95% CI) | Final AR angle (95% CI) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for model in ("scfm", "fno"):
        for stratum in ("id_unseen_base", "ood_geometry"):
            values = payload["fully_autoregressive"][model][stratum]
            markdown.append(
                f"| {model} | {stratum} | {_estimate_cell(values['tf_ang'])} | "
                f"{_estimate_cell(values['ar_ang'])} | {_estimate_cell(values['gap_ang'])} | "
                f"{_estimate_cell(values['final_ar_ang'])} |"
            )
    (args.output_dir / "README.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
