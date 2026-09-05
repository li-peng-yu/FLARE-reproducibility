#!/usr/bin/env python3
"""Summarize seen/unseen target-duration interpolation on the x5 test split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = ("ang", "mse", "q_abs")
SEEN = {1.0, 2.0, 3.0, 5.0}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty interpolation input: {path}")
    return rows


def _case_means(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["run_id"],
            row["control_segment_index"],
            row["start_time_ns"],
            row["requested_duration_ns"],
        )
        grouped[key].append(row)
    out = []
    for values in grouped.values():
        first = values[0]
        duration = round(float(first["requested_duration_ns"]), 6)
        out.append(
            {
                "base_id": first["base_id"],
                "run_id": first["run_id"],
                "control_segment_index": int(first["control_segment_index"]),
                "start_time_ns": round(float(first["start_time_ns"]), 6),
                "duration_ns": duration,
                "split": "seen" if duration in SEEN else "unseen",
                **{
                    metric: float(np.mean([float(row[metric]) for row in values]))
                    for metric in METRICS
                },
            }
        )
    return out


def _case_key(row: dict[str, Any]) -> tuple[str, int, float, float]:
    return (
        str(row["run_id"]),
        int(row["control_segment_index"]),
        round(float(row["start_time_ns"]), 6),
        round(float(row["duration_ns"]), 6),
    )


def _paired_cluster_delta(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    metric: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap a same-case difference after averaging within base condition."""
    left_cases = {_case_key(row): row for row in left}
    right_cases = {_case_key(row): row for row in right}
    if set(left_cases) != set(right_cases):
        raise RuntimeError("interpolation models were evaluated on different case keys")
    grouped: dict[str, list[float]] = defaultdict(list)
    for key in sorted(left_cases):
        left_row = left_cases[key]
        right_row = right_cases[key]
        if str(left_row["base_id"]) != str(right_row["base_id"]):
            raise RuntimeError("base-condition mismatch in interpolation comparison")
        grouped[str(left_row["base_id"])].append(
            float(left_row[metric]) - float(right_row[metric])
        )
    clusters = sorted(grouped)
    values = np.asarray(
        [np.mean(grouped[cluster], dtype=np.float64) for cluster in clusters],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        chosen = rng.integers(0, len(values), size=len(values))
        bootstrap[index] = float(values[chosen].mean())
    return {
        "definition": "left minus right; positive means right is lower",
        "left_minus_right": float(values.mean()),
        "left_minus_right_95ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "probability_right_lower": float(np.mean(bootstrap > 0.0)),
        "base_condition_clusters": len(clusters),
        "paired_cases": len(left_cases),
    }


def _cluster_estimate(
    rows: list[dict[str, Any]], key: str, *, iterations: int, seed: int
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["base_id"])].append(float(row[key]))
    clusters = sorted(grouped)
    if not clusters:
        raise RuntimeError(f"empty interpolation stratum for {key}")
    array = np.asarray([np.mean(grouped[item]) for item in clusters], dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        chosen = rng.integers(0, len(array), size=len(array))
        bootstrap[index] = float(array[chosen].mean())
    return {
        "mean": float(array.mean()),
        "95ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "base_condition_clusters": len(clusters),
        "cases": len(rows),
    }


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=("stage1", "no_target_time", "standard_prior", "cartesian", "direct", "fno"),
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        label: _case_means(_read(args.input_dir / f"{label}.csv"))
        for label in args.labels
    }
    durations = sorted({row["duration_ns"] for rows in data.values() for row in rows})
    table: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for label_index, label in enumerate(args.labels):
        summary[label] = {}
        for duration_index, duration in enumerate(durations):
            selected = [row for row in data[label] if row["duration_ns"] == duration]
            if not selected:
                continue
            estimates = {
                metric: _cluster_estimate(
                    selected,
                    metric,
                    iterations=args.bootstrap,
                    seed=args.seed + 100_003 * label_index + 7_919 * duration_index + metric_index,
                )
                for metric_index, metric in enumerate(METRICS)
            }
            summary[label][str(duration)] = estimates
            table.append(
                {
                    "model": label,
                    "requested_duration_ns": duration,
                    "duration_status": "seen" if duration in SEEN else "unseen_interpolation",
                    **{metric: estimates[metric]["mean"] for metric in METRICS},
                    **{f"{metric}_ci95_low": estimates[metric]["95ci"][0] for metric in METRICS},
                    **{f"{metric}_ci95_high": estimates[metric]["95ci"][1] for metric in METRICS},
                    "cases": estimates["ang"]["cases"],
                    "base_condition_clusters": estimates["ang"]["base_condition_clusters"],
                }
            )
    _write(args.output_dir / "time_interpolation_table.csv", table)

    # Same-case intervals for the target-time ablation.
    comparisons: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    pairs = [
        ("no_target_time", "stage1"),
        ("cartesian", "stage1"),
        ("fno", "stage1"),
    ]
    pairs = [pair for pair in pairs if pair[0] in data and pair[1] in data]
    for pair_index, (left, right) in enumerate(pairs):
        comparison_name = f"{left}_vs_{right}"
        comparisons[comparison_name] = {}
        for duration_index, duration in enumerate(durations):
            left_rows = [row for row in data[left] if row["duration_ns"] == duration]
            right_rows = [row for row in data[right] if row["duration_ns"] == duration]
            if not left_rows and not right_rows:
                continue
            duration_result: dict[str, Any] = {}
            for metric_index, metric in enumerate(METRICS):
                result = _paired_cluster_delta(
                    left_rows,
                    right_rows,
                    metric,
                    iterations=args.bootstrap,
                    seed=(
                        args.seed
                        + 104_729 * pair_index
                        + 7_919 * duration_index
                        + 1_543 * metric_index
                    ),
                )
                duration_result[metric] = result
                comparison_rows.append(
                    {
                        "comparison": comparison_name,
                        "left": left,
                        "right": right,
                        "requested_duration_ns": duration,
                        "duration_status": "seen" if duration in SEEN else "unseen_interpolation",
                        "metric": metric,
                        "left_minus_right": result["left_minus_right"],
                        "ci95_low": result["left_minus_right_95ci"][0],
                        "ci95_high": result["left_minus_right_95ci"][1],
                        "probability_right_lower": result["probability_right_lower"],
                        "base_condition_clusters": result["base_condition_clusters"],
                        "paired_cases": result["paired_cases"],
                    }
                )
            comparisons[comparison_name][str(duration)] = duration_result
    if comparison_rows:
        _write(
            args.output_dir / "time_interpolation_paired_comparisons.csv",
            comparison_rows,
        )

    figure, axis = plt.subplots(figsize=(7.4, 5.1))
    for label in args.labels:
        selected = sorted(
            (row for row in table if row["model"] == label),
            key=lambda row: row["requested_duration_ns"],
        )
        axis.plot(
            [row["requested_duration_ns"] for row in selected],
            [row["ang"] for row in selected],
            marker="o",
            label=label,
        )
    for duration in durations:
        if duration not in SEEN:
            axis.axvline(duration, color="#BBBBBB", linewidth=0.7, linestyle=":")
    axis.set_xlabel("requested duration (ns); dotted = held-out interpolation")
    axis.set_ylabel("angular error (degree)")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(args.output_dir / "time_interpolation_angle.png", dpi=240)
    plt.close(figure)

    payload = {
        "status": "complete",
        "seen_training_durations_ns": sorted(SEEN),
        "unseen_interpolation_durations_ns": [value for value in durations if value not in SEEN],
        "protocol": "balanced test cases per requested duration; base-condition cluster bootstrap",
        "models": summary,
        "paired_comparisons": comparisons,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    markdown = [
        "# Target-time interpolation",
        "",
        "Dotted-duration entries were held out as target durations. "
        "Intervals are 95% base-condition-cluster bootstrap intervals.",
        "",
        "| Model | Duration (ns) | Status | Angle (95% CI) | MSE | |Delta Q| |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in table:
        status = "seen" if row["duration_status"] == "seen" else "unseen interpolation"
        markdown.append(
            f"| {row['model']} | {row['requested_duration_ns']:g} | {status} | "
            f"{row['ang']:.4f} [{row['ang_ci95_low']:.4f}, {row['ang_ci95_high']:.4f}] | "
            f"{row['mse']:.5f} | {row['q_abs']:.4f} |"
        )
    (args.output_dir / "README.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
