#!/usr/bin/env python3
"""Aggregate and plot x5 semigroup-consistency and handoff-robustness results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _base_id(run_id: str) -> str:
    match = re.search(r"(?:^|_)base(\d+)(?:_|$)", run_id)
    return f"base{match.group(1)}" if match else run_id


def _base_weighted_mean(rows: list[dict[str, Any]], key: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[_base_id(str(row["run_id"]))].append(float(row[key]))
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def _cluster_bootstrap_mean(
    rows: list[dict[str, Any]],
    key: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[_base_id(str(row["run_id"]))].append(float(row[key]))
    clusters = sorted(grouped)
    if not clusters:
        raise RuntimeError(f"cannot bootstrap empty values for {key}")
    array = np.asarray(
        [np.mean(grouped[cluster], dtype=np.float64) for cluster in clusters],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        values[index] = float(array[selected].mean())
    return {
        "mean": float(array.mean()),
        "95ci": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
        "base_condition_clusters": len(clusters),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _duration_pair_summary(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pair = (round(float(row["left_ns"]), 6), round(float(row["right_ns"]), 6))
        grouped[pair].append(row)
    return [
        {
            "left_ns": pair[0],
            "right_ns": pair[1],
            "total_ns": pair[0] + pair[1],
            "cases": len(values),
            **{key: _base_weighted_mean(values, key) for key in keys},
        }
        for pair, values in sorted(grouped.items())
    ]


def _cluster_bootstrap_delta(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    key: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    def keyed(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], float]:
        out = {}
        for row in rows:
            item = (row["run_id"], row["start"], row["middle"], row["end"])
            out[item] = float(row[key])
        return out

    a, b = keyed(left), keyed(right)
    if set(a) != set(b):
        raise RuntimeError("semigroup models were not evaluated on identical paired cases")
    common = sorted(a)
    if not common:
        raise RuntimeError("no paired semigroup cases")
    clustered: dict[str, list[float]] = defaultdict(list)
    for item in common:
        clustered[_base_id(item[0])].append(a[item] - b[item])
    clusters = sorted(clustered)
    difference = np.asarray(
        [np.mean(clustered[cluster]) for cluster in clusters], dtype=np.float64
    )
    point = float(difference.mean())
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        chosen = rng.integers(0, len(clusters), size=len(clusters))
        values[index] = float(difference[chosen].mean())
    return {
        "left_minus_right": point,
        "left_minus_right_95ci": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "probability_right_smaller": float(np.mean(values > 0.0)),
        "paired_cases": len(common),
        "base_condition_clusters": len(clusters),
    }


def _cluster_bootstrap_handoff_delta(
    rows: list[dict[str, Any]],
    key: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    by_run_lambda: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        weight = round(float(row["lambda"]), 9)
        if weight in (0.0, 1.0):
            by_run_lambda[(str(row["run_id"]), weight)].append(float(row[key]))
    run_ids = sorted({run_id for run_id, _weight in by_run_lambda})
    if not run_ids:
        raise RuntimeError(f"no lambda endpoint pairs for {key}")
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for run_id in run_ids:
        true_values = by_run_lambda.get((run_id, 0.0), [])
        predicted_values = by_run_lambda.get((run_id, 1.0), [])
        if not true_values or len(true_values) != len(predicted_values):
            raise RuntimeError(f"unpaired handoff endpoints for {run_id}, metric {key}")
        delta = float(np.mean(predicted_values) - np.mean(true_values))
        by_cluster[_base_id(run_id)].append(delta)
    clusters = sorted(by_cluster)
    array = np.asarray(
        [np.mean(by_cluster[cluster], dtype=np.float64) for cluster in clusters],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        samples[index] = float(array[selected].mean())
    return {
        "definition": "lambda=1 predicted-boundary error minus lambda=0 true-boundary error",
        "mean": float(array.mean()),
        "95ci": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "probability_degradation_positive": float(np.mean(samples > 0.0)),
        "base_condition_clusters": len(clusters),
        "trajectories": len(run_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=("stage1", "mixed"))
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    distributions = {
        label: _read(args.input_root / label / "semigroup_distribution.csv")
        for label in args.labels
    }
    handoff = {
        label: _read(args.input_root / label / "handoff_curve.csv")
        for label in args.labels
    }
    sg_keys = (
        "mean_sg_defect_ang",
        "ensemble_energy_distance_ang",
        "q_w1",
        "energy_w1",
    )
    semigroup_summary: dict[str, Any] = {}
    semigroup_rows: list[dict[str, Any]] = []
    for label_index, (label, rows) in enumerate(distributions.items()):
        estimates = {
            key: _cluster_bootstrap_mean(
                rows,
                key,
                iterations=args.bootstrap,
                seed=args.seed + 10_007 * label_index + 1_009 * key_index,
            )
            for key_index, key in enumerate(sg_keys)
        }
        semigroup_summary[label] = {
            **{key: estimate["mean"] for key, estimate in estimates.items()},
            "confidence_intervals": {
                key: estimate["95ci"] for key, estimate in estimates.items()
            },
            "base_condition_clusters": next(iter(estimates.values()))[
                "base_condition_clusters"
            ],
            "cases": len(rows),
            "duration_pairs": _duration_pair_summary(rows, sg_keys),
            "max_non_duration_condition_relative_delta": max(
                float(row.get("non_duration_condition_max_relative_delta", 0.0))
                for row in rows
            ),
        }
        table_row: dict[str, Any] = {
            "model": label,
            "cases": len(rows),
            "base_condition_clusters": semigroup_summary[label][
                "base_condition_clusters"
            ],
            "max_non_duration_condition_relative_delta": semigroup_summary[label][
                "max_non_duration_condition_relative_delta"
            ],
        }
        for key, estimate in estimates.items():
            table_row[key] = estimate["mean"]
            table_row[f"{key}_ci95_low"] = estimate["95ci"][0]
            table_row[f"{key}_ci95_high"] = estimate["95ci"][1]
        semigroup_rows.append(table_row)
    _write_csv(args.output_dir / "semigroup_main_table.csv", semigroup_rows)
    comparisons = {}
    if len(args.labels) >= 2:
        left = args.labels[0]
        for label_index, right in enumerate(args.labels[1:]):
            comparisons[f"{left}_vs_{right}"] = {
                key: _cluster_bootstrap_delta(
                    distributions[left],
                    distributions[right],
                    key,
                    args.bootstrap,
                    args.seed + 101 * label_index + 997 * key_index,
                )
                for key_index, key in enumerate(sg_keys)
            }
    comparison_rows: list[dict[str, Any]] = []
    for comparison, metrics in comparisons.items():
        left, right = comparison.split("_vs_", maxsplit=1)
        for metric, result in metrics.items():
            comparison_rows.append(
                {
                    "comparison": comparison,
                    "left": left,
                    "right": right,
                    "metric": metric,
                    "left_minus_right": result["left_minus_right"],
                    "ci95_low": result["left_minus_right_95ci"][0],
                    "ci95_high": result["left_minus_right_95ci"][1],
                    "probability_right_smaller": result[
                        "probability_right_smaller"
                    ],
                    "base_condition_clusters": result[
                        "base_condition_clusters"
                    ],
                    "paired_cases": result["paired_cases"],
                }
            )
    _write_csv(args.output_dir / "semigroup_paired_comparisons.csv", comparison_rows)

    handoff_summary: dict[str, Any] = {}
    handoff_table_rows: list[dict[str, Any]] = []
    for label, rows in handoff.items():
        grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[float(row["lambda"])].append(row)
        curve = []
        for weight_index, weight in enumerate(sorted(grouped)):
            item: dict[str, Any] = {"lambda": weight}
            for metric_index, metric in enumerate(
                ("output_ang", "output_mse", "output_q_abs", "input_ang")
            ):
                estimate = _cluster_bootstrap_mean(
                    grouped[weight],
                    metric,
                    iterations=args.bootstrap,
                    seed=(
                        args.seed
                        + 100_003 * args.labels.index(label)
                        + 7_919 * weight_index
                        + 1_009 * metric_index
                    ),
                )
                item[metric] = estimate["mean"]
                item[f"{metric}_95ci"] = estimate["95ci"]
                item["base_condition_clusters"] = estimate[
                    "base_condition_clusters"
                ]
            curve.append(item)
            handoff_table_rows.append(
                {
                    "model": label,
                    "lambda": weight,
                    "base_condition_clusters": item[
                        "base_condition_clusters"
                    ],
                    **{
                        metric: item[metric]
                        for metric in (
                            "output_ang",
                            "output_mse",
                            "output_q_abs",
                            "input_ang",
                        )
                    },
                    **{
                        f"{metric}_ci95_low": item[f"{metric}_95ci"][0]
                        for metric in (
                            "output_ang",
                            "output_mse",
                            "output_q_abs",
                            "input_ang",
                        )
                    },
                    **{
                        f"{metric}_ci95_high": item[f"{metric}_95ci"][1]
                        for metric in (
                            "output_ang",
                            "output_mse",
                            "output_q_abs",
                            "input_ang",
                        )
                    },
                }
            )
        weights = np.asarray([row["lambda"] for row in curve], dtype=np.float64)
        angles = np.asarray([row["output_ang"] for row in curve], dtype=np.float64)
        degradation = {
            metric: _cluster_bootstrap_handoff_delta(
                rows,
                metric,
                iterations=args.bootstrap,
                seed=(
                    args.seed
                    + 500_009 * args.labels.index(label)
                    + 10_007 * metric_index
                ),
            )
            for metric_index, metric in enumerate(
                ("output_ang", "output_mse", "output_q_abs")
            )
        }
        handoff_summary[label] = {
            "curve": curve,
            "angle_auc": float(np.trapz(angles, weights)),
            "angle_degradation_lambda1_minus0": float(angles[-1] - angles[0]),
            "paired_endpoint_degradation": degradation,
        }
    _write_csv(args.output_dir / "handoff_curve_table.csv", handoff_table_rows)

    summary = {
        "status": "complete",
        "semigroup": semigroup_summary,
        "semigroup_comparisons": comparisons,
        "handoff": handoff_summary,
        "bootstrap_iterations": args.bootstrap,
        "bootstrap_unit": "held-out base condition",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    for label in args.labels:
        curve = handoff_summary[label]["curve"]
        x = [row["lambda"] for row in curve]
        axes[0].plot(x, [row["output_ang"] for row in curve], marker="o", label=label)
        axes[1].plot(x, [row["output_q_abs"] for row in curve], marker="o", label=label)
    axes[0].set_ylabel("output angular error (degree)")
    axes[1].set_ylabel("output |delta Q|")
    for axis in axes:
        axis.set_xlabel("handoff interpolation lambda (0=true, 1=predicted)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(args.output_dir / "handoff_robustness.png", dpi=240)
    plt.close(figure)
    markdown = [
        "# Semigroup consistency and handoff robustness",
        "",
        "Point estimates average base-condition means; brackets are 95% cluster-bootstrap intervals.",
        "",
        "| Model | SG angular defect | Ensemble angular distance | Q Wasserstein-1 |",
        "|---|---:|---:|---:|",
    ]
    for row in semigroup_rows:
        markdown.append(
            f"| {row['model']} | {row['mean_sg_defect_ang']:.4f} "
            f"[{row['mean_sg_defect_ang_ci95_low']:.4f}, {row['mean_sg_defect_ang_ci95_high']:.4f}] | "
            f"{row['ensemble_energy_distance_ang']:.4f} "
            f"[{row['ensemble_energy_distance_ang_ci95_low']:.4f}, {row['ensemble_energy_distance_ang_ci95_high']:.4f}] | "
            f"{row['q_w1']:.4f} [{row['q_w1_ci95_low']:.4f}, {row['q_w1_ci95_high']:.4f}] |"
        )
    (args.output_dir / "README.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
