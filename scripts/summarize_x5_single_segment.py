#!/usr/bin/env python3
"""Aggregate the unified x5 single-segment main table and causal ablations."""

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


METRICS = ("ang", "mse", "q_abs", "q_fail", "energy_abs", "norm_abs")
DEFAULT_LABELS = (
    "stage1",
    "mixed",
    "cartesian",
    "direct",
    "fno",
    "no_target_time",
    "active_spatial",
    "standard_prior",
    "gt_only",
    "pred_only",
    "topology",
)
MAIN_STEPS = {
    label: (1 if label in {"direct", "fno"} else 10)
    for label in DEFAULT_LABELS
}


def _read(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for key in ("draw", "record_index", "control_segment_index"):
                row[key] = int(row[key])
            for key in ("start_time_ns", "end_time_ns", "duration_ns"):
                row[key] = float(row[key])
            for metric in METRICS:
                row[metric] = float(row[metric])
                persistence = f"persistence_{metric}"
                if persistence in row:
                    row[persistence] = float(row[persistence])
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no rows in {path}")
    return rows


def _case_key(row: dict[str, Any]) -> tuple[str, int, float, float]:
    return (
        str(row["run_id"]),
        int(row["control_segment_index"]),
        round(float(row["start_time_ns"]), 6),
        round(float(row["end_time_ns"]), 6),
    )


def _case_means(rows: list[dict[str, Any]], prefix: str = "") -> dict[tuple, dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_case_key(row)].append(row)
    out: dict[tuple, dict[str, Any]] = {}
    for key, values in grouped.items():
        first = values[0]
        item: dict[str, Any] = {
            "base_id": str(first["base_id"]),
            "run_id": str(first["run_id"]),
            "control_segment_index": int(first["control_segment_index"]),
            "duration_ns": float(first["duration_ns"]),
            "draws": len(values),
        }
        for metric in METRICS:
            item[metric] = float(np.mean([float(row[f"{prefix}{metric}"]) for row in values]))
        out[key] = item
    return out


def _cluster_array(cases: dict[tuple, dict[str, Any]], metric: str) -> tuple[list[str], np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in cases.values():
        grouped[str(row["base_id"])].append(float(row[metric]))
    clusters = sorted(grouped)
    return clusters, np.asarray(
        [np.mean(grouped[cluster], dtype=np.float64) for cluster in clusters],
        dtype=np.float64,
    )


def _mean_ci(values: np.ndarray, *, iterations: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selected = rng.integers(0, values.size, size=values.size)
        samples[index] = float(values[selected].mean())
    return {
        "mean": float(values.mean()),
        "95ci": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "clusters": int(values.size),
    }


def _paired_delta(
    left: dict[tuple, dict[str, Any]],
    right: dict[tuple, dict[str, Any]],
    metric: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if set(left) != set(right):
        raise RuntimeError("single-segment models were evaluated on different case keys")
    grouped: dict[str, list[float]] = defaultdict(list)
    for key in sorted(left):
        cluster = str(left[key]["base_id"])
        if cluster != str(right[key]["base_id"]):
            raise RuntimeError("base-condition mismatch in paired comparison")
        grouped[cluster].append(float(left[key][metric]) - float(right[key][metric]))
    clusters = sorted(grouped)
    values = np.asarray([np.mean(grouped[key]) for key in clusters], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        samples[index] = float(values[selected].mean())
    return {
        "definition": "left minus right; positive means right is lower",
        "left_minus_right": float(values.mean()),
        "left_minus_right_95ci": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "probability_right_lower": float(np.mean(samples > 0.0)),
        "clusters": len(clusters),
        "paired_cases": len(left),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_metric_figure(
    path: Path,
    labels: list[str],
    estimates: dict[str, Any],
) -> None:
    if not labels:
        raise RuntimeError(f"no models available for figure: {path}")
    width = max(10.0, 1.15 * len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(width, 4.9))
    for axis, metric, ylabel in (
        (axes[0], "ang", "angular error (degree)"),
        (axes[1], "q_abs", r"$|\Delta Q|$"),
    ):
        means = [estimates[label][metric]["mean"] for label in labels]
        intervals = [estimates[label][metric]["95ci"] for label in labels]
        errors = np.asarray(
            [
                [mean - ci[0] for mean, ci in zip(means, intervals, strict=True)],
                [ci[1] - mean for mean, ci in zip(means, intervals, strict=True)],
            ],
            dtype=np.float64,
        )
        positions = np.arange(len(labels))
        axis.bar(positions, means, color="#277DA1", alpha=0.88)
        axis.errorbar(
            positions, means, yerr=errors, fmt="none", color="black", capsize=3
        )
        axis.set_xticks(positions, labels, rotation=31, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.23)
    figure.tight_layout()
    figure.savefig(path, dpi=240)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        label: args.input_dir / f"{label}_ode{MAIN_STEPS[label]}.csv"
        for label in args.labels
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    raw = {label: _read(path) for label, path in paths.items()}
    cases = {label: _case_means(rows) for label, rows in raw.items()}
    keysets = [set(values) for values in cases.values()]
    if not all(keys == keysets[0] for keys in keysets[1:]):
        raise RuntimeError("single-segment evaluations do not share identical held-out cases")
    persistence = _case_means(raw["stage1"], prefix="persistence_")

    all_cases = {"persistence": persistence, **cases}
    estimates: dict[str, Any] = {}
    table_rows: list[dict[str, Any]] = []
    for label_index, (label, values) in enumerate(all_cases.items()):
        estimates[label] = {}
        row: dict[str, Any] = {"model": label, "cases": len(values)}
        for metric_index, metric in enumerate(METRICS):
            clusters, array = _cluster_array(values, metric)
            estimate = _mean_ci(
                array,
                iterations=args.bootstrap,
                seed=args.seed + 1009 * label_index + 7919 * metric_index,
            )
            estimates[label][metric] = estimate
            row[metric] = estimate["mean"]
            row[f"{metric}_ci95_low"] = estimate["95ci"][0]
            row[f"{metric}_ci95_high"] = estimate["95ci"][1]
            row["clusters"] = len(clusters)
        metadata_path = paths[label].with_suffix(".json") if label != "persistence" else None
        if metadata_path is not None and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            row["parameters"] = int(metadata["model_parameters"])
            row["latency_seconds_per_transition"] = float(
                metadata["latency_seconds_per_transition"]
            )
            row["peak_gpu_memory_bytes"] = int(metadata["peak_gpu_memory_bytes"])
            estimates[label]["metadata"] = metadata
        else:
            row["parameters"] = ""
            row["latency_seconds_per_transition"] = ""
            row["peak_gpu_memory_bytes"] = ""
        table_rows.append(row)
    _write_csv(args.output_dir / "single_segment_main_table.csv", table_rows)

    pairs = (
        ("persistence", "stage1"),
        ("direct", "stage1"),
        ("fno", "stage1"),
        ("no_target_time", "stage1"),
        # Representation is isolated against the formal Stage-1 model.
        ("cartesian", "stage1"),
        ("active_spatial", "stage1"),
        ("stage1", "gt_only"),
        ("stage1", "mixed"),
        ("gt_only", "mixed"),
        ("gt_only", "pred_only"),
        ("mixed", "pred_only"),
        ("mixed", "topology"),
    )
    comparisons: dict[str, Any] = {}
    for pair_index, (left, right) in enumerate(pairs):
        if left not in all_cases or right not in all_cases:
            continue
        comparisons[f"{left}_vs_{right}"] = {
            metric: _paired_delta(
                all_cases[left],
                all_cases[right],
                metric,
                iterations=args.bootstrap,
                seed=args.seed + 104729 * pair_index + 1543 * metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
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
                    "probability_right_lower": result["probability_right_lower"],
                    "base_condition_clusters": result["clusters"],
                    "paired_cases": result["paired_cases"],
                }
            )
    _write_csv(
        args.output_dir / "single_segment_paired_comparisons.csv",
        comparison_rows,
    )

    duration_rows: list[dict[str, Any]] = []
    durations = sorted({round(float(row["duration_ns"]), 6) for row in persistence.values()})
    for label, values in all_cases.items():
        for duration in durations:
            selected = {
                key: row
                for key, row in values.items()
                if round(float(row["duration_ns"]), 6) == duration
            }
            if not selected:
                continue
            row = {"model": label, "duration_ns": duration, "cases": len(selected)}
            for metric in METRICS:
                _, array = _cluster_array(selected, metric)
                row[metric] = float(array.mean())
            duration_rows.append(row)
    _write_csv(args.output_dir / "single_segment_by_duration.csv", duration_rows)

    summary = {
        "status": "complete",
        "protocol": {
            "point_estimate": "mean of per-base-condition means",
            "draw_handling": "model draws averaged within each held-out transition before statistics",
            "uncertainty": "percentile bootstrap over base-condition clusters",
            "bootstrap_iterations": args.bootstrap,
            "seed": args.seed,
        },
        "models": estimates,
        "paired_comparisons": comparisons,
        "case_count": len(persistence),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    primary_plot_labels = [
        "persistence",
        "direct",
        "fno",
        "cartesian",
        "standard_prior",
        "stage1",
        "mixed",
    ]
    primary_plot_labels = [
        label for label in primary_plot_labels if label in estimates
    ]
    _save_metric_figure(
        args.output_dir / "single_segment_main_metrics.png",
        primary_plot_labels,
        estimates,
    )
    ablation_plot_labels = [
        "stage1",
        "gt_only",
        "mixed",
        "pred_only",
        "no_target_time",
        "active_spatial",
        "topology",
    ]
    ablation_plot_labels = [
        label for label in ablation_plot_labels if label in estimates
    ]
    _save_metric_figure(
        args.output_dir / "single_segment_causal_ablation_metrics.png",
        ablation_plot_labels,
        estimates,
    )

    markdown = [
        "# x5 single-segment main results",
        "",
        "Point estimates average within base condition; intervals are 95% base-cluster bootstraps.",
        "",
        "| Model | Angle (deg, 95% CI) | MSE | |Delta Q| (95% CI) | Q failure |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        markdown.append(
            f"| {row['model']} | {float(row['ang']):.4f} "
            f"[{float(row['ang_ci95_low']):.4f}, {float(row['ang_ci95_high']):.4f}] | "
            f"{float(row['mse']):.5f} | {float(row['q_abs']):.4f} "
            f"[{float(row['q_abs_ci95_low']):.4f}, {float(row['q_abs_ci95_high']):.4f}] | "
            f"{float(row['q_fail']):.4f} |"
        )
    (args.output_dir / "README.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
