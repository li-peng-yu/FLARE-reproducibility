#!/usr/bin/env python3
"""Aggregate paired Stage-1/Stage-2 autoregressive rollout-gap results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = ("ang", "mse", "q_abs", "energy_abs", "q_fail")
MODES = ("tf", "ar", "gap")


def _read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                row: dict[str, Any] = dict(raw)
                for key in (
                    "draw_index",
                    "record_index",
                    "segment_position",
                    "handoff_count",
                    "control_segment_index",
                ):
                    row[key] = int(row[key])
                for mode in MODES:
                    for metric in METRICS:
                        row[f"{mode}_{metric}"] = float(row[f"{mode}_{metric}"])
                rows.append(row)
    return rows


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _ci(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]


def _run_level(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["run_id"]), int(row["segment_position"]))].append(row)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for key, values in grouped.items():
        first = values[0]
        item: dict[str, Any] = {
            "run_id": first["run_id"],
            "base_id": first["base_id"],
            "segment_position": first["segment_position"],
            "handoff_count": first["handoff_count"],
            "control_segment_index": first["control_segment_index"],
            "end_time_ns": _mean([float(row["end_time_ns"]) for row in values]),
            "draws": len(values),
        }
        for mode in MODES:
            for metric in METRICS:
                item[f"{mode}_{metric}"] = _mean(
                    [float(row[f"{mode}_{metric}"]) for row in values]
                )
        out[key] = item
    return out


def _trajectory_rollout_values(
    run_rows: dict[tuple[str, int], dict[str, Any]],
    metric: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (_run_id, position), row in run_rows.items():
        if int(position) > 0:
            grouped[str(row["run_id"])].append(row)
    out: dict[str, dict[str, Any]] = {}
    for run_id, rows in grouped.items():
        out[run_id] = {
            "base_id": str(rows[0]["base_id"]),
            **{
                mode: _mean([float(row[f"{mode}_{metric}"]) for row in rows])
                for mode in MODES
            },
        }
    return out


def _cluster_means(
    values: dict[str, dict[str, Any]],
    mode: str,
    clusters: list[str],
) -> np.ndarray:
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for row in values.values():
        by_cluster[str(row["base_id"])].append(float(row[mode]))
    return np.asarray([_mean(by_cluster[key]) for key in clusters], dtype=np.float64)


def _bootstrap_metric(
    stage1: dict[str, dict[str, Any]],
    stage2: dict[str, dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if set(stage1) != set(stage2):
        missing1 = sorted(set(stage2) - set(stage1))[:5]
        missing2 = sorted(set(stage1) - set(stage2))[:5]
        raise RuntimeError(f"trajectory mismatch: missing_stage1={missing1}, missing_stage2={missing2}")
    clusters = sorted({str(row["base_id"]) for row in stage1.values()})
    if clusters != sorted({str(row["base_id"]) for row in stage2.values()}):
        raise RuntimeError("Stage1 and Stage2 base-condition clusters differ")
    arrays = {
        (label, mode): _cluster_means(values, mode, clusters)
        for label, values in (("stage1", stage1), ("stage2", stage2))
        for mode in MODES
    }
    point = {
        label: {mode: float(arrays[(label, mode)].mean()) for mode in MODES}
        for label in ("stage1", "stage2")
    }
    denominator = point["stage1"]["gap"]
    point_delta = point["stage1"]["gap"] - point["stage2"]["gap"]
    point_reduction = (
        100.0 * point_delta / denominator if abs(denominator) > 1.0e-12 else float("nan")
    )
    point_ar_reduction = (
        100.0
        * (point["stage1"]["ar"] - point["stage2"]["ar"])
        / point["stage1"]["ar"]
        if abs(point["stage1"]["ar"]) > 1.0e-12
        else float("nan")
    )

    rng = np.random.default_rng(int(seed))
    delta_samples = np.empty(iterations, dtype=np.float64)
    reduction_samples = np.empty(iterations, dtype=np.float64)
    ar_reduction_samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        gap1 = float(arrays[("stage1", "gap")][sampled].mean())
        gap2 = float(arrays[("stage2", "gap")][sampled].mean())
        ar1 = float(arrays[("stage1", "ar")][sampled].mean())
        ar2 = float(arrays[("stage2", "ar")][sampled].mean())
        delta_samples[index] = gap1 - gap2
        reduction_samples[index] = 100.0 * (gap1 - gap2) / gap1 if abs(gap1) > 1.0e-12 else np.nan
        ar_reduction_samples[index] = 100.0 * (ar1 - ar2) / ar1 if abs(ar1) > 1.0e-12 else np.nan

    return {
        "stage1": point["stage1"],
        "stage2": point["stage2"],
        "stage1_gap_minus_stage2_gap": point_delta,
        "stage1_gap_minus_stage2_gap_95ci": _ci(delta_samples),
        "autoregressive_rollout_gap_reduction_percent": point_reduction,
        "autoregressive_rollout_gap_reduction_percent_95ci": _ci(reduction_samples),
        "direct_ar_error_reduction_percent": point_ar_reduction,
        "direct_ar_error_reduction_percent_95ci": _ci(ar_reduction_samples),
        "bootstrap_probability_gap_reduction_positive": float(np.mean(delta_samples > 0.0)),
        "clusters": len(clusters),
        "trajectories": len(stage1),
    }


def _per_segment_summary(
    run_rows_by_label: dict[str, dict[tuple[str, int], dict[str, Any]]]
) -> dict[str, Any]:
    positions = sorted(
        {position for rows in run_rows_by_label.values() for (_run, position) in rows}
    )
    out: dict[str, Any] = {}
    for position in positions:
        position_summary: dict[str, Any] = {}
        for label, rows in run_rows_by_label.items():
            selected = [row for (_run, pos), row in rows.items() if pos == position]
            position_summary[label] = {
                "n": len(selected),
                "mean_end_time_ns": _mean([float(row["end_time_ns"]) for row in selected]),
                "angle_deg": {
                    mode: _mean([float(row[f"{mode}_ang"]) for row in selected])
                    for mode in MODES
                },
                "mse": {
                    mode: _mean([float(row[f"{mode}_mse"]) for row in selected])
                    for mode in MODES
                },
                "q_abs": {
                    mode: _mean([float(row[f"{mode}_q_abs"]) for row in selected])
                    for mode in MODES
                },
            }
        out[str(position)] = position_summary
    return out


def _save_figure(path: Path, per_segment: dict[str, Any]) -> None:
    positions = np.asarray(sorted(int(key) for key in per_segment), dtype=np.int64)
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.7))
    colors = {"stage1": "#555555", "stage2": "#1976B5"}
    for label in ("stage1", "stage2"):
        tf = np.asarray([per_segment[str(pos)][label]["angle_deg"]["tf"] for pos in positions])
        ar = np.asarray([per_segment[str(pos)][label]["angle_deg"]["ar"] for pos in positions])
        gap = ar - tf
        axes[0].plot(positions, ar, marker="o", linewidth=2.0, color=colors[label], label=f"{label} AR")
        axes[0].plot(positions, tf, marker="s", linewidth=1.4, linestyle="--", color=colors[label], alpha=0.7, label=f"{label} TF")
        axes[1].plot(positions, gap, marker="o", linewidth=2.0, color=colors[label], label=label)
    axes[0].set_xlabel("handoff count")
    axes[0].set_ylabel("mean angular error (degree)")
    axes[0].set_title("Teacher-forced and autoregressive errors")
    axes[1].axhline(0.0, color="#999999", linewidth=0.8)
    axes[1].set_xlabel("handoff count")
    axes[1].set_ylabel("ARG: AR minus TF (degree)")
    axes[1].set_title("Autoregressive rollout gap")
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
        axis.set_xticks(positions)
    figure.tight_layout()
    figure.savefig(path, dpi=240)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths_by_label = {
        label: [args.input_dir / f"{label}_draw{draw}.csv" for draw in range(args.draws)]
        for label in ("stage1", "stage2")
    }
    for label, paths in paths_by_label.items():
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing {label} inputs: {missing}")
    raw_by_label = {label: _read_rows(paths) for label, paths in paths_by_label.items()}
    run_rows_by_label = {label: _run_level(rows) for label, rows in raw_by_label.items()}
    if set(run_rows_by_label["stage1"]) != set(run_rows_by_label["stage2"]):
        raise RuntimeError("Stage1 and Stage2 evaluated different trajectory/segment keys")

    metrics: dict[str, Any] = {}
    for metric in METRICS:
        values1 = _trajectory_rollout_values(run_rows_by_label["stage1"], metric)
        values2 = _trajectory_rollout_values(run_rows_by_label["stage2"], metric)
        metrics[metric] = _bootstrap_metric(
            values1,
            values2,
            iterations=int(args.bootstrap),
            seed=int(args.seed) + 997 * (METRICS.index(metric) + 1),
        )
    per_segment = _per_segment_summary(run_rows_by_label)

    metadata_by_label: dict[str, list[dict[str, Any]]] = {}
    for label in ("stage1", "stage2"):
        metadata_by_label[label] = [
            json.loads((args.input_dir / f"{label}_draw{draw}.json").read_text(encoding="utf-8"))
            for draw in range(args.draws)
        ]
    summary = {
        "status": "complete",
        "primary_metric": "angular autoregressive rollout gap (ARG_ang = AR_ang - TF_ang)",
        "primary_effect": "ARGR_ang = (ARG_stage1 - ARG_stage2) / ARG_stage1",
        "scope": "held-out test trajectories with at least one control-segment handoff in 1-6 ns",
        "pairing": "same trajectories, EMA state, Heun ODE10, and sampler seeds for Stage1/Stage2 and TF/AR",
        "draws_per_trajectory": int(args.draws),
        "cluster_bootstrap": {
            "unit": "base condition parsed from run_id",
            "iterations": int(args.bootstrap),
            "seed": int(args.seed),
        },
        "metrics": metrics,
        "per_segment": per_segment,
        "inputs": metadata_by_label,
        "causal_limitation": (
            "Stage2 has 25k additional updates. This comparison demonstrates the net effect of the "
            "Stage2 procedure versus Stage1, but a matched 25k true-start-only continuation is needed "
            "to isolate predicted-handoff exposure from additional optimization."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _save_figure(args.output_dir / "angular_arg_stage1_vs_stage2.png", per_segment)

    angle = metrics["ang"]
    markdown = f"""# Stage2 fully autoregressive rollout evaluation

- Test trajectories: {angle['trajectories']} across {angle['clusters']} base-condition clusters
- Draws per trajectory: {args.draws}
- Stage1 ARG (angle): {angle['stage1']['gap']:.6f} degree
- Stage2 ARG (angle): {angle['stage2']['gap']:.6f} degree
- ARG reduction: {angle['autoregressive_rollout_gap_reduction_percent']:.3f}% (95% cluster-bootstrap CI {angle['autoregressive_rollout_gap_reduction_percent_95ci'][0]:.3f}% to {angle['autoregressive_rollout_gap_reduction_percent_95ci'][1]:.3f}%)
- Direct AR angular-error reduction: {angle['direct_ar_error_reduction_percent']:.3f}% (95% CI {angle['direct_ar_error_reduction_percent_95ci'][0]:.3f}% to {angle['direct_ar_error_reduction_percent_95ci'][1]:.3f}%)
- Bootstrap probability that Stage2 reduced ARG: {angle['bootstrap_probability_gap_reduction_positive']:.6f}

This is a Stage1-versus-Stage2 net-effect test. A matched true-start-only 25k continuation remains necessary for a causal training ablation.
"""
    (args.output_dir / "README.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
