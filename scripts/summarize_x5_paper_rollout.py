#!/usr/bin/env python3
"""Aggregate the complete x5 paper TF/fully-AR causal ablation."""

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

from compare_skx_x5_stage2_rollout_arg import (
    METRICS,
    _bootstrap_metric,
    _per_segment_summary,
    _read_rows,
    _run_level,
    _trajectory_rollout_values,
)


DEFAULT_LABELS = (
    "stage1",
    "mixed",
    "cartesian",
    "standard_prior",
    "no_target_time",
    "direct",
    "fno",
    "gt_only",
    "pred_only",
    "topology",
)


def _draw_paths(input_dir: Path, label: str) -> list[Path]:
    def draw_index(path: Path) -> int:
        return int(path.stem.rsplit("draw", 1)[1])

    paths = sorted(input_dir.glob(f"{label}_draw*.csv"), key=draw_index)
    if not paths:
        raise FileNotFoundError(f"no rollout draws for {label} in {input_dir}")
    expected = list(range(len(paths)))
    actual = [draw_index(path) for path in paths]
    if actual != expected:
        raise RuntimeError(
            f"non-contiguous rollout draw ids for {label}: {actual}, expected {expected}"
        )
    return paths


def _model_estimate(
    values: dict[str, dict[str, Any]],
    mode: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in values.values():
        grouped[str(row["base_id"])].append(float(row[mode]))
    clusters = sorted(grouped)
    if not clusters:
        raise RuntimeError("cannot summarize an empty rollout collection")
    array = np.asarray(
        [np.mean(grouped[cluster], dtype=np.float64) for cluster in clusters],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        samples[index] = float(array[selected].mean())
    return {
        "mean": float(array.mean()),
        "95ci": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "base_condition_clusters": len(clusters),
        "trajectories": len(values),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_handoff_figure(path: Path, per_segment: dict[str, Any], labels: list[str]) -> None:
    positions = sorted(int(position) for position in per_segment)
    if not positions:
        raise RuntimeError("no handoff positions available for rollout figure")
    colors = plt.get_cmap("tab10")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for label_index, label in enumerate(labels):
        color = colors(label_index)
        tf = [
            per_segment[str(position)][label]["angle_deg"]["tf"]
            for position in positions
        ]
        ar = [
            per_segment[str(position)][label]["angle_deg"]["ar"]
            for position in positions
        ]
        axes[0].plot(positions, ar, marker="o", color=color, label=f"{label} AR")
        axes[0].plot(
            positions,
            tf,
            marker="s",
            linestyle="--",
            alpha=0.68,
            color=color,
            label=f"{label} TF",
        )
        axes[1].plot(
            positions,
            np.asarray(ar) - np.asarray(tf),
            marker="o",
            color=color,
            label=label,
        )
    axes[0].set_ylabel("angular error (degree)")
    axes[1].set_ylabel("ARG: AR minus TF (degree)")
    axes[1].axhline(0.0, color="#888888", linewidth=0.8)
    for axis in axes:
        axis.set_xlabel("handoff count")
        axis.set_xticks(positions)
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=240)
    plt.close(figure)


def _pair(
    left: str,
    right: str,
    values: dict[str, dict[str, dict[str, Any]]],
    metric: str,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    raw = _bootstrap_metric(
        values[left][metric],
        values[right][metric],
        iterations=bootstrap,
        seed=seed,
    )
    return {
        "left": left,
        "right": right,
        "models": {left: raw["stage1"], right: raw["stage2"]},
        "left_gap_minus_right_gap": raw["stage1_gap_minus_stage2_gap"],
        "left_gap_minus_right_gap_95ci": raw["stage1_gap_minus_stage2_gap_95ci"],
        "right_arg_reduction_percent": raw[
            "autoregressive_rollout_gap_reduction_percent"
        ],
        "right_arg_reduction_percent_95ci": raw[
            "autoregressive_rollout_gap_reduction_percent_95ci"
        ],
        "right_ar_reduction_percent": raw["direct_ar_error_reduction_percent"],
        "right_ar_reduction_percent_95ci": raw[
            "direct_ar_error_reduction_percent_95ci"
        ],
        "probability_right_arg_smaller": raw[
            "bootstrap_probability_gap_reduction_positive"
        ],
        "clusters": raw["clusters"],
        "trajectories": raw["trajectories"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument(
        "--comparison-pairs",
        nargs="*",
        default=None,
        metavar="LEFT:RIGHT",
        help=(
            "Explicit paired comparisons. The default retains the paper's "
            "original Stage-2 comparison set."
        ),
    )
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = {label: _draw_paths(args.input_dir, label) for label in args.labels}
    wrong_draw_counts = {
        label: len(label_paths)
        for label, label_paths in paths.items()
        if len(label_paths) != args.draws
    }
    if wrong_draw_counts:
        raise RuntimeError(
            f"expected exactly {args.draws} draws per label, got {wrong_draw_counts}"
        )
    run_rows = {
        label: _run_level(_read_rows(label_paths))
        for label, label_paths in paths.items()
    }
    keysets = [set(rows) for rows in run_rows.values()]
    if not all(keys == keysets[0] for keys in keysets[1:]):
        raise RuntimeError("models were not evaluated on identical trajectory/segment keys")

    values = {
        label: {
            metric: _trajectory_rollout_values(rows, metric)
            for metric in METRICS
        }
        for label, rows in run_rows.items()
    }
    model_estimates: dict[str, Any] = {}
    model_rows: list[dict[str, Any]] = []
    for label_index, label in enumerate(args.labels):
        model_estimates[label] = {}
        row: dict[str, Any] = {"model": label}
        for metric_index, metric in enumerate(METRICS):
            model_estimates[label][metric] = {}
            for mode_index, mode in enumerate(("tf", "ar", "gap")):
                estimate = _model_estimate(
                    values[label][metric],
                    mode,
                    iterations=args.bootstrap,
                    seed=(
                        args.seed
                        + 100_003 * label_index
                        + 7_919 * metric_index
                        + 1_009 * mode_index
                    ),
                )
                model_estimates[label][metric][mode] = estimate
                if metric == "ang":
                    row[f"{mode}_angle_deg"] = estimate["mean"]
                    row[f"{mode}_angle_ci95_low"] = estimate["95ci"][0]
                    row[f"{mode}_angle_ci95_high"] = estimate["95ci"][1]
                    row["base_condition_clusters"] = estimate[
                        "base_condition_clusters"
                    ]
                    row["trajectories"] = estimate["trajectories"]
        model_rows.append(row)
    _write_csv(args.output_dir / "rollout_angle_table.csv", model_rows)
    if args.comparison_pairs is None:
        pairs = [
            ("stage1", "gt_only"),
            ("stage1", "mixed"),
            ("gt_only", "mixed"),
            ("gt_only", "pred_only"),
            ("mixed", "pred_only"),
            ("mixed", "topology"),
        ]
    else:
        pairs = []
        for raw_pair in args.comparison_pairs:
            parts = raw_pair.split(":")
            if len(parts) != 2 or not all(parts):
                raise ValueError(
                    f"invalid comparison pair {raw_pair!r}; expected LEFT:RIGHT"
                )
            pairs.append((parts[0], parts[1]))
    pairs = [pair for pair in pairs if pair[0] in values and pair[1] in values]
    if args.comparison_pairs is not None and len(pairs) != len(args.comparison_pairs):
        raise ValueError(
            "every explicit comparison label must also be present in --labels"
        )
    comparisons: dict[str, Any] = {}
    for metric_index, metric in enumerate(METRICS):
        comparisons[metric] = {
            f"{left}_vs_{right}": _pair(
                left,
                right,
                values,
                metric,
                args.bootstrap,
                args.seed + 1009 * metric_index + 37 * pair_index,
            )
            for pair_index, (left, right) in enumerate(pairs)
        }
    comparison_rows: list[dict[str, Any]] = []
    for metric, metric_comparisons in comparisons.items():
        for comparison_name, result in metric_comparisons.items():
            comparison_rows.append(
                {
                    "comparison": comparison_name,
                    "left": result["left"],
                    "right": result["right"],
                    "metric": metric,
                    "left_arg_minus_right_arg": result[
                        "left_gap_minus_right_gap"
                    ],
                    "arg_delta_ci95_low": result[
                        "left_gap_minus_right_gap_95ci"
                    ][0],
                    "arg_delta_ci95_high": result[
                        "left_gap_minus_right_gap_95ci"
                    ][1],
                    "right_arg_reduction_percent": result[
                        "right_arg_reduction_percent"
                    ],
                    "arg_reduction_ci95_low": result[
                        "right_arg_reduction_percent_95ci"
                    ][0],
                    "arg_reduction_ci95_high": result[
                        "right_arg_reduction_percent_95ci"
                    ][1],
                    "right_ar_reduction_percent": result[
                        "right_ar_reduction_percent"
                    ],
                    "ar_reduction_ci95_low": result[
                        "right_ar_reduction_percent_95ci"
                    ][0],
                    "ar_reduction_ci95_high": result[
                        "right_ar_reduction_percent_95ci"
                    ][1],
                    "probability_right_arg_smaller": result[
                        "probability_right_arg_smaller"
                    ],
                    "base_condition_clusters": result["clusters"],
                    "trajectories": result["trajectories"],
                }
            )
    _write_csv(args.output_dir / "rollout_paired_comparisons.csv", comparison_rows)
    per_segment = _per_segment_summary(run_rows)
    summary = {
        "status": "complete",
        "definition": {
            "TF": "each segment starts from its held-out MuMax boundary",
            "fully_AR": "only the first segment uses a MuMax start; later segments consume the previous model prediction",
            "ARG": "fully_AR error minus TF error",
        },
        "labels": list(args.labels),
        "draws_per_trajectory": {
            label: len(label_paths) for label, label_paths in paths.items()
        },
        "cluster_bootstrap": {
            "unit": "base condition",
            "iterations": args.bootstrap,
            "seed": args.seed,
        },
        "model_estimates": model_estimates,
        "comparisons": comparisons,
        "per_segment": per_segment,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _save_handoff_figure(
        args.output_dir / "rollout_angle_by_handoff.png",
        per_segment,
        list(args.labels),
    )
    angle = comparisons["ang"]
    rows = []
    for label in args.labels:
        estimate = model_estimates[label]["ang"]
        rows.append(
            f"| {label} | {estimate['tf']['mean']:.4f} "
            f"[{estimate['tf']['95ci'][0]:.4f}, {estimate['tf']['95ci'][1]:.4f}] | "
            f"{estimate['ar']['mean']:.4f} "
            f"[{estimate['ar']['95ci'][0]:.4f}, {estimate['ar']['95ci'][1]:.4f}] | "
            f"{estimate['gap']['mean']:.4f} "
            f"[{estimate['gap']['95ci'][0]:.4f}, {estimate['gap']['95ci'][1]:.4f}] |"
        )
    primary_pair = pairs[0] if pairs else None
    primary_name = (
        f"{primary_pair[0]}_vs_{primary_pair[1]}"
        if primary_pair is not None
        else None
    )
    causal = angle.get(primary_name) if primary_name is not None else None
    causal_text = ""
    if causal is not None:
        causal_text = (
            f"\nThe compute-matched causal comparison "
            f"({causal['left']} vs {causal['right']}) gives "
            f"{causal['right_ar_reduction_percent']:.3f}% fully-AR angular-error "
            f"reduction (95% CI {causal['right_ar_reduction_percent_95ci'][0]:.3f}% "
            f"to {causal['right_ar_reduction_percent_95ci'][1]:.3f}%) and "
            f"{causal['right_arg_reduction_percent']:.3f}% ARG reduction "
            f"(95% CI {causal['right_arg_reduction_percent_95ci'][0]:.3f}% "
            f"to {causal['right_arg_reduction_percent_95ci'][1]:.3f}%)."
        )
    markdown = """# x5 Stage-2 causal rollout ablation

| Model | TF angle (95% CI) | Fully-AR angle (95% CI) | ARG (95% CI) |
|---|---:|---:|---:|
""" + "\n".join(rows) + causal_text + "\n"
    (args.output_dir / "README.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
