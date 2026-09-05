#!/usr/bin/env python3
"""Render paper diagnostics from the exact-control rollout summary."""

# These figures combine physical
# two-segment quality with the separate representative 5-ns speed workload.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
METHODS = (
    ("flare", "FLARE", "#0072B2"),
    ("cartesian_cfm", "Cartesian CFM", "#F0E442"),
    ("direct_unet", "Direct U-Net", "#56B4E9"),
    ("poseidon_t", "Poseidon-T", "#E69F00"),
    ("cno_fm", "CNO-FM", "#009E73"),
    ("dpot_ti", "DPOT-Ti", "#D55E00"),
    ("mpp_avit_ti", "MPP-AViT-Ti", "#7B61A8"),
    ("pdearena_unet", "PDEArena", "#8C564B"),
    ("le_pde", "LE-PDE", "#CC79A7"),
    ("neuralmag_x5", "NeuralMAG", "#4D4D4D"),
)
COMPUTE_METHODS = tuple(
    item
    for item in METHODS
    if item[0] not in {"cartesian_cfm", "direct_unet"}
)

# Offsets keep the newly clustered
# complete-path angular metrics legible without changing any plotted values.
ANGLE_LABEL_POSITIONS = {
    "flare": (35.0, 12.0),
    "cartesian_cfm": (41.0, 17.5),
    "direct_unet": (42.5, 21.0),
    "poseidon_t": (31.2, 27.5),
    "cno_fm": (31.2, 20.0),
    "dpot_ti": (41.0, 30.0),
    "pdearena_unet": (38.0, 24.0),
}
COMPUTE_SCORE_LABEL_OFFSETS = {
    "cno_fm": (-7, 7, "right"),
    "dpot_ti": (6, 8, "left"),
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(f"incomplete summary: {path}")
    if payload.get("reference_mode") != "exact_control_multisegment_rollout_endpoint":
        raise RuntimeError(f"wrong quality protocol: {path}")
    if int(payload.get("conditions", -1)) != 33:
        raise RuntimeError(f"expected 33 complete paths: {path}")
    rows = {str(row["method"]): row for row in payload.get("rows", [])}
    for method, _, _ in METHODS:
        row = rows.get(method)
        if row is None:
            raise RuntimeError(f"missing {method} row: {path}")
        metrics = [float(row[key]) for key in
                   ("distribution_score", "paired_angle_deg", "angular_energy_distance_deg")]
        if not np.isfinite(metrics).all():
            raise RuntimeError(f"non-finite quality metrics for {method}: {path}")
    for method, _, _ in COMPUTE_METHODS:
        latency = float(rows[method]["fastest_batch_ms_per_output"])
        parameters = int(rows[method]["parameter_count"])
        if not np.isfinite(latency) or latency <= 0 or parameters <= 0:
            raise RuntimeError(f"invalid timing or parameter count for {method}: {path}")
    return payload


def _panel(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.0,
        1.025,
        label,
        transform=axis.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="bottom",
    )


def _save(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def render_distribution(payload: dict[str, Any], output: Path) -> None:
    rows = {str(row["method"]): row for row in payload["rows"]}
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12.8, 3.25),
        gridspec_kw={"width_ratios": [1.30, 1.08, 0.92]},
    )

    x = np.arange(len(METHODS), dtype=float)
    values = np.asarray([float(rows[key]["distribution_score"]) for key, _, _ in METHODS])
    lows = np.asarray([float(rows[key]["distribution_score_ci"][0]) for key, _, _ in METHODS])
    highs = np.asarray([float(rows[key]["distribution_score_ci"][1]) for key, _, _ in METHODS])
    colors = [color for _, _, color in METHODS]
    labels = [label for _, label, _ in METHODS]
    axes[0].bar(x, values, width=0.66, color=colors)
    axes[0].errorbar(
        x,
        values,
        yerr=np.vstack((values - lows, highs - values)),
        fmt="none",
        ecolor="#303030",
        linewidth=0.8,
        capsize=2.0,
    )
    axes[0].set_xticks(x, labels, rotation=42, ha="right")
    axes[0].set_ylim(0.0, min(1.0, max(highs) * 1.12))
    axes[0].set_ylabel("complete-path Distribution Score ↑")
    axes[0].grid(axis="y", alpha=0.18)
    axes[0].set_axisbelow(True)
    _panel(axes[0], "a")

    for key, label, color in METHODS:
        row = rows[key]
        marker = "*" if key == "flare" else ("s" if key == "direct_unet" else "o")
        size = 80 if key == "flare" else (38 if key == "direct_unet" else 30)
        axes[1].scatter(
            float(row["paired_angle_deg"]),
            float(row["angular_energy_distance_deg"]),
            marker=marker,
            s=size,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        point = (
            float(row["paired_angle_deg"]),
            float(row["angular_energy_distance_deg"]),
        )
        if key in ANGLE_LABEL_POSITIONS:
            axes[1].annotate(
                label,
                point,
                xytext=ANGLE_LABEL_POSITIONS[key],
                textcoords="data",
                fontsize=7.0,
                ha="left",
                va="center",
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.45},
            )
        else:
            axes[1].annotate(
                label,
                point,
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.0,
            )
    axes[1].set_xlabel("paired angular error (degrees) ↓")
    axes[1].set_ylabel("angular energy distance (degrees) ↓")
    axes[1].set_xlim(30.0, 80.0)
    axes[1].set_ylim(10.0, 82.0)
    axes[1].grid(alpha=0.18)
    axes[1].set_axisbelow(True)
    _panel(axes[1], "b")

    comparisons = payload["flare_vs_all_baselines_distribution_score"]
    baseline_methods = [item for item in METHODS if item[0] != "flare"]
    benefit = np.asarray(
        [float(comparisons[key]["relative_benefit"]) for key, _, _ in baseline_methods]
    )
    ratio = 1.0 + benefit
    low = 1.0 + np.asarray(
        [
            float(comparisons[key]["relative_benefit_bootstrap_95ci"][0])
            for key, _, _ in baseline_methods
        ]
    )
    high = 1.0 + np.asarray(
        [
            float(comparisons[key]["relative_benefit_bootstrap_95ci"][1])
            for key, _, _ in baseline_methods
        ]
    )
    order = np.argsort(ratio)
    y = np.arange(len(order), dtype=float)
    axes[2].errorbar(
        ratio[order],
        y,
        xerr=np.vstack((ratio[order] - low[order], high[order] - ratio[order])),
        fmt="o",
        color="#0072B2",
        ecolor="#0072B2",
        linewidth=1.0,
        capsize=2.0,
    )
    axes[2].axvline(1.0, color="#555555", linestyle="--", linewidth=0.9)
    axes[2].set_xscale("log")
    axes[2].set_yticks(
        y,
        [baseline_methods[int(index)][1] for index in order],
    )
    axes[2].set_xlabel("paired score ratio\nFLARE / baseline")
    axes[2].grid(axis="x", alpha=0.18)
    axes[2].set_axisbelow(True)
    _panel(axes[2], "c")

    figure.subplots_adjust(left=0.055, right=0.992, bottom=0.245, top=0.90, wspace=0.36)
    _save(figure, output)


def render_compute(payload: dict[str, Any], output: Path) -> None:
    rows = {str(row["method"]): row for row in payload["rows"]}
    figure, axes = plt.subplots(1, 2, figsize=(9.3, 3.45))
    for key, label, color in COMPUTE_METHODS:
        row = rows[key]
        latency = float(row["fastest_batch_ms_per_output"])
        score = float(row["distribution_score"])
        parameters = float(row["parameter_count"])
        size = 28.0 + 30.0 * np.sqrt(parameters / 1.0e8)
        marker = "*" if key == "flare" else "o"
        axes[0].scatter(latency, score, s=size, marker=marker, color=color, edgecolor="white")
        dx, dy, alignment = COMPUTE_SCORE_LABEL_OFFSETS.get(
            key, (4, 4, "left")
        )
        axes[0].annotate(
            label,
            (latency, score),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.0,
            ha=alignment,
        )
        axes[1].scatter(parameters / 1.0e6, latency, s=size, marker=marker, color=color, edgecolor="white")
        axes[1].annotate(label, (parameters / 1.0e6, latency), xytext=(4, 4), textcoords="offset points", fontsize=7.0)

    axes[0].set_xscale("log")
    axes[0].set_xlabel("fixed 2+3-ns latency at fastest paper batch (ms/output)")
    axes[0].set_ylabel("complete-path Distribution Score ↑")
    axes[0].grid(alpha=0.18)
    axes[0].set_axisbelow(True)
    _panel(axes[0], "a")

    axes[1].set_yscale("log")
    axes[1].set_xlabel("parameters (millions)")
    axes[1].set_ylabel("fixed 2+3-ns latency (ms/output)")
    axes[1].grid(alpha=0.18)
    axes[1].set_axisbelow(True)
    _panel(axes[1], "b")

    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.90, wspace=0.31)
    _save(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT / "reports/x5_exact_control_rollout_seed78_20260903.json",
    )
    parser.add_argument(
        "--distribution-output",
        type=Path,
        default=PROJECT / "paper/fig/appendix_distribution_diagnostics.pdf",
    )
    parser.add_argument(
        "--compute-output",
        type=Path,
        default=PROJECT / "paper/fig/appendix_compute_tradeoffs.pdf",
    )
    args = parser.parse_args()
    payload = _load(args.summary.resolve())
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    render_distribution(payload, args.distribution_output.resolve())
    render_compute(payload, args.compute_output.resolve())


if __name__ == "__main__":
    main()
