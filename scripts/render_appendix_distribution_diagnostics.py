#!/usr/bin/env python3
"""Render the appendix distribution diagnostic from frozen evaluation summaries."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    color: str
    summary: Path


METHODS = (
    Method(
        "scfm_stage1",
        "FLARE",
        "#0072B2",
        PROJECT
        / "graph/results/x5_distribution_pareto_20260822/"
        "formal_standard_prior_n5_v020/scfm_stage1/summary/run_summary.json",
    ),
    Method(
        "poseidon_t",
        "Poseidon-T",
        "#E69F00",
        PROJECT
        / "graph/results/x5_distribution_pareto_20260816/"
        "formal_33groups_50k_v020/poseidon_t/summary/run_summary.json",
    ),
    Method(
        "cno_fm",
        "CNO-FM",
        "#009E73",
        PROJECT
        / "graph/results/x5_distribution_pareto_20260816/"
        "formal_33groups_50k_v020/cno_fm/summary/run_summary.json",
    ),
    Method(
        "dpot_ti",
        "DPOT-Ti",
        "#D55E00",
        PROJECT
        / "graph/results/x5_distribution_pareto_20260816/"
        "formal_33groups_50k_v020/dpot_ti/summary/run_summary.json",
    ),
    Method(
        "mpp_avit_ti",
        "MPP-AViT-Ti",
        "#7B61A8",
        PROJECT
        / "graph/results/x5_distribution_pareto_20260816/"
        "formal_33groups_50k_v020/mpp_avit_ti/summary/run_summary.json",
    ),
    Method(
        "pdearena_unet",
        "PDEArena",
        "#8C564B",
        PROJECT
        / "graph/results/x5_distribution_pareto_20260816/"
        "formal_33groups_50k_v020/pdearena_unet/summary/run_summary.json",
    ),
    Method(
        "le_pde",
        "LE-PDE",
        "#CC79A7",
        PROJECT
        / "graph/results/x5_distribution_pareto_20260816/"
        "formal_33groups_50k_v020/le_pde/summary/run_summary.json",
    ),
    Method(
        "neuralmag_x5",
        "NeuralMAG",
        "#4D4D4D",
        PROJECT
        / "graph/results/x5_distribution_pareto_20260816/neuralmag_x5/"
        "formal_33groups_50k_v020/summary/run_summary.json",
    ),
)

MATCHED_COMPARISONS = (
    PROJECT
    / "graph/results/x5_distribution_pareto_20260822/"
    "formal_standard_prior_n5_v020/matched_50k_summary.json"
)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _validate(summaries: dict[str, dict[str, Any]]) -> None:
    """Guard against silently rendering a stale or incompatible experiment."""
    expected = {
        "scfm_stage1": (0.8050488022, 31.53563799, 12.98545421),
        "poseidon_t": (0.7091896043, 25.89772484, 14.36270832),
        "cno_fm": (0.6959372278, 26.41837627, 14.35148795),
        "dpot_ti": (0.7288021392, 29.70498840, 17.05451467),
        "mpp_avit_ti": (0.5124499706, 34.64257232, 24.78954189),
        "pdearena_unet": (0.7361567309, 27.05793056, 15.16213848),
        "le_pde": (0.0462503061, 52.66849521, 46.34877362),
        "neuralmag_x5": (0.5457512288, 41.27409379, 23.90384229),
    }
    for method in METHODS:
        payload = summaries[method.key]
        actual = (
            float(payload["score"]["combined"]["geometric_mean_score"]),
            float(payload["paired_angular_error"]["combined"]["mean"]),
            float(payload["angular_energy_distance"]["combined"]["mean"]),
        )
        if not np.allclose(actual, expected[method.key], rtol=0.0, atol=5e-8):
            raise ValueError(
                f"frozen summary mismatch for {method.key}: {actual} != "
                f"{expected[method.key]}"
            )


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.0,
        1.025,
        label,
        transform=axis.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="bottom",
    )


def _distribution_bars(
    axis: plt.Axes, summaries: dict[str, dict[str, Any]]
) -> None:
    x = np.arange(len(METHODS), dtype=float)
    width = 0.245
    series = (
        ("1", -width, "#56B4E9", "drive"),
        ("2", 0.0, "#D55E00", "post-relax"),
        ("combined", width, "#0072B2", "combined"),
    )
    for key, offset, color, label in series:
        values = []
        for method in METHODS:
            score = summaries[method.key]["score"]
            source = score["combined"] if key == "combined" else score["by_segment"][key]
            values.append(float(source["geometric_mean_score"]))
        axis.bar(x + offset, values, width=width, color=color, label=label)
    axis.set_xticks(x, [method.label for method in METHODS], rotation=42, ha="right")
    axis.set_ylim(0.0, 0.92)
    axis.set_ylabel("same-condition Distribution Score ↑")
    axis.grid(axis="y", alpha=0.18)
    axis.set_axisbelow(True)
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.56, 1.02),
        ncol=3,
        fontsize=7.2,
        handlelength=1.5,
        columnspacing=1.0,
    )
    _panel_label(axis, "a")


def _accuracy_scatter(
    axis: plt.Axes, summaries: dict[str, dict[str, Any]]
) -> None:
    # Hand-tuned offsets fan out the crowded
    # lower-left cluster.  The previous exported PDF placed six labels on top
    # of one another and obscured both points and method identities.
    label_positions: dict[str, tuple[float, float, str]] = {
        "scfm_stage1": (32.4, 12.35, "left"),
        "poseidon_t": (24.1, 12.35, "left"),
        "cno_fm": (24.1, 17.0, "left"),
        "dpot_ti": (30.8, 18.9, "left"),
        "mpp_avit_ti": (35.7, 25.9, "left"),
        "pdearena_unet": (27.9, 15.45, "left"),
        "le_pde": (51.6, 48.2, "right"),
        "neuralmag_x5": (42.3, 22.8, "left"),
    }
    for method in METHODS:
        payload = summaries[method.key]
        x = float(payload["paired_angular_error"]["combined"]["mean"])
        y = float(payload["angular_energy_distance"]["combined"]["mean"])
        marker = "*" if method.key == "scfm_stage1" else "o"
        size = 78 if method.key == "scfm_stage1" else 30
        axis.scatter(
            x,
            y,
            marker=marker,
            s=size,
            color=method.color,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        label_x, label_y, align = label_positions[method.key]
        axis.annotate(
            method.label,
            (x, y),
            xytext=(label_x, label_y),
            textcoords="data",
            ha=align,
            va="center",
            fontsize=7.25,
            color="#292929",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15},
            annotation_clip=False,
            arrowprops={
                "arrowstyle": "-",
                "color": "#777777",
                "linewidth": 0.45,
                "shrinkA": 1.5,
                "shrinkB": 2.5,
            },
        )
    axis.set_xlim(23.7, 56.1)
    axis.set_ylim(11.4, 49.0)
    axis.set_xlabel("paired angular error (degrees) ↓")
    axis.set_ylabel("angular energy distance (degrees) ↓")
    axis.grid(alpha=0.18)
    axis.set_axisbelow(True)
    _panel_label(axis, "b")


def _paired_ratios(axis: plt.Axes, comparisons: dict[str, Any]) -> None:
    rows = sorted(
        comparisons["paired_comparisons"],
        key=lambda row: float(row["score_ratio"]),
    )
    label_map = {
        "PDEArena U-Net": "PDEArena",
        "DPOT-Ti": "DPOT-Ti",
        "Poseidon-T": "Poseidon-T",
        "CNO-FM": "CNO-FM",
        "MPP-AViT-Ti": "MPP-AViT-Ti",
        "LE-PDE": "LE-PDE",
    }
    y = np.arange(len(rows), dtype=float)
    points = np.asarray([float(row["score_ratio"]) for row in rows])
    lows = np.asarray([float(row["score_ratio_bootstrap_95ci"][0]) for row in rows])
    highs = np.asarray([float(row["score_ratio_bootstrap_95ci"][1]) for row in rows])
    axis.errorbar(
        points,
        y,
        xerr=np.vstack((points - lows, highs - points)),
        fmt="o",
        color="#0072B2",
        ecolor="#0072B2",
        markersize=5.0,
        linewidth=1.15,
        capsize=2.0,
    )
    axis.axvline(1.0, color="#555555", linestyle="--", linewidth=0.9)
    axis.set_xscale("log")
    axis.set_xlim(0.82, 115.0)
    axis.set_yticks(y, [label_map[str(row["right"])] for row in rows])
    axis.set_xlabel("paired score ratio\nFLARE / baseline")
    axis.grid(axis="x", alpha=0.18)
    axis.set_axisbelow(True)
    _panel_label(axis, "c")


def render(output: Path) -> None:
    summaries = {method.key: _load(method.summary) for method in METHODS}
    _validate(summaries)
    comparisons = _load(MATCHED_COMPARISONS)
    if len(comparisons.get("paired_comparisons", [])) != 6:
        raise ValueError("expected six frozen paired baseline comparisons")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
        }
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12.8, 3.25),
        gridspec_kw={"width_ratios": [1.34, 1.05, 0.91]},
    )
    _distribution_bars(axes[0], summaries)
    _accuracy_scatter(axes[1], summaries)
    _paired_ratios(axes[2], comparisons)
    figure.subplots_adjust(left=0.055, right=0.992, bottom=0.245, top=0.90, wspace=0.34)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "paper/fig/appendix_distribution_diagnostics.pdf",
    )
    args = parser.parse_args()
    render(args.output.resolve())


if __name__ == "__main__":
    main()
