#!/usr/bin/env python3
"""Render the Appendix Stage-2 diagnostic from a frozen rollout summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "stage1": "Stage 1",
    "mixed": "Stage 2 mixed",
    "gt_only": "GT only",
}
COLORS = {
    "stage1": "#1f77b4",
    "mixed": "#d95f02",
    "gt_only": "#1b9e77",
}


def _errorbar(estimate: dict[str, object]) -> np.ndarray:
    mean = float(estimate["mean"])
    low, high = (float(value) for value in estimate["95ci"])
    return np.asarray([[mean - low], [high - mean]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    expected = ["stage1", "mixed", "gt_only"]
    if summary.get("labels") != expected:
        raise ValueError(f"expected labels {expected}, got {summary.get('labels')}")
    if summary.get("cluster_bootstrap", {}).get("iterations") != 50_000:
        raise ValueError("rollout figure requires the frozen 50k-bootstrap summary")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
        }
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12.6, 3.05),
        gridspec_kw={"width_ratios": [1.12, 1.12, 0.96]},
    )

    # (a) Overall exact-boundary and autoregressive errors.
    positions = np.arange(len(expected), dtype=float)
    width = 0.38
    exact_color = "#56B4E9"
    rollout_color = "#D55E00"
    for mode, offset, color, label in (
        ("tf", -width / 2, exact_color, "exact boundary"),
        ("ar", width / 2, rollout_color, "autoregressive"),
    ):
        estimates = [summary["model_estimates"][key]["ang"][mode] for key in expected]
        means = [float(estimate["mean"]) for estimate in estimates]
        yerr = np.concatenate([_errorbar(estimate) for estimate in estimates], axis=1)
        axes[0].bar(
            positions + offset,
            means,
            width,
            yerr=yerr,
            color=color,
            edgecolor="none",
            error_kw={"ecolor": "black", "elinewidth": 1.1, "capsize": 2.8, "capthick": 1.1},
            label=label,
        )
    axes[0].set_xticks(positions, [LABELS[key] for key in expected], rotation=38, ha="right")
    axes[0].set_ylabel("angular error (degrees)")
    axes[0].set_ylim(0, 43)
    # Keep the legend above the axes so it cannot
    # obscure the high autoregressive bars and their confidence intervals.
    axes[0].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        columnspacing=1.2,
    )
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].set_axisbelow(True)
    axes[0].text(-0.12, 1.02, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=11)

    # (b) Overall autoregressive-minus-exact-boundary degradation.
    reverse = list(reversed(expected))
    ypos = np.arange(len(reverse), dtype=float)
    for y, key in zip(ypos, reverse, strict=True):
        estimate = summary["model_estimates"][key]["ang"]["gap"]
        axes[1].barh(
            y,
            float(estimate["mean"]),
            xerr=_errorbar(estimate),
            color=COLORS[key],
            edgecolor="none",
            error_kw={"ecolor": "black", "elinewidth": 1.1, "capsize": 2.8, "capthick": 1.1},
        )
    axes[1].set_yticks(ypos, [LABELS[key] for key in reverse])
    axes[1].set_xlabel("rollout degradation (degrees)")
    axes[1].set_xlim(0, 6.4)
    axes[1].grid(axis="x", alpha=0.22)
    axes[1].set_axisbelow(True)
    axes[1].text(-0.15, 1.02, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=11)

    # (c) Segment-resolved endpoint sequence.  Segment 1 has no generated handoff.
    segments = sorted(summary["per_segment"], key=int)
    x = np.arange(len(segments), dtype=float)
    for key in expected:
        ar = [summary["per_segment"][segment][key]["angle_deg"]["ar"] for segment in segments]
        tf = [summary["per_segment"][segment][key]["angle_deg"]["tf"] for segment in segments]
        axes[2].plot(x, ar, color=COLORS[key], marker="o", linewidth=1.7, label=LABELS[key])
        axes[2].plot(x, tf, color=COLORS[key], marker="x", linestyle="--", linewidth=1.45, alpha=0.70)
    axes[2].set_xticks(x, [f"segment {int(segment) + 1}" for segment in segments])
    axes[2].set_ylabel("angular error (degrees)")
    axes[2].set_ylim(28.8, 39.0)
    axes[2].grid(axis="y", alpha=0.22)
    axes[2].set_axisbelow(True)
    axes[2].legend(frameon=False, loc="upper left")
    axes[2].text(-0.15, 1.02, "c", transform=axes[2].transAxes, fontweight="bold", fontsize=11)

    figure.tight_layout(w_pad=2.4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
