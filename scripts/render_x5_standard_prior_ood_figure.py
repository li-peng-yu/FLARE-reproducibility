#!/usr/bin/env python3
"""Render the Appendix ring-OOD diagnostic from its frozen summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _estimate(payload: dict[str, object], key: str) -> tuple[float, np.ndarray]:
    estimate = payload[key]
    mean = float(estimate["mean"])
    low, high = (float(value) for value in estimate["95ci"])
    return mean, np.asarray([[mean - low], [high - mean]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError("OOD figure requires a complete frozen summary")

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
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 3.05))
    id_color, ood_color = "#56B4E9", "#D55E00"

    # (a) MagFlow single-segment ID/OOD comparison.
    single = summary["single_segment"]["scfm"]
    single_means, single_errors = [], []
    for stratum in ("id_unseen_base", "ood_geometry"):
        mean, error = _estimate(single[stratum], "ang")
        single_means.append(mean)
        single_errors.append(error)
    x = np.arange(2)
    axes[0].bar(
        x,
        single_means,
        yerr=np.concatenate(single_errors, axis=1),
        color=[id_color, ood_color],
        edgecolor="none",
        error_kw={"ecolor": "black", "elinewidth": 1.1, "capsize": 3.0},
    )
    axes[0].set_xticks(x, ["ID, unseen base", "OOD ring"], rotation=10)
    axes[0].set_ylabel("single-segment angle (degrees)")
    axes[0].set_ylim(0, 35)
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].set_axisbelow(True)
    axes[0].text(-0.13, 1.02, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=11)

    # (b) Matched FNO control under teacher-forced and autoregressive evaluation.
    fno = summary["fully_autoregressive"]["fno"]
    modes = ("tf_ang", "ar_ang", "final_ar_ang")
    labels = ("teacher-forced", "autoregressive", "final rollout step")
    positions = np.arange(len(modes), dtype=float)
    for offset, stratum, color, label in (
        (-0.08, "id_unseen_base", id_color, "ID, unseen base"),
        (0.08, "ood_geometry", ood_color, "OOD ring"),
    ):
        means, errors = [], []
        for mode in modes:
            mean, error = _estimate(fno[stratum], mode)
            means.append(mean)
            errors.append(error)
        axes[1].errorbar(
            positions + offset,
            means,
            yerr=np.concatenate(errors, axis=1),
            color=color,
            marker="o",
            linewidth=1.4,
            capsize=3.0,
            label=label,
        )
    axes[1].set_xticks(positions, labels, rotation=24, ha="right")
    axes[1].set_ylabel("multi-stage angle (degrees)")
    axes[1].set_ylim(24, 45)
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].text(-0.13, 1.02, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=11)

    # (c) Relative OOD changes for MagFlow.
    scfm_rollout = summary["fully_autoregressive"]["scfm"]
    relative = [
        100.0
        * (
            float(single["ood_geometry"]["ang"]["mean"])
            - float(single["id_unseen_base"]["ang"]["mean"])
        )
        / float(single["id_unseen_base"]["ang"]["mean"])
    ]
    for mode in modes:
        id_mean = float(scfm_rollout["id_unseen_base"][mode]["mean"])
        ood_mean = float(scfm_rollout["ood_geometry"][mode]["mean"])
        relative.append(100.0 * (ood_mean - id_mean) / id_mean)
    relative_labels = ("single segment", "teacher-forced", "autoregressive", "final rollout step")
    axes[2].bar(np.arange(4), relative, color="#2878B5", edgecolor="none")
    axes[2].set_xticks(np.arange(4), relative_labels, rotation=26, ha="right")
    axes[2].set_ylabel("OOD relative change (%)")
    axes[2].set_ylim(0, 4.2)
    axes[2].grid(axis="y", alpha=0.22)
    axes[2].set_axisbelow(True)
    axes[2].text(-0.13, 1.02, "c", transform=axes[2].transAxes, fontweight="bold", fontsize=11)

    figure.tight_layout(w_pad=2.4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
