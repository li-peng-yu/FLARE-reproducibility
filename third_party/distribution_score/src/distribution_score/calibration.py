"""Aggregate cross-condition probability ranks into group calibration scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .distance import FORMAL_BLOCKS, FORMAL_SHIFT_RADIUS_PX
from .manifest import load_manifest, result_directory
from .statistics import bootstrap_macro_score, calibration_summary
from .version import PACKAGE_VERSION, algorithm_version, distance_identifier


COLORS = (
    "#226F8A",
    "#D1495B",
    "#2A9D6F",
    "#E09F3E",
    "#7656A3",
    "#6B705C",
    "#3D5A80",
    "#B56576",
)


def _save_plots(
    output_dir: Path,
    summaries: dict[str, dict[str, object]],
    groups: dict[str, np.ndarray],
    group_order: tuple[str, ...],
    labels: dict[str, str],
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 6.0))
    axis.plot([0, 1], [0, 1], color="#30343B", linestyle="--", linewidth=1.5, label="Ideal")
    for index, name in enumerate(group_order):
        summary = summaries[name]
        axis.plot(
            summary["alphas"],
            summary["coverage"],
            marker="o",
            linewidth=2,
            color=COLORS[index % len(COLORS)],
            label=f"{labels[name]} (n={summary['n_conditions']})",
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel(r"Nominal probability $\alpha$")
    axis.set_ylabel(r"Observed coverage $\hat{C}_g(\alpha)$")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(output_dir / "calibration_curves_by_group.png", dpi=240)
    plt.close(figure)

    columns = min(3, len(group_order))
    rows = int(np.ceil(len(group_order) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.2 * rows), squeeze=False)
    bins = np.linspace(0.0, 1.0, 11)
    for index, axis in enumerate(axes.flat):
        if index >= len(group_order):
            axis.axis("off")
            continue
        name = group_order[index]
        axis.hist(groups[name], bins=bins, color=COLORS[index % len(COLORS)], edgecolor="white")
        axis.axhline(len(groups[name]) / 10.0, color="#30343B", linestyle="--", linewidth=1.2)
        axis.set_title(labels[name], fontsize=10)
        axis.set_xlabel(r"Probability rank $u$")
        axis.set_ylabel("Conditions")
    figure.tight_layout()
    figure.savefig(output_dir / "probability_rank_histograms_by_group.png", dpi=240)
    plt.close(figure)

    errors = [float(summaries[name]["mean_absolute_calibration_error"]) for name in group_order]
    scores = [float(summaries[name]["calibration_score"]) for name in group_order]
    display_labels = [labels[name] for name in group_order]
    colors = [COLORS[index % len(COLORS)] for index in range(len(group_order))]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, max(4.2, 0.65 * len(group_order))))
    axes[0].barh(display_labels, errors, color=colors)
    axes[0].set_xlabel(r"$E_g$ (lower is better)")
    axes[0].set_xlim(0, max(0.12, max(errors) * 1.15))
    axes[1].barh(display_labels, scores, color=colors)
    axes[1].set_xlabel(r"$S_g=1-2E_g$ (higher is better)")
    axes[1].set_xlim(0, 1)
    figure.tight_layout()
    figure.savefig(output_dir / "calibration_error_and_score_by_group.png", dpi=240)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate probability ranks with equal statistical-group weight."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=FORMAL_BLOCKS)
    parser.add_argument(
        "--shift-radius", type=int, default=FORMAL_SHIFT_RADIUS_PX
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    manifest = load_manifest(args.manifest)
    blocks = int(getattr(args, "blocks", FORMAL_BLOCKS))
    shift_radius = int(
        getattr(args, "shift_radius", FORMAL_SHIFT_RADIUS_PX)
    )
    expected_distance_id = distance_identifier(blocks, shift_radius)
    expected_algorithm_version = algorithm_version(blocks, shift_radius)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else manifest.path.parent / "calibration_summary"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    incompatible: list[dict[str, object]] = []
    for spec in manifest.conditions:
        result_path = result_directory(
            spec,
            args.results_root,
            blocks=blocks,
            shift_radius=shift_radius,
        ) / "probability_rank.json"
        if not result_path.is_file():
            missing.append(str(result_path))
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            missing.append(str(result_path))
            continue
        if (
            result.get("package_version") != PACKAGE_VERSION
            or result.get("algorithm_version") != expected_algorithm_version
            or result.get("distance_id") != expected_distance_id
        ):
            incompatible.append(
                {
                    "condition_id": spec.condition_id,
                    "result_path": str(result_path),
                    "package_version": result.get("package_version"),
                    "algorithm_version": result.get("algorithm_version"),
                    "distance_id": result.get("distance_id"),
                    "expected_package_version": PACKAGE_VERSION,
                    "expected_algorithm_version": expected_algorithm_version,
                    "expected_distance_id": expected_distance_id,
                }
            )
            continue
        rows.append(
            {
                "condition_id": spec.condition_id,
                "statistical_group": spec.group,
                "statistical_group_label": spec.group_label,
                "package_version": result["package_version"],
                "algorithm_version": result["algorithm_version"],
                "distance_id": result["distance_id"],
                "probability_rank_u": float(result["probability_rank_u"]),
                "truth_model_density": float(result["truth_model_density"]),
                "kernel_sigma": float(result["kernel_sigma"]),
            }
        )
    if missing:
        (output_dir / "missing_results.json").write_text(
            json.dumps(missing, indent=2) + "\n", encoding="utf-8"
        )
        raise RuntimeError(
            f"expected {len(manifest.conditions)} complete ranks, found {len(rows)}"
        )
    if incompatible:
        (output_dir / "incompatible_results.json").write_text(
            json.dumps(incompatible, indent=2) + "\n", encoding="utf-8"
        )
        raise RuntimeError(
            f"found {len(incompatible)} ranks from an incompatible algorithm or distance"
        )

    with (output_dir / "condition_probability_ranks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups = {
        name: np.asarray(
            [row["probability_rank_u"] for row in rows if row["statistical_group"] == name],
            dtype=np.float64,
        )
        for name in manifest.group_order
    }
    if any(values.size == 0 for values in groups.values()):
        raise RuntimeError("one or more statistical groups are empty")
    group_summaries = {
        name: calibration_summary(groups[name]) for name in manifest.group_order
    }
    group_scores = [
        float(group_summaries[name]["calibration_score"]) for name in manifest.group_order
    ]
    group_errors = [
        float(group_summaries[name]["mean_absolute_calibration_error"])
        for name in manifest.group_order
    ]
    pooled = calibration_summary(
        np.asarray([row["probability_rank_u"] for row in rows], dtype=np.float64)
    )
    result: dict[str, object] = {
        "status": "complete",
        "package_version": PACKAGE_VERSION,
        "algorithm_version": expected_algorithm_version,
        "distance_id": expected_distance_id,
        "distance": {
            "blocks": [blocks, blocks],
            "patch_edge_px": 256 // blocks,
            "shift_radius_each_axis_px": shift_radius,
        },
        "n_conditions": len(rows),
        "n_statistical_groups": len(manifest.group_order),
        "primary_score_name": "equal_group_macro_calibration_score",
        "primary_score": float(np.mean(group_scores)),
        "primary_macro_error": float(np.mean(group_errors)),
        "primary_formula": "mean_g[max(0, 1 - 2*mean_alpha(|C_g(alpha)-alpha|))]",
        "group_weighting": "all statistical groups have equal weight",
        "group_order": list(manifest.group_order),
        "group_labels": manifest.group_labels,
        "group_scores": group_summaries,
        "primary_score_bootstrap_95ci": bootstrap_macro_score(
            groups,
            manifest.group_order,
            iterations=args.bootstrap,
            seed=args.seed,
        ),
        "pooled_diagnostic_not_used_as_primary": pooled,
        "reason_for_group_macro_average": (
            "prevents opposite coverage biases in different physical regimes from cancelling"
        ),
    }
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "group_calibration_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "statistical_group",
            "statistical_group_label",
            "n_conditions",
            "mean_absolute_calibration_error",
            "calibration_score",
            "maximum_absolute_calibration_error",
            "u_mean",
            "u_median",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name in manifest.group_order:
            summary = group_summaries[name]
            writer.writerow(
                {
                    "statistical_group": name,
                    "statistical_group_label": manifest.group_labels[name],
                    **{key: summary[key] for key in fieldnames[2:]},
                }
            )
    _save_plots(
        output_dir,
        group_summaries,
        groups,
        manifest.group_order,
        manifest.group_labels,
    )
    return result


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
