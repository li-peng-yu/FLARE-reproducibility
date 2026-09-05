#!/usr/bin/env python3
"""Render frozen Table-10 geometry-ablation predictions for visual inspection.

The script uses the exact-anchor, five-draw seed-78 artifacts that underlie the
multi-seed geometry table.  Cases are selected by fixed quantiles of the
Cartesian-CFM minus core mean-draw angular error, rather than by visual review.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
FORMAL = PROJECT / "outputs/skx_bt_1000base_x5/paper_formal_20260822"
CORE_SAMPLES = (
    PROJECT
    / "graph/results/x5_exact_anchor_forecasts_20260822/"
    "standard_prior_samples_25_v010"
)
TRUTH_ROOT = (
    PROJECT
    / "graph/results/x5_distribution_pareto_20260822/"
    "formal_standard_prior_n5_v020/scfm_stage1"
)
SCORE_DIR = "same_condition_score_4x4_shift16"


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    sample_root: Path
    metrics_csv: Path


METHODS = (
    Method(
        "core_s78",
        "Core local axis--angle",
        CORE_SAMPLES,
        FORMAL / "seed_exact_anchor_summary/cart_s78/per_anchor_metrics.csv",
    ),
    Method(
        "cart_s78",
        "Cartesian CFM",
        FORMAL / "seed_exact_anchor_samples/cart_s78",
        FORMAL / "seed_exact_anchor_summary/cart_s78/per_anchor_metrics.csv",
    ),
    Method(
        "a2d_s78",
        "Fixed 2D tangent basis",
        FORMAL / "seed_exact_anchor_samples/a2d_s78",
        FORMAL / "seed_exact_anchor_summary/a2d_s78/per_anchor_metrics.csv",
    ),
    Method(
        "rfm_s78",
        "Riemannian FM",
        FORMAL / "seed_exact_anchor_samples/rfm_s78",
        FORMAL / "seed_exact_anchor_summary/rfm_s78/per_anchor_metrics.csv",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT
            / "outputs/skx_bt_1000base_x5/paper_revision_20260824/"
            "seed_geometry_visualization"
        ),
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=(0.10, 0.50, 0.90, 1.00),
        help="Quantiles of Cartesian-minus-core mean-draw angular error.",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Use a compact, publication-scale overview layout.",
    )
    return parser.parse_args()


def load_metric_rows(method: Method) -> dict[tuple[str, int], dict[str, float]]:
    rows: dict[tuple[str, int], dict[str, float]] = {}
    with method.metrics_csv.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if raw["method"] != method.key:
                continue
            key = (raw["condition_id"], int(raw["truth_repeat_index"]))
            rows[key] = {
                "mean_draw_angle_deg": float(raw["mean_draw_angle_deg"]),
                "ensemble_mean_angle_deg": float(raw["ensemble_mean_angle_deg"]),
                "fair_energy_score": float(raw["fair_energy_score"]),
            }
    if len(rows) != 330:
        raise RuntimeError(
            f"expected 330 exact-anchor rows for {method.key}, found {len(rows)}"
        )
    return rows


def normalize(fields: np.ndarray) -> np.ndarray:
    values = np.asarray(fields, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, np.maximum(norms, 1.0e-8), out=np.zeros_like(values))


def condition_paths(root: Path, condition_id: str) -> tuple[Path, Path]:
    condition = root / "conditions" / condition_id
    return (
        condition / "model_samples/model_samples_f16.npy",
        condition / "model_samples/anchor_repeat_index.npy",
    )


def load_draws(method: Method, condition_id: str, repeat: int) -> np.ndarray:
    sample_path, assignment_path = condition_paths(method.sample_root, condition_id)
    samples = np.load(sample_path, mmap_mode="r")
    assignments = np.asarray(np.load(assignment_path), dtype=np.int64)
    indices = np.flatnonzero(assignments == repeat)
    if indices.size != 5:
        raise RuntimeError(
            f"expected five draws for {method.key}, {condition_id}, repeat {repeat}; "
            f"found {indices.size}"
        )
    return normalize(np.asarray(samples[indices], dtype=np.float32))


def load_truth_and_mask(condition_id: str, repeat: int) -> tuple[np.ndarray, np.ndarray]:
    score_root = TRUTH_ROOT / "conditions" / condition_id / SCORE_DIR
    truth = normalize(
        np.asarray(np.load(score_root / "mumax_targets_f16.npy", mmap_mode="r"))
    )[repeat]
    mask = np.asarray(np.load(score_root / "geometry_mask.npy"), dtype=bool)
    if mask.ndim == 3:
        mask = mask.squeeze()
    if mask.shape != truth.shape[-2:]:
        raise RuntimeError(
            f"mask/field mismatch for {condition_id}: {mask.shape} vs {truth.shape}"
        )
    return truth, mask


def mean_angle_deg(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    dots = np.sum(a * b, axis=0)
    angles = np.arccos(np.clip(dots[mask], -1.0, 1.0))
    return float(np.degrees(angles).mean())


def medoid_index(draws: np.ndarray, mask: np.ndarray) -> int:
    costs = []
    for index in range(draws.shape[0]):
        costs.append(
            np.mean(
                [
                    mean_angle_deg(draws[index], draws[other], mask)
                    for other in range(draws.shape[0])
                    if other != index
                ]
            )
        )
    return int(np.argmin(costs))


def image_mz(axis: plt.Axes, field: np.ndarray, mask: np.ndarray) -> None:
    values = field[2].copy()
    values[~mask] = np.nan
    axis.imshow(values, origin="lower", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.6)


def select_cases(
    core: dict[tuple[str, int], dict[str, float]],
    cart: dict[tuple[str, int], dict[str, float]],
    quantiles: list[float],
) -> list[dict[str, object]]:
    values = sorted(
        (
            cart[key]["mean_draw_angle_deg"] - core[key]["mean_draw_angle_deg"],
            key[0],
            key[1],
        )
        for key in core.keys() & cart.keys()
    )
    gaps = np.asarray([value[0] for value in values], dtype=np.float64)
    selected: list[dict[str, object]] = []
    used: set[tuple[str, int]] = set()
    for quantile in quantiles:
        target = float(np.quantile(gaps, quantile))
        candidates = sorted(values, key=lambda item: (abs(item[0] - target), item[1], item[2]))
        choice = next(item for item in candidates if (item[1], item[2]) not in used)
        used.add((choice[1], choice[2]))
        selected.append(
            {
                "quantile": float(quantile),
                "gap_deg": float(choice[0]),
                "condition_id": choice[1],
                "repeat": int(choice[2]),
            }
        )
    return selected


def render_overview(
    output: Path,
    selected: list[dict[str, object]],
    metrics: dict[str, dict[tuple[str, int], dict[str, float]]],
    *,
    paper: bool = False,
) -> list[dict[str, object]]:
    columns = 1 + len(METHODS)
    figure_size = (7.15, 1.38 * len(selected)) if paper else (2.0 * columns, 2.05 * len(selected))
    figure, axes = plt.subplots(
        len(selected),
        columns,
        figsize=figure_size,
        squeeze=False,
    )
    paper_titles = {
        "core_s78": "Core local\naxis–angle",
        "cart_s78": "Cartesian\nCFM",
        "a2d_s78": "Fixed 2D\ntangent basis",
        "rfm_s78": "Riemannian\nFM",
    }
    records: list[dict[str, object]] = []
    for row_index, selected_case in enumerate(selected):
        condition_id = str(selected_case["condition_id"])
        repeat = int(selected_case["repeat"])
        truth, mask = load_truth_and_mask(condition_id, repeat)
        image_mz(axes[row_index, 0], truth, mask)
        if row_index == 0:
            axes[row_index, 0].set_title(
                "MuMax3\ntarget" if paper else "MuMax3 target",
                fontsize=6.8 if paper else 9,
                pad=2 if paper else 4,
            )
        row_label = (
            f"q={float(selected_case['quantile']):.2f}\n"
            f"base{condition_id[4:8]}, seg. {condition_id.split('segment')[1][:3]}, r{repeat}\n"
            f"gap {float(selected_case['gap_deg']):+.1f}°"
            if paper
            else
            f"q={float(selected_case['quantile']):.2f}\n"
            f"{condition_id.replace('_segment', ' / seg')} / r{repeat}\n"
            f"Cart-core gap {float(selected_case['gap_deg']):+.1f} deg"
        )
        axes[row_index, 0].set_ylabel(
            row_label,
            fontsize=5.4 if paper else 7.2,
            rotation=0,
            ha="right",
            va="center",
            labelpad=4 if paper else 8,
        )
        record: dict[str, object] = dict(selected_case)
        for column_index, method in enumerate(METHODS, start=1):
            draws = load_draws(method, condition_id, repeat)
            medoid = medoid_index(draws, mask)
            image_mz(axes[row_index, column_index], draws[medoid], mask)
            if row_index == 0:
                axes[row_index, column_index].set_title(
                    paper_titles[method.key] if paper else method.label,
                    fontsize=6.8 if paper else 9,
                    pad=2 if paper else 4,
                )
            mean_draw = metrics[method.key][(condition_id, repeat)]["mean_draw_angle_deg"]
            axes[row_index, column_index].text(
                0.03,
                0.04,
                f"{mean_draw:.1f}°" if paper else f"mean {mean_draw:.1f} deg",
                transform=axes[row_index, column_index].transAxes,
                fontsize=4.8 if paper else 6.8,
                color="#111111",
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.2},
            )
            record[f"{method.key}_medoid_draw"] = medoid
            record[f"{method.key}_mean_draw_angle_deg"] = mean_draw
        records.append(record)

    if not paper:
        figure.suptitle(
            "Geometry-ablation predictions (seed 78; ensemble-medoid draw)",
            fontsize=12,
            y=0.995,
        )
        figure.text(
            0.5,
            0.006,
            "Case selection uses fixed quantiles of Cartesian-CFM minus core exact-anchor mean-draw angular error; m_z in [-1, 1].",
            ha="center",
            fontsize=7.5,
        )
    figure.subplots_adjust(
        left=0.135 if paper else 0.19,
        right=0.998 if paper else 0.995,
        top=0.925 if paper else 0.945,
        bottom=0.015 if paper else 0.035,
        wspace=0.025 if paper else 0.035,
        hspace=0.045 if paper else 0.08,
    )
    figure.savefig(output.with_suffix(".png"), dpi=240)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)
    return records


def render_draws(
    output: Path,
    selected_case: dict[str, object],
    metrics: dict[str, dict[tuple[str, int], dict[str, float]]],
) -> dict[str, object]:
    condition_id = str(selected_case["condition_id"])
    repeat = int(selected_case["repeat"])
    truth, mask = load_truth_and_mask(condition_id, repeat)
    figure, axes = plt.subplots(
        len(METHODS),
        6,
        figsize=(12.4, 1.82 * len(METHODS)),
        squeeze=False,
    )
    record: dict[str, object] = dict(selected_case)
    for row_index, method in enumerate(METHODS):
        image_mz(axes[row_index, 0], truth, mask)
        draws = load_draws(method, condition_id, repeat)
        medoid = medoid_index(draws, mask)
        for draw_index in range(5):
            image_mz(axes[row_index, draw_index + 1], draws[draw_index], mask)
            if draw_index == medoid:
                for spine in axes[row_index, draw_index + 1].spines.values():
                    spine.set_color("#111111")
                    spine.set_linewidth(1.6)
        axes[row_index, 0].set_ylabel(
            f"{method.label}\nmean error "
            f"{metrics[method.key][(condition_id, repeat)]['mean_draw_angle_deg']:.1f} deg",
            fontsize=8,
            rotation=0,
            ha="right",
            va="center",
            labelpad=8,
        )
        record[f"{method.key}_medoid_draw"] = medoid
        if row_index == 0:
            axes[row_index, 0].set_title("MuMax3 target", fontsize=9)
            for draw_index in range(5):
                axes[row_index, draw_index + 1].set_title(f"draw {draw_index + 1}", fontsize=9)

    figure.suptitle(
        f"Five exact-anchor draws: {condition_id}, repeat {repeat}\n"
        f"Cartesian-core mean-error gap {float(selected_case['gap_deg']):+.2f} deg",
        fontsize=11,
        y=0.995,
    )
    figure.text(
        0.5,
        0.006,
        "Black border marks each method's ensemble medoid; all panels use the same m_z scale [-1, 1].",
        ha="center",
        fontsize=7.5,
    )
    figure.subplots_adjust(left=0.235, right=0.995, top=0.91, bottom=0.04, wspace=0.035, hspace=0.06)
    figure.savefig(output.with_suffix(".png"), dpi=240)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)
    return record


def main() -> None:
    args = parse_args()
    if any(not 0.0 <= quantile <= 1.0 for quantile in args.quantiles):
        raise ValueError(f"quantiles must lie in [0, 1]: {args.quantiles}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {method.key: load_metric_rows(method) for method in METHODS}
    selected = select_cases(metrics["core_s78"], metrics["cart_s78"], args.quantiles)
    overview_records = render_overview(
        args.output_dir / "seed78_geometry_prediction_overview",
        selected,
        metrics,
        paper=args.paper,
    )
    high_case = selected[-2] if len(selected) > 1 else selected[-1]
    worst_case = selected[-1]
    draw_records = [
        render_draws(
            args.output_dir / "seed78_geometry_prediction_draws_high",
            high_case,
            metrics,
        ),
        render_draws(
            args.output_dir / "seed78_geometry_prediction_draws_worst",
            worst_case,
            metrics,
        ),
    ]
    payload = {
        "status": "complete",
        "selection_policy": (
            "Fixed quantiles of Cartesian-CFM minus core seed-78 exact-anchor "
            "mean-draw angular error; no visual selection."
        ),
        "quantiles": [float(value) for value in args.quantiles],
        "overview_cases": overview_records,
        "draw_cases": draw_records,
        "method_sample_roots": {
            method.key: str(method.sample_root) for method in METHODS
        },
        "truth_root": str(TRUTH_ROOT),
    }
    (args.output_dir / "selection_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
