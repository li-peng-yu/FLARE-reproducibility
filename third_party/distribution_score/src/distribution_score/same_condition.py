"""CLI for repeated MuMax truth versus repeated model samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .distance import (
    FORMAL_BLOCKS,
    FORMAL_SHIFT_RADIUS_PX,
    no_shift_distance,
    patch_shift_distance,
)
from .fields import normalize_batch, normalize_field, read_ovf
from .statistics import bootstrap_mean_interval, same_condition_score
from .texture import evaluate_sample
from .version import PACKAGE_VERSION, algorithm_version, distance_identifier


ARRAY_KEYS = {
    "model_density",
    "truth_density",
    "density_ratio",
    "log_density_ratio",
    "abs_log_density_ratio",
    "per_truth_symmetric_score",
}


def load_truth_repeats(
    condition_dir: Path,
    expected: int | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all complete ``segment_000 -> segment_001`` repeats."""

    targets: list[np.ndarray] = []
    labels: list[str] = []
    anchor: np.ndarray | None = None
    for repeat_dir in sorted((condition_dir / "repeats").glob("rep_*")):
        anchor_path = repeat_dir / "run.out" / "segment_000_end.ovf"
        target_path = repeat_dir / "run.out" / "segment_001_end.ovf"
        if not anchor_path.is_file() or not target_path.is_file():
            continue
        if anchor is None:
            anchor = normalize_field(read_ovf(anchor_path))
        targets.append(normalize_field(read_ovf(target_path)))
        labels.append(repeat_dir.name)
    if anchor is None or len(targets) < 2:
        raise RuntimeError("at least two complete MuMax repeats are required")
    if expected is not None and len(targets) != expected:
        raise RuntimeError(f"expected {expected} complete MuMax repeats, found {len(targets)}")
    return np.stack(targets), anchor, labels


def _periodic_boundary(condition_dir: Path) -> bool:
    params_path = condition_dir / "repeats" / "rep_000" / "params.json"
    if not params_path.is_file():
        return False
    params = json.loads(params_path.read_text(encoding="utf-8"))
    boundary = params.get("boundary", {}) or {}
    return bool(boundary.get("pbc_x", False) or boundary.get("pbc_y", False))


def _progress(device: torch.device):
    def report(index: int, total: int, valid: int) -> None:
        if index % 8 != 0 and index != total:
            return
        memory = (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else 0.0
        )
        print(f"patch={index:02d}/{total} valid={valid} memory_GB={memory:.2f}", flush=True)

    return report


def save_nearest_texture_metrics(
    output_dir: Path,
    model: np.ndarray,
    truth: np.ndarray,
    distance: np.ndarray,
    labels: list[str],
    periodic: bool,
) -> dict[str, float]:
    nearest = distance.argmin(axis=1)
    rows: list[dict[str, float | int | str]] = []
    for truth_index, model_index in enumerate(nearest):
        metrics = evaluate_sample(
            model[int(model_index)],
            truth[truth_index],
            threshold=0.0,
            phase="minority",
            boundary_tolerance=3.0,
            max_boundary_points=6000,
            periodic=periodic,
        )
        rows.append(
            {
                "truth_index": truth_index,
                "truth_label": labels[truth_index],
                "nearest_model_index": int(model_index),
                "patch_shift_distance": float(distance[truth_index, model_index]),
                **metrics,
            }
        )
    with (output_dir / "nearest_texture_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in ("geom_score", "boundary_chamfer_px", "fourier_similarity", "final_score")
    }


def save_diagnostics(
    output_dir: Path,
    model: np.ndarray,
    truth: np.ndarray,
    distance: np.ndarray,
    density_ratio: np.ndarray,
    absolute_log_ratio: np.ndarray,
    point_score: np.ndarray,
) -> None:
    nearest = distance.argmin(axis=1)
    order = np.argsort(absolute_log_ratio)
    row_count = min(20, len(order))
    selected = np.linspace(0, len(order) - 1, row_count).round().astype(int)
    selected_truth = order[selected]
    figure, axes = plt.subplots(row_count, 2, figsize=(5.3, 1.9 * row_count), squeeze=False)
    for row, truth_index in enumerate(selected_truth):
        model_index = int(nearest[truth_index])
        axes[row, 0].imshow(truth[truth_index, 2], origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
        axes[row, 1].imshow(model[model_index, 2], origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
        axes[row, 0].set_ylabel(
            f"truth {truth_index:03d}\nr={density_ratio[truth_index]:.3g} "
            f"s={point_score[truth_index]:.3f}",
            fontsize=7,
        )
        axes[row, 1].set_ylabel(
            f"model {model_index:04d}\nD={distance[truth_index, model_index]:.3f}",
            fontsize=7,
        )
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_title("MuMax truth")
    axes[0, 1].set_title("nearest model sample")
    figure.tight_layout()
    figure.savefig(output_dir / "nearest_matches.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].hist(distance.min(axis=1), bins=20, color="#2F67A8")
    axes[0].set_xlabel("nearest patch-shift distance")
    axes[0].set_ylabel("MuMax count")
    axes[1].hist(point_score, bins=20, range=(0, 1), color="#137C8B")
    axes[1].set_xlabel("per-MuMax density-ratio score")
    axes[2].imshow(distance, aspect="auto", cmap="viridis")
    axes[2].set_xlabel("model sample")
    axes[2].set_ylabel("MuMax repeat")
    figure.tight_layout()
    figure.savefig(output_dir / "distribution_diagnostics.png", dpi=220)
    plt.close(figure)


def _serializable(summary: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in summary.items() if key not in ARRAY_KEYS}


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    condition_dir = args.condition_dir.resolve()
    default_output = condition_dir / (
        f"same_condition_score_{args.blocks}x{args.blocks}_shift{args.shift_radius}"
    )
    output_dir = (args.output_dir or default_output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    model_path = args.model_samples or condition_dir / "model_samples" / "model_samples_f16.npy"
    model_memmap = np.load(model_path, mmap_mode="r", allow_pickle=False)
    if model_memmap.ndim != 4 or model_memmap.shape[1:] != (3, 256, 256):
        raise ValueError(f"expected model shape (N,3,256,256), got {model_memmap.shape}")
    model_count = model_memmap.shape[0] if args.num_model is None else args.num_model
    if model_count <= 0 or model_memmap.shape[0] < model_count:
        raise ValueError(f"requested {model_count} model samples, file contains {model_memmap.shape[0]}")

    truth, anchor, labels = load_truth_repeats(condition_dir, args.num_mumax)
    geometry_mask = np.linalg.norm(anchor, axis=0) > 0.5
    if not geometry_mask.any():
        raise RuntimeError("empty geometry mask")
    model = normalize_batch(np.asarray(model_memmap[:model_count], dtype=np.float32))
    model *= geometry_mask[None, None]
    truth *= geometry_mask[None, None]

    np.save(output_dir / "mumax_targets_f16.npy", truth.astype(np.float16))
    np.save(output_dir / "geometry_mask.npy", geometry_mask)
    model_tensor = torch.from_numpy(model).to(device=device, dtype=torch.float32)
    truth_tensor = torch.from_numpy(truth).to(device=device, dtype=torch.float32)
    mask_tensor = torch.from_numpy(geometry_mask.astype(np.float32))[None, None].to(device)

    started = time.time()
    progress = _progress(device) if args.progress else None
    cross_distance, valid_patches = patch_shift_distance(
        model_tensor,
        truth_tensor,
        mask_tensor,
        blocks=args.blocks,
        shift_radius=args.shift_radius,
        reference_chunk=args.reference_chunk,
        progress=progress,
    )
    self_distance, _ = patch_shift_distance(
        truth_tensor,
        truth_tensor,
        mask_tensor,
        blocks=args.blocks,
        shift_radius=args.shift_radius,
        reference_chunk=min(args.reference_chunk, len(truth)),
        progress=progress,
    )
    np.fill_diagonal(self_distance, 0.0)
    control_cross = no_shift_distance(model_tensor, truth_tensor, mask_tensor)
    control_self = no_shift_distance(truth_tensor, truth_tensor, mask_tensor)
    np.fill_diagonal(control_self, 0.0)

    np.save(output_dir / "truth_to_model_patch_shift_distance.npy", cross_distance)
    np.save(output_dir / "truth_self_patch_shift_distance.npy", self_distance)
    primary = same_condition_score(cross_distance, self_distance)
    control = same_condition_score(control_cross, control_self)
    sigma = float(primary["sigma"])
    sensitivity = {
        str(factor): _serializable(
            same_condition_score(cross_distance, self_distance, sigma=sigma * factor)
        )
        for factor in (0.5, 1.0, 2.0)
    }

    error_interval = bootstrap_mean_interval(
        np.asarray(primary["abs_log_density_ratio"]),
        iterations=args.bootstrap,
        seed=args.seed,
    )
    score_interval = [math.exp(-error_interval[1]), math.exp(-error_interval[0])]
    texture_summary = None
    if not args.skip_auxiliary_texture:
        texture_summary = save_nearest_texture_metrics(
            output_dir,
            model,
            truth,
            cross_distance,
            labels,
            _periodic_boundary(condition_dir),
        )
    save_diagnostics(
        output_dir,
        model,
        truth,
        cross_distance,
        np.asarray(primary["density_ratio"]),
        np.asarray(primary["abs_log_density_ratio"]),
        np.asarray(primary["per_truth_symmetric_score"]),
    )

    result: dict[str, object] = {
        "status": "complete",
        "package_version": PACKAGE_VERSION,
        "algorithm_version": algorithm_version(args.blocks, args.shift_radius),
        "distance_id": distance_identifier(args.blocks, args.shift_radius),
        "condition_id": condition_dir.name,
        "num_model_samples": int(model_count),
        "num_mumax_repeats": int(len(truth)),
        "distance": {
            "resolution": [256, 256],
            "magnetization_components": 3,
            "blocks": [args.blocks, args.blocks],
            "patch_edge_px": 256 // args.blocks,
            "shift_radius_each_axis_px": args.shift_radius,
            "integer_shifts": True,
            "downsampling": False,
            "shift_penalty": 0.0,
            "symmetric": True,
            "valid_geometry_patches": valid_patches,
        },
        "primary_score_definition": {
            "name": "symmetric_local_density_ratio_score",
            "formula": "exp(-mean_i(abs(log((q_model_i+eps)/(q_mumax_loo_i+eps)))))",
            "range": [0.0, 1.0],
            "higher_is_better": True,
        },
        "primary_patch_shift": _serializable(primary),
        "primary_symmetric_log_ratio_mae_bootstrap_95ci": error_interval,
        "primary_symmetric_ratio_score_bootstrap_95ci": score_interval,
        "sigma_sensitivity": sensitivity,
        "no_shift_control": _serializable(control),
        "nearest_texture_metric_mean": texture_summary,
        "elapsed_seconds": time.time() - started,
        "max_cuda_memory_gb": (
            torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
        ),
    }
    np.save(output_dir / "model_density.npy", primary["model_density"])
    np.save(output_dir / "mumax_leave_one_out_density.npy", primary["truth_density"])
    np.save(output_dir / "density_ratio.npy", primary["density_ratio"])
    np.save(output_dir / "per_mumax_symmetric_score.npy", primary["per_truth_symmetric_score"])
    (output_dir / "scores.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score repeated model and MuMax outcomes at one fixed condition."
    )
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--model-samples", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-model", type=int, default=None)
    parser.add_argument("--num-mumax", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=FORMAL_BLOCKS)
    parser.add_argument(
        "--shift-radius", type=int, default=FORMAL_SHIFT_RADIUS_PX
    )
    parser.add_argument("--reference-chunk", type=int, default=256)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--skip-auxiliary-texture", action="store_true")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
