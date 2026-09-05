#!/usr/bin/env python3
"""Final simple magnetic texture evaluator.

The evaluator reports exactly three base metrics:
  - geom_score: real-space domain/boundary similarity, higher is better
  - boundary_chamfer_px: average boundary mismatch in pixels, lower is better
  - fourier_similarity: Fourier power-spectrum similarity, higher is better

It also reports a weighted final_score. Fourier contributes 50% of the score.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_array(path: Path, key: str | None = None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        if key is None:
            keys = list(archive.keys())
            if len(keys) != 1:
                raise ValueError(f"{path} has keys {keys}; pass the desired key")
            key = keys[0]
        return np.asarray(archive[key])
    if suffix in {".pt", ".pth"}:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Reading .pt/.pth files requires torch") from exc
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            if key is None:
                candidates = [
                    name
                    for name, value in obj.items()
                    if hasattr(value, "shape") and len(value.shape) >= 3
                ]
                if len(candidates) != 1:
                    raise ValueError(f"{path} has tensor keys {candidates}; pass the desired key")
                key = candidates[0]
            obj = obj[key]
        if hasattr(obj, "detach"):
            obj = obj.detach().cpu().numpy()
        return np.asarray(obj)
    raise ValueError(f"Unsupported file suffix: {path.suffix}")


def to_bchw(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim < 3:
        raise ValueError(f"{name} must have at least 3 dimensions, got {arr.shape}")
    if 3 not in arr.shape:
        raise ValueError(f"{name} must contain one channel axis of size 3, got {arr.shape}")

    if arr.ndim == 3:
        channel_axis = 0 if arr.shape[0] == 3 else 2
    elif arr.shape[1] == 3:
        channel_axis = 1
    elif arr.shape[-1] == 3:
        channel_axis = arr.ndim - 1
    else:
        channel_axis = [idx for idx, size in enumerate(arr.shape) if size == 3][0]

    arr = np.moveaxis(arr, channel_axis, -3)
    if arr.ndim == 3:
        return arr[None]
    leading = int(np.prod(arr.shape[:-3]))
    return arr.reshape(leading, 3, arr.shape[-2], arr.shape[-1])


def choose_phase(target_mz: np.ndarray, threshold: float, phase: str) -> str:
    if phase != "minority":
        return phase
    negative_area = int((target_mz < threshold).sum())
    positive_area = int((target_mz > threshold).sum())
    return "negative" if negative_area <= positive_area else "positive"


def make_mask(mz: np.ndarray, threshold: float, phase: str) -> np.ndarray:
    if phase == "negative":
        return mz < threshold
    if phase == "positive":
        return mz > threshold
    raise ValueError(f"Unknown phase: {phase}")


def mask_boundary(mask: np.ndarray, periodic: bool) -> np.ndarray:
    mask = mask.astype(bool)
    if periodic:
        boundary = (
            (mask != np.roll(mask, 1, axis=0))
            | (mask != np.roll(mask, -1, axis=0))
            | (mask != np.roll(mask, 1, axis=1))
            | (mask != np.roll(mask, -1, axis=1))
        )
        return np.argwhere(boundary)

    boundary = np.zeros_like(mask, dtype=bool)
    dy = mask[:-1, :] != mask[1:, :]
    dx = mask[:, :-1] != mask[:, 1:]
    boundary[:-1, :] |= dy
    boundary[1:, :] |= dy
    boundary[:, :-1] |= dx
    boundary[:, 1:] |= dx
    return np.argwhere(boundary)


def downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    idx = np.linspace(0, points.shape[0] - 1, max_points).round().astype(np.int64)
    return points[idx]


def min_distances(
    source: np.ndarray,
    target: np.ndarray,
    shape: tuple[int, int],
    periodic: bool,
    chunk: int = 1024,
) -> np.ndarray:
    if source.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    if target.shape[0] == 0:
        return np.full((source.shape[0],), np.inf, dtype=np.float64)

    h, w = shape
    target = target.astype(np.float64)
    result = np.empty((source.shape[0],), dtype=np.float64)
    for start in range(0, source.shape[0], chunk):
        part = source[start : start + chunk].astype(np.float64)
        dy = np.abs(part[:, None, 0] - target[None, :, 0])
        dx = np.abs(part[:, None, 1] - target[None, :, 1])
        if periodic:
            dy = np.minimum(dy, float(h) - dy)
            dx = np.minimum(dx, float(w) - dx)
        result[start : start + chunk] = np.sqrt((dy * dy + dx * dx).min(axis=1))
    return result


def safe_div(num: float, denom: float, empty_value: float = 1.0) -> float:
    return empty_value if denom == 0.0 else num / denom


def geometry_metrics(
    pred_mz: np.ndarray,
    target_mz: np.ndarray,
    threshold: float,
    phase: str,
    boundary_tolerance: float,
    max_boundary_points: int,
    periodic: bool,
) -> tuple[float, float]:
    resolved_phase = choose_phase(target_mz, threshold, phase)
    pred_mask = make_mask(pred_mz, threshold, resolved_phase)
    target_mask = make_mask(target_mz, threshold, resolved_phase)

    pred_area = float(pred_mask.sum())
    target_area = float(target_mask.sum())
    total_area = float(pred_mask.size)
    intersection = float((pred_mask & target_mask).sum())
    dice = safe_div(2.0 * intersection, pred_area + target_area)
    area_error = abs(pred_area - target_area) / max(total_area, 1.0)
    area_score = max(0.0, 1.0 - area_error)

    pred_boundary = downsample_points(mask_boundary(pred_mask, periodic), max_boundary_points)
    target_boundary = downsample_points(mask_boundary(target_mask, periodic), max_boundary_points)

    if pred_boundary.shape[0] == 0 and target_boundary.shape[0] == 0:
        boundary_f1 = 1.0
        boundary_chamfer_px = 0.0
    elif pred_boundary.shape[0] == 0 or target_boundary.shape[0] == 0:
        boundary_f1 = 0.0
        h, w = pred_mask.shape
        boundary_chamfer_px = float((h * h + w * w) ** 0.5)
    else:
        pred_to_target = min_distances(
            pred_boundary,
            target_boundary,
            pred_mask.shape,
            periodic,
        )
        target_to_pred = min_distances(
            target_boundary,
            pred_boundary,
            pred_mask.shape,
            periodic,
        )
        precision = float((pred_to_target <= boundary_tolerance).mean())
        recall = float((target_to_pred <= boundary_tolerance).mean())
        boundary_f1 = safe_div(2.0 * precision * recall, precision + recall, empty_value=0.0)
        boundary_chamfer_px = 0.5 * (float(pred_to_target.mean()) + float(target_to_pred.mean()))

    geom_score = 0.50 * dice + 0.35 * boundary_f1 + 0.15 * area_score
    return geom_score, boundary_chamfer_px


def fourier_similarity(pred_mz: np.ndarray, target_mz: np.ndarray, eps: float = 1e-12) -> float:
    pred = np.asarray(pred_mz, dtype=np.float64)
    target = np.asarray(target_mz, dtype=np.float64)
    pred = pred - float(pred.mean())
    target = target - float(target.mean())

    pred_power = np.abs(np.fft.fftshift(np.fft.fft2(pred))) ** 2
    target_power = np.abs(np.fft.fftshift(np.fft.fft2(target))) ** 2
    pred_power = pred_power / (float(pred_power.sum()) + eps)
    target_power = target_power / (float(target_power.sum()) + eps)

    l1 = float(np.abs(pred_power - target_power).sum())
    return float(np.clip(1.0 - 0.5 * l1, 0.0, 1.0))


def boundary_score_from_chamfer(chamfer_px: float, image_shape: tuple[int, int]) -> float:
    h, w = image_shape
    diag = float((h * h + w * w) ** 0.5)
    return float(np.clip(1.0 - chamfer_px / max(diag, 1.0), 0.0, 1.0))


def evaluate_sample(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float,
    phase: str,
    boundary_tolerance: float,
    max_boundary_points: int,
    periodic: bool,
) -> dict[str, float]:
    pred_mz = pred[2]
    target_mz = target[2]
    geom_score, chamfer_px = geometry_metrics(
        pred_mz,
        target_mz,
        threshold,
        phase,
        boundary_tolerance,
        max_boundary_points,
        periodic,
    )
    fourier = fourier_similarity(pred_mz, target_mz)
    chamfer_score = boundary_score_from_chamfer(chamfer_px, pred_mz.shape)
    final_score = 0.50 * fourier + 0.35 * geom_score + 0.15 * chamfer_score
    return {
        "geom_score": geom_score,
        "boundary_chamfer_px": chamfer_px,
        "fourier_similarity": fourier,
        "final_score": final_score,
    }


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0].keys()
    }


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", *rows[0].keys()])
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow({"sample_index": idx, **row})


def main() -> None:
    parser = argparse.ArgumentParser(description="Final magnetic texture evaluator.")
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--pred-key", default=None)
    parser.add_argument("--target-key", default=None)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--phase",
        choices=["negative", "positive", "minority"],
        default="negative",
        help="Which m_z phase to compare for geometry. Fourier always uses raw m_z.",
    )
    parser.add_argument("--boundary-tolerance", type=float, default=3.0)
    parser.add_argument("--max-boundary-points", type=int, default=6000)
    parser.add_argument("--periodic", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    pred = to_bchw(load_array(args.pred, args.pred_key), "pred")
    target = to_bchw(load_array(args.target, args.target_key), "target")
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch after normalization: {pred.shape} vs {target.shape}")

    rows = [
        evaluate_sample(
            pred[i],
            target[i],
            threshold=args.threshold,
            phase=args.phase,
            boundary_tolerance=args.boundary_tolerance,
            max_boundary_points=args.max_boundary_points,
            periodic=args.periodic,
        )
        for i in range(pred.shape[0])
    ]
    summary = {
        "num_samples": int(pred.shape[0]),
        "shape_bchw": list(pred.shape),
        "settings": {
            "threshold": args.threshold,
            "phase": args.phase,
            "boundary_tolerance": args.boundary_tolerance,
            "periodic": args.periodic,
            "final_score_weights": {
                "fourier_similarity": 0.50,
                "geom_score": 0.35,
                "boundary_chamfer_score": 0.15,
            },
        },
        "mean": mean_rows(rows),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.output_csv:
        write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
