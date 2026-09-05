#!/usr/bin/env python3
"""Measure local anchor-to-target rotation angles on the paper test conditions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from skyrmion_cfm.data.ovf import read_ovf


FALLBACK_COSINE = -0.999
FALLBACK_ANGLE_DEG = float(np.degrees(np.arccos(FALLBACK_COSINE)))
DEFAULT_THRESHOLDS_DEG = (150.0, 165.0, 170.0, 175.0, FALLBACK_ANGLE_DEG, 179.0)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _threshold_key(threshold: float) -> str:
    return f"fraction_ge_{threshold:.6f}_deg".replace(".", "p")


def _pair_angles(anchor_path: Path, target_path: Path, mask: np.ndarray) -> np.ndarray:
    anchor = np.asarray(read_ovf(anchor_path), dtype=np.float32)
    target = np.asarray(read_ovf(target_path), dtype=np.float32)
    if anchor.shape != target.shape or anchor.ndim != 3 or anchor.shape[0] != 3:
        raise ValueError(f"incompatible fields: {anchor.shape}, {target.shape}")
    if mask.shape != anchor.shape[-2:]:
        raise ValueError(f"incompatible mask: {mask.shape}, {anchor.shape}")
    anchor_norm = np.linalg.norm(anchor, axis=0)
    target_norm = np.linalg.norm(target, axis=0)
    valid = mask & (anchor_norm > 1.0e-6) & (target_norm > 1.0e-6)
    if not np.any(valid):
        raise ValueError(f"no valid cells for {anchor_path} -> {target_path}")
    dot = np.sum(anchor * target, axis=0)
    dot = dot / np.maximum(anchor_norm * target_norm, 1.0e-12)
    return np.degrees(np.arccos(np.clip(dot[valid], -1.0, 1.0))).astype(
        np.float32,
        copy=False,
    )


def _clustered_mean(
    rows: list[dict[str, Any]],
    key: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["base_id"])].append(float(row[key]))
    group_ids = sorted(grouped)
    values = np.asarray(
        [np.mean(grouped[group_id], dtype=np.float64) for group_id in group_ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selected = rng.integers(0, len(values), size=len(values))
        draws[index] = values[selected].mean()
    return {
        "base_groups": len(group_ids),
        "pair_equal_base_balanced_mean": float(values.mean()),
        "bootstrap_95ci": np.quantile(draws, [0.025, 0.975]).tolist(),
        "bootstrap_unit": "base group; both segment roles and five repeats retained",
    }


def _summarize_angles(
    arrays: list[np.ndarray],
    rows: list[dict[str, Any]],
    thresholds: tuple[float, ...],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    combined = np.concatenate(arrays).astype(np.float32, copy=False)
    threshold_summary: dict[str, Any] = {}
    for index, threshold in enumerate(thresholds):
        key = _threshold_key(threshold)
        count = int(np.count_nonzero(combined >= threshold))
        threshold_summary[f"{threshold:.6f}"] = {
            "angle_deg": threshold,
            "sites": count,
            "cell_weighted_fraction": count / combined.size,
            "pairs_with_any": int(sum(float(row[key]) > 0.0 for row in rows)),
            "pair_count": len(rows),
            **_clustered_mean(
                rows,
                key,
                iterations=iterations,
                seed=seed + 10_007 * index,
            ),
        }
    return {
        "valid_sites": int(combined.size),
        "pairs": len(rows),
        "angle_quantiles_deg": {
            f"q{quantile:g}": float(np.percentile(combined, quantile))
            for quantile in (50.0, 90.0, 95.0, 99.0, 99.9, 99.99, 100.0)
        },
        "thresholds": threshold_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--thresholds-deg",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS_DEG,
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=208_240_900)
    args = parser.parse_args()
    thresholds = tuple(sorted(set(float(value) for value in args.thresholds_deg)))
    if any(not 0.0 < value <= 180.0 for value in thresholds):
        raise ValueError(f"invalid thresholds: {thresholds}")

    rows: list[dict[str, Any]] = []
    arrays: list[np.ndarray] = []
    arrays_by_role: dict[str, list[np.ndarray]] = defaultdict(list)
    rows_by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    condition_dirs = sorted(
        path
        for path in args.condition_root.iterdir()
        if path.is_dir() and (path / "condition_metadata.json").is_file()
    )
    if not condition_dirs:
        raise RuntimeError(f"no paper conditions below {args.condition_root}")

    for condition_dir in condition_dirs:
        metadata = json.loads(
            (condition_dir / "condition_metadata.json").read_text(encoding="utf-8")
        )
        mask_path = (
            condition_dir
            / "same_condition_score_4x4_shift16"
            / "geometry_mask.npy"
        )
        mask = np.asarray(np.load(mask_path), dtype=np.bool_).squeeze()
        role = str(metadata["segment_role"])
        for repeat in metadata["repeats"]:
            angles = _pair_angles(
                Path(repeat["anchor_ovf"]),
                Path(repeat["truth_ovf"]),
                mask,
            )
            row: dict[str, Any] = {
                "condition_id": metadata["condition_id"],
                "base_id": str(metadata["base_id"]),
                "segment_role": role,
                "repeat_position": int(repeat["repeat_position"]),
                "horizon_ns": float(repeat["horizon_ns"]),
                "valid_sites": int(angles.size),
                "mean_angle_deg": float(angles.mean(dtype=np.float64)),
                "max_angle_deg": float(angles.max()),
                "q99_angle_deg": float(np.percentile(angles, 99.0)),
                "q999_angle_deg": float(np.percentile(angles, 99.9)),
            }
            for threshold in thresholds:
                row[_threshold_key(threshold)] = float(np.mean(angles >= threshold))
            rows.append(row)
            arrays.append(angles)
            rows_by_role[role].append(row)
            arrays_by_role[role].append(angles)

    summary = {
        "schema": "x5_paper_antipodal_rotation_statistics_v1",
        "status": "complete",
        "scope": (
            "the 66 same-condition paper macro-conditions, five matched MuMax "
            "anchor-target repeats per condition"
        ),
        "rotation": "valid-cell principal angle arccos(m_anchor dot m_target)",
        "implementation_fallback": {
            "criterion": "cos(theta) < -0.999",
            "equivalent_angle_deg": FALLBACK_ANGLE_DEG,
        },
        "conditions": len(condition_dirs),
        "pairs": len(rows),
        "thresholds_deg": list(thresholds),
        "combined": _summarize_angles(
            arrays,
            rows,
            thresholds,
            iterations=args.bootstrap,
            seed=args.seed,
        ),
        "by_segment_role": {
            role: _summarize_angles(
                arrays_by_role[role],
                rows_by_role[role],
                thresholds,
                iterations=args.bootstrap,
                seed=args.seed + 100_003 * index,
            )
            for index, role in enumerate(sorted(arrays_by_role))
        },
        "condition_root": str(args.condition_root),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output_dir / "summary.json", summary)
    csv_path = args.output_dir / "pair_statistics.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)

    lines = [
        "# Near-antipodal rotation statistics",
        "",
        f"Scope: {len(condition_dirs)} conditions and {len(rows)} matched MuMax anchor-target pairs.",
        "",
        "| Threshold | Cell-weighted fraction | Pairs with any | Base-balanced mean [95% CI] |",
        "|---:|---:|---:|---:|",
    ]
    for threshold in thresholds:
        item = summary["combined"]["thresholds"][f"{threshold:.6f}"]
        lines.append(
            f"| {threshold:.3f}° | {item['cell_weighted_fraction']:.6g} | "
            f"{item['pairs_with_any']}/{item['pair_count']} | "
            f"{item['pair_equal_base_balanced_mean']:.6g} "
            f"[{item['bootstrap_95ci'][0]:.6g}, {item['bootstrap_95ci'][1]:.6g}] |"
        )
    lines.extend(
        [
            "",
            "The implementation fallback is triggered at cos(theta) < -0.999, "
            f"equivalent to theta > {FALLBACK_ANGLE_DEG:.3f}°.",
            "",
        ]
    )
    _atomic_text(args.output_dir / "REPORT.md", "\n".join(lines))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
