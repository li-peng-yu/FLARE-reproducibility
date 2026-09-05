#!/usr/bin/env python3
"""Independent five-versus-five MuMax3 self-score from x30 repeats."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    PROJECT
    / "outputs/skx_bt_165base_x30/evaluation_20260818/"
    "same_condition_density_4x4_shift16_v020/"
    "x5_scratch50k_20260812_on_x30_test"
)
DEFAULT_OUTPUT = (
    PROJECT
    / "outputs/skx_bt_1000base_x5/paper_formal_20260822/"
    "distribution_sensitivity_observables/x30_mumax_self_score"
)


def _folds(condition_id: str, count: int) -> list[np.ndarray]:
    if count != 30:
        raise ValueError(f"expected 30 independent repeats for {condition_id}, got {count}")
    ordered = sorted(
        range(count),
        key=lambda index: hashlib.sha256(f"20260822|{condition_id}|{index}".encode()).hexdigest(),
    )
    return [np.asarray(ordered[start : start + 5], dtype=np.int64) for start in range(0, 30, 5)]


def _score(reference: np.ndarray, model: np.ndarray, distance: np.ndarray) -> float:
    ref = distance[np.ix_(reference, reference)]
    cross = distance[np.ix_(reference, model)]
    nearest = np.where(np.eye(len(reference), dtype=bool), np.inf, ref).min(axis=1)
    sigma = max(float(np.median(nearest)), 1.0e-12)
    ref_kernel = np.exp(-(ref**2) / (2.0 * sigma**2))
    model_kernel = np.exp(-(cross**2) / (2.0 * sigma**2))
    q_reference = (ref_kernel.sum(axis=1) - np.diag(ref_kernel)) / (len(reference) - 1)
    q_model = model_kernel.mean(axis=1)
    return math.exp(
        -float(np.abs(np.log((q_model + 1.0e-12) / (q_reference + 1.0e-12))).mean())
    )


def _aggregate(rows: list[dict[str, Any]], *, iterations: int, seed: int) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["base_id"])].append(math.log(float(row["self_score"])))
    groups = sorted(grouped)
    observed = np.asarray([value for group in groups for value in grouped[group]], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selected = rng.integers(0, len(groups), size=len(groups))
        values = [value for group_index in selected for value in grouped[groups[int(group_index)]]]
        draws[index] = math.exp(float(np.mean(values)))
    return {
        "geometric_mean": math.exp(float(observed.mean())),
        "bootstrap_95ci": np.quantile(draws, [0.025, 0.975]).tolist(),
        "conditions": len(rows),
        "base_groups": len(groups),
        "bootstrap_unit": "x30 base group, retaining both segments",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for condition in sorted((args.root / "conditions").iterdir()):
        score_dir = condition / "same_condition_score_4x4_shift16"
        metadata_path = condition / "condition_metadata.json"
        distance_path = score_dir / "truth_self_patch_shift_distance.npy"
        if not metadata_path.is_file() or not distance_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text())
        distance = np.load(distance_path).astype(np.float64)
        folds = _folds(condition.name, len(distance))
        pair_scores = [
            _score(folds[reference], folds[model], distance)
            for reference in range(len(folds))
            for model in range(len(folds))
            if reference != model
        ]
        rows.append(
            {
                "condition_id": condition.name,
                "base_id": str(metadata["base_id"]).zfill(4),
                "segment_role": str(metadata["segment_role"]),
                "ordered_disjoint_fold_pairs": len(pair_scores),
                "self_score": math.exp(float(np.log(pair_scores).mean())),
                "pair_score_arithmetic_mean": float(np.mean(pair_scores)),
                "pair_score_min": float(np.min(pair_scores)),
                "pair_score_max": float(np.max(pair_scores)),
            }
        )
    if len(rows) != 34:
        raise RuntimeError(f"expected 34 x30 conditions, found {len(rows)}")

    summary = {
        "status": "complete",
        "design": (
            "Thirty independent MuMax3 repeats are deterministically partitioned into six "
            "disjoint folds of five. Each condition score is the geometric mean over all 30 "
            "ordered reference/model fold pairs. Every pair uses the Table-1 five-versus-five "
            "budget and recomputes bandwidth from its five reference repeats only."
        ),
        "combined": _aggregate(rows, iterations=args.bootstrap, seed=args.seed),
        "by_segment": {
            role: _aggregate(
                [row for row in rows if row["segment_role"] == role],
                iterations=args.bootstrap,
                seed=args.seed + index + 1,
            )
            for index, role in enumerate(sorted({row["segment_role"] for row in rows}))
        },
        "scope": (
            "This is an independent simulator self-consistency diagnostic on the 17-base x30 "
            "subset, not a ceiling estimated on the 33-base main comparison."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "condition_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
