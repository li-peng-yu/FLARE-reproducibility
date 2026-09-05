#!/usr/bin/env python3
"""Assemble deterministic x5 checkpoints into a matched-budget seed ensemble."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DIST_ROOT = Path(str(Path(__file__).resolve().parents[1] / "third_party/distribution_score"))
for path in (PROJECT, DIST_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from distribution_score.distance import FORMAL_BLOCKS, FORMAL_SHIFT_RADIUS_PX  # noqa: E402
from distribution_score.same_condition import run as run_same_condition  # noqa: E402
from scripts.evaluate_skx_x5_same_condition_distribution import (  # noqa: E402
    _aggregate,
    _write_json,
)
from scripts.x5_distribution_metrics import (  # noqa: E402
    aggregate_clustered_mean,
    angular_energy_distance_from_files,
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    temporary.replace(path)


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--source-root", action="append", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=209_030_700)
    args = parser.parse_args()
    if len(args.source_root) < 2:
        raise ValueError("a deep ensemble requires at least two --source-root values")

    sources = [path.resolve() for path in args.source_root]
    member_count = len(sources)
    truth_root = args.truth_root.resolve()
    output_root = args.output_root.resolve()
    source_summaries = [_json(root / "summary/run_summary.json") for root in sources]
    if any(summary.get("status") != "complete" for summary in source_summaries):
        raise RuntimeError("one or more source evaluations are incomplete")

    condition_dirs = sorted(
        path
        for path in (truth_root / "conditions").glob("base*_exact_control*")
        if path.is_dir()
    )
    if len(condition_dirs) != 33:
        raise RuntimeError(f"expected 33 truth conditions, found {len(condition_dirs)}")

    rows: list[dict[str, Any]] = []
    member_counts = np.zeros(member_count, dtype=np.int64)
    for condition_position, truth_condition in enumerate(condition_dirs):
        condition_id = truth_condition.name
        metadata = _json(truth_condition / "condition_metadata.json")
        base_id = str(metadata["base_id"])
        arrays = []
        manifests = []
        for source in sources:
            source_condition = source / "conditions" / condition_id
            arrays.append(
                np.asarray(
                    np.load(
                        source_condition / "model_samples/model_samples_f16.npy",
                        allow_pickle=False,
                    ),
                    dtype=np.float16,
                )
            )
            manifests.append(_json(source_condition / "model_samples/manifest.json"))
        if any(array.shape != (5, 3, 256, 256) for array in arrays):
            raise RuntimeError(
                f"{condition_id}: expected {member_count} [5,3,256,256] "
                "source arrays"
            )

        # Five Table-1-budget draws sampled from the training-seed ensemble.
        # For five members this uses each seed exactly once.  For other sizes,
        # the base-dependent rotation balances any unequal weights across
        # conditions without looking at a target or prediction.
        offset = int(base_id) % member_count
        member_assignment = np.asarray(
            [
                (repeat_index + offset) % member_count
                for repeat_index in range(5)
            ],
            dtype=np.int16,
        )
        member_counts += np.bincount(member_assignment, minlength=member_count)
        matched = np.stack(
            [
                arrays[int(member_assignment[repeat_index])][repeat_index]
                for repeat_index in range(5)
            ]
        )

        distribution_condition = output_root / "distribution/conditions" / condition_id
        exact_condition = output_root / "exact_anchor/conditions" / condition_id
        for target in (distribution_condition, exact_condition):
            target.mkdir(parents=True, exist_ok=True)
            _write_json(target / "condition_metadata.json", metadata)

        distribution_model_dir = distribution_condition / "model_samples"
        _save(distribution_model_dir / "model_samples_f16.npy", matched)
        _save(distribution_model_dir / "member_index.npy", member_assignment)
        _write_json(
            distribution_model_dir / "manifest.json",
            {
                "status": "complete",
                "method": args.label,
                "shape": list(matched.shape),
                "reference_mode": metadata["reference_mode"],
                "distribution_semantics": (
                    "five matched-budget draws sampled from a "
                    f"{member_count}-training-seed deep ensemble; member assignment "
                    "is target-independent"
                ),
                "source_roots": sources,
                "source_manifests": manifests,
                "member_assignment": member_assignment.tolist(),
            },
        )

        # Fair energy is conditional on each exact anchor, so retain every
        # independently trained member prediction for each of five anchors.
        exact = np.stack(
            [
                arrays[member_index][repeat_index]
                for repeat_index in range(5)
                for member_index in range(member_count)
            ]
        )
        anchor_assignment = np.repeat(
            np.arange(5, dtype=np.int16), member_count
        )
        exact_model_dir = exact_condition / "model_samples"
        _save(exact_model_dir / "model_samples_f16.npy", exact)
        _save(exact_model_dir / "anchor_repeat_index.npy", anchor_assignment)
        _save(
            exact_model_dir / "member_index.npy",
            np.tile(np.arange(member_count, dtype=np.int16), 5),
        )
        _write_json(
            exact_model_dir / "manifest.json",
            {
                "status": "complete",
                "method": args.label,
                "shape": list(exact.shape),
                "reference_mode": metadata["reference_mode"],
                "distribution_semantics": (
                    f"{member_count} independently trained member forecasts from "
                    "each exact anchor"
                ),
                "draws_per_anchor": member_count,
                "checkpoint_sha256": ";".join(
                    str(item.get("checkpoint_sha256", "unknown")) for item in manifests
                ),
                "source_roots": sources,
            },
        )

        score_dir = distribution_condition / "same_condition_score_4x4_shift16"
        result = run_same_condition(
            argparse.Namespace(
                condition_dir=truth_condition,
                output_dir=score_dir,
                model_samples=distribution_model_dir / "model_samples_f16.npy",
                num_model=5,
                num_mumax=5,
                blocks=FORMAL_BLOCKS,
                shift_radius=FORMAL_SHIFT_RADIUS_PX,
                reference_chunk=5,
                bootstrap=args.bootstrap,
                seed=args.seed + int(base_id) * 1_000,
                device=args.device,
                progress=False,
                skip_auxiliary_texture=True,
            )
        )
        primary = result["primary_patch_shift"]
        energy = angular_energy_distance_from_files(
            distribution_model_dir / "model_samples_f16.npy",
            score_dir / "mumax_targets_f16.npy",
            score_dir / "geometry_mask.npy",
            num_model=5,
            device=args.device,
        )
        rows.append(
            {
                "condition_id": condition_id,
                "base_id": base_id,
                "control_segment_index": -1,
                "segment_role": "drive_to_post_relax",
                "absolute_start_ns": metadata["absolute_start_ns"],
                "absolute_end_ns": metadata["absolute_end_ns"],
                "horizon_ns": metadata["horizon_ns"],
                "symmetric_ratio_score": primary["symmetric_ratio_score"],
                "symmetric_log_ratio_mae": primary["symmetric_log_ratio_mae"],
                "kernel_sigma": primary["sigma"],
                "score_bootstrap_low": result[
                    "primary_symmetric_ratio_score_bootstrap_95ci"
                ][0],
                "score_bootstrap_high": result[
                    "primary_symmetric_ratio_score_bootstrap_95ci"
                ][1],
                **energy,
            }
        )
        print(
            json.dumps(
                {
                    "position": condition_position + 1,
                    "conditions": len(condition_dirs),
                    "condition_id": condition_id,
                    "score": primary["symmetric_ratio_score"],
                }
            ),
            flush=True,
        )

    distribution_summary = {
        "status": "complete",
        "method": args.label,
        "strategy": f"{member_count}_training_seed_deep_ensemble",
        "source_roots": sources,
        "source_summaries": source_summaries,
        "conditions": len(rows),
        "reference_mode": "exact_control_multisegment_rollout_endpoint",
        "model_predictions_per_condition": 5,
        "member_draw_counts": member_counts.tolist(),
        "score": _aggregate(rows, iterations=args.bootstrap, seed=args.seed + 999),
        "angular_energy_distance": aggregate_clustered_mean(
            rows,
            "angular_energy_distance_deg",
            iterations=args.bootstrap,
            seed=args.seed + 1_999,
        ),
        "paired_angular_error": aggregate_clustered_mean(
            rows,
            "paired_model_mumax_mean_deg",
            iterations=args.bootstrap,
            seed=args.seed + 2_999,
        ),
    }
    distribution_summary_dir = output_root / "distribution/summary"
    _csv(distribution_summary_dir / "condition_scores.csv", rows)
    _write_json(distribution_summary_dir / "run_summary.json", distribution_summary)
    _write_json(
        output_root / "exact_anchor/summary/run_summary.json",
        {
            "status": "complete",
            "mode": f"{member_count}_member_exact_anchor_generation_only",
            "method": args.label,
            "conditions": len(rows),
            "draws_per_anchor": member_count,
            "model_predictions_per_condition": 5 * member_count,
            "source_roots": sources,
        },
    )
    print(json.dumps(distribution_summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
