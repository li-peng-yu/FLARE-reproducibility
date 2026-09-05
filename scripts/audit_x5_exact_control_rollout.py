#!/usr/bin/env python3
# Audit exact time closure,
# control tensors, and prediction-only handoffs for all 33x5 paths.
"""Audit exact-control x5 rollout construction without running any model."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_skx_x5_same_condition_distribution import (
    DEFAULT_REPEAT_DATASET,
    _complete_test_groups,
    _prepare_multisegment_rollout_condition,
    _write_json,
)
from skyrmion_cfm.config import load_config
from skyrmion_cfm.data.fixed_time import build_fixed_time_datasets


TOLERANCE_NS = 2.0e-4


def _audit_repeat(repeat: dict[str, Any]) -> dict[str, Any]:
    segments = list(repeat["segments"])
    handoffs = list(repeat["handoffs"])
    if len(segments) < 2 or len(handoffs) != len(segments) - 1:
        raise RuntimeError(f"invalid segment/handoff counts: {repeat['run_id']}")
    duration_sum = sum(float(row["duration_ns"]) for row in segments)
    span = float(repeat["absolute_end_ns"]) - float(repeat["absolute_start_ns"])
    if not math.isclose(duration_sum, span, rel_tol=0.0, abs_tol=TOLERANCE_NS):
        raise RuntimeError(
            f"time closure failed for {repeat['run_id']}: {duration_sum} != {span}"
        )
    if any(
        not math.isclose(
            float(row["exact_boundary_gap_ns"]),
            0.0,
            rel_tol=0.0,
            abs_tol=TOLERANCE_NS,
        )
        for row in handoffs
    ):
        raise RuntimeError(f"nonzero exact handoff gap: {repeat['run_id']}")
    driven = [row for row in segments if float(row["drive_fraction"]) > 0.5]
    relaxed = [row for row in segments if float(row["drive_fraction"]) <= 0.5]
    if len(driven) != 1 or len(relaxed) != 1:
        raise RuntimeError(
            f"expected one driven and one relaxed segment: {repeat['run_id']}"
        )
    if float(driven[0]["j_field_max_abs_a_m2"]) <= 1.0:
        raise RuntimeError(f"driven segment has zero current: {repeat['run_id']}")
    if float(relaxed[0]["j_field_max_abs_a_m2"]) > 1.0:
        raise RuntimeError(f"relaxed segment has nonzero current: {repeat['run_id']}")
    return {
        "run_id": repeat["run_id"],
        "anchor_ns": repeat["absolute_start_ns"],
        "final_ns": repeat["absolute_end_ns"],
        "segment_duration_sum_ns": duration_sum,
        "absolute_span_ns": span,
        "segments": segments,
        "handoffs": handoffs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-groups", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("data", {}).setdefault("memmap", {})["auto_build"] = False
    cfg["data"]["memmap"]["force_rebuild"] = False
    cfg.setdefault("train", {})["num_workers"] = 0
    cfg["train"]["persistent_workers"] = False
    _, _, dataset = build_fixed_time_datasets(cfg, build_splits={"test"})
    if dataset is None:
        raise RuntimeError("failed to build x5 test split")
    groups = _complete_test_groups(
        dataset,
        repeat_dataset=DEFAULT_REPEAT_DATASET,
        repeats_per_group=5,
    )
    if args.max_groups > 0:
        groups = dict(list(groups.items())[: args.max_groups])

    audited: list[dict[str, Any]] = []
    for base, record_indices in groups.items():
        condition_id = f"base{base}_exact_control_drive_to_post_relax"
        prepared = _prepare_multisegment_rollout_condition(
            dataset,
            base=base,
            record_indices=record_indices,
            condition_dir=args.output_root / "conditions" / condition_id,
        )
        metadata = prepared["metadata"]
        if metadata["reference_mode"] != (
            "exact_control_multisegment_rollout_endpoint"
        ):
            raise RuntimeError(f"wrong reference mode for {condition_id}")
        audited.extend(_audit_repeat(row) for row in metadata["repeats"])

    payload = {
        "status": "complete",
        "protocol": "exact control boundaries with prediction-only handoff",
        "groups": len(groups),
        "trajectories": len(audited),
        "time_tolerance_ns": TOLERANCE_NS,
        "all_time_closures_pass": True,
        "all_exact_handoff_gaps_zero": True,
        "all_control_tensors_match_protocol": True,
        "teacher_forcing_after_first_segment": False,
        "example": audited[0],
    }
    _write_json(args.output_root / "audit.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
