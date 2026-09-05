#!/usr/bin/env python3
"""Validate and summarize exact-control rollout quality and paper timing."""

# Quality follows the exact driven control and predicted handoff into 3.5-ns
# relaxation. Timing uses the separately measured fixed 2+3-ns workload.

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METHODS = (
    "flare",
    # The canonical Cartesian-CFM
    # checkpoint is evaluated on the same complete path as the main methods.
    "cartesian_cfm",
    # Direct U-Net is evaluated on
    # the same two-segment quality/fair-score task.  Its representative 5-ns
    # batch timing was not part of the frozen eight-method speed sweep.
    "direct_unet",
    "poseidon_t",
    "cno_fm",
    "dpot_ti",
    "mpp_avit_ti",
    "pdearena_unet",
    "le_pde",
    "neuralmag_x5",
)
TIMED_METHODS = tuple(
    method
    for method in METHODS
    if method not in {"cartesian_cfm", "direct_unet"}
)
FASTEST_BATCH = {
    "flare": 128,
    "poseidon_t": 2048,
    "cno_fm": 512,
    "dpot_ti": 64,
    "mpp_avit_ti": 128,
    "pdearena_unet": 128,
    "le_pde": 2048,
    "neuralmag_x5": 128,
}
REFERENCE_MODE = "exact_control_multisegment_rollout_endpoint"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(summary: dict[str, Any], key: str) -> tuple[float, list[float]]:
    combined = summary[key]["combined"]
    if key == "score":
        return (
            float(combined["geometric_mean_score"]),
            [float(value) for value in combined["geometric_mean_bootstrap_95ci"]],
        )
    return (
        float(combined["mean"]),
        [float(value) for value in combined["bootstrap_95ci"]],
    )


def _condition_rows(run_dir: Path, method: str) -> list[dict[str, Any]]:
    path = run_dir / "summary/condition_scores.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 33:
        raise RuntimeError(f"{method}: expected 33 condition-score rows, found {len(rows)}")
    if len({str(row["condition_id"]) for row in rows}) != len(rows):
        raise RuntimeError(f"{method}: duplicate condition IDs in {path}")
    return rows


def _named_paths(values: list[str], *, option: str) -> dict[str, Path]:
    """Parse repeatable METHOD=PATH command-line overrides."""
    parsed: dict[str, Path] = {}
    for value in values:
        try:
            method, raw_path = value.split("=", 1)
        except ValueError as error:
            raise ValueError(f"{option} expects METHOD=PATH, got {value!r}") from error
        if method not in METHODS:
            raise ValueError(f"{option}: unknown method {method!r}")
        if method in parsed:
            raise ValueError(f"{option}: duplicate method {method!r}")
        parsed[method] = Path(raw_path).resolve()
    return parsed


def _named_fair_sources(
    values: list[str], *, option: str
) -> dict[str, tuple[str, Path]]:
    """Parse repeatable METHOD=SUMMARY_KEY=RUN_SUMMARY command-line overrides."""
    parsed: dict[str, tuple[str, Path]] = {}
    for value in values:
        parts = value.split("=", 2)
        if len(parts) != 3:
            raise ValueError(
                f"{option} expects METHOD=SUMMARY_KEY=RUN_SUMMARY, got {value!r}"
            )
        method, summary_key, raw_path = parts
        if method not in METHODS:
            raise ValueError(f"{option}: unknown method {method!r}")
        if method in parsed:
            raise ValueError(f"{option}: duplicate method {method!r}")
        parsed[method] = (summary_key, Path(raw_path).resolve())
    return parsed


def _paired_benefit_bootstrap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    key: str,
    geometric: bool,
    higher_is_better: bool,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    left_by_id = {str(row["condition_id"]): row for row in left}
    right_by_id = {str(row["condition_id"]): row for row in right}
    if set(left_by_id) != set(right_by_id):
        raise RuntimeError("paired comparison requires identical condition IDs")

    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for condition_id in sorted(left_by_id):
        lrow = left_by_id[condition_id]
        rrow = right_by_id[condition_id]
        if str(lrow["base_id"]).zfill(4) != str(rrow["base_id"]).zfill(4):
            raise RuntimeError(f"base mismatch for {condition_id}")
        left_value = float(lrow[key])
        right_value = float(rrow[key])
        if left_value <= 0.0 or right_value <= 0.0:
            raise RuntimeError(f"non-positive value for paired ratio: {condition_id}")
        grouped[str(lrow["base_id"]).zfill(4)].append((left_value, right_value))
    groups = sorted(grouped)

    def aggregate(pairs: list[tuple[float, float]]) -> float:
        left_values = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        right_values = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        if geometric:
            ratio = math.exp(float(np.log(left_values).mean() - np.log(right_values).mean()))
        else:
            ratio = float(left_values.mean() / right_values.mean())
        return ratio - 1.0 if higher_is_better else 1.0 - ratio

    observed_pairs = [pair for group in groups for pair in grouped[group]]
    observed = aggregate(observed_pairs)
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selected = rng.integers(0, len(groups), size=len(groups))
        pairs = [
            pair
            for group_index in selected
            for pair in grouped[groups[int(group_index)]]
        ]
        draws[index] = aggregate(pairs)
    return {
        "relative_benefit": observed,
        "relative_benefit_bootstrap_95ci": np.quantile(
            draws, [0.025, 0.975]
        ).tolist(),
        "bootstrap_probability_flare_better": float(np.mean(draws > 0.0)),
        "bootstrap_unit": "paired physical base group",
        "iterations": iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument(
        "--fixed-two-segment-timing-summary",
        type=Path,
        required=True,
        help="Summary generated from the fixed 2+3-ns timing workflow.",
    )
    parser.add_argument("--fair-energy-root", type=Path)
    parser.add_argument(
        "--quality-override",
        action="append",
        default=[],
        metavar="METHOD=RUN_DIR",
        help=(
            "Use RUN_DIR (containing summary/run_summary.json) instead of "
            "QUALITY_ROOT/METHOD. Repeat once per overridden method."
        ),
    )
    parser.add_argument(
        "--fair-energy-override",
        action="append",
        default=[],
        metavar="METHOD=SUMMARY_KEY=RUN_SUMMARY",
        help=(
            "Read one method's fair-energy row from RUN_SUMMARY['methods']"
            "[SUMMARY_KEY]. Repeat once per overridden method."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    quality_overrides = _named_paths(
        args.quality_override, option="--quality-override"
    )
    fair_overrides = _named_fair_sources(
        args.fair_energy_override, option="--fair-energy-override"
    )

    timing_source = args.fixed_two_segment_timing_summary.resolve()
    fixed_timing = _json(timing_source)
    if fixed_timing.get("status") != "complete":
        raise RuntimeError("fixed two-segment timing summary is incomplete")
    protocol = fixed_timing.get("protocol", {})
    if (
        not math.isclose(float(protocol.get("total_horizon_ns", -1.0)), 5.0)
        or [float(value) for value in protocol.get("segment_durations_ns", [])] != [2.0, 3.0]
        or int(protocol.get("handoff_count", -1)) != 1
    ):
        raise RuntimeError("wrong fixed two-segment timing protocol")
    fixed_timing_by_method = {str(row["method"]): row for row in fixed_timing.get("rows", [])}
    if set(fixed_timing_by_method) != set(TIMED_METHODS):
        raise RuntimeError("fixed timing summary does not contain all methods")
    mumax_representative_ms = float(protocol["mumax_reference_ms"])
    fair: dict[str, Any] = {}
    if args.fair_energy_root is not None:
        fair_summary = _json(args.fair_energy_root / "run_summary.json")
        if fair_summary.get("status") != "complete":
            raise RuntimeError("fair-energy summary is incomplete")
        fair = fair_summary["methods"]

    rows: list[dict[str, Any]] = []
    condition_rows: dict[str, list[dict[str, Any]]] = {}
    horizons: list[float] = []
    for method in METHODS:
        quality_run_dir = quality_overrides.get(method, args.quality_root / method)
        quality_path = quality_run_dir / "summary/run_summary.json"
        summary = _json(quality_path)
        if summary.get("status") != "complete":
            raise RuntimeError(f"{method}: incomplete quality summary")
        if summary.get("reference_mode") != REFERENCE_MODE:
            raise RuntimeError(f"{method}: wrong quality reference mode")
        if int(summary.get("conditions", -1)) != 33:
            raise RuntimeError(f"{method}: expected 33 final rollout conditions")
        if [int(value) for value in summary.get("segments", [])] != [-1]:
            raise RuntimeError(f"{method}: expected composed-path segment id -1")

        if method == "flare":
            for path in sorted(
                (quality_run_dir / "conditions").glob(
                    "base*_exact_control*/condition_metadata.json"
                )
            ):
                metadata = _json(path)
                for repeat in metadata["repeats"]:
                    span = float(repeat["absolute_span_ns"])
                    composed = float(repeat["composed_model_horizon_ns"])
                    if abs(span - composed) > 2.0e-4:
                        raise RuntimeError(f"time closure failed in {path}")
                    horizons.append(span)

        score, score_ci = _metric(summary, "score")
        paired, paired_ci = _metric(summary, "paired_angular_error")
        angular_ed, angular_ed_ci = _metric(
            summary, "angular_energy_distance"
        )
        condition_rows[method] = _condition_rows(quality_run_dir, method)
        timing_row = fixed_timing_by_method.get(method)
        if timing_row is not None:
            batch = int(timing_row["new_fastest_batch"])
            latency_ms = float(timing_row["new_paper_batch_ms_per_output"])
            parameter_count: int | None = int(timing_row["parameter_count"])
        else:
            batch = None
            latency_ms = None
            parameter_count = None
        fair_source: str | None = None
        fair_method = fair.get(method) or {}
        if method in fair_overrides:
            fair_key, fair_path = fair_overrides[method]
            fair_payload = _json(fair_path)
            if fair_payload.get("status") != "complete":
                raise RuntimeError(f"{method}: fair-energy override is incomplete")
            try:
                fair_method = fair_payload["methods"][fair_key]
            except KeyError as error:
                raise RuntimeError(
                    f"{method}: fair-energy key {fair_key!r} is absent from {fair_path}"
                ) from error
            fair_source = f"{fair_path}#methods/{fair_key}"
        elif args.fair_energy_root is not None:
            fair_source = str(
                (args.fair_energy_root / "run_summary.json").resolve()
            ) + f"#methods/{method}"
        fair_combined = fair_method.get("combined", {})
        fair_metric = fair_combined.get("fair_energy_score", {})
        rows.append(
            {
                "method": method,
                "conditions": 33,
                "distribution_score": score,
                "distribution_score_ci": score_ci,
                "paired_angle_deg": paired,
                "paired_angle_deg_ci": paired_ci,
                "angular_energy_distance_deg": angular_ed,
                "angular_energy_distance_deg_ci": angular_ed_ci,
                "fair_energy_score": fair_metric.get("mean"),
                "fair_energy_score_ci": fair_metric.get("bootstrap_95ci"),
                "fastest_paper_batch": batch,
                "fastest_batch_ms_per_output": latency_ms,
                "parameter_count": parameter_count,
                "exact_control_temporal_adapter": summary.get(
                    "exact_control_temporal_adapter",
                    (
                        "one exact-duration endpoint call per control segment"
                        if method == "flare"
                        else None
                    ),
                ),
                "representative_5ns_speedup_over_mumax": (
                    mumax_representative_ms / latency_ms
                    if latency_ms is not None
                    else None
                ),
                "quality_summary": str(quality_path.resolve()),
                "fair_energy_summary": fair_source,
                "timing_summary": str(timing_source),
            }
        )

    if len(horizons) != 165:
        raise RuntimeError(f"expected 165 FLARE reference paths, found {len(horizons)}")

    row_by_method = {str(row["method"]): row for row in rows}
    rankings = {
        "distribution_score": sorted(
            METHODS,
            key=lambda method: float(row_by_method[method]["distribution_score"]),
            reverse=True,
        ),
        "paired_angle_deg": sorted(
            METHODS,
            key=lambda method: float(row_by_method[method]["paired_angle_deg"]),
        ),
        "angular_energy_distance_deg": sorted(
            METHODS,
            key=lambda method: float(
                row_by_method[method]["angular_energy_distance_deg"]
            ),
        ),
        "fastest_batch_ms_per_output": sorted(
            TIMED_METHODS,
            key=lambda method: float(
                row_by_method[method]["fastest_batch_ms_per_output"]
            ),
        ),
    }
    if fair and all(
        row_by_method[method]["fair_energy_score"] is not None for method in METHODS
    ):
        rankings["fair_energy_score"] = sorted(
            METHODS,
            key=lambda method: float(row_by_method[method]["fair_energy_score"]),
        )

    comparison_specs = {
        "distribution_score": {
            "csv_key": "symmetric_ratio_score",
            "geometric": True,
            "higher_is_better": True,
        },
        "paired_angle_deg": {
            "csv_key": "paired_model_mumax_mean_deg",
            "geometric": False,
            "higher_is_better": False,
        },
        "angular_energy_distance_deg": {
            "csv_key": "angular_energy_distance_deg",
            "geometric": False,
            "higher_is_better": False,
        },
    }
    flare_vs_strongest_baseline: dict[str, Any] = {}
    for metric_index, (metric, spec) in enumerate(comparison_specs.items()):
        baseline = next(
            method for method in rankings[metric] if method != "flare"
        )
        flare_vs_strongest_baseline[metric] = {
            "baseline": baseline,
            **_paired_benefit_bootstrap(
                condition_rows["flare"],
                condition_rows[baseline],
                key=str(spec["csv_key"]),
                geometric=bool(spec["geometric"]),
                higher_is_better=bool(spec["higher_is_better"]),
                iterations=50000,
                seed=208296000 + metric_index,
            ),
        }

    flare_vs_all_baselines_distribution_score = {
        method: _paired_benefit_bootstrap(
            condition_rows["flare"],
            condition_rows[method],
            key="symmetric_ratio_score",
            geometric=True,
            higher_is_better=True,
            iterations=50000,
            seed=208297000 + method_index,
        )
        for method_index, method in enumerate(METHODS)
        if method != "flare"
    }

    payload = {
        "status": "complete",
        "reference_mode": REFERENCE_MODE,
        "task": (
            "drive-start saved state -> exact protocol drive-off boundary -> "
            "final post-relax endpoint; prediction-only handoff"
        ),
        "conditions": 33,
        "reference_trajectories": 165,
        "horizon_ns": {
            "minimum": min(horizons),
            "maximum": max(horizons),
            "mean": statistics.fmean(horizons),
        },
        "representative_5ns_timing": {
            "scope": (
                "Common fixed 5-ns execution workload used only to normalize speed; "
                "quality trajectories have nonuniform physical forecast spans."
            ),
            "mumax_batch_one_ms": mumax_representative_ms,
            "mumax_derivation": protocol.get("mumax_reference_description"),
            "source": str(timing_source),
        },
        "rankings": rankings,
        "flare_vs_strongest_baseline": flare_vs_strongest_baseline,
        "flare_vs_all_baselines_distribution_score": (
            flare_vs_all_baselines_distribution_score
        ),
        "simulator_reference_interpretation": {
            "matched_final_endpoint_x30_two_repeat_angle_deg": 30.25,
            "matched_final_endpoint_x30_two_repeat_angle_deg_ci": [23.89, 36.66],
            "matched_final_endpoint_x30_angular_energy_distance_deg": 12.20,
            "matched_final_endpoint_x30_angular_energy_distance_deg_ci": [
                9.62,
                14.86,
            ],
            "scope": (
                "Independent x30 final post-relax endpoints provide an absolute-scale "
                "reference, not a lower bound for paired model error. A strict full-path "
                "oracle would require multiple thermal continuations from the identical "
                "drive-start state and is not present in the current corpus."
            ),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
