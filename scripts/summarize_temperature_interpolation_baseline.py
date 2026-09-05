#!/usr/bin/env python3
"""Compare an interpolation temperature with paired neighboring ID endpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from summarize_x5_temperature_ood import (
    base_means,
    distribution_base_values,
    paired_bootstrap,
    read_csv,
)


def midpoint(lower: dict[str, float], upper: dict[str, float]) -> dict[str, float]:
    if set(lower) != set(upper):
        raise RuntimeError("lower/upper ID base cohorts differ")
    return {key: 0.5 * (lower[key] + upper[key]) for key in sorted(lower)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lower-root", type=Path, required=True)
    parser.add_argument("--upper-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lower-temp-k", type=float, default=150.0)
    parser.add_argument("--upper-temp-k", type=float, default=300.0)
    parser.add_argument("--target-temp-k", type=float, default=225.0)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    lower_token = str(int(args.lower_temp_k))
    upper_token = str(int(args.upper_temp_k))
    target_token = str(int(args.target_temp_k))
    if not math.isclose(
        args.target_temp_k,
        0.5 * (args.lower_temp_k + args.upper_temp_k),
    ):
        raise ValueError("target temperature must be the midpoint of the ID endpoints")

    lambda_one = lambda row: math.isclose(float(row["lambda"]), 1.0)
    result = {
        "design": {
            "lower_id_temperature_k": args.lower_temp_k,
            "upper_id_temperature_k": args.upper_temp_k,
            "target_unseen_temperature_k": args.target_temp_k,
            "baseline": (
                "per-base arithmetic midpoint of "
                f"{args.lower_temp_k:g}/{args.upper_temp_k:g} K ID metrics"
            ),
            "bootstrap_unit": "matched base condition",
            "bootstrap_iterations": args.bootstrap,
        }
    }

    lower_single = base_means(
        read_csv(args.lower_root / f"single_segment/id{lower_token}.csv"), "ang"
    )
    upper_single = base_means(
        read_csv(args.upper_root / f"single_segment/id{upper_token}.csv"), "ang"
    )
    target_single = base_means(
        read_csv(args.target_root / f"single_segment/ood{target_token}.csv"), "ang"
    )
    result["single_segment_angular_error_deg"] = paired_bootstrap(
        midpoint(lower_single, upper_single),
        target_single,
        iterations=args.bootstrap,
        seed=args.seed + 101,
    )

    lower_handoff = base_means(
        read_csv(args.lower_root / f"handoff/id{lower_token}/handoff_curve.csv"),
        "output_ang",
        predicate=lambda_one,
    )
    upper_handoff = base_means(
        read_csv(args.upper_root / f"handoff/id{upper_token}/handoff_curve.csv"),
        "output_ang",
        predicate=lambda_one,
    )
    target_handoff = base_means(
        read_csv(args.target_root / f"handoff/ood{target_token}/handoff_curve.csv"),
        "output_ang",
        predicate=lambda_one,
    )
    result["autoregressive_handoff_angular_error_deg"] = paired_bootstrap(
        midpoint(lower_handoff, upper_handoff),
        target_handoff,
        iterations=args.bootstrap,
        seed=args.seed + 211,
    )

    paths = {
        "lower": args.lower_root / f"distribution/id{lower_token}/summary/condition_scores.csv",
        "upper": args.upper_root / f"distribution/id{upper_token}/summary/condition_scores.csv",
        "target": args.target_root / f"distribution/ood{target_token}/summary/condition_scores.csv",
    }
    dist_rows = {key: read_csv(path) for key, path in paths.items()}
    for name, column, geometric, lower_is_better, offset in (
        ("distribution_score", "symmetric_ratio_score", True, False, 401),
        ("angular_energy_distance_deg", "angular_energy_distance_deg", False, True, 503),
    ):
        lower = distribution_base_values(
            dist_rows["lower"], column, geometric=geometric
        )
        upper = distribution_base_values(
            dist_rows["upper"], column, geometric=geometric
        )
        target = distribution_base_values(
            dist_rows["target"], column, geometric=geometric
        )
        result[name] = paired_bootstrap(
            midpoint(lower, upper),
            target,
            iterations=args.bootstrap,
            seed=args.seed + offset,
            lower_is_better=lower_is_better,
            geometric=geometric,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
