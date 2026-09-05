#!/usr/bin/env python3
"""Summarize a matched ID versus temperature-OOD evaluation pair."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


BASE_RE = re.compile(r"(?:^|_)base(\d+)(?:_|$)")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def base_id(row: dict[str, Any]) -> str:
    if row.get("base_id") not in (None, ""):
        value = str(row["base_id"])
        match = re.search(r"(\d+)", value)
        if match:
            return f"base{int(match.group(1)):04d}"
    match = BASE_RE.search(str(row.get("run_id", "")))
    if match is None:
        raise RuntimeError(f"cannot extract base id: {row}")
    return f"base{int(match.group(1)):04d}"


def base_means(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if predicate is not None and not predicate(row):
            continue
        grouped[base_id(row)].append(float(row[metric]))
    if not grouped:
        raise RuntimeError(f"no rows for metric {metric}")
    return {
        key: float(np.mean(values, dtype=np.float64))
        for key, values in sorted(grouped.items())
    }


def paired_bootstrap(
    id_values: dict[str, float],
    ood_values: dict[str, float],
    *,
    iterations: int,
    seed: int,
    lower_is_better: bool = True,
    geometric: bool = False,
) -> dict[str, Any]:
    if set(id_values) != set(ood_values):
        raise RuntimeError(
            "ID/OOD base groups differ: "
            f"id_only={sorted(set(id_values) - set(ood_values))} "
            f"ood_only={sorted(set(ood_values) - set(id_values))}"
        )
    bases = sorted(id_values)
    id_array = np.asarray([id_values[key] for key in bases], dtype=np.float64)
    ood_array = np.asarray([ood_values[key] for key in bases], dtype=np.float64)

    def aggregate(values: np.ndarray) -> float:
        if geometric:
            return float(np.exp(np.log(np.clip(values, 1.0e-12, None)).mean()))
        return float(values.mean())

    id_point = aggregate(id_array)
    ood_point = aggregate(ood_array)
    delta_point = (
        ood_point - id_point if lower_is_better else id_point - ood_point
    )
    percent_point = 100.0 * delta_point / max(abs(id_point), 1.0e-12)

    rng = np.random.default_rng(seed)
    id_samples = np.empty(iterations, dtype=np.float64)
    ood_samples = np.empty(iterations, dtype=np.float64)
    delta_samples = np.empty(iterations, dtype=np.float64)
    percent_samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        chosen = rng.integers(0, len(bases), size=len(bases))
        id_value = aggregate(id_array[chosen])
        ood_value = aggregate(ood_array[chosen])
        delta = ood_value - id_value if lower_is_better else id_value - ood_value
        id_samples[index] = id_value
        ood_samples[index] = ood_value
        delta_samples[index] = delta
        percent_samples[index] = 100.0 * delta / max(abs(id_value), 1.0e-12)

    interval = lambda values: [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]
    return {
        "base_condition_clusters": len(bases),
        "bootstrap_unit": "matched base condition",
        "bootstrap_iterations": iterations,
        "aggregation": "geometric_mean" if geometric else "base_weighted_arithmetic_mean",
        "higher_is_better": not lower_is_better,
        "id": {"mean": id_point, "95ci": interval(id_samples)},
        "ood": {"mean": ood_point, "95ci": interval(ood_samples)},
        "absolute_degradation": {
            "value": delta_point,
            "95ci": interval(delta_samples),
            "definition": "OOD - ID" if lower_is_better else "ID - OOD",
        },
        "percent_degradation": {
            "value": percent_point,
            "95ci": interval(percent_samples),
            "definition": "100 * absolute_degradation / abs(ID)",
        },
        "probability_degradation_positive": float(np.mean(delta_samples > 0.0)),
    }


def distribution_base_values(
    rows: list[dict[str, Any]], metric: str, *, geometric: bool
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[f"base{int(row['base_id']):04d}"].append(float(row[metric]))
    if geometric:
        return {
            key: float(np.exp(np.log(np.clip(values, 1.0e-12, None)).mean()))
            for key, values in sorted(grouped.items())
        }
    return {
        key: float(np.mean(values, dtype=np.float64))
        for key, values in sorted(grouped.items())
    }


def fmt_ci(value: dict[str, Any], digits: int = 3) -> str:
    return (
        f"{value['mean']:.{digits}f} "
        f"[{value['95ci'][0]:.{digits}f}, {value['95ci'][1]:.{digits}f}]"
    )


def write_report(path: Path, result: dict[str, Any]) -> None:
    design = result["design"]
    id_temp = design["id_temperature_k"]
    ood_temp = design["ood_temperature_k"]
    lines = [
        "# Stage-1 FLARE zero-shot temperature OOD",
        "",
        "Frozen checkpoint: `core_seed78.pt` (EMA, ODE-10 Heun).",
        f"ID is {id_temp:g} K and OOD is {ood_temp:g} K on the same 36 test-only base conditions.",
        "All confidence intervals use a paired base-condition cluster bootstrap.",
        "",
        "| Evaluation | ID angular error (deg) | OOD angular error (deg) | Absolute degradation (deg) | Percent degradation |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("single_segment_angular_error_deg", "Single segment"),
        ("autoregressive_handoff_angular_error_deg", "Autoregressive handoff"),
    ):
        item = result[key]
        delta = item["absolute_degradation"]
        percent = item["percent_degradation"]
        lines.append(
            f"| {label} | {fmt_ci(item['id'])} | {fmt_ci(item['ood'])} | "
            f"{delta['value']:.3f} [{delta['95ci'][0]:.3f}, {delta['95ci'][1]:.3f}] | "
            f"{percent['value']:.1f}% [{percent['95ci'][0]:.1f}%, {percent['95ci'][1]:.1f}%] |"
        )
    if "distribution_score" in result:
        score = result["distribution_score"]
        energy = result["angular_energy_distance_deg"]
        lines.extend(
            [
                "",
                "| Distribution metric | ID | OOD | Absolute degradation | Percent degradation |",
                "|---|---:|---:|---:|---:|",
                (
                    f"| Distribution Score (higher is better) | {fmt_ci(score['id'])} | "
                    f"{fmt_ci(score['ood'])} | {score['absolute_degradation']['value']:.3f} "
                    f"[{score['absolute_degradation']['95ci'][0]:.3f}, "
                    f"{score['absolute_degradation']['95ci'][1]:.3f}] | "
                    f"{score['percent_degradation']['value']:.1f}% "
                    f"[{score['percent_degradation']['95ci'][0]:.1f}%, "
                    f"{score['percent_degradation']['95ci'][1]:.1f}%] |"
                ),
                (
                    f"| Angular energy distance (deg; lower is better) | {fmt_ci(energy['id'])} | "
                    f"{fmt_ci(energy['ood'])} | {energy['absolute_degradation']['value']:.3f} "
                    f"[{energy['absolute_degradation']['95ci'][0]:.3f}, "
                    f"{energy['absolute_degradation']['95ci'][1]:.3f}] | "
                    f"{energy['percent_degradation']['value']:.1f}% "
                    f"[{energy['percent_degradation']['95ci'][0]:.1f}%, "
                    f"{energy['percent_degradation']['95ci'][1]:.1f}%] |"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "Absolute degradation is `OOD - ID` for errors/distances and `ID - OOD` for Distribution Score.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/temperature_ood_400k_20260901"),
    )
    parser.add_argument(
        "--id-root",
        type=Path,
        default=None,
        help="optional root containing the ID outputs; defaults to --input-root",
    )
    parser.add_argument(
        "--ood-root",
        type=Path,
        default=None,
        help="optional root containing the OOD outputs; defaults to --input-root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/temperature_ood_400k_20260901/summary"),
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--id-temp-k", type=float, default=300.0)
    parser.add_argument("--ood-temp-k", type=float, default=400.0)
    parser.add_argument("--audit", type=Path, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    id_token = str(int(args.id_temp_k))
    ood_token = str(int(args.ood_temp_k))
    id_root = args.id_root or args.input_root
    ood_root = args.ood_root or args.input_root

    single_id = read_csv(id_root / f"single_segment/id{id_token}.csv")
    single_ood = read_csv(ood_root / f"single_segment/ood{ood_token}.csv")
    handoff_id = read_csv(id_root / f"handoff/id{id_token}/handoff_curve.csv")
    handoff_ood = read_csv(ood_root / f"handoff/ood{ood_token}/handoff_curve.csv")

    result: dict[str, Any] = {
        "design": {
            "id_temperature_k": args.id_temp_k,
            "ood_temperature_k": args.ood_temp_k,
            "matched_test_only_bases": 36,
            "mumax_repeats_per_base": 5,
            "checkpoint_state": "ema",
            "ode_steps": 10,
            "sampler": "heun",
        }
    }
    result["single_segment_angular_error_deg"] = paired_bootstrap(
        base_means(single_id, "ang"),
        base_means(single_ood, "ang"),
        iterations=args.bootstrap,
        seed=args.seed + 101,
    )
    lambda_one = lambda row: math.isclose(float(row["lambda"]), 1.0)
    result["autoregressive_handoff_angular_error_deg"] = paired_bootstrap(
        base_means(handoff_id, "output_ang", predicate=lambda_one),
        base_means(handoff_ood, "output_ang", predicate=lambda_one),
        iterations=args.bootstrap,
        seed=args.seed + 211,
    )
    lambda_zero = lambda row: math.isclose(float(row["lambda"]), 0.0)
    result["ground_truth_boundary_handoff_angular_error_deg"] = paired_bootstrap(
        base_means(handoff_id, "output_ang", predicate=lambda_zero),
        base_means(handoff_ood, "output_ang", predicate=lambda_zero),
        iterations=args.bootstrap,
        seed=args.seed + 307,
    )

    id_distribution = id_root / f"distribution/id{id_token}/summary/condition_scores.csv"
    ood_distribution = ood_root / f"distribution/ood{ood_token}/summary/condition_scores.csv"
    if id_distribution.is_file() and ood_distribution.is_file():
        dist_id = read_csv(id_distribution)
        dist_ood = read_csv(ood_distribution)
        result["distribution_score"] = paired_bootstrap(
            distribution_base_values(
                dist_id, "symmetric_ratio_score", geometric=True
            ),
            distribution_base_values(
                dist_ood, "symmetric_ratio_score", geometric=True
            ),
            iterations=args.bootstrap,
            seed=args.seed + 401,
            lower_is_better=False,
            geometric=True,
        )
        result["angular_energy_distance_deg"] = paired_bootstrap(
            distribution_base_values(
                dist_id, "angular_energy_distance_deg", geometric=False
            ),
            distribution_base_values(
                dist_ood, "angular_energy_distance_deg", geometric=False
            ),
            iterations=args.bootstrap,
            seed=args.seed + 503,
        )

    audit_path = args.audit
    if audit_path is not None and audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        result["leakage_and_completion_audit"] = {
            key: audit[key]
            for key in (
                "status",
                "checkpoint_sha256",
                "checkpoint_step",
                "checkpoint_embedded_dataset_roots",
                "checkpoint_training_temperature_support_k",
                "target_temperature_k",
                "target_temperature_absent_from_checkpoint_train_and_validation",
                "selected_bases_absent_from_checkpoint_train_and_validation",
                "all_new_seeds_unique",
                "no_300k_thermalized_checkpoint_reused",
                "base_conditions",
                "branches",
            )
        }

    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir / "REPORT.md", result)

    table_rows: list[dict[str, Any]] = []
    for name, item in result.items():
        if not isinstance(item, dict) or "absolute_degradation" not in item:
            continue
        table_rows.append(
            {
                "evaluation": name,
                "id": item["id"]["mean"],
                "id_ci95_low": item["id"]["95ci"][0],
                "id_ci95_high": item["id"]["95ci"][1],
                "ood": item["ood"]["mean"],
                "ood_ci95_low": item["ood"]["95ci"][0],
                "ood_ci95_high": item["ood"]["95ci"][1],
                "absolute_degradation": item["absolute_degradation"]["value"],
                "absolute_ci95_low": item["absolute_degradation"]["95ci"][0],
                "absolute_ci95_high": item["absolute_degradation"]["95ci"][1],
                "percent_degradation": item["percent_degradation"]["value"],
                "percent_ci95_low": item["percent_degradation"]["95ci"][0],
                "percent_ci95_high": item["percent_degradation"]["95ci"][1],
                "base_condition_clusters": item["base_condition_clusters"],
            }
        )
    with (args.output_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
