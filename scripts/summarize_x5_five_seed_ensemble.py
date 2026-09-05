#!/usr/bin/env python3
"""Consolidate five-seed deterministic ensembles and 3-of-5 sensitivity."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
METHODS = ("direct_unet", "cno_fm", "dpot_ti", "poseidon_t", "pdearena_unet")
SEEDS = (78, 79, 80, 81, 82)
T95_DF4 = 2.7764451051977987
QUALITY_KEYS = (
    "distribution_score",
    "angular_energy_distance_deg",
    "paired_angle_deg",
)
ENSEMBLE_KEYS = QUALITY_KEYS + (
    "fair_energy_score",
    "ensemble_spread_rms_chord",
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        raise RuntimeError(f"incomplete result: {path}")
    return value


def _quality(summary: dict[str, Any]) -> dict[str, Any]:
    score = summary["score"]["combined"]
    angular = summary["angular_energy_distance"]["combined"]
    paired = summary["paired_angular_error"]["combined"]
    return {
        "distribution_score": float(score["geometric_mean_score"]),
        "distribution_score_ci": score["geometric_mean_bootstrap_95ci"],
        "angular_energy_distance_deg": float(angular["mean"]),
        "angular_energy_distance_deg_ci": angular["bootstrap_95ci"],
        "paired_angle_deg": float(paired["mean"]),
        "paired_angle_deg_ci": paired["bootstrap_95ci"],
    }


def _fair(summary: dict[str, Any], label: str) -> dict[str, Any]:
    combined = summary["methods"][label]["combined"]
    score = combined["fair_energy_score"]
    spread = combined["ensemble_spread_rms_chord"]
    return {
        "fair_energy_score": float(score["mean"]),
        "fair_energy_score_ci": score["bootstrap_95ci"],
        "ensemble_spread_rms_chord": float(spread["mean"]),
        "ensemble_spread_rms_chord_ci": spread["bootstrap_95ci"],
    }


def _source_root(root: Path, table1: Path, direct_old: Path, method: str, seed: int) -> Path:
    if method == "direct_unet":
        if seed <= 80:
            return direct_old / f"rotmse_s{seed}"
        return root / "per_seed" / f"direct_unet_s{seed}"
    if seed == 78:
        return table1 / method
    return root / "per_seed" / f"{method}_s{seed}"


def _sample_summary(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {"replicates": len(rows)}
    for key in keys:
        values = [float(row[key]) for row in rows]
        mean = statistics.fmean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        half_width = T95_DF4 * sd / math.sqrt(len(values)) if len(values) == 5 else None
        result[key] = {
            "mean": mean,
            "sample_sd": sd,
            "min": min(values),
            "max": max(values),
            "seed_t_95ci": (
                [mean - half_width, mean + half_width]
                if half_width is not None
                else None
            ),
        }
    return result


def _delta(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(candidate[key]) - float(reference[key])
        for key in ENSEMBLE_KEYS
        if key in candidate and key in reference
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT / "outputs/x5_deterministic_stochasticity_20260903",
    )
    parser.add_argument(
        "--table1-root",
        type=Path,
        default=(
            PROJECT
            / "outputs/skx_bt_1000base_x5/paper_revision_20260830/"
            "exact_control_rollout_distribution"
        ),
    )
    parser.add_argument(
        "--direct-old-root",
        type=Path,
        default=(
            PROJECT
            / "outputs/skx_bt_1000base_x5/paper_revision_20260902/"
            "flow_necessity_rotation_mse/distribution"
        ),
    )
    parser.add_argument(
        "--table1-summary",
        type=Path,
        default=PROJECT / "reports/x5_exact_control_rollout_seed78_20260903.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "reports/x5_five_seed_ensemble_20260904.json",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    table1_root = args.table1_root.resolve()
    direct_old = args.direct_old_root.resolve()
    table1_summary = _json(args.table1_summary.resolve())
    table1_rows = {str(row["method"]): row for row in table1_summary["rows"]}

    csv_rows: list[dict[str, Any]] = []
    methods: dict[str, Any] = {}
    subset_tags = ["_".join(map(str, seeds)) for seeds in itertools.combinations(SEEDS, 3)]

    for method in METHODS:
        individual: list[dict[str, Any]] = []
        for seed in SEEDS:
            source = _source_root(root, table1_root, direct_old, method, seed)
            quality = _quality(_json(source / "summary/run_summary.json"))
            row = {
                "method": method,
                "strategy": "single_training_seed",
                "seeds": str(seed),
                "source_root": str(source),
                **quality,
            }
            individual.append(row)
            csv_rows.append(row)

        seed_summary = _sample_summary(individual, QUALITY_KEYS)

        deep3_root = root / "deep_ensemble" / method
        deep3_label = f"{method}_deep3"
        deep3 = {
            "method": method,
            "strategy": "deep_ensemble_3seed_original",
            "seeds": "78;79;80",
            **_quality(_json(deep3_root / "distribution/summary/run_summary.json")),
            **_fair(_json(deep3_root / "fair_energy/run_summary.json"), deep3_label),
        }
        csv_rows.append(deep3)

        deep5_root = root / "deep_ensemble5" / method
        deep5_label = f"{method}_deep5"
        deep5 = {
            "method": method,
            "strategy": "deep_ensemble_5seed_official",
            "seeds": ";".join(map(str, SEEDS)),
            **_quality(_json(deep5_root / "distribution/summary/run_summary.json")),
            **_fair(_json(deep5_root / "fair_energy/run_summary.json"), deep5_label),
        }
        csv_rows.append(deep5)

        subsets: list[dict[str, Any]] = []
        for tag in subset_tags:
            subset_root = root / "deep_ensemble3_subsets" / method / tag
            label = f"{method}_deep3_{tag}"
            subset = {
                "method": method,
                "strategy": "deep_ensemble_3_of_5_subset",
                "seeds": tag.replace("_", ";"),
                **_quality(_json(subset_root / "distribution/run_summary.json")),
                **_fair(_json(subset_root / "fair_energy/run_summary.json"), label),
            }
            subsets.append(subset)
            csv_rows.append(subset)

        subset_summary = _sample_summary(subsets, ENSEMBLE_KEYS)
        table1_seed78 = table1_rows[method]
        flare = table1_rows["flare"]
        methods[method] = {
            "individual_seeds": individual,
            "individual_seed_summary": seed_summary,
            "original_3seed_ensemble": deep3,
            "official_5seed_ensemble": deep5,
            "three_of_five_subsets": subsets,
            "three_of_five_subset_summary": subset_summary,
            "deltas": {
                "five_seed_minus_original_three_seed": _delta(deep5, deep3),
                "five_seed_minus_table1_seed78": _delta(deep5, table1_seed78),
                "five_seed_minus_flare": _delta(deep5, flare),
            },
        }

    payload = {
        "status": "complete",
        "protocol": {
            "seeds": list(SEEDS),
            "official_comparison": (
                "5-vs-5: each deterministic deep ensemble has five independently "
                "trained members and each exact anchor uses one prediction per member; "
                "FLARE uses five stochastic draws"
            ),
            "quality_budget": (
                "five outputs per condition with target-independent member assignment; "
                "for five seeds every member is used exactly once"
            ),
            "fair_energy": "five unique training-seed members per exact MuMax anchor",
            "seed_uncertainty": (
                "mean, sample SD, range, and Student-t 95% CI across five training seeds"
            ),
            "subset_sensitivity": "all C(5,3)=10 unique three-member subsets",
            "base_cluster_bootstrap": (
                "each reported evaluation CI resamples base clusters and keeps all "
                "conditions/anchors within a base together"
            ),
        },
        "table1_flare": table1_rows["flare"],
        "methods": methods,
        "rows": csv_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    csv_path = args.output.with_suffix(".csv")
    fieldnames: list[str] = []
    for row in csv_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(
                {
                    key: (
                        ";".join(str(item) for item in value)
                        if isinstance(value, list)
                        else value
                    )
                    for key, value in row.items()
                }
            )
    print(json.dumps({"output": str(args.output), "rows": len(csv_rows)}, indent=2))


if __name__ == "__main__":
    main()
