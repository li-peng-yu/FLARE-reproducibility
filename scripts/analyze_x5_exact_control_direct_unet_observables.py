#!/usr/bin/env python3
"""Compare FLARE and Direct U-Net observables on exact-control complete paths."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.analyze_x5_distribution_sensitivity_observables import (
        _observables,
        _w1_equal_samples,
    )
except ModuleNotFoundError:
    from analyze_x5_distribution_sensitivity_observables import (
        _observables,
        _w1_equal_samples,
    )


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    PROJECT
    / "outputs/skx_bt_1000base_x5/paper_revision_20260830/"
    "exact_control_rollout_distribution"
)
DEFAULT_OUTPUT = (
    PROJECT
    / "outputs/skx_bt_1000base_x5/paper_revision_20260830/"
    "exact_control_rollout_direct_unet_observables"
)
METHODS = ("flare", "direct_unet")
METRICS = (
    "topological_charge_w1",
    "mean_mz_w1",
    "exchange_texture_energy_w1",
)


def _validate_root(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for method in METHODS:
        method_root = root / method
        summary_path = method_root / "summary/run_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("reference_mode") != "exact_control_multisegment_rollout_endpoint":
            raise RuntimeError(f"wrong reference mode for {method}: {summary_path}")
        if summary.get("multisegment_rollout") is not True and summary.get("rollout") is not True:
            raise RuntimeError(f"rollout is not enabled for {method}: {summary_path}")
        if int(summary.get("conditions", -1)) != 33:
            raise RuntimeError(f"expected 33 conditions for {method}: {summary_path}")
        found[method] = method_root
    return found


def _conditions(method_root: Path) -> dict[str, Path]:
    result = {
        path.name: path
        for path in (method_root / "conditions").iterdir()
        if path.is_dir()
        and (path / "condition_metadata.json").is_file()
        and (path / "model_samples/model_samples_f16.npy").is_file()
        and (path / "same_condition_score_4x4_shift16/mumax_targets_f16.npy").is_file()
    }
    if len(result) != 33:
        raise RuntimeError(f"expected 33 complete conditions under {method_root}, found {len(result)}")
    return result


def _rows(method_roots: dict[str, Path]) -> dict[str, list[dict[str, Any]]]:
    conditions_by_method = {
        method: _conditions(method_root) for method, method_root in method_roots.items()
    }
    ids = set(conditions_by_method["flare"])
    if any(set(conditions) != ids for conditions in conditions_by_method.values()):
        raise RuntimeError("FLARE and Direct U-Net condition identifiers do not match")

    rows: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for condition_id in sorted(ids):
        truth_condition = conditions_by_method["flare"][condition_id]
        score_root = truth_condition / "same_condition_score_4x4_shift16"
        mask = np.load(score_root / "geometry_mask.npy", mmap_mode="r")
        truth = _observables(
            np.load(score_root / "mumax_targets_f16.npy", mmap_mode="r"), mask
        )
        for method in METHODS:
            condition = conditions_by_method[method][condition_id]
            metadata = json.loads(
                (condition / "condition_metadata.json").read_text(encoding="utf-8")
            )
            samples = np.load(
                condition / "model_samples/model_samples_f16.npy", mmap_mode="r"
            )
            if samples.shape[0] != 5:
                raise RuntimeError(
                    f"expected five model endpoints for {method}/{condition_id}, "
                    f"found {samples.shape[0]}"
                )
            model = _observables(samples, mask)
            row: dict[str, Any] = {
                "method": method,
                "condition_id": condition_id,
                "base_id": str(metadata["base_id"]).zfill(4),
            }
            for observable in truth:
                row[f"{observable}_w1"] = _w1_equal_samples(
                    model[observable], truth[observable]
                )
            rows[method].append(row)
    return rows


def _bootstrap(
    rows: dict[str, list[dict[str, Any]]], *, iterations: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    count = len(rows["flare"])
    indices = rng.integers(0, count, size=(iterations, count))
    summary: dict[str, Any] = {}
    for metric in METRICS:
        flare = np.asarray([float(row[metric]) for row in rows["flare"]])
        direct = np.asarray([float(row[metric]) for row in rows["direct_unet"]])
        flare_boot = flare[indices].mean(axis=1)
        direct_boot = direct[indices].mean(axis=1)
        difference_boot = direct_boot - flare_boot
        summary[metric] = {
            "flare": {
                "value": float(flare.mean()),
                "bootstrap_95ci": [
                    float(np.quantile(flare_boot, 0.025)),
                    float(np.quantile(flare_boot, 0.975)),
                ],
            },
            "direct_unet": {
                "value": float(direct.mean()),
                "bootstrap_95ci": [
                    float(np.quantile(direct_boot, 0.025)),
                    float(np.quantile(direct_boot, 0.975)),
                ],
            },
            "direct_minus_flare": {
                "value": float((direct - flare).mean()),
                "bootstrap_95ci": [
                    float(np.quantile(difference_boot, 0.025)),
                    float(np.quantile(difference_boot, 0.975)),
                ],
                "probability_direct_worse": float(np.mean(difference_boot > 0.0)),
            },
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    method_roots = _validate_root(args.root.resolve())
    rows = _rows(method_roots)
    summary = {
        "status": "complete",
        "reference_mode": "exact_control_multisegment_rollout_endpoint",
        "conditions": 33,
        "model_endpoints_per_condition": 5,
        "bootstrap": {
            "unit": "paired physical base condition",
            "iterations": args.bootstrap,
            "seed": args.seed,
        },
        "metrics": _bootstrap(rows, iterations=args.bootstrap, seed=args.seed),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "per_condition.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "condition_id", "base_id", *METRICS],
        )
        writer.writeheader()
        for method in METHODS:
            writer.writerows(rows[method])
    print(args.output)


if __name__ == "__main__":
    main()
