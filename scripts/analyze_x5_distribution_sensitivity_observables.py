#!/usr/bin/env python3
"""Paper audit: Distribution-Score sensitivity and stochastic observables.

This script is intentionally read-only with respect to saved model samples and
hand-edited figures.  It consumes the frozen five-versus-five endpoint arrays
and writes numerical JSON/CSV/Markdown summaries only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_MAGFLOW = (
    PROJECT
    / "graph/results/x5_distribution_pareto_20260822/"
    "formal_standard_prior_n5_v020/scfm_stage1"
)
DEFAULT_BASELINES = (
    PROJECT
    / "graph/results/x5_distribution_pareto_20260816/"
    "formal_33groups_50k_v020"
)
DEFAULT_NEURALMAG = (
    PROJECT
    / "graph/results/x5_distribution_pareto_20260816/neuralmag_x5/"
    "formal_33groups_50k_v020"
)
DEFAULT_OUTPUT = (
    PROJECT
    / "outputs/skx_bt_1000base_x5/paper_formal_20260822/"
    "distribution_sensitivity_observables"
)


def _method_roots(
    magflow: Path, baselines: Path, neuralmag: Path
) -> dict[str, Path]:
    return {
        "MagFlow": magflow,
        "Poseidon-T": baselines / "poseidon_t",
        "CNO-FM": baselines / "cno_fm",
        "DPOT-Ti": baselines / "dpot_ti",
        "MPP-AViT-Ti": baselines / "mpp_avit_ti",
        "PDEArena U-Net": baselines / "pdearena_unet",
        "LE-PDE": baselines / "le_pde",
        "NeuralMAG-x5": neuralmag,
    }


def _condition_dirs(root: Path) -> dict[str, Path]:
    conditions = root / "conditions"
    found = {
        path.name: path
        for path in conditions.iterdir()
        if path.is_dir()
        and (path / "condition_metadata.json").is_file()
        and (path / "same_condition_score_4x4_shift16/scores.json").is_file()
    }
    if len(found) != 66:
        raise RuntimeError(f"expected 66 complete conditions under {root}, found {len(found)}")
    return found


def _base_id(metadata: dict[str, Any]) -> str:
    return str(metadata["base_id"]).zfill(4)


def _cluster_bootstrap(
    rows: list[dict[str, Any]],
    key: str,
    *,
    geometric: bool,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[key])
        if geometric and value <= 0.0:
            raise ValueError(f"{key} must be positive for a geometric mean")
        grouped[str(row["base_id"])].append(math.log(value) if geometric else value)
    groups = sorted(grouped)
    if not groups:
        raise ValueError("cannot aggregate no rows")
    observed_flat = np.asarray(
        [value for group in groups for value in grouped[group]], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selected = rng.integers(0, len(groups), size=len(groups))
        flat = np.asarray(
            [value for group_index in selected for value in grouped[groups[int(group_index)]]],
            dtype=np.float64,
        )
        draws[index] = flat.mean()
    if geometric:
        observed = float(np.exp(observed_flat.mean()))
        draws = np.exp(draws)
    else:
        observed = float(observed_flat.mean())
    return {
        "value": observed,
        "bootstrap_95ci": np.quantile(draws, [0.025, 0.975]).tolist(),
        "conditions": len(rows),
        "base_groups": len(groups),
        "aggregation": "equal-condition geometric mean" if geometric else "equal-condition mean",
        "bootstrap_unit": "base group, with both segments retained",
    }


def _paired_ratio_bootstrap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    key: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    left_by_condition = {str(row["condition_id"]): row for row in left}
    right_by_condition = {str(row["condition_id"]): row for row in right}
    if set(left_by_condition) != set(right_by_condition):
        raise ValueError("paired ratio requires identical condition IDs")
    grouped: dict[str, list[float]] = defaultdict(list)
    for condition_id in sorted(left_by_condition):
        lrow = left_by_condition[condition_id]
        rrow = right_by_condition[condition_id]
        grouped[str(lrow["base_id"])].append(
            math.log(float(lrow[key])) - math.log(float(rrow[key]))
        )
    groups = sorted(grouped)
    observed = np.asarray(
        [value for group in groups for value in grouped[group]], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selected = rng.integers(0, len(groups), size=len(groups))
        values = [
            value
            for group_index in selected
            for value in grouped[groups[int(group_index)]]
        ]
        draws[index] = math.exp(float(np.mean(values)))
    return {
        "ratio": math.exp(float(observed.mean())),
        "bootstrap_95ci": np.quantile(draws, [0.025, 0.975]).tolist(),
        "bootstrap_unit": "paired base group, with both segments retained",
    }


def _load_score_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition_id, condition in sorted(_condition_dirs(root).items()):
        metadata = json.loads((condition / "condition_metadata.json").read_text())
        score = json.loads(
            (condition / "same_condition_score_4x4_shift16/scores.json").read_text()
        )
        rows.append(
            {
                "condition_id": condition_id,
                "base_id": _base_id(metadata),
                "segment_role": str(metadata["segment_role"]),
                "sigma_0.5": float(score["sigma_sensitivity"]["0.5"]["symmetric_ratio_score"]),
                "sigma_1.0": float(score["sigma_sensitivity"]["1.0"]["symmetric_ratio_score"]),
                "sigma_2.0": float(score["sigma_sensitivity"]["2.0"]["symmetric_ratio_score"]),
                "no_shift": float(score["no_shift_control"]["symmetric_ratio_score"]),
            }
        )
    return rows


def analyze_score_sensitivity(
    method_roots: dict[str, Path], *, iterations: int, seed: int
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    all_rows = {method: _load_score_rows(root) for method, root in method_roots.items()}
    settings = ("sigma_0.5", "sigma_1.0", "sigma_2.0", "no_shift")
    aggregate: dict[str, Any] = {"settings": {}, "method_order": list(method_roots)}
    for setting_index, setting in enumerate(settings):
        method_summary = {
            method: _cluster_bootstrap(
                rows,
                setting,
                geometric=True,
                iterations=iterations,
                seed=seed + 1000 * setting_index + method_index,
            )
            for method_index, (method, rows) in enumerate(all_rows.items())
        }
        ranking = sorted(method_summary, key=lambda name: method_summary[name]["value"], reverse=True)
        best_baseline = next(name for name in ranking if name != "MagFlow")
        aggregate["settings"][setting] = {
            "methods": method_summary,
            "ranking": ranking,
            "magflow_vs_best_baseline": {
                "baseline": best_baseline,
                **_paired_ratio_bootstrap(
                    all_rows["MagFlow"],
                    all_rows[best_baseline],
                    setting,
                    iterations=iterations,
                    seed=seed + 9000 + setting_index,
                ),
            },
        }
    aggregate["interpretation"] = (
        "sigma_* rescales the MuMax-only nearest-neighbour bandwidth; no_shift "
        "removes local translation matching while retaining the nominal bandwidth rule."
    )
    return aggregate, all_rows


def _normalize_fields(array: np.ndarray) -> np.ndarray:
    fields = np.asarray(array, dtype=np.float32)
    norm = np.linalg.norm(fields, axis=1, keepdims=True)
    return fields / np.maximum(norm, 1.0e-8)


def _observables(array: np.ndarray, mask_array: np.ndarray) -> dict[str, np.ndarray]:
    fields = _normalize_fields(array)
    mask = np.asarray(mask_array, dtype=np.bool_)
    fields = fields * mask[None, None]
    valid = max(int(mask.sum()), 1)
    mean_mz = (fields[:, 2] * mask).sum(axis=(-1, -2)) / valid

    # Match the project's open-boundary central difference: replicate padding
    # followed by a centered half-step difference.
    padded_x = np.pad(fields, ((0, 0), (0, 0), (0, 0), (1, 1)), mode="edge")
    padded_y = np.pad(fields, ((0, 0), (0, 0), (1, 1), (0, 0)), mode="edge")
    dmx = 0.5 * (padded_x[..., 2:] - padded_x[..., :-2])
    dmy = 0.5 * (padded_y[..., 2:, :] - padded_y[..., :-2, :])
    density = np.einsum(
        "nchw,nchw->nhw",
        fields,
        np.cross(dmx, dmy, axisa=1, axisb=1, axisc=1),
        optimize=True,
    )
    charge = density.sum(axis=(-1, -2)) / (4.0 * np.pi)

    horizontal_mask = mask[:, 1:] & mask[:, :-1]
    vertical_mask = mask[1:, :] & mask[:-1, :]
    horizontal = 1.0 - (fields[..., 1:] * fields[..., :-1]).sum(axis=1)
    vertical = 1.0 - (fields[..., 1:, :] * fields[..., :-1, :]).sum(axis=1)
    horizontal_energy = (horizontal * horizontal_mask).sum(axis=(-1, -2))
    vertical_energy = (vertical * vertical_mask).sum(axis=(-1, -2))
    bonds = horizontal_mask.sum() + vertical_mask.sum()
    exchange_texture = (horizontal_energy + vertical_energy) / max(int(bonds), 1)
    return {
        "topological_charge": np.asarray(charge, dtype=np.float64),
        "mean_mz": np.asarray(mean_mz, dtype=np.float64),
        "exchange_texture_energy": np.asarray(exchange_texture, dtype=np.float64),
    }


def _w1_equal_samples(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"equal-sample W1 requires matched shapes, got {left.shape}, {right.shape}")
    return float(np.abs(np.sort(left) - np.sort(right)).mean())


def _mean_pairwise_abs(values: np.ndarray) -> float:
    distances = np.abs(values[:, None] - values[None, :])
    indices = np.triu_indices(len(values), k=1)
    return float(distances[indices].mean())


def _load_observable_rows(
    root: Path,
    truth_root: Path,
    truth_cache: dict[str, tuple[dict[str, np.ndarray], np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    conditions = _condition_dirs(root)
    truth_conditions = _condition_dirs(truth_root)
    if set(conditions) != set(truth_conditions):
        raise RuntimeError(f"condition mismatch between {root} and {truth_root}")
    for condition_id, condition in sorted(conditions.items()):
        metadata = json.loads((condition / "condition_metadata.json").read_text())
        score_dir = condition / "same_condition_score_4x4_shift16"
        if condition_id not in truth_cache:
            truth_score_dir = truth_conditions[condition_id] / "same_condition_score_4x4_shift16"
            mask = np.load(truth_score_dir / "geometry_mask.npy", mmap_mode="r")
            truth_array = np.load(truth_score_dir / "mumax_targets_f16.npy", mmap_mode="r")
            truth_cache[condition_id] = (_observables(truth_array, mask), np.asarray(mask))
        truth, cached_mask = truth_cache[condition_id]
        model_array = np.load(condition / "model_samples/model_samples_f16.npy", mmap_mode="r")
        model = _observables(model_array[:5], cached_mask)
        row: dict[str, Any] = {
            "condition_id": condition_id,
            "base_id": _base_id(metadata),
            "segment_role": str(metadata["segment_role"]),
        }
        for observable in sorted(truth):
            row[f"{observable}_w1"] = _w1_equal_samples(model[observable], truth[observable])
            row[f"{observable}_mumax_pairwise"] = _mean_pairwise_abs(truth[observable])
        rows.append(row)
    return rows


def analyze_observables(
    method_roots: dict[str, Path], *, iterations: int, seed: int
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    truth_root = method_roots["MagFlow"]
    truth_cache: dict[str, tuple[dict[str, np.ndarray], np.ndarray]] = {}
    rows_by_method = {
        method: _load_observable_rows(root, truth_root, truth_cache)
        for method, root in method_roots.items()
    }
    observables = ("topological_charge", "mean_mz", "exchange_texture_energy")
    summary: dict[str, Any] = {
        "metric": "condition-wise empirical W1 between five model endpoints and five MuMax3 endpoints",
        "observables": {},
    }
    for observable_index, observable in enumerate(observables):
        key = f"{observable}_w1"
        method_summary = {
            method: _cluster_bootstrap(
                rows,
                key,
                geometric=False,
                iterations=iterations,
                seed=seed + 20000 + observable_index * 1000 + method_index,
            )
            for method_index, (method, rows) in enumerate(rows_by_method.items())
        }
        ranking = sorted(method_summary, key=lambda name: method_summary[name]["value"])
        reference_rows = rows_by_method["MagFlow"]
        reference_dispersion = _cluster_bootstrap(
            reference_rows,
            f"{observable}_mumax_pairwise",
            geometric=False,
            iterations=iterations,
            seed=seed + 30000 + observable_index,
        )
        summary["observables"][observable] = {
            "methods": method_summary,
            "ranking": ranking,
            "mumax_within_condition_mean_pairwise_absolute_difference": reference_dispersion,
        }
    summary["definitions"] = {
        "topological_charge": "signed open-boundary finite-difference Q",
        "mean_mz": "valid-cell spatial mean of the out-of-plane magnetization",
        "exchange_texture_energy": (
            "valid-neighbour mean of 1 - m_i dot m_j; a dimensionless nearest-neighbour "
            "exchange-texture energy proxy"
        ),
    }
    return summary, rows_by_method


def analyze_nonindependent_self_diagnostics(
    magflow_root: Path, *, bootstrap_replicates: int, seed: int
) -> dict[str, Any]:
    """Quantify, and explicitly quarantine, same-five-repeat self comparisons."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for condition_id, condition in sorted(_condition_dirs(magflow_root).items()):
        metadata = json.loads((condition / "condition_metadata.json").read_text())
        score_dir = condition / "same_condition_score_4x4_shift16"
        distances = np.load(score_dir / "truth_self_patch_shift_distance.npy").astype(np.float64)
        score_json = json.loads((score_dir / "scores.json").read_text())
        sigma = float(score_json["primary_patch_shift"]["sigma"])
        kernel = np.exp(-(distances**2) / (2.0 * sigma**2))
        count = len(kernel)
        reference_density = (kernel.sum(axis=1) - np.diag(kernel)) / (count - 1)
        same_density = kernel.mean(axis=1)
        same_score = math.exp(
            -float(
                np.abs(
                    np.log((same_density + 1.0e-12) / (reference_density + 1.0e-12))
                ).mean()
            )
        )
        bootstrap_scores = np.empty(bootstrap_replicates, dtype=np.float64)
        for index in range(bootstrap_replicates):
            sampled = rng.integers(0, count, size=count)
            resampled_density = kernel[:, sampled].mean(axis=1)
            bootstrap_scores[index] = math.exp(
                -float(
                    np.abs(
                        np.log(
                            (resampled_density + 1.0e-12)
                            / (reference_density + 1.0e-12)
                        )
                    ).mean()
                )
            )
        rows.append(
            {
                "condition_id": condition_id,
                "base_id": _base_id(metadata),
                "same_five_including_self": same_score,
                "empirical_bootstrap_mean": float(bootstrap_scores.mean()),
            }
        )
    return {
        "same_five_including_self": _cluster_bootstrap(
            rows,
            "same_five_including_self",
            geometric=True,
            iterations=5000,
            seed=seed + 1,
        ),
        "empirical_bootstrap_5_from_same_5": _cluster_bootstrap(
            rows,
            "empirical_bootstrap_mean",
            geometric=True,
            iterations=5000,
            seed=seed + 2,
        ),
        "paper_use": "excluded",
        "reason": (
            "Both diagnostics reuse the same five MuMax3 fields used to form the leave-one-out "
            "reference density. They are not an independent MuMax3 self-score and must not be "
            "presented as a calibrated null or upper bound."
        ),
    }


def _write_rows(path: Path, rows_by_method: dict[str, list[dict[str, Any]]]) -> None:
    rows = [dict(method=method, **row) for method, rows in rows_by_method.items() for row in rows]
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(
    sensitivity: dict[str, Any], observables: dict[str, Any], self_diag: dict[str, Any]
) -> str:
    lines = [
        "# Frozen x5 distribution audit",
        "",
        "## Distribution Score sensitivity",
        "",
        "| setting | MagFlow | best comparator | MagFlow/comparator | ranking |",
        "|---|---:|---:|---:|---|",
    ]
    for setting, payload in sensitivity["settings"].items():
        comparison = payload["magflow_vs_best_baseline"]
        baseline = comparison["baseline"]
        magflow = payload["methods"]["MagFlow"]["value"]
        baseline_value = payload["methods"][baseline]["value"]
        lines.append(
            f"| {setting} | {magflow:.4f} | {baseline} {baseline_value:.4f} | "
            f"{comparison['ratio']:.4f} | {' > '.join(payload['ranking'])} |"
        )
    lines += [
        "",
        "## Stochastic physical-observable W1",
        "",
        "Lower is better. Values compare five model endpoints with five MuMax3 endpoints per condition.",
        "",
        "| observable | MagFlow | best comparator | ranking |",
        "|---|---:|---:|---|",
    ]
    for observable, payload in observables["observables"].items():
        ranking = payload["ranking"]
        baseline = next(name for name in ranking if name != "MagFlow")
        lines.append(
            f"| {observable} | {payload['methods']['MagFlow']['value']:.6g} | "
            f"{baseline} {payload['methods'][baseline]['value']:.6g} | "
            f"{' < '.join(ranking)} |"
        )
    lines += [
        "",
        "## MuMax self-score status",
        "",
        f"Same-five diagnostic: {self_diag['same_five_including_self']['value']:.4f}.",
        f"Empirical five-from-five bootstrap diagnostic: "
        f"{self_diag['empirical_bootstrap_5_from_same_5']['value']:.4f}.",
        "",
        f"**Not paper-eligible:** {self_diag['reason']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--magflow-root", type=Path, default=DEFAULT_MAGFLOW)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--neuralmag-root", type=Path, default=DEFAULT_NEURALMAG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--self-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    roots = _method_roots(args.magflow_root, args.baseline_root, args.neuralmag_root)
    for method, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"missing {method} result root: {root}")
    args.output.mkdir(parents=True, exist_ok=True)

    sensitivity, sensitivity_rows = analyze_score_sensitivity(
        roots, iterations=args.bootstrap, seed=args.seed
    )
    observables, observable_rows = analyze_observables(
        roots, iterations=args.bootstrap, seed=args.seed
    )
    self_diag = analyze_nonindependent_self_diagnostics(
        args.magflow_root,
        bootstrap_replicates=args.self_bootstrap,
        seed=args.seed,
    )

    (args.output / "distribution_score_sensitivity.json").write_text(
        json.dumps(sensitivity, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "stochastic_physical_observables.json").write_text(
        json.dumps(observables, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "mumax_self_score_audit.json").write_text(
        json.dumps(self_diag, indent=2) + "\n", encoding="utf-8"
    )
    _write_rows(args.output / "distribution_score_condition_rows.csv", sensitivity_rows)
    _write_rows(args.output / "stochastic_observable_condition_rows.csv", observable_rows)
    (args.output / "REPORT.md").write_text(
        _markdown(sensitivity, observables, self_diag), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
