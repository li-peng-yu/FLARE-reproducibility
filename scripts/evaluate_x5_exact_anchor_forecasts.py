#!/usr/bin/env python3
"""Evaluate exact-anchor probabilistic forecasts with a fair energy score.

For every ``base x forecast condition x repeat anchor`` case, stochastic methods provide
``L`` draws conditioned on the exact MuMax3 initial field.  Deterministic
methods are evaluated as one-point distributions.  The primary metric is the
finite-ensemble-corrected energy score

    mean_j d(y_j, x) - mean_{j<k} d(y_j, y_k) / 2,

where ``d`` is the RMS Euclidean chord distance between the two flattened
valid-cell magnetization fields.  This field distance is a scaled Euclidean
norm, so the population energy score is proper; the off-diagonal correction
is unbiased for independently sampled finite ensembles.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The CLI can evaluate the 33
# complete-path conditions (165 drive-start anchors) in addition to the older
# independent-segment protocol.
SCFM_ROOT = (
    PROJECT_ROOT
    / "graph/results/x5_distribution_pareto_20260816/formal_33groups_v020/scfm_stage1"
)
BASELINE_ROOT = (
    PROJECT_ROOT
    / "graph/results/x5_distribution_pareto_20260816/formal_33groups_50k_v020"
)
NEURALMAG_ROOT = (
    PROJECT_ROOT
    / "graph/results/x5_distribution_pareto_20260816/neuralmag_x5/"
    "formal_33groups_anchor5_50k_v030"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "graph/results/x5_exact_anchor_forecasts_20260822/"
    "formal_330anchors_l5_fair_energy_v010"
)
DEFAULT_METHODS = (
    ("scfm_stage1", SCFM_ROOT),
    ("poseidon_t", BASELINE_ROOT / "poseidon_t"),
    ("cno_fm", BASELINE_ROOT / "cno_fm"),
    ("dpot_ti", BASELINE_ROOT / "dpot_ti"),
    ("mpp_avit_ti", BASELINE_ROOT / "mpp_avit_ti"),
    ("pdearena_unet", BASELINE_ROOT / "pdearena_unet"),
    ("le_pde", BASELINE_ROOT / "le_pde"),
    ("neuralmag_x5", NEURALMAG_ROOT),
)
PROTOCOL_ID = "exact_anchor_fair_energy_rms_chord_l5_v1"
METRIC_KEYS = (
    "fair_energy_score",
    "ensemble_mean_angle_deg",
    "mean_draw_angle_deg",
    "ensemble_spread_rms_chord",
)


@dataclass(frozen=True)
class MethodSpec:
    label: str
    root: Path


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _parse_method(value: str) -> MethodSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("method must be LABEL=/absolute/or/relative/root")
    label, root = value.split("=", 1)
    label = label.strip()
    if not label or not root.strip():
        raise argparse.ArgumentTypeError("method label and root must both be non-empty")
    return MethodSpec(label=label, root=Path(root).expanduser().resolve())


def _normalize_fields(fields: np.ndarray, valid: np.ndarray) -> np.ndarray:
    fields = np.asarray(fields, dtype=np.float32)
    norms = np.linalg.norm(fields, axis=1, keepdims=True)
    normalized = fields / np.maximum(norms, 1.0e-8)
    return normalized * valid[None, None]


def _rms_chord(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> float:
    delta = np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)
    squared = np.sum(delta * delta, axis=0)
    return float(np.sqrt(np.sum(squared * valid) / np.sum(valid)))


def _mean_angle_deg(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> float:
    dot = np.sum(left * right, axis=0)
    cross = np.linalg.norm(np.cross(left, right, axisa=0, axisb=0, axisc=0), axis=0)
    angle = np.degrees(np.arctan2(cross, np.clip(dot, -1.0, 1.0)))
    return float(np.sum(angle * valid) / np.sum(valid))


def _select_anchor_draws(
    model: np.ndarray,
    assignment: np.ndarray | None,
    *,
    repeat_index: int,
    ensemble_size: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    if assignment is None:
        if model.shape[0] <= repeat_index:
            raise RuntimeError(
                f"deterministic source has {model.shape[0]} outputs, "
                f"cannot select repeat {repeat_index}"
            )
        selected = np.asarray([repeat_index], dtype=np.int64)
        return np.asarray(model[selected], dtype=np.float32), selected, "deterministic"
    candidates = np.flatnonzero(assignment == repeat_index)
    # A deterministic endpoint
    # operator can retain an anchor-index file while providing exactly one
    # prediction per exact anchor.  Score that prediction as a one-point
    # distribution instead of requiring five stochastic draws.
    if candidates.size == 1:
        selected = candidates[:1]
        return np.asarray(model[selected], dtype=np.float32), selected, "deterministic"
    if candidates.size < ensemble_size:
        raise RuntimeError(
            f"anchor {repeat_index} has {candidates.size} draws; need {ensemble_size}"
        )
    selected = candidates[:ensemble_size]
    return np.asarray(model[selected], dtype=np.float32), selected, "stochastic"


def _evaluate_method(
    spec: MethodSpec,
    *,
    truth_root: Path,
    ensemble_size: int,
    expected_conditions: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_dirs = sorted(
        set((spec.root / "conditions").glob("base*_segment*"))
        | set((spec.root / "conditions").glob("base*_multisegment*"))
        | set((spec.root / "conditions").glob("base*_exact_control*"))
    )
    condition_dirs: list[Path] = []
    incomplete_condition_dirs: list[str] = []
    for condition_dir in candidate_dirs:
        condition_id = condition_dir.name
        truth_dir = (
            truth_root
            / "conditions"
            / condition_id
            / "same_condition_score_4x4_shift16"
        )
        required = (
            condition_dir / "condition_metadata.json",
            condition_dir / "model_samples/model_samples_f16.npy",
            truth_dir / "mumax_targets_f16.npy",
            truth_dir / "geometry_mask.npy",
        )
        if all(path.is_file() for path in required):
            condition_dirs.append(condition_dir)
        else:
            incomplete_condition_dirs.append(condition_id)
    if expected_conditions > 0 and len(condition_dirs) != expected_conditions:
        raise RuntimeError(
            f"{spec.label}: expected {expected_conditions} conditions, found "
            f"{len(condition_dirs)} complete conditions under {spec.root}; "
            f"ignored {len(incomplete_condition_dirs)} incomplete directories"
        )
    rows: list[dict[str, Any]] = []
    source_manifests: list[dict[str, Any]] = []
    for condition_dir in condition_dirs:
        condition_id = condition_dir.name
        metadata = _json(condition_dir / "condition_metadata.json")
        model_path = condition_dir / "model_samples/model_samples_f16.npy"
        assignment_path = condition_dir / "model_samples/anchor_repeat_index.npy"
        manifest_path = condition_dir / "model_samples/manifest.json"
        truth_dir = truth_root / "conditions" / condition_id / "same_condition_score_4x4_shift16"
        truth_path = truth_dir / "mumax_targets_f16.npy"
        mask_path = truth_dir / "geometry_mask.npy"
        for required in (model_path, truth_path, mask_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        model = np.load(model_path, mmap_mode="r", allow_pickle=False)
        truth = np.load(truth_path, mmap_mode="r", allow_pickle=False)
        valid = np.asarray(np.load(mask_path, allow_pickle=False), dtype=np.float32).squeeze()
        valid = valid > 0.5
        if valid.ndim != 2 or not np.any(valid):
            raise RuntimeError(f"invalid geometry mask: {mask_path}")
        assignment = (
            np.asarray(np.load(assignment_path, allow_pickle=False), dtype=np.int64)
            if assignment_path.is_file()
            else None
        )
        if assignment is not None and assignment.shape != (model.shape[0],):
            raise RuntimeError(
                f"{assignment_path}: assignment {assignment.shape} does not match "
                f"model array {model.shape}"
            )
        if manifest_path.is_file():
            source_manifests.append(_json(manifest_path))
        for repeat_index in range(truth.shape[0]):
            draws, selected, forecast_type = _select_anchor_draws(
                model,
                assignment,
                repeat_index=repeat_index,
                ensemble_size=ensemble_size,
            )
            draws = _normalize_fields(draws, valid)
            target = _normalize_fields(
                np.asarray(truth[repeat_index : repeat_index + 1], dtype=np.float32),
                valid,
            )[0]
            cross = np.asarray(
                [_rms_chord(draw, target, valid) for draw in draws],
                dtype=np.float64,
            )
            pairwise = np.asarray(
                [
                    _rms_chord(draws[j], draws[k], valid)
                    for j in range(len(draws))
                    for k in range(j + 1, len(draws))
                ],
                dtype=np.float64,
            )
            spread = float(pairwise.mean()) if pairwise.size else 0.0
            fair_energy = float(cross.mean() - 0.5 * spread)
            ensemble_mean = np.mean(draws, axis=0)
            ensemble_mean = _normalize_fields(ensemble_mean[None], valid)[0]
            draw_angles = np.asarray(
                [_mean_angle_deg(draw, target, valid) for draw in draws],
                dtype=np.float64,
            )
            rows.append(
                {
                    "method": spec.label,
                    "forecast_type": forecast_type,
                    "condition_id": condition_id,
                    "anchor_condition_id": f"{condition_id}_repeat{repeat_index:02d}",
                    "base_id": str(metadata["base_id"]),
                    "control_segment_index": int(metadata["control_segment_index"]),
                    "segment_role": str(metadata.get("segment_role", "condition")),
                    "truth_repeat_index": repeat_index,
                    "ensemble_size": int(len(draws)),
                    "source_model_indices": ";".join(str(int(i)) for i in selected),
                    "fair_energy_score": fair_energy,
                    "mean_truth_to_draw_rms_chord": float(cross.mean()),
                    "ensemble_spread_rms_chord": spread,
                    "ensemble_mean_angle_deg": _mean_angle_deg(
                        ensemble_mean, target, valid
                    ),
                    "mean_draw_angle_deg": float(draw_angles.mean()),
                }
            )
    checkpoint_ids = sorted(
        {
            str(item.get("checkpoint_sha256") or item.get("checkpoint") or "unknown")
            for item in source_manifests
        }
    )
    source = {
        "label": spec.label,
        "root": str(spec.root),
        "conditions": len(condition_dirs),
        "candidate_condition_directories": len(candidate_dirs),
        "ignored_incomplete_condition_directories": incomplete_condition_dirs,
        "anchor_cases": len(rows),
        "forecast_type": sorted({str(row["forecast_type"]) for row in rows}),
        "checkpoint_identifiers": checkpoint_ids,
    }
    return rows, source


def _group_values(rows: Iterable[dict[str, Any]], key: str) -> tuple[list[str], np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["base_id"])].append(float(row[key]))
    base_ids = sorted(grouped)
    return base_ids, np.asarray(
        [np.mean(grouped[base_id]) for base_id in base_ids], dtype=np.float64
    )


def _cluster_summary(
    rows: list[dict[str, Any]],
    key: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    base_ids, values = _group_values(rows, key)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(iterations, len(values)))
    bootstrap = values[sampled].mean(axis=1)
    all_values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "mean": float(all_values.mean()),
        "bootstrap_95ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "median": float(np.median(all_values)),
        "base_groups": len(base_ids),
        "anchor_cases": len(all_values),
        "bootstrap_unit": "base group; all forecast conditions and repeat anchors kept together",
    }


def _summarize_method(
    rows: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    segments = sorted({int(row["control_segment_index"]) for row in rows})
    result: dict[str, Any] = {
        "method": str(rows[0]["method"]),
        "forecast_type": sorted({str(row["forecast_type"]) for row in rows}),
        "combined": {},
        "by_segment": {},
    }
    for offset, key in enumerate(METRIC_KEYS):
        result["combined"][key] = _cluster_summary(
            rows,
            key,
            iterations=iterations,
            seed=seed + 1009 * (offset + 1),
        )
    for segment in segments:
        selected = [
            row for row in rows if int(row["control_segment_index"]) == segment
        ]
        result["by_segment"][str(segment)] = {
            key: _cluster_summary(
                selected,
                key,
                iterations=iterations,
                seed=seed + 10_007 * segment + 1009 * (offset + 1),
            )
            for offset, key in enumerate(METRIC_KEYS)
        }
    return result


def _paired_comparison(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    key: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    reference = {str(row["anchor_condition_id"]): row for row in reference_rows}
    candidate = {str(row["anchor_condition_id"]): row for row in candidate_rows}
    if set(reference) != set(candidate):
        raise RuntimeError("paired methods do not contain the same exact-anchor cases")
    grouped_reference: dict[str, list[float]] = defaultdict(list)
    grouped_candidate: dict[str, list[float]] = defaultdict(list)
    for anchor_id in sorted(reference):
        base_id = str(reference[anchor_id]["base_id"])
        if base_id != str(candidate[anchor_id]["base_id"]):
            raise RuntimeError(f"base mismatch for {anchor_id}")
        grouped_reference[base_id].append(float(reference[anchor_id][key]))
        grouped_candidate[base_id].append(float(candidate[anchor_id][key]))
    base_ids = sorted(grouped_reference)
    ref = np.asarray(
        [np.mean(grouped_reference[base_id]) for base_id in base_ids],
        dtype=np.float64,
    )
    cand = np.asarray(
        [np.mean(grouped_candidate[base_id]) for base_id in base_ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(base_ids), size=(iterations, len(base_ids)))
    ref_boot = ref[sampled].mean(axis=1)
    cand_boot = cand[sampled].mean(axis=1)
    difference = cand_boot - ref_boot
    relative_reduction = 1.0 - ref_boot / cand_boot
    return {
        "metric": key,
        "candidate_minus_reference": float(cand.mean() - ref.mean()),
        "candidate_minus_reference_bootstrap_95ci": np.quantile(
            difference, [0.025, 0.975]
        ).tolist(),
        "reference_relative_reduction": float(1.0 - ref.mean() / cand.mean()),
        "reference_relative_reduction_bootstrap_95ci": np.quantile(
            relative_reduction, [0.025, 0.975]
        ).tolist(),
        "bootstrap_probability_candidate_greater": float(np.mean(difference > 0.0)),
        "base_groups": len(base_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        action="append",
        type=_parse_method,
        help="Repeatable LABEL=ROOT specification; defaults to the eight paper methods.",
    )
    parser.add_argument("--truth-root", type=Path, default=SCFM_ROOT)
    parser.add_argument("--reference-label", default="scfm_stage1")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument(
        "--expected-conditions",
        type=int,
        default=66,
        help="Require this many conditions; use 0 to accept every complete condition found.",
    )
    parser.add_argument("--bootstrap", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=208220700)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.ensemble_size < 2:
        raise ValueError("stochastic fair energy evaluation requires at least two draws")
    if args.bootstrap <= 0 or args.workers <= 0:
        raise ValueError("bootstrap and workers must be positive")
    methods = tuple(args.method or (MethodSpec(label, root) for label, root in DEFAULT_METHODS))
    labels = [spec.label for spec in methods]
    if len(labels) != len(set(labels)):
        raise ValueError(f"duplicate method labels: {labels}")
    if args.reference_label not in labels:
        raise ValueError(f"reference label {args.reference_label!r} not in {labels}")
    truth_root = args.truth_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(args.workers, len(methods))) as executor:
        futures = {
            executor.submit(
                _evaluate_method,
                spec,
                truth_root=truth_root,
                ensemble_size=args.ensemble_size,
                expected_conditions=args.expected_conditions,
            ): spec
            for spec in methods
        }
        for future in as_completed(futures):
            spec = futures[future]
            rows, source = future.result()
            all_rows[spec.label] = rows
            sources[spec.label] = source
            print(
                json.dumps(
                    {
                        "method": spec.label,
                        "status": "complete",
                        "anchor_cases": len(rows),
                        "fair_energy_score": float(
                            np.mean([row["fair_energy_score"] for row in rows])
                        ),
                    }
                ),
                flush=True,
            )

    ordered_rows = [row for label in labels for row in all_rows[label]]
    _write_csv(output_dir / "per_anchor_metrics.csv", ordered_rows)
    method_summaries = {
        label: _summarize_method(
            all_rows[label],
            iterations=args.bootstrap,
            seed=args.seed + int.from_bytes(
                hashlib.sha256(label.encode("utf-8")).digest()[:4], "little"
            ),
        )
        for label in labels
    }
    summary_rows: list[dict[str, Any]] = []
    for label in labels:
        combined = method_summaries[label]["combined"]
        row: dict[str, Any] = {
            "method": label,
            "forecast_type": ";".join(method_summaries[label]["forecast_type"]),
        }
        for key in METRIC_KEYS:
            metric = combined[key]
            row[key] = metric["mean"]
            row[f"{key}_ci_low"] = metric["bootstrap_95ci"][0]
            row[f"{key}_ci_high"] = metric["bootstrap_95ci"][1]
        summary_rows.append(row)
    _write_csv(output_dir / "method_summary.csv", summary_rows)

    reference_rows = all_rows[args.reference_label]
    paired: dict[str, Any] = {}
    paired_rows: list[dict[str, Any]] = []
    for candidate_label in labels:
        if candidate_label == args.reference_label:
            continue
        comparison_key = f"{candidate_label}_vs_{args.reference_label}"
        paired[comparison_key] = {}
        for metric_offset, key in enumerate(METRIC_KEYS[:3]):
            comparison = _paired_comparison(
                reference_rows,
                all_rows[candidate_label],
                key,
                iterations=args.bootstrap,
                seed=args.seed + 50_021 * (labels.index(candidate_label) + 1) + metric_offset,
            )
            paired[comparison_key][key] = comparison
            paired_rows.append(
                {
                    "reference": args.reference_label,
                    "candidate": candidate_label,
                    **comparison,
                    "candidate_minus_reference_bootstrap_95ci": ";".join(
                        str(value)
                        for value in comparison[
                            "candidate_minus_reference_bootstrap_95ci"
                        ]
                    ),
                    "reference_relative_reduction_bootstrap_95ci": ";".join(
                        str(value)
                        for value in comparison[
                            "reference_relative_reduction_bootstrap_95ci"
                        ]
                    ),
                }
            )
    if paired_rows:
        _write_csv(output_dir / "paired_comparisons.csv", paired_rows)
    summary = {
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "primary_metric": "fair_energy_score",
        "field_distance": (
            "RMS Euclidean chord distance over the flattened valid-cell unit-vector field"
        ),
        "fair_estimator": (
            "mean truth-to-draw distance minus one half of the mean unordered "
            "off-diagonal draw-to-draw distance"
        ),
        "truth_root": str(truth_root),
        "ensemble_size_for_stochastic_methods": args.ensemble_size,
        "deterministic_forecasts": "evaluated as one-point distributions",
        "bootstrap_iterations": args.bootstrap,
        "bootstrap_unit": "base group; all forecast conditions and repeat anchors kept together",
        "reference_label": args.reference_label,
        "sources": sources,
        "methods": method_summaries,
        "paired_comparisons": paired,
    }
    _write_json(output_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
