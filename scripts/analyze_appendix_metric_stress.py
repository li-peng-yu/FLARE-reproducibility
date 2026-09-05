#!/usr/bin/env python3
"""Appendix audit for metric choices and finite ensemble budgets.

This script has three deliberately separate parts:

1. On the independent x30 collection, compare disjoint MuMax3 ensembles at
   5-vs-5, 10-vs-10, and 15-vs-15, and compare FLARE ensembles of size
   1, 2, 5, 10, and 20 against five MuMax3 repeats.
2. On the 33-base primary comparison, recompute the Distribution Score after
   subsampling every method to N=1, 2, and 5 and bootstrap the resulting rank.
   Deterministic baselines contain only five distinct anchor-matched outputs,
   so ranks above N=5 are intentionally not fabricated.
3. Evaluate the full-resolution local-translation distance for 2x2, 4x4,
   and 8x8 patches and windows 0, +/-8, +/-16, and +/-32 on analytic
   sphere-valued skyrmion-shift, topology-loss, and domain-wall-shift cases.

All randomization is deterministic. Confidence intervals resample base
conditions, retaining both physical segments and all within-condition trials.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[1]
DIST_SRC = (
    PROJECT
    / "third_party/distribution_score/src"
)
if str(DIST_SRC) not in sys.path:
    sys.path.insert(0, str(DIST_SRC))

from distribution_score.distance import patch_shift_distance  # noqa: E402


DEFAULT_X30 = (
    PROJECT
    / "outputs/skx_bt_165base_x30/evaluation_20260818/"
    "same_condition_density_4x4_shift16_v020/"
    "x5_scratch50k_20260812_on_x30_test"
)
DEFAULT_OUTPUT = PROJECT / "outputs/paper_artifacts/appendix_metric_stress"
DEFAULT_FIGURE = PROJECT / "outputs/figures/appendix_metric_stress.pdf"
PRIMARY_FLARE = (
    PROJECT
    / "graph/results/x5_distribution_pareto_20260822/"
    "formal_standard_prior_n5_v020/scfm_stage1"
)
PRIMARY_BASELINES = (
    PROJECT
    / "graph/results/x5_distribution_pareto_20260816/"
    "formal_33groups_50k_v020"
)

METHOD_ROOTS = {
    "FLARE": PRIMARY_FLARE,
    "Poseidon-T": PRIMARY_BASELINES / "poseidon_t",
    "CNO-FM": PRIMARY_BASELINES / "cno_fm",
    "DPOT-Ti": PRIMARY_BASELINES / "dpot_ti",
    "MPP-AViT-Ti": PRIMARY_BASELINES / "mpp_avit_ti",
    "PDEArena U-Net": PRIMARY_BASELINES / "pdearena_unet",
    "LE-PDE": PRIMARY_BASELINES / "le_pde",
    "NeuralMAG-x5": (
        PROJECT
        / "graph/results/x5_distribution_pareto_20260816/neuralmag_x5/"
        "formal_33groups_50k_v020"
    ),
}

MODEL_BUDGETS = (1, 2, 5, 10, 20)
SIMULATOR_BUDGETS = (5, 10, 15)
RANK_BUDGETS = (1, 2, 5)
PATCH_BLOCKS = (2, 4, 8)
SHIFT_WINDOWS = (0, 8, 16, 32)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _stable_seed(text: str, base: int) -> int:
    digest = hashlib.sha256(f"{base}|{text}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _normalise(fields: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.normalize(fields.float(), dim=1, eps=1.0e-8) * mask[None, None]


@torch.inference_mode()
def _field_distance_matrices(
    fields: np.ndarray,
    mask_array: np.ndarray,
    *,
    device: torch.device,
    pair_block: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return angular-degree and RMS-chord pairwise matrices."""
    mask = torch.from_numpy(np.array(mask_array, dtype=np.bool_, copy=True)).to(device)
    values = torch.from_numpy(np.asarray(fields, dtype=np.float32)).to(device)
    values = _normalise(values, mask.float())
    count = int(mask.sum().item())
    if count <= 0:
        raise ValueError("empty geometry mask")
    n = values.shape[0]
    angular = torch.empty((n, n), dtype=torch.float32, device=device)
    valid = mask
    for left_start in range(0, n, pair_block):
        left_end = min(n, left_start + pair_block)
        left = values[left_start:left_end, None]
        for right_start in range(left_start, n, pair_block):
            right_end = min(n, right_start + pair_block)
            right = values[None, right_start:right_end]
            dot = (left * right).sum(dim=2).clamp(-1.0, 1.0)
            cross = torch.linalg.vector_norm(torch.cross(left, right, dim=2), dim=2)
            distance = torch.rad2deg(torch.atan2(cross, dot))[..., valid].mean(dim=-1)
            angular[left_start:left_end, right_start:right_end] = distance
            if right_start != left_start:
                angular[right_start:right_end, left_start:left_end] = distance.T
    angular.fill_diagonal_(0.0)

    flattened = values[:, :, valid].reshape(n, -1)
    mean_dot = (flattened @ flattened.T) / float(count)
    chord = torch.sqrt((2.0 - 2.0 * mean_dot).clamp_min(0.0))
    chord.fill_diagonal_(0.0)
    return angular.cpu().numpy(), chord.cpu().numpy()


def _distribution_score(
    reference: np.ndarray,
    model: np.ndarray,
    truth_self: np.ndarray,
    truth_model: np.ndarray,
) -> float:
    ref = truth_self[np.ix_(reference, reference)].astype(np.float64)
    cross = truth_model[np.ix_(reference, model)].astype(np.float64)
    diagonal = np.eye(len(reference), dtype=bool)
    nearest = np.where(diagonal, np.inf, ref).min(axis=1)
    sigma = max(float(np.median(nearest)), 1.0e-6)
    ref_kernel = np.exp(-(ref**2) / (2.0 * sigma**2))
    model_kernel = np.exp(-(cross**2) / (2.0 * sigma**2))
    q_ref = (ref_kernel.sum(axis=1) - np.diag(ref_kernel)) / (len(reference) - 1)
    q_model = model_kernel.mean(axis=1)
    return math.exp(
        -float(np.abs(np.log((q_model + 1.0e-12) / (q_ref + 1.0e-12))).mean())
    )


def _energy_distance(
    left: np.ndarray,
    right: np.ndarray,
    matrix: np.ndarray,
) -> float:
    cross = matrix[np.ix_(left, right)].mean()
    within_left = matrix[np.ix_(left, left)].mean()
    within_right = matrix[np.ix_(right, right)].mean()
    return max(float(2.0 * cross - within_left - within_right), 0.0)


def _fair_energy_score(
    forecasts: np.ndarray,
    targets: np.ndarray,
    chord: np.ndarray,
) -> float | None:
    if len(forecasts) < 2:
        return None
    cross = float(chord[np.ix_(forecasts, targets)].mean())
    within = chord[np.ix_(forecasts, forecasts)]
    correction = float(within.sum()) / (2.0 * len(forecasts) * (len(forecasts) - 1))
    return cross - correction


def _ordinary_energy_score(
    forecasts: np.ndarray,
    targets: np.ndarray,
    chord: np.ndarray,
) -> float:
    cross = float(chord[np.ix_(forecasts, targets)].mean())
    if len(forecasts) == 1:
        return cross
    within = float(chord[np.ix_(forecasts, forecasts)].mean())
    return cross - 0.5 * within


def _condition_dirs(root: Path) -> list[Path]:
    result = sorted((root / "conditions").glob("base*_segment*"))
    if not result:
        raise FileNotFoundError(f"no conditions under {root}")
    return result


def _model_pool_indices(count: int, pool_size: int, condition_id: str) -> np.ndarray:
    if count < pool_size:
        raise ValueError(f"{condition_id}: only {count} model samples; need {pool_size}")
    ordered = sorted(
        range(count),
        key=lambda index: hashlib.sha256(
            f"appendix-metric-stress|{condition_id}|{index}".encode()
        ).hexdigest(),
    )
    return np.asarray(ordered[:pool_size], dtype=np.int64)


def _prepare_x30_matrices(
    condition: Path,
    *,
    output: Path,
    device: torch.device,
    pair_block: int,
    pool_size: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    score_dir = condition / "same_condition_score_4x4_shift16"
    metadata = _json(condition / "condition_metadata.json")
    truth_self = np.load(score_dir / "truth_self_patch_shift_distance.npy").astype(np.float64)
    truth_to_all_model = np.load(
        score_dir / "truth_to_model_patch_shift_distance.npy"
    ).astype(np.float64)
    model_path = condition / "model_samples/model_samples_f16.npy"
    model_memmap = np.load(model_path, mmap_mode="r")
    manifest = _json(condition / "model_samples/manifest.json")
    checkpoint_sha256 = str(manifest.get("checkpoint_sha256", "unknown"))
    pool = _model_pool_indices(model_memmap.shape[0], pool_size, condition.name)
    truth_model = truth_to_all_model[:, pool]
    cache = output / "distance_cache" / f"{condition.name}.npz"
    if cache.is_file():
        loaded = np.load(cache)
        cached_checkpoint = (
            str(loaded["checkpoint_sha256"].item())
            if "checkpoint_sha256" in loaded.files
            else ""
        )
        if (
            cached_checkpoint == checkpoint_sha256
            and np.array_equal(loaded["model_pool_indices"], pool)
        ):
            return (
                metadata,
                truth_self,
                truth_model,
                loaded["angular"],
                loaded["chord"],
                pool,
            )

    truth = np.asarray(
        np.load(score_dir / "mumax_targets_f16.npy", mmap_mode="r"), dtype=np.float32
    )
    model = np.asarray(model_memmap[pool], dtype=np.float32)
    combined = np.concatenate([truth, model], axis=0)
    mask = np.load(score_dir / "geometry_mask.npy")
    angular, chord = _field_distance_matrices(
        combined, mask, device=device, pair_block=pair_block
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        angular=angular,
        chord=chord,
        model_pool_indices=pool,
        checkpoint_sha256=np.asarray(checkpoint_sha256),
    )
    return metadata, truth_self, truth_model, angular, chord, pool


def _x30_budget_rows(
    root: Path,
    *,
    output: Path,
    device: torch.device,
    pair_block: int,
    pool_size: int,
    trials: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_rows: list[dict[str, Any]] = []
    simulator_rows: list[dict[str, Any]] = []
    conditions = _condition_dirs(root)
    if len(conditions) != 34:
        raise RuntimeError(f"expected 34 x30 conditions, found {len(conditions)}")
    for condition_index, condition in enumerate(conditions, start=1):
        print(f"[x30 {condition_index:02d}/{len(conditions)}] {condition.name}", flush=True)
        metadata, patch_truth, patch_model, angular, chord, pool = _prepare_x30_matrices(
            condition,
            output=output,
            device=device,
            pair_block=pair_block,
            pool_size=pool_size,
        )
        n_truth = patch_truth.shape[0]
        n_pool = patch_model.shape[1]
        model_offset = n_truth
        rng = np.random.default_rng(_stable_seed(condition.name, seed))
        for trial in range(trials):
            truth_order = rng.permutation(n_truth)
            model_order = rng.permutation(n_pool)
            reference = truth_order[:5]
            target = truth_order[5:10]
            for budget in MODEL_BUDGETS:
                selected = model_order[:budget]
                angular_truth = reference
                angular_model = model_offset + selected
                fair = None
                ordinary = None
                if str(metadata["segment_role"]) == "drive":
                    fair = _fair_energy_score(angular_model, target, chord)
                    ordinary = _ordinary_energy_score(angular_model, target, chord)
                model_rows.append(
                    {
                        "condition_id": condition.name,
                        "base_id": str(metadata["base_id"]).zfill(4),
                        "segment_role": str(metadata["segment_role"]),
                        "trial": trial,
                        "model_budget": budget,
                        "truth_reference_budget": 5,
                        "distribution_score": _distribution_score(
                            reference, selected, patch_truth, patch_model
                        ),
                        "angular_energy_distance_deg": _energy_distance(
                            angular_truth, angular_model, angular
                        ),
                        "fair_energy_score": fair,
                        "ordinary_energy_score": ordinary,
                        "model_pool_size": n_pool,
                    }
                )
            for budget in SIMULATOR_BUDGETS:
                left = truth_order[:budget]
                right = truth_order[budget : 2 * budget]
                fair = None
                if str(metadata["segment_role"]) == "drive":
                    fair = _fair_energy_score(left, right, chord)
                simulator_rows.append(
                    {
                        "condition_id": condition.name,
                        "base_id": str(metadata["base_id"]).zfill(4),
                        "segment_role": str(metadata["segment_role"]),
                        "trial": trial,
                        "simulator_budget_each_side": budget,
                        "distribution_score": _distribution_score(
                            left, right, patch_truth, patch_truth
                        ),
                        "angular_energy_distance_deg": _energy_distance(
                            left, right, angular
                        ),
                        "fair_energy_score": fair,
                    }
                )
    return model_rows, simulator_rows


def _trial_average(
    rows: list[dict[str, Any]],
    budget_key: str,
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row[budget_key]), str(row["condition_id"]))].append(row)
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (budget, condition_id), selected in grouped.items():
        first = selected[0]
        fair_values = [float(row["fair_energy_score"]) for row in selected if row["fair_energy_score"] is not None]
        ordinary_values = [float(row["ordinary_energy_score"]) for row in selected if row.get("ordinary_energy_score") is not None]
        result[budget].append(
            {
                "condition_id": condition_id,
                "base_id": first["base_id"],
                "segment_role": first["segment_role"],
                "log_distribution_score": float(
                    np.mean([math.log(max(float(row["distribution_score"]), 1.0e-300)) for row in selected])
                ),
                "angular_energy_distance_deg": float(
                    np.mean([float(row["angular_energy_distance_deg"]) for row in selected])
                ),
                "fair_energy_score": float(np.mean(fair_values)) if fair_values else None,
                "ordinary_energy_score": float(np.mean(ordinary_values)) if ordinary_values else None,
            }
        )
    return result


def _cluster_bootstrap_summary(
    rows: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_base[str(row["base_id"])].append(row)
    bases = sorted(by_base)
    rng = np.random.default_rng(seed)

    def estimate(selected_bases: Iterable[str]) -> tuple[float, float, float | None, float | None]:
        selected_rows = [row for base in selected_bases for row in by_base[base]]
        score = math.exp(float(np.mean([row["log_distribution_score"] for row in selected_rows])))
        angular = float(np.mean([row["angular_energy_distance_deg"] for row in selected_rows]))
        fair_values = [row["fair_energy_score"] for row in selected_rows if row["fair_energy_score"] is not None]
        ordinary_values = [row["ordinary_energy_score"] for row in selected_rows if row["ordinary_energy_score"] is not None]
        fair = float(np.mean(fair_values)) if fair_values else None
        ordinary = float(np.mean(ordinary_values)) if ordinary_values else None
        return score, angular, fair, ordinary

    observed = estimate(bases)
    draws = np.full((iterations, 4), np.nan, dtype=np.float64)
    for index in range(iterations):
        sampled = rng.integers(0, len(bases), size=len(bases))
        draws[index] = [
            value if value is not None else np.nan
            for value in estimate([bases[int(i)] for i in sampled])
        ]
    names = (
        "distribution_score",
        "angular_energy_distance_deg",
        "fair_energy_score",
        "ordinary_energy_score",
    )
    summary: dict[str, Any] = {
        "base_groups": len(bases),
        "conditions": len(rows),
        "bootstrap_iterations": iterations,
        "bootstrap_unit": "base condition; both segments retained",
    }
    for column, name in enumerate(names):
        if observed[column] is None:
            summary[name] = None
        else:
            valid = draws[:, column][np.isfinite(draws[:, column])]
            summary[name] = {
                "estimate": observed[column],
                "bootstrap_95ci": np.quantile(valid, [0.025, 0.975]).tolist(),
            }
    return summary


def _summarise_budgets(
    rows: list[dict[str, Any]],
    budget_key: str,
    *,
    bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    averaged = _trial_average(rows, budget_key)
    result = []
    for offset, budget in enumerate(sorted(averaged)):
        summary = _cluster_bootstrap_summary(
            averaged[budget], iterations=bootstrap, seed=seed + offset * 1009
        )
        result.append({budget_key: budget, **summary})
    return result


def _primary_rank_stress(
    *,
    bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Subsample the primary five-output comparison and bootstrap ranks."""
    condition_data: dict[str, dict[str, tuple[str, np.ndarray, np.ndarray]]] = {}
    common_conditions: set[str] | None = None
    for method, root in METHOD_ROOTS.items():
        conditions = _condition_dirs(root)
        ids = {condition.name for condition in conditions}
        common_conditions = ids if common_conditions is None else common_conditions & ids
        loaded: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
        for condition in conditions:
            score_dir = condition / "same_condition_score_4x4_shift16"
            metadata = _json(condition / "condition_metadata.json")
            loaded[condition.name] = (
                str(metadata["base_id"]).zfill(4),
                np.load(score_dir / "truth_self_patch_shift_distance.npy").astype(np.float64),
                np.load(score_dir / "truth_to_model_patch_shift_distance.npy").astype(np.float64),
            )
        condition_data[method] = loaded
    if common_conditions is None or len(common_conditions) != 66:
        raise RuntimeError(f"expected 66 common primary conditions, got {len(common_conditions or [])}")
    condition_ids = sorted(common_conditions)
    bases = sorted({condition_data["FLARE"][condition_id][0] for condition_id in condition_ids})
    if len(bases) != 33:
        raise RuntimeError(f"expected 33 primary bases, got {len(bases)}")

    output_rows: list[dict[str, Any]] = []
    rank_payload: dict[str, Any] = {}
    methods = list(METHOD_ROOTS)
    for budget in RANK_BUDGETS:
        subsets = list(itertools.combinations(range(5), budget))
        scores: dict[str, np.ndarray] = {
            method: np.empty((len(subsets), len(condition_ids)), dtype=np.float64)
            for method in methods
        }
        for method in methods:
            for subset_index, subset in enumerate(subsets):
                model = np.asarray(subset, dtype=np.int64)
                for condition_index, condition_id in enumerate(condition_ids):
                    _, truth_self, truth_model = condition_data[method][condition_id]
                    reference = np.arange(truth_self.shape[0], dtype=np.int64)
                    scores[method][subset_index, condition_index] = _distribution_score(
                        reference, model, truth_self, truth_model
                    )
        observed = {
            method: math.exp(float(np.log(scores[method].clip(1.0e-300)).mean()))
            for method in methods
        }
        rng = np.random.default_rng(seed + budget * 10007)
        score_draws = {method: np.empty(bootstrap, dtype=np.float64) for method in methods}
        rank_draws = {method: np.empty(bootstrap, dtype=np.int64) for method in methods}
        base_to_conditions = {
            base: [
                index
                for index, condition_id in enumerate(condition_ids)
                if condition_data["FLARE"][condition_id][0] == base
            ]
            for base in bases
        }
        for iteration in range(bootstrap):
            sampled_bases = rng.integers(0, len(bases), size=len(bases))
            condition_draw = [
                condition_index
                for base_index in sampled_bases
                for condition_index in base_to_conditions[bases[int(base_index)]]
            ]
            subset_index = int(rng.integers(0, len(subsets)))
            iteration_scores = {}
            for method in methods:
                value = math.exp(
                    float(
                        np.log(scores[method][subset_index, condition_draw].clip(1.0e-300)).mean()
                    )
                )
                score_draws[method][iteration] = value
                iteration_scores[method] = value
            ordering = sorted(methods, key=lambda method: (-iteration_scores[method], method))
            for rank, method in enumerate(ordering, start=1):
                rank_draws[method][iteration] = rank
        rank_payload[str(budget)] = {
            "model_budget": budget,
            "truth_reference_budget": 5,
            "enumerated_model_subsets": len(subsets),
            "methods": {},
        }
        for method in methods:
            row = {
                "model_budget": budget,
                "method": method,
                "distribution_score": observed[method],
                "score_bootstrap_95ci_low": float(np.quantile(score_draws[method], 0.025)),
                "score_bootstrap_95ci_high": float(np.quantile(score_draws[method], 0.975)),
                "median_rank": float(np.median(rank_draws[method])),
                "rank_bootstrap_95ci_low": float(np.quantile(rank_draws[method], 0.025)),
                "rank_bootstrap_95ci_high": float(np.quantile(rank_draws[method], 0.975)),
                "probability_rank_1": float(np.mean(rank_draws[method] == 1)),
            }
            output_rows.append(row)
            rank_payload[str(budget)]["methods"][method] = row
    rank_payload["protocol"] = {
        "description": "All combinations of N outputs from each method's five anchor-matched outputs; base-cluster bootstrap chooses one common subset per draw.",
        "scope": "Ranks above N=5 are not reported because primary deterministic comparators have only five distinct anchor-matched outputs.",
        "base_groups": 33,
        "conditions": 66,
        "bootstrap_iterations": bootstrap,
    }
    return output_rows, rank_payload


def _skyrmions(centres: list[tuple[float, float]], *, radius: float = 25.0, wall: float = 5.0) -> np.ndarray:
    y, x = np.mgrid[0:256, 0:256].astype(np.float64)
    distance_fields = [np.hypot(x - cx, y - cy) for cx, cy in centres]
    distances = np.stack(distance_fields)
    nearest = np.argmin(distances, axis=0)
    radius_map = np.take_along_axis(distances, nearest[None], axis=0)[0]
    cx_map = np.choose(nearest, [centre[0] for centre in centres])
    cy_map = np.choose(nearest, [centre[1] for centre in centres])
    phi = np.arctan2(y - cy_map, x - cx_map)
    theta = 0.5 * np.pi * (1.0 - np.tanh((radius_map - radius) / wall))
    field = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )
    return field.astype(np.float32)


def _domain_wall(position: float, *, width: float = 5.0) -> np.ndarray:
    _, x = np.mgrid[0:256, 0:256].astype(np.float64)
    coordinate = (x - position) / width
    mz = np.tanh(coordinate)
    mx = 1.0 / np.cosh(coordinate)
    my = np.zeros_like(mx)
    return np.stack([mx, my, mz]).astype(np.float32)


def _topological_charge(field: np.ndarray) -> float:
    m = np.asarray(field, dtype=np.float64)
    dmdy = 0.5 * (m[:, 2:, 1:-1] - m[:, :-2, 1:-1])
    dmdx = 0.5 * (m[:, 1:-1, 2:] - m[:, 1:-1, :-2])
    centre = m[:, 1:-1, 1:-1]
    density = np.einsum("ihw,ihw->hw", centre, np.cross(dmdx, dmdy, axisa=0, axisb=0, axisc=0))
    return float(density.sum() / (4.0 * np.pi))


def _pixel_metrics(reference: np.ndarray, query: np.ndarray) -> tuple[float, float]:
    reference_t = F.normalize(torch.from_numpy(reference), dim=0)
    query_t = F.normalize(torch.from_numpy(query), dim=0)
    dot = (reference_t * query_t).sum(dim=0).clamp(-1.0, 1.0)
    cross = torch.linalg.vector_norm(torch.cross(reference_t, query_t, dim=0), dim=0)
    angle = float(torch.rad2deg(torch.atan2(cross, dot)).mean().item())
    mse = float((reference_t - query_t).square().mean().item())
    return angle, mse


def _translation_cases() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "skyrmion displacement": (
            _skyrmions([(128.0, 128.0)]),
            _skyrmions([(140.0, 134.0)]),
        ),
        "topology disappearance": (
            _skyrmions([(92.0, 128.0), (164.0, 128.0)]),
            _skyrmions([(92.0, 128.0)]),
        ),
        "domain-wall shift": (
            _domain_wall(128.0),
            _domain_wall(142.0),
        ),
    }


def _translation_stress(device: torch.device) -> tuple[list[dict[str, Any]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    cases = _translation_cases()
    mask = torch.ones((1, 1, 256, 256), dtype=torch.float32, device=device)
    rows: list[dict[str, Any]] = []
    for case, (reference_np, query_np) in cases.items():
        pixel_angle, pixel_mse = _pixel_metrics(reference_np, query_np)
        reference = torch.from_numpy(reference_np[None]).to(device)
        query = torch.from_numpy(query_np[None]).to(device)
        q_reference = _topological_charge(reference_np)
        q_query = _topological_charge(query_np)
        for blocks in PATCH_BLOCKS:
            for shift in SHIFT_WINDOWS:
                distance, valid_patches = patch_shift_distance(
                    reference,
                    query,
                    mask,
                    blocks=blocks,
                    shift_radius=shift,
                    reference_chunk=1,
                )
                rows.append(
                    {
                        "perturbation": case,
                        "patch_grid": f"{blocks}x{blocks}",
                        "patch_blocks_each_axis": blocks,
                        "shift_radius_px": shift,
                        "translation_distance": float(distance[0, 0]),
                        "pixel_angular_error_deg": pixel_angle,
                        "pixel_mse": pixel_mse,
                        "q_reference": q_reference,
                        "q_query": q_query,
                        "absolute_delta_q": abs(q_reference - q_query),
                        "valid_patches": valid_patches,
                    }
                )
    return rows, cases


def _metric(summary: dict[str, Any], name: str) -> tuple[float, float, float] | None:
    value = summary.get(name)
    if value is None:
        return None
    return (
        float(value["estimate"]),
        float(value["bootstrap_95ci"][0]),
        float(value["bootstrap_95ci"][1]),
    )


def _plot(
    *,
    model_summary: list[dict[str, Any]],
    simulator_summary: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
    translation_rows: list[dict[str, Any]],
    cases: dict[str, tuple[np.ndarray, np.ndarray]],
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colours = {"DS": "#0072B2", "AED": "#D55E00", "FES": "#009E73"}
    figure, axes = plt.subplots(2, 3, figsize=(10.4, 6.0), constrained_layout=True)

    def budget_panel(axis: plt.Axes, rows: list[dict[str, Any]], budget_key: str, title: str) -> None:
        budgets = np.asarray([int(row[budget_key]) for row in rows])
        score = np.asarray([_metric(row, "distribution_score") for row in rows])
        angular = np.asarray([_metric(row, "angular_energy_distance_deg") for row in rows])
        fair_values = [_metric(row, "fair_energy_score") for row in rows]
        axis.errorbar(
            budgets,
            score[:, 0],
            yerr=np.vstack([score[:, 0] - score[:, 1], score[:, 2] - score[:, 0]]),
            color=colours["DS"], marker="o", capsize=2, label="Distribution Score",
        )
        axis.set_ylabel("Distribution Score", color=colours["DS"])
        axis.tick_params(axis="y", labelcolor=colours["DS"])
        twin = axis.twinx()
        twin.errorbar(
            budgets,
            angular[:, 0],
            yerr=np.vstack([angular[:, 0] - angular[:, 1], angular[:, 2] - angular[:, 0]]),
            color=colours["AED"], marker="s", capsize=2, label="Angular ED",
        )
        valid_fair = [(budget, value) for budget, value in zip(budgets, fair_values) if value is not None]
        if valid_fair:
            fair_budget = np.asarray([item[0] for item in valid_fair])
            fair = np.asarray([item[1] for item in valid_fair])
            twin.errorbar(
                fair_budget,
                fair[:, 0] * 30.0,
                yerr=np.vstack([fair[:, 0] - fair[:, 1], fair[:, 2] - fair[:, 0]]) * 30.0,
                color=colours["FES"], marker="^", capsize=2, linestyle="--",
                label="Fair ES x30",
            )
        twin.set_ylabel("Angular ED (deg); fair ES x30", color="#444444")
        axis.set_xlabel("Samples on each indicated side")
        axis.set_xticks(budgets)
        axis.set_title(title, loc="left", fontweight="bold")
        handles_left, labels_left = axis.get_legend_handles_labels()
        handles_right, labels_right = twin.get_legend_handles_labels()
        axis.legend(
            handles_left + handles_right,
            labels_left + labels_right,
            frameon=False,
            loc="best",
        )
        axis.grid(alpha=0.2)

    budget_panel(axes[0, 0], model_summary, "model_budget", "a  FLARE sample pressure (x30)")
    budget_panel(
        axes[0, 1], simulator_summary, "simulator_budget_each_side", "b  MuMax3 self-reference"
    )

    rank_axis = axes[0, 2]
    flare_rank = [row for row in rank_rows if row["method"] == "FLARE"]
    rank_axis.plot(
        [row["model_budget"] for row in flare_rank],
        [row["probability_rank_1"] for row in flare_rank],
        marker="o", color="#0072B2", label="P(FLARE ranks first)",
    )
    rank_axis.set_ylim(-0.03, 1.03)
    rank_axis.set_xticks(RANK_BUDGETS)
    rank_axis.set_xlabel("Model outcomes N (truth M=5)")
    rank_axis.set_ylabel("Bootstrap probability")
    rank_axis.grid(alpha=0.2)
    # This panel is the frozen eight-method
    # single-segment subset, not the expanded complete-path main suite.
    rank_axis.set_title("c  Frozen-subset rank stability", loc="left", fontweight="bold")
    rank_axis.legend(frameon=False)

    heat_axis = axes[1, 0]
    perturbations = list(cases)
    columns = [(block, shift) for block in PATCH_BLOCKS for shift in SHIFT_WINDOWS]
    matrix = np.empty((len(perturbations), len(columns)))
    for row_index, perturbation in enumerate(perturbations):
        for column_index, (block, shift) in enumerate(columns):
            selected = next(
                row for row in translation_rows
                if row["perturbation"] == perturbation
                and row["patch_blocks_each_axis"] == block
                and row["shift_radius_px"] == shift
            )
            matrix[row_index, column_index] = selected["translation_distance"]
    image = heat_axis.imshow(matrix, aspect="auto", cmap="magma_r")
    heat_axis.set_yticks(range(len(perturbations)), ["skyrmion shift", "topology loss", "wall shift"])
    heat_axis.set_xticks(
        range(len(columns)), [f"{block}x{block}\n+/-{shift}" for block, shift in columns], rotation=45, ha="right"
    )
    heat_axis.set_title("d  Translation-score response", loc="left", fontweight="bold")
    figure.colorbar(image, ax=heat_axis, fraction=0.045, label="distance")

    window_axis = axes[1, 1]
    for perturbation, marker in zip(perturbations, ("o", "s", "^")):
        selected = [
            row for row in translation_rows
            if row["perturbation"] == perturbation and row["patch_blocks_each_axis"] == 4
        ]
        selected.sort(key=lambda row: row["shift_radius_px"])
        window_axis.plot(
            [row["shift_radius_px"] for row in selected],
            [row["translation_distance"] for row in selected],
            marker=marker,
            label=perturbation,
        )
    window_axis.set_xticks(SHIFT_WINDOWS)
    window_axis.set_xlabel("Search radius (pixels), 4x4 patches")
    window_axis.set_ylabel("Translation distance")
    window_axis.grid(alpha=0.2)
    window_axis.legend(frameon=False)
    window_axis.set_title("e  Window sensitivity", loc="left", fontweight="bold")

    example_axis = axes[1, 2]
    reference, query = cases["topology disappearance"]
    plate = np.concatenate([reference[2], query[2]], axis=1)
    example_axis.imshow(plate, cmap="RdBu_r", vmin=-1, vmax=1)
    example_axis.axvline(255.5, color="white", linewidth=1)
    example_axis.text(0.02, 0.95, "two skyrmions", color="white", transform=example_axis.transAxes, va="top")
    example_axis.text(0.52, 0.95, "one skyrmion", color="white", transform=example_axis.transAxes, va="top")
    delta_q = next(row["absolute_delta_q"] for row in translation_rows if row["perturbation"] == "topology disappearance")
    example_axis.set_xlabel(
        "local matching cannot certify topology\n"
        rf"$|\Delta Q|={delta_q:.2f}$",
        fontsize=7.8,
    )
    example_axis.set_xticks([])
    example_axis.set_yticks([])
    example_axis.set_title("f  Topology counterexample", loc="left", fontweight="bold")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x30-root", type=Path, default=DEFAULT_X30)
    parser.add_argument("--primary-root", type=Path, help="Portable saved-frame outputs, with one directory per method.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-pool", type=int, default=40)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--pair-block", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate the figure from existing summaries without recomputing x30 trials.",
    )
    args = parser.parse_args()
    if args.primary_root is not None:
        names = {"FLARE": "flare", "Poseidon-T": "poseidon_t", "CNO-FM": "cno_fm", "DPOT-Ti": "dpot_ti", "MPP-AViT-Ti": "mpp_avit_ti", "PDEArena U-Net": "pdearena_unet", "LE-PDE": "le_pde", "NeuralMAG-x5": "neuralmag_x5"}
        for label, name in names.items():
            METHOD_ROOTS[label] = args.primary_root / name
    if args.model_pool < max(MODEL_BUDGETS):
        raise ValueError("model pool must cover the largest requested budget")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        model_summary = _json(args.output / "x30_flare_budget_summary.json")
        simulator_summary = _json(args.output / "x30_mumax_budget_summary.json")
        with (args.output / "primary_distribution_rank_stress.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rank_rows = list(csv.DictReader(handle))
        for row in rank_rows:
            for key in (
                "model_budget",
                "distribution_score",
                "score_bootstrap_95ci_low",
                "score_bootstrap_95ci_high",
                "median_rank",
                "rank_bootstrap_95ci_low",
                "rank_bootstrap_95ci_high",
                "probability_rank_1",
            ):
                row[key] = float(row[key])
        with (args.output / "translation_matching_stress.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            translation_rows = list(csv.DictReader(handle))
        for row in translation_rows:
            for key in (
                "patch_blocks_each_axis",
                "shift_radius_px",
                "translation_distance",
                "pixel_angular_error_deg",
                "pixel_mse",
                "q_reference",
                "q_query",
                "absolute_delta_q",
                "valid_patches",
            ):
                row[key] = float(row[key])
        cases = _translation_cases()
        _plot(
            model_summary=model_summary,
            simulator_summary=simulator_summary,
            rank_rows=rank_rows,
            translation_rows=translation_rows,
            cases=cases,
            output=args.figure,
        )
        print(args.figure)
        return
    model_rows, simulator_rows = _x30_budget_rows(
        args.x30_root,
        output=args.output,
        device=device,
        pair_block=args.pair_block,
        pool_size=args.model_pool,
        trials=args.trials,
        seed=args.seed,
    )
    _write_csv(args.output / "x30_flare_budget_trials.csv", model_rows)
    _write_csv(args.output / "x30_mumax_budget_trials.csv", simulator_rows)
    model_summary = _summarise_budgets(
        model_rows,
        "model_budget",
        bootstrap=args.bootstrap,
        seed=args.seed + 101,
    )
    simulator_summary = _summarise_budgets(
        simulator_rows,
        "simulator_budget_each_side",
        bootstrap=args.bootstrap,
        seed=args.seed + 202,
    )
    _write_json(args.output / "x30_flare_budget_summary.json", model_summary)
    _write_json(args.output / "x30_mumax_budget_summary.json", simulator_summary)

    rank_rows, rank_payload = _primary_rank_stress(
        bootstrap=args.bootstrap,
        seed=args.seed + 303,
    )
    _write_csv(args.output / "primary_distribution_rank_stress.csv", rank_rows)
    _write_json(args.output / "primary_distribution_rank_stress.json", rank_payload)

    translation_rows, cases = _translation_stress(device)
    _write_csv(args.output / "translation_matching_stress.csv", translation_rows)
    _plot(
        model_summary=model_summary,
        simulator_summary=simulator_summary,
        rank_rows=rank_rows,
        translation_rows=translation_rows,
        cases=cases,
        output=args.figure,
    )

    summary = {
        "status": "complete",
        "x30": {
            "conditions": 34,
            "base_groups": 17,
            "independent_mumax_repeats_per_condition": 30,
            "flare_persisted_samples_per_condition": 128,
            "flare_deterministic_pool_size": args.model_pool,
            "subsampling_trials_per_condition": args.trials,
            "flare_model_budgets": MODEL_BUDGETS,
            "mumax_disjoint_budgets": SIMULATOR_BUDGETS,
            "flare": model_summary,
            "mumax": simulator_summary,
        },
        "rank_stress": rank_payload,
        "translation_matching": {
            "patch_grids": [f"{value}x{value}" for value in PATCH_BLOCKS],
            "shift_radii_px": SHIFT_WINDOWS,
            "perturbations": list(cases),
            "rows": translation_rows,
        },
        "uncertainty": "base-condition clustered bootstrap; both segments retained",
        "figure": args.figure,
    }
    _write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
