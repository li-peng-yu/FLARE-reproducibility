#!/usr/bin/env python3
"""Build the paper's objectively selected Generative Necessity figure.

The script has two deliberately separate stages:

``select``
    Scores every condition in the frozen x30 audit without rendering any
    magnetic field, selects four distinct bases with a prespecified
    phenomenon-coverage objective, and writes a complete selection audit.

``render``
    Uses the frozen MuMax3/FLARE arrays plus the exact anchor and Poseidon-T
    output exported by ``export_generative_necessity_poseidon.py``.  Six
    displayed outcomes are facility-location medoids in physical-observable
    space; no image is chosen by eye.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.analyze_x5_distribution_sensitivity_observables import (  # noqa: E402
    _observables as _paper_observables,
)

DEFAULT_AUDIT = (
    PROJECT
    / "outputs/skx_bt_165base_x30/evaluation_20260828/"
    "standard_prior50k_on_x30_test"
)
DEFAULT_OUTPUT = PROJECT / "outputs/paper_artifacts/generative_necessity"
DEFAULT_PAPER_FIG = PROJECT / "outputs/figures"
SCORE_SUBDIR = "same_condition_score_4x4_shift16"
PHENOMENON_SLOTS = (
    "skyrmion_displacement",
    "domain_wall_shift",
    "topology_change",
    "broad_outcome_spread",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise(array: np.ndarray) -> np.ndarray:
    fields = np.asarray(array, dtype=np.float32)
    return fields / np.maximum(
        np.linalg.norm(fields, axis=1, keepdims=True), 1.0e-8
    )


def _weighted_centroid(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = weights.shape[-2:]
    yy, xx = np.mgrid[0:height, 0:width]
    mass = weights.sum(axis=(-1, -2))
    safe = np.maximum(mass, 1.0e-12)
    cx = (weights * xx).sum(axis=(-1, -2)) / safe
    cy = (weights * yy).sum(axis=(-1, -2)) / safe
    cx = np.where(mass > 1.0e-7, cx, np.nan)
    cy = np.where(mass > 1.0e-7, cy, np.nan)
    return cx.astype(np.float64), cy.astype(np.float64)


def _field_features(array: np.ndarray, mask_array: np.ndarray) -> dict[str, np.ndarray]:
    """Observable definitions match the paper's existing physical audit."""
    # Computing every intermediate map for all 128 FLARE draws at once exceeds
    # the login-node memory limit. Chunking is algebraically identical because
    # each observable is computed independently for each sample.
    sample_count = int(array.shape[0])
    if sample_count > 16:
        chunks = [
            _field_features(array[start : start + 16], mask_array)
            for start in range(0, sample_count, 16)
        ]
        return {
            name: np.concatenate([chunk[name] for chunk in chunks], axis=0)
            for name in chunks[0]
        }
    paper_observables = _paper_observables(array, mask_array)
    fields = _normalise(array)
    mask = np.asarray(mask_array, dtype=np.bool_)
    fields = fields * mask[None, None]
    valid = max(int(mask.sum()), 1)

    padded_x = np.pad(fields, ((0, 0), (0, 0), (0, 0), (1, 1)), mode="edge")
    padded_y = np.pad(fields, ((0, 0), (0, 0), (1, 1), (0, 0)), mode="edge")
    dmx = 0.5 * (padded_x[..., 2:] - padded_x[..., :-2])
    dmy = 0.5 * (padded_y[..., 2:, :] - padded_y[..., :-2, :])
    q_density = np.einsum(
        "nchw,nchw->nhw",
        fields,
        np.cross(dmx, dmy, axisa=1, axisb=1, axisc=1),
        optimize=True,
    ) / (4.0 * np.pi)
    charge = paper_observables["topological_charge"]
    mean_mz = paper_observables["mean_mz"]

    horizontal_mask = mask[:, 1:] & mask[:, :-1]
    vertical_mask = mask[1:, :] & mask[:-1, :]
    horizontal = 1.0 - (fields[..., 1:] * fields[..., :-1]).sum(axis=1)
    vertical = 1.0 - (fields[..., 1:, :] * fields[..., :-1, :]).sum(axis=1)
    horizontal_sum = (horizontal * horizontal_mask).sum(axis=(-1, -2))
    vertical_sum = (vertical * vertical_mask).sum(axis=(-1, -2))
    bonds = max(int(horizontal_mask.sum() + vertical_mask.sum()), 1)
    exchange = paper_observables["exchange_texture_energy"]

    # Domain-wall position is measured with the standard 1-m_z^2 wall weight.
    wall_weight = np.maximum(1.0 - np.square(fields[:, 2]), 0.0) * mask[None]
    wall_cx, wall_cy = _weighted_centroid(wall_weight)
    wall_mass = wall_weight.sum(axis=(-1, -2)) / valid

    # The absolute topological-density centroid tracks skyrmion transport even
    # when the signed charge is negative.
    q_weight = np.abs(q_density) * mask[None]
    q_cx, q_cy = _weighted_centroid(q_weight)
    q_abs_mass = q_weight.sum(axis=(-1, -2))

    # Put nearest-neighbour exchange energy back on sites to obtain a generic
    # texture centroid.  This remains informative for non-skyrmionic walls.
    texture = np.zeros_like(q_density, dtype=np.float32)
    texture[..., :, 1:] += 0.5 * horizontal * horizontal_mask
    texture[..., :, :-1] += 0.5 * horizontal * horizontal_mask
    texture[..., 1:, :] += 0.5 * vertical * vertical_mask
    texture[..., :-1, :] += 0.5 * vertical * vertical_mask
    texture_cx, texture_cy = _weighted_centroid(texture)

    return {
        "topological_charge": np.asarray(charge, dtype=np.float64),
        "mean_mz": np.asarray(mean_mz, dtype=np.float64),
        "exchange_texture_energy": np.asarray(exchange, dtype=np.float64),
        "q_centroid_x": q_cx,
        "q_centroid_y": q_cy,
        "q_abs_mass": np.asarray(q_abs_mass, dtype=np.float64),
        "wall_centroid_x": wall_cx,
        "wall_centroid_y": wall_cy,
        "wall_mass": np.asarray(wall_mass, dtype=np.float64),
        "texture_centroid_x": texture_cx,
        "texture_centroid_y": texture_cy,
    }


def _upper_mean(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=np.float64)
    indices = np.triu_indices(values.shape[0], k=1)
    return float(values[indices].mean())


def _centroid_dispersion(x: np.ndarray, y: np.ndarray) -> float:
    points = np.stack((x, y), axis=1)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 2:
        return 0.0
    delta = points[:, None] - points[None]
    distance = np.linalg.norm(delta, axis=-1)
    return _upper_mean(distance)


def _entropy_of_rounded_charge(charge: np.ndarray) -> float:
    _, counts = np.unique(np.rint(charge).astype(np.int64), return_counts=True)
    if len(counts) <= 1:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(len(counts)))


def _condition_summary(condition: Path) -> dict[str, Any]:
    metadata = _json(condition / "condition_metadata.json")
    score = condition / SCORE_SUBDIR
    mask = np.load(score / "geometry_mask.npy", mmap_mode="r")
    truth = np.load(score / "mumax_targets_f16.npy", mmap_mode="r")
    features = _field_features(truth, mask)
    charge = features["topological_charge"]
    mean_mz = features["mean_mz"]
    exchange = features["exchange_texture_energy"]
    distance = np.load(score / "truth_self_patch_shift_distance.npy", mmap_mode="r")
    return {
        "condition_id": condition.name,
        "base_id": str(metadata["base_id"]).zfill(4),
        "segment_role": str(metadata["segment_role"]),
        "control_segment_index": int(metadata["control_segment_index"]),
        "horizon_ns": float(metadata["horizon_ns"]),
        "run_id": str(metadata["repeats"][0]["run_id"]),
        "num_mumax": int(len(truth)),
        "field_pairwise_mean": _upper_mean(distance),
        "q_mean_abs": float(np.mean(np.abs(charge))),
        "q_std": float(np.std(charge, ddof=1)),
        "q_range": float(np.ptp(charge)),
        "q_rounded_states": int(len(np.unique(np.rint(charge)))),
        "q_state_entropy": _entropy_of_rounded_charge(charge),
        "mz_std": float(np.std(mean_mz, ddof=1)),
        "ex_std": float(np.std(exchange, ddof=1)),
        "q_centroid_dispersion_px": _centroid_dispersion(
            features["q_centroid_x"], features["q_centroid_y"]
        ),
        "wall_centroid_dispersion_px": _centroid_dispersion(
            features["wall_centroid_x"], features["wall_centroid_y"]
        ),
        "texture_centroid_dispersion_px": _centroid_dispersion(
            features["texture_centroid_x"], features["texture_centroid_y"]
        ),
        "wall_mass_std": float(np.std(features["wall_mass"], ddof=1)),
    }


def _rank_percentiles(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    result = np.empty_like(array)
    result[order] = np.linspace(0.0, 1.0, len(array))
    return result


def _add_scores(rows: list[dict[str, Any]]) -> None:
    metric_names = (
        "field_pairwise_mean",
        "q_std",
        "q_range",
        "q_state_entropy",
        "mz_std",
        "ex_std",
        "q_centroid_dispersion_px",
        "wall_centroid_dispersion_px",
        "texture_centroid_dispersion_px",
        "wall_mass_std",
    )
    ranks: dict[str, np.ndarray] = {}
    for name in metric_names:
        ranks[name] = _rank_percentiles([float(row[name]) for row in rows])
    for index, row in enumerate(rows):
        rank = {name: float(values[index]) for name, values in ranks.items()}
        # A topological-density centroid is only called skyrmion displacement
        # when the ensemble carries non-negligible topological density.
        topology_gate = min(float(row["q_mean_abs"]) / 0.25, 1.0)
        row["skyrmion_displacement_score"] = topology_gate * float(
            np.mean(
                [
                    rank["q_centroid_dispersion_px"],
                    rank["texture_centroid_dispersion_px"],
                    rank["field_pairwise_mean"],
                ]
            )
        )
        row["domain_wall_shift_score"] = float(
            np.mean(
                [
                    rank["wall_centroid_dispersion_px"],
                    rank["wall_mass_std"],
                    rank["mz_std"],
                    rank["field_pairwise_mean"],
                ]
            )
        )
        row["topology_change_score"] = float(
            np.mean(
                [
                    rank["q_std"],
                    rank["q_range"],
                    rank["q_state_entropy"],
                    rank["field_pairwise_mean"],
                ]
            )
        )
        row["broad_outcome_spread_score"] = float(
            np.mean(
                [
                    rank["field_pairwise_mean"],
                    rank["q_std"],
                    rank["mz_std"],
                    rank["ex_std"],
                    rank["q_centroid_dispersion_px"],
                    rank["wall_centroid_dispersion_px"],
                    rank["wall_mass_std"],
                ]
            )
        )
        row["scatter_y"] = (
            "mean_mz" if rank["mz_std"] >= rank["ex_std"] else "exchange_texture_energy"
        )


def _select_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Globally assign four unique bases to four prespecified evidence slots."""
    # Only segment 1 starts from the base's one shared-relax checkpoint.
    # Segment 2 starts from a repeat-specific stochastic drive endpoint.
    exact_anchor_indices = [
        index
        for index, row in enumerate(rows)
        if int(row["control_segment_index"]) == 1
        and str(row["segment_role"]) == "drive"
    ]
    if len(exact_anchor_indices) != 17:
        raise RuntimeError(
            f"expected 17 exact-anchor x30 drive conditions, found {len(exact_anchor_indices)}"
        )
    broad = np.asarray(
        [rows[index]["broad_outcome_spread_score"] for index in exact_anchor_indices]
    )
    floor = float(np.quantile(broad, 0.40))
    candidates: dict[str, list[int]] = {}
    for slot in PHENOMENON_SLOTS:
        key = f"{slot}_score"
        eligible = [
            index
            for index in exact_anchor_indices
            if float(rows[index]["broad_outcome_spread_score"]) >= floor
        ]
        candidates[slot] = sorted(
            eligible,
            key=lambda index: (-float(rows[index][key]), rows[index]["condition_id"]),
        )[:12]

    best: tuple[float, tuple[str, ...], tuple[int, ...]] | None = None
    for assignment in itertools.product(*(candidates[slot] for slot in PHENOMENON_SLOTS)):
        selected = [rows[index] for index in assignment]
        if len(set(assignment)) != len(assignment):
            continue
        if len({row["base_id"] for row in selected}) != len(selected):
            continue
        score = 0.0
        for slot, row in zip(PHENOMENON_SLOTS, selected, strict=True):
            score += float(row[f"{slot}_score"])
            score += 0.25 * float(row["broad_outcome_spread_score"])
        identifiers = tuple(row["condition_id"] for row in selected)
        candidate = (score, tuple(reversed(identifiers)), tuple(assignment))
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("no four-case assignment satisfies the fixed diversity constraints")
    selected_rows: list[dict[str, Any]] = []
    for slot, index in zip(PHENOMENON_SLOTS, best[2], strict=True):
        row = dict(rows[index])
        row["selection_slot"] = slot
        selected_rows.append(row)
    return selected_rows


def select(audit_root: Path, output_dir: Path) -> dict[str, Any]:
    conditions = sorted(
        path
        for path in (audit_root / "conditions").iterdir()
        if path.is_dir()
        and (path / "condition_metadata.json").is_file()
        and (path / SCORE_SUBDIR / "mumax_targets_f16.npy").is_file()
    )
    if len(conditions) != 34:
        raise RuntimeError(f"expected 34 complete x30 conditions, found {len(conditions)}")
    rows = [_condition_summary(condition) for condition in conditions]
    _add_scores(rows)
    selected = _select_cases(rows)

    _write_csv(output_dir / "all_condition_scores.csv", rows)
    manifest = {
        "schema": "flare_generative_necessity_selection_v1",
        "status": "selected",
        "policy": (
            "No field was rendered before selection. All 34 x30 test conditions were rank-scored "
            "from MuMax3-only endpoint spreads; eligibility was restricted to the 17 segment-1 "
            "drive conditions that start from each base's single shared-relax checkpoint. Four "
            "distinct bases were globally assigned to "
            "prespecified skyrmion-displacement, domain-wall-shift, topology-change, and broad-spread "
            "slots. Post-relaxation conditions were excluded because their anchors are stochastic "
            "drive endpoints rather than one exact field."
        ),
        "scored_condition_count": len(rows),
        "exact_anchor_candidate_count": sum(
            int(row["control_segment_index"]) == 1
            and str(row["segment_role"]) == "drive"
            for row in rows
        ),
        "mumax_repeats_per_condition": 30,
        "flare_draws_per_condition": 30,
        "flare_draw_policy": (
            "thirty fresh ODE-10 calls from explicit fixed per-draw seeds; "
            "the older 128-draw cache is neither read nor subsampled"
        ),
        "selection_slots": list(PHENOMENON_SLOTS),
        "broad_spread_eligibility_quantile": 0.40,
        "slot_candidate_pool_size": 12,
        "audit_root": str(audit_root.resolve()),
        "flare_source": {
            "table1_original_checkpoint_sha256": (
                "ab658f5750b3208c69b0fb5a681ef77d59441294cd60c37f545f85ee1b7128ac"
            ),
            "checkpoint_step": 50000,
            "checkpoint_state": "ema",
            "released_checkpoint": str(
                Path(os.environ.get("FLARE_CHECKPOINT_ROOT", str(PROJECT.parent / "FLARE_checkpoints"))) / "flare/core_seed78.pt"
            ),
            "released_checkpoint_sha256": _sha256(
                Path(os.environ.get("FLARE_CHECKPOINT_ROOT", str(PROJECT.parent / "FLARE_checkpoints"))) / "flare/core_seed78.pt"
            ),
        },
        "poseidon_source": {
            "released_checkpoint": str(
                Path(os.environ.get("FLARE_CHECKPOINT_ROOT", str(PROJECT.parent / "FLARE_checkpoints"))) / "baselines/poseidon_t.pt"
            ),
            "checkpoint_sha256": _sha256(
                Path(os.environ.get("FLARE_CHECKPOINT_ROOT", str(PROJECT.parent / "FLARE_checkpoints"))) / "baselines/poseidon_t.pt"
            ),
        },
        "selected_cases": selected,
    }
    _write_json(output_dir / "selection_manifest.json", manifest)
    print(json.dumps({"selected": [row["condition_id"] for row in selected]}, indent=2))
    return manifest


def audit_physics(audit_root: Path, output_dir: Path) -> dict[str, Any]:
    """Assign conservative labels and choose two main-text cases objectively."""
    selection = _json(output_dir / "selection_manifest.json")
    inference = _json(output_dir / "raw_export_manifest.json")
    if inference.get("fresh_flare_draws_per_case") != 30 or inference.get(
        "cached_flare_draws_used"
    ):
        raise RuntimeError("physical audit requires 30 freshly generated FLARE draws")

    audited: list[dict[str, Any]] = []
    for case in selection["selected_cases"]:
        condition_id = str(case["condition_id"])
        score_dir = audit_root / "conditions" / condition_id / SCORE_SUBDIR
        mask = np.load(score_dir / "geometry_mask.npy", mmap_mode="r")
        truth = np.load(score_dir / "mumax_targets_f16.npy", mmap_mode="r")
        raw_dir = output_dir / "raw" / condition_id
        flare = np.load(raw_dir / "flare_30_f16.npy", mmap_mode="r")
        baseline = np.load(raw_dir / "poseidon_t_f16.npy", mmap_mode="r")
        if truth.shape != (30, 3, 256, 256) or flare.shape != truth.shape:
            raise RuntimeError(
                f"expected matched 30-outcome ensembles for {condition_id}, "
                f"found truth={truth.shape}, FLARE={flare.shape}"
            )
        truth_features = _field_features(truth, mask)
        flare_features = _field_features(flare, mask)
        baseline_features = _field_features(baseline[None], mask)
        q = truth_features["topological_charge"]
        rounded, counts = np.unique(np.rint(q).astype(np.int64), return_counts=True)
        dominant_fraction = float(np.max(counts) / len(q))
        q_interdecile = float(np.quantile(q, 0.90) - np.quantile(q, 0.10))
        wall_dispersion = _centroid_dispersion(
            truth_features["wall_centroid_x"], truth_features["wall_centroid_y"]
        )
        mean_mz_range = float(np.ptp(truth_features["mean_mz"]))

        topology_count_evidence = bool(
            len(rounded) >= 4
            and q_interdecile >= 1.5
            and int(np.sum(counts >= 3)) >= 3
        )
        approximately_preserved_q = bool(dominant_fraction >= 0.80)
        domain_pattern_evidence = bool(
            approximately_preserved_q
            and wall_dispersion >= 1.0
            and mean_mz_range >= 0.05
        )
        if topology_count_evidence:
            physical_label = "Topology-count variability"
            label_basis = (
                "at least four rounded-Q groups, three populated by at least "
                "three repeats, and Q interdecile range >= 1.5"
            )
        elif domain_pattern_evidence:
            physical_label = "Domain-pattern variability"
            label_basis = (
                ">=80% of outcomes retain one rounded-Q group while wall-weight "
                "centroid and mean-mz vary above fixed thresholds"
            )
        else:
            physical_label = "Texture and Q variability"
            label_basis = (
                "outcome spread is verified, but displacement, domain-wall shift, "
                "or a clean topology-count label is not uniquely established"
            )

        observable_audit: dict[str, Any] = {}
        for name in (
            "topological_charge",
            "mean_mz",
            "exchange_texture_energy",
        ):
            mumax_values = truth_features[name]
            flare_values = flare_features[name]
            poseidon_value = float(baseline_features[name][0])
            observable_audit[name] = {
                "mumax_min": float(np.min(mumax_values)),
                "mumax_max": float(np.max(mumax_values)),
                "mumax_std": float(np.std(mumax_values, ddof=1)),
                "flare_min": float(np.min(flare_values)),
                "flare_max": float(np.max(flare_values)),
                "flare_std": float(np.std(flare_values, ddof=1)),
                "flare_vs_mumax_w1_equal_30": float(
                    np.mean(np.abs(np.sort(flare_values) - np.sort(mumax_values)))
                ),
                "poseidon": poseidon_value,
                "poseidon_outside_mumax_range": bool(
                    poseidon_value < float(np.min(mumax_values))
                    or poseidon_value > float(np.max(mumax_values))
                ),
                "poseidon_nearest_mumax_abs": float(
                    np.min(np.abs(mumax_values - poseidon_value))
                ),
            }

        audited.append(
            {
                **case,
                "physical_label": physical_label,
                "physical_label_basis": label_basis,
                "rounded_q_counts": {
                    str(int(value)): int(count)
                    for value, count in zip(rounded, counts, strict=True)
                },
                "dominant_rounded_q_fraction": dominant_fraction,
                "q_interdecile_range": q_interdecile,
                "wall_centroid_pairwise_dispersion_px": wall_dispersion,
                "mean_mz_range": mean_mz_range,
                "topology_count_evidence": topology_count_evidence,
                "approximately_preserved_q": approximately_preserved_q,
                "domain_pattern_evidence": domain_pattern_evidence,
                "observables": observable_audit,
            }
        )

    domain_candidates = [row for row in audited if row["domain_pattern_evidence"]]
    topology_candidates = [row for row in audited if row["topology_count_evidence"]]
    if not domain_candidates or not topology_candidates:
        raise RuntimeError(
            "the fixed physical-label audit did not find both a domain-pattern "
            "and a topology-count case"
        )
    domain_case = max(
        domain_candidates,
        key=lambda row: (float(row["domain_wall_shift_score"]), row["condition_id"]),
    )
    topology_case = max(
        topology_candidates,
        key=lambda row: (float(row["topology_change_score"]), row["condition_id"]),
    )
    if domain_case["condition_id"] == topology_case["condition_id"]:
        raise RuntimeError("main-text cases must be distinct")

    main_caption = (
        "Stochastic outcome variability under identical physical inputs. Each row fixes "
        "the exact magnetization anchor, control condition, and target horizon. Thirty "
        "independent MuMax3 outcomes and thirty fresh fixed-seed FLARE ODE-10 draws form "
        "observable point clouds, whereas deterministic Poseidon-T returns one endpoint. "
        "The six displayed fields from each ensemble are selected by the same deterministic "
        "facility-location medoid rule in a common physical-feature space; scatter plots "
        "contain all 30 outcomes. The two cases were selected by predeclared MuMax3-only "
        "tests for domain-pattern variability and topology-count variability. Point-cloud "
        "spread is interpreted as stochastic outcome variability, without assuming distinct modes."
    )
    appendix_caption = (
        "Extended exact-anchor audit of stochastic outcome variability. For four objectively "
        "screened held-out conditions, each row compares 30 MuMax3 outcomes, 30 fresh fixed-seed "
        "FLARE ODE-10 draws, and the single output of deterministic Poseidon-T under the same "
        "anchor, physical controls, and horizon. Six fields per stochastic ensemble are chosen "
        "with one shared deterministic medoid rule, while every outcome is included in the "
        "observable scatter. Labels follow fixed physical diagnostics and remain neutral where "
        "a unique displacement or topology mechanism is not established; no multimodality claim "
        "is made."
    )
    payload = {
        "schema": "flare_generative_necessity_physical_audit_v1",
        "status": "passed",
        "label_policy": {
            "topology_count": (
                "rounded-Q groups >= 4; Q interdecile range >= 1.5; "
                "at least three groups contain >= 3 repeats"
            ),
            "domain_pattern": (
                "dominant rounded-Q fraction >= 0.80; wall-centroid pairwise "
                "dispersion >= 1 px; mean-mz range >= 0.05"
            ),
            "fallback": "Texture and Q variability (mechanism-neutral)",
        },
        "main_case_selection_policy": (
            "highest original MuMax3-only domain-wall score among verified domain-pattern "
            "cases plus highest original MuMax3-only topology score among verified "
            "topology-count cases"
        ),
        "main_condition_ids": [
            domain_case["condition_id"],
            topology_case["condition_id"],
        ],
        "appendix_condition_ids": [row["condition_id"] for row in audited],
        "cases": audited,
        "main_caption": main_caption,
        "appendix_caption": appendix_caption,
    }
    _write_json(output_dir / "physical_audit_manifest.json", payload)
    _write_text(
        output_dir / "generative_necessity_captions.txt",
        "MAIN-PAPER CAPTION\n\n"
        + main_caption
        + "\n\nAPPENDIX CAPTION\n\n"
        + appendix_caption
        + "\n",
    )
    print(
        json.dumps(
            {
                "main": payload["main_condition_ids"],
                "labels": {
                    row["condition_id"]: row["physical_label"] for row in audited
                },
            },
            indent=2,
        )
    )
    return payload


def _feature_matrix(features: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack(
        [
            features["topological_charge"],
            features["mean_mz"],
            features["exchange_texture_energy"],
            features["q_centroid_x"] / 256.0,
            features["q_centroid_y"] / 256.0,
            features["wall_centroid_x"] / 256.0,
            features["wall_centroid_y"] / 256.0,
            features["wall_mass"],
        ],
        axis=1,
    )


def _scaled_feature_matrix(matrix: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    median = np.nanmedian(reference, axis=0)
    reference_scale = np.nanstd(reference, axis=0)
    q25, q75 = np.nanpercentile(reference, [25.0, 75.0], axis=0)
    scale = np.maximum(q75 - q25, 0.10 * reference_scale)
    scale = np.maximum(scale, np.asarray([0.05, 0.002, 1e-5, 0.01, 0.01, 0.01, 0.01, 0.002]))
    filled = np.where(np.isfinite(matrix), matrix, median[None])
    return (filled - median[None]) / scale[None]


def _facility_medoids(matrix: np.ndarray, count: int = 6) -> list[int]:
    """Deterministic greedy facility location followed by exact one-swaps."""
    if len(matrix) < count:
        raise ValueError(f"need at least {count} samples, found {len(matrix)}")
    distance = np.linalg.norm(matrix[:, None] - matrix[None], axis=-1)
    selected = [int(np.argmin(distance.sum(axis=1)))]
    minimum = distance[:, selected[0]].copy()
    while len(selected) < count:
        costs = np.full(len(matrix), np.inf)
        for candidate in range(len(matrix)):
            if candidate not in selected:
                costs[candidate] = np.minimum(minimum, distance[:, candidate]).sum()
        chosen = int(np.argmin(costs))
        selected.append(chosen)
        minimum = np.minimum(minimum, distance[:, chosen])

    # Deterministic PAM-style one-swap refinement.
    while True:
        current_cost = float(distance[:, selected].min(axis=1).sum())
        best_cost = current_cost
        best: tuple[int, int] | None = None
        selected_set = set(selected)
        for position in range(count):
            for candidate in range(len(matrix)):
                if candidate in selected_set:
                    continue
                proposal = list(selected)
                proposal[position] = candidate
                cost = float(distance[:, proposal].min(axis=1).sum())
                if cost < best_cost - 1.0e-10:
                    best_cost = cost
                    best = (position, candidate)
        if best is None:
            break
        selected[best[0]] = best[1]
    return selected


def _run_context(run_id: str, segment: str, horizon_ns: float) -> str:
    match = re.search(
        r"_T(?P<t>\d+)K_.*?_B(?P<b>\d+)mT_.*?_J(?P<j>[^_]+)_", run_id
    )
    if match is None:
        return f"{segment.replace('_', ' ')}, $\\Delta T={horizon_ns:.2f}$ ns"
    encoded = match.group("j")
    encoded = encoded.replace("p", ".", 1).replace("ep", "e+").replace("em", "e-")
    try:
        current = float(encoded)
        current_text = f"{current / 1e12:.2f}"
    except ValueError:
        current_text = encoded
    return (
        f"{segment.replace('_', ' ')} · $T={int(match.group('t'))}$ K · "
        f"$B_z={int(match.group('b'))}$ mT · $J={current_text}\\times10^{{12}}$ A m$^{{-2}}$ · "
        f"$\\Delta T={horizon_ns:.2f}$ ns"
    )


def _image_axis(
    axis: plt.Axes,
    field: np.ndarray,
    title: str | None = None,
    *,
    title_fontsize: float = 7.1,
) -> Any:
    image = axis.imshow(
        np.asarray(field)[2], origin="lower", cmap="RdBu_r", vmin=-1.0, vmax=1.0,
        interpolation="nearest", rasterized=True,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(0.45)
        spine.set_color("#4a4a4a")
    if title:
        axis.set_title(title, fontsize=title_fontsize, pad=2.0)
    return image


def _group_images(
    figure: plt.Figure,
    subplot_spec: Any,
    fields: np.ndarray,
    indices: list[int],
    title: str,
    *,
    rows: int,
    columns: int,
    title_fontsize: float,
) -> None:
    if rows * columns != len(indices):
        raise ValueError(
            f"representative grid {rows}x{columns} does not fit {len(indices)} fields"
        )
    grid = subplot_spec.subgridspec(rows, columns, wspace=0.035, hspace=0.035)
    for position, sample_index in enumerate(indices):
        axis = figure.add_subplot(grid[position // columns, position % columns])
        _image_axis(axis, fields[sample_index])
    box = subplot_spec.get_position(figure)
    title_offset = 1.0 / (72.0 * figure.get_figheight())
    figure.text(
        0.5 * (box.x0 + box.x1), box.y1 + title_offset, title,
        fontsize=title_fontsize, ha="center", va="bottom",
    )


def _scatter(
    axis: plt.Axes,
    truth: dict[str, np.ndarray],
    flare: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    y_name: str,
    legend: bool,
) -> None:
    y_label = r"$\bar m_z$" if y_name == "mean_mz" else r"$E_{\mathrm{ex}}$"
    axis.scatter(
        truth["topological_charge"], truth[y_name], s=15, c="#3f3f3f",
        alpha=0.72, linewidths=0, label="MuMax3 (30)", zorder=2,
    )
    axis.scatter(
        flare["topological_charge"], flare[y_name], s=13, c="#1b9e77",
        alpha=0.58, linewidths=0, label="FLARE (30)", zorder=1,
    )
    axis.scatter(
        baseline["topological_charge"], baseline[y_name], s=75, marker="*",
        c="#d95f02", edgecolors="white", linewidths=0.45,
        label="Poseidon-T (1)", zorder=4,
    )
    axis.set_xlabel(r"Topological charge $Q$", fontsize=7.1, labelpad=1.2)
    axis.set_ylabel(y_label, fontsize=7.1, labelpad=0.8)
    axis.tick_params(axis="both", labelsize=6.1, length=2.0, width=0.55, pad=1.2)
    axis.grid(True, color="#d8d8d8", linewidth=0.4, alpha=0.7, zorder=0)
    for spine in axis.spines.values():
        spine.set_linewidth(0.55)
    if legend:
        axis.legend(
            loc="best", fontsize=5.5, frameon=True, framealpha=0.90,
            borderpad=0.35, handletextpad=0.35, labelspacing=0.25,
        )


def render(
    audit_root: Path,
    output_dir: Path,
    paper_fig_dir: Path,
    *,
    version: str,
    figure_output_dir: Path | None = None,
    copy_to_paper: bool = True,
    main_scatter_y: str | None = None,
) -> dict[str, Any]:
    if version not in {"main", "appendix"}:
        raise ValueError(version)
    selection_path = output_dir / "selection_manifest.json"
    selection = _json(selection_path)
    physical_audit = _json(output_dir / "physical_audit_manifest.json")
    cases_by_id = {
        str(case["condition_id"]): case for case in physical_audit["cases"]
    }
    condition_ids = physical_audit[
        "main_condition_ids" if version == "main" else "appendix_condition_ids"
    ]
    cases = [cases_by_id[str(condition_id)] for condition_id in condition_ids]
    expected_cases = 2 if version == "main" else 4
    if len(cases) != expected_cases:
        raise RuntimeError(
            f"the {version} layout requires {expected_cases} cases, found {len(cases)}"
        )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    # Render at the physical width used by a two-column paper.  Oversized
    # canvases later shrunk by LaTeX also shrink typography into illegibility.
    figure_height = 5.20 if version == "main" else 7.40
    figure = plt.figure(figsize=(7.15, figure_height), constrained_layout=False)
    representative_rows, representative_columns = (
        (3, 2) if version == "main" else (2, 3)
    )
    outer = figure.add_gridspec(
        len(cases), 6, left=0.060, right=0.985,
        bottom=0.050 if version == "main" else 0.055,
        top=0.940,
        width_ratios=(1.0, 2.05, 2.05, 1.0, 0.48, 2.10)
        if version == "main"
        else (1.0, 2.35, 2.35, 1.0, 0.48, 2.05),
        hspace=0.20 if version == "main" else 0.30,
        wspace=0.10 if version == "main" else 0.12,
    )
    render_cases: list[dict[str, Any]] = []
    last_image = None
    first_anchor_axis: plt.Axes | None = None

    for row_index, case in enumerate(cases):
        condition_id = str(case["condition_id"])
        condition = audit_root / "conditions" / condition_id
        score = condition / SCORE_SUBDIR
        truth_array = np.load(score / "mumax_targets_f16.npy", mmap_mode="r")
        mask = np.load(score / "geometry_mask.npy", mmap_mode="r")
        raw_dir = output_dir / "raw" / condition_id
        flare_array = np.load(raw_dir / "flare_30_f16.npy", mmap_mode="r")
        anchor = np.load(raw_dir / "anchor_f16.npy")
        baseline_array = np.load(raw_dir / "poseidon_t_f16.npy")
        if (
            truth_array.shape != (30, 3, 256, 256)
            or flare_array.shape != truth_array.shape
            or anchor.shape != (3, 256, 256)
            or baseline_array.shape != (3, 256, 256)
        ):
            raise RuntimeError(
                f"wrong field shape for {condition_id}: truth={truth_array.shape}, "
                f"FLARE={flare_array.shape}, anchor={anchor.shape}, "
                f"Poseidon={baseline_array.shape}"
            )

        truth_features = _field_features(truth_array, mask)
        flare_features = _field_features(flare_array, mask)
        baseline_features = _field_features(baseline_array[None], mask)
        truth_matrix = _feature_matrix(truth_features)
        flare_matrix = _feature_matrix(flare_features)
        # Both ensembles use the same feature transform, fitted once to the
        # pooled 30+30 cloud, and the same deterministic medoid algorithm.
        pooled_matrix = np.concatenate((truth_matrix, flare_matrix), axis=0)
        truth_indices = _facility_medoids(
            _scaled_feature_matrix(truth_matrix, pooled_matrix), 6
        )
        flare_indices = _facility_medoids(
            _scaled_feature_matrix(flare_matrix, pooled_matrix), 6
        )

        anchor_axis = figure.add_subplot(outer[row_index, 0])
        if first_anchor_axis is None:
            first_anchor_axis = anchor_axis
        panel_title_fontsize = 7.1 if version == "main" else 6.6
        group_title_fontsize = 7.3 if version == "main" else 6.8
        last_image = _image_axis(
            anchor_axis,
            anchor,
            "Exact anchor",
            title_fontsize=panel_title_fontsize,
        )
        _group_images(
            figure,
            outer[row_index, 1],
            truth_array,
            truth_indices,
            "MuMax3 outcomes (30)",
            rows=representative_rows,
            columns=representative_columns,
            title_fontsize=group_title_fontsize,
        )
        _group_images(
            figure,
            outer[row_index, 2],
            flare_array,
            flare_indices,
            "FLARE draws (30)",
            rows=representative_rows,
            columns=representative_columns,
            title_fontsize=group_title_fontsize,
        )
        baseline_axis = figure.add_subplot(outer[row_index, 3])
        _image_axis(
            baseline_axis,
            baseline_array,
            "Poseidon-T",
            title_fontsize=panel_title_fontsize,
        )
        scatter_axis = figure.add_subplot(outer[row_index, 5])
        rendered_scatter_y = (
            main_scatter_y
            if version == "main" and main_scatter_y is not None
            else str(case["scatter_y"])
        )
        _scatter(
            scatter_axis,
            truth_features,
            flare_features,
            baseline_features,
            rendered_scatter_y,
            legend=row_index == 0,
        )
        if row_index == 0:
            scatter_axis.set_title("Observable distribution", fontsize=8.2, pad=2.5)

        physical_label = str(case["physical_label"])
        context = _run_context(
            str(case["run_id"]), str(case["segment_role"]), float(case["horizon_ns"])
        )
        row_box = outer[row_index, 0].get_position(figure)
        header_offset_points = 14.0 if version == "main" else 10.0
        header_fontsize = 7.2 if version == "main" else 6.7
        header_y = row_box.y1 + header_offset_points / (
            72.0 * figure.get_figheight()
        )
        figure.text(
            row_box.x0 - 0.012, header_y, f"{chr(97 + row_index)}",
            fontsize=10.2, fontweight="bold", ha="left", va="bottom",
        )
        figure.text(
            row_box.x0 + 0.014, header_y, f"{physical_label}  |  {context}",
            fontsize=header_fontsize, fontweight="semibold", ha="left", va="bottom",
        )
        render_cases.append(
            {
                **case,
                "rendered_scatter_y": rendered_scatter_y,
                "mumax_display_indices": truth_indices,
                "flare_display_indices": flare_indices,
                "mumax_observable_range": {
                    "Q": [
                        float(np.min(truth_features["topological_charge"])),
                        float(np.max(truth_features["topological_charge"])),
                    ],
                    "mean_mz": [
                        float(np.min(truth_features["mean_mz"])),
                        float(np.max(truth_features["mean_mz"])),
                    ],
                    "exchange_texture_energy": [
                        float(np.min(truth_features["exchange_texture_energy"])),
                        float(np.max(truth_features["exchange_texture_energy"])),
                    ],
                },
                "flare_observable_range": {
                    "Q": [
                        float(np.min(flare_features["topological_charge"])),
                        float(np.max(flare_features["topological_charge"])),
                    ],
                    "mean_mz": [
                        float(np.min(flare_features["mean_mz"])),
                        float(np.max(flare_features["mean_mz"])),
                    ],
                    "exchange_texture_energy": [
                        float(np.min(flare_features["exchange_texture_energy"])),
                        float(np.max(flare_features["exchange_texture_energy"])),
                    ],
                },
                "poseidon_observables": {
                    "Q": float(baseline_features["topological_charge"][0]),
                    "mean_mz": float(baseline_features["mean_mz"][0]),
                    "exchange_texture_energy": float(
                        baseline_features["exchange_texture_energy"][0]
                    ),
                },
            }
        )

    if last_image is not None and first_anchor_axis is not None:
        # A compact scale beside the first anchor avoids crossing row headers.
        figure.canvas.draw()
        anchor_box = first_anchor_axis.get_position()
        colorbar_axis = figure.add_axes(
            [anchor_box.x0 - 0.022, anchor_box.y0, 0.008, anchor_box.height]
        )
        colorbar = figure.colorbar(last_image, cax=colorbar_axis)
        colorbar.ax.yaxis.set_ticks_position("left")
        colorbar.ax.set_title(r"$m_z$", fontsize=7.0, pad=2.0)
        colorbar.ax.tick_params(labelsize=5.8, length=2, pad=1.5)

    artifact_dir = figure_output_dir or output_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stem = f"generative_necessity_{version}"
    pdf = artifact_dir / f"{stem}.pdf"
    png = artifact_dir / f"{stem}.png"
    pdf_temporary = pdf.with_suffix(".pdf.tmp")
    png_temporary = png.with_suffix(".png.tmp")
    export_dpi = 300 if version == "main" else 220
    figure.savefig(pdf_temporary, format="pdf", dpi=export_dpi)
    figure.savefig(png_temporary, format="png", dpi=export_dpi)
    plt.close(figure)
    pdf_temporary.replace(pdf)
    png_temporary.replace(png)

    paper_pdf: Path | None = None
    paper_png: Path | None = None
    if copy_to_paper:
        paper_fig_dir.mkdir(parents=True, exist_ok=True)
        paper_pdf = paper_fig_dir / pdf.name
        paper_png = paper_fig_dir / png.name
        shutil.copyfile(pdf, paper_pdf)
        shutil.copyfile(png, paper_png)
    payload = {
        "schema": "flare_generative_necessity_render_v2",
        "status": "complete",
        "version": version,
        "selection_manifest": str(selection_path.resolve()),
        "selection_manifest_sha256": _sha256(selection_path),
        "figure_pdf": str(pdf.resolve()),
        "figure_pdf_sha256": _sha256(pdf),
        "figure_png": str(png.resolve()),
        "paper_copy_enabled": copy_to_paper,
        "paper_pdf": str(paper_pdf.resolve()) if paper_pdf is not None else None,
        "paper_png": str(paper_png.resolve()) if paper_png is not None else None,
        "field_channel": "m_z",
        "field_scale": [-1.0, 1.0],
        "ensemble_sizes": {"MuMax3": 30, "FLARE": 30, "Poseidon-T": 1},
        "flare_draw_source": "raw/<condition_id>/flare_30_f16.npy",
        "caption": physical_audit[
            "main_caption" if version == "main" else "appendix_caption"
        ],
        "display_selection": (
            "For both MuMax3 and FLARE: six deterministic facility-location "
            "medoids in the same pooled-standardized (Q, mean_mz, E_ex, "
            "topological-density centroid, wall centroid, wall mass) space"
        ),
        "cases": render_cases,
    }
    _write_json(artifact_dir / f"render_manifest_{version}.json", payload)
    print(json.dumps({"pdf": str(pdf), "png": str(png)}, indent=2))
    return payload


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("select", "audit", "render-main", "render-appendix", "render"),
    )
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--paper-fig-dir", type=Path, default=DEFAULT_PAPER_FIG)
    parser.add_argument("--figure-output-dir", type=Path)
    parser.add_argument("--no-paper-copy", action="store_true")
    parser.add_argument(
        "--main-scatter-y",
        choices=("audited", "mean_mz", "exchange_texture_energy"),
        default="audited",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    render_kwargs = {
        "figure_output_dir": args.figure_output_dir,
        "copy_to_paper": not args.no_paper_copy,
        "main_scatter_y": (
            None if args.main_scatter_y == "audited" else args.main_scatter_y
        ),
    }
    if args.command == "select":
        select(args.audit_root, args.output_dir)
    elif args.command == "audit":
        audit_physics(args.audit_root, args.output_dir)
    elif args.command in {"render-main", "render"}:
        render(
            args.audit_root,
            args.output_dir,
            args.paper_fig_dir,
            version="main",
            **render_kwargs,
        )
        if args.command == "render":
            render(
                args.audit_root,
                args.output_dir,
                args.paper_fig_dir,
                version="appendix",
                **render_kwargs,
            )
    elif args.command == "render-appendix":
        render(
            args.audit_root,
            args.output_dir,
            args.paper_fig_dir,
            version="appendix",
            **render_kwargs,
        )


if __name__ == "__main__":
    main()
