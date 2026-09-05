#!/usr/bin/env python3
"""Summarize matched x5 distribution diagnostics across requested horizons.

Each saved-frame horizon directory is produced by
``evaluate_skx_x5_same_condition_distribution.py`` with 25 generated samples
(five stochastic forecasts for each of five exact MuMax3 anchors) and a
five-versus-five score budget. Horizons absent from this original saved-frame
audit remain explicit rows; rollout-extension evaluations are reported
separately rather than imputed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.analyze_x5_distribution_sensitivity_observables import (
    _cluster_bootstrap,
    _mean_pairwise_abs,
    _observables,
    _w1_equal_samples,
)


SUPPORTED = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
REQUESTED = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)
SEEN_TRAINING_BUCKETS = {1.0, 2.0, 3.0}


def _tag(duration: float) -> str:
    return f"{duration:.1f}".replace(".", "p")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metric(summary: dict[str, Any], *keys: str) -> tuple[float, float, float]:
    value: Any = summary
    for key in keys:
        value = value[key]
    if "geometric_mean_score" in value:
        point = float(value["geometric_mean_score"])
        interval = value["geometric_mean_bootstrap_95ci"]
    elif "mean" in value:
        point = float(value["mean"])
        interval = value["bootstrap_95ci"]
    else:
        raise KeyError(f"unsupported metric payload at {keys}: {value}")
    return point, float(interval[0]), float(interval[1])


def _spread_summary(
    rows: list[dict[str, str]], *, iterations: int, seed: int
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["base_id"])].append(
            (
                float(row["within_model_mean_deg"]),
                float(row["within_mumax_mean_deg"]),
            )
        )
    bases = sorted(grouped)
    if not bases:
        raise ValueError("spread summary received no conditions")

    def aggregate(selected: list[int]) -> tuple[float, float, float]:
        pairs = np.asarray(
            [pair for index in selected for pair in grouped[bases[index]]],
            dtype=np.float64,
        )
        model = float(pairs[:, 0].mean())
        mumax = float(pairs[:, 1].mean())
        return model, mumax, model / max(mumax, 1.0e-12)

    point = aggregate(list(range(len(bases))))
    rng = np.random.default_rng(seed)
    draws = np.empty((iterations, 3), dtype=np.float64)
    for index in range(iterations):
        selected = rng.integers(0, len(bases), size=len(bases)).tolist()
        draws[index] = aggregate(selected)
    return {
        "model_within_ensemble_angle_deg": point[0],
        "mumax_within_ensemble_angle_deg": point[1],
        "model_to_mumax_spread_ratio": point[2],
        "model_bootstrap_95ci": np.quantile(draws[:, 0], [0.025, 0.975]).tolist(),
        "mumax_bootstrap_95ci": np.quantile(draws[:, 1], [0.025, 0.975]).tolist(),
        "ratio_bootstrap_95ci": np.quantile(draws[:, 2], [0.025, 0.975]).tolist(),
        "estimator": (
            "ratio of equal-condition mean biased-V angular within-set distances; "
            "both ensembles contain five samples, so the common finite-N factor cancels"
        ),
        "bootstrap_unit": "base group; both segments retained",
    }


def _condition_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _observable_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in sorted((root / "conditions").glob("base*_segment*")):
        metadata_path = condition / "condition_metadata.json"
        score_dir = condition / "same_condition_score_4x4_shift16"
        model_path = condition / "model_samples/model_samples_f16.npy"
        if not (
            metadata_path.is_file()
            and (score_dir / "geometry_mask.npy").is_file()
            and (score_dir / "mumax_targets_f16.npy").is_file()
            and model_path.is_file()
        ):
            continue
        metadata = _json(metadata_path)
        mask = np.load(score_dir / "geometry_mask.npy", mmap_mode="r")
        truth = _observables(
            np.load(score_dir / "mumax_targets_f16.npy", mmap_mode="r"), mask
        )
        model = _observables(np.load(model_path, mmap_mode="r")[:5], mask)
        row: dict[str, Any] = {
            "condition_id": condition.name,
            "base_id": str(metadata["base_id"]).zfill(4),
            "segment_role": str(metadata["segment_role"]),
        }
        for observable in sorted(truth):
            row[f"{observable}_w1"] = _w1_equal_samples(
                model[observable], truth[observable]
            )
            row[f"{observable}_mumax_pairwise"] = _mean_pairwise_abs(
                truth[observable]
            )
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no complete observable conditions under {root}")
    return rows


def _unsupported_row(duration: float, *, support: str) -> dict[str, Any]:
    return {
        "duration_ns": duration,
        "support": support,
        "legal_conditions": 0,
        **{
            key: ""
            for key in (
                "distribution_score",
                "distribution_score_ci_low",
                "distribution_score_ci_high",
                "paired_angle_deg",
                "paired_angle_deg_ci_low",
                "paired_angle_deg_ci_high",
                "angular_energy_distance_deg",
                "angular_energy_distance_deg_ci_low",
                "angular_energy_distance_deg_ci_high",
                "fair_energy_score",
                "fair_energy_ci_low",
                "fair_energy_ci_high",
                "q_w1",
                "q_w1_ci_low",
                "q_w1_ci_high",
                "mean_mz_w1",
                "mean_mz_w1_ci_low",
                "mean_mz_w1_ci_high",
                "exchange_texture_w1",
                "exchange_texture_w1_ci_low",
                "exchange_texture_w1_ci_high",
                "model_spread_deg",
                "model_spread_ci_low",
                "model_spread_ci_high",
                "mumax_spread_deg",
                "mumax_spread_ci_low",
                "mumax_spread_ci_high",
                "spread_ratio",
                "spread_ratio_ci_low",
                "spread_ratio_ci_high",
            )
        },
    }


def _summarize_duration(
    root: Path, duration: float, *, bootstrap: int, seed: int
) -> dict[str, Any]:
    run = _json(root / "summary/run_summary.json")
    requested = run.get("requested_horizon_ns")
    if requested is None or not math.isclose(float(requested), duration, abs_tol=1.0e-8):
        raise RuntimeError(f"{root}: requested horizon is {requested}, expected {duration}")
    if int(run["model_samples_per_condition"]) != 25:
        raise RuntimeError(f"{root}: expected 25 generated samples per condition")
    if int(run["score_model_samples_per_condition"]) != 5:
        raise RuntimeError(f"{root}: expected a five-versus-five score budget")

    fair = _json(root / "fair_energy/run_summary.json")
    fair_combined = fair["methods"]["FLARE"]["combined"]["fair_energy_score"]
    fair_metric = (
        float(fair_combined["mean"]),
        float(fair_combined["bootstrap_95ci"][0]),
        float(fair_combined["bootstrap_95ci"][1]),
    )

    observable_rows = _observable_rows(root)
    observable_metrics: dict[str, tuple[float, float, float]] = {}
    for offset, observable in enumerate(
        ("topological_charge", "mean_mz", "exchange_texture_energy")
    ):
        summary = _cluster_bootstrap(
            observable_rows,
            f"{observable}_w1",
            geometric=False,
            iterations=bootstrap,
            seed=seed + 1000 * (offset + 1),
        )
        observable_metrics[observable] = (
            float(summary["value"]),
            float(summary["bootstrap_95ci"][0]),
            float(summary["bootstrap_95ci"][1]),
        )

    spread = _spread_summary(
        _condition_rows(root / "summary/condition_scores.csv"),
        iterations=bootstrap,
        seed=seed + 9000,
    )
    ds = _metric(run, "score", "combined")
    angular_energy = _metric(run, "angular_energy_distance", "combined")
    paired_angle = _metric(run, "paired_angular_error", "combined")
    q = observable_metrics["topological_charge"]
    mz = observable_metrics["mean_mz"]
    exchange = observable_metrics["exchange_texture_energy"]
    return {
        "duration_ns": duration,
        "support": "seen" if duration in SEEN_TRAINING_BUCKETS else "interpolation",
        "legal_conditions": int(run["conditions"]),
        "distribution_score": ds[0],
        "distribution_score_ci_low": ds[1],
        "distribution_score_ci_high": ds[2],
        "paired_angle_deg": paired_angle[0],
        "paired_angle_deg_ci_low": paired_angle[1],
        "paired_angle_deg_ci_high": paired_angle[2],
        "angular_energy_distance_deg": angular_energy[0],
        "angular_energy_distance_deg_ci_low": angular_energy[1],
        "angular_energy_distance_deg_ci_high": angular_energy[2],
        "fair_energy_score": fair_metric[0],
        "fair_energy_ci_low": fair_metric[1],
        "fair_energy_ci_high": fair_metric[2],
        "q_w1": q[0],
        "q_w1_ci_low": q[1],
        "q_w1_ci_high": q[2],
        "mean_mz_w1": mz[0],
        "mean_mz_w1_ci_low": mz[1],
        "mean_mz_w1_ci_high": mz[2],
        "exchange_texture_w1": exchange[0],
        "exchange_texture_w1_ci_low": exchange[1],
        "exchange_texture_w1_ci_high": exchange[2],
        "model_spread_deg": spread["model_within_ensemble_angle_deg"],
        "model_spread_ci_low": spread["model_bootstrap_95ci"][0],
        "model_spread_ci_high": spread["model_bootstrap_95ci"][1],
        "mumax_spread_deg": spread["mumax_within_ensemble_angle_deg"],
        "mumax_spread_ci_low": spread["mumax_bootstrap_95ci"][0],
        "mumax_spread_ci_high": spread["mumax_bootstrap_95ci"][1],
        "spread_ratio": spread["model_to_mumax_spread_ratio"],
        "spread_ratio_ci_low": spread["ratio_bootstrap_95ci"][0],
        "spread_ratio_ci_high": spread["ratio_bootstrap_95ci"][1],
    }


def _plot(rows: list[dict[str, Any]], output: Path) -> None:
    valid = [row for row in rows if row["support"] in {"seen", "interpolation"}]
    x = np.asarray([row["duration_ns"] for row in valid], dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.3), sharex=True)

    def band(axis: Any, key: str, color: str, label: str) -> None:
        value = np.asarray([row[key] for row in valid], dtype=np.float64)
        low = np.asarray([row[f"{key}_ci_low"] for row in valid], dtype=np.float64)
        high = np.asarray([row[f"{key}_ci_high"] for row in valid], dtype=np.float64)
        axis.plot(x, value, "o-", color=color, label=label, linewidth=1.8)
        axis.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0)

    band(axes[0, 0], "paired_angle_deg", "#0072B2", "paired angle")
    band(
        axes[0, 0],
        "angular_energy_distance_deg",
        "#D55E00",
        "angular energy distance",
    )
    axes[0, 0].set_ylabel("angle (degrees)")
    axes[0, 0].legend(frameon=False, fontsize=8)

    band(axes[0, 1], "distribution_score", "#009E73", "Distribution Score")
    axes[0, 1].set_ylabel("Distribution Score")
    axes[0, 1].set_ylim(0.0, 1.02)

    band(axes[1, 0], "q_w1", "#CC79A7", "$Q$")
    band(axes[1, 0], "mean_mz_w1", "#56B4E9", "mean $m_z$")
    band(axes[1, 0], "exchange_texture_w1", "#E69F00", "$E_{ex}$ proxy")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("observable $W_1$ (log scale)")
    axes[1, 0].legend(frameon=False, fontsize=8)

    band(axes[1, 1], "spread_ratio", "#000000", "model / MuMax3 spread")
    axes[1, 1].axhline(1.0, color="#777777", linestyle="--", linewidth=1.1)
    axes[1, 1].set_ylabel("within-ensemble spread ratio")

    # This plot describes only the original
    # saved-frame audit; it must not label unsaved endpoints as invalid tasks.
    for axis in axes.flat:
        axis.axvspan(3.5, 6.0, color="#BDBDBD", alpha=0.22, zorder=-10)
        axis.text(
            4.75,
            0.96,
            "not saved in the original audit\n(5 ns extension evaluated separately)",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.5,
            color="#555555",
        )
        axis.set_xlim(1.0, 6.0)
        axis.set_xticks(np.arange(1.0, 6.1, 0.5))
        axis.grid(alpha=0.18, linewidth=0.6)
    axes[1, 0].set_xlabel("requested horizon (ns)")
    axes[1, 1].set_xlabel("requested horizon (ns)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/paper_artifacts/horizon_distribution")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/paper_artifacts/horizon_distribution")
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("paper/fig/appendix_horizon_diagnostics.pdf")
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=208281700)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for index, duration in enumerate(REQUESTED):
        if duration not in SUPPORTED:
            rows.append(
                # Missing stored frames do not
                # make a future-state task invalid; they require a generated
                # solver reference, as done for the separate 5-ns audit.
                _unsupported_row(duration, support="rollout-reference-not-generated")
            )
            continue
        duration_root = args.root / f"duration_{_tag(duration)}ns"
        run = _json(duration_root / "summary/run_summary.json")
        if int(run.get("conditions", 0)) == 0:
            rows.append(
                _unsupported_row(duration, support="no-exact-saved-target")
            )
            continue
        rows.append(
            _summarize_duration(
                duration_root,
                duration,
                bootstrap=args.bootstrap,
                seed=args.seed + 10_000 * index,
            )
        )

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "horizon_metrics.csv", rows)
    payload = {
        "status": "complete",
        "rows": rows,
        "protocol": {
            "quality_budget": "five model endpoints versus five MuMax3 endpoints",
            "fair_score_budget": "five stochastic draws for each of five exact anchors",
            "interval": "base-group clustered bootstrap; both segments retained",
            "support_statement": (
                "The configured 1--6 ns range is an absolute-time filter. "
                "The nominal post-drive segment lasts 3.5 ns, but exact saved-frame "
                "targets are available only through 3.0 ns in the held-out x5 audit; "
                "the separate matched audit generates the 5-ns endpoint by MuMax3 "
                "continuation, while rollout references were not generated here for "
                "the remaining empty horizons."
            ),
            "spread": (
                "ratio of model to MuMax3 equal-condition within-ensemble angular spread"
            ),
        },
    }
    _write_json(args.output / "horizon_metrics.json", payload)
    _plot(rows, args.figure)
    print(args.output / "horizon_metrics.json")


if __name__ == "__main__":
    main()
