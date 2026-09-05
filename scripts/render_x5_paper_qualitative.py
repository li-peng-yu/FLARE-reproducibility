#!/usr/bin/env python3
"""Render metric-selected, fixed-scale x5 paper qualitative panels."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_x5_semigroup_handoff import _slerp
from skyrmion_cfm.config import release_checkpoint_config
from skyrmion_cfm.config import seed_everything
from skyrmion_cfm.data.fixed_time import (
    build_fixed_time_datasets,
    collate_fixed_time_conditions,
)
from skyrmion_cfm.data.stats import load_training_stats
from skyrmion_cfm.eval.conditional_algorithm_diagnostics import (
    _build_sampler,
    _load_state_verified,
)
from skyrmion_cfm.models import build_model
from skyrmion_cfm.train import move_batch


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty input: {path}")
    return rows


def _collate(samples: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    clean = [{key: value for key, value in sample.items() if value is not None} for sample in samples]
    return move_batch(default_collate(clean), device)


def _case_means(
    rows: list[dict[str, str]], metric: str, key_fields: tuple[str, ...]
) -> list[tuple[float, tuple[str, ...], dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in key_fields)].append(row)
    return sorted(
        (
            float(np.mean([float(row[metric]) for row in values])),
            key,
            values[0],
        )
        for key, values in grouped.items()
    )


def _nearest_quantile(
    values: list[tuple[float, tuple[str, ...], dict[str, str]]], quantile: float
) -> tuple[float, tuple[str, ...], dict[str, str]]:
    target = float(np.quantile([item[0] for item in values], quantile))
    return min(values, key=lambda item: (abs(item[0] - target), item[1]))


def _load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
    *,
    ode_steps: int | None = None,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    cfg = release_checkpoint_config(checkpoint["config"])
    cfg.setdefault("train", {})["compile"] = False
    stats = load_training_stats(Path(cfg["data"]["stats_cache"]))
    model = build_model(cfg, stats.condition).to(device)
    report = _load_state_verified(model, checkpoint, "ema")
    del checkpoint
    if float(report.get("matched_parameter_fraction", 0.0)) != 1.0:
        raise RuntimeError(f"partial checkpoint load: {checkpoint_path}: {report}")
    model.eval()
    sampler = _build_sampler(cfg, stats, ode_steps=ode_steps or int(cfg.get("sampler", {}).get("ode_steps", 10)))
    return cfg, model, sampler


@torch.inference_mode()
def _sample(model, sampler, sample: dict[str, Any], device: torch.device, seed: int, init=None):
    batch = _collate([sample], device)
    cond = collate_fixed_time_conditions(batch)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m_init = batch["m_init"] if init is None else init.to(device)
    pred, _ = sampler.sample(model, m_init, cond)
    return pred.detach(), batch["m_init"].detach(), batch["m_t"].detach(), cond


def _single_sample(dataset, row: dict[str, str]) -> dict[str, Any]:
    rec_idx = int(row["record_index"])
    segment = int(row["control_segment_index"])
    for choice_idx, start_frame in dataset._segment_boundary_specs(rec_idx):
        sample = dataset._build_sample(
            rec_idx,
            int(choice_idx),
            np.random.default_rng(20260813 + rec_idx),
            frame_init=int(start_frame),
            apply_augment=False,
        )
        if int(sample["control_segment_index"]) == segment:
            return sample
    raise RuntimeError(f"cannot resolve single-segment case: {row}")


def _mz(tensor: torch.Tensor) -> np.ndarray:
    return tensor[0, 2].detach().float().cpu().numpy()


def _save_grid(
    path: Path,
    rows: list[list[tuple[str, torch.Tensor]]],
    row_labels: list[str],
) -> None:
    columns = max(len(row) for row in rows)
    figure, axes = plt.subplots(
        len(rows), columns, figsize=(2.35 * columns, 2.2 * len(rows)), squeeze=False
    )
    image = None
    for row_index, items in enumerate(rows):
        for column_index, (title, tensor) in enumerate(items):
            axis = axes[row_index, column_index]
            image = axis.imshow(_mz(tensor), cmap="coolwarm", vmin=-1.0, vmax=1.0)
            axis.set_title(title, fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            if column_index == 0:
                axis.set_ylabel(row_labels[row_index], fontsize=9)
        for column_index in range(len(items), columns):
            axes[row_index, column_index].axis("off")
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.72, label=r"$m_z$")
    figure.subplots_adjust(left=0.06, right=0.91, bottom=0.04, top=0.92, wspace=0.06, hspace=0.24)
    figure.savefig(path, dpi=240)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--mixed", type=Path, required=True)
    parser.add_argument("--cartesian", type=Path, required=True)
    parser.add_argument("--fno", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SKYRMION_CFM_FUSED_OPS", "1")
    seed_everything(args.seed)
    device = torch.device(args.device)

    # The Stage1 data object defines the common held-out record indexing.
    base_checkpoint = torch.load(args.stage1, map_location="cpu", weights_only=False, mmap=True)
    data_cfg = release_checkpoint_config(base_checkpoint["config"])
    del base_checkpoint
    data_cfg.setdefault("data", {})["segment_time_range_ns"] = [1.0, 6.0]
    data_cfg["data"].setdefault("memmap", {})["auto_build"] = False
    data_cfg.setdefault("train", {})["num_workers"] = 0
    _, _, dataset = build_fixed_time_datasets(data_cfg, build_splits={"test"})
    if dataset is None:
        raise RuntimeError("failed to build held-out dataset")

    selection: dict[str, Any] = {
        "status": "complete",
        "policy": "fixed numerical quantiles; no visual inspection used for selection",
        "seed": args.seed,
    }
    single_csv = args.root / "single_segment" / "stage1_ode10.csv"
    single_values = _case_means(
        _read(single_csv),
        "ang",
        ("run_id", "control_segment_index", "start_time_ns", "end_time_ns"),
    )
    selected_single = [_nearest_quantile(single_values, q) for q in (0.25, 0.5, 0.9)]
    selection["single_segment"] = [
        {"quantile": q, "stage1_mean_angle": item[0], "case": item[2]}
        for q, item in zip((0.25, 0.5, 0.9), selected_single, strict=True)
    ]
    samples = [_single_sample(dataset, item[2]) for item in selected_single]
    model_paths = {
        "Stage1": args.stage1,
        "Stage2": args.mixed,
        # Use the paper-facing representation-control name.
        "Cartesian CFM": args.cartesian,
        "FNO": args.fno,
    }
    predictions: dict[str, list[torch.Tensor]] = {}
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for model_index, (label, path) in enumerate(model_paths.items()):
        steps = 1 if label == "FNO" else 10
        _cfg, model, sampler = _load_checkpoint_model(path, device, ode_steps=steps)
        values = []
        for case_index, sample in enumerate(samples):
            pred, initial, target, _cond = _sample(
                model, sampler, sample, device, args.seed + 1009 * case_index
            )
            values.append(pred.cpu())
            if model_index == 0:
                inputs.append(initial.cpu())
                targets.append(target.cpu())
        predictions[label] = values
        del model, sampler
        torch.cuda.empty_cache()
    panel_rows = []
    for index in range(len(samples)):
        panel_rows.append(
            [("Initial", inputs[index]), ("MuMax", targets[index])]
            + [(label, predictions[label][index]) for label in model_paths]
        )
    _save_grid(
        args.output_dir / "single_segment_quantiles.png",
        panel_rows,
        ["q25", "q50", "q90"],
    )

    # Fully autoregressive: select median and 90th-percentile final-hop Stage1
    # trajectories, then compare Stage1 and mixed Stage2 with identical seeds.
    rollout_rows = [
        row
        for draw in range(4)
        for row in _read(args.root / "rollout" / "raw" / f"stage1_draw{draw}.csv")
        if int(row["segment_position"]) == 1
    ]
    rollout_values = _case_means(rollout_rows, "ar_ang", ("run_id",))
    selected_rollout = [_nearest_quantile(rollout_values, q) for q in (0.5, 0.9)]
    selection["fully_autoregressive"] = [
        {"quantile": q, "stage1_ar_angle": item[0], "case": item[2]}
        for q, item in zip((0.5, 0.9), selected_rollout, strict=True)
    ]
    ar_panels: list[list[tuple[str, torch.Tensor]]] = []
    for selected in selected_rollout:
        rec_idx = int(selected[2]["record_index"])
        specs = dataset._segment_boundary_specs(rec_idx)
        samples_ar = [
            dataset._build_sample(
                rec_idx,
                int(choice),
                np.random.default_rng(args.seed + rec_idx + position),
                frame_init=int(start),
                apply_augment=False,
            )
            for position, (choice, start) in enumerate(specs)
        ]
        row: list[tuple[str, torch.Tensor]] = [
            ("Initial", samples_ar[0]["m_init"].unsqueeze(0)),
            ("MuMax final", samples_ar[-1]["m_t"].unsqueeze(0)),
        ]
        for label, path in (("Stage1 AR", args.stage1), ("Stage2 AR", args.mixed)):
            _cfg, model, sampler = _load_checkpoint_model(path, device, ode_steps=10)
            state = None
            for position, sample in enumerate(samples_ar):
                pred, _initial, _target, _cond = _sample(
                    model,
                    sampler,
                    sample,
                    device,
                    args.seed + 10_000_019 * position,
                    init=state,
                )
                state = pred
            row.append((label, state.cpu()))
            del model, sampler
            torch.cuda.empty_cache()
        ar_panels.append(row)
    _save_grid(
        args.output_dir / "fully_autoregressive_quantiles.png",
        ar_panels,
        ["q50", "q90"],
    )

    # Direct versus composed: use the median Stage1 mean semigroup defect.
    sg_rows = _read(args.root / "semigroup_handoff" / "stage1" / "semigroup_distribution.csv")
    sg_values = _case_means(
        sg_rows,
        "mean_sg_defect_ang",
        ("run_id", "start", "middle", "end"),
    )
    selected_sg = _nearest_quantile(sg_values, 0.5)
    sg_row = selected_sg[2]
    selection["semigroup"] = {
        "quantile": 0.5,
        "stage1_mean_defect_angle": selected_sg[0],
        "case": sg_row,
    }
    rec_idx = int(sg_row["record_index"])
    start, middle, end = int(sg_row["start"]), int(sg_row["middle"]), int(sg_row["end"])
    s1 = dataset._build_visual_sample_for_frames(rec_idx, start, middle, np.random.default_rng(args.seed))
    s2 = dataset._build_visual_sample_for_frames(rec_idx, middle, end, np.random.default_rng(args.seed + 1))
    sd = dataset._build_visual_sample_for_frames(rec_idx, start, end, np.random.default_rng(args.seed + 2))
    sg_panel = [("Initial", sd["m_init"].unsqueeze(0)), ("MuMax", sd["m_t"].unsqueeze(0))]
    for label, path in (("Stage1", args.stage1), ("Stage2", args.mixed)):
        _cfg, model, sampler = _load_checkpoint_model(path, device, ode_steps=10)
        first, _i, _t, _c = _sample(model, sampler, s1, device, args.seed)
        composed, _i, _t, _c = _sample(model, sampler, s2, device, args.seed + 1, init=first)
        direct, _i, _t, _c = _sample(model, sampler, sd, device, args.seed)
        sg_panel.extend([(f"{label} direct", direct.cpu()), (f"{label} composed", composed.cpu())])
        del model, sampler
        torch.cuda.empty_cache()
    _save_grid(args.output_dir / "direct_vs_composed.png", [sg_panel], ["median SG defect"])

    # Handoff interpolation: median mixed-Stage2 lambda=1 output error.
    handoff_rows = [
        row
        for row in _read(args.root / "semigroup_handoff" / "mixed" / "handoff_curve.csv")
        if abs(float(row["lambda"]) - 1.0) < 1.0e-9
    ]
    handoff_values = _case_means(handoff_rows, "output_ang", ("run_id",))
    selected_handoff = _nearest_quantile(handoff_values, 0.5)
    hrow = selected_handoff[2]
    selection["handoff"] = {
        "quantile": 0.5,
        "mixed_lambda1_angle": selected_handoff[0],
        "case": hrow,
    }
    rec_idx = int(hrow["record_index"])
    specs = dataset._segment_boundary_specs(rec_idx)
    first_sample = dataset._build_sample(rec_idx, specs[0][0], np.random.default_rng(args.seed), frame_init=specs[0][1], apply_augment=False)
    second_sample = dataset._build_sample(rec_idx, specs[1][0], np.random.default_rng(args.seed + 1), frame_init=specs[1][1], apply_augment=False)
    _cfg, model, sampler = _load_checkpoint_model(args.mixed, device, ode_steps=10)
    predicted_boundary, _i, _t, _c = _sample(model, sampler, first_sample, device, args.seed)
    true_boundary = second_sample["m_init"].unsqueeze(0).to(device)
    handoff_panel = [("True boundary", true_boundary.cpu()), ("Pred boundary", predicted_boundary.cpu()), ("MuMax final", second_sample["m_t"].unsqueeze(0))]
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        interpolated = _slerp(true_boundary, predicted_boundary, weight)
        pred, _i, _t, _c = _sample(
            model, sampler, second_sample, device, args.seed + 1, init=interpolated
        )
        handoff_panel.append((f"lambda={weight:g}", pred.cpu()))
    del model, sampler
    torch.cuda.empty_cache()
    _save_grid(args.output_dir / "handoff_interpolation.png", [handoff_panel], ["median lambda=1"])

    (args.output_dir / "selection_manifest.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
