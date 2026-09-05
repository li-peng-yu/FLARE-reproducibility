#!/usr/bin/env python3
"""Regenerate Figure 21 on the formal seed-78 FLARE median case.

The case is frozen by the current formal qualitative-selection manifest before
any Direct U-Net output is inspected.  All learned fields are regenerated on
that exact case so neither the obsolete paper-full case nor its Cartesian
endpoint-delta baseline can leak into the paper figure.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import torch
import torch.nn.functional as F

from scripts.render_x5_paper_qualitative import (
    _load_checkpoint_model,
    _sample,
    _single_sample,
)
from skyrmion_cfm.config import release_checkpoint_config
from skyrmion_cfm.data.fixed_time import build_fixed_time_datasets


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--dataset-checkpoint", type=Path, required=True)
    parser.add_argument("--flare", type=Path, required=True)
    parser.add_argument("--cartesian", type=Path, required=True)
    parser.add_argument("--direct-unet", type=Path, required=True)
    parser.add_argument("--fno", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260822)
    return parser


def _median_case(path: Path) -> tuple[dict[str, str], int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload["single_segment"])
    candidates = [
        (index, row)
        for index, row in enumerate(rows)
        if abs(float(row["quantile"]) - 0.5) < 1.0e-9
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one q=0.5 single-segment case, found {len(candidates)}"
        )
    index, row = candidates[0]
    return dict(row["case"]), int(index), float(row["stage1_mean_angle"])


def _dataset(checkpoint_path: Path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    cfg = release_checkpoint_config(checkpoint["config"])
    del checkpoint
    cfg.setdefault("data", {})["segment_time_range_ns"] = [1.0, 6.0]
    cfg["data"].setdefault("memmap", {})["auto_build"] = False
    cfg.setdefault("train", {})["num_workers"] = 0
    _, _, dataset = build_fixed_time_datasets(cfg, build_splits={"test"})
    if dataset is None:
        raise RuntimeError("failed to build the frozen test split")
    return dataset


def _normalized(field: torch.Tensor) -> torch.Tensor:
    return F.normalize(field.detach().float().cpu(), dim=1, eps=1.0e-8)


def _valid_mask(sample: dict[str, Any], target: torch.Tensor) -> torch.Tensor:
    defect = sample.get("defect_field")
    if torch.is_tensor(defect):
        mask = defect.detach().float().cpu()
        if mask.ndim == 3:
            mask = mask[0]
    else:
        mask = torch.ones_like(target[0, 0])
    return (mask > 0.0) & (target[0].square().sum(dim=0) > 0.25)


def _angle_map(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[np.ndarray, float]:
    dot = (prediction[0] * target[0]).sum(dim=0).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(dot))
    mean = float(angle[valid].mean())
    array = angle.numpy()
    array[~valid.numpy()] = np.nan
    return array, mean


def _render(
    output: Path,
    target: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    angle_maps: dict[str, np.ndarray],
    angles: dict[str, float],
) -> None:
    methods = ["FLARE", "Cartesian CFM", "Direct U-Net", "FNO"]
    titles = ["MuMax3", *methods]
    figure = plt.figure(figsize=(10.34, 4.09))
    grid = figure.add_gridspec(
        2,
        6,
        width_ratios=[1, 1, 1, 1, 1, 0.045],
        height_ratios=[1, 1],
        left=0.055,
        right=0.965,
        bottom=0.075,
        top=0.92,
        wspace=0.055,
        hspace=0.30,
    )
    field_cmap = plt.get_cmap("coolwarm").copy()
    error_cmap = plt.get_cmap("magma").copy()
    field_cmap.set_bad("white")
    error_cmap.set_bad("white")

    fields = [target, *[predictions[method] for method in methods]]
    for column, (title, field) in enumerate(zip(titles, fields, strict=True)):
        axis = figure.add_subplot(grid[0, column])
        values = field[0, 2].numpy()
        axis.imshow(
            values,
            cmap=field_cmap,
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axis.set_title(title, fontsize=10, pad=4)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#444444")
            spine.set_linewidth(0.55)

    reference_axis = figure.add_subplot(grid[1, 0])
    reference_axis.axis("off")
    reference_axis.text(
        0.5,
        0.52,
        "reference\nfield",
        ha="center",
        va="center",
        fontsize=9,
        color="#444444",
    )
    for column, method in enumerate(methods, start=1):
        axis = figure.add_subplot(grid[1, column])
        axis.imshow(
            angle_maps[method],
            cmap=error_cmap,
            vmin=0.0,
            vmax=90.0,
            interpolation="nearest",
        )
        axis.set_title(
            f"angular error\nmean {angles[method]:.1f}°",
            fontsize=8.5,
            pad=3,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#444444")
            spine.set_linewidth(0.55)

    field_bar = figure.colorbar(
        ScalarMappable(norm=Normalize(-1.0, 1.0), cmap=field_cmap),
        cax=figure.add_subplot(grid[0, 5]),
        ticks=(-1.0, 0.0, 1.0),
    )
    field_bar.set_label(r"$m_z$", fontsize=9)
    field_bar.ax.tick_params(labelsize=8)
    error_bar = figure.colorbar(
        ScalarMappable(norm=Normalize(0.0, 90.0), cmap=error_cmap),
        cax=figure.add_subplot(grid[1, 5]),
        ticks=(0.0, 45.0, 90.0),
    )
    error_bar.set_label("degrees", fontsize=9)
    error_bar.ax.tick_params(labelsize=8)

    figure.text(0.013, 0.735, "a", fontsize=12, fontweight="bold", va="center")
    figure.text(
        0.029,
        0.735,
        r"magnetization $m_z$",
        fontsize=9,
        rotation=90,
        va="center",
        ha="center",
    )
    figure.text(0.013, 0.275, "b", fontsize=12, fontweight="bold", va="center")
    figure.text(
        0.029,
        0.275,
        "local angular discrepancy",
        fontsize=9,
        rotation=90,
        va="center",
        ha="center",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=240)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    args = _parser().parse_args()
    case, case_index, selection_mean_angle = _median_case(args.selection_manifest)
    dataset = _dataset(args.dataset_checkpoint)
    sample = _single_sample(dataset, case)
    device = torch.device(args.device)
    sampling_seed = int(args.seed) + 1009 * case_index

    specifications = [
        ("FLARE", args.flare, 10),
        ("Cartesian CFM", args.cartesian, 10),
        ("Direct U-Net", args.direct_unet, 1),
        ("FNO", args.fno, 1),
    ]
    predictions: dict[str, torch.Tensor] = {}
    target: torch.Tensor | None = None
    checkpoint_audits: dict[str, dict[str, Any]] = {}
    for method, checkpoint_path, steps in specifications:
        cfg, model, sampler = _load_checkpoint_model(
            checkpoint_path,
            device,
            ode_steps=steps,
        )
        prediction, _initial, current_target, _condition = _sample(
            model,
            sampler,
            sample,
            device,
            sampling_seed,
        )
        predictions[method] = _normalized(prediction)
        if target is None:
            target = _normalized(current_target)
        checkpoint_audits[method] = {
            "checkpoint": str(checkpoint_path),
            "arch": cfg["model"].get("arch"),
            "state_repr": cfg["bridge"].get("state_repr"),
            "source_mode": cfg["bridge"].get("source_mode"),
            "ode_steps": steps,
        }
        del model, sampler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if target is None:
        raise RuntimeError("no target was loaded")
    valid = _valid_mask(sample, target)
    angle_maps: dict[str, np.ndarray] = {}
    angles: dict[str, float] = {}
    for method, prediction in predictions.items():
        angle_maps[method], angles[method] = _angle_map(prediction, target, valid)

    expected_checkpoints = {
        "FLARE": ("unet", "alpha", "latent", 10),
        "Cartesian CFM": ("unet", "cart", None, 10),
        "Direct U-Net": ("unet", "alpha", "identity", 1),
        "FNO": ("fno", None, None, 1),
    }
    for method, (arch, state_repr, source_mode, steps) in expected_checkpoints.items():
        audit = checkpoint_audits[method]
        if audit["arch"] != arch or audit["ode_steps"] != steps:
            raise RuntimeError(f"wrong {method} checkpoint: {audit}")
        if state_repr is not None and audit["state_repr"] != state_repr:
            raise RuntimeError(f"wrong {method} representation: {audit}")
        if source_mode is not None and audit["source_mode"] != source_mode:
            raise RuntimeError(f"wrong {method} source: {audit}")
    direct = checkpoint_audits["Direct U-Net"]
    if not (
        direct["arch"] == "unet"
        and direct["state_repr"] == "alpha"
        and direct["source_mode"] == "identity"
        and direct["ode_steps"] == 1
    ):
        raise RuntimeError(f"wrong Direct U-Net control: {direct}")

    _render(args.output, target, predictions, angle_maps, angles)
    manifest = {
        "status": "complete",
        "selection": (
            "q=0.5 of the formal seed-78 FLARE single-segment mean angular "
            "error; frozen before inspecting Direct U-Net"
        ),
        "selection_manifest": str(args.selection_manifest),
        "selection_mean_angular_error_deg": selection_mean_angle,
        "case": case,
        "sampling_seed": sampling_seed,
        "mean_angular_error_deg": angles,
        "checkpoints": checkpoint_audits,
        "output_pdf": str(args.output.with_suffix(".pdf")),
        "output_png": str(args.output.with_suffix(".png")),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
