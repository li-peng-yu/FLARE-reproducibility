from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import time
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from skyrmion_cfm.cfm.bridges import (
    RotationVector2DBridge,
    RotationVectorBridge,
    bridge_source_mode,
    bridge_state_repr,
    make_bridge,
)
from skyrmion_cfm.cfm.loss import CFMLoss, LossOutput
from skyrmion_cfm.cfm.prior import CartSourcePrior, RFMSourcePrior, RotationPrior
from skyrmion_cfm.cfm.rfm import slerp_chw
from skyrmion_cfm.cfm.sampler import BridgeSampler
from skyrmion_cfm.cfm.spatial_mask import masked_site_mean
from skyrmion_cfm.config import get_device, load_config, merge_config, seed_everything
from skyrmion_cfm.data.conditions import ConditionStats, audit_scalar_conditions
from skyrmion_cfm.data.fixed_time import (
    build_fixed_time_datasets,
    collate_fixed_time_conditions,
)
from skyrmion_cfm.data.log_map import log_map_chw
from skyrmion_cfm.data.stats import (
    STATS_SCHEMA_VERSION,
    TrainingStats,
    estimate_omega_stats,
    fit_physics_kappa,
    load_training_stats,
    save_training_stats,
)
from skyrmion_cfm.data.trajectory import build_datasets, collate_conditions
from skyrmion_cfm.eval.metrics import (
    angular_error_deg,
    energy_density,
    magnetic_fraction,
    mse_m,
    topological_charge,
    topological_charge_density,
)
from skyrmion_cfm.eval.informative_visualization import (
    add_informative_candidates,
    evenly_spaced_bucket_indices,
    save_informative_visualization,
    select_informative_candidates,
)
from skyrmion_cfm.models import build_model


def _progress_bar(completed: int, total: int, *, width: int = 20) -> str:
    fraction = min(1.0, max(0.0, completed / max(total, 1)))
    filled = min(width, int(fraction * width))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _format_progress_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0.0:
        return "calculating"
    rounded = int(round(seconds))
    days, remainder = divmod(rounded, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _spin_image(m: torch.Tensor, channel: str = "mz") -> torch.Tensor:
    channel_idx = {"mx": 0, "my": 1, "mz": 2}.get(channel, 2)
    return m[channel_idx].detach().float().cpu()


def _save_validation_visualization(
    examples: list[dict[str, Any]],
    out_dir: Path,
    step: int,
    channel: str = "mz",
    dpi: int = 180,
    filename_prefix: str = "",
) -> Path | None:
    if not examples:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on cluster env
        print({"val_visualization": "skipped", "reason": f"matplotlib import failed: {exc}"}, flush=True)
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    if examples and "cells" in examples[0]:
        has_initial_col = any("initial" in row for row in examples)
        rows_per_example = 3
        gap_units = 0.55
        cell_units = 0.82
        initial_units = 0.95
        def _row_units(row: dict[str, Any]) -> float:
            units = initial_units if has_initial_col and "initial" in row else 0.0
            cells = row["cells"]
            gaps = sum(
                1
                for prev, curr in zip(cells, cells[1:])
                if int(prev.get("segment_index", 0)) != int(curr.get("segment_index", 0))
            )
            return units + cell_units * float(len(cells)) + gap_units * float(gaps)

        max_units = max(max(_row_units(row), 1.0) for row in examples)
        fig = plt.figure(figsize=(1.65 * max_units, 1.65 * rows_per_example * len(examples)))
        left, right = 0.055, 0.995
        bottom, top = 0.02, 0.955
        unit_w = (right - left) / max_units
        sample_h = (top - bottom) / max(len(examples), 1)
        image_w_scale = 0.92
        row_gap = 0.035 * sample_h
        panel_h = (0.84 * sample_h - 2.0 * row_gap) / rows_per_example
        for row_idx, row in enumerate(examples):
            sample_bottom = top - (row_idx + 1) * sample_h + 0.065 * sample_h
            run_id = str(row.get("run_id", ""))
            prediction_label = str(row.get("prediction_label", "pred"))
            cursor = 0.0
            if has_initial_col and "initial" in row:
                initial = _spin_image(row["initial"], channel)
                y = sample_bottom + (rows_per_example - 1) * (panel_h + row_gap)
                ax = fig.add_axes([left + cursor * unit_w, y, initial_units * unit_w * image_w_scale, panel_h])
                ax.axis("on")
                ax.imshow(initial.numpy(), cmap="RdBu_r", vmin=-1.0, vmax=1.0, origin="lower")
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title("initial", fontsize=7)
                ax.set_ylabel(f"sample {row_idx + 1}\n{run_id}\ninput", fontsize=7)
                for offset, label in ((1, "pred"), (2, "diff")):
                    y = sample_bottom + (rows_per_example - 1 - offset) * (panel_h + row_gap)
                    ax = fig.add_axes([left + cursor * unit_w, y, initial_units * unit_w * image_w_scale, panel_h])
                    ax.axis("off")
                    ax.set_ylabel(label, fontsize=7)
                cursor += initial_units
            elif not has_initial_col:
                pass
            prev_segment: int | None = None
            for time_idx, item in enumerate(row["cells"]):
                segment = int(item.get("segment_index", 0))
                if prev_segment is not None and segment != prev_segment:
                    cursor += gap_units
                if item.get("kind") == "segment_input":
                    # A segment boundary has two distinct states in a rollout:
                    # the observed MuMax state and the model state carried from
                    # the preceding prediction.  Keep them on their respective
                    # rows; interleaving the model state into the MuMax row makes
                    # the upper strip look like a physically discontinuous
                    # ground-truth trajectory.
                    truth_state = item.get("target_state", item.get("state"))
                    model_state = item.get("pred_state", item.get("state", truth_state))
                    truth = _spin_image(truth_state, channel)
                    model_input = _spin_image(model_state, channel)
                    diff = (model_input - truth).abs()
                    frame = item.get("frame")
                    abs_t_ns = float(item.get("abs_t_ns", 0.0))
                    title = f"segment {segment} start\nt={abs_t_ns:.4f} ns"
                    if frame is not None:
                        title = f"{title}\nf{frame}"
                    diff_vmax = max(float(diff.max()), 1e-6)
                    panels = (
                        ("mumax", truth, "RdBu_r", -1.0, 1.0),
                        (prediction_label, model_input, "RdBu_r", -1.0, 1.0),
                        ("abs diff", diff, "magma", 0.0, diff_vmax),
                    )
                    for offset, (label, image, cmap, vmin, vmax) in enumerate(panels):
                        y = sample_bottom + (rows_per_example - 1 - offset) * (panel_h + row_gap)
                        ax = fig.add_axes([left + cursor * unit_w, y, cell_units * unit_w * image_w_scale, panel_h])
                        ax.axis("on")
                        ax.imshow(
                            image.numpy(),
                            cmap=cmap,
                            vmin=vmin,
                            vmax=vmax,
                            origin="lower",
                        )
                        ax.set_xticks([])
                        ax.set_yticks([])
                        if offset == 0:
                            ax.set_title(title, fontsize=6)
                        if time_idx == 0 and not has_initial_col:
                            ylabel = label
                            if offset == 0:
                                ylabel = f"sample {row_idx + 1}\n{run_id}\n{label}"
                            ax.set_ylabel(ylabel, fontsize=7)
                    cursor += cell_units
                    prev_segment = segment
                    continue
                target = _spin_image(item["target"], channel)
                pred = _spin_image(item["pred"], channel)
                diff = (pred - target).abs()
                spin_vmax = max(
                    float(target.abs().max()),
                    float(pred.abs().max()),
                    1.0,
                )
                diff_vmax = max(float(diff.max()), 1e-6)
                frame_init = item.get("frame_init")
                frame_target = item.get("frame_target")
                abs_t_ns = float(item.get("abs_t_ns", item["t_end_ns"]))
                dt_ns = float(item["t_end_ns"])
                segment_elapsed_ns = item.get("segment_elapsed_ns")
                title = f"t={abs_t_ns:.4f} ns"
                if item.get("rollout_step") is not None:
                    title = f"{title}\nhop_dt={dt_ns:.4f} ns"
                    if bool(item.get("boundary_transition", False)):
                        title = f"{title} (boundary)"
                elif segment_elapsed_ns is not None:
                    title = f"{title}\nseg_dt={float(segment_elapsed_ns):.4f} ns"
                else:
                    title = f"{title}\ndt={dt_ns:.4f} ns"
                if frame_init is not None and frame_target is not None:
                    title = f"{title}\nf{frame_init}->f{frame_target}"
                panels = (
                    ("mumax", target, "RdBu_r", -spin_vmax, spin_vmax),
                    (prediction_label, pred, "RdBu_r", -spin_vmax, spin_vmax),
                    ("abs diff", diff, "magma", 0.0, diff_vmax),
                )
                for offset, (label, image, cmap, vmin, vmax) in enumerate(panels):
                    y = sample_bottom + (rows_per_example - 1 - offset) * (panel_h + row_gap)
                    ax = fig.add_axes([left + cursor * unit_w, y, cell_units * unit_w * image_w_scale, panel_h])
                    ax.axis("on")
                    ax.imshow(image.numpy(), cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if offset == 0:
                        ax.set_title(title, fontsize=6)
                    if time_idx == 0 and not has_initial_col:
                        ylabel = label
                        if offset == 0:
                            ylabel = f"sample {row_idx + 1}\n{run_id}\n{label}"
                        ax.set_ylabel(ylabel, fontsize=7)
                cursor += cell_units
                prev_segment = segment
        mode_title = str(examples[0].get("mode_title", "")).strip()
        suffix = f" — {mode_title}" if mode_title else ""
        fig.suptitle(f"Validation step {step} ({channel}){suffix}", fontsize=12)
        path = out_dir / f"{filename_prefix}val_step_{step:07d}_{channel}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return path

    nrows = len(examples)
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.4, 2.6 * nrows), squeeze=False)
    col_labels = ("mumax", "pred", "abs diff")
    for row_idx, item in enumerate(examples):
        target = _spin_image(item["target"], channel)
        pred = _spin_image(item["pred"], channel)
        diff = (pred - target).abs()
        spin_vmax = max(float(target.abs().max()), float(pred.abs().max()), 1.0)
        diff_vmax = max(float(diff.max()), 1e-6)
        frame_init = item.get("frame_init")
        frame_target = item.get("frame_target")
        row_label = f"t={item['t_end_ns']:.3g} ns"
        if frame_init is not None and frame_target is not None:
            row_label = f"{row_label}\nf{frame_init}->f{frame_target}"
        for col, (label, image, cmap, vmin, vmax) in enumerate(
            (
                ("mumax", target, "RdBu_r", -spin_vmax, spin_vmax),
                ("pred", pred, "RdBu_r", -spin_vmax, spin_vmax),
                ("abs diff", diff, "magma", 0.0, diff_vmax),
            )
        ):
            ax = axes[row_idx][col]
            ax.imshow(image.numpy(), cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(col_labels[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=9)
    fig.suptitle(f"Validation step {step} ({channel})", fontsize=12)
    fig.tight_layout()
    path = out_dir / f"{filename_prefix}val_step_{step:07d}_{channel}.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _filename_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    slug = slug.strip("._-")
    return slug or "stage"


def _collect_validation_examples(
    examples: list[dict[str, Any]],
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: dict[str, Any],
    max_examples: int,
    target_buckets: set[int] | None = None,
) -> None:
    if max_examples <= 0 or len(examples) >= max_examples:
        return
    t_end_ns = batch.get("t_end_ns")
    t_end_index = batch.get("t_end_index", batch.get("dt_index"))
    existing = {int(item["t_end_index"]) for item in examples if item.get("t_end_index") is not None}
    seen = set(existing)
    batch_size = int(pred.shape[0])
    preferred: list[int] = []
    fallback: list[int] = []
    for idx in range(batch_size):
        bucket = int(t_end_index[idx].detach().cpu()) if torch.is_tensor(t_end_index) else None
        wants_bucket = target_buckets is None or bucket in target_buckets
        if bucket is not None and wants_bucket and bucket not in seen:
            preferred.append(idx)
            seen.add(bucket)
        else:
            fallback.append(idx)
    for idx in preferred + fallback:
        if len(examples) >= max_examples:
            break
        bucket = int(t_end_index[idx].detach().cpu()) if torch.is_tensor(t_end_index) else None
        if target_buckets is not None and bucket not in target_buckets and len(examples) < len(target_buckets):
            continue
        examples.append(
            {
                "pred": pred[idx].detach().cpu(),
                "target": target[idx].detach().cpu(),
                "t_end_ns": (
                    float(t_end_ns[idx].detach().cpu())
                    if torch.is_tensor(t_end_ns)
                    else float("nan")
                ),
                "t_end_index": bucket,
                "frame_init": (
                    int(batch["frame_init"][idx].detach().cpu())
                    if torch.is_tensor(batch.get("frame_init"))
                    else (
                        int(batch["frame0"][idx].detach().cpu())
                        if torch.is_tensor(batch.get("frame0"))
                        else None
                    )
                ),
                "frame_target": (
                    int(batch["frame_target"][idx].detach().cpu())
                    if torch.is_tensor(batch.get("frame_target"))
                    else None
                ),
            }
        )
        if bucket is not None:
            existing.add(bucket)


@torch.no_grad()
def _collect_fixed_time_visual_rows(
    dataset: Any,
    model: torch.nn.Module,
    sampler,
    collate,
    device: torch.device,
    max_rows: int,
    visual_mode: str = "trajectory",
) -> list[dict[str, Any]]:
    if max_rows <= 0 or not hasattr(dataset, "visual_rows"):
        return []
    normalized_mode = str(visual_mode).strip().lower().replace("-", "_")
    use_rollout_rows = normalized_mode in {"rollout", "trajectory_rollout"} and hasattr(
        dataset,
        "rollout_visual_rows",
    )
    use_segment_start_rows = normalized_mode in {
        "segment_start",
        "segment_start_independent",
        "independent_segments",
        "segment_independent",
    } and hasattr(dataset, "trajectory_visual_rows")
    use_trajectory_rows = (
        not use_rollout_rows
        and not use_segment_start_rows
        and normalized_mode in {"trajectory", "legacy_trajectory"}
        and hasattr(dataset, "trajectory_visual_rows")
    )
    if use_rollout_rows:
        sample_rows = dataset.rollout_visual_rows(max_rows)
    elif use_segment_start_rows:
        sample_rows = dataset.trajectory_visual_rows(max_rows)
    elif use_trajectory_rows:
        sample_rows = dataset.trajectory_visual_rows(max_rows)
    else:
        sample_rows = dataset.visual_rows(max_rows)
    examples: list[dict[str, Any]] = []
    for row in sample_rows:
        initial = row[0]["m_init"].detach().cpu()
        cells = []
        segment_start_state: torch.Tensor | None = None
        next_segment_start_state: torch.Tensor | None = None
        rollout_state: torch.Tensor | None = None
        active_segment: int | None = None
        for sample in row:
            collatable_sample = {key: value for key, value in sample.items() if value is not None}
            batch = move_batch(default_collate([collatable_sample]), device)
            cond = collate(batch)
            m_init = batch.get("m_init", batch.get("m0"))
            observed_m_init = batch.get("m_observed_init", m_init)
            segment_index = (
                int(batch["visual_control_segment_index"][0].detach().cpu())
                if torch.is_tensor(batch.get("visual_control_segment_index"))
                else (
                    int(batch["control_segment_index"][0].detach().cpu())
                    if torch.is_tensor(batch.get("control_segment_index"))
                    else 0
                )
            )
            if use_rollout_rows:
                if rollout_state is None:
                    rollout_state = m_init
                if active_segment != segment_index:
                    active_segment = segment_index
                    frame_init = int(batch["frame_init"][0].detach().cpu())
                    frame_init_time_ns = (
                        float(batch["frame_init_time_ns"][0].detach().cpu())
                        if torch.is_tensor(batch.get("frame_init_time_ns"))
                        else float("nan")
                    )
                    cells.append(
                        {
                            "kind": "segment_input",
                            "target_state": observed_m_init[0].detach().cpu(),
                            "pred_state": rollout_state[0].detach().cpu(),
                            "segment_index": segment_index,
                            "frame": frame_init,
                            "abs_t_ns": frame_init_time_ns,
                            "run_id": sample.get("run_id"),
                        }
                    )
                sample_input = rollout_state
            elif use_segment_start_rows:
                if active_segment != segment_index:
                    active_segment = segment_index
                    frame_init = int(batch["frame_init"][0].detach().cpu())
                    frame_init_time_ns = (
                        float(batch["frame_init_time_ns"][0].detach().cpu())
                        if torch.is_tensor(batch.get("frame_init_time_ns"))
                        else float("nan")
                    )
                    cells.append(
                        {
                            "kind": "segment_input",
                            "target_state": observed_m_init[0].detach().cpu(),
                            "pred_state": m_init[0].detach().cpu(),
                            "segment_index": segment_index,
                            "frame": frame_init,
                            "abs_t_ns": frame_init_time_ns,
                            "run_id": sample.get("run_id"),
                        }
                    )
                # Every endpoint is predicted directly from its own segment's
                # observed/model-preconditioned start.  No state is inherited
                # from the preceding segment or preceding horizon.
                sample_input = m_init
            elif use_trajectory_rows:
                if active_segment != segment_index:
                    segment_start_state = next_segment_start_state if next_segment_start_state is not None else m_init
                    active_segment = segment_index
                    frame_init = int(batch["frame_init"][0].detach().cpu())
                    frame_init_time_ns = (
                        float(batch["frame_init_time_ns"][0].detach().cpu())
                        if torch.is_tensor(batch.get("frame_init_time_ns"))
                        else float("nan")
                    )
                    cells.append(
                        {
                            "kind": "segment_input",
                            "target_state": observed_m_init[0].detach().cpu(),
                            "pred_state": segment_start_state[0].detach().cpu(),
                            "segment_index": segment_index,
                            "frame": frame_init,
                            "abs_t_ns": frame_init_time_ns,
                            "run_id": sample.get("run_id"),
                        }
                    )
                sample_input = segment_start_state if segment_start_state is not None else m_init
            else:
                sample_input = m_init
            pred_m1, _ = sampler.sample(model, sample_input, cond)
            m_target = batch.get("m_t", batch.get("m1"))
            t_end_ns = float(batch["t_end_ns"][0].detach().cpu())
            abs_t_ns = (
                float(batch["frame_target_time_ns"][0].detach().cpu())
                if torch.is_tensor(batch.get("frame_target_time_ns"))
                else t_end_ns
            )
            segment_elapsed_ns = (
                float(batch["visual_segment_elapsed_ns"][0].detach().cpu())
                if torch.is_tensor(batch.get("visual_segment_elapsed_ns"))
                else None
            )
            is_segment_end = (
                bool(batch["visual_is_segment_end"][0].detach().cpu().item())
                if torch.is_tensor(batch.get("visual_is_segment_end"))
                else False
            )
            cells.append(
                {
                    "pred": pred_m1[0].detach().cpu(),
                    "target": m_target[0].detach().cpu(),
                    "t_end_ns": t_end_ns,
                    "abs_t_ns": abs_t_ns,
                    "t_end_index": int(batch["t_end_index"][0].detach().cpu()),
                    "frame_init": int(batch["frame_init"][0].detach().cpu()),
                    "frame_target": int(batch["frame_target"][0].detach().cpu()),
                    "segment_index": segment_index,
                    "segment_elapsed_ns": segment_elapsed_ns,
                    "rollout_step": (
                        int(batch["visual_rollout_step"][0].detach().cpu())
                        if torch.is_tensor(batch.get("visual_rollout_step"))
                        else None
                    ),
                    "boundary_transition": (
                        bool(batch["visual_boundary_transition"][0].detach().cpu().item())
                        if torch.is_tensor(batch.get("visual_boundary_transition"))
                        else False
                    ),
                    "run_id": sample.get("run_id"),
                }
            )
            if use_rollout_rows:
                rollout_state = pred_m1.detach()
            elif use_trajectory_rows and is_segment_end:
                next_segment_start_state = pred_m1.detach()
        if not use_trajectory_rows and not use_rollout_rows and not use_segment_start_rows:
            cells.sort(
                key=lambda item: (
                    float(item.get("abs_t_ns", item["t_end_ns"])),
                    int(item.get("frame_target", 0)),
                )
            )
        if use_rollout_rows:
            prediction_label = "rollout"
            mode_title = "fully autoregressive rollout"
        elif use_segment_start_rows:
            prediction_label = "segment-start pred"
            mode_title = "independent prediction from each MuMax segment start"
        else:
            prediction_label = "pred"
            mode_title = ""
        example = {
            "run_id": row[0].get("run_id"),
            "cells": cells,
            "prediction_label": prediction_label,
            "mode_title": mode_title,
        }
        if not use_trajectory_rows and not use_rollout_rows and not use_segment_start_rows:
            example["initial"] = initial
        examples.append(example)
    return examples


def _default_visual_bucket_indices(num_buckets: int, max_examples: int) -> set[int] | None:
    return evenly_spaced_bucket_indices(num_buckets, max_examples)


class EMA:
    _DYNAMIC_STATE_NAMES = ("filter", "expanded_bias")

    @classmethod
    def _trackable(cls, key: str, value: torch.Tensor) -> bool:
        return torch.is_floating_point(value) and key.rsplit(".", 1)[-1] not in cls._DYNAMIC_STATE_NAMES

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if self._trackable(k, v)
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for key, value in state.items():
            if key in self.shadow:
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        state = model.state_dict()
        shadow = {k: v for k, v in self.shadow.items() if k in state}
        backup = {k: state[k].detach().clone() for k in shadow}
        state.update(shadow)
        model.load_state_dict(state, strict=True)
        return backup

    def restore(self, model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        state = model.state_dict()
        state.update(backup)
        model.load_state_dict(state, strict=True)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def mask_batch(batch: dict[str, Any], mask: torch.Tensor) -> dict[str, Any]:
    out = {}
    sequence_mask: list[bool] | None = None
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == mask.shape:
            out[key] = value[mask]
        elif isinstance(value, (list, tuple)) and len(value) == mask.shape[0]:
            if sequence_mask is None:
                sequence_mask = mask.detach().to(device="cpu", dtype=torch.bool).tolist()
            selected = [item for item, keep in zip(value, sequence_mask, strict=True) if keep]
            out[key] = tuple(selected) if isinstance(value, tuple) else selected
        else:
            out[key] = value
    return out


def seed_data_worker(worker_id: int) -> None:
    torch.set_num_threads(1)


def dataloader_options(
    cfg: dict[str, Any],
    device: torch.device,
    *,
    split: str = "train",
) -> dict[str, Any]:
    if split not in {"train", "val"}:
        raise ValueError(f"unknown DataLoader split: {split}")
    train_cfg = cfg["train"]
    train_num_workers = int(train_cfg.get("num_workers", 4))
    if split == "train":
        num_workers = train_num_workers
        persistent_workers = bool(train_cfg.get("persistent_workers", True))
        prefetch_factor = int(train_cfg.get("prefetch_factor", 4))
    else:
        num_workers = int(train_cfg.get("val_num_workers", min(train_num_workers, 4)))
        persistent_workers = bool(train_cfg.get("val_persistent_workers", False))
        prefetch_factor = int(train_cfg.get("val_prefetch_factor", train_cfg.get("prefetch_factor", 4)))
    options: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        options["persistent_workers"] = persistent_workers
        options["prefetch_factor"] = prefetch_factor
        options["worker_init_fn"] = seed_data_worker
    return options


def init_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        timeout_minutes = int(os.environ.get("TORCH_DISTRIBUTED_TIMEOUT_MINUTES", "120"))
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_minutes))
    if distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return distributed, rank, local_rank, world_size


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def resolve_device(configured: str, distributed: bool, local_rank: int) -> torch.device:
    if distributed and torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    return get_device(configured)


def cycle_loader(loader: DataLoader, sampler: DistributedSampler | None = None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def _endpoint_spin_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    metric: str,
    mask: torch.Tensor | dict[str, Any] | None = None,
) -> torch.Tensor:
    metric = metric.lower()
    if metric == "mse":
        return masked_site_mean((pred - target).square().mean(dim=1), mask).mean()
    if metric in {"cosine", "angular_proxy"}:
        cos = (pred * target).sum(dim=1).clamp(-1.0, 1.0)
        return masked_site_mean(1.0 - cos, mask).mean()
    if metric == "hybrid":
        cos = (pred * target).sum(dim=1).clamp(-1.0, 1.0)
        mse = masked_site_mean((pred - target).square().mean(dim=1), mask).mean()
        angular = masked_site_mean(1.0 - cos, mask).mean()
        return mse + angular
    raise ValueError("train.rollout.endpoint_loss.metric must be mse, cosine, or hybrid")


def _endpoint_topo_loss(pred: torch.Tensor, target: torch.Tensor, boundary: str, metric: str) -> torch.Tensor:
    diff = topological_charge(pred, boundary=boundary) - topological_charge(target, boundary=boundary)
    metric = metric.lower()
    if metric in {"abs", "l1"}:
        return diff.abs().mean()
    if metric in {"mse", "l2"}:
        return diff.square().mean()
    raise ValueError("train.endpoint_loss.topo_metric must be abs or mse")


def _coarse_topo_density(m: torch.Tensor, boundary: str, kernel: int) -> torch.Tensor:
    density = topological_charge_density(m, boundary=boundary).unsqueeze(1)
    kernel = max(1, int(kernel))
    if kernel <= 1:
        return density
    # Pool patch sums, not means, so each coarse cell remains a local charge.
    return torch.nn.functional.avg_pool2d(density, kernel_size=kernel, stride=kernel) * float(kernel * kernel)


def _endpoint_topo_density_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    boundary: str,
    metric: str,
    kernel: int,
) -> torch.Tensor:
    pred_density = _coarse_topo_density(pred, boundary, kernel)
    target_density = _coarse_topo_density(target, boundary, kernel)
    diff = pred_density - target_density
    metric = metric.lower()
    if metric in {"abs", "l1"}:
        return diff.abs().mean()
    if metric in {"mse", "l2"}:
        return diff.square().mean()
    if metric in {"smooth_l1", "huber"}:
        return torch.nn.functional.smooth_l1_loss(pred_density, target_density)
    raise ValueError("train.endpoint_loss.topo_density_metric must be abs, mse, or smooth_l1")


def pushforward_loss(
    model,
    loss_fn,
    sampler,
    batch,
    *,
    needs_omega: bool,
    roll_ode_steps: int | None = None,
    endpoint_cfg: dict[str, Any] | None = None,
):
    """Pushforward rollout loss for clip batches (fields shaped ``(B, K, ...)``).

    At each hop, apply the normal CFM loss from the current rollout state to
    the true next frame, then (except after the last hop) roll the model forward
    with no grad and feed its detached output into the next supervised hop.
    This keeps the objective local while exposing the model to its own drifted
    input distribution. ``needs_omega`` is True for the α (rotation-vector)
    bridge, which supervises ``Ω = log_S2(m_init, m_t)``.

    ``roll_ode_steps`` optionally lowers the sampler's ODE steps for the no-grad
    rolls only (they dominate cost); supervised hops are direct CFM regressions.
    """
    steps = int(batch["m_init"].shape[1])

    def hop(i: int) -> dict[str, Any]:
        return {k: (v[:, i] if torch.is_tensor(v) else v) for k, v in batch.items()}

    full_ode_steps = sampler.ode_steps
    roll_steps = int(roll_ode_steps) if roll_ode_steps else full_ode_steps
    endpoint_cfg = endpoint_cfg or {}
    endpoint_enabled = bool(endpoint_cfg.get("enabled", False)) and float(endpoint_cfg.get("weight", 0.0)) > 0.0
    endpoint_weight = float(endpoint_cfg.get("weight", 0.0))
    endpoint_metric = str(endpoint_cfg.get("metric", "cosine"))
    endpoint_steps = int(endpoint_cfg.get("ode_steps", 0)) or roll_steps
    endpoint_topo_weight = float(endpoint_cfg.get("topo_weight", 0.0))
    endpoint_topo_metric = str(endpoint_cfg.get("topo_metric", "abs"))
    endpoint_topo_density_weight = float(endpoint_cfg.get("topo_density_weight", 0.0))
    endpoint_topo_density_metric = str(endpoint_cfg.get("topo_density_metric", "smooth_l1"))
    endpoint_topo_density_kernel = int(endpoint_cfg.get("topo_density_kernel", 16))
    state = batch["m_init"][:, 0]
    losses: list[LossOutput] = []
    endpoint = batch["m_init"].new_tensor(0.0)
    endpoint_topo = batch["m_init"].new_tensor(0.0)
    endpoint_topo_density = batch["m_init"].new_tensor(0.0)
    for i in range(steps):
        item = hop(i)
        cond_i = collate_fixed_time_conditions(item)
        true_next = item["m_t"]
        state_detached = state.detach()
        pf_batch: dict[str, Any] = {"m_init": state_detached, "m_t": true_next}
        if needs_omega:
            # state / true_next are batched (B, 3, H, W); log_map_chw consumes CHW.
            pf_batch["omega_target"] = log_map_chw(state_detached, true_next).clone()
        losses.append(loss_fn(model, pf_batch, cond_i))
        if i == steps - 1:
            if endpoint_enabled:
                sampler.ode_steps = endpoint_steps
                try:
                    pred_final, _ = sampler.sample_with_grad(model, state_detached, cond_i)
                finally:
                    sampler.ode_steps = full_ode_steps
                endpoint = _endpoint_spin_loss(pred_final, true_next, endpoint_metric, cond_i)
                if endpoint_topo_weight > 0.0:
                    endpoint_topo = _endpoint_topo_loss(
                        pred_final,
                        true_next,
                        loss_fn.boundary,
                        endpoint_topo_metric,
                    )
                if endpoint_topo_density_weight > 0.0:
                    endpoint_topo_density = _endpoint_topo_density_loss(
                        pred_final,
                        true_next,
                        loss_fn.boundary,
                        endpoint_topo_density_metric,
                        endpoint_topo_density_kernel,
                    )
            break
        sampler.ode_steps = roll_steps
        try:
            state, _ = sampler.sample(model, state, cond_i)  # @torch.no_grad inside
        finally:
            sampler.ode_steps = full_ode_steps

    inv_n = 1.0 / len(losses)
    out = LossOutput(
        total=sum(x.total for x in losses) * inv_n,
        cfm=sum(x.cfm for x in losses) * inv_n,
        unit=sum(x.unit for x in losses) * inv_n,
        llg=sum(x.llg for x in losses) * inv_n,
        topo=sum(x.topo for x in losses) * inv_n,
        endpoint=endpoint.detach(),
    )
    out.total = (
        out.total
        + endpoint_weight * endpoint
        + endpoint_topo_weight * endpoint_topo
        + endpoint_topo_density_weight * endpoint_topo_density
    )
    out.topo = out.topo + endpoint_topo.detach() + endpoint_topo_density.detach()
    return out


def _prefixed_batch(batch: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        key[len(prefix):]: value
        for key, value in batch.items()
        if key.startswith(prefix)
    }


def segment_pushforward_loss(
    model,
    loss_fn,
    batch: dict[str, Any],
    *,
    needs_omega: bool,
    cfg: dict[str, Any] | None = None,
    handoff_cache: SegmentHandoffCache | None = None,
    physical_target_cache: SegmentPhysicalTargetCache | None = None,
) -> LossOutput:
    """Stage-2 segment handoff loss with one initial state per sample.

    Every non-first segment independently chooses either its real initial state
    or one cached stage-1 prediction of the previous segment endpoint. The
    resulting mixed batch is passed through the student exactly once.  In the
    optional ``mumax_replay`` target mode, the cached predicted start and the
    MuMax3 endpoint continued from that exact start are replaced together.
    Teacher inference and MuMax3 execution are never performed in the training
    step.
    """
    cfg = cfg or {}
    true_weight = float(cfg.get("true_weight", 1.0))
    pred_weight = float(cfg.get("pred_weight", 1.0))
    if true_weight < 0.0 or pred_weight < 0.0:
        raise ValueError("segment_pushforward true_weight and pred_weight must be >= 0")
    if true_weight + pred_weight <= 0.0:
        raise ValueError(
            "segment_pushforward true_weight and pred_weight cannot both be zero"
        )

    selected = batch.get("use_predicted_init")
    if not torch.is_tensor(selected):
        if pred_weight > 0.0:
            raise KeyError(
                "segment pushforward batches require use_predicted_init"
            )
        selected = torch.zeros(
            batch["m_init"].shape[0],
            dtype=torch.bool,
            device=batch["m_init"].device,
        )
    predicted_mask = selected.to(dtype=torch.bool)
    has_prev = batch.get("has_prev_segment")
    if torch.is_tensor(has_prev):
        predicted_mask = predicted_mask & has_prev.to(dtype=torch.bool)

    train_batch = batch
    target_mode = str(cfg.get("target_mode", "original_gt")).lower()
    if target_mode not in {"original_gt", "mumax_replay"}:
        raise ValueError(
            "segment_pushforward target_mode must be original_gt|mumax_replay"
        )
    if target_mode == "mumax_replay" and physical_target_cache is None:
        raise RuntimeError(
            "mumax_replay target mode requires a precomputed physical target cache"
        )
    if bool(predicted_mask.any().item()):
        if handoff_cache is None:
            raise RuntimeError(
                "segment pushforward predictions must be precomputed before training"
            )
        selected_batch = mask_batch(batch, predicted_mask)
        pred_init = handoff_cache.lookup(
            selected_batch,
            device=selected_batch["m_init"].device,
            dtype=selected_batch["m_init"].dtype,
        ).detach()
        if target_mode == "mumax_replay" and bool(
            cfg.get("normalize_replay_init", True)
        ):
            pred_norm = pred_init.norm(dim=1, keepdim=True)
            pred_mask = (pred_norm > 0.5).to(dtype=pred_init.dtype)
            pred_init = pred_init / pred_norm.clamp_min(1.0e-8) * pred_mask
        mixed_init = batch["m_init"].clone()
        mixed_init[predicted_mask] = pred_init
        mixed_target = batch["m_t"]
        if target_mode == "mumax_replay":
            assert physical_target_cache is not None
            physical_target = physical_target_cache.lookup(
                selected_batch,
                device=selected_batch["m_t"].device,
                dtype=selected_batch["m_t"].dtype,
            ).detach()
            mixed_target = batch["m_t"].clone()
            mixed_target[predicted_mask] = physical_target
        train_batch = dict(batch)
        train_batch["m_init"] = mixed_init
        train_batch["m0"] = mixed_init
        train_batch["m_t"] = mixed_target
        train_batch["m1"] = mixed_target
        if needs_omega:
            recomputed_omega = log_map_chw(mixed_init, mixed_target)
            omega_target = batch.get("omega_target")
            if torch.is_tensor(omega_target):
                omega_mask = predicted_mask.reshape(
                    predicted_mask.shape[0],
                    *([1] * (recomputed_omega.ndim - 1)),
                )
                omega_target = torch.where(
                    omega_mask,
                    recomputed_omega,
                    omega_target,
                )
            else:
                omega_target = recomputed_omega
            train_batch["omega_target"] = omega_target

    cond = collate_fixed_time_conditions(train_batch)
    return loss_fn(model, train_batch, cond)


def add_endpoint_consistency_loss(
    loss: LossOutput,
    model,
    sampler,
    batch: dict[str, Any],
    cond: dict[str, torch.Tensor],
    endpoint_cfg: dict[str, Any] | None,
    *,
    step: int | None = None,
    boundary: str = "open",
) -> LossOutput:
    endpoint_cfg = endpoint_cfg or {}
    weight = float(endpoint_cfg.get("weight", 0.0))
    if not bool(endpoint_cfg.get("enabled", False)) or weight <= 0.0:
        return loss
    every_n = max(1, int(endpoint_cfg.get("every_n_steps", 1)))
    if step is not None and every_n > 1 and int(step) % every_n != 0:
        return loss
    m_init = batch.get("m_init", batch.get("m0"))
    m_target = batch.get("m_t", batch.get("m1"))
    if m_init is None or m_target is None:
        return loss
    max_batch = int(endpoint_cfg.get("max_batch_per_rank", 0) or 0)
    if max_batch > 0 and int(m_init.shape[0]) > max_batch:
        n = max_batch
        m_init = m_init[:n]
        m_target = m_target[:n]
        cond = {
            key: value[:n] if torch.is_tensor(value) and value.shape[:1] == batch.get("m_init", batch.get("m0")).shape[:1] else value
            for key, value in cond.items()
        }
    full_ode_steps = sampler.ode_steps
    full_cfg = sampler.classifier_free_guidance
    full_grad_checkpoint = bool(getattr(sampler, "grad_checkpoint", False))
    endpoint_steps = int(endpoint_cfg.get("ode_steps", 0)) or full_ode_steps
    metric = str(endpoint_cfg.get("metric", "cosine"))
    topo_weight = float(endpoint_cfg.get("topo_weight", 0.0))
    topo_metric = str(endpoint_cfg.get("topo_metric", "abs"))
    topo_density_weight = float(endpoint_cfg.get("topo_density_weight", 0.0))
    topo_density_metric = str(endpoint_cfg.get("topo_density_metric", "smooth_l1"))
    topo_density_kernel = int(endpoint_cfg.get("topo_density_kernel", 16))
    sampler.ode_steps = endpoint_steps
    sampler.grad_checkpoint = bool(endpoint_cfg.get("checkpoint", full_grad_checkpoint))
    if bool(endpoint_cfg.get("disable_cfg", False)):
        sampler.classifier_free_guidance = {}
    try:
        pred_final, _ = sampler.sample_with_grad(model, m_init, cond)
    finally:
        sampler.ode_steps = full_ode_steps
        sampler.classifier_free_guidance = full_cfg
        sampler.grad_checkpoint = full_grad_checkpoint
    endpoint = _endpoint_spin_loss(pred_final, m_target, metric, cond)
    if topo_weight > 0.0:
        endpoint_topo = _endpoint_topo_loss(pred_final, m_target, boundary, topo_metric)
        loss.total = loss.total + weight * endpoint + topo_weight * endpoint_topo
        loss.topo = loss.topo + endpoint_topo.detach()
    else:
        loss.total = loss.total + weight * endpoint
    if topo_density_weight > 0.0:
        endpoint_topo_density = _endpoint_topo_density_loss(
            pred_final,
            m_target,
            boundary,
            topo_density_metric,
            topo_density_kernel,
        )
        loss.total = loss.total + topo_density_weight * endpoint_topo_density
        loss.topo = loss.topo + endpoint_topo_density.detach()
    loss.endpoint = endpoint.detach()
    return loss


def add_aux_endpoint_cfm_loss(
    loss: LossOutput,
    model,
    loss_fn,
    batch: dict[str, Any],
    aux_cfg: dict[str, Any] | None,
) -> LossOutput:
    aux_cfg = aux_cfg or {}
    weight = float(aux_cfg.get("weight", 0.0))
    if not bool(aux_cfg.get("enabled", False)) or weight <= 0.0 or "aux_m_t" not in batch:
        return loss
    aux_m_t = batch["aux_m_t"]
    if not torch.is_tensor(aux_m_t) or aux_m_t.ndim < 3:
        return loss
    batch_size, aux_count = int(aux_m_t.shape[0]), int(aux_m_t.shape[1])
    if aux_count <= 0:
        return loss
    max_batch = int(aux_cfg.get("max_batch_per_rank", 0) or 0)
    if max_batch > 0 and batch_size > max_batch:
        batch_size = max_batch
        aux_m_t = aux_m_t[:batch_size]
        batch = {
            key: (value[:batch_size] if torch.is_tensor(value) and value.shape[:1] == batch["aux_m_t"].shape[:1] else value)
            for key, value in batch.items()
        }

    aux_batch: dict[str, Any] = {}
    for key, value in batch.items():
        if key.startswith("aux_") or not torch.is_tensor(value):
            continue
        if value.shape[:1] == (batch_size,):
            aux_batch[key] = value.unsqueeze(1).expand(-1, aux_count, *value.shape[1:]).reshape(
                batch_size * aux_count,
                *value.shape[1:],
            )
        else:
            aux_batch[key] = value

    aux_batch["m_t"] = aux_m_t.reshape(batch_size * aux_count, *aux_m_t.shape[2:])
    aux_batch["m1"] = aux_batch["m_t"]
    if "aux_omega_target" in batch:
        aux_omega = batch["aux_omega_target"]
        aux_batch["omega_target"] = aux_omega.reshape(batch_size * aux_count, *aux_omega.shape[2:])
    aux_fields = {
        "t_end_ns": "aux_t_end_ns",
        "t_end_s": "aux_t_end_s",
        "t_end_index": "aux_t_end_index",
        "dt_s": "aux_t_end_s",
        "dt_index": "aux_t_end_index",
        "frame_target": "aux_frame_target",
        "dt_scale": "aux_frame_target",
        "drive_time_s": "aux_drive_time_s",
        "relax_time_s": "aux_relax_time_s",
        "drive_fraction": "aux_drive_fraction",
    }
    for out_key, in_key in aux_fields.items():
        if in_key in batch:
            value = batch[in_key]
            aux_batch[out_key] = value.reshape(batch_size * aux_count, *value.shape[2:])
    aux_cond = collate_fixed_time_conditions(aux_batch)
    aux_loss = loss_fn(model, aux_batch, aux_cond)
    loss.total = loss.total + weight * _loss_item(aux_loss, "total")
    return loss


def per_rank_train_batch_size(cfg: dict[str, Any], world_size: int) -> int:
    batch_size = int(cfg["train"]["batch_size"])
    if world_size <= 1:
        return batch_size
    if batch_size % world_size != 0:
        raise ValueError(
            f"train.batch_size={batch_size} must be divisible by WORLD_SIZE={world_size} "
            "to keep the global batch size unchanged under DDP."
        )
    per_rank = batch_size // world_size
    if per_rank < 1:
        raise ValueError(
            f"train.batch_size={batch_size} is too small for WORLD_SIZE={world_size}; "
            "each rank needs at least one sample."
        )
    return per_rank


def cosine_lr(step: int, cfg: dict[str, Any], base_lr: float) -> float:
    warmup = int(cfg["train"].get("warmup_steps", 0))
    steps = int(cfg["train"]["steps"])
    min_lr = float(cfg["train"].get("min_lr", 0.0))
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _should_run_validation(completed_step: int, total_steps: int, val_every: int) -> bool:
    """Validate on the requested cadence and always after the final update."""
    if val_every <= 0:
        raise ValueError("train.val_every must be positive")
    return completed_step > 0 and (
        completed_step % val_every == 0 or completed_step >= total_steps
    )


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _normalize_t_end_probs(probs: Any, n: int) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(f"t_end probability list must have length {n}, got {arr.shape}")
    if np.any(arr < 0.0) or float(arr.sum()) <= 0.0:
        raise ValueError("t_end probabilities must be non-negative and sum to > 0")
    return arr / arr.sum()


def _t_end_curriculum_stage(
    schedule: list[dict[str, Any]],
    step: int,
) -> tuple[int, np.ndarray | None]:
    for idx, item in enumerate(schedule):
        until_step = item.get("until_step")
        if until_step is None or int(step) < int(until_step):
            probs = item.get("probs")
            return idx, None if probs is None else np.asarray(probs, dtype=np.float64)
    item = schedule[-1]
    probs = item.get("probs")
    return len(schedule) - 1, None if probs is None else np.asarray(probs, dtype=np.float64)


class StepTimer:
    def __init__(self, enabled: bool, device: torch.device) -> None:
        self.enabled = enabled
        self.device = device
        self.totals: dict[str, float] = {}
        self._last = 0.0

    def start(self) -> None:
        if not self.enabled:
            return
        self._sync()
        self._last = time.perf_counter()

    def mark(self, name: str) -> None:
        if not self.enabled:
            return
        self._sync()
        now = time.perf_counter()
        self.totals[name] = self.totals.get(name, 0.0) + now - self._last
        self._last = now

    def reset(self) -> None:
        self.totals.clear()

    def report(self, steps: int) -> dict[str, float]:
        denom = max(1, int(steps))
        return {f"time_{key}_ms": 1000.0 * value / denom for key, value in self.totals.items()}

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


def make_condition_stats(train_records, dt_scales: list[int], dt_raw_ps: float) -> ConditionStats:
    rows = []
    for rec in train_records:
        for scale in dt_scales:
            row = rec.condition_row(dt_s=scale * dt_raw_ps * 1e-12)
            row.update(rec.material_row())
            rows.append(row)
    return ConditionStats.from_records(rows)


def _v4_segment_condition_rows(rec, t_end_ns: list[float]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for seg in rec.params.get("segments", []):
        if not bool(seg.get("pair_sampling_allowed", True)):
            continue
        start_s = float(seg.get("start_s", 0.0))
        end_s = float(seg.get("end_s", start_s))
        if end_s <= start_s:
            continue
        duration_s = end_s - start_s
        for t_ns in t_end_ns:
            horizon_s = min(duration_s, max(0.0, float(t_ns) * 1e-9))
            if horizon_s <= 0.0:
                continue
            anchors = [start_s]
            if duration_s > horizon_s:
                anchors.append(start_s + 0.5 * (duration_s - horizon_s))
                anchors.append(end_s - horizon_s)
            seen: set[float] = set()
            for anchor_s in anchors:
                key = round(anchor_s, 18)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(rec.v4_condition_row(anchor_s, anchor_s + horizon_s))
    return rows


def make_fixed_time_condition_stats(train_records, t_end_ns: list[float]) -> ConditionStats:
    rows = []
    for rec in train_records:
        if getattr(rec, "is_v4", False):
            rows.extend(_v4_segment_condition_rows(rec, t_end_ns))
            continue
        for t_ns in t_end_ns:
            row = rec.condition_row(dt_s=float(t_ns) * 1e-9)
            row["t_end_s"] = float(t_ns) * 1e-9
            row.update(rec.material_row())
            rows.append(row)
    return ConditionStats.from_records(rows)


def make_training_stats(train_ds, cfg: dict[str, Any]) -> TrainingStats:
    data_cfg = cfg["data"]
    fixed_time = bool(data_cfg.get("t_end_ns"))
    if fixed_time:
        t_end_ns = [float(x) for x in data_cfg["t_end_ns"]]
        # TrainingStats keeps the historical name ``dt_scales`` because the
        # empirical prior indexes this tensor by ``dt_index`` / ``t_end_index``.
        # For fixed-time training the stable bucket ids are simply 0..K-1.
        dt_scales = list(range(len(t_end_ns)))
    else:
        dt_scales = list(data_cfg["dt_scales"])
    cache = data_cfg.get("stats_cache")
    if cache is not None and Path(cache).exists():
        stats = load_training_stats(cache)
        cache_matches_dataset = (
            stats.schema_version == STATS_SCHEMA_VERSION
            and stats.dt_scales == dt_scales
            and stats.dataset_root == str(data_cfg.get("dataset_root"))
        )
        cache_has_all_scales = (
            stats.omega_count.numel() == len(dt_scales)
            and bool(torch.all(stats.omega_count > 0))
        )
        cache_has_kappa = (
            not bool(cfg.get("prior", {}).get("fit_kappa", False))
            or stats.physics_kappa is not None
        )
        if cache_matches_dataset and cache_has_all_scales and cache_has_kappa:
            return stats
    if fixed_time:
        cond_stats = make_fixed_time_condition_stats(train_ds.records, t_end_ns)
    else:
        cond_stats = make_condition_stats(
            train_ds.index.records,
            dt_scales,
            float(data_cfg.get("dt_raw_ps", 5.0)),
        )
    omega_std, omega_count = estimate_omega_stats(
        train_ds,
        dt_scales,
        int(data_cfg.get("stats_max_samples", 4096)),
    )
    physics_kappa, physics_kappa_count = fit_physics_kappa(
        train_ds,
        int(data_cfg.get("kappa_fit_max_samples", data_cfg.get("stats_max_samples", 4096))),
    )
    stats = TrainingStats(
        cond_stats,
        dt_scales,
        omega_std,
        omega_count,
        str(data_cfg.get("dataset_root")),
        physics_kappa,
        physics_kappa_count,
    )
    if cache is not None:
        save_training_stats(cache, stats)
    return stats


def make_training_stats_synced(train_ds, cfg: dict[str, Any]) -> TrainingStats:
    if not (dist.is_available() and dist.is_initialized()):
        return make_training_stats(train_ds, cfg)
    payload: list[TrainingStats | None] = [None]
    if is_main_process():
        payload[0] = make_training_stats(train_ds, cfg)
    dist.broadcast_object_list(payload, src=0)
    assert payload[0] is not None
    return payload[0]


def _runtime_needs_omega_target(cfg: dict[str, Any]) -> bool:
    """Whether normal train/val batches need per-sample Ω targets after stats.

    ``prior.fit_kappa`` needs real Ω samples while building ``TrainingStats``.
    Once the cached/fitted stats exist, non-α bridges do not consume
    ``omega_target`` during ordinary training. Leaving it enabled makes every
    fixed-time ``__getitem__`` pay a full per-site log-map cost and starves the
    GPU data path.
    """
    data_cfg = cfg["data"]
    bridge_cfg = cfg.get("bridge", {})
    loss_cfg = cfg.get("train", {}).get("loss", {})
    return (
        bridge_state_repr(bridge_cfg) in {"alpha", "alpha2d"}
        or float(loss_cfg.get("alpha_mix", 0.0)) > 0.0
        or bool(data_cfg.get("force_omega_target", False))
    )


def disable_unused_fixed_time_omega_targets(train_ds, val_ds, cfg: dict[str, Any]) -> None:
    if not bool(cfg["data"].get("t_end_ns")):
        return
    if _runtime_needs_omega_target(cfg):
        return
    for ds in (train_ds, val_ds):
        if hasattr(ds, "compute_omega_target"):
            ds.compute_omega_target = False


def make_prior(cfg: dict[str, Any], stats: TrainingStats | None = None) -> RotationPrior:
    prior_cfg = cfg["prior"]
    prior_type = prior_cfg.get("type", "physics_scaled")
    omega_std = stats.omega_std if stats is not None and prior_type.startswith("empirical") else None
    kappa = float(prior_cfg.get("kappa", 1.25))
    if prior_type == "physics_scaled" and bool(prior_cfg.get("fit_kappa", False)):
        if stats is None or stats.physics_kappa is None:
            raise ValueError(
                "prior.fit_kappa=true requires positive-temperature training samples to fit kappa"
            )
        kappa = float(stats.physics_kappa) * float(prior_cfg.get("kappa_multiplier", 1.0))
    return RotationPrior(
        prior_type=prior_type,
        kappa=kappa,
        min_sigma=float(prior_cfg.get("min_sigma", 1e-4)),
        dt_scales=list(cfg["data"].get("dt_scales", [])),
        omega_std=omega_std,
    )


def make_bridge_and_priors(cfg: dict[str, Any], stats: TrainingStats | None = None):
    """Plan-2 entry point: build (bridge, rotation_prior, cart_prior, rfm_prior)."""
    rotation_prior = make_prior(cfg, stats)
    bridge_cfg = cfg.get("bridge", {})
    bridge = make_bridge(bridge_cfg or {"state_repr": "cart"})
    state_repr = bridge_state_repr(bridge_cfg or {"state_repr": "cart"})
    source_mode = bridge_source_mode(bridge_cfg, state_repr)
    cart_mode = (
        source_mode
        if state_repr == "cart"
        else str(bridge_cfg.get("cart", {}).get("source", "anchored"))
    )
    rfm_mode = (
        source_mode
        if state_repr == "rfm"
        else str(bridge_cfg.get("rfm", {}).get("source", "anchored"))
    )
    cart_prior = CartSourcePrior(
        mode=cart_mode,
        min_sigma=float(cfg.get("prior", {}).get("min_sigma", 1e-4)),
    )
    rfm_prior = RFMSourcePrior(
        mode=rfm_mode,
        min_sigma=float(cfg.get("prior", {}).get("min_sigma", 1e-4)),
        enabled=bool(bridge_cfg.get("rfm", {}).get("source_noise", True)),
    )
    # Consistency assert: the bridge and prior must agree on source mode,
    # otherwise the loss/sampler train one bridge but sample another. plan-2
    # treats anchored / latent as two separate training runs.
    if bridge.mode == "cart" and getattr(bridge, "source", None) != cart_mode:
        raise ValueError(
            f"CartBridge.source={getattr(bridge, 'source', None)!r} != "
            f"prior.cart.source={cart_mode!r}; fix the yaml."
        )
    if bridge.mode == "rfm" and getattr(bridge, "source", None) != rfm_mode:
        raise ValueError(
            f"RFMBridge.source={getattr(bridge, 'source', None)!r} != "
            f"prior.rfm.source={rfm_mode!r}; fix the yaml."
        )
    return bridge, rotation_prior, cart_prior, rfm_prior


def _strip_compile_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    if not any(k.startswith(prefix) for k in state):
        return state
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}


def _resume_model_core(model: torch.nn.Module) -> torch.nn.Module:
    core = model
    while True:
        if hasattr(core, "module"):
            core = core.module
            continue
        if hasattr(core, "_orig_mod"):
            core = core._orig_mod
            continue
        return core


def _mapped_channel_indices(
    source_channels: int,
    target_channels: int,
    *,
    concatenated: bool,
    device: torch.device,
) -> torch.Tensor | None:
    if source_channels > target_channels:
        return None
    if not concatenated:
        return torch.arange(source_channels, device=device)
    if source_channels % 2 or target_channels % 2:
        return None
    source_half = source_channels // 2
    target_half = target_channels // 2
    if source_half > target_half:
        return None
    return torch.cat(
        [
            torch.arange(source_half, device=device),
            target_half + torch.arange(source_half, device=device),
        ]
    )


def _paired_modulation_indices(
    source_features: int,
    target_features: int,
    *,
    concatenated: bool,
    device: torch.device,
) -> torch.Tensor | None:
    if source_features % 2 or target_features % 2:
        return None
    source_channels = source_features // 2
    target_channels = target_features // 2
    channel_indices = _mapped_channel_indices(
        source_channels,
        target_channels,
        concatenated=concatenated,
        device=device,
    )
    if channel_indices is None:
        return None
    return torch.cat([channel_indices, target_channels + channel_indices])


def _adapt_unet_channel_expansion_tensor(
    key: str,
    value: torch.Tensor,
    target_value: torch.Tensor,
) -> torch.Tensor | None:
    """Function-preserving 64→wider U-Net tensor transplant.

    Existing output channels are isolated from newly added input channels at
    step 0, while new output channels keep their normal random initialization.
    The first block on the up path needs a non-contiguous input map because its
    input is ``cat([decoder, skip])``.
    """

    bare_key = key.removeprefix("_orig_mod.")
    width_prefixes = (
        "in_conv.",
        "down_blocks.",
        "downsamples.",
        "mid1.",
        "mid2.",
        "up_blocks.",
        "upsamples.",
        "out_norm.",
        "out_conv.",
    )
    if not bare_key.startswith(width_prefixes):
        return None
    if not torch.is_tensor(value) or value.ndim != target_value.ndim:
        return None
    if any(source > target for source, target in zip(value.shape, target_value.shape)):
        return None

    device = value.device
    adapted = target_value.detach().to(device=device, dtype=value.dtype).clone()
    up_first = re.match(r"up_blocks\.\d+\.0\.", bare_key) is not None

    if value.ndim == 4:
        if value.shape[2:] != target_value.shape[2:]:
            return None
        input_is_concat = up_first and (
            bare_key.endswith(".conv1.weight") or bare_key.endswith(".skip.weight")
        )
        output_indices = _mapped_channel_indices(
            int(value.shape[0]),
            int(target_value.shape[0]),
            concatenated=False,
            device=device,
        )
        input_indices = _mapped_channel_indices(
            int(value.shape[1]),
            int(target_value.shape[1]),
            concatenated=input_is_concat,
            device=device,
        )
        if output_indices is None or input_indices is None:
            return None
        adapted[output_indices] = 0
        adapted[output_indices[:, None], input_indices[None, :]] = value
        return adapted

    modulation = re.search(
        r"(?:norm[12]\.mod\.1|time_mod[12]\.1)\.(?:weight|bias)$",
        bare_key,
    )
    if value.ndim == 2 and modulation is not None:
        if value.shape[1] != target_value.shape[1]:
            return None
        modulation_is_concat = up_first and (
            ".norm1.mod.1." in bare_key or ".time_mod1.1." in bare_key
        )
        output_indices = _paired_modulation_indices(
            int(value.shape[0]),
            int(target_value.shape[0]),
            concatenated=modulation_is_concat,
            device=device,
        )
        if output_indices is None:
            return None
        adapted.index_copy_(0, output_indices, value)
        return adapted

    if value.ndim == 1:
        if modulation is not None:
            modulation_is_concat = up_first and (
                ".norm1.mod.1." in bare_key or ".time_mod1.1." in bare_key
            )
            output_indices = _paired_modulation_indices(
                int(value.shape[0]),
                int(target_value.shape[0]),
                concatenated=modulation_is_concat,
                device=device,
            )
        else:
            vector_is_concat = up_first and ".norm1.norm." in bare_key
            output_indices = _mapped_channel_indices(
                int(value.shape[0]),
                int(target_value.shape[0]),
                concatenated=vector_is_concat,
                device=device,
            )
        if output_indices is None:
            return None
        adapted.index_copy_(0, output_indices, value)
        return adapted

    return None


def _adapt_resume_state_to_model(
    state: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Adapt known-compatible checkpoint shape changes before strict loading.

    Pulse-time conditioning appends features before ``cond_embed.project``.
    Explicit U-Net channel expansion can also transplant a narrow convolutional
    trunk into a wider one while isolating the new channels at step 0.
    """

    target = model.state_dict()
    compile_prefix = "_orig_mod."
    if state and target and not any(key in target for key in state):
        target_prefixed = all(key.startswith(compile_prefix) for key in target)
        state_prefixed = all(key.startswith(compile_prefix) for key in state)
        if target_prefixed and not state_prefixed:
            state = {f"{compile_prefix}{key}": value for key, value in state.items()}
        elif state_prefixed and not target_prefixed:
            state = _strip_compile_prefix(state)

    expand_unet_channels = bool(
        getattr(_resume_model_core(model), "checkpoint_channel_expansion", False)
    )
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        target_value = target.get(key)
        if (
            key.endswith("cond_embed.project.weight")
            and target_value is not None
            and torch.is_tensor(value)
            and value.ndim == 2
            and target_value.ndim == 2
            and value.shape[0] == target_value.shape[0]
            and value.shape[1] < target_value.shape[1]
        ):
            adapted = torch.zeros_like(target_value, device=value.device)
            adapted[:, : value.shape[1]].copy_(value)
            out[key] = adapted
        elif (
            expand_unet_channels
            and target_value is not None
            and torch.is_tensor(value)
            and tuple(value.shape) != tuple(target_value.shape)
        ):
            adapted = _adapt_unet_channel_expansion_tensor(key, value, target_value)
            out[key] = value if adapted is None else adapted
        else:
            out[key] = value
    return out


def build_rectification_teacher(
    rect_cfg: dict[str, Any],
    fallback_cfg: dict[str, Any],
    stats: TrainingStats,
    device: torch.device,
) -> tuple[torch.nn.Module, BridgeSampler] | None:
    if not bool(rect_cfg.get("enabled", False)):
        return None
    if rect_cfg.get("cache_path") and not bool(rect_cfg.get("online", False)):
        return None
    ckpt_path = rect_cfg.get("teacher_checkpoint")
    if not ckpt_path:
        raise ValueError("train.rectification.teacher_checkpoint is required when rectification is enabled")
    ckpt = torch.load(Path(ckpt_path), map_location=device, weights_only=False)
    if rect_cfg.get("teacher_config"):
        teacher_cfg = load_config(rect_cfg["teacher_config"])
    else:
        teacher_cfg = ckpt.get("config", fallback_cfg)
    # Reuse the current dataset stats but keep the teacher's own condition audit
    # if it was saved in the checkpoint config.
    teacher = build_model(teacher_cfg, stats.condition).to(device)
    state_name = str(rect_cfg.get("teacher_state", "ema")).lower()
    if state_name == "raw":
        payload = ckpt["model"]
    else:
        payload = ckpt.get("ema") or ckpt["model"]
    teacher.load_state_dict(_strip_compile_prefix(payload), strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    bridge, rotation_prior, cart_prior, rfm_prior = make_bridge_and_priors(teacher_cfg, stats)
    teacher_sampler_cfg = teacher_cfg.get("sampler", {})
    guidance_cfg = teacher_sampler_cfg.get("classifier_free_guidance")
    if "teacher_cfg_scale" in rect_cfg:
        guidance_cfg = dict(guidance_cfg or {})
        guidance_cfg["enabled"] = float(rect_cfg["teacher_cfg_scale"]) != 1.0
        guidance_cfg["scale"] = float(rect_cfg["teacher_cfg_scale"])
    stochastic_cfg = None
    if bool(rect_cfg.get("teacher_stochastic", False)):
        stochastic_cfg = teacher_sampler_cfg.get("stochastic_sampler")
    sampler = BridgeSampler(
        bridge=bridge,
        rotation_prior=rotation_prior,
        cart_prior=cart_prior,
        rfm_prior=rfm_prior,
        ode_steps=int(rect_cfg.get("teacher_ode_steps", teacher_sampler_cfg.get("ode_steps", 20))),
        method=str(rect_cfg.get("teacher_method", teacher_sampler_cfg.get("method", "heun"))),
        classifier_free_guidance=guidance_cfg,
        stochastic_sampler=stochastic_cfg,
    )
    return teacher, sampler


class RectificationEndpointCache:
    """Random-access teacher endpoint cache keyed by fixed-time sample identity."""

    def __init__(self, path: str | Path, *, strict: bool = True) -> None:
        root = Path(path)
        index_path = root / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Rectification cache index not found: {index_path}")
        with index_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        self.root = root
        self.strict = bool(strict)
        self.shape = tuple(int(x) for x in meta["shape"])
        self.dtype = np.dtype(meta.get("dtype", "float16"))
        self.keys = {str(k): int(v) for k, v in meta["keys"].items()}
        self.data = np.memmap(root / meta["data_file"], mode="r", dtype=self.dtype, shape=self.shape)

    @staticmethod
    def key(run_id: str, frame_init: int, frame_target: int) -> str:
        return f"{run_id}|{int(frame_init)}|{int(frame_target)}"

    def lookup(self, batch: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        run_ids = batch.get("run_id")
        frame_init = batch.get("frame_init")
        frame_target = batch.get("frame_target")
        if run_ids is None or frame_init is None or frame_target is None:
            if self.strict:
                raise KeyError("Rectification cache requires run_id, frame_init, and frame_target in batch")
            return None
        if torch.is_tensor(frame_init):
            frame_init_values = frame_init.detach().cpu().tolist()
        else:
            frame_init_values = list(frame_init)
        if torch.is_tensor(frame_target):
            frame_target_values = frame_target.detach().cpu().tolist()
        else:
            frame_target_values = list(frame_target)
        arrays = []
        missing = []
        for run_id, fi, ft in zip(run_ids, frame_init_values, frame_target_values):
            key = self.key(str(run_id), int(fi), int(ft))
            idx = self.keys.get(key)
            if idx is None:
                missing.append(key)
                continue
            arrays.append(np.asarray(self.data[idx], dtype=np.float32))
        if missing:
            if self.strict:
                preview = ", ".join(missing[:3])
                raise KeyError(f"Rectification cache miss for {len(missing)} samples: {preview}")
            return None
        arr = np.stack(arrays, axis=0)
        return torch.from_numpy(arr).to(device=device, dtype=dtype, non_blocking=True)


def _apply_spin_augment_batch(m: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
    rot_k = batch.get("augment_rot_k")
    spin_flip = batch.get("augment_spin_flip")
    if rot_k is None and spin_flip is None:
        return m
    if rot_k is None:
        rot_values = [0] * int(m.shape[0])
    elif torch.is_tensor(rot_k):
        rot_values = [int(x) % 4 for x in rot_k.detach().cpu().tolist()]
    else:
        rot_values = [int(x) % 4 for x in rot_k]
    if spin_flip is None:
        flip_values = [False] * int(m.shape[0])
    elif torch.is_tensor(spin_flip):
        flip_values = [bool(x) for x in spin_flip.detach().cpu().tolist()]
    else:
        flip_values = [bool(x) for x in spin_flip]
    if not any(rot_values) and not any(flip_values):
        return m
    out = m.clone()
    for i, (k, flip) in enumerate(zip(rot_values, flip_values, strict=True)):
        item = m[i]
        if k:
            item = torch.rot90(item, k=int(k), dims=(-2, -1))
            rotated = item.clone()
            mx, my = item[0].clone(), item[1].clone()
            if k == 1:
                rotated[0], rotated[1] = -my, mx
            elif k == 2:
                rotated[0], rotated[1] = -mx, -my
            else:
                rotated[0], rotated[1] = my, -mx
            item = rotated
        if flip:
            item = -item
        out[i] = item
    return out


SEGMENT_HANDOFF_CACHE_SCHEMA = 3
SEGMENT_PHYSICAL_TARGET_CACHE_SCHEMA = 1


def _segment_handoff_prediction_counts(
    segment_cfg: dict[str, Any],
) -> tuple[int, int]:
    """Return ``(kept, generated)`` stage-1 handoff counts."""

    kept = int(segment_cfg.get("predicted_inits_per_segment", 1))
    generated = int(segment_cfg.get("candidate_inits_per_segment", kept))
    if kept < 1:
        raise ValueError("predicted_inits_per_segment must be >= 1")
    if generated < kept:
        raise ValueError(
            "candidate_inits_per_segment must be >= predicted_inits_per_segment"
        )
    return kept, generated


_DISTRIBUTION_DISTANCE_MODULES: dict[Path, Any] = {}


def _load_distribution_distance_module(selection_cfg: dict[str, Any]):
    configured_root = selection_cfg.get("distribution_score_root")
    root = (
        Path(configured_root).expanduser()
        if configured_root
        else Path(__file__).resolve().parents[1] / "third_party/distribution_score"
    )
    distance_path = (root / "src" / "distribution_score" / "distance.py").resolve()
    cached = _DISTRIBUTION_DISTANCE_MODULES.get(distance_path)
    if cached is not None:
        return cached
    if not distance_path.is_file():
        raise FileNotFoundError(
            "Distribution Score distance implementation was not found at "
            f"{distance_path}. Set candidate_selection.distribution_score_root "
            "to the Distribution_score-master checkout."
        )
    path_digest = hashlib.sha256(str(distance_path).encode()).hexdigest()[:12]
    module_name = f"_skyrmion_distribution_distance_{path_digest}"
    spec = importlib.util.spec_from_file_location(module_name, distance_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Distribution Score distance module: {distance_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "paired_patch_shift_distance"):
        raise ImportError(
            f"{distance_path} does not provide paired_patch_shift_distance"
        )
    _DISTRIBUTION_DISTANCE_MODULES[distance_path] = module
    return module


def _select_segment_handoff_candidates(
    candidates: torch.Tensor,
    truth: torch.Tensor,
    *,
    keep: int,
    device: torch.device,
    selection_cfg: dict[str, Any],
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Keep candidates nearest to each true segment boundary state."""

    if candidates.ndim != 5 or truth.ndim != 4:
        raise ValueError("handoff candidates/truth must use BRCHW/BCHW layouts")
    if candidates.shape[0] != truth.shape[0] or candidates.shape[2:] != truth.shape[1:]:
        raise ValueError(
            f"incompatible handoff candidate/truth shapes: {candidates.shape}, {truth.shape}"
        )
    if keep < 1 or keep > candidates.shape[1]:
        raise ValueError(f"invalid keep={keep} for {candidates.shape[1]} candidates")
    method = str(selection_cfg.get("method", "distribution_score")).lower().replace("-", "_")
    if method not in {"distribution_score", "distribution_score_patch_shift"}:
        raise ValueError(
            "candidate_selection.method must be 'distribution_score' "
            f"(got {method!r})"
        )

    distance_module = _load_distribution_distance_module(selection_cfg)
    blocks = int(selection_cfg.get("blocks", 8))
    shift_radius = int(selection_cfg.get("shift_radius", 32))
    score_batch = int(selection_cfg.get("score_candidate_batch_size", 4))
    reference_chunk = int(selection_cfg.get("reference_chunk", score_batch))
    if score_batch < 1 or reference_chunk < 1:
        raise ValueError(
            "candidate_selection score_candidate_batch_size and reference_chunk must be >= 1"
        )

    truth = truth.detach().to(device=device, dtype=torch.float32)
    geometry_mask = truth.square().sum(dim=1, keepdim=True).sqrt() > 0.5
    if not bool(geometry_mask.flatten(1).any(dim=1).all()):
        raise RuntimeError("cannot select handoffs against an empty geometry mask")
    truth = truth / truth.norm(dim=1, keepdim=True).clamp_min(1.0e-8)
    truth = truth * geometry_mask
    distance_mask = geometry_mask.to(dtype=torch.float32)

    distance_parts: list[np.ndarray] = []
    for start in range(0, candidates.shape[1], score_batch):
        end = min(candidates.shape[1], start + score_batch)
        candidate_part = candidates[:, start:end].to(
            device=device,
            dtype=torch.float32,
            non_blocking=device.type == "cuda",
        )
        candidate_part = candidate_part / candidate_part.norm(
            dim=2,
            keepdim=True,
        ).clamp_min(1.0e-8)
        candidate_part = candidate_part * geometry_mask[:, None]
        distance_part, _ = distance_module.paired_patch_shift_distance(
            candidate_part,
            truth,
            distance_mask,
            blocks=blocks,
            shift_radius=shift_radius,
            reference_chunk=min(reference_chunk, end - start),
        )
        distance_parts.append(np.asarray(distance_part, dtype=np.float32))
        del candidate_part

    distances = np.concatenate(distance_parts, axis=1)
    if distances.shape != candidates.shape[:2] or not np.isfinite(distances).all():
        raise RuntimeError(
            f"Distribution Score returned invalid candidate distances: {distances.shape}"
        )
    selected_indices = np.argsort(distances, axis=1, kind="stable")[:, :keep]
    batch_indices = torch.arange(candidates.shape[0])[:, None]
    selected = candidates[batch_indices, torch.from_numpy(selected_indices)]
    selected_distances = np.take_along_axis(distances, selected_indices, axis=1)
    return selected.contiguous(), selected_distances, selected_indices


class SegmentHandoffCache:
    """Read-only stage-1 predictions used as stage-2 segment initial states."""

    def __init__(self, path: str | Path) -> None:
        root = Path(path)
        index_path = root / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Segment handoff index not found: {index_path}")
        with index_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        schema_version = int(meta.get("schema_version", 0))
        if schema_version != SEGMENT_HANDOFF_CACHE_SCHEMA:
            raise ValueError(
                f"Unsupported segment handoff schema {schema_version}; "
                f"expected {SEGMENT_HANDOFF_CACHE_SCHEMA}"
            )
        self.root = root
        self.fingerprint = str(meta.get("fingerprint", ""))
        self.teacher = meta.get("teacher", {}) or {}
        self.shape = tuple(int(x) for x in meta["shape"])
        if len(self.shape) != 5:
            raise ValueError(
                "Segment handoff data must have shape (S, N, C, H, W), "
                f"got {self.shape}"
            )
        self.predictions_per_segment = int(meta["predictions_per_segment"])
        if self.predictions_per_segment < 1 or self.shape[1] != self.predictions_per_segment:
            raise ValueError("Segment handoff prediction count does not match its data shape")
        self.candidates_per_segment = int(
            meta.get("candidates_per_segment", self.predictions_per_segment)
        )
        if self.candidates_per_segment < self.predictions_per_segment:
            raise ValueError("Segment handoff candidate count is smaller than its kept count")
        self.candidate_selection = meta.get("candidate_selection", {}) or {}
        self.dtype = np.dtype(meta.get("dtype", "float16"))
        self.keys = {str(k): int(v) for k, v in meta["keys"].items()}
        if len(self.keys) != self.shape[0] or set(self.keys.values()) != set(range(self.shape[0])):
            raise ValueError("Segment handoff index is incomplete or contains duplicate rows")
        data_path = root / meta["data_file"]
        expected_bytes = int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize
        if not data_path.exists() or data_path.stat().st_size != expected_bytes:
            raise ValueError(f"Segment handoff data is missing or incomplete: {data_path}")
        self.data = np.memmap(data_path, mode="r", dtype=self.dtype, shape=self.shape)

    @staticmethod
    def key(run_id: str, frame_init: int, frame_target: int) -> str:
        return RectificationEndpointCache.key(run_id, frame_init, frame_target)

    def lookup(
        self,
        batch: dict[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        run_ids = batch.get("run_id")
        frame_init = batch.get("frame_init")
        frame_target = batch.get("frame_target")
        prediction_idx = batch.get("predicted_init_index")
        if run_ids is None or frame_init is None or frame_target is None or prediction_idx is None:
            raise KeyError(
                "Segment handoff lookup requires run_id, frame_init, frame_target, "
                "and predicted_init_index"
            )
        frame_init_values = (
            frame_init.detach().cpu().tolist()
            if torch.is_tensor(frame_init)
            else list(frame_init)
        )
        frame_target_values = (
            frame_target.detach().cpu().tolist()
            if torch.is_tensor(frame_target)
            else list(frame_target)
        )
        prediction_values = (
            prediction_idx.detach().cpu().tolist()
            if torch.is_tensor(prediction_idx)
            else list(prediction_idx)
        )
        arrays = []
        missing = []
        for run_id, fi, ft, pred_idx in zip(
            run_ids,
            frame_init_values,
            frame_target_values,
            prediction_values,
            strict=True,
        ):
            key = self.key(str(run_id), int(fi), int(ft))
            idx = self.keys.get(key)
            if idx is None:
                missing.append(key)
                continue
            pred_idx = int(pred_idx)
            if pred_idx < 0 or pred_idx >= self.predictions_per_segment:
                raise IndexError(
                    f"predicted_init_index={pred_idx} is outside "
                    f"[0, {self.predictions_per_segment})"
                )
            arrays.append(np.asarray(self.data[idx, pred_idx], dtype=np.float32))
        if missing:
            preview = ", ".join(missing[:3])
            raise KeyError(f"Segment handoff data is missing {len(missing)} samples: {preview}")
        out = torch.from_numpy(np.stack(arrays, axis=0)).to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        )
        return _apply_spin_augment_batch(out, batch)


class SegmentPhysicalTargetCache(SegmentHandoffCache):
    """MuMax3 endpoints continued from cached stage-1 handoff states."""

    def __init__(self, path: str | Path) -> None:
        root = Path(path)
        index_path = root / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Segment physical target index not found: {index_path}"
            )
        with index_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        schema_version = int(meta.get("schema_version", 0))
        if schema_version != SEGMENT_PHYSICAL_TARGET_CACHE_SCHEMA:
            raise ValueError(
                f"Unsupported segment physical target schema {schema_version}; "
                f"expected {SEGMENT_PHYSICAL_TARGET_CACHE_SCHEMA}"
            )
        self.root = root
        self.fingerprint = str(meta.get("fingerprint", ""))
        self.teacher = meta.get("teacher", {}) or {}
        self.source_handoff_fingerprint = str(
            meta.get("source_handoff_fingerprint", "")
        )
        self.shape = tuple(int(x) for x in meta["shape"])
        if len(self.shape) != 5:
            raise ValueError(
                "Segment physical target data must have shape (S, N, C, H, W), "
                f"got {self.shape}"
            )
        self.predictions_per_segment = int(meta["predictions_per_segment"])
        self.candidates_per_segment = self.predictions_per_segment
        self.candidate_selection = {"enabled": False, "method": "physical_replay"}
        if (
            self.predictions_per_segment < 1
            or self.shape[1] != self.predictions_per_segment
        ):
            raise ValueError(
                "Segment physical target prediction count does not match its data shape"
            )
        self.dtype = np.dtype(meta.get("dtype", "float16"))
        self.keys = {str(k): int(v) for k, v in meta["keys"].items()}
        if (
            len(self.keys) != self.shape[0]
            or set(self.keys.values()) != set(range(self.shape[0]))
        ):
            raise ValueError(
                "Segment physical target index is incomplete or contains duplicate rows"
            )
        data_path = root / str(meta["data_file"])
        expected_bytes = int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize
        if not data_path.exists() or data_path.stat().st_size != expected_bytes:
            raise ValueError(
                f"Segment physical target data is missing or incomplete: {data_path}"
            )
        self.data = np.memmap(
            data_path,
            mode="r",
            dtype=self.dtype,
            shape=self.shape,
        )


def prepare_segment_physical_target_cache(
    cfg: dict[str, Any],
    handoff_cache: SegmentHandoffCache | None,
) -> SegmentPhysicalTargetCache | None:
    segment_cfg = cfg.get("train", {}).get("segment_pushforward", {}) or {}
    target_mode = str(segment_cfg.get("target_mode", "original_gt")).lower()
    if target_mode == "original_gt":
        return None
    if target_mode != "mumax_replay":
        raise ValueError(
            "segment_pushforward target_mode must be original_gt|mumax_replay"
        )
    cache_path = segment_cfg.get("physical_target_cache_path")
    if not cache_path:
        raise ValueError(
            "mumax_replay requires segment_pushforward.physical_target_cache_path"
        )
    if handoff_cache is None:
        raise ValueError("mumax_replay requires a stage-1 handoff cache")
    cache = SegmentPhysicalTargetCache(cache_path)
    if cache.shape != handoff_cache.shape:
        raise ValueError(
            "Physical target and handoff cache shapes differ: "
            f"{cache.shape} != {handoff_cache.shape}"
        )
    if cache.keys != handoff_cache.keys:
        raise ValueError("Physical target and handoff cache keys differ")
    if (
        cache.source_handoff_fingerprint
        and handoff_cache.fingerprint
        and cache.source_handoff_fingerprint != handoff_cache.fingerprint
    ):
        raise ValueError(
            "Physical target cache was generated from a different handoff cache"
        )
    if is_main_process():
        print(
            {
                "stage2_physical_target_cache": str(Path(cache_path) / "index.json"),
                "shape": list(cache.shape),
                "source_handoff_fingerprint": cache.source_handoff_fingerprint,
            },
            flush=True,
        )
    return cache


def _segment_handoff_cache_path(cfg: dict[str, Any], out_dir: Path) -> Path:
    segment_cfg = cfg.get("train", {}).get("segment_pushforward", {}) or {}
    cache_path = segment_cfg.get("cache_path")
    return Path(cache_path) if cache_path else out_dir / "segment_handoff_cache"


def _iter_segment_handoff_samples(train_ds):
    indices = getattr(train_ds, "_segment_pair_indices", None)
    if indices is None:
        raise ValueError("Segment handoff cache requires a fixed-time segment_pair_mode dataset")
    rng = np.random.default_rng(int(getattr(train_ds, "seed", 0)) + 104_729)
    for rec_idx, segment_idx in indices:
        if int(segment_idx) <= 0:
            continue
        yield train_ds._build_segment_pair(rec_idx, segment_idx, rng, apply_augment=False)


def _count_segment_handoff_samples(train_ds) -> int:
    indices = getattr(train_ds, "_segment_pair_indices", None)
    if indices is None:
        raise ValueError("Segment handoff cache requires a fixed-time segment_pair_mode dataset")
    return sum(1 for _rec_idx, segment_idx in indices if int(segment_idx) > 0)


def _segment_handoff_fingerprint(
    cfg: dict[str, Any],
    train_ds,
    teacher_identity: dict[str, Any],
    *,
    inference_world_size: int = 1,
) -> str:
    indices = getattr(train_ds, "_segment_pair_indices", None)
    if indices is None:
        raise ValueError("Segment handoff data requires a fixed-time segment_pair_mode dataset")
    records = []
    for rec in getattr(train_ds, "records", ()):  # Dataset metadata, not frame contents.
        records.append(
            {
                "run_id": str(getattr(rec, "run_id", "")),
                "path": str(getattr(rec, "path", "")),
                "n_frames": int(getattr(rec, "n_frames", len(getattr(rec, "frames", ())))),
                "params": getattr(rec, "params", {}),
            }
        )
    segment_cfg = cfg.get("train", {}).get("segment_pushforward", {}) or {}
    predictions_per_segment, candidates_per_segment = _segment_handoff_prediction_counts(
        segment_cfg
    )
    payload = {
        "schema_version": SEGMENT_HANDOFF_CACHE_SCHEMA,
        "teacher": teacher_identity,
        "seed": int(cfg.get("seed", 0)),
        "data": cfg.get("data", {}),
        "model": cfg.get("model", {}),
        "bridge": cfg.get("bridge", {}),
        "prior": cfg.get("prior", {}),
        "sampler": cfg.get("sampler", {}),
        "roll_ode_steps": int(segment_cfg.get("roll_ode_steps", 0)),
        "predicted_inits_per_segment": predictions_per_segment,
        "candidate_inits_per_segment": candidates_per_segment,
        "candidate_selection": segment_cfg.get("candidate_selection", {}) or {},
        "inference_seed": int(segment_cfg.get("inference_seed", int(cfg.get("seed", 0)) + 7919)),
        "records": records,
        "segment_pair_indices": [[int(a), int(b)] for a, b in indices],
    }
    if inference_world_size > 1:
        payload["inference_world_size"] = int(inference_world_size)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _distributed_segment_bounds(total: int, rank: int, world_size: int) -> tuple[int, int]:
    if total < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(
            f"invalid distributed segment bounds: total={total}, rank={rank}, "
            f"world_size={world_size}"
        )
    return total * rank // world_size, total * (rank + 1) // world_size


def _valid_segment_handoff_data(
    cache_root: Path,
    *,
    fingerprint: str,
    num_samples: int,
    predictions_per_segment: int,
    candidates_per_segment: int,
) -> bool:
    index_path = cache_root / "index.json"
    if not index_path.exists():
        return False
    try:
        with index_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if int(meta.get("schema_version", 0)) != SEGMENT_HANDOFF_CACHE_SCHEMA:
            return False
        if str(meta.get("fingerprint", "")) != fingerprint:
            return False
        if int(meta.get("predictions_per_segment", 0)) != predictions_per_segment:
            return False
        if int(meta.get("candidates_per_segment", 0)) != candidates_per_segment:
            return False
        shape = tuple(int(x) for x in meta["shape"])
        if len(shape) != 5 or shape[0] != num_samples or shape[1] != predictions_per_segment:
            return False
        keys = {str(k): int(v) for k, v in meta["keys"].items()}
        if len(keys) != num_samples or set(keys.values()) != set(range(num_samples)):
            return False
        dtype = np.dtype(meta.get("dtype", "float16"))
        data_path = cache_root / str(meta["data_file"])
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        return data_path.exists() and data_path.stat().st_size == expected_bytes
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def _prepare_segment_handoff_cache_distributed(
    cfg: dict[str, Any],
    train_ds,
    model: torch.nn.Module,
    sampler: BridgeSampler,
    device: torch.device,
    cache_root: Path,
    *,
    teacher_identity: dict[str, Any] | None,
    predictions_per_segment: int,
    candidates_per_segment: int,
    selection_cfg: dict[str, Any],
    num_samples: int,
) -> SegmentHandoffCache:
    """Build disjoint handoff-cache shards concurrently on every DDP rank."""

    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("distributed handoff precomputation requires initialized DDP")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    main = rank == 0
    segment_cfg = cfg.get("train", {}).get("segment_pushforward", {}) or {}
    selection_enabled = candidates_per_segment > predictions_per_segment
    index_path = cache_root / "index.json"

    state: list[Any] = [None, False, False, None]
    if main:
        if teacher_identity is None:
            raise ValueError("Stage-2 precomputation requires a loaded stage-1 teacher")
        fingerprint = _segment_handoff_fingerprint(
            cfg,
            train_ds,
            teacher_identity,
            inference_world_size=world_size,
        )
        rebuild_on_start = bool(segment_cfg.get("rebuild_on_stage2_start", True))
        force = bool(segment_cfg.get("cache_rebuild", False)) or (
            rebuild_on_start and bool(cfg.get("train", {}).get("resume_reset_step", False))
        )
        ready = _valid_segment_handoff_data(
            cache_root,
            fingerprint=fingerprint,
            num_samples=num_samples,
            predictions_per_segment=predictions_per_segment,
            candidates_per_segment=candidates_per_segment,
        )
        run_id = os.environ.get("RUN_ID")
        build_token = f"{run_id or 'local'}-{os.getpid()}"
        state = [fingerprint, force, ready, build_token]
    dist.broadcast_object_list(state, src=0)
    fingerprint = str(state[0])
    force = bool(state[1])
    ready = bool(state[2])
    build_token = str(state[3])

    if not (force or not ready):
        if main:
            print(
                {
                    "stage2_inference_data_reused": str(index_path),
                    "segments": num_samples,
                    "predicted_inits_per_segment": predictions_per_segment,
                    "candidate_inits_per_segment": candidates_per_segment,
                    "inference_world_size": world_size,
                },
                flush=True,
            )
        dist.barrier()
        cache = SegmentHandoffCache(cache_root)
        if cache.predictions_per_segment != predictions_per_segment:
            raise ValueError(
                "Stage-2 inference data prediction count does not match the active "
                "configuration"
            )
        return cache

    first = next(_iter_segment_handoff_samples(train_ds))["m_t"]
    shape = (num_samples, predictions_per_segment, *tuple(first.shape))
    data_file = f"handoff.{fingerprint[:16]}.float16.dat"
    data_path = cache_root / data_file
    tmp_data_path = cache_root / f".{data_file}.{build_token}.tmp"
    tmp_index_path = cache_root / f".index.{build_token}.tmp"
    rank_meta_paths = [
        cache_root / f".{data_file}.{build_token}.rank{worker_rank}.json"
        for worker_rank in range(world_size)
    ]
    previous_data_file = None
    if main:
        cache_root.mkdir(parents=True, exist_ok=True)
        if index_path.exists():
            try:
                with index_path.open("r", encoding="utf-8") as f:
                    previous_data_file = json.load(f).get("data_file")
            except (OSError, AttributeError, json.JSONDecodeError):
                previous_data_file = None
        initialized = np.memmap(
            tmp_data_path,
            mode="w+",
            dtype=np.float16,
            shape=shape,
        )
        initialized.flush()
        del initialized
    dist.barrier()

    local_start, local_end = _distributed_segment_bounds(num_samples, rank, world_size)
    local_samples = local_end - local_start
    handoff = np.memmap(tmp_data_path, mode="r+", dtype=np.float16, shape=shape)
    keys: dict[str, int] = {}
    batch_size = int(
        segment_cfg.get("cache_batch_size", cfg["train"].get("batch_size", 16))
    )
    roll_ode_steps = int(segment_cfg.get("roll_ode_steps", 0)) or sampler.ode_steps
    old_training = model.training
    model.eval()
    full_ode_steps = sampler.ode_steps
    sampler.ode_steps = roll_ode_steps
    amp_dtype = cfg["train"].get("amp_dtype", "bf16")
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    use_amp = device.type == "cuda"
    inference_seed = int(
        segment_cfg.get("inference_seed", int(cfg.get("seed", 0)) + 7919)
    )
    rank_inference_seed = inference_seed + rank * 1_000_003
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    torch.manual_seed(rank_inference_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(rank_inference_seed)

    if main:
        print(
            {
                "stage2_inference_data": str(cache_root),
                "segments": num_samples,
                "predicted_inits_per_segment": predictions_per_segment,
                "candidate_inits_per_segment": candidates_per_segment,
                "candidate_selection": (
                    str(selection_cfg.get("method", "distribution_score"))
                    if selection_enabled
                    else "none"
                ),
                "shape": list(shape),
                "roll_ode_steps": roll_ode_steps,
                "teacher": teacher_identity,
                "rebuild": bool(force),
                "inference_world_size": world_size,
                "rank_segment_ranges": [
                    list(_distributed_segment_bounds(num_samples, worker_rank, world_size))
                    for worker_rank in range(world_size)
                ],
            },
            flush=True,
        )
    print(
        {
            "stage2_inference_rank": rank,
            "device": str(device),
            "segment_range": [local_start, local_end],
            "segments": local_samples,
            "seed": rank_inference_seed,
        },
        flush=True,
    )

    chunk: list[dict[str, Any]] = []
    write_offset = local_start
    selection_audit = {
        "count": 0,
        "sum": 0.0,
        "min": math.inf,
        "max": -math.inf,
        "candidate_index_histogram": np.zeros(candidates_per_segment, dtype=np.int64),
    }
    pbar = tqdm(
        total=local_samples * candidates_per_segment,
        desc=f"stage2 inference rank {rank}/{world_size}",
        disable=not main,
    )

    def infer_chunk(samples: list[dict[str, Any]], offset: int) -> None:
        batch = move_batch(default_collate(samples), device)
        prev = _prefixed_batch(batch, "prev_")
        prev_cond = collate_fixed_time_conditions(prev)
        candidate_fields = torch.empty(
            (
                len(samples),
                candidates_per_segment,
                *tuple(prev["m_init"].shape[1:]),
            ),
            dtype=torch.float16,
            device="cpu",
        )
        for prediction_idx in range(candidates_per_segment):
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=use_amp,
            ):
                pred_init, _ = sampler.sample(model, prev["m_init"], prev_cond)
            candidate_fields[:, prediction_idx].copy_(
                pred_init.detach().to(device="cpu", dtype=torch.float16)
            )
            pbar.update(len(samples))
        if selection_enabled:
            selected, selected_distances, selected_indices = (
                _select_segment_handoff_candidates(
                    candidate_fields,
                    batch["m_init"],
                    keep=predictions_per_segment,
                    device=device,
                    selection_cfg=selection_cfg,
                )
            )
            selection_audit["count"] += int(selected_distances.size)
            selection_audit["sum"] += float(selected_distances.sum(dtype=np.float64))
            selection_audit["min"] = min(
                float(selection_audit["min"]),
                float(selected_distances.min()),
            )
            selection_audit["max"] = max(
                float(selection_audit["max"]),
                float(selected_distances.max()),
            )
            np.add.at(
                selection_audit["candidate_index_histogram"],
                selected_indices.reshape(-1),
                1,
            )
        else:
            selected = candidate_fields
        handoff[offset : offset + len(samples)] = selected.numpy()
        for local_idx, sample_i in enumerate(samples):
            key = SegmentHandoffCache.key(
                str(sample_i["run_id"]),
                int(sample_i["frame_init"]),
                int(sample_i["frame_target"]),
            )
            if key in keys:
                raise ValueError(f"Duplicate segment handoff key on rank {rank}: {key}")
            keys[key] = offset + local_idx

    try:
        for sample_idx, sample in enumerate(_iter_segment_handoff_samples(train_ds)):
            if sample_idx < local_start:
                continue
            if sample_idx >= local_end:
                break
            chunk.append(sample)
            if len(chunk) < batch_size:
                continue
            infer_chunk(chunk, write_offset)
            write_offset += len(chunk)
            chunk = []
        if chunk:
            infer_chunk(chunk, write_offset)
            write_offset += len(chunk)
        if write_offset != local_end or len(keys) != local_samples:
            raise RuntimeError(
                f"Stage-2 rank {rank} wrote {write_offset - local_start}/"
                f"{local_samples} assigned segments"
            )
        handoff.flush()
        del handoff
        audit_count = int(selection_audit["count"])
        rank_meta = {
            "rank": rank,
            "start": local_start,
            "end": local_end,
            "keys": keys,
            "selection_audit": {
                "count": audit_count,
                "sum": float(selection_audit["sum"]),
                "min": (
                    float(selection_audit["min"]) if audit_count > 0 else None
                ),
                "max": (
                    float(selection_audit["max"]) if audit_count > 0 else None
                ),
                "candidate_index_histogram": selection_audit[
                    "candidate_index_histogram"
                ].tolist(),
            },
        }
        with rank_meta_paths[rank].open("w", encoding="utf-8") as f:
            json.dump(rank_meta, f)
            f.flush()
            os.fsync(f.fileno())
    finally:
        if "handoff" in locals():
            try:
                del handoff
            except UnboundLocalError:
                pass
        pbar.close()
        sampler.ode_steps = full_ode_steps
        model.train(old_training)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, device)

    # Every rank closes and flushes its disjoint memmap range before rank 0
    # atomically publishes the shared data file and combined index.
    dist.barrier()
    if main:
        merged_keys: dict[str, int] = {}
        merged_audit = {
            "count": 0,
            "sum": 0.0,
            "min": math.inf,
            "max": -math.inf,
            "candidate_index_histogram": np.zeros(
                candidates_per_segment,
                dtype=np.int64,
            ),
        }
        expected_start = 0
        for worker_rank, meta_path in enumerate(rank_meta_paths):
            with meta_path.open("r", encoding="utf-8") as f:
                rank_meta = json.load(f)
            start = int(rank_meta["start"])
            end = int(rank_meta["end"])
            if int(rank_meta["rank"]) != worker_rank or start != expected_start:
                raise RuntimeError(
                    f"invalid Stage-2 rank metadata for rank {worker_rank}: "
                    f"range=({start}, {end}), expected_start={expected_start}"
                )
            expected_start = end
            for key, value in rank_meta["keys"].items():
                if key in merged_keys:
                    raise ValueError(f"Duplicate segment handoff key across ranks: {key}")
                merged_keys[str(key)] = int(value)
            audit = rank_meta["selection_audit"]
            count = int(audit["count"])
            merged_audit["count"] += count
            merged_audit["sum"] += float(audit["sum"])
            if count > 0:
                merged_audit["min"] = min(
                    float(merged_audit["min"]),
                    float(audit["min"]),
                )
                merged_audit["max"] = max(
                    float(merged_audit["max"]),
                    float(audit["max"]),
                )
            merged_audit["candidate_index_histogram"] += np.asarray(
                audit["candidate_index_histogram"],
                dtype=np.int64,
            )
        if expected_start != num_samples or len(merged_keys) != num_samples:
            raise RuntimeError(
                f"Distributed Stage-2 cache is incomplete: ranges end at "
                f"{expected_start}/{num_samples}, keys={len(merged_keys)}"
            )
        if set(merged_keys.values()) != set(range(num_samples)):
            raise RuntimeError("Distributed Stage-2 cache offsets are not contiguous")

        os.replace(tmp_data_path, data_path)
        if selection_enabled:
            selected_count = int(merged_audit["count"])
            selection_metadata = {
                "enabled": True,
                "method": str(selection_cfg.get("method", "distribution_score")),
                "target": "current_segment_true_m_init",
                "distance": {
                    "resolution": [256, 256],
                    "magnetization_components": 3,
                    "blocks": [
                        int(selection_cfg.get("blocks", 8)),
                        int(selection_cfg.get("blocks", 8)),
                    ],
                    "shift_radius_each_axis_px": int(
                        selection_cfg.get("shift_radius", 32)
                    ),
                    "integer_shifts": True,
                    "downsampling": False,
                    "shift_penalty": 0.0,
                    "symmetric": True,
                    "lower_is_better": True,
                },
                "selected_distance": {
                    "count": selected_count,
                    "mean": float(merged_audit["sum"]) / max(selected_count, 1),
                    "min": float(merged_audit["min"]),
                    "max": float(merged_audit["max"]),
                },
                "selected_candidate_index_histogram": merged_audit[
                    "candidate_index_histogram"
                ].tolist(),
            }
        else:
            selection_metadata = {"enabled": False, "method": "none"}
        with tmp_index_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": SEGMENT_HANDOFF_CACHE_SCHEMA,
                    "fingerprint": fingerprint,
                    "teacher": teacher_identity,
                    "data_file": data_file,
                    "dtype": "float16",
                    "shape": list(shape),
                    "keys": merged_keys,
                    "roll_ode_steps": roll_ode_steps,
                    "predictions_per_segment": predictions_per_segment,
                    "candidates_per_segment": candidates_per_segment,
                    "candidate_selection": selection_metadata,
                    "inference_seed": inference_seed,
                    "inference_world_size": world_size,
                    "rank_seed_stride": 1_000_003,
                },
                f,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_index_path, index_path)
        if previous_data_file and previous_data_file != data_file:
            previous_data_path = cache_root / str(previous_data_file)
            if (
                previous_data_path.parent.resolve() == cache_root.resolve()
                and previous_data_path.name.startswith("handoff")
                and previous_data_path.exists()
            ):
                previous_data_path.unlink()
        for meta_path in rank_meta_paths:
            meta_path.unlink(missing_ok=True)
        print(
            {
                "stage2_inference_data_complete": str(index_path),
                "segments": num_samples,
                "predicted_inits_per_segment": predictions_per_segment,
                "candidate_inits_per_segment": candidates_per_segment,
                "inference_world_size": world_size,
            },
            flush=True,
        )

    dist.barrier()
    cache = SegmentHandoffCache(cache_root)
    if cache.predictions_per_segment != predictions_per_segment:
        raise ValueError(
            "Stage-2 inference data prediction count does not match the active "
            "configuration"
        )
    return cache


def prepare_segment_handoff_cache(
    cfg: dict[str, Any],
    train_ds,
    model: torch.nn.Module,
    sampler: BridgeSampler,
    device: torch.device,
    out_dir: Path,
    *,
    distributed: bool,
    teacher_identity: dict[str, Any] | None = None,
) -> SegmentHandoffCache | None:
    segment_cfg = cfg.get("train", {}).get("segment_pushforward", {}) or {}
    if (
        not bool(segment_cfg.get("enabled", False))
        or float(segment_cfg.get("pred_weight", 1.0)) <= 0.0
    ):
        return None
    cache_mode = str(segment_cfg.get("cache_mode", "precompute")).lower()
    if cache_mode not in {"precompute", "read_only"}:
        raise ValueError(
            "train.segment_pushforward requires cache_mode: precompute|read_only; "
            "online inference during stage-2 training is not supported"
        )
    predictions_per_segment, candidates_per_segment = _segment_handoff_prediction_counts(
        segment_cfg
    )
    selection_enabled = candidates_per_segment > predictions_per_segment
    selection_cfg = segment_cfg.get("candidate_selection", {}) or {}
    cache_root = _segment_handoff_cache_path(cfg, out_dir)
    index_path = cache_root / "index.json"
    num_samples = _count_segment_handoff_samples(train_ds)
    if num_samples <= 0:
        raise ValueError("Segment pushforward training has no non-first segments to precompute")
    if cache_mode == "read_only":
        cache = SegmentHandoffCache(cache_root)
        if cache.shape[0] != num_samples:
            raise ValueError(
                "Read-only handoff cache segment count does not match the training split: "
                f"{cache.shape[0]} != {num_samples}"
            )
        if cache.predictions_per_segment != predictions_per_segment:
            raise ValueError(
                "Read-only handoff cache prediction count does not match the active "
                "configuration"
            )
        expected_sha = str((teacher_identity or {}).get("sha256", ""))
        cached_sha = str(cache.teacher.get("sha256", ""))
        if expected_sha and cached_sha and expected_sha != cached_sha:
            raise ValueError(
                "Read-only handoff cache teacher SHA does not match the loaded teacher"
            )
        if is_main_process():
            print(
                {
                    "stage2_inference_data_reused_read_only": str(index_path),
                    "segments": num_samples,
                    "predicted_inits_per_segment": predictions_per_segment,
                },
                flush=True,
            )
        return cache
    if distributed:
        return _prepare_segment_handoff_cache_distributed(
            cfg,
            train_ds,
            model,
            sampler,
            device,
            cache_root,
            teacher_identity=teacher_identity,
            predictions_per_segment=predictions_per_segment,
            candidates_per_segment=candidates_per_segment,
            selection_cfg=selection_cfg,
            num_samples=num_samples,
        )
    if is_main_process():
        if teacher_identity is None:
            raise ValueError("Stage-2 precomputation requires a loaded stage-1 teacher")
        fingerprint = _segment_handoff_fingerprint(cfg, train_ds, teacher_identity)
        rebuild_on_start = bool(segment_cfg.get("rebuild_on_stage2_start", True))
        force = bool(segment_cfg.get("cache_rebuild", False)) or (
            rebuild_on_start and bool(cfg.get("train", {}).get("resume_reset_step", False))
        )
        ready = _valid_segment_handoff_data(
            cache_root,
            fingerprint=fingerprint,
            num_samples=num_samples,
            predictions_per_segment=predictions_per_segment,
            candidates_per_segment=candidates_per_segment,
        )
    else:
        fingerprint = ""
        force = False
        ready = False
    if is_main_process() and (force or not ready):
        cache_root.mkdir(parents=True, exist_ok=True)
        previous_data_file = None
        if index_path.exists():
            try:
                with index_path.open("r", encoding="utf-8") as f:
                    previous_data_file = json.load(f).get("data_file")
            except (OSError, AttributeError, json.JSONDecodeError):
                previous_data_file = None
        first = next(_iter_segment_handoff_samples(train_ds))["m_t"]
        shape = (num_samples, predictions_per_segment, *tuple(first.shape))
        data_file = f"handoff.{fingerprint[:16]}.float16.dat"
        data_path = cache_root / data_file
        tmp_data_path = cache_root / f".{data_file}.{os.getpid()}.tmp"
        tmp_index_path = cache_root / f".index.{os.getpid()}.tmp"
        handoff = np.memmap(tmp_data_path, mode="w+", dtype=np.float16, shape=shape)
        keys: dict[str, int] = {}
        batch_size = int(segment_cfg.get("cache_batch_size", cfg["train"].get("batch_size", 16)))
        roll_ode_steps = int(segment_cfg.get("roll_ode_steps", 0)) or sampler.ode_steps
        old_training = model.training
        model.eval()
        full_ode_steps = sampler.ode_steps
        sampler.ode_steps = roll_ode_steps
        amp_dtype = cfg["train"].get("amp_dtype", "bf16")
        dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        use_amp = device.type == "cuda"
        inference_seed = int(segment_cfg.get("inference_seed", int(cfg.get("seed", 0)) + 7919))
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        torch.manual_seed(inference_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(inference_seed)
        print(
            {
                "stage2_inference_data": str(cache_root),
                "segments": num_samples,
                "predicted_inits_per_segment": predictions_per_segment,
                "candidate_inits_per_segment": candidates_per_segment,
                "candidate_selection": (
                    str(selection_cfg.get("method", "distribution_score"))
                    if selection_enabled
                    else "none"
                ),
                "shape": list(shape),
                "roll_ode_steps": roll_ode_steps,
                "teacher": teacher_identity,
                "rebuild": bool(force),
            },
            flush=True,
        )
        chunk: list[dict[str, Any]] = []
        start = 0
        selection_audit = {
            "count": 0,
            "sum": 0.0,
            "min": math.inf,
            "max": -math.inf,
            "candidate_index_histogram": np.zeros(
                candidates_per_segment,
                dtype=np.int64,
            ),
        }
        pbar = tqdm(
            total=num_samples * candidates_per_segment,
            desc="stage2 inference candidates",
        )

        def infer_chunk(samples: list[dict[str, Any]], offset: int) -> None:
            batch = move_batch(default_collate(samples), device)
            prev = _prefixed_batch(batch, "prev_")
            prev_cond = collate_fixed_time_conditions(prev)
            candidate_fields = torch.empty(
                (
                    len(samples),
                    candidates_per_segment,
                    *tuple(prev["m_init"].shape[1:]),
                ),
                dtype=torch.float16,
                device="cpu",
            )
            for prediction_idx in range(candidates_per_segment):
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=use_amp,
                ):
                    pred_init, _ = sampler.sample(model, prev["m_init"], prev_cond)
                candidate_fields[:, prediction_idx].copy_(
                    pred_init.detach().to(device="cpu", dtype=torch.float16)
                )
                pbar.update(len(samples))
            if selection_enabled:
                selected, selected_distances, selected_indices = (
                    _select_segment_handoff_candidates(
                        candidate_fields,
                        batch["m_init"],
                        keep=predictions_per_segment,
                        device=device,
                        selection_cfg=selection_cfg,
                    )
                )
                selection_audit["count"] += int(selected_distances.size)
                selection_audit["sum"] += float(selected_distances.sum(dtype=np.float64))
                selection_audit["min"] = min(
                    float(selection_audit["min"]),
                    float(selected_distances.min()),
                )
                selection_audit["max"] = max(
                    float(selection_audit["max"]),
                    float(selected_distances.max()),
                )
                np.add.at(
                    selection_audit["candidate_index_histogram"],
                    selected_indices.reshape(-1),
                    1,
                )
            else:
                selected = candidate_fields
            handoff[offset : offset + len(samples)] = selected.numpy()
            for local_idx, sample_i in enumerate(samples):
                key = SegmentHandoffCache.key(
                    str(sample_i["run_id"]),
                    int(sample_i["frame_init"]),
                    int(sample_i["frame_target"]),
                )
                if key in keys:
                    raise ValueError(f"Duplicate segment handoff key: {key}")
                keys[key] = offset + local_idx

        try:
            for sample in _iter_segment_handoff_samples(train_ds):
                chunk.append(sample)
                if len(chunk) < batch_size:
                    continue
                infer_chunk(chunk, start)
                start += len(chunk)
                chunk = []
            if chunk:
                infer_chunk(chunk, start)
                start += len(chunk)
            if start != num_samples or len(keys) != num_samples:
                raise RuntimeError(
                    f"Stage-2 inference data is incomplete: wrote {start}/{num_samples} segments"
                )
            handoff.flush()
            del handoff
            os.replace(tmp_data_path, data_path)
            if selection_enabled:
                selected_count = int(selection_audit["count"])
                selection_metadata = {
                    "enabled": True,
                    "method": str(selection_cfg.get("method", "distribution_score")),
                    "target": "current_segment_true_m_init",
                    "distance": {
                        "resolution": [256, 256],
                        "magnetization_components": 3,
                        "blocks": [
                            int(selection_cfg.get("blocks", 8)),
                            int(selection_cfg.get("blocks", 8)),
                        ],
                        "shift_radius_each_axis_px": int(
                            selection_cfg.get("shift_radius", 32)
                        ),
                        "integer_shifts": True,
                        "downsampling": False,
                        "shift_penalty": 0.0,
                        "symmetric": True,
                        "lower_is_better": True,
                    },
                    "selected_distance": {
                        "count": selected_count,
                        "mean": float(selection_audit["sum"]) / max(selected_count, 1),
                        "min": float(selection_audit["min"]),
                        "max": float(selection_audit["max"]),
                    },
                    "selected_candidate_index_histogram": selection_audit[
                        "candidate_index_histogram"
                    ].tolist(),
                }
            else:
                selection_metadata = {"enabled": False, "method": "none"}
            with tmp_index_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "schema_version": SEGMENT_HANDOFF_CACHE_SCHEMA,
                        "fingerprint": fingerprint,
                        "teacher": teacher_identity,
                        "data_file": data_file,
                        "dtype": "float16",
                        "shape": list(shape),
                        "keys": keys,
                        "roll_ode_steps": roll_ode_steps,
                        "predictions_per_segment": predictions_per_segment,
                        "candidates_per_segment": candidates_per_segment,
                        "candidate_selection": selection_metadata,
                        "inference_seed": inference_seed,
                    },
                    f,
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_index_path, index_path)
            if previous_data_file and previous_data_file != data_file:
                previous_data_path = cache_root / str(previous_data_file)
                if (
                    previous_data_path.parent.resolve() == cache_root.resolve()
                    and previous_data_path.name.startswith("handoff")
                    and previous_data_path.exists()
                ):
                    previous_data_path.unlink()
        except Exception:
            if "handoff" in locals():
                try:
                    del handoff
                except UnboundLocalError:
                    pass
            for tmp_path in (tmp_data_path, tmp_index_path):
                if tmp_path.exists():
                    tmp_path.unlink()
            raise
        finally:
            pbar.close()
            sampler.ode_steps = full_ode_steps
            model.train(old_training)
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, device)
        print(
            {
                "stage2_inference_data_complete": str(index_path),
                "segments": num_samples,
                "predicted_inits_per_segment": predictions_per_segment,
                "candidate_inits_per_segment": candidates_per_segment,
            },
            flush=True,
        )
    elif is_main_process():
        print(
            {
                "stage2_inference_data_reused": str(index_path),
                "segments": num_samples,
                "predicted_inits_per_segment": predictions_per_segment,
                "candidate_inits_per_segment": candidates_per_segment,
            },
            flush=True,
        )
    if distributed:
        dist.barrier()
    cache = SegmentHandoffCache(cache_root)
    if cache.predictions_per_segment != predictions_per_segment:
        raise ValueError(
            "Stage-2 inference data prediction count does not match the active configuration"
        )
    return cache


def build_rectification_cache(rect_cfg: dict[str, Any]) -> RectificationEndpointCache | None:
    if not bool(rect_cfg.get("enabled", False)):
        return None
    cache_path = rect_cfg.get("cache_path")
    if not cache_path:
        return None
    return RectificationEndpointCache(
        cache_path,
        strict=bool(rect_cfg.get("cache_strict", True)),
    )


def _normalize_spin_field(m: torch.Tensor) -> torch.Tensor:
    return m / m.norm(dim=1, keepdim=True).clamp_min(1e-8)


def _mix_teacher_true_endpoint(
    m_true: torch.Tensor,
    m_teacher: torch.Tensor,
    mix: float,
    mode: str = "slerp",
) -> torch.Tensor:
    mix = float(mix)
    if mix <= 0.0:
        return m_true
    if mix >= 1.0:
        return _normalize_spin_field(m_teacher)
    mode = str(mode).lower()
    if mode == "slerp":
        tau = m_true.new_full((m_true.shape[0],), mix)
        return slerp_chw(m_true, _normalize_spin_field(m_teacher), tau)
    if mode in {"linear", "lerp"}:
        return _normalize_spin_field((1.0 - mix) * m_true + mix * m_teacher)
    raise ValueError("train.rectification.mix_mode must be slerp or linear")


@torch.no_grad()
def apply_teacher_true_rectification(
    batch: dict[str, Any],
    cond: dict[str, torch.Tensor],
    teacher_bundle: tuple[torch.nn.Module, BridgeSampler] | None,
    rect_cfg: dict[str, Any],
    *,
    needs_omega: bool,
    endpoint_cache: RectificationEndpointCache | None = None,
) -> dict[str, Any]:
    if teacher_bundle is None and endpoint_cache is None:
        return batch
    apply_prob = float(rect_cfg.get("apply_prob", 1.0))
    if apply_prob <= 0.0:
        return batch
    m_init = batch.get("m_init", batch.get("m0"))
    m_true = batch.get("m_t", batch.get("m1"))
    if m_init is None or m_true is None:
        return batch
    if endpoint_cache is not None:
        m_teacher = endpoint_cache.lookup(batch, device=m_init.device, dtype=m_init.dtype)
        if m_teacher is None:
            return batch
    else:
        if teacher_bundle is None:
            return batch
        teacher, teacher_sampler = teacher_bundle
        m_teacher, _ = teacher_sampler.sample(teacher, m_init, cond)
    mixed = _mix_teacher_true_endpoint(
        m_true,
        m_teacher,
        float(rect_cfg.get("endpoint_mix", 0.3)),
        str(rect_cfg.get("mix_mode", "slerp")),
    )
    if apply_prob < 1.0:
        mask = (torch.rand(mixed.shape[0], device=mixed.device) < apply_prob).reshape(
            -1, *([1] * (mixed.ndim - 1))
        )
        mixed = torch.where(mask, mixed, m_true)
    out = dict(batch)
    if "m_t" in out:
        out["m_t"] = mixed
    elif "m1" in out:
        out["m1"] = mixed
    else:
        out["m_t"] = mixed
    if needs_omega:
        out["omega_target"] = log_map_chw(m_init, mixed)
    return out


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: CFMLoss,
    sampler,
    device: torch.device,
    max_batches: int = 20,
    collate_fn=None,
    bucket_key: str = "dt_scale",
    visual_examples: int = 0,
    visual_out_dir: Path | None = None,
    visual_step: int = 0,
    visual_channel: str = "mz",
    visual_dpi: int = 180,
    visual_bucket_indices: set[int] | None = None,
    visual_filename_prefix: str = "",
    visual_mode: str = "informative",
    visual_diff_vmax: float = 1.0,
    visual_zoom: bool = True,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, list[float]] = {
        "loss": [],
        "cfm": [],
        "unit": [],
        "llg": [],
        "topo": [],
        "mse": [],
        "ang": [],
        "q": [],
        "topo_q_abs": [],
        "magnetic_fraction": [],
        "energy_abs": [],
        "energy_delta": [],
        "energy_pred": [],
        "energy_target": [],
    }
    by_scale: dict[int, dict[str, list[float]]] = {}
    examples: list[dict[str, Any]] = []
    informative_groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    collate = collate_fn or collate_conditions
    normalized_visual_mode = str(visual_mode).strip().lower().replace("-", "_")
    informative_visual = visual_examples > 0 and normalized_visual_mode in {
        "informative",
        "compact",
        "informative_pairs",
    }
    trajectory_visual = (
        visual_examples > 0
        and normalized_visual_mode in {
            "trajectory",
            "legacy_trajectory",
            "rollout",
            "trajectory_rollout",
            "segment_start",
            "segment_start_independent",
            "independent_segments",
            "segment_independent",
        }
        and hasattr(loader.dataset, "visual_rows")
    )
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move_batch(batch, device)
        cond = collate(batch)
        loss = loss_fn(model, batch, cond)
        m_init = batch.get("m_init", batch.get("m0"))
        m_target = batch.get("m_t", batch.get("m1"))
        pred_m1, _ = sampler.sample(model, m_init, cond)
        metric_spatial_mask = batch.get("defect_field")
        mse_each = mse_m(pred_m1, m_target, mask=metric_spatial_mask)
        ang_each = angular_error_deg(pred_m1, m_target, mask=metric_spatial_mask)
        magnetic_fraction_each = magnetic_fraction(metric_spatial_mask, pred_m1)
        q_pred_each = topological_charge(pred_m1, boundary=loss_fn.boundary)
        q_target_each = topological_charge(m_target, boundary=loss_fn.boundary)
        q_each = (q_pred_each - q_target_each).abs()
        b_t = cond["b_t"]
        energy_pred_each = energy_density(
            pred_m1,
            b_t,
            boundary=loss_fn.boundary,
            mask=metric_spatial_mask,
        )
        energy_target_each = energy_density(
            m_target,
            b_t,
            boundary=loss_fn.boundary,
            mask=metric_spatial_mask,
        )
        energy_delta_each = energy_pred_each - energy_target_each
        energy_abs_each = energy_delta_each.abs()

        if informative_visual:
            add_informative_candidates(
                informative_groups,
                m_init=m_init,
                pred=pred_m1,
                target=m_target,
                batch=batch,
                mse_each=mse_each,
                ang_each=ang_each,
                q_pred_each=q_pred_each,
                q_target_each=q_target_each,
            )

        endpoint_mask = None
        if isinstance(loss_fn.bridge, (RotationVectorBridge, RotationVector2DBridge)):
            t_end_ns = cond.get("t_end_ns", batch.get("t_end_ns"))
            if torch.is_tensor(t_end_ns):
                endpoint_mask = t_end_ns <= loss_fn.alpha_t_end_ns_max
        if endpoint_mask is not None:
            metric_mask = endpoint_mask.detach()
            has_endpoint_metrics = bool(metric_mask.any().item())
            mse_metric = mse_each[metric_mask]
            ang_metric = ang_each[metric_mask]
            q_metric = q_each[metric_mask]
            magnetic_fraction_metric = magnetic_fraction_each[metric_mask]
            energy_abs_metric = energy_abs_each[metric_mask]
            energy_delta_metric = energy_delta_each[metric_mask]
            energy_pred_metric = energy_pred_each[metric_mask]
            energy_target_metric = energy_target_each[metric_mask]
            visual_pred = pred_m1[metric_mask] if has_endpoint_metrics else pred_m1[:0]
            visual_target = m_target[metric_mask] if has_endpoint_metrics else m_target[:0]
            visual_batch = mask_batch(batch, metric_mask) if has_endpoint_metrics else batch
        else:
            metric_mask = None
            has_endpoint_metrics = True
            mse_metric = mse_each
            ang_metric = ang_each
            q_metric = q_each
            magnetic_fraction_metric = magnetic_fraction_each
            energy_abs_metric = energy_abs_each
            energy_delta_metric = energy_delta_each
            energy_pred_metric = energy_pred_each
            energy_target_metric = energy_target_each
            visual_pred = pred_m1
            visual_target = m_target
            visual_batch = batch

        if visual_examples > 0 and not informative_visual and not trajectory_visual:
            _collect_validation_examples(
                examples,
                visual_pred,
                visual_target,
                visual_batch,
                visual_examples,
                target_buckets=visual_bucket_indices,
            )
        totals["loss"].append(float(loss.total.detach().cpu()))
        totals["cfm"].append(float(loss.cfm.detach().cpu()))
        totals["unit"].append(float(loss.unit.detach().cpu()))
        totals["llg"].append(float(loss.llg.detach().cpu()))
        totals["topo"].append(float(loss.topo.detach().cpu()))
        if has_endpoint_metrics:
            totals["mse"].append(float(mse_metric.mean().cpu()))
            totals["ang"].append(float(ang_metric.mean().cpu()))
            totals["q"].append(float(q_metric.mean().cpu()))
            totals["topo_q_abs"].append(float(q_metric.mean().cpu()))
            totals["magnetic_fraction"].append(float(magnetic_fraction_metric.mean().cpu()))
            totals["energy_abs"].append(float(energy_abs_metric.mean().cpu()))
            totals["energy_delta"].append(float(energy_delta_metric.mean().cpu()))
            totals["energy_pred"].append(float(energy_pred_metric.mean().cpu()))
            totals["energy_target"].append(float(energy_target_metric.mean().cpu()))
        bucket = batch.get(bucket_key, batch.get("dt_scale", batch.get("t_end_index")))
        if bucket is not None:
            bucket = bucket.detach()
            if metric_mask is not None:
                bucket = bucket[metric_mask]
                mse_bucket = mse_metric
                ang_bucket = ang_metric
                q_bucket = q_metric
                magnetic_fraction_bucket = magnetic_fraction_metric
                energy_abs_bucket = energy_abs_metric
                energy_delta_bucket = energy_delta_metric
            else:
                mse_bucket = mse_each
                ang_bucket = ang_each
                q_bucket = q_each
                magnetic_fraction_bucket = magnetic_fraction_each
                energy_abs_bucket = energy_abs_each
                energy_delta_bucket = energy_delta_each
            for b in sorted(bucket.unique().tolist()):
                mask = bucket == b
                scale_totals = by_scale.setdefault(
                    int(b),
                    {
                        "mse": [],
                        "ang": [],
                        "q": [],
                        "topo_q_abs": [],
                        "magnetic_fraction": [],
                        "energy_abs": [],
                        "energy_delta": [],
                    },
                )
                scale_totals["mse"].append(float(mse_bucket[mask].mean().cpu()))
                scale_totals["ang"].append(float(ang_bucket[mask].mean().cpu()))
                scale_totals["q"].append(float(q_bucket[mask].mean().cpu()))
                scale_totals["topo_q_abs"].append(float(q_bucket[mask].mean().cpu()))
                scale_totals["magnetic_fraction"].append(
                    float(magnetic_fraction_bucket[mask].mean().cpu())
                )
                scale_totals["energy_abs"].append(float(energy_abs_bucket[mask].mean().cpu()))
                scale_totals["energy_delta"].append(float(energy_delta_bucket[mask].mean().cpu()))
    if trajectory_visual:
        examples = _collect_fixed_time_visual_rows(
            loader.dataset,
            model,
            sampler,
            collate,
            device,
            visual_examples,
            visual_mode=normalized_visual_mode,
        )
    elif informative_visual:
        examples = select_informative_candidates(
            informative_groups,
            visual_examples,
            target_buckets=visual_bucket_indices,
        )
    model.train()
    if visual_examples > 0 and visual_out_dir is not None:
        if informative_visual:
            path = save_informative_visualization(
                examples,
                visual_out_dir,
                visual_step,
                channel=visual_channel,
                dpi=visual_dpi,
                diff_vmax=visual_diff_vmax,
                zoom=visual_zoom,
                filename_prefix=visual_filename_prefix,
            )
        else:
            path = _save_validation_visualization(
                examples,
                visual_out_dir,
                visual_step,
                channel=visual_channel,
                dpi=visual_dpi,
                filename_prefix=visual_filename_prefix,
            )
        if path is not None:
            print({"val_visualization": str(path)}, flush=True)
    out = {k: sum(v) / max(1, len(v)) for k, v in totals.items()}
    for scale, scale_totals in sorted(by_scale.items()):
        for key, values in scale_totals.items():
            out[f"{key}_dt{scale}"] = sum(values) / max(1, len(values))
    return out


def reduce_validation_metrics(metrics: dict[str, float]) -> dict[str, float]:
    if not dist.is_available() or not dist.is_initialized():
        return metrics
    gathered: list[dict[str, float] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, metrics)
    keys = sorted({key for item in gathered if item is not None for key in item})
    reduced: dict[str, float] = {}
    for key in keys:
        values = [float(item[key]) for item in gathered if item is not None and key in item]
        if values:
            reduced[key] = sum(values) / len(values)
    return reduced


def _validation_log_payload(
    step: int,
    metrics: dict[str, float],
    cfg: dict[str, Any],
    bucket_key: str,
) -> dict[str, Any]:
    summary_keys = (
        "loss",
        "cfm",
        "unit",
        "llg",
        "topo",
        "mse",
        "ang",
        "q",
        "topo_q_abs",
        "magnetic_fraction",
        "energy_abs",
        "energy_delta",
        "energy_pred",
        "energy_target",
    )
    validation: dict[str, Any] = {"step": int(step)}
    for key in summary_keys:
        if key in metrics:
            validation[key] = float(metrics[key])

    if bucket_key == "t_end_index":
        t_end_ns = [float(x) for x in cfg.get("data", {}).get("t_end_ns", [])]
        by_t_end: list[dict[str, float | int]] = []
        for idx, t_ns in enumerate(t_end_ns):
            row: dict[str, float | int] = {"index": idx, "t_end_ns": t_ns}
            for metric in (
                "mse",
                "ang",
                "q",
                "topo_q_abs",
                "magnetic_fraction",
                "energy_abs",
                "energy_delta",
            ):
                old_key = f"{metric}_dt{idx}"
                if old_key in metrics:
                    row[metric] = float(metrics[old_key])
            if len(row) > 2:
                by_t_end.append(row)
        validation["bucket_key"] = "t_end_index"
        validation["by_t_end_ns"] = by_t_end
        alpha_max = cfg.get("train", {}).get("loss", {}).get("alpha_t_end_ns_max")
        if alpha_max is not None:
            validation["metric_t_end_ns_max"] = float(alpha_max)
    else:
        by_bucket: list[dict[str, float | int]] = []
        bucket_indices = sorted(
            {
                int(key.rsplit("_dt", 1)[1])
                for key in metrics
                if re.search(r"_dt\d+$", key)
            }
        )
        for idx in bucket_indices:
            row: dict[str, float | int] = {"index": idx}
            for metric in (
                "mse",
                "ang",
                "q",
                "topo_q_abs",
                "magnetic_fraction",
                "energy_abs",
                "energy_delta",
            ):
                old_key = f"{metric}_dt{idx}"
                if old_key in metrics:
                    row[metric] = float(metrics[old_key])
            by_bucket.append(row)
        validation["bucket_key"] = bucket_key
        validation["by_bucket"] = by_bucket
    return {"validation": validation}


def _format_metric_value(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return str(val)
    abs_val = abs(val)
    if abs_val > 0.0 and (abs_val < 1e-3 or abs_val >= 1e4):
        return f"{val:.3e}"
    return f"{val:.4g}"


def _format_validation_summary(
    step: int,
    metrics: dict[str, float],
    cfg: dict[str, Any],
    bucket_key: str,
) -> str:
    def metric_line(fields: tuple[tuple[str, str], ...]) -> str:
        return "  ".join(
            f"{label}={_format_metric_value(metrics[key])}"
            for key, label in fields
            if key in metrics
        )

    lines = [f"validation step={int(step)}"]
    loss_line = metric_line(
        (
            ("loss", "loss"),
            ("cfm", "cfm"),
            ("unit", "unit"),
            ("llg", "llg"),
            ("topo", "topo"),
        )
    )
    if loss_line:
        lines.append(f"  losses:   {loss_line}")
    metric_line_text = metric_line(
        (
            ("mse", "mse"),
            ("ang", "ang"),
            ("energy_abs", "|dE|"),
            ("energy_delta", "dE"),
            ("q", "dQ"),
            ("magnetic_fraction", "mag"),
        )
    )
    if metric_line_text:
        lines.append(f"  metrics:  {metric_line_text}")

    bucket_indices = sorted(
        {
            int(key.rsplit("_dt", 1)[1])
            for key in metrics
            if re.search(r"_dt\d+$", key)
        }
    )
    if not bucket_indices:
        return "\n".join(lines)

    t_end_ns = [float(x) for x in cfg.get("data", {}).get("t_end_ns", [])]
    lines.append(f"  buckets ({bucket_key}):")
    for idx in bucket_indices:
        if bucket_key == "t_end_index" and idx < len(t_end_ns):
            label = f"{t_end_ns[idx]:g}ns"
        else:
            label = f"{bucket_key}={idx}"
        bucket_values = []
        for metric_key, metric_label in (
            ("mse", "mse"),
            ("ang", "ang"),
            ("energy_abs", "|dE|"),
            ("energy_delta", "dE"),
            ("q", "dQ"),
            ("magnetic_fraction", "mag"),
        ):
            old_key = f"{metric_key}_dt{idx}"
            if old_key in metrics:
                bucket_values.append(f"{metric_label}={_format_metric_value(metrics[old_key])}")
        if bucket_values:
            lines.append(f"    {label:<8} " + "  ".join(bucket_values))
    return "\n".join(lines)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    cfg: dict[str, Any],
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.shadow,
            "config": cfg,
        },
        path,
    )


def _checkpoint_model_state(
    payload: dict[str, Any],
    state_name: str,
) -> dict[str, torch.Tensor]:
    state_name = str(state_name).lower()
    if state_name == "model":
        state = payload.get("model", payload)
    elif state_name == "ema":
        state = payload.get("ema")
        if state is None:
            raise KeyError("Checkpoint does not contain EMA weights")
    else:
        raise ValueError(f"Checkpoint state must be model|ema, got {state_name!r}")
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint {state_name!r} state is not a state dict")
    return state


def _load_checkpoint_model_state(
    model: torch.nn.Module,
    payload: dict[str, Any],
    *,
    state_name: str,
    strict: bool,
) -> None:
    state_name = str(state_name).lower()
    state = _adapt_resume_state_to_model(
        _strip_compile_prefix(_checkpoint_model_state(payload, state_name)),
        model,
    )
    if state_name != "ema":
        model.load_state_dict(state, strict=strict)
        return

    # EMA intentionally tracks only floating model state. Preserve non-floating
    # buffers from the freshly built model while still validating all trainable
    # and floating entries under strict loading.
    target = model.state_dict()
    unexpected = [key for key in state if key not in target]
    missing = [
        key
        for key, value in target.items()
        if EMA._trackable(key, value) and key not in state
    ]
    if strict and (unexpected or missing):
        raise RuntimeError(
            "EMA checkpoint is incompatible with the model: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    target.update({key: value for key, value in state.items() if key in target})
    model.load_state_dict(target, strict=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_segment_pushforward_teacher(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any] | None:
    segment_cfg = cfg.get("train", {}).get("segment_pushforward", {}) or {}
    if (
        not bool(segment_cfg.get("enabled", False))
        or float(segment_cfg.get("pred_weight", 1.0)) <= 0.0
    ):
        return None
    teacher_checkpoint = segment_cfg.get("teacher_checkpoint") or cfg.get("train", {}).get(
        "resume_from"
    )
    if not teacher_checkpoint:
        raise ValueError("segment_pushforward.teacher_checkpoint is required for stage-2 inference")
    path = Path(teacher_checkpoint)
    payload = torch.load(path, map_location=device, weights_only=False)
    state_name = str(segment_cfg.get("teacher_state", "ema")).lower()
    _load_checkpoint_model_state(
        model,
        payload,
        state_name=state_name,
        strict=bool(segment_cfg.get("teacher_strict", True)),
    )
    identity = {
        "checkpoint": str(path.resolve()),
        "sha256": _sha256_file(path) if is_main_process() else "",
        "state": state_name,
        "step": int(payload.get("step", -1)),
    }
    if is_main_process():
        print({"stage2_teacher": identity}, flush=True)
    return identity


def maybe_resume_training(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    device: torch.device,
) -> int:
    resume_from = cfg.get("train", {}).get("resume_from")
    if not resume_from:
        return 0
    path = Path(resume_from)
    payload = torch.load(path, map_location=device, weights_only=False)
    train_cfg = cfg.get("train", {})
    segment_cfg = train_cfg.get("segment_pushforward", {}) or {}
    default_state = (
        str(segment_cfg.get("teacher_state", "ema"))
        if bool(segment_cfg.get("enabled", False))
        else "model"
    )
    state_name = str(train_cfg.get("resume_model_state", default_state)).lower()
    strict = bool(cfg.get("train", {}).get("resume_strict", True))
    _load_checkpoint_model_state(
        model,
        payload,
        state_name=state_name,
        strict=strict,
    )
    if bool(cfg.get("train", {}).get("resume_load_optimizer", True)) and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    load_ema = bool(cfg.get("train", {}).get("resume_load_ema", True))
    if load_ema and "ema" in payload:
        ema_state = _adapt_resume_state_to_model(_strip_compile_prefix(payload["ema"]), model)
        ema.shadow = {
            key: value.detach().clone()
            for key, value in ema_state.items()
            if torch.is_tensor(value) and EMA._trackable(key, value)
        }
    else:
        ema.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if torch.is_tensor(value) and EMA._trackable(key, value)
        }
    if bool(cfg.get("train", {}).get("resume_reset_step", False)):
        step = 0
    else:
        step = int(payload.get("step", -1)) + 1
    if is_main_process():
        print(
            {
                "resume_from": str(path),
                "resume_model_state": state_name,
                "resume_start_step": step,
            },
            flush=True,
        )
    return max(0, step)


def ensure_memmap_for_training(cfg: dict[str, Any], distributed: bool) -> None:
    data_cfg = cfg["data"]
    memmap_cfg = data_cfg.get("memmap", {})
    if not bool(memmap_cfg.get("enabled", False)):
        return

    from skyrmion_cfm.data.memmap import ensure_memmap

    memmap_path = Path(memmap_cfg.get("path", "outputs/skyrmion_cfm/memmap"))
    force = bool(memmap_cfg.get("force_rebuild", False))
    build_started_at = time.time()

    kwargs = {
        "frame_glob": data_cfg.get("frame_glob", "run.out/m*.ovf"),
        "dtype": memmap_cfg.get("dtype", "float16"),
        "include_before_drive": bool(data_cfg.get("include_before_drive", True)),
        "workers": (
            int(os.environ["SKYRMION_MEMMAP_WORKERS"])
            if "SKYRMION_MEMMAP_WORKERS" in os.environ
            else memmap_cfg.get("workers")
        ),
    }

    if not distributed or is_main_process():
        ensure_memmap(
            data_cfg["dataset_root"],
            memmap_path,
            auto_build=bool(memmap_cfg.get("auto_build", False)),
            force=force,
            **kwargs,
        )
        memmap_cfg["force_rebuild"] = False
        return

    # Do not wait inside an NCCL collective while rank 0 builds the memmap:
    # converting thousands of OVF files can take hours and would trip the
    # process-group watchdog. Poll the finished manifest on the filesystem,
    # then use a short barrier after all ranks are ready.
    manifest = memmap_path / "manifest.json"
    while True:
        if manifest.exists() and (not force or manifest.stat().st_mtime >= build_started_at):
            try:
                ensure_memmap(data_cfg["dataset_root"], memmap_path, auto_build=False, force=False, **kwargs)
            except (json.JSONDecodeError, ValueError):
                time.sleep(30.0)
                continue
            memmap_cfg["force_rebuild"] = False
            return
        time.sleep(30.0)


def _loss_item(loss: LossOutput | dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if isinstance(loss, dict):
        if key not in loss and key in {"unit", "llg", "topo", "endpoint", "action", "action_valid_fraction"}:
            total = loss.get("total")
            if torch.is_tensor(total):
                return total.new_tensor(0.0)
        return loss[key]
    value = getattr(loss, key, None)
    if value is None and key in {"unit", "llg", "topo", "endpoint", "action", "action_valid_fraction"}:
        return loss.total.new_tensor(0.0)
    return value


def build_forward_loss_and_sampler(
    cfg: dict[str, Any],
    stats: TrainingStats,
) -> tuple[CFMLoss, BridgeSampler]:
    bridge, rotation_prior, cart_prior, rfm_prior = make_bridge_and_priors(cfg, stats)
    loss_cfg = cfg["train"]["loss"]
    loss_fn = CFMLoss(
        bridge=bridge,
        rotation_prior=rotation_prior,
        cart_prior=cart_prior,
        rfm_prior=rfm_prior,
        cfm_weight=float(loss_cfg.get("cfm_weight", 1.0)),
        unit_weight=float(loss_cfg.get("unit_weight", 0.0)),
        llg_weight=float(loss_cfg.get("llg_weight", 0.0)),
        llg_max_scale=int(loss_cfg.get("llg_max_scale", 5)),
        topo_weight=float(loss_cfg.get("topo_weight", 0.0)),
        alpha_t_end_ns_max=float(loss_cfg.get("alpha_t_end_ns_max", 1.0)),
        alpha_mix=float(loss_cfg.get("alpha_mix", 0.0)),
        tau_sampling=str(loss_cfg.get("tau_sampling", "uniform")),
        weight_mode=str(loss_cfg.get("weight_mode", "uniform")),
        target_rms_reweight=loss_cfg.get("target_rms_reweight"),
        drive_family_reweight=loss_cfg.get("drive_family_reweight"),
        segment_role_reweight=loss_cfg.get("segment_role_reweight"),
        change_aware_reweight=loss_cfg.get("change_aware_reweight"),
        minibatch_ot=loss_cfg.get("minibatch_ot"),
        cfg_dropout=loss_cfg.get("cfg_dropout"),
        action_matching=loss_cfg.get("action_matching"),
        boundary=str(cfg["data"].get("boundary", "open")),
        antipodal_eps=float(loss_cfg.get("antipodal_eps", 0.0)),
        void_loss_weight=float(loss_cfg.get("void_loss_weight", 1.0e-2)),
    )
    sampler_cfg = cfg.get("sampler", {})
    sampler = BridgeSampler(
        bridge=bridge,
        rotation_prior=rotation_prior,
        cart_prior=cart_prior,
        rfm_prior=rfm_prior,
        ode_steps=int(sampler_cfg.get("ode_steps", 20)),
        method=str(sampler_cfg.get("method", "heun")),
        classifier_free_guidance=sampler_cfg.get("classifier_free_guidance"),
        stochastic_sampler=sampler_cfg.get("stochastic_sampler"),
    )
    return loss_fn, sampler


def run_training(
    cfg: dict[str, Any],
    *,
    checkpoint_prefix: str = "checkpoint",
    final_checkpoint_name: str = "checkpoint_final.pt",
    visual_filename_prefix: str = "",
    destroy_distributed: bool = True,
) -> None:
    distributed, rank, local_rank, world_size = init_distributed()
    performance_cfg = cfg.get("performance", {}) or {}
    if bool(performance_cfg.get("use_fused_ops", False)):
        os.environ["SKYRMION_CFM_FUSED_OPS"] = "1"
    if bool(performance_cfg.get("disable_cudnn_sdp", False)) and hasattr(
        torch.backends.cuda, "enable_cudnn_sdp"
    ):
        torch.backends.cuda.enable_cudnn_sdp(False)
        if is_main_process():
            print(
                {
                    "attention_backends": {
                        "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled(),
                        "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
                        "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
                        "math_sdp": torch.backends.cuda.math_sdp_enabled(),
                    }
                },
                flush=True,
            )
    seed_everything(int(cfg.get("seed", 0)))
    device = resolve_device(cfg.get("device", "auto"), distributed, local_rank)
    out_dir = Path(cfg.get("output_dir", "outputs/skyrmion_cfm"))
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    ensure_memmap_for_training(cfg, distributed)
    if distributed:
        dist.barrier()

    use_fixed_time = bool(cfg["data"].get("t_end_ns"))
    if use_fixed_time:
        train_ds, val_ds, _ = build_fixed_time_datasets(cfg)
    else:
        train_ds, val_ds, _ = build_datasets(cfg)
    if use_fixed_time and bool(getattr(train_ds, "quality_sampling_enabled", False)):
        if is_main_process():
            payload = train_ds.prepare_quality_sampling_cache()
            print(
                {
                    "quality_sampling": "ready",
                    "cache": train_ds.quality_sampling.get("cache_path"),
                    "summary": payload.get("summary", {}),
                },
                flush=True,
            )
        if distributed:
            dist.barrier()
            if not is_main_process():
                train_ds.load_quality_sampling_cache()
    stats = make_training_stats_synced(train_ds, cfg)
    disable_unused_fixed_time_omega_targets(train_ds, val_ds, cfg)
    # Run a label audit on the training records so the embedder skips
    # fully-fixed scalars (plan-2 §forward.conditional audit).
    if use_fixed_time:
        records = train_ds.records
    else:
        records = list(train_ds.index.records)
    audit_rows = []
    audit_t_end_ns = [float(x) for x in cfg["data"].get("t_end_ns", [0.25])]
    for rec in records:
        if getattr(rec, "is_v4", False):
            audit_rows.extend(_v4_segment_condition_rows(rec, audit_t_end_ns))
        else:
            row = rec.condition_row(dt_s=0.0)
            row.update(rec.material_row())
            audit_rows.append(row)
    audit = audit_scalar_conditions(audit_rows)
    cfg["condition_audit"] = audit
    raw_model = build_model(cfg, stats.condition).to(device)
    train_model: torch.nn.Module = raw_model
    if bool(cfg["train"].get("compile", False)) and hasattr(torch, "compile"):
        train_model = torch.compile(raw_model)
    model: torch.nn.Module = train_model
    if distributed:
        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank}
        model = DistributedDataParallel(train_model, **ddp_kwargs)

    loss_fn, sampler = build_forward_loss_and_sampler(cfg, stats)
    rect_cfg = cfg.get("train", {}).get("rectification", {}) or {}
    rectification_cache = build_rectification_cache(rect_cfg)
    rectification_teacher = build_rectification_teacher(rect_cfg, cfg, stats, device)

    teacher_identity = load_segment_pushforward_teacher(cfg, raw_model, device)
    segment_handoff_cache = prepare_segment_handoff_cache(
        cfg,
        train_ds,
        raw_model,
        sampler,
        device,
        out_dir,
        distributed=distributed,
        teacher_identity=teacher_identity,
    )
    segment_physical_target_cache = prepare_segment_physical_target_cache(
        cfg,
        segment_handoff_cache,
    )

    arch = cfg["model"].get("arch", "unet")
    base_lr = float(cfg["train"].get("dit_lr" if arch == "dit" else "lr", cfg["train"]["lr"]))
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=base_lr,
        betas=tuple(cfg["train"].get("betas", [0.9, 0.95])),
        weight_decay=float(cfg["train"].get("weight_decay", 0.01)),
    )
    ema = EMA(raw_model, float(cfg["train"].get("ema_decay", 0.9999)))
    start_step = maybe_resume_training(cfg, raw_model, optimizer, ema, device)

    curriculum_steps = int(cfg["train"].get("curriculum_steps", 0))
    curriculum_max_scale = int(cfg["train"].get("curriculum_max_scale", 5))
    if curriculum_steps > 0:
        train_ds.allowed_scales = {s for s in train_ds.dt_scales if s <= curriculum_max_scale}

    train_batch_size = per_rank_train_batch_size(cfg, world_size)
    if distributed and is_main_process():
        print(
            {
                "ddp_world_size": world_size,
                "global_train_batch_size": int(cfg["train"]["batch_size"]),
                "per_rank_train_batch_size": train_batch_size,
            },
            flush=True,
        )

    def make_train_loader() -> tuple[DataLoader, DistributedSampler | None]:
        sampler = (
            DistributedSampler(
                train_ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=int(cfg.get("seed", 0)),
                drop_last=True,
            )
            if distributed
            else None
        )
        return DataLoader(
            train_ds,
            batch_size=train_batch_size,
            shuffle=False,
            sampler=sampler,
            drop_last=True,
            **dataloader_options(cfg, device, split="train"),
        ), sampler

    def make_val_loader() -> DataLoader:
        sampler = (
            DistributedSampler(
                val_ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            if distributed
            else None
        )
        return DataLoader(
            val_ds,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,
            sampler=sampler,
            drop_last=False,
            **dataloader_options(cfg, device, split="val"),
        )

    t_end_curriculum = cfg.get("data", {}).get("t_end_curriculum") if use_fixed_time else None
    t_end_curriculum = t_end_curriculum or []
    current_t_end_stage: int | None = None

    def apply_t_end_curriculum(step: int) -> bool:
        nonlocal current_t_end_stage
        if not t_end_curriculum:
            return False
        stage_idx, raw_probs = _t_end_curriculum_stage(t_end_curriculum, step)
        if stage_idx == current_t_end_stage:
            return False
        if raw_probs is None:
            train_ds.t_end_probs = None
            probs_list = None
        else:
            probs = _normalize_t_end_probs(raw_probs, len(train_ds.t_end_ns))
            train_ds.t_end_probs = probs
            probs_list = [float(x) for x in probs]
        current_t_end_stage = stage_idx
        if is_main_process():
            print(
                {
                    "t_end_curriculum": {
                        "step": int(step),
                        "stage": int(stage_idx),
                        "probs": probs_list,
                    }
                },
                flush=True,
            )
        return True

    apply_t_end_curriculum(start_step)
    train_loader, train_sampler = make_train_loader()
    amp_dtype = cfg["train"].get("amp_dtype", "bf16")
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    use_amp = device.type == "cuda"
    grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    iterator = cycle_loader(train_loader, train_sampler)
    total_train_steps = int(cfg["train"]["steps"])
    # tqdm's carriage-return stream is useful in a terminal but unreadable in
    # run logs. Scheduled runs use stable one-line stdout progress records.
    scheduled_run = bool(os.environ.get("RUN_ID"))
    pbar = tqdm(
        range(start_step, total_train_steps),
        dynamic_ncols=True,
        disable=not is_main_process() or scheduled_run,
    )
    progress_started = time.monotonic()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    active_collate = collate_fixed_time_conditions if use_fixed_time else collate_conditions
    # Pushforward rollout (opt-in, fixed-time only). When enabled the train
    # loader yields clip batches (B, K, ...) and we roll the model forward.
    rollout_cfg = cfg.get("train", {}).get("rollout", {}) or {}
    segment_pf_cfg = cfg.get("train", {}).get("segment_pushforward", {}) or {}
    segment_pf_enabled = use_fixed_time and bool(segment_pf_cfg.get("enabled", False))
    if segment_pf_enabled and is_main_process():
        kept_handoffs, candidate_handoffs = _segment_handoff_prediction_counts(
            segment_pf_cfg
        )
        print(
            {
                "stage2_training_start": start_step,
                "predicted_inits_per_segment": kept_handoffs,
                "candidate_inits_per_segment": candidate_handoffs,
            },
            flush=True,
        )
    rollout_enabled = (
        use_fixed_time
        and not segment_pf_enabled
        and bool(rollout_cfg.get("enabled", True))
        and int(rollout_cfg.get("steps", 0)) >= 2
    )
    rollout_needs_omega = isinstance(loss_fn.bridge, (RotationVectorBridge, RotationVector2DBridge))
    rollout_roll_ode_steps = int(rollout_cfg.get("roll_ode_steps", 0)) or None
    bucket_key = "t_end_index" if use_fixed_time else "dt_scale"
    default_visual_examples = int(cfg["train"].get("visual_examples", 5 if use_fixed_time else 0))
    default_visual_buckets = _default_visual_bucket_indices(
        len(cfg["data"].get("t_end_ns", [])),
        default_visual_examples,
    )
    visual_out_dir_cfg = cfg["train"].get("visual_out_dir")
    visual_out_dir = Path(visual_out_dir_cfg) if visual_out_dir_cfg else out_dir / "val_visualizations"
    visual_mode = str(cfg["train"].get("visual_mode", "informative"))
    visual_diff_vmax = float(cfg["train"].get("visual_diff_vmax", 1.0))
    visual_zoom = bool(cfg["train"].get("visual_zoom", True))
    val_max_batches = int(cfg["train"].get("val_max_batches", 20))
    val_max_batches_local = max(1, math.ceil(val_max_batches / world_size)) if distributed else val_max_batches
    early_stop_metric_key = cfg["train"].get("early_stop_metric")
    early_stop_patience = int(cfg["train"].get("early_stop_patience", 0))
    best_metric: float | None = None
    best_step = -1
    plateau = 0
    timing_cfg = cfg.get("performance", {}).get("step_timing", {}) or {}
    timer = StepTimer(bool(timing_cfg.get("enabled", False)), device)
    timing_every = max(1, int(timing_cfg.get("every", cfg["train"].get("log_every", 50))))
    timing_steps = 0
    quality_batch_reported = False
    for step in pbar:
        timer.start()
        if step == curriculum_steps and curriculum_steps > 0 and not use_fixed_time:
            train_ds.allowed_scales = None
            train_loader, train_sampler = make_train_loader()
            iterator = cycle_loader(train_loader, train_sampler)
        if apply_t_end_curriculum(step):
            train_loader, train_sampler = make_train_loader()
            iterator = cycle_loader(train_loader, train_sampler)
        set_lr(optimizer, cosine_lr(step, cfg, base_lr))
        logs = []
        for accum_step in range(grad_accum):
            raw_batch = next(iterator)
            if (
                is_main_process()
                and not quality_batch_reported
                and "quality_pair_category_index" in raw_batch
            ):
                pair_category = raw_batch["quality_pair_category_index"].reshape(-1)
                segment_category = raw_batch["quality_segment_category_index"].reshape(-1)
                print(
                    {
                        "quality_sampling_first_batch": {
                            "pair_informative": int((pair_category == 0).sum()),
                            "pair_static": int((pair_category == 1).sum()),
                            "pair_noise_only": int((pair_category == 2).sum()),
                            "original_mix": int(raw_batch["quality_original_mix"].sum()),
                            "forced_accept": int(raw_batch["quality_forced_accept"].sum()),
                            "mean_attempts": float(
                                raw_batch["quality_sampling_attempts"].float().mean()
                            ),
                            "segment_informative": int((segment_category == 0).sum()),
                            "segment_static": int((segment_category == 1).sum()),
                            "segment_noise_only": int((segment_category == 2).sum()),
                        }
                    },
                    flush=True,
                )
                quality_batch_reported = True
            timer.mark("data")
            batch = move_batch(raw_batch, device)
            timer.mark("to_device")
            sync_context = (
                model.no_sync()
                if distributed and isinstance(model, DistributedDataParallel) and accum_step < grad_accum - 1
                else nullcontext()
            )
            with sync_context:
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                    if segment_pf_enabled:
                        loss = segment_pushforward_loss(
                            model,
                            loss_fn,
                            batch,
                            needs_omega=rollout_needs_omega,
                            cfg=segment_pf_cfg,
                            handoff_cache=segment_handoff_cache,
                            physical_target_cache=segment_physical_target_cache,
                        )
                    elif rollout_enabled:
                        loss = pushforward_loss(
                            model, loss_fn, sampler, batch,
                            needs_omega=rollout_needs_omega,
                            roll_ode_steps=rollout_roll_ode_steps,
                            endpoint_cfg=rollout_cfg.get("endpoint_loss"),
                        )
                    else:
                        cond = active_collate(batch)
                        timer.mark("collate")
                        rect_batch = apply_teacher_true_rectification(
                            batch,
                            cond,
                            rectification_teacher,
                            rect_cfg,
                            needs_omega=isinstance(loss_fn.bridge, (RotationVectorBridge, RotationVector2DBridge)),
                            endpoint_cache=rectification_cache,
                        )
                        timer.mark("rectify")
                        loss = loss_fn(model, rect_batch, cond)
                        timer.mark("cfm_loss")
                        loss = add_aux_endpoint_cfm_loss(
                            loss,
                            model,
                            loss_fn,
                            rect_batch,
                            cfg.get("train", {}).get("aux_endpoint_loss"),
                        )
                        timer.mark("aux_loss")
                        loss = add_endpoint_consistency_loss(
                            loss,
                            model,
                            sampler,
                            rect_batch,
                            cond,
                            cfg.get("train", {}).get("endpoint_loss"),
                            step=step,
                            boundary=loss_fn.boundary,
                        )
                        timer.mark("endpoint")
                    scaled = _loss_item(loss, "total") / grad_accum
                scaled.backward()
                timer.mark("backward")
            logs.append(loss)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
        timer.mark("clip")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update(raw_model)
        completed_step = step + 1
        timer.mark("optim")
        timing_steps += 1

        log_every = int(cfg["train"].get("log_every", 50))
        if is_main_process() and (
            step % log_every == 0 or completed_step == total_train_steps
        ):
            loss_values = {
                key: sum(
                    float(_loss_item(item, key).detach().cpu()) for item in logs
                )
                / len(logs)
                for key in (
                    "total",
                    "cfm",
                    "llg",
                    "topo",
                    "endpoint",
                    "action",
                    "action_valid_fraction",
                )
            }
            if not scheduled_run:
                pbar.set_postfix(
                    loss=f"{loss_values['total']:.4g}",
                    cfm=f"{loss_values['cfm']:.4g}",
                    llg=f"{loss_values['llg']:.4g}",
                    topo=f"{loss_values['topo']:.4g}",
                    endpoint=f"{loss_values['endpoint']:.4g}",
                    action=f"{loss_values['action']:.4g}",
                    am_valid=f"{loss_values['action_valid_fraction']:.3g}",
                )
            elapsed = max(0.0, time.monotonic() - progress_started)
            steps_this_run = max(0, completed_step - start_step)
            steps_per_second = steps_this_run / elapsed if elapsed > 0.0 else 0.0
            eta_seconds = (
                (total_train_steps - completed_step) / steps_per_second
                if steps_per_second > 0.0
                else None
            )
            finish = (
                datetime.now().astimezone() + timedelta(seconds=eta_seconds)
                if eta_seconds is not None
                else None
            )
            percent = 100.0 * completed_step / max(total_train_steps, 1)
            finish_text = (
                finish.strftime("%Y-%m-%d %H:%M:%S %Z")
                if finish is not None
                else "calculating"
            )
            print(
                f"[TRAIN] {completed_step:>6}/{total_train_steps:<6} "
                f"{percent:6.2f}% {_progress_bar(completed_step, total_train_steps)} "
                f"speed={steps_per_second:.3f} step/s "
                f"elapsed={_format_progress_duration(elapsed)} "
                f"ETA={_format_progress_duration(eta_seconds)} "
                f"finish={finish_text} "
                f"loss={loss_values['total']:.5g} "
                f"cfm={loss_values['cfm']:.5g} "
                f"endpoint={loss_values['endpoint']:.5g}",
                flush=True,
            )
        if timer.enabled and is_main_process() and timing_steps >= timing_every:
            print({"step_timing": step, **timer.report(timing_steps)}, flush=True)
            timer.reset()
            timing_steps = 0
        if _should_run_validation(
            completed_step,
            total_train_steps,
            int(cfg["train"].get("val_every", 5000)),
        ):
            if distributed:
                dist.barrier()
            stop_now = False
            backup = ema.copy_to(raw_model)
            val_loader = make_val_loader()
            try:
                local_metrics = validate(
                    raw_model,
                    val_loader,
                    loss_fn,
                    sampler,
                    device,
                    max_batches=val_max_batches_local,
                    collate_fn=active_collate,
                    bucket_key=bucket_key,
                    visual_examples=default_visual_examples if is_main_process() else 0,
                    visual_out_dir=visual_out_dir if is_main_process() else None,
                    visual_step=completed_step,
                    visual_channel="mz",
                    visual_dpi=180,
                    visual_bucket_indices=default_visual_buckets,
                    visual_filename_prefix=visual_filename_prefix,
                    visual_mode=visual_mode,
                    visual_diff_vmax=visual_diff_vmax,
                    visual_zoom=visual_zoom,
                )
            finally:
                del val_loader
                ema.restore(raw_model, backup)
                model.train()
            metrics = reduce_validation_metrics(local_metrics)
            if is_main_process():
                print(
                    {
                        "validation_parallel": {
                            "world_size": world_size,
                            "max_batches_global": val_max_batches,
                            "max_batches_per_rank": val_max_batches_local,
                        }
                    },
                    flush=True,
                )
                print(
                    f"\n{_format_validation_summary(completed_step, metrics, cfg, bucket_key)}\n",
                    flush=True,
                )
                print(
                    _validation_log_payload(completed_step, metrics, cfg, bucket_key),
                    flush=True,
                )
                if early_stop_metric_key is not None:
                    metric_key = str(early_stop_metric_key)
                    current = float(metrics.get(metric_key, metrics.get("loss", float("inf"))))
                    if best_metric is None or current < best_metric:
                        best_metric = current
                        best_step = step
                        plateau = 0
                        save_checkpoint(
                            out_dir / f"{checkpoint_prefix}_best.pt",
                            raw_model,
                            optimizer,
                            ema,
                            cfg,
                            step,
                        )
                    else:
                        plateau += 1
                        if early_stop_patience > 0 and plateau >= early_stop_patience:
                            print(
                                f"early stop at step {step}: {metric_key}={current:.4g} "
                                f"vs best={best_metric:.4g} (step {best_step})",
                                flush=True,
                            )
                            stop_now = True
            if distributed:
                payload = [stop_now]
                dist.broadcast_object_list(payload, src=0)
                stop_now = bool(payload[0])
            if distributed:
                dist.barrier()
            if stop_now:
                break
        if step > 0 and step % int(cfg["train"].get("save_every", 5000)) == 0:
            if distributed:
                dist.barrier()
            if is_main_process():
                save_checkpoint(out_dir / f"{checkpoint_prefix}_{step:07d}.pt", raw_model, optimizer, ema, cfg, step)
            if distributed:
                dist.barrier()
    if is_main_process():
        save_checkpoint(out_dir / final_checkpoint_name, raw_model, optimizer, ema, cfg, int(cfg["train"]["steps"]))
    if distributed:
        dist.barrier()
        if destroy_distributed:
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a skyrmion CFM velocity model.")
    parser.add_argument("--config", default="skyrmion_cfm/configs/default.yaml")
    parser.add_argument(
        "--stage",
        default=os.environ.get("STAGE") or os.environ.get("SKYRMION_CFM_STAGE"),
        help="Run only one named stage, or a zero/one-based stage index.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    selected_stage = str(args.stage).strip() if args.stage is not None else None
    if selected_stage == "":
        selected_stage = None
    stages = cfg.get("stages") or []
    if not stages:
        if selected_stage is not None:
            raise SystemExit("--stage was provided, but the config has no stages")
        run_training(cfg)
        return
    base_cfg = {key: value for key, value in cfg.items() if key != "stages"}
    matched_stage = False
    for stage_idx, stage in enumerate(stages):
        stage_name = str(stage.get("name", f"stage{stage_idx + 1}"))
        stage_selectors = {stage_name, str(stage_idx), str(stage_idx + 1)}
        if selected_stage is not None and selected_stage not in stage_selectors:
            continue
        matched_stage = True
        stage_slug = _filename_slug(stage_name)
        overrides = stage.get("overrides", {}) or {}
        stage_cfg = merge_config(base_cfg, overrides)
        if int(os.environ.get("RANK", "0")) == 0:
            print({"stage": stage_name, "stage_index": stage_idx, "output_dir": stage_cfg.get("output_dir")}, flush=True)
        run_training(
            stage_cfg,
            checkpoint_prefix=f"{stage_slug}_checkpoint",
            final_checkpoint_name=f"{stage_slug}_checkpoint_final.pt",
            visual_filename_prefix=f"{stage_slug}_",
            destroy_distributed=selected_stage is not None or stage_idx == len(stages) - 1,
        )
    if selected_stage is not None and not matched_stage:
        raise SystemExit(f"stage not found: {selected_stage}")


if __name__ == "__main__":
    main()
