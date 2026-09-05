#!/usr/bin/env python3
"""Paired teacher-forced/autoregressive evaluation for SKX x5 checkpoints.

The first eligible control segment starts from the dataset-provided state.  At
every later control boundary, teacher-forced (TF) evaluation resets to the
matching dataset state while fully autoregressive (AR) evaluation consumes the
previous model prediction.  TF and AR use the same sampler seed at every hop,
so their difference isolates sensitivity to the predicted handoff state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skyrmion_cfm.config import release_checkpoint_config, seed_everything
from skyrmion_cfm.data.fixed_time import (
    build_fixed_time_datasets,
    collate_fixed_time_conditions,
)
from skyrmion_cfm.data.stats import load_training_stats
from skyrmion_cfm.eval.conditional_algorithm_diagnostics import (
    _build_sampler,
    _load_state_verified,
)
from skyrmion_cfm.eval.metrics import (
    angular_error_deg,
    energy_density,
    mse_m,
    topological_charge,
)
from skyrmion_cfm.models import build_model
from skyrmion_cfm.train import disable_unused_fixed_time_omega_targets, move_batch


METRICS = ("mse", "ang", "q_abs", "energy_abs", "q_fail")


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("evaluation produced no rows")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _base_id(run_id: str) -> str:
    match = re.search(r"(?:^|_)base(\d+)(?:_|$)", run_id)
    return f"base{match.group(1)}" if match else run_id


def _set_sampler_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _metric_tensors(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    b_t: torch.Tensor,
    boundary: str,
) -> dict[str, torch.Tensor]:
    q_pred = topological_charge(pred, boundary=boundary)
    q_target = topological_charge(target, boundary=boundary)
    energy_pred = energy_density(pred, b_t, boundary=boundary, mask=mask)
    energy_target = energy_density(target, b_t, boundary=boundary, mask=mask)
    return {
        "mse": mse_m(pred, target, mask=mask),
        "ang": angular_error_deg(pred, target, mask=mask),
        "q_abs": (q_pred - q_target).abs(),
        "energy_abs": (energy_pred - energy_target).abs(),
        "q_fail": (torch.round(q_pred) != torch.round(q_target)).float(),
        "q_pred": q_pred,
        "q_target": q_target,
    }


def _input_metrics(
    ar_input: torch.Tensor,
    tf_input: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    boundary: str,
) -> dict[str, torch.Tensor]:
    q_ar = topological_charge(ar_input, boundary=boundary)
    q_tf = topological_charge(tf_input, boundary=boundary)
    return {
        "input_mse": mse_m(ar_input, tf_input, mask=mask),
        "input_ang": angular_error_deg(ar_input, tf_input, mask=mask),
        "input_q_abs": (q_ar - q_tf).abs(),
    }


def _collate_samples(samples: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    cleaned = [{key: value for key, value in sample.items() if value is not None} for sample in samples]
    return move_batch(default_collate(cleaned), device)


def _eligible_records(dataset: Any, max_records: int) -> tuple[list[int], int]:
    eligible: list[int] = []
    excluded = 0
    expected_segments: int | None = None
    for rec_idx in dataset._valid_record_idx:
        specs = dataset._segment_boundary_specs(int(rec_idx))
        if len(specs) < 2:
            excluded += 1
            continue
        if expected_segments is None:
            expected_segments = len(specs)
        if len(specs) != expected_segments:
            raise RuntimeError(
                "eligible test trajectories have inconsistent segment counts: "
                f"expected {expected_segments}, found {len(specs)} for record {rec_idx}"
            )
        eligible.append(int(rec_idx))
    if max_records > 0:
        eligible = eligible[: int(max_records)]
    if not eligible:
        raise RuntimeError("no test trajectory has at least one predicted handoff")
    return eligible, excluded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", choices=("ema", "raw", "model"), default="ema")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ode-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--draw-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--range-start-ns", type=float, default=1.0)
    parser.add_argument("--range-end-ns", type=float, default=6.0)
    parser.add_argument(
        "--checkpoint-hash",
        choices=("sha256", "skip"),
        default="sha256",
        help="Hashing a 1 GB checkpoint improves provenance but is unnecessary in dense sweeps.",
    )
    return parser


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.draw_index < 0:
        raise ValueError("--draw-index must be non-negative")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("SKYRMION_CFM_FUSED_OPS", "1")
    seed_everything(int(args.seed))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    started = time.time()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    checkpoint_cfg = checkpoint.get("config")
    if not isinstance(checkpoint_cfg, dict):
        raise RuntimeError("checkpoint does not embed its training config")
    cfg = release_checkpoint_config(checkpoint_cfg)
    data_cfg = cfg.setdefault("data", {})
    data_cfg["segment_time_range_ns"] = [
        float(args.range_start_ns),
        float(args.range_end_ns),
    ]
    data_cfg.setdefault("memmap", {})["auto_build"] = False
    data_cfg["memmap"]["force_rebuild"] = False
    cfg.setdefault("train", {})["compile"] = False

    _, _, test_dataset = build_fixed_time_datasets(cfg, build_splits={"test"})
    if test_dataset is None:
        raise RuntimeError("failed to build test split")
    disable_unused_fixed_time_omega_targets(None, test_dataset, cfg)
    eligible, excluded = _eligible_records(test_dataset, int(args.max_records))

    stats_path = Path(data_cfg["stats_cache"])
    stats = load_training_stats(stats_path)
    model = build_model(cfg, stats.condition).to(device)
    load_report = _load_state_verified(model, checkpoint, str(args.state))
    checkpoint_step = int(checkpoint.get("step", -1))
    del checkpoint
    model.eval()
    sampler = _build_sampler(cfg, stats, ode_steps=int(args.ode_steps))
    boundary = str(data_cfg.get("boundary", "open"))

    rows: list[dict[str, Any]] = []
    segment_counts: set[int] = set()
    for chunk_number, chunk_start in enumerate(range(0, len(eligible), int(args.batch_size))):
        record_indices = eligible[chunk_start : chunk_start + int(args.batch_size)]
        paths: list[list[dict[str, Any]]] = []
        for rec_idx in record_indices:
            samples: list[dict[str, Any]] = []
            for segment_position, (choice_idx, start_frame) in enumerate(
                test_dataset._segment_boundary_specs(rec_idx)
            ):
                rng = np.random.default_rng(
                    int(args.seed) + 104_729 * (rec_idx + 1) + 7_919 * (segment_position + 1)
                )
                samples.append(
                    test_dataset._build_sample(
                        rec_idx,
                        int(choice_idx),
                        rng,
                        frame_init=int(start_frame),
                        apply_augment=False,
                    )
                )
            paths.append(samples)
            segment_counts.add(len(samples))
        if len(segment_counts) != 1:
            raise RuntimeError(f"inconsistent segment counts: {sorted(segment_counts)}")

        ar_state: torch.Tensor | None = None
        for segment_position in range(len(paths[0])):
            batch = _collate_samples(
                [samples[segment_position] for samples in paths],
                device,
            )
            cond = collate_fixed_time_conditions(batch)
            tf_input = batch.get("m_init", batch.get("m0"))
            target = batch.get("m_t", batch.get("m1"))
            if not torch.is_tensor(tf_input) or not torch.is_tensor(target):
                raise RuntimeError("fixed-time batch has no input/target tensors")
            current_ar_input = tf_input if ar_state is None else ar_state
            mask = batch.get("defect_field")
            sampler_seed = (
                int(args.seed)
                + 10_000_019 * int(args.draw_index)
                + 100_003 * int(segment_position)
                + 997 * int(chunk_number)
            )
            _set_sampler_seed(sampler_seed)
            tf_pred, _ = sampler.sample(model, tf_input, cond)
            if segment_position == 0:
                ar_pred = tf_pred
            else:
                _set_sampler_seed(sampler_seed)
                ar_pred, _ = sampler.sample(model, current_ar_input, cond)
            ar_state = ar_pred.detach()

            tf_metrics = _metric_tensors(
                tf_pred,
                target,
                mask=mask,
                b_t=cond["b_t"],
                boundary=boundary,
            )
            ar_metrics = _metric_tensors(
                ar_pred,
                target,
                mask=mask,
                b_t=cond["b_t"],
                boundary=boundary,
            )
            input_metrics = _input_metrics(
                current_ar_input,
                tf_input,
                mask=mask,
                boundary=boundary,
            )

            for local_index, rec_idx in enumerate(record_indices):
                run_id = str(paths[local_index][segment_position]["run_id"])
                row: dict[str, Any] = {
                    "label": str(args.label),
                    "draw_index": int(args.draw_index),
                    "record_index": int(rec_idx),
                    "base_id": _base_id(run_id),
                    "run_id": run_id,
                    "segment_position": int(segment_position),
                    "handoff_count": int(segment_position),
                    "control_segment_index": int(
                        batch["control_segment_index"][local_index].detach().cpu()
                    ),
                    "frame_init": int(batch["frame_init"][local_index].detach().cpu()),
                    "frame_target": int(batch["frame_target"][local_index].detach().cpu()),
                    "start_time_ns": float(
                        batch["frame_init_time_ns"][local_index].detach().cpu()
                    ),
                    "end_time_ns": float(
                        batch["frame_target_time_ns"][local_index].detach().cpu()
                    ),
                    "duration_ns": float(batch["t_end_ns"][local_index].detach().cpu()),
                    "sampler_seed": int(sampler_seed),
                    "zeeman_reference_applied": int(
                        bool(
                            batch["zeeman_reference_applied"][local_index]
                            .detach()
                            .cpu()
                        )
                        if "zeeman_reference_applied" in batch
                        else False
                    ),
                }
                for key in METRICS:
                    tf_value = float(tf_metrics[key][local_index].detach().float().cpu())
                    ar_value = float(ar_metrics[key][local_index].detach().float().cpu())
                    row[f"tf_{key}"] = tf_value
                    row[f"ar_{key}"] = ar_value
                    row[f"gap_{key}"] = ar_value - tf_value
                row["tf_q_pred"] = float(
                    tf_metrics["q_pred"][local_index].detach().float().cpu()
                )
                row["ar_q_pred"] = float(
                    ar_metrics["q_pred"][local_index].detach().float().cpu()
                )
                row["q_target"] = float(
                    tf_metrics["q_target"][local_index].detach().float().cpu()
                )
                for key, value in input_metrics.items():
                    row[key] = float(value[local_index].detach().float().cpu())
                rows.append(row)

        completed = min(chunk_start + len(record_indices), len(eligible))
        print(
            json.dumps(
                {
                    "label": args.label,
                    "draw": args.draw_index,
                    "completed_trajectories": completed,
                    "total_trajectories": len(eligible),
                }
            ),
            flush=True,
        )

    checkpoint_digest = (
        _sha256(args.checkpoint) if args.checkpoint_hash == "sha256" else None
    )
    _atomic_csv(args.output, rows)
    metadata = {
        "status": "complete",
        "definition": {
            "teacher_forced": "each control segment consumes its matching dataset/model-preconditioned start",
            "fully_autoregressive": "only the first segment consumes the dataset start; every later segment consumes the previous model prediction",
            "rollout_gap": "AR error minus TF error, paired with the same sampler seed",
        },
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_hash_mode": args.checkpoint_hash,
        "checkpoint_step": checkpoint_step,
        "checkpoint_prior_type": str(cfg.get("prior", {}).get("type", "physics_scaled")),
        "checkpoint_zeeman_precondition_enabled": bool(
            cfg.get("data", {}).get("zeeman_precondition", {}).get("enabled", False)
        ),
        "checkpoint_mask_target_time": bool(
            cfg.get("model", {}).get("mask_target_time", False)
        ),
        "checkpoint_load": load_report,
        "state": args.state,
        "split": "test",
        "segment_time_range_ns": [args.range_start_ns, args.range_end_ns],
        "eligible_trajectories": len(eligible),
        "excluded_without_handoff": excluded,
        "segments_per_trajectory": sorted(segment_counts),
        "draw_index": args.draw_index,
        "seed": args.seed,
        "ode_steps": args.ode_steps,
        "sampler_method": str(cfg.get("sampler", {}).get("method", "heun")),
        "batch_size": args.batch_size,
        "rows": len(rows),
        "output_csv": str(args.output),
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(args.output.with_suffix(".json"), metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
