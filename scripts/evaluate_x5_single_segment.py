#!/usr/bin/env python3
"""Unified held-out single-segment metrics and latency for x5 paper models."""

from __future__ import annotations

import argparse
import csv
import json
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

from skyrmion_cfm.config import release_checkpoint_config
from skyrmion_cfm.config import load_config, seed_everything
from skyrmion_cfm.data.fixed_time import build_fixed_time_datasets, collate_fixed_time_conditions
from skyrmion_cfm.data.stats import load_training_stats
from skyrmion_cfm.eval.conditional_algorithm_diagnostics import _build_sampler, _load_state_verified
from skyrmion_cfm.eval.metrics import angular_error_deg, energy_density, mse_m, topological_charge
from skyrmion_cfm.models import build_model
from skyrmion_cfm.train import disable_unused_fixed_time_omega_targets, move_batch


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("evaluation produced no rows")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _collate(samples: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    cleaned = [{k: v for k, v in sample.items() if v is not None} for sample in samples]
    return move_batch(default_collate(cleaned), device)


def _base_id(run_id: str) -> str:
    match = re.search(r"(?:^|_)base(\d+)(?:_|$)", run_id)
    return f"base{match.group(1)}" if match else run_id


def _metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None, b_t: torch.Tensor, boundary: str) -> dict[str, torch.Tensor]:
    q_pred = topological_charge(pred, boundary=boundary)
    q_target = topological_charge(target, boundary=boundary)
    norm = pred.norm(dim=1)
    if mask is not None:
        weight = mask[:, 0] if mask.ndim == 4 else mask
        norm_error = ((norm - 1.0).abs() * weight).sum((-1, -2)) / weight.sum((-1, -2)).clamp_min(1.0)
    else:
        norm_error = (norm - 1.0).abs().mean((-1, -2))
    return {
        "ang": angular_error_deg(pred, target, mask=mask),
        "mse": mse_m(pred, target, mask=mask),
        "q_abs": (q_pred - q_target).abs(),
        "q_fail": (torch.round(q_pred) != torch.round(q_target)).float(),
        "energy_abs": (
            energy_density(pred, b_t, boundary=boundary, mask=mask)
            - energy_density(target, b_t, boundary=boundary, mask=mask)
        ).abs(),
        "norm_abs": norm_error,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", choices=("ema", "raw"), default="ema")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ode-steps", type=int, default=10)
    parser.add_argument("--method", choices=("heun", "euler"), default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument(
        "--durations-ns",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Evaluate explicit requested durations from each valid control-segment "
            "boundary. This supports held-out-time interpolation tests even when "
            "the durations are absent from the training bucket grid."
        ),
    )
    parser.add_argument(
        "--max-cases-per-duration",
        type=int,
        default=0,
        help="Deterministic balanced cap used with --durations-ns.",
    )
    parser.add_argument(
        "--case-selection",
        choices=("head", "random"),
        default="head",
        help="How to select --max-cases from the complete deterministic case list.",
    )
    parser.add_argument("--range-start-ns", type=float, default=1.0)
    parser.add_argument("--range-end-ns", type=float, default=6.0)
    parser.add_argument(
        "--data-config",
        type=Path,
        default=None,
        help=(
            "Optional evaluation-only config whose data section replaces the "
            "checkpoint data section. The checkpoint training-statistics cache "
            "is retained so condition normalization stays checkpoint-correct."
        ),
    )
    parser.add_argument(
        "--stats-cache-override",
        type=Path,
        default=None,
        help=(
            "Path-only relocation of the checkpoint training-statistics cache. "
            "The cache contents must be identical to the checkpoint cache."
        ),
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    return parser


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SKYRMION_CFM_FUSED_OPS", "1")
    seed_everything(args.seed)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    cfg = release_checkpoint_config(checkpoint["config"])
    checkpoint_prior_type = str(cfg.get("prior", {}).get("type", "physics_scaled"))
    checkpoint_zeeman_enabled = bool(
        cfg.get("data", {}).get("zeeman_precondition", {}).get("enabled", False)
    )
    checkpoint_mask_target_time = bool(
        cfg.get("model", {}).get("mask_target_time", False)
    )
    checkpoint_stats_cache = cfg.get("data", {}).get("stats_cache")
    if args.data_config is not None:
        evaluation_cfg = load_config(args.data_config)
        evaluation_data = deepcopy(evaluation_cfg.get("data", {}))
        if not evaluation_data:
            raise ValueError(f"data config has no data section: {args.data_config}")
        checkpoint_horizons = list(cfg.get("data", {}).get("t_end_ns", []))
        evaluation_horizons = list(evaluation_data.get("t_end_ns", []))
        if checkpoint_horizons != evaluation_horizons:
            raise ValueError(
                "checkpoint and evaluation t_end_ns grids differ: "
                f"{checkpoint_horizons} != {evaluation_horizons}"
            )
        if not checkpoint_stats_cache:
            raise ValueError("checkpoint config has no data.stats_cache")
        evaluation_data["stats_cache"] = (
            str(args.stats_cache_override)
            if args.stats_cache_override is not None
            else checkpoint_stats_cache
        )
        cfg["data"] = evaluation_data
    elif args.stats_cache_override is not None:
        cfg.setdefault("data", {})["stats_cache"] = str(args.stats_cache_override)
    cfg.setdefault("data", {})["segment_time_range_ns"] = [args.range_start_ns, args.range_end_ns]
    cfg["data"].setdefault("memmap", {})["auto_build"] = False
    cfg.setdefault("train", {})["compile"] = False
    train_dataset, val_dataset, test_dataset = build_fixed_time_datasets(
        cfg, build_splits={args.split}
    )
    datasets = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
    dataset = datasets[args.split]
    if dataset is None:
        raise RuntimeError("failed to build held-out split")
    disable_unused_fixed_time_omega_targets(None, dataset, cfg)
    stats = load_training_stats(Path(cfg["data"]["stats_cache"]))
    model = build_model(cfg, stats.condition).to(device)
    load_report = _load_state_verified(model, checkpoint, args.state)
    checkpoint_step = int(checkpoint.get("step", -1))
    del checkpoint
    model.eval()
    sampler = _build_sampler(cfg, stats, ode_steps=args.ode_steps)
    if args.method is not None:
        sampler.method = args.method
    boundary = str(cfg["data"].get("boundary", "open"))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    deterministic_source = (
        str(cfg.get("bridge", {}).get("source_mode", "")).lower() == "identity"
    )
    effective_draws = 1 if deterministic_source else int(args.draws)

    cases: list[dict[str, Any]] = []
    for rec_idx in dataset._valid_record_idx:
        for choice_idx, start_frame in dataset._segment_boundary_specs(int(rec_idx)):
            if args.durations_ns is None:
                cases.append(
                    {
                        "record_index": int(rec_idx),
                        "choice_index": int(choice_idx),
                        "start_frame": int(start_frame),
                        "target_frame": None,
                        "requested_duration_ns": None,
                    }
                )
                continue
            record = dataset.records[int(rec_idx)]
            save_step_ps = float(dataset._record_choices[int(rec_idx)][int(choice_idx)][2])
            seen_targets: set[int] = set()
            for requested_duration in args.durations_ns:
                target_frame = dataset._target_frame_for_duration(
                    int(rec_idx),
                    record,
                    save_step_ps,
                    int(start_frame),
                    float(requested_duration),
                )
                if target_frame is None or int(target_frame) in seen_targets:
                    continue
                seen_targets.add(int(target_frame))
                cases.append(
                    {
                        "record_index": int(rec_idx),
                        "choice_index": int(choice_idx),
                        "start_frame": int(start_frame),
                        "target_frame": int(target_frame),
                        "requested_duration_ns": float(requested_duration),
                    }
                )
    if args.durations_ns is not None and args.max_cases_per_duration > 0:
        balanced: list[dict[str, Any]] = []
        for duration_index, requested_duration in enumerate(args.durations_ns):
            selected = [
                case
                for case in cases
                if case["requested_duration_ns"] == float(requested_duration)
            ]
            cap = min(int(args.max_cases_per_duration), len(selected))
            if cap < len(selected):
                rng = np.random.default_rng(args.seed + 65_537 + 7_919 * duration_index)
                indices = np.sort(rng.choice(len(selected), size=cap, replace=False))
                selected = [selected[int(index)] for index in indices]
            balanced.extend(selected)
        cases = balanced
    if args.max_cases > 0:
        max_cases = min(int(args.max_cases), len(cases))
        if args.case_selection == "random" and max_cases < len(cases):
            rng = np.random.default_rng(args.seed + 65_537)
            selected = np.sort(rng.choice(len(cases), size=max_cases, replace=False))
            cases = [cases[int(index)] for index in selected]
        else:
            cases = cases[:max_cases]
    rows: list[dict[str, Any]] = []
    elapsed_forward = 0.0
    predicted_samples = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch_start in range(0, len(cases), args.batch_size):
        chunk = cases[batch_start : batch_start + args.batch_size]
        samples = [
            dataset._build_sample(
                int(case["record_index"]),
                int(case["choice_index"]),
                np.random.default_rng(args.seed + 104729 * int(case["record_index"])),
                frame_init=int(case["start_frame"]),
                apply_augment=False,
                frame_target_override=case["target_frame"],
            )
            for case in chunk
        ]
        batch = _collate(samples, device)
        cond = collate_fixed_time_conditions(batch)
        mask = batch.get("defect_field")
        # Persistence is evaluated on the exact same transitions as every
        # learned model.  Keep it in each row so downstream paired bootstrap
        # code never has to reconstruct or re-sample the held-out cases.
        persistence_values = _metrics(
            batch["m_init"], batch["m_t"], mask, cond["b_t"], boundary
        )
        for draw in range(effective_draws):
            torch.manual_seed(args.seed + 10_000_019 * draw + 1009 * batch_start)
            torch.cuda.manual_seed_all(args.seed + 10_000_019 * draw + 1009 * batch_start)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            pred, _ = sampler.sample(model, batch["m_init"], cond)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_forward += time.perf_counter() - started
            predicted_samples += len(chunk)
            values = _metrics(pred, batch["m_t"], mask, cond["b_t"], boundary)
            for i, (case, sample) in enumerate(zip(chunk, samples, strict=True)):
                run_id = str(sample["run_id"])
                row: dict[str, Any] = {
                    "label": args.label,
                    "draw": draw,
                    "record_index": int(case["record_index"]),
                    "base_id": _base_id(run_id),
                    "run_id": run_id,
                    "control_segment_index": int(sample["control_segment_index"]),
                    "start_time_ns": float(sample["frame_init_time_ns"]),
                    "end_time_ns": float(sample["frame_target_time_ns"]),
                    "duration_ns": float(sample["t_end_ns"]),
                    "requested_duration_ns": (
                        ""
                        if case["requested_duration_ns"] is None
                        else float(case["requested_duration_ns"])
                    ),
                    "zeeman_reference_applied": int(
                        bool(sample.get("zeeman_reference_applied", False))
                    ),
                }
                for name, tensor in values.items():
                    row[name] = float(tensor[i].detach().float().cpu())
                for name, tensor in persistence_values.items():
                    row[f"persistence_{name}"] = float(
                        tensor[i].detach().float().cpu()
                    )
                rows.append(row)
        print(json.dumps({"completed_cases": min(batch_start + len(chunk), len(cases)), "total_cases": len(cases)}), flush=True)

    _atomic_csv(args.output, rows)
    metadata = {
        "status": "complete",
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "checkpoint_prior_type": checkpoint_prior_type,
        "checkpoint_zeeman_precondition_enabled": checkpoint_zeeman_enabled,
        "checkpoint_mask_target_time": checkpoint_mask_target_time,
        "evaluation_zeeman_precondition_enabled": bool(
            cfg.get("data", {}).get("zeeman_precondition", {}).get("enabled", False)
        ),
        "checkpoint_load": load_report,
        "model_parameters": parameters,
        "state": args.state,
        "split": args.split,
        "evaluation_data_config": (
            str(args.data_config) if args.data_config is not None else None
        ),
        "checkpoint_stats_cache": str(checkpoint_stats_cache),
        "effective_stats_cache": str(cfg["data"]["stats_cache"]),
        "cases": len(cases),
        "requested_durations_ns": args.durations_ns,
        "max_cases_per_duration": args.max_cases_per_duration,
        "case_selection": args.case_selection,
        "requested_draws": args.draws,
        "effective_draws": effective_draws,
        "deterministic_source": deterministic_source,
        "rows": len(rows),
        "ode_steps": args.ode_steps,
        "method": sampler.method,
        "latency_seconds_per_transition": elapsed_forward / max(1, predicted_samples),
        "timed_transitions": predicted_samples,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "zeeman_gate_fraction": float(np.mean([row["zeeman_reference_applied"] for row in rows])),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
