#!/usr/bin/env python3
"""Direct/composed semigroup and interpolated-handoff tests for x5 checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skyrmion_cfm.config import release_checkpoint_config
from skyrmion_cfm.config import load_config, seed_everything
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


def _set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _slerp(a: torch.Tensor, b: torch.Tensor, weight: float) -> torch.Tensor:
    if weight <= 0.0:
        return a
    if weight >= 1.0:
        return b
    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)
    dot = (a * b).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    linear = torch.nn.functional.normalize((1.0 - weight) * a + weight * b, dim=1)
    curved = (
        torch.sin((1.0 - weight) * theta) / sin_theta.clamp_min(1.0e-7) * a
        + torch.sin(weight * theta) / sin_theta.clamp_min(1.0e-7) * b
    )
    return torch.where(sin_theta.abs() < 1.0e-6, linear, curved)


def _metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    b_t: torch.Tensor,
    boundary: str,
) -> dict[str, torch.Tensor]:
    q_pred = topological_charge(pred, boundary=boundary)
    q_target = topological_charge(target, boundary=boundary)
    return {
        "ang": angular_error_deg(pred, target, mask=mask),
        "mse": mse_m(pred, target, mask=mask),
        "q_abs": (q_pred - q_target).abs(),
        "q_fail": (torch.round(q_pred) != torch.round(q_target)).float(),
        "energy_abs": (
            energy_density(pred, b_t, boundary=boundary, mask=mask)
            - energy_density(target, b_t, boundary=boundary, mask=mask)
        ).abs(),
    }


def _append_metrics(row: dict[str, Any], prefix: str, values: dict[str, torch.Tensor], i: int) -> None:
    for name, tensor in values.items():
        row[f"{prefix}_{name}"] = float(tensor[i].detach().float().cpu())


def _condition_max_abs_delta(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    *,
    ignored: tuple[str, ...] = (
        "t_end_index",
        "t_end_ns",
        "t_end_s",
        "dt_index",
        "dt_s",
        "drive_time_s",
        "relax_time_s",
        "drive_fraction",
        # These encode where the queried interval lies inside the segment,
        # rather than a change of the physical control itself.
        "anchor_offset_norm",
        "target_offset_norm",
    ),
) -> float:
    """Largest scale-normalized physical-control change after temporal metadata.

    The x5 protocols are piecewise time homogeneous, but this assertion keeps
    the semigroup claim honest if a future dataset adds a time-varying field,
    temperature ramp, or material schedule inside a nominal control segment.
    """
    maximum = 0.0
    keys = (set(left) & set(right)) - set(ignored)
    for key in keys:
        a, b = left[key], right[key]
        if not torch.is_tensor(a) or not torch.is_tensor(b) or a.shape != b.shape:
            continue
        if not (a.is_floating_point() or a.is_complex()):
            delta = float((a != b).to(torch.float32).max().cpu())
        else:
            a_float, b_float = a.float(), b.float()
            scale = torch.maximum(a_float.abs(), b_float.abs()).clamp_min(1.0)
            delta = float(((a_float - b_float).abs() / scale).max().cpu())
        maximum = max(maximum, delta)
    return maximum


def _ensemble_energy_distance(
    direct: list[torch.Tensor],
    composed: list[torch.Tensor],
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Biased energy distance using masked mean angular field distance."""

    def distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return angular_error_deg(left, right, mask=mask)

    cross = torch.stack([distance(x, y) for x in direct for y in composed]).mean(0)
    within_direct = torch.stack([distance(x, y) for x in direct for y in direct]).mean(0)
    within_composed = torch.stack([distance(x, y) for x in composed for y in composed]).mean(0)
    return (2.0 * cross - within_direct - within_composed).clamp_min(0.0)


def _same_condition_specs(dataset: Any, rec_idx: int) -> list[dict[str, int | float]]:
    record = dataset.records[rec_idx]
    save_step_ps = float(record.params.get("table_dt_s", 2.5e-10)) * 1.0e12
    out: list[dict[str, int | float]] = []
    for segment_index, (segment_start_ns, segment_end_ns) in enumerate(
        dataset._control_segments_for(rec_idx, record, save_step_ps)
    ):
        if not dataset._segment_in_time_range(segment_start_ns, segment_end_ns):
            continue
        start = dataset._frame_at_or_after(record, segment_start_ns, save_step_ps)
        end = dataset._frame_at_or_before(record, segment_end_ns, save_step_ps)
        if start is None or end is None or end - start < 2:
            continue
        # Every available internal split gives a valid direct-vs-composed test.
        for middle in range(int(start) + 1, int(end)):
            left_ns = dataset._duration_ns(record, int(start), middle, save_step_ps)
            right_ns = dataset._duration_ns(record, middle, int(end), save_step_ps)
            total_ns = dataset._duration_ns(record, int(start), int(end), save_step_ps)
            if min(left_ns, right_ns) < 0.20 or total_ns > 3.1:
                continue
            out.append(
                {
                    "segment_index": segment_index,
                    "start": int(start),
                    "middle": middle,
                    "end": int(end),
                    "left_ns": left_ns,
                    "right_ns": right_ns,
                    "total_ns": total_ns,
                }
            )
    return out


def _build_for_frames(dataset: Any, rec_idx: int, start: int, end: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    return dataset._build_visual_sample_for_frames(rec_idx, start, end, rng)


def _eligible_handoff(dataset: Any) -> list[tuple[int, list[tuple[int, int]]]]:
    out = []
    for rec_idx in dataset._valid_record_idx:
        specs = dataset._segment_boundary_specs(int(rec_idx))
        if len(specs) >= 2:
            out.append((int(rec_idx), specs))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state", choices=("ema", "raw"), default="ema")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ode-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--max-semigroup", type=int, default=512)
    parser.add_argument("--max-handoff", type=int, default=736)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--lambdas", type=float, nargs="+", default=(0.0, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument(
        "--data-config",
        type=Path,
        default=None,
        help=(
            "Optional evaluation-only config whose data section replaces the "
            "checkpoint data section. Checkpoint training statistics are retained."
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
    return parser


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SKYRMION_CFM_FUSED_OPS", "1")
    seed_everything(args.seed)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    cfg = release_checkpoint_config(checkpoint["config"])
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
    cfg.setdefault("data", {})["segment_time_range_ns"] = [1.0, 6.0]
    cfg["data"].setdefault("memmap", {})["auto_build"] = False
    cfg.setdefault("train", {})["compile"] = False
    _, _, dataset = build_fixed_time_datasets(cfg, build_splits={"test"})
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
    boundary = str(cfg["data"].get("boundary", "open"))
    deterministic_source = (
        str(cfg.get("bridge", {}).get("source_mode", "")).lower() == "identity"
    )
    effective_draws = 1 if deterministic_source else int(args.draws)

    # Semigroup: select deterministic record/split specs before batching.
    candidates: list[tuple[int, dict[str, int | float]]] = []
    for rec_idx in dataset._valid_record_idx:
        for spec in _same_condition_specs(dataset, int(rec_idx)):
            candidates.append((int(rec_idx), spec))
    if args.max_semigroup > 0:
        # Hash ordering is deterministic across checkpoints but samples the
        # whole held-out split instead of taking only the first trajectories.
        def candidate_digest(item: tuple[int, dict[str, int | float]]) -> str:
            rec_idx, spec = item
            run_id = str(dataset.records[rec_idx].run_id)
            token = (
                f"{args.seed}|{run_id}|{spec['start']}|{spec['middle']}|{spec['end']}"
            )
            return hashlib.sha256(token.encode("utf-8")).hexdigest()

        candidates = sorted(candidates, key=candidate_digest)[: args.max_semigroup]
    semigroup_rows: list[dict[str, Any]] = []
    semigroup_distribution_rows: list[dict[str, Any]] = []
    for batch_start in range(0, len(candidates), args.batch_size):
        chunk = candidates[batch_start : batch_start + args.batch_size]
        first_samples = [
            _build_for_frames(dataset, rec_idx, int(spec["start"]), int(spec["middle"]), args.seed + 11 * rec_idx)
            for rec_idx, spec in chunk
        ]
        second_samples = [
            _build_for_frames(dataset, rec_idx, int(spec["middle"]), int(spec["end"]), args.seed + 13 * rec_idx)
            for rec_idx, spec in chunk
        ]
        direct_samples = [
            _build_for_frames(dataset, rec_idx, int(spec["start"]), int(spec["end"]), args.seed + 17 * rec_idx)
            for rec_idx, spec in chunk
        ]
        b1, b2, bd = (_collate(values, device) for values in (first_samples, second_samples, direct_samples))
        c1, c2, cd = (collate_fixed_time_conditions(values) for values in (b1, b2, bd))
        control_delta_12 = _condition_max_abs_delta(c1, c2)
        control_delta_1d = _condition_max_abs_delta(c1, cd)
        control_delta_2d = _condition_max_abs_delta(c2, cd)
        control_delta = max(control_delta_12, control_delta_1d, control_delta_2d)
        if control_delta > 1.0e-6:
            raise RuntimeError(
                "semigroup candidate is not time homogeneous after excluding duration "
                f"conditions: max absolute delta={control_delta:.6g}"
            )
        init = b1["m_init"]
        target = bd["m_t"]
        mask = bd.get("defect_field")
        direct_draws: list[torch.Tensor] = []
        composed_draws: list[torch.Tensor] = []
        direct_q: list[torch.Tensor] = []
        composed_q: list[torch.Tensor] = []
        direct_energy: list[torch.Tensor] = []
        composed_energy: list[torch.Tensor] = []
        for draw in range(effective_draws):
            seed_value = args.seed + 10_000_019 * draw + 1009 * batch_start
            _set_seed(seed_value)
            first_pred, _ = sampler.sample(model, init, c1)
            _set_seed(seed_value + 1)
            composed_pred, _ = sampler.sample(model, first_pred, c2)
            _set_seed(seed_value)
            direct_pred, _ = sampler.sample(model, init, cd)
            direct_draws.append(direct_pred)
            composed_draws.append(composed_pred)
            direct_q.append(topological_charge(direct_pred, boundary=boundary))
            composed_q.append(topological_charge(composed_pred, boundary=boundary))
            direct_energy.append(energy_density(direct_pred, cd["b_t"], boundary=boundary, mask=mask))
            composed_energy.append(energy_density(composed_pred, cd["b_t"], boundary=boundary, mask=mask))
            direct_metric = _metric(direct_pred, target, mask=mask, b_t=cd["b_t"], boundary=boundary)
            composed_metric = _metric(composed_pred, target, mask=mask, b_t=cd["b_t"], boundary=boundary)
            defect_metric = _metric(composed_pred, direct_pred, mask=mask, b_t=cd["b_t"], boundary=boundary)
            for i, ((rec_idx, spec), sample) in enumerate(zip(chunk, direct_samples, strict=True)):
                row: dict[str, Any] = {
                    "label": args.label,
                    "draw": draw,
                    "record_index": rec_idx,
                    "run_id": str(sample["run_id"]),
                    **spec,
                    "sampler_seed": seed_value,
                    "non_duration_condition_max_relative_delta": control_delta,
                }
                _append_metrics(row, "direct", direct_metric, i)
                _append_metrics(row, "composed", composed_metric, i)
                _append_metrics(row, "paired_sg_defect", defect_metric, i)
                semigroup_rows.append(row)

        mean_direct = F.normalize(torch.stack(direct_draws).mean(0), dim=1)
        mean_composed = F.normalize(torch.stack(composed_draws).mean(0), dim=1)
        mean_defect = _metric(
            mean_composed,
            mean_direct,
            mask=mask,
            b_t=cd["b_t"],
            boundary=boundary,
        )
        energy_distance = _ensemble_energy_distance(direct_draws, composed_draws, mask)
        q_w1 = (
            torch.stack(direct_q).sort(dim=0).values
            - torch.stack(composed_q).sort(dim=0).values
        ).abs().mean(0)
        energy_w1 = (
            torch.stack(direct_energy).sort(dim=0).values
            - torch.stack(composed_energy).sort(dim=0).values
        ).abs().mean(0)
        for i, ((rec_idx, spec), sample) in enumerate(zip(chunk, direct_samples, strict=True)):
            row = {
                "label": args.label,
                "record_index": rec_idx,
                "run_id": str(sample["run_id"]),
                **spec,
                "draws": effective_draws,
                "non_duration_condition_max_relative_delta": control_delta,
                "ensemble_energy_distance_ang": float(energy_distance[i].cpu()),
                "q_w1": float(q_w1[i].cpu()),
                "energy_w1": float(energy_w1[i].cpu()),
            }
            _append_metrics(row, "mean_sg_defect", mean_defect, i)
            semigroup_distribution_rows.append(row)
        print(
            json.dumps(
                {
                    "phase": "semigroup",
                    "completed_cases": len(semigroup_distribution_rows),
                    "total_cases": len(candidates),
                }
            ),
            flush=True,
        )

    # Handoff robustness: interpolate the second segment's true and predicted starts.
    handoff = _eligible_handoff(dataset)
    if args.max_handoff > 0:
        handoff = handoff[: args.max_handoff]
    handoff_rows: list[dict[str, Any]] = []
    for batch_start in range(0, len(handoff), args.batch_size):
        chunk = handoff[batch_start : batch_start + args.batch_size]
        first_samples = []
        second_samples = []
        for rec_idx, specs in chunk:
            c0, s0 = specs[0]
            c1, s1 = specs[1]
            first_samples.append(dataset._build_sample(rec_idx, c0, np.random.default_rng(args.seed + rec_idx), frame_init=s0, apply_augment=False))
            second_samples.append(dataset._build_sample(rec_idx, c1, np.random.default_rng(args.seed + 7 + rec_idx), frame_init=s1, apply_augment=False))
        b1, b2 = (_collate(values, device) for values in (first_samples, second_samples))
        c1, c2 = (collate_fixed_time_conditions(values) for values in (b1, b2))
        true_boundary = b2["m_init"]
        target = b2["m_t"]
        mask = b2.get("defect_field")
        for draw in range(effective_draws):
            seed_value = args.seed + 1_000_003 + 10_000_019 * draw + 1009 * batch_start
            _set_seed(seed_value)
            predicted_boundary, _ = sampler.sample(model, b1["m_init"], c1)
            for weight in args.lambdas:
                interpolated = _slerp(true_boundary, predicted_boundary, float(weight))
                _set_seed(seed_value + 1)
                pred, _ = sampler.sample(model, interpolated, c2)
                metric = _metric(pred, target, mask=mask, b_t=c2["b_t"], boundary=boundary)
                input_metric = _metric(interpolated, true_boundary, mask=mask, b_t=c2["b_t"], boundary=boundary)
                for i, ((rec_idx, _specs), sample) in enumerate(zip(chunk, second_samples, strict=True)):
                    row = {
                        "label": args.label,
                        "draw": draw,
                        "record_index": rec_idx,
                        "run_id": str(sample["run_id"]),
                        "lambda": float(weight),
                        "sampler_seed": seed_value,
                    }
                    _append_metrics(row, "output", metric, i)
                    _append_metrics(row, "input", input_metric, i)
                    handoff_rows.append(row)
        print(json.dumps({"phase": "handoff", "completed_trajectories": min(batch_start + len(chunk), len(handoff)), "total": len(handoff)}), flush=True)

    _atomic_csv(args.output_dir / "semigroup.csv", semigroup_rows)
    _atomic_csv(args.output_dir / "semigroup_distribution.csv", semigroup_distribution_rows)
    _atomic_csv(args.output_dir / "handoff_curve.csv", handoff_rows)
    metadata = {
        "status": "complete",
        "label": args.label,
        "checkpoint": str(args.checkpoint),
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
        "evaluation_data_config": (
            str(args.data_config) if args.data_config is not None else None
        ),
        "checkpoint_stats_cache": str(checkpoint_stats_cache),
        "effective_stats_cache": str(cfg["data"]["stats_cache"]),
        "ode_steps": args.ode_steps,
        "requested_draws": args.draws,
        "effective_draws": effective_draws,
        "deterministic_source": deterministic_source,
        "semigroup_rows": len(semigroup_rows),
        "semigroup_distribution_cases": len(semigroup_distribution_rows),
        "handoff_trajectories": len(handoff),
        "handoff_rows": len(handoff_rows),
        "lambdas": args.lambdas,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
