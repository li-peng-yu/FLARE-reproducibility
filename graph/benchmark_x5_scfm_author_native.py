#!/usr/bin/env python3
"""Time complete SCFM ODE execution at a requested physical horizon."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import default_collate

from graph.benchmark_cost_plot import _autocast_context, _configure_attention_backends
from graph.benchmark_x5_scfm_quality import atomic_json, load_scfm
from graph.x5_author_native_rollout_dataset import (
    X5AuthorNativeRolloutCases,
    first_legal_segment_sample,
)
from skyrmion_cfm.config import seed_everything
from skyrmion_cfm.data.fixed_time import collate_fixed_time_conditions
from skyrmion_cfm.train import move_batch
from scripts.evaluate_skx_x5_same_condition_distribution import (
    DEFAULT_REPEAT_DATASET,
    _complete_test_groups,
    _override_multisegment_timing_durations,
    _prepare_multisegment_rollout_condition,
)


STEP_NS = 0.25


def _retarget_timing_sample(
    sample: dict[str, Any],
    *,
    horizon_ns: float,
    t_end_index: int,
) -> dict[str, Any]:
    """Set a timing-only endpoint condition beyond the stored target frames."""
    drive_time_s = float(sample["drive_time_s"])
    horizon_s = float(horizon_ns) * 1.0e-9
    sample["t_end_ns"] = torch.tensor(horizon_ns, dtype=torch.float32)
    sample["t_end_s"] = torch.tensor(horizon_s, dtype=torch.float32)
    sample["t_end_index"] = torch.tensor(t_end_index, dtype=torch.long)
    sample["frame_target_time_ns"] = sample["frame_init_time_ns"].float() + horizon_ns
    sample["relax_time_s"] = torch.tensor(
        max(0.0, horizon_s - drive_time_s),
        dtype=torch.float32,
    )
    sample["drive_fraction"] = torch.tensor(
        drive_time_s / max(horizon_s, 1.0e-15),
        dtype=torch.float32,
    )
    return sample


def _ranges(size: int, batch_size: int) -> Iterable[range]:
    for start in range(0, size, batch_size):
        yield range(start, min(start + batch_size, size))


def _prepare_batches(
    samples: list[dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
    channels_last: bool,
) -> list[tuple[torch.Tensor, Any]]:
    batches: list[tuple[torch.Tensor, Any]] = []
    for indices in _ranges(len(samples), batch_size):
        batch = move_batch(default_collate([samples[index] for index in indices]), device)
        m_init = batch["m_init"]
        if channels_last:
            m_init = m_init.contiguous(memory_format=torch.channels_last)
        conditions = collate_fixed_time_conditions(batch)
        batches.append((m_init, conditions))
    return batches


def _first_multisegment_path(
    dataset: Any,
    seed: int,
    condition_dir: Path,
    timing_segment_durations_ns: list[float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = _complete_test_groups(
        dataset,
        repeat_dataset=DEFAULT_REPEAT_DATASET,
        repeats_per_group=5,
    )
    base, record_indices = next(iter(groups.items()))
    prepared = _prepare_multisegment_rollout_condition(
        dataset,
        base=base,
        record_indices=record_indices,
        condition_dir=condition_dir,
    )
    if timing_segment_durations_ns is not None:
        prepared = _override_multisegment_timing_durations(
            prepared, dataset, timing_segment_durations_ns
        )
    path = list(prepared["segment_samples"][0])
    if len(path) < 2:
        raise RuntimeError("paper timing condition has fewer than two control segments")
    repeat = prepared["metadata"]["repeats"][0]
    return path, {
        "base_id": base,
        "run_id": str(path[0]["run_id"]),
        "control_segment_indices": prepared["metadata"][
            "control_segment_indices"
        ],
        "segments": repeat["segments"],
        "handoff_count": prepared["metadata"]["handoff_count"],
        "absolute_span_ns": repeat["absolute_span_ns"],
        "composed_model_horizon_ns": repeat["composed_model_horizon_ns"],
        "handoffs": repeat["handoffs"],
        "timing_segment_durations_ns": prepared["metadata"].get(
            "timing_segment_durations_ns"
        ),
        "timing_role": (
            "standardized fixed-duration exact-control-segment rollout"
            if timing_segment_durations_ns is not None
            else "complete exact-control-segment rollout"
        ),
    }


def _sample_multisegment(
    model: torch.nn.Module,
    sampler: Any,
    batches: list[tuple[torch.Tensor, Any]],
    *,
    device: torch.device,
    precision: str,
) -> torch.Tensor:
    prediction: torch.Tensor | None = None
    for model_input, conditions in batches:
        current_input = model_input if prediction is None else prediction
        with _autocast_context(device, precision):
            prediction, _ = sampler.sample(model, current_input, conditions)
    if prediction is None:
        raise RuntimeError("FLARE multisegment timing produced no prediction")
    return prediction


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SCFM native benchmark requires CUDA")
    _configure_attention_backends(args.precision)
    cfg, test_ds, model, sampler, device = load_scfm(
        args.config,
        args.checkpoint,
        args.channels_last,
        args.override_config,
    )
    sampler.ode_steps = args.ode_steps
    horizon_ns = float(args.horizon_ns)
    if args.timing_segment_durations_ns is not None and not args.multisegment_rollout:
        raise ValueError(
            "--timing-segment-durations-ns requires --multisegment-rollout"
        )
    if (not args.multisegment_rollout) and horizon_ns <= 0.0:
        raise ValueError("horizon_ns must be positive")
    horizon_step = int(round(horizon_ns / STEP_NS))
    if (not args.multisegment_rollout) and abs(
        horizon_step * STEP_NS - horizon_ns
    ) > 1.0e-9:
        raise ValueError(f"horizon_ns must be a multiple of {STEP_NS:g} ns")
    multisegment_path: list[dict[str, Any]] | None = None
    if args.multisegment_rollout:
        if args.matched_segment_index is not None:
            raise ValueError(
                "--multisegment-rollout and --matched-segment-index are mutually exclusive"
            )
        if not args.repeat_single_case:
            raise ValueError(
                "multisegment timing currently requires --repeat-single-case"
            )
        multisegment_path, selection = _first_multisegment_path(
            test_ds,
            args.seed,
            args.output.parent / "timing_condition",
            args.timing_segment_durations_ns,
        )
        samples = [multisegment_path[0]]
        horizon_ns = float(selection["composed_model_horizon_ns"])
    elif args.matched_segment_index is not None:
        # Use an admissible post-relaxation
        # query rather than the old frame(0)->frame(5 ns) cross-control view.
        matched_sample, matched_metadata = first_legal_segment_sample(
            test_ds,
            segment_index=int(args.matched_segment_index),
            horizon_ns=horizon_ns,
            seed=args.seed,
            preferred_run_id=args.matched_run_id,
        )
        samples = [matched_sample] * args.samples
        selection = {
            **matched_metadata,
            "samples": args.samples,
            "timing_role": "matched legal single-segment query",
        }
    else:
        cases = X5AuthorNativeRolloutCases(test_ds, limit=args.samples)
        if horizon_ns <= cases.max_horizon_ns:
            samples = [
                cases.direct_sample(index, horizon_step) for index in range(len(cases))
            ]
            selection = cases.selection_metadata()
        else:
            samples = [
                _retarget_timing_sample(
                    cases.direct_sample(index, cases.num_steps),
                    horizon_ns=horizon_ns,
                    t_end_index=test_ds._t_end_bucket_index(horizon_ns),
                )
                for index in range(len(cases))
            ]
            selection = cases.selection_metadata()
            selection.update(
                {
                    "timing_condition_horizon_ns": horizon_ns,
                    "stored_target_horizon_ns": cases.max_horizon_ns,
                    "has_stored_target_frame": False,
                    "purpose": "timing-only target-time query; no 8 ns accuracy claim",
                }
            )

    results_by_batch: dict[str, Any] = {}
    for batch_size in args.batch_sizes:
        if multisegment_path is not None:
            segment_batches = [
                _prepare_batches(
                    [sample] * batch_size,
                    batch_size=batch_size,
                    device=device,
                    channels_last=args.channels_last,
                )[0]
                for sample in multisegment_path
            ]
            _sample_multisegment(
                model,
                sampler,
                segment_batches,
                device=device,
                precision=args.precision,
            )
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            runs: list[float] = []
            for repeat in range(args.repeats):
                seed_everything(args.seed + 10_000 + repeat)
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                _sample_multisegment(
                    model,
                    sampler,
                    segment_batches,
                    device=device,
                    precision=args.precision,
                )
                torch.cuda.synchronize(device)
                runs.append(
                    1000.0 * (time.perf_counter() - started) / batch_size
                )
            results_by_batch[str(batch_size)] = {
                "mean_ms_per_sample": statistics.fmean(runs),
                "std_ms_per_sample": (
                    statistics.stdev(runs) if len(runs) > 1 else 0.0
                ),
                "runs_ms_per_sample": runs,
                "peak_gpu_memory_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
            }
            del segment_batches
            torch.cuda.empty_cache()
            continue
        timing_samples = (
            [samples[0]] * batch_size
            if args.repeat_single_case
            else samples
        )
        batches = _prepare_batches(
            timing_samples,
            batch_size=batch_size,
            device=device,
            channels_last=args.channels_last,
        )
        warm_m, warm_conditions = batches[0]
        with _autocast_context(device, args.precision):
            sampler.sample(model, warm_m, warm_conditions)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        runs: list[float] = []
        for repeat in range(args.repeats):
            seed_everything(args.seed + 10_000 + repeat)
            elapsed = 0.0
            count = 0
            for m_init, conditions in batches:
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                with _autocast_context(device, args.precision):
                    sampler.sample(model, m_init, conditions)
                torch.cuda.synchronize(device)
                elapsed += time.perf_counter() - started
                count += m_init.shape[0]
            runs.append(1000.0 * elapsed / count)
        results_by_batch[str(batch_size)] = {
            "mean_ms_per_sample": statistics.fmean(runs),
            "std_ms_per_sample": statistics.stdev(runs) if len(runs) > 1 else 0.0,
            "runs_ms_per_sample": runs,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
        del batches
        torch.cuda.empty_cache()

    method = str(cfg["sampler"].get("method", "heun"))
    evaluations_per_step = 2 if method == "heun" else 1
    guidance = cfg.get("sampler", {}).get("classifier_free_guidance") or {}
    guidance_multiplier = 2 if bool(guidance.get("enabled", False)) else 1
    # This composed-path timing is
    # diagnostic; paper speed keeps the common representative 5-ns workload.
    payload = {
        "schema": "x5_scfm_complete_horizon_timing_v1",
        "status": "complete",
        "method": "scfm_stage1",
        "label": f"SCFM Stage 1 ODE-{args.ode_steps}",
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "override_config": (
            str(args.override_config) if args.override_config is not None else None
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "precision": args.precision,
        "device": torch.cuda.get_device_name(device),
        "samples": args.samples,
        "repeats": args.repeats,
        "horizon_ns": horizon_ns,
        "multisegment_rollout": args.multisegment_rollout,
        "reference_mode": (
            "exact_control_multisegment_rollout_endpoint"
            if args.multisegment_rollout
            else "timing_horizon_query"
        ),
        "selection": selection,
        "algorithm": {
            "native_temporal_protocol": "target-time conditional flow sampling",
            "sampler": method,
            "ode_steps": args.ode_steps,
            "network_evaluations_per_endpoint": (
                args.ode_steps * evaluations_per_step * guidance_multiplier
            ),
            "segment_count": (
                len(multisegment_path) if multisegment_path is not None else 1
            ),
            "handoff_count": (
                len(multisegment_path) - 1
                if multisegment_path is not None
                else 0
            ),
            "network_evaluations_per_complete_output": (
                args.ode_steps
                * evaluations_per_step
                * guidance_multiplier
                * (len(multisegment_path) if multisegment_path is not None else 1)
            ),
            "rollout_feedback": args.multisegment_rollout,
            "stochastic": True,
        },
        "results_by_batch_size": results_by_batch,
        "batch_scaling_input": (
            "one fixed held-out condition repeated within each batch"
            if args.repeat_single_case
            else "held-out conditions batched without repetition"
        ),
        "timing_scope": (
            (
                "complete FLARE sampling over the exact protocol control "
                "segments with prediction handoff; "
                if args.multisegment_rollout
                else f"complete {horizon_ns:g} ns ODE sampler and output projection; "
            )
            + "excludes data loading, host-to-device transfer, and condition construction"
        ),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--override-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon-ns", type=float, default=5.0)
    parser.add_argument(
        "--multisegment-rollout",
        action="store_true",
        help=(
            "Ignore --horizon-ns and time the exact-control drive->post-relax "
            "path with one FLARE call per physical control segment."
        ),
    )
    parser.add_argument(
        "--timing-segment-durations-ns",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Timing-only durations for --multisegment-rollout, for example "
            "2 3 for a standardized two-segment 5-ns workload."
        ),
    )
    parser.add_argument(
        "--matched-segment-index",
        type=int,
        default=None,
        help=(
            "Require an exact target inside this constant-control segment; use 2 "
            "for the matched 3-ns post-relaxation benchmark."
        ),
    )
    parser.add_argument(
        "--matched-run-id",
        default=None,
        help="Optionally require the exact held-out run used by the quality set.",
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--ode-steps", type=int, default=10)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument(
        "--repeat-single-case",
        action="store_true",
        help=(
            "Repeat one fixed held-out condition to form every requested batch. "
            "This permits powers-of-two throughput sweeps larger than the finite "
            "set of unique timing cases and matches the native-baseline benchmark."
        ),
    )
    parser.add_argument("--seed", type=int, default=78)
    return parser


if __name__ == "__main__":
    evaluate(make_parser().parse_args())
