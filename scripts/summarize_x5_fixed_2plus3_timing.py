#!/usr/bin/env python3
"""Summarize standardized 2+3-ns timing at paper batch sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUPPORTED_BATCHES = {
    "flare": (1, 2, 4, 8, 16, 32, 64, 128),
    "poseidon_t": (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048),
    "cno_fm": (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048),
    "dpot_ti": (1, 2, 4, 8, 16, 32, 64, 128, 256),
    "mpp_avit_ti": (1, 2, 4, 8, 16, 32, 64, 128),
    "pdearena_unet": (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
    "le_pde": (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048),
    "neuralmag_x5": (1, 2, 4, 8, 16, 32, 64, 128),
}
DISPLAY = {
    "flare": "FLARE ODE-10",
    "poseidon_t": "Poseidon-T",
    "cno_fm": "CNO-FM",
    "dpot_ti": "DPOT-Ti",
    "mpp_avit_ti": "MPP-AViT-Ti",
    "pdearena_unet": "PDEArena U-Net",
    "le_pde": "LE-PDE",
    "neuralmag_x5": "NeuralMAG-x5",
}


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(f"incomplete result: {path}")
    return payload


def _validate_common(payload: dict[str, Any], path: Path) -> None:
    if not payload.get("multisegment_rollout"):
        raise RuntimeError(f"not a multisegment timing result: {path}")
    if abs(float(payload.get("horizon_ns", -1.0)) - 5.0) > 1.0e-7:
        raise RuntimeError(f"not a total-5-ns timing result: {path}")
    if payload.get("reference_mode") != "exact_control_multisegment_rollout_endpoint":
        raise RuntimeError(f"wrong reference mode: {path}")


def _native_sweep(
    root: Path, method: str
) -> tuple[dict[int, float], dict[int, float], dict[str, Any]]:
    path = root / method / "sweep.json"
    payload = _json(path)
    _validate_common(payload, path)
    selection = payload.get("selection", {})
    durations = selection.get("timing_segment_durations_ns")
    if durations != [2.0, 3.0]:
        raise RuntimeError(f"wrong segment durations in {path}: {durations}")
    if int(selection.get("handoff_count", -1)) != 1:
        raise RuntimeError(f"wrong handoff count in {path}")
    drive_fractions = [
        float(segment["drive_fraction"])
        for segment in selection.get("segments", [])
    ]
    if drive_fractions != [1.0, 0.0]:
        raise RuntimeError(f"wrong drive/relax controls in {path}: {drive_fractions}")
    if method == "flare":
        algorithm = payload.get("algorithm", {})
        if (
            int(algorithm.get("segment_count", -1)) != 2
            or int(algorithm.get("network_evaluations_per_complete_output", -1))
            != 40
        ):
            raise RuntimeError(f"wrong FLARE segmented execution in {path}")
    else:
        expected_calls = 2 if method in {"poseidon_t", "cno_fm"} else 20
        if int(payload.get("network_calls", -1)) != expected_calls:
            raise RuntimeError(f"wrong model-call count in {path}")
    results = payload["results_by_batch_size"]
    expected = SUPPORTED_BATCHES[method]
    if set(results) != {str(batch) for batch in expected}:
        raise RuntimeError(f"incomplete batch sweep in {path}: {sorted(results)}")
    return (
        {
            batch: float(results[str(batch)]["mean_ms_per_sample"])
            for batch in expected
        },
        {
            batch: float(results[str(batch)]["peak_gpu_memory_bytes"]) / (1024**3)
            for batch in expected
        },
        payload,
    )


def _neuralmag_sweep(
    root: Path,
) -> tuple[dict[int, float], dict[int, float], dict[str, Any]]:
    payloads: dict[int, dict[str, Any]] = {}
    for batch in SUPPORTED_BATCHES["neuralmag_x5"]:
        path = root / "neuralmag_x5" / f"batch_{batch}.json"
        payload = _json(path)
        _validate_common(payload, path)
        if int(payload["batch_size"]) != batch:
            raise RuntimeError(f"wrong NeuralMAG batch in {path}")
        rollout = payload.get("rollout_metadata", [])
        if len(rollout) != 1:
            raise RuntimeError(f"unexpected NeuralMAG repeats in {path}")
        segment_horizons = [
            float(row["horizon_s"]) * 1.0e9
            for row in rollout[0].get("segments", [])
        ]
        if any(
            abs(left - right) > 1.0e-6
            for left, right in zip(segment_horizons, (2.0, 3.0))
        ) or len(segment_horizons) != 2:
            raise RuntimeError(
                f"wrong NeuralMAG segment durations in {path}: {segment_horizons}"
            )
        if int(rollout[0]["rk4_steps"]) != 50_000:
            raise RuntimeError(f"wrong NeuralMAG RK4 count in {path}")
        segment_steps = [
            int(row["rk4_steps"]) for row in rollout[0].get("segments", [])
        ]
        segment_demag_calls = [
            int(row["learned_demag_calls"])
            for row in rollout[0].get("segments", [])
        ]
        if segment_steps != [20_000, 30_000]:
            raise RuntimeError(f"wrong NeuralMAG segment steps in {path}")
        if segment_demag_calls != [80_000, 120_000]:
            raise RuntimeError(f"wrong NeuralMAG demag calls in {path}")
        payloads[batch] = payload
    return (
        {
            batch: float(payloads[batch]["mean_ms_per_sample"])
            for batch in SUPPORTED_BATCHES["neuralmag_x5"]
        },
        {
            batch: float(payloads[batch]["peak_gpu_memory_bytes"]) / (1024**3)
            for batch in SUPPORTED_BATCHES["neuralmag_x5"]
        },
        payloads[max(SUPPORTED_BATCHES["neuralmag_x5"])],
    )


def _fmt_latency(value: float) -> str:
    if value >= 100_000:
        return f"{value:,.0f}"
    if value >= 100:
        return f"{value:,.1f}"
    if value >= 10:
        return f"{value:,.2f}"
    if value >= 1:
        return f"{value:,.3f}"
    return f"{value:,.4f}"


def _fmt_speed(value: float) -> str:
    return f"{value:,.2f}" if value < 10.0 else f"{value:,.1f}"


def _fmt_memory(value: float) -> str:
    return f"{value:.3f}" if value < 1.0 else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mumax-ms", type=float, default=446_125.0,
                        help="5-ns MuMax reference in ms; paper: 361000 + (361000 - 190750) / 2.")
    args = parser.parse_args()
    if args.mumax_ms <= 0:
        parser.error("--mumax-ms must be positive")
    rows: list[dict[str, Any]] = []
    for method in SUPPORTED_BATCHES:
        if method == "neuralmag_x5":
            sweep, peak_gib, payload = _neuralmag_sweep(args.input_root)
        else:
            sweep, peak_gib, payload = _native_sweep(args.input_root, method)
        batch_one_ms = sweep[1]
        paper_batch, paper_batch_ms = min(sweep.items(), key=lambda item: item[1])
        parameter_count = int(payload["parameter_count"])
        if parameter_count <= 0 or any(value <= 0 for value in sweep.values()):
            raise RuntimeError(f"invalid parameter count or latency for {method}")
        # Retain the evaluator's field names for downstream summary compatibility.
        row = {
            "method": method,
            "display": DISPLAY[method],
            "parameter_count": parameter_count,
            "new_fastest_batch": paper_batch,
            "new_batch_one_ms": batch_one_ms,
            "new_batch_one_speedup": args.mumax_ms / batch_one_ms,
            "new_paper_batch_ms_per_output": paper_batch_ms,
            "new_paper_batch_speedup": args.mumax_ms / paper_batch_ms,
            "new_batch_one_peak_gpu_memory_gib": peak_gib[1],
            "new_paper_batch_peak_gpu_memory_gib": peak_gib[paper_batch],
            "source_schema": payload.get("schema"),
            "new_batch_sweep_ms_per_output": {str(batch): value for batch, value in sweep.items()},
            "paper_formatted": {
                "batch_one_latency_ms": _fmt_latency(batch_one_ms),
                "batch_one_speedup": _fmt_speed(args.mumax_ms / batch_one_ms),
                "fastest_batch": paper_batch,
                "fastest_batch_latency_ms_per_output": _fmt_latency(paper_batch_ms),
                "fastest_batch_speedup": _fmt_speed(args.mumax_ms / paper_batch_ms),
                "batch_one_peak_gpu_memory_gib": _fmt_memory(peak_gib[1]),
                "fastest_batch_peak_gpu_memory_gib": _fmt_memory(peak_gib[paper_batch]),
                "batch_sweep_ms_per_output": {str(batch): _fmt_latency(value) for batch, value in sweep.items()},
            },
        }
        rows.append(row)
    output = {
        "status": "complete",
        "protocol": {
            "total_horizon_ns": 5.0,
            "segment_durations_ns": [2.0, 3.0],
            "handoff_count": 1,
            "handoff_state": "previous model prediction",
            "mumax_reference_ms": args.mumax_ms,
            "mumax_reference_description": (
                "Paper reference: T_5ns = T_4ns + 0.5 * (T_4ns - T_2ns), "
                "with T_2ns=190750 ms and T_4ns=361000 ms, FixDt=1e-14 s. "
                "Override --mumax-ms to use a separately measured reference."
            ),
            "batch_policy": "full supported powers-of-two sweep; select minimum measured milliseconds per output",
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
