#!/usr/bin/env python3
"""Evaluate SCFM on the same deterministic x5 endpoint view as external baselines."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from graph.benchmark_cost_plot import (
    HORIZONS_NS,
    _autocast_context,
    _build_condition_audit,
    _clean_state_dict_keys,
    _configure_attention_backends,
    _relocate_checkpoint_paths,
)
from graph.x5_initial_horizon_dataset import X5InitialStateHorizonDataset
from skyrmion_cfm.cfm.sampler import BridgeSampler
from skyrmion_cfm.config import get_device, load_config, merge_config, seed_everything
from skyrmion_cfm.data.fixed_time import (
    build_fixed_time_datasets,
    collate_fixed_time_conditions,
)
from skyrmion_cfm.data.stats import load_training_stats
from skyrmion_cfm.models import build_model
from skyrmion_cfm.train import make_bridge_and_priors, move_batch


def normalize_magnetization(m: torch.Tensor) -> torch.Tensor:
    return m / m.square().sum(dim=1, keepdim=True).clamp_min(1.0e-12).sqrt()


def quality(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    """Match the metric definition used by x5_external_baseline.py exactly."""
    prediction = normalize_magnetization(prediction.float())
    target = normalize_magnetization(target.float())
    mse = F.mse_loss(prediction, target).item()
    dot = (prediction * target).sum(dim=1).clamp(-1.0, 1.0)
    angle_deg = torch.rad2deg(torch.acos(dot)).mean().item()
    return mse, angle_deg


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_scfm(
    config: Path,
    checkpoint: Path,
    channels_last: bool,
    override_config: Path | None = None,
) -> tuple[dict[str, Any], Any, torch.nn.Module, BridgeSampler, torch.device]:
    """Load saved training statistics instead of refitting them at evaluation."""
    cfg = load_config(config)
    if override_config is not None:
        cfg = merge_config(cfg, load_config(override_config))
    cfg = _relocate_checkpoint_paths(cfg)
    seed_everything(int(cfg.get("seed", 0)))
    device = get_device(cfg.get("device", "auto"))
    train_ds, _, test_ds = build_fixed_time_datasets(cfg)
    stats_path = Path(cfg["data"]["stats_cache"])
    if not stats_path.is_file():
        raise FileNotFoundError(f"missing saved training statistics: {stats_path}")
    stats = load_training_stats(stats_path)
    _build_condition_audit(cfg, train_ds)

    model = build_model(cfg, stats.condition).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    checkpoint_payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = checkpoint_payload.get("ema") or checkpoint_payload.get("model") or checkpoint_payload
    missing, unexpected = model.load_state_dict(_clean_state_dict_keys(state), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint load mismatch: missing={len(missing)} unexpected={len(unexpected)}"
        )
    model.eval()

    bridge, rotation_prior, cart_prior, rfm_prior = make_bridge_and_priors(cfg, stats)
    sampler = BridgeSampler(
        bridge=bridge,
        rotation_prior=rotation_prior,
        cart_prior=cart_prior,
        rfm_prior=rfm_prior,
        ode_steps=int(cfg["sampler"].get("ode_steps", 50)),
        method=str(cfg["sampler"].get("method", "heun")),
        classifier_free_guidance=cfg.get("sampler", {}).get("classifier_free_guidance"),
        stochastic_sampler=cfg.get("sampler", {}).get("stochastic_sampler"),
    )
    return cfg, test_ds, model, sampler, device


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("formal SCFM quality evaluation requires CUDA")
    _configure_attention_backends(args.precision)
    cfg, test_ds, model, sampler, device = load_scfm(
        args.config,
        args.checkpoint,
        args.channels_last,
    )
    sampler.ode_steps = args.ode_steps
    results: dict[str, Any] = {}
    selections: dict[str, Any] = {}

    for horizon in HORIZONS_NS:
        subset = X5InitialStateHorizonDataset(test_ds, horizon, args.samples_per_horizon)
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        mse_values: list[float] = []
        angle_values: list[float] = []
        for batch in loader:
            batch = move_batch(batch, device)
            m_init = batch["m_init"]
            if args.channels_last:
                m_init = m_init.contiguous(memory_format=torch.channels_last)
            conditions = collate_fixed_time_conditions(batch)
            with _autocast_context(device, args.precision):
                prediction, _ = sampler.sample(model, m_init, conditions)
            mse, angle = quality(prediction, batch["m_t"])
            mse_values.append(mse)
            angle_values.append(angle)

        key = f"{horizon:g}"
        results[key] = {
            "mse": statistics.fmean(mse_values),
            "mean_angle_deg": statistics.fmean(angle_values),
            "batches": len(mse_values),
            "samples": len(subset),
        }
        selections[key] = subset.selection_metadata()
        print({"horizon_ns": horizon, **results[key]}, flush=True)

    method = str(cfg["sampler"].get("method", "heun"))
    evaluations_per_step = 2 if method == "heun" else 1
    guidance = cfg.get("sampler", {}).get("classifier_free_guidance") or {}
    guidance_multiplier = 2 if bool(guidance.get("enabled", False)) else 1
    payload = {
        "schema": "x5_scfm_quality_v1",
        "method": "scfm_stage1",
        "label": "SCFM Stage 1",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "split": "test",
        "seed": int(cfg.get("seed", 0)),
        "precision": args.precision,
        "batch_size": args.batch_size,
        "samples_per_horizon": args.samples_per_horizon,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "sampler": {
            "method": method,
            "ode_steps": args.ode_steps,
            "network_calls_per_prediction": (
                args.ode_steps * evaluations_per_step * guidance_multiplier
            ),
            "one_seeded_draw_per_sample": True,
        },
        "evaluation_dataset_view": {
            "t_end_ns": list(HORIZONS_NS),
            "control_segmented": False,
            "current_time_mode": "t0",
            "meaning": "full pulse/relax trajectory from the x5 initial state",
        },
        "selection": selections,
        "quality": results,
    }
    atomic_json(args.output, payload)
    return payload


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-horizon", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ode-steps", type=int, default=10)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--channels-last", action="store_true")
    return parser


if __name__ == "__main__":
    arguments = make_parser().parse_args()
    print(json.dumps(evaluate(arguments), indent=2, sort_keys=True), flush=True)
