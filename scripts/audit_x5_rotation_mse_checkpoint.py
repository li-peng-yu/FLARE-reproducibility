#!/usr/bin/env python3
"""Fail loudly unless a checkpoint is the exact rotation-target MSE control."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import torch


def _value(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def _without(mapping: dict[str, Any], *keys: str) -> dict[str, Any]:
    value = deepcopy(mapping)
    for key in keys:
        value.pop(key, None)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument(
        "--expected-reference-seed",
        type=int,
        help=(
            "Expected seed of the matched CFM reference checkpoint. Defaults to "
            "--expected-seed; use a known reference seed when auditing additional "
            "Direct U-Net replicates."
        ),
    )
    args = parser.parse_args()
    expected_reference_seed = (
        args.expected_seed
        if args.expected_reference_seed is None
        else args.expected_reference_seed
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    cfg = checkpoint["config"]
    reference_checkpoint = torch.load(
        args.reference_checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    reference = reference_checkpoint["config"]
    expected = {
        "seed": args.expected_seed,
        "model.arch": "unet",
        "bridge.state_repr": "alpha",
        "bridge.source_mode": "identity",
        "bridge.objective": "residual",
        "train.steps": 50_000,
        "train.loss.cfm_weight": 1.0,
        "train.loss.unit_weight": 0.0,
        "train.loss.llg_weight": 0.0,
        "train.loss.topo_weight": 0.0,
        "train.loss.alpha_mix": 0.0,
        "train.loss.tau_sampling": "zero",
        "train.loss.weight_mode": "uniform",
        "train.loss.change_aware_reweight.enabled": False,
        "train.loss.minibatch_ot.enabled": False,
        "sampler.ode_steps": 1,
        "sampler.method": "euler",
    }
    observed: dict[str, Any] = {}
    for dotted, wanted in expected.items():
        keys = dotted.split(".")
        got = _value(cfg, *keys)
        observed[dotted] = got
        if got != wanted:
            raise RuntimeError(f"{dotted}: expected {wanted!r}, got {got!r}")

    reference_expected = {
        "seed": expected_reference_seed,
        "model.arch": "unet",
        "bridge.state_repr": "alpha",
        "bridge.source_mode": "latent",
        "bridge.objective": "residual",
        "train.loss.tau_sampling": "uniform",
        "sampler.ode_steps": 10,
        "sampler.method": "heun",
    }
    for dotted, wanted in reference_expected.items():
        got = _value(reference, *dotted.split("."))
        if got != wanted:
            raise RuntimeError(
                f"reference {dotted}: expected {wanted!r}, got {got!r}"
            )

    matched_sections = ("data", "model", "prior", "performance")
    for section in matched_sections:
        if cfg.get(section) != reference.get(section):
            raise RuntimeError(f"control/reference {section} configs are not identical")
    if _without(cfg["bridge"], "source_mode") != _without(
        reference["bridge"], "source_mode"
    ):
        raise RuntimeError("bridge differs beyond the intended source_mode change")
    if _without(cfg["train"]["loss"], "tau_sampling", "unit_weight") != _without(
        reference["train"]["loss"], "tau_sampling", "unit_weight"
    ):
        raise RuntimeError("loss differs beyond direct-MSE tau/unit settings")
    if _without(cfg["train"], "loss", "visual_out_dir") != _without(
        reference["train"], "loss", "visual_out_dir"
    ):
        raise RuntimeError("training recipe or optimizer differs from the matched core")
    if int(checkpoint.get("step", -1)) != 50_000:
        raise RuntimeError(f"control checkpoint is not at 50k: {checkpoint.get('step')}")
    if int(reference_checkpoint.get("step", -1)) != 50_000:
        raise RuntimeError(
            f"reference checkpoint is not at 50k: {reference_checkpoint.get('step')}"
        )

    print(
        json.dumps(
            {
                "status": "exact_rotation_target_mse_control",
                "checkpoint": str(args.checkpoint.resolve()),
                "matched_reference_checkpoint": str(
                    args.reference_checkpoint.resolve()
                ),
                "checkpoint_step": int(checkpoint.get("step", -1)),
                "matched_sections": list(matched_sections),
                "checks": observed,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
