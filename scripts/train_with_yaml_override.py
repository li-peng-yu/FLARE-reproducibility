from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from skyrmion_cfm.config import load_config, merge_config
from skyrmion_cfm.train import run_training


def _compact_config_summary(cfg: dict) -> dict:
    """Return the small set of run-defining fields useful in run logs."""

    train = cfg.get("train", {}) or {}
    segment = train.get("segment_pushforward", {}) or {}
    data = cfg.get("data", {}) or {}
    roots = data.get("dataset_root", [])
    if isinstance(roots, (str, Path)):
        roots = [roots]
    return {
        "output_dir": cfg.get("output_dir"),
        "seed": cfg.get("seed"),
        "dataset_roots": [Path(str(root)).name for root in roots],
        "trajectory_index_cache": data.get("trajectory_index_cache"),
        "train_steps": int(train.get("steps", 0)),
        "global_batch": int(train.get("batch_size", 0)),
        "loader_workers_per_rank": int(train.get("num_workers", 0)),
        "resume_from": train.get("resume_from"),
        "segment_pushforward": {
            "enabled": bool(segment.get("enabled", False)),
            "predictions_per_segment": int(
                segment.get("predicted_inits_per_segment", 1)
            ),
            "candidates_per_segment": int(
                segment.get("candidate_inits_per_segment", 1)
            ),
            "sampling_mode": segment.get("sampling_mode"),
            "target_mode": segment.get("target_mode"),
            "handoff_cache": segment.get("cache_path"),
            "physical_target_cache": segment.get("physical_target_cache_path"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train from a complete base YAML plus a small recursive override YAML."
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--override", required=True)
    parser.add_argument(
        "--print-merged",
        action="store_true",
        help="Print the merged config before distributed training starts.",
    )
    args = parser.parse_args()

    cfg = merge_config(load_config(args.base), load_config(args.override))
    if args.print_merged and int(os.environ.get("RANK", "0")) == 0:
        # Full merged YAML is convenient interactively, but thousands of lines
        # from every torchrun rank make run progress logs unusable.  In a
        # scheduled run, emit one compact rank-0 summary instead.
        if os.environ.get("RUN_ID"):
            print(
                "[CONFIG] "
                + json.dumps(_compact_config_summary(cfg), sort_keys=True),
                flush=True,
            )
        else:
            print(yaml.safe_dump(cfg, sort_keys=False), flush=True)
    run_training(cfg)


if __name__ == "__main__":
    main()
