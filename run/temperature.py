#!/usr/bin/env python3
"""Portable frozen-checkpoint temperature experiment (30/90/150/225/300/400 K)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skyrmion_cfm.config import load_config

DATA = Path(os.environ.get("FLARE_DATASET_ROOT", str(ROOT.parent / "FLARE_dataset")))
TEMP = Path(os.environ.get("FLARE_TEMPERATURE_ROOT", str(ROOT / "outputs/temperature_data")))
CHECKPOINT = Path(os.environ.get("FLARE_CHECKPOINT_ROOT", str(ROOT.parent / "FLARE_checkpoints"))) / "flare/core_seed78.pt"
STATS = ROOT / "configs/stats/dataset_stats_both.json"
NAMES = {
    30: "skx_bt_test300_to30_x5_sharedrelax_20260903",
    90: "skx_bt_test300_to90_x5_sharedrelax_20260902",
    150: "skx_bt_test300_to150_x5_sharedrelax_20260902",
    225: "skx_bt_test300_to225_x5_sharedrelax_20260902",
    300: "skx_bt_test300_id_x5_view_20260901",
    400: "skx_bt_test300_to400_x5_sharedrelax_20260901",
}
SEEDS = {30: 3020260903, 90: 90202609, 150: 150202609, 225: 225202609, 400: 40020260901}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check_inputs(temperature: int) -> None:
    for path, expected in ((CHECKPOINT, "44b71927fa4e5454f6583f9e1b43413ae5cefdf1a20b6a01202568ad3a593d96"),
                           (STATS, "d359e497478f3d0b9e9533f5a9b03f9a1a3235ed23f5f37c8c7b6f99ce290236")):
        if digest(path) != expected:
            raise RuntimeError(f"Frozen input SHA256 mismatch: {path}")
    audit = TEMP / NAMES[400 if temperature == 300 else temperature] / "completion_audit.json"
    if json.loads(audit.read_text()).get("status") != "pass_complete":
        raise RuntimeError(f"Dataset must pass the complete leakage/provenance audit: {audit}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "simulate", "audit", "evaluate", "summarize"))
    p.add_argument("--temperatures", nargs="+", type=int, choices=tuple(NAMES), default=[400, 30, 90, 150, 225])
    p.add_argument("--output", type=Path, default=ROOT / "outputs/temperature_eval")
    p.add_argument("--task", type=int, help="MuMax base task, 1..36; omitted runs all 36.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    os.chdir(ROOT)
    output = args.output.resolve()
    python = os.environ.get("PYTHON_BIN", sys.executable)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'third_party/distribution_score/src'}:{env.get('PYTHONPATH', '')}"
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("SKYRMION_CFM_FUSED_OPS", "1")
    planned = []
    def add(script: str, *rest: object) -> None:
        planned.append([python, str(ROOT / script), *map(str, rest)])
    for t in args.temperatures:
        role = "id_reference" if t in (30, 150, 300) else "ood"
        tag = ("id" if role == "id_reference" else "ood") + str(t)
        data_root = TEMP / NAMES[t]
        if args.phase == "prepare":
            if t == 300:
                continue  # The matched 300 K view is created by prepare(400).
            view = TEMP / (NAMES[300] if t == 400 else f"skx_bt_test300_id_x5_view_for{t}_20260902")
            add("dataset_generation/prepare_x5_temperature_ood_400k.py",
                "--source-root", DATA / "primary_x5/skx_bt_1000base_x5_sharedrelax_20260805",
                "--training-root", DATA / "primary_x5/skx_bt_1000base_x5_20260803",
                "--training-root", DATA / "primary_x5/skx_bt_1000base_x5_sharedrelax_20260805",
                "--x30-root", DATA / "self_consistency_x30/skx_bt_165base_x30_sharedrelax_20260813",
                "--target-temp-k", t, "--target-temperature-index", {30: 0, 150: 1, 400: 3}.get(t, -1),
                "--target-role", role, "--seed-generator-seed", SEEDS[t],
                "--output-root", data_root, "--id-view-root", view, "--execute")
        elif args.phase == "simulate":
            if t == 300:
                continue
            indices = [args.task] if args.task is not None else range(1, 37)
            for index in indices:
                if not 1 <= index <= 36:
                    p.error("--task must be in 1..36")
                planned.append(["bash", str(ROOT / "dataset_generation/run_temperature_mumax.sh"), str(data_root), str(t), str(index)])
        elif args.phase == "audit":
            if t != 300:
                add("dataset_generation/audit_x5_temperature_ood_400k.py", "--root", data_root,
                    "--checkpoint", CHECKPOINT, "--target-temp-k", t, "--target-role", role, "--require-complete")
        elif args.phase == "evaluate":
            config = output / "configs" / f"{tag}.yaml"
            if not args.dry_run:
                import yaml
                check_inputs(t)
                template = load_config(ROOT / "configs/data/x5_temperature_ood90_matched_test36.yaml")
                template["data"].update(dataset_root=str(data_root), trajectory_index_cache=str(output / "cache" / (tag + ".pkl")))
                template["data"]["memmap"].update(enabled=False, auto_build=False, path=str(output / "cache" / tag))
                config.parent.mkdir(parents=True, exist_ok=True)
                config.write_text(yaml.safe_dump(template, sort_keys=False))
                (output / "single_segment").mkdir(parents=True, exist_ok=True)
            common = ["--checkpoint", CHECKPOINT, "--state", "ema", "--device", "cuda", "--ode-steps", 10]
            add("scripts/evaluate_x5_single_segment.py", *common, "--stats-cache-override", STATS,
                "--data-config", config, "--label", f"stage1_{tag}", "--output", output / f"single_segment/{tag}.csv",
                "--method", "heun", "--batch-size", 32, "--draws", 4, "--seed", 20260813, "--max-cases", 0, "--split", "test")
            add("scripts/evaluate_x5_semigroup_handoff.py", *common, "--stats-cache-override", STATS,
                "--data-config", config, "--label", f"stage1_{tag}", "--output-dir", output / "handoff" / tag,
                "--batch-size", 32, "--max-semigroup", 512, "--max-handoff", 736, "--draws", 4, "--seed", 20260813)
            add("scripts/evaluate_skx_x5_same_condition_distribution.py", *common, "--stats", STATS,
                "--evaluation-config", config, "--output-root", output / "distribution" / tag,
                "--segments", 2, "--repeat-dataset", NAMES[t], "--repeats-per-group", 5, "--num-model", 5,
                "--score-num-model", 5, "--batch-size", 5, "--bootstrap", 5000, "--seed", 208160700)
    if args.phase == "summarize":
        for lower, upper, target in ((30, 150, 90), (150, 300, 225)):
            add("scripts/summarize_temperature_interpolation_baseline.py", "--lower-root", output,
                "--upper-root", output, "--target-root", output, "--lower-temp-k", lower, "--upper-temp-k", upper,
                "--target-temp-k", target, "--bootstrap", 10000, "--seed", 20260903,
                "--output", output / f"summary/interpolation_{target}k.json")
        add("scripts/summarize_x5_temperature_ood.py", "--input-root", output,
            "--output-dir", output / "summary/extrapolation_400k", "--id-temp-k", 300,
            "--ood-temp-k", 400, "--bootstrap", 10000, "--seed", 20260901,
            "--audit", TEMP / NAMES[400] / "completion_audit.json")
    for command in planned:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
