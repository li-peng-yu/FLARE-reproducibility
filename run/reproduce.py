#!/usr/bin/env python3
"""Run the 2026-09-05 paper workflows, or print their complete commands."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("PYTHON_BIN", sys.executable)


def py(script: str, *args: object) -> list[str]:
    return [PYTHON, str(ROOT / script), *map(str, args)]


def experiment(name: str, index: int = 0) -> list[str]:
    return ["bash", str(ROOT / "run/experiments" / (name + ".sh")), str(index)]


def shell(script: str, *args: object, **variables: object) -> list[str]:
    command = ["bash", str(ROOT / "run" / script), *map(str, args)]
    if variables:
        command = ["env", *[f"{key}={value}" for key, value in variables.items()], *command]
    return command


def auxiliary_commands(output: Path) -> list[list[str]]:
    checkpoints = Path(os.environ.get("FLARE_CHECKPOINT_ROOT", str(ROOT.parent / "FLARE_checkpoints"))) / "flare"
    names = {"stage1": "core", "mixed": "stage2_mixed", "gt_only": "stage2_gt",
             "cartesian": "cartesian", "fno": "fno", "no_target_time": "no_target_time"}
    def evaluate(script: str, label: str, *rest: object) -> list[str]:
        options = {"ODE_STEPS": 1, "METHOD": "euler"} if label == "fno" else {}
        return shell(script, checkpoints / (names[label] + "_seed78.pt"), label, *rest, **options)
    result = [evaluate("evaluate_single_segment.sh", label) for label in names]
    for steps in (5, 10, 20):
        result.append(shell("evaluate_single_segment.sh", checkpoints / "core_seed78.pt", "standard_prior",
                            OUTPUT_ROOT="outputs/solver_sensitivity", ODE_STEPS=steps, METHOD="heun"))
    for draw in range(4):
        result.extend(evaluate("evaluate_rollout.sh", label, draw) for label in ("stage1", "gt_only", "mixed"))
    result.extend(evaluate("evaluate_semigroup.sh", label) for label in ("stage1", "mixed", "cartesian"))
    result.extend(evaluate("evaluate_time_interpolation.sh", label) for label in ("stage1", "no_target_time", "cartesian", "fno"))
    result.extend(shell("evaluate_ood.sh", target) for target in ("flare-single", "fno-single"))
    result.extend(shell("evaluate_ood.sh", "flare-rollout", draw) for draw in range(4))
    result.append(shell("evaluate_ood.sh", "fno-rollout", 0))
    bootstrap = ["--bootstrap", 50000, "--seed", 20260822]
    result.extend([
        py("scripts/summarize_x5_single_segment.py", "--input-dir", "outputs/single_segment",
           "--output-dir", output / "single_segment", "--labels", *names, *bootstrap),
        py("scripts/summarize_x5_paper_rollout.py", "--input-dir", "outputs/rollout/raw",
           "--output-dir", output / "rollout", "--labels", "stage1", "mixed", "gt_only", "--draws", 4, *bootstrap),
        py("scripts/summarize_x5_semigroup_handoff.py", "--input-root", "outputs/semigroup_handoff",
           "--output-dir", output / "semigroup_handoff", "--labels", "stage1", "mixed", "cartesian", *bootstrap),
        py("scripts/summarize_x5_time_interpolation.py", "--input-dir", "outputs/time_interpolation",
           "--output-dir", output / "time_interpolation", "--labels", "stage1", "no_target_time", "cartesian", "fno", *bootstrap),
        py("scripts/summarize_x5_ood_ring.py", "--root", "outputs/ood_ring",
           "--manifest", ROOT / "configs/data/ood_ring_split.json", "--output-dir", output / "ood_ring", *bootstrap),
    ])
    return result


def commands(workflow: str, output: Path) -> list[list[str]]:
    result: list[list[str]] = []
    def tasks(name: str, count: int) -> None:
        result.extend(experiment(name, index) for index in range(count))
    if workflow == "auxiliary":
        return auxiliary_commands(output)
    if workflow == "auxiliary-distribution":
        result.extend(shell("evaluate_auxiliary_distribution.sh", method)
                      for method in ("flare", "poseidon_t", "cno_fm", "dpot_ti", "mpp_avit_ti", "pdearena_unet", "le_pde"))
        result.extend(shell("neuralmag_auxiliary.sh", "score", index) for index in range(11))
        result.append(shell("neuralmag_auxiliary.sh", "merge"))
        return result
    if workflow == "cache":
        return [[PYTHON, "-m", "skyrmion_cfm.data.memmap", "--config", config, "--dtype", "float16"]
                for config in ("configs/base/x5.yaml", "configs/data/x30_evaluation.yaml")]
    if workflow == "geometry":
        tasks("run_x5_seed_geometry_exact_control_rollout", 15)
        tasks("run_x5_seed_geometry_exact_control_finalize_cpu", 1)
        tasks("run_x5_rotation_mse_exact_control", 3)
        tasks("run_x5_rotation_mse_finalize_cpu", 1)
    elif workflow == "external-primary":
        result.extend(experiment("run_x5_multisegment_rollout_distribution", i) for i in range(1, 7))
        tasks("run_x5_multisegment_rollout_neuralmag", 11)
        tasks("run_x5_multisegment_rollout_neuralmag_merge", 1)
        tasks("run_x5_multisegment_exact_anchor_neuralmag", 11)
    elif workflow == "timing":
        tasks("run_x5_fixed_2plus3_batch_sweep", 7)
        tasks("run_x5_fixed_2plus3_neuralmag_sweep", 8)
    elif workflow == "ensemble-training":
        tasks("run_x5_deterministic_deep_ensemble_train", 8)
        tasks("run_x5_deterministic_seed81_82_train", 8)
        tasks("run_x5_direct_unet_seed81_82_train_2gpu", 2)
    elif workflow == "ensemble":
        tasks("run_x5_deterministic_seed_eval", 8)
        tasks("run_x5_deterministic_seed81_82_eval", 8)
        tasks("run_x5_direct_unet_seed81_82_eval", 2)
        tasks("run_x5_five_seed_ensemble_eval", 5)
        tasks("run_x5_three_of_five_subset_eval", 50)
        tasks("run_x5_five_seed_ensemble_finalize_cpu", 1)
    elif workflow == "jitter":
        tasks("run_x5_anchor_jitter", 15)
        tasks("run_x5_deterministic_stochasticity_finalize_cpu", 1)
    elif workflow == "horizons":
        tasks("run_x5_horizon_distribution", 5)
        result.append(py("scripts/summarize_x5_horizon_distribution.py",
                         "--root", "outputs/paper_artifacts/horizon_distribution",
                         "--output", "outputs/paper_artifacts/horizon_distribution",
                         "--figure", output / "appendix_horizon_diagnostics.pdf"))
    elif workflow == "metric-stress":
        tasks("run_x30_standard_prior_metric_stress", 1)
    elif workflow == "finalize-primary":
        return [["bash", str(ROOT / "run/summarize_primary.sh")]]
    else:
        raise ValueError(workflow)
    return result


WORKFLOWS = ("cache", "geometry", "external-primary", "timing", "finalize-primary",
             "ensemble-training", "ensemble", "jitter", "horizons", "auxiliary",
             "auxiliary-distribution", "metric-stress")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", choices=WORKFLOWS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reproduced_paper")
    parser.add_argument("--step", type=int, help="Run one numbered step (zero based).")
    args = parser.parse_args()
    output = args.output.resolve()
    planned = commands(args.workflow, output)
    if args.step is not None:
        if not 0 <= args.step < len(planned):
            parser.error(f"step must be in 0..{len(planned)-1}")
        planned = [planned[args.step]]
    env = dict(os.environ)
    env["PYTHON_BIN"] = PYTHON
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "third_party/distribution_score/src"), env.get("PYTHONPATH", "")])
    env.setdefault("FLARE_DATASET_ROOT", str(ROOT.parent / "FLARE_dataset"))
    env.setdefault("FLARE_CHECKPOINT_ROOT", str(ROOT.parent / "FLARE_checkpoints"))
    env.setdefault("FLARE_TEMPERATURE_ROOT", str(ROOT / "outputs/temperature_data"))
    env.setdefault("X5_AUTHOR_REPO_ROOT", str(ROOT / "third_party"))
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
        (output / (args.workflow + "_commands.json")).write_text(json.dumps(planned, indent=2) + "\n")
    for index, command in enumerate(planned):
        print(f"[{index}] {shlex.join(command)}", flush=True)
        if not args.dry_run:
            with (output / f"{args.workflow}_{args.step if args.step is not None else index:03d}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
