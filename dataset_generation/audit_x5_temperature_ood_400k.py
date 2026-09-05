#!/usr/bin/env python3
"""Audit completed matched-temperature outputs and checkpoint/data non-overlap."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_ROOT = (
    PROJECT_ROOT.parent
    / "FLARE_temperature_ood/skx_bt_test300_to400_x5_sharedrelax_20260901"
)
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/flare/core_seed78.pt"
EXPECTED_TRAINING_ROOT_NAMES = {
    "skx_bt_1000base_x5_20260803",
    "skx_bt_1000base_x5_sharedrelax_20260805",
}
EXPECTED_MUMAX_VERSION = "mumax 3.12"
EXPECTED_MUMAX_COMMIT = "6e5c98bb"
EXPECTED_MUMAX_SHA256 = (
    "37eff7666a86c349089daaa8a637c4f4270a6a79c9135b8fc19a7115e628d06f"
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_count(run_dir: Path) -> int:
    return len(list((run_dir / "run.out").glob("m[0-9]*.ovf")))


def solver_provenance(log_path: Path) -> tuple[str, str, str] | None:
    if not log_path.is_file():
        return None
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = [handle.readline().strip() for _ in range(4)]
    if len(lines) < 4:
        raise RuntimeError(f"truncated MuMax log: {log_path}")
    version = lines[0].removeprefix("//")
    commit = lines[1].removeprefix("//commit hash:").strip()
    gpu = lines[3].removeprefix("//GPU info:").strip()
    if EXPECTED_MUMAX_VERSION not in version or "CUDA-12.0" not in version:
        raise RuntimeError(f"wrong MuMax version in {log_path}: {version}")
    if commit != EXPECTED_MUMAX_COMMIT:
        raise RuntimeError(f"wrong MuMax commit in {log_path}: {commit}")
    return version, commit, gpu


def table_temperatures(table_path: Path, target_temp_k: float) -> set[float]:
    if not table_path.is_file():
        return set()
    with table_path.open("r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().removeprefix("# ").rstrip("\n").split("\t")
        try:
            temp_index = header.index("Temp (K)")
        except ValueError as exc:
            raise RuntimeError(f"Temp (K) missing from {table_path}") from exc
        values = {
            float(fields[temp_index])
            for line in handle
            if line.strip()
            for fields in [line.rstrip("\n").split("\t")]
        }
    if values != {target_temp_k}:
        raise RuntimeError(
            f"non-target temperature values in {table_path}: {sorted(values)}"
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--checkpoint-evidence",
        type=Path,
        default=None,
        help="prior pass_complete audit for the identical frozen checkpoint",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--target-temp-k", type=float, default=400.0)
    parser.add_argument(
        "--target-role", choices=("ood", "id_reference"), default="ood"
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if not float(args.target_temp_k).is_integer():
        raise ValueError("this dataset naming scheme requires an integer temperature")
    target_temp_k = float(args.target_temp_k)
    target_token = str(int(target_temp_k))
    run_re = re.compile(
        rf"_base(?P<base>\d+)_tr(?P<repeat>\d+)_T{re.escape(target_token)}K_"
    )

    root = args.root.absolute()
    if not root.is_dir():
        raise FileNotFoundError(root)
    leakage = read_json(root / "leakage_audit.json")
    if leakage.get("status") != "pass":
        raise RuntimeError("pre-generation leakage audit did not pass")
    target_absent = target_temp_k not in {
        float(value)
        for value in leakage["checkpoint_training_temperature_support_k"]
    }
    if args.target_role == "ood" and not target_absent:
        raise RuntimeError("OOD target appears in checkpoint training support")
    if args.target_role == "id_reference" and target_absent:
        raise RuntimeError("ID-reference target is absent from training support")

    checkpoint_evidence = None
    if args.checkpoint_evidence is None:
        import torch

        checkpoint_payload = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        checkpoint_step = int(checkpoint_payload.get("step", -1))
        raw_roots = checkpoint_payload.get("config", {}).get("data", {}).get(
            "dataset_root", []
        )
        checkpoint_roots = [str(raw_roots)] if isinstance(raw_roots, str) else [
            str(value) for value in raw_roots
        ]
        checkpoint_sha256 = sha256(args.checkpoint)
        del checkpoint_payload
    else:
        checkpoint_evidence = read_json(args.checkpoint_evidence)
        if checkpoint_evidence.get("status") != "pass_complete":
            raise RuntimeError("checkpoint evidence is not pass_complete")
        checkpoint_step = int(checkpoint_evidence["checkpoint_step"])
        checkpoint_roots = [
            str(value)
            for value in checkpoint_evidence["checkpoint_embedded_dataset_roots"]
        ]
        checkpoint_sha256 = str(checkpoint_evidence["checkpoint_sha256"])
    root_names = {Path(value).name for value in checkpoint_roots}
    if root_names != EXPECTED_TRAINING_ROOT_NAMES:
        raise RuntimeError(
            f"checkpoint embeds unexpected dataset roots: {checkpoint_roots}"
        )

    by_base: Counter[int] = Counter()
    temperatures: Counter[float] = Counter()
    splits: Counter[str] = Counter()
    thermal_seeds: set[int] = set()
    incomplete: list[dict[str, Any]] = []
    frame_histogram: Counter[int] = Counter()
    solver_versions: Counter[str] = Counter()
    solver_commits: Counter[str] = Counter()
    solver_gpus: Counter[str] = Counter()
    simulator_temperatures: Counter[float] = Counter()
    run_rows: list[dict[str, Any]] = []
    for params_path in sorted(root.glob("r_*/params.json")):
        run_dir = params_path.parent
        match = run_re.search(run_dir.name)
        if match is None:
            raise RuntimeError(f"invalid target-temperature run name: {run_dir.name}")
        params = read_json(params_path)
        base = int(params["base_index"])
        repeat = int(params["thermal_repeat_index"])
        temp = float(params["temp_k"])
        split = str(params["split"])
        seed = int(params["thermal_seed"])
        if base != int(match.group("base")) or repeat != int(match.group("repeat")):
            raise RuntimeError(f"run identity mismatch: {run_dir}")
        if temp != target_temp_k or split != "test":
            raise RuntimeError(f"invalid OOD metadata: {run_dir}")
        if seed in thermal_seeds:
            raise RuntimeError(f"duplicate OOD thermal seed: {seed}")
        thermal_seeds.add(seed)
        by_base[base] += 1
        temperatures[temp] += 1
        splits[split] += 1
        count = frame_count(run_dir)
        frame_histogram[count] += 1
        required = (
            run_dir / "run.out/m_initial.ovf",
            run_dir / "run.out/m_final.ovf",
            run_dir / "run.out/table.txt",
            run_dir / "run.out/log.txt",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if count < 2:
            missing.append(f"only {count} numbered frames")
        if missing:
            incomplete.append({"run_id": run_dir.name, "missing": missing})
        provenance = solver_provenance(run_dir / "run.out/log.txt")
        if provenance is not None:
            version, commit, gpu = provenance
            solver_versions[version] += 1
            solver_commits[commit] += 1
            solver_gpus[gpu] += 1
        for value in table_temperatures(run_dir / "run.out/table.txt", target_temp_k):
            simulator_temperatures[value] += 1
        run_rows.append(
            {
                "run_id": run_dir.name,
                "base_index": base,
                "thermal_repeat_index": repeat,
                "thermal_seed": seed,
                "numbered_frames": count,
                "complete": not missing,
            }
        )

    relax_incomplete: list[dict[str, Any]] = []
    relax_seeds: set[int] = set()
    for provenance_path in sorted(root.glob("shared_relax/base*/checkpoint_provenance.json")):
        provenance = read_json(provenance_path)
        base = int(provenance["base_index"])
        seed = int(provenance["thermal_seed"])
        if seed in thermal_seeds or seed in relax_seeds:
            raise RuntimeError(f"relax/branch thermal seed collision: {seed}")
        relax_seeds.add(seed)
        relax_dir = provenance_path.parent
        required = (
            relax_dir / "relax.out/m_analytic_initial.ovf",
            relax_dir / "relax.out/m_initial.ovf",
            relax_dir / "relax.out/table.txt",
            relax_dir / "relax.out/log.txt",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            relax_incomplete.append({"base_index": base, "missing": missing})
        provenance = solver_provenance(relax_dir / "relax.out/log.txt")
        if provenance is not None:
            version, commit, gpu = provenance
            solver_versions[version] += 1
            solver_commits[commit] += 1
            solver_gpus[gpu] += 1
        for value in table_temperatures(relax_dir / "relax.out/table.txt", target_temp_k):
            simulator_temperatures[value] += 1

    if len(run_rows) != 180 or len(by_base) != 36 or set(by_base.values()) != {5}:
        raise RuntimeError(
            f"wrong OOD grouping: runs={len(run_rows)} groups={dict(by_base)}"
        )
    if len(relax_seeds) != 36:
        raise RuntimeError(f"wrong relaxation seed count: {len(relax_seeds)}")
    complete = not incomplete and not relax_incomplete
    status_rows = 0
    status_bases: set[int] = set()
    for status_path in sorted(root.glob("generation_status/base*.tsv")):
        with status_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                status_rows += 1
                status_bases.add(int(row["base"]))
                if EXPECTED_MUMAX_VERSION not in row["mumax_version"]:
                    raise RuntimeError(f"wrong solver version in {status_path}")
                if EXPECTED_MUMAX_COMMIT not in row["mumax_commit"]:
                    raise RuntimeError(f"wrong solver commit in {status_path}")
                if row["mumax_sha256"] != EXPECTED_MUMAX_SHA256:
                    raise RuntimeError(f"wrong solver hash in {status_path}")
    if complete and (len(status_bases) != 36 or status_rows != 216):
        raise RuntimeError(
            f"wrong generation status coverage: bases={len(status_bases)} rows={status_rows}"
        )
    if args.require_complete and not complete:
        raise RuntimeError(
            f"dataset incomplete: branches={len(incomplete)} relaxations={len(relax_incomplete)}"
        )

    result = {
        "status": "pass_complete" if complete else "pass_inputs_pending_outputs",
        "root": str(root),
        "checkpoint": str(args.checkpoint.absolute()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_evidence": (
            str(args.checkpoint_evidence.absolute())
            if args.checkpoint_evidence is not None
            else None
        ),
        "checkpoint_step": checkpoint_step,
        "checkpoint_embedded_dataset_roots": checkpoint_roots,
        "checkpoint_training_temperature_support_k": leakage[
            "checkpoint_training_temperature_support_k"
        ],
        "target_temperature_k": target_temp_k,
        "target_role": args.target_role,
        "target_temperature_absent_from_checkpoint_train_and_validation": target_absent,
        "selected_bases_absent_from_checkpoint_train_and_validation": True,
        "split_counts": dict(splits),
        "temperature_counts": {str(key): value for key, value in temperatures.items()},
        "base_conditions": len(by_base),
        "branches": len(run_rows),
        "repeats_per_base": sorted(set(by_base.values())),
        "branch_thermal_seeds": len(thermal_seeds),
        "relaxation_thermal_seeds": len(relax_seeds),
        "all_new_seeds_unique": True,
        "expected_mumax_sha256": EXPECTED_MUMAX_SHA256,
        "solver_versions": dict(solver_versions),
        "solver_commits": dict(solver_commits),
        "solver_gpu_info": dict(solver_gpus),
        "simulator_temperature_counts": {
            str(key): value for key, value in sorted(simulator_temperatures.items())
        },
        "generation_status_files": len(status_bases),
        "generation_status_rows": status_rows,
        "numbered_frame_count_histogram": {
            str(key): value for key, value in sorted(frame_histogram.items())
        },
        "incomplete_branches": incomplete,
        "incomplete_relaxations": relax_incomplete,
        "no_300k_thermalized_checkpoint_reused": True,
        "runs": run_rows,
    }
    output = args.output or (root / "completion_audit.json")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    console_summary = {
        key: value
        for key, value in result.items()
        if key not in {"runs", "incomplete_branches", "incomplete_relaxations"}
    }
    console_summary["incomplete_branch_count"] = len(incomplete)
    console_summary["incomplete_relaxation_count"] = len(relax_incomplete)
    console_summary["audit_file"] = str(output)
    print(json.dumps(console_summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
