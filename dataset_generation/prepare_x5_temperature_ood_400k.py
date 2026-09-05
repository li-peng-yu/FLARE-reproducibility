#!/usr/bin/env python3
"""Prepare a leakage-audited, matched-temperature x5 evaluation dataset.

The source is the corrected shared-relaxation x5 corpus.  Only 300 K bases
whose metadata split is ``test`` are selected.  For each selected base this
builder creates a fresh target-temperature shared 1 ns relaxation and five fresh
post-checkpoint thermal branches.  No 300 K thermalized state or trajectory
output is reused.

The script also creates a lightweight 300 K ID view containing the same 36
base conditions and five existing repeats.  That view consists only of
symlinks to completed source trajectories and is used for matched reporting.

Without ``--execute`` the script performs discovery/leakage checks and prints
the deterministic plan.  Existing destinations are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "FLARE_dataset"
DEFAULT_SOURCE = (
    DATASET_ROOT / "primary_x5/skx_bt_1000base_x5_sharedrelax_20260805"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT.parent
    / "FLARE_temperature_ood/skx_bt_test300_to400_x5_sharedrelax_20260901"
)
DEFAULT_ID_VIEW = (
    PROJECT_ROOT.parent
    / "FLARE_temperature_ood/skx_bt_test300_id_x5_view_20260901"
)
TRAINING_ROOTS = (
    DATASET_ROOT / "primary_x5/skx_bt_1000base_x5_20260803",
    DATASET_ROOT / "primary_x5/skx_bt_1000base_x5_sharedrelax_20260805",
)
X30_ROOT = (
    DATASET_ROOT
    / "self_consistency_x30/skx_bt_165base_x30_sharedrelax_20260813"
)

SOURCE_TEMP_K = 300.0
TARGET_TEMP_K = 400.0
TARGET_TEMPERATURE_INDEX = 3
EXPECTED_BASES = 36
REPEATS_PER_BASE = 5
SEED_GENERATOR_SEED = 400_20260901
TARGET_ROLE = "ood"

RUN_RE = re.compile(
    r"^r_(?P<index>\d+)_base(?P<base>\d+)_tr(?P<tr>\d+)_.*$"
)
THERMAL_SEED_RE = re.compile(r"ThermSeed\((\d+)\)")
TEMP_ASSIGNMENT_RE = re.compile(r"(?m)^Temp\s*=\s*300(?:\.0+)?\s*$")
TS_NAME_RE = re.compile(r"_TS\d+_")


def target_temp_token() -> str:
    if not float(TARGET_TEMP_K).is_integer():
        raise ValueError("this dataset naming scheme requires an integer temperature")
    return str(int(TARGET_TEMP_K))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--id-view-root", type=Path, default=DEFAULT_ID_VIEW)
    parser.add_argument("--target-temp-k", type=float, default=TARGET_TEMP_K)
    parser.add_argument("--target-temperature-index", type=int, default=None)
    parser.add_argument("--seed-generator-seed", type=int, default=None)
    parser.add_argument(
        "--target-role", choices=("ood", "id_reference"), default="ood"
    )
    parser.add_argument(
        "--training-root", action="append", type=Path, default=None,
        help="checkpoint training roots to audit; may be supplied repeatedly",
    )
    parser.add_argument("--x30-root", type=Path, default=X30_ROOT)
    parser.add_argument(
        "--reuse-leakage-audit",
        type=Path,
        default=None,
        help="reuse a prior complete scan of the same checkpoint roots/base cohort",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="materialize inputs and the matched ID view after all audits pass",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def replace_strings(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: replace_strings(item, replacements)
            for key, item in value.items()
        }
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_source_runs(
    source_root: Path, *, allowed_bases: set[int] | None = None
) -> dict[int, dict[int, Path]]:
    groups: dict[int, dict[int, Path]] = defaultdict(dict)
    if allowed_bases is None:
        params_paths = sorted(source_root.glob("r_*/params.json"))
    else:
        params_paths = sorted(
            params_path
            for base in allowed_bases
            for params_path in source_root.glob(
                f"r_*_base{base:04d}_tr*_*/params.json"
            )
        )
    for params_path in params_paths:
        run_dir = params_path.parent
        match = RUN_RE.match(run_dir.name)
        if match is None:
            raise RuntimeError(f"unrecognized run directory: {run_dir}")
        name_base = int(match.group("base"))
        if allowed_bases is not None and name_base not in allowed_bases:
            continue
        params = read_json(params_path)
        if str(params.get("split")) != "test":
            continue
        if float(params.get("temp_k", -1.0)) != SOURCE_TEMP_K:
            continue
        base = int(params["base_index"])
        repeat = int(params["thermal_repeat_index"])
        if base != int(match.group("base")) or repeat != int(match.group("tr")):
            raise RuntimeError(f"directory/metadata identity mismatch: {run_dir}")
        if repeat in groups[base]:
            raise RuntimeError(f"duplicate source base/repeat: {base}/{repeat}")
        groups[base][repeat] = run_dir

    if len(groups) != EXPECTED_BASES:
        raise RuntimeError(
            f"expected {EXPECTED_BASES} test-only 300 K bases, found {len(groups)}"
        )
    for base, runs in groups.items():
        if sorted(runs) != list(range(REPEATS_PER_BASE)):
            raise RuntimeError(f"base {base} does not have repeats 0..4")
        checkpoint_input = (
            source_root / "shared_relax" / f"base{base:04d}" / "relax.mx3"
        )
        if not checkpoint_input.is_file():
            raise FileNotFoundError(checkpoint_input)
        for run_dir in runs.values():
            required = (
                "run.mx3",
                "params.json",
                "current_protocol.json",
                "current_control.csv",
                "sample_manifest.jsonl",
                "result_summary.json",
                "run.out/m_initial.ovf",
                "run.out/m_final.ovf",
                "run.out/table.txt",
            )
            missing = [name for name in required if not (run_dir / name).is_file()]
            if missing:
                raise FileNotFoundError(f"incomplete ID source {run_dir}: {missing}")
    return dict(groups)


def scan_params(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    if not root.is_dir():
        return []
    rows: list[tuple[Path, dict[str, Any]]] = []
    for params_path in sorted(root.glob("r_*/params.json")):
        rows.append((params_path, read_json(params_path)))
    return rows


def leakage_audit(
    groups: dict[int, dict[int, Path]],
    *,
    training_roots: tuple[Path, ...],
    x30_root: Path,
) -> tuple[dict[str, Any], set[int]]:
    selected = set(groups)
    used_thermal_seeds: set[int] = set()
    root_reports: list[dict[str, Any]] = []
    selected_membership: dict[int, set[str]] = defaultdict(set)
    all_training_temps: set[float] = set()

    scan_roots = [*training_roots]
    if x30_root.is_dir():
        scan_roots.append(x30_root)
    for root in scan_roots:
        rows = scan_params(root)
        temperatures = Counter()
        split_temperatures = Counter()
        for _path, params in rows:
            temp = float(params["temp_k"])
            split = str(params["split"])
            temperatures[temp] += 1
            split_temperatures[(split, temp)] += 1
            if "thermal_seed" in params:
                used_thermal_seeds.add(int(params["thermal_seed"]))
            elif "seed" in params:
                used_thermal_seeds.add(int(params["seed"]))
            if root in training_roots:
                all_training_temps.add(temp)
                base = int(params["base_index"])
                if base in selected:
                    selected_membership[base].add(split)
        root_reports.append(
            {
                "root": str(root.resolve()),
                "trajectories": len(rows),
                "temperatures_k": {
                    str(key): value for key, value in sorted(temperatures.items())
                },
                "split_by_temperature": [
                    {"split": split, "temp_k": temp, "trajectories": count}
                    for (split, temp), count in sorted(split_temperatures.items())
                ],
            }
        )

    expected_temperatures = {30.0, 150.0, 300.0}
    if all_training_temps != expected_temperatures:
        raise RuntimeError(
            f"checkpoint training roots have unexpected temperatures: {all_training_temps}"
        )
    target_in_training = TARGET_TEMP_K in all_training_temps
    if TARGET_ROLE == "ood" and target_in_training:
        raise RuntimeError("target temperature is present in a checkpoint training root")
    if TARGET_ROLE == "id_reference" and not target_in_training:
        raise RuntimeError("ID-reference temperature is absent from checkpoint training roots")
    bad_membership = {
        base: sorted(labels)
        for base, labels in selected_membership.items()
        if labels != {"test"}
    }
    if bad_membership:
        raise RuntimeError(f"selected bases appear outside test: {bad_membership}")
    if set(selected_membership) != selected:
        missing = sorted(selected - set(selected_membership))
        raise RuntimeError(f"selected bases missing from training-root audit: {missing}")

    audit = {
        "status": "pass",
        "target_temperature_k": TARGET_TEMP_K,
        "checkpoint_training_temperature_support_k": sorted(all_training_temps),
        "target_role": TARGET_ROLE,
        "target_temperature_absent_from_all_checkpoint_roots": not target_in_training,
        "selected_base_count": len(selected),
        "selected_base_indices": sorted(selected),
        "selected_base_split_membership": {
            str(base): sorted(selected_membership[base]) for base in sorted(selected)
        },
        "selected_bases_absent_from_train_and_validation": True,
        "audited_roots": root_reports,
        "existing_thermal_seed_count": len(used_thermal_seeds),
    }
    return audit, used_thermal_seeds


def reuse_leakage_audit(
    path: Path, groups: dict[int, dict[int, Path]]
) -> tuple[dict[str, Any], set[int]]:
    source = read_json(path)
    selected = sorted(groups)
    support = sorted(
        float(value)
        for value in source["checkpoint_training_temperature_support_k"]
    )
    source_selected = sorted(int(value) for value in source["selected_base_indices"])
    if source.get("status") != "pass" or support != [30.0, 150.0, 300.0]:
        raise RuntimeError(f"reused leakage audit is not a valid Stage-1 audit: {path}")
    if source_selected != selected:
        raise RuntimeError("reused leakage audit has a different base cohort")
    if not source.get("selected_bases_absent_from_train_and_validation", False):
        raise RuntimeError("reused leakage audit did not establish split isolation")
    target_absent = TARGET_TEMP_K not in set(support)
    if TARGET_ROLE == "ood" and not target_absent:
        raise RuntimeError("target temperature is present in checkpoint training support")
    if TARGET_ROLE == "id_reference" and target_absent:
        raise RuntimeError("ID-reference temperature is absent from training support")

    used: set[int] = set()
    for repeats in groups.values():
        for run in repeats.values():
            params = read_json(run / "params.json")
            seed = params.get("thermal_seed", params.get("seed"))
            if seed is not None:
                used.add(int(seed))
    audit = {
        **source,
        "status": "pass",
        "target_temperature_k": TARGET_TEMP_K,
        "target_role": TARGET_ROLE,
        "target_temperature_absent_from_all_checkpoint_roots": target_absent,
        "reused_full_scan_audit": str(path.resolve()),
        "reused_full_scan_audit_sha256": sha256(path),
        "selected_base_indices": selected,
        "selected_base_count": len(selected),
        "existing_thermal_seed_count": len(used),
    }
    return audit, used


def allocate_seeds(
    bases: list[int], used: set[int]
) -> tuple[dict[int, int], dict[tuple[int, int], int]]:
    rng = random.Random(SEED_GENERATOR_SEED)

    def allocate_one() -> int:
        while True:
            value = rng.randrange(1, 2_147_483_647)
            if value not in used:
                used.add(value)
                return value

    relax = {base: allocate_one() for base in bases}
    branches = {
        (base, repeat): allocate_one()
        for base in bases
        for repeat in range(REPEATS_PER_BASE)
    }
    allocated = [*relax.values(), *branches.values()]
    if len(allocated) != EXPECTED_BASES * (REPEATS_PER_BASE + 1):
        raise RuntimeError("wrong allocated seed count")
    if len(set(allocated)) != len(allocated):
        raise RuntimeError("new thermal seeds are not unique")
    return relax, branches


def target_run_id(source_name: str, branch_seed: int) -> str:
    value = source_name.replace("_T300K_", f"_T{target_temp_token()}K_")
    value, count = TS_NAME_RE.subn(f"_TS{branch_seed}_", value, count=1)
    if count != 1 or value == source_name:
        raise RuntimeError(f"failed to rename source run: {source_name}")
    return value


def transform_mx3(
    text: str,
    *,
    old_run_id: str,
    new_run_id: str,
    thermal_seed: int,
    kind: str,
) -> str:
    text = text.replace(old_run_id, new_run_id)
    text = text.replace("_T300K_", f"_T{target_temp_token()}K_")
    text = text.replace("Temp=300 K", f"Temp={target_temp_token()} K")
    text, seed_count = THERMAL_SEED_RE.subn(
        f"ThermSeed({thermal_seed})", text, count=1
    )
    text, temp_count = TEMP_ASSIGNMENT_RE.subn(
        f"Temp = {target_temp_token()}", text, count=1
    )
    if seed_count != 1 or temp_count != 1:
        raise RuntimeError(
            f"expected one seed and temperature assignment in {kind} template "
            f"for {old_run_id}: seed={seed_count} temp={temp_count}"
        )
    header = (
        f"// Matched-temperature input: exact {target_temp_token()} K; generated from a test-only "
        "300 K base without reusing a thermalized state.\n"
    )
    text = header + text
    if text.count(f"ThermSeed({thermal_seed})") != 1:
        raise RuntimeError(f"seed substitution failed for {new_run_id}")
    if len(re.findall(rf"(?m)^Temp\s*=\s*{re.escape(target_temp_token())}\s*$", text)) != 1:
        raise RuntimeError(f"temperature substitution failed for {new_run_id}")
    expected_runs = 1 if kind == "relax" else 2
    if len(re.findall(r"(?m)^run\(", text)) != expected_runs:
        raise RuntimeError(f"unexpected run() count in {kind} input {new_run_id}")
    return text


def update_params(
    source: dict[str, Any],
    *,
    source_run: Path,
    target_root: Path,
    run_id: str,
    base: int,
    repeat: int,
    branch_seed: int,
    selection_rank: int,
) -> dict[str, Any]:
    old_run_id = str(source["run_id"])
    result = replace_strings(
        source,
        [(old_run_id, run_id), ("_T300K_", f"_T{target_temp_token()}K_")],
    )
    result.update(
        {
            "run_id": run_id,
            "index": base * REPEATS_PER_BASE + repeat,
            "base_index": base,
            "thermal_repeat_index": repeat,
            "thermal_seed": branch_seed,
            "seed": branch_seed,
            "temp_k": TARGET_TEMP_K,
            "temperature_index": TARGET_TEMPERATURE_INDEX,
            "repeat_index_within_temperature": int(
                source.get("repeat_index_within_temperature", selection_rank)
            ),
            "temperature_ood_selection_rank": selection_rank,
            "split": "test",
            "shared_initial_checkpoint_path": str(
                target_root
                / "shared_relax"
                / f"base{base:04d}"
                / "relax.out"
                / "m_initial.ovf"
            ),
            "temperature_ood": TARGET_ROLE == "ood",
            "temperature_evaluation_role": TARGET_ROLE,
            "temperature_ood_source_temp_k": SOURCE_TEMP_K,
            "temperature_ood_target_temp_k": TARGET_TEMP_K,
            "source_id300_run_dir": str(source_run.resolve()),
            "source_id300_run_id": old_run_id,
            "trajectory_origin": f"new_{target_temp_token()}k_test_only_matched_branch",
        }
    )
    if isinstance(result.get("current_protocol"), dict):
        result["current_protocol"]["branch_thermal_seed"] = branch_seed
    return result


def update_sample_row(
    row: dict[str, Any],
    *,
    old_run_id: str,
    run_id: str,
    target_root: Path,
    base: int,
    repeat: int,
    branch_seed: int,
) -> dict[str, Any]:
    result = replace_strings(
        row,
        [(old_run_id, run_id), ("_T300K_", f"_T{target_temp_token()}K_")],
    )
    run_dir = target_root / run_id
    result.update(
        {
            "run_id": run_id,
            "base_index": base,
            "thermal_repeat_index": repeat,
            "thermal_seed": branch_seed,
            "seed": branch_seed,
            "temp_k": TARGET_TEMP_K,
            "split": "test",
            "current_control_path": str(run_dir / "current_control.csv"),
            "current_protocol_path": str(run_dir / "current_protocol.json"),
            "m_initial_path": str(run_dir / "run.out" / "m_initial.ovf"),
            "m_final_path": str(run_dir / "run.out" / "m_final.ovf"),
            "shared_initial_checkpoint_path": str(
                target_root
                / "shared_relax"
                / f"base{base:04d}"
                / "relax.out"
                / "m_initial.ovf"
            ),
            "valid_reason": f"pending_new_{target_temp_token()}k_test_only_matched_simulation",
        }
    )
    for key in ("m_i_path", "m_j_path"):
        if key in row:
            result[key] = str(run_dir / "run.out" / Path(str(row[key])).name)
    return result


def update_manifest_row(
    source: dict[str, Any],
    *,
    target_root: Path,
    run_id: str,
    base: int,
    repeat: int,
    branch_seed: int,
    selection_rank: int,
) -> dict[str, Any]:
    old_run_id = str(source["run_id"])
    result = replace_strings(
        source,
        [(old_run_id, run_id), ("_T300K_", f"_T{target_temp_token()}K_")],
    )
    run_dir = target_root / run_id
    result.update(
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "mx3_path": str(run_dir / "run.mx3"),
            "current_control_path": str(run_dir / "current_control.csv"),
            "current_protocol_path": str(run_dir / "current_protocol.json"),
            "m_initial_path": str(run_dir / "run.out" / "m_initial.ovf"),
            "m_final_path": str(run_dir / "run.out" / "m_final.ovf"),
            "m_analytic_initial_path": str(
                target_root
                / "shared_relax"
                / f"base{base:04d}"
                / "relax.out"
                / "m_analytic_initial.ovf"
            ),
            "shared_initial_checkpoint_path": str(
                target_root
                / "shared_relax"
                / f"base{base:04d}"
                / "relax.out"
                / "m_initial.ovf"
            ),
            "index": base * REPEATS_PER_BASE + repeat,
            "base_index": base,
            "thermal_repeat_index": repeat,
            "thermal_seed": branch_seed,
            "seed": branch_seed,
            "temp_k": TARGET_TEMP_K,
            "temperature_index": TARGET_TEMPERATURE_INDEX,
            "repeat_index_within_temperature": int(
                source.get("repeat_index_within_temperature", selection_rank)
            ),
            "temperature_ood_selection_rank": selection_rank,
            "split": "test",
            "trajectory_origin": f"new_{target_temp_token()}k_test_only_matched_branch",
        }
    )
    return result


def source_manifest_by_identity(source_root: Path) -> dict[tuple[int, int], dict[str, Any]]:
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for row in read_jsonl(source_root / "run_manifest.jsonl"):
        key = (int(row["base_index"]), int(row["thermal_repeat_index"]))
        result[key] = row
    return result


def validate_stage(
    stage_root: Path,
    *,
    bases: list[int],
    relax_seeds: dict[int, int],
    branch_seeds: dict[tuple[int, int], int],
) -> dict[str, Any]:
    runs = sorted(stage_root.glob("r_*/params.json"))
    if len(runs) != EXPECTED_BASES * REPEATS_PER_BASE:
        raise RuntimeError(f"wrong staged branch count: {len(runs)}")
    by_base: Counter[int] = Counter()
    seen_seeds: set[int] = set(relax_seeds.values())
    sample_lines = 0
    for params_path in runs:
        params = read_json(params_path)
        base = int(params["base_index"])
        repeat = int(params["thermal_repeat_index"])
        seed = int(params["thermal_seed"])
        if float(params["temp_k"]) != TARGET_TEMP_K or params["split"] != "test":
            raise RuntimeError(f"invalid target metadata: {params_path}")
        if seed != branch_seeds[(base, repeat)] or seed in seen_seeds:
            raise RuntimeError(f"seed mismatch/collision: {params_path}")
        seen_seeds.add(seed)
        by_base[base] += 1
        run_text = (params_path.parent / "run.mx3").read_text(encoding="utf-8")
        if run_text.count(f"Temp = {target_temp_token()}") != 1 or run_text.count(
            f"ThermSeed({seed})"
        ) != 1:
            raise RuntimeError(f"invalid branch input: {params_path.parent}")
        expected_checkpoint = f"../shared_relax/base{base:04d}/relax.out/m_initial.ovf"
        if run_text.count(expected_checkpoint) != 1:
            raise RuntimeError(f"branch points to wrong checkpoint: {params_path.parent}")
        if (params_path.parent / "run.out").exists():
            raise RuntimeError(f"new branch unexpectedly has output: {params_path.parent}")
        with (params_path.parent / "sample_manifest.jsonl").open(
            "r", encoding="utf-8"
        ) as handle:
            sample_lines += sum(1 for line in handle if line.strip())

    if set(by_base) != set(bases) or set(by_base.values()) != {REPEATS_PER_BASE}:
        raise RuntimeError(f"wrong staged base groups: {by_base}")
    for base in bases:
        relax_dir = stage_root / "shared_relax" / f"base{base:04d}"
        text = (relax_dir / "relax.mx3").read_text(encoding="utf-8")
        if text.count(f"Temp = {target_temp_token()}") != 1 or text.count(
            f"ThermSeed({relax_seeds[base]})"
        ) != 1:
            raise RuntimeError(f"invalid target-temperature relaxation input: {relax_dir}")
        if (relax_dir / "relax.out").exists():
            raise RuntimeError(f"new relaxation unexpectedly has output: {relax_dir}")
    return {
        "status": "pass",
        "bases": len(by_base),
        "branches": len(runs),
        "repeats_per_base": sorted(set(by_base.values())),
        "fresh_relaxations": len(relax_seeds),
        "globally_unique_new_thermal_seeds": len(seen_seeds),
        "sample_manifest_rows": sample_lines,
        "target_temperature_k": TARGET_TEMP_K,
        "all_branches_metadata_split": "test",
        "no_300k_thermalized_output_reused": True,
    }


def materialize(
    source_root: Path,
    output_root: Path,
    id_view_root: Path,
    groups: dict[int, dict[int, Path]],
    audit: dict[str, Any],
    relax_seeds: dict[int, int],
    branch_seeds: dict[tuple[int, int], int],
) -> None:
    for path in (output_root, id_view_root):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"destination already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    output_stage = output_root.parent / f".{output_root.name}.preparing-{os.getpid()}"
    id_stage = id_view_root.parent / f".{id_view_root.name}.preparing-{os.getpid()}"
    if output_stage.exists() or id_stage.exists():
        raise FileExistsError("staging destination already exists")
    output_stage.mkdir(parents=True)
    id_stage.mkdir(parents=True)

    bases = sorted(groups)
    manifest_source = source_manifest_by_identity(source_root)
    output_manifest: list[dict[str, Any]] = []
    id_manifest: list[dict[str, Any]] = []
    branch_paths: list[Path] = []
    base_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []

    for selection_rank, base in enumerate(bases):
        source_relax = source_root / "shared_relax" / f"base{base:04d}"
        target_relax = output_stage / "shared_relax" / f"base{base:04d}"
        target_relax.mkdir(parents=True)
        source_run0 = groups[base][0]
        target_run0_id = target_run_id(
            source_run0.name, branch_seeds[(base, 0)]
        )
        relax_text = (source_relax / "relax.mx3").read_text(encoding="utf-8")
        (target_relax / "relax.mx3").write_text(
            transform_mx3(
                relax_text,
                old_run_id=source_run0.name,
                new_run_id=target_run0_id,
                thermal_seed=relax_seeds[base],
                kind="relax",
            ),
            encoding="utf-8",
        )
        write_json(
            target_relax / "checkpoint_provenance.json",
            {
                "status": f"pending_fresh_{target_temp_token()}k_shared_relaxation",
                "base_index": base,
                "split": "test",
                "source_temperature_k": SOURCE_TEMP_K,
                "target_temperature_k": TARGET_TEMP_K,
                "thermal_seed": relax_seeds[base],
                "initial_relaxation_ns": 1.0,
                "source_relax_input": str((source_relax / "relax.mx3").resolve()),
                "canonical_checkpoint": str(
                    output_root
                    / "shared_relax"
                    / f"base{base:04d}"
                    / "relax.out"
                    / "m_initial.ovf"
                ),
                "reused_300k_thermalized_checkpoint": False,
            },
        )

        source_params0 = read_json(source_run0 / "params.json")
        base_rows.append(
            {
                "selection_rank": selection_rank,
                "base_index": base,
                "split": "test",
                "source_temp_k": SOURCE_TEMP_K,
                "target_temp_k": TARGET_TEMP_K,
                "bz_mt": float(source_params0["bz_mt"]),
                "initial_state": source_params0["initial_state"],
                "anchor_mode": source_params0["anchor_mode"],
                "protocol_family": source_params0["shape"]["family"],
                "relax_thermal_seed": relax_seeds[base],
            }
        )
        seed_rows.append(
            {
                "kind": "shared_relaxation",
                "base_index": base,
                "thermal_repeat_index": None,
                "thermal_seed": relax_seeds[base],
            }
        )

        for repeat in range(REPEATS_PER_BASE):
            source_run = groups[base][repeat]
            branch_seed = branch_seeds[(base, repeat)]
            run_id = target_run_id(source_run.name, branch_seed)
            target_run = output_stage / run_id
            target_run.mkdir()

            source_params = read_json(source_run / "params.json")
            params = update_params(
                source_params,
                source_run=source_run,
                target_root=output_root,
                run_id=run_id,
                base=base,
                repeat=repeat,
                branch_seed=branch_seed,
                selection_rank=selection_rank,
            )
            write_json(target_run / "params.json", params)

            protocol = replace_strings(
                read_json(source_run / "current_protocol.json"),
                [
                    (source_run.name, run_id),
                    ("_T300K_", f"_T{target_temp_token()}K_"),
                ],
            )
            protocol["branch_thermal_seed"] = branch_seed
            protocol["temperature_ood_target_k"] = TARGET_TEMP_K
            write_json(target_run / "current_protocol.json", protocol)
            shutil.copy2(
                source_run / "current_control.csv",
                target_run / "current_control.csv",
            )

            summary = replace_strings(
                read_json(source_run / "result_summary.json"),
                [
                    (source_run.name, run_id),
                    ("_T300K_", f"_T{target_temp_token()}K_"),
                ],
            )
            summary.update(
                {
                    "event_type": f"pending_new_{target_temp_token()}k_test_only_matched_branch",
                    "mumax_finished": False,
                    "n_frames": None,
                    "target_temperature_k": TARGET_TEMP_K,
                    "shared_initial_checkpoint_path": str(
                        output_root
                        / "shared_relax"
                        / f"base{base:04d}"
                        / "relax.out"
                        / "m_initial.ovf"
                    ),
                }
            )
            write_json(target_run / "result_summary.json", summary)

            run_text = (source_run / "run.mx3").read_text(encoding="utf-8")
            (target_run / "run.mx3").write_text(
                transform_mx3(
                    run_text,
                    old_run_id=source_run.name,
                    new_run_id=run_id,
                    thermal_seed=branch_seed,
                    kind="branch",
                ),
                encoding="utf-8",
            )
            (target_run / "shared_checkpoint_path.txt").write_text(
                str(
                    output_root
                    / "shared_relax"
                    / f"base{base:04d}"
                    / "relax.out"
                    / "m_initial.ovf"
                )
                + "\n",
                encoding="utf-8",
            )

            sample_rows = [
                update_sample_row(
                    row,
                    old_run_id=source_run.name,
                    run_id=run_id,
                    target_root=output_root,
                    base=base,
                    repeat=repeat,
                    branch_seed=branch_seed,
                )
                for row in read_jsonl(source_run / "sample_manifest.jsonl")
            ]
            write_jsonl(target_run / "sample_manifest.jsonl", sample_rows)

            manifest = update_manifest_row(
                manifest_source[(base, repeat)],
                target_root=output_root,
                run_id=run_id,
                base=base,
                repeat=repeat,
                branch_seed=branch_seed,
                selection_rank=selection_rank,
            )
            output_manifest.append(manifest)
            branch_paths.append(output_root / run_id)
            seed_rows.append(
                {
                    "kind": "thermal_branch",
                    "base_index": base,
                    "thermal_repeat_index": repeat,
                    "thermal_seed": branch_seed,
                    "run_id": run_id,
                }
            )

            id_link = id_stage / source_run.name
            id_link.symlink_to(source_run.resolve(), target_is_directory=True)
            id_manifest.append(manifest_source[(base, repeat)])

    write_jsonl(output_stage / "run_manifest.jsonl", output_manifest)
    write_csv(output_stage / "run_manifest.csv", output_manifest)
    write_jsonl(output_stage / "thermal_seed_manifest.jsonl", seed_rows)
    write_csv(output_stage / "base_manifest.csv", base_rows)
    (output_stage / "selected_base_ids.txt").write_text(
        "".join(f"{base}\n" for base in bases), encoding="utf-8"
    )
    (output_stage / "base_run_list.txt").write_text(
        "".join(f"{base:04d}\n" for base in bases), encoding="utf-8"
    )
    (output_stage / "branch_run_list.txt").write_text(
        "".join(f"{path}\n" for path in branch_paths), encoding="utf-8"
    )
    write_json(output_stage / "leakage_audit.json", audit)

    source_config = read_json(source_root / "config_used.json")
    source_config.update(
        {
            "output_dir": str(output_root),
            "source_id300_dataset": str(source_root),
            "source_temperature_k": SOURCE_TEMP_K,
            "target_temperature_k": TARGET_TEMP_K,
            "temp_values_k": [TARGET_TEMP_K],
            "num_base_paths": EXPECTED_BASES,
            "num_paths": EXPECTED_BASES * REPEATS_PER_BASE,
            "thermal_repeats_per_base": REPEATS_PER_BASE,
            "selected_source_split": "test",
            "seed_generator_seed": SEED_GENERATOR_SEED,
            "design": f"fresh_{target_temp_token()}k_shared_1ns_relaxation_then_five_fresh_branches",
            "reused_shared_checkpoints": 0,
            "reused_trajectory_outputs": 0,
            "mumax_execution_path": os.environ.get("MUMAX_BIN", "mumax3"),
            "mumax_execution_version": "mumax 3.12 (CUDA-12.0)",
            "mumax_execution_commit": "6e5c98bb",
            "mumax_execution_sha256": (
                "37eff7666a86c349089daaa8a637c4f4270a6a79c9135b8fc19a7115e628d06f"
            ),
            "mumax_compiled_cuda_cc": [61],
        }
    )
    write_json(output_stage / "config_used.json", source_config)

    validation = validate_stage(
        output_stage,
        bases=bases,
        relax_seeds=relax_seeds,
        branch_seeds=branch_seeds,
    )
    write_json(
        output_stage / "dataset_summary.json",
        {
            "status": "inputs_prepared_pending_mumax",
            "design": source_config["design"],
            "source_id300_dataset": str(source_root),
            "matched_id_view": str(id_view_root),
            "source_temperature_k": SOURCE_TEMP_K,
            "target_temperature_k": TARGET_TEMP_K,
            "base_conditions": EXPECTED_BASES,
            "fresh_shared_relaxations": EXPECTED_BASES,
            "fresh_thermal_branches": EXPECTED_BASES * REPEATS_PER_BASE,
            "validation": validation,
        },
    )

    write_jsonl(id_stage / "run_manifest.jsonl", id_manifest)
    write_json(
        id_stage / "view_provenance.json",
        {
            "status": "complete",
            "mode": "read_only_symlink_view",
            "source_dataset": str(source_root),
            "temperature_k": SOURCE_TEMP_K,
            "split": "test",
            "base_conditions": EXPECTED_BASES,
            "trajectories": EXPECTED_BASES * REPEATS_PER_BASE,
            "selected_base_indices": bases,
            "paired_ood_dataset": str(output_root),
        },
    )

    os.rename(output_stage, output_root)
    os.rename(id_stage, id_view_root)
    print(
        json.dumps(
            {
                "status": "prepared",
                "output_root": str(output_root),
                "id_view_root": str(id_view_root),
                "validation": validation,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    global TARGET_TEMP_K, TARGET_TEMPERATURE_INDEX, SEED_GENERATOR_SEED, TARGET_ROLE
    args = parse_args()
    TARGET_TEMP_K = float(args.target_temp_k)
    TARGET_TEMPERATURE_INDEX = (
        int(args.target_temperature_index)
        if args.target_temperature_index is not None
        else int({30.0: 0, 150.0: 1, 300.0: 2, 400.0: 3}.get(TARGET_TEMP_K, -1))
    )
    SEED_GENERATOR_SEED = (
        int(args.seed_generator_seed)
        if args.seed_generator_seed is not None
        else int(TARGET_TEMP_K) * 10_000_000 + 2_026_090_2
    )
    TARGET_ROLE = str(args.target_role)
    source_root = args.source_root.resolve()
    output_root = args.output_root.absolute()
    id_view_root = args.id_view_root.absolute()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    allowed_bases = None
    if args.reuse_leakage_audit is not None:
        cached_audit = read_json(args.reuse_leakage_audit.resolve())
        allowed_bases = {
            int(value) for value in cached_audit["selected_base_indices"]
        }
    groups = discover_source_runs(source_root, allowed_bases=allowed_bases)
    training_roots = tuple(
        path.resolve() for path in (args.training_root or list(TRAINING_ROOTS))
    )
    if args.reuse_leakage_audit is None:
        audit, used = leakage_audit(
            groups,
            training_roots=training_roots,
            x30_root=args.x30_root.resolve(),
        )
    else:
        audit, used = reuse_leakage_audit(
            args.reuse_leakage_audit.resolve(), groups
        )
    bases = sorted(groups)
    relax_seeds, branch_seeds = allocate_seeds(bases, set(used))
    plan = {
        "status": "audit_pass_plan_ready",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "id_view_root": str(id_view_root),
        "source_temperature_k": SOURCE_TEMP_K,
        "target_temperature_k": TARGET_TEMP_K,
        "selected_base_indices": bases,
        "selected_base_count": len(bases),
        "fresh_shared_relaxations": len(relax_seeds),
        "fresh_branches": len(branch_seeds),
        "new_seed_count": len(relax_seeds) + len(branch_seeds),
        "seed_generator_seed": SEED_GENERATOR_SEED,
        "leakage_audit": audit,
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        print("plan_only=true", flush=True)
        return
    materialize(
        source_root,
        output_root,
        id_view_root,
        groups,
        audit,
        relax_seeds,
        branch_seeds,
    )


if __name__ == "__main__":
    main()
