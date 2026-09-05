#!/usr/bin/env python3
"""Build a balanced 165-base x 30-branch dataset from the corrected x5 data.

The corrected source dataset has one shared 1 ns relaxation checkpoint per
base and five independent post-checkpoint thermal branches.  This builder:

* selects 11 representative bases from each of the 3 x 5 (T, Bz) cells;
* preserves the source base-level train/val/test split labels;
* reuses the five completed corrected branches with hard-linked run outputs;
* creates 25 new, globally unique thermal branches per selected base; and
* writes complete manifests plus portable pending/reused run lists.

The destination is assembled in a staging directory and atomically renamed
only after all structural and seed checks pass.  Existing destinations are
never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "FLARE_dataset"
DEFAULT_SOURCE = (
    DATASET_ROOT
    / "primary_x5/skx_bt_1000base_x5_sharedrelax_20260805"
)
DEFAULT_OUTPUT = (
    DATASET_ROOT
    / "self_consistency_x30/skx_bt_165base_x30_sharedrelax_20260813"
)

SELECTION_SEED = 20260813
THERMAL_SEED_SEED = 30260813
REPEATS_PER_BASE = 30
REUSED_REPEATS = 5
NEW_REPEATS = REPEATS_PER_BASE - REUSED_REPEATS
BASES_PER_BT_CELL = 11
TEMPERATURES = (30.0, 150.0, 300.0)
BZ_VALUES = (0.0, 8.0, 16.0, 24.0, 32.0)
EXTRA_VAL_CELL = (150.0, 16.0)
EXTRA_TEST_CELLS = {(30.0, 0.0), (300.0, 32.0)}

RUN_RE = re.compile(
    r"^r_(?P<index>\d+)_base(?P<base>\d+)_tr(?P<tr>\d+)_"
    r"(?P<suffix>.*)$"
)
THERMAL_SEED_RE = re.compile(r"ThermSeed\((\d+)\)")
TS_NAME_RE = re.compile(r"_TS\d+_")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="materialize the dataset; without this flag only print the selection plan",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
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


def hardlink_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.readlink(item))
        elif item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(item, target)
        else:
            raise RuntimeError(f"unsupported source entry: {item}")


def discover_runs(source_root: Path) -> dict[int, dict[int, Path]]:
    groups: dict[int, dict[int, Path]] = defaultdict(dict)
    for path in source_root.iterdir():
        if not path.is_dir() or not path.name.startswith("r_"):
            continue
        match = RUN_RE.match(path.name)
        if match is None:
            raise RuntimeError(f"unrecognized run directory: {path}")
        base = int(match.group("base"))
        repeat = int(match.group("tr"))
        if repeat in groups[base]:
            raise RuntimeError(f"duplicate base/repeat: base={base} repeat={repeat}")
        groups[base][repeat] = path
    if len(groups) != 1000:
        raise RuntimeError(f"expected 1000 source bases, found {len(groups)}")
    for base, runs in groups.items():
        if sorted(runs) != list(range(REUSED_REPEATS)):
            raise RuntimeError(f"base {base} does not have exactly repeats 0..4")
        checkpoint = source_root / "shared_relax" / f"base{base:04d}" / "relax.out" / "m_initial.ovf"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        for repeat, run_dir in runs.items():
            required = [
                run_dir / "run.mx3",
                run_dir / "params.json",
                run_dir / "current_protocol.json",
                run_dir / "current_control.csv",
                run_dir / "sample_manifest.jsonl",
                run_dir / "result_summary.json",
                run_dir / "run.out" / "m_final.ovf",
                run_dir / "run.out" / "m_initial.ovf",
                run_dir / "run.out" / "table.txt",
            ]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"incomplete corrected source branch base={base} repeat={repeat}: {missing}"
                )
    return dict(groups)


def base_params(groups: dict[int, dict[int, Path]]) -> dict[int, dict[str, Any]]:
    return {base: read_json(runs[0] / "params.json") for base, runs in groups.items()}


def numeric_features(params: dict[str, Any]) -> list[float]:
    shape = params["shape"]
    j_amp = float(shape["j_amp_a_per_m2"])
    angle = float(shape["angle_rad"])
    orientation = float(params["initial_orientation_rad"])
    phase = float(params["initial_phase_rad"])
    return [
        math.log10(abs(j_amp)),
        float(shape["pulse_s"]) * 1.0e9,
        float(shape["center_x_m"]) * 1.0e9,
        float(shape["center_y_m"]) * 1.0e9,
        float(shape["radius_m"]) * 1.0e9,
        float(shape["width_m"]) * 1.0e9,
        float(shape["aspect_ratio"]),
        math.sin(angle),
        math.cos(angle),
        float(shape["lobe_separation_m"]) * 1.0e9,
        float(shape["secondary_lobe_ratio"]),
        float(params["initial_period_m"]) * 1.0e9,
        float(params["initial_wall_m"]) * 1.0e9,
        float(params["initial_sx"]),
        float(params["initial_sy"]),
        math.sin(orientation),
        math.cos(orientation),
        math.sin(phase),
        math.cos(phase),
        float(params["initial_noise_amp"]),
        float(params["initial_chirality"]),
    ]


def categorical_features(params: dict[str, Any]) -> list[str]:
    shape = params["shape"]
    sign = "positive_J" if float(shape["j_amp_a_per_m2"]) >= 0.0 else "negative_J"
    return [
        f"state={params['initial_state']}",
        f"protocol={shape['family']}",
        f"anchor={params['anchor_mode']}",
        f"sign={sign}",
    ]


def make_feature_matrix(
    params_by_base: dict[int, dict[str, Any]],
) -> tuple[list[int], np.ndarray]:
    base_ids = sorted(params_by_base)
    numeric = np.asarray(
        [numeric_features(params_by_base[base]) for base in base_ids], dtype=np.float64
    )
    means = numeric.mean(axis=0)
    scales = numeric.std(axis=0)
    scales[scales < 1.0e-12] = 1.0
    numeric = (numeric - means) / scales

    category_names = sorted(
        {name for base in base_ids for name in categorical_features(params_by_base[base])}
    )
    category_index = {name: index for index, name in enumerate(category_names)}
    categorical = np.zeros((len(base_ids), len(category_names)), dtype=np.float64)
    for row, base in enumerate(base_ids):
        for name in categorical_features(params_by_base[base]):
            categorical[row, category_index[name]] = 1.5
    return base_ids, np.concatenate([numeric, categorical], axis=1)


def farthest_first(
    candidates: list[int],
    quota: int,
    feature_by_base: dict[int, np.ndarray],
) -> list[int]:
    if quota < 0 or quota > len(candidates):
        raise ValueError(f"invalid quota {quota} for {len(candidates)} candidates")
    if quota == 0:
        return []
    ordered = sorted(candidates)
    matrix = np.stack([feature_by_base[base] for base in ordered])
    centroid = matrix.mean(axis=0)
    centroid_distance = np.sum((matrix - centroid) ** 2, axis=1)
    first_position = min(
        range(len(ordered)), key=lambda index: (centroid_distance[index], ordered[index])
    )
    selected_positions = [first_position]
    selected_set = {first_position}
    min_distance = np.sum((matrix - matrix[first_position]) ** 2, axis=1)
    while len(selected_positions) < quota:
        position = max(
            (index for index in range(len(ordered)) if index not in selected_set),
            key=lambda index: (min_distance[index], -ordered[index]),
        )
        selected_positions.append(position)
        selected_set.add(position)
        distance = np.sum((matrix - matrix[position]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
    return [ordered[position] for position in selected_positions]


def selection_quota(temp_k: float, bz_mt: float, split: str) -> int:
    cell = (temp_k, bz_mt)
    if split == "val":
        return 2 if cell == EXTRA_VAL_CELL else 1
    if split == "test":
        return 2 if cell in EXTRA_TEST_CELLS else 1
    if split == "train":
        return BASES_PER_BT_CELL - selection_quota(temp_k, bz_mt, "val") - selection_quota(
            temp_k, bz_mt, "test"
        )
    raise ValueError(split)


def select_bases(
    params_by_base: dict[int, dict[str, Any]],
) -> tuple[list[int], dict[str, Any]]:
    base_ids, feature_matrix = make_feature_matrix(params_by_base)
    feature_by_base = {
        base: feature_matrix[index] for index, base in enumerate(base_ids)
    }
    pools: dict[tuple[float, float, str], list[int]] = defaultdict(list)
    for base, params in params_by_base.items():
        key = (float(params["temp_k"]), float(params["bz_mt"]), str(params["split"]))
        pools[key].append(base)

    selected: list[int] = []
    strata: list[dict[str, Any]] = []
    for temp_k in TEMPERATURES:
        for bz_mt in BZ_VALUES:
            for split in ("train", "val", "test"):
                key = (temp_k, bz_mt, split)
                candidates = sorted(pools[key])
                quota = selection_quota(temp_k, bz_mt, split)
                chosen = farthest_first(candidates, quota, feature_by_base)
                selected.extend(chosen)
                strata.append(
                    {
                        "temp_k": temp_k,
                        "bz_mt": bz_mt,
                        "split": split,
                        "candidate_count": len(candidates),
                        "quota": quota,
                        "selected_base_indices": sorted(chosen),
                    }
                )

    selected = sorted(selected)
    if len(selected) != 165 or len(set(selected)) != 165:
        raise RuntimeError(f"selection is not 165 unique bases: {len(selected)}")
    bt_counts = Counter(
        (float(params_by_base[base]["temp_k"]), float(params_by_base[base]["bz_mt"]))
        for base in selected
    )
    if set(bt_counts.values()) != {BASES_PER_BT_CELL} or len(bt_counts) != 15:
        raise RuntimeError(f"invalid B-T balance: {bt_counts}")
    split_counts = Counter(str(params_by_base[base]["split"]) for base in selected)
    if split_counts != Counter({"train": 132, "test": 17, "val": 16}):
        raise RuntimeError(f"invalid split counts: {split_counts}")

    discrete = {
        field: dict(
            sorted(
                Counter(
                    (
                        str(params_by_base[base]["shape"]["family"])
                        if field == "protocol_family"
                        else (
                            "positive"
                            if field == "j_sign"
                            and float(params_by_base[base]["shape"]["j_amp_a_per_m2"]) >= 0
                            else (
                                "negative"
                                if field == "j_sign"
                                else str(params_by_base[base][field])
                            )
                        )
                    )
                    for base in selected
                ).items()
            )
        )
        for field in ("initial_state", "anchor_mode", "protocol_family", "j_sign")
    }
    plan = {
        "selection_seed": SELECTION_SEED,
        "algorithm": "split-preserving standardized mixed-feature farthest-first",
        "base_count": len(selected),
        "bases_per_temperature_field_cell": BASES_PER_BT_CELL,
        "split_counts": dict(sorted(split_counts.items())),
        "bt_counts": [
            {"temp_k": temp, "bz_mt": bz, "bases": count}
            for (temp, bz), count in sorted(bt_counts.items())
        ],
        "discrete_counts": discrete,
        "strata": strata,
        "selected_base_indices": selected,
    }
    return selected, plan


def all_used_thermal_seeds(source_root: Path) -> set[int]:
    used: set[int] = set()
    for row in read_jsonl(source_root / "run_manifest.jsonl"):
        used.add(int(row["thermal_seed"]))
    if len(used) != 5000:
        raise RuntimeError(f"source thermal seeds are not globally unique: {len(used)}")
    return used


def allocate_new_seeds(selected_bases: list[int], used: set[int]) -> dict[tuple[int, int], int]:
    rng = random.Random(THERMAL_SEED_SEED)
    allocated: dict[tuple[int, int], int] = {}
    for base in selected_bases:
        for repeat in range(REUSED_REPEATS, REPEATS_PER_BASE):
            while True:
                seed = rng.randrange(1, 2_147_483_647)
                if seed not in used:
                    used.add(seed)
                    allocated[(base, repeat)] = seed
                    break
    if len(allocated) != len(selected_bases) * NEW_REPEATS:
        raise RuntimeError("failed to allocate all new thermal seeds")
    return allocated


def renamed_run_id(template_name: str, base: int, repeat: int, seed: int) -> str:
    match = RUN_RE.match(template_name)
    if match is None:
        raise RuntimeError(f"invalid template run id: {template_name}")
    suffix = match.group("suffix")
    suffix, count = TS_NAME_RE.subn(f"_TS{seed}_", f"_{suffix}", count=1)
    if count != 1:
        raise RuntimeError(f"run id does not contain one TS field: {template_name}")
    suffix = suffix.removeprefix("_")
    run_index = base * REPEATS_PER_BASE + repeat
    return f"r_{run_index:05d}_base{base:04d}_tr{repeat:02d}_{suffix}"


def transform_run_mx3(text: str, old_run_id: str, new_run_id: str, seed: int) -> str:
    text, run_id_count = re.subn(
        rf"(?m)^// run_id={re.escape(old_run_id)};",
        f"// run_id={new_run_id};",
        text,
        count=1,
    )
    if run_id_count != 1:
        raise RuntimeError(f"expected one run-id comment in template {old_run_id}")
    text, seed_count = THERMAL_SEED_RE.subn(f"ThermSeed({seed})", text, count=1)
    if seed_count != 1:
        raise RuntimeError(f"expected one ThermSeed in template {old_run_id}")
    text = text.replace(
        "// All five replicas load the same post-relaxation checkpoint.",
        "// All thirty replicas load the same post-relaxation checkpoint.",
    )
    if text.count(f"ThermSeed({seed})") != 1:
        raise RuntimeError(f"thermal seed replacement failed for {new_run_id}")
    return text


def update_branch_object(
    value: dict[str, Any],
    *,
    source_root: Path,
    output_root: Path,
    old_run_id: str,
    new_run_id: str,
    base: int,
    repeat: int,
    seed: int,
    split: str,
    origin: str,
) -> dict[str, Any]:
    replacements = [
        (str(source_root), str(output_root)),
        (old_run_id, new_run_id),
    ]
    result = replace_strings(value, replacements)
    result["index"] = base * REPEATS_PER_BASE + repeat
    result["run_id"] = new_run_id
    result["base_index"] = base
    result["thermal_repeat_index"] = repeat
    result["thermal_seed"] = seed
    result["seed"] = seed
    result["split"] = split
    result["shared_initial_checkpoint_path"] = str(
        output_root / "shared_relax" / f"base{base:04d}" / "relax.out" / "m_initial.ovf"
    )
    result["trajectory_origin"] = origin
    result["source_corrected_run_id"] = old_run_id
    if isinstance(result.get("current_protocol"), dict):
        result["current_protocol"]["branch_thermal_seed"] = seed
    return result


def copy_shared_checkpoint(
    source_root: Path,
    stage_root: Path,
    output_root: Path,
    base: int,
    selection_rank: int,
) -> None:
    source = source_root / "shared_relax" / f"base{base:04d}"
    target = stage_root / "shared_relax" / f"base{base:04d}"
    target.mkdir(parents=True)
    shutil.copy2(source / "relax.mx3", target / "relax.mx3")
    provenance = replace_strings(
        read_json(source / "checkpoint_provenance.json"),
        [(str(source_root), str(output_root))],
    )
    provenance["curated_selection_rank"] = selection_rank
    provenance["source_corrected_checkpoint"] = str(
        source / "relax.out" / "m_initial.ovf"
    )
    provenance["reuse_method"] = "hardlink_same_filesystem"
    write_json(target / "checkpoint_provenance.json", provenance)
    hardlink_tree(source / "relax.out", target / "relax.out")


def build_branch(
    *,
    source_root: Path,
    stage_root: Path,
    output_root: Path,
    source_run: Path,
    base: int,
    repeat: int,
    seed: int,
    split: str,
    reuse_output: bool,
    root_manifest_template: dict[str, Any],
) -> tuple[Path, dict[str, Any], int]:
    old_run_id = source_run.name
    new_run_id = renamed_run_id(old_run_id, base, repeat, seed)
    target = stage_root / new_run_id
    target.mkdir()
    origin = "reused_corrected_x5_branch" if reuse_output else "new_x30_thermal_branch"

    params = update_branch_object(
        read_json(source_run / "params.json"),
        source_root=source_root,
        output_root=output_root,
        old_run_id=old_run_id,
        new_run_id=new_run_id,
        base=base,
        repeat=repeat,
        seed=seed,
        split=split,
        origin=origin,
    )
    params["source_corrected_run_dir"] = str(source_run)
    write_json(target / "params.json", params)

    protocol = update_branch_object(
        read_json(source_run / "current_protocol.json"),
        source_root=source_root,
        output_root=output_root,
        old_run_id=old_run_id,
        new_run_id=new_run_id,
        base=base,
        repeat=repeat,
        seed=seed,
        split=split,
        origin=origin,
    )
    protocol["branch_thermal_seed"] = seed
    write_json(target / "current_protocol.json", protocol)
    shutil.copy2(source_run / "current_control.csv", target / "current_control.csv")

    summary = update_branch_object(
        read_json(source_run / "result_summary.json"),
        source_root=source_root,
        output_root=output_root,
        old_run_id=old_run_id,
        new_run_id=new_run_id,
        base=base,
        repeat=repeat,
        seed=seed,
        split=split,
        origin=origin,
    )
    if reuse_output:
        summary.update(
            {
                "event_type": "reused_completed_corrected_shared_relaxation_branch",
                "mumax_finished": True,
                "n_frames": len(list((source_run / "run.out").glob("m[0-9]*.ovf"))),
            }
        )
    else:
        summary.update(
            {
                "event_type": "pending_new_x30_shared_relaxation_branch",
                "mumax_finished": False,
                "n_frames": None,
            }
        )
    write_json(target / "result_summary.json", summary)

    run_text = (source_run / "run.mx3").read_text(encoding="utf-8")
    (target / "run.mx3").write_text(
        transform_run_mx3(run_text, old_run_id, new_run_id, seed), encoding="utf-8"
    )
    checkpoint_path = (
        output_root / "shared_relax" / f"base{base:04d}" / "relax.out" / "m_initial.ovf"
    )
    (target / "shared_checkpoint_path.txt").write_text(
        f"{checkpoint_path}\n", encoding="utf-8"
    )

    sample_count = 0
    with (source_run / "sample_manifest.jsonl").open("r", encoding="utf-8") as source_handle, (
        target / "sample_manifest.jsonl"
    ).open("w", encoding="utf-8") as target_handle:
        for line in source_handle:
            if not line.strip():
                continue
            row = update_branch_object(
                json.loads(line),
                source_root=source_root,
                output_root=output_root,
                old_run_id=old_run_id,
                new_run_id=new_run_id,
                base=base,
                repeat=repeat,
                seed=seed,
                split=split,
                origin=origin,
            )
            row["valid_reason"] = (
                "reused_completed_corrected_shared_relaxation"
                if reuse_output
                else "pending_new_x30_shared_relaxation_simulation"
            )
            target_handle.write(json.dumps(row, sort_keys=True))
            target_handle.write("\n")
            sample_count += 1

    if reuse_output:
        hardlink_tree(source_run / "run.out", target / "run.out")
        write_json(
            target / ".reuse_provenance.json",
            {
                "source_run_dir": str(source_run),
                "source_run_id": old_run_id,
                "reuse_method": "hardlink_same_filesystem",
                "thermal_seed": seed,
            },
        )

    manifest = update_branch_object(
        root_manifest_template,
        source_root=source_root,
        output_root=output_root,
        old_run_id=old_run_id,
        new_run_id=new_run_id,
        base=base,
        repeat=repeat,
        seed=seed,
        split=split,
        origin=origin,
    )
    manifest.update(
        {
            "run_dir": str(output_root / new_run_id),
            "mx3_path": str(output_root / new_run_id / "run.mx3"),
            "current_control_path": str(output_root / new_run_id / "current_control.csv"),
            "current_protocol_path": str(output_root / new_run_id / "current_protocol.json"),
            "m_initial_path": str(output_root / new_run_id / "run.out" / "m_initial.ovf"),
            "m_final_path": str(output_root / new_run_id / "run.out" / "m_final.ovf"),
            "m_analytic_initial_path": str(
                output_root
                / "shared_relax"
                / f"base{base:04d}"
                / "relax.out"
                / "m_analytic_initial.ovf"
            ),
            "n_samples_planned": sample_count,
        }
    )
    return output_root / new_run_id, manifest, sample_count


def write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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


def validate_stage(
    *,
    stage_root: Path,
    selected_bases: list[int],
    all_runs: list[Path],
    new_runs: list[Path],
    reused_runs: list[Path],
    seeds: list[int],
    source_groups: dict[int, dict[int, Path]],
) -> dict[str, Any]:
    expected_all = len(selected_bases) * REPEATS_PER_BASE
    expected_new = len(selected_bases) * NEW_REPEATS
    expected_reused = len(selected_bases) * REUSED_REPEATS
    if (len(all_runs), len(new_runs), len(reused_runs)) != (
        expected_all,
        expected_new,
        expected_reused,
    ):
        raise RuntimeError(
            f"invalid run counts: all={len(all_runs)} new={len(new_runs)} reused={len(reused_runs)}"
        )
    if len(seeds) != expected_all or len(set(seeds)) != expected_all:
        raise RuntimeError("target thermal seeds are not globally unique")

    group_counts: Counter[int] = Counter()
    split_run_counts: Counter[str] = Counter()
    sample_counts: Counter[int] = Counter()
    for run_path in all_runs:
        stage_run = stage_root / run_path.name
        params = read_json(stage_run / "params.json")
        base = int(params["base_index"])
        repeat = int(params["thermal_repeat_index"])
        seed = int(params["thermal_seed"])
        group_counts[base] += 1
        split_run_counts[str(params["split"])] += 1
        run_text = (stage_run / "run.mx3").read_text(encoding="utf-8")
        if run_text.count(f"ThermSeed({seed})") != 1:
            raise RuntimeError(f"invalid ThermSeed in {stage_run}")
        checkpoint_relative = f'../shared_relax/base{base:04d}/relax.out/m_initial.ovf'
        if run_text.count(checkpoint_relative) != 1:
            raise RuntimeError(f"invalid checkpoint path in {stage_run}")
        if len(re.findall(r"(?m)^run\(", run_text)) != 2:
            raise RuntimeError(f"expected exactly two run() calls in {stage_run}")
        if repeat < REUSED_REPEATS:
            if not (stage_run / "run.out" / "m_final.ovf").is_file():
                raise RuntimeError(f"reused branch lacks final frame: {stage_run}")
            source_final = source_groups[base][repeat] / "run.out" / "m_final.ovf"
            target_final = stage_run / "run.out" / "m_final.ovf"
            if source_final.stat().st_ino != target_final.stat().st_ino:
                raise RuntimeError(f"reused output is not hard-linked: {stage_run}")
        elif (stage_run / "run.out").exists():
            raise RuntimeError(f"new branch unexpectedly has run.out: {stage_run}")
        with (stage_run / "sample_manifest.jsonl").open("r", encoding="utf-8") as handle:
            sample_count = sum(1 for line in handle if line.strip())
        sample_counts[sample_count] += 1

    if set(group_counts.values()) != {REPEATS_PER_BASE} or len(group_counts) != 165:
        raise RuntimeError(f"invalid target group counts: {group_counts}")
    if split_run_counts != Counter({"train": 3960, "test": 510, "val": 480}):
        raise RuntimeError(f"invalid trajectory split counts: {split_run_counts}")

    for base in selected_bases:
        source_checkpoint = (
            DEFAULT_SOURCE / "shared_relax" / f"base{base:04d}" / "relax.out" / "m_initial.ovf"
        )
        if not source_checkpoint.exists():
            source_checkpoint = (
                source_groups[base][0].parent
                / "shared_relax"
                / f"base{base:04d}"
                / "relax.out"
                / "m_initial.ovf"
            )
        target_checkpoint = (
            stage_root / "shared_relax" / f"base{base:04d}" / "relax.out" / "m_initial.ovf"
        )
        if not target_checkpoint.is_file():
            raise FileNotFoundError(target_checkpoint)

    return {
        "all_runs": len(all_runs),
        "new_runs": len(new_runs),
        "reused_runs": len(reused_runs),
        "base_groups": len(group_counts),
        "repeats_per_base": sorted(set(group_counts.values())),
        "trajectory_split_counts": dict(sorted(split_run_counts.items())),
        "sample_manifest_line_count_histogram": {
            str(key): value for key, value in sorted(sample_counts.items())
        },
        "globally_unique_target_thermal_seeds": len(set(seeds)),
    }


def build_dataset(
    source_root: Path,
    output_root: Path,
    groups: dict[int, dict[int, Path]],
    params_by_base: dict[int, dict[str, Any]],
    selected_bases: list[int],
    selection_plan: dict[str, Any],
) -> None:
    if output_root.exists():
        raise FileExistsError(f"destination already exists: {output_root}")
    stage_root = output_root.parent / f".{output_root.name}.preparing-{os.getpid()}"
    if stage_root.exists():
        raise FileExistsError(f"staging directory already exists: {stage_root}")
    stage_root.mkdir(parents=True)

    source_manifest_rows = read_jsonl(source_root / "run_manifest.jsonl")
    source_manifest: dict[tuple[int, int], dict[str, Any]] = {}
    for row in source_manifest_rows:
        key = (int(row["base_index"]), int(row["thermal_repeat_index"]))
        source_manifest[key] = row
    if len(source_manifest) != 5000:
        raise RuntimeError(f"invalid source manifest size: {len(source_manifest)}")

    used_seeds = all_used_thermal_seeds(source_root)
    new_seed_map = allocate_new_seeds(selected_bases, set(used_seeds))

    all_runs: list[Path] = []
    new_runs: list[Path] = []
    reused_runs: list[Path] = []
    manifest_rows: list[dict[str, Any]] = []
    target_seeds: list[int] = []
    seed_rows: list[dict[str, Any]] = []
    total_samples = 0

    selection_rank_by_base = {base: rank for rank, base in enumerate(selected_bases)}
    for base in selected_bases:
        copy_shared_checkpoint(
            source_root,
            stage_root,
            output_root,
            base,
            selection_rank_by_base[base],
        )

    for selection_rank, base in enumerate(selected_bases):
        split = str(params_by_base[base]["split"])
        for repeat in range(REPEATS_PER_BASE):
            if repeat < REUSED_REPEATS:
                source_repeat = repeat
                source_run = groups[base][repeat]
                seed = int(read_json(source_run / "params.json")["seed"])
                reuse_output = True
            else:
                source_repeat = 0
                source_run = groups[base][source_repeat]
                seed = new_seed_map[(base, repeat)]
                reuse_output = False
            run_path, manifest, sample_count = build_branch(
                source_root=source_root,
                stage_root=stage_root,
                output_root=output_root,
                source_run=source_run,
                base=base,
                repeat=repeat,
                seed=seed,
                split=split,
                reuse_output=reuse_output,
                root_manifest_template=source_manifest[(base, source_repeat)],
            )
            manifest["selection_rank"] = selection_rank
            manifest_rows.append(manifest)
            all_runs.append(run_path)
            target_seeds.append(seed)
            total_samples += sample_count
            if reuse_output:
                reused_runs.append(run_path)
            else:
                new_runs.append(run_path)
            seed_rows.append(
                {
                    "base_index": base,
                    "selection_rank": selection_rank,
                    "thermal_repeat_index": repeat,
                    "thermal_seed": seed,
                    "split": split,
                    "trajectory_origin": manifest["trajectory_origin"],
                    "run_id": run_path.name,
                }
            )
        print(
            f"prepared_base rank={selection_rank + 1}/165 base={base:04d} split={split}",
            flush=True,
        )

    def write_paths(path: Path, values: list[Path]) -> None:
        path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")

    write_paths(stage_root / "run_list.txt", all_runs)
    write_paths(stage_root / "new_run_list.txt", new_runs)
    write_paths(stage_root / "reused_run_list.txt", reused_runs)
    write_jsonl(stage_root / "run_manifest.jsonl", manifest_rows)
    write_manifest_csv(stage_root / "run_manifest.csv", manifest_rows)
    write_jsonl(stage_root / "thermal_seed_manifest.jsonl", seed_rows)
    (stage_root / "selected_base_ids.txt").write_text(
        "".join(f"{base}\n" for base in selected_bases), encoding="utf-8"
    )

    selection_records: list[dict[str, Any]] = []
    for rank, base in enumerate(selected_bases):
        params = params_by_base[base]
        shape = params["shape"]
        selection_records.append(
            {
                "selection_rank": rank,
                "base_index": base,
                "split": params["split"],
                "temp_k": params["temp_k"],
                "bz_mt": params["bz_mt"],
                "initial_state": params["initial_state"],
                "anchor_mode": params["anchor_mode"],
                "protocol_family": shape["family"],
                "j_amp_a_per_m2": shape["j_amp_a_per_m2"],
                "pulse_duration_ns": float(shape["pulse_s"]) * 1.0e9,
                "center_x_nm": float(shape["center_x_m"]) * 1.0e9,
                "center_y_nm": float(shape["center_y_m"]) * 1.0e9,
            }
        )
    selection_plan["selected_bases"] = selection_records
    write_json(stage_root / "selection_manifest.json", selection_plan)

    config = read_json(source_root / "config_used.json")
    config.update(
        {
            "output_dir": str(output_root),
            "num_base_paths": len(selected_bases),
            "num_paths": len(all_runs),
            "thermal_repeats_per_base": REPEATS_PER_BASE,
            "reused_corrected_repeats_per_base": REUSED_REPEATS,
            "new_thermal_repeats_per_base": NEW_REPEATS,
            "source_corrected_dataset": str(source_root),
            "selection_seed": SELECTION_SEED,
            "thermal_seed_generation_seed": THERMAL_SEED_SEED,
            "selection_policy": selection_plan["algorithm"],
            "bt_cell_policy": "exactly_11_bases_per_each_of_3_temperatures_x_5_fields",
            "split_policy": "preserve_source_base_level_split_with_132_train_16_val_17_test",
            "design": "one_shared_1ns_relaxation_then_thirty_independent_thermal_branches",
            "recommended_parallel_workers": 4,
        }
    )
    write_json(stage_root / "config_used.json", config)

    validation = validate_stage(
        stage_root=stage_root,
        selected_bases=selected_bases,
        all_runs=all_runs,
        new_runs=new_runs,
        reused_runs=reused_runs,
        seeds=target_seeds,
        source_groups=groups,
    )
    summary = {
        "design": config["design"],
        "source_corrected_dataset": str(source_root),
        "num_base_paths": len(selected_bases),
        "thermal_repeats_per_base": REPEATS_PER_BASE,
        "num_trajectory_runs": len(all_runs),
        "reused_completed_trajectory_runs": len(reused_runs),
        "new_trajectory_runs_to_compute": len(new_runs),
        "shared_checkpoints_reused": len(selected_bases),
        "shared_initial_relaxation": True,
        "legacy_trajectories_reused": False,
        "corrected_x5_trajectories_reused": True,
        "sample_manifest_rows": total_samples,
        "selection": {
            "bases_per_bt_cell": BASES_PER_BT_CELL,
            "split_counts": selection_plan["split_counts"],
            "discrete_counts": selection_plan["discrete_counts"],
        },
        "validation": validation,
        "parallel_execution": {
            "recommended_workers": 4,
            "tasks": math.ceil(len(new_runs) / 5),
            "runs_per_task": 5,
        },
    }
    write_json(stage_root / "dataset_summary.json", summary)

    os.rename(stage_root, output_root)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"dataset_ready={output_root}", flush=True)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    groups = discover_runs(source_root)
    params_by_base = base_params(groups)
    selected_bases, selection_plan = select_bases(params_by_base)
    print(json.dumps(selection_plan, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        print("plan_only=true", flush=True)
        return
    build_dataset(
        source_root,
        output_root,
        groups,
        params_by_base,
        selected_bases,
        selection_plan,
    )


if __name__ == "__main__":
    main()
