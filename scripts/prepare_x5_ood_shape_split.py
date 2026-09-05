#!/usr/bin/env python3
"""Create a base-group-safe geometry-OOD split for the x5 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _base_key(record: Any) -> str:
    value = record.params.get("base_index")
    if value is None:
        raise ValueError(f"record has no base_index: {record.run_id}")
    return f"base{int(value):04d}"


def _shape_family(record: Any) -> str:
    shape = record.params.get("shape", {})
    value = shape.get("family") if isinstance(shape, dict) else None
    if not value:
        raise ValueError(f"record has no shape.family: {record.run_id}")
    return str(value)


def _digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-family", default="ring")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--id-test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    if args.val_fraction <= 0 or args.id_test_fraction <= 0:
        raise ValueError("validation and ID-test fractions must be positive")
    if args.val_fraction + args.id_test_fraction >= 1:
        raise ValueError("validation and ID-test fractions leave no training groups")

    with args.index_cache.open("rb") as handle:
        cached = pickle.load(handle)
    records = cached.get("records") if isinstance(cached, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError(f"invalid trajectory index cache: {args.index_cache}")

    groups: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        groups[_base_key(record)].append(record)
    family_by_group: dict[str, str] = {}
    for key, values in groups.items():
        families = {_shape_family(record) for record in values}
        if len(families) != 1:
            raise ValueError(f"base group {key} spans shape families: {families}")
        family_by_group[key] = next(iter(families))

    holdout = sorted(
        key for key, family in family_by_group.items() if family == args.holdout_family
    )
    in_domain = sorted(set(groups) - set(holdout))
    if not holdout or len(in_domain) < 3:
        raise RuntimeError("geometry holdout split is empty or underdetermined")
    rng = np.random.default_rng(args.seed)
    shuffled = list(np.asarray(in_domain)[rng.permutation(len(in_domain))])
    val_count = max(1, round(len(shuffled) * args.val_fraction))
    id_test_count = max(1, round(len(shuffled) * args.id_test_fraction))
    val_groups = sorted(str(value) for value in shuffled[:val_count])
    id_test_groups = sorted(
        str(value) for value in shuffled[val_count : val_count + id_test_count]
    )
    train_groups = sorted(
        str(value) for value in shuffled[val_count + id_test_count :]
    )

    def run_ids(base_groups: list[str]) -> list[str]:
        return sorted(str(record.run_id) for key in base_groups for record in groups[key])

    train_ids = run_ids(train_groups)
    val_ids = run_ids(val_groups)
    id_test_ids = run_ids(id_test_groups)
    ood_ids = run_ids(holdout)
    test_ids = sorted(id_test_ids + ood_ids)
    all_ids = train_ids + val_ids + test_ids
    if len(all_ids) != len(set(all_ids)) or len(all_ids) != len(records):
        raise RuntimeError("split does not form an exact partition of the trajectory index")

    payload = {
        "schema": "x5_geometry_ood_split_v1",
        "seed": args.seed,
        "holdout": {"metadata_key": "shape.family", "value": args.holdout_family},
        "splits": {"train": train_ids, "val": val_ids, "test": test_ids},
        "test_strata": {"id_unseen_base": id_test_ids, "ood_geometry": ood_ids},
        "base_groups": {
            "train": train_groups,
            "val": val_groups,
            "id_test": id_test_groups,
            "ood_test": holdout,
        },
        "counts": {
            "records": {
                "train": len(train_ids),
                "val": len(val_ids),
                "test": len(test_ids),
                "id_test": len(id_test_ids),
                "ood_test": len(ood_ids),
            },
            "base_groups": {
                "train": len(train_groups),
                "val": len(val_groups),
                "id_test": len(id_test_groups),
                "ood_test": len(holdout),
            },
        },
        "integrity": {
            "all_run_ids_sha256": _digest(sorted(all_ids)),
            "train_run_ids_sha256": _digest(train_ids),
            "val_run_ids_sha256": _digest(val_ids),
            "test_run_ids_sha256": _digest(test_ids),
        },
        "source_index_cache": str(args.index_cache),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload["counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
