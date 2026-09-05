#!/usr/bin/env python3
"""Audit completion and invariants of the 165-base x30 MuMax dataset."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = (
    Path(__file__).resolve().parents[2]
    / "FLARE_dataset/self_consistency_x30/skx_bt_165base_x30_sharedrelax_20260813"
)
RUN_RE = re.compile(r"_base(?P<base>\d+)_tr(?P<repeat>\d+)_")


def main() -> None:
    rows = [json.loads(line) for line in (ROOT / "run_manifest.jsonl").read_text().splitlines()]
    groups: dict[int, set[int]] = defaultdict(set)
    split_bases: dict[str, set[int]] = defaultdict(set)
    seed_counts: Counter[int] = Counter()
    complete = 0
    pending = 0
    invalid: list[str] = []
    new_complete = 0
    new_total = 0
    for row in rows:
        # Manifests retain the generation host path as provenance; resolve the
        # released run by its stable directory name under the selected root.
        run_dir = ROOT / Path(row["run_dir"]).name
        match = RUN_RE.search(run_dir.name)
        if match is None:
            invalid.append(f"bad_name:{run_dir.name}")
            continue
        base = int(row["base_index"])
        repeat = int(row["thermal_repeat_index"])
        groups[base].add(repeat)
        split_bases[str(row["split"])].add(base)
        seed_counts[int(row["thermal_seed"])] += 1
        is_complete = all(
            (run_dir / relative).is_file()
            for relative in ("run.out/m_final.ovf", "run.out/m_initial.ovf", "run.out/table.txt")
        )
        if is_complete:
            complete += 1
        else:
            pending += 1
        if repeat >= 5:
            new_total += 1
            new_complete += int(is_complete)
        if int(match.group("base")) != base or int(match.group("repeat")) != repeat:
            invalid.append(f"name_metadata_mismatch:{run_dir.name}")
        checkpoint = ROOT / "shared_relax" / f"base{base:04d}" / "relax.out" / "m_initial.ovf"
        if not checkpoint.is_file():
            invalid.append(f"missing_checkpoint:base{base:04d}")

    bad_groups = {
        base: sorted(repeats)
        for base, repeats in groups.items()
        if repeats != set(range(30))
    }
    report = {
        "root": str(ROOT),
        "manifest_runs": len(rows),
        "base_groups": len(groups),
        "complete_runs": complete,
        "pending_runs": pending,
        "new_runs_complete": new_complete,
        "new_runs_total": new_total,
        "split_base_counts": {key: len(value) for key, value in sorted(split_bases.items())},
        "unique_thermal_seeds": len(seed_counts),
        "duplicate_thermal_seed_count": sum(count > 1 for count in seed_counts.values()),
        "bad_group_count": len(bad_groups),
        "bad_groups": bad_groups,
        "invalid_count": len(set(invalid)),
        "invalid_examples": sorted(set(invalid))[:20],
        "ok": (
            len(rows) == 4950
            and len(groups) == 165
            and not bad_groups
            and len(seed_counts) == 4950
            and not invalid
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
