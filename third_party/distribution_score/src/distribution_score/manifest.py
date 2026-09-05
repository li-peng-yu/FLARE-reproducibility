"""Manifest parsing shared by cross-condition evaluation and aggregation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .distance import FORMAL_BLOCKS, FORMAL_SHIFT_RADIUS_PX


@dataclass(frozen=True)
class ConditionSpec:
    condition_id: str
    group: str
    group_label: str
    condition_dir: Path
    model_samples: Path
    anchor: Path
    truth: Path


@dataclass(frozen=True)
class EvaluationManifest:
    path: Path
    conditions: tuple[ConditionSpec, ...]
    group_order: tuple[str, ...]
    group_labels: dict[str, str]


def _resolve(path: str | Path, parent: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else (parent / value).resolve()


def load_manifest(path: str | Path) -> EvaluationManifest:
    """Load and validate the public cross-condition manifest format."""

    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_conditions = payload.get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError("manifest.conditions must be a non-empty list")

    explicit_order: list[str] = []
    labels: dict[str, str] = {}
    for raw_group in payload.get("groups", []):
        if isinstance(raw_group, str):
            name = raw_group
            label = raw_group
        elif isinstance(raw_group, dict):
            name = str(raw_group["name"])
            label = str(raw_group.get("label", name))
        else:
            raise ValueError(f"invalid group definition: {raw_group!r}")
        if name in explicit_order:
            raise ValueError(f"duplicate group: {name}")
        explicit_order.append(name)
        labels[name] = label

    conditions: list[ConditionSpec] = []
    seen_ids: set[str] = set()
    observed_order: list[str] = []
    for raw in raw_conditions:
        condition_id = str(raw["id"])
        group = str(raw["group"])
        if condition_id in seen_ids:
            raise ValueError(f"duplicate condition id: {condition_id}")
        seen_ids.add(condition_id)
        if group not in observed_order:
            observed_order.append(group)
        group_label = str(raw.get("group_label", labels.get(group, group)))
        labels.setdefault(group, group_label)

        condition_dir = _resolve(raw["condition_dir"], path.parent)
        model_samples = _resolve(
            raw.get("model_samples", "model_samples/model_samples_f16.npy"),
            condition_dir,
        )
        anchor = _resolve(
            raw.get("anchor", "repeats/rep_000/run.out/segment_000_end.ovf"),
            condition_dir,
        )
        truth = _resolve(
            raw.get("truth", "repeats/rep_000/run.out/segment_001_end.ovf"),
            condition_dir,
        )
        conditions.append(
            ConditionSpec(
                condition_id=condition_id,
                group=group,
                group_label=group_label,
                condition_dir=condition_dir,
                model_samples=model_samples,
                anchor=anchor,
                truth=truth,
            )
        )

    unknown = set(observed_order) - set(explicit_order)
    group_order = explicit_order + [name for name in observed_order if name in unknown]
    missing = set(explicit_order) - set(observed_order)
    if missing:
        raise ValueError(f"groups contain no conditions: {sorted(missing)}")
    return EvaluationManifest(path, tuple(conditions), tuple(group_order), labels)


def result_directory(
    spec: ConditionSpec,
    output_root: Path | None,
    *,
    blocks: int = FORMAL_BLOCKS,
    shift_radius: int = FORMAL_SHIFT_RADIUS_PX,
) -> Path:
    if output_root is None:
        return spec.condition_dir / (
            f"probability_calibration_{blocks}x{blocks}_shift{shift_radius}"
        )
    return output_root.expanduser().resolve() / spec.condition_id
