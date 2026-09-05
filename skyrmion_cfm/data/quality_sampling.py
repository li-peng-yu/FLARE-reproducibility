"""Quality-aware sampling for fixed-time micromagnetic frame pairs.

The source trajectories are never modified.  A small, deterministic cache
summarises each control segment, while individual candidate pairs are scored
inside the data-loader worker before they are admitted to a training batch.
This deliberately separates two questions:

* Is a whole control segment almost entirely quiet/noise-dominated?
* Is this particular start/target pair useful for learning coherent motion?

Validation and test datasets do not use this policy; it is a training sampler.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


QUALITY_CACHE_SCHEMA_VERSION = 1

PAIR_CATEGORY_TO_INDEX = {
    "informative": 0,
    "static": 1,
    "noise_only": 2,
}
SEGMENT_CATEGORY_TO_INDEX = dict(PAIR_CATEGORY_TO_INDEX)


def _probability(value: Any, name: str) -> float:
    out = float(value)
    if not math.isfinite(out) or not 0.0 <= out <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
    return out


def _positive(value: Any, name: str, *, allow_zero: bool = True) -> float:
    out = float(value)
    lower_ok = out >= 0.0 if allow_zero else out > 0.0
    if not math.isfinite(out) or not lower_ok:
        comparator = ">=" if allow_zero else ">"
        raise ValueError(f"{name} must be finite and {comparator} 0, got {value!r}")
    return out


def normalize_quality_sampling_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and fill defaults for ``data.quality_sampling``."""

    cfg = dict(raw or {})
    pair_raw = dict(cfg.get("pair", {}) or {})
    segment_raw = dict(cfg.get("segment", {}) or {})

    block_factor = int(cfg.get("block_factor", 8))
    max_attempts = int(cfg.get("max_attempts", 8))
    probes_per_segment = int(segment_raw.get("probes_per_segment", 4))
    if block_factor < 1:
        raise ValueError("data.quality_sampling.block_factor must be >= 1")
    if max_attempts < 1:
        raise ValueError("data.quality_sampling.max_attempts must be >= 1")
    if probes_per_segment < 1:
        raise ValueError(
            "data.quality_sampling.segment.probes_per_segment must be >= 1"
        )

    pair = {
        "static_raw_max_deg": _positive(
            pair_raw.get("static_raw_max_deg", 5.0),
            "data.quality_sampling.pair.static_raw_max_deg",
        ),
        "static_coherent_max_deg": _positive(
            pair_raw.get("static_coherent_max_deg", 5.0),
            "data.quality_sampling.pair.static_coherent_max_deg",
        ),
        "noise_raw_min_deg": _positive(
            pair_raw.get("noise_raw_min_deg", 10.0),
            "data.quality_sampling.pair.noise_raw_min_deg",
        ),
        "noise_coherent_max_deg": _positive(
            pair_raw.get("noise_coherent_max_deg", 8.0),
            "data.quality_sampling.pair.noise_coherent_max_deg",
        ),
        "noise_retention_max": _positive(
            pair_raw.get("noise_retention_max", 0.5),
            "data.quality_sampling.pair.noise_retention_max",
        ),
        "noise_neighbor_min_deg": _positive(
            pair_raw.get("noise_neighbor_min_deg", 60.0),
            "data.quality_sampling.pair.noise_neighbor_min_deg",
        ),
        "noise_block_resultant_max": _positive(
            pair_raw.get("noise_block_resultant_max", 0.5),
            "data.quality_sampling.pair.noise_block_resultant_max",
        ),
        "static_keep_probability": _probability(
            pair_raw.get("static_keep_probability", 0.20),
            "data.quality_sampling.pair.static_keep_probability",
        ),
        "noise_keep_probability": _probability(
            pair_raw.get("noise_keep_probability", 0.05),
            "data.quality_sampling.pair.noise_keep_probability",
        ),
    }
    segment = {
        "probes_per_segment": probes_per_segment,
        "static_raw_p90_max_deg": _positive(
            segment_raw.get("static_raw_p90_max_deg", 7.5),
            "data.quality_sampling.segment.static_raw_p90_max_deg",
        ),
        "static_coherent_p90_max_deg": _positive(
            segment_raw.get("static_coherent_p90_max_deg", 5.0),
            "data.quality_sampling.segment.static_coherent_p90_max_deg",
        ),
        "noise_fraction_min": _probability(
            segment_raw.get("noise_fraction_min", 0.75),
            "data.quality_sampling.segment.noise_fraction_min",
        ),
        "noise_coherent_p90_max_deg": _positive(
            segment_raw.get("noise_coherent_p90_max_deg", 10.0),
            "data.quality_sampling.segment.noise_coherent_p90_max_deg",
        ),
        "static_keep_probability": _probability(
            segment_raw.get("static_keep_probability", 0.25),
            "data.quality_sampling.segment.static_keep_probability",
        ),
        "noise_keep_probability": _probability(
            segment_raw.get("noise_keep_probability", 0.10),
            "data.quality_sampling.segment.noise_keep_probability",
        ),
    }

    cache_path = cfg.get("cache_path")
    if cache_path is not None and not str(cache_path).strip():
        raise ValueError("data.quality_sampling.cache_path cannot be empty")

    return {
        "enabled": bool(cfg.get("enabled", False)),
        "cache_path": None if cache_path is None else str(cache_path),
        "auto_build": bool(cfg.get("auto_build", True)),
        "rebuild_cache": bool(cfg.get("rebuild_cache", False)),
        "block_factor": block_factor,
        "max_attempts": max_attempts,
        # This is an unconditional branch through the original sampler.  It
        # prevents a target-dependent policy from erasing the base conditional
        # distribution, even when both segment and pair gates are aggressive.
        "original_mix_probability": _probability(
            cfg.get("original_mix_probability", 0.10),
            "data.quality_sampling.original_mix_probability",
        ),
        "pair": pair,
        "segment": segment,
    }


def _masked_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


@torch.no_grad()
def pair_quality_metrics(
    sample: dict[str, Any],
    *,
    block_factor: int,
    source_key: str = "m_init",
) -> dict[str, float]:
    """Measure raw and spatially coherent motion for one CHW spin pair.

    The default source is the actual bridge input.  In configurations with a
    Zeeman reference this is the preconditioned state, not the observed frame,
    which aligns the admission decision with the residual velocity the model
    must learn.
    """

    source = sample[source_key].detach().float()
    target = sample["m_t"].detach().float()
    if source.ndim != 3 or target.shape != source.shape or source.shape[0] != 3:
        raise ValueError(
            "quality sampling expects m_init/m_t with matching (3,H,W) shapes, "
            f"got {tuple(source.shape)} and {tuple(target.shape)}"
        )

    raw_mask = sample.get("defect_field")
    if torch.is_tensor(raw_mask):
        weight = raw_mask.detach().float()
        if weight.ndim == 3:
            if weight.shape[0] != 1:
                raise ValueError(
                    "quality sampling expects defect_field shape (1,H,W) or (H,W), "
                    f"got {tuple(weight.shape)}"
                )
            weight = weight[0]
    else:
        weight = torch.ones_like(source[0])
    weight = weight.clamp(0.0, 1.0)
    spin_valid = (source.square().sum(dim=0) > 0.25) & (
        target.square().sum(dim=0) > 0.25
    )
    weight = weight * spin_valid.to(weight.dtype)
    if float(weight.sum()) <= 0.0:
        raise ValueError("quality sampling found no magnetic sites in pair")

    source = F.normalize(source, dim=0, eps=1.0e-8)
    target = F.normalize(target, dim=0, eps=1.0e-8)
    raw_dot = (source * target).sum(dim=0).clamp(-1.0, 1.0)
    raw_angle = _masked_mean(torch.rad2deg(torch.acos(raw_dot)), weight)

    factor = min(int(block_factor), int(source.shape[-2]), int(source.shape[-1]))
    factor = max(1, factor)
    weight4 = weight[None, None]
    pooled_weight = F.avg_pool2d(weight4, kernel_size=factor, stride=factor)

    def pool(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = F.avg_pool2d(
            field[None] * weight4,
            kernel_size=factor,
            stride=factor,
        )
        pooled = pooled / pooled_weight.clamp_min(1.0e-8)
        resultant = pooled.norm(dim=1)
        return F.normalize(pooled, dim=1, eps=1.0e-8), resultant

    source_block, source_resultant = pool(source)
    target_block, target_resultant = pool(target)
    block_dot = (source_block * target_block).sum(dim=1).clamp(-1.0, 1.0)
    block_weight = pooled_weight[:, 0]
    coherent_angle = _masked_mean(
        torch.rad2deg(torch.acos(block_dot))[0], block_weight[0]
    )
    lowfreq_mz_change = _masked_mean(
        (source_block[:, 2] - target_block[:, 2]).abs()[0], block_weight[0]
    )
    source_resultant_mean = _masked_mean(source_resultant[0], block_weight[0])
    target_resultant_mean = _masked_mean(target_resultant[0], block_weight[0])

    target_dot_x = (target[:, :, :-1] * target[:, :, 1:]).sum(dim=0).clamp(-1.0, 1.0)
    target_dot_y = (target[:, :-1, :] * target[:, 1:, :]).sum(dim=0).clamp(-1.0, 1.0)
    weight_x = weight[:, :-1] * weight[:, 1:]
    weight_y = weight[:-1, :] * weight[1:, :]
    target_neighbor_angle = (
        (torch.rad2deg(torch.acos(target_dot_x)) * weight_x).sum()
        + (torch.rad2deg(torch.acos(target_dot_y)) * weight_y).sum()
    ) / (weight_x.sum() + weight_y.sum()).clamp_min(1.0)

    raw_value = float(raw_angle.item())
    coherent_value = float(coherent_angle.item())
    retention = coherent_value / max(raw_value, 1.0e-8)
    return {
        "raw_angle_deg": raw_value,
        "coherent_angle_deg": coherent_value,
        "coherent_retention": retention,
        "lowfreq_mz_change": float(lowfreq_mz_change.item()),
        "source_block_resultant": float(source_resultant_mean.item()),
        "target_block_resultant": float(target_resultant_mean.item()),
        "target_neighbor_angle_deg": float(target_neighbor_angle.item()),
    }


def classify_pair_quality(metrics: dict[str, float], cfg: dict[str, Any]) -> str:
    pair = cfg["pair"]
    raw = float(metrics["raw_angle_deg"])
    coherent = float(metrics["coherent_angle_deg"])
    retention = float(metrics["coherent_retention"])
    if (
        raw <= float(pair["static_raw_max_deg"])
        and coherent <= float(pair["static_coherent_max_deg"])
    ):
        return "static"
    lowpass_noise = (
        raw >= float(pair["noise_raw_min_deg"])
        and coherent <= float(pair["noise_coherent_max_deg"])
        and retention <= float(pair["noise_retention_max"])
    )
    rough_noise = (
        raw >= float(pair["noise_raw_min_deg"])
        and float(metrics["target_neighbor_angle_deg"])
        >= float(pair["noise_neighbor_min_deg"])
        and float(metrics["target_block_resultant"])
        <= float(pair["noise_block_resultant_max"])
    )
    if lowpass_noise or rough_noise:
        return "noise_only"
    return "informative"


def pair_keep_probability(category: str, cfg: dict[str, Any]) -> float:
    if category == "static":
        return float(cfg["pair"]["static_keep_probability"])
    if category == "noise_only":
        return float(cfg["pair"]["noise_keep_probability"])
    return 1.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_segment(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        return {
            "category": "informative",
            "probe_count": 0,
            "raw_angle_p90_deg": float("nan"),
            "coherent_angle_p90_deg": float("nan"),
            "static_fraction": 0.0,
            "noise_fraction": 0.0,
            "informative_fraction": 0.0,
        }
    categories = Counter(str(row["category"]) for row in rows)
    count = len(rows)
    raw_p90 = _percentile([float(row["raw_angle_deg"]) for row in rows], 90.0)
    coherent_p90 = _percentile(
        [float(row["coherent_angle_deg"]) for row in rows], 90.0
    )
    static_fraction = categories["static"] / count
    noise_fraction = categories["noise_only"] / count
    informative_fraction = categories["informative"] / count
    segment = cfg["segment"]
    if (
        raw_p90 <= float(segment["static_raw_p90_max_deg"])
        and coherent_p90 <= float(segment["static_coherent_p90_max_deg"])
    ):
        category = "static"
    elif (
        noise_fraction >= float(segment["noise_fraction_min"])
        and coherent_p90 <= float(segment["noise_coherent_p90_max_deg"])
    ):
        category = "noise_only"
    else:
        category = "informative"
    return {
        "category": category,
        "probe_count": count,
        "raw_angle_p90_deg": raw_p90,
        "coherent_angle_p90_deg": coherent_p90,
        "static_fraction": static_fraction,
        "noise_fraction": noise_fraction,
        "informative_fraction": informative_fraction,
    }


def segment_keep_probability(entry: dict[str, Any] | None, cfg: dict[str, Any]) -> float:
    if not entry:
        return 1.0
    category = str(entry.get("category", "informative"))
    if category == "static":
        return float(cfg["segment"]["static_keep_probability"])
    if category == "noise_only":
        return float(cfg["segment"]["noise_keep_probability"])
    return 1.0


def quality_cache_fingerprint(dataset: Any, cfg: dict[str, Any]) -> str:
    record_ids = [str(record.run_id) for record in dataset.records]
    record_digest = hashlib.sha256("\n".join(record_ids).encode("utf-8")).hexdigest()
    payload = {
        "schema_version": QUALITY_CACHE_SCHEMA_VERSION,
        "record_count": len(record_ids),
        "record_digest": record_digest,
        "t_end_ns": [float(value) for value in dataset.t_end_ns],
        "anchor_mode": str(dataset.anchor_mode),
        "segment_policy": str(dataset.segment_policy),
        "segment_time_range_ns": dataset.segment_time_range_ns,
        "current_time_mode": str(dataset.current_time_mode),
        "zeeman_precondition": {
            "enabled": bool(dataset.zeeman_precondition_enabled),
            "min_cycles": float(dataset.zeeman_precondition_min_cycles),
            "gamma_hz_per_t": float(dataset.zeeman_precondition_gamma_hz_per_t),
            "use_spatial_field": bool(dataset.zeeman_precondition_spatial_field),
            "sign": float(dataset.zeeman_precondition_sign),
        },
        "block_factor": int(cfg["block_factor"]),
        "pair_thresholds": cfg["pair"],
        "segment_thresholds": {
            key: value
            for key, value in cfg["segment"].items()
            if key not in {"static_keep_probability", "noise_keep_probability"}
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evenly_spaced(values: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    if len(values) <= count:
        return values
    indices = np.rint(np.linspace(0, len(values) - 1, count)).astype(np.int64)
    return [values[int(index)] for index in sorted(set(indices.tolist()))]


def _range_probe_starts(lo: int, hi: int) -> tuple[int, ...]:
    if hi <= lo:
        return (int(lo),)
    return tuple(sorted({int(lo), int((lo + hi) // 2), int(hi)}))


def segment_probe_specs(dataset: Any, rec_idx: int, max_probes: int) -> dict[int, list[tuple[int, int]]]:
    """Return representative ``(choice_idx, frame_init)`` pairs per segment."""

    rec = dataset.records[int(rec_idx)]
    grouped: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for choice_idx, ranges in enumerate(dataset._choice_start_ranges[int(rec_idx)]):
        t_end_ns, frame_offset, save_step_ps = dataset._record_choices[int(rec_idx)][
            choice_idx
        ]
        for lo, hi in ranges:
            for frame_init in _range_probe_starts(int(lo), int(hi)):
                if dataset.segment_policy != "none":
                    frame_target = dataset._target_frame_for_duration(
                        int(rec_idx),
                        rec,
                        float(save_step_ps),
                        int(frame_init),
                        float(t_end_ns),
                    )
                    if frame_target is None:
                        frame_target = int(frame_init) + int(frame_offset)
                    segment_idx = dataset._control_segment_index_for_span(
                        int(rec_idx),
                        rec,
                        float(save_step_ps),
                        int(frame_init),
                        int(frame_target),
                    )
                else:
                    segment_idx = 0
                grouped[int(segment_idx)].add((int(choice_idx), int(frame_init)))

    out: dict[int, list[tuple[int, int]]] = {}
    for segment_idx, values in grouped.items():
        # Sorting first by horizon, then start time lets the even subsample see
        # both short/long pairs and early/late windows instead of one cluster.
        ordered = sorted(
            values,
            key=lambda item: (
                float(dataset._record_choices[int(rec_idx)][item[0]][0]),
                int(item[1]),
            ),
        )
        out[int(segment_idx)] = _evenly_spaced(ordered, int(max_probes))
    return out


def build_segment_quality_cache(
    dataset: Any,
    cfg: dict[str, Any],
    path: str | Path,
) -> dict[str, Any]:
    """Probe each control segment and atomically write its quality summary."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = quality_cache_fingerprint(dataset, cfg)
    segments: dict[str, dict[str, Any]] = {}
    segment_categories: Counter[str] = Counter()
    pair_categories: Counter[str] = Counter()
    processed_segments = 0
    processed_pairs = 0
    probes_per_segment = int(cfg["segment"]["probes_per_segment"])
    compute_omega_original = bool(dataset.compute_omega_target)
    dataset.compute_omega_target = False
    try:
        for rec_idx in dataset._valid_record_idx:
            rec = dataset.records[int(rec_idx)]
            rec_rows: dict[str, Any] = {}
            for segment_idx, specs in segment_probe_specs(
                dataset, int(rec_idx), probes_per_segment
            ).items():
                rows: list[dict[str, Any]] = []
                for choice_idx, frame_init in specs:
                    rng = np.random.default_rng(
                        int(dataset.seed)
                        + 1_000_003 * (int(rec_idx) + 1)
                        + 9_973 * (int(segment_idx) + 1)
                        + 101 * (int(choice_idx) + 1)
                        + int(frame_init)
                    )
                    sample = dataset._build_sample(
                        int(rec_idx),
                        int(choice_idx),
                        rng,
                        frame_init=int(frame_init),
                        apply_augment=False,
                    )
                    metrics = pair_quality_metrics(
                        sample,
                        block_factor=int(cfg["block_factor"]),
                    )
                    category = classify_pair_quality(metrics, cfg)
                    rows.append({**metrics, "category": category})
                    pair_categories[category] += 1
                    processed_pairs += 1
                entry = summarize_segment(rows, cfg)
                rec_rows[str(int(segment_idx))] = entry
                segment_categories[str(entry["category"])] += 1
                processed_segments += 1
                if processed_segments == 1 or processed_segments % 250 == 0:
                    print(
                        json.dumps(
                            {
                                "quality_cache": "building",
                                "segments": processed_segments,
                                "pairs": processed_pairs,
                                "run_id": rec.run_id,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            if rec_rows:
                segments[str(rec.run_id)] = rec_rows
    finally:
        dataset.compute_omega_target = compute_omega_original

    payload = {
        "schema_version": QUALITY_CACHE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "completed": True,
        "config": cfg,
        "summary": {
            "records": len(segments),
            "segments": processed_segments,
            "probe_pairs": processed_pairs,
            "segment_categories": dict(segment_categories),
            "pair_categories": dict(pair_categories),
        },
        "segments": segments,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "quality_cache": "complete",
                "path": str(output),
                **payload["summary"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return payload


def load_segment_quality_cache(
    dataset: Any,
    cfg: dict[str, Any],
    path: str | Path,
) -> dict[str, Any]:
    cache_path = Path(path)
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != QUALITY_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"quality cache schema mismatch for {cache_path}: "
            f"expected {QUALITY_CACHE_SCHEMA_VERSION}, got {payload.get('schema_version')!r}"
        )
    expected = quality_cache_fingerprint(dataset, cfg)
    actual = str(payload.get("fingerprint", ""))
    if actual != expected:
        raise ValueError(
            f"quality cache fingerprint mismatch for {cache_path}; rebuild the cache"
        )
    if not bool(payload.get("completed", False)):
        raise ValueError(f"quality cache is incomplete: {cache_path}")
    return payload
