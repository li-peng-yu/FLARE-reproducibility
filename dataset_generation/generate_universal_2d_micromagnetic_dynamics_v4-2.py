#!/usr/bin/env python3
"""
Generate MuMax3 trajectories from the v4 universal 2D micromagnetic dataset spec.

The input JSON is intentionally a sampling schema, not only a flat config.  This
generator keeps the reference script's run-directory contract while adding a
segment-level metadata contract:

    run.mx3
    params.json
    trajectory_metadata.json
    segments.json
    segment_schedule.csv
    drive_protocol.json
    sample_manifest.jsonl
    result_summary.json

Every trajectory stores the accepted top-level modes, the sampled/derived
reference physical parameters, every condition segment's drive parameters, and
the analytic temperature schedule parameters. Per-segment temperatures are
midpoint diagnostics; non-isothermal schedules are rendered by internal MuMax3
substeps without increasing the metadata segment count.
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
import struct
import time
import zlib
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


CONFIG: Dict[str, Any] = {
    "spec_path": "Universal_2D_Micromagnetic_Dynamics_Dataset_Spec_v4_generator_CN-3.json",
    "output_dir": "universal_2d_micromagnetic_v4_runs",
    "num_paths": 1000,
    "random_seed": 20260628,
    "split_ratios": (0.8, 0.1, 0.1),
    "max_attempt_factor": 50,
    "quota_max_accepted_proposal_factor": 30,
    "quota_overflow_policy": "discard",
    "enforce_distribution_quota": True,
    "copy_spec_json": True,
    "run_batch_id": "",
    "fail_if_output_exists": False,
    "enable_sot_trajectories": True,
    "force_dataset_profile": "",
    "force_drive_type": "",

    # MuMax3 numerics.
    "solver": 5,
    "max_dt_s": 1e-12,
    "max_err": 1e-6,
    "output_format": "OVF2_BINARY",
    "edge_smooth": 8,
    "save_dt_s": 250e-12,
    "final_relax_autostop": True,
    "final_relax_check_dt_s": None,
    "final_relax_max_checks": 4096,
    "min_save_dt_s": 1e-13,
    "table_save_divisor": 4.0,
    "time_delay_steps": [1, 2, 4, 8, 16],

    # Region approximation for local field/current masks in metadata and mx3.
    "control_grid": (8, 8),
    "local_profile_region_threshold": 1e-4,
    "local_profile_min_px": 8.0,
    "local_profile_min_rmin_factor": 2.0,
    "local_profile_min_region_fraction": 0.25,

    # Internal renderer substeps for analytic temperature schedules. These are
    # not metadata segments; they only approximate time-varying T in run.mx3.
    "temperature_schedule_render_substeps": {
        "isothermal": 1,
        "warmup_hold": 12,
        "anneal_hold": 16,
        "warmup_anneal_cycle_per_cycle": 12,
        "quench_relax": 16,
        "local_temperature_spot": 16,
    },

    # Conservative defaults for drive models rendered to MuMax3.
    "default_pol": 0.62,
    "default_lambda_sl": 1.0,
    "default_epsilon_prime": 0.0,
    "fixed_layer_position": "FIXEDLAYER_TOP",
    # Effective spin-Hall SOT rendered through the mathematically equivalent
    # Lambda=1 Slonczewski basis.  The sampled theta_DL_eff normally comes
    # from the spec; this value is only a backwards-compatible fallback.
    "default_sot_theta_dl_eff": 0.20,
    "sot_max_effective_field_T": 0.50,
    "sot_max_rotation_per_step_rad": 0.02,
    "sot_min_max_dt_s": 1.0e-15,

}


MU0 = 1.2566370614359173e-6
HBAR_J_S = 1.054571817e-34
ELEMENTARY_CHARGE_C = 1.602176634e-19
GAMMA_LL_RAD_PER_T_S = 1.7595e11
T_REF_K = 300.0
MIN_TC_ABOVE_TREF_K = 50.0
MAIN_MIN_LENGTH_SCALE_CELLS = 3.0
EDGE_MIN_LENGTH_SCALE_CELLS = 2.0
MAIN_DMI_PERIOD_CELLS_MIN = 32.0
EDGE_DMI_PERIOD_CELLS_MIN = 18.0
FINITE_OBJECT_GEOMETRIES = {"disk", "ellipse", "ring", "smooth_polygon"}
GAUSSIAN_LOCAL_PROFILES = {
    "gaussian_spot",
    "gaussian_local_channel",
    "disk_gaussian_spot",
    "disk_local_field",
    "multi_spot",
    "two_lobe_multi_channel",
    "two_lobe_multi_spot",
}
STRIPE_LOCAL_PROFILES = {"stripe_antenna", "stripe_electrode", "local_antenna"}
DISK_LOCAL_PROFILES = {"disk_contact", "ellipse", "ellipse_contact"}
RING_LOCAL_PROFILES = {"ring", "ring_contact"}
MASK_RENDERED_GEOMETRIES = {"nanostrip", "notched_strip", "antidot_or_holes", "smooth_polygon"}
MX3_APPROXIMATED_GEOMETRIES = set()
SOT_DRIVE_TYPES = {"sot", "weak_current", "strong_current", "local_write_delete"}
# Legacy markers are retained only so already-generated proxy trajectories can
# still be detected and quarantined deterministically.
SOT_PROXY_DRIVE_TYPES = set(SOT_DRIVE_TYPES)
SOT_PROXY_RENDERED_DRIVE_TYPE = "sot_like_slonczewski_proxy"
SOT_PROXY_TORQUE_MODEL = "slonczewski_fixed_layer_proxy"
SOT_RENDERED_DRIVE_TYPE = "sot"
SOT_RENDERED_TORQUE_MODEL = "effective_spin_hall_sot_via_slonczewski_lambda1"
MASK_IMAGE_ENCODING = "mumax_imageshape_dark_inside_v1"
AFFECTED_REASON_LEGACY_MASK = "legacy_mask_polarity_inverted"
AFFECTED_REASON_SOT_PROXY = "sot_like_slonczewski_proxy"
DRIVE_TYPE_RENDER_ALIASES = {
    "ac_field": "field_pulse_or_local_field",
    "sinc_or_gaussian_field_pulse": "field_pulse_or_local_field",
    "local_antenna": "field_pulse_or_local_field",
    "sot": SOT_RENDERED_DRIVE_TYPE,
    "weak_current": SOT_RENDERED_DRIVE_TYPE,
    "strong_current": SOT_RENDERED_DRIVE_TYPE,
    "local_write_delete": SOT_RENDERED_DRIVE_TYPE,
}
SPATIAL_PROFILE_RENDER_ALIASES = {
    "multi_spot": "gaussian_spot",
    "two_lobe_multi_channel": "gaussian_local_channel",
    "two_lobe_multi_spot": "gaussian_spot",
    "disk_gaussian_spot": "gaussian_spot",
    "ellipse": "disk_contact",
    "ellipse_contact": "disk_contact",
    "local_antenna": "stripe_antenna",
    "stripe_electrode": "stripe_antenna",
}
ENABLE_STANDARD_MX3_MAGNETOELASTIC = False
ENABLE_STANDARD_MX3_DEFECT_DISORDER = False
PMA_NO_DMI_BUBBLE_MIN_THICKNESS_M = 1.0e-9
PMA_NO_DMI_BUBBLE_QK_RANGE = (1.05, 1.5)
PMA_NO_DMI_BUBBLE_GEOMETRIES = {"full_rectangle", "disk", "ellipse", "smooth_polygon"}
QUOTA_FIELD_TO_METADATA = {
    "DATASET_PROFILE": "dataset_profile",
    "MATERIAL_FAMILY": "material_family",
    "GEOMETRY_MODE": "geometry_mode",
    "DRIVE_TYPE": "drive_type",
    "TIME_REGIME": "TIME_REGIME",
    "INIT_FAMILY": "init_family",
}


class SpecSampler:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.trace: List[Dict[str, Any]] = []

    def sample(self, node: Dict[str, Any], path: str = "") -> Any:
        dist = node.get("dist")
        if dist == "fixed":
            value = deepcopy(node.get("value"))
            self._record(path, dist, value)
            return value
        if dist == "categorical":
            item = weighted_choice(node.get("items", []), self.rng)
            value = deepcopy(item.get("value"))
            self._record(path, dist, value)
            return value
        if dist == "uniform":
            lo, hi = as_range(node)
            value = self.rng.uniform(lo, hi)
            self._record(path, dist, value)
            return value
        if dist == "log_uniform":
            lo, hi = as_range(node)
            if lo <= 0 or hi <= 0:
                raise ValueError(f"log_uniform requires positive bounds at {path}: {lo}, {hi}")
            value = 10 ** self.rng.uniform(math.log10(lo), math.log10(hi))
            self._record(path, dist, value)
            return value
        if dist == "integer_uniform":
            lo, hi = as_range(node)
            value = self.rng.randint(int(math.ceil(lo)), int(math.floor(hi)))
            self._record(path, dist, value)
            return value
        if dist == "mixture":
            component = weighted_choice(node.get("components", []), self.rng)
            value = self.sample(component["sample"], path)
            self._record(path, dist, value, component.get("component_name"))
            return value
        if dist == "axis_distribution":
            value = self._sample_axis_distribution(node, path)
            self._record(path, dist, value)
            return value
        raise ValueError(f"Unsupported distribution at {path}: {dist}")

    def sample_params(self, params: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, node in params.items():
            if isinstance(node, dict) and "dist" in node:
                if node["dist"] in {"derived", "derived_range_uniform"}:
                    continue
                out[key] = self.sample(node, f"{path}.{key}" if path else key)
            elif isinstance(node, dict):
                out[key] = deepcopy(node)
            else:
                out[key] = deepcopy(node)
        return out

    def _sample_axis_distribution(self, node: Dict[str, Any], path: str) -> Dict[str, Any]:
        mode = node.get("mode")
        if mode == "in_plane_with_optional_tilt":
            az_deg = float(self.sample(node["azimuth_deg"], f"{path}.azimuth_deg"))
            tilt_deg = float(self.sample(node["tilt_from_plane_deg"], f"{path}.tilt_from_plane_deg"))
            if self.rng.random() < 0.5:
                tilt_deg = -tilt_deg
            az = math.radians(az_deg)
            tilt = math.radians(tilt_deg)
            vec = (math.cos(az) * math.cos(tilt), math.sin(az) * math.cos(tilt), math.sin(tilt))
            return {
                "mode": mode,
                "azimuth_deg": az_deg,
                "tilt_from_plane_deg": tilt_deg,
                "vector": normalize(vec),
            }
        if mode == "near_z_or_minus_z":
            az_deg = float(self.sample(node["azimuth_deg"], f"{path}.azimuth_deg"))
            tilt_deg = float(self.sample(node["tilt_deg"], f"{path}.tilt_deg"))
            polarity = str(self.sample(node["polarity"], f"{path}.polarity"))
            sign = -1.0 if polarity == "-z" else 1.0
            az = math.radians(az_deg)
            tilt = math.radians(tilt_deg)
            vec = (math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), sign * math.cos(tilt))
            return {
                "mode": mode,
                "azimuth_deg": az_deg,
                "tilt_deg": tilt_deg,
                "polarity": polarity,
                "vector": normalize(vec),
            }
        raise ValueError(f"Unsupported axis_distribution mode at {path}: {mode}")

    def _record(self, path: str, dist: str, value: Any, component: str | None = None) -> None:
        entry = {"path": path, "dist": dist, "value": value}
        if component is not None:
            entry["component"] = component
        self.trace.append(entry)


def as_range(node: Dict[str, Any]) -> Tuple[float, float]:
    lo, hi = node["range"]
    return float(lo), float(hi)


def weighted_choice(items: Sequence[Dict[str, Any]], rng: random.Random) -> Dict[str, Any]:
    if not items:
        raise ValueError("Cannot sample from an empty categorical/mixture list.")
    total = sum(float(item.get("weight", 1.0)) for item in items)
    if total <= 0:
        raise ValueError("At least one sampled item must have positive weight.")
    pick = rng.random() * total
    acc = 0.0
    last = items[-1]
    for item in items:
        acc += float(item.get("weight", 1.0))
        if pick <= acc:
            return item
    return last


def weighted_count_targets(total_count: int, items: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    positive = [(str(item["value"]), float(item.get("weight", 1.0))) for item in items if float(item.get("weight", 1.0)) > 0.0]
    if total_count <= 0 or not positive:
        return {value: 0 for value, _weight in positive}
    total_weight = sum(weight for _value, weight in positive)
    raw = [(value, total_count * weight / total_weight) for value, weight in positive]
    out = {value: int(math.floor(count)) for value, count in raw}
    remaining = total_count - sum(out.values())
    remainders = sorted(((count - math.floor(count), value) for value, count in raw), reverse=True)
    for _frac, value in remainders[:remaining]:
        out[value] += 1
    return dict(sorted(out.items()))


def distribution_control_node(spec_json: Dict[str, Any]) -> Dict[str, Any]:
    return spec_json.get("random_combination_semantics", {}).get("accepted_trajectory_distribution_control", {})


def quota_targets_from_spec(spec_json: Dict[str, Any], num_paths: int) -> Dict[str, Any]:
    control = distribution_control_node(spec_json)
    primary_field = str(control.get("primary_quota_by", "DATASET_PROFILE"))
    profile_targets: Dict[str, int] = {}
    material_by_profile_targets: Dict[str, Dict[str, int]] = {}
    if primary_field == "DATASET_PROFILE":
        profile_targets = weighted_count_targets(num_paths, spec_json["dataset_profile"]["sample"].get("items", []))
    secondary_fields = [str(field) for field in control.get("secondary_quota_by", [])]
    if "MATERIAL_FAMILY" in secondary_fields and profile_targets:
        by_profile = spec_json["material_family"]["sample_by_profile"]
        default_node = by_profile.get("core_relax")
        for profile, count in profile_targets.items():
            node = by_profile.get(profile, default_node)
            material_by_profile_targets[profile] = weighted_count_targets(count, node.get("items", []))
    return {
        "enabled": bool(control.get("enabled", False)),
        "primary_field": primary_field,
        "secondary_fields": secondary_fields,
        "target_counts": {
            "DATASET_PROFILE": profile_targets,
            "DATASET_PROFILE x MATERIAL_FAMILY": material_by_profile_targets,
        },
    }


def add_counts(dst: Dict[str, int], src: Dict[str, Any]) -> None:
    for key, value in src.items():
        try:
            dst[str(key)] = dst.get(str(key), 0) + int(value)
        except (TypeError, ValueError):
            continue


class AcceptedTrajectoryQuota:
    def __init__(self, spec_json: Dict[str, Any], num_paths: int):
        self.targets = quota_targets_from_spec(spec_json, num_paths)
        self.enabled = bool(self.targets.get("enabled"))
        self.profile_counts: Dict[str, int] = {}
        self.material_counts_by_profile: Dict[str, Dict[str, int]] = {}
        self.overflow_reason_counts: Dict[str, int] = {}
        self.overflow_counts: Dict[str, Dict[str, int]] = {
            "DATASET_PROFILE": {},
            "MATERIAL_FAMILY": {},
            "GEOMETRY_MODE": {},
            "DRIVE_TYPE": {},
            "TIME_REGIME": {},
        }
        self.overflow_proposal_attempt_count = 0
        self.overflow_rejected_proposal_count = 0
        self.overflow_rejection_reason_counts: Dict[str, int] = {}
        self.overflow_primary_rejection_reason_counts: Dict[str, int] = {}

    def acceptance_rejection_reason(self, meta: Dict[str, Any]) -> str | None:
        if not self.enabled:
            return None
        profile = str(meta["dataset_profile"])
        profile_targets = self.targets["target_counts"].get("DATASET_PROFILE", {})
        if profile in profile_targets and self.profile_counts.get(profile, 0) >= int(profile_targets.get(profile, 0)):
            return "primary_dataset_profile_bucket_full"
        material_targets = self.targets["target_counts"].get("DATASET_PROFILE x MATERIAL_FAMILY", {})
        family = str(meta["material_family"])
        profile_material_targets = material_targets.get(profile)
        if profile_material_targets is not None and family in profile_material_targets:
            family_target = int(profile_material_targets.get(family, 0))
            if self.material_counts_by_profile.get(profile, {}).get(family, 0) >= family_target:
                return "secondary_material_family_bucket_full"
        return None

    def record_accept(self, meta: Dict[str, Any]) -> None:
        profile = str(meta["dataset_profile"])
        family = str(meta["material_family"])
        self.profile_counts[profile] = self.profile_counts.get(profile, 0) + 1
        profile_material_counts = self.material_counts_by_profile.setdefault(profile, {})
        profile_material_counts[family] = profile_material_counts.get(family, 0) + 1

    def record_overflow(self, meta: Dict[str, Any], reason: str) -> None:
        self.overflow_reason_counts[reason] = self.overflow_reason_counts.get(reason, 0) + 1
        for quota_field, meta_field in QUOTA_FIELD_TO_METADATA.items():
            if quota_field not in self.overflow_counts:
                continue
            value = str(meta.get(meta_field, ""))
            self.overflow_counts[quota_field][value] = self.overflow_counts[quota_field].get(value, 0) + 1
        self.overflow_proposal_attempt_count += int(meta.get("proposal_attempt_count", 0) or 0)
        self.overflow_rejected_proposal_count += int(meta.get("rejected_proposal_count", 0) or 0)
        add_counts(self.overflow_rejection_reason_counts, meta.get("rejection_reason_counts", {}))
        add_counts(self.overflow_primary_rejection_reason_counts, meta.get("primary_rejection_reason_counts", {}))

    def summary(self) -> Dict[str, Any]:
        target_counts = self.targets.get("target_counts", {})
        material_targets = target_counts.get("DATASET_PROFILE x MATERIAL_FAMILY", {})
        material_actual = {
            profile: dict(sorted(counts.items()))
            for profile, counts in sorted(self.material_counts_by_profile.items())
        }
        underfilled_profile = {
            profile: {"target": target, "accepted": self.profile_counts.get(profile, 0)}
            for profile, target in sorted(target_counts.get("DATASET_PROFILE", {}).items())
            if self.profile_counts.get(profile, 0) < target
        }
        underfilled_material: Dict[str, Dict[str, Dict[str, int]]] = {}
        for profile, targets in sorted(material_targets.items()):
            for family, target in sorted(targets.items()):
                accepted = self.material_counts_by_profile.get(profile, {}).get(family, 0)
                if accepted < target:
                    underfilled_material.setdefault(profile, {})[family] = {"target": target, "accepted": accepted}
        return {
            "enabled": self.enabled,
            "primary_field": self.targets.get("primary_field"),
            "secondary_fields": self.targets.get("secondary_fields", []),
            "overflow_policy": "discard",
            "target_counts": target_counts,
            "accepted_counts": {
                "DATASET_PROFILE": dict(sorted(self.profile_counts.items())),
                "DATASET_PROFILE x MATERIAL_FAMILY": material_actual,
            },
            "overflow_counts": {
                field: dict(sorted(counts.items()))
                for field, counts in sorted(self.overflow_counts.items())
                if counts
            },
            "overflow_reason_counts": dict(sorted(self.overflow_reason_counts.items())),
            "overflow_proposal_attempt_count": self.overflow_proposal_attempt_count,
            "overflow_rejected_proposal_count": self.overflow_rejected_proposal_count,
            "overflow_rejection_reason_counts": dict(sorted(self.overflow_rejection_reason_counts.items())),
            "overflow_primary_rejection_reason_counts": dict(sorted(self.overflow_primary_rejection_reason_counts.items())),
            "underfilled_buckets": {
                "DATASET_PROFILE": underfilled_profile,
                "DATASET_PROFILE x MATERIAL_FAMILY": underfilled_material,
            },
        }


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize(v: Sequence[float]) -> List[float]:
    n = math.sqrt(sum(float(x) * float(x) for x in v))
    if n <= 0:
        return [0.0, 0.0, 1.0]
    return [float(x) / n for x in v]


def random_unit_sphere(rng: random.Random) -> List[float]:
    z = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2.0 * math.pi)
    r = math.sqrt(max(0.0, 1.0 - z * z))
    return [r * math.cos(phi), r * math.sin(phi), z]


def random_in_plane(rng: random.Random) -> List[float]:
    phi = rng.uniform(0.0, 2.0 * math.pi)
    return [math.cos(phi), math.sin(phi), 0.0]


def near_z_direction(rng: random.Random, max_tilt_deg: float = 10.0) -> List[float]:
    sign = -1.0 if rng.random() < 0.5 else 1.0
    tilt = math.radians(rng.uniform(0.0, max_tilt_deg))
    phi = rng.uniform(0.0, 2.0 * math.pi)
    return normalize([math.sin(tilt) * math.cos(phi), math.sin(tilt) * math.sin(phi), sign * math.cos(tilt)])


def direction_from_spec(value: str, rng: random.Random, reference: Sequence[float] | None = None) -> List[float]:
    if value in {"+z_or_-z", "near_plus_or_minus_z_cone_0_10deg"}:
        return near_z_direction(rng, 10.0)
    if value == "in_plane_random":
        return random_in_plane(rng)
    if value == "random_unit_sphere":
        return random_unit_sphere(rng)
    if value == "+x_or_-x":
        return [1.0 if rng.random() < 0.5 else -1.0, 0.0, 0.0]
    if value == "+y_or_-y":
        return [0.0, 1.0 if rng.random() < 0.5 else -1.0, 0.0]
    if value == "random_in_plane":
        return random_in_plane(rng)
    if value == "z_cross_J_direction":
        j = normalize(reference or random_in_plane(rng))
        return normalize([-j[1], j[0], 0.0])
    if value == "random_unit_or_tilted":
        if rng.random() < 0.5:
            return random_unit_sphere(rng)
        base = random_in_plane(rng)
        tilt = math.radians(rng.uniform(-45.0, 45.0))
        return normalize([base[0] * math.cos(tilt), base[1] * math.cos(tilt), math.sin(tilt)])
    return normalize(reference or [0.0, 0.0, 1.0])


def safe_token(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:.4g}"
    else:
        text = str(value)
    return (
        text.replace("-", "m")
        .replace("+", "p")
        .replace(".", "p")
        .replace("/", "_")
        .replace(" ", "_")
    )


def sign_value(value: Any, default: int = 1) -> int:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return 1 if float(value) >= 0 else -1
    text = str(value).strip().lower()
    if text in {"+", "+1", "1", "plus", "+z"}:
        return 1
    if text in {"-", "-1", "minus", "-z"}:
        return -1
    return default


def make_time_seed() -> int:
    return (time.time_ns() ^ (os.getpid() << 16)) % (2**31 - 1)


def safe_identifier(text: str) -> str:
    ident = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    ident = ident.strip("._-")
    if not ident:
        raise ValueError("identifier is empty after sanitization")
    return ident


def make_timestamp_random_id(prefix: str = "batch") -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    rnd = random.SystemRandom().randrange(0, 10**8)
    return safe_identifier(f"{prefix}_{stamp}_{rnd:08d}")


def derive_thermal_noise_seed(seed: int, index: int, attempt: int) -> int:
    """Derive a positive MuMax3 thermal-noise seed from the trajectory seed."""
    value = (int(seed) * 1103515245 + 12345 + 104729 * int(index) + 1009 * int(attempt)) % (2**31 - 1)
    return value or 1


def assign_split(num_paths: int, ratios: Sequence[float], rng: random.Random) -> List[str]:
    n_train = int(round(float(ratios[0]) * num_paths))
    n_val = int(round(float(ratios[1]) * num_paths))
    n_test = num_paths - n_train - n_val
    labels = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    rng.shuffle(labels)
    return labels


def parse_int_pair(text: str | None, default: Sequence[int] | Tuple[int, int], name: str) -> Tuple[int, int]:
    if text is None:
        return int(default[0]), int(default[1])
    cleaned = text.lower().replace("x", ",").replace("*", ",")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"{name} must look like 8x8 or 8,8.")
    a, b = int(parts[0]), int(parts[1])
    if a <= 0 or b <= 0:
        raise ValueError(f"{name} entries must be positive.")
    return a, b


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_input_path(path: str | Path) -> Path:
    p = Path(path)
    if p.exists() or p.is_absolute():
        return p
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / p,
        script_dir / p.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return p


def load_override_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(f"YAML override requested but PyYAML is not installed: {path}") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Override file must contain an object at top level: {path}")
    return data


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    unknown = sorted(set(override) - set(base))
    if unknown:
        raise ValueError(f"Unknown override keys: {unknown}")
    for key, value in override.items():
        out[key] = value
    return out


def grid_from_spec(spec_json: Dict[str, Any]) -> Dict[str, Any]:
    grid_node = spec_json["global_grid_and_normalization"]["grid"]
    nx = int(grid_node["Nx"]["value"])
    ny = int(grid_node["Ny"]["value"])
    nz = int(grid_node["Nz"]["value"])
    cell_nm = float(grid_node["cell_nm"]["value"])
    dx_m = cell_nm * 1e-9
    dy_m = dx_m
    return {
        "Nx": nx,
        "Ny": ny,
        "Nz": nz,
        "cell_nm": cell_nm,
        "dx_m": dx_m,
        "dy_m": dy_m,
        "dz_m": None,
        "Lx_m": nx * dx_m,
        "Ly_m": ny * dy_m,
    }


def length_resolution_metrics(ms: float, aex: float, ku: float, anis_u: Sequence[float], dx_m: float) -> Dict[str, Any]:
    lex = math.sqrt(2.0 * aex / (MU0 * ms * ms)) if ms > 0 and aex > 0 else None
    near_z = abs(float(anis_u[2])) >= 0.5 if len(anis_u) >= 3 else False
    keff = ku - 0.5 * MU0 * ms * ms if near_z else None
    delta_dw = math.sqrt(aex / keff) if aex > 0 and keff and keff > 0 else None
    delta_cells = delta_dw / dx_m if delta_dw and dx_m > 0 else None
    wall_cells = math.pi * delta_dw / dx_m if delta_dw and dx_m > 0 else None
    length_scales = [x for x in (lex, delta_dw) if x is not None]
    min_length = min(length_scales) if length_scales else None
    return {
        "lex_m": lex,
        "rex": lex / dx_m if lex and dx_m > 0 else None,
        "Keff_J_per_m3": keff,
        "Delta_DW_m": delta_dw,
        "Delta_DW_cells": delta_cells,
        "domain_wall_width_cells": wall_cells,
        "min_physical_length_m": min_length,
        "r_min_cells": min_length / dx_m if min_length and dx_m > 0 else None,
    }


def dmi_period_cells(aex: float, d_ref: float, dx_m: float) -> float | None:
    if aex <= 0 or dx_m <= 0 or d_ref == 0.0:
        return None
    return 4.0 * math.pi * aex / abs(d_ref) / dx_m


def classify_min_length_resolution(r_min: float | None) -> str:
    if r_min is None or not math.isfinite(r_min) or r_min < EDGE_MIN_LENGTH_SCALE_CELLS:
        return "reject"
    if r_min < MAIN_MIN_LENGTH_SCALE_CELLS:
        return "edge_stress_only"
    return "main"


def classify_optional_min_length_resolution(value: float | None) -> str:
    if value is None:
        return "not_applicable"
    return classify_min_length_resolution(value)


def classify_dmi_period_resolution(ld_cells: float | None) -> str:
    if ld_cells is None:
        return "not_applicable"
    if not math.isfinite(ld_cells) or ld_cells < EDGE_DMI_PERIOD_CELLS_MIN:
        return "reject"
    if ld_cells < MAIN_DMI_PERIOD_CELLS_MIN:
        return "edge_stress_only"
    return "main"


def combine_resolution_partition(*classes: str) -> str:
    if any(cls == "reject" for cls in classes):
        return "reject"
    if any(cls == "edge_stress_only" for cls in classes):
        return "edge_stress_only"
    return "main"


def build_resolution_policy(
    normalized: Dict[str, Any],
    segments: Sequence[Dict[str, Any]] | None = None,
    extra_materials: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    ref_rex = normalized.get("rex")
    ref_delta_cells = normalized.get("Delta_DW_cells_ref")
    ref_wall_cells = normalized.get("domain_wall_width_cells_ref")
    ref_r_min = normalized.get("r_min_ref")
    ref_ld_cells = normalized.get("LD_cells_ref")
    segment_rex_values = [
        float(seg["instantaneous_material"]["rex_T"])
        for seg in segments or []
        if seg.get("instantaneous_material", {}).get("rex_T") is not None
    ]
    segment_delta_values = [
        float(seg["instantaneous_material"]["Delta_DW_cells_T"])
        for seg in segments or []
        if seg.get("instantaneous_material", {}).get("Delta_DW_cells_T") is not None
    ]
    segment_wall_values = [
        float(seg["instantaneous_material"]["domain_wall_width_cells_T"])
        for seg in segments or []
        if seg.get("instantaneous_material", {}).get("domain_wall_width_cells_T") is not None
    ]
    segment_r_min_values = [
        float(seg["instantaneous_material"]["r_min_T"])
        for seg in segments or []
        if seg.get("instantaneous_material", {}).get("r_min_T") is not None
    ]
    segment_ld_values = [
        float(seg["instantaneous_material"]["LD_cells_T"])
        for seg in segments or []
        if seg.get("instantaneous_material", {}).get("LD_cells_T") is not None
    ]
    region_materials = [
        value.get("instantaneous_material") or {}
        for seg in segments or []
        for value in seg.get("region_material_values", [])
    ]
    probe_materials = list(extra_materials or [])
    region_rex_values = [
        float(inst["rex_T"])
        for inst in [*region_materials, *probe_materials]
        if inst.get("rex_T") is not None
    ]
    region_delta_values = [
        float(inst["Delta_DW_cells_T"])
        for inst in [*region_materials, *probe_materials]
        if inst.get("Delta_DW_cells_T") is not None
    ]
    region_wall_values = [
        float(inst["domain_wall_width_cells_T"])
        for inst in [*region_materials, *probe_materials]
        if inst.get("domain_wall_width_cells_T") is not None
    ]
    region_r_min_values = [
        float(inst["r_min_T"])
        for inst in [*region_materials, *probe_materials]
        if inst.get("r_min_T") is not None
    ]
    region_ld_values = [
        float(inst["LD_cells_T"])
        for inst in [*region_materials, *probe_materials]
        if inst.get("LD_cells_T") is not None
    ]
    all_rex_values = [float(ref_rex)] if ref_rex is not None else []
    all_rex_values.extend(segment_rex_values)
    all_rex_values.extend(region_rex_values)
    all_delta_values = [float(ref_delta_cells)] if ref_delta_cells is not None else []
    all_delta_values.extend(segment_delta_values)
    all_delta_values.extend(region_delta_values)
    all_wall_values = [float(ref_wall_cells)] if ref_wall_cells is not None else []
    all_wall_values.extend(segment_wall_values)
    all_wall_values.extend(region_wall_values)
    all_r_min_values = [float(ref_r_min)] if ref_r_min is not None else []
    all_r_min_values.extend(segment_r_min_values)
    all_r_min_values.extend(region_r_min_values)
    all_ld_values = [float(ref_ld_cells)] if ref_ld_cells is not None else []
    all_ld_values.extend(segment_ld_values)
    all_ld_values.extend(region_ld_values)

    min_rex = min(all_rex_values) if all_rex_values else None
    min_delta_cells = min(all_delta_values) if all_delta_values else None
    min_wall_cells = min(all_wall_values) if all_wall_values else None
    min_r_min = min(all_r_min_values) if all_r_min_values else None
    min_ld_cells = min(all_ld_values) if all_ld_values else None
    exchange_class = classify_optional_min_length_resolution(min_rex)
    wall_class = classify_optional_min_length_resolution(min_delta_cells)
    min_length_class = classify_min_length_resolution(min_r_min)
    dmi_class = classify_dmi_period_resolution(min_ld_cells)
    partition = combine_resolution_partition(min_length_class, dmi_class)
    return {
        "dataset_resolution_partition": partition,
        "min_length_resolution_class": min_length_class,
        "exchange_resolution_class": exchange_class,
        "pma_domain_wall_resolution_class": wall_class,
        "dmi_period_resolution_class": dmi_class,
        "rex_ref": ref_rex,
        "rex_min_over_segments": min_rex,
        "Delta_DW_cells_ref": ref_delta_cells,
        "Delta_DW_cells_min_over_segments": min_delta_cells,
        "domain_wall_width_cells_ref": ref_wall_cells,
        "domain_wall_width_cells_min_over_segments": min_wall_cells,
        "r_min_ref": ref_r_min,
        "r_min_over_segments": min_r_min,
        "LD_cells_ref": ref_ld_cells,
        "LD_cells_min_over_segments": min_ld_cells,
        "main_r_min_cells": MAIN_MIN_LENGTH_SCALE_CELLS,
        "edge_r_min_cells": EDGE_MIN_LENGTH_SCALE_CELLS,
        "main_LD_cells_min": MAIN_DMI_PERIOD_CELLS_MIN,
        "edge_LD_cells_min": EDGE_DMI_PERIOD_CELLS_MIN,
    }


def resolution_rejection_reasons(
    policy: Dict[str, Any],
    profile: str,
    allow_stress_partition: bool = False,
) -> List[str]:
    reasons: List[str] = []
    if policy["min_length_resolution_class"] == "reject":
        reasons.append("min_physical_length_under_resolved_r_min_below_edge_threshold")
    if policy["dmi_period_resolution_class"] == "reject":
        reasons.append("dmi_period_under_resolved_LD_below_edge_threshold")
    if (
        policy["dataset_resolution_partition"] == "edge_stress_only"
        and profile != "edge_rare"
        and not allow_stress_partition
    ):
        reasons.append("resolution_edge_stress_only_requires_edge_rare_or_stress")
    return reasons


def resolution_validity_tags(policy: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    if policy["min_length_resolution_class"] == "edge_stress_only":
        tags.append("min_length_resolution_edge_validity")
    if policy["exchange_resolution_class"] == "edge_stress_only":
        tags.append("exchange_resolution_edge_validity")
    if policy["pma_domain_wall_resolution_class"] == "edge_stress_only":
        tags.append("pma_domain_wall_resolution_edge_validity")
    if policy["dmi_period_resolution_class"] == "edge_stress_only":
        tags.append("dmi_period_resolution_edge_validity")
    if policy["dataset_resolution_partition"] == "edge_stress_only":
        tags.append("edge_stress_resolution_only")
    return tags


def material_rejection_reasons(raw: Dict[str, Any], normalized: Dict[str, Any], family: str, profile: str) -> List[str]:
    reasons: List[str] = []
    thickness = raw["thickness_m"]
    if not (0.3e-9 <= thickness <= 2e-9):
        reasons.append("thickness_outside_0p3_2nm")
    if raw["Ms_ref_A_per_m"] <= 0:
        reasons.append("Ms_ref_nonpositive")
    if raw["A_ref_J_per_m"] <= 0:
        reasons.append("A_ref_nonpositive")
    if raw["alpha_ref"] <= 0:
        reasons.append("alpha_ref_nonpositive")
    if family == "pma_idmi":
        if not (normalized.get("Keff_ref_J_per_m3") is not None and normalized["Keff_ref_J_per_m3"] > 0):
            reasons.append("pma_idmi_requires_positive_Keff")
        if not normalized.get("has_Dc_ref", False):
            reasons.append("pma_idmi_requires_valid_Dc")
        if normalized.get("LD_cells_ref") is None or float(normalized["LD_cells_ref"]) < EDGE_DMI_PERIOD_CELLS_MIN:
            reasons.append("pma_idmi_requires_LD_cells_ge_edge_threshold")
    if family == "bulk_dmi_2d" and (
        normalized.get("LD_cells_ref") is None or float(normalized["LD_cells_ref"]) < EDGE_DMI_PERIOD_CELLS_MIN
    ):
        reasons.append("bulk_dmi_requires_LD_cells_ge_edge_threshold")
    reasons.extend(resolution_rejection_reasons(build_resolution_policy(normalized), profile, allow_stress_partition=True))
    return reasons


def record_rejection(reason_counts: Dict[str, int], primary_counts: Dict[str, int], reasons: Sequence[str]) -> None:
    if not reasons:
        return
    primary_counts[reasons[0]] = primary_counts.get(reasons[0], 0) + 1
    for reason in reasons:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1


def sample_dataset_profile(spec_json: Dict[str, Any], sampler: SpecSampler) -> str:
    return str(sampler.sample(spec_json["dataset_profile"]["sample"], "dataset_profile.sample"))


def sample_material_family(spec_json: Dict[str, Any], profile: str, sampler: SpecSampler) -> str:
    node = spec_json["material_family"]["sample_by_profile"].get(
        profile, spec_json["material_family"]["sample_by_profile"]["core_relax"]
    )
    return str(sampler.sample(node, f"material_family.sample_by_profile.{profile}"))


def resolve_bulk_axis(choice: str, rng: random.Random) -> Dict[str, Any]:
    if choice == "near_z":
        vec = near_z_direction(rng, 8.0)
    elif choice == "in_plane":
        vec = random_in_plane(rng)
    else:
        vec = random_unit_sphere(rng)
    return {"mode": choice, "vector": vec}


def sample_alpha_ref(spec_json: Dict[str, Any], profile: str, sampler: SpecSampler) -> float:
    alpha_node = spec_json["global_grid_and_normalization"]["sampled_global_scalars"]["alpha_ref"]
    node = alpha_node.get(profile, alpha_node["default_profiles"])
    return float(sampler.sample(node, f"global_grid_and_normalization.sampled_global_scalars.alpha_ref.{profile}"))


def sample_theta_t(spec_json: Dict[str, Any], sampler: SpecSampler) -> float:
    node = spec_json["global_grid_and_normalization"]["sampled_global_scalars"]["theta_t"]["sample"]
    return float(sampler.sample(node, "global_grid_and_normalization.sampled_global_scalars.theta_t"))


def compute_material_derived(
    raw: Dict[str, Any],
    family: str,
    profile: str,
    spec_json: Dict[str, Any],
    sampler: SpecSampler,
    grid: Dict[str, Any],
    rng: random.Random,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    tags: List[str] = []
    ms = float(raw["Ms_ref_A_per_m"])
    aex = float(raw["A_ref_J_per_m"])
    qk = float(raw.get("qK_ref", 0.0))
    ku = 0.5 * qk * MU0 * ms * ms
    alpha = sample_alpha_ref(spec_json, profile, sampler)
    theta_t = sample_theta_t(spec_json, sampler)

    uK_ref = raw.get("uK_ref", {"vector": [0.0, 0.0, 1.0]})
    if isinstance(uK_ref, str):
        uK_ref = resolve_bulk_axis(uK_ref, rng)
        raw["uK_ref"] = uK_ref
    anis_u = normalize(uK_ref.get("vector", [0.0, 0.0, 1.0]))
    length_metrics = length_resolution_metrics(ms, aex, ku, anis_u, grid["dx_m"])
    lex = float(length_metrics["lex_m"] or 0.0)
    thickness = theta_t * lex
    keff = length_metrics["Keff_J_per_m3"]
    delta_dw = length_metrics["Delta_DW_m"]
    dc = 4.0 * math.sqrt(aex * keff) / math.pi if keff and keff > 0 else None

    dmi_type = str(raw.get("DMI_TYPE", "none"))
    d_sign = sign_value(raw.get("D_sign", 1))
    d_ref = float(raw.get("D_ref_J_per_m2", 0.0) or 0.0)
    ld_cells_ref = None
    d_over_dc = 0.0

    if dmi_type == "interfacial":
        if family == "pma_idmi" and dc:
            d_over_dc = float(raw.get("d_D_over_Dc_ref", 0.0))
            d_ref = d_sign * d_over_dc * dc
        elif family == "q1_stripe_competition":
            mode = str(raw.get("D_sampling_mode_if_interfacial", "LD_cells"))
            if mode == "d_over_Dc_when_valid" and dc:
                d_over_dc = float(raw.get("d_D_over_Dc_ref_if_valid", 0.0))
                d_ref = d_sign * d_over_dc * dc
            else:
                ld_cells_ref = float(raw.get("LD_cells_ref_if_Keff_nonpositive_or_LD_mode", 48.0))
                d_ref = d_sign * 4.0 * math.pi * aex / (ld_cells_ref * grid["dx_m"])
                d_over_dc = abs(d_ref) / dc if dc else 0.0
        else:
            d_ref = 0.0
    elif dmi_type == "bulk":
        ld_cells_ref = float(raw.get("LD_cells_ref", 48.0))
        d_ref = d_sign * 4.0 * math.pi * aex / (ld_cells_ref * grid["dx_m"])
        d_over_dc = abs(d_ref) / dc if dc else 0.0
        tags.append("effective_2d_bulk_dmi")

    if d_ref != 0.0:
        ld_cells_ref = dmi_period_cells(aex, d_ref, grid["dx_m"])

    normalized = {
        "lex_ref_m": lex,
        "rex": lex / grid["dx_m"],
        "qK_ref": qk,
        "theta_t": theta_t,
        "kappa_D_ref": d_ref / (MU0 * ms * ms * lex) if lex > 0 else 0.0,
        "LD_cells_ref": ld_cells_ref,
        "has_Dc_ref": bool(dc and dc > 0),
        "d_D_over_Dc_ref": d_over_dc,
        "qC1_ref": 0.0,
        "qC2_ref": 0.0,
        "qME_ref": 0.0,
        "Keff_ref_J_per_m3": keff,
        "Delta_DW_ref_m": delta_dw,
        "Delta_DW_cells_ref": length_metrics["Delta_DW_cells"],
        "domain_wall_width_cells_ref": length_metrics["domain_wall_width_cells"],
        "r_min_ref": length_metrics["r_min_cells"],
        "Dc_ref_J_per_m2": dc,
    }
    raw_out = {
        "Ms_ref_A_per_m": ms,
        "A_ref_J_per_m": aex,
        "Ku_ref_J_per_m3": ku,
        "D_ref_J_per_m2": d_ref,
        "alpha_ref": alpha,
        "thickness_m": thickness,
        "Kc1_ref_J_per_m3": 0.0,
        "Kc2_ref_J_per_m3": 0.0,
        "B1_ref_J_per_m3": 0.0,
        "B2_ref_J_per_m3": 0.0,
        "qK_ref": qk,
        "uK_ref": uK_ref,
        "anis_u": anis_u,
        "DMI_TYPE": dmi_type,
    }
    if theta_t > 0.4:
        tags.append("thinfilm_edge_validity")
    return raw_out, normalized, tags


def material_is_valid(
    raw: Dict[str, Any],
    normalized: Dict[str, Any],
    family: str = "",
    profile: str = "edge_rare",
) -> bool:
    return not material_rejection_reasons(raw, normalized, family, profile)


def sample_material(
    spec_json: Dict[str, Any],
    profile: str,
    sampler: SpecSampler,
    grid: Dict[str, Any],
    rng: random.Random,
) -> Tuple[str, Dict[str, Any], Dict[str, Any], List[str]]:
    family = sample_material_family(spec_json, profile, sampler)
    family_node = spec_json["material_family"]["families"][family]
    raw_sample = sampler.sample_params(
        family_node["sampled_parameters"],
        f"material_family.families.{family}.sampled_parameters",
    )
    raw, normalized, tags = compute_material_derived(raw_sample, family, profile, spec_json, sampler, grid, rng)
    return family, raw, normalized, tags


def sample_temperature_params(
    spec_json: Dict[str, Any],
    family: str,
    sampler: SpecSampler,
) -> Tuple[str, Dict[str, Any]]:
    root = spec_json["temp_param_mode"]
    mode = str(sampler.sample(root["sample"], "temp_param_mode.sample"))
    tc_node = root["common_parameters"]["Tc_K"].get(family, root["common_parameters"]["Tc_K"]["default"])
    sampled_tc = float(sampler.sample(tc_node, f"temp_param_mode.Tc_K.{family}"))
    tc = max(sampled_tc, T_REF_K + MIN_TC_ABOVE_TREF_K)
    params: Dict[str, Any] = {"TEMP_PARAM_MODE": mode, "Tc_K": tc}
    if mode == "temp_params_off":
        params.update({
            "beta_Ms": 0.0,
            "m_red_min": 1.0,
            "m_red_max": 1.0,
            "p_A": 0.0,
            "p_Ku": 0.0,
            "p_D": 0.0,
            "p_Kc": 0.0,
            "p_Kc2": 0.0,
            "p_B1": 0.0,
            "p_B2": 0.0,
            "c_alpha": 0.0,
            "alpha_min": 0.0,
            "alpha_max": 10.0,
        })
        return mode, params
    mode_node = root["modes"][mode]
    if mode == "family_power_law":
        params.update(sampler.sample_params(mode_node["variables_by_family"][family], f"temp_param_mode.{mode}.{family}"))
        params.update(sampler.sample_params(mode_node["shared_variables"], f"temp_param_mode.{mode}.shared"))
    else:
        params.update(sampler.sample_params(mode_node["variables"], f"temp_param_mode.{mode}.variables"))
    params.setdefault("p_Kc", params.get("p_Ku", 0.0))
    params.setdefault("p_Kc2", params.get("p_Kc", 0.0))
    params.setdefault("p_B1", 1.0)
    params.setdefault("p_B2", 1.0)
    params.setdefault("m_red_min", 0.08)
    params.setdefault("m_red_max", 1.3)
    params.setdefault("alpha_min", 0.001)
    params.setdefault("alpha_max", 0.9)
    return mode, params


def temperature_reduced(T_K: float, temp_params: Dict[str, Any]) -> float:
    if temp_params["TEMP_PARAM_MODE"] == "temp_params_off":
        return 1.0
    tc = float(temp_params["Tc_K"])
    beta = float(temp_params.get("beta_Ms", 0.35))
    tref = T_REF_K
    den = max(1e-9, 1.0 - tref / tc)
    num = max(1e-9, 1.0 - T_K / tc)
    m_red = (num / den) ** beta
    return clamp(m_red, float(temp_params.get("m_red_min", 0.08)), float(temp_params.get("m_red_max", 1.3)))


def instantaneous_params(T_K: float, raw: Dict[str, Any], temp_params: Dict[str, Any]) -> Dict[str, Any]:
    if temp_params["TEMP_PARAM_MODE"] == "temp_params_off":
        return {
            "T_K": T_K,
            "m_reduced": 1.0,
            "Ms_T_A_per_m": raw["Ms_ref_A_per_m"],
            "A_T_J_per_m": raw["A_ref_J_per_m"],
            "Ku_T_J_per_m3": raw["Ku_ref_J_per_m3"],
            "D_T_J_per_m2": raw["D_ref_J_per_m2"],
            "alpha_T": raw["alpha_ref"],
            "Kc1_T_J_per_m3": raw["Kc1_ref_J_per_m3"],
            "Kc2_T_J_per_m3": raw["Kc2_ref_J_per_m3"],
            "B1_T_J_per_m3": raw["B1_ref_J_per_m3"],
            "B2_T_J_per_m3": raw["B2_ref_J_per_m3"],
        }
    mred = temperature_reduced(T_K, temp_params)
    alpha = raw["alpha_ref"] * (1.0 + float(temp_params.get("c_alpha", 0.0)) * (T_K - T_REF_K) / T_REF_K)
    alpha = clamp(alpha, float(temp_params.get("alpha_min", 0.001)), float(temp_params.get("alpha_max", 0.9)))
    return {
        "T_K": T_K,
        "m_reduced": mred,
        "Ms_T_A_per_m": raw["Ms_ref_A_per_m"] * mred,
        "A_T_J_per_m": raw["A_ref_J_per_m"] * (mred ** float(temp_params.get("p_A", 2.0))),
        "Ku_T_J_per_m3": raw["Ku_ref_J_per_m3"] * (mred ** float(temp_params.get("p_Ku", 3.0))),
        "D_T_J_per_m2": raw["D_ref_J_per_m2"] * (mred ** float(temp_params.get("p_D", 1.5))),
        "alpha_T": alpha,
        "Kc1_T_J_per_m3": raw["Kc1_ref_J_per_m3"] * (mred ** float(temp_params.get("p_Kc", 3.0))),
        "Kc2_T_J_per_m3": raw["Kc2_ref_J_per_m3"] * (mred ** float(temp_params.get("p_Kc2", 3.0))),
        "B1_T_J_per_m3": raw["B1_ref_J_per_m3"] * (mred ** float(temp_params.get("p_B1", 2.0))),
        "B2_T_J_per_m3": raw["B2_ref_J_per_m3"] * (mred ** float(temp_params.get("p_B2", 2.0))),
    }


def attach_resolution_to_instantaneous_material(
    inst: Dict[str, Any],
    raw: Dict[str, Any],
    grid: Dict[str, Any],
) -> None:
    anis_u = raw.get("anis_u", [0.0, 0.0, 1.0])
    metrics = length_resolution_metrics(
        float(inst["Ms_T_A_per_m"]),
        float(inst["A_T_J_per_m"]),
        float(inst["Ku_T_J_per_m3"]),
        anis_u,
        float(grid["dx_m"]),
    )
    inst["lex_T_m"] = metrics["lex_m"]
    inst["rex_T"] = metrics["rex"]
    inst["Keff_T_J_per_m3"] = metrics["Keff_J_per_m3"]
    inst["Delta_DW_T_m"] = metrics["Delta_DW_m"]
    inst["Delta_DW_cells_T"] = metrics["Delta_DW_cells"]
    inst["domain_wall_width_cells_T"] = metrics["domain_wall_width_cells"]
    inst["r_min_T"] = metrics["r_min_cells"]
    inst["LD_cells_T"] = dmi_period_cells(
        float(inst["A_T_J_per_m"]),
        float(inst["D_T_J_per_m2"]),
        float(grid["dx_m"]),
    )


def attach_instantaneous_resolution(segments: Sequence[Dict[str, Any]], raw: Dict[str, Any], grid: Dict[str, Any]) -> None:
    for seg in segments:
        inst = seg["instantaneous_material"]
        attach_resolution_to_instantaneous_material(inst, raw, grid)


def sample_time_regime(
    spec_json: Dict[str, Any],
    profile: str,
    sampler: SpecSampler,
    cfg: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    root = spec_json["time_regime"]
    node = root["by_profile"].get(profile, root["global_target"])
    mode = str(sampler.sample(node, f"time_regime.by_profile.{profile}"))
    mode_node = root["modes"][mode]
    if mode == "final_relax":
        T_end_ns = float(sampler.sample(mode_node["T_end_ns_timeout"], f"time_regime.modes.{mode}.T_end_ns_timeout"))
        torque_threshold = float(sampler.sample(mode_node["torque_threshold"], f"time_regime.modes.{mode}.torque_threshold"))
        energy_slope_threshold = float(
            sampler.sample(mode_node["energy_slope_threshold"], f"time_regime.modes.{mode}.energy_slope_threshold")
        )
    else:
        T_end_ns = float(sampler.sample(mode_node["T_end_ns"], f"time_regime.modes.{mode}.T_end_ns"))
        torque_threshold = None
        energy_slope_threshold = None
    T_end_s = T_end_ns * 1e-9
    save_dt_s = max(float(cfg["min_save_dt_s"]), float(cfg["save_dt_s"]))
    saved = max(1, int(math.floor((T_end_s + 1e-18) / save_dt_s)) + 1)
    autostop_enabled = bool(cfg.get("final_relax_autostop", True)) and mode == "final_relax"
    return mode, {
        "time_regime": mode,
        "T_end_ns": T_end_ns,
        "T_end_s": T_end_s,
        "condition_T_end_ns": T_end_ns,
        "condition_T_end_s": T_end_s,
        "condition_start_ns": 0.0,
        "condition_start_s": 0.0,
        "saved_frame_count": saved,
        "save_dt_s": save_dt_s,
        "table_dt_s": max(float(cfg["min_save_dt_s"]), save_dt_s / float(cfg["table_save_divisor"])),
        "pre_relaxation_enabled": False,
        "pre_relaxation_duration_ns": 0.0,
        "pre_relaxation_duration_s": 0.0,
        "relaxed": False,
        "stop_reason": "pending_simulation",
        "actual_final_time_ns": None,
        "final_max_torque": None,
        "final_energy": None,
        "torque_threshold": torque_threshold,
        "energy_slope_threshold": energy_slope_threshold,
        "autostop_enabled": autostop_enabled,
        "autostop_metric": "maxTorque" if autostop_enabled else None,
        "autostop_stop_when": ["maxTorque < torque_threshold", "timeout"] if autostop_enabled else None,
        "autostop_note": (
            "MuMax3 run.mx3 checks maxTorque after sparse relaxation chunks; "
            "actual frame count and final time require postprocessing after the run."
            if autostop_enabled else None
        ),
    }


def sample_segment_mode(spec_json: Dict[str, Any], sampler: SpecSampler, drive_type: str) -> Tuple[str, Dict[str, Any]]:
    modes = spec_json["drive_protocol"]["segment_modes"]
    if drive_type in {"none", "initial_spinwave_only_then_free"}:
        mode = "constant_over_pair"
    else:
        items = [{"value": key, "weight": float(value.get("sample_weight", 1.0))} for key, value in modes.items()]
        mode = str(sampler.sample({"dist": "categorical", "items": items}, "drive_protocol.segment_modes"))
    params = sampler.sample_params(modes[mode]["parameters"], f"drive_protocol.segment_modes.{mode}.parameters")
    if "approx_segment_count" in params and "segment_count" not in params:
        params["segment_count"] = params["approx_segment_count"]
    if mode == "smooth_waveform_tokenized":
        params = {
            "segment_count": int(params.get("segment_count", params.get("approx_segment_count", 8))),
            "pair_crosses_segment_boundary": False,
            "rendered_from": "smooth_waveform_tokenized",
        }
        mode = "piecewise_constant"
    return mode, params


def build_segment_bounds(mode: str, params: Dict[str, Any], total_s: float, rng: random.Random) -> List[Tuple[float, float, bool]]:
    if total_s <= 0:
        raise ValueError("total_s must be positive.")
    if mode == "rectangular_pulse_split":
        n = int(params.get("segment_count", 3))
        start = clamp(float(params.get("pulse_start_fraction", 0.2)), 0.0, 0.95)
        width = clamp(float(params.get("pulse_width_fraction", 0.2)), 0.01, 1.0 - start)
        end = min(1.0, start + width)
        if n == 5:
            pre_mid = 0.5 * start
            post_mid = end + 0.5 * (1.0 - end)
            points = [0.0, pre_mid, start, end, post_mid, 1.0]
            active = [False, False, True, False, False]
        else:
            points = [0.0, start, end, 1.0]
            active = [False, True, False]
        return [(points[i] * total_s, points[i + 1] * total_s, active[i]) for i in range(len(points) - 1) if points[i + 1] > points[i]]
    n = max(1, int(params.get("segment_count", 1)))
    if mode == "piecewise_constant" and n > 1:
        min_frac = clamp(float(params.get("segment_duration_fraction_min", 0.05)), 0.0, 0.45)
        min_frac = min(min_frac, 0.95 / n)
        raw = [rng.random() for _ in range(n)]
        s = sum(raw) or 1.0
        remaining = max(0.0, 1.0 - n * min_frac)
        fracs = [min_frac + remaining * x / s for x in raw]
    else:
        fracs = [1.0 / n] * n
    bounds: List[Tuple[float, float, bool]] = []
    acc = 0.0
    for frac in fracs:
        start = acc
        acc += frac
        bounds.append((start * total_s, min(acc, 1.0) * total_s, True))
    bounds[-1] = (bounds[-1][0], total_s, bounds[-1][2])
    return bounds


def strip_legacy_temperature_discretization_controls(variables: Dict[str, Any]) -> Dict[str, Any]:
    ignored: Dict[str, Any] = {}
    for key in ("segment_count", "segment_count_per_cycle", "approx_segment_count"):
        if key in variables:
            ignored[key] = variables.pop(key)
    return ignored


def temperature_schedule_parameters(schedule: Dict[str, Any]) -> Dict[str, Any]:
    return (
        schedule.get("T_schedule_parameters")
        or schedule.get("sampled_variables")
        or schedule.get("local_T_profile_parameters", {})
    )


def temperature_smoothstep(z: float) -> float:
    u = clamp(z, 0.0, 1.0)
    return 3.0 * u * u - 2.0 * u * u * u


def temperature_schedule_value_at_u(schedule: Dict[str, Any], u: float) -> float:
    mode = str(schedule.get("T_schedule_mode", "isothermal"))
    variables = temperature_schedule_parameters(schedule)
    cap = float(schedule.get("T_cap_K", float("inf")))
    u = clamp(u, 0.0, 1.0)
    if mode == "isothermal":
        value = float(variables.get("T0_K", variables.get("T_a_K", 300.0)))
    elif mode == "warmup_hold":
        ta = float(variables["T_start_K"])
        tb = max(ta, float(variables["T_peak_K"]))
        s = max(1e-6, float(variables["ramp_fraction"]))
        value = ta + (tb - ta) * temperature_smoothstep(u / s)
    elif mode == "anneal_hold":
        ta = float(variables["T_start_K"])
        tb = min(ta, float(variables["T_end_K"]))
        s = max(1e-6, float(variables["anneal_fraction"]))
        value = ta + (tb - ta) * temperature_smoothstep(u / s)
    elif mode == "warmup_anneal_cycle":
        ta = float(variables["T_low_K"])
        tb = max(ta, float(variables["T_high_K"]))
        cycles = max(1.0, float(variables["cycle_count"]))
        value = ta + (tb - ta) * 0.5 * (1.0 - math.cos(2.0 * math.pi * cycles * u))
    elif mode == "quench_relax":
        ta = float(variables["T_initial_K"])
        tb = min(ta, float(variables["T_final_K"]))
        tau = max(1e-6, float(variables["tau_quench"]))
        r = (math.exp(-u / tau) - math.exp(-1.0 / tau)) / max(1e-9, 1.0 - math.exp(-1.0 / tau))
        value = tb + (ta - tb) * r
    elif mode == "local_temperature_spot":
        base, peak_delta = local_temperature_components_at_u(schedule, u)
        value = base + 0.25 * peak_delta
    else:
        value = 300.0
    return clamp(value, 0.0, cap)


def local_temperature_components_at_u(schedule: Dict[str, Any], u: float) -> Tuple[float, float]:
    mode = str(schedule.get("T_schedule_mode", "isothermal"))
    variables = temperature_schedule_parameters(schedule)
    u = clamp(u, 0.0, 1.0)
    if mode != "local_temperature_spot":
        return temperature_schedule_value_at_u(schedule, u), 0.0
    base = float(variables["T_base_K"])
    delta = float(variables["delta_T_K"])
    u0 = float(variables["pulse_center_norm"])
    w = max(1e-6, float(variables["pulse_width_norm"]))
    pulse = math.exp(-((u - u0) ** 2) / (2.0 * w * w))
    return base, delta * pulse


def temperature_control_points_K(mode: str, variables: Dict[str, Any], cap: float) -> List[float]:
    if mode == "isothermal":
        values = [float(variables.get("T0_K", 300.0))]
    elif mode == "warmup_hold":
        values = [float(variables["T_start_K"]), float(variables["T_peak_K"])]
    elif mode == "anneal_hold":
        values = [float(variables["T_start_K"]), float(variables["T_end_K"])]
    elif mode == "warmup_anneal_cycle":
        values = [float(variables["T_low_K"]), float(variables["T_high_K"])]
    elif mode == "quench_relax":
        values = [float(variables["T_initial_K"]), float(variables["T_final_K"])]
    elif mode == "local_temperature_spot":
        base = float(variables["T_base_K"])
        values = [base, base + float(variables["delta_T_K"])]
    else:
        values = [300.0]
    return [clamp(value, 0.0, cap) for value in values]


def temperature_probe_us(schedule: Dict[str, Any]) -> List[float]:
    variables = temperature_schedule_parameters(schedule)
    points = {0.0, 0.5, 1.0}
    mode = str(schedule.get("T_schedule_mode", "isothermal"))
    if mode == "local_temperature_spot" and "pulse_center_norm" in variables:
        points.add(clamp(float(variables["pulse_center_norm"]), 0.0, 1.0))
    if mode == "warmup_anneal_cycle":
        cycles = max(1, int(round(float(variables.get("cycle_count", 1)))))
        for k in range(cycles):
            points.add(clamp((k + 0.5) / cycles, 0.0, 1.0))
    points.update(i / 32.0 for i in range(33))
    return sorted(points)


def sample_temperature_schedule(
    spec_json: Dict[str, Any],
    sampler: SpecSampler,
    temp_params: Dict[str, Any],
    segment_bounds: Sequence[Tuple[float, float, bool]],
) -> Tuple[str, Dict[str, Any], List[float]]:
    root = spec_json["T_schedule_mode"]
    mode = str(sampler.sample(root["sample"], "T_schedule_mode.sample"))
    mode_node = root["modes"][mode]
    variables = sampler.sample_params(mode_node.get("variables", {}), f"T_schedule_mode.modes.{mode}.variables")
    ignored_discretization = strip_legacy_temperature_discretization_controls(variables)
    max_proposal_T_K = 350.0
    for key in ("T0_K", "T_start_K", "T_peak_K", "T_end_K", "T_low_K", "T_high_K", "T_initial_K", "T_final_K", "T_base_K"):
        if key in variables:
            variables[key] = clamp(float(variables[key]), 0.0, max_proposal_T_K)
    # Keep endpoint metadata consistent with the monotone/cyclic temperature path semantics.
    if mode == "warmup_hold" and {"T_start_K", "T_peak_K"} <= variables.keys():
        variables["T_peak_K"] = max(float(variables["T_start_K"]), float(variables["T_peak_K"]))
    elif mode == "anneal_hold" and {"T_start_K", "T_end_K"} <= variables.keys():
        t_hi = max(float(variables["T_start_K"]), float(variables["T_end_K"]))
        t_lo = min(float(variables["T_start_K"]), float(variables["T_end_K"]))
        variables["T_start_K"] = t_hi
        variables["T_end_K"] = t_lo
    elif mode == "warmup_anneal_cycle" and {"T_low_K", "T_high_K"} <= variables.keys():
        t_hi = max(float(variables["T_low_K"]), float(variables["T_high_K"]))
        t_lo = min(float(variables["T_low_K"]), float(variables["T_high_K"]))
        variables["T_low_K"] = t_lo
        variables["T_high_K"] = t_hi
    elif mode == "quench_relax" and {"T_initial_K", "T_final_K"} <= variables.keys():
        t_hi = max(float(variables["T_initial_K"]), float(variables["T_final_K"]))
        t_lo = min(float(variables["T_initial_K"]), float(variables["T_final_K"]))
        variables["T_initial_K"] = t_hi
        variables["T_final_K"] = t_lo
    elif mode == "local_temperature_spot" and {"T_base_K", "delta_T_K"} <= variables.keys():
        base = float(variables["T_base_K"])
        delta = max(0.0, float(variables["delta_T_K"]))
        variables["delta_T_K"] = min(delta, max(0.0, max_proposal_T_K - base))
    theta_defaults = root["model_encoding"]["parameter_input"]["default_values"]
    theta = deepcopy(theta_defaults)
    mapping = mode_node.get("theta_T_mapping", {})
    for theta_name, var_name in mapping.items():
        if var_name in variables:
            theta[theta_name] = variables[var_name]

    tc = float(temp_params["Tc_K"])
    cap = 0.92 * tc

    total_s = segment_bounds[-1][1]
    schedule_base = {
        "T_schedule_mode": mode,
        "T_cap_K": cap,
        "T_schedule_parameters": deepcopy(variables),
        "sampled_variables": deepcopy(variables),
    }
    values = []
    base_values = []
    peak_delta_values = []
    for start_s, end_s, _ in segment_bounds:
        mid_u = 0.5 * (start_s + end_s) / total_s
        base_T, peak_delta_T = local_temperature_components_at_u(schedule_base, mid_u)
        values.append(temperature_schedule_value_at_u(schedule_base, mid_u))
        base_values.append(clamp(base_T, 0.0, cap))
        peak_delta_values.append(peak_delta_T)
    probe_us = temperature_probe_us(schedule_base)
    probe_peak_values = []
    for u in probe_us:
        base_T, peak_delta_T = local_temperature_components_at_u(schedule_base, u)
        probe_peak_values.append(clamp(base_T + peak_delta_T, 0.0, cap))

    schedule = {
        "T_ref_K": T_REF_K,
        "T_schedule_mode": mode,
        "T_function_id": mode_node.get("function_id", "piecewise_constant"),
        "T_mode_onehot": one_hot(mode, root["model_encoding"]["mode_input"]["order"]),
        "theta_T": [theta[name] for name in root["model_encoding"]["parameter_input"]["order"]],
        "theta_T_order": root["model_encoding"]["parameter_input"]["order"],
        "T_schedule_parameters": deepcopy(variables),
        "T_control_points_K": temperature_control_points_K(mode, variables, cap),
        "T_segment_values_K": values,
        "T_segment_base_values_K": base_values,
        "T_segment_peak_delta_values_K": peak_delta_values,
        "T_segment_start_s": [x[0] for x in segment_bounds],
        "T_segment_end_s": [x[1] for x in segment_bounds],
        "local_T_profile_type": "gaussian_spot" if mode == "local_temperature_spot" else "global_uniform",
        "local_T_profile_parameters": variables if mode == "local_temperature_spot" else {},
        "max_T_over_Tc": max(probe_peak_values) / tc if tc > 0 and probe_peak_values else None,
        "T_cap_K": cap,
        "sampled_variables": deepcopy(variables),
        "active_theta_fields": mode_node.get("active_theta_fields", []),
    }
    if ignored_discretization:
        schedule["ignored_discretization_parameters"] = ignored_discretization
    return mode, schedule, values


def one_hot(value: str, order: Sequence[str]) -> List[int]:
    return [1 if value == x else 0 for x in order]


def orientation_degrees_from_choice(choice: str, rng: random.Random) -> float:
    if choice == "vertical":
        return 90.0
    if choice == "random_angle":
        return rng.uniform(0.0, 180.0)
    return 0.0


def smooth_polygon_vertices_px(params: Dict[str, Any], rng: random.Random) -> List[List[float]]:
    n = max(3, int(params.get("vertex_count", 8)))
    r_min = float(params.get("radius_min_px", 50.0))
    r_max = max(r_min, float(params.get("radius_max_px", 100.0)))
    smooth = clamp(float(params.get("fourier_smoothing_strength", 0.3)), 0.0, 1.0)
    radii = [rng.uniform(r_min, r_max) for _ in range(n)]
    for _ in range(2):
        radii = [
            (1.0 - smooth) * radii[i] + smooth * (radii[i - 1] + radii[i] + radii[(i + 1) % n]) / 3.0
            for i in range(n)
        ]
    phi0 = math.radians(float(params.get("orientation_deg", 0.0)))
    return [
        [
            radii[i] * math.cos(phi0 + 2.0 * math.pi * i / n),
            radii[i] * math.sin(phi0 + 2.0 * math.pi * i / n),
        ]
        for i in range(n)
    ]


def grid_hole_centers_px(count: int, radius_px: float, jitter_fraction: float, rng: random.Random) -> List[List[float]]:
    count = max(0, count)
    if count == 0:
        return []
    cols = max(1, int(math.ceil(math.sqrt(count))))
    rows = max(1, int(math.ceil(count / cols)))
    usable = 256.0 - 2.4 * radius_px
    dx = usable / max(1, cols)
    dy = usable / max(1, rows)
    centers: List[List[float]] = []
    for idx in range(count):
        col = idx % cols
        row = idx // cols
        x = -0.5 * usable + (col + 0.5) * dx
        y = -0.5 * usable + (row + 0.5) * dy
        if jitter_fraction > 0:
            x += rng.uniform(-0.5, 0.5) * jitter_fraction * dx
            y += rng.uniform(-0.5, 0.5) * jitter_fraction * dy
        centers.append([clamp(x, -128.0 + radius_px, 128.0 - radius_px), clamp(y, -128.0 + radius_px, 128.0 - radius_px)])
    return centers


def random_hole_centers_px(params: Dict[str, Any], rng: random.Random) -> List[List[float]]:
    count = max(0, int(params.get("hole_count", 0)))
    radius = float(params.get("hole_radius_px", 8.0))
    min_sep = float(params.get("minimum_hole_separation_factor", 2.5)) * radius
    edge = float(params.get("minimum_edge_distance_factor", 1.2)) * radius
    lo = -128.0 + max(edge, radius)
    hi = 128.0 - max(edge, radius)
    centers: List[List[float]] = []
    for _ in range(5000):
        if len(centers) >= count:
            break
        x = rng.uniform(lo, hi)
        y = rng.uniform(lo, hi)
        if all((x - cx) ** 2 + (y - cy) ** 2 >= min_sep ** 2 for cx, cy in centers):
            centers.append([x, y])
    if len(centers) < count:
        centers.extend(grid_hole_centers_px(count - len(centers), radius, 0.0, rng))
    return centers[:count]


def enrich_geometry_parameters(mode: str, params: Dict[str, Any], rng: random.Random) -> None:
    if mode in {"nanostrip", "notched_strip"}:
        params["orientation_deg"] = orientation_degrees_from_choice(str(params.get("orientation", "horizontal")), rng)
    if mode == "smooth_polygon":
        params["polygon_vertices_px"] = smooth_polygon_vertices_px(params, rng)
    if mode == "antidot_or_holes":
        layout = str(params.get("hole_layout", "random_nonoverlap"))
        count = max(0, int(params.get("hole_count", 0)))
        radius = float(params.get("hole_radius_px", 8.0))
        jitter = float(params.get("jitter_fraction", 0.0) or 0.0)
        if layout == "random_nonoverlap":
            params["hole_centers_px"] = random_hole_centers_px(params, rng)
        else:
            params["hole_centers_px"] = grid_hole_centers_px(count, radius, jitter if layout == "jittered_grid" else 0.0, rng)


def sample_geometry(
    spec_json: Dict[str, Any],
    sampler: SpecSampler,
    material_family: str,
    normalized: Dict[str, Any],
    rng: random.Random,
) -> Tuple[str, Dict[str, Any], List[str]]:
    geom_root = spec_json["geometry_sampling"]
    family_node = spec_json["material_family"]["families"][material_family]
    compatible = set(family_node.get("compatible_geometry_modes", geom_root["modes"].keys()))
    items = [deepcopy(item) for item in geom_root["sample"]["items"] if item["value"] in compatible]
    if not items:
        items = deepcopy(geom_root["sample"]["items"])
    mode = str(sampler.sample({"dist": "categorical", "items": items}, "geometry_sampling.sample"))
    params = sample_mode_parameters(geom_root["modes"][mode]["parameters"], sampler, f"geometry_sampling.modes.{mode}", normalized, rng)
    enrich_geometry_parameters(mode, params, rng)
    tags: List[str] = []
    if mode == "antidot_or_holes" and params.get("hole_layout") != "random_nonoverlap":
        tags.append("periodic_antidot_lattice")
    return mode, params, tags


def geometry_rejection_reasons(geometry_mode: str, geometry_params: Dict[str, Any], normalized: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    rex = float(normalized.get("rex", 0.0) or 0.0)
    if geometry_mode == "ring":
        outer = float(geometry_params.get("outer_diameter_px", 0.0) or 0.0)
        ratio = float(geometry_params.get("inner_outer_ratio", 1.0) or 1.0)
        ring_width = 0.5 * outer * (1.0 - ratio)
        if ring_width < max(8.0, 2.0 * rex):
            reasons.append("ring_width_under_required_max_8_or_2rex")
    if geometry_mode == "smooth_polygon":
        r_min = float(geometry_params.get("radius_min_px", 0.0) or 0.0)
        r_max = float(geometry_params.get("radius_max_px", 0.0) or 0.0)
        if r_min <= 0 or r_max <= 0 or r_min > r_max:
            reasons.append("smooth_polygon_invalid_radius_bounds")
    return reasons


def sample_mode_parameters(
    params_node: Dict[str, Any],
    sampler: SpecSampler,
    path: str,
    normalized: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, node in params_node.items():
        if isinstance(node, dict) and "dist" in node:
            if node["dist"] == "derived_range_uniform":
                low = eval_low_formula(str(node.get("low_formula", "0")), normalized, out)
                hi = float(node["high"])
                if hi < low:
                    hi = low
                value = rng.uniform(low, hi)
                sampler._record(f"{path}.parameters.{key}", "derived_range_uniform", value)
                out[key] = value
            elif node["dist"] == "derived":
                out[key] = sample_known_derived(str(node.get("formula", "")), normalized, out, rng)
                sampler._record(f"{path}.parameters.{key}", "derived", out[key])
            else:
                out[key] = sampler.sample(node, f"{path}.parameters.{key}")
        else:
            out[key] = deepcopy(node)
    return out


def eval_low_formula(formula: str, normalized: Dict[str, Any], params: Dict[str, Any]) -> float:
    rex = float(normalized.get("rex", 4.0))
    wall_width_px = float(params.get("wall_width_px", rex))
    if "0.75*rex" in formula:
        return max(6.0, 0.75 * rex)
    if "2*wall_width_px" in formula:
        return max(6.0, 2.0 * wall_width_px)
    if "4*rex" in formula:
        return max(12.0, 4.0 * rex)
    if "6*rex" in formula:
        return max(24.0, 6.0 * rex)
    return max(0.0, rex)


def sample_known_derived(formula: str, normalized: Dict[str, Any], params: Dict[str, Any], rng: random.Random) -> float:
    rex = float(normalized.get("rex", 4.0))
    delta_dw = normalized.get("Delta_DW_ref_m")
    dx = 2e-9
    if "Delta_DW_ref_m" in formula and delta_dw:
        return clamp(rng.uniform(0.7, 1.5) * math.pi * float(delta_dw) / dx, 6.0, 48.0)
    if "LD_cells_ref" in formula:
        ld = normalized.get("LD_cells_ref") or 48.0
        return clamp(rng.uniform(0.7, 1.8) * float(ld), 16.0, 128.0)
    if "clip(Uniform(0.7,1.5)*rex,4,10)" in formula:
        return clamp(rng.uniform(0.7, 1.5) * rex, 4.0, 10.0)
    if "max(12,4*rex)" in formula:
        return rng.uniform(max(12.0, 4.0 * rex), 128.0)
    return clamp(rng.uniform(1.0, 3.0) * rex, 6.0, 48.0)


def boundary_sample_node(root: Dict[str, Any], geometry_mode: str, profile: str) -> Dict[str, Any]:
    default_node = root["default_boundary_by_geometry"].get(geometry_mode, root["default_boundary_by_geometry"]["full_rectangle"])
    sample_node = deepcopy(default_node["sample"])
    override = root.get("profile_overrides", {}).get(profile)
    if not override:
        return sample_node
    absorbing_weight = float(override.get("allow_absorbing_edge_weight", 0.0) or 0.0)
    if absorbing_weight <= 0.0 or "absorbing_edge" not in root.get("modes", {}):
        return sample_node
    items = [deepcopy(item) for item in sample_node.get("items", []) if item.get("value") != "absorbing_edge"]
    if override.get("renormalize_other_weights", False):
        total = sum(float(item.get("weight", 1.0)) for item in items) or 1.0
        scale = max(0.0, 1.0 - absorbing_weight) / total
        for item in items:
            item["weight"] = float(item.get("weight", 1.0)) * scale
    items.append({"value": "absorbing_edge", "weight": absorbing_weight})
    out = deepcopy(sample_node)
    out["items"] = items
    return out


def boundary_geometry_interpretation(geometry_mode: str, boundary_mode: str) -> str:
    if boundary_mode == "absorbing_edge":
        return "shape_mask_with_absorbing_edge"
    if geometry_mode == "antidot_or_holes" and boundary_mode == "pbc_xy":
        return "periodic_antidot_lattice"
    if geometry_mode in FINITE_OBJECT_GEOMETRIES and boundary_mode == "pbc_xy":
        return "periodic_object_array"
    if boundary_mode.startswith("pbc"):
        return "periodic_shape_mask_on_fixed_256_grid"
    return "shape_mask_on_fixed_256_grid"


def boundary_validity_tags(geometry_mode: str, boundary_mode: str) -> List[str]:
    tags: List[str] = []
    if geometry_mode in FINITE_OBJECT_GEOMETRIES and boundary_mode == "pbc_xy":
        tags.append("periodic_object_array")
    if geometry_mode == "antidot_or_holes" and boundary_mode == "pbc_xy":
        tags.append("periodic_antidot_lattice")
    return tags


def sample_boundary(
    spec_json: Dict[str, Any],
    geometry_mode: str,
    profile: str,
    sampler: SpecSampler,
) -> Tuple[str, str, Dict[str, Any], List[str]]:
    root = spec_json["boundary"]
    sample_node = boundary_sample_node(root, geometry_mode, profile)
    boundary_mode = str(sampler.sample(sample_node, f"boundary.default_boundary_by_geometry.{geometry_mode}"))
    params = sampler.sample_params(root["modes"][boundary_mode]["parameters"], f"boundary.modes.{boundary_mode}.parameters")
    out = {
        "default_boundary": boundary_mode,
        "boundary_mode": boundary_mode,
        "pbc_x": bool(params.get("pbc_x", False)),
        "pbc_y": bool(params.get("pbc_y", False)),
        "pbc_repeat_x": int(params.get("pbc_repeat_x", 0)),
        "pbc_repeat_y": int(params.get("pbc_repeat_y", 0)),
        "geometry_interpretation": boundary_geometry_interpretation(geometry_mode, boundary_mode),
        "absorbing_edge_width_px": params.get("absorbing_width_px", 0.0),
        "alpha_edge_multiplier": params.get("alpha_edge_multiplier", 1.0),
        "sponge_profile": params.get("sponge_profile", "none"),
        "raw_parameters": params,
    }
    return boundary_mode, boundary_mode, out, boundary_validity_tags(geometry_mode, boundary_mode)


def sample_cubic(
    spec_json: Dict[str, Any],
    sampler: SpecSampler,
    raw: Dict[str, Any],
    rng: random.Random,
) -> Tuple[str, Dict[str, Any], List[str]]:
    root = spec_json["cubic_anisotropy"]
    mode = str(sampler.sample(root["sample"], "cubic_anisotropy.sample"))
    mode_node = root["modes"][mode]
    vars_ = sampler.sample_params(mode_node.get("variables", {}), f"cubic_anisotropy.modes.{mode}.variables")
    fixed = sampler.sample_params(mode_node.get("fixed_parameters", {}), f"cubic_anisotropy.modes.{mode}.fixed_parameters")
    if mode == "cubic_off":
        kc1 = float(fixed.get("Kc1_ref_J_per_m3", 0.0))
        kc2 = float(fixed.get("Kc2_ref_J_per_m3", 0.0))
        qC1 = 0.0
        qC2 = 0.0
    else:
        q_abs = float(vars_.get("qC1_ref_abs", 0.0))
        if mode == "mixed_uniaxial_cubic":
            q_abs = min(q_abs, float(vars_.get("relative_to_qK", 0.3)) * max(float(raw.get("qK_ref", 0.0)), 0.05))
        sign = sign_value(vars_.get("Kc1_sign", 1))
        qC1 = sign * q_abs
        qC2 = qC1 * float(vars_.get("qC2_over_qC1", 0.0))
        kc1 = qC1 * 0.5 * MU0 * raw["Ms_ref_A_per_m"] ** 2
        kc2 = qC2 * 0.5 * MU0 * raw["Ms_ref_A_per_m"] ** 2
    axis_mode = str(vars_.get("cubic_axis_mode", "global_aligned"))
    rotation_deg = float(vars_.get("cubic_axis_rotation_deg", 0.0))
    axes = cubic_axes(axis_mode, rotation_deg, raw.get("anis_u", [0.0, 0.0, 1.0]), rng)
    tags = []
    if mode == "cubic_dominant_edge":
        tags.append("cubic_dominant_edge_validity")
    if abs(qC1) > 0.4 and raw.get("qK_ref", 0.0) < 1.1:
        tags.append("anisotropy_competition_tag")
    out = {
        "CUBIC_ANISOTROPY_MODE": mode,
        "Kc1_ref_J_per_m3": kc1,
        "Kc2_ref_J_per_m3": kc2,
        "qC1_ref": qC1,
        "qC2_ref": qC2,
        "cubic_axis_mode": axis_mode,
        "cubic_axis_1": axes[0],
        "cubic_axis_2": axes[1],
        "cubic_axis_3": axes[2],
        "sampled_variables": vars_,
    }
    return mode, out, tags


def cubic_axes(axis_mode: str, rotation_deg: float, anis_u: Sequence[float], rng: random.Random) -> List[List[float]]:
    if axis_mode == "random_3d_orthonormal":
        a1 = random_unit_sphere(rng)
        temp = random_unit_sphere(rng)
        dot = sum(a * b for a, b in zip(a1, temp))
        a2 = normalize([temp[i] - dot * a1[i] for i in range(3)])
        a3 = normalize([
            a1[1] * a2[2] - a1[2] * a2[1],
            a1[2] * a2[0] - a1[0] * a2[2],
            a1[0] * a2[1] - a1[1] * a2[0],
        ])
        return [a1, a2, a3]
    if axis_mode in {"aligned_to_grid", "global_aligned", "cubic_off"}:
        phi = 0.0
    elif axis_mode == "aligned_with_uK_projection":
        ux = float(anis_u[0]) if len(anis_u) > 0 else 1.0
        uy = float(anis_u[1]) if len(anis_u) > 1 else 0.0
        phi = math.atan2(uy, ux) if ux * ux + uy * uy > 1e-12 else 0.0
    else:
        phi = math.radians(rotation_deg)
    c, s = math.cos(phi), math.sin(phi)
    return [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]


def sample_magnetoelastic(
    spec_json: Dict[str, Any],
    sampler: SpecSampler,
    raw: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], List[str]]:
    root = spec_json["magnetoelastic_coupling"]
    sample_node = root["sample"]
    if not ENABLE_STANDARD_MX3_MAGNETOELASTIC:
        sample_node = {
            "dist": "categorical",
            "items": [{"value": "magnetoelastic_off", "weight": 1.0}],
        }
    mode = str(sampler.sample(sample_node, "magnetoelastic_coupling.sample"))
    mode_node = root["modes"][mode]
    vars_ = sampler.sample_params(mode_node.get("variables", {}), f"magnetoelastic_coupling.modes.{mode}.variables")
    fixed = sampler.sample_params(mode_node.get("fixed_parameters", {}), f"magnetoelastic_coupling.modes.{mode}.fixed_parameters")
    b1 = float(vars_.get("B1_ref_J_per_m3", fixed.get("B1_ref_J_per_m3", 0.0)))
    b2 = float(vars_.get("B2_ref_J_per_m3", fixed.get("B2_ref_J_per_m3", 0.0)))
    strain_values = [
        float(vars_.get("epsilon_xx", vars_.get("epsilon_base", vars_.get("strain_amplitude", 0.0)) or 0.0)),
        float(vars_.get("epsilon_yy", 0.0) or 0.0),
        float(vars_.get("epsilon_xy", 0.0) or 0.0),
        float(vars_.get("epsilon_zz", 0.0) or 0.0),
        float(vars_.get("epsilon_gradient_amplitude", 0.0) or 0.0),
    ]
    max_abs_strain = max(abs(x) for x in strain_values)
    qme = 0.0
    if raw["Ms_ref_A_per_m"] > 0:
        qme = 2.0 * max(abs(b1), abs(b2)) * max_abs_strain / (MU0 * raw["Ms_ref_A_per_m"] ** 2)
    tags = ["magnetoelastic_edge_validity"] if qme > 1.5 else []
    profile_type = vars_.get("profile_type", "uniform" if mode.startswith("uniform") else "none")
    out = {
        "MAGNETOELASTIC_MODE": mode,
        "B1_ref_J_per_m3": b1,
        "B2_ref_J_per_m3": b2,
        "strain_profile_type": profile_type,
        "strain_summary": vars_ if vars_ else fixed,
        "max_abs_strain": max_abs_strain,
        "qME_ref": qme,
        "sampled_variables": vars_,
    }
    return mode, out, tags


def sample_defects(spec_json: Dict[str, Any], profile: str, sampler: SpecSampler) -> Tuple[str, Dict[str, Any]]:
    root = spec_json["defects_and_disorder"]
    if ENABLE_STANDARD_MX3_DEFECT_DISORDER:
        node = root["sample_edge_rare"] if profile == "edge_rare" else root["sample_default"]
    else:
        node = {
            "dist": "categorical",
            "items": [{"value": "uniform_material", "weight": 1.0}],
        }
    mode = str(sampler.sample(node, f"defects_and_disorder.{profile}"))
    params = sampler.sample_params(root["modes"][mode].get("parameters", {}), f"defects_and_disorder.modes.{mode}.parameters")
    return mode, {"DEFECT_MODE": mode, "parameters": params}


def init_mode_is_compatible(mode_node: Dict[str, Any], material_family: str, geometry_mode: str) -> bool:
    compatible_materials = mode_node.get("compatible_materials")
    if compatible_materials and material_family not in set(compatible_materials):
        return False
    compatible_geometries = mode_node.get("compatible_geometries")
    if compatible_geometries and geometry_mode not in set(compatible_geometries):
        return False
    return True


def filter_init_sample_node(
    sample_node: Dict[str, Any],
    init_modes: Dict[str, Any],
    material_family: str,
    geometry_mode: str,
    raw: Dict[str, Any],
    normalized: Dict[str, Any],
) -> Dict[str, Any]:
    out = deepcopy(sample_node)
    out["items"] = [
        deepcopy(item)
        for item in sample_node.get("items", [])
        if init_mode_is_compatible(init_modes[str(item["value"])], material_family, geometry_mode)
        and init_mode_is_allowed_by_physics(str(item["value"]), material_family, geometry_mode, raw, normalized)
    ]
    return out


def init_mode_is_allowed_by_physics(
    init_family: str,
    material_family: str,
    geometry_mode: str,
    raw: Dict[str, Any],
    normalized: Dict[str, Any],
) -> bool:
    if material_family != "pma_no_dmi" or init_family != "bubble_skyrmion":
        return True
    thickness_m = float(raw.get("thickness_m", 0.0) or 0.0)
    qk = float(normalized.get("qK_ref", raw.get("qK_ref", 0.0)) or 0.0)
    qk_min, qk_max = PMA_NO_DMI_BUBBLE_QK_RANGE
    return (
        thickness_m >= PMA_NO_DMI_BUBBLE_MIN_THICKNESS_M
        and qk_min <= qk <= qk_max
        and geometry_mode in PMA_NO_DMI_BUBBLE_GEOMETRIES
    )


def sample_init(
    spec_json: Dict[str, Any],
    material_family: str,
    geometry_mode: str,
    grid: Dict[str, Any],
    raw: Dict[str, Any],
    normalized: Dict[str, Any],
    sampler: SpecSampler,
    rng: random.Random,
) -> Tuple[str, Dict[str, Any], List[str]]:
    family_node = spec_json["material_family"]["families"][material_family]
    init_modes = spec_json["initial_state_sampling"]["modes"]
    init_sample = family_node.get("compatible_init_families", {}).get("sample")
    if init_sample is None:
        init_sample = spec_json["initial_state_sampling"]["global_sample_target"]
    init_sample = filter_init_sample_node(init_sample, init_modes, material_family, geometry_mode, raw, normalized)
    if not init_sample.get("items"):
        init_sample = filter_init_sample_node(
            spec_json["initial_state_sampling"]["global_sample_target"],
            init_modes,
            material_family,
            geometry_mode,
            raw,
            normalized,
        )
    init_family = str(sampler.sample(init_sample, f"material_family.families.{material_family}.compatible_init_families"))
    mode_node = init_modes[init_family]
    params = sample_mode_parameters(mode_node["parameters"], sampler, f"initial_state_sampling.modes.{init_family}", normalized, rng)
    tags = []
    if init_family == "domain_wall":
        params["wall_width_px"] = params.get("wall_width_px_if_Delta_valid") or params.get("wall_width_px_if_Delta_invalid") or 8.0
        wall_type_node = mode_node["parameters"]["wall_type_by_material"]
        if material_family == "inplane_soft" and geometry_mode in {"nanostrip", "notched_strip"}:
            params["wall_type"] = sampler.sample(wall_type_node["inplane_soft_nanostrip"], "initial_state_sampling.domain_wall.wall_type")
        elif raw.get("DMI_TYPE") == "interfacial":
            params["wall_type"] = sampler.sample(wall_type_node["interfacial_dmi"], "initial_state_sampling.domain_wall.wall_type")
        else:
            params["wall_type"] = sampler.sample(wall_type_node["pma_no_dmi"], "initial_state_sampling.domain_wall.wall_type")
    if init_family == "stripe_labyrinth":
        if raw.get("D_ref_J_per_m2", 0.0) != 0.0:
            params["period_px"] = sample_known_derived("LD_cells_ref", normalized, params, rng)
        else:
            params["period_px"] = float(params.get("period_px_if_no_DMI", 48.0))
    if init_family == "bubble_skyrmion":
        if material_family == "pma_no_dmi":
            tags.append("pma_no_dmi_dipolar_bubble_seed_conditioned")
        if params.get("core_polarity") == params.get("background_polarity"):
            params["core_polarity"] = "-" if params.get("background_polarity") == "+" else "+"
            tags.append("bubble_core_polarity_forced_opposite_background")
        count = params.get("object_count", 1)
        if isinstance(count, str):
            count_node = mode_node["parameters"]["object_count_expanded"][count]
            count = sampler.sample(count_node, f"initial_state_sampling.bubble_skyrmion.object_count_expanded.{count}")
        if params.get("intentional_unstable_type") == "two_objects_close_to_collision":
            count = max(2, int(count))
        params["object_count"] = int(count)
        params["radius_px"] = params.get("radius_px") or rng.uniform(max(6.0, 2.0 * normalized["rex"]), 48.0)
        if params.get("intentional_unstable_type") == "radius_too_small_likely_collapse":
            params["radius_px"] = max(3.0, float(params["radius_px"]) * rng.uniform(0.45, 0.75))
        elif params.get("intentional_unstable_type") == "radius_too_large_likely_expansion":
            max_radius = 0.42 * min(float(grid.get("Nx", 256)), float(grid.get("Ny", 256)))
            params["radius_px"] = min(max_radius, float(params["radius_px"]) * rng.uniform(1.2, 1.6))
        dmi_key = "bulk" if raw.get("DMI_TYPE") == "bulk" else ("interfacial" if raw.get("DMI_TYPE") == "interfacial" else "none_pma")
        params["helicity"] = sampler.sample(mode_node["parameters"]["helicity_by_DMI"][dmi_key], "initial_state_sampling.bubble_skyrmion.helicity")
        if params.get("intentional_unstable_type") != "normal_stable_ish":
            tags.append("intentional_unstable_initial_state")
        params["objects"] = place_objects(int(params["object_count"]), float(params["radius_px"]), params, grid, rng)
    return init_family, {"INIT_FAMILY": init_family, "parameters": params}, tags


def place_objects(
    count: int,
    radius_px: float,
    params: Dict[str, Any],
    grid: Dict[str, Any],
    rng: random.Random,
) -> List[Dict[str, float]]:
    objects: List[Dict[str, float]] = []
    nx = float(grid.get("Nx", 256))
    ny = float(grid.get("Ny", 256))
    aspect = max(1.0, float(params.get("aspect_ratio", 1.0) or 1.0))
    minor_radius_px = radius_px / aspect
    margin_x = max(radius_px + 4.0, 12.0)
    margin_y = max(minor_radius_px + 4.0, 12.0)
    min_sep = float(params.get("minimum_center_separation_factor", 1.5) or 1.5) * radius_px
    position_type = str(params.get("position_type", "safely_inside_geometry"))
    near_fraction = clamp(float(params.get("near_boundary_distance_fraction_of_radius", 0.5) or 0.5), 0.0, 1.0)
    deformation = max(0.0, float(params.get("deformation_amplitude_fraction_of_radius", 0.0) or 0.0))
    unstable = str(params.get("intentional_unstable_type", "normal_stable_ish"))

    def random_center() -> Tuple[float, float]:
        if position_type == "near_boundary":
            edge = rng.choice(["left", "right", "bottom", "top"])
            offset = max(2.0, near_fraction * radius_px)
            if edge == "left":
                return margin_x - radius_px + offset, rng.uniform(margin_y, ny - margin_y)
            if edge == "right":
                return nx - margin_x + radius_px - offset, rng.uniform(margin_y, ny - margin_y)
            if edge == "bottom":
                return rng.uniform(margin_x, nx - margin_x), margin_y - minor_radius_px + offset
            return rng.uniform(margin_x, nx - margin_x), ny - margin_y + minor_radius_px - offset
        return rng.uniform(margin_x, nx - margin_x), rng.uniform(margin_y, ny - margin_y)

    def append_object(x_px: float, y_px: float, radius_scale: float = 1.0) -> None:
        objects.append({
            "x_px": clamp(x_px, 0.0, nx),
            "y_px": clamp(y_px, 0.0, ny),
            "radius_px": radius_px * radius_scale,
            "aspect_ratio": max(1.0, aspect * rng.uniform(0.95, 1.05)),
            "orientation_deg": rng.uniform(0.0, 180.0),
            "deformation_amplitude_fraction_of_radius": deformation,
            "deformation_phase_rad": rng.uniform(0.0, 2.0 * math.pi),
        })

    for idx in range(count):
        if idx == 1 and unstable == "two_objects_close_to_collision" and objects:
            first = objects[0]
            theta = rng.uniform(0.0, 2.0 * math.pi)
            separation = rng.uniform(0.55, 0.95) * min_sep
            append_object(float(first["x_px"]) + separation * math.cos(theta), float(first["y_px"]) + separation * math.sin(theta), rng.uniform(0.9, 1.1))
            continue
        placed = False
        for _attempt in range(500):
            x_px, y_px = random_center()
            if unstable == "two_objects_close_to_collision" and not objects:
                append_object(x_px, y_px, rng.uniform(0.9, 1.1))
                placed = True
                break
            if all((x_px - obj["x_px"]) ** 2 + (y_px - obj["y_px"]) ** 2 >= min_sep ** 2 for obj in objects):
                append_object(x_px, y_px, rng.uniform(0.85, 1.15))
                placed = True
                break
        if not placed:
            x_px, y_px = random_center()
            append_object(x_px, y_px, rng.uniform(0.85, 1.15))
    return objects


def add_render_audit(
    audit: List[Dict[str, Any]],
    path: str,
    sampled_value: Any,
    rendered_value: Any,
    status: str,
    reason: str,
) -> None:
    if sampled_value == rendered_value:
        return
    audit.append({
        "path": path,
        "sampled_value": sampled_value,
        "rendered_value": rendered_value,
        "status": status,
        "reason": reason,
    })


def canonicalize_geometry_to_mx3(
    geometry_mode: str,
    geometry_params: Dict[str, Any],
    audit: List[Dict[str, Any]],
) -> None:
    if geometry_mode == "disk":
        old = geometry_params.get("center_jitter_px", 0.0)
        geometry_params["center_jitter_px"] = 0.0
        add_render_audit(
            audit,
            "geometry.parameters.center_jitter_px",
            old,
            0.0,
            "canonicalized_to_rendered",
            "Current MuMax3 disk shape is centered; no transl() is rendered.",
        )
    if geometry_mode == "ellipse":
        old_orientation = geometry_params.get("orientation_deg", 0.0)
        old_jitter = geometry_params.get("center_jitter_px", 0.0)
        geometry_params["orientation_deg"] = 0.0
        geometry_params["center_jitter_px"] = 0.0
        add_render_audit(
            audit,
            "geometry.parameters.orientation_deg",
            old_orientation,
            0.0,
            "canonicalized_to_rendered",
            "Current MuMax3 ellipse shape is axis-aligned; no RotZ() is rendered.",
        )
        add_render_audit(
            audit,
            "geometry.parameters.center_jitter_px",
            old_jitter,
            0.0,
            "canonicalized_to_rendered",
            "Current MuMax3 ellipse shape is centered; no transl() is rendered.",
        )
    if geometry_mode == "ring":
        old = geometry_params.get("orientation_deg", 0.0)
        geometry_params["orientation_deg"] = 0.0
        add_render_audit(
            audit,
            "geometry.parameters.orientation_deg",
            old,
            0.0,
            "canonicalized_to_rendered",
            "Current MuMax3 ring is rotationally symmetric.",
        )


def canonicalize_initial_state_to_mx3(
    init_family: str,
    init: Dict[str, Any],
    audit: List[Dict[str, Any]],
) -> None:
    params = init.get("parameters", {})
    if init_family == "domain_wall":
        replacements = {
            "center_offset_mode": "rendered_centered_spacing",
            "meander_amplitude_fraction_of_spacing": 0.0,
            "meander_wavelength_px": 0.0,
            "wall_type": "rendered_tanh_wall",
        }
        for key, rendered in replacements.items():
            if key in params:
                old = params.get(key)
                params[key] = rendered
                add_render_audit(
                    audit,
                    f"initial_state.parameters.{key}",
                    old,
                    rendered,
                    "canonicalized_to_rendered",
                    "Current MuMax3 init renders straight tanh walls with fixed in-plane rotation.",
                )
    elif init_family == "vortex_antivortex":
        replacements = {
            "core_number": 1,
            "initial_offset_fraction_of_geometry_radius": 0.0,
            "vortex_scenario": "centered_vortex",
        }
        for key, rendered in replacements.items():
            if key in params:
                old = params.get(key)
                params[key] = rendered
                add_render_audit(
                    audit,
                    f"initial_state.parameters.{key}",
                    old,
                    rendered,
                    "canonicalized_to_rendered",
                    "Current MuMax3 init renders one centered vortex core.",
                )
    elif init_family == "spinwave_fmr_seed":
        replacements = {
            "background_state": "uniform",
            "mode_count": 1,
            "gaussian_packet_width_px": 0.0,
            "type": "plane_wave",
        }
        for key, rendered in replacements.items():
            if key in params:
                old = params.get(key)
                params[key] = rendered
                add_render_audit(
                    audit,
                    f"initial_state.parameters.{key}",
                    old,
                    rendered,
                    "canonicalized_to_rendered",
                    "Current MuMax3 init renders a single plane-wave perturbation on a uniform background.",
                )


def canonicalize_drive_segments_to_mx3(
    drive_type: str,
    drive_summary: Dict[str, Any],
    segments: Sequence[Dict[str, Any]],
    audit: List[Dict[str, Any]],
) -> str:
    rendered_drive_type = DRIVE_TYPE_RENDER_ALIASES.get(drive_type, drive_type)
    reason = (
        "Effective spin-Hall SOT is rendered exactly in MuMax3's Lambda=1 Slonczewski basis: "
        "J_solver is out of plane, FixedLayer=sigma_SOT, Pol=theta_DL_eff, and "
        "EpsilonPrime=(Pol/2)*r_FL_DL."
        if drive_type in SOT_DRIVE_TYPES
        else "Current MuMax3 renderer uses segment-wise constant field/current assignments, not analytic AC/sinc/local-antenna waveforms."
    )
    add_render_audit(
        audit,
        "drive_type",
        drive_type,
        rendered_drive_type,
        "canonicalized_to_rendered",
        reason,
    )
    drive_summary["drive_type_sampled"] = drive_type
    drive_summary["drive_type_rendered"] = rendered_drive_type
    drive_summary["drive_profile_type"] = "segment_piecewise_constant"
    if drive_type in SOT_DRIVE_TYPES:
        drive_summary["rendered_torque_model"] = SOT_RENDERED_TORQUE_MODEL
        drive_summary["has_sot"] = any(bool(s["drive"].get("has_sot")) for s in segments)
        drive_summary["has_sot_like_slonczewski_proxy"] = False
    if "rendered_from" in drive_summary.get("segment_mode_parameters", {}):
        add_render_audit(
            audit,
            "drive.segment_mode_parameters.waveform_type",
            drive_summary["segment_mode_parameters"].get("rendered_from"),
            "piecewise_constant_segments",
            "canonicalized_to_rendered",
            "Analytic waveform tokens are approximated as constant MuMax3 run() segments.",
        )

    for seg in segments:
        d = seg["drive"]
        d["drive_type_sampled"] = d.get("drive_type", drive_type)
        if d.get("active_kind") in {"field", "sot"}:
            d["drive_type"] = rendered_drive_type
        else:
            d["drive_type"] = d.get("drive_type", drive_type)
        d["drive_type_rendered"] = d["drive_type"]
        d["drive_profile_type"] = "segment_piecewise_constant"
        profile = d.get("spatial_profile", "global_uniform")
        rendered_profile = SPATIAL_PROFILE_RENDER_ALIASES.get(profile, profile)
        add_render_audit(
            audit,
            f"segments[{seg['segment_index']}].drive.spatial_profile",
            profile,
            rendered_profile,
            "canonicalized_to_rendered",
            "Current MuMax3 local drive renderer uses single-center Gaussian, stripe, disk/contact, or ring tile profiles.",
        )
        d["spatial_profile_sampled"] = profile
        d["spatial_profile"] = rendered_profile
        params = d.setdefault("profile_parameters", {})
        for key, rendered in {"spot_count": 1, "antenna_orientation_deg": 0.0}.items():
            if key in params:
                old = params[key]
                params[key] = rendered
                add_render_audit(
                    audit,
                    f"segments[{seg['segment_index']}].drive.profile_parameters.{key}",
                    old,
                    rendered,
                    "canonicalized_to_rendered",
                    "Current MuMax3 tile profile renderer does not render multi-spot counts or rotated antenna axes.",
                )
    return rendered_drive_type


def rendered_parameter_audit_dict(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "contract": "metadata condition fields are canonicalized to what run.mx3 renders; sampled richer labels are audit-only.",
        "renderer": "MuMax3 segment renderer in generate_universal_2d_micromagnetic_dynamics_v4.py",
        "entry_count": len(entries),
        "entries": list(entries),
    }


def sample_drive_type(spec_json: Dict[str, Any], profile: str, sampler: SpecSampler) -> str:
    node = spec_json["drive_protocol"]["drive_type_by_profile"].get(profile, spec_json["drive_protocol"]["drive_type_by_profile"]["core_relax"])
    return str(sampler.sample(node, f"drive_protocol.drive_type_by_profile.{profile}"))


def canonical_drive_kind(drive_type: str) -> str:
    if drive_type in {"none", "initial_spinwave_only_then_free"}:
        return "none"
    if drive_type in {
        "static_field",
        "field_quench",
        "field_pulse_or_local_field",
        "ac_field",
        "sinc_or_gaussian_field_pulse",
        "local_antenna",
        "strong_field_pulse",
        "topology_collision_setup",
    }:
        return "field"
    if drive_type == "zhang_li_stt":
        return "zhang_li"
    if drive_type == "slonczewski_stt":
        return "slonczewski"
    if drive_type in SOT_DRIVE_TYPES or drive_type in {SOT_RENDERED_DRIVE_TYPE, SOT_PROXY_RENDERED_DRIVE_TYPE}:
        return "sot"
    return "none"


def sot_epsilon_prime(pol: float, r_fl_dl: float) -> float:
    """Map a Gilbert-basis FL/DL ratio to MuMax3 Lambda=1 parameters.

    The basis is ``m x (sigma x m)`` for DL and ``sigma x m`` for FL.  MuMax3
    uses epsilon=Pol/2 when Lambda=1, so the secondary coefficient must be
    scaled by the same factor.  A convention using ``m x sigma`` for FL would
    carry the opposite sign and is deliberately not used by this dataset.
    """
    return 0.5 * float(pol) * float(r_fl_dl)


def sot_field_coefficients_T(
    j_solver_A_per_m2: float,
    pol: float,
    epsilon_prime: float,
    ms_A_per_m: float,
    thickness_m: float,
    alpha: float,
) -> Dict[str, float]:
    """Return MuMax3 raw and explicit SOT coefficients in tesla.

    This mirrors ``cuda/slonczewski2.cu`` for Lambda=1.  Keeping the formula
    here makes metadata, timestep selection, tests, and the actual solver use
    one auditable convention.
    """
    ms = float(ms_A_per_m)
    thickness = float(thickness_m)
    if not all(math.isfinite(x) for x in (j_solver_A_per_m2, pol, epsilon_prime, ms, thickness, alpha)):
        raise ValueError("SOT parameters must be finite")
    if ms <= 0.0 or thickness <= 0.0:
        raise ValueError("SOT requires positive Ms and free-layer thickness")
    beta_T = (HBAR_J_S / ELEMENTARY_CHARGE_C) * float(j_solver_A_per_m2) / (thickness * ms)
    b_dl = beta_T * (0.5 * float(pol))
    b_fl = beta_T * float(epsilon_prime)
    gilbert = 1.0 / (1.0 + float(alpha) ** 2)
    explicit_dl = gilbert * (b_dl + float(alpha) * b_fl)
    explicit_fl = gilbert * (b_fl - float(alpha) * b_dl)
    return {
        "sot_beta_T": beta_T,
        "sot_B_DL_T": b_dl,
        "sot_B_FL_T": b_fl,
        "sot_explicit_B_DL_T": explicit_dl,
        "sot_explicit_B_FL_sigma_cross_m_T": explicit_fl,
        "sot_effective_field_magnitude_T": math.hypot(explicit_dl, explicit_fl),
    }


def annotate_sot_drive_for_material(
    drive: Dict[str, Any],
    instantaneous_material: Dict[str, Any],
    region_material_values: Sequence[Dict[str, Any]],
    thickness_m: float,
    cfg: Dict[str, Any],
    ms_floor_A_per_m: float | None = None,
) -> None:
    """Attach coefficients using the smallest active-region Ms for safety."""
    if not bool(drive.get("has_sot")):
        return
    ms_values = [float(instantaneous_material.get("Ms_T_A_per_m", 0.0))]
    if ms_floor_A_per_m is not None:
        ms_values.append(float(ms_floor_A_per_m))
    for region_value in region_material_values:
        region_inst = region_value.get("instantaneous_material", {})
        if isinstance(region_inst, dict):
            ms_values.append(float(region_inst.get("Ms_T_A_per_m", ms_values[0])))
    positive_ms = [value for value in ms_values if value > 0.0 and math.isfinite(value)]
    ms_min = min(positive_ms) if positive_ms else 0.0
    alpha = float(instantaneous_material.get("alpha_T", 0.0))
    coeffs = sot_field_coefficients_T(
        float(drive.get("J_A_per_m2", 0.0)),
        float(drive.get("Pol", 0.0)),
        float(drive.get("EpsilonPrime", 0.0)),
        ms_min,
        float(thickness_m),
        alpha,
    )
    rotation_field = max(float(coeffs["sot_effective_field_magnitude_T"]), 1.0e-30)
    recommended_max_dt_s = min(
        float(cfg["max_dt_s"]),
        float(cfg["sot_max_rotation_per_step_rad"]) / (GAMMA_LL_RAD_PER_T_S * rotation_field),
    )
    drive.update(coeffs)
    drive.update({
        "FreeLayerThickness_m": float(thickness_m),
        "sot_coefficient_basis": "DL=m_cross_sigma_cross_m;FL=sigma_cross_m;gilbert_input",
        "sot_recommended_max_dt_s": recommended_max_dt_s,
    })


def annotate_sot_segments(
    segments: Sequence[Dict[str, Any]],
    drive_summary: Dict[str, Any],
    raw: Dict[str, Any],
    cfg: Dict[str, Any],
    ms_floor_A_per_m: float | None = None,
) -> None:
    """Attach physical SOT coefficients and a safe per-segment MaxDt."""
    thickness_m = float(raw.get("thickness_m", 0.0))
    active_drives: List[Dict[str, Any]] = []
    for segment in segments:
        drive = segment.get("drive", {})
        if not isinstance(drive, dict) or not bool(drive.get("has_sot")):
            continue
        annotate_sot_drive_for_material(
            drive,
            segment.get("instantaneous_material", {}),
            segment.get("region_material_values", []) or [],
            thickness_m,
            cfg,
            ms_floor_A_per_m,
        )
        active_drives.append(drive)

    if not active_drives:
        return
    drive_summary.update({
        "has_sot": True,
        "has_sot_like_slonczewski_proxy": False,
        "rendered_torque_model": SOT_RENDERED_TORQUE_MODEL,
        "max_abs_sot_B_DL_T": max(abs(float(d["sot_B_DL_T"])) for d in active_drives),
        "max_abs_sot_B_FL_T": max(abs(float(d["sot_B_FL_T"])) for d in active_drives),
        "max_sot_effective_field_magnitude_T": max(
            float(d["sot_effective_field_magnitude_T"]) for d in active_drives
        ),
        "min_sot_recommended_max_dt_s": min(float(d["sot_recommended_max_dt_s"]) for d in active_drives),
    })


def sot_segment_rejection_reasons(
    segments: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> List[str]:
    reasons: List[str] = []
    max_field = float(cfg.get("sot_max_effective_field_T", math.inf))
    min_dt = float(cfg.get("sot_min_max_dt_s", 0.0))
    for segment in segments:
        drive = segment.get("drive", {})
        if not isinstance(drive, dict) or not bool(drive.get("has_sot")):
            continue
        field = float(drive.get("sot_effective_field_magnitude_T", math.inf))
        recommended_dt = float(drive.get("sot_recommended_max_dt_s", 0.0))
        if not math.isfinite(field) or field > max_field:
            reasons.append("sot_effective_field_exceeds_limit")
        if not math.isfinite(recommended_dt) or recommended_dt < min_dt:
            reasons.append("sot_required_timestep_below_limit")
    return list(dict.fromkeys(reasons))


def sample_drive_segment(
    spec_json: Dict[str, Any],
    cfg: Dict[str, Any],
    drive_type: str,
    raw: Dict[str, Any],
    sampler: SpecSampler,
    rng: random.Random,
    active: bool,
    segment_index: int,
) -> Dict[str, Any]:
    kind = canonical_drive_kind(drive_type)
    active = active and kind != "none"
    out: Dict[str, Any] = {
        "drive_type": drive_type,
        "active_kind": kind,
        "active": active,
        "has_field": False,
        "has_zhang_li": False,
        "has_slonczewski": False,
        "has_sot": False,
        "has_sot_like_slonczewski_proxy": False,
        "rendered_torque_model": "none",
        "B_ext_T": [0.0, 0.0, 0.0],
        "J_A_per_m2": 0.0,
        "J_vector_A_per_m2": [0.0, 0.0, 0.0],
        "polarization": [0.0, -1.0, 0.0],
        "spatial_profile": "global_uniform",
        "profile_parameters": {},
        "region_values": [],
        "Pol": cfg["default_pol"],
        "Lambda": cfg["default_lambda_sl"],
        "EpsilonPrime": cfg["default_epsilon_prime"],
    }
    if not active:
        return out

    dp = spec_json["drive_protocol"]
    if kind == "field":
        node = dp["external_field"]
        abs_b = float(sampler.sample(node["magnitude_abs_b"], f"drive_protocol.external_field.segment_{segment_index}.magnitude_abs_b"))
        if drive_type in {"strong_field_pulse", "topology_collision_setup"}:
            abs_b = max(abs_b, rng.uniform(0.3, 0.8))
        direction_name = str(sampler.sample(node["direction"], f"drive_protocol.external_field.segment_{segment_index}.direction"))
        direction = direction_from_spec(direction_name, rng)
        b_t = abs_b * MU0 * raw["Ms_ref_A_per_m"]
        out.update({
            "has_field": True,
            "rendered_torque_model": "field",
            "B_ext_T": [b_t * x for x in direction],
            "normalized_abs_b": abs_b,
            "direction_name": direction_name,
            "direction": direction,
            "spatial_profile": sampler.sample(node["spatial_profile"], f"drive_protocol.external_field.segment_{segment_index}.spatial_profile"),
            "profile_parameters": sampler.sample_params(node["local_profile_parameters"], f"drive_protocol.external_field.segment_{segment_index}.local_profile_parameters"),
        })
    elif kind == "zhang_li":
        node = dp["zhang_li_stt"]
        j = float(sampler.sample(node["raw_current_density_A_per_m2"], f"drive_protocol.zhang_li_stt.segment_{segment_index}.J"))
        direction_name = str(sampler.sample(node["current_direction"], f"drive_protocol.zhang_li_stt.segment_{segment_index}.current_direction"))
        direction = direction_from_spec(direction_name, rng)
        beta = float(sampler.sample(node["beta_ZL"], f"drive_protocol.zhang_li_stt.segment_{segment_index}.beta_ZL"))
        out.update({
            "has_zhang_li": True,
            "rendered_torque_model": "zhang_li",
            "J_A_per_m2": j,
            "J_vector_A_per_m2": [j * x for x in direction],
            "current_direction_name": direction_name,
            "current_direction": direction,
            "beta_ZL": beta,
            "spatial_profile": sampler.sample(node["spatial_profile"], f"drive_protocol.zhang_li_stt.segment_{segment_index}.spatial_profile"),
            "profile_parameters": sampler.sample_params(
                node.get("local_profile_parameters", {}),
                f"drive_protocol.zhang_li_stt.segment_{segment_index}.local_profile_parameters",
            ),
        })
    elif kind == "slonczewski":
        node = dp["slonczewski_stt"]
        j = float(sampler.sample(node["raw_current_density_A_per_m2"], f"drive_protocol.slonczewski_stt.segment_{segment_index}.J"))
        p_name = str(sampler.sample(node["p_SL_direction"], f"drive_protocol.slonczewski_stt.segment_{segment_index}.p_SL_direction"))
        p = direction_from_spec(p_name, rng)
        contact_radius_px = sampler.sample(node["contact_radius_px"], f"drive_protocol.slonczewski_stt.segment_{segment_index}.contact_radius_px")
        profile_parameters = sampler.sample_params(
            node.get("local_profile_parameters", {}),
            f"drive_protocol.slonczewski_stt.segment_{segment_index}.local_profile_parameters",
        )
        profile_parameters.setdefault("contact_radius_px", contact_radius_px)
        out.update({
            "has_slonczewski": True,
            "rendered_torque_model": "slonczewski",
            "J_A_per_m2": j,
            "J_vector_A_per_m2": [0.0, 0.0, j],
            "polarization": p,
            "polarization_name": p_name,
            "spatial_profile": sampler.sample(node["spatial_profile"], f"drive_protocol.slonczewski_stt.segment_{segment_index}.spatial_profile"),
            "contact_radius_px": contact_radius_px,
            "profile_parameters": profile_parameters,
        })
    elif kind == "sot":
        node = dp["sot"]
        j = float(sampler.sample(node["raw_current_density_A_per_m2"], f"drive_protocol.sot.segment_{segment_index}.J"))
        if drive_type in {"strong_current", "local_write_delete"}:
            j = max(j, rng.uniform(8e11, 3e12))
        j_direction = random_in_plane(rng)
        sigma_mode = str(sampler.sample(node["sigma_SOT_mode"], f"drive_protocol.sot.segment_{segment_index}.sigma_SOT_mode"))
        sigma = direction_from_spec(sigma_mode, rng, j_direction)
        theta_node = node.get("theta_DL_eff")
        theta_dl_eff = (
            float(sampler.sample(theta_node, f"drive_protocol.sot.segment_{segment_index}.theta_DL_eff"))
            if isinstance(theta_node, dict)
            else float(cfg["default_sot_theta_dl_eff"])
        )
        if theta_dl_eff < 0.0:
            sigma = [-value for value in sigma]
        pol = abs(theta_dl_eff)
        r_fl_dl = float(sampler.sample(node["r_FL_DL"], f"drive_protocol.sot.segment_{segment_index}.r_FL_DL"))
        epsilon_prime = sot_epsilon_prime(pol, r_fl_dl)
        out.update({
            "has_sot": True,
            "has_sot_like_slonczewski_proxy": False,
            "rendered_torque_model": SOT_RENDERED_TORQUE_MODEL,
            "sampled_torque_intent": "effective_spin_hall_sot",
            "J_A_per_m2": j,
            "J_vector_A_per_m2": [0.0, 0.0, j],
            "solver_J_vector_A_per_m2": [0.0, 0.0, j],
            "charge_current_direction": j_direction,
            "charge_current_vector_A_per_m2": [j * x for x in j_direction],
            "theta_DL_eff": theta_dl_eff,
            "Pol": pol,
            "Lambda": 1.0,
            "r_FL_DL": r_fl_dl,
            "EpsilonPrime": epsilon_prime,
            "sigma_SOT_mode": sigma_mode,
            "sigma_SOT": sigma,
            "polarization": sigma,
            "spatial_profile": sampler.sample(node["spatial_profile"], f"drive_protocol.sot.segment_{segment_index}.spatial_profile"),
            "profile_parameters": sampler.sample_params(
                node.get("local_profile_parameters", {}),
                f"drive_protocol.sot.segment_{segment_index}.local_profile_parameters",
            ),
        })
    return out


def build_segments(
    spec_json: Dict[str, Any],
    cfg: Dict[str, Any],
    profile: str,
    raw: Dict[str, Any],
    temp_params: Dict[str, Any],
    time_info: Dict[str, Any],
    drive_type: str,
    sampler: SpecSampler,
    rng: random.Random,
) -> Tuple[str, Dict[str, Any], str, Dict[str, Any], List[Dict[str, Any]]]:
    segment_mode, segment_mode_params = sample_segment_mode(spec_json, sampler, drive_type)
    condition_total_s = float(time_info.get("condition_T_end_s", time_info["T_end_s"]))
    bounds = build_segment_bounds(segment_mode, segment_mode_params, condition_total_s, rng)
    T_mode, temp_schedule, T_values = sample_temperature_schedule(spec_json, sampler, temp_params, bounds)
    segments: List[Dict[str, Any]] = []
    for i, ((start_s, end_s, drive_active), T_K) in enumerate(zip(bounds, T_values)):
        inst = instantaneous_params(T_K, raw, temp_params)
        drive = sample_drive_segment(spec_json, cfg, drive_type, raw, sampler, rng, drive_active, i)
        segment = {
            "segment_index": i,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": end_s - start_s,
            "start_ns": start_s * 1e9,
            "end_ns": end_s * 1e9,
            "duration_ns": (end_s - start_s) * 1e9,
            "condition_segment_index": i,
            "condition_start_s": start_s,
            "condition_end_s": end_s,
            "condition_start_ns": start_s * 1e9,
            "condition_end_ns": end_s * 1e9,
            "midpoint_u": 0.5 * (start_s + end_s) / condition_total_s,
            "global_midpoint_u": 0.5 * (start_s + end_s) / condition_total_s,
            "segment_role": "condition",
            "condition_segment": True,
            "pre_relaxation_segment": False,
            "T_K": T_K,
            "T_midpoint_K": T_K,
            "temperature": {
                "T_schedule_mode": T_mode,
                "T_function_id": temp_schedule["T_function_id"],
                "theta_T": temp_schedule["theta_T"],
                "theta_T_order": temp_schedule["theta_T_order"],
                "T_schedule_parameters": deepcopy(temp_schedule["T_schedule_parameters"]),
                "T_midpoint_K": T_K,
                "T_midpoint_semantics": "diagnostic sample at this condition segment midpoint",
            },
            "instantaneous_material": inst,
            "drive": drive,
            "pair_sampling_allowed": True,
        }
        segments.append(segment)
    active_torque_models = sorted({
        s["drive"].get("rendered_torque_model", "none")
        for s in segments
        if s["drive"].get("active") and s["drive"].get("rendered_torque_model", "none") != "none"
    })
    drive_summary = {
        "has_field": any(s["drive"]["has_field"] for s in segments),
        "has_zhang_li": any(s["drive"]["has_zhang_li"] for s in segments),
        "has_slonczewski": any(s["drive"]["has_slonczewski"] for s in segments),
        "has_sot": any(s["drive"]["has_sot"] for s in segments),
        "has_sot_like_slonczewski_proxy": any(
            s["drive"].get("has_sot_like_slonczewski_proxy", False) for s in segments
        ),
        "drive_segment_mode": segment_mode,
        "segment_count": len(segments),
        "active_drive_mask": {
            "field": any(s["drive"]["has_field"] for s in segments),
            "zhang_li": any(s["drive"]["has_zhang_li"] for s in segments),
            "slonczewski": any(s["drive"]["has_slonczewski"] for s in segments),
            "sot": any(s["drive"]["has_sot"] for s in segments),
            "sot_like_slonczewski_proxy": any(
                s["drive"].get("has_sot_like_slonczewski_proxy", False) for s in segments
            ),
        },
        "drive_profile_type": "segment_piecewise_constant",
        "rendered_torque_model": active_torque_models[0] if len(active_torque_models) == 1 else "mixed_or_none",
        "segment_mode_parameters": segment_mode_params,
    }
    return segment_mode, drive_summary, T_mode, temp_schedule, segments


def pre_relaxation_settings(spec_json: Dict[str, Any]) -> Dict[str, Any]:
    node = spec_json.get("pre_relaxation_segment", {})
    if not isinstance(node, dict):
        node = {}
    enabled = bool(node.get("enabled", False))
    duration_ns = float(node.get("duration_ns", 1.0))
    if duration_ns <= 0.0:
        enabled = False
        duration_ns = 0.0
    return {
        "enabled": enabled,
        "duration_ns": duration_ns,
        "duration_s": duration_ns * 1e-9,
    }


def update_time_info_for_pre_relaxation(time_info: Dict[str, Any], pre: Dict[str, Any]) -> None:
    condition_s = float(time_info.get("condition_T_end_s", time_info["T_end_s"]))
    condition_ns = condition_s * 1e9
    pre_s = float(pre["duration_s"]) if pre.get("enabled") else 0.0
    total_s = pre_s + condition_s
    save_dt_s = float(time_info["save_dt_s"])
    time_info.update({
        "condition_T_end_s": condition_s,
        "condition_T_end_ns": condition_ns,
        "condition_start_s": pre_s,
        "condition_start_ns": pre_s * 1e9,
        "pre_relaxation_enabled": bool(pre.get("enabled")),
        "pre_relaxation_duration_s": pre_s,
        "pre_relaxation_duration_ns": pre_s * 1e9,
        "T_end_s": total_s,
        "T_end_ns": total_s * 1e9,
        "saved_frame_count": max(1, int(math.floor((total_s + 1e-18) / save_dt_s)) + 1),
    })


def schedule_u_for_time(meta: Dict[str, Any], t_s: float, segment: Dict[str, Any] | None = None) -> float:
    if segment is not None and segment.get("segment_role") == "pre_relaxation":
        return float(segment.get("temperature_schedule_u", 0.0))
    condition_start_s = float(meta.get("time", {}).get("condition_start_s", 0.0))
    condition_total_s = float(meta.get("time", {}).get("condition_T_end_s", meta.get("time", {}).get("T_end_s", 1.0)))
    return clamp((float(t_s) - condition_start_s) / max(condition_total_s, 1e-30), 0.0, 1.0)


def refresh_temperature_schedule_segment_arrays(
    temperature_schedule: Dict[str, Any],
    segments: Sequence[Dict[str, Any]],
    condition_start_s: float,
    condition_total_s: float,
) -> None:
    temperature_schedule["condition_start_s"] = condition_start_s
    temperature_schedule["condition_end_s"] = condition_start_s + condition_total_s
    temperature_schedule["condition_T_end_s"] = condition_total_s
    temperature_schedule["condition_T_segment_start_s"] = [
        seg["condition_start_s"] for seg in segments if seg.get("condition_segment", True)
    ]
    temperature_schedule["condition_T_segment_end_s"] = [
        seg["condition_end_s"] for seg in segments if seg.get("condition_segment", True)
    ]
    temperature_schedule["T_segment_values_K"] = [seg["T_K"] for seg in segments]
    base_values = []
    peak_delta_values = []
    for seg in segments:
        u = float(seg.get("midpoint_u", 0.0))
        base_T, peak_delta_T = local_temperature_components_at_u(temperature_schedule, u)
        base_values.append(base_T)
        peak_delta_values.append(peak_delta_T)
    temperature_schedule["T_segment_base_values_K"] = base_values
    temperature_schedule["T_segment_peak_delta_values_K"] = peak_delta_values
    temperature_schedule["T_segment_start_s"] = [seg["start_s"] for seg in segments]
    temperature_schedule["T_segment_end_s"] = [seg["end_s"] for seg in segments]


def prepend_pre_relaxation_segment(
    spec_json: Dict[str, Any],
    cfg: Dict[str, Any],
    raw: Dict[str, Any],
    temp_params: Dict[str, Any],
    time_info: Dict[str, Any],
    T_mode: str,
    temperature_schedule: Dict[str, Any],
    drive_summary: Dict[str, Any],
    segments: List[Dict[str, Any]],
    sampler: SpecSampler,
    rng: random.Random,
) -> None:
    pre = pre_relaxation_settings(spec_json)
    condition_total_s = float(time_info.get("condition_T_end_s", time_info["T_end_s"]))
    update_time_info_for_pre_relaxation(time_info, pre)
    pre_s = float(time_info["pre_relaxation_duration_s"])
    total_s = float(time_info["T_end_s"])

    for i, seg in enumerate(segments, start=1 if pre.get("enabled") else 0):
        original_start_s = float(seg["start_s"])
        original_end_s = float(seg["end_s"])
        shifted_start_s = original_start_s + pre_s
        shifted_end_s = original_end_s + pre_s
        seg.update({
            "segment_index": i,
            "condition_segment_index": i - 1 if pre.get("enabled") else i,
            "condition_start_s": original_start_s,
            "condition_end_s": original_end_s,
            "condition_start_ns": original_start_s * 1e9,
            "condition_end_ns": original_end_s * 1e9,
            "start_s": shifted_start_s,
            "end_s": shifted_end_s,
            "duration_s": shifted_end_s - shifted_start_s,
            "start_ns": shifted_start_s * 1e9,
            "end_ns": shifted_end_s * 1e9,
            "duration_ns": (shifted_end_s - shifted_start_s) * 1e9,
            "midpoint_u": 0.5 * (original_start_s + original_end_s) / condition_total_s,
            "global_midpoint_u": 0.5 * (shifted_start_s + shifted_end_s) / total_s,
            "segment_role": "condition",
            "condition_segment": True,
            "pre_relaxation_segment": False,
        })

    if pre.get("enabled"):
        pre_T = temperature_schedule_value_at_u(temperature_schedule, 0.0)
        pre_inst = instantaneous_params(pre_T, raw, temp_params)
        pre_drive = sample_drive_segment(spec_json, cfg, "none", raw, sampler, rng, False, 0)
        pre_segment = {
            "segment_index": 0,
            "condition_segment_index": None,
            "start_s": 0.0,
            "end_s": pre_s,
            "duration_s": pre_s,
            "start_ns": 0.0,
            "end_ns": pre_s * 1e9,
            "duration_ns": pre_s * 1e9,
            "condition_start_s": None,
            "condition_end_s": None,
            "condition_start_ns": None,
            "condition_end_ns": None,
            "midpoint_u": 0.0,
            "global_midpoint_u": 0.5 * pre_s / total_s,
            "temperature_schedule_u": 0.0,
            "segment_role": "pre_relaxation",
            "condition_segment": False,
            "pre_relaxation_segment": True,
            "pair_sampling_allowed": True,
            "T_K": pre_T,
            "T_midpoint_K": pre_T,
            "temperature": {
                "T_schedule_mode": T_mode,
                "T_function_id": temperature_schedule["T_function_id"],
                "theta_T": temperature_schedule["theta_T"],
                "theta_T_order": temperature_schedule["theta_T_order"],
                "T_schedule_parameters": deepcopy(temperature_schedule["T_schedule_parameters"]),
                "T_midpoint_K": pre_T,
                "T_midpoint_semantics": "pre-relaxation uses the condition schedule start temperature",
            },
            "instantaneous_material": pre_inst,
            "drive": pre_drive,
        }
        segments.insert(0, pre_segment)

    refresh_temperature_schedule_segment_arrays(
        temperature_schedule,
        segments,
        float(time_info["condition_start_s"]),
        condition_total_s,
    )
    drive_summary["segment_count"] = len(segments)
    drive_summary["condition_segment_count"] = sum(1 for seg in segments if seg.get("condition_segment", True))
    drive_summary["pre_relaxation_enabled"] = bool(pre.get("enabled"))
    drive_summary["pre_relaxation_duration_ns"] = float(time_info["pre_relaxation_duration_ns"])


def local_profile_min_feature_px(cfg: Dict[str, Any], grid: Dict[str, Any], r_min_cells: float | None) -> float:
    gx, gy = tuple(cfg.get("control_grid", (8, 8)))
    tile_px = min(float(grid["Nx"]) / max(1.0, float(gx)), float(grid["Ny"]) / max(1.0, float(gy)))
    candidates = [
        float(cfg.get("local_profile_min_px", 8.0)),
        float(cfg.get("local_profile_min_region_fraction", 0.25)) * tile_px,
    ]
    if r_min_cells is not None and math.isfinite(float(r_min_cells)):
        candidates.append(float(cfg.get("local_profile_min_rmin_factor", 2.0)) * float(r_min_cells))
    return max(candidates)


def enforce_local_drive_profile_resolution(
    cfg: Dict[str, Any],
    grid: Dict[str, Any],
    segments: Sequence[Dict[str, Any]],
) -> int:
    adjusted_count = 0
    for seg in segments:
        drive = seg.get("drive", {})
        profile = str(drive.get("spatial_profile", "global_uniform"))
        if profile == "global_uniform":
            continue
        params = drive.setdefault("profile_parameters", {})
        inst = seg.get("instantaneous_material", {})
        min_px = local_profile_min_feature_px(cfg, grid, inst.get("r_min_T"))
        adjustments: List[Dict[str, Any]] = []

        def raise_to_floor(key: str, current_px: float) -> None:
            nonlocal adjusted_count
            if current_px >= min_px:
                return
            params[key] = min_px
            if key == "contact_radius_px":
                drive["contact_radius_px"] = min_px
            adjustments.append({
                "parameter": key,
                "original_px": current_px,
                "adjusted_px": min_px,
                "min_effective_px": min_px,
                "r_min_cells": inst.get("r_min_T"),
            })
            adjusted_count += 1

        if profile in GAUSSIAN_LOCAL_PROFILES:
            default_sigma = float(params.get("contact_radius_px", params.get("spot_sigma_norm", 0.1) * grid["Nx"]))
            sigma_key = "spot_sigma_px" if "spot_sigma_px" in params or "contact_radius_px" not in params else "contact_radius_px"
            raise_to_floor(sigma_key, float(params.get(sigma_key, default_sigma)))
        if profile in STRIPE_LOCAL_PROFILES:
            raise_to_floor("stripe_width_px", float(params.get("stripe_width_px", 12.0)))
        if profile in DISK_LOCAL_PROFILES or profile in RING_LOCAL_PROFILES:
            radius = float(params.get("contact_radius_px", params.get("spot_sigma_px", 24.0)))
            raise_to_floor("contact_radius_px", radius)

        if adjustments:
            drive["local_profile_resolution_adjustments"] = adjustments
            drive["local_profile_min_effective_px"] = min_px
    return adjusted_count


def enforce_local_magnetoelastic_profile_resolution(
    cfg: Dict[str, Any],
    grid: Dict[str, Any],
    segments: Sequence[Dict[str, Any]],
    magnetoelastic: Dict[str, Any],
) -> int:
    if magnetoelastic.get("MAGNETOELASTIC_MODE") != "local_strain_defect":
        return 0
    variables = magnetoelastic.get("sampled_variables")
    if not isinstance(variables, dict):
        return 0
    r_min_values = [
        float(seg["instantaneous_material"]["r_min_T"])
        for seg in segments
        if seg.get("instantaneous_material", {}).get("r_min_T") is not None
    ]
    min_px = local_profile_min_feature_px(cfg, grid, min(r_min_values) if r_min_values else None)
    adjustments: List[Dict[str, Any]] = []
    for key in ("spot_sigma_px", "smoothing_px"):
        if key not in variables:
            continue
        original_px = float(variables[key])
        if original_px >= min_px:
            continue
        variables[key] = min_px
        if isinstance(magnetoelastic.get("strain_summary"), dict):
            magnetoelastic["strain_summary"][key] = min_px
        adjustments.append({
            "parameter": key,
            "original_px": original_px,
            "adjusted_px": min_px,
            "min_effective_px": min_px,
            "r_min_cells": min(r_min_values) if r_min_values else None,
        })
    if adjustments:
        magnetoelastic["local_profile_resolution_adjustments"] = adjustments
        magnetoelastic["local_profile_min_effective_px"] = min_px
    return len(adjustments)


def make_control_regions(cfg: Dict[str, Any], grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    gx, gy = tuple(cfg["control_grid"])
    gx = int(gx)
    gy = int(gy)
    width_m = grid["Lx_m"] / gx
    height_m = grid["Ly_m"] / gy
    regions: List[Dict[str, Any]] = []
    for iy in range(gy):
        for ix in range(gx):
            regions.append({
                "region_id": 1 + iy * gx + ix,
                "ix": ix,
                "iy": iy,
                "x_m": -0.5 * grid["Lx_m"] + (ix + 0.5) * width_m,
                "y_m": -0.5 * grid["Ly_m"] + (iy + 0.5) * height_m,
                "width_m": width_m,
                "height_m": height_m,
            })
    return regions


def profile_value(region: Dict[str, Any], profile: str, params: Dict[str, Any], grid: Dict[str, Any]) -> float:
    if profile == "global_uniform":
        return 1.0
    x_px = (region["x_m"] / grid["dx_m"]) + 0.5 * grid["Nx"]
    y_px = (region["y_m"] / grid["dy_m"]) + 0.5 * grid["Ny"]
    cx = float(params.get("spot_x0_norm", params.get("x0_norm", 0.5))) * grid["Nx"]
    cy = float(params.get("spot_y0_norm", params.get("y0_norm", 0.5))) * grid["Ny"]
    sigma_default = params.get("contact_radius_px", params.get("spot_sigma_norm", 0.1) * grid["Nx"])
    sigma = float(params.get("spot_sigma_px", sigma_default))
    if profile in GAUSSIAN_LOCAL_PROFILES:
        return math.exp(-((x_px - cx) ** 2 + (y_px - cy) ** 2) / (2.0 * max(1.0, sigma) ** 2))
    if profile in STRIPE_LOCAL_PROFILES:
        width = max(1.0, float(params.get("stripe_width_px", 12.0)))
        return 1.0 if abs(y_px - cy) <= 0.5 * width else 0.0
    if profile in DISK_LOCAL_PROFILES or profile in RING_LOCAL_PROFILES:
        rr = math.sqrt((x_px - cx) ** 2 + (y_px - cy) ** 2)
        radius = float(params.get("contact_radius_px", params.get("spot_sigma_px", 24.0)))
        if profile in RING_LOCAL_PROFILES:
            return 1.0 if abs(rr - radius) < 0.25 * radius else 0.0
        return 1.0 if rr <= radius else 0.0
    return 1.0


def attach_region_material_values(
    cfg: Dict[str, Any],
    grid: Dict[str, Any],
    raw: Dict[str, Any],
    temp_params: Dict[str, Any],
    temperature_schedule: Dict[str, Any],
    segments: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
) -> None:
    """Attach per-region material values that mirror the rendered MuMax3 state."""
    threshold = float(cfg["local_profile_region_threshold"])
    mode = temperature_schedule.get("T_schedule_mode")
    tc = float(temp_params.get("Tc_K", 1e9))
    cap = 0.92 * tc
    local_params = temperature_schedule.get("local_T_profile_parameters", temperature_schedule.get("sampled_variables", {}))
    base_values = temperature_schedule.get("T_segment_base_values_K", [])
    peak_delta_values = temperature_schedule.get("T_segment_peak_delta_values_K", [])

    for seg in segments:
        seg_i = int(seg["segment_index"])
        values = []
        u = float(seg.get("midpoint_u", 0.5))
        if mode == "local_temperature_spot":
            params = temperature_schedule_parameters(temperature_schedule)
            if {"T_base_K", "delta_T_K", "pulse_center_norm", "pulse_width_norm"} <= params.keys():
                base, peak_delta = local_temperature_components_at_u(temperature_schedule, u)
            else:
                base = float(base_values[seg_i] if seg_i < len(base_values) else seg["T_K"])
                peak_delta = float(peak_delta_values[seg_i] if seg_i < len(peak_delta_values) else 0.0)
            for region in regions:
                scale = profile_value(region, "gaussian_spot", local_params, grid)
                if abs(scale) < threshold:
                    scale = 0.0
                temp = clamp(base + peak_delta * scale, 0.0, cap)
                inst = instantaneous_params(temp, raw, temp_params)
                attach_resolution_to_instantaneous_material(inst, raw, grid)
                values.append({
                    "region_id": region["region_id"],
                    "scale": scale,
                    "T_K": temp,
                    "instantaneous_material": inst,
                })
        else:
            inst = seg["instantaneous_material"]
            for region in regions:
                values.append({
                    "region_id": region["region_id"],
                    "scale": 1.0,
                    "T_K": inst["T_K"],
                    "instantaneous_material": deepcopy(inst),
                })
        seg["region_material_values"] = values


def collect_temperature_schedule_probe_materials(
    cfg: Dict[str, Any],
    grid: Dict[str, Any],
    raw: Dict[str, Any],
    temp_params: Dict[str, Any],
    temperature_schedule: Dict[str, Any],
    regions: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    mode = temperature_schedule.get("T_schedule_mode")
    tc = float(temp_params.get("Tc_K", 1e9))
    cap = 0.92 * tc
    us = temperature_probe_us(temperature_schedule)
    materials: List[Dict[str, Any]] = []
    if mode == "local_temperature_spot":
        threshold = float(cfg["local_profile_region_threshold"])
        local_params = temperature_schedule.get(
            "local_T_profile_parameters",
            temperature_schedule_parameters(temperature_schedule),
        )
        for u in us:
            base, peak_delta = local_temperature_components_at_u(temperature_schedule, float(u))
            for region in regions:
                scale = profile_value(region, "gaussian_spot", local_params, grid)
                if abs(scale) < threshold:
                    scale = 0.0
                temp = clamp(base + peak_delta * scale, 0.0, cap)
                inst = instantaneous_params(temp, raw, temp_params)
                attach_resolution_to_instantaneous_material(inst, raw, grid)
                materials.append(inst)
    else:
        for u in us:
            temp = temperature_schedule_value_at_u(temperature_schedule, float(u))
            inst = instantaneous_params(temp, raw, temp_params)
            attach_resolution_to_instantaneous_material(inst, raw, grid)
            materials.append(inst)
    return materials


def attach_region_values(
    cfg: Dict[str, Any],
    grid: Dict[str, Any],
    segments: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
) -> None:
    threshold = float(cfg["local_profile_region_threshold"])
    for seg in segments:
        drive = seg["drive"]
        profile = drive.get("spatial_profile", "global_uniform")
        params = drive.get("profile_parameters", {})
        values = []
        for region in regions:
            scale = profile_value(region, profile, params, grid)
            if abs(scale) < threshold:
                scale = 0.0
            values.append({
                "region_id": region["region_id"],
                "scale": scale,
                "B_ext_T": [scale * x for x in drive.get("B_ext_T", [0.0, 0.0, 0.0])],
                "J_vector_A_per_m2": [scale * x for x in drive.get("J_vector_A_per_m2", [0.0, 0.0, 0.0])],
            })
        drive["region_values"] = values


def unique_drive_type_order(spec_json: Dict[str, Any]) -> List[str]:
    seen: Dict[str, None] = {}
    for node in spec_json["drive_protocol"]["drive_type_by_profile"].values():
        for item in node.get("items", []):
            seen[str(item["value"])] = None
    for rendered in DRIVE_TYPE_RENDER_ALIASES.values():
        seen.setdefault(str(rendered), None)
    seen.setdefault("none", None)
    return list(seen.keys())


def categorical_encoding(value: str, order: Sequence[str]) -> Dict[str, Any]:
    return {"value": value, "order": list(order), "one_hot": one_hot(value, order)}


def build_categorical_encodings(spec_json: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "DMI_TYPE": categorical_encoding(
            str(metadata["raw_physical_params_ref"].get("DMI_TYPE", "none")),
            ["none", "interfacial", "bulk"],
        ),
        "TEMP_PARAM_MODE": categorical_encoding(
            metadata["TEMP_PARAM_MODE"],
            list(spec_json["temp_param_mode"]["modes"].keys()),
        ),
        "T_schedule_mode": categorical_encoding(
            metadata["T_schedule_mode"],
            spec_json["T_schedule_mode"]["model_encoding"]["mode_input"]["order"],
        ),
        "default_boundary": categorical_encoding(
            metadata["default_boundary"],
            list(spec_json["boundary"]["modes"].keys()),
        ),
        "BOUNDARY_MODE": categorical_encoding(
            metadata["boundary_mode"],
            list(spec_json["boundary"]["modes"].keys()),
        ),
        "geometry_mode": categorical_encoding(
            metadata["geometry_mode"],
            list(spec_json["geometry_sampling"]["modes"].keys()),
        ),
        "material_family": categorical_encoding(
            metadata["material_family"],
            list(spec_json["material_family"]["families"].keys()),
        ),
        "CUBIC_ANISOTROPY_MODE": categorical_encoding(
            metadata["CUBIC_ANISOTROPY_MODE"],
            list(spec_json["cubic_anisotropy"]["modes"].keys()),
        ),
        "MAGNETOELASTIC_MODE": categorical_encoding(
            metadata["MAGNETOELASTIC_MODE"],
            list(spec_json["magnetoelastic_coupling"]["modes"].keys()),
        ),
        "drive_type": categorical_encoding(
            metadata["drive_type"],
            unique_drive_type_order(spec_json),
        ),
    }


def sample_trajectory(
    index: int,
    split: str,
    spec_json: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    seed = int(cfg["random_seed"]) + 1000003 * index
    rejection_reason_counts: Dict[str, int] = {}
    primary_rejection_reason_counts: Dict[str, int] = {}
    for attempt in range(int(cfg["max_attempt_factor"])):
        local_rng = random.Random(seed + 7919 * attempt)
        sampler = SpecSampler(local_rng)
        grid = grid_from_spec(spec_json)
        profile = str(cfg.get("force_dataset_profile", "") or sample_dataset_profile(spec_json, sampler))
        family, raw, normalized, validity_tags = sample_material(spec_json, profile, sampler, grid, local_rng)
        rejection_reasons = material_rejection_reasons(raw, normalized, family, profile)
        if rejection_reasons:
            record_rejection(rejection_reason_counts, primary_rejection_reason_counts, rejection_reasons)
            continue

        grid["dz_m"] = raw["thickness_m"]
        validity_tags = list(dict.fromkeys(validity_tags))
        temp_mode, temp_params = sample_temperature_params(spec_json, family, sampler)
        time_regime, time_info = sample_time_regime(spec_json, profile, sampler, cfg)
        geometry_mode, geometry_params, geometry_tags = sample_geometry(spec_json, sampler, family, normalized, local_rng)
        rejection_reasons = geometry_rejection_reasons(geometry_mode, geometry_params, normalized)
        if rejection_reasons:
            record_rejection(rejection_reason_counts, primary_rejection_reason_counts, rejection_reasons)
            continue
        default_boundary, boundary_mode, boundary, boundary_tags = sample_boundary(spec_json, geometry_mode, profile, sampler)
        cubic_mode, cubic, cubic_tags = sample_cubic(spec_json, sampler, raw, local_rng)
        magnetoelastic_mode, magnetoelastic, me_tags = sample_magnetoelastic(spec_json, sampler, raw)
        defect_mode, defect = sample_defects(spec_json, profile, sampler)
        init_family, init, init_tags = sample_init(spec_json, family, geometry_mode, grid, raw, normalized, sampler, local_rng)
        forced_drive_type = str(cfg.get("force_drive_type", "") or "").strip()
        if time_regime == "final_relax":
            # Final relaxation is a true no-drive relaxation; otherwise maxTorque
            # autostop is inconsistent with driven dynamics.
            drive_type = "none"
        else:
            drive_type = forced_drive_type or sample_drive_type(spec_json, profile, sampler)
        if not bool(cfg.get("enable_sot_trajectories", True)) and drive_type in SOT_DRIVE_TYPES:
            record_rejection(
                rejection_reason_counts,
                primary_rejection_reason_counts,
                ["sot_excluded_by_config"],
            )
            continue
        if forced_drive_type and time_regime == "final_relax":
            record_rejection(
                rejection_reason_counts,
                primary_rejection_reason_counts,
                ["forced_drive_incompatible_with_final_relax"],
            )
            continue
        segment_mode, drive, T_mode, temperature_schedule, segments = build_segments(
            spec_json, cfg, profile, raw, temp_params, time_info, drive_type, sampler, local_rng
        )
        prepend_pre_relaxation_segment(
            spec_json,
            cfg,
            raw,
            temp_params,
            time_info,
            T_mode,
            temperature_schedule,
            drive,
            segments,
            sampler,
            local_rng,
        )
        render_audit: List[Dict[str, Any]] = []
        canonicalize_geometry_to_mx3(geometry_mode, geometry_params, render_audit)
        canonicalize_initial_state_to_mx3(init_family, init, render_audit)
        drive_type = canonicalize_drive_segments_to_mx3(drive_type, drive, segments, render_audit)

        raw["Kc1_ref_J_per_m3"] = cubic["Kc1_ref_J_per_m3"]
        raw["Kc2_ref_J_per_m3"] = cubic["Kc2_ref_J_per_m3"]
        raw["B1_ref_J_per_m3"] = magnetoelastic["B1_ref_J_per_m3"]
        raw["B2_ref_J_per_m3"] = magnetoelastic["B2_ref_J_per_m3"]
        normalized["qC1_ref"] = cubic["qC1_ref"]
        normalized["qC2_ref"] = cubic["qC2_ref"]
        normalized["qME_ref"] = magnetoelastic["qME_ref"]
        for seg in segments:
            seg["instantaneous_material"] = instantaneous_params(seg["T_K"], raw, temp_params)
        attach_instantaneous_resolution(segments, raw, grid)
        regions = make_control_regions(cfg, grid)
        attach_region_material_values(cfg, grid, raw, temp_params, temperature_schedule, segments, regions)
        local_profile_adjustment_count = enforce_local_drive_profile_resolution(cfg, grid, segments)
        local_profile_adjustment_count += enforce_local_magnetoelastic_profile_resolution(
            cfg, grid, segments, magnetoelastic
        )
        temperature_probe_materials = collect_temperature_schedule_probe_materials(
            cfg, grid, raw, temp_params, temperature_schedule, regions
        )
        probe_ms = [
            float(material.get("Ms_T_A_per_m", 0.0))
            for material in temperature_probe_materials
            if float(material.get("Ms_T_A_per_m", 0.0)) > 0.0
        ]
        annotate_sot_segments(
            segments,
            drive,
            raw,
            cfg,
            min(probe_ms) if probe_ms else None,
        )
        rejection_reasons = sot_segment_rejection_reasons(segments, cfg)
        if rejection_reasons:
            record_rejection(rejection_reason_counts, primary_rejection_reason_counts, rejection_reasons)
            continue

        resolution_policy = build_resolution_policy(normalized, segments, temperature_probe_materials)
        allow_stress_partition = magnetoelastic_mode != "magnetoelastic_off"
        rejection_reasons = resolution_rejection_reasons(resolution_policy, profile, allow_stress_partition)
        if rejection_reasons:
            record_rejection(rejection_reason_counts, primary_rejection_reason_counts, rejection_reasons)
            continue

        tags = list(dict.fromkeys(
            validity_tags
            + resolution_validity_tags(resolution_policy)
            + geometry_tags
            + boundary_tags
            + cubic_tags
            + me_tags
            + init_tags
        ))
        if local_profile_adjustment_count:
            tags.append("local_profile_resolution_adjusted")
        if render_audit:
            tags.append("rendered_parameter_canonicalized")
        audit_tags: List[str] = []
        if render_audit:
            audit_tags.append("rendered_parameter_canonicalized")
        if geometry_mode in MX3_APPROXIMATED_GEOMETRIES:
            tags.append("mx3_geometry_approximation")
            audit_tags.append("mx3_geometry_approximation")
        if temp_mode == "strong_temp_drift_edge" or temperature_schedule["max_T_over_Tc"] > 0.75:
            tags.append("temperature_strong_drift_validity")
        tags = list(dict.fromkeys(tags))
        attach_region_values(cfg, grid, segments, regions)

        trajectory_id = (
            f"traj_{index:06d}_{profile}_{family}_{geometry_mode}_{init_family}_"
            f"{drive_type}_{time_regime}"
        )
        run_batch_id = str(cfg.get("run_batch_id", "") or "").strip()
        if run_batch_id:
            trajectory_id = f"{safe_identifier(run_batch_id)}_{trajectory_id}"
        metadata = {
            "trajectory_id": trajectory_id,
            "generation_batch_id": safe_identifier(run_batch_id) if run_batch_id else "",
            "random_seed": seed,
            "thermal_noise_seed": derive_thermal_noise_seed(seed, index, attempt),
            "split": split,
            "dataset_profile": profile,
            "material_family": family,
            "geometry_mode": geometry_mode,
            "default_boundary": default_boundary,
            "boundary_mode": boundary_mode,
            "init_family": init_family,
            "drive_type": drive_type,
            "TEMP_PARAM_MODE": temp_mode,
            "T_schedule_mode": T_mode,
            "CUBIC_ANISOTROPY_MODE": cubic_mode,
            "MAGNETOELASTIC_MODE": magnetoelastic_mode,
            "DEFECT_MODE": defect_mode,
            "TIME_REGIME": time_regime,
            "grid": grid,
            "raw_physical_params_ref": raw,
            "temperature_schedule": temperature_schedule,
            "temperature_dependent_params": temp_params,
            "instantaneous_segment_params": [seg["instantaneous_material"] for seg in segments],
            "normalized_params_ref": normalized,
            "resolution_policy": resolution_policy,
            "boundary": boundary,
            "geometry": {"geometry_mode": geometry_mode, "parameters": geometry_params},
            "initial_state": init,
            "cubic_anisotropy": cubic,
            "magnetoelastic": magnetoelastic,
            "defects_and_disorder": defect,
            "drive": drive,
            "segments": segments,
            "control_regions": regions,
            "rendered_parameter_audit": rendered_parameter_audit_dict(render_audit),
            "time": time_info,
            "events": {
                "mandatory_event_frames": spec_json["time_regime"]["mandatory_event_frames"],
                "planned_segment_boundaries_s": [seg["start_s"] for seg in segments] + [segments[-1]["end_s"]],
            },
            "validity_tags": tags,
            "audit_tags": audit_tags,
            "proposal_attempt_count": attempt + 1,
            "rejected_proposal_count": attempt,
            "rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
            "primary_rejection_reason_counts": dict(sorted(primary_rejection_reason_counts.items())),
            "sampling_trace": sampler.trace,
            "source_spec": {
                "schema_version": spec_json.get("schema_version"),
                "source_version": spec_json.get("source_version"),
            },
        }
        metadata["categorical_encodings"] = build_categorical_encodings(spec_json, metadata)
        return metadata

    raise RuntimeError(
        "Unable to sample a valid trajectory for index "
        f"{index}; rejection reasons: {dict(sorted(rejection_reason_counts.items()))}"
    )


def geometry_shape_expr(meta: Dict[str, Any]) -> Tuple[str, str]:
    if geometry_uses_mask(meta):
        return 'imageShape("mask.png")', "mask_png_from_metadata"
    grid = meta["grid"]
    mode = meta["geometry_mode"]
    p = meta["geometry"]["parameters"]
    dx = grid["dx_m"]
    dy = grid["dy_m"]
    lx = grid["Lx_m"]
    ly = grid["Ly_m"]
    note = "exact"
    if mode == "full_rectangle":
        return f"rect({lx:.12g}, {ly:.12g})", note
    if mode == "nanostrip":
        length = float(p.get("length_px", 256.0)) * dx
        width = float(p.get("width_px", 96.0)) * dy
        if p.get("orientation") == "vertical":
            length, width = width, length
        return f"rect({length:.12g}, {width:.12g})", note
    if mode == "disk":
        diameter = max(float(p.get("diameter_px", 160.0)), 12.0 * meta["normalized_params_ref"]["rex"]) * dx
        return f"circle({diameter:.12g})", note
    if mode == "ellipse":
        major = float(p.get("major_axis_px", 180.0)) * dx
        minor = major / max(float(p.get("aspect_ratio", 2.0)), 1e-6)
        return f"ellipse({major:.12g}, {minor:.12g})", note
    if mode == "ring":
        outer = float(p.get("outer_diameter_px", 180.0)) * dx
        inner = outer * float(p.get("inner_outer_ratio", 0.5))
        return f"circle({outer:.12g}).sub(circle({inner:.12g}))", note
    if mode == "notched_strip":
        note = "notches_retained_in_metadata_simulated_as_strip"
        length = float(p.get("strip_length_px", 220.0)) * dx
        width = float(p.get("strip_width_px", 96.0)) * dy
        if p.get("orientation") == "vertical":
            length, width = width, length
        return f"rect({length:.12g}, {width:.12g})", note
    if mode == "antidot_or_holes":
        note = "holes_retained_in_metadata_simulated_as_full_rectangle"
        return f"rect({lx:.12g}, {ly:.12g})", note
    if mode == "smooth_polygon":
        note = "polygon_retained_in_metadata_simulated_as_circle"
        radius = 0.5 * float(p.get("radius_max_px", 110.0)) * dx
        return f"circle({2.0 * radius:.12g})", note
    return f"rect({lx:.12g}, {ly:.12g})", "fallback_rectangle"


def geometry_uses_mask(meta: Dict[str, Any]) -> bool:
    return str(meta.get("geometry_mode", "")) in MASK_RENDERED_GEOMETRIES


def rotate_to_local_px(x: float, y: float, angle_deg: float) -> Tuple[float, float]:
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    return x * c + y * s, -x * s + y * c


def rounded_rect_contains_px(x: float, y: float, length: float, width: float, rounding: float = 0.0) -> bool:
    hx, hy = 0.5 * length, 0.5 * width
    ax, ay = abs(x), abs(y)
    rounding = clamp(rounding, 0.0, max(0.0, min(hx, hy)))
    if rounding <= 0:
        return ax <= hx and ay <= hy
    core_x = hx - rounding
    core_y = hy - rounding
    if ax <= core_x and ay <= hy:
        return True
    if ax <= hx and ay <= core_y:
        return True
    return (ax - core_x) ** 2 + (ay - core_y) ** 2 <= rounding ** 2


def strip_mask_contains_px(mode: str, params: Dict[str, Any], x: float, y: float) -> bool:
    if mode == "nanostrip":
        length = float(params.get("length_px", 256.0))
        width = float(params.get("width_px", 96.0))
        rounding = float(params.get("edge_rounding_px", 0.0) or 0.0)
    else:
        length = float(params.get("strip_length_px", 220.0))
        width = float(params.get("strip_width_px", 96.0))
        rounding = 0.0
    lx, ly = rotate_to_local_px(x, y, float(params.get("orientation_deg", 0.0)))
    if not rounded_rect_contains_px(lx, ly, length, width, rounding):
        return False
    if mode != "notched_strip":
        return True
    depth = clamp(float(params.get("notch_depth_fraction_of_width", 0.25)) * width, 0.0, 0.5 * width)
    notch_width = max(1.0, float(params.get("notch_width_fraction_of_width", 0.25)) * width)
    count = max(1, int(params.get("notch_count", 1)))
    centers = [0.0] if count == 1 else [-0.25 * length, 0.25 * length]
    symmetric = str(params.get("notch_symmetry", "symmetric")) == "symmetric"
    for center in centers:
        if abs(lx - center) <= 0.5 * notch_width:
            if ly >= 0.5 * width - depth:
                return False
            if symmetric and ly <= -0.5 * width + depth:
                return False
    return True


def point_in_polygon_px(x: float, y: float, vertices: Sequence[Sequence[float]]) -> bool:
    inside = False
    n = len(vertices)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = float(vertices[i][0]), float(vertices[i][1])
        xj, yj = float(vertices[j][0]), float(vertices[j][1])
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / max(1e-12, yj - yi) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def geometry_mask_contains_px(meta: Dict[str, Any], x: float, y: float) -> bool:
    mode = str(meta["geometry_mode"])
    params = meta["geometry"]["parameters"]
    if mode in {"nanostrip", "notched_strip"}:
        return strip_mask_contains_px(mode, params, x, y)
    if mode == "antidot_or_holes":
        radius = float(params.get("hole_radius_px", 8.0))
        for cx, cy in params.get("hole_centers_px", []):
            if (x - float(cx)) ** 2 + (y - float(cy)) ** 2 <= radius ** 2:
                return False
        return True
    if mode == "smooth_polygon":
        vertices = params.get("polygon_vertices_px") or smooth_polygon_vertices_px(params, random.Random(int(meta["random_seed"]) + 31))
        return point_in_polygon_px(x, y, vertices)
    return True


def geometry_mask_rows(meta: Dict[str, Any]) -> List[List[int]]:
    """Render a MuMax ImageShape mask (dark inside, light outside)."""
    nx = int(meta["grid"]["Nx"])
    ny = int(meta["grid"]["Ny"])
    rows: List[List[int]] = []
    for j in range(ny):
        y = 0.5 * ny - (j + 0.5)
        row: List[int] = []
        for i in range(nx):
            x = (i + 0.5) - 0.5 * nx
            # MuMax ImageShape treats dark pixels as material. The legacy v4
            # generator wrote this polarity backwards, producing complement
            # geometries for every mask-rendered mode.
            row.append(0 if geometry_mask_contains_px(meta, x, y) else 255)
        rows.append(row)
    return rows


def write_png_grayscale(path: Path, rows: Sequence[Sequence[int]]) -> None:
    height = len(rows)
    width = len(rows[0]) if height else 0
    raw = b"".join(bytes([0]) + bytes(int(clamp(v, 0, 255)) for v in row) for row in rows)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack("!I", len(payload)) + kind + payload + struct.pack("!I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    ihdr = struct.pack("!IIBBBBB", width, height, 8, 0, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def write_geometry_assets(run_dir: Path, meta: Dict[str, Any]) -> None:
    if not geometry_uses_mask(meta):
        return
    write_png_grayscale(run_dir / "mask.png", geometry_mask_rows(meta))
    meta["geometry"]["mask_png"] = "mask.png"
    meta["geometry"]["mask_image_encoding"] = MASK_IMAGE_ENCODING


def render_control_regions(meta: Dict[str, Any]) -> str:
    lines = ["// Static tile regions for local drive approximations."]
    for r in meta["control_regions"]:
        lines.append(
            f"DefRegion({int(r['region_id'])}, rect({r['width_m']:.12g}, {r['height_m']:.12g}).transl({r['x_m']:.12g}, {r['y_m']:.12g}, 0))"
        )
    return "\n".join(lines)


def helicity_angle_rad(helicity: str, raw: Dict[str, Any], seed: int) -> float:
    d_sign = 1.0 if float(raw.get("D_ref_J_per_m2", 0.0) or 0.0) >= 0.0 else -1.0
    if helicity == "D_sign_matched_Neel":
        return 0.0 if d_sign >= 0.0 else math.pi
    if helicity in {"wrong_Neel", "wrong_chirality_Neel"}:
        return math.pi if d_sign >= 0.0 else 0.0
    if helicity == "Bloch_sign_matched":
        return 0.5 * math.pi if d_sign >= 0.0 else -0.5 * math.pi
    if helicity == "wrong_Bloch":
        return -0.5 * math.pi if d_sign >= 0.0 else 0.5 * math.pi
    if helicity == "Bloch":
        return 0.5 * math.pi
    if helicity == "Neel":
        return 0.0
    if helicity == "random":
        return random.Random(seed + 97).choice([0.0, math.pi, 0.5 * math.pi, -0.5 * math.pi])
    return 0.0


def render_initial_state(meta: Dict[str, Any]) -> str:
    init_family = meta["init_family"]
    params = meta["initial_state"]["parameters"]
    seed = int(meta["random_seed"])
    nx = int(meta["grid"]["Nx"])
    ny = int(meta["grid"]["Ny"])
    anis = meta["raw_physical_params_ref"]["anis_u"]
    ax, ay, az = anis
    if init_family == "uniform_near_uniform":
        if params.get("direction_type") == "random_unit_sphere":
            direction = random_unit_sphere(random.Random(seed + 11))
        else:
            direction = anis
        return f"""// Initial state: uniform_near_uniform
m = uniform({direction[0]:.12g}, {direction[1]:.12g}, {direction[2]:.12g})
randSeed({seed})
saveas(m, "m_initial")
"""
    if init_family == "smooth_random":
        bias = float(params.get("bias_strength", 0.2))
        xi = max(1.0, float(params.get("correlation_length_px", 16.0)))
        return f"""// Initial state: smooth_random deterministic Fourier-like texture.
m = uniform({ax:.12g}, {ay:.12g}, {az:.12g})
for i:=0; i<{nx}; i++ {{
    for j:=0; j<{ny}; j++ {{
        x := (i-{nx/2:.12g})/{xi:.12g}
        y := (j-{ny/2:.12g})/{xi:.12g}
        mx := {bias:.12g}*{ax:.12g} + sin(0.73*x + 1.31*y + {seed % 997})
        my := {bias:.12g}*{ay:.12g} + sin(1.17*x - 0.83*y + {seed % 577})
        mz := {bias:.12g}*{az:.12g} + sin(0.41*x + 0.97*y + {seed % 389})
        n := sqrt(mx*mx + my*my + mz*mz)
        m.SetCell(i, j, 0, vector(mx/n, my/n, mz/n))
    }}
}}
randSeed({seed})
saveas(m, "m_initial")
"""
    if init_family == "domain_wall":
        theta = math.radians(float(params.get("orientation_deg", 0.0)))
        c, s = math.cos(theta), math.sin(theta)
        wall_px = max(4.0, float(params.get("wall_width_px", 8.0)))
        wall_m = wall_px * meta["grid"]["dx_m"]
        wall_number = int(params.get("wall_number", 1))
        spacing_m = meta["grid"]["Lx_m"] / max(1, wall_number)
        return f"""// Initial state: domain_wall
m = uniform(0, 0, 1)
for i:=0; i<{nx}; i++ {{
    for j:=0; j<{ny}; j++ {{
        r := Index2Coord(i, j, 0)
        u := r.X()*{c:.12g} + r.Y()*{s:.12g}
        q := 0.0
        for k:=0; k<{wall_number}; k++ {{
            center := ({-0.5 * (wall_number - 1):.12g} + k)*{spacing_m:.12g}
            q += tanh((u-center)/{wall_m:.12g})
        }}
        mz := tanh(q)
        mxy := sqrt(max(0, 1 - mz*mz))
        mx := mxy*{-s:.12g}
        my := mxy*{c:.12g}
        m.SetCell(i, j, 0, vector(mx, my, mz))
    }}
}}
randSeed({seed})
saveas(m, "m_initial")
"""
    if init_family == "stripe_labyrinth":
        period_m = max(8.0, float(params.get("period_px", params.get("period_px_if_no_DMI", 48.0)))) * meta["grid"]["dx_m"]
        theta = math.radians(float(params.get("orientation_deg", 0.0)))
        phase = 2.0 * math.pi * float(params.get("phase_fraction", 0.0))
        c, s = math.cos(theta), math.sin(theta)
        return f"""// Initial state: stripe_labyrinth
m = uniform(0, 0, 1)
for i:=0; i<{nx}; i++ {{
    for j:=0; j<{ny}; j++ {{
        r := Index2Coord(i, j, 0)
        u := r.X()*{c:.12g} + r.Y()*{s:.12g}
        q := sin(2*pi*u/{period_m:.12g} + {phase:.12g})
        mz := tanh(2.5*q)
        mxy := sqrt(max(0, 1 - mz*mz))
        m.SetCell(i, j, 0, vector(mxy*{c:.12g}, mxy*{s:.12g}, mz))
    }}
}}
randSeed({seed})
saveas(m, "m_initial")
"""
    if init_family == "bubble_skyrmion":
        bg = 1.0 if params.get("background_polarity", "+") == "+" else -1.0
        core = 1.0 if params.get("core_polarity", "-") == "+" else -1.0
        bg_s = f"{bg:.1f}"
        core_s = f"{core:.1f}"
        wall_m = float(params.get("wall_width_parameter_px", 6.0)) * meta["grid"]["dx_m"]
        helicity_angle = helicity_angle_rad(str(params.get("helicity", "Neel")), meta["raw_physical_params_ref"], seed)
        object_lines = []
        for obj in params.get("objects", []):
            x_m = (float(obj["x_px"]) - 0.5 * nx) * meta["grid"]["dx_m"]
            y_m = (float(obj["y_px"]) - 0.5 * ny) * meta["grid"]["dy_m"]
            r_m = float(obj["radius_px"]) * meta["grid"]["dx_m"]
            aspect = max(1.0, float(obj.get("aspect_ratio", params.get("aspect_ratio", 1.0)) or 1.0))
            theta = math.radians(float(obj.get("orientation_deg", 0.0) or 0.0))
            c, s = math.cos(theta), math.sin(theta)
            deformation = max(0.0, float(obj.get(
                "deformation_amplitude_fraction_of_radius",
                params.get("deformation_amplitude_fraction_of_radius", 0.0),
            ) or 0.0))
            phase = float(obj.get("deformation_phase_rad", 0.0) or 0.0)
            object_lines.append(f"""
        dx = r.X()-({x_m:.12g})
        dy = r.Y()-({y_m:.12g})
        u = dx*{c:.12g} + dy*{s:.12g}
        v = -dx*{s:.12g} + dy*{c:.12g}
        phi = atan2(v*{aspect:.12g}, u)
        rb = max({0.35 * r_m:.12g}, {r_m:.12g}*(1.0 + {deformation:.12g}*cos(3.0*phi + {phase:.12g})))
        minor = rb/{aspect:.12g}
        rho = sqrt((u/rb)*(u/rb) + (v/minor)*(v/minor))
        weight = 0.5*(1.0-tanh((rho-1.0)*rb/{wall_m:.12g}))
        local_mz = ({bg_s}) + (({core_s})-({bg_s}))*weight
        local_mxy = sqrt(max(0.0, 1.0 - local_mz*local_mz))
        angle = phi + {theta + helicity_angle:.12g}
        mx = mx + local_mxy*cos(angle)*weight
        my = my + local_mxy*sin(angle)*weight
        mz = mz + local_mz - ({bg_s})
""")
        return f"""// Initial state: bubble_skyrmion
// Uses sampled helicity, aspect ratio, deformation, placement, and intentional-instability parameters.
m = uniform(0, 0, {bg_s})
for i:=0; i<{nx}; i++ {{
    for j:=0; j<{ny}; j++ {{
        r := Index2Coord(i, j, 0)
        mx := 0.0
        my := 0.0
        mz := {bg_s}
        dx := 0.0
        dy := 0.0
        u := 0.0
        v := 0.0
        phi := 0.0
        rb := 1.0
        minor := 1.0
        rho := 0.0
        weight := 0.0
        local_mz := {bg_s}
        local_mxy := 0.0
        angle := 0.0
{''.join(object_lines)}
        mz = max(-1.0, min(1.0, mz))
        n := sqrt(mx*mx + my*my + mz*mz)
        m.SetCell(i, j, 0, vector(mx/n, my/n, mz/n))
    }}
}}
randSeed({seed})
saveas(m, "m_initial")
"""
    if init_family == "vortex_antivortex":
        polarity = 1.0 if params.get("polarity", "+") == "+" else -1.0
        circulation = -1.0 if params.get("circulation") == "clockwise" else 1.0
        core_m = float(params.get("core_radius_px", 6.0)) * meta["grid"]["dx_m"]
        return f"""// Initial state: vortex_antivortex
m = uniform(1, 0, 0)
for i:=0; i<{nx}; i++ {{
    for j:=0; j<{ny}; j++ {{
        r := Index2Coord(i, j, 0)
        rr := sqrt(r.X()*r.X() + r.Y()*r.Y())
        phi := atan2(r.Y(), r.X())
        mz := {polarity:.12g}*exp(-(rr*rr)/({core_m:.12g}*{core_m:.12g}))
        mxy := sqrt(max(0, 1 - mz*mz))
        mx := {circulation:.12g}*(-sin(phi))*mxy
        my := {circulation:.12g}*( cos(phi))*mxy
        m.SetCell(i, j, 0, vector(mx, my, mz))
    }}
}}
randSeed({seed})
saveas(m, "m_initial")
"""
    if init_family == "spinwave_fmr_seed":
        wavelength_m = float(params.get("wavelength_px", 32.0)) * meta["grid"]["dx_m"]
        amp = math.radians(float(params.get("angular_amplitude_deg", 3.0)))
        theta = math.radians(float(params.get("propagation_direction_deg", 0.0)))
        c, s = math.cos(theta), math.sin(theta)
        return f"""// Initial state: spinwave_fmr_seed
m = uniform({ax:.12g}, {ay:.12g}, {az:.12g})
for i:=0; i<{nx}; i++ {{
    for j:=0; j<{ny}; j++ {{
        r := Index2Coord(i, j, 0)
        u := r.X()*{c:.12g} + r.Y()*{s:.12g}
        ph := 2*pi*u/{wavelength_m:.12g}
        mx := {ax:.12g} + {amp:.12g}*cos(ph)
        my := {ay:.12g} + {amp:.12g}*sin(ph)
        mz := {az:.12g}
        n := sqrt(mx*mx + my*my + mz*mz)
        m.SetCell(i, j, 0, vector(mx/n, my/n, mz/n))
    }}
}}
randSeed({seed})
saveas(m, "m_initial")
"""
    return f"""// Initial state fallback.
m = uniform({ax:.12g}, {ay:.12g}, {az:.12g})
randSeed({seed})
saveas(m, "m_initial")
"""


def render_material_assignment(raw: Dict[str, Any], inst: Dict[str, Any], cubic: Dict[str, Any]) -> str:
    ax, ay, az = raw["anis_u"]
    c1, c2, _c3 = cubic["cubic_axis_1"], cubic["cubic_axis_2"], cubic["cubic_axis_3"]
    dmi_type = raw.get("DMI_TYPE", "none")
    dind = inst["D_T_J_per_m2"] if dmi_type == "interfacial" else 0.0
    dbulk = inst["D_T_J_per_m2"] if dmi_type == "bulk" else 0.0
    return f"""Msat  = {inst['Ms_T_A_per_m']:.12g}
Aex   = {inst['A_T_J_per_m']:.12g}
Ku1   = {inst['Ku_T_J_per_m3']:.12g}
AnisU = vector({ax:.12g}, {ay:.12g}, {az:.12g})
Dind  = {dind:.12g}
Dbulk = {dbulk:.12g}
alpha = {inst['alpha_T']:.12g}
Kc1 = {inst['Kc1_T_J_per_m3']:.12g}
Kc2 = {inst['Kc2_T_J_per_m3']:.12g}
anisC1 = vector({c1[0]:.12g}, {c1[1]:.12g}, {c1[2]:.12g})
anisC2 = vector({c2[0]:.12g}, {c2[1]:.12g}, {c2[2]:.12g})
Temp = {inst['T_K']:.12g}
"""


def render_region_material_assignments(
    raw: Dict[str, Any],
    region_values: Sequence[Dict[str, Any]],
) -> List[str]:
    dmi_type = raw.get("DMI_TYPE", "none")
    lines: List[str] = []
    for value in region_values:
        rid = int(value["region_id"])
        inst = value["instantaneous_material"]
        dind = inst["D_T_J_per_m2"] if dmi_type == "interfacial" else 0.0
        dbulk = inst["D_T_J_per_m2"] if dmi_type == "bulk" else 0.0
        lines.extend([
            f"Msat.SetRegion({rid}, {inst['Ms_T_A_per_m']:.12g})",
            f"Aex.SetRegion({rid}, {inst['A_T_J_per_m']:.12g})",
            f"Ku1.SetRegion({rid}, {inst['Ku_T_J_per_m3']:.12g})",
            f"Dind.SetRegion({rid}, {dind:.12g})",
            f"Dbulk.SetRegion({rid}, {dbulk:.12g})",
            f"alpha.SetRegion({rid}, {inst['alpha_T']:.12g})",
            f"Kc1.SetRegion({rid}, {inst['Kc1_T_J_per_m3']:.12g})",
            f"Kc2.SetRegion({rid}, {inst['Kc2_T_J_per_m3']:.12g})",
            f"Temp.SetRegion({rid}, {inst['T_K']:.12g})",
        ])
    return lines


def render_region_vector_assignments(quantity: str, region_values: Sequence[Dict[str, Any]], key: str) -> List[str]:
    lines: List[str] = []
    for value in region_values:
        vx, vy, vz = value.get(key, [0.0, 0.0, 0.0])
        lines.append(
            f"{quantity}.SetRegion({int(value['region_id'])}, vector({vx:.12g}, {vy:.12g}, {vz:.12g}))"
        )
    return lines


def render_drive_assignment(segment: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    d = segment["drive"]
    bx, by, bz = d.get("B_ext_T", [0.0, 0.0, 0.0])
    region_values = d.get("region_values", [])
    use_region_profile = bool(region_values) and d.get("spatial_profile", "global_uniform") != "global_uniform"
    if use_region_profile:
        bx, by, bz = 0.0, 0.0, 0.0
    lines = [
        f"B_ext = vector({bx:.12g}, {by:.12g}, {bz:.12g})",
        "J = vector(0, 0, 0)",
        f"MaxDt = {float(cfg['max_dt_s']):.12g}",
    ]
    if d.get("has_zhang_li"):
        jx, jy, jz = d["J_vector_A_per_m2"]
        if use_region_profile:
            jx, jy, jz = 0.0, 0.0, 0.0
        lines += [
            "DisableZhangLiTorque = false",
            "DisableSlonczewskiTorque = true",
            f"Pol = {float(d.get('Pol', cfg['default_pol'])):.12g}",
            f"xi = {float(d.get('beta_ZL', 0.0)):.12g}",
            f"J = vector({jx:.12g}, {jy:.12g}, {jz:.12g})",
        ]
    elif d.get("has_slonczewski") or d.get("has_sot") or d.get("has_sot_like_slonczewski_proxy"):
        px, py, pz = d.get("polarization", [0.0, -1.0, 0.0])
        jx, jy, jz = d.get("J_vector_A_per_m2", [0.0, 0.0, d.get("J_A_per_m2", 0.0)])
        if use_region_profile:
            jx, jy, jz = 0.0, 0.0, 0.0
        if d.get("has_sot"):
            lines.append("// Effective spin-Hall SOT in the exact Lambda=1 Slonczewski torque basis.")
            lines.append(f"MaxDt = {float(d.get('sot_recommended_max_dt_s', cfg['max_dt_s'])):.12g}")
        elif d.get("has_sot_like_slonczewski_proxy"):
            lines.append("// SOT-like sampled drive rendered as Slonczewski fixed-layer proxy.")
        lines += [
            "DisableZhangLiTorque = true",
            "DisableSlonczewskiTorque = false",
            f"Pol = {float(d.get('Pol', cfg['default_pol'])):.12g}",
            f"Lambda = {float(d.get('Lambda', cfg['default_lambda_sl'])):.12g}",
            f"EpsilonPrime = {float(d.get('EpsilonPrime', cfg['default_epsilon_prime'])):.12g}",
            f"FixedLayer = vector({px:.12g}, {py:.12g}, {pz:.12g})",
            f"FixedLayerPosition = {cfg['fixed_layer_position']}",
            f"J = vector({jx:.12g}, {jy:.12g}, {jz:.12g})",
        ]
    else:
        lines += [
            "DisableZhangLiTorque = true",
            "DisableSlonczewskiTorque = true",
        ]
    if region_values:
        lines.append("// Region drive values clear stale overrides and approximate local profiles.")
        lines.extend(render_region_vector_assignments("B_ext", region_values, "B_ext_T"))
        lines.extend(render_region_vector_assignments("J", region_values, "J_vector_A_per_m2"))
    return "\n".join(lines)


def render_temperature_assignment(meta: Dict[str, Any], segment: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    regions = meta.get("control_regions", [])
    if not regions:
        return ""
    region_material_values = segment.get("region_material_values", [])
    if region_material_values:
        lines = ["// Region material and temperature values clear stale overrides."]
        lines.extend(render_region_material_assignments(meta["raw_physical_params_ref"], region_material_values))
    else:
        temp = float(segment["instantaneous_material"]["T_K"])
        lines = ["// Region temperatures clear stale overrides."]
        for region in regions:
            lines.append(f"Temp.SetRegion({int(region['region_id'])}, {temp:.12g})")
    return "\n".join(lines)


def render_absorbing_edge_assignment(meta: Dict[str, Any], segment: Dict[str, Any]) -> str:
    boundary = meta.get("boundary", {})
    if boundary.get("boundary_mode") != "absorbing_edge":
        return ""
    regions = meta.get("control_regions", [])
    if not regions:
        return ""
    width_px = float(boundary.get("absorbing_edge_width_px", 0.0) or 0.0)
    if width_px <= 0:
        return ""
    width_m = width_px * float(meta["grid"]["dx_m"])
    multiplier = float(boundary.get("alpha_edge_multiplier", 1.0) or 1.0)
    profile = str(boundary.get("sponge_profile", "cosine"))
    alpha_base = float(segment["instantaneous_material"]["alpha_T"])
    half_x = 0.5 * float(meta["grid"]["Lx_m"])
    half_y = 0.5 * float(meta["grid"]["Ly_m"])
    lines = ["// Absorbing edge: tile approximation via elevated alpha near boundaries."]
    for region in regions:
        dist_edge = min(
            half_x - abs(float(region["x_m"])),
            half_y - abs(float(region["y_m"])),
        )
        u = clamp(1.0 - dist_edge / max(width_m, 1e-30), 0.0, 1.0)
        if profile == "quadratic":
            scale = u * u
        elif profile == "linear":
            scale = u
        else:
            scale = 0.5 * (1.0 - math.cos(math.pi * u))
        alpha = alpha_base * (1.0 + (multiplier - 1.0) * scale)
        lines.append(f"alpha.SetRegion({int(region['region_id'])}, {alpha:.12g})")
    return "\n".join(lines)


def indent_mx3_block(text: str, prefix: str = "    ") -> str:
    return "\n".join((prefix + line) if line else "" for line in text.splitlines())


def final_relax_check_dt(meta: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    configured = cfg.get("final_relax_check_dt_s")
    if configured is not None and float(configured) > 0:
        check_dt = float(configured)
    else:
        check_dt = float(meta["time"]["save_dt_s"])
    check_dt = max(float(cfg["min_save_dt_s"]), check_dt)
    max_checks = max(1, int(cfg.get("final_relax_max_checks", 256)))
    timeout_s = float(meta["time"].get("condition_T_end_s", meta["time"]["T_end_s"]))
    if timeout_s / check_dt > max_checks:
        check_dt = timeout_s / max_checks
    return check_dt


def temperature_render_substep_count(meta: Dict[str, Any], segment: Dict[str, Any], cfg: Dict[str, Any]) -> int:
    if segment.get("segment_role") == "pre_relaxation":
        return 1
    schedule = meta.get("temperature_schedule", {})
    mode = str(schedule.get("T_schedule_mode", "isothermal"))
    render_cfg = cfg.get("temperature_schedule_render_substeps", {})
    if mode == "warmup_anneal_cycle":
        params = temperature_schedule_parameters(schedule)
        cycles = max(1, int(round(float(params.get("cycle_count", 1)))))
        base = int(render_cfg.get("warmup_anneal_cycle_per_cycle", 12)) * cycles
    else:
        base = int(render_cfg.get(mode, 1))
    values = []
    for u in temperature_probe_us(schedule):
        base_T, peak_delta_T = local_temperature_components_at_u(schedule, u)
        values.append(base_T + peak_delta_T)
    if values and max(values) - min(values) < 1e-9:
        base = 1
    if bool(meta["time"].get("autostop_enabled")):
        check_dt = final_relax_check_dt(meta, cfg)
        base = max(base, int(math.ceil(float(segment["duration_s"]) / max(check_dt, 1e-30))))
    return max(1, base)


def render_region_material_values_at_u(
    meta: Dict[str, Any],
    segment: Dict[str, Any],
    cfg: Dict[str, Any],
    inst: Dict[str, Any],
    u: float,
) -> List[Dict[str, Any]]:
    regions = meta.get("control_regions", [])
    if not regions:
        return []
    schedule = meta.get("temperature_schedule", {})
    mode = schedule.get("T_schedule_mode")
    grid = meta["grid"]
    raw = meta["raw_physical_params_ref"]
    temp_params = meta["temperature_dependent_params"]
    if mode != "local_temperature_spot":
        return [
            {
                "region_id": region["region_id"],
                "scale": 1.0,
                "T_K": inst["T_K"],
                "instantaneous_material": deepcopy(inst),
            }
            for region in regions
        ]

    threshold = float(cfg["local_profile_region_threshold"])
    cap = 0.92 * float(temp_params.get("Tc_K", 1e9))
    local_params = schedule.get("local_T_profile_parameters", temperature_schedule_parameters(schedule))
    base, peak_delta = local_temperature_components_at_u(schedule, u)
    values = []
    for region in regions:
        scale = profile_value(region, "gaussian_spot", local_params, grid)
        if abs(scale) < threshold:
            scale = 0.0
        temp = clamp(base + peak_delta * scale, 0.0, cap)
        region_inst = instantaneous_params(temp, raw, temp_params)
        attach_resolution_to_instantaneous_material(region_inst, raw, grid)
        values.append({
            "region_id": region["region_id"],
            "scale": scale,
            "T_K": temp,
            "instantaneous_material": region_inst,
        })
    return values


def temperature_render_subsegments(
    meta: Dict[str, Any],
    segment: Dict[str, Any],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    n = temperature_render_substep_count(meta, segment, cfg)
    start_s = float(segment["start_s"])
    end_s = float(segment["end_s"])
    raw = meta["raw_physical_params_ref"]
    temp_params = meta["temperature_dependent_params"]
    out: List[Dict[str, Any]] = []
    for i in range(n):
        sub_start = start_s + (end_s - start_s) * i / n
        sub_end = start_s + (end_s - start_s) * (i + 1) / n
        mid_time_s = 0.5 * (sub_start + sub_end)
        mid_u = schedule_u_for_time(meta, mid_time_s, segment)
        temp = temperature_schedule_value_at_u(meta["temperature_schedule"], mid_u)
        inst = instantaneous_params(temp, raw, temp_params)
        attach_resolution_to_instantaneous_material(inst, raw, meta["grid"])
        subseg = deepcopy(segment)
        subseg.update({
            "render_substep_index": i,
            "render_substep_count": n,
            "start_s": sub_start,
            "end_s": sub_end,
            "duration_s": sub_end - sub_start,
            "start_ns": sub_start * 1e9,
            "end_ns": sub_end * 1e9,
            "duration_ns": (sub_end - sub_start) * 1e9,
            "midpoint_u": mid_u,
            "global_midpoint_u": mid_time_s / float(meta["time"]["T_end_s"]),
            "T_K": temp,
            "T_midpoint_K": temp,
            "instantaneous_material": inst,
        })
        subseg["region_material_values"] = render_region_material_values_at_u(meta, segment, cfg, inst, mid_u)
        annotate_sot_drive_for_material(
            subseg["drive"],
            inst,
            subseg["region_material_values"],
            float(raw["thickness_m"]),
            cfg,
        )
        out.append(subseg)
    return out


def render_segment_schedule(meta: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    lines: List[str] = []
    final_relax_autostop = bool(meta["time"].get("autostop_enabled"))
    if final_relax_autostop:
        lines.append("// final_relax auto-stop: sparse run chunks stop when maxTorque drops below threshold.")
        lines.append("v4RelaxDone := false")
        lines.append(f"v4RelaxTorqueThreshold := {float(meta['time']['torque_threshold']):.12g}")
        lines.append(f"v4RelaxCheckDt := {final_relax_check_dt(meta, cfg):.12g}")
    for seg in meta["segments"]:
        seg_i = int(seg["segment_index"])
        header = f"// Segment {seg_i}: {seg['start_ns']:.6g} ns -> {seg['end_ns']:.6g} ns"
        render_subsegments = temperature_render_subsegments(meta, seg, cfg)
        if final_relax_autostop and seg.get("segment_role") != "pre_relaxation":
            block = [header]
            for subseg in render_subsegments:
                sub_i = int(subseg["render_substep_index"])
                sub_n = int(subseg["render_substep_count"])
                sub_block = []
                if sub_n > 1:
                    sub_block.append(
                        f"// Segment {seg_i} temperature render substep {sub_i + 1}/{sub_n}: "
                        f"{subseg['start_ns']:.6g} ns -> {subseg['end_ns']:.6g} ns"
                    )
                sub_block.extend([
                    render_material_assignment(meta["raw_physical_params_ref"], subseg["instantaneous_material"], meta["cubic_anisotropy"]),
                    render_temperature_assignment(meta, subseg, cfg),
                    render_absorbing_edge_assignment(meta, subseg),
                    render_drive_assignment(subseg, cfg),
                    f'run({subseg["duration_s"]:.12g})',
                    "if maxTorque < v4RelaxTorqueThreshold {",
                    "    v4RelaxDone = true",
                    "    TableSave()",
                    "}",
                ])
                block.append("if !v4RelaxDone {")
                block.append(indent_mx3_block("\n".join(sub_block)))
                block.append("}")
            block.append(f'saveas(m, "segment_{seg_i:03d}_end")')
            lines.append("if !v4RelaxDone {")
            lines.append(indent_mx3_block("\n".join(block)))
            lines.append("}")
        else:
            lines.append(header)
            for subseg in render_subsegments:
                sub_i = int(subseg["render_substep_index"])
                sub_n = int(subseg["render_substep_count"])
                if sub_n > 1:
                    lines.append(
                        f"// Segment {seg_i} temperature render substep {sub_i + 1}/{sub_n}: "
                        f"{subseg['start_ns']:.6g} ns -> {subseg['end_ns']:.6g} ns"
                    )
                lines.append(render_material_assignment(meta["raw_physical_params_ref"], subseg["instantaneous_material"], meta["cubic_anisotropy"]))
                lines.append(render_temperature_assignment(meta, subseg, cfg))
                lines.append(render_absorbing_edge_assignment(meta, subseg))
                lines.append(render_drive_assignment(subseg, cfg))
                lines.append(f'run({subseg["duration_s"]:.12g})')
            lines.append(f'saveas(m, "segment_{seg_i:03d}_end")')
    lines.append('saveas(m, "m_final")')
    lines.append("TableSave()")
    lines.append("flush()")
    return "\n".join(lines) + "\n"


def render_mx3(meta: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    grid = meta["grid"]
    boundary = meta["boundary"]
    shape_expr, geom_note = geometry_shape_expr(meta)
    raw = meta["raw_physical_params_ref"]
    first_inst = meta["segments"][0]["instantaneous_material"]
    meta["geometry"]["mx3_geometry_note"] = geom_note
    pbc_x = int(boundary["pbc_repeat_x"] if boundary["pbc_x"] else 0)
    pbc_y = int(boundary["pbc_repeat_y"] if boundary["pbc_y"] else 0)
    return f"""// Auto-generated by generate_universal_2d_micromagnetic_dynamics_v4.py
// trajectory_id: {meta['trajectory_id']}
// split: {meta['split']}
// thermal_noise_seed: {meta['thermal_noise_seed']}
// Segment contract: segments.json records condition segments. Non-isothermal schedules
// may render as internal temperature substeps that share the same metadata segment.

randSeed({int(meta['random_seed'])})
ThermSeed({int(meta['thermal_noise_seed'])})

SetGridSize({grid['Nx']}, {grid['Ny']}, {grid['Nz']})
SetCellSize({grid['dx_m']:.12g}, {grid['dy_m']:.12g}, {grid['dz_m']:.12g})
SetPBC({pbc_x}, {pbc_y}, 0)
EdgeSmooth = {int(cfg['edge_smooth'])}
setGeom({shape_expr})

OutputFormat = {cfg['output_format']}
SnapshotFormat = "png"

SetSolver({int(cfg['solver'])})
MaxDt = {float(cfg['max_dt_s']):.12g}
MaxErr = {float(cfg['max_err']):.12g}

{render_control_regions(meta)}

// Reference material before initial-state construction.
{render_material_assignment(raw, first_inst, meta['cubic_anisotropy'])}
FreeLayerThickness = {grid['dz_m']:.12g}

{render_initial_state(meta)}
save(regions)
saveas(geom, "geom")

TableAdd(E_total)
TableAdd(E_exch)
TableAdd(E_anis)
TableAdd(E_demag)
TableAdd(E_Zeeman)
TableAdd(ext_topologicalcharge)
ext_BubbleMz = -1
TableAdd(ext_bubblepos)
TableAdd(ext_bubbledist)
TableAdd(ext_bubblespeed)
TableAdd(m.Comp(0))
TableAdd(m.Comp(1))
TableAdd(m.Comp(2))
TableAdd(B_ext.Comp(0))
TableAdd(B_ext.Comp(1))
TableAdd(B_ext.Comp(2))
TableAdd(J.Comp(0))
TableAdd(J.Comp(1))
TableAdd(J.Comp(2))
TableAdd(maxTorque)
TableAdd(dt)
TableAdd(LastErr)
TableAdd(NEval)

TableAutoSave({meta['time']['table_dt_s']:.12g})
AutoSave(m, {meta['time']['save_dt_s']:.12g})
TableSave()

{render_segment_schedule(meta, cfg)}
"""


def drive_protocol_dict(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "trajectory_id": meta["trajectory_id"],
        "drive_type": meta["drive_type"],
        "drive_type_sampled": meta["drive"].get("drive_type_sampled"),
        "drive_type_rendered": meta["drive"].get("drive_type_rendered", meta["drive_type"]),
        "rendered_torque_model": meta["drive"].get("rendered_torque_model", "mixed_or_none"),
        "has_sot": meta["drive"].get("has_sot", False),
        "has_sot_like_slonczewski_proxy": meta["drive"].get("has_sot_like_slonczewski_proxy", False),
        "max_abs_sot_B_DL_T": meta["drive"].get("max_abs_sot_B_DL_T", 0.0),
        "max_abs_sot_B_FL_T": meta["drive"].get("max_abs_sot_B_FL_T", 0.0),
        "max_sot_effective_field_magnitude_T": meta["drive"].get("max_sot_effective_field_magnitude_T", 0.0),
        "min_sot_recommended_max_dt_s": meta["drive"].get("min_sot_recommended_max_dt_s", 0.0),
        "drive_segment_mode": meta["drive"]["drive_segment_mode"],
        "segment_count": meta["drive"]["segment_count"],
        "condition_segment_count": meta["drive"].get("condition_segment_count", meta["drive"]["segment_count"]),
        "pre_relaxation_enabled": meta["drive"].get("pre_relaxation_enabled", False),
        "pre_relaxation_duration_ns": meta["drive"].get("pre_relaxation_duration_ns", 0.0),
        "active_drive_mask": meta["drive"]["active_drive_mask"],
        "rendered_parameter_audit": meta.get("rendered_parameter_audit", {}),
        "control_grid": [
            max(r["ix"] for r in meta["control_regions"]) + 1,
            max(r["iy"] for r in meta["control_regions"]) + 1,
        ],
        "region_order": [r["region_id"] for r in meta["control_regions"]],
        "regions": meta["control_regions"],
        "segments": [
            {
                "segment_index": s["segment_index"],
                "segment_role": s.get("segment_role", "condition"),
                "start_s": s["start_s"],
                "end_s": s["end_s"],
                "drive": s["drive"],
            }
            for s in meta["segments"]
        ],
    }


def expected_frames(meta: Dict[str, Any]) -> int:
    return int(meta["time"]["saved_frame_count"])


def initial_result_summary(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mumax_finished": False,
        "event_type": "pending",
        "n_frames": None,
        "n_frames_expected": expected_frames(meta),
        "n_frames_expected_is_timeout_upper_bound": bool(meta["time"].get("autostop_enabled")),
        "q_start": None,
        "q_end": None,
        "skyrmion_count_start": None,
        "skyrmion_count_end": None,
        "max_abs_norm_error": None,
        "notes": (
            "Fill by OVF/table postprocessing after MuMax3 run finishes. "
            "For final_relax autostop runs, regenerate sample pairs from existing OVF frames."
            if bool(meta["time"].get("autostop_enabled"))
            else "Fill by OVF/table postprocessing after MuMax3 run finishes."
        ),
    }


def make_sample_manifest_rows(meta: Dict[str, Any], run_dir: Path, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if bool(meta["time"].get("autostop_enabled")):
        return []
    out_dir = run_dir / "run.out"
    rows: List[Dict[str, Any]] = []
    n_frames = expected_frames(meta)
    save_dt_s = float(meta["time"]["save_dt_s"])
    frame_times = [i * save_dt_s for i in range(n_frames)]
    for seg in meta["segments"]:
        seg_i = int(seg["segment_index"])
        frame_indices = [
            i for i, t in enumerate(frame_times)
            if seg["start_s"] - 1e-18 <= t <= seg["end_s"] + 1e-18
        ]
        frame_set = set(frame_indices)
        for delay in cfg["time_delay_steps"]:
            if delay <= 0:
                continue
            for fi in frame_indices:
                fj = fi + int(delay)
                if fj >= n_frames or fj not in frame_set:
                    continue
                ti = frame_times[fi]
                tj = frame_times[fj]
                temperature_i_K = temperature_schedule_value_at_u(
                    meta["temperature_schedule"],
                    schedule_u_for_time(meta, ti, seg),
                )
                temperature_j_K = temperature_schedule_value_at_u(
                    meta["temperature_schedule"],
                    schedule_u_for_time(meta, tj, seg),
                )
                inst_i = instantaneous_params(
                    temperature_i_K,
                    meta["raw_physical_params_ref"],
                    meta["temperature_dependent_params"],
                )
                attach_resolution_to_instantaneous_material(inst_i, meta["raw_physical_params_ref"], meta["grid"])
                rows.append({
                    "sample_id": f"{meta['trajectory_id']}__seg{seg_i:03d}__d{delay:03d}__i{fi:05d}",
                    "trajectory_id": meta["trajectory_id"],
                    "run_id": meta["trajectory_id"],
                    "thermal_noise_seed": meta["thermal_noise_seed"],
                    "split": meta["split"],
                    "dataset_profile": meta["dataset_profile"],
                    "material_family": meta["material_family"],
                    "geometry_mode": meta["geometry_mode"],
                    "boundary_mode": meta["boundary_mode"],
                    "pbc_x": meta["boundary"]["pbc_x"],
                    "pbc_y": meta["boundary"]["pbc_y"],
                    "pbc_repeat_x": meta["boundary"]["pbc_repeat_x"],
                    "pbc_repeat_y": meta["boundary"]["pbc_repeat_y"],
                    "geometry_interpretation": meta["boundary"]["geometry_interpretation"],
                    "init_family": meta["init_family"],
                    "TEMP_PARAM_MODE": meta["TEMP_PARAM_MODE"],
                    "T_schedule_mode": meta["T_schedule_mode"],
                    "CUBIC_ANISOTROPY_MODE": meta["CUBIC_ANISOTROPY_MODE"],
                    "MAGNETOELASTIC_MODE": meta["MAGNETOELASTIC_MODE"],
                    "DEFECT_MODE": meta["DEFECT_MODE"],
                    "TIME_REGIME": meta["TIME_REGIME"],
                    "segment_index": seg_i,
                    "segment_start_s": seg["start_s"],
                    "segment_end_s": seg["end_s"],
                    "frame_i": fi,
                    "frame_j": fj,
                    "t_i_s": ti,
                    "t_j_s": tj,
                    "delta_t_s": tj - ti,
                    "m_i_path": str((out_dir / f"m{fi:06d}.ovf").resolve()),
                    "m_j_path": str((out_dir / f"m{fj:06d}.ovf").resolve()),
                    "m_initial_path": str((out_dir / "m_initial.ovf").resolve()),
                    "m_final_path": str((out_dir / "m_final.ovf").resolve()),
                    "params_path": str((run_dir / "params.json").resolve()),
                    "segments_path": str((run_dir / "segments.json").resolve()),
                    "drive_protocol_path": str((run_dir / "drive_protocol.json").resolve()),
                    "temperature_K": temperature_i_K,
                    "temperature_i_K": temperature_i_K,
                    "temperature_j_K": temperature_j_K,
                    "temperature_segment_midpoint_K": seg["T_midpoint_K"],
                    "Ms_T_A_per_m": inst_i["Ms_T_A_per_m"],
                    "A_T_J_per_m": inst_i["A_T_J_per_m"],
                    "Ku_T_J_per_m3": inst_i["Ku_T_J_per_m3"],
                    "D_T_J_per_m2": inst_i["D_T_J_per_m2"],
                    "alpha_T": inst_i["alpha_T"],
                    "rex_T": inst_i["rex_T"],
                    "Delta_DW_cells_T": inst_i["Delta_DW_cells_T"],
                    "domain_wall_width_cells_T": inst_i["domain_wall_width_cells_T"],
                    "r_min_T": inst_i["r_min_T"],
                    "LD_cells_T": inst_i["LD_cells_T"],
                    "dataset_resolution_partition": meta["resolution_policy"]["dataset_resolution_partition"],
                    "min_length_resolution_class": meta["resolution_policy"]["min_length_resolution_class"],
                    "exchange_resolution_class": meta["resolution_policy"]["exchange_resolution_class"],
                    "pma_domain_wall_resolution_class": meta["resolution_policy"]["pma_domain_wall_resolution_class"],
                    "dmi_period_resolution_class": meta["resolution_policy"]["dmi_period_resolution_class"],
                    "drive_type": meta["drive_type"],
                    "drive_type_sampled": meta["drive"].get("drive_type_sampled"),
                    "drive_type_rendered": meta["drive"].get("drive_type_rendered", meta["drive_type"]),
                    "rendered_torque_model": seg["drive"].get("rendered_torque_model", "none"),
                    "has_sot": seg["drive"].get("has_sot", False),
                    "has_sot_like_slonczewski_proxy": seg["drive"].get("has_sot_like_slonczewski_proxy", False),
                    "theta_DL_eff": seg["drive"].get("theta_DL_eff", 0.0),
                    "r_FL_DL": seg["drive"].get("r_FL_DL", 0.0),
                    "sot_B_DL_T": seg["drive"].get("sot_B_DL_T", 0.0),
                    "sot_B_FL_T": seg["drive"].get("sot_B_FL_T", 0.0),
                    "drive_active_kind": seg["drive"]["active_kind"],
                    "validity_tags": meta["validity_tags"],
                    "audit_tags": meta["audit_tags"],
                    "valid_reason": "same_condition_segment_with_schedule_parameters",
                })
    return rows


def write_segment_schedule_csv(path: Path, meta: Dict[str, Any]) -> None:
    fieldnames = [
        "segment_index",
        "segment_role",
        "start_ns",
        "end_ns",
        "duration_ns",
        "condition_segment_index",
        "condition_start_ns",
        "condition_end_ns",
        "T_K",
        "T_midpoint_K",
        "T_schedule_mode",
        "T_function_id",
        "Ms_T_A_per_m",
        "A_T_J_per_m",
        "Ku_T_J_per_m3",
        "D_T_J_per_m2",
        "alpha_T",
        "drive_type",
        "drive_type_sampled",
        "drive_type_rendered",
        "drive_active_kind",
        "rendered_torque_model",
        "has_field",
        "has_zhang_li",
        "has_slonczewski",
        "has_sot",
        "has_sot_like_slonczewski_proxy",
        "theta_DL_eff",
        "r_FL_DL",
        "sot_B_DL_T",
        "sot_B_FL_T",
        "sot_recommended_max_dt_s",
        "B_ext_T",
        "J_vector_A_per_m2",
        "spatial_profile",
        "spatial_profile_sampled",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seg in meta["segments"]:
            inst = seg["instantaneous_material"]
            drive = seg["drive"]
            writer.writerow({
                "segment_index": seg["segment_index"],
                "segment_role": seg.get("segment_role", "condition"),
                "start_ns": seg["start_ns"],
                "end_ns": seg["end_ns"],
                "duration_ns": seg["duration_ns"],
                "condition_segment_index": seg.get("condition_segment_index"),
                "condition_start_ns": seg.get("condition_start_ns"),
                "condition_end_ns": seg.get("condition_end_ns"),
                "T_K": seg["T_K"],
                "T_midpoint_K": seg["T_midpoint_K"],
                "T_schedule_mode": meta["T_schedule_mode"],
                "T_function_id": meta["temperature_schedule"]["T_function_id"],
                "Ms_T_A_per_m": inst["Ms_T_A_per_m"],
                "A_T_J_per_m": inst["A_T_J_per_m"],
                "Ku_T_J_per_m3": inst["Ku_T_J_per_m3"],
                "D_T_J_per_m2": inst["D_T_J_per_m2"],
                "alpha_T": inst["alpha_T"],
                "drive_type": drive.get("drive_type", meta["drive_type"]),
                "drive_type_sampled": drive.get("drive_type_sampled"),
                "drive_type_rendered": drive.get("drive_type_rendered", drive.get("drive_type", meta["drive_type"])),
                "drive_active_kind": drive["active_kind"],
                "rendered_torque_model": drive.get("rendered_torque_model", "none"),
                "has_field": drive["has_field"],
                "has_zhang_li": drive["has_zhang_li"],
                "has_slonczewski": drive["has_slonczewski"],
                "has_sot": drive["has_sot"],
                "has_sot_like_slonczewski_proxy": drive.get("has_sot_like_slonczewski_proxy", False),
                "theta_DL_eff": drive.get("theta_DL_eff", 0.0),
                "r_FL_DL": drive.get("r_FL_DL", 0.0),
                "sot_B_DL_T": drive.get("sot_B_DL_T", 0.0),
                "sot_B_FL_T": drive.get("sot_B_FL_T", 0.0),
                "sot_recommended_max_dt_s": drive.get("sot_recommended_max_dt_s", 0.0),
                "B_ext_T": json.dumps(drive["B_ext_T"]),
                "J_vector_A_per_m2": json.dumps(drive["J_vector_A_per_m2"]),
                "spatial_profile": drive["spatial_profile"],
                "spatial_profile_sampled": drive.get("spatial_profile_sampled", drive["spatial_profile"]),
            })


def flatten_for_csv(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in d.items():
        if isinstance(value, (list, tuple, dict)):
            out[key] = json.dumps(value, sort_keys=True)
        else:
            out[key] = value
    return out


def manifest_summary(meta: Dict[str, Any], run_dir: Path, sample_count: int) -> Dict[str, Any]:
    return flatten_for_csv({
        "trajectory_id": meta["trajectory_id"],
        "run_id": meta["trajectory_id"],
        "random_seed": meta["random_seed"],
        "thermal_noise_seed": meta["thermal_noise_seed"],
        "split": meta["split"],
        "dataset_profile": meta["dataset_profile"],
        "material_family": meta["material_family"],
        "geometry_mode": meta["geometry_mode"],
        "default_boundary": meta["default_boundary"],
        "boundary_mode": meta["boundary_mode"],
        "pbc_x": meta["boundary"]["pbc_x"],
        "pbc_y": meta["boundary"]["pbc_y"],
        "pbc_repeat_x": meta["boundary"]["pbc_repeat_x"],
        "pbc_repeat_y": meta["boundary"]["pbc_repeat_y"],
        "geometry_interpretation": meta["boundary"]["geometry_interpretation"],
        "absorbing_edge_width_px": meta["boundary"].get("absorbing_edge_width_px", 0.0),
        "alpha_edge_multiplier": meta["boundary"].get("alpha_edge_multiplier", 1.0),
        "init_family": meta["init_family"],
        "drive_type": meta["drive_type"],
        "drive_type_sampled": meta["drive"].get("drive_type_sampled"),
        "drive_type_rendered": meta["drive"].get("drive_type_rendered", meta["drive_type"]),
        "rendered_torque_model": meta["drive"].get("rendered_torque_model", "mixed_or_none"),
        "has_sot_like_slonczewski_proxy": meta["drive"].get("has_sot_like_slonczewski_proxy", False),
        "rendered_parameter_audit_entry_count": meta.get("rendered_parameter_audit", {}).get("entry_count", 0),
        "TEMP_PARAM_MODE": meta["TEMP_PARAM_MODE"],
        "T_schedule_mode": meta["T_schedule_mode"],
        "CUBIC_ANISOTROPY_MODE": meta["CUBIC_ANISOTROPY_MODE"],
        "MAGNETOELASTIC_MODE": meta["MAGNETOELASTIC_MODE"],
        "DEFECT_MODE": meta["DEFECT_MODE"],
        "TIME_REGIME": meta["TIME_REGIME"],
        "segment_count": len(meta["segments"]),
        "T_end_ns": meta["time"]["T_end_ns"],
        "condition_T_end_ns": meta["time"].get("condition_T_end_ns", meta["time"]["T_end_ns"]),
        "pre_relaxation_enabled": meta["time"].get("pre_relaxation_enabled", False),
        "pre_relaxation_duration_ns": meta["time"].get("pre_relaxation_duration_ns", 0.0),
        "saved_frame_count": meta["time"]["saved_frame_count"],
        "saved_frame_count_is_timeout_upper_bound": bool(meta["time"].get("autostop_enabled")),
        "sample_manifest_requires_postprocess": bool(meta["time"].get("autostop_enabled")),
        "save_dt_ps": meta["time"]["save_dt_s"] * 1e12,
        "Ms_ref_A_per_m": meta["raw_physical_params_ref"]["Ms_ref_A_per_m"],
        "A_ref_J_per_m": meta["raw_physical_params_ref"]["A_ref_J_per_m"],
        "Ku_ref_J_per_m3": meta["raw_physical_params_ref"]["Ku_ref_J_per_m3"],
        "D_ref_J_per_m2": meta["raw_physical_params_ref"]["D_ref_J_per_m2"],
        "alpha_ref": meta["raw_physical_params_ref"]["alpha_ref"],
        "thickness_m": meta["raw_physical_params_ref"]["thickness_m"],
        "DMI_TYPE": meta["raw_physical_params_ref"].get("DMI_TYPE", "none"),
        "rex": meta["normalized_params_ref"]["rex"],
        "qK_ref": meta["normalized_params_ref"]["qK_ref"],
        "kappa_D_ref": meta["normalized_params_ref"]["kappa_D_ref"],
        "Delta_DW_cells_ref": meta["normalized_params_ref"]["Delta_DW_cells_ref"],
        "domain_wall_width_cells_ref": meta["normalized_params_ref"]["domain_wall_width_cells_ref"],
        "r_min_ref": meta["normalized_params_ref"]["r_min_ref"],
        "LD_cells_ref": meta["normalized_params_ref"]["LD_cells_ref"],
        "d_D_over_Dc_ref": meta["normalized_params_ref"]["d_D_over_Dc_ref"],
        "qC1_ref": meta["normalized_params_ref"]["qC1_ref"],
        "qC2_ref": meta["normalized_params_ref"]["qC2_ref"],
        "qME_ref": meta["normalized_params_ref"]["qME_ref"],
        "dataset_resolution_partition": meta["resolution_policy"]["dataset_resolution_partition"],
        "min_length_resolution_class": meta["resolution_policy"]["min_length_resolution_class"],
        "exchange_resolution_class": meta["resolution_policy"]["exchange_resolution_class"],
        "pma_domain_wall_resolution_class": meta["resolution_policy"]["pma_domain_wall_resolution_class"],
        "dmi_period_resolution_class": meta["resolution_policy"]["dmi_period_resolution_class"],
        "rex_min_over_segments": meta["resolution_policy"]["rex_min_over_segments"],
        "Delta_DW_cells_min_over_segments": meta["resolution_policy"]["Delta_DW_cells_min_over_segments"],
        "domain_wall_width_cells_min_over_segments": meta["resolution_policy"]["domain_wall_width_cells_min_over_segments"],
        "r_min_over_segments": meta["resolution_policy"]["r_min_over_segments"],
        "LD_cells_min_over_segments": meta["resolution_policy"]["LD_cells_min_over_segments"],
        "theta_t": meta["normalized_params_ref"]["theta_t"],
        "validity_tags": meta["validity_tags"],
        "audit_tags": meta["audit_tags"],
        "categorical_encodings": meta.get("categorical_encodings", {}),
        "proposal_attempt_count": meta["proposal_attempt_count"],
        "rejected_proposal_count": meta["rejected_proposal_count"],
        "rejection_reason_counts": meta["rejection_reason_counts"],
        "primary_rejection_reason_counts": meta["primary_rejection_reason_counts"],
        "n_samples_planned": sample_count,
        "run_dir": str(run_dir.resolve()),
        "mx3_path": str((run_dir / "run.mx3").resolve()),
        "params_path": str((run_dir / "params.json").resolve()),
        "segments_path": str((run_dir / "segments.json").resolve()),
        "sample_manifest_path": str((run_dir / "sample_manifest.jsonl").resolve()),
    })


def estimate_storage_gb(cfg: Dict[str, Any], spec_json: Dict[str, Any]) -> float:
    grid = grid_from_spec(spec_json)
    save_dt_s = max(float(cfg["min_save_dt_s"]), float(cfg["save_dt_s"]))
    pre_s = float(pre_relaxation_settings(spec_json)["duration_s"])
    time_root = spec_json["time_regime"]
    modes = time_root["modes"]
    weighted_frames = 0.0
    total_weight = 0.0
    for profile_item in spec_json["dataset_profile"]["sample"]["items"]:
        profile = str(profile_item["value"])
        profile_weight = float(profile_item.get("weight", 1.0))
        node = time_root["by_profile"].get(profile, time_root["global_target"])
        regime_items = node["items"]
        regime_weight_sum = sum(float(item.get("weight", 1.0)) for item in regime_items) or 1.0
        for item in regime_items:
            mode = str(item["value"])
            weight = profile_weight * float(item.get("weight", 1.0)) / regime_weight_sum
            key = "T_end_ns_timeout" if mode == "final_relax" else "T_end_ns"
            lo, hi = modes[mode][key]["range"]
            mean_t_s = 0.5 * (float(lo) + float(hi)) * 1e-9
            weighted_frames += weight * (math.floor((mean_t_s + pre_s + 1e-18) / save_dt_s) + 1)
            total_weight += weight
    expected_frames = weighted_frames / max(1e-12, total_weight)
    bytes_per_frame = grid["Nx"] * grid["Ny"] * grid["Nz"] * 3 * 4
    return bytes_per_frame * expected_frames * int(cfg["num_paths"]) / 1e9


def write_accepted_run(root: Path, meta: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[Path, List[Dict[str, Any]]]:
    run_dir = root / meta["trajectory_id"]
    run_dir.mkdir(parents=True, exist_ok=True)

    write_geometry_assets(run_dir, meta)
    (run_dir / "run.mx3").write_text(render_mx3(meta, cfg), encoding="utf-8")
    (run_dir / "params.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "trajectory_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "segments.json").write_text(json.dumps(meta["segments"], indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "drive_protocol.json").write_text(json.dumps(drive_protocol_dict(meta), indent=2, sort_keys=True), encoding="utf-8")
    write_segment_schedule_csv(run_dir / "segment_schedule.csv", meta)
    (run_dir / "result_summary.json").write_text(json.dumps(initial_result_summary(meta), indent=2, sort_keys=True), encoding="utf-8")

    sample_rows = make_sample_manifest_rows(meta, run_dir, cfg)
    with (run_dir / "sample_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in sample_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return run_dir, sample_rows


def generate_dataset(cfg: Dict[str, Any]) -> None:
    spec_path = resolve_input_path(cfg["spec_path"])
    spec_json = load_json(spec_path)
    run_batch_id = str(cfg.get("run_batch_id", "") or "").strip()
    if run_batch_id:
        cfg["run_batch_id"] = safe_identifier(run_batch_id)
    root = Path(cfg["output_dir"])
    if bool(cfg.get("fail_if_output_exists", False)) and root.exists() and any(root.iterdir()):
        raise FileExistsError(f"output directory already exists and is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    if cfg.get("copy_spec_json", True):
        shutil.copy2(spec_path, root / spec_path.name)

    splits = assign_split(int(cfg["num_paths"]), tuple(cfg["split_ratios"]), random.Random(int(cfg["random_seed"]) + 17))
    run_dirs: List[str] = []
    manifest_rows: List[Dict[str, Any]] = []
    all_sample_rows: List[Dict[str, Any]] = []

    quota = AcceptedTrajectoryQuota(spec_json, int(cfg["num_paths"]))
    if not bool(cfg.get("enforce_distribution_quota", True)):
        quota.enabled = False
    if quota.enabled:
        proposal_index = 0
        max_accepted_proposals = max(
            int(cfg["num_paths"]),
            int(cfg["num_paths"]) * int(cfg.get("quota_max_accepted_proposal_factor", cfg["max_attempt_factor"])),
        )
        while len(manifest_rows) < int(cfg["num_paths"]) and proposal_index < max_accepted_proposals:
            split = splits[len(manifest_rows)]
            meta = sample_trajectory(proposal_index, split, spec_json, cfg)
            proposal_index += 1
            quota_reason = quota.acceptance_rejection_reason(meta)
            if quota_reason is not None:
                quota.record_overflow(meta, quota_reason)
                continue
            quota.record_accept(meta)
            run_dir, sample_rows = write_accepted_run(root, meta, cfg)
            all_sample_rows.extend(sample_rows)
            run_dirs.append(str(run_dir.resolve()))
            manifest_rows.append(manifest_summary(meta, run_dir, len(sample_rows)))
        if len(manifest_rows) < int(cfg["num_paths"]):
            print(
                "WARNING: quota-controlled generation underfilled; "
                f"generated {len(manifest_rows)} of {int(cfg['num_paths'])} requested trajectories. "
                "See generation_audit.json for underfilled buckets."
            )
    else:
        for index in range(int(cfg["num_paths"])):
            meta = sample_trajectory(index, splits[index], spec_json, cfg)
            run_dir, sample_rows = write_accepted_run(root, meta, cfg)
            all_sample_rows.extend(sample_rows)
            run_dirs.append(str(run_dir.resolve()))
            manifest_rows.append(manifest_summary(meta, run_dir, len(sample_rows)))

    (root / "run_list.txt").write_text("\n".join(run_dirs) + "\n", encoding="utf-8")
    write_manifest_csv(root / "run_manifest.csv", manifest_rows)
    with (root / "run_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with (root / "sample_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in all_sample_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    (root / "config_used.json").write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    write_audit(root / "generation_audit.json", manifest_rows, all_sample_rows, quota.summary() if quota.enabled else None)

    print(f"Generated {len(run_dirs)} universal v4 segmented MuMax3 runs under: {root.resolve()}")
    print(f"Spec: {spec_path}")
    print(f"Run manifest: {root / 'run_manifest.csv'}")
    print(f"Global sample manifest: {root / 'sample_manifest.jsonl'}")
    print(f"Estimated raw m OVF storage: ~{estimate_storage_gb(cfg, spec_json):.2f} GB")


def write_manifest_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_audit(
    path: Path,
    manifest_rows: List[Dict[str, Any]],
    sample_rows: List[Dict[str, Any]],
    quota_summary: Dict[str, Any] | None = None,
) -> None:
    def counts(field: str, rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for row in rows:
            key = str(row.get(field, ""))
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items()))

    def sum_int(field: str, rows: Iterable[Dict[str, Any]]) -> int:
        total = 0
        for row in rows:
            try:
                total += int(row.get(field, 0) or 0)
            except (TypeError, ValueError):
                pass
        return total

    def aggregate_json_counts(field: str, rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for row in rows:
            value = row.get(field, {})
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    value = {}
            if not isinstance(value, dict):
                continue
            for key, count in value.items():
                try:
                    out[str(key)] = out.get(str(key), 0) + int(count)
                except (TypeError, ValueError):
                    continue
        return dict(sorted(out.items()))

    accepted_rejection_counts = aggregate_json_counts("rejection_reason_counts", manifest_rows)
    accepted_primary_rejection_counts = aggregate_json_counts("primary_rejection_reason_counts", manifest_rows)
    if quota_summary:
        add_counts(accepted_rejection_counts, quota_summary.get("overflow_rejection_reason_counts", {}))
        add_counts(accepted_primary_rejection_counts, quota_summary.get("overflow_primary_rejection_reason_counts", {}))
    attempted = sum_int("proposal_attempt_count", manifest_rows) + int((quota_summary or {}).get("overflow_proposal_attempt_count", 0) or 0)
    rejected = sum_int("rejected_proposal_count", manifest_rows) + int((quota_summary or {}).get("overflow_rejected_proposal_count", 0) or 0)
    audit = {
        "attempted_trajectory_count": attempted,
        "accepted_trajectory_count": len(manifest_rows),
        "rejected_trajectory_count": rejected,
        "accepted_pair_count": len(sample_rows),
        "rejection_reason_counts": accepted_rejection_counts,
        "primary_rejection_reason_counts": accepted_primary_rejection_counts,
        "counts": {
            "DATASET_PROFILE": counts("dataset_profile", manifest_rows),
            "MATERIAL_FAMILY": counts("material_family", manifest_rows),
            "GEOMETRY_MODE": counts("geometry_mode", manifest_rows),
            "BOUNDARY_MODE": counts("boundary_mode", manifest_rows),
            "GEOMETRY_INTERPRETATION": counts("geometry_interpretation", manifest_rows),
            "INIT_FAMILY": counts("init_family", manifest_rows),
            "DRIVE_TYPE": counts("drive_type", manifest_rows),
            "DRIVE_TYPE_SAMPLED": counts("drive_type_sampled", manifest_rows),
            "DRIVE_TYPE_RENDERED": counts("drive_type_rendered", manifest_rows),
            "RENDERED_TORQUE_MODEL": counts("rendered_torque_model", manifest_rows),
            "T_SCHEDULE_MODE": counts("T_schedule_mode", manifest_rows),
            "TEMP_PARAM_MODE": counts("TEMP_PARAM_MODE", manifest_rows),
            "CUBIC_ANISOTROPY_MODE": counts("CUBIC_ANISOTROPY_MODE", manifest_rows),
            "MAGNETOELASTIC_MODE": counts("MAGNETOELASTIC_MODE", manifest_rows),
            "DEFECT_MODE": counts("DEFECT_MODE", manifest_rows),
            "TIME_REGIME": counts("TIME_REGIME", manifest_rows),
            "DATASET_RESOLUTION_PARTITION": counts("dataset_resolution_partition", manifest_rows),
            "MIN_LENGTH_RESOLUTION_CLASS": counts("min_length_resolution_class", manifest_rows),
            "EXCHANGE_RESOLUTION_CLASS": counts("exchange_resolution_class", manifest_rows),
            "PMA_DOMAIN_WALL_RESOLUTION_CLASS": counts("pma_domain_wall_resolution_class", manifest_rows),
            "DMI_PERIOD_RESOLUTION_CLASS": counts("dmi_period_resolution_class", manifest_rows),
        },
        "note": "Generator samples until material and segment-level resolution constraints pass; per-run sampling_trace records accepted proposal details.",
    }
    if quota_summary:
        audit["accepted_trajectory_distribution_control"] = quota_summary
    path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")


def affected_reasons_for_metadata(metadata: Dict[str, Any]) -> List[str]:
    """Return deterministic reasons why a legacy trajectory must be quarantined."""
    reasons: List[str] = []
    geometry_mode = str(metadata.get("geometry_mode", ""))
    geometry = metadata.get("geometry", {})
    if not isinstance(geometry, dict):
        geometry = {}
    if (
        geometry_mode in MASK_RENDERED_GEOMETRIES
        and geometry.get("mask_image_encoding") != MASK_IMAGE_ENCODING
    ):
        reasons.append(AFFECTED_REASON_LEGACY_MASK)

    drive = metadata.get("drive", {})
    if not isinstance(drive, dict):
        drive = {}
    proxy = (
        metadata.get("drive_type") == SOT_PROXY_RENDERED_DRIVE_TYPE
        or metadata.get("drive_type_rendered") == SOT_PROXY_RENDERED_DRIVE_TYPE
        or metadata.get("rendered_torque_model") == SOT_PROXY_TORQUE_MODEL
        or drive.get("drive_type") == SOT_PROXY_RENDERED_DRIVE_TYPE
        or drive.get("drive_type_rendered") == SOT_PROXY_RENDERED_DRIVE_TYPE
        or drive.get("rendered_torque_model") == SOT_PROXY_TORQUE_MODEL
        or bool(drive.get("has_sot_like_slonczewski_proxy", False))
    )
    if not proxy:
        segments = metadata.get("segments", [])
        if isinstance(segments, list):
            proxy = any(
                isinstance(segment, dict)
                and isinstance(segment.get("drive"), dict)
                and (
                    segment["drive"].get("drive_type") == SOT_PROXY_RENDERED_DRIVE_TYPE
                    or segment["drive"].get("drive_type_rendered") == SOT_PROXY_RENDERED_DRIVE_TYPE
                    or segment["drive"].get("rendered_torque_model") == SOT_PROXY_TORQUE_MODEL
                    or bool(segment["drive"].get("has_sot_like_slonczewski_proxy", False))
                )
                for segment in segments
            )
    if proxy:
        reasons.append(AFFECTED_REASON_SOT_PROXY)
    return reasons


def plan_known_affected_quarantine(dataset_root: str | Path, quarantine_root: str | Path) -> Dict[str, Any]:
    """Audit a generated root and return a non-mutating quarantine plan."""
    dataset_root = Path(dataset_root).resolve()
    quarantine_root = Path(quarantine_root).resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"dataset root does not exist or is not a directory: {dataset_root}")
    if quarantine_root == dataset_root or dataset_root in quarantine_root.parents:
        raise ValueError("quarantine root must be outside the dataset root so recursive loaders cannot see it")

    records: List[Dict[str, Any]] = []
    invalid_metadata: List[Dict[str, str]] = []
    trajectory_count = 0
    split_counts: Dict[str, int] = {}
    affected_split_counts: Dict[str, int] = {}
    for run_dir in sorted(dataset_root.iterdir()):
        params_path = run_dir / "params.json"
        if not run_dir.is_dir() or not params_path.is_file():
            continue
        trajectory_count += 1
        try:
            metadata = load_json(params_path)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            invalid_metadata.append({"run_dir": str(run_dir), "error": str(exc)})
            continue
        split = str(metadata.get("split", "unspecified"))
        split_counts[split] = split_counts.get(split, 0) + 1
        reasons = affected_reasons_for_metadata(metadata)
        if not reasons:
            continue
        affected_split_counts[split] = affected_split_counts.get(split, 0) + 1
        records.append(
            {
                "trajectory_id": str(metadata.get("trajectory_id", run_dir.name)),
                "source": str(run_dir),
                "destination": str(quarantine_root / run_dir.name),
                "reasons": reasons,
                "geometry_mode": str(metadata.get("geometry_mode", "")),
                "drive_type": str(metadata.get("drive_type", "")),
            }
        )

    reason_counts: Dict[str, int] = {}
    for record in records:
        for reason in record["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    clean_split_counts = {
        split: count - affected_split_counts.get(split, 0)
        for split, count in split_counts.items()
    }
    return {
        "dataset_root": str(dataset_root),
        "quarantine_root": str(quarantine_root),
        "trajectory_count": trajectory_count,
        "affected_trajectory_count": len(records),
        "clean_trajectory_count": trajectory_count - len(records) - len(invalid_metadata),
        "reason_counts": dict(sorted(reason_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "affected_split_counts": dict(sorted(affected_split_counts.items())),
        "clean_split_counts": dict(sorted(clean_split_counts.items())),
        "multiple_reason_count": sum(len(record["reasons"]) > 1 for record in records),
        "invalid_metadata_count": len(invalid_metadata),
        "invalid_metadata": invalid_metadata,
        "records": records,
    }


def quarantine_known_affected_trajectories(
    dataset_root: str | Path,
    quarantine_root: str | Path,
    *,
    apply: bool = False,
    progress_every: int = 100,
) -> Dict[str, Any]:
    """Dry-run or resumably move affected trajectories to an external root."""
    report = plan_known_affected_quarantine(dataset_root, quarantine_root)
    report["applied"] = bool(apply)
    report["moved_trajectory_count"] = 0
    report["resumed"] = False
    if not apply:
        return report
    if report["invalid_metadata_count"]:
        raise RuntimeError(
            "refusing to quarantine with unreadable trajectory metadata: "
            f"{report['invalid_metadata_count']} invalid params.json files"
        )

    source_root = Path(report["dataset_root"])
    destination_root = Path(report["quarantine_root"])
    if not destination_root.parent.is_dir():
        raise NotADirectoryError(f"quarantine parent does not exist: {destination_root.parent}")
    if os.stat(source_root).st_dev != os.stat(destination_root.parent).st_dev:
        raise OSError("quarantine root must be on the same filesystem; refusing a potentially huge copy")
    manifest_path = destination_root / "quarantine_manifest.jsonl"
    summary_path = destination_root / "quarantine_summary.json"

    if destination_root.exists() and any(destination_root.iterdir()):
        if not manifest_path.is_file() or not summary_path.is_file():
            raise FileExistsError(
                "non-empty quarantine root has no resumable manifest/summary: "
                f"{destination_root}"
            )
        try:
            stored_summary = load_json(summary_path)
            stored_records = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot resume invalid quarantine journal: {destination_root}") from exc
        if (
            str(stored_summary.get("dataset_root")) != str(source_root)
            or str(stored_summary.get("quarantine_root")) != str(destination_root)
        ):
            raise RuntimeError("quarantine journal belongs to a different source or destination root")
        if not stored_records and int(stored_summary.get("affected_trajectory_count", 0)) != 0:
            raise RuntimeError("quarantine journal is missing its trajectory records")

        manifest_sources: set[str] = set()
        manifest_destinations: set[str] = set()
        for record in stored_records:
            source = Path(str(record.get("source", ""))).resolve()
            target = Path(str(record.get("destination", ""))).resolve()
            if source.parent != source_root or target.parent != destination_root or source.name != target.name:
                raise RuntimeError(f"unsafe path in quarantine journal: {record}")
            if str(source) in manifest_sources or str(target) in manifest_destinations:
                raise RuntimeError("quarantine journal contains duplicate source/destination paths")
            manifest_sources.add(str(source))
            manifest_destinations.add(str(target))
        new_sources = {str(record["source"]) for record in report["records"]} - manifest_sources
        if new_sources:
            raise RuntimeError(
                "dataset gained affected trajectories after quarantine began; use a new destination: "
                + ", ".join(sorted(new_sources)[:3])
            )
        expected_entries = {
            "quarantine_manifest.jsonl",
            "quarantine_summary.json",
            *(Path(path).name for path in manifest_destinations),
        }
        unexpected_entries = {path.name for path in destination_root.iterdir()} - expected_entries
        if unexpected_entries:
            raise RuntimeError(
                "quarantine root contains entries absent from its journal: "
                + ", ".join(sorted(unexpected_entries)[:3])
            )
        report = dict(stored_summary)
        report["records"] = stored_records
        report["invalid_metadata"] = []
        report["applied"] = True
        report["resumed"] = True
    else:
        destination_root.mkdir(parents=False, exist_ok=True)
        for record in report["records"]:
            target = Path(record["destination"])
            if target.exists():
                raise FileExistsError(f"quarantine target already exists: {target}")
        manifest_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in report["records"]),
            encoding="utf-8",
        )

    summary = {key: value for key, value in report.items() if key not in {"records", "invalid_metadata"}}
    summary["status"] = "moving"

    moved = 0
    pending: List[Dict[str, Any]] = []
    for record in report["records"]:
        source = Path(record["source"])
        target = Path(record["destination"])
        source_exists = source.is_dir()
        target_exists = target.is_dir()
        if source_exists == target_exists:
            state = "both exist" if source_exists else "neither exists"
            raise RuntimeError(f"cannot resume quarantine because {state}: {source} / {target}")
        if target_exists:
            moved += 1
        else:
            pending.append(record)

    summary["moved_trajectory_count"] = moved
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    for record in pending:
        Path(record["source"]).rename(Path(record["destination"]))
        moved += 1
        if progress_every > 0 and (moved % progress_every == 0 or moved == len(report["records"])):
            summary["moved_trajectory_count"] = moved
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            print(f"Quarantined {moved}/{len(report['records'])} trajectories", flush=True)
    report["moved_trajectory_count"] = moved
    summary["moved_trajectory_count"] = moved
    summary["status"] = "complete"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate segmented MuMax3 trajectories from the universal v4 JSON spec.")
    parser.add_argument(
        "--quarantine-dataset",
        default=None,
        help="Audit an existing generated dataset root for legacy inverted masks and SOT proxy trajectories.",
    )
    parser.add_argument(
        "--quarantine-out",
        default=None,
        help="Destination outside --quarantine-dataset; required when quarantining.",
    )
    parser.add_argument(
        "--apply-quarantine",
        action="store_true",
        help="Actually move affected trajectory directories; without this flag the quarantine command is a dry-run.",
    )
    parser.add_argument("--override", default=None, help="JSON/YAML file with CONFIG keys to override before CLI flags.")
    parser.add_argument("--spec", default=None, help="Path to Universal_2D_Micromagnetic_Dynamics_Dataset_Spec_v4 JSON.")
    parser.add_argument("--out", default=None, help="Output directory.")
    parser.add_argument("--num-paths", type=int, default=None, help="Number of trajectories.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--time-seed", action="store_true", help="Use a time-derived random seed.")
    parser.add_argument("--run-batch-id", default=None, help="Batch ID to prefix trajectory IDs; use 'auto' for timestamp plus random digits.")
    parser.add_argument("--append-run-id-to-out", action="store_true", help="Append the run batch ID to the output directory name.")
    parser.add_argument("--fail-if-output-exists", action="store_true", help="Abort if the output directory already exists and is not empty.")
    parser.add_argument("--control-grid", default=None, help="Local profile tile grid, e.g. 8x8 or 12,12.")
    parser.add_argument(
        "--disable-sot",
        action="store_true",
        help="Exclude effective spin-Hall SOT trajectories; strict SOT is enabled by default.",
    )
    parser.add_argument("--force-drive-type", default=None, help="Force a drive type, primarily for controlled validation runs.")
    parser.add_argument("--force-profile", default=None, help="Force a dataset profile, primarily for controlled validation runs.")
    parser.add_argument("--disable-quota", action="store_true", help="Disable accepted-distribution quotas for controlled validation runs.")
    args = parser.parse_args()

    if args.quarantine_dataset is not None:
        if args.quarantine_out is None:
            parser.error("--quarantine-out is required with --quarantine-dataset")
        report = quarantine_known_affected_trajectories(
            args.quarantine_dataset,
            args.quarantine_out,
            apply=args.apply_quarantine,
        )
        summary = {key: value for key, value in report.items() if key not in {"records", "invalid_metadata"}}
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.quarantine_out is not None or args.apply_quarantine:
        parser.error("--quarantine-out/--apply-quarantine require --quarantine-dataset")

    cfg = dict(CONFIG)
    if args.override is not None:
        cfg = merge_config(cfg, load_override_config(args.override))
    if args.spec is not None:
        cfg["spec_path"] = args.spec
    if args.out is not None:
        cfg["output_dir"] = args.out
    if args.num_paths is not None:
        cfg["num_paths"] = args.num_paths
    if args.seed is not None:
        cfg["random_seed"] = args.seed
    if args.time_seed:
        cfg["random_seed"] = make_time_seed()
        cfg["seed_source"] = "time_ns_xor_pid"
    if args.run_batch_id is not None:
        if args.run_batch_id.strip().lower() == "auto":
            cfg["run_batch_id"] = make_timestamp_random_id("univ2d")
        else:
            cfg["run_batch_id"] = safe_identifier(args.run_batch_id)
    if args.append_run_id_to_out:
        if not cfg.get("run_batch_id"):
            cfg["run_batch_id"] = make_timestamp_random_id("univ2d")
        out_path = Path(cfg["output_dir"])
        cfg["output_dir"] = str(out_path.with_name(f"{out_path.name}_{cfg['run_batch_id']}"))
    if args.fail_if_output_exists:
        cfg["fail_if_output_exists"] = True
    cfg["control_grid"] = parse_int_pair(args.control_grid, cfg["control_grid"], "control_grid")
    if args.disable_sot:
        cfg["enable_sot_trajectories"] = False
    if args.force_drive_type is not None:
        cfg["force_drive_type"] = args.force_drive_type
    if args.force_profile is not None:
        cfg["force_dataset_profile"] = args.force_profile
    if args.disable_quota:
        cfg["enforce_distribution_quota"] = False

    generate_dataset(cfg)


if __name__ == "__main__":
    main()
