from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from skyrmion_cfm.data.ovf import read_ovf


@dataclass(frozen=True)
class SpatialFieldSources:
    """Resolved paths to per-run spatial conditioning sources."""

    run_dir: Path
    geom_path: Path | None
    regions_path: Path | None
    current_protocol_path: Path | None
    current_control_path: Path | None
    drive_protocol_path: Path | None = None
    segments_path: Path | None = None

    @classmethod
    def from_run_dir(cls, run_dir: str | Path) -> "SpatialFieldSources":
        run_dir = Path(run_dir)
        out = run_dir / "run.out"

        def _maybe(*candidates: Path) -> Path | None:
            for c in candidates:
                if c.exists():
                    return c
            return None

        return cls(
            run_dir=run_dir,
            geom_path=_maybe(out / "geom.ovf", run_dir / "geom.ovf"),
            regions_path=_maybe(out / "regions000000.ovf", run_dir / "regions000000.ovf"),
            current_protocol_path=_maybe(run_dir / "current_protocol.json"),
            current_control_path=_maybe(run_dir / "current_control.csv"),
            drive_protocol_path=_maybe(run_dir / "drive_protocol.json"),
            segments_path=_maybe(run_dir / "segments.json"),
        )


def load_defect_field(
    sources: SpatialFieldSources,
    lattice_size: int,
) -> np.ndarray:
    """Return a (1, L, L) defect/geometry mask in [0, 1].

    Uses ``geom.ovf`` when available (1 = magnetic body, 0 = void/defect with
    smooth edge transition). Falls back to a uniform ones field — that is the
    correct prior when the run did not save geometry and the sample is a
    full-lattice square.
    """
    if sources.geom_path is not None:
        arr = read_ovf(sources.geom_path)
        if arr.ndim == 3 and arr.shape[0] == 1:
            field = arr[0]
        elif arr.ndim == 3 and arr.shape[0] == 3:
            field = np.linalg.norm(arr, axis=0)
        else:
            field = np.asarray(arr).reshape(arr.shape[-2], arr.shape[-1])
        field = np.clip(field.astype(np.float32), 0.0, 1.0)
        if field.shape != (lattice_size, lattice_size):
            field = _resize_nearest(field, lattice_size, lattice_size)
        return field[None]
    return np.ones((1, lattice_size, lattice_size), dtype=np.float32)


def _resize_nearest(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    src_h, src_w = arr.shape
    yi = (np.arange(h) * src_h // h).clip(0, src_h - 1)
    xi = (np.arange(w) * src_w // w).clip(0, src_w - 1)
    return arr[yi[:, None], xi[None, :]]


@lru_cache(maxsize=4096)
def _read_protocol(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=4096)
def _read_segments(path: Path) -> tuple[dict[str, Any], ...]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return tuple(x for x in data if isinstance(x, dict))
    return tuple()


@lru_cache(maxsize=4096)
def _read_control(path: Path) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Return (step_start_ns, per-step per-region currents).

    The CSV holds one row per control step with ``t_start_ns``, ``duration_ps``
    and ``J_region_XXX_A_per_m2``. We return the start-time list together with
    a tuple of per-region current tuples ordered by region_id.
    """
    rows: list[dict[str, str]] = []
    region_cols: list[str] | None = None
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return (0.0,), ((),)
        region_cols = sorted(
            [c for c in reader.fieldnames if c.startswith("J_region_") and c.endswith("_A_per_m2")],
            key=lambda c: int(c.split("_")[2]),
        )
        for row in reader:
            rows.append(row)
    if not rows or region_cols is None:
        return (0.0,), ((),)
    starts: list[float] = []
    per_region: list[tuple[float, ...]] = []
    for row in rows:
        starts.append(float(row.get("t_start_ns", "0.0") or 0.0))
        per_region.append(tuple(float(row.get(c, "0.0") or 0.0) for c in region_cols))
    return tuple(starts), tuple(per_region)


def control_grid_shape(protocol: dict[str, Any]) -> tuple[int, int]:
    grid = protocol.get("control_grid", [8, 8])
    if isinstance(grid, (list, tuple)) and len(grid) == 2:
        return int(grid[0]), int(grid[1])
    return 8, 8


def _v4_segment_at_time(protocol_or_segments: dict[str, Any] | tuple[dict[str, Any], ...], t_ns: float) -> dict[str, Any] | None:
    segments = (
        tuple(protocol_or_segments.get("segments", ()))
        if isinstance(protocol_or_segments, dict)
        else protocol_or_segments
    )
    if not segments:
        return None
    t_s = float(t_ns) * 1e-9
    eps = 1e-18
    for seg in segments:
        start = float(seg.get("start_s", 0.0))
        if abs(t_s - start) <= eps:
            return seg
    for seg in segments:
        start = float(seg.get("start_s", 0.0))
        end = float(seg.get("end_s", start))
        if start - eps <= t_s < end - eps:
            return seg
    for seg in segments:
        if abs(t_s - float(seg.get("end_s", 0.0))) <= eps:
            return seg
    return segments[-1]


def _v4_protocol(sources: SpatialFieldSources) -> dict[str, Any] | None:
    if sources.drive_protocol_path is None:
        return None
    return _read_protocol(sources.drive_protocol_path)


def _v4_full_segment(sources: SpatialFieldSources, t_ns: float) -> dict[str, Any] | None:
    if sources.segments_path is None:
        return None
    return _v4_segment_at_time(_read_segments(sources.segments_path), t_ns)


def _v4_drive_grid(
    protocol: dict[str, Any],
    t_ns: float,
    component: str,
) -> np.ndarray:
    gx, gy = control_grid_shape(protocol)
    grid = np.zeros((gy, gx), dtype=np.float32)
    seg = _v4_segment_at_time(protocol, t_ns)
    if seg is None:
        return grid
    drive = seg.get("drive", {})
    if not isinstance(drive, dict):
        return grid
    if component.startswith("J"):
        vec_key = "J_vector_A_per_m2"
        scalar_key = "J_A_per_m2"
        comp_idx = {"J_x": 0, "J_y": 1, "J_z": 2}[component]
    else:
        vec_key = "B_ext_T"
        scalar_key = ""
        comp_idx = {"B_x": 0, "B_y": 1, "B_z": 2}[component]
    region_values = drive.get("region_values") or []
    if region_values:
        rid_to_ix = _region_id_to_grid(protocol)
        for rv in region_values:
            rid = int(rv.get("region_id", 0))
            ixiy = rid_to_ix.get(rid)
            if ixiy is None:
                continue
            ix, iy = ixiy
            vector = rv.get(vec_key)
            value = float(vector[comp_idx]) if isinstance(vector, list) and len(vector) > comp_idx else 0.0
            if 0 <= ix < gx and 0 <= iy < gy:
                grid[iy, ix] = value
        return grid
    vector = drive.get(vec_key)
    if isinstance(vector, list) and len(vector) > comp_idx:
        value = float(vector[comp_idx])
    elif scalar_key:
        value = float(drive.get(scalar_key, 0.0))
    else:
        value = 0.0
    if value != 0.0:
        grid[:, :] = value
    return grid


def _v4_material_grid(
    sources: SpatialFieldSources,
    t_ns: float,
    key: str,
) -> np.ndarray:
    protocol = _v4_protocol(sources)
    if protocol is None:
        return np.zeros((8, 8), dtype=np.float32)
    gx, gy = control_grid_shape(protocol)
    grid = np.zeros((gy, gx), dtype=np.float32)
    seg = _v4_full_segment(sources, t_ns)
    if seg is None:
        return grid
    material_key_by_field = {
        "temperature_field": ("T_K", None),
        "msat_field": ("instantaneous_material", "Ms_T_A_per_m"),
        "aex_field": ("instantaneous_material", "A_T_J_per_m"),
        "ku1_field": ("instantaneous_material", "Ku_T_J_per_m3"),
        "dind_field": ("instantaneous_material", "D_T_J_per_m2"),
        "alpha_field": ("instantaneous_material", "alpha_T"),
    }
    top_key, nested_key = material_key_by_field[key]
    region_values = seg.get("region_material_values") or []
    if region_values:
        for rv in region_values:
            rid = int(rv.get("region_id", 0))
            ix = (rid - 1) % gx
            iy = (rid - 1) // gx
            if nested_key is None:
                value = float(rv.get(top_key, 0.0))
            else:
                value = float((rv.get(top_key) or {}).get(nested_key, 0.0))
            if 0 <= ix < gx and 0 <= iy < gy:
                grid[iy, ix] = value
        return grid
    if nested_key is None:
        value = float(seg.get("T_K", 0.0))
    else:
        value = float((seg.get(top_key) or {}).get(nested_key, 0.0))
    grid[:, :] = value
    return grid


def _region_id_to_grid(protocol: dict[str, Any]) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    gx, gy = control_grid_shape(protocol)
    regions = protocol.get("regions") or protocol.get("region_values") or []
    for region in regions:
        if not isinstance(region, dict) or region.get("region_id") is None:
            continue
        rid = int(region["region_id"])
        default_ix = (rid - 1) % gx
        default_iy = (rid - 1) // gx
        out[rid] = (
            int(region.get("ix", default_ix)),
            int(region.get("iy", default_iy)),
        )
    # Some generators omit the verbose region table because region ids are
    # already defined row-major by the control grid.  Filling missing ids also
    # makes partially specified tables safe without changing explicit entries.
    for rid in range(1, gx * gy + 1):
        out.setdefault(rid, ((rid - 1) % gx, (rid - 1) // gx))
    return out


def pulse_window_ns(sources: SpatialFieldSources) -> tuple[float, float] | None:
    """Return the absolute half-open SOT pulse window ``[start, end)`` in ns.

    Current-control datasets have used two compatible schemas over time.  Old
    files store a top-level ``pulse_duration_ns`` and implicitly start at zero;
    newer B/T files store absolute ``pulse_start_time_ns`` /
    ``pulse_end_time_ns`` and put the duration under ``shape``.  Normalising
    both schemas here prevents a delayed pulse from being mistaken for a pulse
    that starts at t=0.
    """
    if sources.current_protocol_path is None and sources.drive_protocol_path is not None:
        protocol = _read_protocol(sources.drive_protocol_path)
        active: list[tuple[float, float]] = []
        for seg in protocol.get("segments", []):
            drive = seg.get("drive", {})
            if isinstance(drive, dict) and drive.get("active"):
                start_ns = float(seg.get("start_s", 0.0)) * 1e9
                end_ns = float(seg.get("end_s", 0.0)) * 1e9
                if end_ns > start_ns:
                    active.append((start_ns, end_ns))
        if not active:
            return None
        return min(start for start, _ in active), max(end for _, end in active)
    if sources.current_protocol_path is None:
        return None

    protocol = _read_protocol(sources.current_protocol_path)
    explicit_start = protocol.get("pulse_start_time_ns")
    explicit_end = protocol.get("pulse_end_time_ns")
    shape = protocol.get("shape") or {}
    duration = float(
        protocol.get("pulse_duration_ns")
        or (shape.get("pulse_duration_ns") if isinstance(shape, dict) else 0.0)
        or 0.0
    )

    csv_starts: tuple[float, ...] = ()
    if sources.current_control_path is not None:
        csv_starts, _ = _read_control(sources.current_control_path)
    if explicit_start is not None:
        start_ns = float(explicit_start)
    elif csv_starts:
        start_ns = float(min(csv_starts))
    else:
        start_ns = 0.0

    if explicit_end is not None:
        end_ns = float(explicit_end)
    elif duration > 0.0:
        end_ns = start_ns + duration
    else:
        return None
    if end_ns <= start_ns:
        return None
    return start_ns, end_ns


def control_grid_at_time(
    sources: SpatialFieldSources,
    t_ns: float,
    phase: str = "control_on",
) -> np.ndarray:
    """Return the (gx, gy) 8x8 region-current map at simulation time ``t_ns``.

    During ``control_on`` (t < pulse_duration_ns) we read the most recent CSV
    row whose ``t_start_ns`` <= t. After the pulse ends, all regions are zero.
    If no control CSV exists, return an all-zero grid sized from the protocol.
    """
    if sources.current_protocol_path is None and sources.drive_protocol_path is not None:
        protocol = _read_protocol(sources.drive_protocol_path)
        return _v4_drive_grid(protocol, t_ns, "J_z")
    if sources.current_protocol_path is None:
        return np.zeros((8, 8), dtype=np.float32)
    protocol = _read_protocol(sources.current_protocol_path)
    gx, gy = control_grid_shape(protocol)
    grid = np.zeros((gy, gx), dtype=np.float32)

    pulse_window = pulse_window_ns(sources)
    if pulse_window is not None and phase != "always_on":
        pulse_start_ns, pulse_end_ns = pulse_window
        if t_ns < pulse_start_ns - 1e-9 or t_ns >= pulse_end_ns - 1e-9:
            return grid
    if sources.current_control_path is None:
        # No per-step CSV available — fall back to the protocol-level scalar
        # ``j_amp_a_per_m2`` distributed uniformly over its regions.
        shape = protocol.get("shape") or {}
        j_amp = float(
            protocol.get("j_amp_a_per_m2")
            or (shape.get("j_amp_a_per_m2") if isinstance(shape, dict) else 0.0)
            or 0.0
        )
        if j_amp == 0.0:
            return grid
        rid_to_ix = _region_id_to_grid(protocol)
        for rid, (ix, iy) in rid_to_ix.items():
            if 0 <= ix < gx and 0 <= iy < gy:
                grid[iy, ix] = j_amp
        return grid

    starts, per_region = _read_control(sources.current_control_path)
    if not starts:
        return grid
    chosen: int | None = None
    for idx, t_start in enumerate(starts):
        if t_start <= t_ns + 1e-9:
            chosen = idx
    if chosen is None:
        return grid
    rid_to_ix = _region_id_to_grid(protocol)
    region_values = per_region[chosen]
    for rid, (ix, iy) in rid_to_ix.items():
        if 0 <= ix < gx and 0 <= iy < gy and 0 < rid <= len(region_values):
            grid[iy, ix] = float(region_values[rid - 1])
    return grid


def pulse_duration_ns(sources: SpatialFieldSources) -> float:
    """Return the configured SOT pulse duration in ns, or 0 when unavailable."""
    window = pulse_window_ns(sources)
    return 0.0 if window is None else max(0.0, window[1] - window[0])


def control_change_times_ns(
    sources: SpatialFieldSources,
    *,
    include_zero: bool = True,
    include_pulse_end: bool = True,
) -> tuple[float, ...]:
    """Return absolute times where the drive condition may change.

    The returned boundaries are derived from ``current_control.csv`` row start
    times plus the protocol pulse end. They are intentionally independent of a
    particular frame spacing so callers can map them onto whatever saved-frame
    schedule a trajectory used.
    """
    if sources.current_protocol_path is None and sources.drive_protocol_path is not None:
        protocol = _read_protocol(sources.drive_protocol_path)
        times: set[float] = set()
        if include_zero:
            times.add(0.0)
        for seg in protocol.get("segments", []):
            times.add(float(seg.get("start_s", 0.0)) * 1e9)
            times.add(float(seg.get("end_s", 0.0)) * 1e9)
        return tuple(sorted(t for t in times if t >= 0.0))
    if sources.current_protocol_path is None:
        return (0.0,) if include_zero else ()
    pulse_window = pulse_window_ns(sources)
    times: set[float] = set()
    if include_zero:
        times.add(0.0)
    if sources.current_control_path is not None:
        starts, _ = _read_control(sources.current_control_path)
        for value in starts:
            t_ns = float(value)
            if t_ns < 0.0:
                continue
            if pulse_window is None or (
                pulse_window[0] - 1e-9 <= t_ns <= pulse_window[1] + 1e-9
            ):
                times.add(t_ns)
    if pulse_window is not None:
        times.add(pulse_window[0])
        if include_pulse_end:
            times.add(pulse_window[1])
    return tuple(sorted(times))


def control_segments_ns(
    sources: SpatialFieldSources,
    total_time_ns: float,
) -> tuple[tuple[float, float], ...]:
    """Return half-open time segments over which the control condition is fixed.

    Segment boundaries include every known drive change and the pulse-off
    boundary. If a run has no explicit control protocol, the whole trajectory is
    treated as one segment.
    """
    total = max(0.0, float(total_time_ns))
    if total <= 0.0:
        return ()
    if sources.current_protocol_path is None and sources.drive_protocol_path is not None:
        protocol = _read_protocol(sources.drive_protocol_path)
        segments: list[tuple[float, float]] = []
        for seg in protocol.get("segments", []):
            start = float(seg.get("start_s", 0.0)) * 1e9
            end = float(seg.get("end_s", 0.0)) * 1e9
            if end > start:
                segments.append((start, min(end, total)))
        return tuple((s, e) for s, e in segments if e > s)
    if sources.current_protocol_path is None:
        return ((0.0, total),)
    boundaries = {0.0, total}
    for t_ns in control_change_times_ns(sources):
        if 0.0 < t_ns < total:
            boundaries.add(float(t_ns))
    ordered = sorted(boundaries)
    segments: list[tuple[float, float]] = []
    for start, end in zip(ordered, ordered[1:], strict=False):
        if end > start:
            segments.append((float(start), float(end)))
    return tuple(segments)


def rasterize_j_field(
    grid: np.ndarray,
    lattice_size: int,
) -> np.ndarray:
    """Upsample an (gx, gy) region grid to a (1, L, L) piecewise-constant field.

    The control grid is stored row-major over (iy, ix) — that is, ``grid[iy, ix]``
    holds the value of region ``(ix, iy)``. Each region is mapped to a
    ``(L // gx) × (L // gy)`` patch on the lattice (matching the mumax3 region
    layout for static masks).
    """
    gy, gx = grid.shape
    if lattice_size % gx != 0 or lattice_size % gy != 0:
        # Fall back to nearest-neighbour upsampling for non-divisible grids.
        return _resize_nearest(grid, lattice_size, lattice_size)[None]
    block_y = lattice_size // gy
    block_x = lattice_size // gx
    return np.repeat(
        np.repeat(grid[None], block_y, axis=1),
        block_x,
        axis=2,
    ).astype(np.float32)


def build_j_field(
    sources: SpatialFieldSources,
    lattice_size: int,
    t_ns: float = 0.0,
    phase: str = "control_on",
) -> np.ndarray:
    """One-shot helper: control grid at ``t_ns`` → (1, L, L) raster."""
    grid = control_grid_at_time(sources, t_ns=t_ns, phase=phase)
    return rasterize_j_field(grid, lattice_size)


def spatial_condition_fields_at_time(
    sources: SpatialFieldSources,
    lattice_size: int,
    t_ns: float,
    fields: Sequence[str],
) -> dict[str, np.ndarray]:
    """Build configured spatial condition maps for old or v4 run layouts."""
    out: dict[str, np.ndarray] = {}
    protocol = _v4_protocol(sources)
    for field in fields:
        key = str(field)
        if key == "defect_field":
            out[key] = load_defect_field(sources, lattice_size)
        elif key == "j_field":
            out[key] = build_j_field(sources, lattice_size, t_ns=t_ns)
        elif key in {"j_x_field", "j_y_field", "j_z_field"}:
            if protocol is None:
                component = {"j_x_field": 0, "j_y_field": 1, "j_z_field": 2}[key]
                base = control_grid_at_time(sources, t_ns=t_ns) if component == 2 else np.zeros((8, 8), dtype=np.float32)
            else:
                base = _v4_drive_grid(protocol, t_ns, {"j_x_field": "J_x", "j_y_field": "J_y", "j_z_field": "J_z"}[key])
            out[key] = rasterize_j_field(base, lattice_size)
        elif key in {"b_x_field", "b_y_field", "b_z_field"}:
            if protocol is None:
                base = np.zeros((8, 8), dtype=np.float32)
            else:
                base = _v4_drive_grid(protocol, t_ns, {"b_x_field": "B_x", "b_y_field": "B_y", "b_z_field": "B_z"}[key])
            out[key] = rasterize_j_field(base, lattice_size)
        elif key in {"temperature_field", "msat_field", "aex_field", "ku1_field", "dind_field", "alpha_field"}:
            base = _v4_material_grid(sources, t_ns, key)
            out[key] = rasterize_j_field(base, lattice_size)
        else:
            out[key] = np.zeros((1, lattice_size, lattice_size), dtype=np.float32)
    return out
