from __future__ import annotations

import math
from collections.abc import Sequence as SequenceABC
from typing import Any, Sequence

import torch

from skyrmion_cfm.data.conditions import V4_CATEGORICAL_ORDERS, V4_T_THETA_KEYS


MU0 = 1.2566370614359173e-06
T_REF_K = 300.0


def is_v4_metadata(params: dict[str, Any]) -> bool:
    return "raw_physical_params_ref" in params and "segments" in params


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _temperature_params(schedule: dict[str, Any]) -> dict[str, Any]:
    return (
        schedule.get("T_schedule_parameters")
        or schedule.get("sampled_variables")
        or schedule.get("local_T_profile_parameters", {})
    )


def temperature_smoothstep(z: float) -> float:
    u = clamp(float(z), 0.0, 1.0)
    return 3.0 * u * u - 2.0 * u * u * u


def local_temperature_components_at_u(schedule: dict[str, Any], u: float) -> tuple[float, float]:
    mode = str(schedule.get("T_schedule_mode", "isothermal"))
    variables = _temperature_params(schedule)
    u = clamp(u, 0.0, 1.0)
    if mode != "local_temperature_spot":
        return temperature_schedule_value_at_u(schedule, u), 0.0
    base = float(variables.get("T_base_K", 300.0))
    delta = float(variables.get("delta_T_K", 0.0))
    u0 = float(variables.get("pulse_center_norm", 0.5))
    w = max(1e-6, float(variables.get("pulse_width_norm", 0.1)))
    pulse = math.exp(-((u - u0) ** 2) / (2.0 * w * w))
    return base, delta * pulse


def temperature_schedule_value_at_u(schedule: dict[str, Any], u: float) -> float:
    mode = str(schedule.get("T_schedule_mode", "isothermal"))
    variables = _temperature_params(schedule)
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


def segment_at_time(params: dict[str, Any], t_s: float) -> dict[str, Any]:
    segments = list(params.get("segments", ()))
    if not segments:
        return {}
    eps = 1e-18
    for seg in segments:
        start = float(seg.get("start_s", 0.0))
        end = float(seg.get("end_s", start))
        if abs(float(t_s) - start) <= eps:
            return seg
    for seg in segments:
        start = float(seg.get("start_s", 0.0))
        end = float(seg.get("end_s", start))
        if start - eps <= float(t_s) < end - eps:
            return seg
    for seg in segments:
        if abs(float(t_s) - float(seg.get("end_s", 0.0))) <= eps:
            return seg
    return segments[-1]


def schedule_u_for_time(params: dict[str, Any], t_s: float, segment: dict[str, Any] | None = None) -> float:
    if segment is not None and segment.get("segment_role") == "pre_relaxation":
        return float(segment.get("temperature_schedule_u", 0.0))
    time = params.get("time", {})
    condition_start_s = float(time.get("condition_start_s", 0.0))
    condition_total_s = float(time.get("condition_T_end_s", time.get("T_end_s", 1.0)))
    return clamp((float(t_s) - condition_start_s) / max(condition_total_s, 1e-30), 0.0, 1.0)


def temperature_at_time(params: dict[str, Any], t_s: float, segment: dict[str, Any] | None = None) -> float:
    schedule = params.get("temperature_schedule", {})
    return temperature_schedule_value_at_u(schedule, schedule_u_for_time(params, t_s, segment))


def temperature_reduced(T_K: float, temp_params: dict[str, Any]) -> float:
    if temp_params.get("TEMP_PARAM_MODE") == "temp_params_off":
        return 1.0
    tc = float(temp_params.get("Tc_K", 1.0e9))
    beta = float(temp_params.get("beta_Ms", 0.35))
    den = max(1e-9, 1.0 - T_REF_K / tc)
    num = max(1e-9, 1.0 - float(T_K) / tc)
    m_red = (num / den) ** beta
    return clamp(m_red, float(temp_params.get("m_red_min", 0.08)), float(temp_params.get("m_red_max", 1.3)))


def length_resolution_metrics(ms: float, aex: float, ku: float, anis_u: Sequence[float], dx_m: float) -> dict[str, Any]:
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


def instantaneous_params(T_K: float, raw: dict[str, Any], temp_params: dict[str, Any], grid: dict[str, Any]) -> dict[str, Any]:
    if temp_params.get("TEMP_PARAM_MODE") == "temp_params_off":
        inst = {
            "T_K": T_K,
            "m_reduced": 1.0,
            "Ms_T_A_per_m": raw["Ms_ref_A_per_m"],
            "A_T_J_per_m": raw["A_ref_J_per_m"],
            "Ku_T_J_per_m3": raw["Ku_ref_J_per_m3"],
            "D_T_J_per_m2": raw["D_ref_J_per_m2"],
            "alpha_T": raw["alpha_ref"],
            "Kc1_T_J_per_m3": raw.get("Kc1_ref_J_per_m3", 0.0),
            "Kc2_T_J_per_m3": raw.get("Kc2_ref_J_per_m3", 0.0),
            "B1_T_J_per_m3": raw.get("B1_ref_J_per_m3", 0.0),
            "B2_T_J_per_m3": raw.get("B2_ref_J_per_m3", 0.0),
        }
    else:
        mred = temperature_reduced(T_K, temp_params)
        alpha = raw["alpha_ref"] * (1.0 + float(temp_params.get("c_alpha", 0.0)) * (T_K - T_REF_K) / T_REF_K)
        alpha = clamp(alpha, float(temp_params.get("alpha_min", 0.001)), float(temp_params.get("alpha_max", 0.9)))
        inst = {
            "T_K": T_K,
            "m_reduced": mred,
            "Ms_T_A_per_m": raw["Ms_ref_A_per_m"] * mred,
            "A_T_J_per_m": raw["A_ref_J_per_m"] * (mred ** float(temp_params.get("p_A", 2.0))),
            "Ku_T_J_per_m3": raw["Ku_ref_J_per_m3"] * (mred ** float(temp_params.get("p_Ku", 3.0))),
            "D_T_J_per_m2": raw["D_ref_J_per_m2"] * (mred ** float(temp_params.get("p_D", 1.5))),
            "alpha_T": alpha,
            "Kc1_T_J_per_m3": raw.get("Kc1_ref_J_per_m3", 0.0) * (mred ** float(temp_params.get("p_Kc", 3.0))),
            "Kc2_T_J_per_m3": raw.get("Kc2_ref_J_per_m3", 0.0) * (mred ** float(temp_params.get("p_Kc2", 3.0))),
            "B1_T_J_per_m3": raw.get("B1_ref_J_per_m3", 0.0) * (mred ** float(temp_params.get("p_B1", 2.0))),
            "B2_T_J_per_m3": raw.get("B2_ref_J_per_m3", 0.0) * (mred ** float(temp_params.get("p_B2", 2.0))),
        }
    metrics = length_resolution_metrics(
        float(inst["Ms_T_A_per_m"]),
        float(inst["A_T_J_per_m"]),
        float(inst["Ku_T_J_per_m3"]),
        raw.get("anis_u", [0.0, 0.0, 1.0]),
        float(grid.get("dx_m", 1.0)),
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
        float(grid.get("dx_m", 1.0)),
    )
    return inst


def one_hot_from_metadata(params: dict[str, Any], key: str, value: str) -> torch.Tensor:
    # The model dimensions are defined by the canonical runtime vocabulary.
    # Persisted datasets may contain one-hot vectors written by an older
    # vocabulary (for example, 13- and 14-wide ``drive_type`` encodings in the
    # same dataset), which cannot be stacked into a batch.  Re-encode known
    # categories from their string value so every sample has the current,
    # stable width and ordering.
    order = V4_CATEGORICAL_ORDERS.get(key)
    if order is not None:
        return torch.tensor(
            [1.0 if str(value) == x else 0.0 for x in order],
            dtype=torch.float32,
        )
    enc = params.get("categorical_encodings", {}).get(key)
    if isinstance(enc, dict) and "one_hot" in enc:
        return torch.tensor(enc["one_hot"], dtype=torch.float32)
    return torch.empty(0, dtype=torch.float32)


def _vector3(node: Any, default: Sequence[float]) -> tuple[float, float, float]:
    values = list(node) if isinstance(node, SequenceABC) and not isinstance(node, (str, bytes)) else []
    out = list(default)
    for i, value in enumerate(values[:3]):
        out[i] = float(value)
    return float(out[0]), float(out[1]), float(out[2])


def categorical_tensors(params: dict[str, Any], segment: dict[str, Any]) -> dict[str, torch.Tensor]:
    drive = segment.get("drive", {}) if isinstance(segment, dict) else {}
    values = {
        "dataset_profile": params.get("dataset_profile", ""),
        "material_family": params.get("material_family", ""),
        "geometry_mode": params.get("geometry_mode", ""),
        "boundary_mode": params.get("boundary_mode", params.get("default_boundary", "")),
        "init_family": params.get("init_family", ""),
        "DMI_TYPE": params.get("raw_physical_params_ref", {}).get("DMI_TYPE", "none"),
        "TEMP_PARAM_MODE": params.get("TEMP_PARAM_MODE", ""),
        "T_schedule_mode": params.get("T_schedule_mode", ""),
        "CUBIC_ANISOTROPY_MODE": params.get("CUBIC_ANISOTROPY_MODE", ""),
        "MAGNETOELASTIC_MODE": params.get("MAGNETOELASTIC_MODE", ""),
        "DEFECT_MODE": params.get("DEFECT_MODE", ""),
        "TIME_REGIME": params.get("TIME_REGIME", ""),
        "drive_type": params.get("drive_type", ""),
        "drive_type_rendered": drive.get("drive_type_rendered", params.get("drive", {}).get("drive_type_rendered", params.get("drive_type", ""))),
        "rendered_torque_model": drive.get("rendered_torque_model", "none"),
        "drive_active_kind": drive.get("active_kind", "none"),
        "segment_role": segment.get("segment_role", "condition") if isinstance(segment, dict) else "condition",
    }
    return {f"{key}_onehot": one_hot_from_metadata(params, key, str(value)) for key, value in values.items()}


def theta_t_row(params: dict[str, Any]) -> dict[str, float]:
    schedule = params.get("temperature_schedule", {})
    theta = list(schedule.get("theta_T", []))
    order = list(schedule.get("theta_T_order", V4_T_THETA_KEYS))
    values = {str(k): float(v) for k, v in zip(order, theta, strict=False)}
    return {f"theta_T_{key}": float(values.get(key, 0.0)) for key in V4_T_THETA_KEYS}


def pair_condition_row(params: dict[str, Any], t_i_s: float, t_j_s: float) -> dict[str, float]:
    segment = segment_at_time(params, t_i_s)
    raw = params.get("raw_physical_params_ref", {})
    normalized = params.get("normalized_params_ref", {})
    grid = params.get("grid", {})
    temp_params = params.get("temperature_dependent_params", {})
    drive = segment.get("drive", {}) if isinstance(segment, dict) else {}
    b = drive.get("B_ext_T", [0.0, 0.0, 0.0])
    j_vec = drive.get("J_vector_A_per_m2", [0.0, 0.0, 0.0])
    charge_j_vec = drive.get("charge_current_vector_A_per_m2", [0.0, 0.0, 0.0])
    polarization = drive.get("polarization", [0.0, -1.0, 0.0])
    anis_u = _vector3(raw.get("anis_u", [0.0, 0.0, 1.0]), (0.0, 0.0, 1.0))
    cubic = params.get("cubic_anisotropy", {})
    cubic_axis_1 = _vector3(cubic.get("cubic_axis_1", [1.0, 0.0, 0.0]), (1.0, 0.0, 0.0))
    cubic_axis_2 = _vector3(cubic.get("cubic_axis_2", [0.0, 1.0, 0.0]), (0.0, 1.0, 0.0))
    cubic_axis_3 = _vector3(cubic.get("cubic_axis_3", [0.0, 0.0, 1.0]), (0.0, 0.0, 1.0))
    segment_start = float(segment.get("start_s", t_i_s) or 0.0)
    segment_end = float(segment.get("end_s", t_j_s) or 0.0)
    duration = max(0.0, segment_end - segment_start)
    anchor_offset_s = max(0.0, float(t_i_s) - segment_start)
    target_offset_s = max(0.0, float(t_j_s) - segment_start)
    anchor_offset_norm = 0.0 if duration <= 0 else clamp(anchor_offset_s / duration, 0.0, 1.0)
    target_offset_norm = 0.0 if duration <= 0 else clamp(target_offset_s / duration, 0.0, 1.0)
    # Training conditions are intentionally segment-fixed: for every pair in a
    # segment, material scalars use the segment-start value. Continuous
    # temperature/material evolution is represented separately by mode one-hot
    # plus function parameters (theta_T and temp_param_*).
    T_start = temperature_at_time(params, segment_start, segment)
    inst = (
        instantaneous_params(T_start, raw, temp_params, grid)
        if raw and temp_params
        else segment.get("instantaneous_material", {})
    )
    active = bool(drive.get("active", False))
    row = {
        "dt_s": max(0.0, float(t_j_s) - float(t_i_s)),
        "t_end_s": max(0.0, float(t_j_s) - float(t_i_s)),
        "anchor_offset_s": anchor_offset_s,
        "anchor_offset_norm": anchor_offset_norm,
        "target_offset_norm": target_offset_norm,
        "temp_k": T_start,
        "temperature_start_k": T_start,
        "temperature_segment_midpoint_k": float(segment.get("T_midpoint_K", segment.get("T_K", T_start)) or T_start),
        "temperature_over_tc": T_start / max(float(temp_params.get("Tc_K", 1.0e9)), 1e-30),
        "b_x_t": float(b[0]),
        "b_y_t": float(b[1]),
        "b_z_t": float(b[2]),
        "b_ext_x_t": float(b[0]),
        "b_ext_y_t": float(b[1]),
        "b_ext_z_t": float(b[2]),
        "current_a_m2": float(drive.get("J_A_per_m2", j_vec[2] if len(j_vec) >= 3 else 0.0)),
        "j_vector_x_a_per_m2": float(j_vec[0]),
        "j_vector_y_a_per_m2": float(j_vec[1]),
        "j_vector_z_a_per_m2": float(j_vec[2]),
        "j_abs_a_per_m2": math.sqrt(sum(float(x) * float(x) for x in j_vec)),
        "charge_current_x_a_per_m2": float(charge_j_vec[0]),
        "charge_current_y_a_per_m2": float(charge_j_vec[1]),
        "charge_current_z_a_per_m2": float(charge_j_vec[2]),
        "alpha": float(inst.get("alpha_T", raw.get("alpha_ref", 0.1))),
        "aex_j_per_m": float(inst.get("A_T_J_per_m", raw.get("A_ref_J_per_m", 1.0e-11))),
        "dind_j_per_m2": float(inst.get("D_T_J_per_m2", raw.get("D_ref_J_per_m2", 0.0))),
        "ku1_j_per_m3": float(inst.get("Ku_T_J_per_m3", raw.get("Ku_ref_J_per_m3", 0.0))),
        "msat_a_per_m": float(inst.get("Ms_T_A_per_m", raw.get("Ms_ref_A_per_m", 580000.0))),
        "dx_m": float(grid.get("dx_m", 1.0)),
        "dy_m": float(grid.get("dy_m", 1.0)),
        "dz_m": float(grid.get("dz_m", raw.get("thickness_m", 1.0))),
        "pol_eff": float(drive.get("Pol", 0.0)),
        "epsilon_prime": float(drive.get("EpsilonPrime", 0.0)),
        "fixed_layer_x": float(polarization[0]),
        "fixed_layer_y": float(polarization[1]),
        "fixed_layer_z": float(polarization[2]),
        "anis_u_x": anis_u[0],
        "anis_u_y": anis_u[1],
        "anis_u_z": anis_u[2],
        "cubic_axis_1_x": cubic_axis_1[0],
        "cubic_axis_1_y": cubic_axis_1[1],
        "cubic_axis_1_z": cubic_axis_1[2],
        "cubic_axis_2_x": cubic_axis_2[0],
        "cubic_axis_2_y": cubic_axis_2[1],
        "cubic_axis_2_z": cubic_axis_2[2],
        "cubic_axis_3_x": cubic_axis_3[0],
        "cubic_axis_3_y": cubic_axis_3[1],
        "cubic_axis_3_z": cubic_axis_3[2],
        "polarization_x": float(polarization[0]),
        "polarization_y": float(polarization[1]),
        "polarization_z": float(polarization[2]),
        "lambda_sl": float(drive.get("Lambda", 0.0)),
        "beta_zl": float(drive.get("beta_ZL", 0.0)),
        "theta_dl_eff": float(drive.get("theta_DL_eff", 0.0)),
        "r_fl_dl": float(drive.get("r_FL_DL", 0.0)),
        "sot_b_dl_t": float(drive.get("sot_B_DL_T", 0.0)),
        "sot_b_fl_t": float(drive.get("sot_B_FL_T", 0.0)),
        "sot_explicit_b_dl_t": float(drive.get("sot_explicit_B_DL_T", 0.0)),
        "sot_explicit_b_fl_t": float(drive.get("sot_explicit_B_FL_sigma_cross_m_T", 0.0)),
        "m_reduced": float(inst.get("m_reduced", 1.0)),
        "ms_t_a_per_m": float(inst.get("Ms_T_A_per_m", raw.get("Ms_ref_A_per_m", 0.0))),
        "a_t_j_per_m": float(inst.get("A_T_J_per_m", raw.get("A_ref_J_per_m", 0.0))),
        "ku_t_j_per_m3": float(inst.get("Ku_T_J_per_m3", raw.get("Ku_ref_J_per_m3", 0.0))),
        "d_t_j_per_m2": float(inst.get("D_T_J_per_m2", raw.get("D_ref_J_per_m2", 0.0))),
        "alpha_t": float(inst.get("alpha_T", raw.get("alpha_ref", 0.0))),
        "kc1_t_j_per_m3": float(inst.get("Kc1_T_J_per_m3", 0.0) or 0.0),
        "kc2_t_j_per_m3": float(inst.get("Kc2_T_J_per_m3", 0.0) or 0.0),
        "b1_t_j_per_m3": float(inst.get("B1_T_J_per_m3", 0.0) or 0.0),
        "b2_t_j_per_m3": float(inst.get("B2_T_J_per_m3", 0.0) or 0.0),
        "ms_ref_a_per_m": float(raw.get("Ms_ref_A_per_m", 0.0)),
        "a_ref_j_per_m": float(raw.get("A_ref_J_per_m", 0.0)),
        "ku_ref_j_per_m3": float(raw.get("Ku_ref_J_per_m3", 0.0)),
        "d_ref_j_per_m2": float(raw.get("D_ref_J_per_m2", 0.0)),
        "alpha_ref": float(raw.get("alpha_ref", 0.0)),
        "thickness_m": float(raw.get("thickness_m", grid.get("dz_m", 0.0)) or 0.0),
        "tc_k": float(temp_params.get("Tc_K", 0.0)),
        "theta_t": float(normalized.get("theta_t", 0.0) or 0.0),
        "qk_ref": float(normalized.get("qK_ref", 0.0) or 0.0),
        "kappa_d_ref": float(normalized.get("kappa_D_ref", 0.0) or 0.0),
        "d_d_over_dc_ref": float(normalized.get("d_D_over_Dc_ref", 0.0) or 0.0),
        "ld_cells_ref": float(normalized.get("LD_cells_ref", 0.0) or 0.0),
        "rex": float(normalized.get("rex", 0.0) or 0.0),
        "r_min_ref": float(normalized.get("r_min_ref", 0.0) or 0.0),
        "delta_dw_cells_ref": float(normalized.get("Delta_DW_cells_ref", 0.0) or 0.0),
        "domain_wall_width_cells_ref": float(normalized.get("domain_wall_width_cells_ref", 0.0) or 0.0),
        "rex_t": float(inst.get("rex_T", 0.0) or 0.0),
        "r_min_t": float(inst.get("r_min_T", 0.0) or 0.0),
        "ld_cells_t": float(inst.get("LD_cells_T", 0.0) or 0.0),
        "delta_dw_cells_t": float(inst.get("Delta_DW_cells_T", 0.0) or 0.0),
        "domain_wall_width_cells_t": float(inst.get("domain_wall_width_cells_T", 0.0) or 0.0),
        "pre_relaxation_flag": 1.0 if segment.get("segment_role") == "pre_relaxation" else 0.0,
        "drive_active_flag": 1.0 if active else 0.0,
        "has_field": 1.0 if drive.get("has_field") else 0.0,
        "has_zhang_li": 1.0 if drive.get("has_zhang_li") else 0.0,
        "has_slonczewski": 1.0 if drive.get("has_slonczewski") else 0.0,
        "has_sot": 1.0 if drive.get("has_sot") else 0.0,
        "has_sot_like_slonczewski_proxy": 1.0 if drive.get("has_sot_like_slonczewski_proxy") else 0.0,
        "temp_param_beta_ms": float(temp_params.get("beta_Ms", 0.0)),
        "temp_param_m_red_min": float(temp_params.get("m_red_min", 0.0)),
        "temp_param_m_red_max": float(temp_params.get("m_red_max", 0.0)),
        "temp_param_p_a": float(temp_params.get("p_A", 0.0)),
        "temp_param_p_ku": float(temp_params.get("p_Ku", 0.0)),
        "temp_param_p_d": float(temp_params.get("p_D", 0.0)),
        "temp_param_p_kc": float(temp_params.get("p_Kc", 0.0)),
        "temp_param_p_kc2": float(temp_params.get("p_Kc2", 0.0)),
        "temp_param_p_b1": float(temp_params.get("p_B1", 0.0)),
        "temp_param_p_b2": float(temp_params.get("p_B2", 0.0)),
        "temp_param_c_alpha": float(temp_params.get("c_alpha", 0.0)),
        "temp_param_alpha_min": float(temp_params.get("alpha_min", 0.0)),
        "temp_param_alpha_max": float(temp_params.get("alpha_max", 0.0)),
    }
    row.update(theta_t_row(params))
    return row
