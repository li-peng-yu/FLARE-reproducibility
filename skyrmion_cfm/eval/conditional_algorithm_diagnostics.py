"""Decision-oriented diagnostics for conditional micromagnetic generators.

The tests in this module are deliberately paired and training-free.  They
answer four questions before another long training run is started:

1. Does the generated distribution actually depend on the supplied physical
   condition, or mostly on ``m_init`` and the latent source?
2. Is the apparent diversity calibrated, collapsed, or over-dispersed?
3. Are ODE discretisation and the fitted source prior limiting inference?
4. Would factoring out the exactly solvable constant-field LLG motion reduce
   the endpoint-learning burden?

All condition interventions reuse the exact same source state.  Consequently
their output difference cannot be attributed to a lucky/unlucky latent draw.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import statistics
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from skyrmion_cfm.cfm.sampler import BridgeSampler
from skyrmion_cfm.config import get_device, load_config, merge_config, seed_everything
from skyrmion_cfm.data.conditions import audit_scalar_conditions
from skyrmion_cfm.data.fixed_time import (
    build_fixed_time_datasets,
    collate_fixed_time_conditions,
)
from skyrmion_cfm.eval.metrics import angular_error_deg, topological_charge
from skyrmion_cfm.models import build_model
from skyrmion_cfm.train import (
    _v4_segment_condition_rows,
    make_bridge_and_priors,
    make_training_stats,
    move_batch,
)


GAMMA_RAD_S_T = 2.0 * math.pi * 28.0e9


def _autocast(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if hasattr(value, "__dict__"):
        return value.__dict__
    raise TypeError(f"Cannot JSON-encode {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    tmp.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")
    tmp.replace(path)


def _attach_condition_audit(cfg: dict[str, Any], train_ds) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    horizons = [float(x) for x in cfg["data"].get("t_end_ns", [0.25])]
    for rec in train_ds.records:
        if getattr(rec, "is_v4", False):
            rows.extend(_v4_segment_condition_rows(rec, horizons))
        else:
            row = rec.condition_row(dt_s=0.0)
            row.update(rec.material_row())
            rows.append(row)
    audit = audit_scalar_conditions(rows)
    cfg["condition_audit"] = audit
    return audit


def _build_sampler(cfg: dict[str, Any], stats, ode_steps: int | None = None) -> BridgeSampler:
    bridge, rotation_prior, cart_prior, rfm_prior = make_bridge_and_priors(cfg, stats)
    return BridgeSampler(
        bridge=bridge,
        rotation_prior=rotation_prior,
        cart_prior=cart_prior,
        rfm_prior=rfm_prior,
        ode_steps=int(ode_steps if ode_steps is not None else cfg["sampler"].get("ode_steps", 20)),
        method=str(cfg["sampler"].get("method", "heun")),
    )


def _strip_compile_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    if not any(key.startswith(prefix) for key in state):
        return state
    return {
        key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in state.items()
    }


def _load_state_verified(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    state_name: str,
) -> dict[str, Any]:
    if state_name == "ema":
        state = checkpoint.get("ema")
        if state is None:
            raise KeyError("checkpoint has no EMA state")
    elif state_name in {"raw", "model"}:
        state = checkpoint["model"]
    else:
        raise ValueError(f"unknown model state {state_name!r}")
    state = _strip_compile_prefix(state)
    target = model.state_dict()
    matched = {
        key
        for key, value in state.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    matched_numel = sum(target[key].numel() for key in matched)
    total_numel = sum(value.numel() for value in target.values())
    fraction = matched_numel / max(1, total_numel)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if fraction < 0.999 or missing or unexpected:
        raise RuntimeError(
            "checkpoint/model mismatch: "
            f"matched_fraction={fraction:.6f}, missing={list(missing)[:8]}, "
            f"unexpected={list(unexpected)[:8]}"
        )
    return {
        "state": state_name,
        "matched_parameter_fraction": fraction,
        "matched_keys": len(matched),
        "target_keys": len(target),
    }


def _loader(dataset, count: int, batch_size: int, workers: int) -> DataLoader:
    count = min(max(0, int(count)), len(dataset))
    return DataLoader(
        Subset(dataset, range(count)),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(workers > 0),
    )


class _OnePerTrajectoryDataset(Dataset):
    """Deterministically select one fixed condition from each trajectory."""

    def __init__(self, base, seed: int, choice_policy: str = "random") -> None:
        self.base = base
        self.seed = int(seed)
        self.choice_policy = str(choice_policy)
        self.record_indices = [int(x) for x in base._valid_record_idx]

    def __len__(self) -> int:
        return len(self.record_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec_idx = self.record_indices[int(index)]
        choices = self.base._record_choices[rec_idx]
        if not choices:
            raise RuntimeError(f"trajectory {rec_idx} has no valid condition")
        if self.choice_policy == "min":
            choice_idx = min(range(len(choices)), key=lambda i: choices[i][0])
        elif self.choice_policy == "max":
            choice_idx = max(range(len(choices)), key=lambda i: choices[i][0])
        elif self.choice_policy == "random":
            rng_choice = np.random.default_rng(self.seed + 65_537 * (int(index) + 1))
            choice_idx = int(rng_choice.integers(0, len(choices)))
        else:
            raise ValueError(f"unknown trajectory choice policy {self.choice_policy!r}")
        rng = np.random.default_rng(self.seed + 104_729 * (int(index) + 1))
        return self.base._build_sample(
            rec_idx,
            choice_idx,
            rng,
            apply_augment=False,
        )


def _repeat_cond(cond: dict[str, torch.Tensor], repeats: int) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in cond.items():
        if value is None:
            continue
        out[key] = value.repeat_interleave(int(repeats), dim=0)
    return out


def _repeat_first(x: torch.Tensor, repeats: int) -> torch.Tensor:
    return x.repeat_interleave(int(repeats), dim=0)


def _mask_for_metric(batch: dict[str, Any]) -> torch.Tensor | None:
    return batch.get("defect_field")


def _pair_distances(pred: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Return B x C(K,2) image-level angular distances for B x K x C x H x W."""
    batch, draws = pred.shape[:2]
    values: list[torch.Tensor] = []
    for i, j in itertools.combinations(range(draws), 2):
        values.append(angular_error_deg(pred[:, i], pred[:, j], mask=mask))
    if not values:
        return pred.new_zeros(batch, 0)
    return torch.stack(values, dim=1)


def _ensemble_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    m_init: torch.Tensor,
    mask: torch.Tensor | None,
    boundary: str,
) -> dict[str, torch.Tensor]:
    """Metrics per condition for pred shaped B x K x 3 x H x W."""
    batch, draws = pred.shape[:2]
    target_rep = target[:, None].expand(-1, draws, -1, -1, -1)
    mask_rep = None if mask is None else _repeat_first(mask, draws)
    target_dist = angular_error_deg(
        pred.reshape(batch * draws, *pred.shape[2:]),
        target_rep.reshape(batch * draws, *target.shape[1:]),
        mask=mask_rep,
    ).reshape(batch, draws)
    pair = _pair_distances(pred, mask)
    pair_mean = pair.mean(dim=1) if pair.shape[1] else pred.new_zeros(batch)
    spherical_mean = F.normalize(pred.float().mean(dim=1), dim=1, eps=1.0e-8)
    q_pred = topological_charge(
        pred.reshape(batch * draws, *pred.shape[2:]).float(), boundary=boundary
    ).reshape(batch, draws)
    q_target = topological_charge(target.float(), boundary=boundary)
    return {
        "model_target_mean_deg": target_dist.mean(dim=1),
        "model_target_min_deg": target_dist.min(dim=1).values,
        "model_model_mean_deg": pair_mean,
        "model_model_median_deg": pair.median(dim=1).values if pair.shape[1] else pair_mean,
        "energy_score_deg": target_dist.mean(dim=1) - 0.5 * pair_mean,
        "ensemble_mean_target_deg": angular_error_deg(spherical_mean, target, mask=mask),
        "persistence_deg": angular_error_deg(m_init, target, mask=mask),
        "q_target": q_target,
        "q_pred_mean": q_pred.mean(dim=1),
        "q_pred_std": q_pred.std(dim=1, unbiased=False),
        "q_abs_error_mean": (q_pred - q_target[:, None]).abs().mean(dim=1),
    }


def _decode_onehot(batch: dict[str, Any], key: str, index: int) -> int | None:
    value = batch.get(f"{key}_onehot")
    if not torch.is_tensor(value):
        return None
    return int(value[index].argmax().item())


def _metadata_row(batch: dict[str, Any], index: int, global_index: int) -> dict[str, Any]:
    t_ns = float(batch["t_end_ns"][index].item())
    temp = float(batch["temp_k"][index].item())
    b = batch["b_t"][index].float()
    b_mag = float(b.norm().item())
    run_ids = batch.get("run_id")
    run_id = run_ids[index] if isinstance(run_ids, (list, tuple)) else str(run_ids)
    return {
        "sample_index": int(global_index),
        "run_id": str(run_id),
        "t_end_ns": t_ns,
        "temperature_k": temp,
        "b_mag_t": b_mag,
        "n_prec_external": 28.0 * b_mag * t_ns,
        "drive_active_kind_index": _decode_onehot(batch, "drive_active_kind", index),
        "material_family_index": _decode_onehot(batch, "material_family", index),
        "dataset_profile_index": _decode_onehot(batch, "dataset_profile", index),
        "geometry_mode_index": _decode_onehot(batch, "geometry_mode", index),
    }


def _tensor_rows(metrics: dict[str, torch.Tensor], batch_size: int) -> list[dict[str, float]]:
    rows = [dict() for _ in range(batch_size)]
    for key, value in metrics.items():
        vals = value.detach().float().cpu().tolist()
        for i, item in enumerate(vals):
            rows[i][key] = float(item)
    return rows


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    x = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(x, q).item())


def _aggregate(rows: list[dict[str, Any]], metric_keys: Iterable[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for key in metric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
        if not values:
            continue
        out[key] = {
            "mean": float(statistics.fmean(values)),
            "median": float(statistics.median(values)),
            "p10": _quantile(values, 0.1),
            "p90": _quantile(values, 0.9),
        }
    return out


def _temp_bin(value: float) -> str:
    if value < 1.0:
        return "T<1K"
    if value < 50.0:
        return "1-50K"
    if value < 150.0:
        return "50-150K"
    if value < 250.0:
        return "150-250K"
    return "T>=250K"


def _phase_bin(value: float) -> str:
    if value < 0.5:
        return "N<0.5"
    if value < 1.0:
        return "0.5<=N<1"
    if value < 2.0:
        return "1<=N<2"
    if value < 5.0:
        return "2<=N<5"
    return "N>=5"


ENSEMBLE_METRICS = (
    "model_target_mean_deg",
    "model_target_min_deg",
    "model_model_mean_deg",
    "energy_score_deg",
    "ensemble_mean_target_deg",
    "persistence_deg",
    "q_abs_error_mean",
    "q_pred_std",
)


def _stratify_ensemble(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "t_end_ns": defaultdict(list),
        "temperature": defaultdict(list),
        "external_phase": defaultdict(list),
        "drive_active_kind_index": defaultdict(list),
        "material_family_index": defaultdict(list),
        "dataset_profile_index": defaultdict(list),
    }
    for row in rows:
        groups["t_end_ns"][f"{row['t_end_ns']:g}"].append(row)
        groups["temperature"][_temp_bin(row["temperature_k"])].append(row)
        groups["external_phase"][_phase_bin(row["n_prec_external"])].append(row)
        for key in ("drive_active_kind_index", "material_family_index", "dataset_profile_index"):
            groups[key][str(row.get(key))].append(row)
    return {
        group: {name: _aggregate(items, ENSEMBLE_METRICS) for name, items in values.items()}
        for group, values in groups.items()
    }


@torch.no_grad()
def run_ensemble(
    model: torch.nn.Module,
    sampler: BridgeSampler,
    dataset,
    *,
    count: int,
    draws: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    boundary: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    started = time.time()
    loader = _loader(dataset, count, batch_size, workers)
    for batch_index, batch_cpu in enumerate(loader):
        batch = move_batch(batch_cpu, device)
        cond = collate_fixed_time_conditions(batch)
        m_init = batch["m_init"]
        target = batch["m_t"]
        cond_k = _repeat_cond(cond, draws)
        init_k = _repeat_first(m_init, draws)
        with _autocast(device):
            pred, _ = sampler.sample(model, init_k, cond_k)
        pred = pred.float().reshape(m_init.shape[0], draws, *m_init.shape[1:])
        metrics = _ensemble_metrics(
            pred, target, m_init, _mask_for_metric(batch), boundary
        )
        metric_rows = _tensor_rows(metrics, m_init.shape[0])
        for i, metric_row in enumerate(metric_rows):
            rows.append({**_metadata_row(batch_cpu, i, offset + i), **metric_row})
        offset += m_init.shape[0]
        if batch_index % 8 == 0:
            print(
                json.dumps(
                    {"test": "ensemble", "done": offset, "total": min(count, len(dataset))}
                ),
                flush=True,
            )
    summary = {
        "conditions": len(rows),
        "draws_per_condition": int(draws),
        "elapsed_s": time.time() - started,
        "overall": _aggregate(rows, ENSEMBLE_METRICS),
        "stratified": _stratify_ensemble(rows),
    }
    if rows:
        ratios = [
            row["model_model_mean_deg"] / max(row["model_target_mean_deg"], 1.0e-8)
            for row in rows
        ]
        summary["diversity_to_target_ratio"] = {
            "mean": statistics.fmean(ratios),
            "median": statistics.median(ratios),
        }
    return rows, summary


TIME_KEYS = {
    "t_end_index",
    "t_end_ns",
    "t_end_s",
    "dt_index",
    "dt_s",
    "drive_time_s",
    "relax_time_s",
    "drive_fraction",
    "anchor_offset_s",
    "anchor_offset_norm",
    "target_offset_norm",
}


def _condition_groups(keys: Iterable[str]) -> dict[str, set[str]]:
    keys = set(keys)
    field = {
        key
        for key in keys
        if key == "b_t"
        or key.startswith("b_ext_")
        or key in {"b_x_t", "b_y_t", "b_z_t", "has_field"}
        or key in {"b_x_field", "b_y_field", "b_z_field"}
    }
    current = {
        key
        for key in keys
        if key in {
            "current_a_m2",
            "control_grid",
            "pol_eff",
            "epsilon_prime",
            "lambda_sl",
            "beta_zl",
            "theta_dl_eff",
            "r_fl_dl",
        }
        or key.startswith("j_")
        or key.startswith("charge_current_")
        or key.startswith("sot_")
        or key.startswith("polarization_")
        or key.startswith("fixed_layer_")
        or key.startswith("has_zhang")
        or key.startswith("has_slon")
        or key.startswith("has_sot")
    }
    thermal = {
        key
        for key in keys
        if key == "temp_k"
        or "temperature" in key.lower()
        or key.startswith("temp_")
        or key.startswith("theta_T_")
        or key in {"m_reduced", "T_schedule_mode_onehot"}
    }
    material_tokens = (
        "alpha",
        "aex",
        "dind",
        "ku",
        "msat",
        "ms_",
        "a_t_",
        "a_ref_",
        "d_t_",
        "d_ref_",
        "kc",
        "anis_",
        "cubic_",
        "magnetoelastic",
        "b1_",
        "b2_",
        "thickness",
        "tc_k",
        "rex",
        "r_min",
        "ld_cells",
        "delta_dw",
        "domain_wall_width",
        "material_family",
        "DMI_TYPE",
    )
    material = {
        key for key in keys if any(token.lower() in key.lower() for token in material_tokens)
    }
    drive_category = {
        key
        for key in keys
        if key.startswith("drive_")
        or key.startswith("rendered_torque")
        or key in {"segment_role_onehot", "pre_relaxation_flag"}
    }
    protected = {
        "defect_field",
        "geometry_mode_onehot",
        "boundary_mode_onehot",
        "init_family_onehot",
        "dataset_profile_onehot",
    }
    dynamic = keys - protected
    groups = {
        "time": TIME_KEYS & keys,
        "field": field,
        "current": current,
        "thermal": thermal,
        "material": material,
        "drive_bundle": field | current | drive_category,
        "all_physics": dynamic,
    }
    return {name: values for name, values in groups.items() if values}


def _shuffled_condition(
    cond: dict[str, torch.Tensor],
    keys: set[str],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    first = next(iter(cond.values()))
    batch = first.shape[0]
    perm = torch.roll(torch.arange(batch, device=first.device), shifts=1)
    out = dict(cond)
    changed = torch.zeros(batch, dtype=torch.bool, device=first.device)
    for key in keys:
        value = cond.get(key)
        if value is None or value.shape[0] != batch:
            continue
        shuffled = value.index_select(0, perm)
        out[key] = shuffled
        changed |= (value != shuffled).reshape(batch, -1).any(dim=1)
    return out, changed


def _embedding_relative_shift(
    model: torch.nn.Module,
    cond_a: dict[str, torch.Tensor],
    cond_b: dict[str, torch.Tensor],
) -> torch.Tensor:
    batch = next(iter(cond_a.values())).shape[0]
    device = next(iter(cond_a.values())).device
    tau = torch.full((batch,), 0.5, device=device)
    a = model.cond_embed(tau, cond_a).float()
    b = model.cond_embed(tau, cond_b).float()
    return (a - b).norm(dim=1) / a.norm(dim=1).clamp_min(1.0e-8)


@torch.no_grad()
def run_interventions(
    model: torch.nn.Module,
    sampler: BridgeSampler,
    dataset,
    *,
    count: int,
    draws: int,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    started = time.time()
    loader = _loader(dataset, count, max(2, batch_size), workers)
    for batch_index, batch_cpu in enumerate(loader):
        batch = move_batch(batch_cpu, device)
        cond = collate_fixed_time_conditions(batch)
        m_init = batch["m_init"]
        target = batch["m_t"]
        mask = _mask_for_metric(batch)
        batch_n = m_init.shape[0]
        cond_k = _repeat_cond(cond, draws)
        init_k = _repeat_first(m_init, draws)
        target_k = _repeat_first(target, draws)
        mask_k = None if mask is None else _repeat_first(mask, draws)
        state0, _ = sampler._init_state(init_k, cond_k)
        with _autocast(device):
            base, _ = sampler.sample(model, init_k, cond_k, state0=state0)
        base = base.float()
        base_target = angular_error_deg(base, target_k, mask=mask_k).reshape(batch_n, draws)
        base_reshaped = base.reshape(batch_n, draws, *m_init.shape[1:])
        base_pair = _pair_distances(base_reshaped, mask)
        base_diversity = (
            base_pair.mean(dim=1) if base_pair.shape[1] else base.new_zeros(batch_n)
        )
        groups = _condition_groups(cond.keys())
        for group_name, group_keys in groups.items():
            changed_cond, changed = _shuffled_condition(cond, group_keys)
            changed_k = _repeat_cond(changed_cond, draws)
            with _autocast(device):
                perturbed, _ = sampler.sample(model, init_k, changed_k, state0=state0)
            perturbed = perturbed.float()
            shift = angular_error_deg(base, perturbed, mask=mask_k).reshape(batch_n, draws).mean(dim=1)
            pert_target = angular_error_deg(
                perturbed, target_k, mask=mask_k
            ).reshape(batch_n, draws).mean(dim=1)
            emb_shift = _embedding_relative_shift(model, cond, changed_cond)
            for i in range(batch_n):
                rows.append(
                    {
                        **_metadata_row(batch_cpu, i, offset + i),
                        "group": group_name,
                        "changed": bool(changed[i].item()),
                        "changed_key_count": len(group_keys),
                        "base_target_deg": float(base_target[i].mean().item()),
                        "shuffled_target_deg": float(pert_target[i].item()),
                        "target_score_delta_deg": float(
                            pert_target[i].item() - base_target[i].mean().item()
                        ),
                        "paired_output_shift_deg": float(shift[i].item()),
                        "same_condition_diversity_deg": float(base_diversity[i].item()),
                        "embedding_relative_shift": float(emb_shift[i].item()),
                    }
                )
        offset += batch_n
        if batch_index % 4 == 0:
            print(
                json.dumps(
                    {"test": "condition_intervention", "done": offset, "total": min(count, len(dataset))}
                ),
                flush=True,
            )
    metric_keys = (
        "base_target_deg",
        "shuffled_target_deg",
        "target_score_delta_deg",
        "paired_output_shift_deg",
        "same_condition_diversity_deg",
        "embedding_relative_shift",
    )
    by_group: dict[str, Any] = {}
    for group in sorted({row["group"] for row in rows}):
        selected = [row for row in rows if row["group"] == group and row["changed"]]
        summary = _aggregate(selected, metric_keys)
        ratios = [
            row["paired_output_shift_deg"] / max(row["same_condition_diversity_deg"], 1.0e-8)
            for row in selected
        ]
        if ratios:
            summary["condition_shift_to_latent_diversity"] = {
                "mean": statistics.fmean(ratios),
                "median": statistics.median(ratios),
            }
        summary["changed_fraction"] = len(selected) / max(
            1, sum(row["group"] == group for row in rows)
        )
        by_group[group] = summary
    return rows, {
        "conditions": offset,
        "draws_per_condition": int(draws),
        "elapsed_s": time.time() - started,
        "by_group": by_group,
    }


@torch.no_grad()
def run_ode_sweep(
    model: torch.nn.Module,
    sampler: BridgeSampler,
    dataset,
    *,
    count: int,
    steps: list[int],
    batch_size: int,
    workers: int,
    device: torch.device,
) -> dict[str, Any]:
    accum: dict[str, list[float]] = defaultdict(list)
    started = time.time()
    original_steps = sampler.ode_steps
    try:
        for batch_cpu in _loader(dataset, count, batch_size, workers):
            batch = move_batch(batch_cpu, device)
            cond = collate_fixed_time_conditions(batch)
            m_init = batch["m_init"]
            target = batch["m_t"]
            mask = _mask_for_metric(batch)
            state0, _ = sampler._init_state(m_init, cond)
            outputs: dict[int, torch.Tensor] = {}
            for n_steps in steps:
                sampler.ode_steps = int(n_steps)
                with _autocast(device):
                    outputs[n_steps], _ = sampler.sample(model, m_init, cond, state0=state0)
                outputs[n_steps] = outputs[n_steps].float()
                accum[f"target_deg_steps_{n_steps}"].extend(
                    angular_error_deg(outputs[n_steps], target, mask=mask).cpu().tolist()
                )
            reference = outputs[steps[-1]]
            for n_steps in steps[:-1]:
                accum[f"to_steps_{steps[-1]}_deg_steps_{n_steps}"].extend(
                    angular_error_deg(outputs[n_steps], reference, mask=mask).cpu().tolist()
                )
            for left, right in zip(steps[:-1], steps[1:], strict=True):
                accum[f"delta_deg_{left}_to_{right}"].extend(
                    angular_error_deg(outputs[left], outputs[right], mask=mask).cpu().tolist()
                )
    finally:
        sampler.ode_steps = original_steps
    return {
        "conditions": min(count, len(dataset)),
        "steps": steps,
        "elapsed_s": time.time() - started,
        "metrics": {
            key: _aggregate([{"x": value} for value in values], ("x",))["x"]
            for key, values in accum.items()
        },
    }


@torch.no_grad()
def run_prior_scale_sweep(
    model: torch.nn.Module,
    sampler: BridgeSampler,
    dataset,
    *,
    count: int,
    draws: int,
    scales: list[float],
    batch_size: int,
    workers: int,
    device: torch.device,
    boundary: str,
) -> dict[str, Any]:
    rows_by_scale: dict[float, list[dict[str, float]]] = defaultdict(list)
    started = time.time()
    for batch in _loader(dataset, count, batch_size, workers):
        batch = move_batch(batch, device)
        cond = collate_fixed_time_conditions(batch)
        m_init = batch["m_init"]
        target = batch["m_t"]
        mask = _mask_for_metric(batch)
        cond_k = _repeat_cond(cond, draws)
        init_k = _repeat_first(m_init, draws)
        state0, _ = sampler._init_state(init_k, cond_k)
        for scale in scales:
            with _autocast(device):
                pred, _ = sampler.sample(model, init_k, cond_k, state0=state0 * float(scale))
            pred = pred.float().reshape(m_init.shape[0], draws, *m_init.shape[1:])
            metrics = _ensemble_metrics(pred, target, m_init, mask, boundary)
            rows_by_scale[float(scale)].extend(_tensor_rows(metrics, m_init.shape[0]))
    return {
        "conditions": min(count, len(dataset)),
        "draws_per_condition": int(draws),
        "elapsed_s": time.time() - started,
        "by_scale": {
            f"{scale:g}": _aggregate(rows, ENSEMBLE_METRICS)
            for scale, rows in sorted(rows_by_scale.items())
        },
    }


def _analytic_zeeman_llg(
    m_init: torch.Tensor,
    b_field: torch.Tensor,
    alpha: torch.Tensor,
    dt_s: torch.Tensor,
    sign: float,
) -> torch.Tensor:
    """Exact constant-field LLG solution, independently at every lattice site."""
    b_mag = b_field.float().norm(dim=1, keepdim=True)
    active = b_mag > 1.0e-12
    b_hat = b_field.float() / b_mag.clamp_min(1.0e-12)
    u0 = (m_init.float() * b_hat).sum(dim=1, keepdim=True).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    transverse = m_init.float() - u0 * b_hat
    e1 = transverse / transverse.norm(dim=1, keepdim=True).clamp_min(1.0e-8)
    e2 = torch.cross(b_hat, e1, dim=1)
    alpha = alpha.float().reshape(-1, 1, 1, 1)
    dt_s = dt_s.float().reshape(-1, 1, 1, 1)
    phase = GAMMA_RAD_S_T * b_mag * dt_s / (1.0 + alpha.square())
    damp = alpha * phase
    u = torch.tanh(torch.atanh(u0) + damp)
    radius = (1.0 - u.square()).clamp_min(0.0).sqrt()
    rotated = (
        u * b_hat
        + radius
        * (
            torch.cos(phase) * e1
            + float(sign) * torch.sin(phase) * e2
        )
    )
    rotated = F.normalize(rotated, dim=1, eps=1.0e-8)
    return torch.where(active.expand_as(rotated), rotated, m_init.float())


def _spatial_b_field(batch: dict[str, Any]) -> torch.Tensor | None:
    parts: list[torch.Tensor] = []
    for key in ("b_x_field", "b_y_field", "b_z_field"):
        value = batch.get(key)
        if value is None:
            return None
        if value.ndim == 3:
            value = value.unsqueeze(1)
        parts.append(value.float())
    return torch.cat(parts, dim=1)


@torch.no_grad()
def run_analytic_precondition_audit(
    dataset,
    *,
    count: int,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    started = time.time()
    for batch_cpu in _loader(dataset, count, batch_size, workers):
        batch = move_batch(batch_cpu, device)
        m_init = batch["m_init"].float()
        target = batch["m_t"].float()
        mask = _mask_for_metric(batch)
        alpha = batch.get("alpha_t", batch.get("alpha"))
        if alpha is None:
            alpha = torch.zeros(m_init.shape[0], device=device)
        uniform_b = batch["b_t"].float()[:, :, None, None].expand_as(m_init)
        spatial_b = _spatial_b_field(batch)
        variants: dict[str, torch.Tensor] = {
            "uniform_plus": _analytic_zeeman_llg(m_init, uniform_b, alpha, batch["t_end_s"], +1.0),
            "uniform_minus": _analytic_zeeman_llg(m_init, uniform_b, alpha, batch["t_end_s"], -1.0),
        }
        if spatial_b is not None:
            variants["spatial_plus"] = _analytic_zeeman_llg(
                m_init, spatial_b, alpha, batch["t_end_s"], +1.0
            )
            variants["spatial_minus"] = _analytic_zeeman_llg(
                m_init, spatial_b, alpha, batch["t_end_s"], -1.0
            )
        identity = angular_error_deg(m_init, target, mask=mask)
        errors = {
            name: angular_error_deg(pred, target, mask=mask)
            for name, pred in variants.items()
        }
        for i in range(m_init.shape[0]):
            row = {
                **_metadata_row(batch_cpu, i, offset + i),
                "identity_target_deg": float(identity[i].item()),
            }
            for name, value in errors.items():
                row[f"{name}_target_deg"] = float(value[i].item())
                row[f"{name}_improvement_deg"] = float(identity[i].item() - value[i].item())
            rows.append(row)
        offset += m_init.shape[0]
    metric_keys = ["identity_target_deg"]
    for name in ("uniform_plus", "uniform_minus", "spatial_plus", "spatial_minus"):
        if rows and f"{name}_target_deg" in rows[0]:
            metric_keys.extend((f"{name}_target_deg", f"{name}_improvement_deg"))
    phase_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        phase_groups[_phase_bin(row["n_prec_external"])].append(row)
    summary = {
        "conditions": len(rows),
        "elapsed_s": time.time() - started,
        "overall": _aggregate(rows, metric_keys),
        "by_external_phase": {
            key: _aggregate(value, metric_keys) for key, value in phase_groups.items()
        },
    }
    for name in ("uniform_plus", "uniform_minus", "spatial_plus", "spatial_minus"):
        key = f"{name}_improvement_deg"
        if rows and key in rows[0]:
            summary[f"{name}_improved_fraction"] = statistics.fmean(
                float(row[key] > 0.0) for row in rows
            )
    return rows, summary


def _audit_summary(audit: dict[str, Any], stats, model: torch.nn.Module) -> dict[str, Any]:
    disabled_varying = [
        {
            "key": key,
            "mean": float(item.mean),
            "std": float(item.std),
            "coverage": float(item.coverage),
        }
        for key, item in audit.items()
        if not item.enabled and item.std > 0.0 and item.coverage >= 0.9
    ]
    disabled_varying.sort(key=lambda row: row["std"])
    embedder = getattr(model, "cond_embed", None)
    enabled_material = list(getattr(embedder, "enabled_material_keys", ()))
    stats_rows = []
    for key, mean, std in zip(
        stats.condition.keys,
        stats.condition.mean.detach().cpu().tolist(),
        stats.condition.std.detach().cpu().tolist(),
        strict=True,
    ):
        stats_rows.append({"key": key, "mean": float(mean), "stored_std": float(std)})
    return {
        "audit_min_std_is_absolute_si": True,
        "disabled_but_varying": disabled_varying,
        "disabled_but_varying_count": len(disabled_varying),
        "enabled_material_keys": enabled_material,
        "condition_stats": stats_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--config-stage",
        default=None,
        help="Merge the named entry from a staged training config before building the model.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--one-per-trajectory",
        action="store_true",
        help="Use each split trajectory exactly once instead of stochastic dataset indices.",
    )
    parser.add_argument(
        "--trajectory-choice",
        choices=("random", "min", "max"),
        default="random",
    )
    parser.add_argument("--state", choices=("ema", "raw"), default="ema")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ensemble-count", type=int, default=359)
    parser.add_argument("--ensemble-draws", type=int, default=8)
    parser.add_argument("--intervention-count", type=int, default=128)
    parser.add_argument("--intervention-draws", type=int, default=2)
    parser.add_argument("--ode-count", type=int, default=48)
    parser.add_argument("--ode-steps", type=int, nargs="+", default=[10, 20, 40, 80])
    parser.add_argument("--prior-count", type=int, default=64)
    parser.add_argument("--prior-draws", type=int, default=4)
    parser.add_argument("--prior-scales", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25])
    parser.add_argument("--analytic-count", type=int, default=1000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    cfg = load_config(args.config)
    if args.config_stage is not None:
        stages = cfg.get("stages") or []
        matches = [stage for stage in stages if str(stage.get("name")) == args.config_stage]
        if len(matches) != 1:
            available = [str(stage.get("name")) for stage in stages]
            raise ValueError(
                f"config stage {args.config_stage!r} matched {len(matches)} entries; "
                f"available stages: {available}"
            )
        base_cfg = {key: value for key, value in cfg.items() if key != "stages"}
        cfg = merge_config(base_cfg, matches[0].get("overrides", {}) or {})
    if bool(cfg.get("performance", {}).get("use_fused_ops", False)):
        os.environ["SKYRMION_CFM_FUSED_OPS"] = "1"
    cfg.setdefault("data", {}).setdefault("augment", {})["enabled"] = False
    cfg["data"].setdefault("memmap", {})["auto_build"] = False
    cfg["data"]["memmap"]["force_rebuild"] = False
    device = get_device(args.device)

    print(json.dumps({"phase": "build_datasets", "device": str(device)}), flush=True)
    train_ds, val_ds, test_ds = build_fixed_time_datasets(cfg)
    stats = make_training_stats(train_ds, cfg)
    audit = _attach_condition_audit(cfg, train_ds)
    dataset = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split]
    if args.one_per_trajectory:
        dataset = _OnePerTrajectoryDataset(
            dataset,
            seed=args.seed,
            choice_policy=args.trajectory_choice,
        )

    model = build_model(cfg, stats.condition).to(device)
    checkpoint = torch.load(
        Path(args.checkpoint), map_location="cpu", weights_only=False, mmap=True
    )
    load_report = _load_state_verified(model, checkpoint, args.state)
    model.eval()
    sampler = _build_sampler(cfg, stats)
    boundary = str(cfg["data"].get("boundary", "open"))

    report: dict[str, Any] = {
        "config": str(args.config),
        "config_stage": args.config_stage,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_load": load_report,
        "split": args.split,
        "dataset_length": len(dataset),
        "one_per_trajectory": bool(args.one_per_trajectory),
        "trajectory_choice": args.trajectory_choice,
        "seed": args.seed,
        "device": str(device),
        "bridge": type(sampler.bridge).__name__,
        "sampler_method": sampler.method,
        "sampler_ode_steps": sampler.ode_steps,
        "condition_scaling_audit": _audit_summary(audit, stats, model),
    }
    _write_json(args.output_dir / "report.json", report)
    del checkpoint

    print(json.dumps({"phase": "ensemble"}), flush=True)
    ensemble_rows, report["ensemble"] = run_ensemble(
        model,
        sampler,
        dataset,
        count=args.ensemble_count,
        draws=args.ensemble_draws,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        boundary=boundary,
    )
    _write_jsonl(args.output_dir / "ensemble_rows.jsonl", ensemble_rows)
    _write_json(args.output_dir / "report.json", report)

    print(json.dumps({"phase": "condition_interventions"}), flush=True)
    intervention_rows, report["condition_interventions"] = run_interventions(
        model,
        sampler,
        dataset,
        count=args.intervention_count,
        draws=args.intervention_draws,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    _write_jsonl(args.output_dir / "condition_intervention_rows.jsonl", intervention_rows)
    _write_json(args.output_dir / "report.json", report)

    print(json.dumps({"phase": "ode_sweep"}), flush=True)
    report["ode_sweep"] = run_ode_sweep(
        model,
        sampler,
        dataset,
        count=args.ode_count,
        steps=sorted(set(args.ode_steps)),
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    _write_json(args.output_dir / "report.json", report)

    print(json.dumps({"phase": "prior_scale_sweep"}), flush=True)
    report["prior_scale_sweep"] = run_prior_scale_sweep(
        model,
        sampler,
        dataset,
        count=args.prior_count,
        draws=args.prior_draws,
        scales=args.prior_scales,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        boundary=boundary,
    )
    _write_json(args.output_dir / "report.json", report)

    print(json.dumps({"phase": "analytic_precondition_audit"}), flush=True)
    analytic_rows, report["analytic_precondition_audit"] = run_analytic_precondition_audit(
        dataset,
        count=args.analytic_count,
        batch_size=max(args.batch_size, 8),
        workers=args.workers,
        device=device,
    )
    _write_jsonl(args.output_dir / "analytic_precondition_rows.jsonl", analytic_rows)
    report["completed"] = True
    _write_json(args.output_dir / "report.json", report)
    print(json.dumps({"phase": "complete", "output": str(args.output_dir)}), flush=True)


if __name__ == "__main__":
    main()
