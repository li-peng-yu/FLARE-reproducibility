from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


MORPHOLOGY_ORDER = ("localized", "mixed", "extended")


def evenly_spaced_bucket_indices(num_buckets: int, max_examples: int) -> set[int] | None:
    """Choose validation horizons across the full configured time range."""

    if num_buckets <= 0 or max_examples <= 0:
        return None
    if num_buckets <= max_examples:
        return set(range(num_buckets))
    if max_examples == 1:
        return {num_buckets - 1}
    values = np.rint(np.linspace(0, num_buckets - 1, max_examples)).astype(np.int64)
    return {int(value) for value in values}


def _batch_value(value: Any, index: int, default: Any = None) -> Any:
    if torch.is_tensor(value):
        item = value[index].detach().cpu()
        return item.item() if item.numel() == 1 else item
    if isinstance(value, (list, tuple)) and index < len(value):
        return value[index]
    return default


def _information_scores(
    m_init: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Return model-independent structure/dynamics scores for BCHW spins."""

    valid = target.square().sum(dim=1) > 0.25
    weight = valid.to(dtype=target.dtype)
    count = weight.flatten(1).sum(dim=1).clamp_min(1.0)
    target_z = target[:, 2]
    mean_z = (target_z * weight).flatten(1).sum(dim=1) / count
    centered = target_z - mean_z[:, None, None]
    variance = (centered.square() * weight).flatten(1).sum(dim=1) / count
    structure_std = variance.clamp_min(0.0).sqrt()
    # Measure how much target variation survives a local low-pass filter.
    # Pixel-scale stochastic texture has low coherence; domains, walls and
    # compact objects retain most of their variation. Fill void pixels with
    # the per-sample mean so geometry edges do not inflate this score.
    filled_z = torch.where(valid, target_z, mean_z[:, None, None])
    smooth_z = F.avg_pool2d(
        F.pad(filled_z[:, None], (4, 4, 4, 4), mode="replicate"),
        kernel_size=9,
        stride=1,
    )[:, 0]
    smooth_centered = smooth_z - mean_z[:, None, None]
    coherent_variance = (
        (smooth_centered.square() * weight).flatten(1).sum(dim=1) / count
    )
    coherence = (
        coherent_variance.clamp_min(0.0).sqrt() / structure_std.clamp_min(1.0e-6)
    ).clamp(0.0, 1.0)
    active_fraction = (
        ((centered.abs() > 0.25) & valid).to(dtype=target.dtype).flatten(1).sum(dim=1)
        / count
    )
    dot = (m_init * target).sum(dim=1).clamp(-1.0, 1.0)
    dynamics = ((1.0 - dot) * weight).flatten(1).sum(dim=1) / count
    information = (
        structure_std * (0.25 + 0.75 * coherence)
        + 0.40
        * active_fraction.clamp_min(0.0).sqrt()
        * coherence.clamp_min(0.0).sqrt()
        + 0.25 * dynamics.clamp_min(0.0).sqrt()
    )

    morphologies: list[str] = []
    for std_value, fraction_value, coherence_value in zip(
        structure_std.detach().cpu().tolist(),
        active_fraction.detach().cpu().tolist(),
        coherence.detach().cpu().tolist(),
        strict=True,
    ):
        if std_value < 0.04:
            morphology = "uniform"
        elif std_value > 0.08 and coherence_value < 0.60:
            morphology = "noisy"
        elif fraction_value < 0.08:
            morphology = "localized"
        elif fraction_value < 0.35:
            morphology = "mixed"
        else:
            morphology = "extended"
        morphologies.append(morphology)
    return information, active_fraction, dynamics, coherence, morphologies


def add_informative_candidates(
    groups: dict[tuple[int, str], list[dict[str, Any]]],
    *,
    m_init: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: dict[str, Any],
    mse_each: torch.Tensor,
    ang_each: torch.Tensor,
    q_pred_each: torch.Tensor,
    q_target_each: torch.Tensor,
    max_per_group: int = 2,
) -> None:
    """Keep a small deterministic set of informative candidates per horizon/type."""

    bucket_value = batch.get("t_end_index", batch.get("dt_index", batch.get("dt_scale")))
    if bucket_value is None:
        return
    information, active_fraction, dynamics, coherence, morphologies = _information_scores(
        m_init,
        target,
    )
    bucket_cpu = bucket_value.detach().to(device="cpu", dtype=torch.long)
    information_cpu = information.detach().cpu()
    active_cpu = active_fraction.detach().cpu()
    dynamics_cpu = dynamics.detach().cpu()
    coherence_cpu = coherence.detach().cpu()
    mse_cpu = mse_each.detach().cpu()
    ang_cpu = ang_each.detach().cpu()
    q_pred_cpu = q_pred_each.detach().cpu()
    q_target_cpu = q_target_each.detach().cpu()

    for index in range(int(target.shape[0])):
        bucket = int(bucket_cpu[index])
        morphology = morphologies[index]
        run_id = str(_batch_value(batch.get("run_id"), index, ""))
        frame_init = int(_batch_value(batch.get("frame_init", batch.get("frame0")), index, -1))
        frame_target = int(_batch_value(batch.get("frame_target"), index, -1))
        score = float(information_cpu[index])
        identity = (run_id, frame_init, frame_target, bucket)
        key = (bucket, morphology)
        kept = groups.setdefault(key, [])
        limit = max(1, int(max_per_group))
        sort_key = (-score, run_id, frame_init, frame_target)
        if any(tuple(item["identity"]) == identity for item in kept):
            continue
        if len(kept) >= limit:
            worst = kept[-1]
            worst_key = (
                -float(worst["information_score"]),
                str(worst["run_id"]),
                int(worst["frame_init"]),
                int(worst["frame_target"]),
            )
            if sort_key >= worst_key:
                continue
        candidate = {
            "initial": m_init[index].detach().cpu(),
            "pred": pred[index].detach().cpu(),
            "target": target[index].detach().cpu(),
            "t_end_ns": float(_batch_value(batch.get("t_end_ns"), index, float("nan"))),
            "t_end_index": bucket,
            "frame_init": frame_init,
            "frame_target": frame_target,
            "run_id": run_id,
            "morphology": morphology,
            "information_score": score,
            "active_fraction": float(active_cpu[index]),
            "dynamics_score": float(dynamics_cpu[index]),
            "coherence_score": float(coherence_cpu[index]),
            "mse": float(mse_cpu[index]),
            "ang": float(ang_cpu[index]),
            "q_pred": float(q_pred_cpu[index]),
            "q_target": float(q_target_cpu[index]),
            "identity": identity,
        }
        kept.append(candidate)
        kept.sort(
            key=lambda item: (
                -float(item["information_score"]),
                str(item["run_id"]),
                int(item["frame_init"]),
                int(item["frame_target"]),
            )
        )
        del kept[limit:]


def select_informative_candidates(
    groups: dict[tuple[int, str], list[dict[str, Any]]],
    max_examples: int,
    target_buckets: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Select horizon-balanced, morphology-diverse, non-uniform examples."""

    if max_examples <= 0:
        return []
    available_buckets = sorted({bucket for bucket, _ in groups})
    buckets = sorted(target_buckets) if target_buckets else available_buckets
    selected: list[dict[str, Any]] = []
    used: set[tuple[Any, ...]] = set()
    used_runs: set[str] = set()

    for position, bucket in enumerate(buckets):
        if len(selected) >= max_examples:
            break
        preferred = MORPHOLOGY_ORDER[position % len(MORPHOLOGY_ORDER)]
        structured_pool = [
            candidate
            for morphology in MORPHOLOGY_ORDER
            for candidate in groups.get((bucket, morphology), [])
        ]
        if structured_pool:
            high_information_pool = [
                item
                for item in structured_pool
                if float(item["information_score"]) >= 0.35
            ]
            preferred_pool = [
                item
                for item in high_information_pool
                if item["morphology"] == preferred
            ]
            pool = preferred_pool or high_information_pool
            pool.sort(
                key=lambda item: (
                    -float(item["information_score"]),
                    str(item["run_id"]),
                )
            )
        else:
            pool = [
                *groups.get((bucket, "noisy"), []),
                *groups.get((bucket, "uniform"), []),
            ]
        candidate = next(
            (
                item
                for item in pool
                if tuple(item["identity"]) not in used
                and str(item["run_id"]) not in used_runs
            ),
            None,
        )
        if candidate is None:
            candidate = next(
                (item for item in pool if tuple(item["identity"]) not in used),
                None,
            )
        if candidate is not None:
            selected.append(candidate)
            used.add(tuple(candidate["identity"]))
            used_runs.add(str(candidate["run_id"]))

    if len(selected) < max_examples:
        remaining = [
            candidate
            for candidates in groups.values()
            for candidate in candidates
            if tuple(candidate["identity"]) not in used
            and candidate["morphology"] not in {"uniform", "noisy"}
            and float(candidate["information_score"]) >= 0.35
        ]
        remaining.sort(
            key=lambda item: (
                2
                if item["morphology"] == "uniform"
                else 1
                if item["morphology"] == "noisy"
                else 0,
                -float(item["information_score"]),
                int(item["t_end_index"]),
                str(item["run_id"]),
            )
        )
        for candidate in remaining:
            if len(selected) >= max_examples:
                break
            identity = tuple(candidate["identity"])
            if identity in used:
                continue
            selected.append(candidate)
            used.add(identity)
            used_runs.add(str(candidate["run_id"]))
    return selected


def informative_roi_bounds(
    initial: torch.Tensor,
    target: torch.Tensor,
    *,
    min_side: int = 64,
    padding: int = 12,
) -> tuple[int, int, int, int]:
    """Find a square crop around target structure or actual state change."""

    height, width = int(target.shape[-2]), int(target.shape[-1])
    valid = target.square().sum(dim=0) > 0.25
    target_z = target[2]
    valid_values = target_z[valid]
    if valid_values.numel() == 0:
        return 0, height, 0, width
    background = valid_values.median()
    structure = (target_z - background).abs() > 0.20
    dot = (initial * target).sum(dim=0).clamp(-1.0, 1.0)
    changed = (1.0 - dot) > 0.05
    # Prefer the target structure itself. A global initial-to-target rotation
    # otherwise makes ``changed`` cover the whole sample and defeats zooming.
    focus = valid & structure
    if not bool(focus.any()):
        focus = valid & changed
    coords = focus.nonzero(as_tuple=False)
    if coords.numel() == 0:
        return 0, height, 0, width
    y0 = max(0, int(coords[:, 0].min()) - padding)
    y1 = min(height, int(coords[:, 0].max()) + 1 + padding)
    x0 = max(0, int(coords[:, 1].min()) - padding)
    x1 = min(width, int(coords[:, 1].max()) + 1 + padding)
    side = min(
        max(y1 - y0, x1 - x0, min(int(min_side), height, width)),
        height,
        width,
    )
    if side >= int(0.85 * max(height, width)):
        return 0, height, 0, width
    cy = 0.5 * (y0 + y1)
    cx = 0.5 * (x0 + x1)
    y0 = max(0, min(height - side, int(round(cy - 0.5 * side))))
    x0 = max(0, min(width - side, int(round(cx - 0.5 * side))))
    return y0, y0 + side, x0, x0 + side


def _spin_image(m: torch.Tensor, channel: str) -> torch.Tensor:
    channel_index = {"mx": 0, "my": 1, "mz": 2}.get(channel, 2)
    return m[channel_index].detach().float().cpu()


def save_informative_visualization(
    examples: list[dict[str, Any]],
    out_dir: Path,
    step: int,
    *,
    channel: str = "mz",
    dpi: int = 180,
    diff_vmax: float = 1.0,
    zoom: bool = True,
    filename_prefix: str = "",
) -> Path | None:
    if not examples:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as exc:  # pragma: no cover - cluster dependency
        print(
            {"val_visualization": "skipped", "reason": f"matplotlib import failed: {exc}"},
            flush=True,
        )
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    columns = 8 if zoom else 4
    fig, axes = plt.subplots(
        len(examples),
        columns,
        figsize=(2.25 * columns, 2.15 * len(examples)),
        squeeze=False,
    )
    full_titles = ("initial", "MuMax target", "prediction", f"|Δ{channel}|")
    zoom_titles = tuple(f"{title} zoom" for title in full_titles)
    for row_index, item in enumerate(examples):
        initial = item["initial"]
        target = item["target"]
        pred = item["pred"]
        initial_image = _spin_image(initial, channel)
        target_image = _spin_image(target, channel)
        pred_image = _spin_image(pred, channel)
        diff_image = (pred_image - target_image).abs()
        images = (initial_image, target_image, pred_image, diff_image)
        y0, y1, x0, x1 = informative_roi_bounds(initial, target)
        panels = list(images)
        if zoom:
            panels.extend(image[y0:y1, x0:x1] for image in images)
        for column, image in enumerate(panels):
            ax = axes[row_index, column]
            is_diff = column % 4 == 3
            ax.imshow(
                image.numpy(),
                cmap="magma" if is_diff else "RdBu_r",
                vmin=0.0 if is_diff else -1.0,
                vmax=max(float(diff_vmax), 1.0e-6) if is_diff else 1.0,
                origin="lower",
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title((full_titles + zoom_titles)[column], fontsize=9)
            if zoom and column < 3 and (y0, y1, x0, x1) != (
                0,
                int(target.shape[-2]),
                0,
                int(target.shape[-1]),
            ):
                ax.add_patch(
                    Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        fill=False,
                        edgecolor="gold",
                        linewidth=0.8,
                    )
                )
        run_id = str(item.get("run_id", ""))
        if len(run_id) > 38:
            run_id = f"…{run_id[-37:]}"
        q_target = float(item.get("q_target", float("nan")))
        q_pred = float(item.get("q_pred", float("nan")))
        label = (
            f"{run_id}\n"
            f"t={float(item.get('t_end_ns', float('nan'))):.3g} ns  "
            f"{item.get('morphology', '')}\n"
            f"MSE={float(item.get('mse', float('nan'))):.3g}  "
            f"ang={float(item.get('ang', float('nan'))):.1f}°\n"
            f"Q target/pred={q_target:.2f}/{q_pred:.2f}"
        )
        axes[row_index, 0].set_ylabel(label, fontsize=7.5, rotation=0, ha="right", va="center")

    fig.suptitle(
        f"Informative validation samples — step {int(step)} ({channel}); "
        f"shared error scale [0, {float(diff_vmax):g}]",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.11, 0.01, 1.0, 0.98), h_pad=0.75, w_pad=0.25)
    path = out_dir / f"{filename_prefix}val_step_{int(step):07d}_informative_{channel}.png"
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    sample_manifest = [
        {
            key: item.get(key)
            for key in (
                "run_id",
                "t_end_ns",
                "t_end_index",
                "frame_init",
                "frame_target",
                "morphology",
                "information_score",
                "active_fraction",
                "dynamics_score",
                "coherence_score",
                "mse",
                "ang",
                "q_target",
                "q_pred",
            )
        }
        for item in examples
    ]
    path.with_name(f"{path.stem}_samples.json").write_text(
        json.dumps(sample_manifest, indent=2),
        encoding="utf-8",
    )
    return path
