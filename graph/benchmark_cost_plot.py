from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skyrmion_cfm.cfm.sampler import BridgeSampler
from skyrmion_cfm.config import get_device, load_config, seed_everything
from skyrmion_cfm.data.conditions import audit_scalar_conditions
from skyrmion_cfm.data.fixed_time import build_fixed_time_datasets, collate_fixed_time_conditions
from skyrmion_cfm.models import build_model
from skyrmion_cfm.train import make_bridge_and_priors, make_training_stats, move_batch
from graph.x5_initial_horizon_dataset import X5InitialStateHorizonDataset


DEFAULT_CONFIG = "configs/base/x5.yaml"
DEFAULT_CHECKPOINT = Path(os.environ.get("FLARE_CHECKPOINT_ROOT", str(PROJECT_ROOT.parent / "FLARE_checkpoints"))) / "flare/core_seed78.pt"
HORIZONS_NS = (0.25, 0.5, 1.0, 2.0, 4.0)



def _clean_state_dict_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state.items():
        while key.startswith("module."):
            key = key.removeprefix("module.")
        while key.startswith("_orig_mod."):
            key = key.removeprefix("_orig_mod.")
        cleaned[key] = value
    return cleaned


def _project_root() -> Path:
    return PROJECT_ROOT


def _resolve_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser()


def _relocate_checkpoint_paths(value: Any) -> Any:
    from skyrmion_cfm.config import release_checkpoint_config
    return release_checkpoint_config(value)


def _build_condition_audit(cfg: dict[str, Any], train_ds: Any) -> None:
    audit_rows = []
    for rec in train_ds.records:
        row = rec.condition_row(dt_s=0.0)
        row.update(rec.material_row())
        audit_rows.append(row)
    cfg["condition_audit"] = audit_scalar_conditions(audit_rows)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return torch.autocast(device_type="cuda", dtype=dtype)


def _configure_attention_backends(precision: str) -> None:
    if precision == "fp32" or not torch.cuda.is_available():
        return
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)


def _load_ours(
    config: Path,
    checkpoint: Path,
    channels_last: bool,
) -> tuple[dict[str, Any], Any, Any, torch.nn.Module, BridgeSampler, torch.device]:
    cfg = _relocate_checkpoint_paths(load_config(config))
    if bool(cfg.get("performance", {}).get("use_fused_ops", False)):
        os.environ["SKYRMION_CFM_FUSED_OPS"] = "1"
    seed_everything(int(cfg.get("seed", 0)))
    device = get_device(cfg.get("device", "auto"))

    train_ds, val_ds, test_ds = build_fixed_time_datasets(cfg)
    stats = make_training_stats(train_ds, cfg)
    _build_condition_audit(cfg, train_ds)

    model = build_model(cfg, stats.condition).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("ema") or ckpt.get("model") or ckpt
    missing, unexpected = model.load_state_dict(_clean_state_dict_keys(state), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint load mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    bridge, rot_prior, cart_prior, rfm_prior = make_bridge_and_priors(cfg, stats)
    sampler = BridgeSampler(
        bridge=bridge,
        rotation_prior=rot_prior,
        cart_prior=cart_prior,
        rfm_prior=rfm_prior,
        ode_steps=int(cfg["sampler"].get("ode_steps", 50)),
        method=str(cfg["sampler"].get("method", "heun")),
        classifier_free_guidance=cfg.get("sampler", {}).get("classifier_free_guidance"),
        stochastic_sampler=cfg.get("sampler", {}).get("stochastic_sampler"),
    )
    return cfg, val_ds, test_ds, model, sampler, device


@torch.no_grad()
def benchmark_ours(
    config: Path,
    checkpoint: Path,
    checkpoint_label: str,
    split: str,
    samples_per_horizon: int,
    batch_size: int,
    warmup_batches: int,
    repeats: int,
    cache_batches: bool,
    ode_steps: int | None,
    precision: str,
    channels_last: bool,
) -> dict[str, Any]:
    _configure_attention_backends(precision)
    cfg, val_ds, test_ds, model, sampler, device = _load_ours(config, checkpoint, channels_last)
    if ode_steps is not None:
        sampler.ode_steps = int(ode_steps)
    dataset = val_ds if split == "val" else test_ds
    num_workers = 0
    timings: dict[str, Any] = {}
    selections: dict[str, Any] = {}

    for horizon in HORIZONS_NS:
        print({"stage": "construct_initial_horizon_pairs", "horizon_ns": horizon}, flush=True)
        subset = X5InitialStateHorizonDataset(dataset, horizon, samples_per_horizon)
        selections[f"{horizon:g}"] = subset.selection_metadata()
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        cached_batches: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
        if cache_batches:
            print({"stage": "cache_batches_to_gpu", "horizon_ns": horizon, "samples": len(subset)}, flush=True)
            for batch in loader:
                batch = move_batch(batch, device)
                if channels_last:
                    batch["m_init"] = batch["m_init"].contiguous(memory_format=torch.channels_last)
                cached_batches.append((batch["m_init"], collate_fixed_time_conditions(batch)))
        else:
            for batch in loader:
                batch = move_batch(batch, device)
                if channels_last:
                    batch["m_init"] = batch["m_init"].contiguous(memory_format=torch.channels_last)
                cached_batches.append((batch["m_init"], collate_fixed_time_conditions(batch)))

        print({"stage": "warmup", "horizon_ns": horizon, "warmup_batches": warmup_batches}, flush=True)
        for warm_i, (m_init, cond) in enumerate(cached_batches):
            if warm_i >= warmup_batches:
                break
            with _autocast_context(device, precision):
                _ = sampler.sample(model, m_init, cond)
        _sync(device)

        runs: list[float] = []
        observed_samples = 0
        print({"stage": "timed_repeats", "horizon_ns": horizon, "repeats": repeats}, flush=True)
        for repeat_idx in range(repeats):
            total_s = 0.0
            total_n = 0
            for m_init, cond in cached_batches:
                _sync(device)
                start = time.perf_counter()
                with _autocast_context(device, precision):
                    _ = sampler.sample(model, m_init, cond)
                _sync(device)
                total_s += time.perf_counter() - start
                total_n += int(m_init.shape[0])
            observed_samples = max(observed_samples, total_n)
            if total_n:
                runs.append(1000.0 * total_s / total_n)
            print(
                {
                    "stage": "repeat_done",
                    "horizon_ns": horizon,
                    "repeat": repeat_idx + 1,
                    "ms_per_sample": runs[-1] if runs else None,
                    "samples": total_n,
                },
                flush=True,
            )
        timings[f"{horizon:g}"] = {
            "mean_ms": statistics.fmean(runs) if runs else None,
            "std_ms": statistics.stdev(runs) if len(runs) > 1 else 0.0,
            "samples": observed_samples,
            "runs": runs,
        }

    return {
        "unit": "ms / sample",
        "method": "Ours",
        "config": str(config),
        "checkpoint": str(checkpoint),
        "checkpoint_label": checkpoint_label,
        "split": split,
        "batch_size": batch_size,
        "precision": precision,
        "channels_last": channels_last,
        "cache_batches": cache_batches,
        "timing_excludes": ["DataLoader iteration", "CPU-to-GPU batch transfer", "condition collation"],
        "samples_per_horizon": samples_per_horizon,
        "unique_samples_total": int(sum(item["samples"] for item in timings.values())),
        "timed_inferences_total": int(sum(item["samples"] for item in timings.values()) * repeats),
        "repeats": repeats,
        "device": str(device),
        "ode_steps": int(sampler.ode_steps),
        "horizons": timings,
        "evaluation_dataset_view": {
            "t_end_ns": list(HORIZONS_NS),
            "control_segmented": False,
            "current_time_mode": "t0",
            "meaning": "full pulse/relax trajectory from the x5 initial state",
        },
        "selection": selections,
    }


def _load_baselines(path: Path | None) -> dict[str, dict[str, float | None]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    methods = payload.get("methods", payload)
    return {
        str(method): {str(k): (None if v is None else float(v)) for k, v in values.items()}
        for method, values in methods.items()
    }


def _load_ours_result(path: Path | None) -> dict[str, float | None]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text())
    horizons = payload.get("horizons", {})
    return {str(k): (None if v.get("mean_ms") is None else float(v["mean_ms"])) for k, v in horizons.items()}


def plot_costs(
    baselines: dict[str, dict[str, float | None]],
    ours: dict[str, float | None],
    output: Path,
    title: str,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = {
        "mumax3 reference": baselines.get("mumax3 reference", {}),
        "Direct U-Net": baselines.get("Direct U-Net", {}),
        "FNO": baselines.get("FNO", {}),
        "Ours": ours,
    }
    x_labels = [f"{h:g}" for h in HORIZONS_NS]
    x = list(range(len(x_labels)))

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    markers = ["o", "s", "^", "D"]
    plotted = 0
    for marker, (method, values) in zip(markers, methods.items(), strict=True):
        y = [values.get(label) for label in x_labels]
        if all(value is None for value in y):
            continue
        plotted += 1
        y_plot = [float("nan") if value is None else value for value in y]
        ax.plot(x, y_plot, marker=marker, linewidth=2.0, markersize=5.5, label=method)
    if plotted == 0:
        raise ValueError("no numeric cost values available to plot")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:g} ns" for h in HORIZONS_NS])
    ax.set_xlabel("prediction horizon")
    ax.set_ylabel("compute cost (ms / sample)")
    ax.set_title(title)
    ax.grid(True, which="major", color="#d8d8d8", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark compute cost and plot method scaling.")
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--checkpoint-label", default="flare_stage1")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--samples-per-horizon", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ode-steps", type=int, default=None, help="Override cfg sampler.ode_steps.")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument(
        "--cache-batches",
        action="store_true",
        help="Move batches to GPU before timing so upload/collate/DataLoader time is excluded.",
    )
    parser.add_argument("--baselines", type=Path, default=Path("graph/cost_baselines.json"))
    parser.add_argument("--ours-json", type=Path, default=Path("outputs/timing/flare_cost.json"))
    parser.add_argument("--plot", type=Path, default=Path("outputs/figures/compute_cost_by_horizon.png"))
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    checkpoint = _resolve_path(args.checkpoint)
    if not args.plot_only:
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"checkpoint not found: {args.checkpoint}; remapped candidate: {checkpoint}"
            )
        result = benchmark_ours(
            config=args.config,
            checkpoint=checkpoint,
            checkpoint_label=args.checkpoint_label,
            split=args.split,
            samples_per_horizon=args.samples_per_horizon,
            batch_size=args.batch_size,
            warmup_batches=args.warmup_batches,
            repeats=args.repeats,
            cache_batches=args.cache_batches,
            ode_steps=args.ode_steps,
            precision=args.precision,
            channels_last=args.channels_last,
        )
        args.ours_json.parent.mkdir(parents=True, exist_ok=True)
        args.ours_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    baseline_path = args.baselines if args.baselines.exists() else None
    baselines = _load_baselines(baseline_path)
    ours = _load_ours_result(args.ours_json)
    plot_costs(baselines, ours, args.plot, "Compute cost by prediction horizon")


if __name__ == "__main__":
    main()
