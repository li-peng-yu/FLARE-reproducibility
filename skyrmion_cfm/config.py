from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return _expand_paths(yaml.safe_load(f))


def _expand_paths(value: Any) -> Any:
    """Expand release asset locations without changing numerical settings."""
    if isinstance(value, dict):
        return {key: _expand_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_paths(item) for item in value]
    if isinstance(value, str):
        root = Path(__file__).resolve().parents[1]
        defaults = {
            "FLARE_DATASET_ROOT": str(root.parent / "FLARE_dataset"),
            "FLARE_CHECKPOINT_ROOT": str(root.parent / "FLARE_checkpoints"),
            "FLARE_TEMPERATURE_ROOT": str(root / "outputs/temperature_data"),
        }
        for name, default in defaults.items():
            value = value.replace("${" + name + "}", os.environ.get(name, default))
        return os.path.expandvars(value)
    return value


def release_checkpoint_config(config: dict[str, Any]) -> dict[str, Any]:
    """Relocate checkpoint data and statistics for evaluation.

    Model, prior, sampler and training parameters are retained. x5 inputs use
    float16 memmaps. Temperature evaluation supplies its raw-OVF data config
    after relocation. Build input caches before evaluation.
    """
    cfg = deepcopy(config)
    data = cfg.get("data", {})
    root = Path(__file__).resolve().parents[1]
    dataset_root = Path(os.environ.get("FLARE_DATASET_ROOT", str(root.parent / "FLARE_dataset")))
    roots = data.get("dataset_root", [])
    single = isinstance(roots, str)
    roots = [roots] if single else roots
    known = {"skx_bt_1000base_x5_20260803", "skx_bt_1000base_x5_sharedrelax_20260805"}
    x5 = bool(roots) and all(Path(p).name in known for p in roots)
    if x5:
        relocated = [str(dataset_root / "primary_x5" / Path(p).name) for p in roots]
        data["dataset_root"] = relocated[0] if single else relocated
        memmap = data.setdefault("memmap", {})
        memmap.update(enabled=True, dtype="float16", auto_build=False,
                      path=str(root / "cache/memmap_both_fp16"), force_rebuild=False)
        data["trajectory_index_cache"] = str(root / "cache/trajectory_index_both.pkl")
        if data.get("split_manifest"):
            data["split_manifest"] = str(root / "configs/data/ood_ring_split.json")
        old_stats = str(data.get("stats_cache", ""))
        if "ood" in old_stats.lower() or data.get("split_manifest"):
            stats = "ood_ring_fno_stats.json" if cfg.get("model", {}).get("arch") == "fno" else "ood_ring_scfm_stats.json"
        else:
            stats = "dataset_stats_both.json"
        data["stats_cache"] = str(root / "configs/stats" / stats)
    return _expand_paths(cfg)


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_config(out[key], value)
        else:
            out[key] = value
    return out


def get_device(device: str):
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def seed_everything(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
