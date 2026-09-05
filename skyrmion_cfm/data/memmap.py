from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import errno
import json
import mmap
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from skyrmion_cfm.config import load_config
from skyrmion_cfm.data.ovf import read_ovf

DEFAULT_MEMMAP_WORKERS = min(8, os.cpu_count() or 1)
_WORKER_FRAMES: np.memmap | None = None
_WORKER_DTYPE: np.dtype | None = None


@contextlib.contextmanager
def _file_lock(lock_path: Path, poll_seconds: float = 30.0):
    """Small cross-process lock for shared memmap builds across concurrent processes."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()} time={time.time()}\n".encode("utf-8"))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            print(f"Waiting for memmap build lock: {lock_path}", file=sys.stderr, flush=True)
            time.sleep(poll_seconds)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _format_read_error(path: str | Path, exc: BaseException) -> str:
    return f"{path}: {type(exc).__name__}: {exc}"


class OVFMemmapStore:
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open("r", encoding="utf-8") as f:
            self.manifest: dict[str, Any] = json.load(f)
        self.shape = tuple(int(x) for x in self.manifest["shape"])
        self.dtype = np.dtype(self.manifest["dtype"])
        data_path = self.manifest_path.parent / self.manifest["data_file"]
        self.frames = np.memmap(data_path, mode="r", dtype=self.dtype, shape=self.shape)
        # Each dataloader worker maps the full frame store.  On large stores
        # this can make khugepaged spend a lot of CPU scanning terabytes of
        # virtual address space; the access pattern is also random by design.
        mm = getattr(self.frames, "_mmap", None)
        if mm is not None and hasattr(mm, "madvise"):
            for advice in (
                getattr(mmap, "MADV_NOHUGEPAGE", None),
                getattr(mmap, "MADV_RANDOM", None),
            ):
                if advice is None:
                    continue
                try:
                    mm.madvise(advice)
                except (OSError, ValueError):
                    pass
        self.run_to_index = {run_id: i for i, run_id in enumerate(self.manifest["run_ids"])}
        # Per-record frame count: when ``runs_per_record_n_frames`` is present
        # we use it to validate per-run frame access (variable-length runs
        # padded to ``shape[1]``); otherwise we fall back to the uniform
        # ``shape[1]`` for backward compat with v1 manifests.
        per_record = self.manifest.get("runs_per_record_n_frames")
        if per_record is not None:
            self.n_frames_per_run = [int(n) for n in per_record]
        else:
            self.n_frames_per_run = [int(self.shape[1])] * len(self.run_to_index)

    def n_frames(self, run_id: str) -> int:
        idx = self.run_to_index.get(run_id)
        if idx is None:
            raise KeyError(f"Run {run_id!r} is not present in {self.manifest_path}")
        return self.n_frames_per_run[idx]

    def get(self, run_id: str, frame_index: int) -> np.ndarray:
        idx = self.run_to_index.get(run_id)
        if idx is None:
            raise KeyError(f"Run {run_id!r} is not present in {self.manifest_path}")
        n = self.n_frames_per_run[idx]
        if frame_index < 0 or frame_index >= n:
            raise IndexError(
                f"Frame {frame_index} is outside run {run_id!r}'s frame range [0, {n})"
            )
        return np.asarray(self.frames[idx, frame_index], dtype=np.float32)


def manifest_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / "manifest.json"


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_roots(
    dataset_root: str | Path | list[str | Path] | tuple[str | Path, ...],
) -> list[str]:
    if isinstance(dataset_root, (str, Path)):
        return [str(dataset_root)]
    return [str(r) for r in dataset_root]


def _manifest_matches(
    manifest: dict[str, Any],
    dataset_root: str | Path | list[str | Path] | tuple[str | Path, ...],
    frame_glob: str,
    include_before_drive: bool,
    dtype: str,
) -> bool:
    expected_roots = _normalize_roots(dataset_root)
    stored = manifest.get("dataset_roots")
    if stored is None:
        # v1 manifests stored a single string under ``dataset_root``.
        stored = [str(manifest.get("dataset_root"))]
    return (
        list(stored) == expected_roots
        and manifest.get("frame_glob") == frame_glob
        and bool(manifest.get("include_before_drive", False)) == bool(include_before_drive)
        and manifest.get("dtype") == np.dtype(dtype).name
    )


def _init_memmap_worker(
    data_path: str,
    shape: tuple[int, ...],
    dtype_name: str,
) -> None:
    global _WORKER_FRAMES, _WORKER_DTYPE
    _WORKER_DTYPE = np.dtype(dtype_name)
    _WORKER_FRAMES = np.memmap(data_path, mode="r+", dtype=_WORKER_DTYPE, shape=shape)


def _write_memmap_record(task: tuple[int, list[str]]) -> int:
    if _WORKER_FRAMES is None or _WORKER_DTYPE is None:
        raise RuntimeError("Memmap worker was not initialized")
    run_idx, paths = task
    for frame_idx, path in enumerate(paths):
        _WORKER_FRAMES[run_idx, frame_idx] = read_ovf(path).astype(_WORKER_DTYPE, copy=False)
    return run_idx


def _write_memmap_record_into(
    frames: np.memmap,
    dtype: np.dtype,
    task: tuple[int, list[str]],
) -> tuple[int, int, list[str]]:
    run_idx, paths = task
    errors: list[str] = []
    for frame_idx, path in enumerate(paths):
        try:
            frames[run_idx, frame_idx] = read_ovf(path).astype(dtype, copy=False)
        except Exception as exc:  # noqa: BLE001 - bad input data must not kill the build.
            errors.append(_format_read_error(path, exc))
    # A missing frame breaks positional time indexing, so skip the whole run
    # for training instead of compacting later frames into earlier slots.
    return run_idx, 0 if errors else len(paths), errors


def _write_memmap_record_robust(task: tuple[int, list[str]]) -> tuple[int, int, list[str]]:
    """Write one run, returning errors instead of letting corrupt OVFs abort the pool."""
    if _WORKER_FRAMES is None or _WORKER_DTYPE is None:
        raise RuntimeError("Memmap worker was not initialized")
    return _write_memmap_record_into(_WORKER_FRAMES, _WORKER_DTYPE, task)


def _print_memmap_summary(
    *,
    total_runs: int,
    total_frames: int,
    valid_counts: list[int],
    errors_by_run: dict[int, list[str]],
) -> None:
    skipped_runs = sum(1 for n in valid_counts if n == 0)
    skipped_frames = sum(len(v) for v in errors_by_run.values())
    usable_runs = total_runs - skipped_runs
    usable_frames = sum(valid_counts)
    if errors_by_run:
        print("memmap warnings:", file=sys.stderr, flush=True)
        for run_idx in sorted(errors_by_run):
            print(f"  skipped run index {run_idx}:", file=sys.stderr, flush=True)
            for err in errors_by_run[run_idx]:
                print(f"    {err}", file=sys.stderr, flush=True)
    print(
        "memmap summary: "
        f"usable_runs={usable_runs}/{total_runs}, "
        f"usable_frames={usable_frames}/{total_frames}, "
        f"skipped_runs={skipped_runs}, "
        f"bad_ovf_files={skipped_frames}",
        file=sys.stderr,
        flush=True,
    )


def _first_readable_frame_shape(records: list[Any]) -> tuple[int, ...]:
    first_error: str | None = None
    for rec in records:
        for path in rec.frames:
            try:
                return read_ovf(path).shape
            except Exception as exc:  # noqa: BLE001 - keep scanning for a usable shape sample.
                if first_error is None:
                    first_error = _format_read_error(path, exc)
    detail = f"; first error: {first_error}" if first_error else ""
    raise ValueError(f"No readable OVF frames found for memmap shape inference{detail}")


def build_memmap(
    dataset_root: str | Path | list[str | Path] | tuple[str | Path, ...],
    out_dir: str | Path,
    frame_glob: str = "run.out/m*.ovf",
    dtype: str = "float16",
    include_before_drive: bool = False,
    workers: int | None = None,
) -> Path:
    from skyrmion_cfm.data.trajectory import TrajectoryIndex

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = TrajectoryIndex.from_root(dataset_root, frame_glob, include_before_drive)
    if not index.records:
        raise ValueError("No trajectory records found")
    n_runs = len(index.records)
    # plan-2 datasets can mix 25 ps / 250 ps save_step_ps and may live across
    # several generation folders; per-run n_frames is therefore non-uniform.
    # We pad shorter runs along the time axis to ``max(n_frames)`` and store
    # the true length per run so the loader never reads padding.
    per_run_n_frames = [r.n_frames for r in index.records]
    max_n_frames = max(per_run_n_frames)
    frame_shape = _first_readable_frame_shape(index.records)
    shape = (n_runs, max_n_frames, *frame_shape)
    np_dtype = np.dtype(dtype)
    data_file = f"frames.{np_dtype.name}.dat"
    data_path = out_dir / data_file
    frames = np.memmap(data_path, mode="w+", dtype=np_dtype, shape=shape)
    run_ids: list[str] = []
    tasks: list[tuple[int, list[str]]] = []
    for run_idx, rec in enumerate(index.records):
        run_ids.append(rec.run_id)
        tasks.append((run_idx, [str(p) for p in rec.frames]))

    workers = max(1, int(DEFAULT_MEMMAP_WORKERS if workers is None else workers))
    valid_n_frames = [0] * n_runs
    errors_by_run: dict[int, list[str]] = {}
    if workers == 1:
        for task in tqdm(tasks, desc="memmap"):
            run_idx, n_valid, errors = _write_memmap_record_into(frames, np_dtype, task)
            valid_n_frames[run_idx] = n_valid
            if errors:
                errors_by_run[run_idx] = errors
    else:
        del frames
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_memmap_worker,
            initargs=(str(data_path), shape, np_dtype.name),
        ) as executor:
            with tqdm(total=len(tasks), desc=f"memmap ({workers} workers)") as progress:
                for run_idx, n_valid, errors in executor.map(
                    _write_memmap_record_robust, tasks, chunksize=1
                ):
                    valid_n_frames[run_idx] = n_valid
                    if errors:
                        errors_by_run[run_idx] = errors
                    progress.update(1)
        frames = np.memmap(data_path, mode="r+", dtype=np_dtype, shape=shape)
    frames.flush()
    _print_memmap_summary(
        total_runs=n_runs,
        total_frames=sum(per_run_n_frames),
        valid_counts=valid_n_frames,
        errors_by_run=errors_by_run,
    )
    manifest = {
        "dataset_roots": _normalize_roots(dataset_root),
        "data_file": data_file,
        "dtype": np_dtype.name,
        "shape": list(shape),
        "run_ids": run_ids,
        "n_frames": max_n_frames,  # padded uniform length
        "runs_per_record_n_frames": valid_n_frames,  # 0 means skipped due to corrupt input.
        "frame_glob": frame_glob,
        "include_before_drive": include_before_drive,
    }
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def ensure_memmap(
    dataset_root: str | Path | list[str | Path] | tuple[str | Path, ...],
    out_dir: str | Path,
    frame_glob: str = "run.out/m*.ovf",
    dtype: str = "float16",
    include_before_drive: bool = False,
    auto_build: bool = False,
    force: bool = False,
    workers: int | None = None,
) -> Path:
    out_dir = Path(out_dir)
    path = manifest_path(out_dir)
    if path.exists() and not force:
        manifest = _load_manifest(path)
        if _manifest_matches(manifest, dataset_root, frame_glob, include_before_drive, dtype):
            return path
        if not auto_build:
            raise ValueError(
                f"Existing memmap manifest {path} does not match the configured dataset. "
                "Rebuild it with skyrmion-cfm-memmap --force or set data.memmap.auto_build=true."
            )
    if not auto_build and not force:
        raise FileNotFoundError(
            f"Memmap manifest {path} does not exist. Run skyrmion-cfm-memmap or set "
            "data.memmap.auto_build=true."
        )
    with _file_lock(out_dir / ".build.lock"):
        if path.exists() and not force:
            manifest = _load_manifest(path)
            if _manifest_matches(manifest, dataset_root, frame_glob, include_before_drive, dtype):
                return path
        return build_memmap(
            dataset_root,
            out_dir,
            frame_glob=frame_glob,
            dtype=dtype,
            include_before_drive=include_before_drive,
            workers=workers,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preconvert OVF trajectories to a contiguous memmap.")
    parser.add_argument("--config", default="skyrmion_cfm/configs/default.yaml")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of parallel worker processes for OVF conversion "
            f"(default: config, SKYRMION_MEMMAP_WORKERS, or {DEFAULT_MEMMAP_WORKERS})."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    memmap_cfg = data_cfg.get("memmap", {})
    out_dir = args.out_dir or memmap_cfg.get("path", "outputs/skyrmion_cfm/memmap")
    manifest = ensure_memmap(
        data_cfg["dataset_root"],
        out_dir,
        data_cfg.get("frame_glob", "run.out/m*.ovf"),
        dtype=args.dtype,
        include_before_drive=bool(data_cfg.get("include_before_drive", True)),
        auto_build=True,
        force=args.force,
        workers=(
            args.workers
            if args.workers is not None
            else int(os.environ["SKYRMION_MEMMAP_WORKERS"])
            if "SKYRMION_MEMMAP_WORKERS" in os.environ
            else memmap_cfg.get("workers")
        ),
    )
    print(manifest)


if __name__ == "__main__":
    main()
