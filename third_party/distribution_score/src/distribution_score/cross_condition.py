"""CLI for one-truth-per-condition probability-rank evaluation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .distance import FORMAL_BLOCKS, FORMAL_SHIFT_RADIUS_PX, patch_shift_distance
from .fields import normalize_batch, normalize_field, read_ovf
from .manifest import ConditionSpec, load_manifest, result_directory
from .statistics import probability_rank_from_distances
from .version import PACKAGE_VERSION, algorithm_version, distance_identifier


def _progress(device: torch.device):
    def report(index: int, total: int, valid: int) -> None:
        if index % 8 != 0 and index != total:
            return
        memory = (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else 0.0
        )
        print(f"patch={index:02d}/{total} valid={valid} memory_GB={memory:.2f}", flush=True)

    return report


@torch.inference_mode()
def evaluate_condition(
    spec: ConditionSpec,
    *,
    output_root: Path | None,
    num_model: int,
    reference_count: int,
    blocks: int,
    shift_radius: int,
    reference_chunk: int,
    seed: int,
    device: torch.device,
    progress: bool,
) -> dict[str, object]:
    output_dir = result_directory(
        spec,
        output_root,
        blocks=blocks,
        shift_radius=shift_radius,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "probability_rank.json"
    ranking_count = num_model - reference_count
    distance_id = distance_identifier(blocks, shift_radius)
    expected = {
        "package_version": PACKAGE_VERSION,
        "algorithm_version": algorithm_version(blocks, shift_radius),
        "distance_id": distance_id,
        "num_model_samples": num_model,
        "reference_samples": reference_count,
        "ranking_samples": ranking_count,
        "blocks": [blocks, blocks],
        "shift_radius_each_axis_px": shift_radius,
        "split_seed": seed,
    }
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and all(
            existing.get(key) == value for key, value in expected.items()
        ):
            return {**existing, "status": "already_complete"}

    model_memmap = np.load(spec.model_samples, mmap_mode="r", allow_pickle=False)
    if model_memmap.ndim != 4 or model_memmap.shape[1:] != (3, 256, 256):
        raise ValueError(f"unexpected model sample shape: {model_memmap.shape}")
    if model_memmap.shape[0] < num_model:
        raise ValueError(f"requested {num_model} model samples, found {model_memmap.shape[0]}")
    anchor = normalize_field(read_ovf(spec.anchor))
    truth = normalize_field(read_ovf(spec.truth))
    geometry_mask = np.linalg.norm(anchor, axis=0) > 0.5
    if not geometry_mask.any():
        raise RuntimeError("empty geometry mask")

    model = normalize_batch(np.asarray(model_memmap[:num_model], dtype=np.float32))
    model *= geometry_mask[None, None]
    truth *= geometry_mask[None]
    permutation = np.random.default_rng(seed).permutation(num_model)
    reference_indices = permutation[:reference_count]
    ranking_indices = permutation[reference_count:]
    reference_np = np.ascontiguousarray(model[reference_indices])
    ranking_np = np.ascontiguousarray(model[ranking_indices])
    query_np = np.concatenate([ranking_np, truth[None]], axis=0)

    reference = torch.from_numpy(reference_np).to(device=device, dtype=torch.float32)
    query = torch.from_numpy(query_np).to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(geometry_mask.astype(np.float32))[None, None].to(device)
    callback = _progress(device) if progress else None
    started = time.time()
    reference_self_distance, valid_patches = patch_shift_distance(
        reference,
        reference,
        mask,
        blocks=blocks,
        shift_radius=shift_radius,
        reference_chunk=reference_chunk,
        progress=callback,
    )
    query_distance, _ = patch_shift_distance(
        reference,
        query,
        mask,
        blocks=blocks,
        shift_radius=shift_radius,
        reference_chunk=reference_chunk,
        progress=callback,
    )
    rank_summary = probability_rank_from_distances(
        reference_self_distance,
        query_distance,
    )
    ranking_density = np.asarray(rank_summary["ranking_density"])

    np.save(output_dir / "reference_indices.npy", reference_indices.astype(np.int32))
    np.save(output_dir / "ranking_indices.npy", ranking_indices.astype(np.int32))
    np.save(
        output_dir / "reference_self_distance_f16.npy",
        reference_self_distance.astype(np.float16),
    )
    np.save(
        output_dir / "ranking_and_truth_to_reference_distance_f16.npy",
        query_distance.astype(np.float16),
    )
    np.save(output_dir / "ranking_density.npy", ranking_density)
    np.save(output_dir / "geometry_mask.npy", geometry_mask)

    result: dict[str, object] = {
        "status": "complete",
        "package_version": PACKAGE_VERSION,
        "algorithm_version": algorithm_version(blocks, shift_radius),
        "distance_id": distance_id,
        "condition_id": spec.condition_id,
        "statistical_group": spec.group,
        "statistical_group_label": spec.group_label,
        "num_model_samples": num_model,
        "reference_samples": reference_count,
        "ranking_samples": ranking_count,
        "split_seed": seed,
        "blocks": [blocks, blocks],
        "patch_edge_px": 256 // blocks,
        "shift_radius_each_axis_px": shift_radius,
        "integer_shifts": True,
        "downsampling": False,
        "shift_penalty": 0.0,
        "symmetric_distance": True,
        "valid_geometry_patches": valid_patches,
        "kernel_sigma": float(rank_summary["sigma"]),
        "truth_model_density": float(rank_summary["truth_density"]),
        "ranking_density_min": float(ranking_density.min()),
        "ranking_density_median": float(np.median(ranking_density)),
        "ranking_density_max": float(ranking_density.max()),
        "probability_rank_u": float(rank_summary["probability_rank_u"]),
        "rank_formula": "(1 + count(q_B > q_truth)) / (len(B) + 1)",
        "elapsed_seconds": time.time() - started,
        "max_cuda_memory_gb": (
            torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
        ),
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute one probability rank for each distinct condition."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--condition-id", action="append", default=None)
    parser.add_argument("--num-model", type=int, default=500)
    parser.add_argument("--reference-count", type=int, default=250)
    parser.add_argument("--blocks", type=int, default=FORMAL_BLOCKS)
    parser.add_argument(
        "--shift-radius", type=int, default=FORMAL_SHIFT_RADIUS_PX
    )
    parser.add_argument("--reference-chunk", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress", action="store_true")
    return parser


@torch.inference_mode()
def run(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.reference_count <= 1 or args.reference_count >= args.num_model:
        raise ValueError("reference-count must lie strictly between 1 and num-model")
    manifest = load_manifest(args.manifest)
    selected_ids = set(args.condition_id or [])
    if selected_ids:
        known = {spec.condition_id for spec in manifest.conditions}
        unknown = selected_ids - known
        if unknown:
            raise ValueError(f"unknown condition ids: {sorted(unknown)}")
    selected = [
        spec for spec in manifest.conditions
        if not selected_ids or spec.condition_id in selected_ids
    ]
    manifest_position = {
        spec.condition_id: position for position, spec in enumerate(manifest.conditions)
    }

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for position, spec in enumerate(selected):
        try:
            result = evaluate_condition(
                spec,
                output_root=args.output_root,
                num_model=args.num_model,
                reference_count=args.reference_count,
                blocks=args.blocks,
                shift_radius=args.shift_radius,
                reference_chunk=args.reference_chunk,
                seed=args.seed + manifest_position[spec.condition_id] * 10000,
                device=device,
                progress=args.progress,
            )
            results.append(result)
            message = {
                "position": position + 1,
                "total": len(selected),
                "condition_id": spec.condition_id,
                "status": result["status"],
                "probability_rank_u": result["probability_rank_u"],
            }
        except Exception as error:
            failures.append({"condition_id": spec.condition_id, "error": repr(error)})
            message = {
                "position": position + 1,
                "total": len(selected),
                "condition_id": spec.condition_id,
                "status": "failed",
                "error": repr(error),
            }
        print(json.dumps(message, sort_keys=True), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if failures:
        failure_parent = args.output_root or manifest.path.parent
        failure_parent.mkdir(parents=True, exist_ok=True)
        failure_path = failure_parent / "cross_condition_failures.json"
        failure_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"{len(failures)} conditions failed; see {failure_path}")
    return results


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
