"""Exact full-resolution local-patch translation distance."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F


ProgressCallback = Callable[[int, int, int], None]

FORMAL_RESOLUTION = 256
FORMAL_BLOCKS = 4
FORMAL_PATCH_EDGE_PX = FORMAL_RESOLUTION // FORMAL_BLOCKS
FORMAL_SHIFT_RADIUS_PX = FORMAL_PATCH_EDGE_PX // 4


def patch_grid(
    size: int = FORMAL_RESOLUTION,
    blocks: int = FORMAL_BLOCKS,
) -> list[tuple[int, int, int, int]]:
    """Return row-major square-patch bounds."""

    if size <= 0 or blocks <= 0 or size % blocks:
        raise ValueError(f"size={size} must be positive and divisible by blocks={blocks}")
    edge = size // blocks
    return [
        (row * edge, (row + 1) * edge, column * edge, (column + 1) * edge)
        for row in range(blocks)
        for column in range(blocks)
    ]


def _validate_inputs(
    reference: torch.Tensor,
    query: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[int, int]:
    if reference.ndim != 4 or query.ndim != 4:
        raise ValueError("reference and query must use NCHW layout")
    if reference.shape[1:] != query.shape[1:]:
        raise ValueError(f"incompatible field shapes: {reference.shape}, {query.shape}")
    if reference.shape[1] != 3:
        raise ValueError("the metric requires mx, my, and mz")
    height, width = reference.shape[-2:]
    if (height, width) != (256, 256):
        raise ValueError(f"the formal metric requires 256x256 fields, got {(height, width)}")
    if mask.shape != (1, 1, height, width):
        raise ValueError(f"expected mask shape (1,1,{height},{width}), got {mask.shape}")
    if reference.device != query.device or reference.device != mask.device:
        raise ValueError("reference, query, and mask must be on the same device")
    return height, width


@torch.inference_mode()
def patch_shift_distance(
    reference: torch.Tensor,
    query: torch.Tensor,
    mask: torch.Tensor,
    *,
    blocks: int = FORMAL_BLOCKS,
    shift_radius: int = FORMAL_SHIFT_RADIUS_PX,
    reference_chunk: int = 256,
    progress: ProgressCallback | None = None,
) -> tuple[np.ndarray, int]:
    """Return a symmetric ``query x reference`` distance matrix.

    Every source patch is correlated against all integer translations in the
    corresponding target search window. The two directed distances are
    averaged. There is no downsampling and no translation penalty.
    """

    height, width = _validate_inputs(reference, query, mask)
    if shift_radius < 0:
        raise ValueError("shift_radius must be non-negative")
    if reference_chunk <= 0:
        raise ValueError("reference_chunk must be positive")

    n_reference = reference.shape[0]
    n_query = query.shape[0]
    query_to_reference = torch.zeros(
        (n_query, n_reference), dtype=torch.float32, device=reference.device
    )
    reference_to_query = torch.zeros_like(query_to_reference)
    valid_patches = 0
    patches = patch_grid(height, blocks)

    for patch_index, (y0, y1, x0, x1) in enumerate(patches, start=1):
        source_count = float(mask[0, 0, y0:y1, x0:x1].sum().item())
        if source_count < 1.0:
            continue
        valid_patches += 1
        target_y0 = max(0, y0 - shift_radius)
        target_x0 = max(0, x0 - shift_radius)
        target_y1 = min(height, y1 + shift_radius)
        target_x1 = min(width, x1 + shift_radius)
        query_source = query[:, :, y0:y1, x0:x1].contiguous()
        query_search = query[:, :, target_y0:target_y1, target_x0:target_x1].contiguous()

        for start in range(0, n_reference, reference_chunk):
            end = min(n_reference, start + reference_chunk)
            reference_part = reference[start:end]
            reference_search = reference_part[
                :, :, target_y0:target_y1, target_x0:target_x1
            ].contiguous()
            forward_correlation = F.conv2d(reference_search, query_source)
            best_forward = forward_correlation.flatten(2).amax(dim=2).transpose(0, 1)
            best_forward = (best_forward / source_count).clamp(-1.0, 1.0)
            query_to_reference[:, start:end] += (1.0 - best_forward).clamp_min(0.0)

            reference_source = reference_part[:, :, y0:y1, x0:x1].contiguous()
            reverse_correlation = F.conv2d(query_search, reference_source)
            best_reverse = reverse_correlation.flatten(2).amax(dim=2)
            best_reverse = (best_reverse / source_count).clamp(-1.0, 1.0)
            reference_to_query[:, start:end] += (1.0 - best_reverse).clamp_min(0.0)

        if progress is not None:
            progress(patch_index, len(patches), valid_patches)

    if valid_patches == 0:
        raise RuntimeError("the geometry mask contains no magnetic patch")
    symmetric = 0.5 * (
        query_to_reference / valid_patches + reference_to_query / valid_patches
    )
    return symmetric.cpu().numpy(), valid_patches


@torch.inference_mode()
def paired_patch_shift_distance(
    reference: torch.Tensor,
    query: torch.Tensor,
    mask: torch.Tensor,
    *,
    blocks: int = FORMAL_BLOCKS,
    shift_radius: int = FORMAL_SHIFT_RADIUS_PX,
    reference_chunk: int = 256,
    progress: ProgressCallback | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact distances for independent candidate groups.

    ``reference`` has shape ``(B, R, 3, 256, 256)`` and ``query`` has shape
    ``(B, 3, 256, 256)``. Row ``b`` of the returned ``(B, R)`` matrix only
    compares ``query[b]`` with ``reference[b]``. This is mathematically the
    same calculation as calling :func:`patch_shift_distance` ``B`` times, but
    grouped convolutions avoid computing irrelevant cross-group pairs.

    A separate geometry mask is accepted for every group. The second return
    value contains the number of valid geometry patches for each group.
    """

    if reference.ndim != 5 or query.ndim != 4:
        raise ValueError(
            "reference and query must use BRCHW and BCHW layouts, respectively"
        )
    batch, n_reference, channels, height, width = reference.shape
    if batch <= 0 or n_reference <= 0:
        raise ValueError(f"reference must contain non-empty groups, got {reference.shape}")
    if query.shape != (batch, channels, height, width):
        raise ValueError(f"incompatible field shapes: {reference.shape}, {query.shape}")
    if channels != 3:
        raise ValueError("the metric requires mx, my, and mz")
    if (height, width) != (256, 256):
        raise ValueError(f"the formal metric requires 256x256 fields, got {(height, width)}")
    if mask.shape != (batch, 1, height, width):
        raise ValueError(
            f"expected mask shape ({batch},1,{height},{width}), got {mask.shape}"
        )
    if reference.device != query.device or reference.device != mask.device:
        raise ValueError("reference, query, and mask must be on the same device")
    if shift_radius < 0:
        raise ValueError("shift_radius must be non-negative")
    if reference_chunk <= 0:
        raise ValueError("reference_chunk must be positive")

    query_to_reference = torch.zeros(
        (batch, n_reference), dtype=torch.float32, device=reference.device
    )
    reference_to_query = torch.zeros_like(query_to_reference)
    valid_patches = torch.zeros(batch, dtype=torch.long, device=reference.device)
    patches = patch_grid(height, blocks)

    for patch_index, (y0, y1, x0, x1) in enumerate(patches, start=1):
        source_count = mask[:, :, y0:y1, x0:x1].sum(dim=(1, 2, 3))
        valid = source_count >= 1.0
        if not bool(valid.any()):
            continue
        valid_patches += valid.to(dtype=torch.long)
        source_count = source_count.clamp_min(1.0)
        valid_weight = valid.to(dtype=torch.float32)[:, None]
        target_y0 = max(0, y0 - shift_radius)
        target_x0 = max(0, x0 - shift_radius)
        target_y1 = min(height, y1 + shift_radius)
        target_x1 = min(width, x1 + shift_radius)
        query_source = query[:, :, y0:y1, x0:x1].contiguous()
        query_search = query[
            :, :, target_y0:target_y1, target_x0:target_x1
        ].contiguous()
        search_height = target_y1 - target_y0
        search_width = target_x1 - target_x0

        for start in range(0, n_reference, reference_chunk):
            end = min(n_reference, start + reference_chunk)
            chunk = end - start
            reference_part = reference[:, start:end]

            reference_search = reference_part[
                :, :, :, target_y0:target_y1, target_x0:target_x1
            ]
            reference_search = (
                reference_search.permute(1, 0, 2, 3, 4)
                .contiguous()
                .view(chunk, batch * channels, search_height, search_width)
            )
            forward_correlation = F.conv2d(
                reference_search,
                query_source,
                groups=batch,
            )
            best_forward = forward_correlation.flatten(2).amax(dim=2).transpose(0, 1)
            best_forward = (best_forward / source_count[:, None]).clamp(-1.0, 1.0)
            query_to_reference[:, start:end] += (
                (1.0 - best_forward).clamp_min(0.0) * valid_weight
            )

            reference_source = reference_part[:, :, :, y0:y1, x0:x1]
            reference_source = reference_source.contiguous().view(
                batch * chunk,
                channels,
                y1 - y0,
                x1 - x0,
            )
            reverse_correlation = F.conv2d(
                query_search.view(1, batch * channels, search_height, search_width),
                reference_source,
                groups=batch,
            )
            best_reverse = reverse_correlation.view(batch, chunk, -1).amax(dim=2)
            best_reverse = (best_reverse / source_count[:, None]).clamp(-1.0, 1.0)
            reference_to_query[:, start:end] += (
                (1.0 - best_reverse).clamp_min(0.0) * valid_weight
            )

        if progress is not None:
            progress(patch_index, len(patches), int(valid.sum().item()))

    empty = torch.nonzero(valid_patches == 0, as_tuple=False).flatten()
    if empty.numel():
        preview = ", ".join(str(int(index)) for index in empty[:8].cpu())
        raise RuntimeError(f"geometry mask is empty for group(s): {preview}")
    symmetric = 0.5 * (
        query_to_reference / valid_patches[:, None]
        + reference_to_query / valid_patches[:, None]
    )
    return symmetric.cpu().numpy(), valid_patches.cpu().numpy()


@torch.inference_mode()
def no_shift_distance(
    reference: torch.Tensor,
    query: torch.Tensor,
    mask: torch.Tensor,
) -> np.ndarray:
    """Return the global no-translation control distance."""

    _validate_inputs(reference, query, mask)
    pixel_count = float(mask.sum().item())
    dot_product = query.flatten(1) @ reference.flatten(1).T
    distance = 1.0 - (dot_product / max(pixel_count, 1.0)).clamp(-1.0, 1.0)
    return distance.clamp_min(0.0).float().cpu().numpy()
