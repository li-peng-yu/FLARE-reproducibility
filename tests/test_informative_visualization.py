from __future__ import annotations

import torch

from skyrmion_cfm.eval.informative_visualization import (
    add_informative_candidates,
    evenly_spaced_bucket_indices,
    informative_roi_bounds,
    select_informative_candidates,
)


def _uniform_spin(batch: int, size: int = 16) -> torch.Tensor:
    out = torch.zeros((batch, 3, size, size), dtype=torch.float32)
    out[:, 2] = -1.0
    return out


def test_bucket_selection_spans_full_time_range() -> None:
    assert evenly_spaced_bucket_indices(12, 4) == {0, 4, 7, 11}
    assert evenly_spaced_bucket_indices(4, 12) == {0, 1, 2, 3}


def test_informative_selection_prefers_structured_samples_per_bucket() -> None:
    initial = _uniform_spin(4)
    target = initial.clone()
    # Bucket 0: one uniform sample and one localized object.
    target[1, 2, 7:9, 7:9] = 1.0
    # Bucket 1: one uniform sample and one extended domain pattern.
    target[3, 2, :, 8:] = 1.0
    pred = target.clone()
    batch = {
        "t_end_index": torch.tensor([0, 0, 1, 1]),
        "t_end_ns": torch.tensor([0.25, 0.25, 100.0, 100.0]),
        "frame_init": torch.tensor([0, 0, 0, 0]),
        "frame_target": torch.tensor([1, 1, 4, 4]),
        "run_id": ["uniform-0", "localized", "uniform-1", "extended"],
    }
    zeros = torch.zeros(4)
    groups: dict[tuple[int, str], list[dict]] = {}

    add_informative_candidates(
        groups,
        m_init=initial,
        pred=pred,
        target=target,
        batch=batch,
        mse_each=zeros,
        ang_each=zeros,
        q_pred_each=zeros,
        q_target_each=zeros,
    )
    selected = select_informative_candidates(groups, 2, target_buckets={0, 1})

    assert [item["run_id"] for item in selected] == ["localized", "extended"]


def test_roi_focuses_on_localized_structure() -> None:
    initial = _uniform_spin(1, size=64)[0]
    target = initial.clone()
    target[2, 29:35, 30:36] = 1.0

    y0, y1, x0, x1 = informative_roi_bounds(
        initial,
        target,
        min_side=24,
        padding=4,
    )

    assert y1 - y0 == 24
    assert x1 - x0 == 24
    assert y0 <= 29 < 35 <= y1
    assert x0 <= 30 < 36 <= x1


def test_pixel_noise_is_demoted_below_coherent_structure() -> None:
    size = 64
    initial = _uniform_spin(2, size=size)
    target = initial.clone()
    generator = torch.Generator().manual_seed(7)
    target[0, 2] = torch.where(
        torch.rand((size, size), generator=generator) > 0.5,
        1.0,
        -1.0,
    )
    target[1, 2, :, size // 2 :] = 1.0
    batch = {
        "t_end_index": torch.tensor([0, 0]),
        "t_end_ns": torch.tensor([10.0, 10.0]),
        "frame_init": torch.tensor([0, 0]),
        "frame_target": torch.tensor([1, 1]),
        "run_id": ["pixel-noise", "domain-wall"],
    }
    zeros = torch.zeros(2)
    groups: dict[tuple[int, str], list[dict]] = {}

    add_informative_candidates(
        groups,
        m_init=initial,
        pred=target,
        target=target,
        batch=batch,
        mse_each=zeros,
        ang_each=zeros,
        q_pred_each=zeros,
        q_target_each=zeros,
    )
    selected = select_informative_candidates(groups, 1, target_buckets={0})

    assert groups[(0, "noisy")][0]["run_id"] == "pixel-noise"
    assert selected[0]["run_id"] == "domain-wall"
