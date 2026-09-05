from __future__ import annotations

import torch

from scripts.x5_anchor_jitter import apply_anchor_jitter


def _field() -> torch.Tensor:
    field = torch.zeros(2, 3, 16, 16)
    field[:, 2, 2:14, 3:13] = 1.0
    return field


def test_anchor_jitter_preserves_mask_norm_and_requested_rms() -> None:
    source = _field()
    result, reports = apply_anchor_jitter(
        source,
        rms_degrees=0.5,
        correlation_px=2.0,
        seeds=[11, 12],
    )
    valid = source.square().sum(dim=1).sqrt() > 0.5
    assert torch.allclose(result.square().sum(dim=1).sqrt()[valid], torch.ones(240))
    assert torch.count_nonzero(result.masked_select(~valid[:, None])) == 0
    assert all(abs(report["realized_rms_degrees"] - 0.5) < 1.0e-5 for report in reports)


def test_anchor_jitter_is_seed_replayable_and_non_degenerate() -> None:
    source = _field()[:1]
    first, _ = apply_anchor_jitter(
        source, rms_degrees=1.0, correlation_px=1.0, seeds=[91]
    )
    replay, _ = apply_anchor_jitter(
        source, rms_degrees=1.0, correlation_px=1.0, seeds=[91]
    )
    other, _ = apply_anchor_jitter(
        source, rms_degrees=1.0, correlation_px=1.0, seeds=[92]
    )
    assert torch.equal(first, replay)
    assert not torch.equal(first, other)
