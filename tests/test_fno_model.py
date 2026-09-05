from __future__ import annotations

import torch

from skyrmion_cfm.models import build_model


def _config() -> dict:
    return {
        "data": {"t_end_ns": [1.0, 2.0], "boundary": "open"},
        "bridge": {"state_repr": "cart"},
        "model": {
            "arch": "fno",
            "input_repr": "cartesian",
            "cond_dim": 512,
            "use_spatial_cond": True,
            "spatial_cond_fields": ["j_z_field"],
            "spatial_cond_scales": {"j_z_field": 1.0e11},
            "fno": {"width": 8, "depth": 2, "modes": 4, "padding": 2},
        },
    }


def test_fno_velocity_shape_and_gradient() -> None:
    model = build_model(_config())
    batch, size = 2, 16
    m_init = torch.randn(batch, 3, size, size)
    state = torch.randn(batch, 3, size, size)
    tau = torch.rand(batch)
    cond = {
        "t_end_index": torch.tensor([0, 1]),
        "t_end_s": torch.tensor([1.0e-9, 2.0e-9]),
        "temp_k": torch.tensor([30.0, 150.0]),
        "b_t": torch.zeros(batch, 3),
        "current_a_m2": torch.zeros(batch),
        "j_z_field": torch.zeros(batch, 1, size, size),
    }
    output = model(m_init, state, tau, cond)
    assert output.shape == state.shape
    output.square().mean().backward()
    assert model.input_projection.weight.grad is not None
    assert model.blocks[0].spectral.weight_positive.grad is not None
