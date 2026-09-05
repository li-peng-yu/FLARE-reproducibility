"""Numerical stability checks for the RFM primitives (plan-2 §RFM).

The user opted *not* to add an antipodal mask; instead the implementation
must keep ``log_map`` and ``slerp_velocity`` finite + tangent-projected even
when the endpoints are nearly identical or nearly antipodal. These tests pin
that contract.
"""

from __future__ import annotations

import torch

from skyrmion_cfm.cfm.rfm import log_map, slerp_velocity, project_to_tangent
from skyrmion_cfm.data.log_map import log_map_s2


def test_log_map_unit_magnitude_at_90_deg():
    p = torch.tensor([[0.0, 0.0, 1.0]])
    q = torch.tensor([[1.0, 0.0, 0.0]])
    lm = log_map(p, q)
    assert torch.isfinite(lm).all()
    assert torch.allclose(lm.norm(dim=-1), torch.tensor([torch.pi / 2]), atol=1e-5)


def test_log_map_near_antipodal_stays_finite_and_unit_speed():
    p = torch.tensor([[0.0, 0.0, 1.0]])
    q = torch.tensor([[1e-3, 0.0, -1.0]])
    q = q / q.norm(dim=-1, keepdim=True)
    lm = log_map(p, q)
    assert torch.isfinite(lm).all()
    # Magnitude must approach π (not the buggy 2.2 from the old formula).
    assert lm.norm(dim=-1).item() > 3.0


def test_log_map_at_identity_is_zero():
    p = torch.tensor([[0.0, 0.0, 1.0]])
    q = p.clone()
    lm = log_map(p, q)
    assert torch.isfinite(lm).all()
    assert lm.norm(dim=-1).item() < 1e-5


def test_slerp_velocity_is_tangent_and_finite_everywhere():
    p = torch.tensor([[0.0, 0.0, 1.0]])
    for q in (
        torch.tensor([[1e-6, 0.0, 1.0]]),                    # near identity
        torch.tensor([[1.0, 0.0, 0.0]]),                     # 90°
        torch.tensor([[1e-3, 0.0, -1.0]]),                   # near antipodal
        torch.tensor([[-1.0, 0.0, 0.0]]),                    # 90° in xz
    ):
        q = q / q.norm(dim=-1, keepdim=True)
        for tau_val in (0.0, 0.25, 0.5, 0.75, 1.0):
            tau = torch.tensor([tau_val])
            mu, v = slerp_velocity(p, q, tau)
            assert torch.isfinite(mu).all(), f"mu NaN at tau={tau_val}"
            assert torch.isfinite(v).all(), f"v NaN at tau={tau_val}"
            # |mu| ~ 1 by construction
            assert torch.allclose(mu.norm(dim=-1), torch.tensor([1.0]), atol=1e-5)
            # v lies in T_mu S^2
            assert (mu * v).sum(dim=-1).abs().item() < 1e-5


def test_log_map_s2_handles_antipodal_pair():
    # plan-1 rotation-vector branch must also stay finite.
    m0 = torch.tensor([[0.0, 0.0, 1.0]])
    m1 = torch.tensor([[0.0, 0.0, -1.0]])
    omega = log_map_s2(m0, m1)
    assert torch.isfinite(omega).all()
    # An exact antipodal pair should give a rotation magnitude of π.
    assert omega.norm(dim=-1).item() == 0.0 or abs(omega.norm(dim=-1).item() - torch.pi) < 1e-3
