from __future__ import annotations

import pytest
import torch
from torch import nn

from skyrmion_cfm.cfm.bridges import CartBridge, RotationVectorBridge
from skyrmion_cfm.cfm.loss import CFMLoss
from skyrmion_cfm.cfm.prior import CartSourcePrior, RotationPrior
from skyrmion_cfm.cfm.sampler import BridgeSampler
from skyrmion_cfm.data.log_map import rodrigues_chw


def _loss(tau_sampling: str) -> CFMLoss:
    return CFMLoss(
        bridge=CartBridge(source="identity", objective="residual", normalize_path=False),
        rotation_prior=RotationPrior(prior_type="standard_gaussian"),
        cart_prior=CartSourcePrior(mode="identity"),
        tau_sampling=tau_sampling,
    )


def test_zero_tau_sampling_is_direct_regression_state() -> None:
    loss = _loss("zero")
    tau = loss._sample_tau(4, torch.device("cpu"))
    assert torch.equal(tau, torch.zeros(4))

    m_init = torch.randn(4, 3, 2, 2)
    target_delta = torch.randn_like(m_init)
    bridge = loss.bridge.build(m_init, target_delta, tau)
    assert torch.equal(bridge.state, torch.zeros_like(target_delta))
    assert torch.equal(bridge.target_velocity, target_delta)


def test_zero_tau_alpha_identity_is_direct_rotation_regression() -> None:
    loss = CFMLoss(
        bridge=RotationVectorBridge(source="identity", objective="residual"),
        rotation_prior=RotationPrior(prior_type="standard_gaussian"),
        cart_prior=CartSourcePrior(mode="identity"),
        tau_sampling="zero",
    )
    tau = loss._sample_tau(3, torch.device("cpu"))
    m_init = torch.randn(3, 3, 2, 2)
    omega_target = torch.randn_like(m_init)
    source, _ = loss._sample_source(m_init, omega_target, {})
    bridge = loss.bridge.build(
        m_init,
        omega_target,
        tau,
        source_noise=source,
    )

    assert torch.equal(source, torch.zeros_like(omega_target))
    assert torch.equal(bridge.state, torch.zeros_like(omega_target))
    assert torch.equal(bridge.target_velocity, omega_target)


def test_alpha_identity_euler_one_is_one_deterministic_forward() -> None:
    generator = torch.Generator().manual_seed(23)
    m_init = torch.randn((2, 3, 4, 4), generator=generator)
    m_init = m_init / m_init.norm(dim=1, keepdim=True)
    omega = 0.1 * torch.randn(m_init.shape, generator=generator)

    class FixedRotation(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, m_init, state, tau, cond):
            del m_init, state, tau, cond
            self.calls += 1
            return omega

    model = FixedRotation()
    sampler = BridgeSampler(
        bridge=RotationVectorBridge(source="identity", objective="residual"),
        rotation_prior=RotationPrior(prior_type="standard_gaussian"),
        ode_steps=1,
        method="euler",
    )
    prediction, state = sampler.sample(model, m_init, {})

    assert model.calls == 1
    torch.testing.assert_close(state, omega)
    torch.testing.assert_close(prediction, rodrigues_chw(omega, m_init))


def test_unknown_tau_sampling_fails_loudly() -> None:
    with pytest.raises(ValueError, match="tau_sampling"):
        _loss("typo")._sample_tau(1, torch.device("cpu"))


@pytest.mark.parametrize("objective", ["endpoint", "residual"])
def test_raw_cart_auxiliary_endpoint_uses_source_plus_velocity(
    objective: str,
) -> None:
    generator = torch.Generator().manual_seed(17)
    m_init = torch.randn((2, 3, 8, 8), generator=generator)
    m_init = m_init / m_init.norm(dim=1, keepdim=True)
    m_target = torch.randn((2, 3, 8, 8), generator=generator)
    m_target = m_target / m_target.norm(dim=1, keepdim=True)

    class ExactEndpointVelocity(nn.Module):
        def forward(self, m_init, state, tau, cond):
            del state, tau, cond
            return m_target.to(m_init) - m_init

    loss = CFMLoss(
        bridge=CartBridge(
            source="identity",
            objective=objective,
            normalize_path=False,
        ),
        rotation_prior=RotationPrior(prior_type="standard_gaussian"),
        cart_prior=CartSourcePrior(mode="identity"),
        unit_weight=1.0e-3,
        topo_weight=2.0e-2,
        tau_sampling="zero" if objective == "residual" else "uniform",
    )
    result = loss(
        ExactEndpointVelocity(),
        {"m_init": m_init, "m_t": m_target},
        {"t_end_s": torch.ones(2), "temp_k": torch.ones(2)},
    )
    torch.testing.assert_close(result.cfm, torch.tensor(0.0), atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(result.unit, torch.tensor(0.0), atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(result.topo, torch.tensor(0.0), atol=1.0e-6, rtol=0.0)
