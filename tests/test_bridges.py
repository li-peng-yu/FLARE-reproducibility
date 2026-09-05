"""Sanity tests for the plan-2 bridges.

We don't need a GPU here — small lattices and CPU-only forward passes verify
the algebraic identities promised by plan-2.
"""

from __future__ import annotations

import math

import torch

from skyrmion_cfm.cfm.bridges import (
    CartBridge,
    RFMBridge,
    RotationVectorBridge,
    make_bridge,
    project_velocity_to_state,
)
from skyrmion_cfm.cfm.prior import CartSourcePrior, RotationPrior
from skyrmion_cfm.cfm.rfm import (
    exp_map_chw,
    log_map_chw,
    project_to_tangent_chw,
    slerp_velocity_chw,
)
from skyrmion_cfm.cfm.sampler import BridgeSampler
from skyrmion_cfm.data.log_map import normalize_spin
from skyrmion_cfm.data.log_map import log_map_chw as rot_log_map_chw
from skyrmion_cfm.data.log_map import rodrigues_chw


def _random_unit_field(b: int = 2, h: int = 4, w: int = 4, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    x = torch.randn(b, 3, h, w)
    return normalize_spin(x.movedim(1, -1)).movedim(-1, 1)


# ---------------------------------------------------------------------------
# RFM geometry primitives
# ---------------------------------------------------------------------------

def test_log_then_exp_is_identity():
    p = _random_unit_field(seed=1)
    q = _random_unit_field(seed=2)
    v = log_map_chw(p, q)
    # log lives in the tangent: v · p ≈ 0 per site.
    dot = (v * p).sum(dim=1)
    assert dot.abs().max().item() < 1e-5
    q_back = exp_map_chw(p, v)
    # exp(log(q)) should reproduce q.
    err = (q_back - q).abs().max().item()
    assert err < 1e-5, err


def test_rotation_log_handles_antipodal_spins():
    p = _random_unit_field(seed=11)
    q = -p
    omega = rot_log_map_chw(p, q)
    theta = omega.norm(dim=1)
    assert (theta - torch.pi).abs().max().item() < 1e-4
    q_back = rodrigues_chw(omega, p)
    err = (q_back - q).abs().max().item()
    assert err < 1e-5, err


def test_slerp_velocity_matches_finite_difference():
    p = _random_unit_field(seed=3)
    q = _random_unit_field(seed=4)
    tau = torch.tensor([0.5, 0.7])
    mu, v = slerp_velocity_chw(p, q, tau)
    # Finite-difference check: ddot(τ) μ ≈ v
    dtau = 1.0e-4
    tau_plus = tau + dtau
    tau_minus = (tau - dtau).clamp_min(0.0)
    mu_p, _ = slerp_velocity_chw(p, q, tau_plus)
    mu_m, _ = slerp_velocity_chw(p, q, tau_minus)
    v_fd = (mu_p - mu_m) / (2 * dtau)
    err = (v - v_fd).abs().max().item()
    # Tangent-space tolerance — the analytic ‖θâ × μ‖ matches FD to 5 sig figs.
    assert err < 1e-3, err


def test_cart_bridge_target_velocity_is_tangent():
    bridge = CartBridge(source="anchored")
    p = _random_unit_field(seed=5)
    q = _random_unit_field(seed=6)
    tau = torch.tensor([0.5, 0.3])
    noise = 0.05 * torch.randn_like(p)
    out = bridge.build(p, q, tau, source_noise=noise)
    # state must be unit-norm.
    state_norm = out.state.norm(dim=1)
    assert (state_norm - 1).abs().max().item() < 1e-4
    # target velocity must be tangent to state.
    dot = (out.target_velocity * out.state).sum(dim=1)
    assert dot.abs().max().item() < 1e-5
    # The default Cart target is the exact derivative of normalize(z_tau).
    x0 = p + noise
    tau_b = tau.reshape(-1, 1, 1, 1)
    z = (1.0 - tau_b) * x0 + tau_b * q
    norm = z.norm(dim=1, keepdim=True)
    diff = q - x0
    tangent_diff = diff - (diff * out.state).sum(dim=1, keepdim=True) * out.state
    expected = tangent_diff / norm.clamp_min(bridge.min_norm_z)
    err = (out.target_velocity - expected).abs().max().item()
    assert err < 1e-6, err


def test_cart_canonical_affine_bridge_matches_conditional_flow_matching():
    bridge = CartBridge(source="latent", normalize_path=False)
    p = _random_unit_field(seed=15)
    q = _random_unit_field(seed=16)
    tau = torch.tensor([0.2, 0.8])
    x0 = torch.randn_like(p)
    out = bridge.build(p, q, tau, source_noise=x0)

    tau_b = tau.reshape(-1, 1, 1, 1)
    torch.testing.assert_close(out.state, (1.0 - tau_b) * x0 + tau_b * q)
    torch.testing.assert_close(out.target_velocity, q - x0)

    # A raw affine Cartesian field must not be projected onto a tangent plane.
    raw = torch.randn_like(out.state)
    torch.testing.assert_close(project_velocity_to_state(bridge, out.state, raw), raw)


def test_cart_canonical_affine_sampler_integrates_in_ambient_space():
    bridge = CartBridge(source="latent", normalize_path=False)
    sampler = BridgeSampler(
        bridge=bridge,
        rotation_prior=RotationPrior("standard_gaussian"),
        cart_prior=CartSourcePrior("latent"),
        ode_steps=4,
        method="heun",
    )
    m_init = _random_unit_field(seed=17)
    x0 = torch.randn_like(m_init)
    x1 = _random_unit_field(seed=18)
    velocity = x1 - x0

    class ConstantVelocity(torch.nn.Module):
        def forward(self, m_init, state, tau, cond):
            del m_init, state, tau, cond
            return velocity

    state, _ = sampler.sample_state(ConstantVelocity(), m_init, {}, state0=x0)
    torch.testing.assert_close(state, x1, rtol=0.0, atol=2.0e-7)
    physical, _ = sampler.sample(ConstantVelocity(), m_init, {}, state0=x0)
    torch.testing.assert_close(physical, x1, rtol=0.0, atol=2.0e-7)


def test_rfm_bridge_target_velocity_is_tangent():
    bridge = RFMBridge()
    p = _random_unit_field(seed=7)
    q = _random_unit_field(seed=8)
    tau = torch.tensor([0.6, 0.2])
    out = bridge.build(p, q, tau, source_noise=None)
    state_norm = out.state.norm(dim=1)
    assert (state_norm - 1).abs().max().item() < 1e-4
    dot = (out.target_velocity * out.state).sum(dim=1)
    assert dot.abs().max().item() < 1e-4


def test_alpha_bridge_velocity_is_endpoint_diff():
    bridge = RotationVectorBridge()
    p = _random_unit_field(seed=9)
    omega_target = 0.05 * torch.randn_like(p)
    tau = torch.tensor([0.3, 0.5])
    omega0 = 0.05 * torch.randn_like(p)
    out = bridge.build(p, omega_target, tau, source_noise=omega0)
    expected = omega_target - omega0
    err = (out.target_velocity - expected).abs().max().item()
    assert err < 1e-6, err


def test_make_bridge_decouples_target_repr_from_source_mode():
    alpha = make_bridge({"target_repr": "alpha", "source": "anchored"})
    assert isinstance(alpha, RotationVectorBridge)
    assert alpha.source == "anchored"

    alpha_override = make_bridge(
        {
            "target_repr": "alpha",
            "source": "anchored",
            "alpha": {"source": "identity"},
        }
    )
    assert alpha_override.source == "identity"

    cart = make_bridge({"target_repr": "cart", "source": "latent"})
    assert isinstance(cart, CartBridge)
    assert cart.source == "latent"
    assert cart.normalize_path is True

    cart_affine = make_bridge(
        {
            "state_repr": "cart",
            "source_mode": "latent",
            "objective": "endpoint",
            "cart": {"normalize_path": False},
        }
    )
    assert isinstance(cart_affine, CartBridge)
    assert cart_affine.normalize_path is False

    rfm = make_bridge({"target_repr": "rfm", "source": "identity"})
    assert isinstance(rfm, RFMBridge)
    assert rfm.source == "identity"


def test_project_to_tangent_idempotent():
    p = _random_unit_field(seed=10)
    v = torch.randn_like(p)
    proj = project_to_tangent_chw(p, v)
    proj2 = project_to_tangent_chw(p, proj)
    err = (proj - proj2).abs().max().item()
    assert err < 1e-6, err
