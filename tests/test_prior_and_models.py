import copy
import warnings

import torch

from skyrmion_cfm.cfm.loss import CFMLoss
from skyrmion_cfm.cfm.prior import RotationPrior
from skyrmion_cfm.data.conditions import ConditionStats
from skyrmion_cfm.data.stats import fit_physics_kappa
from skyrmion_cfm.models import build_model
from skyrmion_cfm.models.common import build_spatial_condition_image
from skyrmion_cfm.train import _adapt_resume_state_to_model


def _cfg(arch: str):
    return {
        "data": {"lattice_size": 16, "dt_scales": [1, 2, 5]},
        "model": {
            "arch": arch,
            "input_repr": "cartesian",
            "cond_dim": 512,
            "mlp": {"hidden": 32, "depth": 1, "bottleneck": 32, "spatial_size": 8},
            "unet": {
                "base_channels": 4,
                "channel_mults": [1, 2],
                "num_res_blocks": 1,
                "attention": False,
                "bottleneck_downsample": False,
                "groups": 4,
            },
            "dit": {"patch_size": 4, "depth": 1, "hidden": 32, "heads": 4, "mlp_ratio": 2.0},
        },
    }


def _batch():
    m0 = torch.randn(2, 3, 16, 16)
    m0 = m0 / m0.norm(dim=1, keepdim=True).clamp_min(1e-8)
    omega_target = 0.01 * torch.randn_like(m0)
    return {
        "m0": m0,
        "m1": m0,
        "omega_target": omega_target,
        "dt_scale": torch.tensor([1, 5]),
        "dt_index": torch.tensor([0, 2]),
        "dt_s": torch.tensor([5e-12, 25e-12]),
        "temp_k": torch.tensor([100.0, 200.0]),
        "b_t": torch.zeros(2, 3),
        "current_a_m2": torch.zeros(2),
    }


def _cond(batch):
    return {
        "dt_index": batch["dt_index"],
        "dt_s": batch["dt_s"],
        "temp_k": batch["temp_k"],
        "b_t": batch["b_t"],
        "current_a_m2": batch["current_a_m2"],
    }


def test_spatial_condition_image_scales_j_field_only():
    ref = torch.zeros(2, 3, 4, 4)
    cond = {
        "defect_field": torch.ones(2, 1, 4, 4),
        "j_field": torch.full((2, 1, 4, 4), 2.0e12),
    }
    spatial = build_spatial_condition_image(cond, ref, j_field_scale=1.0e12)

    assert spatial.shape == (2, 2, 4, 4)
    assert torch.allclose(spatial[:, 0], torch.ones(2, 4, 4))
    assert torch.allclose(spatial[:, 1], torch.full((2, 4, 4), 2.0))

    missing = build_spatial_condition_image({}, ref, j_field_scale=1.0e12)
    assert missing.shape == (2, 2, 4, 4)
    assert torch.allclose(missing, torch.zeros_like(missing))


def test_empirical_prior_broadcasts_per_scale_and_component():
    prior = RotationPrior(
        "empirical_scaled",
        kappa=2.0,
        dt_scales=[1, 2, 5],
        omega_std=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
    )
    batch = _batch()
    sigma = prior.sigma(_cond(batch), batch["omega_target"])
    assert sigma.shape == (2, 3, 1, 1)
    assert torch.allclose(sigma[0, :, 0, 0], torch.tensor([2.0, 4.0, 6.0]))
    assert torch.allclose(sigma[1, :, 0, 0], torch.tensor([14.0, 16.0, 18.0]))


def test_physics_kappa_fit_matches_sqrt_dt_temperature_scale():
    true_kappa = 3.0
    pattern = torch.tensor(
        [[[-1.0, 1.0], [-1.0, 1.0]], [[1.0, -1.0], [1.0, -1.0]], [[-1.0, -1.0], [1.0, 1.0]]]
    )
    samples = []
    for dt_s, temp_k in [(2.0, 8.0), (5.0, 20.0), (7.0, 11.0)]:
        sigma = true_kappa * (dt_s * temp_k) ** 0.5
        samples.append(
            {
                "dt_s": torch.tensor(dt_s),
                "temp_k": torch.tensor(temp_k),
                "omega_target": pattern * sigma,
            }
        )
    kappa, count = fit_physics_kappa(samples, max_samples=10)
    assert count == len(samples)
    assert kappa is not None
    assert abs(kappa - true_kappa) < 1e-6


def test_all_architectures_forward_and_loss():
    stats = ConditionStats(torch.zeros(6), torch.ones(6))
    batch = _batch()
    cond = _cond(batch)
    prior = RotationPrior("standard_gaussian")
    for arch in ["mlp", "unet", "dit"]:
        model = build_model(_cfg(arch), stats)
        out = model(batch["m0"], batch["omega_target"], torch.rand(2), cond)
        assert out.shape == batch["omega_target"].shape
        loss = CFMLoss(prior)(model, batch, cond).total
        assert torch.isfinite(loss)


def test_unet_independent_time_conditioning_is_zero_initialized_and_checkpoint_compatible():
    stats = ConditionStats(torch.zeros(6), torch.ones(6))
    base_cfg = _cfg("unet")
    time_cfg = copy.deepcopy(base_cfg)
    time_cfg["model"]["time_conditioning"] = {
        "enabled": True,
        "embedding_dim": 32,
        "bucket_dim": 16,
        "fourier_dim": 16,
        "hidden_dim": 32,
        "scale": 1.0,
    }

    torch.manual_seed(123)
    base_model = build_model(base_cfg, stats).eval()
    torch.manual_seed(456)
    time_model = build_model(time_cfg, stats).eval()
    missing, unexpected = time_model.load_state_dict(base_model.state_dict(), strict=False)

    assert not unexpected
    assert missing
    assert all("time_conditioner" in key or "time_mod" in key for key in missing)
    for name, parameter in time_model.named_parameters():
        if "time_mod" in name and (name.endswith("1.weight") or name.endswith("1.bias")):
            assert torch.count_nonzero(parameter) == 0

    batch = _batch()
    cond = _cond(batch)
    tau = torch.tensor([0.25, 0.75])
    with torch.no_grad():
        base_out = base_model(batch["m0"], batch["omega_target"], tau, cond)
        time_out = time_model(batch["m0"], batch["omega_target"], tau, cond)
    torch.testing.assert_close(time_out, base_out, rtol=0.0, atol=0.0)


def test_unet_independent_time_modulation_receives_gradient():
    stats = ConditionStats(torch.zeros(6), torch.ones(6))
    cfg = _cfg("unet")
    cfg["model"]["time_conditioning"] = {
        "enabled": True,
        "embedding_dim": 32,
        "bucket_dim": 16,
        "fourier_dim": 16,
        "hidden_dim": 32,
    }
    model = build_model(cfg, stats)
    batch = _batch()
    output = model(
        batch["m0"],
        batch["omega_target"],
        torch.tensor([0.25, 0.75]),
        _cond(batch),
    )
    output.square().mean().backward()

    gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "time_mod" in name and name.endswith("1.weight")
    ]
    assert gradients
    assert any(gradient is not None and torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_unet_mask_target_time_keeps_lookup_in_backward_graph():
    stats = ConditionStats(torch.zeros(6), torch.ones(6))
    cfg = _cfg("unet")
    cfg["model"]["mask_target_time"] = True
    model = build_model(cfg, stats)
    batch = _batch()

    model(
        batch["m0"],
        batch["omega_target"],
        torch.tensor([0.25, 0.75]),
        _cond(batch),
    ).square().mean().backward()

    gradient = model.cond_embed.t_lookup.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient) == 0


def test_unet_attn_depth_mismatch_is_reported():
    stats = ConditionStats(torch.zeros(6), torch.ones(6))
    cfg = _cfg("unet")
    cfg["model"]["unet"].update(
        {
            "attention": True,
            "bottleneck_mode": "attn",
            "bottleneck_depth": 8,
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_model(cfg, stats)

    assert any("bottleneck_depth is ignored" in str(item.message) for item in caught)


def test_unet_attn_transformer_extension_is_checkpoint_compatible_identity():
    stats = ConditionStats(torch.zeros(6), torch.ones(6))
    base_cfg = _cfg("unet")
    base_cfg["model"]["unet"].update(
        {
            "attention": True,
            "bottleneck_mode": "attn",
            "bottleneck_depth": 1,
        }
    )
    deep_cfg = copy.deepcopy(base_cfg)
    deep_cfg["model"]["unet"].update(
        {
            "bottleneck_mode": "attn_transformer",
            "bottleneck_depth": 2,
            "bottleneck_hidden": 16,
            "bottleneck_heads": 2,
            "bottleneck_mlp_ratio": 2.0,
        }
    )

    torch.manual_seed(123)
    base_model = build_model(base_cfg, stats).eval()
    torch.manual_seed(456)
    deep_model = build_model(deep_cfg, stats).eval()
    missing, unexpected = deep_model.load_state_dict(base_model.state_dict(), strict=False)

    assert not unexpected
    assert missing
    assert all(key.startswith("bottleneck_transformer.") for key in missing)
    assert sum(parameter.numel() for parameter in deep_model.parameters()) > sum(
        parameter.numel() for parameter in base_model.parameters()
    )

    batch = _batch()
    cond = _cond(batch)
    tau = torch.tensor([0.25, 0.75])
    with torch.no_grad():
        base_out = base_model(batch["m0"], batch["omega_target"], tau, cond)
        deep_out = deep_model(batch["m0"], batch["omega_target"], tau, cond)
    # Identity input/output projections add only floating-point GEMM roundoff.
    torch.testing.assert_close(deep_out, base_out, rtol=0.0, atol=1.0e-6)

    deep_model.zero_grad(set_to_none=True)
    deep_model(batch["m0"], batch["omega_target"], tau, cond).square().mean().backward()
    gate_gradients = [
        parameter.grad
        for name, parameter in deep_model.named_parameters()
        if name.startswith("bottleneck_transformer.") and name.endswith("ada.1.weight")
    ]
    assert gate_gradients
    assert any(
        gradient is not None and torch.count_nonzero(gradient) > 0
        for gradient in gate_gradients
    )


def test_unet_extra_zero_initialized_resblocks_are_checkpoint_compatible_identity():
    stats = ConditionStats(torch.zeros(6), torch.ones(6))
    base_cfg = _cfg("unet")
    deep_cfg = copy.deepcopy(base_cfg)
    deep_cfg["model"]["unet"].update(
        {
            "num_res_blocks": 2,
            "zero_init_residual": True,
        }
    )

    torch.manual_seed(123)
    base_model = build_model(base_cfg, stats).eval()
    torch.manual_seed(456)
    deep_model = build_model(deep_cfg, stats).eval()
    missing, unexpected = deep_model.load_state_dict(base_model.state_dict(), strict=False)

    assert not unexpected
    assert missing
    assert all(
        (key.startswith("down_blocks.") or key.startswith("up_blocks."))
        and ".1." in key
        for key in missing
    )

    batch = _batch()
    cond = _cond(batch)
    tau = torch.tensor([0.25, 0.75])
    with torch.no_grad():
        base_out = base_model(batch["m0"], batch["omega_target"], tau, cond)
        deep_out = deep_model(batch["m0"], batch["omega_target"], tau, cond)
    torch.testing.assert_close(deep_out, base_out, rtol=0.0, atol=0.0)

    deep_model.zero_grad(set_to_none=True)
    deep_model(batch["m0"], batch["omega_target"], tau, cond).square().mean().backward()
    new_conv_gradients = [
        parameter.grad
        for name, parameter in deep_model.named_parameters()
        if (name.startswith("down_blocks.") or name.startswith("up_blocks."))
        and ".1.conv2.weight" in name
    ]
    assert new_conv_gradients
    assert any(
        gradient is not None and torch.count_nonzero(gradient) > 0
        for gradient in new_conv_gradients
    )


def test_unet_channel_expansion_transplants_checkpoint_and_preserves_output():
    stats = ConditionStats(torch.zeros(6), torch.ones(6))
    base_cfg = _cfg("unet")
    base_cfg["model"]["unet"].update(
        {
            "base_channels": 4,
            "channel_mults": [1, 2],
            "num_res_blocks": 1,
            "groups": 4,
            "attention": True,
            "bottleneck_mode": "attn",
            "bottleneck_depth": 1,
        }
    )
    wide_cfg = copy.deepcopy(base_cfg)
    wide_cfg["model"]["unet"].update(
        {
            "base_channels": 6,
            "num_res_blocks": 2,
            "groups": 6,
            "zero_init_residual": True,
            "checkpoint_channel_expansion": True,
            "checkpoint_source_base_channels": 4,
            "checkpoint_source_groups": 4,
            "bottleneck_mode": "attn_transformer",
            "bottleneck_legacy_channels": 8,
            "bottleneck_legacy_heads": 8,
            "bottleneck_hidden": 16,
            "bottleneck_depth": 2,
            "bottleneck_heads": 2,
            "bottleneck_mlp_ratio": 2.0,
        }
    )

    torch.manual_seed(123)
    base_model = build_model(base_cfg, stats).eval()
    torch.manual_seed(456)
    wide_model = build_model(wide_cfg, stats).eval()
    adapted = _adapt_resume_state_to_model(base_model.state_dict(), wide_model)
    shape_mismatches = [
        key
        for key, value in adapted.items()
        if key in wide_model.state_dict()
        and tuple(value.shape) != tuple(wide_model.state_dict()[key].shape)
    ]
    assert not shape_mismatches
    missing, unexpected = wide_model.load_state_dict(adapted, strict=False)
    assert not unexpected
    assert missing

    batch = _batch()
    cond = _cond(batch)
    tau = torch.tensor([0.25, 0.75])
    with torch.no_grad():
        base_out = base_model(batch["m0"], batch["omega_target"], tau, cond)
        wide_out = wide_model(batch["m0"], batch["omega_target"], tau, cond)
    # Wider convolutions change accumulation order, so function-preserving
    # channel transplantation is numerically (rather than bitwise) identical.
    torch.testing.assert_close(wide_out, base_out, rtol=0.0, atol=1.0e-6)

    wide_model.zero_grad(set_to_none=True)
    wide_model(batch["m0"], batch["omega_target"], tau, cond).square().mean().backward()
    assert wide_model.out_conv.weight.grad is not None
    assert torch.count_nonzero(wide_model.out_conv.weight.grad[:, 4:]) > 0
