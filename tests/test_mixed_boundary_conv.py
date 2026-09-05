from __future__ import annotations

import torch
import torch.nn.functional as F

from skyrmion_cfm.models.common import (
    BoundaryConv2d,
    V4_BOUNDARY_MODE_ORDER,
    boundary_periodicity_from_modes,
    pad_2d,
)


_REFERENCE_BOUNDARIES = ("open", "pbc_x", "pbc_y", "periodic", "open")


def _reference_forward(
    conv: BoundaryConv2d,
    x: torch.Tensor,
) -> torch.Tensor:
    py, px = conv.boundary_padding
    rows = []
    for index, boundary in enumerate(_REFERENCE_BOUNDARIES):
        sample = pad_2d(x[index : index + 1], (px, px, py, py), boundary)
        rows.append(
            F.conv2d(
                sample,
                conv.weight,
                conv.bias,
                conv.stride,
                conv.padding,
                conv.dilation,
                conv.groups,
            )
        )
    return torch.cat(rows, dim=0)


def test_mixed_boundary_conv_matches_split_reference_and_gradients() -> None:
    for stride in (1, 2):
        torch.manual_seed(7)
        conv = BoundaryConv2d(3, 4, 3, stride=stride, padding=1, boundary="open").double()
        modes = torch.arange(len(V4_BOUNDARY_MODE_ORDER), dtype=torch.long)
        probe = torch.randn(
            5,
            4,
            4 if stride == 2 else 8,
            5 if stride == 2 else 9,
            dtype=torch.double,
        )

        x_new = torch.randn(5, 3, 8, 9, dtype=torch.double, requires_grad=True)
        out_new = conv(x_new, modes)
        loss_new = (out_new * probe).sum()
        grad_x_new, grad_w_new, grad_b_new = torch.autograd.grad(
            loss_new,
            (x_new, conv.weight, conv.bias),
        )

        x_ref = x_new.detach().clone().requires_grad_(True)
        out_ref = _reference_forward(conv, x_ref)
        loss_ref = (out_ref * probe).sum()
        grad_x_ref, grad_w_ref, grad_b_ref = torch.autograd.grad(
            loss_ref,
            (x_ref, conv.weight, conv.bias),
        )

        torch.testing.assert_close(out_new, out_ref, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(grad_x_new, grad_x_ref, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(grad_w_new, grad_w_ref, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(grad_b_new, grad_b_ref, rtol=1e-10, atol=1e-10)


def test_mixed_boundary_conv_accepts_predecoded_periodicity() -> None:
    torch.manual_seed(11)
    conv = BoundaryConv2d(2, 3, 3, padding=1)
    x = torch.randn(5, 2, 7, 6)
    modes = torch.arange(5)
    flags = boundary_periodicity_from_modes(modes, len(modes), x.device)
    torch.testing.assert_close(conv(x, flags), conv(x, modes))


def test_mixed_boundary_conv_is_fullgraph_compilable() -> None:
    if not hasattr(torch, "compile"):
        return
    torch.manual_seed(13)
    conv = BoundaryConv2d(2, 3, 3, padding=1)
    x = torch.randn(5, 2, 7, 6)
    flags = boundary_periodicity_from_modes(torch.arange(5), 5, x.device)
    compiled = torch.compile(conv, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x, flags), conv(x, flags))
