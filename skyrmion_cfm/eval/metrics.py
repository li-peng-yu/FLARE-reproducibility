from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F

from skyrmion_cfm.models.common import central_diff, normalize_boundary, shift_neighbor


def _spatial_weight(mask: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    weight = mask.to(device=ref.device, dtype=ref.dtype)
    if weight.ndim == ref.ndim:
        if weight.shape[1] == 1:
            weight = weight[:, 0]
        else:
            weight = weight.mean(dim=1)
    if weight.ndim == ref.ndim - 1:
        pass
    elif weight.ndim == ref.ndim - 2:
        weight = weight.unsqueeze(0).expand(ref.shape[0], -1, -1)
    else:
        raise ValueError(
            f"spatial mask shape {tuple(mask.shape)} is incompatible with {tuple(ref.shape)}"
        )
    spatial_shape_ok = tuple(weight.shape[-2:]) == tuple(ref.shape[-2:])
    if weight.shape[0] != ref.shape[0] or not spatial_shape_ok:
        if weight.shape[0] == 1 and ref.shape[0] > 1 and spatial_shape_ok:
            weight = weight.expand(ref.shape[0], -1, -1)
        else:
            raise ValueError(
                f"spatial mask shape {tuple(mask.shape)} is incompatible with {tuple(ref.shape)}"
            )
    return weight.clamp(0.0, 1.0)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean(dim=(-1, -2))
    denom = mask.sum(dim=(-1, -2)).clamp_min(1.0)
    return (values * mask).sum(dim=(-1, -2)) / denom


def magnetic_fraction(mask: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
    weight = _spatial_weight(mask, ref)
    if weight is None:
        return ref.new_ones(ref.shape[0])
    return (weight > 0.5).to(dtype=ref.dtype).mean(dim=(-1, -2))


def apply_spatial_mask(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    weight = _spatial_weight(mask, x)
    if weight is None:
        return x
    return x * weight.unsqueeze(1)


def mse_m(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    weight = _spatial_weight(mask, pred)
    return _masked_mean((pred - target).square().sum(dim=1), weight)


def angular_error_deg(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    weight = _spatial_weight(mask, pred)
    dot = (pred * target).sum(dim=1).clamp(-1.0, 1.0)
    return _masked_mean(torch.rad2deg(torch.acos(dot)), weight)


def _ddx(m: torch.Tensor, boundary: str = "open") -> torch.Tensor:
    return central_diff(m, dim=-1, boundary=boundary)


def _ddy(m: torch.Tensor, boundary: str = "open") -> torch.Tensor:
    return central_diff(m, dim=-2, boundary=boundary)


# Backwards-compat shims for callers that imported the periodic helpers by name.
def _ddx_periodic(m: torch.Tensor) -> torch.Tensor:
    return _ddx(m, boundary="periodic")


def _ddy_periodic(m: torch.Tensor) -> torch.Tensor:
    return _ddy(m, boundary="periodic")


def topological_charge(m: torch.Tensor, boundary: str = "open") -> torch.Tensor:
    """Continuum finite-difference topological charge for BCHW magnetization.

    The boundary mode controls how the central differences treat the edges of
    the lattice. ``periodic`` matches the original wrap-around behaviour;
    ``open`` (the default) uses Neumann-like replicate padding, which matches a
    hard simulation boundary such as mumax3 ``setGeom`` without ``SetPBC``.
    Q is no longer a strict integer in the open case but is still a useful
    diagnostic.
    """
    dmx = _ddx(m, boundary)
    dmy = _ddy(m, boundary)
    density = (m * torch.cross(dmx, dmy, dim=1)).sum(dim=1)
    return density.sum(dim=(-1, -2)) / (4.0 * torch.pi)


def topological_charge_density(m: torch.Tensor, boundary: str = "open") -> torch.Tensor:
    """Per-cell finite-difference topological charge density for BCHW fields.

    Summing the returned ``(B, H, W)`` tensor over spatial dimensions gives the
    same value as :func:`topological_charge`. This is useful as a local training
    signal: it tells the model where topology is misplaced, not just how much
    the global charge differs.
    """
    dmx = _ddx(m, boundary)
    dmy = _ddy(m, boundary)
    density = (m * torch.cross(dmx, dmy, dim=1)).sum(dim=1)
    return density / (4.0 * torch.pi)


def structure_factor(m: torch.Tensor) -> torch.Tensor:
    mz = m[:, 2]
    fft = torch.fft.fft2(mz, norm="ortho")
    return torch.fft.fftshift(fft.abs().square(), dim=(-1, -2))


def structure_factor_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(structure_factor(pred), structure_factor(target), reduction="none").mean(
        dim=(-1, -2)
    )


def wasserstein_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Exact empirical W1 distance for one-dimensional empirical samples.

    The last dimension is treated as the sample axis. When the two
    distributions have different sample counts, we interpolate the smaller
    sorted sequence onto the larger's quantile grid (linear interpolation in
    quantile-space) and integrate ``|F_x^{-1}(u) - F_y^{-1}(u)|`` over
    ``u ∈ [0, 1]``. That is the standard W1 estimator for unequal samples;
    plan-2 evaluates against mumax3 reference distributions whose size
    typically differs from the model's draw count.
    """
    x_sorted = x.flatten(start_dim=1).sort(dim=1).values.float()
    y_sorted = y.flatten(start_dim=1).sort(dim=1).values.float()
    nx = x_sorted.shape[1]
    ny = y_sorted.shape[1]
    if nx == 0 or ny == 0:
        return x_sorted.new_zeros(x_sorted.shape[0])
    if nx == ny:
        return (x_sorted - y_sorted).abs().mean(dim=1)
    # Resample the smaller sequence onto the larger's quantile grid.
    if nx >= ny:
        big, small = x_sorted, y_sorted
        swap = False
    else:
        big, small = y_sorted, x_sorted
        swap = True
    n_big = big.shape[1]
    n_small = small.shape[1]
    # Quantile positions of the bigger sequence and where they sit in [0,1].
    q_big = (torch.arange(n_big, device=big.device).float() + 0.5) / n_big
    # Quantile positions of the smaller — also bin midpoints, no endpoint bias.
    q_small = (torch.arange(n_small, device=small.device).float() + 0.5) / n_small
    # Linearly interpolate ``small`` onto ``q_big``.
    # torch.searchsorted on a 1D index tensor; broadcast across batch.
    idx = torch.searchsorted(q_small, q_big).clamp(1, n_small - 1)
    q_lo = q_small[idx - 1]
    q_hi = q_small[idx]
    w = ((q_big - q_lo) / (q_hi - q_lo).clamp_min(1e-12)).clamp(0.0, 1.0)
    small_lo = small[:, idx - 1]
    small_hi = small[:, idx]
    small_interp = small_lo + w * (small_hi - small_lo)
    diff = (big - small_interp).abs()
    if swap:
        # |x - y| is symmetric; the swap only mattered for which side gets
        # interpolated, the integrand is unchanged.
        pass
    return diff.mean(dim=1)


def lifetime_events(nsk: torch.Tensor, dt_s: float, threshold: float = 0.5) -> list[float]:
    """Extract skyrmion lifetimes from an Nsk(t) curve.

    Each connected positive segment is one lifetime. This is intentionally
    defined on an already-computed skyrmion-count observable so it works for
    both predicted and reference rollouts.
    """
    curves = nsk.detach().cpu()
    if curves.ndim == 1:
        curves = curves[None]
    lifetimes: list[float] = []
    for curve in curves:
        alive = curve > threshold
        start: int | None = None
        for idx, is_alive in enumerate(alive.tolist()):
            if is_alive and start is None:
                start = idx
            elif not is_alive and start is not None:
                lifetimes.append((idx - start) * float(dt_s))
                start = None
        if start is not None:
            lifetimes.append((len(alive) - start) * float(dt_s))
    return lifetimes


def histogram_w1(
    pred_values: Iterable[float],
    ref_values: Iterable[float],
    bins: int = 32,
) -> float:
    """W1-like distance between two lifetime histograms."""
    pred = torch.tensor(list(pred_values), dtype=torch.float32)
    ref = torch.tensor(list(ref_values), dtype=torch.float32)
    if pred.numel() == 0 and ref.numel() == 0:
        return 0.0
    if pred.numel() == 0 or ref.numel() == 0:
        return float("inf")
    lo = min(float(pred.min()), float(ref.min()))
    hi = max(float(pred.max()), float(ref.max()))
    if hi <= lo:
        return 0.0
    edges = torch.linspace(lo, hi, bins + 1)
    pred_hist = torch.histc(pred, bins=bins, min=lo, max=hi)
    ref_hist = torch.histc(ref, bins=bins, min=lo, max=hi)
    pred_cdf = pred_hist.cumsum(0) / pred_hist.sum().clamp_min(1.0)
    ref_cdf = ref_hist.cumsum(0) / ref_hist.sum().clamp_min(1.0)
    return float((pred_cdf - ref_cdf).abs().mean() * (hi - lo))


def energy_density(
    m: torch.Tensor,
    b_t: torch.Tensor,
    exchange: float = 1.0,
    dmi: float = 0.0,
    anisotropy: float = 0.0,
    boundary: str = "open",
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Simple dimensionless Heisenberg + interfacial DMI + Zeeman energy."""
    bnd = normalize_boundary(boundary)
    spatial_mask = _spatial_weight(mask, m)
    mxp = shift_neighbor(m, dim=-1, boundary=bnd)
    myp = shift_neighbor(m, dim=-2, boundary=bnd)
    exch_x = (m * mxp).sum(dim=1)
    exch_y = (m * myp).sum(dim=1)
    if spatial_mask is not None:
        wxp = shift_neighbor(spatial_mask[:, None], dim=-1, boundary=bnd)[:, 0]
        wyp = shift_neighbor(spatial_mask[:, None], dim=-2, boundary=bnd)[:, 0]
        exch_x = exch_x * spatial_mask * wxp
        exch_y = exch_y * spatial_mask * wyp
    if bnd != "periodic":
        # The shifted neighbor at the last row/column does not correspond to a
        # real bond when the lattice has a hard boundary; mask those pairs out.
        exch_x = exch_x.clone()
        exch_y = exch_y.clone()
        exch_x[..., :, -1] = 0.0
        exch_y[..., -1, :] = 0.0
    exch = -exchange * (exch_x + exch_y)
    dmx = _ddx(m, bnd)
    dmy = _ddy(m, bnd)
    dmi_density = dmi * (
        m[:, 2] * (dmx[:, 0] + dmy[:, 1])
        - (m[:, 0] * dmx[:, 2] + m[:, 1] * dmy[:, 2])
    )
    while b_t.ndim < m.ndim:
        b_t = b_t[..., None]
    zeeman = -(m * b_t).sum(dim=1)
    anis = -anisotropy * m[:, 2].square()
    return _masked_mean(exch + dmi_density + zeeman + anis, spatial_mask)


def skyrmion_count(
    m: torch.Tensor,
    mz_threshold: float = 0.0,
    boundary: str = "open",
) -> torch.Tensor:
    """Heuristic count of connected reversed-domain regions in each frame.

    This is intentionally simple and dependency-free. It is useful as a rollout
    diagnostic, while topological_charge remains the physically meaningful
    continuous observable. ``boundary='periodic'`` wraps the connectivity at
    the lattice edges; ``open`` (default) does not, matching mumax3 ``setGeom``
    without ``SetPBC``.
    """
    bnd = normalize_boundary(boundary)
    wrap = bnd == "periodic"
    counts = []
    masks = (m[:, 2] < mz_threshold).detach().cpu().numpy()
    for mask in masks:
        h, w = mask.shape
        seen = set()
        count = 0
        for y in range(h):
            for x in range(w):
                if not mask[y, x] or (y, x) in seen:
                    continue
                count += 1
                stack = [(y, x)]
                seen.add((y, x))
                while stack:
                    cy, cx = stack.pop()
                    neighbors = []
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = cy + dy, cx + dx
                        if wrap:
                            ny %= h
                            nx %= w
                        elif ny < 0 or ny >= h or nx < 0 or nx >= w:
                            continue
                        neighbors.append((ny, nx))
                    for ny, nx in neighbors:
                        if mask[ny, nx] and (ny, nx) not in seen:
                            seen.add((ny, nx))
                            stack.append((ny, nx))
        counts.append(count)
    return torch.tensor(counts, device=m.device, dtype=torch.float32)
