#!/usr/bin/env python3
"""Adapt NeuralMAG's released learned-demag solver to the SKX x5 protocol.

This is deliberately a physics rollout, not a direct endpoint adapter.  The
released two-layer U-Net is used only for the demagnetizing field.  Exchange,
interfacial DMI, anisotropy, Zeeman field, spatial Slonczewski torque, thermal
noise, and RK4 time integration are supplied explicitly for x5.

The x5 film is one 1 nm layer.  To respect NeuralMAG's six-channel checkpoint,
we represent it as two tied 0.5 nm sublayers, duplicate the magnetization, and
average the two predicted demagnetizing fields.  ``validate-demag`` compares
that adapter with MAG2305's exact FFT demagnetizing field at the same geometry.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEURALMAG_ROOT = Path(str(Path(__file__).resolve().parents[1] / "third_party/NeuralMAG"))
DIST_ROOT = Path(str(Path(__file__).resolve().parents[1] / "third_party/distribution_score"))
for path in (PROJECT_ROOT, NEURALMAG_ROOT, DIST_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from distribution_score.distance import FORMAL_BLOCKS, FORMAL_SHIFT_RADIUS_PX  # noqa: E402
from distribution_score.same_condition import run as run_same_condition  # noqa: E402
from external_baselines.x5_external_baseline import _configure_cuda, _dataset_config  # noqa: E402
from graph.x5_author_native_rollout_dataset import attach_native_schedule_fields  # noqa: E402
from libs.Unet import UNet  # noqa: E402
from scripts.evaluate_skx_x5_same_condition_distribution import (  # noqa: E402
    DEFAULT_REPEAT_DATASET,
    SEGMENT_LABELS,
    _aggregate,
    _complete_test_groups,
    _condition_supports_duration,
    _override_multisegment_timing_durations,
    _prepare_condition,
    _prepare_multisegment_rollout_condition,
    _prepare_rollout_condition,
    _write_json,
)
from skyrmion_cfm.config import seed_everything  # noqa: E402
from skyrmion_cfm.data.fixed_time import build_fixed_time_datasets  # noqa: E402
from scripts.x5_distribution_metrics import (  # noqa: E402
    aggregate_clustered_mean,
    angular_energy_distance_from_files,
)


CHECKPOINT = NEURALMAG_ROOT / "egs/NMI/ckpt/k16/model.pt"
IMPLEMENTATION_VERSION = "neuralmag_x5_mumax_stencil_v2"
MU0 = 4.0 * math.pi * 1.0e-7
# MuMax3's ``GammaLL`` default.  Using the same value matters for a faithful
# long-horizon comparison (rather than using a newer CODATA electron value).
GAMMA_RAD_PER_T_S = 1.7595e11
HBAR_J_S = 1.054_571_817e-34
ELEMENTARY_CHARGE_C = 1.602_176_634e-19
KB_J_PER_K = 1.380_649e-23

# Exact constants used by generate_bt_nonuniform_sot_pairs.py for x5.
X5_DEFAULTS = {
    "alpha": 0.3,
    "aex_j_per_m": 1.5e-11,
    "dind_j_per_m2": 0.00325,
    "ku1_j_per_m3": 800_000.0,
    "msat_a_per_m": 580_000.0,
    "dx_m": 2.0e-9,
    "dy_m": 2.0e-9,
    "dz_m": 1.0e-9,
    "pol": 0.62,
    "epsilon_prime": 0.0,
    "fixed_layer": (0.0, -1.0, 0.0),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# This hash is the 6,000-step pilot
# previously named ``neuralmag_x5_demag_finetuned_v2.pt``.  Reject by content
# hash so renaming or copying the file cannot make it paper-eligible again.
DEPRECATED_CHECKPOINT_SHA256 = {
    "a93cd49f288c22997de5ea3f3dcc578fd521e705fa3e6135de9ea72543de1930": (
        "6,000-step NeuralMAG pilot; use the verified 50,000-step checkpoint "
        "with SHA-256 f47b2e4b5e1ee5395c7ae99a8438520ac0f27620866cd26b680e2aec8e80547d"
    ),
    # Reproducibility-package copy of the same 6k tensors with sanitized metadata.
    "1ccd6e5d79e06727a3ce1e5d9650436001d270b1bcbcaa8e2c637a5a6e0e8830": (
        "repackaged 6,000-step NeuralMAG pilot; use the verified 50,000-step checkpoint"
    ),
}


def _reject_deprecated_checkpoint(path: Path) -> None:
    digest = _sha256(path)
    reason = DEPRECATED_CHECKPOINT_SHA256.get(digest)
    if reason is not None:
        raise RuntimeError(f"deprecated NeuralMAG checkpoint {path}: {reason}")


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("no rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _author_commit() -> str:
    frozen_revision = NEURALMAG_ROOT / "SOURCE_COMMIT"
    if frozen_revision.is_file():
        return frozen_revision.read_text(encoding="utf-8").strip()
    return subprocess.check_output(
        ["git", "-C", str(NEURALMAG_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


class NeuralMAGDemag(torch.nn.Module):
    """Released NeuralMAG U-Net with a transparent 1-layer x5 adapter."""

    def __init__(self, checkpoint: Path = CHECKPOINT) -> None:
        super().__init__()
        self.network = UNet(kc=16, inc=6, ouc=6)
        _reject_deprecated_checkpoint(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        self.network.load_state_dict(state, strict=True)
        self.network.eval()

    def forward(self, m: torch.Tensor, msat_a_per_m: float = 580_000.0) -> torch.Tensor:
        if m.ndim != 4 or m.shape[1] != 3:
            raise ValueError(f"expected (B,3,H,W), got {tuple(m.shape)}")
        # NeuralMAG channel order is [layer0 xyz, layer1 xyz].
        output = self.network(torch.cat((m, m), dim=1))
        output = output.view(m.shape[0], 2, 3, *m.shape[-2:]).mean(dim=1)
        # Author convention: Hd[Oe] = network_output * Ms[emu/cc] / 1000.
        # Ms[emu/cc] = Ms[A/m] / 1000 and 1 Oe corresponds to 1e-4 T.
        return output * (float(msat_a_per_m) / 1.0e6) * 1.0e-4


def _raw_neuralmag_forward(network: UNet, x: torch.Tensor) -> torch.Tensor:
    """Author training-time forward pass, before the signed exp transform."""
    s1 = network.s1(x)
    s2 = network.s2(network.pool(s1))
    s3 = network.s3(network.pool(s2))
    s4 = network.s4(network.pool(s3))
    s5 = network.s5(network.pool(s4))
    up_1 = network.up_1(s5)
    up_2 = network.up_2(torch.cat((up_1, s4), dim=1))
    up_3 = network.up_3(torch.cat((up_2, s3), dim=1))
    up_4 = network.up_4(torch.cat((up_3, s2), dim=1))
    # ``last_Conv`` itself has no signed exponential; libs.Unet.forward adds it.
    return network.last_Conv(torch.cat((up_4, s1), dim=1))


def _signed_log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def _signed_expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(x.abs().clamp_max(12.0))


class X5ExactDemag:
    """Batched MAG2305 FFT target for two tied 0.5 nm x5 sublayers."""

    def __init__(self, device: torch.device) -> None:
        import libs.MAG2305 as MAG2305

        film = MAG2305.mmModel(
            types="bulk",
            size=(256, 256, 2),
            cell=(2.0, 2.0, 0.5),
            Ms=580.0,
            Ax=1.5e-6,
            Ku=8.0e6,
            Kvec=(0, 0, 1),
            device=str(device),
        )
        film.DemagInit()
        self.kernel = film.FDMW.detach().clone()
        self.fft_shape = tuple(int(value) for value in film.fftsize)
        del film

    @torch.no_grad()
    def __call__(self, m: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = m.shape
        if channels != 3 or (height, width) != (256, 256):
            raise ValueError(tuple(m.shape))
        # MAG2305 layout is (component, x, y, z).  The x5 lattice is square,
        # so the dataset's two spatial axes map directly without resampling.
        layers = m[:, None].expand(batch, 2, channels, height, width)
        magnetic = layers.permute(0, 2, 3, 4, 1) * 580.0
        padded = torch.zeros(
            (batch, 3, *self.fft_shape), device=m.device, dtype=torch.float32
        )
        padded[:, :, :height, :width, :2] = magnetic
        magnetization_fft = torch.fft.rfftn(padded, dim=(2, 3, 4))
        field_fft = -torch.einsum(
            "mnxyz,bnxyz->bmxyz", self.kernel, magnetization_fft
        )
        field = torch.fft.irfftn(
            field_fft, s=self.fft_shape, dim=(2, 3, 4)
        )[:, :, :height, :width, :2]
        field_oe = field * (4.0 * math.pi)
        return field_oe.permute(0, 4, 1, 2, 3).reshape(batch, 6, height, width)


class X5ExactDemagField(torch.nn.Module):
    """Exact-FFT diagnostic with the same Tesla-valued interface as NeuralMAG."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.operator = X5ExactDemag(device)

    def forward(self, m: torch.Tensor, _msat_a_per_m: float = 580_000.0) -> torch.Tensor:
        field_oe = self.operator(m).view(m.shape[0], 2, 3, *m.shape[-2:])
        return field_oe.mean(dim=1) * 1.0e-4


def _mumax_exchange_dmi_field_t(
    m: torch.Tensor,
    *,
    aex_j_per_m: float,
    dind_j_per_m2: float,
    msat_a_per_m: float,
    dx_m: float,
    dy_m: float,
) -> torch.Tensor:
    """MuMax3's uniform-material exchange + interfacial-DMI stencil.

    This follows ``cuda/dmi.cu`` directly.  In particular, positive ``Dind``
    contributes ``(+d_x m_z, +d_y m_z, -d_x m_x-d_y m_y)`` in the interior,
    and the missing edge cells are extrapolated with MuMax3's chiral Neumann
    boundary condition.  A simple replicated pad has the wrong DMI sign and
    the wrong edge field for this benchmark.
    """
    if m.ndim != 4 or m.shape[1] != 3:
        raise ValueError(f"expected (B,3,H,W), got {tuple(m.shape)}")
    aex = float(aex_j_per_m)
    dind = float(dind_j_per_m2)
    msat = float(msat_a_per_m)
    dx = float(dx_m)
    dy = float(dy_m)
    if aex <= 0.0 or msat <= 0.0 or dx <= 0.0 or dy <= 0.0:
        raise ValueError("Aex, Msat, dx, and dy must be positive")

    left = torch.empty_like(m)
    right = torch.empty_like(m)
    lower = torch.empty_like(m)
    upper = torch.empty_like(m)
    left[..., 1:] = m[..., :-1]
    right[..., :-1] = m[..., 1:]
    lower[..., 1:, :] = m[..., :-1, :]
    upper[..., :-1, :] = m[..., 1:, :]

    # Exact ghost-cell extrapolation from MuMax3 cuda/dmi.cu.  The channel
    # slices deliberately retain a singleton component axis.
    gx = dx * (0.5 * dind / aex)
    gy = dy * (0.5 * dind / aex)
    edge_left = m[..., :1]
    edge_right = m[..., -1:]
    left[:, 0:1, :, :1] = edge_left[:, 0:1] + gx * edge_left[:, 2:3]
    left[:, 1:2, :, :1] = edge_left[:, 1:2]
    left[:, 2:3, :, :1] = edge_left[:, 2:3] - gx * edge_left[:, 0:1]
    right[:, 0:1, :, -1:] = edge_right[:, 0:1] - gx * edge_right[:, 2:3]
    right[:, 1:2, :, -1:] = edge_right[:, 1:2]
    right[:, 2:3, :, -1:] = edge_right[:, 2:3] + gx * edge_right[:, 0:1]

    edge_lower = m[..., :1, :]
    edge_upper = m[..., -1:, :]
    lower[:, 0:1, :1, :] = edge_lower[:, 0:1]
    lower[:, 1:2, :1, :] = edge_lower[:, 1:2] + gy * edge_lower[:, 2:3]
    lower[:, 2:3, :1, :] = edge_lower[:, 2:3] - gy * edge_lower[:, 1:2]
    upper[:, 0:1, -1:, :] = edge_upper[:, 0:1]
    upper[:, 1:2, -1:, :] = edge_upper[:, 1:2] - gy * edge_upper[:, 2:3]
    upper[:, 2:3, -1:, :] = edge_upper[:, 2:3] + gy * edge_upper[:, 1:2]

    energy_gradient = (
        (2.0 * aex / (dx * dx)) * (left + right - 2.0 * m)
        + (2.0 * aex / (dy * dy)) * (lower + upper - 2.0 * m)
    )
    energy_gradient[:, 0:1] += (dind / dx) * (
        right[:, 2:3] - left[:, 2:3]
    )
    energy_gradient[:, 1:2] += (dind / dy) * (
        upper[:, 2:3] - lower[:, 2:3]
    )
    energy_gradient[:, 2:3] += (dind / dx) * (
        left[:, 0:1] - right[:, 0:1]
    ) + (dind / dy) * (lower[:, 1:2] - upper[:, 1:2])
    return energy_gradient / msat


class NeuralMAGX5Simulator:
    """Fixed-step stochastic RK4 solver with NeuralMAG demagnetization."""

    def __init__(
        self,
        demag: torch.nn.Module,
        *,
        dt_s: float = 1.0e-13,
        thermal: bool = True,
        sot_scale: float = 1.0,
        dmi_scale: float = 1.0,
    ) -> None:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        self.demag = demag
        self.dt_s = float(dt_s)
        self.thermal = bool(thermal)
        self.sot_scale = float(sot_scale)
        self.dmi_scale = float(dmi_scale)

    @staticmethod
    def _normalize(m: torch.Tensor) -> torch.Tensor:
        return F.normalize(m, dim=1, eps=1.0e-12)

    def effective_field_t(
        self,
        m: torch.Tensor,
        *,
        bz_t: torch.Tensor,
        thermal_b_t: torch.Tensor | None,
    ) -> torch.Tensor:
        p = X5_DEFAULTS
        ms = float(p["msat_a_per_m"])
        aex = float(p["aex_j_per_m"])
        dind = float(p["dind_j_per_m2"]) * self.dmi_scale
        ku1 = float(p["ku1_j_per_m3"])
        dx = float(p["dx_m"])
        dy = float(p["dy_m"])

        b_eff = self.demag(m, ms)
        b_eff = b_eff + _mumax_exchange_dmi_field_t(
            m,
            aex_j_per_m=aex,
            dind_j_per_m2=dind,
            msat_a_per_m=ms,
            dx_m=dx,
            dy_m=dy,
        )
        b_anis = torch.zeros_like(m)
        b_anis[:, 2:3] = (2.0 * ku1 / ms) * m[:, 2:3]
        b_eff = b_eff + b_anis
        b_eff[:, 2:3] = b_eff[:, 2:3] + bz_t[:, None, None, None]
        if thermal_b_t is not None:
            b_eff = b_eff + thermal_b_t
        return b_eff

    def rhs(
        self,
        m: torch.Tensor,
        *,
        bz_t: torch.Tensor,
        current_z_a_per_m2: torch.Tensor,
        thermal_b_t: torch.Tensor | None,
    ) -> torch.Tensor:
        alpha = float(X5_DEFAULTS["alpha"])
        b_eff = self.effective_field_t(m, bz_t=bz_t, thermal_b_t=thermal_b_t)
        mxh = torch.cross(m, b_eff, dim=1)
        mxmxh = torch.cross(m, mxh, dim=1)
        result = -GAMMA_RAD_PER_T_S / (1.0 + alpha * alpha) * (
            mxh + alpha * mxmxh
        )

        ms = float(X5_DEFAULTS["msat_a_per_m"])
        dz = float(X5_DEFAULTS["dz_m"])
        pol = float(X5_DEFAULTS["pol"])
        eps_prime = float(X5_DEFAULTS["epsilon_prime"])
        beta_t = (
            (HBAR_J_S / ELEMENTARY_CHARGE_C)
            * current_z_a_per_m2
            / (dz * ms)
        )
        b_dl = beta_t * (0.5 * pol)
        b_fl = beta_t * eps_prime
        explicit_dl = (b_dl + alpha * b_fl) / (1.0 + alpha * alpha)
        explicit_fl = (b_fl - alpha * b_dl) / (1.0 + alpha * alpha)
        sigma = m.new_tensor(X5_DEFAULTS["fixed_layer"]).view(1, 3, 1, 1)
        sigma = sigma.expand_as(m)
        dl_basis = torch.cross(m, torch.cross(sigma, m, dim=1), dim=1)
        fl_basis = torch.cross(sigma, m, dim=1)
        result = result + self.sot_scale * GAMMA_RAD_PER_T_S * (
            explicit_dl * dl_basis + explicit_fl * fl_basis
        )
        return result

    def _thermal_field(
        self,
        m: torch.Tensor,
        temp_k: torch.Tensor,
        step_dt_s: float,
        generator: torch.Generator,
    ) -> torch.Tensor | None:
        if not self.thermal or float(temp_k.max()) <= 0.0:
            return None
        p = X5_DEFAULTS
        volume = float(p["dx_m"]) * float(p["dy_m"]) * float(p["dz_m"])
        variance = (
            2.0
            * float(p["alpha"])
            * KB_J_PER_K
            * temp_k
            / (
                float(p["msat_a_per_m"])
                * volume
                * GAMMA_RAD_PER_T_S
                * step_dt_s
            )
        )
        std = variance.clamp_min(0.0).sqrt().view(-1, 1, 1, 1)
        noise = torch.randn(
            m.shape,
            device=m.device,
            dtype=m.dtype,
            generator=generator,
        )
        return noise * std

    def rk4_step(
        self,
        m: torch.Tensor,
        *,
        step_dt_s: float,
        bz_t: torch.Tensor,
        current_z_a_per_m2: torch.Tensor,
        temp_k: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        thermal = self._thermal_field(m, temp_k, step_dt_s, generator)
        kwargs = {
            "bz_t": bz_t,
            "current_z_a_per_m2": current_z_a_per_m2,
            "thermal_b_t": thermal,
        }
        k1 = self.rhs(m, **kwargs)
        k2 = self.rhs(m + 0.5 * step_dt_s * k1, **kwargs)
        k3 = self.rhs(m + 0.5 * step_dt_s * k2, **kwargs)
        k4 = self.rhs(m + step_dt_s * k3, **kwargs)
        return self._normalize(m + (step_dt_s / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))

    @torch.inference_mode()
    def rollout(
        self,
        m_init: torch.Tensor,
        *,
        horizon_s: float,
        bz_t: torch.Tensor,
        current_z_a_per_m2: torch.Tensor,
        temp_k: torch.Tensor,
        seed: int,
        generator: torch.Generator | None = None,
        progress_every: int = 0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if horizon_s <= 0.0:
            raise ValueError("horizon_s must be positive")
        batch = int(m_init.shape[0])
        for name, value in (("bz_t", bz_t), ("temp_k", temp_k)):
            if value.shape != (batch,):
                raise ValueError(f"{name} shape {tuple(value.shape)} != ({batch},)")
        if current_z_a_per_m2.shape != (batch, 1, *m_init.shape[-2:]):
            raise ValueError(
                "current_z_a_per_m2 must have shape "
                f"(B,1,H,W), got {tuple(current_z_a_per_m2.shape)}"
            )
        if generator is None:
            generator = torch.Generator(device=m_init.device)
            generator.manual_seed(int(seed))
        m = self._normalize(m_init.float())
        full_steps = int(math.floor(horizon_s / self.dt_s + 1.0e-9))
        remainder = float(horizon_s - full_steps * self.dt_s)
        if remainder < self.dt_s * 1.0e-6:
            remainder = 0.0
        total_steps = full_steps + int(remainder > 0.0)
        started = time.perf_counter()
        for index in range(total_steps):
            step_dt = self.dt_s if index < full_steps else remainder
            m = self.rk4_step(
                m,
                step_dt_s=step_dt,
                bz_t=bz_t,
                current_z_a_per_m2=current_z_a_per_m2,
                temp_k=temp_k,
                generator=generator,
            )
            if progress_every and (index + 1) % progress_every == 0:
                if m.device.type == "cuda":
                    torch.cuda.synchronize(m.device)
                print(
                    {
                        "neuralmag_x5_step": index + 1,
                        "total_steps": total_steps,
                        "elapsed_s": time.perf_counter() - started,
                    },
                    flush=True,
                )
        if m.device.type == "cuda":
            torch.cuda.synchronize(m.device)
        if not bool(torch.isfinite(m).all()):
            raise RuntimeError("NeuralMAG-x5 rollout produced non-finite magnetization")
        return m, {
            "rk4_steps": total_steps,
            "learned_demag_calls": 4 * total_steps,
            "horizon_s": horizon_s,
            "elapsed_seconds": time.perf_counter() - started,
            "batch_size": batch,
        }


def _record_values(record: Any) -> tuple[float, float]:
    params = record.params
    bz = params.get("bz_t", params.get("fixed_bz_t", params.get("bias_bz_t", 0.0)))
    temp = params.get("temp_k", params.get("fixed_temp_k", 0.0))
    return float(bz), float(temp)


def _condition_batch(
    prepared: dict[str, Any],
    dataset: Any,
    record_indices: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    starts: list[torch.Tensor] = []
    currents: list[torch.Tensor] = []
    bz_values: list[float] = []
    temp_values: list[float] = []
    horizons: list[float] = []
    for sample, record_index in zip(prepared["samples"], record_indices, strict=True):
        sample = attach_native_schedule_fields(dataset, int(record_index), sample)
        starts.append(sample.get("m_observed_init", sample["m_init"]).float())
        current = sample["planned_j_z_field"].float()
        if current.ndim == 2:
            current = current.unsqueeze(0)
        if int(sample["control_segment_index"].item()) != 1:
            current = torch.zeros_like(current)
        currents.append(current)
        bz, temp = _record_values(dataset.records[int(record_index)])
        bz_values.append(bz)
        temp_values.append(temp)
        horizons.append(float(sample["t_end_s"].item()))
    # OVF headers and float32 sample metadata can disagree by a few 1e-14 s
    # for repeats of the same nominal segment.  This remains well below one
    # 0.1 ps integration step and is not a physical condition mismatch.
    if max(horizons) - min(horizons) > 5.0e-14:
        raise RuntimeError(f"repeat horizons differ: {horizons}")
    return (
        torch.stack(starts).to(device),
        torch.stack(currents).to(device),
        torch.tensor(bz_values, device=device, dtype=torch.float32),
        torch.tensor(temp_values, device=device, dtype=torch.float32),
        statistics.fmean(horizons),
    )


# Preserve float64 protocol
# durations so the exact 3.5-ns relaxation uses 35,000, not 35,001, RK4 steps.
def _multisegment_condition_batches(
    prepared: dict[str, Any],
    dataset: Any,
    record_indices: list[int],
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]]:
    paths = prepared["segment_samples"]
    if len(paths) != len(record_indices) or not paths:
        raise RuntimeError("misaligned NeuralMAG multisegment paths")
    segment_count = len(paths[0])
    if segment_count < 2 or any(len(path) != segment_count for path in paths):
        raise RuntimeError("invalid NeuralMAG multisegment path shape")
    batches = [
        _condition_batch(
            {"samples": [path[position] for path in paths]},
            dataset,
            record_indices,
            device,
        )
        for position in range(segment_count)
    ]
    # Keep integration time in float64 protocol precision.  The model samples
    # store t_end_s as float32 for neural conditioning; using that rounded
    # value for RK4 can create a spurious near-zero remainder step at 3.5 ns.
    exact_by_repeat = prepared["metadata"]["repeats"]
    corrected: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]
    ] = []
    for position, (starts, currents, bz, temp, _rounded_horizon_s) in enumerate(
        batches
    ):
        durations_ns = [
            float(repeat["segments"][position]["duration_ns"])
            for repeat in exact_by_repeat
        ]
        if max(durations_ns) - min(durations_ns) > 2.0e-4:
            raise RuntimeError(
                f"exact NeuralMAG segment horizons differ: {durations_ns}"
            )
        corrected.append(
            (
                starts,
                currents,
                bz,
                temp,
                statistics.fmean(durations_ns) * 1.0e-9,
            )
        )
    return corrected


def _load_test_dataset(config: Path) -> Any:
    cfg = _dataset_config(config)
    cfg.setdefault("data", {}).setdefault("augment", {})["enabled"] = False
    _, _, dataset = build_fixed_time_datasets(cfg, build_splits={"test"})
    if dataset is None:
        raise RuntimeError("failed to build x5 test split")
    return dataset


def _load_model(
    device: torch.device, checkpoint: Path = CHECKPOINT
) -> NeuralMAGDemag:
    model = NeuralMAGDemag(checkpoint).to(device).eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    return model


def _demag_metrics(
    network: UNet,
    inputs: torch.Tensor,
    exact_oe: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        raw = _raw_neuralmag_forward(network, inputs)
        # Runtime scales the Ms=1000 normalized prediction by x5 Ms/1000.
        predicted_oe = _signed_expm1(raw) * 0.58
        difference = predicted_oe - exact_oe
        pred_flat = predicted_oe.flatten(1)
        exact_flat = exact_oe.flatten(1)
        return {
            "mae_oe": float(difference.abs().mean()),
            "rmse_oe": float(difference.square().mean().sqrt()),
            "relative_l2": float(
                difference.square().sum().sqrt()
                / exact_oe.square().sum().sqrt().clamp_min(1.0e-12)
            ),
            "field_cosine_mean": float(
                F.cosine_similarity(pred_flat, exact_flat, dim=1).mean()
            ),
        }


def finetune_demag(args: argparse.Namespace) -> None:
    """Fine-tune only the released learned-demag subroutine on x5 geometry."""
    device = torch.device(args.device)
    cfg = _dataset_config(args.config)
    cfg.setdefault("data", {}).setdefault("augment", {})["enabled"] = False
    train_dataset, _, test_dataset = build_fixed_time_datasets(
        cfg, build_splits={"train", "test"}
    )
    if train_dataset is None or test_dataset is None:
        raise RuntimeError("failed to build train/test datasets for demag fine-tuning")
    exact_operator = X5ExactDemag(device)
    adapter = _load_model(device, args.checkpoint)
    network = adapter.network
    network.train()
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed)

    def sample_batch(
        dataset: Any, indices: np.ndarray, *, augment: bool = False
    ) -> torch.Tensor:
        states: list[torch.Tensor] = []
        for offset, index in enumerate(indices.tolist()):
            sample = dataset[int(index)]
            if offset % 2:
                state = sample["m_t"]
            else:
                state = sample.get("m_observed_init", sample["m_init"])
            states.append(state.float())
        batch = torch.stack(states).to(device)
        if augment and args.state_noise_std > 0.0:
            # Cover the neighbourhood visited by an autoregressive rollout,
            # rather than fitting only exact MuMax snapshots.  Targets remain
            # exact FFT fields of these synthetic states and no test target is
            # used for optimization.
            amplitude = torch.rand(
                (batch.shape[0], 1, 1, 1), device=device, dtype=batch.dtype
            ) * float(args.state_noise_std)
            batch = F.normalize(batch + amplitude * torch.randn_like(batch), dim=1)
        return batch

    test_indices = np.linspace(
        0, len(test_dataset) - 1, num=args.validation_samples, dtype=np.int64
    )
    validation_m = sample_batch(test_dataset, test_indices)
    validation_input = torch.cat((validation_m, validation_m), dim=1)
    validation_exact = exact_operator(validation_m)
    before = _demag_metrics(network, validation_input, validation_exact)

    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        indices = rng.integers(0, len(train_dataset), size=args.batch_size)
        m = sample_batch(train_dataset, indices, augment=True)
        inputs = torch.cat((m, m), dim=1)
        with torch.no_grad():
            exact_oe = exact_operator(m)
            normalized_target = exact_oe / 0.58
            target_log = _signed_log1p(normalized_target)
        raw = _raw_neuralmag_forward(network, inputs)
        loss_log = F.smooth_l1_loss(raw, target_log, beta=0.25)
        predicted_normalized = _signed_expm1(raw)
        target_scale = normalized_target.square().mean().sqrt().clamp_min(1.0)
        loss_physical = ((predicted_normalized - normalized_target) / target_scale).square().mean()
        loss = loss_log + args.physical_weight * loss_physical
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row: dict[str, float | int] = {
                "step": step,
                "loss": float(loss.detach()),
                "loss_log": float(loss_log.detach()),
                "loss_physical": float(loss_physical.detach()),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            print(row, flush=True)

    network.eval()
    after = _demag_metrics(network, validation_input, validation_exact)
    payload = {
        "state_dict": {key: value.detach().cpu() for key, value in network.state_dict().items()},
        "metadata": {
            "schema": "neuralmag_x5_demag_finetune_v1",
            "base_checkpoint": str(args.checkpoint),
            "base_checkpoint_sha256": _sha256(args.checkpoint),
            "author_repository": str(NEURALMAG_ROOT),
            "author_commit": _author_commit(),
            "training_split": "x5 metadata train split only",
            "validation_split": "x5 metadata test split",
            "target": "MAG2305 exact FFT demag, two tied 0.5nm sublayers",
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "physical_weight": args.physical_weight,
            "state_noise_std": args.state_noise_std,
            "before": before,
            "after": after,
            "history": history,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    report = dict(payload["metadata"])
    report["output"] = str(args.output)
    report["output_sha256"] = _sha256(args.output)
    _write_json(args.output.with_suffix(".json"), report)
    print(json.dumps(report, indent=2), flush=True)


@torch.inference_mode()
def validate_demag(args: argparse.Namespace) -> None:
    import libs.MAG2305 as MAG2305

    device = torch.device(args.device)
    dataset = _load_test_dataset(args.config)
    groups = _complete_test_groups(
        dataset,
        repeat_dataset=DEFAULT_REPEAT_DATASET,
        repeats_per_group=5,
    )
    first_base, record_indices = next(iter(groups.items()))
    prepared = _prepare_condition(
        dataset,
        base=first_base,
        record_indices=record_indices,
        segment_index=1,
        condition_dir=args.output.parent / "validation_condition",
    )
    starts, _currents, _bz, _temp, _horizon = _condition_batch(
        prepared, dataset, record_indices, device
    )
    model = _load_model(device, args.checkpoint)
    predicted = model(starts[: args.samples])

    film = MAG2305.mmModel(
        types="bulk",
        size=(256, 256, 2),
        cell=(2.0, 2.0, 0.5),
        Ms=580.0,
        Ax=1.5e-6,
        Ku=8.0e6,
        Kvec=(0, 0, 1),
        device=str(device),
    )
    film.DemagInit()
    exact_values: list[torch.Tensor] = []
    for m in starts[: args.samples]:
        one_layer = m.permute(1, 2, 0).cpu().numpy()
        two_layer = np.stack((one_layer, one_layer), axis=2)
        film.SpinInit(two_layer)
        film.DemagField_FFT()
        exact = film.Hd.mean(dim=2).permute(2, 0, 1) * 1.0e-4
        exact_values.append(exact)
    exact = torch.stack(exact_values)
    difference = predicted - exact
    exact_norm = exact.square().sum().sqrt()
    pred_flat = predicted.flatten(1)
    exact_flat = exact.flatten(1)
    payload = {
        "status": "complete",
        "schema": "neuralmag_x5_demag_validation_v1",
        "base_id": first_base,
        "samples": args.samples,
        "adapter": "one 1nm layer represented by two tied 0.5nm sublayers",
        "reference": "author MAG2305 exact FFT at cell=(2nm,2nm,0.5nm), two layers",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "mae_t": float(difference.abs().mean()),
        "rmse_t": float(difference.square().mean().sqrt()),
        "relative_l2": float(difference.square().sum().sqrt() / exact_norm.clamp_min(1e-12)),
        "field_cosine_mean": float(F.cosine_similarity(pred_flat, exact_flat, dim=1).mean()),
        "exact_rms_t": float(exact.square().mean().sqrt()),
        "predicted_rms_t": float(predicted.square().mean().sqrt()),
    }
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2), flush=True)


def score_shard(args: argparse.Namespace) -> None:
    if args.draws_per_anchor < 1:
        raise ValueError("draws-per-anchor must be positive")
    if args.duration_ns is not None and args.duration_ns <= 0.0:
        raise ValueError("duration-ns must be positive")
    selected_task_modes = sum(
        (
            args.duration_ns is not None,
            args.rollout_target_root is not None,
            bool(args.multisegment_rollout),
        )
    )
    if selected_task_modes > 1:
        raise ValueError(
            "--duration-ns, --rollout-target-root, and --multisegment-rollout "
            "are mutually exclusive"
        )
    if args.rollout_target_root is not None:
        args.rollout_target_root = args.rollout_target_root.resolve()
        if not (args.rollout_target_root / "manifest.json").is_file():
            raise FileNotFoundError(args.rollout_target_root / "manifest.json")
        args.segments = (2,)
    if args.multisegment_rollout:
        args.segments = (-1,)
    requested_horizon_ns = (
        5.0 if args.rollout_target_root is not None else args.duration_ns
    )
    device = torch.device(args.device)
    dataset = _load_test_dataset(args.config)
    all_groups = list(
        _complete_test_groups(
            dataset,
            repeat_dataset=DEFAULT_REPEAT_DATASET,
            repeats_per_group=5,
        ).items()
    )
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    groups = all_groups[args.shard_index :: args.shard_count]
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    if not groups:
        raise RuntimeError("empty NeuralMAG-x5 group shard")
    model: torch.nn.Module = (
        X5ExactDemagField(device)
        if args.exact_demag
        else _load_model(device, args.checkpoint)
    )
    simulator = NeuralMAGX5Simulator(
        model,
        dt_s=args.dt_s,
        thermal=not args.no_thermal,
        sot_scale=args.sot_scale,
        dmi_scale=args.dmi_scale,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped_conditions: list[dict[str, Any]] = []
    started = time.time()
    position = 0
    total = len(groups) * len(args.segments)
    for base, record_indices in groups:
        for segment_index in args.segments:
            position += 1
            condition_id = (
                f"base{base}_exact_control_drive_to_post_relax"
                if args.multisegment_rollout
                else (
                    f"base{base}_segment{segment_index:03d}_"
                    f"{SEGMENT_LABELS.get(segment_index, 'condition')}"
                )
            )
            # Score NeuralMAG on the same
            # exact legal duration as FLARE and the endpoint baselines.
            if args.duration_ns is not None and not _condition_supports_duration(
                dataset,
                record_indices=record_indices,
                segment_index=segment_index,
                duration_ns=float(args.duration_ns),
            ):
                skipped_conditions.append(
                    {
                        "condition_id": condition_id,
                        "base_id": base,
                        "control_segment_index": segment_index,
                        "requested_horizon_ns": float(args.duration_ns),
                        "reason": "requested target lies outside this control segment",
                    }
                )
                print(
                    json.dumps(
                        {
                            "shard": args.shard_index,
                            "position": position,
                            "total": total,
                            "condition_id": condition_id,
                            "status": "unsupported_horizon",
                        }
                    ),
                    flush=True,
                )
                continue
            condition_dir = args.output_root / "conditions" / condition_id
            condition_dir.mkdir(parents=True, exist_ok=True)
            if args.multisegment_rollout:
                prepared = _prepare_multisegment_rollout_condition(
                    dataset,
                    base=base,
                    record_indices=record_indices,
                    condition_dir=condition_dir,
                )
            elif args.rollout_target_root is not None:
                prepared = _prepare_rollout_condition(
                    dataset,
                    base=base,
                    record_indices=record_indices,
                    condition_dir=condition_dir,
                    target_root=args.rollout_target_root,
                    horizon_ns=5.0,
                )
            else:
                prepared = _prepare_condition(
                    dataset,
                    base=base,
                    record_indices=record_indices,
                    segment_index=segment_index,
                    condition_dir=condition_dir,
                    duration_ns=args.duration_ns,
                )
            model_dir = condition_dir / "model_samples"
            model_path = model_dir / "model_samples_f16.npy"
            manifest_path = model_dir / "manifest.json"
            expected_shape = [
                len(record_indices) * args.draws_per_anchor,
                3,
                256,
                256,
            ]
            anchor_index_path = model_dir / "anchor_repeat_index.npy"
            reuse = False
            if model_path.is_file() and manifest_path.is_file() and not args.force:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                reuse = (
                    manifest.get("status") == "complete"
                    and manifest.get("implementation_version") == IMPLEMENTATION_VERSION
                    and manifest.get("shape") == expected_shape
                    and math.isclose(float(manifest.get("dt_s", -1.0)), args.dt_s)
                    and bool(manifest.get("thermal")) == (not args.no_thermal)
                    and manifest.get("checkpoint_sha256") == _sha256(args.checkpoint)
                    and bool(manifest.get("exact_demag")) == args.exact_demag
                    and math.isclose(float(manifest.get("sot_scale", -99.0)), args.sot_scale)
                    and math.isclose(float(manifest.get("dmi_scale", -99.0)), args.dmi_scale)
                    and int(manifest.get("draws_per_anchor", 1))
                    == args.draws_per_anchor
                    and manifest.get("requested_horizon_ns") == requested_horizon_ns
                    and manifest.get("reference_mode")
                    == prepared["metadata"].get(
                        "reference_mode", "saved_segment_endpoint"
                    )
                    and anchor_index_path.is_file()
                )
            if not reuse:
                if args.multisegment_rollout:
                    segment_batches = _multisegment_condition_batches(
                        prepared, dataset, record_indices, device
                    )
                    starts = segment_batches[0][0]
                else:
                    starts, currents, bz, temp, horizon_s = _condition_batch(
                        prepared, dataset, record_indices, device
                    )
                predictions: list[np.ndarray] = []
                rollout_metas: list[dict[str, Any]] = []
                base_seed = (
                    args.seed + int(base) * 1009 + int(segment_index) * 100_003
                )
                draw_batch_size = (
                    args.draws_per_anchor
                    if args.draw_batch_size <= 0
                    else min(args.draw_batch_size, args.draws_per_anchor)
                )
                for draw_start in range(
                    0, args.draws_per_anchor, draw_batch_size
                ):
                    draw_count = min(
                        draw_batch_size,
                        args.draws_per_anchor - draw_start,
                    )
                    batched_starts = starts.repeat(draw_count, 1, 1, 1)
                    if args.multisegment_rollout:
                        generator = torch.Generator(device=device)
                        generator.manual_seed(
                            base_seed + draw_start * 1_000_003
                        )
                        prediction = batched_starts
                        segment_rollouts: list[dict[str, Any]] = []
                        for segment_position, (
                            _truth_starts,
                            segment_currents,
                            segment_bz,
                            segment_temp,
                            segment_horizon_s,
                        ) in enumerate(segment_batches):
                            prediction, segment_meta = simulator.rollout(
                                prediction,
                                horizon_s=segment_horizon_s,
                                bz_t=segment_bz.repeat(draw_count),
                                current_z_a_per_m2=segment_currents.repeat(
                                    draw_count, 1, 1, 1
                                ),
                                temp_k=segment_temp.repeat(draw_count),
                                seed=base_seed + draw_start * 1_000_003,
                                generator=generator,
                                progress_every=args.progress_every,
                            )
                            segment_meta["segment_position"] = segment_position
                            segment_meta["control_segment_index"] = prepared[
                                "metadata"
                            ]["control_segment_indices"][segment_position]
                            segment_rollouts.append(segment_meta)
                        rollout_meta = {
                            "protocol": "piecewise-control RK4 with continuous thermal RNG",
                            "segments": segment_rollouts,
                            "rk4_steps": sum(
                                int(row["rk4_steps"]) for row in segment_rollouts
                            ),
                            "learned_demag_calls": sum(
                                int(row["learned_demag_calls"])
                                for row in segment_rollouts
                            ),
                            "horizon_s": sum(
                                float(row["horizon_s"])
                                for row in segment_rollouts
                            ),
                            "elapsed_seconds": sum(
                                float(row["elapsed_seconds"])
                                for row in segment_rollouts
                            ),
                            "batch_size": int(prediction.shape[0]),
                        }
                    else:
                        batched_currents = currents.repeat(draw_count, 1, 1, 1)
                        batched_bz = bz.repeat(draw_count)
                        batched_temp = temp.repeat(draw_count)
                        prediction, rollout_meta = simulator.rollout(
                            batched_starts,
                            horizon_s=horizon_s,
                            bz_t=batched_bz,
                            current_z_a_per_m2=batched_currents,
                            temp_k=batched_temp,
                            seed=base_seed + draw_start * 1_000_003,
                            progress_every=args.progress_every,
                        )
                    predictions.append(prediction.cpu().numpy().astype(np.float16))
                    rollout_meta["draw_start"] = draw_start
                    rollout_meta["draw_count"] = draw_count
                    rollout_metas.append(rollout_meta)
                    del prediction, batched_starts
                    if not args.multisegment_rollout:
                        del batched_currents, batched_bz, batched_temp
                values = np.concatenate(predictions, axis=0)
                anchor_indices = np.tile(
                    np.arange(len(record_indices), dtype=np.int16),
                    args.draws_per_anchor,
                )
                if list(values.shape) != expected_shape:
                    raise RuntimeError(f"wrong prediction shape: {values.shape}")
                model_dir.mkdir(parents=True, exist_ok=True)
                temporary = model_path.with_suffix(model_path.suffix + ".tmp")
                with temporary.open("wb") as handle:
                    np.save(handle, values)
                temporary.replace(model_path)
                temporary_anchor = anchor_index_path.with_suffix(
                    anchor_index_path.suffix + ".tmp"
                )
                with temporary_anchor.open("wb") as handle:
                    np.save(handle, anchor_indices)
                temporary_anchor.replace(anchor_index_path)
                _write_json(
                    manifest_path,
                    {
                        "status": "complete",
                        "method": "neuralmag_x5",
                        "implementation_version": IMPLEMENTATION_VERSION,
                        "shape": expected_shape,
                        "dtype": "float16",
                        "dt_s": args.dt_s,
                        "thermal": not args.no_thermal,
                        "checkpoint": str(args.checkpoint),
                        "checkpoint_sha256": _sha256(args.checkpoint),
                        "exact_demag": args.exact_demag,
                        "sot_scale": args.sot_scale,
                        "dmi_scale": args.dmi_scale,
                        "condition_id": condition_id,
                        "requested_horizon_ns": requested_horizon_ns,
                        "reference_mode": prepared["metadata"].get(
                            "reference_mode", "saved_segment_endpoint"
                        ),
                        "multisegment_rollout": args.multisegment_rollout,
                        "realized_horizon_ns": prepared["metadata"]["horizon_ns"],
                        "draws_per_anchor": args.draws_per_anchor,
                        "draw_batch_size": draw_batch_size,
                        "anchor_repeat_index": str(anchor_index_path),
                        "rollouts": rollout_metas,
                        "distribution_semantics": (
                            f"{args.draws_per_anchor} stochastic NeuralMAG-x5 physics "
                            "rollouts from each of five matching MuMax repeat anchors"
                        ),
                    },
                )
                del starts, predictions, values
                if not args.multisegment_rollout:
                    del currents, bz, temp

            if args.samples_only:
                rows.append(
                    {
                        "condition_id": condition_id,
                        "base_id": base,
                        "control_segment_index": segment_index,
                        "draws_per_anchor": args.draws_per_anchor,
                        "status": "samples_complete",
                    }
                )
                print(
                    json.dumps(
                        {
                            "shard": args.shard_index,
                            "position": position,
                            "total": total,
                            "condition_id": condition_id,
                            "status": "samples_complete",
                            "draws_per_anchor": args.draws_per_anchor,
                        }
                    ),
                    flush=True,
                )
                torch.cuda.empty_cache()
                continue

            score_dir = condition_dir / "same_condition_score_4x4_shift16"
            result = run_same_condition(
                argparse.Namespace(
                    condition_dir=condition_dir,
                    output_dir=score_dir,
                    model_samples=model_path,
                    num_model=len(record_indices) * args.draws_per_anchor,
                    num_mumax=len(record_indices),
                    blocks=FORMAL_BLOCKS,
                    shift_radius=FORMAL_SHIFT_RADIUS_PX,
                    reference_chunk=len(record_indices),
                    bootstrap=args.bootstrap,
                    seed=args.seed + int(base) * 1000 + segment_index,
                    device=args.device,
                    progress=False,
                    skip_auxiliary_texture=True,
                )
            )
            primary = result["primary_patch_shift"]
            energy = angular_energy_distance_from_files(
                model_path,
                score_dir / "mumax_targets_f16.npy",
                score_dir / "geometry_mask.npy",
                num_model=5,
                device=args.device,
            )
            rows.append(
                {
                    "condition_id": condition_id,
                    "base_id": base,
                    "control_segment_index": segment_index,
                    "segment_role": prepared["metadata"].get(
                        "segment_role", SEGMENT_LABELS.get(segment_index, "condition")
                    ),
                    "absolute_start_ns": prepared["metadata"]["absolute_start_ns"],
                    "absolute_end_ns": prepared["metadata"]["absolute_end_ns"],
                    "horizon_ns": prepared["metadata"]["horizon_ns"],
                    "symmetric_ratio_score": primary["symmetric_ratio_score"],
                    "symmetric_log_ratio_mae": primary["symmetric_log_ratio_mae"],
                    "kernel_sigma": primary["sigma"],
                    "score_bootstrap_low": result[
                        "primary_symmetric_ratio_score_bootstrap_95ci"
                    ][0],
                    "score_bootstrap_high": result[
                        "primary_symmetric_ratio_score_bootstrap_95ci"
                    ][1],
                    **energy,
                }
            )
            print(
                json.dumps(
                    {
                        "shard": args.shard_index,
                        "position": position,
                        "total": total,
                        "condition_id": condition_id,
                        "score": primary["symmetric_ratio_score"],
                    }
                ),
                flush=True,
            )
            torch.cuda.empty_cache()

    shard_dir = args.output_root / "summary/shards"
    _atomic_csv(shard_dir / f"shard_{args.shard_index:03d}.csv", rows)
    _write_json(
        shard_dir / f"shard_{args.shard_index:03d}.json",
        {
            "status": "complete",
            "implementation_version": IMPLEMENTATION_VERSION,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "conditions": len(rows),
            "elapsed_seconds": time.time() - started,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "dt_s": args.dt_s,
            "thermal": not args.no_thermal,
            "exact_demag": args.exact_demag,
            "sot_scale": args.sot_scale,
            "dmi_scale": args.dmi_scale,
            "draws_per_anchor": args.draws_per_anchor,
            "samples_only": args.samples_only,
            "requested_horizon_ns": requested_horizon_ns,
            "rollout_target_root": args.rollout_target_root,
            "multisegment_rollout": args.multisegment_rollout,
            "reference_mode": (
                "exact_control_multisegment_rollout_endpoint"
                if args.multisegment_rollout
                else (
                    "mumax3_rollout_extension"
                    if args.rollout_target_root is not None
                    else "saved_segment_endpoint"
                )
            ),
            "skipped_conditions": skipped_conditions,
        },
    )


def merge_shards(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    draws_per_anchor: set[int] = set()
    requested_horizons: set[float | None] = set()
    # Preserve the reference provenance in the
    # merged paper artifact, rather than dropping it at the shard boundary.
    rollout_target_roots: set[str | None] = set()
    reference_modes: set[str] = set()
    expected_checkpoint_sha256 = _sha256(args.checkpoint)
    for shard_index in range(args.shard_count):
        shard_root = args.output_root / "summary/shards"
        path = shard_root / f"shard_{shard_index:03d}.csv"
        manifest_path = shard_root / f"shard_{shard_index:03d}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise RuntimeError(f"incomplete NeuralMAG-x5 shard: {manifest_path}")
        if manifest.get("implementation_version") != IMPLEMENTATION_VERSION:
            raise RuntimeError(f"stale NeuralMAG-x5 implementation: {manifest_path}")
        if manifest.get("checkpoint_sha256") != expected_checkpoint_sha256:
            raise RuntimeError(f"mixed NeuralMAG-x5 checkpoint: {manifest_path}")
        if bool(manifest.get("samples_only", False)):
            raise RuntimeError(
                f"cannot build the quality summary from a samples-only shard: {manifest_path}"
            )
        draws_per_anchor.add(int(manifest.get("draws_per_anchor", 1)))
        requested_horizons.add(manifest.get("requested_horizon_ns"))
        rollout_target_roots.add(manifest.get("rollout_target_root"))
        reference_modes.add(
            str(manifest.get("reference_mode", "saved_segment_endpoint"))
        )
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if len(draws_per_anchor) != 1:
        raise RuntimeError(
            f"mixed draws-per-anchor across NeuralMAG-x5 shards: {draws_per_anchor}"
        )
    draw_count = draws_per_anchor.pop()
    if len(requested_horizons) != 1:
        raise RuntimeError(
            "mixed requested horizons across NeuralMAG-x5 shards: "
            f"{requested_horizons}"
        )
    requested_horizon_ns = requested_horizons.pop()
    if len(rollout_target_roots) != 1:
        raise RuntimeError(
            "mixed rollout target roots across NeuralMAG-x5 shards: "
            f"{rollout_target_roots}"
        )
    rollout_target_root = rollout_target_roots.pop()
    if len(reference_modes) != 1:
        raise RuntimeError(
            f"mixed reference modes across NeuralMAG-x5 shards: {reference_modes}"
        )
    reference_mode = reference_modes.pop()
    rows.sort(key=lambda row: (int(row["base_id"]), int(row["control_segment_index"])))
    if len(rows) != args.expected_conditions:
        raise RuntimeError(f"expected {args.expected_conditions} conditions, got {len(rows)}")
    if len({row["condition_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate condition rows across NeuralMAG-x5 shards")
    summary_dir = args.output_root / "summary"
    _atomic_csv(summary_dir / "condition_scores.csv", rows)
    algorithm_version = json.loads(
        (DIST_ROOT / "ALGORITHM_VERSION.json").read_text(encoding="utf-8")
    )
    payload = {
        "status": "complete",
        "method": "neuralmag_x5",
        "implementation_version": IMPLEMENTATION_VERSION,
        "label": "NeuralMAG-x5 (our physics adaptation)",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_step": 0,
        "author_repository": str(NEURALMAG_ROOT),
        "author_commit": _author_commit(),
        "adapter_repository": str(PROJECT_ROOT),
        "implementation": str(DIST_ROOT),
        "algorithm_version": algorithm_version,
        "evaluation_split": "metadata test split",
        "complete_x5_base_groups": len({row["base_id"] for row in rows}),
        "segments": sorted({int(row["control_segment_index"]) for row in rows}),
        "conditions": len(rows),
        "requested_horizon_ns": requested_horizon_ns,
        "rollout_target_root": rollout_target_root,
        "reference_mode": reference_mode,
        "multisegment_rollout": (
            reference_mode == "exact_control_multisegment_rollout_endpoint"
        ),
        "mumax_repeats_per_condition": 5,
        "model_predictions_per_condition": 5 * draw_count,
        "model_draws_per_anchor": draw_count,
        "distribution_semantics": (
            f"{draw_count} stochastic NeuralMAG-x5 physics rollout(s) from each "
            "of five matching MuMax repeat anchors; no ground-truth selection"
        ),
        "x5_physics": [
            "released NeuralMAG learned demagnetizing field",
            "exchange",
            "interfacial DMI",
            "uniaxial anisotropy",
            "Zeeman field",
            "spatial Lambda=1 Slonczewski torque",
            "Brown thermal field",
            "fixed-step RK4",
        ],
        "score": _aggregate(rows, iterations=args.bootstrap, seed=args.seed + 999),
        "angular_energy_distance": aggregate_clustered_mean(
            rows,
            "angular_energy_distance_deg",
            iterations=args.bootstrap,
            seed=args.seed + 1_999,
        ),
        "paired_angular_error": aggregate_clustered_mean(
            rows,
            "paired_model_mumax_mean_deg",
            iterations=args.bootstrap,
            seed=args.seed + 2_999,
        ),
    }
    _write_json(summary_dir / "run_summary.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


def benchmark(args: argparse.Namespace) -> None:
    if (not args.multisegment_rollout) and args.horizon_ns <= 0.0:
        raise ValueError("horizon-ns must be positive")
    if args.timing_segment_durations_ns is not None and not args.multisegment_rollout:
        raise ValueError(
            "--timing-segment-durations-ns requires --multisegment-rollout"
        )
    device = torch.device(args.device)
    dataset = _load_test_dataset(args.config)
    base, record_indices = next(
        iter(
            _complete_test_groups(
                dataset,
                repeat_dataset=DEFAULT_REPEAT_DATASET,
                repeats_per_group=5,
            ).items()
        )
    )
    if args.multisegment_rollout:
        if args.rollout_target_root is not None:
            raise ValueError(
                "--multisegment-rollout and --rollout-target-root are mutually exclusive"
            )
        prepared = _prepare_multisegment_rollout_condition(
            dataset,
            base=base,
            record_indices=record_indices,
            condition_dir=args.output.parent / "timing_condition",
        )
        if args.timing_segment_durations_ns is not None:
            prepared = _override_multisegment_timing_durations(
                prepared, dataset, args.timing_segment_durations_ns
            )
    elif args.rollout_target_root is not None:
        target_root = args.rollout_target_root.resolve()
        if not (target_root / "manifest.json").is_file():
            raise FileNotFoundError(target_root / "manifest.json")
        # Timing needs no target values, but it
        # must construct the same exact-anchor, constant-zero-current 5-ns
        # input as quality instead of searching the finite source trajectory
        # for a saved endpoint that was never requested there.
        prepared = _prepare_rollout_condition(
            dataset,
            base=base,
            record_indices=record_indices,
            condition_dir=args.output.parent / "timing_condition",
            target_root=target_root,
            horizon_ns=args.horizon_ns,
        )
    else:
        prepared = _prepare_condition(
            dataset,
            base=base,
            record_indices=record_indices,
            segment_index=2,
            condition_dir=args.output.parent / "timing_condition",
            duration_ns=args.horizon_ns,
        )
    if args.multisegment_rollout:
        raw_segment_batches = _multisegment_condition_batches(
            prepared, dataset, record_indices, device
        )
        starts = raw_segment_batches[0][0]
    else:
        starts, currents, bz, temp, _horizon = _condition_batch(
            prepared, dataset, record_indices, device
        )
    if args.repeat_conditions:
        batch = int(args.batch_size)
        indices = torch.arange(batch, device=device) % starts.shape[0]
    else:
        batch = min(args.batch_size, starts.shape[0])
        indices = torch.arange(batch, device=device)
    starts = starts.index_select(0, indices)
    if args.multisegment_rollout:
        segment_batches = [
            (
                truth_starts.index_select(0, indices),
                segment_currents.index_select(0, indices),
                segment_bz.index_select(0, indices),
                segment_temp.index_select(0, indices),
                segment_horizon_s,
            )
            for (
                truth_starts,
                segment_currents,
                segment_bz,
                segment_temp,
                segment_horizon_s,
            ) in raw_segment_batches
        ]
    else:
        currents = currents.index_select(0, indices)
        bz = bz.index_select(0, indices)
        temp = temp.index_select(0, indices)
    model = _load_model(device, args.checkpoint)
    simulator = NeuralMAGX5Simulator(
        model,
        dt_s=args.dt_s,
        thermal=not args.no_thermal,
        sot_scale=args.sot_scale,
        dmi_scale=args.dmi_scale,
    )
    runs: list[float] = []
    metas: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    for repeat in range(args.repeats):
        if args.multisegment_rollout:
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + repeat * 1_000_003)
            _prediction = starts
            segment_metas: list[dict[str, Any]] = []
            for segment_position, (
                _truth_starts,
                segment_currents,
                segment_bz,
                segment_temp,
                segment_horizon_s,
            ) in enumerate(segment_batches):
                _prediction, segment_meta = simulator.rollout(
                    _prediction,
                    horizon_s=segment_horizon_s,
                    bz_t=segment_bz,
                    current_z_a_per_m2=segment_currents,
                    temp_k=segment_temp,
                    seed=args.seed + repeat * 1_000_003,
                    generator=generator,
                    progress_every=args.progress_every,
                )
                segment_meta["segment_position"] = segment_position
                segment_metas.append(segment_meta)
            meta = {
                "protocol": "piecewise-control RK4 with continuous thermal RNG",
                "segments": segment_metas,
                "rk4_steps": sum(int(row["rk4_steps"]) for row in segment_metas),
                "learned_demag_calls": sum(
                    int(row["learned_demag_calls"]) for row in segment_metas
                ),
                "horizon_s": sum(float(row["horizon_s"]) for row in segment_metas),
                "elapsed_seconds": sum(
                    float(row["elapsed_seconds"]) for row in segment_metas
                ),
                "batch_size": batch,
            }
        else:
            _prediction, meta = simulator.rollout(
                starts,
                horizon_s=float(args.horizon_ns) * 1.0e-9,
                bz_t=bz,
                current_z_a_per_m2=currents,
                temp_k=temp,
                seed=args.seed + repeat * 1_000_003,
                progress_every=args.progress_every,
            )
        runs.append(1000.0 * float(meta["elapsed_seconds"]) / batch)
        metas.append(meta)
        print({"repeat": repeat, "ms_per_sample": runs[-1]}, flush=True)
    payload = {
        "status": "complete",
        "schema": "neuralmag_x5_complete_horizon_timing_v2",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "method": "neuralmag_x5",
        "implementation_version": IMPLEMENTATION_VERSION,
        "label": "NeuralMAG-x5 (our physics adaptation)",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "author_repository": str(NEURALMAG_ROOT),
        "author_commit": _author_commit(),
        "device": torch.cuda.get_device_name(device),
        "precision": "fp32",
        "batch_size": batch,
        "batch_scaling_input": (
            "five matched held-out conditions cycled to fill the requested batch"
            if args.repeat_conditions
            else "at most five distinct matched held-out conditions"
        ),
        "repeats": args.repeats,
        "dt_s": args.dt_s,
        "thermal": not args.no_thermal,
        "sot_scale": args.sot_scale,
        "dmi_scale": args.dmi_scale,
        "horizon_ns": (
            float(prepared["metadata"]["composed_model_horizon_ns"])
            if args.multisegment_rollout
            else float(args.horizon_ns)
        ),
        "multisegment_rollout": args.multisegment_rollout,
        "reference_mode": prepared["metadata"].get(
            "reference_mode", "saved_segment_endpoint"
        ),
        # Embed the selected condition in
        # every shard so downstream aggregation can reject timing outside the
        # frozen paper-quality population without relying on a sidecar file.
        "timing_condition": {
            "base_id": str(prepared["metadata"]["base_id"]).zfill(4),
            "control_segment_index": int(prepared["metadata"]["control_segment_index"]),
            "control_segment_indices": prepared["metadata"].get(
                "control_segment_indices"
            ),
            "segment_role": str(prepared["metadata"]["segment_role"]),
            "requested_horizon_ns": prepared["metadata"][
                "requested_horizon_ns"
            ],
            "run_ids": [
                str(row["run_id"]) for row in prepared["metadata"]["repeats"]
            ],
        },
        "mean_ms_per_sample": statistics.fmean(runs),
        "std_ms_per_sample": statistics.stdev(runs) if len(runs) > 1 else 0.0,
        "runs_ms_per_sample": runs,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "rollout_metadata": metas,
        "timing_scope": (
            "complete x5 physics RK4 rollout over exact protocol control "
            "segments; excludes data loading and transfer"
            if args.multisegment_rollout
            else "complete x5 physics RK4 rollout; excludes data loading and transfer"
        ),
    }
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, required=True)
    common.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    common.add_argument("--device", default="cuda")
    common.add_argument("--seed", type=int, default=208_160_830)
    common.add_argument("--dt-s", type=float, default=1.0e-13)
    common.add_argument("--no-thermal", action="store_true")
    common.add_argument("--sot-scale", type=float, default=1.0)
    common.add_argument("--dmi-scale", type=float, default=1.0)
    common.add_argument("--exact-demag", action="store_true")
    common.add_argument("--progress-every", type=int, default=1000)

    validate_parser = commands.add_parser("validate-demag", parents=[common])
    validate_parser.add_argument("--output", type=Path, required=True)
    validate_parser.add_argument("--samples", type=int, default=2)
    validate_parser.set_defaults(func=validate_demag)

    finetune_parser = commands.add_parser("finetune-demag", parents=[common])
    finetune_parser.add_argument("--output", type=Path, required=True)
    finetune_parser.add_argument("--steps", type=int, default=2000)
    finetune_parser.add_argument("--batch-size", type=int, default=2)
    finetune_parser.add_argument("--lr", type=float, default=1.0e-4)
    finetune_parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    finetune_parser.add_argument("--physical-weight", type=float, default=0.05)
    finetune_parser.add_argument("--state-noise-std", type=float, default=0.0)
    finetune_parser.add_argument("--validation-samples", type=int, default=5)
    finetune_parser.add_argument("--log-every", type=int, default=50)
    finetune_parser.set_defaults(func=finetune_demag)

    score_parser = commands.add_parser("score-shard", parents=[common])
    score_parser.add_argument("--output-root", type=Path, required=True)
    score_parser.add_argument("--shard-index", type=int, required=True)
    score_parser.add_argument("--shard-count", type=int, required=True)
    score_parser.add_argument("--segments", type=int, nargs="+", default=(1, 2))
    score_parser.add_argument(
        "--duration-ns",
        type=float,
        default=None,
        help=(
            "Override each segment endpoint by this exact within-segment duration; "
            "unsupported conditions are skipped."
        ),
    )
    score_parser.add_argument(
        "--rollout-target-root",
        type=Path,
        default=None,
        help=(
            "[5NS-ROLLOUT FIX 2026-08-29] Score the 5-ns zero-current "
            "NeuralMAG rollout against generated MuMax3 continuation targets."
        ),
    )
    score_parser.add_argument(
        "--multisegment-rollout",
        action="store_true",
        help=(
            "Integrate the saved drive and post-relax control segments in "
            "sequence, carrying both magnetization and thermal RNG state "
            "through the physical control boundary."
        ),
    )
    score_parser.add_argument("--max-groups", type=int)
    score_parser.add_argument("--bootstrap", type=int, default=5000)
    score_parser.add_argument("--draws-per-anchor", type=int, default=1)
    score_parser.add_argument(
        "--draw-batch-size",
        type=int,
        default=0,
        help="draw groups evaluated together; 0 batches all draws",
    )
    score_parser.add_argument("--samples-only", action="store_true")
    score_parser.add_argument("--force", action="store_true")
    score_parser.set_defaults(func=score_shard)

    merge_parser = commands.add_parser("merge")
    merge_parser.add_argument("--output-root", type=Path, required=True)
    merge_parser.add_argument("--checkpoint", type=Path, required=True)
    merge_parser.add_argument("--shard-count", type=int, required=True)
    merge_parser.add_argument("--expected-conditions", type=int, default=66)
    merge_parser.add_argument("--bootstrap", type=int, default=5000)
    merge_parser.add_argument("--seed", type=int, default=208_160_830)
    merge_parser.set_defaults(func=merge_shards)

    benchmark_parser = commands.add_parser("benchmark", parents=[common])
    benchmark_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser.add_argument("--batch-size", type=int, default=5)
    # A 5-ns reference can be generated by a
    # solver rollout even when it is absent from the original saved frames.
    benchmark_parser.add_argument("--horizon-ns", type=float, default=5.0)
    benchmark_parser.add_argument(
        "--multisegment-rollout",
        action="store_true",
        help=(
            "Ignore --horizon-ns and time the exact-control drive->post-relax "
            "piecewise-control path with continuous RK4/thermal state."
        ),
    )
    benchmark_parser.add_argument(
        "--timing-segment-durations-ns",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Timing-only durations for --multisegment-rollout, for example "
            "2 3 for a standardized two-segment 5-ns workload."
        ),
    )
    benchmark_parser.add_argument(
        "--rollout-target-root",
        type=Path,
        default=None,
        help=(
            "Use generated MuMax3 continuation metadata to construct an "
            "unsaved constant-control horizon such as the matched 5-ns task."
        ),
    )
    benchmark_parser.add_argument("--repeats", type=int, default=1)
    benchmark_parser.add_argument(
        "--repeat-conditions",
        action="store_true",
        help="Cycle the five matched timing conditions to form larger batches.",
    )
    benchmark_parser.set_defaults(func=benchmark)

    args = parser.parse_args()
    if args.command != "merge":
        if not torch.cuda.is_available():
            raise RuntimeError("NeuralMAG-x5 requires CUDA")
        _configure_cuda()
        seed_everything(args.seed)
    args.func(args)
    gc.collect()


if __name__ == "__main__":
    main()
