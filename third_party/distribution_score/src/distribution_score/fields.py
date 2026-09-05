"""Magnetization-field normalization and minimal MuMax OVF input."""

from __future__ import annotations

from pathlib import Path

import numpy as np


CHECK_FLOAT = np.float32(1234567.0)


def read_ovf(path: str | Path) -> np.ndarray:
    """Read a MuMax OVF Binary 4 file as ``(valuedim, ny, nx)``."""

    path = Path(path)
    raw = path.read_bytes()
    marker = b"# Begin: Data Binary 4"
    marker_position = raw.find(marker)
    if marker_position < 0:
        raise ValueError(f"not an OVF Binary 4 file: {path}")

    header = raw[:marker_position].decode("utf-8", errors="replace")
    metadata: dict[str, str] = {}
    for line in header.splitlines():
        if line.startswith("#") and ":" in line:
            key, value = line[1:].split(":", 1)
            metadata[key.strip().lower()] = value.strip()

    nx = int(metadata["xnodes"])
    ny = int(metadata["ynodes"])
    nz = int(metadata["znodes"])
    valuedim = int(metadata["valuedim"])
    data_start = raw.find(b"\n", marker_position) + 1
    values = np.frombuffer(raw, dtype="<f4", offset=data_start)
    if values.size < 1 or not np.isclose(values[0], CHECK_FLOAT):
        raise ValueError(f"bad OVF check float: {path}")

    expected = nx * ny * nz * valuedim
    payload = values[1 : 1 + expected]
    if payload.size != expected:
        raise ValueError(f"truncated OVF payload: {path}")
    first_plane = payload.reshape(nz, ny, nx, valuedim)[0]
    return np.moveaxis(first_plane, -1, 0).astype(np.float32, copy=False)


def normalize_field(array: np.ndarray) -> np.ndarray:
    """Return one normalized magnetization field in ``(3, H, W)`` layout."""

    field = np.asarray(array, dtype=np.float32)
    if field.ndim != 3:
        raise ValueError(f"expected one three-dimensional field, got {field.shape}")
    if field.shape[0] != 3 and field.shape[-1] == 3:
        field = np.moveaxis(field, -1, 0)
    if field.shape[0] != 3:
        raise ValueError(f"expected three magnetization components, got {field.shape}")
    norm = np.linalg.norm(field, axis=0, keepdims=True)
    return np.divide(field, norm, out=np.zeros_like(field), where=norm > 1.0e-8)


def normalize_batch(array: np.ndarray) -> np.ndarray:
    """Return normalized magnetization fields in ``(N, 3, H, W)`` layout."""

    fields = np.asarray(array, dtype=np.float32)
    if fields.ndim != 4:
        raise ValueError(f"expected a batch of fields, got {fields.shape}")
    if fields.shape[1] != 3 and fields.shape[-1] == 3:
        fields = np.moveaxis(fields, -1, 1)
    if fields.shape[1] != 3:
        raise ValueError(f"expected three magnetization components, got {fields.shape}")
    norm = np.linalg.norm(fields, axis=1, keepdims=True)
    return np.divide(fields, norm, out=np.zeros_like(fields), where=norm > 1.0e-8)
