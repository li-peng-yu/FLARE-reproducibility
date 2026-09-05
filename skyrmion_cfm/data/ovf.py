from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_HEADER_RE = re.compile(rb"#\s*([^:\n]+):\s*([^\n\r]+)")


@dataclass(frozen=True)
class OVFHeader:
    xnodes: int
    ynodes: int
    znodes: int
    valuedim: int
    data_mode: str
    dtype: str


def _parse_header(raw: bytes) -> tuple[OVFHeader, int]:
    marker = b"# Begin: Data"
    data_pos = raw.find(marker)
    if data_pos < 0:
        raise ValueError("OVF file does not contain a data section")
    line_end = raw.find(b"\n", data_pos)
    if line_end < 0:
        raise ValueError("OVF data marker is not terminated")
    data_line = raw[data_pos:line_end].decode("ascii", errors="replace").strip()
    header_raw = raw[:line_end]
    values: dict[str, str] = {}
    for match in _HEADER_RE.finditer(header_raw):
        key = match.group(1).decode("ascii", errors="ignore").strip().lower()
        val = match.group(2).decode("ascii", errors="ignore").strip()
        values[key] = val
    mode = "text"
    dtype = "float32"
    if "Binary 4" in data_line:
        mode = "binary"
        dtype = "float32"
    elif "Binary 8" in data_line:
        mode = "binary"
        dtype = "float64"
    elif "Text" in data_line:
        mode = "text"
    header = OVFHeader(
        xnodes=int(values["xnodes"]),
        ynodes=int(values["ynodes"]),
        znodes=int(values.get("znodes", "1")),
        valuedim=int(values.get("valuedim", "3")),
        data_mode=mode,
        dtype=dtype,
    )
    return header, line_end + 1


def read_ovf(path: str | Path, dtype: np.dtype = np.float32) -> np.ndarray:
    """Read an OVF vector field as a CHW numpy array.

    MuMax3 writes OVF2 binary files with a 4- or 8-byte check value after the
    data marker. The magnetization data are stored as x-fastest vectors.
    """
    raw = Path(path).read_bytes()
    header, start = _parse_header(raw)
    count = header.xnodes * header.ynodes * header.znodes * header.valuedim
    if header.data_mode == "binary":
        itemsize = np.dtype(header.dtype).itemsize
        start += itemsize
        arr = np.frombuffer(raw, dtype="<" + np.dtype(header.dtype).str[1:], count=count, offset=start)
    else:
        end = raw.find(b"# End: Data", start)
        payload = raw[start:end if end > 0 else None]
        arr = np.fromstring(payload.decode("ascii", errors="ignore"), sep=" ", dtype=np.float64)
        if arr.size < count:
            raise ValueError(f"OVF text payload has {arr.size} values, expected {count}")
        arr = arr[:count]
    field = arr.reshape(header.znodes, header.ynodes, header.xnodes, header.valuedim)
    if header.znodes != 1:
        field = field.mean(axis=0)
    else:
        field = field[0]
    return np.asarray(field.transpose(2, 0, 1), dtype=dtype)


def read_ovf_header(path: str | Path) -> OVFHeader:
    raw = Path(path).read_bytes()
    header, _ = _parse_header(raw)
    return header


def write_ovf_like(
    template_path: str | Path,
    output_path: str | Path,
    field: np.ndarray,
) -> None:
    """Write a CHW field while preserving a MuMax3 OVF template header.

    Stage-2 physical replay must hand a model prediction back to MuMax3
    without changing the grid, cell geometry, coordinate convention, or OVF
    encoding.  Reusing the exact bytes surrounding the template payload is
    safer than synthesizing a second header implementation.  The x5 datasets
    use one-layer OVF2 Binary 4 files; Binary 8 is supported as well.
    """

    template = Path(template_path)
    output = Path(output_path)
    raw = template.read_bytes()
    header, data_start = _parse_header(raw)
    if header.data_mode != "binary":
        raise ValueError(
            "write_ovf_like requires a binary OVF template, "
            f"got {header.data_mode!r}: {template}"
        )
    if header.znodes != 1:
        raise ValueError(
            "write_ovf_like currently requires znodes=1, "
            f"got {header.znodes}: {template}"
        )

    array = np.asarray(field)
    expected_shape = (header.valuedim, header.ynodes, header.xnodes)
    if array.shape != expected_shape:
        raise ValueError(
            f"OVF field shape {array.shape} does not match template "
            f"{expected_shape}: {template}"
        )
    if not np.isfinite(array).all():
        raise ValueError("OVF field contains non-finite values")

    dtype = np.dtype(header.dtype).newbyteorder("<")
    itemsize = dtype.itemsize
    value_count = int(np.prod(expected_shape, dtype=np.int64))
    payload_start = data_start + itemsize  # Preserve the OVF check value.
    payload_end = payload_start + value_count * itemsize
    if payload_end > len(raw):
        raise ValueError(f"OVF template payload is truncated: {template}")
    payload = (
        np.asarray(array.transpose(1, 2, 0), dtype=dtype)
        .reshape(-1)
        .tobytes(order="C")
    )
    temporary = output.with_name(f".{output.name}.tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(raw[:payload_start] + payload + raw[payload_end:])
    temporary.replace(output)
