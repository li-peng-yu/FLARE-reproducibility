from __future__ import annotations

import argparse
import json
import os
import math
import re
import shutil
from pathlib import Path


HORIZONS_NS = (0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_SOURCE_ROOT = Path(os.environ.get("FLARE_DATASET_ROOT", str(Path(__file__).resolve().parents[2] / "FLARE_dataset"))) / "primary_x5/skx_bt_1000base_x5_sharedrelax_20260805"


def _rewrite_scalar(mx3: str, name: str, value: float) -> str:
    replacement = f"{name} = {value:.12g}"
    out, count = re.subn(
        rf"^\s*{re.escape(name)}\s*=\s*[0-9.eE+-]+\s*$",
        replacement,
        mx3,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError(f"failed to rewrite {name}")
    return out


def _insert_fixdt(mx3: str, fix_dt_s: float) -> str:
    replacement = f"FixDt = {fix_dt_s:.12g}"
    out, count = re.subn(
        r"^\s*FixDt\s*=\s*[0-9.eE+-]+\s*$",
        replacement,
        mx3,
        count=1,
        flags=re.MULTILINE,
    )
    if count == 1:
        return out
    out, count = re.subn(
        r"^(\s*MaxErr\s*=\s*[0-9.eE+-]+\s*)$",
        rf"\1\n{replacement}",
        mx3,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError("failed to insert FixDt after MaxErr")
    return out


def _rewrite_initial_state(mx3: str) -> str:
    out, count = re.subn(
        r'^\s*m\.LoadFile\("[^"]+"\)\s*$',
        'm.LoadFile("m_initial.ovf")',
        mx3,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError("failed to rewrite the shared-relaxation checkpoint path")
    return out


def _rewrite_horizon(mx3: str, pulse_duration_ns: float, horizon_ns: float) -> str:
    pattern = re.compile(
        r"^run\(([0-9.eE+-]+)\)\s*\n"
        r"J\s*=\s*vector\(0,\s*0,\s*0\)\s*\n"
        r"run\(([0-9.eE+-]+)\)\s*$",
        flags=re.MULTILINE,
    )
    matches = list(pattern.finditer(mx3))
    if len(matches) != 1:
        raise RuntimeError(f"expected one pulse/relax schedule, found {len(matches)}")
    source_pulse_s = float(matches[0].group(1))
    expected_pulse_s = float(pulse_duration_ns) * 1e-9
    if not math.isclose(source_pulse_s, expected_pulse_s, rel_tol=0.0, abs_tol=5e-15):
        raise RuntimeError(
            f"run.mx3 pulse {source_pulse_s} does not match metadata {expected_pulse_s}"
        )

    horizon_s = float(horizon_ns) * 1e-9
    driven_s = min(horizon_s, expected_pulse_s)
    relaxation_s = max(0.0, horizon_s - expected_pulse_s)
    lines = [f"run({driven_s:.12g})"]
    if relaxation_s > 0.0:
        lines.extend(
            [
                "J = vector(0, 0, 0)",
                f"run({relaxation_s:.12g})",
            ]
        )
    return pattern.sub("\n".join(lines), mx3, count=1)


def _discover_split_runs(source_root: Path, split: str) -> list[tuple[Path, dict]]:
    by_base: dict[int, tuple[Path, dict]] = {}
    for params_path in sorted(source_root.glob("r_*/params.json")):
        params = json.loads(params_path.read_text())
        if str(params.get("split")) != split:
            continue
        repeat = int(params.get("thermal_repeat_index", params.get("repeat_index", 0)))
        if repeat != 0:
            continue
        base = int(params["base_index"])
        if base in by_base:
            raise RuntimeError(f"duplicate repeat-zero run for base {base}")
        run_dir = params_path.parent
        required = [
            run_dir / "run.mx3",
            run_dir / "run.out" / "m_initial.ovf",
            run_dir / "current_protocol.json",
            run_dir / "current_control.csv",
        ]
        if all(path.is_file() for path in required):
            by_base[base] = (run_dir, params)
    if not by_base:
        raise RuntimeError(f"no complete repeat-zero {split} runs under {source_root}")
    return [by_base[base] for base in sorted(by_base)]


def _quantile_select(candidates: list[tuple[Path, dict]], count: int) -> list[tuple[Path, dict]]:
    if count <= 0 or count > len(candidates):
        raise ValueError(f"cannot select {count} structures from {len(candidates)} candidates")
    if count == 1:
        return [candidates[len(candidates) // 2]]
    indices = [round(index * (len(candidates) - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError(f"quantile selection produced duplicate indices: {indices}")
    return [candidates[index] for index in indices]


def _oommf_reference(selected: list[tuple[Path, dict]]) -> tuple[Path, dict]:
    candidates = [item for item in selected if math.isclose(float(item[1]["temp_k"]), 30.0)]
    if not candidates:
        candidates = selected
    return min(
        candidates,
        key=lambda item: (
            abs(float(item[1]["bz_mt"]) - 16.0),
            abs(float(item[1]["pulse_duration_ns"]) - 0.5),
            int(item[1]["base_index"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare fixed-step mumax3 timings from held-out corrected x5 cases."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("graph/results/x5_stage1_compute_cost_20260814/mumax_fixdt_1e-14"),
    )
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--num-structures", type=int, default=8)
    parser.add_argument("--horizons", type=float, nargs="+", default=list(HORIZONS_NS))
    parser.add_argument("--maxdt", type=float, default=1e-12)
    parser.add_argument("--maxerr", type=float, default=1e-6)
    parser.add_argument("--fixdt", type=float, default=1e-14)
    args = parser.parse_args()

    if (args.out_dir / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite existing benchmark: {args.out_dir}")

    candidates = _discover_split_runs(args.source_root, args.split)
    selected = _quantile_select(candidates, args.num_structures)
    oommf_run, oommf_params = _oommf_reference(selected)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs_root = args.out_dir / "runs"
    manifest: list[dict[str, object]] = []
    runlist: list[str] = []

    for structure_index, (source_run, params) in enumerate(selected, start=1):
        original = (source_run / "run.mx3").read_text()
        original = _rewrite_scalar(original, "MaxDt", args.maxdt)
        original = _rewrite_scalar(original, "MaxErr", args.maxerr)
        original = _insert_fixdt(original, args.fixdt)
        original = _rewrite_initial_state(original)
        for horizon_ns in args.horizons:
            run_dir = (
                runs_root
                / f"h_{horizon_ns:g}ns"
                / f"sample_{structure_index:02d}_{source_run.name}"
            )
            run_dir.mkdir(parents=True, exist_ok=False)
            shutil.copy2(source_run / "run.out" / "m_initial.ovf", run_dir / "m_initial.ovf")
            prepared = _rewrite_horizon(
                original,
                pulse_duration_ns=float(params["pulse_duration_ns"]),
                horizon_ns=float(horizon_ns),
            )
            (run_dir / "run.mx3").write_text(prepared)
            resolved = str(run_dir.resolve())
            runlist.append(resolved)
            manifest.append(
                {
                    "run_dir": resolved,
                    "source_run": str(source_run.resolve()),
                    "run_id": source_run.name,
                    "base_index": int(params["base_index"]),
                    "thermal_repeat_index": int(params["thermal_repeat_index"]),
                    "split": args.split,
                    "structure_index": structure_index,
                    "horizon_ns": float(horizon_ns),
                    "temperature_k": float(params["temp_k"]),
                    "bz_mt": float(params["bz_mt"]),
                    "pulse_duration_ns": float(params["pulse_duration_ns"]),
                    "max_dt_s": args.maxdt,
                    "max_err": args.maxerr,
                    "fix_dt_s": args.fixdt,
                }
            )

    selection = {
        "schema": "x5_compute_cost_selection_v1",
        "source_root": str(args.source_root.resolve()),
        "split": args.split,
        "candidate_base_conditions": len(candidates),
        "selection": "even quantiles of sorted held-out base_index; thermal repeat 0",
        "mumax_structures": [
            {
                "source_run": str(run.resolve()),
                "base_index": int(params["base_index"]),
                "temperature_k": float(params["temp_k"]),
                "bz_mt": float(params["bz_mt"]),
                "pulse_duration_ns": float(params["pulse_duration_ns"]),
            }
            for run, params in selected
        ],
        "oommf_structure": {
            "source_run": str(oommf_run.resolve()),
            "base_index": int(oommf_params["base_index"]),
            "temperature_k": float(oommf_params["temp_k"]),
            "bz_mt": float(oommf_params["bz_mt"]),
            "pulse_duration_ns": float(oommf_params["pulse_duration_ns"]),
            "selection": "30 K selected case nearest Bz=16 mT and pulse=0.5 ns",
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out_dir / "run_list.txt").write_text("\n".join(runlist) + "\n")
    (args.out_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (args.out_dir / "oommf_source_run_list.txt").write_text(str(oommf_run.resolve()) + "\n")
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "runs": len(runlist),
                "structures": len(selected),
                "horizons_ns": args.horizons,
                "oommf_source_run": str(oommf_run),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
