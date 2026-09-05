from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from distribution_score.calibration import run
from distribution_score.manifest import load_manifest, result_directory


class ManifestAndAggregationTests(unittest.TestCase):
    def test_generic_groups_and_result_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            conditions = []
            ranks = {
                "a0": ("group_a", 0.25),
                "a1": ("group_a", 0.75),
                "b0": ("group_b", 0.20),
                "b1": ("group_b", 0.80),
            }
            for condition_id, (group, rank) in ranks.items():
                conditions.append(
                    {
                        "id": condition_id,
                        "group": group,
                        "condition_dir": f"data/{condition_id}",
                    }
                )
                result_dir = root / "results" / condition_id
                result_dir.mkdir(parents=True)
                (result_dir / "probability_rank.json").write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "package_version": "0.2.0",
                            "algorithm_version": "0.2.0",
                            "distance_id": "full_vector_patch_shift_4x4_shift16_symmetric",
                            "probability_rank_u": rank,
                            "truth_model_density": 0.5,
                            "kernel_sigma": 0.1,
                        }
                    ),
                    encoding="utf-8",
                )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "groups": [
                            {"name": "group_a", "label": "Group A"},
                            {"name": "group_b", "label": "Group B"},
                        ],
                        "conditions": conditions,
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_manifest(manifest_path)
            self.assertEqual(manifest.group_order, ("group_a", "group_b"))
            self.assertEqual(manifest.conditions[0].condition_dir, root / "data" / "a0")
            self.assertEqual(
                result_directory(manifest.conditions[0], None),
                root / "data" / "a0" / "probability_calibration_4x4_shift16",
            )

            output_dir = root / "summary"
            result = run(
                argparse.Namespace(
                    manifest=manifest_path,
                    results_root=root / "results",
                    output_dir=output_dir,
                    bootstrap=20,
                    seed=7,
                )
            )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["n_statistical_groups"], 2)
            self.assertTrue((output_dir / "calibration_summary.json").is_file())
            self.assertTrue((output_dir / "calibration_curves_by_group.png").is_file())

    def test_aggregation_rejects_old_distance_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "groups": ["group_a"],
                        "conditions": [
                            {
                                "id": "a0",
                                "group": "group_a",
                                "condition_dir": "data/a0",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result_dir = root / "results" / "a0"
            result_dir.mkdir(parents=True)
            (result_dir / "probability_rank.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "package_version": "0.1.0",
                        "algorithm_version": "0.1.0",
                        "distance_id": "full_vector_patch_shift_8x8_shift32_symmetric",
                        "probability_rank_u": 0.5,
                        "truth_model_density": 0.5,
                        "kernel_sigma": 0.1,
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "summary"
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                run(
                    argparse.Namespace(
                        manifest=manifest_path,
                        results_root=root / "results",
                        output_dir=output_dir,
                        bootstrap=20,
                        seed=7,
                        blocks=4,
                        shift_radius=16,
                    )
                )
            self.assertTrue((output_dir / "incompatible_results.json").is_file())


if __name__ == "__main__":
    unittest.main()
