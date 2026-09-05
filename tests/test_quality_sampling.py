from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from skyrmion_cfm.data.fixed_time import FixedTimePairDataset
from skyrmion_cfm.data.quality_sampling import (
    build_segment_quality_cache,
    classify_pair_quality,
    load_segment_quality_cache,
    normalize_quality_sampling_config,
    pair_quality_metrics,
    summarize_segment,
)


def _uniform_pair(size: int = 16) -> dict[str, torch.Tensor | str]:
    initial = torch.zeros((3, size, size), dtype=torch.float32)
    initial[2] = -1.0
    return {
        "m_init": initial,
        "m_t": initial.clone(),
        "defect_field": torch.ones((1, size, size), dtype=torch.float32),
        "control_segment_index": torch.tensor(0),
        "run_id": "run-0",
    }


class QualityMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = normalize_quality_sampling_config({"enabled": True})

    def test_unchanged_pair_is_static(self) -> None:
        metrics = pair_quality_metrics(_uniform_pair(), block_factor=4)
        self.assertAlmostEqual(metrics["raw_angle_deg"], 0.0, places=5)
        self.assertAlmostEqual(metrics["coherent_angle_deg"], 0.0, places=5)
        self.assertEqual(classify_pair_quality(metrics, self.cfg), "static")

    def test_pixel_noise_is_noise_only(self) -> None:
        sample = _uniform_pair()
        target = sample["m_t"].clone()
        rows = torch.arange(target.shape[-2])[:, None]
        cols = torch.arange(target.shape[-1])[None, :]
        target[2] = torch.where((rows + cols) % 2 == 0, 1.0, -1.0)
        sample["m_t"] = target
        metrics = pair_quality_metrics(sample, block_factor=4)
        self.assertGreater(metrics["target_neighbor_angle_deg"], 170.0)
        self.assertLess(metrics["target_block_resultant"], 0.01)
        self.assertEqual(classify_pair_quality(metrics, self.cfg), "noise_only")

    def test_coherent_domain_change_is_informative(self) -> None:
        sample = _uniform_pair()
        target = sample["m_t"].clone()
        target[2, :, target.shape[-1] // 2 :] = 1.0
        sample["m_t"] = target
        metrics = pair_quality_metrics(sample, block_factor=4)
        self.assertGreater(metrics["coherent_angle_deg"], 30.0)
        self.assertEqual(classify_pair_quality(metrics, self.cfg), "informative")

    def test_segment_p90_preserves_a_sparse_event(self) -> None:
        rows = [
            {
                "category": "static",
                "raw_angle_deg": 1.0,
                "coherent_angle_deg": 1.0,
            }
            for _ in range(3)
        ]
        rows.append(
            {
                "category": "informative",
                "raw_angle_deg": 60.0,
                "coherent_angle_deg": 55.0,
            }
        )
        summary = summarize_segment(rows, self.cfg)
        self.assertEqual(summary["category"], "informative")
        self.assertGreater(summary["coherent_angle_p90_deg"], 5.0)


class QualityDatasetSamplingTests(unittest.TestCase):
    def _dataset(self, samples: list[dict]) -> FixedTimePairDataset:
        dataset = FixedTimePairDataset.__new__(FixedTimePairDataset)
        dataset.quality_sampling = normalize_quality_sampling_config(
            {
                "enabled": True,
                "original_mix_probability": 0.0,
                "max_attempts": len(samples),
                "pair": {
                    "static_keep_probability": 0.0,
                    "noise_keep_probability": 0.0,
                },
            }
        )
        dataset._quality_segment_entries = {}
        dataset._draw_standard_pair_spec = lambda rng: (0, 0, 0)
        pending = list(samples)

        def build_sample(*args, **kwargs):
            return pending.pop(0)

        dataset._build_sample = build_sample
        return dataset

    def test_rejects_static_pair_then_accepts_informative_pair(self) -> None:
        static = _uniform_pair()
        informative = _uniform_pair()
        target = informative["m_t"].clone()
        target[2, :, target.shape[-1] // 2 :] = 1.0
        informative["m_t"] = target
        dataset = self._dataset([static, informative])

        sample = dataset._build_quality_sample(np.random.default_rng(7))

        self.assertEqual(int(sample["quality_pair_category_index"]), 0)
        self.assertEqual(int(sample["quality_sampling_attempts"]), 2)
        self.assertFalse(bool(sample["quality_forced_accept"]))

    def test_forced_fallback_keeps_training_finite(self) -> None:
        static = _uniform_pair()
        dataset = self._dataset([static])

        sample = dataset._build_quality_sample(np.random.default_rng(11))

        self.assertEqual(int(sample["quality_pair_category_index"]), 1)
        self.assertTrue(bool(sample["quality_forced_accept"]))
        self.assertEqual(int(sample["quality_sampling_attempts"]), 1)


class QualityCacheTests(unittest.TestCase):
    def test_segment_cache_round_trip(self) -> None:
        dataset = SimpleNamespace(
            records=[SimpleNamespace(run_id="run-0")],
            _valid_record_idx=[0],
            _record_choices=[[(1.0, 1, 1000.0)]],
            _choice_start_ranges=[[[(0, 0)]]],
            t_end_ns=(1.0,),
            anchor_mode="random",
            segment_policy="none",
            segment_time_range_ns=None,
            current_time_mode="t0",
            zeeman_precondition_enabled=False,
            zeeman_precondition_min_cycles=5.0,
            zeeman_precondition_gamma_hz_per_t=2.8e10,
            zeeman_precondition_spatial_field=True,
            zeeman_precondition_sign=1.0,
            compute_omega_target=True,
            seed=3,
        )
        dataset._build_sample = lambda *args, **kwargs: _uniform_pair()
        cfg = normalize_quality_sampling_config(
            {"enabled": True, "segment": {"probes_per_segment": 1}}
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "quality.json"
            built = build_segment_quality_cache(dataset, cfg, path)
            loaded = load_segment_quality_cache(dataset, cfg, path)

        self.assertTrue(built["completed"])
        self.assertEqual(loaded["segments"]["run-0"]["0"]["category"], "static")
        self.assertTrue(dataset.compute_omega_target)


if __name__ == "__main__":
    unittest.main()
