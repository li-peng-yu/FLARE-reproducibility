from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import skyrmion_cfm.train as train_module
from skyrmion_cfm.data.fixed_time import FixedTimePairDataset
from skyrmion_cfm.train import (
    EMA,
    _load_checkpoint_model_state,
    mask_batch,
    maybe_resume_training,
    prepare_segment_handoff_cache,
    segment_pushforward_loss,
)


class _Dataset:
    seed = 11
    _segment_pair_indices = [(0, 0), (0, 1)]
    records = [
        SimpleNamespace(
            run_id="run-0",
            path=Path("/dataset/run-0"),
            n_frames=3,
            params={"save_step_ps": 250.0},
        )
    ]

    def _build_segment_pair(self, rec_idx, segment_idx, rng, *, apply_augment):
        assert rec_idx == 0
        assert segment_idx == 1
        assert apply_augment is False
        state = torch.zeros(3, 2, 2)
        return {
            "run_id": "run-0",
            "frame_init": torch.tensor(1),
            "frame_target": torch.tensor(2),
            "m_init": state.clone(),
            "m_t": state.clone(),
            "prev_m_init": state.clone(),
        }


class _Sampler:
    ode_steps = 20

    def __init__(self) -> None:
        self.calls = 0

    def sample(self, model, m_init, cond):
        self.calls += 1
        return m_init + float(self.calls), None


class _PredictionRng:
    def integers(self, low: int, high: int) -> int:
        assert (low, high) == (0, 4)
        return 3

    def random(self) -> float:
        return 0.25


class _TrueInitRng:
    def integers(self, low: int, high: int) -> int:
        raise AssertionError("a true-init sample must not choose a predicted variant")

    def random(self) -> float:
        return 0.75


class _LookupCache:
    def __init__(self) -> None:
        self.calls = 0
        self.batch = None

    def lookup(self, batch, *, device, dtype):
        self.calls += 1
        self.batch = batch
        return torch.full_like(batch["m_init"], 9.0, device=device, dtype=dtype)


def _cfg(cache_path: Path) -> dict:
    return {
        "seed": 78,
        "data": {},
        "model": {},
        "bridge": {},
        "prior": {},
        "sampler": {"ode_steps": 20, "method": "heun"},
        "train": {
            "batch_size": 2,
            "amp_dtype": "bf16",
            "resume_reset_step": True,
            "segment_pushforward": {
                "enabled": True,
                "cache_mode": "precompute",
                "cache_path": str(cache_path),
                "pred_weight": 1.0,
                "predicted_inits_per_segment": 3,
                "roll_ode_steps": 20,
            },
        },
    }


class SegmentPushforwardTests(unittest.TestCase):
    def test_distribution_score_selection_keeps_nearest_candidate(self):
        truth = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
        truth[:, 2] = 1.0
        candidates = torch.stack([-truth[0], truth[0]], dim=0).unsqueeze(0).half()

        selected, distances, indices = train_module._select_segment_handoff_candidates(
            candidates,
            truth,
            keep=1,
            device=torch.device("cpu"),
            selection_cfg={
                "method": "distribution_score",
                "blocks": 1,
                "shift_radius": 0,
                "score_candidate_batch_size": 2,
                "reference_chunk": 1,
            },
        )

        np.testing.assert_array_equal(indices, [[1]])
        np.testing.assert_allclose(distances, [[0.0]], atol=1.0e-6)
        self.assertTrue(torch.equal(selected, candidates[:, 1:2]))

    def test_segment_pair_selects_one_precomputed_variant(self):
        dataset = FixedTimePairDataset.__new__(FixedTimePairDataset)
        dataset.segment_pair_predicted_inits = 4
        dataset.segment_pair_predicted_probability = 0.5
        dataset.augment = False
        dataset._segment_boundary_specs = lambda rec_idx: [(0, 10), (1, 20)]

        def build_sample(rec_idx, choice_idx, rng, *, frame_init, apply_augment):
            state = torch.full((3, 2, 2), float(frame_init))
            return {
                "run_id": "run-0",
                "frame_init": torch.tensor(frame_init),
                "frame_target": torch.tensor(frame_init + 1),
                "m_init": state,
                "m_t": state + 1.0,
            }

        dataset._build_sample = build_sample
        sample = dataset._build_segment_pair(0, 1, _PredictionRng(), apply_augment=False)

        self.assertEqual(int(sample["predicted_init_index"]), 3)
        self.assertTrue(bool(sample["use_predicted_init"]))
        self.assertTrue(bool(sample["has_prev_segment"]))
        self.assertTrue(torch.all(sample["prev_m_init"] == 10.0))

        true_sample = dataset._build_segment_pair(
            0,
            1,
            _TrueInitRng(),
            apply_augment=False,
        )
        self.assertEqual(int(true_sample["predicted_init_index"]), 0)
        self.assertFalse(bool(true_sample["use_predicted_init"]))

        first = dataset._build_segment_pair(0, 0, _PredictionRng(), apply_augment=False)
        self.assertEqual(int(first["predicted_init_index"]), 0)
        self.assertFalse(bool(first["use_predicted_init"]))
        self.assertFalse(bool(first["has_prev_segment"]))

    def test_segment_loss_mixes_selected_inits_in_one_forward(self):
        m_init = (
            torch.arange(4, dtype=torch.float32)
            .view(4, 1, 1, 1)
            .expand(4, 3, 2, 2)
            .clone()
        )
        batch = {
            "run_id": ["first", "pred-a", "true", "pred-b"],
            "frame_init": torch.tensor([0, 1, 2, 3]),
            "frame_target": torch.tensor([1, 2, 3, 4]),
            "predicted_init_index": torch.tensor([0, 1, 0, 3]),
            "m_init": m_init,
            "m0": m_init,
            "m_t": torch.zeros_like(m_init),
            "omega_target": torch.full_like(m_init, -5.0),
            "prev_m_init": torch.zeros_like(m_init),
            "has_prev_segment": torch.tensor([False, True, True, True]),
            "use_predicted_init": torch.tensor([True, True, False, True]),
        }
        cache = _LookupCache()
        forwarded_batches = []
        log_map_inputs = []

        def loss_fn(_model, train_batch, _cond):
            forwarded_batches.append(train_batch)
            total = train_batch["m_init"].mean()
            zero = total * 0.0
            return train_module.LossOutput(
                total=total,
                cfm=total,
                unit=zero,
                llg=zero,
                topo=zero,
                endpoint=zero,
            )

        def fake_log_map(m0, m1):
            log_map_inputs.append((m0.clone(), m1.clone()))
            return m0 + 100.0

        with (
            mock.patch.object(
                train_module,
                "collate_fixed_time_conditions",
                lambda _: {},
            ),
            mock.patch.object(
                train_module,
                "log_map_chw",
                side_effect=fake_log_map,
            ),
        ):
            segment_pushforward_loss(
                None,
                loss_fn,
                batch,
                needs_omega=True,
                cfg={"true_weight": 1.0, "pred_weight": 1.0},
                handoff_cache=cache,
            )

        self.assertEqual(len(forwarded_batches), 1)
        self.assertEqual(cache.calls, 1)
        self.assertEqual(cache.batch["run_id"], ["pred-a", "pred-b"])
        mixed = forwarded_batches[0]
        self.assertTrue(
            torch.equal(
                mixed["m_init"][:, 0, 0, 0],
                torch.tensor([0.0, 9.0, 2.0, 9.0]),
            )
        )
        self.assertTrue(torch.equal(mixed["m0"], mixed["m_init"]))
        self.assertEqual(len(log_map_inputs), 1)
        self.assertEqual(log_map_inputs[0][0].shape, m_init.shape)
        self.assertTrue(torch.equal(log_map_inputs[0][0], mixed["m_init"]))
        self.assertTrue(
            torch.equal(
                mixed["omega_target"][:, 0, 0, 0],
                torch.tensor([-5.0, 109.0, -5.0, 109.0]),
            )
        )

    def test_precompute_generates_all_variants_once_and_reuses_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = _cfg(tmp_path / "handoffs")
            sampler = _Sampler()
            teacher = {
                "checkpoint": "/checkpoints/stage1.pt",
                "sha256": "teacher-sha",
                "state": "ema",
                "step": 100,
            }

            with mock.patch.object(
                train_module,
                "collate_fixed_time_conditions",
                lambda batch: {},
            ):
                cache = prepare_segment_handoff_cache(
                    cfg,
                    _Dataset(),
                    torch.nn.Identity(),
                    sampler,
                    torch.device("cpu"),
                    tmp_path,
                    distributed=False,
                    teacher_identity=teacher,
                )

                self.assertEqual(sampler.calls, 3)
                self.assertIsNotNone(cache)
                self.assertEqual(cache.predictions_per_segment, 3)
                selected = cache.lookup(
                    {
                        "run_id": ["run-0"],
                        "frame_init": torch.tensor([1]),
                        "frame_target": torch.tensor([2]),
                        "predicted_init_index": torch.tensor([2]),
                    },
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                self.assertTrue(torch.all(selected == 3.0))
                self.assertEqual(sampler.calls, 3)

                # Stage 2 masks first segments before cache lookup. Keep
                # non-tensor sample identities aligned with the masked frame
                # tensors so lookup cannot combine one run with another run's
                # frame range.
                mixed = mask_batch(
                    {
                        "run_id": ["first-segment", "run-0"],
                        "frame_init": torch.tensor([0, 1]),
                        "frame_target": torch.tensor([1, 2]),
                        "predicted_init_index": torch.tensor([0, 2]),
                    },
                    torch.tensor([False, True]),
                )
                self.assertEqual(mixed["run_id"], ["run-0"])
                selected = cache.lookup(
                    mixed,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                self.assertTrue(torch.all(selected == 3.0))

                resume_cfg = deepcopy(cfg)
                resume_cfg["train"]["resume_reset_step"] = False
                reused = prepare_segment_handoff_cache(
                    resume_cfg,
                    _Dataset(),
                    torch.nn.Identity(),
                    sampler,
                    torch.device("cpu"),
                    tmp_path,
                    distributed=False,
                    teacher_identity=teacher,
                )
                self.assertIsNotNone(reused)
                self.assertEqual(sampler.calls, 3)

    def test_precompute_overgenerates_and_keeps_only_selected_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = _cfg(tmp_path / "handoffs")
            segment_cfg = cfg["train"]["segment_pushforward"]
            segment_cfg["predicted_inits_per_segment"] = 2
            segment_cfg["candidate_inits_per_segment"] = 5
            segment_cfg["candidate_selection"] = {"method": "distribution_score"}
            sampler = _Sampler()
            teacher = {
                "checkpoint": "/checkpoints/stage1.pt",
                "sha256": "teacher-sha",
                "state": "ema",
                "step": 100,
            }

            def select(candidates, truth, *, keep, device, selection_cfg):
                self.assertEqual(tuple(candidates.shape), (1, 5, 3, 2, 2))
                self.assertEqual(keep, 2)
                self.assertTrue(torch.all(truth == 0.0))
                indices = np.array([[3, 1]], dtype=np.int64)
                distances = np.array([[0.1, 0.2]], dtype=np.float32)
                return candidates[:, [3, 1]], distances, indices

            with (
                mock.patch.object(
                    train_module,
                    "collate_fixed_time_conditions",
                    lambda batch: {},
                ),
                mock.patch.object(
                    train_module,
                    "_select_segment_handoff_candidates",
                    side_effect=select,
                ) as selector,
            ):
                cache = prepare_segment_handoff_cache(
                    cfg,
                    _Dataset(),
                    torch.nn.Identity(),
                    sampler,
                    torch.device("cpu"),
                    tmp_path,
                    distributed=False,
                    teacher_identity=teacher,
                )

            self.assertEqual(sampler.calls, 5)
            self.assertEqual(selector.call_count, 1)
            self.assertEqual(cache.predictions_per_segment, 2)
            self.assertEqual(cache.candidates_per_segment, 5)
            self.assertTrue(cache.candidate_selection["enabled"])
            first = cache.lookup(
                {
                    "run_id": ["run-0"],
                    "frame_init": torch.tensor([1]),
                    "frame_target": torch.tensor([2]),
                    "predicted_init_index": torch.tensor([0]),
                },
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            second = cache.lookup(
                {
                    "run_id": ["run-0"],
                    "frame_init": torch.tensor([1]),
                    "frame_target": torch.tensor([2]),
                    "predicted_init_index": torch.tensor([1]),
                },
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            self.assertTrue(torch.all(first == 4.0))
            self.assertTrue(torch.all(second == 2.0))

    def test_candidate_count_cannot_be_smaller_than_kept_count(self):
        cfg = _cfg(Path("handoffs"))
        segment_cfg = cfg["train"]["segment_pushforward"]
        segment_cfg["predicted_inits_per_segment"] = 3
        segment_cfg["candidate_inits_per_segment"] = 2
        with self.assertRaisesRegex(ValueError, "candidate_inits_per_segment"):
            prepare_segment_handoff_cache(
                cfg,
                _Dataset(),
                torch.nn.Identity(),
                _Sampler(),
                torch.device("cpu"),
                Path("."),
                distributed=False,
            )

    def test_segment_loss_refuses_online_inference(self):
        batch = {
            "m_init": torch.zeros(1, 3, 2, 2),
            "m_t": torch.zeros(1, 3, 2, 2),
            "prev_m_init": torch.zeros(1, 3, 2, 2),
            "has_prev_segment": torch.tensor([True]),
            "use_predicted_init": torch.tensor([True]),
        }

        with self.assertRaisesRegex(RuntimeError, "must be precomputed"):
            segment_pushforward_loss(
                None,
                None,
                batch,
                needs_omega=False,
                cfg={"true_weight": 0.0, "pred_weight": 1.0},
                handoff_cache=None,
            )

    def test_online_cache_mode_is_rejected(self):
        cfg = {
            "train": {
                "segment_pushforward": {
                    "enabled": True,
                    "cache_mode": "online",
                    "pred_weight": 1.0,
                }
            }
        }

        with self.assertRaisesRegex(ValueError, "online inference"):
            prepare_segment_handoff_cache(
                cfg,
                None,
                torch.nn.Identity(),
                _Sampler(),
                torch.device("cpu"),
                Path("."),
                distributed=False,
            )

    def test_ema_state_can_initialize_the_trainable_model(self):
        model = torch.nn.Linear(2, 1)
        payload = {
            "model": {
                "weight": torch.full_like(model.weight, 1.0),
                "bias": torch.full_like(model.bias, 1.0),
            },
            "ema": {
                "weight": torch.full_like(model.weight, 2.0),
                "bias": torch.full_like(model.bias, 3.0),
            },
        }

        _load_checkpoint_model_state(model, payload, state_name="ema", strict=True)

        self.assertTrue(torch.all(model.weight == 2.0))
        self.assertTrue(torch.all(model.bias == 3.0))

    def test_stage2_resume_initializes_model_and_new_ema_from_stage1_ema(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "stage1.pt"
            source = torch.nn.Linear(2, 1)
            torch.save(
                {
                    "step": 100,
                    "model": {
                        "weight": torch.full_like(source.weight, 1.0),
                        "bias": torch.full_like(source.bias, 1.0),
                    },
                    "ema": {
                        "weight": torch.full_like(source.weight, 2.0),
                        "bias": torch.full_like(source.bias, 3.0),
                    },
                },
                checkpoint,
            )
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters())
            ema = EMA(model, 0.9999)
            cfg = {
                "train": {
                    "resume_from": str(checkpoint),
                    "resume_model_state": "ema",
                    "resume_reset_step": True,
                    "resume_load_optimizer": False,
                    "resume_load_ema": False,
                }
            }

            step = maybe_resume_training(
                cfg,
                model,
                optimizer,
                ema,
                torch.device("cpu"),
            )

            self.assertEqual(step, 0)
            self.assertTrue(torch.all(model.weight == 2.0))
            self.assertTrue(torch.all(model.bias == 3.0))
            self.assertTrue(torch.all(ema.shadow["weight"] == 2.0))
            self.assertTrue(torch.all(ema.shadow["bias"] == 3.0))


if __name__ == "__main__":
    unittest.main()
