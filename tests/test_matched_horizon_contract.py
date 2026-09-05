"""Regression guards for the revised paper's physical execution protocols."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math
import pytest
import torch

from skyrmion_cfm.config import release_checkpoint_config


def test_checkpoint_relocation_preserves_model_and_training_and_input_precision(monkeypatch, tmp_path):
    config = {'data': {'dataset_root': ['/old/skx_bt_1000base_x5_20260803'],
                       'memmap': {'enabled': False, 'dtype': 'float16'}},
              'model': {'arch': 'unet', 'width': 128},
              'prior': {'type': 'standard_gaussian'},
              'train': {'steps': 50000}, 'sampler': {'ode_steps': 10}}
    original = deepcopy(config)
    monkeypatch.setenv('FLARE_DATASET_ROOT', str(tmp_path))
    relocated = release_checkpoint_config(config)
    assert config == original
    for field in ('model', 'prior', 'train', 'sampler'):
        assert relocated[field] == config[field]
    assert relocated['data']['dataset_root'] == [str(tmp_path / 'primary_x5/skx_bt_1000base_x5_20260803')]
    assert relocated['data']['memmap']['enabled'] is True
    assert relocated['data']['memmap']['dtype'] == 'float16'


@pytest.mark.parametrize('duration', [0.2509511836, 0.79319019858, 3.5])
def test_native_fractional_tail_closes_control_switch(monkeypatch, duration):
    from external_baselines import x5_author_native as native
    monkeypatch.setattr(native, 'attach_native_schedule_fields', lambda dataset, record, sample: sample)
    dataset = SimpleNamespace(_t_end_bucket_index=lambda value: 0)
    sample = {'t_end_ns': duration, 'frame_init_time_ns': 1.0,
              'frame_target_time_ns': 1.0 + duration,
              'run_id': 'synthetic_control', 'drive_fraction': 1.0}
    steps = native._exact_control_substeps(dataset, 0, sample)
    assert math.isclose(sum(float(s['t_end_ns']) for s in steps), duration, abs_tol=1e-7)
    assert all(0 < float(s['t_end_ns']) <= 0.25 + 1e-7 for s in steps)
    for left, right in zip(steps, steps[1:]):
        assert torch.equal(left['frame_target_time_ns'], right['frame_init_time_ns'])
    assert math.isclose(float(steps[-1]['frame_target_time_ns']), 1.0 + duration, abs_tol=1e-6)


def test_native_two_segment_timing_uses_predicted_handoff():
    from external_baselines.x5_author_native import _timed_multisegment_rollout
    class ToyModel:
        method = 'poseidon_t'
        def __init__(self):
            self.seen = []
        def predict(self, condition):
            self.seen.append(condition.clone())
            return condition[:, :3] + 1
    first = torch.zeros(1, 5, 2, 2)
    second = torch.full_like(first, 99)
    model = ToyModel()
    result = _timed_multisegment_rollout(model, [first, second], precision='fp32', segment_step_counts=[1,1])
    torch.testing.assert_close(result, torch.full((1,3,2,2), 2.0))
    torch.testing.assert_close(model.seen[1][:,:3], torch.ones(1,3,2,2))
    torch.testing.assert_close(model.seen[1][:,3:], second[:,3:])
