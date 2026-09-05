from __future__ import annotations

import torch

from skyrmion_cfm.data.fixed_time import FixedTimePairDataset


class _Rng:
    def integers(self, low: int, high: int) -> int:
        return 1

    def random(self) -> float:
        return 0.0


def test_fixed_time_rot90_augment_rotates_b_t():
    ds = FixedTimePairDataset.__new__(FixedTimePairDataset)
    ds.augment = True
    ds.augment_rot90 = True
    ds.augment_spin_flip_prob = 0.0

    m = torch.zeros(3, 2, 2)
    defect = torch.zeros(1, 2, 2)
    j_field = torch.zeros(1, 2, 2)
    b_t = torch.tensor([2.0, 3.0, 5.0])

    *_, b_rot = ds._augment(m, m, defect, j_field, b_t, _Rng())

    assert torch.allclose(b_rot, torch.tensor([-3.0, 2.0, 5.0]))


def test_fixed_time_spin_flip_augment_flips_b_t():
    ds = FixedTimePairDataset.__new__(FixedTimePairDataset)
    ds.augment = True
    ds.augment_rot90 = False
    ds.augment_spin_flip_prob = 1.0

    m = torch.zeros(3, 2, 2)
    defect = torch.zeros(1, 2, 2)
    j_field = torch.zeros(1, 2, 2)
    b_t = torch.tensor([0.0, 0.0, 5.0])

    *_, b_flip = ds._augment(m, m, defect, j_field, b_t, _Rng())

    assert torch.allclose(b_flip, torch.tensor([0.0, 0.0, -5.0]))


def test_fixed_time_rot90_rotates_all_directional_condition_metadata():
    ds = FixedTimePairDataset.__new__(FixedTimePairDataset)
    sample = {
        "b_x_t": torch.tensor(2.0),
        "b_y_t": torch.tensor(3.0),
        "b_ext_x_t": torch.tensor(4.0),
        "b_ext_y_t": torch.tensor(5.0),
        "j_vector_x_a_per_m2": torch.tensor(6.0),
        "j_vector_y_a_per_m2": torch.tensor(7.0),
        "charge_current_x_a_per_m2": torch.tensor(12.0),
        "charge_current_y_a_per_m2": torch.tensor(13.0),
        "polarization_x": torch.tensor(8.0),
        "polarization_y": torch.tensor(9.0),
        "fixed_layer_x": torch.tensor(10.0),
        "fixed_layer_y": torch.tensor(11.0),
        "anis_u_x": torch.tensor(12.0),
        "anis_u_y": torch.tensor(13.0),
        "cubic_axis_1_x": torch.tensor(14.0),
        "cubic_axis_1_y": torch.tensor(15.0),
        "cubic_axis_2_x": torch.tensor(16.0),
        "cubic_axis_2_y": torch.tensor(17.0),
        "cubic_axis_3_x": torch.tensor(18.0),
        "cubic_axis_3_y": torch.tensor(19.0),
        "theta_T_x0_norm": torch.tensor(0.2),
        "theta_T_y0_norm": torch.tensor(0.7),
        "boundary_mode_onehot": torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]),
        "dx_m": torch.tensor(2.0e-9),
        "dy_m": torch.tensor(3.0e-9),
    }

    ds._rotate_condition_metadata_in_place(sample, rot_k=1)

    for x_key, y_key, old_x, old_y in (
        ("b_x_t", "b_y_t", 2.0, 3.0),
        ("b_ext_x_t", "b_ext_y_t", 4.0, 5.0),
        ("j_vector_x_a_per_m2", "j_vector_y_a_per_m2", 6.0, 7.0),
        ("charge_current_x_a_per_m2", "charge_current_y_a_per_m2", 12.0, 13.0),
        ("polarization_x", "polarization_y", 8.0, 9.0),
        ("fixed_layer_x", "fixed_layer_y", 10.0, 11.0),
        ("anis_u_x", "anis_u_y", 12.0, 13.0),
        ("cubic_axis_1_x", "cubic_axis_1_y", 14.0, 15.0),
        ("cubic_axis_2_x", "cubic_axis_2_y", 16.0, 17.0),
        ("cubic_axis_3_x", "cubic_axis_3_y", 18.0, 19.0),
    ):
        assert torch.allclose(sample[x_key], torch.tensor(-old_y))
        assert torch.allclose(sample[y_key], torch.tensor(old_x))
    assert torch.allclose(sample["theta_T_x0_norm"], torch.tensor(0.7))
    assert torch.allclose(sample["theta_T_y0_norm"], torch.tensor(0.8))
    assert torch.equal(
        sample["boundary_mode_onehot"],
        torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]),
    )
    assert torch.allclose(sample["dx_m"], torch.tensor(3.0e-9))
    assert torch.allclose(sample["dy_m"], torch.tensor(2.0e-9))


def test_fixed_time_rot90_metadata_covers_all_quarter_turns():
    ds = FixedTimePairDataset.__new__(FixedTimePairDataset)
    expected = {
        1: ((-3.0, 2.0), (0.7, 0.8), 2),
        2: ((-2.0, -3.0), (0.8, 0.3), 1),
        3: ((3.0, -2.0), (0.3, 0.2), 2),
    }
    for k, (vector, position, boundary_index) in expected.items():
        sample = {
            "anis_u_x": torch.tensor(2.0),
            "anis_u_y": torch.tensor(3.0),
            "theta_T_x0_norm": torch.tensor(0.2),
            "theta_T_y0_norm": torch.tensor(0.7),
            "boundary_mode_onehot": torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]),
        }
        ds._rotate_condition_metadata_in_place(sample, rot_k=k)
        assert torch.allclose(sample["anis_u_x"], torch.tensor(vector[0]))
        assert torch.allclose(sample["anis_u_y"], torch.tensor(vector[1]))
        assert torch.allclose(sample["theta_T_x0_norm"], torch.tensor(position[0]))
        assert torch.allclose(sample["theta_T_y0_norm"], torch.tensor(position[1]))
        assert int(sample["boundary_mode_onehot"].argmax()) == boundary_index


def test_fixed_time_rot90_segment_path_rotates_spatial_and_scalar_conditions():
    ds = FixedTimePairDataset.__new__(FixedTimePairDataset)
    scalar = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    jx = scalar.unsqueeze(0)
    jy = (scalar + 10.0).unsqueeze(0)
    bx = (scalar + 20.0).unsqueeze(0)
    by = (scalar + 30.0).unsqueeze(0)
    sample = {
        "m_init": torch.zeros(3, 2, 2),
        "m_t": torch.zeros(3, 2, 2),
        "defect_field": scalar.unsqueeze(0),
        "j_field": scalar.unsqueeze(0),
        "b_t": torch.tensor([2.0, 3.0, 5.0]),
        "omega_target": torch.zeros(3, 2, 2),
        "control_grid": scalar.clone(),
        "j_x_field": jx.clone(),
        "j_y_field": jy.clone(),
        "j_z_field": (scalar + 40.0).unsqueeze(0),
        "b_x_field": bx.clone(),
        "b_y_field": by.clone(),
        "b_z_field": (scalar + 50.0).unsqueeze(0),
        "temperature_field": (scalar + 60.0).unsqueeze(0),
        "anis_u_x": torch.tensor(6.0),
        "anis_u_y": torch.tensor(7.0),
        "boundary_mode_onehot": torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]),
    }

    ds._augment_sample_in_place(sample, rot_k=1, spin_flip=False)

    assert torch.allclose(sample["b_t"], torch.tensor([-3.0, 2.0, 5.0]))
    assert torch.equal(sample["control_grid"], torch.rot90(scalar, k=1, dims=(-2, -1)))
    assert torch.equal(sample["j_x_field"], -torch.rot90(jy, k=1, dims=(-2, -1)))
    assert torch.equal(sample["j_y_field"], torch.rot90(jx, k=1, dims=(-2, -1)))
    assert torch.equal(sample["b_x_field"], -torch.rot90(by, k=1, dims=(-2, -1)))
    assert torch.equal(sample["b_y_field"], torch.rot90(bx, k=1, dims=(-2, -1)))
    assert torch.equal(
        sample["temperature_field"],
        torch.rot90((scalar + 60.0).unsqueeze(0), k=1, dims=(-2, -1)),
    )
    assert torch.allclose(sample["anis_u_x"], torch.tensor(-7.0))
    assert torch.allclose(sample["anis_u_y"], torch.tensor(6.0))
    assert torch.equal(
        sample["boundary_mode_onehot"],
        torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]),
    )
    assert int(sample["augment_rot_k"]) == 1
    assert not bool(sample["augment_spin_flip"])
