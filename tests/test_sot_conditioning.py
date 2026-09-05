from __future__ import annotations

from skyrmion_cfm.data.conditions import (
    DEFAULT_EMBEDDED_SCALAR_KEYS,
    V4_CATEGORICAL_ORDERS,
    V4_EMBEDDED_SCALAR_KEYS,
)
from skyrmion_cfm.data.v4_metadata import (
    categorical_tensors,
    one_hot_from_metadata,
    pair_condition_row,
)


def test_categorical_encoding_normalizes_legacy_metadata_width() -> None:
    params = {
        "categorical_encodings": {
            # Older dataset shards persisted a trailing proxy slot, producing
            # a 14-wide vector even for the strict ``sot`` class.
            "drive_type": {"one_hot": [0.0] * 6 + [1.0] + [0.0] * 7},
        }
    }

    encoded = one_hot_from_metadata(
        params,
        "drive_type",
        "sot",
    )

    order = V4_CATEGORICAL_ORDERS["drive_type"]
    assert len(order) == 13
    assert "sot_like_slonczewski_proxy" not in order
    assert encoded.shape == (len(order),)
    assert int(encoded.argmax()) == order.index("sot")
    assert encoded.sum().item() == 1.0


def test_v4_sot_solver_parameters_reach_existing_model_conditioning() -> None:
    drive = {
        "active": True,
        "has_sot": True,
        "has_sot_like_slonczewski_proxy": False,
        "J_A_per_m2": 4.0e11,
        "J_vector_A_per_m2": [0.0, 0.0, 4.0e11],
        "charge_current_vector_A_per_m2": [4.0e11, 0.0, 0.0],
        "polarization": [0.0, 1.0, 0.0],
        "Pol": 0.2,
        "Lambda": 1.0,
        "EpsilonPrime": 0.03,
        "theta_DL_eff": 0.2,
        "r_FL_DL": 0.3,
        "active_kind": "sot",
        "rendered_torque_model": "effective_spin_hall_sot_via_slonczewski_lambda1",
        "sot_B_DL_T": 0.08,
        "sot_B_FL_T": 0.024,
        "sot_explicit_B_DL_T": 0.081,
        "sot_explicit_B_FL_sigma_cross_m_T": 0.016,
    }
    params = {
        "drive_type": "sot",
        "segments": [
            {
                "start_s": 0.0,
                "end_s": 1.0e-9,
                "T_K": 300.0,
                "drive": drive,
                "instantaneous_material": {
                    "Ms_T_A_per_m": 5.8e5,
                    "A_T_J_per_m": 1.0e-11,
                    "Ku_T_J_per_m3": 5.0e5,
                    "D_T_J_per_m2": 3.0e-3,
                    "alpha_T": 0.1,
                },
            }
        ],
        "temperature_schedule": {
            "T_schedule_mode": "isothermal",
            "T_schedule_parameters": {"T0_K": 300.0},
        },
        "grid": {"dx_m": 2.0e-9, "dy_m": 2.0e-9, "dz_m": 1.0e-9},
    }

    row = pair_condition_row(params, 0.0, 2.5e-10)

    assert row["current_a_m2"] == 4.0e11
    assert row["pol_eff"] == 0.2
    assert row["epsilon_prime"] == 0.03
    assert row["fixed_layer_y"] == 1.0
    assert row["theta_dl_eff"] == 0.2
    assert row["r_fl_dl"] == 0.3
    assert row["charge_current_x_a_per_m2"] == 4.0e11
    assert "pol_eff" in DEFAULT_EMBEDDED_SCALAR_KEYS
    assert "epsilon_prime" in DEFAULT_EMBEDDED_SCALAR_KEYS
    assert "has_sot" in V4_EMBEDDED_SCALAR_KEYS
    assert "j_vector_z_a_per_m2" in V4_EMBEDDED_SCALAR_KEYS

    encoded = categorical_tensors(params, params["segments"][0])["rendered_torque_model_onehot"]
    torque_order = V4_CATEGORICAL_ORDERS["rendered_torque_model"]
    strict_index = torque_order.index("effective_spin_hall_sot_via_slonczewski_lambda1")
    assert int(encoded.argmax()) == strict_index
    drive_encoded = categorical_tensors(params, params["segments"][0])["drive_type_onehot"]
    drive_order = V4_CATEGORICAL_ORDERS["drive_type"]
    assert len(drive_order) == 13
    assert int(drive_encoded.argmax()) == drive_order.index("sot")
    # Reusing the old proxy slot keeps categorical_linear checkpoint shapes
    # compatible while changing its semantics to the strict solver mapping.
    assert len(torque_order) == 7
