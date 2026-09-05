import math
import unittest

from skyrmion_cfm.data.fixed_time import (
    _normalize_segment_drive_filter,
    _segment_j_max_abs_a_m2,
)


class SegmentDriveFilterTest(unittest.TestCase):
    def test_checks_global_charge_and_region_currents(self):
        self.assertEqual(
            _segment_j_max_abs_a_m2(
                {"drive": {"J_vector_A_per_m2": [0.0, 0.0, 0.0]}}
            ),
            0.0,
        )
        self.assertEqual(_segment_j_max_abs_a_m2({"drive": {"J_A_per_m2": -3.0}}), 3.0)
        self.assertEqual(
            _segment_j_max_abs_a_m2(
                {"drive": {"charge_current_vector_A_per_m2": [0.0, 4.0, 0.0]}}
            ),
            4.0,
        )
        self.assertEqual(
            _segment_j_max_abs_a_m2(
                {
                    "drive": {
                        "J_vector_A_per_m2": [0.0, 0.0, 0.0],
                        "region_values": [
                            {"region_id": 1, "J_vector_A_per_m2": [0.0, -5.0, 0.0]},
                        ],
                    }
                }
            ),
            5.0,
        )

    def test_ignores_inactive_torque_family_intent(self):
        segment = {
            "drive": {
                "active": False,
                "active_kind": "sot",
                "has_sot": False,
                "J_A_per_m2": 0.0,
                "J_vector_A_per_m2": [0.0, 0.0, 0.0],
            }
        }
        self.assertEqual(_segment_j_max_abs_a_m2(segment), 0.0)

    def test_normalization_and_validation(self):
        self.assertEqual(_normalize_segment_drive_filter(None), "all")
        self.assertEqual(_normalize_segment_drive_filter("zero_j"), "no_j")
        self.assertEqual(_normalize_segment_drive_filter("no_current"), "no_j")
        with self.assertRaisesRegex(ValueError, "all\\|no_j"):
            _normalize_segment_drive_filter("field_only")

    def test_non_finite_metadata_fails_closed(self):
        segment = {
            "drive": {
                "J_A_per_m2": math.nan,
                "J_vector_A_per_m2": [0.0, math.inf, 0.0],
            }
        }
        self.assertTrue(math.isinf(_segment_j_max_abs_a_m2(segment)))


if __name__ == "__main__":
    unittest.main()
