from __future__ import annotations

import unittest

import numpy as np
import torch

from distribution_score.distance import (
    FORMAL_BLOCKS,
    FORMAL_PATCH_EDGE_PX,
    FORMAL_SHIFT_RADIUS_PX,
    patch_grid,
    patch_shift_distance,
)
from distribution_score.version import (
    FORMAL_ALGORITHM_VERSION,
    FORMAL_DISTANCE_ID,
    PACKAGE_VERSION,
    algorithm_version,
    distance_identifier,
)


class PatchShiftDistanceTests(unittest.TestCase):
    def test_formal_defaults_are_4x4_with_quarter_patch_shift(self) -> None:
        self.assertEqual(PACKAGE_VERSION, "0.2.0")
        self.assertEqual(FORMAL_ALGORITHM_VERSION, "0.2.0")
        self.assertEqual(FORMAL_BLOCKS, 4)
        self.assertEqual(FORMAL_PATCH_EDGE_PX, 64)
        self.assertEqual(FORMAL_SHIFT_RADIUS_PX, 16)
        self.assertEqual(
            FORMAL_DISTANCE_ID,
            distance_identifier(FORMAL_BLOCKS, FORMAL_SHIFT_RADIUS_PX),
        )
        self.assertEqual(
            distance_identifier(8, 32),
            "full_vector_patch_shift_8x8_shift32_symmetric",
        )
        self.assertEqual(algorithm_version(4, 16), "0.2.0")
        self.assertEqual(algorithm_version(8, 32), "0.1.0")
        patches = patch_grid()
        self.assertEqual(len(patches), 16)
        self.assertEqual(patches[0], (0, 64, 0, 64))
        self.assertEqual(patches[-1], (192, 256, 192, 256))

    def test_distance_uses_formal_defaults(self) -> None:
        reference = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
        reference[:, 2] = 1.0
        mask = torch.ones((1, 1, 256, 256), dtype=torch.float32)
        distance, valid_patches = patch_shift_distance(
            reference,
            reference,
            mask,
            reference_chunk=1,
        )
        self.assertEqual(valid_patches, 16)
        np.testing.assert_allclose(distance, 0.0, atol=1.0e-6)

    def test_identical_and_opposite_uniform_fields(self) -> None:
        reference = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
        reference[:, 2] = 1.0
        identical = reference.clone()
        opposite = -reference
        query = torch.cat([identical, opposite], dim=0)
        mask = torch.ones((1, 1, 256, 256), dtype=torch.float32)
        distance, valid_patches = patch_shift_distance(
            reference,
            query,
            mask,
            blocks=8,
            shift_radius=0,
            reference_chunk=1,
        )
        self.assertEqual(valid_patches, 64)
        np.testing.assert_allclose(distance[:, 0], [0.0, 2.0], atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
