"""Package and formal metric version identifiers."""

PACKAGE_VERSION = "0.2.0"
FORMAL_ALGORITHM_VERSION = "0.2.0"
FORMAL_DISTANCE_ID = "full_vector_patch_shift_4x4_shift16_symmetric"


def distance_identifier(blocks: int, shift_radius: int) -> str:
    """Return an identifier that records the distance actually computed."""

    return (
        f"full_vector_patch_shift_{blocks}x{blocks}_"
        f"shift{shift_radius}_symmetric"
    )


def algorithm_version(blocks: int, shift_radius: int) -> str:
    """Return the formal protocol version matching the requested distance."""

    if (blocks, shift_radius) == (4, 16):
        return FORMAL_ALGORITHM_VERSION
    if (blocks, shift_radius) == (8, 32):
        return "0.1.0"
    return "custom"
