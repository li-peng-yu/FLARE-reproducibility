"""Distribution-aware scores for stochastic magnetic structures."""

from .statistics import (
    calibration_summary,
    density_ratio_summary,
    nearest_neighbor_bandwidth,
    probability_rank_from_distances,
    same_condition_score,
)
from .version import PACKAGE_VERSION

__all__ = [
    "calibration_summary",
    "density_ratio_summary",
    "nearest_neighbor_bandwidth",
    "probability_rank_from_distances",
    "same_condition_score",
]

__version__ = PACKAGE_VERSION
