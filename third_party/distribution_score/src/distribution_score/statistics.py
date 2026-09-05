"""Statistical definitions used by both distribution scores."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


DEFAULT_ALPHAS = np.arange(0.1, 1.0, 0.1, dtype=np.float64)


def nearest_neighbor_bandwidth(distance: np.ndarray, floor: float = 1.0e-6) -> float:
    """Median nearest-neighbor distance, excluding each sample itself."""

    matrix = np.asarray(distance, dtype=np.float64).copy()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError(f"expected a square self-distance matrix with at least 2 rows, got {matrix.shape}")
    np.fill_diagonal(matrix, np.inf)
    nearest = matrix.min(axis=1)
    finite = nearest[np.isfinite(nearest)]
    if finite.size == 0:
        raise ValueError("cannot estimate a bandwidth from the supplied distances")
    return max(float(np.median(finite)), float(floor))


def kernel_density(distance: np.ndarray, sigma: float) -> np.ndarray:
    """Average Gaussian-kernel similarity for each query row."""

    matrix = np.asarray(distance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError(f"expected a non-empty distance matrix, got {matrix.shape}")
    sigma = max(float(sigma), 1.0e-8)
    return np.exp(-np.square(matrix) / (2.0 * sigma * sigma)).mean(axis=1)


def leave_one_out_density(self_distance: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-kernel self density with the diagonal removed."""

    matrix = np.asarray(self_distance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError(f"expected a square self-distance matrix with at least 2 rows, got {matrix.shape}")
    sigma = max(float(sigma), 1.0e-8)
    kernel = np.exp(-np.square(matrix) / (2.0 * sigma * sigma))
    np.fill_diagonal(kernel, 0.0)
    return kernel.sum(axis=1) / (kernel.shape[1] - 1)


def density_ratio_summary(
    model_density: np.ndarray,
    truth_density: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
) -> dict[str, float | np.ndarray]:
    """Compare local model and truth densities symmetrically around ratio one."""

    model_density = np.asarray(model_density, dtype=np.float64)
    truth_density = np.asarray(truth_density, dtype=np.float64)
    if model_density.ndim != 1 or model_density.shape != truth_density.shape or model_density.size == 0:
        raise ValueError(
            f"expected equal non-empty one-dimensional densities, got "
            f"{model_density.shape} and {truth_density.shape}"
        )
    log_ratio = np.log(model_density + epsilon) - np.log(truth_density + epsilon)
    absolute_log_ratio = np.abs(log_ratio)
    point_score = np.exp(-absolute_log_ratio)
    mean_error = float(absolute_log_ratio.mean())
    model_nll = float((-np.log(model_density + epsilon)).mean())
    truth_nll = float((-np.log(truth_density + epsilon)).mean())
    delta_nll = model_nll - truth_nll
    return {
        "density_epsilon": float(epsilon),
        "symmetric_log_ratio_mae": mean_error,
        "symmetric_ratio_score": float(math.exp(-mean_error)),
        "model_nll": model_nll,
        "truth_leave_one_out_nll": truth_nll,
        "delta_nll": delta_nll,
        "forward_kl_style_score": float(
            math.exp(float(np.clip(-delta_nll, -50.0, 50.0)))
        ),
        "model_density": model_density,
        "truth_density": truth_density,
        "density_ratio": np.exp(log_ratio),
        "log_density_ratio": log_ratio,
        "abs_log_density_ratio": absolute_log_ratio,
        "per_truth_symmetric_score": point_score,
    }


def same_condition_score(
    truth_to_model_distance: np.ndarray,
    truth_self_distance: np.ndarray,
    *,
    sigma: float | None = None,
    epsilon: float = 1.0e-12,
) -> dict[str, float | np.ndarray]:
    """Compute the repeated-truth local-density-ratio score."""

    cross = np.asarray(truth_to_model_distance, dtype=np.float64)
    self_distance = np.asarray(truth_self_distance, dtype=np.float64)
    if cross.ndim != 2 or self_distance.shape != (cross.shape[0], cross.shape[0]):
        raise ValueError(f"incompatible cross/self distances: {cross.shape} and {self_distance.shape}")
    bandwidth = nearest_neighbor_bandwidth(self_distance) if sigma is None else float(sigma)
    summary = density_ratio_summary(
        kernel_density(cross, bandwidth),
        leave_one_out_density(self_distance, bandwidth),
        epsilon=epsilon,
    )
    return {"sigma": bandwidth, **summary}


def probability_rank_from_distances(
    reference_self_distance: np.ndarray,
    ranking_and_truth_to_reference_distance: np.ndarray,
    *,
    sigma: float | None = None,
) -> dict[str, float | np.ndarray]:
    """Return the density rank of the final query row, which is the truth."""

    queries = np.asarray(ranking_and_truth_to_reference_distance, dtype=np.float64)
    reference = np.asarray(reference_self_distance, dtype=np.float64)
    if queries.ndim != 2 or queries.shape[0] < 2 or queries.shape[1] != reference.shape[0]:
        raise ValueError(f"incompatible reference/query distances: {reference.shape}, {queries.shape}")
    bandwidth = nearest_neighbor_bandwidth(reference) if sigma is None else float(sigma)
    query_density = kernel_density(queries, bandwidth)
    ranking_density = query_density[:-1]
    truth_density = float(query_density[-1])
    rank = float(
        (1 + np.count_nonzero(ranking_density > truth_density))
        / (ranking_density.size + 1)
    )
    return {
        "sigma": bandwidth,
        "ranking_density": ranking_density,
        "truth_density": truth_density,
        "probability_rank_u": rank,
    }


def calibration_summary(
    values: np.ndarray,
    alphas: np.ndarray = DEFAULT_ALPHAS,
) -> dict[str, float | int | list[float]]:
    """Summarize how closely probability ranks follow Uniform(0, 1)."""

    values = np.asarray(values, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("probability ranks must be a non-empty vector")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probability ranks must lie in [0, 1]")
    coverage = np.asarray([(values <= alpha).mean() for alpha in alphas])
    signed_deviation = coverage - alphas
    error = float(np.abs(signed_deviation).mean())
    return {
        "n_conditions": int(values.size),
        "alphas": alphas.tolist(),
        "coverage": coverage.tolist(),
        "signed_deviation": signed_deviation.tolist(),
        "mean_absolute_calibration_error": error,
        "calibration_score": float(max(0.0, 1.0 - 2.0 * error)),
        "maximum_absolute_calibration_error": float(np.abs(signed_deviation).max()),
        "u_mean": float(values.mean()),
        "u_median": float(np.median(values)),
    }


def bootstrap_macro_score(
    groups: Mapping[str, np.ndarray],
    group_order: Sequence[str],
    *,
    iterations: int = 2000,
    seed: int = 0,
) -> list[float]:
    """Bootstrap the equal-group macro calibration score."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    rng = np.random.default_rng(seed)
    scores = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        group_scores = []
        for name in group_order:
            values = np.asarray(groups[name], dtype=np.float64)
            draw = values[rng.integers(0, len(values), size=len(values))]
            group_scores.append(float(calibration_summary(draw)["calibration_score"]))
        scores[iteration] = float(np.mean(group_scores))
    return [float(value) for value in np.quantile(scores, [0.025, 0.975])]


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    iterations: int = 2000,
    seed: int = 0,
) -> list[float]:
    """Percentile bootstrap interval for a scalar sample mean."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty vector")
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        means[iteration] = values[rng.integers(0, len(values), size=len(values))].mean()
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]
