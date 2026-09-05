from __future__ import annotations

import math

from scripts.summarize_x5_temperature_ood import base_means, paired_bootstrap


def test_paired_bootstrap_error_degradation_direction() -> None:
    result = paired_bootstrap(
        {"base0001": 1.0, "base0002": 2.0, "base0003": 3.0},
        {"base0001": 2.0, "base0002": 3.0, "base0003": 4.0},
        iterations=500,
        seed=7,
    )

    assert result["id"]["mean"] == 2.0
    assert result["ood"]["mean"] == 3.0
    assert result["absolute_degradation"]["value"] == 1.0
    assert all(
        math.isclose(value, 1.0)
        for value in result["absolute_degradation"]["95ci"]
    )
    assert math.isclose(result["percent_degradation"]["value"], 50.0)


def test_paired_bootstrap_score_degradation_direction() -> None:
    result = paired_bootstrap(
        {"base0001": 0.8, "base0002": 0.6},
        {"base0001": 0.5, "base0002": 0.3},
        iterations=500,
        seed=11,
        lower_is_better=False,
    )

    assert math.isclose(result["id"]["mean"], 0.7)
    assert math.isclose(result["ood"]["mean"], 0.4)
    assert math.isclose(result["absolute_degradation"]["value"], 0.3)
    assert result["absolute_degradation"]["definition"] == "ID - OOD"


def test_base_means_clusters_draws_and_repeats_by_base() -> None:
    rows = [
        {"run_id": "r_00001_base0007_tr00_T300K", "ang": "10", "lambda": "1"},
        {"run_id": "r_00002_base0007_tr01_T300K", "ang": "14", "lambda": "1"},
        {"run_id": "r_00003_base0008_tr00_T300K", "ang": "20", "lambda": "1"},
        {"run_id": "r_00004_base0008_tr01_T300K", "ang": "99", "lambda": "0"},
    ]

    result = base_means(
        rows,
        "ang",
        predicate=lambda row: math.isclose(float(row["lambda"]), 1.0),
    )

    assert result == {"base0007": 12.0, "base0008": 20.0}
