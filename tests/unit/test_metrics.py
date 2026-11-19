"""Metric correctness, especially MAPE's suitability handling."""

from __future__ import annotations

import numpy as np
import pytest

from interest_rate_mlops.evaluation.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    regression_metrics,
    root_mean_squared_error,
)


def test_mae_known_value():
    assert mean_absolute_error([1.0, 2.0, 3.0], [1.5, 2.0, 2.0]) == pytest.approx(0.5)


def test_rmse_known_value():
    # errors: 1, 2 -> mean square 2.5 -> rmse sqrt(2.5)
    assert root_mean_squared_error([1.0, 2.0], [0.0, 4.0]) == pytest.approx(np.sqrt(2.5))


def test_rmse_penalises_large_errors_more_than_mae():
    actual = [0.0, 0.0]
    predicted = [0.0, 6.0]
    assert root_mean_squared_error(actual, predicted) > mean_absolute_error(actual, predicted)


def test_mape_ignores_zero_and_near_zero_actuals():
    result = mean_absolute_percentage_error([0.0, 0.1, 2.0, 10.0], [0.5, 0.5, 2.2, 9.0])
    # only 2.0 and 10.0 qualify (>= 0.5)
    assert result.n_evaluated == 2
    assert result.excluded_near_zero == 2
    assert result.coverage_pct == pytest.approx(50.0)
    assert result.value == pytest.approx(((0.2 / 2.0) + (1.0 / 10.0)) / 2 * 100)


def test_mape_returns_none_when_nothing_qualifies():
    result = mean_absolute_percentage_error([0.0, 0.0, 0.1], [0.1, 0.2, 0.2])
    assert result.value is None
    assert result.n_evaluated == 0
    assert result.coverage_pct == 0.0


def test_mape_threshold_is_configurable():
    result = mean_absolute_percentage_error([0.2], [0.22], min_abs_actual=0.0)
    assert result.value == pytest.approx(10.0)
    assert result.n_evaluated == 1


def test_regression_metrics_contains_primary_and_coverage():
    metrics = regression_metrics([1.0, 2.0, 0.0], [1.1, 2.1, 0.1])
    assert set(metrics) >= {
        "mae",
        "rmse",
        "mape",
        "mape_coverage_pct",
        "n_samples",
        "mean_actual",
    }
    assert metrics["n_samples"] == 3.0
    assert metrics["mape_coverage_pct"] == pytest.approx(66.67, abs=0.01)


def test_metric_input_validation():
    with pytest.raises(ValueError, match="same length"):
        mean_absolute_error([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="empty"):
        mean_absolute_error([], [])
    with pytest.raises(ValueError, match="NaN or infinite"):
        mean_absolute_error([1.0, np.nan], [1.0, 2.0])
