"""Performance monitoring: only realized accuracy can trigger degradation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from interest_rate_mlops.config import load_monitoring_config
from interest_rate_mlops.monitoring.drift import _classify
from interest_rate_mlops.monitoring.performance import evaluate_model_performance

FEATURES = ["lag_1", "lag_2"]


class ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.value)


class BrokenModel:
    def predict(self, frame: pd.DataFrame):
        raise RuntimeError("model backend exploded")


def _window(n: int = 30, target: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lag_1": np.full(n, target),
            "lag_2": np.full(n, target),
            "target": np.full(n, target),
        }
    )


CFG = load_monitoring_config()


def test_missing_model_returns_unknown_status():
    result = evaluate_model_performance(_window(), None, {}, CFG, feature_names=FEATURES)
    assert result.available is False
    assert result.status == "unknown"
    assert "could not be loaded" in result.reason


def test_too_few_realized_observations_skips_performance():
    frame = _window(5)
    result = evaluate_model_performance(
        frame, ConstantModel(1.0), {"test_rmse": 0.1}, CFG, feature_names=FEATURES
    )
    assert result.available is False
    assert "at least" in result.reason


def test_perfect_model_is_ok():
    result = evaluate_model_performance(
        _window(), ConstantModel(1.0), {"test_rmse": 0.1}, CFG, feature_names=FEATURES
    )
    assert result.available is True
    assert result.status == "ok"
    assert result.metrics["rmse"] == pytest.approx(0.0)
    assert result.rmse_degradation_pct == pytest.approx(-100.0)


def test_severe_degradation_is_critical():
    result = evaluate_model_performance(
        _window(), ConstantModel(3.0), {"test_rmse": 0.1}, CFG, feature_names=FEATURES
    )
    assert result.status == "critical"
    assert result.rmse_degradation_pct > CFG.performance.rmse_degradation_critical_pct


def test_moderate_degradation_is_warning():
    result = evaluate_model_performance(
        _window(), ConstantModel(1.15), {"test_rmse": 0.1}, CFG, feature_names=FEATURES
    )
    assert result.status == "warning"


def test_mape_is_ignored_when_near_zero_actuals_dominate():
    # Actuals near zero: MAPE is meaningless, so only RMSE drives the verdict.
    frame = _window(30, target=0.1)
    result = evaluate_model_performance(
        frame, ConstantModel(0.1), {"test_rmse": 0.1, "test_mape": 5.0}, CFG, feature_names=FEATURES
    )
    assert result.mape_degradation_pct is None
    assert "MAPE ignored" in result.reason
    assert result.status == "ok"


def test_prediction_failure_is_reported_not_raised():
    result = evaluate_model_performance(
        _window(), BrokenModel(), {"test_rmse": 0.1}, CFG, feature_names=FEATURES
    )
    assert result.available is False
    assert "prediction failed" in result.reason


def test_drift_status_classification_thresholds():
    assert _classify(0.0, CFG) == "ok"
    assert _classify(CFG.drift.warning_share, CFG) == "warning"
    assert _classify(CFG.drift.critical_share, CFG) == "critical"
