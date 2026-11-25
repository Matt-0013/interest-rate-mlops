"""Monitoring: performance gates, drift effect sizing and the reference window."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from interest_rate_mlops.config import Settings, load_monitoring_config, load_pipeline_config
from interest_rate_mlops.monitoring.drift import (
    _apply_effect_size_gate,
    _classify,
    _ks_statistic,
)
from interest_rate_mlops.monitoring.performance import evaluate_model_performance
from interest_rate_mlops.pipelines.monitoring_pipeline import resolve_reference_window
from interest_rate_mlops.training.splits import chronological_split

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


# ---------------------------------------------------------------------------
# Drift effect-size gate
# ---------------------------------------------------------------------------
def test_ks_statistic_is_bounded_and_unit_free():
    assert _ks_statistic(np.arange(10.0), np.arange(10.0)) == pytest.approx(0.0)
    assert _ks_statistic(np.zeros(10), np.ones(10) + 1.0) == pytest.approx(1.0)


def test_effect_size_gate_suppresses_a_barely_shifted_distribution():
    # Same distribution, different sample sizes: the test rejects the null on
    # volume alone, which is exactly what the magnitude floor must absorb.
    rng = np.random.default_rng(7)
    reference = pd.DataFrame({"lag_1": rng.normal(5.0, 1.0, 800)})
    current = pd.DataFrame({"lag_1": rng.normal(5.0, 1.0, 2000)})

    gated, suppressed = _apply_effect_size_gate(
        {"lag_1": {"drift_detected": True}}, reference, current, CFG
    )

    assert gated["lag_1"]["ks_statistic"] < CFG.drift.min_effect_size
    assert suppressed == 1
    # The raw verdict is preserved so the report never hides it.
    assert gated["lag_1"]["statistical_drift"] is True
    assert gated["lag_1"]["drift_detected"] is False


def test_effect_size_gate_keeps_a_material_shift():
    reference = pd.DataFrame({"lag_1": np.linspace(0.0, 10.0, 400)})
    current = pd.DataFrame({"lag_1": np.linspace(9.0, 10.0, 60)})

    gated, suppressed = _apply_effect_size_gate(
        {"lag_1": {"drift_detected": True}}, reference, current, CFG
    )

    assert suppressed == 0
    assert gated["lag_1"]["drift_detected"] is True


def test_effect_size_gate_counts_a_spread_collapse_without_a_mean_shift():
    # A variance-only change is still a distribution change: gating on the mean
    # alone would hide the zero-bound regime, so the KS statistic decides.
    rng = np.random.default_rng(3)
    reference = pd.DataFrame({"diff_1": rng.normal(0.0, 0.6, 500)})
    current = pd.DataFrame({"diff_1": rng.normal(0.0, 0.03, 60)})

    gated, suppressed = _apply_effect_size_gate(
        {"diff_1": {"drift_detected": True}}, reference, current, CFG
    )

    assert suppressed == 0
    assert gated["diff_1"]["mean_shift_sd"] < 0.2
    assert gated["diff_1"]["ks_statistic"] >= CFG.drift.min_effect_size
    assert gated["diff_1"]["drift_detected"] is True


# ---------------------------------------------------------------------------
# Reference window provenance
# ---------------------------------------------------------------------------
def _feature_matrix(n: int = 700, start: str = "1959-01-01") -> pd.DataFrame:
    index = pd.date_range(start=start, periods=n, freq="MS", name="origin_date")
    return pd.DataFrame({"lag_1": np.arange(n, dtype=float)}, index=index)


def _settings(tmp_path) -> Settings:
    return Settings(
        app_env="test",
        reports_dir=str(tmp_path),
        enable_db_persistence=False,
        log_format="console",
    )


def _write_split_record(tmp_path, *, train_end: str, n_train: int) -> None:
    directory = tmp_path / "evaluation" / "latest"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model_comparison.json").write_text(
        json.dumps({"split": {"train_end": train_end, "n_train": n_train}}), encoding="utf-8"
    )


def test_reference_window_uses_the_split_the_training_run_recorded(tmp_path):
    features = _feature_matrix()
    train_end = str(features.index[557].date())
    _write_split_record(tmp_path, train_end=train_end, n_train=558)

    window = resolve_reference_window(features, load_pipeline_config(), CFG, _settings(tmp_path))

    assert window.source == "recorded_split"
    assert window.train_end == "2005-06-01"
    assert len(window.frame) == 558
    # The reference must stop where training stopped, holdout never leaks in.
    assert str(window.frame.index[-1].date()) == window.train_end


def test_reference_window_falls_back_to_the_splitter_without_a_record(tmp_path):
    features = _feature_matrix()
    pipeline_cfg = load_pipeline_config()

    window = resolve_reference_window(features, pipeline_cfg, CFG, _settings(tmp_path))

    assert window.source == "chronological_split"
    assert len(window.frame) == chronological_split(features.index, pipeline_cfg.split).n_train
    assert len(window.frame) < len(features)


def test_reference_window_ignores_a_record_that_disagrees_with_the_matrix(tmp_path):
    features = _feature_matrix()
    _write_split_record(tmp_path, train_end=str(features.index[420].date()), n_train=999)

    window = resolve_reference_window(features, load_pipeline_config(), CFG, _settings(tmp_path))

    assert window.source == "chronological_split"
