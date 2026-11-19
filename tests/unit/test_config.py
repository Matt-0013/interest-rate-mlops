"""Configuration loading: typed, strict and explicit about failures."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from interest_rate_mlops.config import (
    FeatureConfig,
    Settings,
    load_data_config,
    load_monitoring_config,
    load_pipeline_config,
    load_yaml,
    resolve_path,
)
from interest_rate_mlops.exceptions import ConfigurationError


def test_data_config_matches_verified_dataset_schema():
    cfg = load_data_config()
    assert cfg.target.column == "Effective Federal Funds Rate"
    assert cfg.target.slug == "effective_federal_funds_rate"
    assert cfg.target.frequency == "monthly"
    assert cfg.raw_filename == "index.csv"
    assert cfg.date_columns == ["Year", "Month", "Day"]
    assert cfg.panel.start.isoformat() == "1958-01-01"
    assert cfg.validation.allowed_target_range == (0.0, 25.0)
    assert cfg.enabled_exogenous == []  # macro series carried but not modelled


def test_pipeline_config_is_typed_and_ordered():
    cfg = load_pipeline_config()
    # diff_12 needs both t and t-12, so the lookback is 13 observations.
    assert cfg.features.lookback == 13
    assert cfg.features.min_observations == 13
    assert cfg.features.names()[:3] == ["lag_1", "lag_2", "lag_3"]
    assert cfg.features.names()[-2:] == ["diff_1", "diff_12"]
    assert set(cfg.models) == {"linear_regression", "random_forest", "xgboost"}
    assert cfg.mlflow.registered_model_name == "interest_rate_forecaster"
    assert cfg.promotion.min_rmse_improvement_pct > 0


def test_monitoring_config_thresholds_are_ordered():
    cfg = load_monitoring_config()
    assert cfg.drift.warning_share < cfg.drift.critical_share
    assert cfg.current.window_size >= cfg.current.min_rows_for_performance
    assert cfg.paths.status_path.name == "drift_status.json"


def test_missing_file_raises_configuration_error(tmp_path: Path):
    with pytest.raises(ConfigurationError, match="not found"):
        load_yaml(tmp_path / "nope.yaml")
    with pytest.raises(ConfigurationError):
        load_data_config(tmp_path / "nope.yaml")


def test_invalid_yaml_shape_raises(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="mapping"):
        load_yaml(path)


def test_unknown_config_key_is_rejected(tmp_path: Path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        load_pipeline_config()
        .model_dump_json()
        .replace('"features"', '"unknown_key": 1, "features"'),
        encoding="utf-8",
    )
    from interest_rate_mlops.config import PipelineConfig

    with pytest.raises(ValidationError):
        PipelineConfig.model_validate_json(path.read_text(encoding="utf-8"))


def test_min_observations_must_cover_lookback():
    with pytest.raises(ValidationError, match="min_observations"):
        FeatureConfig(
            lags=[1, 12],
            rolling_windows=[3],
            rolling_stats=["mean"],
            diffs=[1],
            min_observations=5,
        )


def test_empty_lags_rejected():
    with pytest.raises(ValidationError):
        FeatureConfig(
            lags=[], rolling_windows=[3], rolling_stats=["mean"], diffs=[1], min_observations=3
        )


def test_resolve_path_is_root_relative():
    resolved = resolve_path("configs/data.yaml")
    assert resolved.is_absolute()
    assert resolved.name == "data.yaml"


def test_settings_defaults_are_local_only():
    settings = Settings()
    assert settings.database_url.startswith("postgresql+psycopg://")
    assert settings.resolved_tracking_uri.startswith("sqlite:///")
    assert settings.model_alias == "champion"
    assert settings.is_test is False
