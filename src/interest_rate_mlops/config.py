"""Typed configuration.

Three sources, each with one responsibility:

* ``configs/*.yaml`` - declarative pipeline configuration (committed, no secrets)
* environment variables - deployment-specific values and credentials
* CLI arguments - per-run overrides

YAML files are parsed into Pydantic models so a typo fails fast with a precise
message instead of surfacing later as a silent default.
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from interest_rate_mlops.exceptions import ConfigurationError

# Repository root. ``config.py`` lives at ``<root>/src/interest_rate_mlops``.
PROJECT_ROOT = Path(
    os.environ.get("MLOPS_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()


class StrictModel(BaseModel):
    """Pydantic base that rejects unknown keys, so config typos are loud."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# data.yaml
# ---------------------------------------------------------------------------
class TargetConfig(StrictModel):
    column: str
    slug: str
    unit: str
    frequency: Literal["monthly", "weekly", "daily"]
    horizon: int = Field(default=1, ge=1, le=24)


class ExogenousConfig(StrictModel):
    column: str
    slug: str
    frequency: Literal["monthly", "quarterly", "weekly", "daily"]
    asof_fill: Literal["none", "last_known"] = "last_known"
    enabled: bool = False


class PanelConfig(StrictModel):
    start: date | None = None
    end: date | None = None
    anchor: Literal["month_start"] = "month_start"
    drop_intra_period_rows: bool = True
    require_contiguous_target: bool = True


class ValidationConfig(StrictModel):
    required_columns: list[str]
    duplicate_key: list[str]
    allowed_target_range: tuple[float, float]
    max_missing_fraction: float = Field(default=0.95, ge=0.0, le=1.0)
    max_missing_fraction_target: float = Field(default=0.01, ge=0.0, le=1.0)
    outlier_iqr_factor: float = Field(default=3.0, gt=0.0)


class DatasetConfig(StrictModel):
    name: str
    source: Literal["kaggle", "local_file", "fred"]
    kaggle_dataset: str | None = None
    fred_series: list[str] = Field(default_factory=list)
    raw_dir: str
    raw_filename: str
    processed_dir: str
    date_columns: list[str]
    target: TargetConfig
    exogenous: list[ExogenousConfig] = Field(default_factory=list)
    panel: PanelConfig = Field(default_factory=PanelConfig)
    validation: ValidationConfig

    @model_validator(mode="after")
    def _check_source_fields(self) -> DatasetConfig:
        if self.source == "kaggle" and not self.kaggle_dataset:
            raise ValueError("dataset.kaggle_dataset is required when source is 'kaggle'")
        if self.source == "fred" and not self.fred_series:
            raise ValueError("dataset.fred_series is required when source is 'fred'")
        if len(self.date_columns) != 3:
            raise ValueError("dataset.date_columns must list exactly [year, month, day]")
        return self

    @property
    def raw_path(self) -> Path:
        return resolve_path(self.raw_dir) / self.raw_filename

    @property
    def raw_directory(self) -> Path:
        return resolve_path(self.raw_dir)

    @property
    def processed_directory(self) -> Path:
        return resolve_path(self.processed_dir)

    @property
    def panel_path(self) -> Path:
        return self.processed_directory / "panel.parquet"

    @property
    def features_path(self) -> Path:
        return self.processed_directory / "features.parquet"

    @property
    def enabled_exogenous(self) -> list[ExogenousConfig]:
        return [series for series in self.exogenous if series.enabled]

    def series_slugs(self) -> dict[str, str]:
        """Map raw column name -> canonical panel column name."""
        mapping = {self.target.column: self.target.slug}
        mapping.update({series.column: series.slug for series in self.exogenous})
        return mapping


# ---------------------------------------------------------------------------
# model.yaml
# ---------------------------------------------------------------------------
class FeatureConfig(StrictModel):
    lags: list[int]
    rolling_windows: list[int]
    rolling_stats: list[Literal["mean", "std"]]
    diffs: list[int]
    min_observations: int = Field(ge=2)

    @model_validator(mode="after")
    def _check_lookbacks(self) -> FeatureConfig:
        if not self.lags:
            raise ValueError("features.lags must not be empty")
        if any(lag < 1 for lag in self.lags):
            raise ValueError("features.lags must all be >= 1")
        if any(window < 2 for window in self.rolling_windows):
            raise ValueError("features.rolling_windows must all be >= 2")
        if any(diff < 1 for diff in self.diffs):
            raise ValueError("features.diffs must all be >= 1")
        required = max(max(self.lags), max(self.diffs) + 1, max(self.rolling_windows))
        if self.min_observations < required:
            raise ValueError(
                f"features.min_observations={self.min_observations} is too small; "
                f"features require at least {required} observations"
            )
        return self

    @property
    def lookback(self) -> int:
        """Number of observations needed before one feature row can be built."""
        return max(max(self.lags), max(self.diffs) + 1, max(self.rolling_windows))

    def names(self) -> list[str]:
        """Deterministic feature order used at training *and* serving time."""
        return (
            [f"lag_{lag}" for lag in self.lags]
            + [
                f"roll_{stat}_{window}"
                for window in self.rolling_windows
                for stat in self.rolling_stats
            ]
            + [f"diff_{diff}" for diff in self.diffs]
        )


class SplitConfig(StrictModel):
    strategy: Literal["chronological"] = "chronological"
    test_fraction: float = Field(gt=0.0, lt=1.0)
    min_test_observations: int = Field(default=24, ge=1)


class CVConfig(StrictModel):
    strategy: Literal["expanding_window"] = "expanding_window"
    n_splits: int = Field(default=5, ge=2, le=20)
    gap: int = Field(default=0, ge=0)


class ModelSpec(StrictModel):
    type: Literal["linear_regression", "random_forest", "xgboost"]
    params: dict[str, Any] = Field(default_factory=dict)


class SelectionConfig(StrictModel):
    primary_metric: str = "cv_rmse"
    secondary_metric: str = "cv_mae"
    minimize: bool = True


class MLflowConfig(StrictModel):
    experiment_name: str
    registered_model_name: str
    artifact_path: str = "model"
    champion_alias: str = "champion"
    challenger_alias: str = "challenger"
    register_best_only: bool = True
    tracking_uri: str | None = None


class PromotionConfig(StrictModel):
    max_test_rmse: float = Field(gt=0.0)
    max_mape: float = Field(gt=0.0)
    max_cv_rmse_std_ratio: float = Field(gt=0.0)
    min_rmse_improvement_pct: float = Field(ge=0.0)
    bootstrap_when_no_champion: bool = True


class PipelineConfig(StrictModel):
    features: FeatureConfig
    split: SplitConfig
    cv: CVConfig
    models: dict[str, ModelSpec]
    selection: SelectionConfig
    mlflow: MLflowConfig
    promotion: PromotionConfig

    @model_validator(mode="after")
    def _check_selection_metrics(self) -> PipelineConfig:
        known = {"cv_rmse", "cv_mae", "test_rmse", "test_mae", "cv_mape", "test_mape"}
        for metric in (self.selection.primary_metric, self.selection.secondary_metric):
            if metric not in known:
                raise ValueError(f"selection metric {metric!r} is not one of {sorted(known)}")
        if not self.models:
            raise ValueError("models must define at least one candidate")
        return self


# ---------------------------------------------------------------------------
# monitoring.yaml
# ---------------------------------------------------------------------------
class ReferenceConfig(StrictModel):
    strategy: Literal["training_split"] = "training_split"
    min_reference_rows: int = Field(default=60, ge=2)


class CurrentWindowConfig(StrictModel):
    window_size: int = Field(default=60, ge=2)
    min_rows_for_performance: int = Field(default=20, ge=2)


class DriftConfig(StrictModel):
    evidently_preset: str = "data_drift"
    warning_share: float = Field(default=0.30, ge=0.0, le=1.0)
    critical_share: float = Field(default=0.60, ge=0.0, le=1.0)
    min_current_rows: int = Field(default=20, ge=2)
    # A column is only counted as drifted when the statistical test rejects *and*
    # the shift is material: the two-sample Kolmogorov-Smirnov statistic D must
    # reach this value. Large windows make the p-value reject on differences of
    # no practical size, which is what this floor prevents. 0.0 disables it.
    min_effect_size: float = Field(default=0.10, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_ordering(self) -> DriftConfig:
        if self.warning_share > self.critical_share:
            raise ValueError("drift.warning_share must be <= drift.critical_share")
        return self


class PerformanceConfig(StrictModel):
    rmse_degradation_warning_pct: float = Field(default=20.0, gt=0.0)
    rmse_degradation_critical_pct: float = Field(default=50.0, gt=0.0)
    mape_degradation_warning_pct: float = Field(default=25.0, gt=0.0)
    mape_degradation_critical_pct: float = Field(default=60.0, gt=0.0)
    min_mape_coverage_pct: float = Field(default=50.0, ge=0.0, le=100.0)


class MonitoringPaths(StrictModel):
    reports_dir: str = "reports/monitoring"
    status_file: str = "reports/drift_status.json"
    latest_summary: str = "reports/monitoring/latest_summary.json"

    @property
    def reports_path(self) -> Path:
        return resolve_path(self.reports_dir)

    @property
    def status_path(self) -> Path:
        return resolve_path(self.status_file)

    @property
    def latest_summary_path(self) -> Path:
        return resolve_path(self.latest_summary)


class AlertConfig(StrictModel):
    retrain_when_drift_status: Literal["warning", "critical", "never"] = "critical"
    retrain_when_performance_status: Literal["warning", "critical", "never"] = "critical"


class MonitoringConfig(StrictModel):
    reference: ReferenceConfig
    current: CurrentWindowConfig
    drift: DriftConfig
    performance: PerformanceConfig
    paths: MonitoringPaths
    alerts: AlertConfig


# ---------------------------------------------------------------------------
# Environment-driven settings
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """Deployment-specific settings, loaded from environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        protected_namespaces=(),  # allow model_name / model_alias fields
    )

    app_env: Literal["development", "test", "staging", "production"] = "development"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    # Application database (kept separate from MLflow's tracking database).
    database_url: str = "postgresql+psycopg://mlops:mlops@localhost:5432/mlops"
    database_echo: bool = False
    database_pool_size: int = 5

    # MLflow tracking + registry.
    mlflow_tracking_uri: str | None = None
    mlflow_artifact_root: str | None = None
    model_name: str = "interest_rate_forecaster"
    model_alias: str = "champion"
    model_cache_ttl_seconds: int = Field(default=300, ge=0)

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "Interest Rate Forecasting API"
    api_version: str = "0.1.0"

    # Config file locations (relative to the project root unless absolute).
    data_config: str = "configs/data.yaml"
    pipeline_config: str = "configs/model.yaml"
    monitoring_config: str = "configs/monitoring.yaml"

    reports_dir: str = "reports"
    enable_db_persistence: bool = True
    # DVC is a repository-level tool; containers and tests can switch it off
    # without changing pipeline behaviour.
    enable_dvc: bool = True

    @property
    def resolved_tracking_uri(self) -> str:
        """Explicit env URI wins; otherwise a local SQLite tracking store."""
        if self.mlflow_tracking_uri:
            return self.mlflow_tracking_uri
        db_path = (PROJECT_ROOT / "mlruns" / "mlflow.db").as_posix()
        return f"sqlite:///{db_path}"

    @property
    def reports_path(self) -> Path:
        return resolve_path(self.reports_dir)

    @property
    def is_test(self) -> bool:
        return self.app_env == "test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def resolve_path(value: str | Path, root: Path | None = None) -> Path:
    """Resolve ``value`` against the project root unless it is absolute."""
    path = Path(value)
    return path if path.is_absolute() else (root or PROJECT_ROOT) / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping, raising :class:`ConfigurationError` on any problem."""
    resolved = resolve_path(path)
    if not resolved.exists():
        raise ConfigurationError(f"configuration file not found: {resolved}")
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ConfigurationError(f"invalid YAML in {resolved}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"{resolved} must contain a YAML mapping at the top level")
    return data


@lru_cache(maxsize=8)
def _data_config_cached(path: str) -> DatasetConfig:
    return DatasetConfig.model_validate(load_yaml(path)["dataset"])


@lru_cache(maxsize=8)
def _pipeline_config_cached(path: str) -> PipelineConfig:
    return PipelineConfig.model_validate(load_yaml(path))


@lru_cache(maxsize=8)
def _monitoring_config_cached(path: str) -> MonitoringConfig:
    return MonitoringConfig.model_validate(load_yaml(path))


def load_data_config(path: str | Path | None = None) -> DatasetConfig:
    """Load ``data.yaml`` (cached per path)."""
    resolved = resolve_path(path or os.environ.get("DATA_CONFIG", "configs/data.yaml"))
    if not resolved.exists():
        raise ConfigurationError(f"data config not found: {resolved}")
    return _data_config_cached(str(resolved))


def load_pipeline_config(path: str | Path | None = None) -> PipelineConfig:
    """Load ``model.yaml`` (cached per path)."""
    resolved = resolve_path(path or os.environ.get("PIPELINE_CONFIG", "configs/model.yaml"))
    if not resolved.exists():
        raise ConfigurationError(f"pipeline config not found: {resolved}")
    return _pipeline_config_cached(str(resolved))


def load_monitoring_config(path: str | Path | None = None) -> MonitoringConfig:
    """Load ``monitoring.yaml`` (cached per path)."""
    resolved = resolve_path(path or os.environ.get("MONITORING_CONFIG", "configs/monitoring.yaml"))
    if not resolved.exists():
        raise ConfigurationError(f"monitoring config not found: {resolved}")
    return _monitoring_config_cached(str(resolved))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
