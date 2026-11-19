"""Shared pytest fixtures.

All fixtures are synthetic and small: tests must never depend on the historical
Kaggle snapshot so they stay fast and deterministic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from interest_rate_mlops.config import (
    DatasetConfig,
    FeatureConfig,
    Settings,
    get_settings,
    load_data_config,
    load_monitoring_config,
    load_pipeline_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CSV = PROJECT_ROOT / "tests" / "fixtures" / "sample_raw.csv"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    """Small synthetic raw frame with the real dataset's column names."""
    return pd.read_csv(FIXTURE_CSV, dtype=str)


@pytest.fixture()
def dataset_config(tmp_path: Path) -> DatasetConfig:
    """Dataset config pointed at the synthetic fixture and a temp data dir."""
    base = load_data_config()
    return base.model_copy(
        update={
            "raw_dir": str(FIXTURE_CSV.parent),
            "raw_filename": FIXTURE_CSV.name,
            "processed_dir": str(tmp_path / "processed"),
        }
    )


@pytest.fixture()
def feature_config() -> FeatureConfig:
    return FeatureConfig(
        lags=[1, 2, 3],
        rolling_windows=[3],
        rolling_stats=["mean", "std"],
        diffs=[1],
        min_observations=4,
    )


@pytest.fixture()
def pipeline_config():
    return load_pipeline_config()


@pytest.fixture()
def monitoring_config():
    return load_monitoring_config()


@pytest.fixture()
def settings() -> Settings:
    """Isolated settings: SQLite DB, no file-model fallback, test env."""
    return Settings(
        app_env="test",
        database_url="sqlite+pysqlite:///:memory:",
        mlflow_tracking_uri=None,
        enable_db_persistence=False,
        log_format="console",
    )


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Keep the cached settings singleton from leaking across tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def make_panel(
    n: int = 60,
    start: str = "2010-01-01",
    slope: float = 0.05,
    base: float = 2.0,
    extra_columns: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """Deterministic monthly panel with a linear trend and mild noise."""
    index = pd.date_range(start=start, periods=n, freq="MS", name="date")
    rng = np.random.default_rng(42)
    values = base + slope * np.arange(n) + rng.normal(0, 0.05, n)
    panel = pd.DataFrame({"target": values}, index=index)
    for name, series in (extra_columns or {}).items():
        panel[name] = series
    return panel
