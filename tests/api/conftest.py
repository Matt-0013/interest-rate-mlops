"""API test fixtures.

A stub champion keeps the tests independent of MLflow, while the database is a
real SQLite schema so persistence is genuinely exercised.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.dependencies.database import Database
from api.dependencies.model import ModelHandle
from api.main import create_app
from interest_rate_mlops.config import Settings, load_pipeline_config
from interest_rate_mlops.database.base import Base
from interest_rate_mlops.exceptions import ModelNotAvailableError
from interest_rate_mlops.features import features_from_history

FEATURES = load_pipeline_config().features


class StubModel:
    """Deterministic regressor stand-in."""

    def __init__(self, value: float = 2.5) -> None:
        self.value = value
        self.calls = 0

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self.calls += 1
        return np.full(len(frame), self.value)


class StubModelService:
    """Minimal ``ModelService`` substitute with a working feature contract."""

    def __init__(self, settings: Settings, value: float = 2.5, available: bool = True) -> None:
        self.settings = settings
        self.model = StubModel(value)
        self.available = available
        self.handle = ModelHandle(
            model=self.model,
            name=settings.model_name,
            version="7",
            alias="champion",
            run_id="run-7",
            dataset_version="sha256:test",
            feature_names=list(FEATURES.names()),
            target="effective_federal_funds_rate",
            unit="percent per annum",
            horizon=1,
            lookback_observations=FEATURES.lookback,
            metrics={"test_rmse": 0.1585, "test_mae": 0.1125},
            loaded_at=datetime.now(UTC),
            tracking_uri=settings.resolved_tracking_uri,
        )

    def get(self, refresh: bool = False) -> ModelHandle:
        if not self.available:
            raise ModelNotAvailableError("no champion model is promoted")
        return self.handle


@pytest.fixture()
def api_settings(tmp_path) -> Settings:
    return Settings(
        app_env="test",
        database_url="sqlite+pysqlite:///:memory:",
        enable_db_persistence=True,
        reports_dir=str(tmp_path / "reports"),
        log_format="console",
        model_cache_ttl_seconds=0,
    )


@pytest.fixture()
def api_db(api_settings: Settings) -> Database:
    database = Database(api_settings)
    assert database.connect(), "in-memory sqlite should connect"
    Base.metadata.create_all(database.engine)
    yield database
    database.dispose()


@pytest.fixture()
def model_service(api_settings: Settings) -> StubModelService:
    return StubModelService(api_settings)


@pytest.fixture()
def client(api_settings: Settings, api_db: Database, model_service: StubModelService):
    app = create_app(settings=api_settings, database=api_db, model_service=model_service)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def history_payload() -> list[dict[str, float | str]]:
    index = pd.date_range("2016-01-01", periods=FEATURES.lookback + 1, freq="MS")
    return [
        {"date": stamp.strftime("%Y-%m-%d"), "value": round(0.3 + 0.01 * position, 3)}
        for position, stamp in enumerate(index)
    ]


@pytest.fixture()
def features_payload(history_payload) -> dict[str, float]:
    result = features_from_history(history_payload, FEATURES, target_name="target")
    return result["features"]
