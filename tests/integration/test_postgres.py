"""PostgreSQL integration.

Skipped unless a scratch database is provided via ``TEST_DATABASE_URL``. The
test creates the schema if missing and rolls its data back, so it can be run
against a throwaway database without leaving rows behind.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from interest_rate_mlops.database import Base, create_engine_from_url, create_session_factory
from interest_rate_mlops.database.repositories import (
    count_predictions,
    latest_monitoring_run,
    record_monitoring_run,
    record_prediction,
)


def _database_url() -> str:
    return os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL") or ""


@pytest.mark.integration
def test_postgres_roundtrip():
    url = _database_url()
    if not url.startswith("postgresql"):
        pytest.skip("set TEST_DATABASE_URL to a scratch PostgreSQL database to run this test")

    engine = create_engine_from_url(url)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"PostgreSQL not reachable: {exc}")

    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    with factory() as session:
        record_prediction(
            session,
            request_id="integration-request",
            model_name="forecaster",
            model_version="1",
            prediction=1.23,
            features={"lag_1": 1.2},
        )
        record_monitoring_run(
            session,
            model_name="forecaster",
            model_version="1",
            status="ok",
            data_drift=False,
            drift_share=0.0,
            performance_status="ok",
            retrain_recommended=False,
            n_reference_rows=10,
            n_current_rows=5,
            report_path=None,
            summary={"source": "integration-test"},
        )
        session.flush()

        assert count_predictions(session, model_name="forecaster") >= 1
        monitoring = latest_monitoring_run(session)
        assert monitoring is not None
        assert monitoring.status == "ok"
        session.rollback()

    engine.dispose()
