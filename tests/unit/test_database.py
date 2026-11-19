"""Persistence layer behaviour (SQLite engine, same SQLAlchemy models as Postgres)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select

from interest_rate_mlops.config import Settings
from interest_rate_mlops.database import (
    Base,
    ModelMetadata,
    PredictionRecord,
    create_engine_from_url,
    create_session_factory,
)
from interest_rate_mlops.database.repositories import (
    count_predictions,
    get_active_model,
    latest_monitoring_run,
    latest_pipeline_executions,
    list_predictions,
    record_deployment,
    record_ingestion,
    record_monitoring_run,
    record_pipeline_execution,
    record_prediction,
    record_training_run,
    upsert_model_metadata,
)


@pytest.fixture()
def session():
    settings = Settings(app_env="test", database_url="sqlite+pysqlite:///:memory:")
    engine = create_engine_from_url(settings.database_url)
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    with factory() as db_session:
        yield db_session
    engine.dispose()


def test_record_training_run_is_idempotent(session):
    payload = {
        "run_id": "run-1",
        "model_name": "linear_regression",
        "model_type": "linear_regression",
        "experiment_name": "exp",
        "dataset_name": "fed",
        "dataset_version": "sha256:abc",
        "params": {"fit_intercept": True},
        "metrics": {"test_rmse": 0.1},
    }
    record_training_run(session, **payload)
    record_training_run(session, **{**payload, "metrics": {"test_rmse": 0.09}})

    rows = session.scalars(select(PredictionRecord)).all()
    assert rows == []
    from interest_rate_mlops.database.models import TrainingRun

    stored = session.scalars(select(TrainingRun)).all()
    assert len(stored) == 1
    assert stored[0].metrics["test_rmse"] == 0.09


def test_model_metadata_enforces_single_active_version(session):
    upsert_model_metadata(
        session,
        model_name="forecaster",
        version="1",
        alias="champion",
        stage="champion",
        metrics={"test_rmse": 0.2},
        is_active=True,
    )
    upsert_model_metadata(
        session,
        model_name="forecaster",
        version="2",
        alias="champion",
        stage="champion",
        metrics={"test_rmse": 0.15},
        is_active=True,
    )

    active = get_active_model(session, "forecaster")
    assert active is not None
    assert active.version == "2"

    versions = {row.version: row for row in session.scalars(select(ModelMetadata)).all()}
    assert versions["1"].is_active is False
    assert versions["1"].stage == "archived"
    assert versions["2"].is_active is True


def test_predictions_are_listed_newest_first_and_counted(session):
    for index in range(3):
        record_prediction(
            session,
            request_id=f"req-{index}",
            model_name="forecaster",
            model_version="2",
            prediction=1.0 + index,
            features={"lag_1": 1.0},
            latency_ms=1.5,
        )
    session.flush()

    records = list_predictions(session, limit=2)
    assert len(records) == 2
    assert records[0].prediction == pytest.approx(3.0)
    assert count_predictions(session) == 3
    assert count_predictions(session, model_name="other") == 0


def test_monitoring_run_is_retrievable(session):
    record_monitoring_run(
        session,
        model_name="forecaster",
        model_version="2",
        status="warning",
        data_drift=True,
        drift_share=0.4,
        performance_status="ok",
        retrain_recommended=False,
        n_reference_rows=100,
        n_current_rows=30,
        report_path="reports/monitoring/x.html",
        summary={"alerts": ["drift"]},
    )
    session.flush()
    latest = latest_monitoring_run(session)
    assert latest is not None
    assert latest.status == "warning"
    assert latest.summary["alerts"] == ["drift"]


def test_ingestion_deployment_and_pipeline_records(session):
    record_ingestion(
        session,
        dataset_name="fed",
        source="kaggle",
        raw_path="data/raw/index.csv",
        dataset_version="sha256:abc",
        rows_in=904,
        rows_out=710,
        validation_passed=True,
        report={"passed": True},
    )
    record_deployment(
        session,
        model_name="forecaster",
        to_version="2",
        from_version="1",
        action="promotion",
        reason="approved",
        decision={"approved": True},
    )
    record_pipeline_execution(
        session, pipeline="ingestion", status="succeeded", duration_seconds=1.2
    )
    record_pipeline_execution(
        session, pipeline="training", status="succeeded", duration_seconds=3.4
    )
    record_pipeline_execution(session, pipeline="training", status="failed", duration_seconds=0.1)
    session.flush()

    executions = latest_pipeline_executions(session)
    by_name = {execution.pipeline: execution for execution in executions}
    assert set(by_name) == {"ingestion", "training"}
    assert by_name["training"].status == "failed"
    assert by_name["training"].duration_seconds == pytest.approx(0.1)
    assert by_name["ingestion"].status == "succeeded"


def test_prediction_created_at_is_timezone_aware(session):
    record = record_prediction(
        session,
        request_id="req-tz",
        model_name="forecaster",
        model_version="2",
        prediction=1.0,
        features={},
    )
    session.flush()
    assert record.created_at.tzinfo is not None
    assert record.created_at <= datetime.now(UTC)


def test_table_set_excludes_unused_models(session):
    tables = set(Base.metadata.tables)
    assert tables == {
        "training_runs",
        "model_metadata",
        "model_deployment_history",
        "prediction_records",
        "data_ingestion_logs",
        "monitoring_runs",
        "pipeline_executions",
    }
    assert session.scalar(select(func.count()).select_from(ModelMetadata)) == 0
