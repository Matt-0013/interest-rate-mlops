"""Repository functions.

Keeping persistence in plain functions (rather than hidden in the pipelines)
makes the write side of the system easy to test with an in-memory SQLite session
and keeps transaction boundaries visible at the call site.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Select, desc, select
from sqlalchemy.orm import Session

from interest_rate_mlops.database.models import (
    DataIngestionLog,
    ModelDeploymentHistory,
    ModelMetadata,
    MonitoringRun,
    PipelineExecution,
    PredictionRecord,
    TrainingRun,
    utcnow,
)


def record_training_run(
    session: Session,
    *,
    run_id: str,
    model_name: str,
    model_type: str,
    experiment_name: str,
    dataset_name: str,
    dataset_version: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    cv_metrics: dict[str, Any] | None = None,
    model_version: str | None = None,
    git_commit: str | None = None,
    duration_seconds: float | None = None,
    status: str = "completed",
) -> TrainingRun:
    """Insert (or update) the training-run record for one candidate."""
    existing = session.scalars(
        select(TrainingRun).where(
            TrainingRun.run_id == run_id, TrainingRun.model_name == model_name
        )
    ).first()
    record = existing or TrainingRun(run_id=run_id, model_name=model_name)
    record.model_type = model_type
    record.experiment_name = experiment_name
    record.dataset_name = dataset_name
    record.dataset_version = dataset_version
    record.params = params
    record.metrics = metrics
    record.cv_metrics = cv_metrics or {}
    record.model_version = model_version
    record.git_commit = git_commit
    record.duration_seconds = duration_seconds
    record.status = status
    if existing is None:
        session.add(record)
    session.flush()
    return record


def upsert_model_metadata(
    session: Session,
    *,
    model_name: str,
    version: str,
    alias: str,
    stage: str,
    metrics: dict[str, Any],
    run_id: str | None = None,
    dataset_version: str | None = None,
    is_active: bool = False,
) -> ModelMetadata:
    """Insert or update a model version's promotion state."""
    record = session.scalars(
        select(ModelMetadata).where(
            ModelMetadata.model_name == model_name, ModelMetadata.version == version
        )
    ).first()
    if record is None:
        record = ModelMetadata(model_name=model_name, version=version)
        session.add(record)
    record.alias = alias
    record.stage = stage
    record.metrics = metrics
    record.run_id = run_id or record.run_id
    record.dataset_version = dataset_version or record.dataset_version
    record.is_active = is_active
    if is_active:
        # Only one active version per registered model.
        for other in session.scalars(
            select(ModelMetadata).where(
                ModelMetadata.model_name == model_name,
                ModelMetadata.is_active.is_(True),
                ModelMetadata.version != version,
            )
        ):
            other.is_active = False
            other.stage = "archived"
        record.promoted_at = utcnow()
    session.flush()
    return record


def record_deployment(
    session: Session,
    *,
    model_name: str,
    to_version: str,
    action: str,
    reason: str,
    decision: dict[str, Any] | None = None,
    from_version: str | None = None,
) -> ModelDeploymentHistory:
    """Append a promotion/rejection decision to the audit trail."""
    record = ModelDeploymentHistory(
        model_name=model_name,
        from_version=from_version,
        to_version=to_version,
        action=action,
        reason=reason,
        decision=decision or {},
    )
    session.add(record)
    session.flush()
    return record


def record_prediction(
    session: Session,
    *,
    request_id: str,
    model_name: str,
    model_version: str,
    prediction: float,
    features: dict[str, Any],
    target: str = "target",
    unit: str = "percent",
    origin_date: str | None = None,
    latency_ms: float | None = None,
    source: str = "api",
) -> PredictionRecord:
    """Persist one served prediction."""
    record = PredictionRecord(
        request_id=request_id,
        model_name=model_name,
        model_version=model_version,
        prediction=prediction,
        features=features,
        target=target,
        unit=unit,
        origin_date=origin_date,
        latency_ms=latency_ms,
        source=source,
    )
    session.add(record)
    session.flush()
    return record


def list_predictions(
    session: Session,
    *,
    limit: int = 50,
    offset: int = 0,
    model_name: str | None = None,
) -> list[PredictionRecord]:
    """Most recent predictions first."""
    statement: Select[tuple[PredictionRecord]] = (
        select(PredictionRecord)
        .order_by(desc(PredictionRecord.created_at), desc(PredictionRecord.id))
        .limit(limit)
        .offset(offset)
    )
    if model_name:
        statement = statement.where(PredictionRecord.model_name == model_name)
    return list(session.scalars(statement))


def count_predictions(session: Session, *, model_name: str | None = None) -> int:
    """Total stored predictions (used by the API pagination envelope)."""
    from sqlalchemy import func

    statement = select(func.count(PredictionRecord.id))
    if model_name:
        statement = statement.where(PredictionRecord.model_name == model_name)
    return int(session.scalar(statement) or 0)


def record_ingestion(
    session: Session,
    *,
    dataset_name: str,
    source: str,
    raw_path: str,
    dataset_version: str,
    rows_in: int,
    rows_out: int,
    validation_passed: bool,
    report: dict[str, Any],
) -> DataIngestionLog:
    """Persist an ingestion attempt (including failed validation)."""
    record = DataIngestionLog(
        dataset_name=dataset_name,
        source=source,
        raw_path=raw_path,
        dataset_version=dataset_version,
        rows_in=rows_in,
        rows_out=rows_out,
        validation_passed=validation_passed,
        report=report,
    )
    session.add(record)
    session.flush()
    return record


def record_monitoring_run(
    session: Session,
    *,
    model_name: str,
    model_version: str | None,
    status: str,
    data_drift: bool,
    drift_share: float,
    performance_status: str,
    retrain_recommended: bool,
    n_reference_rows: int,
    n_current_rows: int,
    report_path: str | None,
    summary: dict[str, Any],
) -> MonitoringRun:
    """Persist the outcome of one Evidently monitoring run."""
    record = MonitoringRun(
        model_name=model_name,
        model_version=model_version,
        status=status,
        data_drift=data_drift,
        drift_share=drift_share,
        performance_status=performance_status,
        retrain_recommended=retrain_recommended,
        n_reference_rows=n_reference_rows,
        n_current_rows=n_current_rows,
        report_path=report_path,
        summary=summary,
    )
    session.add(record)
    session.flush()
    return record


def latest_monitoring_run(session: Session) -> MonitoringRun | None:
    """Most recent monitoring result (backs ``GET /drift/status``)."""
    return session.scalars(
        select(MonitoringRun)
        .order_by(desc(MonitoringRun.created_at), desc(MonitoringRun.id))
        .limit(1)
    ).first()


def record_pipeline_execution(
    session: Session,
    *,
    pipeline: str,
    status: str,
    duration_seconds: float | None = None,
    dataset_version: str | None = None,
    model_version: str | None = None,
    details: dict[str, Any] | None = None,
) -> PipelineExecution:
    """Persist a pipeline outcome for the infrastructure dashboards."""
    record = PipelineExecution(
        pipeline=pipeline,
        status=status,
        duration_seconds=duration_seconds,
        dataset_version=dataset_version,
        model_version=model_version,
        details=details or {},
    )
    session.add(record)
    session.flush()
    return record


def latest_pipeline_executions(session: Session) -> list[PipelineExecution]:
    """Most recent execution per pipeline name."""
    results: list[PipelineExecution] = []
    for pipeline in (
        "ingestion",
        "training",
        "promotion",
        "monitoring",
    ):
        record = session.scalars(
            select(PipelineExecution)
            .where(PipelineExecution.pipeline == pipeline)
            .order_by(desc(PipelineExecution.created_at), desc(PipelineExecution.id))
            .limit(1)
        ).first()
        if record is not None:
            results.append(record)
    return results


def get_active_model(session: Session, model_name: str) -> ModelMetadata | None:
    """Currently promoted version according to the application database."""
    return session.scalars(
        select(ModelMetadata).where(
            ModelMetadata.model_name == model_name, ModelMetadata.is_active.is_(True)
        )
    ).first()
