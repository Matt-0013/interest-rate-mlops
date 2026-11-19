"""Application database schema.

Deliberately separate from MLflow's tracking database: MLflow owns experiments,
runs and registry state, while these tables own the *operational* record that the
API and dashboards need (what was served, when, with which model version, and
what the pipelines did).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from interest_rate_mlops.database.base import Base


def utcnow() -> datetime:
    """Timezone-aware UTC timestamp (server-side default is set in the DDL)."""
    return datetime.now(UTC)


class TrainingRun(Base):
    """One trained candidate and its scores."""

    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(32))
    experiment_name: Mapped[str] = mapped_column(String(128), nullable=False)
    dataset_name: Mapped[str] = mapped_column(String(128), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    git_commit: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="completed")
    params: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    cv_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )

    __table_args__ = (UniqueConstraint("run_id", "model_name", name="uq_training_runs_run_model"),)


class ModelMetadata(Base):
    """Promotion state of a registered model version."""

    __tablename__ = "model_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    run_id: Mapped[str | None] = mapped_column(String(64))
    alias: Mapped[str] = mapped_column(String(32), nullable=False, default="challenger")
    stage: Mapped[str] = mapped_column(String(32), nullable=False, default="challenger")
    dataset_version: Mapped[str | None] = mapped_column(String(64))
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )

    __table_args__ = (
        UniqueConstraint("model_name", "version", name="uq_model_metadata_name_version"),
        Index("ix_model_metadata_active", "model_name", "is_active"),
    )


class ModelDeploymentHistory(Base):
    """Audit trail of promotion and rejection decisions."""

    __tablename__ = "model_deployment_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    from_version: Mapped[str | None] = mapped_column(String(32))
    to_version: Mapped[str] = mapped_column(String(32), nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)  # promotion | rejection
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    decision: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )


class PredictionRecord(Base):
    """Every served prediction, with the model version that produced it."""

    __tablename__ = "prediction_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    target: Mapped[str] = mapped_column(String(64), nullable=False, default="target")
    origin_date: Mapped[str | None] = mapped_column(String(16))
    prediction: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(32), nullable=False, default="percent")
    features: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="api")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )

    __table_args__ = (Index("ix_prediction_records_created_desc", "created_at", "id"),)


class DataIngestionLog(Base):
    """Ingestion runs, including failed validation attempts."""

    __tablename__ = "data_ingestion_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="local_file")
    raw_path: Mapped[str] = mapped_column(String(512), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    rows_in: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rows_out: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    validation_passed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    report: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )


class MonitoringRun(Base):
    """Drift / performance monitoring executions."""

    __tablename__ = "monitoring_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_version: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), nullable=False)  # ok | warning | critical
    data_drift: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    drift_share: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    performance_status: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    retrain_recommended: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    n_reference_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_current_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    report_path: Mapped[str | None] = mapped_column(String(512))
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )


class PipelineExecution(Base):
    """Pipeline-level status used by the Prometheus file collector."""

    __tablename__ = "pipeline_executions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pipeline: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)  # succeeded | failed
    dataset_version: Mapped[str | None] = mapped_column(String(64))
    model_version: Mapped[str | None] = mapped_column(String(32))
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )
