"""System-level response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DependencyStatus(BaseModel):
    """Health of one downstream dependency."""

    name: str
    healthy: bool
    detail: str = ""


class HealthResponse(BaseModel):
    """Liveness/readiness payload."""

    status: str = Field(description="ok | degraded | error")
    ready: bool
    version: str
    environment: str
    uptime_seconds: float
    dependencies: list[DependencyStatus] = Field(default_factory=list)
    checks: dict[str, bool] = Field(default_factory=dict)


class DriftStatusResponse(BaseModel):
    """Latest monitoring status (data drift and model performance separately)."""

    available: bool
    generated_at: str | None = None
    status: str = "unknown"
    data_drift: bool | None = None
    drift_share: float | None = None
    drift_status: str | None = None
    performance_status: str | None = None
    retrain_recommended: bool | None = None
    alerts: list[str] = Field(default_factory=list)
    windows: dict[str, int] = Field(default_factory=dict)
    report_path: str | None = None
    notes: list[str] = Field(default_factory=list)
    source: str = Field(default="none", description="database | file | none")
    extra: dict[str, Any] = Field(default_factory=dict)
