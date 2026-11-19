"""Application services shared by the REST routes and the GraphQL schema.

Keeping the logic here means both API surfaces enforce the same validation,
persist the same record and emit the same metrics.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from api import metrics
from api.dependencies.model import ModelHandle
from api.schemas.prediction import PredictionRequest, PredictionResponse
from api.schemas.system import DriftStatusResponse
from interest_rate_mlops.config import (
    Settings,
    load_data_config,
    load_monitoring_config,
    load_pipeline_config,
)
from interest_rate_mlops.database.repositories import latest_monitoring_run, record_prediction
from interest_rate_mlops.exceptions import MLOpsError
from interest_rate_mlops.features import features_from_history, validate_feature_payload
from interest_rate_mlops.logging_utils import get_logger
from interest_rate_mlops.storage import read_json

logger = get_logger(__name__)


def perform_prediction(
    payload: PredictionRequest,
    handle: ModelHandle,
    session: Session,
    settings: Settings,
    *,
    request_id: str | None = None,
    source: str = "api",
) -> PredictionResponse:
    """Validate input, score it, persist the record and update metrics."""
    request_id = request_id or str(uuid.uuid4())
    started = time.perf_counter()
    data_cfg = load_data_config()
    allowed_range = data_cfg.validation.allowed_target_range

    if payload.features is not None:
        frame = validate_feature_payload(payload.features, handle.feature_names, allowed_range)
        origin_date = None
        input_mode = "features"
    else:
        if not payload.history:  # pragma: no cover - schema enforces one mode
            raise MLOpsError("history mode requires at least one observation")
        computed = features_from_history(
            [observation.model_dump() for observation in payload.history],
            load_pipeline_config().features,
            target_name="target",
        )
        frame = validate_feature_payload(computed["features"], handle.feature_names, allowed_range)
        origin_date = computed["origin_date"]
        input_mode = "history"

    inference_started = time.perf_counter()
    raw_prediction = handle.model.predict(frame)
    inference_seconds = time.perf_counter() - inference_started

    prediction = float(np.asarray(raw_prediction).ravel()[0])
    if not np.isfinite(prediction):
        raise MLOpsError("model returned a non-finite prediction")

    latency_ms = (time.perf_counter() - started) * 1000.0
    feature_values = {name: float(value) for name, value in frame.iloc[0].items()}

    record = record_prediction(
        session,
        request_id=request_id,
        model_name=handle.name,
        model_version=handle.version,
        prediction=prediction,
        features=feature_values,
        target=handle.target,
        unit=handle.unit,
        origin_date=origin_date,
        latency_ms=round(latency_ms, 3),
        source=source,
    )

    metrics.PREDICTION_COUNT.labels(handle.name, handle.version).inc()
    metrics.INFERENCE_LATENCY.labels(handle.name, handle.version).observe(inference_seconds)
    metrics.PREDICTION_VALUE.labels(handle.name, handle.version).observe(prediction)

    logger.info(
        "prediction served",
        extra={
            "event": "prediction",
            "request_id": request_id,
            "model_name": handle.name,
            "model_version": handle.version,
            "status_code": 200,
            "duration_ms": round(latency_ms, 2),
        },
    )

    return PredictionResponse(
        request_id=request_id,
        prediction=round(prediction, 6),
        unit=handle.unit,
        target=handle.target,
        model_name=handle.name,
        model_version=handle.version,
        model_alias=handle.alias,
        input_mode=input_mode,
        origin_date=origin_date,
        features=feature_values,
        latency_ms=round(latency_ms, 3),
        created_at=record.created_at or datetime.now(UTC),
    )


def latest_feature_row(handle: ModelHandle) -> dict[str, Any]:
    """Freshest engineered feature row, so clients can try ``POST /predict``."""
    data_cfg = load_data_config()
    if not data_cfg.features_path.exists():
        raise MLOpsError(
            f"feature matrix not found at {data_cfg.features_path}; run the training "
            "pipeline first (`make train`)"
        )

    frame = pd.read_parquet(data_cfg.features_path)
    missing = [name for name in handle.feature_names if name not in frame.columns]
    if missing:
        raise MLOpsError(
            f"persisted feature matrix is missing features required by the champion: {missing}"
        )

    complete = frame[handle.feature_names].dropna()
    if complete.empty:
        raise MLOpsError("no complete feature row is available in the persisted matrix")

    row = complete.iloc[-1]
    origin = complete.index[-1]
    origin_date = origin[:10] if isinstance(origin, str) else str(pd.Timestamp(origin).date())
    return {
        "origin_date": origin_date,
        "features": {name: float(value) for name, value in row.items()},
        "notes": [
            "Features are the freshest complete row of the persisted training matrix; "
            "they predict the following month's observation."
        ],
    }


def read_drift_status(session: Session | None, settings: Settings) -> DriftStatusResponse:
    """Latest monitoring status, preferring the database and falling back to file."""
    monitoring_cfg = load_monitoring_config()

    if session is not None:
        try:
            record = latest_monitoring_run(session)
        except Exception as exc:  # noqa: BLE001 - status must stay readable
            logger.warning("drift status database read failed: %s", exc)
            record = None
        if record is not None:
            summary = record.summary or {}
            drift = summary.get("drift") or {}
            return DriftStatusResponse(
                available=True,
                generated_at=record.created_at.isoformat() if record.created_at else None,
                status=record.status,
                data_drift=record.data_drift,
                drift_share=record.drift_share,
                drift_status=drift.get("status"),
                performance_status=record.performance_status,
                retrain_recommended=record.retrain_recommended,
                alerts=summary.get("alerts", []),
                windows={
                    "reference_rows": record.n_reference_rows,
                    "current_rows": record.n_current_rows,
                },
                report_path=record.report_path,
                notes=summary.get("notes", []),
                source="database",
            )

    document = read_json(monitoring_cfg.paths.status_path, default=None)
    if isinstance(document, dict) and document:
        return DriftStatusResponse(
            available=True,
            generated_at=document.get("generated_at"),
            status=str(document.get("status", "unknown")),
            data_drift=document.get("data_drift"),
            drift_share=document.get("drift_share"),
            drift_status=document.get("drift_status"),
            performance_status=document.get("performance_status"),
            retrain_recommended=document.get("retrain_recommended"),
            alerts=document.get("alerts", []),
            windows=document.get("windows", {}),
            report_path=(document.get("drift") or {}).get("report_path"),
            notes=document.get("notes", []),
            source="file",
        )

    return DriftStatusResponse(
        available=False,
        notes=[
            "no monitoring run has been recorded yet; run `make monitor` or the "
            "monitoring Airflow DAG"
        ],
    )
