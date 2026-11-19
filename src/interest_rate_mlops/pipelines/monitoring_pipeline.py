"""Monitoring pipeline (Airflow DAG 4, ``make monitor``).

Produces an Evidently drift report over the champion's input features, evaluates
performance only where realised observations exist, and publishes a single status
document that ``GET /drift/status`` and Grafana can read.

Data drift and performance degradation stay separate fields all the way through:
a drift alert never asserts that accuracy fell.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from interest_rate_mlops.config import (
    DatasetConfig,
    MonitoringConfig,
    PipelineConfig,
    Settings,
    get_settings,
    load_data_config,
    load_monitoring_config,
    load_pipeline_config,
)
from interest_rate_mlops.database.repositories import (
    record_monitoring_run,
    record_pipeline_execution,
)
from interest_rate_mlops.exceptions import MLOpsError, ModelNotAvailableError
from interest_rate_mlops.logging_utils import get_logger
from interest_rate_mlops.monitoring import (
    DriftReportResult,
    PerformanceResult,
    evaluate_model_performance,
    run_drift_report,
)
from interest_rate_mlops.pipelines.common import (
    Timer,
    optional_session,
    record_pipeline_state,
    utc_now,
    utc_stamp,
)
from interest_rate_mlops.registry import ModelRegistry
from interest_rate_mlops.storage import read_parquet, write_json

logger = get_logger(__name__)

STATUS_ORDER = {"ok": 0, "unknown": 0, "warning": 1, "critical": 2}


@dataclass(frozen=True)
class MonitoringOutcome:
    """Result of one monitoring run."""

    status: str
    drift: DriftReportResult
    performance: PerformanceResult
    retrain_recommended: bool
    model_name: str
    model_version: str | None
    status_path: str
    summary_path: str
    report_paths: dict[str, str] = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)
    db_persisted: bool = False
    duration_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "drift": self.drift.as_dict(),
            "performance": self.performance.as_dict(),
            "retrain_recommended": self.retrain_recommended,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "status_path": self.status_path,
            "summary_path": self.summary_path,
            "alerts": list(self.alerts),
            "db_persisted": self.db_persisted,
            "duration_seconds": round(self.duration_seconds, 3),
        }


def _load_champion(
    settings: Settings, pipeline_cfg: PipelineConfig
) -> tuple[Any, dict[str, Any], str | None]:
    """Load the champion model and its baseline metrics (never fatal)."""
    registry = ModelRegistry(
        settings.resolved_tracking_uri, pipeline_cfg.mlflow.registered_model_name
    )
    champion = registry.get_version_by_alias(pipeline_cfg.mlflow.champion_alias)
    if champion is None:
        logger.warning(
            "no champion alias found; performance evaluation will be skipped",
            extra={"event": "champion_missing"},
        )
        return None, {}, None
    try:
        model = registry.load_model(pipeline_cfg.mlflow.champion_alias)
    except ModelNotAvailableError as exc:
        logger.warning("champion load failed: %s", exc, extra={"event": "champion_load_failed"})
        return None, champion.metrics, champion.version
    return model, champion.metrics, champion.version


def run_monitoring(
    *,
    settings: Settings | None = None,
    data_cfg: DatasetConfig | None = None,
    pipeline_cfg: PipelineConfig | None = None,
    monitoring_cfg: MonitoringConfig | None = None,
) -> MonitoringOutcome:
    """Run drift + performance monitoring and publish the status document."""
    settings = settings or get_settings()
    data_cfg = data_cfg or load_data_config()
    pipeline_cfg = pipeline_cfg or load_pipeline_config()
    monitoring_cfg = monitoring_cfg or load_monitoring_config()

    if not data_cfg.features_path.exists():
        raise MLOpsError(
            f"feature matrix not found at {data_cfg.features_path}; run the training "
            "pipeline first (`make train` or DAG `training_pipeline`)"
        )

    features = read_parquet(data_cfg.features_path)
    feature_names = list(pipeline_cfg.features.names())
    missing = [name for name in feature_names if name not in features.columns]
    if missing:
        raise MLOpsError(
            f"persisted feature matrix is missing configured features: {missing}; "
            "retrain so features and monitoring agree"
        )

    training_rows = int(len(features) * (1.0 - pipeline_cfg.split.test_fraction))
    reference = features.iloc[: max(training_rows, monitoring_cfg.reference.min_reference_rows)]
    current = features.iloc[-monitoring_cfg.current.window_size :]

    with Timer() as timer:
        alerts: list[str] = []
        drift = run_drift_report(
            reference[feature_names],
            current[feature_names],
            monitoring_cfg,
            output_dir=monitoring_cfg.paths.reports_path,
            tag="data_drift",
        )

        model, baseline_metrics, champion_version = _load_champion(settings, pipeline_cfg)
        performance = evaluate_model_performance(
            current,
            model,
            baseline_metrics,
            monitoring_cfg,
            feature_names=feature_names,
            target_column="target",
        )

        worst = max(
            [drift.status, performance.status],
            key=lambda value: STATUS_ORDER.get(value, 0),
        )
        retrain = (
            monitoring_cfg.alerts.retrain_when_drift_status != "never"
            and drift.status == monitoring_cfg.alerts.retrain_when_drift_status
        ) or (
            monitoring_cfg.alerts.retrain_when_performance_status != "never"
            and performance.status == monitoring_cfg.alerts.retrain_when_performance_status
        )

        if drift.status != "ok":
            alerts.append(
                f"data drift {drift.status}: {drift.n_drifted}/{drift.n_features} features "
                f"drifted (share {drift.drift_share})"
            )
        if performance.status != "ok" and performance.rmse_degradation_pct is not None:
            baseline_rmse = performance.baseline_metrics.get("rmse")
            current_rmse = performance.metrics.get("rmse")
            alerts.append(
                f"model performance {performance.status}: RMSE {current_rmse:.4f} vs "
                f"baseline {baseline_rmse:.4f} ({performance.rmse_degradation_pct:+.1f}% change)"
            )
        if retrain:
            alerts.append("retraining recommended by the configured monitoring thresholds")

        status_document = {
            "generated_at": utc_now().isoformat(),
            "run_id": utc_stamp(),
            "model": {
                "name": pipeline_cfg.mlflow.registered_model_name,
                "version": champion_version,
                "alias": pipeline_cfg.mlflow.champion_alias,
            },
            "status": worst,
            "data_drift": drift.dataset_drift,
            "drift_share": drift.drift_share,
            "drift_status": drift.status,
            "performance_status": performance.status,
            "performance_available": performance.available,
            "retrain_recommended": retrain,
            "drift": drift.as_dict(),
            "performance": performance.as_dict(),
            "alerts": alerts,
            "windows": {
                "reference_rows": int(len(reference)),
                "current_rows": int(len(current)),
            },
            "notes": [
                "Drift measures input feature distributions only; it does not imply "
                "that forecast accuracy degraded.",
                "Performance is reported only over realised observations in the current window.",
            ],
        }
        status_path = write_json(monitoring_cfg.paths.status_path, status_document)
        summary_path = write_json(monitoring_cfg.paths.latest_summary_path, status_document)

        db_persisted = False
        with optional_session(settings) as session:
            if session is not None:
                record_monitoring_run(
                    session,
                    model_name=pipeline_cfg.mlflow.registered_model_name,
                    model_version=champion_version,
                    status=worst,
                    data_drift=drift.dataset_drift,
                    drift_share=drift.drift_share,
                    performance_status=performance.status,
                    retrain_recommended=retrain,
                    n_reference_rows=int(len(reference)),
                    n_current_rows=int(len(current)),
                    report_path=drift.report_path,
                    summary=status_document,
                )
                record_pipeline_execution(
                    session,
                    pipeline="monitoring",
                    status="succeeded",
                    duration_seconds=timer.seconds,
                    details={
                        "drift_status": drift.status,
                        "performance_status": performance.status,
                    },
                )
                db_persisted = True

        record_pipeline_state(
            settings,
            pipeline="monitoring",
            status=worst,
            duration_seconds=timer.seconds,
            details={
                "drift_status": drift.status,
                "performance_status": performance.status,
                "retrain_recommended": retrain,
            },
        )

    logger.info(
        "monitoring pipeline finished",
        extra={"event": "monitoring_complete", "status_code": 200 if worst == "ok" else 409},
    )
    return replace(
        MonitoringOutcome(
            status=worst,
            drift=drift,
            performance=performance,
            retrain_recommended=retrain,
            model_name=pipeline_cfg.mlflow.registered_model_name,
            model_version=champion_version,
            status_path=str(status_path),
            summary_path=str(summary_path),
            report_paths={"html": drift.report_path, "summary": drift.summary_path},
            alerts=alerts,
        ),
        db_persisted=db_persisted,
        duration_seconds=timer.seconds,
    )
