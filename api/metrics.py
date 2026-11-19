"""Prometheus metrics for the API and the ML pipelines.

Two metric families coexist deliberately:

* **infrastructure** - request counts, latency, error rates (this module's
  counters/histograms, updated by middleware);
* **ML/pipeline** - drift share, performance status, pipeline status, read from
  the JSON state files the pipelines write (``FileStateCollector``).

Keeping the ML signals file-backed means Airflow jobs never need network access
to the API and the API never needs a database query per scrape.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from prometheus_client import Counter, Gauge, Histogram
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

from interest_rate_mlops.logging_utils import get_logger

logger = get_logger(__name__)

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total HTTP requests handled by the API.",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
HTTP_ERRORS = Counter(
    "api_http_errors_total",
    "HTTP responses with status >= 400.",
    ["endpoint", "status"],
)
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Predictions served, by model version.",
    ["model_name", "model_version"],
)
INFERENCE_LATENCY = Histogram(
    "model_inference_duration_seconds",
    "Model inference duration in seconds.",
    ["model_name", "model_version"],
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0),
)
PREDICTION_VALUE = Histogram(
    "model_prediction_value_percent",
    "Distribution of predicted interest rates (percentage points).",
    ["model_name", "model_version"],
    buckets=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0),
)
MODEL_INFO = Gauge(
    "mlops_model_info",
    "Currently served model (value is always 1).",
    ["model_name", "model_version", "alias"],
)
MODEL_LOAD_FAILURES = Counter(
    "mlops_model_load_failures_total",
    "Champion model load attempts that failed.",
)

_STATUS_VALUES = {"ok": 0, "unknown": -1, "warning": 1, "critical": 2}
_SUCCESS_STATES = {"succeeded", "ok", "completed", "skipped"}


def _read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _parse_timestamp(value: object) -> float:
    if not isinstance(value, str):
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


class FileStateCollector(Collector):
    """Expose pipeline and monitoring state files as Prometheus gauges."""

    def __init__(self, reports_dir: str | Path) -> None:
        self.reports_dir = Path(reports_dir)

    def collect(self):
        yield from self._pipeline_metrics()
        yield from self._monitoring_metrics()

    # -- pipelines ---------------------------------------------------------
    def _pipeline_metrics(self):
        state = _read_json(self.reports_dir / "state" / "pipeline_status.json")

        status = GaugeMetricFamily(
            "mlops_pipeline_status",
            "1 when the last run of the pipeline succeeded, else 0.",
            labels=["pipeline"],
        )
        last_run = GaugeMetricFamily(
            "mlops_pipeline_last_run_timestamp_seconds",
            "Unix timestamp of the last pipeline run.",
            labels=["pipeline"],
        )
        duration = GaugeMetricFamily(
            "mlops_pipeline_duration_seconds",
            "Duration of the last pipeline run in seconds.",
            labels=["pipeline"],
        )
        success = GaugeMetricFamily(
            "mlops_pipeline_last_success_timestamp_seconds",
            "Unix timestamp of the last successful pipeline run.",
            labels=["pipeline"],
        )

        for pipeline, payload in state.items():
            if not isinstance(payload, dict):
                continue
            pipeline_status = str(payload.get("status", "unknown"))
            timestamp = _parse_timestamp(payload.get("timestamp"))
            status.add_metric([pipeline], 1.0 if pipeline_status in _SUCCESS_STATES else 0.0)
            last_run.add_metric([pipeline], timestamp)
            duration.add_metric([pipeline], float(payload.get("duration_seconds") or 0.0))
            if pipeline_status in _SUCCESS_STATES:
                success.add_metric([pipeline], timestamp)

        yield status
        yield last_run
        yield duration
        yield success

    # -- ML monitoring -----------------------------------------------------
    def _monitoring_metrics(self):
        document = _read_json(self.reports_dir / "drift_status.json")

        drift_share = GaugeMetricFamily(
            "mlops_data_drift_share",
            "Share of model input features flagged as drifted.",
        )
        drift_detected = GaugeMetricFamily(
            "mlops_data_drift_detected",
            "1 when Evidently reports dataset drift, else 0.",
        )
        performance_status = GaugeMetricFamily(
            "mlops_performance_status",
            "Performance status (-1 unknown, 0 ok, 1 warning, 2 critical).",
        )
        rmse = GaugeMetricFamily(
            "mlops_model_rmse_percent",
            "RMSE of the champion on the monitoring window (percentage points).",
        )
        baseline_rmse = GaugeMetricFamily(
            "mlops_model_baseline_rmse_percent",
            "Champion RMSE on its held-out test block (percentage points).",
        )
        retrain = GaugeMetricFamily(
            "mlops_retrain_recommended",
            "1 when the monitoring thresholds recommend retraining, else 0.",
        )
        status_info = GaugeMetricFamily(
            "mlops_monitoring_status",
            "Overall monitoring status (-1 unknown, 0 ok, 1 warning, 2 critical).",
        )

        if document:
            metrics = (document.get("performance") or {}).get("metrics") or {}
            baseline = (document.get("performance") or {}).get("baseline_metrics") or {}
            overall = str(document.get("status", "unknown"))
            perf = str(document.get("performance_status", "unknown"))

            drift_share.add_metric([], float(document.get("drift_share") or 0.0))
            drift_detected.add_metric([], 1.0 if document.get("data_drift") else 0.0)
            performance_status.add_metric([], float(_STATUS_VALUES.get(perf, -1)))
            status_info.add_metric([], float(_STATUS_VALUES.get(overall, -1)))
            retrain.add_metric([], 1.0 if document.get("retrain_recommended") else 0.0)
            if metrics.get("rmse") is not None:
                rmse.add_metric([], float(metrics["rmse"]))
            if baseline.get("rmse") is not None:
                baseline_rmse.add_metric([], float(baseline["rmse"]))

        yield drift_share
        yield drift_detected
        yield performance_status
        yield rmse
        yield baseline_rmse
        yield retrain
        yield status_info
