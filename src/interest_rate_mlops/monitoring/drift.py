"""Data drift monitoring with Evidently.

Scope is deliberately narrow: this module measures **input distribution drift**
only. It never claims that forecast accuracy degraded - that requires realised
targets and is handled by :mod:`interest_rate_mlops.monitoring.performance`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from interest_rate_mlops.config import MonitoringConfig
from interest_rate_mlops.exceptions import MLOpsError
from interest_rate_mlops.logging_utils import get_logger
from interest_rate_mlops.storage import write_json

logger = get_logger(__name__)


@dataclass(frozen=True)
class DriftReportResult:
    """Outcome of one drift evaluation."""

    report_path: str
    summary_path: str
    dataset_drift: bool
    drift_share: float
    n_features: int
    n_drifted: int
    status: str
    n_reference_rows: int
    n_current_rows: int
    thresholds: dict[str, float]
    per_feature: dict[str, dict[str, Any]] = field(default_factory=dict)
    generated_at: str = ""
    warning: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _classify(share: float, cfg: MonitoringConfig) -> str:
    if share >= cfg.drift.critical_share:
        return "critical"
    if share >= cfg.drift.warning_share:
        return "warning"
    return "ok"


def _extract_evidently_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Pull dataset-level and per-column drift results from an Evidently report."""
    dataset_level: dict[str, Any] | None = None
    per_feature: dict[str, dict[str, Any]] = {}

    for entry in payload.get("metrics", []):
        result = entry.get("result", {}) or {}
        # Evidently 0.6 emits DatasetDriftMetric (dataset level) and DataDriftTable
        # (per column, under ``drift_by_columns``).
        if "drift_by_columns" in result:
            for column, detail in (result.get("drift_by_columns") or {}).items():
                per_feature[str(column)] = {
                    "drift_detected": bool(detail.get("drift_detected")),
                    "stattest": str(detail.get("stattest_name", detail.get("stattest", ""))),
                    "drift_score": detail.get("drift_score"),
                    "threshold": detail.get("threshold"),
                }
        elif "dataset_drift" in result and "number_of_columns" in result:
            dataset_level = result

    if dataset_level is None:
        raise MLOpsError(
            "Evidently report did not contain a dataset-level drift metric; "
            "the installed Evidently version may be incompatible with this preset"
        )
    return {"dataset": dataset_level, "per_feature": per_feature}


def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    cfg: MonitoringConfig,
    *,
    output_dir: str | Path,
    tag: str = "drift",
) -> DriftReportResult:
    """Run an Evidently data-drift report and persist HTML + JSON summaries.

    Args:
        reference: feature rows the champion was trained on.
        current: most recent feature rows.
        cfg: monitoring thresholds and paths.
        output_dir: directory receiving the HTML report and summary JSON.
        tag: filename prefix.
    """
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    if len(reference) < cfg.reference.min_reference_rows:
        raise MLOpsError(
            f"reference window has {len(reference)} rows; at least "
            f"{cfg.reference.min_reference_rows} are required"
        )
    if len(current) < cfg.drift.min_current_rows:
        raise MLOpsError(
            f"current window has {len(current)} rows; at least "
            f"{cfg.drift.min_current_rows} are required for a drift evaluation"
        )

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    html_path = directory / f"{tag}_{stamp}.html"
    summary_path = directory / f"{tag}_{stamp}.json"

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=ColumnMapping(),
    )
    report.save_html(str(html_path))

    payload = report.as_dict()
    extracted = _extract_evidently_metrics(payload)
    dataset = extracted["dataset"]
    drift_share = float(dataset.get("share_of_drifted_columns", 0.0) or 0.0)
    n_drifted = int(dataset.get("number_of_drifted_columns", 0) or 0)
    n_features = int(dataset.get("number_of_columns", reference.shape[1]) or reference.shape[1])
    status = _classify(drift_share, cfg)

    result = DriftReportResult(
        report_path=str(html_path),
        summary_path=str(summary_path),
        dataset_drift=bool(dataset.get("dataset_drift", False)),
        drift_share=round(drift_share, 4),
        n_features=n_features,
        n_drifted=n_drifted,
        status=status,
        n_reference_rows=int(len(reference)),
        n_current_rows=int(len(current)),
        thresholds={
            "warning_share": cfg.drift.warning_share,
            "critical_share": cfg.drift.critical_share,
        },
        per_feature=extracted["per_feature"],
        generated_at=stamp,
    )
    write_json(summary_path, result.as_dict())

    logger.info(
        "data drift report generated",
        extra={
            "event": "drift_report",
            "status_code": 200 if status == "ok" else 409,
        },
    )
    return result
