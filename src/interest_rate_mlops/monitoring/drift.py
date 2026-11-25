"""Data drift monitoring with Evidently.

Scope is deliberately narrow: this module measures **input distribution drift**
only. It never claims that forecast accuracy degraded - that requires realised
targets and is handled by :mod:`interest_rate_mlops.monitoring.performance`.

A feature is only counted as drifted when both conditions hold: the statistical
test rejects the null (Evidently's p-value) **and** the shift is material, i.e.
the Kolmogorov-Smirnov statistic reaches ``drift.min_effect_size``. P-values
shrink with the reference window, so on their own they flag differences of no
practical size.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
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
    #: True when ``drift_share`` reaches the configured warning share, so the
    #: flag always agrees with the status this run publishes.
    dataset_drift: bool
    drift_share: float
    n_features: int
    n_drifted: int
    status: str
    n_reference_rows: int
    n_current_rows: int
    thresholds: dict[str, float]
    #: Features whose test rejected but whose shift was below the effect-size floor.
    n_suppressed: int = 0
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


def _ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: the largest gap between CDFs.

    Computed on the pooled order statistics, which is where the supremum of the
    two empirical CDFs is necessarily attained. Bounded in [0, 1] and unit-free,
    so it is comparable across features measured in different scales.
    """
    ref = np.sort(reference)
    cur = np.sort(current)
    grid = np.concatenate([ref, cur])
    cdf_ref = np.searchsorted(ref, grid, side="right") / ref.size
    cdf_cur = np.searchsorted(cur, grid, side="right") / cur.size
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def _effect_sizes(reference: pd.Series, current: pd.Series) -> dict[str, float | None]:
    """How large is the shift, independent of how many rows were compared."""
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
    if ref.size < 2 or cur.size < 2:
        return {"ks_statistic": None, "mean_shift_sd": None}

    spread = float(np.std(ref, ddof=1))
    mean_shift = float(np.mean(cur) - np.mean(ref))
    if not np.isfinite(spread) or spread <= 0.0:
        # A constant reference column has no spread to standardise with; fall
        # back to the relative shift instead of dividing by zero.
        magnitude = abs(float(np.mean(ref))) or abs(float(np.mean(cur))) or 1.0
        standardised = abs(mean_shift) / magnitude
    else:
        standardised = abs(mean_shift) / spread
    return {
        "ks_statistic": round(_ks_statistic(ref, cur), 4),
        "mean_shift_sd": round(float(standardised), 4),
    }


def _apply_effect_size_gate(
    per_feature: dict[str, dict[str, Any]],
    reference: pd.DataFrame,
    current: pd.DataFrame,
    cfg: MonitoringConfig,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Demote statistically-significant but practically tiny shifts.

    Returns the gated per-feature records and the number of features the gate
    suppressed. A record keeps its raw statistical verdict under
    ``statistical_drift`` so the report never hides what Evidently said.
    """
    floor = cfg.drift.min_effect_size
    gated: dict[str, dict[str, Any]] = {}
    suppressed = 0
    for column, detail in per_feature.items():
        record = dict(detail)
        statistical = bool(detail.get("drift_detected"))
        if column in reference.columns and column in current.columns:
            record.update(_effect_sizes(reference[column], current[column]))
        else:
            record.update({"ks_statistic": None, "mean_shift_sd": None})
        effect = record.get("ks_statistic")
        # An unmeasurable effect size must not be used to hide a rejection.
        material = effect is None or float(effect) >= floor
        record["statistical_drift"] = statistical
        record["materially_drifted"] = bool(material)
        record["drift_detected"] = bool(statistical and material)
        if statistical and not material:
            suppressed += 1
        gated[column] = record
    return gated, suppressed


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
    per_feature, n_suppressed = _apply_effect_size_gate(
        extracted["per_feature"], reference, current, cfg
    )
    if per_feature:
        # The gate changes which columns count, so the share has to be recomputed
        # from the gated verdicts instead of trusting Evidently's own counter.
        n_features = len(per_feature)
        n_drifted = sum(1 for record in per_feature.values() if record["drift_detected"])
        drift_share = n_drifted / n_features
    else:
        logger.warning(
            "Evidently returned no per-column drift detail; the effect-size gate "
            "cannot be applied and its raw verdict is used unchanged",
            extra={"event": "drift_gate_unavailable"},
        )
        n_features = int(
            dataset.get("number_of_columns", reference.shape[1]) or reference.shape[1]
        )
        n_drifted = int(dataset.get("number_of_drifted_columns", 0) or 0)
        drift_share = float(dataset.get("share_of_drifted_columns", 0.0) or 0.0)
    status = _classify(drift_share, cfg)

    result = DriftReportResult(
        report_path=str(html_path),
        summary_path=str(summary_path),
        dataset_drift=drift_share >= cfg.drift.warning_share,
        drift_share=round(drift_share, 4),
        n_features=n_features,
        n_drifted=n_drifted,
        status=status,
        n_reference_rows=int(len(reference)),
        n_current_rows=int(len(current)),
        thresholds={
            "warning_share": cfg.drift.warning_share,
            "critical_share": cfg.drift.critical_share,
            "min_effect_size": cfg.drift.min_effect_size,
        },
        n_suppressed=n_suppressed,
        per_feature=per_feature,
        generated_at=stamp,
    )
    write_json(summary_path, result.as_dict())

    logger.info(
        "data drift report generated: %s/%s features drifted (%s suppressed by effect size)",
        n_drifted,
        n_features,
        n_suppressed,
        extra={
            "event": "drift_report",
            "status_code": 200 if status == "ok" else 409,
        },
    )
    return result
