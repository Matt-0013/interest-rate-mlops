"""Model performance monitoring.

Performance is evaluated **only** when realised target values exist for the
current window. Drift in inputs is a warning sign; it is not evidence that the
forecast got worse, so the two signals are never merged into one claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from interest_rate_mlops.config import MonitoringConfig
from interest_rate_mlops.evaluation.metrics import regression_metrics
from interest_rate_mlops.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PerformanceResult:
    """Current-window performance compared with the champion's baseline."""

    available: bool
    status: str
    n_actuals: int
    metrics: dict[str, float | None] = field(default_factory=dict)
    baseline_metrics: dict[str, float | None] = field(default_factory=dict)
    rmse_degradation_pct: float | None = None
    mape_degradation_pct: float | None = None
    reason: str = ""
    thresholds: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _classify(
    rmse_delta: float | None,
    mape_delta: float | None,
    cfg: MonitoringConfig,
) -> str:
    statuses: list[str] = []
    if rmse_delta is not None:
        if rmse_delta >= cfg.performance.rmse_degradation_critical_pct:
            statuses.append("critical")
        elif rmse_delta >= cfg.performance.rmse_degradation_warning_pct:
            statuses.append("warning")
    if mape_delta is not None:
        if mape_delta >= cfg.performance.mape_degradation_critical_pct:
            statuses.append("critical")
        elif mape_delta >= cfg.performance.mape_degradation_warning_pct:
            statuses.append("warning")

    if "critical" in statuses:
        return "critical"
    if "warning" in statuses:
        return "warning"
    return "ok"


def evaluate_model_performance(
    features_with_actuals: pd.DataFrame,
    model: Any,
    baseline_metrics: dict[str, Any],
    cfg: MonitoringConfig,
    *,
    feature_names: list[str],
    target_column: str = "target",
) -> PerformanceResult:
    """Score the champion on the labelled part of the current window.

    Args:
        features_with_actuals: feature rows plus the realised ``target``.
        model: loaded champion (anything with ``.predict``).
        baseline_metrics: champion metrics from the registry (``test_rmse`` etc.).
        cfg: monitoring thresholds.
        feature_names: exact feature contract of the champion.
        target_column: name of the realised-target column.
    """
    thresholds = {
        "rmse_degradation_warning_pct": cfg.performance.rmse_degradation_warning_pct,
        "rmse_degradation_critical_pct": cfg.performance.rmse_degradation_critical_pct,
        "mape_degradation_warning_pct": cfg.performance.mape_degradation_warning_pct,
        "mape_degradation_critical_pct": cfg.performance.mape_degradation_critical_pct,
    }

    if model is None:
        return PerformanceResult(
            available=False,
            status="unknown",
            n_actuals=0,
            reason="champion model could not be loaded from the registry",
            thresholds=thresholds,
        )

    labelled = features_with_actuals.dropna(subset=[target_column])
    if len(labelled) < cfg.current.min_rows_for_performance:
        return PerformanceResult(
            available=False,
            status="unknown",
            n_actuals=int(len(labelled)),
            reason=(
                f"{len(labelled)} realised observations available; at least "
                f"{cfg.current.min_rows_for_performance} are required"
            ),
            thresholds=thresholds,
        )

    try:
        predictions = model.predict(labelled[feature_names])
    except Exception as exc:  # noqa: BLE001 - model backends raise many types
        return PerformanceResult(
            available=False,
            status="unknown",
            n_actuals=int(len(labelled)),
            reason=f"prediction failed: {exc}",
            thresholds=thresholds,
        )

    metrics = regression_metrics(labelled[target_column], predictions)
    baseline = {
        "rmse": baseline_metrics.get("test_rmse"),
        "mae": baseline_metrics.get("test_mae"),
        "mape": baseline_metrics.get("test_mape"),
    }

    rmse_delta = None
    if metrics["rmse"] is not None and baseline["rmse"]:
        rmse_delta = (metrics["rmse"] - baseline["rmse"]) / baseline["rmse"] * 100.0

    # MAPE is only meaningful when enough realised values are away from zero.
    mape_coverage = float(metrics.get("mape_coverage_pct") or 0.0)
    mape_delta = None
    mape_note = ""
    if metrics["mape"] is None:
        mape_note = (
            "MAPE ignored for degradation: not computable in this window " "(no |actual| >= 0.5 pp)"
        )
    elif baseline["mape"]:
        if mape_coverage >= cfg.performance.min_mape_coverage_pct:
            mape_delta = (metrics["mape"] - baseline["mape"]) / baseline["mape"] * 100.0
        else:
            mape_note = (
                f"MAPE ignored for degradation: only {mape_coverage:.1f}% of the window "
                f"has |actual| >= 0.5 pp (threshold "
                f"{cfg.performance.min_mape_coverage_pct}%)"
            )

    status = _classify(rmse_delta, mape_delta, cfg)
    logger.info(
        "model performance evaluated",
        extra={"event": "performance_evaluated", "status_code": 200 if status == "ok" else 409},
    )
    return PerformanceResult(
        available=True,
        status=status,
        n_actuals=int(len(labelled)),
        metrics=metrics,
        baseline_metrics=baseline,
        rmse_degradation_pct=None if rmse_delta is None else round(rmse_delta, 3),
        mape_degradation_pct=None if mape_delta is None else round(mape_delta, 3),
        reason="; ".join(
            part
            for part in (
                "performance measured on realised observations in the current window",
                mape_note,
            )
            if part
        ),
        thresholds=thresholds,
    )
