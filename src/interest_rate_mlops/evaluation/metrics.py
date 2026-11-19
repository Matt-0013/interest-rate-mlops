"""Regression metrics for rate forecasting.

MAE and RMSE are the primary metrics and are always defined. MAPE is reported
only where it is meaningful: the fed funds rate has traded as low as 0.07%, so a
plain percentage error explodes near zero. MAPE is therefore evaluated on actual
values with ``|y| >= min_abs_actual`` (default 0.5 percentage points) and the
coverage is reported alongside it. When nothing qualifies, MAPE is ``None``
rather than a misleading number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

DEFAULT_MAPE_MIN_ABS_ACTUAL = 0.5


@dataclass(frozen=True)
class MAPEResult:
    """MAPE value plus how much of the sample it is based on."""

    value: float | None
    min_abs_actual: float
    n_evaluated: int
    n_total: int
    coverage_pct: float
    excluded_near_zero: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_arrays(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype=float).ravel()
    predicted = np.asarray(y_pred, dtype=float).ravel()
    if actual.shape != predicted.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length (got {actual.shape} vs {predicted.shape})"
        )
    if actual.size == 0:
        raise ValueError("cannot compute metrics on empty arrays")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("y_true/y_pred contain NaN or infinite values")
    return actual, predicted


def mean_absolute_error(y_true: Any, y_pred: Any) -> float:
    """Mean absolute error in the target's units (percentage points)."""
    actual, predicted = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))


def root_mean_squared_error(y_true: Any, y_pred: Any) -> float:
    """Root mean squared error in the target's units."""
    actual, predicted = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mean_absolute_percentage_error(
    y_true: Any,
    y_pred: Any,
    min_abs_actual: float = DEFAULT_MAPE_MIN_ABS_ACTUAL,
) -> MAPEResult:
    """MAPE computed only on rows with a non-negligible actual value.

    Args:
        y_true: actual values.
        y_pred: predicted values.
        min_abs_actual: minimum ``|y|`` for a row to be included.

    Returns:
        A :class:`MAPEResult` whose ``value`` is ``None`` when no row qualifies.
    """
    actual, predicted = _as_arrays(y_true, y_pred)
    mask = np.abs(actual) >= min_abs_actual
    n_evaluated = int(mask.sum())
    if n_evaluated == 0:
        value: float | None = None
    else:
        value = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100.0)

    return MAPEResult(
        value=value,
        min_abs_actual=float(min_abs_actual),
        n_evaluated=n_evaluated,
        n_total=int(actual.size),
        coverage_pct=round(100.0 * n_evaluated / actual.size, 2),
        excluded_near_zero=int(actual.size - n_evaluated),
    )


def regression_metrics(
    y_true: Any,
    y_pred: Any,
    min_abs_actual: float = DEFAULT_MAPE_MIN_ABS_ACTUAL,
) -> dict[str, float | None]:
    """All reported metrics, flattened for MLflow/JSON/DB storage."""
    mape = mean_absolute_percentage_error(y_true, y_pred, min_abs_actual=min_abs_actual)
    actual, _ = _as_arrays(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mape": mape.value,
        "mape_coverage_pct": mape.coverage_pct,
        "n_samples": float(actual.size),
        "mean_actual": float(actual.mean()),
    }
