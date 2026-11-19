"""Feature engineering for one-step-ahead forecasting.

Semantics (pinned by unit tests):

* a feature row is indexed by its **origin** month ``t``, the last month whose
  observation is known at prediction time;
* that row predicts the target at ``t + horizon`` (default next month);
* ``lag_k`` is the observation ``k`` steps before the *target*, i.e. ``k-1``
  steps before the origin, so ``lag_1`` is the freshest known value;
* rolling statistics end **at** ``t`` and use complete windows only;
* ``diff_k`` is ``value[t] - value[t-k]``.

Consequences: no future value can enter a feature row, and a missing observation
propagates as ``NaN`` and drops the row instead of being imputed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from interest_rate_mlops.config import FeatureConfig
from interest_rate_mlops.exceptions import DataValidationError
from interest_rate_mlops.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FeatureMatrix:
    """Supervised design matrix with explicit time bookkeeping."""

    X: pd.DataFrame
    y: pd.Series
    target_dates: pd.Series
    feature_names: list[str]
    target_name: str

    @property
    def origins(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.X.index, name="origin")

    def __len__(self) -> int:
        return len(self.X)

    def metadata(self) -> dict[str, Any]:
        """Serialisable description stored next to every trained model."""
        return {
            "target": self.target_name,
            "feature_names": list(self.feature_names),
            "n_samples": int(len(self.X)),
            "start": str(self.origins.min().date()),
            "end": str(self.origins.max().date()),
            "horizon": 1,
        }


def _target_series(panel: pd.DataFrame, target_col: str) -> pd.Series:
    if target_col not in panel.columns:
        raise DataValidationError(
            f"target column {target_col!r} is not in the panel "
            f"(available: {list(panel.columns)})",
            failures=["missing_target"],
        )
    series = panel[target_col].astype(float)
    if series.isna().any():
        raise DataValidationError(
            "target series contains missing observations; resolve them before "
            "feature engineering (the pipeline never imputes the target)",
            failures=["target_gaps"],
        )
    if not panel.index.is_monotonic_increasing:
        raise DataValidationError("panel index must be sorted ascending", failures=["index_order"])
    return series


def build_features_frame(
    panel: pd.DataFrame,
    cfg: FeatureConfig,
    target_col: str = "target",
) -> pd.DataFrame:
    """Build the engineered feature frame, indexed by origin month."""
    series = _target_series(panel, target_col)
    features = pd.DataFrame(index=panel.index)

    for lag in cfg.lags:
        # lag_1 == value at the origin, lag_2 == one month earlier, ...
        features[f"lag_{lag}"] = series.shift(lag - 1)

    for window in cfg.rolling_windows:
        # min_periods == window: burn-in rows stay NaN rather than using a
        # shorter window (no partial-window surprises between train and serve).
        rolling = series.rolling(window=window, min_periods=window)
        if "mean" in cfg.rolling_stats:
            features[f"roll_mean_{window}"] = rolling.mean()
        if "std" in cfg.rolling_stats:
            features[f"roll_std_{window}"] = rolling.std(ddof=1)

    for diff in cfg.diffs:
        features[f"diff_{diff}"] = series.diff(diff)

    return features[cfg.names()].astype(float)


def build_supervised_features(
    panel: pd.DataFrame,
    cfg: FeatureConfig,
    target_col: str = "target",
    horizon: int = 1,
) -> FeatureMatrix:
    """Align features at origin ``t`` with the target observed at ``t + horizon``."""
    if horizon < 1:
        raise DataValidationError("horizon must be >= 1", failures=["horizon"])

    features = build_features_frame(panel, cfg, target_col=target_col)
    target = panel[target_col].astype(float).shift(-horizon)
    target_dates = pd.Series(panel.index, index=panel.index).shift(-horizon)

    mask = features.notna().all(axis=1) & target.notna()
    X = features.loc[mask]
    y = target.loc[mask].rename(target_col)
    dates = target_dates.loc[mask].rename("target_date")

    if X.empty:
        raise DataValidationError(
            "no complete feature rows were produced; the history is shorter than "
            f"the required lookback ({cfg.lookback} observations)",
            failures=["insufficient_history"],
        )
    if len(y) < 2:
        raise DataValidationError(
            f"only {len(y)} supervised samples available; at least 2 are required",
            failures=["insufficient_history"],
        )

    # Contract checks: strictly increasing time and no NaN anywhere.
    if not X.index.is_monotonic_increasing or not X.index.is_unique:
        raise DataValidationError("feature index must be unique and increasing")
    if y.isna().any() or X.isna().any().any():  # pragma: no cover - defensive
        raise DataValidationError("feature matrix contains NaN after filtering")
    if not (pd.DatetimeIndex(dates) > pd.DatetimeIndex(X.index)).all():  # pragma: no cover
        raise DataValidationError("target dates must be later than origin dates")

    logger.info(
        "built supervised feature matrix",
        extra={
            "event": "features_built",
            "n_features": len(X.columns),
            "samples": len(X),
            "start": str(X.index.min().date()),
            "end": str(X.index.max().date()),
        },
    )
    return FeatureMatrix(
        X=X,
        y=y,
        target_dates=dates,
        feature_names=list(X.columns),
        target_name=target_col,
    )


def build_feature_row(
    panel: pd.DataFrame,
    cfg: FeatureConfig,
    target_col: str = "target",
) -> tuple[pd.Timestamp, dict[str, float]]:
    """Compute the freshest complete feature row (used for serving/backtests)."""
    features = build_features_frame(panel, cfg, target_col=target_col)
    complete = features.dropna()
    if complete.empty:
        raise DataValidationError(
            f"history is too short: {len(panel)} observations provided, "
            f"{required_history_length(cfg)} required",
            failures=["insufficient_history"],
        )
    origin = pd.Timestamp(complete.index[-1])
    row = complete.iloc[-1].to_dict()
    return origin, {name: float(value) for name, value in row.items()}


def required_history_length(cfg: FeatureConfig) -> int:
    """Minimum number of observations needed for one complete feature row."""
    return int(cfg.lookback)


def feature_metadata(
    cfg: FeatureConfig,
    target_name: str,
    target_unit: str,
    horizon: int,
) -> dict[str, Any]:
    """Serialisable feature contract stored with the model artifact."""
    return {
        "target": target_name,
        "target_unit": target_unit,
        "horizon": horizon,
        "lookback_observations": required_history_length(cfg),
        "feature_names": cfg.names(),
        "lags": cfg.lags,
        "rolling_windows": cfg.rolling_windows,
        "rolling_stats": cfg.rolling_stats,
        "diffs": cfg.diffs,
    }
