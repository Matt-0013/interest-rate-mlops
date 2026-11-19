"""Serving-time feature handling.

Two request shapes are supported by the API and both converge here:

* engineered ``features`` - the caller supplies the model's exact feature
  contract (validated against the champion's stored feature names);
* raw ``history`` - a contiguous monthly series that is converted into a panel
  and pushed through the *same* :mod:`builders` code used in training, which is
  what prevents train/serve skew.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from interest_rate_mlops.config import FeatureConfig
from interest_rate_mlops.exceptions import DataValidationError
from interest_rate_mlops.features.builders import build_feature_row, required_history_length


def validate_feature_payload(
    payload: Mapping[str, Any],
    expected_names: Sequence[str],
    allowed_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Validate an engineered feature payload and return a one-row frame.

    The frame column order always matches ``expected_names`` so the estimator
    receives features in the order it was trained on.

    Raises:
        DataValidationError: on missing/extra features, non-numeric or
            non-finite values, or values outside the plausibility band.
    """
    expected = list(expected_names)
    if not expected:
        raise DataValidationError("model feature contract is empty", failures=["feature_contract"])

    missing = [name for name in expected if name not in payload]
    if missing:
        raise DataValidationError(
            f"missing required features: {missing}", failures=["missing_features"]
        )
    extra = [name for name in payload if name not in expected]
    if extra:
        raise DataValidationError(
            f"unexpected features (schema mismatch with the champion model): {extra}",
            failures=["unexpected_features"],
        )

    values: dict[str, float] = {}
    for name in expected:
        raw = payload[name]
        if isinstance(raw, bool) or not isinstance(raw, int | float | np.integer | np.floating):
            raise DataValidationError(
                f"feature {name!r} must be a number, got {type(raw).__name__}",
                failures=["invalid_feature_type"],
            )
        value = float(raw)
        if not np.isfinite(value):
            raise DataValidationError(
                f"feature {name!r} must be finite", failures=["invalid_feature_value"]
            )
        values[name] = value

    if allowed_range is not None:
        low, high = allowed_range
        rate_features = {
            name: value for name, value in values.items() if name.startswith(("lag_", "roll_mean_"))
        }
        outside = {name: value for name, value in rate_features.items() if not low <= value <= high}
        if outside:
            raise DataValidationError(
                f"rate features outside the plausible band [{low}, {high}]: {outside}",
                failures=["feature_out_of_range"],
            )

    return pd.DataFrame([values], columns=expected)


def history_to_panel(
    observations: Sequence[Mapping[str, Any]],
    target_name: str = "target",
) -> pd.DataFrame:
    """Convert ``[{date, value}, ...]`` into a contiguous monthly panel.

    The history must be month-start observations with no gaps: a gap means an
    observation is missing, and the feature builder is not allowed to bridge it.
    """
    if len(observations) < 2:
        raise DataValidationError(
            "at least two observations are required", failures=["insufficient_history"]
        )

    records: list[tuple[pd.Timestamp, float]] = []
    for item in observations:
        raw_date = item.get("date")
        raw_value = item.get("value")
        try:
            timestamp = pd.Timestamp(raw_date)
        except (ValueError, TypeError) as exc:
            raise DataValidationError(
                f"invalid observation date: {raw_date!r}", failures=["invalid_dates"]
            ) from exc
        if timestamp.day != 1:
            raise DataValidationError(
                f"observations must be month-start dates, got {timestamp.date()}",
                failures=["invalid_dates"],
            )
        try:
            value = float(raw_value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"invalid observation value for {timestamp.date()}: {raw_value!r}",
                failures=["invalid_feature_value"],
            ) from exc
        if not np.isfinite(value):
            raise DataValidationError(
                f"observation value for {timestamp.date()} must be finite",
                failures=["invalid_feature_value"],
            )
        records.append((timestamp, value))

    frame = pd.DataFrame.from_records(records, columns=["date", "value"])
    frame["date"] = pd.to_datetime(frame["date"])
    if frame["date"].duplicated().any():
        duplicates = sorted(frame.loc[frame["date"].duplicated(), "date"].dt.strftime("%Y-%m"))
        raise DataValidationError(
            f"duplicate observation dates: {duplicates[:5]}", failures=["duplicate_records"]
        )

    frame = frame.sort_values("date").set_index("date")
    calendar = pd.date_range(frame.index.min(), frame.index.max(), freq="MS")
    if len(calendar) != len(frame):
        missing = sorted(set(calendar) - set(frame.index))
        raise DataValidationError(
            f"history is not a contiguous monthly series; {len(missing)} month(s) "
            f"missing (first: {[str(value.date()) for value in missing[:5]]})",
            failures=["history_gaps"],
        )
    frame.index.name = "date"
    return frame.rename(columns={"value": target_name})


def features_from_history(
    observations: Sequence[Mapping[str, Any]],
    cfg: FeatureConfig,
    target_name: str = "target",
) -> dict[str, Any]:
    """Build the freshest feature row from a raw monthly history."""
    panel = history_to_panel(observations, target_name=target_name)
    minimum = required_history_length(cfg)
    if len(panel) < minimum:
        raise DataValidationError(
            f"{len(panel)} observations provided, at least {minimum} are required",
            failures=["insufficient_history"],
        )
    origin, row = build_feature_row(panel, cfg, target_col=target_name)
    return {
        "origin_date": origin.date().isoformat(),
        "features": row,
        "observations_used": int(len(panel)),
    }


def parse_iso_month(value: str | date) -> pd.Timestamp:
    """Parse an ISO date string and assert it is a month start."""
    timestamp = pd.Timestamp(value)
    if timestamp.day != 1:
        raise DataValidationError(
            f"{value!r} is not a month-start date", failures=["invalid_dates"]
        )
    return timestamp
